#!/usr/bin/env python3
"""Lightweight video segment annotation helper.

This script fetches frames from parquets, groups them into windows, calls a
vLLM/OpenAI compatible endpoint, and writes frame-level answers to JSONL.  The
implementation purposely avoids complex orchestration so that behaviour stays
transparent and easy to debug.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
import pyarrow.parquet as pq

SYSTEM_PROMPT = (
    "You are a precise robot video annotator. Return JSON ONLY (no prose).\n"
    "IMPORTANT DEFINITIONS:\n"
    "- 'Gripper' = the two parallel metal fingers at the robot end effector.\n"
    "- If no object is in DIRECT contact with the gripper in a frame, use null for q1_contact_object AND q2_in_direct_contact.\n"
    "- 2D bounding boxes: absolute pixel coords [x1, y1, x2, y2].\n"
    "- 3D bounding boxes: if you cannot estimate 3D, set q4_moving_towards_bbox_3d = {\"estimable\": false}.\n"
    "- Output MUST be a JSON ARRAY where EACH ELEMENT corresponds to ONE FRAME in the same order."
)

QUESTION_BLOCK = """
The task description in this video is: {task}.
For each frame, please answer the following questions in JSON format.
Q1: What is the object that the gripper is in contact with? If no object is in direct contact with the gripper, answer None.
Q2: Is the robot arm gripper in direct contact with this object? Pay attention to the two metal fingers at the end of the robot's end effector.
Q3: What is the object that the robot arm gripper is moving towards?
Q4: Locate the moving-towards object in Question 3 (report in 2D bounding box, and 3D bounding box format), and also provide a unique text description of the object.
Q5: What is the current accomplishment status?
Q6: If there are multiple trials over grasping or placing, answer whether this frame contains a successful trial. For example, if the gripper tried to grasp the object multiple times, answer yes or no according to the current frame status. Otherwise, answer N/A.
Return a JSON array where each element corresponds to one frame in order.
""".strip()


@dataclass
class Segment:
    segment_id: str
    parquet_idx: int
    start: int
    end: int
    task: str


def decode_image_bytes(cell: Any) -> np.ndarray:
    arr = np.frombuffer(cell["bytes"], np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


def frames_to_mp4(frames: Sequence[np.ndarray], fps: int) -> bytes:
    if not frames:
        raise ValueError("no frames supplied")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    with open(path, "rb") as f:
        payload = f.read()
    Path(path).unlink(missing_ok=True)
    return payload


def make_windows(n: int, window: int, stride: int) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    starts = list(range(0, n, stride))
    return [(s, min(s + window, n)) for s in starts]


def load_manifest(path: Path) -> List[str]:
    entries: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
    if not entries:
        raise ValueError(f"manifest {path} is empty")
    return entries


def parquet_row_counts(paths: Sequence[str]) -> List[int]:
    lengths: List[int] = []
    for p in paths:
        meta = pq.ParquetFile(p).metadata
        lengths.append(meta.num_rows if meta is not None else 0)
    return lengths


def cumulative_starts(lengths: Sequence[int]) -> List[int]:
    starts = [0]
    total = 0
    for n in lengths[:-1]:
        total += n
        starts.append(total)
    return starts


def split_total_range(
    lengths: Sequence[int], start_total: int, end_total_exclusive: int
) -> List[Tuple[int, int, int]]:
    if end_total_exclusive <= start_total:
        return []
    starts = cumulative_starts(lengths)
    chunks: List[Tuple[int, int, int]] = []
    for idx, count in enumerate(lengths):
        g0 = starts[idx]
        g1 = g0 + count
        lo = max(start_total, g0)
        hi = min(end_total_exclusive, g1)
        if hi > lo:
            chunks.append((idx, lo - g0, hi - g0))
        if hi >= end_total_exclusive:
            break
    return chunks


def load_segments_json(
    path: Path,
    *,
    default_task: str,
    parquet_lengths: Sequence[int],
    end_inclusive_local: bool = True,
    end_inclusive_total: bool = True,
) -> List[Segment]:
    raw = json.loads(path.read_text())
    entries = raw.get("segments", raw)
    if not isinstance(entries, list):
        raise ValueError("segments JSON must be a list or contain a 'segments' list")

    results: List[Segment] = []
    cum = cumulative_starts(parquet_lengths)

    for idx, seg in enumerate(entries):
        task = seg.get("task") or default_task
        seg_id = seg.get("segment_id") or f"segment_{idx}"
        pq_idx = seg.get("parquet_idx")

        if pq_idx is None:
            g0 = int(seg["start_total_frame"])
            g1 = int(seg["end_total_frame"])
            if end_inclusive_total:
                g1 += 1
            for part_i, (p_idx, l0, l1) in enumerate(split_total_range(parquet_lengths, g0, g1)):
                part_id = f"{seg_id}_p{part_i}" if part_i else seg_id
                results.append(Segment(part_id, p_idx, l0, l1, task))
            continue

        pq_idx = int(pq_idx)
        l0 = int(seg["start_index"])
        l1 = int(seg["end_index"])
        if end_inclusive_local:
            l1 += 1

        n_rows = parquet_lengths[pq_idx]
        first_hi = min(l1, n_rows)
        if first_hi > l0:
            results.append(Segment(seg_id, pq_idx, max(0, l0), first_hi, task))

        spill = l1 - first_hi
        if spill <= 0:
            continue
        g0 = cum[pq_idx] + first_hi
        g1 = cum[pq_idx] + l1
        for part_i, (p_idx, a, b) in enumerate(split_total_range(parquet_lengths, g0, g1), start=1):
            results.append(Segment(f"{seg_id}_spill{part_i}", p_idx, a, b, task))

    return results


class SimpleAnnotator:
    def __init__(self, base_url: str, api_key: str, model: str, concurrency: int, mode: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.sem = asyncio.Semaphore(max(1, concurrency))
        self.mode = mode

    async def annotate_window(
        self,
        *,
        segment_id: str,
        frame_indices: Sequence[int],
        frames: Sequence[np.ndarray],
        task_text: str,
        fps: int,
    ) -> Dict[int, Dict[str, Any]]:
        if not frames:
            return {}

        content: List[Dict[str, Any]] = []
        if self.mode == "video":
            clip = frames_to_mp4(frames, fps=fps)
            data_url = "data:video/mp4;base64," + base64.b64encode(clip).decode("utf-8")
            content.append({"type": "text", "text": f"Segment {segment_id} with {len(frames)} frames."})
            content.append({"type": "video_url", "video_url": {"url": data_url}})
        else:
            content.append({"type": "text", "text": f"Frames from segment {segment_id}."})
            for idx, frame in zip(frame_indices, frames):
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not ok:
                    raise RuntimeError("failed to encode JPEG frame")
                data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")
                content.append({"type": "text", "text": f"Frame {idx}"})
                content.append({"type": "image_url", "image_url": {"url": data_url}})

        content.append({"type": "text", "text": QUESTION_BLOCK.format(task=task_text)})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        async with self.sem:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )

        text = response.choices[0].message.content.strip()
        data = self._parse_json_array(text)

        out: Dict[int, Dict[str, Any]] = {}
        for idx, obj in enumerate(data):
            frame_idx = obj.get("frame_index")
            if frame_idx is None and idx < len(frame_indices):
                frame_idx = frame_indices[idx]
                obj["frame_index"] = frame_idx
            out[int(frame_idx)] = obj
        return out

    @staticmethod
    def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
        first = raw.find("[")
        last = raw.rfind("]")
        snippet = raw[first : last + 1] if first != -1 and last != -1 and last > first else raw
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as exc:  # pragma: no cover - surface textual output
            raise RuntimeError(f"model did not return valid JSON: {raw[:200]}...") from exc
        if not isinstance(parsed, list):
            raise RuntimeError("model response is not a JSON array")
        cleaned: List[Dict[str, Any]] = []
        for item in parsed:
            if isinstance(item, dict):
                cleaned.append(item)
        return cleaned


async def annotate_segment(
    *,
    seg: Segment,
    annotator: SimpleAnnotator,
    df: pd.DataFrame,
    image_column: str,
    window: int,
    stride: int,
    fps: int,
    task_text: str,
) -> List[Dict[str, Any]]:
    start = max(0, seg.start)
    end = min(seg.end, len(df))
    if end <= start:
        return []

    local_indices = df["_local_index"].iloc[start:end].tolist()
    image_cells = df[image_column].iloc[start:end].tolist()

    frame_map: Dict[int, Dict[str, Any]] = {}
    for ws, we in make_windows(len(local_indices), window, stride):
        indices = local_indices[ws:we]
        frames = [decode_image_bytes(cell) for cell in image_cells[ws:we]]
        result = await annotator.annotate_window(
            segment_id=seg.segment_id,
            frame_indices=indices,
            frames=frames,
            task_text=task_text,
            fps=fps,
        )
        frame_map.update(result)

    ordered: List[Dict[str, Any]] = []
    for frame_idx in sorted(frame_map):
        obj = frame_map[frame_idx]
        obj.setdefault("frame_index", frame_idx)
        obj["_segment_id"] = seg.segment_id
        obj["_parquet_idx"] = seg.parquet_idx
        ordered.append(obj)
    return ordered


async def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.parquet_manifest)
    segments_path = Path(args.segments)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    lengths = parquet_row_counts(manifest)
    segments = load_segments_json(
        segments_path,
        default_task=args.default_task,
        parquet_lengths=lengths,
        end_inclusive_local=args.end_inclusive,
        end_inclusive_total=True,
    )
    if not segments:
        out_path.write_text("", encoding="utf-8")
        print("No segments to annotate; wrote empty file.")
        return

    annotator = SimpleAnnotator(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        concurrency=args.concurrency,
        mode=args.mode,
    )

    df_cache: Dict[int, pd.DataFrame] = {}
    total_frames = 0

    with out_path.open("w", encoding="utf-8") as sink:
        for seg in segments:
            if seg.parquet_idx not in df_cache:
                df = pd.read_parquet(manifest[seg.parquet_idx], columns=[args.image_col], engine="pyarrow")
                df["_local_index"] = np.arange(len(df), dtype=np.int64)
                df_cache[seg.parquet_idx] = df
            else:
                df = df_cache[seg.parquet_idx]

            try:
                frame_rows = await annotate_segment(
                    seg=seg,
                    annotator=annotator,
                    df=df,
                    image_column=args.image_col,
                    window=args.window,
                    stride=args.stride,
                    fps=args.fps,
                    task_text=seg.task,
                )
            except Exception as exc:
                print(f"[ERROR] Segment {seg.segment_id} failed: {exc}", file=sys.stderr)
                raise

            for row in frame_rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_frames += len(frame_rows)
            print(f"Segment {seg.segment_id}: wrote {len(frame_rows)} frames")

    print(f"Done. Wrote {total_frames} frame annotations to {out_path}.")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Annotate segments with a vLLM endpoint.")
    ap.add_argument("--parquet_manifest", required=True, help="Text file containing parquet paths (one per line).")
    ap.add_argument("--segments", required=True, help="JSON file describing segments to annotate.")
    ap.add_argument("--out", default="annotations.jsonl", help="Output JSONL file.")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key", default="token-abc123")
    ap.add_argument("--default_task", default="pick up the white mug and place it onto the plate, then move the chocolate bar to the left of the plate.")
    ap.add_argument("--image_col", default="image")
    ap.add_argument("--end_inclusive", action="store_true", default=True, help="Treat segment end_index as inclusive.")
    ap.add_argument("--mode", choices=["video", "multi-image"], default="video")
    ap.add_argument("--fps", type=int, default=2, help="FPS when packaging frames as video clips.")
    ap.add_argument("--window", type=int, default=64, help="Frames per request window.")
    ap.add_argument("--stride", type=int, default=64, help="Stride between windows.")
    ap.add_argument("--concurrency", type=int, default=2, help="Max simultaneous requests.")
    return ap


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
