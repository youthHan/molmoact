#!/usr/bin/env python3
"""Simple Qwen3-VL annotator using the HuggingFace backend.

This script loads frames from parquet shards, groups them into windows, and
runs Qwen3-VL via `transformers` to answer the fixed set of annotation
questions per frame.  It aims to be transparent and easy to tweak—no
multiprocessing or elaborate async machinery—so you can experiment quickly.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import pyarrow.parquet as pq
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

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


# -----------------------------------------------------------------------------
# Frame helpers
# -----------------------------------------------------------------------------


def decode_image_bytes(cell: Any) -> np.ndarray:
    arr = np.frombuffer(cell["bytes"], np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


def save_frames_as_video(frames: Sequence[np.ndarray], fps: int) -> Path:
    if not frames:
        raise ValueError("no frames supplied")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()
    return Path(tmp.name)


def save_frames_as_images(frames: Sequence[np.ndarray]) -> List[Path]:
    paths: List[Path] = []
    for frame in frames:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        success, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            raise RuntimeError("failed to encode JPEG frame")
        Path(tmp.name).write_bytes(buf.tobytes())
        paths.append(Path(tmp.name))
    return paths


# -----------------------------------------------------------------------------
# Segment utilities
# -----------------------------------------------------------------------------


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
    return [pq.ParquetFile(p).metadata.num_rows for p in paths]


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


def make_windows(n: int, window: int, stride: int) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    return [(start, min(start + window, n)) for start in range(0, n, stride)]


# -----------------------------------------------------------------------------
# HF inference wrapper
# -----------------------------------------------------------------------------


def parse_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str is None or dtype_str.lower() == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[key]


class ParseError(RuntimeError):
    pass


class HFAnnotator:
    def __init__(
        self,
        model_name: str,
        device_map: str,
        torch_dtype: Optional[torch.dtype],
        max_new_tokens: int,
        mode: str,
        video_min_pixels: Optional[int],
        video_max_pixels: Optional[int],
        video_total_pixels: Optional[int],
        video_sample_fps: Optional[float],
        debug_dir: Optional[Path],
    ) -> None:
        print(f"Loading processor {model_name} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_name)
        load_kwargs: Dict[str, Any] = {"device_map": device_map, "trust_remote_code": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        print(f"Loading model {model_name} ...", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.mode = mode
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.video_total_pixels = video_total_pixels
        self.video_sample_fps = video_sample_fps
        self.image_patch_size = getattr(self.processor.image_processor, "patch_size", 16)
        self.debug_dir = debug_dir
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def annotate_window(
        self,
        *,
        segment_id: str,
        frame_indices: Sequence[int],
        frames: Sequence[np.ndarray],
        task_text: str,
        fps: int,
    ) -> Dict[int, Dict[str, Any]]:
        temp_files: List[Path] = []
        try:
            user_content: List[Dict[str, Any]] = []
            if self.mode == "video":
                video_path = save_frames_as_video(frames, fps)
                temp_files.append(video_path)
                video_entry: Dict[str, Any] = {"type": "video", "video": video_path.resolve().as_uri()}
                if self.video_sample_fps is not None:
                    video_entry["fps"] = float(self.video_sample_fps)
                else:
                    video_entry["fps"] = float(fps)
                if self.video_min_pixels is not None:
                    video_entry["min_pixels"] = self.video_min_pixels
                if self.video_max_pixels is not None:
                    video_entry["max_pixels"] = self.video_max_pixels
                if self.video_total_pixels is not None:
                    video_entry["total_pixels"] = self.video_total_pixels
                user_content.append(video_entry)
            else:
                image_paths = save_frames_as_images(frames)
                temp_files.extend(image_paths)
                # Represent as a video with explicit frames list (per README guidance)
                frame_uris = [path.resolve().as_uri() for path in image_paths]
                video_entry = {"type": "video", "video": frame_uris}
                if self.video_sample_fps is not None:
                    video_entry["sample_fps"] = str(self.video_sample_fps)
                user_content.append(video_entry)

            user_content.append({"type": "text", "text": QUESTION_BLOCK.format(task=task_text)})

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.image_patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            if videos is not None:
                videos, video_meta = zip(*videos)
                videos = list(videos)
                video_meta = list(video_meta)
            else:
                video_meta = None

            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_meta,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            outputs = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            text_output = outputs[0] if outputs else "[]"
        finally:
            for path in temp_files:
                path.unlink(missing_ok=True)

        try:
            return self._json_array_to_dict(text_output, frame_indices)
        except ParseError as exc:
            self._handle_parse_failure(segment_id, text_output, exc)
            return {}

    def _handle_parse_failure(self, segment_id: str, raw: str, exc: Exception) -> None:
        msg = f"[WARN] Segment {segment_id}: model output not valid JSON ({exc}); skipping window"
        print(msg, file=sys.stderr)
        if self.debug_dir is not None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            name = f"{segment_id.replace('/', '_')}_{ts}.txt"
            path = self.debug_dir / name
            try:
                path.write_text(raw, encoding="utf-8")
                print(f"[INFO] Saved raw response to {path}", file=sys.stderr)
            except Exception as save_exc:
                print(f"[WARN] Failed to save debug response: {save_exc}", file=sys.stderr)

    @staticmethod
    def _json_array_to_dict(raw: str, frame_indices: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        first = raw.find("[")
        last = raw.rfind("]")
        snippet = raw[first : last + 1] if first != -1 and last != -1 and last > first else raw
        # Strip common markdown code fences
        snippet = snippet.strip()
        if snippet.startswith("```"):
            lines = snippet.splitlines()
            if lines:
                lines = lines[1:]
            while lines and lines[-1].strip().startswith("```"):
                lines.pop()
            snippet = "\n".join(lines)
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise ParseError(f"JSON decode error: {exc}") from exc
        if not isinstance(parsed, list):
            raise ParseError("Model response is not a JSON array")

        result: Dict[int, Dict[str, Any]] = {}
        for idx, obj in enumerate(parsed):
            if not isinstance(obj, dict):
                continue
            frame_idx = obj.get("frame_index")
            if frame_idx is None and idx < len(frame_indices):
                frame_idx = frame_indices[idx]
                obj["frame_index"] = frame_idx
            if frame_idx is None:
                continue
            result[int(frame_idx)] = obj
        return result


# -----------------------------------------------------------------------------
# Annotation driver
# -----------------------------------------------------------------------------


def annotate_segment(
    *,
    seg: Segment,
    annotator: HFAnnotator,
    df: pd.DataFrame,
    image_column: str,
    window: int,
    stride: int,
    fps: int,
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
        window_result = annotator.annotate_window(
            segment_id=seg.segment_id,
            frame_indices=indices,
            frames=frames,
            task_text=seg.task,
            fps=fps,
        )
        frame_map.update(window_result)

    ordered: List[Dict[str, Any]] = []
    for frame_idx in sorted(frame_map):
        obj = frame_map[frame_idx]
        obj.setdefault("frame_index", frame_idx)
        obj["_segment_id"] = seg.segment_id
        obj["_parquet_idx"] = seg.parquet_idx
        ordered.append(obj)
    return ordered


def run(args: argparse.Namespace) -> None:
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

    dtype = parse_dtype(args.dtype)
    annotator = HFAnnotator(
        model_name=args.model,
        device_map=args.device_map,
        torch_dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        video_total_pixels=args.video_total_pixels,
        video_sample_fps=args.video_sample_fps,
        debug_dir=out_path.parent / f"{out_path.name}.responses",
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

            frame_rows = annotate_segment(
                seg=seg,
                annotator=annotator,
                df=df,
                image_column=args.image_col,
                window=args.window,
                stride=args.stride,
                fps=args.fps,
            )

            for row in frame_rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_frames += len(frame_rows)
            print(f"Segment {seg.segment_id}: wrote {len(frame_rows)} frames")

    print(f"Done. Wrote {total_frames} frame annotations to {out_path}.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Annotate segments using Qwen3-VL (HF backend).")
    ap.add_argument("--parquet_manifest", required=True, help="Text file with parquet paths (one per line).")
    ap.add_argument("--segments", required=True, help="JSON segments file.")
    ap.add_argument("--out", default="annotations.jsonl", help="Output JSONL path.")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", help="Model name or path.")
    ap.add_argument("--device_map", default="auto", help="Device map for model loading (e.g., auto, cuda, cpu).")
    ap.add_argument("--dtype", default="auto", help="Torch dtype (auto, float16, bfloat16, float32).")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--default_task", default="pick up the white mug and place it onto the plate, then move the chocolate bar to the left of the plate.")
    ap.add_argument("--image_col", default="image")
    ap.add_argument("--end_inclusive", action="store_true", default=True, help="Treat segment end_index as inclusive.")
    ap.add_argument("--mode", choices=["video", "multi-image"], default="video")
    ap.add_argument("--fps", type=int, default=2, help="FPS when packaging frames into temp videos.")
    ap.add_argument("--window", type=int, default=64, help="Frames per request window.")
    ap.add_argument("--stride", type=int, default=64, help="Stride between windows.")
    ap.add_argument("--video_min_pixels", type=int, default=None)
    ap.add_argument("--video_max_pixels", type=int, default=None)
    ap.add_argument("--video_total_pixels", type=int, default=None)
    ap.add_argument("--video_sample_fps", type=float, default=None, help="Override FPS metadata sent to the model.")
    return ap


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
