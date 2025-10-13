import os, io, json, base64, asyncio, tempfile, argparse, math, sys, time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import multiprocessing as mp


def log_event(message: str, *, prefix: str = "", stream = sys.stdout) -> None:
    """Print a timestamped log line for easier runtime tracing."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix_text = f"{prefix.strip()} " if prefix else ""
    print(f"[{timestamp}] {prefix_text}{message}", file=stream, flush=True)

import numpy as np
import pandas as pd
import cv2
# from pydantic import BaseModel, conlist
from openai import AsyncOpenAI
import pyarrow.parquet as pq

# =============================
# Strict output schema (what we expect back)
# =============================
from typing import List, Optional


def _nullable(type_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {"anyOf": [{"type": "null"}, type_dict]}


BBOX_2D_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "bbox_2d": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
        }
    },
    "required": ["bbox_2d"],
    "additionalProperties": False,
}

BBOX_3D_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "estimable": {"type": "boolean"},
        "x_min": {"type": ["number", "null"]},
        "y_min": {"type": ["number", "null"]},
        "z_min": {"type": ["number", "null"]},
        "x_max": {"type": ["number", "null"]},
        "y_max": {"type": ["number", "null"]},
        "z_max": {"type": ["number", "null"]},
        "note": {"type": ["string", "null"]},
    },
    "required": ["estimable"],
    "additionalProperties": False,
}

FRAME_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_index": {"type": "integer"},
        "q1_contact_object": {"type": ["string", "null"]},
        "q2_in_direct_contact": {"type": ["boolean", "null"]},
        "q3_moving_towards_object": {"type": ["string", "null"]},
        "q4_moving_towards_bbox_2d": _nullable(BBOX_2D_SCHEMA),
        "q4_moving_towards_bbox_3d": _nullable(BBOX_3D_SCHEMA),
        "q4_moving_towards_unique_description": {"type": ["string", "null"]},
        "q5_status": {"type": "string"},
        "q6_success_trial": {"type": "string", "enum": ["yes", "no", "N/A"]},
    },
    "required": ["frame_index", "q5_status", "q6_success_trial"],
    "additionalProperties": False,
}


def json_schema_array_of_frames():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "frame_annotations",
            "schema": {
                "type": "array",
                "items": FRAME_SCHEMA,
            },
        },
    }

# =============================
# Prompts
# =============================

SYSTEM_MSG = (
    "You are a precise robot video annotator. Return JSON ONLY (no prose). JSON.\n"
    "IMPORTANT DEFINITIONS:\n"
    "- 'Gripper' = the two parallel metal fingers at the robot end effector.\n"
    "- If no object is in DIRECT contact with the gripper in a frame, use null for q1_contact_object AND q2_in_direct_contact.\n"
    "- 2D bounding boxes: absolute pixel coords [x1, y1, x2, y2] on the frames you see (top-left, bottom-right).\n"
    "- 3D bounding boxes: if you cannot estimate 3D, set q4_moving_towards_bbox_3d = {\"estimable\": false}.\n"
    "- Keep text concise; do NOT add extra keys.\n"
    "- Output MUST be a JSON ARRAY where EACH ELEMENT corresponds to ONE FRAME in the same order."
)

QUESTION_BLOCK = """
Please answer the following questions for EACH frame:

Q1: What is the object that the gripper is in contact with? If no object is in direct contact with the gripper, answer None.

Q2: Is the robot arm gripper in direct contact with this object? Pay attention to the two metal fingers at the end of the robot's end effector.

Q3: What is the object that the robot arm gripper is moving towards?

Q4: Locate the moving-towards object in Question 3 (report in 2D bounding box, and 3D bounding box format), and also provide a unique text description of the object.

Q5: What is the current accomplishment status? Describe it briefly.

Q6: If there are multiple trials over grasping or placing, answer whether this frame contains a successful trial. For example, if the gripper tried to grasp the object multiple times, answer yes or no according to the current frame status. Otherwise, answer N/A.
""".strip()

SCHEMA_BLOCK = """
Return a JSON ARRAY. For EACH frame (in order), include ONE object with these exact keys:

{
  "frame_index": <int>,
  "q1_contact_object": <string or null>,             // Q1
  "q2_in_direct_contact": <true|false|null>,         // Q2
  "q3_moving_towards_object": <string or null>,      // Q3
  "q4_moving_towards_bbox_2d": { "bbox_2d": [<int>,<int>,<int>,<int>] } or null,  // Q4
  "q4_moving_towards_bbox_3d": {
      "estimable": <bool>,
      "x_min": <number|null>, "y_min": <number|null>, "z_min": <number|null>,
      "x_max": <number|null>, "y_max": <number|null>, "z_max": <number|null>,
      "note": <string|null>
  } or null,                                         // Q4
  "q4_moving_towards_unique_description": <string or null>,   // Q4
  "q5_status": <string>,                             // Q5, <= 15 words
  "q6_success_trial": "yes" | "no" | "N/A"           // Q6
}
""".strip()

def user_instructions(task_text: str) -> str:
    return (
        f"Task description: {task_text}\n\n"
        f"{QUESTION_BLOCK}\n\n"
        "ADDITIONAL REQUIREMENTS:\n"
        "- Each object in the JSON array must correspond to the matching frame, in order.\n"
        "- Use absolute pixel coordinates for 2D boxes on the frames you see (no normalization).\n"
        "- If 3D cannot be estimated, set {\"estimable\": false} and keep other 3D fields null.\n\n"
        f"{SCHEMA_BLOCK}\n"
        "Output JSON array ONLY."
    )

# =============================
# Media helpers
# =============================

def decode_image_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b['bytes'], np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img  # BGR

def to_data_url_mime(buf: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(buf).decode("utf-8")

def image_to_data_url(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return to_data_url_mime(buf.tobytes(), "image/jpeg")

def frames_to_mp4_data_url(frames_bgr: List[np.ndarray], fps: int) -> str:
    """Encode frames into a temp .mp4 and return as data: URL (works with vLLM OpenAI server)."""
    if not frames_bgr:
        raise ValueError("No frames to encode")
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames_bgr:
        if f.shape[0] != h or f.shape[1] != w:
            f = cv2.resize(f, (w, h))
        vw.write(f)
    vw.release()
    with open(path, "rb") as f:
        buf = f.read()
    try:
        os.remove(path)
    except Exception:
        pass
    return to_data_url_mime(buf, "video/mp4")

# =============================
# Windows (sub-clips per request)
# =============================

def make_windows_len(n: int, window: int, stride: int) -> List[Tuple[int, int]]:
    starts = list(range(0, n, stride))
    return [(s, min(s + window, n)) for s in starts if s < n]


@dataclass
class SegmentResult:
    order: int
    segment_id: str
    parquet_idx: int
    rows: List[Dict[str, Any]]
    serialized: List[str] = field(default_factory=list, init=False, repr=False)

    def json_lines(self) -> List[str]:
        if not self.serialized:
            self.serialized = [json.dumps(row, ensure_ascii=False) for row in self.rows]
        return self.serialized


class SegmentWriter:
    def __init__(self, output_path: Path, start_order: int, flush: bool = True) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", encoding="utf-8")
        self._flush = flush
        self._next_order = start_order
        self._pending: Dict[int, SegmentResult] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def submit(self, result: SegmentResult) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("SegmentWriter is already closed")
            self._pending[result.order] = result
            await self._flush_ready_locked()

    async def _flush_ready_locked(self) -> None:
        while self._next_order in self._pending:
            result = self._pending.pop(self._next_order)
            for line in result.json_lines():
                self._fh.write(line + "\n")
            if self._flush:
                self._fh.flush()
            self._next_order += 1

    async def finalize(self) -> None:
        async with self._lock:
            if self._closed:
                return
            if self._pending:
                pending = sorted(self._pending.keys())
                raise RuntimeError(
                    f"Cannot finalize writer; pending segments remain: {pending[:8]}"
                    + (" ..." if len(pending) > 8 else "")
                )
            self._fh.flush()
            self._fh.close()
            self._closed = True

    def abort(self) -> None:
        if not self._closed:
            try:
                self._fh.flush()
            except Exception:
                pass
            self._fh.close()
            self._closed = True
        self._pending.clear()


class ProgressPrinter:
    def __init__(self, total_segments: int, prefix: str) -> None:
        self.total_segments = max(1, total_segments)
        self.prefix = prefix
        self._completed = 0
        self._frames = 0
        self._start_ts = time.time()
        self._lock = asyncio.Lock()

    async def update(self, *, segment_id: str, frames: int) -> None:
        async with self._lock:
            self._completed += 1
            self._frames += max(0, frames)
            elapsed = time.time() - self._start_ts
            avg_fps = (self._frames / elapsed) if elapsed > 0 else 0.0
            log_event(
                f"Segment {self._completed}/{self.total_segments} ({segment_id}) wrote {frames} frames; cumulative_frames={self._frames}, avg_fps={avg_fps:.2f}",
                prefix=self.prefix,
            )

# =============================
# Parquet manifest & segment JSON
# =============================

def parquet_lengths_from_manifest(manifest_paths: list[str]) -> list[int]:
    """Return number of rows in each parquet path without loading data."""
    lengths = []
    for p in manifest_paths:
        pf = pq.ParquetFile(p)
        lengths.append(pf.metadata.num_rows)
    if not lengths:
        raise ValueError("No parquet files found in manifest")
    return lengths

def _cumulative_starts(lengths: list[int]) -> list[int]:
    starts = [0]
    s = 0
    for n in lengths[:-1]:
        s += n
        starts.append(s)
    return starts  # len == len(lengths)

def split_total_range_over_parquets(
    lengths: list[int],
    start_total: int,
    end_total_exclusive: int
) -> list[tuple[int, int, int]]:
    """
    Given global [start_total, end_total_exclusive), return a list of
    (parquet_idx, local_start, local_end_exclusive) chunks that cover the range.
    """
    if end_total_exclusive <= start_total:
        return []
    starts = _cumulative_starts(lengths)  # total index at start of each parquet
    out = []
    for i, n in enumerate(lengths):
        g0 = starts[i]
        g1 = g0 + n  # exclusive
        # overlap with [start_total, end_total_exclusive)
        lo = max(start_total, g0)
        hi = min(end_total_exclusive, g1)
        if hi > lo:
            local_start = lo - g0
            local_end = hi - g0
            out.append((i, local_start, local_end))
        if hi >= end_total_exclusive:
            break
    return out

def load_parquet_manifest(manifest_path: str) -> List[str]:
    """Manifest is a text file with one parquet path per line. Line number == parquet_idx."""
    paths = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p and not p.startswith("#"):
                paths.append(p)
    if not paths:
        raise ValueError("Parquet manifest is empty")
    return paths

def load_segments_json(
    segments_json: str,
    default_task: str,
    parquet_lengths: list[int],
    end_inclusive_local: bool = True,
    end_inclusive_total: bool = True,
) -> list[dict]:
    """
    Supports segments where:
      A) parquet_idx is provided with local start_index/end_index (local to that parquet), or
      B) parquet_idx is None, but start_total_frame/end_total_frame are provided (global).
    It will also split any segment that spills past the end of its parquet into the next one(s).
    Returns a flat list of per-parquet subsegments:
      {segment_id, parquet_idx, start, end, task}
    where 'start' and 'end' are local [start, end) indices in that parquet (end exclusive).
    """
    with open(segments_json, "r", encoding="utf-8") as f:
        root = json.load(f)

    segs_raw = root.get("segments", [])
    if not segs_raw:
        raise ValueError("No segments found in JSON")

    out: list[dict] = []
    cum_starts = _cumulative_starts(parquet_lengths)

    def make_seg_id(base: str, part_idx: int | None) -> str:
        return f"{base}_part{part_idx}" if part_idx is not None else base

    for idx, s in enumerate(segs_raw):
        task_text = s.get("task") or default_task

        # Case B: parquet_idx is missing/None -> use total frame indices
        if s.get("parquet_idx") is None:
            if "start_total_frame" not in s or "end_total_frame" not in s:
                raise ValueError(
                    f"Segment {idx} has parquet_idx=None but lacks start_total_frame/end_total_frame"
                )
            g0 = int(s["start_total_frame"])
            g1 = int(s["end_total_frame"])
            if end_inclusive_total:
                g1 = g1 + 1  # make exclusive
            chunks = split_total_range_over_parquets(parquet_lengths, g0, g1)
            if not chunks:
                continue
            base_id = s.get("segment_id") or f"global_{g0}-{g1-1}"
            for part_i, (pq_i, l0, l1) in enumerate(chunks):
                out.append({
                    "segment_id": make_seg_id(base_id, part_i if len(chunks) > 1 else None),
                    "parquet_idx": pq_i,
                    "start": int(l0),
                    "end": int(l1),
                    "task": task_text,
                })
            continue

        # Case A: parquet_idx is given -> use local indices, but fix spillovers
        pq_i = int(s["parquet_idx"])
        if "start_index" not in s or "end_index" not in s:
            raise ValueError(f"Segment {idx} missing start_index/end_index")

        l0 = int(s["start_index"])
        l1 = int(s["end_index"])
        if end_inclusive_local:
            l1 = l1 + 1  # make exclusive

        # First chunk in this parquet
        n = parquet_lengths[pq_i]
        first_lo = max(0, l0)
        first_hi = min(n, l1)

        base_id = s.get("segment_id") or f"pq{pq_i}_{l0}-{(l1-1 if not end_inclusive_local else l1-2)}"
        part_counter = 0

        if first_hi > first_lo:
            out.append({
                "segment_id": make_seg_id(base_id, None),  # may add parts if spills further
                "parquet_idx": pq_i,
                "start": int(first_lo),
                "end": int(first_hi),
                "task": task_text,
            })
        # Spill to later parquets if needed
        remaining = (l1 - first_hi)
        cur_total_start = cum_starts[pq_i] + first_hi  # global start of remaining
        cur_total_end = cum_starts[pq_i] + l1
        if remaining > 0:
            chunks = split_total_range_over_parquets(
                parquet_lengths, cur_total_start, cum_starts[pq_i] + l1
            )
            # The first chunk overlaps the same parquet we already added; skip that one
            chunks = [(i, a, b) for (i, a, b) in chunks if i != pq_i or a >= first_hi]
            for sub in chunks:
                part_counter += 1
                i2, a2, b2 = sub
                out.append({
                    "segment_id": make_seg_id(base_id, part_counter),
                    "parquet_idx": i2,
                    "start": int(a2),
                    "end": int(b2),
                    "task": task_text,
                })

    return out

# =============================
# vLLM OpenAI client
# =============================

class VLLMAnnotator:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        concurrency: int = 8,
        structured_output: str = "json_schema",
        log_prefix: str = "",
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.structured_output = structured_output  # "json_schema" | "json_object" | "none"
        self.log_prefix = log_prefix.strip()

    async def annotate_window(
        self,
        segment_id: str,
        frame_indices: List[int],           # local indices within the parquet file
        frames_bgr: List[np.ndarray],
        mode: str,                          # "video" | "multi-image"
        task_text: str,
        fps_for_video: int = 2,
        jpeg_quality: int = 90
    ) -> Dict[int, Dict[str, Any]]:
        if not frames_bgr:
            return {}

        prefix = self.log_prefix
        if prefix:
            try:
                first_idx = frame_indices[0]
                last_idx = frame_indices[-1]
            except IndexError:
                first_idx = last_idx = -1
            log_event(
                f"Submitting window: segment={segment_id}, frames={len(frames_bgr)}, range=[{first_idx},{last_idx}], mode={mode}",
                prefix=prefix,
            )

        if mode == "video":
            clip_url = frames_to_mp4_data_url(frames_bgr, fps=fps_for_video)
            content = [
                {"type": "text", "text": f"Video clip from segment_id={segment_id}. Frames: {len(frames_bgr)} at {fps_for_video} FPS."},
                {"type": "video_url", "video_url": {"url": clip_url}},
                {"type": "text", "text": user_instructions(task_text)}
            ]
        elif mode == "multi-image":
            content = [{"type": "text", "text": f"Ordered frames from segment_id={segment_id}. Treat as a short continuous clip."}]
            for idx, img in zip(frame_indices, frames_bgr):
                content.append({"type": "text", "text": f"Frame {idx}:"})
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img, quality=jpeg_quality)}})
            content.append({"type": "text", "text": user_instructions(task_text)})
        else:
            raise ValueError("mode must be 'video' or 'multi-image'")

        messages = [{"role":"system","content":SYSTEM_MSG}, {"role":"user","content":content}]
        rf = None
        if self.structured_output == "json_schema":
            rf = json_schema_array_of_frames()
        elif self.structured_output == "json_object":
            rf = {"type": "json_object"}

        async with self.sem:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0, response_format=rf
            )

        txt = resp.choices[0].message.content.strip()
        if prefix:
            log_event(
                f"Received response for segment={segment_id}, frames={len(frames_bgr)}",
                prefix=prefix,
            )
        # In json_schema/json_object modes, it's already valid JSON:
        try:
            data = json.loads(txt)
        except Exception:
            # fallback: try to extract the first JSON array
            start, end = txt.find('['), txt.rfind(']')
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(f"Cannot parse JSON list from model output: {txt[:300]}...")
            data = json.loads(txt[start:end+1])

        out = {}
        for i, obj in enumerate(data):
            fi = obj.get("frame_index")
            if fi is None:
                # Align by order if the model omitted indices
                fi = frame_indices[i]
                obj["frame_index"] = fi
            out[int(fi)] = obj
        return out

# =============================
# Main driver
# =============================

async def run(
    parquet_manifest: str,
    segments_json: str,
    model: str,
    base_url: str,
    api_key: str,
    default_task: str,
    image_col: str = "image",
    end_inclusive: bool = True,       # local inclusive (start_index/end_index)
    mode: str = "video",
    fps_for_video: int = 2,
    window: int = 48,
    stride: int = 48,
    concurrency: int = 16,
    structured_output: str = "json_schema",
    jpeg_quality: int = 90,
    out_path: str = "annotations.jsonl",
    segments_override: Optional[List[Dict[str, Any]]] = None,
    segment_order_start: Optional[int] = None,
    stream_flush: bool = True,
    log_prefix: str = "",
) -> int:
    prefix = log_prefix.strip()

    parquet_paths = load_parquet_manifest(parquet_manifest)
    parquet_lengths = parquet_lengths_from_manifest(parquet_paths)

    if segments_override is not None:
        segments: List[Dict[str, Any]] = []
        base_order = segment_order_start if segment_order_start is not None else 0
        for idx, seg in enumerate(segments_override):
            seg_copy = dict(seg)
            seg_copy.setdefault("__order", base_order + idx)
            segments.append(seg_copy)
    else:
        loaded_segments = load_segments_json(
            segments_json,
            default_task=default_task,
            parquet_lengths=parquet_lengths,
            end_inclusive_local=end_inclusive,
            end_inclusive_total=True
        )
        segments = []
        for idx, seg in enumerate(loaded_segments):
            seg_copy = dict(seg)
            seg_copy["__order"] = idx
            segments.append(seg_copy)

    order_start = min((seg["__order"] for seg in segments), default=segment_order_start or 0)
    writer = SegmentWriter(Path(out_path), start_order=order_start, flush=stream_flush)
    if not segments:
        log_event("No segments provided; output will be empty.", prefix=prefix)
        await writer.finalize()
        return 0

    log_event(
        f"Beginning annotation for {len(segments)} segments (window={window}, stride={stride}, mode={mode}).",
        prefix=prefix,
    )

    progress = ProgressPrinter(len(segments), prefix)

    # Lazy cache of parquet -> pandas df (only the image column)
    pq_cache: Dict[int, pd.DataFrame] = {}

    async def load_df_if_needed(pq_idx: int) -> pd.DataFrame:
        if pq_idx in pq_cache:
            return pq_cache[pq_idx]
        path = parquet_paths[pq_idx]
        log_event(f"Loading parquet[{pq_idx}] from {path}", prefix=prefix)
        df = pd.read_parquet(path, columns=[image_col], engine="pyarrow")
        df["_local_index"] = np.arange(len(df), dtype=np.int64)
        pq_cache[pq_idx] = df
        return df

    annotator = VLLMAnnotator(
        base_url=base_url,
        api_key=api_key,
        model=model,
        concurrency=concurrency,
        structured_output=structured_output,
        log_prefix=prefix,
    )

    async def handle_segment(seg: Dict[str, Any]) -> int:
        seg_id = seg["segment_id"]
        pq_idx = seg["parquet_idx"]
        start = int(seg["start"])
        end = int(seg["end"])  # exclusive
        task_txt = seg["task"]

        log_event(
            f"Preparing segment {seg_id} (order={seg['__order']}, parquet={pq_idx}, frames={end - start})",
            prefix=prefix,
        )

        df = await load_df_if_needed(pq_idx)

        start = max(0, start)
        end = min(end, len(df))
        if end <= start:
            log_event(
                f"WARN: Empty segment {seg_id} in pq[{pq_idx}] ({start}, {end}); skipping",
                prefix=prefix,
                stream=sys.stderr,
            )
            return 0

        local_idxs_all = df["_local_index"].iloc[start:end].tolist()
        imgs_bytes = df[image_col].iloc[start:end].tolist()

        windows = make_windows_len(len(local_idxs_all), window=window, stride=stride)
        log_event(
            f"Segment {seg_id} split into {len(windows)} window(s)",
            prefix=prefix,
        )

        tasks = []
        for ws, we in windows:
            widxs = local_idxs_all[ws:we]
            frames_bgr = [decode_image_bytes(b) for b in imgs_bytes[ws:we]]
            tasks.append(annotator.annotate_window(
                segment_id=seg_id,
                frame_indices=widxs,
                frames_bgr=frames_bgr,
                mode=mode,
                task_text=task_txt,
                fps_for_video=fps_for_video,
                jpeg_quality=jpeg_quality
            ))

        window_outputs = await asyncio.gather(*tasks)
        frame_rows: List[Tuple[int, Dict[str, Any]]] = []
        for window_out in window_outputs:
            for local_idx, obj in window_out.items():
                obj["_segment_id"] = seg_id
                obj["_parquet_idx"] = pq_idx
                obj["_segment_order"] = seg["__order"]
                frame_rows.append((local_idx, obj))

        if not frame_rows:
            return 0

        frame_rows.sort(key=lambda item: item[0])
        ordered_objs = [obj for _, obj in frame_rows]

        result = SegmentResult(
            order=int(seg["__order"]),
            segment_id=seg_id,
            parquet_idx=pq_idx,
            rows=ordered_objs,
        )
        await writer.submit(result)
        await progress.update(segment_id=seg_id, frames=len(ordered_objs))
        return len(ordered_objs)

    frame_counts: List[int] = []
    try:
        if segments:
            frame_counts = await asyncio.gather(*[handle_segment(s) for s in segments])
    except Exception:
        writer.abort()
        raise
    else:
        await writer.finalize()

    total_frames = sum(frame_counts)
    log_event(
        f"Completed annotation: {total_frames} frames across {len(segments)} segments → {out_path}",
        prefix=prefix,
    )
    return total_frames


def _split_ranges(total: int, num_chunks: int) -> List[Tuple[int, int]]:
    if total <= 0:
        return []
    num_chunks = max(1, min(num_chunks, total))
    chunk = math.ceil(total / num_chunks)
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + chunk, total)
        ranges.append((start, end))
        start = end
    return ranges


def _merge_part_files(parts: List[Tuple[int, Path]], final_path: Path) -> int:
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    with final_path.open("w", encoding="utf-8") as fout:
        for _, part in sorted(parts, key=lambda item: item[0]):
            if not part.exists():
                continue
            with part.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    total_lines += 1
    return total_lines


def _worker_entry(
    worker_id: int,
    cfg: Dict[str, Any],
    segments_chunk: List[Dict[str, Any]],
    out_path: str,
    stream_flush: bool,
    result_queue: mp.Queue,
) -> None:
    log_prefix = f"[worker-{worker_id}]"
    log_event(
        f"Starting worker on {len(segments_chunk)} segments → {out_path}",
        prefix=log_prefix,
    )
    try:
        total = asyncio.run(
            run(
                parquet_manifest=cfg["parquet_manifest"],
                segments_json=cfg["segments_json"],
                model=cfg["model"],
                base_url=cfg["base_url"],
                api_key=cfg["api_key"],
                default_task=cfg["default_task"],
                image_col=cfg["image_col"],
                end_inclusive=cfg["end_inclusive"],
                mode=cfg["mode"],
                fps_for_video=cfg["fps"],
                window=cfg["window"],
                stride=cfg["stride"],
                concurrency=cfg["concurrency"],
                structured_output=cfg["structured_output"],
                jpeg_quality=cfg["jpeg_quality"],
                out_path=out_path,
                segments_override=segments_chunk,
                segment_order_start=segments_chunk[0]["__order"] if segments_chunk else 0,
                stream_flush=stream_flush,
                log_prefix=log_prefix,
            )
        )
        log_event(
            f"Finished worker; annotated {total} frames", prefix=log_prefix
        )
        result_queue.put((worker_id, total))
    except Exception as exc:  # pragma: no cover - surface worker failures
        log_event(f"Worker failed: {exc}", prefix=log_prefix, stream=sys.stderr)
        result_queue.put((worker_id, exc))
        raise

# =============================
# CLI
# =============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_manifest", required=True, help="Text file: one parquet path per line. Line index == parquet_idx.")
    ap.add_argument("--segments", required=True, help="Segment JSON (with 'segments': [{start_index,end_index,parquet_idx,...}])")
    ap.add_argument("--model", default="/mnt/bn/kinetics-lp-maliva/pretrain_models/Qwen3-VL-235B-A22B-Instruct/")
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:22002/v1"))
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "token-abc123"))
    ap.add_argument("--default_task", default="pick up the white mug and place it onto the plate, then move the chocolate bar to the left of the plate.")
    ap.add_argument("--image_col", default="image")
    ap.add_argument("--end_inclusive", action="store_true", default=True, help="Interpret end_index as inclusive (your JSON uses this) local idx only.")
    ap.add_argument("--mode", choices=["video","multi-image"], default="video")
    ap.add_argument("--fps", type=int, default=2, help="FPS for encoded sub-clips (video mode).")
    ap.add_argument("--window", type=int, default=48, help="Frames per request (if a segment is long).")
    ap.add_argument("--stride", type=int, default=48, help="Stride between windows.")
    ap.add_argument("--concurrency", type=int, default=16, help="Max concurrent requests.")
    ap.add_argument("--structured_output", choices=["json_schema","json_object","none"], default="json_schema")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--out", default="annotations.jsonl")
    ap.add_argument("--num_processes", type=int, default=1, help="Number of worker processes for segment batches.")
    ap.add_argument("--no_stream_flush", action="store_true", help="Disable flushing output file after each segment (higher throughput, less crash resilience).")
    args = ap.parse_args()

    root_prefix = "[main]"
    stream_flush = not args.no_stream_flush
    log_event(
        f"CLI start: processes={args.num_processes}, stream_flush={stream_flush}, out={args.out}",
        prefix=root_prefix,
    )

    cfg = {
        "parquet_manifest": args.parquet_manifest,
        "segments_json": args.segments,
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "default_task": args.default_task,
        "image_col": args.image_col,
        "end_inclusive": args.end_inclusive,
        "mode": args.mode,
        "fps": args.fps,
        "window": args.window,
        "stride": args.stride,
        "concurrency": args.concurrency,
        "structured_output": args.structured_output,
        "jpeg_quality": args.jpeg_quality,
    }

    if args.num_processes <= 1:
        log_event("Running in single-process mode", prefix=root_prefix)
        asyncio.run(run(
            parquet_manifest=cfg["parquet_manifest"],
            segments_json=cfg["segments_json"],
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            default_task=cfg["default_task"],
            image_col=cfg["image_col"],
            end_inclusive=cfg["end_inclusive"],
            mode=cfg["mode"],
            fps_for_video=cfg["fps"],
            window=cfg["window"],
            stride=cfg["stride"],
            concurrency=cfg["concurrency"],
            structured_output=cfg["structured_output"],
            jpeg_quality=cfg["jpeg_quality"],
            out_path=args.out,
            stream_flush=stream_flush,
            log_prefix=root_prefix,
        ))
        return

    log_event(
        f"Running in multi-process mode with {args.num_processes} workers", prefix=root_prefix
    )
    parquet_paths = load_parquet_manifest(cfg["parquet_manifest"])
    parquet_lengths = parquet_lengths_from_manifest(parquet_paths)
    segments_all = load_segments_json(
        cfg["segments_json"],
        default_task=cfg["default_task"],
        parquet_lengths=parquet_lengths,
        end_inclusive_local=cfg["end_inclusive"],
        end_inclusive_total=True,
    )

    if not segments_all:
        final_path = Path(args.out)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text("", encoding="utf-8")
        log_event(
            f"No segments to annotate; wrote 0 frame annotations to {final_path}",
            prefix=root_prefix,
        )
        return

    for idx, seg in enumerate(segments_all):
        seg["__order"] = idx

    ranges = _split_ranges(len(segments_all), args.num_processes)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes: List[Tuple[int, mp.Process]] = []
    parts: List[Tuple[int, Path]] = []

    for worker_id, (start, end) in enumerate(ranges):
        chunk = [dict(seg) for seg in segments_all[start:end]]
        if not chunk:
            continue
        part_path = Path(f"{args.out}.part{worker_id:02d}")
        if part_path.exists():
            try:
                part_path.unlink()
            except Exception as exc:
                log_event(
                    f"WARN: could not remove stale shard {part_path}: {exc}",
                    prefix=root_prefix,
                    stream=sys.stderr,
                )
        log_event(
            f"Launching worker-{worker_id} for segments[{start}:{end}) → {part_path}",
            prefix=root_prefix,
        )
        proc = ctx.Process(
            target=_worker_entry,
            args=(worker_id, cfg, chunk, str(part_path), stream_flush, result_queue),
        )
        proc.start()
        processes.append((worker_id, proc))
        parts.append((start, part_path))

    worker_results: Dict[int, int] = {}
    worker_errors: Dict[int, Exception] = {}

    for _ in processes:
        wid, payload = result_queue.get()
        if isinstance(payload, Exception):
            worker_errors[wid] = payload
        else:
            worker_results[wid] = int(payload)
            log_event(
                f"Worker-{wid} reported {worker_results[wid]} frames", prefix=root_prefix
            )

    for wid, proc in processes:
        proc.join()

    result_queue.close()
    result_queue.join_thread()

    for wid, proc in processes:
        if proc.exitcode != 0:
            worker_errors.setdefault(wid, RuntimeError(f"exit code {proc.exitcode}"))

    if worker_errors:
        for wid, err in sorted(worker_errors.items()):
            log_event(f"worker-{wid} failed: {err}", prefix=root_prefix, stream=sys.stderr)
        raise SystemExit(1)

    total_lines = _merge_part_files(parts, Path(args.out))
    total_frames = sum(worker_results.get(wid, 0) for wid, _ in processes)
    log_event(
        f"Merged {len(parts)} shard files into {args.out} ({total_lines} lines)",
        prefix=root_prefix,
    )
    log_event(
        f"Annotated ~{total_frames} frames across {len(segments_all)} segments using {len(processes)} workers",
        prefix=root_prefix,
    )

if __name__ == "__main__":
    main()
