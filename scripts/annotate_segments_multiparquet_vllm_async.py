import os, io, json, base64, asyncio, tempfile, argparse
from typing import List, Dict, Any, Optional, Tuple

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
from pydantic import BaseModel, Field
try:
    # Pydantic v2
    from typing import Annotated
    from pydantic.version import VERSION as PYD_VERSION
    IS_PYD_V2 = PYD_VERSION.startswith("2.")
except Exception:
    IS_PYD_V2 = False

if IS_PYD_V2:
    # v2: use Annotated[List[int], Field(min_length=4, max_length=4)]
    List4Ints = Annotated[List[int], Field(min_length=4, max_length=4)]
else:
    # v1: use conlist with min_items/max_items
    from pydantic import conlist
    List4Ints = conlist(int, min_items=4, max_items=4)


# --- models (replace your BBox2D and json_schema helper) ---
class BBox2D(BaseModel):
    bbox_2d: List4Ints

class BBox3D(BaseModel):
    estimable: bool
    x_min: Optional[float] = None
    y_min: Optional[float] = None
    z_min: Optional[float] = None
    x_max: Optional[float] = None
    y_max: Optional[float] = None
    z_max: Optional[float] = None
    note: Optional[str] = None

class FrameAnswer(BaseModel):
    frame_index: int
    q1_contact_object: Optional[str] = None
    q2_in_direct_contact: Optional[bool] = None
    q3_moving_towards_object: Optional[str] = None
    q4_moving_towards_bbox_2d: Optional[BBox2D] = None
    q4_moving_towards_bbox_3d: Optional[BBox3D] = None
    q4_moving_towards_unique_description: Optional[str] = None
    q5_status: str
    q6_success_trial: str  # "yes" | "no" | "N/A"

def _model_schema(model_cls):
    # v2: model_json_schema(); v1: schema()
    return (getattr(model_cls, "model_json_schema", None) or getattr(model_cls, "schema"))()

def json_schema_array_of_frames():
    item_schema = _model_schema(FrameAnswer)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "frame_annotations",
            "schema": {"type": "array", "items": item_schema}
        }
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
    def __init__(self, base_url: str, api_key: str, model: str, concurrency: int = 8, structured_output: str = "json_schema"):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.structured_output = structured_output  # "json_schema" | "json_object" | "none"

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
    out_path: str = "annotations.jsonl"
):
    # Load manifest and parquet lengths
    parquet_paths = load_parquet_manifest(parquet_manifest)
    parquet_lengths = parquet_lengths_from_manifest(parquet_paths)

    # Load segments and normalize/split them into per-parquet subsegments
    segments = load_segments_json(
        segments_json,
        default_task=default_task,
        parquet_lengths=parquet_lengths,
        end_inclusive_local=end_inclusive,
        end_inclusive_total=True  # your JSON's *_total_frame look inclusive
    )

    # Lazy cache of parquet -> pandas df (only the image column)
    pq_cache: Dict[int, pd.DataFrame] = {}

    async def load_df_if_needed(pq_idx: int) -> pd.DataFrame:
        if pq_idx in pq_cache:
            return pq_cache[pq_idx]
        path = parquet_paths[pq_idx]
        df = pd.read_parquet(path, columns=[image_col], engine="pyarrow")
        # Create an implicit local index column to align with your "line number within the parquet"
        df["_local_index"] = np.arange(len(df), dtype=np.int64)
        pq_cache[pq_idx] = df
        return df

    annotator = VLLMAnnotator(base_url=base_url, api_key=api_key, model=model,
                              concurrency=concurrency, structured_output=structured_output)

    results: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

    async def handle_segment(seg: Dict[str, Any]):
        seg_id   = seg["segment_id"]
        pq_idx   = seg["parquet_idx"]
        start    = int(seg["start"])
        end      = int(seg["end"])   # exclusive
        task_txt = seg["task"]

        df = await load_df_if_needed(pq_idx)

        # Clip to bounds
        start = max(0, start)
        end   = min(end, len(df))
        if end <= start:
            print(f"[WARN] Empty segment {seg_id} in pq[{pq_idx}] ({start}, {end}); skipping")
            return

        # Slice once; we’ll decode per window to keep RAM stable
        local_idxs_all = df["_local_index"].iloc[start:end].tolist()
        imgs_bytes = df[image_col].iloc[start:end].tolist()

        tasks = []
        for ws, we in make_windows_len(len(local_idxs_all), window=window, stride=stride):
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
        for window_out in window_outputs:
            for local_idx, obj in window_out.items():
                obj["_segment_id"] = seg_id
                obj["_parquet_idx"] = pq_idx
                results[(seg_id, pq_idx, local_idx)] = obj

    # Fire all segments
    await asyncio.gather(*[handle_segment(s) for s in segments])

    # Write JSONL (sorted by parquet_idx, then segment, then local frame)
    with open(out_path, "w", encoding="utf-8") as f:
        for (_, pq_idx, local_idx), obj in sorted(results.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} frame annotations to {out_path}")

# =============================
# CLI
# =============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_manifest", required=True, help="Text file: one parquet path per line. Line index == parquet_idx.")
    ap.add_argument("--segments", required=True, help="Segment JSON (with 'segments': [{start_index,end_index,parquet_idx,...}])")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
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
    args = ap.parse_args()

    asyncio.run(run(
        parquet_manifest=args.parquet_manifest,
        segments_json=args.segments,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        default_task=args.default_task,
        image_col=args.image_col,
        end_inclusive=args.end_inclusive,
        mode=args.mode,
        fps_for_video=args.fps,
        window=args.window,
        stride=args.stride,
        concurrency=args.concurrency,
        structured_output=args.structured_output,
        jpeg_quality=args.jpeg_quality,
        out_path=args.out
    ))

if __name__ == "__main__":
    main()
