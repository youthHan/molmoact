"""Export sampled trajectory segments as videos for manual inspection.

Example::

    python3 MolmoAct/scripts/export_segment_videos.py \
        --segments-json run_01_segments.json \
        --parquet /mnt/data/libero/train-00000-of-00025.parquet \
        --parquet-image-column image \
        --sample-count 3 \
        --output-dir segment_videos \
        --fps 15

Supports both directory-based frames (``--frames-dir`` + ``--pattern``) and
parquet shards (``--parquet`` + ``--parquet-image-column``). Segment indices
must align with the frame order used when tracking.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

try:
    import imageio.v2 as imageio
except ImportError as exc:  # pragma: no cover
    raise ImportError("imageio is required for video export") from exc

try:  # optional dependency for parquet reading
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - handled at runtime
    pq = None


@dataclass
class Segment:
    start_index: int
    end_index: int
    length: int
    start_total_frame: int
    end_total_frame: int
    start_local_frame: int
    end_local_frame: int
    parquet_idx: Optional[int] = None


# -----------------------------------------------------------------------------
# Utilities for loading segments and frames
# -----------------------------------------------------------------------------


def load_segments(path: Path) -> List[Segment]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "segments" in payload:
        segments_raw = payload["segments"]
    elif isinstance(payload, list):
        segments_raw = payload
    else:
        raise ValueError("segments JSON must be a list or have a 'segments' field")
    segments: List[Segment] = []
    for entry in segments_raw:
        start_index = int(entry.get("start_index", entry.get("start_frame", 0)))
        end_index = int(entry.get("end_index", entry.get("end_frame", start_index)))
        length = int(entry.get("length", end_index - start_index + 1))
        start_total = int(entry.get("start_total_frame", entry.get("start_frame", start_index)))
        end_total = int(entry.get("end_total_frame", entry.get("end_frame", end_index)))
        start_local = int(entry.get("start_frame", start_index))
        end_local = int(entry.get("end_frame", end_index))
        parquet_idx = entry.get("parquet_idx")
        parquet_idx = int(parquet_idx) if parquet_idx is not None else None
        segments.append(
            Segment(
                start_index=start_index,
                end_index=end_index,
                length=length,
                start_total_frame=start_total,
                end_total_frame=end_total,
                start_local_frame=start_local,
                end_local_frame=end_local,
                parquet_idx=parquet_idx,
            )
        )
    return segments


def list_frames(directory: Path, pattern: Optional[str]) -> List[Path]:
    pattern = pattern or "*.png"
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No frames found in {directory} matching {pattern}")
    return paths


def select_segments(
    segments: Sequence[Segment],
    explicit_indices: Optional[Sequence[int]],
    sample_count: int,
    seed: Optional[int],
) -> List[tuple[int, Segment]]:
    if explicit_indices:
        chosen = []
        for idx in explicit_indices:
            if idx < 0 or idx >= len(segments):
                raise IndexError(f"Segment index {idx} out of range (len={len(segments)})")
            chosen.append((idx, segments[idx]))
        return chosen

    population = list(enumerate(segments))
    if sample_count >= len(population):
        return population
    rng = random.Random(seed)
    sampled = rng.sample(population, sample_count)
    sampled.sort(key=lambda x: x[0])
    return sampled


def expand_parquet_tokens(tokens: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for token in tokens:
        expanded = os.path.expanduser(token)
        candidate = Path(expanded)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.parquet")))
            continue
        if candidate.is_file() and candidate.suffix == ".parquet":
            paths.append(candidate)
            continue
        for match in glob.glob(expanded, recursive=True):
            match_path = Path(match)
            if match_path.is_file() and match_path.suffix == ".parquet":
                paths.append(match_path)
    unique: List[Path] = []
    seen = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def read_parquet_range(
    parquet_tokens: Iterable[str],
    image_column: str,
    start: int,
    end: int,
    shard_idx: Optional[int] = None,
) -> List[np.ndarray]:
    if pq is None:
        raise RuntimeError("pyarrow is required for parquet operations")
    parquet_files = expand_parquet_tokens(parquet_tokens)
    if not parquet_files:
        raise FileNotFoundError("No parquet files located for the provided arguments")

    def read_from_file(pfile: "pq.ParquetFile", start_local: int, end_local: int, path: Path) -> List[np.ndarray]:
        frames_local: List[np.ndarray] = []
        current_local = 0
        target = end_local - start_local + 1
        collected_local = 0
        metadata_local = pfile.metadata
        num_groups = metadata_local.num_row_groups if metadata_local is not None else 1
        for rg_idx in range(num_groups):
            table = pfile.read_row_group(rg_idx, columns=[image_column])
            rows = table.to_pylist()
            for row in rows:
                if current_local > end_local:
                    return frames_local
                if current_local < start_local:
                    current_local += 1
                    continue
                value = row.get(image_column)
                if value is None:
                    raise ValueError(f"Row {current_local} in {path} missing column '{image_column}'")
                frames_local.append(to_rgb_array(value))
                collected_local += 1
                current_local += 1
                if collected_local >= target:
                    return frames_local
            current_local += 0  # explicit for clarity
        return frames_local

    if shard_idx is not None:
        if shard_idx < 0 or shard_idx >= len(parquet_files):
            raise IndexError(f"parquet_idx {shard_idx} out of range (len={len(parquet_files)})")

        cumulative = 0
        target_path = parquet_files[shard_idx]
        target_file = pq.ParquetFile(target_path)
        target_rows_meta = target_file.metadata.num_rows if target_file.metadata is not None else None
        target_rows = target_rows_meta if target_rows_meta is not None else end - start + 1
        for i in range(shard_idx):
            meta = pq.ParquetFile(parquet_files[i]).metadata
            if meta is not None:
                cumulative += meta.num_rows

        local_start = start - cumulative
        local_end = end - cumulative
        if local_start < 0 or local_end >= target_rows:
            # assume provided indices are already local to the shard
            local_start = max(0, start)
            local_end = min(target_rows - 1, end)
        return read_from_file(target_file, local_start, local_end, target_path)

    # Fallback: iterate across shards sequentially using global indices
    frames: List[np.ndarray] = []
    current = 0
    target_count = end - start + 1
    collected = 0
    for path in parquet_files:
        pq_file = pq.ParquetFile(path)
        rows = pq_file.metadata.num_rows if pq_file.metadata is not None else 0
        if current + rows <= start:
            current += rows
            continue
        frames.extend(
            read_from_file(
                pq_file,
                max(0, start - current),
                min(rows - 1, end - current),
                path,
            )
        )
        collected = len(frames)
        if collected >= target_count:
            return frames[:target_count]
        current += rows
        if current > end:
            break
    return frames


def to_rgb_array(value) -> np.ndarray:
    if isinstance(value, Image.Image):
        return np.array(value.convert("RGB"), dtype=np.uint8)
    if isinstance(value, (bytes, bytearray)):
        with Image.open(BytesIO(value)) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)
    if isinstance(value, memoryview):
        with Image.open(BytesIO(value.tobytes())) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)
    if isinstance(value, np.ndarray):
        arr = value
        if arr.dtype in (np.float32, np.float64):
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr
    if isinstance(value, dict):
        if value.get("path"):
            with Image.open(value["path"]) as img:
                return np.array(img.convert("RGB"), dtype=np.uint8)
        if value.get("bytes") is not None:
            with Image.open(BytesIO(value["bytes"])) as img:
                return np.array(img.convert("RGB"), dtype=np.uint8)
    raise ValueError(f"Unsupported image container from parquet: {type(value)!r}")


# -----------------------------------------------------------------------------
# Video export helpers
# -----------------------------------------------------------------------------


def segment_to_frames(
    segment: Segment,
    frame_paths: Optional[List[Path]],
    parquet_tokens: Optional[Sequence[str]],
    image_column: Optional[str],
) -> List[np.ndarray]:
    if frame_paths is not None:
        start = segment.start_total_frame
        end = segment.end_total_frame
        slice_paths = frame_paths[start : end + 1]
        frames = []
        for path in slice_paths:
            with Image.open(path) as img:
                frames.append(np.array(img.convert("RGB"), dtype=np.uint8))
        return frames
    if parquet_tokens is not None and image_column is not None:
        return read_parquet_range(
            parquet_tokens,
            image_column,
            segment.start_local_frame,
            segment.end_local_frame,
            shard_idx=segment.parquet_idx,
        )
    raise ValueError("Either frame_paths or parquet_tokens must be provided")


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for img in frames:
            writer.append_data(img)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sampled trajectory segments as videos")
    parser.add_argument("--segments-json", required=True, help="Segments JSON from segment_tracked_trajectory.py")
    parser.add_argument("--frames-dir", help="Directory containing ordered frames")
    parser.add_argument("--pattern", help="Glob pattern for frames (default: *.png)")
    parser.add_argument("--parquet", action="append", default=[], help="Parquet file(s)/directories/globs")
    parser.add_argument("--parquet-image-column", help="Column name containing images in parquet")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of segments to sample")
    parser.add_argument(
        "--segments",
        nargs="*",
        type=int,
        help="Explicit segment indices to export (overrides --sample-count)",
    )
    parser.add_argument("--random-seed", type=int, help="Random seed for sampling segments")
    parser.add_argument("--output-dir", required=True, help="Directory to write individual videos")
    parser.add_argument("--concat-output", help="Optional single video containing all sampled segments")
    parser.add_argument("--fps", type=int, default=15, help="FPS for exported videos")
    parser.add_argument(
        "--gap-frames",
        type=int,
        default=10,
        help="Blank frames inserted between segments when using --concat-output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    segments_path = Path(args.segments_json)
    segments = load_segments(segments_path)
    if not segments:
        print("[WARN] No segments found; exiting")
        return

    frame_paths: Optional[List[Path]] = None
    parquet_tokens: Optional[List[str]] = None
    image_column: Optional[str] = None

    if args.frames_dir:
        frame_paths = list_frames(Path(args.frames_dir), args.pattern)
    elif args.parquet:
        parquet_tokens = args.parquet
        if not args.parquet_image_column:
            raise ValueError("--parquet-image-column is required when using --parquet")
        image_column = args.parquet_image_column
    else:
        raise ValueError("Provide either --frames-dir or --parquet inputs")

    chosen = select_segments(segments, args.segments, args.sample_count, args.random_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concat_writer = None
    gap_frame = None
    if args.concat_output:
        concat_path = Path(args.concat_output)
        concat_writer = imageio.get_writer(concat_path, fps=args.fps)
        gap_frame = None

    for idx, segment in chosen:
        frames = segment_to_frames(segment, frame_paths, parquet_tokens, image_column)
        video_path = output_dir / f"segment_{idx:04d}.mp4"
        write_video(video_path, frames, fps=args.fps)
        print(f"[INFO] Wrote {video_path} ({segment.length} frames)")

        if concat_writer is not None:
            for array in frames:
                concat_writer.append_data(array)
            if args.gap_frames > 0:
                if gap_frame is None:
                    if frames:
                        gap_frame = np.zeros_like(frames[0])
                    elif frame_paths:
                        with Image.open(frame_paths[0]) as img:  # type: ignore[index]
                            template = np.array(img.convert("RGB"), dtype=np.uint8)
                        gap_frame = np.zeros_like(template)
                if gap_frame is not None:
                    for _ in range(args.gap_frames):
                        concat_writer.append_data(gap_frame)

    if concat_writer is not None:
        concat_writer.close()
        print(f"[INFO] Wrote concatenated video to {args.concat_output}")


if __name__ == "__main__":
    main()
