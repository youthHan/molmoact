"""Track gripper trajectories on pre-segmented clips.

This script consumes the segment metadata emitted by
``segment_tracked_trajectory.py`` together with the original raw frame source
(frame directory or parquet shards) and reruns the DINO-based tracker on a
per-segment basis. The output is one trajectory JSON per segment plus an
optional combined file.

Features
--------
- Honors segments that may span multiple parquet shards or be expressed in
  global frame coordinates.
- Can inherit initial patch indices from the original trajectory JSON or accept
  explicit overrides.
- Supports all tracker knobs: references, history toggles, EMA, distance
  penalty, etc.
- Optional visualization overlays per segment reuse the same annotation style
  as ``export_segment_videos.py``.

Example::

    python3 MolmoAct/scripts/track_segments.py \
        --segments-json run_01_segments.json \
        --trajectory-json run_01_traj.json \
        --parquet /mnt/data/libero/train-00000-of-00025.parquet \
        --parquet /mnt/data/libero/train-00001-of-00025.parquet \
        --parquet-image-column image \
        --initial-patch-source trajectory \
        --output-dir segment_tracks \
        --visualize-dir segment_overlays \
        --weight-prev 0.8 --distance-penalty 0.01

    # Sharded invocation (e.g., four GPUs)
    python3 MolmoAct/scripts/track_segments.py \\
        --segments-json run_01_segments.json \\
        --trajectory-json run_01_traj.json \\
        --parquet /mnt/data/libero/train-*.parquet \\
        --parquet-image-column image \\
        --num-shards 4 --shard-index 0 \\
        --output-dir segment_tracks_shard0

"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from MolmoAct.gripper_tracking import (
    DINOGripperTracker,
    ReferencePatch,
    TrajectoryPoint as TrackerTrajectoryPoint,
)

try:  # optional dependency for parquet reading
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class Segment:
    segment_idx: int
    start_index: int
    end_index: int
    length: int
    start_total_frame: int
    end_total_frame: int
    start_local_frame: int
    end_local_frame: int
    parquet_idx: Optional[int]
    task: Optional[str]


@dataclass
class TrajPoint:
    frame_idx: int
    total_frame_idx: int
    patch_idx: int
    x: float
    y: float
    smoothed_x: float
    smoothed_y: float
    score: float
    parquet_idx: Optional[int]
    task: Optional[str]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def load_segments(path: Path) -> List[Segment]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "segments" in data:
        segments_raw = data["segments"]
    elif isinstance(data, list):
        segments_raw = data
    else:
        raise ValueError("segments JSON must be a list or contain 'segments'")

    segments: List[Segment] = []
    for idx, entry in enumerate(segments_raw):
        start_index = int(entry.get("start_index", entry.get("start_frame", 0)))
        end_index = int(entry.get("end_index", entry.get("end_frame", start_index)))
        length = int(entry.get("length", end_index - start_index + 1))
        start_total = int(entry.get("start_total_frame", entry.get("start_frame", start_index)))
        end_total = int(entry.get("end_total_frame", entry.get("end_frame", end_index)))
        start_local = int(entry.get("start_frame", start_index))
        end_local = int(entry.get("end_frame", end_index))
        parquet_idx = entry.get("parquet_idx")
        parquet_idx = int(parquet_idx) if parquet_idx is not None else None
        task = entry.get("task")
        segments.append(
            Segment(
                segment_idx=idx,
                start_index=start_index,
                end_index=end_index,
                length=length,
                start_total_frame=start_total,
                end_total_frame=end_total,
                start_local_frame=start_local,
                end_local_frame=end_local,
                parquet_idx=parquet_idx,
                task=task,
            )
        )
    return segments


def load_trajectory(path: Path) -> List[TrajPoint]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, Sequence):
        raise ValueError("trajectory JSON must be a list")
    traj: List[TrajPoint] = []
    for entry in entries:
        frame_idx = int(entry.get("frame_idx", 0))
        total_idx = int(entry.get("total_frame_idx", frame_idx))
        parquet_idx = entry.get("parquet_idx")
        parquet_idx = int(parquet_idx) if parquet_idx is not None else None
        sx = entry.get("smoothed_x")
        sy = entry.get("smoothed_y")
        traj.append(
            TrajPoint(
                frame_idx=frame_idx,
                total_frame_idx=total_idx,
                patch_idx=int(entry.get("patch_idx", 0)),
                x=float(entry.get("x", 0.0)),
                y=float(entry.get("y", 0.0)),
                smoothed_x=float(sx if sx is not None else entry.get("x", 0.0)),
                smoothed_y=float(sy if sy is not None else entry.get("y", 0.0)),
                score=float(entry.get("score", 0.0)),
                parquet_idx=parquet_idx,
                task=entry.get("task"),
            )
        )
    return traj


def expand_parquet_tokens(tokens: Sequence[str]) -> List[Path]:
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
            path = Path(match)
            if path.is_file() and path.suffix == ".parquet":
                paths.append(path)
    unique: List[Path] = []
    seen = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


# -----------------------------------------------------------------------------
# Frame access helpers
# -----------------------------------------------------------------------------


def iter_segment_frames(
    segment: Segment,
    frame_paths: Optional[List[Path]],
    parquet_files: Optional[List[Path]],
    image_column: Optional[str],
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (global_frame_idx, image_array) for frames in segment."""

    if frame_paths is not None:
        for global_idx in range(segment.start_total_frame, segment.end_total_frame + 1):
            path = frame_paths[global_idx]
            with Image.open(path) as img:
                yield global_idx, np.array(img.convert("RGB"), dtype=np.uint8)
        return

    if parquet_files is None or image_column is None:
        raise ValueError("Either frame paths or parquet inputs must be provided")
    if pq is None:
        raise RuntimeError("pyarrow is required for parquet inputs")

    # If segment has single parquet index, treat indices as local
    if segment.parquet_idx is not None:
        shard_idx = segment.parquet_idx
        if shard_idx >= len(parquet_files):
            raise IndexError(
                f"Segment {segment.segment_idx} references parquet_idx={shard_idx} but only {len(parquet_files)} shards provided"
            )
        path = parquet_files[shard_idx]
        pf = pq.ParquetFile(path)
        for local_idx, array in _iter_parquet_range(pf, image_column, segment.start_local_frame, segment.end_local_frame):
            global_idx = segment.start_total_frame + (local_idx - segment.start_local_frame)
            yield global_idx, array
        return

    # Fall back to global indices across shards
    remaining = segment.length
    target_start = segment.start_total_frame
    target_end = segment.end_total_frame
    offset = 0
    for shard_idx, path in enumerate(parquet_files):
        if remaining <= 0:
            break
        pf = pq.ParquetFile(path)
        shard_rows = pf.metadata.num_rows if pf.metadata is not None else 0
        shard_start = offset
        shard_end = offset + shard_rows - 1
        if shard_end < target_start:
            offset += shard_rows
            continue
        local_start = max(0, target_start - shard_start)
        local_end = min(shard_rows - 1, target_end - shard_start)
        for local_idx, array in _iter_parquet_range(pf, image_column, local_start, local_end):
            global_idx = shard_start + local_idx
            if global_idx < target_start or global_idx > target_end:
                continue
            yield global_idx, array
            remaining -= 1
            if remaining <= 0:
                break
        offset += shard_rows


def _iter_parquet_range(
    pf: "pq.ParquetFile",
    column: str,
    start: int,
    end: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    current = 0
    for rg_idx in range(pf.metadata.num_row_groups if pf.metadata is not None else 1):
        table = pf.read_row_group(rg_idx, columns=[column])
        rows = table.to_pylist()
        for row_idx, row in enumerate(rows):
            if current > end:
                return
            if current < start:
                current += 1
                continue
            value = row.get(column) if isinstance(row, dict) else row
            yield current, to_rgb_array(value)
            current += 1


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
# Tracking
# -----------------------------------------------------------------------------


def build_reference_patches(ref_specs: Sequence[str]) -> List[ReferencePatch]:
    refs: List[ReferencePatch] = []
    for spec in ref_specs:
        parts = {}
        for item in spec.split(","):
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Reference spec '{item}' missing '='")
            key, value = item.split("=", 1)
            parts[key.strip()] = value.strip()
        if "path" not in parts or "patch" not in parts:
            raise ValueError("Reference spec must include path and patch keys")
        refs.append(
            ReferencePatch(
                image_path=parts["path"],
                patch_idx=int(parts["patch"]),
                weight=float(parts.get("weight", 1.0)),
                start_frame=int(parts.get("start", 0)),
                end_frame=int(parts["end"]) if "end" in parts else None,
                description=parts.get("desc"),
            )
        )
    return refs


def track_segment(
    tracker: DINOGripperTracker,
    frames_iter: Iterable[np.ndarray],
    initial_patch_idx: int,
    references: Optional[Sequence[ReferencePatch]],
    weight_prev: float,
    distance_penalty: float,
    ema_alpha: Optional[float],
    disable_history: bool,
    initial_reference_weight: float,
) -> List[TrackerTrajectoryPoint]:
    frames_list = [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in frames_iter]
    trajectory = tracker.track(
        frames=frames_list,
        initial_patch_idx=initial_patch_idx,
        references=references,
        weight_prev=weight_prev,
        distance_penalty=distance_penalty,
        ema_alpha=ema_alpha,
        use_history_similarity=not disable_history,
        initial_reference_weight=initial_reference_weight,
    )
    for frame in frames_list:
        frame.close()
    return trajectory


# -----------------------------------------------------------------------------
# Visualization helper (optional)
# -----------------------------------------------------------------------------


def annotate_frame(
    array: np.ndarray,
    point: TrackerTrajectoryPoint,
    global_idx: int,
    task: Optional[str],
    overlay: bool,
    marker_radius: int,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    if not overlay:
        return array
    img = Image.fromarray(array)
    draw = ImageDraw.Draw(img)
    x = point.smoothed_x if point.smoothed_x is not None else point.x
    y = point.smoothed_y if point.smoothed_y is not None else point.y
    r = marker_radius
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 50, 50), width=max(2, r // 2))
    text_lines = [
        f"global:{global_idx}",
        f"seg_frame:{point.frame_idx}",
        f"({x:.1f},{y:.1f})"
    ]
    if task:
        text_lines.append(task)
    text = "\n".join(text_lines)
    draw.text((x + r + 4, y - r - 4), text, fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(img, dtype=np.uint8)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track gripper trajectories per segment")
    parser.add_argument("--segments-json", required=True, help="Segment metadata JSON")
    parser.add_argument("--trajectory-json", required=True, help="Original trajectory JSON (to seed patches)")
    parser.add_argument("--frames-dir", help="Directory containing ordered frames")
    parser.add_argument("--pattern", help="Glob pattern for frames (default: *.png)")
    parser.add_argument("--parquet", action="append", default=[], help="Parquet file(s)/directories/globs")
    parser.add_argument("--parquet-image-column", help="Column name containing images in parquet")

    parser.add_argument("--segments", nargs="*", type=int, help="Segment indices to track (default all)")
    parser.add_argument("--sample-count", type=int, default=None, help="Randomly sample N segments")
    parser.add_argument("--random-seed", type=int, help="Seed for segment sampling")

    parser.add_argument("--initial-patch-source", choices=["trajectory", "first"], default="trajectory")
    parser.add_argument("--initial-patch", type=int, help="Override initial patch index for all segments")

    parser.add_argument("--reference", action="append", default=[], help="Optional reference specs path=...,patch=... etc")
    parser.add_argument("--weight-prev", type=float, default=1.0)
    parser.add_argument("--distance-penalty", type=float, default=0.0)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--disable-history", action="store_true")
    parser.add_argument("--initial-reference-weight", type=float, default=0.0)

    parser.add_argument("--model-id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--device", default=None)
    parser.add_argument("--patch-size-override", type=int, default=None)

    parser.add_argument("--output-dir", required=True, help="Directory to store per-segment trajectories")
    parser.add_argument("--combined-output", help="Optional combined trajectory JSON")
    parser.add_argument("--visualize-dir", help="Optional directory for overlay frames")
    parser.add_argument("--overlay-marker-radius", type=int, default=8)
    parser.add_argument("--overlay-disable", action="store_true")
    parser.add_argument("--fps", type=int, default=15, help="If visualizing, set assumed FPS for logging")
    parser.add_argument("--num-shards", type=int, help="Total number of shards for distributed runs")
    parser.add_argument("--shard-index", type=int, help="Zero-based shard index for this process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    segments = load_segments(Path(args.segments_json))
    if not segments:
        print("[WARN] No segments found; exiting")
        return

    traj_points = load_trajectory(Path(args.trajectory_json))
    traj_by_index = {pt.total_frame_idx: pt for pt in traj_points}

    frame_paths: Optional[List[Path]] = None
    parquet_files: Optional[List[Path]] = None
    if args.frames_dir:
        pattern = args.pattern or "*.png"
        frame_paths = sorted(Path(args.frames_dir).glob(pattern))
        if not frame_paths:
            raise FileNotFoundError(f"No frames matching {pattern} in {args.frames_dir}")
    elif args.parquet:
        parquet_files = expand_parquet_tokens(args.parquet)
        if not parquet_files:
            raise FileNotFoundError("No parquet files located for tracking")
        if not args.parquet_image_column:
            raise ValueError("--parquet-image-column is required when using --parquet")
    else:
        raise ValueError("Provide either --frames-dir or --parquet inputs")

    # Segment selection
    selected_segments: List[Segment]
    if args.num_shards is not None or args.shard_index is not None:
        if args.num_shards is None or args.shard_index is None:
            raise ValueError("Both --num-shards and --shard-index must be provided together")
        if args.num_shards < 1:
            raise ValueError("--num-shards must be >= 1")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard-index must be in [0, num_shards)")
        if args.sample_count is not None:
            raise ValueError("--sample-count cannot be used when sharding")

    if args.segments:
        selected_segments = [segments[idx] for idx in args.segments]
    elif args.sample_count:
        import random

        rng = random.Random(args.random_seed)
        selected_segments = rng.sample(segments, min(args.sample_count, len(segments)))
        selected_segments.sort(key=lambda s: s.segment_idx)
    else:
        selected_segments = segments

    if args.num_shards is not None:
        selected_segments = [
            seg for seg in selected_segments if seg.segment_idx % args.num_shards == args.shard_index
        ]
        print(
            f"[INFO] Shard {args.shard_index}/{args.num_shards} processing {len(selected_segments)} segments"
        )

    references = build_reference_patches(args.reference)

    tracker = DINOGripperTracker(
        model_id=args.model_id,
        device=args.device,
        patch_size_override=args.patch_size_override,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_dir = Path(args.visualize_dir) if args.visualize_dir else None
    if visualize_dir:
        visualize_dir.mkdir(parents=True, exist_ok=True)
        font = ImageFont.load_default()
    else:
        font = None

    combined_records: List[Dict] = []

    for segment in selected_segments:
        print(f"[INFO] Tracking segment {segment.segment_idx} length={segment.length} task={segment.task}")
        frames_iter = iter_segment_frames(
            segment,
            frame_paths,
            parquet_files,
            args.parquet_image_column,
        )

        if args.initial_patch is not None:
            initial_patch_idx = args.initial_patch
        elif args.initial_patch_source == "first":
            first_pt = traj_by_index.get(segment.start_total_frame)
            initial_patch_idx = first_pt.patch_idx if first_pt else 0
        else:  # trajectory
            first_pt = traj_by_index.get(segment.start_total_frame)
            if first_pt is None:
                raise ValueError(
                    f"Segment {segment.segment_idx} start frame {segment.start_total_frame} not found in trajectory JSON"
                )
            initial_patch_idx = first_pt.patch_idx

        tracker_points = track_segment(
            tracker,
            (frame for _, frame in frames_iter),
            initial_patch_idx=initial_patch_idx,
            references=references,
            weight_prev=args.weight_prev,
            distance_penalty=args.distance_penalty,
            ema_alpha=None if args.ema_alpha < 0 else args.ema_alpha,
            disable_history=args.disable_history,
            initial_reference_weight=args.initial_reference_weight,
        )

        segment_records: List[Dict] = []
        for step_idx, tracker_point in enumerate(tracker_points):
            global_idx = segment.start_total_frame + step_idx
            record = {
                "segment_idx": segment.segment_idx,
                "segment_frame_idx": step_idx,
                "global_frame_idx": global_idx,
                "patch_idx": tracker_point.patch_idx,
                "x": tracker_point.x,
                "y": tracker_point.y,
                "smoothed_x": tracker_point.smoothed_x,
                "smoothed_y": tracker_point.smoothed_y,
                "score": tracker_point.score,
                "similarity_prev": tracker_point.similarity_prev,
                "similarity_refs": tracker_point.similarity_refs,
                "task": segment.task,
            }
            segment_records.append(record)
            combined_records.append(record)

        out_path = output_dir / f"segment_{segment.segment_idx:04d}.json"
        out_path.write_text(json.dumps(segment_records, indent=2))
        print(f"[INFO] Wrote {out_path}")

        if visualize_dir and font is not None:
            # re-generate frames for visualization (could be optimized by caching)
            frame_iter = iter_segment_frames(
                segment,
                frame_paths,
                parquet_files,
                args.parquet_image_column,
            )
            for step_idx, (global_idx, frame) in enumerate(frame_iter):
                if step_idx >= len(tracker_points):
                    break
                overlay = annotate_frame(
                    frame,
                    tracker_points[step_idx],
                    global_idx=global_idx,
                    task=segment.task,
                    overlay=not args.overlay_disable,
                    marker_radius=args.overlay_marker_radius,
                    font=font,
                )
                img = Image.fromarray(overlay)
                img.save(visualize_dir / f"segment_{segment.segment_idx:04d}_{step_idx:05d}.png")
                img.close()

    if args.combined_output:
        Path(args.combined_output).write_text(json.dumps(combined_records, indent=2))
        print(f"[INFO] Wrote combined trajectory to {args.combined_output}")


if __name__ == "__main__":
    main()
