"""Command line entry-point for DINO-based gripper tracking.

The script wraps :class:`MolmoAct.gripper_tracking.dino_gripper_tracker.DINOGripperTracker`
and exposes a simple workflow for trajectory recovery:

1. Provide frames either from parquet shards (``--parquet``) or individual files
   via ``--frames`` / ``--frames-dir`` + ``--pattern``.
2. Specify the gripper patch index in the seed frame with ``--initial-patch``.
3. (Optional) Attach reference crops using ``--reference path=...,patch=...`` to
   bias the tracker when the gripper is occluded or visually ambiguous.

Examples::

    # Directory-based frames
    python3 MolmoAct/scripts/track_gripper_trajectory.py \
        --frames-dir data/demos/run_01/frames --pattern "frame_*.png" \
        --initial-patch 187 \
        --reference path=refs/gripper_close.png,patch=42,weight=1.5,desc=closeup \
        --reference path=refs/gripper_side.png,patch=19,weight=0.8,start=10 \
        --output run_01_traj.json \
        --visualize-dir run_01_viz

    # Parquet-based frames
    python3 MolmoAct/scripts/track_gripper_trajectory.py \
        --parquet data/libero/run_01.parquet \
        --parquet-image-column image \
        --initial-patch 96 \
        --output run_01_traj.json \
        --visualize-dir run_01_viz

    # Dump patch indices for frame 0 and a reference crop, then exit
    python3 MolmoAct/scripts/track_gripper_trajectory.py \
        --frames-dir data/demos/run_01/frames \
        --grid-frame frame=0,out=run_01_frame0_grid.png \
        --grid-reference path=refs/gripper_close.png,out=ref_grid.png \
        --grid-only

Run with ``--print-example`` to see both templates inside the CLI.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw

from MolmoAct.gripper_tracking import DINOGripperTracker, ReferencePatch, TrajectoryPoint

try:  # Optional dependency for parquet reading
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - handled at runtime
    pq = None


FrameInput = Union[str, Image.Image]


def _to_pil_from_any(value: object) -> Image.Image:
    """Best-effort conversion of parquet cell contents into a PIL image."""

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, memoryview):
        return Image.open(io.BytesIO(value.tobytes())).convert("RGB")
    if isinstance(value, str) and os.path.exists(value):
        return Image.open(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    try:  # optional numpy support for array-like columns
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            arr = value
            if arr.dtype in (np.float32, np.float64):
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr)
    except Exception:  # pragma: no cover - numpy optional
        pass
    raise ValueError(f"Unsupported image container fetched from parquet: {type(value)!r}")


def render_overlays(frames: Sequence[FrameInput], trajectory: Iterable[TrajectoryPoint], output_dir: Path) -> None:
    """Write per-frame overlays with the tracked coordinates.

    A small red circle shows the smoothed gripper position, yellow lines connect
    consecutive frames, and the frame index is annotated next to the point. This
    provides a lightweight monitoring view without relying on a full GUI.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_inputs: List[FrameInput] = list(frames)
    points_by_idx = {point.frame_idx: point for point in trajectory}
    previous_xy: Optional[Tuple[float, float]] = None

    for local_idx, frame_input in enumerate(frame_inputs):
        frame_id = local_idx
        point = points_by_idx.get(frame_id)
        if point is None:
            continue
        current_xy = _point_to_tuple(point)

        if isinstance(frame_input, str):
            with Image.open(frame_input).convert("RGB") as img:
                _draw_overlay(img, point, frame_id, previous_xy)
                previous_xy = current_xy
                out_path = output_dir / f"{frame_id:05d}.png"
                img.save(out_path)
        else:
            img = frame_input.copy()
            try:
                _draw_overlay(img, point, frame_id, previous_xy)
                previous_xy = current_xy
                out_path = output_dir / f"{frame_id:05d}.png"
                img.save(out_path)
            finally:
                img.close()


def _point_to_tuple(point: TrajectoryPoint) -> Tuple[float, float]:
    x = point.smoothed_x if point.smoothed_x is not None else point.x
    y = point.smoothed_y if point.smoothed_y is not None else point.y
    return (x, y)


def _draw_overlay(img: Image.Image, point: TrajectoryPoint, frame_id: int, previous_xy: Optional[Tuple[float, float]]) -> None:
    draw = ImageDraw.Draw(img)
    x = point.smoothed_x if point.smoothed_x is not None else point.x
    y = point.smoothed_y if point.smoothed_y is not None else point.y
    radius = 8
    bbox = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bbox, outline=(255, 50, 50), width=3)

    if previous_xy is not None:
        draw.line((*previous_xy, x, y), fill=(255, 220, 0), width=2)
    draw.text((x + radius + 2, y - radius - 2), f"{frame_id}", fill=(255, 255, 255))


def parse_kv_string(arg: str) -> Dict[str, str]:
    parts: Dict[str, str] = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Segment '{item}' must look like key=value")
        key, value = item.split("=", 1)
        parts[key.strip()] = value.strip()
    return parts


def parse_reference_arg(arg: str) -> ReferencePatch:
    """Parse reference spec of the form key=value,key=value."""
    parts = parse_kv_string(arg)

    if "patch" not in parts:
        raise ValueError("Reference spec must provide 'patch=<idx>'")

    patch_idx = int(parts["patch"])
    image_path = parts.get("path")
    weight = float(parts.get("weight", 1.0))
    start_frame = int(parts.get("start", 0))
    end_frame = int(parts["end"]) if "end" in parts else None
    description = parts.get("desc")

    if not image_path:
        raise ValueError("Reference spec must include 'path=<image_path>'")

    return ReferencePatch(
        image_path=image_path,
        patch_idx=patch_idx,
        weight=weight,
        start_frame=start_frame,
        end_frame=end_frame,
        description=description,
    )


@dataclass
class GridDumpSpec:
    kind: str  # 'frame' or 'path'
    output: Path
    frame_idx: Optional[int] = None
    path: Optional[str] = None
    highlight: Optional[int] = None


def parse_grid_frame_arg(arg: str) -> GridDumpSpec:
    parts = parse_kv_string(arg)
    if "frame" not in parts or "out" not in parts:
        raise ValueError("Grid frame spec requires 'frame=<idx>,out=<path>'")
    frame_idx = int(parts["frame"])
    output = Path(parts["out"])
    highlight = int(parts["highlight"]) if "highlight" in parts else None
    return GridDumpSpec(kind="frame", output=output, frame_idx=frame_idx, highlight=highlight)


def parse_grid_reference_arg(arg: str) -> GridDumpSpec:
    parts = parse_kv_string(arg)
    if "path" not in parts or "out" not in parts:
        raise ValueError("Grid reference spec requires 'path=<image>,out=<path>'")
    path = parts["path"]
    output = Path(parts["out"])
    highlight = int(parts["highlight"]) if "highlight" in parts else None
    return GridDumpSpec(kind="path", output=output, path=path, highlight=highlight)


def expand_parquet_inputs(tokens: Iterable[str]) -> List[Path]:
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
            m_path = Path(match)
            if m_path.is_file() and m_path.suffix == ".parquet":
                paths.append(m_path)
    unique = []
    seen = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def load_parquet_frames(
    parquet_tokens: Iterable[str],
    image_column: str,
    start_index: int = 0,
    stop_index: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> List[Image.Image]:
    if pq is None:
        raise RuntimeError("pyarrow is required when using --parquet inputs.")

    parquet_files = expand_parquet_inputs(parquet_tokens)
    if not parquet_files:
        raise FileNotFoundError("No .parquet files found for the provided --parquet arguments")

    frames: List[Image.Image] = []
    global_index = 0
    collected = 0
    stop = stop_index if stop_index is not None else float("inf")

    for path in parquet_files:
        pq_file = pq.ParquetFile(path)
        metadata = pq_file.metadata
        num_row_groups = metadata.num_row_groups if metadata is not None else 1
        for rg_idx in range(num_row_groups):
            table = pq_file.read_row_group(rg_idx, columns=[image_column])
            for row in table.to_pylist():
                if global_index < start_index:
                    global_index += 1
                    continue
                if global_index >= stop:
                    return frames
                if max_frames is not None and collected >= max_frames:
                    return frames
                value = row.get(image_column)
                if value is None:
                    raise ValueError(
                        f"Row {global_index} in {path} is missing column '{image_column}'."
                    )
                frames.append(_to_pil_from_any(value))
                collected += 1
                global_index += 1
            if global_index >= stop or (max_frames is not None and collected >= max_frames):
                return frames

    return frames


def collect_frames(args: argparse.Namespace) -> List[FrameInput]:
    if args.parquet:
        if args.frames or args.frames_dir:
            raise ValueError("When using --parquet, do not also provide --frames or --frames-dir")
        frames = load_parquet_frames(
            args.parquet,
            image_column=args.parquet_image_column,
            start_index=args.parquet_start,
            stop_index=args.parquet_stop,
            max_frames=args.max_frames,
        )
        if not frames:
            raise RuntimeError("No frames extracted from parquet inputs")
        return frames

    if args.frames:
        frames = [str(Path(p)) for p in args.frames]
    else:
        if not args.frames_dir:
            raise ValueError("Either --frames, --frames-dir, or --parquet must be provided")
        directory = Path(args.frames_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Frame directory '{directory}' not found")
        pattern = args.pattern or "*.png"
        frames = sorted(str(path) for path in directory.glob(pattern))
        if not frames:
            raise FileNotFoundError(f"No frames matching pattern '{pattern}' in '{directory}'")

    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    return frames


def dump_patch_grids(
    tracker: DINOGripperTracker,
    frames: Sequence[FrameInput],
    frame_specs: Sequence[GridDumpSpec],
    reference_specs: Sequence[GridDumpSpec],
    annotate: bool,
) -> None:
    for spec in frame_specs:
        assert spec.frame_idx is not None
        if spec.frame_idx < 0 or spec.frame_idx >= len(frames):
            raise IndexError(f"--grid-frame index {spec.frame_idx} is out of bounds (len={len(frames)})")
        frame_input = frames[spec.frame_idx]
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        overlay = tracker.render_patch_grid(frame_input, highlight_idx=spec.highlight, annotate=annotate)
        overlay.save(spec.output)
        overlay.close()

    for spec in reference_specs:
        assert spec.path is not None
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        overlay = tracker.render_patch_grid(spec.path, highlight_idx=spec.highlight, annotate=annotate)
        overlay.save(spec.output)
        overlay.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track gripper trajectory using DINO patch similarities",
        epilog="Tip: use --print-example to see a fully-populated reference command.",
    )
    parser.add_argument("--frames", nargs="*", help="Explicit list of frame image paths (ordered)")
    parser.add_argument("--frames-dir", help="Directory containing frame images", default=None)
    parser.add_argument("--pattern", help="Glob pattern for frames within --frames-dir (default: *.png)", default=None)
    parser.add_argument(
        "--parquet",
        action="append",
        default=[],
        help="Parquet file(s), directories, or glob patterns providing frames",
    )
    parser.add_argument(
        "--parquet-image-column",
        default="image",
        help="Column in the parquet rows that stores RGB frames (default: image)",
    )
    parser.add_argument(
        "--parquet-start",
        type=int,
        default=0,
        help="Inclusive start row index when reading from parquet",
    )
    parser.add_argument(
        "--parquet-stop",
        type=int,
        default=None,
        help="Exclusive stop row index when reading from parquet",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames processed",
    )
    parser.add_argument(
        "--grid-frame",
        action="append",
        default=[],
        help="Dump patch grid for a frame: frame=<idx>,out=<path>[,highlight=<idx>]",
    )
    parser.add_argument(
        "--grid-reference",
        action="append",
        default=[],
        help="Dump patch grid for an image file: path=<img>,out=<path>[,highlight=<idx>]",
    )
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Generate requested patch grids and exit without tracking",
    )
    parser.add_argument(
        "--grid-hide-indices",
        action="store_true",
        help="Skip numeric annotations when dumping patch grids",
    )
    parser.add_argument("--initial-patch", type=int, required=True, help="Patch index for the seed frame")
    parser.add_argument("--initial-frame", type=int, default=0, help="Frame index for the seed patch")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Reference crop spec as key=value pairs (repeatable). Required keys: path,patch",
    )
    parser.add_argument("--weight-prev", type=float, default=1.0, help="Weight for previous-frame similarity")
    parser.add_argument("--distance-penalty", type=float, default=0.0, help="Distance penalty per pixel")
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.3,
        help="EMA factor for smoothed pixel coordinates (set negative to disable)",
    )
    parser.add_argument("--model-id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--output", help="Path to save trajectory as JSON", default=None)
    parser.add_argument(
        "--visualize-dir",
        help="Optional directory to write frame overlays with tracked coordinates",
        default=None,
    )
    parser.add_argument(
        "--print-example",
        action="store_true",
        help="Print an example command line showing reference usage and exit",
    )

    args = parser.parse_args()

    grid_frame_specs = [parse_grid_frame_arg(spec) for spec in args.grid_frame]
    grid_reference_specs = [parse_grid_reference_arg(spec) for spec in args.grid_reference]

    if args.print_example:
        examples = [
            (
                "Directory frames",
                "python3 MolmoAct/scripts/track_gripper_trajectory.py "
                "--frames-dir demo/run_01/frames --pattern 'frame_*.png' "
                "--initial-patch 187 "
                "--reference path=refs/gripper_close.png,patch=42,weight=1.5,desc=closeup "
                "--reference path=refs/gripper_side.png,patch=19,weight=0.8,start=10 "
                "--weight-prev 1.0 --distance-penalty 0.02 --ema-alpha 0.3 "
                "--output run_01_traj.json --visualize-dir run_01_viz",
            ),
            (
                "Parquet frames",
                "python3 MolmoAct/scripts/track_gripper_trajectory.py "
                "--parquet data/libero/run_01.parquet "
                "--parquet-image-column image "
                "--parquet-start 0 --max-frames 300 "
                "--initial-patch 96 "
                "--output run_01_traj.json --visualize-dir run_01_viz",
            ),
        ]
        for title, cmd in examples:
            print(f"{title}:\n  {cmd}\n")
        return
    frames = collect_frames(args)
    references = [parse_reference_arg(spec) for spec in args.reference]
    ema_alpha = None if args.ema_alpha < 0 else args.ema_alpha

    tracker = DINOGripperTracker(model_id=args.model_id)

    if grid_frame_specs or grid_reference_specs:
        dump_patch_grids(
            tracker,
            frames,
            grid_frame_specs,
            grid_reference_specs,
            annotate=not args.grid_hide_indices,
        )
        if args.grid_only:
            return

    trajectory = tracker.track(
        frames=frames,
        initial_patch_idx=args.initial_patch,
        initial_frame_idx=args.initial_frame,
        references=references,
        weight_prev=args.weight_prev,
        distance_penalty=args.distance_penalty,
        ema_alpha=ema_alpha,
    )

    if args.output:
        payload = [
            {
                "frame_idx": point.frame_idx,
                "patch_idx": point.patch_idx,
                "x": point.x,
                "y": point.y,
                "smoothed_x": point.smoothed_x,
                "smoothed_y": point.smoothed_y,
                "score": point.score,
                "similarity_prev": point.similarity_prev,
                "similarity_refs": point.similarity_refs,
            }
            for point in trajectory
        ]
        Path(args.output).write_text(json.dumps(payload, indent=2))

    if args.visualize_dir:
        render_overlays(frames, trajectory, Path(args.visualize_dir))

    print("frame_idx\tpatch_idx\tx\ty\tscore")
    for point in trajectory:
        sx = point.smoothed_x if point.smoothed_x is not None else point.x
        sy = point.smoothed_y if point.smoothed_y is not None else point.y
        print(f"{point.frame_idx}\t{point.patch_idx}\t{sx:.2f}\t{sy:.2f}\t{point.score:.4f}")


if __name__ == "__main__":
    main()
