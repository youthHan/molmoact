"""Command line entry-point for DINO-based gripper tracking.

The script wraps :class:`MolmoAct.gripper_tracking.dino_gripper_tracker.DINOGripperTracker`
and exposes a simple workflow for trajectory recovery:

1. Provide a frame sequence via ``--frames`` or ``--frames-dir`` + ``--pattern``.
2. Specify the gripper patch index in the seed frame with ``--initial-patch``.
3. (Optional) Attach reference crops using ``--reference path=...,patch=...`` to
   bias the tracker when the gripper is occluded or visually ambiguous.

Example (seed frame 0, two references, output JSON + overlays)::

    python3 MolmoAct/scripts/track_gripper_trajectory.py \
        --frames-dir data/demos/run_01/frames --pattern "frame_*.png" \
        --initial-patch 187 \
        --reference path=refs/gripper_close.png,patch=42,weight=1.5,desc=closeup \
        --reference path=refs/gripper_side.png,patch=19,weight=0.8,start=10 \
        --output run_01_traj.json \
        --visualize-dir run_01_viz

Run with ``--print-example`` to see the same template inside the CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw

from MolmoAct.gripper_tracking import DINOGripperTracker, ReferencePatch, TrajectoryPoint


def render_overlays(frames: Iterable[str], trajectory: Iterable[TrajectoryPoint], output_dir: Path) -> None:
    """Write per-frame overlays with the tracked coordinates.

    A small red circle shows the smoothed gripper position, yellow lines connect
    consecutive frames, and the frame index is annotated next to the point. This
    provides a lightweight monitoring view without relying on a full GUI.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = list(frames)
    points_by_idx = {point.frame_idx: point for point in trajectory}
    previous_xy: Optional[Tuple[float, float]] = None

    for local_idx, frame_path in enumerate(frame_paths):
        frame_id = local_idx
        point = points_by_idx.get(frame_id)
        if point is None:
            continue

        with Image.open(frame_path).convert("RGB") as img:
            draw = ImageDraw.Draw(img)
            x = point.smoothed_x if point.smoothed_x is not None else point.x
            y = point.smoothed_y if point.smoothed_y is not None else point.y
            radius = 8
            bbox = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(bbox, outline=(255, 50, 50), width=3)

            if previous_xy is not None:
                draw.line((*previous_xy, x, y), fill=(255, 220, 0), width=2)
            draw.text((x + radius + 2, y - radius - 2), f"{frame_id}", fill=(255, 255, 255))

            previous_xy = (x, y)

            out_path = output_dir / f"{frame_id:05d}.png"
            img.save(out_path)


def parse_reference_arg(arg: str) -> ReferencePatch:
    """Parse reference spec of the form key=value,key=value."""
    parts = {}
    for item in arg.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Reference segment '{item}' must look like key=value")
        key, value = item.split("=", 1)
        parts[key.strip()] = value.strip()

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


def collect_frames(args: argparse.Namespace) -> List[str]:
    if args.frames:
        return [str(Path(p)) for p in args.frames]
    if not args.frames_dir:
        raise ValueError("Either --frames or --frames-dir must be provided")
    directory = Path(args.frames_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Frame directory '{directory}' not found")
    pattern = args.pattern or "*.png"
    frames = sorted(str(path) for path in directory.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No frames matching pattern '{pattern}' in '{directory}'")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track gripper trajectory using DINO patch similarities",
        epilog="Tip: use --print-example to see a fully-populated reference command.",
    )
    parser.add_argument("--frames", nargs="*", help="Explicit list of frame image paths (ordered)")
    parser.add_argument("--frames-dir", help="Directory containing frame images", default=None)
    parser.add_argument("--pattern", help="Glob pattern for frames within --frames-dir (default: *.png)")
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

    if args.print_example:
        example = (
            "python3 MolmoAct/scripts/track_gripper_trajectory.py "
            "--frames-dir demo/run_01/frames --pattern 'frame_*.png' "
            "--initial-patch 187 "
            "--reference path=refs/gripper_close.png,patch=42,weight=1.5,desc=closeup "
            "--reference path=refs/gripper_side.png,patch=19,weight=0.8,start=10 "
            "--weight-prev 1.0 --distance-penalty 0.02 --ema-alpha 0.3 "
            "--output run_01_traj.json --visualize-dir run_01_viz"
        )
        print("Example command:\n" + example)
        return
    frames = collect_frames(args)
    references = [parse_reference_arg(spec) for spec in args.reference]
    ema_alpha = None if args.ema_alpha < 0 else args.ema_alpha

    tracker = DINOGripperTracker(model_id=args.model_id)
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
