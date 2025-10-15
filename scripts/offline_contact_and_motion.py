"""Offline contact detection and motion segmentation around a tracked gripper.

This script consumes:
- A sequence of frames (directory pattern or parquet column)
- A gripper trajectory JSON (from track_gripper_trajectory.py)

And produces:
- Detected contact frame indices (one or more)
- Approximated gripper-facing edge points near contact
- Motion clusters from dense/sparse point tracks (CoTracker or LK fallback)
- Per-cluster binary masks per frame (optional overlays)
- A compact JSON summary of results

Dependencies:
- numpy, PIL
- Optional: torch + co-tracker (if checkpoints provided)
- Optional: OpenCV (for LK fallback and simple optical flow-based edge proxy)

Usage example:

  python3 MolmoAct/scripts/offline_contact_and_motion.py \
    --frames-dir demo/run_01/frames --pattern "frame_*.png" \
    --gripper-json run_01_traj.json \
    --cotracker-checkpoint co-tracker/checkpoints/scaled_offline.pth \
    --grid-size 20 \
    --output-dir outputs/run_01 \
    --write-overlays

Notes:
- If --cotracker-checkpoint is missing, the script falls back to a
  pyramidal LK tracker on a regular grid of points.
- Edge estimation for the gripper near contact uses a local optical-flow
  affinity test when OpenCV is available; otherwise, it approximates the
  forward edge by projecting along the gripper velocity.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


# -----------------------------
# Utilities: frame ingestion
# -----------------------------

try:  # parquet is optional; only used when --parquet is provided
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None


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
        for match in list(Path().glob(expanded)):
            m = Path(match)
            if m.is_file() and m.suffix == ".parquet":
                paths.append(m)
    unique: List[Path] = []
    seen = set()
    for p in sorted(paths):
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def _to_pil_from_any(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        from io import BytesIO

        buf = bytes(value) if isinstance(value, memoryview) else value
        return Image.open(BytesIO(buf)).convert("RGB")
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            arr = value
            if arr.dtype in (np.float32, np.float64):
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr)
    except Exception:  # pragma: no cover
        pass
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            from io import BytesIO

            return Image.open(BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str) and Path(value).exists():
        return Image.open(value).convert("RGB")
    raise ValueError(f"Unsupported parquet image cell type: {type(value)!r}")


def load_frames_from_parquet(
    parquet_tokens: Sequence[str],
    image_column: str,
    start_index: int = 0,
    stop_index: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> List[Image.Image]:
    if pq is None:
        raise RuntimeError("pyarrow is required for --parquet inputs")
    files = expand_parquet_tokens(parquet_tokens)
    if not files:
        raise FileNotFoundError("No parquet files located for --parquet inputs")
    frames: List[Image.Image] = []
    global_idx = 0
    stop = stop_index if stop_index is not None else float("inf")
    for path in files:
        pf = pq.ParquetFile(path)
        meta = pf.metadata
        num_groups = meta.num_row_groups if meta is not None else 1
        for rg in range(num_groups):
            table = pf.read_row_group(rg, columns=[image_column])
            for row in table.to_pylist():
                if global_idx < start_index:
                    global_idx += 1
                    continue
                if global_idx >= stop:
                    return frames
                if max_frames is not None and len(frames) >= max_frames:
                    return frames
                value = row.get(image_column) if isinstance(row, dict) else row
                frames.append(_to_pil_from_any(value).convert("RGB"))
                global_idx += 1
    return frames


def collect_frames(args: argparse.Namespace) -> List[Image.Image]:
    if args.parquet:
        return load_frames_from_parquet(
            args.parquet,
            image_column=args.parquet_image_column,
            start_index=args.parquet_start,
            stop_index=args.parquet_stop,
            max_frames=args.max_frames,
        )
    # directory pattern
    if not args.frames_dir:
        raise ValueError("Provide --frames-dir or --parquet inputs")
    pattern = args.pattern or "*.png"
    paths = sorted(Path(args.frames_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No frames matching {pattern} in {args.frames_dir}")
    if args.max_frames is not None:
        paths = paths[: args.max_frames]
    frames: List[Image.Image] = []
    for p in paths:
        with Image.open(p) as img:
            frames.append(img.convert("RGB"))
    return frames


# -----------------------------
# Gripper trajectory
# -----------------------------


@dataclass
class GripperPoint:
    frame_idx: int
    x: float
    y: float


def load_gripper_track(path: Path) -> List[GripperPoint]:
    data = json.loads(path.read_text())
    if not isinstance(data, Sequence):
        raise ValueError("Gripper JSON must be a list")
    pts: List[GripperPoint] = []
    for row in data:
        sx = row.get("smoothed_x")
        sy = row.get("smoothed_y")
        x = float(sx if sx is not None else row.get("x", 0.0))
        y = float(sy if sy is not None else row.get("y", 0.0))
        pts.append(GripperPoint(frame_idx=int(row.get("frame_idx", len(pts))), x=x, y=y))
    pts.sort(key=lambda p: p.frame_idx)
    return pts


def smooth_signal(seq: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1:
        return seq
    pad = win // 2
    kernel = np.ones(win, dtype=np.float64) / win
    padded = np.pad(seq, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def derive_velocity_and_jerk(gripper: Sequence[GripperPoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array([p.x for p in gripper], dtype=np.float64)
    ys = np.array([p.y for p in gripper], dtype=np.float64)
    vx = np.gradient(xs)
    vy = np.gradient(ys)
    speed = np.hypot(vx, vy)
    ax = np.gradient(vx)
    ay = np.gradient(vy)
    jerk = np.hypot(ax, ay)
    return speed, jerk, np.stack([vx, vy], axis=-1)


# -----------------------------
# Tracking: CoTracker or LK fallback
# -----------------------------


def try_cotracker_predictor(checkpoint: Optional[str], window_len: int = 60):
    if checkpoint is None:
        return None
    try:
        from cotracker.predictor import CoTrackerPredictor  # type: ignore

        if not Path(checkpoint).exists():
            print(f"[WARN] CoTracker checkpoint not found: {checkpoint}; falling back to LK")
            return None
        model = CoTrackerPredictor(checkpoint=checkpoint, offline=True, window_len=window_len)
        return model
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Failed to initialize CoTracker: {e}; falling back to LK")
        return None


def frames_to_tensor(frames: Sequence[Image.Image]) -> "torch.Tensor":  # type: ignore
    import torch

    arr = np.stack([np.array(im, dtype=np.uint8) for im in frames], axis=0)  # T,H,W,3
    arr = arr.transpose(0, 3, 1, 2)  # T,3,H,W
    tensor = torch.from_numpy(arr).float() / 255.0
    return tensor.unsqueeze(0)  # 1,T,3,H,W


def run_cotracker(
    frames: Sequence[Image.Image],
    grid_size: int,
    checkpoint: str,
    query_frame: int = 0,
    backward: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    model = try_cotracker_predictor(checkpoint)
    if model is None:
        return run_lk_grid(frames, grid_size)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    video = frames_to_tensor(frames).to(device)
    tracks, vis = model(video, queries=None, segm_mask=None, grid_size=grid_size, grid_query_frame=query_frame, backward_tracking=backward)
    # shapes: tracks [B,T,N,2], vis [B,T,N]
    tracks_np = tracks[0].detach().cpu().numpy()
    vis_np = vis[0].detach().cpu().numpy().astype(bool)
    return tracks_np, vis_np


def run_lk_grid(frames: Sequence[Image.Image], grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV is required for LK fallback but is not available") from e

    imgs = [np.array(im.convert("L"), dtype=np.uint8) for im in frames]
    H, W = imgs[0].shape

    xs = np.linspace(0, W - 1, grid_size)
    ys = np.linspace(0, H - 1, grid_size)
    gx, gy = np.meshgrid(xs, ys)
    pts0 = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1).astype(np.float32)

    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    N = pts0.shape[0]
    T = len(imgs)
    traj = np.zeros((T, N, 2), dtype=np.float32)
    vis = np.zeros((T, N), dtype=bool)
    traj[0] = pts0
    vis[0] = True

    prev = imgs[0]
    prev_pts = pts0
    prev_status = np.ones((N,), dtype=np.uint8)
    for t in range(1, T):
        next_img = imgs[t]
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev, next_img, prev_pts, None, **lk_params)
        status = status.reshape(-1)
        # Persist previous positions for lost points
        prev_pts = np.where(status[:, None] == 1, next_pts, prev_pts)
        traj[t] = prev_pts
        vis[t] = status.astype(bool)
        prev = next_img
    return traj.astype(np.float32), vis


# -----------------------------
# Contact detection
# -----------------------------


def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)


def detect_contact_frames(
    gripper_xy: np.ndarray,  # [T,2]
    gripper_speed: np.ndarray,  # [T]
    gripper_jerk: np.ndarray,  # [T]
    tracks_xy: np.ndarray,  # [T,N,2]
    vis: np.ndarray,  # [T,N]
    local_radius: float = 32.0,
    min_gap: int = 8,
    jerk_z: float = 2.5,
    local_speed_z: float = 2.0,
) -> List[int]:
    T, N, _ = tracks_xy.shape
    # distance to nearest track point
    diffs = tracks_xy - gripper_xy[:, None, :]
    dists = np.linalg.norm(diffs, axis=-1)
    dists = np.where(vis, dists, np.inf)

    # local neighborhood mask
    local_mask = (dists <= local_radius) & vis

    # local speeds
    vx = np.gradient(tracks_xy[..., 0], axis=0)
    vy = np.gradient(tracks_xy[..., 1], axis=0)
    local_speed = np.linalg.norm(np.stack([vx, vy], axis=-1), axis=-1)  # T,N
    local_speed = np.where(local_mask, local_speed, 0.0)
    local_speed_sum = local_speed.sum(axis=1)

    jerk_zs = robust_zscore(gripper_jerk)
    local_zs = robust_zscore(np.gradient(local_speed_sum))

    score = np.maximum(0.0, jerk_zs) + 0.7 * np.maximum(0.0, local_zs)

    # peaks with thresholds
    candidates: List[int] = []
    last = -min_gap
    for t in range(1, T - 1):
        if t - last < min_gap:
            continue
        if jerk_zs[t] >= jerk_z and local_zs[t] >= local_speed_z:
            if score[t] >= score[t - 1] and score[t] >= score[t + 1]:
                candidates.append(t)
                last = t
    return candidates


# -----------------------------
# Gripper forward-edge approximation
# -----------------------------


def estimate_gripper_edge(
    frames: Sequence[Image.Image],
    gripper_xy: np.ndarray,  # [T,2]
    vel: np.ndarray,  # [T,2]
    t: int,
    window: int = 2,
    roi: int = 48,
) -> Tuple[float, float]:
    """Estimate forward-facing edge point near frame t.

    Try local optical-flow affinity if OpenCV is available; otherwise project a
    fixed offset along velocity.
    """
    v = vel[t]
    if np.linalg.norm(v) < 1e-3:
        return float(gripper_xy[t, 0]), float(gripper_xy[t, 1])

    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover
        # Fallback: project along velocity by small offset
        unit = v / (np.linalg.norm(v) + 1e-8)
        edge = gripper_xy[t] + unit * (roi * 0.4)
        return float(edge[0]), float(edge[1])

    # use flow between t-1 and t (or earlier if needed)
    t0 = max(0, t - max(1, window))
    img0 = np.array(frames[t0].convert("L"), dtype=np.uint8)
    img1 = np.array(frames[t].convert("L"), dtype=np.uint8)
    H, W = img0.shape
    cx, cy = gripper_xy[t].astype(np.float32)
    x0 = int(max(0, cx - roi))
    y0 = int(max(0, cy - roi))
    x1 = int(min(W, cx + roi))
    y1 = int(min(H, cy + roi))
    patch0 = img0[y0:y1, x0:x1]
    patch1 = img1[y0:y1, x0:x1]

    flow = cv2.calcOpticalFlowFarneback(
        patch0, patch1, None, pyr_scale=0.5, levels=3, winsize=21, iterations=5, poly_n=7, poly_sigma=1.2, flags=0
    )
    # gripper motion vector
    gv = vel[t]
    if np.linalg.norm(gv) < 1e-6:
        gv = np.array([1.0, 0.0], dtype=np.float32)
    unit = gv / (np.linalg.norm(gv) + 1e-8)

    # pixels whose flow aligns with gripper displacement
    fx = flow[..., 0]
    fy = flow[..., 1]
    dot = fx * unit[0] + fy * unit[1]
    mag = np.sqrt(fx * fx + fy * fy) + 1e-6
    cos = dot / mag
    align = (cos > 0.7) & (mag > 0.3)  # aligned and non-trivial speed

    if not np.any(align):
        edge = gripper_xy[t] + unit * (roi * 0.4)
        return float(edge[0]), float(edge[1])

    # among aligned pixels, pick the farthest along +unit direction
    ys, xs = np.nonzero(align)
    xs_world = xs + x0
    ys_world = ys + y0
    rel = np.stack([xs_world - cx, ys_world - cy], axis=-1)
    proj = rel @ unit  # projection length
    idx = int(np.argmax(proj))
    edge = np.array([xs_world[idx], ys_world[idx]], dtype=np.float32)
    return float(edge[0]), float(edge[1])


# -----------------------------
# Motion clustering and mask export
# -----------------------------


def simple_motion_clusters(
    tracks_xy: np.ndarray,  # [T,N,2]
    vis: np.ndarray,  # [T,N]
    start_t: int,
    min_disp: float = 5.0,
    angle_thr_deg: float = 30.0,
    mag_rel_thr: float = 0.5,
    spatial_radius: float = 64.0,
) -> Dict[int, List[int]]:
    """Greedy clustering of points by similar displacement vector and proximity.

    Returns a mapping cluster_id -> list of point indices. Cluster 0 is reserved
    for near-static background (|disp| < min_disp).
    """
    T, N, _ = tracks_xy.shape
    end_t = T - 1
    disp = tracks_xy[end_t] - tracks_xy[start_t]
    vis_ok = vis[start_t] & vis[end_t]
    mags = np.linalg.norm(disp, axis=-1)
    clusters: Dict[int, List[int]] = {0: []}
    assigned = np.full(N, -1, dtype=int)

    static_idx = np.where((mags < min_disp) | (~vis_ok))[0]
    clusters[0].extend(static_idx.tolist())
    assigned[static_idx] = 0

    # candidates sorted by magnitude
    order = np.argsort(-mags)
    cid = 1
    for i in order:
        if assigned[i] >= 0:
            continue
        seed = i
        seed_vec = disp[seed]
        seed_mag = mags[seed]
        if seed_mag < min_disp:
            continue
        group = [seed]
        assigned[seed] = cid
        # add neighbors with similar direction/magnitude and spatially close at start
        unit_seed = seed_vec / (seed_mag + 1e-8)
        cos_thr = math.cos(math.radians(angle_thr_deg))
        for j in order:
            if assigned[j] >= 0:
                continue
            v = disp[j]
            m = mags[j]
            if m < min_disp:
                continue
            unit = v / (m + 1e-8)
            if unit_seed @ unit < cos_thr:
                continue
            if not (mag_rel_thr <= (m / (seed_mag + 1e-8)) <= (1.0 / max(mag_rel_thr, 1e-3))):
                continue
            # spatial proximity (at start time)
            d = np.linalg.norm(tracks_xy[start_t, j] - tracks_xy[start_t, seed])
            if d > spatial_radius:
                continue
            assigned[j] = cid
            group.append(j)
        clusters[cid] = group
        cid += 1
    return clusters


def draw_cluster_masks(
    frames: Sequence[Image.Image],
    clusters: Dict[int, List[int]],
    tracks_xy: np.ndarray,  # [T,N,2]
    out_dir: Path,
    radius_px: int = 6,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    draw_overlays: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    T, N, _ = tracks_xy.shape
    H, W = frames[0].size[1], frames[0].size[0]

    # Pre-choose colors
    if color_map is None:
        rng = np.random.RandomState(42)
        color_map = {cid: tuple(int(x) for x in rng.randint(60, 220, size=3)) for cid in clusters.keys()}

    for t in range(T):
        # one mask per cluster
        for cid, idxs in clusters.items():
            mask = Image.new("L", frames[0].size, 0)
            draw = ImageDraw.Draw(mask)
            for j in idxs:
                x, y = tracks_xy[t, j]
                r = radius_px
                draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
            mask_path = out_dir / f"mask_{cid:02d}_{t:05d}.png"
            mask.save(mask_path)

        if draw_overlays:
            rgb = frames[t].copy()
            overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for cid, idxs in clusters.items():
                color = color_map[cid]
                for j in idxs:
                    x, y = tracks_xy[t, j]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(*color, 220))
            rgb = rgb.convert("RGBA")
            rgb.alpha_composite(overlay)
            rgb = rgb.convert("RGB")
            rgb.save(out_dir / f"overlay_{t:05d}.png")


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline contact detection + motion segmentation around a gripper track")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-dir", help="Directory with frames")
    p.add_argument("--pattern", default="*.png", help="Glob pattern for frames-dir")
    src.add_argument("--parquet", action="append", default=[], help="Parquet file(s)/dir/glob with frames")
    p.add_argument("--parquet-image-column", default="image")
    p.add_argument("--parquet-start", type=int, default=0)
    p.add_argument("--parquet-stop", type=int, default=None)

    p.add_argument("--max-frames", type=int, default=None)

    p.add_argument("--gripper-json", required=True, help="Trajectory JSON from track_gripper_trajectory.py")

    p.add_argument("--cotracker-checkpoint", default=None, help="Path to CoTracker checkpoint .pth")
    p.add_argument("--grid-size", type=int, default=20, help="Grid size for CoTracker/LK (N x N points)")

    p.add_argument("--local-radius", type=float, default=32.0, help="Radius around gripper for contact detection")
    p.add_argument("--min-gap", type=int, default=8, help="Min frames between contact events")
    p.add_argument("--jerk-z", type=float, default=2.5)
    p.add_argument("--local-speed-z", type=float, default=2.0)

    p.add_argument("--output-dir", required=True, help="Directory to store outputs")
    p.add_argument("--write-overlays", action="store_true", help="Write cluster overlays per frame")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # frames
    frames = collect_frames(args)
    T = len(frames)
    if T < 3:
        raise ValueError("Need at least 3 frames for tracking and contact detection")

    # gripper track
    g_points = load_gripper_track(Path(args.gripper_json))
    if len(g_points) < T:
        # pad/restrict
        T_effective = min(T, len(g_points))
        frames = frames[:T_effective]
        g_points = g_points[:T_effective]
        T = T_effective

    g_xy = np.stack([[p.x, p.y] for p in g_points], axis=0)
    speed, jerk, vel = derive_velocity_and_jerk(g_points)

    # tracks
    tracks_xy, vis = run_cotracker(frames, args.grid_size, args.cotracker_checkpoint)
    if tracks_xy.shape[0] != T:
        T_eff = min(T, tracks_xy.shape[0])
        frames = frames[:T_eff]
        g_points = g_points[:T_eff]
        g_xy = g_xy[:T_eff]
        speed = speed[:T_eff]
        jerk = jerk[:T_eff]
        vel = vel[:T_eff]
        vis = vis[:T_eff]
        tracks_xy = tracks_xy[:T_eff]
        T = T_eff

    # detect contacts
    contacts = detect_contact_frames(
        g_xy, speed, jerk, tracks_xy, vis, local_radius=args.local_radius, min_gap=args.min_gap, jerk_z=args.jerk_z, local_speed_z=args.local_speed_z
    )

    # estimate gripper forward edge at each contact
    edges: List[Tuple[int, float, float]] = []
    for t in contacts:
        ex, ey = estimate_gripper_edge(frames, g_xy, vel, t)
        edges.append((t, ex, ey))

    # cluster motions after first contact (or from 0 if none)
    start_t = contacts[0] if contacts else 0
    clusters = simple_motion_clusters(tracks_xy, vis, start_t=start_t)

    # draw masks and overlays
    masks_dir = out_dir / "masks"
    draw_cluster_masks(frames, clusters, tracks_xy, masks_dir, draw_overlays=args.write_overlays)

    # save a summary JSON
    summary = {
        "frames": T,
        "grid_points": int(tracks_xy.shape[1]),
        "contacts": [{"frame": int(t), "edge": [float(x), float(y)]} for (t, x, y) in edges],
        "clusters": {int(cid): [int(i) for i in idxs] for cid, idxs in clusters.items()},
        "params": {
            "grid_size": args.grid_size,
            "local_radius": args.local_radius,
            "jerk_z": args.jerk_z,
            "local_speed_z": args.local_speed_z,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # optional: quick contact visualization
    if contacts:
        ov_dir = out_dir / "contact_overlays"
        ov_dir.mkdir(exist_ok=True)
        for (t, ex, ey) in edges:
            img = frames[t].copy()
            draw = ImageDraw.Draw(img)
            gx, gy = g_xy[t]
            draw.ellipse((gx - 6, gy - 6, gx + 6, gy + 6), outline=(255, 50, 50), width=3)
            draw.ellipse((ex - 5, ey - 5, ex + 5, ey + 5), outline=(50, 220, 255), width=3)
            img.save(ov_dir / f"contact_{t:05d}.png")

    print(f"[INFO] Done. Summary -> {out_dir / 'summary.json'}; masks -> {masks_dir}")


if __name__ == "__main__":
    main()

