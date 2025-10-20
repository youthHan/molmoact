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
import glob as _glob
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

os.sys.path.append("/mnt/bn/kinetics-lp-maliva/playground_projects/MolmoAct")
from gripper_tracking import DINOGripperTracker


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
        for match in _glob.glob(expanded):
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


def compute_shard_offsets(parquet_files: List[Path]) -> List[Tuple[Path, int, int]]:
    offsets: List[Tuple[Path, int, int]] = []
    total = 0
    for path in parquet_files:
        pf = pq.ParquetFile(path)
        rows = pf.metadata.num_rows if pf.metadata is not None else 0
        offsets.append((path, total, total + rows - 1))
        total += rows
    return offsets


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
        if value.get("bytes") is not None:
            with Image.open(BytesIO(value["bytes"])) as img:
                return np.array(img.convert("RGB"), dtype=np.uint8)
        if value.get("path"):
            with Image.open(value["path"]) as img:
                return np.array(img.convert("RGB"), dtype=np.uint8)
    raise ValueError(f"Unsupported image container from parquet: {type(value)!r}")


def _iter_parquet_range(
    pf: "pq.ParquetFile",
    column: str,
    start: int,
    end: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    current = 0
    for rg_idx in range(pf.metadata.num_row_groups if pf.metadata is not None else 1):
        table = pf.read_row_group(rg_idx, columns=[column])
        rows = table.to_pylist()
        for row in rows:
            if current > end:
                return
            if current < start:
                current += 1
                continue
            value = row.get(column) if isinstance(row, dict) else row
            yield current, to_rgb_array(value)
            current += 1


def load_frames_from_parquet_by_indices(
    parquet_tokens: Sequence[str],
    image_column: str,
    global_indices: Sequence[int],
) -> List[Image.Image]:
    if pq is None:
        raise RuntimeError("pyarrow is required for --parquet inputs")
    files = expand_parquet_tokens(parquet_tokens)
    if not files:
        raise FileNotFoundError("No parquet files located for --parquet inputs")

    indices = [int(idx) for idx in global_indices]
    if not indices:
        return []

    pos_map: Dict[int, List[int]] = {}
    for pos, idx in enumerate(indices):
        pos_map.setdefault(idx, []).append(pos)

    unique_sorted = sorted(set(indices))
    offsets = compute_shard_offsets(files)
    results: List[Optional[Image.Image]] = [None] * len(indices)

    for path, start, end in offsets:
        needed = [idx for idx in unique_sorted if start <= idx <= end]
        if not needed:
            continue
        local_min = needed[0] - start
        local_max = needed[-1] - start
        pf = pq.ParquetFile(path)
        for local_idx, array in _iter_parquet_range(pf, image_column, local_min, local_max):
            global_idx = start + local_idx
            slots = pos_map.get(global_idx)
            if not slots:
                continue
            img = Image.fromarray(array)
            for pos in slots:
                results[pos] = img.copy()
            img.close()

    missing = [i for i, im in enumerate(results) if im is None]
    if missing:
        raise RuntimeError(
            f"Failed to fetch {len(missing)} frames by global indices. Example missing positions: {missing[:10]}"
        )
    return [img for img in results if img is not None]


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
    global_frame_idx: Optional[int] = None


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
        gfi = row.get("global_frame_idx")
        if gfi is None:
            gfi = row.get("total_frame_idx")
        gfi = int(gfi) if gfi is not None else None
        pts.append(GripperPoint(frame_idx=int(row.get("frame_idx", len(pts))), x=x, y=y, global_frame_idx=gfi))
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


def try_cotracker_predictor(checkpoint: Optional[str], window_len: int = 60, vis_thr: float = 0.9):
    if checkpoint is None:
        return None
    try:
        from cotracker.predictor import CoTrackerPredictor  # type: ignore

        if not Path(checkpoint).exists():
            print(f"[WARN] CoTracker checkpoint not found: {checkpoint}; falling back to LK")
            return None
        model = CoTrackerPredictor(checkpoint=checkpoint, offline=True, window_len=window_len, vis_thr=vis_thr)
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
    vis_thr: float = 0.9,
    segm_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model = try_cotracker_predictor(checkpoint, vis_thr=vis_thr)
    if model is None:
        return run_lk_grid(frames, grid_size)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    video = frames_to_tensor(frames).to(device)
    mask_t = None
    if segm_mask is not None:
        m = segm_mask.astype(np.float32)
        if m.ndim == 2:
            m = m[None, None, ...]
        elif m.ndim == 3:
            m = m[None, ...]
        mask_t = torch.from_numpy(m).to(device)
    tracks, vis = model(
        video,
        queries=None,
        segm_mask=mask_t,
        grid_size=grid_size,
        grid_query_frame=query_frame,
        backward_tracking=backward,
    )
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
    local_speed_level_z: Optional[float] = None,
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
    level_zs = robust_zscore(local_speed_sum)

    score = np.maximum(0.0, jerk_zs) + 0.7 * np.maximum(0.0, local_zs)

    # peaks with thresholds
    candidates: List[int] = []
    last = -min_gap
    for t in range(1, T - 1):
        if t - last < min_gap:
            continue
        local_ok = local_zs[t] >= local_speed_z
        if not local_ok and local_speed_level_z is not None:
            local_ok = level_zs[t] >= local_speed_level_z
        if jerk_zs[t] >= jerk_z and local_ok:
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
    window_frames: Optional[int] = None,
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
    end_t = T - 1 if window_frames is None else min(T - 1, start_t + max(1, int(window_frames)))
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


def prepost_and_exclusion_candidates(
    g_xy: np.ndarray,  # [T,2]
    tracks_xy: np.ndarray,  # [T,N,2]
    vis: np.ndarray,  # [T,N]
    contact_t: int,
    pre_frames: int = 12,
    post_frames: int = 12,
    dpre_min: float = 32.0,
    dpost_max: float = 20.0,
    onset_lag: int = 2,
    speed_ratio_min: float = 0.7,
    speed_ratio_max: float = 1.3,
    dir_cos_min: float = 0.9,
    excl_radius: float = 22.0,
    excl_percent: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (candidate_mask[N], gripper_mask[N]) selecting likely object points.

    - candidate: far from gripper before contact, close after, onset near contact,
      and moving with similar speed and direction to gripper post-contact.
    - gripper: points that stay too close to the gripper for most of post window.
    """
    T, N, _ = tracks_xy.shape
    pre_s = max(0, contact_t - pre_frames)
    pre_e = max(0, contact_t - 1)
    post_s = contact_t
    post_e = min(T - 1, contact_t + post_frames)
    if pre_e <= pre_s or post_e <= post_s:
        return np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)

    # distances to gripper
    d = np.linalg.norm(tracks_xy - g_xy[:, None, :], axis=-1)  # T,N
    d_pre = np.median(d[pre_s:pre_e + 1], axis=0)
    d_post = np.median(d[post_s:post_e + 1], axis=0)

    # exclusion: near gripper most of post window
    near = (d[post_s:post_e + 1] <= excl_radius) & vis[post_s:post_e + 1]
    near_ratio = near.mean(axis=0)
    gripper_mask = near_ratio >= excl_percent

    # onset near contact: speed rise
    vx = np.gradient(tracks_xy[..., 0], axis=0)
    vy = np.gradient(tracks_xy[..., 1], axis=0)
    speed_i = np.linalg.norm(np.stack([vx, vy], axis=-1), axis=-1)  # T,N
    g_vx = np.gradient(g_xy[:, 0])
    g_vy = np.gradient(g_xy[:, 1])
    g_speed = np.hypot(g_vx, g_vy)  # T
    # choose peak onset within lag frames after contact
    onset_ok = np.zeros(N, dtype=bool)
    if onset_lag >= 0:
        s_pre = np.median(speed_i[max(0, contact_t - 3):contact_t], axis=0)
        s_post = np.median(speed_i[contact_t:min(T - 1, contact_t + onset_lag) + 1], axis=0)
        onset_ok = s_post > s_pre

    # speed/direction similarity post-contact
    # compute medians over post window
    gv_post = np.stack([np.median(g_vx[post_s:post_e + 1]), np.median(g_vy[post_s:post_e + 1])])
    gmag = np.linalg.norm(gv_post) + 1e-6
    dir_ok = np.zeros(N, dtype=bool)
    spd_ok = np.zeros(N, dtype=bool)
    if gmag > 1e-5:
        ivx = np.median(vx[post_s:post_e + 1], axis=0)
        ivy = np.median(vy[post_s:post_e + 1], axis=0)
        imags = np.hypot(ivx, ivy) + 1e-6
        # cosine similarity with gripper median velocity
        cos = (ivx * gv_post[0] + ivy * gv_post[1]) / (imags * gmag)
        dir_ok = cos >= dir_cos_min
        # speed ratio band
        ratio = imags / gmag
        spd_ok = (ratio >= speed_ratio_min) & (ratio <= speed_ratio_max)

    candidate = (d_pre >= dpre_min) & (d_post <= dpost_max) & onset_ok & dir_ok & spd_ok & (~gripper_mask)
    candidate = candidate & np.any(vis[post_s:post_e + 1], axis=0)
    return candidate, gripper_mask


def dino_gripper_gating(
    frames: Sequence[Image.Image],
    g_xy: np.ndarray,  # [T,2]
    contact_t: int,
    points_xy: np.ndarray,  # [T,N,2]
    vis: np.ndarray,  # [T,N]
    model_id: str,
    sim_thr: float = 0.88,
    pre_frames: int = 2,
    post_frames: int = 8,
) -> np.ndarray:
    """Return mask[N] of points that are NOT too similar to gripper appearance.

    Computes a gripper patch embedding at a reference frame (contact-1 if possible)
    and compares nearest-patch embeddings for points within a small time window.
    """
    tracker = DINOGripperTracker(model_id=model_id)
    t_ref = max(0, contact_t - 1)
    # Encode reference frame and pick nearest patch to gripper coord
    ref_grid = tracker._encode_image(frames[t_ref])
    gx, gy = g_xy[t_ref]
    dx = ref_grid.xs - gx
    dy = ref_grid.ys - gy
    dist = np.hypot(dx, dy)
    ref_idx = int(np.argmin(dist))
    g_embed = ref_grid.normalized[ref_idx]

    # Time window frames to encode
    t0 = max(0, contact_t - pre_frames)
    t1 = min(len(frames) - 1, contact_t + post_frames)

    # Pre-encode grids for window
    grids = [tracker._encode_image(frames[t]) for t in range(t0, t1 + 1)]

    N = points_xy.shape[1]
    keep = np.ones(N, dtype=bool)
    # For each point, compute max similarity over the window
    for j in range(N):
        max_sim = -1.0
        for ti, t in enumerate(range(t0, t1 + 1)):
            if not vis[t, j]:
                continue
            grid = grids[ti]
            x, y = points_xy[t, j]
            dx = grid.xs - x
            dy = grid.ys - y
            idx = int(np.argmin(np.hypot(dx, dy)))
            sim = float(grid.normalized[idx] @ g_embed)
            if sim > max_sim:
                max_sim = sim
        if max_sim >= sim_thr:
            keep[j] = False
    return keep


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
    p.add_argument("--grid-size", type=int, default=40, help="Grid size for CoTracker/LK (N x N points)")
    p.add_argument("--cotracker-vis-thr", type=float, default=0.9, help="Visibility threshold for CoTracker points")
    p.add_argument("--cotracker-backward", action="store_true", help="Enable backward tracking to bridge occlusions")
    p.add_argument("--roi-densify", action="store_true", help="Restrict initial grid to ROI around gripper path")
    p.add_argument("--roi-radius", type=float, default=18.0, help="ROI densification radius around gripper path (px)")

    p.add_argument("--local-radius", type=float, default=32.0, help="Radius around gripper for contact detection")
    p.add_argument("--min-gap", type=int, default=8, help="Min frames between contact events")
    p.add_argument("--jerk-z", type=float, default=2.5)
    p.add_argument("--local-speed-z", type=float, default=2.0)
    p.add_argument("--local-speed-level-z", type=float, default=None, help="Optional z-threshold on local speed level (not just derivative)")

    p.add_argument("--output-dir", required=True, help="Directory to store outputs")
    p.add_argument("--write-overlays", action="store_true", help="Write cluster overlays per frame")
    p.add_argument("--debug-overlays", action="store_true", help="Write detailed debug overlays for contacts and gating")
    p.add_argument("--disable-candidate-gating", action="store_true", help="Bypass pre/post + exclusion gating and keep clustered points")
    p.add_argument("--baseline-v0", action="store_true", help="Use safe baseline defaults and disable gating")
    p.add_argument("--fallback-start", default="auto", choices=["auto", "max-local-deriv", "max-local-level", "max-jerk", "mid"], help="Start frame when no contact is found")
    p.add_argument("--tag", default=None, help="Optional run tag stored in summary.json")
    
    # Clustering and gating
    p.add_argument("--cluster-window", type=int, default=15, help="Frames after contact for displacement window")
    p.add_argument("--min-disp", type=float, default=5.0)
    p.add_argument("--angle-thr-deg", type=float, default=20.0)
    p.add_argument("--mag-rel-thr", type=float, default=0.8)
    p.add_argument("--spatial-radius", type=float, default=40.0)

    p.add_argument("--exclude-near-gripper-radius", type=float, default=22.0)
    p.add_argument("--exclude-near-gripper-percent", type=float, default=0.75)
    p.add_argument("--pre-frames", type=int, default=12)
    p.add_argument("--post-frames", type=int, default=12)
    p.add_argument("--dpre-min", type=float, default=32.0)
    p.add_argument("--dpost-max", type=float, default=20.0)
    p.add_argument("--onset-lag", type=int, default=2)
    p.add_argument("--speed-ratio-min", type=float, default=0.7)
    p.add_argument("--speed-ratio-max", type=float, default=1.3)
    p.add_argument("--dir-cos-min", type=float, default=0.9)

    p.add_argument("--dino-gating", action="store_true")
    p.add_argument("--dino-model-id", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--dino-sim-gripper-thr", type=float, default=0.88)
    p.add_argument("--dino-pre-frames", type=int, default=2)
    p.add_argument("--dino-post-frames", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply baseline v0 preset if requested (safe defaults)
    if args.baseline_v0:
        args.cotracker_backward = True
        args.cotracker_vis_thr = 0.8 if args.cotracker_vis_thr is None or args.cotracker_vis_thr > 0.8 else args.cotracker_vis_thr
        args.roi_densify = False
        args.dino_gating = False
        args.local_radius = max(args.local_radius, 48.0)
        args.jerk_z = min(args.jerk_z, 2.2)
        args.local_speed_z = min(args.local_speed_z, 2.2)
        args.local_speed_level_z = 2.0 if args.local_speed_level_z is None else args.local_speed_level_z
        args.cluster_window = 15
        args.angle_thr_deg = 20.0
        args.mag_rel_thr = 0.8
        args.spatial_radius = 36.0
        args.disable_candidate_gating = True

    # Load gripper first to support parquet index-based fetch
    g_points = load_gripper_track(Path(args.gripper_json))

    # frames
    if args.parquet:
        indices = [p.global_frame_idx for p in g_points]
        if all(idx is not None for idx in indices):
            frames = load_frames_from_parquet_by_indices(
                args.parquet, image_column=args.parquet_image_column, global_indices=[int(i) for i in indices]  # type: ignore
            )
        else:
            frames = collect_frames(args)
    else:
        frames = collect_frames(args)

    T = len(frames)
    if T < 3:
        raise ValueError("Need at least 3 frames for tracking and contact detection")

    if len(g_points) != T:
        T_effective = min(T, len(g_points))
        frames = frames[:T_effective]
        g_points = g_points[:T_effective]
        T = T_effective

    g_xy = np.stack([[p.x, p.y] for p in g_points], axis=0)
    speed, jerk, vel = derive_velocity_and_jerk(g_points)

    # Optional ROI densification mask built from gripper path
    roi_mask: Optional[np.ndarray] = None
    if args.roi_densify:
        H, W = frames[0].size[1], frames[0].size[0]
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        r = int(max(1, round(args.roi_radius)))
        yy, xx = np.ogrid[:H, :W]
        for (x, y) in g_xy.astype(int):
            # draw a filled disk
            mask = (xx - int(x)) ** 2 + (yy - int(y)) ** 2 <= r * r
            roi_mask[mask] = 1

    # tracks
    tracks_xy, vis = run_cotracker(
        frames,
        args.grid_size,
        args.cotracker_checkpoint,
        backward=args.cotracker_backward,
        vis_thr=args.cotracker_vis_thr,
        segm_mask=roi_mask,
    )
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
        g_xy,
        speed,
        jerk,
        tracks_xy,
        vis,
        local_radius=args.local_radius,
        min_gap=args.min_gap,
        jerk_z=args.jerk_z,
        local_speed_z=args.local_speed_z,
        local_speed_level_z=args.local_speed_level_z,
    )

    # Debug overlays: ROI mask visualization
    if args.debug_overlays and roi_mask is not None:
        dbg_dir = out_dir / "debug_overlays"
        dbg_dir.mkdir(exist_ok=True)
        img = frames[0].copy().convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ov = ImageDraw.Draw(overlay)
        # paint ROI as translucent green
        H, W = roi_mask.shape
        # draw squares for efficiency
        for y in range(0, H, 2):
            x_idxs = np.where(roi_mask[y] > 0)[0]
            for x in x_idxs:
                overlay.putpixel((int(x), int(y)), (0, 255, 100, 90))
        img.alpha_composite(overlay)
        img.convert("RGB").save(dbg_dir / "roi_mask_frame0.png")

    # estimate gripper forward edge at each contact
    edges: List[Tuple[int, float, float]] = []
    for t in contacts:
        ex, ey = estimate_gripper_edge(frames, g_xy, vel, t)
        edges.append((t, ex, ey))

    # Debug overlays for contact signals
    if args.debug_overlays:
        dbg_dir = out_dir / "debug_overlays"
        dbg_dir.mkdir(exist_ok=True)
        # recompute local signal series for annotation
        diffs = tracks_xy - g_xy[:, None, :]
        dists = np.linalg.norm(diffs, axis=-1)
        local_mask_all = (dists <= args.local_radius) & vis
        vx = np.gradient(tracks_xy[..., 0], axis=0)
        vy = np.gradient(tracks_xy[..., 1], axis=0)
        local_speed_all = np.linalg.norm(np.stack([vx, vy], axis=-1), axis=-1)
        local_speed_sum = (local_speed_all * local_mask_all).sum(axis=1)
        jerk_zs = robust_zscore(jerk)
        local_zs = robust_zscore(np.gradient(local_speed_sum))
        level_zs = robust_zscore(local_speed_sum)
        for (t, ex, ey) in edges:
            img = frames[t].copy()
            draw = ImageDraw.Draw(img)
            gx, gy = g_xy[t]
            # draw local radius
            r = args.local_radius
            draw.ellipse((gx - r, gy - r, gx + r, gy + r), outline=(0, 200, 255), width=2)
            # draw points in local mask
            idxs = np.where(local_mask_all[t])[0]
            for j in idxs:
                x, y = tracks_xy[t, j]
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 220, 0))
            # gripper and edge
            draw.ellipse((gx - 5, gy - 5, gx + 5, gy + 5), outline=(255, 50, 50), width=2)
            draw.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), outline=(50, 220, 255), width=2)
            # annotate values
            txt = f"t={t} jerk_z={jerk_zs[t]:.2f} local_dz={local_zs[t]:.2f} level_z={level_zs[t]:.2f}"
            draw.text((10, 10), txt, fill=(255, 255, 255))
            img.save(dbg_dir / f"contact_debug_t{t:05d}.png")

    # cluster motions after first contact (or from fallback if none)
    if contacts:
        start_t = contacts[0]
    else:
        # Fallback start_t selection based on configured strategy
        diffs_fb = tracks_xy - g_xy[:, None, :]
        dists_fb = np.linalg.norm(diffs_fb, axis=-1)
        local_mask_fb = (dists_fb <= args.local_radius) & vis
        vx_fb = np.gradient(tracks_xy[..., 0], axis=0)
        vy_fb = np.gradient(tracks_xy[..., 1], axis=0)
        local_speed_fb = np.linalg.norm(np.stack([vx_fb, vy_fb], axis=-1), axis=-1)
        local_speed_sum_fb = (local_speed_fb * local_mask_fb).sum(axis=1)
        z_local_deriv = robust_zscore(np.gradient(local_speed_sum_fb))
        z_local_level = robust_zscore(local_speed_sum_fb)
        z_jerk = robust_zscore(jerk)
        if args.fallback_start == "max-local-deriv":
            start_t = int(np.argmax(z_local_deriv))
        elif args.fallback_start == "max-local-level":
            start_t = int(np.argmax(z_local_level))
        elif args.fallback_start == "max-jerk":
            start_t = int(np.argmax(z_jerk))
        elif args.fallback_start == "mid":
            start_t = int(len(frames) // 2)
        else:  # auto blend
            score_fb = z_local_deriv + 0.5 * z_local_level + 0.5 * z_jerk
            start_t = int(np.argmax(score_fb))
        # In fallback, default to disabling candidate gating to avoid wiping output
        if not args.disable_candidate_gating:
            args.disable_candidate_gating = True

    # Pre/Post and exclusion candidates (enabled only when we have a contact and not disabled)
    gating_enabled = (len(contacts) > 0) and (not args.disable_candidate_gating)
    if gating_enabled:
        cand_mask, grip_mask = prepost_and_exclusion_candidates(
            g_xy,
            tracks_xy,
            vis,
            contact_t=start_t,
            pre_frames=args.pre_frames,
            post_frames=args.post_frames,
            dpre_min=args.dpre_min,
            dpost_max=args.dpost_max,
            onset_lag=args.onset_lag,
            speed_ratio_min=args.speed_ratio_min,
            speed_ratio_max=args.speed_ratio_max,
            dir_cos_min=args.dir_cos_min,
            excl_radius=args.exclude_near_gripper_radius,
            excl_percent=args.exclude_near_gripper_percent,
        )

        # Optional DINO gating to remove gripper-appearance points
        if args.dino_gating:
            keep_mask = dino_gripper_gating(
                frames,
                g_xy,
                contact_t=start_t,
                points_xy=tracks_xy,
                vis=vis,
                model_id=args.dino_model_id,
                sim_thr=args.dino_sim_gripper_thr,
                pre_frames=args.dino_pre_frames,
                post_frames=args.dino_post_frames,
            )
            cand_mask = cand_mask & keep_mask

    # Build clusters, then filter members by candidate mask
    clusters = simple_motion_clusters(
        tracks_xy,
        vis,
        start_t=start_t,
        window_frames=args.cluster_window,
        min_disp=args.min_disp,
        angle_thr_deg=args.angle_thr_deg,
        mag_rel_thr=args.mag_rel_thr,
        spatial_radius=args.spatial_radius,
    )
    # Apply candidate filtering only if enabled; otherwise keep clusters as-is
    if gating_enabled:
        filtered: Dict[int, List[int]] = {}
        new_cid = 1
        static = []
        for cid, idxs in clusters.items():
            if cid == 0:
                static = idxs
                continue
            kept = [j for j in idxs if cand_mask[j]]
            if kept:
                filtered[new_cid] = kept
                new_cid += 1
        # Background cluster: original static plus non-candidate members
        non_cand = [j for j in range(tracks_xy.shape[1]) if not cand_mask[j]]
        filtered[0] = sorted(set(static + non_cand))
        clusters = filtered

    # Debug overlay: gating result at start_t
    if args.debug_overlays and gating_enabled:
        dbg_dir = out_dir / "debug_overlays"
        img = frames[start_t].copy()
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # red: gripper excluded
        for j in np.where(grip_mask)[0]:
            x, y = tracks_xy[start_t, j]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 60, 60, 220))
        # green: candidates kept
        for j in np.where(cand_mask & (~grip_mask))[0]:
            x, y = tracks_xy[start_t, j]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(60, 220, 100, 220))
        img = img.convert("RGBA")
        img.alpha_composite(overlay)
        img.convert("RGB").save(dbg_dir / f"gating_start_t{start_t:05d}.png")

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
            "cluster_window": args.cluster_window,
            "min_disp": args.min_disp,
            "angle_thr_deg": args.angle_thr_deg,
            "mag_rel_thr": args.mag_rel_thr,
            "spatial_radius": args.spatial_radius,
            "exclude_near_gripper_radius": args.exclude_near_gripper_radius,
            "exclude_near_gripper_percent": args.exclude_near_gripper_percent,
            "dpre_min": args.dpre_min,
            "dpost_max": args.dpost_max,
            "onset_lag": args.onset_lag,
            "speed_ratio_min": args.speed_ratio_min,
            "speed_ratio_max": args.speed_ratio_max,
            "dir_cos_min": args.dir_cos_min,
            "dino_gating": args.dino_gating,
            "dino_sim_gripper_thr": args.dino_sim_gripper_thr if args.dino_gating else None,
            "baseline_v0": args.baseline_v0,
            "disable_candidate_gating": args.disable_candidate_gating,
            "fallback_start": None if contacts else args.fallback_start,
        },
        "tag": args.tag,
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
