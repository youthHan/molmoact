"""
Action event detection utilities for LIBERO evaluation.

What this provides:
- detect_gripper_events: rising/falling edges (open/close) from the 7th action dim
- detect_turn_segments: knob-like turning segments from rotation vs. translation

Assumptions:
- Actions are 7-D per step: (dx, dy, dz, r1, r2, r3, gripper)
- The last dim is the gripper command in [-1, +1] after any post-processing
  (see run_libero_eval[_vllm].py where gripper is normalized and potentially inverted)
- Rotation dims (r1..r3) represent end-effector orientation deltas (env-dependent units).

CLI usage examples:
  python -m MolmoAct.experiments.LIBERO.action_events \
      --actions npy_path.npy \
      --gripper-thresh 0.2 --min-hold 3 \
      --rot-thresh 0.08 --ratio-thresh 3.0 --window 8

  # Or analyze a JSON list of actions (list[list[float]]):
  python -m MolmoAct.experiments.LIBERO.action_events --actions actions.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GripperEvent:
    t: int
    kind: str  # 'open' or 'close'
    value: float


@dataclass
class TurnSegment:
    t_start: int
    t_end: int
    axis: int        # 0, 1, or 2 for the dominant rotation axis in dims 3..5
    direction: int   # +1 or -1 based on net rotation sign on the dominant axis
    rot_sum: float   # signed sum of rotation on dominant axis
    rot_mag_sum: float
    trans_mag_sum: float


def _to_np(actions: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(actions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"Expected array of shape [T,7+], got {arr.shape}")
    return arr


def detect_gripper_events(
    actions: Sequence[Sequence[float]],
    thresh: float = 0.2,
    min_hold: int = 2,
    assume_inverted: bool = True,
) -> List[GripperEvent]:
    """
    Detect open/close edges on the gripper signal.

    - `assume_inverted=True` matches run_libero_eval[_vllm].py, where the gripper sign is flipped so that
      +1 ≈ close, -1 ≈ open. If your actions are pre-inversion, set to False.
    - `thresh` and `min_hold` debias small jitters; event triggers when the sign crosses and holds ≥ min_hold steps.
    """
    a = _to_np(actions)
    g = a[:, -1].copy()
    if assume_inverted:
        g *= 1.0  # already inverted by caller environment; keep as is

    # Ternary state {-1, 0, +1}
    st = np.zeros_like(g, dtype=int)
    st[g >= +thresh] = +1
    st[g <= -thresh] = -1

    events: List[GripperEvent] = []
    last_state = st[0]
    hold = 1
    for t in range(1, len(st)):
        if st[t] == last_state:
            hold += 1
            continue
        # state changed
        if st[t] != 0 and hold >= min_hold:
            kind = 'close' if st[t] > 0 else 'open'
            events.append(GripperEvent(t=t, kind=kind, value=float(g[t])))
        last_state = st[t]
        hold = 1
    return events


def detect_turn_segments(
    actions: Sequence[Sequence[float]],
    window: int = 8,
    rot_thresh: float = 0.08,
    ratio_thresh: float = 3.0,
    require_closed: bool = False,
    gripper_thresh: float = 0.2,
) -> List[TurnSegment]:
    """
    Find knob-like turning segments where rotation dominates translation.

    Heuristic (tuned conservatively to avoid false positives):
      - Use dims 0:3 as translation deltas, 3:6 as rotation deltas.
      - Over a sliding window, require mean(|rot|) > rot_thresh and mean(|rot|) / (mean(|trans|)+1e-6) > ratio_thresh.
      - Dominant axis is argmax of mean absolute rotation; direction from the window’s signed sum on that axis.
      - Optionally require the gripper to be closed in the window (if the task expects grasp-then-turn).

    Notes:
      - Units depend on your dataset stats. This works best on unnormalized dims 0..5 (as produced by MolmoAct parsers).
      - Adjust thresholds per task suite if needed.
    """
    a = _to_np(actions)
    trans = a[:, 0:3]
    rot = a[:, 3:6]
    grip = a[:, -1]

    T = len(a)
    W = max(int(window), 1)
    out: List[TurnSegment] = []

    t = 0
    while t + W <= T:
        sl = slice(t, t + W)
        rot_mag = np.mean(np.linalg.norm(rot[sl], axis=1))
        trans_mag = np.mean(np.linalg.norm(trans[sl], axis=1))
        ratio = rot_mag / max(trans_mag, 1e-6)
        if rot_mag >= rot_thresh and ratio >= ratio_thresh:
            if require_closed:
                if np.mean(grip[sl] >= gripper_thresh) < 0.8:  # ≥80% steps closed
                    t += 1
                    continue
            # dominant axis & direction
            abs_means = np.mean(np.abs(rot[sl]), axis=0)
            axis = int(np.argmax(abs_means))
            signed_sum = float(np.sum(rot[sl, axis]))
            direction = +1 if signed_sum >= 0 else -1
            seg = TurnSegment(
                t_start=t,
                t_end=t + W - 1,
                axis=axis,
                direction=direction,
                rot_sum=signed_sum,
                rot_mag_sum=float(np.sum(np.linalg.norm(rot[sl], axis=1))),
                trans_mag_sum=float(np.sum(np.linalg.norm(trans[sl], axis=1))),
            )
            out.append(seg)
            # jump past this window to avoid excessive overlap
            t += W
        else:
            t += 1
    return out


def _load_actions(path: str) -> np.ndarray:
    if path.endswith('.npy') or path.endswith('.npz'):
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # pick the first array
            key = list(arr.keys())[0]
            data = arr[key]
        else:
            data = arr
        return _to_np(data)
    # JSON: either list[list[float]] or dict with key 'actions'
    with open(path, 'r') as f:
        obj = json.load(f)
    if isinstance(obj, dict) and 'actions' in obj:
        obj = obj['actions']
    return _to_np(obj)


def main():
    ap = argparse.ArgumentParser(description="Detect gripper open/close and turning segments from 7-D actions")
    ap.add_argument('--actions', required=True, help="Path to .npy/.npz or JSON (list of 7-D actions)")
    ap.add_argument('--gripper-thresh', type=float, default=0.2)
    ap.add_argument('--min-hold', type=int, default=2)
    ap.add_argument('--no-invert', action='store_true', help="Set if your gripper signal is not inverted (+1=open)")
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--rot-thresh', type=float, default=0.08)
    ap.add_argument('--ratio-thresh', type=float, default=3.0)
    ap.add_argument('--require-closed', action='store_true', help="Require closed gripper during turn")
    args = ap.parse_args()

    acts = _load_actions(args.actions)
    ge = detect_gripper_events(
        acts,
        thresh=args.gripper_thresh,
        min_hold=args.min_hold,
        assume_inverted=not args.no_invert,
    )
    turns = detect_turn_segments(
        acts,
        window=args.window,
        rot_thresh=args.rot_thresh,
        ratio_thresh=args.ratio_thresh,
        require_closed=args.require_closed,
        gripper_thresh=args.gripper_thresh,
    )

    print("Gripper events:")
    for ev in ge:
        print(f"  t={ev.t:4d}  {ev.kind:5s}  value={ev.value:+.3f}")

    print("\nTurning segments:")
    for seg in turns:
        axis_name = {0: 'r1', 1: 'r2', 2: 'r3'}.get(seg.axis, str(seg.axis))
        direction = 'ccw(+)' if seg.direction > 0 else 'cw(-)'
        print(
            f"  t=[{seg.t_start:4d},{seg.t_end:4d}]  axis={axis_name}  dir={direction}  "
            f"rot_sum={seg.rot_sum:+.3f}  |rot|={seg.rot_mag_sum:.3f}  |trans|={seg.trans_mag_sum:.3f}"
        )


if __name__ == '__main__':
    main()

