# Offline Contact + Motion Segmentation

This document explains the full pipeline, all arguments, and the logic behind `offline_contact_and_motion.py`, with code references where each concept is implemented.

## Overview

- Inputs
  - Frames: either a directory of images or sharded parquet with an image column.
  - Gripper trajectory JSON (from `track_gripper_trajectory.py`).
- Core steps
  - Frame ingestion and optional parquet-by-global-index fetching.
  - CoTracker (or LK) dense/sparse tracking; optional ROI densification near the gripper path.
  - Contact detection using gripper jerk and local scene motion near the gripper.
  - Forward-edge estimate for the gripper near contact (for debugging and future gating).
  - Candidate selection: pre vs. post distance, onset timing, velocity alignment; gripper exclusion.
  - Optional DINO appearance gating (remove gripper-looking points).
  - Short-window motion clustering and mask export.
  - Debug overlays for contact signals, ROI, and gating decisions.

## Entry Points (CLI)

- Script: `MolmoAct/scripts/offline_contact_and_motion.py:836`
- Arguments parsing: `MolmoAct/scripts/offline_contact_and_motion.py:836`

Key source inputs
- `--frames-dir` (images) or `--parquet` + `--parquet-image-column` (parquet): `MolmoAct/scripts/offline_contact_and_motion.py:839`, `MolmoAct/scripts/offline_contact_and_motion.py:841`
- `--gripper-json`: `MolmoAct/scripts/offline_contact_and_motion.py:848`

Tracking & occlusion handling
- `--cotracker-checkpoint`: `MolmoAct/scripts/offline_contact_and_motion.py:850`
- `--grid-size` (default 40): `MolmoAct/scripts/offline_contact_and_motion.py:851`
- `--cotracker-vis-thr` (visibility threshold): `MolmoAct/scripts/offline_contact_and_motion.py:852`
- `--cotracker-backward` (bidirectional tracking): `MolmoAct/scripts/offline_contact_and_motion.py:853`
- `--roi-densify`, `--roi-radius`: `MolmoAct/scripts/offline_contact_and_motion.py:854`, `MolmoAct/scripts/offline_contact_and_motion.py:855`

Contact detection
- `--local-radius`: `MolmoAct/scripts/offline_contact_and_motion.py:857`
- `--min-gap`: `MolmoAct/scripts/offline_contact_and_motion.py:858`
- `--jerk-z`: `MolmoAct/scripts/offline_contact_and_motion.py:859`
- `--local-speed-z`: `MolmoAct/scripts/offline_contact_and_motion.py:860`
- `--local-speed-level-z` (optional, for steady pulls): `MolmoAct/scripts/offline_contact_and_motion.py:861`

Clustering and gating
- Short window: `--cluster-window`: `MolmoAct/scripts/offline_contact_and_motion.py:866`
- Thresholds: `--min-disp`, `--angle-thr-deg`, `--mag-rel-thr`, `--spatial-radius`: `MolmoAct/scripts/offline_contact_and_motion.py:867`, `MolmoAct/scripts/offline_contact_and_motion.py:868`, `MolmoAct/scripts/offline_contact_and_motion.py:869`, `MolmoAct/scripts/offline_contact_and_motion.py:870`
- Gripper exclusion & pre/post distance: `--exclude-near-gripper-*`, `--pre-frames`, `--post-frames`, `--dpre-min`, `--dpost-max`, `--onset-lag`, `--speed-ratio-*`, `--dir-cos-min`: `MolmoAct/scripts/offline_contact_and_motion.py:872`–`MolmoAct/scripts/offline_contact_and_motion.py:880`
- DINO gating: `--dino-gating`, `--dino-model-id`, `--dino-sim-gripper-thr`, `--dino-pre-frames`, `--dino-post-frames`: `MolmoAct/scripts/offline_contact_and_motion.py:882`–`MolmoAct/scripts/offline_contact_and_motion.py:886`

Outputs & debug
- Overlays: `--write-overlays`, `--debug-overlays`: `MolmoAct/scripts/offline_contact_and_motion.py:864`, `MolmoAct/scripts/offline_contact_and_motion.py:865`
- Output dir: `--output-dir`: `MolmoAct/scripts/offline_contact_and_motion.py:863`

## Frame Ingestion

- Expand parquet tokens/globs: `MolmoAct/scripts/offline_contact_and_motion.py:63`
- Parquet sequential loader (start/stop): `MolmoAct/scripts/offline_contact_and_motion.py:120`
- Global-index aware loader across shards: `MolmoAct/scripts/offline_contact_and_motion.py:215`
- Gripper trajectory JSON
  - Load and honor `smoothed_x/smoothed_y`, `global_frame_idx` (or `total_frame_idx`): `MolmoAct/scripts/offline_contact_and_motion.py:301`–`MolmoAct/scripts/offline_contact_and_motion.py:317`
- When `--parquet` is given and JSON has global indices, frames are fetched exactly by those indices (see `main`, ROI mask build and call into tracker): `MolmoAct/scripts/offline_contact_and_motion.py:912`–`MolmoAct/scripts/offline_contact_and_motion.py:945`

## Tracking (CoTracker or LK)

- CoTracker predictor with visibility threshold: `co-tracker/cotracker/predictor.py:14`, `co-tracker/cotracker/predictor.py:35`, `co-tracker/cotracker/predictor.py:172`
- Wrapper API: `run_cotracker`: `MolmoAct/scripts/offline_contact_and_motion.py:371`
  - Accepts `vis_thr`, `backward`, and an optional `segm_mask` for ROI densification.
  - Builds video tensor: `MolmoAct/scripts/offline_contact_and_motion.py:386`
  - Creates mask tensor if provided: `MolmoAct/scripts/offline_contact_and_motion.py:388`–`MolmoAct/scripts/offline_contact_and_motion.py:395`
- ROI densification (optional)
  - Mask built from gripper path: disks of radius `--roi-radius` centered at gripper positions along the track: `MolmoAct/scripts/offline_contact_and_motion.py:912`–`MolmoAct/scripts/offline_contact_and_motion.py:927`
  - Passed to CoTracker via `segm_mask`: `MolmoAct/scripts/offline_contact_and_motion.py:933`–`MolmoAct/scripts/offline_contact_and_motion.py:945`
- LK fallback for environments without checkpoints/GPU: `MolmoAct/scripts/offline_contact_and_motion.py:410`

## Contact Detection (Definition)

Contact is the earliest local maximum in a joint score where:
1) Gripper jerk z-score ≥ `--jerk-z`, and
2) Either the derivative z-score of the local scene motion within `--local-radius` is ≥ `--local-speed-z`, or the level z-score of that local motion ≥ `--local-speed-level-z` (if provided).

Implementation
- Local neighborhood mask (points within radius of gripper): `MolmoAct/scripts/offline_contact_and_motion.py:477`
- Local speeds per point over time: `MolmoAct/scripts/offline_contact_and_motion.py:481`–`MolmoAct/scripts/offline_contact_and_motion.py:485`
- Robust z-scores: `MolmoAct/scripts/offline_contact_and_motion.py:487`–`MolmoAct/scripts/offline_contact_and_motion.py:489`
- Joint peak selection with `min-gap`: `MolmoAct/scripts/offline_contact_and_motion.py:493`–`MolmoAct/scripts/offline_contact_and_motion.py:506`

Why this definition: the jerk spike captures the gripper’s acceleration change at touch; the local scene motion spike (or high level) captures the environment’s response. For steady pulls (e.g., opening a drawer), set `--local-speed-level-z` to a positive value (e.g., 2.0) so constant elevated local motion is sufficient evidence.

## Gripper Forward-Edge Estimate

- Optical-flow affinity inside a square ROI of size `2*roi` centered on the gripper; pick farthest aligned pixel along gripper direction: `MolmoAct/scripts/offline_contact_and_motion.py:539`–`MolmoAct/scripts/offline_contact_and_motion.py:552`, `MolmoAct/scripts/offline_contact_and_motion.py:560`–`MolmoAct/scripts/offline_contact_and_motion.py:571`
- Fallback: project gripper position along velocity: `MolmoAct/scripts/offline_contact_and_motion.py:531`–`MolmoAct/scripts/offline_contact_and_motion.py:537`

## Candidate Selection (Pre/Post + Exclusion)

Goals
- Keep points that become associated with the gripper at contact (object), and drop persistent gripper-body points.

Logic (per point)
- Pre vs. Post distance to gripper:
  - Pre median distance ≥ `--dpre-min`; post median distance ≤ `--dpost-max`: `MolmoAct/scripts/offline_contact_and_motion.py:685`–`MolmoAct/scripts/offline_contact_and_motion.py:688`, `MolmoAct/scripts/offline_contact_and_motion.py:726`
- Onset near contact: post median speed > pre median speed within `--onset-lag` frames: `MolmoAct/scripts/offline_contact_and_motion.py:705`–`MolmoAct/scripts/offline_contact_and_motion.py:707`
- Direction/speed consistency with gripper in the post window: cosine ≥ `--dir-cos-min`, speed ratio in [`--speed-ratio-min`, `--speed-ratio-max`]: `MolmoAct/scripts/offline_contact_and_motion.py:711`–`MolmoAct/scripts/offline_contact_and_motion.py:724`
- Gripper exclusion: near the gripper for ≥ `--exclude-near-gripper-percent` of post frames if within `--exclude-near-gripper-radius`: `MolmoAct/scripts/offline_contact_and_motion.py:690`–`MolmoAct/scripts/offline_contact_and_motion.py:693`

Output: boolean masks `(candidate_mask, gripper_mask)`: `MolmoAct/scripts/offline_contact_and_motion.py:728`

## DINO Appearance Gating (Optional)

- Build a gripper embedding from the nearest DINO patch to the gripper position at `t_ref = contact-1`: `MolmoAct/scripts/offline_contact_and_motion.py:747`–`MolmoAct/scripts/offline_contact_and_motion.py:756`
- Over a small time window around contact, for each point pick the nearest patch and compute max cosine similarity to this gripper embedding; drop points with similarity ≥ `--dino-sim-gripper-thr`: `MolmoAct/scripts/offline_contact_and_motion.py:762`–`MolmoAct/scripts/offline_contact_and_motion.py:783`

## Motion Clustering (Short Window)

- Displacements computed from `start_t = contact` over `--cluster-window` frames: `MolmoAct/scripts/offline_contact_and_motion.py:573`
- Greedy grouping by:
  - Minimum displacement `--min-disp`: `MolmoAct/scripts/offline_contact_and_motion.py:637`–`MolmoAct/scripts/offline_contact_and_motion.py:639`
  - Direction similarity (cosine of displacement vectors, angle threshold `--angle-thr-deg`): `MolmoAct/scripts/offline_contact_and_motion.py:629`–`MolmoAct/scripts/offline_contact_and_motion.py:641`
  - Relative magnitude similarity band `--mag-rel-thr`: `MolmoAct/scripts/offline_contact_and_motion.py:642`–`MolmoAct/scripts/offline_contact_and_motion.py:643`
  - Spatial proximity at `start_t` within `--spatial-radius`: `MolmoAct/scripts/offline_contact_and_motion.py:644`–`MolmoAct/scripts/offline_contact_and_motion.py:647`

Candidate filtering of clusters
- After clustering, keep only candidate points (as defined above) in non-zero clusters; background cluster 0 is the union of static points and non-candidate points: `MolmoAct/scripts/offline_contact_and_motion.py:1067`–`MolmoAct/scripts/offline_contact_and_motion.py:1093`

## Debug Overlays & Outputs

- ROI mask overlay: `debug_overlays/roi_mask_frame0.png`: `MolmoAct/scripts/offline_contact_and_motion.py:957`–`MolmoAct/scripts/offline_contact_and_motion.py:973`
- Contact overlays: local-radius circle, local points, gripper and edge markers, and signal annotations: `debug_overlays/contact_debug_t*.png`: `MolmoAct/scripts/offline_contact_and_motion.py:980`–`MolmoAct/scripts/offline_contact_and_motion.py:1006`
- Gating overlay: red (gripper excluded) and green (object candidates) at `start_t`: `debug_overlays/gating_start_t*.png`: `MolmoAct/scripts/offline_contact_and_motion.py:1095`–`MolmoAct/scripts/offline_contact_and_motion.py:1111`
- Cluster masks and point overlays: `masks/mask_*.png`, `masks/overlay_*.png`: `MolmoAct/scripts/offline_contact_and_motion.py:786`–`MolmoAct/scripts/offline_contact_and_motion.py:828`
- Summary JSON with parameters and contacts: `summary.json`: `MolmoAct/scripts/offline_contact_and_motion.py:1117`–`MolmoAct/scripts/offline_contact_and_motion.py:1145`
- Contact frame thumbnails (gripper + forward-edge): `contact_overlays/contact_*.png`: `MolmoAct/scripts/offline_contact_and_motion.py:1147`–`MolmoAct/scripts/offline_contact_and_motion.py:1157`

## Tuning Tips (256×256)

- Occlusions/small objects
  - `--cotracker-vis-thr 0.8 --cotracker-backward` (keeps marginal points; bridges occlusions).
  - `--roi-densify --roi-radius 16`–`22` for coverage near gripper path.
- Contact accuracy
  - If early/late: adjust `--jerk-z` and `--local-speed-z` ±0.3; for steady pulls add `--local-speed-level-z 2.0`.
- Avoid gripper absorption
  - `--exclude-near-gripper-radius 20 --exclude-near-gripper-percent 0.75`.
  - DINO: `--dino-gating --dino-sim-gripper-thr 0.88`.
- Clustering
  - `--cluster-window 12`–`18`, `--angle-thr-deg 18`–`24`, `--mag-rel-thr 0.8`, `--spatial-radius 36`–`44`.

## Example Commands

- Baseline v0 (safe defaults, no ROI/DINO, forgiving contact; recommended to verify signals):

  ```
  python MolmoAct/scripts/offline_contact_and_motion.py \
    --parquet "/path/to/shards/*.parquet" --parquet-image-column image \
    --gripper-json gripper.json \
    --grid-size 40 --cotracker-checkpoint co-tracker/checkpoints/scaled_offline.pth \
    --cotracker-vis-thr 0.8 --cotracker-backward \
    --local-radius 48 --jerk-z 2.2 --local-speed-z 2.2 --local-speed-level-z 2.0 \
    --cluster-window 15 --angle-thr-deg 20 --mag-rel-thr 0.8 --spatial-radius 36 \
    --disable-candidate-gating --write-overlays --debug-overlays --baseline-v0 \
    --output-dir outputs/baseline_v0
  ```

- With parquet, occlusion handling, ROI densification, debug overlays:

  ```
  python MolmoAct/scripts/offline_contact_and_motion.py \
    --parquet "/path/to/shards/*.parquet" --parquet-image-column image \
    --gripper-json gripper.json \
    --grid-size 40 --cotracker-checkpoint co-tracker/checkpoints/scaled_offline.pth \
    --cotracker-vis-thr 0.8 --cotracker-backward \
    --roi-densify --roi-radius 18 \
    --local-radius 48 --jerk-z 2.2 --local-speed-z 2.2 --local-speed-level-z 2.0 \
    --cluster-window 15 --angle-thr-deg 20 --mag-rel-thr 0.8 --spatial-radius 36 \
    --exclude-near-gripper-radius 20 --exclude-near-gripper-percent 0.75 \
    --pre-frames 12 --post-frames 12 --dpre-min 28 --dpost-max 18 --onset-lag 2 \
    --dino-gating --dino-sim-gripper-thr 0.88 \
    --write-overlays --debug-overlays \
    --output-dir outputs/run_01
  ```

## Presets, Tags, and Fallbacks

- `--baseline-v0`: applies safe defaults (occlusion-friendly CoTracker, no ROI/DINO, permissive contact, `--disable-candidate-gating`) to quickly verify signal presence.
- `--disable-candidate-gating`: bypasses pre/post distance + exclusion + DINO filters; keeps clusters as-is.
- `--fallback-start {auto|max-local-deriv|max-local-level|max-jerk|mid}`: when no contact is found, selects a reasonable start frame for clustering; default `auto` blends local-deriv/level and jerk z-scores.
- `--tag`: an arbitrary string saved to `summary.json` to annotate runs.

## Notes & Limitations

- DINO gating computes embeddings within a small time window; it’s best-effort and optional.
- Optical-flow based forward-edge can be noisy when textureless; it’s for debugging and heuristics.
- Parquet-by-index fetch requires the trajectory JSON to include `global_frame_idx` (or `total_frame_idx` fallback): `MolmoAct/scripts/offline_contact_and_motion.py:311`–`MolmoAct/scripts/offline_contact_and_motion.py:315`.
