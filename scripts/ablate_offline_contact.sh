#!/usr/bin/env bash
set -euo pipefail

# Ablation runner for offline_contact_and_motion.py
#
# Usage:
#   bash MolmoAct/scripts/ablate_offline_contact.sh \
#     <SEG_ID> \
#     "/mnt/.../train-*-of-00013.parquet" \
#     dino_tracker_in_seg_vis/goal_mixp_new_trc_ema0.3_hist_shard4 \
#     dino_tracker_in_seg_vis/cotracker_libero_goal_viz \
#     [--run]
#
# Notes:
# - SEG_ID can be '12' or '0012'. The script normalizes to 4 digits.
# - Add --run to actually execute; otherwise commands are printed.

SEG_RAW=${1:?"SEG_ID required (e.g., 12)"}
PARQUET_GLOB=${2:?"Parquet glob required"}
GRIPPER_DIR=${3:?"Directory containing segment_XXXX.json"}
OUTPUT_ROOT=${4:?"Output root directory"}
DO_RUN=${5:-}

SEG=$(printf "%04d" "${SEG_RAW}")
GRIPPER_JSON="${GRIPPER_DIR}/segment_${SEG}.json"

PY=python
SCRIPT="MolmoAct/scripts/offline_contact_and_motion.py"

# Common fixed inputs
COMMON=(
  --parquet "${PARQUET_GLOB}" \
  --parquet-image-column image \
  --gripper-json "${GRIPPER_JSON}" \
  --grid-size 40 \
  --cotracker-backward \
  --cotracker-vis-thr 0.8 \
  --write-overlays --debug-overlays
)

run_cmd() {
  local tag=$1; shift
  local outdir="${OUTPUT_ROOT}/seg${SEG}_${tag}"
  echo "# ${tag}"
  echo "${PY} ${SCRIPT} \"${COMMON[@]}\" $* --output-dir \"${outdir}\" --tag ${tag}"
  if [[ "${DO_RUN}" == "--run" ]]; then
    ${PY} ${SCRIPT} "${COMMON[@]}" "$@" --output-dir "${outdir}" --tag "${tag}"
  fi
}

echo "[INFO] Using gripper JSON: ${GRIPPER_JSON}"

# 0) Baseline v0 (safe defaults, no gating)
run_cmd baseline_v0 --baseline-v0 --disable-candidate-gating

# 1) Grid density ablation
run_cmd grid30 --grid-size 30 --baseline-v0 --disable-candidate-gating
run_cmd grid50 --grid-size 50 --baseline-v0 --disable-candidate-gating

# 2) Occlusion threshold ablation (visibility)
run_cmd vis090 --cotracker-vis-thr 0.9 --baseline-v0 --disable-candidate-gating
run_cmd vis075 --cotracker-vis-thr 0.75 --baseline-v0 --disable-candidate-gating

# 3) ROI densify ablation (off vs on with varying radius)
run_cmd roi18 --roi-densify --roi-radius 18 --baseline-v0 --disable-candidate-gating
run_cmd roi24 --roi-densify --roi-radius 24 --baseline-v0 --disable-candidate-gating
run_cmd roi30 --roi-densify --roi-radius 30 --baseline-v0 --disable-candidate-gating

# 4) Contact thresholds (soft vs strict; include level z for steady pulls)
run_cmd contact_soft --local-radius 48 --jerk-z 2.0 --local-speed-z 1.7 --local-speed-level-z 2.0 --disable-candidate-gating
run_cmd contact_strict --local-radius 40 --jerk-z 2.8 --local-speed-z 2.4 --disable-candidate-gating

# 5) Clustering window ablation
run_cmd clust_w12 --cluster-window 12 --angle-thr-deg 20 --mag-rel-thr 0.8 --spatial-radius 36 --disable-candidate-gating
run_cmd clust_w20 --cluster-window 20 --angle-thr-deg 20 --mag-rel-thr 0.8 --spatial-radius 36 --disable-candidate-gating

# 6) Candidate gating (mild) – pre/post + exclusion
run_cmd gate_mild \
  --cluster-window 15 --angle-thr-deg 20 --mag-rel-thr 0.8 --spatial-radius 36 \
  --exclude-near-gripper-radius 20 --exclude-near-gripper-percent 0.7 \
  --pre-frames 12 --post-frames 12 --dpre-min 24 --dpost-max 24 --onset-lag 3 \
  --speed-ratio-min 0.6 --speed-ratio-max 1.6 --dir-cos-min 0.7

# 7) Candidate gating (stronger) + DINO gating
run_cmd gate_strong_dino \
  --cluster-window 15 --angle-thr-deg 20 --mag-rel-thr 0.8 --spatial-radius 36 \
  --exclude-near-gripper-radius 20 --exclude-near-gripper-percent 0.75 \
  --pre-frames 12 --post-frames 12 --dpre-min 28 --dpost-max 20 --onset-lag 2 \
  --speed-ratio-min 0.7 --speed-ratio-max 1.3 --dir-cos-min 0.9 \
  --dino-gating --dino-sim-gripper-thr 0.90

echo "[INFO] Commands prepared. Append --run to execute."

