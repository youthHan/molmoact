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

# Detect --run anywhere after fixed args
DO_RUN=""
for arg in "${@:5}"; do
  if [[ "$arg" == "--run" ]]; then
    DO_RUN="--run"
    break
  fi
done

SEG=$(printf "%04d" "${SEG_RAW}")
# Normalize paths (strip trailing slashes)
GRIPPER_JSON="${GRIPPER_DIR%/}/segment_${SEG}.json"
OUTPUT_ROOT="/mnt/bn/kinetics-lp-maliva/playground_projects/MolmoAct/${OUTPUT_ROOT%/}"

PY=python
SCRIPT="scripts/offline_contact_and_motion.py"

# Common fixed inputs
COMMON=(
  --parquet "${PARQUET_GLOB}" \
  --parquet-image-column image \
  --gripper-json "${GRIPPER_JSON}" \
  --grid-size 40 \
  --cotracker-vis-thr 0.8 \
  --cotracker-checkpoint ../co-tracker/checkpoints/scaled_offline.pth \
  --write-overlays --debug-overlays \
  --cotracker-backward
)

run_cmd() {
  # local tag=$1; shift
  local tag=$1xia_backwards; shift
  local outdir="${OUTPUT_ROOT}/seg${SEG}_${tag}"
  echo "# ${tag}"
  # Build command as an array for correctness
  local cmd=("${PY}" "${SCRIPT}" "${COMMON[@]}" "$@" --output-dir "${outdir}" --tag "${tag}")
  # Pretty-print with shell-escaped tokens
  local line=""
  for tok in "${cmd[@]}"; do
    printf -v esc '%q' "$tok"
    line+="$esc "
  done
  echo "$line"
  if [[ "${DO_RUN}" == "--run" ]]; then
    "${cmd[@]}"
    ffmpeg -framerate 30 -start_number 0 -i ${outdir}/masks/overlay_%05d.png     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p    ${outdir}/overlay.mp4 
    mv  ${outdir}/overlay.mp4 /mnt/bn/kinetics-lp-maliva/playground_projects/MolmoAct/motion_videos/seg${SEG}_${tag}.mp4
  fi
}

echo "[INFO] Using gripper JSON: ${GRIPPER_JSON}"
if [[ ! -f "${GRIPPER_JSON}" ]]; then
  echo "[WARN] Gripper JSON not found: ${GRIPPER_JSON}" 1>&2
fi

# 0) Baseline v0 (safe defaults, no gating)
run_cmd baseline_v0 --baseline-v0 --disable-candidate-gating

# 1) Grid density ablation
run_cmd grid30 --grid-size 30 --baseline-v0 --disable-candidate-gating
# run_cmd grid50 --grid-size 50 --baseline-v0 --disable-candidate-gating

# # 2) Occlusion threshold ablation (visibility)
# run_cmd vis090 --cotracker-vis-thr 0.9 --baseline-v0 --disable-candidate-gating
# run_cmd vis075 --cotracker-vis-thr 0.75 --baseline-v0 --disable-candidate-gating

# # 3) ROI densify ablation (off vs on with varying radius)
# run_cmd roi18 --roi-densify --roi-radius 18 --baseline-v0 --disable-candidate-gating
# run_cmd roi24 --roi-densify --roi-radius 24 --baseline-v0 --disable-candidate-gating
# run_cmd roi30 --roi-densify --roi-radius 30 --baseline-v0 --disable-candidate-gating

# # 4) Contact thresholds (soft vs strict; include level z for steady pulls)
# run_cmd contact_soft --local-radius 48 --jerk-z 2.0 --local-speed-z 1.7 --local-speed-level-z 2.0 --disable-candidate-gating
# run_cmd contact_strict --local-radius 40 --jerk-z 2.8 --local-speed-z 2.4 --disable-candidate-gating

# # 5) Clustering window ablation
# run_cmd clust_w12 --cluster-window 12 --angle-thr-deg 20 --mag-rel-thr 0.8 --spatial-radius 36 --disable-candidate-gating
# run_cmd clust_w20 --cluster-window 20 --angle-thr-deg 20 --mag-rel-thr 0.8 --spatial-radius 36 --disable-candidate-gating

# # 6) Candidate gating (mild) – pre/post + exclusion
# run_cmd gate_mild \
#   --cluster-window 15 --angle-thr-deg 20 --mag-rel-thr 0.8 --spatial-radius 36 \
#   --exclude-near-gripper-radius 20 --exclude-near-gripper-percent 0.7 \
#   --pre-frames 12 --post-frames 12 --dpre-min 24 --dpost-max 24 --onset-lag 3 \
#   --speed-ratio-min 0.6 --speed-ratio-max 1.6 --dir-cos-min 0.7

# # 7) Candidate gating (stronger) + DINO gating
# run_cmd gate_strong_dino \
#   --cluster-window 15 --angle-thr-deg 20 --mag-rel-thr 0.8 --spatial-radius 36 \
#   --exclude-near-gripper-radius 20 --exclude-near-gripper-percent 0.75 \
#   --pre-frames 12 --post-frames 12 --dpre-min 28 --dpost-max 20 --onset-lag 2 \
#   --speed-ratio-min 0.7 --speed-ratio-max 1.3 --dir-cos-min 0.9 \
#   --dino-gating --dino-sim-gripper-thr 0.90

echo "[INFO] Commands prepared. Append --run to execute."


# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_baseline_v0/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_baseline_v0/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_clust_w12/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_clust_w12/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_clust_w20/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_clust_w20/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_contact_soft/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_contact_soft/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_contact_strict/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_contact_strict/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_gate_mild/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_gate_mild/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_gate_strong_dino/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_gate_strong_dino/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_vis090/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_vis090/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_roi30/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_roi30/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_roi24/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_roi24/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_roi18/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_roi18/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_grid30/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_grid30/overlay.mp4 
# ffmpeg -framerate 30 -start_number 0 -i 'dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_grid50/masks/overlay_%05d.png'     -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p      dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_grid50/overlay.mp4 

# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_baseline_v0/overlay.mp4 motion_videos/seg0012_baseline_v0.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_roi18/overlay.mp4 motion_videos/seg0012_roi18.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_roi24/overlay.mp4 motion_videos/seg0012_roi24.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_roi30/overlay.mp4 motion_videos/seg0012_roi30.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_vis090/overlay.mp4 motion_videos/seg0012_vis090.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_gate_strong_dino/overlay.mp4 motion_videos/seg0012_gate_strong_dino.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_gate_mild/overlay.mp4 motion_videos/seg0012_gate_mild.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_contact_strict/overlay.mp4 motion_videos/seg0012_contact_strict.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_contact_soft/overlay.mp4 motion_videos/seg0012_contact_soft.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_clust_w20/overlay.mp4 motion_videos/seg0012_clust_w20.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_clust_w12/overlay.mp4 motion_videos/seg0012_clust_w12.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_grid30/overlay.mp4 motion_videos/seg0012_grid30.mp4
# mv  dino_tracker_in_seg_vis/cortacker_libero_goal_viz/seg0012_grid50/overlay.mp4 motion_videos/seg0012_grid50.mp4