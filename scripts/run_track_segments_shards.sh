#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <num_shards> <output_prefix> <common track_segments.py args>" >&2
  echo "Example: $0 4 segment_tracks \
    --segments-json run_01_segments.json --trajectory-json run_01_traj.json \
    --parquet /mnt/data/libero/train-*.parquet --parquet-image-column image" >&2
  exit 1
fi

NUM_SHARDS=$1
OUTPUT_PREFIX=$2
shift 2
COMMON_ARGS=("$@")

if [[ $NUM_SHARDS -le 0 ]]; then
  echo "num_shards must be positive" >&2
  exit 1
fi

pids=()
for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  echo "[INFO] Launching shard $shard/$NUM_SHARDS"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$shard} \
  python3 scripts/track_segments.py \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$shard" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUTPUT_PREFIX}_shard${shard}" \
    --visualize-dir "${OUTPUT_PREFIX}_shard${shard}_viz" &
  pids+=("$!")
  sleep 0.5
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

exit $status
