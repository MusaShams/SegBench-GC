#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conditions=(
  "original:0:0:true"
  "robust-offset24:25:24:true"
  "naive-offset24:25:24:false"
)

for specification in "${conditions[@]}"; do
  IFS=: read -r condition interval offset bootstrap <<<"$specification"
  source_dir="runs/segmentation-cube-double-100k-gate/${condition}/seed_0"
  resume_checkpoint="$source_dir/agent.pt"
  output_dir="runs/segmentation-cube-double-full-seed0-gate/${condition}/seed_0"
  log_dir="logs/segmentation-cube-double-full-seed0-gate/${condition}"

  mkdir -p "$log_dir"
  if [[ -f "$log_dir/SUCCESS" ]]; then
    continue
  fi
  if [[ ! -f "$resume_checkpoint" ]]; then
    echo "Missing 100k checkpoint: $resume_checkpoint" >&2
    exit 1
  fi

  rm -rf "$output_dir"
  if /usr/bin/time -v \
    -o "$log_dir/time.txt" \
    .venv/bin/python scripts/train.py \
      --config configs/experiment/ogbench_full.yaml \
      --config configs/env/ogbench_cube_double_official_goals.yaml \
      --config configs/algo/static_mixture_iql_official_matched.yaml \
      --output-dir "$output_dir" \
      --set device=cuda \
      --set seed=0 \
      --set resume_checkpoint_path="$resume_checkpoint" \
      --set steps=900000 \
      --set log_interval=5000 \
      --set checkpoint_interval=100000 \
      --set rollout_episodes=50 \
      --set rollout_max_steps=1000 \
      --set eval_batch_size=4096 \
      --set backup_segmentation_interval="$interval" \
      --set backup_segmentation_offset="$offset" \
      --set bootstrap_at_backup_boundaries="$bootstrap" \
      2>&1 | tee "$log_dir/train.log"; then
    touch "$log_dir/SUCCESS"
  else
    status=${PIPESTATUS[0]}
    touch "$log_dir/FAILED"
    exit "$status"
  fi
done
