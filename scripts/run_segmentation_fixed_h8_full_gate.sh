#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conditions=(
  "robust-offset24:true"
  "naive-offset24:false"
)

for seed in 0 1 2; do
  for specification in "${conditions[@]}"; do
    IFS=: read -r condition bootstrap <<<"$specification"
    source_dir="runs/segmentation-fixed-h8-100k-gate/${condition}/seed_${seed}"
    resume_checkpoint="$source_dir/agent.pt"
    output_dir="runs/segmentation-fixed-h8-full-gate/${condition}/seed_${seed}"
    log_dir="logs/segmentation-fixed-h8-full-gate/${condition}-seed${seed}"

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
        --config configs/env/ogbench_official_goals.yaml \
        --config configs/algo/fixed_horizon_iql_h8_official_matched.yaml \
        --output-dir "$output_dir" \
        --set device=cuda \
        --set seed="$seed" \
        --set resume_checkpoint_path="$resume_checkpoint" \
        --set steps=900000 \
        --set log_interval=5000 \
        --set checkpoint_interval=100000 \
        --set rollout_episodes=50 \
        --set rollout_max_steps=1000 \
        --set eval_batch_size=4096 \
        --set backup_segmentation_interval=25 \
        --set backup_segmentation_offset=24 \
        --set bootstrap_at_backup_boundaries="$bootstrap" \
        2>&1 | tee "$log_dir/train.log"; then
      touch "$log_dir/SUCCESS"
    else
      status=${PIPESTATUS[0]}
      touch "$log_dir/FAILED"
      exit "$status"
    fi
  done
done
