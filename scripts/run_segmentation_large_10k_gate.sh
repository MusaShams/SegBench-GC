#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conditions=(
  "original:0:0:true"
  "robust-offset24:25:24:true"
  "naive-offset24:25:24:false"
)

for seed in 0 1 2; do
  for specification in "${conditions[@]}"; do
    IFS=: read -r condition interval offset bootstrap <<<"$specification"
    output_dir="runs/segmentation-large-10k-gate/${condition}/seed_${seed}"
    log_dir="logs/segmentation-large-10k-gate/${condition}-seed${seed}"

    mkdir -p "$log_dir"
    if [[ -f "$log_dir/SUCCESS" ]]; then
      continue
    fi

    rm -rf "$output_dir"
    if /usr/bin/time -v \
      -o "$log_dir/time.txt" \
      .venv/bin/python scripts/train.py \
        --config configs/experiment/ogbench_full.yaml \
        --config configs/env/ogbench_large_official_goals.yaml \
        --config configs/algo/static_mixture_iql_official_matched.yaml \
        --output-dir "$output_dir" \
        --set device=cuda \
        --set seed="$seed" \
        --set steps=10000 \
        --set log_interval=1000 \
        --set checkpoint_interval=0 \
        --set rollout_episodes=2 \
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
done
