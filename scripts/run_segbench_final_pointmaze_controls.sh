#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_job() {
  local condition="$1"
  local seed="$2"
  local interval="$3"
  local offset="$4"
  local bootstrap="$5"
  local output_dir="runs/segbench-final-pointmaze-controls/${condition}/seed_${seed}"
  local log_dir="logs/segbench-final-pointmaze-controls/${condition}-seed${seed}"
  local checkpoint="$output_dir/checkpoints/latest.pt"
  local metadata="${checkpoint}.metadata.json"
  local remaining_steps=1000000
  local resume_args=()

  mkdir -p "$log_dir"
  if [[ -f "$log_dir/SUCCESS" ]]; then
    echo "Skipping completed ${condition} seed ${seed}"
    return
  fi

  if [[ -f "$checkpoint" && -f "$metadata" ]]; then
    completed_steps="$(
      .venv/bin/python -c \
        'import json, sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["step"]))' \
        "$metadata"
    )"
    if (( completed_steps < 0 || completed_steps > 1000000 )); then
      echo "Invalid checkpoint step ${completed_steps} in ${metadata}" >&2
      return 1
    fi
    remaining_steps=$((1000000 - completed_steps))
    resume_args=(
      --set "resume_checkpoint_path=$checkpoint"
      --set "steps=$remaining_steps"
    )
  fi

  if /usr/bin/time -v \
    -o "$log_dir/time.txt" \
    .venv/bin/python scripts/train.py \
      --config configs/experiment/ogbench_full.yaml \
      --config configs/env/ogbench_official_goals.yaml \
      --config configs/algo/static_mixture_iql_official_matched.yaml \
      --output-dir "$output_dir" \
      --set device=cuda \
      --set seed="$seed" \
      --set steps="$remaining_steps" \
      --set log_interval=5000 \
      --set checkpoint_interval=100000 \
      --set rollout_episodes=50 \
      --set rollout_max_steps=1000 \
      --set eval_batch_size=4096 \
      --set backup_segmentation_interval="$interval" \
      --set backup_segmentation_offset="$offset" \
      --set bootstrap_at_backup_boundaries="$bootstrap" \
      "${resume_args[@]}" \
      2>&1 | tee "$log_dir/train.log"; then
    touch "$log_dir/SUCCESS"
  else
    status=${PIPESTATUS[0]}
    touch "$log_dir/FAILED"
    return "$status"
  fi
}

for seed in 0 1 2 3 4; do
  run_job "original" "$seed" 0 0 true
done

for seed in 3 4; do
  run_job "robust-offset24" "$seed" 25 24 true
  run_job "naive-offset24" "$seed" 25 24 false
done
