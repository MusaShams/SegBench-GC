#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for seed in 1 2; do
  output_dir="runs/hf-components/consistency-only/seed_${seed}"
  log_dir="logs/hf-components/consistency-only-seed${seed}"
  checkpoint="$output_dir/checkpoints/latest.pt"
  metadata="${checkpoint}.metadata.json"
  remaining_steps=100000
  resume_args=()

  mkdir -p "$log_dir"
  if [[ -f "$log_dir/SUCCESS" ]]; then
    echo "Skipping completed consistency-only seed ${seed}"
    continue
  fi

  rm -f "$log_dir/FAILED"
  if [[ -f "$checkpoint" && -f "$metadata" ]]; then
    completed_steps="$(
      .venv/bin/python -c \
        'import json, sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["step"]))' \
        "$metadata"
    )"
    if (( completed_steps < 0 || completed_steps > 100000 )); then
      echo "Invalid checkpoint step ${completed_steps} in ${metadata}" >&2
      exit 1
    fi
    remaining_steps=$((100000 - completed_steps))
    resume_args=(
      --set "resume_checkpoint_path=$checkpoint"
      --set "steps=$remaining_steps"
    )
  fi

  echo "Starting consistency-only seed ${seed}: $(date -Is)" \
    | tee "$log_dir/status.log"
  if /usr/bin/time -v \
    -o "$log_dir/time.txt" \
    .venv/bin/python scripts/train.py \
      --config configs/experiment/ogbench_full.yaml \
      --config configs/env/ogbench_official_goals.yaml \
      --config configs/algo/consistent_mixture_iql_official_matched.yaml \
      --output-dir "$output_dir" \
      --set device=cuda \
      --set seed="$seed" \
      --set steps="$remaining_steps" \
      --set log_interval=5000 \
      --set checkpoint_interval=10000 \
      --set rollout_episodes=10 \
      --set rollout_max_steps=1000 \
      --set eval_batch_size=4096 \
      "${resume_args[@]}" \
      2>&1 | tee "$log_dir/train.log"; then
    touch "$log_dir/SUCCESS"
  else
    status=${PIPESTATUS[0]}
    touch "$log_dir/FAILED"
    exit "$status"
  fi
done
