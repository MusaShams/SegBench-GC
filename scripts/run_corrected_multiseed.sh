#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seeds=("$@")
if [[ ${#seeds[@]} -eq 0 ]]; then
  seeds=(1 2)
fi

run_job() {
  local method="$1"
  local seed="$2"
  local config="$3"
  local output_dir="runs/full/${method}/seed_${seed}"
  local log_dir="logs/full-${method}-seed${seed}"
  local checkpoint="$output_dir/checkpoints/latest.pt"
  local metadata="${checkpoint}.metadata.json"
  local remaining_steps=1000000
  local resume_args=()

  mkdir -p "$log_dir"
  if [[ -f "$log_dir/SUCCESS" ]]; then
    echo "Skipping completed ${method} seed ${seed}"
    return
  fi

  rm -f "$log_dir/FAILED"
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
    echo "Resuming ${method} seed ${seed} from step ${completed_steps}"
  fi

  echo "Starting ${method} seed ${seed}: $(date -Is)" | tee "$log_dir/status.log"
  if /usr/bin/time -v \
    -o "$log_dir/time.txt" \
    .venv/bin/python scripts/train.py \
      --config configs/experiment/ogbench_full.yaml \
      --config configs/env/ogbench_official_goals.yaml \
      --config "$config" \
      --output-dir "$output_dir" \
      --set device=cuda \
      --set seed="$seed" \
      --set chunk_size=1 \
      --set action_chunk_size=1 \
      "${resume_args[@]}" \
      2>&1 | tee "$log_dir/train.log"; then
    echo "Completed ${method} seed ${seed}: $(date -Is)" | tee -a "$log_dir/status.log"
    touch "$log_dir/SUCCESS"
  else
    status=${PIPESTATUS[0]}
    echo "Failed ${method} seed ${seed} with status ${status}: $(date -Is)" \
      | tee -a "$log_dir/status.log"
    touch "$log_dir/FAILED"
    return "$status"
  fi
}

for seed in "${seeds[@]}"; do
  run_job \
    "fixed-h8" \
    "$seed" \
    "configs/algo/fixed_horizon_iql_h8_official_matched.yaml"
  run_job \
    "adaptive-corrected" \
    "$seed" \
    "configs/algo/adaptive_iql_official_matched.yaml"
done
