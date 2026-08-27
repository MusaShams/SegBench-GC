#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_job() {
  local method="$1"
  local config="$2"
  local output_dir="runs/hf-gciql-go-no-go/${method}/seed_0"
  local log_dir="logs/hf-gciql-go-no-go/${method}"

  mkdir -p "$log_dir"
  if [[ -f "$log_dir/SUCCESS" ]]; then
    echo "Skipping completed ${method}"
    return
  fi

  rm -rf "$output_dir"
  rm -f "$log_dir/FAILED"
  echo "Starting ${method}: $(date -Is)" | tee "$log_dir/status.log"
  if /usr/bin/time -v \
    -o "$log_dir/time.txt" \
    .venv/bin/python scripts/train.py \
      --config configs/experiment/ogbench_full.yaml \
      --config configs/env/ogbench_official_goals.yaml \
      --config "$config" \
      --output-dir "$output_dir" \
      --set device=cuda \
      --set seed=0 \
      --set steps=10000 \
      --set log_interval=1000 \
      --set checkpoint_interval=10000 \
      --set rollout_episodes=5 \
      --set rollout_max_steps=1000 \
      --set eval_batch_size=4096 \
      2>&1 | tee "$log_dir/train.log"; then
    echo "Completed ${method}: $(date -Is)" | tee -a "$log_dir/status.log"
    touch "$log_dir/SUCCESS"
  else
    status=${PIPESTATUS[0]}
    echo "Failed ${method} with status ${status}: $(date -Is)" \
      | tee -a "$log_dir/status.log"
    touch "$log_dir/FAILED"
    return "$status"
  fi
}

run_job \
  "static-mixture" \
  "configs/algo/static_mixture_iql_official_matched.yaml"
run_job \
  "hf-gciql" \
  "configs/algo/hf_gciql_official_matched.yaml"
