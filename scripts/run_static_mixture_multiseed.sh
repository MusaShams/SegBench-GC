#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seeds=("$@")
if [[ ${#seeds[@]} -eq 0 ]]; then
  seeds=(1 2 3 4)
fi

for seed in "${seeds[@]}"; do
  output_dir="runs/full/static-mixture/seed_${seed}"
  log_dir="logs/full-static-mixture-seed${seed}"
  checkpoint="$output_dir/checkpoints/latest.pt"
  metadata="${checkpoint}.metadata.json"
  remaining_steps=1000000
  resume_args=()

  mkdir -p "$log_dir"
  if [[ -f "$log_dir/SUCCESS" ]]; then
    echo "Skipping completed static-mixture seed ${seed}"
    continue
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
      exit 1
    fi
    remaining_steps=$((1000000 - completed_steps))
    resume_args=(
      --set "resume_checkpoint_path=$checkpoint"
      --set "steps=$remaining_steps"
    )
    echo "Resuming static-mixture seed ${seed} from step ${completed_steps}"
  fi

  echo "Starting static-mixture seed ${seed}: $(date -Is)" \
    | tee "$log_dir/status.log"
  if /usr/bin/time -v \
    -o "$log_dir/time.txt" \
    .venv/bin/python scripts/train.py \
      --config configs/experiment/ogbench_full.yaml \
      --config configs/env/ogbench_official_goals.yaml \
      --config configs/algo/static_mixture_iql_official_matched.yaml \
      --output-dir "$output_dir" \
      --set device=cuda \
      --set seed="$seed" \
      "${resume_args[@]}" \
      2>&1 | tee "$log_dir/train.log"; then
    echo "Completed static-mixture seed ${seed}: $(date -Is)" \
      | tee -a "$log_dir/status.log"
    touch "$log_dir/SUCCESS"
  else
    status=${PIPESTATUS[0]}
    echo "Failed static-mixture seed ${seed} with status ${status}: $(date -Is)" \
      | tee -a "$log_dir/status.log"
    touch "$log_dir/FAILED"
    exit "$status"
  fi
done
