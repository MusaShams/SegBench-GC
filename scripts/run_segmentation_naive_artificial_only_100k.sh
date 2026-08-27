#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Naive artificial-terminal condition for the reported 100k matched-count
# PointMaze study. Source-trajectory boundaries remain continuation-valid;
# only the 35k artificial cuts are treated as absorbing.

TRAIN_STEPS="${TRAIN_STEPS:-100000}"
SEGMENTATION_COUNT="${SEGMENTATION_COUNT:-35000}"
OPT_SEEDS="${OPT_SEEDS:-0 1 2}"
SEGMENTATION_SEEDS="${SEGMENTATION_SEEDS:-101 102 103}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-10}"
ROLLOUT_MAX_STEPS="${ROLLOUT_MAX_STEPS:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/segmentation-matched-count-100000-naive-artificial-only}"
LOG_ROOT="${LOG_ROOT:-logs/segmentation-matched-count-100000-naive-artificial-only}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python; create the project environment first." >&2
  exit 1
fi

run_job() {
  local opt_seed="$1"
  local segmentation_seed="$2"
  local suffix="opt${opt_seed}-seg${segmentation_seed}"
  local output_dir="${OUTPUT_ROOT}/naive/${suffix}"
  local log_dir="${LOG_ROOT}/naive/${suffix}"
  local checkpoint="${output_dir}/checkpoints/latest.pt"
  local metadata="${checkpoint}.metadata.json"
  local remaining_steps="$TRAIN_STEPS"
  local resume_args=()

  mkdir -p "$log_dir"

  if [[ -f "$log_dir/SUCCESS" ]]; then
    echo "Skipping completed naive-artificial-only ${suffix}"
    return
  fi

  if [[ -f "$checkpoint" && -f "$metadata" ]]; then
    completed_steps="$(
      .venv/bin/python -c \
        'import json, sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["step"]))' \
        "$metadata"
    )"
    if (( completed_steps < 0 || completed_steps > TRAIN_STEPS )); then
      echo "Invalid checkpoint step ${completed_steps} in ${metadata}" >&2
      return 1
    fi
    remaining_steps=$((TRAIN_STEPS - completed_steps))
    if (( remaining_steps == 0 )); then
      echo "Checkpoint complete but SUCCESS marker absent: ${output_dir}" >&2
      return 1
    fi
    resume_args=(
      --set "resume_checkpoint_path=$checkpoint"
      --set "steps=$remaining_steps"
    )
  fi

  echo "Running naive-artificial-only ${suffix}: steps=${TRAIN_STEPS}, cuts=${SEGMENTATION_COUNT}"
  rm -f "$log_dir/FAILED"

  if /usr/bin/time -v \
    -o "$log_dir/time.txt" \
    .venv/bin/python scripts/train.py \
      --config configs/experiment/ogbench_full.yaml \
      --config configs/env/ogbench_official_goals.yaml \
      --config configs/algo/static_mixture_iql_official_matched.yaml \
      --output-dir "$output_dir" \
      --set device=cuda \
      --set seed="$opt_seed" \
      --set steps="$remaining_steps" \
      --set log_interval=5000 \
      --set checkpoint_interval="$TRAIN_STEPS" \
      --set rollout_episodes="$ROLLOUT_EPISODES" \
      --set rollout_max_steps="$ROLLOUT_MAX_STEPS" \
      --set eval_batch_size="$EVAL_BATCH_SIZE" \
      --set backup_segmentation_count="$SEGMENTATION_COUNT" \
      --set backup_segmentation_seed="$segmentation_seed" \
      --set artificial_boundaries_are_continuing=false \
      --set bootstrap_at_backup_boundaries=true \
      "${resume_args[@]}" \
      2>&1 | tee "$log_dir/train.log"; then
    touch "$log_dir/SUCCESS"
  else
    status=${PIPESTATUS[0]}
    touch "$log_dir/FAILED"
    return "$status"
  fi
}

for opt_seed in $OPT_SEEDS; do
  for segmentation_seed in $SEGMENTATION_SEEDS; do
    run_job "$opt_seed" "$segmentation_seed"
  done
done
