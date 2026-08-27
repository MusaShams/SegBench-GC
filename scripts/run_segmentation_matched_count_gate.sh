#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Matched-count SegBench-GC protocol. Original has no artificial cuts. CVT and
# naive use the same seeded cut sets; source-trajectory boundaries remain
# continuation-valid in both segmented conditions. Only artificial-cut
# continuation differs between CVT and naive.
#
# Defaults are a low-cost 10k gate. Reproduce the reported 100k matrix with:
#   TRAIN_STEPS=100000 SEGMENTATION_COUNT=35000 \
#   OPT_SEEDS="0 1 2" SEGMENTATION_SEEDS="101 102 103" \
#   bash scripts/run_segmentation_matched_count_gate.sh

TRAIN_STEPS="${TRAIN_STEPS:-10000}"
SEGMENTATION_COUNT="${SEGMENTATION_COUNT:-35000}"
OPT_SEEDS="${OPT_SEEDS:-0 1 2}"
SEGMENTATION_SEEDS="${SEGMENTATION_SEEDS:-101 102 103}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-10}"
ROLLOUT_MAX_STEPS="${ROLLOUT_MAX_STEPS:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/segmentation-matched-count-${TRAIN_STEPS}-gate}"
LOG_ROOT="${LOG_ROOT:-logs/segmentation-matched-count-${TRAIN_STEPS}-gate}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python; create the project environment first." >&2
  exit 1
fi

run_job() {
  local condition="$1"
  local opt_seed="$2"
  local segmentation_seed="$3"
  local artificial_continuing="$4"
  local count="$5"

  local suffix="opt${opt_seed}"
  if [[ "$condition" != "original" ]]; then
    suffix="${suffix}-seg${segmentation_seed}"
  fi

  local output_dir="${OUTPUT_ROOT}/${condition}/${suffix}"
  local log_dir="${LOG_ROOT}/${condition}/${suffix}"
  local checkpoint="${output_dir}/checkpoints/latest.pt"
  local metadata="${checkpoint}.metadata.json"
  local remaining_steps="$TRAIN_STEPS"
  local resume_args=()

  mkdir -p "$log_dir"

  if [[ -f "$log_dir/SUCCESS" ]]; then
    echo "Skipping completed ${condition} ${suffix}"
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

  segmentation_args=()
  if [[ "$condition" != "original" ]]; then
    segmentation_args=(
      --set "backup_segmentation_count=$count"
      --set "backup_segmentation_seed=$segmentation_seed"
      --set "bootstrap_at_backup_boundaries=true"
      --set "artificial_boundaries_are_continuing=$artificial_continuing"
    )
  fi

  echo "Running ${condition} ${suffix}: steps=${TRAIN_STEPS}, cuts=${count}"
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
      "${segmentation_args[@]}" \
      "${resume_args[@]}" \
      2>&1 | tee "$log_dir/train.log"; then
    touch "$log_dir/SUCCESS"
  else
    status=${PIPESTATUS[0]}
    touch "$log_dir/FAILED"
    return "$status"
  fi
}

# Original is independent of segmentation seed, so run it once per optimization
# seed.
for opt_seed in $OPT_SEEDS; do
  run_job "original" "$opt_seed" 0 true 0
done

# Keep the historical output directory label "robust" for compatibility; this
# condition is CVT in the paper. CVT and naive use identical cut sets and differ
# only in artificial-cut continuation.
for opt_seed in $OPT_SEEDS; do
  for segmentation_seed in $SEGMENTATION_SEEDS; do
    run_job "robust" "$opt_seed" "$segmentation_seed" true "$SEGMENTATION_COUNT"
    run_job "naive" "$opt_seed" "$segmentation_seed" false "$SEGMENTATION_COUNT"
  done
done
