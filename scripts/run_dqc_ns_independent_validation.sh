#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Publication-scale independent validation for the published DQC-paper NS n=25
# baseline. Keep the segmentation realization fixed across optimization seeds so
# that the only replicated nuisance here is learner optimization stochasticity.
# Seed 100001 was used for the pilot; by default this launcher runs the two
# remaining seeds needed for a three-seed validation.

OPT_SEEDS="${OPT_SEEDS:-200002 300003}"
SEGMENTATION_SEED="${SEGMENTATION_SEED:-101}"
OFFLINE_STEPS="${OFFLINE_STEPS:-250000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/dqc-ns-independent-250k}"
LOG_ROOT="${LOG_ROOT:-logs/dqc-ns-independent-250k}"

PYTHON=".dqc-venv/bin/python"
RUNNER="scripts/run_dqc_ns_segbench.py"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON; run scripts/setup_dqc_baseline.py first." >&2
  exit 1
fi

run_job() {
  local mode="$1"
  local seed="$2"
  local suffix="seed${seed}"
  local seg_args=()

  if [[ "$mode" != "original" ]]; then
    suffix="${suffix}-seg${SEGMENTATION_SEED}"
    seg_args=(--segmentation-seed "$SEGMENTATION_SEED")
  fi

  local out="${OUTPUT_ROOT}/${mode}/${suffix}"
  local log_dir="${LOG_ROOT}/${mode}/${suffix}"
  local metrics="${out}/metrics.jsonl"
  local checkpoint="${out}/params_${OFFLINE_STEPS}.pkl"
  mkdir -p "$log_dir"

  if [[ -f "$checkpoint" ]] && grep -q '"event": "rollout_eval"' "$metrics" 2>/dev/null; then
    echo "Skipping completed ${mode} ${suffix}"
    return
  fi

  echo "Running ${mode} ${suffix}"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    /usr/bin/time -v -o "${log_dir}/time.txt" \
    "$PYTHON" "$RUNNER" \
      --mode "$mode" \
      --seed "$seed" \
      "${seg_args[@]}" \
      --offline-steps "$OFFLINE_STEPS" \
      --log-interval "$LOG_INTERVAL" \
      --eval-episodes "$EVAL_EPISODES" \
      --output-dir "$out" \
      2>&1 | tee "${log_dir}/train.log"
}

for seed in $OPT_SEEDS; do
  # Fresh unsegmented control for each optimization seed.
  run_job original "$seed"
  # Exact same segmentation realization and optimization seed for CVT/naive.
  run_job robust "$seed"
  run_job naive "$seed"
done
