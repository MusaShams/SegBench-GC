#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=".dqc-venv/bin/python"
EVALUATOR="scripts/evaluate_dqc_ns_checkpoint.py"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_SEED="${EVAL_SEED:-4242}"
SEGMENTATION_SEED="${SEGMENTATION_SEED:-101}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/dqc-ns-independent-250k-eval50}"
LOG_ROOT="${LOG_ROOT:-logs/dqc-ns-independent-250k-eval50}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON; run scripts/setup_dqc_baseline.py first." >&2
  exit 1
fi

run_eval() {
  local condition="$1"
  local seed="$2"
  local checkpoint="$3"
  local suffix="seed${seed}"
  local seg_args=()
  if [[ "$condition" != "original" ]]; then
    suffix="${suffix}-seg${SEGMENTATION_SEED}"
    seg_args=(--segmentation-seed "$SEGMENTATION_SEED")
  fi

  local output="${OUTPUT_ROOT}/${condition}/${suffix}.json"
  local log_dir="${LOG_ROOT}/${condition}/${suffix}"
  mkdir -p "$log_dir"

  if [[ -s "$output" ]]; then
    echo "Skipping completed ${condition} ${suffix}"
    return
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    return 1
  fi

  echo "Evaluating ${condition} ${suffix}"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    /usr/bin/time -v -o "${log_dir}/time.txt" \
    "$PYTHON" "$EVALUATOR" \
      --checkpoint "$checkpoint" \
      --seed "$seed" \
      --condition "$condition" \
      "${seg_args[@]}" \
      --eval-episodes "$EVAL_EPISODES" \
      --eval-seed "$EVAL_SEED" \
      --output "$output" \
      2>&1 | tee "${log_dir}/eval.log"
}

# Pilot seed checkpoints.
run_eval original 100001 \
  runs/dqc-ns-original-250k/seed100001/params_250000.pkl
run_eval robust 100001 \
  runs/dqc-ns-250k/robust/seed100001-seg101/params_250000.pkl
run_eval naive 100001 \
  runs/dqc-ns-250k/naive/seed100001-seg101/params_250000.pkl

# Two preregistered replication optimization seeds.
for seed in 200002 300003; do
  run_eval original "$seed" \
    "runs/dqc-ns-independent-250k/original/seed${seed}/params_250000.pkl"
  run_eval robust "$seed" \
    "runs/dqc-ns-independent-250k/robust/seed${seed}-seg101/params_250000.pkl"
  run_eval naive "$seed" \
    "runs/dqc-ns-independent-250k/naive/seed${seed}-seg101/params_250000.pkl"
done

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    rows.append(record)

expected = 9
if len(rows) != expected:
    raise SystemExit(f"Expected {expected} evaluation records, found {len(rows)}")

print("\n=== DQC NS independent validation: final reevaluation ===")
for condition in ("original", "robust", "naive"):
    subset = sorted(
        [row for row in rows if row["condition"] == condition],
        key=lambda row: row["seed"],
    )
    values = np.asarray([row["rollout_success_rate"] for row in subset], dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    seed_text = ", ".join(f"{row['seed']}={row['rollout_success_rate']:.3f}" for row in subset)
    print(f"{condition:8s} mean={mean:.4f} sample_sd={sd:.4f} | {seed_text}")

original = {row["seed"]: row["rollout_success_rate"] for row in rows if row["condition"] == "original"}
robust = {row["seed"]: row["rollout_success_rate"] for row in rows if row["condition"] == "robust"}
naive = {row["seed"]: row["rollout_success_rate"] for row in rows if row["condition"] == "naive"}
seeds = sorted(original)
robust_minus_original = np.asarray([robust[s] - original[s] for s in seeds])
robust_minus_naive = np.asarray([robust[s] - naive[s] for s in seeds])
print(
    "paired robust-original: "
    f"mean={robust_minus_original.mean():.4f}, "
    f"sample_sd={robust_minus_original.std(ddof=1):.4f}, "
    f"per_seed={robust_minus_original.tolist()}"
)
print(
    "paired robust-naive: "
    f"mean={robust_minus_naive.mean():.4f}, "
    f"sample_sd={robust_minus_naive.std(ddof=1):.4f}, "
    f"per_seed={robust_minus_naive.tolist()}"
)
PY
