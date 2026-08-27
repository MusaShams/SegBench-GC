#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EVAL_EPISODES="${EVAL_EPISODES:-50}"
ROLLOUT_MAX_STEPS="${ROLLOUT_MAX_STEPS:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
INPUT_ROOT="${INPUT_ROOT:-runs/segmentation-matched-count-100000-naive-artificial-only}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/segmentation-matched-count-100000-naive-artificial-only-eval50}"
LOG_ROOT="${LOG_ROOT:-logs/segmentation-matched-count-100000-naive-artificial-only-eval50}"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python; create the project environment first." >&2
  exit 1
fi

run_eval() {
  local opt_seed="$1"
  local segmentation_seed="$2"
  local suffix="opt${opt_seed}-seg${segmentation_seed}"
  local checkpoint="${INPUT_ROOT}/naive/${suffix}/checkpoints/latest.pt"
  local output="${OUTPUT_ROOT}/naive/${suffix}.json"
  local log_dir="${LOG_ROOT}/naive/${suffix}"

  mkdir -p "$log_dir" "$(dirname "$output")"

  if [[ -s "$output" ]]; then
    echo "Skipping completed eval ${suffix}"
    return
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    return 1
  fi

  echo "Evaluating naive artificial-only ${suffix} at ${EVAL_EPISODES} episodes/task"
  /usr/bin/time -v -o "${log_dir}/time.txt" \
    .venv/bin/python scripts/evaluate_checkpoint.py \
      --config configs/experiment/ogbench_full.yaml \
      --config configs/env/ogbench_official_goals.yaml \
      --config configs/algo/static_mixture_iql_official_matched.yaml \
      --checkpoint "$checkpoint" \
      --set device=cuda \
      --set seed="$opt_seed" \
      --set rollout_episodes="$EVAL_EPISODES" \
      --set rollout_max_steps="$ROLLOUT_MAX_STEPS" \
      --set eval_batch_size="$EVAL_BATCH_SIZE" \
      --output "$output" \
      2>&1 | tee "${log_dir}/eval.log"
}

for opt_seed in 0 1 2; do
  for segmentation_seed in 101 102 103; do
    run_eval "$opt_seed" "$segmentation_seed"
  done
done

.venv/bin/python - "$OUTPUT_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "naive").glob("opt*-seg*.json")):
    match = re.fullmatch(r"opt(\d+)-seg(\d+)", path.stem)
    if match is None:
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollout = payload.get("rollout_eval") or {}
    value = rollout.get("rollout_success_rate")
    if value is None:
        raise SystemExit(f"Missing rollout_success_rate in {path}")
    rows.append((int(match.group(1)), int(match.group(2)), float(value)))

if len(rows) != 9:
    raise SystemExit(f"Expected 9 evaluation records, found {len(rows)}")

print("\n=== Naive artificial-only: 50 episodes/task ===")
for opt_seed, seg_seed, value in rows:
    print(f"opt{opt_seed}-seg{seg_seed}: {value:.4f}")

seed_means = []
seed_stds = []
for opt_seed in sorted({row[0] for row in rows}):
    vals = np.asarray([row[2] for row in rows if row[0] == opt_seed], dtype=float)
    seed_means.append(float(vals.mean()))
    seed_stds.append(float(vals.std(ddof=1)))
    print(
        f"opt{opt_seed}: segmentation_mean={vals.mean():.4f} "
        f"segmentation_sd={vals.std(ddof=1):.4f} values={vals.tolist()}"
    )

seed_means = np.asarray(seed_means, dtype=float)
print(
    "aggregate_after_segmentation_first: "
    f"mean={seed_means.mean():.4f} "
    f"sample_sd={seed_means.std(ddof=1):.4f} "
    f"seed_means={seed_means.tolist()}"
)
PY
