#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-paper/tables/boundary_bias_100k_corrected}"
MAX_DISTANCE="${MAX_DISTANCE:-8}"
SAMPLES_PER_DISTANCE="${SAMPLES_PER_DISTANCE:-2048}"
DIAGNOSTIC_SEED="${DIAGNOSTIC_SEED:-2027}"

mkdir -p "$OUTPUT_ROOT"

for opt_seed in 0 1 2; do
  for segmentation_seed in 101 102 103; do
    suffix="opt${opt_seed}-seg${segmentation_seed}"
    echo "=== corrected boundary bias ${suffix} ==="

    .venv/bin/python scripts/analyze_boundary_local_bias.py \
      --original-run "runs/segmentation-matched-count-100000-gate/original/opt${opt_seed}" \
      --robust-run "runs/segmentation-matched-count-100000-gate/robust/${suffix}" \
      --naive-run "runs/segmentation-matched-count-100000-naive-artificial-only/naive/${suffix}" \
      --max-distance "$MAX_DISTANCE" \
      --samples-per-distance "$SAMPLES_PER_DISTANCE" \
      --seed "$DIAGNOSTIC_SEED" \
      --output "$OUTPUT_ROOT/${suffix}.csv" \
      --contrast-output "$OUTPUT_ROOT/${suffix}-contrasts.csv" \
      > "$OUTPUT_ROOT/${suffix}.log"
  done
done

.venv/bin/python - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
columns = [
    "robust_minus_original_value",
    "naive_minus_original_value",
    "robust_minus_original_q",
    "naive_minus_original_q",
    "naive_minus_robust_value",
    "naive_minus_robust_q",
]

pairs = []
for path in sorted(root.glob("opt*-seg*-contrasts.csv")):
    match = re.fullmatch(r"opt(\d+)-seg(\d+)-contrasts", path.stem)
    if match is None:
        continue
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    pair = {
        "opt_seed": int(match.group(1)),
        "segmentation_seed": int(match.group(2)),
        "distances": len(rows),
    }
    for column in columns:
        pair[column] = float(np.mean([float(row[column]) for row in rows]))
    pairs.append(pair)

if len(pairs) != 9:
    raise SystemExit(f"Expected 9 pair summaries, found {len(pairs)}")

summary = {"pairs": pairs, "aggregate": {}}
print("\n=== Corrected boundary-local bias: pair means over distances 0..8 ===")
for pair in pairs:
    print(
        f"opt{pair['opt_seed']}-seg{pair['segmentation_seed']}: "
        f"naive-robust V={pair['naive_minus_robust_value']:+.4f}, "
        f"Q={pair['naive_minus_robust_q']:+.4f}"
    )

print("\n=== Aggregate across 9 opt x segmentation pairs ===")
for column in columns:
    values = np.asarray([pair[column] for pair in pairs], dtype=float)
    stats = {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    summary["aggregate"][column] = stats
    print(
        f"{column}: mean={stats['mean']:+.4f} "
        f"sample_sd={stats['sample_sd']:.4f} "
        f"range=[{stats['min']:+.4f}, {stats['max']:+.4f}]"
    )

output = root / "summary.json"
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote {output}")
PY
