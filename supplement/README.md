# SegBench-GC anonymous supplement

This archive contains the code, configurations, tests, manuscript source, and aggregate artifacts needed to inspect the SegBench-GC experiments reported in the anonymous submission.

## Environment

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.13.0
.venv/bin/pip install -e '.[dev,rl,benchmarks,analysis]'
```

The reported primary runs used Python 3.10, PyTorch 2.13.0+cu130, an NVIDIA L4, MuJoCo 3.3.7, and OGBench 1.2.1.

## Validation

```bash
.venv/bin/python -m pytest -q
TECTONIC=/path/to/tectonic bash scripts/compile_paper.sh
.venv/bin/python scripts/check_iclr_submission.py
```

Raw checkpoints and full run logs are omitted from the anonymous archive because of size. Aggregate CSV/JSON/LaTeX artifacts are included. Training runners record resolved configuration, optimization seed, boundary counts, continuation assumptions, update step, task-level rollout success, and checkpoint metadata.

## Primary matched-count protocol

The primary study uses PointMaze Medium Stitch, three optimization seeds, three artificial segmentation seeds, and exactly 35,000 artificial cuts per segmentation. Source-trajectory continuation semantics are fixed across conditions. CVT and naive use the same cut set and differ only in whether artificial cuts retain continuation value.

The launcher defaults to a 10k gate. Run the reported 100k protocol with:

```bash
TRAIN_STEPS=100000 \
SEGMENTATION_COUNT=35000 \
OPT_SEEDS="0 1 2" \
SEGMENTATION_SEEDS="101 102 103" \
bash scripts/run_segmentation_matched_count_gate.sh
```

The on-disk label `robust` corresponds to CVT in the paper and is retained for compatibility with artifact paths.

## Mechanism diagnostics

Direct target-identity check:

```bash
.venv/bin/python scripts/analyze_target_identity.py \
  --robust-run runs/segmentation-matched-count-100000-gate/robust/opt0-seg101 \
  --samples 32768 \
  --output runs/target-identity-100k/opt0-seg101.json
```

Boundary-local value/Q diagnostic:

```bash
bash scripts/run_boundary_local_bias_corrected_100k.sh
```

The one-step GCIQL negative control is covered by the paired replay/protocol tests in `tests/`.

## Independent published n-step validation

Install the pinned Decoupled Q-Chunking baseline:

```bash
.venv/bin/python scripts/setup_dqc_baseline.py
```

Run the reported NS (`n=25`) validation and 50-episode-per-task reevaluation:

```bash
bash scripts/run_dqc_ns_independent_validation.sh
bash scripts/run_dqc_ns_eval50.sh
```

The adapter uses policy chunk size one, backup horizon 25, no chunk critic, and a fixed 3.5% artificial-cut realization. The external DQC source is downloaded at a pinned revision and stored outside the repository tree.

## Archive contents

The anonymous package includes an `ARTIFACT_MANIFEST.json` with per-file SHA-256 hashes. The paper reports the matched-count primary study, its mechanism diagnostics and one-step negative control, and the independent published NS validation. Additional development utilities are retained for reproducibility provenance but are not part of the reported empirical claim.
