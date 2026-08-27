# SegBench-GC

**Testing segmentation invariance in multi-step offline goal-conditioned reinforcement learning.**

SegBench-GC asks whether a trajectory-aware offline GCRL learner changes when the same logged process is partitioned at arbitrary nonterminal backup boundaries. The benchmark holds transitions, source trajectories, goal sampling, optimization settings, and evaluation fixed while varying only artificial backup boundaries and their continuation semantics.

## Protocol

SegBench-GC separates **source-trajectory boundaries** from **artificial backup boundaries**. Source trajectories remain fixed for future-goal sampling. Artificial cuts affect only multi-step target construction.

- **Original:** no artificial backup cuts.
- **CVT:** an artificial cut stops reward accumulation but retains the continuation bootstrap from the stored successor.
- **Naive:** the same artificial cut is treated as absorbing and its continuation bootstrap is zeroed.

Source-boundary semantics are identical in all three conditions. CVT applies standard nonterminal continuation semantics; it is not proposed as a new Bellman rule.

## Main results

### PointMaze Medium, 100k updates

The primary study uses three optimization seeds, three independently seeded segmentations, exactly 35,000 artificial cuts per segmentation, and 50 evaluation episodes per task.

| Condition | Five-task success |
| --- | ---: |
| Original | 50.5% ± 15.4% |
| CVT | 39.1% ± 5.1% |
| Naive | 19.1% ± 0.2% |

The paired CVT-minus-naive difference is **+20.0 ± 5.4 percentage points**. Across the three segmentation realizations, naive mean success is **20.5%, 31.9%, and 4.8%**, while CVT mean success is **39.6%, 42.3%, and 35.5%**. The corresponding segmentation dispersions are **13.6 points** for naive handling and **3.4 points** for CVT.

CVT is not perfectly invariant at this budget: it retains **77.4%** of the uncut mean.

### Published n-step validation

We apply the same boundary intervention to the published **NS (n=25)** baseline from the Decoupled Q-Chunking codebase on Puzzle-4x5. With three optimization seeds, one fixed 3.5% artificial-cut realization, and 50 evaluation episodes per task:

| Condition | Five-task success |
| --- | ---: |
| Original | 47.2% ± 14.3% |
| CVT | 58.5% ± 19.5% |
| Naive | 0.27% ± 0.23% |

The paired CVT-minus-naive difference is **+58.3 ± 19.8 percentage points**. This validation uses a separate published n-step learner and tests whether the failure extends beyond the primary horizon-indexed implementation.

## Mechanism diagnostics

At an artificial cut after `k` valid steps,

```text
y_naive - y_CVT = - gamma^k V(s_{t+k}, g)
```

A direct target diagnostic verifies this identity to a maximum absolute residual of approximately `3.8e-6` on cut-affected targets. With the negative goal reward used by the primary learner, 99.8% of affected targets shift upward when continuation is dropped.

The learned-critic diagnostic, aggregated across all nine optimization/segmentation pairs and distances 0 through 8 from a cut, gives:

| Contrast | Value shift | Q shift |
| --- | ---: | ---: |
| CVT - Original | -0.07 | +0.17 |
| Naive - Original | +15.47 | +16.02 |
| Naive - CVT | +15.54 | +15.84 |

The one-step GCIQL negative control does not consume artificial backup-boundary metadata, and paired replay samples remain identical under resegmentation.

## Repository layout

```text
configs/      Experiment, environment, and learner configs
src/          SegBench-GC implementation
scripts/      Training, evaluation, diagnostics, and paper utilities
tests/        Unit and protocol tests
paper/        ICLR/arXiv manuscript sources and aggregate tables
supplement/   Anonymous-submission reproduction guide
```

Large run directories, checkpoints, external repositories, and local virtual environments are intentionally excluded from source control.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,rl,benchmarks,analysis]"
python -m pytest -q
```

## Reproduce the primary protocol

The matched-count launcher defaults to a low-cost 10k gate. Run the reported 100k matrix with:

```bash
TRAIN_STEPS=100000 \
SEGMENTATION_COUNT=35000 \
OPT_SEEDS="0 1 2" \
SEGMENTATION_SEEDS="101 102 103" \
bash scripts/run_segmentation_matched_count_gate.sh
```

The launcher preserves source-boundary continuation in both segmented conditions. CVT and naive use identical artificial cut sets and differ only in whether artificial cuts retain continuation value. Checkpoints can be reevaluated with `scripts/evaluate_checkpoint.py` using the desired rollout budget.

## Diagnostics

Direct target identity:

```bash
python scripts/analyze_target_identity.py \
  --robust-run runs/segmentation-matched-count-100000-gate/robust/opt0-seg101 \
  --samples 32768 \
  --output runs/target-identity-100k/opt0-seg101.json
```

Boundary-local critic analysis:

```bash
bash scripts/run_boundary_local_bias_corrected_100k.sh
```

The on-disk condition label `robust` is retained for compatibility with the experiment artifacts; it corresponds to CVT in the paper.

## Published NS baseline

Set up the pinned Decoupled Q-Chunking source and environment:

```bash
python scripts/setup_dqc_baseline.py
```

Run the three-seed validation and final reevaluation:

```bash
bash scripts/run_dqc_ns_independent_validation.sh
bash scripts/run_dqc_ns_eval50.sh
```

The adapter uses policy chunk size 1, backup horizon 25, and no chunk critic. The external DQC repository is pinned by the setup script and stored under `.external/` rather than vendored here.

## Paper and submission artifacts

Compile and check the anonymous ICLR manuscript:

```bash
bash scripts/compile_paper.sh
python scripts/check_iclr_submission.py
```

Build the author-identified arXiv manuscript and source package:

```bash
bash scripts/compile_paper.sh arxiv.tex
python scripts/build_arxiv_source.py
```

Build the deterministic anonymous supplement:

```bash
python scripts/build_anonymous_supplement.py
```

## Scope

SegBench-GC does not claim a new Bellman update or universal failure across all offline RL algorithms. It provides a controlled diagnostic for whether multi-step offline GCRL is sensitive to benign resegmentation, and the reported experiments show that artificial-terminal handling can materially change learned values and goal-reaching behavior even when the underlying data and goal-sampling semantics are unchanged.

## Citation and license

Citation metadata is provided in `CITATION.cff`. The code is released under the MIT License; see `LICENSE`.
