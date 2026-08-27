"""Verify the CVT-vs-naive target identity on a trained primary checkpoint.

The diagnostic reconstructs the exact artificial cut set from a CVT run, then
samples paired targets whose source transitions, goals, accumulated rewards,
effective horizons, and continuation states are identical. Source-trajectory
boundary semantics are held fixed. The only CVT-vs-naive difference is whether
an artificial cut retains its continuation discount.

For every affected target slot we verify numerically

    y_naive - y_CVT = -gamma**k * V(s_{t+k}, g).

The script also reports how many sampled target slots the historical global
``bootstrap_at_backup_boundaries=false`` control additionally changed at source
trajectory boundaries. That bookkeeping is useful for auditing the final
artificial-only control; it is not part of the identity itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_gcrl.algorithms.factory import create_agent
from adaptive_gcrl.data.official_goal_sampling import (
    OfficialGoalReplayBuffer,
    OfficialGoalSamplingConfig,
)
from adaptive_gcrl.data.segmentation import (
    boundary_continuation_mask,
    fixed_count_segmentation_boundaries,
    periodic_segmentation_boundaries,
    random_segmentation_boundaries,
)
from adaptive_gcrl.training.checkpoints import load_agent_checkpoint
from train import load_training_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robust-run", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_train_start(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run log: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "train_start":
                return record
    raise ValueError(f"No train_start event in {path}")


def resolve_checkpoint(run_dir: Path) -> Path:
    for candidate in (
        run_dir / "checkpoints" / "latest.pt",
        run_dir / "agent.pt",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {run_dir}")


def reconstruct_backup_boundaries(
    terminals: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    interval = int(config.get("backup_segmentation_interval", 0))
    probability = float(config.get("backup_segmentation_probability", 0.0))
    count = int(config.get("backup_segmentation_count", 0))
    active = sum((interval > 0, probability > 0.0, count > 0))
    if active != 1:
        raise ValueError(
            "Expected exactly one artificial segmentation mode in the CVT run."
        )
    if interval > 0:
        return periodic_segmentation_boundaries(
            terminals,
            interval=interval,
            offset=int(config.get("backup_segmentation_offset", 0)),
        )
    segmentation_seed = int(config.get("backup_segmentation_seed", 0))
    if probability > 0.0:
        return random_segmentation_boundaries(
            terminals,
            cut_probability=probability,
            seed=segmentation_seed,
        )
    return fixed_count_segmentation_boundaries(
        terminals,
        num_cuts=count,
        seed=segmentation_seed,
    )


def make_sampling_config(config: dict[str, Any]) -> OfficialGoalSamplingConfig:
    algorithm = str(config.get("algorithm", ""))
    horizons = (
        tuple(int(value) for value in config.get("horizons", (1, 2, 4, 8)))
        if algorithm in {"adaptive_iql", "torch_adaptive_iql"}
        or algorithm.startswith("fixed_horizon_iql")
        else None
    )
    if horizons is None:
        raise ValueError("Target-identity diagnostic requires multi-step horizons.")
    return OfficialGoalSamplingConfig(
        discount=float(config.get("discount", 0.99)),
        value_p_curgoal=float(config.get("value_p_curgoal", 0.2)),
        value_p_trajgoal=float(config.get("value_p_trajgoal", 0.5)),
        value_p_randomgoal=float(config.get("value_p_randomgoal", 0.3)),
        value_geom_sample=bool(config.get("value_geom_sample", True)),
        actor_p_curgoal=float(config.get("actor_p_curgoal", 0.0)),
        actor_p_trajgoal=float(config.get("actor_p_trajgoal", 0.5)),
        actor_p_randomgoal=float(config.get("actor_p_randomgoal", 0.5)),
        actor_geom_sample=bool(config.get("actor_geom_sample", False)),
        gc_negative=bool(config.get("gc_negative", True)),
        horizons=horizons,
    )


def assert_same_except_discounts(first, second) -> None:
    for field in (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "terminals",
        "goals",
        "actor_goals",
        "masks",
    ):
        left = getattr(first, field)
        right = getattr(second, field)
        if left is None or right is None:
            if left is not right:
                raise AssertionError(f"Paired batches differ in optional field {field}.")
            continue
        np.testing.assert_array_equal(left, right, err_msg=f"Paired batches differ in {field}.")

    if first.horizon_targets is None or second.horizon_targets is None:
        raise AssertionError("Paired batches are missing horizon targets.")
    if first.horizon_targets.horizons != second.horizon_targets.horizons:
        raise AssertionError("Paired horizon sets differ.")
    for field in ("returns", "next_observations", "effective_steps"):
        np.testing.assert_array_equal(
            getattr(first.horizon_targets, field),
            getattr(second.horizon_targets, field),
            err_msg=f"Paired horizon targets differ in {field}.",
        )


def matching_values(agent, batch, *, inference_batch_size: int) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Target-identity diagnostic requires PyTorch.") from exc

    targets = batch.horizon_targets
    if targets is None or batch.goals is None:
        raise ValueError("Expected goal-conditioned horizon targets.")
    horizons = targets.horizons
    next_observations = np.asarray(targets.next_observations)
    goals = np.asarray(batch.goals)
    batch_size, num_horizons, observation_dim = next_observations.shape
    if num_horizons != len(horizons):
        raise ValueError("Horizon target shape does not match horizon tuple.")

    output = np.empty((batch_size, num_horizons), dtype=np.float32)
    for start in range(0, batch_size, inference_batch_size):
        stop = min(start + inference_batch_size, batch_size)
        count = stop - start
        repeated_goals = np.repeat(goals[start:stop, None, :], num_horizons, axis=1)
        state_goal = agent._state_goal(
            next_observations[start:stop].reshape(count * num_horizons, observation_dim),
            repeated_goals.reshape(count * num_horizons, -1),
        )
        with torch.no_grad():
            all_values = agent.value(state_goal).reshape(
                count,
                num_horizons,
                num_horizons,
            )
            selected = torch.diagonal(all_values, dim1=1, dim2=2)
        output[start:stop] = selected.detach().cpu().numpy()
    return output


def summarize_horizon(
    horizon: int,
    affected: np.ndarray,
    values: np.ndarray,
    cvt_discounts: np.ndarray,
    cvt_targets: np.ndarray,
    naive_targets: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, Any]:
    count = int(np.count_nonzero(affected))
    if count == 0:
        return {
            "horizon": int(horizon),
            "affected_slots": 0,
        }
    shifts = naive_targets[affected] - cvt_targets[affected]
    selected_values = values[affected]
    return {
        "horizon": int(horizon),
        "affected_slots": count,
        "continuation_value_mean": float(selected_values.mean()),
        "continuation_value_sample_std": float(selected_values.std(ddof=1)) if count > 1 else 0.0,
        "cvt_discount_mean": float(cvt_discounts[affected].mean()),
        "cvt_target_mean": float(cvt_targets[affected].mean()),
        "naive_target_mean": float(naive_targets[affected].mean()),
        "naive_minus_cvt_mean": float(shifts.mean()),
        "naive_minus_cvt_sample_std": float(shifts.std(ddof=1)) if count > 1 else 0.0,
        "positive_shift_fraction": float(np.mean(shifts > 0.0)),
        "max_abs_identity_residual": float(np.max(np.abs(residuals[affected]))),
    }


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive.")
    if args.inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive.")

    start = read_train_start(args.robust_run)
    config = dict(start["config"])
    if not bool(config.get("source_boundaries_are_continuing", False)):
        raise ValueError(
            "This diagnostic expects the finalized OGBench protocol with continuing source boundaries."
        )
    if not bool(config.get("bootstrap_at_backup_boundaries", True)):
        raise ValueError("--robust-run must be a CVT/robust checkpoint.")

    seed = int(config.get("seed", 0))
    source_batch = load_training_batch(config, seed)
    source_boundaries = np.asarray(source_batch.terminals, dtype=bool)
    backup_boundaries = reconstruct_backup_boundaries(source_boundaries, config)
    artificial_boundaries = backup_boundaries & ~source_boundaries
    if not np.any(artificial_boundaries):
        raise ValueError("CVT run contains no artificial backup boundaries.")

    robust_continuations = boundary_continuation_mask(
        source_boundaries,
        backup_boundaries,
        source_continues=True,
        artificial_continues=True,
    )
    naive_continuations = boundary_continuation_mask(
        source_boundaries,
        backup_boundaries,
        source_continues=True,
        artificial_continues=False,
    )
    sampling_config = make_sampling_config(config)

    robust_replay = OfficialGoalReplayBuffer(
        source_batch,
        sampling_config,
        backup_boundaries=backup_boundaries,
        bootstrap_at_boundaries=True,
        boundary_continuations=robust_continuations,
    )
    naive_replay = OfficialGoalReplayBuffer(
        source_batch,
        sampling_config,
        backup_boundaries=backup_boundaries,
        bootstrap_at_boundaries=True,
        boundary_continuations=naive_continuations,
    )
    legacy_naive_replay = OfficialGoalReplayBuffer(
        source_batch,
        sampling_config,
        backup_boundaries=backup_boundaries,
        bootstrap_at_boundaries=False,
        boundary_continuations=robust_continuations,
    )

    robust_batch = robust_replay.sample(args.samples, np.random.default_rng(args.seed))
    naive_batch = naive_replay.sample(args.samples, np.random.default_rng(args.seed))
    legacy_naive_batch = legacy_naive_replay.sample(
        args.samples,
        np.random.default_rng(args.seed),
    )
    assert_same_except_discounts(robust_batch, naive_batch)
    assert_same_except_discounts(robust_batch, legacy_naive_batch)

    robust_targets = robust_batch.horizon_targets
    naive_targets = naive_batch.horizon_targets
    legacy_targets = legacy_naive_batch.horizon_targets
    assert robust_targets is not None
    assert naive_targets is not None
    assert legacy_targets is not None

    cvt_discounts = np.asarray(robust_targets.discounts, dtype=np.float32)
    naive_discounts = np.asarray(naive_targets.discounts, dtype=np.float32)
    legacy_discounts = np.asarray(legacy_targets.discounts, dtype=np.float32)
    affected = cvt_discounts != naive_discounts
    if not np.any(affected):
        raise AssertionError("Sample contained no artificial-cut target differences.")
    if not np.all(naive_discounts[affected] == 0.0):
        raise AssertionError("Naive artificial-only targets retained a cut bootstrap.")

    agent = create_agent(config, source_batch)
    if not hasattr(agent, "_state_goal") or not hasattr(agent, "value"):
        raise TypeError("Primary checkpoint does not expose horizon value heads.")
    load_agent_checkpoint(agent, resolve_checkpoint(args.robust_run))
    values = matching_values(
        agent,
        robust_batch,
        inference_batch_size=args.inference_batch_size,
    )

    returns = np.asarray(robust_targets.returns, dtype=np.float32)
    cvt_targets = returns + cvt_discounts * values
    corrected_naive_targets = returns + naive_discounts * values
    expected_shift = (naive_discounts - cvt_discounts) * values
    actual_shift = corrected_naive_targets - cvt_targets
    residual = actual_shift - expected_shift

    effective_steps = np.asarray(robust_targets.effective_steps, dtype=np.int32)
    discount = float(sampling_config.discount)
    expected_cvt_discount = np.power(discount, effective_steps, dtype=np.float64)
    if not np.allclose(
        cvt_discounts[affected],
        expected_cvt_discount[affected],
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError("CVT discount does not equal gamma**k at an affected cut.")

    max_residual = float(np.max(np.abs(residual[affected])))
    if max_residual > 1e-5:
        raise AssertionError(
            f"Target identity residual too large: {max_residual:.3e}"
        )

    legacy_extra_source = naive_discounts != legacy_discounts
    if np.any(legacy_extra_source & affected):
        raise AssertionError(
            "Historical source-boundary audit overlaps artificial-only differences."
        )

    per_horizon = []
    for column, horizon in enumerate(robust_targets.horizons):
        per_horizon.append(
            summarize_horizon(
                int(horizon),
                affected[:, column],
                values[:, column],
                cvt_discounts[:, column],
                cvt_targets[:, column],
                corrected_naive_targets[:, column],
                residual[:, column],
            )
        )

    shifts = actual_shift[affected]
    total_slots = int(np.prod(cvt_discounts.shape))
    result = {
        "event": "target_identity",
        "status": "PASS",
        "run": str(args.robust_run),
        "checkpoint": str(resolve_checkpoint(args.robust_run)),
        "samples": int(args.samples),
        "horizons": [int(value) for value in robust_targets.horizons],
        "target_slots": total_slots,
        "artificial_cut_affected_slots": int(np.count_nonzero(affected)),
        "artificial_cut_affected_fraction": float(np.mean(affected)),
        "source_boundary_count": int(np.count_nonzero(source_boundaries)),
        "artificial_boundary_count": int(np.count_nonzero(artificial_boundaries)),
        "historical_global_naive_extra_source_slots": int(
            np.count_nonzero(legacy_extra_source)
        ),
        "historical_global_naive_extra_source_fraction": float(
            np.mean(legacy_extra_source)
        ),
        "continuation_value_mean_at_artificial_cuts": float(values[affected].mean()),
        "naive_minus_cvt_mean": float(shifts.mean()),
        "naive_minus_cvt_sample_std": float(shifts.std(ddof=1))
        if shifts.size > 1
        else 0.0,
        "positive_shift_fraction": float(np.mean(shifts > 0.0)),
        "max_abs_identity_residual": max_residual,
        "per_horizon": per_horizon,
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
