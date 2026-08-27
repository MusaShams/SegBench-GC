"""Measure learned value/Q bias as a function of distance to artificial cuts."""

from __future__ import annotations

import argparse
import csv
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
    fixed_count_segmentation_boundaries,
    periodic_segmentation_boundaries,
    random_segmentation_boundaries,
)
from adaptive_gcrl.training.checkpoints import load_agent_checkpoint
from train import load_training_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--robust-run", type=Path, required=True)
    parser.add_argument("--naive-run", type=Path, required=True)
    parser.add_argument("--max-distance", type=int, default=8)
    parser.add_argument("--samples-per-distance", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/tables/segbench_gc_boundary_local_bias.csv"),
    )
    parser.add_argument(
        "--contrast-output",
        type=Path,
        default=Path("paper/tables/segbench_gc_boundary_local_bias_contrasts.csv"),
    )
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
    candidates = (
        run_dir / "checkpoints" / "latest.pt",
        run_dir / "agent.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found under {run_dir}; checked "
        + ", ".join(str(path) for path in candidates)
    )


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
            "Diagnostic robust run must configure exactly one artificial "
            "segmentation mode."
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


def indices_by_distance_to_next_artificial_cut(
    terminals: np.ndarray,
    backup_boundaries: np.ndarray,
    *,
    max_distance: int,
) -> dict[int, np.ndarray]:
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative.")
    terminals = np.asarray(terminals, dtype=bool)
    backup_boundaries = np.asarray(backup_boundaries, dtype=bool)
    if terminals.shape != backup_boundaries.shape:
        raise ValueError("terminals and backup_boundaries must have equal shape.")

    artificial = backup_boundaries & ~terminals
    cut_indices = np.flatnonzero(artificial)
    if cut_indices.size == 0:
        raise ValueError("No artificial cuts are present.")

    source_ends = np.flatnonzero(terminals)
    if source_ends.size == 0 or source_ends[-1] != terminals.size - 1:
        source_ends = np.concatenate([source_ends, [terminals.size - 1]])

    indices = np.arange(terminals.size)
    next_cut_positions = np.searchsorted(cut_indices, indices, side="left")
    has_next_cut = next_cut_positions < cut_indices.size
    next_cuts = np.full(terminals.size, terminals.size, dtype=int)
    next_cuts[has_next_cut] = cut_indices[next_cut_positions[has_next_cut]]
    source_final = source_ends[np.searchsorted(source_ends, indices, side="left")]

    valid = has_next_cut & (next_cuts <= source_final)
    distances = next_cuts - indices
    output: dict[int, np.ndarray] = {}
    for distance in range(max_distance + 1):
        output[distance] = np.flatnonzero(valid & (distances == distance))
    return output


def deterministic_future_goals(
    batch,
    config: dict[str, Any],
    indices: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    sampler = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(
            discount=float(config.get("discount", 0.99)),
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=True,
            gc_negative=bool(config.get("gc_negative", True)),
            horizons=None,
        ),
    )
    rng = np.random.default_rng(seed)
    goal_indices = sampler._sample_goals(
        indices,
        p_curgoal=0.0,
        p_trajgoal=1.0,
        geom_sample=True,
        rng=rng,
    )
    return np.asarray(batch.observations[goal_indices])


def normalized_core_config(config: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "backup_segmentation_interval",
        "backup_segmentation_offset",
        "backup_segmentation_probability",
        "backup_segmentation_count",
        "backup_segmentation_seed",
        "bootstrap_at_backup_boundaries",
        "artificial_boundaries_are_continuing",
        "resume_checkpoint_path",
        "steps",
        "output_dir",
        "checkpoint_path",
        "periodic_checkpoint_path",
    }
    return {
        key: value
        for key, value in config.items()
        if key not in ignored
    }


def validate_run_compatibility(starts: dict[str, dict[str, Any]]) -> None:
    original = starts["original"]
    robust = starts["robust"]
    naive = starts["naive"]

    if normalized_core_config(robust["config"]) != normalized_core_config(naive["config"]):
        raise ValueError("Robust and naive resolved configs differ beyond segmentation semantics.")
    if normalized_core_config(original["config"]) != normalized_core_config(robust["config"]):
        raise ValueError("Original and robust resolved configs differ beyond segmentation semantics.")

    for field in ("source_boundary_sha256", "source_boundary_count"):
        values = {starts[mode].get(field) for mode in starts}
        if len(values) > 1:
            raise ValueError(f"Runs differ in {field}: {values}")

    robust_hash = robust.get("backup_boundary_sha256")
    naive_hash = naive.get("backup_boundary_sha256")
    if robust_hash is not None and naive_hash is not None and robust_hash != naive_hash:
        raise ValueError("Robust and naive runs do not share the same backup-boundary hash.")


def horizon_weights(config: dict[str, Any], num_horizons: int) -> np.ndarray:
    configured = config.get("static_horizon_weights")
    if configured is None:
        return np.full(num_horizons, 1.0 / num_horizons)
    weights = np.asarray(configured, dtype=float)
    if weights.shape != (num_horizons,):
        raise ValueError(
            f"static_horizon_weights has shape {weights.shape}; expected {(num_horizons,)}"
        )
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("static_horizon_weights must have positive sum.")
    return weights / total


def evaluate_mode(
    mode: str,
    run_dir: Path,
    start: dict[str, Any],
    batch,
    sampled: dict[int, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Boundary-local diagnostics require PyTorch.") from exc

    config = dict(start["config"])
    agent = create_agent(config, batch)
    load_agent_checkpoint(agent, resolve_checkpoint(run_dir))

    if not hasattr(agent, "_state_goal") or not hasattr(agent, "value"):
        raise TypeError(
            f"{type(agent).__name__} does not expose the horizon value interface "
            "required by this diagnostic."
        )
    if not hasattr(agent, "_qs"):
        raise TypeError(
            f"{type(agent).__name__} does not expose critic heads required by this diagnostic."
        )

    horizons = tuple(int(value) for value in config.get("horizons", (1,)))
    weights = horizon_weights(config, len(horizons))
    rows: list[dict[str, Any]] = []

    for distance, (indices, goals) in sorted(sampled.items()):
        state_goal = agent._state_goal(batch.observations[indices], goals)
        actions = torch.as_tensor(
            np.asarray(batch.actions[indices]),
            dtype=torch.float32,
            device=agent.device,
        )
        with torch.no_grad():
            values = agent.value(state_goal).detach().cpu().numpy()
            q1, q2 = agent._qs(state_goal, actions)
            q_values = torch.minimum(q1, q2).detach().cpu().numpy()

        if values.ndim == 1:
            values = values[:, None]
        if q_values.ndim == 1:
            q_values = q_values[:, None]
        if values.shape[1] != len(horizons) or q_values.shape[1] != len(horizons):
            raise ValueError("Checkpoint horizon-head shape does not match resolved config.")

        weighted_values = values @ weights
        weighted_q = q_values @ weights
        row: dict[str, Any] = {
            "mode": mode,
            "distance_to_cut": distance,
            "samples": len(indices),
            "weighted_value_mean": float(weighted_values.mean()),
            "weighted_value_std": float(weighted_values.std(ddof=1)) if len(indices) > 1 else 0.0,
            "weighted_q_mean": float(weighted_q.mean()),
            "weighted_q_std": float(weighted_q.std(ddof=1)) if len(indices) > 1 else 0.0,
        }
        for column, horizon in enumerate(horizons):
            row[f"value_h{horizon}_mean"] = float(values[:, column].mean())
            row[f"q_h{horizon}_mean"] = float(q_values[:, column].mean())
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contrast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["mode"], int(row["distance_to_cut"])): row
        for row in rows
    }
    distances = sorted(
        int(row["distance_to_cut"])
        for row in rows
        if row["mode"] == "original"
    )
    output: list[dict[str, Any]] = []
    for distance in distances:
        original = lookup[("original", distance)]
        robust = lookup[("robust", distance)]
        naive = lookup[("naive", distance)]
        output.append(
            {
                "distance_to_cut": distance,
                "samples": original["samples"],
                "robust_minus_original_value": robust["weighted_value_mean"]
                - original["weighted_value_mean"],
                "naive_minus_original_value": naive["weighted_value_mean"]
                - original["weighted_value_mean"],
                "robust_minus_original_q": robust["weighted_q_mean"]
                - original["weighted_q_mean"],
                "naive_minus_original_q": naive["weighted_q_mean"]
                - original["weighted_q_mean"],
                "naive_minus_robust_value": naive["weighted_value_mean"]
                - robust["weighted_value_mean"],
                "naive_minus_robust_q": naive["weighted_q_mean"]
                - robust["weighted_q_mean"],
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.samples_per_distance <= 0:
        raise ValueError("samples_per_distance must be positive.")

    run_dirs = {
        "original": args.original_run,
        "robust": args.robust_run,
        "naive": args.naive_run,
    }
    starts = {
        mode: read_train_start(run_dir)
        for mode, run_dir in run_dirs.items()
    }
    validate_run_compatibility(starts)

    robust_config = dict(starts["robust"]["config"])
    seed = int(robust_config.get("seed", 0))
    batch = load_training_batch(robust_config, seed)
    backup_boundaries = reconstruct_backup_boundaries(
        batch.terminals,
        robust_config,
    )
    candidate_indices = indices_by_distance_to_next_artificial_cut(
        batch.terminals,
        backup_boundaries,
        max_distance=args.max_distance,
    )

    rng = np.random.default_rng(args.seed)
    sampled: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for distance, candidates in sorted(candidate_indices.items()):
        if candidates.size == 0:
            raise ValueError(f"No states found at distance {distance} from a cut.")
        count = min(args.samples_per_distance, candidates.size)
        selected = np.sort(rng.choice(candidates, size=count, replace=False))
        goals = deterministic_future_goals(
            batch,
            robust_config,
            selected,
            seed=args.seed + distance,
        )
        sampled[distance] = (selected, goals)

    rows: list[dict[str, Any]] = []
    for mode in ("original", "robust", "naive"):
        rows.extend(
            evaluate_mode(
                mode,
                run_dirs[mode],
                starts[mode],
                batch,
                sampled,
            )
        )

    contrasts = contrast_rows(rows)
    write_csv(args.output, rows)
    write_csv(args.contrast_output, contrasts)

    print(json.dumps(contrasts, indent=2))
    print(f"Wrote {args.output} and {args.contrast_output}")


if __name__ == "__main__":
    main()
