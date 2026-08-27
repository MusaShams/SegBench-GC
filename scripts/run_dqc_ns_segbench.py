"""Run SegBench-GC on the published DQC paper's n-step (NS) baseline.

The DQC paper's NS baseline uses a 25-step return backup with a one-action
policy and no chunk critic. That makes the intervention clean: artificial cuts
shorten only the multi-step backup. CVT keeps the continuation bootstrap at the
stored successor state, while the naive control zeros that continuation mask.
The pinned external DQC source remains unmodified under ``.external/dqc``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


DQC_REVISION = "df898256a77f3594b54a7268bd5f89915981da35"
# Primary matched-count PointMaze used 35k cuts over roughly one million
# eligible nonterminal locations. Preserve that intervention density rather
# than the absolute count on Puzzle-4x5's roughly three-million-state dataset.
DEFAULT_SEGMENTATION_FRACTION = 0.035


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(".external/dqc"))
    parser.add_argument("--env-name", default="puzzle-4x5-play-oraclerep-v0")
    parser.add_argument("--mode", choices=("original", "robust", "naive"), required=True)
    parser.add_argument("--seed", type=int, default=100001)
    parser.add_argument("--segmentation-seed", type=int, default=101)
    parser.add_argument(
        "--segmentation-fraction",
        type=float,
        default=DEFAULT_SEGMENTATION_FRACTION,
        help="Fraction of eligible nonterminal successor states selected as cuts.",
    )
    parser.add_argument(
        "--segmentation-count",
        type=int,
        default=None,
        help="Exact artificial-cut count. Overrides --segmentation-fraction.",
    )
    parser.add_argument("--offline-steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=4242)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-final-checkpoint", action="store_true")
    parser.add_argument(
        "--verify-pairing",
        action="store_true",
        help="Assert robust/naive batches are identical except continuation masks.",
    )
    parser.add_argument("--pairing-check-batch-size", type=int, default=4096)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run the pairing assertion and exit before agent initialization/training.",
    )
    return parser.parse_args()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape).encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def eligible_artificial_cut_states(terminals: np.ndarray, valids: np.ndarray) -> np.ndarray:
    """Return successor-state indices whose predecessor is a valid nonterminal.

    A cut at state ``c`` means a backup beginning before ``c`` may stop at the
    stored continuation state ``s_c``. Requiring transition ``c-1`` to be valid
    and nonterminal prevents cuts from crossing original source boundaries.
    """
    terminals = np.asarray(terminals, dtype=bool)
    valids = np.asarray(valids, dtype=bool)
    if terminals.ndim != 1 or valids.ndim != 1 or terminals.shape != valids.shape:
        raise ValueError("terminals and valids must be same-shaped vectors.")
    if terminals.size <= 1:
        return np.empty(0, dtype=np.int64)
    predecessor_ok = valids[:-1] & ~terminals[:-1]
    return np.flatnonzero(predecessor_ok).astype(np.int64) + 1


def make_artificial_cut_states(
    terminals: np.ndarray,
    valids: np.ndarray,
    *,
    seed: int,
    fraction: float,
    count: int | None,
) -> tuple[np.ndarray, int]:
    candidates = eligible_artificial_cut_states(terminals, valids)
    if count is None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("segmentation_fraction must be in (0, 1].")
        count = int(round(float(fraction) * candidates.size))
    if count < 0:
        raise ValueError("segmentation_count must be non-negative.")
    if count > candidates.size:
        raise ValueError(
            f"Requested {count} cuts but only {candidates.size} eligible locations exist."
        )
    if count == 0:
        return np.empty(0, dtype=np.int64), candidates.size
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=count, replace=False)
    return np.sort(selected.astype(np.int64)), candidates.size


def compute_cut_plan(
    idxs: np.ndarray,
    original_horizons: np.ndarray,
    cut_states: np.ndarray,
    *,
    dataset_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the first artificial cut strictly inside each original backup."""
    idxs = np.asarray(idxs, dtype=np.int64)
    original_horizons = np.asarray(original_horizons, dtype=np.int64)
    cut_states = np.asarray(cut_states, dtype=np.int64)
    if idxs.shape != original_horizons.shape:
        raise ValueError("idxs and original_horizons must have the same shape.")

    positions = np.searchsorted(cut_states, idxs, side="right")
    has_cut = positions < cut_states.size
    next_cut = np.full(idxs.shape, int(dataset_size), dtype=np.int64)
    if np.any(has_cut):
        next_cut[has_cut] = cut_states[positions[has_cut]]
    cut_steps = next_cut - idxs
    should_cut = has_cut & (cut_steps > 0) & (cut_steps < original_horizons)
    new_horizons = original_horizons.copy()
    new_horizons[should_cut] = cut_steps[should_cut]
    return should_cut, next_cut, new_horizons


def apply_segmentation_to_batch(
    batch: dict[str, Any],
    *,
    idxs: np.ndarray,
    cut_states: np.ndarray,
    dataset_size: int,
    get_observations: Callable[[np.ndarray], Any],
    mode: str,
) -> dict[str, Any]:
    if mode not in {"robust", "naive"}:
        raise ValueError("apply_segmentation_to_batch requires robust or naive mode.")
    if cut_states.size == 0:
        return batch

    original_horizons = np.asarray(batch["high_value_backup_horizon"], dtype=np.int64)
    should_cut, next_cut, new_horizons = compute_cut_plan(
        idxs,
        original_horizons,
        cut_states,
        dataset_size=dataset_size,
    )
    if not np.any(should_cut):
        return batch

    # Copy only arrays that are changed; all source observations/actions/goals
    # and sampled trajectory semantics remain those of the published dataset.
    result = dict(batch)
    next_observations = np.array(batch["high_value_next_observations"], copy=True)
    next_observations[should_cut] = np.asarray(get_observations(next_cut[should_cut]))
    result["high_value_next_observations"] = next_observations
    result["high_value_backup_horizon"] = new_horizons

    # A synthetic cut is not goal completion. Because should_cut is strictly
    # before the learner's original goal/source endpoint, its accumulated
    # success reward is exactly zero under this positive-reward formulation.
    rewards = np.asarray(batch["high_value_rewards"], dtype=float).copy()
    rewards[should_cut] = 0.0
    result["high_value_rewards"] = rewards

    masks = np.asarray(batch["high_value_masks"], dtype=float).copy()
    masks[should_cut] = 1.0 if mode == "robust" else 0.0
    result["high_value_masks"] = masks
    return result


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        import jax

        if hasattr(value, "shape"):
            return np.asarray(jax.device_get(value)).tolist()
    except ImportError:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    args = parse_args()
    if args.offline_steps <= 0 and not args.verify_only:
        raise ValueError("offline_steps must be positive.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if args.eval_episodes <= 0 and not args.verify_only:
        raise ValueError("eval_episodes must be positive.")
    if args.pairing_check_batch_size <= 0:
        raise ValueError("pairing_check_batch_size must be positive.")
    if args.verify_only and not args.verify_pairing:
        raise ValueError("--verify-only requires --verify-pairing.")

    source_dir = args.source_dir.resolve()
    marker = source_dir / "agents" / "dqc.py"
    if not marker.is_file():
        raise FileNotFoundError(
            f"Pinned DQC source is missing at {source_dir}. "
            "Run scripts/setup_dqc_baseline.py first."
        )
    sys.path.insert(0, str(source_dir))

    import jax
    import jax.tree_util
    import tqdm
    from agents.dqc import DQCAgent, get_config
    from envs.env_utils import make_ogbench_env_and_datasets
    from utils.datasets import CGCDataset, Dataset
    from utils.evaluation import evaluate
    from utils.flax_utils import save_agent

    config = get_config()
    # Published Puzzle-4x5 NS (n=25) configuration from experiments/reproduce.py.
    config["num_qs"] = 2
    config["policy_chunk_size"] = 1
    config["backup_horizon"] = 25
    config["use_chunk_critic"] = False
    config["distill_method"] = "expectile"
    config["implicit_backup_type"] = "quantile"
    config["q_agg"] = "mean"
    config["kappa_b"] = 0.7
    config["kappa_d"] = 0.5

    env, train_raw, _ = make_ogbench_env_and_datasets(
        args.env_name,
        dataset_path=None,
        compact_dataset=True,
        add_info=True,
    )
    eval_env = make_ogbench_env_and_datasets(
        args.env_name,
        dataset_path=None,
        compact_dataset=True,
        env_only=True,
    )

    eps = 1e-5
    train_raw["actions"] = np.clip(train_raw["actions"], -1 + eps, 1 - eps)
    train_base = Dataset.create(**train_raw)

    if args.mode == "original":
        cut_states = np.empty(0, dtype=np.int64)
        candidate_count = eligible_artificial_cut_states(
            train_base["terminals"], train_base["valids"]
        ).size
    else:
        cut_states, candidate_count = make_artificial_cut_states(
            train_base["terminals"],
            train_base["valids"],
            seed=args.segmentation_seed,
            fraction=args.segmentation_fraction,
            count=args.segmentation_count,
        )

    class SegBenchCGCDataset(CGCDataset):
        def __init__(self, dataset, cfg, *, mode: str, cuts: np.ndarray) -> None:
            self.segbench_mode = mode
            self.segbench_cut_states = np.asarray(cuts, dtype=np.int64)
            super().__init__(dataset, cfg)

        def sample(self, batch_size: int, idxs=None, evaluation=False):
            if idxs is None:
                idxs = self.dataset.get_random_idxs(batch_size)
            idxs = np.asarray(idxs, dtype=np.int64)
            batch = super().sample(batch_size, idxs=idxs, evaluation=evaluation)
            if self.segbench_mode == "original":
                return batch
            return apply_segmentation_to_batch(
                batch,
                idxs=idxs,
                cut_states=self.segbench_cut_states,
                dataset_size=self.size,
                get_observations=self.get_observations,
                mode=self.segbench_mode,
            )

    def make_dataset(mode: str) -> SegBenchCGCDataset:
        return SegBenchCGCDataset(train_base, config, mode=mode, cuts=cut_states)

    if args.verify_pairing:
        robust_dataset = make_dataset("robust")
        naive_dataset = make_dataset("naive")
        valid_idxs = np.asarray(robust_dataset.dataset.valid_idxs, dtype=np.int64)
        check_size = min(args.pairing_check_batch_size, valid_idxs.size)
        rng = np.random.default_rng(args.segmentation_seed + 1000003)
        idxs = rng.choice(valid_idxs, size=check_size, replace=False)

        sampling_seed = args.seed + 7000001
        np.random.seed(sampling_seed)
        robust_batch = robust_dataset.sample(check_size, idxs=idxs)
        np.random.seed(sampling_seed)
        naive_batch = naive_dataset.sample(check_size, idxs=idxs)
        if robust_batch.keys() != naive_batch.keys():
            raise AssertionError("Robust/naive batch keys differ.")
        for key in robust_batch:
            if key == "high_value_masks":
                continue
            np.testing.assert_array_equal(
                np.asarray(robust_batch[key]),
                np.asarray(naive_batch[key]),
                err_msg=f"Robust/naive mismatch outside continuation mask: {key}",
            )
        robust_masks = np.asarray(robust_batch["high_value_masks"])
        naive_masks = np.asarray(naive_batch["high_value_masks"])
        changed = robust_masks != naive_masks
        if not np.any(changed):
            raise AssertionError("Pairing check sampled no cut-affected backups.")
        np.testing.assert_array_equal(robust_masks[~changed], naive_masks[~changed])
        np.testing.assert_array_equal(robust_masks[changed], np.ones(np.sum(changed)))
        np.testing.assert_array_equal(naive_masks[changed], np.zeros(np.sum(changed)))
        print(
            json.dumps(
                {
                    "event": "pairing_check",
                    "batch_size": int(check_size),
                    "cut_affected_backups": int(np.sum(changed)),
                    "status": "PASS",
                },
                sort_keys=True,
            )
        )
        if args.verify_only:
            return

    random.seed(args.seed)
    np.random.seed(args.seed)
    train_dataset = make_dataset(args.mode)
    example_batch = train_dataset.sample(1)
    agent = DQCAgent.create(args.seed, example_batch, config)

    if args.output_dir is None:
        suffix = f"seed{args.seed}"
        if args.mode != "original":
            suffix += f"-seg{args.segmentation_seed}"
        output_dir = Path("runs/dqc-ns-segbench") / args.mode / suffix
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    source_terminals = np.asarray(train_base["terminals"]) > 0
    source_valids = np.asarray(train_base["valids"]) > 0
    effective_fraction = 0.0 if candidate_count == 0 else cut_states.size / candidate_count
    start_record = {
        "event": "train_start",
        "external_repository": "ColinQiyangLi/dqc",
        "external_revision": DQC_REVISION,
        "external_method": "NS n=25",
        "env_name": args.env_name,
        "mode": args.mode,
        "seed": args.seed,
        "segmentation_seed": None if args.mode == "original" else args.segmentation_seed,
        "segmentation_count": int(cut_states.size),
        "segmentation_candidate_count": int(candidate_count),
        "segmentation_fraction": float(effective_fraction),
        "cut_state_sha256": sha256_array(cut_states),
        "source_terminal_sha256": sha256_array(source_terminals),
        "source_valid_sha256": sha256_array(source_valids),
        "dataset_size": int(train_base.size),
        "config": to_jsonable(config.to_dict()),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
    }

    with metrics_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(start_record, sort_keys=True) + "\n")
        log.flush()

        first_time = time.time()
        interval_start = first_time
        last_logged_step = 0
        last_info: dict[str, Any] = {}
        for step in tqdm.trange(1, args.offline_steps + 1, smoothing=0.1, dynamic_ncols=True):
            batch = train_dataset.sample(int(config["batch_size"]))
            agent, update_info = agent.update(batch)
            last_info = update_info

            if step == 1 or step % args.log_interval == 0 or step == args.offline_steps:
                host_info = jax.tree_util.tree_map(
                    lambda value: np.asarray(jax.device_get(value)), update_info
                )
                now = time.time()
                steps_since_log = step - last_logged_step
                record = {
                    "event": "train_step",
                    "step": step,
                    "seconds_per_step": (now - interval_start) / max(1, steps_since_log),
                    "elapsed_seconds": now - first_time,
                    "metrics": to_jsonable(host_info),
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                interval_start = now
                last_logged_step = step

        if last_info:
            jax.tree_util.tree_map(
                lambda value: np.asarray(jax.device_get(value)), last_info
            )

        # Reset evaluation-side RNGs so each condition uses the same rollout RNG
        # sequence. The learned policy is the only intended difference.
        random.seed(args.eval_seed)
        np.random.seed(args.eval_seed)
        try:
            eval_env.reset(seed=args.eval_seed)
        except TypeError:
            eval_env.reset()

        task_infos = (
            eval_env.unwrapped.task_infos
            if hasattr(eval_env.unwrapped, "task_infos")
            else eval_env.task_infos
        )
        action_dim = int(example_batch["actions"].shape[-1])
        task_success: dict[int, float] = {}
        for task_id in range(1, len(task_infos) + 1):
            eval_info, _, _ = evaluate(
                agent=agent,
                agent_name=config["agent_name"],
                env=eval_env,
                goal_conditioned=True,
                task_id=task_id,
                num_eval_episodes=args.eval_episodes,
                num_video_episodes=0,
                action_dim=action_dim,
            )
            task_success[task_id] = float(eval_info.get("success", 0.0))

        rollout_record = {
            "event": "rollout_eval",
            "step": args.offline_steps,
            "rollout_success_rate": float(np.mean(list(task_success.values()))),
            "task_success_rates": task_success,
            "episodes_per_task": args.eval_episodes,
            "eval_seed": args.eval_seed,
        }
        log.write(json.dumps(rollout_record, sort_keys=True) + "\n")
        log.flush()

        if not args.no_final_checkpoint:
            save_agent(agent, str(output_dir), args.offline_steps)
            log.write(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "step": args.offline_steps,
                        "path": str(output_dir / f"params_{args.offline_steps}.pkl"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    print(json.dumps(rollout_record, indent=2, sort_keys=True))
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
