"""Run SegBench-GC on the independent n-step GCSAC+BC implementation.

This adapter subclasses the pinned Horizon Reduction ``HGCDataset`` without
changing its original source-trajectory boundaries or goal sampler. Artificial
cuts only shorten the high-value backup. CVT keeps the continuation mask at a
cut; the naive control zeros it. The external repository itself remains
unmodified under ``.external/``.

For large published OGBench datasets, ``--dataset-dir`` may point to a directory
of training ``.npz`` shards. The adapter follows Horizon Reduction's reference
training loop by cycling through sorted shards every
``--dataset-replace-interval`` updates (1000 by default). Validation shards are
not required because this adapter does not compute validation losses.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


HORIZON_REDUCTION_REVISION = "c298aedcc505bc7a7b44b4d0c9318993f8b3f3fd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(".external/horizon-reduction"),
    )
    parser.add_argument(
        "--env-name",
        default="pointmaze-medium-stitch-v0",
    )
    parser.add_argument(
        "--mode",
        choices=("original", "robust", "naive"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segmentation-seed", type=int, default=101)
    parser.add_argument(
        "--segmentation-count",
        type=int,
        default=35000,
        help="Exact artificial-cut count per active training shard.",
    )
    parser.add_argument("--offline-steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Training dataset file or directory of .npz training shards. "
            "Files ending in -val.npz are ignored."
        ),
    )
    parser.add_argument(
        "--dataset-replace-interval",
        type=int,
        default=1000,
        help=(
            "Cycle to the next dataset shard after this many updates, matching "
            "the Horizon Reduction reference trainer. Set 0 to disable rotation."
        ),
    )
    parser.add_argument(
        "--no-final-checkpoint",
        action="store_true",
        help="Skip the final checkpoint (useful for integration smokes).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
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


def sha256_text_records(records: list[str]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def discover_dataset_paths(dataset_dir: Path | None) -> list[Path | None]:
    """Resolve the default dataset, one file, or a directory of train shards."""
    if dataset_dir is None:
        return [None]

    path = dataset_dir.expanduser().resolve()
    if path.is_file():
        if path.name.endswith("-val.npz"):
            raise ValueError("--dataset-dir must point to a training shard, not -val.npz.")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    paths = sorted(
        candidate
        for candidate in path.glob("*.npz")
        if not candidate.name.endswith("-val.npz")
    )
    if not paths:
        raise FileNotFoundError(f"No training .npz shards found in {path}")
    return paths


def dataset_manifest_records(paths: list[Path | None]) -> list[str]:
    if paths == [None]:
        return ["ogbench-default"]
    records: list[str] = []
    for path in paths:
        assert path is not None
        records.append(f"{path.name}\t{path.stat().st_size}")
    return records


def shard_segmentation_seed(base_seed: int, shard_label: str) -> int:
    """Derive a stable per-shard seed shared by robust and naive conditions."""
    digest = hashlib.sha256(f"{base_seed}:{shard_label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def make_artificial_cut_states(
    dataset,
    *,
    num_cuts: int,
    seed: int,
) -> np.ndarray:
    if num_cuts < 0:
        raise ValueError("segmentation_count must be non-negative.")
    terminals = np.asarray(dataset["terminals"]) > 0
    if "valids" in dataset:
        valids = np.asarray(dataset["valids"]) > 0
    else:
        valids = np.ones_like(terminals, dtype=bool)

    # A cut is placed *after* an eligible transition c, so the continuation
    # state is c + 1. Excluding terminal transitions guarantees that c + 1
    # remains inside the same original source trajectory.
    eligible = valids & ~terminals
    if eligible.size:
        eligible[-1] = False
    candidate_transitions = np.flatnonzero(eligible)
    if num_cuts > candidate_transitions.size:
        raise ValueError(
            f"Requested {num_cuts} cuts but only "
            f"{candidate_transitions.size} eligible transitions exist."
        )
    if num_cuts == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected_transitions = rng.choice(
        candidate_transitions,
        size=num_cuts,
        replace=False,
    )
    return np.sort(selected_transitions.astype(np.int64) + 1)


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
    if args.offline_steps <= 0:
        raise ValueError("offline_steps must be positive.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if args.dataset_replace_interval < 0:
        raise ValueError("dataset_replace_interval must be non-negative.")
    if args.mode == "original" and args.segmentation_count != 35000:
        # The count is ignored for original; avoid pretending otherwise.
        print("note: segmentation_count is ignored for original mode")

    source_dir = args.source_dir.resolve()
    marker = source_dir / "agents" / "ngcsacbc.py"
    if not marker.is_file():
        raise FileNotFoundError(
            f"Pinned Horizon Reduction source is missing at {source_dir}. "
            "Run scripts/setup_horizon_reduction_baseline.py first."
        )
    sys.path.insert(0, str(source_dir))

    import jax
    import jax.tree_util
    import ogbench
    import tqdm
    from agents.ngcsacbc import NGCSACBCAgent, get_config
    from envs.env_utils import make_env_and_datasets
    from ogbench.relabel_utils import add_oracle_reps, relabel_dataset
    from utils.datasets import Dataset, HGCDataset
    from utils.evaluation import evaluate
    from utils.flax_utils import save_agent

    class SegBenchHGCDataset(HGCDataset):
        def __init__(
            self,
            dataset,
            config,
            *,
            mode: str,
            cut_states: np.ndarray,
        ) -> None:
            super().__init__(dataset, config)
            self.segbench_mode = mode
            self.segbench_cut_states = np.asarray(cut_states, dtype=np.int64)

        def sample(self, batch_size: int, idxs=None, evaluation=False):
            if idxs is None:
                idxs = self.dataset.get_random_idxs(batch_size)
            idxs = np.asarray(idxs, dtype=np.int64)
            batch = super().sample(
                batch_size,
                idxs=idxs,
                evaluation=evaluation,
            )
            if self.segbench_mode == "original" or self.segbench_cut_states.size == 0:
                return batch

            # Work on writable arrays because the underlying Dataset is frozen.
            batch = {
                key: np.array(value, copy=True)
                if hasattr(value, "shape")
                else value
                for key, value in batch.items()
            }

            positions = np.searchsorted(
                self.segbench_cut_states,
                idxs,
                side="right",
            )
            has_cut = positions < self.segbench_cut_states.size
            next_cut = np.full(batch_size, self.size, dtype=np.int64)
            next_cut[has_cut] = self.segbench_cut_states[positions[has_cut]]
            cut_steps = next_cut - idxs

            original_steps = np.asarray(
                batch["high_value_subgoal_steps"],
                dtype=np.int64,
            )
            should_cut = has_cut & (cut_steps > 0) & (cut_steps < original_steps)
            if not np.any(should_cut):
                return batch

            replacement_indices = next_cut[should_cut]
            replacement_observations = self.get_observations(replacement_indices)
            next_observations = np.asarray(
                batch["high_value_next_observations"]
            ).copy()
            next_observations[should_cut] = np.asarray(replacement_observations)
            batch["high_value_next_observations"] = next_observations

            new_steps = original_steps.copy()
            new_steps[should_cut] = cut_steps[should_cut]
            batch["high_value_subgoal_steps"] = new_steps

            masks = np.asarray(batch["high_value_masks"], dtype=float).copy()
            masks[should_cut] = 1.0 if self.segbench_mode == "robust" else 0.0
            batch["high_value_masks"] = masks

            rewards = np.asarray(batch["high_value_rewards"], dtype=float).copy()
            if bool(self.config["gc_negative"]):
                gamma = float(self.config["discount"])
                k = cut_steps[should_cut].astype(float)
                if np.isclose(gamma, 1.0):
                    rewards[should_cut] = -k
                else:
                    rewards[should_cut] = -(1.0 - gamma**k) / (1.0 - gamma)
            else:
                # The artificial cut is not goal completion. Robust and naive
                # receive the same accumulated reward; only continuation differs.
                rewards[should_cut] = 0.0
            batch["high_value_rewards"] = rewards
            batch["segbench_cut_applied"] = should_cut.astype(np.float32)
            return batch

    def external_env_name(dataset_name: str) -> tuple[str, bool, bool, bool]:
        """Mirror OGBench dataset-name parsing needed to load training-only shards."""
        splits = dataset_name.split("-")
        if "singletask" in splits:
            pos = splits.index("singletask")
            env_name = "-".join(splits[: pos - 1] + splits[pos:])
            return env_name, True, True, False
        if "oraclerep" in splits:
            env_name = "-".join(splits[:-3] + splits[-1:])
            return env_name, True, False, True
        env_name = "-".join(splits[:-2] + splits[-1:])
        return env_name, False, False, False

    def load_training_shard(path: Path, env) -> Dataset:
        """Load exactly the training half of an OGBench shard, without a val file."""
        env_name, dataset_add_info, is_singletask, is_oraclerep = external_env_name(
            args.env_name
        )
        ob_dtype = (
            np.uint8
            if ("visual" in env_name or "powderworld" in env_name)
            else np.float32
        )
        action_dtype = np.int32 if "powderworld" in env_name else np.float32
        raw = ogbench.load_dataset(
            str(path),
            ob_dtype=ob_dtype,
            action_dtype=action_dtype,
            compact_dataset=True,
            add_info=dataset_add_info,
        )

        if is_singletask:
            relabel_dataset(env_name, env, raw)
        if is_oraclerep:
            add_oracle_reps(env_name, env, raw)

        # Match OGBench's make_env_and_datasets(..., add_info=False) behavior.
        for key in ("qpos", "qvel", "button_states"):
            raw.pop(key, None)

        # Match Horizon Reduction's env_utils action clipping.
        eps = 1e-5
        raw["actions"] = np.clip(raw["actions"], -1 + eps, 1 - eps)
        return Dataset.create(**raw)

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_paths = discover_dataset_paths(args.dataset_dir)
    manifest_records = dataset_manifest_records(dataset_paths)
    manifest_sha256 = sha256_text_records(manifest_records)

    if dataset_paths == [None]:
        env, first_base, _ = make_env_and_datasets(
            args.env_name,
            dataset_path=None,
        )
        # The reference helper already returns its Dataset wrapper.
        if not isinstance(first_base, Dataset):
            first_base = Dataset.create(**first_base)
    else:
        # Build the environment without forcing a paired validation shard. The
        # training shard itself is loaded below with the same OGBench compact
        # semantics and oracle-representation preprocessing as the reference.
        env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
        env.reset()
        first_base = load_training_shard(dataset_paths[0], env)

    config = get_config()
    # Config-file parsing in the reference main normally resolves this
    # placeholder. Set it explicitly in the standalone adapter.
    config["target_entropy"] = None

    def build_train_dataset(
        shard_index: int,
        *,
        preloaded_base: Dataset | None = None,
    ) -> tuple[SegBenchHGCDataset, dict[str, Any]]:
        path = dataset_paths[shard_index]
        if preloaded_base is not None:
            train_base = preloaded_base
        elif path is None:
            _, train_base, _ = make_env_and_datasets(
                args.env_name,
                dataset_path=None,
            )
            if not isinstance(train_base, Dataset):
                train_base = Dataset.create(**train_base)
        else:
            train_base = load_training_shard(path, env)

        shard_label = "ogbench-default" if path is None else path.name
        derived_seed = shard_segmentation_seed(args.segmentation_seed, shard_label)
        if args.mode == "original":
            cut_states = np.empty(0, dtype=np.int64)
        else:
            cut_states = make_artificial_cut_states(
                train_base,
                num_cuts=args.segmentation_count,
                seed=derived_seed,
            )

        wrapped = SegBenchHGCDataset(
            train_base,
            config,
            mode=args.mode,
            cut_states=cut_states,
        )
        source_terminals = np.asarray(train_base["terminals"]) > 0
        metadata = {
            "shard_index": shard_index,
            "shard_label": shard_label,
            "shard_path": None if path is None else str(path),
            "num_transitions": int(train_base.size),
            "source_terminal_sha256": sha256_array(source_terminals),
            "segmentation_seed": None if args.mode == "original" else int(derived_seed),
            "segmentation_count": 0 if args.mode == "original" else int(len(cut_states)),
            "cut_state_sha256": sha256_array(cut_states),
        }
        return wrapped, metadata

    dataset_idx = 0
    train_dataset, shard_metadata = build_train_dataset(
        dataset_idx,
        preloaded_base=first_base,
    )
    del first_base

    example_batch = train_dataset.sample(1)
    agent = NGCSACBCAgent.create(
        args.seed,
        example_batch,
        config,
    )

    if args.output_dir is None:
        suffix = f"seed{args.seed}"
        if args.mode != "original":
            suffix += f"-seg{args.segmentation_seed}"
        output_dir = Path(
            "runs/horizon-reduction-segbench"
        ) / args.mode / suffix
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    start_record = {
        "event": "train_start",
        "external_repository": "seohongpark/horizon-reduction",
        "external_revision": HORIZON_REDUCTION_REVISION,
        "env_name": args.env_name,
        "mode": args.mode,
        "seed": args.seed,
        "segmentation_seed": None if args.mode == "original" else args.segmentation_seed,
        "segmentation_count_per_shard": (
            0 if args.mode == "original" else args.segmentation_count
        ),
        "dataset_shard_count": len(dataset_paths),
        "dataset_replace_interval": args.dataset_replace_interval,
        "dataset_manifest_sha256": manifest_sha256,
        "dataset_manifest": manifest_records,
        # Keep first-shard hashes at top level for compatibility with the
        # original single-dataset logs; every shard is also logged explicitly.
        "cut_state_sha256": shard_metadata["cut_state_sha256"],
        "source_terminal_sha256": shard_metadata["source_terminal_sha256"],
        "config": to_jsonable(config.to_dict()),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
    }

    with metrics_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(start_record, sort_keys=True) + "\n")
        log.write(
            json.dumps(
                {
                    "event": "dataset_shard",
                    "effective_step": 1,
                    **shard_metadata,
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()

        first_time = time.time()
        interval_start = first_time
        last_log_step = 0
        last_info: dict[str, Any] = {}
        for step in tqdm.trange(
            1,
            args.offline_steps + 1,
            smoothing=0.1,
            dynamic_ncols=True,
        ):
            batch = train_dataset.sample(int(config["batch_size"]))
            agent, update_info = agent.update(batch)
            last_info = update_info

            if step == 1 or step % args.log_interval == 0 or step == args.offline_steps:
                host_info = jax.tree_util.tree_map(
                    lambda value: np.asarray(jax.device_get(value)),
                    update_info,
                )
                denominator = step - last_log_step
                record = {
                    "event": "train_step",
                    "step": step,
                    "seconds_per_step": (time.time() - interval_start) / denominator,
                    "elapsed_seconds": time.time() - first_time,
                    "metrics": to_jsonable(host_info),
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                interval_start = time.time()
                last_log_step = step

            if args.save_interval > 0 and step % args.save_interval == 0:
                save_agent(agent, str(output_dir), step)
                log.write(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "step": step,
                            "path": str(output_dir / f"params_{step}.pkl"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                log.flush()

            # Match the reference trainer: the current shard serves the update
            # at the boundary step, then the next shard becomes active.
            if (
                len(dataset_paths) > 1
                and args.dataset_replace_interval > 0
                and step % args.dataset_replace_interval == 0
                and step < args.offline_steps
            ):
                dataset_idx = (dataset_idx + 1) % len(dataset_paths)
                old_dataset = train_dataset
                train_dataset, shard_metadata = build_train_dataset(dataset_idx)
                del old_dataset
                del batch
                gc.collect()
                log.write(
                    json.dumps(
                        {
                            "event": "dataset_shard",
                            "effective_step": step + 1,
                            **shard_metadata,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                log.flush()

        # Synchronize the final update before timing/evaluation.
        if last_info:
            jax.tree_util.tree_map(
                lambda value: np.asarray(jax.device_get(value)),
                last_info,
            )

        task_infos = (
            env.unwrapped.task_infos
            if hasattr(env.unwrapped, "task_infos")
            else env.task_infos
        )
        task_success: dict[int, float] = {}
        for task_id in range(1, len(task_infos) + 1):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=env,
                env_name=args.env_name,
                goal_conditioned=True,
                task_id=task_id,
                config=config,
                num_eval_episodes=args.eval_episodes,
                num_video_episodes=0,
                eval_temperature=0,
            )
            task_success[task_id] = float(eval_info.get("success", 0.0))

        rollout_record = {
            "event": "rollout_eval",
            "step": args.offline_steps,
            "rollout_success_rate": float(np.mean(list(task_success.values()))),
            "task_success_rates": task_success,
            "episodes_per_task": args.eval_episodes,
        }
        log.write(json.dumps(rollout_record, sort_keys=True) + "\n")

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
