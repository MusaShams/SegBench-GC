"""Deterministically re-evaluate a saved DQC-paper NS n=25 checkpoint.

This script reconstructs the published NS architecture/configuration, restores a
saved agent, and runs goal-conditioned Puzzle-4x5 evaluation without modifying
training state. It is intended for higher-episode publication reevaluation of
SegBench-GC independent-validation checkpoints.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np


DQC_REVISION = "df898256a77f3594b54a7268bd5f89915981da35"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(".external/dqc"))
    parser.add_argument("--env-name", default="puzzle-4x5-play-oraclerep-v0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Optimization seed used for the saved run.")
    parser.add_argument("--condition", choices=("original", "robust", "naive"), required=True)
    parser.add_argument("--segmentation-seed", type=int, default=101)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=4242)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    source_dir = args.source_dir.resolve()
    marker = source_dir / "agents" / "dqc.py"
    if not marker.is_file():
        raise FileNotFoundError(
            f"Pinned DQC source is missing at {source_dir}. "
            "Run scripts/setup_dqc_baseline.py first."
        )
    sys.path.insert(0, str(source_dir))

    import flax
    import jax
    from agents.dqc import DQCAgent, get_config
    from envs.env_utils import make_ogbench_env_and_datasets
    from utils.datasets import CGCDataset, Dataset
    from utils.evaluation import evaluate

    config = get_config()
    # Published Puzzle-4x5 NS (n=25) configuration used by the training runner.
    config["num_qs"] = 2
    config["policy_chunk_size"] = 1
    config["backup_horizon"] = 25
    config["use_chunk_critic"] = False
    config["distill_method"] = "expectile"
    config["implicit_backup_type"] = "quantile"
    config["q_agg"] = "mean"
    config["kappa_b"] = 0.7
    config["kappa_d"] = 0.5

    eval_env, train_raw, _ = make_ogbench_env_and_datasets(
        args.env_name,
        dataset_path=None,
        compact_dataset=True,
        add_info=True,
    )
    eps = 1e-5
    train_raw["actions"] = np.clip(train_raw["actions"], -1 + eps, 1 - eps)
    train_base = Dataset.create(**train_raw)
    train_dataset = CGCDataset(train_base, config)

    # Reproduce the same architecture before restoring the exact saved state.
    random.seed(args.seed)
    np.random.seed(args.seed)
    example_batch = train_dataset.sample(1)
    agent = DQCAgent.create(args.seed, example_batch, config)
    with args.checkpoint.open("rb") as handle:
        saved = pickle.load(handle)
    agent = flax.serialization.from_state_dict(agent, saved["agent"])

    # Match the training runner's evaluation-side RNG reset exactly.
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

    record = {
        "event": "dqc_ns_checkpoint_eval",
        "external_repository": "ColinQiyangLi/dqc",
        "external_revision": DQC_REVISION,
        "external_method": "NS n=25",
        "condition": args.condition,
        "seed": args.seed,
        "segmentation_seed": None if args.condition == "original" else args.segmentation_seed,
        "checkpoint": str(args.checkpoint),
        "episodes_per_task": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "rollout_success_rate": float(np.mean(list(task_success.values()))),
        "task_success_rates": task_success,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
