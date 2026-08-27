"""Evaluate a saved agent checkpoint on offline batches and optional OGBench rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_gcrl.algorithms.factory import create_agent
from adaptive_gcrl.data.replay_buffer import ReplayBuffer
from adaptive_gcrl.evaluation.offline_metrics import evaluate_agent_batch
from adaptive_gcrl.training.checkpoints import checkpoint_metadata_path, load_agent_checkpoint, read_metadata
from adaptive_gcrl.utils.config import apply_overrides, load_config_files
from adaptive_gcrl.utils.seeding import seed_everything
from train import load_training_batch, maybe_run_rollout_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help="Override config values with key=value syntax.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config_files(args.config), args.set)
    seed = int(config.get("seed", 0))
    rng = seed_everything(seed)
    batch = load_training_batch(config, seed)
    agent = create_agent(config, batch)
    load_agent_checkpoint(agent, args.checkpoint)

    eval_batch_size = int(config.get("eval_batch_size", batch.size))
    replay_buffer = ReplayBuffer(batch)
    eval_batch = batch if eval_batch_size >= batch.size else replay_buffer.sample(eval_batch_size, rng)
    evaluation = evaluate_agent_batch(agent, eval_batch)
    rollout_metrics = maybe_run_rollout_eval(config, agent, seed)
    metadata_path = checkpoint_metadata_path(args.checkpoint)
    metadata = read_metadata(metadata_path) if metadata_path.exists() else None
    payload = {
        "checkpoint": str(args.checkpoint),
        "metadata": None if metadata is None else metadata.__dict__,
        "eval": evaluation.metrics,
        "rollout_eval": rollout_metrics,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
