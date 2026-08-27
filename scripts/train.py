"""Training entry point for adaptive GCRL experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from adaptive_gcrl.algorithms.factory import create_agent
from adaptive_gcrl.algorithms.action_chunking import ActionChunkingAgent
from adaptive_gcrl.data.action_chunks import make_action_chunk_batch
from adaptive_gcrl.data.horizon_targets import attach_horizon_targets, compute_goal_horizon_targets
from adaptive_gcrl.data.official_goal_sampling import (
    OfficialGoalReplayBuffer,
    OfficialGoalSamplingConfig,
)
from adaptive_gcrl.data.ogbench import OGBenchSpec, load_ogbench_env, load_ogbench_transition_batch
from adaptive_gcrl.data.replay_buffer import ReplayBuffer, TransitionBatch
from adaptive_gcrl.data.segmentation import (
    boundary_continuation_mask,
    fixed_count_segmentation_boundaries,
    periodic_segmentation_boundaries,
    random_segmentation_boundaries,
)
from adaptive_gcrl.data.synthetic import SyntheticGCRLConfig, make_synthetic_gcrl_batch
from adaptive_gcrl.evaluation.offline_metrics import evaluate_agent_batch
from adaptive_gcrl.evaluation.rollouts import evaluate_goal_conditioned_policy
from adaptive_gcrl.training.checkpoints import (
    CheckpointMetadata,
    checkpoint_metadata_path,
    load_agent_checkpoint,
    read_metadata,
    save_agent_checkpoint,
)
from adaptive_gcrl.training.trainer import OfflineTrainer, TrainState
from adaptive_gcrl.utils.config import apply_overrides, load_config_files
from adaptive_gcrl.utils.logging import MetricLogger
from adaptive_gcrl.utils.provenance import (
    array_sha256,
    canonical_config_sha256,
    current_git_state,
    runtime_versions,
)
from adaptive_gcrl.utils.seeding import seed_everything
from adaptive_gcrl.utils.tfvc import current_tfvc_changeset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help="Override config values with key=value syntax.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_training_batch(config: dict, seed: int) -> TransitionBatch:
    suite = str(config.get("suite", "synthetic"))
    if suite == "synthetic":
        return make_synthetic_gcrl_batch(
            SyntheticGCRLConfig(
                num_transitions=int(config.get("num_transitions", 256)),
                observation_dim=int(config.get("observation_dim", 4)),
                goal_dim=int(config.get("goal_dim", config.get("observation_dim", 4))),
                action_dim=int(config.get("action_dim", 2)),
                noise_std=float(config.get("noise_std", 0.01)),
            ),
            seed=seed,
        )
    if suite == "ogbench":
        return load_ogbench_transition_batch(
            OGBenchSpec(
                task=str(config["task"]),
                dataset=str(config.get("dataset", "default")),
                observation_mode=str(config.get("observation_mode", "state")),
                goal_reward_mode=str(config.get("goal_reward_mode", "dense_negative_distance")),
                success_threshold=float(config.get("success_threshold", 1.0)),
            ),
            seed=seed,
        )
    raise ValueError(f"Unsupported training suite: {suite}")


def maybe_run_rollout_eval(config: dict, agent, seed: int) -> Optional[dict[str, float]]:
    if str(config.get("suite", "synthetic")) != "ogbench":
        return None
    episodes = int(config.get("rollout_episodes", 0))
    if episodes <= 0:
        return None
    env = load_ogbench_env(
        OGBenchSpec(
            task=str(config["task"]),
            dataset=str(config.get("dataset", "default")),
            observation_mode=str(config.get("observation_mode", "state")),
            goal_reward_mode=str(config.get("goal_reward_mode", "dense_negative_distance")),
            success_threshold=float(config.get("success_threshold", 1.0)),
        )
    )
    max_steps = config.get("rollout_max_steps")
    task_ids = config.get("rollout_task_ids")
    summary = evaluate_goal_conditioned_policy(
        env,
        agent,
        episodes=episodes,
        seed=seed,
        max_steps=None if max_steps is None else int(max_steps),
        task_ids=None
        if task_ids is None
        else tuple(int(task_id) for task_id in task_ids),
    )
    return summary.as_metrics()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config_files(args.config), args.set)
    seed = int(config.get("seed", 0))
    rng = seed_everything(seed)

    output_dir = args.output_dir or Path(str(config.get("output_dir", "runs/debug")))
    batch = load_training_batch(config, seed)
    action_chunk_size = int(config.get("action_chunk_size", 1))
    goal_sampling_mode = str(config.get("goal_sampling_mode", "static"))
    if goal_sampling_mode == "official" and action_chunk_size > 1:
        raise ValueError("Official dynamic goal sampling does not yet support action_chunk_size > 1.")
    primitive_action_dim = int(batch.actions.shape[1])
    if action_chunk_size > 1:
        batch = make_action_chunk_batch(batch, action_chunk_size, discount=float(config.get("discount", 1.0)))
    algorithm = str(config.get("algorithm", ""))
    if goal_sampling_mode != "official" and (
        algorithm in {"adaptive_iql", "torch_adaptive_iql"}
        or algorithm.startswith("fixed_horizon_iql")
    ):
        horizons = tuple(int(horizon) for horizon in config.get("horizons", (1, 2, 4, 8)))
        batch = attach_horizon_targets(
            batch,
            compute_goal_horizon_targets(
                batch,
                horizons,
                discount=float(config.get("discount", 0.99)),
                reward_mode=str(config.get("goal_reward_mode", "dense_negative_distance")),
                success_threshold=float(config.get("success_threshold", 1.0)),
            ),
        )

    source_boundaries_are_continuing = False
    artificial_boundaries_are_continuing: Optional[bool] = None
    source_boundary_mask = np.asarray(batch.terminals, dtype=bool)
    effective_backup_boundaries = source_boundary_mask.copy()
    boundary_continuations = np.zeros(batch.size, dtype=bool)
    replay_bootstrap_at_boundaries = True

    if goal_sampling_mode == "official":
        horizons = (
            tuple(int(horizon) for horizon in config.get("horizons", (1, 2, 4, 8)))
            if algorithm in {"adaptive_iql", "torch_adaptive_iql"}
            or algorithm.startswith("fixed_horizon_iql")
            else None
        )
        segmentation_interval = int(
            config.get("backup_segmentation_interval", 0)
        )
        segmentation_probability = float(
            config.get("backup_segmentation_probability", 0.0)
        )
        segmentation_count = int(
            config.get("backup_segmentation_count", 0)
        )
        if segmentation_count < 0:
            raise ValueError("backup_segmentation_count must be non-negative.")
        active_segmentation_modes = sum(
            (
                segmentation_interval > 0,
                segmentation_probability > 0.0,
                segmentation_count > 0,
            )
        )
        if active_segmentation_modes > 1:
            raise ValueError(
                "Configure only one of periodic, Bernoulli, or fixed-count "
                "backup segmentation."
            )
        backup_boundaries = None
        if segmentation_interval > 0:
            backup_boundaries = periodic_segmentation_boundaries(
                batch.terminals,
                interval=segmentation_interval,
                offset=int(config.get("backup_segmentation_offset", 0)),
            )
        elif segmentation_probability > 0.0:
            backup_boundaries = random_segmentation_boundaries(
                batch.terminals,
                cut_probability=segmentation_probability,
                seed=int(config.get("backup_segmentation_seed", seed)),
            )
        elif segmentation_count > 0:
            backup_boundaries = fixed_count_segmentation_boundaries(
                batch.terminals,
                num_cuts=segmentation_count,
                seed=int(config.get("backup_segmentation_seed", seed)),
            )
        effective_backup_boundaries = (
            np.asarray(batch.terminals, dtype=bool)
            if backup_boundaries is None
            else np.asarray(backup_boundaries, dtype=bool)
        )
        source_boundaries_are_continuing = bool(
            config.get("source_boundaries_are_continuing", False)
        )
        source_boundary_mask = np.asarray(
            batch.terminals,
            dtype=bool,
        )
        if source_boundaries_are_continuing:
            boundary_next_observations = np.asarray(
                batch.next_observations[source_boundary_mask]
            )
            if not np.all(np.isfinite(boundary_next_observations)):
                raise ValueError(
                    "Continuing source boundaries require finite stored successor observations."
                )

        explicit_artificial_continuation = config.get(
            "artificial_boundaries_are_continuing"
        )
        if explicit_artificial_continuation is None:
            # Legacy behavior is retained for reproducibility of historical runs:
            # the global bootstrap flag applies to every continuation-valid backup
            # boundary, including source boundaries marked as continuing.
            boundary_continuations = effective_backup_boundaries.copy()
            if not source_boundaries_are_continuing:
                boundary_continuations[source_boundary_mask] = False
            replay_bootstrap_at_boundaries = bool(
                config.get("bootstrap_at_backup_boundaries", True)
            )
        else:
            # Final controlled protocol: source-boundary semantics are fixed and
            # only the newly inserted artificial boundaries change between CVT
            # and the naive artificial-terminal control.
            artificial_boundaries_are_continuing = bool(
                explicit_artificial_continuation
            )
            boundary_continuations = boundary_continuation_mask(
                source_boundary_mask,
                effective_backup_boundaries,
                source_continues=source_boundaries_are_continuing,
                artificial_continues=artificial_boundaries_are_continuing,
            )
            replay_bootstrap_at_boundaries = True

        replay_buffer = OfficialGoalReplayBuffer(
            batch,
            OfficialGoalSamplingConfig(
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
            ),
            backup_boundaries=backup_boundaries,
            bootstrap_at_boundaries=replay_bootstrap_at_boundaries,
            boundary_continuations=boundary_continuations,
        )
    else:
        replay_buffer = ReplayBuffer(batch)
    agent = create_agent(config, batch)
    if action_chunk_size > 1:
        agent = ActionChunkingAgent(agent, action_chunk_size, primitive_action_dim)
    initial_step = 0
    resume_checkpoint = config.get("resume_checkpoint_path")
    if resume_checkpoint:
        resume_path = Path(str(resume_checkpoint))
        load_agent_checkpoint(agent, resume_path)
        metadata_path = checkpoint_metadata_path(resume_path)
        if metadata_path.exists():
            resume_metadata = read_metadata(metadata_path)
            initial_step = resume_metadata.step
            if resume_metadata.rng_state is not None:
                rng.bit_generator.state = resume_metadata.rng_state
    trainer = OfflineTrainer(
        agent=agent,
        replay_buffer=replay_buffer,
        batch_size=int(config.get("batch_size", 32)),
        rng=rng,
        state=TrainState(step=initial_step),
    )
    tfvc_changeset = current_tfvc_changeset(
        collection=str(config.get("tfvc_collection", "")),
        workspace=str(config.get("tfvc_workspace", "")),
    )
    repo_root = Path(__file__).resolve().parents[1]
    provenance = {
        **current_git_state(repo_root),
        "config_sha256": canonical_config_sha256(config),
        "source_boundary_sha256": array_sha256(source_boundary_mask),
        "backup_boundary_sha256": array_sha256(effective_backup_boundaries),
        **runtime_versions(),
    }
    total_steps = int(config.get("steps", 1))
    log_interval = int(config.get("log_interval", 1))
    checkpoint_interval = int(config.get("checkpoint_interval", 0))
    if log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative.")

    with MetricLogger(output_dir / "metrics.jsonl") as logger:
        logger.write(
            {
                "event": "train_start",
                "config": config,
                "seed": seed,
                "tfvc_changeset": tfvc_changeset,
                **provenance,
                "num_transitions": batch.size,
                "backup_boundary_count": int(
                    replay_buffer.backup_boundary_locs.size
                )
                if isinstance(replay_buffer, OfficialGoalReplayBuffer)
                else 0,
                "artificial_backup_boundary_count": int(
                    np.count_nonzero(
                        effective_backup_boundaries & ~source_boundary_mask
                    )
                )
                if isinstance(replay_buffer, OfficialGoalReplayBuffer)
                else 0,
                "source_boundaries_are_continuing": bool(
                    source_boundaries_are_continuing
                ),
                "artificial_boundaries_are_continuing": artificial_boundaries_are_continuing,
                "source_boundary_count": int(
                    np.count_nonzero(source_boundary_mask)
                ),
                "continuing_source_boundary_count": int(
                    np.count_nonzero(
                        source_boundary_mask
                        & boundary_continuations
                    )
                ),
                "continuing_artificial_boundary_count": int(
                    np.count_nonzero(
                        effective_backup_boundaries
                        & ~source_boundary_mask
                        & boundary_continuations
                    )
                ),
                "replay_bootstrap_at_boundaries": bool(
                    replay_bootstrap_at_boundaries
                ),
                "resume_checkpoint_path": None if resume_checkpoint is None else str(resume_checkpoint),
            }
        )
        for _ in range(total_steps):
            result = trainer.train_step()
            if (
                result.step == initial_step + 1
                or result.step % log_interval == 0
                or result.step == initial_step + total_steps
            ):
                logger.write(
                    {
                        "event": "train_step",
                        "step": result.step,
                        "metrics": result.metrics,
                    }
                )
            if checkpoint_interval and result.step % checkpoint_interval == 0:
                periodic_path = Path(
                    str(
                        config.get(
                            "periodic_checkpoint_path",
                            output_dir / "checkpoints" / "latest.pt",
                        )
                    )
                )
                save_agent_checkpoint(
                    agent,
                    periodic_path,
                    CheckpointMetadata(
                        step=result.step,
                        seed=seed,
                        tfvc_changeset=tfvc_changeset,
                        rng_state=rng.bit_generator.state,
                    ),
                )
                logger.write(
                    {
                        "event": "checkpoint",
                        "kind": "periodic",
                        "step": result.step,
                        "path": str(periodic_path),
                    }
                )
        eval_batch_size = int(config.get("eval_batch_size", batch.size))
        eval_batch = batch if eval_batch_size >= batch.size else replay_buffer.sample(eval_batch_size, rng)
        evaluation = evaluate_agent_batch(agent, eval_batch)
        logger.write({"event": "eval", "step": trainer.state.step, "metrics": evaluation.metrics})
        rollout_metrics = maybe_run_rollout_eval(config, agent, seed)
        if rollout_metrics is not None:
            logger.write({"event": "rollout_eval", "step": trainer.state.step, "metrics": rollout_metrics})
        if bool(config.get("save_checkpoint", False)):
            checkpoint_path = Path(str(config.get("checkpoint_path", output_dir / "agent.pt")))
            save_agent_checkpoint(
                agent,
                checkpoint_path,
                CheckpointMetadata(
                    step=trainer.state.step,
                    seed=seed,
                    tfvc_changeset=tfvc_changeset,
                    rng_state=rng.bit_generator.state,
                ),
            )
            logger.write(
                {
                    "event": "checkpoint",
                    "kind": "final",
                    "step": trainer.state.step,
                    "path": str(checkpoint_path),
                }
            )


if __name__ == "__main__":
    main()
