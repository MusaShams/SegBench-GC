"""Agent factory used by command-line training scripts."""

from __future__ import annotations

from typing import Any, Mapping

from adaptive_gcrl.algorithms.adaptive_horizon import AdaptiveHorizonBaselineConfig, make_adaptive_horizon_training_agent
from adaptive_gcrl.algorithms.base import OfflineAgent
from adaptive_gcrl.algorithms.bc import BehavioralCloningConfig, make_behavioral_cloning_agent
from adaptive_gcrl.algorithms.fixed_temporal import FixedTemporalConfig, make_fixed_temporal_baseline
from adaptive_gcrl.data.replay_buffer import TransitionBatch


def create_agent(config: Mapping[str, Any], batch: TransitionBatch) -> OfflineAgent:
    algorithm = str(config.get("algorithm", "behavioral_cloning"))
    if algorithm in {"behavioral_cloning", "bc"}:
        return make_behavioral_cloning_agent(
            batch,
            BehavioralCloningConfig(
                learning_rate=float(config.get("learning_rate", 0.05)),
                l2_regularization=float(config.get("l2_regularization", 0.0)),
            ),
        )
    if algorithm in {"fixed_horizon", "fixed_chunk", "fixed_temporal"}:
        return make_fixed_temporal_baseline(
            batch,
            FixedTemporalConfig(
                horizon=int(config.get("horizon", 8)),
                chunk_size=int(config.get("chunk_size", 1)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                l2_regularization=float(config.get("l2_regularization", 0.0)),
            ),
        )
    if algorithm in {"iql", "goal_conditioned_iql", "torch_iql"}:
        from adaptive_gcrl.algorithms.torch_iql import TorchIQLConfig, make_torch_iql_agent

        return make_torch_iql_agent(
            batch,
            TorchIQLConfig(
                learning_rate=float(config.get("learning_rate", 1e-3)),
                hidden_dim=int(config.get("hidden_dim", 256)),
                discount=float(config.get("discount", 0.99)),
                expectile=float(config.get("expectile", 0.7)),
                advantage_temperature=float(config.get("advantage_temperature", 3.0)),
                advantage_clip=float(config.get("advantage_clip", 100.0)),
                target_tau=float(config.get("target_tau", 0.005)),
                hidden_layers=int(config.get("hidden_layers", 2)),
                activation=str(config.get("activation", "relu")),
                layer_norm=bool(config.get("layer_norm", True)),
                actor_loss_mode=str(config.get("actor_loss_mode", "awr")),
                actor_alpha=float(config.get("actor_alpha", config.get("advantage_temperature", 3.0))),
                normalize_inputs=bool(config.get("normalize_inputs", False)),
                actor_output_activation=str(config.get("actor_output_activation", "identity")),
                goal_direction_loss_weight=float(config.get("goal_direction_loss_weight", 0.0)),
                device=str(config.get("device", "cpu")),
            ),
        )
    if algorithm in {"adaptive_horizon", "adaptive_horizon_chunking"}:
        horizons = config.get("horizons", (1, 2, 4, 8, 16))
        return make_adaptive_horizon_training_agent(
            batch,
            AdaptiveHorizonBaselineConfig(
                horizons=tuple(int(horizon) for horizon in horizons),
                chunk_size=int(config.get("chunk_size", 4)),
                temperature=float(config.get("temperature", 1.0)),
                uncertainty_penalty=float(config.get("uncertainty_penalty", 0.0)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                critic_learning_rate=float(config.get("critic_learning_rate", 0.005)),
                l2_regularization=float(config.get("l2_regularization", 0.0)),
                learned_gate=bool(config.get("learned_gate", False)),
                gate_learning_rate=float(config.get("gate_learning_rate", 1e-3)),
                gate_hidden_dim=int(config.get("gate_hidden_dim", 32)),
            ),
        )
    if algorithm in {"adaptive_iql", "torch_adaptive_iql"} or algorithm.startswith("fixed_horizon_iql"):
        from adaptive_gcrl.algorithms.torch_adaptive_iql import TorchAdaptiveIQLConfig, make_torch_adaptive_iql_agent

        horizons = config.get("horizons", (1, 2, 4, 8))
        return make_torch_adaptive_iql_agent(
            batch,
            TorchAdaptiveIQLConfig(
                horizons=tuple(int(horizon) for horizon in horizons),
                chunk_size=int(config.get("chunk_size", 4)),
                learning_rate=float(config.get("learning_rate", 1e-3)),
                gate_learning_rate=float(config.get("gate_learning_rate", 1e-3)),
                hidden_dim=int(config.get("hidden_dim", 256)),
                gate_hidden_dim=int(config.get("gate_hidden_dim", 32)),
                discount=float(config.get("discount", 0.99)),
                expectile=float(config.get("expectile", 0.7)),
                advantage_temperature=float(config.get("advantage_temperature", 3.0)),
                advantage_clip=float(config.get("advantage_clip", 100.0)),
                target_tau=float(config.get("target_tau", 0.005)),
                hidden_layers=int(config.get("hidden_layers", 2)),
                activation=str(config.get("activation", "relu")),
                layer_norm=bool(config.get("layer_norm", True)),
                actor_loss_mode=str(config.get("actor_loss_mode", "awr")),
                actor_alpha=float(config.get("actor_alpha", config.get("advantage_temperature", 3.0))),
                uncertainty_penalty=float(config.get("uncertainty_penalty", 0.25)),
                horizon_value_weight=float(config.get("horizon_value_weight", 1.0)),
                horizon_value_mode=str(config.get("horizon_value_mode", "cumulative")),
                horizon_penalty=float(config.get("horizon_penalty", 0.0)),
                horizon_prior_center=None
                if config.get("horizon_prior_center") in {None, "", "none", "None"}
                else float(config.get("horizon_prior_center")),
                horizon_prior_penalty=float(config.get("horizon_prior_penalty", 0.0)),
                gate_target_smoothing=float(config.get("gate_target_smoothing", 0.0)),
                gate_entropy_regularization=float(config.get("gate_entropy_regularization", 0.0)),
                gate_selection_strategy=str(config.get("gate_selection_strategy", "argmax")),
                actor_horizon_weighting=str(config.get("actor_horizon_weighting", "selected")),
                gate_execution_strategy=str(config.get("gate_execution_strategy", "argmax")),
                static_horizon_weights=None
                if config.get("static_horizon_weights") is None
                or config.get("static_horizon_weights") in ("", "none", "None")
                else tuple(
                    float(weight)
                    for weight in config.get("static_horizon_weights")
                ),
                support_temperature=None
                if config.get("support_temperature") is None
                else float(config.get("support_temperature")),
                cross_horizon_consistency_weight=float(
                    config.get("cross_horizon_consistency_weight", 0.0)
                ),
                normalize_inputs=bool(config.get("normalize_inputs", False)),
                actor_output_activation=str(config.get("actor_output_activation", "identity")),
                goal_direction_loss_weight=float(config.get("goal_direction_loss_weight", 0.0)),
                device=str(config.get("device", "cpu")),
            ),
        )
    raise ValueError(f"Unsupported algorithm for executable training harness: {algorithm}")
