"""Shared offline evaluation helpers."""

from __future__ import annotations

import numpy as np

from adaptive_gcrl.algorithms.base import EvaluationResult
from adaptive_gcrl.data.replay_buffer import TransitionBatch


def evaluate_agent_batch(agent, batch: TransitionBatch) -> EvaluationResult:
    """Evaluate a batch while measuring actor error against actor goals.

    Official OGBench goal sampling draws value and actor goals separately. Some
    agent implementations historically computed ``eval_action_mse`` using the
    value goal even though the actor is optimized and executed against the actor
    goal. This wrapper preserves every agent-provided metric but recomputes the
    action MSE with actor goals whenever they are available.
    """
    evaluation = agent.evaluate_batch(batch)
    if batch.actor_goals is None:
        return evaluation

    predictions = agent.predict(batch.observations, batch.actor_goals)
    evaluation.metrics["eval_action_mse"] = float(
        np.mean((predictions - batch.actions) ** 2)
    )
    return evaluation
