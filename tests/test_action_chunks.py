import numpy as np

from adaptive_gcrl.algorithms.action_chunking import ActionChunkingAgent
from adaptive_gcrl.algorithms.base import EvaluationResult, TrainResult
from adaptive_gcrl.data.action_chunks import make_action_chunk_batch
from adaptive_gcrl.data.replay_buffer import TransitionBatch


def make_batch() -> TransitionBatch:
    return TransitionBatch(
        observations=np.arange(12, dtype=float).reshape(6, 2),
        actions=np.arange(12, dtype=float).reshape(6, 2),
        rewards=np.ones(6),
        next_observations=np.arange(2, 14, dtype=float).reshape(6, 2),
        terminals=np.array([False, False, False, False, False, True]),
        goals=np.zeros((6, 2)),
    )


def test_make_action_chunk_batch_flattens_action_windows() -> None:
    chunked = make_action_chunk_batch(make_batch(), chunk_size=3, discount=1.0)

    assert chunked.actions.shape == (4, 6)
    np.testing.assert_array_equal(chunked.actions[0], np.arange(6, dtype=float))
    assert chunked.rewards[0] == 3.0


class FixedChunkInnerAgent:
    def __init__(self) -> None:
        self.calls = 0

    def train_step(self, batch, rng):
        return TrainResult(step=1, metrics={})

    def predict(self, observations, goals=None):
        self.calls += 1
        return np.array([[1.0, 2.0, 3.0, 4.0]])

    def evaluate_batch(self, batch):
        return EvaluationResult(metrics={"eval_action_mse": 0.0})

    def predict_with_info(self, observations, goals=None):
        return self.predict(observations, goals), {"selected_horizon": np.array([4])}


def test_action_chunking_agent_caches_primitive_actions() -> None:
    inner = FixedChunkInnerAgent()
    agent = ActionChunkingAgent(inner, chunk_size=2, primitive_action_dim=2)

    first = agent.predict(np.zeros((1, 2)), np.zeros((1, 2)))
    second = agent.predict(np.zeros((1, 2)), np.zeros((1, 2)))

    np.testing.assert_array_equal(first, np.array([[1.0, 2.0]]))
    np.testing.assert_array_equal(second, np.array([[3.0, 4.0]]))
    assert inner.calls == 1


def test_action_chunking_agent_preserves_horizon_info_across_cached_actions() -> None:
    agent = ActionChunkingAgent(FixedChunkInnerAgent(), chunk_size=2, primitive_action_dim=2)

    _, first_info = agent.predict_with_info(np.zeros((1, 2)), np.zeros((1, 2)))
    _, second_info = agent.predict_with_info(np.zeros((1, 2)), np.zeros((1, 2)))

    np.testing.assert_array_equal(first_info["selected_horizon"], np.array([4.0]))
    np.testing.assert_array_equal(second_info["selected_horizon"], np.array([4.0]))
