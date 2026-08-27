import numpy as np
import pytest

from adaptive_gcrl.data.replay_buffer import ReplayBuffer, TransitionBatch


def make_batch(size: int = 5) -> TransitionBatch:
    return TransitionBatch(
        observations=np.zeros((size, 3)),
        actions=np.zeros((size, 2)),
        rewards=np.zeros((size,)),
        next_observations=np.ones((size, 3)),
        terminals=np.zeros((size,), dtype=bool),
        goals=np.ones((size, 3)),
    )


def test_transition_batch_validates_batch_dimensions() -> None:
    with pytest.raises(ValueError, match="actions"):
        TransitionBatch(
            observations=np.zeros((5, 3)),
            actions=np.zeros((4, 2)),
            rewards=np.zeros((5,)),
            next_observations=np.ones((5, 3)),
            terminals=np.zeros((5,), dtype=bool),
        )


def test_replay_buffer_samples_requested_size() -> None:
    buffer = ReplayBuffer(make_batch())

    sample = buffer.sample(3, np.random.default_rng(0))

    assert sample.size == 3
    assert sample.observations.shape == (3, 3)
