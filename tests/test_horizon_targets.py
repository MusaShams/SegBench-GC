import numpy as np

from adaptive_gcrl.data.horizon_targets import attach_horizon_targets, compute_goal_horizon_targets
from adaptive_gcrl.data.replay_buffer import ReplayBuffer, TransitionBatch


def test_goal_horizon_targets_use_start_goal_and_stop_at_terminal() -> None:
    batch = TransitionBatch(
        observations=np.array([[0.0], [1.0], [10.0], [11.0]]),
        actions=np.zeros((4, 1)),
        rewards=np.zeros(4),
        next_observations=np.array([[1.0], [2.0], [11.0], [12.0]]),
        terminals=np.array([False, True, False, True]),
        goals=np.array([[2.0], [100.0], [12.0], [100.0]]),
    )

    targets = compute_goal_horizon_targets(
        batch,
        (1, 2, 3),
        discount=0.5,
        reward_mode="sparse_success",
        success_threshold=0.1,
    )

    np.testing.assert_allclose(targets.returns[0], np.array([0.0, 0.5, 0.5]))
    np.testing.assert_allclose(targets.discounts[0], np.array([0.5, 0.0, 0.0]))
    np.testing.assert_allclose(targets.next_observations[0, :, 0], np.array([1.0, 2.0, 2.0]))
    np.testing.assert_array_equal(targets.effective_steps[0], np.array([1, 2, 2]))


def test_replay_sampling_preserves_precomputed_horizon_targets() -> None:
    batch = TransitionBatch(
        observations=np.arange(8, dtype=float).reshape(4, 2),
        actions=np.zeros((4, 1)),
        rewards=np.zeros(4),
        next_observations=np.arange(2, 10, dtype=float).reshape(4, 2),
        terminals=np.array([False, False, False, True]),
        goals=np.ones((4, 2)),
    )
    targets = compute_goal_horizon_targets(
        batch,
        (1, 2),
        discount=0.99,
        reward_mode="dense_negative_distance",
        success_threshold=1.0,
    )
    sampled = ReplayBuffer(attach_horizon_targets(batch, targets)).sample(3, np.random.default_rng(0))

    assert sampled.horizon_targets is not None
    assert sampled.horizon_targets.horizons == (1, 2)
    assert sampled.horizon_targets.returns.shape == (3, 2)
    assert sampled.horizon_targets.effective_steps.shape == (3, 2)


def test_goal_horizon_targets_stop_at_success_before_terminal() -> None:
    batch = TransitionBatch(
        observations=np.array([[0.0], [1.0], [2.0]]),
        actions=np.zeros((3, 1)),
        rewards=np.zeros(3),
        next_observations=np.array([[1.0], [2.0], [3.0]]),
        terminals=np.array([False, False, True]),
        goals=np.array([[1.0], [100.0], [100.0]]),
    )

    targets = compute_goal_horizon_targets(
        batch,
        (1, 3),
        discount=0.99,
        reward_mode="sparse_success",
        success_threshold=0.1,
    )

    np.testing.assert_allclose(targets.returns[0], [1.0, 1.0])
    np.testing.assert_allclose(targets.discounts[0], [0.0, 0.0])
    np.testing.assert_array_equal(targets.effective_steps[0], [1, 1])
