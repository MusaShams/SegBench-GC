import numpy as np
import pytest

from adaptive_gcrl.data.offline_dataset import goal_relabel_batch_from_dataset, goal_relabel_rewards, transition_batch_from_dataset


def test_transition_batch_from_dataset_uses_explicit_next_observations() -> None:
    batch = transition_batch_from_dataset(
        {
            "observations": np.zeros((3, 2)),
            "actions": np.ones((3, 1)),
            "rewards": np.arange(3),
            "next_observations": np.ones((3, 2)),
            "terminals": np.array([False, False, True]),
            "desired_goals": np.full((3, 2), 2.0),
        }
    )

    assert batch.size == 3
    assert batch.goals is not None
    assert batch.terminals.dtype == bool


def test_transition_batch_from_dataset_can_infer_next_observations() -> None:
    batch = transition_batch_from_dataset(
        {
            "observations": np.arange(8).reshape(4, 2),
            "actions": np.ones((4, 1)),
            "rewards": np.arange(4),
            "dones": np.array([False, False, False, True]),
        }
    )

    assert batch.size == 3
    assert batch.next_observations[0, 0] == 2


def test_transition_batch_from_dataset_requires_terminal_flags() -> None:
    with pytest.raises(KeyError, match="terminals or dones"):
        transition_batch_from_dataset(
            {
                "observations": np.zeros((3, 2)),
                "actions": np.ones((3, 1)),
                "rewards": np.arange(3),
                "next_observations": np.ones((3, 2)),
            }
        )


def test_goal_relabel_batch_from_reward_free_dataset() -> None:
    batch = goal_relabel_batch_from_dataset(
        {
            "observations": np.arange(12, dtype=float).reshape(6, 2),
            "actions": np.ones((6, 1)),
            "next_observations": np.arange(2, 14, dtype=float).reshape(6, 2),
            "terminals": np.array([False, False, True, False, False, True]),
        },
        seed=0,
    )

    assert batch.size == 6
    assert batch.goals is not None
    assert batch.goals.shape == (6, 2)
    assert batch.rewards.shape == (6,)


def test_goal_relabel_rewards_support_sparse_success() -> None:
    rewards = goal_relabel_rewards(
        achieved_goals=np.array([[0.0, 0.0], [2.0, 0.0]]),
        goals=np.array([[0.5, 0.0], [0.0, 0.0]]),
        reward_mode="sparse_success",
        success_threshold=1.0,
    )

    np.testing.assert_array_equal(rewards, np.array([1.0, 0.0]))


def test_goal_relabel_batch_can_use_sparse_rewards() -> None:
    batch = goal_relabel_batch_from_dataset(
        {
            "observations": np.arange(8, dtype=float).reshape(4, 2),
            "actions": np.ones((4, 1)),
            "next_observations": np.arange(2, 10, dtype=float).reshape(4, 2),
            "terminals": np.array([False, False, False, True]),
        },
        seed=0,
        reward_mode="sparse_success",
        success_threshold=1.0,
    )

    assert set(np.unique(batch.rewards)) <= {0.0, 1.0}
