import numpy as np
import pytest

from adaptive_gcrl.data.official_goal_sampling import (
    OfficialGoalReplayBuffer,
    OfficialGoalSamplingConfig,
)
from adaptive_gcrl.data.replay_buffer import TransitionBatch


def make_ordered_batch() -> TransitionBatch:
    return TransitionBatch(
        observations=np.arange(12, dtype=np.float32).reshape(6, 2),
        actions=np.zeros((6, 2), dtype=np.float32),
        rewards=np.zeros(6, dtype=np.float32),
        next_observations=np.arange(2, 14, dtype=np.float32).reshape(6, 2),
        terminals=np.array([False, False, True, False, False, True]),
    )


def test_official_current_goals_produce_zero_rewards_and_masks() -> None:
    replay = OfficialGoalReplayBuffer(
        make_ordered_batch(),
        OfficialGoalSamplingConfig(
            value_p_curgoal=1.0,
            value_p_trajgoal=0.0,
            value_p_randomgoal=0.0,
            actor_p_curgoal=1.0,
            actor_p_trajgoal=0.0,
            actor_p_randomgoal=0.0,
        ),
    )

    sample = replay.sample(16, np.random.default_rng(0))

    np.testing.assert_array_equal(sample.rewards, np.zeros(16))
    np.testing.assert_array_equal(sample.masks, np.zeros(16))
    np.testing.assert_array_equal(sample.goals, sample.observations)
    np.testing.assert_array_equal(sample.actor_goals, sample.observations)


def test_official_stitch_sampling_carries_actor_goals_and_horizon_targets() -> None:
    replay = OfficialGoalReplayBuffer(
        make_ordered_batch(),
        OfficialGoalSamplingConfig(horizons=(1, 2, 4)),
    )

    sample = replay.sample(8, np.random.default_rng(1))

    assert sample.actor_goals is not None
    assert sample.masks is not None
    assert sample.horizon_targets is not None
    assert sample.horizon_targets.horizons == (1, 2, 4)
    assert sample.horizon_targets.returns.shape == (8, 3)
    assert sample.horizon_targets.effective_steps.shape == (8, 3)


def test_horizon_targets_bootstrap_at_trajectory_boundaries() -> None:
    replay = OfficialGoalReplayBuffer(
        make_ordered_batch(),
        OfficialGoalSamplingConfig(horizons=(1, 4)),
    )

    targets = replay._horizon_targets(
        np.array([1]),
        value_goal_indices=np.array([4]),
    )

    assert targets is not None
    np.testing.assert_allclose(targets.returns[0], [-1.0, -1.99])
    np.testing.assert_allclose(targets.discounts[0], [0.99, 0.99**2])
    np.testing.assert_array_equal(targets.effective_steps[0], [1, 2])
    np.testing.assert_array_equal(
        targets.next_observations[0, 1],
        make_ordered_batch().next_observations[2],
    )


def test_artificial_backup_boundaries_do_not_change_goal_trajectories() -> None:
    batch = make_ordered_batch()
    backup_boundaries = np.array(
        [False, True, True, False, False, True]
    )
    replay = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(horizons=(4,)),
        backup_boundaries=backup_boundaries,
    )

    assert replay._final_state_indices(np.array([0]))[0] == 2
    assert replay._backup_final_indices(np.array([0]))[0] == 1


def test_naive_artificial_boundary_zeroes_bootstrap() -> None:
    batch = make_ordered_batch()
    backup_boundaries = np.array(
        [False, True, True, False, False, True]
    )
    robust = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(horizons=(4,)),
        backup_boundaries=backup_boundaries,
        bootstrap_at_boundaries=True,
    )
    naive = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(horizons=(4,)),
        backup_boundaries=backup_boundaries,
        bootstrap_at_boundaries=False,
    )

    robust_targets = robust._horizon_targets(
        np.array([0]),
        value_goal_indices=np.array([2]),
    )
    naive_targets = naive._horizon_targets(
        np.array([0]),
        value_goal_indices=np.array([2]),
    )

    assert robust_targets is not None
    assert naive_targets is not None
    np.testing.assert_allclose(robust_targets.discounts[0], [0.99**2])
    np.testing.assert_allclose(naive_targets.discounts[0], [0.0])
    np.testing.assert_array_equal(
        robust_targets.effective_steps,
        naive_targets.effective_steps,
    )


def test_noncontinuing_boundary_never_bootstraps() -> None:
    batch = make_ordered_batch()
    boundaries = np.array(
        [False, True, True, False, False, True]
    )
    continuations = np.array(
        [False, False, True, False, False, True]
    )
    replay = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(horizons=(4,)),
        backup_boundaries=boundaries,
        bootstrap_at_boundaries=True,
        boundary_continuations=continuations,
    )

    targets = replay._horizon_targets(
        np.array([0]),
        value_goal_indices=np.array([2]),
    )

    assert targets is not None
    np.testing.assert_allclose(targets.discounts[0], [0.0])


def test_continuation_mask_rejects_non_boundary_entries() -> None:
    batch = make_ordered_batch()
    with pytest.raises(
        ValueError,
        match="may only mark backup boundaries",
    ):
        OfficialGoalReplayBuffer(
            batch,
            OfficialGoalSamplingConfig(horizons=(4,)),
            boundary_continuations=np.array(
                [True, False, True, False, False, True]
            ),
        )


def test_boundary_bootstrap_preserves_bellman_consistent_target() -> None:
    batch = make_ordered_batch()
    uncut = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(discount=0.9, horizons=(4,)),
    )
    backup_boundaries = np.array(
        [False, True, True, False, False, True]
    )
    robust = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(discount=0.9, horizons=(4,)),
        backup_boundaries=backup_boundaries,
        bootstrap_at_boundaries=True,
    )
    naive = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(discount=0.9, horizons=(4,)),
        backup_boundaries=backup_boundaries,
        bootstrap_at_boundaries=False,
    )

    uncut_target = uncut._horizon_targets(
        np.array([0]),
        value_goal_indices=np.array([4]),
    )
    robust_target = robust._horizon_targets(
        np.array([0]),
        value_goal_indices=np.array([4]),
    )
    naive_target = naive._horizon_targets(
        np.array([0]),
        value_goal_indices=np.array([4]),
    )
    assert uncut_target is not None
    assert robust_target is not None
    assert naive_target is not None
    exact_value = -10.0

    uncut_backup = (
        uncut_target.returns[0, 0]
        + uncut_target.discounts[0, 0] * exact_value
    )
    robust_backup = (
        robust_target.returns[0, 0]
        + robust_target.discounts[0, 0] * exact_value
    )
    naive_backup = (
        naive_target.returns[0, 0]
        + naive_target.discounts[0, 0] * exact_value
    )

    assert robust_backup == pytest.approx(uncut_backup)
    assert naive_backup != pytest.approx(uncut_backup)


def test_one_step_sampling_is_invariant_to_backup_boundaries() -> None:
    batch = make_ordered_batch()
    backup_boundaries = np.array(
        [False, True, True, False, False, True]
    )
    original = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(horizons=None),
    )
    resegmented = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(horizons=None),
        backup_boundaries=backup_boundaries,
        bootstrap_at_boundaries=False,
    )

    original_sample = original.sample(
        16,
        np.random.default_rng(8),
    )
    resegmented_sample = resegmented.sample(
        16,
        np.random.default_rng(8),
    )

    np.testing.assert_array_equal(
        original_sample.observations,
        resegmented_sample.observations,
    )
    np.testing.assert_array_equal(
        original_sample.goals,
        resegmented_sample.goals,
    )
    np.testing.assert_array_equal(
        original_sample.actor_goals,
        resegmented_sample.actor_goals,
    )
    np.testing.assert_array_equal(
        original_sample.rewards,
        resegmented_sample.rewards,
    )
    assert original_sample.horizon_targets is None
    assert resegmented_sample.horizon_targets is None
