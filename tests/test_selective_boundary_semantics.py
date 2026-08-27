import numpy as np

from adaptive_gcrl.data.official_goal_sampling import (
    OfficialGoalReplayBuffer,
    OfficialGoalSamplingConfig,
)
from adaptive_gcrl.data.replay_buffer import TransitionBatch
from adaptive_gcrl.data.segmentation import boundary_continuation_mask


def make_batch() -> TransitionBatch:
    return TransitionBatch(
        observations=np.arange(12, dtype=np.float32).reshape(6, 2),
        actions=np.zeros((6, 2), dtype=np.float32),
        rewards=np.zeros(6, dtype=np.float32),
        next_observations=np.arange(2, 14, dtype=np.float32).reshape(6, 2),
        terminals=np.array([False, False, True, False, False, True]),
    )


def test_naive_artificial_only_preserves_source_boundary_bootstrap() -> None:
    batch = make_batch()
    source = np.asarray(batch.terminals, dtype=bool)
    backup = np.array([False, True, True, False, False, True])
    continuations = boundary_continuation_mask(
        source,
        backup,
        source_continues=True,
        artificial_continues=False,
    )
    replay = OfficialGoalReplayBuffer(
        batch,
        OfficialGoalSamplingConfig(discount=0.9, horizons=(4,)),
        backup_boundaries=backup,
        bootstrap_at_boundaries=True,
        boundary_continuations=continuations,
    )

    artificial_target = replay._horizon_targets(
        np.array([0]),
        value_goal_indices=np.array([2]),
    )
    source_target = replay._horizon_targets(
        np.array([2]),
        value_goal_indices=np.array([4]),
    )

    assert artificial_target is not None
    assert source_target is not None
    np.testing.assert_allclose(artificial_target.discounts[0], [0.0])
    np.testing.assert_allclose(source_target.discounts[0], [0.9])
