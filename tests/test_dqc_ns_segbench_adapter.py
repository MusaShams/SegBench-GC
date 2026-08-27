import numpy as np

from scripts.run_dqc_ns_segbench import (
    compute_cut_plan,
    eligible_artificial_cut_states,
    make_artificial_cut_states,
)


def test_eligible_cut_states_do_not_cross_source_boundaries():
    # Two compact trajectories of length five. Indices 3/4 and 8/9 are the
    # compact terminal/final-state pair; index 5 starts the second trajectory.
    terminals = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.float32)
    valids = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=np.float32)

    states = eligible_artificial_cut_states(terminals, valids)

    np.testing.assert_array_equal(states, np.array([1, 2, 3, 6, 7, 8]))
    assert 5 not in states  # reset state after an original source boundary
    assert 4 not in states  # successor of the source-terminal transition
    assert 9 not in states


def test_fixed_count_cut_sampling_is_exact_paired_and_deterministic():
    terminals = np.zeros(101, dtype=np.float32)
    valids = np.ones(101, dtype=np.float32)
    terminals[-2:] = 1
    valids[-1] = 0

    first, candidates = make_artificial_cut_states(
        terminals,
        valids,
        seed=101,
        fraction=0.035,
        count=17,
    )
    second, candidates_again = make_artificial_cut_states(
        terminals,
        valids,
        seed=101,
        fraction=0.9,
        count=17,
    )

    assert candidates == candidates_again
    assert first.size == 17
    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first) > 0)


def test_fraction_mode_rounds_to_exact_count():
    terminals = np.zeros(1001, dtype=np.float32)
    valids = np.ones(1001, dtype=np.float32)
    terminals[-2:] = 1
    valids[-1] = 0

    cuts, candidates = make_artificial_cut_states(
        terminals,
        valids,
        seed=7,
        fraction=0.035,
        count=None,
    )

    assert cuts.size == round(0.035 * candidates)


def test_cut_plan_uses_first_strictly_internal_cut_only():
    idxs = np.array([0, 5, 10, 15], dtype=np.int64)
    original_horizons = np.array([8, 4, 5, 2], dtype=np.int64)
    cut_states = np.array([3, 9, 12, 17], dtype=np.int64)

    should_cut, next_cut, new_horizons = compute_cut_plan(
        idxs,
        original_horizons,
        cut_states,
        dataset_size=20,
    )

    # 0->3 and 10->12 are shortened. 5->9 and 15->17 land exactly at the
    # original endpoint and therefore must not be reclassified as artificial.
    np.testing.assert_array_equal(should_cut, np.array([True, False, True, False]))
    np.testing.assert_array_equal(next_cut, np.array([3, 9, 12, 17]))
    np.testing.assert_array_equal(new_horizons, np.array([3, 4, 2, 2]))
