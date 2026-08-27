import numpy as np
import pytest

from adaptive_gcrl.data.segmentation import (
    boundary_continuation_mask,
    fixed_count_segmentation_boundaries,
    periodic_segmentation_boundaries,
    random_segmentation_boundaries,
)


def test_periodic_segmentation_preserves_source_boundaries() -> None:
    terminals = np.array([False, False, True, False, False, True])

    boundaries = periodic_segmentation_boundaries(
        terminals,
        interval=2,
        offset=1,
    )

    np.testing.assert_array_equal(
        boundaries,
        np.array([False, True, True, True, False, True]),
    )


def test_periodic_segmentation_validates_offset() -> None:
    with pytest.raises(ValueError, match="offset"):
        periodic_segmentation_boundaries(
            np.zeros(4, dtype=bool),
            interval=2,
            offset=2,
        )


def test_random_segmentation_is_reproducible_and_preserves_terminals() -> None:
    terminals = np.array([False, False, True, False, False, True])

    first = random_segmentation_boundaries(
        terminals,
        cut_probability=0.5,
        seed=4,
    )
    second = random_segmentation_boundaries(
        terminals,
        cut_probability=0.5,
        seed=4,
    )

    np.testing.assert_array_equal(first, second)
    assert np.all(first[terminals])
    assert first[-1]


def test_random_segmentation_validates_probability() -> None:
    with pytest.raises(ValueError, match="cut_probability"):
        random_segmentation_boundaries(
            np.zeros(4, dtype=bool),
            cut_probability=0.0,
            seed=0,
        )


def test_fixed_count_segmentation_adds_exact_number_of_cuts() -> None:
    terminals = np.array(
        [False, False, True, False, False, False, False, True]
    )

    boundaries = fixed_count_segmentation_boundaries(
        terminals,
        num_cuts=3,
        seed=9,
    )

    artificial = boundaries & ~terminals
    assert int(np.count_nonzero(artificial)) == 3
    assert np.all(boundaries[terminals])
    assert boundaries[-1]


def test_fixed_count_segmentation_is_reproducible_and_seed_sensitive() -> None:
    terminals = np.zeros(20, dtype=bool)
    terminals[[4, 9, 14, 19]] = True

    first = fixed_count_segmentation_boundaries(
        terminals,
        num_cuts=5,
        seed=11,
    )
    second = fixed_count_segmentation_boundaries(
        terminals,
        num_cuts=5,
        seed=11,
    )
    different = fixed_count_segmentation_boundaries(
        terminals,
        num_cuts=5,
        seed=12,
    )

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.count_nonzero(first & ~terminals) == 5
    assert np.count_nonzero(different & ~terminals) == 5


def test_fixed_count_segmentation_does_not_count_forced_final_boundary() -> None:
    terminals = np.array([False, False, False, False])

    boundaries = fixed_count_segmentation_boundaries(
        terminals,
        num_cuts=1,
        seed=0,
    )

    assert boundaries[-1]
    assert np.count_nonzero(boundaries[:-1]) == 1


def test_fixed_count_segmentation_rejects_too_many_cuts() -> None:
    terminals = np.array([False, True, False, True])

    with pytest.raises(ValueError, match="eligible locations"):
        fixed_count_segmentation_boundaries(
            terminals,
            num_cuts=3,
            seed=0,
        )


def test_boundary_continuation_mask_separates_source_and_artificial_semantics() -> None:
    source = np.array([False, False, True, False, False, True])
    backup = np.array([False, True, True, False, True, True])

    robust = boundary_continuation_mask(
        source,
        backup,
        source_continues=True,
        artificial_continues=True,
    )
    naive_artificial_only = boundary_continuation_mask(
        source,
        backup,
        source_continues=True,
        artificial_continues=False,
    )

    np.testing.assert_array_equal(robust, backup)
    np.testing.assert_array_equal(naive_artificial_only, source)


def test_boundary_continuation_mask_requires_source_boundaries_to_be_retained() -> None:
    source = np.array([False, True, False, True])
    backup = np.array([False, False, True, True])

    with pytest.raises(ValueError, match="retain every source boundary"):
        boundary_continuation_mask(
            source,
            backup,
            source_continues=True,
            artificial_continues=False,
        )
