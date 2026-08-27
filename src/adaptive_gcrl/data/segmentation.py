"""Synthetic trajectory resegmentation for boundary-robustness studies."""

from __future__ import annotations

import numpy as np


def periodic_segmentation_boundaries(
    terminals: np.ndarray,
    *,
    interval: int,
    offset: int = 0,
) -> np.ndarray:
    """Add artificial boundaries without removing source trajectory ends."""
    if interval <= 0:
        raise ValueError("interval must be positive.")
    if not 0 <= offset < interval:
        raise ValueError("offset must be in [0, interval).")
    source_boundaries = np.asarray(terminals, dtype=bool)
    if source_boundaries.ndim != 1:
        raise ValueError("terminals must be a vector.")
    boundaries = source_boundaries.copy()
    candidate_indices = np.arange(offset, boundaries.size, interval)
    boundaries[candidate_indices] = True
    if boundaries.size:
        boundaries[-1] = True
    return boundaries


def random_segmentation_boundaries(
    terminals: np.ndarray,
    *,
    cut_probability: float,
    seed: int,
) -> np.ndarray:
    """Add reproducible Bernoulli backup cuts to source trajectories."""
    if not 0.0 < cut_probability <= 1.0:
        raise ValueError("cut_probability must be in (0, 1].")
    source_boundaries = np.asarray(terminals, dtype=bool)
    if source_boundaries.ndim != 1:
        raise ValueError("terminals must be a vector.")
    rng = np.random.default_rng(seed)
    boundaries = source_boundaries | (
        rng.random(source_boundaries.size) < cut_probability
    )
    if boundaries.size:
        boundaries[-1] = True
    return boundaries


def fixed_count_segmentation_boundaries(
    terminals: np.ndarray,
    *,
    num_cuts: int,
    seed: int,
) -> np.ndarray:
    """Add exactly ``num_cuts`` reproducible artificial nonterminal cuts.

    Existing source boundaries are always preserved and do not count toward
    ``num_cuts``. The final dataset index is also forced to be a backup boundary
    for safety but is excluded from the artificial-cut count when it was not
    already a source boundary.
    """
    source_boundaries = np.asarray(terminals, dtype=bool)
    if source_boundaries.ndim != 1:
        raise ValueError("terminals must be a vector.")
    if num_cuts < 0:
        raise ValueError("num_cuts must be non-negative.")

    candidate_mask = ~source_boundaries
    if candidate_mask.size:
        candidate_mask[-1] = False
    candidate_indices = np.flatnonzero(candidate_mask)
    if num_cuts > candidate_indices.size:
        raise ValueError(
            f"Requested {num_cuts} artificial cuts but only "
            f"{candidate_indices.size} eligible locations exist."
        )

    boundaries = source_boundaries.copy()
    if num_cuts:
        rng = np.random.default_rng(seed)
        selected = rng.choice(
            candidate_indices,
            size=num_cuts,
            replace=False,
        )
        boundaries[selected] = True
    if boundaries.size:
        boundaries[-1] = True
    return boundaries


def boundary_continuation_mask(
    source_boundaries: np.ndarray,
    backup_boundaries: np.ndarray,
    *,
    source_continues: bool,
    artificial_continues: bool,
) -> np.ndarray:
    """Mark which backup boundaries retain a valid continuation bootstrap.

    Source-trajectory ends and artificial cuts are controlled independently so
    a segmentation intervention can change only the semantics of the newly
    inserted artificial cuts while holding source-boundary treatment fixed.
    """
    source = np.asarray(source_boundaries, dtype=bool)
    backup = np.asarray(backup_boundaries, dtype=bool)
    if source.ndim != 1 or backup.ndim != 1 or source.shape != backup.shape:
        raise ValueError("source_boundaries and backup_boundaries must be same-shaped vectors.")
    if np.any(source & ~backup):
        raise ValueError("backup_boundaries must retain every source boundary.")

    continuations = np.zeros_like(backup, dtype=bool)
    if source_continues:
        continuations[source] = True
    if artificial_continues:
        continuations[backup & ~source] = True
    return continuations
