"""Apply Phase 0's interference-matrix analysis to Phase 1 models.

This is the actual "does the mixing primitive affect feature isolation, not
just task accuracy" measurement doc 4 was built around -- everything up to
this point (recall accuracy, packing capacity) is either a performance
metric or a per-feature-averaged summary; the interference matrix is the
first thing here that can reveal *cross-feature* structure, i.e. whether
one feature being active makes another harder to reconstruct because the
architecture is packing them into a shared, overlapping direction.
"""

from __future__ import annotations

import torch

from superposition_zoo.metrics.packing import interference_matrix


def position_masked_interference(
    target: torch.Tensor,
    predicted: torch.Tensor,
    active_mask: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    """:func:`interference_matrix`, restricted to a subset of sequence positions.

    Lets the same interference analysis be run separately on, e.g., only
    pointer positions vs. only content positions -- the interesting
    question is whether a mixing primitive shows *more* cross-feature
    interference specifically where it has to move information across the
    sequence, not just in the trivial local-copy case.

    Args:
        target: ``(batch, seq_len, n_features)``.
        predicted: ``(batch, seq_len, n_features)``.
        active_mask: ``(batch, seq_len, n_features)`` ground-truth activity.
        position_mask: ``(batch, seq_len)`` bool -- which positions to include.

    Returns:
        ``(n_features, n_features)`` interference matrix, computed only
        over the selected positions.
    """
    selected_target = target[position_mask]
    selected_predicted = predicted[position_mask]
    selected_active = active_mask[position_mask]
    per_example_squared_error = (selected_target - selected_predicted) ** 2
    return interference_matrix(selected_active, per_example_squared_error)
