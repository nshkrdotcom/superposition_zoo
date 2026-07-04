"""Ground-truth-anchored superposition/feature-packing metrics (Phase 0).

These compare model output directly against known ground-truth features --
no probing, no inference of what the "true" feature is.
"""

from __future__ import annotations

import torch


def per_feature_reconstruction_error(
    true_values: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error per feature, averaged over the batch dimension.

    Args:
        true_values: ``(batch, n_features)`` ground truth.
        reconstructed: ``(batch, n_features)`` model output.

    Returns:
        ``(n_features,)`` tensor of per-feature MSE.
    """
    return ((true_values - reconstructed) ** 2).mean(dim=0)


def interference_matrix(
    active_mask: torch.Tensor,
    per_example_squared_error: torch.Tensor,
) -> torch.Tensor:
    """How much feature ``i``'s reconstruction error changes when feature ``j`` is active.

    ``result[i, j]`` is ``mean(error_i | j active) - mean(error_i | j inactive)``.
    A positive value means feature ``j`` being active makes feature ``i`` harder
    to reconstruct -- evidence the two features are sharing a packed direction.
    The diagonal is defined to be zero (a feature does not "interfere with
    itself" in this sense). Cells with no contrast (feature ``j`` constant
    across the whole batch) are ``NaN`` rather than a fabricated number.

    Args:
        active_mask: ``(batch, n_features)`` boolean ground-truth activity.
        per_example_squared_error: ``(batch, n_features)`` squared error per
            example per feature (not yet averaged over the batch).

    Returns:
        ``(n_features, n_features)`` interference matrix.
    """
    _, n_features = active_mask.shape
    result = torch.zeros(n_features, n_features)

    for j in range(n_features):
        active_j = active_mask[:, j]
        inactive_j = ~active_j
        if not active_j.any() or not inactive_j.any():
            result[:, j] = float("nan")
            continue
        mean_error_active = per_example_squared_error[active_j].mean(dim=0)
        mean_error_inactive = per_example_squared_error[inactive_j].mean(dim=0)
        result[:, j] = mean_error_active - mean_error_inactive

    result.fill_diagonal_(0.0)
    return result


def importance_weighted_loss(
    true_values: torch.Tensor,
    reconstructed: torch.Tensor,
    importance: torch.Tensor,
) -> torch.Tensor:
    """Importance-weighted reconstruction loss, as in Elhage et al. (2022).

    ``loss = mean_over_batch( sum_over_features( importance * (true - reconstructed)^2 ) )``

    Args:
        true_values: ``(batch, n_features)`` ground truth.
        reconstructed: ``(batch, n_features)`` model output.
        importance: ``(n_features,)`` per-feature weight.

    Returns:
        A scalar loss tensor.
    """
    squared_error = (true_values - reconstructed) ** 2
    weighted = squared_error * importance
    return weighted.sum(dim=-1).mean()


def capacity_summary(
    per_feature_mse: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    """Summarize how many features were reconstructed below an error threshold.

    Args:
        per_feature_mse: ``(n_features,)`` per-feature MSE, e.g. from
            :func:`per_feature_reconstruction_error`.
        threshold: a feature counts as "well reconstructed" if its MSE is
            strictly below this value.

    Returns:
        A dict with ``num_features``, ``num_well_reconstructed``, and
        ``fraction_well_reconstructed``.
    """
    num_features = per_feature_mse.shape[0]
    well_reconstructed = per_feature_mse < threshold
    num_well_reconstructed = int(well_reconstructed.sum().item())
    return {
        "num_features": num_features,
        "num_well_reconstructed": num_well_reconstructed,
        "fraction_well_reconstructed": num_well_reconstructed / num_features,
    }
