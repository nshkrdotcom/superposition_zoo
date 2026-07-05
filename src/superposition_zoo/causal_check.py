"""Ground-truth causal verification against a real trained model.

Aggregates :func:`~superposition_zoo.metrics.causal.patch_and_verify` over
many pointer positions in a batch. Doc 4's whole methodological point was
that ground truth lets you *causally* verify a model's retrieval rather
than trusting accuracy alone -- this is the module that actually exercises
that against a real checkpoint, not just a hand-written oracle in a test.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from superposition_zoo.metrics.causal import patch_and_verify
from superposition_zoo.models.sequence_model import SequenceModel
from superposition_zoo.recall_task import RecallBatch


def load_trained_model(
    checkpoint_path: str | Path,
    n_features: int,
    d_model: int,
    mixing_primitive_name: str,
    n_layers: int = 1,
    mixing_kwargs: dict | None = None,
    device: str = "cpu",
) -> SequenceModel:
    """Reconstruct a :class:`SequenceModel` and load trained weights into it."""
    model = SequenceModel(
        n_features=n_features,
        d_model=d_model,
        mixing_primitive_name=mixing_primitive_name,
        n_layers=n_layers,
        mixing_kwargs=mixing_kwargs,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def causal_check_report(
    model: nn.Module,
    batch: RecallBatch,
    n_checks: int,
    generator: torch.Generator,
) -> dict[str, float | int]:
    """Run :func:`patch_and_verify` across many random pointer positions and aggregate.

    Args:
        model: any module implementing ``forward(input_features, control) -> predicted``.
            Wrapped in ``eval()``/``no_grad()`` internally.
        batch: a :class:`RecallBatch` to check against.
        n_checks: how many distinct pointer positions to sample (capped at
            however many actually exist in ``batch``).
        generator: an explicitly-seeded ``torch.Generator``.

    Returns:
        A dict with ``n_checks``, ``moved_toward_substitute_fraction``,
        ``moved_away_from_original_fraction``, and the mean
        distance-to-substitute before/after patching across all checks. A
        model whose retrieval is genuinely causally driven by the true
        source position should show both fractions near ``1.0``; a model
        with no real causal dependence on the source will show both near
        ``0.0``.
    """
    model.eval()
    pointer_positions = batch.is_pointer.nonzero()
    n_available = pointer_positions.shape[0]
    n_checks = min(n_checks, n_available)
    sample_idx = torch.randperm(n_available, generator=generator)[:n_checks]

    # Not every model passed here has parameters (test doubles like a
    # hand-written oracle may not) -- fall back to CPU rather than crash.
    params = list(model.parameters())
    model_device = params[0].device if params else torch.device("cpu")

    def predict_fn(input_features: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(input_features.to(model_device), control.to(model_device)).cpu()

    results = []
    for i in sample_idx.tolist():
        b, t = pointer_positions[i].tolist()
        substitute = torch.rand(batch.input_features.shape[-1], generator=generator)
        results.append(
            patch_and_verify(batch, predict_fn, batch_index=b, pointer_time=t, substitute_value=substitute)
        )

    n = len(results)
    mean_dist_to_substitute_before = sum(r["dist_to_substitute_before"] for r in results) / n
    mean_dist_to_substitute_after = sum(r["dist_to_substitute_after"] for r in results) / n

    # The boolean moved_toward_substitute_fraction can read 1.0 even for a
    # negligible effect (any infinitesimal movement in the right direction
    # satisfies a strict inequality) -- found for real when causal-checking
    # round-1's near-chance linear_attention/delta_net/ssm checkpoints,
    # which all showed "100% moved toward substitute" despite barely-there
    # actual movement. This normalizes the movement by the starting
    # distance so a real effect (attention: ~92%) is distinguishable from a
    # technically-positive but negligible one (ssm: ~0.2%).
    if mean_dist_to_substitute_before > 0:
        relative_movement = (
            mean_dist_to_substitute_before - mean_dist_to_substitute_after
        ) / mean_dist_to_substitute_before
    else:
        relative_movement = float("nan")

    return {
        "n_checks": n,
        "moved_toward_substitute_fraction": sum(r["moved_toward_substitute"] for r in results) / n,
        "moved_away_from_original_fraction": sum(r["moved_away_from_original"] for r in results) / n,
        "mean_dist_to_substitute_before": mean_dist_to_substitute_before,
        "mean_dist_to_substitute_after": mean_dist_to_substitute_after,
        "mean_dist_to_original_before": sum(r["dist_to_original_before"] for r in results) / n,
        "mean_dist_to_original_after": sum(r["dist_to_original_after"] for r in results) / n,
        "relative_movement_toward_substitute": relative_movement,
    }
