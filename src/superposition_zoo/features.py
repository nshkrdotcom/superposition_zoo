"""Ground-truth sparse feature generation (Phase 0).

Every downstream metric depends on this module being correct: the whole
point of the benchmark is that we know the true features by construction,
so there is nothing to infer here, only to generate faithfully.
"""

from __future__ import annotations

import torch


def _validate_sparsity(sparsity: float | torch.Tensor, n_features: int) -> torch.Tensor:
    if isinstance(sparsity, torch.Tensor):
        if sparsity.shape != (n_features,):
            raise ValueError(
                f"sparsity tensor must have shape ({n_features},), got {tuple(sparsity.shape)}"
            )
        if torch.any(sparsity < 0.0) or torch.any(sparsity > 1.0):
            raise ValueError("sparsity values must all lie in [0, 1]")
        return sparsity
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    return torch.full((n_features,), float(sparsity))


def generate_features(
    n_features: int,
    batch_size: int,
    sparsity: float | torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ground-truth sparse feature vectors.

    Each feature ``i`` is independently active with probability
    ``1 - sparsity[i]``; when active its magnitude is drawn from
    ``Uniform(0, 1)``, and it is exactly ``0.0`` otherwise.

    Args:
        n_features: number of ground-truth features.
        batch_size: number of independent examples to generate.
        sparsity: scalar in ``[0, 1]`` applied to every feature, or a
            ``(n_features,)`` tensor giving a per-feature sparsity.
        generator: an explicitly-seeded ``torch.Generator``. Required (not
            optional) so every call site is deterministic by construction.

    Returns:
        ``(values, active_mask)`` where ``values`` has shape
        ``(batch_size, n_features)`` and ``active_mask`` is the boolean
        ground truth of which features are active.
    """
    per_feature_sparsity = _validate_sparsity(sparsity, n_features)
    density = (1.0 - per_feature_sparsity).unsqueeze(0).expand(batch_size, n_features)

    draws = torch.rand(batch_size, n_features, generator=generator)
    active_mask = draws < density

    magnitudes = torch.rand(batch_size, n_features, generator=generator)
    values = magnitudes * active_mask

    return values, active_mask
