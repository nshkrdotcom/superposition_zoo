"""Ground-truth activation patching for the recall benchmark (Phase 1).

This is the direct replacement for the old ``attention_lab`` Tier-1 patching/
restoration machinery: because ``RecallBatch.source_position`` is known
ground truth (we generated it), verifying that a model's retrieval is
causally driven by the true source position -- rather than a coincidental
correlation -- is a direct measurement (swap the source, check the output
followed the swap) instead of an inference requiring null families, matched
controls, and FDR correction.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from superposition_zoo.recall_task import RecallBatch

PredictFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def patch_and_verify(
    batch: RecallBatch,
    predict_fn: PredictFn,
    batch_index: int,
    pointer_time: int,
    substitute_value: torch.Tensor,
) -> dict[str, float | bool]:
    """Substitute the true source position's content and check the effect.

    Args:
        batch: a :class:`RecallBatch` (ground truth for pointer resolution).
        predict_fn: a callable ``(input_features, control) -> predicted``,
            both shaped ``(batch, seq_len, n_features)``. This is
            deliberately not a specific model class, so the metric can be
            unit-tested against a hand-written oracle/negative-control
            predictor, independent of any real trained model.
        batch_index: which batch element to patch.
        pointer_time: which pointer position to inspect; must satisfy
            ``batch.is_pointer[batch_index, pointer_time]``.
        substitute_value: ``(n_features,)`` replacement content for the true
            source position.

    Returns:
        A dict of distances before/after patching and two booleans:
        ``moved_toward_substitute`` and ``moved_away_from_original``. A
        model with a real, causally load-bearing retrieval mechanism should
        show both as ``True``; a model that ignores the source position
        entirely (or reads from the wrong place) will not.
    """
    assert bool(batch.is_pointer[batch_index, pointer_time]), (
        f"position {pointer_time} in batch element {batch_index} is not a pointer position"
    )
    source = int(batch.source_position[batch_index, pointer_time].item())
    original_source_content = batch.input_features[batch_index, source].clone()

    original_predicted = predict_fn(batch.input_features, batch.control)
    original_pred_at_pointer = original_predicted[batch_index, pointer_time]

    patched_input = batch.input_features.clone()
    patched_input[batch_index, source] = substitute_value
    patched_predicted = predict_fn(patched_input, batch.control)
    patched_pred_at_pointer = patched_predicted[batch_index, pointer_time]

    dist_to_original_before = torch.norm(original_pred_at_pointer - original_source_content).item()
    dist_to_substitute_before = torch.norm(original_pred_at_pointer - substitute_value).item()
    dist_to_original_after = torch.norm(patched_pred_at_pointer - original_source_content).item()
    dist_to_substitute_after = torch.norm(patched_pred_at_pointer - substitute_value).item()

    return {
        "dist_to_original_before": dist_to_original_before,
        "dist_to_substitute_before": dist_to_substitute_before,
        "dist_to_original_after": dist_to_original_after,
        "dist_to_substitute_after": dist_to_substitute_after,
        "moved_toward_substitute": dist_to_substitute_after < dist_to_substitute_before,
        "moved_away_from_original": dist_to_original_after > dist_to_original_before,
    }
