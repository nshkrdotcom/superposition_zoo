from __future__ import annotations

import torch

from superposition_zoo.metrics.causal import patch_and_verify
from superposition_zoo.recall_task import generate_recall_batch


def _oracle_predict_fn(input_features: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """A hand-written 'perfect' predictor: resolves every pointer by finding
    the earlier position whose key exactly matches, exactly as an ideal
    content-addressed (MQAR-style) model would. Used to test the
    causal-verification *metric*, independent of whether any real trained
    model exists yet.
    """
    batch, seq_len, _ = input_features.shape
    output = input_features.clone()
    for b in range(batch):
        for t in range(seq_len):
            if control[b, t, 0] > 0.5:
                pointer_key = control[b, t, 1:]
                for s in range(t):
                    if control[b, s, 0] <= 0.5 and torch.equal(control[b, s, 1:], pointer_key):
                        output[b, t] = input_features[b, s]
                        break
    return output


def _identity_predict_fn(input_features: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """A predictor that ignores pointers entirely -- a stand-in for a model
    with no real cross-token mixing, used as the negative-control case.
    """
    return input_features.clone()


def _batch_with_a_pointer():
    generator = torch.Generator()
    generator.manual_seed(0)
    batch = generate_recall_batch(
        batch_size=2, seq_len=16, n_features=6, n_pointers=3, sparsity=0.4, min_gap=2, generator=generator
    )
    b, t = (batch.is_pointer.nonzero()[0]).tolist()
    return batch, b, t


def test_oracle_predictor_shows_full_causal_effect():
    batch, b, t = _batch_with_a_pointer()
    substitute = torch.rand(batch.input_features.shape[-1])

    result = patch_and_verify(
        batch, _oracle_predict_fn, batch_index=b, pointer_time=t, substitute_value=substitute
    )

    assert result["moved_toward_substitute"] is True
    assert result["moved_away_from_original"] is True
    assert result["dist_to_substitute_after"] < 1e-5
    assert result["dist_to_original_before"] < 1e-5


def test_identity_predictor_shows_no_causal_effect():
    batch, b, t = _batch_with_a_pointer()
    substitute = torch.rand(batch.input_features.shape[-1])

    result = patch_and_verify(
        batch, _identity_predict_fn, batch_index=b, pointer_time=t, substitute_value=substitute
    )

    assert result["moved_toward_substitute"] is False
    # identity predictor's output at the (zeroed) pointer position is
    # unaffected by patching an earlier position at all
    assert result["dist_to_substitute_before"] == result["dist_to_substitute_after"]


def test_raises_if_requested_position_is_not_actually_a_pointer():
    batch, b, t = _batch_with_a_pointer()
    non_pointer_t = int((~batch.is_pointer[b]).nonzero()[0].item())
    substitute = torch.rand(batch.input_features.shape[-1])
    try:
        patch_and_verify(
            batch, _oracle_predict_fn, batch_index=b, pointer_time=non_pointer_t, substitute_value=substitute
        )
        raised = False
    except AssertionError:
        raised = True
    assert raised
