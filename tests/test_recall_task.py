from __future__ import annotations

import pytest
import torch

from superposition_zoo.recall_task import generate_recall_batch


def test_shapes():
    batch = generate_recall_batch(
        batch_size=4, seq_len=32, n_features=10, n_pointers=5, sparsity=0.6, generator=_gen()
    )
    assert batch.input_features.shape == (4, 32, 10)
    assert batch.control.shape == (4, 32, 2)
    assert batch.target.shape == (4, 32, 10)
    assert batch.is_pointer.shape == (4, 32)
    assert batch.is_pointer.dtype == torch.bool
    assert batch.source_position.shape == (4, 32)
    assert batch.active_mask.shape == (4, 32, 10)
    assert batch.active_mask.dtype == torch.bool


def test_content_positions_reconstruct_themselves():
    batch = generate_recall_batch(
        batch_size=8, seq_len=40, n_features=12, n_pointers=8, sparsity=0.5, generator=_gen()
    )
    content_mask = ~batch.is_pointer
    assert torch.equal(batch.target[content_mask], batch.input_features[content_mask])


def test_pointer_positions_have_zeroed_local_input():
    batch = generate_recall_batch(
        batch_size=8, seq_len=40, n_features=12, n_pointers=8, sparsity=0.5, generator=_gen()
    )
    pointer_mask = batch.is_pointer
    assert torch.all(batch.input_features[pointer_mask] == 0.0)


def test_pointer_target_exactly_equals_source_positions_input():
    batch = generate_recall_batch(
        batch_size=8, seq_len=40, n_features=12, n_pointers=8, sparsity=0.5, generator=_gen()
    )
    for b in range(8):
        for t in range(40):
            if batch.is_pointer[b, t]:
                s = batch.source_position[b, t].item()
                assert torch.equal(batch.target[b, t], batch.input_features[b, s])


def test_source_position_is_strictly_causal():
    batch = generate_recall_batch(
        batch_size=8, seq_len=40, n_features=12, n_pointers=8, sparsity=0.5, min_gap=2, generator=_gen()
    )
    for b in range(8):
        for t in range(40):
            if batch.is_pointer[b, t]:
                s = batch.source_position[b, t].item()
                assert s <= t - 2


def test_sources_are_never_themselves_pointers():
    batch = generate_recall_batch(
        batch_size=8, seq_len=40, n_features=12, n_pointers=10, sparsity=0.5, generator=_gen()
    )
    for b in range(8):
        for t in range(40):
            if batch.is_pointer[b, t]:
                s = batch.source_position[b, t].item()
                assert not batch.is_pointer[b, s].item()


def test_pointer_count_matches_request_with_generous_params():
    batch = generate_recall_batch(
        batch_size=16, seq_len=64, n_features=10, n_pointers=6, sparsity=0.5, generator=_gen()
    )
    assert torch.all(batch.is_pointer.sum(dim=1) == 6)


def test_non_pointer_positions_have_negative_one_sentinel_source():
    batch = generate_recall_batch(
        batch_size=8, seq_len=40, n_features=12, n_pointers=8, sparsity=0.5, generator=_gen()
    )
    assert torch.all(batch.source_position[~batch.is_pointer] == -1)


def test_control_channel_encodes_pointer_flag_and_normalized_offset():
    seq_len = 40
    batch = generate_recall_batch(
        batch_size=8, seq_len=seq_len, n_features=12, n_pointers=8, sparsity=0.5, generator=_gen()
    )
    assert torch.equal(batch.control[..., 0], batch.is_pointer.float())
    assert torch.all(batch.control[~batch.is_pointer][..., 1] == 0.0)
    for b in range(8):
        for t in range(seq_len):
            if batch.is_pointer[b, t]:
                s = batch.source_position[b, t].item()
                expected_offset = (t - s) / seq_len
                assert batch.control[b, t, 1].item() == pytest.approx(expected_offset)


def test_deterministic_given_same_seed():
    gen1 = torch.Generator()
    gen1.manual_seed(7)
    gen2 = torch.Generator()
    gen2.manual_seed(7)
    b1 = generate_recall_batch(
        batch_size=4, seq_len=20, n_features=8, n_pointers=3, sparsity=0.5, generator=gen1
    )
    b2 = generate_recall_batch(
        batch_size=4, seq_len=20, n_features=8, n_pointers=3, sparsity=0.5, generator=gen2
    )
    assert torch.equal(b1.input_features, b2.input_features)
    assert torch.equal(b1.target, b2.target)
    assert torch.equal(b1.source_position, b2.source_position)


def test_rejects_negative_n_pointers():
    with pytest.raises(ValueError):
        generate_recall_batch(
            batch_size=2, seq_len=20, n_features=8, n_pointers=-1, sparsity=0.5, generator=_gen()
        )


def test_rejects_too_many_pointers_for_seq_len():
    with pytest.raises(ValueError):
        generate_recall_batch(
            batch_size=2, seq_len=5, n_features=8, n_pointers=10, sparsity=0.5, generator=_gen()
        )


def test_rejects_min_gap_below_one():
    with pytest.raises(ValueError):
        generate_recall_batch(
            batch_size=2, seq_len=20, n_features=8, n_pointers=3, sparsity=0.5, min_gap=0, generator=_gen()
        )


def _gen() -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(999)
    return generator
