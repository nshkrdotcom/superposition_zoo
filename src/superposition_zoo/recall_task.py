"""Cross-token superposition/recall benchmark (Phase 1).

Extends the Phase 0 sparse-feature setup (see ``features.py``) with a
sequence dimension and explicit "pointer" positions that must retrieve an
earlier position's feature combination. This is the benchmark that actually
requires a sequence-mixing primitive to do real work: a purely local
(per-position) processor cannot solve it, because the correct output at a
pointer position depends on content that lives at a different position.

This is a *content-addressed* pointer/copy task, matching the standard
multi-query associative-recall (MQAR) design used to compare sequence-
mixing primitives in the Zoology/Based line of work (Arora et al.): every
position carries a random "key" vector, and a pointer's key is set to
*exactly match* its true source position's key. Retrieval is "find the
earlier position whose key matches mine," which is precisely the kind of
content-similarity matching softmax/QK attention is naturally suited to
learn -- unlike this benchmark's first version, which encoded the source as
a continuous relative-offset scalar and required the model to perform
positional arithmetic from it. That version turned out to be architecturally
much harder to learn (verified by actually training standard attention on
it -- recall accuracy stayed near zero even after fixing a separate,
necessary positional-encoding bug), which is exactly the kind of thing a
real training run catches that a unit test alone does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from superposition_zoo.features import generate_features

KEY_DIM = 8
"""Dimensionality of the per-position random key vector."""

CONTROL_CHANNELS = 1 + KEY_DIM
"""``[is_pointer_flag, key_0, ..., key_{KEY_DIM-1}]``."""


@dataclass
class RecallBatch:
    """Ground truth for one batch of the cross-token recall benchmark.

    All fields have a leading ``(batch, seq_len, ...)`` shape.
    """

    input_features: torch.Tensor
    """``(B, T, F)``: local content, zeroed at pointer positions."""

    control: torch.Tensor
    """``(B, T, CONTROL_CHANNELS)``: ``[is_pointer_flag, key...]`` per position.
    A pointer's key is set to exactly match its true source position's key."""

    target: torch.Tensor
    """``(B, T, F)``: what the model should reconstruct at every position."""

    is_pointer: torch.Tensor
    """``(B, T)`` bool."""

    source_position: torch.Tensor
    """``(B, T)`` long; ``-1`` sentinel where ``is_pointer`` is ``False``."""

    active_mask: torch.Tensor
    """``(B, T, F)`` bool: ground truth of which features are active in ``target``."""


def generate_recall_batch(
    batch_size: int,
    seq_len: int,
    n_features: int,
    n_pointers: int,
    sparsity: float | torch.Tensor,
    generator: torch.Generator,
    min_gap: int = 1,
) -> RecallBatch:
    """Generate a batch of cross-token superposition/recall sequences.

    Args:
        batch_size: number of independent sequences.
        seq_len: length of each sequence.
        n_features: number of ground-truth features (see ``features.py``).
        n_pointers: number of pointer (recall) events per sequence.
        sparsity: passed through to :func:`generate_features`.
        generator: an explicitly-seeded ``torch.Generator``.
        min_gap: minimum distance between a pointer and the earliest position
            it may source from; also enforces strict causality (a pointer can
            never source from itself or the future).

    Returns:
        A :class:`RecallBatch`.

    Note:
        In rare configurations (very high ``n_pointers`` relative to
        ``seq_len``) a chosen pointer position may have no valid,
        non-pointer earlier source and is silently left as a content
        position instead. Callers who need an exact pointer count should
        choose generous ``seq_len``/``n_pointers``/``min_gap`` combinations,
        as the tests in this repo do.
    """
    if n_pointers < 0:
        raise ValueError(f"n_pointers must be >= 0, got {n_pointers}")
    if min_gap < 1:
        raise ValueError(f"min_gap must be >= 1, got {min_gap}")
    max_possible_pointers = max(0, seq_len - min_gap)
    if n_pointers > max_possible_pointers:
        raise ValueError(
            f"n_pointers={n_pointers} exceeds max_possible_pointers={max_possible_pointers} "
            f"for seq_len={seq_len} and min_gap={min_gap}"
        )

    content_values, content_active = generate_features(
        n_features=n_features,
        batch_size=batch_size * seq_len,
        sparsity=sparsity,
        generator=generator,
    )
    content_values = content_values.view(batch_size, seq_len, n_features)
    content_active = content_active.view(batch_size, seq_len, n_features)

    input_features = content_values.clone()
    target = content_values.clone()
    active_mask = content_active.clone()
    is_pointer = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    source_position = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    key = torch.randn(batch_size, seq_len, KEY_DIM, generator=generator)

    if n_pointers > 0:
        eligible = torch.arange(min_gap, seq_len)
        n_eligible = eligible.numel()

        # Pick n_pointers distinct positions per row via random priority + topk
        # (equivalent to sampling without replacement, fully vectorized across
        # the batch instead of a per-row torch.randperm call).
        priorities = torch.rand(batch_size, n_eligible, generator=generator)
        _, top_idx = torch.topk(priorities, n_pointers, dim=1)
        chosen = eligible[top_idx]  # (batch_size, n_pointers) absolute positions

        # A position's pointer status depends only on which positions were
        # chosen, not on the order sources get resolved in (any earlier
        # pointer position that could act as a source is already known here,
        # regardless of processing order) -- so every pointer flag can be set
        # up front, before resolving any sources.
        is_pointer.scatter_(1, chosen, True)

        # Validity mask for every (row, pointer, candidate-source) triple:
        # causally earlier than min_gap, and not itself a pointer.
        positions = torch.arange(seq_len).view(1, 1, seq_len)
        causal_ok = positions <= (chosen.unsqueeze(-1) - min_gap)
        not_pointer = (~is_pointer).unsqueeze(1)
        valid = causal_ok & not_pointer  # (batch_size, n_pointers, seq_len)
        has_valid_source = valid.any(dim=-1)

        # Uniform-among-valid-candidates sampling via the random-priority /
        # argmax trick: assigning iid continuous priorities and taking the
        # argmax among valid entries is equivalent to sampling uniformly
        # among them, without needing a per-triple Python-level scan.
        source_priorities = torch.rand(batch_size, n_pointers, seq_len, generator=generator)
        source_priorities = source_priorities.masked_fill(~valid, -1.0)
        chosen_source = source_priorities.argmax(dim=-1)  # (batch_size, n_pointers)

        # Only a small, O(batch_size * n_pointers) loop remains (vs. the
        # original O(batch_size * n_pointers * seq_len) candidate scan) --
        # each iteration does O(1) tensor writes, no scanning.
        for b in range(batch_size):
            for i in range(n_pointers):
                t = int(chosen[b, i])
                if not has_valid_source[b, i]:
                    is_pointer[b, t] = False  # no valid source: revert to a content position
                    continue
                s = int(chosen_source[b, i])
                source_position[b, t] = s
                input_features[b, t] = 0.0
                target[b, t] = content_values[b, s]
                active_mask[b, t] = content_active[b, s]
                key[b, t] = key[b, s]  # exact key match: this is the retrieval signal

    control = torch.cat([is_pointer.float().unsqueeze(-1), key], dim=-1)

    return RecallBatch(
        input_features=input_features,
        control=control,
        target=target,
        is_pointer=is_pointer,
        source_position=source_position,
        active_mask=active_mask,
    )
