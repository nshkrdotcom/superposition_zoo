"""Frozen random-binding (VSA-style) mixing.

The strongest test of "legible by construction" in the zoo (doc 4 §7's
optional stretch goal #6): binding/unbinding uses fixed, never-trained
Rademacher (+-1) role codes, one per absolute position, so there is no
learned routing decision anywhere in the mixing step itself -- only a
learned content projection in and a learned readout out.

Mechanism: bind each position's (learned) content with its own frozen code
(elementwise product), bundle causally via cumulative sum (a literal
superposition of every position's binding so far, in the vector-symbolic-
architecture sense), then unbind at each position with its own code.
Rademacher codes are self-inverse under elementwise multiplication
(code * code = 1 everywhere), so a position's own contribution is recovered
exactly; every other position's contribution becomes cross-talk that does
not cancel (interference, scaling down as `d_model` grows -- the classic
VSA capacity/interference tradeoff, and a direct instance of the
superposition phenomenon this whole project studies).

Read this primitive's limitation honestly rather than hide it: this scheme
has no content-comparison operation at all. It can bind and recover
*its own* position's content, and it can be *read out* by a learned
projection that might exploit whatever partial signal survives in the
cross-talk -- but it cannot search for "the earlier position whose key
matches mine" the way `recall_task.py`'s content-addressed design requires,
because nothing here compares two vectors for similarity (that is what
attention's QK dot product provides and this primitive deliberately does
not have). Tested and reported as such, not assumed -- see
`tests/test_vsa_binding.py` and `EXPERIMENT_LOG.md` for what actually
happens when this is trained on the recall benchmark.
"""

from __future__ import annotations

import torch
from torch import nn

from superposition_zoo.mixing.base import MixingPrimitive


def causal_bind_bundle_unbind(content: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Bind, causally bundle, and unbind -- the VSA core, factored out for testing.

    Args:
        content: ``(batch, seq_len, d_model)``.
        codes: ``(seq_len, d_model)``, entries in ``{-1, +1}``.

    Returns:
        ``(batch, seq_len, d_model)``.
    """
    bound = content * codes.unsqueeze(0)
    memory = torch.cumsum(bound, dim=1)
    return memory * codes.unsqueeze(0)


class VSABinding(MixingPrimitive):
    def __init__(self, d_model: int, max_seq_len: int = 2048, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Frozen: generated once from a fixed, training-independent seed and
        # registered as a buffer (not a Parameter), so it is identical across
        # every instance and every run and is never touched by the optimizer.
        code_generator = torch.Generator().manual_seed(0)
        role_codes = torch.randint(0, 2, (max_seq_len, d_model), generator=code_generator).float() * 2 - 1
        self.register_buffer("role_codes", role_codes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        content = self.value_proj(x)
        codes = self.role_codes[:seq_len]
        retrieved = causal_bind_bundle_unbind(content, codes)
        return self.out_proj(retrieved)
