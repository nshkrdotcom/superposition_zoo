"""The mixing-primitive contract and registry.

Every entry in the zoo (doc 4/5: standard attention, linear attention,
DeltaNet, a minimal SSM, hard top-1 routing, ...) is a drop-in replacement
for "the block that mixes information across sequence positions": same
``forward(x: [B, T, D]) -> [B, T, D]`` contract, same strict-causality
requirement, same ``d_model``. This is what makes matched-budget, apples-to-
apples comparison across structurally different primitives possible at all.
"""

from __future__ import annotations

from torch import nn


class MixingPrimitive(nn.Module):
    """Base class every mixing primitive subclasses.

    Contract: ``forward(x)`` takes and returns a ``(batch, seq_len, d_model)``
    tensor, and must be strictly causal (output at position ``t`` may never
    depend on input at position ``> t``). See ``tests/mixing_helpers.py`` for
    the shared test contract every primitive is checked against.
    """

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MixingRegistry:
    """A small, explicit name -> constructor registry.

    Mirrors ``attention_lab``'s ``build_attention()`` string-dispatch
    pattern (the one piece of that codebase's design worth reusing) but as
    an instantiable class rather than a module-level singleton, so tests can
    construct their own isolated registry instead of mutating shared global
    state.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[MixingPrimitive]] = {}

    def register(self, name: str, cls: type[MixingPrimitive]) -> None:
        if name in self._registry:
            raise ValueError(f"mixing primitive '{name}' is already registered")
        self._registry[name] = cls

    def build(self, name: str, d_model: int, **kwargs) -> MixingPrimitive:
        if name not in self._registry:
            raise NotImplementedError(
                f"mixing primitive '{name}' is not implemented or not yet registered. "
                f"Available: {self.available()}"
            )
        return self._registry[name](d_model=d_model, **kwargs)

    def available(self) -> list[str]:
        return sorted(self._registry)
