"""The zoo: every mixing primitive registers itself here on import.

``REGISTRY`` is the single shared instance used by real training code (as
opposed to the isolated per-test ``MixingRegistry()`` instances used in
``tests/test_mixing_base.py``).
"""

from __future__ import annotations

from superposition_zoo.mixing.base import MixingPrimitive, MixingRegistry
from superposition_zoo.mixing.delta_net import DeltaNet
from superposition_zoo.mixing.hard_routing import HardRoutingAttention
from superposition_zoo.mixing.linear_attention import LinearAttention
from superposition_zoo.mixing.ssm import MinimalSSM
from superposition_zoo.mixing.standard_attention import StandardAttention
from superposition_zoo.mixing.vsa_binding import VSABinding

REGISTRY = MixingRegistry()
REGISTRY.register("standard_attention", StandardAttention)
REGISTRY.register("linear_attention", LinearAttention)
REGISTRY.register("delta_net", DeltaNet)
REGISTRY.register("hard_routing", HardRoutingAttention)
REGISTRY.register("ssm", MinimalSSM)
REGISTRY.register("vsa_binding", VSABinding)

__all__ = ["REGISTRY", "MixingPrimitive", "MixingRegistry"]
