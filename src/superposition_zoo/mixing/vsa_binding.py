"""Frozen random-binding (VSA-style) mixing -- intentionally unimplemented.

Doc 4 §7's optional stretch goal #6: the strongest possible test of the
"legible by construction" hypothesis, since binding/unbinding via fixed
random (e.g. Rademacher) role codes gives no learned routing to discover at
all. Registered but unimplemented, following the same precedent as
``attention_lab``'s own ``trilinear_cp`` placeholder: honest incompleteness
(doc 5 §2 principle 6) rather than a stub that silently looks real. Use
``standard_attention``, ``linear_attention``, ``delta_net``, ``hard_routing``,
or ``ssm`` until this is designed and implemented.
"""

from __future__ import annotations

from superposition_zoo.mixing.base import MixingPrimitive


class VSABinding(MixingPrimitive):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "vsa_binding is an intentionally unimplemented stretch goal. Frozen "
            "random-binding mixing needs a binding/unbinding scheme (e.g. circular "
            "convolution or elementwise product with fixed Rademacher role codes) that is "
            "not yet designed. Available implemented primitives: standard_attention, "
            "linear_attention, delta_net, hard_routing, ssm."
        )
