from __future__ import annotations

import pytest


def test_vsa_binding_is_an_honest_not_implemented_stub():
    from superposition_zoo.mixing import REGISTRY

    with pytest.raises(NotImplementedError, match="vsa_binding"):
        REGISTRY.build("vsa_binding", d_model=16)


def test_vsa_binding_is_listed_as_a_stub_not_silently_missing():
    from superposition_zoo.mixing.vsa_binding import VSABinding

    with pytest.raises(NotImplementedError):
        VSABinding(d_model=16)
