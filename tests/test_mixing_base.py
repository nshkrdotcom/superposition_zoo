from __future__ import annotations

import pytest
import torch
from torch import nn

from superposition_zoo.mixing.base import MixingPrimitive, MixingRegistry


class _DummyPrimitive(MixingPrimitive):
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_build_returns_instance_of_registered_class():
    registry = MixingRegistry()
    registry.register("dummy", _DummyPrimitive)
    instance = registry.build("dummy", d_model=8)
    assert isinstance(instance, _DummyPrimitive)
    assert isinstance(instance, MixingPrimitive)


def test_available_lists_registered_names_sorted():
    registry = MixingRegistry()
    registry.register("zeta", _DummyPrimitive)
    registry.register("alpha", _DummyPrimitive)
    assert registry.available() == ["alpha", "zeta"]


def test_duplicate_registration_raises():
    registry = MixingRegistry()
    registry.register("dummy", _DummyPrimitive)
    with pytest.raises(ValueError):
        registry.register("dummy", _DummyPrimitive)


def test_unknown_primitive_raises_not_implemented_with_available_list():
    registry = MixingRegistry()
    registry.register("dummy", _DummyPrimitive)
    with pytest.raises(NotImplementedError) as excinfo:
        registry.build("does_not_exist", d_model=8)
    assert "dummy" in str(excinfo.value)


def test_registries_are_independent_instances():
    registry_a = MixingRegistry()
    registry_b = MixingRegistry()
    registry_a.register("dummy", _DummyPrimitive)
    assert registry_a.available() == ["dummy"]
    assert registry_b.available() == []


def test_num_parameters_default_implementation():
    registry = MixingRegistry()
    registry.register("dummy", _DummyPrimitive)
    instance = registry.build("dummy", d_model=4)
    # nn.Linear(4, 4): weight (4*4) + bias (4) = 20
    assert instance.num_parameters() == 20


def test_real_registry_is_populated_on_import():
    from superposition_zoo.mixing import REGISTRY

    assert "standard_attention" in REGISTRY.available()
