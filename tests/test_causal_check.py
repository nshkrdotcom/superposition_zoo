from __future__ import annotations

import torch
from torch import nn

from superposition_zoo.causal_check import causal_check_report
from superposition_zoo.recall_task import generate_recall_batch


class _OracleModel(nn.Module):
    """Wraps the same 'find the earlier position with a matching key' oracle
    logic used in test_metrics_causal.py as an nn.Module, so
    causal_check_report (which expects a model, not a bare predict_fn) can
    be tested against a case with a known, full causal effect.
    """

    def forward(self, input_features: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
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


class _IdentityModel(nn.Module):
    def forward(self, input_features: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return input_features.clone()


def _eval_batch():
    generator = torch.Generator()
    generator.manual_seed(0)
    return generate_recall_batch(
        batch_size=8, seq_len=32, n_features=8, n_pointers=4, sparsity=0.4, min_gap=2, generator=generator
    )


class _WeakEffectModel(nn.Module):
    """A model with a technically-positive but tiny causal effect -- the
    real scenario found when causal-checking round-1's weak (near-chance)
    linear_attention/delta_net/ssm checkpoints: patching the source moved
    the output toward the substitute by a barely-there margin (e.g. mean
    distance 1.876 -> 1.872), satisfying the strict "moved_toward_substitute"
    boolean while the actual effect size was negligible. Used to test that
    the relative-movement metric (not just the boolean) can tell the two
    situations apart.
    """

    def forward(self, input_features: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = input_features.shape
        baseline = input_features.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        output = baseline.clone()
        for b in range(batch):
            for t in range(seq_len):
                if control[b, t, 0] > 0.5:
                    pointer_key = control[b, t, 1:]
                    for s in range(t):
                        if control[b, s, 0] <= 0.5 and torch.equal(control[b, s, 1:], pointer_key):
                            output[b, t] = 0.99 * baseline[b, t] + 0.01 * input_features[b, s]
                            break
        return output


def test_oracle_model_shows_strong_causal_effect():
    batch = _eval_batch()
    report = causal_check_report(
        _OracleModel(), batch, n_checks=10, generator=torch.Generator().manual_seed(1)
    )

    assert report["n_checks"] == 10
    assert report["moved_toward_substitute_fraction"] == 1.0
    assert report["moved_away_from_original_fraction"] == 1.0
    assert report["mean_dist_to_substitute_after"] < 1e-4
    assert report["relative_movement_toward_substitute"] > 0.9


def test_weak_effect_model_satisfies_the_boolean_but_shows_small_relative_movement():
    # Regression: a naive reader could see moved_toward_substitute_fraction
    # == 1.0 and conclude a real causal effect exists. The relative-movement
    # metric must reveal that this is a negligible effect, not a strong one,
    # even though the strict boolean is satisfied for every check.
    batch = _eval_batch()
    report = causal_check_report(
        _WeakEffectModel(), batch, n_checks=10, generator=torch.Generator().manual_seed(1)
    )
    assert report["moved_toward_substitute_fraction"] == 1.0
    assert report["relative_movement_toward_substitute"] < 0.1


def test_identity_model_shows_no_causal_effect():
    batch = _eval_batch()
    report = causal_check_report(
        _IdentityModel(), batch, n_checks=10, generator=torch.Generator().manual_seed(1)
    )

    assert report["moved_toward_substitute_fraction"] == 0.0
    assert report["mean_dist_to_substitute_before"] == report["mean_dist_to_substitute_after"]


def test_n_checks_is_capped_at_available_pointers():
    batch = _eval_batch()
    n_available = int(batch.is_pointer.sum().item())
    report = causal_check_report(
        _OracleModel(), batch, n_checks=n_available + 1000, generator=torch.Generator().manual_seed(1)
    )
    assert report["n_checks"] == n_available


def test_works_with_a_real_parameterized_model_not_just_zero_param_test_doubles():
    # Regression: an earlier version crashed with StopIteration when a
    # model had zero parameters (the oracle/identity test doubles above),
    # and separately shipped a device-mismatch bug (model on cuda, batch
    # generated on cpu) that only ever showed up on real GPU hardware, not
    # in this CPU-only test suite. This exercises the parameter-detection
    # path directly; the actual cross-device move is verified by running
    # `szoo causal-check --device cuda` for real, not by this test.
    from superposition_zoo.models.sequence_model import SequenceModel

    batch = _eval_batch()
    model = SequenceModel(n_features=8, d_model=16, mixing_primitive_name="standard_attention", n_layers=1)
    report = causal_check_report(model, batch, n_checks=4, generator=torch.Generator().manual_seed(0))
    assert report["n_checks"] == 4
    assert 0.0 <= report["moved_toward_substitute_fraction"] <= 1.0


def test_report_is_deterministic_given_same_generator_seed():
    batch = _eval_batch()
    report_a = causal_check_report(
        _OracleModel(), batch, n_checks=6, generator=torch.Generator().manual_seed(42)
    )
    report_b = causal_check_report(
        _OracleModel(), batch, n_checks=6, generator=torch.Generator().manual_seed(42)
    )
    assert report_a == report_b
