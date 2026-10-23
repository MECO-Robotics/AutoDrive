"""Network contracts only; these tests do not validate simulator behavior."""
import pytest
import torch

from frc_defense.architecture_dense import BaselineMLP, FieldFNOPolicy, ResidualMLP
from frc_defense.tensor_training_runtime import ActorCritic


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(73)
    return {
        "scalars": torch.randn(2, 137, generator=generator),
        "field": torch.randn(2, 12, 32, 64, generator=generator),
        "entities": torch.randn(2, 16, 12, generator=generator),
        "entity_mask": torch.ones(2, 16, dtype=torch.bool),
    }


@pytest.mark.parametrize("policy_class", [BaselineMLP, ResidualMLP, FieldFNOPolicy])
def test_forward_backward_finite(policy_class, inputs):
    torch.manual_seed(17)
    model = policy_class()
    logits, value = model(inputs)
    assert logits.shape == (2, 8)
    assert value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    loss = logits.square().mean() + value.square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer.step()
    logits, value = model(inputs)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_baseline_matches_existing_policy(inputs):
    torch.manual_seed(29)
    existing = ActorCritic(137, 8, action_kind="categorical")
    candidate = BaselineMLP()
    candidate.load_state_dict(existing.state_dict())
    expected_logits, _, expected_value = existing(inputs["scalars"])
    logits, value = candidate(inputs)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(value, expected_value, rtol=0, atol=0)


def test_fourier_policy_uses_field_and_has_spectral_gradients(inputs):
    torch.manual_seed(41)
    model = FieldFNOPolicy()
    inputs["field"].requires_grad_()
    logits, value = model(inputs)
    (logits[:, 0].sum() + value.sum()).backward()
    assert inputs["field"].grad.abs().sum() > 0
    for spectral in model.spectral:
        assert spectral.weights.grad.abs().sum() > 0
    changed = dict(inputs, field=torch.zeros_like(inputs["field"]))
    changed_logits, changed_value = model(changed)
    assert not torch.allclose(logits, changed_logits)
    assert not torch.allclose(value, changed_value)
