"""Network contracts only; these checks do not validate simulator behavior."""
import pytest
import torch

from frc_defense.architecture_cnn import FullFieldCNN, FullFieldResNet


@pytest.mark.parametrize("network", (FullFieldCNN, FullFieldResNet))
@pytest.mark.parametrize("empty", (False, True))
def test_full_field_network_forward_backward(network, empty):
    torch.manual_seed(728)
    model = network()
    field = torch.zeros(2, 12, 32, 64) if empty else torch.randn(2, 12, 32, 64)
    field.requires_grad_(True)
    scalars = torch.randn(2, 137, requires_grad=True)
    logits, value = model({"field": field, "scalars": scalars})
    assert logits.shape == (2, 8)
    assert value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    (logits.square().mean() + value.square().mean()).backward()
    assert torch.isfinite(field.grad).all()
    assert torch.isfinite(scalars.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("network", (FullFieldCNN, FullFieldResNet))
def test_full_field_network_retains_absolute_location(network):
    torch.manual_seed(728)
    model = network()
    field = torch.zeros(2, 12, 32, 64)
    field[0, 0, 8, 8] = 1.0
    field[1, 0, 8, 48] = 1.0
    logits, value = model({"field": field, "scalars": torch.zeros(2, 137)})
    assert not torch.allclose(logits[0], logits[1], atol=1e-8, rtol=0)
