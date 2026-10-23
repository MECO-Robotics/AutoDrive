"""Neural contracts only; these tests do not validate simulated game behavior."""

import pytest
import torch

from frc_defense.architecture_entities import DeepSetsPolicy, EntityAttentionPolicy


@pytest.mark.parametrize("policy_type", [DeepSetsPolicy, EntityAttentionPolicy])
def test_entity_order_and_padding_do_not_change_policy(policy_type):
    torch.manual_seed(17)
    policy = policy_type()
    inputs = {
        "scalars": torch.randn(3, 137),
        "entities": torch.randn(3, 7, 12),
        "entity_mask": torch.tensor([
            [True, True, False, False, False, False, False],
            [True, True, True, True, True, True, True],
            [False, False, False, False, False, False, False],
        ]),
    }
    logits, value = policy(inputs)
    permutation = torch.tensor([6, 0, 3, 1, 4, 5, 2])
    changed = dict(inputs)
    changed["entities"] = inputs["entities"][:, permutation].clone()
    changed["entity_mask"] = inputs["entity_mask"][:, permutation]
    changed["entities"][~changed["entity_mask"]] = float("nan")
    reordered_logits, reordered_value = policy(changed)
    torch.testing.assert_close(logits, reordered_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(value, reordered_value, atol=1e-6, rtol=1e-5)
    assert logits.shape == (3, 8)
    assert value.shape == (3,)
    (logits.square().mean() + value.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())


@pytest.mark.parametrize("policy_type", [DeepSetsPolicy, EntityAttentionPolicy])
def test_empty_set_is_finite_and_valid_tracks_affect_output(policy_type):
    torch.manual_seed(18)
    policy = policy_type()
    scalars = torch.randn(2, 137)
    empty = {
        "scalars": scalars,
        "entities": torch.empty(2, 0, 12),
        "entity_mask": torch.empty(2, 0, dtype=torch.bool),
    }
    logits, value = policy(empty)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    (logits.square().mean() + value.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    observed = {
        "scalars": scalars,
        "entities": torch.randn(2, 3, 12),
        "entity_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    observed_logits, _ = policy(observed)
    assert not torch.allclose(logits, observed_logits)
