"""Guardrails for the full HIP strategic-observation path."""
from __future__ import annotations

import importlib

import torch

from frc_defense import tensor_sim
from frc_defense import tensor_strategic_observation_full_proto as full_proto


def test_default_environment_setting_enables_full_assembler(monkeypatch):
    monkeypatch.delenv("AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_FULL_HIP", raising=False)
    module = importlib.reload(full_proto)
    assert module.FUSED_FULL_STRATEGIC_OBSERVATION_HIP_ENABLED is True


def test_full_assembler_can_be_disabled_and_cpu_falls_back(monkeypatch):
    # Force device availability so this checks the flag itself, independent of
    # whether the test host has HIP hardware.
    monkeypatch.setattr(full_proto, "available", lambda device: True)
    monkeypatch.setattr(full_proto,
        "FUSED_FULL_STRATEGIC_OBSERVATION_HIP_ENABLED", False)
    assert not full_proto.integrated_available("cuda:0")
    monkeypatch.setattr(full_proto,
        "FUSED_FULL_STRATEGIC_OBSERVATION_HIP_ENABLED", True)
    assert full_proto.integrated_available("cuda:0")
    monkeypatch.setattr(full_proto,
        "FUSED_FULL_STRATEGIC_OBSERVATION_HIP_ENABLED", False)

    env = tensor_sim.TensorDefenseEnv(num_envs=4, task="counter_defense",
        device="cpu", seed=811, action_mode="strategic", randomize=False,
        reuse_strategic_own_candidates=True)
    expected = env._strategic_observation(0)

    def unexpected_full_path(*args, **kwargs):
        raise AssertionError("full HIP assembler must not run while disabled")

    monkeypatch.setattr(tensor_sim, "_full_strategic_observation", unexpected_full_path)
    actual = env._strategic_observation(0)
    assert torch.equal(expected, actual)


def test_candidate_cache_active_rows_are_shared_by_observation_paths():
    env = tensor_sim.TensorDefenseEnv(num_envs=6, task="counter_defense",
        device="cpu", seed=812, action_mode="strategic", randomize=False,
        reuse_strategic_own_candidates=True)
    active = torch.tensor((True, False, True, False, True, False), dtype=torch.bool)
    _, current_data, current_mask = env._strategic_observation(
        0, return_candidate_data=True, active_mask=active)
    pending = env._pending_strategic_own_candidates
    assert pending is not None
    for got, now in zip(pending[0], current_data):
        assert torch.equal(got[active], now[active])
        assert torch.equal(got[~active], torch.zeros_like(got[~active]))
    assert torch.equal(pending[1][active], current_mask[active])
    assert not bool(pending[1][~active].any())
