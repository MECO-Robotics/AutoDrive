"""Guardrails for optional HIP fuel-candidate distance/mask fusion."""
from __future__ import annotations

import pytest
import torch

from frc_defense import tensor_sim
from frc_defense import tensor_fuel_candidate_rank as candidate_rank


def _reference(points, robot, free):
    distance = (points - robot[:, None, :]).norm(dim=-1).masked_fill(~free, float("inf"))
    nearest, indices = distance.topk(4, dim=-1, largest=False)
    return indices, torch.isfinite(nearest), nearest


def test_candidate_rank_feature_flag_gates_dispatch(monkeypatch):
    monkeypatch.setattr(candidate_rank, "available", lambda device: True)
    monkeypatch.setattr(candidate_rank, "FUSED_FUEL_CANDIDATE_RANK_HIP_ENABLED", False)
    assert not candidate_rank.integrated_available("cuda:0")
    monkeypatch.setattr(candidate_rank, "FUSED_FUEL_CANDIDATE_RANK_HIP_ENABLED", True)
    assert candidate_rank.integrated_available("cuda:0")


def test_candidate_extension_failure_disables_dispatch(monkeypatch):
    monkeypatch.setattr(candidate_rank, "FUSED_FUEL_CANDIDATE_RANK_HIP_ENABLED", True)
    monkeypatch.setattr(candidate_rank, "available", lambda device: False)
    assert not candidate_rank.integrated_available("cuda:0")


def test_cpu_candidate_path_falls_back_to_torch(monkeypatch):
    monkeypatch.setattr(candidate_rank, "FUSED_FUEL_CANDIDATE_RANK_HIP_ENABLED", True)
    env = tensor_sim.TensorDefenseEnv(num_envs=8, task="counter_defense", device="cpu",
        seed=932, action_mode="strategic", randomize=False)
    points, _, free = env._perceived_fuel(0)
    expected = _reference(points, env.sim.pose[:, 0, :2], free)
    actual = env._fuel_candidates(0)
    assert not candidate_rank.integrated_available("cpu")
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[0][actual[1]], expected[0][expected[1]])


def test_squared_distance_mode_keeps_torch_fallback(monkeypatch):
    monkeypatch.setattr(candidate_rank, "integrated_available", lambda device: True)
    env = tensor_sim.TensorDefenseEnv(num_envs=8, task="counter_defense", device="cpu",
        seed=933, action_mode="strategic", randomize=False,
        squared_fuel_candidate_distance=True)
    points, _, free = env._perceived_fuel(0)
    delta = points - env.sim.pose[:, 0, None, :2]
    distance_sq = delta.square().sum(-1).masked_fill(~free, float("inf"))
    nearest_sq, indices = distance_sq.topk(4, dim=-1, largest=False)
    expected = indices, torch.isfinite(nearest_sq), nearest_sq.clamp_min(0.).sqrt()
    actual = env._fuel_candidates(0)
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[0][actual[1]], expected[0][expected[1]])


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.version.hip),
                    reason="HIP device required")
def test_hip_random_candidate_outputs_match_torch_exactly():
    assert candidate_rank.integrated_available("cuda:0")
    torch.manual_seed(934)
    n, p = 257, 504
    points = torch.randn((n, p, 2), device="cuda")
    robot = torch.randn((n, 2), device="cuda")
    free = torch.rand((n, p), device="cuda") > .25
    actual = candidate_rank.candidates(points, robot, free)
    expected = _reference(points, robot, free)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.version.hip),
                    reason="HIP device required")
def test_hip_ties_preserve_distances_validity_and_finite_membership():
    assert candidate_rank.integrated_available("cuda:0")
    n, p = 8, 504
    points = torch.randn((n, p, 2), device="cuda")
    robot = torch.zeros((n, 2), device="cuda")
    free = torch.zeros((n, p), dtype=torch.bool, device="cuda")
    # Eight finite candidates at unit distance make the top-4 boundary tied.
    points[0, :8] = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.],
                                  [.6, .8], [-.6, .8], [.8, -.6], [-.8, -.6]], device="cuda")
    free[0, :8] = True
    for row, count in enumerate((0, 1, 2, 3), start=1):
        free[row, :count] = True
    active = torch.tensor([True, False, True, False, True, False, True, False],
                          device="cuda")
    actual = candidate_rank.candidates(points, robot, free, active_mask=active)
    expected = _reference(points, robot, free)
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])
    tied_indices = actual[0][0].cpu().tolist()
    assert len(set(tied_indices)) == 4
    assert set(tied_indices).issubset(set(range(8)))
    # All selected finite ties have the same distance; their particular index
    # order/subset is backend-unstable and is not asserted against one run.
    assert torch.equal(actual[2][0], torch.ones((4,), device="cuda"))
    for row, count in enumerate((0, 1, 2, 3), start=1):
        assert int(actual[1][row].sum().item()) == count
        assert torch.equal(actual[0][row, :count], expected[0][row, :count])
        # Invalid +inf tied candidate indices intentionally have no assertion.


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.version.hip),
                    reason="HIP device required")
def test_env_dispatches_to_fused_rank_when_enabled_and_falls_back_when_disabled(monkeypatch):
    env = tensor_sim.TensorDefenseEnv(num_envs=32, task="counter_defense", device="cuda:0",
        seed=935, action_mode="strategic", randomize=True)
    points, _, free = env._perceived_fuel(0)
    baseline = _reference(points, env.sim.pose[:, 0, :2], free)
    calls = []
    original = candidate_rank.candidates

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate_rank, "candidates", spy)
    monkeypatch.setattr(candidate_rank, "FUSED_FUEL_CANDIDATE_RANK_HIP_ENABLED", False)
    torch_fallback = env._fuel_candidates(0)
    assert not calls
    monkeypatch.setattr(candidate_rank, "FUSED_FUEL_CANDIDATE_RANK_HIP_ENABLED", True)
    fused = env._fuel_candidates(0)
    assert calls
    assert torch.equal(torch_fallback[1], fused[1])
    assert torch.equal(torch_fallback[2], fused[2])
    assert torch.equal(torch_fallback[0][torch_fallback[1]], fused[0][fused[1]])
    assert torch.equal(torch_fallback[0][torch_fallback[1]], baseline[0][baseline[1]])
