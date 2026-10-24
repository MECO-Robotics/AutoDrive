import torch

from frc_defense.meco_phase4_opponents import (
    POLICY_FACTOR_IDS, POLICY_POOL, opponent_velocity_command, policy_pool_identity,
)


def _state():
    return {
        "own_xy": torch.tensor([[1., 1.], [1., 1.]]),
        "target_xy": torch.tensor([[4., 1.], [4., 1.]]),
        "target_velocity_xy": torch.tensor([[0., 1.], [0., 1.]]),
        "target_valid": torch.tensor([True, False]),
        "anchor_xy": torch.tensor([[2., 2.], [2., 2.]]),
        "objective_xy": torch.tensor([[8., 1.], [8., 1.]]),
        "phase": torch.tensor([0.5, 0.5]),
    }


def test_pool_covers_frozen_design_axis_and_has_stable_identity():
    assert POLICY_POOL == (
        "stationary_blockage", "direct_pursuit", "trench_interceptor",
        "telegraph_reversal", "reactive_defender_a", "reactive_defender_b",
        "orbit_to_pursuit_switch")
    assert set(POLICY_FACTOR_IDS) == set(POLICY_POOL)
    assert policy_pool_identity() == policy_pool_identity()
    assert len(policy_pool_identity()["config_sha256"]) == 64


def test_commands_are_deterministic_bounded_and_track_masked():
    state = _state()
    for policy in POLICY_POOL:
        first = opponent_velocity_command(policy, state, tick=100)
        second = opponent_velocity_command(policy, state, tick=100)
        torch.testing.assert_close(first, second, rtol=0., atol=0.)
        assert first.shape == (2, 2)
        assert torch.isfinite(first).all()
        assert bool((first.norm(dim=-1) <= 1.500001).all())
        if policy in {"direct_pursuit", "trench_interceptor", "reactive_defender_a",
                      "reactive_defender_b", "orbit_to_pursuit_switch"}:
            torch.testing.assert_close(first[1], torch.zeros(2))


def test_stationary_feint_and_switch_have_explicit_deterministic_stimuli():
    state = _state()
    torch.testing.assert_close(
        opponent_velocity_command("stationary_blockage", state, tick=0),
        torch.zeros((2, 2)))
    before = opponent_velocity_command("telegraph_reversal", state, tick=359)
    after = opponent_velocity_command("telegraph_reversal", state, tick=360)
    assert not torch.equal(before, after)
    orbit = opponent_velocity_command("orbit_to_pursuit_switch", state, tick=359)
    pursuit = opponent_velocity_command("orbit_to_pursuit_switch", state, tick=360)
    assert not torch.equal(orbit[0], pursuit[0])
    torch.testing.assert_close(pursuit[0], torch.tensor([1.5, 0.]))


def test_per_robot_speed_limits_remain_device_shaped():
    command = opponent_velocity_command("direct_pursuit", _state(), tick=0,
                                        max_speed_mps=torch.tensor([0.4, 0.8]))
    assert command.shape == (2, 2)
    torch.testing.assert_close(command.norm(dim=-1), torch.tensor([0.4, 0.]))


def test_pool_rejects_unknown_policy_and_malformed_observation():
    state = _state()
    try:
        opponent_velocity_command("unregistered", state, tick=0)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown policy accepted")
    del state["target_xy"]
    try:
        opponent_velocity_command("direct_pursuit", state, tick=0)
    except ValueError as exc:
        assert "target_xy" in str(exc)
    else:
        raise AssertionError("incomplete sensed state accepted")


def test_nonfinite_sensed_state_and_speed_fail_closed_to_finite_zero_commands():
    state = _state()
    state["target_xy"][0, 0] = float("nan")
    state["target_velocity_xy"][1, 1] = float("inf")
    for policy in ("direct_pursuit", "trench_interceptor", "reactive_defender_b",
                   "orbit_to_pursuit_switch"):
        command = opponent_velocity_command(policy, state, tick=400)
        assert torch.isfinite(command).all()
        torch.testing.assert_close(command, torch.zeros_like(command))

    command = opponent_velocity_command(
        "stationary_blockage", _state(), tick=0,
        max_speed_mps=torch.tensor(float("nan")))
    assert torch.isfinite(command).all()
    torch.testing.assert_close(command, torch.zeros_like(command))


def test_nonfinite_batch_speed_fails_closed_only_for_corrupt_rows():
    command = opponent_velocity_command(
        "direct_pursuit", _state(), tick=0,
        max_speed_mps=torch.tensor([float("nan"), 0.8]))
    assert torch.isfinite(command).all()
    torch.testing.assert_close(command[0], torch.zeros(2))
    torch.testing.assert_close(command[1], torch.zeros(2))
