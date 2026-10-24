import torch
import pytest
from types import SimpleNamespace

from frc_defense.meco_phase4_opponents import (
    POLICY_FACTOR_IDS, POLICY_POOL, compile_opponent_policy_schedule,
    opponent_velocity_command, policy_pool_identity,
    opponent_policy_runtime_slots, validate_opponent_policy_schedule,
    TRENCH_INTERCEPTION_CATEGORIES,
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


def test_validation_schedule_binds_policy_case_seeds_and_sensor_boundary():
    case = {"layout_id": "layout-a", "defender_family": "held-family",
            "start_condition": "start-a", "sensor_profile": "noisy"}
    schedule = compile_opponent_policy_schedule(policy_id="reactive_defender_a",
        scenario_id="offense_vs_scripted_pressure", categories=("active/trench/level_4_reactive",),
        qualification_case=case, seeds=(10, 20), split="validation")
    assert schedule["executable"] is True
    assert schedule["qualification_eligible"] is False
    assert "sensed" in schedule["policy_inputs"]
    assert schedule["policy_config_sha256"] == policy_pool_identity()["config_sha256"]
    assert validate_opponent_policy_schedule(schedule,
        scenario_id="offense_vs_scripted_pressure",
        categories=("active/trench/level_4_reactive",),
        qualification_case=case, seeds=(10, 20), split="validation") == schedule
    with pytest.raises(ValueError, match="exact protocol case/seed/config"):
        validate_opponent_policy_schedule(schedule,
            scenario_id="offense_vs_scripted_pressure",
            categories=("active/trench/level_4_reactive",),
            qualification_case=case, seeds=(10, 21), split="validation")
    with pytest.raises(ValueError, match="validation-only"):
        compile_opponent_policy_schedule(policy_id="reactive_defender_a",
            scenario_id="offense_vs_scripted_pressure", categories=("active/trench/level_4_reactive",),
            qualification_case=case, seeds=(10, 20), split="test")


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


def test_nonfinite_or_nonbinary_validity_metadata_fails_closed():
    state = _state()
    state["target_valid"] = torch.tensor([float("nan"), 2.])
    for policy in ("direct_pursuit", "trench_interceptor", "reactive_defender_b",
                   "orbit_to_pursuit_switch"):
        command = opponent_velocity_command(policy, state, tick=400)
        assert torch.isfinite(command).all()
        torch.testing.assert_close(command, torch.zeros_like(command))


def test_trench_interceptor_requires_exact_registered_validation_domain():
    case = {"layout_id": "layout-a", "defender_family": "held-family",
            "start_condition": "start-a", "sensor_profile": "noisy"}
    assert len(TRENCH_INTERCEPTION_CATEGORIES) == 14
    schedule = compile_opponent_policy_schedule(
        policy_id="trench_interceptor",
        scenario_id="offense_vs_scripted_pressure",
        categories=("active/trench/level_4_reactive",), qualification_case=case,
        seeds=(10, 20), split="validation",
        factor_id="opponent.trench_interception", protocol_id="held-validation-v1")
    assert schedule["protocol_id"] == "held-validation-v1"
    assert schedule["category_domain"] == list(TRENCH_INTERCEPTION_CATEGORIES)
    assert schedule["configured_pressure_slots"] == {
        "active/trench/level_4_reactive": [0]}
    assert opponent_policy_runtime_slots(
        "active/trench/level_0", allow_level_zero=True) == ()
    with pytest.raises(ValueError, match="pressure levels 1 through 5"):
        opponent_policy_runtime_slots("active/trench/level_0")
    with pytest.raises(ValueError, match="category subset"):
        compile_opponent_policy_schedule(
            policy_id="trench_interceptor",
            scenario_id="offense_vs_scripted_pressure",
            categories=("active/open/level_4_reactive",),
            qualification_case=case, seeds=(10,), split="validation",
            factor_id="opponent.trench_interception", protocol_id="held-validation-v1")


@pytest.mark.parametrize("slots", [(0,), (0, 1)])
def test_trench_schedule_reaches_runtime_command_for_bound_slots(slots):
    from frc_defense.curriculum_scenarios import CurriculumEnv

    category = ("active/trench/level_5_mixed" if len(slots) == 2
                else "active/trench/level_4_reactive")
    commands = torch.full((1, 6, 3), -7.)
    env = SimpleNamespace(
        curriculum={"pressure": {"level": 5 if len(slots) == 2 else 4,
                                  "slots": list(slots), "phase": 0.}},
        phase4_factor_config=None, phase4_opponent_policy_id="trench_interceptor",
        phase4_opponent_runtime={"configured_slots": list(slots),
                                 "applied_slots": [], "applied_command_calls": 0},
        steps=4, match_elapsed=torch.tensor([0.]), field_length=16., field_width=8.,
        policy_pose=torch.tensor([[[1., 1., 0.]] * 6]),
        opponent_pose=torch.tensor([[[4., 1., 0.]] * 6]),
        opponent_velocity=torch.tensor([[[0., 1., 0.]] * 6]),
        opponent_valid=torch.ones((1, 6), dtype=torch.bool),
        phase4_defender_anchor=torch.tensor([[[2., 2., 0.]] * 6]),
        sim=SimpleNamespace(speed=torch.ones((1, 6))),
    )
    output = CurriculumEnv._curriculum_commands(
        env, commands, torch.ones((1,), dtype=torch.bool))
    assert env.phase4_opponent_runtime["applied_slots"] == list(slots)
    assert env.phase4_opponent_runtime["applied_command_calls"] == len(slots)
    for slot in slots:
        assert not torch.equal(output[0, slot, :2], torch.tensor([-7., -7.]))
    for slot in set(range(6)) - set(slots):
        torch.testing.assert_close(output[0, slot], torch.tensor([-7., -7., -7.]))
