"""Pure device-buffer contracts; execution is gated by simulator review."""
from types import SimpleNamespace

import pytest
import torch

from frc_defense.gamepiece_actions import (
    _capture_fuel_score_awards, initialize_fuel_score_event_buffer,
    materialize_fuel_score_awards, reset_fuel_score_event_buffer,
)
from frc_defense.meco_simulated_score_ledger import (
    ESTIMAND_SCHEMA, OUTCOME_NAME, PHASE0_PLAN_SHA256, RULES, RULES_ID,
    RULES_SHA256, SOURCE_PLAN_SHA256, MecoScoreEventRecorder,
    estimand_approval_digest,
)
from test_meco_simulated_score_ledger import _context, _simulator_elapsed


SOURCE = "a" * 64


def _approval():
    return {
        "schema": ESTIMAND_SCHEMA,
        "approval_id": "gpu-score-buffer-fixture",
        "approved": True,
        "approver": "test-only",
        "approved_at": "2026-10-25T12:00:00Z",
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "phase0_plan_sha256": PHASE0_PLAN_SHA256,
        "simulator_source_sha256": SOURCE,
        "rules_id": RULES_ID,
        "rules_sha256": RULES_SHA256,
        "primary_estimand": OUTCOME_NAME,
        "formula": "sum score_units by alliance over accepted FUEL score-award events",
        "unit": RULES["unit"],
        "included_outcomes": RULES["included_outcomes"],
        "excluded_outcomes": RULES["excluded_outcomes"],
        "interpretation_limits": "Fixture only; not official points.",
        "intended_use": "Unit contract only.",
    }


def _env(world_count=2, horizon=4):
    return SimpleNamespace(
        n=world_count, horizon=horizon, device=torch.device("cpu"),
        sim=SimpleNamespace(num_robots=6, dt=0.02),
        robot_score_interval_values=(0.10,) * 6,
        match_elapsed=torch.zeros(world_count, dtype=torch.float32),
        steps=torch.ones(world_count, dtype=torch.long),
        team_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        _meco_piece_generation=torch.zeros((world_count, 8), dtype=torch.long),
        fuel_physics=SimpleNamespace(_gpu_qualifying=True),
        reward=torch.tensor([3., 4.]), generator=torch.Generator().manual_seed(9))


def _recorder(world_count=2):
    approval = _approval()
    context = _context(world_count=world_count)
    return MecoScoreEventRecorder(context, approval,
                                  estimand_approval_digest(approval))


def test_buffer_preserves_vector_and_loop_order_and_ledger_digest():
    env, recorder = _env(), _recorder()
    initialize_fuel_score_event_buffer(env, recorder)
    env._meco_score_event_recorder = recorder
    generator_state = env.generator.get_state().clone()
    reward = env.reward.clone()

    vectorized = torch.zeros((2, 6, 8), dtype=torch.bool)
    vectorized[0, 3, 0] = True
    vectorized[1, 0, 1] = True
    vectorized[1, 4, 2] = True
    env.match_elapsed.fill_(_simulator_elapsed(0))
    _capture_fuel_score_awards(env, vectorized, source_site=(
        "gamepiece_actions.update_fuel_actions.vectorized_score_mutation"))

    env.steps.fill_(2)
    env.match_elapsed.fill_(_simulator_elapsed(1))
    loop_robot_1 = torch.zeros((2, 8), dtype=torch.bool)
    loop_robot_1[0, 3] = True
    loop_robot_1[1, 4] = True
    _capture_fuel_score_awards(env, loop_robot_1, robot_slot=1, source_site=(
        "gamepiece_actions.update_fuel_actions.robot_loop_score_mutation"))
    loop_robot_3 = torch.zeros((2, 8), dtype=torch.bool)
    loop_robot_3[1, 5] = True
    _capture_fuel_score_awards(env, loop_robot_3, robot_slot=3, source_site=(
        "gamepiece_actions.update_fuel_actions.robot_loop_score_mutation"))

    complete, reason = materialize_fuel_score_awards(env, recorder)
    assert complete and reason is None
    assert env.generator.get_state().equal(generator_state)
    torch.testing.assert_close(env.reward, reward, rtol=0, atol=0)

    expected = _recorder()
    expected_rows = (
        (0, 0, 3, 0, "vectorized_score_mutation"),
        (0, 1, 0, 1, "vectorized_score_mutation"),
        (0, 1, 4, 2, "vectorized_score_mutation"),
        (1, 0, 1, 3, "robot_loop_score_mutation"),
        (1, 1, 1, 4, "robot_loop_score_mutation"),
        (1, 1, 3, 5, "robot_loop_score_mutation"),
    )
    for tick, world, robot, piece, suffix in expected_rows:
        expected.record_award(tick=tick, match_elapsed_s=_simulator_elapsed(tick),
            world_slot=world, alliance=0 if robot < 3 else 1,
            robot_slot=robot, piece_slot=piece, piece_generation=0,
            source_site=("gamepiece_actions.update_fuel_actions." + suffix))
    assert recorder.events == expected.events
    captured = recorder.finish(terminal_tick=2, terminal_state="complete",
                               terminal_score_counts=[3, 3])
    baseline = expected.finish(terminal_tick=2, terminal_state="complete",
                               terminal_score_counts=[3, 3])
    assert captured["event_stream"] == baseline["event_stream"]
    assert captured["ledger"]["events_sha256"] == baseline["ledger"]["events_sha256"]


def test_selected_reset_preserves_other_world_rows_and_counts():
    env, recorder = _env(), _recorder()
    initialize_fuel_score_event_buffer(env, recorder)
    env._meco_score_event_recorder = recorder
    chosen = torch.zeros((2, 6, 8), dtype=torch.bool)
    chosen[0, 0, 0] = True
    chosen[1, 2, 1] = True
    _capture_fuel_score_awards(env, chosen, source_site=(
        "gamepiece_actions.update_fuel_actions.vectorized_score_mutation"))
    buffer = env._meco_gpu_score_event_buffer
    buffer["overflow_flags"].fill_(2)
    reset_fuel_score_event_buffer(env, torch.tensor([True, False]))
    assert buffer["world_counts"].tolist() == [0, 1]
    assert int(buffer["aggregate_count"]) == 1
    assert not bool(buffer["event_mask"][0].any())
    assert bool(buffer["event_mask"][1].any())
    assert buffer["reset_seen"].tolist() == [True, False]
    assert int(buffer["overflow_flags"]) == 2
    complete, reason = materialize_fuel_score_awards(env, recorder)
    assert not complete and "reset crossed" in reason


def test_all_false_reset_is_exact_noop():
    env, recorder = _env(), _recorder()
    initialize_fuel_score_event_buffer(env, recorder)
    chosen = torch.zeros((2, 6, 8), dtype=torch.bool)
    chosen[1, 2, 1] = True
    _capture_fuel_score_awards(env, chosen, source_site=(
        "gamepiece_actions.update_fuel_actions.vectorized_score_mutation"))
    buffer = env._meco_gpu_score_event_buffer
    before = {name: value.clone() for name, value in buffer.items()
              if isinstance(value, torch.Tensor)}
    reset_fuel_score_event_buffer(env, torch.tensor([False, False]))
    for name, value in before.items():
        assert buffer[name].equal(value), name


def test_capacity_bound_and_overflow_fail_closed_without_partial_stream():
    supported = _env(world_count=1, horizon=8000)
    initialize_fuel_score_event_buffer(supported, _recorder(world_count=1))
    assert supported._meco_gpu_score_event_buffer["capacity"] <= 32768

    oversized = _env(world_count=10, horizon=8000)
    with pytest.raises(ValueError, match="aggregate episode cap"):
        initialize_fuel_score_event_buffer(
            oversized, _recorder(world_count=10))

    env, recorder = _env(), _recorder()
    initialize_fuel_score_event_buffer(env, recorder)
    env._meco_gpu_score_event_buffer["overflow_flags"].fill_(2)
    complete, reason = materialize_fuel_score_awards(env, recorder)
    assert not complete and "overflow" in reason
    incomplete = recorder.finish_incomplete(terminal_tick=0,
        terminal_state="complete", terminal_score_counts=[0, 0], reason=reason)
    assert incomplete["ledger"]["status"] == "incomplete"
    assert incomplete["ledger"]["qualification_eligible"] is False
    assert incomplete["ledger"]["official_match_points"] is None
    assert incomplete["event_stream"] is None
