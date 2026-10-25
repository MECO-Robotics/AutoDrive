import json
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frc_defense.meco_phase4 import phase4_factor_config
from frc_defense.meco_phase4 import PHASE4_FACTORS
from frc_defense.meco_phase4_runtime_adapter import (
    FirstRouteProgressInterval,
    PHASE4_CONTACT_ROUTE_GOAL_TOLERANCE_RESOLUTION_FACTOR,
    PHASE4_REQUIREMENT_BINDINGS,
    phase4_factor_exposure,
    phase4_pair_contact_event_record,
    route_remaining_distance_m,
    phase4_requirement_binding,
    requirement_ids_for_factor,
    validate_phase4_capture,
)
from frc_defense.meco_track_training import _observe_phase4_contact_route_progress
from frc_defense.curriculum_scenarios import CurriculumEnv
import frc_defense.curriculum_scenarios as curriculum_scenarios
from frc_defense.tensor_physics_collision import pair_contact_substep_endpoint


def test_stationary_blocker_overrides_only_its_slot_and_masks_inactive_envs(monkeypatch):
    env = object.__new__(CurriculumEnv)
    env.curriculum = {"pressure": {"level": 1, "slots": [0, 1],
        "family": "stationary_blocker", "phase": 0.}}
    env.phase4_factor_config = {"kind": "stationary_blockage", "blocker_slot": 0,
        "pose_tolerance_m": .02, "heading_tolerance_rad": .02}
    env.stationary_blockage_assigned_pose = torch.zeros((2, 3))
    env.phase4_runtime_metrics = {"stationary_blockage_pose_valid": torch.ones(2, dtype=torch.bool)}
    env.sim = SimpleNamespace(pose=torch.zeros((2, 6, 3)),
                              speed=torch.ones((2, 6)))
    env.policy_pose = torch.zeros((2, 6, 3))
    env.opponent_pose = torch.zeros((2, 6, 3))
    env.opponent_velocity = torch.zeros((2, 6, 2))
    env.opponent_valid = torch.ones((2, 6), dtype=torch.bool)
    env.phase4_defender_anchor = torch.zeros((2, 6, 3))
    env.field_length, env.field_width = 16., 8.
    env.match_elapsed = torch.zeros(2)
    env.steps = torch.zeros(2, dtype=torch.long)
    env.phase4_opponent_policy_id = "scheduled"
    seen_slots = []

    def scheduled_opponent(_policy_id, sensed, *, tick, max_speed_mps):
        seen_slots.append(sensed["own_xy"].shape[1] if sensed["own_xy"].ndim > 1 else tick)
        return torch.ones((2, 2))

    monkeypatch.setattr(curriculum_scenarios, "opponent_velocity_command", scheduled_opponent)
    commands = torch.full((2, 6, 3), 9.)
    result = env._curriculum_commands(commands, torch.tensor([True, False]))
    assert torch.equal(result[0, 0], torch.zeros(3))
    assert torch.equal(result[0, 1, :2], torch.ones(2))
    assert torch.equal(result[1], torch.zeros((6, 3)))
    assert len(seen_slots) == 1
from frc_defense.meco_ablation_matrix import _validate_phase4_condition_records


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_binding_covers_exact_registry_requirements_and_fails_closed():
    registry = json.loads((ROOT / "qualifications/meco/phase4_design_registry.json").read_text())
    expected = {row["requirement_id"] for row in registry["requirements"]}
    assert set(PHASE4_REQUIREMENT_BINDINGS) == expected
    assert all(not row["stimulus_supported"] or row["runtime_factor_kind"]
               for row in PHASE4_REQUIREMENT_BINDINGS.values())
    for row in registry["requirements"]:
        binding = phase4_requirement_binding(row["requirement_id"])
        assert binding["factor_ids"] == row["factor_ids"]
        assert binding["required_signals"] == row["required_outcome_signals"]
        assert set(binding["runtime_factor_by_factor_id"]) <= set(binding["factor_ids"])
        assert set(binding["runtime_event_by_factor_id"]) <= set(binding["factor_ids"])
        assert set(binding["runtime_factor_by_factor_id"].values()) <= (
            set(PHASE4_FACTORS) | {"legal_team_start_profile"})
        assert not set(binding["required_signals"]).issubset(
            binding["authoritative_signals"])
    contact = phase4_requirement_binding("bumps_and_contact")
    assert contact["runtime_event_by_factor_id"] == {
        "contact.robot_bump": "phase4_pair_contact_event"}
    assert contact["stimulus_supported"] is False
    starts = phase4_requirement_binding("multiple_legal_team_start_states")
    assert starts["stimulus_supported"] is True
    assert starts["runtime_event_by_factor_id"] == {
        "start.legal_team_poses":
            "phase4_start_profile.profile_applied_before_observation"}
    assert starts["authoritative_signals"] == []
    focal = phase4_requirement_binding("observation_only_focal_pose_mislocalization")
    assert focal["runtime_factor_kind"] is None
    assert focal["runtime_factor_by_factor_id"] == {}
    assert focal["runtime_event"] is None
    assert focal["runtime_event_by_factor_id"] == {}
    assert focal["stimulus_supported"] is False
    assert focal["assignment_runtime_event"] == "phase4_focal_pose_assignment"
    assert focal["assignment_applied_event"] == (
        "records[].phase4_focal_pose.applied_before_policy_observation")
    assert focal["assignment_path_supported"] is True
    assert not requirement_ids_for_factor({"kind": "focal_pose_offset"})
    assert focal["authoritative_signals"] == []
    assert focal["qualification_eligible"] is False
    assert any("initial-only" in limitation for limitation in focal["limitations"])


@pytest.mark.parametrize("kind,event_field,event_value", [
    ("feinting_opponent", "feint_reversal_tick", 360),
    ("abrupt_opponent_policy_change", "policy_change_tick", 360),
    ("observation_delay", "delayed_observation_ticks", 6),
    ("independent_low_confidence", "confidence_cap_refresh_count", 12),
])
def test_capture_binds_only_realized_existing_stimuli_and_marks_missing_outcomes(
        kind, event_field, event_value):
    factor = phase4_factor_config(kind)
    source = {"repository": "fixture", "commit": "abc123"}
    runtime = {event_field: event_value}
    if kind == "independent_low_confidence":
        runtime.update(confidence_cap_applied=factor["fuel_confidence_cap"],
                       confidence_cap_reduced_track_values=4)
    capture = phase4_factor_exposure(factor, runtime,
        {"success": 1.0, "safety_robot_contact_rate": 0.0},
        source_identity=source, split="validation", seed=21,
        scenario_id="offense_vs_scripted_pressure", category="active/trench/level_1_stationary",
        qualification_case={"layout_id": "layout-a", "defender_family": "family-a",
            "start_condition": "start-a", "sensor_profile": "sensor-a"})
    assert capture["status"] == "realized_stimulus_outcomes_incomplete"
    assert capture["runtime_event"]["realized"] is True
    assert capture["required_signals_unavailable"]
    assert capture["qualification_eligible"] is False
    validate_phase4_capture(capture, expected_source_identity=source)
    with pytest.raises(ValueError, match="source identity mismatch"):
        validate_phase4_capture(capture, expected_source_identity={"commit": "different"})
    validate_phase4_capture(capture, expected_source_identity_sha256=
        capture["source_identity_sha256"], expected_episode_seed=21,
        expected_split="validation", expected_qualification_case=capture["qualification_case"])


def test_capture_accepts_only_complete_route_outcome_with_exact_episode_binding():
    from frc_defense.meco_phase4_runtime_adapter import phase4_route_completion_record

    factor = phase4_factor_config("feinting_opponent")
    source = {"repository": "fixture", "commit": "abc123"}
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    binding = {"episode_id": "episode-1", "policy_id": "A1",
        "scenario_id": "case", "category": "category", "case_id": "case-1",
        "seed": 21, "split": "validation", "factor_id": "opponent.feint_reversal",
        "condition": "factor_treatment", "source_identity_sha256": source_hash}
    route = phase4_route_completion_record([{
        "macro_index": 0, "episode_id": "episode-1",
        "source_identity_sha256": source_hash, "goal_lifecycle_id": "goal-1",
        "selected_goal_id": "goal-1", "route_id": "route-1", "goal_xy_m": [0., 0.],
        "goal_heading_rad": None, "execution_valid": True,
        "execution_exclusion_reasons": [], "executed_action_trace": [0],
        "terminal_goal_distance_m": .01, "goal_heading_required": False,
        "terminal_heading_error_rad": None, "terminal_pose_xy_m": [.01, 0.],
        "terminal_pose_heading_rad": 0.}], binding=binding)
    capture = phase4_factor_exposure(factor, {"feint_reversal_tick": 360}, {},
        source_identity=source, split="validation", seed=21,
        scenario_id="case", category="category", episode_id="episode-1",
        policy_id="A1", route_completion_by_factor={
            "opponent.feint_reversal": route})
    observed = capture["route_outcome_observation"]
    assert observed["route_completion_rate"] == 1.
    assert observed["binding"] == binding
    assert observed["authoritative_signal_available"] is False
    assert capture["qualification_eligible"] is False
    validate_phase4_capture(capture, expected_source_identity=source)
    wrong_condition = copy.deepcopy(capture)
    wrong_condition["route_outcome_observation"]["binding"]["condition"] = "no_factor"
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_phase4_capture(wrong_condition, expected_source_identity=source)
    inconsistent_counts = copy.deepcopy(capture)
    inconsistent_observation = inconsistent_counts["route_outcome_observation"]
    # Preserve the completed lifecycle row while falsifying aggregate counts.
    assert inconsistent_observation["attempts"][0]["outcome"] == "completed"
    inconsistent_observation["completed"] = 0
    inconsistent_observation["observed_failure"] = 1
    inconsistent_observation["route_completion_rate"] = 0.
    with pytest.raises(ValueError, match="incomplete or invalid"):
        validate_phase4_capture(inconsistent_counts,
            expected_source_identity=source)

    wrong_episode = copy.deepcopy(route)
    wrong_episode["binding"]["episode_id"] = "episode-2"
    with pytest.raises(ValueError, match="binding mismatch"):
        phase4_factor_exposure(factor, {"feint_reversal_tick": 360}, {},
            source_identity=source, split="validation", seed=21,
            scenario_id="case", category="category", episode_id="episode-1",
            policy_id="A1", route_completion_by_factor={
                "opponent.feint_reversal": wrong_episode})

    censored = copy.deepcopy(route)
    censored["censored"] = 1
    censored_capture = phase4_factor_exposure(factor,
        {"feint_reversal_tick": 360}, {}, source_identity=source,
        split="validation", seed=21, scenario_id="case", category="category",
        episode_id="episode-1", policy_id="A1", route_completion_by_factor={
            "opponent.feint_reversal": censored})
    assert censored_capture["route_outcome_observation"] is None


def test_stationary_blockage_is_registered_and_capture_requires_reset_assignment():
    factor = phase4_factor_config("stationary_blockage")
    source = {"repository": "fixture", "commit": "abc123"}
    capture = phase4_factor_exposure(factor,
        {"stationary_blockage_assigned": True,
         "stationary_blockage_pose_valid": True}, {}, source_identity=source,
        split="validation", seed=21,
        scenario_id="offense_vs_scripted_pressure",
        category="active/trench/level_1_stationary",
        qualification_case={"layout_id": "layout-a",
            "defender_family": "family-a", "start_condition": "start-a",
            "sensor_profile": "sensor-a"})
    assert capture["runtime_event"]["realized"] is True
    assert capture["runtime_event"]["evidence"]["blocker_slot"] == 0
    validate_phase4_capture(capture, expected_source_identity=source)
    drifted = phase4_factor_exposure(factor,
        {"stationary_blockage_assigned": True,
         "stationary_blockage_pose_valid": False}, {}, source_identity=source,
        split="validation", seed=21,
        scenario_id="offense_vs_scripted_pressure",
        category="active/trench/level_1_stationary",
        qualification_case={"layout_id": "layout-a",
            "defender_family": "family-a", "start_condition": "start-a",
            "sensor_profile": "sensor-a"})
    assert drifted["status"] == "stimulus_not_realized"
    assert drifted["qualification_eligible"] is False


@pytest.mark.parametrize("mutation,error", [
    (lambda row: row.update(factor_sha256="0" * 64), "factor registration or digest"),
    (lambda row: row.update(status="stimulus_not_realized"), "not a realized"),
    (lambda row: row["runtime_event"].update(realized=False), "was not realized"),
    (lambda row: row["runtime_event"].update(observed=359), "does not match"),
    (lambda row: row.update(required_signals_unavailable=[]), "signal accounting"),
    (lambda row: row.update(seed=22), "seed differs"),
    (lambda row: row.update(qualification_case={"layout_id": "wrong"}), "qualification case"),
])
def test_capture_validator_rejects_provenance_and_realization_tampering(mutation, error):
    factor = phase4_factor_config("feinting_opponent")
    source = {"commit": "abc123"}
    case = {"layout_id": "layout-a", "defender_family": "family-a",
            "start_condition": "start-a", "sensor_profile": "sensor-a"}
    capture = phase4_factor_exposure(factor, {"feint_reversal_tick": 360}, {},
        source_identity=source, split="validation", seed=21,
        scenario_id="offense_vs_scripted_pressure", category="active/trench/level_1_stationary",
        qualification_case=case)
    tampered = copy.deepcopy(capture)
    mutation(tampered)
    with pytest.raises(ValueError, match=error):
        validate_phase4_capture(tampered, expected_source_identity=source,
            expected_episode_seed=21, expected_split="validation",
            expected_qualification_case=case)


def test_capture_does_not_claim_unrealized_stimulus_or_expose_test_seed():
    factor = phase4_factor_config("feinting_opponent")
    source = {"commit": "abc123"}
    capture = phase4_factor_exposure(factor, {"feint_reversal_tick": None}, {},
        source_identity=source, split="test", seed=987654321,
        scenario_id="offense_vs_scripted_pressure", category="active/trench/level_1_stationary")
    assert capture["status"] == "stimulus_not_realized"
    assert capture["runtime_event"]["realized"] is False
    assert capture["seed"] is None
    assert capture["seed_redacted"] is True
    assert "987654321" not in json.dumps(capture)
    with pytest.raises(ValueError, match="not a realized"):
        validate_phase4_capture(capture, expected_source_identity=source)


def test_realized_test_capture_keeps_seed_redacted_and_mismatch_is_rejected():
    factor = phase4_factor_config("feinting_opponent")
    source = {"commit": "abc123"}
    capture = phase4_factor_exposure(factor, {"feint_reversal_tick": 360}, {},
        source_identity=source, split="test", seed=987654321,
        scenario_id="offense_vs_scripted_pressure", category="active/trench/level_1_stationary")
    validate_phase4_capture(capture, expected_source_identity=source,
                            expected_episode_seed=987654321, expected_split="test")
    assert capture["seed"] is None and capture["seed_redacted"] is True
    leaked = copy.deepcopy(capture)
    leaked.update(seed=987654321, seed_redacted=False)
    with pytest.raises(ValueError, match="must redact"):
        validate_phase4_capture(leaked, expected_source_identity=source,
                                expected_split="test")


def test_matrix_intake_rejects_unrealized_or_episode_mismatched_capture():
    factor = phase4_factor_config("feinting_opponent")
    source = {"commit": "abc123"}
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    case = {"layout_id": "layout-a", "defender_family": "family-a",
            "start_condition": "start-a", "sensor_profile": "sensor-a"}
    capture = phase4_factor_exposure(factor, {"feint_reversal_tick": 360}, {},
        source_identity=source, split="validation", seed=21,
        scenario_id="offense_vs_scripted_pressure", category="active/trench/level_1_stationary",
        qualification_case=case)
    row = {"phase4_realization_capture": capture, "scenario_id": capture["scenario_id"],
        "category": capture["category"], "seed": 21, **case}
    _validate_phase4_condition_records([row], factor=factor,
        source_hash=source_hash, split="validation", qualification_case=case)

    unrealized = copy.deepcopy(row)
    unrealized["phase4_realization_capture"]["status"] = "stimulus_not_realized"
    with pytest.raises(ValueError, match="not a realized"):
        _validate_phase4_condition_records([unrealized], factor=factor,
            source_hash=source_hash, split="validation", qualification_case=case)
    event_not_realized = copy.deepcopy(row)
    event_not_realized["phase4_realization_capture"]["runtime_event"]["realized"] = False
    with pytest.raises(ValueError, match="was not realized"):
        _validate_phase4_condition_records([event_not_realized], factor=factor,
            source_hash=source_hash, split="validation", qualification_case=case)
    wrong_seed = copy.deepcopy(row)
    wrong_seed["phase4_realization_capture"]["seed"] = 22
    with pytest.raises(ValueError, match="seed differs"):
        _validate_phase4_condition_records([wrong_seed], factor=factor,
            source_hash=source_hash, split="validation", qualification_case=case)


def test_pair_contact_reader_binds_simulator_event_summary_to_contact_factor():
    class ContractFixture:
        num_robots = 6
        _collision_pair_i = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
        _collision_pair_j = torch.tensor([1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5])
        pair_contact_event_count = torch.zeros((1, 15), dtype=torch.long)
        pair_contact_solver_report_count = torch.zeros_like(pair_contact_event_count)
        pair_contact_first_substep = torch.full_like(pair_contact_event_count, -1)
        pair_contact_last_substep = torch.full_like(pair_contact_event_count, -1)
        pair_contact_first_time_s = torch.full((1, 15), float("nan"), dtype=torch.float64)
        pair_contact_last_time_s = torch.full_like(pair_contact_first_time_s, float("nan"))
        pair_contact_impulse_sum = torch.zeros((1, 15, 4))

    fixture = ContractFixture()
    pair_slot = 9  # Lexicographic pair slot (2, 3).
    fixture.pair_contact_event_count[0, pair_slot] = 2
    fixture.pair_contact_solver_report_count[0, pair_slot] = 6
    fixture.pair_contact_first_substep[0, pair_slot] = 1
    fixture.pair_contact_last_substep[0, pair_slot] = 2
    fixture.pair_contact_first_time_s[0, pair_slot] = .02
    fixture.pair_contact_last_time_s[0, pair_slot] = .04
    fixture.pair_contact_impulse_sum[0, pair_slot] = torch.tensor([3., -1., 2., 4.])
    record = phase4_pair_contact_event_record(fixture)
    assert record["factor_id"] == "contact.robot_bump"
    assert record["pair_slots"] == [2, 3]
    assert record["event_realized_by_world"] == [True]
    assert record["distinct_contact_substeps_by_world"] == [2]
    assert record["contact_solver_reports_by_world"] == [6]
    assert record["first_contact_physics_substep_by_world"] == [1]
    assert record["first_contact_physics_time_s_by_world"] == [.02]
    assert record["last_contact_physics_substep_by_world"] == [2]
    assert record["accumulated_impulse_by_world"] == [{
        "normal_n_s": 3., "tangential_n_s": -1.,
        "net_on_second_robot_n_s": [2., 4.]}]
    assert "not a complete event stream" in record["event_scope"]
    assert record["recovery_time_after_bump_s"] is None
    assert record["route_completion_rate"] is None
    with pytest.raises(ValueError, match="requires contact.robot_bump"):
        phase4_pair_contact_event_record(fixture, factor_id="contact.other")


def test_first_contact_clock_uses_completed_substep_endpoint():
    dt = .02
    assert pair_contact_substep_endpoint(0, 0.0, dt) == (1, dt)
    next_substep, next_time = pair_contact_substep_endpoint(9, .18, dt)
    assert next_substep == 10
    assert next_time == pytest.approx(.20)
    substep, time_s = pair_contact_substep_endpoint(
        torch.tensor([0]), torch.tensor([0.], dtype=torch.float64), dt)
    assert substep.tolist() == [1]
    assert time_s.tolist() == pytest.approx([dt])
    hip_source = (ROOT / "frc_defense/csrc/tensor_collision_multi_hip_kernel.hip").read_text()
    assert "physics_substep_index[world] + 1" in hip_source
    assert "physics_time_s[world] + step_dt" in hip_source
    assert "physics_substep_index[0] + 1" in hip_source
    assert "physics_time_s[0] + step_dt" in hip_source


def test_route_progress_interval_is_outer_tick_censored_and_contact_resets_baseline():
    observer = FirstRouteProgressInterval()
    def sample(tick, count, distance, goal="g"):
        return {"outer_tick": tick, "outer_tick_time_s": tick * .02,
            "distinct_contact_substeps": count,
            "last_contact_physics_time_s": count * .02,
            "last_contact_physics_substep": count,
            "selected_goal_id": goal, "selected_goal_xy_m": [1., 0.],
            "route_remaining_distance_m": distance}
    observer.observe(sample(1, 1, 1.0))
    observer.observe(sample(2, 1, .9))
    assert observer.record()["first_route_progress_interval"] is not None
    interval = observer.record()["first_route_progress_interval"]
    assert interval["contact_physics_time_lower_bound_s"] == .02
    assert interval["baseline_route_sample_outer_tick_time_s"] == .02
    assert interval["route_progress_outer_tick_upper_bound_s"] == .04
    assert observer.record()["recovery_time_after_bump_s"] is None
    assert observer.record()["qualification_eligible"] is False
    observer.observe(sample(3, 2, .7))
    assert observer.record()["first_route_progress_interval"] is None
    assert observer.record()["null_reason"] == "horizon_without_route_progress"

    reset = FirstRouteProgressInterval()
    reset.observe(sample(1, 1, 1.0))
    reset.observe(sample(2, 2, .99))
    reset.observe(sample(3, 2, .90))
    assert reset.record()["first_route_progress_interval"][
        "contact_physics_time_lower_bound_s"] == .04
    changed = FirstRouteProgressInterval()
    changed.observe(sample(1, 1, 1.0))
    changed.observe(sample(2, 1, .9, goal="other"))
    assert changed.record()["null_reason"] == "goal_changed"


def test_route_projection_and_null_reasons_fail_closed():
    tol = .2 * PHASE4_CONTACT_ROUTE_GOAL_TOLERANCE_RESOLUTION_FACTOR
    assert route_remaining_distance_m([[0., 0.], [1., 0.], [1., 1.]],
        [.5, .1], [1., 1.], goal_tolerance_m=tol) == pytest.approx(1.5)
    assert route_remaining_distance_m([[0., 0.], [1., 0.]], [0., 0.], [2., 0.],
        goal_tolerance_m=tol) is None
    no_contact = FirstRouteProgressInterval().record()
    assert no_contact["null_reason"] == "no_contact_observed"
    invalid = FirstRouteProgressInterval()
    invalid.observe({"outer_tick": 1, "outer_tick_time_s": .02, "distinct_contact_substeps": 1,
        "last_contact_physics_time_s": .015, "last_contact_physics_substep": 1,
        "selected_goal_id": "g", "selected_goal_xy_m": [1., 0.],
        "route_remaining_distance_m": None})
    assert invalid.record()["null_reason"] == "no_valid_executable_route"
    horizon = FirstRouteProgressInterval()
    horizon.observe({"outer_tick": 1, "outer_tick_time_s": .02, "distinct_contact_substeps": 1,
        "last_contact_physics_time_s": .02,
        "last_contact_physics_substep": 1,
        "selected_goal_id": "g", "selected_goal_xy_m": [1., 0.],
        "route_remaining_distance_m": 1.})
    assert horizon.record()["null_reason"] == "horizon_without_route_progress"

    small_progress = FirstRouteProgressInterval()
    small_progress.observe({"outer_tick": 1, "outer_tick_time_s": .02, "distinct_contact_substeps": 1,
        "last_contact_physics_time_s": .02, "last_contact_physics_substep": 1,
        "selected_goal_id": "g",
        "selected_goal_xy_m": [1., 0.], "route_remaining_distance_m": 1.})
    small_progress.observe({"outer_tick": 2, "outer_tick_time_s": .04, "distinct_contact_substeps": 1,
        "last_contact_physics_time_s": .02, "last_contact_physics_substep": 1,
        "selected_goal_id": "g",
        "selected_goal_xy_m": [1., 0.], "route_remaining_distance_m": .999})
    assert small_progress.record()["first_route_progress_interval"] is not None

    nonmonotonic = FirstRouteProgressInterval()
    nonmonotonic.observe({"outer_tick": 2, "outer_tick_time_s": .04,
        "distinct_contact_substeps": 1, "last_contact_physics_time_s": .03,
        "selected_goal_id": "g", "selected_goal_xy_m": [1., 0.],
        "route_remaining_distance_m": 1.})
    nonmonotonic.observe({"outer_tick": 2, "outer_tick_time_s": .06,
        "distinct_contact_substeps": 1, "selected_goal_id": "g",
        "selected_goal_xy_m": [1., 0.], "route_remaining_distance_m": .9})
    assert nonmonotonic.record()["null_reason"] == "non_monotonic_outer_tick_or_time"
    regressed = FirstRouteProgressInterval()
    regressed.observe({"outer_tick": 1, "outer_tick_time_s": .02,
        "distinct_contact_substeps": 2, "last_contact_physics_time_s": .02,
        "selected_goal_id": "g", "selected_goal_xy_m": [1., 0.],
        "route_remaining_distance_m": 1.})
    regressed.observe({"outer_tick": 2, "outer_tick_time_s": .04,
        "distinct_contact_substeps": 1, "selected_goal_id": "g",
        "selected_goal_xy_m": [1., 0.], "route_remaining_distance_m": .9})
    assert regressed.record()["null_reason"] == "contact_substep_count_regressed"

    solver_time_regressed = FirstRouteProgressInterval()
    solver_time_regressed.observe({"outer_tick": 1, "outer_tick_time_s": .02,
        "distinct_contact_substeps": 1, "last_contact_physics_time_s": .015,
        "last_contact_physics_substep": 4, "selected_goal_id": "g",
        "selected_goal_xy_m": [1., 0.], "route_remaining_distance_m": 1.})
    solver_time_regressed.observe({"outer_tick": 2, "outer_tick_time_s": .04,
        "distinct_contact_substeps": 2, "last_contact_physics_time_s": .015,
        "last_contact_physics_substep": 5, "selected_goal_id": "g",
        "selected_goal_xy_m": [1., 0.], "route_remaining_distance_m": .9})
    assert solver_time_regressed.record()["first_route_progress_interval"] is None
    assert solver_time_regressed.record()["null_reason"] == \
        "solver_contact_time_or_substep_regressed"

    solver_step_regressed = FirstRouteProgressInterval()
    solver_step_regressed.observe({"outer_tick": 1, "outer_tick_time_s": .02,
        "distinct_contact_substeps": 1, "last_contact_physics_time_s": .015,
        "last_contact_physics_substep": 4, "selected_goal_id": "g",
        "selected_goal_xy_m": [1., 0.], "route_remaining_distance_m": 1.})
    solver_step_regressed.observe({"outer_tick": 2, "outer_tick_time_s": .04,
        "distinct_contact_substeps": 2, "last_contact_physics_time_s": .018,
        "last_contact_physics_substep": 4, "selected_goal_id": "g",
        "selected_goal_xy_m": [1., 0.], "route_remaining_distance_m": .9})
    assert solver_step_regressed.record()["first_route_progress_interval"] is None
    assert solver_step_regressed.record()["null_reason"] == \
        "solver_contact_time_or_substep_regressed"

    missing_count = FirstRouteProgressInterval()
    missing_count.observe({"outer_tick": 1, "outer_tick_time_s": .02,
        "distinct_contact_substeps": 1, "last_contact_physics_time_s": .015,
        "last_contact_physics_substep": 4, "selected_goal_id": "g",
        "selected_goal_xy_m": [1., 0.], "route_remaining_distance_m": 1.})
    missing_count.observe({"outer_tick": 2, "outer_tick_time_s": .04,
        "last_contact_physics_time_s": .015, "last_contact_physics_substep": 4,
        "selected_goal_id": "g", "selected_goal_xy_m": [1., 0.],
        "route_remaining_distance_m": .9})
    assert missing_count.record()["first_route_progress_interval"] is None
    assert missing_count.record()["null_reason"] == \
        "contact_substep_count_missing_or_invalid"

    non_integer_count = FirstRouteProgressInterval()
    non_integer_count.observe({"outer_tick": 1, "outer_tick_time_s": .02,
        "distinct_contact_substeps": 1.5, "selected_goal_id": "g",
        "selected_goal_xy_m": [1., 0.], "route_remaining_distance_m": 1.})
    assert non_integer_count.record()["first_route_progress_interval"] is None
    assert non_integer_count.record()["null_reason"] == \
        "contact_substep_count_missing_or_invalid"


def test_fake_contact_outer_step_sampler_records_truth_goal_route_and_solver_clock():
    class Sim:
        num_robots = 6
        _collision_pair_i = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
        _collision_pair_j = torch.tensor([1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5])
        pair_contact_event_count = torch.zeros((1, 15), dtype=torch.long)
        pair_contact_solver_report_count = torch.zeros((1, 15), dtype=torch.long)
        pair_contact_first_substep = torch.full((1, 15), -1, dtype=torch.long)
        pair_contact_last_substep = torch.full((1, 15), -1, dtype=torch.long)
        pair_contact_first_time_s = torch.full((1, 15), float("nan"), dtype=torch.float64)
        pair_contact_last_time_s = torch.full_like(pair_contact_first_time_s, float("nan"))
        pair_contact_impulse_sum = torch.zeros((1, 15, 4))
        pose = torch.zeros((1, 6, 3))
    sim = Sim()
    sim.pair_contact_event_count[0, 9] = 1
    sim.pair_contact_last_substep[0, 9] = 1
    sim.pair_contact_last_time_s[0, 9] = .02
    planner = type("Planner", (), {})()
    planner.last_path = torch.zeros((1, 6, 3, 2))
    planner.last_path[0, 3] = torch.tensor([[0., 0.], [.5, 0.], [1., 0.]])
    planner.last_lengths = torch.tensor([[0, 0, 0, 3, 0, 0]])
    planner.last_goal = torch.zeros((1, 6, 2))
    planner.last_goal[0, 3] = torch.tensor([1., 0.])
    planner.resolution = .2
    env = type("Env", (), {"n": 1, "sim": sim, "planner": planner,
        "steps": torch.tensor([1]), "dt": .02})()
    observer = FirstRouteProgressInterval()
    _observe_phase4_contact_route_progress(env, observer,
        selected_goal_id="chosen", selected_goal_xy_m=[1., 0.])
    sample = observer.record()["trace"][0]
    assert sample["contact_pair_slots"] == [2, 3]
    assert sample["contact_substep_delta"] == 1
    assert sample["last_contact_physics_time_s"] == .02
    assert sample["slot3_truth_xy_m"] == [0., 0.]
    assert sample["selected_goal_id"] == "chosen"
    assert sample["route_matches_selected_goal"] is True
    assert sample["route_goal_tolerance_m"] == pytest.approx(
        .2 * PHASE4_CONTACT_ROUTE_GOAL_TOLERANCE_RESOLUTION_FACTOR)
    assert sample["route_remaining_distance_m"] == pytest.approx(1.)
    first_route_fingerprint = sample["route_fingerprint_sha256"]
    env.steps[0] = 2
    planner.last_path[0, 3] = torch.tensor([[0., 0.], [0., .5], [1., 0.]])
    _observe_phase4_contact_route_progress(env, observer,
        selected_goal_id="chosen", selected_goal_xy_m=[1., 0.])
    second_sample = observer.record()["trace"][-1]
    assert second_sample["selected_goal_id"] == sample["selected_goal_id"]
    assert second_sample["route_fingerprint_sha256"] != first_route_fingerprint
