import json
import copy
import hashlib
from pathlib import Path

import pytest
import torch

from frc_defense.meco_phase4 import phase4_factor_config
from frc_defense.meco_phase4 import PHASE4_FACTORS
from frc_defense.meco_phase4_runtime_adapter import (
    PHASE4_REQUIREMENT_BINDINGS,
    phase4_factor_exposure,
    phase4_pair_contact_event_record,
    phase4_requirement_binding,
    validate_phase4_capture,
)
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
        assert set(binding["runtime_factor_by_factor_id"].values()) <= set(PHASE4_FACTORS)
        assert not set(binding["required_signals"]).issubset(
            binding["authoritative_signals"])
    contact = phase4_requirement_binding("bumps_and_contact")
    assert contact["runtime_event_by_factor_id"] == {
        "contact.robot_bump": "phase4_pair_contact_event"}
    assert contact["stimulus_supported"] is False


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
    fixture.pair_contact_first_substep[0, pair_slot] = 8
    fixture.pair_contact_last_substep[0, pair_slot] = 10
    fixture.pair_contact_first_time_s[0, pair_slot] = .16
    fixture.pair_contact_last_time_s[0, pair_slot] = .20
    fixture.pair_contact_impulse_sum[0, pair_slot] = torch.tensor([3., -1., 2., 4.])
    record = phase4_pair_contact_event_record(fixture)
    assert record["factor_id"] == "contact.robot_bump"
    assert record["pair_slots"] == [2, 3]
    assert record["event_realized_by_world"] == [True]
    assert record["distinct_contact_substeps_by_world"] == [2]
    assert record["contact_solver_reports_by_world"] == [6]
    assert record["first_contact_physics_substep_by_world"] == [8]
    assert record["last_contact_physics_substep_by_world"] == [10]
    assert record["accumulated_impulse_by_world"] == [{
        "normal_n_s": 3., "tangential_n_s": -1.,
        "net_on_second_robot_n_s": [2., 4.]}]
    assert "not a complete event stream" in record["event_scope"]
    assert record["recovery_time_after_bump_s"] is None
    assert record["route_completion_rate"] is None
    with pytest.raises(ValueError, match="requires contact.robot_bump"):
        phase4_pair_contact_event_record(fixture, factor_id="contact.other")
