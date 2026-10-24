import json
from pathlib import Path

import pytest

from frc_defense.meco_phase4 import phase4_factor_config
from frc_defense.meco_phase4 import PHASE4_FACTORS
from frc_defense.meco_phase4_runtime_adapter import (
    PHASE4_REQUIREMENT_BINDINGS,
    phase4_factor_exposure,
    phase4_requirement_binding,
    validate_phase4_capture,
)


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
        assert set(binding["runtime_event_by_factor_id"]) == set(
            binding["runtime_factor_by_factor_id"])
        assert set(binding["runtime_factor_by_factor_id"].values()) <= set(PHASE4_FACTORS)
        assert not set(binding["required_signals"]).issubset(
            binding["authoritative_signals"])


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
        scenario_id="offense_vs_scripted_pressure", category="active/trench/level_1_stationary")
    assert capture["status"] == "realized_stimulus_outcomes_incomplete"
    assert capture["runtime_event"]["realized"] is True
    assert capture["required_signals_unavailable"]
    assert capture["qualification_eligible"] is False
    validate_phase4_capture(capture, expected_source_identity=source)
    with pytest.raises(ValueError, match="source identity mismatch"):
        validate_phase4_capture(capture, expected_source_identity={"commit": "different"})


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
