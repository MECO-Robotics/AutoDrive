from copy import deepcopy
from types import SimpleNamespace

from frc_defense.meco_metric_ledger import MecoEpisodeMetricLedger
from frc_defense.meco_phase4_runtime_adapter import (
    phase4_selected_goal_endpoint_completion_record,
)


def _attempt(**overrides):
    row = {
        "selected_goal_id": "goal-1", "goal_xy_m": [1.0, 2.0],
        "goal_heading_required": False, "goal_heading_rad": None,
        "terminal_goal_distance_m": .03, "terminal_heading_error_rad": None,
        "execution_valid": True, "execution_exclusion_reasons": [],
        "episode_id": "episode-1", "source_identity_sha256": "a" * 64,
        "geometry_id": "crossing_contact",
    }
    row.update(overrides)
    return row


def test_selected_goal_endpoint_rate_counts_only_bound_clean_attempts():
    rows = [
        _attempt(),
        _attempt(selected_goal_id="goal-2", goal_heading_required=True,
            goal_heading_rad=1.0, terminal_goal_distance_m=.02,
            terminal_heading_error_rad=.08),
        _attempt(selected_goal_id="goal-3", goal_heading_required=True,
            goal_heading_rad=1.0, terminal_goal_distance_m=.01,
            terminal_heading_error_rad=.12),
        _attempt(execution_valid=False,
            execution_exclusion_reasons=["action_remasked"]),
        _attempt(selected_goal_id="goal-contradictory", execution_valid=True,
            execution_exclusion_reasons=["action_remasked"]),
        _attempt(episode_id="wrong-episode"),
        _attempt(terminal_goal_distance_m=None),
    ]
    result = phase4_selected_goal_endpoint_completion_record(rows,
        episode_id="episode-1", source_identity_sha256="a" * 64,
        geometry_id="crossing_contact")
    assert result["numerator"] == 2
    assert result["denominator"] == 3
    assert result["selected_goal_endpoint_completion_rate"] == 2 / 3
    assert result["selected_goal_ids"] == ["goal-1", "goal-2", "goal-3"]
    assert result["attempted_goal_rows"] == 7
    assert result["excluded_goal_rows"] == 4
    assert result["exclusions_by_reason"] == {
        "action_remasked": 2,
        "execution_validity_conflicts_with_exclusion_reasons": 1,
        "episode_source_or_geometry_binding_mismatch": 1,
        "terminal_endpoint_measurement_unavailable": 1,
    }
    assert result["route_completion_rate"] is None
    assert result["recovery_time_after_bump_s"] is None
    assert result["authoritative_signal_available"] is False
    assert result["qualification_eligible"] is False


def test_selected_goal_endpoint_requires_heading_when_selected_goal_has_heading():
    rows = [_attempt(goal_heading_required=True, goal_heading_rad=0.5,
                     terminal_heading_error_rad=None)]
    result = phase4_selected_goal_endpoint_completion_record(rows,
        episode_id="episode-1", source_identity_sha256="a" * 64,
        geometry_id="crossing_contact")
    assert result["numerator"] == 0
    assert result["denominator"] == 0
    assert result["selected_goal_endpoint_completion_rate"] is None
    assert result["exclusions_by_reason"] == {
        "required_heading_measurement_unavailable": 1}


def test_empty_or_unbound_attempts_return_null_rate_with_exclusion_reason():
    empty = phase4_selected_goal_endpoint_completion_record(None,
        episode_id="episode-1", source_identity_sha256="a" * 64,
        geometry_id="crossing_contact")
    assert empty["selected_goal_endpoint_completion_rate"] is None
    assert empty["empty_denominator_reason"] == "no_valid_bound_selected_goal_attempts"
    malformed = deepcopy(_attempt())
    malformed["selected_goal_id"] = None
    result = phase4_selected_goal_endpoint_completion_record([malformed],
        episode_id="episode-1", source_identity_sha256="a" * 64,
        geometry_id="crossing_contact")
    assert result["denominator"] == 0
    assert result["exclusions_by_reason"] == {"selected_goal_identity_invalid": 1}


def test_goal_attempt_ledger_preserves_binding_and_execution_exclusions():
    ledger = MecoEpisodeMetricLedger(SimpleNamespace())
    ledger.observe_goal_attempt(action=2, route_id="route-1", goal_xy=(1., 2.),
        goal_heading=.5, terminal_goal_distance_m=.02,
        terminal_heading_error_rad=.03, heading_alignment_opportunity=True,
        selected_goal_id="goal-1", episode_id="episode-1",
        source_identity_sha256="a" * 64, geometry_id="crossing_contact",
        execution_valid=False, execution_exclusion_reasons=("action_remasked",))
    row = ledger.to_record()["goal_attempts"][0]
    assert row["selected_goal_id"] == "goal-1"
    assert row["episode_id"] == "episode-1"
    assert row["source_identity_sha256"] == "a" * 64
    assert row["geometry_id"] == "crossing_contact"
    assert row["goal_heading_required"] is True
    assert row["execution_valid"] is False
    assert row["execution_exclusion_reasons"] == ["action_remasked"]
