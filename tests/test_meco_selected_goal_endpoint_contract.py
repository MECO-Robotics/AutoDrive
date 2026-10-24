from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from frc_defense.meco_metric_ledger import MecoEpisodeMetricLedger
from frc_defense.meco_phase4_runtime_adapter import (
    phase4_selected_goal_endpoint_completion_record,
    phase4_route_completion_record,
    phase4_route_completion_campaign_record,
    phase4_route_condition,
    PHASE4_ROUTE_OUTCOME_POLICY_IDS,
    PHASE4_ROUTE_OUTCOME_FACTOR_IDS,
)
from frc_defense.meco_track_training import _phase4_route_goal_lifecycle_id


def test_route_factor_condition_binds_stationary_treatment_and_baseline():
    factor = {"kind": "stationary_blockage"}
    assert phase4_route_condition("blockage.stationary",
        phase4_factor=factor, contact_factor_id=None) == "factor_treatment"
    assert phase4_route_condition("opponent.feint_reversal",
        phase4_factor=factor, contact_factor_id=None) == "no_factor"


def _attempt(**overrides):
    row = {
        "macro_index": 0,
        "selected_goal_id": "goal-1", "goal_xy_m": [1.0, 2.0],
        "goal_lifecycle_id": "lifecycle-1",
        "goal_heading_required": False, "goal_heading_rad": None,
        "terminal_goal_distance_m": .03, "terminal_heading_error_rad": None,
        "terminal_pose_xy_m": [1.03, 2.0], "terminal_pose_heading_rad": 0.0,
        "execution_valid": True, "execution_exclusion_reasons": [],
        "episode_id": "episode-1", "source_identity_sha256": "a" * 64,
        "geometry_id": "crossing_contact",
    }
    row.update(overrides)
    distance, goal_xy = row.get("terminal_goal_distance_m"), row.get("goal_xy_m")
    if isinstance(distance, (int, float)) and isinstance(goal_xy, list):
        row["terminal_pose_xy_m"] = [goal_xy[0] + distance, goal_xy[1]]
    if (row.get("goal_heading_required") is True and
            isinstance(row.get("goal_heading_rad"), (int, float)) and
            isinstance(row.get("terminal_heading_error_rad"), (int, float))):
        row["terminal_pose_heading_rad"] = row["goal_heading_rad"] + row[
            "terminal_heading_error_rad"]
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


def test_route_completion_campaign_preserves_censored_attempts_and_exact_pair_coverage():
    provenance = {"source_identity_sha256": "a" * 64,
        "episode_id": "episode-1", "scenario_id": "scenario-1",
        "category": "active/trench/level_1",
        "execution_device": json.dumps({"device": "cuda:0", "backend": "cuda"},
            sort_keys=True, separators=(",", ":")),
        "checkpoint_sha256": "b" * 64, "condition_id": "condition-1"}
    expected = [
        {"policy_id": policy, "factor_id": factor, "case_id": "case-1",
         "seed": 5, "condition": condition, **provenance}
        for policy in PHASE4_ROUTE_OUTCOME_POLICY_IDS
        for factor in PHASE4_ROUTE_OUTCOME_FACTOR_IDS
        for condition in ("no_factor", "factor_treatment")
    ]
    attempts = []
    for index, assignment in enumerate(expected):
        event = ({"schema": "meco-evaluator-route-endpoint-v1",
            "source": "evaluator_terminal_slot3_truth_after_executed_macro",
            "macro_index": 2, "distance_m": .03, "heading_required": False,
            "heading_error_rad": None, "goal_xy_m": [1., 2.],
            "terminal_pose_xy_m": [1.03, 2.], "terminal_pose_heading_rad": 0.,
            "endpoint_rule": "distance <= 0.04 m and, when required, absolute heading error <= 0.1 rad"}
            if index == 0 else None)
        attempts.append({**assignment, "binding": assignment,
            "attempt_id": f"attempt-{index}",
            "route_attempt_id": f"attempt-{index}",
            "executed_action_trace": [2, 2, 3],
            "selected_goal_id": "goal-1", "route_id": "route-1",
            "goal_xy_m": [1., 2.], "goal_heading_rad": None,
            "goal_heading_required": False, "end_macro_index": 2,
            "terminal_goal_distance_m": .03,
            "terminal_heading_error_rad": None,
            "terminal_pose_xy_m": [1.03, 2.],
            "terminal_pose_heading_rad": 0.,
            "source_identity_sha256": "a" * 64,
            "outcome": ("completed" if index == 0 else None),
            "objective_event": event,
            "censor_reason": "objective_event_unavailable"})
    result = phase4_route_completion_campaign_record(attempts, expected)
    assert result["attempt_row_count"] == 88
    assert result["expected_policy_ids"] == list(PHASE4_ROUTE_OUTCOME_POLICY_IDS)
    assert result["required_factor_ids"] == list(PHASE4_ROUTE_OUTCOME_FACTOR_IDS)
    assert result["route_completion_rate"] is None
    assert result["authoritative_signal_available"] is False
    assert result["qualification_eligible"] is False
    assert sum(row["assigned"] for row in result["by_assignment"]) == 88
    assert sum(row["censored"] for row in result["by_assignment"]) == 87
    assert sum(row["completed"] for row in result["by_assignment"]) == 1
    assert all(row["route_completion_rate"] is None
               for row in result["by_assignment"] if row["censored"])


def test_route_completion_campaign_rejects_missing_paired_policy_condition():
    provenance = {"source_identity_sha256": "a" * 64,
        "episode_id": "episode-1", "scenario_id": "scenario-1",
        "category": "active/trench/level_1",
        "execution_device": json.dumps({"device": "cuda:0", "backend": "cuda"},
            sort_keys=True, separators=(",", ":")),
        "checkpoint_sha256": "b" * 64, "condition_id": "condition-1"}
    expected = [
        {"policy_id": policy, "factor_id": factor, "case_id": "case-1",
         "seed": 5, "condition": condition, **provenance}
        for policy in PHASE4_ROUTE_OUTCOME_POLICY_IDS
        for factor in PHASE4_ROUTE_OUTCOME_FACTOR_IDS
        for condition in ("no_factor", "factor_treatment")
    ]
    expected.pop()
    with pytest.raises(ValueError, match="exact policy/condition paired coverage"):
        phase4_route_completion_campaign_record([], expected)


def test_route_completion_campaign_keeps_unscheduled_contact_lane_missing():
    provenance = {"source_identity_sha256": "a" * 64,
        "episode_id": "episode-1", "scenario_id": "scenario-1",
        "category": "active/trench/level_1",
        "execution_device": json.dumps({"device": "cuda:0", "backend": "cuda"},
            sort_keys=True, separators=(",", ":")),
        "checkpoint_sha256": "b" * 64, "condition_id": "condition-1"}
    expected = [
        {"policy_id": policy, "factor_id": factor, "case_id": "case-1",
         "seed": 5, "condition": condition, **provenance}
        for policy in PHASE4_ROUTE_OUTCOME_POLICY_IDS
        for factor in PHASE4_ROUTE_OUTCOME_FACTOR_IDS
        for condition in ("no_factor", "factor_treatment")
    ]
    for row in expected:
        if row["factor_id"] == "contact.robot_bump" and row["condition"] == "factor_treatment":
            row.update(episode_id=None, provenance_status="treatment_not_scheduled",
                condition_id="missing:contact.robot_bump:treatment_not_scheduled")
    result = phase4_route_completion_campaign_record([], expected)
    assert result["route_completion_rate"] is None
    assert result["expected_assignment_count"] == 88
    assert len(result["assignments_without_route_lifecycle_records"]) == 88


def test_route_completion_campaign_recomputes_endpoint_event_evidence():
    provenance = {"source_identity_sha256": "a" * 64,
        "episode_id": "episode-1", "scenario_id": "scenario-1",
        "category": "active/trench/level_1",
        "execution_device": json.dumps({"device": "cuda:0", "backend": "cuda"},
            sort_keys=True, separators=(",", ":")),
        "checkpoint_sha256": "b" * 64, "condition_id": "condition-1"}
    expected = [{"policy_id": policy, "factor_id": factor, "case_id": "case-1",
        "seed": 5, "condition": condition, **provenance}
        for policy in PHASE4_ROUTE_OUTCOME_POLICY_IDS
        for factor in PHASE4_ROUTE_OUTCOME_FACTOR_IDS
        for condition in ("no_factor", "factor_treatment")]
    attempts = []
    for index, assignment in enumerate(expected):
        event = ({"schema": "meco-evaluator-route-endpoint-v1",
            "source": "evaluator_terminal_slot3_truth_after_executed_macro",
            "macro_index": 2, "distance_m": .001, "heading_required": False,
            "heading_error_rad": None, "goal_xy_m": [1., 2.],
            "terminal_pose_xy_m": [1.03, 2.], "terminal_pose_heading_rad": 0.,
            "endpoint_rule": "distance <= 0.04 m and, when required, absolute heading error <= 0.1 rad"}
            if index == 0 else None)
        attempts.append({**assignment, "binding": assignment,
            "attempt_id": f"attempt-{index}", "route_attempt_id": f"attempt-{index}",
            "executed_action_trace": [2], "selected_goal_id": "goal-1",
            "route_id": "route-1", "goal_xy_m": [1., 2.],
            "goal_heading_rad": None, "goal_heading_required": False,
            "end_macro_index": 2, "terminal_goal_distance_m": .03,
            "terminal_heading_error_rad": None,
            "terminal_pose_xy_m": [1.03, 2.], "terminal_pose_heading_rad": 0.,
            "source_identity_sha256": "a" * 64,
            "outcome": "completed" if index == 0 else None,
            "objective_event": event})
    result = phase4_route_completion_campaign_record(attempts, expected)
    assert result["route_completion_rate"] is None
    assert sum(row["completed"] for row in result["by_assignment"]) == 0
    assert sum(row["censored"] for row in result["by_assignment"]) == 88


def test_goal_ledger_keeps_actual_action_trace_and_does_not_guess_route_outcome():
    ledger = MecoEpisodeMetricLedger(SimpleNamespace())
    ledger.observe_goal_attempt(action=2, route_id="route-1", goal_xy=(1., 2.),
        goal_heading=None, terminal_goal_distance_m=.01,
        terminal_heading_error_rad=None, heading_alignment_opportunity=False,
        selected_goal_id="goal-1", executed_action_trace=[2, 2, 3])
    row = ledger.to_record()["goal_attempts"][0]
    assert row["executed_action_trace"] == [2, 2, 3]
    assert row["route_outcome_status"] == "censored"
    assert row["route_outcome_censor_reason"] == (
        "registered_route_completion_objective_event_unavailable")


def test_route_completion_record_groups_macro_rows_into_stable_lifecycles():
    binding = {"episode_id": "episode-1", "policy_id": "A1",
        "scenario_id": "full_match_offense", "category": "active/trench/level_1",
        "case_id": "case-sha", "seed": 5, "split": "validation",
        "factor_id": "contact.robot_bump", "condition": "factor_treatment",
        "source_identity_sha256": "a" * 64}
    rows = [
        _attempt(route_id="route-1", executed_action_trace=[2, 2],
            terminal_goal_distance_m=.12),
        _attempt(route_id="route-1", executed_action_trace=[3],
            terminal_goal_distance_m=.03),
        _attempt(selected_goal_id="goal-2", goal_lifecycle_id="lifecycle-2", route_id="route-2",
            executed_action_trace=[2, 3], terminal_goal_distance_m=.08),
        _attempt(selected_goal_id="goal-3", goal_lifecycle_id="lifecycle-3", route_id="route-3",
            execution_valid=False, execution_exclusion_reasons=["action_remasked"],
            executed_action_trace=[2, 7], terminal_goal_distance_m=.01),
    ]
    for row in rows:
        row["route_id"] = row.get("route_id", "route-default")
    for index, row in enumerate(rows):
        row["macro_index"] = index
        row["executed_action_trace"] = row.get("executed_action_trace", [2])
    result = phase4_route_completion_record(rows, binding=binding)
    assert result["attempted"] == 3
    assert result["completed"] == 1
    assert result["observed_failure"] == 0
    assert result["censored"] == 2
    assert result["route_completion_rate"] is None
    assert result["attempts"][0]["executed_action_trace"] == [2, 2, 3]
    assert result["attempts"][0]["outcome"] == "completed"
    assert "route_replaced_before_completion" in result["attempts"][1]["censor_reasons"]
    assert "action_remasked" in result["attempts"][2]["censor_reasons"]


def test_route_completion_record_marks_native_d0_lifecycle_capture_unsupported():
    binding = {"episode_id": "episode-1", "policy_id": "D0",
        "scenario_id": "full_match_offense", "category": "active/trench/level_1",
        "case_id": "case-sha", "seed": 5, "split": "validation",
        "factor_id": "contact.robot_bump", "condition": "factor_treatment",
        "source_identity_sha256": "a" * 64}
    result = phase4_route_completion_record([], binding=binding)
    assert result["attempted"] == 0
    assert result["route_completion_rate"] is None
    assert result["policy_support_status"] == (
        "unsupported_native_d0_has_no_goal_attempt_ledger")
    assert result["authoritative_signal_available"] is False
    assert result["qualification_eligible"] is False


def test_route_completion_censors_macro_gaps_and_distinguishes_missing_ledger():
    binding = {"episode_id": "episode-1", "policy_id": "A1",
        "scenario_id": "full_match_offense", "category": "active/trench/level_1",
        "case_id": "case-sha", "seed": 5, "split": "validation",
        "factor_id": "contact.robot_bump", "condition": "factor_treatment",
        "source_identity_sha256": "a" * 64}
    first = _attempt(macro_index=4, terminal_goal_distance_m=.2,
                     route_id="route-1", executed_action_trace=[2])
    endpoint = _attempt(macro_index=6, terminal_goal_distance_m=.01,
                        route_id="route-1", executed_action_trace=[3])
    gapped = phase4_route_completion_record([first, endpoint], binding=binding)
    assert gapped["attempted"] == 2
    assert gapped["completed"] == 1
    assert gapped["censored"] == 1
    assert "macro_index_gap_or_invalid" in gapped["attempts"][0]["censor_reasons"]
    assert gapped["route_completion_rate"] is None

    missing = phase4_route_completion_record(None, binding=binding)
    empty = phase4_route_completion_record([], binding=binding)
    malformed = phase4_route_completion_record({}, binding=binding)
    assert missing["ledger_input_status"] == "goal_ledger_unavailable"
    assert empty["ledger_input_status"] == "goal_ledger_available"
    assert empty["attempted"] == 0
    assert malformed["ledger_input_status"] == "malformed_goal_ledger_input"


def test_route_id_replacement_censors_prior_lifecycle_at_same_goal_endpoint():
    binding = {"episode_id": "episode-1", "policy_id": "A1",
        "scenario_id": "full_match_offense", "category": "active/trench/level_1",
        "case_id": "case-sha", "seed": 5, "split": "validation",
        "factor_id": "blockage.stationary", "condition": "factor_treatment",
        "source_identity_sha256": "a" * 64}
    rows = [
        _attempt(macro_index=0, route_id="stable-route-1",
            terminal_goal_distance_m=.2, executed_action_trace=[2]),
        _attempt(macro_index=1, route_id="stable-route-2",
            terminal_goal_distance_m=.01, executed_action_trace=[3]),
    ]
    result = phase4_route_completion_record(rows, binding=binding)
    assert result["attempted"] == 1
    assert result["attempts"][0]["outcome"] == "completed"
    assert result["attempts"][0]["route_ids_observed"] == [
        "stable-route-1", "stable-route-2"]


def test_goal_lifecycle_replacement_censors_even_when_route_and_endpoint_match():
    binding = {"episode_id": "episode-1", "policy_id": "A1",
        "scenario_id": "full_match_offense", "category": "active/trench/level_1",
        "case_id": "case-sha", "seed": 5, "split": "validation",
        "factor_id": "blockage.stationary", "condition": "factor_treatment",
        "source_identity_sha256": "a" * 64}
    rows = [
        _attempt(macro_index=0, goal_lifecycle_id="intent-a", route_id="node-1",
            terminal_goal_distance_m=.2, executed_action_trace=[2]),
        _attempt(macro_index=1, goal_lifecycle_id="intent-b", route_id="node-2",
            terminal_goal_distance_m=.01, executed_action_trace=[3]),
    ]
    result = phase4_route_completion_record(rows, binding=binding)
    assert result["attempted"] == 2
    assert result["attempts"][0]["outcome"] == "censored"
    assert "route_replaced_before_completion" in result["attempts"][0]["censor_reasons"]
    assert result["attempts"][1]["outcome"] == "completed"


def test_goal_switch_and_revisit_creates_three_lifecycle_segments():
    binding = {"episode_id": "episode-1", "policy_id": "A1",
        "scenario_id": "full_match_offense", "category": "active/trench/level_1",
        "case_id": "case-sha", "seed": 5, "split": "validation",
        "factor_id": "blockage.stationary", "condition": "factor_treatment",
        "source_identity_sha256": "a" * 64}
    lifecycle_ids = [_phase4_route_goal_lifecycle_id(action=7,
        plan_a_node_id=node, behavior_id=0, descriptor_id=-1)
        for node in (4, 19, 4)]
    rows = [_attempt(macro_index=i, goal_lifecycle_id=lifecycle,
        selected_goal_id=f"candidate-{i}", route_id=f"candidate-{i}",
        terminal_goal_distance_m=.2, executed_action_trace=[2])
        for i, lifecycle in enumerate(lifecycle_ids)]
    result = phase4_route_completion_record(rows, binding=binding)
    assert result["attempted"] == 3
    assert result["censored"] == 3
    assert all("route_replaced_before_completion" in row["censor_reasons"]
        for row in result["attempts"][:2])
    assert "episode_ended_before_completion" in result["attempts"][2]["censor_reasons"]


def test_plan_a_lifecycle_tracks_selected_goal_node_not_candidate_route_id():
    first = _phase4_route_goal_lifecycle_id(action=7, plan_a_node_id=4,
        behavior_id=0, descriptor_id=-1, goal_xy=(1.0, 2.0))
    same_goal_new_route_candidate = _phase4_route_goal_lifecycle_id(
        action=7, plan_a_node_id=4, behavior_id=0, descriptor_id=-1,
        goal_xy=(1.0, 2.0))
    switched_goal = _phase4_route_goal_lifecycle_id(action=7,
        plan_a_node_id=19, behavior_id=0, descriptor_id=-1,
        goal_xy=(2.0, 2.5))
    assert first == same_goal_new_route_candidate
    assert first != switched_goal


def test_reused_target_buffer_slot_has_no_stable_lifecycle_identity():
    first_generation = _phase4_route_goal_lifecycle_id(action=1,
        plan_a_node_id=4, behavior_id=1, descriptor_id=2,
        goal_xy=(1.0, 2.0))
    reused_slot_new_target = _phase4_route_goal_lifecycle_id(action=1,
        plan_a_node_id=19, behavior_id=1, descriptor_id=2,
        goal_xy=(8.0, 4.0))
    assert first_generation is None
    assert reused_slot_new_target is None
    binding = {"episode_id": "episode-1", "policy_id": "A1",
        "scenario_id": "full_match_offense", "category": "active/trench/level_1",
        "case_id": "case-sha", "seed": 5, "split": "validation",
        "factor_id": "blockage.stationary", "condition": "factor_treatment",
        "source_identity_sha256": "a" * 64}
    rows = [_attempt(macro_index=index, action=1, goal_lifecycle_id=None,
        route_id=f"collect:track:2:generation-{index}",
        terminal_goal_distance_m=.01, executed_action_trace=[1])
        for index in range(2)]
    result = phase4_route_completion_record(rows, binding=binding)
    assert result["attempted"] == result["completed"] == result["censored"] == 0
    assert result["unsupported_identity_row_count"] == 2
    assert result["unsupported_identity_rows"][0]["reason"] == (
        "stable_route_goal_identity_unavailable")
    assert result["policy_support_status"] == (
        "unsupported_goal_lifecycle_identity_missing")


def test_pending_assigned_target_goal_still_fails_closed_without_target_generation():
    lifecycle_id = _phase4_route_goal_lifecycle_id(action=1,
        plan_a_node_id=10, behavior_id=1, descriptor_id=2,
        goal_xy=(1.0, 2.0), assigned_goal_id="assigned-route-7")
    assert lifecycle_id is None
