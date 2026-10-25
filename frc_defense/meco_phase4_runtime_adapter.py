"""Phase 4 validation requirement/runtime/signal contract.

This adapter makes the boundary between a planned factor and an executed
stimulus explicit. It does not make a requirement executable unless its
stimulus and every required outcome signal have an authoritative runtime
source. Captures bind observations to the exact evaluator source identity.
"""
from __future__ import annotations

import hashlib
import json
import math

from .meco_phase4 import PHASE4_FACTORS, phase4_factor_digest


# Keep route endpoint matching aligned with evaluate_routes' grid tolerance.
PHASE4_CONTACT_ROUTE_GOAL_TOLERANCE_RESOLUTION_FACTOR = 0.72

# The full paired design includes the five tracks, three P=0 controls, and
# three deterministic controls. Keep this list aligned with the evaluator's
# exact comparison grid; aliases or partial policy sets are not accepted.
PHASE4_ROUTE_OUTCOME_POLICY_IDS = (
    "A1", "A2", "A3", "C1", "C2", "A1_P0", "A2_P0", "A3_P0",
    "D0", "D1", "D2")
PHASE4_ROUTE_OUTCOME_FACTOR_IDS = (
    "blockage.stationary", "opponent.feint_reversal",
    "opponent.abrupt_policy_change", "contact.robot_bump")
PHASE4_ROUTE_FACTOR_ID_BY_KIND = {
    "stationary_blockage": "blockage.stationary",
    "feinting_opponent": "opponent.feint_reversal",
    "abrupt_opponent_policy_change": "opponent.abrupt_policy_change",
}


def phase4_route_condition(factor_id: str, *, phase4_factor: dict | None,
                           contact_factor_id: str | None) -> str:
    """Bind paired route outcome rows to the scheduled treatment condition."""
    active_factor_id = contact_factor_id
    if active_factor_id is None and isinstance(phase4_factor, dict):
        active_factor_id = PHASE4_ROUTE_FACTOR_ID_BY_KIND.get(
            phase4_factor.get("kind"))
    return "factor_treatment" if factor_id == active_factor_id else "no_factor"


_COMMON = {
    "runtime_event": None,
    "runtime_factor_kind": None,
    "runtime_factor_by_factor_id": {},
    "runtime_event_by_factor_id": {},
    "required_signals": (),
    "authoritative_signals": (),
    "stimulus_supported": False,
    "limitations": (),
}

# These rows mirror all requirements in qualifications/meco/phase4_design_registry.json.
# A helper or generated axis is recorded as available scaffolding, not runtime.
PHASE4_REQUIREMENT_BINDINGS = {
    "stationary_blockage": {**_COMMON,
        "factor_ids": ("blockage.stationary",),
        "runtime_factor_kind": "stationary_blockage",
        "runtime_factor_by_factor_id": {"blockage.stationary": "stationary_blockage"},
        "runtime_event": "phase4_runtime_metrics.stationary_blockage_assigned",
        "runtime_event_by_factor_id": {"blockage.stationary":
            "phase4_runtime_metrics.stationary_blockage_assigned"},
        "required_signals": ("route_completion_rate", "time_lost_to_blockage_s", "successful_alternate_crossing_rate"),
        "stimulus_supported": True,
        "limitations": ("registered slot-0 blocker pose/action is wired, but simulator placement and blockage outcome remain unvalidated", "all three required outcome signals remain non-authoritative")},
    "direct_pursuit": {**_COMMON,
        "factor_ids": ("opponent.direct_pursuit",),
        "runtime_factor_kind": "direct_pursuit",
        "runtime_factor_by_factor_id": {"opponent.direct_pursuit": "direct_pursuit"},
        "runtime_event": "records[].phase4_direct_pursuit",
        "runtime_event_by_factor_id": {"opponent.direct_pursuit":
            "records[].phase4_direct_pursuit.contact_events+objective_route.progress_intervals"},
        "required_signals": ("interception_rate", "time_lost_to_opponents_s", "robot_contact_rate"),
        "authoritative_signals": ("interception_rate", "time_lost_to_opponents_s", "robot_contact_rate"),
        "stimulus_supported": True,
        "assignment_runtime_event": "phase4_opponent_schedule",
        "assignment_applied_event": "records[].phase4_direct_pursuit.applied_slots",
        "limitations": ("simulator-grounded events and paired fixed-objective time loss are captured, but no held-out simulator replay/dashboard review evidence has been accepted; executable promotion remains fail-closed",)},
    "trench_interception": {**_COMMON,
        "factor_ids": ("opponent.trench_interception",),
        "runtime_factor_kind": "trench_interception",
        "runtime_factor_by_factor_id": {
            "opponent.trench_interception": "trench_interception"},
        "runtime_event": "records[].phase4_trench_interception",
        "runtime_event_by_factor_id": {"opponent.trench_interception":
            "records[].phase4_trench_interception.contact_events+crossing_events"},
        "assignment_runtime_event": "phase4_opponent_schedule",
        "assignment_applied_event": "records[].phase4_opponent_policy.runtime.applied_slots",
        "assignment_path_supported": True,
        "required_signals": ("interception_rate", "alternate_crossing_success_rate", "trench_crossing_time_s"),
        "authoritative_signals": ("interception_rate", "alternate_crossing_success_rate", "trench_crossing_time_s"),
        "stimulus_supported": True,
        "qualification_eligible": False,
        "limitations": ("the 14-category trench domain remains unqualified; this tranche defines authoritative record validators but has not been validated in a representative simulator replay", "the candidate cell is limited to active/trench/level_4_reactive with slot 0 and requires exact 11-policy paired coverage plus fresh replay/dashboard review before candidate executability", "qualification remains disabled")},
    "feints": {**_COMMON,
        "factor_ids": ("opponent.feint_reversal",),
        "runtime_factor_kind": "feinting_opponent",
        "runtime_factor_by_factor_id": {"opponent.feint_reversal": "feinting_opponent"},
        "runtime_event": "phase4_runtime_metrics.feint_reversal_tick",
        "runtime_event_by_factor_id": {"opponent.feint_reversal":
            "phase4_runtime_metrics.feint_reversal_tick"},
        "required_signals": ("reaction_latency_after_reversal_s", "interception_rate", "route_completion_rate"),
        "authoritative_signals": (),
        "stimulus_supported": True,
        "limitations": ("reversal event is captured, but required reaction/interception/route signals are unavailable",)},
    "reactive_defenders_and_opponent_pools": {**_COMMON,
        "factor_ids": ("defender.reactive_pool",),
        "runtime_event": None,
        "required_signals": ("interception_rate", "policy_family_coverage", "paired_outcome_by_family"),
        "limitations": ("deterministic policy IDs and sensed-track command hook are bound to validation schedules, but no representative simulator run has validated slot application", "interception and paired policy-family outcome signals unavailable")},
    "multiple_legal_team_start_states": {**_COMMON,
        "factor_ids": ("start.legal_team_poses",),
        "runtime_factor_kind": "legal_team_start_profile",
        "runtime_factor_by_factor_id": {
            "start.legal_team_poses": "legal_team_start_profile"},
        "runtime_event": "phase4_start_profile.profile_applied_before_observation",
        "runtime_event_by_factor_id": {
            "start.legal_team_poses":
                "phase4_start_profile.profile_applied_before_observation"},
        "stimulus_supported": True,
        "required_signals": ("legal_start_validation_rate", "outcome_by_start_state"),
        "authoritative_signals": ("legal_start_validation_rate", "outcome_by_start_state"),
        "signal_sources": {
            "legal_start_validation_rate":
                "records[].phase4_start_profile.per_world_legal + validated_pose_world_count; recomputed across this profile cell as sum(legal booleans)/sum(world counts)",
            "outcome_by_start_state":
                "paired validation report metrics reconciled by exact policy/category/seed against scenario_defined and this assigned registry_profile_id only"},
        "candidate_cell_gate": {
            "profile_scope": "one registry_profile_id per candidate cell",
            "every_expected_world_must_be_legal": True,
            "required_state_ids": ("scenario_defined", "assigned_registry_profile_id"),
            "qualification_eligible": False},
        "limitations": ("registered full-team poses are applied after reset and before perception; legal-start rate and stratified outcomes still lack authoritative simulator signals",)},
    "teammate_behaviors": {**_COMMON,
        "factor_ids": ("teammate.behavior_pool",),
        "runtime_event": None,
        "required_signals": ("outcome_by_teammate_behavior", "teammate_collision_rate", "score_contribution"),
        "limitations": ("teammate schedule helper is not called in validation", "behavior-specific outcomes, collision, and match-point contribution unavailable")},
    "sensor_delay_and_low_confidence": {**_COMMON,
        "factor_ids": ("sensor.delay_120ms", "sensor.low_confidence"),
        "runtime_factor_kind": "observation_delay|independent_low_confidence",
        "runtime_factor_by_factor_id": {
            "sensor.delay_120ms": "observation_delay",
            "sensor.low_confidence": "independent_low_confidence"},
        "runtime_event": "phase4_runtime_metrics.delayed_observation_ticks|confidence_cap_refresh_count",
        "runtime_event_by_factor_id": {
            "sensor.delay_120ms": "phase4_runtime_metrics.delayed_observation_ticks",
            "sensor.low_confidence": "phase4_runtime_metrics.confidence_cap_refresh_count"},
        "required_signals": ("outcome_by_sensor_condition", "stale_observation_reaction_latency_s", "outcome_by_confidence_band"),
        "stimulus_supported": True,
        "limitations": ("delay and confidence exposure are captured, but required outcome-by-condition, latency, and confidence-band signals are unavailable",)},
    "abrupt_opponent_change": {**_COMMON,
        "factor_ids": ("opponent.abrupt_policy_change",),
        "runtime_factor_kind": "abrupt_opponent_policy_change",
        "runtime_factor_by_factor_id": {
            "opponent.abrupt_policy_change": "abrupt_opponent_policy_change"},
        "runtime_event": "phase4_runtime_metrics.policy_change_tick",
        "runtime_event_by_factor_id": {"opponent.abrupt_policy_change":
            "phase4_runtime_metrics.policy_change_tick"},
        "required_signals": ("reaction_latency_after_switch_s", "interception_rate", "route_completion_rate"),
        "stimulus_supported": True,
        "limitations": ("switch event is captured, but required reaction/interception/route signals are unavailable",)},
    "bumps_and_contact": {**_COMMON,
        "factor_ids": ("contact.robot_bump",),
        "runtime_event": "phase4_pair_contact_event",
        "runtime_event_by_factor_id": {
            "contact.robot_bump": "phase4_pair_contact_event"},
        "required_signals": ("safety_robot_contact_rate", "recovery_time_after_bump_s", "route_completion_rate"),
        # The schema has an aggregate robot-contact metric, but it is not
        # bump-specific, so it is not authoritative for this requirement.
        "authoritative_signals": (),
        "limitations": ("registered contact geometry and sensed slot-2 approach are wired, but no representative simulator replay has validated contact geometry; pair-contact event summary is source-bound; recovery and route completion outcomes remain unavailable; qualification remains disabled")},
    "observation_only_focal_pose_mislocalization": {**_COMMON,
        "factor_ids": ("observation.focal_pose_offset",),
        "assignment_runtime_event": "phase4_focal_pose_assignment",
        "assignment_applied_event": "records[].phase4_focal_pose.applied_before_policy_observation",
        "assignment_path_supported": True,
        "required_signals": ("relocalization_time_s", "pose_error_recovery_rate", "safety_contact_rate"),
        "authoritative_signals": (),
        "qualification_eligible": False,
        "limitations": ("focal pose uses a separate protocol-bound assignment/report path and is not supported by generic PHASE4_FACTORS capture", "registered paired no-offset/offset assignment is applied after final scenario reset before perception and the first policy observation; physical simulator state is unchanged", "truth-versus-estimate diagnostics are initial-only evaluator records; no representative simulator validation, estimator correction, or later relocalization event exists", "relocalization, pose-error recovery, and safety contact outcomes are unavailable; qualification remains disabled")},
}


def phase4_requirement_binding(requirement_id: str) -> dict:
    """Return a copy-like JSON-safe binding for one registered requirement."""
    try:
        row = PHASE4_REQUIREMENT_BINDINGS[requirement_id]
    except KeyError as exc:
        raise ValueError(f"unknown Phase 4 requirement: {requirement_id}") from exc
    return json.loads(json.dumps(row))


def requirement_ids_for_factor(factor: dict | None) -> tuple[str, ...]:
    if factor is None:
        return ()
    kind = factor.get("kind") if isinstance(factor, dict) else None
    result = []
    for requirement_id, row in PHASE4_REQUIREMENT_BINDINGS.items():
        if row["runtime_factor_kind"] and kind in row["runtime_factor_kind"].split("|"):
            result.append(requirement_id)
    return tuple(result)


def phase4_pair_contact_event_record(sim, *, factor_id: str = "contact.robot_bump",
                                     pair_slots: tuple[int, int] = (2, 3)) -> dict:
    """Read simulator-originated contact-pair events for the registered factor."""
    if factor_id != "contact.robot_bump":
        raise ValueError("pair-contact event reader requires contact.robot_bump")
    if (getattr(sim, "num_robots", None) != 6 or pair_slots != (2, 3) or
            not all(hasattr(sim, name) for name in (
                "pair_contact_event_count", "pair_contact_first_substep",
                "pair_contact_first_time_s", "pair_contact_last_substep",
                "pair_contact_last_time_s", "pair_contact_solver_report_count",
                "pair_contact_impulse_sum"))):
        raise ValueError("simulator does not expose the registered six-robot contact pair")
    pair_i, pair_j = sim._collision_pair_i.tolist(), sim._collision_pair_j.tolist()
    try:
        slot = next(index for index, pair in enumerate(zip(pair_i, pair_j))
                    if pair == pair_slots)
    except StopIteration as exc:
        raise ValueError("registered robot pair is absent from simulator collision slots") from exc
    counts = sim.pair_contact_event_count[:, slot].tolist()
    reports = sim.pair_contact_solver_report_count[:, slot].tolist()
    substeps = sim.pair_contact_first_substep[:, slot].tolist()
    times = sim.pair_contact_first_time_s[:, slot].tolist()
    last_substeps = sim.pair_contact_last_substep[:, slot].tolist()
    last_times = sim.pair_contact_last_time_s[:, slot].tolist()
    impulses = sim.pair_contact_impulse_sum[:, slot, :].tolist()
    return {
        "schema": "meco-phase4-pair-contact-event-v1",
        "factor_id": factor_id,
        "pair_slots": list(pair_slots),
        "event_source": "simulator_robot_pair_sat_contact_solver",
        "event_realized_by_world": [count > 0 for count in counts],
        "distinct_contact_substeps_by_world": counts,
        "contact_solver_reports_by_world": reports,
        "first_contact_physics_substep_by_world": [
            None if substep < 0 else substep for substep in substeps],
        "first_contact_physics_time_s_by_world": [
            None if count == 0 else time for count, time in zip(counts, times)],
        "last_contact_physics_substep_by_world": [
            None if substep < 0 else substep for substep in last_substeps],
        "last_contact_physics_time_s_by_world": [
            None if count == 0 else time for count, time in zip(counts, last_times)],
        "accumulated_impulse_by_world": [
            {"normal_n_s": value[0], "tangential_n_s": value[1],
             "net_on_second_robot_n_s": value[2:4]} for value in impulses],
        "recovery_time_after_bump_s": None,
        "route_completion_rate": None,
        "qualification_eligible": False,
        "outcome_status": "contact_event_available_recovery_and_route_outcomes_unavailable",
        "event_scope": "first_and_last_contact_substep_plus_distinct_substep_count_and_episode_impulse_sum; not a complete event stream",
    }


def phase4_selected_goal_endpoint_completion_record(goal_attempts, *, episode_id: str,
        source_identity_sha256: str, geometry_id: str) -> dict:
    """Summarize only correctly bound, cleanly executed selected-goal endpoints."""
    binding = {"episode_id": episode_id,
        "source_identity_sha256": source_identity_sha256,
        "geometry_id": geometry_id}
    valid_binding = all(isinstance(value, str) and value
                        for value in binding.values())
    exclusions: dict[str, int] = {}
    selected_goal_ids = set()
    complete_count = valid_count = attempted_rows = 0
    for row in goal_attempts if isinstance(goal_attempts, list) else []:
        attempted_rows += 1
        if not isinstance(row, dict):
            exclusions["malformed_attempt_row"] = exclusions.get("malformed_attempt_row", 0) + 1
            continue
        reasons = row.get("execution_exclusion_reasons")
        if row.get("execution_valid") is not True or reasons:
            reason_list = reasons if isinstance(reasons, list) and reasons else [
                "execution_validity_unavailable"]
            if row.get("execution_valid") is True and reasons:
                exclusions["execution_validity_conflicts_with_exclusion_reasons"] = (
                    exclusions.get(
                        "execution_validity_conflicts_with_exclusion_reasons", 0) + 1)
            for reason in set(str(value) for value in reason_list):
                exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        if not valid_binding or any(row.get(key) != value
                                    for key, value in binding.items()):
            exclusions["episode_source_or_geometry_binding_mismatch"] = (
                exclusions.get("episode_source_or_geometry_binding_mismatch", 0) + 1)
            continue
        goal_id = row.get("selected_goal_id")
        goal_xy = row.get("goal_xy_m")
        if (not isinstance(goal_id, str) or not goal_id or
                not isinstance(goal_xy, list) or len(goal_xy) != 2 or
                any(type(value) not in (int, float) or
                    not math.isfinite(value) for value in goal_xy)):
            exclusions["selected_goal_identity_invalid"] = (
                exclusions.get("selected_goal_identity_invalid", 0) + 1)
            continue
        distance = row.get("terminal_goal_distance_m")
        if type(distance) not in (int, float) or not math.isfinite(distance):
            exclusions["terminal_endpoint_measurement_unavailable"] = (
                exclusions.get("terminal_endpoint_measurement_unavailable", 0) + 1)
            continue
        heading_required = row.get("goal_heading_required")
        if type(heading_required) is not bool:
            exclusions["heading_requirement_unavailable"] = (
                exclusions.get("heading_requirement_unavailable", 0) + 1)
            continue
        goal_heading = row.get("goal_heading_rad")
        if heading_required and (type(goal_heading) not in (int, float) or
                                not math.isfinite(goal_heading)):
            exclusions["selected_goal_heading_invalid"] = (
                exclusions.get("selected_goal_heading_invalid", 0) + 1)
            continue
        heading_error = row.get("terminal_heading_error_rad")
        if heading_required and (type(heading_error) not in (int, float) or
                                 not math.isfinite(heading_error)):
            exclusions["required_heading_measurement_unavailable"] = (
                exclusions.get("required_heading_measurement_unavailable", 0) + 1)
            continue
        valid_count += 1
        selected_goal_ids.add(goal_id)
        if distance <= .04 and (not heading_required or heading_error <= .1):
            complete_count += 1
    return {
        "schema": "meco-phase4-selected-goal-endpoint-diagnostic-v1",
        "diagnostic_only": True,
        "observation_scope": "episode-wide selected-goal attempts during assigned contact geometry; not contact-caused recovery",
        "binding": binding,
        "contact_pair_slots": [2, 3],
        "selected_goal_ids": sorted(selected_goal_ids),
        "numerator": complete_count,
        "denominator": valid_count,
        "selected_goal_endpoint_completion_rate": (
            complete_count / valid_count if valid_count else None),
        "attempted_goal_rows": attempted_rows,
        "excluded_goal_rows": attempted_rows - valid_count,
        "exclusions_by_reason": exclusions,
        "heading_required_when": "the selected goal row carries a finite goal_heading_rad; completion requires absolute terminal heading error <= 0.1 rad",
        "endpoint_rule": "terminal distance to selected goal <= 0.04 m",
        "empty_denominator_reason": (None if valid_count else
            "no_valid_bound_selected_goal_attempts"),
        "route_completion_rate": None,
        "recovery_time_after_bump_s": None,
        "authoritative_signal_available": False,
        "qualification_eligible": False,
    }


def phase4_route_completion_record(goal_attempts, *, binding: dict):
    """Fold consecutive macro rows into stable route/goal lifecycles.

    A lifecycle completes only on the evaluator-truth joint endpoint event.
    Replacement, episode horizon, execution gaps, and measurement gaps censor
    unfinished lifecycles; no failure is inferred from abandonment or timeout.
    """
    required_binding = ("episode_id", "policy_id", "scenario_id", "category",
                        "case_id", "seed", "split", "factor_id", "condition",
                        "source_identity_sha256")
    if (not isinstance(binding, dict) or any(name not in binding for name in
            required_binding) or any(not isinstance(binding[name], str) or
            not binding[name] for name in required_binding if name != "seed") or
            type(binding.get("seed")) is not int or binding["seed"] < 0 or
            len(binding["source_identity_sha256"]) != 64 or any(
                char not in "0123456789abcdef" for char in
                binding["source_identity_sha256"])):
        raise ValueError("route completion has incomplete episode binding")
    ledger_status = ("goal_ledger_unavailable" if goal_attempts is None else
                     "goal_ledger_available" if isinstance(goal_attempts, list) else
                     "malformed_goal_ledger_input")
    attempts = goal_attempts if isinstance(goal_attempts, list) else []
    outcomes = []
    unsupported_identity_rows = []
    censored_reasons = {}
    completed = observed_failure = censored = 0
    current = None

    def close(reason=None):
        nonlocal current, completed, observed_failure, censored
        if current is None:
            return
        if reason:
            current["censor_reasons"].append(reason)
        current["censor_reasons"] = sorted(set(current["censor_reasons"]))
        if current["explicit_failure_event"] and not current["censor_reasons"]:
            current["outcome"] = "observed_failure"
            observed_failure += 1
        elif current["completion_event"] and not current["censor_reasons"]:
            current["outcome"] = "completed"
            completed += 1
        else:
            current["outcome"] = "censored"
            censored += 1
            if not current["censor_reasons"]:
                current["censor_reasons"].append("registered_objective_event_unavailable")
            for censor_reason in current["censor_reasons"]:
                censored_reasons[censor_reason] = censored_reasons.get(censor_reason, 0) + 1
        current["objective_event"] = ({
            "schema": "meco-evaluator-route-endpoint-v1",
            "source": "evaluator_terminal_slot3_truth_after_executed_macro",
            "macro_index": current["end_macro_index"],
            "distance_m": current.get("terminal_goal_distance_m"),
            "heading_required": current.get("goal_heading_required"),
            "heading_error_rad": current.get("terminal_heading_error_rad"),
            "goal_xy_m": current.get("goal_xy_m"),
            "terminal_pose_xy_m": current.get("terminal_pose_xy_m"),
            "terminal_pose_heading_rad": current.get("terminal_pose_heading_rad"),
            "endpoint_rule": "distance <= 0.04 m and, when required, absolute heading error <= 0.1 rad",
        } if current["completion_event"] else {
            "schema": "meco-registered-route-objective-failure-v1",
            "source": "registered_route_objective_failure_event",
        } if current["explicit_failure_event"] else None)
        current.pop("completion_event", None)
        current.pop("explicit_failure_event", None)
        current["binding"] = dict(binding)
        current["source_identity_sha256"] = binding["source_identity_sha256"]
        outcomes.append(current)
        current = None

    previous_macro_index = None
    for index, row in enumerate(attempts):
        row = row if isinstance(row, dict) else {}
        macro_index = row.get("macro_index")
        valid_macro_index = type(macro_index) is int and macro_index >= 0
        gap = (not valid_macro_index or (previous_macro_index is not None and
                macro_index != previous_macro_index + 1))
        if gap:
            close("macro_index_gap_or_invalid")
        previous_macro_index = macro_index if valid_macro_index else None
        goal_xy = row.get("goal_xy_m")
        goal_heading = row.get("goal_heading_rad")
        route_id = row.get("route_id")
        goal_lifecycle_id = row.get("goal_lifecycle_id")
        valid_xy = (isinstance(goal_xy, list) and len(goal_xy) == 2 and
                    all(type(value) in (int, float) and math.isfinite(value)
                        for value in goal_xy))
        valid_identity = (isinstance(goal_lifecycle_id, str) and goal_lifecycle_id and
                          isinstance(row.get("selected_goal_id"), str) and
                          row["selected_goal_id"] and isinstance(route_id, str) and route_id and
                          valid_xy and (goal_heading is None or
                          (type(goal_heading) in (int, float) and math.isfinite(goal_heading))))
        if not valid_identity:
            close("stable_route_goal_identity_unavailable")
            unsupported_identity_rows.append({
                "ledger_row_index": index,
                "macro_index": macro_index if valid_macro_index else None,
                "selected_goal_id": row.get("selected_goal_id"),
                "route_id": route_id,
                "reason": "stable_route_goal_identity_unavailable",
            })
            continue
        if current is not None and goal_lifecycle_id != current["goal_lifecycle_id"]:
            close("route_replaced_before_completion")
        if current is None:
            lifecycle_key = {"binding": binding, "goal_lifecycle_id": goal_lifecycle_id,
                             "start_macro_index": macro_index if valid_macro_index else index}
            lifecycle_id = hashlib.sha256(json.dumps(lifecycle_key,
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            current = {"route_attempt_id": lifecycle_id,
                "start_macro_index": macro_index if valid_macro_index else index,
                "end_macro_index": macro_index if valid_macro_index else index,
                "selected_goal_id": row.get("selected_goal_id"),
                "goal_lifecycle_id": goal_lifecycle_id,
                "route_id": route_id,
                "goal_xy_m": goal_xy,
                "goal_heading_rad": goal_heading,
                "executed_action_trace": [], "censor_reasons": [],
                "completion_event": False, "explicit_failure_event": False,
                "route_goal_identity": goal_lifecycle_id}
        current["selected_goal_id"] = row.get("selected_goal_id")
        current["goal_xy_m"] = goal_xy
        current["goal_heading_rad"] = goal_heading
        current["end_macro_index"] = macro_index if valid_macro_index else index
        if not valid_macro_index:
            current["censor_reasons"].append("macro_index_unavailable")
        if not isinstance(row.get("episode_id"), str) or row["episode_id"] != binding["episode_id"]:
            current["censor_reasons"].append("episode_binding_mismatch")
        if (not isinstance(row.get("source_identity_sha256"), str) or
                row["source_identity_sha256"] != binding["source_identity_sha256"]):
            current["censor_reasons"].append("source_binding_mismatch")
        current.setdefault("route_ids_observed", [])
        if isinstance(row.get("route_id"), str) and row["route_id"]:
            current["route_ids_observed"].append(row["route_id"])
            current["route_id"] = row["route_id"]
        else:
            current["censor_reasons"].append("planner_route_record_unavailable")
        if row.get("execution_valid") is not True:
            current["censor_reasons"].extend(row.get("execution_exclusion_reasons") or
                                               ["execution_validity_unavailable"])
        elif row.get("execution_exclusion_reasons"):
            current["censor_reasons"].append(
                "execution_validity_conflicts_with_exclusion_reasons")
        trace = row.get("executed_action_trace")
        if (not isinstance(trace, list) or not trace or any(
                type(action) is not int or action < 0 for action in trace)):
            current["censor_reasons"].append("executed_outer_action_trace_missing_or_invalid")
        else:
            current["executed_action_trace"].extend(trace)
        distance = row.get("terminal_goal_distance_m")
        heading_required = row.get("goal_heading_required")
        heading_error = row.get("terminal_heading_error_rad")
        terminal_xy = row.get("terminal_pose_xy_m")
        terminal_heading = row.get("terminal_pose_heading_rad")
        valid_terminal_xy = (isinstance(terminal_xy, list) and len(terminal_xy) == 2 and
            all(type(value) in (int, float) and math.isfinite(value) for value in terminal_xy))
        computed_distance = (math.dist(terminal_xy, goal_xy) if valid_terminal_xy and valid_xy
                             else None)
        computed_heading_error = None
        if (heading_required is True and valid_xy and
                type(goal_heading) in (int, float) and
                type(terminal_heading) in (int, float) and
                math.isfinite(goal_heading) and math.isfinite(terminal_heading)):
            computed_heading_error = abs(math.atan2(
                math.sin(terminal_heading - goal_heading),
                math.cos(terminal_heading - goal_heading)))
        valid_endpoint = (type(distance) in (int, float) and math.isfinite(distance) and
            type(heading_required) is bool and computed_distance is not None and
            math.isclose(distance, computed_distance, rel_tol=0., abs_tol=1e-6) and
            ((not heading_required and heading_error is None) or
             (heading_required and computed_heading_error is not None and
              type(heading_error) in (int, float) and math.isfinite(heading_error) and
              math.isclose(heading_error, computed_heading_error, rel_tol=0., abs_tol=1e-6))))
        if not valid_endpoint:
            current["censor_reasons"].append("terminal_endpoint_measurement_unavailable")
        elif computed_distance <= .04 and (not heading_required or computed_heading_error <= .1):
            current["terminal_goal_distance_m"] = distance
            current["goal_heading_required"] = heading_required
            current["terminal_heading_error_rad"] = heading_error
            current["terminal_pose_xy_m"] = terminal_xy
            current["terminal_pose_heading_rad"] = terminal_heading
            current["completion_event"] = True
            close()
        # observed_failure is only accepted from a future explicitly registered
        # evaluator event; current runtime records do not emit one.
        elif row.get("route_failure_event_schema") == \
                "meco-registered-route-objective-failure-v1" and \
                row.get("route_failure_event") is True:
            current["explicit_failure_event"] = True
            close()
    close("episode_ended_before_completion")
    denominator = len(outcomes)
    fully_observed = denominator > 0 and censored == 0
    return {
        "schema": "meco-phase4-route-completion-attempts-v1",
        "binding": dict(binding),
        "objective": "stable selected route-goal lifecycle reaches within 0.04 m; when heading is required, absolute heading error <= 0.1 rad",
        "attempt_denominator_semantics": "consecutive macro rows sharing the stable evaluator goal_lifecycle_id form one lifecycle even when per-macro candidate or planner route IDs advance; lifecycle replacement, actual macro-index gaps, missing goal geometry, and episode horizon censor unfinished lifecycles",
        "unsupported_identity_row_count": len(unsupported_identity_rows),
        "unsupported_identity_rows": unsupported_identity_rows,
        "ledger_input_status": ledger_status,
        "goal_attempt_coverage_available": binding.get("goal_attempt_coverage_available") is True,
        "attempted": denominator, "completed": completed,
        "observed_failure": observed_failure, "censored": censored,
        "censored_reasons": censored_reasons,
        "route_completion_rate": (completed / denominator
            if fully_observed else None),
        "rate_null_reason": (None if fully_observed else
            "no_route_goal_lifecycles_or_one_or_more_censored_attempts"),
        "attempts": outcomes,
        "policy_support_status": (
            "unsupported_native_d0_has_no_goal_attempt_ledger"
            if binding["policy_id"] == "D0" and not attempts else
            "unsupported_goal_ledger_missing_or_malformed" if not isinstance(goal_attempts, list) else
            "unsupported_goal_attempt_coverage_incomplete" if binding.get("goal_attempt_coverage_available") is False else
            "unsupported_goal_lifecycle_identity_missing" if unsupported_identity_rows else
            "unsupported_no_route_goal_lifecycle_records" if not attempts else
            "evaluator_goal_attempt_records_available"),
        "authoritative_signal_available": False,
        "qualification_eligible": False,
    }


def phase4_route_completion_campaign_record(attempt_rows, expected_assignments):
    """Aggregate explicitly observed route outcomes without inventing labels.

    Each attempt must be bound to its paired design assignment and include the
    actual outer-action trace and planner goal identity. `outcome` is supplied
    only by an evaluator with a registered route-completion objective event;
    absent/invalid outcomes remain censored. A rate is emitted only when every
    assigned attempt in that aggregate has an observed completed/failure label.
    """
    if not isinstance(expected_assignments, list) or not expected_assignments:
        raise ValueError("route completion requires expected paired assignments")
    expected = {}
    binding_keys = ("policy_id", "factor_id", "case_id", "seed", "condition")
    provenance_keys = ("source_identity_sha256", "episode_id", "scenario_id",
                       "category", "execution_device", "checkpoint_sha256",
                       "condition_id")
    for row in expected_assignments:
        if not isinstance(row, dict) or any(key not in row for key in binding_keys):
            raise ValueError("expected route assignment has incomplete identity")
        key = tuple(row[key] for key in binding_keys)
        if (row["policy_id"] not in PHASE4_ROUTE_OUTCOME_POLICY_IDS or
                row["factor_id"] not in PHASE4_ROUTE_OUTCOME_FACTOR_IDS or
                row["condition"] not in ("no_factor", "factor_treatment") or
                not isinstance(row["case_id"], str) or not row["case_id"] or
                type(row["seed"]) is not int or row["seed"] < 0 or key in expected):
            raise ValueError("expected route assignment is invalid or duplicated")
        missing_provenance = []
        for name in provenance_keys:
            value = row.get(name)
            if (name == "episode_id" and value is None and
                    row.get("provenance_status") in (
                        "treatment_not_scheduled", "record_missing")):
                continue
            if (name == "checkpoint_sha256" and value is None and
                    row.get("policy_id") in ("D0", "D1", "D2")):
                continue
            if not isinstance(value, str) or not value:
                missing_provenance.append(name)
        if missing_provenance:
            raise ValueError("expected route assignment lacks full provenance")
        try:
            device_binding = json.loads(row["execution_device"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("expected route assignment has unnormalized device provenance") from exc
        if (not isinstance(device_binding, dict) or
                json.dumps(device_binding, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) != row["execution_device"]):
            raise ValueError("expected route assignment has unnormalized device provenance")
        for digest_name in ("source_identity_sha256", "checkpoint_sha256"):
            digest = row[digest_name]
            if digest is None and row.get("policy_id") in ("D0", "D1", "D2"):
                continue
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("expected route assignment has invalid provenance digest")
        expected[key] = 0
    present_policy_ids = {row["policy_id"] for row in expected_assignments}
    if present_policy_ids != set(PHASE4_ROUTE_OUTCOME_POLICY_IDS):
        raise ValueError("route assignments do not cover all 11 registered policy IDs")
    present_factor_ids = {row["factor_id"] for row in expected_assignments}
    if present_factor_ids != set(PHASE4_ROUTE_OUTCOME_FACTOR_IDS):
        raise ValueError("route assignments do not cover all four route-outcome factors")
    paired_groups = {(row["factor_id"], row["case_id"], row["seed"])
                     for row in expected_assignments}
    for factor_id, case_id, seed in paired_groups:
        observed = {(row["policy_id"], row["condition"])
                    for row in expected_assignments
                    if (row["factor_id"], row["case_id"], row["seed"]) ==
                       (factor_id, case_id, seed)}
        required = {(policy_id, condition)
                    for policy_id in PHASE4_ROUTE_OUTCOME_POLICY_IDS
                    for condition in ("no_factor", "factor_treatment")}
        if observed != required:
            raise ValueError("route assignments lack exact policy/condition paired coverage")
    if not isinstance(attempt_rows, list):
        raise ValueError("route attempts must be a list")

    counts = {key: {"assigned": 0, "completed": 0,
                    "observed_failure": 0, "censored": 0,
                    "censor_reasons": {}} for key in expected}
    seen_attempts = set()
    malformed = 0
    for row in attempt_rows:
        if not isinstance(row, dict):
            malformed += 1
            continue
        key = tuple(row.get(name) for name in binding_keys)
        if key not in expected:
            raise ValueError("route attempt is outside the expected paired design")
        expected_row = next(item for item in expected_assignments
                            if tuple(item[name] for name in binding_keys) == key)
        bound = row.get("binding")
        if (not isinstance(bound, dict) or any(
                bound.get(name) != expected_row.get(name)
                for name in (*binding_keys, *provenance_keys))):
            raise ValueError("route attempt provenance does not match expected assignment")
        attempt_id = row.get("route_attempt_id", row.get("attempt_id"))
        if not isinstance(attempt_id, str) or not attempt_id or (key, attempt_id) in seen_attempts:
            raise ValueError("route attempt identity is missing or duplicated")
        seen_attempts.add((key, attempt_id))
        expected[key] += 1
        aggregate = counts[key]
        aggregate["assigned"] += 1
        # Require actual executed actions and the planner's goal record. These
        # fields do not themselves imply that the route was completed.
        trace = row.get("executed_action_trace")
        reasons = []
        if not isinstance(trace, list) or not trace or any(
                type(action) is not int or action < 0 for action in trace):
            reasons.append("executed_outer_action_trace_missing_or_invalid")
        goal_id, route_id = row.get("selected_goal_id"), row.get("route_id")
        source_hash = row.get("source_identity_sha256")
        if (not isinstance(goal_id, str) or not goal_id or
                not isinstance(route_id, str) or not route_id or
                not isinstance(source_hash, str) or len(source_hash) != 64 or
                any(char not in "0123456789abcdef" for char in source_hash)):
            reasons.append("planner_goal_or_source_binding_missing_or_invalid")
        outcome = row.get("outcome")
        event = row.get("objective_event")
        valid_objective_event = False
        if (outcome == "completed" and isinstance(event, dict) and
                event.get("schema") == "meco-evaluator-route-endpoint-v1" and
                event.get("source") == "evaluator_terminal_slot3_truth_after_executed_macro"):
            goal_xy = row.get("goal_xy_m")
            terminal_xy = event.get("terminal_pose_xy_m")
            distance = event.get("distance_m")
            heading_required = event.get("heading_required")
            heading_error = event.get("heading_error_rad")
            terminal_heading = event.get("terminal_pose_heading_rad")
            goal_heading = row.get("goal_heading_rad")
            geometry_valid = (isinstance(goal_xy, list) and len(goal_xy) == 2 and
                isinstance(terminal_xy, list) and len(terminal_xy) == 2 and
                all(type(value) in (int, float) and math.isfinite(value)
                    for value in (*goal_xy, *terminal_xy)) and
                type(distance) in (int, float) and math.isfinite(distance))
            recomputed_distance = (math.dist(goal_xy, terminal_xy)
                if geometry_valid else None)
            recomputed_heading_error = None
            if (heading_required is True and type(goal_heading) in (int, float) and
                    type(terminal_heading) in (int, float) and
                    math.isfinite(goal_heading) and math.isfinite(terminal_heading)):
                recomputed_heading_error = abs(math.atan2(
                    math.sin(terminal_heading - goal_heading),
                    math.cos(terminal_heading - goal_heading)))
            event_rule_valid = (event.get("endpoint_rule") ==
                "distance <= 0.04 m and, when required, absolute heading error <= 0.1 rad")
            valid_objective_event = (geometry_valid and event_rule_valid and
                event.get("macro_index") == row.get("end_macro_index") and
                event.get("goal_xy_m") == goal_xy and
                event.get("terminal_pose_xy_m") == row.get("terminal_pose_xy_m") and
                event.get("terminal_pose_heading_rad") == row.get(
                    "terminal_pose_heading_rad") and
                event.get("distance_m") == row.get("terminal_goal_distance_m") and
                heading_required is row.get("goal_heading_required") and
                event.get("heading_error_rad") == row.get(
                    "terminal_heading_error_rad") and
                math.isclose(distance, recomputed_distance, rel_tol=0., abs_tol=1e-6) and
                distance <= .04 and
                ((heading_required is False and row.get("goal_heading_required") is False
                  and heading_error is None) or
                 (heading_required is True and row.get("goal_heading_required") is True and
                  type(heading_error) in (int, float) and
                  math.isclose(heading_error, recomputed_heading_error,
                               rel_tol=0., abs_tol=1e-6) and heading_error <= .1)))
        if outcome == "observed_failure":
            valid_objective_event = (isinstance(event, dict) and
                event.get("schema") == "meco-registered-route-objective-failure-v1")
        if outcome not in ("completed", "observed_failure") or not valid_objective_event:
            outcome = None
        if outcome not in ("completed", "observed_failure"):
            reasons.append(row.get("censor_reason") if isinstance(
                row.get("censor_reason"), str) and row["censor_reason"] else
                "registered_route_objective_event_unavailable")
        if reasons:
            aggregate["censored"] += 1
            for reason in reasons:
                aggregate["censor_reasons"][reason] = (
                    aggregate["censor_reasons"].get(reason, 0) + 1)
        else:
            aggregate[outcome] += 1

    if malformed:
        raise ValueError("route attempt list contains malformed rows without assignment identity")
    missing_assignments = [key for key, count in expected.items() if count == 0]
    by_assignment = []
    for key in sorted(counts, key=lambda value: tuple(str(v) for v in value)):
        row = dict(zip(binding_keys, key))
        values = counts[key]
        fully_observed = values["censored"] == 0
        row.update(values)
        row["route_lifecycle_records_missing"] = values["assigned"] == 0
        row["route_completion_rate"] = (
            values["completed"] / (values["completed"] + values["observed_failure"])
            if fully_observed and values["assigned"] else None)
        row["rate_null_reason"] = (None if row["route_completion_rate"] is not None
            else "one_or_more_attempts_censored_or_objective_event_unavailable")
        row["authoritative_signal_available"] = False
        row["qualification_eligible"] = False
        by_assignment.append(row)
    total_completed = sum(row["completed"] for row in by_assignment)
    total_failures = sum(row["observed_failure"] for row in by_assignment)
    total_censored = sum(row["censored"] for row in by_assignment)
    overall_observed = (not missing_assignments and total_censored == 0 and
                        total_completed + total_failures > 0)
    return {
        "schema": "meco-phase4-route-completion-campaign-diagnostic-v1",
        "attempt_outcome_semantics": "completed_or_observed_failure_only_from_registered_route_objective_event; otherwise censored",
        "expected_policy_ids": list(PHASE4_ROUTE_OUTCOME_POLICY_IDS),
        "required_factor_ids": list(PHASE4_ROUTE_OUTCOME_FACTOR_IDS),
        "expected_assignment_count": len(expected),
        "assignments_without_route_lifecycle_records": [list(key)
            for key in sorted(missing_assignments, key=lambda value: tuple(str(v) for v in value))],
        "attempt_row_count": len(attempt_rows),
        "by_assignment": by_assignment,
        "route_completion_rate": (total_completed / (total_completed + total_failures)
            if overall_observed else None),
        "null_reason": ("one_or_more_assignments_lack_route_lifecycle_records_or_censoring_exists"
            if not overall_observed
            else "representative simulator validation is not complete"),
        "authoritative_signal_available": False,
        "qualification_eligible": False,
    }


def route_remaining_distance_m(path_xy, pose_xy, goal_xy, *, goal_tolerance_m):
    """Return route distance after nearest polyline projection, or None if invalid."""
    try:
        points = [[float(v) for v in point] for point in path_xy]
        pose = [float(v) for v in pose_xy]
        goal = [float(v) for v in goal_xy]
        if (len(points) < 2 or len(pose) != 2 or len(goal) != 2 or
                any(len(point) != 2 for point in points) or
                any(not math.isfinite(v) for point in points for v in point) or
                any(not math.isfinite(v) for v in (*pose, *goal))):
            return None
        if math.dist(points[-1], goal) > goal_tolerance_m:
            return None
        lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
        best = None
        traversed = 0.
        best_remaining = None
        for index, (a, b) in enumerate(zip(points, points[1:])):
            dx, dy = b[0] - a[0], b[1] - a[1]
            denom = dx * dx + dy * dy
            ratio = 0. if denom == 0. else max(0., min(1.,
                ((pose[0] - a[0]) * dx + (pose[1] - a[1]) * dy) / denom))
            projected = (a[0] + ratio * dx, a[1] + ratio * dy)
            distance = math.dist(pose, projected)
            remaining = (traversed + lengths[index] * (1. - ratio) +
                         sum(lengths[index + 1:]))
            if best is None or distance < best:
                best, best_remaining = distance, remaining
            traversed += lengths[index]
        return float(best_remaining)
    except (TypeError, ValueError, OverflowError):
        return None


class FirstRouteProgressInterval:
    """Outer-tick interval-censored route progress observation; never qualification evidence."""
    def __init__(self):
        self.previous_contact_count = 0
        self.samples = []
        self.baseline = None
        self.result = None
        self.last_reason = "no_contact_observed"
        self.last_outer_tick = None
        self.last_outer_time = None
        self.last_solver_contact_time = None
        self.last_solver_contact_substep = None

    def observe(self, sample):
        sample = dict(sample)
        raw_count = sample.get("distinct_contact_substeps")
        count_invalid = type(raw_count) is not int or raw_count < 0
        count = None if count_invalid else raw_count
        outer_tick = sample.get("outer_tick")
        now = sample.get("outer_tick_time_s")
        monotonic_time = isinstance(now, (int, float)) and math.isfinite(now)
        monotonic_tick = type(outer_tick) is int
        if self.last_outer_tick is not None:
            monotonic_tick &= outer_tick > self.last_outer_tick
        if self.last_outer_time is not None:
            monotonic_time &= isinstance(now, (int, float)) and now > self.last_outer_time
        count_regressed = count is not None and count < self.previous_contact_count
        if count_regressed:
            count = None
        invalid_clock = not monotonic_tick or not monotonic_time
        invalid_sample = invalid_clock or count_regressed or count_invalid
        delta = 0 if count is None else max(0, count - self.previous_contact_count)
        if count is not None:
            self.previous_contact_count = count
        if monotonic_tick and type(outer_tick) is int:
            self.last_outer_tick = outer_tick
        if monotonic_time and isinstance(now, (int, float)) and math.isfinite(now):
            self.last_outer_time = float(now)
        sample["contact_substep_delta"] = delta
        if delta or self.baseline is not None or invalid_sample:
            self.samples.append(sample)
        contact_time = sample.get("last_contact_physics_time_s")
        route = sample.get("route_remaining_distance_m")
        goal = (sample.get("selected_goal_id"), sample.get("selected_goal_xy_m"))
        valid_clock = isinstance(now, (int, float)) and math.isfinite(now)
        if invalid_sample:
            self.baseline = None
            self.result = None
            self.last_reason = ("non_monotonic_outer_tick_or_time" if invalid_clock
                else "contact_substep_count_regressed" if count_regressed else
                "contact_substep_count_missing_or_invalid")
            sample["window_event"] = "window_censored_invalid_clock_or_contact_count"
            return
        if delta:
            self.result = None
            valid_contact_clock = (isinstance(contact_time, (int, float)) and
                math.isfinite(contact_time) and contact_time <= now)
            contact_substep = sample.get("last_contact_physics_substep")
            valid_contact_substep = type(contact_substep) is int and contact_substep >= 0
            solver_clock_regressed = (
                valid_contact_clock and valid_contact_substep and
                ((self.last_solver_contact_time is not None and
                  contact_time <= self.last_solver_contact_time) or
                 (self.last_solver_contact_substep is not None and
                  contact_substep <= self.last_solver_contact_substep)))
            if solver_clock_regressed:
                self.baseline = None
                self.last_reason = "solver_contact_time_or_substep_regressed"
                sample["window_event"] = "window_censored_solver_contact_clock_regressed"
                return
            if not valid_contact_clock or not valid_contact_substep:
                self.baseline = None
                self.last_reason = "invalid_solver_contact_clock"
                sample["window_event"] = "window_censored_invalid_solver_contact_clock"
                return
            self.last_solver_contact_time = float(contact_time)
            self.last_solver_contact_substep = contact_substep
            if (valid_clock and valid_contact_clock and
                    isinstance(route, (int, float)) and math.isfinite(route) and
                    goal[0] is not None):
                self.baseline = {"time": float(now), "contact_time": float(contact_time),
                                 "distance": float(route), "goal": goal}
                self.last_reason = "contact_window_open"
            else:
                self.baseline = None
                self.last_reason = ("no_valid_executable_route" if route is None else
                    "no_valid_solver_contact_clock" if not valid_contact_clock else
                    "no_valid_clock_or_goal")
            sample["window_event"] = "contact_baseline_reset"
            return
        if self.baseline is None:
            sample["window_event"] = None
            return
        if not valid_clock:
            self.baseline = None
            self.last_reason = "no_valid_clock"
            sample["window_event"] = "window_censored_no_valid_clock"
            return
        if goal != self.baseline["goal"]:
            self.baseline = None
            self.last_reason = "goal_changed"
            sample["window_event"] = "window_censored_goal_changed"
            return
        if not isinstance(route, (int, float)) or not math.isfinite(route):
            self.baseline = None
            self.last_reason = "no_valid_executable_route"
            sample["window_event"] = "window_censored_no_valid_executable_route"
            return
        if float(route) < self.baseline["distance"]:
            if self.result is None:
                self.result = {"contact_physics_time_lower_bound_s": self.baseline["contact_time"],
                    "baseline_route_sample_outer_tick_time_s": self.baseline["time"],
                    "route_progress_outer_tick_upper_bound_s": float(now),
                    "baseline_route_remaining_distance_m": self.baseline["distance"],
                    "progress_route_remaining_distance_m": float(route),
                    "selected_goal_id": goal[0], "selected_goal_xy_m": goal[1],
                    "semantics": "interval_censored_first_route_progress_after_contact; outer_tick_bounds_only"}
            self.last_reason = "interval_observed"
            self.baseline = None
            sample["window_event"] = "first_route_progress_interval_observed"
            return
        sample["window_event"] = None

    def record(self, *, trace_limit=None):
        reason = ("horizon_without_route_progress" if self.result is None and
                  self.baseline is not None else self.last_reason)
        trace = self.samples if trace_limit is None else self.samples[-trace_limit:]
        return {"first_route_progress_interval": self.result,
            "null_reason": None if self.result is not None else reason,
            "trace": trace, "trace_total_samples": len(self.samples),
            "trace_truncated": len(trace) < len(self.samples),
            "trace_clock": "outer_20ms_environment_ticks; contact time is solver summary timestamp",
            "recovery_time_after_bump_s": None,
            "authoritative_signal_available": False,
            "qualification_eligible": False}


def phase4_factor_exposure(factor: dict | None, runtime_metrics: dict | None,
                           metrics: dict | None, *, source_identity: dict,
                           split: str, seed: int, scenario_id: str,
                           category: str,
                           qualification_case: dict | None = None,
                           episode_id: str | None = None,
                           policy_id: str | None = None,
                           route_completion_by_factor: dict | None = None,
                           contact_factor_id: str | None = None) -> dict | None:
    """Build an immutable-provenance capture; missing/unrealized signals fail closed."""
    if factor is None:
        return None
    if factor.get("kind") not in PHASE4_FACTORS or PHASE4_FACTORS[factor["kind"]] != factor:
        raise ValueError("cannot capture an unregistered Phase 4 factor")
    if split not in ("validation", "test", "train", "calibration", "diagnostic"):
        raise ValueError("invalid Phase 4 capture split")
    if type(seed) is not int or seed < 0:
        raise ValueError("Phase 4 capture seed must be a non-negative integer")
    if not isinstance(source_identity, dict) or not source_identity:
        raise ValueError("Phase 4 capture requires evaluator source identity")
    if qualification_case is not None and (
            not isinstance(qualification_case, dict) or
            set(qualification_case) != {"layout_id", "defender_family",
                                        "start_condition", "sensor_profile"} or
            any(not isinstance(value, str) or not value
                for value in qualification_case.values())):
        raise ValueError("Phase 4 capture has invalid qualification case")
    runtime = runtime_metrics if isinstance(runtime_metrics, dict) else {}
    outcomes = metrics if isinstance(metrics, dict) else {}
    # Match the evaluator's established source_identity_sha256 serialization.
    source_hash = hashlib.sha256(json.dumps(source_identity, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    kind = factor["kind"]
    if kind == "feinting_opponent":
        event = runtime.get("feint_reversal_tick")
        event_expected = factor["reversal_tick"]
        event_field = "feint_reversal_tick"
        event_evidence = {}
    elif kind == "abrupt_opponent_policy_change":
        event = runtime.get("policy_change_tick")
        event_expected = factor["switch_tick"]
        event_field = "policy_change_tick"
        event_evidence = {}
    elif kind == "stationary_blockage":
        event = runtime.get("stationary_blockage_assigned")
        event_expected = True
        event_field = "stationary_blockage_assigned"
        pose_valid = runtime.get("stationary_blockage_pose_valid")
        if hasattr(pose_valid, "detach"):
            pose_valid = bool(pose_valid.detach().all().item())
        event_evidence = {"blocker_slot": factor["blocker_slot"],
            "blocker_x_m": factor["blocker_x_m"],
            "lane_binding": factor["lane_binding"],
            "pose_valid_for_full_episode": pose_valid,
            "pose_tolerance_m": factor["pose_tolerance_m"],
            "heading_tolerance_rad": factor["heading_tolerance_rad"]}
    elif kind == "observation_delay":
        event = runtime.get("delayed_observation_ticks", 0)
        event_expected = ">0"
        event_field = "delayed_observation_ticks"
        event_evidence = {}
    elif kind == "independent_low_confidence":
        event = runtime.get("confidence_cap_refresh_count", 0)
        event_expected = ">0"
        event_field = "confidence_cap_refresh_count"
        event_evidence = {
            "confidence_cap_applied": runtime.get("confidence_cap_applied"),
            "confidence_cap_reduced_track_values": runtime.get(
                "confidence_cap_reduced_track_values", 0)}
    else:  # guarded by validate_phase4_factor, kept fail-closed for future additions
        event, event_expected, event_field = None, None, None
    if kind in ("feinting_opponent", "abrupt_opponent_policy_change"):
        realized = type(event) is int and event == event_expected
    elif kind == "stationary_blockage":
        realized = event is True
        realized = (realized and event_evidence["pose_valid_for_full_episode"] is True)
    elif kind == "observation_delay":
        realized = type(event) is int and event > 0
    else:
        realized = (type(event) is int and event > 0 and
                    runtime.get("confidence_cap_applied") == factor["fuel_confidence_cap"] and
                    runtime.get("confidence_cap_reduced_track_values", 0) > 0)
    requirements = requirement_ids_for_factor(factor)
    bindings = [PHASE4_REQUIREMENT_BINDINGS[item] for item in requirements]
    required_signals = sorted({signal for row in bindings for signal in row["required_signals"]})
    authoritative_signals = sorted({signal for row in bindings for signal in row["authoritative_signals"]})
    # Capture only named metrics the evaluator actually emitted. A generic metric
    # is not promoted into a requirement-specific signal by this adapter.
    observed = {name: outcomes[name] for name in authoritative_signals if name in outcomes}
    unavailable = sorted(set(required_signals) - set(observed))
    route_factor_id = {
        "stationary_blockage": "blockage.stationary",
        "feinting_opponent": "opponent.feint_reversal",
        "abrupt_opponent_policy_change": "opponent.abrupt_policy_change",
    }.get(kind)
    route_observation = None
    if realized and route_factor_id is not None and route_completion_by_factor is not None:
        expected_condition = phase4_route_condition(route_factor_id,
            phase4_factor=factor, contact_factor_id=contact_factor_id)
        route_observation = _phase4_route_completion_observation(
            route_completion_by_factor.get(route_factor_id),
            expected_binding={"episode_id": episode_id,
                "policy_id": policy_id, "factor_id": route_factor_id,
                "condition": expected_condition,
                "source_identity_sha256": source_hash})
    return {
        "schema": "meco-phase4-realized-stimulus-capture-v1",
        "status": "realized_stimulus_outcomes_incomplete" if realized else "stimulus_not_realized",
        "qualification_eligible": False,
        "split": split,
        "scenario_id": scenario_id,
        "category": category,
        "qualification_case": qualification_case,
        "episode_id": episode_id,
        "policy_id": policy_id,
        "contact_factor_id": contact_factor_id,
        "seed": seed if split != "test" else None,
        "seed_redacted": split == "test",
        "source_identity": source_identity,
        "source_identity_sha256": source_hash,
        "factor": factor,
        "factor_sha256": phase4_factor_digest(factor),
        "requirement_ids": list(requirements),
        "runtime_event": {"field": event_field, "observed": event,
                          "expected": event_expected, "realized": realized,
                          "evidence": event_evidence},
        "authoritative_observed_signals": observed,
        "required_signals_unavailable": unavailable,
        # Route completion is preserved as a separately validated diagnostic.
        # The registry does not yet consider it authoritative for qualification.
        "route_outcome_observation": route_observation,
        "missing_runtime_sources": sorted({limitation for row in bindings
                                             for limitation in row["limitations"]}),
        "disclosure": {"contains_final_test_seed_assignments": False,
                       "seed_is_episode_seed_only": True},
    }


def _phase4_route_completion_observation(record: dict | None, *,
                                        expected_binding: dict) -> dict | None:
    """Accept only complete route lifecycle records with the exact episode binding."""
    if record is None:
        return None
    if not isinstance(record, dict) or record.get("schema") != \
            "meco-phase4-route-completion-attempts-v1":
        raise ValueError("route completion capture has invalid record schema")
    binding = record.get("binding")
    if (not isinstance(binding, dict) or any(value is None for value in
            expected_binding.values()) or any(binding.get(key) != value
            for key, value in expected_binding.items())):
        raise ValueError("route completion capture episode/source/factor/condition binding mismatch")
    attempts = record.get("attempts")
    attempted = record.get("attempted")
    completed = record.get("completed")
    failures = record.get("observed_failure")
    censored = record.get("censored")
    rate = record.get("route_completion_rate")
    if (not isinstance(attempts, list) or type(attempted) is not int or attempted <= 0 or
            len(attempts) != attempted or type(completed) is not int or completed < 0 or
            type(failures) is not int or failures < 0 or type(censored) is not int or
            censored != 0):
        return None
    observed_completed = observed_failures = 0
    for attempt in attempts:
        if (not isinstance(attempt, dict) or attempt.get("outcome") not in
                ("completed", "observed_failure") or attempt.get("censor_reasons") or
                attempt.get("binding") != binding or
                attempt.get("source_identity_sha256") !=
                    expected_binding["source_identity_sha256"]):
            return None
        event = attempt.get("objective_event")
        if attempt["outcome"] == "completed":
            if (not isinstance(event, dict) or event.get("schema") !=
                    "meco-evaluator-route-endpoint-v1" or event.get("source") !=
                    "evaluator_terminal_slot3_truth_after_executed_macro"):
                return None
            observed_completed += 1
        elif (not isinstance(event, dict) or event.get("schema") !=
                "meco-registered-route-objective-failure-v1" or event.get("source") !=
                "registered_route_objective_failure_event"):
            return None
        else:
            observed_failures += 1
    if (completed != observed_completed or failures != observed_failures or
            completed + failures != attempted or type(rate) not in (int, float) or
            not math.isfinite(rate) or not math.isclose(
                rate, observed_completed / attempted, rel_tol=0., abs_tol=1e-12)):
        return None
    return {"schema": "meco-phase4-route-outcome-observation-v1",
        "factor_id": expected_binding["factor_id"],
        "condition": expected_binding["condition"],
        "binding": dict(binding), "attempted": attempted,
        "completed": completed, "observed_failure": failures,
        "censored": 0, "route_completion_rate": rate,
        "attempts": attempts,
        "source_record_schema": record["schema"],
        "authoritative_signal_available": False,
        "qualification_eligible": False}


def validate_phase4_capture(capture: dict, *,
                            expected_source_identity: dict | None = None,
                            expected_source_identity_sha256: str | None = None,
                            expected_episode_seed: int | None = None,
                            expected_split: str | None = None,
                            expected_qualification_case: dict | None = None) -> None:
    """Validate exact source, factor, realization, split, episode, and signals."""
    if not isinstance(capture, dict) or capture.get("schema") != "meco-phase4-realized-stimulus-capture-v1":
        raise ValueError("invalid Phase 4 realization capture schema")
    embedded_source = capture.get("source_identity")
    if not isinstance(embedded_source, dict) or not embedded_source:
        raise ValueError("Phase 4 capture source identity is missing")
    embedded_hash = hashlib.sha256(json.dumps(embedded_source,
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    expected_hash = expected_source_identity_sha256
    if expected_source_identity is not None:
        if embedded_source != expected_source_identity:
            raise ValueError("Phase 4 capture source identity mismatch")
        expected_hash = hashlib.sha256(json.dumps(expected_source_identity,
            sort_keys=True, allow_nan=False).encode()).hexdigest()
    if expected_hash is None or capture.get("source_identity_sha256") != expected_hash or embedded_hash != expected_hash:
        raise ValueError("Phase 4 capture source digest mismatch")
    factor = capture.get("factor")
    if (not isinstance(factor, dict) or factor.get("kind") not in PHASE4_FACTORS or
            factor != PHASE4_FACTORS[factor["kind"]] or
            capture.get("factor_sha256") != phase4_factor_digest(factor)):
        raise ValueError("Phase 4 capture factor registration or digest mismatch")
    if capture.get("qualification_eligible") is not False:
        raise ValueError("runtime capture alone cannot claim qualification")
    if capture.get("status") != "realized_stimulus_outcomes_incomplete":
        raise ValueError("Phase 4 capture is not a realized, outcome-incomplete event")
    split = capture.get("split")
    if split not in ("validation", "test", "train", "calibration", "diagnostic"):
        raise ValueError("Phase 4 capture split is invalid")
    if expected_split is not None and split != expected_split:
        raise ValueError("Phase 4 capture split differs from its episode")
    seed = capture.get("seed")
    redacted = capture.get("seed_redacted")
    if split == "test":
        if seed is not None or redacted is not True:
            raise ValueError("test-split Phase 4 capture must redact its seed")
    elif type(seed) is not int or seed < 0 or redacted is not False:
        raise ValueError("non-test Phase 4 capture must contain its episode seed")
    if expected_episode_seed is not None and split != "test" and seed != expected_episode_seed:
        raise ValueError("Phase 4 capture seed differs from its episode")
    disclosure = capture.get("disclosure")
    if (not isinstance(disclosure, dict) or
            disclosure.get("contains_final_test_seed_assignments") is not False):
        raise ValueError("Phase 4 capture final-test disclosure metadata is invalid")
    case = capture.get("qualification_case")
    if case is not None and (
            not isinstance(case, dict) or
            set(case) != {"layout_id", "defender_family", "start_condition", "sensor_profile"} or
            any(not isinstance(value, str) or not value for value in case.values())):
        raise ValueError("Phase 4 capture qualification case is invalid")
    if expected_qualification_case is not None and case != expected_qualification_case:
        raise ValueError("Phase 4 capture qualification case differs from its episode")

    requirements = requirement_ids_for_factor(factor)
    if not requirements or capture.get("requirement_ids") != list(requirements):
        raise ValueError("Phase 4 capture requirement binding differs from its factor")
    rows = [PHASE4_REQUIREMENT_BINDINGS[item] for item in requirements]
    required = sorted({signal for row in rows for signal in row["required_signals"]})
    authoritative = sorted({signal for row in rows for signal in row["authoritative_signals"]})
    observed = capture.get("authoritative_observed_signals")
    unavailable = capture.get("required_signals_unavailable")
    if (not isinstance(observed, dict) or not all(isinstance(key, str) for key in observed) or
            not set(observed).issubset(authoritative) or
            not set(observed).issubset(required) or
            not isinstance(unavailable, list) or
            not all(isinstance(signal, str) for signal in unavailable) or
            len(unavailable) != len(set(unavailable)) or
            unavailable != sorted(set(required) - set(observed)) or not unavailable):
        raise ValueError("Phase 4 capture required/unavailable signal accounting is inconsistent")

    route_observation = capture.get("route_outcome_observation")
    route_factor_id = {"stationary_blockage": "blockage.stationary",
        "feinting_opponent": "opponent.feint_reversal",
        "abrupt_opponent_policy_change": "opponent.abrupt_policy_change"}.get(
            factor["kind"])
    if route_observation is not None:
        if route_factor_id is None:
            raise ValueError("Phase 4 route outcome is unsupported for this factor")
        expected_condition = phase4_route_condition(route_factor_id,
            phase4_factor=factor,
            contact_factor_id=capture.get("contact_factor_id"))
        expected_binding = {"episode_id": capture.get("episode_id"),
            "policy_id": capture.get("policy_id"), "factor_id": route_factor_id,
            "condition": expected_condition,
            "source_identity_sha256": embedded_hash}
        revalidated = _phase4_route_completion_observation({
            "schema": "meco-phase4-route-completion-attempts-v1",
            "binding": route_observation.get("binding"),
            "attempts": route_observation.get("attempts"),
            "attempted": route_observation.get("attempted"),
            "completed": route_observation.get("completed"),
            "observed_failure": route_observation.get("observed_failure"),
            "censored": route_observation.get("censored"),
            "route_completion_rate": route_observation.get("route_completion_rate")},
            expected_binding=expected_binding)
        if revalidated is None or any(route_observation.get(key) != value
                for key, value in revalidated.items()):
            raise ValueError("Phase 4 route outcome observation is incomplete or invalid")

    event = capture.get("runtime_event")
    if not isinstance(event, dict) or event.get("realized") is not True:
        raise ValueError("Phase 4 runtime stimulus was not realized")
    kind = factor["kind"]
    if kind == "feinting_opponent":
        expected_field, expected_value = "feint_reversal_tick", factor["reversal_tick"]
        valid_observed = type(event.get("observed")) is int and event["observed"] == expected_value
    elif kind == "abrupt_opponent_policy_change":
        expected_field, expected_value = "policy_change_tick", factor["switch_tick"]
        valid_observed = type(event.get("observed")) is int and event["observed"] == expected_value
    elif kind == "stationary_blockage":
        expected_field, expected_value = "stationary_blockage_assigned", True
        valid_observed = event.get("observed") is True
        evidence = event.get("evidence")
        valid_observed = (valid_observed and isinstance(evidence, dict) and
            evidence == {"blocker_slot": factor["blocker_slot"],
                "blocker_x_m": factor["blocker_x_m"],
                "lane_binding": factor["lane_binding"],
                "pose_valid_for_full_episode": True,
                "pose_tolerance_m": factor["pose_tolerance_m"],
                "heading_tolerance_rad": factor["heading_tolerance_rad"]})
    elif kind == "observation_delay":
        expected_field, expected_value = "delayed_observation_ticks", ">0"
        valid_observed = type(event.get("observed")) is int and event["observed"] > 0
    elif kind == "independent_low_confidence":
        expected_field, expected_value = "confidence_cap_refresh_count", ">0"
        evidence = event.get("evidence")
        valid_observed = (type(event.get("observed")) is int and event["observed"] > 0 and
            isinstance(evidence, dict) and
            evidence.get("confidence_cap_applied") == factor["fuel_confidence_cap"] and
            type(evidence.get("confidence_cap_reduced_track_values")) is int and
            evidence["confidence_cap_reduced_track_values"] > 0)
    else:
        raise ValueError("Phase 4 capture factor has no registered runtime event contract")
    if (event.get("field") != expected_field or event.get("expected") != expected_value or
            not valid_observed):
        raise ValueError("Phase 4 runtime event does not match the registered factor")
