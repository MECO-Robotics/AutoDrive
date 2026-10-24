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
        "runtime_event": None,
        "required_signals": ("route_completion_rate", "time_lost_to_blockage_s", "successful_alternate_crossing_rate"),
        "limitations": ("no validation-run stationary-blockage assignment", "all three required signals unavailable")},
    "direct_pursuit": {**_COMMON,
        "factor_ids": ("opponent.direct_pursuit",),
        "runtime_event": None,
        "required_signals": ("interception_rate", "time_lost_to_opponents_s", "robot_contact_rate"),
        "limitations": ("opponent policy helper is not wired to validation slot assignment", "pursuit-specific interception and time-loss signals unavailable")},
    "trench_interception": {**_COMMON,
        "factor_ids": ("opponent.trench_interception",),
        "runtime_event": None,
        "required_signals": ("interception_rate", "alternate_crossing_success_rate", "trench_crossing_time_s"),
        "limitations": ("opponent policy helper is not wired to validation slot assignment", "all three required signals unavailable")},
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
        "runtime_event": None,
        "required_signals": ("legal_start_validation_rate", "outcome_by_start_state"),
        "limitations": ("legal start helpers are not applied by validation reset", "legal-start and stratified-outcome signals unavailable")},
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
        "runtime_event": None,
        "required_signals": ("relocalization_time_s", "pose_error_recovery_rate", "safety_contact_rate"),
        "authoritative_signals": (),
        "limitations": ("pose offset helper is not applied at validation observation boundary", "relocalization and pose-error recovery signals unavailable")},
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
                           qualification_case: dict | None = None) -> dict | None:
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
    return {
        "schema": "meco-phase4-realized-stimulus-capture-v1",
        "status": "realized_stimulus_outcomes_incomplete" if realized else "stimulus_not_realized",
        "qualification_eligible": False,
        "split": split,
        "scenario_id": scenario_id,
        "category": category,
        "qualification_case": qualification_case,
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
        "missing_runtime_sources": sorted({limitation for row in bindings
                                             for limitation in row["limitations"]}),
        "disclosure": {"contains_final_test_seed_assignments": False,
                       "seed_is_episode_seed_only": True},
    }


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
