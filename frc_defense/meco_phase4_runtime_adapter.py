"""Phase 4 validation requirement/runtime/signal contract.

This adapter makes the boundary between a planned factor and an executed
stimulus explicit. It does not make a requirement executable unless its
stimulus and every required outcome signal have an authoritative runtime
source. Captures bind observations to the exact evaluator source identity.
"""
from __future__ import annotations

import hashlib
import json

from .meco_phase4 import PHASE4_FACTORS, phase4_factor_digest


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
        "limitations": ("seven-policy helper is not wired into validation episode scheduling", "policy-family outcome signals unavailable")},
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
        "runtime_event": None,
        "required_signals": ("safety_robot_contact_rate", "recovery_time_after_bump_s", "route_completion_rate"),
        # The schema has an aggregate robot-contact metric, but it is not
        # bump-specific, so it is not authoritative for this requirement.
        "authoritative_signals": (),
        "limitations": ("contact geometry helper is not wired into validation reset", "pair impulse/contact timestamps and route-following recovery signal unavailable; proxy metrics do not qualify")},
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
