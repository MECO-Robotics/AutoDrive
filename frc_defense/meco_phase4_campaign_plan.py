"""Compile and verify the non-executing Phase 4 validation campaign schedule.

This compiler binds the schedule to frozen inputs and describes exact matched
coverage without materializing episodes, exposing seed values, or running a
simulator. Design-axis expansions are explicitly not executable until runtime
stimuli and evaluator wiring are implemented and verified.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

from .meco_phase4_design import (
    CASE_AXES,
    _canonical_bytes,
    _read_json,
    _read_protocol_validation,
    _sha256,
    compile_phase4_design_overlay,
)
from .meco_phase4_runtime_adapter import phase4_requirement_binding
from .meco_phase4_opponents import (POLICY_CONFIG, POLICY_FACTOR_IDS,
                                    opponent_policy_runtime_domain_supported,
                                    opponent_policy_runtime_slots, policy_assignment_identity,
                                    policy_pool_identity,
                                    trench_interception_runtime_domain_supported,
                                    compile_opponent_policy_schedule)
from .meco_phase4_context import phase4_start_profile_assignment


PLAN_SCHEMA = "meco-phase4-validation-campaign-plan-v1"
POLICIES = ("D0", "D1", "D2", "A1", "A2", "A3", "C1", "C2")
DIRECT_PURSUIT_POLICY_IDS = tuple(sorted((*POLICIES, "A1_P0", "A2_P0", "A3_P0")))
CONDITIONS = ("no_factor", "factor_treatment")
EXPECTED_FACTOR_CASE_AXIS_CELLS = 6488
VALIDATION_SEEDS_PER_CASE = 100
TRENCH_POLICY_IDS = tuple(sorted((*POLICIES, "A1_P0", "A2_P0", "A3_P0")))


def _trench_interception_review_gate(rows: list[dict], *, protocol: dict,
        protocol_sha256: str, registry_sha256: str, case: dict,
        seed_sha256: str, assignment_sha256: str,
        expected_schedule_sha256: str) -> dict | None:
    """Accept only a complete, independently reviewed paired trench evidence row."""
    matches = [row for row in rows if isinstance(row, dict) and
        row.get("scenario_id") == case["scenario_id"] and
        row.get("category") == case["category"] and
        row.get("qualification_case") == {axis: case[axis] for axis in (
            "layout_id", "defender_family", "start_condition", "sensor_profile")}]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("trench-interception review evidence duplicates a protocol case")
    row = matches[0]
    exact = (row.get("schema") == "meco-phase4-trench-interception-review-evidence-v1"
        and row.get("protocol_id") == protocol.get("protocol_id")
        and row.get("protocol_sha256") == protocol_sha256
        and row.get("registry_sha256") == registry_sha256
        and row.get("factor_id") == "opponent.trench_interception"
        and row.get("policy_id") == "trench_interceptor"
        and row.get("split") == "validation"
        and row.get("validation_seed_assignment_sha256") == seed_sha256
        and row.get("assignment_identity_sha256") == assignment_sha256
        and row.get("control_policy_ids") == list(TRENCH_POLICY_IDS)
        and row.get("assigned_pressure_slots") == [0]
        and row.get("paired_rows_per_condition") == {
            "no_factor": len(TRENCH_POLICY_IDS) * VALIDATION_SEEDS_PER_CASE,
            "factor_treatment": len(TRENCH_POLICY_IDS) * VALIDATION_SEEDS_PER_CASE}
        and row.get("exact_pair_coverage") is True
        and row.get("all_pairs_full_horizon_valid") is True
        and row.get("outcome_signals_complete") is True
        and row.get("treatment_interception_valid_count") ==
            len(TRENCH_POLICY_IDS) * VALIDATION_SEEDS_PER_CASE
        and row.get("treatment_interception_invalid_count") == 0
        and row.get("unaccounted_assignments") == 0
        and row.get("invalid_pair_count") == 0
        and row.get("qualification_eligible") is False
        and isinstance(row.get("source_identity_sha256"), str)
        and bool(row["source_identity_sha256"])
        and isinstance(row.get("checkpoint_bundle_sha256"), dict)
        and bool(row["checkpoint_bundle_sha256"])
        and isinstance(row.get("execution_device"), dict)
        and bool(row["execution_device"])
        and row.get("schedule_sha256") == expected_schedule_sha256
        and isinstance(row.get("seed_manifest_sha256"), str)
        and bool(row["seed_manifest_sha256"])
        and row.get("policy_config_sha256") == policy_pool_identity()["config_sha256"]
        and isinstance(row.get("frontend_asset_manifest_sha256"), str)
        and bool(row["frontend_asset_manifest_sha256"])
        and isinstance(row.get("paired_outcomes_artifact_sha256"), str)
        and bool(row["paired_outcomes_artifact_sha256"]))
    replay = row.get("replay_review")
    replays = replay.get("paired_replays") if isinstance(replay, dict) else None
    exact = exact and isinstance(replay, dict) and replay.get("review_accepted") is True
    exact = exact and all(replay.get(flag) is True for flag in (
        "dashboard_backend_current", "dashboard_frontend_current", "endpoint_healthy",
        "replay_current_checkout", "replay_seed_is_in_exact_validation_manifest"))
    exact = exact and replay.get("backend_source_identity_sha256") == row.get(
        "source_identity_sha256")
    exact = exact and replay.get("frontend_asset_manifest_sha256") == row.get(
        "frontend_asset_manifest_sha256")
    exact = exact and replay.get("seed_manifest_sha256") == row.get("seed_manifest_sha256")
    exact = exact and type(replay.get("replay_seed")) is int
    exact = exact and replay.get("replay_seed") == next(iter(
        {item.get("seed") for item in replay.get("paired_replays", [])
         if isinstance(item, dict)}), None)
    exact = exact and isinstance(replays, list) and len(replays) == 2
    if not isinstance(replays, list):
        exact = False
    elif any(not isinstance(item, dict) or type(item.get("seed")) is not int
            for item in replays):
        exact = False
    exact = exact and {item.get("condition") for item in replays if isinstance(item, dict)} == set(CONDITIONS)
    exact = exact and len({item.get("seed") for item in replays if isinstance(item, dict)}) == 1
    exact = exact and all(isinstance(item, dict)
        and item.get("scenario_id") == case["scenario_id"]
        and item.get("category") == case["category"]
        and item.get("qualification_case") == {axis: case[axis] for axis in (
            "layout_id", "defender_family", "start_condition", "sensor_profile")}
        and item.get("split") == "validation"
        and item.get("seed_manifest_sha256") == row.get("seed_manifest_sha256")
        and item.get("source_identity_sha256") == row.get("source_identity_sha256")
        and item.get("checkpoint_bundle_sha256") == row.get("checkpoint_bundle_sha256")
        and item.get("execution_device") == row.get("execution_device")
        and item.get("source_identity_sha256") == row.get("source_identity_sha256")
        and item.get("frontend_asset_manifest_sha256") == row.get(
            "frontend_asset_manifest_sha256")
        and item.get("policy_id") == "trench_interceptor"
        and item.get("schedule_sha256") == row.get("schedule_sha256")
        and item.get("assignment_identity_sha256") == assignment_sha256
        and item.get("policy_config_sha256") == policy_pool_identity()["config_sha256"]
        and item.get("seed") == replay.get("replay_seed")
        and item.get("seed_manifest_sha256") == replay.get("seed_manifest_sha256")
        and item.get("backend_source_identity_sha256") == row.get("source_identity_sha256")
        and item.get("assigned_pressure_slots") == [0]
        and item.get("applied_slots") == ([] if item.get("condition") == "no_factor" else [0])
        and isinstance(item.get("replay_id"), str) and bool(item["replay_id"])
        and isinstance(item.get("replay_sha256"), str) and bool(item["replay_sha256"])
        for item in replays)
    exact = exact and len({item.get("replay_id") for item in replays if isinstance(item, dict)}) == 2
    exact = exact and isinstance(replay.get("reviewer"), str) and bool(replay["reviewer"])
    if not exact:
        return {"accepted": False, "reason": "trench review evidence is incomplete or mismatched"}
    return {"accepted": True, "evidence_sha256": _sha256(_canonical_bytes(row)),
        "source_identity_sha256": row["source_identity_sha256"],
        "checkpoint_bundle_sha256": row["checkpoint_bundle_sha256"],
        "execution_device": row["execution_device"],
        "schedule_sha256": row["schedule_sha256"],
        "frontend_asset_manifest_sha256": row["frontend_asset_manifest_sha256"],
        "paired_outcomes_artifact_sha256": row["paired_outcomes_artifact_sha256"]}


def _seed_digest(seeds: list[int]) -> str:
    """Bind exact validation seed assignment without returning its values."""
    return _sha256(_canonical_bytes(seeds))


def _direct_pursuit_review_gate(rows: list[dict], *, protocol: dict,
        protocol_sha256: str, registry_sha256: str, case: dict, seed_sha256: str,
        assignment_sha256: str, seed_count: int,
        configured_pressure_slots: list[int]) -> dict | None:
    matches = [row for row in rows if isinstance(row, dict) and
        row.get("scenario_id") == case["scenario_id"] and
        row.get("category") == case["category"] and
        row.get("qualification_case") == {axis: case[axis] for axis in (
            "layout_id", "defender_family", "start_condition", "sensor_profile")}]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("direct-pursuit review evidence duplicates a protocol case")
    row = matches[0]
    replay = row.get("replay_review")
    outcomes = row.get("paired_outcomes")
    paired_replays = replay.get("paired_replays") if isinstance(replay, dict) else None
    pair_count = seed_count * len(DIRECT_PURSUIT_POLICY_IDS)
    schedule_sha256 = row.get("schedule_sha256")
    seed_manifest_sha256 = row.get("seed_manifest_sha256")
    interception = outcomes.get("interception_rate") if isinstance(outcomes, dict) else None
    contact_rate = outcomes.get("robot_contact_rate") if isinstance(outcomes, dict) else None
    time_loss = outcomes.get("paired_time_loss_to_opponents_s") if isinstance(outcomes, dict) else None
    exact_binding = (row.get("schema") == "meco-phase4-direct-pursuit-review-evidence-v1"
        and row.get("protocol_id") == protocol.get("protocol_id")
        and row.get("protocol_sha256") == protocol_sha256
        and row.get("registry_sha256") == registry_sha256
        and row.get("factor_id") == "opponent.direct_pursuit"
        and row.get("policy_id") == "direct_pursuit"
        and row.get("policy_config_sha256") == policy_pool_identity()["config_sha256"]
        and row.get("split") == "validation"
        and row.get("validation_seed_assignment_sha256") == seed_sha256
        and row.get("assignment_identity_sha256") == assignment_sha256
        and isinstance(schedule_sha256, str) and bool(schedule_sha256)
        and isinstance(seed_manifest_sha256, str) and bool(seed_manifest_sha256)
        and row.get("configured_pressure_slots") == configured_pressure_slots
        and row.get("control_policy_ids") == list(DIRECT_PURSUIT_POLICY_IDS)
        and isinstance(row.get("paired_rows_per_condition"), dict)
        and set(row["paired_rows_per_condition"]) == {
            "no_factor", "factor_treatment"}
        and row["paired_rows_per_condition"] == {
            "no_factor": pair_count, "factor_treatment": pair_count}
        and isinstance(row.get("source_identity_sha256"), str)
        and bool(row.get("source_identity_sha256"))
        and isinstance(row.get("checkpoint_bundle_sha256"), dict)
        and bool(row.get("checkpoint_bundle_sha256"))
        and all(isinstance(value, str) and value
                for value in row["checkpoint_bundle_sha256"].values())
        and isinstance(row.get("execution_device"), dict)
        and bool(row.get("execution_device"))
        and row.get("exact_pair_coverage") is True
        and row.get("all_pairs_uncensored") is True
        and row.get("qualification_eligible") is False
        and row.get("eligible_pair_count") == pair_count
        and isinstance(row.get("paired_outcomes_artifact_sha256"), str)
        and bool(row.get("paired_outcomes_artifact_sha256"))
        and isinstance(outcomes, dict)
        and isinstance(interception, dict)
        and interception.get("denominator_eligible_episodes") == pair_count
        and type(interception.get("numerator_intercepted_episodes")) is int
        and 0 <= interception["numerator_intercepted_episodes"] <= pair_count
        and isinstance(interception.get("rate"), (int, float))
        and math.isfinite(interception["rate"])
        and abs(interception["rate"] - interception[
            "numerator_intercepted_episodes"] / pair_count) <= 1e-12
        and isinstance(contact_rate, dict)
        and type(contact_rate.get("numerator_distinct_pair_contact_substeps")) is int
        and contact_rate["numerator_distinct_pair_contact_substeps"] >= 0
        and isinstance(contact_rate.get("denominator_pair_seconds"), (int, float))
        and math.isfinite(contact_rate["denominator_pair_seconds"])
        and contact_rate["denominator_pair_seconds"] > 0
        and isinstance(contact_rate.get(
            "rate_distinct_pair_contact_substeps_per_pair_second"), (int, float))
        and math.isfinite(contact_rate[
            "rate_distinct_pair_contact_substeps_per_pair_second"])
        and isinstance(time_loss, list) and len(time_loss) == pair_count
        and all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in time_loss)
        and isinstance(replay, dict)
        and replay.get("review_accepted") is True
        and replay.get("validation_seed_assignment_sha256") == seed_sha256
        and replay.get("dashboard_backend_current") is True
        and replay.get("dashboard_frontend_current") is True
        and replay.get("endpoint_healthy") is True
        and replay.get("replay_seed_is_in_exact_validation_manifest") is True
        and replay.get("replay_current_checkout") is True
        and type(replay.get("replay_seed")) is int
        and 0 <= replay["replay_seed"] < 2**63
        and replay.get("seed_manifest_sha256") == seed_manifest_sha256
        and replay.get("backend_source_identity_sha256") == row.get(
            "source_identity_sha256")
        and isinstance(replay.get("frontend_asset_manifest_sha256"), str)
        and bool(replay.get("frontend_asset_manifest_sha256"))
        and isinstance(paired_replays, list) and len(paired_replays) == 2
        and {item.get("condition") for item in paired_replays if isinstance(item, dict)} ==
            {"no_factor", "factor_treatment"}
        and len({item.get("seed") for item in paired_replays if isinstance(item, dict)}) == 1
        and all(isinstance(item, dict)
            and item.get("scenario_id") == case["scenario_id"]
            and item.get("category") == case["category"]
            and item.get("qualification_case") == {axis: case[axis] for axis in (
                "layout_id", "defender_family", "start_condition", "sensor_profile")}
            and item.get("split") == "validation"
            and type(item.get("seed")) is int and 0 <= item["seed"] < 2**63
            and item.get("seed") == replay.get("replay_seed")
            and item.get("seed_manifest_sha256") == seed_manifest_sha256
            and item.get("schedule_sha256") == schedule_sha256
            and item.get("assignment_identity_sha256") == assignment_sha256
            and item.get("source_identity_sha256") == row.get("source_identity_sha256")
            and item.get("frontend_asset_manifest_sha256") == replay.get(
                "frontend_asset_manifest_sha256")
            and item.get("checkpoint_bundle_sha256") == row.get("checkpoint_bundle_sha256")
            and item.get("execution_device") == row.get("execution_device")
            and item.get("policy_config_sha256") == policy_pool_identity()["config_sha256"]
            and item.get("policy_id") == "direct_pursuit"
            and item.get("configured_slots") == configured_pressure_slots
            and item.get("policy_ids") == list(DIRECT_PURSUIT_POLICY_IDS)
            and ((item.get("condition") == "no_factor" and item.get("applied_slots") == [])
                or (item.get("condition") == "factor_treatment" and
                    item.get("applied_slots") == configured_pressure_slots))
            and isinstance(item.get("replay_id"), str) and bool(item["replay_id"])
            and isinstance(item.get("replay_sha256"), str) and bool(item["replay_sha256"])
            for item in paired_replays)
        and isinstance(replay.get("reviewer"), str) and bool(replay["reviewer"]))
    if not exact_binding:
        return {"accepted": False, "reason": "direct-pursuit review evidence is incomplete or mismatched"}
    return {"accepted": True, "evidence_sha256": _sha256(_canonical_bytes(row)),
        "source_identity_sha256": row["source_identity_sha256"],
        "checkpoint_bundle_sha256": row["checkpoint_bundle_sha256"],
        "execution_device": row["execution_device"],
        "schedule_sha256": schedule_sha256,
        "registry_sha256": registry_sha256,
        "seed_manifest_sha256": seed_manifest_sha256,
        "paired_replay_sha256": _sha256(_canonical_bytes(paired_replays))}


def _validate_seed_partition(protocol: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    """Validate selected assignments; protocol reader supplied validation rows only."""
    rows = protocol.get("cases")
    if not isinstance(rows, list):
        raise ValueError("protocol case table is missing")
    result: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != "validation":
            raise ValueError("selective protocol reader returned a non-validation case")
        seeds = row.get("seeds")
        if (not isinstance(seeds, list) or not seeds or
                any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds) or
                len(seeds) != len(set(seeds))):
            raise ValueError("validation case has malformed or duplicate seeds")
        if len(seeds) != VALIDATION_SEEDS_PER_CASE:
            raise ValueError("validation case does not have exactly 100 seeds")
        if any(not isinstance(row.get(axis), str) or not row[axis] for axis in CASE_AXES):
            raise ValueError("validation case has incomplete case axes")
        key = tuple(row[axis] for axis in CASE_AXES)
        if key in result:
            raise ValueError("duplicate validation case axis identity")
        result[key] = {"case": {axis: row[axis] for axis in CASE_AXES},
                       "seed_count": len(seeds), "seed_sha256": _seed_digest(seeds),
                       "_seed_values_internal": tuple(seeds)}
    return result


def compile_phase4_campaign_plan(protocol_path: str | Path,
                                registry_path: str | Path,
                                overlay_path: str | Path,
                                direct_pursuit_review_evidence_path: str | Path | None = None,
                                trench_interception_review_evidence_path: str | Path | None = None
                                ) -> dict[str, Any]:
    """Return a compact, deterministic schedule plan, never episode rows."""
    protocol, protocol_raw = _read_protocol_validation(protocol_path)
    registry, registry_raw = _read_json(registry_path)
    overlay, overlay_raw = _read_json(overlay_path)
    direct_review_rows = []
    direct_review_raw_sha256 = None
    trench_review_rows = []
    trench_review_raw_sha256 = None
    if direct_pursuit_review_evidence_path is not None:
        review_path = Path(direct_pursuit_review_evidence_path)
        review_raw = review_path.read_bytes()
        review_payload = json.loads(review_raw)
        if (not isinstance(review_payload, dict) or review_payload.get("schema") !=
                "meco-phase4-direct-pursuit-review-evidence-bundle-v1" or
                not isinstance(review_payload.get("cases"), list)):
            raise ValueError("invalid direct-pursuit replay-review evidence bundle")
        direct_review_rows = review_payload["cases"]
        direct_review_raw_sha256 = _sha256(review_raw)
    if trench_interception_review_evidence_path is not None:
        review_path = Path(trench_interception_review_evidence_path)
        review_raw = review_path.read_bytes()
        review_payload = json.loads(review_raw)
        if (not isinstance(review_payload, dict) or review_payload.get("schema") !=
                "meco-phase4-trench-interception-review-evidence-bundle-v1" or
                not isinstance(review_payload.get("cases"), list)):
            raise ValueError("invalid trench-interception replay-review evidence bundle")
        trench_review_rows = review_payload["cases"]
        trench_review_raw_sha256 = _sha256(review_raw)
    expected_overlay = compile_phase4_design_overlay(protocol_path, registry_path)
    if overlay != expected_overlay:
        raise ValueError("Phase 4 overlay does not exactly match the bound protocol and registry")
    if overlay.get("overlay_id") != _sha256(_canonical_bytes({
            key: value for key, value in overlay.items() if key != "overlay_id"})):
        raise ValueError("Phase 4 overlay ID digest mismatch")
    if overlay["protocol_binding"].get("protocol_raw_file_sha256") != _sha256(protocol_raw):
        raise ValueError("overlay protocol raw digest mismatch")
    if overlay["factor_registry_binding"].get("registry_raw_file_sha256") != _sha256(registry_raw):
        raise ValueError("overlay registry raw digest mismatch")

    pairing = registry.get("pairing", {})
    policies = pairing.get("same_checkpoint_controls")
    conditions = pairing.get("conditions")
    if policies != list(POLICIES) or conditions != list(CONDITIONS):
        raise ValueError("registry policy/condition grid differs from exact paired design")
    if pairing.get("seed_policy") != "inherit_validation_protocol_without_serializing_or_exporting_seed_values":
        raise ValueError("unsupported Phase 4 validation seed policy")

    validation = _validate_seed_partition(protocol)
    overlay_cases = overlay.get("validation_case_axes")
    overlay_case_map = {tuple(c.get(axis) for axis in CASE_AXES): c
                        for c in overlay_cases if isinstance(c, dict)}
    if len(overlay_case_map) != len(overlay_cases) or set(overlay_case_map) != set(validation):
        raise ValueError("overlay validation case inventory has missing, extra, or duplicate cases")

    registry_requirements = {r["requirement_id"]: r for r in registry["requirements"]}
    plan_requirements = []
    seen_cells: set[str] = set()
    planned_cells = 0
    total_rows = 0
    for compiled in overlay["requirements"]:
        req_id = compiled["requirement_id"]
        runtime_binding = phase4_requirement_binding(req_id)
        source_req = registry_requirements[req_id]
        factors = source_req["factor_ids"]
        selectors = source_req["validation_case_axes"]
        applicable = []
        for case_tuple, record in validation.items():
            case = record["case"]
            if not all(case.get(axis) in values
                       for axis, values in compiled["resolved_protocol_constraints"].items()):
                continue
            scope = compiled["implicit_scenario_scope"]
            if scope and case["scenario_id"] not in scope:
                continue
            applicable.append((case, record))
        expected_base = [{"factor_id": factor, **case}
                         for factor in factors for case, _ in applicable]
        if (len(expected_base) != len(compiled["protocol_bound_case_rows"]) or
                {_canonical_bytes(row) for row in expected_base} !=
                {_canonical_bytes(row) for row in compiled["protocol_bound_case_rows"]}):
            raise ValueError(f"{req_id} exact protocol-case coverage differs from overlay")

        design_domains: dict[str, list[Any]] = {}
        for axis, selector in selectors.items():
            if isinstance(selector, str) and selector.startswith("design_axis."):
                name = selector.removeprefix("design_axis.")
                domain = registry.get("design_axes", {}).get(name)
                if not isinstance(domain, list) or not domain or len(domain) != len(set(map(str, domain))):
                    raise ValueError(f"{req_id} has invalid design axis {name}")
                design_domains[name] = domain
        names = tuple(sorted(design_domains))
        design_products = (list(itertools.product(*(design_domains[n] for n in names)))
                           if names else [()])
        cells = []
        for factor in factors:
            for case, seed_info in applicable:
                for product in design_products:
                    design_axes = dict(zip(names, product))
                    reactive_assignment = None
                    direct_pursuit_assignment = None
                    direct_review_gate = None
                    trench_interception_assignment = None
                    trench_review_gate = None
                    start_profile_assignment = None
                    if req_id == "reactive_defenders_and_opponent_pools":
                        policy_id = design_axes.get("policy_pool")
                        if policy_id not in POLICY_CONFIG["policies"]:
                            raise ValueError("reactive policy cell has an unknown policy assignment")
                        config_sha256 = policy_pool_identity()["config_sha256"]
                        case_payload = {"scenario_id": case["scenario_id"],
                            "category": case["category"],
                            "qualification_case": {axis: case[axis] for axis in (
                                "layout_id", "defender_family", "start_condition",
                                "sensor_profile")},
                            "split": "validation",
                            "validation_seed_assignment_sha256": seed_info["seed_sha256"]}
                        reactive_assignment = {
                            "schema": "meco-phase4-opponent-schedule-assignment-v1",
                            "policy_id": policy_id,
                            "policy_family": POLICY_FACTOR_IDS[policy_id],
                            "configured_pressure_slots": (list(opponent_policy_runtime_slots(
                                case["category"])) if opponent_policy_runtime_domain_supported(
                                    case["scenario_id"], case["category"]) else []),
                            "policy_config_sha256": config_sha256,
                            "protocol_bound_evaluation_case": case_payload,
                            "stimulus_identity_sha256": _sha256(_canonical_bytes({
                                "case": case_payload, "factor_id": factor,
                                "policy_id": policy_id,
                                "policy_config_sha256": config_sha256})),
                            "factor_treatment_seeded_assignment_sha256": _sha256(
                                _canonical_bytes([
                                    policy_assignment_identity(policy_id=policy_id,
                                        scenario_id=case["scenario_id"],
                                        category=case["category"],
                                        qualification_case=case_payload["qualification_case"],
                                        seed=seed, condition="factor_treatment",
                                        factor_id=factor)
                                    for seed in seed_info["_seed_values_internal"]])),
                            "per_seed_identity_derivation": (
                                "sha256(canonical validation case, exact seed, factor, policy ID, config hash)"),
                            "policy_inputs": POLICY_CONFIG["sensing_boundary"],
                            "runtime_domain_supported": opponent_policy_runtime_domain_supported(
                                case["scenario_id"], case["category"]),
                            "planned_policy_rows_per_control_per_category": seed_info["seed_count"],
                            "control_policy_ids": list(POLICIES),
                            "pair_conditions": list(CONDITIONS),
                            "paired_condition_semantics": {
                                "no_factor": "registered category scripted opponent",
                                "factor_treatment": policy_id},
                            "paired_control_category_seed_coverage_sha256": _sha256(
                                _canonical_bytes([[control, case["category"], seed, condition]
                                    for control in POLICIES
                                    for seed in seed_info["_seed_values_internal"]
                                    for condition in CONDITIONS])),
                            "paired_control_category_seed_row_count": (
                                len(POLICIES) * seed_info["seed_count"] * len(CONDITIONS)),
                            "exact_pair_coverage": True,
                            "executed": False,
                            "qualification_eligible": False,
                        }
                    if req_id == "direct_pursuit":
                        if factor != "opponent.direct_pursuit":
                            raise ValueError("direct-pursuit requirement has an unexpected factor")
                        supported = opponent_policy_runtime_domain_supported(
                            case["scenario_id"], case["category"])
                        if supported:
                            case_payload = {"scenario_id": case["scenario_id"],
                                "category": case["category"],
                                "qualification_case": {axis: case[axis] for axis in (
                                    "layout_id", "defender_family", "start_condition",
                                    "sensor_profile")}, "split": "validation",
                                "validation_seed_assignment_sha256": seed_info["seed_sha256"]}
                            registered_pressure_slots = list(opponent_policy_runtime_slots(
                                case["category"]))
                            if not registered_pressure_slots:
                                raise ValueError("direct pursuit requires registered pressure slots")
                            assigned_pursuit_slot = registered_pressure_slots[0]
                            direct_pursuit_assignment = {
                                "schema": "meco-phase4-direct-pursuit-assignment-v1",
                                "policy_id": "direct_pursuit", "factor_id": factor,
                                "policy_config_sha256": policy_pool_identity()["config_sha256"],
                                "registered_pressure_slots": registered_pressure_slots,
                                "assigned_pursuit_slot": assigned_pursuit_slot,
                                "configured_pressure_slots": [assigned_pursuit_slot],
                                "preserved_scripted_slots": registered_pressure_slots[1:],
                                "control_policy_ids": list(DIRECT_PURSUIT_POLICY_IDS),
                                "paired_control_category_seed_coverage_sha256": _sha256(
                                    _canonical_bytes([[control, case["category"], seed, condition]
                                        for control in DIRECT_PURSUIT_POLICY_IDS
                                        for seed in seed_info["_seed_values_internal"]
                                        for condition in CONDITIONS])),
                                "protocol_bound_evaluation_case": case_payload,
                                "factor_treatment_seeded_assignment_sha256": _sha256(
                                    _canonical_bytes([{**policy_assignment_identity(
                                        policy_id="direct_pursuit",
                                        scenario_id=case["scenario_id"],
                                        category=case["category"],
                                        qualification_case=case_payload[
                                            "qualification_case"], seed=seed,
                                        condition="factor_treatment", factor_id=factor),
                                        "category": case["category"],
                                        "registered_pressure_slots": registered_pressure_slots,
                                        "configured_pressure_slots": [assigned_pursuit_slot],
                                        "preserved_scripted_slots": registered_pressure_slots[1:]}
                                        for seed in seed_info["_seed_values_internal"]])),
                                "runtime_domain_supported": True,
                                "paired_control_category_seed_row_count": (
                                    len(DIRECT_PURSUIT_POLICY_IDS) * seed_info["seed_count"] * len(CONDITIONS)),
                                "exact_pair_coverage": True,
                                "executed": False, "qualification_eligible": False,
                            }
                            direct_review_gate = _direct_pursuit_review_gate(
                                direct_review_rows, protocol=protocol,
                                protocol_sha256=_sha256(protocol_raw),
                                registry_sha256=_sha256(registry_raw), case=case,
                                seed_sha256=seed_info["seed_sha256"],
                                assignment_sha256=direct_pursuit_assignment[
                                    "factor_treatment_seeded_assignment_sha256"],
                                seed_count=seed_info["seed_count"],
                                configured_pressure_slots=direct_pursuit_assignment[
                                    "configured_pressure_slots"])
                    if (req_id == "trench_interception" and
                            factor == "opponent.trench_interception" and
                            case["scenario_id"] == "offense_vs_scripted_pressure" and
                            case["category"] == "active/trench/level_4_reactive" and
                            case["layout_id"] == "2026_rebuilt_shifted_structures_v1" and
                            case["defender_family"] == "orbiting_interceptor" and
                            case["start_condition"] == "offset_upper_lane" and
                            case["sensor_profile"] == "noisy" and
                            seed_info["seed_count"] == VALIDATION_SEEDS_PER_CASE):
                        if not trench_interception_runtime_domain_supported(
                                case["scenario_id"], case["category"]):
                            raise ValueError("frozen trench candidate is outside runtime domain")
                        case_payload = {"scenario_id": case["scenario_id"],
                            "category": case["category"], "qualification_case": {
                                axis: case[axis] for axis in ("layout_id", "defender_family",
                                    "start_condition", "sensor_profile")},
                            "split": "validation",
                            "validation_seed_assignment_sha256": seed_info["seed_sha256"]}
                        schedule = compile_opponent_policy_schedule(
                            policy_id="trench_interceptor", scenario_id=case["scenario_id"],
                            categories=(case["category"],),
                            qualification_case=case_payload["qualification_case"],
                            seeds=seed_info["_seed_values_internal"], split="validation",
                            factor_id=factor, protocol_id=protocol["protocol_id"])
                        if schedule["configured_pressure_slots"][case["category"]] != [0]:
                            raise ValueError("frozen trench candidate must replace only slot 0")
                        trench_interception_assignment = {
                            "schema": "meco-phase4-trench-interception-assignment-v1",
                            "policy_id": "trench_interceptor", "factor_id": factor,
                            "policy_family": schedule["policy_family"],
                            "policy_config_sha256": schedule["policy_config_sha256"],
                            "configured_pressure_slots": [0],
                            "control_policy_ids": list(TRENCH_POLICY_IDS),
                            "schedule": schedule,
                            "schedule_sha256": schedule["schedule_sha256"],
                            "protocol_bound_evaluation_case": case_payload,
                            "factor_treatment_seeded_assignment_sha256":
                                schedule["per_episode_assignment_sha256"],
                            "paired_condition_semantics": {
                                "no_factor": "registered level-4 reactive slot-0 script",
                                "factor_treatment": "trench_interceptor replaces slot 0"},
                            "paired_rows_per_condition": len(TRENCH_POLICY_IDS) * seed_info["seed_count"],
                            "exact_pair_coverage_required": True,
                            "runtime_domain_supported": True,
                            "executed": False, "qualification_eligible": False}
                        trench_review_gate = _trench_interception_review_gate(
                            trench_review_rows, protocol=protocol,
                            protocol_sha256=_sha256(protocol_raw),
                            registry_sha256=_sha256(registry_raw), case=case,
                            seed_sha256=seed_info["seed_sha256"],
                            assignment_sha256=schedule["per_episode_assignment_sha256"],
                            expected_schedule_sha256=schedule["schedule_sha256"])
                    if req_id == "multiple_legal_team_start_states":
                        registry_profile_id = design_axes.get("legal_team_pose_sets")
                        start_profile_assignment = phase4_start_profile_assignment(
                            registry_profile_id,
                            scenario_id=case["scenario_id"], split="validation",
                            protocol_id=protocol["protocol_id"],
                            protocol_sha256=_sha256(protocol_raw),
                            qualification_case={axis: case[axis] for axis in (
                                "layout_id", "defender_family", "start_condition",
                                "sensor_profile")},
                            categories=(case["category"],),
                            seeds=seed_info["_seed_values_internal"])
                    status = ("protocol_bound_policy_assignment_planned_not_executed"
                              if reactive_assignment is not None or direct_pursuit_assignment is not None or
                                trench_interception_assignment is not None else
                              "protocol_bound_start_profile_assignment_planned_not_executed"
                              if start_profile_assignment is not None else
                              "design_only_unassigned_to_protocol_case" if design_axes else
                              "planned_validation_not_executed")
                    cell = {
                        "requirement_id": req_id,
                        "factor_id": factor,
                        "case_axes": case,
                        "design_axes": design_axes,
                        "seed_count": seed_info["seed_count"],
                        "validation_seed_assignment_sha256": seed_info["seed_sha256"],
                        "planned_pair_rows_per_policy": 2 * seed_info["seed_count"],
                        "status": status,
                        "schedulable": True,
                        "runtime_factor_kind": runtime_binding[
                            "runtime_factor_by_factor_id"].get(factor),
                        "runtime_event": runtime_binding[
                            "runtime_event_by_factor_id"].get(factor),
                        "runtime_stimulus_supported": (
                            factor in runtime_binding["runtime_factor_by_factor_id"]
                            and runtime_binding["stimulus_supported"]),
                        "scheduler_assignment_supported": (
                            reactive_assignment is not None or
                            direct_pursuit_assignment is not None or
                            trench_interception_assignment is not None or
                            start_profile_assignment is not None),
                        "opponent_schedule_assignment": reactive_assignment,
                        "direct_pursuit_assignment": direct_pursuit_assignment,
                        "trench_interception_assignment": trench_interception_assignment,
                        "start_profile_assignment": start_profile_assignment,
                        "required_outcome_signals_complete": set(
                            runtime_binding["required_signals"]).issubset(
                                runtime_binding["authoritative_signals"]),
                        "executable": (not bool(design_axes)
                            and (req_id != "direct_pursuit" or
                                bool(direct_review_gate and direct_review_gate.get("accepted")))
                            and (req_id != "trench_interception" or
                                bool(trench_interception_assignment and trench_review_gate and
                                    trench_review_gate.get("accepted")))
                            and factor in runtime_binding["runtime_factor_by_factor_id"]
                            and runtime_binding["stimulus_supported"]
                            and set(runtime_binding["required_signals"]).issubset(
                                runtime_binding["authoritative_signals"])),
                        "executed": False,
                        "simulator_behavior_validated": bool(
                            (direct_review_gate and direct_review_gate.get("accepted")) or
                            (trench_review_gate and trench_review_gate.get("accepted"))),
                        "direct_pursuit_review_gate": direct_review_gate,
                        "trench_interception_review_gate": trench_review_gate,
                        "trench_interception_tranche_status": (
                            "review_accepted_candidate_cell_only" if
                            trench_review_gate and trench_review_gate.get("accepted") and
                                bool(trench_interception_assignment) else
                            "planned_not_executed_or_reviewed" if
                            trench_interception_assignment else None),
                    }
                    schedule_identity = {"factor_id": factor, "case_axes": case,
                                        "design_axes": design_axes}
                    schedule_key = _sha256(_canonical_bytes(schedule_identity))
                    cell["case_cell_key"] = _sha256(_canonical_bytes(cell))
                    if schedule_key in seen_cells:
                        raise ValueError("duplicate factor/case/design cell")
                    seen_cells.add(schedule_key)
                    cells.append(cell)
        count = len(cells)
        planned_cells += count
        req_rows = count * VALIDATION_SEEDS_PER_CASE * len(POLICIES) * len(CONDITIONS)
        total_rows += req_rows
        plan_requirements.append({
            "requirement_id": req_id,
            "factor_ids": factors,
            "applicable_protocol_case_count": len(applicable),
            "design_axis_domains": design_domains,
            "factor_case_axis_cell_count": count,
            "pairing_grid": [{"policy_id": p, "condition": c}
                             for p in POLICIES for c in CONDITIONS],
            "pairing_grid_sha256": _sha256(_canonical_bytes([
                {"policy_id": p, "condition": c} for p in POLICIES for c in CONDITIONS])),
            "pair_coverage": "exactly_one_control_and_one_treatment_per_validation_seed_policy_cell",
            "runtime_binding": runtime_binding,
            "scheduled_row_count_theoretical": req_rows,
            "cells": cells,
        })
    if planned_cells != EXPECTED_FACTOR_CASE_AXIS_CELLS:
        raise ValueError(f"factor/case/design-axis cell total {planned_cells} != 6488")
    expected_rows = planned_cells * VALIDATION_SEEDS_PER_CASE * len(POLICIES) * len(CONDITIONS)
    if total_rows != expected_rows or total_rows != 10_380_800:
        raise ValueError("theoretical schedule row count does not match exact paired coverage")
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "planned_not_executed",
        "qualification_eligible": False,
        "protocol_binding": {"protocol_id": protocol["protocol_id"],
                             "protocol_raw_file_sha256": _sha256(protocol_raw)},
        "registry_binding": {"registry_schema": registry["schema"],
                             "registry_raw_file_sha256": _sha256(registry_raw)},
        "overlay_binding": {"overlay_id": overlay["overlay_id"],
                            "overlay_raw_file_sha256": _sha256(overlay_raw)},
        "validation_seed_policy": "digest_only_exact_validation_case_assignment_no_seed_values",
        "validation_case_count": len(validation),
        "policies": list(POLICIES),
        "pair_conditions": list(CONDITIONS),
        "factor_case_axis_cell_count": planned_cells,
        "scheduled_row_count_theoretical": total_rows,
        "episode_rows_materialized": False,
        "final_test_rows_or_seed_values_included": False,
        "design_only_cells_executable": False,
        "executable_cell_count": sum(1 for requirement in plan_requirements
                                      for cell in requirement["cells"]
                                      if cell["executable"]),
        "runtime_capture_schema": "meco-phase4-realized-stimulus-capture-v1",
        "requirements": plan_requirements,
        "runtime_and_evaluator_wiring_required": True,
        "direct_pursuit_review_evidence_sha256": direct_review_raw_sha256,
        "trench_interception_review_evidence_sha256": trench_review_raw_sha256,
    }
    plan["plan_id"] = _sha256(_canonical_bytes(plan))
    return plan


def write_phase4_campaign_plan(protocol_path: str | Path, registry_path: str | Path,
                               overlay_path: str | Path,
                               output_path: str | Path,
                               direct_pursuit_review_evidence_path: str | Path | None = None,
                               trench_interception_review_evidence_path: str | Path | None = None
                               ) -> dict[str, Any]:
    plan = compile_phase4_campaign_plan(protocol_path, registry_path, overlay_path,
        direct_pursuit_review_evidence_path=direct_pursuit_review_evidence_path,
        trench_interception_review_evidence_path=trench_interception_review_evidence_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--direct-pursuit-review-evidence", type=Path,
        help="reviewed held-out simulator/replay evidence required to open direct-pursuit executability")
    parser.add_argument("--trench-interception-review-evidence", type=Path,
        help="reviewed held-out paired trench evidence required to open the candidate cell gate")
    args = parser.parse_args(argv)
    plan = write_phase4_campaign_plan(args.protocol, args.registry, args.overlay, args.output,
        direct_pursuit_review_evidence_path=args.direct_pursuit_review_evidence,
        trench_interception_review_evidence_path=args.trench_interception_review_evidence)
    print(json.dumps({"plan_id": plan["plan_id"],
                      "factor_case_axis_cell_count": plan["factor_case_axis_cell_count"],
                      "scheduled_row_count_theoretical": plan["scheduled_row_count_theoretical"],
                      "episode_rows_materialized": plan["episode_rows_materialized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
