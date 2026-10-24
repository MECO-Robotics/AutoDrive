"""Compile a validation-only design overlay for source-plan Phase 4.

This module describes planned factor coverage. It neither adds simulator
stimuli nor runs evaluations, and it never serializes protocol seed values.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from .meco_phase4 import PHASE4_FACTORS


OVERLAY_SCHEMA = "meco-phase4-validation-design-overlay-v1"
CASE_AXES = ("scenario_id", "category", "layout_id", "defender_family",
             "start_condition", "sensor_profile")
SOURCE_PLAN_REQUIREMENTS = frozenset({
    "stationary_blockage", "direct_pursuit", "trench_interception", "feints",
    "reactive_defenders_and_opponent_pools", "multiple_legal_team_start_states",
    "teammate_behaviors", "sensor_delay_and_low_confidence",
    "abrupt_opponent_change", "bumps_and_contact",
    "observation_only_focal_pose_mislocalization",
})


def _resolve_selector(axis: str, selector: Any, protocol: dict,
                      registry: dict) -> tuple[set[Any] | None, dict[str, list[Any]],
                                               set[str]]:
    """Resolve a registry selector to protocol constraints or design-only axes."""
    design_axes = registry.get("design_axes", {})
    if isinstance(selector, list):
        values = selector
        implicit_scenarios: set[str] = set()
    elif not isinstance(selector, str) or not selector:
        raise ValueError(f"{axis} selector must be a nonempty string or list")
    elif selector.startswith("design_axis."):
        name = selector.removeprefix("design_axis.")
        if name not in design_axes or not isinstance(design_axes[name], list) or not design_axes[name]:
            raise ValueError(f"unknown or empty design-axis selector: {selector}")
        return None, {axis: list(design_axes[name])}, set()
    elif selector.startswith("validation."):
        name = selector.removeprefix("validation.")
        if name in ("layouts", "defender_families"):
            values = protocol.get("validation", {}).get(name)
        elif name in ("start_conditions", "sensor_profiles"):
            values = protocol.get(name, {}).get("validation")
        else:
            raise ValueError(f"unknown protocol-axis selector: {selector}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"unresolved protocol-axis selector: {selector}")
        implicit_scenarios = set()
    elif selector.endswith(".required_categories"):
        scenario = selector.removesuffix(".required_categories")
        if scenario not in protocol.get("scenario_ids", []):
            raise ValueError(f"unknown scenario category selector: {selector}")
        values = sorted({row.get("category") for row in protocol.get("cases", [])
                         if row.get("split") == "validation" and
                         row.get("scenario_id") == scenario and
                         isinstance(row.get("category"), str)})
        if not values:
            raise ValueError(f"scenario has no validation categories: {selector}")
        implicit_scenarios = {scenario}
    else:
        values = [selector]
        implicit_scenarios = set()

    if not isinstance(values, list) or not values or any(
            not isinstance(value, (str, int, float, bool)) for value in values):
        raise ValueError(f"unresolved or malformed selector for {axis}: {selector}")
    if axis not in CASE_AXES:
        raise ValueError(f"{axis} is neither a protocol case axis nor a design axis")
    known = {row.get(axis) for row in protocol.get("cases", [])
             if row.get("split") == "validation"}
    if not set(values).issubset(known):
        raise ValueError(f"selector for {axis} does not resolve to validation cases: {selector}")
    return set(values), {}, implicit_scenarios


def _expand_design_rows(factor_ids: list[str], design_selectors: dict[str, list[Any]]) -> list[dict]:
    if not design_selectors:
        return []
    names = tuple(sorted(design_selectors))
    products = list(itertools.product(*(design_selectors[name] for name in names)))
    return [{"factor_id": factor, "design_axes": dict(zip(names, values)),
             "status": "design_only_unassigned_to_protocol_case",
             "counts_as_protocol_coverage": False,
             "counts_as_executed_coverage": False}
            for factor in factor_ids for values in products]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: str | Path) -> tuple[dict, bytes]:
    raw = Path(path).expanduser().resolve().read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Phase 4 design inputs must be JSON objects")
    return value, raw


def compile_phase4_design_overlay(protocol_path: str | Path,
                                  registry_path: str | Path) -> dict:
    """Bind factor design and planned paired validation coverage to exact inputs.

    Only validation/train axis names and validation case identifiers leave this
    function. Seed values and final-test rows are intentionally never copied.
    """
    protocol, protocol_raw = _read_json(protocol_path)
    registry, registry_raw = _read_json(registry_path)
    expected_protocol_id = _sha256(_canonical_bytes({
        key: value for key, value in protocol.items() if key != "protocol_id"}))
    if protocol.get("protocol_id") != expected_protocol_id:
        raise ValueError("qualification protocol ID does not match its contents")
    if registry.get("schema") != "meco-phase4-design-registry-v1":
        raise ValueError("unsupported Phase 4 design registry")

    train = protocol.get("training")
    validation = protocol.get("validation")
    if not isinstance(train, dict) or not isinstance(validation, dict):
        raise ValueError("protocol must declare training and validation axes")
    # The protocol's "none" family means no defender policy; it is not a
    # defense-policy family and cannot count as a held-out-family violation.
    train_families = [x for x in train.get("defender_families", []) if x != "none"]
    validation_families = [x for x in validation.get("defender_families", []) if x != "none"]
    if (not isinstance(train_families, list) or not train_families or
            not isinstance(validation_families, list) or not validation_families or
            any(not isinstance(x, str) or not x for x in train_families + validation_families)):
        raise ValueError("training and validation must declare defender-family sets")
    overlap = sorted(set(train_families) & set(validation_families))
    if overlap:
        raise ValueError("whole defender families overlap between train and validation")

    rows = protocol.get("cases")
    if not isinstance(rows, list):
        raise ValueError("protocol case table is missing")
    validation_cases = []
    observed_train_families: set[str] = set()
    observed_validation_families: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("protocol has a malformed case row")
        split = row.get("split")
        if split not in ("train", "validation", "test"):
            raise ValueError("protocol case has an unsupported split")
        if split == "train" and row.get("defender_family") != "none":
            observed_train_families.add(row.get("defender_family"))
        if split == "validation" and row.get("defender_family") != "none":
            observed_validation_families.add(row.get("defender_family"))
        if split != "validation":
            continue
        if any(not isinstance(row.get(field), str) or not row[field]
               for field in CASE_AXES):
            raise ValueError("validation case has incomplete factor axes")
        if (row["defender_family"] != "none" and
                row["defender_family"] not in validation_families):
            raise ValueError("validation case uses an undeclared defender family")
        validation_cases.append({field: row[field] for field in CASE_AXES})
    if not validation_cases:
        raise ValueError("protocol has no validation cases")
    if (not observed_train_families.issubset(set(train_families)) or
            not observed_validation_families.issubset(set(validation_families))):
        raise ValueError("protocol case rows use undeclared defender families")
    # A set removes duplicates without allowing any seed-bearing row into output.
    validation_cases = sorted({tuple(case[field] for field in CASE_AXES)
                               for case in validation_cases})
    axis_rows = [dict(zip(CASE_AXES, values)) for values in validation_cases]

    requirements = registry.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("design registry has no source-plan requirements")
    seen_requirements: set[str] = set()
    planned = []
    for req in requirements:
        if not isinstance(req, dict):
            raise ValueError("malformed Phase 4 requirement")
        req_id = req.get("requirement_id")
        factors = req.get("factor_ids")
        if not isinstance(req_id, str) or not req_id or req_id in seen_requirements:
            raise ValueError("Phase 4 requirement IDs must be unique and nonempty")
        seen_requirements.add(req_id)
        if not isinstance(factors, list) or not factors or any(
                not isinstance(x, str) or not x for x in factors):
            raise ValueError(f"{req_id} has no explicit factor IDs")
        runtime_ids = req.get("existing_runtime_factor_ids", [])
        if not isinstance(runtime_ids, list) or any(x not in PHASE4_FACTORS for x in runtime_ids):
            raise ValueError(f"{req_id} refers to an unknown existing runtime factor")
        selectors = req.get("validation_case_axes")
        if not isinstance(selectors, dict) or not selectors:
            raise ValueError(f"{req_id} has no validation case-axis selectors")
        protocol_constraints: dict[str, set[Any]] = {}
        design_selectors: dict[str, list[Any]] = {}
        implicit_scenarios: set[str] = set()
        for axis, selector in selectors.items():
            values, design, scoped_scenarios = _resolve_selector(
                axis, selector, protocol, registry)
            if values is not None:
                protocol_constraints[axis] = values
            for design_axis, design_values in design.items():
                if design_axis in design_selectors:
                    raise ValueError(f"{req_id} repeats design axis {design_axis}")
                design_selectors[design_axis] = design_values
            implicit_scenarios.update(scoped_scenarios)
        applicable = [case for case in axis_rows
                      if (not implicit_scenarios or
                          case["scenario_id"] in implicit_scenarios)
                      and all(case[axis] in values
                              for axis, values in protocol_constraints.items())]
        if not applicable:
            raise ValueError(f"{req_id} has no matching validation case axes")
        signals = req.get("required_outcome_signals")
        states = req.get("signal_status")
        if (not isinstance(signals, list) or not signals or not isinstance(states, dict)
                or set(signals) != set(states)):
            raise ValueError(f"{req_id} must status every required outcome signal")
        protocol_rows = [{"factor_id": factor, **case} for factor in factors
                         for case in applicable]
        design_rows = _expand_design_rows(factors, design_selectors)
        planned.append({
            "requirement_id": req_id,
            "factor_ids": list(factors),
            "existing_runtime_factor_ids": list(runtime_ids),
            "stimulus_status": req.get("stimulus_status"),
            "resolved_protocol_constraints": {
                axis: sorted(values) for axis, values in sorted(protocol_constraints.items())},
            "implicit_scenario_scope": sorted(implicit_scenarios),
            "protocol_bound_case_rows": protocol_rows,
            "protocol_bound_case_count_per_factor": len(applicable),
            "planned_paired_protocol_condition_rows_per_factor": 2 * len(applicable),
            "design_only_rows": design_rows,
            "design_only_row_count_per_factor": len(design_rows) // len(factors),
            "design_only_rows_are_protocol_coverage": False,
            "design_only_rows_are_executed_coverage": False,
            "required_outcome_signals": list(signals),
            "signal_status": dict(states),
        })
    if seen_requirements != SOURCE_PLAN_REQUIREMENTS:
        missing = sorted(SOURCE_PLAN_REQUIREMENTS - seen_requirements)
        extra = sorted(seen_requirements - SOURCE_PLAN_REQUIREMENTS)
        raise ValueError(f"registry requirement IDs differ from source plan; missing={missing}, extra={extra}")

    # No final-test case, seed, seed count, or test axis is projected. The raw
    # digest still binds the overlay to the complete protocol bytes.
    overlay = {
        "schema": OVERLAY_SCHEMA,
        "status": "design_only_not_executed",
        "qualification_eligible": False,
        "protocol_binding": {
            "protocol_id": protocol["protocol_id"],
            "protocol_raw_file_sha256": _sha256(protocol_raw),
        },
        "factor_registry_binding": {
            "registry_schema": registry["schema"],
            "registry_raw_file_sha256": _sha256(registry_raw),
        },
        "family_separation": {
            "axis": "defender_family",
            "train_families": sorted(set(train_families)),
            "validation_families": sorted(set(validation_families)),
            "overlap": [],
            "whole_family_holdout_verified": True,
        },
        "validation_case_axes": axis_rows,
        "requirements": planned,
        "planned_pairing": registry["pairing"],
        "design_axes": registry.get("design_axes", {}),
        "prohibitions": registry.get("prohibitions", []),
        "disclosure": {
            "validation_case_count": len(axis_rows),
            "contains_seed_values": False,
            "contains_final_test_rows": False,
            "safety_margins": None,
            "qualification_claim": False,
            "planned_coverage_is_not_executed_coverage": True,
        },
    }
    overlay["overlay_id"] = _sha256(_canonical_bytes(overlay))
    return overlay


def write_phase4_design_overlay(protocol_path: str | Path,
                                registry_path: str | Path,
                                output_path: str | Path) -> dict:
    overlay = compile_phase4_design_overlay(protocol_path, registry_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overlay, indent=2, sort_keys=True,
                              allow_nan=False) + "\n", encoding="utf-8")
    return overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, help="frozen MECO protocol JSON")
    parser.add_argument("--registry", required=True, help="Phase 4 design registry JSON")
    parser.add_argument("--output", required=True, help="validation-only overlay output JSON")
    args = parser.parse_args(argv)
    overlay = write_phase4_design_overlay(args.protocol, args.registry, args.output)
    print(json.dumps({"overlay_id": overlay["overlay_id"],
                      "protocol_id": overlay["protocol_binding"]["protocol_id"],
                      "validation_case_count": overlay["disclosure"]["validation_case_count"],
                      "requirement_count": len(overlay["requirements"]),
                      "qualification_eligible": False}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
