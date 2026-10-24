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


PLAN_SCHEMA = "meco-phase4-validation-campaign-plan-v1"
POLICIES = ("D0", "D1", "D2", "A1", "A2", "A3", "C1", "C2")
CONDITIONS = ("no_factor", "factor_treatment")
EXPECTED_FACTOR_CASE_AXIS_CELLS = 6488
VALIDATION_SEEDS_PER_CASE = 100


def _seed_digest(seeds: list[int]) -> str:
    """Bind exact validation seed assignment without returning its values."""
    return _sha256(_canonical_bytes(seeds))


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
                       "seed_count": len(seeds), "seed_sha256": _seed_digest(seeds)}
    return result


def compile_phase4_campaign_plan(protocol_path: str | Path,
                                registry_path: str | Path,
                                overlay_path: str | Path) -> dict[str, Any]:
    """Return a compact, deterministic schedule plan, never episode rows."""
    protocol, protocol_raw = _read_protocol_validation(protocol_path)
    registry, registry_raw = _read_json(registry_path)
    overlay, overlay_raw = _read_json(overlay_path)
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
                    cell = {
                        "requirement_id": req_id,
                        "factor_id": factor,
                        "case_axes": case,
                        "design_axes": design_axes,
                        "seed_count": seed_info["seed_count"],
                        "validation_seed_assignment_sha256": seed_info["seed_sha256"],
                        "planned_pair_rows_per_policy": 2 * seed_info["seed_count"],
                        "status": ("design_only_unassigned_to_protocol_case" if design_axes
                                   else "planned_validation_not_executed"),
                        "schedulable": True,
                        "runtime_factor_kind": runtime_binding[
                            "runtime_factor_by_factor_id"].get(factor),
                        "runtime_event": runtime_binding[
                            "runtime_event_by_factor_id"].get(factor),
                        "runtime_stimulus_supported": (
                            factor in runtime_binding["runtime_factor_by_factor_id"]
                            and runtime_binding["stimulus_supported"]),
                        "required_outcome_signals_complete": set(
                            runtime_binding["required_signals"]).issubset(
                                runtime_binding["authoritative_signals"]),
                        "executable": (not bool(design_axes)
                            and factor in runtime_binding["runtime_factor_by_factor_id"]
                            and runtime_binding["stimulus_supported"]
                            and set(runtime_binding["required_signals"]).issubset(
                                runtime_binding["authoritative_signals"])),
                        "executed": False,
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
    }
    plan["plan_id"] = _sha256(_canonical_bytes(plan))
    return plan


def write_phase4_campaign_plan(protocol_path: str | Path, registry_path: str | Path,
                               overlay_path: str | Path,
                               output_path: str | Path) -> dict[str, Any]:
    plan = compile_phase4_campaign_plan(protocol_path, registry_path, overlay_path)
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
    args = parser.parse_args(argv)
    plan = write_phase4_campaign_plan(args.protocol, args.registry, args.overlay, args.output)
    print(json.dumps({"plan_id": plan["plan_id"],
                      "factor_case_axis_cell_count": plan["factor_case_axis_cell_count"],
                      "scheduled_row_count_theoretical": plan["scheduled_row_count_theoretical"],
                      "episode_rows_materialized": plan["episode_rows_materialized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
