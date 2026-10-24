#!/usr/bin/env python3
"""Post-hoc cluster-bootstrap summaries for frozen MECO matrix reports.

This tool is diagnostic only. It does not evaluate simulator behavior or infer
promotion eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

# Direct execution sets sys.path to scripts/, while imports from the project
# root already have the repository package available.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from frc_defense.meco_phase4 import PHASE4_FACTORS, phase4_factor_digest


STRATUM_FIELDS = ("layout_id", "defender_family", "start_condition", "sensor_profile")
DEFAULT_REPLICATES = 10_000
DEFAULT_SEED = 20261024
PHASE4_POLICIES = ("A1", "A2", "A3", "C1", "C2", "D0", "D1", "D2")
MATRIX_SCHEMA_VERSION = 4


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot read evaluation report {path}: {exc}") from exc


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON report must contain an object: {path}")
    return value


def _finite_number(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number")
    return float(value)


def _cluster_bootstrap_ci(groups: dict[tuple[str, str], list[float]], *,
                          seed: int, replicates: int) -> list[float]:
    """Match qualification's deterministic cluster-then-row bootstrap."""
    keys = sorted(groups)
    if not keys or any(not groups[key] for key in keys):
        raise ValueError("cluster bootstrap requires populated strata")
    state = seed & ((1 << 64) - 1)

    def randbelow(limit: int) -> int:
        nonlocal state
        state ^= (state << 13) & ((1 << 64) - 1)
        state ^= state >> 7
        state ^= (state << 17) & ((1 << 64) - 1)
        return state % limit

    means = []
    for _ in range(replicates):
        sampled = []
        for _ in keys:
            rows = groups[keys[randbelow(len(keys))]]
            for _ in rows:
                sampled.append(rows[randbelow(len(rows))])
        means.append(sum(sampled) / len(sampled))
    means.sort()
    return [means[int(.025 * (replicates - 1))],
            means[int(.975 * (replicates - 1))]]


def _resolve_conditions(matrix: dict, matrix_path: Path) -> tuple[dict, dict]:
    protocol = matrix.get("protocol")
    conditions = matrix.get("conditions")
    if not isinstance(protocol, dict) or not isinstance(conditions, dict):
        raise ValueError("matrix report lacks protocol or condition mapping")
    if (matrix.get("schema_version") != MATRIX_SCHEMA_VERSION or
            protocol.get("schema_version") != MATRIX_SCHEMA_VERSION):
        raise ValueError("matrix report/protocol must use schema version 4")
    expected_matrix_hash = matrix.get("report_sha256")
    unhashed = dict(matrix)
    unhashed.pop("report_sha256", None)
    if not isinstance(expected_matrix_hash, str) or _canonical_hash(unhashed) != expected_matrix_hash:
        raise ValueError("matrix report SHA-256 does not match its contents")
    if protocol.get("matrix_id") != matrix.get("matrix_id"):
        raise ValueError("matrix protocol ID differs from report matrix ID")

    common = {key: matrix.get(key) for key in (
        "checkpoint_sha256", "training_configuration_sha256",
        "training_protocol_ids", "source_identity_sha256", "execution_device",
        "evaluation_design", "evaluation_split", "qualification_protocol_id",
        "qualification_protocol", "qualification_case", "seed_split_assignments")}
    for key, value in common.items():
        protocol_key = key
        if protocol.get(protocol_key) != value:
            raise ValueError(f"matrix and protocol provenance differ for {key}")
    if (not isinstance(protocol.get("seed_list"), list) or
            len(set(protocol["seed_list"]) ) != len(protocol["seed_list"]) or
            not protocol["seed_list"]):
        raise ValueError("matrix seed list must be nonempty and unique")
    if (not isinstance(protocol.get("categories"), list) or
            len(set(protocol["categories"])) != len(protocol["categories"]) or
            not protocol["categories"]):
        raise ValueError("matrix category list must be nonempty and unique")
    if (common["evaluation_split"] in ("validation", "test") and
            common["evaluation_design"] != "frozen_protocol_held_out"):
        raise ValueError("held-out matrix split lacks frozen protocol evaluation design")
    if common["evaluation_split"] not in ("validation", "test") and (
            common["evaluation_design"] != "diagnostic_only_unbound_seeds" or
            common["qualification_protocol_id"] is not None or
            common["qualification_protocol"] is not None):
        raise ValueError("unbound matrix split has inconsistent protocol provenance")
    if common["evaluation_split"] in ("validation", "test"):
        qualification_protocol = common["qualification_protocol"]
        if (not isinstance(qualification_protocol, dict) or
                not isinstance(qualification_protocol.get("protocol_id"), str) or
                not qualification_protocol["protocol_id"] or
                not isinstance(qualification_protocol.get("protocol_sha256"), str) or
                not qualification_protocol["protocol_sha256"] or
                qualification_protocol["protocol_id"] != common["qualification_protocol_id"]):
            raise ValueError("held-out matrix lacks bound protocol-v2 ID/SHA-256 provenance")
    assignments = common["seed_split_assignments"]
    if assignments is not None:
        for category in protocol["categories"]:
            selected = assignments.get(category)
            if not isinstance(selected, dict):
                raise ValueError(f"matrix has no split assignment for {category}")
            if any(selected.get(str(seed)) != common["evaluation_split"]
                   for seed in protocol["seed_list"]):
                raise ValueError(f"matrix seed split differs for {category}")

    reports, file_info = {}, {}
    by_id = {}
    condition_specs = protocol.get("conditions")
    if not isinstance(condition_specs, list) or any(not isinstance(spec, dict)
                                                    for spec in condition_specs):
        raise ValueError("matrix protocol condition specs are malformed")
    specs = {spec.get("condition_name"): spec for spec in condition_specs}
    if (len(specs) != len(condition_specs) or
            set(specs) != set(conditions)):
        raise ValueError("matrix condition names differ from frozen protocol specs")
    base = matrix_path.parent
    for name, condition in conditions.items():
        if not isinstance(condition, dict) or not condition.get("condition_id"):
            raise ValueError(f"invalid condition manifest: {name}")
        condition_id = condition["condition_id"]
        if condition_id in by_id:
            raise ValueError(f"duplicate condition ID: {condition_id}")
        by_id[condition_id] = name
        raw_path = condition.get("report_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"condition {name} has no report path")
        report_path = Path(raw_path)
        if not report_path.is_absolute():
            report_path = base / report_path
        report_path = report_path.resolve()
        digest = _file_sha256(report_path)
        report = _load_json(report_path)
        if report.get("schema_version") != 1:
            raise ValueError(f"evaluation report schema version is unsupported: {name}")
        experiment = report.get("experiment_matrix")
        spec = specs.get(name)
        if (not isinstance(experiment, dict) or spec is None or
                experiment.get("matrix_id") != matrix.get("matrix_id") or
                experiment.get("condition_id") != condition_id or
                experiment.get("condition") != spec):
            raise ValueError(f"condition report mapping mismatch: {name}")
        for key in ("evaluation_design", "evaluation_split",
                    "qualification_protocol_id", "qualification_protocol",
                    "seed_split_assignments"):
            if experiment.get(key) != common.get(key):
                raise ValueError(f"condition report {key} differs from matrix provenance: {name}")
        checks = {
            "evaluation_id": report.get("evaluation_id"),
            "protocol_id": report.get("protocol_id"),
            "source_identity_sha256": report.get("source_identity_sha256"),
            "checkpoint_sha256": condition.get("checkpoint_sha256"),
            "qualification_case": report.get("qualification_case"),
            "evaluation_split": experiment.get("evaluation_split"),
        }
        if condition.get("evaluation_id") != checks["evaluation_id"]:
            raise ValueError(f"evaluation ID mismatch: {name}")
        if condition.get("protocol_id") != checks["protocol_id"]:
            raise ValueError(f"evaluation protocol mismatch: {name}")
        if (condition.get("qualification_protocol") != common["qualification_protocol"] or
                report.get("qualification_protocol_id") != common["qualification_protocol_id"]):
            raise ValueError(f"qualification protocol provenance mismatch: {name}")
        if checks["source_identity_sha256"] != common["source_identity_sha256"]:
            raise ValueError(f"source provenance mismatch: {name}")
        if condition.get("checkpoint_sha256") != common["checkpoint_sha256"]:
            raise ValueError(f"checkpoint provenance mismatch: {name}")
        if condition.get("qualification_case") != common["qualification_case"]:
            raise ValueError(f"condition stratum mismatch: {name}")
        if condition.get("execution_device") != common["execution_device"]:
            raise ValueError(f"condition device mismatch: {name}")
        if checks["evaluation_split"] != common["evaluation_split"]:
            raise ValueError(f"evaluation split mismatch: {name}")
        if report.get("scenario_id") != protocol.get("scenario_id"):
            raise ValueError(f"scenario mismatch: {name}")
        if report.get("execution_device") != common["execution_device"]:
            raise ValueError(f"execution device mismatch: {name}")
        cats = spec.get("categories") or protocol["categories"]
        if report.get("categories") != cats or report.get("seeds") != protocol["seed_list"]:
            raise ValueError(f"category/seed coverage mismatch: {name}")
        if common["evaluation_split"] in ("validation", "test"):
            frozen = common["qualification_protocol"]
            binding = report.get("evaluation_case")
            expected_seeds = protocol["seed_list"]
            if (report.get("evaluation_protocol_id") != frozen["protocol_id"] or
                    report.get("evaluation_protocol_sha256") != frozen["protocol_sha256"] or
                    report.get("evaluation_design") != "frozen_protocol_held_out" or
                    report.get("split") != common["evaluation_split"] or
                    report.get("exact_episode_seeds") != expected_seeds or
                    not isinstance(binding, dict) or
                    binding.get("protocol_id") != frozen["protocol_id"] or
                    binding.get("protocol_sha256") != frozen["protocol_sha256"] or
                    binding.get("protocol_file") != frozen.get("protocol_file") or
                    binding.get("split") != common["evaluation_split"] or
                    binding.get("scenario_id") != protocol.get("scenario_id") or
                    binding.get("categories") != cats or
                    binding.get("qualification_case") != common["qualification_case"] or
                    binding.get("seeds") != expected_seeds):
                raise ValueError(f"evaluation protocol-v2 binding differs from matrix: {name}")
            case_ids = binding.get("case_ids")
            expected_case_ids = [{"category": category, **common["qualification_case"]}
                                 for category in cats]
            if case_ids != expected_case_ids:
                raise ValueError(f"evaluation protocol-v2 case identity differs from matrix: {name}")
        elif (report.get("evaluation_protocol_id") is not None or
                report.get("evaluation_protocol_sha256") is not None or
                report.get("evaluation_case") is not None or
                report.get("exact_episode_seeds") is not None or
                report.get("evaluation_design") != "diagnostic_unbound"):
            raise ValueError(f"diagnostic report carries held-out protocol binding: {name}")
        manifest = report.get("policy_manifest")
        if not isinstance(manifest, dict) or condition.get("policy_manifest") != manifest:
            raise ValueError(f"policy manifest missing: {name}")
        for policy, entry in manifest.items():
            if not isinstance(entry, dict):
                raise ValueError(f"policy manifest entry is malformed for {policy}")
            track = policy if policy in common["checkpoint_sha256"] else None
            if track is None and policy.endswith("_P0"):
                track = policy[:-3]
            if track is None and "__INFERENCE_DISABLED_TO_D1" in policy:
                track = policy.split("__INFERENCE_DISABLED_TO_D1", 1)[0]
            if track in common["checkpoint_sha256"]:
                expected = {
                    "checkpoint_sha256": common["checkpoint_sha256"][track],
                    "training_configuration_sha256": common["training_configuration_sha256"][track],
                    "training_protocol_id": common["training_protocol_ids"][track],
                    "source_identity_sha256": common["source_identity_sha256"],
                }
                if any(entry.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"policy provenance mismatch for {policy} in {name}")
        records = {}
        report_categories, report_seeds = set(report.get("categories", [])), set(report.get("seeds", []))
        for row in report.get("records", []):
            key = (row.get("policy"), row.get("category"), row.get("seed"))
            if row.get("category") not in report_categories or row.get("seed") not in report_seeds:
                raise ValueError(f"unexpected evaluation record outside seed/category sample: {key}")
            if key in records:
                raise ValueError(f"duplicate evaluation record {key} in {name}")
            stratum = {field: row.get(field) for field in STRATUM_FIELDS}
            if stratum != common["qualification_case"]:
                raise ValueError(f"record stratum mismatch for {key} in {name}")
            records[key] = row
        reports[condition_id] = (report, records)
        file_info[condition_id] = {"path": str(report_path), "sha256": digest,
                                   "evaluation_id": report["evaluation_id"]}
    common["scenario_id"] = protocol.get("scenario_id")
    return {"common": common, "by_id": by_id, "specs": specs,
            "conditions": conditions, "reports": reports, "files": file_info}


def _check_common_contrast_provenance(contrast: dict, common: dict) -> None:
    observed = contrast.get("common_bundle_provenance")
    if not isinstance(observed, dict):
        raise ValueError("contrast lacks common bundle provenance")
    for key in ("checkpoint_sha256", "training_configuration_sha256",
                "training_protocol_ids", "source_identity_sha256", "execution_device",
                "evaluation_design", "evaluation_split", "qualification_case",
                "qualification_protocol_id", "qualification_protocol",
                "seed_split_assignments", "scenario_id"):
        if observed.get(key) != common.get(key):
            raise ValueError(f"contrast common provenance mismatch for {key}")


def _row_for(report_records: dict, policy: str, category: str, seed: int,
             where: str) -> dict:
    row = report_records.get((policy, category, seed))
    if row is None:
        raise ValueError(f"missing {where} record for {policy}/{category}/{seed}")
    return row


def _check_record_policy_id(report: dict, row: dict, policy: str) -> None:
    manifest = report.get("policy_manifest", {}).get(policy)
    if not isinstance(manifest, dict) or not manifest.get("policy_id"):
        raise ValueError(f"evaluation manifest lacks policy ID for {policy}")
    if row.get("policy_id") != manifest["policy_id"]:
        raise ValueError(f"record policy ID differs from manifest for {policy}")


def _stratum(row: dict) -> tuple[str, str]:
    layout, family = row.get("layout_id"), row.get("defender_family")
    if not isinstance(layout, str) or not layout or not isinstance(family, str) or not family:
        raise ValueError("record lacks valid layout_id/defender_family bootstrap cluster")
    return layout, family


def _add_metric_difference(output: list, *, contrast_name: str, category: str,
                           metric: str, paired: list[tuple[dict, dict, dict]],
                           candidate_policy: str, reference_policy: str,
                           bootstrap_seed: int, replicates: int) -> None:
    groups: dict[tuple[str, str], list[float]] = {}
    values = []
    for candidate_row, reference_row, copied in paired:
        cm = candidate_row.get("metrics")
        rm = reference_row.get("metrics")
        if not isinstance(cm, dict) or not isinstance(rm, dict):
            raise ValueError("paired evaluation record lacks metrics object")
        if metric not in cm or metric not in rm:
            raise ValueError(f"unpaired metric {metric} in {contrast_name}/{category}")
        cvalue, rvalue = _finite_number(cm[metric], metric), _finite_number(rm[metric], metric)
        if copied is not None:
            for label, expected in (("candidate", cvalue), ("reference", rvalue)):
                source = copied[f"{label}_metrics"]
                if (not isinstance(source, dict) or metric not in source or
                        _finite_number(source[metric], metric) != expected):
                    raise ValueError(f"contrast copied {label} metric differs from source report")
        if _stratum(candidate_row) != _stratum(reference_row):
            raise ValueError("paired rows have incompatible bootstrap clusters")
        diff = cvalue - rvalue
        cluster = _stratum(candidate_row)
        groups.setdefault(cluster, []).append(diff)
        values.append(diff)
    if not values:
        raise ValueError("contrast contains no paired metric observations")
    output.append({"contrast": contrast_name, "category": category,
        "candidate_policy": candidate_policy, "reference_policy": reference_policy,
        "metric": metric, "n_pairs": len(values), "mean_difference": sum(values) / len(values),
        "ci_95": _cluster_bootstrap_ci(groups, seed=bootstrap_seed,
                                        replicates=replicates),
        "cluster_count": len(groups), "cluster_fields": list(STRATUM_FIELDS[:2]),
        "difference_direction": "candidate_minus_reference",
        "_observations": [{"seed": pair[0].get("seed"),
            "cluster": list(_stratum(pair[0])), "difference":
                _finite_number(pair[0]["metrics"][metric], metric) -
                _finite_number(pair[1]["metrics"][metric], metric)}
            for pair in [(item[0], item[1]) for item in paired]]})


def summarize_matrix(matrix_path: str | Path, *, bootstrap_seed: int = DEFAULT_SEED,
                     replicates: int = DEFAULT_REPLICATES) -> dict:
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if type(replicates) is not int or replicates < 100:
        raise ValueError("replicates must be an integer of at least 100")
    path = Path(matrix_path).resolve()
    matrix = _load_json(path)
    if not isinstance(matrix.get("contrasts"), list) or not matrix["contrasts"]:
        raise ValueError("matrix report has no contrast rows")
    loaded = _resolve_conditions(matrix, path)
    summaries = []
    seen_rows: set[tuple] = set()
    matched_coverage: dict[tuple[str, str], set[tuple[str, str, int]]] = {}
    recall_groups: dict[tuple[str, str, int], list[dict]] = {}
    phase4_groups: dict[tuple[str, str, str, str], list[dict]] = {}
    phase4_seen: set[tuple[str, str, str, int]] = set()
    phase4_coverage: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for contrast in matrix.get("contrasts", []):
        if not isinstance(contrast, dict) or contrast.get("paired") is not True:
            raise ValueError("every summarized contrast must be marked paired")
        _check_common_contrast_provenance(contrast, loaded["common"])
        kind = contrast.get("contrast")
        if kind == "candidate_recall_K_sweep":
            condition_id = contrast.get("condition_id")
            if condition_id not in loaded["reports"]:
                raise ValueError("candidate recall references unknown condition")
            report, records = loaded["reports"][condition_id]
            policy = contrast.get("candidate_policy")
            match = re.fullmatch(r"([A-Z][0-9])@K=(8|16|32)", str(policy))
            if not match:
                raise ValueError("candidate recall policy has invalid K identity")
            track, k = match.group(1), int(match.group(2))
            category, seed = contrast.get("category"), contrast.get("seed")
            row = _row_for(records, track, category, seed, "candidate recall")
            _check_record_policy_id(report, row, track)
            if _finite_number(contrast.get("candidate_recall"), "candidate_recall") != _finite_number(row.get("candidate_recall"), "candidate_recall"):
                raise ValueError("candidate recall contrast differs from source record")
            candidate_k = report.get("candidate_k", {}).get("effective")
            if candidate_k != k:
                raise ValueError("candidate recall K differs from evaluation report")
            key = (kind, track, category, seed, k)
            if key in seen_rows:
                raise ValueError("duplicate candidate recall category/seed/K row")
            seen_rows.add(key)
            recall_groups.setdefault((track, category, k), []).append({
                "seed": seed, "cluster": list(_stratum(row)),
                "value": _finite_number(row.get("candidate_recall"), "candidate_recall")})
            continue

        if kind == "matched_prediction_current_vs_cv":
            ids = contrast.get("condition_ids", {})
            if set(ids) != {"current", "cv"}:
                raise ValueError("matched prediction contrast lacks current/CV condition mapping")
            policy_cv, policy_current = contrast.get("candidate_policy"), contrast.get("reference_policy")
            if (not isinstance(policy_cv, str) or not policy_cv.endswith("@cv") or
                    not isinstance(policy_current, str) or not policy_current.endswith("@current") or
                    policy_cv[:-3] != policy_current[:-8]):
                raise ValueError("matched prediction policy mapping is inconsistent")
            track = policy_cv[:-3]
            current_report, current_records = loaded["reports"].get(ids["current"], (None, None))
            cv_report, cv_records = loaded["reports"].get(ids["cv"], (None, None))
            if current_report is None or cv_report is None:
                raise ValueError("matched prediction references unknown condition")
            category, seed = contrast.get("category"), contrast.get("seed")
            coverage_key = (ids["current"], ids["cv"])
            matched_coverage.setdefault(coverage_key, set()).add((track, category, seed))
            current_row = _row_for(current_records, track, category, seed, "current mode")
            cv_row = _row_for(cv_records, track, category, seed, "CV mode")
            _check_record_policy_id(current_report, current_row, track)
            _check_record_policy_id(cv_report, cv_row, track)
            for key in ("current_metrics", "cv_metrics"):
                source = current_row.get("metrics") if key == "current_metrics" else cv_row.get("metrics")
                if contrast.get(key) != source:
                    raise ValueError(f"matched prediction {key} differs from source report")
            if _stratum(current_row) != _stratum(cv_row):
                raise ValueError("matched prediction rows have mismatched strata")
            current_metrics, cv_metrics = current_row.get("metrics", {}), cv_row.get("metrics", {})
            common_metrics = sorted(key for key in set(current_metrics) & set(cv_metrics)
                if type(current_metrics[key]) in (int, float) and math.isfinite(current_metrics[key])
                and type(cv_metrics[key]) in (int, float) and math.isfinite(cv_metrics[key]))
            for metric in common_metrics:
                pair_key = (kind, track, category, seed, metric)
                if pair_key in seen_rows:
                    raise ValueError("duplicate matched prediction category/seed row")
                # Each individual contrast is one paired observation, so combine below by key.
                seen_rows.add(pair_key)
                diff = (_finite_number(cv_row["metrics"].get(metric), metric) -
                        _finite_number(current_row["metrics"].get(metric), metric))
                summaries.append({"contrast": kind, "category": category,
                    "policy": track, "seed": seed, "metric": metric,
                    "candidate_policy": policy_cv, "reference_policy": policy_current,
                    "difference": diff, "difference_direction": "cv_minus_current",
                    "cluster": list(_stratum(cv_row)), "_observations": [{
                        "seed": seed, "cluster": list(_stratum(cv_row)),
                        "difference": diff}]})
            continue

        if kind == "phase4_factor_vs_no_factor":
            factor = contrast.get("factor")
            factor_id = contrast.get("condition_ids", {}).get("factor")
            baseline_id = contrast.get("condition_ids", {}).get("baseline")
            if (not isinstance(factor, str) or factor not in PHASE4_FACTORS or
                    not isinstance(factor_id, str) or not isinstance(baseline_id, str)):
                raise ValueError("Phase 4 contrast has an unregistered factor or invalid condition IDs")
            if factor_id not in loaded["reports"] or baseline_id not in loaded["reports"]:
                raise ValueError("Phase 4 contrast references unknown condition")
            factor_name = loaded["by_id"].get(factor_id)
            baseline_name = loaded["by_id"].get(baseline_id)
            factor_condition = loaded["conditions"][factor_name]
            baseline_condition = loaded["conditions"][baseline_name]
            factor_spec = loaded["specs"].get(factor_name, {}).get("phase4_factor")
            if (factor_spec != PHASE4_FACTORS[factor] or
                    factor_condition.get("phase4_factor") != factor_spec or
                    baseline_condition.get("phase4_factor") is not None or
                    loaded["specs"].get(baseline_name, {}).get("phase4_factor") is not None):
                raise ValueError("Phase 4 factor identity differs from condition manifest")
            factor_report, factor_records = loaded["reports"][factor_id]
            baseline_report, baseline_records = loaded["reports"][baseline_id]
            factor_digest = phase4_factor_digest(factor_spec)
            if (factor_report.get("phase4_factor") != factor_spec or
                    baseline_report.get("phase4_factor") is not None or
                    contrast.get("factor_sha256") != factor_digest):
                raise ValueError("Phase 4 factor config/digest differs from evaluation reports")
            policy = contrast.get("candidate_policy")
            if not isinstance(policy, str) or contrast.get("reference_policy") != policy:
                raise ValueError("Phase 4 comparison must pair the same policy")
            category, seed = contrast.get("category"), contrast.get("seed")
            case = contrast.get("qualification_case")
            if case != loaded["common"]["qualification_case"]:
                raise ValueError("Phase 4 contrast qualification stratum differs from matrix")
            key = (factor, policy, category, seed)
            if key in phase4_seen:
                raise ValueError("duplicate Phase 4 factor/policy/category/seed row")
            phase4_seen.add(key)
            candidate_row = _row_for(factor_records, policy, category, seed, "Phase 4 factor")
            baseline_row = _row_for(baseline_records, policy, category, seed, "Phase 4 baseline")
            _check_record_policy_id(factor_report, candidate_row, policy)
            _check_record_policy_id(baseline_report, baseline_row, policy)
            if candidate_row.get("policy_id") != baseline_row.get("policy_id"):
                raise ValueError("Phase 4 treatment/baseline policy IDs differ")
            candidate_metrics, baseline_metrics = candidate_row.get("metrics"), baseline_row.get("metrics")
            if (not isinstance(candidate_metrics, dict) or not isinstance(baseline_metrics, dict) or
                    contrast.get("factor_metrics") != candidate_metrics or
                    contrast.get("baseline_metrics") != baseline_metrics):
                raise ValueError("Phase 4 copied metrics differ from source evaluation records")
            if set(candidate_metrics) != set(baseline_metrics):
                raise ValueError("Phase 4 treatment/baseline metric schemas differ")
            if ({field: candidate_row.get(field) for field in STRATUM_FIELDS} != case or
                    {field: baseline_row.get(field) for field in STRATUM_FIELDS} != case):
                raise ValueError("Phase 4 treatment/baseline records have mismatched strata")
            coverage_key = (factor, policy)
            phase4_coverage.setdefault(coverage_key, set()).add((category, seed))
            for metric in sorted(candidate_metrics):
                candidate_value, baseline_value = candidate_metrics[metric], baseline_metrics[metric]
                if candidate_value is None and baseline_value is None:
                    continue
                if (type(candidate_value) not in (int, float) or
                        not math.isfinite(candidate_value) or
                        type(baseline_value) not in (int, float) or
                        not math.isfinite(baseline_value)):
                    raise ValueError(f"Phase 4 paired metric {metric} is missing or non-finite")
                phase4_groups.setdefault((factor, policy, category, metric), []).append({
                    "seed": seed, "cluster": list(_stratum(candidate_row)),
                    "difference": float(candidate_value) - float(baseline_value)})
            continue

        # Standard within-report and cross-condition paired metric contrasts.
        condition_ids = contrast.get("condition_ids")
        if condition_ids:
            if not isinstance(condition_ids, dict) or set(condition_ids) != {"factor", "baseline"}:
                raise ValueError("cross-condition contrast has unsupported condition mapping")
            candidate_id, reference_id = condition_ids["factor"], condition_ids["baseline"]
        else:
            candidate_id = reference_id = contrast.get("condition_id")
        if candidate_id not in loaded["reports"] or reference_id not in loaded["reports"]:
            raise ValueError("paired contrast references unknown condition")
        candidate_report, candidate_records = loaded["reports"][candidate_id]
        reference_report, reference_records = loaded["reports"][reference_id]
        candidate_policy, reference_policy = contrast.get("candidate_policy"), contrast.get("reference_policy")
        if not isinstance(candidate_policy, str) or not isinstance(reference_policy, str):
            raise ValueError("contrast lacks policy mapping")
        rows = contrast.get("paired_episode_rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("contrast lacks paired episode rows")
        grouped: dict[str, dict[str, list[tuple[dict, dict, dict]]]] = {}
        local_keys = set()
        for item in rows:
            category, seed = item.get("category"), item.get("seed")
            rowkey = (kind, candidate_policy, reference_policy, category, seed)
            if rowkey in local_keys or rowkey in seen_rows:
                raise ValueError("duplicate paired category/seed row")
            local_keys.add(rowkey)
            cr = _row_for(candidate_records, candidate_policy, category, seed, "candidate")
            rr = _row_for(reference_records, reference_policy, category, seed, "reference")
            if item.get("candidate_record_policy_id") != cr.get("policy_id") or item.get("reference_record_policy_id") != rr.get("policy_id"):
                raise ValueError("contrast policy ID differs from source record")
            cm, rm = cr.get("metrics"), rr.get("metrics")
            if item.get("candidate_metrics") != cm or item.get("reference_metrics") != rm:
                raise ValueError("contrast copied metrics differ from source report")
            if _stratum(cr) != _stratum(rr):
                raise ValueError("paired category/seed rows have mismatched strata")
            grouped.setdefault(category, {}).setdefault("pairs", []).append((cr, rr, item))
            seen_rows.add(rowkey)
        # Require full exact seed coverage per category, with no unpaired/missing rows.
        expected_categories = set(candidate_report.get("categories", []))
        if set(grouped) != expected_categories:
            raise ValueError("paired contrast has missing or unexpected category rows")
        for category, data in grouped.items():
            requested = [seed for seed in candidate_report.get("seeds", []) if category in candidate_report.get("categories", [])]
            observed = [item[0].get("seed") for item in data["pairs"]]
            if len(observed) != len(set(observed)) or set(observed) != set(requested):
                raise ValueError(f"paired contrast has missing/duplicate seed rows for {category}")
            numeric_metrics = set.intersection(*[
                {key for key, value in pair[0].get("metrics", {}).items()
                 if type(value) in (int, float) and math.isfinite(value)} &
                {key for key, value in pair[1].get("metrics", {}).items()
                 if type(value) in (int, float) and math.isfinite(value)}
                for pair in data["pairs"]])
            for metric in sorted(numeric_metrics):
                _add_metric_difference(summaries, contrast_name=kind, category=category,
                    metric=metric, paired=data["pairs"], candidate_policy=candidate_policy,
                    reference_policy=reference_policy, bootstrap_seed=bootstrap_seed,
                    replicates=replicates)

    for (current_id, cv_id), observed in matched_coverage.items():
        current_report, _ = loaded["reports"][current_id]
        cv_report, _ = loaded["reports"][cv_id]
        if (current_report.get("categories") != cv_report.get("categories") or
                current_report.get("seeds") != cv_report.get("seeds")):
            raise ValueError("matched prediction conditions have incompatible sample coverage")
        expected = {(track, category, seed)
            for track in loaded["common"]["checkpoint_sha256"]
            for category in current_report["categories"]
            for seed in current_report["seeds"]}
        if observed != expected:
            raise ValueError("matched prediction contrast has missing track/category/seed rows")

    for (track, category, k), observations in sorted(recall_groups.items()):
        groups: dict[tuple[str, str], list[float]] = {}
        for observation in observations:
            groups.setdefault(tuple(observation["cluster"]), []).append(observation["value"])
        values = [observation["value"] for observation in observations]
        summaries.append({"contrast": "candidate_recall_K_sweep", "category": category,
            "policy": track, "candidate_k": k, "n_episodes": len(values),
            "mean_candidate_recall": sum(values) / len(values),
            "ci_95": _cluster_bootstrap_ci(groups, seed=bootstrap_seed ^ k,
                                            replicates=replicates),
            "cluster_count": len(groups), "cluster_fields": list(STRATUM_FIELDS[:2]),
            "_observations": observations})

    expected_factors = {spec["phase4_factor"]["kind"]
        for spec in loaded["specs"].values()
        if isinstance(spec.get("phase4_factor"), dict)}
    observed_factors = {factor for factor, _ in phase4_coverage}
    if observed_factors != expected_factors:
        raise ValueError("Phase 4 contrast coverage differs from registered treatment conditions")
    expected_factor_policies = {(factor, policy) for factor in expected_factors
                                for policy in PHASE4_POLICIES}
    if set(phase4_coverage) != expected_factor_policies:
        raise ValueError("Phase 4 contrast coverage is missing a required factor/policy pair")
    for (factor, policy), observed in phase4_coverage.items():
        treatment_id = next(condition_id for condition_id, name in loaded["by_id"].items()
            if loaded["specs"][name].get("phase4_factor", {}).get("kind") == factor)
        treatment_report, _ = loaded["reports"][treatment_id]
        expected = {(category, seed) for category in treatment_report["categories"]
                    for seed in treatment_report["seeds"]}
        if observed != expected:
            raise ValueError(f"Phase 4 contrast has incomplete category/seed coverage for {factor}/{policy}")
    phase4_summaries: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for (factor, policy, category, metric), observations in phase4_groups.items():
        phase4_summaries.setdefault((factor, policy, category), {})[metric] = observations
    for (factor, policy, category), metric_groups in sorted(phase4_summaries.items()):
        for metric, observations in sorted(metric_groups.items()):
            clusters: dict[tuple[str, str], list[float]] = {}
            values = []
            for observation in observations:
                value = observation["difference"]
                clusters.setdefault(tuple(observation["cluster"]), []).append(value)
                values.append(value)
            summaries.append({"contrast": "phase4_factor_vs_no_factor", "factor": factor,
                "category": category, "candidate_policy": policy, "reference_policy": policy,
                "metric": metric, "n_pairs": len(values), "mean_difference": sum(values) / len(values),
                "ci_95": _cluster_bootstrap_ci(clusters, seed=bootstrap_seed,
                                                replicates=replicates),
                "cluster_count": len(clusters), "cluster_fields": list(STRATUM_FIELDS[:2]),
                "difference_direction": "candidate_minus_reference",
                "_observations": observations})

    return {"schema": "meco-ablation-bootstrap-summary-v1",
        "diagnostic_only": True,
        "promotion_inference": False,
        "matrix": {"path": str(path), "file_sha256": _file_sha256(path),
                   "report_sha256": matrix["report_sha256"],
                   "matrix_id": matrix.get("matrix_id")},
        "evaluation_reports": loaded["files"],
        "provenance": loaded["common"],
        "bootstrap": {"method": "cluster_then_paired_episode_resampling",
            "cluster_fields": ["layout_id", "defender_family"],
            "seed": bootstrap_seed, "replicates": replicates,
            "interval": "percentile_95", "percentiles": [2.5, 97.5]},
        "summaries": summaries,
        "interpretation": "Diagnostic paired intervals only; no promotion or qualification conclusion."}


def summarize_matrices(matrix_paths: list[str | Path], *,
                       bootstrap_seed: int = DEFAULT_SEED,
                       replicates: int = DEFAULT_REPLICATES) -> dict:
    """Pool compatible held-out matrices after rejecting sample overlap."""
    if not matrix_paths:
        raise ValueError("at least one input matrix is required")
    results = [summarize_matrix(path, bootstrap_seed=bootstrap_seed,
                                replicates=replicates) for path in matrix_paths]
    common = results[0]["provenance"]
    compatible_fields = ("checkpoint_sha256", "training_configuration_sha256",
        "training_protocol_ids", "source_identity_sha256", "execution_device",
        "evaluation_design", "evaluation_split", "qualification_protocol_id",
        "scenario_id")

    def protocol_identity(value: Any) -> tuple[str, str] | None:
        if value is None:
            return None
        if (not isinstance(value, dict) or not isinstance(value.get("protocol_id"), str)
                or not isinstance(value.get("protocol_sha256"), str)):
            raise ValueError("qualification protocol provenance lacks ID or SHA-256")
        return value["protocol_id"], value["protocol_sha256"]

    common_protocol = protocol_identity(common.get("qualification_protocol"))
    for result in results[1:]:
        for field in compatible_fields:
            if result["provenance"].get(field) != common.get(field):
                raise ValueError(f"input matrices have incompatible {field}")
        if protocol_identity(result["provenance"].get("qualification_protocol")) != common_protocol:
            raise ValueError("input matrices have incompatible qualification protocol ID/SHA-256")
    seen_samples = set()
    groups: dict[tuple, list[dict]] = {}
    for result in results:
        matrix_samples = set()
        case = result["provenance"].get("qualification_case")
        if not isinstance(case, dict) or set(case) != set(STRATUM_FIELDS):
            raise ValueError("input matrix has incomplete frozen qualification stratum")
        for item in result["summaries"]:
            for observation in item["_observations"]:
                sample = (item["category"], observation["seed"],
                          tuple(case[field] for field in STRATUM_FIELDS))
                matrix_samples.add(sample)
                signature = tuple(item.get(field) for field in (
                    "contrast", "factor", "category", "candidate_policy", "reference_policy",
                    "policy", "metric", "candidate_k"))
                groups.setdefault(signature, []).append({**observation,
                    "full_stratum": case})
        if seen_samples.intersection(matrix_samples):
            raise ValueError("input matrices contain overlapping category/seed/stratum samples")
        seen_samples.update(matrix_samples)

    summaries = []
    for signature, observations in sorted(groups.items(), key=lambda item: repr(item[0])):
        contrast, factor, category, candidate, reference, policy, metric, k = signature
        clusters: dict[tuple[str, str], list[float]] = {}
        value_key = "value" if contrast == "candidate_recall_K_sweep" else "difference"
        values = []
        for observation in observations:
            value = _finite_number(observation[value_key], value_key)
            cluster = tuple(observation["cluster"])
            clusters.setdefault(cluster, []).append(value)
            values.append(value)
        row = {"contrast": contrast, "category": category,
            "n_episodes": len(values), "cluster_count": len(clusters),
            "cluster_fields": list(STRATUM_FIELDS[:2]),
            "ci_95": _cluster_bootstrap_ci(clusters,
                seed=bootstrap_seed ^ (k or 0), replicates=replicates)}
        if contrast == "candidate_recall_K_sweep":
            row.update({"policy": policy, "candidate_k": k,
                        "mean_candidate_recall": sum(values) / len(values)})
        elif contrast == "phase4_factor_vs_no_factor":
            row.update({"factor": factor, "candidate_policy": candidate,
                        "reference_policy": reference, "metric": metric,
                        "mean_difference": sum(values) / len(values),
                        "difference_direction": "candidate_minus_reference"})
        else:
            row.update({"candidate_policy": candidate, "reference_policy": reference,
                        "metric": metric, "mean_difference": sum(values) / len(values),
                        "difference_direction": ("cv_minus_current" if contrast ==
                            "matched_prediction_current_vs_cv" else "candidate_minus_reference")})
            if policy:
                row["policy"] = policy
        summaries.append(row)
    matrix_info = [result["matrix"] for result in results]
    report_info = {}
    for result in results:
        report_info.update(result["evaluation_reports"])
    return {"schema": "meco-ablation-bootstrap-summary-v1",
        "diagnostic_only": True, "promotion_inference": False,
        "matrices": matrix_info, "evaluation_reports": report_info,
        "provenance": {key: common.get(key) for key in compatible_fields},
        "qualification_protocol": common.get("qualification_protocol"),
        "strata": [result["provenance"]["qualification_case"] for result in results],
        "bootstrap": {"method": "cluster_then_paired_episode_resampling",
            "cluster_fields": ["layout_id", "defender_family"],
            "seed": bootstrap_seed, "replicates": replicates,
            "interval": "percentile_95", "percentiles": [2.5, 97.5]},
        "summaries": summaries,
        "interpretation": "Diagnostic paired intervals only; no promotion or qualification conclusion."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-matrix", required=True, action="append", type=Path,
                        help="matrix JSON to include; repeat for compatible held-out strata")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    args = parser.parse_args(argv)
    result = summarize_matrices(args.input_matrix, bootstrap_seed=args.bootstrap_seed,
                                replicates=args.replicates)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
