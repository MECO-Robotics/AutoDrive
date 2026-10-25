"""Frozen MECO qualification protocol and promotion report consumer.

Diagnostic evaluations remain separate. Qualification reports must contain
one immutable row for every protocol case, policy, and exact paired seed.
This module does not run simulations at import time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import argparse
import re
import uuid

from .training_curriculum import (CURRICULUM_VERSION, CategoryResult,
    CurriculumState, EvaluationResult, GPU_NUMERIC_GATE_PROTOCOL_VERSION,
    SCENARIOS, _ancestors, evaluate_promotion, seed_partition)
from .meco_stage1_preflight_contract import (MATCHED_REFERENCE_CANDIDATE_K,
    MATCHED_REFERENCE_VARIANCE_VERSION, ROUTE_COST_POINTS_PER_SECOND,
    validate_matched_reference_calibration_metadata, validate_phase0_chain,
    validate_phase0_identity_binding, validate_phase0_index)

PROTOCOL_VERSION = "meco-qualification-v1"
LEGACY_CALIBRATED_PROTOCOL_VERSION = "meco-qualification-v2"
D0_PAIRED_REFERENCE_PROTOCOL_VERSION = "meco-qualification-v3"
CALIBRATED_PROTOCOL_VERSION = "meco-qualification-v4"
GPU_BOOTSTRAP_PROTOCOL_VERSION = GPU_NUMERIC_GATE_PROTOCOL_VERSION
GPU_BOOTSTRAP_IMPLEMENTATION_ID = "meco-v4-cluster-episode-device-percentile-v1"
D0_CALIBRATION_VERSION = "meco-d0-calibration-v2"
D0_CALIBRATION_SHARD_MANIFEST = "meco-d0-calibration-shard-manifest-v1"
D0_CALIBRATION_SHARD_ARTIFACT = "meco-d0-calibration-shard-v1"
MIN_EPISODES = 100
TRAIN_SEED_POOL_SIZE = 100
BOOTSTRAP_REPLICATES = 10_000
POLICIES = ("A1", "A2", "A3", "C1", "C2")
REFERENCE_POLICY = "D0"
MATCHED_REFERENCE_POLICIES = frozenset(("D1", "D2"))
MATCHED_REFERENCE_VARIANTS = {
    "D1/current": ("D1", "D1_hand_weighted", "current"),
    "D2/cv": ("D2", "D2_kinematic_prediction", "cv"),
    "D2/multi_model_v1": ("D2", "D2_kinematic_prediction", "multi_model_v1"),
}


def _reference_variant(value: str) -> str:
    # Keep the earlier API spellings as aliases for protocol construction.
    aliases = {"D1": "D1/current", "D2": "D2/cv"}
    variant = aliases.get(value, value)
    if variant not in MATCHED_REFERENCE_VARIANTS:
        raise ValueError("matched reference must be D1/current, D2/cv, or D2/multi_model_v1")
    return variant


def _matched_reference_identity(policy: str, mode: str | None = None) -> dict:
    if mode is None:
        variant = _reference_variant(policy)
        policy, policy_id, mode = MATCHED_REFERENCE_VARIANTS[variant]
    else:
        policy_id = {"D1": "D1_hand_weighted", "D2": "D2_kinematic_prediction"}.get(policy)
        if policy_id is None or not (
                (policy == "D1" and mode == "current")
                or (policy == "D2" and mode in ("cv", "multi_model_v1"))):
            raise ValueError("matched reference policy/mode is unsupported")
        variant = f"{policy}/{mode}"
    from .meco_prediction_contract import prediction_contract
    return {"selection_id": variant, "policy": policy, "policy_id": policy_id,
        "prediction_mode": mode, "prediction_contract": prediction_contract(mode),
        "checkpoint_sha256": None}


def _control_prediction_modes(protocol: dict) -> dict[str, str]:
    """Keep D1 at current pose while applying the frozen mode to D2."""
    reference = protocol.get("paired_reference_policy")
    mode = protocol.get("paired_reference_mode")
    if reference == "D1":
        if mode != "current":
            raise ValueError("D1 reference is only supported with current-pose prediction")
        return {"D1": "current", "D2": "cv"}
    if reference == "D2" and mode in ("cv", "multi_model_v1"):
        return {"D1": "current", "D2": mode}
    raise ValueError("frozen reference policy/mode is unsupported")


def _reference_control_kwargs(protocol: dict) -> dict:
    if protocol.get("protocol_version") in (CALIBRATED_PROTOCOL_VERSION,
            GPU_BOOTSTRAP_PROTOCOL_VERSION):
        return {"control_prediction_modes": _control_prediction_modes(protocol)}
    if protocol.get("protocol_version") == D0_PAIRED_REFERENCE_PROTOCOL_VERSION:
        return {"control_prediction_modes": _control_prediction_modes(protocol)}
    return {}
SAFETY_METRICS_SCHEMA = "meco-safety-rates-v1"
SAFETY_METRIC_NAMES = frozenset(("safety_robot_contact_rate",
    "safety_field_contact_rate", "safety_wall_contact_rate"))

# Each named layout/noise/family is applied by the simulator constructor or
# per-step scenario controller. New variants require a corresponding runtime
# implementation before they can appear in this frozen case table.
TRAIN_LAYOUTS = ("2026_rebuilt",)
VALIDATION_LAYOUTS = ("2026_rebuilt_shifted_structures_v1",
                      "2026_rebuilt_shifted_structures_v2")
FINAL_TEST_LAYOUTS = ("2026_rebuilt_shifted_structures_v3",)
TRAIN_DEFENDER_FAMILIES = ("category_scripted",)
VALIDATION_DEFENDER_FAMILIES = ("orbiting_interceptor", "split_chaser")
FINAL_TEST_DEFENDER_FAMILIES = ("split_chaser",)
TRAIN_START_CONDITIONS = ("scenario_defined",)
VALIDATION_START_CONDITIONS = ("offset_upper_lane",)
FINAL_TEST_START_CONDITIONS = ("offset_lower_lane",)
TRAIN_SENSOR_PROFILES = ("standard",)
VALIDATION_SENSOR_PROFILES = ("noisy",)
FINAL_TEST_SENSOR_PROFILES = ("noisy",)


@dataclass(frozen=True, order=True)
class QualificationCase:
    scenario_id: str
    category: str
    split: str
    layout_id: str
    defender_family: str
    start_condition: str
    sensor_profile: str
    seeds: tuple[int, ...]


def _split_axes(split):
    if split == "train":
        return TRAIN_LAYOUTS, TRAIN_DEFENDER_FAMILIES, TRAIN_START_CONDITIONS, TRAIN_SENSOR_PROFILES
    if split == "validation":
        return VALIDATION_LAYOUTS, VALIDATION_DEFENDER_FAMILIES, VALIDATION_START_CONDITIONS, VALIDATION_SENSOR_PROFILES
    if split == "test":
        return FINAL_TEST_LAYOUTS, FINAL_TEST_DEFENDER_FAMILIES, FINAL_TEST_START_CONDITIONS, FINAL_TEST_SENSOR_PROFILES
    if split == "calibration":
        return VALIDATION_LAYOUTS, VALIDATION_DEFENDER_FAMILIES, VALIDATION_START_CONDITIONS, VALIDATION_SENSOR_PROFILES
    raise ValueError("split must be train, validation, test, or calibration")


def protocol_cases(scenario_id: str, split: str, *, episodes: int = MIN_EPISODES,
                   master_seed: int = 20261024) -> tuple[QualificationCase, ...]:
    """Return frozen, exact seed lists for every scenario category and stratum."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario_id}")
    if type(episodes) is not int or episodes < MIN_EPISODES:
        raise ValueError(f"qualification requires at least {MIN_EPISODES} episodes per category")
    layouts, families, starts, sensors = _split_axes(split)
    if scenario_id not in ("offense_vs_scripted_pressure", "full_match_offense"):
        families = ("none",)
    cases = []
    for category in SCENARIOS[scenario_id].required_categories:
        for layout in layouts:
            for family in families:
                for start in starts:
                    for sensor in sensors:
                        # Namespace independent case seed pools by scenario,
                        # split, and stratum while preserving split bits.
                        namespace = int.from_bytes(hashlib.sha256(
                            f"{layout}/{family}/{start}/{sensor}".encode()).digest()[:4], "big")
                        seeds = seed_partition(scenario_id, split, episodes,
                            master_seed=(master_seed << 32) | namespace)
                        cases.append(QualificationCase(scenario_id, category, split,
                            layout, family, start, sensor, seeds))
    return tuple(cases)


def _scenario_closure(target: str) -> tuple[str, ...]:
    closure = {target, *_ancestors(target)}
    return tuple(scenario for scenario in SCENARIOS if scenario in closure)


def _retention_scenarios(target: str) -> tuple[str, ...]:
    return tuple(scenario for scenario in _scenario_closure(target) if scenario != target)


def _validate_calibration_runtime_device(runtime_device: object) -> None:
    """Validate the typed CUDA/HIP identity emitted by v2 D0 calibration."""
    if not isinstance(runtime_device, dict):
        raise ValueError("D0 calibration runtime device identity is invalid")
    device = runtime_device.get("device")
    if not isinstance(device, str) or re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", device) is None:
        raise ValueError("D0 calibration runtime device must be an indexed CUDA device")
    backend = runtime_device.get("backend")
    torch_version = runtime_device.get("torch_version")
    hip_version = runtime_device.get("hip_version")
    device_name = runtime_device.get("device_name")
    if (backend not in ("hip", "cuda")
            or not isinstance(torch_version, str) or not torch_version.strip()
            or torch_version != torch_version.strip()
            or not isinstance(device_name, str) or not device_name.strip()):
        raise ValueError("D0 calibration runtime device/version identity is invalid")
    if backend == "hip":
        if (not isinstance(hip_version, str) or not hip_version.strip()
                or hip_version != hip_version.strip()):
            raise ValueError("HIP calibration runtime requires a typed HIP version")
    elif hip_version is not None:
        raise ValueError("CUDA calibration runtime must not report a HIP version")


def _validate_calibration_d0_manifest(policy_manifest: object, source_identity: str) -> None:
    """Bind D0's production manifest entry to the report's exact source."""
    if not isinstance(policy_manifest, dict):
        raise ValueError("D0 calibration policy manifest is invalid")
    d0 = policy_manifest.get(REFERENCE_POLICY)
    if (not isinstance(d0, dict)
            or d0.get("policy_id") != "D0_production_deterministic"
            or d0.get("source_identity_sha256") != source_identity
            or d0.get("checkpoint_sha256") is not None):
        raise ValueError("D0 calibration manifest source/checkpoint provenance is invalid")


def _calibration_report_index(reports: dict, scenarios: tuple[str, ...]) -> tuple[dict, dict, str, dict]:
    """Validate every D0 report and return its IDs, exact seed pools, and shared source."""
    if set(reports) != set(scenarios):
        raise ValueError("D0 calibration bundle must cover the target and every prerequisite exactly")
    ids, pools, report_hashes, source_hash = {}, {}, {}, None
    calibration_seed_union = set()
    for scenario in scenarios:
        report = reports[scenario]
        if isinstance(report, dict) and report.get("artifact_type") in (
                D0_CALIBRATION_SHARD_MANIFEST, D0_CALIBRATION_SHARD_ARTIFACT):
            raise ValueError("D0 calibration shard artifacts must be merged before protocol intake")
        if (not isinstance(report, dict) or report.get("schema_version") != 2
                or report.get("calibration") is not True
                or report.get("scenario_id") != scenario
                or report.get("protocol_version") != D0_CALIBRATION_VERSION):
            raise ValueError(f"D0 calibration bundle report is invalid for {scenario}")
        unsigned = {key: value for key, value in report.items() if key != "calibration_id"}
        digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if report.get("calibration_id") != digest:
            raise ValueError(f"D0 calibration report digest is invalid for {scenario}")
        episodes, repetitions, master_seed = (report.get("episodes_per_category_stratum"),
            report.get("repetitions"), report.get("master_seed"))
        if (type(episodes) is not int or episodes < MIN_EPISODES
                or type(repetitions) is not int or repetitions < 2
                or type(master_seed) is not int or master_seed < 0):
            raise ValueError(f"D0 calibration report settings are invalid for {scenario}")
        cases = protocol_cases(scenario, "calibration", episodes=episodes,
                               master_seed=master_seed)
        expected_case_ids = [{"category": c.category, "layout_id": c.layout_id,
            "defender_family": c.defender_family, "start_condition": c.start_condition,
            "sensor_profile": c.sensor_profile} for c in cases]
        if report.get("case_ids") != expected_case_ids:
            raise ValueError(f"D0 calibration case coverage is invalid for {scenario}")
        expected_rows = {(rep, c.category, c.layout_id, c.defender_family,
            c.start_condition, c.sensor_profile, seed)
            for rep in range(repetitions) for c in cases for seed in c.seeds}
        rows = report.get("records")
        if not isinstance(rows, list) or len(rows) != len(expected_rows):
            raise ValueError(f"D0 calibration raw coverage is incomplete for {scenario}")
        found = set()
        throughput_by_stratum_seed = {}
        for row in rows:
            key = (row.get("repetition"), row.get("category"), row.get("layout_id"),
                row.get("defender_family"), row.get("start_condition"),
                row.get("sensor_profile"), row.get("seed"))
            metrics = row.get("metrics")
            metric = metrics.get("useful_throughput") if isinstance(metrics, dict) else None
            if (key not in expected_rows or key in found or row.get("policy") != REFERENCE_POLICY
                    or row.get("split") != "calibration"
                    or row.get("policy_id") != "D0_production_deterministic"
                    or row.get("checkpoint_sha256") is not None
                    or type(row.get("repetition")) is not int
                    or type(metric) not in (int, float) or not math.isfinite(metric) or metric < 0):
                raise ValueError(f"D0 calibration raw row is invalid for {scenario}")
            found.add(key)
            stratum = key[1:6]
            throughput_by_stratum_seed.setdefault(stratum, {}).setdefault(key[6], []).append(float(metric))
        if found != expected_rows:
            raise ValueError(f"D0 calibration seed/repetition coverage is incomplete for {scenario}")
        current_source = report.get("source_identity_sha256")
        if (not isinstance(current_source, str) or len(current_source) != 64
                or any(c not in "0123456789abcdef" for c in current_source)
                or report.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
                or report.get("safety_metrics_source_identity_sha256") != current_source
                or any(row.get("source_identity_sha256") != current_source for row in rows)):
            raise ValueError(f"D0 calibration source/controller provenance is invalid for {scenario}")
        _validate_calibration_d0_manifest(report.get("policy_manifest"), current_source)
        runtime_device = report.get("runtime_device")
        try:
            _validate_calibration_runtime_device(runtime_device)
        except ValueError as exc:
            raise ValueError(f"D0 calibration GPU provenance is invalid for {scenario}") from exc
        normalized_runtime = report.get("normalized_runtime_identity")
        expected_normalized_runtime = _normalized_calibration_runtime(runtime_device)
        if normalized_runtime != expected_normalized_runtime:
            raise ValueError(f"D0 calibration normalized runtime/row provenance is invalid for {scenario}")
        try:
            row_runtime_matches = all(
                _normalized_calibration_runtime(row.get("calibration_runtime_device"))
                == expected_normalized_runtime for row in rows)
        except ValueError as exc:
            raise ValueError(f"D0 calibration row runtime provenance is invalid for {scenario}") from exc
        if not row_runtime_matches:
            raise ValueError(f"D0 calibration row runtime provenance is invalid for {scenario}")
        for row in rows:
            metrics = row.get("metrics")
            safety_keys = ({key for key in metrics
                            if isinstance(key, str) and key.startswith("safety_")}
                           if isinstance(metrics, dict) else set())
            if (not isinstance(metrics, dict)
                    or safety_keys != SAFETY_METRIC_NAMES
                    or any(type(metrics.get(name)) not in (int, float)
                        or not math.isfinite(metrics[name]) or not 0 <= metrics[name] <= 1
                        for name in SAFETY_METRIC_NAMES)):
                raise ValueError(f"D0 calibration safety metric coverage is invalid for {scenario}")
        # The uncertainty summaries drive frozen promotion margins. Recompute
        # them from the exact raw rows and reject missing, extra, or edited
        # summary strata before any margin derivation can consume them.
        summaries = report.get("summary")
        expected_summary = {}
        for case in cases:
            stratum = (case.category, case.layout_id, case.defender_family,
                case.start_condition, case.sensor_profile)
            by_seed = throughput_by_stratum_seed.get(stratum, {})
            if set(by_seed) != set(case.seeds) or any(
                    len(values) != repetitions for values in by_seed.values()):
                raise ValueError(f"D0 calibration summary source rows are incomplete for {scenario}")
            seed_means = [sum(by_seed[seed]) / repetitions for seed in case.seeds]
            repeat_variances = [_variance(by_seed[seed]) for seed in case.seeds]
            expected_summary["/".join(stratum)] = {
                "category": case.category, "layout_id": case.layout_id,
                "defender_family": case.defender_family,
                "start_condition": case.start_condition,
                "sensor_profile": case.sensor_profile,
                "seed_count": len(case.seeds), "repetition_count": repetitions,
                "mean_useful_throughput": sum(seed_means) / len(seed_means),
                "between_seed_variance": _variance(seed_means),
                "mean_within_seed_repeat_variance": sum(repeat_variances) / len(repeat_variances),
                "seed_digest": hashlib.sha256(json.dumps(list(case.seeds),
                    separators=(",", ":")).encode()).hexdigest()}
        if not isinstance(summaries, dict) or summaries != expected_summary:
            raise ValueError(f"D0 calibration summaries do not exactly match raw records for {scenario}")
        if source_hash is None:
            source_hash = current_source
        elif source_hash != current_source:
            raise ValueError("D0 calibration bundle mixes source identities")
        seeds = {seed for case in cases for seed in case.seeds}
        if calibration_seed_union & seeds:
            raise ValueError("D0 calibration seed pools overlap across prerequisite scenarios")
        calibration_seed_union.update(seeds)
        ids[scenario] = digest
        report_hashes[scenario] = hashlib.sha256(json.dumps(report, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        pools[scenario] = {"master_seed": master_seed, "episodes": episodes,
            "repetitions": repetitions, "cases": [
                {"category": c.category, "layout_id": c.layout_id,
                 "defender_family": c.defender_family,
                 "start_condition": c.start_condition,
                 "sensor_profile": c.sensor_profile, "seeds": list(c.seeds)}
                for c in cases]}
    return ids, pools, source_hash, report_hashes


def _reference_variance_report_index(reports, scenarios, *, source_hash,
                                     reference_variant, d0_reports):
    if isinstance(reports, dict) and "calibration_reports" in reports:
        reports = reports["calibration_reports"]
    if not isinstance(reports, dict) or set(reports) != set(scenarios):
        raise ValueError("matched reference variance bundle must cover the exact prerequisite closure")
    indexed, hashes, ids, seed_union, runtime_identity = {}, {}, {}, set(), None
    expected_variant = _reference_variant(reference_variant)
    for scenario in scenarios:
        report = reports[scenario]
        _validate_matched_reference_variance_report(report)
        if (not isinstance(report, dict)
                or report.get("artifact_type") != MATCHED_REFERENCE_VARIANCE_VERSION
                or report.get("scenario_id") != scenario
                or report.get("config", {}).get("reference_selection_id") != expected_variant
                or report.get("source_identity_sha256") != source_hash
                or report.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
                or report.get("safety_metrics_source_identity_sha256") != source_hash):
            raise ValueError(f"matched reference variance report is incompatible for {scenario}")
        unsigned = {key: value for key, value in report.items() if key != "calibration_id"}
        calibration_id = _canonical_sha256(unsigned)
        if report.get("calibration_id") != calibration_id:
            raise ValueError(f"matched reference variance digest is invalid for {scenario}")
        config = report.get("config", {})
        if (config.get("prediction_contract") != _matched_reference_identity(expected_variant)["prediction_contract"]
                or report.get("policy_manifest", {}).get(config.get("reference_policy"), {}).get("policy_id")
                    != MATCHED_REFERENCE_VARIANTS[expected_variant][1]
                or report.get("policy_manifest", {}).get(config.get("reference_policy"), {}).get("prediction_mode")
                    != MATCHED_REFERENCE_VARIANTS[expected_variant][2]):
            raise ValueError(f"matched reference policy/mode/contract differs for {scenario}")
        if report.get("policy_manifest", {}).get(REFERENCE_POLICY, {}).get("policy_id") != "D0_production_deterministic":
            raise ValueError(f"matched reference report lacks the D0 safety-floor policy for {scenario}")
        d0_report = d0_reports[scenario]
        d0_cases = d0_report.get("case_ids")
        reference_cases = [{key: value for key, value in row.items() if key != "reference_policy"}
                           for row in report.get("case_ids", [])]
        d0_episodes = d0_report.get("episodes_per_category_stratum")
        d0_master_seed = d0_report.get("master_seed")
        d0_protocol_cases = protocol_cases(scenario, "calibration",
            episodes=d0_episodes, master_seed=d0_master_seed)
        expected_d0_cases = [{"category": case.category, "layout_id": case.layout_id,
            "defender_family": case.defender_family,
            "start_condition": case.start_condition,
            "sensor_profile": case.sensor_profile} for case in d0_protocol_cases]
        expected_reference_cases = [{**row, "seeds": list(case.seeds)}
            for row, case in zip(expected_d0_cases, d0_protocol_cases)]
        if (d0_cases != expected_d0_cases or reference_cases != expected_reference_cases
                or report.get("config", {}).get("episodes_per_category_stratum") != d0_episodes
                or report.get("config", {}).get("master_seed") != d0_master_seed):
            raise ValueError(f"matched reference and D0 calibration seed tables differ for {scenario}")
        episodes = config.get("episodes_per_category_stratum")
        master_seed = config.get("master_seed")
        _validate_d0_calibration_case_table(scenario, episodes, master_seed, d0_cases)
        seeds = {seed for case in reference_cases for seed in case["seeds"]}
        if seeds & seed_union:
            raise ValueError("matched reference calibration seed pools overlap across prerequisite scenarios")
        seed_union.update(seeds)
        runtime = report.get("normalized_runtime_identity")
        if runtime != _normalized_calibration_runtime(report.get("runtime_device")):
            raise ValueError(f"matched reference runtime identity is invalid for {scenario}")
        if runtime_identity is not None and runtime_identity != runtime:
            raise ValueError("matched reference reports have runtime identity drift")
        d0_runtime = d0_report.get("normalized_runtime_identity")
        if d0_runtime is not None and d0_runtime != runtime:
            raise ValueError("matched reference and D0 calibration runtime identities differ")
        runtime_identity = runtime
        indexed[scenario] = report
        ids[scenario] = calibration_id
        hashes[scenario] = _canonical_sha256(report)
    return indexed, ids, hashes, runtime_identity


def _validate_matched_reference_variance_report(report):
    """Revalidate raw repeated rows and summaries before protocol binding."""
    metadata = validate_matched_reference_calibration_metadata(report,
        variant_table=MATCHED_REFERENCE_VARIANTS,
        variant_normalizer=_reference_variant,
        identity_builder=_matched_reference_identity,
        protocol_cases_builder=protocol_cases,
        d0_case_validator=_validate_d0_calibration_case_table,
        runtime_normalizer=_normalized_calibration_runtime,
        minimum_episodes=MIN_EPISODES,
        candidate_k=MATCHED_REFERENCE_CANDIDATE_K,
        route_cost=ROUTE_COST_POINTS_PER_SECOND,
        safety_metric_names=SAFETY_METRIC_NAMES,
        safety_metrics_schema=SAFETY_METRICS_SCHEMA,
        safety_floor_policy=REFERENCE_POLICY)
    config, scenario = metadata["config"], metadata["scenario"]
    variant, policy, policy_id, mode = (metadata[key] for key in
        ("variant", "policy", "policy_id", "mode"))
    identity = metadata["identity"]
    candidate_configuration = metadata["candidate_configuration"]
    cases, expected_cases = metadata["cases"], metadata["expected_cases"]
    source_hash, runtime = metadata["source_hash"], metadata["runtime"]
    manifest = report.get("policy_manifest")
    if (not isinstance(manifest, dict)
            or manifest.get(REFERENCE_POLICY, {}).get("policy_id") != "D0_production_deterministic"
            or manifest.get(policy, {}).get("policy_id") != policy_id
            or manifest.get(policy, {}).get("checkpoint_sha256") is not None
            or manifest.get(policy, {}).get("source_identity_sha256") != report.get("source_identity_sha256")
            or manifest.get(policy, {}).get("prediction_mode") != mode
            or manifest.get(policy, {}).get("prediction_contract") != identity["prediction_contract"]
            or manifest.get(policy, {}).get("calibration_candidate_configuration")
                != candidate_configuration):
        raise ValueError("matched reference variance policy manifest is invalid")
    expected = {(rep, name, case.category, case.layout_id, case.defender_family,
        case.start_condition, case.sensor_profile, seed)
        for rep in range(config["repetitions"])
        for name in (REFERENCE_POLICY, policy) for case in cases for seed in case.seeds}
    case_bindings = {}
    for case in expected_cases:
        key_fields = (case["category"], case["layout_id"], case["defender_family"],
            case["start_condition"], case["sensor_profile"])
        case_bindings[key_fields] = {"scenario_id": scenario,
            "master_seed": config["master_seed"],
            "episodes": config["episodes_per_category_stratum"],
            "category": case["category"],
            "qualification_case": {field: case[field] for field in
                ("layout_id", "defender_family", "start_condition", "sensor_profile")},
            "episode_seeds": case["seeds"]}
    found = {}
    for row in report.get("records", []):
        if not isinstance(row, dict):
            raise ValueError("matched reference variance record is malformed")
        if type(row.get("repetition")) is not int or type(row.get("seed")) is not int:
            raise ValueError("matched reference repetition and seed must be exact integers")
        key = (row.get("repetition"), row.get("policy"), row.get("category"),
            row.get("layout_id"), row.get("defender_family"),
            row.get("start_condition"), row.get("sensor_profile"), row.get("seed"))
        if key not in expected or key in found:
            raise ValueError("matched reference variance contains duplicate or unexpected rows")
        if row.get("calibration_control_case_binding") != case_bindings.get(key[2:7]):
            raise ValueError("matched reference row is not bound to its frozen evaluator case and seed tuple")
        expected_id = "D0_production_deterministic" if key[1] == REFERENCE_POLICY else policy_id
        if (row.get("policy_id") != expected_id
                or row.get("split") != "calibration"
                or row.get("scenario_id") != scenario
                or row.get("source_identity_sha256") != source_hash
                or row.get("calibration_runtime_device") != runtime
                or (key[1] == policy and (row.get("prediction_mode") != mode
                    or row.get("prediction_contract") != identity["prediction_contract"]))):
            raise ValueError("matched reference variance row provenance differs from its frozen report")
        metrics = row.get("metrics")
        if (not isinstance(metrics, dict)
                or {key for key in metrics if isinstance(key, str) and key.startswith("safety_")}
                    != SAFETY_METRIC_NAMES
                or any(type(metrics.get(metric)) not in (int, float)
                    or not math.isfinite(metrics[metric]) or not 0 <= metrics[metric] <= 1
                    for metric in SAFETY_METRIC_NAMES)
                or type(metrics.get("useful_throughput")) not in (int, float)
                or not math.isfinite(metrics["useful_throughput"])
                or metrics["useful_throughput"] < 0):
            raise ValueError("matched reference variance row metrics are incomplete")
        found[key] = row
    if set(found) != expected:
        raise ValueError("matched reference variance report has incomplete repeated seed coverage")
    expected_summary = _matched_reference_variance_summary(expected_cases,
        config["repetitions"], list(found.values()))
    if report.get("summary") != expected_summary:
        raise ValueError("matched reference variance summary does not match raw repeated outcomes")
    unsigned = {key: value for key, value in report.items() if key != "calibration_id"}
    if report.get("calibration_id") != _canonical_sha256(unsigned):
        raise ValueError("matched reference variance report digest is invalid")
    return report


def _validate_reference_margin_review(review, *, target, reference_variant,
                                     reference_ids, reference_hashes,
                                     safety_margins):
    categories = set(SCENARIOS[target].required_categories)
    required_keys = {"schema_version", "reviewed_by", "reviewed_utc", "decision",
        "reference_variant", "reference_calibration_ids", "reference_report_sha256",
        "minimum_useful_gain", "safety_noninferiority_margins"}
    if (not isinstance(review, dict) or review.get("schema_version") != 1
            or set(review) != required_keys
            or review.get("reference_variant") != reference_variant
            or review.get("reference_calibration_ids") != reference_ids
            or review.get("reference_report_sha256") != reference_hashes
            or not isinstance(review.get("reviewed_by"), str)
            or not review["reviewed_by"].strip()
            or not isinstance(review.get("reviewed_utc"), str)
            or not review["reviewed_utc"].strip()
            or review.get("decision") != "accepted"):
        raise ValueError("reference variance margin review must bind the frozen reports and record an explicit approval")
    gains = review.get("minimum_useful_gain")
    if (not isinstance(gains, dict) or set(gains) != categories
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in gains.values())):
        raise ValueError("review must provide an explicit nonnegative useful-gain margin per target category")
    reviewed_safety = review.get("safety_noninferiority_margins")
    if (not isinstance(reviewed_safety, dict) or reviewed_safety != safety_margins
            or set(reviewed_safety) != SAFETY_METRIC_NAMES):
        raise ValueError("reviewed safety margins must exactly match the separately declared safety margin file")
    return {"schema_version": 1, "reviewed_by": review["reviewed_by"],
        "reviewed_utc": review["reviewed_utc"], "decision": "accepted",
        "reference_variant": reference_variant,
        "reference_calibration_ids": reference_ids,
        "reference_report_sha256": reference_hashes,
        "minimum_useful_gain": dict(gains),
        "safety_noninferiority_margins": dict(reviewed_safety),
        "review_sha256": _canonical_sha256(review)}


def protocol_document(scenario_id: str | None = None, *, episodes: int = MIN_EPISODES,
                      master_seed: int = 20261024, calibration_report: dict | None = None,
                      minimum_useful_gain: dict | None = None,
                      safety_noninferiority_margins: dict[str, float] | None = None,
                      reference_policy: str = "D1/current",
                      matched_reference_calibration_report: dict | None = None,
                      reference_margin_review: dict | None = None,
                      phase0_identity_binding: dict | None = None) -> dict:
    """Canonical target-scoped protocol; the digest is the protocol identity."""
    if scenario_id is not None and scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario_id}")
    calibrated = calibration_report is not None
    if calibrated:
        reference_policy = _reference_variant(reference_policy)
        if minimum_useful_gain is not None:
            raise ValueError("v4 useful-gain margins come only from the explicit reference margin review")
    if calibrated and scenario_id is None:
        raise ValueError("calibrated protocol requires an explicit promotion target")
    target = scenario_id
    scenarios = (_scenario_closure(target) if calibrated else
        tuple(SCENARIOS) if target is None else (target,))
    cases = [asdict(case) | {"seeds": list(case.seeds)} for scenario in scenarios
             for split in ("train", "validation", "test")
             for case in protocol_cases(scenario, split,
                 episodes=TRAIN_SEED_POOL_SIZE if split == "train" else episodes,
                 master_seed=master_seed)]
    if not calibrated:
        # Keep the legacy v1 document byte-for-byte schema compatible.
        body = {"protocol_version": PROTOCOL_VERSION,
            "curriculum_version": CURRICULUM_VERSION,
            "scenario_ids": list(scenarios),
            "scenario_versions": {scenario: SCENARIOS[scenario].version for scenario in scenarios},
            "episodes_per_category_stratum": episodes,
            "master_seed": master_seed,
            "training": {"layouts": list(TRAIN_LAYOUTS),
                "defender_families": ["none", *TRAIN_DEFENDER_FAMILIES],
                "seed_pool_size": TRAIN_SEED_POOL_SIZE,
                "runtime_seed_binding": "MecoTaskScheduler selects these exact train-split seeds by scenario cycle modulo pool size; scheduler master_seed must match this protocol",
                "runtime_seed_lists_enforced": True},
            "validation": {"layouts": list(VALIDATION_LAYOUTS),
                "defender_families": ["none", *VALIDATION_DEFENDER_FAMILIES]},
            "final_test": {"layouts": list(FINAL_TEST_LAYOUTS),
                "defender_families": ["none", *FINAL_TEST_DEFENDER_FAMILIES],
                "isolated_until_selected_frozen_bundle": True},
            "start_conditions": {"train": list(TRAIN_START_CONDITIONS),
                "validation": list(VALIDATION_START_CONDITIONS),
                "test": list(FINAL_TEST_START_CONDITIONS)},
            "sensor_profiles": {"train": list(TRAIN_SENSOR_PROFILES),
                "validation": list(VALIDATION_SENSOR_PROFILES),
                "test": list(FINAL_TEST_SENSOR_PROFILES)},
            "paired_reference_policy": REFERENCE_POLICY,
            "gate_calibration_status": "provisional_pending_baseline_variance_review",
            "promotion_gates": {scenario: asdict(SCENARIOS[scenario].gate)
                for scenario in scenarios},
            "cases": cases}
        body["protocol_id"] = hashlib.sha256(json.dumps(body, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return body
    if calibrated:
        if target == "pickup_simple" and phase0_identity_binding is None:
            raise ValueError("pickup_simple Stage 1 v5 protocol requires a Phase 0 identity binding")
        reports = ({calibration_report["scenario_id"]: calibration_report}
                   if isinstance(calibration_report, dict) and "scenario_id" in calibration_report
                   else calibration_report)
        if not isinstance(reports, dict):
            raise ValueError("calibration input must be a report or scenario-to-report bundle")
        calibration_ids, calibration_seed_pools, source_hash, calibration_report_hashes = _calibration_report_index(
            reports, scenarios)
        if (not isinstance(safety_noninferiority_margins, dict)
                or set(safety_noninferiority_margins) != SAFETY_METRIC_NAMES
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or not 0 <= value <= 1 for value in safety_noninferiority_margins.values())):
            raise ValueError("provide the exact supported safety metric names and margins")
        reference_reports, reference_ids, reference_hashes, reference_runtime = \
            _reference_variance_report_index(matched_reference_calibration_report,
                scenarios, source_hash=source_hash, reference_variant=reference_policy,
                d0_reports=reports)
        reviewed = _validate_reference_margin_review(reference_margin_review,
            target=target, reference_variant=reference_policy,
            reference_ids=reference_ids, reference_hashes=reference_hashes,
            safety_margins=safety_noninferiority_margins)
        minimum_useful_gain = reviewed["minimum_useful_gain"]
        useful_gain_derivation = {"schema_version": 1,
            "method": "explicit_reviewed_reference_variance_margins",
            "minimum_useful_gain": minimum_useful_gain,
            "review_sha256": reviewed["review_sha256"],
            "reviewed_by": reviewed["reviewed_by"],
            "reviewed_utc": reviewed["reviewed_utc"]}
        baseline_id = hashlib.sha256(json.dumps(calibration_ids, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        reference_variance_calibration = {"version": MATCHED_REFERENCE_VARIANCE_VERSION,
            "calibration_ids": reference_ids, "report_sha256": reference_hashes,
            "normalized_runtime_identity": reference_runtime,
            "seed_table_sha256": {name: _canonical_sha256(reference_reports[name]["case_ids"])
                                  for name in scenarios},
            "margin_review": reviewed}
        if phase0_identity_binding is not None:
            validate_phase0_identity_binding(phase0_identity_binding)
            if phase0_identity_binding["source_identity_sha256"] != source_hash:
                raise ValueError("Phase 0 evaluator source identity does not match calibration reports")
    else:
        calibration_ids, calibration_seed_pools, calibration_report_hashes, source_hash, baseline_id = {}, {}, {}, None, None
        reference_reports, reference_ids, reference_hashes = {}, {}, {}
        reviewed, reference_variance_calibration = None, None
    body = {"protocol_version": GPU_BOOTSTRAP_PROTOCOL_VERSION if calibrated else PROTOCOL_VERSION,
        "promotion_target_scenario": target,
        "scenario_ids": list(scenarios),
        "retention_scenario_ids": list(_retention_scenarios(target)) if calibrated else [],
        "scenario_versions": {scenario: SCENARIOS[scenario].version for scenario in scenarios},
        "episodes_per_category_stratum": episodes,
        "master_seed": master_seed,
        "training": {"layouts": list(TRAIN_LAYOUTS),
            "defender_families": ["none", *TRAIN_DEFENDER_FAMILIES],
            "seed_pool_size": TRAIN_SEED_POOL_SIZE,
            "runtime_seed_binding": "MecoTaskScheduler selects these exact train-split seeds by scenario cycle modulo pool size; scheduler master_seed must match this protocol",
            "runtime_seed_lists_enforced": True},
        "validation": {"layouts": list(VALIDATION_LAYOUTS),
            "defender_families": ["none", *VALIDATION_DEFENDER_FAMILIES]},
        "final_test": {"layouts": list(FINAL_TEST_LAYOUTS),
            "defender_families": ["none", *FINAL_TEST_DEFENDER_FAMILIES],
            "isolated_until_selected_frozen_bundle": True},
        "start_conditions": {"train": list(TRAIN_START_CONDITIONS),
            "validation": list(VALIDATION_START_CONDITIONS),
            "test": list(FINAL_TEST_START_CONDITIONS)},
        "sensor_profiles": {"train": list(TRAIN_SENSOR_PROFILES),
            "validation": list(VALIDATION_SENSOR_PROFILES),
            "test": list(FINAL_TEST_SENSOR_PROFILES)},
        "paired_reference_policy": (MATCHED_REFERENCE_VARIANTS[reference_policy][0]
            if calibrated else REFERENCE_POLICY),
        "paired_reference_mode": (MATCHED_REFERENCE_VARIANTS[reference_policy][2]
            if calibrated else None),
        "paired_reference_variant": reference_policy if calibrated else None,
        "paired_reference_identity": (_matched_reference_identity(reference_policy)
            if calibrated else None),
        "reference_selection_basis": ("predeclared_matched_baseline_before_heldout_evaluation"
            if calibrated else None),
        "safety_floor_policy": REFERENCE_POLICY if calibrated else None,
        "gate_calibration_status": ("matched_reference_variance_and_reviewed_margin_frozen" if calibrated
            else "provisional_pending_baseline_variance_review"),
        "calibration_id": baseline_id, "calibration_ids": calibration_ids,
        "calibration_seed_pools": calibration_seed_pools,
        "calibration_report_sha256": calibration_report_hashes if calibrated else {},
        "calibration_source_identity_sha256": source_hash,
        "minimum_useful_gain": minimum_useful_gain if calibrated else None,
        "useful_gain_derivation": useful_gain_derivation if calibrated else None,
        "useful_gain_calibration_basis": ({"policy": REFERENCE_POLICY,
            "calibration_id": baseline_id,
            "calibration_report_sha256": calibration_report_hashes,
            "role": "separate_safety_floor_and_seeded_runtime_control"}
            if calibrated else None),
        "matched_reference_variance_calibration": reference_variance_calibration,
        "safety_noninferiority_margins": safety_noninferiority_margins if calibrated else None,
        "safety_gate_status": "metrics_schema_verified" if calibrated else "not_calibrated",
        "promotion_gates": {scenario: asdict(SCENARIOS[scenario].gate) for scenario in scenarios},
        "cases": cases}
    if calibrated:
        if phase0_identity_binding is not None:
            body["phase0_identity_binding"] = dict(phase0_identity_binding)
        body["safety_metrics_schema"] = SAFETY_METRICS_SCHEMA
        body["safety_metrics_source_identity_sha256"] = source_hash
        body["gpu_statistics_implementation"] = _gpu_statistics_implementation(
            reference_variance_calibration["normalized_runtime_identity"])
    body["protocol_id"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return body


def _gpu_statistics_implementation(runtime_identity):
    return {"implementation_id": GPU_BOOTSTRAP_IMPLEMENTATION_ID,
        "generator": "torch.Generator(device); torch.randint and float64 torch.rand",
        "seed_derivation": "validation: sha256(evaluation_id/category)[:16], safety XOR 0x5AFE and 0xD00D; final test: sha256(evaluation_id/category/metric/ref|D0)[:16]",
        "cluster_order": "ascending (layout_id, defender_family)",
        "resampling": "same number of clusters; episodes sampled within each selected cluster using that cluster's original count; pooled episode mean with variable replicate denominator",
        "replicate_count": BOOTSTRAP_REPLICATES,
        "quantile_indices": [int(.025 * (BOOTSTRAP_REPLICATES - 1)),
                             int(.975 * (BOOTSTRAP_REPLICATES - 1))],
        "runtime_identity": runtime_identity}


def _validate_v2_protocol(protocol: dict, scenario: str) -> None:
    """Validate a target-scoped calibrated protocol for target or retention use."""
    if not isinstance(protocol, dict) or protocol.get("protocol_version") not in (
            GPU_BOOTSTRAP_PROTOCOL_VERSION, CALIBRATED_PROTOCOL_VERSION, D0_PAIRED_REFERENCE_PROTOCOL_VERSION,
            LEGACY_CALIBRATED_PROTOCOL_VERSION):
        raise ValueError("promotion requires a calibrated v2 protocol")
    body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if protocol.get("protocol_id") != digest:
        raise ValueError("qualification protocol file hash is invalid")
    target = protocol.get("promotion_target_scenario")
    if target not in SCENARIOS or scenario not in _scenario_closure(target):
        raise ValueError("scenario is outside the calibrated target/prerequisite closure")
    scenarios = _scenario_closure(target)
    if protocol.get("scenario_ids") != list(scenarios):
        raise ValueError("calibrated protocol prerequisite closure is invalid")
    if protocol.get("retention_scenario_ids") != list(_retention_scenarios(target)):
        raise ValueError("calibrated protocol retention scenario list is invalid")
    if (protocol.get("scenario_versions") != {name: SCENARIOS[name].version for name in scenarios}
            or protocol.get("promotion_gates") != {name: asdict(SCENARIOS[name].gate) for name in scenarios}):
        raise ValueError("calibrated protocol scenario versions/gates are invalid")
    legacy_v2 = protocol.get("protocol_version") == LEGACY_CALIBRATED_PROTOCOL_VERSION
    legacy_v3 = protocol.get("protocol_version") == D0_PAIRED_REFERENCE_PROTOCOL_VERSION
    expected_gate_status = ("calibrated_baseline_frozen" if legacy_v2 else
        "d0_operational_margin_frozen_reference_variance_unverified" if legacy_v3 else
        "matched_reference_variance_and_reviewed_margin_frozen")
    if (protocol.get("gate_calibration_status") != expected_gate_status
            or protocol.get("safety_gate_status") != "metrics_schema_verified"):
        raise ValueError("calibrated protocol status fields are invalid or unsupported")
    reference = protocol.get("paired_reference_policy")
    if legacy_v2:
        if reference != REFERENCE_POLICY:
            raise ValueError("legacy calibrated v2 protocol must retain its D0 reference")
    else:
        variant = protocol.get("paired_reference_variant")
        if variant not in MATCHED_REFERENCE_VARIANTS:
            raise ValueError("calibrated protocol must freeze D1/current, D2/cv, or D2/multi_model_v1")
        expected_policy, _, expected_mode = MATCHED_REFERENCE_VARIANTS[variant]
        expected_basis = ({"policy": REFERENCE_POLICY,
            "calibration_id": protocol.get("calibration_id"),
            "calibration_report_sha256": protocol.get("calibration_report_sha256"),
            "reference_variance_calibration_status": "not_available"} if legacy_v3 else {
            "policy": REFERENCE_POLICY,
            "calibration_id": protocol.get("calibration_id"),
            "calibration_report_sha256": protocol.get("calibration_report_sha256"),
            "role": "separate_safety_floor_and_seeded_runtime_control"})
        if (reference != expected_policy or protocol.get("paired_reference_mode") != expected_mode
                or protocol.get("paired_reference_identity") != _matched_reference_identity(variant)
                or protocol.get("reference_selection_basis")
                    != "predeclared_matched_baseline_before_heldout_evaluation"
                or protocol.get("safety_floor_policy") != REFERENCE_POLICY
                or protocol.get("useful_gain_calibration_basis") != expected_basis):
            raise ValueError("calibrated protocol reference identity, D0 calibration basis, or safety floor is invalid")
    for field in ("calibration_id", "calibration_source_identity_sha256"):
        value = protocol.get(field)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"calibrated protocol has invalid {field}")
    if (protocol.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
            or protocol.get("safety_metrics_source_identity_sha256")
                != protocol.get("calibration_source_identity_sha256")):
        raise ValueError("calibrated protocol safety schema/source binding is invalid")
    if "phase0_identity_binding" in protocol:
        validate_phase0_identity_binding(protocol["phase0_identity_binding"])
        if protocol["phase0_identity_binding"]["source_identity_sha256"] != protocol["calibration_source_identity_sha256"]:
            raise ValueError("calibrated protocol Phase 0 evaluator source identity is invalid")
    if protocol.get("protocol_version") == GPU_BOOTSTRAP_PROTOCOL_VERSION:
        reference_runtime = protocol.get("matched_reference_variance_calibration", {}).get(
            "normalized_runtime_identity")
        if (not isinstance(reference_runtime, dict)
                or protocol.get("gpu_statistics_implementation") !=
                    _gpu_statistics_implementation(reference_runtime)):
            raise ValueError("GPU bootstrap implementation/runtime binding is invalid")
    ids = protocol.get("calibration_ids")
    pools = protocol.get("calibration_seed_pools")
    report_hashes = protocol.get("calibration_report_sha256")
    if (not isinstance(ids, dict) or set(ids) != set(scenarios)
            or not isinstance(pools, dict) or set(pools) != set(scenarios)
            or not isinstance(report_hashes, dict) or set(report_hashes) != set(scenarios)):
        raise ValueError("calibrated protocol is missing prerequisite calibration reports")
    if any(not isinstance(value, str) or len(value) != 64
           or any(c not in "0123456789abcdef" for c in value)
           for value in (*ids.values(), *report_hashes.values())):
        raise ValueError("calibrated protocol contains an invalid calibration report ID")
    combined_id = hashlib.sha256(json.dumps(ids, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if protocol.get("calibration_id") != combined_id:
        raise ValueError("calibration bundle identity mismatch")
    # Categories sharing a qualification stratum intentionally share seeds so
    # their outcomes can be paired. Reuse across distinct strata/scenarios is
    # still forbidden.
    calibration_seed_strata = {}
    for name in scenarios:
        pool = pools[name]
        if (not isinstance(pool, dict) or type(pool.get("master_seed")) is not int
                or pool["master_seed"] < 0 or type(pool.get("episodes")) is not int
                or pool["episodes"] < MIN_EPISODES or type(pool.get("repetitions")) is not int
                or pool["repetitions"] < 2):
            raise ValueError(f"calibration seed pool is invalid for {name}")
        expected_pool = [{"category": c.category, "layout_id": c.layout_id,
            "defender_family": c.defender_family, "start_condition": c.start_condition,
            "sensor_profile": c.sensor_profile, "seeds": list(c.seeds)}
            for c in protocol_cases(name, "calibration", episodes=pool["episodes"],
                master_seed=pool["master_seed"])]
        if pool.get("cases") != expected_pool:
            raise ValueError(f"calibration seed pool case table is invalid for {name}")
        for case in expected_pool:
            for seed in case["seeds"]:
                stratum = (name, case["layout_id"], case["defender_family"],
                    case["start_condition"], case["sensor_profile"])
                if seed % 4 != 3 or (seed in calibration_seed_strata
                        and calibration_seed_strata[seed] != stratum):
                    raise ValueError("calibration seed pools overlap or use a non-calibration partition")
                calibration_seed_strata[seed] = stratum
    categories = set(SCENARIOS[target].required_categories)
    gain_margins = protocol.get("minimum_useful_gain")
    safety_margins = protocol.get("safety_noninferiority_margins")
    if (not isinstance(gain_margins, dict) or set(gain_margins) != categories
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in gain_margins.values())):
        raise ValueError("calibrated protocol useful-gain categories or margins are invalid")
    derivation = protocol.get("useful_gain_derivation")
    if protocol.get("protocol_version") in (CALIBRATED_PROTOCOL_VERSION,
            GPU_BOOTSTRAP_PROTOCOL_VERSION):
        if (not isinstance(derivation, dict)
                or derivation.get("schema_version") != 1
                or derivation.get("method") != "explicit_reviewed_reference_variance_margins"
                or derivation.get("minimum_useful_gain") != gain_margins
                or not isinstance(derivation.get("review_sha256"), str)
                or derivation.get("review_sha256") != protocol.get(
                    "matched_reference_variance_calibration", {}).get("margin_review", {}).get("review_sha256")
                or not isinstance(derivation.get("reviewed_by"), str)
                or not derivation["reviewed_by"].strip()
                or not isinstance(derivation.get("reviewed_utc"), str)
                or not derivation["reviewed_utc"].strip()):
            raise ValueError("v4 useful-gain margins lack an explicit reviewed reference-variance decision")
        reference_calibration = protocol.get("matched_reference_variance_calibration")
        review = reference_calibration.get("margin_review") if isinstance(reference_calibration, dict) else None
        if (not isinstance(reference_calibration, dict)
                or reference_calibration.get("version") != MATCHED_REFERENCE_VARIANCE_VERSION
                or not isinstance(reference_calibration.get("calibration_ids"), dict)
                or set(reference_calibration["calibration_ids"]) != set(scenarios)
                or not isinstance(reference_calibration.get("report_sha256"), dict)
                or set(reference_calibration["report_sha256"]) != set(scenarios)
                or not isinstance(reference_calibration.get("seed_table_sha256"), dict)
                or set(reference_calibration["seed_table_sha256"]) != set(scenarios)
                or not isinstance(reference_calibration.get("normalized_runtime_identity"), dict)
                or not isinstance(review, dict)
                or any(not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                    for mapping_name in ("calibration_ids", "report_sha256", "seed_table_sha256")
                    for value in reference_calibration.get(mapping_name, {}).values())
                or review.get("reviewed_by") != derivation.get("reviewed_by")
                or review.get("reviewed_utc") != derivation.get("reviewed_utc")
                or review.get("reference_variant") != protocol.get("paired_reference_variant")
                or review.get("reference_calibration_ids") != reference_calibration.get("calibration_ids")
                or review.get("reference_report_sha256") != reference_calibration.get("report_sha256")
                or review.get("minimum_useful_gain") != gain_margins
                or review.get("safety_noninferiority_margins") != safety_margins
                or review.get("review_sha256") != _canonical_sha256({key: value for key, value in review.items()
                    if key != "review_sha256"})):
            raise ValueError("v4 matched-reference calibration or explicit margin review binding is invalid")
    elif (not isinstance(derivation, dict)
            or derivation.get("schema_version") != 1
            or derivation.get("method") != "declared_operational_threshold_plus_multiplier_times_max_stratum_standard_error"
            or derivation.get("calibration_id") != ids[target]
            or not isinstance(derivation.get("operational_gain_threshold"), dict)
            or set(derivation["operational_gain_threshold"]) != categories
            or not isinstance(derivation.get("uncertainty_by_category"), dict)
            or set(derivation["uncertainty_by_category"]) != categories
            or type(derivation.get("uncertainty_multiplier")) not in (int, float)
            or not math.isfinite(derivation["uncertainty_multiplier"])
            or derivation["uncertainty_multiplier"] < 0):
        raise ValueError("calibrated protocol useful-gain derivation provenance is invalid")
    for category in categories:
        if protocol.get("protocol_version") in (CALIBRATED_PROTOCOL_VERSION,
                GPU_BOOTSTRAP_PROTOCOL_VERSION):
            break
        uncertainty = derivation["uncertainty_by_category"][category]
        threshold = derivation["operational_gain_threshold"][category]
        se = uncertainty.get("maximum_stratum_standard_error") if isinstance(uncertainty, dict) else None
        if (type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0
                or type(se) not in (int, float) or not math.isfinite(se) or se < 0
                or gain_margins[category] != threshold + derivation["uncertainty_multiplier"] * se):
            raise ValueError("calibrated protocol gain margin does not match its declared derivation")
    if (not isinstance(safety_margins, dict) or set(safety_margins) != SAFETY_METRIC_NAMES
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not 0 <= value <= 1 for value in safety_margins.values())):
        raise ValueError("calibrated protocol safety metric names or margins are invalid")
    episodes, master_seed = protocol.get("episodes_per_category_stratum"), protocol.get("master_seed")
    if type(episodes) is not int or episodes < MIN_EPISODES or type(master_seed) is not int:
        raise ValueError("calibrated protocol episode or seed settings are invalid")
    expected_cases = [asdict(case) | {"seeds": list(case.seeds)}
        for name in scenarios for split in ("train", "validation", "test")
        for case in protocol_cases(name, split,
            episodes=TRAIN_SEED_POOL_SIZE if split == "train" else episodes,
            master_seed=master_seed)]
    if protocol.get("cases") != expected_cases:
        raise ValueError("calibrated protocol case table or exact seed lists are invalid")
    protocol_seeds = {seed for row in expected_cases for seed in row["seeds"]}
    if set(calibration_seed_strata) & protocol_seeds:
        raise ValueError("calibration seed pools overlap train/validation/test seeds")


_EVIDENCE_FIELDS = ("schema_version", "evidence_schema_version", "qualification", "evaluation_id", "protocol_id",
    "scenario_id", "split", "candidate_policy", "bundle_id", "policy_manifest",
    "retention_scenario_ids",
    "source_identity_sha256", "gate_definition", "gate_calibration_status",
    "protocol_version", "calibration_id", "calibration_ids", "calibration_seed_pools",
    "calibration_report_sha256", "minimum_useful_gain", "useful_gain_derivation",
    "useful_gain_calibration_basis",
    "safety_noninferiority_margins", "safety_gate_status",
    "safety_metrics_schema", "safety_metrics_source_identity_sha256",
    "paired_reference_policy", "paired_reference_mode", "paired_reference_variant",
    "paired_reference_identity", "reference_selection_basis",
    "safety_floor_policy", "matched_reference_variance_calibration",
    "paired", "episodes_per_category_stratum", "gpu_statistics_implementation",
    "statistics_runtime_device", "confidence_intervals", "gpu_numeric_verdict",
    "master_seed", "records", "case_ids", "reference_report", "generated_utc",
    "final_test_isolated", "final_test_results")


_HISTORICAL_EVIDENCE_FIELDS = tuple(key for key in _EVIDENCE_FIELDS
    if key not in ("gpu_statistics_implementation", "statistics_runtime_device",
                   "confidence_intervals", "gpu_numeric_verdict"))
_LEGACY_EVIDENCE_FIELDS = tuple(key for key in _HISTORICAL_EVIDENCE_FIELDS
    if key not in ("evidence_schema_version", "final_test_isolated", "final_test_results",
                   "useful_gain_calibration_basis", "paired_reference_identity",
                   "paired_reference_mode", "paired_reference_variant",
                   "reference_selection_basis", "safety_floor_policy",
                   "matched_reference_variance_calibration"))
_D0_PAIRED_EVIDENCE_FIELDS = tuple(key for key in _HISTORICAL_EVIDENCE_FIELDS
    if key != "matched_reference_variance_calibration")


def _evidence_digest(report: dict) -> str:
    version = report.get("evidence_schema_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("qualification evidence schema version is unsupported")
    if report.get("protocol_version") in (GPU_BOOTSTRAP_PROTOCOL_VERSION,
            CALIBRATED_PROTOCOL_VERSION, D0_PAIRED_REFERENCE_PROTOCOL_VERSION) and version != 2:
        raise ValueError("calibrated v3/v4/v5 reports require evidence schema v2")
    fields = (_LEGACY_EVIDENCE_FIELDS if version == 1 else
        _D0_PAIRED_EVIDENCE_FIELDS if report.get("protocol_version") == D0_PAIRED_REFERENCE_PROTOCOL_VERSION
        else _EVIDENCE_FIELDS if report.get("protocol_version") == GPU_BOOTSTRAP_PROTOCOL_VERSION
        else _HISTORICAL_EVIDENCE_FIELDS)
    evidence = {key: report.get(key) for key in fields}
    return hashlib.sha256(json.dumps(evidence, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _validate_evidence_digest(report: dict) -> None:
    if (not isinstance(report, dict)
            or type(report.get("evidence_schema_version", 1)) is not int
            or report.get("evidence_schema_version", 1) not in (1, 2)
            or (report.get("protocol_version") in (GPU_BOOTSTRAP_PROTOCOL_VERSION,
                CALIBRATED_PROTOCOL_VERSION, D0_PAIRED_REFERENCE_PROTOCOL_VERSION)
                and report.get("evidence_schema_version") != 2)
            or report.get("evidence_sha256") != _evidence_digest(report)):
        raise ValueError("qualification evidence digest is missing or invalid")


def _validate_report_protocol_binding(report: dict, protocol: dict) -> None:
    bindings = ("protocol_id", "protocol_version", "calibration_id",
        "calibration_ids", "calibration_seed_pools",
        "calibration_report_sha256", "useful_gain_derivation",
        "retention_scenario_ids",
        "gate_calibration_status", "minimum_useful_gain",
        "safety_noninferiority_margins", "safety_gate_status",
        "safety_metrics_schema", "safety_metrics_source_identity_sha256",
        "episodes_per_category_stratum", "master_seed")
    if protocol.get("protocol_version") != LEGACY_CALIBRATED_PROTOCOL_VERSION:
        bindings += ("paired_reference_policy", "paired_reference_mode",
                     "paired_reference_variant", "paired_reference_identity",
                     "reference_selection_basis", "safety_floor_policy",
                     "useful_gain_calibration_basis")
    if protocol.get("protocol_version") in (CALIBRATED_PROTOCOL_VERSION,
            GPU_BOOTSTRAP_PROTOCOL_VERSION):
        bindings += ("matched_reference_variance_calibration",)
    if protocol.get("protocol_version") == GPU_BOOTSTRAP_PROTOCOL_VERSION:
        bindings += ("gpu_statistics_implementation",)
    if any(report.get(field) != protocol.get(field) for field in bindings):
        raise ValueError("qualification report calibration/protocol provenance mismatch")
    if protocol.get("protocol_version") == GPU_BOOTSTRAP_PROTOCOL_VERSION:
        runtime = report.get("statistics_runtime_device")
        if not isinstance(runtime, dict) or runtime != _require_calibration_gpu(
                runtime.get("device")):
            raise ValueError("GPU statistics report runtime/device identity is invalid")
        if (_normalized_calibration_runtime(runtime) !=
                protocol["gpu_statistics_implementation"]["runtime_identity"]):
            raise ValueError("GPU statistics runtime differs from the frozen protocol")
        verdict = report.get("gpu_numeric_verdict")
        if verdict is not None and (not isinstance(verdict, dict)
                or verdict.get("implementation_id") != GPU_BOOTSTRAP_IMPLEMENTATION_ID
                or verdict.get("evaluation_id") != report.get("evaluation_id")
                or verdict.get("protocol_id") != report.get("protocol_id")
                or verdict.get("runtime_device") != runtime):
            raise ValueError("GPU numeric verdict is not bound to this report/runtime")
    legacy_v2 = report.get("protocol_version") == LEGACY_CALIBRATED_PROTOCOL_VERSION
    valid_reference = (report.get("paired_reference_policy") == REFERENCE_POLICY
        if legacy_v2 else report.get("paired_reference_policy") in MATCHED_REFERENCE_POLICIES)
    if not valid_reference or report.get("paired") is not True:
        raise ValueError("qualification report reference pairing provenance is invalid")
    reference_report = report.get("reference_report")
    if (not isinstance(reference_report, dict) or (not legacy_v2 and (
            reference_report.get("paired_reference_policy") != report.get("paired_reference_policy")
            or reference_report.get("paired_reference_mode") != report.get("paired_reference_mode")
            or reference_report.get("paired_reference_variant") != report.get("paired_reference_variant")
            or reference_report.get("paired_reference_identity") != report.get("paired_reference_identity")
            or reference_report.get("reference_selection_basis") != report.get("reference_selection_basis")
            or reference_report.get("safety_floor_policy") != report.get("safety_floor_policy")
            or (report.get("protocol_version") in (CALIBRATED_PROTOCOL_VERSION,
                GPU_BOOTSTRAP_PROTOCOL_VERSION)
                and reference_report.get("matched_reference_variance_calibration")
                    != report.get("matched_reference_variance_calibration"))))):
        raise ValueError("qualification reference report declaration differs from its frozen protocol")


def _variance(values):
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _require_calibration_gpu(device: str) -> dict:
    """Require an explicit, visible CUDA/HIP device before D0 calibration."""
    import torch
    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid calibration device {device!r}") from exc
    if selected.type != "cuda" or selected.index is None:
        raise ValueError("D0 calibration requires an explicit GPU device such as cuda:0")
    if (selected.index < 0 or not torch.cuda.is_available()
            or selected.index >= torch.cuda.device_count()):
        raise RuntimeError(f"requested calibration GPU {selected} is not available")
    name = torch.cuda.get_device_name(selected.index)
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError(f"could not identify requested calibration GPU {selected}")
    return {"device": str(selected), "device_name": name,
        "backend": "hip" if torch.version.hip else "cuda",
        "torch_version": str(torch.__version__),
        "hip_version": str(torch.version.hip) if torch.version.hip is not None else None}


def _canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _calibration_source_identity_sha256() -> str:
    from .curriculum_training import source_identity
    # Match evaluate_meco's established source identity hash byte for byte.
    # Its default JSON separators are part of the existing v2 report contract.
    return hashlib.sha256(json.dumps(source_identity(), sort_keys=True,
        allow_nan=False).encode()).hexdigest()


def _calibration_case_table(scenario_id, episodes, repetitions, master_seed):
    cases = protocol_cases(scenario_id, "calibration", episodes=episodes,
                           master_seed=master_seed)
    return [{"category": case.category, "layout_id": case.layout_id,
        "defender_family": case.defender_family,
        "start_condition": case.start_condition,
        "sensor_profile": case.sensor_profile, "seeds": list(case.seeds)}
        for case in cases]


def _calibration_shard_config(scenario_id, episodes, repetitions, master_seed):
    return {"scenario_id": scenario_id, "split": "calibration",
        "episodes_per_category_stratum": episodes, "repetitions": repetitions,
        "master_seed": master_seed, "seed_partition_code": 3,
        "repeat_semantics": "same exact calibration seeds repeated independently",
        "policy": REFERENCE_POLICY, "policy_id": "D0_production_deterministic",
        "evaluation_policies": [], "control_policies": [REFERENCE_POLICY],
        "include_controls": True}


def _validate_d0_calibration_case_table(scenario_id, episodes, master_seed, cases):
    calibration_seeds = {seed for case in cases for seed in case["seeds"]}
    if any(seed % 4 != 3 for seed in calibration_seeds):
        raise ValueError("calibration seed partition is not isolated")
    for split in ("train", "validation", "test"):
        other = {seed for case in protocol_cases(scenario_id, split,
            episodes=episodes, master_seed=master_seed) for seed in case.seeds}
        if calibration_seeds & other:
            raise ValueError(f"calibration seeds overlap the {split} partition")


def _calibration_work_units(case_table, repetitions):
    return [{"repetition": repetition, **{key: case[key] for key in (
        "category", "layout_id", "defender_family", "start_condition",
        "sensor_profile")}, "seed": seed}
        for repetition in range(repetitions) for case in case_table
        for seed in case["seeds"]]


def _unit_key(unit) -> str:
    return _canonical_sha256(unit)


def _normalized_calibration_runtime(runtime_device):
    _validate_calibration_runtime_device(runtime_device)
    normalized = {key: value for key, value in runtime_device.items() if key != "device"}
    normalized["device_model"] = runtime_device.get("device_name")
    return normalized


def _d0_calibration_shard_manifests(*, scenario_id, episodes, repetitions,
                                    master_seed, shard_count, device):
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("D0 calibration requires at least two repetitions")
    if type(episodes) is not int or episodes < MIN_EPISODES:
        raise ValueError(f"D0 calibration requires at least {MIN_EPISODES} episodes per stratum")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("calibration master seed must be a nonnegative integer")
    if type(shard_count) is not int or shard_count < 1:
        raise ValueError("calibration shard count must be a positive integer")
    runtime = _require_calibration_gpu(device)
    cases = _calibration_case_table(scenario_id, episodes, repetitions, master_seed)
    config = _calibration_shard_config(scenario_id, episodes, repetitions, master_seed)
    _validate_d0_calibration_case_table(scenario_id, episodes, master_seed, cases)
    protocol_identity = _canonical_sha256({"protocol_version": D0_CALIBRATION_VERSION,
        "config": config, "case_table": cases})
    source_identity = _calibration_source_identity_sha256()
    units = _calibration_work_units(cases, repetitions)
    if shard_count > len(units):
        raise ValueError("calibration shard count cannot exceed the number of work units")
    buckets = [[] for _ in range(shard_count)]
    # Stable hash ranking plus round-robin assignment gives deterministic,
    # disjoint, near-balanced shards without Python's randomized hash().
    ranked_units = sorted(units, key=lambda unit: _unit_key({
        "protocol": protocol_identity, "unit": unit}))
    for rank, unit in enumerate(ranked_units):
        buckets[rank % shard_count].append(unit)
    common = {"schema_version": 1,
        "artifact_type": D0_CALIBRATION_SHARD_MANIFEST,
        "protocol_version": D0_CALIBRATION_VERSION,
        "config": config, "case_table": cases,
        "protocol_identity_sha256": protocol_identity,
        "source_identity_sha256": source_identity,
        "runtime_device": runtime,
        "normalized_runtime_identity": _normalized_calibration_runtime(runtime),
        "shard_count": shard_count,
        "work_unit_count": len(units)}
    manifests = []
    for shard_index, assigned in enumerate(buckets):
        body = {**common, "shard_index": shard_index,
            "work_units": assigned}
        body["manifest_id"] = _canonical_sha256(body)
        manifests.append(body)
    return manifests


def _validate_d0_shard_manifest(manifest):
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("artifact_type") != D0_CALIBRATION_SHARD_MANIFEST
            or manifest.get("protocol_version") != D0_CALIBRATION_VERSION):
        raise ValueError("unsupported D0 calibration shard manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("manifest_id") != _canonical_sha256(body):
        raise ValueError("D0 calibration shard manifest digest is invalid")
    config, cases = manifest.get("config"), manifest.get("case_table")
    if not isinstance(config, dict) or config.get("split") != "calibration":
        raise ValueError("D0 calibration shard config is invalid")
    expected_config = _calibration_shard_config(config.get("scenario_id"),
        config.get("episodes_per_category_stratum"), config.get("repetitions"),
        config.get("master_seed"))
    if config != expected_config:
        raise ValueError("D0 calibration shard settings differ from the frozen command contract")
    expected_cases = _calibration_case_table(config.get("scenario_id"),
        config.get("episodes_per_category_stratum"), config.get("repetitions"),
        config.get("master_seed"))
    if cases != expected_cases:
        raise ValueError("D0 calibration shard case table differs from frozen protocol")
    _validate_d0_calibration_case_table(config["scenario_id"],
        config["episodes_per_category_stratum"], config["master_seed"], cases)
    expected_protocol_identity = _canonical_sha256({
        "protocol_version": D0_CALIBRATION_VERSION, "config": config,
        "case_table": expected_cases})
    if manifest.get("protocol_identity_sha256") != expected_protocol_identity:
        raise ValueError("D0 calibration shard protocol identity is invalid")
    if (type(manifest.get("shard_count")) is not int or manifest["shard_count"] < 1
            or type(manifest.get("shard_index")) is not int
            or not 0 <= manifest["shard_index"] < manifest["shard_count"]):
        raise ValueError("D0 calibration shard index/count is invalid")
    current_units = _calibration_work_units(expected_cases, config["repetitions"])
    if manifest["shard_count"] > len(current_units):
        raise ValueError("calibration shard count exceeds the work unit count")
    all_buckets = [[] for _ in range(manifest["shard_count"])]
    ranked_units = sorted(current_units, key=lambda unit: _unit_key({
        "protocol": expected_protocol_identity, "unit": unit}))
    for rank, unit in enumerate(ranked_units):
        all_buckets[rank % manifest["shard_count"]].append(unit)
    if (manifest.get("work_unit_count") != len(current_units)
            or manifest.get("work_units") != all_buckets[manifest["shard_index"]]):
        raise ValueError("D0 calibration shard assignment is invalid")
    runtime = manifest.get("runtime_device")
    if manifest.get("normalized_runtime_identity") != _normalized_calibration_runtime(runtime):
        raise ValueError("D0 calibration shard normalized runtime identity is invalid")
    for key in ("source_identity_sha256",):
        digest = manifest.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("D0 calibration shard source identity is invalid")
    return manifest


def _validate_d0_shard_entries(artifact, manifest, *, require_complete):
    expected = {_unit_key(unit): unit for unit in manifest["work_units"]}
    found = {}
    for entry in artifact.get("entries", []):
        if not isinstance(entry, dict):
            raise ValueError("D0 calibration shard entry is malformed")
        unit_key = entry.get("work_unit_sha256")
        unit = expected.get(unit_key)
        row = entry.get("record")
        if unit is None or unit_key in found or not isinstance(row, dict):
            raise ValueError("D0 calibration shard has unexpected or duplicate work unit")
        if any(row.get(field) != value for field, value in unit.items()):
            raise ValueError("D0 calibration shard record differs from its assigned work unit")
        if (row.get("policy") != REFERENCE_POLICY
                or row.get("split") != "calibration"
                or row.get("policy_id") != "D0_production_deterministic"
                or row.get("checkpoint_sha256") is not None
                or row.get("scenario_id") != manifest["config"]["scenario_id"]
                or row.get("source_identity_sha256") != manifest["source_identity_sha256"]
                or row.get("calibration_runtime_device") != manifest["runtime_device"]):
            raise ValueError("D0 calibration shard record provenance is invalid")
        metrics = row.get("metrics")
        if (not isinstance(metrics, dict)
                or type(metrics.get("useful_throughput")) not in (int, float)
                or not math.isfinite(metrics["useful_throughput"])
                or metrics["useful_throughput"] < 0
                or {key for key in metrics if isinstance(key, str) and key.startswith("safety_")}
                    != SAFETY_METRIC_NAMES
                or any(type(metrics.get(name)) not in (int, float)
                    or not math.isfinite(metrics[name]) or not 0 <= metrics[name] <= 1
                    for name in SAFETY_METRIC_NAMES)):
            raise ValueError("D0 calibration shard row metrics are invalid")
        found[unit_key] = entry
    if require_complete and set(found) != set(expected):
        raise ValueError("D0 calibration shard record coverage is incomplete")
    return [found[_unit_key(unit)] for unit in manifest["work_units"] if _unit_key(unit) in found]


def _validate_d0_shard_artifact(artifact, manifest):
    if (not isinstance(artifact, dict) or artifact.get("schema_version") != 1
            or artifact.get("artifact_type") != D0_CALIBRATION_SHARD_ARTIFACT
            or artifact.get("manifest_id") != manifest["manifest_id"]
            or artifact.get("manifest") != manifest):
        raise ValueError("D0 calibration shard artifact does not match its manifest")
    unsigned = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if artifact.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise ValueError("D0 calibration shard artifact digest is invalid")
    entries = _validate_d0_shard_entries(artifact, manifest, require_complete=True)
    if (not isinstance(artifact.get("policy_manifest"), dict)
            or artifact.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
            or artifact.get("safety_metrics_source_identity_sha256") != manifest["source_identity_sha256"]):
        raise ValueError("D0 calibration shard policy/safety provenance is invalid")
    _validate_calibration_d0_manifest(artifact["policy_manifest"],
                                      manifest["source_identity_sha256"])
    return entries


def _d0_calibration_shard_artifact(manifest, entries, policy_manifest):
    artifact = {"schema_version": 1,
        "artifact_type": D0_CALIBRATION_SHARD_ARTIFACT,
        "manifest_id": manifest["manifest_id"], "manifest": manifest,
        "policy_manifest": policy_manifest,
        "safety_metrics_schema": SAFETY_METRICS_SCHEMA,
        "safety_metrics_source_identity_sha256": manifest["source_identity_sha256"],
        "entries": entries}
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def run_d0_calibration_shard(manifest, *, device: str, output: str | Path) -> dict:
    """Run or resume one immutable D0 calibration shard artifact."""
    _validate_d0_shard_manifest(manifest)
    runtime = _require_calibration_gpu(device)
    if runtime != manifest["runtime_device"]:
        raise ValueError("runtime device identity differs from D0 calibration shard manifest")
    if _calibration_source_identity_sha256() != manifest["source_identity_sha256"]:
        raise ValueError("source identity differs from D0 calibration shard manifest")
    destination = Path(output)
    if destination.is_file():
        completed = _load_json(destination)
        _validate_d0_shard_artifact(completed, manifest)
        return completed
    partial_path = destination.with_suffix(destination.suffix + ".partial")
    entries = []
    policy_manifest = None
    if partial_path.is_file():
        partial = _load_json(partial_path)
        if (partial.get("schema_version") != 1
                or partial.get("artifact_type") != "meco-d0-calibration-shard-progress-v1"
                or partial.get("manifest") != manifest
                or partial.get("manifest_id") != manifest["manifest_id"]
                or partial.get("runtime_device") != runtime
                or partial.get("source_identity_sha256") != manifest["source_identity_sha256"]):
            raise ValueError("D0 calibration resume state does not match its immutable manifest")
        partial_unsigned = {key: value for key, value in partial.items() if key != "progress_sha256"}
        if partial.get("progress_sha256") != _canonical_sha256(partial_unsigned):
            raise ValueError("D0 calibration resume state digest is invalid")
        entries = _validate_d0_shard_entries(partial, manifest, require_complete=False)
        policy_manifest = partial.get("policy_manifest")
        if entries and not isinstance(policy_manifest, dict):
            raise ValueError("D0 calibration resume state is missing its policy manifest")
    completed_units = {entry["work_unit_sha256"] for entry in entries}
    config = manifest["config"]
    cases_by_key = {(case["category"], case["layout_id"], case["defender_family"],
        case["start_condition"], case["sensor_profile"]): case
        for case in manifest["case_table"]}
    # The checkpoint boundary is one case/repetition group. An interruption
    # during a group leaves that group absent and safely rerunnable.
    groups = {}
    for unit in manifest["work_units"]:
        key = (unit["repetition"], unit["category"], unit["layout_id"],
            unit["defender_family"], unit["start_condition"], unit["sensor_profile"])
        groups.setdefault(key, []).append(unit)
    from .meco_track_training import evaluate_meco
    for group_key, units in groups.items():
        missing = [unit for unit in units if _unit_key(unit) not in completed_units]
        if not missing:
            continue
        repetition, category, layout_id, defender_family, start_condition, sensor_profile = group_key
        result = evaluate_meco({}, scenario_id=config["scenario_id"],
            categories=(category,), episodes=len(missing),
            master_seed=config["master_seed"],
            episode_seeds=tuple(unit["seed"] for unit in missing),
            device=device, split="calibration", evaluation_policies=(),
            control_policies=(REFERENCE_POLICY,), include_controls=True,
            qualification_case={"layout_id": layout_id,
                "defender_family": defender_family,
                "start_condition": start_condition,
                "sensor_profile": sensor_profile})
        if (result.get("source_identity_sha256") != manifest["source_identity_sha256"]
                or result.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
                or result.get("safety_metrics_source_identity_sha256") != manifest["source_identity_sha256"]):
            raise ValueError("D0 calibration shard evaluator provenance differs from its manifest")
        current_policy_manifest = result.get("policy_manifest")
        if (not isinstance(current_policy_manifest, dict)
                or current_policy_manifest.get(REFERENCE_POLICY, {}).get("policy_id")
                    != config["policy_id"]):
            raise ValueError("D0 calibration shard did not evaluate the production deterministic controller")
        if policy_manifest is not None and policy_manifest != current_policy_manifest:
            raise ValueError("D0 policy manifest changed while resuming calibration shard")
        policy_manifest = current_policy_manifest
        rows_by_seed = {}
        for row in result.get("records", []):
            if row.get("policy") != REFERENCE_POLICY or row.get("seed") in rows_by_seed:
                raise ValueError("D0 calibration shard evaluator returned unexpected/duplicate records")
            rows_by_seed[row["seed"]] = row
        if set(rows_by_seed) != {unit["seed"] for unit in missing}:
            raise ValueError("D0 calibration shard evaluator returned incomplete seed coverage")
        for unit in missing:
            record = {"repetition": repetition, **rows_by_seed[unit["seed"]],
                "calibration_runtime_device": runtime}
            entries.append({"work_unit_sha256": _unit_key(unit), "record": record})
        progress = {"schema_version": 1,
            "artifact_type": "meco-d0-calibration-shard-progress-v1",
            "manifest_id": manifest["manifest_id"], "manifest": manifest,
            "runtime_device": runtime,
            "source_identity_sha256": manifest["source_identity_sha256"],
            "policy_manifest": policy_manifest,
            "entries": sorted(entries, key=lambda entry: entry["work_unit_sha256"])}
        progress["progress_sha256"] = _canonical_sha256(progress)
        _atomic_json(partial_path, progress)
        completed_units.update(entry["work_unit_sha256"] for entry in entries)
    artifact = _d0_calibration_shard_artifact(manifest, entries, policy_manifest)
    _validate_d0_shard_artifact(artifact, manifest)
    write_immutable_report(destination, artifact)
    partial_path.unlink(missing_ok=True)
    return artifact


def _d0_calibration_summary(cases, repetitions, records, scenario_id):
    summaries = {}
    for case in cases:
        key = (case["category"], case["layout_id"], case["defender_family"],
               case["start_condition"], case["sensor_profile"])
        by_seed = {}
        for row in records:
            if (row["category"], row["layout_id"], row["defender_family"],
                row["start_condition"], row["sensor_profile"]) == key:
                by_seed.setdefault(row["seed"], []).append(
                    float(row["metrics"]["useful_throughput"]))
        expected = set(case["seeds"])
        if set(by_seed) != expected or any(len(values) != repetitions for values in by_seed.values()):
            raise ValueError("merged calibration report has incomplete repeated seed coverage")
        seed_means = [sum(by_seed[seed]) / repetitions for seed in case["seeds"]]
        repeat_variances = [_variance(by_seed[seed]) for seed in case["seeds"]]
        summaries["/".join(key)] = {"category": case["category"],
            "layout_id": case["layout_id"], "defender_family": case["defender_family"],
            "start_condition": case["start_condition"], "sensor_profile": case["sensor_profile"],
            "seed_count": len(case["seeds"]), "repetition_count": repetitions,
            "mean_useful_throughput": sum(seed_means) / len(seed_means),
            "between_seed_variance": _variance(seed_means),
            "mean_within_seed_repeat_variance": sum(repeat_variances) / len(repeat_variances),
            "seed_digest": hashlib.sha256(json.dumps(list(case["seeds"]),
                separators=(",", ":")).encode()).hexdigest()}
    return summaries


def _merged_d0_calibration_report(manifests, artifacts):
    first = manifests[0]
    count = first["shard_count"]
    if len(manifests) != count or len(artifacts) != count:
        raise ValueError("D0 calibration merge requires every shard exactly once")
    by_index = {}
    for manifest, artifact in zip(manifests, artifacts):
        _validate_d0_shard_manifest(manifest)
        if manifest["manifest_id"] in by_index.values():
            raise ValueError("D0 calibration merge contains a duplicate manifest")
        if manifest["shard_count"] != count:
            raise ValueError("D0 calibration merge mixes shard counts")
        index = manifest["shard_index"]
        if index in by_index:
            raise ValueError("D0 calibration merge contains a duplicate shard index")
        by_index[index] = manifest["manifest_id"]
        _validate_d0_shard_artifact(artifact, manifest)
    if set(by_index) != set(range(count)):
        raise ValueError("D0 calibration merge is missing a shard index")
    common_fields = ("protocol_version", "config", "case_table", "protocol_identity_sha256",
        "source_identity_sha256", "normalized_runtime_identity", "shard_count", "work_unit_count")
    base = manifests[0]
    for manifest in manifests[1:]:
        if any(manifest.get(key) != base.get(key) for key in common_fields):
            raise ValueError("D0 calibration shard manifests disagree on protocol/source/runtime identity")
    if _calibration_source_identity_sha256() != base["source_identity_sha256"]:
        raise ValueError("calibration source identity changed since shard planning")
    runtime_identity = base["normalized_runtime_identity"]
    if (runtime_identity.get("backend") not in ("cuda", "hip")
            or runtime_identity.get("device_model") != base["runtime_device"].get("device_name")):
        raise ValueError("calibration shard runtime identity is inconsistent")
    policy_manifests = [artifact.get("policy_manifest") for artifact in artifacts]
    if any(value != policy_manifests[0] for value in policy_manifests[1:]):
        raise ValueError("D0 calibration shards disagree on controller policy manifest")
    _validate_calibration_d0_manifest(policy_manifests[0], base["source_identity_sha256"])
    entries = []
    for manifest, artifact in zip(manifests, artifacts):
        entries.extend(_validate_d0_shard_artifact(artifact, manifest))
    expected_units = _calibration_work_units(base["case_table"], base["config"]["repetitions"])
    expected_keys = {_unit_key(unit) for unit in expected_units}
    actual_keys = [entry["work_unit_sha256"] for entry in entries]
    if set(actual_keys) != expected_keys or len(actual_keys) != len(expected_keys):
        raise ValueError("D0 calibration merge has missing, duplicate, or unexpected work units")
    records = [dict(entry["record"]) for entry in sorted(entries,
        key=lambda entry: (entry["record"]["repetition"], entry["record"]["category"],
            entry["record"]["layout_id"], entry["record"]["defender_family"],
            entry["record"]["start_condition"], entry["record"]["sensor_profile"],
            entry["record"]["seed"]))]
    config = base["config"]
    case_table = base["case_table"]
    summaries = _d0_calibration_summary(case_table, config["repetitions"], records,
        config["scenario_id"])
    for record in records:
        # The evaluator's D0 identity is stable; the shard artifact binds that
        # policy ID even though D0 has no checkpoint file.
        if record.get("policy_id") != config["policy_id"]:
            raise ValueError("D0 calibration shard policy identity changed")
    # Reconstruct the exact schema-v2 case identities consumed by protocol intake.
    body = {"schema_version": 2, "calibration": True,
        "protocol_version": D0_CALIBRATION_VERSION, "scenario_id": config["scenario_id"],
        "master_seed": config["master_seed"], "split": "calibration",
        "seed_partition_code": 3,
        "episodes_per_category_stratum": config["episodes_per_category_stratum"],
        "repetitions": config["repetitions"],
        "repeat_semantics": config["repeat_semantics"],
        "case_ids": [{key: case[key] for key in ("category", "layout_id", "defender_family",
            "start_condition", "sensor_profile")} for case in case_table],
        "policy_manifest": policy_manifests[0],
        "source_identity_sha256": base["source_identity_sha256"],
        "runtime_device": dict(base["runtime_device"]),
        "normalized_runtime_identity": runtime_identity,
        "safety_metrics_schema": SAFETY_METRICS_SCHEMA,
        "safety_metrics_source_identity_sha256": base["source_identity_sha256"],
        "summary": summaries, "records": records,
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()}
    body["calibration_id"] = _canonical_sha256(body)
    return body


def derive_useful_gain_margins(reports: dict, target: str, declaration: dict) -> tuple[dict, dict]:
    """Add declared operational gain to a deterministic calibration uncertainty allowance."""
    categories = set(SCENARIOS[target].required_categories)
    if (not isinstance(declaration, dict)
            or declaration.get("schema_version") != 1
            or not isinstance(declaration.get("operational_gain_threshold"), dict)
            or set(declaration["operational_gain_threshold"]) != categories
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in declaration["operational_gain_threshold"].values())):
        raise ValueError("useful-gain declaration must give a nonnegative operational threshold for every target category")
    multiplier = declaration.get("uncertainty_multiplier")
    if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError("useful-gain declaration requires a finite nonnegative uncertainty_multiplier")
    target_report = reports[target]
    summaries = target_report.get("summary")
    if not isinstance(summaries, dict):
        raise ValueError("target calibration report has no variance summaries")
    margins, uncertainty = {}, {}
    for category in sorted(categories):
        strata = [row for row in summaries.values() if row.get("category") == category]
        if not strata:
            raise ValueError(f"calibration report has no uncertainty summary for {category}")
        # Conservative across the frozen target strata. The operational threshold
        # is declared by the operator; calibration variance only widens it.
        bounds = []
        for row in strata:
            n = row.get("seed_count")
            reps = row.get("repetition_count")
            between = row.get("between_seed_variance")
            within = row.get("mean_within_seed_repeat_variance")
            if (type(n) is not int or n <= 0 or type(reps) is not int or reps < 2
                    or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                           for x in (between, within))):
                raise ValueError("calibration report variance summary is incomplete or invalid")
            bounds.append(math.sqrt(between / n + within / (n * reps)))
        se = max(bounds)
        threshold = float(declaration["operational_gain_threshold"][category])
        margins[category] = threshold + float(multiplier) * se
        uncertainty[category] = {"maximum_stratum_standard_error": se,
            "stratum_count": len(strata)}
    declaration_digest = hashlib.sha256(json.dumps(declaration, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    provenance = {"schema_version": 1,
        "method": "declared_operational_threshold_plus_multiplier_times_max_stratum_standard_error",
        "standard_error": "sqrt(between_seed_variance/seed_count + mean_within_seed_repeat_variance/(seed_count*repetition_count))",
        "operational_gain_threshold": declaration["operational_gain_threshold"],
        "uncertainty_multiplier": float(multiplier), "uncertainty_by_category": uncertainty,
        "declaration_sha256": declaration_digest,
        "calibration_id": target_report["calibration_id"]}
    return margins, provenance


def run_d0_calibration(*, scenario_id: str, episodes: int, repetitions: int,
                       master_seed: int, device: str) -> dict:
    """Collect repeated D0 outcomes on a dedicated, disjoint seed partition."""
    device_identity = _require_calibration_gpu(device)
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("D0 calibration requires at least two repeated runs")
    if type(episodes) is not int or episodes < MIN_EPISODES:
        raise ValueError(f"D0 calibration requires at least {MIN_EPISODES} seeds per category/stratum")
    from .meco_track_training import evaluate_meco
    cases = protocol_cases(scenario_id, "calibration", episodes=episodes,
                           master_seed=master_seed)
    calibration_seeds = {seed for case in cases for seed in case.seeds}
    if any(seed % 4 != 3 for seed in calibration_seeds):
        raise ValueError("calibration seed partition is not isolated")
    for split in ("train", "validation", "test"):
        other = {seed for case in protocol_cases(scenario_id, split,
            episodes=episodes, master_seed=master_seed) for seed in case.seeds}
        if calibration_seeds & other:
            raise ValueError(f"calibration seeds overlap the {split} partition")
    records = []
    source_hash = None
    safety_source_hash = None
    safety_schema = None
    manifest = None
    # Repetitions intentionally reuse each calibration seed. This estimates
    # repeat-run variation separately from variation across seed means.
    for repetition in range(repetitions):
        for case in cases:
            result = evaluate_meco({}, scenario_id=scenario_id,
                categories=(case.category,), episodes=len(case.seeds),
                master_seed=case.seeds[0], episode_seeds=case.seeds,
                device=device, split="calibration", evaluation_policies=(),
                control_policies=(REFERENCE_POLICY,), include_controls=True,
                qualification_case={"layout_id": case.layout_id,
                    "defender_family": case.defender_family,
                    "start_condition": case.start_condition,
                    "sensor_profile": case.sensor_profile})
            if manifest is None:
                manifest = result["policy_manifest"]
                source_hash = result["source_identity_sha256"]
                safety_schema = result.get("safety_metrics_schema")
                safety_source_hash = result.get("safety_metrics_source_identity_sha256")
            elif (manifest != result["policy_manifest"]
                    or source_hash != result["source_identity_sha256"]
                    or safety_schema != result.get("safety_metrics_schema")
                    or safety_source_hash != result.get("safety_metrics_source_identity_sha256")):
                raise ValueError("D0 source identity changed during calibration")
            for row in result["records"]:
                records.append({"repetition": repetition, **row,
                    "calibration_runtime_device": device_identity})
    summaries = {}
    for case in cases:
        key = (case.category, case.layout_id, case.defender_family,
               case.start_condition, case.sensor_profile)
        by_seed = {}
        for row in records:
            if (row["category"], row["layout_id"], row["defender_family"],
                row["start_condition"], row["sensor_profile"]) == key:
                by_seed.setdefault(row["seed"], []).append(
                    float(row["metrics"]["useful_throughput"]))
        expected = set(case.seeds)
        if set(by_seed) != expected or any(len(values) != repetitions for values in by_seed.values()):
            raise ValueError("calibration report has incomplete repeated seed coverage")
        seed_means = [sum(by_seed[seed]) / repetitions for seed in case.seeds]
        repeat_variances = [(_variance(values) if len(values) > 1 else 0.)
                            for values in by_seed.values()]
        summaries["/".join(key)] = {"category": case.category,
            "layout_id": case.layout_id, "defender_family": case.defender_family,
            "start_condition": case.start_condition, "sensor_profile": case.sensor_profile,
            "seed_count": len(case.seeds), "repetition_count": repetitions,
            "mean_useful_throughput": sum(seed_means) / len(seed_means),
            "between_seed_variance": _variance(seed_means),
            "mean_within_seed_repeat_variance": sum(repeat_variances) / len(repeat_variances),
            "seed_digest": hashlib.sha256(json.dumps(list(case.seeds),
                separators=(",", ":")).encode()).hexdigest()}
    body = {"schema_version": 2, "calibration": True,
        "protocol_version": D0_CALIBRATION_VERSION, "scenario_id": scenario_id,
        "master_seed": master_seed, "split": "calibration",
        "seed_partition_code": 3, "episodes_per_category_stratum": episodes,
        "repetitions": repetitions, "repeat_semantics": "same exact calibration seeds repeated independently",
        "case_ids": [{"category": c.category, "layout_id": c.layout_id,
            "defender_family": c.defender_family, "start_condition": c.start_condition,
            "sensor_profile": c.sensor_profile} for c in cases],
        "policy_manifest": manifest, "source_identity_sha256": source_hash,
        "runtime_device": device_identity,
        "normalized_runtime_identity": _normalized_calibration_runtime(device_identity),
        "safety_metrics_schema": safety_schema,
        "safety_metrics_source_identity_sha256": safety_source_hash,
        "summary": summaries, "records": records,
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()}
    body["calibration_id"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return body


def _matched_reference_variance_summary(cases, repetitions, records):
    metrics = ("useful_throughput", *sorted(SAFETY_METRIC_NAMES))
    summary = {}
    for case in cases:
        key_fields = ("category", "layout_id", "defender_family",
                      "start_condition", "sensor_profile")
        key = tuple(case[name] for name in key_fields)
        for policy in (REFERENCE_POLICY, "D1", "D2"):
            if policy not in (REFERENCE_POLICY, case["reference_policy"]):
                continue
            by_seed = {}
            for row in records:
                if (row["policy"] == policy
                        and tuple(row[name] for name in key_fields) == key):
                    by_seed.setdefault(row["seed"], []).append(row["metrics"])
            if set(by_seed) != set(case["seeds"]) or any(
                    len(values) != repetitions for values in by_seed.values()):
                raise ValueError("matched reference calibration repeated seed coverage is incomplete")
            for metric in metrics:
                values_by_seed = []
                for seed in case["seeds"]:
                    values = [row.get(metric) for row in by_seed[seed]]
                    if any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in values):
                        raise ValueError(f"matched reference calibration is missing {metric} records")
                    values_by_seed.append([float(value) for value in values])
                seed_means = [sum(values) / repetitions for values in values_by_seed]
                repeat_variances = [_variance(values) for values in values_by_seed]
                summary["/".join((policy, metric, *key))] = {
                    "policy": policy, "metric": metric,
                    **dict(zip(key_fields, key)),
                    "seed_count": len(case["seeds"]),
                    "repetition_count": repetitions,
                    "mean": sum(seed_means) / len(seed_means),
                    "between_seed_variance": _variance(seed_means),
                    "mean_within_seed_repeat_variance": sum(repeat_variances) / len(repeat_variances),
                    "seed_digest": hashlib.sha256(json.dumps(list(case["seeds"]),
                        separators=(",", ":")).encode()).hexdigest()}
    return summary


def run_matched_reference_variance_calibration(*, scenario_id: str, episodes: int,
        repetitions: int, master_seed: int, reference_policy: str, device: str) -> dict:
    """Measure repeat and seed variance for one frozen learned reference plus D0 safety floor."""
    variant = _reference_variant(reference_policy)
    policy, policy_id, mode = MATCHED_REFERENCE_VARIANTS[variant]
    runtime = _require_calibration_gpu(device)
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("matched reference calibration requires at least two repetitions")
    if type(episodes) is not int or episodes < MIN_EPISODES:
        raise ValueError(f"matched reference calibration requires at least {MIN_EPISODES} episodes per stratum")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("calibration master seed must be a nonnegative integer")
    from .meco_track_training import evaluate_meco
    cases = protocol_cases(scenario_id, "calibration", episodes=episodes,
                           master_seed=master_seed)
    case_table = [{"category": case.category, "layout_id": case.layout_id,
        "defender_family": case.defender_family, "start_condition": case.start_condition,
        "sensor_profile": case.sensor_profile, "seeds": list(case.seeds),
        "reference_policy": policy} for case in cases]
    _validate_d0_calibration_case_table(scenario_id, episodes, master_seed,
        [{key: value for key, value in row.items() if key != "reference_policy"}
         for row in case_table])
    source_hash = _calibration_source_identity_sha256()
    identity = _matched_reference_identity(variant)
    candidate_configuration = {"candidate_k": MATCHED_REFERENCE_CANDIDATE_K,
        "route_cost_points_per_second": ROUTE_COST_POINTS_PER_SECOND,
        "action_ticks": 12}
    config = {"scenario_id": scenario_id, "split": "calibration",
        "episodes_per_category_stratum": episodes, "repetitions": repetitions,
        "master_seed": master_seed, "seed_partition_code": 3,
        "repeat_semantics": "same exact calibration seeds repeated independently",
        "reference_selection_id": variant, "reference_policy": policy,
        "reference_policy_id": policy_id, "prediction_mode": mode,
        "prediction_contract": identity["prediction_contract"],
        "candidate_configuration": candidate_configuration,
        "safety_floor_policy": REFERENCE_POLICY,
        "required_safety_metrics": sorted(SAFETY_METRIC_NAMES)}
    protocol_identity = _canonical_sha256({"version": MATCHED_REFERENCE_VARIANCE_VERSION,
        "config": config, "case_table": case_table, "source_identity_sha256": source_hash,
        "runtime_identity": _normalized_calibration_runtime(runtime)})
    control_modes = {policy: mode}
    records, policy_manifest = [], None
    for repetition in range(repetitions):
        for case in cases:
            case_binding = {"scenario_id": scenario_id,
                "master_seed": master_seed, "episodes": len(case.seeds),
                "category": case.category,
                "qualification_case": {"layout_id": case.layout_id,
                    "defender_family": case.defender_family,
                    "start_condition": case.start_condition,
                    "sensor_profile": case.sensor_profile},
                "episode_seeds": list(case.seeds)}
            result = evaluate_meco({}, scenario_id=scenario_id,
                categories=(case.category,), episodes=len(case.seeds),
                master_seed=master_seed, episode_seeds=case.seeds,
                device=device, split="calibration", evaluation_policies=(),
                control_policies=(REFERENCE_POLICY, policy), include_controls=True,
                control_prediction_modes=control_modes,
                calibration_control_policy=policy,
                candidate_k_override=MATCHED_REFERENCE_CANDIDATE_K,
                calibration_route_cost_points_per_second=ROUTE_COST_POINTS_PER_SECOND,
                qualification_case=case_binding["qualification_case"])
            if result.get("calibration_control_case_binding") != case_binding:
                raise ValueError("matched reference evaluator did not bind the frozen calibration case and seed tuple")
            if (result.get("source_identity_sha256") != source_hash
                    or result.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
                    or result.get("safety_metrics_source_identity_sha256") != source_hash):
                raise ValueError("matched reference source/safety provenance drifted during calibration")
            expected_execution_device = {"device": runtime["device"],
                "accelerator_backend": "ROCm" if runtime["backend"] == "hip" else "CUDA"}
            if result.get("execution_device") != expected_execution_device:
                raise ValueError("matched reference execution device/runtime identity drifted")
            current_manifest = result.get("policy_manifest")
            if (not isinstance(current_manifest, dict)
                    or current_manifest.get(REFERENCE_POLICY, {}).get("policy_id")
                        != "D0_production_deterministic"
                    or any(current_manifest.get(name, {}).get(field) != expected
                        for name, field, expected in ((policy, "policy_id", policy_id),
                            (policy, "prediction_mode", mode),
                            (policy, "prediction_contract", identity["prediction_contract"])) )):
                raise ValueError("matched reference policy or prediction contract differs from the frozen selection")
            if current_manifest.get(policy, {}).get("calibration_candidate_configuration") != candidate_configuration:
                raise ValueError("matched reference candidate configuration differs from the frozen selection")
            if policy_manifest is not None and policy_manifest != current_manifest:
                raise ValueError("matched reference policy manifest changed during calibration")
            policy_manifest = current_manifest
            eval_rows = result.get("records", [])
            by_key = {(row.get("policy"), row.get("seed")): row for row in eval_rows}
            expected = {(name, seed) for name in (REFERENCE_POLICY, policy)
                        for seed in case.seeds}
            if set(by_key) != expected or len(by_key) != len(eval_rows):
                raise ValueError("matched reference/D0 safety-floor records have incomplete seed coverage")
            for name, seed in sorted(expected):
                row = by_key[(name, seed)]
                metrics = row.get("metrics")
                if (not isinstance(metrics, dict)
                        or {key for key in metrics if isinstance(key, str) and key.startswith("safety_")}
                            != SAFETY_METRIC_NAMES
                        or any(type(metrics.get(metric)) not in (int, float)
                            or not math.isfinite(metrics[metric]) or not 0 <= metrics[metric] <= 1
                            for metric in SAFETY_METRIC_NAMES)
                        or type(metrics.get("useful_throughput")) not in (int, float)
                        or not math.isfinite(metrics["useful_throughput"])):
                    raise ValueError("matched reference calibration row lacks throughput or required safety metrics")
                records.append({"repetition": repetition,
                    "calibration_control_case_binding": case_binding, **row,
                    "calibration_runtime_device": runtime})
    body = {"schema_version": 1, "calibration": True,
        "artifact_type": MATCHED_REFERENCE_VARIANCE_VERSION,
        "scenario_id": scenario_id, "config": config,
        "protocol_identity_sha256": protocol_identity,
        "case_ids": case_table, "policy_manifest": policy_manifest,
        "source_identity_sha256": source_hash, "runtime_device": runtime,
        "normalized_runtime_identity": _normalized_calibration_runtime(runtime),
        "safety_metrics_schema": SAFETY_METRICS_SCHEMA,
        "safety_metrics_source_identity_sha256": source_hash,
        "summary": _matched_reference_variance_summary(case_table, repetitions, records),
        "records": records,
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()}
    body["calibration_id"] = _canonical_sha256(body)
    return body


# Versioned extension used by the resumable paired D0/D1/D2 campaign. The
# historical D0-only and single-reference artifacts above remain readable.
PAIRED_CALIBRATION_VERSION = "meco-paired-d0-d2-calibration-v2"
PAIRED_CALIBRATION_CAMPAIGN = "meco-paired-calibration-campaign-v2"
PAIRED_CALIBRATION_SHARD = "meco-paired-calibration-shard-manifest-v2"
PAIRED_CALIBRATION_ARTIFACT = "meco-paired-calibration-shard-v2"
PAIRED_CALIBRATION_PROGRESS = "meco-paired-calibration-shard-progress-v2"
PAIRED_CALIBRATION_FINAL_SHARD = "meco-paired-calibration-shard-final-v2"
PAIRED_CALIBRATION_REPLAY_CONTRACT = "dashboard-replay-generation-v1"
PAIRED_CALIBRATION_TUPLE_FIELDS = (
    "scenario_id", "case_id", "category", "defender_family", "layout_id",
    "start_condition", "sensor_profile", "seed", "repetition", "evaluation_phase")
PAIRED_CALIBRATION_PHASE0_DIGEST_FIELDS = frozenset({
    "capture_receipt_sha256", "baseline_freeze_sha256", "partition_sha256",
    "tuple_registry_sha256", "assignment_table_sha256"})
PAIRED_CALIBRATION_CONTROLS = (
    "D0", "D1/current", "D2/cv", "D2/multi_model_v1",
)
_PAIRED_CALIBRATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")


def _paired_artifact_binding(value, *, kind):
    fields = {"path", "artifact_sha256", "schema"}
    identity_field = {"baseline freeze": "freeze_sha256",
        "partition": "partition_sha256", "estimand approval": "approval_sha256"}.get(kind)
    if identity_field is None:
        raise ValueError("unknown paired calibration gate artifact kind")
    fields.add(identity_field)
    if kind == "partition":
        fields.add("baseline_freeze_sha256")
    if kind == "baseline freeze":
        fields.update({"source_commit", "source_tree", "dependency_lock_sha256"})
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"paired calibration {kind} binding is incomplete")
    path = value.get("path")
    if (not isinstance(path, str) or not _PAIRED_CALIBRATION_ID.fullmatch(path)
            or path.startswith("/") or ".." in Path(path).parts
            or "\\" in path):
        raise ValueError(f"paired calibration {kind} artifact path is unsafe")
    digest_fields = fields.intersection({"artifact_sha256", identity_field,
        "baseline_freeze_sha256", "dependency_lock_sha256"})
    for key in digest_fields:
        if not isinstance(value.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise ValueError(f"paired calibration {kind} artifact digest is invalid")
    for key in fields.intersection({"source_commit", "source_tree"}):
        if not isinstance(value.get(key), str) or re.fullmatch(r"[0-9a-f]{40}", value[key]) is None:
            raise ValueError(f"paired calibration {kind} source identity is invalid")
    if not isinstance(value.get("schema"), str) or not value["schema"]:
        raise ValueError(f"paired calibration {kind} artifact schema is missing")
    return dict(value)


def _paired_worker_assignments(assignments):
    if not isinstance(assignments, list) or len(assignments) != 2:
        raise ValueError("paired calibration requires exactly two explicit WX9100 workers")
    normalized = []
    for assignment in assignments:
        required = {"worker_id", "device", "device_index", "device_model",
                    "device_uuid", "backend", "backend_version", "torch_version"}
        if not isinstance(assignment, dict) or set(assignment) != required:
            raise ValueError("paired calibration worker identity is incomplete")
        if (not isinstance(assignment["worker_id"], str)
                or not assignment["worker_id"].strip()
                or type(assignment["device_index"]) is not int
                or assignment["device_index"] < 0
                or assignment["device"] != f"cuda:{assignment['device_index']}"
                or not isinstance(assignment["device_model"], str)
                or "wx9100" not in re.sub(r"[^a-z0-9]", "",
                    assignment["device_model"].lower())
                or not isinstance(assignment["device_uuid"], str)
                or not assignment["device_uuid"].strip()
                or assignment["backend"] != "hip"
                or any(not isinstance(assignment[key], str) or not assignment[key].strip()
                       for key in ("backend_version", "torch_version"))):
            raise ValueError("paired calibration worker must bind an indexed WX9100 HIP device")
        normalized.append(dict(assignment))
    if (len({row["worker_id"] for row in normalized}) != 2
            or len({row["device_index"] for row in normalized}) != 2
            or len({row["device_uuid"] for row in normalized}) != 2):
        raise ValueError("paired calibration workers must identify two distinct devices")
    if len({(row["backend"], row["backend_version"], row["torch_version"])
            for row in normalized}) != 1:
        raise ValueError("paired calibration workers have mixed runtime identities")
    return sorted(normalized, key=lambda row: row["device_index"])


def _paired_control_specs():
    from .meco_prediction_contract import prediction_contract

    candidate = {"candidate_k": MATCHED_REFERENCE_CANDIDATE_K,
        "route_cost_points_per_second": ROUTE_COST_POINTS_PER_SECOND,
        "action_ticks": 12}
    rows = [{"control_id": "D0", "policy": "D0",
        "policy_id": "D0_production_deterministic", "prediction_mode": None,
        "prediction_contract": None, "prediction_contract_sha256": None,
        "candidate_configuration": candidate,
        "execution_supported": True, "support_limit": None}]
    for variant in PAIRED_CALIBRATION_CONTROLS[1:]:
        identity = _matched_reference_identity(variant)
        body = {"control_id": variant, "policy": identity["policy"],
            "policy_id": identity["policy_id"],
            "prediction_mode": identity["prediction_mode"],
            "prediction_contract": identity["prediction_contract"],
            "prediction_contract_sha256": _canonical_sha256(identity["prediction_contract"]),
            "candidate_configuration": candidate, "execution_supported": True,
            "support_limit": None}
        body["control_binding_sha256"] = _canonical_sha256(body)
        rows.append(body)
    rows[0]["control_binding_sha256"] = _canonical_sha256(rows[0])
    return rows


def _paired_shard_control_bindings(controls):
    return {control["control_id"]: control["control_binding_sha256"]
        for control in controls}


def _paired_control_contracts_sha256(controls):
    return _canonical_sha256(controls)


def _validate_effective_phase0_bundle(bundle, repo_root):
    """Narrow dependency boundary for the separately reviewed effective-freeze API."""
    try:
        from .meco_baseline_freeze import validate_effective_phase0_bundle
    except ImportError as exc:
        raise RuntimeError("effective Phase 0 bundle validator is not integrated") from exc
    digests = validate_effective_phase0_bundle(bundle, repo_root)
    if not isinstance(digests, dict) or set(digests) != PAIRED_CALIBRATION_PHASE0_DIGEST_FIELDS:
        raise ValueError("effective Phase 0 bundle validator returned an incomplete digest map")
    for name, digest in digests.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"effective Phase 0 digest is malformed: {name}")
    return digests


def _validate_effective_phase0_configuration_bytes(bundle, configuration_artifacts):
    """Match caller-loaded bytes to the configuration snapshot authenticated by v3."""
    import base64

    receipt = bundle.get("capture_receipt") if isinstance(bundle, dict) else None
    artifacts = receipt.get("configuration_artifacts") if isinstance(receipt, dict) else None
    if not isinstance(artifacts, list) or not isinstance(configuration_artifacts, dict):
        raise ValueError("effective Phase 0 configuration artifact bytes are missing")
    expected = {}
    for artifact in artifacts:
        if (not isinstance(artifact, dict) or set(artifact) != {
                "artifact_id", "content_sha256", "content_base64"}
                or not isinstance(artifact["artifact_id"], str)):
            raise ValueError("effective Phase 0 configuration artifact row is malformed")
        try:
            content = base64.b64decode(artifact["content_base64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("effective Phase 0 configuration artifact bytes are malformed") from exc
        if hashlib.sha256(content).hexdigest() != artifact["content_sha256"]:
            raise ValueError("effective Phase 0 configuration artifact digest changed")
        expected[artifact["artifact_id"]] = content
    if len(expected) != len(artifacts) or configuration_artifacts != expected:
        raise ValueError("effective Phase 0 configuration bytes differ from caller artifacts")
    return expected


def _paired_calibration_registered_tuples(bundle):
    """Select only eligible tuple IDs assigned to calibration by the closed bundle."""
    if not isinstance(bundle, dict):
        raise ValueError("effective Phase 0 bundle is malformed")
    materialization = bundle.get("partition_materialization")
    if not isinstance(materialization, dict):
        raise ValueError("effective Phase 0 tuple partition materialization is missing")
    partitions = materialization.get("partitions")
    registry = materialization.get("tuple_registry")
    assignment_table = materialization.get("assignment_table")
    if (not isinstance(partitions, dict) or "calibration" not in partitions
            or not isinstance(registry, dict) or not isinstance(registry.get("tuples"), list)
            or not isinstance(assignment_table, dict)
            or not isinstance(assignment_table.get("assignments"), dict)):
        raise ValueError("effective Phase 0 calibration tuple registry is malformed")
    calibration_ids = partitions["calibration"].get("tuple_ids")
    assignments = assignment_table["assignments"]
    if (not isinstance(calibration_ids, list) or not calibration_ids
            or len(calibration_ids) != len(set(calibration_ids))):
        raise ValueError("effective Phase 0 calibration tuple IDs are empty or duplicated")
    by_id = {}
    for row in registry["tuples"]:
        if not isinstance(row, dict):
            raise ValueError("effective Phase 0 tuple registry row is malformed")
        fields = {key: row.get(key) for key in PAIRED_CALIBRATION_TUPLE_FIELDS}
        tuple_id = row.get("tuple_id")
        if (not isinstance(tuple_id, str) or tuple_id != _canonical_sha256(fields)
                or tuple_id in by_id or type(row.get("eligible")) is not bool):
            raise ValueError("effective Phase 0 tuple ID does not bind its registered tuple")
        by_id[tuple_id] = row
    selected = []
    for tuple_id in calibration_ids:
        row = by_id.get(tuple_id)
        if (row is None or row.get("eligible") is not True
                or assignments.get(tuple_id) != "calibration"
                or row.get("evaluation_phase") != "calibration"):
            raise ValueError("paired schedule contains a non-calibration or excluded tuple")
        selected.append({key: row[key] for key in PAIRED_CALIBRATION_TUPLE_FIELDS}
            | {"tuple_id": tuple_id})
    if set(calibration_ids) != {tuple_id for tuple_id, row in by_id.items()
            if row.get("eligible") is True and assignments.get(tuple_id) == "calibration"}:
        raise ValueError("paired schedule omits an eligible registered calibration tuple")
    return sorted(selected, key=lambda row: row["tuple_id"])


def _paired_phase0_bundle_binding(bundle, digest_map):
    receipt = bundle.get("capture_receipt") if isinstance(bundle, dict) else None
    freeze = bundle.get("baseline_freeze") if isinstance(bundle, dict) else None
    if (not isinstance(receipt, dict) or not isinstance(receipt.get("source"), dict)
            or not isinstance(receipt.get("runtime"), dict)
            or not isinstance(freeze, dict) or not isinstance(freeze.get("source"), dict)):
        raise ValueError("effective Phase 0 bundle lacks source/runtime-bound receipt and freeze")
    source = receipt["source"]
    runtime = receipt["runtime"]
    if (freeze.get("source") != source
            or freeze.get("runtime") != runtime
            or not isinstance(source.get("commit"), str)
            or not isinstance(source.get("tree"), str)
            or not isinstance(runtime.get("dependency_lock_sha256"), str)):
        raise ValueError("effective Phase 0 bundle source/runtime identities disagree")
    binding = {**digest_map,
        "bundle_sha256": _canonical_sha256(bundle),
        "source_commit": source["commit"], "source_tree": source["tree"],
        "dependency_lock_sha256": runtime["dependency_lock_sha256"]}
    return binding


def _validate_paired_phase0_binding(binding):
    expected = set(PAIRED_CALIBRATION_PHASE0_DIGEST_FIELDS) | {
        "bundle_sha256", "source_commit", "source_tree", "dependency_lock_sha256"}
    if not isinstance(binding, dict) or set(binding) != expected:
        raise ValueError("paired calibration effective Phase 0 binding is incomplete")
    for key in PAIRED_CALIBRATION_PHASE0_DIGEST_FIELDS | {
            "bundle_sha256", "dependency_lock_sha256"}:
        if not isinstance(binding.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", binding[key]) is None:
            raise ValueError(f"paired calibration Phase 0 digest is invalid: {key}")
    for key in ("source_commit", "source_tree"):
        if not isinstance(binding.get(key), str) or re.fullmatch(r"[0-9a-f]{40}", binding[key]) is None:
            raise ValueError(f"paired calibration Phase 0 source {key} is invalid")
    return dict(binding)


def _paired_calibration_work_units(registered_tuples):
    """Expand only preregistered calibration tuple IDs across matched controls."""
    if not isinstance(registered_tuples, list) or not registered_tuples:
        raise ValueError("paired calibration requires registered calibration tuples")
    units = []
    for item in registered_tuples:
        if (not isinstance(item, dict) or set(item) !=
                set(PAIRED_CALIBRATION_TUPLE_FIELDS) | {"tuple_id"}
                or item.get("evaluation_phase") != "calibration"
                or item.get("tuple_id") != _canonical_sha256({key: item[key]
                    for key in PAIRED_CALIBRATION_TUPLE_FIELDS})):
            raise ValueError("paired calibration tuple is not bound to the registered ID")
        pair = {key: item[key] for key in PAIRED_CALIBRATION_TUPLE_FIELDS}
        pair["split"] = "calibration"
        for control_id in PAIRED_CALIBRATION_CONTROLS[1:]:
            pair_id = _canonical_sha256({"tuple_id": item["tuple_id"],
                "paired_control_id": control_id})
            for policy_control_id in ("D0", control_id):
                units.append({**pair, "tuple_id": item["tuple_id"],
                    "control_id": policy_control_id,
                    "paired_control_id": control_id, "pair_id": pair_id})
    return sorted(units, key=lambda unit: (unit["tuple_id"], unit["control_id"]))


def _paired_source_binding(value):
    fields = {"commit", "tree", "source_identity_sha256",
        "source_tree_identity_sha256", "dependency_lock_sha256"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("paired calibration source/runtime binding is incomplete")
    for key in ("commit", "tree"):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{40}", value[key]) is None:
            raise ValueError(f"paired calibration source {key} is invalid")
    for key in fields - {"commit", "tree"}:
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise ValueError(f"paired calibration source/runtime digest {key} is invalid")
    return dict(value)


def _scenario_registry_binding():
    snapshot = [asdict(definition) for definition in SCENARIOS.values()]
    return {"scenario_ids": [item["scenario_id"] for item in snapshot],
        "registry": snapshot, "registry_sha256": _canonical_sha256(snapshot)}


def _paired_shard_assignments(units, shard_count, campaign_identity):
    if type(shard_count) is not int or shard_count < 2:
        raise ValueError("paired calibration needs at least two shards")
    if shard_count > len(units):
        raise ValueError("paired calibration shard count exceeds work-unit count")
    pair_groups = {}
    for unit in units:
        pair_groups.setdefault(unit["pair_id"], []).append(unit)
    if any({unit["control_id"] for unit in group} != {"D0", group[0]["paired_control_id"]}
            or len(group) != 2 for group in pair_groups.values()):
        raise ValueError("paired calibration schedule has an incomplete control pair")
    evaluation_groups = {}
    case_fields = ("scenario_id", "case_id", "category", "layout_id",
        "defender_family", "start_condition", "sensor_profile", "repetition",
        "evaluation_phase", "paired_control_id")
    for pair_group in pair_groups.values():
        representative = pair_group[0]
        evaluation_key = _canonical_sha256({field: representative[field]
            for field in case_fields})
        evaluation_groups.setdefault(evaluation_key, []).extend(pair_group)
    if shard_count > len(evaluation_groups):
        raise ValueError("paired calibration shard count exceeds complete evaluation-group count")
    ranked = sorted(evaluation_groups.items(), key=lambda item: _canonical_sha256({
        "campaign": campaign_identity, "evaluation_group": item[0]}))
    buckets = [[] for _ in range(shard_count)]
    for index, (_, group) in enumerate(ranked):
        buckets[index % shard_count].extend(sorted(group,
            key=lambda unit: (unit["seed"], unit["tuple_id"], unit["control_id"])))
    return buckets


def build_paired_calibration_campaign(*, shard_count, source_binding,
        phase0_binding, calibration_tuples, evaluation_budget,
        estimand_approval_binding, worker_assignments):
    """Build an immutable schedule from the pre-registered Phase 0 tuple set."""
    source = _paired_source_binding(source_binding)
    phase0 = _validate_paired_phase0_binding(phase0_binding)
    approval = _paired_artifact_binding(estimand_approval_binding, kind="estimand approval")
    if approval["schema"] != "meco-simulated-score-estimand-review-v1":
        raise ValueError("paired calibration gate artifact schema is unsupported")
    if any(source[key] != phase0[phase_key] for key, phase_key in (
            ("commit", "source_commit"), ("tree", "source_tree"),
            ("dependency_lock_sha256", "dependency_lock_sha256"))):
        raise ValueError("paired calibration source/runtime binding differs from baseline freeze")
    if type(shard_count) is not int or shard_count < 2:
        raise ValueError("paired calibration needs at least two shards")
    if (not isinstance(evaluation_budget, dict)
            or set(evaluation_budget) != {"episodes_per_tuple", "repetitions"}
            or any(type(value) is not int or value < 1
                for value in evaluation_budget.values())):
        raise ValueError("paired calibration budget must come from the effective tuple partition")
    worker_rows = _paired_worker_assignments(worker_assignments)
    from .meco_simulated_score_ledger import (EVENT_STREAM_SCHEMA, LEDGER_SCHEMA,
        RULES_ID, RULES_SHA256, OUTCOME_NAME, RULES)
    scoring = {"ledger_schema": LEDGER_SCHEMA, "rules_id": RULES_ID,
        "rules_sha256": RULES_SHA256, "rules": RULES,
        "event_stream_schema": EVENT_STREAM_SCHEMA,
        "replay_contract_schema": PAIRED_CALIBRATION_REPLAY_CONTRACT,
        "estimand_approval_sha256": approval["approval_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "outcome_name": OUTCOME_NAME, "unit": RULES["unit"],
        "official_match_points": None}
    controls = _paired_control_specs()
    tuple_rows = sorted(calibration_tuples, key=lambda row: row.get("tuple_id", ""))
    units = _paired_calibration_work_units(tuple_rows)
    if shard_count < len(worker_rows):
        raise ValueError("each paired calibration worker must receive a shard")
    config = {"split": "calibration", "evaluation_budget": evaluation_budget,
        "shard_count": shard_count, "control_ids": list(PAIRED_CALIBRATION_CONTROLS),
        "repeat_semantics": "exact registered tuple ID, seed, and repetition",
        "summary_method": "paired_seed_cluster_bootstrap_v1",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES}
    schedule_sha = _canonical_sha256(units)
    registry = _scenario_registry_binding()
    body = {"schema_version": 2, "campaign_type": PAIRED_CALIBRATION_VERSION,
        "artifact_type": PAIRED_CALIBRATION_CAMPAIGN, "config": config,
        "source": source, "scenario_registry": registry,
        "phase0_binding": phase0, "calibration_tuples": tuple_rows,
        "estimand_approval": approval, "scoring": scoring,
        "controls": controls, "worker_assignments": worker_rows,
        "work_unit_count": len(units), "schedule_sha256": schedule_sha}
    campaign_identity = _canonical_sha256(body)
    buckets = _paired_shard_assignments(units, shard_count, campaign_identity)
    shards = []
    for index, assigned in enumerate(buckets):
        worker = worker_rows[index % len(worker_rows)]
        phase0_binding_sha256 = _canonical_sha256(phase0)
        shard_body = {"schema_version": 2,
            "artifact_type": PAIRED_CALIBRATION_SHARD,
            "campaign_id": campaign_identity, "shard_index": index,
            "shard_count": shard_count, "worker_id": worker["worker_id"],
            "phase0_binding_sha256": phase0_binding_sha256,
            "tuple_registry_sha256": phase0["tuple_registry_sha256"],
            "assignment_table_sha256": phase0["assignment_table_sha256"],
            "tuple_ids": sorted({unit["tuple_id"] for unit in assigned}),
            "worker_assignment_sha256": _canonical_sha256(worker),
            "control_binding_sha256": _canonical_sha256(
                _paired_shard_control_bindings(controls)),
            "control_bindings": _paired_shard_control_bindings(controls),
            "controls": controls,
            "controls_sha256": _paired_control_contracts_sha256(controls),
            "work_units": assigned}
        shards.append({**shard_body, "manifest_id": _canonical_sha256(shard_body)})
    campaign = {**body, "campaign_id": campaign_identity, "shards": shards}
    return validate_paired_calibration_campaign(campaign)


def validate_paired_calibration_campaign(campaign):
    if (not isinstance(campaign, dict)
            or campaign.get("artifact_type") != PAIRED_CALIBRATION_CAMPAIGN
            or campaign.get("campaign_type") != PAIRED_CALIBRATION_VERSION
            or campaign.get("schema_version") != 2):
        raise ValueError("unsupported paired calibration campaign schema")
    config = campaign.get("config")
    expected_config_keys = {"split", "evaluation_budget", "shard_count",
        "control_ids", "repeat_semantics",
        "summary_method", "bootstrap_replicates"}
    if (not isinstance(config, dict) or set(config) != expected_config_keys
            or config.get("split") != "calibration"
            or config.get("control_ids") != list(PAIRED_CALIBRATION_CONTROLS)
            or config.get("repeat_semantics") != "exact registered tuple ID, seed, and repetition"
            or config.get("summary_method") != "paired_seed_cluster_bootstrap_v1"
            or config.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES):
        raise ValueError("paired calibration campaign config is invalid")
    budget = config.get("evaluation_budget")
    if (not isinstance(budget, dict) or set(budget) != {"episodes_per_tuple", "repetitions"}
            or any(type(value) is not int or value < 1 for value in budget.values())):
        raise ValueError("paired calibration evaluation budget is invalid")
    tuples = campaign.get("calibration_tuples")
    if (not isinstance(tuples, list) or not tuples
            or tuples != sorted(tuples, key=lambda row: row.get("tuple_id", ""))
            or len({row.get("tuple_id") for row in tuples if isinstance(row, dict)}) != len(tuples)):
        raise ValueError("paired calibration registered tuple set is empty, duplicated, or unordered")
    for row in tuples:
        if (not isinstance(row, dict) or set(row) !=
                set(PAIRED_CALIBRATION_TUPLE_FIELDS) | {"tuple_id"}
                or row.get("evaluation_phase") != "calibration"
                or row.get("tuple_id") != _canonical_sha256({key: row[key]
                    for key in PAIRED_CALIBRATION_TUPLE_FIELDS})):
            raise ValueError("paired calibration campaign contains an invalid registered tuple")
    units = _paired_calibration_work_units(tuples)
    if (campaign.get("work_unit_count") != len(units)
            or campaign.get("schedule_sha256") != _canonical_sha256(units)):
        raise ValueError("paired calibration frozen schedule is incomplete or altered")
    if (config.get("control_ids") != list(PAIRED_CALIBRATION_CONTROLS)
            or campaign.get("controls") != _paired_control_specs()):
        raise ValueError("paired calibration control set or predictor identity changed")
    workers = _paired_worker_assignments(campaign.get("worker_assignments"))
    if workers != campaign["worker_assignments"]:
        raise ValueError("paired calibration worker assignments are not canonical")
    phase0 = _validate_paired_phase0_binding(campaign.get("phase0_binding"))
    approval = _paired_artifact_binding(campaign.get("estimand_approval"),
        kind="estimand approval")
    if approval["schema"] != "meco-simulated-score-estimand-review-v1":
        raise ValueError("paired calibration estimand approval schema changed")
    source = _paired_source_binding(campaign.get("source"))
    if any(source[key] != phase0[phase_key] for key, phase_key in (
            ("commit", "source_commit"), ("tree", "source_tree"),
            ("dependency_lock_sha256", "dependency_lock_sha256"))):
        raise ValueError("paired calibration source/runtime binding differs from baseline freeze")
    registry = campaign.get("scenario_registry")
    if registry != _scenario_registry_binding():
        raise ValueError("paired calibration scenario registry identity changed")
    from .meco_simulated_score_ledger import (EVENT_STREAM_SCHEMA, LEDGER_SCHEMA,
        RULES_ID, RULES_SHA256, OUTCOME_NAME, RULES)
    expected_scoring = {"ledger_schema": LEDGER_SCHEMA, "rules_id": RULES_ID,
        "rules_sha256": RULES_SHA256, "rules": RULES,
        "event_stream_schema": EVENT_STREAM_SCHEMA,
        "replay_contract_schema": PAIRED_CALIBRATION_REPLAY_CONTRACT,
        "estimand_approval_sha256": campaign["estimand_approval"]["approval_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "outcome_name": OUTCOME_NAME, "unit": RULES["unit"],
        "official_match_points": None}
    if campaign.get("scoring") != expected_scoring:
        raise ValueError("paired calibration score/replay/approval contract changed")
    body = {key: value for key, value in campaign.items()
        if key not in ("campaign_id", "shards")}
    identity = _canonical_sha256(body)
    if campaign.get("campaign_id") != identity:
        raise ValueError("paired calibration campaign digest is invalid")
    shard_count = config.get("shard_count")
    if type(shard_count) is not int or shard_count < 2 or len(campaign.get("shards", [])) != shard_count:
        raise ValueError("paired calibration shard set is incomplete")
    expected_buckets = _paired_shard_assignments(units, shard_count, identity)
    seen = set()
    for index, (shard, expected_units) in enumerate(zip(campaign["shards"], expected_buckets)):
        expected_control_bindings = _paired_shard_control_bindings(campaign["controls"])
        if (not isinstance(shard, dict)
                or shard.get("campaign_id") != identity
                or shard.get("shard_index") != index
                or shard.get("shard_count") != shard_count
                or shard.get("schema_version") != 2
                or shard.get("phase0_binding_sha256") != _canonical_sha256(phase0)
                or shard.get("tuple_registry_sha256") != phase0["tuple_registry_sha256"]
                or shard.get("assignment_table_sha256") != phase0["assignment_table_sha256"]
                or shard.get("tuple_ids") != sorted({unit["tuple_id"] for unit in expected_units})
                or shard.get("work_units") != expected_units
                or shard.get("control_bindings") != expected_control_bindings
                or shard.get("control_binding_sha256")
                    != _canonical_sha256(expected_control_bindings)
                or shard.get("controls") != campaign["controls"]
                or shard.get("controls_sha256")
                    != _paired_control_contracts_sha256(campaign["controls"])):
            raise ValueError("paired calibration shard assignment is invalid")
        worker = workers[index % len(workers)]
        body = {key: value for key, value in shard.items() if key != "manifest_id"}
        if (shard.get("worker_id") != worker["worker_id"]
                or shard.get("worker_assignment_sha256") != _canonical_sha256(worker)
                or shard.get("manifest_id") != _canonical_sha256(body)):
            raise ValueError("paired calibration shard worker or manifest identity is invalid")
        for unit in shard["work_units"]:
            unit_id = _canonical_sha256(unit)
            if unit_id in seen:
                raise ValueError("paired calibration shards overlap")
            seen.add(unit_id)
    if seen != {_canonical_sha256(unit) for unit in units}:
        raise ValueError("paired calibration shards do not cover the frozen schedule")
    return campaign


def paired_calibration_execution_blockers(campaign):
    """List preregistered controls that currently lack an executable contract."""
    validate_paired_calibration_campaign(campaign)
    return [row["control_id"] for row in campaign["controls"]
        if row.get("execution_supported") is not True]


def _paired_shard_for_index(campaign, shard_index):
    validate_paired_calibration_campaign(campaign)
    if type(shard_index) is not int or not 0 <= shard_index < len(campaign["shards"]):
        raise ValueError("paired calibration shard index is invalid")
    return campaign["shards"][shard_index]


def paired_calibration_replay_binding(campaign, shard_index, work_unit):
    """Return the exact immutable context to pass through evaluate_meco into replay."""
    shard = _paired_shard_for_index(campaign, shard_index)
    if work_unit not in shard["work_units"]:
        raise ValueError("paired calibration binding work unit is outside its shard")
    control = next(item for item in campaign["controls"]
        if item["control_id"] == work_unit["control_id"])
    return {"campaign_id": campaign["campaign_id"],
        "shard_index": shard_index, "shard_manifest_id": shard["manifest_id"],
        "work_unit_id": _canonical_sha256(work_unit), "work_unit": work_unit,
        "pair_id": work_unit["pair_id"], "control_id": work_unit["control_id"],
        "source_binding": campaign["source"],
        "phase0_binding": campaign["phase0_binding"],
        "phase0_binding_sha256": _canonical_sha256(campaign["phase0_binding"]),
        "tuple_id": work_unit["tuple_id"],
        "estimand_approval_sha256": campaign["estimand_approval"]["approval_sha256"],
        "control_binding": control,
        "control_binding_sha256": control["control_binding_sha256"],
        "scoring": campaign["scoring"]}


def paired_calibration_condition_id(binding):
    """Stable unique replay condition for one comparator/pair/repetition row."""
    if not isinstance(binding, dict) or not isinstance(binding.get("work_unit"), dict):
        raise ValueError("paired calibration replay binding is malformed")
    unit = binding["work_unit"]
    control = binding.get("control_id")
    if (unit.get("control_id") != control or unit.get("pair_id") != binding.get("pair_id")
            or type(unit.get("repetition")) is not int or unit["repetition"] < 0):
        raise ValueError("paired calibration replay binding identity is inconsistent")
    return (f"paired-cal-{binding['campaign_id'][:12]}-{binding['pair_id'][:16]}-"
        f"{control.replace('/', '-')}-r{unit['repetition']:04d}")


def _paired_calibration_evaluation_groups(shard):
    """Group seed rows that one exact protocol-case evaluation call scores."""
    if not isinstance(shard, dict) or not isinstance(shard.get("work_units"), list):
        raise ValueError("paired calibration shard work units are malformed")
    groups = {}
    fields = ("scenario_id", "case_id", "category", "layout_id", "defender_family",
        "start_condition", "sensor_profile", "repetition", "evaluation_phase",
        "paired_control_id")
    for unit in shard["work_units"]:
        key = tuple(unit.get(field) for field in fields)
        groups.setdefault(key, {}).setdefault(unit.get("control_id"), []).append(unit)
    normalized = []
    for key, controls in groups.items():
        if set(controls) != {"D0", key[-1]}:
            raise ValueError("paired calibration evaluation group has incomplete controls")
        controls = {control: sorted(rows, key=lambda row: row["seed"])
            for control, rows in controls.items()}
        seed_lists = [tuple(row["seed"] for row in rows) for rows in controls.values()]
        if (seed_lists[0] != seed_lists[1]
                or len(seed_lists[0]) != len(set(seed_lists[0]))):
            raise ValueError("paired calibration evaluation group has unpaired seed coverage")
        for rows in controls.values():
            if any(tuple(row.get(field) for field in fields) != key for row in rows):
                raise ValueError("paired calibration evaluation group case tuple changed")
        normalized.append({"key": key, "controls": controls,
            "seeds": list(seed_lists[0])})
    return sorted(normalized, key=lambda group: repr(group["key"]))


def _paired_prepare_output_directory(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("paired calibration output directory uses a symlink")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("paired calibration output directory became a symlink")
    return path


def run_paired_calibration_shard(campaign, shard_index, *, repo_root,
        replay_root, output_root, phase0_bundle, configuration_artifacts):
    """Run/resume one shard after every actual Pass 1 gate is revalidated.

    The coordinator refuses work while any preregistered control lacks a
    reviewed execution contract. Device availability and identity are checked
    again inside the evaluator against this shard's exact assignment.
    """
    validate_paired_calibration_campaign(campaign)
    blockers = paired_calibration_execution_blockers(campaign)
    if blockers:
        raise ValueError("paired calibration execution blocked by controls: "
            + ", ".join(blockers))
    # Observe the user-stopped r4 campaign and both assigned physical devices
    # before importing the evaluator, creating output roots, or preparing HIP.
    from .meco_external_release_gate import verify_external_release_before_paired_run
    release_observation = verify_external_release_before_paired_run(campaign)
    gate = validate_paired_calibration_gate_artifacts(campaign,
        repo_root=repo_root, phase0_bundle=phase0_bundle,
        configuration_artifacts=configuration_artifacts)
    shard = _paired_shard_for_index(campaign, shard_index)
    worker = campaign["worker_assignments"][shard_index %
        len(campaign["worker_assignments"])]
    from .meco_track_training import evaluate_meco

    output_root, replay_root = Path(output_root), Path(replay_root)
    output_root = _paired_prepare_output_directory(output_root)
    replay_root = _paired_prepare_output_directory(replay_root)
    progress_path = output_root / f"shard-{shard_index:04d}.progress.json"
    final_path = output_root / f"shard-{shard_index:04d}.final.json"
    completed_rows, previous_progress_id = [], None
    if progress_path.exists() or progress_path.is_symlink():
        if progress_path.is_symlink():
            raise ValueError("paired calibration progress path is symlinked")
        previous_raw = progress_path.read_bytes()
        previous = json.loads(previous_raw)
        if previous_raw != json.dumps(previous, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8"):
            raise ValueError("existing paired calibration progress bytes are not canonical")
        validate_paired_calibration_progress(campaign, shard_index, previous,
            replay_root=replay_root, approval=gate["approval"])
        completed_rows = list(previous["completed_rows"])
        previous_progress_id = previous["progress_id"]
    completed_ids = {row["work_unit_id"] for row in completed_rows}
    control_by_id = {control["control_id"]: control for control in campaign["controls"]}
    for group in _paired_calibration_evaluation_groups(shard):
        comparator_id = group["key"][-1]
        comparator = control_by_id[comparator_id]
        if comparator.get("execution_supported") is not True:
            raise ValueError(f"paired calibration control is not executable: {comparator_id}")
        group_fields = ("scenario_id", "case_id", "category", "defender_family",
            "layout_id", "start_condition", "sensor_profile", "repetition",
            "evaluation_phase")
        group_values = (group["key"][0], group["key"][1], group["key"][2],
            group["key"][4], group["key"][5], group["key"][6],
            group["key"][7], group["key"][8], group["key"][9])
        expected_seeds = sorted(row["seed"] for row in campaign["calibration_tuples"]
            if tuple(row[field] for field in group_fields) == group_values)
        if group["seeds"] != expected_seeds:
            raise ValueError("paired calibration group differs from registered tuple seed set")
        qualification_case = dict(zip(("layout_id", "defender_family",
            "start_condition", "sensor_profile"), group["key"][3:7]))
        scenario_id, _, category = group["key"][:3]
        for selected_control_id in ("D0", comparator_id):
            selected = control_by_id[selected_control_id]
            units = group["controls"][selected_control_id]
            bindings = {unit["seed"]: paired_calibration_replay_binding(
                campaign, shard_index, unit) for unit in units}
            missing = [unit for unit in units
                if _canonical_sha256(unit) not in completed_ids]
            if not missing:
                continue
            policy = selected["policy"]
            replay_selection = {policy: list(group["seeds"])}
            group_digest = _canonical_sha256({"campaign_id": campaign["campaign_id"],
                "shard_index": shard_index, "group": group["key"],
                "selected_control_id": selected_control_id})
            replay_dir = replay_root / f"group-{group_digest}"
            _paired_prepare_output_directory(replay_dir)
            eval_dir = _paired_prepare_output_directory(output_root / "evaluations")
            attempt_id = uuid.uuid4().hex
            eval_path = eval_dir / f"group-{group_digest}-{attempt_id}.json"
            replay_dir = _paired_prepare_output_directory(replay_dir / attempt_id)
            result = evaluate_meco({}, scenario_id=scenario_id,
                categories=(category,), episodes=len(group["seeds"]),
                master_seed=min(group["seeds"]),
                episode_seeds=tuple(group["seeds"]), device=worker["device"],
                split="calibration", include_controls=True, output=str(eval_path),
                qualification_case=qualification_case, evaluation_policies=(),
                control_policies=("D0", comparator["policy"]),
                control_prediction_modes={comparator["policy"]: comparator["prediction_mode"]},
                calibration_control_policy=comparator["policy"],
                candidate_k_override=comparator["candidate_configuration"]["candidate_k"],
                calibration_route_cost_points_per_second=
                    comparator["candidate_configuration"]["route_cost_points_per_second"],
                replay_output_dir=replay_dir, replay_selection=replay_selection,
                simulated_score_estimand=str(Path(repo_root) /
                    campaign["estimand_approval"]["path"]),
                paired_calibration_bindings=bindings,
                paired_calibration_worker_identity=worker,
                paired_calibration_campaign=campaign,
                paired_calibration_shard_index=shard_index,
                paired_calibration_phase0_bundle=phase0_bundle,
                paired_calibration_configuration_artifacts=configuration_artifacts)
            records = result.get("records") if isinstance(result, dict) else None
            selected_records = {}
            for record in records if isinstance(records, list) else ():
                if record.get("policy") == policy and record.get("seed") in group["seeds"]:
                    if record["seed"] in selected_records:
                        raise ValueError("paired calibration evaluator returned duplicate policy/seed rows")
                    selected_records[record["seed"]] = record
            if set(selected_records) != set(group["seeds"]):
                raise ValueError("paired calibration evaluator did not return exact selected policy/seed coverage")
            for unit in missing:
                record = selected_records[unit["seed"]]
                replay_ref = record.get("dashboard_replay")
                if not isinstance(replay_ref, dict):
                    raise ValueError("paired calibration selected row lacks replay provenance")
                result_path = Path(replay_ref.get("result_path", ""))
                try:
                    relative_path = (result_path.absolute().relative_to(replay_root.absolute())
                        if result_path.is_absolute() else result_path)
                except ValueError as exc:
                    raise ValueError("paired calibration replay output escaped replay root") from exc
                relative = relative_path.as_posix()
                _, replay_raw, _ = _paired_repo_json_file(replay_root, relative,
                    kind="paired replay output")
                normalized_ref = {"simulation_id": replay_ref.get("simulation_id"),
                    "result_path": relative,
                    "evaluation_record_digest": replay_ref.get("evaluation_record_digest")}
                row = build_paired_calibration_outcome_row(campaign, shard_index,
                    unit, record, replay_raw, normalized_ref, gate["approval"])
                completed_rows.append(row)
                completed_ids.add(row["work_unit_id"])
            progress = build_paired_calibration_progress(campaign, shard_index,
                completed_rows, replay_root=replay_root, approval=gate["approval"])
            validate_paired_calibration_progress(campaign, shard_index, progress,
                replay_root=replay_root, approval=gate["approval"])
            write_paired_calibration_progress_atomic(progress_path, progress,
                expected_previous_progress_id=previous_progress_id)
            previous_progress_id = progress["progress_id"]
    progress = build_paired_calibration_progress(campaign, shard_index,
        completed_rows, replay_root=replay_root, approval=gate["approval"])
    validate_paired_calibration_progress(campaign, shard_index, progress,
        replay_root=replay_root, approval=gate["approval"])
    if not progress_path.exists():
        write_paired_calibration_progress_atomic(progress_path, progress)
    final = build_paired_calibration_final_shard(campaign, shard_index, progress,
        replay_root=replay_root, approval=gate["approval"],
        release_observation=release_observation)
    digest = write_paired_calibration_artifact_immutable(final_path, final)
    return {"campaign_id": campaign["campaign_id"], "shard_index": shard_index,
        "worker_id": worker["worker_id"], "completed_work_units": len(completed_rows),
        "progress_id": progress["progress_id"], "final_shard_sha256": digest,
        "final_shard": str(final_path),
        "external_release_observation": release_observation.record()}


def build_paired_calibration_progress(campaign, shard_index, completed_rows, *,
        replay_root, approval):
    """Compile progress only after reloading and revalidating every replay row."""
    shard = _paired_shard_for_index(campaign, shard_index)
    if not isinstance(completed_rows, list):
        raise ValueError("paired calibration progress rows must be a list")
    expected = {_canonical_sha256(unit): unit for unit in shard["work_units"]}
    seen = set()
    rows = []
    for row in completed_rows:
        if not isinstance(row, dict) or row.get("schema") != PAIRED_CALIBRATION_ARTIFACT:
            raise ValueError("paired calibration progress contains an unsupported row")
        unit_id = row.get("work_unit_id")
        if unit_id not in expected or unit_id in seen:
            raise ValueError("paired calibration progress has an unknown or duplicate work unit")
        if row.get("work_unit") != expected[unit_id]:
            raise ValueError("paired calibration progress row work unit changed")
        if row.get("campaign_id") != campaign["campaign_id"]:
            raise ValueError("paired calibration progress row campaign changed")
        validate_paired_calibration_outcome_row(campaign, shard_index, row,
            replay_root=replay_root, approval=approval)
        seen.add(unit_id)
        rows.append(row)
    rows.sort(key=lambda row: row["work_unit_id"])
    body = {"schema": PAIRED_CALIBRATION_PROGRESS,
        "campaign_id": campaign["campaign_id"],
        "shard_index": shard_index, "shard_manifest_id": shard["manifest_id"],
        "completed_work_unit_ids": [row["work_unit_id"] for row in rows],
        "completed_rows": rows,
        "complete": seen == set(expected)}
    return {**body, "progress_id": _canonical_sha256(body)}


def validate_paired_calibration_progress(campaign, shard_index, progress, *,
        replay_root, approval):
    """Validate resume bytes semantically; never repair or relabel stale progress."""
    shard = _paired_shard_for_index(campaign, shard_index)
    if not isinstance(progress, dict) or progress.get("schema") != PAIRED_CALIBRATION_PROGRESS:
        raise ValueError("paired calibration progress schema is unsupported")
    if set(progress) != {"schema", "campaign_id", "shard_index",
            "shard_manifest_id", "completed_work_unit_ids", "completed_rows",
            "complete", "progress_id"}:
        raise ValueError("paired calibration progress schema is extended or incomplete")
    body = {key: value for key, value in progress.items() if key != "progress_id"}
    if progress.get("progress_id") != _canonical_sha256(body):
        raise ValueError("paired calibration progress digest is invalid")
    if (progress.get("campaign_id") != campaign["campaign_id"]
            or progress.get("shard_index") != shard_index
            or progress.get("shard_manifest_id") != shard["manifest_id"]):
        raise ValueError("paired calibration progress belongs to another manifest")
    expected = {_canonical_sha256(unit): unit for unit in shard["work_units"]}
    rows = progress.get("completed_rows")
    ids = progress.get("completed_work_unit_ids")
    if not isinstance(rows, list) or not isinstance(ids, list):
        raise ValueError("paired calibration progress completion index is malformed")
    row_ids = []
    for row in rows:
        if (not isinstance(row, dict) or row.get("schema") != PAIRED_CALIBRATION_ARTIFACT
                or row.get("campaign_id") != campaign["campaign_id"]):
            raise ValueError("paired calibration progress contains an invalid outcome row")
        unit_id = row.get("work_unit_id")
        if unit_id not in expected or row.get("work_unit") != expected[unit_id]:
            raise ValueError("paired calibration progress contains a changed or foreign work unit")
        validate_paired_calibration_outcome_row(campaign, shard_index, row,
            replay_root=replay_root, approval=approval)
        row_ids.append(unit_id)
    if (row_ids != sorted(row_ids) or len(row_ids) != len(set(row_ids))
            or ids != row_ids or any(value not in expected for value in ids)):
        raise ValueError("paired calibration progress has duplicate, unordered, or foreign completion")
    if progress.get("complete") is not (set(ids) == set(expected)):
        raise ValueError("paired calibration progress completion flag is inconsistent")
    return progress


def build_paired_calibration_final_shard(campaign, shard_index, progress, *,
        replay_root, approval, release_observation):
    """Freeze a complete shard; refuse finalization while any control is blocked."""
    shard = _paired_shard_for_index(campaign, shard_index)
    blockers = paired_calibration_execution_blockers(campaign)
    if blockers:
        raise ValueError("paired calibration finalization blocked by controls: "
            + ", ".join(blockers))
    validate_paired_calibration_progress(campaign, shard_index, progress,
        replay_root=replay_root, approval=approval)
    if progress.get("complete") is not True:
        raise ValueError("paired calibration shard progress is incomplete")
    from .meco_external_release_gate import require_live_release_capability
    release_observation, release_observation_sha256 = require_live_release_capability(
        release_observation, campaign)
    body = {"schema": PAIRED_CALIBRATION_FINAL_SHARD,
        "campaign_id": campaign["campaign_id"], "shard_index": shard_index,
        "shard_manifest_id": shard["manifest_id"],
        "worker_id": shard["worker_id"],
        "worker_assignment_sha256": shard["worker_assignment_sha256"],
        "source_binding": campaign["source"],
        "phase0_binding": campaign["phase0_binding"],
        "phase0_binding_sha256": _canonical_sha256(campaign["phase0_binding"]),
        "tuple_ids": shard["tuple_ids"],
        "estimand_approval_sha256": campaign["estimand_approval"]["approval_sha256"],
        "scoring": campaign["scoring"],
        "control_bindings": shard["control_bindings"],
        "control_binding_sha256": shard["control_binding_sha256"],
        "controls": campaign["controls"],
        "controls_sha256": shard["controls_sha256"],
        "progress_id": progress["progress_id"],
        "release_observation": release_observation,
        "release_observation_sha256": release_observation_sha256,
        "completed_work_unit_ids": progress["completed_work_unit_ids"],
        "completed_rows": progress["completed_rows"], "complete": True}
    return {**body, "final_shard_id": _canonical_sha256(body)}


def _paired_variance(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _paired_seed_cluster_bootstrap_ci(seed_clusters, *, seed,
        replicates=BOOTSTRAP_REPLICATES):
    """Resample matched seed clusters while retaining every repetition in each draw."""
    keys = sorted(seed_clusters, key=repr)
    if not keys or any(not seed_clusters[key] for key in keys):
        raise ValueError("paired seed bootstrap requires populated seed clusters")
    state = seed & ((1 << 64) - 1)

    def randbelow(limit):
        nonlocal state
        state ^= (state << 13) & ((1 << 64) - 1)
        state ^= state >> 7
        state ^= (state << 17) & ((1 << 64) - 1)
        return state % limit

    means = []
    for _ in range(replicates):
        sampled = []
        for _ in keys:
            cluster = seed_clusters[keys[randbelow(len(keys))]]
            sampled.extend(cluster)
        means.append(sum(sampled) / len(sampled))
    means.sort()
    return [means[int(.025 * (replicates - 1))],
            means[int(.975 * (replicates - 1))]]


def _paired_control_summary(campaign, control_id, pairs):
    strata = {}
    outcome_deltas = {"red": [], "blue": []}
    safety_deltas = {name: [] for name in sorted(SAFETY_METRIC_NAMES)}
    throughput_deltas = []
    seed_groups = {}
    for pair_id, rows in pairs.items():
        if set(rows) != {"D0", control_id}:
            raise ValueError("paired calibration merge has incomplete or extra control pairing")
        baseline, candidate = rows["D0"], rows[control_id]
        left = {key: value for key, value in baseline["work_unit"].items()
            if key not in ("control_id", "paired_control_id", "pair_id")}
        right = {key: value for key, value in candidate["work_unit"].items()
            if key not in ("control_id", "paired_control_id", "pair_id")}
        if (left != right
                or baseline["work_unit"]["paired_control_id"] != control_id
                or candidate["work_unit"]["paired_control_id"] != control_id):
            raise ValueError("paired calibration merge pair tuple/repetition differs")
        tuple_row = candidate["work_unit"]
        stratum = tuple(tuple_row[key] for key in ("category", "layout_id",
            "defender_family", "start_condition", "sensor_profile"))
        deltas = {side: candidate["simulated_match_score"][side]
            - baseline["simulated_match_score"][side] for side in ("red", "blue")}
        strata.setdefault(stratum, {**{side: [] for side in ("red", "blue")},
            "safety": {name: [] for name in sorted(SAFETY_METRIC_NAMES)}})
        for side in ("red", "blue"):
            strata[stratum][side].append(float(deltas[side]))
            outcome_deltas[side].append(float(deltas[side]))
        seed_key = (stratum, tuple_row["seed"])
        seed_groups.setdefault(seed_key, {side: [] for side in ("red", "blue")})
        for side in ("red", "blue"):
            seed_groups[seed_key][side].append(float(deltas[side]))
        for name in safety_deltas:
            safety_delta = candidate["safety_metrics"][name] - baseline["safety_metrics"][name]
            safety_deltas[name].append(safety_delta)
            strata[stratum]["safety"][name].append(safety_delta)
        throughput_deltas.append(candidate["useful_throughput"]
            - baseline["useful_throughput"])
    if not outcome_deltas["red"]:
        raise ValueError("paired calibration merge has no completed pairs")
    summary = {"control_id": control_id, "pair_count": len(pairs),
        "simulated_match_score_difference": {}}
    for side in ("red", "blue"):
        seed_clusters = {repr(seed_key): values[side]
            for seed_key, values in seed_groups.items()}
        bootstrap_seed = int(_canonical_sha256({"campaign_id": campaign["campaign_id"],
            "control_id": control_id, "alliance": side})[:16], 16)
        differences = outcome_deltas[side]
        within = [_paired_variance(group[side]) for group in seed_groups.values()]
        seed_means = [sum(group[side]) / len(group[side])
            for group in seed_groups.values()]
        summary["simulated_match_score_difference"][side] = {
            "mean": sum(differences) / len(differences),
            "cluster_bootstrap_ci": _paired_seed_cluster_bootstrap_ci(seed_clusters,
                seed=bootstrap_seed, replicates=BOOTSTRAP_REPLICATES),
            "within_seed_variance_mean": sum(within) / len(within),
            "between_seed_variance": _paired_variance(seed_means)}
    summary["safety_rate_difference"] = {name: {
        "mean": sum(values) / len(values)} for name, values in safety_deltas.items()}
    stratum_summaries = []
    for stratum in sorted(strata, key=repr):
        stratum_key = dict(zip(("category", "layout_id", "defender_family",
            "start_condition", "sensor_profile"), stratum))
        values = strata[stratum]
        score_differences = {}
        for side in ("red", "blue"):
            seed = int(_canonical_sha256({"campaign_id": campaign["campaign_id"],
                "control_id": control_id, "stratum": stratum_key,
                "alliance": side})[:16], 16)
            seed_clusters = {seed: seed_values[side]
                for (seed_stratum, seed), seed_values in seed_groups.items()
                if seed_stratum == stratum}
            score_differences[side] = {"mean": sum(values[side]) / len(values[side]),
                "cluster_bootstrap_ci": _paired_seed_cluster_bootstrap_ci(
                    seed_clusters, seed=seed, replicates=BOOTSTRAP_REPLICATES)}
        stratum_summaries.append({**stratum_key, "pair_count": len(values["red"]),
            "simulated_match_score_difference": score_differences,
            "safety_rate_difference": {name: {"mean": sum(deltas) / len(deltas)}
                for name, deltas in values["safety"].items()}})
    summary["stratum_summaries"] = stratum_summaries
    summary["useful_throughput_difference_mean"] = (
        sum(throughput_deltas) / len(throughput_deltas))
    return summary


def _build_paired_control_bundle(campaign, summary, final_shard_sha256):
    control = next((row for row in campaign["controls"]
        if row["control_id"] == summary["control_id"]), None)
    if control is None or not isinstance(control.get("control_binding_sha256"), str):
        raise ValueError("paired control bundle lacks its exact frozen control binding")
    body = {"schema": "meco-paired-calibration-control-bundle-v2",
        "campaign_id": campaign["campaign_id"],
        "phase0_binding": campaign["phase0_binding"],
        "phase0_binding_sha256": _canonical_sha256(campaign["phase0_binding"]),
        "tuple_ids": sorted({row["tuple_id"] for row in campaign["calibration_tuples"]}),
        "control_id": summary["control_id"],
        "control": control,
        "controls_sha256": _paired_control_contracts_sha256(campaign["controls"]),
        "control_binding_sha256": control["control_binding_sha256"],
        "final_shard_sha256": list(final_shard_sha256),
        "summary": summary, "official_match_points": None,
        "qualification_eligible": False,
        "interpretation": "descriptive simulated outcome calibration; not official match points, qualification, or margin approval"}
    bundle = {**body, "bundle_id": _canonical_sha256(body)}
    validate_paired_calibration_control_bundle(campaign, bundle)
    raw = json.dumps(bundle, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return bundle, hashlib.sha256(raw).hexdigest()


def validate_paired_calibration_control_bundle(campaign, bundle):
    """Verify a published per-control bundle retains its complete frozen control."""
    if (not isinstance(campaign, dict) or not isinstance(bundle, dict)
            or not isinstance(campaign.get("controls"), list)
            or not isinstance(campaign.get("calibration_tuples"), list)
            or not isinstance(campaign.get("phase0_binding"), dict)):
        raise ValueError("paired control bundle is malformed")
    control = next((row for row in campaign["controls"]
        if row["control_id"] == bundle.get("control_id")), None)
    body = {key: value for key, value in bundle.items() if key != "bundle_id"}
    expected_fields = {"schema", "campaign_id", "phase0_binding",
        "phase0_binding_sha256", "tuple_ids", "control_id", "control",
        "controls_sha256", "control_binding_sha256", "final_shard_sha256",
        "summary", "official_match_points", "qualification_eligible",
        "interpretation", "bundle_id"}
    if (control is None or set(bundle) != expected_fields
            or bundle.get("controls_sha256") != _paired_control_contracts_sha256(campaign["controls"])
            or control.get("control_binding_sha256") != _canonical_sha256({key: value
                for key, value in control.items() if key != "control_binding_sha256"})
            or bundle.get("schema") != "meco-paired-calibration-control-bundle-v2"
            or bundle.get("campaign_id") != campaign["campaign_id"]
            or bundle.get("phase0_binding") != campaign["phase0_binding"]
            or bundle.get("phase0_binding_sha256") != _canonical_sha256(campaign["phase0_binding"])
            or bundle.get("tuple_ids") != sorted(row["tuple_id"] for row in campaign["calibration_tuples"])
            or bundle.get("control") != control
            or bundle.get("controls_sha256") != _paired_control_contracts_sha256(campaign["controls"])
            or bundle.get("control_binding_sha256") != control["control_binding_sha256"]
            or bundle.get("official_match_points") is not None
            or bundle.get("qualification_eligible") is not False
            or bundle.get("bundle_id") != _canonical_sha256(body)):
        raise ValueError("paired control bundle identity or full control contract changed")
    return bundle


def build_paired_calibration_merged_report(campaign, final_shard_raws, *,
        replay_root, approval):
    """Strictly merge complete canonical shard bytes and produce preregistered summaries."""
    validate_paired_calibration_campaign(campaign)
    blockers = paired_calibration_execution_blockers(campaign)
    if blockers:
        raise ValueError("paired calibration merge blocked by controls: "
            + ", ".join(blockers))
    if not isinstance(final_shard_raws, list) or len(final_shard_raws) != len(campaign["shards"]):
        raise ValueError("paired calibration merge requires exactly one final artifact per shard")
    final_by_index = {}
    final_hashes = {}
    for raw in final_shard_raws:
        if not isinstance(raw, bytes) or not raw:
            raise ValueError("paired calibration final shard must be supplied as exact bytes")
        try:
            artifact = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("paired calibration final shard is invalid JSON") from exc
        canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")
        if raw != canonical:
            raise ValueError("paired calibration final shard bytes are not canonical")
        index = artifact.get("shard_index") if isinstance(artifact, dict) else None
        if type(index) is not int or index in final_by_index:
            raise ValueError("paired calibration merge has duplicate or invalid shard index")
        shard = _paired_shard_for_index(campaign, index)
        body = {key: value for key, value in artifact.items() if key != "final_shard_id"}
        expected_fields = {"schema", "campaign_id", "shard_index",
            "shard_manifest_id", "worker_id", "worker_assignment_sha256",
            "source_binding", "phase0_binding", "phase0_binding_sha256", "tuple_ids",
            "estimand_approval_sha256", "scoring", "control_bindings",
            "control_binding_sha256", "controls", "controls_sha256", "progress_id",
            "release_observation", "release_observation_sha256",
            "completed_work_unit_ids", "completed_rows", "complete", "final_shard_id"}
        from .meco_external_release_gate import validate_persisted_release_observation_record
        try:
            release_observation_sha256 = validate_persisted_release_observation_record(
                artifact.get("release_observation"), campaign)
        except ValueError as exc:
            raise ValueError("paired calibration final shard release observation is invalid") from exc
        if (set(artifact) != expected_fields
                or artifact.get("schema") != PAIRED_CALIBRATION_FINAL_SHARD
                or artifact.get("final_shard_id") != _canonical_sha256(body)
                or artifact.get("campaign_id") != campaign["campaign_id"]
                or artifact.get("shard_manifest_id") != shard["manifest_id"]
                or artifact.get("worker_id") != shard["worker_id"]
                or artifact.get("worker_assignment_sha256") != shard["worker_assignment_sha256"]
                or artifact.get("source_binding") != campaign["source"]
                or artifact.get("phase0_binding") != campaign["phase0_binding"]
                or artifact.get("phase0_binding_sha256") != _canonical_sha256(campaign["phase0_binding"])
                or artifact.get("tuple_ids") != shard["tuple_ids"]
                or artifact.get("estimand_approval_sha256") != campaign["estimand_approval"]["approval_sha256"]
                or artifact.get("scoring") != campaign["scoring"]
                or artifact.get("control_bindings") != shard["control_bindings"]
                or artifact.get("control_binding_sha256") != shard["control_binding_sha256"]
                or artifact.get("controls") != campaign["controls"]
                or artifact.get("controls_sha256")
                    != _paired_control_contracts_sha256(campaign["controls"])
                or artifact.get("release_observation_sha256") != release_observation_sha256
                or artifact.get("complete") is not True):
            raise ValueError("paired calibration final shard identity or bindings changed")
        expected_ids = sorted(_canonical_sha256(unit) for unit in shard["work_units"])
        if artifact.get("completed_work_unit_ids") != expected_ids:
            raise ValueError("paired calibration final shard has incomplete or extra work")
        progress_body = {"schema": PAIRED_CALIBRATION_PROGRESS,
            "campaign_id": campaign["campaign_id"], "shard_index": index,
            "shard_manifest_id": shard["manifest_id"],
            "completed_work_unit_ids": expected_ids,
            "completed_rows": artifact.get("completed_rows"), "complete": True}
        if artifact.get("progress_id") != _canonical_sha256(progress_body):
            raise ValueError("paired calibration final shard progress digest is invalid")
        row_ids = []
        for row in artifact.get("completed_rows", []):
            validate_paired_calibration_outcome_row(campaign, index, row,
                replay_root=replay_root, approval=approval)
            row_ids.append(row["work_unit_id"])
        if row_ids != expected_ids:
            raise ValueError("paired calibration final shard rows do not exactly cover schedule")
        final_by_index[index] = artifact
        final_hashes[index] = hashlib.sha256(raw).hexdigest()
    if set(final_by_index) != set(range(len(campaign["shards"]))):
        raise ValueError("paired calibration merge shard set is incomplete")
    all_rows = [row for index in range(len(campaign["shards"]))
        for row in final_by_index[index]["completed_rows"]]
    row_by_unit = {row["work_unit_id"]: row for row in all_rows}
    if len(row_by_unit) != len(all_rows):
        raise ValueError("paired calibration merge has overlapping work units")
    expected_units = [unit for shard in campaign["shards"] for unit in shard["work_units"]]
    if set(row_by_unit) != {_canonical_sha256(unit) for unit in expected_units}:
        raise ValueError("paired calibration merge does not cover the frozen schedule")
    pairs = {}
    for row in all_rows:
        pairs.setdefault(row["pair_id"], {})[row["control_id"]] = row
    summaries = [_paired_control_summary(campaign, control_id,
        {pair_id: rows for pair_id, rows in pairs.items() if control_id in rows})
        for control_id in PAIRED_CALIBRATION_CONTROLS[1:]]
    control_bundles = []
    bundle_refs = []
    for summary in summaries:
        bundle, artifact_sha256 = _build_paired_control_bundle(campaign,
            summary, [final_hashes[index] for index in range(len(campaign["shards"]))])
        control_bundles.append(bundle)
        bundle_refs.append({"control_id": summary["control_id"],
            "bundle_id": bundle["bundle_id"],
            "artifact_sha256": artifact_sha256})
    body = {"schema": "meco-paired-calibration-merged-report-v2",
        "campaign_id": campaign["campaign_id"],
        "phase0_binding": campaign["phase0_binding"],
        "phase0_binding_sha256": _canonical_sha256(campaign["phase0_binding"]),
        "tuple_ids": sorted({unit["tuple_id"] for unit in expected_units}),
        "control_binding_sha256": _canonical_sha256(
            _paired_shard_control_bindings(campaign["controls"])),
        "controls": campaign["controls"],
        "controls_sha256": _paired_control_contracts_sha256(campaign["controls"]),
        "final_shard_sha256": [final_hashes[index]
            for index in range(len(campaign["shards"]))],
        "work_unit_count": len(all_rows), "control_bundles": bundle_refs,
        "official_match_points": None, "qualification_eligible": False,
        "interpretation": "descriptive simulated outcome calibration; not official match points, qualification, or margin approval"}
    return {**body, "report_id": _canonical_sha256(body),
        "_control_bundle_artifacts": control_bundles}


def write_paired_calibration_artifact_immutable(path, artifact):
    """Atomically publish canonical JSON without replacing conflicting bytes."""
    path = Path(path)
    if not isinstance(artifact, dict):
        raise ValueError("paired calibration artifact must be an object")
    raw = json.dumps(artifact, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    absolute = path.absolute()
    for part in (absolute, *absolute.parents):
        if part.is_symlink():
            raise ValueError("paired calibration artifact output path uses a symlink")
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
            created = True
        except FileExistsError:
            created = False
        if not created:
            if path.is_symlink() or path.read_bytes() != raw:
                raise ValueError("paired calibration immutable artifact ID has conflicting bytes")
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(raw).hexdigest()


def write_paired_calibration_progress_atomic(path, progress, *,
        expected_previous_progress_id=None):
    """Atomically advance one validated progress file with compare-and-swap.

    Callers must first validate the campaign, approval, and every replay row.
    Progress is mutable only by monotonic row addition; immutable final and
    campaign artifacts continue to use ``write_paired_calibration_artifact_immutable``.
    """
    if not isinstance(progress, dict) or progress.get("schema") != PAIRED_CALIBRATION_PROGRESS:
        raise ValueError("paired calibration progress schema is unsupported")
    body = {key: value for key, value in progress.items() if key != "progress_id"}
    if progress.get("progress_id") != _canonical_sha256(body):
        raise ValueError("paired calibration progress digest is invalid")
    path = Path(path)
    parent = path.parent.absolute()
    for part in (parent, *parent.parents):
        if part.is_symlink():
            raise ValueError("paired calibration progress path uses a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{path.name}.lock"
    if lock_path.is_symlink():
        raise ValueError("paired calibration progress lock is symlinked")
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock_path, lock_flags, 0o600)
    import fcntl
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if path.is_symlink():
            raise ValueError("paired calibration progress destination is symlinked")
        existing = None
        if path.exists():
            try:
                existing = json.loads(path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("existing paired calibration progress is unreadable") from exc
        actual_previous = None if existing is None else existing.get("progress_id")
        if actual_previous != expected_previous_progress_id:
            raise ValueError("paired calibration progress changed concurrently or CAS token is stale")
        if existing is not None:
            if (existing.get("campaign_id") != progress.get("campaign_id")
                    or existing.get("shard_index") != progress.get("shard_index")
                    or existing.get("shard_manifest_id") != progress.get("shard_manifest_id")
                    or not set(existing.get("completed_work_unit_ids", [])).issubset(
                        progress.get("completed_work_unit_ids", []))):
                raise ValueError("paired calibration progress advance is not monotonic")
        raw = json.dumps(progress, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")
        temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        temporary_fd = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(temporary_fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if path.is_symlink():
                raise ValueError("paired calibration progress destination became symlinked")
            os.replace(temporary, path)
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return hashlib.sha256(raw).hexdigest()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def build_paired_calibration_outcome_row(campaign, shard_index, work_unit,
        evaluation_record, replay_raw, replay_ref, approval):
    """Bind one evaluated work unit to exact replay bytes and reconstructed score."""
    shard = _paired_shard_for_index(campaign, shard_index)
    unit_id = _canonical_sha256(work_unit)
    if work_unit not in shard["work_units"]:
        raise ValueError("paired calibration outcome work unit is outside its shard")
    control = next((row for row in campaign["controls"]
        if row["control_id"] == work_unit["control_id"]), None)
    if control is None or control.get("execution_supported") is not True:
        raise ValueError("paired calibration control is not executable")
    if not isinstance(evaluation_record, dict) or not isinstance(replay_ref, dict):
        raise ValueError("paired calibration evaluation or replay reference is malformed")
    for field in ("scenario_id", "category", "layout_id", "defender_family",
            "start_condition", "sensor_profile", "split", "seed"):
        if evaluation_record.get(field) != work_unit[field]:
            raise ValueError("paired calibration evaluation tuple differs from frozen work unit")
    if any(evaluation_record.get(key) != control.get(key) for key in
            ("policy", "policy_id", "prediction_mode", "prediction_contract")):
        raise ValueError("paired calibration evaluation control differs from frozen registry")
    contract = evaluation_record.get("prediction_contract")
    if (control.get("prediction_contract_sha256") !=
            (None if contract is None else _canonical_sha256(contract))
            or control.get("control_binding_sha256") != _canonical_sha256(
                {key: value for key, value in control.items()
                    if key != "control_binding_sha256"})):
        raise ValueError("paired calibration predictor contract digest differs from frozen control")
    source_hash = campaign["source"]["source_identity_sha256"]
    if evaluation_record.get("source_identity_sha256") != source_hash:
        raise ValueError("paired calibration evaluation source identity changed")
    metrics = evaluation_record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("paired calibration evaluation metrics are missing")
    required_metrics = set(SAFETY_METRIC_NAMES) | {"success", "useful_throughput"}
    if (not required_metrics <= set(metrics)
            or type(metrics.get("success")) not in (int, float)
            or not math.isfinite(float(metrics["success"]))
            or not 0 <= float(metrics["success"]) <= 1
            or type(metrics.get("useful_throughput")) not in (int, float)
            or not math.isfinite(float(metrics["useful_throughput"]))
            or metrics["useful_throughput"] < 0
            or any(type(metrics.get(name)) not in (int, float)
                or not math.isfinite(float(metrics[name]))
                or not 0 <= float(metrics[name]) <= 1 for name in SAFETY_METRIC_NAMES)):
        raise ValueError("paired calibration evaluation metrics are incomplete or invalid")
    if (evaluation_record.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
            or evaluation_record.get("safety_metrics_source_identity_sha256") != source_hash):
        raise ValueError("paired calibration safety metrics lack exact source binding")
    safety_metrics = evaluation_record.get("meco_metric_ledger", {}).get("safety_metrics")
    if (not isinstance(safety_metrics, dict)
            or set(safety_metrics) != SAFETY_METRIC_NAMES
            or any(type(safety_metrics.get(name)) not in (int, float)
                or not math.isfinite(float(safety_metrics[name]))
                or not 0 <= float(safety_metrics[name]) <= 1 for name in SAFETY_METRIC_NAMES)):
        raise ValueError("paired calibration exact safety ledger is missing or invalid")
    replay_path = replay_ref.get("result_path")
    replay_id = replay_ref.get("simulation_id")
    replay_record_digest = replay_ref.get("evaluation_record_digest")
    if (not isinstance(replay_raw, bytes) or not replay_raw
            or not isinstance(replay_path, str) or not replay_path.strip()
            or not isinstance(replay_id, str) or not replay_id.strip()
            or not isinstance(replay_record_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", replay_record_digest) is None):
        raise ValueError("paired calibration replay bytes or reference are incomplete")
    replay_relative = Path(replay_path)
    if (replay_relative.is_absolute()
            or any(part in ("", ".", "..") for part in replay_relative.parts)
            or "\\" in replay_path):
        raise ValueError("paired calibration replay path is unsafe")
    expected_record_digest = _canonical_sha256({key: value
        for key, value in evaluation_record.items() if key != "dashboard_replay"})
    if replay_record_digest != expected_record_digest:
        raise ValueError("paired calibration replay reference evaluation digest differs")
    try:
        replay = json.loads(replay_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("paired calibration replay is invalid JSON") from exc
    if (not isinstance(replay, dict) or replay.get("simulation_id") != replay_id
            or replay.get("evaluation_record_digest") != replay_record_digest
            or replay.get("replay_contract_schema") != PAIRED_CALIBRATION_REPLAY_CONTRACT
            or replay.get("paired_calibration_binding") != paired_calibration_replay_binding(
                campaign, shard_index, work_unit)
            or replay.get("condition_id") != paired_calibration_condition_id(
                paired_calibration_replay_binding(campaign, shard_index, work_unit))
            or replay.get("source_identity_sha256") != source_hash):
        raise ValueError("paired calibration replay identity/source differs from its reference")
    for field in ("scenario_id", "category", "seed", "split", "policy", "policy_id",
            "prediction_mode"):
        expected = evaluation_record.get(field)
        if replay.get(field) != expected:
            raise ValueError("paired calibration replay tuple/control differs from evaluation")
    from .meco_simulated_score_ledger import (estimand_approval_digest,
        validate_estimand_approval, validate_replay_score_binding)
    validate_estimand_approval(approval, source_identity_sha256=source_hash)
    approval_digest = estimand_approval_digest(approval)
    if approval_digest != campaign["estimand_approval"]["approval_sha256"]:
        raise ValueError("paired calibration score approval differs from campaign")
    terminal_counts = evaluation_record.get("score_terminal_counters")
    ledger = validate_replay_score_binding(replay, evaluation_record,
        terminal_counts, approval)
    from .meco_simulated_score_ledger import (EVENT_STREAM_SCHEMA, LEDGER_SCHEMA,
        RULES_ID, RULES_SHA256)
    worker = campaign["worker_assignments"][shard_index % len(campaign["worker_assignments"])]
    if (evaluation_record.get("execution_device", {}).get("device") != worker["device"]
            or evaluation_record.get("execution_device", {}).get("accelerator_backend") != "ROCm"):
        raise ValueError("paired calibration evaluation did not use its assigned indexed WX9100")
    outcome_digest = _canonical_sha256({"metrics": metrics,
        "safety_metrics": safety_metrics,
        "execution_failures": evaluation_record.get("execution_failures")})
    safety_digest = _canonical_sha256(safety_metrics)
    body = {"schema": PAIRED_CALIBRATION_ARTIFACT,
        "campaign_id": campaign["campaign_id"], "shard_index": shard_index,
        "shard_manifest_id": shard["manifest_id"], "work_unit_id": unit_id,
        "work_unit": work_unit, "tuple_id": work_unit["tuple_id"],
        "phase0_binding_sha256": _canonical_sha256(campaign["phase0_binding"]),
        "pair_id": work_unit["pair_id"],
        "control_id": work_unit["control_id"], "replay_id": replay_id,
        "replay_path": replay_path,
        "replay_sha256": hashlib.sha256(replay_raw).hexdigest(),
        "evaluation_record_sha256": replay_record_digest,
        "worker_assignment": worker,
        "worker_assignment_sha256": _canonical_sha256(worker),
        "control_binding": control,
        "control_binding_sha256": control["control_binding_sha256"],
        "paired_calibration_binding": paired_calibration_replay_binding(
            campaign, shard_index, work_unit),
        "paired_calibration_binding_sha256": _canonical_sha256(
            paired_calibration_replay_binding(campaign, shard_index, work_unit)),
        "evaluation_metrics": metrics,
        "evaluation_metrics_sha256": _canonical_sha256(metrics),
        "execution_failures": evaluation_record.get("execution_failures"),
        "outcome_digest": outcome_digest, "safety_digest": safety_digest,
        "ledger_schema": LEDGER_SCHEMA, "rules_id": RULES_ID,
        "rules_sha256": RULES_SHA256,
        "event_stream_schema": EVENT_STREAM_SCHEMA,
        "replay_contract_schema": PAIRED_CALIBRATION_REPLAY_CONTRACT,
        "event_stream_sha256": replay["score_event_stream"]["events_sha256"],
        "ledger_sha256": ledger["ledger_sha256"],
        "estimand_approval_sha256": approval_digest,
        "terminal_score_counts": terminal_counts,
        "score_binding": {key: evaluation_record.get(key) for key in (
            "simulated_score_ledger", "simulated_match_score",
            "source_identity_sha256", "official_match_points",
            "qualification_eligible")},
        "simulated_match_score": ledger["simulated_match_score"],
        "official_match_points": None, "qualification_eligible": False,
        "useful_throughput": float(metrics["useful_throughput"]),
        "raw_fuel_counts": list(terminal_counts),
        "safety_metrics_schema": SAFETY_METRICS_SCHEMA,
        "safety_metrics_source_identity_sha256": source_hash,
        "safety_metrics": {name: float(safety_metrics[name])
            for name in sorted(SAFETY_METRIC_NAMES)}}
    return {**body, "outcome_row_sha256": _canonical_sha256(body)}


def validate_paired_calibration_outcome_row(campaign, shard_index, row, *,
        replay_root, approval):
    """Fail closed unless row schema, worker, exact replay bytes, and score agree."""
    shard = _paired_shard_for_index(campaign, shard_index)
    expected_fields = {"schema", "campaign_id", "shard_index", "shard_manifest_id",
        "work_unit_id", "work_unit", "tuple_id", "phase0_binding_sha256",
        "pair_id", "control_id", "replay_id",
        "replay_path", "replay_sha256", "evaluation_record_sha256",
        "worker_assignment", "worker_assignment_sha256", "control_binding",
        "control_binding_sha256", "ledger_schema", "rules_id",
        "paired_calibration_binding", "paired_calibration_binding_sha256",
        "rules_sha256", "event_stream_schema", "replay_contract_schema",
        "event_stream_sha256", "ledger_sha256", "estimand_approval_sha256",
        "evaluation_metrics", "evaluation_metrics_sha256", "outcome_digest", "safety_digest",
        "execution_failures",
        "terminal_score_counts", "score_binding", "simulated_match_score",
        "official_match_points", "qualification_eligible", "useful_throughput",
        "raw_fuel_counts", "safety_metrics_schema",
        "safety_metrics_source_identity_sha256", "safety_metrics",
        "outcome_row_sha256"}
    if not isinstance(row, dict) or set(row) != expected_fields:
        raise ValueError("paired calibration outcome row schema is incomplete or extended")
    body = {key: value for key, value in row.items() if key != "outcome_row_sha256"}
    if row.get("outcome_row_sha256") != _canonical_sha256(body):
        raise ValueError("paired calibration outcome row digest is invalid")
    unit = row.get("work_unit")
    unit_id = row.get("work_unit_id")
    if (row.get("schema") != PAIRED_CALIBRATION_ARTIFACT
            or row.get("campaign_id") != campaign["campaign_id"]
            or row.get("shard_index") != shard_index
            or row.get("shard_manifest_id") != shard["manifest_id"]
            or not isinstance(unit, dict) or unit not in shard["work_units"]
            or unit_id != _canonical_sha256(unit)
            or row.get("tuple_id") != unit.get("tuple_id")
            or row.get("phase0_binding_sha256") != _canonical_sha256(campaign["phase0_binding"])
            or row.get("pair_id") != unit["pair_id"]
            or row.get("control_id") != unit["control_id"]):
        raise ValueError("paired calibration outcome row is foreign to its manifest/work unit")
    control = next(item for item in campaign["controls"]
        if item["control_id"] == unit["control_id"])
    if control.get("execution_supported") is not True:
        raise ValueError("paired calibration outcome row uses a blocked control")
    worker = campaign["worker_assignments"][shard_index % len(campaign["worker_assignments"])]
    if (row.get("worker_assignment") != worker
            or row.get("worker_assignment_sha256") != _canonical_sha256(worker)):
        raise ValueError("paired calibration outcome worker/device identity changed")
    if (row.get("control_binding") != control
            or row.get("control_binding_sha256") != control["control_binding_sha256"]
            or control.get("control_binding_sha256") != _canonical_sha256(
                {key: value for key, value in control.items()
                    if key != "control_binding_sha256"})):
        raise ValueError("paired calibration outcome predictor/control identity changed")
    replay_binding = paired_calibration_replay_binding(campaign, shard_index, unit)
    if (row.get("paired_calibration_binding") != replay_binding
            or row.get("paired_calibration_binding_sha256") != _canonical_sha256(replay_binding)):
        raise ValueError("paired calibration outcome binding differs from campaign shard")
    from .meco_simulated_score_ledger import (EVENT_STREAM_SCHEMA, LEDGER_SCHEMA,
        RULES_ID, RULES_SHA256, estimand_approval_digest,
        validate_estimand_approval, validate_replay_score_binding)
    if (row.get("ledger_schema") != LEDGER_SCHEMA or row.get("rules_id") != RULES_ID
            or row.get("rules_sha256") != RULES_SHA256
            or row.get("event_stream_schema") != EVENT_STREAM_SCHEMA
            or row.get("replay_contract_schema") != PAIRED_CALIBRATION_REPLAY_CONTRACT
            or row.get("estimand_approval_sha256") != campaign["estimand_approval"]["approval_sha256"]):
        raise ValueError("paired calibration replay/scorer/approval contract changed")
    source_hash = campaign["source"]["source_identity_sha256"]
    validate_estimand_approval(approval, source_identity_sha256=source_hash)
    if estimand_approval_digest(approval) != row["estimand_approval_sha256"]:
        raise ValueError("paired calibration approval bytes do not match row digest")
    replay_path = row.get("replay_path")
    resolved, raw, replay = _paired_repo_json_file(replay_root, replay_path,
        kind="paired replay")
    if hashlib.sha256(raw).hexdigest() != row["replay_sha256"]:
        raise ValueError("paired calibration replay artifact bytes changed")
    if (replay.get("simulation_id") != row["replay_id"]
            or replay.get("evaluation_record_digest") != row["evaluation_record_sha256"]
            or replay.get("source_identity_sha256") != source_hash
            or replay.get("paired_calibration_worker_identity") != worker
            or replay.get("paired_calibration_binding") != replay_binding
            or replay.get("condition_id") != paired_calibration_condition_id(replay_binding)
            or replay.get("execution_device") != {"device": worker["device"],
                "accelerator_backend": "ROCm"}
            or replay.get("device") != {"device": worker["device"],
                "accelerator_backend": "ROCm"}
            or replay.get("simulation_device") != worker["device"]
            or replay.get("score_event_stream", {}).get("schema") != EVENT_STREAM_SCHEMA
            or replay.get("score_event_stream", {}).get("events_sha256") != row["event_stream_sha256"]):
        raise ValueError("paired calibration replay identity/event stream changed")
    for field in ("scenario_id", "category", "seed", "split", "policy", "policy_id",
            "prediction_mode"):
        if replay.get(field) != ({"scenario_id": unit["scenario_id"],
                "category": unit["category"], "seed": unit["seed"],
                "split": unit["split"], "policy": control["policy"],
                "policy_id": control["policy_id"],
                "prediction_mode": control["prediction_mode"]}[field]):
            raise ValueError("paired calibration replay tuple/control differs from work unit")
    expected_case = {key: unit[key] for key in (
        "layout_id", "defender_family", "start_condition", "sensor_profile")}
    replay_evaluation = replay.get("evaluation_record")
    if (replay.get("qualification_case") != expected_case
            or not isinstance(replay_evaluation, dict)
            or replay_evaluation.get("metrics") != row["evaluation_metrics"]
            or replay_evaluation.get("safety_metrics") != row["safety_metrics"]
            or replay.get("outcome_digest") != row["outcome_digest"]
            or replay.get("safety_digest") != row["safety_digest"]
            or _canonical_sha256(replay_evaluation.get("metrics"))
                != row["evaluation_metrics_sha256"]):
        raise ValueError("paired calibration replay outcome/case/safety digest differs")
    if (not isinstance(row.get("execution_failures"), list)
            or _canonical_sha256({"metrics": row["evaluation_metrics"],
                "safety_metrics": row["safety_metrics"],
                "execution_failures": row["execution_failures"]}) != row["outcome_digest"]
            or _canonical_sha256(row["safety_metrics"]) != row["safety_digest"]):
        raise ValueError("paired calibration evaluation outcome/safety digest is invalid")
    rebuilt = validate_replay_score_binding(replay, row["score_binding"],
        row["terminal_score_counts"], approval)
    if (rebuilt["ledger_sha256"] != row["ledger_sha256"]
            or rebuilt["simulated_match_score"] != row["simulated_match_score"]
            or row["score_binding"].get("simulated_match_score") != row["simulated_match_score"]
            or row.get("official_match_points") is not None
            or row.get("qualification_eligible") is not False
            or row.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
            or row.get("safety_metrics_source_identity_sha256") != source_hash
            or set(row.get("safety_metrics", {})) != SAFETY_METRIC_NAMES
            or any(type(value) not in (int, float) or not math.isfinite(value)
                or not 0 <= value <= 1 for value in row["safety_metrics"].values())
            or type(row.get("useful_throughput")) not in (int, float)
            or not math.isfinite(row["useful_throughput"])
            or row["useful_throughput"] < 0
            or row.get("raw_fuel_counts") != row.get("terminal_score_counts")
            or not isinstance(row.get("terminal_score_counts"), list)
            or len(row["terminal_score_counts"]) != 2
            or any(type(value) is not int or value < 0
                for value in row["terminal_score_counts"])):
        raise ValueError("paired calibration outcome row disagrees with replay score/safety contract")
    return row


def _read_paired_bound_json(repo_root, binding, *, kind):
    resolved, raw, value = _paired_repo_json_file(repo_root, binding["path"], kind=kind)
    if hashlib.sha256(raw).hexdigest() != binding["artifact_sha256"]:
        raise ValueError(f"paired calibration {kind} artifact bytes changed")
    return value


def _paired_repo_json_file(repo_root, relative_path, *, kind):
    root = Path(repo_root).resolve()
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"paired calibration {kind} path is unsafe")
    candidate = root / relative
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError(f"paired calibration {kind} path uses a symlink")
    resolved = candidate.resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"paired calibration {kind} artifact is missing or outside checkout")
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"paired calibration {kind} artifact is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"paired calibration {kind} artifact must contain an object")
    return resolved, raw, value


def prepare_paired_calibration_gate_bindings(*, repo_root, phase0_bundle,
        estimand_approval_path, configuration_artifacts):
    """Validate effective Phase 0 intake and the separate human score approval."""
    approval_path, approval_raw, _ = _paired_repo_json_file(
        repo_root, estimand_approval_path, kind="estimand approval")
    phase0_digests = _validate_effective_phase0_bundle(phase0_bundle, repo_root)
    phase0 = _paired_phase0_bundle_binding(phase0_bundle, phase0_digests)
    freeze = phase0_bundle["baseline_freeze"]
    registered_tuples = _paired_calibration_registered_tuples(phase0_bundle)
    materialization = phase0_bundle["partition_materialization"]
    budget = materialization.get("evaluation_budget")
    if (not isinstance(budget, dict) or set(budget) != {"episodes_per_tuple", "repetitions"}
            or any(type(value) is not int or value < 1 for value in budget.values())):
        raise ValueError("effective Phase 0 bundle evaluation budget is invalid")
    from .curriculum_training import source_identity, source_tree_identity
    source_hash = hashlib.sha256(json.dumps(source_identity(), sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    source_tree_hash = hashlib.sha256(json.dumps(source_tree_identity(), sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    root = Path(repo_root).resolve()
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        tree = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
            check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("paired calibration cannot inspect source commit/tree") from exc
    from .meco_simulated_score_ledger import (load_estimand_approval,
        estimand_approval_digest)
    approval, approval_raw_sha = load_estimand_approval(approval_path,
        source_identity_sha256=source_hash)
    if approval_raw_sha != hashlib.sha256(approval_raw).hexdigest():
        raise ValueError("paired calibration approval bytes changed while loading")
    if (phase0["source_commit"] != commit or phase0["source_tree"] != tree
            or phase0["dependency_lock_sha256"] == ""):
        raise ValueError("paired calibration baseline source/runtime binding is stale")
    _validate_effective_phase0_configuration_bytes(phase0_bundle, configuration_artifacts)
    if source_hash != _calibration_source_identity_sha256():
        raise ValueError("paired calibration source identity helpers disagree")
    source = {"commit": commit, "tree": tree,
        "source_identity_sha256": source_hash,
        "source_tree_identity_sha256": source_tree_hash,
        "dependency_lock_sha256": phase0["dependency_lock_sha256"]}
    approval_binding = {"path": approval_path.relative_to(root).as_posix(),
        "artifact_sha256": approval_raw_sha,
        "schema": approval["schema"],
        "approval_sha256": estimand_approval_digest(approval)}
    return {"source_binding": source, "phase0_binding": phase0,
        "calibration_tuples": registered_tuples,
        "evaluation_budget": dict(budget),
        "estimand_approval_binding": approval_binding}


def validate_paired_calibration_gate_artifacts(campaign, *, repo_root, phase0_bundle,
        configuration_artifacts):
    """Revalidate effective Phase 0 bundle bytes and the separate human approval."""
    validate_paired_calibration_campaign(campaign)
    source_hash = _calibration_source_identity_sha256()
    if campaign["source"]["source_identity_sha256"] != source_hash:
        raise ValueError("paired calibration source identity differs from current checkout")
    from .curriculum_training import source_identity, source_tree_identity
    current_identity = hashlib.sha256(json.dumps(source_identity(), sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    current_tree_identity = hashlib.sha256(json.dumps(source_tree_identity(), sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    root = Path(repo_root).resolve()
    try:
        current_commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        current_tree = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
            check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("paired calibration cannot verify source commit/tree") from exc
    if (current_identity != source_hash
            or current_tree_identity != campaign["source"]["source_tree_identity_sha256"]
            or current_commit != campaign["source"]["commit"]
            or current_tree != campaign["source"]["tree"]):
        raise ValueError("paired calibration source commit/tree/runtime identity changed")
    phase0_digests = _validate_effective_phase0_bundle(phase0_bundle, root)
    actual_phase0 = _paired_phase0_bundle_binding(phase0_bundle, phase0_digests)
    if actual_phase0 != campaign["phase0_binding"]:
        raise ValueError("paired calibration effective Phase 0 bundle bytes or digests changed")
    _validate_effective_phase0_configuration_bytes(phase0_bundle, configuration_artifacts)
    freeze = phase0_bundle["baseline_freeze"]
    if (freeze.get("source", {}).get("commit") != campaign["source"]["commit"]
            or freeze.get("source", {}).get("tree") != campaign["source"]["tree"]
            or freeze.get("runtime", {}).get("dependency_lock_sha256")
                != campaign["source"]["dependency_lock_sha256"]):
        raise ValueError("paired calibration baseline source/runtime lock differs from campaign")
    expected_backend = str(freeze["runtime"]["backend"]).lower()
    if expected_backend in ("rocm", "amd", "hip"):
        expected_backend = "hip"
    if any(worker["backend"] != expected_backend
            or worker["backend_version"] != freeze["runtime"]["backend_version"]
            or worker["torch_version"] != freeze["runtime"]["torch_version"]
            for worker in campaign["worker_assignments"]):
        raise ValueError("paired calibration worker runtime differs from baseline freeze")
    validated_tuples = _paired_calibration_registered_tuples(phase0_bundle)
    if validated_tuples != campaign["calibration_tuples"]:
        raise ValueError("paired calibration tuple registry differs from frozen Phase 0 materialization")
    materialization = phase0_bundle["partition_materialization"]
    ledger = materialization.get("final_test_access_ledger")
    if (not isinstance(ledger, dict) or ledger.get("status") != "locked"
            or ledger.get("authorization_count") != 0
            or ledger.get("outcomes_read") is not False
            or ledger.get("outcomes_included") is not False
            or ledger.get("consumption_receipts") != []):
        raise ValueError("paired calibration requires the final-test access ledger to remain locked")
    approval_binding = campaign["estimand_approval"]
    _read_paired_bound_json(repo_root, approval_binding, kind="estimand approval")
    approval_path = Path(repo_root).resolve() / approval_binding["path"]
    from .meco_simulated_score_ledger import (load_estimand_approval,
        estimand_approval_digest)
    approval, raw_sha256 = load_estimand_approval(approval_path,
        source_identity_sha256=source_hash)
    if (raw_sha256 != approval_binding["artifact_sha256"]
            or estimand_approval_digest(approval) != approval_binding["approval_sha256"]):
        raise ValueError("paired calibration human approval bytes or semantic digest changed")
    return {"baseline_freeze": freeze,
        "partition": materialization, "phase0_binding": actual_phase0,
        "approval": approval,
        "source_identity_sha256": source_hash}


def bundle_identity(policy_manifest: dict) -> str:
    """Hash the exact policy checkpoint bundle used by a qualification round."""
    if not isinstance(policy_manifest, dict):
        raise ValueError("policy manifest is required")
    selected = {}
    for policy, item in policy_manifest.items():
        if policy not in POLICIES + ("D0", "D1", "D2") or not isinstance(item, dict):
            continue
        digest = item.get("checkpoint_sha256")
        if policy in POLICIES and (not isinstance(digest, str) or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"missing immutable checkpoint hash for {policy}")
        selected[policy] = {"checkpoint_sha256": digest,
            "source_identity_sha256": item.get("source_identity_sha256")}
    if not any(policy in POLICIES for policy in selected):
        raise ValueError("bundle has no learned policy checkpoint")
    return hashlib.sha256(json.dumps(selected, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def frozen_bundle_identity(checkpoint: str | Path, candidate_policy: str) -> str:
    """Compute selected checkpoint plus current deterministic-reference source identity."""
    if candidate_policy not in POLICIES:
        raise ValueError("unknown candidate policy")
    from .curriculum_training import source_identity
    sources = source_identity()
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    return bundle_identity({candidate_policy: {
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "source_identity_sha256": source_hash},
        REFERENCE_POLICY: {"checkpoint_sha256": None,
                           "source_identity_sha256": source_hash}})


def _bootstrap_ci(values, *, seed: int, replicates: int = BOOTSTRAP_REPLICATES):
    """Deterministic percentile bootstrap CI for the paired episode mean."""
    if not values or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("bootstrap requires finite episode values")
    # Small local PRNG avoids a NumPy dependency and makes the resampling
    # algorithm part of this protocol implementation.
    state = seed & ((1 << 64) - 1)
    def randbelow(limit):
        nonlocal state
        state ^= (state << 13) & ((1 << 64) - 1)
        state ^= state >> 7
        state ^= (state << 17) & ((1 << 64) - 1)
        return state % limit
    means = []
    for _ in range(replicates):
        means.append(sum(float(values[randbelow(len(values))])
                         for _ in values) / len(values))
    means.sort()
    return [means[int(.025 * (replicates - 1))],
            means[int(.975 * (replicates - 1))]]


def _cluster_bootstrap_ci(groups, *, seed: int, replicates: int = BOOTSTRAP_REPLICATES):
    """Resample layout/family clusters, then paired episodes within clusters."""
    keys = sorted(groups)
    if not keys or any(not groups[key] for key in keys):
        raise ValueError("cluster bootstrap requires populated strata")
    state = seed & ((1 << 64) - 1)
    def randbelow(limit):
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


def _device_cluster_bootstrap_ci(groups, *, seed: int, device,
                                 replicates: int = BOOTSTRAP_REPLICATES,
                                 replicate_chunk: int = 128):
    """GPU cluster-then-episode bootstrap with the v4 variable denominator.

    ``groups`` is an already paired mapping from stable cluster key to one
    finite 1-D device tensor of episode deltas. Each replicate samples the same
    number of complete clusters; for each selected cluster it samples that
    cluster's original episode count with replacement, then averages all
    sampled episode values together. No cluster macro-average is introduced.
    """
    import torch

    selected = torch.device(device)
    if selected.type != "cuda" or selected.index is None or torch.version.hip is None:
        raise ValueError("GPU cluster bootstrap requires an explicitly indexed HIP device")
    if type(seed) is not int or seed < 0 or type(replicates) is not int or replicates < 1:
        raise ValueError("GPU cluster bootstrap seed/replicate count is invalid")
    if type(replicate_chunk) is not int or replicate_chunk < 1:
        raise ValueError("GPU cluster bootstrap chunk size is invalid")
    keys = sorted(groups)
    if not keys:
        raise ValueError("GPU cluster bootstrap requires at least one cluster")
    values = []
    for key in keys:
        tensor = groups[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or
                tensor.numel() < 1 or tensor.device != selected or
                not tensor.is_floating_point()):
            raise ValueError("GPU cluster bootstrap groups must be nonempty same-device vectors")
        values.append(tensor.to(dtype=torch.float64))
    lengths = torch.tensor([value.numel() for value in values],
                           device=selected, dtype=torch.long)
    maximum = max(value.numel() for value in values)
    padded = torch.zeros((len(values), maximum), device=selected,
                         dtype=torch.float64)
    for index, value in enumerate(values):
        padded[index, :value.numel()] = value
    generator = torch.Generator(device=selected)
    generator.manual_seed(seed)
    means = []
    for start in range(0, replicates, replicate_chunk):
        count = min(replicate_chunk, replicates - start)
        cluster_index = torch.randint(len(keys), (count, len(keys)),
            device=selected, generator=generator)
        selected_lengths = lengths.index_select(0, cluster_index.reshape(-1)).reshape(
            count, len(keys))
        selected_values = padded.index_select(0, cluster_index.reshape(-1)).reshape(
            count, len(keys), maximum)
        uniforms = torch.rand((count, len(keys), maximum), device=selected,
            dtype=torch.float64, generator=generator)
        episode_index = torch.floor(
            uniforms * selected_lengths[..., None].to(torch.float64)).to(torch.long)
        sampled = selected_values.gather(2, episode_index)
        valid_episode = torch.arange(maximum, device=selected)[None, None, :] < \
            selected_lengths[..., None]
        numerator = torch.where(valid_episode, sampled,
                                torch.zeros_like(sampled)).sum(dim=(1, 2))
        denominator = selected_lengths.sum(dim=1).to(torch.float64)
        means.append(numerator / denominator)
    ordered = torch.sort(torch.cat(means)).values
    lower = int(.025 * (replicates - 1))
    upper = int(.975 * (replicates - 1))
    return ordered[lower], ordered[upper]


def _validate_case_rows(report, expected_cases, policy):
    expected = {(case.category, case.layout_id, case.defender_family,
                 case.start_condition, case.sensor_profile, seed)
                for case in expected_cases for seed in case.seeds}
    rows = [row for row in report.get("records", ())
            if isinstance(row, dict) and row.get("policy") == policy]
    found = {}
    expected_scenario = expected_cases[0].scenario_id if expected_cases else None
    for row in rows:
        if row.get("scenario_id") != expected_scenario:
            raise ValueError("qualification record scenario identity mismatch")
        key = (row.get("category"), row.get("layout_id"),
               row.get("defender_family"), row.get("start_condition"),
               row.get("sensor_profile"), row.get("seed"))
        if key not in expected or key in found:
            raise ValueError(f"unexpected or duplicate qualification row: {key}")
        if row.get("split") != report.get("split") or row.get("protocol_id") != report.get("protocol_id"):
            raise ValueError("qualification record split/protocol provenance mismatch")
        manifest = report.get("policy_manifest", {}).get(policy, {})
        if (row.get("policy_id") != manifest.get("policy_id")
                or row.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
                or row.get("source_identity_sha256") != report.get("source_identity_sha256")):
            raise ValueError("qualification record policy/checkpoint/source provenance mismatch")
        if (row.get("prediction_mode") != manifest.get("prediction_mode")
                or row.get("prediction_contract") != manifest.get("prediction_contract")):
            raise ValueError("qualification row prediction mode/contract differs from policy manifest")
        metrics = row.get("metrics")
        safety_keys = ({key for key in metrics if isinstance(key, str) and key.startswith("safety_")}
                       if isinstance(metrics, dict) else set())
        if safety_keys - SAFETY_METRIC_NAMES:
            raise ValueError("qualification row contains an unknown safety metric")
        if report.get("protocol_version") in (GPU_BOOTSTRAP_PROTOCOL_VERSION,
                CALIBRATED_PROTOCOL_VERSION,
                D0_PAIRED_REFERENCE_PROTOCOL_VERSION, LEGACY_CALIBRATED_PROTOCOL_VERSION):
            source_hash = report.get("source_identity_sha256")
            if (not isinstance(source_hash, str) or len(source_hash) != 64
                    or any(char not in "0123456789abcdef" for char in source_hash)
                    or safety_keys != SAFETY_METRIC_NAMES
                    or report.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
                    or report.get("safety_metrics_source_identity_sha256") != source_hash):
                raise ValueError("calibrated qualification row lacks exact source-bound safety metrics")
        elif report.get("protocol_version") == PROTOCOL_VERSION:
            # Historical v1 rows may predate safety-rate instrumentation.
            if safety_keys and safety_keys != SAFETY_METRIC_NAMES:
                raise ValueError("qualification row has incomplete safety metric coverage")
            if safety_keys and (report.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
                    or report.get("safety_metrics_source_identity_sha256") != report.get("source_identity_sha256")):
                raise ValueError("qualification safety metrics lack evaluator schema/source provenance")
        else:
            raise ValueError("qualification report protocol version is unsupported")
        if any(type(metrics.get(key)) not in (int, float)
               or not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1
               for key in safety_keys):
            raise ValueError("qualification safety metric must be a finite rate in [0, 1]")
        if (not isinstance(metrics, dict) or type(metrics.get("success")) not in
                (bool, int) or not math.isfinite(float(metrics["success"]))
                or not 0 <= float(metrics["success"]) <= 1
                or type(metrics.get("useful_throughput")) not in (int, float)
                or not math.isfinite(metrics["useful_throughput"])
                or metrics["useful_throughput"] < 0):
            raise ValueError("qualification row has invalid outcome metrics")
        found[key] = row
    if set(found) != expected:
        raise ValueError(f"incomplete qualification coverage: {len(found)}/{len(expected)}")
    return found


def _device_category_statistics(category, cases, candidate_rows,
        reference_rows, d0_rows, *, device, evaluation_id, protocol_id,
        gate, useful_margin, safety_margins, seed, reference_policy,
        compute_gate=True, safety_seed_map=None):
    """Aggregate one category and its paired CIs on the selected HIP device."""
    import torch

    selected = torch.device(device)
    if selected.type != "cuda" or selected.index is None or torch.version.hip is None:
        raise ValueError("v4 statistics require an explicitly indexed HIP device")
    if not (len(cases) == len(candidate_rows) == len(reference_rows) == len(d0_rows)
            and candidate_rows):
        raise ValueError("GPU category statistics require complete aligned policy rows")
    candidate_metrics = [row["metrics"] for row in candidate_rows]
    reference_metrics = [row["metrics"] for row in reference_rows]
    d0_metrics = [row["metrics"] for row in d0_rows]
    throughput = torch.tensor([row["useful_throughput"] for row in candidate_metrics],
        device=selected, dtype=torch.float64)
    reference_throughput = torch.tensor(
        [row["useful_throughput"] for row in reference_metrics],
        device=selected, dtype=torch.float64)
    success = torch.tensor([bool(row["success"]) for row in candidate_metrics],
        device=selected, dtype=torch.bool)
    paired_delta = throughput - reference_throughput
    cluster_indices = {}
    for index, case in enumerate(cases):
        cluster_indices.setdefault((case.layout_id, case.defender_family), []).append(index)
    paired_groups = {key: paired_delta.index_select(0, torch.tensor(indices,
        device=selected, dtype=torch.long)) for key, indices in cluster_indices.items()}
    lower, upper = _device_cluster_bootstrap_ci(paired_groups, seed=seed,
                                                 device=selected)
    candidate_mean = throughput.mean()
    reference_mean = reference_throughput.mean()
    success_count = success.sum()
    success_rate = success_count.to(torch.float64) / success.numel()
    verdict_tensors = {
        "success_rate_pass": success_rate >= gate.success_rate,
        "throughput_ratio_pass": ((candidate_mean >= gate.throughput_ratio * reference_mean)
            & (reference_mean > 0)
            if gate.throughput_ratio else torch.ones((), device=selected, dtype=torch.bool)),
        "minimum_useful_throughput_pass": candidate_mean >= gate.min_useful_throughput,
        "useful_gain_ci_pass": lower > float(useful_margin),
    }
    safety_tensors = {}
    safety_cis = {}
    if set(safety_margins) != SAFETY_METRIC_NAMES:
        raise ValueError("GPU v4 statistics require the complete frozen safety margin set")
    for metric, margin in sorted(safety_margins.items()):
        candidate_value = torch.tensor([row[metric] for row in candidate_metrics],
            device=selected, dtype=torch.float64)
        reference_value = torch.tensor([row[metric] for row in reference_metrics],
            device=selected, dtype=torch.float64)
        d0_value = torch.tensor([row[metric] for row in d0_metrics],
            device=selected, dtype=torch.float64)
        candidate_reference = candidate_value - reference_value
        candidate_d0 = candidate_value - d0_value
        reference_groups = {key: candidate_reference.index_select(0,
            torch.tensor(indices, device=selected, dtype=torch.long))
            for key, indices in cluster_indices.items()}
        d0_groups = {key: candidate_d0.index_select(0,
            torch.tensor(indices, device=selected, dtype=torch.long))
            for key, indices in cluster_indices.items()}
        safety_seed = (safety_seed_map or {}).get(metric, (seed ^ 0x5AFE,
                                                           seed ^ 0xD00D))[0]
        d0_safety_seed = (safety_seed_map or {}).get(metric, (seed ^ 0x5AFE,
                                                              seed ^ 0xD00D))[1]
        safety_lower, safety_upper = _device_cluster_bootstrap_ci(
            reference_groups, seed=safety_seed, device=selected)
        d0_lower, d0_upper = _device_cluster_bootstrap_ci(
            d0_groups, seed=d0_safety_seed, device=selected)
        safety_pass = (safety_upper <= float(margin)) & (d0_upper <= float(margin))
        safety_tensors[metric] = safety_pass
        safety_cis[metric] = (safety_lower, safety_upper, d0_lower, d0_upper,
                              float(margin))
    verdict_tensors["safety_pass"] = torch.stack(
        [safety_tensors[name] for name in sorted(safety_tensors)]).all()
    # Transfer only completed aggregates and Boolean decisions for report/state
    # orchestration. All means, CI selection, and numeric comparisons above are
    # device operations.
    scalar_tensors = [success_count, success_rate, candidate_mean, reference_mean,
        lower, upper, *[value for values in safety_cis.values() for value in values[:4]],
        *verdict_tensors.values(), *[safety_tensors[name]
            for name in sorted(safety_tensors)]]
    scalars = torch.stack([value.to(torch.float64) for value in scalar_tensors]
                          ).detach().cpu().tolist()
    cursor = 0
    success_count_value = int(scalars[cursor]); cursor += 1
    success_rate_value = float(scalars[cursor]); cursor += 1
    candidate_mean_value = float(scalars[cursor]); cursor += 1
    reference_mean_value = float(scalars[cursor]); cursor += 1
    lower_value, upper_value = float(scalars[cursor]), float(scalars[cursor + 1]); cursor += 2
    copied_safety = {}
    for metric in sorted(safety_cis):
        copied_safety[metric] = [float(value) for value in scalars[cursor:cursor + 4]]
        cursor += 4
    verdict_names = tuple(verdict_tensors)
    verdict_values = {name: bool(value) for name, value in
        zip(verdict_names, scalars[cursor:cursor + len(verdict_names)])}
    cursor += len(verdict_names)
    safety_values = {name: bool(value) for name, value in
        zip(sorted(safety_tensors), scalars[cursor:])}
    failures = []
    if compute_gate:
        if not verdict_values["success_rate_pass"]:
            failures.append(f"{category}: success gate failed")
        if not verdict_values["throughput_ratio_pass"]:
            failures.append(f"{category}: qualified paired throughput gate failed")
        if not verdict_values["minimum_useful_throughput_pass"]:
            failures.append(f"{category}: absolute delivery minimum failed")
        if not verdict_values["useful_gain_ci_pass"]:
            failures.append(f"{category}: paired useful-throughput CI does not exceed frozen margin")
        for metric, passed in safety_values.items():
            if not passed:
                failures.append(f"{category}: safety noninferiority CI exceeds {metric} margin")
    return {
        "category_result": {"category": category, "episodes": len(candidate_rows),
            "successes": success_count_value,
            "success_rate": success_rate_value,
            "useful_throughput": candidate_mean_value,
            "reference_throughput": reference_mean_value},
        "confidence_intervals": {"candidate_mean": candidate_mean_value,
            "reference_mean": reference_mean_value,
            "paired_difference_95_ci": [lower_value, upper_value],
            "clusters": len(cluster_indices), "episodes": len(candidate_rows),
            "cluster_keys": [list(key) for key in sorted(cluster_indices)],
            "safety_noninferiority": {
                metric: {"candidate_minus_reference_95_ci": values[:2],
                    "reference_policy": reference_policy,
                    "candidate_minus_D0_safety_floor_95_ci": values[2:4],
                    "noninferiority_margin": values[4]}
                for metric, values in copied_safety.items()}},
        "category_verdict": {**verdict_values,
            "safety_metric_pass": safety_values,
            "safety_assessment": "evaluated"},
        "failures": failures,
    }


def consume_validation_round(report: dict, *, protocol: dict,
                             state: CurriculumState,
                             retention_reports: tuple[dict, ...] = ()):
    """Validate complete paired coverage, compute clustered CIs, and promote.

    ``report`` and retention reports must be fresh qualification-run reports;
    diagnostic MECO evaluations are rejected because they lack case identity.
    Final-test reports are never accepted by this promotion entry point.
    """
    if report.get("qualification") is not True or report.get("split") != "validation":
        raise ValueError("only qualification validation reports may advance promotion")
    scenario = report.get("scenario_id")
    if scenario != protocol.get("promotion_target_scenario"):
        raise ValueError("only the frozen target scenario may advance promotion")
    _validate_v2_protocol(protocol, scenario)
    if protocol.get("protocol_version") != GPU_BOOTSTRAP_PROTOCOL_VERSION:
        raise ValueError("historical host-bootstrap protocols cannot enter a promotion streak")
    _validate_evidence_digest(report)
    _validate_report_protocol_binding(report, protocol)
    from .curriculum_training import source_identity
    current_source_hash = hashlib.sha256(json.dumps(source_identity(), sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    if protocol.get("calibration_source_identity_sha256") != current_source_hash:
        raise ValueError("calibration source identity differs from current evaluation source")
    protocol_id = protocol.get("protocol_id")
    if not isinstance(protocol_id, str) or report.get("protocol_id") != protocol_id:
        raise ValueError("frozen qualification protocol mismatch")
    if protocol.get("promotion_gates", {}).get(scenario) != asdict(SCENARIOS[scenario].gate):
        raise ValueError("qualification gate definition differs from the frozen protocol")
    if report.get("gate_definition") != protocol.get("promotion_gates", {}).get(scenario):
        raise ValueError("qualification report gate definition differs from the frozen protocol")
    expected_cases = tuple(case for case in protocol_cases(scenario, "validation",
        episodes=protocol["episodes_per_category_stratum"],
        master_seed=protocol["master_seed"]))
    expected_case_ids = [{"category": case.category, "layout_id": case.layout_id,
        "defender_family": case.defender_family, "start_condition": case.start_condition,
        "sensor_profile": case.sensor_profile} for case in expected_cases]
    if report.get("case_ids") != expected_case_ids:
        raise ValueError("qualification case coverage does not match frozen protocol")
    if report.get("bundle_id") != bundle_identity(report.get("policy_manifest")):
        raise ValueError("checkpoint bundle identity mismatch")
    if report.get("policy_manifest") != report.get("reference_report", {}).get("policy_manifest"):
        raise ValueError("paired policy/reference bundle provenance mismatch")
    candidate_policy = report.get("candidate_policy")
    if candidate_policy not in POLICIES:
        raise ValueError("qualification report names an unknown candidate policy")
    reference_policy = protocol["paired_reference_policy"]
    legacy_v2 = protocol.get("protocol_version") == LEGACY_CALIBRATED_PROTOCOL_VERSION
    expected_reference = protocol.get("paired_reference_identity")
    reference_manifest = report.get("policy_manifest", {}).get(reference_policy)
    if (not legacy_v2 and (report.get("paired_reference_identity") != expected_reference
            or not isinstance(reference_manifest, dict)
            or any(reference_manifest.get(key) != expected_reference.get(key)
                   for key in ("policy_id", "prediction_mode", "prediction_contract",
                               "checkpoint_sha256")))):
        raise ValueError("paired reference identity differs from the frozen protocol")
    per_policy = {policy: _validate_case_rows(report, expected_cases, policy)
                  for policy in (candidate_policy, reference_policy, REFERENCE_POLICY)}
    categories = []
    ci = {}
    category_verdicts = {}
    gate_failures = []
    if (report.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
            or report.get("safety_metrics_source_identity_sha256") != report.get("source_identity_sha256")):
        gate_failures.append("safety metric schema/source provenance is unavailable")
    if protocol.get("safety_gate_status") != "metrics_schema_verified":
        gate_failures.append("protocol does not authorize a verified safety metric implementation")
    if report.get("source_identity_sha256") != protocol.get("calibration_source_identity_sha256"):
        raise ValueError("qualification report source identity differs from calibrated protocol")
    selected_device = report["statistics_runtime_device"]["device"]
    for category in SCENARIOS[scenario].required_categories:
        case_rows = [case for case in expected_cases if case.category == category]
        seeds = tuple(seed for case in case_rows for seed in case.seeds)
        candidate_rows = [per_policy[candidate_policy][
            (case.category, case.layout_id, case.defender_family,
             case.start_condition, case.sensor_profile, seed)]
            for case in case_rows for seed in case.seeds]
        reference_rows = [per_policy[reference_policy][
            (case.category, case.layout_id, case.defender_family,
             case.start_condition, case.sensor_profile, seed)]
            for case in case_rows for seed in case.seeds]
        seed = int(hashlib.sha256(f"{report['evaluation_id']}/{category}".encode()).hexdigest()[:16], 16)
        margin = protocol.get("minimum_useful_gain", {}).get(category)
        if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
            raise ValueError(f"{category}: frozen useful gain margin is invalid")
        d0_rows = [per_policy[REFERENCE_POLICY][
            (case.category, case.layout_id, case.defender_family,
             case.start_condition, case.sensor_profile, seed)]
            for case in case_rows for seed in case.seeds]
        computed = _device_category_statistics(category,
            [case for case in case_rows for _ in case.seeds],
            candidate_rows, reference_rows, d0_rows, device=selected_device,
            evaluation_id=report["evaluation_id"], protocol_id=protocol_id,
            gate=SCENARIOS[scenario].gate, useful_margin=margin,
            safety_margins=protocol["safety_noninferiority_margins"], seed=seed,
            reference_policy=reference_policy)
        summary_row = computed["category_result"]
        ci[category] = computed["confidence_intervals"]
        category_verdicts[category] = computed["category_verdict"]
        gate_failures.extend(computed["failures"])
        categories.append(CategoryResult(category, summary_row["episodes"],
            summary_row["successes"], summary_row["useful_throughput"],
            summary_row["reference_throughput"], True, True, protocol_id,
            seeds, seeds))
    numeric_verdict = {"schema_version": 1,
        "verdict_type": "meco-gpu-numeric-gate-verdict-v1",
        "implementation_id": GPU_BOOTSTRAP_IMPLEMENTATION_ID,
        "verdict_scope": "qualification",
        "evaluation_id": report["evaluation_id"], "protocol_id": protocol_id,
        "runtime_device": report["statistics_runtime_device"],
        "category_verdicts": category_verdicts, "failures": gate_failures}
    if report.get("gpu_numeric_verdict") is not None and report["gpu_numeric_verdict"] != numeric_verdict:
        raise ValueError("stored GPU numeric verdict differs from recomputed device result")
    report["confidence_intervals"] = ci
    report["gpu_numeric_verdict"] = numeric_verdict
    report["evidence_sha256"] = _evidence_digest(report)
    _validate_evidence_digest(report)
    result = EvaluationResult(report["evaluation_id"], scenario, "validation",
        tuple(categories), report["bundle_id"],
        SCENARIOS[scenario].version, CURRICULUM_VERSION, protocol_id,
        numeric_verdict, protocol.get("protocol_version"))
    retention_results = tuple(_retention_result(item, protocol, report,
                                device=device, state=state)
                              for item in retention_reports)
    decision = evaluate_promotion(state, result,
                                  retention_results=retention_results)
    return decision, {"evaluation_result": asdict(result), "confidence_intervals": ci,
        "promotion": asdict(decision), "state": state.to_dict()}


def _retention_result(report, protocol, primary, *, device, state):
    """Convert a complete retention report; promotion gate enforces same bundle."""
    _validate_evidence_digest(report)
    _validate_v2_protocol(protocol, protocol.get("scenario_ids", [None])[0])
    _validate_report_protocol_binding(report, protocol)
    if (report.get("qualification") is not True or report.get("split") != "validation"
            or report.get("protocol_id") != protocol.get("protocol_id")
            or report.get("evaluation_id") != primary.get("evaluation_id")
            or report.get("bundle_id") != primary.get("bundle_id")):
        raise ValueError("retention must use the same round, frozen protocol, and checkpoint bundle")
    if report.get("source_identity_sha256") != protocol.get("calibration_source_identity_sha256"):
        raise ValueError("retention report source identity differs from calibrated protocol")
    if report.get("bundle_id") != bundle_identity(report.get("policy_manifest")):
        raise ValueError("retention checkpoint bundle identity mismatch")
    scenario = report["scenario_id"]
    if report.get("gate_definition") != protocol.get("promotion_gates", {}).get(scenario):
        raise ValueError("retention report gate definition differs from the frozen protocol")
    cases = protocol_cases(scenario, "validation",
        episodes=protocol["episodes_per_category_stratum"],
        master_seed=protocol["master_seed"])
    candidate = report["candidate_policy"]
    expected_case_ids = [{"category": c.category, "layout_id": c.layout_id,
        "defender_family": c.defender_family, "start_condition": c.start_condition,
        "sensor_profile": c.sensor_profile} for c in cases]
    if report.get("case_ids") != expected_case_ids:
        raise ValueError("retention report case coverage differs from the frozen protocol")
    records = _validate_case_rows(report, cases, candidate)
    categories = []
    verdict_by_category = {}
    numeric_failures = []
    for category in SCENARIOS[scenario].required_categories:
        rows = [row for key, row in records.items() if key[0] == category]
        seeds = tuple(row["seed"] for row in rows)
        import torch
        selected = torch.device(device)
        metrics = [row["metrics"] for row in rows]
        throughput_tensor = torch.tensor([row["useful_throughput"] for row in metrics],
            device=selected, dtype=torch.float64)
        success_tensor = torch.tensor([bool(row["success"]) for row in metrics],
            device=selected, dtype=torch.bool)
        mean_tensor = throughput_tensor.mean()
        success_count_tensor = success_tensor.sum()
        gate = SCENARIOS[scenario].gate
        success_pass_tensor = (success_count_tensor.to(torch.float64) /
                               success_tensor.numel()) >= gate.success_rate
        throughput_ratio_pass_tensor = ((mean_tensor >= gate.throughput_ratio * mean_tensor)
            & (mean_tensor > 0)
            if gate.throughput_ratio else torch.ones((), device=selected, dtype=torch.bool))
        minimum_pass_tensor = mean_tensor >= gate.min_useful_throughput
        baseline = state.qualified_throughput.get(scenario, {}).get(category)
        baseline_pass_tensor = (mean_tensor >= .9 * float(baseline)
            if baseline is not None else torch.zeros((), device=selected, dtype=torch.bool))
        copied = torch.stack((mean_tensor, success_count_tensor.to(torch.float64),
            success_pass_tensor.to(torch.float64), throughput_ratio_pass_tensor.to(torch.float64),
            minimum_pass_tensor.to(torch.float64), baseline_pass_tensor.to(torch.float64))
            ).detach().cpu().tolist()
        throughput, successes, success_pass, ratio_pass, minimum_pass, baseline_pass = copied
        status = {"success_rate_pass": bool(success_pass),
            "throughput_ratio_pass": bool(ratio_pass),
            "minimum_useful_throughput_pass": bool(minimum_pass),
            "useful_gain_ci_pass": True, "safety_pass": None,
            "safety_assessment": "not_in_retention_gate",
            "safety_metric_pass": {},
            "retention_baseline_pass": bool(baseline_pass)}
        verdict_by_category[category] = status
        if not status["success_rate_pass"]:
            numeric_failures.append(f"{category}: success gate failed")
        if not status["throughput_ratio_pass"]:
            numeric_failures.append(f"{category}: qualified paired throughput gate failed")
        if not status["minimum_useful_throughput_pass"]:
            numeric_failures.append(f"{category}: absolute delivery minimum failed")
        if not status["retention_baseline_pass"]:
            numeric_failures.append(f"{scenario}/{category}: retention throughput regression or missing baseline")
        categories.append(CategoryResult(category, len(rows),
            int(successes), float(throughput), float(throughput), True, True,
            protocol["protocol_id"], seeds, seeds))
    numeric_verdict = {"schema_version": 1,
        "verdict_type": "meco-gpu-numeric-gate-verdict-v1",
        "implementation_id": GPU_BOOTSTRAP_IMPLEMENTATION_ID,
        "verdict_scope": "retention",
        "evaluation_id": report["evaluation_id"], "protocol_id": report["protocol_id"],
        "runtime_device": report["statistics_runtime_device"],
        "category_verdicts": verdict_by_category, "failures": numeric_failures}
    return EvaluationResult(report["evaluation_id"], scenario, "validation",
        tuple(categories), report["bundle_id"], SCENARIOS[scenario].version,
        CURRICULUM_VERSION, protocol["protocol_id"], numeric_verdict,
        protocol.get("protocol_version"))


def write_immutable_report(path: str | Path, report: dict) -> str:
    """Create a report once and fail if its content-addressed destination exists."""
    destination = Path(path)
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return digest


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def run_validation_round(*, scenario_id: str, candidate_policy: str,
        checkpoints: dict[str, str], protocol: dict, state: CurriculumState,
        device: str, output: str | Path, retention_reports=(), round_id: str | None = None,
        collect_only: bool = False, split: str = "validation"):
    """Run a full frozen split suite; diagnostics use `meco_track_training evaluate`."""
    if split not in ("validation", "test"):
        raise ValueError("qualification runs support validation or final test only")
    if (split == "test" and protocol.get("protocol_version") !=
            GPU_BOOTSTRAP_PROTOCOL_VERSION):
        raise ValueError("historical host-bootstrap protocols cannot run final-test statistics")
    from .meco_track_training import evaluate_meco
    from .meco_device import require_indexed_cuda_device
    from .fuel_physics_hip import prepare_native_fuel_extension
    device = str(require_indexed_cuda_device(device,
        purpose="MECO qualification"))
    prepare_native_fuel_extension(device)
    if candidate_policy not in POLICIES or set(checkpoints) != {candidate_policy}:
        raise ValueError("qualification needs exactly the selected candidate checkpoint")
    if scenario_id not in protocol.get("scenario_ids", ()):
        raise ValueError("scenario is not in the frozen qualification protocol")
    if split == "validation" and scenario_id != protocol.get("promotion_target_scenario") and not collect_only:
        raise ValueError("validation promotion is target-only; prerequisites are retention evidence")
    _validate_v2_protocol(protocol, scenario_id)
    statistics_runtime_device = None
    if protocol.get("protocol_version") == GPU_BOOTSTRAP_PROTOCOL_VERSION:
        statistics_runtime_device = _require_calibration_gpu(device)
        if (_normalized_calibration_runtime(statistics_runtime_device) !=
                protocol["gpu_statistics_implementation"]["runtime_identity"]):
            raise ValueError("selected validation device/runtime differs from frozen GPU statistics protocol")
    if protocol.get("promotion_gates", {}).get(scenario_id) != asdict(SCENARIOS[scenario_id].gate):
        raise ValueError("protocol gate definition does not match curriculum version")
    episodes = protocol.get("episodes_per_category_stratum")
    master_seed = protocol.get("master_seed")
    if type(episodes) is not int or episodes < MIN_EPISODES or type(master_seed) is not int:
        raise ValueError("invalid frozen qualification episode/seed settings")
    protocol_body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    protocol_id = hashlib.sha256(json.dumps(protocol_body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if protocol.get("protocol_id") != protocol_id:
        raise ValueError("qualification protocol hash mismatch")
    if protocol.get("protocol_version") in (GPU_BOOTSTRAP_PROTOCOL_VERSION,
            CALIBRATED_PROTOCOL_VERSION,
            D0_PAIRED_REFERENCE_PROTOCOL_VERSION, LEGACY_CALIBRATED_PROTOCOL_VERSION):
        from .curriculum_training import source_identity
        current_source_hash = hashlib.sha256(json.dumps(source_identity(), sort_keys=True,
            allow_nan=False).encode()).hexdigest()
        if protocol.get("calibration_source_identity_sha256") != current_source_hash:
            raise ValueError("calibration source identity differs from current evaluation source")
    raw_cases = [item for item in protocol["cases"]
                 if item.get("scenario_id") == scenario_id and item.get("split") == split]
    expected = protocol_cases(scenario_id, split, episodes=episodes,
                               master_seed=master_seed)
    expected_case_ids = [{"category": case.category, "layout_id": case.layout_id,
        "defender_family": case.defender_family, "start_condition": case.start_condition,
        "sensor_profile": case.sensor_profile} for case in expected]
    raw_case_ids = [{key: row.get(key) for key in ("category", "layout_id",
        "defender_family", "start_condition", "sensor_profile")} for row in raw_cases]
    if raw_case_ids != expected_case_ids:
        raise ValueError("frozen case coverage does not match the versioned protocol")
    if split == "test":
        if scenario_id not in state.qualified:
            raise ValueError("final test requires a previously promoted scenario")
        if state.qualified_protocol.get(scenario_id) != protocol_id:
            raise ValueError("final test protocol differs from qualified protocol")
        actual_bundle = frozen_bundle_identity(checkpoints[candidate_policy], candidate_policy)
        if state.last_qualified_checkpoint.get(scenario_id) != actual_bundle:
            raise ValueError("final test requires the exact selected frozen checkpoint bundle")
    by_key = {(item.category, item.layout_id, item.defender_family,
               item.start_condition, item.sensor_profile): item for item in expected}
    if len(raw_cases) != len(expected):
        raise ValueError("frozen protocol has incomplete validation case table")
    run_id = round_id or uuid.uuid4().hex
    records, manifest, source_hash = [], None, None
    safety_schema, safety_source_hash = None, None
    have_safety_binding = False
    for row in raw_cases:
        key = (row["category"], row["layout_id"], row["defender_family"],
               row["start_condition"], row["sensor_profile"])
        case = by_key.get(key)
        if case is None or tuple(row.get("seeds", ())) != case.seeds:
            raise ValueError("case identity or exact seed list differs from frozen protocol")
        report = evaluate_meco(checkpoints, scenario_id=scenario_id,
            categories=(case.category,), episodes=len(case.seeds),
            master_seed=case.seeds[0], episode_seeds=case.seeds,
            split=split, qualification_protocol_id=protocol_id,
            qualification_case={"layout_id": case.layout_id,
                "defender_family": case.defender_family,
                "start_condition": case.start_condition,
                "sensor_profile": case.sensor_profile},
            evaluation_protocol=protocol,
            evaluation_case={"layout_id": case.layout_id,
                "defender_family": case.defender_family,
                "start_condition": case.start_condition,
                "sensor_profile": case.sensor_profile},
            evaluation_policies=(candidate_policy,),
            control_policies=tuple(dict.fromkeys((REFERENCE_POLICY,
                protocol["paired_reference_policy"]))),
            include_controls=True, device=device,
            **_reference_control_kwargs(protocol))
        if manifest is None:
            manifest = report["policy_manifest"]
            source_hash = report["source_identity_sha256"]
        elif manifest != report["policy_manifest"] or source_hash != report["source_identity_sha256"]:
            raise ValueError("checkpoint/source identity changed during qualification round")
        report_safety_schema = report.get("safety_metrics_schema")
        report_safety_source_hash = report.get("safety_metrics_source_identity_sha256")
        if protocol.get("protocol_version") in (GPU_BOOTSTRAP_PROTOCOL_VERSION,
                CALIBRATED_PROTOCOL_VERSION,
                D0_PAIRED_REFERENCE_PROTOCOL_VERSION, LEGACY_CALIBRATED_PROTOCOL_VERSION):
            if (report_safety_schema != protocol.get("safety_metrics_schema")
                    or report_safety_source_hash != protocol.get("safety_metrics_source_identity_sha256")
                    or report_safety_source_hash != protocol.get("calibration_source_identity_sha256")
                    or report.get("source_identity_sha256")
                        != protocol.get("calibration_source_identity_sha256")):
                raise ValueError("evaluator safety schema/source differs from calibrated protocol")
        if not have_safety_binding:
            safety_schema, safety_source_hash = report_safety_schema, report_safety_source_hash
            have_safety_binding = True
        elif (safety_schema != report_safety_schema
                or safety_source_hash != report_safety_source_hash):
            raise ValueError("safety schema/source identity changed during qualification round")
        records.extend(report["records"])
    bundle_id = bundle_identity(manifest)
    round_report = {"schema_version": 1, "evidence_schema_version": 2,
        "qualification": True,
        "evaluation_id": run_id, "protocol_id": protocol_id,
        "scenario_id": scenario_id, "split": split,
        "candidate_policy": candidate_policy, "bundle_id": bundle_id,
        "policy_manifest": manifest, "source_identity_sha256": source_hash,
        "gate_definition": asdict(SCENARIOS[scenario_id].gate),
        "gate_calibration_status": protocol["gate_calibration_status"],
        "protocol_version": protocol.get("protocol_version"),
        "calibration_id": protocol.get("calibration_id"),
        "calibration_ids": protocol.get("calibration_ids"),
        "calibration_seed_pools": protocol.get("calibration_seed_pools"),
        "calibration_report_sha256": protocol.get("calibration_report_sha256"),
        "retention_scenario_ids": protocol.get("retention_scenario_ids"),
        "minimum_useful_gain": protocol.get("minimum_useful_gain"),
        "useful_gain_derivation": protocol.get("useful_gain_derivation"),
        "useful_gain_calibration_basis": protocol.get("useful_gain_calibration_basis"),
        "matched_reference_variance_calibration": protocol.get("matched_reference_variance_calibration"),
        "gpu_statistics_implementation": protocol.get("gpu_statistics_implementation"),
        "statistics_runtime_device": statistics_runtime_device,
        "confidence_intervals": None, "gpu_numeric_verdict": None,
        "safety_noninferiority_margins": protocol.get("safety_noninferiority_margins"),
        "safety_gate_status": protocol.get("safety_gate_status"),
        "safety_metrics_schema": safety_schema,
        "safety_metrics_source_identity_sha256": safety_source_hash,
        "paired_reference_policy": protocol.get("paired_reference_policy"),
        "paired_reference_mode": protocol.get("paired_reference_mode"),
        "paired_reference_variant": protocol.get("paired_reference_variant"),
        "paired_reference_identity": protocol.get("paired_reference_identity"),
        "reference_selection_basis": protocol.get("reference_selection_basis"),
        "safety_floor_policy": REFERENCE_POLICY if protocol.get("protocol_version") in (
            GPU_BOOTSTRAP_PROTOCOL_VERSION, CALIBRATED_PROTOCOL_VERSION, D0_PAIRED_REFERENCE_PROTOCOL_VERSION,
            LEGACY_CALIBRATED_PROTOCOL_VERSION) else None,
        "paired": True,
        "episodes_per_category_stratum": episodes,
        "master_seed": master_seed, "records": records,
        "case_ids": [{key: row[key] for key in ("category", "layout_id",
            "defender_family", "start_condition", "sensor_profile")} for row in raw_cases],
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()}
    round_report["reference_report"] = {
        "policy_manifest": manifest,
        "paired_reference_policy": protocol.get("paired_reference_policy"),
        "paired_reference_mode": protocol.get("paired_reference_mode"),
        "paired_reference_variant": protocol.get("paired_reference_variant"),
        "paired_reference_identity": protocol.get("paired_reference_identity"),
        "reference_selection_basis": protocol.get("reference_selection_basis"),
        "safety_floor_policy": REFERENCE_POLICY if protocol.get("protocol_version") in (
            GPU_BOOTSTRAP_PROTOCOL_VERSION, CALIBRATED_PROTOCOL_VERSION, D0_PAIRED_REFERENCE_PROTOCOL_VERSION,
            LEGACY_CALIBRATED_PROTOCOL_VERSION) else None,
        "matched_reference_variance_calibration": protocol.get("matched_reference_variance_calibration")}
    round_report["evidence_sha256"] = _evidence_digest(round_report)
    if collect_only:
        return None, round_report
    if split == "test":
        candidate_map = _validate_case_rows(round_report, expected, candidate_policy)
        reference_policy = protocol["paired_reference_policy"]
        reference_map = _validate_case_rows(round_report, expected, reference_policy)
        d0_map = _validate_case_rows(round_report, expected, REFERENCE_POLICY)
        reference_manifest = manifest.get(reference_policy, {})
        if (protocol.get("protocol_version") != LEGACY_CALIBRATED_PROTOCOL_VERSION
                and any(reference_manifest.get(key) != protocol["paired_reference_identity"].get(key)
                    for key in ("policy_id", "prediction_mode", "prediction_contract", "checkpoint_sha256"))):
            raise ValueError("final-test reference identity differs from the preselected protocol")
        summary = {}
        final_test_cis = {}
        for category in SCENARIOS[scenario_id].required_categories:
            cases = [case for case in expected if case.category == category]
            keys = [(case, seed) for case in cases for seed in case.seeds]
            cand = [candidate_map[(case.category, case.layout_id,
                case.defender_family, case.start_condition, case.sensor_profile, seed)]
                for case, seed in keys]
            ref = [reference_map[(case.category, case.layout_id,
                case.defender_family, case.start_condition, case.sensor_profile, seed)]
                for case, seed in keys]
            d0 = [d0_map[(case.category, case.layout_id, case.defender_family,
                case.start_condition, case.sensor_profile, seed)] for case, seed in keys]
            if protocol.get("protocol_version") == GPU_BOOTSTRAP_PROTOCOL_VERSION:
                seed = int(hashlib.sha256(f"{run_id}/{category}".encode()).hexdigest()[:16], 16)
                safety_seed_map = {metric: (
                    int(hashlib.sha256(f"{run_id}/{category}/{metric}/ref".encode()).hexdigest()[:16], 16),
                    int(hashlib.sha256(f"{run_id}/{category}/{metric}/D0".encode()).hexdigest()[:16], 16))
                    for metric in protocol["safety_noninferiority_margins"]}
                computed = _device_category_statistics(category,
                    [case for case in cases for _ in case.seeds], cand, ref, d0,
                    device=statistics_runtime_device["device"], evaluation_id=run_id,
                    protocol_id=protocol_id, gate=SCENARIOS[scenario_id].gate,
                    useful_margin=protocol["minimum_useful_gain"][category],
                    safety_margins=protocol["safety_noninferiority_margins"],
                    seed=seed, reference_policy=reference_policy, compute_gate=False,
                    safety_seed_map=safety_seed_map)
                values = computed["category_result"]
                ci_row = computed["confidence_intervals"]
                final_test_cis[category] = ci_row
                summary[category] = {"episodes": values["episodes"],
                    "successes": values["successes"],
                    "success_rate": values["success_rate"],
                    "candidate_useful_throughput": values["useful_throughput"],
                    "reference_useful_throughput": values["reference_throughput"],
                    "paired_reference_policy": reference_policy,
                    "paired_reference_mode": protocol.get("paired_reference_mode"),
                    "paired_reference_variant": protocol.get("paired_reference_variant"),
                    "safety_noninferiority": ci_row["safety_noninferiority"],
                    "paired_difference_95_ci": ci_row["paired_difference_95_ci"]}
            else:  # Defensive assertion; old protocols are rejected at entry.
                raise ValueError("final-test statistics require the GPU bootstrap protocol")
        round_report["final_test_isolated"] = True
        round_report["final_test_results"] = summary
        round_report["confidence_intervals"] = final_test_cis
        round_report["evidence_sha256"] = _evidence_digest(round_report)
        _validate_evidence_digest(round_report)
        write_immutable_report(output, round_report)
        return None, round_report
    retention_reports = list(retention_reports)
    if not retention_reports:
        for ancestor in sorted(_ancestors(scenario_id)):
            _, ancestor_report = run_validation_round(scenario_id=ancestor,
                candidate_policy=candidate_policy, checkpoints=checkpoints,
                protocol=protocol, state=state, device=device, output=output,
                round_id=run_id, collect_only=True, split="validation")
            retention_reports.append(ancestor_report)
    round_report["retention_reports"] = retention_reports
    staged_state = CurriculumState.from_dict(state.to_dict())
    decision, summary = consume_validation_round(round_report, protocol=protocol,
        state=staged_state, retention_reports=tuple(retention_reports))
    round_report["confidence_intervals"] = summary["confidence_intervals"]
    round_report["promotion"] = summary["promotion"]
    round_report["state"] = summary["state"]
    write_immutable_report(output, round_report)
    _replace_curriculum_state(state, staged_state)
    return decision, round_report


def _replace_curriculum_state(destination: CurriculumState, source: CurriculumState):
    """Copy validated staged state only after its immutable report is durable."""
    destination.qualified = set(source.qualified)
    destination.consecutive_passes = dict(source.consecutive_passes)
    destination.last_qualified_checkpoint = dict(source.last_qualified_checkpoint)
    destination.qualified_throughput = {key: dict(value)
                                        for key, value in source.qualified_throughput.items()}
    destination.seen_evaluation_ids = set(source.seen_evaluation_ids)
    destination.qualified_protocol = dict(source.qualified_protocol)
    destination.consecutive_protocol = dict(source.consecutive_protocol)
    destination.consecutive_checkpoint = dict(source.consecutive_checkpoint)


def _load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


_STAGE1_PREFLIGHT_ARTIFACTS = (
    "capture_receipt", "effective_snapshot", "effective_freeze",
    "tuple_registry", "partition_materialization", "score_estimand_review",
    "d0_calibration_bundle", "matched_reference_calibration_bundle",
    "margin_review",
)
_STAGE1_CHAIN_ARTIFACTS = ("capture_receipt", "effective_snapshot",
    "effective_freeze", "tuple_registry", "partition_materialization")
_STAGE1_PREFLIGHT_SCHEMA = "meco-stage1-readiness-preflight-v1"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _stage1_resolved_path(value, *, label, strict=True):
    path = Path(value).expanduser()
    try:
        return path.resolve(strict=strict)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} cannot be resolved") from exc


def _stage1_inside(path, roots):
    return any(path == root or root in path.parents for root in roots)


def _stage1_json(path, *, label, canonical=False):
    try:
        with path.open("rb") as stream:
            raw = stream.read()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if canonical:
        from .meco_simulated_score_ledger import canonical_bytes
        if raw != canonical_bytes(value):
            raise ValueError(f"{label} must use canonical immutable JSON bytes")
    return value, hashlib.sha256(raw).hexdigest()


def _stage1_digest(value, label):
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _stage1_reject_protected_references(value, *, base, roots):
    """Inspect path declarations without opening the referenced files."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and (key == "path" or key.endswith("_path")):
                if isinstance(item, str) and item:
                    candidate = Path(item).expanduser()
                    if not candidate.is_absolute():
                        candidate = base / candidate
                    resolved = _stage1_resolved_path(candidate, label="referenced evidence path", strict=False)
                    if _stage1_inside(resolved, roots):
                        raise ValueError("referenced_evidence_path_protected")
            _stage1_reject_protected_references(item, base=base, roots=roots)
    elif isinstance(value, list):
        for item in value:
            _stage1_reject_protected_references(item, base=base, roots=roots)


def _stage1_save_report(destination, report):
    report["report_sha256"] = _canonical_sha256(report)
    if destination.exists():
        existing, _ = _stage1_json(destination, label="existing output")
        if existing != report:
            raise ValueError("refusing to replace a non-identical existing output")
        return report
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, destination)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return report


def _stage1_blocked_report(destination, checks, blocker_codes):
    report = {"schema": _STAGE1_PREFLIGHT_SCHEMA, "status": "blocked",
        "scope": "metadata_consistent_only", "ready": False,
        "current_source_factory_reproduction": "not_performed",
        "checkpoint_digest_semantics": "declared_metadata_only_not_verified_against_checkpoint_bytes",
        "checkpoint_bytes_opened": False, "outcomes_read": False,
        "curriculum_state_changed": False, "qualification_decision": False,
        "tracks": [{"policy_id": policy, "status": "blocked",
            "blocker_code": "preflight_input_safety_blocked"} for policy in POLICIES],
        "checks": checks, "blocker_codes": sorted(set(blocker_codes))}
    return _stage1_save_report(destination, report)


def _stage1_check_phase0_artifact(name, value, protocol):
    if not isinstance(value, dict):
        raise ValueError(f"Phase 0 {name} artifact is malformed")
    if name == "score_estimand_review":
        from .meco_simulated_score_ledger import validate_estimand_approval
        validate_estimand_approval(value,
            source_identity_sha256=protocol.get("calibration_source_identity_sha256"))
        return {"approval_id": value["approval_id"],
            "source_identity_sha256": value["simulator_source_sha256"],
            "rules_sha256": value["rules_sha256"]}
    if name == "d0_calibration_bundle":
        reports = value.get("calibration_reports")
        if value.get("schema_version") != 1 or not isinstance(reports, dict) or not reports:
            raise ValueError("D0 calibration bundle is incomplete")
        ids, _, source_hash, _ = _calibration_report_index(reports,
            _scenario_closure("pickup_simple"))
        if ids != protocol.get("calibration_ids"):
            raise ValueError("D0 calibration bundle does not match protocol calibration IDs")
        if source_hash != protocol.get("calibration_source_identity_sha256"):
            raise ValueError("D0 calibration source does not match protocol source identity")
        return {"calibration_ids": ids, "source_identity_sha256": source_hash}
    if name == "matched_reference_calibration_bundle":
        reports = value.get("calibration_reports")
        if value.get("schema_version") != 1 or not isinstance(reports, dict) or not reports:
            raise ValueError("matched-reference calibration bundle is incomplete")
        hashes = {}
        ids = {}
        for scenario, report in reports.items():
            if not isinstance(report, dict):
                raise ValueError(f"matched-reference report is malformed for {scenario}")
            _validate_matched_reference_variance_report(report)
            ids[scenario] = report["calibration_id"]
            hashes[scenario] = _canonical_sha256(report)
        embedded = protocol.get("matched_reference_variance_calibration", {})
        if (ids != embedded.get("calibration_ids")
                or hashes != embedded.get("report_sha256")):
            raise ValueError("matched-reference bundle does not match protocol calibration binding")
        return {"calibration_ids": ids, "report_sha256": hashes}
    if name == "margin_review":
        ref = protocol.get("matched_reference_variance_calibration", {})
        _validate_reference_margin_review(value, target="pickup_simple",
            reference_variant=protocol.get("paired_reference_variant"),
            reference_ids=ref.get("calibration_ids"),
            reference_hashes=ref.get("report_sha256"),
            safety_margins=protocol.get("safety_noninferiority_margins"))
        return {"review_sha256": _canonical_sha256(value)}
    raise ValueError("unknown Phase 0 artifact")


def preflight_stage1(*, protocol_path, phase0_index_path, campaign_manifest_path,
                     checkpoint_manifest_paths, protected_campaign_roots,
                     protected_checkpoint_roots, output_path):
    """Read allowlisted metadata only and emit a fail-closed Stage 1 readiness report."""
    if len(checkpoint_manifest_paths) != len(POLICIES):
        raise ValueError("supply exactly five checkpoint manifest paths")
    protected = []
    for label, values in (("campaign", protected_campaign_roots),
                          ("checkpoint", protected_checkpoint_roots)):
        if not values:
            raise ValueError(f"at least one protected {label} root is required")
        for value in values:
            root = _stage1_resolved_path(value, label=f"protected {label} root")
            if not root.is_dir():
                raise ValueError(f"protected {label} root is not a directory")
            protected.append(root)
    protected = list(dict.fromkeys(protected))

    input_values = [protocol_path, phase0_index_path, campaign_manifest_path,
        *checkpoint_manifest_paths]
    inputs = []
    caller_input_issues = {}
    for value in input_values:
        path = _stage1_resolved_path(value, label="input", strict=False)
        if _stage1_inside(path, protected):
            caller_input_issues[path] = "input_path_protected"
        elif not path.is_file():
            caller_input_issues[path] = "input_path_missing"
        inputs.append(path)
    output_candidate = Path(output_path).expanduser()
    parent = _stage1_resolved_path(output_candidate.parent, label="output parent")
    destination = (parent / output_candidate.name).resolve(strict=False)
    if _stage1_inside(destination, protected):
        raise ValueError("output resolves inside a protected campaign/checkpoint root")
    if destination in inputs:
        raise ValueError("input/output alias is forbidden")

    # Read the campaign declaration first, then expand the denylist before
    # opening any other supplied or referenced artifact.
    if _stage1_inside(inputs[2], protected) or not inputs[2].is_file():
        blocker = "campaign_manifest_path_protected" if _stage1_inside(inputs[2], protected) else "campaign_manifest_missing"
        rows = [{"check_id": "input_path", "status": "blocked",
            "evidence_path": str(path), "evidence_sha256": None,
            "blocker_code": caller_input_issues.get(path, "input_not_opened")}
            for path in inputs]
        return _stage1_blocked_report(destination, rows,
            [blocker, *[row["blocker_code"] for row in rows]])
    try:
        campaign, campaign_raw_sha = _stage1_json(inputs[2], label="campaign manifest")
    except ValueError:
        return _stage1_blocked_report(destination, [{"check_id": "campaign_manifest",
            "status": "blocked", "evidence_path": str(inputs[2]),
            "evidence_sha256": hashlib.sha256(inputs[2].read_bytes()).hexdigest(),
            "blocker_code": "campaign_manifest_json_invalid"}],
            ["campaign_manifest_json_invalid"])
    for field in ("campaign_roots", "checkpoint_roots"):
        roots = campaign.get(field)
        if not isinstance(roots, list) or not roots:
            return _stage1_blocked_report(destination, [{"check_id": "campaign_roots",
                "status": "blocked", "evidence_path": str(inputs[2]),
                "evidence_sha256": campaign_raw_sha,
                "blocker_code": "campaign_protected_roots_missing"}],
                ["campaign_protected_roots_missing"])
        for raw_root in roots:
            if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
                return _stage1_blocked_report(destination, [{"check_id": "campaign_roots",
                    "status": "blocked", "evidence_path": str(inputs[2]),
                    "evidence_sha256": campaign_raw_sha,
                    "blocker_code": "campaign_protected_root_invalid"}],
                    ["campaign_protected_root_invalid"])
            try:
                root = _stage1_resolved_path(raw_root, label=f"declared {field} root")
            except ValueError:
                raise
            if not root.is_dir():
                raise ValueError(f"declared {field} root is not a directory")
            protected.append(root)
    protected = list(dict.fromkeys(protected))
    expanded_violations = [path for path in inputs if _stage1_inside(path, protected)]
    if _stage1_inside(destination, protected):
        raise ValueError("output resolves inside a campaign-declared protected root")
    try:
        _stage1_reject_protected_references(campaign, base=inputs[2].parent, roots=protected)
    except ValueError:
        return _stage1_blocked_report(destination, [{"check_id": "campaign_references",
            "status": "blocked", "evidence_path": str(inputs[2]),
            "evidence_sha256": campaign_raw_sha,
            "blocker_code": "referenced_evidence_path_protected"}],
            ["referenced_evidence_path_protected"])
    expanded_input_issues = dict(caller_input_issues)
    expanded_input_issues.update({path: "input_path_protected" for path in expanded_violations})
    if expanded_input_issues:
        rows = [{"check_id": "input_path", "status": "blocked",
            "evidence_path": str(path), "evidence_sha256": None,
            "blocker_code": expanded_input_issues.get(path, "input_not_opened")}
            for path in inputs]
        codes = [row["blocker_code"] for row in rows if row["blocker_code"] != "input_not_opened"]
        return _stage1_blocked_report(destination, rows, codes)

    protocol_load_error = phase0_load_error = None
    try:
        protocol, protocol_raw_sha = _stage1_json(inputs[0], label="protocol")
    except ValueError:
        protocol, protocol_raw_sha, protocol_load_error = {}, hashlib.sha256(inputs[0].read_bytes()).hexdigest(), True
    try:
        phase0, phase0_sha = _stage1_json(inputs[1], label="Phase 0 evidence index")
    except ValueError:
        phase0, phase0_sha, phase0_load_error = {}, hashlib.sha256(inputs[1].read_bytes()).hexdigest(), True
    try:
        if not protocol_load_error:
            _stage1_reject_protected_references(protocol, base=inputs[0].parent, roots=protected)
        if not phase0_load_error:
            _stage1_reject_protected_references(phase0, base=inputs[1].parent, roots=protected)
    except ValueError:
        return _stage1_blocked_report(destination, [{"check_id": "input_references",
            "status": "blocked", "evidence_path": str(inputs[0]),
            "evidence_sha256": None if protocol_load_error else protocol_raw_sha,
            "blocker_code": "referenced_evidence_path_protected"}],
            ["referenced_evidence_path_protected"])
    checkpoint_paths = inputs[3:]
    checkpoint_rows = []
    checkpoint_load_errors = []
    for path in checkpoint_paths:
        try:
            checkpoint_rows.append(_stage1_json(path, label="checkpoint manifest"))
            checkpoint_load_errors.append(False)
        except ValueError:
            checkpoint_rows.append(({}, hashlib.sha256(path.read_bytes()).hexdigest()))
            checkpoint_load_errors.append(True)
    checkpoints = [row[0] for row in checkpoint_rows]
    checks = []
    blockers = []
    track_statuses = {policy: {"policy_id": policy, "status": "blocked",
        "blocker_code": "five_track_binding_not_validated"} for policy in POLICIES}

    def record(check_id, evidence_path, evidence_sha, error=None, related=None):
        row = {"check_id": check_id, "status": "blocked" if error else "passed",
            "evidence_path": str(evidence_path), "evidence_sha256": evidence_sha}
        if related:
            row["related_evidence"] = related
        if error:
            row["blocker_code"] = error
            blockers.append(error)
        checks.append(row)

    try:
        if protocol_load_error:
            raise ValueError("protocol_json_invalid")
        if protocol.get("protocol_version") != GPU_BOOTSTRAP_PROTOCOL_VERSION:
            raise ValueError("protocol_not_v5_gpu_bootstrap")
        _validate_v2_protocol(protocol, "pickup_simple")
        if protocol.get("promotion_target_scenario") != "pickup_simple":
            raise ValueError("protocol_wrong_target_scenario")
        if protocol.get("gpu_statistics_implementation", {}).get("implementation_id") != GPU_BOOTSTRAP_IMPLEMENTATION_ID:
            raise ValueError("protocol_wrong_implementation")
        record("protocol_v5", inputs[0], protocol_raw_sha)
    except Exception as exc:
        record("protocol_v5", inputs[0], protocol_raw_sha, str(exc) if str(exc) in {
            "protocol_not_v5_gpu_bootstrap", "protocol_wrong_target_scenario",
            "protocol_wrong_implementation"} else "protocol_invalid")

    try:
        if phase0_load_error:
            raise ValueError("phase0_index_json_invalid")
        artifacts = phase0.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(_STAGE1_PREFLIGHT_ARTIFACTS):
            raise ValueError("phase0_artifact_set_incomplete")
        refs = {}
        for name in _STAGE1_PREFLIGHT_ARTIFACTS:
            ref = artifacts[name]
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
                raise ValueError("phase0_artifact_reference_invalid")
            raw_path = ref["path"]
            if not isinstance(raw_path, str) or Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
                raise ValueError("phase0_artifact_path_invalid")
            path = _stage1_resolved_path(inputs[1].parent / raw_path, label="Phase 0 evidence path")
            if _stage1_inside(path, protected) or not path.is_file():
                raise ValueError("phase0_artifact_path_protected_or_missing")
            if path in inputs or path == destination:
                raise ValueError("phase0_input_output_alias")
            refs[name] = path
        if len(set(refs.values())) != len(_STAGE1_PREFLIGHT_ARTIFACTS):
            raise ValueError("phase0_artifact_reference_duplicate")
        loaded_artifacts = {}
        artifact_raw_sha256 = {}
        phase0_evidence = []
        for name, path in refs.items():
            artifact, raw_sha = _stage1_json(path, label=f"Phase 0 {name}",
                canonical=(name == "score_estimand_review"))
            _stage1_reject_protected_references(artifact, base=path.parent, roots=protected)
            if _stage1_digest(artifacts[name]["sha256"], f"{name} file digest") != raw_sha:
                raise ValueError("phase0_artifact_digest_mismatch")
            loaded_artifacts[name] = artifact
            artifact_raw_sha256[name] = raw_sha
            phase0_evidence.append({"artifact": name, "evidence_path": str(path),
                "evidence_sha256": raw_sha})
        chain = {name: loaded_artifacts[name] for name in _STAGE1_CHAIN_ARTIFACTS}
        chain_binding = validate_phase0_chain(chain)
        if chain_binding["allocation_authorized"] is not True:
            raise ValueError("phase0_allocation_authorization_not_validated")
        phase0_binding = validate_phase0_index(phase0, chain, artifact_raw_sha256)
        if (phase0_binding["source_identity_sha256"] != protocol.get("calibration_source_identity_sha256")
                or protocol.get("phase0_identity_binding") != {
                    "effective_freeze_sha256": phase0_binding["effective_freeze_sha256"],
                    "source_commit": phase0_binding["source"]["commit"],
                    "source_tree": phase0_binding["source"]["tree"],
                    "source_plan_path": phase0_binding["source"]["source_plan_path"],
                    "source_plan_sha256": phase0_binding["source"]["source_plan_sha256"],
                    "runtime_sha256": phase0_binding["runtime_sha256"],
                    "dependency_lock_sha256": phase0_binding["dependency_lock_sha256"],
                    "normalized_runtime_sha256": phase0_binding["normalized_runtime_sha256"],
                    "config_sha256": phase0_binding["config_sha256"],
                    "configuration_artifacts": phase0_binding["configuration_artifacts"],
                    "source_identity_sha256": phase0_binding["source_identity_sha256"]}):
            raise ValueError("phase0_protocol_identity_binding_mismatch")
        if campaign.get("phase0_identity_binding") != protocol.get("phase0_identity_binding"):
            raise ValueError("phase0_campaign_identity_binding_mismatch")
        phase0["_verified_stage1_seeds"] = {"validation_seeds": chain_binding["validation_seeds"],
            "final_test_seeds": chain_binding["final_test_seeds"],
            "baseline_freeze_sha256": chain_binding["effective_freeze_sha256"]}
        for name in ("score_estimand_review", "d0_calibration_bundle",
                "matched_reference_calibration_bundle", "margin_review"):
            _stage1_check_phase0_artifact(name, loaded_artifacts[name], protocol)
        review = loaded_artifacts["score_estimand_review"]
        if review.get("simulator_source_sha256") != phase0_binding["source_identity_sha256"]:
            raise ValueError("phase0_estimand_source_binding_invalid")
        if campaign.get("source_identity_sha256") != phase0_binding["source_identity_sha256"]:
            raise ValueError("phase0_campaign_source_binding_invalid")
        record("phase0_prerequisites", inputs[1], phase0_sha,
            related=phase0_evidence)
    except Exception as exc:
        record("phase0_prerequisites", inputs[1], phase0_sha,
            str(exc) if str(exc).startswith("phase0_") else "phase0_evidence_invalid")

    try:
        if any(checkpoint_load_errors):
            raise ValueError("checkpoint_manifest_json_invalid")
        declared = _stage1_validate_campaign_and_checkpoints(campaign, checkpoints, protocol, phase0)
        track_statuses = {policy: {"policy_id": policy, "status": "passed",
            "declared_checkpoint_sha256": declared[policy]} for policy in POLICIES}
        related = [{"policy_id": item.get("policy_id"),
            "evidence_path": str(path), "evidence_sha256": checkpoint_rows[index][1]}
            for index, (path, item) in enumerate(zip(checkpoint_paths, checkpoints))]
        record("five_track_binding", inputs[2], hashlib.sha256(inputs[2].read_bytes()).hexdigest(),
            related=related)
    except Exception as exc:
        known = str(exc)
        track_error = known if re.fullmatch(r"[a-z0-9_]+", known) else "campaign_or_checkpoint_binding_invalid"
        counts = {policy: sum(item.get("policy_id") == policy for item in checkpoints)
            for policy in POLICIES}
        track_statuses = {policy: {"policy_id": policy, "status": "blocked",
            "blocker_code": ("track_manifest_missing" if counts[policy] == 0 else
                "track_manifest_duplicate" if counts[policy] > 1 else track_error)}
            for policy in POLICIES}
        related = [{"policy_id": item.get("policy_id"),
            "evidence_path": str(path), "evidence_sha256": checkpoint_rows[index][1]}
            for index, (path, item) in enumerate(zip(checkpoint_paths, checkpoints))]
        record("five_track_binding", inputs[2], hashlib.sha256(inputs[2].read_bytes()).hexdigest(),
            known if re.fullmatch(r"[a-z0-9_]+", known) else "campaign_or_checkpoint_binding_invalid",
            related=related)

    report = {"schema": _STAGE1_PREFLIGHT_SCHEMA,
        "status": "ready" if not blockers else "blocked",
        "scope": "metadata_consistent_only",
        "current_source_factory_reproduction": "not_performed",
        "ready": not blockers,
        "checkpoint_digest_semantics": "declared_metadata_only_not_verified_against_checkpoint_bytes",
        "checkpoint_bytes_opened": False, "outcomes_read": False,
        "curriculum_state_changed": False, "qualification_decision": False,
        "tracks": [track_statuses[policy] for policy in POLICIES],
        "checks": checks, "blocker_codes": sorted(set(blockers))}
    report["report_sha256"] = _canonical_sha256(report)
    if destination.exists():
        existing, _ = _stage1_json(destination, label="existing output")
        if existing != report:
            raise ValueError("refusing to replace a non-identical existing output")
        return report
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, destination)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return report


def _stage1_validate_campaign_and_checkpoints(campaign, checkpoints, protocol, phase0):
    campaign_fields = {"schema", "qualification_eligible", "evaluation_split",
        "scenario_id", "scenario_version", "protocol_id", "source_identity_sha256",
        "config_sha256", "promotion_comparator", "safety_floor_policy",
        "validation_seeds", "final_test_seeds", "matched_campaign_ancestry",
        "campaign_roots", "checkpoint_roots", "phase0_identity_binding"}
    if (set(campaign) != campaign_fields
            or campaign.get("schema") != "meco-five-track-stage1-campaign-manifest-v1"):
        raise ValueError("campaign_manifest_schema_invalid")
    if campaign.get("qualification_eligible") is not True or campaign.get("evaluation_split") != "validation":
        raise ValueError("diagnostic_campaign_not_qualification_eligible")
    if (campaign.get("scenario_id") != "pickup_simple"
            or campaign.get("scenario_version") != SCENARIOS["pickup_simple"].version):
        raise ValueError("campaign_scenario_binding_invalid")
    if campaign.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("campaign_protocol_mismatch")
    source = protocol.get("calibration_source_identity_sha256")
    if campaign.get("source_identity_sha256") != source:
        raise ValueError("campaign_source_mismatch")
    binding = protocol.get("phase0_identity_binding")
    if not isinstance(binding, dict) or campaign.get("phase0_identity_binding") != binding:
        raise ValueError("campaign_phase0_identity_binding_mismatch")
    if campaign.get("promotion_comparator") != protocol.get("paired_reference_variant"):
        raise ValueError("campaign_comparator_mismatch")
    if campaign.get("safety_floor_policy") != REFERENCE_POLICY:
        raise ValueError("campaign_safety_floor_invalid")
    partition = phase0.get("_verified_stage1_seeds")
    if not isinstance(partition, dict):
        raise ValueError("phase0_partition_missing")
    validation = campaign.get("validation_seeds")
    final_test = campaign.get("final_test_seeds")
    expected_validation = sorted({seed for row in protocol.get("cases", [])
        if row.get("scenario_id") == "pickup_simple" and row.get("split") == "validation"
        for seed in row.get("seeds", [])})
    expected_test = sorted({seed for row in protocol.get("cases", [])
        if row.get("scenario_id") == "pickup_simple" and row.get("split") == "test"
        for seed in row.get("seeds", [])})
    if (not isinstance(validation, list) or not validation
            or not isinstance(final_test, list) or not final_test
            or any(type(seed) is not int for seed in validation + final_test)
            or len(set(validation)) != len(validation) or len(set(final_test)) != len(final_test)
            or set(validation) & set(final_test)):
        raise ValueError("campaign_seed_partition_invalid")
    if (validation != partition.get("validation_seeds")
            or final_test != partition.get("final_test_seeds")):
        raise ValueError("campaign_seed_partition_mismatch")
    if validation != expected_validation or final_test != expected_test:
        raise ValueError("campaign_protocol_seed_mismatch")
    ancestry = campaign.get("matched_campaign_ancestry")
    config_digest = _stage1_digest(campaign.get("config_sha256"), "campaign config digest")
    if config_digest != binding.get("config_sha256"):
        raise ValueError("campaign_phase0_config_mismatch")
    if not isinstance(ancestry, str) or not ancestry.strip():
        raise ValueError("campaign_ancestry_missing")
    found = {}
    declared_digests = set()
    for item in checkpoints:
        checkpoint_fields = {"schema", "policy_id", "checkpoint_sha256",
            "source_identity_sha256", "config_sha256", "protocol_id",
            "scenario_id", "scenario_version", "validation_seed_partition_sha256",
            "final_test_seed_partition_sha256", "matched_campaign_ancestry"}
        if (set(item) != checkpoint_fields
                or item.get("schema") != "meco-stage1-checkpoint-manifest-v1"):
            raise ValueError("checkpoint_manifest_schema_invalid")
        policy = item.get("policy_id")
        if policy not in POLICIES or policy in found:
            raise ValueError("checkpoint_policy_missing_duplicate_or_invalid")
        digest = _stage1_digest(item.get("checkpoint_sha256"), "declared checkpoint digest")
        if digest in declared_digests:
            raise ValueError("checkpoint_digest_duplicate")
        declared_digests.add(digest)
        if (item.get("source_identity_sha256") != source
                or item.get("config_sha256") != config_digest
                or item.get("protocol_id") != protocol.get("protocol_id")
                or item.get("scenario_id") != "pickup_simple"
                or item.get("scenario_version") != SCENARIOS["pickup_simple"].version
                or item.get("validation_seed_partition_sha256") != _canonical_sha256(validation)
                or item.get("final_test_seed_partition_sha256") != _canonical_sha256(final_test)
                or item.get("matched_campaign_ancestry") != ancestry):
            raise ValueError("checkpoint_binding_mismatch")
        found[policy] = digest
    if set(found) != set(POLICIES):
        raise ValueError("five_track_checkpoint_set_incomplete")
    return found


def _write_immutable_or_verify(path, payload):
    destination = Path(path)
    if destination.exists():
        existing = _load_json(destination)
        if existing != payload:
            raise ValueError(f"immutable artifact already exists with different content: {destination}")
        return _canonical_sha256(payload)
    return write_immutable_report(destination, payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="MECO frozen qualification and promotion")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight-stage1",
        help="metadata-only readiness check for the five-track pickup_simple Stage 1 gate")
    preflight.add_argument("--protocol", required=True)
    preflight.add_argument("--phase0-index", required=True)
    preflight.add_argument("--campaign-manifest", required=True)
    preflight.add_argument("--checkpoint-manifest", action="append", required=True,
        help="per-track manifest; supply exactly once for each A1, A2, A3, C1, and C2")
    preflight.add_argument("--protected-campaign-root", action="append", required=True,
        help="active campaign root; repeat for every protected campaign root")
    preflight.add_argument("--protected-checkpoint-root", action="append", required=True,
        help="active checkpoint root; repeat for every protected checkpoint root")
    preflight.add_argument("--output", required=True,
        help="new JSON report destination outside all campaign/checkpoint roots")
    create = commands.add_parser("create-protocol")
    create.add_argument("--output", required=True)
    create.add_argument("--episodes", type=int, default=MIN_EPISODES)
    create.add_argument("--master-seed", type=int, default=20261024)
    create.add_argument("--scenario")
    create.add_argument("--calibration-report")
    create.add_argument("--calibration-bundle-json")
    create.add_argument("--minimum-useful-gain-json",
        help="JSON with schema_version, per-category operational_gain_threshold, and uncertainty_multiplier")
    create.add_argument("--safety-margins-json",
        help="JSON mapping each supported safety rate to its explicit noninferiority tolerance")
    create.add_argument("--matched-reference-calibration-bundle-json",
        help="matched-reference variance reports for the exact prerequisite closure")
    create.add_argument("--reference-margin-review-json",
        help="explicit reviewed final useful-gain and safety margins bound to matched-reference variance reports")
    create.add_argument("--phase0-identity-binding-json",
        help="validated Stage 1 Phase 0 identity binding; required for pickup_simple v5")
    create.add_argument("--reference-policy",
        choices=(*MATCHED_REFERENCE_VARIANTS, "D1", "D2"),
        default="D1/current",
        help="preselect D1/current, D2/cv, or D2/multi_model_v1 before held-out evaluation")
    calibrate = commands.add_parser("calibrate-d0")
    calibrate.add_argument("--scenario", required=True)
    calibrate.add_argument("--episodes", type=int, default=MIN_EPISODES)
    calibrate.add_argument("--repetitions", type=int, default=3)
    calibrate.add_argument("--master-seed", type=int, default=20261024)
    calibrate.add_argument("--device", required=True,
        help="explicit GPU device (for example cuda:0); CPU calibration is unsupported")
    calibrate.add_argument("--output", required=True)
    ref_calibrate = commands.add_parser("calibrate-matched-reference")
    ref_calibrate.add_argument("--scenario", required=True)
    ref_calibrate.add_argument("--reference-policy", choices=(*MATCHED_REFERENCE_VARIANTS, "D1", "D2"),
        required=True)
    ref_calibrate.add_argument("--episodes", type=int, default=MIN_EPISODES)
    ref_calibrate.add_argument("--repetitions", type=int, default=3)
    ref_calibrate.add_argument("--master-seed", type=int, default=20261024)
    ref_calibrate.add_argument("--device", required=True,
        help="explicit indexed GPU, for example cuda:0")
    ref_calibrate.add_argument("--output", required=True)
    ref_bundle = commands.add_parser("bundle-matched-reference-calibrations")
    ref_bundle.add_argument("--report", action="append", required=True,
        help="matched-reference calibration report; repeat for target and each prerequisite")
    ref_bundle.add_argument("--output", required=True)
    plan_shards = commands.add_parser("plan-d0-calibration-shards")
    plan_shards.add_argument("--scenario", required=True)
    plan_shards.add_argument("--episodes", type=int, default=MIN_EPISODES)
    plan_shards.add_argument("--repetitions", type=int, default=3)
    plan_shards.add_argument("--master-seed", type=int, default=20261024)
    plan_shards.add_argument("--shard-count", type=int, required=True)
    plan_shards.add_argument("--device", required=True,
        help="explicit GPU device used to bind manifests, for example cuda:0")
    plan_shards.add_argument("--manifest-dir", required=True)
    run_shard = commands.add_parser("calibrate-d0-shard")
    run_shard.add_argument("--manifest", required=True)
    run_shard.add_argument("--device", required=True,
        help="the exact indexed GPU device bound by the manifest")
    run_shard.add_argument("--output", required=True)
    merge_shards = commands.add_parser("merge-d0-calibration-shards")
    merge_shards.add_argument("--shard", action="append", required=True,
        help="completed shard artifact; provide every shard exactly once")
    merge_shards.add_argument("--output", required=True)
    bundle = commands.add_parser("bundle-calibrations")
    bundle.add_argument("--report", action="append", required=True,
        help="D0 calibration report path; repeat once for target and each prerequisite")
    bundle.add_argument("--output", required=True)
    run = commands.add_parser("run-validation")
    run.add_argument("--scenario", required=True)
    run.add_argument("--candidate-policy", choices=POLICIES, required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--protocol", required=True)
    run.add_argument("--state", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", required=True,
        help="available CUDA or ROCm/HIP device, for example cuda:0")
    test = commands.add_parser("run-final-test")
    test.add_argument("--scenario", required=True)
    test.add_argument("--candidate-policy", choices=POLICIES, required=True)
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--protocol", required=True)
    test.add_argument("--state", required=True)
    test.add_argument("--output", required=True)
    test.add_argument("--device", required=True,
        help="available CUDA or ROCm/HIP device, for example cuda:0")
    args = parser.parse_args(argv)
    if args.command == "preflight-stage1":
        report = preflight_stage1(protocol_path=args.protocol,
            phase0_index_path=args.phase0_index,
            campaign_manifest_path=args.campaign_manifest,
            checkpoint_manifest_paths=args.checkpoint_manifest,
            protected_campaign_roots=args.protected_campaign_root,
            protected_checkpoint_roots=args.protected_checkpoint_root,
            output_path=args.output)
        print(json.dumps(report, sort_keys=True, indent=2))
        return
    if args.command == "create-protocol":
        if args.calibration_report and args.calibration_bundle_json:
            raise ValueError("use either one calibration report or a calibration bundle")
        calibration_path = args.calibration_bundle_json or args.calibration_report
        calibration = _load_json(calibration_path) if calibration_path else None
        if isinstance(calibration, dict) and "calibration_reports" in calibration:
            calibration = calibration["calibration_reports"]
        gain_margins = _load_json(args.minimum_useful_gain_json) if args.minimum_useful_gain_json else None
        safety_margins = _load_json(args.safety_margins_json) if args.safety_margins_json else None
        reference_calibration = (_load_json(args.matched_reference_calibration_bundle_json)
            if args.matched_reference_calibration_bundle_json else None)
        margin_review = (_load_json(args.reference_margin_review_json)
            if args.reference_margin_review_json else None)
        phase0_binding = (_load_json(args.phase0_identity_binding_json)
            if args.phase0_identity_binding_json else None)
        write_immutable_report(args.output, protocol_document(
            scenario_id=args.scenario, episodes=args.episodes, master_seed=args.master_seed,
            calibration_report=calibration, minimum_useful_gain=gain_margins,
            safety_noninferiority_margins=safety_margins,
            reference_policy=args.reference_policy,
            matched_reference_calibration_report=reference_calibration,
            reference_margin_review=margin_review,
            phase0_identity_binding=phase0_binding))
        return
    if args.command == "calibrate-d0":
        report = run_d0_calibration(scenario_id=args.scenario, episodes=args.episodes,
            repetitions=args.repetitions, master_seed=args.master_seed, device=args.device)
        report_digest = write_immutable_report(args.output, report)
        print(json.dumps({"calibration_id": report["calibration_id"],
            "report_sha256": report_digest, "output": args.output}, indent=2))
        return
    if args.command == "calibrate-matched-reference":
        report = run_matched_reference_variance_calibration(
            scenario_id=args.scenario, episodes=args.episodes,
            repetitions=args.repetitions, master_seed=args.master_seed,
            reference_policy=args.reference_policy, device=args.device)
        report_digest = write_immutable_report(args.output, report)
        print(json.dumps({"calibration_id": report["calibration_id"],
            "report_sha256": _canonical_sha256(report), "file_sha256": report_digest,
            "scenario_id": args.scenario,
            "reference_variant": report["config"]["reference_selection_id"],
            "output": args.output}, indent=2))
        return
    if args.command == "bundle-matched-reference-calibrations":
        indexed = {}
        common = None
        seen_seeds = set()
        for path in args.report:
            report = _load_json(path)
            _validate_matched_reference_variance_report(report)
            scenario = report.get("scenario_id")
            if not isinstance(scenario, str) or scenario in indexed:
                raise ValueError("matched reference calibration scenarios must be present and unique")
            config = report["config"]
            identity = (config["reference_selection_id"], report["source_identity_sha256"],
                report["normalized_runtime_identity"], config["master_seed"],
                config["episodes_per_category_stratum"], config["repetitions"])
            if common is not None and common != identity:
                raise ValueError("matched reference calibration bundle mixes variant/source/runtime/seed protocols")
            common = identity
            seeds = {seed for case in report["case_ids"] for seed in case["seeds"]}
            if seeds & seen_seeds:
                raise ValueError("matched reference calibration seed pools overlap across scenarios")
            seen_seeds.update(seeds)
            indexed[scenario] = report
        bundle = {"schema_version": 1, "calibration_reports": indexed}
        write_immutable_report(args.output, bundle)
        print(json.dumps({"scenario_ids": sorted(indexed),
            "reports": {scenario: {"calibration_id": report["calibration_id"],
                "report_sha256": _canonical_sha256(report)}
                for scenario, report in sorted(indexed.items())},
            "output": args.output}, indent=2))
        return
    if args.command == "plan-d0-calibration-shards":
        manifests = _d0_calibration_shard_manifests(
            scenario_id=args.scenario, episodes=args.episodes,
            repetitions=args.repetitions, master_seed=args.master_seed,
            shard_count=args.shard_count, device=args.device)
        manifest_dir = Path(args.manifest_dir)
        for manifest in manifests:
            name = f"d0-calibration-{args.scenario}-shard-{manifest['shard_index']:04d}-of-{args.shard_count:04d}.json"
            _write_immutable_or_verify(manifest_dir / name, manifest)
        print(json.dumps({"manifest_count": len(manifests),
            "work_unit_count": manifests[0]["work_unit_count"],
            "manifest_ids": [manifest["manifest_id"] for manifest in manifests],
            "manifest_dir": str(manifest_dir)}, indent=2))
        return
    if args.command == "calibrate-d0-shard":
        manifest = _load_json(args.manifest)
        artifact = run_d0_calibration_shard(manifest, device=args.device,
            output=args.output)
        print(json.dumps({"manifest_id": artifact["manifest_id"],
            "artifact_sha256": artifact["artifact_sha256"],
            "entry_count": len(artifact["entries"]), "output": args.output}, indent=2))
        return
    if args.command == "merge-d0-calibration-shards":
        artifacts = [_load_json(path) for path in args.shard]
        manifests = [artifact.get("manifest") for artifact in artifacts]
        if any(manifest is None for manifest in manifests):
            raise ValueError("D0 calibration shard artifact is missing its manifest")
        report = _merged_d0_calibration_report(manifests, artifacts)
        report_digest = write_immutable_report(args.output, report)
        print(json.dumps({"calibration_id": report["calibration_id"],
            "report_sha256": report_digest,
            "scenario_id": report["scenario_id"],
            "output": args.output}, indent=2))
        return
    if args.command == "bundle-calibrations":
        reports = [_load_json(path) for path in args.report]
        indexed = {report.get("scenario_id"): report for report in reports}
        if None in indexed or len(indexed) != len(reports):
            raise ValueError("calibration report scenarios must be present and unique")
        bundle_report = {"schema_version": 1, "calibration_reports": indexed}
        digest = write_immutable_report(args.output, bundle_report)
        print(json.dumps({"scenario_ids": sorted(indexed),
            "bundle_sha256": digest, "output": args.output}, indent=2))
        return
    protocol = _load_json(args.protocol)
    state_path = Path(args.state)
    if state_path.is_file():
        envelope = _load_json(state_path)
        if (envelope.get("schema_version") != 1
                or envelope.get("candidate_policy") != args.candidate_policy
                or not isinstance(envelope.get("state"), dict)):
            raise ValueError("qualification state belongs to a different policy or schema")
        state = CurriculumState.from_dict(envelope["state"])
    else:
        state = CurriculumState()
    if args.command == "run-final-test":
        _, report = run_validation_round(scenario_id=args.scenario,
            candidate_policy=args.candidate_policy,
            checkpoints={args.candidate_policy: args.checkpoint}, protocol=protocol,
            state=state, device=args.device, output=args.output, split="test")
        _write_dashboard_status(state_path, args.scenario, args.candidate_policy,
            report, state)
        print(json.dumps({"evaluation_id": report["evaluation_id"],
            "protocol_id": report["protocol_id"], "bundle_id": report["bundle_id"],
            "final_test_results": report["final_test_results"],
            "output": args.output}, indent=2))
        return
    state_before_round = state.to_dict()
    try:
        decision, report = run_validation_round(scenario_id=args.scenario,
            candidate_policy=args.candidate_policy,
            checkpoints={args.candidate_policy: args.checkpoint}, protocol=protocol,
            state=state, device=args.device, output=args.output,
            retention_reports=())
        _atomic_json(state_path, {"schema_version": 1,
            "candidate_policy": args.candidate_policy, "state": state.to_dict(),
            "last_round": {"evaluation_id": report["evaluation_id"],
                "scenario_id": args.scenario, "protocol_id": report["protocol_id"],
                "checkpoint_bundle_id": report["bundle_id"],
                "passed": decision.passed, "promoted": decision.promoted,
                "failures": list(decision.failures),
                "streak_reset": not decision.passed}})
    except BaseException as error:
        _replace_curriculum_state(state, CurriculumState.from_dict(state_before_round))
        state.consecutive_passes[args.scenario] = 0
        state.consecutive_protocol.pop(args.scenario, None)
        state.consecutive_checkpoint.pop(args.scenario, None)
        failure = {"scenario_id": args.scenario,
            "protocol_id": protocol.get("protocol_id"),
            "checkpoint_path": args.checkpoint, "passed": False,
            "failures": [f"round aborted: {type(error).__name__}: {error}"[:2048]],
            "streak_reset": True}
        try:
            _atomic_json(state_path, {"schema_version": 1,
                "candidate_policy": args.candidate_policy, "state": state.to_dict(),
                "last_round": failure})
        except BaseException as persistence_error:
            failure["failures"].append(
                f"state persistence failed: {type(persistence_error).__name__}: {persistence_error}")
        finally:
            try:
                _write_dashboard_status(state_path, args.scenario, args.candidate_policy,
                    {"evaluation_id": None, "protocol_id": protocol.get("protocol_id"),
                     "split": "validation", "bundle_id": None, "policy_manifest": {},
                     "gate_definition": asdict(SCENARIOS[args.scenario].gate),
                     "gate_calibration_status": protocol.get("gate_calibration_status"),
                     "protocol_version": protocol.get("protocol_version"),
                     "calibration_id": protocol.get("calibration_id"),
                     "calibration_ids": protocol.get("calibration_ids"),
                     "retention_scenario_ids": protocol.get("retention_scenario_ids"),
                     "safety_gate_status": protocol.get("safety_gate_status"),
                     "promotion": {"passed": False, "promoted": False,
                                   "failures": failure["failures"]},
                     "failure": failure,
                     "generated_utc": __import__("datetime").datetime.now(
                         __import__("datetime").timezone.utc).isoformat()}, state)
            except BaseException:
                # The original round failure remains authoritative; dashboard
                # refresh is best-effort even when state storage is unavailable.
                pass
        raise
    _write_dashboard_status(state_path, args.scenario, args.candidate_policy,
                             report, state)
    print(json.dumps({"evaluation_id": report["evaluation_id"],
        "protocol_id": report["protocol_id"], "bundle_id": report["bundle_id"],
        "promotion": asdict(decision), "output": args.output,
        "state": args.state}, indent=2))


def _write_dashboard_status(state_path, scenario, candidate_policy, report, state):
    policy_info = report.get("policy_manifest", {}).get(candidate_policy, {})
    streak = {"count": state.consecutive_passes.get(scenario, 0),
        "protocol_id": state.consecutive_protocol.get(scenario),
        "bundle_id": state.consecutive_checkpoint.get(scenario)}
    _atomic_json(Path(state_path).with_name("qualification-status.json"), {
        "schema_version": 1, "updated_utc": report["generated_utc"],
        "evaluation_id": report["evaluation_id"], "scenario_id": scenario,
        "candidate_policy": candidate_policy, "split": report["split"],
        "protocol_id": report["protocol_id"], "bundle_id": report["bundle_id"],
        "checkpoint_sha256": policy_info.get("checkpoint_sha256"),
        "gate_definition": report["gate_definition"],
        "gate_calibration_status": report["gate_calibration_status"],
        "protocol_version": report.get("protocol_version"),
        "calibration_id": report.get("calibration_id"),
        "calibration_ids": report.get("calibration_ids"),
        "calibration_seed_pools": report.get("calibration_seed_pools"),
        "retention_scenario_ids": report.get("retention_scenario_ids"),
        "minimum_useful_gain": report.get("minimum_useful_gain"),
        "safety_noninferiority_margins": report.get("safety_noninferiority_margins"),
        "safety_gate_status": report.get("safety_gate_status"),
        "confidence_intervals": report.get("confidence_intervals",
                                            report.get("final_test_results", {})),
        "promotion": report.get("promotion", {"passed": False, "promoted": False,
            "failures": ["final test does not change qualification state"]}), "streak": streak,
        "failure": report.get("failure"),
        "qualified": scenario in state.qualified,
        "final_test_isolated": report["split"] == "test"})


if __name__ == "__main__":
    main()
