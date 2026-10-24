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
from pathlib import Path
import argparse
import re
import uuid

from .training_curriculum import (CURRICULUM_VERSION, CategoryResult,
    CurriculumState, EvaluationResult, SCENARIOS, _ancestors, evaluate_promotion,
    seed_partition)

PROTOCOL_VERSION = "meco-qualification-v1"
LEGACY_CALIBRATED_PROTOCOL_VERSION = "meco-qualification-v2"
D0_PAIRED_REFERENCE_PROTOCOL_VERSION = "meco-qualification-v3"
CALIBRATED_PROTOCOL_VERSION = "meco-qualification-v4"
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
    from .meco_tracks import prediction_contract
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
    if protocol.get("protocol_version") == CALIBRATED_PROTOCOL_VERSION:
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
    if (not isinstance(report, dict)
            or report.get("artifact_type") != MATCHED_REFERENCE_VARIANCE_VERSION
            or report.get("schema_version") != 1
            or report.get("calibration") is not True):
        raise ValueError("unsupported matched reference variance report")
    config = report.get("config")
    scenario = report.get("scenario_id")
    if not isinstance(config, dict) or config.get("scenario_id") != scenario:
        raise ValueError("matched reference variance config is invalid")
    variant = _reference_variant(config.get("reference_selection_id"))
    policy, policy_id, mode = MATCHED_REFERENCE_VARIANTS[variant]
    identity = _matched_reference_identity(variant)
    from .meco_track_training import ROUTE_COST_POINTS_PER_SECOND
    candidate_configuration = {"candidate_k": MATCHED_REFERENCE_CANDIDATE_K,
        "route_cost_points_per_second": ROUTE_COST_POINTS_PER_SECOND,
        "action_ticks": 12}
    if (config.get("reference_policy") != policy
            or config.get("reference_policy_id") != policy_id
            or config.get("prediction_mode") != mode
            or config.get("prediction_contract") != identity["prediction_contract"]
            or config.get("candidate_configuration") != candidate_configuration
            or config.get("safety_floor_policy") != REFERENCE_POLICY
            or config.get("required_safety_metrics") != sorted(SAFETY_METRIC_NAMES)
            or config.get("split") != "calibration"
            or type(config.get("episodes_per_category_stratum")) is not int
            or config["episodes_per_category_stratum"] < MIN_EPISODES
            or type(config.get("repetitions")) is not int or config["repetitions"] < 2
            or type(config.get("master_seed")) is not int or config["master_seed"] < 0):
        raise ValueError("matched reference variance policy, mode, seed, or runtime settings are invalid")
    cases = protocol_cases(scenario, "calibration",
        episodes=config["episodes_per_category_stratum"], master_seed=config["master_seed"])
    expected_cases = [{"category": case.category, "layout_id": case.layout_id,
        "defender_family": case.defender_family, "start_condition": case.start_condition,
        "sensor_profile": case.sensor_profile, "seeds": list(case.seeds),
        "reference_policy": policy} for case in cases]
    if report.get("case_ids") != expected_cases:
        raise ValueError("matched reference variance case/seed table differs from the frozen calibration protocol")
    _validate_d0_calibration_case_table(scenario, config["episodes_per_category_stratum"],
        config["master_seed"], [{key: value for key, value in row.items()
            if key != "reference_policy"} for row in expected_cases])
    source_hash = report.get("source_identity_sha256")
    runtime = report.get("runtime_device")
    normalized_runtime = _normalized_calibration_runtime(runtime)
    if (report.get("normalized_runtime_identity") != normalized_runtime
            or report.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
            or report.get("safety_metrics_source_identity_sha256") != source_hash):
        raise ValueError("matched reference variance source/runtime/safety identity is invalid")
    expected_protocol = _canonical_sha256({"version": MATCHED_REFERENCE_VARIANCE_VERSION,
        "config": config, "case_table": expected_cases,
        "source_identity_sha256": source_hash, "runtime_identity": normalized_runtime})
    if report.get("protocol_identity_sha256") != expected_protocol:
        raise ValueError("matched reference variance protocol identity is invalid")
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
                      reference_margin_review: dict | None = None) -> dict:
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
    else:
        calibration_ids, calibration_seed_pools, calibration_report_hashes, source_hash, baseline_id = {}, {}, {}, None, None
        reference_reports, reference_ids, reference_hashes = {}, {}, {}
        reviewed, reference_variance_calibration = None, None
    body = {"protocol_version": CALIBRATED_PROTOCOL_VERSION if calibrated else PROTOCOL_VERSION,
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
        body["safety_metrics_schema"] = SAFETY_METRICS_SCHEMA
        body["safety_metrics_source_identity_sha256"] = source_hash
    body["protocol_id"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return body


def _validate_v2_protocol(protocol: dict, scenario: str) -> None:
    """Validate a target-scoped calibrated protocol for target or retention use."""
    if not isinstance(protocol, dict) or protocol.get("protocol_version") not in (
            CALIBRATED_PROTOCOL_VERSION, D0_PAIRED_REFERENCE_PROTOCOL_VERSION,
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
    if protocol.get("protocol_version") == CALIBRATED_PROTOCOL_VERSION:
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
        if protocol.get("protocol_version") == CALIBRATED_PROTOCOL_VERSION:
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
    "paired", "episodes_per_category_stratum",
    "master_seed", "records", "case_ids", "reference_report", "generated_utc",
    "final_test_isolated", "final_test_results")


_LEGACY_EVIDENCE_FIELDS = tuple(key for key in _EVIDENCE_FIELDS
    if key not in ("evidence_schema_version", "final_test_isolated", "final_test_results",
                   "useful_gain_calibration_basis", "paired_reference_identity",
                   "paired_reference_mode", "paired_reference_variant",
                   "reference_selection_basis", "safety_floor_policy",
                   "matched_reference_variance_calibration"))
_D0_PAIRED_EVIDENCE_FIELDS = tuple(key for key in _EVIDENCE_FIELDS
    if key != "matched_reference_variance_calibration")


def _evidence_digest(report: dict) -> str:
    version = report.get("evidence_schema_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("qualification evidence schema version is unsupported")
    if report.get("protocol_version") in (CALIBRATED_PROTOCOL_VERSION,
            D0_PAIRED_REFERENCE_PROTOCOL_VERSION) and version != 2:
        raise ValueError("calibrated v3/v4 reports require evidence schema v2")
    fields = (_LEGACY_EVIDENCE_FIELDS if version == 1 else
        _D0_PAIRED_EVIDENCE_FIELDS if report.get("protocol_version") == D0_PAIRED_REFERENCE_PROTOCOL_VERSION
        else _EVIDENCE_FIELDS)
    evidence = {key: report.get(key) for key in fields}
    return hashlib.sha256(json.dumps(evidence, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _validate_evidence_digest(report: dict) -> None:
    if (not isinstance(report, dict)
            or type(report.get("evidence_schema_version", 1)) is not int
            or report.get("evidence_schema_version", 1) not in (1, 2)
            or (report.get("protocol_version") in (CALIBRATED_PROTOCOL_VERSION,
                D0_PAIRED_REFERENCE_PROTOCOL_VERSION)
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
    if protocol.get("protocol_version") == CALIBRATED_PROTOCOL_VERSION:
        bindings += ("matched_reference_variance_calibration",)
    if any(report.get(field) != protocol.get(field) for field in bindings):
        raise ValueError("qualification report calibration/protocol provenance mismatch")
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
            or (report.get("protocol_version") == CALIBRATED_PROTOCOL_VERSION
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


MATCHED_REFERENCE_VARIANCE_VERSION = "meco-matched-reference-variance-v1"
MATCHED_REFERENCE_CANDIDATE_K = 16


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
    from .meco_track_training import ROUTE_COST_POINTS_PER_SECOND
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
        if report.get("protocol_version") in (CALIBRATED_PROTOCOL_VERSION,
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
    gate_failures = []
    if (report.get("safety_metrics_schema") != SAFETY_METRICS_SCHEMA
            or report.get("safety_metrics_source_identity_sha256") != report.get("source_identity_sha256")):
        gate_failures.append("safety metric schema/source provenance is unavailable")
    if protocol.get("safety_gate_status") != "metrics_schema_verified":
        gate_failures.append("protocol does not authorize a verified safety metric implementation")
    if report.get("source_identity_sha256") != protocol.get("calibration_source_identity_sha256"):
        raise ValueError("qualification report source identity differs from calibrated protocol")
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
        successes = sum(int(bool(row["metrics"]["success"])) for row in candidate_rows)
        candidate_rate = sum(float(row["metrics"]["useful_throughput"])
                             for row in candidate_rows) / len(candidate_rows)
        reference_rate = sum(float(row["metrics"]["useful_throughput"])
                             for row in reference_rows) / len(reference_rows)
        # Paired, stratified bootstrap: resample complete (family, layout)
        # clusters, then episodes within each selected cluster.
        groups = {}
        for case, candidate, reference in zip(
                (case for case in case_rows for _ in case.seeds),
                candidate_rows, reference_rows):
            key = (case.layout_id, case.defender_family)
            groups.setdefault(key, []).append((candidate, reference))
        paired_by_cluster = {}
        for key, pairs in groups.items():
            paired_by_cluster[key] = [float(a["metrics"]["useful_throughput"])
                - float(b["metrics"]["useful_throughput"]) for a, b in pairs]
        seed = int(hashlib.sha256(f"{report['evaluation_id']}/{category}".encode()).hexdigest()[:16], 16)
        ci[category] = {"candidate_mean": candidate_rate,
            "reference_mean": reference_rate,
            "paired_difference_95_ci": _cluster_bootstrap_ci(paired_by_cluster, seed=seed),
            "paired_reference_policy": reference_policy,
            "clusters": len(groups), "episodes": len(candidate_rows),
            "cluster_keys": [list(key) for key in sorted(groups)]}
        margin = protocol.get("minimum_useful_gain", {}).get(category)
        if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
            gate_failures.append(f"{category}: frozen useful gain margin missing")
        elif ci[category]["paired_difference_95_ci"][0] <= margin:
            gate_failures.append(f"{category}: paired useful-throughput CI does not exceed frozen margin")
        safety_cis = {}
        d0_rows = [per_policy[REFERENCE_POLICY][
            (case.category, case.layout_id, case.defender_family,
             case.start_condition, case.sensor_profile, seed)]
            for case in case_rows for seed in case.seeds]
        for metric, safety_margin in protocol.get("safety_noninferiority_margins", {}).items():
            safety_groups = {key: [] for key in groups}
            d0_safety_groups = {key: [] for key in groups}
            valid = True
            for case, candidate, reference, d0 in zip(
                    (case for case in case_rows for _ in case.seeds),
                    candidate_rows, reference_rows, d0_rows):
                candidate_value = candidate["metrics"].get(metric)
                reference_value = reference["metrics"].get(metric)
                d0_value = d0["metrics"].get(metric)
                if (type(candidate_value) not in (int, float)
                        or type(reference_value) not in (int, float)
                        or type(d0_value) not in (int, float)
                        or not math.isfinite(candidate_value)
                        or not math.isfinite(reference_value)
                        or not math.isfinite(d0_value)):
                    valid = False
                    break
                cluster = (case.layout_id, case.defender_family)
                safety_groups[cluster].append(
                    float(candidate_value) - float(reference_value))
                d0_safety_groups[cluster].append(float(candidate_value) - float(d0_value))
            if not valid:
                gate_failures.append(f"{category}: safety metric {metric} is absent or invalid")
                safety_cis[metric] = None
            else:
                interval = _cluster_bootstrap_ci(safety_groups, seed=seed ^ 0x5AFE)
                d0_interval = _cluster_bootstrap_ci(d0_safety_groups,
                    seed=seed ^ 0xD00D)
                safety_cis[metric] = {
                    "candidate_minus_reference_95_ci": interval,
                    "reference_policy": reference_policy,
                    "candidate_minus_D0_safety_floor_95_ci": d0_interval,
                    "noninferiority_margin": safety_margin}
                if interval[1] > safety_margin:
                    gate_failures.append(f"{category}: safety noninferiority CI exceeds {metric} margin")
                if d0_interval[1] > safety_margin:
                    gate_failures.append(f"{category}: D0 safety floor CI exceeds {metric} margin")
        ci[category]["safety_noninferiority"] = safety_cis
        categories.append(CategoryResult(category, len(candidate_rows), successes,
            candidate_rate, reference_rate, True, True, protocol_id,
            seeds, seeds))
    result = EvaluationResult(report["evaluation_id"], scenario, "validation",
        tuple(categories), report["bundle_id"],
        SCENARIOS[scenario].version, CURRICULUM_VERSION, protocol_id)
    retention_results = tuple(_retention_result(item, protocol, report)
                              for item in retention_reports)
    decision = evaluate_promotion(state, result,
                                  retention_results=retention_results,
                                  additional_failures=gate_failures)
    return decision, {"evaluation_result": asdict(result), "confidence_intervals": ci,
        "promotion": asdict(decision), "state": state.to_dict()}


def _retention_result(report, protocol, primary):
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
    for category in SCENARIOS[scenario].required_categories:
        rows = [row for key, row in records.items() if key[0] == category]
        seeds = tuple(row["seed"] for row in rows)
        throughput = sum(row["metrics"]["useful_throughput"] for row in rows) / len(rows)
        categories.append(CategoryResult(category, len(rows),
            sum(bool(row["metrics"]["success"]) for row in rows),
            throughput, throughput, True, True, protocol["protocol_id"], seeds, seeds))
    return EvaluationResult(report["evaluation_id"], scenario, "validation",
        tuple(categories), report["bundle_id"], SCENARIOS[scenario].version,
        CURRICULUM_VERSION, protocol["protocol_id"])


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
    from .meco_track_training import evaluate_meco
    from .meco_device import require_indexed_cuda_device
    from .fuel_physics_hip import prepare_native_fuel_extension
    device = str(require_indexed_cuda_device(device,
        purpose="MECO qualification"))
    prepare_native_fuel_extension(device)
    if split not in ("validation", "test"):
        raise ValueError("qualification runs support validation or final test only")
    if candidate_policy not in POLICIES or set(checkpoints) != {candidate_policy}:
        raise ValueError("qualification needs exactly the selected candidate checkpoint")
    if scenario_id not in protocol.get("scenario_ids", ()):
        raise ValueError("scenario is not in the frozen qualification protocol")
    if split == "validation" and scenario_id != protocol.get("promotion_target_scenario") and not collect_only:
        raise ValueError("validation promotion is target-only; prerequisites are retention evidence")
    _validate_v2_protocol(protocol, scenario_id)
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
    if protocol.get("protocol_version") in (CALIBRATED_PROTOCOL_VERSION,
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
        if protocol.get("protocol_version") in (CALIBRATED_PROTOCOL_VERSION,
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
            CALIBRATED_PROTOCOL_VERSION, D0_PAIRED_REFERENCE_PROTOCOL_VERSION,
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
            CALIBRATED_PROTOCOL_VERSION, D0_PAIRED_REFERENCE_PROTOCOL_VERSION,
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
            success = sum(bool(row["metrics"]["success"]) for row in cand)
            grouped = {}
            for (case, _), a, b in zip(keys, cand, ref):
                grouped.setdefault((case.layout_id, case.defender_family), []).append(
                    float(a["metrics"]["useful_throughput"])
                    - float(b["metrics"]["useful_throughput"]))
            safety_summary = {}
            for metric, margin in protocol.get("safety_noninferiority_margins", {}).items():
                reference_diffs = {}
                d0_diffs = {}
                for (case, _), candidate_row, reference_row, d0_row in zip(keys, cand, ref, d0):
                    key = (case.layout_id, case.defender_family)
                    reference_diffs.setdefault(key, []).append(
                        candidate_row["metrics"][metric] - reference_row["metrics"][metric])
                    d0_diffs.setdefault(key, []).append(
                        candidate_row["metrics"][metric] - d0_row["metrics"][metric])
                safety_summary[metric] = {
                    "candidate_minus_reference_95_ci": _cluster_bootstrap_ci(reference_diffs,
                        seed=int(hashlib.sha256(f"{run_id}/{category}/{metric}/ref".encode()).hexdigest()[:16], 16)),
                    "candidate_minus_D0_safety_floor_95_ci": _cluster_bootstrap_ci(d0_diffs,
                        seed=int(hashlib.sha256(f"{run_id}/{category}/{metric}/D0".encode()).hexdigest()[:16], 16)),
                    "noninferiority_margin": margin}
            summary[category] = {"episodes": len(cand), "successes": success,
                "success_rate": success / len(cand),
                "candidate_useful_throughput": sum(row["metrics"]["useful_throughput"] for row in cand) / len(cand),
                "reference_useful_throughput": sum(row["metrics"]["useful_throughput"] for row in ref) / len(ref),
                "paired_reference_policy": reference_policy,
                "paired_reference_mode": protocol.get("paired_reference_mode"),
                "paired_reference_variant": protocol.get("paired_reference_variant"),
                "safety_noninferiority": safety_summary,
                "paired_difference_95_ci": _cluster_bootstrap_ci(grouped,
                    seed=int(hashlib.sha256(f"{run_id}/{category}".encode()).hexdigest()[:16], 16))}
        round_report["final_test_isolated"] = True
        round_report["final_test_results"] = summary
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
        write_immutable_report(args.output, protocol_document(
            scenario_id=args.scenario, episodes=args.episodes, master_seed=args.master_seed,
            calibration_report=calibration, minimum_useful_gain=gain_margins,
            safety_noninferiority_margins=safety_margins,
            reference_policy=args.reference_policy,
            matched_reference_calibration_report=reference_calibration,
            reference_margin_review=margin_review))
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
