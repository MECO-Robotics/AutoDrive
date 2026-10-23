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
import uuid

from .training_curriculum import (CURRICULUM_VERSION, CategoryResult,
    CurriculumState, EvaluationResult, SCENARIOS, _ancestors, evaluate_promotion,
    seed_partition)

PROTOCOL_VERSION = "meco-qualification-v1"
CALIBRATED_PROTOCOL_VERSION = "meco-qualification-v2"
MIN_EPISODES = 100
TRAIN_SEED_POOL_SIZE = 100
BOOTSTRAP_REPLICATES = 10_000
POLICIES = ("A1", "A2", "A3", "C1", "C2")
REFERENCE_POLICY = "D0"

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


def protocol_document(scenario_id: str | None = None, *, episodes: int = MIN_EPISODES,
                      master_seed: int = 20261024, calibration_report: dict | None = None,
                      minimum_useful_gain: dict[str, float] | None = None,
                      safety_noninferiority_margins: dict[str, float] | None = None) -> dict:
    """Canonical protocol document; its digest is the protocol identity."""
    scenarios = (tuple(SCENARIOS) if scenario_id is None else (scenario_id,))
    if any(scenario not in SCENARIOS for scenario in scenarios):
        raise ValueError(f"unknown scenario: {scenario_id}")
    cases = [asdict(case) | {"seeds": list(case.seeds)} for scenario in scenarios
             for split in ("train", "validation", "test")
             for case in protocol_cases(scenario, split,
                 episodes=TRAIN_SEED_POOL_SIZE if split == "train" else episodes,
                 master_seed=master_seed)]
    calibrated = calibration_report is not None
    if calibrated:
        if (calibration_report.get("calibration") is not True
                or calibration_report.get("scenario_id") not in scenarios
                or calibration_report.get("protocol_version") != "meco-d0-calibration-v1"):
            raise ValueError("calibrated protocol requires a matching D0 calibration report")
        if tuple(scenarios) != (calibration_report["scenario_id"],):
            raise ValueError("calibrated protocols are frozen one scenario at a time")
        unsigned = {key: value for key, value in calibration_report.items()
                    if key != "calibration_id"}
        calibration_id = hashlib.sha256(json.dumps(unsigned, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if calibration_report.get("calibration_id") != calibration_id:
            raise ValueError("D0 calibration report digest is invalid")
        cal_scenario = calibration_report["scenario_id"]
        cal_episodes = calibration_report.get("episodes_per_category_stratum")
        cal_repetitions = calibration_report.get("repetitions")
        cal_master_seed = calibration_report.get("master_seed")
        if (type(cal_episodes) is not int or cal_episodes < MIN_EPISODES
                or type(cal_repetitions) is not int or cal_repetitions < 2
                or type(cal_master_seed) is not int or cal_master_seed < 0):
            raise ValueError("D0 calibration report has invalid seed/repetition settings")
        cal_cases = protocol_cases(cal_scenario, "calibration", episodes=cal_episodes,
                                   master_seed=cal_master_seed)
        expected_cal_ids = [{"category": c.category, "layout_id": c.layout_id,
            "defender_family": c.defender_family, "start_condition": c.start_condition,
            "sensor_profile": c.sensor_profile} for c in cal_cases]
        if calibration_report.get("case_ids") != expected_cal_ids:
            raise ValueError("D0 calibration case coverage does not match its seed protocol")
        expected_count = sum(len(c.seeds) for c in cal_cases) * cal_repetitions
        cal_records = calibration_report.get("records")
        if not isinstance(cal_records, list) or len(cal_records) != expected_count:
            raise ValueError("D0 calibration report has incomplete raw record coverage")
        expected_rows = {(rep, c.category, c.layout_id, c.defender_family,
            c.start_condition, c.sensor_profile, seed)
            for rep in range(cal_repetitions) for c in cal_cases for seed in c.seeds}
        found_rows = set()
        for row in cal_records:
            key = (row.get("repetition"), row.get("category"), row.get("layout_id"),
                   row.get("defender_family"), row.get("start_condition"),
                   row.get("sensor_profile"), row.get("seed"))
            if key not in expected_rows or key in found_rows or row.get("policy") != REFERENCE_POLICY:
                raise ValueError("D0 calibration raw row identity is invalid or duplicated")
            if (row.get("split") != "calibration"
                    or row.get("policy_id") != "D0_production_deterministic"
                    or row.get("checkpoint_sha256") is not None
                    or row.get("source_identity_sha256") != calibration_report.get("source_identity_sha256")):
                raise ValueError("D0 calibration raw row provenance is invalid")
            metric = row.get("metrics", {}).get("useful_throughput")
            if type(metric) not in (int, float) or not math.isfinite(metric) or metric < 0:
                raise ValueError("D0 calibration raw outcome is invalid")
            found_rows.add(key)
        if found_rows != expected_rows:
            raise ValueError("D0 calibration raw seed/repetition coverage is incomplete")
        source_hash = calibration_report.get("source_identity_sha256")
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or any(char not in "0123456789abcdef" for char in source_hash)
                or calibration_report.get("policy_manifest", {}).get(REFERENCE_POLICY, {}).get(
                    "policy_id") != "D0_production_deterministic"):
            raise ValueError("D0 calibration report lacks source/controller provenance")
        expected = {category for scenario in scenarios
                    for category in SCENARIOS[scenario].required_categories}
        if (not isinstance(minimum_useful_gain, dict)
                or set(minimum_useful_gain) != expected
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or value < 0 for value in minimum_useful_gain.values())):
            raise ValueError("provide a nonnegative minimum useful-throughput gain for every category")
        if (not isinstance(safety_noninferiority_margins, dict)
                or not safety_noninferiority_margins
                or any(not isinstance(key, str) or not key.startswith("safety_")
                       or type(value) not in (int, float)
                       or not math.isfinite(value) or not 0 <= value <= 1
                       for key, value in safety_noninferiority_margins.items())):
            raise ValueError("provide nonnegative safety noninferiority margins")
    body = {"protocol_version": CALIBRATED_PROTOCOL_VERSION if calibrated else PROTOCOL_VERSION,
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
        "gate_calibration_status": ("calibrated_baseline_frozen" if calibrated
                                     else "provisional_pending_baseline_variance_review"),
        "calibration_id": calibration_report.get("calibration_id") if calibrated else None,
        "calibration_source_identity_sha256": calibration_report.get("source_identity_sha256") if calibrated else None,
        "minimum_useful_gain": minimum_useful_gain if calibrated else None,
        "safety_noninferiority_margins": safety_noninferiority_margins if calibrated else None,
        "safety_gate_status": "awaiting_runtime_contact_metric_support" if calibrated else "not_calibrated",
        "promotion_gates": {scenario: asdict(SCENARIOS[scenario].gate)
                             for scenario in scenarios},
        "cases": cases}
    body["protocol_id"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return body


def _variance(values):
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def run_d0_calibration(*, scenario_id: str, episodes: int, repetitions: int,
                       master_seed: int, device: str = "cpu") -> dict:
    """Collect repeated D0 outcomes on a dedicated, disjoint seed partition."""
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
            elif manifest != result["policy_manifest"] or source_hash != result["source_identity_sha256"]:
                raise ValueError("D0 source identity changed during calibration")
            for row in result["records"]:
                records.append({"repetition": repetition, **row})
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
    body = {"schema_version": 1, "calibration": True,
        "protocol_version": "meco-d0-calibration-v1", "scenario_id": scenario_id,
        "master_seed": master_seed, "split": "calibration",
        "seed_partition_code": 3, "episodes_per_category_stratum": episodes,
        "repetitions": repetitions, "repeat_semantics": "same exact calibration seeds repeated independently",
        "case_ids": [{"category": c.category, "layout_id": c.layout_id,
            "defender_family": c.defender_family, "start_condition": c.start_condition,
            "sensor_profile": c.sensor_profile} for c in cases],
        "policy_manifest": manifest, "source_identity_sha256": source_hash,
        "summary": summaries, "records": records,
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()}
    body["calibration_id"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
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
        metrics = row.get("metrics")
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
    protocol_id = protocol.get("protocol_id")
    if not isinstance(protocol_id, str) or report.get("protocol_id") != protocol_id:
        raise ValueError("frozen qualification protocol mismatch")
    protocol_body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    computed_protocol_id = hashlib.sha256(json.dumps(protocol_body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if computed_protocol_id != protocol_id:
        raise ValueError("qualification protocol file hash is invalid")
    calibrated = protocol.get("protocol_version") == CALIBRATED_PROTOCOL_VERSION
    if not calibrated:
        raise ValueError("provisional v1 protocol cannot advance promotion; freeze a calibrated v2 protocol")
    scenario = report.get("scenario_id")
    if protocol.get("promotion_gates", {}).get(scenario) != asdict(SCENARIOS[scenario].gate):
        raise ValueError("qualification gate definition differs from the frozen protocol")
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
    per_policy = {policy: _validate_case_rows(report, expected_cases, policy)
                  for policy in (candidate_policy, REFERENCE_POLICY)}
    categories = []
    ci = {}
    gate_failures = []
    for category in SCENARIOS[scenario].required_categories:
        case_rows = [case for case in expected_cases if case.category == category]
        seeds = tuple(seed for case in case_rows for seed in case.seeds)
        candidate_rows = [per_policy[candidate_policy][
            (case.category, case.layout_id, case.defender_family,
             case.start_condition, case.sensor_profile, seed)]
            for case in case_rows for seed in case.seeds]
        reference_rows = [per_policy[REFERENCE_POLICY][
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
            "clusters": len(groups), "episodes": len(candidate_rows),
            "cluster_keys": [list(key) for key in sorted(groups)]}
        margin = protocol.get("minimum_useful_gain", {}).get(category)
        if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
            gate_failures.append(f"{category}: frozen useful gain margin missing")
        elif ci[category]["paired_difference_95_ci"][0] <= margin:
            gate_failures.append(f"{category}: paired useful-throughput CI does not exceed frozen margin")
        safety_cis = {}
        for metric, safety_margin in protocol.get("safety_noninferiority_margins", {}).items():
            safety_groups = {key: [] for key in groups}
            valid = True
            for case, candidate, reference in zip(
                    (case for case in case_rows for _ in case.seeds),
                    candidate_rows, reference_rows):
                candidate_value = candidate["metrics"].get(metric)
                reference_value = reference["metrics"].get(metric)
                if (type(candidate_value) not in (int, float)
                        or type(reference_value) not in (int, float)
                        or not math.isfinite(candidate_value)
                        or not math.isfinite(reference_value)):
                    valid = False
                    break
                safety_groups[(case.layout_id, case.defender_family)].append(
                    float(candidate_value) - float(reference_value))
            if not valid:
                gate_failures.append(f"{category}: safety metric {metric} is absent or invalid")
                safety_cis[metric] = None
            else:
                interval = _cluster_bootstrap_ci(safety_groups, seed=seed ^ 0x5AFE)
                safety_cis[metric] = {"candidate_minus_D0_95_ci": interval,
                                      "noninferiority_margin": safety_margin}
                if interval[1] > safety_margin:
                    gate_failures.append(f"{category}: safety noninferiority CI exceeds {metric} margin")
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
    if (report.get("qualification") is not True or report.get("split") != "validation"
            or report.get("protocol_id") != protocol.get("protocol_id")
            or report.get("evaluation_id") != primary.get("evaluation_id")
            or report.get("bundle_id") != primary.get("bundle_id")):
        raise ValueError("retention must use the same round, frozen protocol, and checkpoint bundle")
    scenario = report["scenario_id"]
    cases = protocol_cases(scenario, "validation",
        episodes=protocol["episodes_per_category_stratum"],
        master_seed=protocol["master_seed"])
    candidate = report["candidate_policy"]
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
    if split not in ("validation", "test"):
        raise ValueError("qualification runs support validation or final test only")
    if candidate_policy not in POLICIES or set(checkpoints) != {candidate_policy}:
        raise ValueError("qualification needs exactly the selected candidate checkpoint")
    if scenario_id not in protocol.get("scenario_ids", ()):
        raise ValueError("scenario is not in the frozen qualification protocol")
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
    if protocol.get("protocol_version") == CALIBRATED_PROTOCOL_VERSION:
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
            evaluation_policies=(candidate_policy,), control_policies=(REFERENCE_POLICY,),
            include_controls=True, device=device)
        if manifest is None:
            manifest = report["policy_manifest"]
            source_hash = report["source_identity_sha256"]
        elif manifest != report["policy_manifest"] or source_hash != report["source_identity_sha256"]:
            raise ValueError("checkpoint/source identity changed during qualification round")
        records.extend(report["records"])
    bundle_id = bundle_identity(manifest)
    round_report = {"schema_version": 1, "qualification": True,
        "evaluation_id": run_id, "protocol_id": protocol_id,
        "scenario_id": scenario_id, "split": split,
        "candidate_policy": candidate_policy, "bundle_id": bundle_id,
        "policy_manifest": manifest, "source_identity_sha256": source_hash,
        "gate_definition": asdict(SCENARIOS[scenario_id].gate),
        "gate_calibration_status": protocol["gate_calibration_status"],
        "protocol_version": protocol.get("protocol_version"),
        "calibration_id": protocol.get("calibration_id"),
        "minimum_useful_gain": protocol.get("minimum_useful_gain"),
        "safety_noninferiority_margins": protocol.get("safety_noninferiority_margins"),
        "safety_gate_status": protocol.get("safety_gate_status"),
        "paired_reference_policy": REFERENCE_POLICY, "paired": True,
        "episodes_per_category_stratum": episodes,
        "master_seed": master_seed, "records": records,
        "case_ids": [{key: row[key] for key in ("category", "layout_id",
            "defender_family", "start_condition", "sensor_profile")} for row in raw_cases],
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()}
    round_report["reference_report"] = {"policy_manifest": manifest}
    if collect_only:
        return None, round_report
    if split == "test":
        candidate_map = _validate_case_rows(round_report, expected, candidate_policy)
        reference_map = _validate_case_rows(round_report, expected, REFERENCE_POLICY)
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
            success = sum(bool(row["metrics"]["success"]) for row in cand)
            grouped = {}
            for (case, _), a, b in zip(keys, cand, ref):
                grouped.setdefault((case.layout_id, case.defender_family), []).append(
                    float(a["metrics"]["useful_throughput"])
                    - float(b["metrics"]["useful_throughput"]))
            summary[category] = {"episodes": len(cand), "successes": success,
                "success_rate": success / len(cand),
                "candidate_useful_throughput": sum(row["metrics"]["useful_throughput"] for row in cand) / len(cand),
                "reference_useful_throughput": sum(row["metrics"]["useful_throughput"] for row in ref) / len(ref),
                "paired_difference_95_ci": _cluster_bootstrap_ci(grouped,
                    seed=int(hashlib.sha256(f"{run_id}/{category}".encode()).hexdigest()[:16], 16))}
        round_report["final_test_isolated"] = True
        round_report["final_test_results"] = summary
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="MECO frozen qualification and promotion")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-protocol")
    create.add_argument("--output", required=True)
    create.add_argument("--episodes", type=int, default=MIN_EPISODES)
    create.add_argument("--master-seed", type=int, default=20261024)
    create.add_argument("--scenario")
    create.add_argument("--calibration-report")
    create.add_argument("--minimum-useful-gain-json")
    create.add_argument("--safety-margins-json")
    calibrate = commands.add_parser("calibrate-d0")
    calibrate.add_argument("--scenario", required=True)
    calibrate.add_argument("--episodes", type=int, default=MIN_EPISODES)
    calibrate.add_argument("--repetitions", type=int, default=3)
    calibrate.add_argument("--master-seed", type=int, default=20261024)
    calibrate.add_argument("--device", default="cpu")
    calibrate.add_argument("--output", required=True)
    run = commands.add_parser("run-validation")
    run.add_argument("--scenario", required=True)
    run.add_argument("--candidate-policy", choices=POLICIES, required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--protocol", required=True)
    run.add_argument("--state", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cpu")
    test = commands.add_parser("run-final-test")
    test.add_argument("--scenario", required=True)
    test.add_argument("--candidate-policy", choices=POLICIES, required=True)
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--protocol", required=True)
    test.add_argument("--state", required=True)
    test.add_argument("--output", required=True)
    test.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "create-protocol":
        calibration = _load_json(args.calibration_report) if args.calibration_report else None
        gain_margins = _load_json(args.minimum_useful_gain_json) if args.minimum_useful_gain_json else None
        safety_margins = _load_json(args.safety_margins_json) if args.safety_margins_json else None
        write_immutable_report(args.output, protocol_document(
            scenario_id=args.scenario, episodes=args.episodes, master_seed=args.master_seed,
            calibration_report=calibration, minimum_useful_gain=gain_margins,
            safety_noninferiority_margins=safety_margins))
        return
    if args.command == "calibrate-d0":
        report = run_d0_calibration(scenario_id=args.scenario, episodes=args.episodes,
            repetitions=args.repetitions, master_seed=args.master_seed, device=args.device)
        report_digest = write_immutable_report(args.output, report)
        print(json.dumps({"calibration_id": report["calibration_id"],
            "report_sha256": report_digest, "output": args.output}, indent=2))
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
            "failures": [f"round aborted: {type(error).__name__}: {error}"],
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
