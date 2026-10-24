"""Compile exact validation-only inputs for the frozen MECO matrix."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .meco_ablation_matrix import (
    DEFAULT_EVALUATION_STRATUM,
    default_conditions,
)
from .meco_track_training import load_evaluation_protocol
from .meco_tracks import MECO_TRACKS
from .meco_qualification import MIN_EPISODES, protocol_cases
from .training_curriculum import SCENARIOS


PLAN_SCHEMA = "meco-validation-campaign-plan-v1"
SEED_MANIFEST_SCHEMA = "meco-exact-seed-manifest-v1"
CASE_FIELDS = tuple(DEFAULT_EVALUATION_STRATUM)
BASE_POLICIES = (*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _artifact_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_exact_validation_cases(protocol: dict) -> None:
    episodes = protocol.get("episodes_per_category_stratum")
    master_seed = protocol.get("master_seed")
    if type(episodes) is not int or episodes < MIN_EPISODES:
        raise ValueError("protocol validation episode count is invalid")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("protocol master seed is invalid")
    expected = [asdict(case) for scenario in SCENARIOS
        for case in protocol_cases(scenario, "validation", episodes=episodes,
                                   master_seed=master_seed)]
    expected_rows = Counter(_canonical_bytes(row) for row in expected)
    observed_rows = Counter(_canonical_bytes(row) for row in protocol.get("cases", [])
        if isinstance(row, dict) and row.get("split") == "validation")
    if observed_rows != expected_rows:
        raise ValueError("protocol validation cases differ from the complete deterministic scenario/stratum/category/seed schedule")


def _load_protocol(path: str | Path, *, qualification_claim: bool):
    protocol_path = Path(path).expanduser().resolve()
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    if not isinstance(protocol, dict):
        raise ValueError("qualification protocol must be a JSON object")
    protocol_sha256 = hashlib.sha256(raw).hexdigest()
    scenarios = tuple(SCENARIOS)
    if protocol.get("scenario_ids") != list(scenarios):
        raise ValueError("validation campaign requires a frozen protocol covering all registered scenarios")
    if qualification_claim and (
            protocol.get("protocol_version") != "meco-qualification-v2" or
            protocol.get("gate_calibration_status") != "calibrated_baseline_frozen" or
            protocol.get("safety_gate_status") != "metrics_schema_verified"):
        raise ValueError("qualification claim requires a calibrated v2 protocol with frozen safety gates")
    if protocol.get("gate_calibration_status") == "provisional_pending_baseline_variance_review" and qualification_claim:
        raise ValueError("provisional protocol cannot support a qualification claim")
    expected_id = _sha({key: value for key, value in protocol.items()
                        if key != "protocol_id"})
    if protocol.get("protocol_id") != expected_id:
        raise ValueError("qualification protocol ID does not match its contents")
    _validate_exact_validation_cases(protocol)
    # The shared loader validates protocol schema, exact seeds and split disjointness.
    for scenario in scenarios:
        loaded, protocol_id, _, _ = load_evaluation_protocol(protocol,
            scenario_id=scenario)
        if loaded != protocol or protocol_id != expected_id:
            raise ValueError("qualification protocol validation was inconsistent across scenarios")
    return protocol, protocol_path.name, protocol_sha256, expected_id


def _validation_groups(protocol: dict) -> list[dict]:
    by_stratum: dict[tuple, dict[str, tuple[int, ...]]] = {}
    case_rows = protocol.get("cases")
    if not isinstance(case_rows, list):
        raise ValueError("qualification protocol case table is missing")
    known_scenarios = set(SCENARIOS)
    for row in case_rows:
        if not isinstance(row, dict):
            raise ValueError("qualification protocol contains a malformed case")
        scenario = row.get("scenario_id")
        if scenario not in known_scenarios:
            raise ValueError("qualification protocol contains an unknown scenario")
        if row.get("split") != "validation":
            continue
        category = row.get("category")
        if category not in SCENARIOS[scenario].required_categories:
            raise ValueError("qualification protocol contains an unknown validation category")
        values = tuple(row.get(field) for field in CASE_FIELDS)
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("validation case stratum is incomplete")
        seeds = row.get("seeds")
        if (not isinstance(seeds, list) or not seeds or
                any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds) or
                len(seeds) != len(set(seeds))):
            raise ValueError("validation case seeds must be unique signed-63-bit integers")
        key = (scenario, *values)
        category_map = by_stratum.setdefault(key, {})
        if category in category_map:
            raise ValueError("qualification protocol has duplicate validation case identity")
        category_map[category] = tuple(seeds)

    expected_categories = {name: tuple(definition.required_categories)
                           for name, definition in SCENARIOS.items()}
    groups = []
    seen_seed_owner: dict[int, tuple] = {}
    for scenario in SCENARIOS:
        scenario_keys = sorted((key for key in by_stratum if key[0] == scenario),
                               key=lambda key: key[1:])
        if not scenario_keys:
            raise ValueError(f"qualification protocol has no validation cases for {scenario}")
        for key in scenario_keys:
            category_map = by_stratum[key]
            if tuple(category for category in expected_categories[scenario]
                     if category in category_map) != expected_categories[scenario]:
                raise ValueError(f"validation category coverage is incomplete for {scenario}/{key[1:]}")
            seed_lists = {category_map[category] for category in expected_categories[scenario]}
            if len(seed_lists) != 1:
                raise ValueError("categories may be grouped only when their exact validation seed lists match")
            seeds = next(iter(seed_lists))
            owner = key
            for seed in seeds:
                previous = seen_seed_owner.setdefault(seed, owner)
                if previous != owner:
                    raise ValueError("validation seed is reused across distinct scenario/stratum cases")
            groups.append({"scenario_id": scenario,
                "qualification_case": dict(zip(CASE_FIELDS, key[1:])),
                "categories": list(expected_categories[scenario]),
                "seed_list": list(seeds)})
    return groups


def _condition_policies(condition_name: str) -> tuple[str, ...]:
    if condition_name == "inference_disabled_fallback_current_k16":
        return (*BASE_POLICIES,
                *(f"{track}__INFERENCE_DISABLED_TO_D1" for track in MECO_TRACKS))
    return BASE_POLICIES


def compile_validation_campaign(*, protocol_path: str | Path,
        checkpoint_paths: dict[str, str | Path] | None = None,
        source_identity: dict | None = None,
        training_configuration_sha256: dict[str, str] | None = None,
        qualification_claim: bool = False) -> dict:
    """Build a deterministic validation plan; never exports final-test seeds."""
    protocol, protocol_file, protocol_sha256, protocol_id = _load_protocol(
        protocol_path, qualification_claim=qualification_claim)
    groups = _validation_groups(protocol)
    checkpoints = None
    if checkpoint_paths is not None:
        if set(checkpoint_paths) != set(MECO_TRACKS):
            raise ValueError(f"checkpoint paths must cover exactly {MECO_TRACKS}")
        checkpoints = {track: _sha_file(Path(checkpoint_paths[track]))
                       for track in MECO_TRACKS}
    if training_configuration_sha256 is not None:
        if set(training_configuration_sha256) != set(MECO_TRACKS):
            raise ValueError("training configuration hashes must cover all five tracks")
        training_configuration_sha256 = {
            track: _validate_sha(training_configuration_sha256[track],
                                 f"{track} training configuration hash")
            for track in MECO_TRACKS}
    source_binding = None
    if source_identity is not None:
        if not isinstance(source_identity, dict) or not source_identity:
            raise ValueError("source identity must be a nonempty JSON object")
        source_binding = {"identity": source_identity,
                          "sha256": _sha(source_identity)}

    jobs = []
    for group in groups:
        scenario = group["scenario_id"]
        categories = tuple(group["categories"])
        qualification_case = group["qualification_case"]
        seeds = tuple(group["seed_list"])
        seed_manifest = {"schema": SEED_MANIFEST_SCHEMA,
            "protocol_id": protocol_id,
            "protocol_sha256": protocol_sha256,
            "scenario_id": scenario,
            "categories": list(categories),
            "split": "validation",
            "qualification_case": qualification_case,
            "seed_list": list(seeds)}
        seed_manifest_sha256 = hashlib.sha256(_artifact_bytes(seed_manifest)).hexdigest()
        seed_manifest_file = ("seed-manifests/" + scenario + "-" +
                              _sha(qualification_case)[:16] + ".json")
        conditions = default_conditions(scenario, categories=categories,
                                        qualification_case=qualification_case)
        planned_conditions = []
        for condition in conditions:
            condition_data = asdict(condition)
            condition_categories = list(condition.categories or categories)
            policies = list(_condition_policies(condition.condition_name))
            keys = sorted((policy, category, seed) for policy in policies
                          for category in condition_categories for seed in seeds)
            condition_data.update({
                "expected_coverage": {
                    "policies": policies,
                    "categories": condition_categories,
                    "seed_list": list(seeds),
                    "record_count": len(keys),
                    "policy_category_seed_keys_sha256": hashlib.sha256(
                        _canonical_bytes([list(key) for key in keys])).hexdigest()},
                "replay_selection": {},
            })
            if condition.condition_name == "architecture_matched_cv_k16":
                replay_policies = list((*MECO_TRACKS, "D0", "D1", "D2"))
                replay_seed = seeds[0]
                selection = {policy: [replay_seed] for policy in replay_policies}
                condition_data["replay_selection"] = selection
                condition_data["expected_replay_keys"] = sorted(
                    [policy, category, replay_seed]
                    for policy in replay_policies for category in categories)
            planned_conditions.append(condition_data)
        job = {**group,
            "seed_manifest": seed_manifest,
            "seed_manifest_file": seed_manifest_file,
            "seed_manifest_sha256": seed_manifest_sha256,
            "conditions": planned_conditions,
            "checkpoint_sha256": checkpoints,
            "source_identity": source_binding,
            "training_configuration_sha256": training_configuration_sha256,
        }
        job["job_id"] = _sha(job)
        jobs.append(job)

    body = {"schema": PLAN_SCHEMA,
        "qualification_claim": qualification_claim,
        "qualification_eligible": (
            protocol.get("protocol_version") == "meco-qualification-v2" and
            protocol.get("gate_calibration_status") == "calibrated_baseline_frozen" and
            protocol.get("safety_gate_status") == "metrics_schema_verified"),
        "protocol": {"file": protocol_file, "protocol_id": protocol_id,
                     "protocol_sha256": protocol_sha256,
                     "protocol_version": protocol.get("protocol_version"),
                     "gate_calibration_status": protocol.get("gate_calibration_status")},
        "split": "validation",
        "scenario_ids": list(SCENARIOS),
        "job_count": len(jobs),
        "jobs": jobs,
        "final_test_seed_handling": "not_exported",
    }
    body["plan_digest"] = _sha(body)
    return body
