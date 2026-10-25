"""Offline contracts for immutable Phase 0 baseline and split freezes.

The MECO track evaluator calls ``make_curriculum_env`` from
``meco_track_training.evaluate_meco``. That factory builds ``CurriculumEnv``
from ``TensorThreeVsThreeEnv`` and applies scenario, category, layout, sensor,
and opponent assignments. ``tensor_3v3.py`` imports the shared low-level
``TensorVectorizedSimulator`` from ``tensor_sim.py``; only that shared physics
dependency is included in the source map. The map also follows scenario and
opponent selection through the evaluator and their registries. Constructor
defaults and per-run overrides are not an effective campaign configuration,
so callers must supply observed values, runtime identity, and the content
bytes of every required outcome-affecting configuration artifact. Config
artifacts are stored by stable logical ID and content digest; filesystem
locations are not identity. This module does not instantiate a simulator or
create a production freeze.

The simulator reports scored FUEL piece counts; it does not assign official
match points. Do not use those counts as a ``match_points`` field or as a
qualification gate.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


BASELINE_FREEZE_SCHEMA = "meco-baseline-freeze-v2"
PARTITION_CONTRACT_SCHEMA = "meco-heldout-partitions-v1"

# Every field is an effective value to be extracted from a real campaign
# instance. Source hashes identify the code that defines or consumes it; they
# are not treated as proof of the value by themselves.
AUTHORITATIVE_FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    "robot_dynamics": (
        "frc_defense/curriculum_scenarios.py",
        "frc_defense/tensor_3v3.py",
        "frc_defense/tensor_sim.py",
        "frc_defense/tensor_physics.py",
        "frc_defense/fuel_physics_runtime.py",
    ),
    "field_geometry": (
        "frc_defense/curriculum_scenarios.py",
        "frc_defense/tensor_3v3.py",
        "frc_defense/field.py",
    ),
    "legal_connectivity": (
        "frc_defense/curriculum_scenarios.py",
        "frc_defense/tensor_3v3.py",
        "frc_defense/field.py",
        "frc_defense/tensor_adstar.py",
        "frc_defense/_tensor_3v3_navigation.py",
    ),
    "opponent_controllers": (
        "frc_defense/curriculum_scenarios.py",
        "frc_defense/training_curriculum.py",
        "frc_defense/meco_track_training.py",
        "frc_defense/meco_phase4_opponents.py",
        "frc_defense/tensor_3v3.py",
        "frc_defense/_tensor_3v3_actions.py",
    ),
    "action_update_cadence": (
        "frc_defense/curriculum_scenarios.py",
        "frc_defense/tensor_3v3.py",
        "frc_defense/_tensor_3v3_actions.py",
        "frc_defense/tensor_physics.py",
    ),
    "sensor_noise_profiles": (
        "frc_defense/curriculum_scenarios.py",
        "frc_defense/tensor_3v3.py",
        "frc_defense/_tensor_3v3_observation.py",
        "frc_defense/tensor_perception.py",
    ),
    "scoring_observation_semantics": (
        "frc_defense/curriculum_scenarios.py",
        "frc_defense/tensor_3v3.py",
        "frc_defense/_tensor_3v3_gamepieces.py",
        "frc_defense/meco_track_training.py",
        "frc_defense/meco_metric_ledger.py",
    ),
    "safety_event_semantics": (
        "frc_defense/tensor_physics.py",
        "frc_defense/tensor_3v3.py",
        "frc_defense/_tensor_3v3_actions.py",
        "frc_defense/curriculum_scenarios.py",
        "frc_defense/meco_track_training.py",
        "frc_defense/meco_metric_ledger.py",
    ),
}

RUNTIME_IDENTITY_FIELDS = frozenset({
    "simulator_id",
    "simulator_version",
    "simulator_build_sha256",
    "python_version",
    "torch_version",
    "backend",
    "backend_version",
    "device_id",
    "dependency_lock_sha256",
})

PARTITION_NAMES = ("train", "calibration", "validation", "final_test")
PARTITION_AXES = (
    "seeds",
    "defender_families",
    "layouts",
    "start_conditions",
)
FINAL_TEST_FORBIDDEN_USES = frozenset({
    "training", "calibration", "reference_selection", "margin_review",
})
PARTITION_WORKFLOWS = {
    "train": frozenset({"training"}),
    "calibration": frozenset({"calibration", "reference_selection", "margin_review"}),
    "validation": frozenset({"validation", "model_selection"}),
    "final_test": frozenset({"final_evaluation"}),
}
REQUIRED_CONFIGURATION_ARTIFACT_IDS = frozenset({
    "campaign_assignment",
    "evaluation_protocol",
    "drivetrain_physics",
    "gamepiece_rules",
    "field_layout",
    "scenario_registry",
    "opponent_schedule",
    "sensor_profile",
})

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RELATIVE_PATH = re.compile(r"^[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)*$")
_CONFIG_ARTIFACT_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_NONIDENTITY_VALUES = {"", "unknown", "unavailable", "unspecified", "none", "null"}


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data using stable, compact UTF-8 encoding."""
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _repo_path(root: Path, relative_path: str) -> Path:
    if (not isinstance(relative_path, str)
            or not _SAFE_RELATIVE_PATH.fullmatch(relative_path)
            or relative_path.startswith(".")):
        raise ValueError("repository path is missing or unsafe")
    root = root.resolve()
    candidate = root / relative_path
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError(f"symlinked repository path is not accepted: {relative_path}")
    resolved = candidate.resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"repository file is missing or escapes root: {relative_path}")
    return resolved


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"could not inspect source identity with git: {args[0]}") from exc
    return result.stdout.strip()


def capture_source_identity(repo_root: str | Path, source_plan_path: str) -> dict[str, Any]:
    """Read actual clean checkout and source-plan identity; reject dirty trees."""
    root = Path(repo_root).resolve()
    git_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if git_root != root:
        raise ValueError("repo_root must be the checkout root")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError("baseline freeze requires a clean source checkout")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if not _HEX_40.fullmatch(commit) or not _HEX_40.fullmatch(tree):
        raise ValueError("git did not provide exact source commit/tree identities")
    plan_file = _repo_path(root, source_plan_path)
    return {
        "commit": commit,
        "tree": tree,
        "clean": True,
        "source_plan_path": source_plan_path,
        "source_plan_sha256": hashlib.sha256(plan_file.read_bytes()).hexdigest(),
    }


def _validate_runtime_identity(runtime: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(runtime, Mapping) or set(runtime) != RUNTIME_IDENTITY_FIELDS:
        raise ValueError("runtime identity must provide every required identity field exactly")
    clean: dict[str, str] = {}
    for key in sorted(RUNTIME_IDENTITY_FIELDS):
        value = runtime[key]
        if not isinstance(value, str) or value.strip().lower() in _NONIDENTITY_VALUES:
            raise ValueError(f"runtime identity field is missing or ambiguous: {key}")
        clean[key] = value.strip()
    for key in ("simulator_build_sha256", "dependency_lock_sha256"):
        if not _HEX_64.fullmatch(clean[key]):
            raise ValueError(f"runtime identity has invalid digest: {key}")
    return clean


def _normalize_configuration_artifacts(
    artifacts: Mapping[str, bytes],
) -> list[dict[str, str]]:
    """Bind supplied config contents without persisting their filesystem paths."""
    if (not isinstance(artifacts, Mapping)
            or any(not isinstance(key, str) for key in artifacts)):
        raise ValueError("configuration artifacts must provide every required config content")
    missing = REQUIRED_CONFIGURATION_ARTIFACT_IDS - set(artifacts)
    extra = set(artifacts) - REQUIRED_CONFIGURATION_ARTIFACT_IDS
    if missing or extra:
        raise ValueError(
            "configuration artifact IDs must match the required set exactly; "
            f"missing={sorted(missing)}, extra={sorted(extra)}")
    normalized = []
    for artifact_id in sorted(artifacts):
        content = artifacts[artifact_id]
        if (not isinstance(artifact_id, str)
                or not _CONFIG_ARTIFACT_ID.fullmatch(artifact_id)
                or not isinstance(content, bytes)):
            raise ValueError("configuration artifact needs a stable logical ID and content bytes")
        normalized.append({
            "artifact_id": artifact_id,
            "content_sha256": hashlib.sha256(content).hexdigest(),
        })
    return normalized


def _validate_field_records(
    repo_root: Path,
    records: Mapping[str, Any],
    *,
    is_normalized: bool = False,
) -> dict[str, Any]:
    if not isinstance(records, Mapping) or set(records) != set(AUTHORITATIVE_FIELD_SOURCES):
        missing = sorted(set(AUTHORITATIVE_FIELD_SOURCES) - set(records or ()))
        extra = sorted(set(records or ()) - set(AUTHORITATIVE_FIELD_SOURCES))
        raise ValueError(f"baseline fields must be complete and exact; missing={missing}, extra={extra}")
    normalized: dict[str, Any] = {}
    for field, required_paths in AUTHORITATIVE_FIELD_SOURCES.items():
        record = records[field]
        expected_record_keys = {"value", "selector", "sources"}
        if is_normalized:
            expected_record_keys.add("value_sha256")
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            raise ValueError(f"{field} needs exactly value, selector, and sources")
        value = record["value"]
        if not isinstance(value, (dict, list)) or not value:
            raise ValueError(f"{field} must contain a nonempty structured value")
        if field == "scoring_observation_semantics":
            required_score_semantics = {
                "simulator_metric": "fuel_score_count",
                "metric_semantics": "scored_fuel_piece_count",
            }
            if (not isinstance(value, dict)
                    or set(value) != {
                        "simulator_metric", "metric_semantics",
                        "official_match_points_available",
                    }
                    or any(value.get(key) != expected
                           for key, expected in required_score_semantics.items())
                    or value.get("official_match_points_available") is not False):
                raise ValueError(
                    "current simulator scoring must remain explicitly FUEL-piece-count-only"
                )
        try:
            value_hash = canonical_sha256(value)
        except ValueError as exc:
            raise ValueError(f"{field} value is not valid finite JSON data") from exc
        if is_normalized and record["value_sha256"] != value_hash:
            raise ValueError(f"{field} value digest mismatch")
        selector = record["selector"]
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError(f"{field} requires an extraction selector")
        refs = record["sources"]
        if not isinstance(refs, list) or len(refs) != len(required_paths):
            raise ValueError(f"{field} must bind every authoritative source exactly once")
        by_path: dict[str, str] = {}
        for ref in refs:
            if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
                raise ValueError(f"{field} has malformed source reference")
            path, digest = ref["path"], ref["sha256"]
            if path in by_path or path not in required_paths or not isinstance(digest, str) or not _HEX_64.fullmatch(digest):
                raise ValueError(f"{field} has duplicate, non-authoritative, or invalid source reference")
            actual = hashlib.sha256(_repo_path(repo_root, path).read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"{field} source digest mismatch: {path}")
            by_path[path] = digest
        if set(by_path) != set(required_paths):
            raise ValueError(f"{field} omits an authoritative source file")
        normalized[field] = {
            "value": value,
            "value_sha256": value_hash,
            "selector": selector.strip(),
            "sources": [{"path": path, "sha256": by_path[path]} for path in required_paths],
        }
    return normalized


def compile_baseline_freeze(
    repo_root: str | Path,
    *,
    source_plan_path: str,
    runtime_identity: Mapping[str, Any],
    field_records: Mapping[str, Any],
    configuration_artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    """Compile only a complete, clean-source, explicitly runtime-bound freeze.

    No defaults are supplied for effective simulator fields or runtime identity.
    Missing observations make compilation fail rather than creating an
    incomplete artifact that could be mistaken for a usable baseline.
    """
    root = Path(repo_root).resolve()
    source = capture_source_identity(root, source_plan_path)
    runtime = _validate_runtime_identity(runtime_identity)
    fields = _validate_field_records(root, field_records)
    configs = _normalize_configuration_artifacts(configuration_artifacts)
    body = {
        "schema": BASELINE_FREEZE_SCHEMA,
        "status": "complete_source_runtime_bound_baseline",
        "source": source,
        "runtime": runtime,
        "runtime_sha256": canonical_sha256(runtime),
        "configuration_artifacts": configs,
        "configuration_artifacts_sha256": canonical_sha256(configs),
        "fields": fields,
        "qualification_eligible": False,
    }
    return {**body, "freeze_sha256": canonical_sha256(body)}


def validate_baseline_freeze(
    freeze: Mapping[str, Any],
    repo_root: str | Path,
    *,
    configuration_artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    """Validate a compiled freeze against current source files and checkout."""
    if not isinstance(freeze, Mapping):
        raise ValueError("baseline freeze must be an object")
    expected_keys = {
        "schema", "status", "source", "runtime", "runtime_sha256",
        "configuration_artifacts", "configuration_artifacts_sha256", "fields",
        "qualification_eligible", "freeze_sha256",
    }
    if set(freeze) != expected_keys:
        raise ValueError("baseline freeze has missing or unexpected fields")
    if (freeze["schema"] != BASELINE_FREEZE_SCHEMA
            or freeze["status"] != "complete_source_runtime_bound_baseline"
            or freeze["qualification_eligible"] is not False):
        raise ValueError("unsupported or qualification-eligible baseline freeze")
    source = freeze["source"]
    if not isinstance(source, Mapping) or set(source) != {
            "commit", "tree", "clean", "source_plan_path", "source_plan_sha256"}:
        raise ValueError("baseline source identity is incomplete")
    if source["clean"] is not True:
        raise ValueError("baseline source checkout is not clean")
    current_source = capture_source_identity(repo_root, source["source_plan_path"])
    if dict(source) != current_source:
        raise ValueError("baseline source or source plan identity drifted")
    runtime = _validate_runtime_identity(freeze["runtime"])
    if freeze["runtime_sha256"] != canonical_sha256(runtime):
        raise ValueError("runtime identity digest mismatch")
    configs = _normalize_configuration_artifacts(configuration_artifacts)
    if (freeze["configuration_artifacts"] != configs
            or freeze["configuration_artifacts_sha256"] != canonical_sha256(configs)):
        raise ValueError("configuration artifact identity drifted")
    normalized_fields = _validate_field_records(
        Path(repo_root).resolve(), freeze["fields"], is_normalized=True)
    if normalized_fields != freeze["fields"]:
        raise ValueError("baseline field values or source bindings are not canonical")
    body = {key: freeze[key] for key in expected_keys if key != "freeze_sha256"}
    if freeze["freeze_sha256"] != canonical_sha256(body):
        raise ValueError("baseline freeze digest mismatch")
    return dict(freeze)


def _normalize_partition_axes(partition_name: str, partition: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(partition, Mapping) or set(partition) != set(PARTITION_AXES):
        raise ValueError(f"{partition_name} must provide all partition axes exactly")
    normalized: dict[str, Any] = {}
    for axis in PARTITION_AXES:
        values = partition[axis]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{partition_name}.{axis} must be a nonempty exact table")
        if axis == "seeds":
            if any(type(seed) is not int or seed < 0 for seed in values):
                raise ValueError(f"{partition_name}.seeds must contain nonnegative integers")
            if list(values) != sorted(values):
                raise ValueError(f"{partition_name}.seeds must be sorted for canonical identity")
        elif any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError(f"{partition_name}.{axis} must contain nonempty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{partition_name}.{axis} contains duplicate assignments")
        normalized[axis] = list(values)
    return normalized


def compile_partition_contract(
    *,
    baseline_freeze_sha256: str,
    partitions: Mapping[str, Any],
    allowed_uses: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile a disjoint four-way seed/family/layout/start partition contract."""
    if not isinstance(baseline_freeze_sha256, str) or not _HEX_64.fullmatch(baseline_freeze_sha256):
        raise ValueError("partition contract must bind an exact baseline freeze digest")
    if not isinstance(partitions, Mapping) or set(partitions) != set(PARTITION_NAMES):
        raise ValueError("exact train/calibration/validation/final_test partitions are required")
    normalized = {
        name: _normalize_partition_axes(name, partitions[name])
        for name in PARTITION_NAMES
    }
    for axis in PARTITION_AXES:
        owners: dict[Any, str] = {}
        for name in PARTITION_NAMES:
            for value in normalized[name][axis]:
                if value in owners:
                    raise ValueError(
                        f"{axis} assignment leaks across {owners[value]} and {name}: {value!r}"
                    )
                owners[value] = name
    if not isinstance(allowed_uses, Mapping) or set(allowed_uses) != set(PARTITION_NAMES):
        raise ValueError("workflow access policy must declare every partition exactly")
    clean_uses: dict[str, list[str]] = {}
    for name, expected in PARTITION_WORKFLOWS.items():
        values = allowed_uses[name]
        if (not isinstance(values, (list, tuple, set, frozenset))
                or any(not isinstance(value, str) or not value.strip() for value in values)
                or len(set(values)) != len(values)
                or set(values) != expected):
            raise ValueError(f"{name} workflow access policy is not the preregistered policy")
        clean_uses[name] = sorted(values)
    if FINAL_TEST_FORBIDDEN_USES.intersection(clean_uses["final_test"]):
        raise ValueError("final-test data is accessible to a prohibited workflow")
    body = {
        "schema": PARTITION_CONTRACT_SCHEMA,
        "status": "disjoint_contract_only",
        "baseline_freeze_sha256": baseline_freeze_sha256,
        "partitions": normalized,
        "allowed_uses": clean_uses,
        "qualification_eligible": False,
    }
    return {**body, "partition_sha256": canonical_sha256(body)}


def validate_partition_contract(
    contract: Mapping[str, Any], *, baseline_freeze_sha256: str,
) -> dict[str, Any]:
    """Re-run the compiler's structural and leakage checks on a partition."""
    if not isinstance(contract, Mapping):
        raise ValueError("partition contract must be an object")
    expected_keys = {
        "schema", "status", "baseline_freeze_sha256", "partitions", "allowed_uses",
        "qualification_eligible", "partition_sha256",
    }
    if set(contract) != expected_keys:
        raise ValueError("partition contract has missing or unexpected fields")
    if (contract["schema"] != PARTITION_CONTRACT_SCHEMA
            or contract["status"] != "disjoint_contract_only"
            or contract["qualification_eligible"] is not False):
        raise ValueError("unsupported or qualification-eligible partition contract")
    if (not isinstance(baseline_freeze_sha256, str)
            or not _HEX_64.fullmatch(baseline_freeze_sha256)
            or contract["baseline_freeze_sha256"] != baseline_freeze_sha256):
        raise ValueError("partition contract does not bind the validated baseline freeze")
    compiled = compile_partition_contract(
        baseline_freeze_sha256=contract["baseline_freeze_sha256"],
        partitions=contract["partitions"],
        allowed_uses=contract["allowed_uses"],
    )
    if compiled != dict(contract):
        raise ValueError("partition contract digest or canonical form mismatch")
    return dict(contract)
