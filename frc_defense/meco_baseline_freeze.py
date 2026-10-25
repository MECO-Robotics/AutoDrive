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

import base64
import hashlib
import importlib
import json
import re
import subprocess
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BASELINE_FREEZE_SCHEMA = "meco-baseline-freeze-v2"
PARTITION_CONTRACT_SCHEMA = "meco-heldout-partitions-v1"

# Approval is unavailable until an independently reviewed content digest is
# pinned here. Empty sets intentionally keep materialized drafts non-authorizing.
TRUSTED_ALLOCATION_ARTIFACT_SHA256: frozenset[str] = frozenset()
TRUSTED_FINAL_TEST_APPROVAL_SHA256: frozenset[str] = frozenset()
FINAL_TEST_LEDGER_ROOT = Path(__file__).resolve().parents[1] / "qualifications/meco/final-test-ledgers"

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

CAPTURE_RECEIPT_SCHEMA = "meco-effective-capture-receipt-v1"
EFFECTIVE_SNAPSHOT_SCHEMA = "meco-effective-configuration-snapshot-v1"
EFFECTIVE_FREEZE_SCHEMA = "meco-baseline-freeze-v3"
TUPLE_REGISTRY_SCHEMA = "meco-evaluation-tuple-registry-v1"
PARTITION_MATERIALIZATION_SCHEMA = "meco-heldout-partitions-v2"
FIVE_TRACK_IDS = ("A1", "A2", "A3", "C1", "C2")
TUPLE_FIELDS = ("scenario_id", "case_id", "category", "defender_family",
    "layout_id", "start_condition", "sensor_profile", "seed", "repetition",
    "evaluation_phase")
ARTIFACT_CONFIG_IDS = tuple(sorted(REQUIRED_CONFIGURATION_ARTIFACT_IDS))

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
    runtime_identity: Mapping[str, Any] | None = None,
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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _strict_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_pinned_json_artifact(reference: Mapping[str, Any], *,
        trusted_digests: frozenset[str], label: str) -> tuple[dict[str, Any], str]:
    if not isinstance(reference, Mapping) or set(reference) != {
            "content_base64", "content_sha256"}:
        raise ValueError(f"{label} must include exact content bytes and digest")
    try:
        content = base64.b64decode(reference["content_base64"], validate=True)
        def reject_duplicate_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"{label} contains duplicate JSON keys")
                result[key] = value
            return result
        parsed = json.loads(content, object_pairs_hook=reject_duplicate_pairs)
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} content is not valid canonical JSON") from exc
    if not isinstance(parsed, dict) or _canonical_json_bytes(parsed) != content:
        raise ValueError(f"{label} content is not canonical JSON")
    digest = hashlib.sha256(content).hexdigest()
    if digest != reference["content_sha256"]:
        raise ValueError(f"{label} content digest mismatch")
    if digest not in trusted_digests:
        raise ValueError(f"{label} is not authenticated by a pinned trusted digest")
    return parsed, digest


def _validate_factory_module_origins(factory: Any, repo_root: str | Path) -> None:
    root = Path(repo_root).resolve()
    expected_factory = (root / "frc_defense/curriculum_scenarios.py").resolve()
    filename = Path(factory.__code__.co_filename).resolve()
    module_path = Path(importlib.import_module(factory.__module__).__file__).resolve()
    if filename != expected_factory or module_path != expected_factory:
        raise ValueError("loaded curriculum factory does not originate from the captured source root")
    authoritative = {path for paths in AUTHORITATIVE_FIELD_SOURCES.values() for path in paths}
    for relative_path in authoritative:
        path = (root / relative_path).resolve()
        if not path.is_file():
            raise ValueError(f"authoritative source is absent from capture root: {relative_path}")
    package_root = (root / "frc_defense").resolve()
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not (module_name == "frc_defense"
                or module_name.startswith("frc_defense.")):
            continue
        module_file = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        spec_origin = getattr(spec, "origin", None) if spec is not None else None
        # A behavior module with no verifiable filesystem origin can still be a
        # preloaded module from another checkout (or an injected module). Fail
        # closed instead of silently omitting it from the source closure.
        module_origin = module_file or spec_origin
        if not isinstance(module_origin, str) or not module_origin:
            raise ValueError(f"loaded behavior module {module_name} has no verifiable source origin")
        if module_origin in {"built-in", "frozen"}:
            raise ValueError(f"loaded behavior module {module_name} has no verifiable source origin")
        origin = Path(module_origin).resolve()
        try:
            origin.relative_to(package_root)
        except ValueError as exc:
            raise ValueError(f"loaded behavior module {module_name} comes from another checkout") from exc


def _strict_utc_timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} is malformed") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must use UTC")
    return value


def _validate_effective_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    required = RUNTIME_IDENTITY_FIELDS | {"device_model", "device_index"}
    if not isinstance(runtime, Mapping) or set(runtime) != required:
        raise ValueError("effective runtime identity must include exact simulator, dependency, and indexed device fields")
    normalized = {}
    for name, value in runtime.items():
        if name == "device_index":
            if type(value) is not int or value < 0:
                raise ValueError("effective runtime device_index must be nonnegative")
            normalized[name] = value
        elif not isinstance(value, str) or value.strip().lower() in _NONIDENTITY_VALUES:
            raise ValueError(f"effective runtime identity is missing or ambiguous: {name}")
        else:
            normalized[name] = value.strip()
    for name in ("simulator_build_sha256", "dependency_lock_sha256"):
        _strict_digest(normalized[name], name)
    if normalized["device_id"].startswith(("auto", "cuda", "hip", "rocm")):
        raise ValueError("effective runtime device_id must identify a physical device, not an auto/index selector")
    if not normalized["device_model"].strip():
        raise ValueError("effective runtime requires a device model")
    return normalized


def _effective_snapshot_body(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping) or set(snapshot) != {
            "schema", "producer", "factory_arguments", "resolved_fields", "configuration_artifacts",
            "runtime_identity", "snapshot_sha256"}:
        raise ValueError("effective snapshot has missing or unexpected fields")
    body = {key: snapshot[key] for key in snapshot if key != "snapshot_sha256"}
    if snapshot["schema"] != EFFECTIVE_SNAPSHOT_SCHEMA:
        raise ValueError("unsupported effective configuration snapshot schema")
    producer = snapshot["producer"]
    if (not isinstance(producer, Mapping) or set(producer) != {
            "module", "function", "source_identity_sha256"}
            or producer.get("module") != "frc_defense.curriculum_scenarios"
            or producer.get("function") != "make_curriculum_env"):
        raise ValueError("snapshot must come from the current curriculum environment factory")
    _strict_digest(producer["source_identity_sha256"], "snapshot producer source identity")
    if snapshot["snapshot_sha256"] != canonical_sha256(body):
        raise ValueError("effective snapshot digest mismatch")
    factory_arguments = snapshot["factory_arguments"]
    if not isinstance(factory_arguments, Mapping) or not factory_arguments:
        raise ValueError("effective snapshot must retain exact public factory arguments")
    if any(not isinstance(key, str) or key.startswith("_") for key in factory_arguments):
        raise ValueError("effective snapshot factory arguments cannot contain private controls")
    return body


def compile_effective_capture_receipt(
    repo_root: str | Path,
    *,
    source_plan_path: str,
    factory_arguments: Mapping[str, Any],
    runtime_identity: Mapping[str, Any] | None = None,
    captured_at_utc: str | None = None,
) -> dict[str, Any]:
    """Ask the live factory resolver for its exact constructor snapshot, then bind it.

    The resolver has a snapshot-only path that shares its normalized constructor
    object with environment creation and returns before device initialization.
    """
    import base64
    import inspect

    root = Path(repo_root).resolve()
    if not isinstance(factory_arguments, Mapping):
        raise ValueError("factory arguments must be an explicit object")
    from .curriculum_scenarios import make_curriculum_env
    parameters = inspect.signature(make_curriculum_env).parameters
    public_arguments = {name for name, parameter in parameters.items()
        if parameter.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        and not name.startswith("_")}
    required_arguments = {"scenario_id", "category", "num_envs", "device", "seed",
        "fuel_physics", "production_d0", "layout_id", "defender_family",
        "start_condition", "sensor_profile", "evaluation_protocol_object",
        "evaluation_protocol_content_base64", "evaluation_protocol_content_sha256"}
    if set(factory_arguments) - public_arguments or not required_arguments <= set(factory_arguments):
        raise ValueError("effective capture must bind every explicit curriculum factory setting")
    _validate_factory_module_origins(make_curriculum_env, root)
    effective_snapshot = make_curriculum_env(**dict(factory_arguments),
        _effective_snapshot_only=True, _effective_snapshot_runtime_identity=runtime_identity,
        _effective_snapshot_repo_root=root,
        _effective_snapshot_source_plan_path=source_plan_path)
    body = _effective_snapshot_body(effective_snapshot)
    source = capture_source_identity(root, source_plan_path)
    if body["producer"]["source_identity_sha256"] != canonical_sha256(source):
        raise ValueError("effective snapshot was emitted by a different source checkout")
    fields = body["resolved_fields"]
    if not isinstance(fields, Mapping) or set(fields) != set(AUTHORITATIVE_FIELD_SOURCES):
        raise ValueError("effective snapshot must resolve every authoritative field")
    clean_fields = {}
    for field, required_paths in AUTHORITATIVE_FIELD_SOURCES.items():
        record = fields[field]
        expected = {"value", "selector", "runtime_owner", "resolution_state", "derivation"}
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError(f"effective {field} provenance is incomplete")
        if record["resolution_state"] not in {"resolved", "overridden", "inherited"}:
            raise ValueError(f"effective {field} is default-only, intended-only, or unresolved")
        if not isinstance(record["selector"], str) or not record["selector"].strip():
            raise ValueError(f"effective {field} requires the actual runtime selector")
        if not isinstance(record["runtime_owner"], str) or not record["runtime_owner"].strip():
            raise ValueError(f"effective {field} requires its runtime owner")
        derivation = record["derivation"]
        if (not isinstance(derivation, Mapping) or set(derivation) != {"kind", "input_sha256"}
                or derivation["kind"] not in {"factory_argument", "resolved_scenario", "runtime_object"}):
            raise ValueError(f"effective {field} requires a reproducible derivation reference")
        _strict_digest(derivation["input_sha256"], f"{field} derivation input")
        value_hash = canonical_sha256(record["value"])
        sources = [{"path": path,
            "sha256": hashlib.sha256(_repo_path(root, path).read_bytes()).hexdigest()}
            for path in required_paths]
        clean_fields[field] = {"value": record["value"], "value_sha256": value_hash,
            "selector": record["selector"], "runtime_owner": record["runtime_owner"],
            "resolution_state": record["resolution_state"],
            "derivation": dict(derivation), "sources": sources}
    runtime = _validate_effective_runtime(body["runtime_identity"])
    artifacts = body["configuration_artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != REQUIRED_CONFIGURATION_ARTIFACT_IDS:
        raise ValueError("effective snapshot must carry bytes for exactly the eight logical configuration artifacts")
    clean_artifacts = []
    for artifact_id in sorted(artifacts):
        artifact = artifacts[artifact_id]
        if (not isinstance(artifact, Mapping) or set(artifact) != {
                "content_base64", "content_sha256"}):
            raise ValueError(f"{artifact_id} effective configuration must bind canonical content bytes")
        try:
            content = base64.b64decode(artifact["content_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{artifact_id} effective configuration bytes are invalid") from exc
        try:
            decoded = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{artifact_id} configuration content is not JSON") from exc
        if _canonical_json_bytes(decoded) != content:
            raise ValueError(f"{artifact_id} configuration bytes are not canonical JSON")
        if hashlib.sha256(content).hexdigest() != artifact["content_sha256"]:
            raise ValueError(f"{artifact_id} configuration content digest mismatch")
        clean_artifacts.append({"artifact_id": artifact_id,
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "content_base64": base64.b64encode(content).decode("ascii")})
    captured = captured_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    _strict_utc_timestamp(captured, "capture time")
    receipt_body = {"schema": CAPTURE_RECEIPT_SCHEMA,
        "capture_tool_version": 1, "captured_at_utc": captured,
        "source": {**source, "authoritative_source_sha256": {
            path: hashlib.sha256(_repo_path(root, path).read_bytes()).hexdigest()
            for path in sorted({p for paths in AUTHORITATIVE_FIELD_SOURCES.values() for p in paths})}},
        "effective_snapshot_sha256": effective_snapshot["snapshot_sha256"],
        "effective_snapshot": effective_snapshot,
        "fields": clean_fields, "configuration_artifacts": clean_artifacts,
        "runtime": runtime, "qualification_eligible": False}
    return {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}


def validate_effective_capture_receipt(receipt: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    """Verify current source, config bytes, field hashes, and receipt digest."""
    import base64

    if not isinstance(receipt, Mapping):
        raise ValueError("capture receipt must be an object")
    expected = {"schema", "capture_tool_version", "captured_at_utc", "source",
        "effective_snapshot_sha256", "effective_snapshot", "fields", "configuration_artifacts",
        "runtime", "qualification_eligible", "receipt_sha256"}
    if set(receipt) != expected or receipt["schema"] != CAPTURE_RECEIPT_SCHEMA:
        raise ValueError("unsupported or noncanonical capture receipt")
    if receipt["qualification_eligible"] is not False or receipt["capture_tool_version"] != 1:
        raise ValueError("capture receipt cannot claim eligibility")
    source = receipt["source"]
    current = capture_source_identity(repo_root, source["source_plan_path"])
    if any(source.get(key) != value for key, value in current.items()):
        raise ValueError("capture receipt source tree or plan drifted")
    for path, digest in source.get("authoritative_source_sha256", {}).items():
        if hashlib.sha256(_repo_path(Path(repo_root).resolve(), path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"capture receipt authoritative source drifted: {path}")
    if set(source.get("authoritative_source_sha256", {})) != {
            p for paths in AUTHORITATIVE_FIELD_SOURCES.values() for p in paths}:
        raise ValueError("capture receipt omits authoritative source hashes")
    snapshot = receipt["effective_snapshot"]
    snapshot_body = _effective_snapshot_body(snapshot)
    if (snapshot.get("snapshot_sha256") != receipt["effective_snapshot_sha256"]
            or snapshot["producer"]["source_identity_sha256"] != canonical_sha256(current)
            or _validate_effective_runtime(receipt["runtime"])
                != _validate_effective_runtime(snapshot["runtime_identity"])):
        raise ValueError("capture receipt runtime or snapshot binding mismatch")
    # Re-run only the factory's snapshot path from the captured public arguments.
    # It returns before environment construction; the runtime probe is read-only.
    from .curriculum_scenarios import make_curriculum_env
    _validate_factory_module_origins(make_curriculum_env, repo_root)
    reproduced_snapshot = make_curriculum_env(**dict(snapshot["factory_arguments"]),
        _effective_snapshot_only=True,
        _effective_snapshot_runtime_identity=snapshot["runtime_identity"],
        _effective_snapshot_repo_root=Path(repo_root).resolve(),
        _effective_snapshot_source_plan_path=source["source_plan_path"])
    if reproduced_snapshot != snapshot:
        raise ValueError("capture receipt snapshot does not reproduce from the current factory")
    for field, record in receipt["fields"].items():
        if field not in AUTHORITATIVE_FIELD_SOURCES or set(record) != {
                "value", "value_sha256", "selector", "runtime_owner",
                "resolution_state", "derivation", "sources"}:
            raise ValueError(f"capture receipt field schema mismatch: {field}")
        if canonical_sha256(record["value"]) != record["value_sha256"]:
            raise ValueError(f"capture receipt field value digest mismatch: {field}")
        if (not isinstance(record["selector"], str) or not record["selector"].strip()
                or not isinstance(record["runtime_owner"], str) or not record["runtime_owner"].strip()
                or record["resolution_state"] not in {"resolved", "overridden", "inherited"}
                or not isinstance(record["derivation"], Mapping)
                or set(record["derivation"]) != {"kind", "input_sha256"}
                or record["derivation"]["kind"] not in {"factory_argument", "resolved_scenario", "runtime_object"}):
            raise ValueError(f"capture receipt field provenance is incomplete: {field}")
        _strict_digest(record["derivation"]["input_sha256"], f"{field} derivation input")
        source_refs = record["sources"]
        expected_paths = set(AUTHORITATIVE_FIELD_SOURCES[field])
        if (not isinstance(source_refs, list)
                or len(source_refs) != len(expected_paths)
                or {ref.get("path") for ref in source_refs if isinstance(ref, Mapping)} != expected_paths):
            raise ValueError(f"capture receipt field source closure mismatch: {field}")
        for ref in record["sources"]:
            if (not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}
                    or source["authoritative_source_sha256"].get(ref["path"]) != ref["sha256"]):
                raise ValueError(f"capture receipt field source binding mismatch: {field}")
    if set(receipt["fields"]) != set(AUTHORITATIVE_FIELD_SOURCES):
        raise ValueError("capture receipt omits effective fields")
    if ({name: {"value": record["value"], "selector": record["selector"],
            "runtime_owner": record["runtime_owner"],
            "resolution_state": record["resolution_state"],
            "derivation": record["derivation"]}
            for name, record in receipt["fields"].items()} != snapshot["resolved_fields"]):
        raise ValueError("capture receipt effective fields differ from reproduced factory snapshot")
    artifacts = receipt["configuration_artifacts"]
    if (not isinstance(artifacts, list) or len(artifacts) != len(REQUIRED_CONFIGURATION_ARTIFACT_IDS)
            or {item.get("artifact_id") for item in artifacts if isinstance(item, Mapping)}
                != REQUIRED_CONFIGURATION_ARTIFACT_IDS):
        raise ValueError("capture receipt configuration artifact closure mismatch")
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {
                "artifact_id", "content_sha256", "content_base64"}:
            raise ValueError("capture receipt configuration artifact schema mismatch")
        raw = base64.b64decode(item["content_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != item["content_sha256"]:
            raise ValueError(f"capture receipt configuration bytes changed: {item['artifact_id']}")
        if _canonical_json_bytes(json.loads(raw)) != raw:
            raise ValueError(f"capture receipt configuration is not canonical: {item['artifact_id']}")
    snapshot_artifacts = snapshot["configuration_artifacts"]
    receipt_artifact_map = {item["artifact_id"]: {
        "content_base64": item["content_base64"],
        "content_sha256": item["content_sha256"]} for item in artifacts}
    if receipt_artifact_map != snapshot_artifacts:
        raise ValueError("capture receipt config bytes differ from reproduced factory snapshot")
    body = {key: receipt[key] for key in expected if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != canonical_sha256(body):
        raise ValueError("capture receipt digest mismatch")
    return dict(receipt)


def compile_effective_baseline_freeze(receipt: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    """Compile the v3 freeze from a fully verified capture receipt."""
    validated = validate_effective_capture_receipt(receipt, repo_root)
    body = {"schema": EFFECTIVE_FREEZE_SCHEMA,
        "status": "complete_source_runtime_bound_effective_baseline",
        "capture_receipt_sha256": validated["receipt_sha256"],
        "source": validated["source"], "runtime": validated["runtime"],
        "fields": validated["fields"],
        "configuration_artifacts": validated["configuration_artifacts"],
        "qualification_eligible": False}
    return {**body, "freeze_sha256": canonical_sha256(body)}


def validate_effective_baseline_freeze(freeze: Mapping[str, Any], receipt: Mapping[str, Any],
                                       repo_root: str | Path) -> dict[str, Any]:
    expected = {"schema", "status", "capture_receipt_sha256", "source", "runtime",
        "fields", "configuration_artifacts", "qualification_eligible", "freeze_sha256"}
    if not isinstance(freeze, Mapping) or set(freeze) != expected:
        raise ValueError("effective baseline freeze has missing or unexpected fields")
    validated_receipt = validate_effective_capture_receipt(receipt, repo_root)
    expected_freeze = compile_effective_baseline_freeze(validated_receipt, repo_root)
    if dict(freeze) != expected_freeze:
        raise ValueError("effective baseline freeze differs from its validated receipt")
    return dict(freeze)


def validate_effective_phase0_bundle(bundle: Mapping[str, Any], repo_root: str | Path) -> dict[str, str]:
    """Validate the complete receipt→freeze→tuple partition chain for intake."""
    if not isinstance(bundle, Mapping) or set(bundle) != {
            "capture_receipt", "baseline_freeze", "partition_materialization"}:
        raise ValueError("Phase 0 intake requires capture receipt, freeze, and partition materialization")
    receipt = validate_effective_capture_receipt(bundle["capture_receipt"], repo_root)
    freeze = validate_effective_baseline_freeze(bundle["baseline_freeze"], receipt, repo_root)
    partition = bundle["partition_materialization"]
    if not isinstance(partition, Mapping):
        raise ValueError("Phase 0 partition materialization is missing")
    normalized_registry = partition.get("tuple_registry")
    if not isinstance(normalized_registry, Mapping):
        raise ValueError("Phase 0 tuple registry is missing")
    raw_registry = {"schema": normalized_registry.get("schema"),
        "tuples": [{key: item[key] for key in (*TUPLE_FIELDS, "eligible")}
            | ({"exclusion_reason": item["exclusion_reason"]}
                if item.get("eligible") is False else {})
            for item in normalized_registry.get("tuples", [])]}
    validate_tuple_partition_materialization(partition,
        freeze_sha256=freeze["freeze_sha256"],
        registry=raw_registry,
        assignment_table=partition.get("assignment_table"))
    if partition.get("allocation_authorized") is not True:
        raise ValueError("Phase 0 intake requires a preregistration artifact pinned by a trusted digest")
    return {"capture_receipt_sha256": receipt["receipt_sha256"],
        "baseline_freeze_sha256": freeze["freeze_sha256"],
        "partition_sha256": partition["partition_sha256"],
        "tuple_registry_sha256": partition["tuple_registry_sha256"],
        "assignment_table_sha256": partition["assignment_table_sha256"]}


def _normalize_tuple_registry(registry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict]]:
    if (not isinstance(registry, Mapping) or set(registry) != {"schema", "tuples"}
            or registry.get("schema") != TUPLE_REGISTRY_SCHEMA
            or not isinstance(registry.get("tuples"), list) or not registry["tuples"]):
        raise ValueError("tuple registry must be a nonempty versioned registry")
    normalized, by_id = [], {}
    for item in registry["tuples"]:
        if not isinstance(item, Mapping):
            raise ValueError("tuple registry entries must be objects")
        eligible = item.get("eligible")
        required = set(TUPLE_FIELDS) | {"eligible"}
        if not isinstance(eligible, bool) or set(item) != required | ({"exclusion_reason"} if not eligible else set()):
            raise ValueError("tuple registry entry has missing fields or invalid eligibility closure")
        value = {name: item[name] for name in TUPLE_FIELDS}
        for key in ("scenario_id", "case_id", "category", "defender_family", "layout_id",
                    "start_condition", "sensor_profile", "evaluation_phase"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError(f"tuple registry {key} must be a nonempty identifier")
        for key in ("seed", "repetition"):
            if type(value[key]) is not int or value[key] < 0:
                raise ValueError(f"tuple registry {key} must be a nonnegative integer")
        if not eligible and (not isinstance(item.get("exclusion_reason"), str)
                or not item["exclusion_reason"].strip()):
            raise ValueError("excluded tuples require an explicit reason code")
        tuple_id = canonical_sha256(value)
        if tuple_id in by_id:
            raise ValueError("tuple registry contains a duplicate tuple")
        record = {**value, "tuple_id": tuple_id, "eligible": eligible}
        if not eligible:
            record["exclusion_reason"] = item["exclusion_reason"]
        normalized.append(record)
        by_id[tuple_id] = record
    normalized.sort(key=lambda item: item["tuple_id"])
    return {"schema": TUPLE_REGISTRY_SCHEMA, "tuples": normalized}, by_id


def compile_tuple_partition_materialization(*, freeze_sha256: str,
        registry: Mapping[str, Any], assignment_table: Mapping[str, Any]) -> dict[str, Any]:
    """Close a predeclared four-way tuple assignment over all five tracks."""
    _strict_digest(freeze_sha256, "baseline freeze digest")
    normalized_registry, by_id = _normalize_tuple_registry(registry)
    if (not isinstance(assignment_table, Mapping) or set(assignment_table) != {
            "schema", "assignments", "model_tracks", "evaluation_budget", "allocation_provenance"}
            or assignment_table.get("schema") != "meco-partition-assignment-table-v1"):
        raise ValueError("partition requires an explicit preregistered assignment table")
    eligible_ids = {key for key, item in by_id.items() if item["eligible"]}
    assignments = assignment_table["assignments"]
    if not isinstance(assignments, Mapping) or set(assignments) != eligible_ids:
        raise ValueError("partition assignment table must cover every eligible tuple exactly once")
    by_partition = {name: [] for name in PARTITION_NAMES}
    for tuple_id, partition in assignments.items():
        if partition not in PARTITION_NAMES:
            raise ValueError("eligible tuple is assigned to an unknown partition")
        if by_id[tuple_id]["evaluation_phase"] != partition:
            raise ValueError("tuple evaluation_phase differs from its assigned partition")
        by_partition[partition].append(tuple_id)
    for name in by_partition:
        by_partition[name].sort()
        if not by_partition[name]:
            raise ValueError(f"partition {name} has no eligible tuples")
    for axis in ("seed", "defender_family", "layout_id", "start_condition"):
        owners = {}
        for partition, ids in by_partition.items():
            for tuple_id in ids:
                value = by_id[tuple_id][axis]
                if value in owners:
                    raise ValueError(f"{axis} leaks across {owners[value]} and {partition}")
                owners[value] = partition
    tracks = assignment_table["model_tracks"]
    if not isinstance(tracks, Mapping) or set(tracks) != set(FIVE_TRACK_IDS):
        raise ValueError("partition must define exactly A1, A2, A3, C1, and C2")
    budget = assignment_table["evaluation_budget"]
    if not isinstance(budget, Mapping) or set(budget) != {"episodes_per_tuple", "repetitions"}:
        raise ValueError("fixed evaluation budget must specify episodes_per_tuple and repetitions")
    if any(type(value) is not int or value < 1 for value in budget.values()):
        raise ValueError("fixed evaluation budget values must be positive integers")
    for track in FIVE_TRACK_IDS:
        item = tracks[track]
        if (not isinstance(item, Mapping) or set(item) != {"tuple_ids", "evaluation_budget"}
                or item["tuple_ids"] != sorted(eligible_ids) or item["evaluation_budget"] != budget):
            raise ValueError(f"{track} partition membership and budget must match all tracks exactly")
    provenance = assignment_table["allocation_provenance"]
    provenance_keys = {"authored_by", "authored_at_utc", "source_sha256", "outcomes_accessed"}
    if (not isinstance(provenance, Mapping) or set(provenance) not in (
            provenance_keys, provenance_keys | {"authorization_artifact"})
            or not isinstance(provenance["authored_by"], str) or not provenance["authored_by"].strip()
            or not isinstance(provenance["authored_at_utc"], str)
            or provenance["outcomes_accessed"] is not False):
        raise ValueError("allocation must be explicitly preregistered before outcomes are accessed")
    _strict_digest(provenance["source_sha256"], "assignment source digest")
    allocation_authorized = False
    allocation_artifact_sha256 = None
    if "authorization_artifact" in provenance:
        artifact_body, allocation_artifact_sha256 = _validate_pinned_json_artifact(
            provenance["authorization_artifact"],
            trusted_digests=TRUSTED_ALLOCATION_ARTIFACT_SHA256,
            label="predeclared tuple assignment artifact")
        expected_artifact_body = {key: assignment_table[key]
            for key in ("schema", "assignments", "model_tracks", "evaluation_budget")}
        if artifact_body != expected_artifact_body:
            raise ValueError("authenticated assignment artifact differs from the materialized table")
        allocation_authorized = True
    uses = {name: sorted(values) for name, values in PARTITION_WORKFLOWS.items()}
    registry_sha = canonical_sha256(normalized_registry)
    assignment_sha = canonical_sha256(dict(assignment_table))
    partitions = {name: {"tuple_ids": ids,
        "tuple_set_sha256": canonical_sha256(ids), "allowed_uses": uses[name]}
        for name, ids in by_partition.items()}
    registry_exclusions = [{"tuple_id": item["tuple_id"],
        "reason": item["exclusion_reason"]} for item in normalized_registry["tuples"]
        if not item["eligible"]]
    body = {"schema": PARTITION_MATERIALIZATION_SCHEMA,
        "status": "closed_tuple_partition_contract_only",
        "baseline_freeze_sha256": freeze_sha256,
        "tuple_registry": normalized_registry, "tuple_registry_sha256": registry_sha,
        "assignment_table": dict(assignment_table), "assignment_table_sha256": assignment_sha,
        "partitions": partitions, "model_tracks": dict(tracks),
        "evaluation_budget": dict(budget),
        "allocation_authorized": allocation_authorized,
        "allocation_artifact_sha256": allocation_artifact_sha256,
        "final_test_access_ledger": {"schema": "meco-final-test-access-ledger-v1",
            "authorization_schema": "meco-final-test-single-use-authorization-v1",
            "max_authorizations": 1, "authorization_count": 0,
            "status": "locked", "outcomes_included": False,
            "outcomes_read": False, "consumption_receipts": []},
        "excluded_tuple_reasons": registry_exclusions,
        "qualification_eligible": False}
    body["final_test_access_ledger"]["partition_sha256"] = canonical_sha256(body)
    return {**body, "partition_sha256": canonical_sha256(body)}


def validate_tuple_partition_materialization(contract: Mapping[str, Any], *,
        freeze_sha256: str, registry: Mapping[str, Any], assignment_table: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise ValueError("tuple partition materialization must be an object")
    expected = compile_tuple_partition_materialization(freeze_sha256=freeze_sha256,
        registry=registry, assignment_table=assignment_table)
    if dict(contract) != expected:
        raise ValueError("tuple partition closure, freeze, registry, or assignment digest mismatch")
    ledger = contract.get("final_test_access_ledger", {})
    body_without_ledger_binding = {key: contract[key] for key in contract
        if key != "partition_sha256"}
    body_without_ledger_binding["final_test_access_ledger"] = dict(ledger)
    body_without_ledger_binding["final_test_access_ledger"].pop("partition_sha256", None)
    if ledger.get("partition_sha256") != canonical_sha256(body_without_ledger_binding):
        raise ValueError("final-test access ledger does not bind the partition body")
    return dict(contract)


def _build_final_test_authorization_transition(partition_contract: Mapping[str, Any], *,
        prior_ledger_state: Mapping[str, Any], authorization: Mapping[str, Any],
        consumer_id: str, consumed_at_utc: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate authorization claims and build a receipt without reading outcomes."""
    if not isinstance(partition_contract, Mapping) or not isinstance(prior_ledger_state, Mapping):
        raise ValueError("partition contract and prior ledger state must be objects")
    if not isinstance(authorization, Mapping):
        raise ValueError("final-test authorization must be an object")
    contract_body = {key: value for key, value in partition_contract.items()
        if key != "partition_sha256"}
    partition_sha = canonical_sha256(contract_body)
    if partition_contract.get("partition_sha256") != partition_sha:
        raise ValueError("final-test authorization requires an intact partition contract")
    partition_ledger = partition_contract.get("final_test_access_ledger")
    if (not isinstance(partition_ledger, Mapping)
            or partition_ledger.get("partition_sha256") != canonical_sha256({
                key: value for key, value in contract_body.items()
                if key != "final_test_access_ledger"} | {
                    "final_test_access_ledger": {
                        key: value for key, value in partition_ledger.items()
                        if key != "partition_sha256"}})):
        raise ValueError("final-test ledger does not bind the immutable partition")
    expected_initial = {"schema": "meco-final-test-access-ledger-v1",
        "authorization_schema": "meco-final-test-single-use-authorization-v1",
        "max_authorizations": 1, "authorization_count": 0, "status": "locked",
        "outcomes_included": False, "outcomes_read": False,
        "consumption_receipts": [], "partition_sha256":
            partition_ledger.get("partition_sha256")}
    if dict(prior_ledger_state) == expected_initial:
        prior = dict(prior_ledger_state)
    else:
        prior = dict(prior_ledger_state)
        if (prior.get("partition_sha256") != expected_initial["partition_sha256"]
                or prior.get("authorization_count") != 0
                or prior.get("status") != "locked"
                or prior.get("consumption_receipts") != []
                or prior.get("outcomes_read") is not False
                or prior.get("outcomes_included") is not False):
            raise ValueError("final-test authorization slot is already consumed or invalid")
    if prior.get("authorization_count") != 0:
        raise ValueError("final-test authorization is single-use")
    required = {"schema", "partition_sha256", "approval_artifact_sha256",
        "approval_artifact",
        "authorized_by", "purpose", "issued_at_utc", "authorization_sha256"}
    if set(authorization) != required or authorization.get("schema") != \
            "meco-final-test-single-use-authorization-v1":
        raise ValueError("final-test authorization schema must include authenticated approval content bytes")
    unsigned_authorization = {key: value for key, value in authorization.items()
        if key != "authorization_sha256"}
    approval, approval_digest = _validate_pinned_json_artifact(
        authorization["approval_artifact"],
        trusted_digests=TRUSTED_FINAL_TEST_APPROVAL_SHA256,
        label="human final-test approval artifact")
    if (set(approval) != {"schema", "partition_sha256", "authorized_by", "purpose",
            "issued_at_utc"}
            or approval.get("schema") != "meco-final-test-human-approval-v1"
            or authorization["approval_artifact_sha256"] != approval_digest
            or authorization["partition_sha256"] != approval.get("partition_sha256")
            or authorization["authorized_by"] != approval.get("authorized_by")
            or authorization["purpose"] != approval.get("purpose")
            or authorization["issued_at_utc"] != approval.get("issued_at_utc")):
        raise ValueError("human approval artifact does not authenticate the final-test authorization")
    if (authorization["partition_sha256"] != partition_sha
            or authorization["purpose"] != "final_evaluation"
            or not isinstance(authorization["authorized_by"], str)
            or not authorization["authorized_by"].strip()
            or authorization["authorization_sha256"] != canonical_sha256(unsigned_authorization)):
        raise ValueError("final-test authorization does not match the partition or approval")
    if not isinstance(consumer_id, str) or not consumer_id.strip():
        raise ValueError("final-test consumer identity is required")
    _strict_utc_timestamp(authorization["issued_at_utc"], "authorization issue time")
    _strict_utc_timestamp(consumed_at_utc, "final-test transition time")
    receipt = {"schema": "meco-final-test-authorization-consumption-v1",
        "partition_sha256": partition_sha,
        "authorization_sha256": authorization["authorization_sha256"],
        "authorized_by": authorization["authorized_by"],
        "consumer_id": consumer_id, "consumed_at_utc": consumed_at_utc,
        "outcomes_read": False}
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    next_state = {"schema": "meco-final-test-access-ledger-state-v1",
        "partition_sha256": expected_initial["partition_sha256"],
        "authorization_count": 1, "status": "authorization_consumed",
        "outcomes_included": False, "outcomes_read": False,
        "consumption_receipts": [receipt["receipt_sha256"]]}
    return next_state, receipt


def consume_final_test_authorization(partition_contract: Mapping[str, Any], *,
        prior_ledger_state: Mapping[str, Any], authorization: Mapping[str, Any],
        consumer_id: str, consumed_at_utc: str,
        ledger_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Atomically consume one trusted approval through a persisted compare-and-swap ledger."""
    if ledger_path is None:
        raise ValueError("single-use final-test authorization requires a persisted ledger path")
    try:
        import fcntl
    except ImportError as exc:
        raise ValueError("this platform cannot enforce the final-test ledger lock") from exc
    path = Path(ledger_path).resolve()
    expected_path = (Path(FINAL_TEST_LEDGER_ROOT).resolve()
        / f"{partition_contract.get('partition_sha256')}.json")
    if path != expected_path:
        raise ValueError("final-test ledger path must match the policy root and immutable partition digest")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                try:
                    persisted_state = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError("persisted final-test ledger state is unreadable") from exc
                if persisted_state != dict(prior_ledger_state):
                    raise ValueError("final-test ledger compare-and-swap failed; authorization already consumed or stale")
            else:
                persisted_state = None
            next_state, receipt = _build_final_test_authorization_transition(
                partition_contract, prior_ledger_state=prior_ledger_state,
                authorization=authorization, consumer_id=consumer_id,
                consumed_at_utc=consumed_at_utc)
            initial_ledger = partition_contract.get("final_test_access_ledger")
            if persisted_state is None and dict(prior_ledger_state) != initial_ledger:
                raise ValueError("first ledger transition must start from the immutable locked partition ledger")
            receipt_dir = path.with_name(path.name + ".receipts")
            receipt_dir.mkdir(parents=True, exist_ok=True)
            receipt_path = receipt_dir / f"{authorization['authorization_sha256']}.json"
            receipt_data = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
            try:
                receipt_fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as exc:
                raise ValueError("final-test authorization already has an immutable consume receipt") from exc
            with os.fdopen(receipt_fd, "wb") as stream:
                stream.write(receipt_data)
                stream.flush()
                os.fsync(stream.fileno())
            receipt_directory_fd = os.open(receipt_dir, os.O_RDONLY)
            try:
                os.fsync(receipt_directory_fd)
            finally:
                os.close(receipt_directory_fd)
            data = json.dumps(next_state, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.",
                suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return next_state, receipt
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def write_immutable_artifact(directory: str | Path, artifact: Mapping[str, Any], *,
                             id_field: str) -> Path:
    """Atomically create a content-addressed JSON artifact without replacement."""
    if not isinstance(artifact, Mapping) or id_field not in artifact:
        raise ValueError("immutable artifact must carry its content-addressed ID")
    artifact_id = artifact[id_field]
    _strict_digest(artifact_id, id_field)
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{artifact_id}.json"
    content = json.dumps(artifact, sort_keys=True, indent=2, ensure_ascii=False,
        allow_nan=False).encode("utf-8") + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{artifact_id}.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != content:
                raise ValueError("refusing to overwrite an immutable artifact ID")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target
