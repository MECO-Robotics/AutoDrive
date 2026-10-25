"""Pure validators for serialized Stage 1 Phase 0 evidence.

This module intentionally uses only the Python standard library. It validates
captured declarations and digest links; it does not reproduce a factory or
inspect the current checkout.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import PurePosixPath

CAPTURE_RECEIPT_SCHEMA = "meco-effective-capture-receipt-v1"
EFFECTIVE_SNAPSHOT_SCHEMA = "meco-effective-configuration-snapshot-v1"
EFFECTIVE_FREEZE_SCHEMA = "meco-baseline-freeze-v3"
TUPLE_REGISTRY_SCHEMA = "meco-evaluation-tuple-registry-v1"
PARTITION_MATERIALIZATION_SCHEMA = "meco-heldout-partitions-v2"
CONFIG_IDS = frozenset({"campaign_assignment", "evaluation_protocol", "drivetrain_physics",
    "gamepiece_rules", "field_layout", "scenario_registry", "opponent_schedule", "sensor_profile"})
RUNTIME_FIELDS = frozenset({"simulator_id", "simulator_version", "simulator_build_sha256",
    "python_version", "torch_version", "backend", "backend_version", "device_id",
    "dependency_lock_sha256"})
TUPLE_FIELDS = ("scenario_id", "case_id", "category", "defender_family", "layout_id",
    "start_condition", "sensor_profile", "seed", "repetition", "evaluation_phase")
PUBLIC_FACTORY_ARGUMENTS = frozenset({"scenario_id", "category", "num_envs", "device",
    "seed", "horizon", "fuel_physics", "production_d0", "layout_id",
    "defender_family", "start_condition", "sensor_profile", "phase4_factor",
    "contact_geometry_id", "contact_geometry_config", "phase4_opponent_policy_id",
    "phase4_opponent_schedule", "phase4_opponent_seed",
    "phase4_start_profile_assignment", "phase4_focal_pose_assignment",
    "phase4_teammate_behavior_assignment", "evaluation_protocol_content_base64",
    "evaluation_protocol_object", "evaluation_protocol_content_sha256"})
PARTITIONS = ("train", "calibration", "validation", "final_test")
TRUSTED_ALLOCATION_ARTIFACT_SHA256 = frozenset()
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SAFE_RELATIVE_PATH = re.compile(r"^[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)*$")
ROUTE_COST_POINTS_PER_SECOND = 0.05
MATCHED_REFERENCE_VARIANCE_VERSION = "meco-matched-reference-variance-v1"
MATCHED_REFERENCE_CANDIDATE_K = 16


def validate_phase0_identity_binding(binding):
    """Validate typed, separate Phase 0 and evaluator source identities."""
    fields = {"effective_freeze_sha256", "source_commit", "source_tree",
        "source_plan_path", "source_plan_sha256", "runtime_sha256",
        "dependency_lock_sha256", "normalized_runtime_sha256", "config_sha256",
        "configuration_artifacts", "source_identity_sha256"}
    if not isinstance(binding, dict) or set(binding) != fields:
        raise ValueError("phase0_identity_binding_schema_invalid")
    for name in ("effective_freeze_sha256", "source_plan_sha256", "runtime_sha256",
            "dependency_lock_sha256", "normalized_runtime_sha256", "config_sha256",
            "source_identity_sha256"):
        _digest(binding[name], f"phase0_identity_{name}")
    for name in ("source_commit", "source_tree"):
        if not isinstance(binding[name], str) or HEX40.fullmatch(binding[name]) is None:
            raise ValueError("phase0_identity_git_digest_invalid")
    if not isinstance(binding["source_plan_path"], str) or not binding["source_plan_path"] or binding["source_plan_path"].startswith("/") or ".." in binding["source_plan_path"].split("/"):
        raise ValueError("phase0_identity_source_plan_path_invalid")
    config = binding["configuration_artifacts"]
    if not isinstance(config, dict) or set(config) != CONFIG_IDS:
        raise ValueError("phase0_identity_configuration_set_invalid")
    for digest in config.values():
        _digest(digest, "phase0_identity_configuration")
    if canonical_sha256(config) != binding["config_sha256"]:
        raise ValueError("phase0_identity_configuration_digest_mismatch")
    return binding


def validate_matched_reference_calibration_metadata(report, *, variant_table,
        variant_normalizer, identity_builder, protocol_cases_builder, d0_case_validator,
        runtime_normalizer, minimum_episodes, candidate_k, route_cost,
        safety_metric_names, safety_metrics_schema, safety_floor_policy):
    """Validate immutable matched-reference policy/runtime/case metadata.

    Pure callbacks supply the existing scenario and report contracts; this
    function itself performs no imports, device checks, or simulator work.
    """
    if (not isinstance(report, dict)
            or report.get("artifact_type") != MATCHED_REFERENCE_VARIANCE_VERSION
            or report.get("schema_version") != 1 or report.get("calibration") is not True):
        raise ValueError("unsupported matched reference variance report")
    config, scenario = report.get("config"), report.get("scenario_id")
    if not isinstance(config, dict) or config.get("scenario_id") != scenario:
        raise ValueError("matched reference variance config is invalid")
    variant = variant_normalizer(config.get("reference_selection_id"))
    if variant not in variant_table:
        raise ValueError("matched_reference_variant_invalid")
    policy, policy_id, mode = variant_table[variant]
    identity = identity_builder(variant)
    candidate_configuration = {"candidate_k": candidate_k,
        "route_cost_points_per_second": route_cost, "action_ticks": 12}
    episodes = config.get("episodes_per_category_stratum")
    repetitions, master_seed = config.get("repetitions"), config.get("master_seed")
    if (config.get("reference_policy") != policy
            or config.get("reference_policy_id") != policy_id
            or config.get("prediction_mode") != mode
            or config.get("prediction_contract") != identity["prediction_contract"]
            or config.get("candidate_configuration") != candidate_configuration
            or config.get("safety_floor_policy") != safety_floor_policy
            or config.get("required_safety_metrics") != sorted(safety_metric_names)
            or config.get("split") != "calibration"
            or type(episodes) is not int or episodes < minimum_episodes
            or type(repetitions) is not int or repetitions < 2
            or type(master_seed) is not int or master_seed < 0):
        raise ValueError("matched reference variance policy, mode, seed, or runtime settings are invalid")
    cases = protocol_cases_builder(scenario, "calibration", episodes=episodes,
        master_seed=master_seed)
    expected_cases = [{"category": case.category, "layout_id": case.layout_id,
        "defender_family": case.defender_family, "start_condition": case.start_condition,
        "sensor_profile": case.sensor_profile, "seeds": list(case.seeds),
        "reference_policy": policy} for case in cases]
    if report.get("case_ids") != expected_cases:
        raise ValueError("matched reference variance case/seed table differs from the frozen calibration protocol")
    d0_case_validator(scenario, episodes, master_seed,
        [{key: value for key, value in row.items() if key != "reference_policy"}
         for row in expected_cases])
    source_hash = report.get("source_identity_sha256")
    runtime = report.get("runtime_device")
    normalized_runtime = runtime_normalizer(runtime)
    if (report.get("normalized_runtime_identity") != normalized_runtime
            or report.get("safety_metrics_schema") != safety_metrics_schema
            or report.get("safety_metrics_source_identity_sha256") != source_hash):
        raise ValueError("matched reference variance source/runtime/safety identity is invalid")
    expected_protocol = _qualification_sha256({"version": MATCHED_REFERENCE_VARIANCE_VERSION,
        "config": config, "case_table": expected_cases,
        "source_identity_sha256": source_hash, "runtime_identity": normalized_runtime})
    if report.get("protocol_identity_sha256") != expected_protocol:
        raise ValueError("matched reference variance protocol identity is invalid")
    return {"config": config, "scenario": scenario, "variant": variant,
        "policy": policy, "policy_id": policy_id, "mode": mode,
        "identity": identity, "candidate_configuration": candidate_configuration,
        "episodes": episodes, "repetitions": repetitions,
        "master_seed": master_seed, "cases": cases,
        "expected_cases": expected_cases, "source_hash": source_hash,
        "runtime": runtime, "normalized_runtime": normalized_runtime}

# Kept byte-for-byte aligned with frc_defense.meco_baseline_freeze's authoritative closure.
AUTHORITATIVE_FIELD_SOURCES = {
    "robot_dynamics": ("frc_defense/curriculum_scenarios.py", "frc_defense/tensor_3v3.py", "frc_defense/tensor_sim.py", "frc_defense/tensor_physics.py", "frc_defense/fuel_physics_runtime.py"),
    "field_geometry": ("frc_defense/curriculum_scenarios.py", "frc_defense/tensor_3v3.py", "frc_defense/field.py"),
    "legal_connectivity": ("frc_defense/curriculum_scenarios.py", "frc_defense/tensor_3v3.py", "frc_defense/field.py", "frc_defense/tensor_adstar.py", "frc_defense/_tensor_3v3_navigation.py"),
    "opponent_controllers": ("frc_defense/curriculum_scenarios.py", "frc_defense/training_curriculum.py", "frc_defense/meco_track_training.py", "frc_defense/meco_phase4_opponents.py", "frc_defense/tensor_3v3.py", "frc_defense/_tensor_3v3_actions.py"),
    "action_update_cadence": ("frc_defense/curriculum_scenarios.py", "frc_defense/tensor_3v3.py", "frc_defense/_tensor_3v3_actions.py", "frc_defense/tensor_physics.py"),
    "sensor_noise_profiles": ("frc_defense/curriculum_scenarios.py", "frc_defense/tensor_3v3.py", "frc_defense/_tensor_3v3_observation.py", "frc_defense/tensor_perception.py"),
    "scoring_observation_semantics": ("frc_defense/curriculum_scenarios.py", "frc_defense/tensor_3v3.py", "frc_defense/_tensor_3v3_gamepieces.py", "frc_defense/meco_track_training.py", "frc_defense/meco_metric_ledger.py"),
    "safety_event_semantics": ("frc_defense/tensor_physics.py", "frc_defense/tensor_3v3.py", "frc_defense/_tensor_3v3_actions.py", "frc_defense/curriculum_scenarios.py", "frc_defense/meco_track_training.py", "frc_defense/meco_metric_ledger.py"),
}


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _qualification_sha256(value):
    """Preserve qualification's existing ensure_ascii=True digest encoding."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _digest(value, label):
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{label}_digest_invalid")
    return value


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("phase0_snapshot_protocol_duplicate_key")
        result[key] = value
    return result


def _runtime(value):
    if not isinstance(value, dict) or set(value) != RUNTIME_FIELDS | {"device_model", "device_index"}:
        raise ValueError("phase0_runtime_schema_invalid")
    for key, item in value.items():
        if key == "device_index":
            if type(item) is not int or item < 0:
                raise ValueError("phase0_runtime_device_index_invalid")
        elif not isinstance(item, str) or item.strip().lower() in {"", "unknown", "unavailable", "unspecified", "none", "null"}:
            raise ValueError("phase0_runtime_identity_incomplete")
        elif item != item.strip():
            raise ValueError("phase0_runtime_identity_not_normalized")
    for key in ("simulator_build_sha256", "dependency_lock_sha256"):
        _digest(value[key], f"phase0_runtime_{key}")
    if value["device_id"].startswith(("auto", "cuda", "hip", "rocm")):
        raise ValueError("phase0_runtime_device_id_invalid")
    return value


def _normalized_calibration_runtime(runtime):
    """Mirror qualification's normalized, already-typed calibration runtime."""
    backend = runtime.get("backend") if isinstance(runtime, dict) else None
    if backend not in {"hip", "cuda"}:
        raise ValueError("phase0_runtime_backend_invalid")
    if (not isinstance(runtime.get("device"), str)
            or re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", runtime["device"]) is None
            or not isinstance(runtime.get("torch_version"), str)
            or not runtime["torch_version"].strip()
            or runtime["torch_version"] != runtime["torch_version"].strip()
            or not isinstance(runtime.get("device_name"), str)
            or not runtime["device_name"].strip()
            or runtime["device_name"] != runtime["device_name"].strip()):
        raise ValueError("phase0_runtime_calibration_identity_invalid")
    hip_version = runtime.get("hip_version")
    if backend == "hip":
        if (not isinstance(hip_version, str) or not hip_version.strip()
                or hip_version != hip_version.strip()):
            raise ValueError("phase0_runtime_hip_version_invalid")
    elif hip_version is not None:
        raise ValueError("phase0_runtime_cuda_hip_version_invalid")
    # Mirror _normalized_calibration_runtime: omit device selector and expose
    # the runtime device name under both `device_name` and `device_model`.
    normalized = {key: value for key, value in runtime.items() if key != "device"}
    normalized["device_model"] = runtime["device_name"]
    return normalized


def _phase0_runtime_as_calibration_runtime(runtime):
    backend = {"ROCm": "hip", "CUDA": "cuda"}.get(runtime["backend"])
    if backend is None:
        raise ValueError("phase0_runtime_backend_invalid")
    return _normalized_calibration_runtime({
        "device": f"cuda:{runtime['device_index']}",
        "device_name": runtime["device_model"], "backend": backend,
        "torch_version": runtime["torch_version"],
        "hip_version": runtime["backend_version"] if backend == "hip" else None})


def _config_artifacts(value, *, snapshot=False):
    if snapshot:
        if not isinstance(value, dict) or set(value) != CONFIG_IDS:
            raise ValueError("phase0_snapshot_config_set_invalid")
        rows = []
        for artifact_id in sorted(CONFIG_IDS):
            item = value[artifact_id]
            if not isinstance(item, dict) or set(item) != {"content_base64", "content_sha256"}:
                raise ValueError("phase0_snapshot_config_schema_invalid")
            raw = base64.b64decode(item["content_base64"], validate=True)
            parsed = json.loads(raw)
            if json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() != raw:
                raise ValueError("phase0_config_content_not_canonical")
            digest = hashlib.sha256(raw).hexdigest()
            if digest != _digest(item["content_sha256"], "phase0_config_content"):
                raise ValueError("phase0_config_content_digest_mismatch")
            rows.append({"artifact_id": artifact_id, "content_sha256": digest})
        return {row["artifact_id"]: row["content_sha256"] for row in rows}
    if not isinstance(value, list) or len(value) != len(CONFIG_IDS):
        raise ValueError("phase0_receipt_config_set_invalid")
    result = {}
    ids = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"artifact_id", "content_sha256", "content_base64"}:
            raise ValueError("phase0_receipt_config_schema_invalid")
        artifact = item["artifact_id"]
        if artifact not in CONFIG_IDS or artifact in result:
            raise ValueError("phase0_receipt_config_id_invalid")
        ids.append(artifact)
        raw = base64.b64decode(item["content_base64"], validate=True)
        parsed = json.loads(raw)
        if json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() != raw:
            raise ValueError("phase0_config_content_not_canonical")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != _digest(item["content_sha256"], "phase0_config_content"):
            raise ValueError("phase0_config_content_digest_mismatch")
        result[artifact] = digest
    if set(result) != CONFIG_IDS or ids != sorted(CONFIG_IDS):
        raise ValueError("phase0_receipt_config_set_invalid")
    return result


def _validate_snapshot(snapshot, receipt_source, receipt_runtime):
    keys = {"schema", "producer", "factory_arguments", "resolved_fields", "configuration_artifacts", "runtime_identity", "snapshot_sha256"}
    if not isinstance(snapshot, dict) or set(snapshot) != keys or snapshot.get("schema") != EFFECTIVE_SNAPSHOT_SCHEMA:
        raise ValueError("phase0_snapshot_schema_invalid")
    body = {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}
    if _digest(snapshot["snapshot_sha256"], "phase0_snapshot") != canonical_sha256(body):
        raise ValueError("phase0_snapshot_digest_mismatch")
    producer = snapshot["producer"]
    if not isinstance(producer, dict) or set(producer) != {"module", "function", "source_identity_sha256"} or producer.get("module") != "frc_defense.curriculum_scenarios" or producer.get("function") != "make_curriculum_env":
        raise ValueError("phase0_snapshot_producer_invalid")
    _digest(producer.get("source_identity_sha256"), "phase0_snapshot_source_identity")
    source_identity = {key: receipt_source[key] for key in
        ("commit", "tree", "clean", "source_plan_path", "source_plan_sha256")}
    if producer["source_identity_sha256"] != canonical_sha256(source_identity):
        raise ValueError("phase0_snapshot_source_identity_mismatch")
    required_arguments = {"scenario_id", "category", "num_envs", "device", "seed",
        "fuel_physics", "production_d0", "layout_id", "defender_family",
        "start_condition", "sensor_profile", "evaluation_protocol_object",
        "evaluation_protocol_content_base64", "evaluation_protocol_content_sha256"}
    if (not isinstance(snapshot["factory_arguments"], dict)
            or not required_arguments <= set(snapshot["factory_arguments"])
            or not set(snapshot["factory_arguments"]) <= PUBLIC_FACTORY_ARGUMENTS
            or any(not isinstance(k, str) or k.startswith("_") for k in snapshot["factory_arguments"])):
        raise ValueError("phase0_snapshot_factory_arguments_invalid")
    snapshot_config = _config_artifacts(snapshot["configuration_artifacts"], snapshot=True)
    arguments = snapshot["factory_arguments"]
    protocol_artifact = snapshot["configuration_artifacts"]["evaluation_protocol"]
    try:
        protocol_bytes = base64.b64decode(arguments["evaluation_protocol_content_base64"], validate=True)
        artifact_bytes = base64.b64decode(protocol_artifact["content_base64"], validate=True)
        protocol_object = json.loads(protocol_bytes, object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("phase0_snapshot_protocol_bytes_invalid") from exc
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if (not isinstance(protocol_object, dict)
            or protocol_bytes != artifact_bytes
            or protocol_sha256 != _digest(arguments["evaluation_protocol_content_sha256"],
                "phase0_snapshot_protocol")
            or protocol_sha256 != _digest(protocol_artifact["content_sha256"],
                "phase0_snapshot_protocol_artifact")
            or protocol_object != arguments["evaluation_protocol_object"]):
        raise ValueError("phase0_snapshot_protocol_binding_mismatch")
    if snapshot["runtime_identity"] != receipt_runtime:
        raise ValueError("phase0_snapshot_runtime_binding_mismatch")
    if _runtime(snapshot["runtime_identity"]) != _runtime(receipt_runtime):
        raise ValueError("phase0_snapshot_runtime_invalid")
    fields = snapshot["resolved_fields"]
    if not isinstance(fields, dict) or set(fields) != set(AUTHORITATIVE_FIELD_SOURCES):
        raise ValueError("phase0_snapshot_field_set_invalid")
    if not isinstance(receipt_source, dict):
        raise ValueError("phase0_receipt_source_invalid")
    return snapshot_config


def _validate_receipt(receipt):
    required = {"schema", "capture_tool_version", "captured_at_utc", "source", "effective_snapshot_sha256", "effective_snapshot", "fields", "configuration_artifacts", "runtime", "qualification_eligible", "receipt_sha256"}
    if not isinstance(receipt, dict) or set(receipt) != required or receipt.get("schema") != CAPTURE_RECEIPT_SCHEMA or receipt.get("capture_tool_version") != 1 or receipt.get("qualification_eligible") is not False:
        raise ValueError("phase0_receipt_schema_invalid")
    captured = receipt.get("captured_at_utc")
    if not isinstance(captured, str) or not captured.endswith("Z"):
        raise ValueError("phase0_receipt_timestamp_invalid")
    try:
        parsed = datetime.fromisoformat(captured[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("phase0_receipt_timestamp_invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("phase0_receipt_timestamp_invalid")
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if _digest(receipt["receipt_sha256"], "phase0_receipt") != canonical_sha256(body):
        raise ValueError("phase0_receipt_digest_mismatch")
    source = receipt["source"]
    if not isinstance(source, dict) or set(source) != {"commit", "tree", "clean", "source_plan_path", "source_plan_sha256", "authoritative_source_sha256"}:
        raise ValueError("phase0_source_identity_schema_invalid")
    if any(not isinstance(source.get(k), str) or HEX40.fullmatch(source[k]) is None for k in ("commit", "tree")) or source.get("clean") is not True:
        raise ValueError("phase0_source_commit_tree_invalid")
    plan = source.get("source_plan_path")
    if (not isinstance(plan, str) or not SAFE_RELATIVE_PATH.fullmatch(plan)
            or plan.startswith(".")):
        raise ValueError("phase0_source_plan_path_invalid")
    _digest(source.get("source_plan_sha256"), "phase0_source_plan")
    authoritative = source.get("authoritative_source_sha256")
    expected_paths = {p for paths in AUTHORITATIVE_FIELD_SOURCES.values() for p in paths}
    if not isinstance(authoritative, dict) or set(authoritative) != expected_paths:
        raise ValueError("phase0_authoritative_source_closure_invalid")
    for path, digest in authoritative.items():
        if not isinstance(path, str) or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("phase0_authoritative_source_path_invalid")
        _digest(digest, "phase0_authoritative_source")
    runtime = _runtime(receipt["runtime"])
    snapshot = receipt["effective_snapshot"]
    if not isinstance(snapshot, dict):
        raise ValueError("phase0_receipt_snapshot_invalid")
    if _digest(receipt["effective_snapshot_sha256"], "phase0_snapshot_link") != snapshot.get("snapshot_sha256"):
        raise ValueError("phase0_receipt_snapshot_link_mismatch")
    config = _config_artifacts(receipt["configuration_artifacts"])
    snapshot_config = _validate_snapshot(snapshot, source, runtime)
    if config != snapshot_config:
        raise ValueError("phase0_receipt_snapshot_config_mismatch")
    fields = receipt["fields"]
    if not isinstance(fields, dict) or set(fields) != set(AUTHORITATIVE_FIELD_SOURCES):
        raise ValueError("phase0_receipt_field_set_invalid")
    for name, paths in AUTHORITATIVE_FIELD_SOURCES.items():
        field = fields[name]
        if not isinstance(field, dict) or set(field) != {"value", "value_sha256", "selector", "runtime_owner", "resolution_state", "derivation", "sources"}:
            raise ValueError("phase0_receipt_field_schema_invalid")
        if not isinstance(field["value"], (dict, list)) or not field["value"]:
            raise ValueError("phase0_receipt_field_value_invalid")
        if name == "scoring_observation_semantics" and field["value"] != {
                "simulator_metric": "fuel_score_count",
                "metric_semantics": "scored_fuel_piece_count",
                "official_match_points_available": False}:
            raise ValueError("phase0_scoring_semantics_invalid")
        if canonical_sha256(field["value"]) != _digest(field["value_sha256"], "phase0_field_value"):
            raise ValueError("phase0_receipt_field_digest_mismatch")
        if not isinstance(field["selector"], str) or not field["selector"].strip() or not isinstance(field["runtime_owner"], str) or not field["runtime_owner"].strip() or field["resolution_state"] not in {"resolved", "overridden", "inherited"}:
            raise ValueError("phase0_receipt_field_provenance_invalid")
        derivation = field["derivation"]
        if not isinstance(derivation, dict) or set(derivation) != {"kind", "input_sha256"} or derivation["kind"] not in {"factory_argument", "resolved_scenario", "runtime_object"}:
            raise ValueError("phase0_receipt_field_derivation_invalid")
        _digest(derivation["input_sha256"], "phase0_field_derivation")
        refs = field["sources"]
        if not isinstance(refs, list) or len(refs) != len(paths) or {r.get("path") for r in refs if isinstance(r, dict)} != set(paths):
            raise ValueError("phase0_field_source_closure_invalid")
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or authoritative.get(ref["path"]) != _digest(ref["sha256"], "phase0_field_source"):
                raise ValueError("phase0_field_source_link_invalid")
        snap_field = snapshot["resolved_fields"].get(name)
        expected_snap = {k: field[k] for k in ("value", "selector", "runtime_owner", "resolution_state", "derivation")}
        if snap_field != expected_snap:
            raise ValueError("phase0_receipt_snapshot_field_mismatch")
    return {"receipt": receipt, "snapshot": snapshot, "source": source,
        "runtime": runtime, "config_sha256": canonical_sha256(config),
        "configuration_artifacts": config}


def _normalize_registry(registry):
    if not isinstance(registry, dict) or set(registry) != {"schema", "tuples"} or registry.get("schema") != TUPLE_REGISTRY_SCHEMA or not isinstance(registry.get("tuples"), list) or not registry["tuples"]:
        raise ValueError("phase0_tuple_registry_invalid")
    normalized, by_id = [], {}
    for item in registry["tuples"]:
        if not isinstance(item, dict) or type(item.get("eligible")) is not bool:
            raise ValueError("phase0_tuple_entry_invalid")
        eligible = item["eligible"]
        if set(item) != set(TUPLE_FIELDS) | {"eligible"} | ({"exclusion_reason"} if not eligible else set()):
            raise ValueError("phase0_tuple_entry_schema_invalid")
        row = {k: item[k] for k in TUPLE_FIELDS}
        for key in ("scenario_id", "case_id", "category", "defender_family", "layout_id", "start_condition", "sensor_profile", "evaluation_phase"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("phase0_tuple_identity_invalid")
        for key in ("seed", "repetition"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("phase0_tuple_number_invalid")
        if not eligible and (not isinstance(item.get("exclusion_reason"), str) or not item["exclusion_reason"].strip()):
            raise ValueError("phase0_tuple_exclusion_reason_invalid")
        tuple_id = canonical_sha256(row)
        if tuple_id in by_id:
            raise ValueError("phase0_tuple_duplicate")
        normalized_row = row | {"tuple_id": tuple_id, "eligible": eligible}
        if not eligible:
            normalized_row["exclusion_reason"] = item["exclusion_reason"]
        normalized.append(normalized_row)
        by_id[tuple_id] = normalized_row
    normalized.sort(key=lambda r: r["tuple_id"])
    return {"schema": TUPLE_REGISTRY_SCHEMA, "tuples": normalized}, by_id


def validate_phase0_chain(chain):
    """Validate receipt → snapshot → freeze → tuple registry/partition objects."""
    if not isinstance(chain, dict) or set(chain) != {"capture_receipt", "effective_snapshot", "effective_freeze", "tuple_registry", "partition_materialization"}:
        raise ValueError("phase0_chain_schema_invalid")
    receipt_result = _validate_receipt(chain["capture_receipt"])
    receipt = chain["capture_receipt"]
    snapshot = chain["effective_snapshot"]
    if snapshot != receipt["effective_snapshot"]:
        raise ValueError("phase0_chain_snapshot_receipt_mismatch")
    freeze = chain["effective_freeze"]
    freeze_fields = {"schema", "status", "capture_receipt_sha256", "source", "runtime", "fields", "configuration_artifacts", "qualification_eligible", "freeze_sha256"}
    if not isinstance(freeze, dict) or set(freeze) != freeze_fields or freeze.get("schema") != EFFECTIVE_FREEZE_SCHEMA or freeze.get("status") != "complete_source_runtime_bound_effective_baseline" or freeze.get("qualification_eligible") is not False:
        raise ValueError("phase0_freeze_schema_invalid")
    freeze_body = {k: v for k, v in freeze.items() if k != "freeze_sha256"}
    if _digest(freeze["freeze_sha256"], "phase0_freeze") != canonical_sha256(freeze_body):
        raise ValueError("phase0_freeze_digest_mismatch")
    if (freeze.get("capture_receipt_sha256") != receipt["receipt_sha256"] or freeze.get("source") != receipt["source"] or freeze.get("runtime") != receipt["runtime"] or freeze.get("fields") != receipt["fields"] or freeze.get("configuration_artifacts") != receipt["configuration_artifacts"]):
        raise ValueError("phase0_freeze_receipt_link_mismatch")
    registry, by_id = _normalize_registry(chain["tuple_registry"])
    partition = chain["partition_materialization"]
    if not isinstance(partition, dict):
        raise ValueError("phase0_partition_schema_invalid")
    assignment = partition.get("assignment_table")
    if not isinstance(assignment, dict) or set(assignment) != {"schema", "assignments", "model_tracks", "evaluation_budget", "allocation_provenance"} or assignment.get("schema") != "meco-partition-assignment-table-v1":
        raise ValueError("phase0_assignment_table_invalid")
    if partition.get("tuple_registry") != registry or partition.get("tuple_registry_sha256") != canonical_sha256(registry):
        raise ValueError("phase0_partition_registry_link_invalid")
    if partition.get("baseline_freeze_sha256") != freeze["freeze_sha256"] or partition.get("assignment_table_sha256") != canonical_sha256(assignment):
        raise ValueError("phase0_partition_digest_link_invalid")
    assignments = assignment["assignments"]
    eligible = {tid for tid, row in by_id.items() if row["eligible"]}
    if not isinstance(assignments, dict) or set(assignments) != eligible:
        raise ValueError("phase0_assignment_coverage_invalid")
    part_ids = {p: [] for p in PARTITIONS}
    for tid, name in assignments.items():
        if name not in PARTITIONS or by_id[tid]["evaluation_phase"] != name:
            raise ValueError("phase0_assignment_phase_invalid")
        part_ids[name].append(tid)
    if any(not values for values in part_ids.values()):
        raise ValueError("phase0_partition_empty")
    for axis in ("seed", "defender_family", "layout_id", "start_condition"):
        owners = {}
        for name, tids in part_ids.items():
            for tid in tids:
                value = by_id[tid][axis]
                if value in owners:
                    raise ValueError("phase0_partition_leakage")
                owners[value] = name
    tracks = assignment["model_tracks"]
    if not isinstance(tracks, dict) or set(tracks) != {"A1", "A2", "A3", "C1", "C2"}:
        raise ValueError("phase0_partition_track_set_invalid")
    budget = assignment["evaluation_budget"]
    if not isinstance(budget, dict) or set(budget) != {"episodes_per_tuple", "repetitions"} or any(type(v) is not int or v < 1 for v in budget.values()):
        raise ValueError("phase0_partition_budget_invalid")
    for track in tracks.values():
        if not isinstance(track, dict) or set(track) != {"tuple_ids", "evaluation_budget"} or track["tuple_ids"] != sorted(eligible) or track["evaluation_budget"] != budget:
            raise ValueError("phase0_partition_track_binding_invalid")
    provenance = assignment["allocation_provenance"]
    pkeys = {"authored_by", "authored_at_utc", "source_sha256", "outcomes_accessed"}
    if not isinstance(provenance, dict) or set(provenance) not in (pkeys, pkeys | {"authorization_artifact"}) or not isinstance(provenance.get("authored_by"), str) or not provenance["authored_by"].strip() or provenance.get("outcomes_accessed") is not False:
        raise ValueError("phase0_allocation_provenance_invalid")
    if not isinstance(provenance.get("authored_at_utc"), str) or not provenance["authored_at_utc"].strip():
        raise ValueError("phase0_allocation_timestamp_invalid")
    _digest(provenance.get("source_sha256"), "phase0_allocation_source")
    authorized = False
    auth_sha = None
    if "authorization_artifact" in provenance:
        artifact = provenance["authorization_artifact"]
        if not isinstance(artifact, dict) or set(artifact) != {"content_base64", "content_sha256"}:
            raise ValueError("phase0_allocation_authorization_invalid")
        try:
            raw = base64.b64decode(artifact["content_base64"], validate=True)
            def reject_duplicate_pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("phase0_allocation_authorization_duplicate_key")
                    result[key] = value
                return result
            authorization = json.loads(raw, object_pairs_hook=reject_duplicate_pairs)
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("phase0_allocation_authorization_json_invalid") from exc
        if not isinstance(authorization, dict) or json.dumps(authorization,
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False).encode("utf-8") != raw:
            raise ValueError("phase0_allocation_authorization_not_canonical")
        auth_sha = hashlib.sha256(raw).hexdigest()
        if auth_sha != _digest(artifact["content_sha256"], "phase0_allocation_authorization"):
            raise ValueError("phase0_allocation_authorization_digest_mismatch")
        expected_authorization = {key: assignment[key] for key in
            ("schema", "assignments", "model_tracks", "evaluation_budget")}
        if authorization != expected_authorization:
            raise ValueError("phase0_allocation_authorization_content_mismatch")
        authorized = auth_sha in TRUSTED_ALLOCATION_ARTIFACT_SHA256
        if not authorized:
            raise ValueError("phase0_allocation_authorization_untrusted")
    for p, ids in part_ids.items():
        ids.sort()
        expected = {"tuple_ids": ids, "tuple_set_sha256": canonical_sha256(ids), "allowed_uses": {
            "train": ["training"], "calibration": ["calibration", "margin_review", "reference_selection"],
            "validation": ["model_selection", "validation"], "final_test": ["final_evaluation"]}[p]}
        if partition.get("partitions", {}).get(p) != expected:
            raise ValueError("phase0_partition_closure_invalid")
    uses = {"train": ["training"], "calibration": ["calibration", "margin_review", "reference_selection"],
        "validation": ["model_selection", "validation"], "final_test": ["final_evaluation"]}
    excluded = [{"tuple_id": row["tuple_id"], "reason": row["exclusion_reason"]}
        for row in registry["tuples"] if not row["eligible"]]
    expected_body = {"schema": PARTITION_MATERIALIZATION_SCHEMA,
        "status": "closed_tuple_partition_contract_only",
        "baseline_freeze_sha256": freeze["freeze_sha256"],
        "tuple_registry": registry, "tuple_registry_sha256": canonical_sha256(registry),
        "assignment_table": assignment, "assignment_table_sha256": canonical_sha256(assignment),
        "partitions": {name: {"tuple_ids": part_ids[name],
            "tuple_set_sha256": canonical_sha256(part_ids[name]), "allowed_uses": uses[name]}
            for name in PARTITIONS},
        "model_tracks": tracks, "evaluation_budget": budget,
        "allocation_authorized": authorized, "allocation_artifact_sha256": auth_sha,
        "final_test_access_ledger": {"schema": "meco-final-test-access-ledger-v1",
            "authorization_schema": "meco-final-test-single-use-authorization-v1",
            "max_authorizations": 1, "authorization_count": 0, "status": "locked",
            "outcomes_included": False, "outcomes_read": False, "consumption_receipts": []},
        "excluded_tuple_reasons": excluded, "qualification_eligible": False}
    ledger_bound = dict(expected_body)
    ledger_bound["final_test_access_ledger"] = dict(expected_body["final_test_access_ledger"])
    ledger_bound["final_test_access_ledger"]["partition_sha256"] = canonical_sha256(ledger_bound)
    expected_body["final_test_access_ledger"]["partition_sha256"] = ledger_bound["final_test_access_ledger"]["partition_sha256"]
    if partition != expected_body | {"partition_sha256": canonical_sha256(expected_body)}:
        raise ValueError("phase0_partition_self_digest_invalid")
    if partition.get("allocation_authorized") is not authorized or auth_sha != partition.get("allocation_artifact_sha256"):
        raise ValueError("phase0_partition_authorization_binding_invalid")
    result = {"capture_receipt_sha256": receipt["receipt_sha256"],
        "effective_snapshot_sha256": snapshot["snapshot_sha256"],
        "effective_freeze_sha256": freeze["freeze_sha256"],
        "source": freeze["source"], "runtime": freeze["runtime"],
        "runtime_sha256": canonical_sha256(freeze["runtime"]),
        "dependency_lock_sha256": freeze["runtime"]["dependency_lock_sha256"],
        "normalized_runtime": _phase0_runtime_as_calibration_runtime(freeze["runtime"]),
        "normalized_runtime_sha256": canonical_sha256(
            _phase0_runtime_as_calibration_runtime(freeze["runtime"])),
        "config_sha256": receipt_result["config_sha256"],
        "configuration_artifacts": receipt_result["configuration_artifacts"],
        "tuple_registry_sha256": partition["tuple_registry_sha256"],
        "partition_sha256": partition["partition_sha256"],
        "assignment_table_sha256": partition["assignment_table_sha256"],
        "allocation_authorized": authorized,
        "validation_seeds": sorted({row["seed"] for tid, row in by_id.items() if assignments[tid] == "validation"}),
        "final_test_seeds": sorted({row["seed"] for tid, row in by_id.items() if assignments[tid] == "final_test"})}
    return result


def validate_phase0_index(index, chain, artifact_sha256s):
    """Cross-bind the canonical Phase 0 index to a validated serialized chain."""
    required = {"schema", "source_identity_sha256", "source", "runtime_sha256", "dependency_lock_sha256", "normalized_runtime_sha256", "config_sha256", "configuration_artifacts", "capture_receipt_sha256", "effective_snapshot_sha256", "effective_freeze_sha256", "tuple_registry_sha256", "partition_sha256", "assignment_table_sha256", "artifacts", "index_sha256"}
    if not isinstance(index, dict) or set(index) != required or index.get("schema") != "meco-stage1-phase0-evidence-index-v1":
        raise ValueError("phase0_index_schema_invalid")
    if _digest(index["index_sha256"], "phase0_index") != canonical_sha256({k: v for k, v in index.items() if k != "index_sha256"}):
        raise ValueError("phase0_index_digest_invalid")
    _digest(index["source_identity_sha256"], "phase0_evaluator_source")
    chain_ids = validate_phase0_chain(chain)
    if index["source"] != {k: chain_ids["source"][k] for k in ("commit", "tree", "source_plan_path", "source_plan_sha256")}:
        raise ValueError("phase0_index_source_binding_mismatch")
    for key in ("runtime_sha256", "dependency_lock_sha256", "normalized_runtime_sha256", "config_sha256", "capture_receipt_sha256", "effective_snapshot_sha256", "effective_freeze_sha256", "tuple_registry_sha256", "partition_sha256", "assignment_table_sha256"):
        if index[key] != chain_ids.get(key):
            raise ValueError(f"phase0_index_{key}_mismatch")
    if index["dependency_lock_sha256"] != chain_ids["runtime"]["dependency_lock_sha256"]:
        raise ValueError("phase0_index_dependency_lock_runtime_mismatch")
    if index["configuration_artifacts"] != chain_ids["configuration_artifacts"]:
        raise ValueError("phase0_index_configuration_binding_mismatch")
    artifact_names = {"capture_receipt", "effective_snapshot", "effective_freeze",
        "tuple_registry", "partition_materialization", "score_estimand_review",
        "d0_calibration_bundle", "matched_reference_calibration_bundle", "margin_review"}
    if not isinstance(index["artifacts"], dict) or set(index["artifacts"]) != artifact_names or not isinstance(artifact_sha256s, dict) or set(artifact_sha256s) != artifact_names:
        raise ValueError("phase0_index_artifact_binding_mismatch")
    for name, ref in index["artifacts"].items():
        ref_path = ref.get("path") if isinstance(ref, dict) else None
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref_path, str) or not ref_path
                or PurePosixPath(ref_path).is_absolute() or ".." in PurePosixPath(ref_path).parts):
            raise ValueError("phase0_index_artifact_reference_invalid")
        if _digest(ref["sha256"], "phase0_index_artifact") != artifact_sha256s[name]:
            raise ValueError("phase0_index_artifact_byte_digest_mismatch")
    return chain_ids | {"source_identity_sha256": index["source_identity_sha256"], "source": index["source"], "index_sha256": index["index_sha256"]}
