"""Frozen, paired MECO A/B matrix orchestration.

The matrix stores exact checkpoint/source/seed provenance and delegates each
condition to ``evaluate_meco``. It reports paired episode rows; it does not
compute promotion confidence intervals or claim simulator qualification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .meco_tracks import MECO_TRACKS, PREDICTION_MODES
from .meco_phase4 import PHASE4_FACTORS, phase4_factor_config, phase4_factor_digest
from .meco_device import normalized_device_info
from .training_curriculum import get_scenario


MATRIX_SCHEMA_VERSION = 4
POLICY_CONTRASTS = {
    "A1_vs_C2": (("A1", "C2"),),
    "C1_vs_C2": (("C1", "C2"),),
    "A1_A2_A3": (("A1", "A2"), ("A1", "A3"), ("A2", "A3")),
    "C1_vs_D1": (("C1", "D1"),),
    "D1_vs_D2": (("D1", "D2"),),
}
DEFAULT_EVALUATION_STRATUM = {
    "layout_id": "2026_rebuilt",
    "defender_family": "category_scripted",
    "start_condition": "scenario_defined",
    "sensor_profile": "standard",
}


@dataclass(frozen=True)
class MatrixCondition:
    condition_name: str
    prediction_mode: str
    control_prediction_mode: str | None
    candidate_k: int
    inference_disable_tick: int | None = None
    phase4_factor: dict | None = None
    categories: tuple[str, ...] | None = None
    qualification_case: dict | None = None

    def validate(self):
        if self.prediction_mode not in PREDICTION_MODES:
            raise ValueError("unknown learned-track prediction mode")
        if (self.control_prediction_mode is not None and
                self.control_prediction_mode not in PREDICTION_MODES):
            raise ValueError("unknown control prediction mode")
        if self.candidate_k not in (8, 16, 32):
            raise ValueError("candidate K must be 8, 16, or 32")
        if (self.inference_disable_tick is not None and
                (type(self.inference_disable_tick) is not int or
                 self.inference_disable_tick <= 0)):
            raise ValueError("inference disable tick must be a positive integer")
        if self.phase4_factor is not None:
            if (not isinstance(self.phase4_factor, dict) or
                    self.phase4_factor.get("kind") not in PHASE4_FACTORS or
                    self.phase4_factor != PHASE4_FACTORS[self.phase4_factor["kind"]]):
                raise ValueError("Phase 4 factor differs from its frozen contract")
        if self.qualification_case is not None:
            if (not isinstance(self.qualification_case, dict) or
                    set(self.qualification_case) != set(DEFAULT_EVALUATION_STRATUM) or
                    any(not isinstance(value, str) or not value
                        for value in self.qualification_case.values())):
                raise ValueError("qualification case must freeze the four stratum identifiers")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _checkpoint_hashes(checkpoints: dict[str, str]) -> dict[str, str]:
    return {track: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for track, path in sorted(checkpoints.items())}


def _source_identity_hash(source_identity: dict) -> str:
    return hashlib.sha256(json.dumps(source_identity, sort_keys=True,
        allow_nan=False).encode()).hexdigest()


def _checkpoint_configuration_provenance(checkpoints: dict[str, str]):
    """Freeze exact saved config/protocol metadata before any condition runs."""
    import torch

    config_hashes, protocol_ids = {}, {}
    shared_config = None
    for track in MECO_TRACKS:
        payload = torch.load(checkpoints[track], map_location="cpu",
                             weights_only=False)
        config = payload.get("config")
        if not isinstance(config, dict) or config.get("track") != track:
            raise ValueError(f"{track} checkpoint has no matching saved configuration")
        immutable = {key: value for key, value in config.items() if key != "updates"}
        config_hashes[track] = hashlib.sha256(json.dumps(immutable,
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        shared = {key: value for key, value in immutable.items()
                  if key not in ("track", "device", "output")}
        if shared_config is None:
            shared_config = shared
        elif shared != shared_config:
            raise ValueError("checkpoints do not share immutable training settings")
        protocol_ids[track] = payload.get("protocol_id")
    if any(not isinstance(value, str) or not value for value in protocol_ids.values()):
        raise ValueError("checkpoint is missing its supervisor protocol ID")
    if any(value != protocol_ids[MECO_TRACKS[0]] for value in protocol_ids.values()):
        raise ValueError("checkpoints do not share one supervisor protocol")
    return config_hashes, protocol_ids, shared_config


def _held_out_seed_assignments(protocol: dict, scenario_id: str,
                               categories: tuple[str, ...],
                               seeds: tuple[int, ...]) -> dict[str, dict[str, str]]:
    """Require each requested case seed to belong to its frozen validation/test split."""
    assignments = {}
    for category in categories:
        cases = [case for case in protocol.get("cases", [])
                 if case.get("scenario_id") == scenario_id
                 and case.get("category") == category]
        category_assignments = {}
        for seed in seeds:
            matches = [case for case in cases if seed in case.get("seeds", [])]
            if (len(matches) != 1 or
                    matches[0]["split"] not in ("validation", "test")):
                raise ValueError(
                    f"matrix seed {seed} for {scenario_id}/{category} is not assigned "
                    "to exactly one frozen validation/test split")
            category_assignments[str(seed)] = matches[0]["split"]
        assignments[category] = category_assignments
    return assignments


def _validate_condition_manifest(report: dict, *,
                                 frozen_checkpoint_hashes: dict[str, str],
                                 frozen_source_hash: str,
                                 expected_qualification_protocol_id: str | None,
                                 expected_configs: dict[str, str] | None,
                                 expected_protocols: dict[str, str] | None,
                                 expected_shared_config: dict | None,
                                 expected_device: dict[str, str],
                                 expected_qualification_case: dict):
    """Reject an evaluation whose loaded bundle differs from matrix inputs."""
    manifest = report.get("policy_manifest")
    if not isinstance(manifest, dict) or set(MECO_TRACKS) - set(manifest):
        raise ValueError("evaluation manifest is missing one or more learned tracks")
    if report.get("source_identity_sha256") != frozen_source_hash:
        raise ValueError("evaluation source identity differs from frozen matrix provenance")
    if report.get("qualification_protocol_id") != expected_qualification_protocol_id:
        raise ValueError("evaluation qualification protocol ID differs from frozen matrix provenance")
    if report.get("execution_device") != expected_device:
        raise ValueError("evaluation report device/backend differs from frozen matrix provenance")
    if report.get("qualification_case") != expected_qualification_case:
        raise ValueError("evaluation report stratum differs from frozen matrix qualification case")
    component_timing = report.get("component_timing")
    timing_provenance = (component_timing.get("provenance")
                         if isinstance(component_timing, dict) else None)
    if (not isinstance(timing_provenance, dict)
            or timing_provenance.get("device") != expected_device["device"]
            or timing_provenance.get("accelerator_backend") != expected_device["accelerator_backend"]):
        raise ValueError("evaluation timing provenance device/backend differs from frozen matrix provenance")
    observed_configs, observed_protocols = {}, {}
    for track in MECO_TRACKS:
        policy = manifest[track]
        if policy.get("checkpoint_sha256") != frozen_checkpoint_hashes[track]:
            raise ValueError(f"evaluation loaded a different {track} checkpoint hash")
        if policy.get("source_identity_sha256") != frozen_source_hash:
            raise ValueError(f"evaluation {track} manifest has a different source identity")
        config_hash = policy.get("training_configuration_sha256")
        protocol_id = policy.get("training_protocol_id")
        if not isinstance(config_hash, str) or not config_hash:
            raise ValueError(f"evaluation manifest lacks {track} configuration hash")
        observed_configs[track] = config_hash
        if not isinstance(protocol_id, str) or not protocol_id:
            raise ValueError(f"evaluation manifest lacks {track} training protocol ID")
        observed_protocols[track] = protocol_id
        if expected_configs is not None and config_hash != expected_configs[track]:
            raise ValueError(f"{track} checkpoint configuration changed between conditions")
        if expected_protocols is not None and protocol_id != expected_protocols[track]:
            raise ValueError(f"{track} checkpoint protocol changed between conditions")
    if any(value != observed_protocols[MECO_TRACKS[0]]
           for value in observed_protocols.values()):
        raise ValueError("five evaluation checkpoints do not share one protocol")
    shared_config = report.get("shared_training_configuration")
    if not isinstance(shared_config, dict):
        raise ValueError("evaluation report lacks its shared training configuration")
    if expected_shared_config is not None and shared_config != expected_shared_config:
        raise ValueError("shared training configuration changed between conditions")
    return observed_configs, observed_protocols, shared_config


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def default_conditions(scenario_id: str, *, categories: tuple[str, ...] | None = None,
                       inference_disable_tick: int | None = None,
                       qualification_case: dict | None = None
                       ) -> tuple[MatrixCondition, ...]:
    """Return the plan's architecture/prediction/K/fault cases and eligible Phase 4 factors."""
    horizon = math.ceil(get_scenario(scenario_id).time_budget_s / .02)
    fault_tick = inference_disable_tick
    if fault_tick is None:
        # Default to a reproducible mid-episode macro-boundary request.
        fault_tick = (horizon // 2 // 12) * 12
    if fault_tick <= 0 or fault_tick >= horizon:
        raise ValueError("fallback fault tick must be inside the scenario horizon")
    conditions = (
        MatrixCondition("architecture_matched_cv_k16", "cv", "cv", 16),
        MatrixCondition("controls_default_d1_current_d2_cv_k16", "cv", None, 16),
        MatrixCondition("matched_prediction_current_k16", "current", "current", 16),
        MatrixCondition("candidate_recall_k8_cv", "cv", "cv", 8),
        MatrixCondition("candidate_recall_k32_cv", "cv", "cv", 32),
        MatrixCondition("inference_disabled_fallback_current_k16", "current",
                        "current", 16, fault_tick),
    )
    conditions = tuple(MatrixCondition(**{**asdict(condition),
        "qualification_case": qualification_case}) for condition in conditions)
    pressure_domain = scenario_id in ("offense_vs_scripted_pressure", "full_match_offense")
    supported_categories = tuple(category for category in (categories or ())
        if _registered_pressure_level(category) in (1, 2, 3, 4, 5))
    if pressure_domain and supported_categories:
        conditions += tuple(MatrixCondition(
            f"phase4_{name}", "cv", "cv", 16,
            phase4_factor=phase4_factor_config(name),
            categories=supported_categories,
            qualification_case=qualification_case)
            for name in PHASE4_FACTORS)
    for condition in conditions:
        condition.validate()
    return conditions


def _registered_pressure_level(category: str) -> int:
    marker = next((part for part in category.split("/")
                   if part.startswith("level_")), "level_0")
    try:
        return int(marker.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _records_by_key(report: dict) -> dict[tuple[str, str, int], dict]:
    result = {}
    for row in report.get("records", ()):
        key = (row.get("policy"), row.get("category"), row.get("seed"))
        if key in result:
            raise ValueError(f"duplicate policy/category/seed record: {key}")
        result[key] = row
    return result


def _record_metrics(row: dict | None) -> dict | None:
    if row is None:
        return None
    metrics = row.get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _contrast_pair(*, name: str, condition_id: str, report: dict,
                   records: dict, candidate_policy: str,
                   reference_policy: str) -> dict:
    rows = []
    for category in report["categories"]:
        for seed in report["seeds"]:
            candidate = records.get((candidate_policy, category, seed))
            reference = records.get((reference_policy, category, seed))
            if candidate is None or reference is None:
                raise ValueError(f"contrast {name} lacks paired records for {category}/{seed}")
            rows.append({"category": category, "seed": seed,
                "candidate_policy": candidate_policy,
                "reference_policy": reference_policy,
                "candidate_metrics": _record_metrics(candidate),
                "reference_metrics": _record_metrics(reference),
                "candidate_record_policy_id": candidate.get("policy_id"),
                "reference_record_policy_id": reference.get("policy_id")})
    return {"contrast": name,
        "candidate_policy": candidate_policy,
        "reference_policy": reference_policy,
        "paired": True,
        "condition_id": condition_id,
        "report_provenance": {"evaluation_id": report["evaluation_id"],
            "protocol_id": report["protocol_id"],
            "source_identity_sha256": report["source_identity_sha256"],
            "seed_list": report["seeds"],
            "policy_manifest": report["policy_manifest"]},
        "paired_episode_rows": rows}


def _architecture_track_vs_d0(*, track: str, condition_id: str,
                              report: dict, records: dict,
                              categories: tuple[str, ...],
                              seeds: tuple[int, ...]) -> dict:
    """Build exact architecture-condition pairs; D0 retains its current control."""
    if track not in MECO_TRACKS:
        raise ValueError(f"unknown learned track for D0 contrast: {track}")
    if (report.get("categories") != list(categories) or
            report.get("seeds") != list(seeds)):
        raise ValueError("architecture D0 contrast report category/seed coverage differs")
    expected = {(policy, category, seed)
                for policy in (track, "D0")
                for category in categories for seed in seeds}
    observed = {key for key in records if key[0] in (track, "D0")}
    if observed != expected:
        raise ValueError(
            f"architecture D0 contrast coverage differs for {track}: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")
    policy_manifest = report.get("policy_manifest", {})
    candidate_manifest = policy_manifest.get(track, {})
    d0_manifest = policy_manifest.get("D0", {})
    candidate_manifest_id = candidate_manifest.get("policy_id")
    d0_manifest_id = d0_manifest.get("policy_id")
    if (not isinstance(candidate_manifest_id, str) or not candidate_manifest_id or
            not isinstance(d0_manifest_id, str) or not d0_manifest_id):
        raise ValueError("architecture D0 contrast report lacks candidate/D0 manifest policy IDs")
    rows = []
    for category in categories:
        for seed in seeds:
            candidate = records.get((track, category, seed))
            reference = records.get(("D0", category, seed))
            if candidate is None or reference is None:
                raise ValueError(
                    f"architecture condition {track}/D0 lacks paired records for {category}/{seed}")
            candidate_metrics = _record_metrics(candidate)
            reference_metrics = _record_metrics(reference)
            candidate_policy_id = candidate.get("policy_id")
            reference_policy_id = reference.get("policy_id")
            if (not isinstance(candidate_metrics, dict) or
                    not isinstance(reference_metrics, dict) or
                    not isinstance(candidate_policy_id, str) or not candidate_policy_id or
                    not isinstance(reference_policy_id, str) or not reference_policy_id):
                raise ValueError(
                    f"architecture D0 contrast lacks policy IDs/metrics for {category}/{seed}")
            if (candidate_policy_id != candidate_manifest_id or
                    reference_policy_id != d0_manifest_id):
                raise ValueError(
                    f"architecture D0 record policy ID differs from report manifest "
                    f"for {track}/{category}/{seed}")
            rows.append({"category": category, "seed": seed,
                "candidate_policy": track, "reference_policy": "D0",
                "candidate_record_policy_id": candidate_policy_id,
                "reference_record_policy_id": reference_policy_id,
                "candidate_metrics": candidate_metrics,
                "reference_metrics": reference_metrics})
    prediction = report.get("prediction", {})
    candidate_k = report.get("candidate_k", {})
    if (prediction.get("learned_mode") != "cv" or
            candidate_k.get("effective") != 16 or
            d0_manifest.get("policy_id") != "D0_production_deterministic" or
            d0_manifest.get("prediction_mode") == "cv"):
        raise ValueError(
            "architecture D0 contrast report does not prove CV/K=16 learned tracks "
            "and production deterministic D0 semantics")
    return {"contrast": f"{track}_vs_D0_architecture_condition",
        "comparison_scope": "architecture_condition_outcomes",
        "candidate_policy": track, "reference_policy": "D0",
        "paired": True, "condition_id": condition_id,
        "condition_name": "architecture_matched_cv_k16",
        "condition_semantics": {
            "learned_track_prediction_mode": prediction.get("learned_mode", "cv"),
            "d0_prediction_mode": d0_manifest.get("prediction_mode"),
            "d0_policy_id": d0_manifest.get("policy_id"),
            "fairness_caveat": (
                "Architecture-condition outcome contrast: learned tracks use CV; "
                "D0 is the production deterministic controller and is not assigned "
                "the learned-track CV prediction mode. This is not a "
                "prediction-mode-matched architecture comparison.")},
        "report_provenance": {"evaluation_id": report["evaluation_id"],
            "protocol_id": report["protocol_id"],
            "source_identity_sha256": report["source_identity_sha256"],
            "seed_list": list(report["seeds"]),
            "policy_manifest": report["policy_manifest"]},
        "paired_episode_rows": rows}


def run_matrix(*, checkpoints: dict[str, str], scenario_id: str,
               categories: tuple[str, ...], seeds: tuple[int, ...],
               device: str, output_dir: str | Path,
               inference_disable_tick: int | None = None,
               qualification_protocol: str | Path | None = None,
               qualification_case: dict | None = None,
               replay_output_dir: str | Path | None = None,
               replay_seeds: tuple[int, ...] | None = None) -> dict:
    """Execute all fixed comparisons on exact shared checkpoint/seed bundles."""
    from .curriculum_training import source_identity
    from .meco_track_training import evaluate_meco, load_evaluation_protocol

    if set(checkpoints) != set(MECO_TRACKS):
        raise ValueError(f"checkpoint keys must be exactly {MECO_TRACKS}")
    if not seeds or len(set(seeds)) != len(seeds) or any(
            type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds):
        raise ValueError("seeds must be a nonempty unique list of signed-63-bit integers")
    if (replay_output_dir is None) != (replay_seeds is None):
        raise ValueError("replay_output_dir and explicit replay_seeds must be supplied together")
    if replay_seeds is not None and (not replay_seeds or len(set(replay_seeds)) != len(replay_seeds)
            or any(type(seed) is not int or seed not in seeds for seed in replay_seeds)):
        raise ValueError("replay_seeds must be a nonempty unique subset of matrix seeds")
    if not categories:
        raise ValueError("at least one scenario category is required")
    definition = get_scenario(scenario_id)
    if any(category not in definition.required_categories for category in categories):
        raise ValueError("category does not belong to the selected scenario")
    from .fuel_physics_hip import prepare_native_fuel_extension
    prepare_native_fuel_extension(device)
    qualification_provenance = None
    if qualification_protocol is None:
        seed_split_assignments = None
        evaluation_design = "diagnostic_only_unbound_seeds"
        evaluation_split = "diagnostic"
    else:
        protocol, protocol_id, protocol_sha256, protocol_file = load_evaluation_protocol(
            Path(qualification_protocol), scenario_id=scenario_id)
        seed_split_assignments = _held_out_seed_assignments(
            protocol, scenario_id, categories, seeds)
        splits = {split for assignments in seed_split_assignments.values()
                  for split in assignments.values()}
        if len(splits) != 1:
            raise ValueError(
                "one matrix invocation must use a single frozen split; run validation "
                "and test seeds in separate matrices")
        evaluation_split = next(iter(splits))
        evaluation_design = "frozen_protocol_held_out"
        qualification_provenance = {"protocol_id": protocol_id,
            "protocol_sha256": protocol_sha256,
            "protocol_file": protocol_file}
    output_root = Path(output_dir)
    execution_device = normalized_device_info(device)
    bundle_hashes = _checkpoint_hashes(checkpoints)
    source_hash = _source_identity_hash(source_identity())
    config_hashes, protocol_ids, frozen_shared_config = (
        _checkpoint_configuration_provenance(checkpoints))
    qualification_case = dict(DEFAULT_EVALUATION_STRATUM if qualification_case is None
                              else qualification_case)
    if (set(qualification_case) != set(DEFAULT_EVALUATION_STRATUM) or
            any(not isinstance(value, str) or not value
                for value in qualification_case.values())):
        raise ValueError("qualification_case must contain nonempty layout, defender, start, and sensor identifiers")
    conditions = default_conditions(scenario_id, categories=categories,
        inference_disable_tick=inference_disable_tick,
        qualification_case=qualification_case)
    phase4_enabled = any(condition.phase4_factor is not None for condition in conditions)
    pressure_domain = scenario_id in ("offense_vs_scripted_pressure", "full_match_offense")
    unsupported_phase4 = ([{"category": category,
        "reason": ("scenario is outside the registered pressure domain" if not pressure_domain
                   else "level_0 has no scripted pressure slot" if _registered_pressure_level(category) == 0
                   else "category is outside the registered level_1 through level_5 pressure domain")}
        for category in categories
        if not pressure_domain or _registered_pressure_level(category) not in (1, 2, 3, 4, 5)])
    condition_specs = [asdict(condition) for condition in conditions]
    matrix_protocol = {"schema_version": MATRIX_SCHEMA_VERSION,
        "scenario_id": scenario_id, "categories": list(categories),
        "seed_list": list(seeds), "checkpoint_sha256": bundle_hashes,
        "evaluation_design": evaluation_design,
        "evaluation_split": evaluation_split,
        "qualification_protocol_id": (None if qualification_provenance is None else
            qualification_provenance["protocol_id"]),
        "qualification_protocol": qualification_provenance,
        "seed_split_assignments": seed_split_assignments,
        "source_identity_sha256": source_hash,
        "execution_device": execution_device,
        "qualification_case": qualification_case,
        "training_configuration_sha256": config_hashes,
        "training_protocol_ids": protocol_ids,
        "shared_training_configuration": frozen_shared_config,
        "conditions": condition_specs,
        "phase4_factor_schema": ("meco-phase4-five-level-pressure-v2" if phase4_enabled else None),
        "phase4_unsupported_categories": unsupported_phase4,
        "unsupported_conditions_open": ([] if phase4_enabled else [
            "explicit_feint_behavior", "sensor_delay_ablation",
            "separate_low_confidence_profile", "abrupt_opponent_policy_change"])}
    matrix_id = _canonical_hash(matrix_protocol)
    matrix_protocol["matrix_id"] = matrix_id
    results: dict[str, dict] = {}
    evaluation_reports: dict[str, dict] = {}
    expected_configs = config_hashes
    expected_protocols = protocol_ids
    expected_shared_config = frozen_shared_config
    for condition in conditions:
        if _checkpoint_hashes(checkpoints) != bundle_hashes:
            raise ValueError("checkpoint files drifted from frozen matrix inputs before a condition")
        if _source_identity_hash(source_identity()) != source_hash:
            raise ValueError("source identity drifted from frozen matrix inputs before a condition")
        spec = asdict(condition)
        condition_id = _canonical_hash({"matrix_id": matrix_id,
            "condition": spec, "checkpoint_sha256": bundle_hashes,
            "source_identity_sha256": source_hash, "seed_list": list(seeds),
            "execution_device": execution_device})
        report_path = output_root / f"{condition_id}.evaluation.json"
        condition_categories = condition.categories or categories
        report = evaluate_meco(checkpoints, scenario_id=scenario_id,
            categories=condition_categories, episodes=len(seeds), master_seed=seeds[0],
            episode_seeds=seeds, device=device, split=evaluation_split,
            qualification_case=condition.qualification_case,
            qualification_protocol_id=(None if qualification_provenance is None else
                qualification_provenance["protocol_id"]),
            evaluation_protocol=(qualification_protocol
                if evaluation_split in ("validation", "test") else None),
            evaluation_case=(condition.qualification_case or qualification_case
                if evaluation_split in ("validation", "test") else None),
            candidate_k_override=condition.candidate_k,
            prediction_mode=condition.prediction_mode,
            control_prediction_mode=condition.control_prediction_mode,
            inference_disable_tick=condition.inference_disable_tick,
            phase4_factor=condition.phase4_factor,
            replay_output_dir=replay_output_dir,
            replay_seeds=replay_seeds,
            condition_id=condition_id,
            output=str(report_path))
        if report.get("phase4_factor") != condition.phase4_factor:
            raise ValueError("evaluation report Phase 4 factor differs from the frozen condition")
        observed_configs, observed_protocols, shared_config = _validate_condition_manifest(
            report, frozen_checkpoint_hashes=bundle_hashes,
            frozen_source_hash=source_hash,
            expected_qualification_protocol_id=(None if qualification_provenance is None else
                qualification_provenance["protocol_id"]),
            expected_configs=expected_configs,
            expected_protocols=expected_protocols,
            expected_shared_config=expected_shared_config,
            expected_device=execution_device,
            expected_qualification_case=qualification_case)
        if _checkpoint_hashes(checkpoints) != bundle_hashes:
            raise ValueError("checkpoint files drifted while an evaluation condition was running")
        if _source_identity_hash(source_identity()) != source_hash:
            raise ValueError("source identity drifted while an evaluation condition was running")
        report["experiment_matrix"] = {"matrix_id": matrix_id,
            "condition_id": condition_id, "condition": spec,
            "evaluation_design": evaluation_design,
            "evaluation_split": evaluation_split,
            "qualification_protocol_id": (None if qualification_provenance is None else
                qualification_provenance["protocol_id"]),
            "qualification_protocol": qualification_provenance,
            "seed_split_assignments": seed_split_assignments}
        _atomic_json(report_path, report)
        condition_records = _records_by_key(report)
        results[condition.condition_name] = {"condition_id": condition_id,
            "evaluation_id": report["evaluation_id"],
            "dashboard_replay_count": len(report.get("dashboard_replays", [])),
            "protocol_id": report["protocol_id"],
            "qualification_protocol_id": report["qualification_protocol_id"],
            "evaluation_design": evaluation_design,
            "evaluation_split": evaluation_split,
            "qualification_protocol": qualification_provenance,
            "seed_split_assignments": seed_split_assignments,
            "report_path": str(report_path),
            "checkpoint_sha256": bundle_hashes,
            "source_identity_sha256": source_hash,
            "execution_device": execution_device,
            "qualification_case": report["qualification_case"],
            "policy_manifest": report["policy_manifest"],
            "effective_candidate_k": report["candidate_k"],
            "prediction": report["prediction"],
            "phase4_factor": condition.phase4_factor,
            "fault": report["inference_disable_fault"]}
        evaluation_reports[condition.condition_name] = {
            "report": report, "records": condition_records,
            "condition_id": condition_id}

    contrasts = []
    architecture_name = "architecture_matched_cv_k16"
    architecture = evaluation_reports[architecture_name]
    for contrast_name, pairs in POLICY_CONTRASTS.items():
        condition_name = ("controls_default_d1_current_d2_cv_k16"
                          if contrast_name == "D1_vs_D2"
                          else architecture_name)
        selected_report = evaluation_reports[condition_name]
        for candidate, reference in pairs:
            contrasts.append(_contrast_pair(name=contrast_name,
                condition_id=selected_report["condition_id"],
                report=selected_report["report"],
                records=selected_report["records"],
                candidate_policy=candidate, reference_policy=reference))
    # Architecture-condition outcomes: all five learned tracks against D0 on
    # exact common category/seed records. D0 keeps its production behavior.
    for track in MECO_TRACKS:
        contrasts.append(_architecture_track_vs_d0(
            track=track, condition_id=architecture["condition_id"],
            report=architecture["report"], records=architecture["records"],
            categories=categories, seeds=seeds))
    # Matched forecast mode is an episode-paired cross-condition contrast.
    current = evaluation_reports["matched_prediction_current_k16"]
    cv = architecture
    current_records, cv_records = current["records"], cv["records"]
    for track in MECO_TRACKS:
        for category in categories:
            for seed in seeds:
                current_row = current_records.get((track, category, seed))
                cv_row = cv_records.get((track, category, seed))
                if current_row is None or cv_row is None:
                    raise ValueError(f"matched prediction contrast lacks {track}/{category}/{seed}")
                contrasts.append({"contrast": "matched_prediction_current_vs_cv",
                    "candidate_policy": f"{track}@cv",
                    "reference_policy": f"{track}@current", "paired": True,
                    "condition_ids": {"current": current["condition_id"],
                                      "cv": cv["condition_id"]},
                    "report_provenance": {
                        "current": {"evaluation_id": current["report"]["evaluation_id"],
                                    "protocol_id": current["report"]["protocol_id"]},
                        "cv": {"evaluation_id": cv["report"]["evaluation_id"],
                               "protocol_id": cv["report"]["protocol_id"]}},
                    "category": category, "seed": seed,
                    "current_metrics": _record_metrics(current_row),
                    "cv_metrics": _record_metrics(cv_row)})
    # Same-checkpoint P=0 inference ablation; both rows share exact seeds.
    for track in ("A1", "A2", "A3"):
        contrasts.append(_contrast_pair(name=f"{track}_P0",
            condition_id=architecture["condition_id"],
            report=architecture["report"], records=architecture["records"],
            candidate_policy=f"{track}_P0", reference_policy=track))
    # Candidate K sweep: report row-level recall at each common K, without
    # treating the bounded expanded-goal set as a simulator-optimal oracle.
    for track in ("C1", "C2"):
        for name in ("candidate_recall_k8_cv", "architecture_matched_cv_k16",
                     "candidate_recall_k32_cv"):
            item = evaluation_reports[name]
            for category in categories:
                for seed in seeds:
                    row = item["records"].get((track, category, seed))
                    if row is None:
                        raise ValueError(f"candidate recall contrast lacks {track}/{category}/{seed}")
                    contrasts.append({"contrast": "candidate_recall_K_sweep",
                        "candidate_policy": f"{track}@K={item['report']['candidate_k']}",
                        "reference_policy": "expanded_route_qualified_candidate_pool",
                        "paired": True, "condition_id": item["condition_id"],
                        "report_provenance": {"evaluation_id": item["report"]["evaluation_id"],
                            "protocol_id": item["report"]["protocol_id"],
                            "source_identity_sha256": source_hash,
                            "checkpoint_sha256": bundle_hashes},
                        "category": category, "seed": seed,
                        "candidate_recall": row.get("candidate_recall")})
    # Fault report contains both normal and disabled-inference episodes.
    fault = evaluation_reports["inference_disabled_fallback_current_k16"]
    fault_report = fault["report"]
    fault_records = fault["records"]
    for track in MECO_TRACKS:
        candidate_policy = f"{track}__INFERENCE_DISABLED_TO_D1"
        contrasts.append(_contrast_pair(name="inference_disable_to_D1",
            condition_id=fault["condition_id"], report=fault_report,
            records=fault_records, candidate_policy=candidate_policy,
            reference_policy=track))

    # Each Phase 4 treatment is paired against the same-checkpoint, same-seed
    # no-factor scenario condition. Every row covers all five learned tracks
    # and D0/D1/D2 controls; the factor is the only condition-level difference.
    if phase4_enabled:
        baseline = evaluation_reports[architecture_name]
        factor_policies = (*MECO_TRACKS, "D0", "D1", "D2")
        for name in PHASE4_FACTORS:
            treatment_name = f"phase4_{name}"
            if treatment_name not in evaluation_reports:
                continue
            treatment = evaluation_reports[treatment_name]
            treatment_categories = treatment["report"]["categories"]
            for policy in factor_policies:
                for category in treatment_categories:
                    for seed in seeds:
                        treated = treatment["records"].get((policy, category, seed))
                        control = baseline["records"].get((policy, category, seed))
                        if treated is None or control is None:
                            raise ValueError(
                                f"Phase 4 paired contrast lacks {policy}/{category}/{seed}")
                        expected_stratum = qualification_case
                        treated_stratum = {key: treated.get(key)
                            for key in DEFAULT_EVALUATION_STRATUM}
                        control_stratum = {key: control.get(key)
                            for key in DEFAULT_EVALUATION_STRATUM}
                        if (treated_stratum != expected_stratum or
                                control_stratum != expected_stratum or
                                treated_stratum != control_stratum):
                            raise ValueError(
                                f"Phase 4 pair has mismatched evaluation strata for {policy}/{category}/{seed}")
                        contrasts.append({"contrast": "phase4_factor_vs_no_factor",
                            "factor": name,
                            "factor_sha256": phase4_factor_digest(
                                treatment["report"].get("phase4_factor")),
                            "candidate_policy": policy, "reference_policy": policy,
                            "paired": True,
                            "condition_ids": {"factor": treatment["condition_id"],
                                "baseline": baseline["condition_id"]},
                            "qualification_case": expected_stratum,
                            "category": category, "seed": seed,
                            "factor_metrics": _record_metrics(treated),
                            "baseline_metrics": _record_metrics(control),
                            "factor_runtime": treated.get("phase4_factor_runtime")})

    # Repeat the frozen bundle identity on every contrast so a contrast row
    # remains auditable when copied out of the matrix-level report.
    common_provenance = {
        "execution_device": execution_device,
        "qualification_case": qualification_case,
        "evaluation_design": evaluation_design,
        "evaluation_split": evaluation_split,
        "qualification_protocol_id": (None if qualification_provenance is None else
            qualification_provenance["protocol_id"]),
        "qualification_protocol": qualification_provenance,
        "seed_split_assignments": seed_split_assignments,
        "checkpoint_sha256": bundle_hashes,
        "training_configuration_sha256": {
            track: config_hashes[track] for track in MECO_TRACKS},
        "training_protocol_ids": {
            track: protocol_ids[track] for track in MECO_TRACKS},
        "source_identity_sha256": source_hash,
        "seed_list": list(seeds),
        "category_list": list(categories),
        "scenario_id": scenario_id,
    }
    for contrast in contrasts:
        contrast["common_bundle_provenance"] = common_provenance

    matrix_report = {"schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": matrix_id,
        "protocol": matrix_protocol,
        "evaluation_design": evaluation_design,
        "evaluation_split": evaluation_split,
        "qualification_protocol_id": (None if qualification_provenance is None else
            qualification_provenance["protocol_id"]),
        "qualification_protocol": qualification_provenance,
        "seed_split_assignments": seed_split_assignments,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity_sha256": source_hash,
        "execution_device": execution_device,
        "dashboard_replay_capture": ({"enabled": replay_output_dir is not None,
            "output_dir": str(Path(replay_output_dir).resolve()),
            "seed_list": list(replay_seeds or ())}
            if replay_output_dir is not None else {"enabled": False}),
        "checkpoint_sha256": bundle_hashes,
        "training_configuration_sha256": {
            track: results[architecture_name]["policy_manifest"][track][
                "training_configuration_sha256"] for track in MECO_TRACKS},
        "training_protocol_ids": {
            track: results[architecture_name]["policy_manifest"][track][
                "training_protocol_id"] for track in MECO_TRACKS},
        "conditions": results,
        "contrasts": contrasts,
        "phase4_unsupported_categories": unsupported_phase4,
        "interpretation_limits": [
            "Paired rows are reported without confidence intervals; use the calibrated qualification gate for promotion.",
            "Candidate recall is relative to the declared bounded route-qualified reference pool, not a simulator-optimality oracle.",
            "P=0 is an inference-time ablation using the same learned A checkpoint.",
            "A1/A2/A3/C1/C2 versus D0 rows are architecture-condition outcomes paired on exact category/seed; learned tracks use CV while D0 retains its production deterministic behavior, so these are not prediction-mode-matched contrasts.",
            "Fault rows hand off at the first 12-step macro boundary at or after the requested tick.",
            ("Each Phase 4 factor replaces only registered slot-0 pressure behavior and is paired with the no-factor row at the same category and seed; unsupported categories are listed explicitly. Metrics remain diagnostic until held-out evaluation and calibrated gates." if phase4_enabled else
             "Feint, sensor-delay, separate low-confidence, and abrupt policy-change conditions require registered level_1 through level_5 pressure categories; level_0 has no scripted pressure slot.")],
    }
    matrix_report["report_sha256"] = _canonical_hash(matrix_report)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / f"{matrix_id}.matrix.json", matrix_report)
    return matrix_report
