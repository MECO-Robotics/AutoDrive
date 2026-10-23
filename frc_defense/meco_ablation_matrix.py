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


@dataclass(frozen=True)
class MatrixCondition:
    condition_name: str
    prediction_mode: str
    control_prediction_mode: str | None
    candidate_k: int
    inference_disable_tick: int | None = None
    phase4_factor: dict | None = None

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
                                 expected_device: dict[str, str]):
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
                       inference_disable_tick: int | None = None
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
    phase4_supported = (scenario_id in ("offense_vs_scripted_pressure",
        "full_match_offense") and categories is not None and bool(categories)
        and all(category.endswith("level_4_reactive") for category in categories))
    if phase4_supported:
        conditions += tuple(MatrixCondition(
            f"phase4_{name}", "cv", "cv", 16,
            phase4_factor=phase4_factor_config(name))
            for name in PHASE4_FACTORS)
    for condition in conditions:
        condition.validate()
    return conditions


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


def run_matrix(*, checkpoints: dict[str, str], scenario_id: str,
               categories: tuple[str, ...], seeds: tuple[int, ...],
               device: str, output_dir: str | Path,
               inference_disable_tick: int | None = None,
               qualification_protocol: str | Path | None = None) -> dict:
    """Execute all fixed comparisons on exact shared checkpoint/seed bundles."""
    from .curriculum_training import source_identity
    from .meco_track_training import evaluate_meco

    if set(checkpoints) != set(MECO_TRACKS):
        raise ValueError(f"checkpoint keys must be exactly {MECO_TRACKS}")
    if not seeds or len(set(seeds)) != len(seeds) or any(
            type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds):
        raise ValueError("seeds must be a nonempty unique list of signed-63-bit integers")
    if not categories:
        raise ValueError("at least one scenario category is required")
    definition = get_scenario(scenario_id)
    if any(category not in definition.required_categories for category in categories):
        raise ValueError("category does not belong to the selected scenario")
    qualification_provenance = None
    if qualification_protocol is None:
        seed_split_assignments = None
        evaluation_design = "diagnostic_only_unbound_seeds"
        evaluation_split = "diagnostic"
    else:
        from .meco_supervised import _load_protocol
        protocol, protocol_id, protocol_sha256, protocol_path = _load_protocol(
            Path(qualification_protocol))
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
            "protocol_file": protocol_path.name}
    output_root = Path(output_dir)
    execution_device = normalized_device_info(device)
    bundle_hashes = _checkpoint_hashes(checkpoints)
    source_hash = _source_identity_hash(source_identity())
    config_hashes, protocol_ids, frozen_shared_config = (
        _checkpoint_configuration_provenance(checkpoints))
    conditions = default_conditions(scenario_id, categories=categories,
        inference_disable_tick=inference_disable_tick)
    phase4_enabled = any(condition.phase4_factor is not None for condition in conditions)
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
        "training_configuration_sha256": config_hashes,
        "training_protocol_ids": protocol_ids,
        "shared_training_configuration": frozen_shared_config,
        "conditions": condition_specs,
        "phase4_factor_schema": ("meco-phase4-four-factor-v1" if phase4_enabled else None),
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
        report = evaluate_meco(checkpoints, scenario_id=scenario_id,
            categories=categories, episodes=len(seeds), master_seed=seeds[0],
            episode_seeds=seeds, device=device, split=evaluation_split,
            qualification_protocol_id=(None if qualification_provenance is None else
                qualification_provenance["protocol_id"]),
            candidate_k_override=condition.candidate_k,
            prediction_mode=condition.prediction_mode,
            control_prediction_mode=condition.control_prediction_mode,
            inference_disable_tick=condition.inference_disable_tick,
            phase4_factor=condition.phase4_factor,
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
            expected_device=execution_device)
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
            treatment = evaluation_reports[treatment_name]
            for policy in factor_policies:
                for category in categories:
                    for seed in seeds:
                        treated = treatment["records"].get((policy, category, seed))
                        control = baseline["records"].get((policy, category, seed))
                        if treated is None or control is None:
                            raise ValueError(
                                f"Phase 4 paired contrast lacks {policy}/{category}/{seed}")
                        contrasts.append({"contrast": "phase4_factor_vs_no_factor",
                            "factor": name,
                            "factor_sha256": phase4_factor_digest(
                                treatment["report"].get("phase4_factor")),
                            "candidate_policy": policy, "reference_policy": policy,
                            "paired": True,
                            "condition_ids": {"factor": treatment["condition_id"],
                                "baseline": baseline["condition_id"]},
                            "category": category, "seed": seed,
                            "factor_metrics": _record_metrics(treated),
                            "baseline_metrics": _record_metrics(control),
                            "factor_runtime": treated.get("phase4_factor_runtime")})

    # Repeat the frozen bundle identity on every contrast so a contrast row
    # remains auditable when copied out of the matrix-level report.
    common_provenance = {
        "execution_device": execution_device,
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
        "checkpoint_sha256": bundle_hashes,
        "training_configuration_sha256": {
            track: results[architecture_name]["policy_manifest"][track][
                "training_configuration_sha256"] for track in MECO_TRACKS},
        "training_protocol_ids": {
            track: results[architecture_name]["policy_manifest"][track][
                "training_protocol_id"] for track in MECO_TRACKS},
        "conditions": results,
        "contrasts": contrasts,
        "interpretation_limits": [
            "Paired rows are reported without confidence intervals; use the calibrated qualification gate for promotion.",
            "Candidate recall is relative to the declared bounded route-qualified reference pool, not a simulator-optimality oracle.",
            "P=0 is an inference-time ablation using the same learned A checkpoint.",
            "Fault rows hand off at the first 12-step macro boundary at or after the requested tick.",
            ("All four Phase 4 factors use explicit same-seed baseline pairs; metrics remain diagnostic until held-out evaluation and calibrated gates." if phase4_enabled else
             "Feint, sensor-delay, separate low-confidence, and abrupt policy-change conditions require the registered level_4_reactive pressure categories.")],
    }
    matrix_report["report_sha256"] = _canonical_hash(matrix_report)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / f"{matrix_id}.matrix.json", matrix_report)
    return matrix_report
