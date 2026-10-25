"""Resumable validation-only execution and aggregation for MECO campaigns."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable

from .meco_ablation_matrix import _records_by_key
from .meco_tracks import MECO_TRACKS
from .training_curriculum import SCENARIOS
from .meco_qualification import (CALIBRATED_PROTOCOL_VERSION,
    D0_PAIRED_REFERENCE_PROTOCOL_VERSION, LEGACY_CALIBRATED_PROTOCOL_VERSION,
    PROTOCOL_VERSION)
from .meco_validation_campaign import (CALIBRATED_GATE_STATUSES, _artifact_bytes,
    _canonical_bytes, _sha, compile_validation_campaign)
from .meco_device_lease import lease_devices


CAMPAIGN_SCHEMA = "meco-validation-campaign-complete-v1"
JOB_INPUT_SCHEMA = "meco-validation-campaign-job-input-v1"
JOB_COMPLETE_SCHEMA = "meco-validation-campaign-job-complete-v1"


def run_phase4_opponent_cell(plan: dict, cell_key: str, *,
        protocol_path: str | Path, checkpoint_paths: dict[str, str],
        matrix_script: str | Path, output_dir: str | Path, device: str,
        runner: Callable[[list[str]], int] | None = None) -> dict:
    """Dispatch one protocol-bound reactive-opponent diagnostic cell.

    The normal campaign runner groups categories. This narrow path keeps the
    Phase 4 category/policy/seed assignment atomic and passes it through the
    same matrix CLI into ``evaluate_meco``. It makes no qualification claim.
    """
    if (not isinstance(plan, dict) or plan.get("schema") !=
            "meco-phase4-validation-campaign-plan-v1"):
        raise ValueError("invalid Phase 4 validation campaign plan")
    if plan.get("plan_id") != _sha({key: value for key, value in plan.items()
                                    if key != "plan_id"}):
        raise ValueError("Phase 4 campaign plan digest mismatch")
    requirement = next((row for row in plan.get("requirements", [])
        if row.get("requirement_id") in ("reactive_defenders_and_opponent_pools",
                                          "direct_pursuit", "trench_interception") and any(
            cell.get("case_cell_key") == cell_key for cell in row.get("cells", []))), None)
    if requirement is None:
        raise ValueError("Phase 4 plan lacks the requested opponent requirement")
    matches = [cell for cell in requirement.get("cells", [])
               if cell.get("case_cell_key") == cell_key]
    if len(matches) != 1:
        raise ValueError("Phase 4 reactive cell key is missing or duplicated")
    cell = matches[0]
    direct_pursuit = requirement.get("requirement_id") == "direct_pursuit"
    trench_interception = requirement.get("requirement_id") == "trench_interception"
    assignment = (cell.get("direct_pursuit_assignment") if direct_pursuit else
                 cell.get("trench_interception_assignment") if trench_interception else
                 cell.get("opponent_schedule_assignment"))
    if (not isinstance(assignment, dict) or
            cell.get("scheduler_assignment_supported") is not True or
            assignment.get("qualification_eligible") is not False):
        raise ValueError("Phase 4 reactive cell lacks a diagnostic schedule assignment")
    if (not isinstance(checkpoint_paths, dict) or
            set(checkpoint_paths) != set(MECO_TRACKS)):
        raise ValueError("Phase 4 matrix dispatch requires all five checkpoints")

    from .meco_phase4_design import CASE_AXES, _read_protocol_validation
    from .meco_phase4_opponents import (compile_opponent_policy_schedule,
        opponent_policy_runtime_domain_supported,
        trench_interception_runtime_domain_supported)
    protocol, raw = _read_protocol_validation(protocol_path)
    if hashlib.sha256(raw).hexdigest() != plan.get("protocol_binding", {}).get(
            "protocol_raw_file_sha256"):
        raise ValueError("Phase 4 dispatch protocol differs from the frozen plan")
    case_axes = cell.get("case_axes")
    if not isinstance(case_axes, dict) or set(case_axes) != set(CASE_AXES):
        raise ValueError("Phase 4 dispatch cell has an invalid protocol case")
    domain_supported = (
        trench_interception_runtime_domain_supported(
            case_axes["scenario_id"], case_axes["category"])
        if trench_interception else
        opponent_policy_runtime_domain_supported(
            case_axes["scenario_id"], case_axes["category"]))
    if not domain_supported:
        raise ValueError("Phase 4 opponent dispatch is outside pressure-category runtime domain")
    protocol_rows = [row for row in protocol["cases"]
        if all(row.get(axis) == case_axes[axis] for axis in CASE_AXES)]
    if len(protocol_rows) != 1:
        raise ValueError("Phase 4 dispatch case is not unique in validation protocol")
    seeds = protocol_rows[0].get("seeds")
    if (_sha(seeds) != cell.get(
            "validation_seed_assignment_sha256") or len(seeds) != cell.get("seed_count")):
        raise ValueError("Phase 4 dispatch seeds differ from the protocol-bound cell")
    qualification_case = {axis: case_axes[axis] for axis in (
        "layout_id", "defender_family", "start_condition", "sensor_profile")}
    schedule = compile_opponent_policy_schedule(
        policy_id=("direct_pursuit" if direct_pursuit else
            "trench_interceptor" if trench_interception else assignment["policy_id"]),
        scenario_id=case_axes["scenario_id"],
        categories=(case_axes["category"],), qualification_case=qualification_case,
        seeds=seeds, split="validation", factor_id=cell["factor_id"],
        protocol_id=protocol["protocol_id"])
    if (schedule["policy_config_sha256"] != assignment.get("policy_config_sha256")
            or schedule["factor_id"] != cell.get("factor_id")
            or (not direct_pursuit and schedule["policy_family"] != assignment.get("policy_family"))
            or schedule["per_episode_assignment_sha256"] !=
                assignment.get("factor_treatment_seeded_assignment_sha256")
            or schedule["configured_pressure_slots"].get(case_axes["category"])
                != assignment.get("configured_pressure_slots")
            or (direct_pursuit and (
                schedule.get("registered_pressure_slots", {}).get(case_axes["category"])
                    != assignment.get("registered_pressure_slots")
                or schedule.get("preserved_scripted_slots", {}).get(case_axes["category"])
                    != assignment.get("preserved_scripted_slots")
                or assignment.get("assigned_pursuit_slot") not in
                    assignment.get("configured_pressure_slots", ())
                or assignment.get("configured_pressure_slots") !=
                    [assignment.get("assigned_pursuit_slot")]
                or assignment.get("control_policy_ids") != sorted(
                    (*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2"))))
            or (trench_interception and (
                schedule["configured_pressure_slots"].get(case_axes["category"]) != [0]
                or assignment.get("configured_pressure_slots") != [0]
                or assignment.get("control_policy_ids") != sorted(
                    (*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2"))))):
        raise ValueError("runtime schedule differs from the planner's exact policy/config/seed assignment")
    case_dir = Path(output_dir).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    seed_manifest = {"schema": "meco-exact-seed-manifest-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "scenario_id": case_axes["scenario_id"],
        "categories": [case_axes["category"]], "split": "validation",
        "qualification_case": qualification_case, "seed_list": seeds}
    seed_path = case_dir / "seed-manifest.json"
    schedule_path = case_dir / "opponent-schedule.json"
    direct_binding = {**schedule, "schedule": schedule,
        "protocol_sha256": hashlib.sha256(raw).hexdigest()}
    direct_binding_path = case_dir / "direct-pursuit-binding.json"
    trench_binding = {"schema": "meco-phase4-trench-interception-binding-v1",
        "factor_id": "opponent.trench_interception", "policy_id": "trench_interceptor",
        "scenario_id": case_axes["scenario_id"], "category": case_axes["category"],
        "qualification_case": qualification_case, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(raw).hexdigest(), "split": "validation",
        "seed_list": seeds, "seed_manifest_sha256": hashlib.sha256(
            _artifact_bytes(seed_manifest)).hexdigest(),
        "schedule": schedule, "schedule_sha256": schedule["schedule_sha256"],
        "policy_config_sha256": schedule["policy_config_sha256"],
        "per_episode_assignment_sha256": schedule["per_episode_assignment_sha256"]}
    trench_binding_path = case_dir / "trench-interception-binding.json"
    case_path = case_dir / "qualification-case.json"
    _write_immutable(seed_path, seed_manifest)
    _write_immutable(schedule_path, schedule)
    if direct_pursuit:
        _write_immutable(direct_binding_path, direct_binding)
    if trench_interception:
        _write_immutable(trench_binding_path, trench_binding)
    _write_immutable(case_path, qualification_case)
    base_argv = [sys.executable, str(Path(matrix_script).resolve()),
        "--scenario", case_axes["scenario_id"], "--categories", case_axes["category"],
        "--seed-manifest", str(seed_path), "--qualification-protocol",
        str(Path(protocol_path).resolve()), "--qualification-case-json", str(case_path),
        "--device", device, "--conditions", "architecture_matched_cv_k16"]
    for track in MECO_TRACKS:
        base_argv += [f"--{track.lower()}-checkpoint",
                      str(Path(checkpoint_paths[track]).resolve())]
    execute = runner or (lambda command: subprocess.run(command, check=False).returncode)
    pair_results = {}
    for condition in ("no_factor", "factor_treatment"):
        argv = [*base_argv, "--output-dir", str(case_dir / condition)]
        if direct_pursuit:
            argv += ["--phase4-direct-pursuit-binding-json", str(direct_binding_path)]
        if trench_interception:
            argv += ["--phase4-trench-interception-binding-json", str(trench_binding_path)]
        if condition == "factor_treatment":
            argv += ["--phase4-opponent-schedule-json", str(schedule_path)]
        pair_results[condition] = execute(argv)
    pairing = None
    if all(result == 0 for result in pair_results.values()):
        pairing = (_validate_phase4_trench_interception_pair(case_dir, trench_binding,
            policy_ids=assignment["control_policy_ids"], category=case_axes["category"],
            seeds=seeds, qualification_case=qualification_case)
            if trench_interception else _validate_phase4_opponent_pair(case_dir, schedule,
                categories=[case_axes["category"]], seeds=seeds,
                qualification_case=qualification_case,
                direct_pursuit=direct_pursuit))
    return {"case_cell_key": cell_key,
        "policy_id": assignment["policy_id"],
        "schedule_sha256": schedule["schedule_sha256"],
        "runtime_status": ("outcomes_captured_pending_replay_review" if
            (direct_pursuit or trench_interception) and pairing and
                pairing.get("outcome_signals_complete") else
            "paired_outcomes_rejected" if trench_interception and pairing else
            "scheduled_not_simulator_validated"),
        "simulator_behavior_validated": False,
        "outcome_signals_complete": bool((direct_pursuit or trench_interception) and pairing and
            pairing.get("outcome_signals_complete")),
        "qualification_eligible": False,
        "required_family_outcomes": ({"interception_rate": None,
            "policy_family_coverage": pairing.get("policy_family_coverage"),
            "paired_outcome_by_family": pairing.get("paired_outcome_by_family")}
            if (not direct_pursuit and not trench_interception and pairing) else
            {"interception_rate": None, "policy_family_coverage": None,
             "paired_outcome_by_family": None}
            if not direct_pursuit and not trench_interception else None),
        "paired_coverage": pairing,
        "promotion_gate": ({"simulator_outcomes_captured": bool(pairing and
            pairing.get("outcome_signals_complete")),
            "held_out_replay_review_accepted": False,
            "dashboard_backend_current": False,
            "dashboard_frontend_current": False,
            "executable_promotion_allowed": False} if direct_pursuit or trench_interception else None),
        "dispatch_return_codes": pair_results,
        "executed": all(result == 0 for result in pair_results.values())}


def run_phase4_start_profile_cell(plan: dict, cell_key: str, *,
        protocol_path: str | Path, checkpoint_paths: dict[str, str],
        matrix_script: str | Path, output_dir: str | Path, device: str,
        runner: Callable[[list[str]], int] | None = None) -> dict:
    """Run matched ordinary-reset/profile arms for one protocol-bound pose set.

    The matrix itself covers every selected policy on identical validation
    seeds. This diagnostic pairing compares the ordinary scenario reset with
    the one frozen full-team profile and never claims qualification.
    """
    if (not isinstance(plan, dict) or plan.get("schema") !=
            "meco-phase4-validation-campaign-plan-v1" or
            plan.get("plan_id") != _sha({key: value for key, value in plan.items()
                                         if key != "plan_id"})):
        raise ValueError("invalid Phase 4 validation campaign plan")
    requirement = next((row for row in plan.get("requirements", [])
        if row.get("requirement_id") == "multiple_legal_team_start_states"), None)
    if requirement is None:
        raise ValueError("Phase 4 plan lacks the legal full-team start requirement")
    matches = [row for row in requirement.get("cells", [])
               if row.get("case_cell_key") == cell_key]
    if len(matches) != 1:
        raise ValueError("Phase 4 start-profile cell key is missing or duplicated")
    cell = matches[0]
    assignment = cell.get("start_profile_assignment")
    if (cell.get("factor_id") != "start.legal_team_poses" or
            cell.get("scheduler_assignment_supported") is not True or
            not isinstance(assignment, dict) or
            assignment.get("qualification_eligible") is not False):
        raise ValueError("Phase 4 cell lacks a diagnostic legal-start assignment")
    if (not isinstance(checkpoint_paths, dict) or
            set(checkpoint_paths) != set(MECO_TRACKS)):
        raise ValueError("Phase 4 start-profile dispatch requires all five checkpoints")

    dispatch_binding = _capture_phase4_dispatch_binding(checkpoint_paths, device)

    from .meco_phase4_design import CASE_AXES, _read_protocol_validation
    from .meco_phase4_context import (phase4_start_profile_assignment,
        validate_phase4_start_profile_assignment)
    protocol, raw = _read_protocol_validation(protocol_path)
    protocol_sha256 = hashlib.sha256(raw).hexdigest()
    if protocol_sha256 != plan.get("protocol_binding", {}).get(
            "protocol_raw_file_sha256"):
        raise ValueError("Phase 4 dispatch protocol differs from the frozen plan")
    case_axes = cell.get("case_axes")
    if not isinstance(case_axes, dict) or set(case_axes) != set(CASE_AXES):
        raise ValueError("Phase 4 start-profile cell has an invalid protocol case")
    protocol_rows = [row for row in protocol.get("cases", [])
        if all(row.get(axis) == case_axes[axis] for axis in CASE_AXES)]
    if len(protocol_rows) != 1:
        raise ValueError("Phase 4 start-profile case is not unique in validation protocol")
    seeds = protocol_rows[0].get("seeds")
    if (_sha(seeds) != cell.get("validation_seed_assignment_sha256") or
            len(seeds) != cell.get("seed_count")):
        raise ValueError("Phase 4 start-profile seeds differ from the protocol-bound cell")
    qualification_case = {axis: case_axes[axis] for axis in (
        "layout_id", "defender_family", "start_condition", "sensor_profile")}
    expected_assignment = phase4_start_profile_assignment(
        cell.get("design_axes", {}).get("legal_team_pose_sets"),
        scenario_id=case_axes["scenario_id"], split="validation",
        protocol_id=protocol["protocol_id"], protocol_sha256=protocol_sha256,
        qualification_case=qualification_case,
        categories=(case_axes["category"],), seeds=seeds)
    validate_phase4_start_profile_assignment(assignment)
    if assignment != expected_assignment:
        raise ValueError("start-profile cell assignment differs from exact protocol/case/seeds")

    case_dir = Path(output_dir).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    seed_manifest = {"schema": "meco-exact-seed-manifest-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "scenario_id": case_axes["scenario_id"],
        "categories": [case_axes["category"]], "split": "validation",
        "qualification_case": qualification_case, "seed_list": seeds}
    seed_path = case_dir / "seed-manifest.json"
    case_path = case_dir / "qualification-case.json"
    _write_immutable(seed_path, seed_manifest)
    _write_immutable(case_path, qualification_case)
    base_argv = [sys.executable, str(Path(matrix_script).resolve()),
        "--scenario", case_axes["scenario_id"], "--categories", case_axes["category"],
        "--seed-manifest", str(seed_path), "--qualification-protocol",
        str(Path(protocol_path).resolve()), "--qualification-case-json", str(case_path),
        "--device", device, "--conditions", "architecture_matched_cv_k16"]
    for track in MECO_TRACKS:
        base_argv += [f"--{track.lower()}-checkpoint",
                      str(Path(checkpoint_paths[track]).resolve())]
    execute = runner or (lambda command: subprocess.run(command, check=False).returncode)
    return_codes = {}
    for arm in ("no_factor", "factor_treatment"):
        argv = [*base_argv, "--output-dir", str(case_dir / arm)]
        if arm == "factor_treatment":
            argv += ["--phase4-start-profile", assignment["registry_profile_id"]]
        return_codes[arm] = execute(argv)
    pairing = None
    paired_outcomes_artifact = None
    if all(code == 0 for code in return_codes.values()):
        pairing = _validate_phase4_start_profile_pair(case_dir, assignment,
            categories=[case_axes["category"]], seeds=seeds,
            expected_binding=dispatch_binding)
        current_source_hash = _source_identity_sha256()
        if current_source_hash != dispatch_binding["source_identity_sha256"]:
            raise ValueError("Phase 4 start-profile source identity drifted during paired dispatch")
        if (_source_tree_identity_sha256() !=
                dispatch_binding["source_tree_identity_sha256"]):
            raise ValueError("Phase 4 start-profile source tree changed during paired dispatch")
        if _dispatch_checkpoint_hashes(checkpoint_paths) != dispatch_binding[
                "checkpoint_sha256"]:
            raise ValueError("Phase 4 start-profile checkpoint files drifted during paired dispatch")
        outcome_payload = {
            "schema": "meco-phase4-start-profile-paired-outcomes-v1",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "registry_sha256": plan["registry_binding"][
                "registry_raw_file_sha256"],
            "scenario_id": case_axes["scenario_id"],
            "category": case_axes["category"],
            "qualification_case": qualification_case,
            "split": "validation",
            "validation_seed_assignment_sha256": _sha(seeds),
            "seed_manifest_sha256": _file_sha(seed_path),
            "registry_profile_id": assignment["registry_profile_id"],
            "profile_config_sha256": assignment["profile_config_sha256"],
            "assignment_sha256": assignment["assignment_sha256"],
            "source_identity_sha256": dispatch_binding[
                "source_identity_sha256"],
            "source_tree_identity": dispatch_binding["source_tree_identity"],
            "source_tree_identity_sha256": dispatch_binding[
                "source_tree_identity_sha256"],
            "checkpoint_sha256": dispatch_binding["checkpoint_sha256"],
            "execution_device": dispatch_binding["execution_device"],
            "condition_report_sha256": pairing["condition_report_sha256"],
            "condition_report_paths": pairing["condition_report_paths"],
            "condition_matrix_sha256": pairing["condition_matrix_sha256"],
            "condition_matrix_paths": pairing["condition_matrix_paths"],
            "condition_ids": pairing["condition_ids"],
            "paired_keys": pairing["paired_keys"],
            "per_world_legality_records": pairing[
                "per_world_legality_records"],
            "state_ids": ["scenario_defined", assignment["registry_profile_id"]],
            "outcome_by_start_state": pairing["outcome_by_start_state"],
            "exact_pair_coverage": pairing["exact_pair_coverage"],
            "paired_deltas_reconciled": pairing["paired_deltas_reconciled"],
            "coverage_digest_sha256": pairing["coverage_digest_sha256"],
            "legal_world_count": pairing["legal_world_count"],
            "legal_world_legal_count": pairing["legal_world_legal_count"],
            "legal_start_validation_rate": pairing[
                "legal_start_validation_rate"],
            "all_treatment_worlds_legal": pairing[
                "all_treatment_worlds_legal"],
            "qualification_eligible": False,
        }
        outcome_path = case_dir / "start-profile-paired-outcomes.json"
        _write_immutable(outcome_path, outcome_payload)
        paired_outcomes_artifact = {"path": str(outcome_path),
            "sha256": _file_sha(outcome_path)}
    return {"case_cell_key": cell_key,
        "registry_profile_id": assignment["registry_profile_id"],
        "profile_config_sha256": assignment["profile_config_sha256"],
        "assignment_sha256": assignment["assignment_sha256"],
        "dispatch_binding": dispatch_binding,
        "runtime_status": ("paired_outcomes_captured_pending_replay_review"
            if pairing is not None and pairing[
                "all_treatment_worlds_legal"] else
            "paired_outcomes_rejected_world_legality" if pairing is not None else
            "scheduled_not_simulator_validated"),
        "simulator_behavior_validated": False,
        "legal_start_validation_rate": (None if pairing is None else
            pairing["legal_start_validation_rate"]),
        "outcome_by_start_state": (None if pairing is None else
            pairing["outcome_by_start_state"]),
        "coverage_digest_sha256": (None if pairing is None else
            pairing["coverage_digest_sha256"]),
        "qualification_eligible": False,
        "paired_coverage": pairing,
        "paired_outcomes_artifact": paired_outcomes_artifact,
        "dispatch_return_codes": return_codes,
        "executed": all(code == 0 for code in return_codes.values())}


def run_phase4_context_cell(plan: dict, cell_key: str, *, factor: str,
        protocol_path: str | Path, checkpoint_paths: dict[str, str],
        matrix_script: str | Path, output_dir: str | Path, device: str,
        runner: Callable[[list[str]], int] | None = None) -> dict:
    """Dispatch one exact paired focal-pose or teammate-pool validation cell.

    The matrix API creates both arms in one invocation, sharing its frozen
    protocol case, seed list, checkpoints, source identity, and device.
    Returned coverage is metadata evidence only; behavioral outcomes remain
    unavailable and qualification remains disabled.
    """
    if (not isinstance(plan, dict) or plan.get("schema") !=
            "meco-phase4-validation-campaign-plan-v1" or
            plan.get("plan_id") != _sha({key: value for key, value in plan.items()
                                          if key != "plan_id"})):
        raise ValueError("invalid Phase 4 validation campaign plan")
    requirement_id, design_axis = {
        "focal_pose": ("observation_only_focal_pose_mislocalization",
                       "focal_pose_offsets"),
        "teammate_behavior": ("teammate_behaviors", "teammate_behaviors"),
    }.get(factor, (None, None))
    if requirement_id is None:
        raise ValueError("unsupported paired Phase 4 context factor")
    requirements = [row for row in plan.get("requirements", [])
                    if row.get("requirement_id") == requirement_id]
    if len(requirements) != 1:
        raise ValueError("Phase 4 plan lacks a unique context requirement")
    matches = [cell for cell in requirements[0].get("cells", [])
               if cell.get("case_cell_key") == cell_key]
    if len(matches) != 1:
        raise ValueError("Phase 4 context cell key is missing or duplicated")
    cell = matches[0]
    if (not isinstance(checkpoint_paths, dict) or
            set(checkpoint_paths) != set(MECO_TRACKS)):
        raise ValueError("Phase 4 context dispatch requires all five checkpoints")

    from .meco_phase4_design import CASE_AXES, _read_protocol_validation
    from .meco_phase4_context import (TEAMMATE_BEHAVIORS,
        phase4_focal_pose_assignment, phase4_teammate_behavior_assignment,
        validate_phase4_focal_pose_assignment,
        validate_phase4_teammate_behavior_assignment)
    protocol, raw = _read_protocol_validation(protocol_path)
    protocol_sha256 = hashlib.sha256(raw).hexdigest()
    if protocol_sha256 != plan.get("protocol_binding", {}).get(
            "protocol_raw_file_sha256"):
        raise ValueError("Phase 4 context dispatch protocol differs from the frozen plan")
    case_axes = cell.get("case_axes")
    if (not isinstance(case_axes, dict) or set(case_axes) != set(CASE_AXES) or
            case_axes.get("scenario_id") != "full_match_offense"):
        raise ValueError("Phase 4 context cell is not an exact full-match protocol case")
    protocol_rows = [row for row in protocol.get("cases", [])
        if all(row.get(axis) == case_axes[axis] for axis in CASE_AXES)]
    if len(protocol_rows) != 1:
        raise ValueError("Phase 4 context case is not unique in the frozen protocol")
    seeds = protocol_rows[0].get("seeds")
    if (_sha(seeds) != cell.get("validation_seed_assignment_sha256") or
            len(seeds) != cell.get("seed_count")):
        raise ValueError("Phase 4 context seeds differ from the protocol-bound cell")
    categories = [case_axes["category"]]
    qualification_case = {axis: case_axes[axis] for axis in (
        "layout_id", "defender_family", "start_condition", "sensor_profile")}
    design_axes = cell.get("design_axes")
    value = design_axes.get(design_axis) if isinstance(design_axes, dict) else None
    if factor == "focal_pose":
        from .meco_phase4_context import phase4_context_identity
        phase4_context_identity("focal_pose_offsets", value)
        common = dict(paired_offset_name=value, scenario_id=case_axes["scenario_id"],
            split="validation", protocol_id=protocol["protocol_id"],
            protocol_sha256=protocol_sha256, qualification_case=qualification_case,
            categories=categories, seeds=seeds)
        assignments = {"no_factor": phase4_focal_pose_assignment(None, **common),
            "factor_treatment": phase4_focal_pose_assignment(value, **common)}
        for assignment in assignments.values():
            validate_phase4_focal_pose_assignment(assignment)
        condition_names = [f"phase4_focal_pose_{value}_no_offset",
                           f"phase4_focal_pose_{value}_offset"]
        extra_args = ["--phase4-focal-pose-offset", value]
        outcome_fields = ["relocalization_time_s", "pose_error_recovery_rate",
                          "safety_contact_rate"]
    else:
        if value not in TEAMMATE_BEHAVIORS:
            raise ValueError("teammate-behavior cell has an unknown behavior axis value")
        # The named design-axis value fixes the deterministic pool rotation.
        pool_seed = TEAMMATE_BEHAVIORS.index(value)
        common = dict(scenario_id=case_axes["scenario_id"], split="validation",
            protocol_id=protocol["protocol_id"], protocol_sha256=protocol_sha256,
            qualification_case=qualification_case, categories=categories,
            seeds=seeds, pool_seed=pool_seed)
        assignments = {"no_factor": phase4_teammate_behavior_assignment(
                arm="no_factor", **common),
            "factor_treatment": phase4_teammate_behavior_assignment(
                arm="behavior_pool", **common)}
        for assignment in assignments.values():
            validate_phase4_teammate_behavior_assignment(assignment)
        condition_names = ["phase4_teammate_behaviors_no_factor",
                           "phase4_teammate_behaviors_behavior_pool"]
        extra_args = ["--phase4-teammate-behavior-pool-seed", str(pool_seed)]
        outcome_fields = ["outcome_by_teammate_behavior", "teammate_collision_rate",
                          "score_contribution"]

    case_dir = Path(output_dir).resolve()
    if case_dir.exists():
        raise ValueError("Phase 4 context output path already exists; dispatch is not resumable")
    case_dir.mkdir(parents=True, exist_ok=False)
    dispatch_binding = _capture_phase4_dispatch_binding(checkpoint_paths, device)
    seed_manifest = {"schema": "meco-exact-seed-manifest-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "scenario_id": case_axes["scenario_id"], "categories": categories,
        "split": "validation", "qualification_case": qualification_case,
        "seed_list": seeds}
    seed_path, case_path = case_dir / "seed-manifest.json", case_dir / "qualification-case.json"
    _write_immutable(seed_path, seed_manifest)
    _write_immutable(case_path, qualification_case)
    argv = [sys.executable, str(Path(matrix_script).resolve()),
        "--scenario", case_axes["scenario_id"], "--categories", categories[0],
        "--seed-manifest", str(seed_path), "--qualification-protocol",
        str(Path(protocol_path).resolve()), "--qualification-case-json", str(case_path),
        "--device", device, "--conditions", ",".join(condition_names),
        "--output-dir", str(case_dir), *extra_args]
    for track in MECO_TRACKS:
        argv += [f"--{track.lower()}-checkpoint",
                 str(Path(checkpoint_paths[track]).resolve())]
    execute = runner or (lambda command: subprocess.run(command, check=False).returncode)
    return_code = execute(argv)
    pairing = None
    if return_code == 0:
        pairing = _validate_phase4_context_matrix_pair(case_dir, factor=factor,
            assignments=assignments, categories=categories, seeds=seeds,
            condition_names=condition_names, expected_binding=dispatch_binding,
            outcome_fields=outcome_fields, protocol_path=Path(protocol_path).resolve(),
            protocol_id=protocol["protocol_id"], protocol_sha256=protocol_sha256,
            seed_manifest_path=seed_path)
        if (_source_identity_sha256() != dispatch_binding["source_identity_sha256"] or
                _dispatch_checkpoint_hashes(checkpoint_paths) !=
                    dispatch_binding["checkpoint_sha256"]):
            raise ValueError("Phase 4 context source/checkpoint binding drifted during dispatch")
    pair_sha = (assignments["no_factor"].get("pair_assignment_sha256")
                if factor == "focal_pose" else assignments["no_factor"].get("pair_sha256"))
    return {"case_cell_key": cell_key, "factor": factor, "design_axis_value": value,
        "pair_assignment_sha256": pair_sha, "assignment_sha256": {
            arm: assignment.get("assignment_sha256", assignment.get("pair_sha256"))
            for arm, assignment in assignments.items()},
        "dispatch_binding": dispatch_binding,
        "runtime_status": "scheduled_not_simulator_validated",
        "simulator_behavior_validated": False,
        "outcomes": (pairing.get("outcomes") if pairing is not None else {
            field: None for field in outcome_fields}), "qualification_eligible": False,
        "paired_coverage": pairing, "dispatch_return_code": return_code,
        "executed": return_code == 0}


def _validate_phase4_context_matrix_pair(case_dir: Path, *, factor: str,
        assignments: dict, categories: list[str], seeds: list[int],
        condition_names: list[str], expected_binding: dict,
        outcome_fields: list[str], protocol_path: Path, protocol_id: str,
        protocol_sha256: str, seed_manifest_path: Path) -> dict:
    """Bind both matrix arms to exact assignments and row keys; no outcomes inferred."""
    policies = set((*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2"))
    expected_keys = {(policy, category, seed) for policy in policies
                     for category in categories for seed in seeds}
    paths = sorted(case_dir.glob("*.matrix.json"))
    if len(paths) != 1:
        raise ValueError("Phase 4 context dispatch requires exactly one matrix report")
    matrix = _read_json(paths[0])
    protocol = matrix.get("protocol", {})
    if (matrix.get("qualification_eligible") is not False or
            protocol.get("qualification_eligible") is not False or
            protocol.get("categories") != categories or
            protocol.get("seed_list") != seeds or
            protocol.get("evaluation_split") != "validation" or
            protocol.get("qualification_protocol_id") != protocol_id or
            protocol.get("qualification_protocol") != {
                "protocol_id": protocol_id, "protocol_sha256": protocol_sha256,
                "protocol_file": str(protocol_path)} or
            protocol.get("seed_manifest") != {
                "path": str(seed_manifest_path),
                "sha256": _file_sha(seed_manifest_path),
                "schema": "meco-exact-seed-manifest-v1"} or
            protocol.get("qualification_case") != assignments["no_factor"]["qualification_case"]):
        raise ValueError("Phase 4 context matrix differs from the frozen validation pair")
    for name, report in (("matrix", matrix), ("matrix protocol", protocol)):
        _validate_phase4_dispatch_report_binding(report, expected_binding,
            context=f"paired context {name}")
    conditions = matrix.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(condition_names):
        raise ValueError("Phase 4 context matrix has missing or unexpected paired conditions")
    rows_by_arm = {}
    report_hashes = {}
    case_root = case_dir.resolve()
    for arm, condition_name in zip(("no_factor", "factor_treatment"), condition_names):
        condition = conditions[condition_name]
        expected_assignment = assignments[arm]
        assignment_field = ("phase4_focal_pose_assignment" if factor == "focal_pose"
                            else "phase4_teammate_behavior_assignment")
        expected_assignment_json = json.loads(_canonical_bytes(expected_assignment))
        if condition.get(assignment_field) != expected_assignment_json:
            raise ValueError(f"Phase 4 {arm} condition has a mismatched assignment")
        _validate_phase4_dispatch_report_binding(condition, expected_binding,
            context=f"{arm} matrix condition")
        report_path = Path(condition["report_path"])
        if not report_path.is_absolute():
            report_path = case_root / report_path
        try:
            resolved_report_path = report_path.resolve(strict=True)
            resolved_report_path.relative_to(case_root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Phase 4 {arm} report path is missing or outside its fresh case directory") from exc
        if not resolved_report_path.is_file():
            raise ValueError(f"Phase 4 {arm} report path is not a regular file")
        report_bytes = resolved_report_path.read_bytes()
        report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        if condition.get("evaluation_report_sha256") != report_sha256:
            raise ValueError(f"Phase 4 {arm} report byte hash differs from matrix condition binding")
        report_hashes[condition_name] = report_sha256
        report = json.loads(report_bytes)
        if not isinstance(report, dict):
            raise ValueError(f"Phase 4 {arm} evaluation report is not a JSON object")
        _validate_phase4_dispatch_report_binding(report, expected_binding,
            context=f"{arm} evaluation report")
        rows = _records_by_key(report)
        if (set(rows) != expected_keys or report.get("qualification_eligible") is not False or
                report.get(assignment_field) != expected_assignment_json):
            raise ValueError(f"Phase 4 {arm} evaluation lacks exact policy/category/seed coverage")
        for key, row in rows.items():
            policy = key[0]
            track = policy[:-3] if policy.endswith("_P0") else policy
            if (row.get("checkpoint_sha256") != expected_binding["checkpoint_sha256"].get(track) or
                    row.get("source_identity_sha256") != expected_binding["source_identity_sha256"] or
                    row.get("execution_device") != expected_binding["execution_device"]):
                raise ValueError(f"Phase 4 context row has mismatched source/checkpoint/device: {key}")
            if factor == "focal_pose":
                captured = row.get("phase4_focal_pose")
                if arm == "no_factor":
                    if captured is None:
                        raise ValueError(f"Phase 4 control row lacks no-offset assignment evidence: {key}")
                if (not isinstance(captured, dict) or
                        captured.get("assignment_sha256") != expected_assignment["assignment_sha256"] or
                        captured.get("pair_assignment_sha256") != expected_assignment["pair_assignment_sha256"] or
                        captured.get("arm") != expected_assignment["arm"] or
                        captured.get("requested_offset_name") != expected_assignment["requested_offset_name"] or
                        captured.get("applied_before_policy_observation") is not True or
                        captured.get("physical_truth_unchanged") is not True or
                        captured.get("qualification_eligible") is not False):
                    raise ValueError(f"Phase 4 context row lacks exact focal-pose evidence: {key}")
                for field in outcome_fields:
                    if captured.get(field) is not None:
                        raise ValueError(f"Phase 4 focal-pose outcome {field} must remain unavailable")
            else:
                captured = row.get("phase4_teammate_behavior")
                seed = key[2]
                schedule = expected_assignment["behavior_schedule"]
                expected_behaviors = (None if arm == "no_factor" else
                    {str(slot): behavior for slot, behavior in
                     schedule[expected_assignment["seeds"].index(seed)]["assignments"].items()})
                if (not isinstance(captured, dict) or
                        captured.get("assignment_pair_sha256") != expected_assignment["pair_sha256"] or
                        captured.get("arm") != expected_assignment["arm"] or
                        captured.get("assignments") != expected_behaviors or
                        captured.get("assignment_applied") != (expected_behaviors is not None) or
                        captured.get("scenario_id") != row.get("scenario_id") or
                        captured.get("category") != key[1] or
                        captured.get("seed") != key[2] or
                        captured.get("policy_id") != row.get("policy_id") or
                        captured.get("condition_id") != row.get("condition_id") or
                        captured.get("protocol_id") != protocol_id or
                        captured.get("protocol_sha256") != protocol_sha256 or
                        captured.get("qualification_case") != expected_assignment[
                            "qualification_case"] or
                        captured.get("action_event_coverage", {}).get("complete") is not True or
                        captured.get("whole_team_episode_outcome_metrics") != {
                            name: row.get("metrics", {}).get(name) for name in (
                                "success", "scores", "useful_throughput",
                                "completed_cycles")} or
                        captured.get("source_identity_sha256") != row.get(
                            "source_identity_sha256") or
                        captured.get("checkpoint_sha256") != row.get(
                            "checkpoint_sha256") or
                        captured.get("execution_device") != row.get(
                            "execution_device") or
                        captured.get("teammate_collision_rate") is not None or
                        captured.get("score_contribution") is not None or
                        captured.get("qualification_eligible") is not False):
                    raise ValueError(f"Phase 4 context row lacks exact teammate-behavior evidence: {key}")
                if arm == "factor_treatment" and (
                        captured["action_event_coverage"].get("observed_action_ticks") !=
                        captured["action_event_coverage"].get("expected_action_ticks") or
                        captured["action_event_coverage"].get("expected_action_ticks") !=
                        row.get("elapsed_ticks") or
                        captured.get("outcome_semantics") !=
                        "whole-team paired episode outcome while this behavior assignment was active; not an individual causal contribution"):
                    raise ValueError(f"Phase 4 teammate treatment action coverage is incomplete: {key}")
        rows_by_arm[arm] = rows
    if set(rows_by_arm["no_factor"]) != set(rows_by_arm["factor_treatment"]):
        raise ValueError("Phase 4 context arms have missing or unexpected paired rows")
    paired_behavior_outcomes = None
    if factor == "teammate_behavior":
        behavior_rows = []
        expected_behavior_keys = set()
        expected_behaviors = set()
        for key in sorted(expected_keys):
            treatment = rows_by_arm["factor_treatment"][key][
                "phase4_teammate_behavior"]
            control_metrics = rows_by_arm["no_factor"][key].get("metrics", {})
            for slot_text, behavior in treatment["assignments"].items():
                expected_behaviors.add(behavior)
                expected_behavior_keys.add((*key, behavior))
                behavior_rows.append({
                    "policy_id": key[0], "category": key[1], "seed": key[2],
                    "behavior": behavior,
                    "slot": int(slot_text),
                    "control_whole_team_outcome": {name: control_metrics.get(name)
                        for name in ("success", "scores", "useful_throughput",
                                     "completed_cycles")},
                    "treatment_whole_team_outcome": treatment[
                        "whole_team_episode_outcome_metrics"],
                    "outcome_semantics": "paired whole-team episode outcomes while this behavior was present; concurrent behavior slots prevent individual score attribution"})
        from .meco_phase4_outcome_bridge import reduce_teammate_behavior_outcomes
        paired_behavior_outcomes = reduce_teammate_behavior_outcomes(
            behavior_rows, expected_keys=expected_behavior_keys,
            expected_behaviors=expected_behaviors)
    else:
        paired_behavior_outcomes = {field: None for field in outcome_fields}
    return {"schema": "meco-phase4-context-pair-coverage-v1", "factor": factor,
        "condition_names": condition_names, "categories": categories,
        "seed_count": len(seeds), "paired_row_count": len(expected_keys),
        "paired_rows_per_condition": {name: len(expected_keys) for name in condition_names},
        "assignment_sha256": {arm: assignment.get("assignment_sha256",
            assignment.get("pair_sha256")) for arm, assignment in assignments.items()},
        "evaluation_report_sha256": report_hashes,
        "matrix_sha256": _file_sha(paths[0]), "checkpoint_sha256":
            expected_binding["checkpoint_sha256"], "source_identity_sha256":
            expected_binding["source_identity_sha256"], "execution_device":
            expected_binding["execution_device"], "outcomes":
            paired_behavior_outcomes, "qualification_eligible": False}


def _validate_phase4_start_profile_pair(case_dir: Path, assignment: dict, *,
        categories: list[str], seeds: list[int], expected_binding: dict) -> dict:
    """Validate paired coverage and aggregate realized legal-start outcomes."""
    from .meco_tracks import MECO_TRACKS
    policies = set((*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2"))
    expected_keys = {(policy, category, seed) for policy in policies
                     for category in categories for seed in seeds}
    arms = {}
    condition_report_sha256 = {}
    condition_report_paths = {}
    condition_matrix_sha256 = {}
    condition_matrix_paths = {}
    condition_ids = {}
    for arm in ("no_factor", "factor_treatment"):
        paths = sorted((case_dir / arm).glob("*.matrix.json"))
        if len(paths) != 1:
            raise ValueError(f"Phase 4 {arm} start-profile arm lacks exactly one matrix report")
        matrix_bytes = paths[0].read_bytes()
        condition_matrix_sha256[arm] = hashlib.sha256(matrix_bytes).hexdigest()
        evidence_matrix_path = case_dir / "paired-evidence-reports" / f"{arm}.matrix.json"
        _write_immutable_bytes(evidence_matrix_path, matrix_bytes)
        condition_matrix_paths[arm] = evidence_matrix_path.relative_to(
            case_dir.resolve()).as_posix()
        matrix = json.loads(matrix_bytes)
        protocol = matrix.get("protocol", {})
        if (matrix.get("qualification_eligible") is not False or
                protocol.get("categories") != categories or
                protocol.get("seed_list") != seeds or
                protocol.get("qualification_case") != assignment["qualification_case"]):
            raise ValueError(f"Phase 4 {arm} start-profile matrix differs from frozen paired case")
        _validate_phase4_dispatch_report_binding(matrix, expected_binding,
            context=f"{arm} matrix")
        _validate_phase4_dispatch_report_binding(protocol, expected_binding,
            context=f"{arm} matrix protocol")
        expected_assignment = assignment if arm == "factor_treatment" else None
        if (protocol.get("phase4_start_profile_assignment") != expected_assignment or
                matrix.get("phase4_start_profile_assignment") != expected_assignment):
            raise ValueError(f"Phase 4 {arm} start-profile matrix has a mismatched assignment")
        conditions = matrix.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != {
                "architecture_matched_cv_k16"}:
            raise ValueError(f"Phase 4 {arm} matrix has unexpected condition coverage")
        condition = conditions["architecture_matched_cv_k16"]
        condition_ids[arm] = condition.get("condition_id")
        report_path = Path(condition["report_path"]).resolve()
        report_bytes = report_path.read_bytes()
        condition_report_sha256[arm] = hashlib.sha256(report_bytes).hexdigest()
        evidence_report_path = case_dir / "paired-evidence-reports" / f"{arm}.json"
        _write_immutable_bytes(evidence_report_path, report_bytes)
        condition_report_paths[arm] = evidence_report_path.relative_to(
            case_dir.resolve()).as_posix()
        report = json.loads(report_bytes)
        _validate_phase4_dispatch_report_binding(condition, expected_binding,
            context=f"{arm} matrix condition")
        _validate_phase4_dispatch_report_binding(report, expected_binding,
            context=f"{arm} per-policy evaluation report")
        rows = _records_by_key(report)
        if (set(rows) != expected_keys or report.get("qualification_eligible") is not False or
                report.get("phase4_start_profile_assignment") != expected_assignment):
            raise ValueError(f"Phase 4 {arm} evaluation lacks exact profile/seed/policy coverage")
        for key, row in rows.items():
            policy = key[0]
            track = policy[:-3] if policy.endswith("_P0") else policy
            expected_checkpoint = expected_binding["checkpoint_sha256"].get(track)
            if (row.get("checkpoint_sha256") != expected_checkpoint or
                    row.get("source_identity_sha256") !=
                        expected_binding["source_identity_sha256"] or
                    row.get("execution_device") != expected_binding["execution_device"]):
                raise ValueError(f"Phase 4 {arm} row has mismatched checkpoint/source/device binding: {key}")
            captured = row.get("phase4_start_profile")
            if arm == "no_factor":
                if captured is not None:
                    raise ValueError("no-factor start-profile control unexpectedly applied a profile")
            elif (not isinstance(captured, dict) or
                    captured.get("factor_id") != assignment["factor_id"] or
                    captured.get("registry_profile_id") != assignment["registry_profile_id"] or
                    captured.get("runtime_profile_name") != assignment["runtime_profile_name"] or
                    captured.get("profile_config_sha256") != assignment["profile_config_sha256"] or
                    captured.get("assignment_sha256") != assignment["assignment_sha256"] or
                    captured.get("profile_applied_before_observation") is not True or
                    captured.get("qualification_eligible") is not False):
                raise ValueError(f"Phase 4 treatment row lacks exact profile provenance: {key}")
            else:
                legal_worlds = captured.get("per_world_legal")
                world_count = captured.get("validated_pose_world_count")
                world_indices = captured.get("validated_pose_world_indices")
                legal_count = captured.get("validated_pose_legal_world_count")
                rate = captured.get("legal_start_validation_rate")
                if (type(world_count) is not int or world_count < 1 or
                        not isinstance(legal_worlds, list) or
                        len(legal_worlds) != world_count or
                        world_indices != list(range(world_count)) or
                        captured.get("validated_pose_world_index_semantics") !=
                            "zero_based_simulator_batch_index" or
                        any(type(value) is not bool for value in legal_worlds) or
                        type(legal_count) is not int or
                        legal_count != sum(legal_worlds) or
                        type(rate) not in (int, float) or
                        not math.isfinite(float(rate)) or
                        float(rate) != sum(legal_worlds) / world_count):
                    raise ValueError(f"Phase 4 treatment row has missing or partial per-world legality: {key}")
                outcome = captured.get("outcome_metrics")
                metrics = row.get("metrics")
                names = ("success", "scores", "useful_throughput", "completed_cycles")
                if (not isinstance(outcome, dict) or not isinstance(metrics, dict) or
                        set(outcome) != set(names)):
                    raise ValueError(f"Phase 4 treatment row has missing outcome metrics: {key}")
                for name in names:
                    value = outcome[name]
                    if (type(value) not in (bool, int, float) or
                            (type(value) is not bool and not math.isfinite(float(value))) or
                            metrics.get(name) != value):
                        raise ValueError(f"Phase 4 treatment row has invalid outcome metric {name}: {key}")
        manifest = report.get("policy_manifest")
        if not isinstance(manifest, dict):
            raise ValueError(f"Phase 4 {arm} evaluation lacks per-policy provenance")
        for policy in policies:
            track = policy[:-3] if policy.endswith("_P0") else policy
            expected_checkpoint = expected_binding["checkpoint_sha256"].get(track)
            provenance = manifest.get(policy)
            if (not isinstance(provenance, dict) or
                    provenance.get("checkpoint_sha256") != expected_checkpoint or
                    provenance.get("source_identity_sha256") !=
                        expected_binding["source_identity_sha256"] or
                    provenance.get("execution_device") != expected_binding["execution_device"]):
                raise ValueError(f"Phase 4 {arm} policy manifest has mismatched binding: {policy}")
        arms[arm] = rows
    if set(arms["no_factor"]) != set(arms["factor_treatment"]):
        raise ValueError("Phase 4 start-profile arms do not have exact paired rows")
    metric_names = ("success", "scores", "useful_throughput", "completed_cycles")
    by_state = {}
    legal_world_total = 0
    legal_world_count_total = 0
    for state, arm in (("scenario_defined", "no_factor"),
                       (assignment["registry_profile_id"], "factor_treatment")):
        by_policy = {}
        for policy in sorted(policies):
            records = [arms[arm][(policy, category, seed)]
                for category in categories for seed in seeds]
            values = {name: [] for name in metric_names}
            for row in records:
                metrics = row.get("metrics")
                if not isinstance(metrics, dict):
                    raise ValueError(f"Phase 4 {arm} row has missing outcome metrics")
                for name in metric_names:
                    value = metrics.get(name)
                    if (type(value) not in (bool, int, float) or
                            (type(value) is not bool and not math.isfinite(float(value)))):
                        raise ValueError(f"Phase 4 {arm} row has null/nonfinite {name}")
                    values[name].append(float(value))
                if arm == "factor_treatment":
                    captured = row["phase4_start_profile"]
                    worlds = captured["per_world_legal"]
                    count = captured["validated_pose_world_count"]
                    legal_world_total += sum(worlds)
                    legal_world_count_total += count
            by_policy[policy] = {"n": len(records), "metrics": {
                "success_rate": sum(values["success"]) / len(records),
                **{name: sum(values[name]) / len(records)
                   for name in metric_names if name != "success"}}}
        by_state[state] = {"n": len(expected_keys), "by_policy": by_policy}
    paired_deltas = {}
    for policy in sorted(policies):
        by_metric = {}
        for name in metric_names:
            deltas = []
            for category in categories:
                for seed in seeds:
                    control = arms["no_factor"][(policy, category, seed)]["metrics"][name]
                    treatment = arms["factor_treatment"][(policy, category, seed)]["metrics"][name]
                    deltas.append({"category": category, "seed": seed,
                        "delta": float(treatment) - float(control)})
            by_metric[name] = {"mean_delta": sum(row["delta"] for row in deltas) / len(deltas),
                               "paired_n": len(deltas), "per_category_seed": deltas}
        paired_deltas[policy] = by_metric
    outcome_by_start_state = {"states": by_state,
        "paired_deltas_vs_scenario_defined": paired_deltas}
    legality_records = []
    for key in sorted(expected_keys):
        captured = arms["factor_treatment"][key]["phase4_start_profile"]
        legality_records.append({"policy": key[0], "category": key[1],
            "seed": key[2], "world_count": captured["validated_pose_world_count"],
            "world_index_semantics": captured[
                "validated_pose_world_index_semantics"],
            "world_indices": captured["validated_pose_world_indices"],
            "per_world_legal": captured["per_world_legal"],
            "legal_world_count": captured[
                "validated_pose_legal_world_count"],
            "legal_start_validation_rate": captured["legal_start_validation_rate"]})
    coverage_digest = _sha({"expected_keys": [list(key) for key in sorted(expected_keys)],
        "assignment_sha256": assignment["assignment_sha256"],
        "checkpoint_sha256": expected_binding["checkpoint_sha256"],
        "source_identity_sha256": expected_binding["source_identity_sha256"],
        "execution_device": expected_binding["execution_device"],
        "condition_report_sha256": condition_report_sha256,
        "condition_matrix_sha256": condition_matrix_sha256,
        "per_world_legality_records": legality_records,
        "outcome_by_start_state": outcome_by_start_state})
    per_world_legality_records = legality_records
    return {"policies": sorted(policies), "categories": categories,
        "condition_names": ["architecture_matched_cv_k16"],
        "seed_count": len(seeds), "paired_row_count": len(expected_keys),
        "paired_rows_per_condition": {
            "architecture_matched_cv_k16": len(expected_keys)},
        "profile_config_sha256": assignment["profile_config_sha256"],
        "assignment_sha256": assignment["assignment_sha256"],
        "checkpoint_sha256": expected_binding["checkpoint_sha256"],
        "source_identity_sha256": expected_binding["source_identity_sha256"],
        "execution_device": expected_binding["execution_device"],
        "legal_start_validation_rate": legal_world_total / legal_world_count_total,
        "legal_world_count": legal_world_count_total,
        "legal_world_legal_count": legal_world_total,
        "condition_report_sha256": condition_report_sha256,
        "condition_report_paths": condition_report_paths,
        "condition_matrix_sha256": condition_matrix_sha256,
        "condition_matrix_paths": condition_matrix_paths,
        "condition_ids": condition_ids,
        "paired_keys": [list(key) for key in sorted(expected_keys)],
        "per_world_legality_records": per_world_legality_records,
        "exact_pair_coverage": True,
        "paired_deltas_reconciled": True,
        "all_treatment_worlds_legal": legal_world_total == legal_world_count_total,
        "outcome_by_start_state": outcome_by_start_state,
        "coverage_digest_sha256": coverage_digest, "qualification_eligible": False}


def _dispatch_checkpoint_hashes(checkpoint_paths: dict[str, str]) -> dict[str, str]:
    return {track: _file_sha(Path(checkpoint_paths[track]).resolve())
            for track in sorted(checkpoint_paths)}


def _source_identity_sha256() -> str:
    from .curriculum_training import source_identity
    return hashlib.sha256(json.dumps(source_identity(), sort_keys=True,
        allow_nan=False).encode()).hexdigest()


def _source_tree_identity() -> dict:
    from .curriculum_training import source_tree_identity
    return source_tree_identity()


def _source_tree_identity_sha256(identity: dict | None = None) -> str:
    identity = _source_tree_identity() if identity is None else identity
    return hashlib.sha256(json.dumps(identity, sort_keys=True,
        allow_nan=False).encode()).hexdigest()


def _capture_phase4_dispatch_binding(checkpoint_paths: dict[str, str], device: str) -> dict:
    # Match matrix capture order: native runtime preparation precedes source identity.
    from .fuel_physics_hip import prepare_native_fuel_extension
    from .meco_device import normalized_device_info
    prepare_native_fuel_extension(device)
    source_tree = _source_tree_identity()
    return {"checkpoint_sha256": _dispatch_checkpoint_hashes(checkpoint_paths),
        "source_identity_sha256": _source_identity_sha256(),
        "source_tree_identity": source_tree,
        "source_tree_identity_sha256": _source_tree_identity_sha256(source_tree),
        "execution_device": normalized_device_info(device)}


def _validate_phase4_dispatch_report_binding(report: dict, expected: dict, *,
        context: str) -> None:
    if (not isinstance(report, dict) or
            report.get("checkpoint_sha256") != expected["checkpoint_sha256"] or
            report.get("source_identity_sha256") != expected["source_identity_sha256"] or
            report.get("execution_device") != expected["execution_device"]):
        raise ValueError(f"Phase 4 {context} checkpoint/source/device binding mismatch")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_phase4_opponent_pair(case_dir: Path, schedule: dict,
        *, categories: list[str], seeds: list[int], qualification_case: dict,
        direct_pursuit: bool = False) -> dict:
    """Require exact paired controls/category/seeds across both matrix arms."""
    from .meco_tracks import MECO_TRACKS
    policy_names = set((*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2"))
    matrix_files = {}
    reports_by_arm = {}
    paired_matrix_binding = None
    expected_keys = {(policy, category, seed) for policy in policy_names
                     for category in categories for seed in seeds}
    for arm in ("no_factor", "factor_treatment"):
        paths = sorted((case_dir / arm).glob("*.matrix.json"))
        if len(paths) != 1:
            raise ValueError(f"Phase 4 {arm} arm lacks exactly one matrix report")
        matrix = _read_json(paths[0])
        protocol = matrix.get("protocol", {})
        expected_schedule = schedule if arm == "factor_treatment" else None
        direct_binding = protocol.get("phase4_direct_pursuit_binding")
        if (protocol.get("qualification_case") != qualification_case
                or protocol.get("categories") != categories
                or protocol.get("seed_list") != seeds
                or protocol.get("condition_selection") != ["architecture_matched_cv_k16"]
                or [condition.get("condition_name") for condition in protocol.get("conditions", [])]
                    != ["architecture_matched_cv_k16"]
                or protocol.get("phase4_opponent_schedule") != expected_schedule
                or (direct_pursuit and (not isinstance(direct_binding, dict)
                    or direct_binding.get("schedule") != schedule
                    or direct_binding.get("schedule_sha256") != schedule["schedule_sha256"]
                    or direct_binding.get("checkpoint_bundle_sha256") != protocol.get(
                        "checkpoint_sha256")
                    or direct_binding.get("source_identity_sha256") != protocol.get(
                        "source_identity_sha256")
                    or direct_binding.get("execution_device") != protocol.get(
                        "execution_device")
                    or not isinstance(direct_binding.get("pair_binding_id"), str)
                    or not direct_binding.get("pair_binding_id")
                    or direct_binding.get("pair_binding") !=
                        protocol.get("phase4_direct_pursuit_binding", {}).get(
                            "pair_binding")))):
            raise ValueError(f"Phase 4 {arm} matrix differs from its frozen paired assignment")
        binding = {key: protocol.get(key) for key in (
            "scenario_id", "qualification_protocol_id", "qualification_protocol",
            "checkpoint_sha256", "source_identity_sha256",
            "training_configuration_sha256", "execution_device",
            "qualification_case", "categories", "seed_list")}
        if paired_matrix_binding is None:
            paired_matrix_binding = binding
        elif binding != paired_matrix_binding:
            raise ValueError("Phase 4 pair arms differ in protocol/checkpoint/source/device binding")
        condition_map = matrix.get("conditions")
        if (not isinstance(condition_map, dict)
                or set(condition_map) != {"architecture_matched_cv_k16"}):
            raise ValueError(f"Phase 4 {arm} matrix has no condition reports")
        arm_reports = {}
        for condition_name, entry in condition_map.items():
            report_path = Path(entry["report_path"]).resolve(strict=True)
            try:
                report_path.relative_to(case_dir.resolve())
            except ValueError as exc:
                raise ValueError("Phase 4 opponent report escapes the fresh case directory") from exc
            report_sha = _file_sha(report_path)
            if entry.get("evaluation_report_sha256") != report_sha:
                raise ValueError("Phase 4 opponent report bytes differ from the matrix digest")
            report = _read_json(report_path)
            if (report.get("scenario_id") != protocol.get("scenario_id")
                    or report.get("categories") != categories
                    or report.get("split") != "validation"
                    or report.get("evaluation_protocol_id") !=
                        protocol.get("qualification_protocol_id")
                    or report.get("evaluation_protocol_sha256") !=
                        protocol.get("qualification_protocol", {}).get("protocol_sha256")
                    or report.get("exact_episode_seeds") != seeds
                    or report.get("execution_device") != protocol.get("execution_device")
                    or report.get("source_identity_sha256") !=
                        protocol.get("source_identity_sha256")
                    or report.get("checkpoint_sha256") != protocol.get(
                        "checkpoint_sha256")):
                raise ValueError(f"Phase 4 {arm} evaluator report differs from frozen matrix provenance")
            rows = _records_by_key(report)
            if set(rows) != expected_keys:
                raise ValueError(f"Phase 4 {arm}/{condition_name} has incomplete policy/category/seed coverage")
            if report.get("qualification_eligible") is not False:
                raise ValueError("Phase 4 opponent outputs cannot be qualification eligible")
            for key, row in rows.items():
                track = key[0][:-3] if key[0].endswith("_P0") else key[0]
                if (row.get("source_identity_sha256") != protocol.get(
                        "source_identity_sha256") or
                        row.get("checkpoint_sha256") != protocol.get(
                            "checkpoint_sha256", {}).get(track) or
                        row.get("execution_device") != protocol.get("execution_device")):
                    raise ValueError(f"Phase 4 opponent row has mismatched source/checkpoint/device: {key}")
                captured = row.get("phase4_opponent_policy")
                if direct_pursuit:
                    outcome = row.get("phase4_direct_pursuit")
                    if (not isinstance(outcome, dict)
                            or outcome.get("scenario_id") != protocol.get("scenario_id")
                            or outcome.get("category") != key[1]
                            or outcome.get("qualification_case") != qualification_case
                            or outcome.get("split") != "validation"
                            or outcome.get("seed") != key[2]
                            or outcome.get("factor_id") != schedule["factor_id"]
                            or outcome.get("policy_id") != schedule["policy_id"]
                            or outcome.get("schedule_sha256") != schedule["schedule_sha256"]
                            or outcome.get("pair_binding_id") != direct_binding.get(
                                "pair_binding_id")
                            or outcome.get("policy_config_sha256") != schedule["policy_config_sha256"]
                            or outcome.get("configured_slots") != schedule[
                                "configured_pressure_slots"][key[1]]
                            or outcome.get("checkpoint_bundle_sha256") != protocol.get(
                                "checkpoint_sha256")
                            or outcome.get("policy_checkpoint_sha256") != row.get(
                                "checkpoint_sha256")
                            or outcome.get("source_identity_sha256") !=
                                protocol.get("source_identity_sha256")
                            or outcome.get("execution_device") != protocol.get("execution_device")
                            or outcome.get("condition") != arm):
                        raise ValueError(f"Phase 4 direct-pursuit row lacks exact simulator binding: {key}")
                    if arm == "factor_treatment" and (
                            outcome.get("applied_slots") != outcome.get("configured_slots")
                            or outcome.get("contact_events") is None):
                        raise ValueError(f"direct-pursuit treatment lacks applied slot/event capture: {key}")
                    if arm == "no_factor" and outcome.get("applied_slots") != []:
                        raise ValueError(f"direct-pursuit control unexpectedly applied policy: {key}")
                    continue
                if arm == "no_factor" and captured is not None:
                    raise ValueError("no-factor control unexpectedly applied a reactive policy")
                if arm == "factor_treatment":
                    expected_assignment = next((item for item in schedule.get(
                        "per_episode_assignment", ()) if item.get("seed") == key[2]
                            and item.get("category") == key[1]), None)
                    if (not isinstance(captured, dict)
                            or captured.get("policy_id") != schedule["policy_id"]
                            or captured.get("factor_id") != schedule["factor_id"]
                            or captured.get("policy_family") != schedule["policy_family"]
                            or captured.get("schedule_sha256") != schedule["schedule_sha256"]
                            or captured.get("scenario_id") != protocol.get("scenario_id")
                            or captured.get("category") != key[1]
                            or captured.get("seed") != key[2]
                            or captured.get("condition") != arm
                            or captured.get("qualification_case") != qualification_case
                            or captured.get("condition_id") != row.get("condition_id")
                            or captured.get("source_identity_sha256") != row.get(
                                "source_identity_sha256")
                            or captured.get("checkpoint_sha256") != row.get(
                                "checkpoint_sha256")
                            or captured.get("execution_device") != row.get(
                                "execution_device")
                            or not isinstance(expected_assignment, dict)
                            or captured.get("assignment_sha256") != expected_assignment.get(
                                "assignment_sha256")
                            or captured.get("configured_slots") != expected_assignment.get(
                                "configured_pressure_slots")
                            or captured.get("applied_slots") != expected_assignment.get(
                                "configured_pressure_slots")
                            or captured.get("applied_event_coverage_complete") is not True
                            or captured.get("applied_command_calls_by_slot") != {
                                str(slot): row.get("elapsed_ticks") for slot in
                                expected_assignment.get("configured_pressure_slots", ())}
                            or captured.get("episode_outcome_metrics") != {
                                name: row.get("metrics", {}).get(name) for name in (
                                    "success", "scores", "useful_throughput",
                                    "completed_cycles")}
                            or captured.get("interception_rate") is not None
                            or captured.get("paired_outcome_by_family") is not None):
                        raise ValueError(f"Phase 4 treatment row lacks policy provenance/null outcomes: {key}")
            arm_reports[condition_name] = rows
        arm_reports_by_name = arm_reports
        if arm == "no_factor":
            condition_names = set(arm_reports_by_name)
        elif set(arm_reports_by_name) != condition_names:
            raise ValueError("Phase 4 matrix arms have different condition coverage")
        reports_by_arm[arm] = arm_reports_by_name
        matrix_files[arm] = paths[0]
    pair_counts = {}
    direct_paired = []
    reactive_paired = []
    for condition_name in sorted(condition_names):
        control = reports_by_arm["no_factor"][condition_name]
        treatment = reports_by_arm["factor_treatment"][condition_name]
        if set(control) != set(treatment):
            raise ValueError(f"Phase 4 pair key coverage differs for {condition_name}")
        pair_counts[condition_name] = len(control)
        if direct_pursuit:
            from .meco_phase4_direct_pursuit import paired_direct_pursuit_outcomes
            for key in sorted(control):
                direct_paired.append(paired_direct_pursuit_outcomes(
                    control[key]["phase4_direct_pursuit"],
                    treatment[key]["phase4_direct_pursuit"]))
        elif schedule.get("factor_id") == "defender.reactive_pool":
            for key in sorted(control):
                captured = treatment[key].get("phase4_opponent_policy")
                if not isinstance(captured, dict):
                    raise ValueError(f"reactive treatment lacks applied policy capture: {key}")
                family = captured["policy_family"]
                reactive_paired.append({"policy_id": key[0],
                    "opponent_policy_id": schedule["policy_id"],
                    "policy_family": family, "category": key[1], "seed": key[2],
                    "control_episode_outcome": {name: control[key].get(
                        "metrics", {}).get(name) for name in ("success", "scores",
                            "useful_throughput", "completed_cycles")},
                    "treatment_episode_outcome": captured["episode_outcome_metrics"],
                    "outcome_semantics": "whole-team paired episode outcome while the policy was assigned; association by policy family, not per-policy causal contribution"})
    reactive_family_summary = None
    if schedule.get("factor_id") == "defender.reactive_pool":
        from .meco_phase4_outcome_bridge import reduce_policy_family_outcomes
        reactive_family_summary = reduce_policy_family_outcomes(reactive_paired,
            expected_policy_ids=policy_names, categories=categories, seeds=seeds,
            family_by_policy={policy: schedule["policy_family"]
                for policy in policy_names})
    pairing = {"schema": ("meco-phase4-direct-pursuit-pair-outcomes-v1" if direct_pursuit
                           else "meco-phase4-reactive-opponent-pair-coverage-v1"),
        "policy_id": schedule["policy_id"],
        "schedule_sha256": schedule["schedule_sha256"],
        "condition_names": sorted(condition_names),
        "control_policy_ids": sorted(policy_names),
        "categories": categories,
        "validation_seed_assignment_sha256": _sha(seeds),
        "paired_rows_per_condition": pair_counts,
        "exact_policy_category_seed_pair_coverage": True,
        "runtime_status": ("outcomes_captured_pending_replay_review" if direct_pursuit
                            else "scheduled_not_simulator_validated"),
        "simulator_behavior_validated": False,
        "outcome_signals_complete": bool(direct_pursuit and direct_paired),
        "paired_outcome_by_family": (reactive_family_summary[
            "paired_outcome_by_family"] if reactive_family_summary else None),
        "policy_family_coverage": (reactive_family_summary[
            "policy_family_coverage"] if reactive_family_summary else None),
        "paired_outcome_semantics": (reactive_family_summary[
            "paired_outcome_semantics"] if reactive_family_summary else None),
        "interception_rate": ({
            "numerator_intercepted_episodes": sum(
                row["interception_episode_numerator"] for row in direct_paired),
            "denominator_eligible_episodes": sum(
                row["interception_eligible_episode_denominator"]
                for row in direct_paired),
            "rate": (sum(row["interception_episode_numerator"]
                for row in direct_paired) / sum(
                    row["interception_eligible_episode_denominator"]
                    for row in direct_paired)) if direct_paired else None}
            if direct_pursuit else None),
        "contact_frequency": ({
            "numerator_distinct_pair_contact_substeps": sum(
                sum(pair["distinct_contact_substep_count"] for pair in
                    reports_by_arm["factor_treatment"][condition_name][key][
                        "phase4_direct_pursuit"]["contact_pairs"])
                for condition_name in sorted(condition_names)
                for key in sorted(reports_by_arm["factor_treatment"][condition_name])),
            "denominator_pair_seconds": sum(
                reports_by_arm["factor_treatment"][condition_name][key][
                    "phase4_direct_pursuit"]["robot_contact_rate_denominator_pair_seconds"]
                for condition_name in sorted(condition_names)
                for key in sorted(reports_by_arm["factor_treatment"][condition_name])),
            "rate_distinct_pair_contact_substeps_per_pair_second": (
                sum(reports_by_arm["factor_treatment"][condition_name][key][
                    "phase4_direct_pursuit"]["robot_contact_rate_numerator_distinct_substeps"]
                    for condition_name in sorted(condition_names)
                    for key in sorted(reports_by_arm["factor_treatment"][condition_name])) /
                sum(reports_by_arm["factor_treatment"][condition_name][key][
                    "phase4_direct_pursuit"]["robot_contact_rate_denominator_pair_seconds"]
                    for condition_name in sorted(condition_names)
                    for key in sorted(reports_by_arm["factor_treatment"][condition_name])))
                if direct_pursuit and treatment else None}
            if direct_pursuit else None),
        "robot_contact_rate": ({"mean_pair_contact_events_per_pair_second":
            sum(treatment[key]["phase4_direct_pursuit"]["robot_contact_rate"]
                for key in treatment) / len(treatment)} if direct_pursuit and treatment else None),
        "paired_time_loss_to_opponents_s": ([row["time_lost_to_opponents_s"]
            for row in direct_paired] if direct_pursuit else None),
        "paired_time_loss_by_policy": ({policy: {
            "eligible_pairs": sum(row["policy_id"] == policy for row in direct_paired),
            "mean_time_lost_to_opponents_s": sum(row["time_lost_to_opponents_s"]
                for row in direct_paired if row["policy_id"] == policy) /
                sum(row["policy_id"] == policy for row in direct_paired)}
            for policy in sorted({row["policy_id"] for row in direct_paired})}
            if direct_pursuit and direct_paired else None),
        "paired_direct_pursuit_outcomes": direct_paired if direct_pursuit else None,
        "qualification_eligible": False,
        "matrix_sha256": {arm: _file_sha(path) for arm, path in matrix_files.items()}}
    pairing["pairing_sha256"] = _sha(pairing)
    artifact_path = case_dir / "paired-coverage.json"
    _write_immutable(artifact_path, pairing)
    return pairing


def _validate_phase4_trench_interception_pair(case_dir: Path, binding: dict, *,
        policy_ids: list[str], category: str, seeds: list[int],
        qualification_case: dict) -> dict:
    """Reconcile all assigned trench rows and retain invalid policy/seed pairs."""
    from .meco_phase4_trench_interception import (
        reduce_trench_interception_pair, validate_trench_interception_record)
    expected_policy_ids = sorted((*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2"))
    if policy_ids != expected_policy_ids or len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("trench pair validator requires the exact 11-by-100 frozen matrix")
    expected_keys = {(policy, category, seed) for policy in policy_ids for seed in seeds}
    outcomes, row_errors, arm_rows, matrix_files = {}, {"no_factor": [], "factor_treatment": []}, {}, {}
    paired_binding = None
    for arm in CONDITIONS:
        matrix_paths = sorted((case_dir / arm).glob("*.matrix.json"))
        if len(matrix_paths) != 1:
            raise ValueError(f"trench {arm} arm lacks exactly one matrix report")
        matrix = _read_json(matrix_paths[0])
        protocol = matrix.get("protocol", {})
        arm_binding = protocol.get("phase4_trench_interception_binding")
        expected_schedule = binding["schedule"] if arm == "factor_treatment" else None
        if (protocol.get("qualification_case") != qualification_case
                or protocol.get("categories") != [category]
                or protocol.get("seed_list") != seeds
                or protocol.get("phase4_opponent_schedule") != expected_schedule
                or not isinstance(arm_binding, dict)
                or arm_binding.get("schedule_sha256") != binding["schedule_sha256"]
                or not arm_binding.get("pair_binding_id")
                or arm_binding.get("pair_binding", {}).get("source_identity_sha256") !=
                    protocol.get("source_identity_sha256")
                or arm_binding.get("pair_binding", {}).get("execution_device") !=
                    protocol.get("execution_device")):
            raise ValueError(f"trench {arm} matrix differs from frozen assignment/provenance")
        if paired_binding is None:
            paired_binding = arm_binding
        elif arm_binding != paired_binding:
            raise ValueError("trench arms do not share the exact stable pair binding")
        matrix_files[arm] = _file_sha(matrix_paths[0])
        entries = matrix.get("conditions")
        if not isinstance(entries, dict) or set(entries) != {"architecture_matched_cv_k16"}:
            raise ValueError(f"trench {arm} matrix has unexpected condition coverage")
        report = _read_json(Path(entries["architecture_matched_cv_k16"]["report_path"]).resolve())
        rows = _records_by_key(report)
        if set(rows) != expected_keys or report.get("qualification_eligible") is not False:
            raise ValueError(f"trench {arm} report has missing/extra assignments or qualification state")
        outcomes[arm], row_errors[arm], arm_rows[arm] = {}, [], {}
        for key in sorted(expected_keys):
            row = rows[key]
            record = row.get("phase4_trench_interception")
            try:
                if not isinstance(record, dict):
                    raise ValueError("missing trench outcome record")
                check = validate_trench_interception_record(record,
                    binding=arm_binding, expected_condition=arm,
                    expected_policy_id=key[0], expected_seed=key[2],
                    expected_source_sha256=protocol["source_identity_sha256"],
                    expected_checkpoint_sha256=row.get("checkpoint_sha256"),
                    expected_device=protocol["execution_device"])
                arm_rows[arm][key] = (record, check)
            except (KeyError, TypeError, ValueError) as exc:
                row_errors[arm].append({"policy_id": key[0], "seed": key[2],
                    "reason": str(exc)})
                arm_rows[arm][key] = (record, None)
    paired, pair_statuses = [], []
    for key in sorted(expected_keys):
        control, control_check = arm_rows["no_factor"][key]
        treatment, treatment_check = arm_rows["factor_treatment"][key]
        valid = (control_check is not None and treatment_check is not None and
            control_check["crossing_row_valid"] and treatment_check["crossing_row_valid"] and
            control_check["full_horizon_pose_valid"] and
            treatment_check["full_horizon_pose_valid"])
        if valid:
            try:
                paired.append(reduce_trench_interception_pair(control, treatment,
                    binding=paired_binding, policy_id=key[0], seed=key[2],
                    source_identity_sha256=paired_binding["source_identity_sha256"],
                    checkpoint_sha256=control["policy_checkpoint_sha256"],
                    execution_device=paired_binding["execution_device"]))
            except (KeyError, TypeError, ValueError) as exc:
                valid = False
                reason = str(exc)
        else:
            reason = "one_or_both_arm_rows_failed_full_horizon_or_contact_validation"
        if not valid:
            row_errors["no_factor" if control_check is None else "factor_treatment"].append(
                {"policy_id": key[0], "seed": key[2], "reason": reason})
        pair_statuses.append({"policy_id": key[0], "seed": key[2], "valid_pair": bool(valid),
            "invalid_reason": None if valid else reason,
            "treatment_interception_row_valid": bool(treatment_check is not None and
                treatment_check.get("interception_row_valid"))})
    valid_pair_keys = {(row["policy_id"], row["seed"]) for row in pair_statuses
        if row.get("valid_pair") is True}
    paired_valid = [row for row in paired
        if (row.get("policy_id"), row.get("seed")) in valid_pair_keys and
        "alternate_crossing_success_delta" in row]
    pair_count = len(expected_keys)
    per_arm = {}
    for arm in CONDITIONS:
        full_rows = [(record, check) for record, check in arm_rows[arm].values()
            if check is not None and check.get("full_horizon_pose_valid") is True]
        crossing_rows = [(record, check) for record, check in full_rows
            if check.get("crossing_row_valid") is True]
        per_arm[arm] = {
            "assigned_count": pair_count,
            "full_horizon_valid_count": len(full_rows),
            "invalid_or_censored_count": pair_count - len(full_rows),
            "alternate_crossing_success_numerator": sum(
                check["alternate_crossing_success"] is True for _, check in crossing_rows),
            "alternate_crossing_success_denominator": len(crossing_rows),
            "crossing_time_rmst_numerator_s": sum(
                check["trench_crossing_time_s"] for _, check in crossing_rows),
            "crossing_time_rmst_denominator": len(crossing_rows),
            "crossing_event_count": sum(not check["crossing_right_censored"]
                for _, check in crossing_rows),
            "horizon_censor_count": sum(check["crossing_right_censored"]
                for _, check in crossing_rows),
            "restricted_mean_crossing_time_s": (
                sum(check["trench_crossing_time_s"] for _, check in crossing_rows) /
                len(crossing_rows) if crossing_rows else None),
            "interception_numerator": (sum(check["interception_episode_numerator"] == 1
                for _, check in full_rows if check.get("interception_episode_numerator") is not None)
                if arm == "factor_treatment" else None),
            "interception_denominator": (sum(check.get("interception_row_valid") is True
                for _, check in full_rows) if arm == "factor_treatment" else None)}
    coverage = {"schema": "meco-phase4-trench-interception-pair-outcomes-v1",
        "factor_id": "opponent.trench_interception",
        "policy_id": "trench_interceptor",
        "protocol_id": binding["protocol_id"],
        "protocol_sha256": binding["protocol_sha256"],
        "scenario_id": binding["scenario_id"],
        "qualification_case": binding["qualification_case"],
        "split": "validation",
        "policy_ids": policy_ids, "category": category, "seed_count": len(seeds),
        "assigned_rows_per_condition": {arm: pair_count for arm in CONDITIONS},
        "paired_rows_per_condition": {arm: pair_count for arm in CONDITIONS},
        "invalid_pair_count": pair_count - len(paired_valid),
        "valid_pair_count": len(paired_valid),
        "treatment_interception_valid_count": per_arm["factor_treatment"][
            "interception_denominator"],
        "treatment_interception_invalid_count": pair_count - per_arm[
            "factor_treatment"]["interception_denominator"],
        "unaccounted_assignments": 0,
        "arm_invalid_row_count": {arm: len(row_errors[arm]) for arm in CONDITIONS},
        "arm_invalid_rows": row_errors,
        "per_arm_outcomes": per_arm,
        "paired_outcomes": paired_valid,
        "exact_pair_coverage": len(pair_statuses) == pair_count,
        "all_pairs_full_horizon_valid": len(paired_valid) == pair_count,
        "outcome_signals_complete": (len(paired_valid) == pair_count and
            per_arm["factor_treatment"]["interception_denominator"] == pair_count),
        "runtime_status": "outcomes_captured_pending_replay_review",
        "source_identity_sha256": paired_binding.get("source_identity_sha256"),
        "checkpoint_bundle_sha256": paired_binding.get("checkpoint_bundle_sha256"),
        "execution_device": paired_binding.get("execution_device"),
        "schedule_sha256": binding["schedule_sha256"],
        "policy_config_sha256": binding["policy_config_sha256"],
        "seed_manifest_sha256": paired_binding.get("seed_manifest_sha256"),
        "validation_seed_assignment_sha256": _sha(seeds),
        "assigned_pressure_slots": [0],
        "control_policy_ids": policy_ids,
        "qualification_eligible": False,
        "matrix_sha256": matrix_files}
    coverage["pairing_sha256"] = _sha(coverage)
    coverage["paired_outcomes_artifact_sha256"] = _write_immutable(
        case_dir / "paired-trench-outcomes.json", coverage)
    return coverage


def _write_immutable(path: Path, value: dict) -> str:
    return _write_immutable_bytes(path, _artifact_bytes(value))


def _write_immutable_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable campaign artifact conflicts: {path}")
        return hashlib.sha256(data).hexdigest()
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"immutable campaign artifact conflicts: {path}")
        dfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        temp.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def _validate_plan(plan: dict) -> None:
    if plan.get("schema") != "meco-validation-campaign-plan-v1":
        raise ValueError("unsupported validation plan schema")
    unhashed = {key: value for key, value in plan.items() if key != "plan_digest"}
    if plan.get("plan_digest") != _sha(unhashed):
        raise ValueError("validation plan digest mismatch")
    if (plan.get("split") != "validation" or plan.get("qualification_claim") is not False
            or plan.get("final_test_seed_handling") != "not_exported"):
        raise ValueError("campaign execution is restricted to diagnostic validation plans")
    protocol = plan.get("protocol", {})
    version = protocol.get("protocol_version")
    gate_status = protocol.get("gate_calibration_status")
    version_matches_status = (version in CALIBRATED_GATE_STATUSES
        and gate_status == CALIBRATED_GATE_STATUSES[version]
        and protocol.get("safety_gate_status") == "metrics_schema_verified")
    eligible_v4 = (version == CALIBRATED_PROTOCOL_VERSION and version_matches_status)
    provisional = (version == PROTOCOL_VERSION
        and gate_status == "provisional_pending_baseline_variance_review")
    legacy_diagnostic = (version in (LEGACY_CALIBRATED_PROTOCOL_VERSION,
        D0_PAIRED_REFERENCE_PROTOCOL_VERSION) and version_matches_status)
    if (not eligible_v4 and not legacy_diagnostic and not provisional) or (
            plan.get("qualification_eligible") is not eligible_v4):
        raise ValueError("campaign plan calibration eligibility/status is inconsistent")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 28 or plan.get("job_count") != 28:
        raise ValueError("campaign requires the exact 28-job all-scenario validation matrix")
    if plan.get("scenario_ids") != list(SCENARIOS):
        raise ValueError("campaign plan scenario registry differs from the current curriculum")
    seen = set()
    seen_case_ids = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("campaign plan contains a non-validation job")
        jid = job.get("job_id")
        if not isinstance(jid, str) or jid != _sha({k: v for k, v in job.items() if k != "job_id"}) or jid in seen:
            raise ValueError("job ID is invalid or duplicated")
        seen.add(jid)
        case_identity = (job.get("scenario_id"), _canonical_bytes(job.get("qualification_case")))
        if (job.get("scenario_id") not in SCENARIOS or case_identity in seen_case_ids
                or not set(job.get("categories", ())).issubset(
                    set(SCENARIOS[job["scenario_id"]].required_categories))):
            raise ValueError("job scenario/category/qualification-case identity is invalid or duplicated")
        seen_case_ids.add(case_identity)
        if (not isinstance(job.get("checkpoint_sha256"), dict)
                or set(job["checkpoint_sha256"]) != set(MECO_TRACKS)
                or not isinstance(job.get("source_identity"), dict)
                or not isinstance(job.get("training_configuration_sha256"), dict)
                or set(job["training_configuration_sha256"]) != set(MECO_TRACKS)):
            raise ValueError("each job requires all five checkpoint/config hashes and a bound source identity")
        manifest = job.get("seed_manifest")
        if (not isinstance(manifest, dict) or manifest.get("split") != "validation"
                or manifest.get("scenario_id") != job.get("scenario_id")
                or manifest.get("seed_list") != job.get("seed_list")
                or hashlib.sha256(_artifact_bytes(manifest)).hexdigest() != job.get("seed_manifest_sha256")):
            raise ValueError("job seed manifest binding is invalid")
        if not job.get("conditions"):
            raise ValueError("job has no matrix conditions")
        for track in MECO_TRACKS:
            for binding in (job["checkpoint_sha256"][track],
                            job["training_configuration_sha256"][track]):
                if (not isinstance(binding, str) or len(binding) != 64
                        or any(c not in "0123456789abcdef" for c in binding)):
                    raise ValueError(f"{track} job binding is not a SHA-256 digest")


def _expected_keys(condition: dict) -> set[tuple[str, str, int]]:
    coverage = condition.get("expected_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("condition lacks expected coverage contract")
    expected = {(policy, category, seed) for policy in coverage["policies"]
                for category in coverage["categories"] for seed in coverage["seed_list"]}
    if len(expected) != coverage.get("record_count"):
        raise ValueError("condition expected coverage count is inconsistent")
    return expected


def _validate_replay_artifact(replay_record: dict, replay_ref: dict, *,
                              report: dict, report_path: Path, row: dict,
                              report_index: int, job: dict) -> None:
    """Bind a producer-shaped replay bundle back to its exact evaluation row."""
    provenance = replay_record.get("replay_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("replay bundle lacks producer replay_provenance")
    expected_record_digest = _sha({key: value for key, value in row.items()
                                   if key != "dashboard_replay"})
    expected_policy = row.get("policy")
    expected_checkpoint = row.get("checkpoint_sha256")
    expected_case = job["qualification_case"]
    if (replay_record.get("evaluation_id") != report.get("evaluation_id")
            or replay_record.get("condition_id") != report.get("experiment_matrix", {}).get("condition_id")
            or replay_ref.get("simulation_id") != replay_record.get("simulation_id")
            or replay_record.get("scenario_id") != job["scenario_id"]
            or replay_record.get("category") != row.get("category")
            or replay_record.get("seed") != row.get("seed")
            or replay_record.get("policy") != expected_policy
            or replay_record.get("split") != "validation"
            or replay_record.get("qualification_case") != expected_case):
        raise ValueError("replay root identity differs from its selected evaluation row")
    if (replay_ref.get("evaluation_report_path") != str(report_path.resolve())
            or replay_record.get("evaluation_report_path") != str(report_path.resolve())
            or replay_ref.get("evaluation_record_index") != report_index
            or replay_record.get("evaluation_record_index") != report_index
            or replay_ref.get("evaluation_record_digest") != expected_record_digest
            or replay_record.get("evaluation_record_digest") != expected_record_digest):
        raise ValueError("replay pointer/digest does not bind the exact evaluation row")
    source_sha = job["source_identity"]["sha256"]
    protocol_id = job["seed_manifest"]["protocol_id"]
    if (replay_record.get("source_identity_sha256") != source_sha
            or replay_record.get("replay_source_sha256") != source_sha
            or provenance.get("replay_source_sha256") != source_sha
            or replay_record.get("qualification_protocol_id") != protocol_id
            or provenance.get("qualification_protocol_id") != protocol_id
            or replay_record.get("protocol_id") != row.get("protocol_id")
            or provenance.get("protocol_id") != row.get("protocol_id")
            or replay_record.get("checkpoint_sha256") != expected_checkpoint
            or provenance.get("checkpoint_sha256") != expected_checkpoint
            or provenance.get("evaluation_id") != report.get("evaluation_id")
            or provenance.get("condition_id") != replay_record.get("condition_id")
            or provenance.get("scenario_id") != job["scenario_id"]
            or provenance.get("category") != row.get("category")
            or provenance.get("seed") != row.get("seed")
            or provenance.get("policy") != expected_policy
            or provenance.get("split") != "validation"
            or provenance.get("qualification_case") != expected_case
            or provenance.get("evaluation_record_index") != report_index
            or provenance.get("evaluation_record_digest") != expected_record_digest):
        raise ValueError("replay provenance differs from campaign source/protocol/row binding")
    safety_metrics = row.get("meco_metric_ledger", {}).get("safety_metrics")
    route_digest = _sha({"route_trace": row.get("route_trace"),
        "executed_planner_route_trace": row.get("executed_planner_route_trace")})
    outcome_digest = _sha({"metrics": row.get("metrics"),
        "safety_metrics": safety_metrics,
        "execution_failures": row.get("execution_failures")})
    safety_digest = _sha(safety_metrics)
    for field, expected in (("route_digest", route_digest),
                            ("outcome_digest", outcome_digest),
                            ("safety_digest", safety_digest)):
        if replay_record.get(field) != expected or provenance.get(field) != expected:
            raise ValueError(f"replay {field} differs from its source evaluation record")
    if expected_policy in MECO_TRACKS and expected_checkpoint != job["checkpoint_sha256"][expected_policy]:
        raise ValueError("replay checkpoint differs from the bound learned-track checkpoint")


def validate_job_artifacts(job: dict, job_dir: Path, *, device: str) -> dict:
    """Validate a complete matrix and every condition/replay before permitting a seal."""
    matrix_dir = job_dir / "matrix"
    matrices = sorted(matrix_dir.glob("*.matrix.json"))
    if len(matrices) != 1:
        raise ValueError("job does not contain exactly one completed matrix report")
    matrix_path = matrices[0]
    # This performs condition report digest/provenance, paired-contrast and replay
    # reference checks. It is diagnostic-only and does not execute simulations.
    from scripts.summarize_meco_ablation_matrix import summarize_matrix
    summarize_matrix(matrix_path, replicates=100)
    matrix = _read_json(matrix_path)
    protocol = matrix.get("protocol", {})
    if (matrix.get("evaluation_split") != "validation"
            or matrix.get("evaluation_design") != "frozen_protocol_held_out"
            or matrix.get("qualification_protocol_id") != job["seed_manifest"]["protocol_id"]
            or matrix.get("scenario_id") != job["scenario_id"]
            or matrix.get("qualification_case") != job["qualification_case"]
            or matrix.get("protocol", {}).get("qualification_protocol", {}).get("protocol_sha256")
                != job["seed_manifest"]["protocol_sha256"]):
        raise ValueError("matrix protocol/split/scenario differs from immutable job input")
    if (protocol.get("seed_list") != job["seed_list"]
            or protocol.get("categories") != job["categories"]
            or protocol.get("qualification_case") != job["qualification_case"]
            or protocol.get("checkpoint_sha256") != job["checkpoint_sha256"]
            or protocol.get("training_configuration_sha256") != job["training_configuration_sha256"]
            or protocol.get("source_identity_sha256") != job["source_identity"]["sha256"]
            or protocol.get("execution_device", {}).get("device") != device):
        raise ValueError("matrix inputs differ from immutable campaign job binding")
    expected_replay_selector = {condition["condition_name"]: condition["replay_selection"]
        for condition in job["conditions"] if condition.get("replay_selection")}
    if protocol.get("replay_selection") != expected_replay_selector:
        raise ValueError("matrix replay selector differs from immutable campaign inputs")
    condition_map = matrix.get("conditions")
    if not isinstance(condition_map, dict) or set(condition_map) != {c["condition_name"] for c in job["conditions"]}:
        raise ValueError("matrix condition table differs from exact planned condition set")
    report_digests = {}
    replay_digests = {}
    referenced_replays = set()
    for condition in job["conditions"]:
        condition_fields = ("condition_name", "prediction_mode", "control_prediction_mode",
            "candidate_k", "inference_disable_tick", "phase4_factor", "categories",
            "qualification_case")
        expected_spec = {key: condition.get(key) for key in condition_fields}
        observed_specs = {item.get("condition_name"): item for item in protocol.get("conditions", [])}
        if _canonical_bytes(observed_specs.get(condition["condition_name"])) != _canonical_bytes(expected_spec):
            raise ValueError("matrix condition specification differs from the planned condition")
        entry = condition_map[condition["condition_name"]]
        if (entry.get("execution_device", {}).get("device") != device
                or entry.get("source_identity_sha256") != job["source_identity"]["sha256"]
                or entry.get("checkpoint_sha256") != job["checkpoint_sha256"]
                or entry.get("training_configuration_sha256") != job["training_configuration_sha256"]):
            raise ValueError("condition provenance differs from campaign input")
        report_path = Path(entry["report_path"]).resolve()
        if not report_path.is_file():
            raise ValueError("condition evaluation report is missing")
        report = _read_json(report_path)
        records = _records_by_key(report)
        if set(records) != _expected_keys(condition):
            raise ValueError(f"condition record coverage differs: {condition['condition_name']}")
        if report.get("split") != "validation" or report.get("scenario_id") != job["scenario_id"]:
            raise ValueError("condition report split/scenario mismatch")
        report_digests[str(report_path)] = _file_sha(report_path)
        selector = condition.get("replay_selection", {})
        selected = {(policy, category, seed) for policy, seeds in selector.items()
                    for seed in seeds for category in job["categories"]}
        exports = report.get("dashboard_replays", [])
        exported = set()
        for replay in exports:
            index = replay.get("evaluation_record_index")
            if type(index) is not int or not 0 <= index < len(report.get("records", [])):
                raise ValueError("replay export record index is invalid")
            row = report["records"][index]
            key = (row.get("policy"), row.get("category"), row.get("seed"))
            exported.add(key)
            replay_path = Path(replay.get("result_path", "")).resolve()
            if not replay_path.is_file():
                raise ValueError("selected replay artifact is missing")
            replay_record = _read_json(replay_path)
            _validate_replay_artifact(replay_record, replay, report=report,
                report_path=report_path, row=row, report_index=index, job=job)
            replay_digests[str(replay_path)] = _file_sha(replay_path)
            referenced_replays.add(replay_path)
        if exported != selected:
            raise ValueError(f"replay export coverage differs: {condition['condition_name']}")
    replay_root = job_dir / "replays"
    if replay_root.exists():
        actual_replays = {path.resolve() for path in replay_root.glob("*.result.json")}
        if actual_replays != referenced_replays:
            raise ValueError("replay directory has missing or unreferenced partial artifacts")
    seed_path = job_dir / job["seed_manifest_file"]
    case_path = job_dir / "qualification-case.json"
    selector_path = job_dir / "replay-selection.json"
    if not seed_path.is_file() or _read_json(seed_path) != job["seed_manifest"]:
        raise ValueError("immutable per-job seed manifest is missing or changed")
    if not case_path.is_file() or _read_json(case_path) != job["qualification_case"]:
        raise ValueError("immutable per-job qualification case is missing or changed")
    expected_selector = {condition["condition_name"]: condition["replay_selection"]
        for condition in job["conditions"] if condition.get("replay_selection")}
    if expected_selector:
        if not selector_path.is_file() or _read_json(selector_path) != expected_selector:
            raise ValueError("immutable replay selector is missing or changed")
        selector_digest = _file_sha(selector_path)
    else:
        selector_digest = None
    return {"matrix_path": str(matrix_path), "matrix_sha256": _file_sha(matrix_path),
            "condition_report_sha256": report_digests,
            "replay_sha256": replay_digests,
            "seed_manifest_sha256": _file_sha(seed_path),
            "qualification_case_sha256": _file_sha(case_path),
            "replay_selection_sha256": selector_digest,
            "matrix_report_sha256": matrix.get("report_sha256"),
            "matrix_id": matrix.get("matrix_id")}


def _completion_valid(job: dict, job_dir: Path, *, device: str,
                     expected_plan_digest: str) -> dict | None:
    seal_path = job_dir / "job.complete.json"
    if not seal_path.is_file():
        return None
    seal = _read_json(seal_path)
    payload = {key: value for key, value in seal.items() if key != "completion_sha256"}
    if (seal.get("schema") != JOB_COMPLETE_SCHEMA or seal.get("job_id") != job["job_id"]
            or seal.get("plan_digest") != expected_plan_digest
            or seal.get("completion_sha256") != _sha(payload)):
        raise ValueError("job completion seal is invalid")
    if seal.get("device") != device or seal.get("diagnostic_only") is not True:
        raise ValueError("job completion seal has invalid device or diagnostic gate")
    input_path = job_dir / "job.input.json"
    job_input = _read_json(input_path) if input_path.is_file() else {}
    if (not input_path.is_file() or _file_sha(input_path) != seal.get("job_input_sha256")
            or _canonical_bytes(job_input.get("job")) != _canonical_bytes(job)
            or job_input.get("plan_digest") != seal.get("plan_digest")):
        raise ValueError("job completion seal is not bound to its immutable job input")
    validated = validate_job_artifacts(job, job_dir, device=device)
    if seal.get("artifacts") != validated:
        raise ValueError("sealed job artifacts changed after completion")
    return seal


def run_campaign(plan: dict, *, output_dir: str | Path, device: str,
                 checkpoint_paths: dict[str, str], matrix_script: str | Path,
                 protocol_path: str | Path,
                 current_source_identity: dict,
                 runner: Callable[[list[str]], int] | None = None) -> dict:
    """Execute/resume a diagnostic validation campaign; no test split is accepted."""
    _validate_plan(plan)
    if re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", device) is None:
        raise ValueError("device must be an indexed CUDA/HIP accelerator")
    if set(checkpoint_paths) != set(MECO_TRACKS):
        raise ValueError("checkpoint paths must cover exactly all five tracks")
    for track, path in checkpoint_paths.items():
        if _file_sha(Path(path)) != plan["jobs"][0]["checkpoint_sha256"][track]:
            raise ValueError(f"{track} checkpoint differs from plan")
    protocol_path = Path(protocol_path).resolve()
    if _file_sha(protocol_path) != plan["protocol"]["protocol_sha256"]:
        raise ValueError("raw qualification protocol SHA-256 differs from immutable plan")
    planned_job = plan["jobs"][0]
    if planned_job["source_identity"]["identity"] != current_source_identity:
        raise ValueError("current prepared source identity differs from frozen plan")
    rebuilt = compile_validation_campaign(protocol_path=protocol_path,
        checkpoint_paths=checkpoint_paths, source_identity=current_source_identity,
        training_configuration_sha256=planned_job["training_configuration_sha256"],
        qualification_claim=False)
    if _canonical_bytes(rebuilt) != _canonical_bytes(plan):
        raise ValueError("plan differs from deterministic all-scenario validation compilation")
    root = Path(output_dir).resolve()
    campaign_lock = root / ".campaign.lock"
    device_lease = lease_devices((device,), owner=f"validation:{plan['plan_digest']}")
    device_lease.acquire()
    acquired = []
    success = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        for lock in (campaign_lock,):
            try:
                fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as exc:
                raise ValueError(f"campaign/device lock exists and must be inspected manually: {lock}") from exc
            acquired.append(lock)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"device": device, "plan_digest": plan["plan_digest"],
                    "pid": os.getpid()}, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        # A newly acquired lock is removed only when the acquisition itself
        # failed before work began; existing locks are never cleared.
        for lock in acquired:
            lock.unlink(missing_ok=True)
        device_lease.release()
        raise
    try:
        plan_path = root / "plan.json"
        _write_immutable(plan_path, plan)
        for job in plan["jobs"]:
            job_dir = root / "jobs" / job["job_id"]
            job_dir.mkdir(parents=True, exist_ok=True)
            input_payload = {"schema": JOB_INPUT_SCHEMA, "plan_digest": plan["plan_digest"],
                "job": job, "device": device, "diagnostic_only": True,
                "qualification_claim": False}
            input_path = job_dir / "job.input.json"
            input_digest = _write_immutable(input_path, input_payload)
            if (job_dir / "job.complete.json").exists():
                seal = _completion_valid(job, job_dir, device=device,
                    expected_plan_digest=plan["plan_digest"])
                if seal["job_input_sha256"] != input_digest:
                    raise ValueError("completed job input digest differs")
                continue
            matrix_dir = job_dir / "matrix"
            existing = list(job_dir.iterdir())
            unexpected = [p for p in existing if p.name not in
                          ("job.input.json", "matrix")]
            if unexpected:
                raise ValueError("unsealed job directory contains unexpected/partial artifacts")
            if matrix_dir.exists() and any(matrix_dir.iterdir()):
                # An unsealed directory may be sealed only if the matrix and all
                # linked condition/replay artifacts validate in full.
                artifacts = validate_job_artifacts(job, job_dir, device=device)
            else:
                if matrix_dir.exists():
                    matrix_dir.rmdir()
                matrix_dir.mkdir(parents=True, exist_ok=True)
                seed_path = job_dir / job["seed_manifest_file"]
                seed_digest = _write_immutable(seed_path, job["seed_manifest"])
                selector = {condition["condition_name"]: condition["replay_selection"]
                    for condition in job["conditions"] if condition.get("replay_selection")}
                argv = [sys.executable, str(Path(matrix_script).resolve()),
                    "--scenario", job["scenario_id"], "--categories", ",".join(job["categories"]),
                    "--seed-manifest", str(seed_path.resolve()), "--qualification-protocol",
                    str(protocol_path), "--qualification-case-json", "",
                    "--device", device, "--output-dir", str(matrix_dir)]
                # The matrix reads only the named case object, so use a dedicated
                # immutable case file instead of the broader job input payload.
                case_path = job_dir / "qualification-case.json"
                case_digest = _write_immutable(case_path, job["qualification_case"])
                argv[argv.index("")] = str(case_path.resolve())
                if selector:
                    selector_path = job_dir / "replay-selection.json"
                    selector_digest = _write_immutable(selector_path, selector)
                    argv += ["--replay-selection-json", str(selector_path.resolve()),
                             "--replay-output-dir", str((job_dir / "replays").resolve())]
                else:
                    selector_digest = None
                for track in MECO_TRACKS:
                    argv += [f"--{track.lower()}-checkpoint", str(Path(checkpoint_paths[track]).resolve())]
                execute = runner or (lambda command: subprocess.run(command, check=False).returncode)
                result = execute(argv)
                if result != 0:
                    raise RuntimeError(f"matrix runner failed for job {job['job_id']}: exit {result}")
                artifacts = validate_job_artifacts(job, job_dir, device=device)
                if (artifacts["seed_manifest_sha256"] != seed_digest
                        or artifacts["qualification_case_sha256"] != case_digest
                        or artifacts["replay_selection_sha256"] != selector_digest):
                    raise ValueError("validated job input artifact digests changed")
            seal = {"schema": JOB_COMPLETE_SCHEMA, "job_id": job["job_id"],
                "plan_digest": plan["plan_digest"], "job_input_sha256": input_digest,
                "device": device, "diagnostic_only": True, "artifacts": artifacts}
            seal["completion_sha256"] = _sha(seal)
            _write_immutable(job_dir / "job.complete.json", seal)
        completions = []
        for job in plan["jobs"]:
            seal = _completion_valid(job, root / "jobs" / job["job_id"], device=device,
                expected_plan_digest=plan["plan_digest"])
            completions.append({"job_id": job["job_id"],
                "completion_sha256": seal["completion_sha256"]})
        campaign = {"schema": CAMPAIGN_SCHEMA, "plan_digest": plan["plan_digest"],
            "diagnostic_only": True, "qualification_claim": False,
            "qualification_eligible": False, "split": "validation", "device": device,
            "job_count": 28, "jobs": completions}
        campaign["campaign_sha256"] = _sha(campaign)
        _write_immutable(root / "campaign.complete.json", campaign)
        success = True
        return campaign
    finally:
        # Keep lock on failure as a manual inspection signal. Remove only after
        # successful sealing, to serialize simultaneous campaigns safely.
        if success:
            for lock in acquired:
                lock.unlink(missing_ok=True)
        device_lease.release()


def aggregate_campaign(plan: dict, campaign_path: str | Path) -> dict:
    """Verify an exact complete 28-job set and preserve per-scenario summaries."""
    _validate_plan(plan)
    path = Path(campaign_path).resolve()
    campaign = _read_json(path)
    unsigned = {key: value for key, value in campaign.items() if key != "campaign_sha256"}
    if (campaign.get("schema") != CAMPAIGN_SCHEMA or campaign.get("campaign_sha256") != _sha(unsigned)
            or campaign.get("plan_digest") != plan["plan_digest"] or campaign.get("split") != "validation"
            or campaign.get("diagnostic_only") is not True or campaign.get("qualification_claim") is not False
            or campaign.get("qualification_eligible") is not False):
        raise ValueError("campaign completion manifest is invalid or not diagnostic validation")
    by_id = {job["job_id"]: job for job in plan["jobs"]}
    rows = campaign.get("jobs")
    if not isinstance(rows, list) or len(rows) != 28 or {row.get("job_id") for row in rows} != set(by_id):
        raise ValueError("campaign manifest does not cover the exact 28 planned jobs")
    scenario_rows: dict[str, list[dict]] = {}
    sample_identities = set()
    for row in rows:
        job = by_id[row["job_id"]]
        case = job["qualification_case"]
        for category in job["categories"]:
            for seed in job["seed_list"]:
                sample = (job["scenario_id"], _canonical_bytes(case), category, seed)
                if sample in sample_identities:
                    raise ValueError("campaign contains duplicate scenario-aware sample identity")
                sample_identities.add(sample)
        # The aggregate's sibling job directory is resolved from the campaign root.
        job_dir = path.parent / "jobs" / job["job_id"]
        seal = _completion_valid(job, job_dir, device=campaign["device"],
            expected_plan_digest=plan["plan_digest"])
        if seal["completion_sha256"] != row.get("completion_sha256"):
            raise ValueError("campaign links to a different job completion seal")
        matrix_path = Path(seal["artifacts"]["matrix_path"])
        from scripts.summarize_meco_ablation_matrix import summarize_matrix
        summary = summarize_matrix(matrix_path, replicates=100)
        scenario_rows.setdefault(job["scenario_id"], []).append({
            "job_id": job["job_id"], "matrix_id": summary["matrix"]["matrix_id"],
            "matrix_sha256": summary["matrix"]["file_sha256"],
            "summary": [{**item, "scenario_id": job["scenario_id"],
                         "job_id": job["job_id"]} for item in summary["summaries"]]})
    scenarios = [{"scenario_id": scenario, "jobs": scenario_rows[scenario]}
                 for scenario in sorted(scenario_rows)]
    if len(scenarios) != len({job["scenario_id"] for job in plan["jobs"]}):
        raise ValueError("campaign scenario coverage is incomplete")
    result = {"schema": "meco-validation-campaign-diagnostic-summary-v1",
        "campaign_sha256": campaign["campaign_sha256"], "plan_digest": plan["plan_digest"],
        "diagnostic_only": True, "promotion_inference": False,
        "pooled_estimate": None,
        "pooling_note": "No cross-scenario pooled estimate is computed; operational weighting is unspecified.",
        "scenario_count": len(scenarios), "job_count": len(rows), "scenarios": scenarios}
    result["summary_sha256"] = _sha(result)
    return result
