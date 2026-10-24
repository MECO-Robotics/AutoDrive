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
import re
import tempfile
from typing import Any

from .meco_tracks import MECO_TRACKS, PREDICTION_MODES
from .meco_phase4 import PHASE4_FACTORS, phase4_factor_config, phase4_factor_digest
from .meco_phase4_runtime_adapter import (
    PHASE4_ROUTE_FACTOR_ID_BY_KIND, PHASE4_ROUTE_OUTCOME_FACTOR_IDS,
    PHASE4_ROUTE_OUTCOME_POLICY_IDS, phase4_route_completion_campaign_record,
    validate_phase4_capture)
from .meco_phase4_context import (CONTACT_GEOMETRIES,
    phase4_start_profile_assignment as build_start_profile_assignment,
    phase4_context_identity, validate_contact_geometry_split,
    validate_phase4_start_profile_assignment,
    phase4_focal_pose_assignment as build_focal_pose_assignment,
    validate_phase4_focal_pose_assignment,
    phase4_teammate_behavior_assignment as build_teammate_behavior_assignment)
from .meco_device import normalized_device_info
from .training_curriculum import get_scenario


MATRIX_SCHEMA_VERSION = 7
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
    contact_geometry_id: str | None = None
    contact_geometry_config: dict | None = None
    focal_pose_pair_offset: str | None = None
    focal_pose_offset: str | None = None
    teammate_behavior_arm: str | None = None
    teammate_behavior_pool_seed: int | None = None

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
        if self.contact_geometry_id is None:
            if self.contact_geometry_config is not None:
                raise ValueError("contact geometry config requires a geometry ID")
        elif (self.contact_geometry_id not in CONTACT_GEOMETRIES or
              self.contact_geometry_config != phase4_context_identity(
                  "robot_contact_geometry", self.contact_geometry_id)):
            raise ValueError("contact geometry differs from its frozen registry identity")
        if self.focal_pose_pair_offset is None:
            if self.focal_pose_offset is not None:
                raise ValueError("focal pose treatment requires its registered pair identity")
        else:
            phase4_context_identity("focal_pose_offsets", self.focal_pose_pair_offset)
            if self.focal_pose_offset not in (None, self.focal_pose_pair_offset):
                raise ValueError("focal pose treatment must match its registered pair identity")
        if self.teammate_behavior_arm is None:
            if self.teammate_behavior_pool_seed is not None:
                raise ValueError("teammate behavior pool seed requires an assigned arm")
        elif (self.teammate_behavior_arm not in ("no_factor", "behavior_pool") or
              type(self.teammate_behavior_pool_seed) is not int or
              self.teammate_behavior_pool_seed < 0):
            raise ValueError("teammate behavior lane requires a valid arm and pool seed")
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
                               seeds: tuple[int, ...],
                               qualification_case: dict) -> dict[str, dict[str, str]]:
    """Require each requested case seed to belong to its frozen validation/test split."""
    assignments = {}
    for category in categories:
        cases = [case for case in protocol.get("cases", [])
                 if case.get("scenario_id") == scenario_id
                 and case.get("category") == category]
        cases = [case for case in cases if all(
            case.get(field) == value for field, value in qualification_case.items())]
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


def _load_seed_manifest(path: str | Path, *, protocol_id: str,
                        protocol_sha256: str, scenario_id: str,
                        categories: tuple[str, ...], split: str,
                        qualification_case: dict) -> tuple[int, ...]:
    """Load an exact seed list whose protocol, split, and stratum are explicit."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema", "protocol_id", "protocol_sha256", "scenario_id",
                "categories", "split", "qualification_case", "seed_list"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("seed manifest must contain exactly the frozen binding fields")
    expected = {"schema": "meco-exact-seed-manifest-v1",
        "protocol_id": protocol_id, "protocol_sha256": protocol_sha256,
        "scenario_id": scenario_id, "categories": list(categories),
        "split": split, "qualification_case": qualification_case}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("seed manifest protocol, scenario, category, split, or stratum mismatch")
    seeds = payload.get("seed_list")
    if (not isinstance(seeds, list) or not seeds or
            any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds) or
            len(seeds) != len(set(seeds))):
        raise ValueError("seed manifest must contain a nonempty unique signed-63-bit seed list")
    return tuple(seeds)


def _validate_replay_selection(selection, conditions, categories, seeds):
    if not isinstance(selection, dict) or not selection:
        raise ValueError("replay selection must map conditions to policy seed lists")
    names = {condition.condition_name for condition in conditions}
    allowed_policies = set(MECO_TRACKS) | {f"{track}_P0" for track in ("A1", "A2", "A3")} | {"D0", "D1", "D2"} | {
        f"{track}__INFERENCE_DISABLED_TO_D1" for track in MECO_TRACKS}
    normalized = {}
    for condition_name, policy_seeds in selection.items():
        if condition_name not in names or not isinstance(policy_seeds, dict) or not policy_seeds:
            raise ValueError("replay selection references an unknown or empty condition")
        normalized[condition_name] = {}
        for policy, selected_seeds in policy_seeds.items():
            if (policy not in allowed_policies or not isinstance(selected_seeds, list) or
                    not selected_seeds or len(selected_seeds) != len(set(selected_seeds)) or
                    any(type(seed) is not int or seed not in seeds for seed in selected_seeds)):
                raise ValueError("replay selection has an unknown policy or invalid matrix seed")
            normalized[condition_name][policy] = list(selected_seeds)
    return normalized


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


def _seal_final_test_consumption(path: Path, *, matrix_id: str,
                                 protocol: dict, checkpoint_hashes: dict,
                                 source_hash: str, seed_list: tuple[int, ...],
                                 scenario_id: str, categories: tuple[str, ...],
                                 qualification_case: dict) -> None:
    """Reserve the single final-test event before simulator execution begins."""
    payload = {"schema": "meco-sealed-final-test-consumption-v1",
        "state": "consumed_before_execution",
        "matrix_id": matrix_id,
        "matrix_sha256": matrix_id,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "scenario_id": scenario_id, "categories": list(categories),
        "split": "test", "qualification_case": qualification_case,
        "seed_list": list(seed_list),
        "checkpoint_sha256": checkpoint_hashes,
        "source_identity_sha256": source_hash,
        "consumed_utc": datetime.now(timezone.utc).isoformat(),
        "retraining_or_repeat_test_forbidden": True}
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("sealed final test was already consumed for this output campaign") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Keep a partial reservation if anything may have started; it must not
        # be silently retried with exposed final-test seeds.
        raise


def final_test_ledger_path(protocol_path: str | Path, protocol_id: str) -> Path:
    """Stable protocol-scoped release ledger shared by evaluators and trainers."""
    return Path(protocol_path).resolve().with_name(
        f"{protocol_id}.sealed-final-test-consumption.json")


def default_conditions(scenario_id: str, *, categories: tuple[str, ...] | None = None,
                       inference_disable_tick: int | None = None,
                       qualification_case: dict | None = None,
                       contact_geometries: tuple[str, ...] = (),
                       focal_pose_offsets: tuple[str, ...] = (),
                       teammate_behavior_pool_seed: int | None = None
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
    if contact_geometries:
        if (not isinstance(contact_geometries, (tuple, list)) or
                any(not isinstance(geometry, str) for geometry in contact_geometries) or
                len(set(contact_geometries)) != len(contact_geometries) or
                any(geometry not in CONTACT_GEOMETRIES for geometry in contact_geometries)):
            raise ValueError("contact_geometries must be unique registered geometry IDs")
        if not pressure_domain or not supported_categories:
            raise ValueError("contact geometries require registered pressure/match categories")
        conditions += tuple(MatrixCondition(
            f"contact_{geometry_id}", "cv", "cv", 16,
            categories=supported_categories,
            qualification_case=qualification_case,
            contact_geometry_id=geometry_id,
            contact_geometry_config=phase4_context_identity(
                "robot_contact_geometry", geometry_id))
            for geometry_id in contact_geometries)
    if focal_pose_offsets:
        if (scenario_id != "full_match_offense" or
                not isinstance(focal_pose_offsets, (tuple, list)) or
                len(set(focal_pose_offsets)) != len(focal_pose_offsets)):
            raise ValueError("focal pose pairs require unique offsets in full-match validation")
        for offset_name in focal_pose_offsets:
            phase4_context_identity("focal_pose_offsets", offset_name)
            for arm, treatment in (("no_offset", None), ("offset", offset_name)):
                conditions += (MatrixCondition(
                    f"phase4_focal_pose_{offset_name}_{arm}", "cv", "cv", 16,
                    categories=categories, qualification_case=qualification_case,
                    focal_pose_pair_offset=offset_name, focal_pose_offset=treatment),)
    if teammate_behavior_pool_seed is not None:
        if scenario_id != "full_match_offense" or type(teammate_behavior_pool_seed) is not int or teammate_behavior_pool_seed < 0:
            raise ValueError("teammate behavior pools require full_match_offense and a non-negative pool seed")
        for arm, name in (("no_factor", "no_factor"),
                          ("behavior_pool", "behavior_pool")):
            conditions += (MatrixCondition(
                f"phase4_teammate_behaviors_{name}", "cv", "cv", 16,
                categories=categories, qualification_case=qualification_case,
                teammate_behavior_arm=arm,
                teammate_behavior_pool_seed=teammate_behavior_pool_seed),)
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


def _validate_focal_pose_condition_selection(conditions, focal_pose_offsets):
    selected = {}
    for condition in conditions:
        if condition.focal_pose_pair_offset is not None:
            selected.setdefault(condition.focal_pose_pair_offset, set()).add(
                condition.focal_pose_offset)
    if (set(selected) != set(focal_pose_offsets) or
            any(arms != {None, name} for name, arms in selected.items())):
        raise ValueError("condition selection must include both arms of every focal-pose pair")


def _validate_teammate_behavior_condition_selection(conditions, pool_seed):
    selected = {condition.teammate_behavior_arm for condition in conditions
                if condition.teammate_behavior_arm is not None}
    if pool_seed is None:
        if selected:
            raise ValueError("teammate behavior lanes require their explicit pool seed")
    elif selected != {"no_factor", "behavior_pool"}:
        raise ValueError("condition selection must include both teammate behavior pair arms")


def _matrix_condition_selection_mode(conditions, condition_names):
    """Validate supported diagnostic selections and choose a contrast path."""
    if condition_names is None:
        return "full"
    names = tuple(condition_names)
    if names == ("architecture_matched_cv_k16",):
        return "architecture_only"
    if len(names) == 2:
        selected = {condition.condition_name: condition for condition in conditions}
        if set(names) == set(selected) and all(
                selected[name].focal_pose_pair_offset is not None for name in names):
            offsets = {selected[name].focal_pose_pair_offset for name in names}
            arms = {selected[name].focal_pose_offset for name in names}
            if len(offsets) == 1 and arms == {None, next(iter(offsets))}:
                return "paired_context_only"
        if set(names) == set(selected) and all(
                selected[name].teammate_behavior_arm is not None for name in names):
            arms = {selected[name].teammate_behavior_arm for name in names}
            seeds = {selected[name].teammate_behavior_pool_seed for name in names}
            if arms == {"no_factor", "behavior_pool"} and len(seeds) == 1:
                return "paired_context_only"
    raise ValueError("diagnostic condition selection must be architecture-only or one complete paired context factor")


def _validate_matrix_contrast_inputs(selection_mode, evaluation_reports):
    """Require baseline contrast inputs only for selections that use them."""
    if selection_mode == "paired_context_only":
        return
    if "architecture_matched_cv_k16" not in evaluation_reports:
        raise ValueError("matrix contrast selection lacks architecture_matched_cv_k16")


def _records_by_key(report: dict) -> dict[tuple[str, str, int], dict]:
    result = {}
    for row in report.get("records", ()):
        key = (row.get("policy"), row.get("category"), row.get("seed"))
        if key in result:
            raise ValueError(f"duplicate policy/category/seed record: {key}")
        result[key] = row
    return result


def _expected_contact_policy_ids(policy_manifest: dict) -> dict[str, str]:
    """List every policy row evaluated by a contact matrix condition."""
    if not isinstance(policy_manifest, dict):
        raise ValueError("contact evaluation report lacks a policy manifest")
    policy_names = (*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2")
    expected = {}
    for policy_name in policy_names:
        manifest = policy_manifest.get(policy_name)
        policy_id = manifest.get("policy_id") if isinstance(manifest, dict) else None
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError(
                f"contact evaluation report lacks expected policy ID: {policy_name}")
        expected[policy_name] = policy_id
    return expected


def _validate_phase4_condition_records(records: list[dict], *, factor: dict,
                                       source_hash: str, split: str,
                                       qualification_case: dict) -> None:
    """Require every treatment episode to carry a realized, exact capture."""
    for row in records:
        capture = row.get("phase4_realization_capture")
        try:
            validate_phase4_capture(capture,
                expected_source_identity_sha256=source_hash,
                expected_episode_seed=row.get("seed"),
                expected_split=split,
                expected_qualification_case=qualification_case)
        except ValueError as exc:
            raise ValueError(
                "Phase 4 episode has an invalid source-bound realization capture: "
                f"{exc}") from exc
        if (capture.get("factor") != factor or
                capture.get("scenario_id") != row.get("scenario_id") or
                capture.get("category") != row.get("category") or
                capture.get("seed") != (None if split == "test" else row.get("seed"))):
            raise ValueError("Phase 4 capture episode identity differs from its record")
        row_case = {name: row.get(name) for name in DEFAULT_EVALUATION_STRATUM}
        if capture.get("qualification_case") != row_case:
            raise ValueError("Phase 4 capture qualification case differs from its record")


def _validate_contact_condition_records(records: list[dict], *, geometry_id: str,
                                        geometry_config: dict,
                                        expected_policy_ids: dict[str, str],
                                        expected_category_seed_keys: set[tuple[str, int]]) -> dict:
    """Require reset exposure and solver event provenance; never infer qualification."""
    if not records:
        raise ValueError("contact condition has no episode records")
    if (not isinstance(expected_policy_ids, dict) or not expected_policy_ids or
            any(not isinstance(policy, str) or not policy or
                not isinstance(policy_id, str) or not policy_id
                for policy, policy_id in expected_policy_ids.items())):
        raise ValueError("contact condition requires explicit expected policy IDs")
    if (not isinstance(expected_category_seed_keys, set) or
            not expected_category_seed_keys or any(
                not isinstance(key, tuple) or len(key) != 2 or
                not isinstance(key[0], str) or type(key[1]) is not int
                for key in expected_category_seed_keys)):
        raise ValueError("contact condition requires explicit category/seed keys")
    paired_rows: dict[tuple[str, int], dict[str, dict]] = {}
    observed_keys = set()
    for row in records:
        exposure = row.get("phase4_contact_exposure")
        event = row.get("phase4_pair_contact_event")
        if (not isinstance(exposure, dict) or
                exposure.get("geometry_id") != geometry_id or
                exposure.get("geometry_config") != geometry_config or
                exposure.get("exposure_assigned") is not True or
                exposure.get("pair_slots") != [2, 3] or
                exposure.get("opponent_action") != 5 or
                exposure.get("opponent_action_source") != "fresh_sensed_track_action_5" or
                exposure.get("slot2_env_control_mode") != "nn" or
                exposure.get("route_seen") not in (True, False) or
                (exposure.get("route_seen") is True and
                 (not isinstance(exposure.get("route_seen_time_s"), (int, float)) or
                  not math.isfinite(exposure["route_seen_time_s"]))) or
                (exposure.get("route_seen") is False and
                 exposure.get("route_seen_time_s") is not None) or
                type(exposure.get("approach_action_ticks")) is not int or
                exposure["approach_action_ticks"] < 0 or
                exposure.get("focal_action_overridden") is not False or
                exposure.get("qualification_eligible") is not False or
                exposure.get("recovery_time_after_bump_s") is not None or
                exposure.get("route_completion_rate") is not None):
            raise ValueError("contact condition lacks exact assigned exposure metadata")
        if (not isinstance(event, dict) or
                event.get("event_source") != "simulator_robot_pair_sat_contact_solver" or
                event.get("pair_slots") != [2, 3] or
                event.get("qualification_eligible") is not False):
            raise ValueError("contact condition lacks authoritative pair-contact event metadata")
        category, seed, policy = row.get("category"), row.get("seed"), row.get("policy")
        if (not isinstance(category, str) or type(seed) is not int or
                policy not in expected_policy_ids):
            raise ValueError("contact condition record has an unexpected category/seed/policy key")
        if row.get("policy_id") != expected_policy_ids[policy]:
            raise ValueError("contact condition record policy ID differs from expected manifest")
        pair_key = (category, seed)
        if pair_key not in expected_category_seed_keys:
            raise ValueError("contact condition record has an unexpected category/seed key")
        key = (policy, category, seed)
        if key in observed_keys:
            raise ValueError("contact condition has duplicate cross-policy exposure rows")
        observed_keys.add(key)
        paired_rows.setdefault(pair_key, {})[policy] = exposure
    expected_keys = {(policy, category, seed)
        for policy in expected_policy_ids for category, seed in expected_category_seed_keys}
    missing, extra = expected_keys - observed_keys, observed_keys - expected_keys
    if missing or extra:
        raise ValueError("contact exposure rows do not cover expected policy/category/seed keys: "
            f"missing={sorted(missing)}, extra={sorted(extra)}")
    signatures = set()
    action_ticks = []
    for pair_key in expected_category_seed_keys:
        group = paired_rows.get(pair_key, {})
        if set(group) != set(expected_policy_ids):
            raise ValueError("contact exposure rows are not cross-policy paired")
        for exposure in group.values():
            signatures.add((exposure["geometry_id"],
                json.dumps(exposure["geometry_config"], sort_keys=True),
                tuple(exposure["pair_slots"]), exposure["opponent_action"],
                exposure["opponent_action_source"],
                exposure["slot2_env_control_mode"],
                exposure["focal_action_overridden"]))
            action_ticks.append(exposure["approach_action_ticks"])
    if len(signatures) != 1:
        raise ValueError("paired contact rows do not share one exposure assignment")
    return {"paired_by_category_seed": True,
        "policy_set": sorted(expected_policy_ids),
        "expected_category_seed_keys": [list(key) for key in sorted(expected_category_seed_keys)],
        "approach_action_tick_counts_match": len(set(action_ticks)) == 1}


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
               categories: tuple[str, ...], seeds: tuple[int, ...] | None,
               device: str, output_dir: str | Path,
               inference_disable_tick: int | None = None,
               qualification_protocol: str | Path | None = None,
               qualification_case: dict | None = None,
               replay_output_dir: str | Path | None = None,
               replay_seeds: tuple[int, ...] | None = None,
               seed_manifest: str | Path | None = None,
               replay_selection: dict | None = None,
               contact_geometries: tuple[str, ...] = (),
               phase4_opponent_schedule: dict | None = None,
               start_profile_registry_id: str | None = None,
               focal_pose_offsets: tuple[str, ...] = (),
               teammate_behavior_pool_seed: int | None = None,
               condition_names: tuple[str, ...] | None = None) -> dict:
    """Execute all fixed comparisons on exact shared checkpoint/seed bundles."""
    from .curriculum_training import source_identity
    from .meco_track_training import evaluate_meco, load_evaluation_protocol

    if set(checkpoints) != set(MECO_TRACKS):
        raise ValueError(f"checkpoint keys must be exactly {MECO_TRACKS}")
    if not isinstance(device, str) or re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", device) is None:
        raise ValueError("matched MECO evaluation requires an explicit indexed CUDA/HIP device; CPU is unsupported")
    seed_manifest_payload = None
    if seed_manifest is not None:
        seed_manifest_payload = json.loads(Path(seed_manifest).read_text(encoding="utf-8"))
        manifest_seeds = seed_manifest_payload.get("seed_list") if isinstance(seed_manifest_payload, dict) else None
        if seeds is None:
            seeds = tuple(manifest_seeds or ())
        elif tuple(manifest_seeds or ()) != tuple(seeds):
            raise ValueError("explicit matrix seeds differ from seed manifest")
    if not seeds or len(set(seeds)) != len(seeds) or any(
            type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds):
        raise ValueError("seeds must be a nonempty unique list of signed-63-bit integers")
    qualification_case = dict(DEFAULT_EVALUATION_STRATUM if qualification_case is None
                              else qualification_case)
    if (set(qualification_case) != set(DEFAULT_EVALUATION_STRATUM) or
            any(not isinstance(value, str) or not value for value in qualification_case.values())):
        raise ValueError("qualification_case must contain nonempty layout, defender, start, and sensor identifiers")
    if replay_selection is not None and replay_seeds is not None:
        raise ValueError("use condition/policy replay_selection or legacy replay_seeds, not both")
    has_replay_selector = replay_seeds is not None or replay_selection is not None
    if (replay_output_dir is None) != (not has_replay_selector):
        raise ValueError("replay_output_dir and explicit replay_seeds must be supplied together")
    if replay_seeds is not None and (not replay_seeds or len(set(replay_seeds)) != len(replay_seeds)
            or any(type(seed) is not int or seed not in seeds for seed in replay_seeds)):
        raise ValueError("replay_seeds must be a nonempty unique subset of matrix seeds")
    if not categories:
        raise ValueError("at least one scenario category is required")
    definition = get_scenario(scenario_id)
    if any(category not in definition.required_categories for category in categories):
        raise ValueError("category does not belong to the selected scenario")
    qualification_provenance = None
    if qualification_protocol is None:
        if seed_manifest_payload is not None:
            raise ValueError("a seed manifest requires its frozen qualification protocol")
        seed_split_assignments = None
        evaluation_design = "diagnostic_only_unbound_seeds"
        evaluation_split = "diagnostic"
    else:
        protocol, protocol_id, protocol_sha256, protocol_file = load_evaluation_protocol(
            Path(qualification_protocol), scenario_id=scenario_id)
        seed_split_assignments = _held_out_seed_assignments(
            protocol, scenario_id, categories, seeds, qualification_case)
        splits = {split for assignments in seed_split_assignments.values()
                  for split in assignments.values()}
        if len(splits) != 1:
            raise ValueError(
                "one matrix invocation must use a single frozen split; run validation "
                "and test seeds in separate matrices")
        evaluation_split = next(iter(splits))
        if seed_manifest_payload is not None:
            bound_seeds = _load_seed_manifest(seed_manifest,
                protocol_id=protocol_id, protocol_sha256=protocol_sha256,
                scenario_id=scenario_id, categories=categories,
                split=evaluation_split, qualification_case=qualification_case)
            if bound_seeds != tuple(seeds):
                raise ValueError("seed manifest exact seeds differ from matrix seed list")
        evaluation_design = "frozen_protocol_held_out"
        qualification_provenance = {"protocol_id": protocol_id,
            "protocol_sha256": protocol_sha256,
            "protocol_file": protocol_file}
    if focal_pose_offsets and (qualification_provenance is None or evaluation_split != "validation"):
        raise ValueError("focal pose pairs require a frozen full-match validation protocol")
    if teammate_behavior_pool_seed is not None and (
            qualification_provenance is None or evaluation_split != "validation" or
            scenario_id != "full_match_offense"):
        raise ValueError("teammate behavior pool requires a frozen full-match validation protocol")
    if phase4_opponent_schedule is not None:
        from .meco_phase4_opponents import validate_opponent_policy_schedule
        phase4_opponent_schedule = validate_opponent_policy_schedule(
            phase4_opponent_schedule, scenario_id=scenario_id,
            categories=categories, qualification_case=qualification_case,
            seeds=seeds, split=evaluation_split,
            factor_id=phase4_opponent_schedule.get("factor_id"),
            protocol_id=(qualification_provenance["protocol_id"]
                         if qualification_provenance is not None else None))
    start_profile_assignment = None
    if start_profile_registry_id is not None:
        if qualification_provenance is None:
            raise ValueError("full-team start profile requires a frozen qualification protocol")
        start_profile_assignment = build_start_profile_assignment(
            start_profile_registry_id, scenario_id=scenario_id,
            split=evaluation_split,
            protocol_id=qualification_provenance["protocol_id"],
            protocol_sha256=qualification_provenance["protocol_sha256"],
            qualification_case=qualification_case, categories=categories,
            seeds=seeds)
        validate_phase4_start_profile_assignment(start_profile_assignment)
        if contact_geometries:
            raise ValueError("full-team start profiles cannot be combined with contact geometry resets")
    output_root = Path(output_dir)
    sealed_test_ledger = (final_test_ledger_path(qualification_protocol,
        qualification_provenance["protocol_id"])
        if qualification_provenance is not None else
        output_root / "sealed-final-test-consumption.json")
    if sealed_test_ledger.exists():
        raise ValueError("this matrix/protocol is sealed after final-test consumption")
    # HIP preparation changes the native-extension component of source_identity;
    # capture source provenance only after preparing the exact evaluation runtime.
    from .fuel_physics_hip import prepare_native_fuel_extension
    prepare_native_fuel_extension(device)
    execution_device = normalized_device_info(device)
    bundle_hashes = _checkpoint_hashes(checkpoints)
    source_hash = _source_identity_hash(source_identity())
    config_hashes, protocol_ids, frozen_shared_config = (
        _checkpoint_configuration_provenance(checkpoints))
    conditions = default_conditions(scenario_id, categories=categories,
        inference_disable_tick=inference_disable_tick,
        qualification_case=qualification_case,
        contact_geometries=contact_geometries,
        focal_pose_offsets=focal_pose_offsets,
        teammate_behavior_pool_seed=teammate_behavior_pool_seed)
    if condition_names is not None:
        if (not condition_names or len(set(condition_names)) != len(condition_names)
                or any(not isinstance(name, str) or not name for name in condition_names)):
            raise ValueError("condition_names must be a nonempty unique condition selection")
        available = {condition.condition_name: condition for condition in conditions}
        if any(name not in available for name in condition_names):
            raise ValueError("condition_names contains an unknown matrix condition")
        conditions = tuple(available[name] for name in condition_names)
        _validate_focal_pose_condition_selection(conditions, focal_pose_offsets)
        _validate_teammate_behavior_condition_selection(
            conditions, teammate_behavior_pool_seed)
    selection_mode = _matrix_condition_selection_mode(conditions, condition_names)
    if replay_seeds is not None:
        # Backward compatible seed-only requests now capture once, for the
        # primary architecture condition, instead of duplicating every replay.
        replay_selection = {"architecture_matched_cv_k16": {
            policy: list(replay_seeds) for policy in (*MECO_TRACKS, "D0", "D1", "D2")}}
    if replay_selection is not None:
        replay_selection = _validate_replay_selection(
            replay_selection, conditions, categories, seeds)
    qualification_case = dict(qualification_case)
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
        "seed_manifest": (None if seed_manifest_payload is None else {
            "path": str(Path(seed_manifest).resolve()),
            "sha256": hashlib.sha256(Path(seed_manifest).read_bytes()).hexdigest(),
            "schema": seed_manifest_payload.get("schema")}),
        "replay_selection": replay_selection or {},
        "evaluation_design": evaluation_design,
        "evaluation_split": evaluation_split,
        "qualification_protocol_id": (None if qualification_provenance is None else
            qualification_provenance["protocol_id"]),
        "qualification_protocol": qualification_provenance,
        "qualification_eligible": False,
        "seed_split_assignments": seed_split_assignments,
        "source_identity_sha256": source_hash,
        "execution_device": execution_device,
        "qualification_case": qualification_case,
        "phase4_opponent_schedule": phase4_opponent_schedule,
        "phase4_start_profile_assignment": start_profile_assignment,
        "training_configuration_sha256": config_hashes,
        "training_protocol_ids": protocol_ids,
        "shared_training_configuration": frozen_shared_config,
        "conditions": condition_specs,
        "condition_selection": (None if condition_names is None else list(condition_names)),
        "contact_geometry_ids": [condition.contact_geometry_id
            for condition in conditions if condition.contact_geometry_id is not None],
        "phase4_factor_schema": ("meco-phase4-five-level-pressure-v2" if phase4_enabled else None),
        "phase4_unsupported_categories": unsupported_phase4,
        "unsupported_conditions_open": ([] if phase4_enabled else [
            "explicit_feint_behavior", "sensor_delay_ablation",
            "separate_low_confidence_profile", "abrupt_opponent_policy_change"])}
    matrix_id = _canonical_hash(matrix_protocol)
    matrix_protocol["matrix_id"] = matrix_id
    if sealed_test_ledger.exists():
        raise ValueError("this matrix/protocol is sealed after final-test consumption")
    if evaluation_split == "test":
        for condition in conditions:
            validate_contact_geometry_split(condition.contact_geometry_id, evaluation_split)
        if qualification_provenance is None:
            raise ValueError("final test requires frozen qualification protocol provenance")
        _seal_final_test_consumption(sealed_test_ledger, matrix_id=matrix_id,
            protocol=qualification_provenance, checkpoint_hashes=bundle_hashes,
            source_hash=source_hash, seed_list=seeds, scenario_id=scenario_id,
            categories=categories, qualification_case=qualification_case)
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
        condition_start_assignment = (None if start_profile_assignment is None else
            build_start_profile_assignment(start_profile_registry_id,
                scenario_id=scenario_id, split=evaluation_split,
                protocol_id=qualification_provenance["protocol_id"],
                protocol_sha256=qualification_provenance["protocol_sha256"],
                qualification_case=(condition.qualification_case or qualification_case),
                categories=condition_categories, seeds=seeds))
        if condition.focal_pose_pair_offset is not None and (
                qualification_provenance is None or evaluation_split != "validation"):
            raise ValueError("focal pose pairs require a frozen validation protocol")
        condition_focal_pose_assignment = (None if condition.focal_pose_pair_offset is None else
            build_focal_pose_assignment(condition.focal_pose_offset,
                paired_offset_name=condition.focal_pose_pair_offset,
                scenario_id=scenario_id, split=evaluation_split,
                protocol_id=qualification_provenance["protocol_id"],
                protocol_sha256=qualification_provenance["protocol_sha256"],
                qualification_case=(condition.qualification_case or qualification_case),
                categories=condition_categories, seeds=seeds))
        condition_teammate_behavior_assignment = (
            None if condition.teammate_behavior_arm is None else
            build_teammate_behavior_assignment(
                arm=condition.teammate_behavior_arm, scenario_id=scenario_id,
                split=evaluation_split,
                protocol_id=qualification_provenance["protocol_id"],
                protocol_sha256=qualification_provenance["protocol_sha256"],
                qualification_case=(condition.qualification_case or qualification_case),
                categories=condition_categories, seeds=seeds,
                pool_seed=condition.teammate_behavior_pool_seed))
        condition_replay_selection = (replay_selection or {}).get(
            condition.condition_name)
        route_factor_id_by_kind = {
            "stationary_blockage": "blockage.stationary",
            "feinting_opponent": "opponent.feint_reversal",
            "abrupt_opponent_policy_change": "opponent.abrupt_policy_change",
        }
        if condition.phase4_factor is not None:
            route_factor_id = route_factor_id_by_kind.get(
                condition.phase4_factor.get("kind"))
            route_outcome_factor_ids = ((route_factor_id,)
                if route_factor_id is not None else None)
        elif condition.contact_geometry_id is not None:
            route_outcome_factor_ids = ("contact.robot_bump",)
        elif condition.condition_name == "architecture_matched_cv_k16":
            # The no-factor baseline is shared by the route-requiring Phase 4
            # contrasts and must carry the same policy/case/seed bindings.
            route_outcome_factor_ids = PHASE4_ROUTE_OUTCOME_FACTOR_IDS
        else:
            route_outcome_factor_ids = None
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
            phase4_opponent_schedule=phase4_opponent_schedule,
            phase4_contact_factor_id=("contact.robot_bump"
                if condition.contact_geometry_id is not None else None),
            phase4_route_outcome_factor_ids=route_outcome_factor_ids,
            contact_geometry_id=condition.contact_geometry_id,
            contact_geometry_config=condition.contact_geometry_config,
            phase4_start_profile_assignment=condition_start_assignment,
            phase4_focal_pose_assignment=condition_focal_pose_assignment,
            phase4_teammate_behavior_assignment=condition_teammate_behavior_assignment,
            replay_output_dir=(replay_output_dir if condition_replay_selection else None),
            replay_seeds=None,
            replay_selection=condition_replay_selection,
            condition_id=condition_id,
            output=str(report_path))
        if report.get("phase4_factor") != condition.phase4_factor:
            raise ValueError("evaluation report Phase 4 factor differs from the frozen condition")
        if report.get("phase4_start_profile_assignment") != condition_start_assignment:
            raise ValueError("evaluation report start profile differs from the frozen matrix assignment")
        if report.get("phase4_focal_pose_assignment") != condition_focal_pose_assignment:
            raise ValueError("evaluation report focal pose differs from the frozen matrix assignment")
        if report.get("phase4_teammate_behavior_assignment") != condition_teammate_behavior_assignment:
            raise ValueError("evaluation report teammate behavior differs from the frozen matrix assignment")
        if (report.get("contact_geometry_id") != condition.contact_geometry_id or
                report.get("contact_geometry_config") != condition.contact_geometry_config):
            raise ValueError("evaluation report contact geometry differs from frozen condition")
        if condition.contact_geometry_id is not None:
            policy_manifest = report.get("policy_manifest", {})
            expected_policy_ids = _expected_contact_policy_ids(policy_manifest)
            if (report.get("categories") != list(condition_categories) or
                    report.get("seeds") != list(seeds)):
                raise ValueError("contact evaluation report category/seed coverage differs from matrix")
            expected_category_seed_keys = {
                (category, seed) for category in condition_categories for seed in seeds}
            contact_pairing = _validate_contact_condition_records(report.get("records", ()),
                geometry_id=condition.contact_geometry_id,
                geometry_config=condition.contact_geometry_config,
                expected_policy_ids=expected_policy_ids,
                expected_category_seed_keys=expected_category_seed_keys)
        else:
            contact_pairing = None
        if condition.phase4_factor is not None:
            _validate_phase4_condition_records(report.get("records", ()),
                factor=condition.phase4_factor, source_hash=source_hash,
                split=evaluation_split, qualification_case=qualification_case)
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
        report["experiment_matrix"]["replay_selection"] = condition_replay_selection or {}
        report["experiment_matrix"]["qualification_eligible"] = False
        report["experiment_matrix"]["contact_exposure_pairing"] = contact_pairing
        _atomic_json(report_path, report)
        report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
        condition_records = _records_by_key(report)
        results[condition.condition_name] = {"condition_id": condition_id,
            "evaluation_id": report["evaluation_id"],
            "dashboard_replay_count": len(report.get("dashboard_replays", [])),
            "replay_selection": condition_replay_selection or {},
            "protocol_id": report["protocol_id"],
            "qualification_protocol_id": report["qualification_protocol_id"],
            "evaluation_design": evaluation_design,
            "evaluation_split": evaluation_split,
            "qualification_protocol": qualification_provenance,
            "seed_split_assignments": seed_split_assignments,
            "report_path": str(report_path),
            "evaluation_report_sha256": report_sha256,
            "checkpoint_sha256": bundle_hashes,
            "source_identity_sha256": source_hash,
            "execution_device": execution_device,
            "qualification_case": report["qualification_case"],
            "policy_manifest": report["policy_manifest"],
            "effective_candidate_k": report["candidate_k"],
            "prediction": report["prediction"],
            "phase4_factor": condition.phase4_factor,
            "phase4_teammate_behavior_assignment": condition_teammate_behavior_assignment,
            "phase4_start_profile_assignment": condition_start_assignment,
            "phase4_focal_pose_assignment": condition_focal_pose_assignment,
            "contact_geometry_id": condition.contact_geometry_id,
            "contact_geometry_config": condition.contact_geometry_config,
            "contact_exposure_pairing": contact_pairing,
            "qualification_eligible": False,
            "fault": report["inference_disable_fault"]}
        evaluation_reports[condition.condition_name] = {
            "report": report, "records": condition_records,
            "condition_id": condition_id}

    route_expected_by_key = {}
    route_attempt_rows = []
    route_records_missing = []
    route_policy_support_statuses = []
    route_summary_keys_seen = set()
    # Freeze the planned route-outcome grid from the matrix design itself. This
    # remains complete even when a report record or its nested ledger is absent.
    for condition in conditions:
        evaluation = evaluation_reports.get(condition.condition_name)
        if evaluation is None:
            continue
        report = evaluation["report"]
        phase4_factor = condition.phase4_factor
        if phase4_factor is not None:
            factor_id = PHASE4_ROUTE_FACTOR_ID_BY_KIND.get(phase4_factor.get("kind"))
            factor_conditions = ((factor_id, "factor_treatment"),) if factor_id else ()
        elif condition.contact_geometry_id is not None:
            factor_conditions = (("contact.robot_bump", "factor_treatment"),)
        elif condition.condition_name == "architecture_matched_cv_k16":
            factor_conditions = tuple((factor_id, "no_factor")
                for factor_id in PHASE4_ROUTE_OUTCOME_FACTOR_IDS)
        else:
            factor_conditions = ()
        qual_case = report.get("qualification_case")
        eval_case = (qual_case if evaluation_split in ("validation", "test") else None)
        policy_manifest = report.get("policy_manifest", {})
        device_identity = json.dumps(report.get("execution_device"),
            sort_keys=True, separators=(",", ":"), allow_nan=False)
        for category in report.get("categories", ()):
            if (any(condition_label == "no_factor" for _, condition_label in factor_conditions)
                    and _registered_pressure_level(category) not in (1, 2, 3, 4, 5)):
                continue
            case_id = hashlib.sha256(json.dumps({"evaluation_case": eval_case,
                "qualification_case": qual_case, "category": category},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            for seed in report.get("seeds", ()):
                for policy_id in PHASE4_ROUTE_OUTCOME_POLICY_IDS:
                    manifest = policy_manifest.get(policy_id, {})
                    for factor_id, condition_label in factor_conditions:
                        assignment = {"policy_id": policy_id,
                            "factor_id": factor_id, "case_id": case_id,
                            "seed": int(seed), "condition": condition_label,
                            "source_identity_sha256": report.get("source_identity_sha256"),
                            "episode_id": None, "provenance_status": "record_missing",
                            "scenario_id": report.get("scenario_id"),
                            "category": category, "execution_device": device_identity,
                            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                            "condition_id": evaluation["condition_id"]}
                        key = tuple(assignment[name] for name in (
                            "policy_id", "factor_id", "case_id", "seed", "condition"))
                        if key in route_expected_by_key:
                            raise ValueError("frozen route outcome design contains duplicate assignment")
                        route_expected_by_key[key] = assignment
    for condition_name, evaluation in evaluation_reports.items():
        for record in evaluation["report"].get("records", ()):
            for planned in route_expected_by_key.values():
                if (planned.get("condition_id") == evaluation["condition_id"] and
                        planned.get("policy_id") == record.get("policy") and
                        planned.get("category") == record.get("category") and
                        planned.get("seed") == record.get("seed") and
                        isinstance(record.get("episode_id"), str) and
                        record["episode_id"]):
                    planned["episode_id"] = record["episode_id"]
                    planned.pop("provenance_status", None)
            by_factor = record.get("phase4_route_completion_by_factor")
            if not isinstance(by_factor, dict):
                planned = [row for row in route_expected_by_key.values()
                    if row.get("condition_id") == evaluation["condition_id"] and
                    row.get("policy_id") == record.get("policy") and
                    row.get("category") == record.get("category") and
                    row.get("seed") == record.get("seed")]
                for row in planned:
                    route_records_missing.append({"condition_name": condition_name,
                        "policy": row["policy_id"], "category": row["category"],
                        "seed": row["seed"], "factor_id": row["factor_id"],
                        "condition": row["condition"],
                        "reason": "phase4_route_completion_by_factor_missing_or_malformed"})
                continue
            for factor_id, summary in by_factor.items():
                if not isinstance(summary, dict) or not isinstance(summary.get("binding"), dict):
                    route_records_missing.append({"condition_name": condition_name,
                        "policy": record.get("policy"), "category": record.get("category"),
                        "seed": record.get("seed"), "factor_id": factor_id,
                        "reason": "route_lifecycle_record_or_binding_missing"})
                    continue
                bound = summary["binding"]
                key = tuple(bound.get(name) for name in (
                    "policy_id", "factor_id", "case_id", "seed", "condition"))
                expected_assignment = route_expected_by_key.get(key)
                if expected_assignment is None:
                    route_records_missing.append({"condition_name": condition_name,
                        "policy": record.get("policy"), "category": record.get("category"),
                        "seed": record.get("seed"), "factor_id": factor_id,
                        "reason": "observed_route_summary_outside_frozen_matrix_design"})
                    continue
                expected_assignment["episode_id"] = bound.get("episode_id")
                expected_assignment.pop("provenance_status", None)
                route_summary_keys_seen.add(key)
                for provenance_key in ("source_identity_sha256", "scenario_id",
                        "category", "execution_device", "checkpoint_sha256", "condition_id"):
                    if bound.get(provenance_key) != expected_assignment.get(provenance_key):
                        route_records_missing.append({"condition_name": condition_name,
                            "policy": record.get("policy"), "category": record.get("category"),
                            "seed": record.get("seed"), "factor_id": factor_id,
                            "reason": f"route_summary_{provenance_key}_mismatch"})
                route_policy_support_statuses.append({
                    "policy_id": bound.get("policy_id"),
                    "factor_id": factor_id,
                    "condition_name": condition_name,
                    "status": summary.get("policy_support_status"),
                    "ledger_input_status": summary.get("ledger_input_status"),
                    "attempted": summary.get("attempted"),
                    "coverage_available": summary.get(
                        "goal_attempt_coverage_available"),
                })
                for attempt in summary.get("attempts", ()):
                    route_attempt_rows.append({**attempt, **bound})
    for key, assignment in list(route_expected_by_key.items()):
        if assignment.get("condition_id", "").startswith("missing:"):
            route_policy_support_statuses.append({
                "policy_id": assignment["policy_id"],
                "factor_id": assignment["factor_id"],
                "condition": assignment["condition"],
                "status": "factor_treatment_not_scheduled",
                "ledger_input_status": "not_applicable",
                "attempted": 0, "coverage_available": False})
            continue
        if key not in route_summary_keys_seen:
            route_policy_support_statuses.append({
                "policy_id": assignment["policy_id"],
                "factor_id": assignment["factor_id"],
                "condition": assignment["condition"],
                "status": "route_lifecycle_summary_missing",
                "ledger_input_status": "missing_record",
                "attempted": 0, "coverage_available": False})
            route_records_missing.append({"policy": assignment["policy_id"],
                "category": assignment["category"], "seed": assignment["seed"],
                "factor_id": assignment["factor_id"],
                "condition": assignment["condition"],
                "reason": "route_lifecycle_summary_missing_from_frozen_assignment"})
    # Missing treatment lanes remain explicit expected assignments. In
    # particular, contact.robot_bump is opt-in; absent geometry is never
    # mistaken for completed paired coverage.
    for baseline in list(route_expected_by_key.values()):
        if baseline["condition"] != "no_factor":
            continue
        treatment = {**baseline, "condition": "factor_treatment",
            "condition_id": f"missing:{baseline['factor_id']}:treatment_not_scheduled",
            "episode_id": None,
            "provenance_status": "treatment_not_scheduled"}
        key = tuple(treatment[name] for name in (
            "policy_id", "factor_id", "case_id", "seed", "condition"))
        if key not in route_expected_by_key:
            route_expected_by_key[key] = treatment
            route_records_missing.append({"policy": treatment["policy_id"],
                "category": treatment["category"], "seed": treatment["seed"],
                "factor_id": treatment["factor_id"],
                "condition": "factor_treatment",
                "reason": "factor_treatment_not_scheduled"})
    route_expected_assignments = list(route_expected_by_key.values())
    route_completion_campaign = None
    if route_expected_assignments:
        try:
            route_completion_campaign = phase4_route_completion_campaign_record(
                route_attempt_rows, route_expected_assignments)
        except ValueError as exc:
            route_completion_campaign = {"schema": "meco-phase4-route-completion-campaign-diagnostic-v1",
                "route_completion_rate": None,
                "status": "paired_design_or_provenance_incomplete",
                "reason": str(exc),
                "missing_route_records": route_records_missing,
                "expected_assignment_count": len(route_expected_assignments),
                "authoritative_signal_available": False,
                "qualification_eligible": False}
    else:
        route_completion_campaign = {"schema": "meco-phase4-route-completion-campaign-diagnostic-v1",
            "route_completion_rate": None, "status": "route_lifecycle_records_missing",
            "missing_route_records": route_records_missing,
            "authoritative_signal_available": False, "qualification_eligible": False}

    architecture_name = "architecture_matched_cv_k16"
    _validate_matrix_contrast_inputs(selection_mode, evaluation_reports)
    if selection_mode == "architecture_only":
        contrasts = []
        architecture = evaluation_reports[architecture_name]
        for track in MECO_TRACKS:
            contrasts.append(_architecture_track_vs_d0(
                track=track, condition_id=architecture["condition_id"],
                report=architecture["report"], records=architecture["records"],
                categories=categories, seeds=seeds))
    elif selection_mode == "paired_context_only":
        # The matrix contains only the exact no-factor/treatment pair. Pairing
        # evidence is retained in condition reports; outcome contrasts remain
        # unavailable until a validated summary consumes both records.
        contrasts = []
    else:
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
                            contrasts[-1]["realization_capture"] = treated.get(
                                "phase4_realization_capture")

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
        "qualification_eligible": False,
        "contact_geometry_ids": [condition.contact_geometry_id
            for condition in conditions if condition.contact_geometry_id is not None],
        "phase4_start_profile_assignment": start_profile_assignment,
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
        "qualification_eligible": False,
        "seed_split_assignments": seed_split_assignments,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity_sha256": source_hash,
        "execution_device": execution_device,
        "phase4_start_profile_assignment": start_profile_assignment,
        "dashboard_replay_capture": ({"enabled": replay_output_dir is not None,
            "output_dir": str(Path(replay_output_dir).resolve()),
            "selection": replay_selection or {}}
            if replay_output_dir is not None else {"enabled": False}),
        "checkpoint_sha256": bundle_hashes,
        "training_configuration_sha256": {
            track: config_hashes[track] for track in MECO_TRACKS},
        "training_protocol_ids": {
            track: protocol_ids[track] for track in MECO_TRACKS},
        "conditions": results,
        "phase4_route_completion_campaign": route_completion_campaign,
        "phase4_route_completion_policy_support": route_policy_support_statuses,
        "contrasts": contrasts,
        "phase4_unsupported_categories": unsupported_phase4,
        "interpretation_limits": [
            "Paired rows are reported without confidence intervals; use the calibrated qualification gate for promotion.",
            "Candidate recall is relative to the declared bounded route-qualified reference pool, not a simulator-optimality oracle.",
            "P=0 is an inference-time ablation using the same learned A checkpoint.",
            "A1/A2/A3/C1/C2 versus D0 rows are architecture-condition outcomes paired on exact category/seed; learned tracks use CV while D0 retains its production deterministic behavior, so these are not prediction-mode-matched contrasts.",
            "Fault rows hand off at the first 12-step macro boundary at or after the requested tick.",
            ("Each Phase 4 factor replaces only registered slot-0 pressure behavior and is paired with the no-factor row at the same category and seed; unsupported categories are listed explicitly. Metrics remain diagnostic until held-out evaluation and calibrated gates." if phase4_enabled else
             "Feint, sensor-delay, separate low-confidence, and abrupt policy-change conditions require registered level_1 through level_5 pressure categories; level_0 has no scripted pressure slot."),
            ("Contact geometry rows are diagnostic only, paired on category/seed and exact reset/action assignment; realized contact is reported separately from exposure. Recovery and route-completion outcomes are unavailable, and qualification eligibility remains false." if contact_geometries else
             "Contact geometry rows are opt-in; recovery and route-completion outcomes remain unavailable and qualification eligibility remains false.")],
    }
    matrix_report["report_sha256"] = _canonical_hash(matrix_report)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / f"{matrix_id}.matrix.json", matrix_report)
    return matrix_report
