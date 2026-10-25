"""Compile and verify the non-executing Phase 4 validation campaign schedule.

This compiler binds the schedule to frozen inputs and describes exact matched
coverage without materializing episodes, exposing seed values, or running a
simulator. Design-axis expansions are explicitly not executable until runtime
stimuli and evaluator wiring are implemented and verified.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.parse import urlsplit

from .meco_phase4_design import (
    CASE_AXES,
    _canonical_bytes,
    _read_json,
    _read_protocol_validation,
    _sha256,
    compile_phase4_design_overlay,
)
from .meco_phase4_runtime_adapter import phase4_requirement_binding
from .meco_phase4_opponents import (POLICY_CONFIG, POLICY_FACTOR_IDS,
                                    opponent_policy_runtime_domain_supported,
                                    opponent_policy_runtime_slots, policy_assignment_identity,
                                    policy_pool_identity,
                                    trench_interception_runtime_domain_supported,
                                    compile_opponent_policy_schedule)
from .meco_phase4_context import phase4_start_profile_assignment


PLAN_SCHEMA = "meco-phase4-validation-campaign-plan-v1"
POLICIES = ("D0", "D1", "D2", "A1", "A2", "A3", "C1", "C2")
DIRECT_PURSUIT_POLICY_IDS = tuple(sorted((*POLICIES, "A1_P0", "A2_P0", "A3_P0")))
CONDITIONS = ("no_factor", "factor_treatment")
EXPECTED_FACTOR_CASE_AXIS_CELLS = 6488
VALIDATION_SEEDS_PER_CASE = 100
TRENCH_POLICY_IDS = tuple(sorted((*POLICIES, "A1_P0", "A2_P0", "A3_P0")))
START_PROFILE_TRACK_IDS = ("A1", "A2", "A3", "C1", "C2")


def _read_bound_trench_artifact(root: Path, row: dict, *, path_key: str,
        sha_key: str) -> tuple[dict, str] | None:
    relative = row.get(path_key)
    if not isinstance(relative, str) or not relative:
        return None
    path_value = Path(relative)
    if path_value.is_absolute() or ".." in path_value.parts:
        return None
    root = root.resolve()
    target = (root / path_value).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return None
    raw = target.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if row.get(sha_key) != digest:
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return (payload, digest) if isinstance(payload, dict) else None


def _bound_evidence_bytes(root: Path, relative: Any, expected_sha256: Any
        ) -> bytes | None:
    """Read a retained evidence file only when its safe path and digest match."""
    if (not isinstance(relative, str) or not relative or
            not isinstance(expected_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
        return None
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        return None
    root = root.resolve()
    target = (root / value).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return None
    raw = target.read_bytes()
    return raw if hashlib.sha256(raw).hexdigest() == expected_sha256 else None


def _parse_http_capture(raw: bytes) -> tuple[int, dict[str, str], bytes] | None:
    """Parse a retained complete HTTP response, including its captured headers."""
    marker = raw.find(b"\r\n\r\n")
    if marker < 0:
        return None
    try:
        lines = raw[:marker].decode("iso-8859-1").split("\r\n")
        match = re.fullmatch(r"HTTP/1\.[01] ([0-9]{3})(?: .*)?", lines[0])
        if match is None:
            return None
        headers = {}
        for line in lines[1:]:
            name, separator, value = line.partition(":")
            if not separator or not name or name.lower() in headers:
                return None
            headers[name.lower()] = value.strip()
        body = raw[marker + 4:]
        content_length = headers.get("content-length")
        if (content_length is None or not content_length.isdecimal() or
                int(content_length) != len(body)):
            return None
        return int(match.group(1)), headers, body
    except (UnicodeDecodeError, ValueError):
        return None


def _endpoint_url_matches(url: Any, path: str) -> bool:
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return (parsed.scheme in ("http", "https") and bool(parsed.netloc) and
        parsed.path == path)


def _checkout_matches_source_identity(service: dict, expected_tree: dict,
        expected_tree_sha256: str) -> bool:
    """Rehash source files under the declared checkout and verify its Git head."""
    root_text = service.get("checkout_root")
    identity = service.get("checkout_source_identity")
    if (not isinstance(root_text, str) or not Path(root_text).is_absolute() or
            not isinstance(identity, dict)):
        return False
    root = Path(root_text).resolve()
    if not root.is_dir():
        return False
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    if commit != service.get("checkout_commit"):
        return False
    source_digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    if service.get("checkout_source_identity_sha256") != source_digest:
        return False
    tree_identity = {"source_files": {key: value for key, value in identity.items()
            if not key.startswith("_")},
        "drivetrain_config_sha256": identity.get("_drivetrain_config_sha256")}
    tree_digest = hashlib.sha256(json.dumps(tree_identity, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    if (tree_identity != expected_tree or tree_digest != expected_tree_sha256):
        return False
    for relative, expected in identity.items():
        if relative.startswith("_"):
            continue
        value = Path(relative)
        if (value.is_absolute() or ".." in value.parts or
                not isinstance(expected, str) or
                re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            return False
        path = (root / value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            return False
    return True


def _service_process_matches_checkout(service: dict) -> bool:
    """Bind captured service PID/start data to a live dashboard process."""
    pid = service.get("main_pid")
    start_ticks = service.get("proc_start_ticks")
    checkout_root = service.get("checkout_root")
    if (type(pid) is not int or pid <= 1 or type(start_ticks) is not int or
            not isinstance(checkout_root, str) or
            not Path(checkout_root).is_absolute()):
        return False
    proc = Path("/proc") / str(pid)
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace")
        actual_root = (proc / "cwd").resolve()
        stat = (proc / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        actual_start_ticks = int(fields[19])
    except (OSError, ValueError, IndexError):
        return False
    expected_root = Path(checkout_root).resolve()
    return (actual_root == expected_root and actual_start_ticks == start_ticks and
        ("frc_defense.dashboard" in command or "frc_defense/dashboard.py" in command))


def _trench_interception_review_gate(rows: list[dict], *, protocol: dict,
        protocol_sha256: str, registry_sha256: str, case: dict,
        seed_sha256: str, assignment_sha256: str,
        expected_schedule_sha256: str, seed_values: list[int],
        evidence_root: Path) -> dict | None:
    """Accept only a complete, independently reviewed paired trench evidence row."""
    matches = [row for row in rows if isinstance(row, dict) and
        row.get("scenario_id") == case["scenario_id"] and
        row.get("category") == case["category"] and
        row.get("qualification_case") == {axis: case[axis] for axis in (
            "layout_id", "defender_family", "start_condition", "sensor_profile")}]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("trench-interception review evidence duplicates a protocol case")
    row = matches[0]
    outcome_artifact = _read_bound_trench_artifact(evidence_root, row,
        path_key="paired_outcomes_artifact_path",
        sha_key="paired_outcomes_artifact_sha256")
    review_artifact = _read_bound_trench_artifact(evidence_root, row,
        path_key="replay_review_artifact_path",
        sha_key="replay_review_artifact_sha256")
    if outcome_artifact is None or review_artifact is None:
        return {"accepted": False,
            "reason": "paired outcomes or independent replay review artifact is missing or hash-mismatched"}
    outcome_payload, outcome_sha256 = outcome_artifact
    replay, review_sha256 = review_artifact
    outcome_body = dict(outcome_payload)
    recorded_pairing_sha = outcome_body.pop("pairing_sha256", None)
    expected_keys = {(policy, seed) for policy in TRENCH_POLICY_IDS for seed in seed_values}
    paired_outcomes = outcome_payload.get("paired_outcomes")
    valid_outcome_items = (isinstance(paired_outcomes, list) and all(
        isinstance(item, dict) and item.get("policy_id") in TRENCH_POLICY_IDS and
        type(item.get("seed")) is int for item in paired_outcomes))
    outcome_keys = ({(item["policy_id"], item["seed"]) for item in paired_outcomes}
        if valid_outcome_items else set())
    outcome_valid = (outcome_payload.get("schema") ==
        "meco-phase4-trench-interception-pair-outcomes-v1"
        and outcome_payload.get("factor_id") == "opponent.trench_interception"
        and outcome_payload.get("policy_id") == "trench_interceptor"
        and outcome_payload.get("protocol_id") == protocol.get("protocol_id")
        and outcome_payload.get("protocol_sha256") == protocol_sha256
        and outcome_payload.get("scenario_id") == case["scenario_id"]
        and outcome_payload.get("category") == case["category"]
        and outcome_payload.get("qualification_case") == {axis: case[axis] for axis in (
            "layout_id", "defender_family", "start_condition", "sensor_profile")}
        and outcome_payload.get("split") == "validation"
        and recorded_pairing_sha == _sha256(_canonical_bytes(outcome_body))
        and outcome_payload.get("schedule_sha256") == expected_schedule_sha256
        and outcome_payload.get("validation_seed_assignment_sha256") == seed_sha256
        and outcome_payload.get("policy_ids") == list(TRENCH_POLICY_IDS)
        and outcome_payload.get("assigned_pressure_slots") == [0]
        and outcome_payload.get("control_policy_ids") == list(TRENCH_POLICY_IDS)
        and outcome_payload.get("policy_config_sha256") == policy_pool_identity()["config_sha256"]
        and outcome_payload.get("seed_manifest_sha256") == row.get("seed_manifest_sha256")
        and outcome_payload.get("seed_count") == len(seed_values) == VALIDATION_SEEDS_PER_CASE
        and outcome_payload.get("assigned_rows_per_condition") == {
            "no_factor": len(expected_keys), "factor_treatment": len(expected_keys)}
        and outcome_payload.get("paired_rows_per_condition") == {
            "no_factor": len(expected_keys), "factor_treatment": len(expected_keys)}
        and outcome_keys == expected_keys and len(paired_outcomes) == len(expected_keys)
        and outcome_payload.get("exact_pair_coverage") is True
        and outcome_payload.get("all_pairs_full_horizon_valid") is True
        and outcome_payload.get("outcome_signals_complete") is True
        and outcome_payload.get("invalid_pair_count") == 0
        and outcome_payload.get("unaccounted_assignments") == 0
        and outcome_payload.get("treatment_interception_valid_count") == len(expected_keys)
        and outcome_payload.get("treatment_interception_invalid_count") == 0
        and outcome_payload.get("source_identity_sha256") == row.get("source_identity_sha256")
        and outcome_payload.get("checkpoint_bundle_sha256") == row.get("checkpoint_bundle_sha256")
        and outcome_payload.get("execution_device") == row.get("execution_device"))
    per_arm = outcome_payload.get("per_arm_outcomes")
    outcome_valid = outcome_valid and isinstance(per_arm, dict) and all(
        isinstance(per_arm.get(arm), dict) and
        per_arm[arm].get("assigned_count") == len(expected_keys) and
        per_arm[arm].get("full_horizon_valid_count") == len(expected_keys) and
        per_arm[arm].get("alternate_crossing_success_denominator") == len(expected_keys) and
        per_arm[arm].get("crossing_time_rmst_denominator") == len(expected_keys)
        for arm in CONDITIONS)
    outcome_valid = outcome_valid and per_arm["factor_treatment"].get(
        "interception_denominator") == len(expected_keys)
    reviewer = replay.get("reviewer_identity") if isinstance(replay, dict) else None
    if not outcome_valid:
        return {"accepted": False, "reason": "retained paired outcomes fail exact assignment reconciliation"}
    exact = (row.get("schema") == "meco-phase4-trench-interception-review-evidence-v1"
        and row.get("protocol_id") == protocol.get("protocol_id")
        and row.get("protocol_sha256") == protocol_sha256
        and row.get("registry_sha256") == registry_sha256
        and row.get("factor_id") == "opponent.trench_interception"
        and row.get("policy_id") == "trench_interceptor"
        and row.get("split") == "validation"
        and row.get("validation_seed_assignment_sha256") == seed_sha256
        and row.get("assignment_identity_sha256") == assignment_sha256
        and row.get("control_policy_ids") == list(TRENCH_POLICY_IDS)
        and row.get("assigned_pressure_slots") == [0]
        and row.get("paired_rows_per_condition") == {
            "no_factor": len(TRENCH_POLICY_IDS) * VALIDATION_SEEDS_PER_CASE,
            "factor_treatment": len(TRENCH_POLICY_IDS) * VALIDATION_SEEDS_PER_CASE}
        and row.get("exact_pair_coverage") is True
        and row.get("all_pairs_full_horizon_valid") is True
        and row.get("outcome_signals_complete") is True
        and row.get("treatment_interception_valid_count") ==
            len(TRENCH_POLICY_IDS) * VALIDATION_SEEDS_PER_CASE
        and row.get("treatment_interception_invalid_count") == 0
        and row.get("unaccounted_assignments") == 0
        and row.get("invalid_pair_count") == 0
        and row.get("qualification_eligible") is False
        and isinstance(row.get("source_identity_sha256"), str)
        and bool(row["source_identity_sha256"])
        and isinstance(row.get("checkpoint_bundle_sha256"), dict)
        and bool(row["checkpoint_bundle_sha256"])
        and isinstance(row.get("execution_device"), dict)
        and bool(row["execution_device"])
        and row.get("schedule_sha256") == expected_schedule_sha256
        and isinstance(row.get("seed_manifest_sha256"), str)
        and bool(row["seed_manifest_sha256"])
        and row.get("policy_config_sha256") == policy_pool_identity()["config_sha256"]
        and isinstance(row.get("frontend_asset_manifest_sha256"), str)
        and bool(row["frontend_asset_manifest_sha256"])
        and row.get("paired_outcomes_artifact_sha256") == outcome_sha256
        and row.get("replay_review_artifact_sha256") == review_sha256)
    replays = replay.get("paired_replays") if isinstance(replay, dict) else None
    exact = (exact and isinstance(replay, dict) and replay.get("schema") ==
        "meco-phase4-trench-interception-replay-review-v1" and
        replay.get("accepted") is True)
    exact = exact and isinstance(reviewer, dict) and all(
        isinstance(reviewer.get(key), str) and reviewer.get(key)
        for key in ("subject_id", "identity_provider", "review_record_id"))
    exact = exact and replay.get("paired_outcomes_artifact_sha256") == outcome_sha256
    exact = exact and replay.get("protocol_id") == protocol.get("protocol_id")
    exact = exact and replay.get("protocol_sha256") == protocol_sha256
    exact = exact and replay.get("factor_id") == "opponent.trench_interception"
    exact = exact and replay.get("policy_id") == "trench_interceptor"
    exact = exact and replay.get("policy_config_sha256") == policy_pool_identity()["config_sha256"]
    exact = exact and replay.get("assignment_identity_sha256") == assignment_sha256
    exact = exact and replay.get("schedule_sha256") == expected_schedule_sha256
    exact = exact and replay.get("scenario_id") == case["scenario_id"]
    exact = exact and replay.get("category") == case["category"]
    exact = exact and replay.get("qualification_case") == {axis: case[axis] for axis in (
        "layout_id", "defender_family", "start_condition", "sensor_profile")}
    exact = exact and replay.get("split") == "validation"
    exact = exact and all(replay.get(flag) is True for flag in (
        "dashboard_backend_current", "dashboard_frontend_current", "endpoint_healthy",
        "replay_current_checkout", "replay_seed_is_in_exact_validation_manifest"))
    exact = exact and replay.get("backend_source_identity_sha256") == row.get(
        "source_identity_sha256")
    exact = exact and replay.get("frontend_asset_manifest_sha256") == row.get(
        "frontend_asset_manifest_sha256")
    exact = exact and replay.get("seed_manifest_sha256") == row.get("seed_manifest_sha256")
    exact = exact and type(replay.get("replay_seed")) is int
    exact = exact and replay.get("replay_seed") == next(iter(
        {item.get("seed") for item in replays if isinstance(item, dict)}), None)
    exact = exact and isinstance(replays, list) and len(replays) == 2
    if not isinstance(replays, list):
        exact = False
    elif any(not isinstance(item, dict) or type(item.get("seed")) is not int
            for item in replays):
        exact = False
    exact = exact and {item.get("condition") for item in replays if isinstance(item, dict)} == set(CONDITIONS)
    exact = exact and len({item.get("seed") for item in replays if isinstance(item, dict)}) == 1
    exact = exact and all(isinstance(item, dict)
        and item.get("scenario_id") == case["scenario_id"]
        and item.get("category") == case["category"]
        and item.get("qualification_case") == {axis: case[axis] for axis in (
            "layout_id", "defender_family", "start_condition", "sensor_profile")}
        and item.get("split") == "validation"
        and item.get("seed_manifest_sha256") == row.get("seed_manifest_sha256")
        and item.get("source_identity_sha256") == row.get("source_identity_sha256")
        and item.get("checkpoint_bundle_sha256") == row.get("checkpoint_bundle_sha256")
        and item.get("execution_device") == row.get("execution_device")
        and item.get("source_identity_sha256") == row.get("source_identity_sha256")
        and item.get("frontend_asset_manifest_sha256") == row.get(
            "frontend_asset_manifest_sha256")
        and item.get("policy_id") == "trench_interceptor"
        and item.get("schedule_sha256") == row.get("schedule_sha256")
        and item.get("assignment_identity_sha256") == assignment_sha256
        and item.get("policy_config_sha256") == policy_pool_identity()["config_sha256"]
        and item.get("seed") == replay.get("replay_seed")
        and item.get("seed_manifest_sha256") == replay.get("seed_manifest_sha256")
        and item.get("backend_source_identity_sha256") == row.get("source_identity_sha256")
        and item.get("assigned_pressure_slots") == [0]
        and item.get("applied_slots") == ([] if item.get("condition") == "no_factor" else [0])
        and isinstance(item.get("replay_id"), str) and bool(item["replay_id"])
        and isinstance(item.get("replay_sha256"), str) and bool(item["replay_sha256"])
        for item in replays)
    exact = exact and len({item.get("replay_id") for item in replays if isinstance(item, dict)}) == 2
    exact = exact and isinstance(reviewer, dict) and all(
        isinstance(reviewer.get(key), str) and reviewer.get(key)
        for key in ("subject_id", "identity_provider", "review_record_id"))
    if not exact:
        return {"accepted": False, "reason": "trench review evidence is incomplete or mismatched"}
    return {"accepted": True, "evidence_sha256": _sha256(_canonical_bytes(row)),
        "source_identity_sha256": row["source_identity_sha256"],
        "checkpoint_bundle_sha256": row["checkpoint_bundle_sha256"],
        "execution_device": row["execution_device"],
        "schedule_sha256": row["schedule_sha256"],
        "frontend_asset_manifest_sha256": row["frontend_asset_manifest_sha256"],
        "paired_outcomes_artifact_sha256": row["paired_outcomes_artifact_sha256"],
        "replay_review_artifact_sha256": row["replay_review_artifact_sha256"],
        "reviewer_identity": reviewer}


def _seed_digest(seeds: list[int]) -> str:
    """Bind exact validation seed assignment without returning its values."""
    return _sha256(_canonical_bytes(seeds))


def _direct_pursuit_review_gate(rows: list[dict], *, protocol: dict,
        protocol_sha256: str, registry_sha256: str, case: dict, seed_sha256: str,
        assignment_sha256: str, seed_count: int,
        configured_pressure_slots: list[int]) -> dict | None:
    matches = [row for row in rows if isinstance(row, dict) and
        row.get("scenario_id") == case["scenario_id"] and
        row.get("category") == case["category"] and
        row.get("qualification_case") == {axis: case[axis] for axis in (
            "layout_id", "defender_family", "start_condition", "sensor_profile")}]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("direct-pursuit review evidence duplicates a protocol case")
    row = matches[0]
    replay = row.get("replay_review")
    outcomes = row.get("paired_outcomes")
    paired_replays = replay.get("paired_replays") if isinstance(replay, dict) else None
    pair_count = seed_count * len(DIRECT_PURSUIT_POLICY_IDS)
    schedule_sha256 = row.get("schedule_sha256")
    seed_manifest_sha256 = row.get("seed_manifest_sha256")
    interception = outcomes.get("interception_rate") if isinstance(outcomes, dict) else None
    contact_rate = outcomes.get("robot_contact_rate") if isinstance(outcomes, dict) else None
    time_loss = outcomes.get("paired_time_loss_to_opponents_s") if isinstance(outcomes, dict) else None
    exact_binding = (row.get("schema") == "meco-phase4-direct-pursuit-review-evidence-v1"
        and row.get("protocol_id") == protocol.get("protocol_id")
        and row.get("protocol_sha256") == protocol_sha256
        and row.get("registry_sha256") == registry_sha256
        and row.get("factor_id") == "opponent.direct_pursuit"
        and row.get("policy_id") == "direct_pursuit"
        and row.get("policy_config_sha256") == policy_pool_identity()["config_sha256"]
        and row.get("split") == "validation"
        and row.get("validation_seed_assignment_sha256") == seed_sha256
        and row.get("assignment_identity_sha256") == assignment_sha256
        and isinstance(schedule_sha256, str) and bool(schedule_sha256)
        and isinstance(seed_manifest_sha256, str) and bool(seed_manifest_sha256)
        and row.get("configured_pressure_slots") == configured_pressure_slots
        and row.get("control_policy_ids") == list(DIRECT_PURSUIT_POLICY_IDS)
        and isinstance(row.get("paired_rows_per_condition"), dict)
        and set(row["paired_rows_per_condition"]) == {
            "no_factor", "factor_treatment"}
        and row["paired_rows_per_condition"] == {
            "no_factor": pair_count, "factor_treatment": pair_count}
        and isinstance(row.get("source_identity_sha256"), str)
        and bool(row.get("source_identity_sha256"))
        and isinstance(row.get("checkpoint_bundle_sha256"), dict)
        and bool(row.get("checkpoint_bundle_sha256"))
        and all(isinstance(value, str) and value
                for value in row["checkpoint_bundle_sha256"].values())
        and isinstance(row.get("execution_device"), dict)
        and bool(row.get("execution_device"))
        and row.get("exact_pair_coverage") is True
        and row.get("all_pairs_uncensored") is True
        and row.get("qualification_eligible") is False
        and row.get("eligible_pair_count") == pair_count
        and isinstance(row.get("paired_outcomes_artifact_sha256"), str)
        and bool(row.get("paired_outcomes_artifact_sha256"))
        and isinstance(outcomes, dict)
        and isinstance(interception, dict)
        and interception.get("denominator_eligible_episodes") == pair_count
        and type(interception.get("numerator_intercepted_episodes")) is int
        and 0 <= interception["numerator_intercepted_episodes"] <= pair_count
        and isinstance(interception.get("rate"), (int, float))
        and math.isfinite(interception["rate"])
        and abs(interception["rate"] - interception[
            "numerator_intercepted_episodes"] / pair_count) <= 1e-12
        and isinstance(contact_rate, dict)
        and type(contact_rate.get("numerator_distinct_pair_contact_substeps")) is int
        and contact_rate["numerator_distinct_pair_contact_substeps"] >= 0
        and isinstance(contact_rate.get("denominator_pair_seconds"), (int, float))
        and math.isfinite(contact_rate["denominator_pair_seconds"])
        and contact_rate["denominator_pair_seconds"] > 0
        and isinstance(contact_rate.get(
            "rate_distinct_pair_contact_substeps_per_pair_second"), (int, float))
        and math.isfinite(contact_rate[
            "rate_distinct_pair_contact_substeps_per_pair_second"])
        and isinstance(time_loss, list) and len(time_loss) == pair_count
        and all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in time_loss)
        and isinstance(replay, dict)
        and replay.get("review_accepted") is True
        and replay.get("validation_seed_assignment_sha256") == seed_sha256
        and replay.get("dashboard_backend_current") is True
        and replay.get("dashboard_frontend_current") is True
        and replay.get("endpoint_healthy") is True
        and replay.get("replay_seed_is_in_exact_validation_manifest") is True
        and replay.get("replay_current_checkout") is True
        and type(replay.get("replay_seed")) is int
        and 0 <= replay["replay_seed"] < 2**63
        and replay.get("seed_manifest_sha256") == seed_manifest_sha256
        and replay.get("backend_source_identity_sha256") == row.get(
            "source_identity_sha256")
        and isinstance(replay.get("frontend_asset_manifest_sha256"), str)
        and bool(replay.get("frontend_asset_manifest_sha256"))
        and isinstance(paired_replays, list) and len(paired_replays) == 2
        and {item.get("condition") for item in paired_replays if isinstance(item, dict)} ==
            {"no_factor", "factor_treatment"}
        and len({item.get("seed") for item in paired_replays if isinstance(item, dict)}) == 1
        and all(isinstance(item, dict)
            and item.get("scenario_id") == case["scenario_id"]
            and item.get("category") == case["category"]
            and item.get("qualification_case") == {axis: case[axis] for axis in (
                "layout_id", "defender_family", "start_condition", "sensor_profile")}
            and item.get("split") == "validation"
            and type(item.get("seed")) is int and 0 <= item["seed"] < 2**63
            and item.get("seed") == replay.get("replay_seed")
            and item.get("seed_manifest_sha256") == seed_manifest_sha256
            and item.get("schedule_sha256") == schedule_sha256
            and item.get("assignment_identity_sha256") == assignment_sha256
            and item.get("source_identity_sha256") == row.get("source_identity_sha256")
            and item.get("frontend_asset_manifest_sha256") == replay.get(
                "frontend_asset_manifest_sha256")
            and item.get("checkpoint_bundle_sha256") == row.get("checkpoint_bundle_sha256")
            and item.get("execution_device") == row.get("execution_device")
            and item.get("policy_config_sha256") == policy_pool_identity()["config_sha256"]
            and item.get("policy_id") == "direct_pursuit"
            and item.get("configured_slots") == configured_pressure_slots
            and item.get("policy_ids") == list(DIRECT_PURSUIT_POLICY_IDS)
            and ((item.get("condition") == "no_factor" and item.get("applied_slots") == [])
                or (item.get("condition") == "factor_treatment" and
                    item.get("applied_slots") == configured_pressure_slots))
            and isinstance(item.get("replay_id"), str) and bool(item["replay_id"])
            and isinstance(item.get("replay_sha256"), str) and bool(item["replay_sha256"])
            for item in paired_replays)
        and isinstance(replay.get("reviewer"), str) and bool(replay["reviewer"]))
    if not exact_binding:
        return {"accepted": False, "reason": "direct-pursuit review evidence is incomplete or mismatched"}
    return {"accepted": True, "evidence_sha256": _sha256(_canonical_bytes(row)),
        "source_identity_sha256": row["source_identity_sha256"],
        "checkpoint_bundle_sha256": row["checkpoint_bundle_sha256"],
        "execution_device": row["execution_device"],
        "schedule_sha256": schedule_sha256,
        "registry_sha256": registry_sha256,
        "seed_manifest_sha256": seed_manifest_sha256,
        "paired_replay_sha256": _sha256(_canonical_bytes(paired_replays))}


def _reconcile_start_profile_outcomes(outcomes: dict, evidence_root: Path, *,
        outcome_artifact_path: Path,
        assignment: dict, expected_keys: set[tuple], categories: list[str],
        seeds: list[int]) -> bool:
    """Recompute report coverage, summaries, legality, deltas, and digest."""
    from .meco_ablation_matrix import _records_by_key
    from .meco_validation_campaign import _canonical_bytes

    paths = outcomes.get("condition_report_paths")
    hashes = outcomes.get("condition_report_sha256")
    matrix_paths = outcomes.get("condition_matrix_paths")
    matrix_hashes = outcomes.get("condition_matrix_sha256")
    arms = {}
    evidence_root = evidence_root.resolve()
    root = outcome_artifact_path.resolve().parent
    if not root.is_relative_to(evidence_root):
        return False
    for arm in CONDITIONS:
        relative_matrix = Path(matrix_paths[arm])
        if relative_matrix.is_absolute() or ".." in relative_matrix.parts:
            return False
        matrix_path = (root / relative_matrix).resolve()
        if not matrix_path.is_relative_to(root) or not matrix_path.is_file():
            return False
        matrix_raw = matrix_path.read_bytes()
        if hashlib.sha256(matrix_raw).hexdigest() != matrix_hashes.get(arm):
            return False
        matrix = json.loads(matrix_raw)
        expected_assignment = assignment if arm == "factor_treatment" else None
        matrix_protocol = matrix.get("protocol") if isinstance(matrix, dict) else None
        matrix_conditions = matrix.get("conditions") if isinstance(matrix, dict) else None
        if (not isinstance(matrix, dict) or
                matrix.get("qualification_eligible") is not False or
                matrix.get("phase4_start_profile_assignment") != expected_assignment or
                not isinstance(matrix_protocol, dict) or
                matrix_protocol.get("phase4_start_profile_assignment") != expected_assignment or
                matrix_protocol.get("qualification_case") !=
                    outcomes.get("qualification_case") or
                matrix_protocol.get("categories") != categories or
                matrix_protocol.get("seed_list") != seeds or
                not isinstance(matrix_conditions, dict) or
                set(matrix_conditions) != {"architecture_matched_cv_k16"}):
            return False
        relative = Path(paths[arm])
        if relative.is_absolute() or ".." in relative.parts:
            return False
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return False
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != hashes.get(arm):
            return False
        report = json.loads(raw)
        matrix_condition = matrix_conditions["architecture_matched_cv_k16"]
        if (not isinstance(report, dict) or
                report.get("qualification_eligible") is not False or
                report.get("phase4_start_profile_assignment") != expected_assignment or
                report.get("scenario_id") != outcomes.get("scenario_id") or
                report.get("categories") != categories or
                report.get("qualification_case") != outcomes.get("qualification_case") or
                report.get("evaluation_split") != "validation" or
                report.get("source_identity_sha256") !=
                    outcomes.get("source_identity_sha256") or
                report.get("checkpoint_sha256") != outcomes.get("checkpoint_sha256") or
                report.get("execution_device") != outcomes.get("execution_device")):
            return False
        if (matrix_condition.get("evaluation_report_sha256") != hashes.get(arm) or
                outcomes.get("condition_ids", {}).get(arm) !=
                    matrix_condition.get("condition_id") or
                matrix_condition.get("qualification_case") !=
                    outcomes.get("qualification_case") or
                matrix_condition.get("evaluation_split") != "validation" or
                matrix_condition.get("checkpoint_sha256") !=
                    outcomes.get("checkpoint_sha256") or
                matrix_condition.get("source_identity_sha256") !=
                    outcomes.get("source_identity_sha256") or
                matrix_condition.get("execution_device") !=
                    outcomes.get("execution_device")):
            return False
        experiment_matrix = report.get("experiment_matrix")
        if (not isinstance(experiment_matrix, dict) or
                experiment_matrix.get("qualification_eligible") is not False or
                experiment_matrix.get("condition_id") !=
                    matrix_condition.get("condition_id") or
                experiment_matrix.get("evaluation_split") != "validation"):
            return False
        rows_by_key = _records_by_key(report)
        if set(rows_by_key) != expected_keys:
            return False
        manifest = report.get("policy_manifest")
        if not isinstance(manifest, dict):
            return False
        for policy in START_PROFILE_POLICY_IDS:
            track = policy[:-3] if policy.endswith("_P0") else policy
            provenance = manifest.get(policy)
            if (not isinstance(provenance, dict) or
                    provenance.get("checkpoint_sha256") !=
                        outcomes["checkpoint_sha256"].get(track) or
                    provenance.get("source_identity_sha256") !=
                        outcomes.get("source_identity_sha256") or
                    provenance.get("execution_device") !=
                        outcomes.get("execution_device")):
                return False
        for key, row in rows_by_key.items():
            track = key[0][:-3] if key[0].endswith("_P0") else key[0]
            metrics = row.get("metrics")
            if (row.get("checkpoint_sha256") !=
                    outcomes["checkpoint_sha256"].get(track) or
                    row.get("source_identity_sha256") !=
                        outcomes.get("source_identity_sha256") or
                    row.get("execution_device") != outcomes.get("execution_device") or
                    not isinstance(metrics, dict)):
                return False
            for name in ("success", "scores", "useful_throughput", "completed_cycles"):
                value = metrics.get(name)
                if (type(value) not in (bool, int, float) or
                        (type(value) is not bool and not math.isfinite(float(value)))):
                    return False
            captured = row.get("phase4_start_profile")
            if arm == "no_factor":
                if captured is not None:
                    return False
            elif (not isinstance(captured, dict) or
                    captured.get("factor_id") != assignment.get("factor_id") or
                    captured.get("registry_profile_id") !=
                        assignment.get("registry_profile_id") or
                    captured.get("profile_config_sha256") !=
                        assignment.get("profile_config_sha256") or
                    captured.get("assignment_sha256") !=
                        assignment.get("assignment_sha256") or
                    captured.get("profile_applied_before_observation") is not True or
                    captured.get("qualification_eligible") is not False):
                return False
        arms[arm] = rows_by_key

    # Rebuild per-world records from treatment rows and reject any missing,
    # duplicated, reordered, or rate-mismatched world evidence.
    legality = []
    total_worlds = total_legal = 0
    for key in sorted(expected_keys):
        captured = arms["factor_treatment"][key]["phase4_start_profile"]
        count = captured.get("validated_pose_world_count")
        world_indices = captured.get("validated_pose_world_indices")
        values = captured.get("per_world_legal")
        legal_count = captured.get("validated_pose_legal_world_count")
        rate = captured.get("legal_start_validation_rate")
        if (type(count) is not int or count < 1 or
                world_indices != list(range(count)) or
                captured.get("validated_pose_world_index_semantics") !=
                    "zero_based_simulator_batch_index" or
                not isinstance(values, list) or len(values) != count or
                any(type(value) is not bool for value in values) or
                type(legal_count) is not int or legal_count != sum(values) or
                type(rate) not in (int, float) or not math.isfinite(float(rate)) or
                float(rate) != legal_count / count):
            return False
        legality.append({"policy": key[0], "category": key[1], "seed": key[2],
            "world_count": count,
            "world_index_semantics": "zero_based_simulator_batch_index",
            "world_indices": world_indices, "per_world_legal": values,
            "legal_world_count": legal_count, "legal_start_validation_rate": rate})
        total_worlds += count
        total_legal += legal_count
    if (outcomes.get("per_world_legality_records") != legality or
            outcomes.get("legal_world_count") != total_worlds or
            outcomes.get("legal_world_legal_count") != total_legal or
            outcomes.get("legal_start_validation_rate") != total_legal / total_worlds or
            outcomes.get("all_treatment_worlds_legal") is not
                (total_legal == total_worlds)):
        return False

    metrics = ("success", "scores", "useful_throughput", "completed_cycles")
    expected_states = {"scenario_defined", assignment["registry_profile_id"]}
    observed = outcomes.get("outcome_by_start_state")
    if (not isinstance(observed, dict) or
            set(observed.get("states", {})) != expected_states or
            set(observed.get("paired_deltas_vs_scenario_defined", {})) !=
                set(START_PROFILE_POLICY_IDS)):
        return False
    for state, arm in (("scenario_defined", "no_factor"),
                       (assignment["registry_profile_id"], "factor_treatment")):
        state_record = observed["states"][state]
        if not isinstance(state_record, dict):
            return False
        by_policy = state_record.get("by_policy")
        if (state_record.get("n") != len(expected_keys) or
                not isinstance(by_policy, dict) or
                set(by_policy) != set(START_PROFILE_POLICY_IDS)):
            return False
        for policy in START_PROFILE_POLICY_IDS:
            policy_record = by_policy[policy]
            if not isinstance(policy_record, dict):
                return False
            records = [arms[arm][(policy, category, seed)]
                for category in categories for seed in seeds]
            raw_values = {name: [float(row["metrics"][name]) for row in records]
                for name in metrics}
            expected_metric_means = {"success_rate": sum(raw_values["success"])
                / len(records), **{name: sum(raw_values[name]) / len(records)
                    for name in metrics if name != "success"}}
            if (policy_record.get("n") != len(records) or
                    not isinstance(policy_record.get("metrics"), dict) or
                    set(policy_record["metrics"]) != set(expected_metric_means) or
                    any(type(policy_record["metrics"].get(name)) not in (int, float) or
                        not math.isfinite(float(policy_record["metrics"][name])) or
                        abs(float(policy_record["metrics"][name]) - value) > 1e-12
                        for name, value in expected_metric_means.items())):
                return False

    for policy in START_PROFILE_POLICY_IDS:
        policy_deltas = observed["paired_deltas_vs_scenario_defined"][policy]
        if not isinstance(policy_deltas, dict) or set(policy_deltas) != set(metrics):
            return False
        for metric in metrics:
            delta_record = policy_deltas[metric]
            if not isinstance(delta_record, dict):
                return False
            expected_per_key = [{"category": category, "seed": seed,
                "delta": float(arms["factor_treatment"][(policy, category, seed)][
                    "metrics"][metric]) - float(arms["no_factor"][(policy, category, seed)][
                    "metrics"][metric])}
                for category in categories for seed in seeds]
            if (delta_record.get("paired_n") != len(expected_per_key) or
                    delta_record.get("per_category_seed") != expected_per_key or
                    type(delta_record.get("mean_delta")) not in (int, float) or
                    not math.isfinite(float(delta_record["mean_delta"])) or
                    abs(float(delta_record["mean_delta"]) - sum(
                        item["delta"] for item in expected_per_key) /
                        len(expected_per_key)) > 1e-12):
                return False

    recomputed_digest = hashlib.sha256(_canonical_bytes({
        "expected_keys": [list(key) for key in sorted(expected_keys)],
        "assignment_sha256": assignment["assignment_sha256"],
        "checkpoint_sha256": outcomes["checkpoint_sha256"],
        "source_identity_sha256": outcomes["source_identity_sha256"],
        "execution_device": outcomes["execution_device"],
        "condition_report_sha256": hashes,
        "condition_matrix_sha256": matrix_hashes,
        "per_world_legality_records": legality,
        "outcome_by_start_state": observed})).hexdigest()
    return (outcomes.get("coverage_digest_sha256") == recomputed_digest and
            total_worlds > 0 and total_legal == total_worlds)


START_PROFILE_POLICY_IDS = tuple(sorted((
    "A1", "A2", "A3", "C1", "C2", "A1_P0", "A2_P0", "A3_P0",
    "D0", "D1", "D2")))


def _start_profile_review_gate(rows: list[dict], *, protocol: dict,
        protocol_sha256: str, registry_sha256: str, case: dict,
        seed_sha256: str, assignment: dict, seed_values: list[int],
        evidence_root: Path) -> dict | None:
    """Accept reviewed replay evidence for exactly one registered profile cell."""
    case_axes = {axis: case[axis] for axis in (
        "layout_id", "defender_family", "start_condition", "sensor_profile")}
    matches = [row for row in rows if isinstance(row, dict) and
        row.get("scenario_id") == case["scenario_id"] and
        row.get("category") == case["category"] and
        row.get("qualification_case") == case_axes and
        row.get("registry_profile_id") == assignment.get("registry_profile_id")]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("start-profile review evidence duplicates a candidate cell")
    row = matches[0]
    outcome_bound = _read_bound_trench_artifact(evidence_root, row,
        path_key="paired_outcomes_artifact_path",
        sha_key="paired_outcomes_artifact_sha256")
    review_bound = _read_bound_trench_artifact(evidence_root, row,
        path_key="replay_review_artifact_path",
        sha_key="replay_review_artifact_sha256")
    dashboard_bound = _read_bound_trench_artifact(evidence_root, row,
        path_key="dashboard_evidence_artifact_path",
        sha_key="dashboard_evidence_artifact_sha256")
    if outcome_bound is None or review_bound is None or dashboard_bound is None:
        return {"accepted": False,
            "reason": "paired outcomes, dashboard evidence, or replay review artifact is missing or hash-mismatched"}
    outcomes, outcome_sha = outcome_bound
    replay, review_sha = review_bound
    dashboard, dashboard_sha = dashboard_bound
    expected_keys = {(policy, case["category"], seed) for policy in
        (*POLICIES, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2")
        for seed in seed_values}
    keys = outcomes.get("paired_keys")
    normalized_keys = {(item[0], item[1], item[2]) for item in keys
        if isinstance(item, list) and len(item) == 3 and
        isinstance(item[0], str) and isinstance(item[1], str) and
        type(item[2]) is int} if isinstance(keys, list) else set()
    metadata_exact = (outcomes.get("schema") ==
        "meco-phase4-start-profile-paired-outcomes-v1" and
        outcomes.get("protocol_id") == protocol.get("protocol_id") and
        outcomes.get("protocol_sha256") == protocol_sha256 and
        outcomes.get("registry_sha256") == registry_sha256 and
        outcomes.get("scenario_id") == case["scenario_id"] and
        outcomes.get("category") == case["category"] and
        outcomes.get("qualification_case") == case_axes and
        outcomes.get("split") == "validation" and
        outcomes.get("validation_seed_assignment_sha256") == seed_sha256 and
        outcomes.get("registry_profile_id") == assignment.get("registry_profile_id") and
        outcomes.get("profile_config_sha256") == assignment.get("profile_config_sha256") and
        outcomes.get("assignment_sha256") == assignment.get("assignment_sha256") and
        isinstance(outcomes.get("seed_manifest_sha256"), str) and
        len(outcomes["seed_manifest_sha256"]) == 64 and
        normalized_keys == expected_keys and len(keys) == len(expected_keys) and
        outcomes.get("qualification_eligible") is False and
        outcomes.get("state_ids") == ["scenario_defined", assignment["registry_profile_id"]] and
        isinstance(outcomes.get("condition_ids"), dict) and
        set(outcomes["condition_ids"]) == set(CONDITIONS) and
        all(isinstance(value, str) and value for value in
            outcomes["condition_ids"].values()) and
        isinstance(outcomes.get("source_identity_sha256"), str) and
        bool(outcomes.get("source_identity_sha256")) and
        isinstance(outcomes.get("source_tree_identity"), dict) and
        isinstance(outcomes.get("source_tree_identity_sha256"), str) and
        re.fullmatch(r"[0-9a-f]{64}", outcomes[
            "source_tree_identity_sha256"]) is not None and
        isinstance(outcomes.get("checkpoint_sha256"), dict) and
        set(outcomes["checkpoint_sha256"]) == set(START_PROFILE_TRACK_IDS) and
        all(isinstance(value, str) and value for value in
            outcomes["checkpoint_sha256"].values()) and
        isinstance(outcomes.get("execution_device"), dict) and
        bool(outcomes.get("execution_device")) and
        isinstance(outcomes.get("condition_report_sha256"), dict) and
        set(outcomes["condition_report_sha256"]) == set(CONDITIONS) and
        all(isinstance(value, str) and len(value) == 64 for value in
            outcomes["condition_report_sha256"].values()) and
        isinstance(outcomes.get("condition_report_paths"), dict) and
        set(outcomes["condition_report_paths"]) == set(CONDITIONS) and
        isinstance(outcomes.get("condition_matrix_sha256"), dict) and
        set(outcomes["condition_matrix_sha256"]) == set(CONDITIONS) and
        all(isinstance(value, str) and len(value) == 64 for value in
            outcomes["condition_matrix_sha256"].values()) and
        isinstance(outcomes.get("condition_matrix_paths"), dict) and
        set(outcomes["condition_matrix_paths"]) == set(CONDITIONS) and
        isinstance(outcomes.get("per_world_legality_records"), list))
    exact_outcomes = (metadata_exact and
        _reconcile_start_profile_outcomes(outcomes, evidence_root,
            outcome_artifact_path=(evidence_root / row[
                "paired_outcomes_artifact_path"]).resolve(),
            assignment=assignment, expected_keys=expected_keys,
            categories=[case["category"]], seeds=seed_values))
    if not exact_outcomes:
        return {"accepted": False,
            "reason": "paired outcomes fail exact single-profile coverage or binding reconciliation"}
    dashboard_tree = outcomes.get("source_tree_identity_sha256")
    endpoints = dashboard.get("endpoint_evidence") if isinstance(dashboard, dict) else None
    asset_manifest = dashboard.get("frontend_asset_manifest") if isinstance(
        dashboard, dict) else None
    required_endpoints = {"/", "/meco", "/api/meco-status",
                          "/api/curriculum-status"}
    required_assets = {"/assets/dashboard.css", "/assets/dashboard.js",
                       "/assets/meco.css", "/assets/meco.js"}
    asset_paths = ({item.get("path") for item in asset_manifest
                    if isinstance(item, dict) and
                    isinstance(item.get("path"), str)}
                   if isinstance(asset_manifest, list) else set())
    assets_well_formed = (isinstance(asset_manifest, list) and
        all(isinstance(item, dict) and
            isinstance(item.get("path"), str) and
            item["path"].startswith("/assets/") and
            isinstance(item.get("checkout_sha256"), str) and
            len(item["checkout_sha256"]) == 64 and
            isinstance(item.get("served_sha256"), str) and
            len(item["served_sha256"]) == 64 and
            isinstance(item.get("checkout_path"), str) and
            isinstance(item.get("served_path"), str) and
            item["checkout_sha256"] == item["served_sha256"]
            for item in asset_manifest) and
        len(asset_paths) == len(asset_manifest) and
        required_assets.issubset(asset_paths) and
        asset_manifest == sorted(asset_manifest, key=lambda item: item["path"]))
    manifest_sha = (_sha256(_canonical_bytes(asset_manifest))
        if assets_well_formed else None)
    endpoint_paths = set(endpoints) if isinstance(endpoints, dict) else set()
    required_endpoint_paths = required_endpoints | required_assets | asset_paths
    endpoint_evidence_exact = (isinstance(endpoints, dict) and
        required_endpoint_paths.issubset(endpoint_paths) and
        all(isinstance(endpoints.get(path), dict) and
            endpoints[path].get("http_status") == 200 and
            isinstance(endpoints[path].get("response_sha256"), str) and
            len(endpoints[path]["response_sha256"]) == 64 and
            isinstance(endpoints[path].get("capture_path"), str) and
            isinstance(endpoints[path].get("capture_sha256"), str) and
            len(endpoints[path]["capture_sha256"]) == 64 and
            _endpoint_url_matches(endpoints[path].get("url"), path)
            for path in required_endpoint_paths) and
        all(endpoints[path].get("backend_source_identity_sha256") ==
                dashboard.get("backend_source_identity_sha256") for path in
            ("/api/meco-status", "/api/curriculum-status")) and
        all(endpoints[path].get("response_sha256") == next(
                item["served_sha256"] for item in asset_manifest
                if item["path"] == path) for path in asset_paths)
        if assets_well_formed else False)
    service = dashboard.get("service") if isinstance(dashboard, dict) else None
    checkout_exact = (isinstance(service, dict) and
        _checkout_matches_source_identity(service,
            outcomes.get("source_tree_identity"), dashboard_tree))
    endpoint_capture_records = []
    if endpoint_evidence_exact:
        for path in required_endpoint_paths:
            capture = _bound_evidence_bytes(evidence_root,
                endpoints[path].get("capture_path"),
                endpoints[path].get("capture_sha256"))
            parsed = _parse_http_capture(capture) if capture is not None else None
            endpoint_capture_records.append((path, parsed))
    endpoint_captures_exact = (endpoint_evidence_exact and
        len(endpoint_capture_records) == len(required_endpoint_paths) and
        all(parsed is not None and parsed[0] == 200 and parsed[2] and
            hashlib.sha256(parsed[2]).hexdigest() ==
                endpoints[path]["response_sha256"] and
            parsed[1].get("x-autodrive-source-identity-sha256") ==
                dashboard.get("backend_source_identity_sha256") and
            parsed[1].get("x-autodrive-source-tree-identity-sha256") == dashboard_tree and
            (path not in asset_paths or hashlib.sha256(parsed[2]).hexdigest() ==
                next(asset["served_sha256"] for asset in asset_manifest
                     if asset["path"] == path))
            for path, parsed in endpoint_capture_records))
    assets_bytes_exact = False
    if isinstance(service, dict) and assets_well_formed:
        asset_checks = []
        for item in asset_manifest:
            checkout_bytes = _bound_evidence_bytes(
                Path(service["checkout_root"]), item.get("checkout_path"),
                item.get("checkout_sha256"))
            served_bytes = _bound_evidence_bytes(evidence_root,
                item.get("served_path"), item.get("served_sha256"))
            asset_checks.append(checkout_bytes is not None and
                served_bytes is not None and checkout_bytes == served_bytes)
        assets_bytes_exact = all(asset_checks)
    dashboard_valid = (isinstance(dashboard, dict) and
        dashboard.get("schema") ==
            "meco-phase4-start-profile-dashboard-evidence-v1" and
        isinstance(service, dict) and
        isinstance(dashboard.get("backend_source_identity_sha256"), str) and
        re.fullmatch(r"[0-9a-f]{64}", dashboard[
            "backend_source_identity_sha256"]) is not None and
        dashboard.get("backend_source_identity_sha256") == service.get(
            "checkout_source_identity_sha256") and
        dashboard.get("checkout_source_identity_sha256") == service.get(
            "checkout_source_identity_sha256") and
        dashboard.get("backend_source_tree_identity_sha256") == dashboard_tree and
        dashboard.get("checkout_source_tree_identity_sha256") == dashboard_tree and
        service.get("name") == "frc-defense-dashboard.service" and
        service.get("active") is True and type(service.get("main_pid")) is int and
        service["main_pid"] > 1 and
        isinstance(service.get("started_at_utc"), str) and
        bool(service["started_at_utc"]) and
        _service_process_matches_checkout(service) and
        isinstance(service.get("checkout_root"), str) and
        Path(service["checkout_root"]).is_absolute() and
        isinstance(service.get("checkout_commit"), str) and
        re.fullmatch(r"[0-9a-f]{40}", service["checkout_commit"]) is not None and
        isinstance(service.get("checkout_source_identity_sha256"), str) and
        checkout_exact and assets_bytes_exact and endpoint_captures_exact and
        assets_well_formed and dashboard.get(
            "frontend_asset_manifest_sha256") == manifest_sha and
        dashboard.get("checkout_frontend_asset_manifest_sha256") == manifest_sha and
        endpoint_evidence_exact)
    endpoint_sha = (_sha256(_canonical_bytes(endpoints))
        if endpoint_evidence_exact else None)
    if not dashboard_valid:
        return {"accepted": False,
            "reason": "dashboard backend, endpoint responses, or frontend asset manifest is not source-bound"}
    reviewer = replay.get("reviewer_identity") if isinstance(replay, dict) else None
    replays = replay.get("paired_replays") if isinstance(replay, dict) else None
    replay_artifacts_exact = (isinstance(replays, list) and len(replays) == 2)
    replay_payloads = []
    if replay_artifacts_exact:
        for item in replays:
            raw = (_bound_evidence_bytes(evidence_root, item.get("artifact_path"),
                item.get("replay_sha256")) if isinstance(item, dict) else None)
            try:
                payload = json.loads(raw) if raw is not None else None
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            replay_payloads.append(payload)
        replay_artifacts_exact = (all(isinstance(item, dict) and
                isinstance(item.get("simulation_id"), str)
            for item in replay_payloads) and
            len({item.get("simulation_id") for item in replay_payloads
                 if isinstance(item, dict)}) == 2 and
            all(isinstance(review_item, dict) and
                isinstance(item.get("simulation_id"), str) and
                item.get("simulation_id") == review_item.get("replay_id") and
                item.get("scenario_id") == case["scenario_id"] and
                item.get("category") == case["category"] and
                item.get("qualification_case") == case_axes and
                item.get("split") == "validation" and
                review_item.get("condition") in CONDITIONS and
                item.get("condition_id") == outcomes.get("condition_ids", {}).get(
                    review_item.get("condition")) and
                item.get("seed") == review_item.get("seed") ==
                    replay.get("replay_seed") and
                item.get("phase4_start_profile_assignment") ==
                    (assignment if review_item.get("condition") ==
                        "factor_treatment" else None) and
                item.get("source_identity_sha256") == outcomes.get(
                    "source_identity_sha256") and
                item.get("source_tree_identity_sha256") == outcomes.get(
                    "source_tree_identity_sha256") and
                isinstance(item.get("policy"), str) and
                item.get("policy") in START_PROFILE_POLICY_IDS and
                item.get("checkpoint_sha256") == outcomes.get("checkpoint_sha256", {}).get(
                    item["policy"][:-3] if item["policy"].endswith("_P0") else
                    item["policy"]) and
                isinstance(outcomes.get("execution_device"), dict) and
                set(outcomes["execution_device"]) == {"device", "accelerator_backend"} and
                item.get("device") == outcomes["execution_device"] and
                item.get("execution_device") == outcomes["execution_device"] and
                item.get("simulation_device") ==
                    outcomes["execution_device"]["device"]
                for review_item, item in zip(replays, replay_payloads)))
    exact_review = (row.get("schema") ==
        "meco-phase4-start-profile-review-evidence-v1" and
        row.get("protocol_id") == protocol.get("protocol_id") and
        row.get("protocol_sha256") == protocol_sha256 and
        row.get("registry_sha256") == registry_sha256 and
        row.get("validation_seed_assignment_sha256") == seed_sha256 and
        row.get("registry_profile_id") == assignment.get("registry_profile_id") and
        row.get("profile_config_sha256") == assignment.get("profile_config_sha256") and
        row.get("assignment_sha256") == assignment.get("assignment_sha256") and
        row.get("seed_manifest_sha256") == outcomes.get("seed_manifest_sha256") and
        row.get("paired_outcomes_artifact_sha256") == outcome_sha and
        row.get("source_identity_sha256") == outcomes.get("source_identity_sha256") and
        row.get("checkpoint_sha256") == outcomes.get("checkpoint_sha256") and
        row.get("execution_device") == outcomes.get("execution_device") and
        row.get("condition_report_sha256") == outcomes.get("condition_report_sha256") and
        row.get("qualification_eligible") is False and
        row.get("dashboard_evidence_artifact_sha256") == dashboard_sha and
        row.get("dashboard_backend_source_identity_sha256") == dashboard.get(
            "backend_source_identity_sha256") and
        row.get("dashboard_backend_source_tree_identity_sha256") == dashboard_tree and
        row.get("endpoint_evidence_sha256") == endpoint_sha and
        isinstance(row.get("seed_manifest_sha256"), str) and
        len(row["seed_manifest_sha256"]) == 64 and
        row.get("frontend_asset_manifest_sha256") == manifest_sha and
        isinstance(replay, dict) and replay.get("schema") ==
        "meco-phase4-start-profile-replay-review-v1" and
        replay.get("accepted") is True and
        replay.get("protocol_id") == protocol.get("protocol_id") and
        replay.get("protocol_sha256") == protocol_sha256 and
        replay.get("registry_sha256") == registry_sha256 and
        replay.get("scenario_id") == case["scenario_id"] and
        replay.get("category") == case["category"] and
        replay.get("qualification_case") == case_axes and
        replay.get("split") == "validation" and
        replay.get("validation_seed_assignment_sha256") == seed_sha256 and
        replay.get("paired_outcomes_artifact_sha256") == outcome_sha and
        replay.get("registry_profile_id") == assignment.get("registry_profile_id") and
        replay.get("profile_config_sha256") == assignment.get("profile_config_sha256") and
        replay.get("assignment_sha256") == assignment.get("assignment_sha256") and
        replay.get("source_identity_sha256") == outcomes.get("source_identity_sha256") and
        replay.get("checkpoint_sha256") == outcomes.get("checkpoint_sha256") and
        replay.get("execution_device") == outcomes.get("execution_device") and
        replay.get("condition_report_sha256") == outcomes.get("condition_report_sha256") and
        replay.get("dashboard_evidence_artifact_sha256") == dashboard_sha and
        replay.get("dashboard_backend_source_identity_sha256") == dashboard.get(
            "backend_source_identity_sha256") and
        replay.get("dashboard_backend_source_tree_identity_sha256") == dashboard_tree and
        replay.get("endpoint_evidence_sha256") == endpoint_sha and
        replay.get("seed_manifest_sha256") == row.get("seed_manifest_sha256") and
        replay.get("frontend_asset_manifest_sha256") ==
            manifest_sha and
        replay.get("replay_checkout_source_identity_sha256") == dashboard_tree and
        type(replay.get("replay_seed")) is int and
        replay.get("replay_seed") in seed_values and
        isinstance(replays, list) and len(replays) == 2 and
        all(isinstance(item, dict) and
            isinstance(item.get("condition"), str) and
            item.get("condition") in CONDITIONS and
            type(item.get("seed")) is int for item in replays) and
        {item.get("condition") for item in replays if isinstance(item, dict)} ==
            set(CONDITIONS) and
        len({item.get("seed") for item in replays if isinstance(item, dict)}) == 1 and
        replay_artifacts_exact and all(isinstance(item, dict) and
            item.get("scenario_id") == case["scenario_id"] and
            item.get("category") == case["category"] and
            item.get("qualification_case") == case_axes and
            item.get("split") == "validation" and
            item.get("validation_seed_assignment_sha256") == seed_sha256 and
            item.get("registry_profile_id") == assignment.get("registry_profile_id") and
            item.get("profile_config_sha256") == assignment.get("profile_config_sha256") and
            item.get("assignment_sha256") == assignment.get("assignment_sha256") and
            item.get("source_identity_sha256") == outcomes.get("source_identity_sha256") and
            item.get("checkpoint_sha256") == outcomes.get("checkpoint_sha256") and
            item.get("execution_device") == outcomes.get("execution_device") and
            item.get("condition_report_sha256") == outcomes.get("condition_report_sha256") and
            item.get("dashboard_evidence_artifact_sha256") == dashboard_sha and
            item.get("dashboard_backend_source_identity_sha256") == dashboard.get(
                "backend_source_identity_sha256") and
            item.get("dashboard_backend_source_tree_identity_sha256") == dashboard_tree and
            item.get("endpoint_evidence_sha256") == endpoint_sha and
            item.get("seed_manifest_sha256") == row.get("seed_manifest_sha256") and
            item.get("frontend_asset_manifest_sha256") == manifest_sha and
            item.get("replay_checkout_source_identity_sha256") == dashboard_tree and
            item.get("seed") == replay.get("replay_seed") and
            isinstance(item.get("replay_id"), str) and bool(item["replay_id"]) and
            isinstance(item.get("replay_sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", item["replay_sha256"]) is not None
            for item in replays) and
        isinstance(reviewer, dict) and all(isinstance(reviewer.get(key), str) and
            reviewer.get(key) for key in ("subject_id", "identity_provider",
                                         "review_record_id")))
    if not exact_review:
        return {"accepted": False,
            "reason": "replay review/dashboard evidence is incomplete or mismatched"}
    return {"accepted": True, "evidence_sha256": _sha256(_canonical_bytes(row)),
        "paired_outcomes_artifact_sha256": outcome_sha,
        "replay_review_artifact_sha256": review_sha,
        "dashboard_evidence_artifact_sha256": dashboard_sha,
        "dashboard_backend_source_identity_sha256": dashboard.get(
            "backend_source_identity_sha256"),
        "dashboard_backend_source_tree_identity_sha256": dashboard_tree,
        "endpoint_evidence_sha256": endpoint_sha,
        "frontend_asset_manifest_sha256": manifest_sha,
        "registry_profile_id": assignment["registry_profile_id"],
        "qualification_eligible": False}


def _validate_seed_partition(protocol: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    """Validate selected assignments; protocol reader supplied validation rows only."""
    rows = protocol.get("cases")
    if not isinstance(rows, list):
        raise ValueError("protocol case table is missing")
    result: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != "validation":
            raise ValueError("selective protocol reader returned a non-validation case")
        seeds = row.get("seeds")
        if (not isinstance(seeds, list) or not seeds or
                any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds) or
                len(seeds) != len(set(seeds))):
            raise ValueError("validation case has malformed or duplicate seeds")
        if len(seeds) != VALIDATION_SEEDS_PER_CASE:
            raise ValueError("validation case does not have exactly 100 seeds")
        if any(not isinstance(row.get(axis), str) or not row[axis] for axis in CASE_AXES):
            raise ValueError("validation case has incomplete case axes")
        key = tuple(row[axis] for axis in CASE_AXES)
        if key in result:
            raise ValueError("duplicate validation case axis identity")
        result[key] = {"case": {axis: row[axis] for axis in CASE_AXES},
                       "seed_count": len(seeds), "seed_sha256": _seed_digest(seeds),
                       "_seed_values_internal": tuple(seeds)}
    return result


def compile_phase4_campaign_plan(protocol_path: str | Path,
                                registry_path: str | Path,
                                overlay_path: str | Path,
                                direct_pursuit_review_evidence_path: str | Path | None = None,
                                trench_interception_review_evidence_path: str | Path | None = None,
                                start_profile_review_evidence_path: str | Path | None = None
                                ) -> dict[str, Any]:
    """Return a compact, deterministic schedule plan, never episode rows."""
    protocol, protocol_raw = _read_protocol_validation(protocol_path)
    registry, registry_raw = _read_json(registry_path)
    overlay, overlay_raw = _read_json(overlay_path)
    direct_review_rows = []
    direct_review_raw_sha256 = None
    trench_review_rows = []
    trench_review_raw_sha256 = None
    trench_review_root = None
    start_profile_review_rows = []
    start_profile_review_raw_sha256 = None
    start_profile_review_root = None
    if direct_pursuit_review_evidence_path is not None:
        review_path = Path(direct_pursuit_review_evidence_path)
        review_raw = review_path.read_bytes()
        review_payload = json.loads(review_raw)
        if (not isinstance(review_payload, dict) or review_payload.get("schema") !=
                "meco-phase4-direct-pursuit-review-evidence-bundle-v1" or
                not isinstance(review_payload.get("cases"), list)):
            raise ValueError("invalid direct-pursuit replay-review evidence bundle")
        direct_review_rows = review_payload["cases"]
        direct_review_raw_sha256 = _sha256(review_raw)
    if trench_interception_review_evidence_path is not None:
        review_path = Path(trench_interception_review_evidence_path)
        review_raw = review_path.read_bytes()
        review_payload = json.loads(review_raw)
        if (not isinstance(review_payload, dict) or review_payload.get("schema") !=
                "meco-phase4-trench-interception-review-evidence-bundle-v1" or
                not isinstance(review_payload.get("cases"), list)):
            raise ValueError("invalid trench-interception replay-review evidence bundle")
        trench_review_rows = review_payload["cases"]
        trench_review_raw_sha256 = _sha256(review_raw)
        trench_review_root = review_path.resolve().parent
    if start_profile_review_evidence_path is not None:
        review_path = Path(start_profile_review_evidence_path)
        review_raw = review_path.read_bytes()
        review_payload = json.loads(review_raw)
        if (not isinstance(review_payload, dict) or review_payload.get("schema") !=
                "meco-phase4-start-profile-review-evidence-bundle-v1" or
                not isinstance(review_payload.get("cases"), list)):
            raise ValueError("invalid start-profile replay-review evidence bundle")
        start_profile_review_rows = review_payload["cases"]
        start_profile_review_raw_sha256 = _sha256(review_raw)
        start_profile_review_root = review_path.resolve().parent
    expected_overlay = compile_phase4_design_overlay(protocol_path, registry_path)
    if overlay != expected_overlay:
        raise ValueError("Phase 4 overlay does not exactly match the bound protocol and registry")
    if overlay.get("overlay_id") != _sha256(_canonical_bytes({
            key: value for key, value in overlay.items() if key != "overlay_id"})):
        raise ValueError("Phase 4 overlay ID digest mismatch")
    if overlay["protocol_binding"].get("protocol_raw_file_sha256") != _sha256(protocol_raw):
        raise ValueError("overlay protocol raw digest mismatch")
    if overlay["factor_registry_binding"].get("registry_raw_file_sha256") != _sha256(registry_raw):
        raise ValueError("overlay registry raw digest mismatch")

    pairing = registry.get("pairing", {})
    policies = pairing.get("same_checkpoint_controls")
    conditions = pairing.get("conditions")
    if policies != list(POLICIES) or conditions != list(CONDITIONS):
        raise ValueError("registry policy/condition grid differs from exact paired design")
    if pairing.get("seed_policy") != "inherit_validation_protocol_without_serializing_or_exporting_seed_values":
        raise ValueError("unsupported Phase 4 validation seed policy")

    validation = _validate_seed_partition(protocol)
    overlay_cases = overlay.get("validation_case_axes")
    overlay_case_map = {tuple(c.get(axis) for axis in CASE_AXES): c
                        for c in overlay_cases if isinstance(c, dict)}
    if len(overlay_case_map) != len(overlay_cases) or set(overlay_case_map) != set(validation):
        raise ValueError("overlay validation case inventory has missing, extra, or duplicate cases")

    registry_requirements = {r["requirement_id"]: r for r in registry["requirements"]}
    plan_requirements = []
    seen_cells: set[str] = set()
    planned_cells = 0
    total_rows = 0
    for compiled in overlay["requirements"]:
        req_id = compiled["requirement_id"]
        runtime_binding = phase4_requirement_binding(req_id)
        source_req = registry_requirements[req_id]
        factors = source_req["factor_ids"]
        selectors = source_req["validation_case_axes"]
        applicable = []
        for case_tuple, record in validation.items():
            case = record["case"]
            if not all(case.get(axis) in values
                       for axis, values in compiled["resolved_protocol_constraints"].items()):
                continue
            scope = compiled["implicit_scenario_scope"]
            if scope and case["scenario_id"] not in scope:
                continue
            applicable.append((case, record))
        expected_base = [{"factor_id": factor, **case}
                         for factor in factors for case, _ in applicable]
        if (len(expected_base) != len(compiled["protocol_bound_case_rows"]) or
                {_canonical_bytes(row) for row in expected_base} !=
                {_canonical_bytes(row) for row in compiled["protocol_bound_case_rows"]}):
            raise ValueError(f"{req_id} exact protocol-case coverage differs from overlay")

        design_domains: dict[str, list[Any]] = {}
        for axis, selector in selectors.items():
            if isinstance(selector, str) and selector.startswith("design_axis."):
                name = selector.removeprefix("design_axis.")
                domain = registry.get("design_axes", {}).get(name)
                if not isinstance(domain, list) or not domain or len(domain) != len(set(map(str, domain))):
                    raise ValueError(f"{req_id} has invalid design axis {name}")
                design_domains[name] = domain
        names = tuple(sorted(design_domains))
        design_products = (list(itertools.product(*(design_domains[n] for n in names)))
                           if names else [()])
        cells = []
        for factor in factors:
            for case, seed_info in applicable:
                for product in design_products:
                    design_axes = dict(zip(names, product))
                    reactive_assignment = None
                    direct_pursuit_assignment = None
                    direct_review_gate = None
                    trench_interception_assignment = None
                    trench_review_gate = None
                    start_profile_assignment = None
                    if req_id == "reactive_defenders_and_opponent_pools":
                        policy_id = design_axes.get("policy_pool")
                        if policy_id not in POLICY_CONFIG["policies"]:
                            raise ValueError("reactive policy cell has an unknown policy assignment")
                        config_sha256 = policy_pool_identity()["config_sha256"]
                        case_payload = {"scenario_id": case["scenario_id"],
                            "category": case["category"],
                            "qualification_case": {axis: case[axis] for axis in (
                                "layout_id", "defender_family", "start_condition",
                                "sensor_profile")},
                            "split": "validation",
                            "validation_seed_assignment_sha256": seed_info["seed_sha256"]}
                        reactive_assignment = {
                            "schema": "meco-phase4-opponent-schedule-assignment-v1",
                            "policy_id": policy_id,
                            "policy_family": POLICY_FACTOR_IDS[policy_id],
                            "configured_pressure_slots": (list(opponent_policy_runtime_slots(
                                case["category"])) if opponent_policy_runtime_domain_supported(
                                    case["scenario_id"], case["category"]) else []),
                            "policy_config_sha256": config_sha256,
                            "protocol_bound_evaluation_case": case_payload,
                            "stimulus_identity_sha256": _sha256(_canonical_bytes({
                                "case": case_payload, "factor_id": factor,
                                "policy_id": policy_id,
                                "policy_config_sha256": config_sha256})),
                            "factor_treatment_seeded_assignment_sha256": _sha256(
                                _canonical_bytes([
                                    policy_assignment_identity(policy_id=policy_id,
                                        scenario_id=case["scenario_id"],
                                        category=case["category"],
                                        qualification_case=case_payload["qualification_case"],
                                        seed=seed, condition="factor_treatment",
                                        factor_id=factor)
                                    for seed in seed_info["_seed_values_internal"]])),
                            "per_seed_identity_derivation": (
                                "sha256(canonical validation case, exact seed, factor, policy ID, config hash)"),
                            "policy_inputs": POLICY_CONFIG["sensing_boundary"],
                            "runtime_domain_supported": opponent_policy_runtime_domain_supported(
                                case["scenario_id"], case["category"]),
                            "planned_policy_rows_per_control_per_category": seed_info["seed_count"],
                            "control_policy_ids": list(POLICIES),
                            "pair_conditions": list(CONDITIONS),
                            "paired_condition_semantics": {
                                "no_factor": "registered category scripted opponent",
                                "factor_treatment": policy_id},
                            "paired_control_category_seed_coverage_sha256": _sha256(
                                _canonical_bytes([[control, case["category"], seed, condition]
                                    for control in POLICIES
                                    for seed in seed_info["_seed_values_internal"]
                                    for condition in CONDITIONS])),
                            "paired_control_category_seed_row_count": (
                                len(POLICIES) * seed_info["seed_count"] * len(CONDITIONS)),
                            "exact_pair_coverage": True,
                            "executed": False,
                            "qualification_eligible": False,
                        }
                    if req_id == "direct_pursuit":
                        if factor != "opponent.direct_pursuit":
                            raise ValueError("direct-pursuit requirement has an unexpected factor")
                        supported = opponent_policy_runtime_domain_supported(
                            case["scenario_id"], case["category"])
                        if supported:
                            case_payload = {"scenario_id": case["scenario_id"],
                                "category": case["category"],
                                "qualification_case": {axis: case[axis] for axis in (
                                    "layout_id", "defender_family", "start_condition",
                                    "sensor_profile")}, "split": "validation",
                                "validation_seed_assignment_sha256": seed_info["seed_sha256"]}
                            registered_pressure_slots = list(opponent_policy_runtime_slots(
                                case["category"]))
                            if not registered_pressure_slots:
                                raise ValueError("direct pursuit requires registered pressure slots")
                            assigned_pursuit_slot = registered_pressure_slots[0]
                            direct_pursuit_assignment = {
                                "schema": "meco-phase4-direct-pursuit-assignment-v1",
                                "policy_id": "direct_pursuit", "factor_id": factor,
                                "policy_config_sha256": policy_pool_identity()["config_sha256"],
                                "registered_pressure_slots": registered_pressure_slots,
                                "assigned_pursuit_slot": assigned_pursuit_slot,
                                "configured_pressure_slots": [assigned_pursuit_slot],
                                "preserved_scripted_slots": registered_pressure_slots[1:],
                                "control_policy_ids": list(DIRECT_PURSUIT_POLICY_IDS),
                                "paired_control_category_seed_coverage_sha256": _sha256(
                                    _canonical_bytes([[control, case["category"], seed, condition]
                                        for control in DIRECT_PURSUIT_POLICY_IDS
                                        for seed in seed_info["_seed_values_internal"]
                                        for condition in CONDITIONS])),
                                "protocol_bound_evaluation_case": case_payload,
                                "factor_treatment_seeded_assignment_sha256": _sha256(
                                    _canonical_bytes([{**policy_assignment_identity(
                                        policy_id="direct_pursuit",
                                        scenario_id=case["scenario_id"],
                                        category=case["category"],
                                        qualification_case=case_payload[
                                            "qualification_case"], seed=seed,
                                        condition="factor_treatment", factor_id=factor),
                                        "category": case["category"],
                                        "registered_pressure_slots": registered_pressure_slots,
                                        "configured_pressure_slots": [assigned_pursuit_slot],
                                        "preserved_scripted_slots": registered_pressure_slots[1:]}
                                        for seed in seed_info["_seed_values_internal"]])),
                                "runtime_domain_supported": True,
                                "paired_control_category_seed_row_count": (
                                    len(DIRECT_PURSUIT_POLICY_IDS) * seed_info["seed_count"] * len(CONDITIONS)),
                                "exact_pair_coverage": True,
                                "executed": False, "qualification_eligible": False,
                            }
                            direct_review_gate = _direct_pursuit_review_gate(
                                direct_review_rows, protocol=protocol,
                                protocol_sha256=_sha256(protocol_raw),
                                registry_sha256=_sha256(registry_raw), case=case,
                                seed_sha256=seed_info["seed_sha256"],
                                assignment_sha256=direct_pursuit_assignment[
                                    "factor_treatment_seeded_assignment_sha256"],
                                seed_count=seed_info["seed_count"],
                                configured_pressure_slots=direct_pursuit_assignment[
                                    "configured_pressure_slots"])
                    if (req_id == "trench_interception" and
                            factor == "opponent.trench_interception" and
                            case["scenario_id"] == "offense_vs_scripted_pressure" and
                            case["category"] == "active/trench/level_4_reactive" and
                            case["layout_id"] == "2026_rebuilt_shifted_structures_v1" and
                            case["defender_family"] == "orbiting_interceptor" and
                            case["start_condition"] == "offset_upper_lane" and
                            case["sensor_profile"] == "noisy" and
                            seed_info["seed_count"] == VALIDATION_SEEDS_PER_CASE):
                        if not trench_interception_runtime_domain_supported(
                                case["scenario_id"], case["category"]):
                            raise ValueError("frozen trench candidate is outside runtime domain")
                        case_payload = {"scenario_id": case["scenario_id"],
                            "category": case["category"], "qualification_case": {
                                axis: case[axis] for axis in ("layout_id", "defender_family",
                                    "start_condition", "sensor_profile")},
                            "split": "validation",
                            "validation_seed_assignment_sha256": seed_info["seed_sha256"]}
                        schedule = compile_opponent_policy_schedule(
                            policy_id="trench_interceptor", scenario_id=case["scenario_id"],
                            categories=(case["category"],),
                            qualification_case=case_payload["qualification_case"],
                            seeds=seed_info["_seed_values_internal"], split="validation",
                            factor_id=factor, protocol_id=protocol["protocol_id"])
                        if schedule["configured_pressure_slots"][case["category"]] != [0]:
                            raise ValueError("frozen trench candidate must replace only slot 0")
                        trench_interception_assignment = {
                            "schema": "meco-phase4-trench-interception-assignment-v1",
                            "policy_id": "trench_interceptor", "factor_id": factor,
                            "policy_family": schedule["policy_family"],
                            "policy_config_sha256": schedule["policy_config_sha256"],
                            "configured_pressure_slots": [0],
                            "control_policy_ids": list(TRENCH_POLICY_IDS),
                            "schedule": schedule,
                            "schedule_sha256": schedule["schedule_sha256"],
                            "protocol_bound_evaluation_case": case_payload,
                            "factor_treatment_seeded_assignment_sha256":
                                schedule["per_episode_assignment_sha256"],
                            "paired_condition_semantics": {
                                "no_factor": "registered level-4 reactive slot-0 script",
                                "factor_treatment": "trench_interceptor replaces slot 0"},
                            "paired_rows_per_condition": len(TRENCH_POLICY_IDS) * seed_info["seed_count"],
                            "exact_pair_coverage_required": True,
                            "runtime_domain_supported": True,
                            "executed": False, "qualification_eligible": False}
                        trench_review_gate = _trench_interception_review_gate(
                            trench_review_rows, protocol=protocol,
                            protocol_sha256=_sha256(protocol_raw),
                            registry_sha256=_sha256(registry_raw), case=case,
                            seed_sha256=seed_info["seed_sha256"],
                            assignment_sha256=schedule["per_episode_assignment_sha256"],
                            expected_schedule_sha256=schedule["schedule_sha256"],
                            seed_values=seed_info["_seed_values_internal"],
                            evidence_root=trench_review_root)
                    if req_id == "multiple_legal_team_start_states":
                        registry_profile_id = design_axes.get("legal_team_pose_sets")
                        start_profile_assignment = phase4_start_profile_assignment(
                            registry_profile_id,
                            scenario_id=case["scenario_id"], split="validation",
                            protocol_id=protocol["protocol_id"],
                            protocol_sha256=_sha256(protocol_raw),
                            qualification_case={axis: case[axis] for axis in (
                                "layout_id", "defender_family", "start_condition",
                                "sensor_profile")},
                            categories=(case["category"],),
                            seeds=seed_info["_seed_values_internal"])
                        start_profile_review_gate = _start_profile_review_gate(
                            start_profile_review_rows, protocol=protocol,
                            protocol_sha256=_sha256(protocol_raw),
                            registry_sha256=_sha256(registry_raw), case=case,
                            seed_sha256=seed_info["seed_sha256"],
                            assignment=start_profile_assignment,
                            seed_values=seed_info["_seed_values_internal"],
                            evidence_root=start_profile_review_root or Path.cwd())
                    else:
                        start_profile_review_gate = None
                    status = ("protocol_bound_policy_assignment_planned_not_executed"
                              if reactive_assignment is not None or direct_pursuit_assignment is not None or
                                trench_interception_assignment is not None else
                              "protocol_bound_start_profile_assignment_planned_not_executed"
                              if start_profile_assignment is not None else
                              "design_only_unassigned_to_protocol_case" if design_axes else
                              "planned_validation_not_executed")
                    cell = {
                        "requirement_id": req_id,
                        "factor_id": factor,
                        "case_axes": case,
                        "design_axes": design_axes,
                        "seed_count": seed_info["seed_count"],
                        "validation_seed_assignment_sha256": seed_info["seed_sha256"],
                        "planned_pair_rows_per_policy": 2 * seed_info["seed_count"],
                        "status": status,
                        "schedulable": True,
                        "runtime_factor_kind": runtime_binding[
                            "runtime_factor_by_factor_id"].get(factor),
                        "runtime_event": runtime_binding[
                            "runtime_event_by_factor_id"].get(factor),
                        "runtime_stimulus_supported": (
                            factor in runtime_binding["runtime_factor_by_factor_id"]
                            and runtime_binding["stimulus_supported"]),
                        "scheduler_assignment_supported": (
                            reactive_assignment is not None or
                            direct_pursuit_assignment is not None or
                            trench_interception_assignment is not None or
                            start_profile_assignment is not None),
                        "opponent_schedule_assignment": reactive_assignment,
                        "direct_pursuit_assignment": direct_pursuit_assignment,
                        "trench_interception_assignment": trench_interception_assignment,
                        "start_profile_assignment": start_profile_assignment,
                        "required_outcome_signals_complete": set(
                            runtime_binding["required_signals"]).issubset(
                                runtime_binding["authoritative_signals"]),
                        "executable": (not bool(design_axes) or
                            (req_id == "multiple_legal_team_start_states" and
                                set(design_axes) == {"legal_team_pose_sets"})
                            and (req_id != "direct_pursuit" or
                                bool(direct_review_gate and direct_review_gate.get("accepted")))
                            and (req_id != "trench_interception" or
                                bool(trench_interception_assignment and trench_review_gate and
                                    trench_review_gate.get("accepted")))
                            and (req_id != "multiple_legal_team_start_states" or
                                bool(start_profile_assignment and start_profile_review_gate and
                                    start_profile_review_gate.get("accepted")))
                            and factor in runtime_binding["runtime_factor_by_factor_id"]
                            and runtime_binding["stimulus_supported"]
                            and set(runtime_binding["required_signals"]).issubset(
                                runtime_binding["authoritative_signals"])),
                        "executed": False,
                        "simulator_behavior_validated": bool(
                            (direct_review_gate and direct_review_gate.get("accepted")) or
                            (trench_review_gate and trench_review_gate.get("accepted")) or
                            (start_profile_review_gate and
                                start_profile_review_gate.get("accepted"))),
                        "start_profile_review_gate": start_profile_review_gate,
                        "direct_pursuit_review_gate": direct_review_gate,
                        "trench_interception_review_gate": trench_review_gate,
                        "trench_interception_tranche_status": (
                            "review_accepted_candidate_cell_only" if
                            trench_review_gate and trench_review_gate.get("accepted") and
                                bool(trench_interception_assignment) else
                            "planned_not_executed_or_reviewed" if
                            trench_interception_assignment else None),
                    }
                    schedule_identity = {"factor_id": factor, "case_axes": case,
                                        "design_axes": design_axes}
                    schedule_key = _sha256(_canonical_bytes(schedule_identity))
                    cell["case_cell_key"] = _sha256(_canonical_bytes(cell))
                    if schedule_key in seen_cells:
                        raise ValueError("duplicate factor/case/design cell")
                    seen_cells.add(schedule_key)
                    cells.append(cell)
        count = len(cells)
        planned_cells += count
        req_rows = count * VALIDATION_SEEDS_PER_CASE * len(POLICIES) * len(CONDITIONS)
        total_rows += req_rows
        plan_requirements.append({
            "requirement_id": req_id,
            "factor_ids": factors,
            "applicable_protocol_case_count": len(applicable),
            "design_axis_domains": design_domains,
            "factor_case_axis_cell_count": count,
            "pairing_grid": [{"policy_id": p, "condition": c}
                             for p in POLICIES for c in CONDITIONS],
            "pairing_grid_sha256": _sha256(_canonical_bytes([
                {"policy_id": p, "condition": c} for p in POLICIES for c in CONDITIONS])),
            "pair_coverage": "exactly_one_control_and_one_treatment_per_validation_seed_policy_cell",
            "runtime_binding": runtime_binding,
            "scheduled_row_count_theoretical": req_rows,
            "cells": cells,
        })
    if planned_cells != EXPECTED_FACTOR_CASE_AXIS_CELLS:
        raise ValueError(f"factor/case/design-axis cell total {planned_cells} != 6488")
    expected_rows = planned_cells * VALIDATION_SEEDS_PER_CASE * len(POLICIES) * len(CONDITIONS)
    if total_rows != expected_rows or total_rows != 10_380_800:
        raise ValueError("theoretical schedule row count does not match exact paired coverage")
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "planned_not_executed",
        "qualification_eligible": False,
        "protocol_binding": {"protocol_id": protocol["protocol_id"],
                             "protocol_raw_file_sha256": _sha256(protocol_raw)},
        "registry_binding": {"registry_schema": registry["schema"],
                             "registry_raw_file_sha256": _sha256(registry_raw)},
        "overlay_binding": {"overlay_id": overlay["overlay_id"],
                            "overlay_raw_file_sha256": _sha256(overlay_raw)},
        "validation_seed_policy": "digest_only_exact_validation_case_assignment_no_seed_values",
        "validation_case_count": len(validation),
        "policies": list(POLICIES),
        "pair_conditions": list(CONDITIONS),
        "factor_case_axis_cell_count": planned_cells,
        "scheduled_row_count_theoretical": total_rows,
        "episode_rows_materialized": False,
        "final_test_rows_or_seed_values_included": False,
        "design_only_cells_executable": False,
        "executable_cell_count": sum(1 for requirement in plan_requirements
                                      for cell in requirement["cells"]
                                      if cell["executable"]),
        "runtime_capture_schema": "meco-phase4-realized-stimulus-capture-v1",
        "requirements": plan_requirements,
        "runtime_and_evaluator_wiring_required": True,
        "direct_pursuit_review_evidence_sha256": direct_review_raw_sha256,
        "trench_interception_review_evidence_sha256": trench_review_raw_sha256,
        "start_profile_review_evidence_sha256": start_profile_review_raw_sha256,
    }
    plan["plan_id"] = _sha256(_canonical_bytes(plan))
    return plan


def write_phase4_campaign_plan(protocol_path: str | Path, registry_path: str | Path,
                               overlay_path: str | Path,
                               output_path: str | Path,
                               direct_pursuit_review_evidence_path: str | Path | None = None,
                               trench_interception_review_evidence_path: str | Path | None = None,
                               start_profile_review_evidence_path: str | Path | None = None
                               ) -> dict[str, Any]:
    plan = compile_phase4_campaign_plan(protocol_path, registry_path, overlay_path,
        direct_pursuit_review_evidence_path=direct_pursuit_review_evidence_path,
        trench_interception_review_evidence_path=trench_interception_review_evidence_path,
        start_profile_review_evidence_path=start_profile_review_evidence_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--direct-pursuit-review-evidence", type=Path,
        help="reviewed held-out simulator/replay evidence required to open direct-pursuit executability")
    parser.add_argument("--trench-interception-review-evidence", type=Path,
        help="reviewed held-out paired trench evidence required to open the candidate cell gate")
    parser.add_argument("--start-profile-review-evidence", type=Path,
        help="reviewed per-profile paired outcomes and fresh replay evidence required to open one candidate cell")
    args = parser.parse_args(argv)
    plan = write_phase4_campaign_plan(args.protocol, args.registry, args.overlay, args.output,
        direct_pursuit_review_evidence_path=args.direct_pursuit_review_evidence,
        trench_interception_review_evidence_path=args.trench_interception_review_evidence,
        start_profile_review_evidence_path=args.start_profile_review_evidence)
    print(json.dumps({"plan_id": plan["plan_id"],
                      "factor_case_axis_cell_count": plan["factor_case_axis_cell_count"],
                      "scheduled_row_count_theoretical": plan["scheduled_row_count_theoretical"],
                      "episode_rows_materialized": plan["episode_rows_materialized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
