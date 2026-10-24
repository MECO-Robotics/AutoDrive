#!/usr/bin/env python3
"""Audit historical MECO exposure and export explicit actor-only transitions.

This tool is diagnostic and initialization-only. It never marks historical
training as qualification evidence and never resumes optimizer/RNG state.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.meco_candidates import CANDIDATE_BASE_FEATURE_COUNT
from frc_defense.curriculum_training import source_identity
from frc_defense.meco_supervised import _load_protocol
from frc_defense.meco_tracks import MECO_TRACKS, make_meco_track
from frc_defense.meco_track_training import ValueHead
from frc_defense.meco_track_training import load_evaluation_protocol
from frc_defense.meco_qualification import TRAIN_SEED_POOL_SIZE, protocol_cases
from frc_defense.meco_phase4 import validate_phase4_training_assignment


ARTIFACT_SCHEMA = "meco-weights-only-protocol-transition-v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _json(path: Path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"actor state {name} is not a tensor")
        cpu = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(_canonical(list(cpu.shape)))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _source_digest(sources):
    return _sha(json.dumps(sources, sort_keys=True, allow_nan=False).encode())


def _assignment_train_check(protocol, assignment):
    if not isinstance(assignment, dict):
        raise ValueError("scheduler history contains a malformed assignment")
    scenario, category, seed = (assignment.get("scenario_id"),
                                assignment.get("category"),
                                assignment.get("seed"))
    matches = [case for case in protocol.get("cases", [])
        if case.get("scenario_id") == scenario and case.get("category") == category
        and case.get("split") == "train"]
    if not matches or type(seed) is not int or not any(seed in row.get("seeds", [])
                                                       for row in matches):
        raise ValueError("scheduler assignment seed is not in the exact train split")
    if any(seed in row.get("seeds", []) for row in protocol.get("cases", [])
           if row.get("split") in ("validation", "test")):
        raise ValueError("scheduler assignment seed overlaps validation/test")


def _validate_scheduler_assignment(protocol, scheduler, assignment):
    master_seed = scheduler.get("master_seed")
    if type(master_seed) is not int or protocol.get("master_seed") != master_seed:
        raise ValueError("scheduler master seed differs from frozen protocol")
    train_case = protocol_cases(assignment["scenario_id"], "train",
        episodes=TRAIN_SEED_POOL_SIZE, master_seed=master_seed)[0]
    expected_pool = _sha(json.dumps({"scenario_id": assignment["scenario_id"],
        "seeds": list(train_case.seeds)}, sort_keys=True, allow_nan=False).encode())
    if assignment.get("training_seed_pool_id") != expected_pool:
        raise ValueError("scheduler assignment training seed pool identity is invalid")
    payload = {"cycle": assignment["cycle"],
        "scenario_id": assignment["scenario_id"], "category": assignment["category"],
        "seed": assignment["seed"], "master_seed": master_seed,
        "training_seed_pool_id": expected_pool}
    if assignment.get("phase4_factor") is not None:
        validate_phase4_training_assignment(assignment)
        payload.update(phase4_training_version=assignment["phase4_training_version"],
            phase4_factor=assignment["phase4_factor"],
            phase4_split=assignment["phase4_split"],
            phase4_training_assignment_id=assignment["phase4_training_assignment_id"],
            phase4_training_protocol_id=assignment["phase4_training_protocol_id"])
    assignment_id = _sha(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    if assignment.get("assignment_id") != assignment_id:
        raise ValueError("scheduler assignment identity does not bind its seed/task fields")


def _assert_supervisor_quiescent(root: Path):
    status_path = root / "supervisor-status.json"
    if not status_path.is_file():
        raise ValueError("supervisor terminal status is missing")
    status = _json(status_path)
    if status.get("status") not in ("complete", "stopped", "failed"):
        raise ValueError("supervisor status is not terminal")
    lanes = status.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != len(MECO_TRACKS):
        raise ValueError("supervisor terminal status lacks all five lane records")
    if {lane.get("track") for lane in lanes} != set(MECO_TRACKS):
        raise ValueError("supervisor terminal status has duplicate or missing lanes")
    if any(lane.get("pid") is not None for lane in lanes):
        raise ValueError("supervisor status still records a worker PID")

    queue_path = root / "queue-status.json"
    if not queue_path.is_file():
        raise ValueError("supervisor queue status is missing; cannot inventory evaluations")
    queue = _json(queue_path)
    if (queue.get("phase") not in ("complete", "stopped", "failed")
            or queue.get("evaluation", {}).get("active")
            or queue.get("evaluation", {}).get("pending", 0) != 0):
        raise ValueError("supervisor queue or evaluator is not terminal and drained")

    lock_path = root / ".supervisor.lock"
    if not lock_path.is_file():
        raise ValueError("supervisor lock file is missing; cannot establish quiescence")
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise ValueError("supervisor output lock is still held") from exc

    proc_root = Path("/proc")
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    errors="replace")
            except OSError:
                continue
            if str(root) in command and ("meco_supervisor" in command or
                                         "meco_track_training" in command):
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                raise ValueError("an MECO supervisor or trainer process still references the segment")
    return fd


def _validate_evaluation_report(report, checkpoints,
                                *, evaluation_protocol_path, queue_status,
                                supervisor_status, status):
    split = report.get("split")
    scenario = report.get("scenario_id")
    categories = report.get("categories")
    seeds = report.get("seeds")
    records = report.get("records")
    if (not isinstance(scenario, str) or not isinstance(categories, list)
            or not categories or not isinstance(seeds, list) or not seeds
            or not isinstance(records, list) or not records):
        raise ValueError("evaluation lacks a recognized case/seed manifest")
    if split == "diagnostic":
        if (report.get("evaluation_design") != "diagnostic_unbound"
                or report.get("evaluation_protocol_id") is not None
                or report.get("evaluation_protocol_sha256") is not None
                or report.get("evaluation_case") is not None
                or report.get("exact_episode_seeds") is not None
                or queue_status.get("evaluation_split") != "diagnostic"
                or supervisor_status.get("evaluation_split") != "diagnostic"
                or any(payload.get(key) is not None for payload in
                    (queue_status, supervisor_status) for key in (
                        "evaluation_protocol_id", "evaluation_protocol_sha256",
                        "evaluation_binding", "exact_episode_seeds"))
                or status.get("evaluation_protocol_id") is not None
                or status.get("evaluation_protocol_sha256") is not None
                or status.get("evaluation_case") is not None
                or status.get("exact_episode_seeds") is not None):
            raise ValueError("unbound evaluation is not explicitly diagnostic and nonqualifying")
        for record in records:
            if (not isinstance(record, dict) or record.get("split") != "diagnostic"
                    or record.get("evaluation_protocol_id") is not None
                    or record.get("evaluation_protocol_sha256") is not None
                    or record.get("evaluation_case_id") is not None):
                raise ValueError("diagnostic record carries held-out protocol provenance")
        return
    if split not in ("validation", "test"):
        raise ValueError("evaluation split is neither diagnostic nor a recognized held-out split")
    if evaluation_protocol_path is None:
        raise ValueError("held-out transition audit requires --evaluation-protocol")
    try:
        eval_protocol, eval_id, eval_sha, _ = load_evaluation_protocol(
            evaluation_protocol_path, scenario_id=scenario)
    except Exception as exc:
        raise ValueError(f"cannot validate frozen evaluation protocol: {exc}") from exc
    binding = report.get("evaluation_case")
    queue_binding = queue_status.get("evaluation_binding")
    status_binding = supervisor_status.get("evaluation_binding")
    expected_binding = queue_binding
    if (report.get("evaluation_design") != "frozen_protocol_held_out"
            or report.get("evaluation_protocol_id") != eval_id
            or report.get("evaluation_protocol_sha256") != eval_sha
            or status.get("evaluation_protocol_id") != eval_id
            or status.get("evaluation_protocol_sha256") != eval_sha
            or status.get("evaluation_case") != binding
            or status.get("qualification_protocol_id") != report.get("qualification_protocol_id")
            or report.get("evaluation_protocol_id") != queue_status.get("evaluation_protocol_id")
            or report.get("evaluation_protocol_sha256") != queue_status.get("evaluation_protocol_sha256")
            or report.get("evaluation_protocol_id") != supervisor_status.get("evaluation_protocol_id")
            or report.get("evaluation_protocol_sha256") != supervisor_status.get("evaluation_protocol_sha256")
            or report.get("split") != queue_status.get("evaluation_split")
            or report.get("split") != supervisor_status.get("evaluation_split")
            or report.get("split") != status.get("split")
            or not isinstance(binding, dict) or not isinstance(expected_binding, dict)
            or not isinstance(status_binding, dict)
            or binding.get("protocol_id") != eval_id
            or binding.get("protocol_sha256") != eval_sha
            or binding.get("split") != split
            or binding.get("scenario_id") != scenario
            or binding.get("categories") != categories
            or binding.get("seeds") != seeds
            or report.get("exact_episode_seeds") != seeds
            or queue_status.get("exact_episode_seeds") != seeds
            or supervisor_status.get("exact_episode_seeds") != seeds
            or status.get("exact_episode_seeds") != seeds
            or binding.get("protocol_id") != queue_binding.get("protocol_id")
            or binding.get("protocol_sha256") != queue_binding.get("protocol_sha256")
            or binding.get("split") != queue_binding.get("split")
            or binding.get("scenario_id") != queue_binding.get("scenario_id")
            or binding.get("categories") != queue_binding.get("categories")
            or binding.get("qualification_case") != queue_binding.get("qualification_case")
            or binding.get("seeds") != queue_binding.get("seeds")
            or queue_status.get("evaluation_case") != binding
            or supervisor_status.get("evaluation_case") != binding
            or status_binding != queue_binding):
        raise ValueError("held-out evaluation protocol/case/seeds/split disagree with supervisor or protocol bytes")
    case = binding.get("qualification_case")
    if not isinstance(case, dict):
        raise ValueError("held-out evaluation case stratum is malformed")
    # Training qualification provenance is retained as its own field. It is
    # deliberately not inferred from, or substituted for, evaluation identity.
    try:
        from frc_defense.meco_track_training import resolve_evaluation_protocol
        verified_binding = resolve_evaluation_protocol(evaluation_protocol_path,
            scenario_id=scenario, categories=tuple(categories), split=split,
            episode_seeds=seeds, qualification_case=case)
    except Exception as exc:
        raise ValueError(f"held-out evaluation case is invalid for frozen protocol: {exc}") from exc
    if verified_binding != binding:
        raise ValueError("report evaluation case binding differs from frozen protocol resolution")
    protocol_cases = [row for row in eval_protocol.get("cases", [])
                      if row.get("scenario_id") == scenario and row.get("split") == split]
    if not protocol_cases:
        raise ValueError("evaluation scenario/split is absent from the checked protocol")
    if len(set(categories)) != len(categories) or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation top-level case has duplicate categories or seeds")
    allowed_by_case = {(row.get("category"), row.get("layout_id"),
        row.get("defender_family"), row.get("start_condition"), row.get("sensor_profile")):
        set(row.get("seeds", [])) for row in protocol_cases}
    used = set()
    policy_cases = set()
    for record in records:
        if (not isinstance(record, dict) or record.get("scenario_id") != scenario
                or record.get("split") != split
                or record.get("evaluation_protocol_id") != eval_id
                or record.get("evaluation_protocol_sha256") != eval_sha
                or record.get("qualification_protocol_id") != report.get("qualification_protocol_id")
                or record.get("evaluation_case_id") != binding.get("case_ids")
                or record.get("source_identity_sha256") != report.get("source_identity_sha256")):
            raise ValueError("evaluation record has missing or inconsistent protocol provenance")
        category, seed = record.get("category"), record.get("seed")
        if category not in categories or seed not in seeds or type(seed) is not int:
            raise ValueError("evaluation record category/seed is absent from its exact manifest")
        case_key = (category, record.get("layout_id"), record.get("defender_family"),
                    record.get("start_condition"), record.get("sensor_profile"))
        allowed = allowed_by_case.get(case_key)
        if allowed is None or seed not in allowed:
            raise ValueError("evaluation record has unknown stratum or seed for its declared split")
        if any(seed in row.get("seeds", []) for row in eval_protocol.get("cases", [])
               if row.get("scenario_id") == scenario and row.get("category") == category
               and row.get("split") != split):
            raise ValueError("evaluation seed leaks across train/validation/test splits")
        for field in ("layout_id", "defender_family", "start_condition", "sensor_profile"):
            if case.get(field) != record.get(field):
                raise ValueError("evaluation record stratum differs from report evaluation case")
        policy = record.get("policy")
        if not isinstance(policy, str) or not policy:
            raise ValueError("evaluation record has no policy identity")
        base_policy = next((track for track in MECO_TRACKS
                            if policy == track or policy.startswith(track + "_")
                            or policy.startswith(track + "__")), policy)
        item = checkpoints.get(base_policy)
        if item is None or record.get("checkpoint_sha256") != item.get("sha256"):
            # Controls may be present in a matched evaluation but have no learned checkpoint.
            if policy in ("D0", "D1", "D2") and record.get("checkpoint_sha256") is None:
                pass
            else:
                raise ValueError("evaluation record does not bind a recognized exact checkpoint")
        uniqueness = (policy, category, seed, *case_key[1:])
        if uniqueness in policy_cases:
            raise ValueError("evaluation contains duplicate policy/category/seed/stratum rows")
        policy_cases.add(uniqueness)
        used.add((category, seed, *case_key[1:]))
    expected_category_seeds = {(category, seed) for category in categories for seed in seeds}
    observed_category_seeds = {(category, key_seed) for category, key_seed, *_ in used}
    if observed_category_seeds != expected_category_seeds:
        raise ValueError("evaluation records do not cover the declared exact category/seed set")
    expected_policy_cases = {(track, category, seed) for track in MECO_TRACKS
                             for category, seed in expected_category_seeds}
    observed_policy_cases = {(policy, category, seed) for policy, category, seed, *_ in policy_cases
                             if policy in MECO_TRACKS}
    if observed_policy_cases != expected_policy_cases:
        raise ValueError("evaluation does not contain every MECO actor for each declared category/seed")


def _validate_evaluation_snapshots(evaluation_root, report_path, report, history,
                                   latest_checkpoints, sources, shared_config):
    match = re.fullmatch(r"cycle-(\d{6})-[0-9a-f]{8}\.json", report_path.name)
    if match is None:
        raise ValueError("evaluation filename does not identify a supervisor checkpoint cycle")
    cycle = int(match.group(1))
    if cycle < 1 or cycle > len(history):
        raise ValueError("evaluation cycle is outside completed scheduler history")
    assignment = history[cycle - 1]["assignment"]
    target_update = assignment["target_update"]
    manifest = report.get("policy_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("evaluation lacks policy checkpoint manifest")
    snapshot_dir = evaluation_root / "checkpoint-snapshots" / report_path.stem
    snapshot_identities = {}
    for track in MECO_TRACKS:
        snapshot_path = snapshot_dir / f"{track}.pt"
        if not snapshot_path.is_file():
            raise ValueError(f"evaluation is missing its {track} checkpoint snapshot")
        digest = _sha(snapshot_path.read_bytes())
        payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        policy = manifest.get(track)
        if (not isinstance(policy, dict) or policy.get("checkpoint_sha256") != digest
                or payload.get("update") != target_update
                or payload.get("task_assignment") != assignment
                or payload.get("run_id") != latest_checkpoints[track]["run_id"]
                or payload.get("sources") != sources):
            raise ValueError(f"evaluation {track} snapshot does not match its scheduler cycle")
        config = payload.get("config")
        if not isinstance(config, dict) or config.get("track") != track:
            raise ValueError(f"evaluation {track} snapshot has malformed training configuration")
        immutable = {key: value for key, value in config.items() if key != "updates"}
        config_digest = _sha(json.dumps(immutable, sort_keys=True, allow_nan=False).encode())
        normalized = {key: value for key, value in config.items()
                      if key not in ("track", "device", "output", "updates")}
        if (config_digest != policy.get("training_configuration_sha256")
                or normalized != shared_config
                or policy.get("training_protocol_id") != payload.get("protocol_id")):
            raise ValueError(f"evaluation {track} snapshot configuration/protocol mismatch")
        snapshot_identities[track] = {"path": str(snapshot_path), "sha256": digest,
            "run_id": payload.get("run_id"), "update": payload.get("update"),
            "protocol_id": payload.get("protocol_id"),
            "configuration_sha256": config_digest,
            "source_identity_sha256": _source_digest(sources)}
    return cycle, snapshot_identities


def _evaluation_inventory(evaluation_root: Path, queue_status_path: Path):
    root = Path(evaluation_root).resolve()
    if not root.is_dir():
        raise ValueError("an explicit existing evaluation root is required for complete inventory")
    reports = sorted(path for path in root.glob("*.json")
                     if not path.name.endswith(".status.json")
                     and path.name != "evaluation-history.json")
    statuses = sorted(root.glob("*.json.status.json"))
    status_by_report = {path.with_suffix("").with_suffix(""): path for path in statuses}
    if set(reports) != set(status_by_report):
        raise ValueError("evaluation root contains a report without status or orphan status file")
    history_path = root / "evaluation-history.json"
    if not history_path.is_file():
        raise ValueError("legacy evaluation root has no durable history index; completeness is unverifiable")
    history_bytes = history_path.read_bytes()
    history = json.loads(history_bytes)
    history_rows = history.get("entries") if history.get("schema_version") == 1 else None
    if not isinstance(history_rows, list):
        raise ValueError("durable evaluation history index is malformed")
    indexed = {}
    previous = None
    for sequence, row in enumerate(history_rows, 1):
        if not isinstance(row, dict):
            raise ValueError("durable evaluation history row is malformed")
        digest = row.get("entry_sha256")
        body = {key: value for key, value in row.items() if key != "entry_sha256"}
        row_digest = _sha(json.dumps(body, sort_keys=True, allow_nan=False).encode())
        if (row.get("sequence") != sequence
                or row.get("previous_entry_sha256") != previous
                or digest != row_digest):
            raise ValueError("durable evaluation history chain is broken")
        previous = digest
        name = row.get("output_file")
        if not isinstance(name, str) or name in indexed:
            raise ValueError("durable evaluation history has duplicate or malformed output")
        indexed[name] = row
    try:
        queue = _json(Path(queue_status_path))
        anchor = queue.get("evaluation_history") or {}
    except (OSError, ValueError, json.JSONDecodeError):
        raise ValueError("supervisor queue lacks a durable evaluation history anchor")
    if (anchor.get("count") != len(history_rows)
            or anchor.get("head_sha256") != previous):
        raise ValueError("durable evaluation history differs from supervisor queue anchor")
    if set(indexed) != {path.name for path in reports}:
        raise ValueError("durable evaluation history does not cover the complete report inventory")
    entries = []
    for report_path in reports:
        status_path = status_by_report[report_path]
        report = _json(report_path)
        status = _json(status_path)
        if status.get("status") != "complete":
            raise ValueError(f"evaluation report is not terminal-complete: {report_path.name}")
        if (status.get("evaluation_id") != report.get("evaluation_id")
                or status.get("protocol_id") != report.get("protocol_id")
                or status.get("output_file") != report_path.name):
            raise ValueError(f"evaluation report/status identity mismatch: {report_path.name}")
        report_sha, status_sha = _sha(report_path.read_bytes()), _sha(status_path.read_bytes())
        history_row = indexed[report_path.name]
        if (history_row.get("status") != "complete"
                or history_row.get("evaluation_id") != report.get("evaluation_id")
                or history_row.get("protocol_id") != report.get("protocol_id")
                or history_row.get("report_sha256") != report_sha
                or history_row.get("status_file") != status_path.name
                or history_row.get("status_sha256") != status_sha):
            raise ValueError(f"evaluation report/status disagrees with durable history: {report_path.name}")
        snapshot_dir = root / "checkpoint-snapshots" / report_path.stem
        if (not snapshot_dir.is_dir()
                or any(path.is_dir() for path in snapshot_dir.rglob("*"))):
            raise ValueError(f"evaluation checkpoint snapshot directory is malformed: {report_path.name}")
        snapshots = sorted(path for path in snapshot_dir.rglob("*") if path.is_file())
        if any(path.is_symlink() for path in snapshots):
            raise ValueError(f"evaluation checkpoint snapshot uses a symlink: {report_path.name}")
        snapshot_rows = [{"path": str(path.resolve()), "sha256": _sha(path.read_bytes())}
            for path in snapshots]
        indexed_snapshots = history_row.get("checkpoint_snapshots")
        indexed_snapshot_rows = sorted(({"path": row.get("path"), "sha256": row.get("sha256")}
            for row in indexed_snapshots or []), key=lambda row: row["path"] or "")
        if (len(snapshot_rows) != len(MECO_TRACKS)
                or {Path(row["path"]).name for row in snapshot_rows} !=
                    {f"{track}.pt" for track in MECO_TRACKS}
                or snapshot_rows != indexed_snapshot_rows):
            raise ValueError(f"evaluation checkpoint snapshot inventory disagrees with durable history: {report_path.name}")
        entries.append({"path": str(report_path), "sha256": report_sha,
            "status_path": str(status_path), "status_sha256": status_sha,
            "snapshot_inventory": snapshot_rows,
            "evaluation_id": report.get("evaluation_id"), "report": report,
            "status": status})
    inventory_digest = _sha(_canonical({"history_sha256": _sha(history_bytes),
        "entries": [{key: value for key, value in row.items() if key != "report"}
                    for row in entries]}))
    return root, entries, inventory_digest


def audit_segment(segment_dir: Path, qualification_protocol: Path,
                  evaluation_root: Path, evaluation_protocol: Path | None = None) -> dict:
    root = Path(segment_dir).resolve()
    lock_fd = _assert_supervisor_quiescent(root)
    try:
        return _audit_segment_locked(root, qualification_protocol, evaluation_root,
                                     evaluation_protocol)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _audit_segment_locked(segment_dir: Path, qualification_protocol: Path,
                           evaluation_root: Path,
                           evaluation_protocol: Path | None = None) -> dict:
    """Reconcile scheduler history, all five lane frontiers, and report hashes."""
    root = Path(segment_dir).resolve()
    protocol, protocol_id, protocol_sha, _ = _load_protocol(Path(qualification_protocol))
    scheduler = _json(root / "task-scheduler-state.json")
    if scheduler.get("pending_assignment") is not None:
        raise ValueError("segment has a pending assignment; frontier is not quiescent")
    completed = scheduler.get("completed_cycle")
    history = scheduler.get("history")
    if type(completed) is not int or completed < 1 or not isinstance(history, list) or len(history) != completed:
        raise ValueError("scheduler completed-cycle/history frontier is inconsistent")
    for assignment_row in history:
        if not isinstance(assignment_row, dict):
            raise ValueError("scheduler history row is malformed")
        _validate_scheduler_assignment(protocol, scheduler, assignment_row.get("assignment"))

    previous_target = 0
    assignment_frontiers = []
    for index, row in enumerate(history, 1):
        if not isinstance(row, dict):
            raise ValueError("scheduler history row is malformed")
        assignment = row.get("assignment")
        if assignment.get("cycle") != index:
            raise ValueError("scheduler cycles are missing, duplicated, or out of order")
        _assignment_train_check(protocol, assignment)
        target = assignment.get("target_update")
        if type(target) is not int or target <= previous_target:
            raise ValueError("scheduler target-update history is invalid")
        previous_target = target
        assignment_frontiers.append((target, assignment))
        results = row.get("lane_results")
        if not isinstance(results, dict) or set(results) != set(MECO_TRACKS):
            raise ValueError("scheduler cycle does not contain exactly five lane results")
        for track in MECO_TRACKS:
            result = results[track]
            if (not isinstance(result, dict)
                    or result.get("assignment_id") != assignment.get("assignment_id")
                    or result.get("cycle") != index
                    or result.get("scenario_id") != assignment.get("scenario_id")
                    or result.get("category") != assignment.get("category")
                    or ("seed" in result and result.get("seed") != assignment.get("seed"))
                    or result.get("completed_updates") != target):
                raise ValueError(f"{track} scheduler outcome disagrees with assignment frontier")
            counters = [result.get(name) for name in (
                "episodes", "successes", "failures", "aborted_no_legal_support")]
            if (any(type(value) is not int or value < 0 for value in counters)
                    or result["successes"] + result["failures"] != result["episodes"]):
                raise ValueError(f"{track} scheduler outcome counters are invalid")

    source_value = None
    shared_config = None
    common_protocol = None
    checkpoints = {}
    latest_assignment = history[-1]["assignment"]
    for track in MECO_TRACKS:
        lane = root / track
        checkpoint_path, run_path = lane / "latest.pt", lane / "run.json"
        if not checkpoint_path.is_file() or not run_path.is_file():
            raise ValueError(f"missing {track} checkpoint or run manifest")
        manifest = _json(run_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"{track} checkpoint is malformed")
        if (manifest.get("track") != track or manifest.get("status") != "complete"
                or manifest.get("completed_updates") != previous_target
                or checkpoint.get("update") != previous_target):
            raise ValueError(f"{track} run/checkpoint update frontier is not the common completed frontier")
        if (manifest.get("run_id") != checkpoint.get("run_id")
                or manifest.get("protocol_id") != checkpoint.get("protocol_id")
                or manifest.get("task_assignment") != latest_assignment
                or checkpoint.get("task_assignment") != latest_assignment):
            raise ValueError(f"{track} run/checkpoint identity disagrees with scheduler frontier")
        progress = checkpoint.get("task_progress")
        if (not isinstance(progress, dict)
                or progress.get("assignment_id") != latest_assignment.get("assignment_id")
                or progress.get("cycle") != completed):
            raise ValueError(f"{track} checkpoint task progress disagrees with scheduler")
        if manifest.get("sources") != checkpoint.get("sources"):
            raise ValueError(f"{track} run/checkpoint source manifests disagree")
        sources = checkpoint.get("sources")
        if not isinstance(sources, dict):
            raise ValueError(f"{track} source identity is missing")
        digest = _source_digest(sources)
        if manifest.get("source_identity_sha256") != digest:
            raise ValueError(f"{track} source identity digest is invalid")
        if source_value is None:
            source_value = sources
        elif source_value != sources:
            raise ValueError("matched lane checkpoints have different full source identities")
        config = checkpoint.get("config")
        if not isinstance(config, dict) or config.get("track") != track:
            raise ValueError(f"{track} checkpoint config is malformed")
        if manifest.get("config") != config:
            raise ValueError(f"{track} run/checkpoint configurations disagree")
        config_digest = _sha(json.dumps({key: value for key, value in config.items()
            if key != "updates"}, sort_keys=True, allow_nan=False).encode())
        if manifest.get("configuration_sha256") != config_digest:
            raise ValueError(f"{track} immutable configuration digest is invalid")
        if common_protocol is None:
            common_protocol = checkpoint.get("protocol_id")
        elif checkpoint.get("protocol_id") != common_protocol:
            raise ValueError("matched lane checkpoints have different supervisor protocol identities")
        normalized = {key: value for key, value in config.items()
                      if key not in ("track", "device", "output", "updates")}
        if shared_config is None:
            shared_config = normalized
        elif shared_config != normalized:
            raise ValueError("matched lane checkpoints have different shared configurations")
        lane_history = checkpoint.get("history")
        if (not isinstance(lane_history, list) or len(lane_history) != previous_target
                or [item.get("update") for item in lane_history] != list(range(1, previous_target + 1))):
            raise ValueError(f"{track} checkpoint history does not cover the common update frontier")
        for update_index, item in enumerate(lane_history, 1):
            expected_assignment = next(assignment for frontier, assignment in assignment_frontiers
                                       if update_index <= frontier)
            if item.get("task_assignment_id") != expected_assignment.get("assignment_id"):
                raise ValueError(f"{track} update history disagrees with scheduler assignments")
        expected_outcome = history[-1]["lane_results"][track]
        if manifest.get("task_outcomes") != expected_outcome:
            raise ValueError(f"{track} run outcome disagrees with scheduler history")
        for key in ("assignment_id", "cycle", "scenario_id", "category", "episodes",
                    "successes", "failures", "aborted_no_legal_support"):
            if key in expected_outcome and progress.get(key) != expected_outcome[key]:
                raise ValueError(f"{track} checkpoint task progress disagrees with scheduler outcome")
        model_state = checkpoint.get("model")
        if not isinstance(model_state, dict) or not model_state:
            raise ValueError(f"{track} checkpoint has no policy actor weights")
        checkpoint_digest = _sha(checkpoint_path.read_bytes())
        checkpoints[track] = {"path": str(checkpoint_path),
            "sha256": checkpoint_digest, "run_id": checkpoint["run_id"],
            "update": previous_target, "protocol_id": checkpoint.get("protocol_id"),
            "configuration_sha256": config_digest,
            "actor_state_sha256": _state_digest(model_state),
            "actor_keys": sorted(model_state)}

    evaluation_root, inventory_entries, inventory_sha = _evaluation_inventory(
        evaluation_root, root / "queue-status.json")
    # queue-status is the durable evaluation state; supervisor-status may only
    # contain the dispatch snapshot from its final loop.
    queue_status = _json(root / "queue-status.json")
    supervisor_status = _json(root / "supervisor-status.json")
    supervisor_evaluation = queue_status.get("evaluation") or {}
    latest_evaluation = supervisor_evaluation.get("output_file")
    inventoried_names = {Path(row["path"]).name for row in inventory_entries}
    if latest_evaluation is None:
        raise ValueError("terminal supervisor has no evaluation inventory anchor")
    if latest_evaluation not in inventoried_names:
        raise ValueError("evaluation root inventory is not anchored to supervisor's terminal evaluation record")
    terminal_evaluation = supervisor_status.get("evaluation") or {}
    if (terminal_evaluation.get("output_file") != latest_evaluation
            or terminal_evaluation.get("evaluation_id") != supervisor_evaluation.get("evaluation_id")
            or terminal_evaluation.get("protocol_id") != supervisor_evaluation.get("protocol_id")
            or terminal_evaluation.get("status") != supervisor_evaluation.get("status")):
        raise ValueError("terminal supervisor evaluation record disagrees with durable queue status")
    if latest_evaluation is not None:
        latest_entry = next(row for row in inventory_entries
                            if Path(row["path"]).name == latest_evaluation)
        if (supervisor_evaluation.get("status") != "complete"
                or supervisor_evaluation.get("evaluation_id") != latest_entry["evaluation_id"]
                or supervisor_evaluation.get("protocol_id") !=
                    latest_entry["report"].get("protocol_id")):
            raise ValueError("supervisor terminal evaluation record does not match latest report/status")
    eval_index = []
    for entry in inventory_entries:
        report_path = Path(entry["path"])
        report = entry.pop("report")
        evaluation_status = entry.pop("status")
        if report.get("source_identity_sha256") != _source_digest(source_value):
            raise ValueError(f"evaluation source identity mismatch: {report_path}")
        evaluation_cycle, evaluation_checkpoints = _validate_evaluation_snapshots(
            evaluation_root, report_path, report, history, checkpoints,
            source_value, shared_config)
        _validate_evaluation_report(report, evaluation_checkpoints,
                                    evaluation_protocol_path=evaluation_protocol,
                                    queue_status=queue_status,
                                    supervisor_status=supervisor_status,
                                    status=evaluation_status)
        policy_manifest = report.get("policy_manifest")
        if not isinstance(policy_manifest, dict):
            raise ValueError(f"evaluation lacks policy manifest: {report_path}")
        for track in MECO_TRACKS:
            item = policy_manifest.get(track)
            if (not isinstance(item, dict) or
                    item.get("checkpoint_sha256") != evaluation_checkpoints[track]["sha256"]):
                raise ValueError(f"evaluation does not bind exact {track} checkpoint: {report_path}")
        eval_index.append({"path": str(report_path), "sha256": entry["sha256"],
            "status_path": entry["status_path"], "status_sha256": entry["status_sha256"],
            "evaluation_id": report.get("evaluation_id"),
            "protocol_id": report.get("protocol_id"),
            "evaluation_protocol_id": report.get("evaluation_protocol_id"),
            "evaluation_protocol_sha256": report.get("evaluation_protocol_sha256"),
            "evaluation_case": report.get("evaluation_case"),
            "exact_episode_seeds": report.get("exact_episode_seeds"),
            "split": report.get("split"),
            "training_qualification_protocol_id": report.get("qualification_protocol_id"),
            "cycle": evaluation_cycle, "checkpoint_snapshots": evaluation_checkpoints,
            "snapshot_inventory": entry["snapshot_inventory"],
            "source_identity_sha256": report["source_identity_sha256"]})

    return {"schema_version": 1, "audit_status": "consistent_historical_segment",
        "qualification_status": "unbound_nonqualifying_historical_segment",
        "qualification_eligible": False,
        "qualification_note": "Retrospective train-split membership does not bind this historical segment to a prospective qualification protocol.",
        "segment_path": str(root), "completed_cycle": completed,
        "completed_update_frontier": previous_target,
        "pending_assignment": None,
        "scheduler_history_sha256": _sha((root / "task-scheduler-state.json").read_bytes()),
        "qualification_protocol_id_checked": protocol_id,
        "qualification_protocol_sha256_checked": protocol_sha,
        "training_qualification_protocol_id": queue_status.get("qualification_protocol_id"),
        "training_qualification_protocol_sha256": queue_status.get("qualification_protocol_sha256"),
        "evaluation_protocol_path_checked": (str(Path(evaluation_protocol).resolve())
            if evaluation_protocol is not None else None),
        "evaluation_protocol_id_checked": queue_status.get("evaluation_protocol_id"),
        "evaluation_protocol_sha256_checked": (_sha(Path(evaluation_protocol).read_bytes())
            if evaluation_protocol is not None else None),
        "supervisor_status_sha256": _sha((root / "supervisor-status.json").read_bytes()),
        "queue_status_sha256": _sha((root / "queue-status.json").read_bytes()),
        "evaluation_inventory_root": str(evaluation_root),
        "evaluation_inventory_sha256": inventory_sha,
        "historical_protocol_binding": "unbound",
        "source_identity": source_value,
        "source_identity_sha256": _source_digest(source_value),
        "shared_training_configuration": shared_config,
        "checkpoints": checkpoints, "evaluations": eval_index}


def export_transition(audit_path: Path, output_path: Path) -> dict:
    audit = _json(Path(audit_path).resolve())
    segment_root = Path(audit.get("segment_path", "")).resolve()
    lock_fd = _assert_supervisor_quiescent(segment_root)
    try:
        return _export_transition_locked(audit_path, output_path, audit)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _export_transition_locked(audit_path: Path, output_path: Path, audit: dict) -> dict:
    """Export matched policy actors only; evidence remains explicitly nonqualifying."""
    audit_path, output_path = Path(audit_path).resolve(), Path(output_path).resolve()
    if audit.get("qualification_eligible") is not False or audit.get("audit_status") != "consistent_historical_segment":
        raise ValueError("transition export requires a consistent, explicitly nonqualifying audit")
    segment_root = Path(audit["segment_path"]).resolve()
    if (_sha((segment_root / "supervisor-status.json").read_bytes()) !=
            audit.get("supervisor_status_sha256") or
            _sha((segment_root / "queue-status.json").read_bytes()) !=
            audit.get("queue_status_sha256") or
            _sha((segment_root / "task-scheduler-state.json").read_bytes()) !=
            audit.get("scheduler_history_sha256")):
        raise ValueError("supervisor or scheduler state changed since audit")
    held_out_evaluations = [row for row in audit.get("evaluations", [])
        if row.get("split") in ("validation", "test")]
    if held_out_evaluations:
        protocol_path = Path(audit.get("evaluation_protocol_path_checked") or "")
        if (not protocol_path.is_file()
                or _sha(protocol_path.read_bytes()) !=
                    audit.get("evaluation_protocol_sha256_checked")):
            raise ValueError("frozen evaluation protocol artifact changed or disappeared since audit")
        for evaluation in held_out_evaluations:
            try:
                _, eval_id, eval_sha, _ = load_evaluation_protocol(protocol_path,
                    scenario_id=evaluation["evaluation_case"]["scenario_id"])
            except Exception as exc:
                raise ValueError(f"frozen evaluation protocol no longer validates: {exc}") from exc
            if (eval_id != evaluation.get("evaluation_protocol_id")
                    or eval_sha != evaluation.get("evaluation_protocol_sha256")
                    or eval_id != audit.get("evaluation_protocol_id_checked")):
                raise ValueError("frozen evaluation protocol identity changed since audit")
    evaluation_root, current_evaluations, inventory_sha = _evaluation_inventory(
        Path(audit.get("evaluation_inventory_root", "")),
        segment_root / "queue-status.json")
    if (str(evaluation_root) != audit.get("evaluation_inventory_root")
            or inventory_sha != audit.get("evaluation_inventory_sha256")):
        raise ValueError("evaluation inventory changed since audit")
    indexed_evaluations = {row["path"]: row for row in audit.get("evaluations", [])}
    for evaluation in current_evaluations:
        prior = indexed_evaluations.get(evaluation["path"])
        if (prior is None or prior.get("sha256") != evaluation["sha256"]
                or prior.get("status_sha256") != evaluation["status_sha256"]
                or prior.get("snapshot_inventory") != evaluation["snapshot_inventory"]):
            raise ValueError("evaluation report/status/snapshot inventory changed or was omitted since audit")
        for snapshot in prior.get("checkpoint_snapshots", {}).values():
            snapshot_path = Path(snapshot.get("path", ""))
            if (not snapshot_path.is_file()
                    or _sha(snapshot_path.read_bytes()) != snapshot.get("sha256")):
                raise ValueError("evaluation checkpoint snapshot changed since audit")
    if set(indexed_evaluations) != {row["path"] for row in current_evaluations}:
        raise ValueError("evaluation inventory differs from audited report set")
    actors = {}
    for track in MECO_TRACKS:
        item = audit.get("checkpoints", {}).get(track)
        if not isinstance(item, dict):
            raise ValueError(f"audit is missing {track} checkpoint identity")
        checkpoint_path = Path(item["path"])
        if not checkpoint_path.is_file() or _sha(checkpoint_path.read_bytes()) != item.get("sha256"):
            raise ValueError(f"{track} checkpoint changed since audit")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor = checkpoint.get("model")
        if _state_digest(actor) != item.get("actor_state_sha256"):
            raise ValueError(f"{track} actor weights changed since audit")
        model = make_meco_track(track, candidate_features=CANDIDATE_BASE_FEATURE_COUNT)
        model.load_state_dict(actor, strict=True)
        actors[track] = {key: value.detach().cpu().clone() for key, value in actor.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or sidecar_path.exists():
        raise FileExistsError("refusing to replace an existing transition artifact or sidecar")
    artifact = {"schema": ARTIFACT_SCHEMA, "transition_id": uuid.uuid4().hex,
        "audit_sha256": _sha(audit_path.read_bytes()),
        "historical_qualification_status": audit["qualification_status"],
        "qualification_eligible": False,
        "historical_source_identity": audit["source_identity"],
        "historical_source_identity_sha256": audit["source_identity_sha256"],
        "checked_qualification_protocol_id": audit["qualification_protocol_id_checked"],
        "checked_qualification_protocol_sha256": audit["qualification_protocol_sha256_checked"],
        "checkpoint_identities": audit["checkpoints"],
        "evaluation_identities": audit["evaluations"],
        "shared_training_configuration": audit["shared_training_configuration"],
        "model_contract": {track: {"state_sha256": _state_digest(actors[track]),
            "keys": sorted(actors[track]),
            "shapes": {key: list(value.shape) for key, value in actors[track].items()},
            "dtypes": {key: str(value.dtype) for key, value in actors[track].items()}}
            for track in MECO_TRACKS},
        "transition_contract": {"mode": "actor_weights_only_fresh_initialization",
            "load": "strict actor state_dict only",
            "critic": "fresh initialization required",
            "optimizer": "fresh initialization required; historical optimizer forbidden",
            "rng": "new explicit seed required; historical RNG forbidden",
            "counters": {"update": 0, "episode": 0, "history": []},
            "run_id": "new UUID required",
            "resume": "not a resume artifact; existing resume identity checks unchanged"},
        "actors": actors}
    fd, payload_tmp = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp",
                                       dir=output_path.parent)
    os.close(fd)
    sidecar_tmp = None
    try:
        torch.save(artifact, payload_tmp)
        artifact_sha = _sha(Path(payload_tmp).read_bytes())
        sidecar = {key: value for key, value in artifact.items() if key != "actors"}
        sidecar.update(artifact_sha256=artifact_sha,
            actor_state_sha256={track: _state_digest(actors[track]) for track in MECO_TRACKS})
        fd, sidecar_tmp = tempfile.mkstemp(prefix=f".{sidecar_path.name}.", suffix=".tmp",
                                           dir=output_path.parent)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(sidecar, sort_keys=True, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(payload_tmp, output_path)
        os.replace(sidecar_tmp, sidecar_path)
    except BaseException:
        for candidate in (payload_tmp, sidecar_tmp):
            if candidate and os.path.exists(candidate):
                os.unlink(candidate)
        for candidate in (output_path, sidecar_path):
            if candidate.exists():
                candidate.unlink()
        raise
    return sidecar


def consume_transition(path: Path, *, seed: int,
                      target_qualification_protocol: Path,
                      models: dict | None = None):
    """Strictly load only actors into fresh models and return reset-state metadata.

    Callers create critic/optimizer/RNG state from scratch using the returned
    initialization seed. No historical trainer state is present in the artifact.
    """
    path = Path(path).resolve()
    sidecar = _json(path.with_suffix(path.suffix + ".json"))
    if _sha(path.read_bytes()) != sidecar.get("artifact_sha256"):
        raise ValueError("transition artifact SHA-256 does not match sidecar")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != ARTIFACT_SCHEMA or artifact.get("qualification_eligible") is not False:
        raise ValueError("unsupported or qualification-claiming transition artifact")
    _, target_protocol_id, target_protocol_sha, _ = _load_protocol(
        Path(target_qualification_protocol))
    if (target_protocol_id != artifact.get("checked_qualification_protocol_id")
            or target_protocol_sha != artifact.get("checked_qualification_protocol_sha256")):
        raise ValueError("target qualification protocol identity/hash differs from transition lineage")
    current_sources = source_identity()
    if (not isinstance(current_sources, dict)
            or current_sources != artifact.get("historical_source_identity")
            or _source_digest(current_sources) != artifact.get("historical_source_identity_sha256")):
        raise ValueError("current source identity does not match the actor model source lineage")
    if models is None:
        models = {track: make_meco_track(track,
            candidate_features=CANDIDATE_BASE_FEATURE_COUNT) for track in MECO_TRACKS}
        registered_types = {track: type(models[track]) for track in MECO_TRACKS}
    else:
        registered_types = {}
        for track in MECO_TRACKS:
            registered_template = make_meco_track(track,
                candidate_features=CANDIDATE_BASE_FEATURE_COUNT)
            registered_types[track] = type(registered_template)
            del registered_template
    if set(models) != set(MECO_TRACKS):
        raise ValueError("consumer requires exactly the five MECO track models")
    if type(seed) is not int or seed < 0:
        raise ValueError("transition consumer requires an explicit nonnegative fresh RNG seed")
    torch.manual_seed(seed)
    random.seed(seed)
    loaded = {}
    for track in MECO_TRACKS:
        actor = artifact.get("actors", {}).get(track)
        contract = artifact.get("model_contract", {}).get(track, {})
        if not isinstance(actor, dict) or _state_digest(actor) != contract.get("state_sha256"):
            raise ValueError(f"{track} actor weights fail transition contract")
        if type(models[track]) is not registered_types[track]:
            raise ValueError(f"{track} consumer model class differs from registered constructor")
        actual = models[track].state_dict()
        actual_keys = sorted(actual)
        actual_shapes = {key: list(value.shape) for key, value in actual.items()}
        actual_dtypes = {key: str(value.dtype) for key, value in actual.items()}
        actor_shapes = {key: list(value.shape) for key, value in actor.items()}
        actor_dtypes = {key: str(value.dtype) for key, value in actor.items()}
        if (actual_keys != contract.get("keys") or actor_shapes != contract.get("shapes")
                or actor_dtypes != contract.get("dtypes")
                or actual_shapes != contract.get("shapes")
                or actual_dtypes != contract.get("dtypes")):
            raise ValueError(f"{track} current model keys/shapes/dtypes differ from transition contract")
        models[track].load_state_dict(actor, strict=True)
        loaded[track] = models[track]
    learning_rate = artifact.get("shared_training_configuration", {}).get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("transition artifact lacks a valid fresh optimizer learning rate")
    critics = {track: ValueHead() for track in MECO_TRACKS}
    optimizers = {track: torch.optim.Adam(
        list(loaded[track].parameters()) + list(critics[track].parameters()),
        lr=float(learning_rate)) for track in MECO_TRACKS}
    return {"models": loaded, "critics": critics, "optimizers": optimizers,
        "initialization": "actor_weights_only_fresh_initialization", "rng_seed": seed,
        "run_id": uuid.uuid4().hex, "update": 0, "episode": 0, "history": [],
        "rng": "new Python/PyTorch seeds initialized; historical RNG not loaded",
        "qualification_eligible": False,
        "lineage": {key: artifact.get(key) for key in (
            "transition_id", "audit_sha256", "historical_source_identity",
            "historical_source_identity_sha256", "checked_qualification_protocol_id",
            "checked_qualification_protocol_sha256", "checkpoint_identities")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--segment-dir", type=Path, required=True)
    audit_parser.add_argument("--qualification-protocol", type=Path, required=True)
    audit_parser.add_argument("--evaluation-protocol", type=Path,
        help="frozen held-out evaluation protocol; required when the segment contains held-out reports")
    audit_parser.add_argument("--evaluation-root", type=Path, required=True,
        help="complete supervisor evaluation output directory; every report/status is inventoried")
    audit_parser.add_argument("--output", type=Path, required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--audit", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        report = audit_segment(args.segment_dir, args.qualification_protocol,
            args.evaluation_root, args.evaluation_protocol)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2,
                                          allow_nan=False) + "\n")
    else:
        report = export_transition(args.audit, args.output)
    print(json.dumps({key: report.get(key) for key in (
        "audit_status", "qualification_status", "qualification_eligible",
        "completed_cycle", "completed_update_frontier", "artifact_sha256")
        if key in report}, sort_keys=True))


if __name__ == "__main__":
    main()
