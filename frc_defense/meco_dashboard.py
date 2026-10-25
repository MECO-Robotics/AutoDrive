"""Read-only v1 status projection for the MECO five-track experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time

from .meco_path_penalty import PLAN_A_PPO_RECOMPUTATION_CONTRACT


TRACKS = ("A1", "A2", "A3", "C1", "C2")
ABLATIONS = ("A1_P0", "A2_P0", "A3_P0")
CONTROLS = ("D0", "D1", "D2")
POLICIES = TRACKS + ABLATIONS + CONTROLS
PENALTY_PROJECTION = "mean_over_legal_heading_bins_xy_grid_v1"
PENALTY_MAP_UNITS = "dimensionless_additive_penalty_per_cardinal_grid_step"
PENALTY_PATH_UNITS = "normalized_grid_edge_units"
LEARNED_ROUTE_COST_PREFIX = (
    "meco_legal_heading_penalty_projection_v1:schema=xy_grid_f64le_v2:")
LEGACY_ROUTE_COST_PROVENANCE = "legacy_bump_cost_path_integral_v2"
STATUS_SCHEMA_VERSION = 1
HEARTBEAT_STALE_SECONDS = 90
WORKER_STALL_SECONDS = 600
QUEUE_STALE_SECONDS = 180
EVALUATION_HEARTBEAT_STALE_SECONDS = 90
MAX_STATUS_FILE_BYTES = 32 * 1024 * 1024
MAX_DASHBOARD_INLINE_REPORT_BYTES = MAX_STATUS_FILE_BYTES
MAX_DASHBOARD_SUMMARY_RAW_BYTES = 128 * 1024 * 1024
MAX_DASHBOARD_SUMMARY_FILE_BYTES = 1024 * 1024
MAX_DASHBOARD_SUMMARY_RECORDS = 50_000
MAX_DASHBOARD_SUMMARY_JSON_NODES = 2_000_000
MAX_DASHBOARD_SUMMARY_FILES_PER_SCAN = 8
MAX_DASHBOARD_SUMMARY_BYTES_PER_SCAN = 4 * 1024 * 1024
MAX_DASHBOARD_SUMMARY_RAW_BYTES_PER_SCAN = 512 * 1024 * 1024
MAX_DASHBOARD_SUMMARY_RECORDS_PER_SCAN = 50_000
DASHBOARD_SUMMARY_SCHEMA_VERSION = 1
DASHBOARD_SUMMARY_SUFFIX = ".dashboard-summary.json"
_DASHBOARD_SUMMARY_READS = threading.BoundedSemaphore(2)
_DASHBOARD_SUMMARY_WRITES = threading.BoundedSemaphore(1)
DIAGNOSTIC_MONITOR_ERROR_LIMIT = 300
DIAGNOSTIC_MONITOR_FRONTIER_LIMIT = 100


def _read_json(path: Path):
    try:
        if path.stat().st_size > MAX_STATUS_FILE_BYTES:
            return None, "file exceeds dashboard size limit"
        data = json.loads(path.read_text())
        return (data, None) if isinstance(data, dict) else (None, "expected a JSON object")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"could not read JSON: {type(exc).__name__}"


def _utc_timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _read_json_within(path: Path, root: Path):
    """Read a status file only when its resolved path remains under root."""
    if not _inside(path, root):
        return None, "path resolves outside selected MECO data root"
    return _read_json(path)


def _process_evidence(pid, *, expected_run_id=None):
    """Read raw procfs state and confirm that PID is a MECO trainer."""
    if type(pid) is not int or pid <= 0:
        return {"confirmed": False, "state": None, "error": "invalid PID"}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"confirmed": False, "state": None, "error": "process missing"}
    except PermissionError:
        pass
    except OSError:
        return {"confirmed": False, "state": None, "error": "process unavailable"}
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return {"confirmed": False, "state": None, "error": "proc command unavailable"}
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return {"confirmed": False, "state": None, "error": "proc state unavailable"}
    # comm is parenthesized and can itself contain spaces or closing parens.
    closing = stat.rfind(")")
    if closing < 0:
        return {"confirmed": False, "state": None, "error": "proc state malformed"}
    fields = stat[closing + 1:].split()
    process_state = fields[0] if fields and len(fields[0]) == 1 else None
    if process_state is None:
        return {"confirmed": False, "state": None, "error": "proc state malformed"}
    is_trainer = any(b"meco_track_training" in argument for argument in arguments)
    has_expected_run = (expected_run_id is None or
        isinstance(expected_run_id, str) and
        any(argument == b"--run-id" and index + 1 < len(arguments)
            and arguments[index + 1] == expected_run_id.encode()
            for index, argument in enumerate(arguments)))
    if not is_trainer or not has_expected_run:
        return {"confirmed": False, "state": process_state,
                "error": "PID command/run-ID mismatch"}
    if process_state not in {"D", "R", "S", "I", "P", "W"}:
        return {"confirmed": False, "state": process_state,
                "error": "process is not in an active state"}
    return {"confirmed": True, "state": process_state, "error": None}


def _process_alive(pid) -> bool:
    """Confirm a live PID still names a MECO trainer in procfs."""
    return _process_evidence(pid)["confirmed"]


def _heartbeat(run_dir: Path, manifest: dict, now: float, data_root: Path,
               process_probe=_process_alive):
    heartbeat, error = _read_json_within(run_dir / "heartbeat.json", data_root)
    if not heartbeat:
        return {"state": "missing", "age_seconds": None, "fresh": False,
                "process_alive": False, "process_confirmed": False,
                "process_state": None, "error": error if error else None}
    stamp = _utc_timestamp(heartbeat.get("heartbeat_utc"))
    age = None if stamp is None else max(0., now - stamp)
    fresh = age is not None and age <= HEARTBEAT_STALE_SECONDS and stamp <= now + 5.
    pid = heartbeat.get("pid")
    alive = bool(process_probe(pid))
    process = _process_evidence(pid, expected_run_id=heartbeat.get("run_id"))
    process_confirmed = (process["confirmed"] if process_probe is _process_alive else alive)
    track = manifest.get("track", manifest.get("config", {}).get("track")
                         if isinstance(manifest.get("config"), dict) else None)
    consistent = (heartbeat.get("schema_version") == 1
                  and heartbeat.get("run_id") == manifest.get("run_id")
                  and heartbeat.get("track") == track)
    state = ("confirmed" if fresh and process_confirmed and consistent else
             "stale" if stamp is not None else "invalid")
    return {"state": state, "age_seconds": age, "fresh": fresh,
            "process_alive": alive, "pid": pid if type(pid) is int else None,
            "process_confirmed": process_confirmed,
            "process_state": process["state"],
            "process_error": process["error"],
            "run_id": heartbeat.get("run_id"),
            "provenance_consistent": consistent,
            "completed_updates": heartbeat.get("completed_updates")
                if type(heartbeat.get("completed_updates")) is int else None,
            "error": None if consistent else "heartbeat provenance mismatch"}


def _training_curriculum_projection(root: Path, queue_payload: dict,
                                    queue_state: str, data_root: Path):
    """Project only scheduler exposure that matches the fresh queue snapshot."""
    from .meco_phase4 import PHASE4_FACTORS, validate_phase4_training_assignment
    from .training_curriculum import SCENARIOS

    unavailable = {"state": "unavailable", "assignment": None,
        "scenario_exposure_chunks": None, "category_exposure_chunks": None,
        "phase4_exposure_by_factor": None,
        "semantics": "training exposure only; not evaluation or qualification"}
    assignment = queue_payload.get("task_assignment")
    if queue_state != "fresh":
        return {**unavailable, "state": "stale" if queue_state == "stale" else "unavailable",
                "error": "queue snapshot is not fresh"}
    scheduler_summary = queue_payload.get("task_scheduler")
    queue_cycle = queue_payload.get("cycle")
    if (not isinstance(assignment, dict) or not isinstance(scheduler_summary, dict)
            or type(queue_cycle) is not int or type(assignment.get("cycle")) is not int
            or assignment.get("cycle") != queue_cycle
            or not isinstance(assignment.get("scenario_id"), str)
            or assignment.get("scenario_id") not in SCENARIOS
            or not isinstance(assignment.get("category"), str)
            or assignment.get("category") not in SCENARIOS[
                assignment.get("scenario_id")].required_categories
            or type(assignment.get("seed")) is not int
            or assignment["seed"] < 0
            or not isinstance(assignment.get("assignment_id"), str)
            or not assignment["assignment_id"]
            or assignment.get("sampling_role") not in
                ("recovery", "ancestor_retention", "stage_exposure")):
        return {**unavailable, "error": "queue task assignment is malformed"}
    projected_assignment = {key: assignment.get(key) for key in (
        "cycle", "scenario_id", "category", "seed", "sampling_role", "assignment_id")}
    try:
        factor = validate_phase4_training_assignment(assignment)
    except (ValueError, TypeError, KeyError):
        return {**unavailable, "error": "queue Phase 4 assignment is malformed"}
    if factor is None:
        if "phase4_exposure_key" in assignment:
            return {**unavailable, "error": "queue Phase 4 exposure key is unexpected"}
    else:
        factor_id = factor["kind"]
        expected_key = _scheduler_phase4_exposure_key(
            assignment["scenario_id"], assignment["category"],
            assignment["seed"], factor_id)
        if assignment.get("phase4_exposure_key") != expected_key:
            return {**unavailable, "error": "queue Phase 4 exposure key is invalid"}
        projected_assignment.update(phase4_factor=factor_id,
                                    phase4_split=assignment["phase4_split"])

    scheduler, error = _read_json_within(root / "task-scheduler-state.json", data_root)
    if scheduler is None:
        return {**unavailable, "error": error or "scheduler state is unavailable"}
    counter_keys = ("exposure_chunks", "category_exposures", "recovery_pressure",
                    "phase4_exposure_counts")
    snapshot_pending = scheduler_summary.get("pending_assignment")
    if (scheduler.get("schema_version") != 1
            or type(scheduler.get("completed_cycle")) is not int
            or scheduler.get("completed_cycle") < 0
            or not isinstance(scheduler.get("pending_assignment"), dict)
            or scheduler.get("pending_assignment") != assignment
            or snapshot_pending != assignment
            or scheduler_summary.get("completed_cycle") != scheduler.get("completed_cycle")
            or queue_cycle != scheduler.get("completed_cycle") + 1
            or any(not isinstance(scheduler.get(key), dict)
                   or scheduler_summary.get(key) != scheduler.get(key)
                   for key in counter_keys)):
        return {**unavailable, "error": "queue and scheduler state do not match"}

    history = scheduler.get("history")
    if (not isinstance(history, list)
            or len(history) != scheduler["completed_cycle"]):
        return {**unavailable, "error": "scheduler history is incomplete"}
    expected_scenarios = {}
    expected_categories = {}
    expected_phase4_hashes = {}
    phase4_by_factor = {name: 0 for name in PHASE4_FACTORS}
    for cycle, row in enumerate(history, start=1):
        previous = row.get("assignment") if isinstance(row, dict) else None
        if (not isinstance(previous, dict)
                or previous.get("cycle") != cycle
                or not isinstance(previous.get("scenario_id"), str)
                or previous.get("scenario_id") not in SCENARIOS
                or not isinstance(previous.get("category"), str)
                or previous.get("category") not in SCENARIOS[
                    previous.get("scenario_id")].required_categories
                or type(previous.get("seed")) is not int
                or previous["seed"] < 0):
            return {**unavailable, "error": "scheduler history assignment is malformed"}
        scenario = previous["scenario_id"]
        category = previous["category"]
        expected_scenarios[scenario] = expected_scenarios.get(scenario, 0) + 1
        category_key = f"{scenario}/{category}"
        expected_categories[category_key] = expected_categories.get(category_key, 0) + 1
        try:
            historical_factor = validate_phase4_training_assignment(previous)
        except (ValueError, TypeError, KeyError):
            return {**unavailable, "error": "scheduler history factor is malformed"}
        if historical_factor is None:
            if "phase4_exposure_key" in previous:
                return {**unavailable, "error": "scheduler history has an unexpected Phase 4 key"}
        else:
            factor_id = historical_factor["kind"]
            hash_key = _scheduler_phase4_exposure_key(
                scenario, category, previous["seed"], factor_id)
            if previous.get("phase4_exposure_key") != hash_key:
                return {**unavailable, "error": "scheduler history factor key is invalid"}
            expected_phase4_hashes[hash_key] = expected_phase4_hashes.get(hash_key, 0) + 1
            phase4_by_factor[factor_id] += 1
    if (scheduler["exposure_chunks"] != expected_scenarios
            or scheduler["category_exposures"] != expected_categories
            or scheduler["phase4_exposure_counts"] != expected_phase4_hashes):
        return {**unavailable, "error": "scheduler counters do not match completed history"}
    return {"state": "fresh", "assignment": projected_assignment,
        "scenario_exposure_chunks": expected_scenarios,
        "category_exposure_chunks": expected_categories,
        "phase4_exposure_by_factor": phase4_by_factor,
        "semantics": "training exposure only; not evaluation or qualification"}


def _scheduler_phase4_exposure_key(scenario, category, seed, factor_id):
    value = {"scenario_id": scenario, "category": category,
             "seed": seed, "factor": factor_id}
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        allow_nan=False).encode()).hexdigest()


def _process_evidence(pid, *, expected_run_id=None):
    """Read raw procfs state and confirm that PID is a MECO trainer."""
    if type(pid) is not int or pid <= 0:
        return {"confirmed": False, "state": None, "error": "invalid PID"}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"confirmed": False, "state": None, "error": "process missing"}
    except PermissionError:
        pass
    except OSError:
        return {"confirmed": False, "state": None, "error": "process unavailable"}
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return {"confirmed": False, "state": None, "error": "proc command unavailable"}
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return {"confirmed": False, "state": None, "error": "proc state unavailable"}
    # comm is parenthesized and can itself contain spaces or closing parens.
    closing = stat.rfind(")")
    if closing < 0:
        return {"confirmed": False, "state": None, "error": "proc state malformed"}
    fields = stat[closing + 1:].split()
    process_state = fields[0] if fields and len(fields[0]) == 1 else None
    if process_state is None:
        return {"confirmed": False, "state": None, "error": "proc state malformed"}
    is_trainer = any(b"meco_track_training" in argument for argument in arguments)
    has_expected_run = (expected_run_id is None or
        isinstance(expected_run_id, str) and
        any(argument == b"--run-id" and index + 1 < len(arguments)
            and arguments[index + 1] == expected_run_id.encode()
            for index, argument in enumerate(arguments)))
    if not is_trainer or not has_expected_run:
        return {"confirmed": False, "state": process_state,
                "error": "PID command/run-ID mismatch"}
    if process_state not in {"D", "R", "S", "I", "P", "W"}:
        return {"confirmed": False, "state": process_state,
                "error": "process is not in an active state"}
    return {"confirmed": True, "state": process_state, "error": None}


def _process_alive(pid) -> bool:
    """Confirm a live PID still names a MECO trainer in procfs."""
    return _process_evidence(pid)["confirmed"]


def _heartbeat(run_dir: Path, manifest: dict, now: float, data_root: Path,
               process_probe=_process_alive):
    heartbeat, error = _read_json_within(run_dir / "heartbeat.json", data_root)
    if not heartbeat:
        return {"state": "missing", "age_seconds": None, "fresh": False,
                "process_alive": False, "process_confirmed": False,
                "process_state": None, "error": error if error else None}
    stamp = _utc_timestamp(heartbeat.get("heartbeat_utc"))
    age = None if stamp is None else max(0., now - stamp)
    fresh = age is not None and age <= HEARTBEAT_STALE_SECONDS and stamp <= now + 5.
    pid = heartbeat.get("pid")
    alive = bool(process_probe(pid))
    process = _process_evidence(pid, expected_run_id=heartbeat.get("run_id"))
    process_confirmed = (process["confirmed"] if process_probe is _process_alive else alive)
    track = manifest.get("track", manifest.get("config", {}).get("track")
                         if isinstance(manifest.get("config"), dict) else None)
    consistent = (heartbeat.get("schema_version") == 1
                  and heartbeat.get("run_id") == manifest.get("run_id")
                  and heartbeat.get("track") == track)
    state = ("confirmed" if fresh and process_confirmed and consistent else
             "stale" if stamp is not None else "invalid")
    return {"state": state, "age_seconds": age, "fresh": fresh,
            "process_alive": alive, "pid": pid if type(pid) is int else None,
            "process_confirmed": process_confirmed,
            "process_state": process["state"],
            "process_error": process["error"],
            "run_id": heartbeat.get("run_id"),
            "provenance_consistent": consistent,
            "completed_updates": heartbeat.get("completed_updates")
                if type(heartbeat.get("completed_updates")) is int else None,
            "error": None if consistent else "heartbeat provenance mismatch"}



def _queue_state(root: Path, now: float, data_root: Path):
    path = root / "queue-status.json"
    payload, error = _read_json_within(path, data_root)
    if payload is None:
        return {"state": "unavailable", "age_seconds": None,
                "lanes": {}, "error": error}
    stamp = _utc_timestamp(payload.get("updated_utc"))
    age = None if stamp is None else max(0., now - stamp)
    fresh = (payload.get("schema_version") == 1 and age is not None
             and age <= QUEUE_STALE_SECONDS and stamp <= now + 5.)
    lanes = {}
    malformed_lanes = False
    lane_error = None
    raw_lanes = payload.get("lanes")
    valid_lane_statuses = {"queued", "running", "stalled", "complete", "stopped",
                           "failed", "aborted_due_to_sibling_failure",
                           "deferred_for_evaluation"}
    if isinstance(raw_lanes, dict):
        lane_rows = []
        if set(raw_lanes) != set(TRACKS):
            malformed_lanes = True
            lane_error = "queue dict lanes do not match the five-track schema"
        for track, value in raw_lanes.items():
            if track not in TRACKS or not isinstance(value, dict):
                malformed_lanes = True
                lane_error = lane_error or "queue lane record is malformed"
                continue
            lane = dict(value)
            if "track" in lane and lane["track"] != track:
                malformed_lanes = True
                lane_error = lane_error or "queue lane key does not match its track"
                continue
            lane["track"] = track
            lane_rows.append(lane)
    elif isinstance(raw_lanes, list):
        lane_rows = raw_lanes
    else:
        lane_rows = None
        error = "queue lanes are malformed"
        malformed_lanes = True
        lane_error = error
    seen_tracks = set()
    if lane_rows is not None:
        for lane in lane_rows:
            if (not isinstance(lane, dict) or lane.get("track") not in TRACKS
                    or lane.get("track") in seen_tracks
                    or not isinstance(lane.get("run_id"), str)
                    or not lane["run_id"].strip()
                    or not isinstance(lane.get("status"), str)
                    or lane.get("status") not in valid_lane_statuses
                    or (lane.get("pid") is not None and type(lane.get("pid")) is not int)
                    or not isinstance(lane.get("device"), str)
                    or not isinstance(lane.get("run_directory"), str)
                    or type(lane.get("completed_updates")) is not int
                    or lane["completed_updates"] < 0
                    or type(lane.get("chunk_updates")) is not int
                    or lane["chunk_updates"] < 0):
                malformed_lanes = True
                lane_error = (lane_error or
                    ("queue lane record is duplicate or malformed"
                     if isinstance(lane, dict) and lane.get("track") in seen_tracks
                     else "queue lane record is malformed"))
                continue
            seen_tracks.add(lane["track"])
            pid = lane.get("pid") if type(lane.get("pid")) is int else None
            process = _process_evidence(pid, expected_run_id=lane["run_id"])
            lanes[lane["track"]] = {"run_id": lane["run_id"],
                "status": lane.get("status", "unknown"),
                "pid": pid,
                "process_confirmed": process["confirmed"],
                "process_state": process["state"],
                "process_error": process["error"],
                "worker_state": ("unconfirmed" if lane.get("status") in ("running", "stalled")
                                 else "queued" if lane.get("status") == "deferred_for_evaluation"
                                 else lane.get("status", "unknown")),
                "device": lane.get("device") if isinstance(lane.get("device"), str) else None,
                "run_directory": lane.get("run_directory") if isinstance(
                    lane.get("run_directory"), str) else None,
                "completed_updates": lane.get("completed_updates")
                    if type(lane.get("completed_updates")) is int else None,
                "chunk_updates": lane.get("chunk_updates")
                    if type(lane.get("chunk_updates")) is int else None}
    if seen_tracks != set(TRACKS):
        malformed_lanes = True
        lane_error = lane_error or "queue lanes do not contain exactly one record per track"
    fresh = fresh and not malformed_lanes
    if malformed_lanes:
        error = lane_error
    active_lanes = [lane for lane in lanes.values()
                    if lane.get("status") in ("running", "stalled")]
    worker_state = ("unknown" if not fresh else
                    "unconfirmed" if active_lanes else
                    "queued" if any(lane.get("status") in ("queued", "deferred_for_evaluation")
                                     for lane in lanes.values()) else "idle")
    raw_evaluation = (payload.get("evaluation")
        if isinstance(payload.get("evaluation"), dict) else None)
    monitor = payload.get("diagnostic_monitor")
    if monitor is None and raw_evaluation is not None:
        monitor = raw_evaluation.get("diagnostic_monitor")
    monitor_error = payload.get("diagnostic_monitor_error")
    if monitor_error is None and raw_evaluation is not None:
        monitor_error = raw_evaluation.get("diagnostic_monitor_error")
    # The monitor is exposed through its dedicated allowlisted projection;
    # do not also leak its nested checkpoint paths through evaluator status.
    evaluation = dict(raw_evaluation) if raw_evaluation is not None else None
    if evaluation is not None:
        evaluation.pop("diagnostic_monitor", None)
        evaluation.pop("diagnostic_monitor_error", None)
    diagnostic_monitor = _diagnostic_monitor_projection(monitor, monitor_error,
                                                         fresh=fresh)
    return {"state": "fresh" if fresh else "stale", "worker_state": worker_state,
            "age_seconds": age,
            "lanes": lanes,
            "curriculum": _training_curriculum_projection(
                root, payload, "fresh" if fresh else "stale", data_root),
            "phase": payload.get("phase") if isinstance(payload.get("phase"), str) else None,
            "cycle": payload.get("cycle") if type(payload.get("cycle")) is int else None,
            "evaluation": evaluation,
            "evaluation_schedule": payload.get("evaluation_schedule")
                if isinstance(payload.get("evaluation_schedule"), dict) else None,
            "diagnostic_monitor": diagnostic_monitor,
            "supervisor_id": payload.get("supervisor_id")
                if isinstance(payload.get("supervisor_id"), str) else None,
            "error": error if not fresh else None}


def _diagnostic_monitor_projection(raw, raw_error, *, fresh):
    """Return an allowlisted view of the diagnostic-only monitor journal summary."""
    error = (raw_error[:DIAGNOSTIC_MONITOR_ERROR_LIMIT]
             if isinstance(raw_error, str) else None)
    safe_base = {"configured": False, "enabled": False,
        "diagnostic_only": True, "qualification_eligible": False,
        "state": "not_configured", "fresh": False, "journal_sha256": None,
        "journal_events": None, "completed_count": 0, "active_frontier": None,
        "queued_frontier": None, "pending_frontier": None,
        "skipped_frontiers": [], "missed_frontiers": [], "skipped_count": 0,
        "missed_count": 0, "error": error}
    if raw is None:
        return safe_base
    safe_base["configured"] = True
    if not isinstance(raw, dict):
        safe_base.update(state="untrusted", error=error or "monitor summary is malformed")
        return safe_base

    def count(value):
        return value if type(value) is int and value >= 0 else None

    def frontier(value, *, skipped=False):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("frontier summary is malformed")
        result = {}
        frontier_id = value.get("frontier_id")
        if not isinstance(frontier_id, str) or not frontier_id or len(frontier_id) > 160:
            raise ValueError("frontier identity is malformed")
        result["frontier_id"] = frontier_id
        for key, minimum in (("cycle", 1), ("update", 0)):
            numeric = value.get(key)
            if type(numeric) is not int or numeric < minimum:
                raise ValueError("frontier identity is malformed")
            result[key] = numeric
        status = "skipped" if skipped else value.get("status")
        if not isinstance(status, str) or status not in {
                "queued", "running", "failed", "verified", "complete", "skipped"}:
            raise ValueError("frontier status is malformed")
        result["status"] = status
        attempts = value.get("attempts")
        if attempts is not None:
            if not isinstance(attempts, list) or any(not isinstance(row, dict) for row in attempts):
                raise ValueError("frontier attempts are malformed")
            if attempts:
                last = attempts[-1]
                attempt_number = last.get("attempt")
                if type(attempt_number) is not int or attempt_number < 1:
                    raise ValueError("frontier attempt is malformed")
                result["attempt"] = attempt_number
        elif "attempt" in value:
            attempt_number = value.get("attempt")
            if type(attempt_number) is not int or attempt_number < 1:
                raise ValueError("frontier attempt is malformed")
            result["attempt"] = attempt_number
        raw_result = value.get("result")
        if raw_result is not None:
            if not isinstance(raw_result, dict):
                raise ValueError("frontier result is malformed")
            result_summary = {}
            evaluation_id = raw_result.get("evaluation_id")
            if evaluation_id is not None:
                if not isinstance(evaluation_id, str) or not evaluation_id or len(evaluation_id) > 160:
                    raise ValueError("frontier result identity is malformed")
                result_summary["evaluation_id"] = evaluation_id
            for hash_field in ("report_sha256", "status_sha256"):
                hash_value = raw_result.get(hash_field)
                if hash_value is not None:
                    if not isinstance(hash_value, str) or not re.fullmatch(r"[0-9a-f]{64}", hash_value):
                        raise ValueError("frontier result hash is malformed")
                    result_summary[hash_field] = hash_value
            for flag in ("diagnostic_only", "qualification_eligible"):
                flag_value = raw_result.get(flag)
                if flag_value is not None:
                    if type(flag_value) is not bool or (flag == "qualification_eligible" and flag_value):
                        raise ValueError("frontier result flags are malformed")
                    result_summary[flag] = flag_value
            result["result"] = result_summary
        return result

    try:
        enabled = raw.get("enabled")
        if enabled is not True or raw.get("diagnostic_only") is not True or raw.get(
                "qualification_eligible") is not False:
            raise ValueError("monitor flags do not confirm diagnostic-only status")
        journal = raw.get("journal_sha256")
        if not isinstance(journal, str) or not re.fullmatch(r"[0-9a-f]{64}", journal):
            raise ValueError("monitor journal hash is malformed")
        journal_events = count(raw.get("journal_events"))
        completed = count(raw.get("completed_count"))
        if journal_events is None or completed is None:
            raise ValueError("monitor counts are malformed")
        skipped = raw.get("skipped_frontiers")
        missed = raw.get("missed_frontiers")
        if not isinstance(skipped, list) or not isinstance(missed, list):
            raise ValueError("monitor frontier history is malformed")
        skipped_rows = [frontier(row, skipped=True)
                        for row in skipped[-DIAGNOSTIC_MONITOR_FRONTIER_LIMIT:]]
        missed_rows = []
        for row in missed[-DIAGNOSTIC_MONITOR_FRONTIER_LIMIT:]:
            if not isinstance(row, dict) or type(row.get("cycle")) is not int or row["cycle"] < 1 or \
                    type(row.get("update")) is not int or row["update"] < 0 or \
                    not isinstance(row.get("reason"), str) or not row["reason"]:
                raise ValueError("missed frontier summary is malformed")
            missed_rows.append({"cycle": row["cycle"], "update": row["update"],
                "reason": row["reason"][:DIAGNOSTIC_MONITOR_ERROR_LIMIT]})
        active = frontier(raw.get("active_frontier"))
        queued = frontier(raw.get("queued_frontier"))
        pending = frontier(raw.get("pending_frontier"))
        if active is not None and active["status"] == "queued":
            raise ValueError("active frontier status is malformed")
        if queued is not None and queued["status"] != "queued":
            raise ValueError("queued frontier status is malformed")
        if pending is not None and pending["status"] not in {"queued", "running", "failed", "verified"}:
            raise ValueError("pending frontier status is malformed")
        if pending != (active or queued):
            raise ValueError("pending frontier does not match the active or queued frontier")
    except ValueError as exc:
        safe_base.update(state="untrusted", error=error or str(exc))
        return safe_base

    state = "fresh" if fresh else "stale"
    safe_base.update(configured=True, enabled=True, diagnostic_only=True,
        qualification_eligible=False, state=state, fresh=bool(fresh),
        journal_sha256=journal, journal_events=journal_events,
        completed_count=completed, skipped_frontiers=skipped_rows,
        missed_frontiers=missed_rows, skipped_count=len(skipped),
        missed_count=len(missed), error=error)
    # A stale queue may retain historical counts, but must never present a saved
    # pending frontier as current work.
    if fresh:
        safe_base.update(active_frontier=active, queued_frontier=queued,
                         pending_frontier=pending)
    return safe_base


def _worker_stalled(*, manifest, heartbeat, stats, latest, lane, queue, now):
    """Require fresh matching queue/worker provenance and a static update frontier."""
    if (manifest.get("status") != "running"
            or queue.get("state") != "fresh"
            or queue.get("phase") not in ("training", "catching_up")
            or not heartbeat.get("process_confirmed")
            or heartbeat.get("fresh")
            or not heartbeat.get("provenance_consistent")
            or heartbeat.get("run_id") != manifest.get("run_id")
            or type(heartbeat.get("pid")) is not int
            or lane.get("status") != "running"
            or lane.get("run_id") != manifest.get("run_id")
            or lane.get("pid") != heartbeat.get("pid")
            or lane.get("completed_updates") != heartbeat.get("completed_updates")):
        return False, None

    frontier = heartbeat.get("completed_updates")
    if type(frontier) is not int:
        return False, None
    if isinstance(stats, dict):
        if stats.get("run_id") != manifest.get("run_id") or stats.get("update") != frontier:
            return False, None
        if not isinstance(latest, dict):
            return False, None
        update_stamp = _utc_timestamp(latest.get("update_utc"))
    else:
        update_stamp = None

    started = _utc_timestamp(manifest.get("started_utc"))
    if started is None:
        return False, None
    chunk_start = manifest.get("chunk_start_update")
    if update_stamp is None:
        if frontier != chunk_start:
            return False, None
        frontier_stamp = started
    else:
        frontier_stamp = max(started, update_stamp)
    unchanged_seconds = max(0., now - frontier_stamp)
    stale_seconds = heartbeat.get("age_seconds")
    if (stale_seconds is None or stale_seconds <= HEARTBEAT_STALE_SECONDS
            or unchanged_seconds < WORKER_STALL_SECONDS):
        return False, unchanged_seconds
    return True, unchanged_seconds


def _queue_attempt_projection(track, lane, *, queue_fresh, previous_run=None):
    """Project a queue-only attempt without borrowing evidence from an older run."""
    lane_state = lane.get("status")
    worker_state = ("queued" if queue_fresh and lane_state in
                    ("queued", "deferred_for_evaluation")
        else "unconfirmed")
    queue_process_confirmed = (queue_fresh
        and lane_state in ("running", "stalled")
        and lane.get("process_confirmed") is True)
    worker_pid = lane.get("pid") if queue_process_confirmed else None
    projected = {"run_id": lane.get("run_id"), "track": track,
        "run_directory": lane.get("run_directory"),
        "manifest_state": "missing", "queue_status": lane_state,
        "training_state": worker_state,
        "heartbeat": {"state": "not_started"
            if queue_fresh and lane_state in ("queued", "deferred_for_evaluation") else "missing",
            "fresh": False, "process_alive": False, "process_confirmed": False,
            "pid": None},
        "queue_process_probe": {"confirmed": queue_process_confirmed,
            "pid": worker_pid},
        "worker_pid": worker_pid, "device": lane.get("device"),
        "sources": {}, "config": {}, "latest_checkpoint": None,
        "completed_updates": (lane.get("completed_updates")
            if type(lane.get("completed_updates")) is int else 0),
        "latest_training_metrics": None}
    if previous_run is not None:
        projected["previous_run"] = {key: previous_run.get(key) for key in (
            "run_id", "run_directory", "manifest_state", "training_state")}
    return projected


def _safe_metrics(record):
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    names = ("success", "useful_throughput", "scores", "acquisitions", "deliveries",
             "collisions", "safety_events", "fallback_count", "fallback_rate",
             "feasible_rate", "heading_violations", "swept_violations",
             "p50_latency_ms", "p95_latency_ms", "p99_latency_ms")
    selected = {name: value for name, value in metrics.items()
                if name in names and type(value) in (bool, int, float)
                and value == value and abs(value) != float("inf")}
    recall = record.get("candidate_recall")
    if type(recall) in (int, float) and recall == recall and abs(recall) != float("inf"):
        selected["candidate_recall"] = recall
    elif isinstance(recall, dict):
        selected["candidate_recall"] = {
            str(key): value for key, value in recall.items()
            if isinstance(key, str) and type(value) in (int, float)
            and value == value and abs(value) != float("inf")}
    latency = record.get("latency_ms")
    if isinstance(latency, dict):
        selected["latency_ms"] = {str(key): value for key, value in latency.items()
                                  if isinstance(key, str) and type(value) in (int, float)
                                  and value == value and abs(value) != float("inf")}
    return selected


def _collection_action_bouts(record):
    """Validate one episode's descriptive collection-bout aggregate."""
    ledger = record.get("meco_metric_ledger")
    if not isinstance(ledger, dict):
        return None
    schema_version = ledger.get("schema_version")
    if schema_version not in (3, 4):
        raise ValueError("unsupported meco_metric_ledger schema")
    availability = ledger.get("availability")
    available = (availability.get("collection_action_bouts")
                 if isinstance(availability, dict) else None)
    if available is not True:
        return None
    bouts = ledger.get("collection_action_bouts")
    if not isinstance(bouts, dict):
        raise ValueError("available collection_action_bouts must be an object")
    count, acquired = bouts.get("count"), bouts.get("acquired_count")
    unsuccessful = bouts.get("unsuccessful_collection_action_bouts")
    proportion = bouts.get("acquired_proportion")
    if (type(count) is not int or count < 0
            or type(acquired) is not int or acquired < 0 or acquired > count):
        raise ValueError("invalid available collection_action_bouts counts")
    if unsuccessful is None and schema_version == 3:
        # Older v3 records predate this field. Preserve their other bout
        # metrics while exposing this new metric as unavailable.
        pass
    elif (type(unsuccessful) is not int or unsuccessful < 0
            or unsuccessful > count or acquired + unsuccessful != count):
        raise ValueError("invalid unsuccessful_collection_action_bouts count")
    if count == 0:
        if proportion is not None:
            raise ValueError("zero collection action bouts must have null proportion")
    elif (type(proportion) not in (int, float)
            or not _finite_number(proportion)
            or not 0 <= proportion <= 1
            or abs(float(proportion) - acquired / count) > 1e-12):
        raise ValueError("invalid available collection_action_bouts proportion")
    return {"count": count, "acquired_count": acquired,
            "unsuccessful_collection_action_bouts": unsuccessful,
            "acquired_proportion": proportion}


def _summarize_collection_action_bouts(rows):
    """Pool only available episode aggregates; keep no-data values null."""
    values = [value for value in (_collection_action_bouts(row) for row in rows)
              if value is not None]
    count = sum(value["count"] for value in values) if values else None
    acquired = sum(value["acquired_count"] for value in values) if values else None
    unsuccessful_values = [value["unsuccessful_collection_action_bouts"]
                            for value in values
                            if value["unsuccessful_collection_action_bouts"] is not None]
    unsuccessful = sum(unsuccessful_values) if unsuccessful_values else None
    return {
        "available_records": len(values),
        "count": count,
        "acquired_count": acquired,
        "unsuccessful_collection_action_bouts": unsuccessful,
        "unsuccessful_collection_action_bouts_available_records": len(unsuccessful_values),
        "unsuccessful_collection_action_bouts_total_records": len(rows),
        "acquired_proportion": (acquired / count if count else None),
        "semantics": "unsuccessful_collection_action_bouts counts finalized focal collection-action bouts with authoritative acquired quantity known to be zero; it is a descriptive bout outcome, not causal intake failure",
    }


def _finite_number(value):
    return type(value) in (bool, int, float) and value == value and abs(value) != float("inf")


def _summarize(rows):
    """Expose category means and a transparent normal-approximation interval."""
    numeric = {}
    for row in rows:
        for name, value in row.items():
            if _finite_number(value):
                numeric.setdefault(name, []).append(float(value))
    summary = {}
    for name, values in numeric.items():
        count = len(values)
        mean = sum(values) / count
        if count > 1:
            variance = sum((value - mean) ** 2 for value in values) / (count - 1)
            half = 1.96 * (variance / count) ** .5
            interval = [mean - half, mean + half]
        else:
            interval = None
        summary[name] = {"n": count, "mean": mean, "normal_95_interval": interval}
    return summary


def _valid_plan_a_penalty_contract(penalty):
    """Validate versioned Plan A penalty semantics and scale metadata."""
    return (isinstance(penalty, dict)
            and penalty.get("trained_mode") in ("learned", "p0")
            and penalty.get("p0_ablation_mode") == "p0"
            and penalty.get("projection") == PENALTY_PROJECTION
            and penalty.get("map_units") == PENALTY_MAP_UNITS
            and penalty.get("path_units") == PENALTY_PATH_UNITS
            and penalty.get("ppo_recomputation") == PLAN_A_PPO_RECOMPUTATION_CONTRACT
            and _finite_number(penalty.get("cost_scale"))
            and penalty["cost_scale"] >= 0
            and _finite_number(penalty.get("points_per_grid_edge"))
            and penalty["points_per_grid_edge"] >= 0)


def _comparison(report):
    """Validate a paired all-policy evaluation without inferring quality."""
    if (report.get("schema_version") != 1
            or report.get("tracks") != list(TRACKS)
            or report.get("ablations") != list(ABLATIONS)
            or report.get("controls") != list(CONTROLS)
            or report.get("paired") is not True
            or not isinstance(report.get("protocol_id"), str)
            or not report["protocol_id"]
            or not isinstance(report.get("evaluation_id"), str)
            or not report["evaluation_id"]
            or not isinstance(report.get("sources"), dict)
            or not report["sources"]
            or type(report.get("diagnostic")) is not bool
            or report.get("split") != "validation"
            or not isinstance(report.get("control_definitions"), dict)
            or report["control_definitions"].get("D0") !=
               "production deterministic offense controller; native 20 ms physics-step cadence; actual executed actions recorded; not constrained by the MECO candidate table"
            or not isinstance(report.get("policy_manifest"), dict)
            or set(report["policy_manifest"]) != set(POLICIES)
            or not _valid_plan_a_penalty_contract(report.get("plan_a_penalty"))
            or any(not isinstance(value, dict)
                   or not isinstance(value.get("policy_id"), str)
                   or not value.get("policy_id")
                   for value in report["policy_manifest"].values())):
        return None
    scenario = report.get("scenario_id")
    categories, seeds, records = report.get("categories"), report.get("seeds"), report.get("records")
    episodes = report.get("episodes_per_category")
    if (type(episodes) is not int or episodes < 1
            or not isinstance(scenario, str) or not scenario
            or not isinstance(categories, list) or not categories
            or any(not isinstance(category, str) or not category for category in categories)
            or len(set(categories)) != len(categories)
            or not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds) or episodes != len(seeds)
            or report["diagnostic"] is not (episodes < 100)
            or not isinstance(records, list)):
        return None
    try:
        canonical_source_hash = hashlib.sha256(json.dumps(
        report["sources"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError):
        return None
    if (not isinstance(report.get("source_identity_sha256"), str)
            or report["source_identity_sha256"] != canonical_source_hash):
        return None
    expected = {(category, seed, policy) for category in categories
                for seed in seeds for policy in POLICIES}
    found = {}
    route_provenance_summary = {}
    for row in records:
        if (not isinstance(row, dict) or row.get("scenario_id") != scenario
                or row.get("category") not in categories
                or type(row.get("seed")) is not int
                or row.get("policy") not in POLICIES
                or row.get("protocol_id") != report["protocol_id"]
                or row.get("split") != report["split"]
                or row.get("source_identity_sha256") != report["source_identity_sha256"]
                or not isinstance(row.get("policy_id"), str)
                or row.get("policy_id") != report["policy_manifest"].get(row.get("policy"), {}).get("policy_id")
                or (row.get("p_mode") != report["policy_manifest"].get(row.get("policy"), {}).get("p_mode")
                    if row.get("policy") in TRACKS + ABLATIONS else row.get("p_mode") is not None)
                or row.get("checkpoint_sha256") != report["policy_manifest"].get(row.get("policy"), {}).get("checkpoint_sha256")
                or not isinstance(row.get("metrics"), dict)):
            return None
        provenance = report["policy_manifest"][row["policy"]]
        expected_checkpoint = provenance.get("checkpoint_sha256")
        if row["policy"] in TRACKS or row["policy"] in ABLATIONS:
            if (not isinstance(expected_checkpoint, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint)):
                return None
            if row["policy"] in ABLATIONS and provenance.get("p_mode") != "p0":
                return None
        elif expected_checkpoint is not None:
            return None
        if provenance.get("source_identity_sha256") != canonical_source_hash:
            return None
        metrics = row["metrics"]
        if (type(metrics.get("success")) not in (bool, int, float)
                or not _finite_number(metrics.get("success"))
                or not 0 <= metrics["success"] <= 1
                or not _finite_number(metrics.get("useful_throughput"))
                or metrics["useful_throughput"] < 0
                or any(not _finite_number(value) for value in metrics.values()
                       if type(value) in (int, float))):
            return None
        try:
            _collection_action_bouts(row)
        except ValueError:
            return None
        if row["policy"] == "D0":
            trace, actions = row.get("executed_action_trace"), row.get("actions")
            ticks = row.get("elapsed_ticks")
            if (row.get("action_source") != "production_deterministic"
                    or row.get("controller_cadence") != "native_20ms_physics_step"
                    or row.get("action_count_semantics") != "executed_per_physics_step"
                    or type(ticks) is not int or ticks < 1
                    or not isinstance(trace, list) or len(trace) != ticks
                    or any(type(action) is not int or not 0 <= action < 8 for action in trace)
                    or not isinstance(actions, list) or len(actions) != 8
                    or any(type(count) is not int or count < 0 for count in actions)):
                return None
            histogram = [trace.count(action) for action in range(8)]
            if actions != histogram:
                return None
        if row["policy"] in ("A1", "A2", "A3") + ABLATIONS:
            expected_mode = "p0" if row["policy"] in ABLATIONS else row["p_mode"]
            route_trace = row.get("route_trace")
            if (expected_mode not in ("learned", "p0")
                    or not isinstance(route_trace, list) or not route_trace):
                return None
            for route in route_trace:
                penalty = route.get("ppo_path_penalty_points") if isinstance(route, dict) else None
                if (not isinstance(route, dict)
                        or route.get("p_mode") != expected_mode
                        or route.get("ppo_path_penalty_units") != "reward_points"
                        or route.get("ppo_path_penalty_points") is None
                        or not _finite_number(penalty)
                        or route.get("p_map_scale") != report["plan_a_penalty"]["cost_scale"]
                        or route.get("p_points_per_grid_edge") != report["plan_a_penalty"]["points_per_grid_edge"]
                        or route.get("path_points_m") is None
                        or route.get("route_cost_units") != PENALTY_PATH_UNITS):
                    return None
                route_provenance = route.get("route_cost_provenance")
                if expected_mode == "p0":
                    if (abs(float(penalty)) > 1e-9
                            or route_provenance != LEGACY_ROUTE_COST_PROVENANCE):
                        return None
                    summary = route_provenance_summary.setdefault(row["policy"], {
                        "mode": "p0", "route_cost_source": "legacy_no_learned_map",
                        "projection": None, "map_scale": 0.,
                        "route_records": 0,
                        "sample_map_digests": []})
                else:
                    scale_suffix = f"scale={float(report['plan_a_penalty']['cost_scale']).hex()}:path_integral_grid_edge_v2"
                    digest_match = (re.search(r"sha256=([0-9a-f]{64}):", route_provenance)
                                    if isinstance(route_provenance, str) else None)
                    if (not isinstance(route_provenance, str)
                            or not route_provenance.startswith(LEARNED_ROUTE_COST_PREFIX)
                            or digest_match is None
                            or not route_provenance.endswith(scale_suffix)):
                        return None
                    summary = route_provenance_summary.setdefault(row["policy"], {
                        "mode": "learned", "route_cost_source": "weighted_ad_star_map",
                        "projection": PENALTY_PROJECTION,
                        "map_scale": report["plan_a_penalty"]["cost_scale"],
                        "route_records": 0,
                        "sample_map_digests": [], "_digest_samples": set()})
                    digest = digest_match.group(1)
                    if (digest not in summary["_digest_samples"]
                            and len(summary["sample_map_digests"]) < 3):
                        summary["_digest_samples"].add(digest)
                        if len(summary["sample_map_digests"]) < 3:
                            summary["sample_map_digests"].append(digest)
                summary["route_records"] += 1
        key = (row["category"], row["seed"], row["policy"])
        if key in found:
            return None
        found[key] = row
    if set(found) != expected:
        return None
    try:
        source_hash = canonical_source_hash
        signature = {"scenario_id": scenario, "categories": categories,
                     "seeds": seeds, "protocol_id": report["protocol_id"],
                     "source_hash": source_hash}
        comparison_group = hashlib.sha256(json.dumps(signature, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError):
        return None
    for summary in route_provenance_summary.values():
        summary.pop("_digest_samples", None)
    policy_metrics = {policy: {category: _summarize([
        _safe_metrics(found[(category, seed, policy)]) for seed in seeds])
        for category in categories} for policy in POLICIES}
    for policy in POLICIES:
        for category in categories:
            policy_metrics[policy][category]["collection_action_bouts"] = (
                _summarize_collection_action_bouts([
                    found[(category, seed, policy)] for seed in seeds]))
    return {"evaluation_id": report["evaluation_id"],
            "generated_utc": report.get("generated_utc"),
            "comparison_group": comparison_group,
            "protocol_id": report["protocol_id"], "source_hash": source_hash,
            "scenario_id": scenario, "categories": categories, "seeds": seeds,
            "diagnostic": report.get("diagnostic") is True,
            "matched_status": "diagnostic matched" if report.get("diagnostic") is True else "non-diagnostic matched",
            "policy_manifest": report["policy_manifest"],
            "control_definitions": report["control_definitions"],
            "candidate_boundary": report.get("candidate_boundary"),
            "route_cost_provenance": route_provenance_summary,
            "policy_metrics": policy_metrics}


def _dashboard_summary_json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _bounded_json_nodes(value):
    stack = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_DASHBOARD_SUMMARY_JSON_NODES or depth > 48:
            return False
        if isinstance(current, dict):
            if any(not isinstance(key, str) or len(key) > 4096 for key in current):
                return False
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, str) and len(current) > 1024 * 1024:
            return False
        elif current is not None and type(current) not in (str, int, float, bool):
            return False
    return True


def _stable_file_sha256(path: Path, *, max_bytes: int):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError("report is not a bounded regular file")
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    try:
        current = path.lstat()
    except OSError as exc:
        raise ValueError("report changed while being hashed") from exc
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError("report changed while being hashed")
    return digest.hexdigest(), identity(after), after.st_size


def _read_bounded_regular_file(path: Path, *, max_bytes: int):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError("summary is not a bounded regular file")
        data = handle.read(max_bytes + 1)
        after = os.fstat(handle.fileno())
    try:
        current = path.lstat()
    except OSError as exc:
        raise ValueError("summary changed while being read") from exc
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if (len(data) > max_bytes or identity(before) != identity(after)
            or identity(after) != identity(current)):
        raise ValueError("summary changed while being read")
    return data, identity(after)


def _atomic_summary_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_dashboard_comparison_summary_unlocked(report_path: Path, report: dict) -> bool:
    """Atomically publish a bounded projection derived by the dashboard validator."""
    report_path = Path(report_path)
    records = report.get("records") if isinstance(report, dict) else None
    categories = report.get("categories") if isinstance(report, dict) else None
    seeds = report.get("seeds") if isinstance(report, dict) else None
    if (not isinstance(records, list) or len(records) > MAX_DASHBOARD_SUMMARY_RECORDS
            or not isinstance(categories, list) or not isinstance(seeds, list)
            or len(categories) * len(seeds) * len(POLICIES) != len(records)
            or not _bounded_json_nodes(report)):
        return False
    raw_sha256, raw_identity, _ = _stable_file_sha256(
        report_path, max_bytes=MAX_DASHBOARD_SUMMARY_RAW_BYTES)
    comparison = _comparison(report)
    if comparison is None:
        return False
    try:
        current = report_path.lstat()
    except OSError:
        return False
    if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != raw_identity:
        return False
    comparison_bytes = _dashboard_summary_json_bytes(comparison)
    payload = {
        "summary_schema_version": DASHBOARD_SUMMARY_SCHEMA_VERSION,
        "summary_type": "meco-dashboard-comparison",
        "raw_report_sha256": raw_sha256,
        "report_schema_version": report.get("schema_version"),
        "evaluation_id": report.get("evaluation_id"),
        "protocol_id": report.get("protocol_id"),
        "split": report.get("split"),
        "source_identity_sha256": report.get("source_identity_sha256"),
        "record_count": len(records),
        "comparison_sha256": hashlib.sha256(comparison_bytes).hexdigest(),
        "comparison": comparison,
    }
    data = _dashboard_summary_json_bytes(payload)
    if len(data) > MAX_DASHBOARD_SUMMARY_FILE_BYTES:
        return False
    _atomic_summary_write(Path(str(report_path) + DASHBOARD_SUMMARY_SUFFIX), data)
    return True


def _write_dashboard_comparison_summary(report_path: Path, report: dict) -> bool:
    """Allow one bounded projection writer per process at a time."""
    if not _DASHBOARD_SUMMARY_WRITES.acquire(blocking=False):
        return False
    try:
        return _write_dashboard_comparison_summary_unlocked(report_path, report)
    finally:
        _DASHBOARD_SUMMARY_WRITES.release()


def _copy_dashboard_comparison_summary(source_path: Path, destination_path: Path) -> bool:
    """Copy a verified sidecar only when the published raw report is byte-identical."""
    source_path, destination_path = Path(source_path), Path(destination_path)
    source_sidecar = Path(str(source_path) + DASHBOARD_SUMMARY_SUFFIX)
    try:
        source_digest, _, _ = _stable_file_sha256(
            source_path, max_bytes=MAX_DASHBOARD_SUMMARY_RAW_BYTES)
        destination_digest, _, _ = _stable_file_sha256(
            destination_path, max_bytes=MAX_DASHBOARD_SUMMARY_RAW_BYTES)
        if source_digest != destination_digest:
            return False
        data, _ = _read_bounded_regular_file(
            source_sidecar, max_bytes=MAX_DASHBOARD_SUMMARY_FILE_BYTES)
        payload = json.loads(data)
        if _dashboard_summary_valid(payload, source_digest) is None:
            return False
        _atomic_summary_write(Path(str(destination_path) + DASHBOARD_SUMMARY_SUFFIX), data)
        return True
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _dashboard_summary_valid(payload, raw_sha256):
    keys = {"summary_schema_version", "summary_type", "raw_report_sha256",
            "report_schema_version", "evaluation_id", "protocol_id", "split",
            "source_identity_sha256", "record_count", "comparison_sha256",
            "comparison"}
    if (not isinstance(payload, dict) or set(payload) != keys
            or type(payload.get("summary_schema_version")) is not int
            or payload.get("summary_schema_version") != DASHBOARD_SUMMARY_SCHEMA_VERSION
            or payload.get("summary_type") != "meco-dashboard-comparison"
            or payload.get("raw_report_sha256") != raw_sha256
            or type(payload.get("report_schema_version")) is not int
            or payload.get("report_schema_version") != 1
            or not isinstance(payload.get("evaluation_id"), str)
            or not payload["evaluation_id"] or len(payload["evaluation_id"]) > 160
            or not isinstance(payload.get("protocol_id"), str)
            or not payload["protocol_id"] or len(payload["protocol_id"]) > 512
            or payload.get("split") != "validation"
            or not isinstance(payload.get("source_identity_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", payload["source_identity_sha256"])
            or type(payload.get("record_count")) is not int
            or not 1 <= payload["record_count"] <= MAX_DASHBOARD_SUMMARY_RECORDS
            or not isinstance(payload.get("comparison_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", payload["comparison_sha256"])):
        return None
    comparison = payload.get("comparison")
    comparison_keys = {"evaluation_id", "generated_utc", "comparison_group",
        "protocol_id", "source_hash", "scenario_id", "categories", "seeds",
        "diagnostic", "matched_status", "policy_manifest", "control_definitions",
        "candidate_boundary", "route_cost_provenance", "policy_metrics"}
    if (not isinstance(comparison, dict) or set(comparison) != comparison_keys
            or comparison.get("evaluation_id") != payload["evaluation_id"]
            or comparison.get("protocol_id") != payload["protocol_id"]
            or comparison.get("source_hash") != payload["source_identity_sha256"]
            or not isinstance(comparison.get("comparison_group"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", comparison["comparison_group"])
            or type(comparison.get("diagnostic")) is not bool
            or comparison.get("matched_status") != (
                "diagnostic matched" if comparison["diagnostic"] else "non-diagnostic matched")):
        return None
    generated = comparison.get("generated_utc")
    scenario = comparison.get("scenario_id")
    if (generated is not None and (not isinstance(generated, str) or len(generated) > 128)
            or not isinstance(scenario, str) or not scenario or len(scenario) > 128):
        return None
    categories, seeds = comparison.get("categories"), comparison.get("seeds")
    if (not isinstance(categories, list) or not categories or len(categories) > 64
            or any(not isinstance(value, str) or not value or len(value) > 128 for value in categories)
            or len(set(categories)) != len(categories)
            or not isinstance(seeds, list) or not seeds
            or len(seeds) > MAX_DASHBOARD_SUMMARY_RECORDS
            or any(type(value) is not int for value in seeds)
            or len(set(seeds)) != len(seeds)
            or len(categories) * len(seeds) * len(POLICIES) != payload["record_count"]):
        return None
    signature = {"scenario_id": scenario, "categories": categories, "seeds": seeds,
                 "protocol_id": payload["protocol_id"],
                 "source_hash": payload["source_identity_sha256"]}
    expected_group = hashlib.sha256(json.dumps(signature, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if comparison["comparison_group"] != expected_group:
        return None
    manifests = comparison.get("policy_manifest")
    if (not isinstance(manifests, dict) or set(manifests) != set(POLICIES)
            or not isinstance(comparison.get("control_definitions"), dict)
            or set(comparison["control_definitions"]) != set(CONTROLS)
            or comparison["control_definitions"].get("D0") !=
                "production deterministic offense controller; native 20 ms physics-step cadence; actual executed actions recorded; not constrained by the MECO candidate table"
            or any(not isinstance(value, str) or len(value) > 1024
                   for value in comparison["control_definitions"].values())
            or (comparison.get("candidate_boundary") is not None
                and not _bounded_json_nodes(comparison["candidate_boundary"]))
            or not isinstance(comparison.get("route_cost_provenance"), dict)
            or not set(comparison["route_cost_provenance"]) <= set(POLICIES)
            or not isinstance(comparison.get("policy_metrics"), dict)
            or set(comparison["policy_metrics"]) != set(POLICIES)):
        return None
    for policy, manifest in manifests.items():
        if (not isinstance(manifest, dict)
                or not isinstance(manifest.get("policy_id"), str)
                or not manifest["policy_id"] or len(manifest["policy_id"]) > 512
                or manifest.get("source_identity_sha256") != payload["source_identity_sha256"]
                or (manifest.get("checkpoint_sha256") is not None
                    and (not isinstance(manifest["checkpoint_sha256"], str)
                         or not re.fullmatch(r"[0-9a-f]{64}", manifest["checkpoint_sha256"])) )
                or (policy in TRACKS + ABLATIONS
                    and not isinstance(manifest.get("checkpoint_sha256"), str))
                or (policy in CONTROLS and manifest.get("checkpoint_sha256") is not None)
                or (policy in ABLATIONS and manifest.get("p_mode") != "p0")
                or (policy in TRACKS[:3]
                    and manifest.get("p_mode") not in ("learned", "p0"))
                or not _bounded_json_nodes(manifest)):
            return None
    for policy, route in comparison["route_cost_provenance"].items():
        if (not isinstance(route, dict)
                or set(route) != {"mode", "route_cost_source", "projection", "map_scale",
                                  "route_records", "sample_map_digests"}
                or route.get("mode") not in ("p0", "learned")
                or not isinstance(route.get("route_cost_source"), str)
                or route.get("projection") not in (None, PENALTY_PROJECTION)
                or not _finite_number(route.get("map_scale"))
                or type(route.get("route_records")) is not int
                or not 0 <= route["route_records"] <= payload["record_count"]
                or not isinstance(route.get("sample_map_digests"), list)
                or len(route["sample_map_digests"]) > 3
                or any(not isinstance(digest, str)
                       or not re.fullmatch(r"[0-9a-f]{64}", digest)
                       for digest in route["sample_map_digests"])):
            return None
    metric_names = {"success", "useful_throughput", "scores", "acquisitions",
        "deliveries", "collisions", "safety_events", "fallback_count", "fallback_rate",
        "feasible_rate", "heading_violations", "swept_violations", "p50_latency_ms",
        "p95_latency_ms", "p99_latency_ms", "candidate_recall"}
    for policy in POLICIES:
        metrics = comparison["policy_metrics"].get(policy)
        if not isinstance(metrics, dict) or set(metrics) != set(categories):
            return None
        for category_metrics in metrics.values():
            if not isinstance(category_metrics, dict) or "collection_action_bouts" not in category_metrics:
                return None
            if not {"success", "useful_throughput"} <= set(category_metrics):
                return None
            if not set(category_metrics) - {"collection_action_bouts"} <= metric_names:
                return None
            for name, summary in category_metrics.items():
                if name == "collection_action_bouts":
                    bouts_keys = {"available_records", "count", "acquired_count",
                        "unsuccessful_collection_action_bouts",
                        "unsuccessful_collection_action_bouts_available_records",
                        "unsuccessful_collection_action_bouts_total_records",
                        "acquired_proportion", "semantics"}
                    if (not isinstance(summary, dict) or set(summary) != bouts_keys
                            or any(type(summary[key]) is not int or summary[key] < 0
                                   for key in ("available_records",
                                       "unsuccessful_collection_action_bouts_available_records",
                                       "unsuccessful_collection_action_bouts_total_records"))
                            or summary["available_records"] > len(seeds)
                            or summary["unsuccessful_collection_action_bouts_available_records"]
                               > summary["available_records"]
                            or summary["unsuccessful_collection_action_bouts_total_records"] != len(seeds)
                            or summary.get("semantics") != "unsuccessful_collection_action_bouts counts finalized focal collection-action bouts with authoritative acquired quantity known to be zero; it is a descriptive bout outcome, not causal intake failure"):
                        return None
                    count, acquired = summary.get("count"), summary.get("acquired_count")
                    unsuccessful = summary.get("unsuccessful_collection_action_bouts")
                    proportion = summary.get("acquired_proportion")
                    if (count is None and acquired is None):
                        if (summary["available_records"] != 0 or unsuccessful is not None
                                or proportion is not None):
                            return None
                    elif (type(count) is not int or count < 0
                            or type(acquired) is not int or acquired < 0 or acquired > count
                            or (summary["available_records"] == 0)
                            or (summary["unsuccessful_collection_action_bouts_available_records"] == 0
                                and unsuccessful is not None)
                            or (summary["unsuccessful_collection_action_bouts_available_records"] > 0
                                and (type(unsuccessful) is not int or unsuccessful < 0))
                            or (unsuccessful is not None and (unsuccessful > count
                                or (summary["unsuccessful_collection_action_bouts_available_records"]
                                    == summary["available_records"]
                                    and acquired + unsuccessful != count)
                                or (summary["unsuccessful_collection_action_bouts_available_records"]
                                    < summary["available_records"]
                                    and acquired + unsuccessful > count)))
                            or (count == 0 and proportion is not None)
                            or (count > 0 and (not _finite_number(proportion)
                                or not 0 <= proportion <= 1
                                or abs(float(proportion) - acquired / count) > 1e-12))):
                        return None
                    continue
                if (not isinstance(summary, dict)
                        or set(summary) != {"n", "mean", "normal_95_interval"}
                        or type(summary.get("n")) is not int
                        or not 1 <= summary["n"] <= len(seeds)
                        or not _finite_number(summary.get("mean"))):
                    return None
                if (name in ("success", "useful_throughput")
                        and summary["n"] != len(seeds)):
                    return None
                if (name == "success" and not 0 <= summary["mean"] <= 1
                        or name == "useful_throughput" and summary["mean"] < 0):
                    return None
                interval = summary.get("normal_95_interval")
                if (summary["n"] == 1 and interval is not None
                        or summary["n"] > 1 and interval is None):
                    return None
                if (interval is not None and (not isinstance(interval, list)
                        or len(interval) != 2
                        or any(not _finite_number(value) for value in interval)
                        or interval[0] > summary["mean"] or interval[1] < summary["mean"])):
                    return None
    try:
        digest = hashlib.sha256(_dashboard_summary_json_bytes(comparison)).hexdigest()
    except (TypeError, ValueError):
        return None
    return comparison if digest == payload["comparison_sha256"] else None


def _report_binding(report):
    records = report.get("records") if isinstance(report, dict) else None
    return {"report_schema_version": report.get("schema_version"),
            "evaluation_id": report.get("evaluation_id"),
            "protocol_id": report.get("protocol_id"),
            "split": report.get("split"),
            "source_identity_sha256": report.get("source_identity_sha256"),
            "record_count": len(records) if isinstance(records, list) else None}


def _read_evaluation_comparison(path: Path, data_root: Path, budget: dict, *, status=None):
    """Read a small raw report or validate its bounded digest-bound sidecar."""
    if path.is_symlink() or not _inside(path, data_root):
        return None, None, "evaluation report is not a regular in-root file", False
    try:
        size = path.stat().st_size
    except OSError:
        return None, None, "evaluation report is unavailable", False
    if size <= MAX_DASHBOARD_INLINE_REPORT_BYTES:
        report, error = _read_json_within(path, data_root)
        if report is None:
            return None, None, error or "evaluation report is unavailable", False
        comparison = _comparison(report)
        if comparison is None:
            return None, None, "evaluation report is incomplete or invalid", True
        return comparison, _report_binding(report), None, True
    if size > MAX_DASHBOARD_SUMMARY_RAW_BYTES:
        return None, None, "evaluation report exceeds the bounded summary limit", False
    if status is None:
        status, _ = _read_json_within(Path(str(path) + ".status.json"), data_root)
    if (not isinstance(status, dict) or status.get("output_file") != path.name):
        return None, None, "oversized report lacks evaluator summary binding", True
    if status.get("status") == "running":
        return None, None, None, True
    expected_summary_sha256 = status.get("dashboard_summary_sha256")
    if (status.get("status") != "complete"
            or not isinstance(expected_summary_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_summary_sha256)):
        return None, None, "oversized report lacks evaluator summary binding", True
    sidecar_path = Path(str(path) + DASHBOARD_SUMMARY_SUFFIX)
    if sidecar_path.is_symlink() or not _inside(sidecar_path, data_root):
        return None, None, "evaluation summary is not a regular in-root file", False
    try:
        sidecar_identity = sidecar_path.lstat()
        if not stat.S_ISREG(sidecar_identity.st_mode):
            raise ValueError("dashboard summary is not a regular file")
        sidecar_size = sidecar_identity.st_size
    except OSError:
        return None, None, "oversized evaluation report has no dashboard summary", False
    except ValueError:
        return None, None, "dashboard summary is not a regular in-root file", False
    if (sidecar_size > MAX_DASHBOARD_SUMMARY_FILE_BYTES
            or budget["files"] >= MAX_DASHBOARD_SUMMARY_FILES_PER_SCAN
            or budget["summary_bytes"] + sidecar_size > MAX_DASHBOARD_SUMMARY_BYTES_PER_SCAN
            or budget["raw_bytes"] + size > MAX_DASHBOARD_SUMMARY_RAW_BYTES_PER_SCAN):
        return None, None, "dashboard summary scan budget exceeded", False
    budget["files"] += 1
    budget["summary_bytes"] += sidecar_size
    budget["raw_bytes"] += size
    try:
        with _DASHBOARD_SUMMARY_READS:
            raw_sha256, raw_identity, _ = _stable_file_sha256(
                path, max_bytes=MAX_DASHBOARD_SUMMARY_RAW_BYTES)
            summary_bytes, read_identity = _read_bounded_regular_file(
                sidecar_path, max_bytes=MAX_DASHBOARD_SUMMARY_FILE_BYTES)
            if hashlib.sha256(summary_bytes).hexdigest() != expected_summary_sha256:
                raise ValueError("dashboard summary differs from evaluator terminal binding")
            if (len(summary_bytes) != sidecar_size
                    or read_identity != (sidecar_identity.st_dev, sidecar_identity.st_ino,
                        sidecar_identity.st_size, sidecar_identity.st_mtime_ns)):
                raise ValueError("dashboard summary changed while being read")
            payload = json.loads(summary_bytes)
            comparison = _dashboard_summary_valid(payload, raw_sha256)
            if comparison is None:
                raise ValueError("dashboard summary schema or binding is invalid")
            current = path.lstat()
            current_identity = (current.st_dev, current.st_ino,
                                current.st_size, current.st_mtime_ns)
            if current_identity != raw_identity:
                raise ValueError("evaluation report changed while being verified")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None, None, "dashboard summary is unavailable or invalid", False
    binding = {key: payload[key] for key in (
        "report_schema_version", "evaluation_id", "protocol_id", "split",
        "source_identity_sha256", "record_count")}
    if any(status.get(key) != binding[key]
           for key in ("evaluation_id", "protocol_id", "split")):
        return None, None, "dashboard summary status binding is invalid", False
    if budget["records"] + binding["record_count"] > MAX_DASHBOARD_SUMMARY_RECORDS_PER_SCAN:
        return None, None, "dashboard summary record budget exceeded", False
    budget["records"] += binding["record_count"]
    return comparison, binding, None, True


def _qualification_status(project: Path, errors: list, data_root: Path):
    root = project / "qualifications" / "meco"
    path = root / "qualification-status.json"
    status, error = _read_json_within(path, data_root)
    if status is None:
        if error and path.exists():
            errors.append({"source": str(path.relative_to(project)), "error": error})
        return None
    failure = status.get("failure")
    promotion = status.get("promotion")
    bundle_id = status.get("bundle_id")
    valid_bundle = (isinstance(bundle_id, str)
                    and re.fullmatch(r"[0-9a-f]{64}", bundle_id))
    # Qualification provenance is intentionally projected field by field. In
    # particular, the calibration seed pools and raw qualification evidence
    # remain private to the qualification artifact.
    from .meco_qualification import (CALIBRATED_PROTOCOL_VERSION,
        D0_PAIRED_REFERENCE_PROTOCOL_VERSION, LEGACY_CALIBRATED_PROTOCOL_VERSION,
        PROTOCOL_VERSION)
    from .training_curriculum import SCENARIOS, _ancestors

    protocol_version = status.get("protocol_version")
    scenario_id = status.get("scenario_id")
    calibration_id = status.get("calibration_id")
    calibration_ids = status.get("calibration_ids")
    retention_scenario_ids = status.get("retention_scenario_ids")
    safety_gate_status = status.get("safety_gate_status")
    valid_hash = lambda value: (type(value) is str
                                and re.fullmatch(r"[0-9a-f]{64}", value) is not None)
    valid_scenario = type(scenario_id) is str and scenario_id in SCENARIOS
    closure = ({scenario_id, *_ancestors(scenario_id)} if valid_scenario else set())
    expected_calibration_keys = {name for name in SCENARIOS if name in closure}
    expected_retention = [name for name in SCENARIOS
                          if name in closure and name != scenario_id]
    valid_calibration_ids = (
        type(calibration_ids) is dict and len(calibration_ids) <= 32
        and all(type(key) is str and key in SCENARIOS and valid_hash(value)
                for key, value in calibration_ids.items()))
    valid_retention_ids = (
        type(retention_scenario_ids) is list and len(retention_scenario_ids) <= 32
        and all(type(name) is str and name in SCENARIOS
                for name in retention_scenario_ids)
        and len(set(retention_scenario_ids)) == len(retention_scenario_ids))
    calibrated_status = {
        LEGACY_CALIBRATED_PROTOCOL_VERSION: "calibrated_baseline_frozen",
        D0_PAIRED_REFERENCE_PROTOCOL_VERSION:
            "d0_operational_margin_frozen_reference_variance_unverified",
        CALIBRATED_PROTOCOL_VERSION:
            "matched_reference_variance_and_reviewed_margin_frozen"}
    if protocol_version in calibrated_status:
        combined_calibration_id = (hashlib.sha256(json.dumps(calibration_ids,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if valid_calibration_ids else None)
        valid_calibration = (valid_hash(calibration_id) and valid_calibration_ids
            and set(calibration_ids) == expected_calibration_keys
            and all(calibration_ids[name] for name in expected_calibration_keys)
            and calibration_id == combined_calibration_id
            and valid_retention_ids and retention_scenario_ids == expected_retention
            and safety_gate_status == "metrics_schema_verified"
            and status.get("gate_calibration_status") == calibrated_status[protocol_version])
    elif protocol_version == PROTOCOL_VERSION:
        # Older v1 status files predate these projection fields. Permit only
        # absent/null or canonical empty calibration containers for them.
        valid_calibration = (
            calibration_id is None
            and (calibration_ids is None or
                 type(calibration_ids) is dict and not calibration_ids)
            and (retention_scenario_ids is None or
                 type(retention_scenario_ids) is list and not retention_scenario_ids)
            and safety_gate_status in (None, "not_calibrated")
            and status.get("gate_calibration_status") in
                (None, "provisional_pending_baseline_variance_review"))
    else:
        valid_calibration = False
    failure_messages = failure.get("failures") if isinstance(failure, dict) else None
    failed_round = (
        protocol_version in (PROTOCOL_VERSION, *calibrated_status)
        and valid_scenario
        and isinstance(promotion, dict) and promotion.get("passed") is False
        and isinstance(failure, dict)
        and failure.get("scenario_id") == scenario_id
        and failure.get("protocol_id") == status.get("protocol_id")
        and failure.get("passed") is False and failure.get("streak_reset") is True
        and isinstance(failure_messages, list) and 0 < len(failure_messages) <= 32
        and all(type(message) is str and len(message) <= 2048
                for message in failure_messages)
        and status.get("evaluation_id") is None and bundle_id is None
        and status.get("split") == "validation")
    if (status.get("schema_version") != 1
            or status.get("split") not in ("validation", "test")
            or not valid_scenario
            or not isinstance(status.get("protocol_id"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", status["protocol_id"])
            or (not valid_bundle and not failed_round)
            or (status.get("checkpoint_sha256") is not None
                and not valid_hash(status.get("checkpoint_sha256")))
            or not isinstance(status.get("confidence_intervals"), dict)
            or not isinstance(status.get("promotion"), dict)
            or not valid_calibration):
        errors.append({"source": str(path.relative_to(project)),
                       "error": "qualification status has invalid provenance"})
        return None
    projected = {key: status.get(key) for key in (
        "updated_utc", "evaluation_id", "scenario_id", "candidate_policy", "split",
        "protocol_id", "bundle_id", "checkpoint_sha256", "gate_definition",
        "gate_calibration_status", "protocol_version", "calibration_id",
        "calibration_ids", "retention_scenario_ids", "safety_gate_status",
        "confidence_intervals", "promotion", "streak", "qualified",
        "final_test_isolated")}
    if isinstance(failure, dict):
        # Expose the reset outcome and protocol binding without projecting the
        # checkpoint path or free-form exception text from the raw artifact.
        projected["failure"] = {
            "scenario_id": failure.get("scenario_id"),
            "protocol_id": failure.get("protocol_id"),
            "passed": failure.get("passed"),
            "streak_reset": failure.get("streak_reset"),
            "failure_count": len(failure_messages) if isinstance(failure_messages, list) else None}
        if failed_round:
            projected["promotion"] = {"passed": False, "promoted": False,
                                       "failure_count": len(failure_messages)}
    return projected


def _collect_status(checkpoint_root: Path, evaluation_root: Path, *, project_root=None, now=None,
                    campaign_passes=None,
                    queue_root: Path | None = None,
                    process_probe=_process_alive):
    now = time.time() if now is None else float(now)
    root = checkpoint_root.resolve()
    eval_root = evaluation_root.resolve()
    data_root = (Path(project_root).resolve() if project_root is not None
                 else root.parents[1].resolve())
    if not _inside(root, data_root) or not _inside(eval_root, data_root):
        raise ValueError("MECO status roots must remain inside the selected data root")
    queue_path_root = (Path(queue_root).resolve() if queue_root is not None else root)
    if not _inside(queue_path_root, data_root):
        raise ValueError("MECO queue root must remain inside the selected data root")
    queue = _queue_state(queue_path_root, now, data_root)
    run_by_track = {track: [] for track in TRACKS}
    errors = []
    if root.is_dir():
        # Only immediate child directories under the pinned root are trusted.
        for path in sorted(root.iterdir()):
            if not path.is_dir() or not _inside(path, root):
                continue
            manifest, error = _read_json_within(path / "run.json", data_root)
            if manifest is None:
                if (path / "run.json").exists():
                    errors.append({"source": path.name, "error": error or "invalid run manifest"})
                continue
            config = manifest.get("config")
            track = manifest.get("track", config.get("track") if isinstance(config, dict) else None)
            if track not in TRACKS:
                continue
            heartbeat = _heartbeat(path, manifest, now, data_root, process_probe)
            state = manifest.get("status", "unknown")
            stats, stats_error = _read_json_within(path / "stats.json", data_root)
            latest = None
            if stats:
                history = stats.get("history", [])
                if isinstance(history, list) and history:
                    candidate = history[-1]
                    if isinstance(candidate, dict):
                        latest = {key: candidate[key] for key in (
                            "update", "episode", "entropy", "rollout_return",
                            "policy_loss", "value_loss", "update_utc") if key in candidate}
            lane = queue["lanes"].get(track, {})
            if state == "running":
                stalled, unchanged_seconds = _worker_stalled(
                    manifest=manifest, heartbeat=heartbeat, stats=stats,
                    latest=latest, lane=lane, queue=queue, now=now)
                heartbeat["unchanged_update_frontier_seconds"] = unchanged_seconds
                heartbeat["worker_stall_threshold_seconds"] = WORKER_STALL_SECONDS
                state = ("stalled" if stalled else
                         "running" if heartbeat["state"] == "confirmed" else "unconfirmed")
            elif state == "queued":
                state = ("queued" if queue["state"] == "fresh"
                         and lane.get("run_id") == manifest.get("run_id") else "unconfirmed")
            elif state not in ("complete", "stopped", "failed",
                                "aborted_due_to_sibling_failure", "unknown"):
                state = "unknown"
            if (lane.get("run_id") == manifest.get("run_id")
                    and lane.get("pid") == heartbeat.get("pid")):
                lane["worker_state"] = (state if state in ("running", "stalled") else
                                         "unconfirmed" if manifest.get("status") == "running" else
                                         state)
            checkpoint = path / "latest.pt"
            checkpoint_info = None
            if checkpoint.is_file() and _inside(checkpoint, root):
                try:
                    stat = checkpoint.stat()
                    checkpoint_info = {"path": str(checkpoint.relative_to(root)),
                                       "bytes": stat.st_size,
                                       "modified_utc": datetime.fromtimestamp(
                                           stat.st_mtime, timezone.utc).isoformat()}
                except OSError:
                    pass
            run_by_track[track].append({"run_id": manifest.get("run_id"),
                "run_directory": path.name, "training_state": state,
                "manifest_state": manifest.get("status", "unknown"),
                "heartbeat": heartbeat,
                "track": track, "config": manifest.get("config", {}),
                "source_identity_sha256": manifest.get("source_identity_sha256"),
                "sources": manifest.get("sources", {}),
                "graph": manifest.get("graph"),
                "candidate_boundary": manifest.get("candidate_boundary"),
                "completed_updates": (stats.get("update") if stats else
                                      manifest.get("completed_updates", 0)),
                "latest_training_metrics": latest,
                "latest_checkpoint": checkpoint_info,
                "error": stats_error if stats is None and (path / "stats.json").exists() else None})

    if queue.get("state") == "fresh":
        active_lanes = [lane for lane in queue.get("lanes", {}).values()
                        if lane.get("status") in ("running", "stalled")]
        queue["worker_state"] = ("stalled" if any(lane.get("worker_state") == "stalled"
                                                    for lane in active_lanes) else
                                  "unconfirmed" if any(lane.get("worker_state") != "running"
                                                       for lane in active_lanes) else
                                  "active" if active_lanes else
                                  "queued" if any(lane.get("status") in (
                                      "queued", "deferred_for_evaluation")
                                                  for lane in queue.get("lanes", {}).values())
                                  else "idle")

    comparisons = []
    evaluation_runs = []
    comparison_cache = {}
    diagnostic_status_outputs = set()
    summary_budget = {"files": 0, "summary_bytes": 0, "raw_bytes": 0, "records": 0}

    def comparison_for(report_path):
        cache_key = str(report_path)
        if cache_key not in comparison_cache:
            comparison_cache[cache_key] = _read_evaluation_comparison(
                report_path, data_root, summary_budget)
        return comparison_cache[cache_key]

    if eval_root.is_dir():
        for status_path in sorted(eval_root.glob("*.status.json")):
            status, error = _read_json_within(status_path, data_root)
            if status is None:
                errors.append({"source": status_path.name, "error": error or "invalid evaluator status"})
                evaluation_runs.append({"evaluation_id": None,
                    "status": "unconfirmed/invalid",
                    "execution_status": "unconfirmed/invalid",
                    "report_validation": "unconfirmed",
                    "evidence_error": error or "invalid evaluator status",
                    "heartbeat_state": "stale/missing",
                    "heartbeat_age_seconds": None, "process_alive": False,
                    "protocol_id": None, "split": None, "diagnostic": False,
                    "progress": None, "output_file": None,
                    "error": error or "invalid evaluator status"})
                continue
            stamp = _utc_timestamp(status.get("heartbeat_utc"))
            age = None if stamp is None else max(0., now - stamp)
            fresh = age is not None and age <= EVALUATION_HEARTBEAT_STALE_SECONDS and stamp <= now + 5.
            output_name = status.get("output_file")
            safe_name = (isinstance(output_name, str) and bool(output_name)
                and "\x00" not in output_name and len(output_name) <= 255
                and output_name not in (".", "..")
                and Path(output_name).name == output_name)
            report_path = eval_root / output_name if safe_name else eval_root / "__invalid__"
            safe_output_path = (safe_name and _inside(report_path, data_root)
                and not report_path.is_symlink()
                and (not report_path.exists() or report_path.is_file()))
            if safe_output_path and status.get("split") == "diagnostic":
                diagnostic_status_outputs.add(output_name)
            valid_report, binding, report_error, report_visible = (
                _read_evaluation_comparison(report_path, data_root, summary_budget,
                    status=status) if safe_output_path else (None, None,
                    "evaluator output file is not a safe in-root basename", False))
            if safe_output_path:
                comparison_cache[str(report_path)] = (
                    valid_report, binding, report_error, report_visible)
            identity_valid = (status.get("schema_version") == 1
                and isinstance(status.get("evaluation_id"), str)
                and bool(status.get("evaluation_id"))
                and len(status["evaluation_id"]) <= 160
                and isinstance(status.get("protocol_id"), str)
                and bool(status.get("protocol_id"))
                and len(status["protocol_id"]) <= 160
                and status.get("split") in ("validation", "test", "diagnostic")
                and safe_output_path)
            report_matches_status = (valid_report is not None
                and isinstance(binding, dict)
                and binding.get("evaluation_id") == status.get("evaluation_id")
                and binding.get("protocol_id") == status.get("protocol_id")
                and binding.get("split") == status.get("split"))
            diagnostic_complete = (identity_valid
                and status.get("status") == "complete"
                and status.get("split") == "diagnostic")
            consistent = (identity_valid and
                (status.get("status") == "running" or report_matches_status))
            alive = bool(process_probe(status.get("pid")))
            state = status.get("status", "unknown")
            if state == "running":
                state = "running" if fresh and alive and consistent else "stale/unconfirmed"
            elif diagnostic_complete:
                state = "status-reported complete"
            elif state == "complete" and not consistent:
                state = "invalid"
            elif state not in ("complete", "failed", "stopped"):
                state = "unknown"
            if not safe_output_path:
                report_error = "evaluator output file is not a safe in-root basename"
            if report_error is not None:
                if state == "running" and report_error == "evaluation report is unavailable":
                    report_validation = "pending"
                    evidence_error = None
                elif report_error == "oversized report lacks evaluator summary binding":
                    report_validation = "unvalidated"
                    evidence_error = report_error
                elif report_error in ("evaluation report is unavailable",
                        "evaluation report is not a regular in-root file"):
                    report_validation = "unavailable"
                    evidence_error = report_error
                else:
                    report_validation = "invalid"
                    evidence_error = report_error
            elif valid_report is not None and report_matches_status:
                report_validation = "validated"
                evidence_error = None
            elif state == "running":
                report_validation = "pending"
                evidence_error = None
            else:
                report_validation = "invalid" if valid_report is not None else "unconfirmed"
                evidence_error = ("evaluator status/report provenance mismatch"
                    if valid_report is not None else None)
            projected_status = state
            projected_error = (evidence_error if diagnostic_complete else
                None if consistent else "evaluator status/report provenance mismatch")
            evaluation_runs.append({"evaluation_id": status.get("evaluation_id"),
                "status": projected_status, "execution_status": projected_status,
                "diagnostic": diagnostic_complete,
                "report_validation": report_validation,
                "evidence_error": evidence_error,
                "heartbeat_state": "fresh" if fresh else "stale/missing",
                "heartbeat_age_seconds": age, "process_alive": alive,
                "protocol_id": status.get("protocol_id"), "split": status.get("split"),
                "progress": status.get("progress"), "output_file": output_name,
                "error": projected_error})
        for path in sorted(eval_root.glob("*.json")):
            if (path.name.endswith(".status.json")
                    or path.name.endswith(DASHBOARD_SUMMARY_SUFFIX)):
                continue
            if not path.is_file() or not _inside(path, data_root):
                continue
            comparison, _, error, _ = comparison_for(path)
            if path.name in diagnostic_status_outputs and comparison is not None:
                comparison = None
                error = "diagnostic evaluator status is not comparison evidence"
            if error:
                errors.append({"source": path.name, "error": error or "invalid evaluation"})
                continue
            if comparison is not None:
                comparison["evidence_file"] = path.name
                try:
                    comparison["_mtime"] = path.stat().st_mtime
                except OSError:
                    comparison["_mtime"] = 0
                comparisons.append(comparison)
    grouped = {}
    for row in comparisons:
        grouped.setdefault(row["comparison_group"], []).append(row)
    groups = [{"comparison_group": key,
               "evaluation_ids": [row["evaluation_id"] for row in rows],
               "evidence_files": [row["evidence_file"] for row in rows],
               "protocol_id": rows[-1]["protocol_id"],
               "source_hash": rows[-1]["source_hash"],
               "scenario_id": rows[-1]["scenario_id"],
               "categories": rows[-1]["categories"], "seeds": rows[-1]["seeds"],
               "diagnostic": any(row["diagnostic"] for row in rows),
               "matched_status": rows[-1]["matched_status"],
               "record_count": len(rows[-1]["seeds"]) * len(rows[-1]["categories"]),
               "route_cost_provenance": rows[-1]["route_cost_provenance"],
               "candidate_boundary": rows[-1].get("candidate_boundary"),
               "policy_metrics": rows[-1]["policy_metrics"]}
              for key, rows in sorted(grouped.items())]
    latest_by_policy = {}
    for comparison in sorted(comparisons, key=lambda row: (row["_mtime"], row["evaluation_id"])):
        for policy, metrics in comparison["policy_metrics"].items():
            latest_by_policy[policy] = {"evaluation_id": comparison["evaluation_id"],
                "comparison_group": comparison["comparison_group"],
                "protocol_id": comparison["protocol_id"],
                "source_hash": comparison["source_hash"],
                "scenario_id": comparison["scenario_id"],
                "diagnostic": comparison["diagnostic"],
                "matched_status": comparison["matched_status"],
                "policy_manifest": comparison["policy_manifest"][policy],
                "route_cost_provenance": comparison["route_cost_provenance"].get(policy),
                "records": metrics}
    policies = []
    for policy in POLICIES:
        runs = run_by_track.get(policy, [])
        lane = queue.get("lanes", {}).get(policy) if policy in TRACKS else None
        current_queue_attempt = (queue.get("state") == "fresh" and lane is not None
            and not any(run.get("run_id") == lane.get("run_id") for run in runs))
        if current_queue_attempt:
            history = sorted(runs, key=lambda row: (
                row.get("training_state") in ("running", "queued", "unconfirmed", "stalled"),
                row.get("heartbeat", {}).get("fresh", False),
                row.get("manifest_state") == "queued",
                row.get("completed_updates", 0), row["run_id"] or ""))
            latest_run = _queue_attempt_projection(policy, lane, queue_fresh=True,
                previous_run=history[-1] if history else None)
        else:
            runs.sort(key=lambda row: (
                row.get("training_state") in ("running", "queued", "unconfirmed", "stalled"),
                row.get("heartbeat", {}).get("fresh", False),
                row.get("manifest_state") == "queued",
                row.get("completed_updates", 0), row["run_id"] or ""))
            latest_run = runs[-1] if runs else None
        if latest_run is None and lane:
            latest_run = _queue_attempt_projection(policy, lane,
                queue_fresh=False)
        evidence = latest_by_policy.get(policy)
        policies.append({"policy_id": policy,
            "kind": ("learned_track" if policy in TRACKS else
                     "P=0 ablation" if policy in ABLATIONS else "deterministic_control"),
            "training_state": latest_run["training_state"] if latest_run else
                ("not_applicable" if policy in CONTROLS or policy in ABLATIONS else "not_started"),
            "evaluation_state": evidence["matched_status"] if evidence else "not_evaluated",
            "run": latest_run, "latest_evidence": evidence})
    qualification = _qualification_status(data_root, errors, data_root)
    return {"schema_version": STATUS_SCHEMA_VERSION,
        "generated_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "data_root": str(data_root),
        "checkpoint_root": str(root), "evaluation_root": str(eval_root),
        "heartbeat_stale_seconds": HEARTBEAT_STALE_SECONDS,
        "worker_stall_seconds": WORKER_STALL_SECONDS,
        "queue_stale_seconds": QUEUE_STALE_SECONDS,
        "evaluation_heartbeat_stale_seconds": EVALUATION_HEARTBEAT_STALE_SECONDS,
        "queue": queue, "policies": policies, "evaluation_runs": evaluation_runs,
        "implementation_passes": campaign_passes or [],
        "qualification": qualification,
        "comparison_groups": groups,
        "errors": errors[:20]}


def meco_status_payload(project_root: Path | str, *, data_root: Path | str | None = None) -> bytes:
    """Read run evidence from data_root while keeping workflow evidence checkout-local."""
    project = Path(project_root).resolve()
    data_project = Path(data_root).resolve() if data_root is not None else project
    if not data_project.is_dir():
        raise ValueError("MECO data root must be an existing directory")
    legacy_checkpoint_root = data_project / "checkpoints" / "meco-five"
    legacy_evaluation_root = data_project / "evaluations" / "meco-five"
    direct_checkpoint_root = data_project / "checkpoints"
    direct_tracks = [direct_checkpoint_root / track for track in TRACKS]
    direct_layout_present = any(path.exists() or path.is_symlink() for path in direct_tracks)
    legacy_layout_present = (legacy_checkpoint_root.exists() or
                             legacy_evaluation_root.exists())
    if direct_layout_present and legacy_layout_present:
        raise ValueError("MECO data root contains ambiguous legacy and direct campaign layouts")
    if direct_layout_present:
        direct_evaluation_root = data_project / "evaluations"
        if (direct_checkpoint_root.is_symlink() or not direct_checkpoint_root.is_dir()
                or not _inside(direct_checkpoint_root, data_project)):
            raise ValueError("direct MECO checkpoints root must be a real in-root directory")
        if (direct_evaluation_root.exists() or direct_evaluation_root.is_symlink()) and (
                direct_evaluation_root.is_symlink() or not direct_evaluation_root.is_dir()
                or not _inside(direct_evaluation_root, data_project)):
            raise ValueError("direct MECO evaluations root must be a real in-root directory")
        if not all(path.is_dir() and not path.is_symlink()
                   and _inside(path, data_project) for path in direct_tracks):
            raise ValueError(
                "direct MECO campaign layout must contain five real, non-symlink "
                "in-root track directories")
        checkpoint_root = direct_checkpoint_root.resolve()
        evaluation_root = direct_evaluation_root.resolve()
        queue_root = data_project
    else:
        checkpoint_root = legacy_checkpoint_root.resolve()
        evaluation_root = legacy_evaluation_root.resolve()
        queue_root = checkpoint_root
    if not _inside(checkpoint_root, data_project) or not _inside(evaluation_root, data_project):
        raise ValueError("MECO status roots must remain inside the selected data root")
    payload = _collect_status(checkpoint_root, evaluation_root, project_root=data_project,
                              queue_root=queue_root)
    pass_path = project / "docs" / "MECO_CAMPAIGN_PASSES.json"
    passes, pass_error = _read_json(pass_path)
    valid_passes = None
    if (passes is not None and passes.get("schema_version") == 1
            and passes.get("pass_status_is_agent_liveness") is False
            and isinstance(passes.get("passes"), list)
            and [item.get("pass") for item in passes["passes"] if isinstance(item, dict)]
                == ["planner", "implementer", "reviewer"]
            and all(item.get("status") in ("pending", "in_progress", "complete", "blocked")
                    and isinstance(item.get("provenance"), list)
                    for item in passes["passes"])):
        valid_passes = {"campaign": passes.get("campaign"),
            "agent_liveness": False, "passes": passes["passes"]}
        factor_status = passes.get("phase4_factors")
        valid_factor_statuses = {"planned", "pending", "in_progress", "complete", "blocked"}
        expected_factor_ids = ["feinting_opponent", "observation_delay",
            "independent_low_confidence", "abrupt_opponent_policy_change"]
        if (isinstance(factor_status, dict)
                and factor_status.get("schema_version") == 1
                and factor_status.get("planner_status") in valid_factor_statuses
                and isinstance(factor_status.get("factors"), list)
                and [factor.get("factor_id") for factor in factor_status["factors"]
                     if isinstance(factor, dict)] == expected_factor_ids
                and all(isinstance(factor, dict)
                    and isinstance(factor.get("factor_id"), str)
                    and factor.get("status") in valid_factor_statuses
                    and factor.get("planner_status") in valid_factor_statuses
                    and factor.get("implementation_status") in valid_factor_statuses
                    and factor.get("review_status") in valid_factor_statuses
                    and factor.get("evaluation_status") in valid_factor_statuses
                    and isinstance(factor.get("summary"), str)
                    for factor in factor_status["factors"])):
            valid_passes["phase4_planner_status"] = factor_status["planner_status"]
            valid_passes["phase4_factors"] = factor_status["factors"]
    payload["implementation_passes"] = valid_passes
    payload["campaign_ledger_path"] = str(pass_path)
    if valid_passes is None:
        payload["errors"].append({"source": str(pass_path.relative_to(project)),
            "error": pass_error or "invalid implementation pass status manifest"})
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
