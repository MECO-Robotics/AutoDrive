"""Read-only v1 status projection for the MECO five-track experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


TRACKS = ("A1", "A2", "A3", "C1", "C2")
ABLATIONS = ("A1_P0", "A2_P0", "A3_P0")
CONTROLS = ("D0", "D1", "D2")
POLICIES = TRACKS + ABLATIONS + CONTROLS
PENALTY_PROJECTION = "mean_over_legal_heading_bins_xy_grid_v1"
PENALTY_MAP_UNITS = "dimensionless_additive_penalty_per_cardinal_grid_step"
PENALTY_PATH_UNITS = "normalized_grid_edge_units"
PENALTY_PPO_CONTRACT = "current_policy_penalty_integrated_on_frozen_rollout_routes"
LEARNED_ROUTE_COST_PREFIX = (
    "meco_legal_heading_penalty_projection_v1:schema=xy_grid_f64le_v2:")
LEGACY_ROUTE_COST_PROVENANCE = "legacy_bump_cost_path_integral_v2"
STATUS_SCHEMA_VERSION = 1
HEARTBEAT_STALE_SECONDS = 90
WORKER_STALL_SECONDS = 600
QUEUE_STALE_SECONDS = 180
EVALUATION_HEARTBEAT_STALE_SECONDS = 90
MAX_STATUS_FILE_BYTES = 32 * 1024 * 1024
TRUSTED_GIT_EXECUTABLE = "/usr/bin/git"


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
    raw_lanes = payload.get("lanes")
    if isinstance(raw_lanes, list):
        for lane in raw_lanes:
            if (not isinstance(lane, dict) or lane.get("track") not in TRACKS
                    or not isinstance(lane.get("run_id"), str)):
                continue
            pid = lane.get("pid") if type(lane.get("pid")) is int else None
            process = _process_evidence(pid, expected_run_id=lane["run_id"])
            lanes[lane["track"]] = {"run_id": lane["run_id"],
                "status": lane.get("status", "unknown"),
                "pid": pid,
                "process_confirmed": process["confirmed"],
                "process_state": process["state"],
                "process_error": process["error"],
                "worker_state": ("unconfirmed" if lane.get("status") in ("running", "stalled")
                                 else lane.get("status", "unknown")),
                "device": lane.get("device") if isinstance(lane.get("device"), str) else None,
                "run_directory": lane.get("run_directory") if isinstance(
                    lane.get("run_directory"), str) else None,
                "completed_updates": lane.get("completed_updates")
                    if type(lane.get("completed_updates")) is int else None,
                "chunk_updates": lane.get("chunk_updates")
                    if type(lane.get("chunk_updates")) is int else None}
    else:
        error = "queue lanes are malformed"
    active_lanes = [lane for lane in lanes.values()
                    if lane.get("status") in ("running", "stalled")]
    worker_state = ("unknown" if not fresh else
                    "unconfirmed" if active_lanes else
                    "queued" if any(lane.get("status") == "queued"
                                     for lane in lanes.values()) else "idle")
    return {"state": "fresh" if fresh else "stale", "worker_state": worker_state,
            "age_seconds": age,
            "lanes": lanes,
            "phase": payload.get("phase") if isinstance(payload.get("phase"), str) else None,
            "cycle": payload.get("cycle") if type(payload.get("cycle")) is int else None,
            "evaluation": payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else None,
            "supervisor_id": payload.get("supervisor_id")
                if isinstance(payload.get("supervisor_id"), str) else None,
            "error": error if not fresh else None}


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
    availability = ledger.get("availability")
    available = (availability.get("collection_action_bouts")
                 if isinstance(availability, dict) else None)
    if available is not True:
        return None
    bouts = ledger.get("collection_action_bouts")
    if not isinstance(bouts, dict):
        raise ValueError("available collection_action_bouts must be an object")
    count, acquired = bouts.get("count"), bouts.get("acquired_count")
    proportion = bouts.get("acquired_proportion")
    if (type(count) is not int or count < 0
            or type(acquired) is not int or acquired < 0 or acquired > count):
        raise ValueError("invalid available collection_action_bouts counts")
    if count == 0:
        if proportion is not None:
            raise ValueError("zero collection action bouts must have null proportion")
    elif (type(proportion) not in (int, float)
            or not _finite_number(proportion)
            or not 0 <= proportion <= 1
            or abs(float(proportion) - acquired / count) > 1e-12):
        raise ValueError("invalid available collection_action_bouts proportion")
    return {"count": count, "acquired_count": acquired,
            "acquired_proportion": proportion}


def _summarize_collection_action_bouts(rows):
    """Pool only available episode aggregates; keep no-data values null."""
    values = [value for value in (_collection_action_bouts(row) for row in rows)
              if value is not None]
    count = sum(value["count"] for value in values) if values else None
    acquired = sum(value["acquired_count"] for value in values) if values else None
    return {
        "available_records": len(values),
        "count": count,
        "acquired_count": acquired,
        "acquired_proportion": (acquired / count if count else None),
        "semantics": "descriptive acquisition proportion among contiguous executed collection-action bouts; not failed-collection rate or causal policy success",
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
            or not isinstance(report.get("plan_a_penalty"), dict)
            or report["plan_a_penalty"].get("trained_mode") not in ("learned", "p0")
            or report["plan_a_penalty"].get("p0_ablation_mode") != "p0"
            or report["plan_a_penalty"].get("projection") != PENALTY_PROJECTION
            or report["plan_a_penalty"].get("map_units") != PENALTY_MAP_UNITS
            or report["plan_a_penalty"].get("path_units") != PENALTY_PATH_UNITS
            or report["plan_a_penalty"].get("ppo_recomputation") != PENALTY_PPO_CONTRACT
            or not _finite_number(report["plan_a_penalty"].get("cost_scale"))
            or report["plan_a_penalty"].get("cost_scale") < 0
            or not _finite_number(report["plan_a_penalty"].get("points_per_grid_edge"))
            or report["plan_a_penalty"].get("points_per_grid_edge") < 0
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
    from .meco_qualification import CALIBRATED_PROTOCOL_VERSION, PROTOCOL_VERSION
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
    if protocol_version == CALIBRATED_PROTOCOL_VERSION:
        combined_calibration_id = (hashlib.sha256(json.dumps(calibration_ids,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if valid_calibration_ids else None)
        valid_calibration = (valid_hash(calibration_id) and valid_calibration_ids
            and set(calibration_ids) == expected_calibration_keys
            and all(calibration_ids[name] for name in expected_calibration_keys)
            and calibration_id == combined_calibration_id
            and valid_retention_ids and retention_scenario_ids == expected_retention
            and safety_gate_status == "metrics_schema_verified"
            and status.get("gate_calibration_status") == "calibrated_baseline_frozen")
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
        protocol_version in (PROTOCOL_VERSION, CALIBRATED_PROTOCOL_VERSION)
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
                    process_probe=_process_alive):
    now = time.time() if now is None else float(now)
    root = checkpoint_root.resolve()
    eval_root = evaluation_root.resolve()
    data_root = (Path(project_root).resolve() if project_root is not None
                 else root.parents[1].resolve())
    if not _inside(root, data_root) or not _inside(eval_root, data_root):
        raise ValueError("MECO status roots must remain inside the selected data root")
    queue = _queue_state(root, now, data_root)
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
                                  "queued" if any(lane.get("status") == "queued"
                                                  for lane in queue.get("lanes", {}).values())
                                  else "idle")

    comparisons = []
    evaluation_runs = []
    if eval_root.is_dir():
        for status_path in sorted(eval_root.glob("*.status.json")):
            status, error = _read_json_within(status_path, data_root)
            if status is None:
                errors.append({"source": status_path.name, "error": error or "invalid evaluator status"})
                continue
            stamp = _utc_timestamp(status.get("heartbeat_utc"))
            age = None if stamp is None else max(0., now - stamp)
            fresh = age is not None and age <= EVALUATION_HEARTBEAT_STALE_SECONDS and stamp <= now + 5.
            output_name = status.get("output_file")
            safe_name = isinstance(output_name, str) and Path(output_name).name == output_name
            report_path = eval_root / output_name if safe_name else eval_root / "__invalid__"
            report, _ = (_read_json_within(report_path, data_root)
                         if safe_name else (None, None))
            valid_report = _comparison(report) if report is not None else None
            consistent = (status.get("schema_version") == 1
                and isinstance(status.get("evaluation_id"), str)
                and isinstance(status.get("protocol_id"), str)
                and status.get("split") == "validation"
                and ((status.get("status") == "running" and report is None)
                    or (valid_report is not None
                        and report.get("evaluation_id") == status.get("evaluation_id")
                        and report.get("protocol_id") == status.get("protocol_id")
                        and report.get("split") == status.get("split"))))
            alive = bool(process_probe(status.get("pid")))
            state = status.get("status", "unknown")
            if state == "running":
                state = "running" if fresh and alive and consistent else "stale/unconfirmed"
            elif state == "complete" and not consistent:
                state = "invalid"
            elif state not in ("complete", "failed", "stopped"):
                state = "unknown"
            evaluation_runs.append({"evaluation_id": status.get("evaluation_id"),
                "status": state, "heartbeat_state": "fresh" if fresh else "stale/missing",
                "heartbeat_age_seconds": age, "process_alive": alive,
                "protocol_id": status.get("protocol_id"), "split": status.get("split"),
                "progress": status.get("progress"), "output_file": output_name,
                "error": None if consistent else "evaluator status/report provenance mismatch"})
        for path in sorted(eval_root.glob("*.json")):
            if path.name.endswith(".status.json"):
                continue
            if not path.is_file() or not _inside(path, data_root):
                continue
            report, error = _read_json_within(path, data_root)
            if report is None:
                errors.append({"source": path.name, "error": error or "invalid evaluation"})
                continue
            comparison = _comparison(report)
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
               "control_definitions": {"D0": rows[-1]["control_definitions"]["D0"]},
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
        runs.sort(key=lambda row: (
            row.get("training_state") in ("running", "queued", "unconfirmed", "stalled"),
            row.get("heartbeat", {}).get("fresh", False),
            row.get("manifest_state") == "queued",
            row.get("completed_updates", 0), row["run_id"] or ""))
        latest_run = runs[-1] if runs else None
        lane = queue.get("lanes", {}).get(policy) if policy in TRACKS else None
        if latest_run is None and lane:
            lane_state = lane.get("status")
            latest_run = {"run_id": lane.get("run_id"), "track": policy,
                "manifest_state": lane_state,
                "training_state": "queued" if queue["state"] == "fresh" and lane_state == "queued" else "unconfirmed",
                "heartbeat": {"state": "not_started", "fresh": False,
                              "pid": lane.get("pid")}, "config": {},
                "worker_pid": lane.get("pid"), "device": lane.get("device"),
                "sources": {}, "latest_checkpoint": None,
                "completed_updates": lane.get("completed_updates") or 0,
                "latest_training_metrics": None}
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
    checkpoint_root = (data_project / "checkpoints" / "meco-five").resolve()
    evaluation_root = (data_project / "evaluations" / "meco-five").resolve()
    if not _inside(checkpoint_root, data_project) or not _inside(evaluation_root, data_project):
        raise ValueError("MECO status roots must remain inside the selected data root")
    payload = _collect_status(checkpoint_root, evaluation_root, project_root=data_project)
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
    payload["three_pass_provenance"] = _three_pass_provenance(project)
    payload["campaign_ledger_path"] = str(pass_path)
    if valid_passes is None:
        payload["errors"].append({"source": str(pass_path.relative_to(project)),
            "error": pass_error or "invalid implementation pass status manifest"})
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()


_EXPECTED_THREE_PASS_PROVENANCE_SHA256 = "fe9a0a624e1ebd194eef81a7151f536ec6855ff99d22450ac47e19874b1507bf"
_EXPECTED_PLAN_SHA256 = "743dba515c7a732bb65161302a62756fbb7306255c90b8d4da1f29f66fc9ba47"


def _three_pass_provenance(project: Path) -> dict:
    """Validate candidate-local scoped summaries; never infer transcripts or outcomes."""
    import hashlib
    import re

    manifest_path = project / "docs" / "MECO_THREE_PASS_PROVENANCE.json"
    plan_path = project / "docs" / "MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md"
    commit_re = re.compile(r"^[0-9a-f]{40}$")
    digest_re = re.compile(r"^[0-9a-f]{64}$")

    def artifact(relpath, expected_hash):
        if (not isinstance(relpath, str) or not relpath or Path(relpath).is_absolute()
                or "\\" in relpath or ".." in Path(relpath).parts
                or not isinstance(expected_hash, str) or not digest_re.fullmatch(expected_hash)):
            raise ValueError("summary artifact path or digest is invalid")
        candidate = project / relpath
        # Reject symlinks in every component, then require a regular in-repo file.
        current = project
        for part in Path(relpath).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("summary artifact path is symlinked")
        resolved_root, resolved = project.resolve(), candidate.resolve()
        if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
            raise ValueError("summary artifact is missing or outside the checkout")
        content = resolved.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("summary artifact hash mismatch")
        return content.decode("utf-8")

    try:
        if (manifest_path.is_symlink() or plan_path.is_symlink()
                or not manifest_path.is_file() or not plan_path.is_file()):
            raise ValueError("manifest or plan is missing or symlinked")
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != _EXPECTED_THREE_PASS_PROVENANCE_SHA256:
            raise ValueError("manifest hash does not match the dashboard-pinned digest")
        data = json.loads(manifest_bytes)
        if (not isinstance(data, dict)
                or data.get("schema") != "meco-three-pass-provenance-v2"
                or data.get("status") != "validated-summary-only"
                or data.get("plan_path") != "docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md"
                or data.get("plan_sha256") != _EXPECTED_PLAN_SHA256
                or hashlib.sha256(plan_path.read_bytes()).hexdigest() != _EXPECTED_PLAN_SHA256
                or not isinstance(data.get("scope"), str)
                or not isinstance(data.get("limits"), list)
                or not all(isinstance(x, str) for x in data["limits"])):
            raise ValueError("manifest schema or plan binding is invalid")
        tranches = data.get("tranches")
        if not isinstance(tranches, list) or not tranches:
            raise ValueError("manifest has no scoped tranche records")
        validated = []
        seen = set()
        required_roles = {"planner", "implementer", "reviewer"}
        for tranche in tranches:
            if not isinstance(tranche, dict):
                raise ValueError("tranche record is invalid")
            tranche_id = tranche.get("tranche_id")
            if not isinstance(tranche_id, str) or not tranche_id or tranche_id in seen:
                raise ValueError("tranche identity is invalid or duplicated")
            seen.add(tranche_id)
            commits = tranche.get("commits")
            if (not isinstance(commits, dict)
                    or set(commits) != {"feature_commit", "reviewer_approved_commit"}
                    or not all(isinstance(v, str) and commit_re.fullmatch(v) for v in commits.values())
                    or commits["feature_commit"] != commits["reviewer_approved_commit"]):
                raise ValueError("feature and reviewer-approved commits must match exact commit ids")
            summaries = tranche.get("summaries")
            if not isinstance(summaries, dict) or set(summaries) != required_roles:
                raise ValueError("tranche must contain planner, implementer, and reviewer summaries")
            summary_text = {}
            for role, record in summaries.items():
                if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                    raise ValueError("summary record fields are invalid")
                summary_text[role] = artifact(record["path"], record["sha256"])
            # A syntactically valid hash is not provenance unless it resolves to
            # an actual commit object in this checkout. Values are strictly
            # validated above before they reach git; shell parsing is never used.
            for commit in commits.values():
                try:
                    checked = subprocess.run(
                        [TRUSTED_GIT_EXECUTABLE, "cat-file", "-e", f"{commit}^{{commit}}"],
                        cwd=project, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=2, check=False,
                        # Never let an inherited PATH select the git executable
                        # (or helpers) used to validate dashboard provenance.
                        env={"PATH": "/usr/bin:/bin",
                             "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"},
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise ValueError("provenance commit object could not be checked") from exc
                if checked.returncode != 0:
                    raise ValueError("provenance commit is not present as a commit object in this checkout")
            validated.append({"tranche_id": tranche_id, "title": tranche.get("title"),
                "commits": commits, "summaries": summary_text})
        return {"status": "validated_summary_only", "evidence_class": "scoped_summaries",
                "tranches": validated, "message": data["scope"], "limits": data["limits"]}
    except (OSError, ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"status": "untrusted", "evidence_class": "unavailable", "tranches": [],
                "untrusted_reason": str(exc)}
