"""Durable ordered queue for manifest-bound MECO evaluation cases."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .meco_tracks import MECO_TRACKS


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        allow_nan=False).encode()).hexdigest()


def verify_evaluation_artifacts(schedule_case, output_path, return_code,
                                checkpoint_paths):
    """Verify a zero-exit evaluator report/status against its frozen job."""
    if return_code != 0:
        raise ValueError(f"evaluator exited with status {return_code}")
    output_path = Path(output_path)
    status_path = Path(str(output_path) + ".status.json")
    try:
        report = json.loads(output_path.read_text())
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("evaluator report or status is missing/malformed") from exc
    if not isinstance(report, dict) or not isinstance(status, dict):
        raise ValueError("evaluator report or status is malformed")
    evaluation_id = report.get("evaluation_id")
    seeds = schedule_case["seeds"]
    if (status.get("schema_version") != 1 or status.get("status") != "complete" or
            not isinstance(evaluation_id, str) or not evaluation_id or
            status.get("evaluation_id") != evaluation_id or
            status.get("protocol_id") != report.get("protocol_id") or
            status.get("output_file") != output_path.name or
            status.get("split") != schedule_case["split"] or
            status.get("schedule_case_binding") != schedule_case or
            report.get("schedule_case_binding") != schedule_case):
        raise ValueError("evaluator status/report does not bind the scheduled case")
    if (report.get("scenario_id") != schedule_case["scenario_id"] or
            report.get("categories") != [schedule_case["category"]] or
            report.get("seeds") != seeds or report.get("split") != schedule_case["split"] or
            status.get("exact_episode_seeds") != seeds):
        raise ValueError("evaluator report/status case or exact seed list mismatch")
    for key in ("protocol_id", "protocol_sha256"):
        report_key = "evaluation_" + key
        status_key = report_key
        if report.get(report_key) != schedule_case[key] or status.get(status_key) != schedule_case[key]:
            raise ValueError("evaluator report/status protocol binding mismatch")
    if schedule_case["split"] == "validation":
        if report.get("qualification_case") != schedule_case.get("qualification_case"):
            raise ValueError("evaluator report qualification case mismatch")
    if set(checkpoint_paths) != set(MECO_TRACKS):
        raise ValueError("scheduled evaluation requires exact five-track checkpoint snapshots")
    checkpoint_hashes = {track: hashlib.sha256(Path(checkpoint_paths[track]).read_bytes()).hexdigest()
        for track in MECO_TRACKS}
    if report.get("checkpoint_sha256") != checkpoint_hashes:
        raise ValueError("evaluator report checkpoint hashes differ from scheduled snapshots")
    policy_manifest = report.get("policy_manifest")
    if not isinstance(policy_manifest, dict) or any(
            not isinstance(policy_manifest.get(track), dict) or
            policy_manifest[track].get("checkpoint_sha256") != checkpoint_hashes[track]
            for track in MECO_TRACKS):
        raise ValueError("evaluator policy manifest does not bind all five scheduled checkpoints")
    return {"evaluation_id": evaluation_id,
        "report_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "status_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
        "status_path": str(status_path), "output_path": str(output_path),
        "checkpoint_sha256": checkpoint_hashes}


def validate_history_indexes(rows, *, schedule_count=None, expected_schedule_sha256=None):
    """Require scheduled history indices to be a unique contiguous prefix."""
    scheduled = [row for row in rows if row.get("schedule_index") is not None]
    if not scheduled:
        return []
    indices = [row.get("schedule_index") for row in scheduled]
    if (any(type(index) is not int for index in indices) or
            indices != list(range(len(indices)))):
        raise ValueError("evaluation history schedule indices are duplicated or contain skips")
    for row in scheduled:
        case = row.get("schedule_case")
        if (not isinstance(case, dict) or case.get("index") != row["schedule_index"] or
                (expected_schedule_sha256 is not None and
                    row.get("evaluation_schedule_sha256") != expected_schedule_sha256)):
            raise ValueError("evaluation history schedule case/digest binding mismatch")
    if schedule_count is not None and len(indices) > schedule_count:
        raise ValueError("evaluation history exceeds the frozen schedule length")
    return indices


def _write(path, state):
    body = dict(state)
    body.pop("state_sha256", None)
    payload = {**body, "state_sha256": _digest(body)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    state.clear()
    state.update(payload)


class MecoEvaluationSchedule:
    """Ordered, manifest-bound evaluation cursor with retry-preserving cases."""

    def __init__(self, path, *, manifest_sha256, configuration_sha256,
                 schedule, schedule_sha256, protocol_sha256, resume=False):
        self.path = Path(path)
        self.schedule = schedule
        if not isinstance(schedule, list) or _digest(schedule) != schedule_sha256:
            raise ValueError("evaluation schedule digest does not match its frozen cases")
        self.binding = {"campaign_manifest_sha256": manifest_sha256,
            "configuration_sha256": configuration_sha256,
            "evaluation_schedule_sha256": schedule_sha256,
            "evaluation_protocol_sha256": protocol_sha256}
        if self.path.exists():
            if not resume:
                raise ValueError("evaluation schedule state already exists; use explicit campaign resume")
            try:
                self.state = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("evaluation schedule state is malformed") from exc
            self._validate()
        else:
            if resume:
                raise ValueError("cannot resume scheduled campaign without durable evaluation state")
            self.state = {"schema_version": 1, **self.binding,
                "schedule_count": len(schedule), "next_enqueue_index": 0,
                "next_index": 0, "items": []}
            self._save()

    def _validate(self):
        if not isinstance(self.state, dict):
            raise ValueError("evaluation schedule state is malformed")
        state = dict(self.state)
        digest = state.pop("state_sha256", None)
        if digest != _digest(state):
            raise ValueError("evaluation schedule state digest mismatch")
        if (state.get("schema_version") != 1 or
                any(state.get(key) != value for key, value in self.binding.items()) or
                state.get("schedule_count") != len(self.schedule) or
                not isinstance(state.get("items"), list)):
            raise ValueError("evaluation schedule state binding/configuration mismatch")
        items = state["items"]
        if (state.get("next_enqueue_index") != len(items) or
                not 0 <= state.get("next_index", -1) <= len(items) <= len(self.schedule)):
            raise ValueError("evaluation schedule cursor is malformed")
        for index, item in enumerate(items):
            if (not isinstance(item, dict) or item.get("index") != index or
                    item.get("schedule_case") != self.schedule[index] or
                    item.get("cycle") != index + 1 or
                    item.get("status") not in ("queued", "running", "failed", "verified", "complete") or
                    not isinstance(item.get("attempts"), list)):
                raise ValueError("evaluation schedule queue item is malformed or out of order")
            snapshots = item.get("checkpoint_snapshots")
            if (not isinstance(snapshots, list) or len(snapshots) != len(MECO_TRACKS) or
                    [row.get("track") for row in snapshots if isinstance(row, dict)] !=
                        sorted(MECO_TRACKS) or
                    any(not isinstance(row.get("path"), str) or
                        not isinstance(row.get("sha256"), str) for row in snapshots)):
                raise ValueError("evaluation schedule checkpoint snapshot binding is malformed")
            for row in snapshots:
                path = Path(row["path"])
                if (path.name != f"{row['track']}.pt" or not path.is_file() or
                        hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
                    raise ValueError("evaluation schedule checkpoint snapshot is missing or changed")
        if any(item["status"] != "complete" for item in items[:state["next_index"]]):
            raise ValueError("evaluation schedule cursor skipped an incomplete case")
        if state["next_index"] < len(items) and any(
                item["status"] == "complete" for item in items[state["next_index"]:]):
            raise ValueError("evaluation schedule completed cases out of order")
        running = [item for item in items if item["status"] == "running"]
        if len(running) > 1 or (running and running[0]["index"] != state["next_index"]):
            raise ValueError("evaluation schedule has an invalid active case")

    def _save(self):
        _write(self.path, self.state)

    def enqueue(self, index, cycle, schedule_case, checkpoint_snapshots):
        if type(index) is not int or index < 0 or index >= len(self.schedule):
            raise ValueError("evaluation schedule index is outside the frozen schedule")
        if index < self.state["next_enqueue_index"]:
            item = self.state["items"][index]
            if (item["cycle"] != cycle or item["schedule_case"] != schedule_case or
                    item["checkpoint_snapshots"] != checkpoint_snapshots):
                raise ValueError("duplicate evaluation schedule index has different case/snapshot")
            return item
        if index != self.state["next_enqueue_index"] or cycle != index + 1:
            raise ValueError("evaluation schedule enqueue would skip or reorder a case")
        item = {"index": index, "cycle": cycle, "schedule_case": schedule_case,
            "checkpoint_snapshots": checkpoint_snapshots, "status": "queued",
            "attempts": []}
        self.state["items"].append(item)
        self.state["next_enqueue_index"] += 1
        self._save()
        return item

    def head(self):
        index = self.state["next_index"]
        if index >= self.state["next_enqueue_index"]:
            return None
        return self.state["items"][index]

    def begin_attempt(self, output_file):
        item = self.head()
        if item is None or item["status"] not in ("queued", "failed"):
            return None
        attempt = {"number": len(item["attempts"]) + 1,
            "output_file": str(output_file), "status": "running"}
        item["attempts"].append(attempt)
        item.update(status="running", last_error=None)
        self._save()
        return item

    def fail_attempt(self, error):
        item = self.head()
        if item is None or item["status"] != "running":
            raise ValueError("no running evaluation schedule attempt to fail")
        item["attempts"][-1].update(status="failed", error=str(error))
        item.update(status="failed", last_error=str(error))
        self._save()
        return item

    def verify_attempt(self, artifact):
        item = self.head()
        if item is None or item["status"] != "running":
            raise ValueError("no running evaluation schedule attempt to verify")
        item["attempts"][-1].update(status="verified", artifact=artifact)
        item.update(status="verified", verified_artifact=artifact)
        self._save()
        return item

    def complete(self, history_entry_sha256):
        item = self.head()
        if item is None or item["status"] != "verified":
            raise ValueError("evaluation schedule case must be verified before completion")
        item.update(status="complete", history_entry_sha256=history_entry_sha256)
        self.state["next_index"] += 1
        self._save()
        return item

    def reconcile_history(self, rows):
        indices = validate_history_indexes(rows,
            schedule_count=len(self.schedule),
            expected_schedule_sha256=self.binding["evaluation_schedule_sha256"])
        scheduled_rows = [row for row in rows if row.get("schedule_index") is not None]
        for index in indices:
            if index >= len(self.state["items"]):
                raise ValueError("durable evaluation history is ahead of its schedule queue")
            item = self.state["items"][index]
            row = scheduled_rows[index]
            history_snapshots = row.get("checkpoint_snapshots")
            history_binding = ([{key: snapshot.get(key) for key in ("track", "path", "sha256")}
                for snapshot in history_snapshots] if isinstance(history_snapshots, list) else None)
            if (row.get("schedule_case") != item["schedule_case"] or
                    row.get("status") != "complete" or row.get("return_code") != 0 or
                    history_binding != item["checkpoint_snapshots"]):
                raise ValueError("durable evaluation history differs from verified schedule queue")
            output_path = self.path.parent / row.get("output_file", "")
            checkpoint_paths = {snapshot["track"]: Path(snapshot["path"])
                for snapshot in item["checkpoint_snapshots"]}
            artifact = verify_evaluation_artifacts(row["schedule_case"],
                output_path, row["return_code"], checkpoint_paths)
            if (row.get("report_sha256") != artifact["report_sha256"] or
                    row.get("status_sha256") != artifact["status_sha256"]):
                raise ValueError("durable evaluation history report/status digest changed")
            if item["status"] not in ("verified", "complete"):
                raise ValueError("durable evaluation history advanced an unverified schedule item")
            item.update(status="complete", history_entry_sha256=row.get("entry_sha256"))
        self.state["next_index"] = len(indices)
        item = self.head()
        if item is not None and item["status"] == "running":
            self.recover_interrupted_attempt()
        else:
            self._save()
        return self.head()

    def recover_interrupted_attempt(self):
        item = self.head()
        if item is not None and item["status"] == "running":
            item["attempts"][-1].update(status="failed",
                error="supervisor restarted before verified completion")
            item.update(status="failed",
                last_error="supervisor restarted before verified completion")
            self._save()
        return item

    def summary(self):
        items = self.state["items"]
        return {"schedule_sha256": self.binding["evaluation_schedule_sha256"],
            "schedule_count": self.state["schedule_count"],
            "enqueued_count": len(items), "next_index": self.state["next_index"],
            "completed_count": self.state["next_index"],
            "pending_count": self.state["schedule_count"] - self.state["next_index"],
            "failed_count": sum(attempt["status"] == "failed"
                for item in items for attempt in item["attempts"]),
            "active_index": (self.head()["index"] if self.head() and
                self.head()["status"] in ("running", "verified") else None)}
