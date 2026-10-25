"""Durable latest-only journal for manifest-bound diagnostic frontiers."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .meco_tracks import MECO_TRACKS

_ZERO = "0" * 64


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MecoDiagnosticMonitor:
    """Append-only event chain with a recoverable cursor and one pending frontier."""

    def __init__(self, root, *, manifest_sha256, configuration_sha256,
                 protocol_sha256, monitor, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.root / "diagnostic-monitor.jsonl"
        self.cursor_path = self.root / "diagnostic-monitor-cursor.json"
        self.binding = {"campaign_manifest_sha256": manifest_sha256,
            "configuration_sha256": configuration_sha256,
            "protocol_sha256": protocol_sha256,
            "monitor_sha256": _sha(monitor)}
        self.monitor = monitor
        if self.journal_path.exists() or self.cursor_path.exists():
            if not resume or not self.journal_path.is_file() or not self.cursor_path.is_file():
                raise ValueError("diagnostic monitor state exists or is incomplete; explicit resume required")
            self.events = self._read_events()
            self._validate_cursor()
            self._validate_persisted_results()
            self._recover_interrupted_attempt()
        else:
            if resume:
                raise ValueError("cannot resume diagnostic monitor without durable journal and cursor")
            self.events = []
            with self.journal_path.open("x") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            self._write_cursor()

    def _read_events(self):
        events, previous = [], _ZERO
        try:
            lines = self.journal_path.read_text().splitlines()
        except OSError as exc:
            raise ValueError("diagnostic monitor journal is unreadable") from exc
        for seq, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("diagnostic monitor journal has a malformed record") from exc
            body = dict(event)
            digest = body.pop("event_sha256", None)
            if (body.get("sequence") != seq or body.get("previous_sha256") != previous or
                    body.get("binding") != self.binding or digest != _sha(body)):
                raise ValueError("diagnostic monitor journal hash chain or binding is invalid")
            payload = body.get("payload", {})
            if body.get("kind") == "frontier_queued":
                self._validate_frontier_snapshots(payload)
            elif body.get("kind") == "attempt_verified":
                result = payload.get("result", {})
                if (not isinstance(result, dict) or
                        _file_sha(result.get("report_path", "")) != result.get("report_sha256") or
                        _file_sha(result.get("status_path", "")) != result.get("status_sha256")):
                    raise ValueError("diagnostic monitor report/status bytes changed")
            events.append(event)
            previous = digest
        return events

    def _validate_cursor(self):
        try:
            cursor = json.loads(self.cursor_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("diagnostic monitor cursor is malformed") from exc
        body = dict(cursor)
        digest = body.pop("cursor_sha256", None)
        if digest != _sha(body) or body.get("binding") != self.binding:
            raise ValueError("diagnostic monitor cursor hash or binding is invalid")
        seq = body.get("sequence")
        head = self.events[-1]["event_sha256"] if self.events else _ZERO
        expected_seq = len(self.events) - 1
        if seq > expected_seq or (seq == expected_seq and body.get("journal_sha256") != head):
            raise ValueError("diagnostic monitor cursor is ahead of or detached from its journal")
        # A journal append may survive a crash before cursor replacement; replay is authoritative.
        if seq < expected_seq:
            self._write_cursor()

    def _validate_result_files(self, frontier, result):
        try:
            report_path, status_path = Path(result["report_path"]), Path(result["status_path"])
            report = json.loads(report_path.read_text())
            status = json.loads(status_path.read_text())
        except (KeyError, OSError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("diagnostic monitor report/status is missing or malformed") from exc
        hashes = {row["track"]: row["sha256"] for row in frontier["checkpoint_snapshots"]}
        binding = {"schema_version": 1, "frontier_id": frontier["frontier_id"],
            "cycle": frontier["cycle"], "update": frontier["update"],
            "checkpoint_sha256": hashes, "source_sha256": frontier["source_sha256"],
            "configuration_sha256": frontier["configuration_sha256"],
            "protocol_id": self.monitor["protocol_id"],
            "protocol_sha256": frontier["protocol_sha256"]}
        case = {"index": frontier["cycle"] - 1,
            "scenario_id": self.monitor["scenario_id"],
            "category": self.monitor["category"], "seeds": self.monitor["seeds"],
            "split": "diagnostic", "protocol_id": self.monitor["protocol_id"],
            "protocol_sha256": self.monitor["protocol_sha256"],
            "qualification_case": None}
        try:
            report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            status_digest = hashlib.sha256(status_path.read_bytes()).hexdigest()
            expected_report_digest = result["report_sha256"]
            expected_status_digest = result["status_sha256"]
        except (KeyError, OSError, TypeError) as exc:
            raise ValueError("diagnostic monitor report/status digest is malformed") from exc
        if (report_digest != expected_report_digest or
                status_digest != expected_status_digest):
            raise ValueError("diagnostic monitor report/status digest changed")
        if (report.get("diagnostic_frontier_binding") != binding or
                status.get("diagnostic_frontier_binding") != binding or
                report.get("schedule_case_binding") != case or
                status.get("schedule_case_binding") != case or
                report.get("checkpoint_sha256") != hashes or
                report.get("exact_episode_seeds") != self.monitor["seeds"] or
                status.get("exact_episode_seeds") != self.monitor["seeds"] or
                report.get("evaluation_protocol_id") != self.monitor["protocol_id"] or
                status.get("evaluation_protocol_id") != self.monitor["protocol_id"] or
                report.get("evaluation_protocol_sha256") != frontier["protocol_sha256"] or
                status.get("evaluation_protocol_sha256") != frontier["protocol_sha256"] or
                report.get("qualification_eligible") is not False or
                status.get("qualification_eligible") is not False or
                status.get("diagnostic_only") is not True or
                not isinstance(report.get("sources"), dict) or
                hashlib.sha256(json.dumps(report["sources"], sort_keys=True,
                    allow_nan=False).encode()).hexdigest() != frontier["source_sha256"]):
            raise ValueError("diagnostic monitor report/status does not bind its frozen frontier")

    def _validate_persisted_results(self):
        active, _, completed = self._frontiers()
        rows = list(completed)
        if active and active["status"] == "verified":
            rows.append(active)
        for frontier in rows:
            self._validate_result_files(frontier, frontier["result"])

    def _write_cursor(self):
        body = {"schema_version": 1, "binding": self.binding,
            "sequence": len(self.events) - 1,
            "journal_sha256": self.events[-1]["event_sha256"] if self.events else _ZERO}
        payload = {**body, "cursor_sha256": _sha(body)}
        temporary = self.cursor_path.with_name(self.cursor_path.name + ".tmp")
        with temporary.open("w") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.cursor_path)
        fd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _append(self, kind, payload):
        previous = self.events[-1]["event_sha256"] if self.events else _ZERO
        body = {"sequence": len(self.events), "previous_sha256": previous,
            "binding": self.binding, "kind": kind, "payload": payload}
        event = {**body, "event_sha256": _sha(body)}
        with self.journal_path.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)
        self._write_cursor()
        return event

    def _frontiers(self):
        active = None
        queued = None
        completed = []
        for event in self.events:
            payload = event["payload"]
            if event["kind"] == "frontier_queued":
                if queued is not None:
                    raise ValueError("diagnostic monitor replaced a queued frontier without a skip record")
                queued = {**payload, "status": "queued", "attempts": []}
            elif event["kind"] == "frontier_skipped":
                target = (active if active and active["frontier_id"] == payload["frontier_id"]
                    else queued)
                if target is None or target["frontier_id"] != payload["frontier_id"]:
                    raise ValueError("diagnostic monitor skipped frontier chain is inconsistent")
                if target is active:
                    active = None
                else:
                    queued = None
            elif event["kind"] == "attempt_started":
                if queued is None or queued["frontier_id"] != payload.get("frontier_id") or active:
                    raise ValueError("diagnostic monitor attempt does not bind the queued frontier")
                active, queued = queued, None
                active["attempts"].append({**payload, "status": "running"})
                active["status"] = "running"
            elif event["kind"] == "attempt_retry_queued":
                self._require_pending(active, payload)
                if active["status"] != "failed" or queued is not None:
                    raise ValueError("diagnostic monitor retry transition is invalid")
                queued, active = active, None
                queued["status"] = "queued"
            elif event["kind"] == "attempt_failed":
                self._require_pending(active, payload)
                if not active["attempts"] or active["attempts"][-1]["status"] != "running":
                    raise ValueError("diagnostic monitor failure has no running attempt")
                active["attempts"][-1].update(status="failed", error=payload["error"])
                active["status"] = "failed"
            elif event["kind"] == "attempt_verified":
                self._require_pending(active, payload)
                if not active["attempts"] or active["attempts"][-1]["status"] != "running":
                    raise ValueError("diagnostic monitor result has no running attempt")
                active["attempts"][-1].update(status="verified", result=payload["result"])
                active["status"] = "verified"
                active["result"] = payload["result"]
            elif event["kind"] == "frontier_completed":
                self._require_pending(active, payload)
                if active["status"] != "verified":
                    raise ValueError("diagnostic monitor completed an unverified result")
                active["status"] = "complete"
                active["result"] = payload["result"]
                completed.append(active)
                active = None
            else:
                raise ValueError("diagnostic monitor journal contains an unknown event")
        return active, queued, completed

    @staticmethod
    def _require_pending(pending, payload):
        if pending is None or pending["frontier_id"] != payload.get("frontier_id"):
            raise ValueError("diagnostic monitor event does not bind the pending frontier")

    def _recover_interrupted_attempt(self):
        active, _, _ = self._frontiers()
        if active and active["status"] == "running":
            self.fail_attempt("monitor restarted before verified completion")
        elif active and active["status"] == "verified":
            # The report/status were revalidated before adoption; finish the durable cursor.
            self.complete()

    @property
    def pending(self):
        active, queued, _ = self._frontiers()
        return active or queued

    def enqueue(self, frontier):
        required = {"frontier_id", "cycle", "update", "checkpoint_snapshots",
            "source_sha256", "configuration_sha256", "protocol_sha256"}
        if not isinstance(frontier, dict) or set(frontier) != required:
            raise ValueError("diagnostic monitor frontier fields are malformed")
        if (not isinstance(frontier["frontier_id"], str) or not frontier["frontier_id"] or
                type(frontier["cycle"]) is not int or frontier["cycle"] < 1 or
                type(frontier["update"]) is not int or frontier["update"] < 0 or
                frontier["configuration_sha256"] != self.binding["configuration_sha256"] or
                frontier["protocol_sha256"] != self.binding["protocol_sha256"] or
                not isinstance(frontier["source_sha256"], str) or
                len(frontier["source_sha256"]) != 64):
            raise ValueError("diagnostic monitor frontier identity is invalid")
        self._validate_frontier_snapshots(frontier)
        active, queued, _ = self._frontiers()
        existing = queued or active
        if existing and existing["frontier_id"] == frontier["frontier_id"]:
            if {key: existing[key] for key in required} != frontier:
                raise ValueError("repeated diagnostic frontier has different snapshots or binding")
            return existing
        if queued:
            self._append("frontier_skipped", {"frontier_id": queued["frontier_id"],
                "superseded_by": frontier["frontier_id"], "cycle": queued["cycle"],
                "update": queued["update"]})
        self._append("frontier_queued", frontier)
        return self.pending

    @staticmethod
    def _validate_frontier_snapshots(frontier):
        rows = frontier.get("checkpoint_snapshots", [])
        if (not isinstance(rows, list) or len(rows) != len(MECO_TRACKS) or
                [row.get("track") for row in rows if isinstance(row, dict)] != sorted(MECO_TRACKS)):
            raise ValueError("diagnostic monitor frontier requires the exact five snapshots")
        for row in rows:
            path = Path(row.get("path", ""))
            if (set(row) != {"track", "path", "sha256"} or
                    path.name != f"{row['track']}.pt" or not path.is_file() or
                    _file_sha(path) != row["sha256"]):
                raise ValueError("diagnostic monitor checkpoint snapshot changed or is malformed")

    def begin_attempt(self, output_path):
        active, queued, _ = self._frontiers()
        if active and active["status"] == "failed" and queued:
            self._append("frontier_skipped", {"frontier_id": active["frontier_id"],
                "superseded_by": queued["frontier_id"], "cycle": active["cycle"],
                "update": active["update"]})
            active = None
        elif active and active["status"] == "failed":
            self._append("attempt_retry_queued", {"frontier_id": active["frontier_id"]})
            active = None
            _, queued, _ = self._frontiers()
        pending = active or queued
        if pending is None or pending["status"] not in ("queued", "failed"):
            return None
        payload = {"frontier_id": pending["frontier_id"],
            "attempt": len(pending["attempts"]) + 1,
            "output_path": str(Path(output_path)), "started_at_unix": time.time()}
        self._append("attempt_started", payload)
        return self.pending

    def fail_attempt(self, error):
        pending, _, _ = self._frontiers()
        if pending is None or pending["status"] != "running":
            raise ValueError("diagnostic monitor has no running attempt")
        self._append("attempt_failed", {"frontier_id": pending["frontier_id"],
            "error": str(error), "failed_at_unix": time.time()})
        active, queued, _ = self._frontiers()
        if queued is not None and active is not None:
            self._append("frontier_skipped", {"frontier_id": active["frontier_id"],
                "superseded_by": queued["frontier_id"], "cycle": active["cycle"],
                "update": active["update"]})
        return self.pending

    def verify_attempt(self, result):
        pending, _, _ = self._frontiers()
        if pending is None or pending["status"] != "running":
            raise ValueError("diagnostic monitor has no running attempt")
        hashes = {row["track"]: row["sha256"] for row in pending["checkpoint_snapshots"]}
        expected_binding = {"schema_version": 1,
            "frontier_id": pending["frontier_id"], "cycle": pending["cycle"],
            "update": pending["update"], "checkpoint_sha256": hashes,
            "source_sha256": pending["source_sha256"],
            "configuration_sha256": pending["configuration_sha256"],
            "protocol_id": self.monitor["protocol_id"],
            "protocol_sha256": pending["protocol_sha256"]}
        if (not isinstance(result, dict) or result.get("checkpoint_sha256") != hashes or
                result.get("seeds") != self.monitor["seeds"] or
                result.get("frontier_id") != pending["frontier_id"] or
                result.get("source_sha256") != pending["source_sha256"] or
                result.get("configuration_sha256") != pending["configuration_sha256"] or
                result.get("protocol_sha256") != pending["protocol_sha256"] or
                result.get("frontier_binding_sha256") != _sha(expected_binding) or
                result.get("qualification_eligible") is not False or
                result.get("diagnostic_only") is not True or
                any(not isinstance(result.get(key), str) or len(result[key]) != 64
                    for key in ("report_sha256", "status_sha256"))):
            raise ValueError("diagnostic monitor result is missing exact integrity bindings")
        self._validate_result_files(pending, result)
        self._append("attempt_verified", {"frontier_id": pending["frontier_id"],
            "result": result})
        return self.pending

    def complete(self):
        pending, _, _ = self._frontiers()
        if pending is None or pending["status"] != "verified":
            raise ValueError("diagnostic monitor frontier must be verified before completion")
        self._append("frontier_completed", {"frontier_id": pending["frontier_id"],
            "result": pending["result"]})
        return self.summary()

    def summary(self):
        active, queued, completed = self._frontiers()
        skipped = [event["payload"] for event in self.events
            if event["kind"] == "frontier_skipped"]
        frontier_cycles = [event["payload"]["cycle"] for event in self.events
            if event["kind"] == "frontier_queued"]
        return {"enabled": True, "diagnostic_only": True,
            "qualification_eligible": False,
            "journal_sha256": self.events[-1]["event_sha256"] if self.events else _ZERO,
            "journal_events": len(self.events), "completed_count": len(completed),
            "skipped_frontiers": skipped, "frontier_cycles": frontier_cycles,
            "active_frontier": active, "queued_frontier": queued,
            "pending_frontier": active or queued}
