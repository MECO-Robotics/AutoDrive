import hashlib
import fcntl
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from frc_defense.meco_qualification import MIN_EPISODES, TRAIN_SEED_POOL_SIZE, protocol_cases
from frc_defense.meco_tracks import MECO_TRACKS
from frc_defense.meco_supervisor import _append_evaluation_history
from scripts import meco_protocol_transition as transition


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


MASTER_SEED = 20261024
TRAIN_CASE = protocol_cases("pickup_simple", "train", episodes=TRAIN_SEED_POOL_SIZE,
    master_seed=MASTER_SEED)[0]
VALIDATION_CASE = protocol_cases("pickup_simple", "validation", episodes=MIN_EPISODES,
    master_seed=MASTER_SEED)[0]
TEST_CASE = protocol_cases("pickup_simple", "test", episodes=MIN_EPISODES,
    master_seed=MASTER_SEED)[0]
TRAIN_SEED = TRAIN_CASE.seeds[0]
VALIDATION_SEED = VALIDATION_CASE.seeds[0]


def _protocol(path):
    cases = []
    for split, case in (("train", TRAIN_CASE), ("validation", VALIDATION_CASE),
                        ("test", TEST_CASE)):
        cases.append({"scenario_id": "pickup_simple", "category": case.category,
            "split": split, "layout_id": case.layout_id,
            "defender_family": case.defender_family,
            "start_condition": case.start_condition,
            "sensor_profile": case.sensor_profile, "seeds": list(case.seeds)})
    body = {"protocol_version": "meco-qualification-v1", "master_seed": MASTER_SEED,
        "cases": cases}
    body["protocol_id"] = hashlib.sha256(json.dumps({k: v for k, v in body.items()},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(body))
    return body


def _segment(root, protocol, *, pending=None, seed=TRAIN_SEED, missing_track=None, update=2):
    root.mkdir(parents=True, exist_ok=True)
    sources = {"meco_track_training.py": "a" * 64}
    source_sha = transition._source_digest(sources)
    pool_id = transition._sha(json.dumps({"scenario_id": "pickup_simple",
        "seeds": list(TRAIN_CASE.seeds)}, sort_keys=True, allow_nan=False).encode())
    assignment_payload = {"cycle": 1, "scenario_id": "pickup_simple", "category": "single",
        "seed": seed, "master_seed": MASTER_SEED, "training_seed_pool_id": pool_id}
    assignment_id = transition._sha(json.dumps(assignment_payload,
        sort_keys=True, allow_nan=False).encode())
    assignment = {**assignment_payload, "training_seed_pool_id": pool_id,
        "assignment_id": assignment_id, "target_update": update}
    results = {track: {"assignment_id": assignment_id, "cycle": 1,
        "scenario_id": "pickup_simple", "category": "single", "episodes": 1,
        "successes": 1, "failures": 0, "aborted_no_legal_support": 0,
        "completed_updates": update} for track in MECO_TRACKS}
    scheduler = {"schema_version": 1, "master_seed": MASTER_SEED,
        "completed_cycle": 1,
        "pending_assignment": pending,
        "history": [{"assignment": assignment, "lane_results": results}]}
    (root / "task-scheduler-state.json").write_text(json.dumps(scheduler))
    (root / ".supervisor.lock").touch()
    (root / "supervisor-status.json").write_text(json.dumps({"status": "complete",
        "lanes": [{"track": track, "pid": None} for track in MECO_TRACKS]}))
    (root / "queue-status.json").write_text(json.dumps({"phase": "complete",
        "evaluation_history": {"count": 0, "head_sha256": None},
        "evaluation": {"active": False, "pending": 0, "status": "idle",
            "output_file": None, "evaluation_id": None}}))
    for track in MECO_TRACKS:
        if track == missing_track:
            continue
        lane = root / track
        lane.mkdir()
        config = {"track": track, "updates": update, "device": "cuda:1",
            "output": str(lane), "learning_rate": 0.0003}
        cfg_sha = transition._sha(json.dumps({k: v for k, v in config.items() if k != "updates"},
            sort_keys=True, allow_nan=False).encode())
        actor = {"weight": torch.tensor([float(ord(track[0]))]),
                 "bias": torch.tensor([0.25])}
        progress = {"assignment_id": assignment_id, "cycle": 1,
            "scenario_id": "pickup_simple", "category": "single", "episodes": 1,
            "successes": 1, "failures": 0, "aborted_no_legal_support": 0}
        checkpoint = {"config": config, "run_id": f"run-{track}", "update": update,
            "episode": 4, "protocol_id": "historical-protocol", "sources": sources,
            "task_assignment": assignment,
            "task_progress": progress,
            "history": [{"update": i, "task_assignment_id": assignment_id}
                for i in range(1, update + 1)], "model": actor}
        torch.save(checkpoint, lane / "latest.pt")
        manifest = {"track": track, "status": "complete", "completed_updates": update,
            "run_id": f"run-{track}", "task_assignment": assignment,
            "task_outcomes": results[track], "protocol_id": "historical-protocol", "sources": sources,
            "source_identity_sha256": source_sha, "config": config,
            "configuration_sha256": cfg_sha}
        (lane / "run.json").write_text(json.dumps(manifest))
    if missing_track is None:
        case = next(row for row in protocol["cases"]
            if row["scenario_id"] == "pickup_simple" and row["category"] == "single"
            and row["split"] == "validation")
        eval_seed = case["seeds"][0]
        eval_stem = "cycle-000001-1234abcd"
        eval_root = root.parent / "evaluations"
        snapshot_dir = eval_root / "checkpoint-snapshots" / eval_stem
        snapshot_dir.mkdir(parents=True)
        source_sha = transition._source_digest(sources)
        policy_manifest, records = {}, []
        for track in MECO_TRACKS:
            source_path = root / track / "latest.pt"
            snapshot_path = snapshot_dir / f"{track}.pt"
            snapshot_path.write_bytes(source_path.read_bytes())
            checkpoint = torch.load(snapshot_path, map_location="cpu", weights_only=False)
            lane_manifest = json.loads((root / track / "run.json").read_text())
            policy_manifest[track] = {"checkpoint_sha256": hashlib.sha256(
                snapshot_path.read_bytes()).hexdigest(),
                "training_protocol_id": checkpoint["protocol_id"],
                "training_configuration_sha256": lane_manifest["configuration_sha256"]}
            records.append({"scenario_id": "pickup_simple", "category": "single",
                "seed": eval_seed, "policy": track,
                "checkpoint_sha256": policy_manifest[track]["checkpoint_sha256"],
                "split": "validation", "qualification_protocol_id": protocol["protocol_id"],
                "layout_id": case["layout_id"], "defender_family": case["defender_family"],
                "start_condition": case["start_condition"],
                "sensor_profile": case["sensor_profile"],
                "source_identity_sha256": source_sha})
        report = {"evaluation_id": "eval-1", "protocol_id": "evaluation-protocol",
            "qualification_protocol_id": protocol["protocol_id"],
            "scenario_id": "pickup_simple", "split": "validation",
            "categories": ["single"], "seeds": [eval_seed],
            "qualification_case": {key: case[key] for key in (
                "layout_id", "defender_family", "start_condition", "sensor_profile")},
            "source_identity_sha256": source_sha,
            "policy_manifest": policy_manifest, "records": records}
        report_path = eval_root / f"{eval_stem}.json"
        _write_evaluation(report_path, report)
        _mark_supervisor_evaluation(root, report_path)
        _append_evaluation_history(eval_root, cycle=1, output=report_path,
            completion={"status": "complete", "evaluation_id": report["evaluation_id"],
                "protocol_id": report["protocol_id"], "return_code": 0},
            checkpoints={track: snapshot_dir / f"{track}.pt" for track in MECO_TRACKS})
        _mark_history_anchor(root, eval_root)
    return assignment


def _audit(segment, protocol_path):
    evaluation_root = segment.parent / "evaluations"
    evaluation_root.mkdir(exist_ok=True)
    return transition.audit_segment(segment, protocol_path, evaluation_root)


def _write_evaluation(path, report):
    path.write_text(json.dumps(report))
    path.with_suffix(path.suffix + ".status.json").write_text(json.dumps({
        "status": "complete", "evaluation_id": report["evaluation_id"],
        "output_file": path.name}))


def _mark_supervisor_evaluation(segment, path):
    status_path = segment / "supervisor-status.json"
    status = json.loads(status_path.read_text())
    report = json.loads(path.read_text())
    status["evaluation"] = {"status": "complete", "output_file": path.name,
        "evaluation_id": report["evaluation_id"]}
    status_path.write_text(json.dumps(status))
    queue_path = segment / "queue-status.json"
    queue = json.loads(queue_path.read_text())
    queue["evaluation"] = {"active": False, "pending": 0, "status": "complete",
        "output_file": path.name, "evaluation_id": report["evaluation_id"]}
    queue_path.write_text(json.dumps(queue))


def _mark_history_anchor(segment, evaluation_root):
    history = json.loads((evaluation_root / "evaluation-history.json").read_text())["entries"]
    queue_path = segment / "queue-status.json"
    queue = json.loads(queue_path.read_text())
    queue["evaluation_history"] = {"count": len(history),
        "head_sha256": history[-1]["entry_sha256"] if history else None}
    queue_path.write_text(json.dumps(queue))


def _refresh_history_report(segment, report_path):
    history_path = report_path.parent / "evaluation-history.json"
    history = json.loads(history_path.read_text())
    row = next(item for item in history["entries"] if item["output_file"] == report_path.name)
    row["evaluation_id"] = json.loads(report_path.read_text())["evaluation_id"]
    row["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    status_path = report_path.with_suffix(report_path.suffix + ".status.json")
    row["status_sha256"] = hashlib.sha256(status_path.read_bytes()).hexdigest()
    body = {key: value for key, value in row.items() if key != "entry_sha256"}
    row["entry_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    history_path.write_text(json.dumps(history))
    _mark_history_anchor(segment, report_path.parent)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))


class _WrongTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.bias = nn.Parameter(torch.zeros(1))


class _SameShapeDifferentClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))


class MecoProtocolTransitionTests(unittest.TestCase):
    def test_valid_audit_is_explicitly_nonqualifying_and_hashes_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            _protocol(protocol_path)
            _segment(root / "segment", _protocol(protocol_path))
            audit = _audit(root / "segment", protocol_path)
            self.assertEqual(audit["audit_status"], "consistent_historical_segment")
            self.assertFalse(audit["qualification_eligible"])
            self.assertEqual(audit["historical_protocol_binding"], "unbound")
            self.assertEqual(set(audit["checkpoints"]), set(MECO_TRACKS))
            self.assertEqual(len(audit["evaluations"]), 1)
            self.assertIn("queue_status_sha256", audit)

    def test_audit_binds_hash_of_exact_evaluation_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol)
            evaluation_root = root / "evaluations"
            evaluation_path = evaluation_root / "cycle-000001-1234abcd.json"
            audited = transition.audit_segment(root / "segment", protocol_path, evaluation_root)
            self.assertEqual(audited["evaluations"][0]["sha256"],
                hashlib.sha256(evaluation_path.read_bytes()).hexdigest())
            report = json.loads(evaluation_path.read_text())
            report["qualification_protocol_id"] = None
            for record in report["records"]:
                record["qualification_protocol_id"] = None
            _write_evaluation(evaluation_path, report)
            _refresh_history_report(root / "segment", evaluation_path)
            with self.assertRaisesRegex(ValueError, "unbound supervisor evaluation"):
                transition.audit_segment(root / "segment", protocol_path, evaluation_root)

    def test_validation_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            _protocol(protocol_path)
            _segment(root / "segment", _protocol(protocol_path), seed=VALIDATION_SEED)
            with self.assertRaisesRegex(ValueError, "exact train split"):
                _audit(root / "segment", protocol_path)

    def test_missing_lane_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol, missing_track="C2")
            with self.assertRaisesRegex(ValueError, "missing C2"):
                _audit(root / "segment", protocol_path)

    def test_pending_frontier_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol, pending={"assignment_id": "next"})
            with self.assertRaisesRegex(ValueError, "pending assignment"):
                _audit(root / "segment", protocol_path)

    def test_nonterminal_supervisor_and_held_lock_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            segment = root / "segment"
            _segment(segment, protocol)
            status_path = segment / "supervisor-status.json"
            status_path.write_text(json.dumps({"status": "running", "lanes": [
                {"track": track, "pid": None} for track in MECO_TRACKS]}))
            with self.assertRaisesRegex(ValueError, "not terminal"):
                _audit(segment, protocol_path)
            status_path.write_text(json.dumps({"status": "complete", "lanes": [
                {"track": track, "pid": None} for track in MECO_TRACKS]}))
            fd = os.open(segment / ".supervisor.lock", os.O_RDONLY)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.assertRaisesRegex(ValueError, "lock is still held"):
                    _audit(segment, protocol_path)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def test_scheduler_lane_outcome_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            segment = root / "segment"
            _segment(segment, protocol)
            path = segment / "task-scheduler-state.json"
            scheduler = json.loads(path.read_text())
            scheduler["history"][0]["lane_results"]["A1"]["scenario_id"] = "other"
            path.write_text(json.dumps(scheduler))
            with self.assertRaisesRegex(ValueError, "outcome disagrees"):
                _audit(segment, protocol_path)
            scheduler["history"][0]["lane_results"]["A1"]["scenario_id"] = "pickup_simple"
            scheduler["history"][0]["lane_results"]["A1"]["successes"] = 2
            path.write_text(json.dumps(scheduler))
            with self.assertRaisesRegex(ValueError, "counters are invalid"):
                _audit(segment, protocol_path)

    def test_evaluation_unknown_split_case_or_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            segment = root / "segment"
            _segment(segment, protocol)
            audit = _audit(segment, protocol_path)
            evaluation_root = root / "evaluations"
            report_path = evaluation_root / "cycle-000001-1234abcd.json"
            report = json.loads(report_path.read_text())
            report["records"][0]["seed"] = 9
            _write_evaluation(report_path, report)
            _mark_supervisor_evaluation(segment, report_path)
            _refresh_history_report(segment, report_path)
            with self.assertRaisesRegex(ValueError, "category/seed|unknown stratum"):
                transition.audit_segment(segment, protocol_path, evaluation_root)

    def test_checkpoint_tamper_after_audit_blocks_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol)
            audit = _audit(root / "segment", protocol_path)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            checkpoint = root / "segment" / "A1" / "latest.pt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "changed since audit"):
                transition.export_transition(audit_path, root / "transition.pt")

    def test_snapshot_tamper_after_audit_blocks_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol)
            audit = _audit(root / "segment", protocol_path)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            snapshot = root / "evaluations" / "checkpoint-snapshots" / "cycle-000001-1234abcd" / "A1.pt"
            snapshot.write_bytes(snapshot.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "history|snapshot"):
                transition.export_transition(audit_path, root / "transition.pt")

    def test_durable_history_detects_deleted_older_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            segment = root / "segment"
            _segment(segment, protocol)
            evaluation_root = root / "evaluations"
            first_report = evaluation_root / "cycle-000001-1234abcd.json"
            second_report = evaluation_root / "cycle-000001-5678abcd.json"
            second_snapshots = evaluation_root / "checkpoint-snapshots" / second_report.stem
            shutil.copytree(evaluation_root / "checkpoint-snapshots" / first_report.stem,
                second_snapshots)
            report = json.loads(first_report.read_text())
            report["evaluation_id"] = "eval-2"
            _write_evaluation(second_report, report)
            _mark_supervisor_evaluation(segment, second_report)
            _append_evaluation_history(evaluation_root, cycle=1, output=second_report,
                completion={"status": "complete", "evaluation_id": report["evaluation_id"],
                    "protocol_id": report["protocol_id"], "return_code": 0},
                checkpoints={track: second_snapshots / f"{track}.pt" for track in MECO_TRACKS})
            _mark_history_anchor(segment, evaluation_root)
            audit = transition.audit_segment(segment, protocol_path, evaluation_root)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            first_report.unlink()
            first_report.with_suffix(first_report.suffix + ".status.json").unlink()
            shutil.rmtree(evaluation_root / "checkpoint-snapshots" / first_report.stem)
            with self.assertRaisesRegex(ValueError, "complete report inventory"):
                transition.export_transition(audit_path, root / "transition.pt")
            with self.assertRaisesRegex(ValueError, "complete report inventory"):
                transition.audit_segment(segment, protocol_path, evaluation_root)

    def test_legacy_evaluation_root_without_history_is_not_claimed_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            segment = root / "segment"
            _segment(segment, protocol)
            (root / "evaluations" / "evaluation-history.json").unlink()
            with self.assertRaisesRegex(ValueError, "legacy evaluation root"):
                _audit(segment, protocol_path)

    def test_export_and_consumer_load_actor_only_and_reset_run_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol)
            audit = _audit(root / "segment", protocol_path)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            models = {track: _Tiny() for track in MECO_TRACKS}
            with mock.patch.object(transition, "make_meco_track", return_value=_Tiny()), \
                 mock.patch.object(transition, "source_identity", return_value=audit["source_identity"]):
                transition.export_transition(audit_path, root / "transition.pt")
                consumed = transition.consume_transition(root / "transition.pt", seed=123,
                    target_qualification_protocol=protocol_path, models=models)
            self.assertEqual(consumed["update"], 0)
            self.assertEqual(consumed["episode"], 0)
            self.assertEqual(consumed["history"], [])
            self.assertEqual(consumed["rng_seed"], 123)
            self.assertTrue(all(not optimizer.state for optimizer in consumed["optimizers"].values()))
            self.assertEqual(consumed["qualification_eligible"], False)
            self.assertEqual(consumed["lineage"]["historical_source_identity_sha256"], audit["source_identity_sha256"])
            self.assertEqual({tuple(model.state_dict()["weight"].tolist()) for model in models.values()},
                {(float(ord(track[0])),) for track in MECO_TRACKS})

    def test_consumer_requires_matching_protocol_source_and_model_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol)
            audit = _audit(root / "segment", protocol_path)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            artifact_path = root / "transition.pt"
            models = {track: _Tiny() for track in MECO_TRACKS}
            with mock.patch.object(transition, "make_meco_track", return_value=_Tiny()), \
                 mock.patch.object(transition, "source_identity", return_value=audit["source_identity"]):
                transition.export_transition(audit_path, artifact_path)
                with mock.patch.object(transition, "source_identity", return_value={"wrong": "source"}):
                    with self.assertRaisesRegex(ValueError, "current source identity"):
                        transition.consume_transition(artifact_path, seed=3,
                            target_qualification_protocol=protocol_path, models=models)
                wrong_protocol = root / "wrong-protocol.json"
                changed = {key: value for key, value in protocol.items() if key != "protocol_id"}
                changed["extra"] = "different target"
                changed["protocol_id"] = hashlib.sha256(json.dumps(changed,
                    sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                wrong_protocol.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "target qualification protocol"):
                    transition.consume_transition(artifact_path, seed=3,
                        target_qualification_protocol=wrong_protocol, models=models)
                with mock.patch.object(transition, "source_identity", return_value=audit["source_identity"]):
                    bad_models = {track: _WrongTiny() for track in MECO_TRACKS}
                    with self.assertRaisesRegex(ValueError, "class differs from registered constructor"):
                        transition.consume_transition(artifact_path, seed=3,
                            target_qualification_protocol=protocol_path, models=bad_models)
                    wrong_class = {track: _SameShapeDifferentClass() for track in MECO_TRACKS}
                    with self.assertRaisesRegex(ValueError, "model class differs"):
                        transition.consume_transition(artifact_path, seed=3,
                            target_qualification_protocol=protocol_path, models=wrong_class)


if __name__ == "__main__":
    unittest.main()
