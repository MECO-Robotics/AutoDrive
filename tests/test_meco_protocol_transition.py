import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from frc_defense.meco_tracks import MECO_TRACKS
from scripts import meco_protocol_transition as transition


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _protocol(path, *, train_seed=7):
    body = {"protocol_version": "meco-qualification-v1", "cases": [
        {"scenario_id": "pickup_simple", "category": "single", "split": "train", "seeds": [train_seed]},
        {"scenario_id": "pickup_simple", "category": "single", "split": "validation", "seeds": [8]},
        {"scenario_id": "pickup_simple", "category": "single", "split": "test", "seeds": [9]}]}
    body["protocol_id"] = hashlib.sha256(json.dumps({k: v for k, v in body.items()},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(body))
    return body


def _segment(root, protocol, *, pending=None, seed=7, missing_track=None, update=2):
    root.mkdir(parents=True, exist_ok=True)
    sources = {"meco_track_training.py": "a" * 64}
    source_sha = transition._source_digest(sources)
    assignment = {"cycle": 1, "scenario_id": "pickup_simple", "category": "single",
        "seed": seed, "assignment_id": "assignment-1", "target_update": update}
    results = {track: {"assignment_id": "assignment-1", "cycle": 1,
        "scenario_id": "pickup_simple", "category": "single", "episodes": 1,
        "successes": 1, "failures": 0, "aborted_no_legal_support": 0,
        "completed_updates": update} for track in MECO_TRACKS}
    scheduler = {"schema_version": 1, "completed_cycle": 1,
        "pending_assignment": pending,
        "history": [{"assignment": assignment, "lane_results": results}]}
    (root / "task-scheduler-state.json").write_text(json.dumps(scheduler))
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
        progress = {"assignment_id": "assignment-1", "cycle": 1,
            "scenario_id": "pickup_simple", "category": "single", "episodes": 1,
            "successes": 1, "failures": 0, "aborted_no_legal_support": 0}
        checkpoint = {"config": config, "run_id": f"run-{track}", "update": update,
            "episode": 4, "protocol_id": "historical-protocol", "sources": sources,
            "task_assignment": assignment,
            "task_progress": progress,
            "history": [{"update": i, "task_assignment_id": "assignment-1"}
                for i in range(1, update + 1)], "model": actor}
        torch.save(checkpoint, lane / "latest.pt")
        manifest = {"track": track, "status": "complete", "completed_updates": update,
            "run_id": f"run-{track}", "task_assignment": assignment,
            "task_outcomes": results[track], "protocol_id": "historical-protocol", "sources": sources,
            "source_identity_sha256": source_sha, "config": config,
            "configuration_sha256": cfg_sha}
        (lane / "run.json").write_text(json.dumps(manifest))
    return assignment


class _Tiny(nn.Module):
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
            audit = transition.audit_segment(root / "segment", protocol_path)
            self.assertEqual(audit["audit_status"], "consistent_historical_segment")
            self.assertFalse(audit["qualification_eligible"])
            self.assertEqual(audit["historical_protocol_binding"], "unbound")
            self.assertEqual(set(audit["checkpoints"]), set(MECO_TRACKS))

    def test_audit_binds_hash_of_exact_evaluation_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol)
            preliminary = transition.audit_segment(root / "segment", protocol_path)
            evaluation_path = root / "evaluation.json"
            report = {"evaluation_id": "eval-1", "protocol_id": "eval-protocol",
                "source_identity_sha256": preliminary["source_identity_sha256"],
                "policy_manifest": {track: {"checkpoint_sha256": preliminary["checkpoints"][track]["sha256"]}
                    for track in MECO_TRACKS}}
            evaluation_path.write_text(json.dumps(report))
            audited = transition.audit_segment(root / "segment", protocol_path, [evaluation_path])
            self.assertEqual(audited["evaluations"][0]["sha256"],
                hashlib.sha256(evaluation_path.read_bytes()).hexdigest())

    def test_validation_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            _protocol(protocol_path)
            _segment(root / "segment", _protocol(protocol_path), seed=8)
            with self.assertRaisesRegex(ValueError, "exact train split"):
                transition.audit_segment(root / "segment", protocol_path)

    def test_missing_lane_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol, missing_track="C2")
            with self.assertRaisesRegex(ValueError, "missing C2"):
                transition.audit_segment(root / "segment", protocol_path)

    def test_pending_frontier_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol, pending={"assignment_id": "next"})
            with self.assertRaisesRegex(ValueError, "pending assignment"):
                transition.audit_segment(root / "segment", protocol_path)

    def test_checkpoint_tamper_after_audit_blocks_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol)
            audit = transition.audit_segment(root / "segment", protocol_path)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            checkpoint = root / "segment" / "A1" / "latest.pt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "changed since audit"):
                transition.export_transition(audit_path, root / "transition.pt")

    def test_export_and_consumer_load_actor_only_and_reset_run_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol = _protocol(protocol_path)
            _segment(root / "segment", protocol)
            audit = transition.audit_segment(root / "segment", protocol_path)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            models = {track: _Tiny() for track in MECO_TRACKS}
            with mock.patch.object(transition, "make_meco_track", return_value=_Tiny()):
                transition.export_transition(audit_path, root / "transition.pt")
                consumed = transition.consume_transition(root / "transition.pt", seed=123, models=models)
            self.assertEqual(consumed["update"], 0)
            self.assertEqual(consumed["episode"], 0)
            self.assertEqual(consumed["history"], [])
            self.assertEqual(consumed["rng_seed"], 123)
            self.assertTrue(all(not optimizer.state for optimizer in consumed["optimizers"].values()))
            self.assertEqual(consumed["qualification_eligible"], False)
            self.assertEqual(consumed["lineage"]["historical_source_identity_sha256"], audit["source_identity_sha256"])
            self.assertEqual({tuple(model.state_dict()["weight"].tolist()) for model in models.values()},
                {(float(ord(track[0])),) for track in MECO_TRACKS})


if __name__ == "__main__":
    unittest.main()
