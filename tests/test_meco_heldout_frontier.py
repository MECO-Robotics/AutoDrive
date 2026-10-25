import hashlib
import json

import torch

import pytest

from frc_defense.meco_heldout_frontier import (
    compile_heldout_frontier,
    validate_heldout_frontier,
)
from frc_defense.meco_qualification import protocol_document
from frc_defense.meco_tracks import MECO_TRACKS


def _inputs(tmp_path):
    protocol = protocol_document()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(__import__("json").dumps(protocol, sort_keys=True, indent=2) + "\n")
    checkpoints = {}
    configs = {}
    for track in MECO_TRACKS:
        path = tmp_path / f"{track}.pt"
        config = {"track": track, "updates": 123, "learning_rate": 0.001,
                  "device": "cuda:0", "output": f"/runs/{track}"}
        torch.save({"schema_version": 1, "config": config,
                    "protocol_id": "fixture-protocol"}, path)
        checkpoints[track] = path
        immutable = {key: value for key, value in config.items() if key != "updates"}
        configs[track] = hashlib.sha256(json.dumps(
            immutable, sort_keys=True, allow_nan=False).encode()).hexdigest()
    source = {"checkout": "fixture", "files": ["policy.py", "planner.py"]}
    return protocol_path, checkpoints, source, configs


def test_frontier_binds_protocol_source_all_tracks_and_full_roster(tmp_path):
    protocol_path, checkpoints, source, configs = _inputs(tmp_path)
    frontier = compile_heldout_frontier(protocol_path=protocol_path,
        checkpoint_paths=checkpoints, source_identity=source,
        training_configuration_sha256=configs)
    assert frontier["version_status"] == "provisional"
    assert frontier["split"] == "validation"
    assert frontier["qualification_claim"] is False
    assert frontier["qualification_eligible"] is False
    assert frontier["final_test_seed_handling"] == "not_exported"
    assert set(frontier["checkpoints_sha256"]) == set(MECO_TRACKS)
    assert set(frontier["training_configuration_sha256"]) == set(MECO_TRACKS)
    assert any(row["id"] == "candidate-recall:C2:K16" for row in frontier["contrast_roster"])
    assert all(len(item["sha256"]) == 64 for item in frontier["seed_manifests"])
    validate_heldout_frontier(frontier, protocol_path=protocol_path,
        checkpoint_paths=checkpoints, source_identity=source,
        training_configuration_sha256=configs)


@pytest.mark.parametrize("field", ["qualification_eligible", "split", "final_test_seed_handling"])
def test_frontier_rejects_claim_or_nonvalidation_mutation(tmp_path, field):
    protocol_path, checkpoints, source, configs = _inputs(tmp_path)
    frontier = compile_heldout_frontier(protocol_path=protocol_path,
        checkpoint_paths=checkpoints, source_identity=source,
        training_configuration_sha256=configs)
    bad = dict(frontier)
    bad[field] = {"qualification_eligible": True, "split": "diagnostic",
                  "final_test_seed_handling": "included"}[field]
    with pytest.raises(ValueError):
        validate_heldout_frontier(bad, protocol_path=protocol_path,
            checkpoint_paths=checkpoints, source_identity=source,
            training_configuration_sha256=configs)


@pytest.mark.parametrize("binding", ["source", "checkpoint", "config", "protocol"])
def test_frontier_rejects_changed_bound_inputs(tmp_path, binding):
    protocol_path, checkpoints, source, configs = _inputs(tmp_path)
    frontier = compile_heldout_frontier(protocol_path=protocol_path,
        checkpoint_paths=checkpoints, source_identity=source,
        training_configuration_sha256=configs)
    if binding == "source":
        source = {**source, "files": ["changed.py"]}
    elif binding == "checkpoint":
        payload = torch.load(checkpoints["C2"], map_location="cpu", weights_only=False)
        payload["extra_payload_marker"] = "changed"
        torch.save(payload, checkpoints["C2"])
    elif binding == "config":
        configs["A2"] = hashlib.sha256(b"changed config").hexdigest()
    else:
        protocol_path.write_text(protocol_path.read_text() + " ")
    expected_error = ("supplied training configuration hashes differ"
                      if binding == "config" else "differs from frozen frontier")
    with pytest.raises(ValueError, match=expected_error):
        validate_heldout_frontier(frontier, protocol_path=protocol_path,
            checkpoint_paths=checkpoints, source_identity=source,
            training_configuration_sha256=configs)


def test_frontier_rejects_missing_track_hash(tmp_path):
    protocol_path, checkpoints, source, configs = _inputs(tmp_path)
    del configs["A3"]
    with pytest.raises(ValueError, match="all five tracks"):
        compile_heldout_frontier(protocol_path=protocol_path,
            checkpoint_paths=checkpoints, source_identity=source,
            training_configuration_sha256=configs)


def test_frontier_rejects_false_checkpoint_config_association(tmp_path):
    protocol_path, checkpoints, source, configs = _inputs(tmp_path)
    configs["C1"] = hashlib.sha256(b"unrelated immutable config").hexdigest()
    with pytest.raises(ValueError, match="differ from checkpoint embedded immutable configs"):
        compile_heldout_frontier(protocol_path=protocol_path,
            checkpoint_paths=checkpoints, source_identity=source,
            training_configuration_sha256=configs)


def test_checkpoint_update_count_is_excluded_from_immutable_config_hash(tmp_path):
    protocol_path, checkpoints, source, configs = _inputs(tmp_path)
    payload = torch.load(checkpoints["A1"], map_location="cpu", weights_only=False)
    payload["config"]["updates"] = 999
    torch.save(payload, checkpoints["A1"])
    frontier = compile_heldout_frontier(protocol_path=protocol_path,
        checkpoint_paths=checkpoints, source_identity=source,
        training_configuration_sha256=configs)
    assert frontier["training_configuration_sha256"] == configs


def test_frontier_rejects_checkpoints_from_different_training_protocols(tmp_path):
    protocol_path, checkpoints, source, configs = _inputs(tmp_path)
    payload = torch.load(checkpoints["C2"], map_location="cpu", weights_only=False)
    payload["protocol_id"] = "other-training-protocol"
    torch.save(payload, checkpoints["C2"])
    with pytest.raises(ValueError, match="do not share one training protocol ID"):
        compile_heldout_frontier(protocol_path=protocol_path,
            checkpoint_paths=checkpoints, source_identity=source,
            training_configuration_sha256=configs)


def test_frontier_rejects_unsupported_checkpoint_format(tmp_path):
    protocol_path, checkpoints, source, configs = _inputs(tmp_path)
    torch.save({"config": {"track": "A1"}}, checkpoints["A1"])
    with pytest.raises(ValueError, match="supported MECO schema-1 payload"):
        compile_heldout_frontier(protocol_path=protocol_path,
            checkpoint_paths=checkpoints, source_identity=source,
            training_configuration_sha256=configs)
