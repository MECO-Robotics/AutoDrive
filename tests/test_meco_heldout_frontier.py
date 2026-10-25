import hashlib

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
        path.write_bytes(("checkpoint:" + track).encode())
        checkpoints[track] = path
        configs[track] = hashlib.sha256(("config:" + track).encode()).hexdigest()
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
        checkpoints["C2"].write_bytes(b"changed checkpoint")
    elif binding == "config":
        configs["A2"] = hashlib.sha256(b"changed config").hexdigest()
    else:
        protocol_path.write_text(protocol_path.read_text() + " ")
    with pytest.raises(ValueError, match="differs from frozen frontier"):
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

