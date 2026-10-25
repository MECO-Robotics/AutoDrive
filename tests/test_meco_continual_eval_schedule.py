import hashlib
import json

import pytest

from frc_defense import meco_training_manifest as manifest_api
from frc_defense.meco_supervisor import (
    _append_evaluation_history, _evaluation_command, _scheduled_evaluation_case)
from frc_defense.meco_tracks import MECO_TRACKS


def _protocol():
    body = {"protocol_version": "meco-qualification-v1", "master_seed": 17,
        "cases": [
            {"scenario_id": "pickup_simple", "category": "single",
                "split": "train", "seeds": [1, 2]},
            {"scenario_id": "pickup_simple", "category": "single",
                "split": "validation", "seeds": [3], "layout_id": "layout-a",
                "defender_family": "none", "start_condition": "standard",
                "sensor_profile": "default"},
            {"scenario_id": "pickup_simple", "category": "single",
                "split": "test", "seeds": [4]},
        ]}
    body["protocol_id"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    sha = hashlib.sha256(json.dumps(body).encode()).hexdigest()
    return body, sha


def _schedule(protocol, sha):
    return [
        {"scenario_id": "pickup_simple", "category": "single", "seeds": [100],
            "split": "diagnostic", "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha},
        {"scenario_id": "pickup_simple", "category": "single", "seeds": [3],
            "split": "validation", "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha, "qualification_case": {
                "layout_id": "layout-a", "defender_family": "none",
                "start_condition": "standard", "sensor_profile": "default"}},
    ]


def test_schedule_progression_is_one_case_per_cycle_and_resume_deterministic():
    from frc_defense.meco_track_training import _EVALUATION_CASE_FIELDS
    assert manifest_api.EVALUATION_CASE_FIELDS == _EVALUATION_CASE_FIELDS
    protocol, sha = _protocol()
    schedule = manifest_api._validate_evaluation_schedule(
        _schedule(protocol, sha), protocol, sha)
    assert [row["index"] for row in schedule] == [0, 1]
    assert [_scheduled_evaluation_case(schedule, cycle)["index"]
        for cycle in (1, 2, 3, 4)] == [0, 1, 0, 1]
    # A resumed supervisor derives the same case from the frozen schedule and cycle.
    assert _scheduled_evaluation_case(schedule, 3) == schedule[0]
    assert _scheduled_evaluation_case(schedule, 4) == schedule[1]


def test_schedule_rejects_protocol_drift_and_split_leakage():
    protocol, sha = _protocol()
    schedule = _schedule(protocol, sha)
    schedule[0]["seeds"] = [1]
    with pytest.raises(ValueError, match="leaks"):
        manifest_api._validate_evaluation_schedule(schedule, protocol, sha)
    schedule = _schedule(protocol, sha)
    schedule[1]["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol binding"):
        manifest_api._validate_evaluation_schedule(schedule, protocol, sha)
    schedule = _schedule(protocol, sha)
    schedule[1]["qualification_case"]["layout_id"] = "wrong-layout"
    with pytest.raises(ValueError, match="exactly match protocol"):
        manifest_api._validate_evaluation_schedule(schedule, protocol, sha)


def test_evaluation_command_preserves_exact_diagnostic_and_validation_seeds(tmp_path):
    checkpoints = {track: tmp_path / f"{track}.pt" for track in MECO_TRACKS}
    base = dict(checkpoints=checkpoints, scenario="pickup_simple", categories=("single",),
        episodes=2, seed=100, device="cuda:1", output=tmp_path / "eval.json",
        episode_seeds=(100, 900))
    diagnostic = _evaluation_command(**base, split="diagnostic")
    assert diagnostic[diagnostic.index("--episode-seeds") + 1] == "100,900"
    assert "--evaluation-protocol" not in diagnostic
    validation = _evaluation_command(**base, split="validation",
        evaluation_protocol=tmp_path / "protocol.json",
        evaluation_case={"layout_id": "l", "defender_family": "none",
            "start_condition": "standard", "sensor_profile": "default"})
    assert validation[validation.index("--episode-seeds") + 1] == "100,900"
    assert validation[validation.index("--evaluation-protocol") + 1] == str(
        tmp_path / "protocol.json")


def test_history_pins_all_five_checkpoint_hashes_and_schedule_case(tmp_path):
    protocol, sha = _protocol()
    case = manifest_api._validate_evaluation_schedule(
        _schedule(protocol, sha), protocol, sha)[1]
    checkpoints = {}
    expected_hashes = {}
    for track in MECO_TRACKS:
        path = tmp_path / f"{track}.pt"
        path.write_bytes(track.encode())
        checkpoints[track] = path
        expected_hashes[track] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "eval.json"
    output.write_text("{}")
    completion = {"status": "complete", "schedule_case": case,
        "evaluation_schedule_sha256": "a" * 64}
    result = _append_evaluation_history(tmp_path, cycle=2, output=output,
        completion=completion, checkpoints=checkpoints)
    history = json.loads((tmp_path / "evaluation-history.json").read_text())
    row = history["entries"][0]
    assert result["count"] == 1
    assert row["schedule_case"] == case
    assert row["evaluation_schedule_sha256"] == "a" * 64
    assert {item["track"]: item["sha256"] for item in row["checkpoint_snapshots"]} == expected_hashes


def test_history_rejects_missing_track_snapshot(tmp_path):
    output = tmp_path / "eval.json"
    output.write_text("{}")
    checkpoints = {}
    for track in MECO_TRACKS[:-1]:
        path = tmp_path / f"{track}.pt"
        path.write_bytes(track.encode())
        checkpoints[track] = path
    with pytest.raises(ValueError, match="all five tracks"):
        _append_evaluation_history(tmp_path, cycle=1, output=output,
            completion={"status": "complete"}, checkpoints=checkpoints)
