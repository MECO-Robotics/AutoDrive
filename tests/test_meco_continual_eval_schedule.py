import hashlib
import json

import pytest

from frc_defense import meco_training_manifest as manifest_api
from frc_defense.meco_supervisor import (
    _append_evaluation_history, _evaluation_command, _scheduled_evaluation_case,
    _validate_evaluation_history)
from frc_defense.meco_tracks import MECO_TRACKS
from scripts import meco_protocol_transition


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


def test_schedule_advances_by_cycle_and_repeated_selection_is_deterministic():
    from frc_defense.meco_track_training import _EVALUATION_CASE_FIELDS
    assert manifest_api.EVALUATION_CASE_FIELDS == _EVALUATION_CASE_FIELDS
    protocol, sha = _protocol()
    schedule = manifest_api._validate_evaluation_schedule(
        _schedule(protocol, sha), protocol, sha)
    assert [row["index"] for row in schedule] == [0, 1]
    assert [_scheduled_evaluation_case(schedule, cycle)["index"]
        for cycle in (1, 2, 3, 4)] == [0, 1, 0, 1]
    # Repeating selection for the same cycle within one supervisor lifetime is stable.
    cycle_two = _scheduled_evaluation_case(schedule, 2)
    assert _scheduled_evaluation_case(schedule, 2) == cycle_two
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
    assert row["campaign_binding"] == {"campaign_manifest_sha256": None,
        "configuration_sha256": None, "phase4_train_exposure": False}
    assert all(item["campaign_binding"] == row["campaign_binding"]
        for item in row["checkpoint_snapshots"])
    assert _validate_evaluation_history(tmp_path / "evaluation-history.json",
        expected_campaign_binding=row["campaign_binding"]) == history


def test_history_binds_manifest_config_and_exposure_to_all_snapshots(tmp_path):
    checkpoints = {}
    for track in MECO_TRACKS:
        path = tmp_path / f"{track}.pt"
        path.write_bytes(track.encode())
        checkpoints[track] = path
    output = tmp_path / "eval.json"
    output.write_text("{}")
    binding = {"campaign_manifest_sha256": "a" * 64,
        "configuration_sha256": "b" * 64, "phase4_train_exposure": True}
    _append_evaluation_history(tmp_path, cycle=1, output=output,
        completion={"status": "complete"}, checkpoints=checkpoints,
        campaign_manifest_sha256=binding["campaign_manifest_sha256"],
        configuration_sha256=binding["configuration_sha256"],
        phase4_train_exposure=True)
    history_path = tmp_path / "evaluation-history.json"
    history = json.loads(history_path.read_text())
    row = history["entries"][0]
    assert row["campaign_binding"] == binding
    assert all(snapshot["campaign_binding"] == binding
        for snapshot in row["checkpoint_snapshots"])
    _validate_evaluation_history(history_path, expected_campaign_binding=binding)

    row["campaign_binding"]["phase4_train_exposure"] = False
    for snapshot in row["checkpoint_snapshots"]:
        snapshot["campaign_binding"]["phase4_train_exposure"] = False
    body = {key: value for key, value in row.items() if key != "entry_sha256"}
    row["entry_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    history_path.write_text(json.dumps(history))
    with pytest.raises(ValueError, match="campaign/configuration binding changed"):
        _validate_evaluation_history(history_path, expected_campaign_binding=binding)


def test_history_hash_detects_binding_tamper(tmp_path):
    checkpoints = {}
    for track in MECO_TRACKS:
        path = tmp_path / f"{track}.pt"
        path.write_bytes(track.encode())
        checkpoints[track] = path
    output = tmp_path / "eval.json"
    output.write_text("{}")
    _append_evaluation_history(tmp_path, cycle=1, output=output,
        completion={"status": "complete"}, checkpoints=checkpoints)
    history_path = tmp_path / "evaluation-history.json"
    history = json.loads(history_path.read_text())
    history["entries"][0]["campaign_binding"]["phase4_train_exposure"] = True
    history_path.write_text(json.dumps(history))
    with pytest.raises(ValueError, match="hash chain is broken"):
        _validate_evaluation_history(history_path)


def test_transition_history_audit_binds_five_snapshots_to_queue_configuration(tmp_path):
    checkpoints = {}
    for track in MECO_TRACKS:
        path = tmp_path / "checkpoint-snapshots" / "cycle-000001-1234abcd" / f"{track}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(track.encode())
        checkpoints[track] = path
    output = tmp_path / "cycle-000001-1234abcd.json"
    output.write_text(json.dumps({"evaluation_id": "eval-1", "protocol_id": "protocol-1"}))
    status_path = output.with_suffix(output.suffix + ".status.json")
    status_path.write_text(json.dumps({"status": "complete", "evaluation_id": "eval-1",
        "protocol_id": "protocol-1", "output_file": output.name}))
    result = _append_evaluation_history(tmp_path, cycle=1, output=output,
        completion={"status": "complete", "evaluation_id": "eval-1",
            "protocol_id": "protocol-1", "return_code": 0},
        checkpoints=checkpoints, campaign_manifest_sha256="a" * 64,
        configuration_sha256="b" * 64, phase4_train_exposure=True)
    queue_path = tmp_path.parent / f"{tmp_path.name}-queue-status.json"
    queue = {"configuration_sha256": "b" * 64,
        "campaign_manifest": {"canonical_sha256": "a" * 64},
        "task_scheduler": {"phase4_train_exposure": True},
        "evaluation_history": result}
    queue_path.write_text(json.dumps(queue))
    _, indexed, _ = meco_protocol_transition._evaluation_inventory(tmp_path, queue_path)
    assert len(indexed[0]["snapshot_inventory"]) == len(MECO_TRACKS)

    history_path = tmp_path / "evaluation-history.json"
    history = json.loads(history_path.read_text())
    original_history = json.loads(json.dumps(history))
    row = history["entries"][0]
    row["checkpoint_snapshots"][1]["track"] = row["checkpoint_snapshots"][0]["track"]
    body = {key: value for key, value in row.items() if key != "entry_sha256"}
    row["entry_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    history_path.write_text(json.dumps(history))
    queue["evaluation_history"]["head_sha256"] = row["entry_sha256"]
    queue_path.write_text(json.dumps(queue))
    with pytest.raises(ValueError, match="duplicate or unknown track labels"):
        meco_protocol_transition._evaluation_inventory(tmp_path, queue_path)

    history = original_history
    row = history["entries"][0]
    row["campaign_binding"]["phase4_train_exposure"] = False
    for snapshot in row["checkpoint_snapshots"]:
        snapshot["campaign_binding"]["phase4_train_exposure"] = False
    body = {key: value for key, value in row.items() if key != "entry_sha256"}
    row["entry_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    history_path.write_text(json.dumps(history))
    queue["evaluation_history"]["head_sha256"] = row["entry_sha256"]
    queue_path.write_text(json.dumps(queue))
    with pytest.raises(ValueError, match="campaign/configuration binding mismatch"):
        meco_protocol_transition._evaluation_inventory(tmp_path, queue_path)


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
