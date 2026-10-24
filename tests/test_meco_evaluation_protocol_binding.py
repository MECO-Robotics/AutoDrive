import contextlib
import hashlib
import io
import json
from pathlib import Path
from unittest import mock

import pytest

from frc_defense import meco_supervisor, meco_track_training


STRATUM = {"layout_id": "layout-v", "defender_family": "scripted-v",
    "start_condition": "offset-v", "sensor_profile": "noisy-v"}
CATEGORIES = ("single", "cluster")
SEEDS = [101, 102]


def _write_protocol(path, *, extra_stratum=False, leak=False):
    cases = []
    for category in CATEGORIES:
        cases.extend([
            {"scenario_id": "pickup_simple", "category": category, "split": "train",
                **STRATUM, "seeds": [1, 2]},
            {"scenario_id": "pickup_simple", "category": category, "split": "validation",
                **STRATUM, "seeds": ([1, 102] if leak else SEEDS)},
            {"scenario_id": "pickup_simple", "category": category, "split": "test",
                **STRATUM, "seeds": [201, 202]},
        ])
        if extra_stratum:
            cases.append({"scenario_id": "pickup_simple", "category": category,
                "split": "validation", "layout_id": "layout-other",
                "defender_family": "scripted-v", "start_condition": "offset-v",
                "sensor_profile": "noisy-v", "seeds": [111, 112]})
    body = {"protocol_version": "meco-qualification-v1", "master_seed": 77,
        "cases": cases}
    body["protocol_id"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(body, indent=2))
    return body


def test_valid_binding_records_id_sha_stratum_and_exact_seeds(tmp_path):
    path = tmp_path / "eval-protocol.json"
    protocol = _write_protocol(path)
    case, seeds = meco_track_training.resolve_evaluation_case(path,
        scenario_id="pickup_simple", categories=CATEGORIES, split="validation")
    binding = meco_track_training.resolve_evaluation_protocol(path,
        scenario_id="pickup_simple", categories=CATEGORIES, split="validation",
        episode_seeds=seeds, qualification_case=case)
    assert case == STRATUM
    assert seeds == tuple(SEEDS)
    assert binding["protocol_id"] == protocol["protocol_id"]
    assert binding["protocol_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert binding["split"] == "validation"
    assert binding["seeds"] == SEEDS
    assert binding["protocol_id"] != "independent-training-qualification-id"
    with mock.patch.object(meco_track_training.torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match="requires an explicit available"):
            meco_track_training.evaluate_meco({}, scenario_id="pickup_simple",
                categories=CATEGORIES, episodes=len(seeds), master_seed=seeds[0],
                device="cuda:0", split="validation", evaluation_protocol=path,
                evaluation_case=case, episode_seeds=seeds)


def test_missing_or_tampered_protocol_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="explicit evaluation protocol"):
        meco_track_training.resolve_evaluation_protocol(None,
            scenario_id="pickup_simple", categories=CATEGORIES, split="validation",
            episode_seeds=SEEDS, qualification_case=STRATUM)
    path = tmp_path / "bad.json"
    protocol = _write_protocol(path)
    protocol["cases"][0]["seeds"] = [999]
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="ID does not match"):
        meco_track_training.resolve_evaluation_protocol(path,
            scenario_id="pickup_simple", categories=CATEGORIES, split="validation",
            episode_seeds=SEEDS, qualification_case=STRATUM)


def test_seed_membership_and_split_leakage_are_rejected(tmp_path):
    path = tmp_path / "protocol.json"
    _write_protocol(path)
    with pytest.raises(ValueError, match="exact protocol case seed membership"):
        meco_track_training.resolve_evaluation_protocol(path,
            scenario_id="pickup_simple", categories=CATEGORIES, split="validation",
            episode_seeds=[102], qualification_case=STRATUM)
    _write_protocol(path, leak=True)
    with pytest.raises(ValueError, match="split seed pools overlap"):
        meco_track_training.resolve_evaluation_protocol(path,
            scenario_id="pickup_simple", categories=CATEGORIES, split="validation",
            episode_seeds=[1, 102], qualification_case=STRATUM)


def test_ambiguous_case_requires_explicit_stratum(tmp_path):
    path = tmp_path / "protocol.json"
    _write_protocol(path, extra_stratum=True)
    with pytest.raises(ValueError, match="ambiguous"):
        meco_track_training.resolve_evaluation_case(path,
            scenario_id="pickup_simple", categories=CATEGORIES, split="validation")


def test_supervisor_defaults_to_diagnostic_and_rejects_automatic_test(tmp_path):
    assert meco_supervisor.resolve_supervisor_evaluation(None,
        scenario="pickup_simple", categories=CATEGORIES, split="diagnostic") == (
            None, None, None)
    path = tmp_path / "protocol.json"
    _write_protocol(path)
    with pytest.raises(ValueError, match="final-test"):
        meco_supervisor.resolve_supervisor_evaluation(path,
            scenario="pickup_simple", categories=CATEGORIES, split="test")
    case, seeds, binding = meco_supervisor.resolve_supervisor_evaluation(path,
        scenario="pickup_simple", categories=CATEGORIES, split="validation")
    assert case == STRATUM and seeds == tuple(SEEDS)
    assert binding["protocol_id"] != "training-qualification-id"


def test_heldout_api_label_without_binding_fails_before_device_setup():
    with pytest.raises(ValueError, match="explicit evaluation protocol"):
        meco_track_training.evaluate_meco({}, scenario_id="pickup_simple",
            categories=("single",), episodes=1, master_seed=5, device="cuda:0",
            split="validation", qualification_case=STRATUM)


def test_evaluator_cli_forwards_distinct_binding_and_exact_seeds(tmp_path):
    path = tmp_path / "protocol.json"
    _write_protocol(path)
    fake = {"evaluation_id": "eval", "protocol_id": "report-protocol", "records": []}
    output = io.StringIO()
    args = ["evaluate", "--scenario", "pickup_simple", "--categories", "single,cluster",
        "--episodes", "2", "--seed", "101", "--device", "cuda:0", "--split", "validation",
        "--evaluation-protocol", str(path), "--evaluation-case-json", json.dumps(STRATUM),
        "--episode-seeds", "101,102", "--output", str(tmp_path / "report.json")]
    for track in ("a1", "a2", "a3", "c1", "c2"):
        args.extend((f"--{track}-checkpoint", str(tmp_path / f"{track}.pt")))
    with mock.patch.object(meco_track_training, "evaluate_meco", return_value=fake) as evaluate, \
         contextlib.redirect_stdout(output):
        meco_track_training.main(args)
    kwargs = evaluate.call_args.kwargs
    assert kwargs["split"] == "validation"
    assert kwargs["evaluation_protocol"] == str(path)
    assert kwargs["evaluation_case"] == STRATUM
    assert kwargs["episode_seeds"] == (101, 102)


def test_supervisor_evaluation_command_carries_exact_binding(tmp_path):
    path = tmp_path / "protocol.json"
    _write_protocol(path)
    command = meco_supervisor._evaluation_command(checkpoints={track: Path(f"{track}.pt")
        for track in ("A1", "A2", "A3", "C1", "C2")}, scenario="pickup_simple",
        categories=CATEGORIES, episodes=2, seed=101, device="cuda:1",
        output=tmp_path / "eval.json", split="validation", evaluation_protocol=path,
        evaluation_case=STRATUM, episode_seeds=SEEDS)
    assert command[command.index("--split") + 1] == "validation"
    assert command[command.index("--episode-seeds") + 1] == "101,102"
    assert command[command.index("--evaluation-protocol") + 1] == str(path)


def test_supervisor_status_keeps_training_and_evaluation_protocols_separate(
        tmp_path, monkeypatch):
    supervisor = meco_supervisor.MecoSupervisor.__new__(meco_supervisor.MecoSupervisor)
    supervisor.queue_path = tmp_path / "queue.json"
    supervisor.status_path = tmp_path / "status.json"
    supervisor.supervisor_id = "sup"
    supervisor.protocol_id = "run-protocol"
    supervisor.prediction_mode = "cv"
    supervisor.source_hash = "source"
    supervisor.qualification_protocol_id = "training-qualification"
    supervisor.qualification_protocol_sha256 = "training-sha"
    supervisor.evaluation_split = "validation"
    supervisor.evaluation_protocol_id = "heldout-evaluation"
    supervisor.evaluation_protocol_sha256 = "evaluation-sha"
    supervisor.evaluation_binding = {"protocol_id": "heldout-evaluation",
        "split": "validation", "seeds": SEEDS}
    supervisor.evaluation_episode_seeds = SEEDS
    supervisor.warm_start_hashes = {}
    supervisor.configuration_hash = "config"
    supervisor.completed_cycles = 0
    supervisor.completed_chunks = 0
    supervisor.current_assignment = None
    supervisor.request_stop = False
    supervisor._lane_payload = lambda track, _statuses: {"track": track, "pid": None}
    supervisor._task_scheduler_status = lambda: None
    supervisor._evaluation_status = lambda: {"active": False}
    saved = []
    monkeypatch.setattr(meco_supervisor, "_atomic_json",
        lambda path, payload: saved.append((path, payload)))
    supervisor._publish(cycle=0, phase="training",
        statuses={track: "queued" for track in ("A1", "A2", "A3", "C1", "C2")})
    queue, status = (entry[1] for entry in saved)
    for payload in (queue, status):
        assert payload["qualification_protocol_id"] == "training-qualification"
        assert payload["evaluation_protocol_id"] == "heldout-evaluation"
        assert payload["evaluation_protocol_sha256"] == "evaluation-sha"
        assert payload["evaluation_split"] == "validation"
