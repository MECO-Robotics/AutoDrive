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
    with pytest.raises(ValueError, match="overlap"):
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


def test_matrix_accepts_and_binds_canonical_v2_protocol(tmp_path, monkeypatch):
    from frc_defense import curriculum_training, fuel_physics_hip
    from frc_defense import meco_ablation_matrix as matrix
    from frc_defense import meco_qualification as qualification
    from frc_defense.meco_ablation_matrix import MatrixCondition
    from frc_defense.meco_tracks import MECO_TRACKS

    scenario = "pickup_simple"
    categories = ("single", "cluster")
    calibration = {"scenario_id": scenario}
    calibration_ids = {scenario: "c" * 64}
    calibration_pools = {scenario: {"master_seed": 77, "episodes": 100,
        "repetitions": 2, "cases": [{"category": case.category,
            "layout_id": case.layout_id, "defender_family": case.defender_family,
            "start_condition": case.start_condition, "sensor_profile": case.sensor_profile,
            "seeds": list(case.seeds)} for case in
            qualification.protocol_cases(scenario, "calibration", episodes=100,
                master_seed=77)]}}
    monkeypatch.setattr(qualification, "_calibration_report_index",
        lambda _reports, scenarios: (
            {name: calibration_ids[name] for name in scenarios},
            {name: calibration_pools[name] for name in scenarios}, "a" * 64,
            {name: "b" * 64 for name in scenarios}))
    def derive_margins(_reports, target, _configured):
        return ({category: .1 for category in categories}, {
            "schema_version": 1,
            "method": "declared_operational_threshold_plus_multiplier_times_max_stratum_standard_error",
            "calibration_id": calibration_ids[target],
            "operational_gain_threshold": {category: .1 for category in categories},
            "uncertainty_by_category": {category: {
                "maximum_stratum_standard_error": 0.0} for category in categories},
            "uncertainty_multiplier": 0.0})
    monkeypatch.setattr(qualification, "derive_useful_gain_margins", derive_margins)
    protocol = qualification.protocol_document(scenario, episodes=100,
        master_seed=77, calibration_report=calibration,
        safety_noninferiority_margins={name: .1
            for name in qualification.SAFETY_METRIC_NAMES})
    protocol_path = tmp_path / "canonical-v2.json"
    protocol_path.write_text(json.dumps(protocol, indent=2))
    case_row = next(row for row in protocol["cases"]
        if row["scenario_id"] == scenario and row["category"] == "single"
        and row["split"] == "validation")
    case = {key: case_row[key] for key in STRATUM}
    seeds = tuple(case_row["seeds"])
    cluster_case = next(row for row in protocol["cases"]
        if row["scenario_id"] == scenario and row["category"] == "cluster"
        and row["split"] == "validation" and all(row[key] == case[key] for key in STRATUM))
    assert tuple(cluster_case["seeds"]) == seeds
    condition = MatrixCondition("architecture_matched_cv_k16", "cv", None, 16,
        qualification_case=case)
    monkeypatch.setattr(fuel_physics_hip, "prepare_native_fuel_extension", lambda _device: None)
    monkeypatch.setattr(matrix, "normalized_device_info", lambda _device: {
        "device": "cuda:0", "backend": "hip", "device_name": "test"})
    monkeypatch.setattr(matrix, "_checkpoint_hashes", lambda _checkpoints: {
        track: f"{index + 1:064x}" for index, track in enumerate(MECO_TRACKS)})
    monkeypatch.setattr(matrix, "_checkpoint_configuration_provenance", lambda _checkpoints: (
        {track: "config" for track in MECO_TRACKS},
        {track: "training-protocol" for track in MECO_TRACKS}, {"shared": True}))
    monkeypatch.setattr(curriculum_training, "source_identity", lambda: {"source": "test"})
    monkeypatch.setattr(matrix, "default_conditions", lambda *_args, **_kwargs: [condition])
    observed = {}

    def stop_at_evaluator(*_args, **kwargs):
        observed.update(kwargs)
        raise RuntimeError("evaluator handoff reached")

    monkeypatch.setattr(meco_track_training, "evaluate_meco", stop_at_evaluator)
    with pytest.raises(RuntimeError, match="evaluator handoff reached"):
        matrix.run_matrix(checkpoints={track: f"{track}.pt" for track in MECO_TRACKS},
            scenario_id=scenario, categories=categories, seeds=seeds,
            device="cuda:0", output_dir=tmp_path / "matrix",
            qualification_protocol=protocol_path, qualification_case=case)
    assert observed["split"] == "validation"
    assert observed["evaluation_protocol"] == protocol_path
    assert observed["qualification_protocol_id"] == protocol["protocol_id"]
    assert observed["evaluation_case"] == case
