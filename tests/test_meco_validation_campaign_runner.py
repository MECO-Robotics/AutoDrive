import hashlib
import json
from pathlib import Path

import pytest

from frc_defense.meco_qualification import protocol_document
from frc_defense.meco_tracks import MECO_TRACKS
from frc_defense.meco_validation_campaign import compile_validation_campaign
from frc_defense.meco_validation_campaign_runner import (
    _sha,
    _validate_plan,
    _validate_replay_artifact,
    aggregate_campaign,
    run_campaign,
)


def _inputs(tmp_path, *, v4=False, monkeypatch=None):
    protocol_path = tmp_path / "protocol.json"
    protocol = protocol_document()
    if v4:
        from frc_defense import meco_qualification
        protocol.update({"protocol_version": meco_qualification.CALIBRATED_PROTOCOL_VERSION,
            "gate_calibration_status": "matched_reference_variance_and_reviewed_margin_frozen",
            "safety_gate_status": "metrics_schema_verified",
            "safety_metrics_schema": meco_qualification.SAFETY_METRICS_SCHEMA,
            "safety_metrics_source_identity_sha256": "a" * 64})
        protocol["protocol_id"] = hashlib.sha256(json.dumps(
            {key: value for key, value in protocol.items() if key != "protocol_id"},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        monkeypatch.setattr(meco_qualification, "_validate_v2_protocol", lambda *_: None)
    protocol_path.write_text(json.dumps(protocol, sort_keys=True,
        indent=2) + "\n", encoding="utf-8")
    checkpoints = {}
    configs = {}
    for track in MECO_TRACKS:
        checkpoint = tmp_path / f"{track}.checkpoint"
        checkpoint.write_bytes(track.encode())
        checkpoints[track] = str(checkpoint)
        configs[track] = hashlib.sha256((track + " config").encode()).hexdigest()
    plan = compile_validation_campaign(protocol_path=protocol_path,
        checkpoint_paths=checkpoints, source_identity={"prepared": "hip-device"},
        training_configuration_sha256=configs)
    return protocol_path, checkpoints, plan


def test_v4_protocol_plan_is_admitted_as_validation_only(tmp_path, monkeypatch):
    _, _, plan = _inputs(tmp_path, v4=True, monkeypatch=monkeypatch)
    assert plan["qualification_claim"] is False
    assert plan["qualification_eligible"] is True
    _validate_plan(plan)
    assert plan["protocol"]["gate_calibration_status"] == (
        "matched_reference_variance_and_reviewed_margin_frozen")
    mismatched = json.loads(json.dumps(plan))
    mismatched["protocol"]["gate_calibration_status"] = "calibrated_baseline_frozen"
    mismatched["plan_digest"] = _sha({key: value for key, value in mismatched.items()
        if key != "plan_digest"})
    with pytest.raises(ValueError, match="calibration eligibility/status"):
        _validate_plan(mismatched)


def test_runner_resumes_only_valid_complete_mocked_jobs_and_aggregates_per_scenario(
        tmp_path, monkeypatch):
    protocol_path, checkpoints, plan = _inputs(tmp_path)
    executed = []

    def fake_validator(job, job_dir, *, device):
        marker = job_dir / "matrix" / "mock.matrix.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        assert marker.is_file()
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return {"matrix_path": str(marker), "matrix_sha256": hashlib.sha256(
            marker.read_bytes()).hexdigest(), "condition_report_sha256": {},
            "replay_sha256": {}, "seed_manifest_sha256": sha(job_dir / job["seed_manifest_file"]),
            "qualification_case_sha256": sha(job_dir / "qualification-case.json"),
            "replay_selection_sha256": (sha(job_dir / "replay-selection.json")
                if (job_dir / "replay-selection.json").exists() else None),
            "matrix_report_sha256": "report", "matrix_id": job["job_id"]}

    monkeypatch.setattr(
        "frc_defense.meco_validation_campaign_runner.validate_job_artifacts", fake_validator)

    def mock_matrix(argv):
        assert argv[0] == __import__("sys").executable
        assert argv[argv.index("--device") + 1] == "cuda:1"
        assert "--qualification-protocol" in argv
        assert "--seed-manifest" in argv
        output = Path(argv[argv.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "mock.matrix.json").write_text("{}\n", encoding="utf-8")
        executed.append(argv)
        return 0

    result = run_campaign(plan, output_dir=tmp_path / "run", device="cuda:1",
        checkpoint_paths=checkpoints, protocol_path=protocol_path,
        matrix_script=tmp_path / "mock_matrix.py",
        current_source_identity={"prepared": "hip-device"}, runner=mock_matrix)
    assert result["job_count"] == 28
    assert result["diagnostic_only"] is True
    assert len(executed) == 28

    # A second invocation resumes every sealed job without invoking matrix execution.
    resumed = run_campaign(plan, output_dir=tmp_path / "run", device="cuda:1",
        checkpoint_paths=checkpoints, protocol_path=protocol_path,
        matrix_script=tmp_path / "mock_matrix.py",
        current_source_identity={"prepared": "hip-device"},
        runner=lambda _argv: pytest.fail("completed matrix must not be rerun"))
    assert resumed == result

    def fake_summary(matrix_path, *, replicates):
        assert replicates == 100
        return {"matrix": {"matrix_id": Path(matrix_path).parent.name,
                            "file_sha256": hashlib.sha256(Path(matrix_path).read_bytes()).hexdigest()},
                "summaries": [{"scenario_id": "preserved-by-campaign-row"}]}

    monkeypatch.setattr("scripts.summarize_meco_ablation_matrix.summarize_matrix", fake_summary)
    summary = aggregate_campaign(plan, tmp_path / "run" / "campaign.complete.json")
    assert summary["job_count"] == 28
    assert summary["pooled_estimate"] is None
    assert summary["scenario_count"] == len({job["scenario_id"] for job in plan["jobs"]})
    assert all("scenario_id" in row for row in summary["scenarios"])


def test_runner_rejects_plan_without_required_checkpoint_and_config_bindings(tmp_path):
    protocol_path, checkpoints, plan = _inputs(tmp_path)
    broken = json.loads(json.dumps(plan))
    broken["jobs"][0]["checkpoint_sha256"] = None
    broken["jobs"][0]["job_id"] = hashlib.sha256(json.dumps(
        {key: value for key, value in broken["jobs"][0].items() if key != "job_id"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    body = {key: value for key, value in broken.items() if key != "plan_digest"}
    broken["plan_digest"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match="all five checkpoint/config hashes"):
        run_campaign(broken, output_dir=tmp_path / "out", device="cuda:0",
            checkpoint_paths=checkpoints, protocol_path=protocol_path,
            matrix_script=tmp_path / "matrix.py",
            current_source_identity={"prepared": "hip-device"}, runner=lambda _argv: 0)


def test_campaign_aggregator_rejects_missing_job(tmp_path):
    protocol_path, _checkpoints, plan = _inputs(tmp_path)
    campaign = {"schema": "meco-validation-campaign-complete-v1",
        "plan_digest": plan["plan_digest"], "diagnostic_only": True,
        "qualification_claim": False, "qualification_eligible": False,
        "split": "validation", "device": "cuda:0", "job_count": 28,
        "jobs": []}
    campaign["campaign_sha256"] = hashlib.sha256(json.dumps(campaign,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path = tmp_path / "campaign.complete.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="exact 28 planned jobs"):
        aggregate_campaign(plan, path)


def test_replay_validator_accepts_actual_replay_provenance_schema(tmp_path):
    source = {"sha256": "a" * 64}
    checkpoint = "b" * 64
    case = {"layout_id": "layout", "defender_family": "scripted",
        "start_condition": "start", "sensor_profile": "standard"}
    safety = {"robot_contact_ticks": 0}
    row = {"scenario_id": "pickup_simple", "category": "single", "seed": 17,
        "policy": "A1", "split": "validation", "protocol_id": "train-protocol",
        "checkpoint_sha256": checkpoint, "metrics": {"success": 1.0},
        "meco_metric_ledger": {"safety_metrics": safety},
        "execution_failures": [], "route_trace": [],
        "executed_planner_route_trace": []}
    record_digest = _sha(row)
    report_path = tmp_path / "condition.evaluation.json"
    condition_id = "condition-1"
    evaluation_id = "evaluation-1"
    route_digest = _sha({"route_trace": row["route_trace"],
        "executed_planner_route_trace": row["executed_planner_route_trace"]})
    outcome_digest = _sha({"metrics": row["metrics"], "safety_metrics": safety,
        "execution_failures": row["execution_failures"]})
    safety_digest = _sha(safety)
    replay_ref = {"simulation_id": "scenario-meco-sample",
        "result_path": str(tmp_path / "scenario-meco-sample.result.json"),
        "evaluation_report_path": str(report_path), "evaluation_record_index": 3,
        "evaluation_record_digest": record_digest}
    row["dashboard_replay"] = replay_ref
    provenance = {"evaluation_id": evaluation_id, "condition_id": condition_id,
        "evaluation_report_path": str(report_path), "evaluation_record_index": 3,
        "evaluation_record_digest": record_digest, "replay_source_sha256": source["sha256"],
        "checkpoint_sha256": checkpoint, "protocol_id": row["protocol_id"],
        "qualification_protocol_id": "qualification-protocol",
        "scenario_id": row["scenario_id"], "category": row["category"],
        "split": row["split"], "seed": row["seed"], "qualification_case": case,
        "policy": row["policy"], "route_digest": route_digest,
        "outcome_digest": outcome_digest, "safety_digest": safety_digest}
    replay = {"simulation_id": replay_ref["simulation_id"],
        "scenario_id": row["scenario_id"], "category": row["category"],
        "seed": row["seed"], "policy": row["policy"], "split": row["split"],
        "qualification_case": case, "evaluation_id": evaluation_id,
        "condition_id": condition_id, "evaluation_report_path": str(report_path),
        "evaluation_record_index": 3, "evaluation_record_digest": record_digest,
        "source_identity_sha256": source["sha256"],
        "replay_source_sha256": source["sha256"],
        "qualification_protocol_id": "qualification-protocol",
        "protocol_id": row["protocol_id"], "checkpoint_sha256": checkpoint,
        "route_digest": route_digest, "outcome_digest": outcome_digest,
        "safety_digest": safety_digest, "replay_provenance": provenance}
    report = {"evaluation_id": evaluation_id,
        "experiment_matrix": {"condition_id": condition_id}}
    job = {"scenario_id": row["scenario_id"], "qualification_case": case,
        "source_identity": source,
        "seed_manifest": {"protocol_id": "qualification-protocol"},
        "checkpoint_sha256": {"A1": checkpoint}}

    _validate_replay_artifact(replay, replay_ref, report=report,
        report_path=report_path, row=row, report_index=3, job=job)
    bad = json.loads(json.dumps(replay))
    bad["replay_provenance"]["evaluation_record_digest"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source/protocol/row binding"):
        _validate_replay_artifact(bad, replay_ref, report=report,
            report_path=report_path, row=row, report_index=3, job=job)
