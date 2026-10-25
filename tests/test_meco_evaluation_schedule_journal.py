import hashlib
import json

import pytest

from frc_defense.meco_evaluation_schedule import (
    MecoEvaluationSchedule, _digest, validate_history_indexes,
    verify_evaluation_artifacts,
)
from frc_defense.meco_tracks import MECO_TRACKS


def _case(index):
    return {"index": index, "scenario_id": "pickup_simple",
        "category": "single", "seeds": [100 + index], "split": "diagnostic",
        "protocol_id": "protocol-id", "protocol_sha256": "a" * 64}


def _snapshots(root):
    rows = []
    for track in sorted(MECO_TRACKS):
        path = root / f"{track}.pt"
        path.write_bytes(track.encode())
        rows.append({"track": track, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def _journal(root, schedule, *, resume=False):
    return MecoEvaluationSchedule(root / "state.json",
        manifest_sha256="m" * 64, configuration_sha256="c" * 64,
        schedule=schedule, schedule_sha256=_digest(schedule),
        protocol_sha256="a" * 64, resume=resume)


def test_slow_evaluator_keeps_every_cycle_and_retries_same_head(tmp_path):
    schedule = [_case(i) for i in range(3)]
    journal = _journal(tmp_path, schedule)
    rows = _snapshots(tmp_path)
    # Training completes three cycles while evaluation case 0 is still active.
    for i, case in enumerate(schedule):
        journal.enqueue(i, i + 1, case, rows)
    assert journal.head()["index"] == 0
    assert journal.summary()["pending_count"] == 3
    assert journal.summary()["enqueued_count"] == 3
    journal.begin_attempt(tmp_path / "attempt-1.json")
    journal.fail_attempt("nonzero exit")
    assert journal.head()["index"] == 0
    journal.begin_attempt(tmp_path / "attempt-2.json")
    journal.verify_attempt({"output_path": "verified.json"})
    journal.complete("history-entry")
    assert journal.head()["index"] == 1
    assert journal.summary()["completed_count"] == 1
    assert journal.summary()["pending_count"] == 2


def test_long_running_attempt_persists_failure_time_separately_from_start(tmp_path,
                                                                          monkeypatch):
    from frc_defense import meco_evaluation_schedule as schedule_module

    schedule = [_case(0)]
    journal = _journal(tmp_path, schedule)
    rows = _snapshots(tmp_path)
    journal.enqueue(0, 1, schedule[0], rows)
    clock = [10.]
    monkeypatch.setattr(schedule_module.time, "time", lambda: clock[0])
    journal.begin_attempt(tmp_path / "long-attempt.json")
    clock[0] = 1010.
    journal.fail_attempt("evaluator failed after a long run")
    attempt = journal.head()["attempts"][0]
    assert attempt["started_at_unix"] == 10.
    assert attempt["failed_at_unix"] == 1010.


def test_restart_reconciles_verified_history_and_retries_interrupted_attempt(tmp_path,
                                                                            monkeypatch):
    from frc_defense import meco_evaluation_schedule as schedule_module
    from frc_defense import meco_supervisor as supervisor_module

    schedule = [_case(i) for i in range(2)]
    journal = _journal(tmp_path, schedule)
    rows = _snapshots(tmp_path)
    journal.enqueue(0, 1, schedule[0], rows)
    journal.enqueue(1, 2, schedule[1], rows)
    clock = [50.]
    monkeypatch.setattr(schedule_module.time, "time", lambda: clock[0])
    journal.begin_attempt(tmp_path / "active.json")
    resumed = _journal(tmp_path, schedule, resume=True)
    clock[0] = 100.
    resumed.reconcile_history([])
    assert resumed.head()["status"] == "failed"
    assert resumed.head()["index"] == 0
    assert resumed.head()["attempts"][0]["failed_at_unix"] == 100.

    supervisor = object.__new__(supervisor_module.MecoSupervisor)
    supervisor.evaluator = None
    supervisor.request_stop = False
    supervisor.evaluation_device = "cpu"
    supervisor.evaluation_schedule_journal = resumed
    monkeypatch.setattr(supervisor_module.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("restart retry must respect failure backoff"))
    supervisor._start_next_evaluation()
    assert resumed.head()["index"] == 0
    assert resumed.head()["status"] == "failed"
    assert len(resumed.head()["attempts"]) == 1

    resumed.begin_attempt(tmp_path / "retry.json")
    output, checkpoint_paths, _, _ = _valid_artifacts(tmp_path, schedule[0])
    artifact = verify_evaluation_artifacts(schedule[0], output, 0, checkpoint_paths)
    resumed.verify_attempt(artifact)
    snapshots = [{**row, "campaign_binding": {
        "campaign_manifest_sha256": "m" * 64,
        "configuration_sha256": "c" * 64, "phase4_train_exposure": False}}
        for row in rows]
    resumed.reconcile_history([{"schedule_index": 0,
        "schedule_case": schedule[0], "evaluation_schedule_sha256": _digest(schedule),
        "status": "complete", "return_code": 0,
        "checkpoint_snapshots": snapshots, "output_file": output.name,
        "report_sha256": artifact["report_sha256"],
        "status_sha256": artifact["status_sha256"], "entry_sha256": "entry0"}])
    assert resumed.head()["index"] == 1


def test_evaluator_spawn_failure_retains_same_schedule_index(tmp_path, monkeypatch):
    from frc_defense import meco_supervisor as supervisor_module

    schedule = [_case(0)]
    journal = _journal(tmp_path, schedule)
    snapshots = _snapshots(tmp_path)
    journal.enqueue(0, 1, schedule[0], snapshots)
    supervisor = object.__new__(supervisor_module.MecoSupervisor)
    supervisor.evaluator = None
    supervisor.request_stop = False
    supervisor.evaluation_device = "cpu"
    supervisor.evaluation_schedule_journal = journal
    supervisor.evaluation_output = tmp_path
    supervisor.evaluation_prediction_mode = "current_pose"
    supervisor.scenario = "pickup_simple"
    supervisor.evaluation_categories = ("single",)
    supervisor.evaluation_episodes = 1
    supervisor.evaluation_protocol_path = None
    supervisor.schedule_evaluation_protocol_path = None
    supervisor.evaluation_case = None
    supervisor.evaluator_job = None
    supervisor.evaluator_log = None
    supervisor.last_evaluation = None

    launches = []
    clock = [supervisor_module.time.time()]
    monkeypatch.setattr(supervisor_module.time, "time", lambda: clock[0])
    def fail_spawn(*_args, **_kwargs):
        launches.append(1)
        clock[0] += 1000  # Simulate elapsed time before failure is recorded.
        raise OSError("simulated spawn failure")

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", fail_spawn)
    supervisor._start_next_evaluation()
    assert journal.head()["index"] == 0
    assert journal.head()["status"] == "failed"
    assert journal.head()["attempts"][0]["status"] == "failed"
    attempt = journal.head()["attempts"][0]
    started_at = attempt["started_at_unix"]
    failed_at = attempt["failed_at_unix"]
    assert failed_at == started_at + 1000
    clock[0] = failed_at + 1
    supervisor._start_next_evaluation()
    assert len(launches) == 1
    clock[0] = failed_at + 2
    supervisor._start_next_evaluation()
    assert len(launches) == 2


def test_first_evaluator_poll_with_no_active_job_is_safe(tmp_path):
    from frc_defense.meco_supervisor import MecoSupervisor

    supervisor = object.__new__(MecoSupervisor)
    supervisor.stop_file = tmp_path / "STOP"
    supervisor.request_stop = False
    supervisor.evaluator = None
    supervisor.last_evaluation = None
    supervisor.evaluation_schedule_journal = None
    supervisor.pending_evaluations = []
    supervisor.started = False
    supervisor._start_next_evaluation = lambda **_kwargs: setattr(supervisor, "started", True)

    supervisor._poll_evaluator(0, {})
    assert supervisor.started is False


def test_manifest_state_initialization_requires_claimed_roots(tmp_path):
    from frc_defense.meco_supervisor import MecoSupervisor

    supervisor = object.__new__(MecoSupervisor)
    supervisor.manifest = {"campaign": {"id": "fixture"}}
    supervisor._campaign_roots_claimed = False
    supervisor.output = tmp_path / "campaign"
    supervisor.evaluation_output = tmp_path / "evaluations"

    with pytest.raises(RuntimeError, match="requires claimed roots"):
        supervisor._read_or_create_runs()
    assert not supervisor.output.exists()
    assert not supervisor.evaluation_output.exists()


def test_run_acquires_lease_claims_fresh_roots_then_initializes(tmp_path, monkeypatch):
    from frc_defense.meco_supervisor import MecoSupervisor

    supervisor = object.__new__(MecoSupervisor)
    supervisor.devices = ("cuda:0",)
    supervisor.evaluation_device = "cuda:1"
    supervisor.manifest_digest = "m" * 64
    supervisor.manifest = {"campaign": {"id": "fixture"}}
    supervisor.resume_campaign = False
    supervisor._campaign_roots_claimed = False
    supervisor.output = tmp_path / "campaign"
    supervisor.evaluation_output = tmp_path / "evaluations"
    events = []

    class FakeLease:
        def acquire(self):
            events.append("lease")
        def release(self):
            events.append("release")

    monkeypatch.setattr("frc_defense.meco_device_lease.lease_devices",
        lambda *_args, **_kwargs: FakeLease())
    claim_roots = supervisor._claim_fresh_manifest_roots
    def record_claim():
        claim_roots()
        events.append("claim")
    supervisor._claim_fresh_manifest_roots = record_claim
    def run_locked():
        assert supervisor._campaign_roots_claimed
        assert supervisor.output.is_dir()
        assert supervisor.evaluation_output.is_dir()
        events.append("initialize")
        return {"status": "stub"}
    supervisor._run_locked = run_locked

    assert not supervisor.output.exists()
    assert not supervisor.evaluation_output.exists()
    assert supervisor.run() == {"status": "stub"}
    assert events == ["lease", "claim", "initialize", "release"]


def test_restart_recovers_last_completed_cycle_missing_queue_snapshot(tmp_path, monkeypatch):
    from frc_defense import meco_supervisor as supervisor_module

    schedule = [_case(0)]
    journal = _journal(tmp_path, schedule)
    supervisor = object.__new__(supervisor_module.MecoSupervisor)
    supervisor.evaluation_schedule_journal = journal
    supervisor.completed_cycles = 1
    supervisor.completed_updates = {track: 1 for track in MECO_TRACKS}
    supervisor.chunk_updates = 1
    supervisor.evaluation_schedule = schedule
    supervisor.evaluation_schedule_sha256 = _digest(schedule)
    supervisor.manifest = {"evaluation_protocol": {"protocol_id": "protocol-id",
        "sha256": "a" * 64}}
    supervisor.evaluation_output = tmp_path / "evaluations"
    supervisor.sources = {"source": "pinned"}
    supervisor.track_dirs = {}
    for track in MECO_TRACKS:
        track_dir = tmp_path / "runs" / track
        track_dir.mkdir(parents=True)
        (track_dir / "latest.pt").write_bytes(track.encode())
        supervisor.track_dirs[track] = track_dir
    monkeypatch.setattr(supervisor_module, "source_identity",
        lambda: supervisor.sources)

    supervisor._reconcile_completed_schedule_cycles()
    assert journal.head()["index"] == 0
    assert journal.head()["cycle"] == 1
    assert len(journal.head()["checkpoint_snapshots"]) == 5


@pytest.mark.parametrize("indexes", [[0, 0], [1], [0, 2]])
def test_history_schedule_index_audit_rejects_duplicate_or_missing(indexes):
    rows = [{"schedule_index": index, "schedule_case": {"index": index},
        "evaluation_schedule_sha256": "s" * 64} for index in indexes]
    with pytest.raises(ValueError, match="duplicated or contain skips"):
        validate_history_indexes(rows, schedule_count=3,
            expected_schedule_sha256="s" * 64)


def _valid_artifacts(root, case):
    checkpoints = {}
    hashes = {}
    policies = {}
    for track in MECO_TRACKS:
        path = root / f"{track}.pt"
        path.write_bytes(track.encode())
        checkpoints[track] = path
        hashes[track] = hashlib.sha256(path.read_bytes()).hexdigest()
        policies[track] = {"checkpoint_sha256": hashes[track]}
    output = root / "eval.json"
    report = {"evaluation_id": "eval-1", "protocol_id": "eval-protocol",
        "scenario_id": case["scenario_id"], "categories": [case["category"]],
        "seeds": case["seeds"], "split": case["split"],
        "evaluation_protocol_id": case["protocol_id"],
        "evaluation_protocol_sha256": case["protocol_sha256"],
        "schedule_case_binding": case, "checkpoint_sha256": hashes,
        "policy_manifest": policies}
    output.write_text(json.dumps(report))
    status = {"schema_version": 1, "status": "complete", "evaluation_id": "eval-1",
        "protocol_id": "eval-protocol", "output_file": output.name,
        "split": case["split"], "schedule_case_binding": case,
        "exact_episode_seeds": case["seeds"],
        "evaluation_protocol_id": case["protocol_id"],
        "evaluation_protocol_sha256": case["protocol_sha256"]}
    (root / "eval.json.status.json").write_text(json.dumps(status))
    return output, checkpoints, report, status


def test_artifacts_require_exit_success_and_exact_report_status_bindings(tmp_path):
    case = _case(0)
    output, checkpoints, report, status = _valid_artifacts(tmp_path, case)
    assert verify_evaluation_artifacts(case, output, 0, checkpoints)["evaluation_id"] == "eval-1"
    status_path = tmp_path / "eval.json.status.json"
    status_bytes = status_path.read_bytes()
    status_path.unlink()
    with pytest.raises(ValueError, match="missing/malformed"):
        verify_evaluation_artifacts(case, output, 0, checkpoints)
    status_path.write_bytes(status_bytes)
    with pytest.raises(ValueError, match="exited"):
        verify_evaluation_artifacts(case, output, 7, checkpoints)
    output.unlink()
    with pytest.raises(ValueError, match="missing/malformed"):
        verify_evaluation_artifacts(case, output, 0, checkpoints)
    output.write_text(json.dumps({**report, "seeds": [999]}))
    with pytest.raises(ValueError, match="case or exact seed"):
        verify_evaluation_artifacts(case, output, 0, checkpoints)
    output.write_text(json.dumps(report))
    status["schedule_case_binding"] = _case(1)
    (tmp_path / "eval.json.status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="bind the scheduled case"):
        verify_evaluation_artifacts(case, output, 0, checkpoints)
