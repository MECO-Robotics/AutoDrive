import copy
import hashlib
import io
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from frc_defense import meco_training_manifest as manifest_api
from frc_defense.meco_diagnostic_monitor import MecoDiagnosticMonitor
from frc_defense.meco_supervisor import _evaluation_command
from frc_defense.meco_track_training import validate_diagnostic_frontier_binding
from frc_defense.meco_tracks import MECO_TRACKS


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol = {"protocol_version": "meco-qualification-v1", "master_seed": 17,
        "cases": [
            {"scenario_id": "pickup_simple", "category": "single", "split": "train", "seeds": [1, 2]},
            {"scenario_id": "pickup_simple", "category": "single", "split": "validation", "seeds": [3]},
            {"scenario_id": "pickup_simple", "category": "single", "split": "test", "seeds": [4]}]}
    protocol["protocol_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(manifest_api, "source_identity", lambda: {"source": "fixture"})
    monkeypatch.setattr(manifest_api, "_git_binding", lambda _: {
        "commit": "a" * 40, "clean": True, "status_sha256": "b" * 64})
    monkeypatch.setattr(manifest_api, "_model_schema", lambda track: {
        "sha256": hashlib.sha256(json.dumps([track, "schema"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "parameters": [track, "schema"]})
    seed_partitions = {"train": [1, 2], "validation": [3], "test": [4]}
    campaign_id = "meco-five-fixture"
    root = tmp_path / campaign_id
    tracks = {track: {"track": track, "scenario_id": "pickup_simple",
        "category": "single", "seed": 17, "device": "cuda:0",
        "output": str(root / "checkpoints" / track)} for track in MECO_TRACKS}
    config = {"seed": 17, "campaign_id": campaign_id,
        "campaign_root": str(root), "evaluation_output": str(root / "evaluations"),
        "qualification_protocol": str(protocol_path),
        "evaluation_protocol": str(protocol_path),
        "training_devices": ["cuda:0", "cuda:1"],
        "lease_contract": manifest_api.LEASE_CONTRACT,
        "evaluation_device": "cuda:1", "tracks": tracks,
        "scheduler": {"algorithm": "MecoTaskScheduler", "version": 1,
            "master_seed": 17, "seed_partitions": seed_partitions,
            "task_assignment_contract": ("sequential-cycle deterministic curriculum assignment; all five tracks share "
                "scenario/category/seed; train seed indexed by cycle modulo frozen protocol pool")},
        "evaluation": {"split": "diagnostic", "scenario_id": "pickup_simple",
            "category": "single", "episode_seeds": [100]}}
    return tmp_path, config


def _monitor(protocol, protocol_sha):
    return {"enabled": True, "cadence_cycles": 3, "episode_budget": 2,
        "device": "cuda:1", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha, "scenario_id": "pickup_simple",
        "category": "single", "seeds": [100, 101],
        "backlog_policy": "latest_only"}


def _frontier(root, cycle):
    snapshot_root = root / f"snapshot-{cycle}"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for track in sorted(MECO_TRACKS):
        path = snapshot_root / f"{track}.pt"
        path.write_bytes(f"{track}-{cycle}".encode())
        rows.append({"track": track, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    sources = {"source": "fixture"}
    source_sha = hashlib.sha256(json.dumps(sources, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    body = {"cycle": cycle, "update": cycle * 10,
        "checkpoint_snapshots": rows, "source_sha256": source_sha,
        "configuration_sha256": "c" * 64, "protocol_sha256": "p" * 64}
    return {"frontier_id": hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest(), **body}


def _journal(root, *, resume=False):
    monitor = {"enabled": True, "cadence_cycles": 1, "episode_budget": 2,
        "device": "cuda:1", "protocol_id": "protocol", "protocol_sha256": "p" * 64,
        "scenario_id": "pickup_simple", "category": "single", "seeds": [100, 101],
        "backlog_policy": "latest_only", "split": "diagnostic",
        "qualification_eligible": False}
    return MecoDiagnosticMonitor(root, manifest_sha256="m" * 64,
        configuration_sha256="c" * 64, protocol_sha256="p" * 64,
        monitor=monitor, resume=resume)


def _valid_result(root, frontier):
    report = root / f"report-{frontier['cycle']}.json"
    status = root / f"status-{frontier['cycle']}.json"
    hashes = {row["track"]: row["sha256"] for row in frontier["checkpoint_snapshots"]}
    binding = {"schema_version": 1, "frontier_id": frontier["frontier_id"],
        "cycle": frontier["cycle"], "update": frontier["update"],
        "checkpoint_sha256": hashes,
        "source_sha256": frontier["source_sha256"],
        "configuration_sha256": frontier["configuration_sha256"],
        "protocol_id": "protocol", "protocol_sha256": frontier["protocol_sha256"]}
    case = {"index": frontier["cycle"] - 1, "scenario_id": "pickup_simple",
        "category": "single", "seeds": [100, 101], "split": "diagnostic",
        "protocol_id": "protocol", "protocol_sha256": frontier["protocol_sha256"],
        "qualification_case": None}
    sources = {"source": "fixture"}
    report.write_text(json.dumps({"diagnostic_frontier_binding": binding,
        "schedule_case_binding": case, "checkpoint_sha256": hashes,
        "exact_episode_seeds": [100, 101], "evaluation_protocol_id": "protocol",
        "evaluation_protocol_sha256": frontier["protocol_sha256"],
        "qualification_eligible": False, "sources": sources}))
    status.write_text(json.dumps({"diagnostic_frontier_binding": binding,
        "schedule_case_binding": case, "exact_episode_seeds": [100, 101],
        "evaluation_protocol_id": "protocol",
        "evaluation_protocol_sha256": frontier["protocol_sha256"],
        "qualification_eligible": False, "diagnostic_only": True}))
    return {"report_path": str(report), "status_path": str(status),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "status_sha256": hashlib.sha256(status.read_bytes()).hexdigest(),
        "checkpoint_sha256": {row["track"]: row["sha256"]
            for row in frontier["checkpoint_snapshots"]},
        "frontier_id": frontier["frontier_id"],
        "source_sha256": frontier["source_sha256"],
        "configuration_sha256": frontier["configuration_sha256"],
        "protocol_sha256": frontier["protocol_sha256"],
        "frontier_binding_sha256": hashlib.sha256(json.dumps(binding,
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest(),
        "seeds": [100, 101], "diagnostic_only": True,
        "qualification_eligible": False}


def test_manifest_freezes_monitor_case_and_rejects_split_overlap(campaign):
    tmp_path, config = campaign
    protocol_path = config["evaluation_protocol"]
    protocol = json.loads(open(protocol_path).read())
    digest = hashlib.sha256(open(protocol_path, "rb").read()).hexdigest()
    config = copy.deepcopy(config)
    config["diagnostic_monitor"] = _monitor(protocol, digest)
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    assert manifest["diagnostic_monitor"]["qualification_eligible"] is False
    assert manifest_api.validate_campaign_manifest(manifest,
        source_root=tmp_path)["valid"]

    bad = copy.deepcopy(manifest)
    bad["diagnostic_monitor"]["seeds"] = [1, 100]
    bad["diagnostic_monitor"]["episode_budget"] = 2
    bad = _seal(bad)
    with pytest.raises(ValueError, match="overlap"):
        manifest_api.validate_campaign_manifest(bad, source_root=tmp_path)


def test_monitor_seeds_must_also_avoid_qualification_protocol(campaign):
    tmp_path, config = campaign
    evaluation_protocol_path = Path(config["evaluation_protocol"])
    evaluation_protocol = json.loads(evaluation_protocol_path.read_text())
    evaluation_digest = hashlib.sha256(evaluation_protocol_path.read_bytes()).hexdigest()
    qualification = copy.deepcopy(evaluation_protocol)
    qualification["cases"] = [
        {"scenario_id": "pickup_simple", "category": "single", "split": "train", "seeds": [100, 102]},
        {"scenario_id": "pickup_simple", "category": "single", "split": "validation", "seeds": [103]},
        {"scenario_id": "pickup_simple", "category": "single", "split": "test", "seeds": [104]}]
    qualification["protocol_id"] = hashlib.sha256(json.dumps({key: value
        for key, value in qualification.items() if key != "protocol_id"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    qualification_path = tmp_path / "qualification-with-different-partitions.json"
    qualification_path.write_text(json.dumps(qualification))
    config = copy.deepcopy(config)
    config["qualification_protocol"] = str(qualification_path)
    config["scheduler"]["seed_partitions"] = {
        "train": [100, 102], "validation": [103], "test": [104]}
    config["diagnostic_monitor"] = _monitor(evaluation_protocol, evaluation_digest)
    with pytest.raises(ValueError, match="qualification protocol partitions"):
        manifest_api.describe_campaign(config, source_root=tmp_path)


def _seal(value):
    value["canonical_sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in value.items() if k != "canonical_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    return value


def test_manifest_keeps_finite_schedule_separate_from_monitor(campaign):
    tmp_path, config = campaign
    protocol_path = config["evaluation_protocol"]
    protocol = json.loads(open(protocol_path).read())
    digest = hashlib.sha256(open(protocol_path, "rb").read()).hexdigest()
    config = copy.deepcopy(config)
    config["diagnostic_monitor"] = _monitor(protocol, digest)
    config["evaluation_schedule"] = [{"scenario_id": "pickup_simple",
        "category": "single", "seeds": [102], "split": "diagnostic",
        "protocol_id": protocol["protocol_id"], "protocol_sha256": digest}]
    with pytest.raises(ValueError, match="separate modes"):
        manifest_api.describe_campaign(config, source_root=tmp_path)


def test_latest_only_coalesces_waiting_frontiers_but_preserves_active(tmp_path):
    journal = _journal(tmp_path / "monitor")
    first, second, third = (_frontier(tmp_path, cycle) for cycle in (1, 2, 3))
    journal.enqueue(first)
    journal.begin_attempt(tmp_path / "attempt-1.json")
    journal.enqueue(second)
    journal.enqueue(third)
    summary = journal.summary()
    assert summary["active_frontier"]["frontier_id"] == first["frontier_id"]
    assert summary["queued_frontier"]["frontier_id"] == third["frontier_id"]
    assert [row["frontier_id"] for row in summary["skipped_frontiers"]] == [
        second["frontier_id"]]
    journal.verify_attempt(_valid_result(tmp_path, first))
    journal.complete()
    assert journal.pending["frontier_id"] == third["frontier_id"]


def test_cycle_three_enqueue_failure_then_cycle_four_resume_records_miss(tmp_path,
        monkeypatch):
    from frc_defense import meco_supervisor as supervisor_api

    root = tmp_path / "monitor"
    monitor = _journal(root)
    supervisor = object.__new__(supervisor_api.MecoSupervisor)
    supervisor.diagnostic_monitor = monitor
    supervisor.diagnostic_monitor_config = {"cadence_cycles": 3}
    supervisor.evaluation_schedule = None
    supervisor.pending_evaluations = []
    supervisor.chunk_updates = 10
    supervisor.last_monitor_error = None
    supervisor._enqueue_diagnostic_frontier = lambda _cycle: (_ for _ in ()).throw(
        RuntimeError("injected cycle-3 snapshot failure"))
    supervisor._queue_evaluation(3)
    assert monitor.summary()["missed_frontiers"][0]["cycle"] == 3

    resumed = _journal(root, resume=True)
    supervisor.diagnostic_monitor = resumed
    supervisor.completed_cycles = 4
    supervisor.track_dirs = {track: tmp_path / track for track in MECO_TRACKS}
    for track_dir in supervisor.track_dirs.values():
        track_dir.mkdir()
        (track_dir / "latest.pt").write_bytes(b"cycle-four")
    supervisor.evaluation_output = tmp_path
    supervisor.source_hash = "s" * 64
    supervisor.configuration_hash = "c" * 64
    supervisor.sources = {"source": "fixture"}
    monkeypatch.setattr(supervisor_api, "source_identity", lambda: supervisor.sources)
    monkeypatch.setattr(supervisor_api, "_checkpoint_info", lambda _path: {"update": 40})
    supervisor._enqueue_diagnostic_frontier = (
        supervisor_api.MecoSupervisor._enqueue_diagnostic_frontier.__get__(
            supervisor, supervisor_api.MecoSupervisor))
    supervisor._reconcile_completed_diagnostic_frontier()
    assert resumed.summary()["frontier_cycles"] == [3]
    assert resumed.summary()["missed_frontiers"][0]["cycle"] == 3
    assert resumed.pending is None


def test_recovery_refuses_latest_checkpoint_for_older_cycle(tmp_path, monkeypatch):
    from frc_defense import meco_supervisor as supervisor_api

    supervisor = object.__new__(supervisor_api.MecoSupervisor)
    supervisor.diagnostic_monitor = _journal(tmp_path / "monitor")
    supervisor.diagnostic_monitor_config = {"cadence_cycles": 3}
    supervisor.completed_cycles = 4
    supervisor.chunk_updates = 10
    supervisor.track_dirs = {track: tmp_path / track for track in MECO_TRACKS}
    for track_dir in supervisor.track_dirs.values():
        track_dir.mkdir()
        (track_dir / "latest.pt").write_bytes(b"cycle-four")
    supervisor.evaluation_output = tmp_path
    supervisor.source_hash = "s" * 64
    supervisor.configuration_hash = "c" * 64
    supervisor.sources = {"source": "fixture"}
    supervisor.last_monitor_error = None
    monkeypatch.setattr(supervisor_api, "source_identity", lambda: supervisor.sources)
    monkeypatch.setattr(supervisor_api, "_checkpoint_info", lambda _path: {"update": 40})
    supervisor._reconcile_completed_diagnostic_frontier()
    summary = supervisor.diagnostic_monitor.summary()
    assert summary["missed_frontiers"][0]["cycle"] == 3
    assert summary["pending_frontier"] is None
    assert summary["frontier_cycles"] == [3]


def test_monitor_restart_turns_running_attempt_into_retryable_failure(tmp_path):
    root = tmp_path / "monitor"
    journal = _journal(root)
    frontier = _frontier(tmp_path, 1)
    journal.enqueue(frontier)
    journal.begin_attempt(tmp_path / "attempt-1.json")
    resumed = _journal(root, resume=True)
    assert resumed.pending["status"] == "failed"
    resumed.begin_attempt(tmp_path / "attempt-2.json")
    resumed.verify_attempt(_valid_result(tmp_path, frontier))
    resumed.complete()
    assert resumed.summary()["completed_count"] == 1
    assert resumed.summary()["qualification_eligible"] is False


def test_monitor_restart_adopts_verified_report_and_advances_cursor(tmp_path):
    root = tmp_path / "monitor"
    journal = _journal(root)
    frontier = _frontier(tmp_path, 1)
    journal.enqueue(frontier)
    journal.begin_attempt(tmp_path / "attempt-1.json")
    journal.verify_attempt(_valid_result(tmp_path, frontier))

    resumed = _journal(root, resume=True)
    assert resumed.summary()["completed_count"] == 1
    assert resumed.pending is None


@pytest.mark.parametrize("target", ["journal", "cursor", "snapshot"])
def test_monitor_tamper_is_rejected(tmp_path, target):
    root = tmp_path / "monitor"
    journal = _journal(root)
    frontier = _frontier(tmp_path, 1)
    journal.enqueue(frontier)
    if target == "journal":
        path = journal.journal_path
        path.write_text(path.read_text().replace("frontier_queued", "altered"))
    elif target == "cursor":
        path = journal.cursor_path
        path.write_text(path.read_text().replace("monitor_sha256", "changed_sha256"))
    else:
        Path(frontier["checkpoint_snapshots"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        _journal(root, resume=True)


def test_dispatch_uses_frozen_monitor_seeds_and_protocol_binding(tmp_path):
    monitor_case = {"index": 4, "scenario_id": "pickup_simple",
        "category": "single", "seeds": [100, 101], "split": "diagnostic",
        "protocol_id": "frozen-protocol", "protocol_sha256": "p" * 64,
        "qualification_case": None}
    checkpoints = {track: tmp_path / f"{track}.pt" for track in MECO_TRACKS}
    for path in checkpoints.values():
        path.write_text("stub")
    command = _evaluation_command(checkpoints=checkpoints,
        scenario=monitor_case["scenario_id"], categories=(monitor_case["category"],),
        episodes=2, seed=100, device="cuda:1", output=tmp_path / "eval.json",
        split="diagnostic", episode_seeds=monitor_case["seeds"],
        schedule_case_binding=monitor_case)
    assert command[command.index("--episode-seeds") + 1] == "100,101"
    binding = json.loads(command[command.index("--schedule-case-binding-json") + 1])
    assert binding == monitor_case
    assert binding["protocol_id"] == "frozen-protocol"
    assert binding["protocol_sha256"] == "p" * 64


def test_evaluator_frontier_binding_checks_exact_checkpoint_bytes(tmp_path):
    checkpoints = {}
    checkpoint_hashes = {}
    for track in MECO_TRACKS:
        path = tmp_path / f"{track}.pt"
        path.write_bytes(track.encode())
        checkpoints[track] = path
        checkpoint_hashes[track] = hashlib.sha256(path.read_bytes()).hexdigest()
    case = {"index": 0, "scenario_id": "pickup_simple", "category": "single",
        "seeds": [100, 101], "split": "diagnostic", "protocol_id": "protocol",
        "protocol_sha256": "a" * 64, "qualification_case": None}
    binding = {"schema_version": 1, "frontier_id": "frontier-1", "cycle": 1,
        "update": 10, "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": "b" * 64, "configuration_sha256": "c" * 64,
        "protocol_id": "protocol", "protocol_sha256": "a" * 64}
    assert validate_diagnostic_frontier_binding(binding,
        schedule_case_binding=case, split="diagnostic",
        checkpoints=checkpoints) == binding
    bad = copy.deepcopy(binding)
    bad["checkpoint_sha256"][MECO_TRACKS[0]] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint snapshots changed"):
        validate_diagnostic_frontier_binding(bad, schedule_case_binding=case,
            split="diagnostic", checkpoints=checkpoints)


def test_supervisor_monitor_dispatch_uses_only_eval_device_and_defers_its_lane(
        tmp_path, monkeypatch):
    from frc_defense import meco_supervisor as supervisor_api

    monitor = _journal(tmp_path / "monitor")
    frontier = _frontier(tmp_path, 4)
    monitor.enqueue(frontier)
    supervisor = object.__new__(supervisor_api.MecoSupervisor)
    supervisor.evaluator = None
    supervisor.request_stop = False
    supervisor.evaluation_device = "cuda:1"
    supervisor.evaluation_schedule_journal = None
    supervisor.diagnostic_monitor = monitor
    supervisor.diagnostic_monitor_config = {"scenario_id": "pickup_simple",
        "category": "single", "seeds": [100, 101], "protocol_id": "protocol",
        "protocol_sha256": "p" * 64}
    supervisor.pending_evaluations = []
    supervisor.evaluation_output = tmp_path
    supervisor.evaluation_prediction_mode = "current_pose"
    supervisor.scenario = "pickup_simple"
    supervisor.evaluation_categories = ("single",)
    supervisor.evaluation_episodes = 1
    supervisor.evaluation_protocol_path = None
    supervisor.schedule_evaluation_protocol_path = None
    supervisor.evaluation_case = None
    supervisor.evaluation_split = "diagnostic"
    supervisor.evaluator_job = None
    supervisor.evaluator_log = None
    supervisor.last_evaluation = None
    supervisor.last_monitor_error = None
    supervisor._publish = lambda **_kwargs: None
    process = SimpleNamespace(pid=777, poll=lambda: None)
    launched = {}

    def fake_popen(command, **_kwargs):
        launched["command"] = command
        return process

    monkeypatch.setattr(supervisor_api.subprocess, "Popen", fake_popen)
    supervisor._start_next_evaluation()
    command = launched["command"]
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--episode-seeds") + 1] == "100,101"
    binding = json.loads(command[command.index("--schedule-case-binding-json") + 1])
    assert binding["protocol_id"] == "protocol"
    assert binding["protocol_sha256"] == "p" * 64
    assert supervisor.evaluator_job["diagnostic_monitor"] is True
    frontier_binding = json.loads(command[command.index(
        "--diagnostic-frontier-binding-json") + 1])
    assert frontier_binding["source_sha256"] == frontier["source_sha256"]
    assert frontier_binding["configuration_sha256"] == frontier["configuration_sha256"]
    assert frontier_binding["protocol_sha256"] == frontier["protocol_sha256"]
    assert frontier_binding["checkpoint_sha256"] == {
        row["track"]: row["sha256"] for row in frontier["checkpoint_snapshots"]}
    assert {track: hashlib.sha256(path.read_bytes()).hexdigest()
        for track, path in supervisor.evaluator_job["checkpoints"].items()} == {
            row["track"]: row["sha256"] for row in frontier["checkpoint_snapshots"]}

    supervisor.devices = ("cuda:0", "cuda:1")
    supervisor.pending_evaluations = []
    supervisor.configs = {track: SimpleNamespace(device=("cuda:1" if i == 0 else "cuda:0"))
        for i, track in enumerate(MECO_TRACKS)}
    supervisor.run_ids = {track: track for track in MECO_TRACKS}
    supervisor.worker_pids = {}
    supervisor.completed_updates = {track: 0 for track in MECO_TRACKS}
    supervisor.chunk_updates = 10
    supervisor.track_dirs = {track: tmp_path / track for track in MECO_TRACKS}
    assert supervisor._training_device_deferred_for_evaluation("cuda:1") is True
    assert supervisor._training_device_deferred_for_evaluation("cuda:0") is False
    assert supervisor._lane_payload(MECO_TRACKS[0], {MECO_TRACKS[0]: "queued"})[
        "status"] == "deferred_for_evaluation"
    assert supervisor._lane_payload(MECO_TRACKS[1], {MECO_TRACKS[1]: "queued"})[
        "status"] == "queued"


def test_monitor_only_campaign_can_resume_without_finite_schedule(campaign, monkeypatch):
    from frc_defense import meco_supervisor as supervisor_api

    tmp_path, config = campaign
    protocol_path = config["evaluation_protocol"]
    protocol = json.loads(Path(protocol_path).read_text())
    digest = hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest()
    config = copy.deepcopy(config)
    config["diagnostic_monitor"] = _monitor(protocol, digest)
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    campaign_root = Path(manifest["campaign"]["root"])
    evaluation_root = Path(manifest["campaign"]["evaluation_output"])
    campaign_root.mkdir(parents=True)
    evaluation_root.mkdir(parents=True)
    identity = {"status": "present", "module": "autodrive_fuel_physics_hip",
        "path": "/native/fuel.so", "sha256": "a" * 64,
        "build_fingerprint": "b" * 64, "device": 1, "inode": 2,
        "size": 3, "mtime_ns": 4}
    monkeypatch.setattr(supervisor_api, "source_identity", lambda: {"source": "fixture"})
    monkeypatch.setattr(supervisor_api, "_validate_runtime_source_binding",
        lambda *_args, **_kwargs: "c" * 64)
    monkeypatch.setattr("frc_defense.meco_device.require_indexed_cuda_device",
        lambda *_args, **_kwargs: None)
    monkeypatch.setattr("frc_defense.fuel_physics_hip.prepare_native_fuel_extension",
        lambda _device: identity)
    monkeypatch.setattr("frc_defense.fuel_physics_hip.native_extension_identity",
        lambda: identity)
    cfg = manifest["tracks"][MECO_TRACKS[0]]["config"]
    supervisor = supervisor_api.MecoSupervisor(output=str(campaign_root),
        evaluation_output=str(evaluation_root),
        devices=tuple(manifest["devices"]["training_pool"]),
        scenario=cfg["scenario_id"], category=cfg["category"], seed=cfg["seed"],
        chunk_updates=cfg["updates"], rollout_steps=cfg["rollout_steps"],
        epochs=cfg["epochs"], action_ticks=cfg["action_ticks"],
        learning_rate=cfg["learning_rate"], gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"], clip_coef=cfg["clip_coef"],
        entropy_coef=cfg["entropy_coef"], value_coef=cfg["value_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        route_cost_weight=cfg["route_cost_points_per_second"],
        candidate_k=cfg["candidate_k"], evaluation_categories=("single",),
        evaluation_episodes=1, evaluation_seed=101,
        evaluation_device=manifest["devices"]["evaluation"], cycles=0,
        stop_file=None, p_mode=cfg["p_mode"], p_cost_scale=cfg["p_cost_scale"],
        p_points_per_grid_edge=cfg["p_points_per_grid_edge"],
        prediction_mode=cfg["prediction_mode"],
        qualification_protocol=manifest["qualification_protocol"]["path"],
        evaluation_split="diagnostic", campaign_manifest=manifest,
        resume_campaign=True)
    assert supervisor.evaluation_schedule is None
    assert supervisor.diagnostic_monitor_config["enabled"] is True


def test_monitor_failure_and_stop_file_do_not_fail_or_launch_training(tmp_path, monkeypatch):
    from frc_defense import meco_supervisor as supervisor_api

    supervisor = object.__new__(supervisor_api.MecoSupervisor)
    supervisor.diagnostic_monitor = SimpleNamespace()
    supervisor.diagnostic_monitor_config = {"cadence_cycles": 1}
    supervisor.evaluation_schedule = None
    supervisor.pending_evaluations = []
    supervisor.chunk_updates = 10
    supervisor._enqueue_diagnostic_frontier = lambda _cycle: (_ for _ in ()).throw(
        RuntimeError("mock journal failure"))
    supervisor.diagnostic_monitor.record_missed_frontier = lambda **_kwargs: None
    supervisor._queue_evaluation(1)
    assert supervisor.last_monitor_error == "RuntimeError: mock journal failure"
    assert supervisor.pending_evaluations == []

    supervisor.request_stop = True
    supervisor.evaluator = None
    supervisor.evaluation_device = "cuda:1"
    monkeypatch.setattr(supervisor_api.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("stop intent must prevent monitor launch"))
    supervisor._start_next_evaluation()


@pytest.mark.parametrize("failure_stage", ["command", "cursor"])
def test_launch_stage_failure_fails_attempt_and_releases_eval_lane(
        tmp_path, monkeypatch, failure_stage):
    from frc_defense import meco_supervisor as supervisor_api

    monitor = _journal(tmp_path / "monitor")
    monitor.enqueue(_frontier(tmp_path, 1))
    supervisor = object.__new__(supervisor_api.MecoSupervisor)
    supervisor.evaluator = None
    supervisor.request_stop = False
    supervisor.evaluation_device = "cuda:1"
    supervisor.evaluation_schedule_journal = None
    supervisor.diagnostic_monitor = monitor
    supervisor.diagnostic_monitor_config = {"scenario_id": "pickup_simple",
        "category": "single", "seeds": [100, 101], "protocol_id": "protocol",
        "protocol_sha256": "p" * 64}
    supervisor.pending_evaluations = []
    supervisor.evaluation_output = tmp_path
    supervisor.evaluation_prediction_mode = "current_pose"
    supervisor.scenario = "pickup_simple"
    supervisor.evaluation_categories = ("single",)
    supervisor.evaluation_episodes = 1
    supervisor.evaluation_protocol_path = None
    supervisor.schedule_evaluation_protocol_path = None
    supervisor.evaluation_case = None
    supervisor.evaluation_split = "diagnostic"
    supervisor.evaluator_job = None
    supervisor.evaluator_log = None
    supervisor.last_evaluation = None
    supervisor.last_monitor_error = None
    if failure_stage == "command":
        monkeypatch.setattr(supervisor_api, "_evaluation_command",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected argument error")))
        expected_error = "RuntimeError: injected argument error"
    else:
        write_cursor = monitor._write_cursor
        fail_once = {"pending": True}

        def injected_cursor_failure():
            if fail_once["pending"]:
                fail_once["pending"] = False
                raise OSError("injected cursor persistence error")
            write_cursor()

        monkeypatch.setattr(monitor, "_write_cursor", injected_cursor_failure)
        expected_error = "OSError: injected cursor persistence error"

    supervisor._start_next_evaluation()

    assert supervisor.diagnostic_monitor is None
    assert supervisor.last_monitor_error == expected_error
    assert monitor.pending["status"] == "failed"
    assert supervisor._training_device_deferred_for_evaluation("cuda:1") is False


def test_nonzero_monitor_evaluator_exit_releases_eval_lane(tmp_path):
    from frc_defense import meco_supervisor as supervisor_api

    monitor = _journal(tmp_path / "monitor")
    frontier = _frontier(tmp_path, 1)
    monitor.enqueue(frontier)
    monitor.begin_attempt(tmp_path / "attempt.json")
    supervisor = object.__new__(supervisor_api.MecoSupervisor)
    supervisor.evaluator = SimpleNamespace(poll=lambda: 9)
    supervisor.evaluator_log = io.BytesIO()
    supervisor.evaluator_job = {"diagnostic_monitor": True, "cycle": 1,
        "output": tmp_path / "missing-report.json"}
    supervisor.evaluation_schedule_journal = None
    supervisor.diagnostic_monitor = monitor
    supervisor.diagnostic_monitor_config = {"enabled": True}
    supervisor.evaluation_device = "cuda:1"
    supervisor.pending_evaluations = []
    supervisor.last_evaluation = None
    supervisor.last_monitor_error = None
    supervisor.request_stop = False
    supervisor._observe_stop = lambda: None
    supervisor._publish = lambda **_kwargs: None

    supervisor._poll_evaluator(1, {})

    assert supervisor.evaluator is None
    assert supervisor.diagnostic_monitor is None
    assert monitor.pending["status"] == "failed"
    assert supervisor.last_evaluation["status"] == "failed"
    assert supervisor._training_device_deferred_for_evaluation("cuda:1") is False
