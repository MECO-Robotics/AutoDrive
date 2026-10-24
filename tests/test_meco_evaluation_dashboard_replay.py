"""Dashboard replay contracts for the in-loop MECO capture path; no simulation."""
import gzip
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

import pytest

from frc_defense.dashboard import create_handler
from frc_defense.dashboard_simulation import (
    replay_code_state, scenario_playback_view, write_dashboard_replay_bundle,
    resolve_dashboard_replay_generation,
)
from frc_defense import dashboard_simulation
from frc_defense.meco_ablation_matrix import run_matrix
import torch
from frc_defense.meco_track_training import (
    _capture_meco_replay_frame, _evaluation_record_digest,
    _validate_meco_replay_alignment, _validate_replay_seed_selection,
)
from scripts import run_meco_ablation_matrix as matrix_cli
from frc_defense.meco_tracks import MECO_TRACKS


def _row():
    return {"scenario_id": "pickup_simple", "category": "single", "seed": 91,
        "policy": "C1", "split": "validation", "elapsed_ticks": 24,
        "metrics": {"success": 1, "useful_throughput": 2.},
        "meco_metric_ledger": {"safety_metrics": {"safety_robot_contact_rate": 0.}},
        "execution_failures": [], "route_trace": [{"candidate_id": "fuel:4"}],
        "executed_planner_route_trace": [
            {"start_tick": 0, "end_tick": 12, "selected_action": 0,
             "executed_action_trace": [0]},
            {"start_tick": 12, "end_tick": 24, "selected_action": 4,
             "executed_action_trace": [4]}]}


def _frames():
    return [{"tick": 0, "meco_policy_decision": None},
        {"tick": 12, "meco_policy_decision": {"policy": "C1", "kind": "policy_decision",
            "macro_start_tick": 0, "macro_end_tick": 12, "action": 0,
            "executed_actions": [0]}},
        {"tick": 24, "meco_policy_decision": {"policy": "C1", "kind": "policy_decision",
            "macro_start_tick": 12, "macro_end_tick": 24, "action": 4,
            "executed_actions": [4]}}]


def test_replay_alignment_rejects_seed_record_tick_and_policy_divergence():
    record = _row()
    frames = _frames()
    _validate_meco_replay_alignment(record, frames, final_tick=24)
    assert len(_evaluation_record_digest(record)) == 64
    with pytest.raises(ValueError, match="terminal frame"):
        _validate_meco_replay_alignment(record, frames, final_tick=23)
    bad = _frames()
    bad[1]["meco_policy_decision"]["policy"] = "A1"
    with pytest.raises(ValueError, match="scored policy decision"):
        _validate_meco_replay_alignment(record, bad, final_tick=24)
    bad = _frames()
    bad[2]["tick"] = 12
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_meco_replay_alignment(record, bad, final_tick=24)


def test_replay_seed_capture_is_explicit_and_subset_bounded(tmp_path):
    assert _validate_replay_seed_selection(None, None, (90, 91)) == set()
    assert _validate_replay_seed_selection(tmp_path, (91,), (90, 91)) == {91}
    with pytest.raises(ValueError, match="supplied together"):
        _validate_replay_seed_selection(tmp_path, None, (90, 91))
    with pytest.raises(ValueError, match="subset"):
        _validate_replay_seed_selection(tmp_path, (92,), (90, 91))
    with pytest.raises(ValueError, match="subset"):
        _validate_replay_seed_selection(tmp_path, (91, 91), (90, 91))


def test_frame_capture_reads_renderer_fields_without_advancing_sim_state():
    env = SimpleNamespace(
        n=1, piece_active=torch.tensor([[True, False]]),
        piece_pos=torch.tensor([[[1., 2.], [3., 4.]]]),
        piece_owner=torch.tensor([[0, -1]]),
        _current_fuel_visibility=torch.ones((1, 6, 2), dtype=torch.bool),
        track_mask=torch.ones((1, 6, 2), dtype=torch.bool),
        track_pos=torch.arange(24, dtype=torch.float32).reshape(1, 6, 2, 2),
        track_age=torch.zeros((1, 6, 2)),
        opponent_valid=torch.zeros((1, 6), dtype=torch.bool),
        opponent_pose=torch.zeros((1, 6, 3)), opponent_age=torch.zeros((1, 6)),
        planner=SimpleNamespace(last_lengths=torch.zeros(6, dtype=torch.long),
            last_path=torch.zeros((6, 3, 2)), last_goal=torch.zeros((6, 2))),
        sim=SimpleNamespace(pose=torch.zeros((1, 6, 3)),
            length=torch.ones((1, 6)), width=torch.ones((1, 6))),
        team_ids=torch.tensor([0, 0, 0, 1, 1, 1]), control_modes=("nn",) * 6,
        robot_roles=("offense",) * 6, last_actions=torch.full((1, 6), 7),
        perception_fov_degrees=90., perception_range=8.,
        hub_centers=torch.zeros((2, 2)), hub_active=torch.ones((1, 2), dtype=torch.bool),
        fuel_score_count=torch.zeros((1, 2)), fuel_acquisition_count=torch.zeros((1, 6)),
        match_elapsed=torch.zeros(1), match_remaining=torch.ones(1),
        _last_audit_targets=torch.zeros((1, 6, 2)),
        _last_audit_fuel_targets=torch.zeros((1, 6, 2)),
        _target_collecting=torch.zeros((1, 6), dtype=torch.bool),
        _last_audit_cluster_count=torch.zeros((1, 6), dtype=torch.long))
    before = env.sim.pose.clone()
    frame = _capture_meco_replay_frame(env, 12, {"policy": "C1"})
    assert frame["tick"] == 12
    assert frame["robots"] == before[0].tolist()
    assert frame["fuel_pieces"] == [[1., 2., 0.]]
    assert frame["robot_detected_fuel"][0] == [[0., 1.], [2., 3.]]
    assert frame["meco_policy_decision"] == {"policy": "C1"}
    assert torch.equal(env.sim.pose, before)


def test_dashboard_bundle_contains_compact_decisions_and_staleness_metadata(tmp_path):
    record = {"simulation_id": "scenario-meco-contract", "task": "3v3",
        "seed": 91, "scenario_seed": 91, "behavior_mode": "match",
        "scenario_id": "pickup_simple", "category": "single", "policy": "C1",
        "split": "validation", "evaluation_id": "eval-abc", "condition_id": "cond-def",
        "evaluation_record_digest": "a" * 64, "checkpoint_sha256": "b" * 64,
        "checkpoint_source_outdated": True, "replay_source_sha256": "c" * 64,
        "replay_provenance": {"evaluation_id": "eval-abc", "condition_id": "cond-def"},
        "code_state": replay_code_state(), "simulation_ticks": 24,
        "simulated_seconds": .48, "field": {"length": 16, "width": 8},
        "robot_types": ["dumper"] * 6,
        "scenarios": [{"id": "pickup_simple", "label": "MECO C1 seed 91",
            "frames": _frames()}], "frames": _frames()}
    result_path = tmp_path / (record["simulation_id"] + ".result.json")
    write_dashboard_replay_bundle(result_path, record)
    generation, _manifest = resolve_dashboard_replay_generation(tmp_path,
        record["simulation_id"])
    assert (generation / "result.json").is_file()
    assert (generation / "result.meta.json").is_file()
    assert (generation / "result.json.gz").is_file()
    assert (generation / "result.view.json.gz").is_file()
    metadata = json.loads((generation / "result.meta.json").read_text())
    assert metadata["condition_id"] == "cond-def"
    assert metadata["checkpoint_source_outdated"] is True
    compact = json.loads(gzip.decompress((generation / "result.view.json.gz").read_bytes()))
    assert compact["scenarios"][0]["frames"][1]["meco_policy_decision"]["action"] == 0
    assert compact["evaluation_record_digest"] == "a" * 64
    assert compact["replay_provenance"]["condition_id"] == "cond-def"
    assert scenario_playback_view(record)["scenarios"][0]["frames"] == compact["scenarios"][0]["frames"]
    generation, manifest = resolve_dashboard_replay_generation(tmp_path,
        record["simulation_id"])
    assert manifest["generation_id"] == generation.name
    assert json.loads((generation / "result.json").read_text()) == record


def test_replay_listing_surfaces_source_and_checkpoint_staleness(tmp_path):
    progress = tmp_path / ".scenario-progress"
    progress.mkdir()
    current_state = replay_code_state()
    record = {"simulation_id": "scenario-meco-stale", "task": "3v3", "seed": 91,
        "behavior_mode": "match", "robot_types": ["dumper"] * 6,
        "code_state": {**current_state, "commit": "old-evaluation-commit"},
        "checkpoint_source_outdated": True,
        "replay_source_sha256": "c" * 64, "checkpoint_sha256": "b" * 64,
        "evaluation_record_digest": "a" * 64, "condition_id": "cond",
        "simulation_ticks": 0, "simulated_seconds": 0,
        "scenarios": [{"id": "pickup_simple", "label": "C1 validation seed 91", "frames": []}],
        "frames": []}
    write_dashboard_replay_bundle(progress / "scenario-meco-stale.result.json", record)
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        listing = json.loads(urlopen(
            f"http://127.0.0.1:{server.server_port}/api/scenario-replays").read())
        replay = listing["replays"][0]
        assert replay["checkpoint_source_outdated"] is True
        assert replay["source_outdated"] is True
        assert replay["outdated"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("failure_stage", ["asset:result.json", "manifest",
    "generation", "generation_parent_synced", "ready_temp_written", "before_pointer"])
def test_replay_publication_failure_keeps_previous_generation_and_retry_succeeds(
        tmp_path, monkeypatch, failure_stage):
    progress = tmp_path / ".scenario-progress"
    progress.mkdir()
    record = {"simulation_id": "scenario-atomic", "seed": 1,
        "simulation_ticks": 10,
        "scenarios": [{"label": "first", "frames": []}]}
    write_dashboard_replay_bundle(progress / "scenario-atomic.result.json", record)
    old_pointer = json.loads((progress / "scenario-atomic.ready.json").read_text())
    changed = {**record, "seed": 2, "simulation_ticks": 99}
    def fail_before_pointer(stage):
        if stage == failure_stage:
            raise OSError("injected crash before publication")
    monkeypatch.setattr(dashboard_simulation, "_replay_publication_checkpoint",
                        fail_before_pointer)
    with pytest.raises(OSError, match="injected crash"):
        write_dashboard_replay_bundle(progress / "scenario-atomic.result.json", changed)
    assert json.loads((progress / "scenario-atomic.ready.json").read_text()) == old_pointer
    old_generation, old_manifest = resolve_dashboard_replay_generation(progress, "scenario-atomic")
    assert json.loads((old_generation / "result.json").read_text())["seed"] == 1
    assert old_manifest["ready_status"]["tick"] == 10
    generations = progress / "scenario-atomic.generations"
    assert list(generations.iterdir()) == [old_generation]
    assert not list(progress.glob(".*.tmp"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status = json.loads(urlopen(
            f"http://127.0.0.1:{server.server_port}/api/zone-playback-progress"
            "?simulation_id=scenario-atomic").read())
        assert status["tick"] == 10
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    monkeypatch.setattr(dashboard_simulation, "_replay_publication_checkpoint",
                        lambda _stage: None)
    write_dashboard_replay_bundle(progress / "scenario-atomic.result.json", changed)
    new_generation, _ = resolve_dashboard_replay_generation(progress, "scenario-atomic")
    assert new_generation != old_generation
    new_generation, new_manifest = resolve_dashboard_replay_generation(progress, "scenario-atomic")
    assert json.loads((new_generation / "result.json").read_text())["seed"] == 2
    assert new_manifest["ready_status"]["tick"] == 99


def test_replay_discovery_and_load_reject_corrupted_generation(tmp_path):
    progress = tmp_path / ".scenario-progress"
    progress.mkdir()
    simulation_id = "scenario-corrupt"
    write_dashboard_replay_bundle(progress / f"{simulation_id}.result.json",
        {"simulation_id": simulation_id, "scenarios": [{"frames": []}]})
    generation, _ = resolve_dashboard_replay_generation(progress, simulation_id)
    with (generation / "result.view.json.gz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="verification failed"):
        resolve_dashboard_replay_generation(progress, simulation_id)
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        listing = json.loads(urlopen(
            f"http://127.0.0.1:{server.server_port}/api/scenario-replays").read())
        assert listing["replays"] == []
        with pytest.raises(HTTPError) as error:
            urlopen(f"http://127.0.0.1:{server.server_port}/api/zone-playback-result"
                    f"?simulation_id={simulation_id}")
        assert error.value.code == 500
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_replay_alignment_matches_selected_and_executed_actions_including_remask():
    record = _row()
    record["executed_planner_route_trace"] = [{"start_tick": 0, "end_tick": 24,
        "selected_action": 0, "executed_action_trace": [0, 1]}]
    frames = [{"tick": 0, "meco_policy_decision": None},
        {"tick": 24, "meco_policy_decision": {"policy": "C1",
         "kind": "policy_decision", "macro_start_tick": 0, "macro_end_tick": 24,
         "action": 0, "executed_actions": [0, 1]}}]
    _validate_meco_replay_alignment(record, frames, final_tick=24)
    frames[1]["meco_policy_decision"]["executed_actions"] = [0, 0]
    with pytest.raises(ValueError, match="actions differ"):
        _validate_meco_replay_alignment(record, frames, final_tick=24)


def test_replay_alignment_compares_full_no_route_macro_trace_and_allows_remask():
    record = _row()
    record.update({"policy": "C1", "elapsed_ticks": 3,
        "executed_planner_route_trace": [], "executed_action_trace": [0, 2, 1],
        "selected_action_trace": [0]})
    frames = [{"tick": 0, "meco_policy_decision": None},
        {"tick": 3, "meco_policy_decision": {"policy": "C1",
         "kind": "policy_decision", "macro_start_tick": 0, "macro_end_tick": 3,
         "action": 0, "executed_actions": [0, 2, 1]}}]
    _validate_meco_replay_alignment(record, frames, final_tick=3)
    # Preserve the selected action and final executed action while corrupting
    # only the middle tick: endpoint-only comparisons would miss this change.
    frames[1]["meco_policy_decision"]["executed_actions"] = [0, 9, 1]
    with pytest.raises(ValueError, match="macro slice"):
        _validate_meco_replay_alignment(record, frames, final_tick=3)


def test_matrix_capture_requires_explicit_seeds_before_gpu_setup(monkeypatch, tmp_path):
    monkeypatch.setattr("frc_defense.fuel_physics_hip.prepare_native_fuel_extension",
        lambda device: pytest.fail("GPU setup occurred before capture validation"))
    with pytest.raises(ValueError, match="supplied together"):
        run_matrix(checkpoints={track: f"{track}.pt" for track in MECO_TRACKS},
            scenario_id="pickup_simple", categories=("single",), seeds=(91,),
            device="cuda:0", output_dir=tmp_path,
            replay_output_dir=tmp_path / ".scenario-progress")


def test_matrix_cli_forwards_explicit_capture_selection(monkeypatch, tmp_path, capsys):
    observed = {}
    def fake_run_matrix(**kwargs):
        observed.update(kwargs)
        return {"matrix_id": "matrix", "conditions": [], "contrasts": []}
    monkeypatch.setattr(matrix_cli, "run_matrix", fake_run_matrix)
    checkpoint_args = []
    for track in MECO_TRACKS:
        checkpoint_args.extend([f"--{track.lower()}-checkpoint", f"{track}.pt"])
    assert matrix_cli.main(["--scenario", "pickup_simple", "--categories", "single",
        "--episodes", "3", "--seed", "90", "--replay-seeds", "91",
        "--replay-output-dir", str(tmp_path / ".scenario-progress"),
        *checkpoint_args]) == 0
    assert observed["replay_seeds"] == (91,)
    assert observed["replay_output_dir"] == tmp_path / ".scenario-progress"
    capsys.readouterr()
