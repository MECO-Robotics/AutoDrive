from pathlib import Path
import json

import pytest

from frc_defense import meco_supervisor as module
from frc_defense.meco_supervisor import MecoSupervisor
from frc_defense.meco_track_training import MecoTrainConfig
from frc_defense.meco_tracks import MECO_TRACKS


class FakeProcess:
    def __init__(self, track, ticks, result, finish):
        self.track = track
        self.ticks = ticks
        self.result = result
        self.finish = finish
        self.pid = 1000 + MECO_TRACKS.index(track)
        self.returncode = None

    def poll(self):
        self.ticks -= 1
        if self.ticks <= 0 and self.returncode is None:
            self.returncode = self.result
            self.finish(self.track, self.result)
        return self.returncode


def make_supervisor(tmp_path, monkeypatch, *, plans=None, completed=None,
                    stop_on=None):
    supervisor = MecoSupervisor.__new__(MecoSupervisor)
    supervisor.output = tmp_path
    for track in MECO_TRACKS:
        (tmp_path / track).mkdir()
    supervisor.devices = ("cuda:0", "cuda:1")
    supervisor.evaluation_device = "cuda:1"
    supervisor.evaluator = None
    supervisor.evaluator_log = None
    supervisor.evaluator_job = None
    supervisor.pending_evaluations = []
    supervisor.chunk_updates = 10
    supervisor.stop_file = tmp_path / "STOP"
    supervisor.supervisor_id = "dispatch-test"
    supervisor.protocol_id = "protocol-test"
    supervisor.source_hash = "source-hash"
    supervisor.sources = {"source": "same"}
    supervisor.run_ids = {track: track for track in MECO_TRACKS}
    supervisor.configs = {track: MecoTrainConfig(track=track,
        device="cuda:0" if index % 2 == 0 else "cuda:1")
        for index, track in enumerate(MECO_TRACKS)}
    supervisor.warm_start_paths = {}
    supervisor.warm_start_hashes = {}
    supervisor.qualification_protocol_path = None
    supervisor.worker_pids = {}
    supervisor.request_stop = False
    supervisor.completed_chunks = 0
    completed = dict(completed or {track: 0 for track in MECO_TRACKS})
    supervisor._completed_updates = lambda track: completed[track]
    supervisor._refresh_progress = lambda track: None
    supervisor._write_queued_manifests = lambda statuses: None
    supervisor._poll_evaluator = lambda cycle, statuses, **kwargs: None
    supervisor.published = []
    supervisor._publish = lambda **kwargs: supervisor.published.append(
        (kwargs["phase"], dict(kwargs["statuses"])))

    plans = plans or {track: (1, 0) for track in MECO_TRACKS}
    active_devices = set()
    active_tracks = set()
    started = []
    events = []
    process_by_track = {}

    def finish(track, result):
        active_devices.remove(supervisor.configs[track].device)
        active_tracks.remove(track)
        events.append(("finish", track, set(active_tracks)))
        if result == 0:
            completed[track] = 10

    def popen(command, **kwargs):
        track = command[command.index("--track") + 1]
        device = supervisor.configs[track].device
        assert device not in active_devices, f"two workers active on {device}"
        assert (supervisor.evaluator is None or
                device != supervisor.evaluation_device), "trainer overlapped evaluator GPU lease"
        active_devices.add(device)
        started.append(track)
        active_tracks.add(track)
        events.append(("start", track, set(active_tracks)))
        ticks, result = plans[track]
        process = FakeProcess(track, ticks, result, finish)
        process_by_track[track] = process
        return process

    monkeypatch.setattr(module, "source_identity", lambda: supervisor.sources)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module, "_train_command",
        lambda **kwargs: ["--track", kwargs["config"].track])
    if stop_on:
        original_popen = popen

        def stop_popen(command, **kwargs):
            process = original_popen(command, **kwargs)
            if process.track == stop_on:
                old_finish = process.finish

                def stop_after_finish(track, result):
                    old_finish(track, result)
                    supervisor.stop_file.touch()
                process.finish = stop_after_finish
            return process

        monkeypatch.setattr(module.subprocess, "Popen", stop_popen)
    return supervisor, completed, started, process_by_track, events


def test_dispatch_fills_freed_devices_without_waiting_for_other_tracks(
        tmp_path, monkeypatch):
    plans = {"A1": (1, 0), "A2": (3, 0), "A3": (4, 0),
             "C1": (5, 0), "C2": (1, 0)}
    supervisor, completed, started, _, events = make_supervisor(
        tmp_path, monkeypatch, plans=plans)
    statuses = {track: "queued" for track in MECO_TRACKS}
    supervisor._run_training_batch(list(MECO_TRACKS), 1, statuses, 10, {})
    assert started == list(MECO_TRACKS)
    a1_finish = events.index(next(event for event in events
                                  if event[:2] == ("finish", "A1")))
    a3_start = events.index(next(event for event in events
                                 if event[:2] == ("start", "A3")))
    a2_finish = events.index(next(event for event in events
                                  if event[:2] == ("finish", "A2")))
    a3_finish = events.index(next(event for event in events
                                  if event[:2] == ("finish", "A3")))
    c1_start = events.index(next(event for event in events
                                 if event[:2] == ("start", "C1")))
    c2_start = events.index(next(event for event in events
                                 if event[:2] == ("start", "C2")))
    assert a1_finish < a3_start < a2_finish
    assert c1_start < a3_finish < c2_start
    assert "C1" in events[c2_start][2]
    assert completed == {track: 10 for track in MECO_TRACKS}
    assert statuses == {track: "complete" for track in MECO_TRACKS}


def test_dispatch_stops_queue_after_first_worker_failure(tmp_path, monkeypatch):
    plans = {"A1": (1, 3), "A2": (6, 0), "A3": (1, 0),
             "C1": (1, 0), "C2": (1, 0)}
    supervisor, _completed, started, _, _events = make_supervisor(
        tmp_path, monkeypatch, plans=plans)
    statuses = {track: "queued" for track in MECO_TRACKS}
    with pytest.raises(RuntimeError, match="training workers failed"):
        supervisor._run_training_batch(list(MECO_TRACKS), 1, statuses, 10, {})
    assert started == ["A1", "A2"]
    assert (tmp_path / ".sibling-aborts" / "dispatch-test-cycle-000001.json").is_file()
    assert statuses["A3"] == "aborted_due_to_sibling_failure"
    assert json.loads((tmp_path / "A3" / "run.json").read_text())[
        "status"] == "aborted_due_to_sibling_failure"


def test_stop_prevents_dispatching_queued_tracks(tmp_path, monkeypatch):
    plans = {track: (1, 0) for track in MECO_TRACKS}
    supervisor, _completed, started, _, _events = make_supervisor(
        tmp_path, monkeypatch, plans=plans, stop_on="A1")
    statuses = {track: "queued" for track in MECO_TRACKS}
    supervisor._run_training_batch(list(MECO_TRACKS), 1, statuses, 10, {})
    assert started == ["A1", "A2"]
    assert statuses["A3"] == "stopped"
    assert statuses["C1"] == "stopped"
    assert json.loads((tmp_path / "A3" / "run.json").read_text())[
        "status"] == "stopped"


def test_dispatch_rejects_duplicate_queue_tracks(tmp_path, monkeypatch):
    supervisor, _completed, started, _, _events = make_supervisor(
        tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duplicate tracks"):
        supervisor._run_training_batch(["A1", "A1"], 1,
            {track: "queued" for track in MECO_TRACKS}, 10, {})
    assert started == []


def test_evaluation_device_lease_defers_only_matching_lanes_and_resumes(
        tmp_path, monkeypatch):
    supervisor, completed, started, _, events = make_supervisor(
        tmp_path, monkeypatch, plans={track: (1, 0) for track in MECO_TRACKS})
    supervisor.evaluator = object()
    eval_ticks = {"left": 3}

    def poll_eval(_cycle, _statuses, *, occupied_devices=()):
        if supervisor.evaluator is None:
            return
        assert supervisor.evaluation_device not in occupied_devices
        events.append(("evaluation_poll", None, set(occupied_devices)))
        eval_ticks["left"] -= 1
        if eval_ticks["left"] == 0:
            supervisor.evaluator = None
            events.append(("evaluation_released", None, set(occupied_devices)))

    supervisor._poll_evaluator = poll_eval
    statuses = {track: "queued" for track in MECO_TRACKS}
    supervisor._run_training_batch(list(MECO_TRACKS), 1, statuses, 10, {})

    assert all(value == 10 for value in completed.values())
    assert statuses == {track: "complete" for track in MECO_TRACKS}
    first_eval_gpu_lane = next(i for i, event in enumerate(events)
                               if event[:2] == ("start", "A2"))
    release_index = next(i for i, event in enumerate(events)
                         if event[0] == "evaluation_released")
    assert len([event for event in events if event[:2] == ("finish", "A1")]) == 1
    assert release_index < first_eval_gpu_lane
    assert any(i < release_index and event[:2] in {
            ("finish", "A1"), ("finish", "A3"), ("finish", "C2")}
        for i, event in enumerate(events))


def test_deferred_device_lane_is_visible_in_queue_payload(tmp_path, monkeypatch):
    supervisor, *_ = make_supervisor(tmp_path, monkeypatch)
    supervisor.evaluator = object()
    supervisor.worker_pids = {}
    lane = supervisor._lane_payload("A2", {track: "queued" for track in MECO_TRACKS})
    assert lane["status"] == "deferred_for_evaluation"
    assert lane["deferred_reason"] == "evaluation_device_lease"


def test_queued_evaluation_waits_for_its_device_lease(tmp_path, monkeypatch):
    supervisor, *_ = make_supervisor(tmp_path, monkeypatch)
    queued = {"job": "must-remain-queued"}
    supervisor.pending_evaluations = [queued]
    launched = []
    monkeypatch.setattr(module.subprocess, "Popen",
        lambda *args, **kwargs: launched.append(args))
    supervisor._start_next_evaluation(occupied_devices={"cuda:1"})
    assert supervisor.pending_evaluations == [queued]
    assert supervisor.evaluator is None
    assert launched == []


def test_target_barrier_and_restart_catchup_dispatch_only_lagging_tracks(
        tmp_path, monkeypatch):
    initial = {"A1": 10, "A2": 0, "A3": 10, "C1": 0, "C2": 10}
    plans = {track: (1, 0) for track in MECO_TRACKS}
    supervisor, completed, started, _, _events = make_supervisor(
        tmp_path, monkeypatch, plans=plans, completed=initial)
    statuses = {track: ("complete" if value == 10 else "queued")
                for track, value in initial.items()}
    supervisor._train_to_update(10, 2, statuses, list(MECO_TRACKS), {})
    assert started == ["A2", "C1"]
    assert all(value == 10 for value in completed.values())
    assert statuses == {track: "complete" for track in MECO_TRACKS}


def test_target_barrier_rejects_a_track_that_did_not_reach_frontier(
        tmp_path, monkeypatch):
    supervisor, completed, _started, _, _events = make_supervisor(
        tmp_path, monkeypatch)
    completed["C2"] = 9
    monkeypatch.setattr(supervisor, "_run_training_batch", lambda *args: None)
    with pytest.raises(RuntimeError, match="matched target"):
        supervisor._train_to_update(10, 1,
            {track: "queued" for track in MECO_TRACKS}, list(MECO_TRACKS), {})
