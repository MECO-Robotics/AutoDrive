"""Fair round-robin supervisor for continuous five-track MECO research runs.

The supervisor is opt-in and is not started as part of import or dashboard
startup. It runs at most one trainer process per configured device, gives each
learned track the same update chunk and immutable training settings, and runs a
paired all-track/control evaluation after every complete five-track cycle.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import torch

from .curriculum_training import source_identity
from .meco_candidates import CANDIDATE_BOUNDARY
from .meco_track_training import CHUNK_RESET_POLICY, MecoTrainConfig, ROOT
from .meco_task_scheduler import MecoTaskScheduler
from .meco_tracks import MECO_TRACKS, PREDICTION_MODES, prediction_contract


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checkpoint_info(path: Path):
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload


def _train_command(*, config: MecoTrainConfig, output: Path, run_id: str,
                   stop_file: Path, resume: Path | None, protocol_id: str,
                   task_assignment: dict, warm_start: Path | None = None,
                   qualification_protocol: Path | None = None):
    command = [sys.executable, "-m", "frc_defense.meco_track_training", "train",
        "--track", config.track, "--scenario", config.scenario_id,
        "--category", config.category, "--updates", str(config.updates),
        "--rollout-steps", str(config.rollout_steps), "--epochs", str(config.epochs),
        "--action-ticks", str(config.action_ticks), "--seed", str(config.seed),
        "--candidate-k", str(config.candidate_k),
        "--device", config.device, "--learning-rate", str(config.learning_rate),
        "--gamma", str(config.gamma), "--gae-lambda", str(config.gae_lambda),
        "--clip-coef", str(config.clip_coef), "--entropy-coef", str(config.entropy_coef),
        "--value-coef", str(config.value_coef), "--max-grad-norm", str(config.max_grad_norm),
        "--route-cost-points-per-second", str(config.route_cost_points_per_second),
        "--p-mode", config.p_mode,
        "--p-cost-scale", str(config.p_cost_scale),
        "--p-points-per-grid-edge", str(config.p_points_per_grid_edge),
        "--prediction-mode", config.prediction_mode,
        "--output", str(output), "--stop-file", str(stop_file), "--run-id", run_id,
        "--protocol-id", protocol_id]
    command.extend(("--task-assignment-json", json.dumps(
        task_assignment, sort_keys=True, allow_nan=False)))
    if resume is not None:
        command.extend(("--resume", str(resume)))
    if warm_start is not None:
        command.extend(("--warm-start", str(warm_start),
                        "--qualification-protocol", str(qualification_protocol)))
    elif qualification_protocol is not None:
        command.extend(("--qualification-protocol", str(qualification_protocol)))
    return command


def _evaluation_command(*, checkpoints, scenario, categories, episodes, seed,
                        device, output, prediction_mode=None):
    command = [sys.executable, "-m", "frc_defense.meco_track_training", "evaluate",
        "--scenario", scenario, "--categories", ",".join(categories),
        "--episodes", str(episodes), "--seed", str(seed), "--device", device,
        "--output", str(output)]
    if prediction_mode is not None:
        command.extend(("--prediction-mode", prediction_mode))
    for track in MECO_TRACKS:
        command.extend((f"--{track.lower()}-checkpoint", str(checkpoints[track])))
    return command


class MecoSupervisor:
    def __init__(self, *, output: str, evaluation_output: str, devices: tuple[str, ...],
                 scenario: str, category: str, seed: int, chunk_updates: int,
                 rollout_steps: int, epochs: int, action_ticks: int, learning_rate: float,
                 gamma: float, gae_lambda: float, clip_coef: float, entropy_coef: float,
                 value_coef: float, max_grad_norm: float, route_cost_weight: float,
                 candidate_k: int,
                 evaluation_categories: tuple[str, ...], evaluation_episodes: int,
                 evaluation_seed: int, evaluation_device: str | None,
                 cycles: int, stop_file: str | None,
                 stage_exposure_chunks: int = 10, p_mode: str = "learned",
                 p_cost_scale: float = 1.0, p_points_per_grid_edge: float = .05,
                 prediction_mode: str = "cv",
                 evaluation_prediction_mode: str | None = None,
                 warm_start_artifacts: dict[str, str] | None = None,
                 qualification_protocol: str | None = None):
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("devices must be a nonempty list without duplicates")
        if any(device.startswith("cuda") for device in devices) and not torch.cuda.is_available():
            raise ValueError("a configured ROCm/CUDA device is unavailable")
        if cycles < 0 or chunk_updates < 1 or evaluation_episodes < 1:
            raise ValueError("cycles must be nonnegative; chunk/evaluation counts positive")
        self.output = Path(output).expanduser().resolve()
        self.evaluation_output = Path(evaluation_output).expanduser().resolve()
        expected_checkpoint_root = (ROOT / "checkpoints" / "meco-five").resolve()
        expected_evaluation_root = (ROOT / "evaluations" / "meco-five").resolve()
        if (self.output != expected_checkpoint_root
                or self.evaluation_output != expected_evaluation_root):
            raise ValueError("supervisor roots must match the dashboard's pinned MECO roots")
        self.devices = devices
        self.warm_start_paths = {}
        self.warm_start_hashes = {}
        self.qualification_protocol_path = (Path(qualification_protocol).expanduser().resolve()
            if qualification_protocol else None)
        self.qualification_protocol_id = None
        self.qualification_protocol_sha256 = None
        self.qualification_protocol = None
        warm_start_artifacts = dict(warm_start_artifacts or {})
        if warm_start_artifacts and set(warm_start_artifacts) != set(MECO_TRACKS):
            raise ValueError("provide one warm-start artifact for every MECO track")
        if warm_start_artifacts and self.qualification_protocol_path is None:
            raise ValueError("per-track warm starts require --qualification-protocol")
        if self.qualification_protocol_path is not None:
            from .meco_supervised import _load_protocol
            (self.qualification_protocol, self.qualification_protocol_id,
             self.qualification_protocol_sha256, self.qualification_protocol_path) = _load_protocol(
                self.qualification_protocol_path)
            if self.qualification_protocol.get("master_seed") != seed:
                raise ValueError("supervisor seed must match the frozen qualification protocol master seed")
        self.scenario = scenario
        self.category = category
        self.seed = seed
        self.candidate_k = candidate_k
        self.p_mode = p_mode
        self.p_cost_scale = p_cost_scale
        self.p_points_per_grid_edge = p_points_per_grid_edge
        if prediction_mode not in PREDICTION_MODES:
            raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
        self.prediction_mode = prediction_mode
        if (evaluation_prediction_mode is not None and
                evaluation_prediction_mode not in PREDICTION_MODES):
            raise ValueError(f"evaluation_prediction_mode must be one of {PREDICTION_MODES}")
        self.evaluation_prediction_mode = evaluation_prediction_mode
        self.stage_exposure_chunks = stage_exposure_chunks
        self.chunk_updates = chunk_updates
        self.evaluation_categories = evaluation_categories
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_seed = evaluation_seed
        self.evaluation_device = evaluation_device or "cpu"
        self.cycles = cycles  # zero means continue until stopped
        self.stop_file = Path(stop_file).expanduser().resolve() if stop_file else self.output / "STOP"
        self.queue_path = self.output / "queue-status.json"
        self.status_path = self.output / "supervisor-status.json"
        self.request_stop = False
        self.evaluator = None
        self.evaluator_log = None
        self.evaluator_job = None
        self.pending_evaluations = []
        self.last_evaluation = None
        self.current_assignment = None
        self.run_ids = {}
        self.worker_pids = {}
        self.completed_updates = {track: 0 for track in MECO_TRACKS}
        self.configs = {}
        for index, track in enumerate(MECO_TRACKS):
            config = MecoTrainConfig(track=track, scenario_id=scenario, category=category,
                updates=chunk_updates, rollout_steps=rollout_steps, epochs=epochs,
                action_ticks=action_ticks, seed=seed, device=devices[index % len(devices)],
                learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
                clip_coef=clip_coef, entropy_coef=entropy_coef, value_coef=value_coef,
                max_grad_norm=max_grad_norm, route_cost_points_per_second=route_cost_weight,
                p_mode=p_mode, p_cost_scale=p_cost_scale,
                p_points_per_grid_edge=p_points_per_grid_edge,
                prediction_mode=prediction_mode,
                candidate_k=candidate_k,
                output=str(self.output / track))
            config.validate()
            self.configs[track] = config
        self.sources = source_identity()
        self.source_hash = _hash(self.sources)
        if warm_start_artifacts:
            from .meco_supervised import validate_warm_start_artifact
            for track in MECO_TRACKS:
                artifact = Path(warm_start_artifacts[track]).expanduser().resolve()
                validate_warm_start_artifact(track, artifact,
                    qualification_protocol=self.qualification_protocol_path,
                    expected_sources=self.sources,
                    expected_prediction_mode=self.prediction_mode)
                self.warm_start_paths[track] = artifact
                self.warm_start_hashes[track] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        self.hyperparameters = asdict(next(iter(self.configs.values())))
        self.hyperparameters.pop("track")
        self.hyperparameters.pop("device")
        self.hyperparameters.pop("output")
        self.hyperparameters.pop("updates")
        self.protocol_id = _hash({"source_identity_sha256": self.source_hash,
            "training_hyperparameters": self.hyperparameters,
            "scenario": scenario, "category": category, "seed": seed,
            "updates_per_track_per_cycle": chunk_updates,
            "devices": devices, "evaluation_categories": evaluation_categories,
            "evaluation_episodes": evaluation_episodes,
            "evaluation_seed": evaluation_seed,
            "evaluation_prediction_mode": evaluation_prediction_mode,
            "qualification_protocol_id": self.qualification_protocol_id,
            "qualification_protocol_sha256": self.qualification_protocol_sha256,
            "warm_start_artifact_sha256": dict(sorted(self.warm_start_hashes.items())),
            "candidate_boundary": CANDIDATE_BOUNDARY,
            "chunk_reset_policy": CHUNK_RESET_POLICY,
            "task_scheduler": {"version": 1,
                "stage_exposure_chunks": stage_exposure_chunks,
                "semantics": "DAG exposure progression; not promotion qualification"},
            "D0_control": "production deterministic slot-3 controller; native 20 ms cadence; outside candidate table"})

    def _lane_payload(self, track, statuses):
        return {"track": track, "run_id": self.run_ids[track],
                "run_directory": track, "device": self.configs[track].device,
                "status": statuses.get(track, "queued"),
                "pid": self.worker_pids.get(track),
                "completed_updates": self._heartbeat_updates(track),
                "chunk_updates": self.chunk_updates}

    def _heartbeat_updates(self, track):
        try:
            heartbeat = json.loads((self.output / track / "heartbeat.json").read_text())
            if (heartbeat.get("run_id") == self.run_ids[track]
                    and type(heartbeat.get("completed_updates")) is int):
                return heartbeat["completed_updates"]
        except (OSError, json.JSONDecodeError):
            pass
        return self._completed_updates(track)

    def _completed_updates(self, track):
        return self.completed_updates.get(track, 0)

    def _task_scheduler_status(self):
        if not hasattr(self, "task_scheduler"):
            return None
        state = self.task_scheduler.state
        return {key: state[key] for key in (
            "completed_cycle", "pending_assignment", "exposure_chunks",
            "category_exposures", "recovery_pressure")}

    def _refresh_progress(self, track):
        payload = _checkpoint_info(self.output / track / "latest.pt")
        if payload:
            self.completed_updates[track] = int(payload.get("update", 0))

    def _publish(self, *, cycle, phase, statuses, evaluation=None, error=None):
        now = _utc()
        queue = {"schema_version": 1, "updated_utc": now,
            "supervisor_id": self.supervisor_id, "protocol_id": self.protocol_id,
            "prediction_mode": self.prediction_mode,
            "prediction_contract": prediction_contract(self.prediction_mode),
            "source_identity_sha256": self.source_hash,
            "qualification_protocol_id": self.qualification_protocol_id,
            "qualification_protocol_sha256": self.qualification_protocol_sha256,
            "warm_start_artifact_sha256": dict(sorted(self.warm_start_hashes.items())),
            "configuration_sha256": self.configuration_hash,
            "cycle": cycle, "phase": phase,
            "candidate_boundary": CANDIDATE_BOUNDARY,
            "lanes": [self._lane_payload(track, statuses) for track in MECO_TRACKS],
            "task_assignment": self.current_assignment,
            "task_scheduler": self._task_scheduler_status(),
            "evaluation": self._evaluation_status()}
        _atomic_json(self.queue_path, queue)
        _atomic_json(self.status_path, {"schema_version": 1,
            "supervisor_id": self.supervisor_id, "protocol_id": self.protocol_id,
            "prediction_mode": self.prediction_mode,
            "prediction_contract": prediction_contract(self.prediction_mode),
            "source_identity_sha256": self.source_hash,
            "qualification_protocol_id": self.qualification_protocol_id,
            "qualification_protocol_sha256": self.qualification_protocol_sha256,
            "warm_start_artifact_sha256": dict(sorted(self.warm_start_hashes.items())),
            "configuration_sha256": self.configuration_hash,
            "heartbeat_utc": now, "status": phase, "cycle": cycle,
            "completed_cycles": self.completed_cycles,
            "completed_training_chunks": self.completed_chunks,
            "lanes": [self._lane_payload(track, statuses) for track in MECO_TRACKS],
            "task_assignment": self.current_assignment,
            "task_scheduler": self._task_scheduler_status(),
            "evaluation": evaluation, "error": error,
            "stop_requested": self.request_stop,
            "candidate_boundary": CANDIDATE_BOUNDARY})

    def _evaluation_status(self):
        status_path = (Path(str(self.evaluator_job["output"]) + ".status.json")
                       if self.evaluator_job else None)
        try:
            detail = json.loads(status_path.read_text()) if status_path else None
        except (OSError, json.JSONDecodeError):
            detail = None
        return {"active": self.evaluator is not None,
            "pending": len(self.pending_evaluations),
            "evaluation_id": detail.get("evaluation_id") if detail else (
                self.last_evaluation.get("evaluation_id") if self.last_evaluation else None),
            "protocol_id": detail.get("protocol_id") if detail else (
                self.last_evaluation.get("protocol_id") if self.last_evaluation else None),
            "status": detail.get("status") if detail else (
                self.last_evaluation.get("status") if self.last_evaluation else
                "queued" if self.pending_evaluations else "idle"),
            "progress": detail.get("progress") if detail else (
                self.last_evaluation.get("progress") if self.last_evaluation else None),
            "pid": self.evaluator.pid if self.evaluator is not None else None,
            "seed": self.evaluator_job.get("seed") if self.evaluator_job else None,
            "output_file": (self.evaluator_job["output"].name if self.evaluator_job
                else self.last_evaluation.get("output_file") if self.last_evaluation else None)}

    def _read_or_create_runs(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.evaluation_output.mkdir(parents=True, exist_ok=True)
        if self.stop_file.exists():
            raise RuntimeError(f"stop file already exists: {self.stop_file}; remove it to resume")
        for track in MECO_TRACKS:
            run_dir = self.output / track
            run_dir.mkdir(parents=True, exist_ok=True)
            latest = run_dir / "latest.pt"
            payload = _checkpoint_info(latest)
            if payload:
                if payload.get("sources") != self.sources:
                    raise ValueError(f"{track} checkpoint source differs from current source")
                if payload.get("protocol_id") != self.protocol_id:
                    raise ValueError(f"{track} checkpoint protocol differs from supervisor configuration")
                saved = dict(payload.get("config", {}))
                for key in ("updates",):
                    saved.pop(key, None)
                requested = asdict(self.configs[track])
                for key in ("updates",):
                    requested.pop(key, None)
                if saved != requested:
                    raise ValueError(f"{track} checkpoint has a different immutable configuration")
                if track in self.warm_start_hashes:
                    initialization = payload.get("initialization") or {}
                    if (initialization.get("mode") != "weights_only_supervised_warm_start" or
                            initialization.get("warm_start_artifact_sha256") !=
                            self.warm_start_hashes[track]):
                        raise ValueError(f"{track} checkpoint was not initialized from the configured warm-start artifact")
                self.run_ids[track] = payload["run_id"]
                self.completed_updates[track] = int(payload.get("update", 0))
            else:
                self.run_ids[track] = uuid.uuid4().hex
        self.configuration = {"scenario_id": self.scenario, "category": self.category,
            "seed": self.seed, "chunk_updates": self.chunk_updates,
            "training": self.hyperparameters, "devices": list(self.devices),
            "evaluation_categories": list(self.evaluation_categories),
            "evaluation_episodes": self.evaluation_episodes,
            "evaluation_seed": self.evaluation_seed,
            "evaluation_device": self.evaluation_device,
            "prediction_mode": self.prediction_mode,
            "evaluation_prediction_mode": self.evaluation_prediction_mode,
            "qualification_protocol_id": self.qualification_protocol_id,
            "qualification_protocol_sha256": self.qualification_protocol_sha256,
            "warm_start_artifact_sha256": dict(sorted(self.warm_start_hashes.items())),
            "chunk_reset_policy": CHUNK_RESET_POLICY,
            "task_scheduler": {"version": 1,
                "stage_exposure_chunks": self.stage_exposure_chunks,
                "semantics": "DAG exposure progression; not promotion qualification"},
            "candidate_boundary": CANDIDATE_BOUNDARY}
        self.configuration_hash = _hash(self.configuration)
        self.scheduler_path = self.output / "task-scheduler-state.json"
        if self.scheduler_path.is_file():
            scheduler_state = json.loads(self.scheduler_path.read_text())
            self.task_scheduler = MecoTaskScheduler(self.seed,
                self.stage_exposure_chunks, scheduler_state)
        else:
            self.task_scheduler = MecoTaskScheduler(self.seed,
                self.stage_exposure_chunks)
            _atomic_json(self.scheduler_path, self.task_scheduler.state)

    def _task_for_cycle(self, cycle, target_update):
        assignment = self.task_scheduler.schedule(cycle, target_update)
        if self.qualification_protocol is not None:
            from .meco_supervised import validate_training_assignment
            validate_training_assignment(self.qualification_protocol, assignment)
        self.current_assignment = assignment
        _atomic_json(self.scheduler_path, self.task_scheduler.state)
        return assignment

    def _observe_stop(self):
        """Convert an externally created STOP file into graceful stop intent."""
        if self.stop_file.exists():
            self.request_stop = True
        return self.request_stop

    def _complete_task_cycle(self, assignment):
        results = {}
        for track in MECO_TRACKS:
            try:
                payload = json.loads((self.output / track / "task-outcomes.json").read_text())
            except (OSError, json.JSONDecodeError):
                payload = {}
            if payload.get("assignment_id") == assignment["assignment_id"]:
                results[track] = payload
        missing = set(MECO_TRACKS) - set(results)
        if missing:
            raise RuntimeError(f"task outcome records missing for matched lanes: {sorted(missing)}")
        self.task_scheduler.complete(assignment, results)
        self.current_assignment = None
        _atomic_json(self.scheduler_path, self.task_scheduler.state)

    def _write_queued_manifests(self, statuses):
        for track in MECO_TRACKS:
            if statuses.get(track) != "queued":
                continue
            run_dir = self.output / track
            manifest_path = run_dir / "run.json"
            if (run_dir / "latest.pt").is_file():
                try:
                    manifest = json.loads(manifest_path.read_text())
                except (OSError, json.JSONDecodeError):
                    manifest = {}
                manifest.update(schema_version=1, run_id=self.run_ids[track], track=track,
                    config=asdict(self.configs[track]), sources=self.sources,
                    source_identity_sha256=self.source_hash, status="queued",
                    updates_semantics="updates per invocation chunk",
                    chunk_reset_policy=CHUNK_RESET_POLICY,
                    candidate_boundary=CANDIDATE_BOUNDARY, protocol_id=self.protocol_id)
                _atomic_json(manifest_path, manifest)

    def _run_training_batch(self, tracks, cycle, statuses, target_update,
                            task_assignment):
        processes = {}
        logs = {}
        failed = {}
        try:
            for track in tracks:
                if self.request_stop:
                    break
                if source_identity() != self.sources:
                    raise RuntimeError("source identity changed; refusing to continue queued training")
                remaining_updates = target_update - self._completed_updates(track)
                if remaining_updates <= 0:
                    statuses[track] = "complete"
                    continue
                base_config = self.configs[track]
                config = MecoTrainConfig(**{**asdict(base_config),
                    "updates": min(self.chunk_updates, remaining_updates)})
                latest = self.output / track / "latest.pt"
                warm_artifact = (self.warm_start_paths.get(track)
                    if not latest.is_file() else None)
                if warm_artifact is not None and hashlib.sha256(
                        warm_artifact.read_bytes()).hexdigest() != self.warm_start_hashes[track]:
                    raise RuntimeError(f"{track} warm-start artifact changed after supervisor initialization")
                command = _train_command(config=config, output=self.output / track,
                    run_id=self.run_ids[track], stop_file=self.stop_file,
                    resume=latest if latest.is_file() else None,
                    protocol_id=self.protocol_id, task_assignment=task_assignment,
                    warm_start=warm_artifact,
                    qualification_protocol=self.qualification_protocol_path)
                log_path = self.output / track / "supervisor.log"
                logs[track] = log_path.open("ab")
                process = subprocess.Popen(command, cwd=ROOT,
                    stdout=logs[track], stderr=subprocess.STDOUT,
                    start_new_session=True)
                processes[track] = process
                self.worker_pids[track] = process.pid
                statuses[track] = "running"
            self._write_queued_manifests(statuses)
            self._publish(cycle=cycle, phase="training", statuses=statuses)
            while processes:
                self._observe_stop()
                if self.request_stop and not self.stop_file.exists():
                    self.stop_file.parent.mkdir(parents=True, exist_ok=True)
                    self.stop_file.touch()
                for track, process in list(processes.items()):
                    result = process.poll()
                    if result is None:
                        continue
                    logs[track].close()
                    del processes[track]
                    self.worker_pids.pop(track, None)
                    self._refresh_progress(track)
                    if result != 0:
                        failed[track] = result
                        statuses[track] = "failed"
                    elif self.stop_file.exists():
                        statuses[track] = "stopped"
                    else:
                        statuses[track] = "complete"
                        self.completed_chunks += 1
                self._publish(cycle=cycle, phase="stopping" if self.request_stop else "training",
                              statuses=statuses, error=failed or None)
                self._poll_evaluator(cycle, statuses)
                if processes:
                    time.sleep(.5)
        except BaseException:
            # Child processes are detached into their own sessions. On every
            # setup or runtime failure, persist stop intent and reap all workers
            # before the output-root lock can be released. If stop-file creation
            # fails, terminate groups directly rather than waiting on a signal
            # path that cannot reach the children.
            stop_file_ready = False
            try:
                self.stop_file.parent.mkdir(parents=True, exist_ok=True)
                self.stop_file.touch(exist_ok=True)
                stop_file_ready = True
            except OSError:
                pass
            for process in processes.values():
                if process.poll() is None and not stop_file_ready:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except OSError:
                        pass
            for process in processes.values():
                if process.poll() is None:
                    try:
                        process.wait(timeout=30. if stop_file_ready else 5.)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=5.)
                        except (OSError, subprocess.TimeoutExpired):
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except OSError:
                                pass
                            process.wait()
            raise
        finally:
            for track, process in list(processes.items()):
                self.worker_pids.pop(track, None)
                self._refresh_progress(track)
            for log in logs.values():
                if not log.closed:
                    log.close()
        if failed:
            raise RuntimeError(f"training workers failed: {failed}")

    def _queue_evaluation(self, cycle):
        if source_identity() != self.sources:
            raise RuntimeError("source identity changed before paired evaluation")
        suffix = f"cycle-{cycle:06d}-{uuid.uuid4().hex[:8]}"
        output = self.evaluation_output / f"{suffix}.json"
        snapshot_dir = self.evaluation_output / "checkpoint-snapshots" / suffix
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        latest = {track: self.output / track / "latest.pt" for track in MECO_TRACKS}
        if any(not path.is_file() for path in latest.values()):
            raise RuntimeError("all five learned tracks need a checkpoint before evaluation")
        checkpoints = {}
        for track, source in latest.items():
            destination = snapshot_dir / f"{track}.pt"
            temporary = snapshot_dir / f".{track}.pt.tmp"
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
            checkpoints[track] = destination
        job = {"cycle": cycle, "output": output, "checkpoints": checkpoints,
               "seed": self.evaluation_seed + (cycle - 1) * self.evaluation_episodes}
        # Bound queued snapshot storage: preserve the active evaluation and
        # replace any older pending snapshot with the newest matched cycle.
        for stale in self.pending_evaluations:
            shutil.rmtree(stale["checkpoints"][MECO_TRACKS[0]].parent,
                          ignore_errors=True)
        self.pending_evaluations = [job]

    def _start_next_evaluation(self):
        if self.evaluator is not None or not self.pending_evaluations or self.request_stop:
            return
        job = self.pending_evaluations.pop(0)
        command = _evaluation_command(checkpoints=job["checkpoints"], scenario=self.scenario,
            categories=self.evaluation_categories, episodes=self.evaluation_episodes,
            seed=job["seed"], device=self.evaluation_device, output=job["output"],
            prediction_mode=self.evaluation_prediction_mode)
        try:
            self.evaluator_job = job
            self.evaluator_log = job["output"].with_suffix(".log").open("ab")
            self.evaluator = subprocess.Popen(command, cwd=ROOT,
                stdout=self.evaluator_log, stderr=subprocess.STDOUT,
                start_new_session=True)
        except OSError as exc:
            if self.evaluator_log is not None:
                self.evaluator_log.close()
            self.evaluator = None
            self.evaluator_log = None
            self.evaluator_job = None
            self.last_evaluation = {"cycle": job["cycle"],
                "output_file": job["output"].name, "status": "failed_to_start",
                "error": f"{type(exc).__name__}: {exc}"}

    def _poll_evaluator(self, cycle, statuses):
        self._observe_stop()
        if self.evaluator is None:
            self._start_next_evaluation()
        if self.evaluator is None:
            return
        result = self.evaluator.poll()
        if result is None:
            return
        self.evaluator_log.close()
        output = self.evaluator_job["output"]
        try:
            evaluator_status = json.loads(Path(str(output) + ".status.json").read_text())
        except (OSError, json.JSONDecodeError):
            evaluator_status = {}
        completion = {"cycle": self.evaluator_job["cycle"],
            "output_file": output.name, "status_file": output.name + ".status.json",
            "return_code": result,
            "evaluation_id": evaluator_status.get("evaluation_id"),
            "protocol_id": evaluator_status.get("protocol_id"),
            "progress": evaluator_status.get("progress"),
            "seed": self.evaluator_job.get("seed")}
        self.evaluator = None
        self.evaluator_log = None
        self.evaluator_job = None
        completion["status"] = "stopped" if self.request_stop else (
            "complete" if result == 0 else "failed")
        self.last_evaluation = completion
        self._start_next_evaluation()
        self._publish(cycle=cycle, phase="stopping" if self.request_stop else "training",
                      statuses=statuses, evaluation=completion)

    def _drain_evaluations(self, cycle, statuses):
        if self.request_stop:
            self.pending_evaluations.clear()
        while self.evaluator is not None:
            self._poll_evaluator(cycle, statuses)
            self._publish(cycle=cycle, phase="stopping" if self.request_stop else "evaluation",
                statuses=statuses, evaluation=getattr(self, "last_evaluation", None))
            if self.evaluator is not None:
                time.sleep(.5)
        if not self.request_stop:
            self._start_next_evaluation()
            while self.evaluator is not None:
                self._poll_evaluator(cycle, statuses)
                if self.evaluator is not None:
                    time.sleep(.5)

    def _mark_queued_stopped(self, statuses):
        for track in MECO_TRACKS:
            if statuses.get(track) != "queued":
                continue
            run_dir = self.output / track
            manifest_path = run_dir / "run.json"
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                manifest = {}
            manifest.update(schema_version=1, run_id=self.run_ids[track], track=track,
                config=asdict(self.configs[track]), sources=self.sources,
                source_identity_sha256=self.source_hash, status="stopped",
                completed_updates=self._completed_updates(track),
                updates_semantics="updates per invocation chunk",
                chunk_reset_policy=CHUNK_RESET_POLICY,
                candidate_boundary=CANDIDATE_BOUNDARY, protocol_id=self.protocol_id,
                stopped_utc=_utc())
            _atomic_json(manifest_path, manifest)
            statuses[track] = "stopped"

    def _train_to_update(self, target_update, cycle, statuses, order,
                         task_assignment):
        remaining = [track for track in order
                     if self._completed_updates(track) < target_update]
        while remaining and not self.request_stop:
            batch = []
            for device in self.devices:
                selected = next((track for track in remaining
                                 if self.configs[track].device == device), None)
                if selected is not None:
                    batch.append(selected)
                    remaining.remove(selected)
            self._run_training_batch(batch, cycle, statuses, target_update,
                                     task_assignment)
            remaining = [track for track in order
                         if self._completed_updates(track) < target_update]
            self._poll_evaluator(cycle, statuses)
        if not self.request_stop:
            wrong = {track: self._completed_updates(track) for track in order
                     if self._completed_updates(track) != target_update}
            if wrong:
                raise RuntimeError(f"track updates did not reach matched target {target_update}: {wrong}")

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        lock_handle = (self.output / ".supervisor.lock").open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_handle.close()
            raise RuntimeError("another MECO supervisor already owns this output root") from exc
        try:
            return self._run_locked()
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    def _run_locked(self):
        self._read_or_create_runs()
        self.supervisor_id = uuid.uuid4().hex
        self.configuration_hash = _hash(self.configuration)
        matched_update = max(self.completed_updates.values())
        self.completed_cycles = matched_update // self.chunk_updates
        self.completed_chunks = 0
        original_handlers = {}
        def handle_stop(signum, _frame):
            self.request_stop = True
            try:
                self.stop_file.parent.mkdir(parents=True, exist_ok=True)
                self.stop_file.touch(exist_ok=True)
            except OSError:
                pass
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.signal(signum, handle_stop)
        statuses = {track: "queued" for track in MECO_TRACKS}
        cycle = 0
        evaluation = None
        try:
            statuses = {track: ("complete" if self._completed_updates(track) == matched_update
                               else "queued") for track in MECO_TRACKS}
            pending = self.task_scheduler.state.get("pending_assignment")
            needs_catchup = (pending is not None or
                any(self._completed_updates(track) < matched_update for track in MECO_TRACKS))
            if needs_catchup:
                if pending is not None:
                    cycle = pending["cycle"]
                    catchup_target = pending["target_update"]
                    assignment = self._task_for_cycle(cycle, catchup_target)
                else:
                    catchup_target = ((matched_update + self.chunk_updates - 1)
                                      // self.chunk_updates) * self.chunk_updates
                    cycle = max(1, (catchup_target - 1) // self.chunk_updates + 1)
                    assignment = self._task_for_cycle(cycle, catchup_target)
                ahead = {track: self._completed_updates(track)
                         for track in MECO_TRACKS
                         if self._completed_updates(track) > catchup_target}
                if ahead:
                    raise RuntimeError(f"lane checkpoints exceed pending task frontier: {ahead}")
                target_order = list(MECO_TRACKS)
                self._publish(cycle=cycle, phase="catching_up", statuses=statuses)
                if any(self._completed_updates(track) < catchup_target for track in target_order):
                    self._train_to_update(catchup_target, cycle, statuses,
                                          target_order, assignment)
                if not self.request_stop and all(
                        self._completed_updates(track) == catchup_target
                        for track in target_order):
                    self._complete_task_cycle(assignment)
                    statuses = {track: "complete" for track in MECO_TRACKS}
                    self.completed_cycles = cycle
                    self._queue_evaluation(cycle)
                    self._start_next_evaluation()
            cycles_run = 0
            while not self.request_stop and (self.cycles == 0 or cycles_run < self.cycles):
                self._observe_stop()
                if self.request_stop:
                    break
                target_update = max(self.completed_updates.values()) + self.chunk_updates
                cycle = target_update // self.chunk_updates
                assignment = self._task_for_cycle(cycle, target_update)
                # Rotate the first scheduled track each cycle to avoid a fixed
                # ordering advantage when the final device batch is partial.
                offset = (cycle - 1) % len(MECO_TRACKS)
                order = list(MECO_TRACKS[offset:]) + list(MECO_TRACKS[:offset])
                statuses = {track: ("complete" if self._completed_updates(track) >= target_update
                                    else "queued") for track in MECO_TRACKS}
                self._publish(cycle=cycle, phase="queued", statuses=statuses)
                self._train_to_update(target_update, cycle, statuses, order,
                                      assignment)
                if self.request_stop:
                    break
                self._complete_task_cycle(assignment)
                self.completed_cycles = cycle
                cycles_run += 1
                self._queue_evaluation(cycle)
                self._start_next_evaluation()
                evaluation = {"active": self.evaluator is not None,
                    "pending": len(self.pending_evaluations),
                    "cycle": cycle, "completed_training_chunks": self.completed_chunks}
            self._observe_stop()
            if self.request_stop and not self.stop_file.exists():
                self.stop_file.parent.mkdir(parents=True, exist_ok=True)
                self.stop_file.touch()
            self._drain_evaluations(cycle, statuses)
            if self.request_stop:
                self.pending_evaluations.clear()
                self._mark_queued_stopped(statuses)
            final_state = "stopped" if self.request_stop else "complete"
            self._publish(cycle=cycle, phase=final_state, statuses=statuses,
                          evaluation=evaluation)
            return {"status": final_state, "cycle": cycle,
                    "completed_cycles": self.completed_cycles,
                    "completed_training_chunks": self.completed_chunks,
                    "queue_status": str(self.queue_path),
                    "supervisor_status": str(self.status_path)}
        except BaseException as exc:
            if self.evaluator is not None and self.evaluator.poll() is None:
                try:
                    os.killpg(self.evaluator.pid, signal.SIGTERM)
                    self.evaluator.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(self.evaluator.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    self.evaluator.wait()
                if self.evaluator_log is not None:
                    self.evaluator_log.close()
                self.evaluator = None
                self.evaluator_log = None
                self.evaluator_job = None
            self._publish(cycle=cycle, phase="failed", statuses=statuses,
                          evaluation=evaluation,
                          error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            for signum, handler in original_handlers.items():
                signal.signal(signum, handler)


def _parser():
    parser = argparse.ArgumentParser(description="Continuous round-robin MECO supervisor")
    parser.add_argument("--output", default=str(ROOT / "checkpoints" / "meco-five"))
    parser.add_argument("--evaluation-output", default=str(ROOT / "evaluations" / "meco-five"))
    default_devices = (",".join(f"cuda:{index}" for index in range(torch.cuda.device_count()))
                       if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", default=default_devices,
        help="comma-separated devices; at most one trainer runs on each device")
    parser.add_argument("--scenario", default="pickup_simple")
    parser.add_argument("--category", default="single")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-updates", type=int, default=100)
    parser.add_argument("--stage-exposure-chunks", type=int, default=10,
        help="chunks exposed per scenario before its dependants can enter training mixture")
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--action-ticks", type=int, default=12)
    parser.add_argument("--candidate-k", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=.993)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--clip-coef", type=float, default=.2)
    parser.add_argument("--entropy-coef", type=float, default=.01)
    parser.add_argument("--value-coef", type=float, default=.5)
    parser.add_argument("--max-grad-norm", type=float, default=.5)
    parser.add_argument("--route-cost-weight", type=float, default=.05)
    parser.add_argument("--p-mode", choices=("learned", "p0"), default="learned")
    parser.add_argument("--p-cost-scale", type=float, default=1.0)
    parser.add_argument("--p-points-per-grid-edge", type=float, default=.05)
    parser.add_argument("--prediction-mode", choices=PREDICTION_MODES, default="cv",
        help="shared Plan A/C forecast mode; default cv preserves current learned behavior")
    parser.add_argument("--evaluation-prediction-mode", choices=PREDICTION_MODES,
        help="evaluate D1 and D2 under the same forecast mode; default preserves D1=current, D2=cv")
    parser.add_argument("--evaluation-categories", default="single,cluster")
    parser.add_argument("--evaluation-episodes", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=20261024)
    parser.add_argument("--evaluation-device", default="cpu",
        help="device for the paired evaluator; defaults to CPU so learners keep their devices")
    parser.add_argument("--qualification-protocol",
        help="frozen qualification protocol used to constrain scheduled scenario/category/seeds")
    parser.add_argument("--warm-start-artifacts", type=json.loads, default=None,
        help="JSON object mapping every MECO track to a validated weights-only artifact")
    parser.add_argument("--cycles", type=int, default=0,
        help="number of complete five-track chunks; zero continues until stopped")
    parser.add_argument("--stop-file")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    categories = tuple(item.strip() for item in args.evaluation_categories.split(",")
                       if item.strip())
    supervisor = MecoSupervisor(output=args.output, evaluation_output=args.evaluation_output,
        devices=devices, scenario=args.scenario, category=args.category, seed=args.seed,
        chunk_updates=args.chunk_updates, rollout_steps=args.rollout_steps,
        epochs=args.epochs, action_ticks=args.action_ticks,
        learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef, entropy_coef=args.entropy_coef,
        value_coef=args.value_coef, max_grad_norm=args.max_grad_norm,
        route_cost_weight=args.route_cost_weight,
        p_mode=args.p_mode, p_cost_scale=args.p_cost_scale,
        p_points_per_grid_edge=args.p_points_per_grid_edge,
        prediction_mode=args.prediction_mode,
        evaluation_prediction_mode=args.evaluation_prediction_mode,
        candidate_k=args.candidate_k,
        evaluation_categories=categories, evaluation_episodes=args.evaluation_episodes,
        evaluation_seed=args.evaluation_seed, evaluation_device=args.evaluation_device,
        cycles=args.cycles, stop_file=args.stop_file,
        stage_exposure_chunks=args.stage_exposure_chunks,
        qualification_protocol=args.qualification_protocol,
        warm_start_artifacts=args.warm_start_artifacts)
    print(json.dumps(supervisor.run(), indent=2))


if __name__ == "__main__":
    main()
