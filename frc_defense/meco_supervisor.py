"""Fair round-robin supervisor for continuous five-track MECO research runs.

The supervisor is opt-in and is not started as part of import or dashboard
startup. It runs at most one trainer process per configured device, gives each
learned track the same update chunk and immutable training settings, and runs a
paired all-track/control evaluation after every complete five-track cycle. A
per-cycle queue refills a device as soon as its current track reaches the target.
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
from .meco_final_test_ledger import assert_training_not_released_after_final_test
from .meco_candidates import CANDIDATE_BOUNDARY
from .meco_track_training import (CHUNK_RESET_POLICY, MecoTrainConfig, ROOT,
    resolve_evaluation_case, resolve_evaluation_protocol)
from .meco_task_scheduler import MecoTaskScheduler
from .meco_phase4 import (phase4_train_protocol_id,
    phase4_training_metadata)
from .meco_tracks import MECO_TRACKS, PREDICTION_MODES, prediction_contract


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _native_extension_attestation(identity):
    from .meco_training_manifest import _valid_native_identity
    if not _valid_native_identity(identity):
        raise RuntimeError("loaded native HIP extension identity is missing or malformed")
    return {"schema": 1, "identity": dict(identity),
        "identity_sha256": _hash(identity)}


def _validate_native_extension_attestation(attestation, expected_identity):
    if (not isinstance(attestation, dict) or attestation.get("schema") != 1 or
            not isinstance(attestation.get("identity"), dict) or
            attestation.get("identity_sha256") != _hash(attestation["identity"])):
        raise RuntimeError("native HIP extension attestation is missing or malformed")
    expected = _native_extension_attestation(expected_identity)
    if attestation != expected:
        raise RuntimeError("loaded native HIP extension differs from its pinned attestation")
    return attestation


def _validate_runtime_source_binding(sources, loaded_identity, attestation,
                                    expected_source_projection=None):
    _validate_native_extension_attestation(attestation, loaded_identity)
    if not isinstance(sources, dict) or sources.get("_native_fuel_extension") != loaded_identity:
        raise RuntimeError("runtime source identity does not bind the attested loaded HIP binary")
    if expected_source_projection is not None:
        from .meco_training_manifest import source_identity_projection
        if source_identity_projection(sources) != expected_source_projection:
            raise RuntimeError("prepared runtime source/config identity differs from campaign manifest")
    return _hash(sources)


def _validate_runtime_warm_start(track, artifact, *, qualification_protocol,
                                 expected_sources, prediction_mode,
                                 native_attestation):
    loaded_identity = (expected_sources.get("_native_fuel_extension")
                       if isinstance(expected_sources, dict) else None)
    _validate_native_extension_attestation(native_attestation, loaded_identity)
    from .meco_supervised import validate_warm_start_artifact
    return validate_warm_start_artifact(track, artifact,
        qualification_protocol=qualification_protocol,
        expected_sources=expected_sources,
        expected_prediction_mode=prediction_mode)


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


def _append_evaluation_history(evaluation_root: Path, *, cycle: int, output: Path,
                              completion: dict, checkpoints: dict):
    """Persist every terminal evaluation before queue status can advance."""
    evaluation_root, output = Path(evaluation_root), Path(output)
    history_path = evaluation_root / "evaluation-history.json"
    try:
        history = json.loads(history_path.read_text())
    except FileNotFoundError:
        history = {"schema_version": 1, "entries": []}
    if history.get("schema_version") != 1 or not isinstance(history.get("entries"), list):
        raise ValueError("evaluation history is malformed")
    entries = history["entries"]
    prior = entries[-1].get("entry_sha256") if entries else None
    if any(row.get("output_file") == output.name for row in entries):
        raise ValueError("evaluation output is already present in durable history")
    snapshot_rows = []
    for track, path in sorted(checkpoints.items()):
        path = Path(path)
        snapshot_rows.append({"track": track, "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    status_path = Path(str(output) + ".status.json")
    row = {"sequence": len(entries) + 1, "previous_entry_sha256": prior,
        "cycle": cycle, "output_file": output.name,
        "status": completion.get("status"),
        "evaluation_id": completion.get("evaluation_id"),
        "protocol_id": completion.get("protocol_id"),
        "return_code": completion.get("return_code"),
        "report_sha256": (hashlib.sha256(output.read_bytes()).hexdigest()
            if output.is_file() else None),
        "status_file": status_path.name,
        "status_sha256": (hashlib.sha256(status_path.read_bytes()).hexdigest()
            if status_path.is_file() else None),
        "checkpoint_snapshots": snapshot_rows}
    row["entry_sha256"] = _hash(row)
    entries.append(row)
    _atomic_json(history_path, history)
    return {"count": len(entries), "head_sha256": row["entry_sha256"]}


def _checkpoint_info(path: Path):
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload


def _reap_worker_groups(processes, *, graceful_seconds=1800., term_seconds=10.,
                        kill_seconds=10.):
    """Give workers time to stop, then bound and verify process-group teardown."""
    def group_exists(process):
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def alive_groups():
        return [process for process in processes.values()
                if process.poll() is None or group_exists(process)]

    deadline = time.monotonic() + graceful_seconds
    while time.monotonic() < deadline:
        alive = alive_groups()
        if not alive:
            break
        time.sleep(min(.5, max(0., deadline - time.monotonic())))
    for process in processes.values():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + term_seconds
    while time.monotonic() < deadline:
        alive = alive_groups()
        if not alive:
            break
        time.sleep(min(.25, max(0., deadline - time.monotonic())))
    for process in processes.values():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
    deadline = time.monotonic() + kill_seconds
    while time.monotonic() < deadline:
        alive = alive_groups()
        if not alive:
            break
        time.sleep(min(.1, max(0., deadline - time.monotonic())))
    return [process.pid for process in alive_groups()]


def _train_command(*, config: MecoTrainConfig, output: Path, run_id: str,
                   stop_file: Path, abort_file: Path, invocation_id: str,
                   resume: Path | None, protocol_id: str,
                   task_assignment: dict, source_hash: str,
                   native_extension_attestation: dict,
                   warm_start: Path | None = None,
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
        "--output", str(output), "--stop-file", str(stop_file),
        "--abort-file", str(abort_file), "--invocation-id", invocation_id,
        "--run-id", run_id,
        "--protocol-id", protocol_id,
        "--expected-source-sha256", source_hash,
        "--native-extension-attestation-json", json.dumps(
            native_extension_attestation, sort_keys=True, allow_nan=False)]
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
                        device, output, prediction_mode=None, split="diagnostic",
                        evaluation_protocol=None, evaluation_case=None,
                        episode_seeds=None):
    command = [sys.executable, "-m", "frc_defense.meco_track_training", "evaluate",
        "--scenario", scenario, "--categories", ",".join(categories),
        "--episodes", str(episodes), "--seed", str(seed), "--device", device,
        "--output", str(output), "--split", split]
    if evaluation_protocol is not None:
        command.extend(("--evaluation-protocol", str(evaluation_protocol),
                        "--evaluation-case-json", json.dumps(evaluation_case,
                            sort_keys=True, allow_nan=False),
                        "--episode-seeds", ",".join(str(value) for value in episode_seeds)))
    if prediction_mode is not None:
        command.extend(("--prediction-mode", prediction_mode))
    for track in MECO_TRACKS:
        command.extend((f"--{track.lower()}-checkpoint", str(checkpoints[track])))
    return command


def resolve_supervisor_evaluation(protocol_path, *, scenario, categories, split,
                                  qualification_case=None):
    """Resolve the supervisor's optional validation-only evaluation contract."""
    if protocol_path is None:
        if split != "diagnostic" or qualification_case is not None:
            raise ValueError("unbound supervisor evaluation must remain diagnostic")
        return None, None, None
    if split == "test":
        raise ValueError("automatic final-test selection is forbidden")
    if split != "validation":
        raise ValueError("bound supervisor evaluation requires validation split")
    case, seeds = resolve_evaluation_case(protocol_path, scenario_id=scenario,
        categories=categories, split=split, qualification_case=qualification_case)
    binding = resolve_evaluation_protocol(protocol_path, scenario_id=scenario,
        categories=categories, split=split, episode_seeds=seeds,
        qualification_case=case)
    return case, seeds, binding


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
                 qualification_protocol: str | None = None,
                 evaluation_protocol: str | None = None,
                 evaluation_split: str = "diagnostic",
                 evaluation_case: dict | None = None,
                 phase4_train_exposure: bool = False,
                 campaign_manifest: dict | None = None):
        self.manifest = None
        self.manifest_validation = None
        self.manifest_digest = None
        self.manifest_configs = None
        if campaign_manifest is not None:
            from .meco_training_manifest import validate_campaign_manifest
            self.manifest_validation = validate_campaign_manifest(campaign_manifest)
            if not self.manifest_validation["valid"]:
                raise ValueError("campaign manifest failed validation")
            self.manifest = campaign_manifest
            self.manifest_digest = self.manifest_validation["canonical_sha256"]
            campaign = campaign_manifest["campaign"]
            output = campaign["root"]
            evaluation_output = campaign["evaluation_output"]
            if Path(output).exists() or Path(evaluation_output).exists():
                raise ValueError("manifest diagnostic launch requires fresh, absent campaign output roots")
            self.manifest_configs = {track: MecoTrainConfig(**campaign_manifest["tracks"][track]["config"])
                                     for track in MECO_TRACKS}
            expected_devices = tuple(campaign_manifest["devices"]["training_pool"])
            expected_eval_device = campaign_manifest["devices"]["evaluation"]
            if tuple(devices) != expected_devices or evaluation_device != expected_eval_device:
                raise ValueError("runtime device arguments conflict with validated campaign manifest")
            if (self.manifest_configs[MECO_TRACKS[0]].scenario_id != scenario or
                    self.manifest_configs[MECO_TRACKS[0]].category != category or
                    self.manifest_configs[MECO_TRACKS[0]].seed != seed or
                    self.manifest_configs[MECO_TRACKS[0]].updates != chunk_updates):
                raise ValueError("runtime training arguments conflict with validated campaign manifest")
        from .meco_device import require_indexed_cuda_device
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("devices must be a nonempty list without duplicates")
        for device in devices:
            require_indexed_cuda_device(device, purpose="MECO supervisor training")
        require_indexed_cuda_device(evaluation_device,
            purpose="MECO supervisor evaluation")
        if cycles < 0 or chunk_updates < 1 or evaluation_episodes < 1:
            raise ValueError("cycles must be nonnegative; chunk/evaluation counts positive")
        self.output = Path(output).expanduser().resolve()
        self.evaluation_output = Path(evaluation_output).expanduser().resolve()
        expected_checkpoint_root = (ROOT / "checkpoints" / "meco-five").resolve()
        expected_evaluation_root = (ROOT / "evaluations" / "meco-five").resolve()
        if (self.manifest is None and (self.output != expected_checkpoint_root
                or self.evaluation_output != expected_evaluation_root)):
            raise ValueError("supervisor roots must match the dashboard's pinned MECO roots")
        self.devices = devices
        from .fuel_physics_hip import prepare_native_fuel_extension
        from .fuel_physics_hip import native_extension_identity
        loaded_native_identities = []
        for device in sorted(set(devices) | {evaluation_device}):
            prepare_native_fuel_extension(device)
            identity = native_extension_identity()
            self.native_extension_attestation = _native_extension_attestation(identity)
            loaded_native_identities.append(identity)
        if any(identity != loaded_native_identities[0]
               for identity in loaded_native_identities[1:]):
            raise RuntimeError("indexed devices loaded different native HIP binary identities")
        self.native_extension_attestation = _native_extension_attestation(
            loaded_native_identities[0])
        self.sources = source_identity()
        self.source_hash = _validate_runtime_source_binding(self.sources,
            loaded_native_identities[0], self.native_extension_attestation,
            expected_source_projection=(self.manifest["source"]["identity"]
                if self.manifest is not None else None))
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
            assert_training_not_released_after_final_test(
                self.qualification_protocol_path, self.qualification_protocol_id,
                self.qualification_protocol_sha256, _hash(source_identity()))
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
        self.evaluation_protocol_path = (Path(evaluation_protocol).expanduser().resolve()
            if evaluation_protocol else None)
        self.evaluation_split = evaluation_split
        self.evaluation_protocol_id = None
        self.evaluation_protocol_sha256 = None
        self.evaluation_binding = None
        self.evaluation_case = None
        self.evaluation_episode_seeds = None
        (self.evaluation_case, self.evaluation_episode_seeds,
         self.evaluation_binding) = resolve_supervisor_evaluation(
            self.evaluation_protocol_path, scenario=scenario,
            categories=evaluation_categories, split=evaluation_split,
            qualification_case=evaluation_case)
        if self.evaluation_binding is not None:
            self.evaluation_protocol_id = self.evaluation_binding["protocol_id"]
            self.evaluation_protocol_sha256 = self.evaluation_binding["protocol_sha256"]
        self.stage_exposure_chunks = stage_exposure_chunks
        if type(phase4_train_exposure) is not bool:
            raise ValueError("phase4_train_exposure must be boolean")
        self.phase4_train_exposure = phase4_train_exposure
        self.chunk_updates = chunk_updates
        self.evaluation_categories = evaluation_categories
        if self.evaluation_binding is not None:
            evaluation_episodes = len(self.evaluation_episode_seeds)
            evaluation_seed = self.evaluation_episode_seeds[0]
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_seed = evaluation_seed
        self.evaluation_device = evaluation_device
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
            config = self.manifest_configs[track] if self.manifest_configs else MecoTrainConfig(track=track, scenario_id=scenario, category=category,
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
        self.track_dirs = {track: Path(self.configs[track].output).expanduser().resolve()
            for track in MECO_TRACKS}
        self.source_hash = _hash(self.sources)
        if warm_start_artifacts:
            for track in MECO_TRACKS:
                artifact = Path(warm_start_artifacts[track]).expanduser().resolve()
                _validate_runtime_warm_start(track, artifact,
                    qualification_protocol=self.qualification_protocol_path,
                    expected_sources=self.sources,
                    expected_prediction_mode=self.prediction_mode,
                    native_attestation=self.native_extension_attestation)
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
            "evaluation_split": self.evaluation_split,
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "evaluation_binding": self.evaluation_binding,
            "evaluation_prediction_mode": evaluation_prediction_mode,
            "qualification_protocol_id": self.qualification_protocol_id,
            "qualification_protocol_sha256": self.qualification_protocol_sha256,
            "warm_start_artifact_sha256": dict(sorted(self.warm_start_hashes.items())),
            "candidate_boundary": CANDIDATE_BOUNDARY,
            "chunk_reset_policy": CHUNK_RESET_POLICY,
            "task_scheduler": {"version": 1,
                "stage_exposure_chunks": stage_exposure_chunks,
                "phase4_train_exposure": self.phase4_train_exposure,
                "semantics": "DAG exposure progression; not promotion qualification"},
            "phase4_training_protocol_id": (phase4_train_protocol_id()
                if self.phase4_train_exposure else None),
            "D0_control": "production deterministic slot-3 controller; native 20 ms cadence; outside candidate table"})

    def _lane_payload(self, track, statuses):
        status = statuses.get(track, "queued")
        deferred = (status == "queued" and
            self.configs[track].device == self.evaluation_device and
            (self.evaluator is not None or self.pending_evaluations))
        return {"track": track, "run_id": self.run_ids[track],
                "run_directory": track, "device": self.configs[track].device,
                "status": "deferred_for_evaluation" if deferred else status,
                "deferred_reason": "evaluation_device_lease" if deferred else None,
                "pid": self.worker_pids.get(track),
                "completed_updates": self._heartbeat_updates(track),
                "chunk_updates": self.chunk_updates}

    def _heartbeat_updates(self, track):
        try:
            heartbeat = json.loads((self.track_dirs[track] / "heartbeat.json").read_text())
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
            "category_exposures", "recovery_pressure", "phase4_exposure_counts")}

    def _refresh_progress(self, track):
        payload = _checkpoint_info(self.track_dirs[track] / "latest.pt")
        if payload:
            self.completed_updates[track] = int(payload.get("update", 0))

    def _publish(self, *, cycle, phase, statuses, evaluation=None, error=None):
        now = _utc()
        history_path = self.evaluation_output / "evaluation-history.json"
        try:
            history_entries = json.loads(history_path.read_text()).get("entries", [])
        except (OSError, json.JSONDecodeError):
            history_entries = []
        queue = {"schema_version": 1, "updated_utc": now,
            "supervisor_id": self.supervisor_id, "protocol_id": self.protocol_id,
            "prediction_mode": self.prediction_mode,
            "prediction_contract": prediction_contract(self.prediction_mode),
            "source_identity_sha256": self.source_hash,
            "native_extension_attestation": self.native_extension_attestation,
            "qualification_protocol_id": self.qualification_protocol_id,
            "qualification_protocol_sha256": self.qualification_protocol_sha256,
            "evaluation_split": self.evaluation_split,
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "evaluation_binding": self.evaluation_binding,
            "warm_start_artifact_sha256": dict(sorted(self.warm_start_hashes.items())),
            "configuration_sha256": self.configuration_hash,
            "cycle": cycle, "phase": phase,
            "candidate_boundary": CANDIDATE_BOUNDARY,
            "lanes": [self._lane_payload(track, statuses) for track in MECO_TRACKS],
            "task_assignment": self.current_assignment,
            "task_scheduler": self._task_scheduler_status(),
            "evaluation_case": self.evaluation_binding,
            "exact_episode_seeds": self.evaluation_episode_seeds,
            "evaluation_history": {"count": len(history_entries),
                "head_sha256": (history_entries[-1].get("entry_sha256")
                    if history_entries else None)},
            "evaluation": self._evaluation_status()}
        if getattr(self, "manifest", None) is not None:
            binding = self._manifest_status_binding()
            queue["campaign_manifest"] = binding
        _atomic_json(self.queue_path, queue)
        status = {"schema_version": 1,
            "supervisor_id": self.supervisor_id, "protocol_id": self.protocol_id,
            "prediction_mode": self.prediction_mode,
            "prediction_contract": prediction_contract(self.prediction_mode),
            "source_identity_sha256": self.source_hash,
            "native_extension_attestation": self.native_extension_attestation,
            "qualification_protocol_id": self.qualification_protocol_id,
            "qualification_protocol_sha256": self.qualification_protocol_sha256,
            "evaluation_split": self.evaluation_split,
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "evaluation_binding": self.evaluation_binding,
            "warm_start_artifact_sha256": dict(sorted(self.warm_start_hashes.items())),
            "configuration_sha256": self.configuration_hash,
            "heartbeat_utc": now, "status": phase, "cycle": cycle,
            "completed_cycles": self.completed_cycles,
            "completed_training_chunks": self.completed_chunks,
            "lanes": [self._lane_payload(track, statuses) for track in MECO_TRACKS],
            "task_assignment": self.current_assignment,
            "task_scheduler": self._task_scheduler_status(),
            "evaluation_case": self.evaluation_binding,
            "exact_episode_seeds": self.evaluation_episode_seeds,
            "evaluation": evaluation, "error": error,
            "stop_requested": self.request_stop,
            "candidate_boundary": CANDIDATE_BOUNDARY}
        if getattr(self, "manifest", None) is not None:
            status["campaign_manifest"] = self._manifest_status_binding()
        _atomic_json(self.status_path, status)

    def _manifest_status_binding(self):
        manifest = self.manifest
        return {"canonical_sha256": self.manifest_digest,
            "campaign_id": manifest["campaign"]["id"],
            "campaign_root": str(self.output),
            "evaluation_output": str(self.evaluation_output),
            "source": manifest["source"],
            "native_extension_attestation": self.native_extension_attestation,
            "track_config_sha256": {t: manifest["tracks"][t]["config_sha256"]
                for t in MECO_TRACKS},
            "qualification_protocol": manifest["qualification_protocol"],
            "evaluation_protocol": manifest["evaluation_protocol"],
            "device_lease": manifest["devices"]["indexed_device_leases"],
            "launch_profile": "diagnostic_fresh_start",
            "phase2_ready": self.manifest_validation["phase2_ready"],
            "phase2_complete": False, "qualification_eligible": False}

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
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "split": self.evaluation_split,
            "evaluation_case": self.evaluation_binding,
            "exact_episode_seeds": self.evaluation_episode_seeds,
            "output_file": (self.evaluator_job["output"].name if self.evaluator_job
                else self.last_evaluation.get("output_file") if self.last_evaluation else None)}

    def _read_or_create_runs(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.evaluation_output.mkdir(parents=True, exist_ok=True)
        if self.stop_file.exists():
            raise RuntimeError(f"stop file already exists: {self.stop_file}; remove it to resume")
        for track in MECO_TRACKS:
            run_dir = self.track_dirs[track]
            run_dir.mkdir(parents=True, exist_ok=True)
            latest = run_dir / "latest.pt"
            payload = _checkpoint_info(latest)
            if payload:
                if payload.get("sources") != self.sources:
                    raise ValueError(f"{track} checkpoint source differs from current source")
                if payload.get("protocol_id") != self.protocol_id:
                    raise ValueError(f"{track} checkpoint protocol differs from supervisor configuration")
                checkpoint_task = payload.get("task_assignment") or {}
                expected_phase4 = phase4_training_metadata(checkpoint_task)
                if payload.get("phase4_training") != expected_phase4:
                    raise ValueError(f"{track} checkpoint Phase 4 training provenance mismatch")
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
                "phase4_train_exposure": self.phase4_train_exposure,
                "semantics": "DAG exposure progression; not promotion qualification"},
            "phase4_training_protocol_id": (phase4_train_protocol_id()
                if self.phase4_train_exposure else None),
            "candidate_boundary": CANDIDATE_BOUNDARY}
        self.configuration_hash = _hash(self.configuration)
        self.scheduler_path = self.output / "task-scheduler-state.json"
        if self.scheduler_path.is_file():
            scheduler_state = json.loads(self.scheduler_path.read_text())
            self.task_scheduler = MecoTaskScheduler(self.seed,
                self.stage_exposure_chunks, scheduler_state,
                self.phase4_train_exposure)
        else:
            self.task_scheduler = MecoTaskScheduler(self.seed,
                self.stage_exposure_chunks,
                phase4_train_exposure=self.phase4_train_exposure)
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

    def _verify_worker_provenance(self, track, target_update, assignment,
                                  worker_manifest, *, checkpoint=None,
                                  require_complete=True):
        if checkpoint is None:
            checkpoint = _checkpoint_info(self.track_dirs[track] / "latest.pt")
        if worker_manifest is None:
            try:
                worker_manifest = json.loads((self.track_dirs[track] / "run.json").read_text())
            except (OSError, json.JSONDecodeError):
                worker_manifest = None
        expected_attestation = self.native_extension_attestation
        if not isinstance(worker_manifest, dict) or worker_manifest.get("sources") != self.sources:
            raise RuntimeError(f"{track} worker run record has mismatched source provenance")
        if (worker_manifest.get("source_identity_sha256") != self.source_hash
                or worker_manifest.get("supervisor_source_identity_sha256") != self.source_hash
                or worker_manifest.get("supervisor_native_extension_attestation") != expected_attestation
                or worker_manifest.get("run_id") != self.run_ids[track]
                or worker_manifest.get("track") != track
                or worker_manifest.get("task_assignment") != assignment
                or worker_manifest.get("completed_updates") != target_update
                or worker_manifest.get("protocol_id") != self.protocol_id
                or (require_complete and worker_manifest.get("status") != "complete")):
            raise RuntimeError(f"{track} worker run record does not match supervisor binding")
        if (not isinstance(checkpoint, dict) or checkpoint.get("sources") != self.sources
                or not isinstance(checkpoint.get("config"), dict)
                or checkpoint["config"].get("track") != track
                or checkpoint.get("run_id") != self.run_ids[track]
                or checkpoint.get("protocol_id") != self.protocol_id
                or checkpoint.get("update") != target_update
                or _hash(checkpoint.get("sources")) != self.source_hash
                or checkpoint.get("task_assignment") != assignment):
            raise RuntimeError(f"{track} final checkpoint provenance differs from supervisor binding")

    def _recover_task_outcome(self, track, assignment):
        checkpoint_path = self.track_dirs[track] / "latest.pt"
        checkpoint = _checkpoint_info(checkpoint_path)
        self._verify_worker_provenance(track, assignment.get("target_update"),
            assignment, None, checkpoint=checkpoint, require_complete=False)
        target_update = assignment.get("target_update")
        progress = checkpoint.get("task_progress") if isinstance(checkpoint, dict) else None
        checkpoint_assignment = checkpoint.get("task_assignment", {}) if isinstance(
            checkpoint, dict) else {}
        if (not isinstance(checkpoint, dict)
                or checkpoint.get("update") != target_update
                or not isinstance(progress, dict)
                or progress.get("assignment_id") != assignment.get("assignment_id")
                or checkpoint_assignment.get("assignment_id") != assignment.get("assignment_id")
                or checkpoint_assignment.get("target_update") != target_update
                or progress.get("cycle") != assignment.get("cycle")
                or progress.get("scenario_id") != assignment.get("scenario_id")
                or progress.get("category") != assignment.get("category")):
            raise RuntimeError(
                f"{track} checkpoint cannot recover task outcome for pending assignment "
                f"{assignment.get('assignment_id')}")
        counters = [progress.get(name) for name in (
            "episodes", "successes", "failures", "aborted_no_legal_support")]
        if (any(type(value) is not int or value < 0 for value in counters)
                or progress["successes"] + progress["failures"] != progress["episodes"]):
            raise RuntimeError(f"{track} checkpoint has invalid task outcome counters")
        outcome = {key: progress[key] for key in (
            "assignment_id", "cycle", "scenario_id", "category", "episodes",
            "successes", "failures", "aborted_no_legal_support")}
        outcome["completed_updates"] = target_update
        outcome_path = self.track_dirs[track] / "task-outcomes.json"
        try:
            current = json.loads(outcome_path.read_text())
        except (OSError, json.JSONDecodeError):
            current = {}
        if current != outcome:
            _atomic_json(outcome_path, outcome)
        manifest_path = self.track_dirs[track] / "run.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = {}
        manifest.update(schema_version=1, run_id=checkpoint.get("run_id"),
            track=track, status="complete", completed_updates=target_update,
            task_assignment=assignment, task_outcomes=outcome,
            phase4_training=phase4_training_metadata(assignment),
            recovered_task_outcome=True, recovered_utc=_utc())
        _atomic_json(manifest_path, manifest)
        return outcome

    def _complete_task_cycle(self, assignment):
        results = {}
        for track in MECO_TRACKS:
            results[track] = self._recover_task_outcome(track, assignment)
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
            run_dir = self.track_dirs[track]
            manifest_path = run_dir / "run.json"
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                manifest = {}
            manifest.update(schema_version=1, run_id=self.run_ids[track], track=track,
                config=asdict(self.configs[track]), sources=self.sources,
                source_identity_sha256=self.source_hash,
                native_extension_attestation=self.native_extension_attestation,
                status="queued", updates_semantics="updates per invocation chunk",
                chunk_reset_policy=CHUNK_RESET_POLICY,
                candidate_boundary=CANDIDATE_BOUNDARY, protocol_id=self.protocol_id)
            _atomic_json(manifest_path, manifest)

    def _run_training_batch(self, tracks, cycle, statuses, target_update,
                            task_assignment):
        processes = {}
        logs = {}
        failed = {}
        abort_file = (self.output / ".sibling-aborts" /
            f"{self.supervisor_id}-cycle-{cycle:06d}.json")
        abort_started = None
        force_terminated = set()
        if len(tracks) != len(set(tracks)):
            raise ValueError("training dispatch queue contains duplicate tracks")
        pending = [track for track in tracks
                   if self._completed_updates(track) < target_update]

        def dispatch(track):
            if source_identity() != self.sources:
                raise RuntimeError("source identity changed; refusing to continue queued training")
            remaining_updates = target_update - self._completed_updates(track)
            if remaining_updates <= 0:
                statuses[track] = "complete"
                return
            base_config = self.configs[track]
            config = MecoTrainConfig(**{**asdict(base_config),
                "updates": min(self.chunk_updates, remaining_updates)})
            from .fuel_physics_hip import (prepare_native_fuel_extension,
                native_extension_identity)
            prepare_native_fuel_extension(config.device)
            loaded_identity = native_extension_identity()
            _validate_native_extension_attestation(
                self.native_extension_attestation, loaded_identity)
            if source_identity() != self.sources:
                raise RuntimeError("source identity changed after native HIP attestation")
            latest = self.track_dirs[track] / "latest.pt"
            warm_artifact = (self.warm_start_paths.get(track)
                if not latest.is_file() else None)
            if warm_artifact is not None and hashlib.sha256(
                    warm_artifact.read_bytes()).hexdigest() != self.warm_start_hashes[track]:
                raise RuntimeError(f"{track} warm-start artifact changed after supervisor initialization")
            command = _train_command(config=config, output=self.track_dirs[track],
                run_id=self.run_ids[track], stop_file=self.stop_file,
                abort_file=abort_file, invocation_id=self.supervisor_id,
                resume=latest if latest.is_file() else None,
                protocol_id=self.protocol_id, task_assignment=task_assignment,
                source_hash=self.source_hash,
                native_extension_attestation=self.native_extension_attestation,
                warm_start=warm_artifact,
                qualification_protocol=self.qualification_protocol_path)
            # Persist the exact loaded binary binding before any subprocess can
            # inherit the GPU runtime or begin training.
            self._write_queued_manifests(statuses)
            log_path = self.track_dirs[track] / "supervisor.log"
            logs[track] = log_path.open("ab")
            try:
                process = subprocess.Popen(command, cwd=ROOT,
                    stdout=logs[track], stderr=subprocess.STDOUT,
                    start_new_session=True)
            except BaseException:
                logs[track].close()
                logs.pop(track, None)
                raise
            processes[track] = process
            self.worker_pids[track] = process.pid
            statuses[track] = "running"
            self._publish(cycle=cycle, phase="training", statuses=statuses)

        try:
            def fill_free_devices():
                nonlocal abort_started
                if abort_started is not None or self._observe_stop():
                    return
                for device in self.devices:
                    if abort_started is not None or self.request_stop:
                        break
                    occupied = any(self.configs[track].device == device
                                   for track in processes)
                    if occupied:
                        continue
                    # A queued or running evaluator owns its indexed device.
                    # Keep that lane pending until the evaluation releases it.
                    if (device == self.evaluation_device and
                            (self.evaluator is not None or self.pending_evaluations)):
                        continue
                    selected = next((track for track in pending
                                     if self.configs[track].device == device), None)
                    if selected is None:
                        continue
                    pending.remove(selected)
                    try:
                        dispatch(selected)
                    except Exception as exc:
                        failed[selected] = repr(exc)
                        statuses[selected] = "failed"
                        _atomic_json(self.track_dirs[selected] / "run.json", {
                            "schema_version": 1, "run_id": self.run_ids[selected],
                            "track": selected, "config": asdict(self.configs[selected]),
                            "sources": self.sources,
                            "source_identity_sha256": self.source_hash,
                            "protocol_id": self.protocol_id, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "completed_updates": self._completed_updates(selected),
                            "failed_utc": _utc()})
                        abort_started = time.monotonic()
                        _atomic_json(abort_file, {
                            "schema_version": 1, "supervisor_id": self.supervisor_id,
                            "invocation_id": self.supervisor_id, "cycle": cycle,
                            "protocol_id": self.protocol_id,
                            "source_identity_sha256": self.source_hash,
                            "failed_track": selected, "failed_exit_code": None,
                            "error": f"{type(exc).__name__}: {exc}", "created_utc": _utc()})
                        break

            fill_free_devices()
            while processes or (pending and abort_started is None and not self.request_stop):
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
                        try:
                            worker_manifest = json.loads(
                                (self.track_dirs[track] / "run.json").read_text())
                        except (OSError, json.JSONDecodeError):
                            worker_manifest = {}
                        if (abort_started is not None and
                                worker_manifest.get("status") ==
                                    "aborted_due_to_sibling_failure" and
                                worker_manifest.get("supervisor_invocation_id") ==
                                    self.supervisor_id):
                            statuses[track] = "aborted_due_to_sibling_failure"
                            worker_manifest.update(status=statuses[track],
                                sibling_abort_file=str(abort_file),
                                sibling_abort_invocation_id=self.supervisor_id,
                                sibling_abort_observed_utc=_utc(),
                                worker_exit_code=result)
                            _atomic_json(self.track_dirs[track] / "run.json", worker_manifest)
                        else:
                            failed[track] = result
                            statuses[track] = "failed"
                            worker_manifest.update(status="failed",
                                error=worker_manifest.get("error") or
                                      f"worker exited with status {result}",
                                failed_utc=worker_manifest.get("failed_utc") or _utc(),
                                worker_exit_code=result,
                                completed_updates=self._completed_updates(track))
                            _atomic_json(self.track_dirs[track] / "run.json", worker_manifest)
                            if abort_started is None:
                                abort_started = time.monotonic()
                                _atomic_json(abort_file, {
                                    "schema_version": 1,
                                    "supervisor_id": self.supervisor_id,
                                    "invocation_id": self.supervisor_id,
                                    "cycle": cycle,
                                    "protocol_id": self.protocol_id,
                                    "source_identity_sha256": self.source_hash,
                                    "failed_track": track,
                                    "failed_exit_code": result,
                                    "created_utc": _utc()})
                    elif self.stop_file.exists():
                        statuses[track] = "stopped"
                    else:
                        try:
                            worker_manifest = json.loads(
                                (self.track_dirs[track] / "run.json").read_text())
                        except (OSError, json.JSONDecodeError):
                            worker_manifest = {}
                        worker_status = worker_manifest.get("status")
                        if track in force_terminated:
                            statuses[track] = "aborted_due_to_sibling_failure"
                            worker_manifest.update(status=statuses[track],
                                sibling_abort_file=str(abort_file),
                                sibling_abort_invocation_id=self.supervisor_id,
                                sibling_abort_reap="process_group_killed",
                                completed_updates=self._completed_updates(track))
                            _atomic_json(self.track_dirs[track] / "run.json", worker_manifest)
                        elif abort_started is not None and worker_status == "aborted_due_to_sibling_failure":
                            statuses[track] = worker_status
                            worker_manifest.update(sibling_abort_file=str(abort_file),
                                sibling_abort_invocation_id=self.supervisor_id,
                                sibling_abort_marker=json.loads(abort_file.read_text()))
                            _atomic_json(self.track_dirs[track] / "run.json", worker_manifest)
                        else:
                            completed = self._completed_updates(track)
                            if completed < target_update:
                                statuses[track] = "failed"
                                failed[track] = result
                                worker_manifest.update(status="failed",
                                    error=(worker_manifest.get("error") or
                                        f"worker exited successfully at update {completed}, below matched target {target_update}"),
                                    worker_exit_code=result, completed_updates=completed,
                                    failed_utc=worker_manifest.get("failed_utc") or _utc())
                                _atomic_json(self.track_dirs[track] / "run.json", worker_manifest)
                            else:
                                self._verify_worker_provenance(track, target_update,
                                    task_assignment, worker_manifest)
                                statuses[track] = "complete"
                                self.completed_chunks += 1
                    self._publish(cycle=cycle,
                        phase="stopping" if self.request_stop else "training",
                        statuses=statuses, error=failed or None)
                if abort_started is None and not self.request_stop:
                    fill_free_devices()
                if abort_started is not None and processes and \
                        time.monotonic() - abort_started >= 1800.:
                    force_terminated.update(processes)
                    unreaped = _reap_worker_groups(processes,
                        graceful_seconds=0., term_seconds=10., kill_seconds=10.)
                    for track, process in list(processes.items()):
                        if process.poll() is not None:
                            logs[track].close()
                            del processes[track]
                            self.worker_pids.pop(track, None)
                            self._refresh_progress(track)
                            statuses[track] = "aborted_due_to_sibling_failure"
                            try:
                                worker_manifest = json.loads(
                                    (self.track_dirs[track] / "run.json").read_text())
                            except (OSError, json.JSONDecodeError):
                                worker_manifest = {}
                            worker_manifest.update(status=statuses[track],
                                sibling_abort_file=str(abort_file),
                                sibling_abort_invocation_id=self.supervisor_id,
                                sibling_abort_reap="process_group_killed",
                                worker_exit_code=process.returncode,
                                completed_updates=self._completed_updates(track))
                            _atomic_json(self.track_dirs[track] / "run.json", worker_manifest)
                    if unreaped:
                        raise RuntimeError(
                            f"worker process groups remained after bounded sibling-abort reap: {unreaped}")
                self._publish(cycle=cycle, phase="stopping" if self.request_stop else "training",
                              statuses=statuses, error=failed or None)
                self._poll_evaluator(cycle, statuses,
                    occupied_devices={self.configs[track].device
                                      for track in processes})
                if processes or self.evaluator is not None:
                    time.sleep(.5)
        except BaseException:
            # Child processes are detached into their own sessions. On setup
            # failure, persist stop intent and bound process-group reaping before
            # releasing the output-root lock.
            stop_file_ready = False
            try:
                self.stop_file.parent.mkdir(parents=True, exist_ok=True)
                self.stop_file.touch(exist_ok=True)
                stop_file_ready = True
            except OSError:
                pass
            unreaped = _reap_worker_groups(processes,
                graceful_seconds=30. if stop_file_ready else 0.)
            if unreaped:
                raise RuntimeError(f"worker process groups remained after bounded reap: {unreaped}")
            raise
        finally:
            for track, process in list(processes.items()):
                self.worker_pids.pop(track, None)
                self._refresh_progress(track)
            for log in logs.values():
                if not log.closed:
                    log.close()
        for track in pending:
            if statuses.get(track) == "queued":
                statuses[track] = ("stopped" if self.request_stop else
                                   "aborted_due_to_sibling_failure" if abort_started is not None
                                   else "queued")
                if statuses[track] in ("stopped", "aborted_due_to_sibling_failure"):
                    try:
                        manifest = json.loads((self.track_dirs[track] / "run.json").read_text())
                    except (OSError, json.JSONDecodeError):
                        manifest = {}
                    manifest.update(schema_version=1, run_id=self.run_ids[track],
                        track=track, config=asdict(self.configs[track]),
                        sources=self.sources, source_identity_sha256=self.source_hash,
                        protocol_id=self.protocol_id, status=statuses[track],
                        completed_updates=self._completed_updates(track),
                        cancelled_utc=_utc())
                    if abort_started is not None:
                        manifest.update(sibling_abort_file=str(abort_file),
                            sibling_abort_invocation_id=self.supervisor_id)
                    _atomic_json(self.track_dirs[track] / "run.json", manifest)
        if pending:
            self._write_queued_manifests(statuses)
            self._publish(cycle=cycle,
                phase="stopping" if self.request_stop else "training",
                statuses=statuses, error=failed or None)
        if failed:
            raise RuntimeError(f"training workers failed: {failed}")

    def _queue_evaluation(self, cycle):
        if source_identity() != self.sources:
            raise RuntimeError("source identity changed before paired evaluation")
        suffix = f"cycle-{cycle:06d}-{uuid.uuid4().hex[:8]}"
        output = self.evaluation_output / f"{suffix}.json"
        snapshot_dir = self.evaluation_output / "checkpoint-snapshots" / suffix
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        latest = {track: self.track_dirs[track] / "latest.pt" for track in MECO_TRACKS}
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
               "seed": (self.evaluation_episode_seeds[0]
                   if self.evaluation_episode_seeds is not None else
                   self.evaluation_seed + (cycle - 1) * self.evaluation_episodes),
               "episode_seeds": self.evaluation_episode_seeds,
               "evaluation_binding": self.evaluation_binding}
        # Bound queued snapshot storage: preserve the active evaluation and
        # replace any older pending snapshot with the newest matched cycle.
        for stale in self.pending_evaluations:
            shutil.rmtree(stale["checkpoints"][MECO_TRACKS[0]].parent,
                          ignore_errors=True)
        self.pending_evaluations = [job]

    def _start_next_evaluation(self, *, occupied_devices=()):
        if self.evaluator is not None or not self.pending_evaluations or self.request_stop:
            return
        if self.evaluation_device in occupied_devices:
            return
        job = self.pending_evaluations.pop(0)
        command = _evaluation_command(checkpoints=job["checkpoints"], scenario=self.scenario,
            categories=self.evaluation_categories, episodes=self.evaluation_episodes,
            seed=job["seed"], device=self.evaluation_device, output=job["output"],
            prediction_mode=self.evaluation_prediction_mode, split=self.evaluation_split,
            evaluation_protocol=self.evaluation_protocol_path,
            evaluation_case=self.evaluation_case,
            episode_seeds=job["episode_seeds"])
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

    def _poll_evaluator(self, cycle, statuses, *, occupied_devices=()):
        self._observe_stop()
        if self.evaluator is None:
            self._start_next_evaluation(occupied_devices=occupied_devices)
        if self.evaluator is None:
            return
        result = self.evaluator.poll()
        if result is None:
            return
        self.evaluator_log.close()
        job = self.evaluator_job
        output = job["output"]
        try:
            evaluator_status = json.loads(Path(str(output) + ".status.json").read_text())
        except (OSError, json.JSONDecodeError):
            evaluator_status = {}
        completion = {"cycle": job["cycle"],
            "output_file": output.name, "status_file": output.name + ".status.json",
            "return_code": result,
            "evaluation_id": evaluator_status.get("evaluation_id"),
            "protocol_id": evaluator_status.get("protocol_id"),
            "progress": evaluator_status.get("progress"),
            "seed": job.get("seed"),
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "split": self.evaluation_split,
            "evaluation_case": self.evaluation_binding,
            "exact_episode_seeds": self.evaluation_episode_seeds}
        self.evaluator = None
        self.evaluator_log = None
        self.evaluator_job = None
        completion["status"] = "stopped" if self.request_stop else (
            "complete" if result == 0 else "failed")
        self.last_evaluation = completion
        _append_evaluation_history(self.evaluation_output,
            cycle=job["cycle"], output=output, completion=completion,
            checkpoints=job["checkpoints"])
        self._start_next_evaluation(occupied_devices=occupied_devices)
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
            if statuses.get(track) not in ("queued", "stopped"):
                continue
            run_dir = self.track_dirs[track]
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
        if remaining and not self.request_stop:
            self._run_training_batch(remaining, cycle, statuses, target_update,
                                     task_assignment)
            self._poll_evaluator(cycle, statuses)
        if not self.request_stop:
            wrong = {track: self._completed_updates(track) for track in order
                     if self._completed_updates(track) != target_update}
            if wrong:
                raise RuntimeError(f"track updates did not reach matched target {target_update}: {wrong}")

    def _claim_fresh_manifest_roots(self):
        if self.manifest is None:
            self.output.mkdir(parents=True, exist_ok=True)
            return
        if self.output.exists() or self.evaluation_output.exists():
            raise ValueError("manifest campaign roots must remain absent until device leases are acquired")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.output.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise ValueError("manifest campaign root was claimed by another process") from exc
        if self.evaluation_output.exists():
            try:
                self.output.rmdir()
            except OSError:
                pass
            raise ValueError("manifest evaluation root appeared while claiming the campaign root")
        try:
            self.evaluation_output.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            try:
                self.output.rmdir()
            except OSError:
                pass
            raise ValueError("manifest evaluation root was claimed by another process") from exc
        except OSError:
            try:
                self.output.rmdir()
            except OSError:
                pass
            raise

    def run(self):
        from .meco_device_lease import lease_devices
        device_lease = lease_devices(tuple(sorted(set(self.devices) | {self.evaluation_device})),
            owner=(f"supervisor:{self.manifest_digest}" if self.manifest_digest else "supervisor"))
        device_lease.acquire()
        try:
            self._claim_fresh_manifest_roots()
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
        finally:
            device_lease.release()

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
    parser.add_argument("--campaign-manifest", help="source-bound isolated campaign manifest")
    parser.add_argument("--diagnostic-launch", action="store_true",
        help="explicitly start a fresh diagnostic campaign from a validated manifest; never qualifies")
    parser.add_argument("--evaluation-output", default=str(ROOT / "evaluations" / "meco-five"))
    parser.add_argument("--devices", required=True,
        help="comma-separated explicit indexed CUDA/ROCm devices; at most one trainer runs on each device")
    parser.add_argument("--scenario", default="pickup_simple")
    parser.add_argument("--category", default="single")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-updates", type=int, default=100)
    parser.add_argument("--stage-exposure-chunks", type=int, default=10,
        help="chunks exposed per scenario before its dependants can enter training mixture")
    parser.add_argument("--phase4-train-exposure", action="store_true",
        help="opt in to balanced Phase 4 train-split adversarial assignments after DAG prerequisites")
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
    parser.add_argument("--evaluation-device", required=True,
        help="explicit indexed CUDA/ROCm device for the paired evaluator")
    parser.add_argument("--qualification-protocol",
        help="frozen qualification protocol used to constrain scheduled scenario/category/seeds")
    parser.add_argument("--evaluation-protocol",
        help="separate frozen protocol for automatic held-out evaluation; selects validation only")
    parser.add_argument("--evaluation-split", choices=("diagnostic", "validation", "test"),
        default="diagnostic",
        help="automatic final-test is forbidden; use an explicit offline evaluation command")
    parser.add_argument("--evaluation-case-json",
        help="JSON object selecting an exact validation stratum when protocol cases are ambiguous")
    parser.add_argument("--warm-start-artifacts", type=json.loads, default=None,
        help="JSON object mapping every MECO track to a validated weights-only artifact")
    parser.add_argument("--cycles", type=int, default=0,
        help="number of complete five-track chunks; zero continues until stopped")
    parser.add_argument("--stop-file")
    return parser


def main(argv=None):
    import sys
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--campaign-manifest" in raw_argv:
        allowed = {"--campaign-manifest", "--diagnostic-launch"}
        flags = [item for item in raw_argv if item.startswith("--")]
        if any(item not in allowed for item in flags):
            raise ValueError("manifest mode derives all configuration; CLI overrides are forbidden")
        parser = argparse.ArgumentParser(description="Launch a source-bound MECO diagnostic campaign")
        parser.add_argument("--campaign-manifest", required=True)
        parser.add_argument("--diagnostic-launch", action="store_true")
        args = parser.parse_args(raw_argv)
        manifest_path = Path(args.campaign_manifest).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text())
        from .meco_training_manifest import validate_campaign_manifest
        validation = validate_campaign_manifest(manifest)
        campaign = manifest["campaign"]
        status = {"valid": True, "manifest_sha256": validation["canonical_sha256"],
            "campaign_id": campaign["id"], "campaign_root": campaign["root"],
            "phase2_ready": validation["phase2_ready"], "phase2_complete": False,
            "qualification_eligible": False,
            "diagnostic_launch_requested": args.diagnostic_launch,
            "launch_ready": False}
        if not args.diagnostic_launch:
            status["blocker"] = "pass --diagnostic-launch to start an isolated fresh diagnostic campaign"
            print(json.dumps(status, indent=2))
            return status
        if not validation["implementation_ready"] or not validation["valid"]:
            raise ValueError("validated manifest is not implementation-ready")
        cfg = manifest["tracks"][MECO_TRACKS[0]]["config"]
        ev = manifest["evaluation_protocol"]
        devices = tuple(manifest["devices"]["training_pool"])
        supervisor = MecoSupervisor(output=campaign["root"],
            evaluation_output=campaign["evaluation_output"], devices=devices,
            scenario=cfg["scenario_id"], category=cfg["category"], seed=cfg["seed"],
            chunk_updates=cfg["updates"], rollout_steps=cfg["rollout_steps"],
            epochs=cfg["epochs"], action_ticks=cfg["action_ticks"],
            learning_rate=cfg["learning_rate"], gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"], clip_coef=cfg["clip_coef"],
            entropy_coef=cfg["entropy_coef"], value_coef=cfg["value_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            route_cost_weight=cfg["route_cost_points_per_second"],
            p_mode=cfg["p_mode"], p_cost_scale=cfg["p_cost_scale"],
            p_points_per_grid_edge=cfg["p_points_per_grid_edge"],
            prediction_mode=cfg["prediction_mode"], candidate_k=cfg["candidate_k"],
            evaluation_categories=(ev["category"],), evaluation_episodes=len(ev["episode_seeds"]),
            evaluation_seed=ev["episode_seeds"][0],
            evaluation_device=manifest["devices"]["evaluation"], cycles=0, stop_file=None,
            qualification_protocol=manifest["qualification_protocol"]["path"],
            evaluation_protocol=ev["path"], evaluation_split=ev["split"],
            campaign_manifest=manifest)
        result = supervisor.run()
        print(json.dumps(result, indent=2))
        return result
    args = _parser().parse_args(raw_argv)
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
        evaluation_protocol=args.evaluation_protocol,
        evaluation_split=args.evaluation_split,
        evaluation_case=(json.loads(args.evaluation_case_json)
            if args.evaluation_case_json else None),
        warm_start_artifacts=args.warm_start_artifacts,
        phase4_train_exposure=args.phase4_train_exposure)
    print(json.dumps(supervisor.run(), indent=2))


if __name__ == "__main__":
    main()
