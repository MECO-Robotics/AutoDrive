"""Matched PPO training and seeded simulator evaluation for MECO A/C tracks.

This is an opt-in research trainer. It never changes the production controller
or relaxes the simulator's existing action legality and safety checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
from functools import wraps
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

import torch
from torch import nn

from .curriculum_scenarios import make_curriculum_env, reset_curriculum_scenario
from .curriculum_training import source_identity, source_tree_identity
from .training_curriculum import get_scenario
from .meco_candidates import (CANDIDATE_BASE_FEATURE_COUNT, CANDIDATE_BOUNDARY, CandidateCommitment,
    ROUTE_COST_POINTS_PER_SECOND,
    RouteFeasibleCandidates, build_route_feasible_candidates)
from .meco_pose_graph import LegalPoseGraph, build_legal_pose_graph
from .meco_path_penalty import (PENALTY_MAP_PROJECTION, PENALTY_MAP_UNITS,
    PENALTY_PATH_UNITS, plan_a_ppo_contract_metadata,
    integrate_frozen_routes, project_pose_penalty)
from .meco_tracks import (MECO_TRACKS, MECO_CANDIDATE_RAW_FEATURE_COUNT,
    make_meco_track, ProductionMecoAdapter, MecoInferenceRequest,
    MECO_INPUT_NAMES, MECO_INFERENCE_SCHEMA, MECO_MAX_INFERENCE_AGE_S,
    candidate_policy_logits, categorical_from_candidate_logits,
    ordered_candidate_mask_digest,
    managed_meco_adapters,
    sample_candidate_action, score_candidates, build_safe_action_candidates,
    candidate_digest, opponent_forecast, opponent_risk_penalty, prediction_contract, PREDICTION_MODES,
    PREDICTION_MAX_SPEED_MPS, PREDICTION_MAX_AGE_S,
    PREDICTION_THREAT_RADIUS_M, PREDICTION_MAX_PENALTY_POINTS,
    PREDICTION_MODEL_ID, MULTIMODEL_HYPOTHESES, MULTIMODEL_MAX_HORIZON_S,
    MULTIMODEL_TURN_RATE_MIN_RAD_S, MULTIMODEL_TURN_RATE_MAX_RAD_S,
    MULTIMODEL_DAMPING_PER_S)
from .meco_plan_a_policy import (PlanAActionSupport, build_plan_a_support,
    gather_pose_behavior_logits, graph_path_integral, plan_a_support_digest,
    PlanAEmptySupportError, SUPPORT_ORDER)
from .meco_metric_ledger import MecoEpisodeMetricLedger
from .meco_input_provenance import input_contract_digest, validate_serialized_inputs
from .meco_phase4_runtime_adapter import (phase4_factor_exposure,
    phase4_pair_contact_event_record, route_remaining_distance_m,
    phase4_route_completion_record, phase4_route_condition,
    phase4_route_condition,
    phase4_selected_goal_endpoint_completion_record,
    FirstRouteProgressInterval,
    PHASE4_CONTACT_ROUTE_GOAL_TOLERANCE_RESOLUTION_FACTOR,
    PHASE4_ROUTE_OUTCOME_FACTOR_IDS)


ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_DT = .02
DEFAULT_CATEGORIES = {"pickup_simple": ("single", "cluster")}
CHUNK_RESET_POLICY = ("reset environment and GRU hidden state at each invocation chunk; "
    "shared scheduled task seed initializes the episode; subsequent episode seeds "
    "are task seed + global episode index")
TRACK_LOSS_RESET_POLICY = {
    "id": "opponent-track-loss-v1",
    "signal": "opponent-confidence-and-signed-time observation raster channels 4 and 5",
    "available_when": "any cell has confidence > 0 and signed_time >= -2.0 seconds",
    "reset_when": "previous decision had an available opponent track and current decision does not",
    "episode_start": "GRU hidden state is already zero; tracking availability starts unknown",
}
TRACK_LOSS_STALE_SECONDS = 2.0
PRODUCTION_INFERENCE_MAX_AGE_S = MECO_MAX_INFERENCE_AGE_S


def _opponent_tracking_available(raster):
    """Use only the shared observation raster to apply the protocol stale-time rule."""
    confidence = raster[:, 4]
    signed_time = raster[:, 5]
    return bool(((confidence > 0.) &
                 (signed_time >= -TRACK_LOSS_STALE_SECONDS)).any().item())


def _tracking_reset_hidden(hidden, previously_available, raster):
    available = _opponent_tracking_available(raster)
    reset = previously_available is True and not available
    return (None if reset else hidden), available, reset


@dataclass(frozen=True)
class MecoTrainConfig:
    track: str
    scenario_id: str = "pickup_simple"
    category: str = "single"
    updates: int = 1000
    rollout_steps: int = 32
    epochs: int = 4
    action_ticks: int = 12
    seed: int = 7
    device: str | None = None
    learning_rate: float = 3e-4
    gamma: float = .993
    gae_lambda: float = .95
    clip_coef: float = .2
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = .5
    route_cost_points_per_second: float = ROUTE_COST_POINTS_PER_SECOND
    p_mode: str = "learned"
    p_cost_scale: float = 1.0
    p_points_per_grid_edge: float = .05
    prediction_mode: str = "cv"
    candidate_k: int = 16
    output: str = ""

    def validate(self):
        from .meco_device import require_indexed_cuda_device
        if self.track not in MECO_TRACKS:
            raise ValueError(f"track must be one of {MECO_TRACKS}")
        definition = get_scenario(self.scenario_id)
        if self.category not in definition.required_categories:
            raise ValueError("category is not registered for scenario")
        for name in ("updates", "rollout_steps", "epochs", "action_ticks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.candidate_k) is not int or self.candidate_k not in (8, 16, 32):
            raise ValueError("candidate_k must be one of 8, 16, or 32")
        if self.p_mode not in ("learned", "p0"):
            raise ValueError("p_mode must be 'learned' or 'p0'")
        if self.prediction_mode not in PREDICTION_MODES:
            raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
        require_indexed_cuda_device(self.device, purpose="MECO training")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_coef",
                     "entropy_coef", "value_coef", "max_grad_norm",
                     "route_cost_points_per_second", "p_cost_scale",
                     "p_points_per_grid_edge"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate <= 0 or self.gamma > 1 or self.gae_lambda > 1 or self.max_grad_norm <= 0:
            raise ValueError("invalid optimizer or discount configuration")


class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(71, 128), nn.Tanh(),
                                 nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, globals_, hidden):
        return self.net(torch.cat((globals_, hidden), -1)).squeeze(-1)


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, allow_nan=False)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_torch(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _model_id(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _candidate_replay_digest(table, route_cost, risk_cost):
    payload = {"table_digest": candidate_digest(table),
        "route_cost": route_cost.detach().cpu().reshape(-1).tolist(),
        "risk_cost": risk_cost.detach().cpu().reshape(-1).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _tensor_evidence(tensor):
    """Return stable identity and runtime metadata for a saved policy tensor."""
    if not isinstance(tensor, torch.Tensor):
        return {"present": False}
    value = tensor.detach().contiguous()
    digest = hashlib.sha256(value.cpu().reshape(-1).view(
        torch.uint8).numpy().tobytes()).hexdigest()
    finite = (bool(torch.isfinite(value).all())
              if value.is_floating_point() or value.is_complex() else True)
    return {"present": True, "sha256": digest, "shape": list(value.shape),
        "dtype": str(value.dtype), "device": str(value.device),
        "finite": finite}


def _cpu_replay_value(value):
    """Copy the exact C1 replay inputs into a portable CPU artifact."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{item.name: _cpu_replay_value(getattr(value, item.name))
                              for item in fields(value)})
    if isinstance(value, dict):
        return {key: _cpu_replay_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_replay_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_replay_value(item) for item in value)
    return value


def _json_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _logits_evidence(tensor):
    if not isinstance(tensor, torch.Tensor):
        return None
    return [_json_number(value) for value in tensor.detach().reshape(-1).cpu().tolist()]


def _candidate_identity_evidence(candidate_set):
    if candidate_set is None:
        return None
    table = candidate_set.table
    features = table.features
    canonical = json.dumps(features, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    return {"candidate_ids": list(table.candidate_ids),
        "valid_mask": [bool(value) for value in table.valid_mask],
        "feature_names": list(table.feature_names),
        "features_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def _candidate_replay_fingerprints(row, model, *, model_parameters_sha256=None):
    candidate_set = row.get("candidate_set")
    if model_parameters_sha256 is None:
        model_parameters_sha256 = _model_id(model)
    return {"raster": _tensor_evidence(row.get("raster")),
        "globals": _tensor_evidence(row.get("globals")),
        "hidden": _tensor_evidence(row.get("hidden_before")),
        "candidate_features": _candidate_identity_evidence(candidate_set),
        "candidate_digest": (candidate_digest(candidate_set.table)
            if candidate_set is not None else None),
        "candidate_replay_digest": row.get("candidate_replay_digest"),
        "route_cost": _tensor_evidence(row.get("route_cost")),
        "risk_cost": _tensor_evidence(row.get("risk_cost")),
        "model_parameters_sha256": model_parameters_sha256}


def _logprob_replay_diagnostic(track, model, row, replay, replay_logprob,
                               *, update_number, row_index, config, output,
                               model_parameters_sha256=None):
    saved_logprob = row["logprob"].reshape(())
    replay_scalar = replay_logprob.reshape(())
    absolute_delta = float((replay_scalar - saved_logprob).abs().detach().cpu())
    relative_delta = absolute_delta / max(
        abs(float(saved_logprob.detach().cpu())), 1e-12)
    rollout_logits = row.get("rollout_logits")
    replay_logits = replay.get("logits")
    max_logits_delta = None
    finite_valid_max_logits_delta = None
    per_index_logits_delta = None
    candidate_set = row.get("candidate_set")
    replay_candidate_ids = (list(candidate_set.table.candidate_ids)
                            if candidate_set is not None else None)
    replay_valid_mask = ([bool(value) for value in candidate_set.table.valid_mask]
                  if candidate_set is not None else [])
    rollout_candidate_ids = row.get("rollout_candidate_ids")
    rollout_valid_mask = row.get("rollout_valid_mask")
    if (isinstance(rollout_logits, torch.Tensor) and
            isinstance(replay_logits, torch.Tensor) and
            rollout_logits.shape == replay_logits.shape):
        rollout_flat = rollout_logits.detach().reshape(-1)
        replay_flat = replay_logits.detach().reshape(-1)
        deltas = (rollout_flat - replay_flat).abs()
        max_logits_delta = float(deltas.max().cpu()) if deltas.numel() else None
        finite = torch.isfinite(rollout_flat) & torch.isfinite(replay_flat)
        if replay_valid_mask and len(replay_valid_mask) == finite.numel():
            finite &= torch.tensor(replay_valid_mask, device=finite.device, dtype=torch.bool)
        finite_deltas = deltas[finite]
        finite_valid_max_logits_delta = (float(finite_deltas.max().cpu())
            if finite_deltas.numel() else None)
        per_index_logits_delta = [
            (_json_number(deltas[index].cpu())
             if bool(finite[index].cpu()) else None)
            for index in range(deltas.numel())]
    named_tensors = {"rollout_logits": rollout_logits,
        "replay_logits": replay_logits, "rollout_logprob": saved_logprob,
        "replay_logprob": replay_scalar, "chosen_action": row.get("action"),
        "raster": row.get("raster"), "globals": row.get("globals"),
        "hidden_before": row.get("hidden_before"),
        "hidden_next": row.get("hidden_next")}
    replay_fingerprints = _candidate_replay_fingerprints(row, model,
        model_parameters_sha256=model_parameters_sha256)
    replay_fingerprints["route_cost_sha256"] = replay_fingerprints[
        "route_cost"]["sha256"]
    replay_fingerprints["risk_cost_sha256"] = replay_fingerprints[
        "risk_cost"]["sha256"]
    repeated_replay = []
    if track in ("C1", "C2"):
        # Only pay for these extra model forwards on an actual strict-guard
        # mismatch. Each pass reuses the identical saved row and hidden state.
        for repeat_index in range(3):
            try:
                repeated = _policy_inputs_for_row(track, model, row,
                    row["hidden_before"], None)
                repeated_logits = repeated["logits"]
                repeated_logprob = torch.distributions.Categorical(
                    logits=repeated_logits).log_prob(
                        row["action"].reshape(1)).reshape(())
                repeated_replay.append({"repeat": repeat_index + 1,
                    "logits": _logits_evidence(repeated_logits),
                    "logprob": _json_number(repeated_logprob.detach().cpu()),
                    "logits_fingerprint": _tensor_evidence(repeated_logits)})
            except Exception as exc:
                repeated_replay.append({"repeat": repeat_index + 1,
                    "error": f"{type(exc).__name__}: {exc}"})
    payload = {"schema_version": 3, "track": track,
        "update": int(update_number), "row_index": int(row_index),
        "episode_id": row.get("episode_id"),
        "absolute_logprob_delta": _json_number(absolute_delta),
        "relative_logprob_delta": _json_number(relative_delta),
        "expected_logprob": _json_number(saved_logprob.detach().cpu()),
        "actual_replay_logprob": _json_number(replay_scalar.detach().cpu()),
        "max_logits_delta": (None if max_logits_delta is None else
                             _json_number(max_logits_delta)),
        "finite_valid_max_logits_delta": (None if finite_valid_max_logits_delta is None
                                          else _json_number(finite_valid_max_logits_delta)),
        "per_index_logits_delta": per_index_logits_delta,
        "rollout_logits_values": _logits_evidence(rollout_logits),
        "replay_logits_values": _logits_evidence(replay_logits),
        "candidate_table": _candidate_identity_evidence(candidate_set),
        "candidate_identity_comparison": {
            "rollout_candidate_ids": rollout_candidate_ids,
            "replay_candidate_ids": replay_candidate_ids,
            "candidate_ids_match": rollout_candidate_ids == replay_candidate_ids,
            "rollout_valid_mask": rollout_valid_mask,
            "replay_valid_mask": replay_valid_mask,
            "valid_mask_match": rollout_valid_mask == replay_valid_mask},
        "rollout_fingerprints": row.get("replay_fingerprints"),
        "replay_fingerprints": replay_fingerprints,
        "repeated_exact_row_replays": repeated_replay,
        "model_parameters_match_rollout": (
            (row.get("replay_fingerprints") or {}).get("model_parameters_sha256") ==
            replay_fingerprints.get("model_parameters_sha256")),
        "chosen_action": int(row["action"].reshape(-1)[0].detach().cpu()),
        "model_training": bool(model.training),
        "rollout_inference_state": row.get("inference_provenance"),
        "replay_inference_state": replay.get("inference_provenance"),
        "input_finite_checks": {name: evidence.get("finite")
            for name, evidence in replay_fingerprints.items()
            if isinstance(evidence, dict) and "finite" in evidence},
        "tensors": {name: _tensor_evidence(tensor)
                    for name, tensor in named_tensors.items()}}
    path = Path(output) / (f"ppo-logprob-replay-failure-{track}-update-"
        f"{update_number:08d}-row-{row_index:04d}.json")
    _atomic_json(path, payload)
    replay_path = path.with_suffix(".pt")
    replay_row = {name: row.get(name) for name in (
        "episode_id", "raster", "globals", "graph", "candidate_set",
        "candidate_digest", "candidate_replay_digest", "route_cost", "risk_cost",
        "action", "logprob", "rollout_logits", "hidden_before", "hidden_next",
        "inference_provenance", "rollout_candidate_ids", "rollout_valid_mask")}
    _atomic_torch(replay_path, {"schema_version": 1, "track": track,
        "update": int(update_number), "row_index": int(row_index),
        "episode_id": row.get("episode_id"),
        "config": asdict(config) if config is not None else None,
        "row": _cpu_replay_value(replay_row),
        "model_state_dict": _cpu_replay_value(model.state_dict()),
        "model_parameters_sha256": replay_fingerprints[
            "model_parameters_sha256"],
        "rollout_logits": _cpu_replay_value(row.get("rollout_logits")),
        "replay_logits": _cpu_replay_value(replay_logits),
        "replay_logprob": _cpu_replay_value(replay_scalar)})
    payload["replay_tensor_artifact"] = str(replay_path)
    _atomic_json(path, payload)
    return path, payload


def _logprob_replay_error(message, track, model, row, replay, replay_logprob,
                          *, update_number, row_index, config, output,
                          model_parameters_sha256=None):
    """Preserve the strict replay failure even if writing evidence fails."""
    try:
        path, diagnostic = _logprob_replay_diagnostic(track, model, row,
            replay, replay_logprob, update_number=update_number,
            row_index=row_index, config=config, output=output,
            model_parameters_sha256=model_parameters_sha256)
    except Exception as exc:
        return RuntimeError(
            f"{message}; diagnostic recording failed: {type(exc).__name__}: {exc}")
    return RuntimeError(
        f"{message}; abs_delta={diagnostic['absolute_logprob_delta']}, "
        f"rel_delta={diagnostic['relative_logprob_delta']}, "
        f"max_logits_delta={diagnostic['max_logits_delta']}, "
        f"action={diagnostic['chosen_action']}, diagnostics={path}")


def _episode_id(run_id: str, episode: int) -> str:
    return f"{uuid.uuid4().hex}:{run_id}-episode-{episode:08d}"


def _canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _validate_paired_calibration_replay_binding(binding, worker, *, device,
        execution_device, scenario_id, split, policy, prediction_mode, seed, qualification_case,
        condition_id, runtime_identity):
    """Check replay bindings before permitting paired calibration export."""
    binding_fields = {"campaign_id", "shard_index", "shard_manifest_id",
        "work_unit_id", "work_unit", "pair_id", "control_id", "source_binding",
        "baseline_freeze_sha256", "partition_sha256",
        "estimand_approval_sha256", "scoring", "control_binding",
        "control_binding_sha256"}
    worker_fields = {"worker_id", "device", "device_index", "device_model",
        "device_uuid", "backend", "backend_version", "torch_version"}
    if not isinstance(binding, dict) or set(binding) != binding_fields:
        raise ValueError("paired calibration binding has an invalid schema")
    if not isinstance(worker, dict) or set(worker) != worker_fields:
        raise ValueError("paired calibration worker identity has an invalid schema")
    if (type(binding["shard_index"]) is not int or binding["shard_index"] < 0
            or not all(isinstance(binding[key], str) and binding[key]
                for key in ("campaign_id", "shard_manifest_id", "work_unit_id",
                    "pair_id", "control_id", "baseline_freeze_sha256",
                    "partition_sha256", "estimand_approval_sha256"))
            or not isinstance(binding["work_unit"], dict)
            or _canonical_digest(binding["work_unit"]) != binding["work_unit_id"]
            or binding["work_unit"].get("pair_id") != binding["pair_id"]
            or binding["work_unit"].get("control_id") != binding["control_id"]
            or binding["work_unit"].get("seed") != seed
            or binding["work_unit"].get("split") != split
            or binding["work_unit"].get("scenario_id") != scenario_id
            or not isinstance(binding["scoring"], dict)
            or binding["scoring"].get("replay_contract_schema") !=
                "dashboard-replay-generation-v1"
            or not isinstance(binding["control_binding"], dict)
            or not isinstance(binding["control_binding_sha256"], str)
            or binding["control_binding"].get("control_binding_sha256") !=
                binding["control_binding_sha256"]
            or binding["control_binding"].get("prediction_contract_sha256") !=
                (_canonical_digest(binding["control_binding"].get("prediction_contract"))
                 if binding["control_binding"].get("prediction_contract") is not None else None)):
        raise ValueError("paired calibration binding does not match this replay")
    if (binding["work_unit"].get("category") not in (None, qualification_case.get("category"))
            or any(binding["work_unit"].get(key) != value
                for key, value in qualification_case.items())):
        raise ValueError("paired calibration work unit differs from frozen case")
    if (not isinstance(worker["worker_id"], str) or not worker["worker_id"]
            or worker["device"] != device
            or type(worker["device_index"]) is not int
            or device != f"cuda:{worker['device_index']}"
            or execution_device != {"device": device, "accelerator_backend": "ROCm"}
            or worker["backend"] != "hip"
            or not isinstance(worker["device_model"], str)
            or "wx9100" not in worker["device_model"].lower()
            or not isinstance(worker["device_uuid"], str) or not worker["device_uuid"]
            or any(worker[key] != runtime_identity.get(key) for key in
                ("device_model", "device_uuid", "backend", "backend_version", "torch_version"))):
        raise ValueError("paired calibration worker identity differs from the assigned ROCm device")
    if binding["work_unit"].get("policy") not in (None, policy):
        raise ValueError("paired calibration work unit differs from replay policy")
    if binding["control_id"] != policy and not binding["control_id"].startswith(policy + "/"):
        raise ValueError("paired calibration control ID differs from replay policy")
    if (binding["control_binding"].get("policy") != policy
            or binding["control_binding"].get("prediction_mode") !=
                prediction_mode):
        raise ValueError("paired calibration control binding differs from replay policy/mode")
    repetition = binding["work_unit"].get("repetition")
    expected_condition_id = (f"paired-cal-{binding['campaign_id'][:12]}-"
        f"{binding['pair_id'][:16]}-{binding['control_id'].replace('/', '-')}-"
        f"r{repetition:04d}" if type(repetition) is int and repetition >= 0 else None)
    if condition_id != expected_condition_id:
        raise ValueError("paired calibration condition ID differs from campaign pair/control")


def _paired_calibration_replay_condition_id(binding):
    if (not isinstance(binding, dict) or not isinstance(binding.get("work_unit"), dict)
            or type(binding["work_unit"].get("repetition")) is not int
            or binding["work_unit"]["repetition"] < 0
            or not all(isinstance(binding.get(key), str) and binding[key]
                for key in ("campaign_id", "pair_id", "control_id"))):
        raise ValueError("paired calibration condition binding is malformed")
    unit = binding["work_unit"]
    return (f"paired-cal-{binding['campaign_id'][:12]}-{binding['pair_id'][:16]}-"
        f"{binding['control_id'].replace('/', '-')}-r{unit['repetition']:04d}")


def _validate_paired_calibration_replay_bindings(bindings, seeds, worker, *,
        device, execution_device, scenario_id, split, policy, prediction_mode,
        qualification_case, runtime_identity):
    if (not isinstance(bindings, dict) or any(type(seed) is not int for seed in bindings)
            or set(bindings) != set(seeds)):
        raise ValueError("paired calibration bindings must cover every exact scored seed")
    for seed, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ValueError("paired calibration per-seed binding must be an object")
        _validate_paired_calibration_replay_binding(binding, worker,
            device=device, execution_device=execution_device,
            scenario_id=scenario_id, split=split, policy=policy,
            prediction_mode=prediction_mode, seed=seed,
            qualification_case=qualification_case,
            condition_id=_paired_calibration_replay_condition_id(binding),
            runtime_identity=runtime_identity)
    values = [bindings[seed] for seed in seeds]
    if any(binding.get("campaign_id") != values[0].get("campaign_id")
            or binding.get("pair_id") != values[0].get("pair_id")
            or binding.get("control_id") != values[0].get("control_id")
            or binding.get("shard_index") != values[0].get("shard_index")
            or binding.get("shard_manifest_id") != values[0].get("shard_manifest_id")
            or binding["work_unit"].get("repetition") !=
                values[0]["work_unit"].get("repetition")
            for binding in values[1:]):
        raise ValueError("paired calibration per-seed bindings mix campaign pair/control units")


def _evaluation_record_digest(record: dict) -> str:
    """Hash the scored row while excluding its later-added replay pointer."""
    return _canonical_digest({key: value for key, value in record.items()
                              if key != "dashboard_replay"})


def _meco_score_ledger_context(env, *, episode_id, seed, scenario_id,
                               source_identity_sha256):
    from .meco_simulated_score_ledger import RULES_ID, RULES_SHA256, sha256

    team_slots = [int(value) for value in env.team_ids.detach().cpu().tolist()]
    reset_generation = int(env._meco_world_reset_generation[0].item())
    score_episode_id = f"{episode_id}:world-0-reset-{reset_generation}"
    schedule = getattr(env, "curriculum_schedule", None)
    effective_config = {
        "rules_id": RULES_ID,
        "rules_sha256": RULES_SHA256,
        "dt": float(env.dt),
        "horizon_ticks": int(env.horizon),
        "alliance_zone_depth_m": float(env.alliance_zone_depth),
        "robot_team_slots": team_slots,
        "fuel_count": int(env.fuel_count),
        "robot_control_modes": list(env.control_modes),
        "robot_types": list(env.robot_types),
        "robot_roles": list(env.robot_roles),
        "robot_fuel_capacities": list(env.robot_fuel_capacity_values),
        "robot_score_interval_s": list(env.robot_score_interval_values),
        "robot_dimensions_m": {
            "length": env.sim.length[0].detach().cpu().tolist(),
            "width": env.sim.width[0].detach().cpu().tolist(),
        },
        "score_intent_source": "effective policy action trace passed to update_fuel_actions",
        "robot_readiness": {
            "dumper_stationary_linear_mps": 0.20,
            "dumper_stationary_angular_radps": 0.20,
            "dumper_aim_tolerance_rad": math.pi / 9,
            "bump_clearance_source": "TensorThreeVsThreeEnv._robots_clear_of_bumps",
        },
        "field_geometry": {
            "field_length_m": float(env.sim.field_length),
            "field_width_m": float(env.sim.field_width),
            "hub_centers_m": env.hub_centers.detach().cpu().tolist(),
            "bump_regions": env.sim.bump_regions.detach().cpu().tolist(),
        },
        "hub_schedule": (None if schedule is None else [list(item) for item in schedule]),
        "behavior_probe": getattr(env, "behavior_probe", None),
        "behavior_probe_hub_active": getattr(env, "behavior_probe_hub_active", None),
    }
    return {
        "episode_id": score_episode_id,
        "evaluation_episode_id": episode_id,
        "seed": int(seed),
        "scenario_id": scenario_id,
        "robot_team_slots": team_slots,
        "world_count": int(env.n),
        "fuel_count": int(env.fuel_count),
        "source_identity_sha256": source_identity_sha256,
        "effective_scoring_configuration": effective_config,
        "effective_scoring_configuration_sha256": sha256(effective_config),
        "terminal_state": "complete",
        "terminal_tick": int(env.horizon),
    }


def _assert_score_estimand_unchanged(path, expected_sha256):
    if path is None:
        return
    current = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if current != expected_sha256:
        raise ValueError("score estimand approval changed during outcome collection")


def _evaluation_action_was_remasked(selected_action: int,
        executed_action_trace) -> bool:
    """Return whether any recorded tick action differs from the selection."""
    return any(value != selected_action for value in executed_action_trace)


def _evaluation_execution_failure_reason(*, action_was_remasked: bool,
        native_preempted: bool, selected_target_lost: bool,
        policy_fallback: bool, fallback_reason: str | None) -> str:
    """Classify observed failures without calling inference fallback a remask."""
    if selected_target_lost:
        return "selected_target_lost"
    if native_preempted:
        return "native_behavior_preempted_selected_goal"
    if action_was_remasked:
        return "action_was_remasked"
    if policy_fallback:
        return fallback_reason or "policy_inference_fallback"
    raise ValueError("execution failure reason requested without a failure")


def _validate_replay_seed_selection(output_dir, replay_seeds, scored_seeds) -> set[int]:
    if (output_dir is None) != (replay_seeds is None):
        raise ValueError("replay_output_dir and explicit replay_seeds must be supplied together")
    if replay_seeds is None:
        return set()
    if (not replay_seeds or len(set(replay_seeds)) != len(replay_seeds) or
            any(type(seed) is not int or seed not in scored_seeds for seed in replay_seeds)):
        raise ValueError("replay_seeds must be a nonempty unique subset of the exact scored seeds")
    return set(replay_seeds)


def _d0_replay_action_ticks(action_ticks: int | None) -> int:
    """Return D0's replay grouping interval without changing evaluation cadence metadata."""
    return 1 if action_ticks is None else action_ticks


def _d0_replay_decision(*, elapsed_ticks: int, action_ticks: int | None,
                        executed_actions: list[int], policy: str, action: int,
                        candidate_k: int, prediction_mode: str,
                        terminal: bool = False) -> dict | None:
    """Build a D0 replay decision at each configured grouping boundary."""
    interval = _d0_replay_action_ticks(action_ticks)
    if elapsed_ticks % interval and not terminal:
        return None
    start_tick = ((elapsed_ticks - 1) // interval) * interval
    return {"kind": "policy_decision", "policy": policy, "action": action,
        "action_source": "production_deterministic", "macro_start_tick": start_tick,
        "macro_end_tick": elapsed_ticks,
        "executed_actions": executed_actions[start_tick:elapsed_ticks],
        "candidate_k": candidate_k, "prediction_mode": prediction_mode}


def _capture_meco_replay_frame(env, tick: int, decision: dict | None) -> dict:
    """Capture renderer state from the evaluated environment without stepping it."""
    audit_valid = getattr(env, "_last_action_audit_valid", None)
    action_audit_available = (audit_valid is not None and bool(audit_valid[0].item()))
    def action_audit(name):
        value = getattr(env, name, None)
        return value[0].tolist() if action_audit_available and value is not None else None

    active = env.piece_active[0].bool()
    pieces = torch.cat((env.piece_pos[0], env.piece_owner[0, :, None]), -1)[active]
    visible = env._current_fuel_visibility[0].bool()
    tracked = env.track_mask[0].bool()
    opponents = env.opponent_valid[0].bool()
    path_lengths = env.planner.last_lengths.reshape(env.n, 6)[0].tolist()
    path_values = env.planner.last_path.reshape(env.n, 6, -1, 2)[0]
    sizes = torch.stack((env.sim.length[0], env.sim.width[0]), -1).flatten().tolist()
    frame = {
        "tick": int(tick),
        # Keep the renderer's established `robots` field as physical output,
        # and spell out its authority next to the policy-facing estimate so
        # replay consumers cannot mistake hidden truth for a model input.
        "robots": env.sim.pose[0].tolist(),
        "physical_truth_pose": env.sim.pose[0].tolist(),
        "policy_pose_estimate": env.policy_pose[0].tolist(),
        "pose_authority": {
            "robots": "physical_truth_for_replay_rendering_only",
            "physical_truth_pose": "evaluator_only",
            "policy_pose_estimate": "learner_localization_input",
        },
        "behavior_mode": "match",
        "robot_teams": env.team_ids.tolist(),
        "robot_control_modes": list(env.control_modes),
        "robot_roles": list(env.robot_roles),
        "robot_actions": env.last_actions[0].tolist(),
        "robot_opponent_visible": opponents.tolist(),
        "robot_detected_opponents": [
            [*env.opponent_pose[0, robot, :2].tolist(),
             float(env.opponent_age[0, robot].item())] if bool(opponents[robot]) else None
            for robot in range(6)],
        "robot_detected_fuel": [
            env.track_pos[0, robot, visible[robot]].tolist() for robot in range(6)],
        "robot_fuel_history": [torch.cat((env.track_pos[0, robot],
            env.track_age[0, robot, :, None]), -1)[tracked[robot] & ~visible[robot]].tolist()
            for robot in range(6)],
        "perception_track_timeout_s": getattr(env, "perception_track_timeout", 2.),
        "perception_fov_degrees": env.perception_fov_degrees,
        "perception_range": env.perception_range,
        "camera_layout": {"height_m": .508,
            "horizontal_fov_degrees": env.perception_fov_degrees,
            "stereo_baseline_m": getattr(env, "stereo_baseline_m", None),
            "mounts": ["intake", "opposite"]},
        "sizes": sizes,
        "adstar_paths": [path[:int(length)].tolist()
                         for path, length in zip(path_values, path_lengths)],
        "fuel_pieces": pieces.tolist(),
        "hub_centers": env.hub_centers.tolist(),
        "hub_active": env.hub_active[0].tolist(),
        "fuel_score_count": env.fuel_score_count[0].tolist(),
        "fuel_acquisition_count": env.fuel_acquisition_count[0].tolist(),
        "match_elapsed": float(env.match_elapsed[0].item()),
        "match_remaining": float(env.match_remaining[0].item()),
        "robot_action_audit_available": action_audit_available,
        "robot_targets": action_audit("_last_audit_targets"),
        "robot_fuel_targets": action_audit("_last_audit_fuel_targets"),
        "robot_route_goals": env.planner.last_goal.reshape(env.n, 6, 2)[0].tolist(),
        "robot_collecting": env._target_collecting[0].tolist(),
        "robot_cluster_counts": action_audit("_last_audit_cluster_count"),
        "meco_policy_decision": decision,
    }
    progress_observer = getattr(env, "_phase4_first_route_progress_observer", None)
    if progress_observer is not None:
        frame["first_route_progress_interval"] = progress_observer.record(trace_limit=1)
    return frame


def _observe_phase4_contact_route_progress(env, observer, *, selected_goal_id,
                                           selected_goal_xy_m):
    """Sample evaluator-only slot-3 truth and current executable planner route."""
    event = phase4_pair_contact_event_record(env.sim)
    planner = env.planner
    route_goal_tolerance = None
    route_fingerprint = None
    try:
        path = planner.last_path.reshape(env.n, 6, planner.last_path.shape[-2], 2)[0, 3]
        length = int(planner.last_lengths.reshape(env.n, 6)[0, 3].item())
        goal = planner.last_goal.reshape(env.n, 6, 2)[0, 3].tolist()
        route_goal_tolerance = (float(planner.resolution) *
            PHASE4_CONTACT_ROUTE_GOAL_TOLERANCE_RESOLUTION_FACTOR)
        path_points = path[:length].tolist()
        normalized_path = [[float(value) for value in point] for point in path_points]
        if (length >= 2 and all(len(point) == 2 and
                all(math.isfinite(value) for value in point)
                for point in normalized_path)):
            route_fingerprint = hashlib.sha256(json.dumps(normalized_path,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        truth_xy = env.sim.pose[0, 3, :2].tolist()
        pinned_xy = (None if selected_goal_xy_m is None else
                     [float(value) for value in selected_goal_xy_m])
        route_matches_goal = (pinned_xy is not None and len(goal) == 2 and
            math.isfinite(route_goal_tolerance) and route_goal_tolerance > 0. and
            all(math.isfinite(float(v)) for v in (*goal, *pinned_xy)) and
            math.dist([float(v) for v in goal], pinned_xy) <= route_goal_tolerance)
        remaining = (route_remaining_distance_m(path_points, truth_xy, pinned_xy,
            goal_tolerance_m=route_goal_tolerance)
                     if route_matches_goal else None)
        if remaining is None:
            route_fingerprint = None
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        truth_xy, pinned_xy, remaining, route_fingerprint = None, None, None, None
        route_matches_goal = False
    world = 0
    counts = event["distinct_contact_substeps_by_world"]
    last_times = event["last_contact_physics_time_s_by_world"]
    last_steps = event["last_contact_physics_substep_by_world"]
    clock = float(env.steps[0].item()) * float(env.dt)
    observer.observe({
        "outer_tick": int(env.steps[0].item()), "outer_tick_time_s": clock,
        "contact_factor_id": event["factor_id"], "contact_pair_slots": event["pair_slots"],
        "distinct_contact_substeps": int(counts[world]),
        "last_contact_physics_substep": last_steps[world],
        "last_contact_physics_time_s": last_times[world],
        "slot3_truth_xy_m": truth_xy,
        "selected_goal_id": selected_goal_id,
        "selected_goal_xy_m": pinned_xy,
        "route_matches_selected_goal": bool(route_matches_goal and remaining is not None),
        "route_goal_tolerance_m": route_goal_tolerance,
        "route_fingerprint_sha256": route_fingerprint,
        "route_remaining_distance_m": remaining,
    })


def _validate_meco_replay_alignment(record: dict, frames: list[dict], *,
                                    final_tick: int) -> None:
    if not frames or int(frames[0].get("tick", -1)) != 0:
        raise ValueError("MECO replay must begin with the evaluation's initial state")
    ticks = [frame.get("tick") for frame in frames]
    if any(type(tick) is not int for tick in ticks) or any(
            right <= left for left, right in zip(ticks, ticks[1:])):
        raise ValueError("MECO replay frame ticks must be strictly increasing integers")
    if ticks[-1] != final_tick or ticks[-1] > int(record["elapsed_ticks"]):
        raise ValueError("MECO replay terminal frame does not align with the scored episode")
    decisions = [frame.get("meco_policy_decision") for frame in frames[1:]]
    if any(not isinstance(decision, dict) or
           decision.get("policy") != record.get("policy") or
           decision.get("macro_end_tick") != frame.get("tick")
           for decision, frame in zip(decisions, frames[1:])):
        raise ValueError("each post-initial replay frame must identify its scored policy decision")
    macro_decisions = [decision for decision in decisions
        if decision.get("kind") == "policy_decision"]
    learned_policy = record.get("policy") in MECO_TRACKS
    executed_policy = record.get("executed_policy_decision_trace")
    if learned_policy and (record.get("executed_policy_decision_trace_schema") !=
            "meco-executed-policy-decision-v1" or not isinstance(executed_policy, list)):
        raise ValueError("learned-track replay requires the versioned executed policy decision trace")
    if executed_policy is not None:
        if (not isinstance(executed_policy, list) or
                len(executed_policy) != len(macro_decisions)):
            raise ValueError("replay macro count differs from executed policy decision trace")
        for decision, row in zip(macro_decisions, executed_policy):
            if (row.get("schema") != "meco-executed-policy-decision-v1" or
                    not isinstance(row.get("candidate_id"), str) or not row["candidate_id"]):
                raise ValueError("executed policy decision trace lacks versioned candidate identity")
            expected = (row.get("selected_action"), row.get("actual_tick_actions"),
                        row.get("start_tick"), row.get("end_tick"),
                        row.get("candidate_id"), row.get("selected_goal_xy_m"))
            actual = (decision.get("action"), decision.get("executed_actions"),
                      decision.get("macro_start_tick"), decision.get("macro_end_tick"),
                      decision.get("candidate_id"), decision.get("goal_xy_m"))
            if actual != expected:
                raise ValueError("replay decision differs from executed policy decision trace")
    route_rows = record.get("executed_planner_route_trace") or []
    if route_rows:
        if len(route_rows) != len(macro_decisions):
            raise ValueError("replay macro count differs from evaluator route trace")
        for decision, row in zip(macro_decisions, route_rows):
            expected = (row.get("selected_action"), row.get("executed_action_trace"),
                        row.get("start_tick"), row.get("end_tick"))
            actual = (decision.get("action"), decision.get("executed_actions"),
                      decision.get("macro_start_tick"), decision.get("macro_end_tick"))
            if actual != expected:
                raise ValueError("replay selected or executed actions differ from evaluator route trace")
    else:
        # Some evaluators do not emit route rows for the policy. Their canonical
        # action trace is tick-level, so compare the complete macro slice.
        executed = record.get("executed_action_trace")
        selected = record.get("selected_action_trace")
        if not isinstance(executed, list) or not executed:
            raise ValueError("evaluator tick action trace is required for replay alignment")
        if len(executed) != final_tick:
            raise ValueError("replay does not cover the complete evaluator action trace")
        if len(macro_decisions) != len(frames) - 1:
            raise ValueError("replay macro count differs from evaluator action trace")
        if isinstance(selected, list) and selected and len(selected) != len(macro_decisions):
            raise ValueError("replay macro count differs from evaluator selected action trace")
        for index, (decision, frame, previous_frame) in enumerate(zip(
                macro_decisions, frames[1:], frames[:-1])):
            start_tick, end_tick = (decision.get("macro_start_tick"),
                                    decision.get("macro_end_tick"))
            trace = decision.get("executed_actions")
            if (type(start_tick) is not int or type(end_tick) is not int or
                    start_tick != previous_frame.get("tick") or
                    end_tick != frame.get("tick") or
                    start_tick < 0 or end_tick <= start_tick or end_tick > len(executed) or
                    not isinstance(trace, list) or trace != executed[start_tick:end_tick]):
                raise ValueError("replay executed action trace differs from evaluator macro slice")
            if isinstance(selected, list) and selected:
                if decision.get("action") != selected[index]:
                    raise ValueError("replay selected action differs from evaluator selected action trace")
            elif decision.get("action") != trace[-1]:
                # D0 chooses anew at every physics tick; its replay action is
                # the actual choice on the final tick of the sampled macro.
                raise ValueError("replay action differs from evaluator tick action trace")


def _selected_policy_candidate_id(prepared, candidate_set, choice):
    """Return the exact selected support/table identity, independent of route capture."""
    if prepared is not None and "support" in prepared:
        support = prepared["support"]
        node = int(support.pair_node_ids[choice].item())
        behavior = int(support.behavior_ids[choice].item())
        descriptor = int(support.descriptor_ids[choice].item())
        return f"node:{node}:behavior:{behavior}:descriptor:{descriptor}"
    if candidate_set is not None and 0 <= int(choice) < len(candidate_set.table.candidate_ids):
        return str(candidate_set.table.candidate_ids[int(choice)])
    return None


def _phase4_route_goal_lifecycle_id(*, action, plan_a_node_id=None,
                                    behavior_id=None, descriptor_id=None,
                                    goal_xy=None, assigned_goal_id=None):
    """Hash a selected objective destination, excluding candidate/path IDs."""
    if type(action) is int and action in range(4):
        # Current sensor slots have no durable allocation generation. A slot
        # may be reused after pickup/respawn, including during assigned-goal
        # continuation, so target-bound attempts always fail closed.
        return None
    if isinstance(assigned_goal_id, str) and assigned_goal_id:
        identity = {"kind": "assigned_goal", "id": assigned_goal_id}
    elif type(action) is int and type(plan_a_node_id) is int and plan_a_node_id >= 0:
        identity = {"kind": "selected_plan_a_objective", "action": action,
            "behavior_id": behavior_id, "descriptor_id": descriptor_id,
            "goal_node_id": plan_a_node_id}
    elif type(action) is int and isinstance(goal_xy, (list, tuple)) and len(goal_xy) == 2:
        values = [float(value) for value in goal_xy]
        if not all(math.isfinite(value) for value in values):
            return None
        identity = {"kind": "selected_route_objective", "action": action,
            "goal_xy_m": values}
    else:
        return None
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return f"phase4-goal-lifecycle-v1:{digest}"


def _snapshot_tensors(snapshot, *, device):
    """Recover the exact six-channel/seven-scalar tensors bound by a snapshot."""
    raster = torch.tensor(snapshot.raster, device=device, dtype=torch.float32).reshape(
        1, *snapshot.raster_shape)
    globals_ = torch.tensor(snapshot.globals, device=device, dtype=torch.float32).reshape(1, 7)
    return raster, globals_


def _route_cost_tensor(candidate_set, weight, *, device):
    eta = [0. if row["eta_s"] is None else float(row["eta_s"])
           for row in candidate_set.route_records]
    return torch.tensor(eta, device=device, dtype=torch.float32)[None] * float(weight)


def _path_geometry(env):
    return {"x": env.planner.x, "y": env.planner.y,
        "nx": env.planner.nx, "ny": env.planner.ny,
        "resolution_m": float(env.planner.resolution),
        "field_length_m": float(env.sim.field_length),
        "field_width_m": float(env.sim.field_width)}


def _path_cost_for_output(track, output, candidate_set, legal_lookup,
                          planner_geometry, config):
    if track not in ("A1", "A2", "A3"):
        return None
    cost_map = project_pose_penalty(track, output["penalty"], legal_lookup,
        planner_geometry, planner_geometry["field_length_m"],
        planner_geometry["field_width_m"], mode=config.p_mode)
    return integrate_frozen_routes(cost_map, candidate_set.route_records,
        planner_geometry["resolution_m"], map_scale=config.p_cost_scale,
        points_per_grid_edge=config.p_points_per_grid_edge)


def _apply_plan_a_path_cost(track, decision, candidate_set, route_cost, risk_cost,
                            legal_lookup, planner_geometry, config, *, record=False):
    path_cost = _path_cost_for_output(track, decision["raw"], candidate_set,
        legal_lookup, planner_geometry, config)
    if path_cost is None:
        return decision, None
    valid = torch.tensor(candidate_set.table.valid_mask, device=path_cost.device,
                         dtype=torch.bool)[None]
    decision["path_cost"] = path_cost
    decision["logits"] = candidate_policy_logits(decision["utility"][None],
        valid, route_cost, risk_cost, path_cost)
    if record:
        for route, cost in zip(candidate_set.route_records,
                               path_cost[0].detach().cpu().tolist()):
            route["ppo_path_penalty_points"] = float(cost)
            route["ppo_path_penalty_units"] = "reward_points"
            route["p_mode"] = config.p_mode
            route["p_map_scale"] = config.p_cost_scale
            route["p_points_per_grid_edge"] = config.p_points_per_grid_edge
    return decision, path_cost


def _plan_a_logits(track, raw, support, graph, raster, env, config, *,
                   route_nodes=None, risk_nodes=None, planner=None,
                   penalty_map=None):
    """Score all frozen node × behavior pairs with current U, P and behavior heads."""
    planner = (env.planner if env is not None else planner)
    logits = gather_pose_behavior_logits(track, raw, graph, raster, support,
        field_length=(float(env.sim.field_length) if env is not None else float(planner.length)),
        field_width=(float(env.sim.field_width) if env is not None else float(planner.width)))
    node_count = graph.node_count
    eta = support.route_cost / max(float(env.sim.speed[0, 3]), .1) if env is not None else None
    route = (eta * float(config.route_cost_points_per_second) if route_nodes is None
             else route_nodes)
    risk = torch.zeros_like(route) if risk_nodes is None else risk_nodes
    if risk_nodes is None and env is not None:
        risk, _ = _plan_a_prediction_risk(env, support, eta,
                                          config.prediction_mode)
    path_cost = torch.zeros_like(route)
    if config.p_mode == "learned":
        if penalty_map is None:
            field_length = (float(env.sim.field_length) if env is not None
                            else float(planner.length))
            field_width = (float(env.sim.field_width) if env is not None
                           else float(planner.width))
            penalty_map = project_pose_penalty(track, raw["penalty"], graph.lookup,
                planner, field_length, field_width, mode="learned")
        ix = torch.floor(support.goal_xy[:, 0] / graph.resolution_m).long().clamp(0, graph.nx - 1)
        iy = torch.floor(support.goal_xy[:, 1] / graph.resolution_m).long().clamp(0, graph.ny - 1)
        per_node = penalty_map[ix, iy]
        graph_integral = graph_path_integral(per_node, support) / graph.resolution_m
        path_cost = graph_integral * config.p_cost_scale * config.p_points_per_grid_edge
    pair_node = support.pair_node_ids
    logits = logits - route[pair_node] - risk[pair_node] - path_cost[pair_node]
    return logits.masked_fill(~support.valid_mask, -torch.inf), path_cost


def _plan_a_prediction_risk(env, support, eta, prediction_mode):
    """Compute Plan A endpoint risk under the shared prediction contract."""
    if prediction_mode not in PREDICTION_MODES:
        raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
    if eta.ndim != 1 or eta.shape[0] != support.goal_xy.shape[0]:
        raise ValueError("Plan A ETA must contain one value per legal graph node")
    pair_valid = support.valid_mask.reshape(-1, 8)
    valid_nodes = pair_valid.any(dim=-1)
    finite_nonnegative_eta = torch.isfinite(eta) & (eta >= 0.)
    if bool((valid_nodes & ~finite_nonnegative_eta).any()):
        raise ValueError("Plan A support-valid node has nonfinite or negative ETA")
    valid_eta = eta[valid_nodes]
    has_valid_endpoint = bool(valid_eta.numel())
    horizon_min = float(valid_eta.min().item()) if has_valid_endpoint else None
    horizon_max = float(valid_eta.max().item()) if has_valid_endpoint else None
    horizon = horizon_max
    risk = torch.zeros_like(eta)
    meta = prediction_contract(prediction_mode, horizon_s=horizon)
    horizon_semantics = (
        "observed_current_pose" if prediction_mode == "current" else
        "per_support_node_eta_clamped_to_multi_model_max_horizon" if prediction_mode == "multi_model_v1" else
        "maximum_eta_over_support_valid_pair_nodes; unreachable_and_masked_nodes_excluded; null_when_no_valid_endpoint")
    valid = bool(env.opponent_valid[0, 3])
    if not valid:
        return risk, {**meta, "horizon_semantics": horizon_semantics,
                      "horizon_min_s": horizon_min, "horizon_max_s": horizon_max,
                      "fallback": "none_no_valid_observation", "used_cv": False}
    position = env.opponent_pose[0, 3, :2]
    if not bool(torch.isfinite(position).all()):
        return risk, {**meta, "horizon_semantics": horizon_semantics,
                      "horizon_min_s": horizon_min, "horizon_max_s": horizon_max,
                      "fallback": "none_nonfinite_observation", "used_cv": False}
    safe_eta = torch.where(finite_nonnegative_eta, eta, torch.zeros_like(eta))
    predicted = position[None].expand(support.goal_xy.shape[0], -1).clone()
    fallback = None if prediction_mode == "current" else "observed_current_pose"
    used_cv = False
    age = float(env.opponent_age[0, 3])
    velocity = env.opponent_velocity[0, 3, :2]
    if (prediction_mode == "cv" and math.isfinite(age) and
            0. <= age <= PREDICTION_MAX_AGE_S and
            bool(torch.isfinite(velocity).all())):
        speed = velocity.norm().clamp_min(1e-8)
        bounded_velocity = velocity * torch.clamp(
            PREDICTION_MAX_SPEED_MPS / speed, max=1.)
        predicted = position[None] + safe_eta[:, None] * bounded_velocity[None]
        fallback = None
        used_cv = True
    used_multimodel = False
    if prediction_mode == "multi_model_v1":
        fresh_motion = (math.isfinite(age) and 0. <= age <= PREDICTION_MAX_AGE_S
                        and bool(torch.isfinite(velocity).all()))
        if fresh_motion:
            t = safe_eta.clamp_max(MULTIMODEL_MAX_HORIZON_S)
            speed = velocity.norm().clamp_min(1e-8)
            bounded_velocity = velocity * torch.clamp(
                PREDICTION_MAX_SPEED_MPS / speed, max=1.)
            omega_observed = (float(env.opponent_velocity[0, 3, 2])
                if env.opponent_velocity.shape[-1] > 2 else 0.)
            if not math.isfinite(omega_observed):
                omega_observed = 0.
            turn_rate = min(MULTIMODEL_TURN_RATE_MAX_RAD_S,
                max(MULTIMODEL_TURN_RATE_MIN_RAD_S, abs(omega_observed)))
            weighted_risk = torch.zeros_like(eta)
            for name, weight in MULTIMODEL_HYPOTHESES:
                if name == "constant_velocity":
                    endpoint = position[None] + t[:, None] * bounded_velocity[None]
                elif name == "damped_velocity":
                    distance_time = -torch.expm1(-MULTIMODEL_DAMPING_PER_S * t) / MULTIMODEL_DAMPING_PER_S
                    endpoint = position[None] + distance_time[:, None] * bounded_velocity[None]
                else:
                    omega = turn_rate * (1. if name == "turn_left" else -1.)
                    angle = omega * t
                    sin_angle, cos_angle = torch.sin(angle), torch.cos(angle)
                    dx = (bounded_velocity[0] * sin_angle + bounded_velocity[1] * (cos_angle - 1.)) / omega
                    dy = (bounded_velocity[0] * (1. - cos_angle) + bounded_velocity[1] * sin_angle) / omega
                    endpoint = position[None] + torch.stack((dx, dy), dim=-1)
                separation = (support.goal_xy - endpoint).norm(dim=-1)
                penalty = PREDICTION_MAX_PENALTY_POINTS * (
                    (PREDICTION_THREAT_RADIUS_M - separation) /
                    PREDICTION_THREAT_RADIUS_M).clamp(0., 1.)
                weighted_risk = weighted_risk + float(weight) * penalty
            risk = torch.where(valid_nodes, weighted_risk, risk)
            fallback = None
            used_multimodel = True
    separation = (support.goal_xy - predicted).norm(dim=-1)
    if prediction_mode != "multi_model_v1" or not used_multimodel:
        risk_values = PREDICTION_MAX_PENALTY_POINTS * (
            (PREDICTION_THREAT_RADIUS_M - separation) /
            PREDICTION_THREAT_RADIUS_M).clamp(0., 1.)
        risk = torch.where(valid_nodes, risk_values, risk)
    age = float(env.opponent_age[0, 3])
    return risk, {**meta, "horizon_semantics": horizon_semantics,
        "fallback": fallback, "used_cv": used_cv,
        "used_multimodel": used_multimodel,
        "risk_semantics": ("weighted_expected_penalty_over_hypothesis_endpoints"
            if prediction_mode == "multi_model_v1" else "single_endpoint_penalty"),
        "observed_age_s": age if math.isfinite(age) else None,
        "horizon_min_s": horizon_min,
        "horizon_max_s": horizon_max}


def _prepare_plan_a_decision(track, model, env, graph, episode_id, hidden, *,
                             route_cost_points_per_second, action_ticks,
                             candidate_k, deterministic, value_head,
                             p_mode, p_cost_scale, p_points_per_grid_edge,
                             prediction_mode="cv", tracking_was_available=None):
    """Build stable full-pose support and sample from its exact masked softmax."""
    # Use the existing observation snapshot seam, but do not build or score a
    # bounded candidate table for Plan A. Candidate K remains a Plan C setting.
    safe = build_safe_action_candidates(env, env._last_obs, world=0, slot=3,
                                        episode_id=episode_id)
    support = build_plan_a_support(env, graph, env._last_obs,
                                   safe.observation.snapshot_id,
                                   require_hip_shortest_path=True)
    raster, globals_ = _snapshot_tensors(safe.observation, device=env.device)
    hidden_before, tracking_available, tracking_reset = _tracking_reset_hidden(
        hidden, tracking_was_available, raster)
    graph_data = _graph_inputs(track, graph, env, raster)
    production_inference_failed = False
    inference_request = None
    inference_handle = None
    response = None
    if isinstance(model, ProductionMecoAdapter):
        hidden_input = (hidden_before if hidden_before is not None else
                        raster.new_zeros((raster.shape[0], 64)))
        model_inputs = {"raster": raster, "globals_": globals_, "hidden": hidden_input}
        model_inputs.update(graph_data or {})
        ordered_inputs = {name: model_inputs[name] for name in MECO_INPUT_NAMES[track]}
        pair_nodes = support.pair_node_ids.detach().cpu().tolist()
        behaviors = support.behavior_ids.detach().cpu().tolist()
        descriptors = support.descriptor_ids.detach().cpu().tolist()
        candidate_ids = tuple(f"node:{node}:behavior:{behavior}:descriptor:{descriptor}"
                              for node, behavior, descriptor in zip(pair_nodes, behaviors, descriptors))
        candidate_mask = tuple(bool(value) for value in
                               support.valid_mask.detach().cpu().tolist())
        request_dispatch = time.monotonic()
        request = None
        if request_dispatch <= safe.observation.capture_monotonic + PRODUCTION_INFERENCE_MAX_AGE_S:
            try:
                request_sequence, generation_id = model.next_request_identity()
                input_contract_sha256 = validate_serialized_inputs(track, ordered_inputs)
                request = MecoInferenceRequest(
                    track=track, snapshot_id=safe.observation.snapshot_id,
                    checkpoint_sha256=model.checkpoint_sha256, protocol_id=model.protocol_id,
                    observation_schema=safe.observation.schema_version,
                    observation_timestamp=safe.observation.timestamp,
                    capture_monotonic=safe.observation.capture_monotonic,
                    dispatch_monotonic=request_dispatch,
                    expires_monotonic=(safe.observation.capture_monotonic
                                       + PRODUCTION_INFERENCE_MAX_AGE_S),
                    input_names=MECO_INPUT_NAMES[track],
                    input_shapes=tuple(tuple(ordered_inputs[name].shape)
                                       for name in MECO_INPUT_NAMES[track]),
                    candidate_ids=candidate_ids,
                    valid_mask=candidate_mask,
                    fallback_index=support.fallback_pair,
                    request_sequence=request_sequence, generation_id=generation_id,
                    candidate_mask_digest=ordered_candidate_mask_digest(
                        candidate_ids, candidate_mask),
                    input_contract_sha256=input_contract_sha256,
                    legal_pose_order_digest=support.support_digest,
                    model_id=f"meco:{track}")
            except Exception:
                request = None
        if request is not None:
            inference_handle = model.submit_inference(request, ordered_inputs)
        if track == "A1":
            utility_shape = (1, 8, *raster.shape[-2:])
        else:
            utility_shape = (1, graph.node_count)
        # Deterministic zero-U/P fallback is prepared while the worker scores.
        raw = {"utility": raster.new_zeros(utility_shape),
               "penalty": raster.new_zeros(utility_shape),
               "behavior_logits": raster.new_zeros((1, 5)),
               "hidden": hidden_input}
    else:
        raw = model(raster=raster, globals_=globals_, hidden=hidden_before,
                    **(graph_data or {}))
    seed_decision = {"raw": raw, "hidden": raw["hidden"]}
    penalty_map = None
    if p_mode == "learned":
        penalty_map = project_pose_penalty(track, raw["penalty"], graph.lookup,
            env.planner, float(env.sim.field_length), float(env.sim.field_width),
            mode="learned")
    config = SimpleNamespace(route_cost_points_per_second=route_cost_points_per_second,
        p_mode=p_mode, p_cost_scale=p_cost_scale,
        p_points_per_grid_edge=p_points_per_grid_edge,
        prediction_mode=prediction_mode)
    logits, path_cost = _plan_a_logits(track, seed_decision["raw"], support,
                                       graph, raster, env, config,
                                       penalty_map=penalty_map)
    eta = support.route_cost / max(float(env.sim.speed[0, 3]), .1)
    route_nodes = eta * float(route_cost_points_per_second)
    risk_nodes, prediction_metadata = _plan_a_prediction_risk(
        env, support, eta, config.prediction_mode)
    if isinstance(model, ProductionMecoAdapter):
        response = (model.poll_inference(request, inference_handle)
                    if request is not None and inference_handle is not None else None)
        response_valid = (response is not None and request is not None
            and response.schema == MECO_INFERENCE_SCHEMA
            and response.track == track
            and response.model_id == request.model_id
            and response.checkpoint_sha256 == model.checkpoint_sha256
            and response.protocol_id == model.protocol_id
            and response.request_sequence == request.request_sequence
            and response.generation_id == request.generation_id
            and response.snapshot_id == request.snapshot_id == safe.observation.snapshot_id
            and response.candidate_mask_digest == request.candidate_mask_digest
            and response.legal_pose_order_digest == support.support_digest
            and request.legal_pose_order_digest == support.support_digest
            and response.observation_timestamp == safe.observation.timestamp
            and response.capture_monotonic == safe.observation.capture_monotonic
            and response.dispatch_monotonic == request.dispatch_monotonic
            and response.expires_monotonic == request.expires_monotonic
            and math.isfinite(response.completed_monotonic)
            and request.dispatch_monotonic <= response.completed_monotonic
            and response.completed_monotonic <= request.expires_monotonic
            and time.monotonic() <= request.expires_monotonic)
        if response_valid:
            raw = response.outputs
            seed_decision = {"raw": raw, "hidden": raw["hidden"]}
            penalty_map = None
            if p_mode == "learned":
                penalty_map = project_pose_penalty(track, raw["penalty"], graph.lookup,
                    env.planner, float(env.sim.field_length), float(env.sim.field_width),
                    mode="learned")
            logits, path_cost = _plan_a_logits(track, raw, support, graph, raster,
                env, config, penalty_map=penalty_map)
        else:
            production_inference_failed = True
        inference_provenance = {"schema": MECO_INFERENCE_SCHEMA,
            "model_id": f"meco:{track}", "checkpoint_sha256": model.checkpoint_sha256,
            "protocol_id": model.protocol_id,
            "observation_timestamp": safe.observation.timestamp,
            "request_sequence": request.request_sequence if request else None,
            "generation_id": request.generation_id if request else None,
            "snapshot_id": safe.observation.snapshot_id,
            "candidate_mask_digest": request.candidate_mask_digest if request else None,
            "legal_pose_order_digest": support.support_digest,
            "captured_monotonic": safe.observation.capture_monotonic,
            "dispatch_monotonic": request.dispatch_monotonic if request else None,
            "completed_monotonic": response.completed_monotonic if response_valid else None,
            "expires_monotonic": request.expires_monotonic if request else None,
            "status": "accepted" if response_valid else "deterministic_fallback",
            "counters": model.inference_counters}
    finite_decision = bool(torch.isfinite(torch.logsumexp(logits[None], dim=-1)).all())
    fallback_used = production_inference_failed or not finite_decision
    fallback_reason = ("policy_inference_fallback" if production_inference_failed else
                       "nonfinite_policy_decision" if not finite_decision else None)
    if not finite_decision:
        action = torch.tensor([support.fallback_pair], device=env.device, dtype=torch.long)
        logprob = torch.zeros((1,), device=env.device, dtype=logits.dtype)
        entropy = torch.zeros_like(logprob)
    else:
        action, logprob, entropy = sample_candidate_action(logits[None],
            deterministic=(True if production_inference_failed else deterministic))
    value = (_value(value_head, globals_, seed_decision["hidden"])
             if value_head is not None else None)
    return {"snapshot": safe.observation, "candidate_set": None,
        "support": support, "raster": raster, "globals": globals_,
        "hidden_before": hidden_before,
        "tracking_available": tracking_available,
        "tracking_reset": tracking_reset,
        "graph": (graph_data or {}), "decision": {**seed_decision, "logits": logits[None]},
        "action": action, "logprob": logprob, "entropy": entropy,
        "hidden_next": seed_decision["hidden"], "value": value,
        "route_cost": route_nodes[None], "risk_cost": risk_nodes[None],
        "path_cost": path_cost, "legal_lookup": graph.lookup,
        "planner_traversal_cost": penalty_map,
        "planner_traversal_cost_scale": (p_cost_scale if penalty_map is not None else 1.),
        "path_geometry": _path_geometry(env), "route_cost_nodes": route_nodes,
        "risk_cost_nodes": risk_nodes, "graph_object": graph,
        "planner_object": env.planner, "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "prediction": prediction_metadata,
        "inference_provenance": inference_provenance if isinstance(model, ProductionMecoAdapter) else None}


def _graph_inputs(track: str, graph: LegalPoseGraph, env, raster):
    if track not in ("A2", "A3"):
        return None
    data = {"node_features": graph.node_features(env, raster),
            "edges": graph.edges, "edge_mask": graph.edge_mask,
            "region_ids": graph.region_ids}
    if hasattr(graph, "coarse_edges"):
        data.update(coarse_edges=graph.coarse_edges,
                    coarse_edge_mask=graph.coarse_edge_mask)
    return data


def _prepare_decision(track, model, env, graph, episode_id, hidden,
                      *, route_cost_points_per_second=ROUTE_COST_POINTS_PER_SECOND,
                      action_ticks=12, candidate_k=16, deterministic=False, value_head=None,
                      record_path_penalty=False,
                      p_mode="learned", p_cost_scale=1., p_points_per_grid_edge=.05,
                      prediction_mode="cv",
                      commitment: CandidateCommitment | None = None,
                      tracking_was_available=None):
    observation = env._last_obs
    if track in ("A1", "A2", "A3"):
        return _prepare_plan_a_decision(track, model, env, graph, episode_id,
            hidden, route_cost_points_per_second=route_cost_points_per_second,
            action_ticks=action_ticks, candidate_k=candidate_k,
            deterministic=deterministic, value_head=value_head,
            p_mode=p_mode, p_cost_scale=p_cost_scale,
            p_points_per_grid_edge=p_points_per_grid_edge,
            prediction_mode=prediction_mode,
            tracking_was_available=tracking_was_available)
    candidate_set = build_route_feasible_candidates(env, observation, graph,
        episode_id=episode_id, action_ticks=action_ticks, candidate_k=candidate_k,
        prediction_mode=prediction_mode,
        commitment=commitment)
    raster, globals_ = _snapshot_tensors(candidate_set.observation, device=env.device)
    hidden_before, tracking_available, tracking_reset = _tracking_reset_hidden(
        hidden, tracking_was_available, raster)
    graph_data = _graph_inputs(track, graph, env, raster)
    path_geometry = _path_geometry(env)
    p_config = SimpleNamespace(p_mode=p_mode, p_cost_scale=p_cost_scale,
        p_points_per_grid_edge=p_points_per_grid_edge)
    if track in ("A1", "A2", "A3"):
        preliminary_cost = _route_cost_tensor(candidate_set,
            route_cost_points_per_second, device=env.device)
        preliminary_risk = candidate_set.risk_cost_points.unsqueeze(0)
        preliminary = score_candidates(track, model, raster=raster,
            globals_=globals_, table=candidate_set.table, hidden=hidden_before,
            graph=graph_data, route_cost=preliminary_cost, risk_cost=preliminary_risk,
            observation_timestamp=candidate_set.observation.timestamp,
            capture_monotonic=candidate_set.observation.capture_monotonic)
        preliminary_map = project_pose_penalty(track, preliminary["raw"]["penalty"],
            graph.lookup, env.planner, path_geometry["field_length_m"],
            path_geometry["field_width_m"], mode=p_mode)
        if p_mode == "learned":
            candidate_set = build_route_feasible_candidates(env, observation, graph,
                episode_id=episode_id, action_ticks=action_ticks,
                candidate_k=candidate_k, learned_traversal_cost=preliminary_map,
                learned_cost_scale=p_cost_scale, prediction_mode=prediction_mode)
        route_cost = _route_cost_tensor(candidate_set,
            route_cost_points_per_second, device=env.device)
        risk_cost = candidate_set.risk_cost_points.unsqueeze(0)
        decision = score_candidates(track, model, raster=raster, globals_=globals_,
            table=candidate_set.table, hidden=hidden_before, graph=graph_data,
            route_cost=route_cost, risk_cost=risk_cost,
            observation_timestamp=candidate_set.observation.timestamp,
            capture_monotonic=candidate_set.observation.capture_monotonic)
        decision, path_cost = _apply_plan_a_path_cost(track, decision,
            candidate_set, route_cost, risk_cost, graph.lookup, path_geometry,
            p_config, record=record_path_penalty)
    else:
        route_cost = _route_cost_tensor(candidate_set,
            route_cost_points_per_second, device=env.device)
        risk_cost = candidate_set.risk_cost_points.unsqueeze(0)
        decision = score_candidates(track, model, raster=raster, globals_=globals_,
            table=candidate_set.table, hidden=hidden_before, graph=graph_data,
            route_cost=route_cost, risk_cost=risk_cost,
            observation_timestamp=candidate_set.observation.timestamp,
            capture_monotonic=candidate_set.observation.capture_monotonic)
        decision["logits"] = _apply_candidate_hysteresis(
            decision["logits"], candidate_set.table)
        path_cost = None
    action, logprob, entropy = sample_candidate_action(decision["logits"],
        deterministic=(True if decision.get("inference_status") ==
                       "deterministic_fallback" else deterministic))
    value = (_value(value_head, globals_, decision["hidden"])
             if value_head is not None else None)
    return {"snapshot": candidate_set.observation, "candidate_set": candidate_set,
            "hidden_before": hidden_before,
            "tracking_available": tracking_available,
            "tracking_reset": tracking_reset,
            "candidate_digest": candidate_digest(candidate_set.table),
            "candidate_replay_digest": _candidate_replay_digest(
                candidate_set.table, route_cost, risk_cost),
            "raster": raster, "globals": globals_, "graph": graph_data,
            "decision": decision, "action": action, "logprob": logprob,
            "entropy": entropy, "hidden_next": decision["hidden"],
            "value": value, "route_cost": route_cost, "risk_cost": risk_cost,
            "path_cost": path_cost, "legal_lookup": graph.lookup,
            "path_geometry": path_geometry}


def _value(value_head, globals_, hidden):
    return value_head(globals_, hidden)


def _apply_candidate_hysteresis(logits, table):
    """Apply the shared committed-task switch cost without changing feasibility."""
    if "commitment_switch_cost" not in table.feature_names:
        return logits
    column = table.feature_names.index("commitment_switch_cost")
    switch = torch.tensor([float(row[column] or 0.) for row in table.features],
                          device=logits.device, dtype=logits.dtype)[None]
    return logits - switch * CANDIDATE_BOUNDARY["commitment_switch_cost_points"]


def _physics_safety_snapshot(env):
    """Read cumulative focal contact ticks at a macro boundary."""
    names = {
        "active_physics_ticks": "safety_active_physics_ticks",
        "robot_contact_ticks": "safety_robot_contact_ticks",
        "field_contact_ticks": "safety_field_contact_ticks",
        "wall_contact_ticks": "safety_wall_contact_ticks",
    }
    snapshot = {}
    for key, attr in names.items():
        value = getattr(env.sim, attr, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 1 or value.shape[1] <= 3:
            return None
        snapshot[key] = int(value[0, 3].detach().item())
    return snapshot


def _run_macro_step(env, action_index: int, action_ticks: int, goal_xy=None, *,
                    goal_heading=None,
                    goal_track_index=None,
                    goal_track_source_xy=None,
                    allow_action7_traversal=False,
                    planner_traversal_cost_override=None,
                    planner_traversal_cost_scale=1.0,
                    planner_force_goal_replan=False,
                    planner_route_trace: list | None = None,
                    safety_counts: dict | None = None,
                    metric_ledger: MecoEpisodeMetricLedger | None = None,
                    contact_progress_observer=None,
                    direct_pursuit_capture=None,
                    trench_interception_capture=None,
                    contact_progress_goal_id=None,
                    return_execution_flags: bool = False):
    actions = torch.zeros((env.n, 6), device=env.device, dtype=torch.long)
    actions[0, 3] = int(action_index)
    total = torch.zeros((), device=env.device)
    ended = timeout = False
    executed_actions = []
    heading_alignment_opportunity = False
    native_goal_preempted = False
    selected_target_lost = False
    selected_target_action_remasked = False
    frozen_traversal_cost = (None if planner_traversal_cost_override is None else
        torch.as_tensor(planner_traversal_cost_override, device=env.device,
                        dtype=env.sim.pose.dtype).detach().clone())
    if safety_counts is not None and "_last_snapshot" not in safety_counts:
        safety_counts["_last_snapshot"] = _physics_safety_snapshot(env)
    for tick_index in range(action_ticks):
        planner_kwargs = {}
        if frozen_traversal_cost is not None:
            planner_kwargs = {
                "planner_traversal_cost_override": frozen_traversal_cost,
                "planner_traversal_cost_override_action": int(action_index),
                "planner_traversal_cost_override_scale": planner_traversal_cost_scale,
            }
        _, _, done, truncated, _ = env.step(actions, capture_info=False,
            planner_goal_override=goal_xy,
            planner_goal_override_action=action_index if goal_xy is not None else None,
            planner_goal_override_target_track=goal_track_index,
            planner_goal_override_target_xy=goal_track_source_xy,
            planner_goal_override_traversal=allow_action7_traversal,
            planner_goal_heading_override=goal_heading,
            planner_force_goal_replan=(planner_force_goal_replan and tick_index == 0),
            **planner_kwargs)
        if direct_pursuit_capture is not None:
            direct_pursuit_capture.observe_physics_step()
        if trench_interception_capture is not None:
            trench_interception_capture.observe_physics_step()
        if contact_progress_observer is not None:
            _observe_phase4_contact_route_progress(env, contact_progress_observer,
                selected_goal_id=contact_progress_goal_id,
                selected_goal_xy_m=goal_xy)
        if planner_route_trace is not None:
            planner = env.planner
            route_length = int(planner.last_lengths[3].detach().item())
            if route_length > 0:
                points = planner.last_path[3, :route_length].detach().cpu().tolist()
                normalized_points = [[float(value) for value in point] for point in points]
                route_sha = hashlib.sha256(json.dumps(normalized_points,
                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                if (not planner_route_trace or
                        planner_route_trace[-1]["route_sha256"] != route_sha):
                    applied = getattr(env,
                        "_last_applied_planner_traversal_signature", None)
                    planner_route_trace.append({"tick_index": tick_index,
                        "route_points_m": normalized_points,
                        "route_sha256": route_sha,
                        "p_map_applied": bool(frozen_traversal_cost is not None and
                            isinstance(applied, dict) and
                            applied.get("map") is frozen_traversal_cost)})
        if metric_ledger is not None:
            metric_ledger.observe_outer_step(int(env.last_actions[0, 3].item()))
        executed_actions.append(int(env.last_actions[0, 3].item()))
        selected_target_lost |= (
            getattr(env, "_last_planner_collection_target_valid", None) is False)
        selected_target_action_remasked |= (
            getattr(env, "_last_planner_collection_action_match", None) is False)
        if safety_counts is not None:
            current = _physics_safety_snapshot(env)
            previous = safety_counts.get("_last_snapshot")
            if current is None or previous is None:
                safety_counts["metrics_available"] = False
                safety_counts["_last_snapshot"] = current
            else:
                for key in ("active_physics_ticks", "robot_contact_ticks",
                            "field_contact_ticks", "wall_contact_ticks"):
                    safety_counts[key] += current[key] - previous[key]
                safety_counts["_last_snapshot"] = current
        if goal_xy is not None:
            current_xy = env.sim.pose[0, 3, :2]
            goal_distance = float((current_xy - torch.as_tensor(
                goal_xy, device=env.device, dtype=current_xy.dtype)).norm())
            heading_alignment_opportunity |= goal_distance <= .04
            if action_index == 4:
                native_goal_preempted |= (
                    int(env.fuel_scored_event[0, 3].item()) > 0 and goal_distance > .04)
            if action_index == 6:
                native_goal_preempted |= (
                    int(env.fuel_passed_event[0, 3].item()) > 0 and goal_distance > .04)
        total += env.curriculum_metrics.reward_delta()[0]
        ended = bool(done[0].item())
        timeout = bool(truncated[0].item())
        if (ended or timeout or selected_target_lost or
                selected_target_action_remasked):
            break
    env._last_obs = env.observe()
    final_heading_error = None
    final_goal_distance = None
    final_heading_target = (goal_heading if goal_heading is not None else
        getattr(env, "_last_planner_collection_goal_heading", None))
    if final_heading_target is not None and not selected_target_lost:
        difference = torch.atan2(
            torch.sin(env.sim.pose[0, 3, 2] - float(final_heading_target)),
            torch.cos(env.sim.pose[0, 3, 2] - float(final_heading_target))).abs()
        final_heading_error = float(difference)
    if goal_xy is not None:
        final_goal_distance = float((env.sim.pose[0, 3, :2] - torch.as_tensor(
            goal_xy, device=env.device, dtype=env.sim.pose.dtype)).norm())
    native_goal_preempted &= final_goal_distance is not None and final_goal_distance > .04
    result = (total, ended, timeout, executed_actions, final_heading_error,
            final_goal_distance, heading_alignment_opportunity, native_goal_preempted,
            selected_target_lost)
    if return_execution_flags:
        return (*result, selected_target_action_remasked)
    return result


def _reset_episode(env, seed: int):
    reset_curriculum_scenario(env, int(seed))
    env._last_obs = env.observe()
    return env._last_obs


def _gae(rewards, dones, values, next_value, gamma, lam):
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for index in reversed(range(len(rewards))):
        next_nonterminal = 1. - dones[index]
        next_v = next_value if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + gamma * next_v * next_nonterminal - values[index]
        last = delta + gamma * lam * next_nonterminal * last
        advantages[index] = last
    return advantages, advantages + values


def _policy_inputs_for_row(track, model, row, hidden, config):
    if track in ("A1", "A2", "A3") and "support" in row:
        raw = model(raster=row["raster"], globals_=row["globals"],
                    hidden=hidden, **(row["graph"] or {}))
        decision = {"raw": raw, "hidden": raw["hidden"]}
        logits, path_cost = _plan_a_logits(track, decision["raw"], row["support"],
            row["graph_object"], row["raster"], None, config,
            route_nodes=row["route_cost_nodes"], risk_nodes=row["risk_cost_nodes"],
            planner=row["planner_object"])
        decision["logits"] = logits[None]
        decision["path_cost"] = path_cost
        return decision
    decision = score_candidates(track, model, raster=row["raster"],
        globals_=row["globals"], table=row["candidate_set"].table,
        hidden=hidden, graph=row["graph"],
        route_cost=row["route_cost"], risk_cost=row["risk_cost"])
    decision["logits"] = _apply_candidate_hysteresis(
        decision["logits"], row["candidate_set"].table)
    if track in ("A1", "A2", "A3"):
        decision, _ = _apply_plan_a_path_cost(track, decision,
            row["candidate_set"], row["route_cost"], row["risk_cost"],
            row["legal_lookup"], row["path_geometry"], config)
    return decision


def _ppo_update(track, model, value_head, optimizer, rows, config, bootstrap_value,
                *, update_number, diagnostic_output):
    rewards = torch.stack([row["reward"] for row in rows])
    dones = torch.tensor([row["done"] for row in rows], device=rewards.device,
                         dtype=rewards.dtype)
    old_logprob = torch.stack([row["logprob"] for row in rows]).detach()
    old_value = torch.stack([row["value"] for row in rows]).detach()
    # Validate exact rollout/replay identity for Plan A before any optimizer
    # step. The support tensors are shared by each row, so digesting catches
    # accidental in-place mutation of pair IDs, descriptors, mask, or paths.
    with torch.no_grad():
        current_model_parameters_sha256 = (
            _model_id(model) if track in ("C1", "C2") else None)
        for row_index, row in enumerate(rows):
            support = row.get("support")
            if support is None:
                continue
            expected_digest = row.get("support_digest")
            actual_digest = plan_a_support_digest(support, row["graph_object"])
            if (not expected_digest or expected_digest != support.support_digest or
                    actual_digest != expected_digest):
                raise RuntimeError("Plan A frozen support changed between rollout and PPO update")
            if row.get("execution_failure", False):
                continue
            replay = _policy_inputs_for_row(track, model, row,
                                            row["hidden_before"], config)
            replay_logprob = categorical_from_candidate_logits(
                replay["logits"]).log_prob(row["action"].reshape(1)).reshape(())
            if not torch.allclose(replay_logprob, row["logprob"].reshape(()),
                                  atol=5e-5, rtol=5e-5):
                raise _logprob_replay_error(
                    "Plan A rollout log-probability does not replay under frozen support",
                    track, model, row, replay, replay_logprob,
                    update_number=update_number, row_index=row_index,
                    config=config, output=diagnostic_output)
        for row_index, row in enumerate(rows):
            if track not in ("C1", "C2") or row.get("candidate_set") is None:
                continue
            expected = row.get("candidate_digest")
            actual = candidate_digest(row["candidate_set"].table)
            if not expected or expected != actual:
                raise RuntimeError("Plan C immutable candidate table changed between rollout and PPO update")
            replay_digest = _candidate_replay_digest(row["candidate_set"].table,
                row["route_cost"], row["risk_cost"])
            if replay_digest != row.get("candidate_replay_digest"):
                raise RuntimeError("Plan C candidate route/risk costs changed between rollout and PPO update")
            replay = _policy_inputs_for_row(track, model, row,
                                            row["hidden_before"], config)
            replay_logprob = categorical_from_candidate_logits(
                replay["logits"]).log_prob(row["action"].reshape(1)).reshape(())
            rollout_fingerprints = row.get("replay_fingerprints")
            replay_fingerprints = _candidate_replay_fingerprints(row, model,
                model_parameters_sha256=current_model_parameters_sha256)
            if (not isinstance(rollout_fingerprints, dict) or
                    rollout_fingerprints != replay_fingerprints):
                raise _logprob_replay_error(
                    "Plan C frozen rollout inputs changed before PPO replay",
                    track, model, row, replay, replay_logprob,
                    update_number=update_number, row_index=row_index,
                    config=config, output=diagnostic_output,
                    model_parameters_sha256=current_model_parameters_sha256)
            if not torch.allclose(replay_logprob, row["logprob"].reshape(()),
                                  atol=5e-5, rtol=5e-5):
                raise _logprob_replay_error(
                    "Plan C rollout log-probability does not replay under frozen candidate table",
                    track, model, row, replay, replay_logprob,
                    update_number=update_number, row_index=row_index,
                    config=config, output=diagnostic_output,
                    model_parameters_sha256=current_model_parameters_sha256)
    with torch.no_grad():
        next_value = (torch.zeros_like(old_value[-1]) if rows[-1]["done"]
                      else bootstrap_value.reshape(()))
    advantage, returns = _gae(rewards, dones, old_value, next_value,
                              config.gamma, config.gae_lambda)
    actor_rows = torch.tensor([not row.get("execution_failure", False) for row in rows],
                              device=advantage.device, dtype=torch.bool)
    actor_advantage = torch.zeros_like(advantage)
    if bool(actor_rows.any()):
        eligible_advantage = advantage[actor_rows]
        actor_advantage[actor_rows] = (eligible_advantage - eligible_advantage.mean()) / \
            eligible_advantage.std(unbiased=False).clamp_min(1e-8)
    policy_losses = []; value_losses = []; entropies = []
    count = len(rows)
    for _ in range(config.epochs):
        order = torch.randperm(count, device=rewards.device)
        for idx in order.tolist():
            row = rows[idx]
            current = _policy_inputs_for_row(track, model, row,
                                             row["hidden_before"], config)
            logits = current["logits"]
            value = _value(value_head, row["globals"], current["hidden"]).reshape(())
            value_loss = .5 * (value - returns[idx]).square()
            if row.get("execution_failure", False):
                # Do not credit a safely remasked fallback as the sampled PPO
                # action. Keep the transition for critic learning and audit.
                loss = config.value_coef * value_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(value_head.parameters()),
                                         config.max_grad_norm)
                optimizer.step()
                policy_losses.append(0.)
                value_losses.append(float(value_loss.detach()))
                entropies.append(0.)
                continue
            dist = categorical_from_candidate_logits(logits)
            action = row["action"].reshape(1)
            logprob = dist.log_prob(action).reshape(())
            entropy = dist.entropy().mean()
            ratio = (logprob - old_logprob[idx]).exp()
            unclipped = ratio * actor_advantage[idx]
            clipped = ratio.clamp(1. - config.clip_coef, 1. + config.clip_coef) * actor_advantage[idx]
            policy_loss = -torch.minimum(unclipped, clipped)
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(value_head.parameters()),
                                     config.max_grad_norm)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy.detach()))
    return {"policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
            "value_loss": sum(value_losses) / max(1, len(value_losses)),
            "entropy": sum(entropies) / max(1, len(entropies)),
            "rollout_return": float(rewards.sum().detach()),
            "rollout_steps": count}


def train_meco(config: MecoTrainConfig, *, resume: str | None = None,
               warm_start: str | None = None,
               qualification_protocol: str | None = None,
               stop_file: str | None = None, abort_file: str | None = None,
               invocation_id: str | None = None,
               run_id_override: str | None = None,
               protocol_id: str | None = None,
               task_assignment: dict | None = None,
               expected_source_sha256: str | None = None,
               native_extension_attestation: dict | None = None):
    config.validate()
    if (abort_file is None) != (invocation_id is None):
        raise ValueError("--abort-file and --invocation-id must be supplied together")
    if invocation_id is not None and (not isinstance(invocation_id, str)
                                      or not invocation_id.strip()):
        raise ValueError("--invocation-id must be a nonempty string")
    if resume is not None and warm_start is not None:
        raise ValueError("--resume and --warm-start are separate initialization modes")
    if warm_start is not None and (qualification_protocol is None or not protocol_id):
        raise ValueError("weights-only warm start requires a frozen qualification protocol and training protocol ID")
    task_assignment = dict(task_assignment or {})
    task_scenario = task_assignment.get("scenario_id", config.scenario_id)
    task_category = task_assignment.get("category", config.category)
    task_seed = task_assignment.get("seed")
    if task_seed is None:
        task_seed = config.seed
    task_definition = get_scenario(task_scenario)
    if task_category not in task_definition.required_categories:
        raise ValueError("scheduled task category is not registered for scenario")
    if type(task_seed) is not int or task_seed < 0:
        raise ValueError("scheduled task seed must be nonnegative")
    from .meco_phase4 import (phase4_training_metadata,
        validate_phase4_training_assignment)
    phase4_factor = validate_phase4_training_assignment(task_assignment)
    phase4_training = phase4_training_metadata(task_assignment)
    qualification_protocol_data = None
    qualification_protocol_id = None
    qualification_protocol_sha256 = None
    if qualification_protocol is not None:
        from .meco_supervised import _load_protocol, validate_training_assignment
        (qualification_protocol_data, qualification_protocol_id,
         qualification_protocol_sha256, _) = _load_protocol(Path(qualification_protocol))
        validate_training_assignment(qualification_protocol_data,
            {"scenario_id": task_scenario, "category": task_category, "seed": task_seed})
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(config.device))
    from .fuel_physics_hip import prepare_native_fuel_extension
    prepare_native_fuel_extension(config.device)
    from .fuel_physics_hip import native_extension_identity
    loaded_native_identity = native_extension_identity()
    if (expected_source_sha256 is None) != (native_extension_attestation is None):
        raise ValueError("expected source hash and native extension attestation must be supplied together")
    sources = source_identity()
    source_sha256 = hashlib.sha256(json.dumps(sources, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    if expected_source_sha256 is not None:
        _validate_supervisor_training_binding(sources, loaded_native_identity,
            expected_source_sha256, native_extension_attestation)
    run_id = run_id_override or uuid.uuid4().hex
    output = Path(config.output or ROOT / "checkpoints" / "meco-five" / config.track)
    if qualification_protocol_data is not None:
        from .meco_final_test_ledger import assert_training_not_released_after_final_test
        source_hash = hashlib.sha256(json.dumps(
            sources, sort_keys=True, allow_nan=False).encode()).hexdigest()
        assert_training_not_released_after_final_test(
            Path(qualification_protocol).resolve(), qualification_protocol_id,
            qualification_protocol_sha256, source_hash)
    output.mkdir(parents=True, exist_ok=True)
    model = make_meco_track(config.track,
        candidate_features=CANDIDATE_BASE_FEATURE_COUNT).to(config.device)
    warm_start_provenance = None
    initialization_record = None
    if warm_start is not None:
        if protocol_id is None:
            raise ValueError("weights-only warm start requires a frozen protocol ID")
        from .meco_supervised import load_warm_start
        warm_start_provenance = load_warm_start(config.track, model, Path(warm_start),
            qualification_protocol=Path(qualification_protocol), expected_sources=sources,
            expected_prediction_mode=config.prediction_mode)
        if (warm_start_provenance.get("artifact_kind") ==
                "weights_only_counterfactual_plan_a_p_route_v1" and
                (config.p_mode != "learned" or
                 config.p_cost_scale != warm_start_provenance.get("p_cost_scale"))):
            raise ValueError(
                "Plan A P route warm start requires matching learned p_mode and p_cost_scale")
        if (warm_start_provenance.get("artifact_kind") ==
                "weights_only_plan_a_joint_u_b_p_v1" and
                (config.p_mode != "learned" or
                 config.p_cost_scale != warm_start_provenance.get("p_cost_scale"))):
            raise ValueError(
                "joint Plan A U+B+P warm start requires matching learned p_mode and p_cost_scale")
        initialization_record = {"mode": "weights_only_supervised_warm_start",
            "warm_start_artifact_sha256": hashlib.sha256(
                Path(warm_start).read_bytes()).hexdigest(),
            "dataset_sha256": warm_start_provenance.get("dataset_sha256"),
            "warm_start_source_identity_sha256": warm_start_provenance.get("source_identity_sha256"),
            "plan_a_p_mode": warm_start_provenance.get("p_mode"),
            "p_route_dataset_sha256": (warm_start_provenance.get("dataset_sha256")
                if warm_start_provenance.get("artifact_kind") ==
                    "weights_only_counterfactual_plan_a_p_route_v1" else None),
            "qualification_protocol_id": warm_start_provenance.get("protocol_id"),
            "qualification_protocol_sha256": warm_start_provenance.get("qualification_protocol_sha256")}
    value_head = ValueHead().to(config.device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(value_head.parameters()),
                                 lr=config.learning_rate)
    update = 0
    episode = 0
    history = []
    resume_rng_torch = None
    resume_rng_python = None
    resume_rng_device = None
    resume_task_progress = None
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("sources") != sources:
            raise ValueError("MECO checkpoint source/schema mismatch; refusing incompatible resume")
        if payload.get("protocol_id") != protocol_id:
            raise ValueError("MECO checkpoint protocol mismatch; refusing incompatible resume")
        saved_config = payload.get("config")
        requested_config = asdict(config)
        if isinstance(saved_config, dict):
            saved_config = dict(saved_config)
            saved_config.pop("updates", None)
        requested_config.pop("updates", None)
        if saved_config != requested_config:
            raise ValueError("MECO immutable run configuration mismatch")
        model.load_state_dict(payload["model"])
        initialization_record = payload.get("initialization") or {"mode": "random_initialization"}
        value_head.load_state_dict(payload["value_head"])
        optimizer.load_state_dict(payload["optimizer"])
        update, episode, history = payload["update"], payload["episode"], payload["history"]
        prior_task = payload.get("task_assignment") or {}
        if (task_assignment.get("assignment_id") is not None
                and prior_task.get("cycle") == task_assignment.get("cycle")
                and prior_task.get("assignment_id") != task_assignment.get("assignment_id")):
            raise ValueError("checkpoint task assignment differs within the same scheduled cycle")
        run_id = payload["run_id"]
        resume_task_progress = payload.get("task_progress")
        if run_id_override and run_id_override != run_id:
            raise ValueError("resume run ID mismatch")
        resume_rng_torch = payload["rng_torch"]
        resume_rng_python = payload["rng_python"]
        resume_rng_device = payload["rng_device"]
    env = make_curriculum_env(task_scenario, task_category, 1,
        config.device, task_seed, phase4_factor=phase4_factor)
    chunk_seed = task_seed
    _reset_episode(env, chunk_seed)
    if resume_rng_torch is not None:
        torch.set_rng_state(resume_rng_torch)
        random.setstate(resume_rng_python)
        if resume_rng_device is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(resume_rng_device, torch.device(config.device))
    try:
        graph = build_legal_pose_graph(env, slot=3)
    except BaseException:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise
    run_manifest = {"schema_version": 1, "run_id": run_id,
        "track": config.track, "config": asdict(config), "task_assignment": task_assignment,
        "phase4_training": phase4_training,
        "updates_semantics": "updates per invocation chunk",
        "protocol_id": protocol_id,
        "input_contract_sha256": input_contract_digest(),
        "configuration_sha256": hashlib.sha256(json.dumps(
            {key: value for key, value in asdict(config).items() if key != "updates"},
            sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "sources": sources,
        "source_identity_sha256": source_sha256,
        "supervisor_source_identity_sha256": expected_source_sha256,
        "supervisor_native_extension_attestation": native_extension_attestation,
        "graph": {"nodes": graph.node_count, "regions": graph.region_count,
                  "resolution_m": graph.resolution_m, "heading_bins": 8},
        "candidate_boundary": CANDIDATE_BOUNDARY,
        "actor_execution_contract": (
            "A1/A2/A3: each macro executes the action sampled at that observation; "
            "the saved PPO log-probability is for that action; a fresh action is "
            "sampled at the next macro boundary. Incomplete goals are not censored. "
            "Sampled fallback, native remask, and native goal preemption are "
            "critic-only and counted separately by reason. C1/C2 semantics unchanged."),
        "plan_a_empty_support_contract": (
            "An observation with no legal node/behavior pair emits a typed state "
            "diagnostic; no action is sampled or executed. The episode is counted "
            "as aborted and reset at the next deterministic task seed. Unconsumed "
            "rollout rows from that episode are discarded before PPO, so no action "
            "or credit is fabricated for the unsupported state. A rollout with no "
            "eligible rows fails the run instead of training on an empty batch."),
        "plan_a_route_execution_contract": (
            "A1/A2/A3 learned mode: freeze the projected P map at the policy decision "
            "and pass it through the full macro to native AD* for focal slot 3 on "
            "every replan and eligible depot retry; other robot routes are unweighted. "
            "P0 passes no map. Map/action/goal binding and map shape/value/scale are "
            "validated. This is an unvalidated implementation seam until a replay "
            "exercises and inspects the resulting route."),
        "prediction_contract": prediction_contract(config.prediction_mode),
        "tracking_loss_reset_policy": TRACK_LOSS_RESET_POLICY,
        "qualification_protocol_id": qualification_protocol_id,
        "qualification_protocol_sha256": qualification_protocol_sha256,
        "candidate_k": config.candidate_k,
        "chunk_reset_policy": CHUNK_RESET_POLICY,
        "chunk_start_update": update,
        "chunk_start_seed": chunk_seed,
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "supervisor_invocation_id": invocation_id}
    run_manifest["initialization"] = initialization_record or {"mode": "random_initialization"}
    _atomic_json(output / "run.json", run_manifest)
    heartbeat_path = output / "heartbeat.json"
    def update_heartbeat():
        _atomic_json(heartbeat_path, {"schema_version": 1, "run_id": run_id,
            "track": config.track, "pid": os.getpid(),
            "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "completed_updates": update})
    update_heartbeat()
    hidden = None
    tracking_was_available = None
    commitment = None
    task_successes = 0
    task_failures = 0
    task_episodes = 0
    task_aborts = 0
    if (isinstance(resume_task_progress, dict)
            and task_assignment.get("assignment_id") is not None
            and resume_task_progress.get("assignment_id") ==
                task_assignment.get("assignment_id")):
        for name in ("episodes", "successes", "failures", "aborted_no_legal_support"):
            count = resume_task_progress.get(name)
            if type(count) is not int or count < 0:
                raise ValueError(f"checkpoint task progress has invalid {name}")
        task_episodes = resume_task_progress["episodes"]
        task_successes = resume_task_progress["successes"]
        task_failures = resume_task_progress["failures"]
        task_aborts = resume_task_progress["aborted_no_legal_support"]
    tracking_loss_reset_count = 0
    empty_support_diagnostics = []
    episode_generation = _episode_id(run_id, episode)
    stop_path = Path(stop_file) if stop_file else None
    abort_path = Path(abort_file) if abort_file else None

    def sibling_abort_marker():
        if abort_path is None or not abort_path.exists():
            return None
        try:
            marker = json.loads(abort_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("sibling abort marker is unreadable") from exc
        if (not isinstance(marker, dict) or marker.get("schema_version") != 1
                or not isinstance(marker.get("invocation_id"), str)
                or not marker["invocation_id"]):
            raise RuntimeError("sibling abort marker has invalid invocation provenance")
        if marker["invocation_id"] != invocation_id:
            raise RuntimeError("sibling abort marker belongs to a different supervisor invocation")
        return marker

    def task_progress_payload():
        return {"assignment_id": task_assignment.get("assignment_id"),
            "cycle": task_assignment.get("cycle"),
            "scenario_id": task_scenario, "category": task_category,
            "episodes": task_episodes, "successes": task_successes,
            "failures": task_failures,
            "aborted_no_legal_support": task_aborts}

    def task_outcomes_payload():
        return {**task_progress_payload(), "completed_updates": update}

    def persist_task_outcomes():
        _atomic_json(output / "task-outcomes.json", task_outcomes_payload())

    def abort_empty_support(exc, rows, update_number, update_events):
        nonlocal episode, task_aborts, hidden, commitment, episode_generation
        nonlocal tracking_was_available
        aborted_episode_id = episode_generation
        retained = [row for row in rows if row.get("episode_id") != aborted_episode_id]
        discarded_count = len(rows) - len(retained)
        rows[:] = retained
        if rows and not rows[-1]["done"]:
            discarded_count += len(rows)
            rows.clear()
        diagnostic = asdict(exc.diagnostic)
        next_episode = episode + 1
        event = {"track": config.track, "update": update_number,
                 "episode": episode, "episode_id": aborted_episode_id,
                 "reason": "no_legal_plan_a_pair", "credited_to_ppo": False,
                 "reset_seed": task_seed + next_episode,
                 "discarded_current_rollout_rows": discarded_count,
                 "diagnostic": diagnostic}
        empty_support_diagnostics.append(event)
        update_events.append(event)
        task_aborts += 1
        run_manifest["plan_a_empty_support_abort_count"] = len(empty_support_diagnostics)
        run_manifest["latest_plan_a_empty_support_abort"] = event
        diagnostic_path = output / f"plan-a-empty-support-aborts-{run_id}.json"
        run_manifest["plan_a_empty_support_diagnostics"] = str(diagnostic_path)
        _atomic_json(diagnostic_path, {
            "schema_version": 1, "run_id": run_id,
            "events": empty_support_diagnostics})
        episode = next_episode
        _reset_episode(env, task_seed + episode)
        hidden = None
        tracking_was_available = None
        commitment = None
        episode_generation = _episode_id(run_id, episode)
        update_heartbeat()

    try:
        chunk_target = update + config.updates
        while update < chunk_target and not (stop_path and stop_path.exists()) \
                and sibling_abort_marker() is None:
            rows = []
            update_empty_support_events = []
            rollout_model_fingerprint = (_model_id(model)
                if config.track in ("C1", "C2") else None)
            for _ in range(config.rollout_steps):
                if source_identity() != sources:
                    raise RuntimeError("source/runtime changed during MECO training; stopping before checkpoint write")
                try:
                    with torch.no_grad():
                        prepared = _prepare_decision(config.track, model, env, graph,
                            episode_generation, hidden,
                            route_cost_points_per_second=config.route_cost_points_per_second,
                            action_ticks=config.action_ticks, candidate_k=config.candidate_k,
                            p_mode=config.p_mode, p_cost_scale=config.p_cost_scale,
                            p_points_per_grid_edge=config.p_points_per_grid_edge,
                            prediction_mode=config.prediction_mode,
                            commitment=commitment,
                            tracking_was_available=tracking_was_available,
                            value_head=value_head)
                except PlanAEmptySupportError as exc:
                    abort_empty_support(exc, rows, update, update_empty_support_events)
                    continue
                action_index = int(prepared["action"][0].item())
                hidden_before = (None if prepared["hidden_before"] is None else
                                 prepared["hidden_before"].detach().clone())
                tracking_was_available = prepared["tracking_available"]
                tracking_loss_reset_count += int(prepared["tracking_reset"])
                hidden = prepared["hidden_next"].detach()
                value = prepared["value"].detach().reshape(())
                candidate = prepared["candidate_set"]
                selected_target_track = None
                selected_target_xy = None
                if "support" in prepared:
                    support = prepared["support"]
                    action = int(support.action_ids[action_index].item())
                    node = int(support.pair_node_ids[action_index].item())
                    goal_xy = tuple(float(v) for v in support.goal_xy[node].tolist())
                    goal_heading = float(support.goal_heading[node].item())
                else:
                    action = candidate.action_indices[action_index]
                    cols = {name: i for i, name in enumerate(candidate.table.feature_names)}
                    selected_row = candidate.table.features[action_index]
                    goal_xy = (float(selected_row[cols["goal_x_m"]]),
                               float(selected_row[cols["goal_y_m"]]))
                    goal_heading = None
                    selected_target_track = candidate.route_records[
                        action_index].get("sensor_track_buffer_index")
                    selected_target_xy = candidate.route_records[
                        action_index].get("source_fuel_xy_m")
                selected_candidate_id = (candidate.table.candidate_ids[action_index]
                    if "support" not in prepared else None)
                (reward, ended, timeout, executed_trace, heading_error,
                 goal_distance, heading_opportunity, native_preempted,
                 selected_target_lost) = _run_macro_step(
                    env, action, config.action_ticks, goal_xy,
                    goal_heading=goal_heading,
                    goal_track_index=selected_target_track,
                    goal_track_source_xy=selected_target_xy,
                    allow_action7_traversal=(prepared is not None and
                                              "support" in prepared and action == 7),
                    planner_traversal_cost_override=(
                        None if prepared.get("fallback_used", False) else
                        prepared.get("planner_traversal_cost")),
                    planner_traversal_cost_scale=prepared.get(
                        "planner_traversal_cost_scale", 1.))
                update_heartbeat()
                remasked = any(item != action for item in executed_trace)
                sampled_fallback = bool(prepared.get("fallback_used", False))
                failure_reasons = []
                if sampled_fallback:
                    failure_reasons.append("sampled_fallback")
                if remasked:
                    failure_reasons.append("action_remasked")
                if native_preempted:
                    failure_reasons.append("native_goal_preempted")
                if selected_target_lost:
                    failure_reasons.append("selected_target_lost")
                row = {"episode_id": episode_generation,
                    "raster": prepared["raster"].detach(),
                    "globals": prepared["globals"].detach(),
                    "graph": prepared["graph"],
                    "candidate_set": prepared["candidate_set"],
                    "candidate_digest": prepared.get("candidate_digest"),
                    "candidate_replay_digest": prepared.get("candidate_replay_digest"),
                    "inference_provenance": prepared["decision"].get(
                        "inference_provenance"),
                    "rollout_candidate_ids": (
                        list(prepared["candidate_set"].table.candidate_ids)
                        if config.track in ("C1", "C2") else None),
                    "rollout_valid_mask": (
                        [bool(value) for value in
                         prepared["candidate_set"].table.valid_mask]
                        if config.track in ("C1", "C2") else None),
                    # Diagnostic instrumentation adds five tensor SHA-256
                    # copies and one candidate-feature JSON hash per C-track
                    # decision; model parameters are hashed once per update.
                    "replay_fingerprints": (
                        {"raster": _tensor_evidence(prepared["raster"]),
                         "globals": _tensor_evidence(prepared["globals"]),
                         "hidden": _tensor_evidence(hidden_before),
                         "candidate_features": _candidate_identity_evidence(
                             prepared["candidate_set"]),
                         "candidate_digest": prepared.get("candidate_digest"),
                         "candidate_replay_digest": prepared.get(
                             "candidate_replay_digest"),
                         "route_cost": _tensor_evidence(prepared["route_cost"]),
                         "risk_cost": _tensor_evidence(prepared["risk_cost"]),
                         "model_parameters_sha256": rollout_model_fingerprint}
                        if config.track in ("C1", "C2") else None),
                    "candidate_commitment": (None if candidate is None or
                                              candidate.commitment is None
                                              else asdict(candidate.commitment)),
                    **({"support": prepared["support"],
                        "support_digest": prepared["support"].support_digest,
                        "route_cost_nodes": prepared["route_cost_nodes"].detach(),
                        "risk_cost_nodes": prepared["risk_cost_nodes"].detach(),
                        "graph_object": prepared["graph_object"],
                        "planner_object": prepared["planner_object"]}
                       if "support" in prepared else {}),
                    "route_cost": prepared["route_cost"].detach(),
                    "risk_cost": prepared["risk_cost"].detach(),
                    "execution_failure": bool(failure_reasons),
                    "execution_failure_reasons": failure_reasons,
                    "prediction": prepared.get("prediction"),
                    "executed_action_trace": executed_trace,
                    "terminal_heading_error": heading_error,
                    "terminal_goal_distance_m": goal_distance,
                    "heading_alignment_opportunity": heading_opportunity,
                    "native_goal_preempted": native_preempted,
                    "selected_target_track_lost": selected_target_lost,
                    "executed_high_level_action": action,
                    "action": torch.tensor([action_index], device=config.device),
                    "rollout_logits": prepared["decision"]["logits"].detach(),
                    "logprob": prepared["logprob"].detach().reshape(()),
                    "hidden_before": hidden_before,
                    "hidden_next": hidden.detach().clone(), "value": value,
                    "reward": reward.detach().reshape(()), "done": ended or timeout,
                    "legal_lookup": graph.lookup,
                    "path_geometry": prepared["path_geometry"],
                    "candidate_recall": (None if "support" in prepared
                                          else candidate.candidate_recall)}
                goal_complete = (goal_distance is not None and goal_distance <= .04 and
                                 (heading_error is None or heading_error <= .1))
                rows.append(row)
                if config.track in ("C1", "C2"):
                    if selected_candidate_id and selected_candidate_id.startswith("continue:"):
                        commitment = candidate.commitment
                    elif selected_candidate_id is not None:
                        route = candidate.route_records[action_index]
                        source_xy = route.get("source_fuel_xy_m")
                        sequence = candidate.observation.sequence
                        commitment = CandidateCommitment(
                            candidate_id=selected_candidate_id,
                            action_index=int(action),
                            goal_variant=int(route.get("goal_variant", 0)),
                            source_xy_m=(None if source_xy is None else
                                         (float(source_xy[0]), float(source_xy[1]))),
                            created_sequence=sequence,
                            expires_sequence=sequence + CANDIDATE_BOUNDARY[
                                "commitment_expiry_macro_actions"] * config.action_ticks)
                    if goal_complete or ended or timeout:
                        commitment = None
                hard_failure = remasked or native_preempted or selected_target_lost
                if config.track in ("C1", "C2") and hard_failure:
                    # A remasked action, preemption, or lost sensed target
                    # invalidates the carried task commitment.
                    commitment = None
                if ended or timeout:
                    outcome = env.curriculum_metrics.outcomes()["success"]
                    success = bool(outcome.reshape(-1)[0].item())
                    task_episodes += 1
                    task_successes += int(success)
                    task_failures += int(not success)
                    episode += 1
                    episode_seed = task_seed + episode
                    _reset_episode(env, episode_seed)
                    hidden = None
                    tracking_was_available = None
                    episode_generation = _episode_id(run_id, episode)
            if not rows:
                raise RuntimeError(
                    "Plan A rollout produced no eligible transitions after empty-support episode aborts")
            if rows[-1]["done"]:
                bootstrap_value = torch.zeros_like(rows[-1]["value"])
            else:
                try:
                    with torch.no_grad():
                        bootstrap = _prepare_decision(config.track, model, env, graph,
                            episode_generation, hidden,
                            route_cost_points_per_second=config.route_cost_points_per_second,
                            action_ticks=config.action_ticks, candidate_k=config.candidate_k,
                            p_mode=config.p_mode, p_cost_scale=config.p_cost_scale,
                            p_points_per_grid_edge=config.p_points_per_grid_edge,
                            prediction_mode=config.prediction_mode,
                            commitment=commitment,
                            tracking_was_available=tracking_was_available,
                            deterministic=True,
                            value_head=value_head)
                    bootstrap_value = bootstrap["value"].detach().reshape(())
                except PlanAEmptySupportError as exc:
                    abort_empty_support(exc, rows, update, update_empty_support_events)
                    if not rows:
                        raise RuntimeError(
                            "Plan A bootstrap found empty support and no eligible transitions remain") from exc
                    if not rows[-1]["done"]:
                        rows.clear()
                        raise RuntimeError(
                            "empty-support bootstrap discarded the rollout to preserve episode boundaries") from exc
                    bootstrap_value = torch.zeros_like(rows[-1]["value"])
            stats = _ppo_update(config.track, model, value_head, optimizer, rows,
                                config, bootstrap_value, update_number=update,
                                diagnostic_output=output)
            update += 1
            update_heartbeat()
            stats.update(update=update, episode=episode, track=config.track,
                         scenario_id=task_scenario, category=task_category,
                         task_assignment_id=task_assignment.get("assignment_id"),
                         learning_rate=config.learning_rate,
                         candidate_k=(None if config.track in ("A1", "A2", "A3")
                                      else config.candidate_k),
                         graph_nodes=graph.node_count,
                         update_utc=datetime.now(timezone.utc).isoformat())
            stats["tracking_loss_reset_count"] = tracking_loss_reset_count
            if update_empty_support_events:
                stats["plan_a_empty_support_abort_count"] = len(update_empty_support_events)
                stats["plan_a_empty_support_aborts"] = update_empty_support_events
            if config.track in ("A1", "A2", "A3"):
                stats.update(policy_support_order=SUPPORT_ORDER,
                    policy_support_pair_count=rows[0]["support"].pair_count,
                    policy_support_digests=[row["support"].support_digest for row in rows],
                    execution_failure_count=sum(bool(row["execution_failure"]) for row in rows),
                    execution_failure_reason_counts={
                        reason: sum(reason in row["execution_failure_reasons"] for row in rows)
                        for reason in ("sampled_fallback", "action_remasked", "native_goal_preempted")},
                    actor_loss_skipped_execution_failures=sum(
                        bool(row["execution_failure"]) for row in rows))
            else:
                stats["candidate_recall"] = sum(
                    row["candidate_recall"]["candidate_recall"]
                    for row in rows) / len(rows)
            history.append(stats)
            payload = {"schema_version": 1, "config": asdict(config),
                "prediction_contract": prediction_contract(config.prediction_mode),
                "qualification_protocol_id": qualification_protocol_id,
                "qualification_protocol_sha256": qualification_protocol_sha256,
                "protocol_id": protocol_id,
                "input_contract_sha256": run_manifest["input_contract_sha256"],
                "initialization": run_manifest["initialization"],
                "task_assignment": task_assignment,
                "phase4_training": run_manifest["phase4_training"],
                "chunk_reset_policy": CHUNK_RESET_POLICY,
                "sources": sources, "run_id": run_id, "update": update,
                "episode": episode, "model": model.state_dict(),
                "task_progress": task_progress_payload(),
                "value_head": value_head.state_dict(), "optimizer": optimizer.state_dict(),
                "rng_torch": torch.get_rng_state(), "rng_python": random.getstate(),
                "rng_device": torch.cuda.get_rng_state(torch.device(config.device)) if config.device.startswith("cuda") else None,
                "history": history}
            _atomic_torch(output / "latest.pt", payload)
            _atomic_json(output / "stats.json", {"run_id": run_id, "track": config.track,
                "update": update, "episode": episode,
                "input_contract_sha256": run_manifest["input_contract_sha256"],
                "history": history})
            persist_task_outcomes()
        abort_marker = sibling_abort_marker()
        worker_status = ("stopped" if stop_path and stop_path.exists() else
                         "aborted_due_to_sibling_failure"
                         if abort_marker is not None and update < chunk_target
                         else "complete")
        run_manifest.update(status=worker_status,
                            completed_updates=update,
                            stopped_utc=datetime.now(timezone.utc).isoformat())
        if worker_status == "aborted_due_to_sibling_failure":
            run_manifest.update(sibling_abort_file=str(abort_path),
                sibling_abort_invocation_id=abort_marker["invocation_id"],
                sibling_abort_marker=abort_marker,
                sibling_abort_observed_utc=datetime.now(timezone.utc).isoformat())
        run_manifest["task_outcomes"] = task_outcomes_payload()
        _atomic_json(output / "task-outcomes.json", run_manifest["task_outcomes"])
        _atomic_json(output / "run.json", run_manifest)
        update_heartbeat()
    except BaseException as exc:
        run_manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                            failed_utc=datetime.now(timezone.utc).isoformat())
        _atomic_json(output / "run.json", run_manifest)
        update_heartbeat()
        raise
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return {"track": config.track, "update": update, "checkpoint": str(output / "latest.pt"),
            "run_id": run_id, "status": run_manifest["status"]}


def _validate_supervisor_training_binding(sources, loaded_native_identity,
                                          expected_source_sha256, attestation):
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    attested_identity = attestation.get("identity") if isinstance(attestation, dict) else None
    attestation_digest = (hashlib.sha256(json.dumps(attested_identity, sort_keys=True,
        allow_nan=False).encode()).hexdigest() if isinstance(attested_identity, dict) else None)
    if (not isinstance(attestation, dict) or attestation.get("schema") != 1
            or attestation.get("identity_sha256") != attestation_digest
            or attested_identity != loaded_native_identity
            or not isinstance(sources, dict)
            or sources.get("_native_fuel_extension") != loaded_native_identity
            or source_digest != expected_source_sha256):
        raise RuntimeError("worker source/native identity differs from supervisor binding")


def _d1_d2_scores(candidate_set: RouteFeasibleCandidates, env, policy: str,
                  prediction_mode: str | None = None,
                  route_cost_points_per_second=ROUTE_COST_POINTS_PER_SECOND):
    table = candidate_set.table
    cols = {name: i for i, name in enumerate(table.feature_names)}
    values = []
    for index, (name, row, feasible) in enumerate(zip(table.candidate_ids,
                                                      table.features,
                                                      table.valid_mask)):
        if not feasible:
            values.append(float("-inf"))
            continue
        action = int(row[cols["action_index"]])
        conf = row[cols["track_confidence"]]
        age = row[cols["track_age_s"]]
        nearby = row[cols["nearby_sensed_track_count"]]
        if action < 4:
            score = (1. + .2 * float(nearby or 0.) + .25 * float(conf or 0.)
                     - .04 * float(age or 0.))
        else:
            score = {"score": .8, "intercept": .1, "ferry": .45, "hold": .05}.get(name, 0.)
        route_cost = (float(candidate_set.route_cost_points[index].item()) /
                      ROUTE_COST_POINTS_PER_SECOND * route_cost_points_per_second)
        score -= (route_cost +
                  float(candidate_set.risk_cost_points[index].item()))
        switch_col = cols.get("commitment_switch_cost")
        if switch_col is not None:
            score -= (float(row[switch_col] or 0.) *
                      CANDIDATE_BOUNDARY["commitment_switch_cost_points"])
        values.append(score)
    return torch.tensor(values, device=env.device, dtype=torch.float32)[None]


def _timing_sync(env):
    """Synchronize the measured device before/after a host wall interval."""
    if torch.device(env.device).type == "cuda":
        torch.cuda.synchronize(env.device)


def _timing_summary(samples):
    """Summarize measured milliseconds using nearest-rank percentiles."""
    values = sorted(float(value) for value in samples)
    def percentile(p):
        if not values:
            return None
        return values[max(0, math.ceil(p * len(values)) - 1)]
    return {"sample_count": len(values), "p50_ms": percentile(.50),
            "p95_ms": percentile(.95), "p99_ms": percentile(.99),
            "mean_ms": (sum(values) / len(values)) if values else None}


def _metric_snapshot(metrics):
    return {name: value.detach().cpu().reshape(-1)[0].item()
            for name, value in metrics.outcomes().items()}


_EVALUATION_CASE_FIELDS = frozenset(("layout_id", "defender_family",
    "start_condition", "sensor_profile"))


def load_evaluation_protocol(value, *, scenario_id=None):
    if isinstance(value, dict):
        protocol = value
        raw = json.dumps(protocol, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()
        path_name = None
    else:
        protocol_path = Path(value).expanduser().resolve()
        raw = protocol_path.read_bytes()
        protocol = json.loads(raw)
        path_name = protocol_path.name
    if not isinstance(protocol, dict):
        raise ValueError("evaluation protocol must be a JSON object")
    protocol_id = protocol.get("protocol_id")
    body = {key: item for key, item in protocol.items() if key != "protocol_id"}
    expected_id = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if protocol_id != expected_id:
        raise ValueError("evaluation protocol ID does not match its contents")
    from .meco_qualification import (CALIBRATED_PROTOCOL_VERSION,
        D0_PAIRED_REFERENCE_PROTOCOL_VERSION, LEGACY_CALIBRATED_PROTOCOL_VERSION,
        PROTOCOL_VERSION)
    version = protocol.get("protocol_version")
    if version not in (PROTOCOL_VERSION, LEGACY_CALIBRATED_PROTOCOL_VERSION,
                       D0_PAIRED_REFERENCE_PROTOCOL_VERSION,
                       CALIBRATED_PROTOCOL_VERSION):
        raise ValueError("unsupported evaluation protocol version")
    if version == PROTOCOL_VERSION and not isinstance(value, dict):
        from .meco_supervised import _load_protocol
        checked, checked_id, checked_sha, checked_path = _load_protocol(Path(value))
        return checked, checked_id, checked_sha, checked_path.name
    if version in (LEGACY_CALIBRATED_PROTOCOL_VERSION,
                   D0_PAIRED_REFERENCE_PROTOCOL_VERSION, CALIBRATED_PROTOCOL_VERSION):
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("calibrated evaluation protocol requires an explicit scenario")
        from .meco_qualification import _validate_v2_protocol
        _validate_v2_protocol(protocol, scenario_id)
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation protocol has no frozen cases")
    split_seeds = {name: set() for name in ("train", "validation", "test")}
    for case in cases:
        if (not isinstance(case, dict) or case.get("split") not in split_seeds
                or not isinstance(case.get("seeds"), list)
                or any(type(seed) is not int or seed < 0 for seed in case["seeds"])
                or len(set(case["seeds"])) != len(case["seeds"])
                or not all(isinstance(case.get(field), str) and case[field]
                    for field in ("scenario_id", "category", *_EVALUATION_CASE_FIELDS))):
            raise ValueError("evaluation protocol contains malformed case data")
        split_seeds[case["split"]].update(case["seeds"])
    if (split_seeds["train"] & split_seeds["validation"]
            or split_seeds["train"] & split_seeds["test"]
            or split_seeds["validation"] & split_seeds["test"]):
        raise ValueError("evaluation protocol split seed pools overlap")
    return protocol, protocol_id, hashlib.sha256(raw).hexdigest(), path_name


def resolve_evaluation_protocol(path, *, scenario_id, categories, split,
                                episode_seeds, qualification_case):
    """Validate an exact held-out evaluation binding without starting a simulator."""
    if split not in ("validation", "test"):
        raise ValueError("evaluation protocol binding is only valid for validation or test")
    if path is None:
        raise ValueError("held-out evaluation requires an explicit evaluation protocol")
    protocol, protocol_id, protocol_sha256, protocol_file = load_evaluation_protocol(
        path, scenario_id=scenario_id)
    if not isinstance(qualification_case, dict) or set(qualification_case) != _EVALUATION_CASE_FIELDS:
        raise ValueError("held-out evaluation requires an exact four-field case stratum")
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("held-out evaluation categories must be nonempty and unique")
    if not isinstance(episode_seeds, (tuple, list)) or not episode_seeds:
        raise ValueError("held-out evaluation requires an explicit exact seed list")
    seeds = tuple(episode_seeds)
    if (any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("evaluation protocol seed list is malformed")
    selected_cases = []
    for category in categories:
        matches = [case for case in protocol.get("cases", [])
            if case.get("scenario_id") == scenario_id and case.get("category") == category
            and case.get("split") == split and all(
                case.get(field) == qualification_case[field]
                for field in _EVALUATION_CASE_FIELDS)]
        if len(matches) != 1:
            raise ValueError("evaluation protocol case is missing or ambiguous for category")
        if tuple(matches[0].get("seeds", ())) != seeds:
            raise ValueError("evaluation seeds differ from exact protocol case seed membership")
        selected_cases.append(matches[0])
    for case in protocol.get("cases", []):
        if (case.get("scenario_id") == scenario_id and case.get("category") in categories
                and case.get("split") != split and set(seeds).intersection(case.get("seeds", []))):
            raise ValueError("evaluation seed list overlaps another protocol split")
    return {"protocol_id": protocol_id, "protocol_sha256": protocol_sha256,
        "protocol_file": protocol_file, "split": split,
        "scenario_id": scenario_id, "categories": list(categories),
        "qualification_case": dict(qualification_case), "seeds": list(seeds),
        "case_ids": [{"category": case["category"], **{
            field: case[field] for field in _EVALUATION_CASE_FIELDS}}
            for case in selected_cases]}


def resolve_evaluation_case(path, *, scenario_id, categories, split,
                            qualification_case=None):
    """Resolve one shared stratum, refusing implicit selection when ambiguous."""
    protocol, _, _, _ = load_evaluation_protocol(path, scenario_id=scenario_id)
    if qualification_case is not None:
        if not isinstance(qualification_case, dict) or set(qualification_case) != _EVALUATION_CASE_FIELDS:
            raise ValueError("evaluation case must contain exactly the four frozen stratum identifiers")
        chosen = dict(qualification_case)
    else:
        candidates = []
        for category in categories:
            cases = [case for case in protocol.get("cases", [])
                if case.get("scenario_id") == scenario_id and case.get("category") == category
                and case.get("split") == split]
            if not cases:
                raise ValueError("evaluation protocol has no case for requested scenario/category/split")
            candidates.append({(tuple(case.get(field) for field in sorted(_EVALUATION_CASE_FIELDS)),
                                tuple(case.get("seeds", []))) for case in cases})
        shared = set.intersection(*candidates) if candidates else set()
        if len(shared) != 1:
            raise ValueError("evaluation protocol case is missing or ambiguous; select a stratum explicitly")
        stratum, _ = next(iter(shared))
        chosen = dict(zip(sorted(_EVALUATION_CASE_FIELDS), stratum))
    matching_seed_lists = []
    for category in categories:
        matches = [case for case in protocol.get("cases", [])
            if case.get("scenario_id") == scenario_id and case.get("category") == category
            and case.get("split") == split and all(
                case.get(field) == chosen[field] for field in _EVALUATION_CASE_FIELDS)]
        if len(matches) != 1:
            raise ValueError("evaluation protocol case is missing or ambiguous for requested category")
        matching_seed_lists.append(tuple(matches[0].get("seeds", [])))
    if not matching_seed_lists or any(seeds != matching_seed_lists[0] for seeds in matching_seed_lists):
        raise ValueError("evaluation categories do not share one exact seed list")
    return chosen, matching_seed_lists[0]


def _manage_evaluation_adapters(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with managed_meco_adapters():
            return function(*args, **kwargs)
    return wrapped


def validate_diagnostic_frontier_binding(binding, *, schedule_case_binding,
        split, checkpoints):
    """Validate frozen frontier metadata and checkpoint bytes without simulation."""
    required = {"schema_version", "frontier_id", "cycle", "update",
        "checkpoint_sha256", "source_sha256", "configuration_sha256",
        "protocol_id", "protocol_sha256"}
    if (split != "diagnostic" or schedule_case_binding is None or
            not isinstance(binding, dict) or set(binding) != required or
            binding.get("schema_version") != 1 or
            not isinstance(binding.get("frontier_id"), str) or not binding["frontier_id"] or
            type(binding.get("cycle")) is not int or binding["cycle"] < 1 or
            type(binding.get("update")) is not int or binding["update"] < 0 or
            binding.get("protocol_id") != schedule_case_binding.get("protocol_id") or
            binding.get("protocol_sha256") != schedule_case_binding.get("protocol_sha256") or
            any(not isinstance(binding.get(key), str) or len(binding[key]) != 64 or
                any(char not in "0123456789abcdef" for char in binding[key])
                for key in ("source_sha256", "configuration_sha256", "protocol_sha256")) or
            not isinstance(binding.get("checkpoint_sha256"), dict) or
            set(binding["checkpoint_sha256"]) != set(MECO_TRACKS) or
            any(not isinstance(value, str) or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value)
                for value in binding["checkpoint_sha256"].values())):
        raise ValueError("diagnostic frontier binding is malformed or conflicts with its case")
    actual = {track: hashlib.sha256(Path(checkpoints[track]).read_bytes()).hexdigest()
        for track in MECO_TRACKS}
    if actual != binding["checkpoint_sha256"]:
        raise ValueError("diagnostic frontier checkpoint snapshots changed before evaluation")
    return binding


@_manage_evaluation_adapters
@torch.no_grad()
def evaluate_meco(checkpoints: dict[str, str], *, scenario_id: str,
                  categories: tuple[str, ...], episodes: int, master_seed: int,
                  device: str, include_controls=True, output: str | None = None,
                  episode_seeds: tuple[int, ...] | None = None,
                  split: str = "diagnostic", qualification_case: dict | None = None,
                  qualification_protocol_id: str | None = None,
                  evaluation_protocol: str | Path | None = None,
                  evaluation_case: dict | None = None,
                  evaluation_policies: tuple[str, ...] | None = None,
                  control_policies: tuple[str, ...] | None = None,
                  prediction_mode: str | None = None,
                  candidate_k_override: int | None = None,
                  control_prediction_mode: str | None = None,
                  control_prediction_modes: dict[str, str] | None = None,
                  calibration_control_policy: str | None = None,
                  calibration_route_cost_points_per_second: float | None = None,
                  inference_disable_tick: int | None = None,
                  phase4_factor: dict | None = None,
                  phase4_opponent_schedule: dict | None = None,
                  phase4_direct_pursuit_binding: dict | None = None,
                  phase4_trench_interception_binding: dict | None = None,
                  phase4_contact_factor_id: str | None = None,
                  phase4_route_outcome_factor_ids: tuple[str, ...] | None = None,
                  contact_geometry_id: str | None = None,
                  contact_geometry_config: dict | None = None,
                  phase4_start_profile_assignment: dict | None = None,
                  phase4_focal_pose_assignment: dict | None = None,
                  phase4_teammate_behavior_assignment: dict | None = None,
                  schedule_case_binding: dict | None = None,
                  diagnostic_frontier_binding: dict | None = None,
                  replay_output_dir: str | Path | None = None,
                  replay_seeds: tuple[int, ...] | None = None,
                  replay_selection: dict[str, list[int]] | None = None,
                  condition_id: str | None = None,
                  simulated_score_estimand: str | Path | None = None,
                  paired_calibration_bindings: dict[int, dict] | None = None,
                  paired_calibration_worker_identity: dict | None = None):
    """Paired seeded simulator evaluation for D0/D1/D2 and five checkpoints."""
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be positive")
    if episode_seeds is not None:
        if (not isinstance(episode_seeds, (tuple, list))
                or any(type(seed) is not int or not 0 <= seed < 2**63
                    for seed in episode_seeds)
                or len(episode_seeds) != episodes
                or len(set(episode_seeds)) != episodes):
            raise ValueError("episode_seeds must be a unique exact seed list matching episodes")
    scored_seeds = (tuple(episode_seeds) if episode_seeds is not None else
                    tuple(master_seed + index for index in range(episodes)))
    if replay_selection is None:
        replay_seed_set = _validate_replay_seed_selection(
            replay_output_dir, replay_seeds, scored_seeds)
        replay_policy_seeds = None
    else:
        if replay_output_dir is None or replay_seeds is not None:
            raise ValueError("replay_selection requires replay_output_dir and excludes replay_seeds")
        if (not isinstance(replay_selection, dict) or not replay_selection or
                any(not isinstance(policy, str) or not policy or
                    not isinstance(selected, list) or not selected or
                    len(selected) != len(set(selected)) or
                    any(type(seed) is not int or seed not in scored_seeds for seed in selected)
                    for policy, selected in replay_selection.items())):
            raise ValueError("replay_selection must bind policies to unique scored seed lists")
        replay_policy_seeds = {policy: set(selected)
                               for policy, selected in replay_selection.items()}
        replay_seed_set = set().union(*replay_policy_seeds.values())
    paired_replay_requested = (paired_calibration_bindings is not None or
        paired_calibration_worker_identity is not None)
    if paired_replay_requested and (paired_calibration_bindings is None or
            paired_calibration_worker_identity is None):
        raise ValueError("paired calibration replay requires per-seed campaign bindings and worker identity")
    if paired_replay_requested and (split != "calibration" or
            calibration_control_policy is None):
        raise ValueError("paired replay bindings are accepted only for calibration controls")
    if paired_replay_requested and (not isinstance(paired_calibration_bindings, dict)
            or set(paired_calibration_bindings) != set(scored_seeds)):
        raise ValueError("paired calibration bindings must cover every exact scored seed")
    if simulated_score_estimand is not None:
        if (replay_output_dir is None or
                (replay_selection is not None and not paired_replay_requested)
                or replay_seed_set != set(scored_seeds)):
            raise ValueError("simulated score collection requires replay export for every policy and scored seed")
    if split not in ("validation", "test", "train", "calibration", "diagnostic"):
        raise ValueError("split must be train, validation, test, calibration, or diagnostic")
    if schedule_case_binding is not None:
        if (not isinstance(schedule_case_binding, dict) or
                set(schedule_case_binding) != {"index", "scenario_id", "category", "seeds",
                    "split", "protocol_id", "protocol_sha256", "qualification_case"} or
                type(schedule_case_binding.get("index")) is not int or
                schedule_case_binding["index"] < 0 or
                schedule_case_binding.get("scenario_id") != scenario_id or
                categories != (schedule_case_binding.get("category"),) or
                schedule_case_binding.get("split") != split or
                schedule_case_binding.get("seeds") != list(scored_seeds) or
                not isinstance(schedule_case_binding.get("protocol_id"), str) or
                not isinstance(schedule_case_binding.get("protocol_sha256"), str)):
            raise ValueError("scheduled evaluation binding differs from evaluator arguments")
    if diagnostic_frontier_binding is not None:
        validate_diagnostic_frontier_binding(diagnostic_frontier_binding,
            schedule_case_binding=schedule_case_binding, split=split,
            checkpoints=checkpoints)
    evaluation_binding = None
    if split in ("validation", "test"):
        if evaluation_case is None:
            evaluation_case = qualification_case
        evaluation_binding = resolve_evaluation_protocol(evaluation_protocol,
            scenario_id=scenario_id, categories=categories, split=split,
            episode_seeds=episode_seeds, qualification_case=evaluation_case)
        if qualification_case is not None and qualification_case != evaluation_case:
            raise ValueError("qualification and evaluation case strata disagree")
        qualification_case = evaluation_case
    elif evaluation_protocol is not None or evaluation_case is not None:
        raise ValueError("evaluation protocol/case cannot bind diagnostic or training evaluation")
    phase4_opponent_policy_id = None
    if phase4_opponent_schedule is not None:
        from .meco_phase4_opponents import validate_opponent_policy_schedule
        validated_schedule = validate_opponent_policy_schedule(
            phase4_opponent_schedule, scenario_id=scenario_id,
            categories=categories, qualification_case=qualification_case or {},
            seeds=scored_seeds, split=split,
            factor_id=phase4_opponent_schedule.get("factor_id"),
            protocol_id=(evaluation_binding["protocol_id"]
                         if evaluation_binding is not None else None))
        phase4_opponent_policy_id = validated_schedule["policy_id"]
    if phase4_direct_pursuit_binding is not None:
        if (split != "validation" or evaluation_binding is None or
                scenario_id != "offense_vs_scripted_pressure" or
                not isinstance(phase4_direct_pursuit_binding, dict) or
                phase4_direct_pursuit_binding.get("policy_id") != "direct_pursuit" or
                phase4_direct_pursuit_binding.get("factor_id") != "opponent.direct_pursuit" or
                phase4_direct_pursuit_binding.get("protocol_id") != evaluation_binding["protocol_id"] or
                phase4_direct_pursuit_binding.get("protocol_sha256") != evaluation_binding["protocol_sha256"] or
                phase4_direct_pursuit_binding.get("seed_list") != list(scored_seeds) or
                phase4_direct_pursuit_binding.get("scenario_id") != scenario_id or
                phase4_direct_pursuit_binding.get("qualification_case") != qualification_case):
            raise ValueError("direct-pursuit binding differs from exact held-out evaluator case/seed")
        if phase4_opponent_schedule is not None and (
                phase4_opponent_schedule.get("schedule_sha256") !=
                phase4_direct_pursuit_binding.get("schedule_sha256")):
            raise ValueError("direct-pursuit treatment schedule differs from outcome binding")
    if phase4_trench_interception_binding is not None:
        trench_binding = phase4_trench_interception_binding
        if (split != "validation" or evaluation_binding is None or
                scenario_id != "offense_vs_scripted_pressure" or
                not isinstance(trench_binding, dict) or
                trench_binding.get("schema") !=
                    "meco-phase4-trench-interception-binding-v1" or
                trench_binding.get("factor_id") != "opponent.trench_interception" or
                trench_binding.get("policy_id") != "trench_interceptor" or
                trench_binding.get("protocol_id") != evaluation_binding["protocol_id"] or
                trench_binding.get("protocol_sha256") != evaluation_binding["protocol_sha256"] or
                trench_binding.get("seed_list") != list(scored_seeds) or
                trench_binding.get("scenario_id") != scenario_id or
                trench_binding.get("category") not in categories or
                trench_binding.get("qualification_case") != qualification_case or
                trench_binding.get("split") != "validation" or
                not isinstance(trench_binding.get("pair_binding_id"), str) or
                not trench_binding.get("pair_binding_id")):
            raise ValueError("trench-interception binding differs from exact held-out case/seed")
        expected_trench_schedule = trench_binding.get("schedule")
        if (not isinstance(expected_trench_schedule, dict) or
                trench_binding.get("schedule_sha256") !=
                    expected_trench_schedule.get("schedule_sha256") or
                trench_binding.get("per_episode_assignment_sha256") !=
                    expected_trench_schedule.get("per_episode_assignment_sha256") or
                trench_binding.get("policy_config_sha256") !=
                    expected_trench_schedule.get("policy_config_sha256") or
                (phase4_opponent_schedule is not None and
                 phase4_opponent_schedule != expected_trench_schedule) or
                (phase4_opponent_schedule is None and
                 trench_binding.get("condition") not in (None, "no_factor"))):
            raise ValueError("trench treatment schedule differs from exact outcome binding")
    if prediction_mode is not None and prediction_mode not in PREDICTION_MODES:
        raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
    if control_prediction_mode is not None and control_prediction_mode not in PREDICTION_MODES:
        raise ValueError(f"control_prediction_mode must be one of {PREDICTION_MODES}")
    if control_prediction_mode is not None and control_prediction_modes is not None:
        raise ValueError("use either control_prediction_mode or control_prediction_modes, not both")
    if control_prediction_modes is not None and (
            not isinstance(control_prediction_modes, dict)
            or not control_prediction_modes
            or set(control_prediction_modes) - {"D1", "D2"}
            or any(mode not in PREDICTION_MODES for mode in control_prediction_modes.values())):
        raise ValueError("control_prediction_modes must map D1 and D2 to known modes")
    if candidate_k_override is not None and candidate_k_override not in (8, 16, 32):
        raise ValueError("candidate_k_override must be 8, 16, or 32")
    if calibration_control_policy is not None and calibration_control_policy not in ("D1", "D2"):
        raise ValueError("calibration_control_policy must be D1 or D2")
    if calibration_control_policy is not None and phase4_factor is not None:
        raise ValueError("calibration control evaluation does not permit phase4_factor")
    if calibration_route_cost_points_per_second is not None and (
            type(calibration_route_cost_points_per_second) not in (int, float)
            or not math.isfinite(calibration_route_cost_points_per_second)
            or calibration_route_cost_points_per_second < 0):
        raise ValueError("calibration route-cost weight must be finite and nonnegative")
    if inference_disable_tick is not None and (type(inference_disable_tick) is not int
            or inference_disable_tick <= 0):
        raise ValueError("inference_disable_tick must be a positive physics tick")
    if (not isinstance(device, str) or not device.startswith("cuda:")
            or not torch.cuda.is_available()):
        raise RuntimeError("MECO checkpoint evaluation requires an explicit available CUDA/HIP device")
    selected_device = torch.device(device)
    if (selected_device.type != "cuda" or selected_device.index is None
            or selected_device.index < 0 or selected_device.index >= torch.cuda.device_count()):
        raise RuntimeError(f"MECO checkpoint evaluation device is unavailable: {device}")
    from .meco_phase4 import validate_phase4_factor
    from .meco_device import normalized_device_info
    from .fuel_physics_hip import prepare_native_fuel_extension
    prepare_native_fuel_extension(device)
    phase4_factor = validate_phase4_factor(phase4_factor)
    if phase4_contact_factor_id not in (None, "contact.robot_bump"):
        raise ValueError("unsupported Phase 4 contact factor")
    from .meco_phase4_context import (CONTACT_GEOMETRIES,
        phase4_context_identity, validate_contact_geometry_split)
    if contact_geometry_id is None:
        if contact_geometry_config is not None or phase4_contact_factor_id is not None:
            raise ValueError("contact event capture requires a registered contact geometry")
    elif (contact_geometry_id not in CONTACT_GEOMETRIES or
            contact_geometry_config != phase4_context_identity(
                "robot_contact_geometry", contact_geometry_id) or
            phase4_contact_factor_id != "contact.robot_bump"):
        raise ValueError("contact geometry and event factor must match the frozen registry")
    validate_contact_geometry_split(contact_geometry_id, split)
    execution_device = normalized_device_info(device)
    if paired_replay_requested:
        import re
        match = re.fullmatch(r"cuda:(\d+)", device)
        if match is None or not torch.version.hip:
            raise ValueError("paired calibration replay requires an indexed ROCm device")
        props = torch.cuda.get_device_properties(int(match.group(1)))
        runtime_identity = {
            "device_model": str(props.name),
            "device_uuid": str(getattr(props, "uuid", "")),
            "backend": "hip",
            "backend_version": str(torch.version.hip),
            "torch_version": str(torch.__version__),
        }
        paired_policy = next(iter(replay_selection)) if replay_selection else None
        if paired_policy not in ("D0", calibration_control_policy):
            raise ValueError("paired replay selection differs from its calibrated control")
        _validate_paired_calibration_replay_bindings(paired_calibration_bindings,
            scored_seeds, paired_calibration_worker_identity, device=device,
            execution_device=execution_device, scenario_id=scenario_id,
            split=split, policy=paired_policy,
            prediction_mode=("current" if paired_policy in ("D0", "D1") else
                (control_prediction_modes or {}).get("D2", "cv")),
            qualification_case={"category": categories[0], **(qualification_case or {})},
            runtime_identity=runtime_identity)
    if condition_id is not None and (not isinstance(condition_id, str) or not condition_id):
        raise ValueError("condition_id must be a nonempty string when provided")
    if qualification_case is not None:
        required_case_fields = {"layout_id", "defender_family", "start_condition", "sensor_profile"}
        if not isinstance(qualification_case, dict) or set(qualification_case) != required_case_fields:
            raise ValueError("qualification_case must contain the four frozen stratum identifiers")
    if phase4_start_profile_assignment is not None:
        from .meco_phase4_context import (
            phase4_start_profile_assignment as build_start_profile_assignment,
            validate_phase4_start_profile_assignment)
        captured_start_assignment = validate_phase4_start_profile_assignment(
            phase4_start_profile_assignment)
        if evaluation_binding is None or qualification_case is None:
            raise ValueError("full-team start profiles require an exact validation protocol case")
        expected_start_assignment = build_start_profile_assignment(
            captured_start_assignment["registry_profile_id"],
            scenario_id=scenario_id, split=split,
            protocol_id=evaluation_binding["protocol_id"],
            protocol_sha256=evaluation_binding["protocol_sha256"],
            qualification_case=qualification_case, categories=categories,
            seeds=scored_seeds)
        if captured_start_assignment != expected_start_assignment:
            raise ValueError("full-team start profile differs from the exact evaluation protocol/seeds")
    if phase4_focal_pose_assignment is not None:
        from .meco_phase4_context import (
            phase4_focal_pose_assignment as build_focal_pose_assignment,
            validate_phase4_focal_pose_assignment)
        captured_focal_assignment = validate_phase4_focal_pose_assignment(
            phase4_focal_pose_assignment)
        if evaluation_binding is None or qualification_case is None:
            raise ValueError("focal pose validation requires an exact frozen protocol case")
        expected_focal_assignment = build_focal_pose_assignment(
            captured_focal_assignment["requested_offset_name"],
            paired_offset_name=captured_focal_assignment["paired_offset_name"],
            scenario_id=scenario_id, split=split,
            protocol_id=evaluation_binding["protocol_id"],
            protocol_sha256=evaluation_binding["protocol_sha256"],
            qualification_case=qualification_case, categories=categories,
            seeds=scored_seeds)
        if captured_focal_assignment != expected_focal_assignment:
            raise ValueError("focal pose assignment differs from exact evaluation protocol/case/seeds")
    if phase4_teammate_behavior_assignment is not None:
        from .meco_phase4_context import (
            phase4_teammate_behavior_assignment as build_teammate_assignment,
            validate_phase4_teammate_behavior_assignment,
            phase4_teammate_behavior_record)
        captured_teammate_assignment = validate_phase4_teammate_behavior_assignment(
            phase4_teammate_behavior_assignment)
        if evaluation_binding is None or qualification_case is None:
            raise ValueError("teammate behavior pool requires an exact validation protocol case")
        expected_teammate_assignment = build_teammate_assignment(
            arm=captured_teammate_assignment["arm"], scenario_id=scenario_id,
            split=split, protocol_id=evaluation_binding["protocol_id"],
            protocol_sha256=evaluation_binding["protocol_sha256"],
            qualification_case=qualification_case, categories=categories,
            seeds=scored_seeds, pool_seed=captured_teammate_assignment["pool_seed"])
        if captured_teammate_assignment != expected_teammate_assignment:
            raise ValueError("teammate behavior assignment differs from exact protocol/case/seeds")
    if qualification_protocol_id is not None and (
            split not in ("validation", "test") or
            not isinstance(qualification_protocol_id, str) or
            not qualification_protocol_id):
        raise ValueError("qualification protocol identity is only valid for held-out splits")
    definition = get_scenario(scenario_id)
    if any(category not in definition.required_categories for category in categories):
        raise ValueError("category is not registered for scenario")
    if inference_disable_tick is not None:
        scenario_horizon = math.ceil(definition.time_budget_s / OBSERVATION_DT)
        if inference_disable_tick >= scenario_horizon:
            raise ValueError("inference_disable_tick must be below the scenario horizon "
                f"({scenario_horizon} physics ticks) for every category")
    if split not in ("validation", "test", "train", "calibration", "diagnostic"):
        raise ValueError("split must be train, validation, test, calibration, or diagnostic")
    selected = tuple(MECO_TRACKS if evaluation_policies is None else evaluation_policies)
    if len(set(selected)) != len(selected) or not set(selected) <= set(MECO_TRACKS):
        raise ValueError("evaluation_policies must be a unique subset of MECO tracks")
    if set(checkpoints) != set(selected):
        raise ValueError("evaluation needs exactly the selected policy checkpoints")
    controls = tuple(("D0", "D1", "D2") if control_policies is None else control_policies)
    if len(set(controls)) != len(controls) or not set(controls) <= {"D0", "D1", "D2"}:
        raise ValueError("control_policies must be a unique subset of D0/D1/D2")
    if include_controls and "D0" not in controls:
        raise ValueError("matched evaluation requires D0 as the paired reference")
    effective_control_prediction_modes = {"D1": "current", "D2": "cv"}
    if control_prediction_mode is not None:
        effective_control_prediction_modes = {"D1": control_prediction_mode,
                                              "D2": control_prediction_mode}
    elif control_prediction_modes is not None:
        effective_control_prediction_modes.update(control_prediction_modes)
    calibration_control_case_binding = None
    if calibration_control_policy is not None:
        expected_mode = ("current" if calibration_control_policy == "D1" else
                         (control_prediction_modes or {}).get("D2"))
        valid_mode = (expected_mode == "current" if calibration_control_policy == "D1"
                      else expected_mode in ("cv", "multi_model_v1"))
        if (split != "calibration" or evaluation_policies != () or selected
                or checkpoints != {} or not include_controls
                or controls != ("D0", calibration_control_policy)
                or inference_disable_tick is not None
                or ((replay_output_dir is not None or replay_seeds is not None
                    or replay_selection is not None) and not paired_replay_requested)
                or qualification_case is None
                or control_prediction_modes != {calibration_control_policy: expected_mode}
                or not valid_mode or candidate_k_override != 16
                or calibration_route_cost_points_per_second is None):
            raise ValueError("calibration control opt-in requires calibration split, empty checkpoints/policies, "
                "D0 plus exactly the selected D1/D2 reference, an explicit supported mode and frozen case, "
                "no inference fault, candidate_k=16, and explicit route-cost weight")
        if paired_replay_requested and (replay_output_dir is None or replay_selection is None
                or simulated_score_estimand is None or len(replay_selection) != 1
                or next(iter(replay_selection)) not in ("D0", calibration_control_policy)
                or next(iter(replay_selection.values())) != list(scored_seeds)):
            raise ValueError("paired calibration replay requires one D0/comparator policy, exact seeds, and approved score estimand")
        from .meco_qualification import protocol_cases
        frozen_cases = protocol_cases(scenario_id, "calibration", episodes=episodes,
            master_seed=master_seed)
        matching_cases = [case for case in frozen_cases
            if case.category in categories
            and {"layout_id": case.layout_id, "defender_family": case.defender_family,
                "start_condition": case.start_condition, "sensor_profile": case.sensor_profile}
                == qualification_case]
        if (len(categories) != 1 or len(matching_cases) != 1
                or tuple(episode_seeds or ()) != matching_cases[0].seeds):
            raise ValueError("calibration control evaluation must match one frozen calibration case and its exact seed tuple")
        frozen_case = matching_cases[0]
        calibration_control_case_binding = {"scenario_id": scenario_id,
            "master_seed": master_seed, "episodes": episodes,
            "category": frozen_case.category,
            "qualification_case": dict(qualification_case),
            "episode_seeds": list(frozen_case.seeds)}
    elif calibration_route_cost_points_per_second is not None:
        raise ValueError("calibration route-cost weight requires calibration_control_policy opt-in")
    if not selected and (controls != ("D0",) or not include_controls):
        if calibration_control_policy is None or not include_controls:
            raise ValueError("control-only evaluation permits D0 only unless calibration_control_policy is explicit")
    sources = source_identity()
    source_hash = hashlib.sha256(json.dumps(
        sources, sort_keys=True, allow_nan=False).encode()).hexdigest()
    score_estimand_approval = None
    score_estimand_approval_sha256 = None
    score_estimand_rules_sha256 = None
    if simulated_score_estimand is not None:
        from .meco_simulated_score_ledger import (RULES_SHA256,
            load_estimand_approval)
        score_estimand_approval, score_estimand_approval_sha256 = load_estimand_approval(
            simulated_score_estimand, source_identity_sha256=source_hash)
        score_estimand_rules_sha256 = RULES_SHA256
        score_estimand_path = Path(simulated_score_estimand)
    else:
        score_estimand_path = None
    if paired_replay_requested and any(
            not isinstance(binding.get("source_binding"), dict)
            or binding["source_binding"].get("source_identity_sha256") != source_hash
            or binding.get("estimand_approval_sha256") != score_estimand_approval_sha256
            for binding in paired_calibration_bindings.values()):
        raise ValueError("paired calibration campaign source or score approval differs from this evaluation")
    source_tree = source_tree_identity()
    source_tree_hash = hashlib.sha256(json.dumps(source_tree, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    loaded = {}
    policy_manifest = {}
    route_weight = None
    checkpoint_candidate_k = None
    common_training_config = None
    common_training_protocol = None
    protocol_initialized = False
    p_mode = None
    p_cost_scale = None
    p_points_per_grid_edge = None
    action_ticks = None
    learned_prediction_mode = None
    for track, path in checkpoints.items():
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload["config"]["track"] != track or payload["sources"] != sources:
            raise ValueError(f"{track} checkpoint is incompatible with current source")
        track_route_weight = float(payload["config"].get(
            "route_cost_points_per_second", ROUTE_COST_POINTS_PER_SECOND))
        track_candidate_k = int(payload["config"].get("candidate_k", 8))
        if track_candidate_k not in (8, 16, 32):
            raise ValueError(f"{track} checkpoint has invalid candidate_k")
        if not math.isfinite(track_route_weight) or track_route_weight < 0:
            raise ValueError(f"{track} checkpoint has invalid route cost weight")
        if route_weight is None:
            route_weight = track_route_weight
        elif track_route_weight != route_weight:
            raise ValueError("paired evaluation checkpoints must share one route-cost weight")
        if checkpoint_candidate_k is None:
            checkpoint_candidate_k = track_candidate_k
        elif track_candidate_k != checkpoint_candidate_k:
            raise ValueError("paired evaluation checkpoints must share one candidate_k")
        model = make_meco_track(track,
            candidate_features=CANDIDATE_BASE_FEATURE_COUNT).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        checkpoint_digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        checkpoint_protocol = payload.get("protocol_id")
        if not isinstance(checkpoint_protocol, str) or not checkpoint_protocol:
            raise ValueError(f"{track} checkpoint has no production protocol identity")
        loaded[track] = ProductionMecoAdapter(track, model,
            checkpoint_sha256=checkpoint_digest, protocol_id=checkpoint_protocol)
        track_config = payload.get("config", {})
        track_p_mode = track_config.get("p_mode", "learned")
        track_p_scale = float(track_config.get("p_cost_scale", 1.))
        track_p_edge_points = float(track_config.get("p_points_per_grid_edge", .05))
        track_action_ticks = int(track_config.get("action_ticks", 12))
        track_prediction_mode = track_config.get("prediction_mode", "cv")
        expected_prediction = prediction_contract(track_prediction_mode)
        if payload.get("prediction_contract", expected_prediction) != expected_prediction:
            raise ValueError(f"{track} checkpoint prediction provenance mismatch")
        if (track_p_mode not in ("learned", "p0")
                or track_prediction_mode not in PREDICTION_MODES
                or not all(math.isfinite(v) and v >= 0 for v in
                           (track_p_scale, track_p_edge_points))
                or track_action_ticks < 1):
            raise ValueError(f"{track} checkpoint has invalid Plan A penalty settings")
        if p_mode is None:
            p_mode, p_cost_scale = track_p_mode, track_p_scale
            p_points_per_grid_edge, action_ticks = track_p_edge_points, track_action_ticks
        elif (track_p_mode, track_p_scale, track_p_edge_points, track_action_ticks) != (
                p_mode, p_cost_scale, p_points_per_grid_edge, action_ticks):
            raise ValueError("paired checkpoints must share Plan A penalty and cadence settings")
        if learned_prediction_mode is None:
            learned_prediction_mode = track_prediction_mode
        elif learned_prediction_mode != track_prediction_mode:
            raise ValueError("paired checkpoints must share prediction_mode")
        immutable_track_config = {key: value for key, value in track_config.items()
                                 if key != "updates"}
        shared_track_config = {key: value for key, value in immutable_track_config.items()
                               if key not in ("track", "device", "output")}
        if common_training_config is None:
            common_training_config = shared_track_config
        elif shared_track_config != common_training_config:
            raise ValueError("paired checkpoints have different immutable training settings")
        if not protocol_initialized:
            common_training_protocol = checkpoint_protocol
            protocol_initialized = True
        elif checkpoint_protocol != common_training_protocol:
            raise ValueError("paired checkpoints have different supervisor protocols")
        policy_manifest[track] = {"policy_id": _model_id(model),
            "checkpoint_sha256": checkpoint_digest,
            "execution_device": execution_device,
            "input_contract_sha256": payload.get("input_contract_sha256"),
            "p_mode": track_p_mode if track in ("A1", "A2", "A3") else None,
            "p_cost_scale": track_p_scale if track in ("A1", "A2", "A3") else None,
            "p_points_per_grid_edge": (track_p_edge_points
                                       if track in ("A1", "A2", "A3") else None),
            "prediction_mode": track_prediction_mode,
            "prediction_contract": prediction_contract(track_prediction_mode),
            "training_protocol_id": payload.get("protocol_id"),
            "training_configuration_sha256": hashlib.sha256(json.dumps(
                immutable_track_config, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "source_identity_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    if calibration_control_policy is not None:
        route_weight = float(calibration_route_cost_points_per_second)
        action_ticks = 12
    candidate_k = (checkpoint_candidate_k if candidate_k_override is None
                   else candidate_k_override)
    if inference_disable_tick is not None:
        first_macro_boundary = ((inference_disable_tick + action_ticks - 1) //
                                action_ticks) * action_ticks
        scenario_horizon = math.ceil(definition.time_budget_s / OBSERVATION_DT)
        if first_macro_boundary >= scenario_horizon:
            raise ValueError("inference disable tick has no macro boundary before episode horizon")
    effective_prediction_mode = prediction_mode or learned_prediction_mode or "cv"
    effective_control_modes = effective_control_prediction_modes
    for track in ("A1", "A2", "A3"):
        if track not in policy_manifest:
            continue
        base = policy_manifest[track]
        policy_manifest[f"{track}_P0"] = {
            **base,
            "policy_id": f"{base['policy_id']}|p_mode=p0",
            "p_mode": "p0",
            "base_track": track,
        }
    for control in controls:
        control_policy_id = {"D0": "D0_production_deterministic",
                             "D1": "D1_hand_weighted",
                             "D2": "D2_kinematic_prediction"}[control]
        policy_manifest[control] = {"policy_id": control_policy_id, "checkpoint_sha256": None,
            "execution_device": execution_device,
            "input_contract_sha256": input_contract_digest(),
            "prediction_mode": effective_control_prediction_modes.get(control),
            "prediction_contract": (prediction_contract(effective_control_prediction_modes[control])
                if control in effective_control_prediction_modes else None),
            "source_identity_sha256": hashlib.sha256(
                json.dumps(sources, sort_keys=True, allow_nan=False).encode()).hexdigest()}
        if control == calibration_control_policy:
            policy_manifest[control]["calibration_candidate_configuration"] = {
                "candidate_k": candidate_k,
                "route_cost_points_per_second": route_weight,
                "action_ticks": action_ticks}
    if inference_disable_tick is not None:
        for track in selected:
            base = policy_manifest[track]
            policy_manifest[f"{track}__INFERENCE_DISABLED_TO_D1"] = {
                **base,
                "policy_id": f"{base['policy_id']}|inference_disabled_tick={inference_disable_tick}|fallback=D1",
                "base_policy": track,
                "inference_fault": "model_inference_disabled_at_macro_boundary_then_shared_D1"}
    protocol = hashlib.sha256(json.dumps({"sources": sources, "scenario": scenario_id,
        "categories": categories, "episodes": episodes, "master_seed": master_seed,
        "episode_seeds": list(episode_seeds) if episode_seeds is not None else None,
        "split": split,
        "qualification_protocol_id": qualification_protocol_id,
        "evaluation_binding": evaluation_binding,
        "qualification_case": qualification_case,
        "calibration_control_case_binding": calibration_control_case_binding,
        "candidate_k": {"checkpoint": checkpoint_candidate_k,
            "evaluation_override": candidate_k_override, "effective": candidate_k},
        "candidate_service": {"contract": CANDIDATE_BOUNDARY,
            "route_cost_points_per_second": ROUTE_COST_POINTS_PER_SECOND,
            "candidate_builder": "shared_C1_C2_D1_D2_route_feasible_candidates"},
        "prediction": {"learned_mode": effective_prediction_mode,
            "control_override": control_prediction_mode,
            "control_mode_overrides": control_prediction_modes,
            "control_modes": effective_control_modes,
            "contract": prediction_contract(effective_prediction_mode)},
        "calibration_control_policy": calibration_control_policy,
        "calibration_candidate_configuration": ({
            "candidate_k": candidate_k,
            "route_cost_points_per_second": route_weight,
            "action_ticks": action_ticks}
            if calibration_control_policy is not None else None),
        "inference_disable_tick": inference_disable_tick,
        "phase4_factor": phase4_factor,
        "phase4_contact": {"factor_id": phase4_contact_factor_id,
            "geometry_id": contact_geometry_id,
            "geometry_config": contact_geometry_config,
            "qualification_eligible": False,
            "recovery_outcome": "unavailable",
            "route_completion_outcome": "unavailable"},
        "execution_device": execution_device,
        "dt": OBSERVATION_DT, "action_ticks": action_ticks, "route_weight": route_weight,
        "plan_a_penalty": {"p_mode": p_mode, "p_cost_scale": p_cost_scale,
            "points_per_grid_edge": p_points_per_grid_edge,
            "projection": PENALTY_MAP_PROJECTION, "map_units": PENALTY_MAP_UNITS,
            "path_units": PENALTY_PATH_UNITS,
            **plan_a_ppo_contract_metadata(),
            "native_route_execution": "projected P map routed to focal slot-3 AD* on every replan and eligible depot retry; other robots unweighted; P0 passes no map",
            "runtime_route_validation": "pending_representative_simulator_replay"},
        "inference_disable_fault": {"requested_tick": inference_disable_tick,
            "fallback": "D1 deterministic scorer using checkpoint prediction mode",
            "trigger": "first macro decision boundary at or after requested physics tick"},
        "timing_instrumentation": {"clock": "synchronized host perf_counter wall time",
            "warmup_count": 0, "decision_boundary": "candidate/feature/model/scoring pipeline"},
        "D0": {"controller": "production deterministic offense", "decision_cadence": "native physics step (20 ms)",
               "execution": "records actual env.last_actions at every physics step"},
        "learned_and_D1_D2": {"decision_cadence": "12 physics steps per macro decision",
            "candidate_contract": "Plan A: fixed full legal-graph node/behavior support with model-independent shortest-path tree and behavior-specific descriptors; Plan C C1/C2 and controls D1/D2: one shared stable configurable K=8/16/32 route-qualified candidate table, including distance-ordered direct goals for sensed targets beyond native top-four, common action/route eligibility and prediction risk, one appended snapshot-revalidated continue row when committed, ten-macro expiry, target-loss/invalid-route/completion/failure/episode clearing, and a shared fixed 0.10-point switch-cost hysteresis. Selected collection tracks flow through the paired slot-3 collect goal; each physics step revalidates the buffer index, freshness, and <=1 m snapshot displacement, dynamically aims terminal intake heading at the current track position using native swept turn-clearance checks, and cancels the macro on loss; physical pickup remains geometric and non-exclusive; only candidate utility scoring differs"}},
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    eval_id = uuid.uuid4().hex
    output_path = Path(output) if output else None
    status_path = (output_path.with_name(output_path.name + ".status.json")
                   if output_path is not None else None)
    status_last_written = 0.
    status_last_completed = -1
    def update_evaluation_status(status, completed=0, *, dashboard_summary_sha256=None):
        nonlocal status_last_written, status_last_completed
        if status_path is None:
            return
        current = time.monotonic()
        if (status == "running" and completed == status_last_completed
                and current - status_last_written < 5.):
            return
        _atomic_json(status_path, {"schema_version": 1, "evaluation_id": eval_id,
            "protocol_id": protocol,
            "qualification_protocol_id": qualification_protocol_id,
            "evaluation_protocol_id": (evaluation_binding["protocol_id"]
                if evaluation_binding else schedule_case_binding["protocol_id"]
                if schedule_case_binding else None),
            "evaluation_protocol_sha256": (evaluation_binding["protocol_sha256"]
                if evaluation_binding else schedule_case_binding["protocol_sha256"]
                if schedule_case_binding else None),
            "evaluation_case": evaluation_binding,
            "schedule_case_binding": schedule_case_binding,
            "diagnostic_frontier_binding": diagnostic_frontier_binding,
            "diagnostic_only": split == "diagnostic",
            "qualification_eligible": False,
            "calibration_control_case_binding": calibration_control_case_binding,
            "exact_episode_seeds": (evaluation_binding["seeds"] if evaluation_binding else
                schedule_case_binding["seeds"] if schedule_case_binding else None),
            "split": split, "status": status,
            "execution_device": execution_device,
            "pid": os.getpid(), "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "output_file": output_path.name,
            "dashboard_summary_sha256": dashboard_summary_sha256,
            "progress": {"completed_records": completed,
                         "total_records": len(categories) * episodes *
                             (len(selected) * (2 if inference_disable_tick is not None else 1) +
                              sum(track in selected for track in ("A1", "A2", "A3")) +
                              (len(controls) if include_controls else 0))}})
        status_last_written = current
        status_last_completed = completed
    records = []
    replay_exports = []
    replay_code_state_snapshot = None
    write_replay_bundle = None
    if replay_output_dir is not None:
        from .dashboard_simulation import replay_code_state, write_dashboard_replay_bundle
        replay_code_state_snapshot = replay_code_state()
        write_replay_bundle = write_dashboard_replay_bundle
    component_timing_ms = {name: [] for name in (
        "legal_graph_build", "policy_decision_pipeline", "d1_fallback_pipeline",
        "simulator_macro_step")}
    timing_device = torch.device(execution_device["device"])
    timing_provenance = {
        "clock": "host perf_counter wall time, device synchronized before and after each measured component",
        "cuda_synchronization": timing_device.type == "cuda",
        "warmup_count": 0,
        "warmup_policy": "no unrecorded simulator warmup; all measured decisions belong to reported episodes",
        **execution_device,
        "device_name": (torch.cuda.get_device_name(timing_device)
                        if timing_device.type == "cuda" else "CPU"),
        "instrumentation": "component intervals include completion of asynchronous device work; decision pipeline is combined candidate/feature/model/scoring preparation and is not inference-only",
    }
    try:
        update_evaluation_status("running")
        for category in categories:
            for episode_index in range(episodes):
                seed = (episode_seeds[episode_index] if episode_seeds is not None
                        else master_seed + episode_index)
                # Same simulator reset seed and task for every control and model.
                base_policies = (list(selected) + [f"{track}_P0" for track in ("A1", "A2", "A3")
                    if track in selected] + (list(controls) if include_controls else []))
                policy_runs = [(policy, False) for policy in base_policies]
                if inference_disable_tick is not None:
                    policy_runs.extend((policy, True) for policy in selected)
                for policy, inference_fault_run in policy_runs:
                    record_policy = (f"{policy}__INFERENCE_DISABLED_TO_D1"
                                     if inference_fault_run else policy)
                    episode_id = _episode_id(f"eval-{eval_id}-{category}-{record_policy}",
                                             episode_index)
                    episode_wall_started = time.perf_counter()
                    is_production_d0 = policy == "D0"
                    env = make_curriculum_env(scenario_id, category, 1, device, seed,
                        production_d0=is_production_d0,
                        phase4_factor=phase4_factor,
                        phase4_opponent_policy_id=phase4_opponent_policy_id,
                        phase4_opponent_schedule=phase4_opponent_schedule,
                        phase4_opponent_seed=(seed if phase4_opponent_schedule is not None else None),
                        contact_geometry_id=contact_geometry_id,
                        contact_geometry_config=contact_geometry_config,
                        phase4_start_profile_assignment=phase4_start_profile_assignment,
                        phase4_focal_pose_assignment=phase4_focal_pose_assignment,
                        phase4_teammate_behavior_assignment=phase4_teammate_behavior_assignment,
                        **(qualification_case or {}))
                    score_event_recorder = None
                    if score_estimand_approval is not None:
                        from .meco_simulated_score_ledger import MecoScoreEventRecorder
                        score_context = _meco_score_ledger_context(env,
                            episode_id=episode_id, seed=seed, scenario_id=scenario_id,
                            source_identity_sha256=source_hash)
                        score_event_recorder = MecoScoreEventRecorder(
                            score_context, score_estimand_approval,
                            score_estimand_approval_sha256)
                        env._meco_score_event_recorder = score_event_recorder
                    direct_pursuit_capture = None
                    if phase4_direct_pursuit_binding is not None:
                        from .meco_phase4_direct_pursuit import DirectPursuitEpisodeCapture
                        manifest = policy_manifest[policy]
                        checkpoint_hash = manifest.get("checkpoint_sha256")
                        direct_pursuit_capture = DirectPursuitEpisodeCapture(
                            env, phase4_direct_pursuit_binding,
                            condition=("factor_treatment" if phase4_opponent_schedule
                                       is not None else "no_factor"),
                            episode_id=episode_id, source_identity_sha256=source_hash,
                            checkpoint_sha256=checkpoint_hash,
                            scenario_id=scenario_id, category=category,
                            qualification_case=qualification_case or {}, seed=seed,
                            split=split, policy_id=policy, execution_device=execution_device,
                            condition_id=condition_id)
                    trench_interception_capture = None
                    if phase4_trench_interception_binding is not None:
                        from .meco_phase4_trench_interception import TrenchInterceptionEpisodeCapture
                        trench_interception_capture = TrenchInterceptionEpisodeCapture(
                            env, phase4_trench_interception_binding,
                            condition=("factor_treatment" if phase4_opponent_schedule
                                       is not None else "no_factor"),
                            episode_id=episode_id, source_identity_sha256=source_hash,
                            checkpoint_sha256=policy_manifest[policy].get("checkpoint_sha256"),
                            policy_id=policy, seed=seed,
                            execution_device=execution_device, condition_id=condition_id)
                    phase4_focal_pose_record = None
                    if phase4_focal_pose_assignment is not None:
                        from .meco_phase4_context import phase4_focal_pose_evaluator_diagnostics
                        phase4_focal_pose_record = phase4_focal_pose_evaluator_diagnostics(
                            env, env.phase4_focal_pose_runtime)
                    episode_metric_ledger = None
                    replay_frames = []
                    try:
                        if (inference_disable_tick is not None and
                                inference_disable_tick >= env.horizon):
                            raise ValueError("inference_disable_tick must be below the "
                                f"evaluated environment horizon ({env.horizon} physics ticks)")
                        if is_production_d0 and env.control_modes[3] != "deterministic":
                            raise RuntimeError("production D0 evaluation did not receive the deterministic slot-3 controller")
                        graph = (None if is_production_d0 else
                                 None)
                        if not is_production_d0:
                            _timing_sync(env)
                            graph_started = time.perf_counter()
                            graph = build_legal_pose_graph(env, slot=3)
                            _timing_sync(env)
                            component_timing_ms["legal_graph_build"].append(
                                (time.perf_counter() - graph_started) * 1000.)
                        hidden = None
                        tracking_was_available = None
                        tracking_loss_reset_count = 0
                        episode_metric_ledger = MecoEpisodeMetricLedger(
                            env, route_tracking_available=not is_production_d0,
                            goal_tracking_available=not is_production_d0)
                        contact_progress_observer = (FirstRouteProgressInterval()
                            if contact_geometry_id is not None else None)
                        if contact_progress_observer is not None:
                            env._phase4_first_route_progress_observer = contact_progress_observer
                        capture_replay = (replay_output_dir is not None and
                            seed in replay_seed_set and
                            (replay_policy_seeds is None or
                             seed in replay_policy_seeds.get(record_policy, set())))
                        if capture_replay:
                            replay_frames.append(_capture_meco_replay_frame(
                                env, int(env.steps[0].item()), None))
                        commitment = None
                        actions_count = [0] * 8
                        routes_seen = []
                        executed_planner_route_trace = []
                        executed_policy_decision_trace = []
                        executed_action_trace = []
                        selected_action_trace = []
                        execution_failures = []
                        last_candidate_recall = None
                        elapsed = 0
                        pending_eval_goal = None
                        inference_fault_reached = False
                        inference_fault_actual_tick = None
                        pre_fault_action = None
                        post_fault_actions = []
                        post_fault_executed_actions = [0] * 8
                        post_fault_macro_count = 0
                        post_fault_safety = {"active_physics_ticks": 0,
                            "robot_contact_ticks": 0, "field_contact_ticks": 0,
                            "wall_contact_ticks": 0, "metrics_available": True}
                        post_fault_metrics_start = None
                        fault_fallback_count = 0
                        decision_latency_samples = []
                        inference_latency_samples = []
                        simulator_macro_latency_samples = []
                        policy_decision_count = 0
                        policy_fallback_count = 0
                        decision_prediction_mode = (
                            effective_prediction_mode
                            if policy in MECO_TRACKS or policy in ("A1_P0", "A2_P0", "A3_P0")
                            else effective_control_modes.get(policy))
                        while elapsed < env.horizon:
                            if (inference_fault_run and not inference_fault_reached
                                    and elapsed >= inference_disable_tick):
                                inference_fault_reached = True
                                inference_fault_actual_tick = elapsed
                                pre_fault_action = (selected_action_trace[-1]
                                                    if selected_action_trace else None)
                                hidden = None
                                tracking_was_available = None
                                commitment = None
                                pending_eval_goal = None
                                post_fault_metrics_start = _metric_snapshot(
                                    env.curriculum_metrics)
                            prepared = None
                            if is_production_d0:
                                # Native deterministic mode recomputes from
                                # current state at every 20 ms simulator step.
                                native_actions = torch.zeros((env.n, 6), device=env.device,
                                                             dtype=torch.long)
                                _, _, done, truncated, _ = env.step(
                                    native_actions, capture_info=False)
                                if direct_pursuit_capture is not None:
                                    direct_pursuit_capture.observe_physics_step()
                                if trench_interception_capture is not None:
                                    trench_interception_capture.observe_physics_step()
                                if contact_progress_observer is not None:
                                    planner_goal = env.planner.last_goal.reshape(
                                        env.n, 6, 2)[0, 3].tolist()
                                    planner_goal = [float(value) for value in planner_goal]
                                    goal_id = ("d0_native:" + hashlib.sha256(
                                        json.dumps(planner_goal, separators=(",", ":"),
                                            allow_nan=False).encode()).hexdigest()[:16]
                                        if all(math.isfinite(value) for value in planner_goal)
                                        else None)
                                    _observe_phase4_contact_route_progress(env,
                                        contact_progress_observer,
                                        selected_goal_id=goal_id,
                                        selected_goal_xy_m=planner_goal if goal_id else None)
                                executed = int(env.last_actions[0, 3].item())
                                episode_metric_ledger.observe_outer_step(executed)
                                executed_action_trace.append(executed)
                                actions_count[executed] += 1
                                elapsed += 1
                                terminal = bool(done[0].item()) or bool(truncated[0].item())
                                decision = (_d0_replay_decision(
                                    elapsed_ticks=elapsed, action_ticks=action_ticks,
                                    executed_actions=executed_action_trace,
                                    policy=record_policy, action=executed,
                                    candidate_k=candidate_k,
                                    prediction_mode=decision_prediction_mode,
                                    terminal=terminal) if capture_replay else None)
                                if decision is not None:
                                    decision["macro_end_tick"] = int(env.steps[0].item())
                                    replay_frames.append(_capture_meco_replay_frame(
                                        env, int(env.steps[0].item()), decision))
                                update_evaluation_status("running", len(records))
                                if bool(done[0].item()) or bool(truncated[0].item()):
                                    break
                                continue
                            obs = env._last_obs
                            prepared = None
                            use_d1_fallback = inference_fault_run and inference_fault_reached
                            if (policy in MECO_TRACKS or policy in ("A1_P0", "A2_P0", "A3_P0")) and not use_d1_fallback:
                                model_track = policy[:2] if policy.endswith("_P0") else policy
                                override_mode = "p0" if policy.endswith("_P0") else p_mode
                                _timing_sync(env)
                                decision_started = time.perf_counter()
                                prepared = _prepare_decision(model_track, loaded[model_track],
                                    env, graph, episode_id, hidden,
                                    route_cost_points_per_second=route_weight,
                                    action_ticks=action_ticks, candidate_k=candidate_k,
                                    deterministic=True, p_mode=override_mode,
                                    p_cost_scale=p_cost_scale,
                                    p_points_per_grid_edge=p_points_per_grid_edge,
                                    prediction_mode=decision_prediction_mode,
                                    record_path_penalty=True,
                                    commitment=commitment,
                                    tracking_was_available=tracking_was_available)
                                _timing_sync(env)
                                decision_latency = (time.perf_counter() - decision_started) * 1000.
                                component_timing_ms["policy_decision_pipeline"].append(decision_latency)
                                decision_latency_samples.append(decision_latency)
                                policy_decision_count += 1
                                provenance = prepared.get("inference_provenance")
                                if isinstance(provenance, dict):
                                    dispatch = provenance.get("dispatch_monotonic")
                                    completed = provenance.get("completed_monotonic")
                                    if (provenance.get("status") == "accepted" and
                                            isinstance(dispatch, (int, float)) and
                                            isinstance(completed, (int, float)) and
                                            completed >= dispatch):
                                        inference_latency_samples.append((completed - dispatch) * 1000.)
                                policy_fallback_count += int(bool(prepared.get("fallback_used")))
                                candidate_set = prepared["candidate_set"]
                                last_candidate_recall = (None if "support" in prepared
                                                         else candidate_set.candidate_recall)
                                choice = int(prepared["action"].item())
                                hidden = prepared["hidden_next"]
                                tracking_was_available = prepared["tracking_available"]
                                tracking_loss_reset_count += int(prepared["tracking_reset"])
                                action_source = "model"
                            else:
                                _timing_sync(env)
                                fallback_started = time.perf_counter()
                                candidate_set = build_route_feasible_candidates(env, obs, graph,
                                    episode_id=episode_id, action_ticks=action_ticks,
                                    candidate_k=candidate_k,
                                    prediction_mode=(effective_prediction_mode if use_d1_fallback
                                        else effective_control_modes.get(policy, "current")),
                                    commitment=commitment)
                                last_candidate_recall = candidate_set.candidate_recall
                                logits = _d1_d2_scores(candidate_set, env,
                                    "D1" if use_d1_fallback else policy,
                                    prediction_mode=(effective_prediction_mode if use_d1_fallback
                                        else effective_control_modes.get(policy, "current")),
                                    route_cost_points_per_second=route_weight)
                                choice = int(logits.argmax(-1).item())
                                _timing_sync(env)
                                decision_latency = (time.perf_counter() - fallback_started) * 1000.
                                component_timing_ms["d1_fallback_pipeline" if use_d1_fallback
                                    else "policy_decision_pipeline"].append(decision_latency)
                                decision_latency_samples.append(decision_latency)
                                policy_decision_count += 1
                                action_source = "d1_inference_disable_fallback" if use_d1_fallback else policy.lower()
                                if use_d1_fallback:
                                    fault_fallback_count += 1
                                    policy_fallback_count += 1
                            selected_candidate_id = _selected_policy_candidate_id(
                                prepared, candidate_set, choice)
                            routes_before_choice = len(routes_seen)
                            plan_a_support = (prepared is not None and
                                              "support" in prepared)
                            behavior_index = None
                            descriptor = None
                            selected_target_track = None
                            selected_target_xy = None
                            selected_goal_node = None
                            if prepared is not None and "support" in prepared:
                                support = prepared["support"]
                                action = int(support.action_ids[choice].item())
                                node = int(support.pair_node_ids[choice].item())
                                selected_goal_node = node
                                behavior_index = int(support.behavior_ids[choice].item())
                                descriptor = int(support.descriptor_ids[choice].item())
                                goal_xy = tuple(float(v) for v in
                                                support.goal_xy[node].tolist())
                                parent_chain = [node]
                                parent = int(support.parent_ids[node].item())
                                while parent >= 0:
                                    parent_chain.append(parent)
                                    parent = int(support.parent_ids[parent].item())
                                parent_chain.reverse()
                                routes_seen.append({"candidate_id": f"node:{node}:behavior:{behavior_index}:descriptor:{descriptor}",
                                    "support_order": SUPPORT_ORDER,
                                    "support_digest": support.support_digest,
                                    "inference_provenance": prepared.get("inference_provenance"),
                                    "policy_fallback": prepared.get("fallback_used", False),
                                    "behavior": ("drive", "intake", "score", "ferry", "defense")[behavior_index],
                                    "behavior_descriptor": descriptor,
                                    "goal_xy_m": list(goal_xy),
                                    "goal_heading_rad": float(support.goal_heading[node].item()),
                                    "graph_path_node_ids": parent_chain,
                                    "graph_path_points_m": [support.goal_xy[i].detach().cpu().tolist()
                                                            for i in parent_chain],
                                    "graph_path_heading_rad": [float(support.goal_heading[i].item())
                                                               for i in parent_chain],
                                    "root_connector_distance_m": support.root_connector_m,
                                    "graph_route_distance_m": float(support.distances_m[node].item()),
                                    "graph_route_cost_m_equivalent": float(support.route_cost[node].item()),
                                    "eta_s": float(prepared["route_cost_nodes"][node].item()) /
                                        max(route_weight, 1e-12) if route_weight > 0 else
                                        float(support.route_cost[node].item()) /
                                        max(float(env.sim.speed[0, 3]), .1),
                                    "risk_penalty_points": float(prepared["risk_cost_nodes"][node].item()),
                                    "prediction": prepared.get("prediction"),
                                    "ppo_path_penalty_points": float(prepared["path_cost"][node].item()),
                                    "p_mode": override_mode,
                                    "p_map_scale": p_cost_scale if override_mode == "learned" else 0.,
                                    "p_points_per_grid_edge": p_points_per_grid_edge,
                                    "route_feasibility": "model_independent_legal_graph_reachability"})
                            else:
                                action = candidate_set.action_indices[choice]
                                cols = {name: i for i, name in enumerate(candidate_set.table.feature_names)}
                                selected_row = candidate_set.table.features[choice]
                                goal_xy = (float(selected_row[cols["goal_x_m"]]),
                                           float(selected_row[cols["goal_y_m"]]))
                                routes_seen.append({**candidate_set.route_records[choice],
                                    "inference_provenance": (prepared.get("decision", {}).get(
                                        "inference_provenance") if prepared is not None else None)})
                                selected_target_track = candidate_set.route_records[
                                    choice].get("sensor_track_buffer_index")
                                selected_goal_node = candidate_set.route_records[
                                    choice].get("legal_node_index")
                                selected_target_xy = candidate_set.route_records[
                                    choice].get("source_fuel_xy_m")
                            goal_heading = (float(support.goal_heading[node].item())
                                            if prepared is not None and "support" in prepared
                                            else None)
                            forced_eval_goal = pending_eval_goal is not None
                            if forced_eval_goal:
                                action = pending_eval_goal["action"]
                                goal_xy = pending_eval_goal["goal_xy"]
                                goal_heading = pending_eval_goal["goal_heading"]
                                selected_candidate_id = (pending_eval_goal.get("route_id") or
                                                         selected_candidate_id)
                                selected_target_track = None
                                selected_target_xy = None
                                if len(routes_seen) > routes_before_choice:
                                    routes_seen.pop()
                            actions_count[action] += 1
                            selected_action_trace.append(action)
                            if use_d1_fallback:
                                post_fault_actions.append(action)
                            _timing_sync(env)
                            macro_started = time.perf_counter()
                            (_, ended, timeout, actual_trace, heading_error,
                             goal_distance, heading_opportunity,
                             native_preempted, selected_target_lost) = _run_macro_step(
                                env, action, action_ticks, goal_xy,
                                goal_heading=goal_heading,
                                goal_track_index=selected_target_track,
                                goal_track_source_xy=selected_target_xy,
                                allow_action7_traversal=(prepared is not None and
                                                          "support" in prepared and action == 7),
                                planner_traversal_cost_override=(
                                    prepared.get("planner_traversal_cost")
                                    if (prepared is not None and not use_d1_fallback and
                                        not prepared.get("fallback_used", False)) else None),
                                planner_traversal_cost_scale=(
                                    prepared.get("planner_traversal_cost_scale", 1.)
                                    if (prepared is not None and not use_d1_fallback and
                                        not prepared.get("fallback_used", False)) else 1.),
                                metric_ledger=episode_metric_ledger,
                                contact_progress_observer=contact_progress_observer,
                                direct_pursuit_capture=direct_pursuit_capture,
                                trench_interception_capture=trench_interception_capture,
                                contact_progress_goal_id=(selected_candidate_id or
                                    ("selected_goal:" + json.dumps(
                                        [round(float(v), 6) for v in goal_xy],
                                        separators=(",", ":")) if goal_xy is not None else None)),
                                safety_counts=(post_fault_safety if use_d1_fallback else None))
                            if policy in ("A1", "A2", "A3", "A1_P0", "A2_P0", "A3_P0"):
                                planner = env.planner
                                path_batch = planner.last_path.reshape(
                                    env.n, 6, planner.last_path.shape[-2], 2)
                                lengths_batch = planner.last_lengths.reshape(env.n, 6)
                                goals_batch = planner.last_goal.reshape(env.n, 6, 2)
                                route_length = int(lengths_batch[0, 3].item())
                                route_points = path_batch[0, 3, :route_length].detach().cpu().tolist()
                                observed_goal = goals_batch[0, 3].detach().cpu().tolist()
                                p0_macro = policy.endswith("_P0")
                                applied_signature = getattr(env,
                                    "_last_applied_planner_traversal_signature", None)
                                applied_map = (None if p0_macro or applied_signature is None
                                               else applied_signature["map"])
                                map_sha256 = None
                                map_scale = None
                                map_shape = None
                                map_dtype = None
                                map_goal = None
                                map_action = None
                                if applied_map is not None:
                                    # Match _run_macro_step's frozen map: cast to the
                                    # simulator pose dtype before cloning and routing.
                                    map_tensor = torch.as_tensor(applied_map,
                                        device=env.device, dtype=env.sim.pose.dtype
                                        ).detach().clone().contiguous().cpu()
                                    map_bytes = map_tensor.numpy().tobytes()
                                    map_shape = list(map_tensor.shape)
                                    map_dtype = str(map_tensor.dtype)
                                    map_identity = json.dumps({"shape": map_shape,
                                        "dtype": map_dtype}, sort_keys=True).encode() + b"\0" + map_bytes
                                    map_sha256 = hashlib.sha256(map_identity).hexdigest()
                                    map_scale = float(applied_signature["scale"])
                                    map_goal = list(applied_signature["goal"])
                                    map_action = int(applied_signature["action"])
                                executed_planner_route_trace.append({
                                    "macro_index": len(executed_planner_route_trace),
                                    "start_tick": elapsed,
                                    "end_tick": elapsed + len(actual_trace),
                                    "selected_action": int(action),
                                    "executed_action_trace": list(actual_trace),
                                    "action_source": action_source,
                                    "selected_goal_xy_m": [float(goal_xy[0]), float(goal_xy[1])],
                                    "observed_goal_xy_m": [float(v) for v in observed_goal],
                                    "p_mode": "p0" if p0_macro else p_mode,
                                    "p_map_applied": applied_map is not None,
                                    "p_map_sha256": map_sha256,
                                    "p_map_shape": map_shape,
                                    "p_map_dtype": map_dtype,
                                    "p_map_scale": map_scale,
                                    "p_map_applied_goal_xy_m": map_goal,
                                    "p_map_applied_action": map_action,
                                    "route_length": route_length,
                                    "route_points_m": route_points,
                                })
                            if policy in MECO_TRACKS:
                                inference = (prepared.get("inference_provenance")
                                    if prepared is not None else None)
                                if inference is None and prepared is not None:
                                    inference = prepared.get("decision", {}).get(
                                        "inference_provenance")
                                inference_outcome = None
                                if inference and isinstance(loaded.get(policy), ProductionMecoAdapter):
                                    counters = loaded[policy].inference_counters
                                    seq = inference.get("request_sequence")
                                    inference_outcome = next((row for row in reversed(
                                        counters.get("recent_outcomes", []))
                                        if row.get("request_sequence") == seq), None)
                                candidate_id = selected_candidate_id
                                selected_goal = [float(goal_xy[0]), float(goal_xy[1])]
                                executed_policy_decision_trace.append({
                                    "schema": "meco-executed-policy-decision-v1",
                                    "candidate_id": candidate_id,
                                    "selected_action": int(action),
                                    "selected_goal_xy_m": selected_goal,
                                    "start_tick": int(elapsed),
                                    "end_tick": int(elapsed + len(actual_trace)),
                                    "actual_tick_actions": list(actual_trace),
                                    "divergence": _evaluation_action_was_remasked(
                                        action, actual_trace),
                                    "fallback": bool(not forced_eval_goal and prepared is not None and
                                        prepared.get("fallback_used", False)),
                                    "status": ("pending_goal_continuation" if forced_eval_goal else
                                        "inference_disabled_fallback" if use_d1_fallback else
                                        "deterministic_fallback" if prepared is not None and
                                        prepared.get("fallback_used", False) else "selected"),
                                    "inference_outcome": inference_outcome,
                                    "inference_provenance": inference})
                            if use_d1_fallback:
                                post_fault_macro_count += 1
                                for executed_action in actual_trace:
                                    post_fault_executed_actions[executed_action] += 1
                            _timing_sync(env)
                            macro_elapsed_ms = (time.perf_counter() - macro_started) * 1000.
                            component_timing_ms["simulator_macro_step"].append(macro_elapsed_ms)
                            simulator_macro_latency_samples.append(macro_elapsed_ms)
                            goal_complete = (goal_distance is not None and goal_distance <= .04 and
                                             (heading_error is None or heading_error <= .1))
                            remasked = _evaluation_action_was_remasked(action,
                                actual_trace)
                            hard_failure = (remasked or native_preempted or
                                            selected_target_lost)
                            if policy in ("C1", "C2", "D1", "D2"):
                                selected_id = candidate_set.table.candidate_ids[choice]
                                if selected_id.startswith("continue:"):
                                    commitment = candidate_set.commitment
                                else:
                                    route = candidate_set.route_records[choice]
                                    source_xy = route.get("source_fuel_xy_m")
                                    sequence = candidate_set.observation.sequence
                                    commitment = CandidateCommitment(
                                        candidate_id=selected_id, action_index=int(action),
                                        goal_variant=int(route.get("goal_variant", 0)),
                                        source_xy_m=(None if source_xy is None else
                                            (float(source_xy[0]), float(source_xy[1]))),
                                        created_sequence=sequence,
                                        expires_sequence=sequence + CANDIDATE_BOUNDARY[
                                            "commitment_expiry_macro_actions"] * action_ticks)
                                if goal_complete or hard_failure or ended or timeout:
                                    commitment = None
                            route_id = (pending_eval_goal["route_id"] if forced_eval_goal
                                        else routes_seen[-1].get("candidate_id")
                                        if routes_seen else None)
                            policy_fallback = bool(not forced_eval_goal and
                                prepared is not None and prepared.get("fallback_used", False))
                            fallback_reason = (prepared.get("fallback_reason")
                                if prepared is not None else None)
                            selected_goal_id = (selected_candidate_id or route_id)
                            goal_identity_is_finite = (goal_xy is not None and
                                len(goal_xy) == 2 and all(math.isfinite(float(value))
                                    for value in goal_xy) and
                                (goal_heading is None or
                                 math.isfinite(float(goal_heading))))
                            if not selected_goal_id and goal_identity_is_finite:
                                goal_identity_payload = {"goal_xy_m": [float(goal_xy[0]),
                                    float(goal_xy[1])], "goal_heading_rad": goal_heading,
                                    "action": int(action)}
                                selected_goal_id = "selected_goal:" + hashlib.sha256(
                                    json.dumps(goal_identity_payload, sort_keys=True,
                                        separators=(",", ":"), allow_nan=False
                                    ).encode()).hexdigest()
                            goal_lifecycle_id = _phase4_route_goal_lifecycle_id(
                                action=int(action),
                                plan_a_node_id=(selected_goal_node if plan_a_support else None),
                                behavior_id=(behavior_index if plan_a_support else None),
                                descriptor_id=(descriptor if plan_a_support else None),
                                goal_xy=goal_xy,
                                assigned_goal_id=(pending_eval_goal.get("route_id")
                                    if forced_eval_goal and pending_eval_goal else None))
                            goal_exclusion_reasons = []
                            if policy_fallback:
                                goal_exclusion_reasons.append("policy_inference_fallback")
                            if use_d1_fallback:
                                goal_exclusion_reasons.append("inference_disabled_d1_fallback")
                            if remasked:
                                goal_exclusion_reasons.append("action_remasked")
                            if native_preempted:
                                goal_exclusion_reasons.append("native_goal_preempted_selection")
                            if selected_target_lost:
                                goal_exclusion_reasons.append("selected_target_lost")
                            if not actual_trace:
                                goal_exclusion_reasons.append("no_executed_outer_action")
                            episode_metric_ledger.observe_route(route_id)
                            episode_metric_ledger.observe_goal_attempt(
                                action=action, route_id=route_id, goal_xy=goal_xy,
                                goal_heading=goal_heading,
                                terminal_goal_distance_m=goal_distance,
                                terminal_heading_error_rad=heading_error,
                                terminal_pose_xy=(float(env.sim.pose[0, 3, 0].item()),
                                    float(env.sim.pose[0, 3, 1].item())),
                                terminal_pose_heading=float(env.sim.pose[0, 3, 2].item()),
                                heading_alignment_opportunity=heading_opportunity,
                                selected_goal_id=selected_goal_id,
                                goal_lifecycle_id=goal_lifecycle_id,
                                macro_index=len(selected_action_trace) - 1,
                                episode_id=episode_id,
                                source_identity_sha256=source_hash,
                                geometry_id=contact_geometry_id,
                                execution_valid=not goal_exclusion_reasons,
                                execution_exclusion_reasons=tuple(goal_exclusion_reasons),
                                executed_action_trace=(actual_trace
                                    if (phase4_route_outcome_factor_ids or
                                        phase4_contact_factor_id is not None or
                                        (phase4_factor or {}).get("kind") in (
                                            "feinting_opponent",
                                            "abrupt_opponent_policy_change"))
                                    else None))
                            if policy_fallback or hard_failure:
                                execution_failures.append({
                                    "selected_action": action,
                                    "executed_action_trace": actual_trace,
                                    "action_was_remasked": remasked,
                                    "policy_inference_fallback": policy_fallback,
                                    "fallback_reason": fallback_reason,
                                    "terminal_heading_error": heading_error,
                                    "terminal_goal_distance_m": goal_distance,
                                    "heading_alignment_opportunity": heading_opportunity,
                                    "reason": _evaluation_execution_failure_reason(
                                        action_was_remasked=remasked,
                                        native_preempted=native_preempted,
                                        selected_target_lost=selected_target_lost,
                                        policy_fallback=policy_fallback,
                                        fallback_reason=fallback_reason),
                                    "native_goal_preempted": native_preempted,
                                    "selected_target_track_lost": selected_target_lost,
                                    "policy_fallback": (prepared.get("fallback_used", False)
                                                        if prepared is not None else False),
                                    "route_id": route_id})
                                if pending_eval_goal is not None:
                                    pending_eval_goal = None
                            elif plan_a_support and not goal_complete and not forced_eval_goal:
                                pending_eval_goal = {"action": action, "goal_xy": goal_xy,
                                    "goal_heading": goal_heading, "route_id": route_id}
                            elif goal_complete and forced_eval_goal:
                                pending_eval_goal = None
                            elapsed += action_ticks
                            if capture_replay:
                                decision = {"kind": "policy_decision", "policy": record_policy,
                                    "action": int(action), "action_source": action_source,
                                    "candidate_id": route_id,
                                    "goal_xy_m": [float(goal_xy[0]), float(goal_xy[1])],
                                    "goal_heading_rad": goal_heading,
                                    "macro_start_tick": int(elapsed - action_ticks),
                                    "macro_end_tick": int(env.steps[0].item()),
                                    "executed_actions": list(actual_trace),
                                    "candidate_k": candidate_k,
                                    "prediction_mode": decision_prediction_mode,
                                    "p_mode": ("p0" if policy.endswith("_P0") else
                                               p_mode if policy in ("A1", "A2", "A3") else None),
                                    "inference_disabled": bool(use_d1_fallback)}
                                if policy in MECO_TRACKS:
                                    decision.update({
                                        "candidate_id": selected_candidate_id,
                                        "goal_xy_m": [float(goal_xy[0]), float(goal_xy[1])]})
                                replay_frames.append(_capture_meco_replay_frame(
                                    env, int(env.steps[0].item()), decision))
                            update_evaluation_status("running", len(records))
                            if ended or timeout:
                                if pending_eval_goal is not None:
                                    execution_failures.append({
                                        "selected_action": pending_eval_goal["action"],
                                        "executed_action_trace": [],
                                        "terminal_goal_distance_m": goal_distance,
                                        "terminal_heading_error": heading_error,
                                        "heading_alignment_opportunity": heading_opportunity,
                                        "reason": "selected_goal_incomplete_at_episode_end",
                                        "route_id": pending_eval_goal["route_id"]})
                                    pending_eval_goal = None
                                break
                        outcomes = env.curriculum_metrics.outcomes()
                        metrics = {name: value.detach().cpu().reshape(-1)[0].item()
                                   for name, value in outcomes.items()}
                        episode_metric_ledger.close_collection_bout(
                            "episode_finalization")
                        metric_ledger_record = episode_metric_ledger.to_record()
                        metrics.update(metric_ledger_record["safety_metrics"])
                        phase4_runtime_record = dict(env.phase4_runtime_metrics)
                        if (phase4_factor and phase4_factor.get("kind") ==
                                "stationary_blockage"):
                            slot = phase4_factor["blocker_slot"]
                            target = env.stationary_blockage_assigned_pose
                            pose = env.sim.pose[:, slot]
                            distance = torch.linalg.vector_norm(
                                pose[:, :2] - target[:, :2], dim=-1)
                            angle_delta = torch.atan2(torch.sin(
                                pose[:, 2] - target[:, 2]), torch.cos(
                                pose[:, 2] - target[:, 2])).abs()
                            prior = env.phase4_runtime_metrics[
                                "stationary_blockage_pose_valid"]
                            phase4_runtime_record["stationary_blockage_pose_valid"] = bool((
                                prior & (distance <= phase4_factor["pose_tolerance_m"]) &
                                (angle_delta <= phase4_factor[
                                    "heading_tolerance_rad"])).all().item())
                        diagnostic_performance = {
                            "scope": "diagnostic_only_not_qualification_primary_gates",
                            "end_to_end_wall_ms": (time.perf_counter() - episode_wall_started) * 1000.,
                            "route_policy_pipeline_ms": _timing_summary(decision_latency_samples),
                            "inference_ms": _timing_summary(inference_latency_samples),
                            "simulator_macro_ms": _timing_summary(simulator_macro_latency_samples),
                            "decision_count": policy_decision_count,
                            "fallback_count": policy_fallback_count,
                            "fallback_rate": (policy_fallback_count / policy_decision_count
                                if policy_decision_count else None),
                            "component_semantics": {
                                "route_policy_pipeline_ms": "synchronized wall time for candidate, observation, feature, inference orchestration, and scoring work; not a route-only kernel measurement",
                                "inference_ms": "accepted asynchronous model dispatch-to-completion time; excludes deterministic fallback requests",
                                "simulator_macro_ms": "synchronized wall time to execute one 12-physics-step macro",
                                "end_to_end_wall_ms": "wall time from environment construction through final per-record metric collection"},
                            "fallback_semantics": "deterministic inference fallback or explicitly injected inference-disable handoff per policy decision"}
                        post_fault_metrics = None
                        if inference_fault_run and inference_fault_reached:
                            progress_fields = ("scores", "acquisitions", "deliveries",
                                "team_conversions", "staged_conversions",
                                "initial_staged_conversions", "inactive_conversions",
                                "inactive_deliveries", "post_boundary_scores",
                                "post_boundary_deliveries", "completed_cycles",
                                "score_cycles", "ferry_cycles")
                            post_fault_metrics = {
                                "final_episode_success": metrics["success"],
                                "final_episode_useful_throughput": metrics["useful_throughput"],
                                "progress_delta": {name: metrics[name] -
                                    post_fault_metrics_start[name]
                                    for name in progress_fields
                                    if name in metrics and name in post_fault_metrics_start},
                                "route_seen_at_episode_end": metrics.get("route_seen"),
                            }
                        fault_pair_id = (hashlib.sha256(json.dumps(
                            {"evaluation_id": eval_id, "scenario": scenario_id,
                             "category": category, "seed": seed, "policy": policy,
                             "split": split}, sort_keys=True).encode()).hexdigest()
                            if inference_disable_tick is not None and policy in selected
                            else None)
                        phase4_route_completion_records = ({
                            factor_id: phase4_route_completion_record(
                                metric_ledger_record.get("goal_attempts"), binding={
                                    "episode_id": episode_id,
                                    "policy_id": record_policy,
                                    "checkpoint_policy_id": policy_manifest[
                                        record_policy]["policy_id"],
                                    "checkpoint_sha256": policy_manifest[
                                        record_policy]["checkpoint_sha256"],
                                    "scenario_id": scenario_id,
                                    "category": category,
                                    "execution_device": json.dumps(execution_device,
                                        sort_keys=True, separators=(",", ":"),
                                        allow_nan=False),
                                    "condition_id": condition_id or "unpaired_evaluation",
                                    "case_id": hashlib.sha256(json.dumps({
                                        "evaluation_case": evaluation_case,
                                        "qualification_case": qualification_case,
                                        "category": category}, sort_keys=True,
                                        separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest(),
                                    "seed": int(seed), "split": split,
                                    "factor_id": factor_id,
                                    "condition": phase4_route_condition(factor_id,
                                        phase4_factor=phase4_factor,
                                        contact_factor_id=phase4_contact_factor_id),
                                    "source_identity_sha256": source_hash,
                                    "goal_attempt_coverage_available":
                                        metric_ledger_record.get(
                                            "goal_attempt_coverage_available") is True})
                            for factor_id in phase4_route_outcome_factor_ids
                        } if phase4_route_outcome_factor_ids else None)
                        record = {"scenario_id": scenario_id, "category": category,
                            "diagnostic_performance": diagnostic_performance,
                            "input_contract_sha256": input_contract_digest(),
                            "seed": seed, "policy": record_policy,
                            "paired_baseline_policy": policy if inference_fault_run else None,
                            "paired_record_policy": (policy if inference_fault_run else
                                record_policy if fault_pair_id else None),
                            "inference_disable_pair_id": fault_pair_id,
                            "inference_disable_fault": ({
                                "requested_tick": inference_disable_tick,
                                "actual_trigger_tick": inference_fault_actual_tick,
                                "reached": inference_fault_reached,
                                "trigger_semantics": "first macro decision boundary at or after requested physics tick",
                                "pre_fault_selected_action": pre_fault_action,
                                "post_fault_selected_actions": post_fault_actions,
                                "fallback": "deterministic D1 hand-weighted candidate scorer using the checkpoint prediction_mode",
                                "prediction_mode": decision_prediction_mode,
                                "model_inference_disabled_after_trigger": inference_fault_reached,
                                "cleared_state": ["Plan A pending goal", "C1/C2/D1/D2 commitment", "recurrent hidden state"],
                                "fallback_count": fault_fallback_count,
                                "post_fault_macro_count": post_fault_macro_count,
                                "post_fault_executed_action_counts": post_fault_executed_actions,
                                "post_fault_outcome_and_progress_delta": post_fault_metrics,
                                "post_fault_safety_ticks": ({key: value for key, value
                                    in post_fault_safety.items() if not key.startswith("_")}
                                    if inference_fault_reached else None),
                                "post_fault_safety_scope": {
                                    "robot_contact_ticks": "low-level physics ticks with a robot-pair contact involving controlled slot 3",
                                    "field_contact_ticks": "low-level physics ticks with field contact for controlled slot 3",
                                    "wall_contact_ticks": "low-level physics ticks with wall contact for controlled slot 3",
                                    "active_physics_ticks": "active low-level physics ticks for controlled slot 3",
                                    "claim_scope": "observed contact events, not policy-rule violations",
                                },
                                "recovery_claim": "not inferred; fallback activation and post-trigger outcome/safety are reported separately",
                            } if inference_fault_run else None),
                            "action_source": ("production_deterministic" if is_production_d0
                                              else action_source),
                            "controller_cadence": ("native_20ms_physics_step" if is_production_d0
                                                    else "12_physics_step_macro"),
                            "action_count_semantics": ("executed_per_physics_step" if is_production_d0
                                                       else "model_then_d1_fallback_per_macro" if inference_fault_run
                                                       else "selected_node_behavior_pair_per_macro" if policy in ("A1", "A2", "A3", "A1_P0", "A2_P0", "A3_P0")
                                                       else "selected_candidate_per_macro"),
                            "policy_id": policy_manifest[record_policy]["policy_id"],
                            "checkpoint_sha256": policy_manifest[record_policy]["checkpoint_sha256"],
                            "execution_device": execution_device,
                            "p_mode": ("p0" if policy.endswith("_P0") else
                                       p_mode if policy in ("A1", "A2", "A3") else None),
                            "prediction_mode": decision_prediction_mode,
                            "prediction_contract": (prediction_contract(decision_prediction_mode)
                                if decision_prediction_mode in PREDICTION_MODES else None),
                            "protocol_id": protocol,
                            "qualification_protocol_id": qualification_protocol_id,
                            "evaluation_protocol_id": (evaluation_binding["protocol_id"]
                                if evaluation_binding else None),
                            "evaluation_protocol_sha256": (evaluation_binding["protocol_sha256"]
                                if evaluation_binding else None),
                            "evaluation_case_id": (evaluation_binding["case_ids"]
                                if evaluation_binding else None),
                            "split": split,
                            **(qualification_case or {}),
                            "source_identity_sha256": source_hash,
                            "executed_policy_decision_trace_schema": (
                                "meco-executed-policy-decision-v1" if policy in MECO_TRACKS else None),
                            "safety_metrics_schema": metric_ledger_record["safety_metrics_schema"],
                            "safety_metrics_source_identity_sha256": source_hash,
                            "actions": actions_count, "executed_action_trace": executed_action_trace,
                            "selected_action_trace": selected_action_trace,
                            "execution_failures": execution_failures,
                            "elapsed_ticks": elapsed,
                            "metrics": metrics, "candidate_recall": last_candidate_recall,
                            "candidate_k": candidate_k,
                            "tracking_loss_reset_count": tracking_loss_reset_count,
                            "tracking_loss_reset_policy": TRACK_LOSS_RESET_POLICY,
                            "phase4_factor": phase4_factor,
                            "phase4_start_profile": ({
                                "factor_id": phase4_start_profile_assignment["factor_id"],
                                "registry_profile_id": phase4_start_profile_assignment[
                                    "registry_profile_id"],
                                "runtime_profile_name": phase4_start_profile_assignment[
                                    "runtime_profile_name"],
                                "profile_config_sha256": phase4_start_profile_assignment[
                                    "profile_config_sha256"],
                                "assignment_sha256": phase4_start_profile_assignment[
                                    "assignment_sha256"],
                                "profile_applied_before_observation": bool(
                                    env.phase4_start_profile_applied),
                                "validated_pose_world_count": int(
                                    env.phase4_start_profile_validated_world_count),
                                "validated_pose_legal_world_count": int(sum(
                                    env.phase4_start_profile_legality or ())),
                                "validated_pose_world_indices": list(range(
                                    int(env.phase4_start_profile_validated_world_count))),
                                "validated_pose_world_index_semantics":
                                    "zero_based_simulator_batch_index",
                                "per_world_legal": list(
                                    env.phase4_start_profile_legality or ()),
                                "legal_start_validation_rate": float(
                                    env.phase4_start_profile_validation_rate),
                                "outcome_metrics": {
                                    name: metrics[name] for name in (
                                        "success", "scores", "useful_throughput",
                                        "completed_cycles")},
                                "qualification_eligible": False,
                            } if phase4_start_profile_assignment is not None else None),
                            "phase4_focal_pose": phase4_focal_pose_record,
                            "phase4_teammate_behavior": (
                                phase4_teammate_behavior_record(
                                    phase4_teammate_behavior_assignment,
                                    env.phase4_teammate_behavior_runtime,
                                    episode_id=episode_id, scenario_id=scenario_id,
                                    category=category, seed=seed,
                                    policy_id=policy_manifest[record_policy]["policy_id"],
                                    evaluation_policy_key=record_policy,
                                    condition_id=condition_id,
                                    source_identity_sha256=source_hash,
                                    outcome_metrics=metrics,
                                    expected_action_ticks=int(
                                        env.sim.physics_substep_index[0].item()),
                                    checkpoint_sha256=policy_manifest[
                                        record_policy]["checkpoint_sha256"],
                                    execution_device=execution_device)
                                if phase4_teammate_behavior_assignment is not None else None),
                            "phase4_opponent_policy": ({
                                "policy_id": phase4_opponent_policy_id,
                                "factor_id": phase4_opponent_schedule["factor_id"],
                                "policy_family": phase4_opponent_schedule["policy_family"],
                                "schedule_sha256": phase4_opponent_schedule[
                                    "schedule_sha256"],
                                "policy_config_sha256": phase4_opponent_schedule[
                                    "policy_config_sha256"],
                                "scenario_id": scenario_id,
                                "category": category,
                                "seed": seed,
                                "condition": "factor_treatment",
                                "qualification_case": qualification_case,
                                "condition_id": condition_id,
                                "source_identity_sha256": source_hash,
                                "checkpoint_sha256": policy_manifest[record_policy][
                                    "checkpoint_sha256"],
                                "execution_device": execution_device,
                                "assignment_sha256": next(item["assignment_sha256"]
                                    for item in phase4_opponent_schedule[
                                        "per_episode_assignment"]
                                    if item["seed"] == seed and
                                       item["category"] == category),
                                "inputs": "own policy_pose plus validity-masked sensed opponent_pose/opponent_velocity and static scenario configuration",
                                "runtime": env.phase4_opponent_runtime,
                                "configured_slots": phase4_opponent_schedule[
                                    "configured_pressure_slots"][category],
                                "applied_slots": env.phase4_opponent_runtime[
                                    "applied_slots"],
                                "applied_command_calls_by_slot": env.phase4_opponent_runtime[
                                    "applied_command_calls_by_slot"],
                                "applied_event_coverage_complete": bool(
                                    env.phase4_opponent_runtime["applied_slots"] ==
                                    phase4_opponent_schedule[
                                        "configured_pressure_slots"][category] and
                                    all(env.phase4_opponent_runtime[
                                        "applied_command_calls_by_slot"].get(str(slot)) ==
                                        int(env.sim.physics_substep_index[0].item())
                                        for slot in phase4_opponent_schedule[
                                            "configured_pressure_slots"][category])),
                                "episode_outcome_metrics": {name: metrics.get(name)
                                    for name in ("success", "scores",
                                        "useful_throughput", "completed_cycles")},
                                "interception_rate": None,
                                "paired_outcome_by_family": None,
                                "qualification_eligible": False,
                                "outcome_semantics": "whole-team episode metrics while this policy was assigned; no interception or per-policy causal contribution is inferred",
                                "qualification_eligible": False,
                            } if phase4_opponent_schedule is not None else None),
                            "phase4_direct_pursuit": (
                                direct_pursuit_capture.finish(
                                    episode_censored=(int(env.sim.physics_substep_index[0].item())
                                        < int(env.horizon)))
                                if direct_pursuit_capture is not None else None),
                            "phase4_trench_interception": (
                                trench_interception_capture.finish(
                                    episode_censored=(int(env.sim.physics_substep_index[0].item())
                                        < int(env.horizon)))
                                if trench_interception_capture is not None else None),
                            "phase4_pair_contact_event": (
                                phase4_pair_contact_event_record(env.sim,
                                    factor_id=phase4_contact_factor_id)
                                if phase4_contact_factor_id is not None else None),
                            "phase4_route_completion_by_factor": phase4_route_completion_records,
                            "phase4_contact_exposure": ({
                                "factor_id": "contact.robot_bump",
                                "geometry_id": contact_geometry_id,
                                "geometry_config": contact_geometry_config,
                                "exposure_assigned": env.contact_geometry_runtime[
                                    "exposure_assigned"],
                                "pair_slots": env.contact_geometry_runtime["pair_slots"],
                                "opponent_action": env.contact_geometry_runtime[
                                    "opponent_action"],
                                "opponent_action_source": env.contact_geometry_runtime[
                                    "opponent_action_source"],
                                "slot2_env_control_mode": env.contact_geometry_runtime[
                                    "slot2_env_control_mode"],
                                "approach_action_ticks": int(env.contact_geometry_runtime[
                                    "approach_action_ticks"][0].item()),
                                "focal_action_overridden": False,
                                "route_seen": bool(env.contact_geometry_runtime[
                                    "route_seen"][0].item()),
                                "route_seen_time_s": (None if float(env.contact_geometry_runtime[
                                    "route_seen_time_s"][0].item()) < 0. else float(
                                        env.contact_geometry_runtime["route_seen_time_s"][0].item())),
                                "qualification_eligible": False,
                                "recovery_time_after_bump_s": None,
                                "route_completion_rate": None,
                                "selected_goal_endpoint_completion_rate":
                                    phase4_selected_goal_endpoint_completion_record(
                                        metric_ledger_record.get("goal_attempts"),
                                        episode_id=episode_id,
                                        source_identity_sha256=source_hash,
                                        geometry_id=contact_geometry_id),
                                "status": "exposure_route_crossing_and_selected_goal_endpoint_diagnostic_only_recovery_unavailable",
                            } if contact_geometry_id is not None else None),
                            "first_route_progress_interval": (
                                contact_progress_observer.record()
                                if contact_progress_observer is not None else None),
                            "phase4_factor_runtime": ({
                                **phase4_runtime_record,
                                "fuel_track_confidence_mean": float(
                                    env.track_confidence[0, 3][env.track_mask[0, 3]].mean().item())
                                    if bool(env.track_mask[0, 3].any()) else None,
                                "confidence_semantics": "requested and applied caps are reported separately; refresh count and reduced sensed-track values prove application; final mean is over sensed learner tracks; positions, velocities, ages, masks, dropout and noise are unchanged",
                            } if phase4_factor is not None else None),
                            "phase4_realization_capture": phase4_factor_exposure(
                                phase4_factor, phase4_runtime_record,
                                metrics, source_identity=sources, split=split,
                                seed=seed, scenario_id=scenario_id,
                                category=category,
                                qualification_case=qualification_case,
                                episode_id=episode_id, policy_id=record_policy,
                                route_completion_by_factor=phase4_route_completion_records,
                                contact_factor_id=phase4_contact_factor_id),
                            "phase4_response_metrics": ({name: metrics.get(name)
                                for name in ("success", "useful_throughput",
                                    "route_switches", "route_aba_proxy",
                                    "safety_robot_contact_rate",
                                    "safety_field_contact_rate",
                                    "safety_wall_contact_rate")}
                                if phase4_factor is not None else None),
                            "base_policy": policy if inference_fault_run else None,
                            "meco_metric_ledger": metric_ledger_record,
                            "route_trace": routes_seen,
                            "executed_planner_route_trace": executed_planner_route_trace,
                            "executed_policy_decision_trace": (
                                executed_policy_decision_trace if policy in MECO_TRACKS else None),
                            "episode_id": episode_id}
                        score_capture = None
                        if score_event_recorder is not None:
                            final_score_tick = int(env.steps[0].item())
                            score_terminal_state = ("complete" if final_score_tick >= env.horizon
                                                    else "censored")
                            terminal_score_counts = [int(value) for value in
                                env.fuel_score_count[0].detach().cpu().tolist()]
                            score_capture = score_event_recorder.finish(
                                terminal_tick=final_score_tick,
                                terminal_state=score_terminal_state,
                                terminal_score_counts=terminal_score_counts)
                            record.update({
                                "simulated_score_ledger": score_capture["ledger"],
                                "simulated_match_score": score_capture["ledger"][
                                    "simulated_match_score"],
                                "official_match_points": None,
                                "qualification_eligible": False,
                                "score_terminal_counters": terminal_score_counts,
                                "score_estimand_approval_sha256":
                                    score_estimand_approval_sha256,
                            })
                        if capture_replay:
                            final_tick = int(env.steps[0].item())
                            if not replay_frames or replay_frames[-1]["tick"] != final_tick:
                                replay_frames.append(_capture_meco_replay_frame(env,
                                    final_tick, {"kind": "episode_end", "policy": record_policy,
                                        "terminal": True, "action": None,
                                        "macro_end_tick": final_tick}))
                            _validate_meco_replay_alignment(record, replay_frames,
                                final_tick=final_tick)
                            record_digest = _evaluation_record_digest(record)
                            route_digest = _canonical_digest({
                                "route_trace": record["route_trace"],
                                "executed_planner_route_trace":
                                    record["executed_planner_route_trace"]})
                            outcome_digest = _canonical_digest({
                                "metrics": record["metrics"],
                                "safety_metrics": record["meco_metric_ledger"].get(
                                    "safety_metrics"),
                                "execution_failures": record["execution_failures"]})
                            paired_replay_binding = (paired_calibration_bindings[seed]
                                if paired_replay_requested else None)
                            replay_condition_id = (_paired_calibration_replay_condition_id(
                                paired_replay_binding) if paired_replay_binding is not None else
                                condition_id or _canonical_digest({
                                "scenario_id": scenario_id, "category": category,
                                "split": split, "policy": record_policy,
                                "candidate_k": candidate_k,
                                "prediction_mode": decision_prediction_mode,
                                "phase4_factor": phase4_factor,
                                "contact_geometry_id": contact_geometry_id,
                                "phase4_start_profile_assignment": phase4_start_profile_assignment,
                                "contact_geometry_config": contact_geometry_config,
                                "qualification_eligible": False,
                                "inference_disable_tick": inference_disable_tick}))
                            replay_identity = _canonical_digest({
                                "evaluation_id": eval_id, "condition_id": replay_condition_id,
                                "record_digest": record_digest, "seed": seed,
                                "policy": record_policy})
                            simulation_id = f"scenario-meco-{replay_identity[:24]}"
                            track = (policy[:2] if policy.endswith("_P0") else
                                     policy if policy in MECO_TRACKS else None)
                            manifest = policy_manifest[record_policy]
                            training_source_hash = (manifest.get("source_identity_sha256")
                                if track is not None else source_hash)
                            source_checkpoint_outdated = (
                                training_source_hash != record["source_identity_sha256"])
                            code_state = replay_code_state_snapshot
                            replay_record = {
                                "replay_contract_schema": "dashboard-replay-generation-v1",
                                "input_contract_sha256": input_contract_digest(),
                                "simulation_id": simulation_id, "task": "3v3",
                                "matchup": scenario_id, "scenario_id": scenario_id,
                                "category": category, "seed": seed, "scenario_seed": seed,
                                "split": split, "qualification_case": qualification_case,
                                "behavior_mode": "match", "policy": record_policy,
                                "policy_id": record["policy_id"],
                                "evaluation_id": eval_id,
                                "condition_id": replay_condition_id,
                                "phase4_start_profile_assignment":
                                    phase4_start_profile_assignment,
                                "evaluation_report_path": (str(output_path.resolve())
                                    if output_path is not None else None),
                                "evaluation_record_index": len(records),
                                "evaluation_record_digest": record_digest,
                                "outcome_digest": outcome_digest,
                                "safety_digest": _canonical_digest(
                                    record["meco_metric_ledger"].get("safety_metrics")),
                                "route_digest": route_digest,
                                "checkpoint": (checkpoints.get(track) if track else None),
                                "checkpoint_sha256": record["checkpoint_sha256"],
                                "checkpoint_source_sha256": training_source_hash,
                                "checkpoint_source_outdated": source_checkpoint_outdated,
                                "replay_source_sha256": source_hash,
                                "source_identity_sha256": source_hash,
                                "source_tree_identity_sha256": source_tree_hash,
                                "protocol_id": record["protocol_id"],
                                "qualification_protocol_id": qualification_protocol_id,
                                "training_protocol_id": manifest.get("training_protocol_id"),
                                "training_configuration_sha256": manifest.get(
                                    "training_configuration_sha256"),
                                "candidate_k": candidate_k,
                                "prediction_mode": decision_prediction_mode,
                                "p_mode": record["p_mode"],
                                "device": execution_device,
                                "execution_device": execution_device,
                                "simulation_device": device,
                                "code_state": code_state,
                                "evaluation_record": {"scenario_id": scenario_id,
                                    "category": category, "seed": seed,
                                    "policy": record_policy, "split": split,
                                    "metrics": record["metrics"],
                                    "safety_metrics": record["meco_metric_ledger"].get(
                                        "safety_metrics"),
                                    "elapsed_ticks": record["elapsed_ticks"],
                                    "route_digest": route_digest,
                                    "outcome_digest": outcome_digest},
                                "field": {"length": env.sim.field_length,
                                    "width": env.sim.field_width,
                                    "alliance_zone_depth": env.alliance_zone_depth,
                                    "elements": [box.as_dict() for box in env.field_boxes]},
                                "robot_teams": env.team_ids.tolist(),
                                "robot_count": 6, "robots_per_alliance": 3,
                                "robot_types": list(env.robot_types),
                                "control_modes": list(env.control_modes),
                                "robot_roles": list(env.robot_roles),
                                "simulation_ticks": final_tick,
                                "simulated_seconds": final_tick * env.dt,
                                "dt": env.dt, "capture_stride": (
                                    _d0_replay_action_ticks(action_ticks)
                                    if is_production_d0 else action_ticks),
                                "action_repeat": (1 if is_production_d0 else action_ticks),
                                "perception_interval": env.perception_interval,
                                "simulation_constraints": {"fuel_physics": True,
                                    "perception_simulation": True,
                                    "max_fuel_per_robot": list(
                                        env.robot_fuel_capacity_values)},
                                "replay_provenance": {"evaluation_id": eval_id,
                                    "condition_id": replay_condition_id,
                                    "evaluation_report_path": (str(output_path.resolve())
                                        if output_path is not None else None),
                                    "evaluation_record_index": len(records),
                                    "evaluation_record_digest": record_digest,
                                    "replay_source_sha256": source_hash,
                                    "checkpoint_sha256": record["checkpoint_sha256"],
                                    "checkpoint_source_sha256": training_source_hash,
                                    "checkpoint_source_outdated": source_checkpoint_outdated,
                                    "protocol_id": record["protocol_id"],
                                    "qualification_protocol_id": qualification_protocol_id,
                                    "training_protocol_id": manifest.get("training_protocol_id"),
                                    "training_configuration_sha256": manifest.get(
                                        "training_configuration_sha256"),
                                    "scenario_id": scenario_id, "category": category,
                                    "split": split, "seed": seed,
                                    "qualification_case": qualification_case,
                                    "policy": record_policy, "policy_id": record["policy_id"],
                                    "candidate_k": candidate_k,
                                    "prediction_mode": decision_prediction_mode,
                                    "p_mode": record["p_mode"], "device": execution_device,
                                    "route_digest": route_digest,
                                    "outcome_digest": outcome_digest,
                                    "safety_digest": _canonical_digest(
                                        record["meco_metric_ledger"].get("safety_metrics"))},
                                "checkpoint_source_outdated": source_checkpoint_outdated,
                                "metrics": record["metrics"],
                                "policy_decisions": [frame["meco_policy_decision"]
                                    for frame in replay_frames[1:]],
                                "scenarios": [{"id": scenario_id,
                                    "label": (f"MECO {record_policy}: {scenario_id}/{category} "
                                        f"{split} seed {seed}"),
                                    "frames": replay_frames}],
                                "frames": replay_frames}
                            if paired_replay_requested:
                                replay_record["paired_calibration_binding"] = paired_replay_binding
                                replay_record["paired_calibration_worker_identity"] = paired_calibration_worker_identity
                            if score_capture is not None:
                                _assert_score_estimand_unchanged(
                                    score_estimand_path,
                                    score_estimand_approval_sha256)
                                score_summary = score_capture["ledger"][
                                    "simulated_match_score"]
                                replay_record.update({
                                    "score_event_stream": score_capture["event_stream"],
                                    "simulated_score_ledger": score_capture["ledger"],
                                    "simulated_match_score": score_summary,
                                    "score_terminal_counters": score_capture[
                                        "terminal_score_counts"],
                                    "score_estimand_approval_sha256":
                                        score_estimand_approval_sha256,
                                })
                                replay_record["evaluation_record"].update({
                                    "simulated_score_ledger_sha256": score_capture[
                                        "ledger"]["ledger_sha256"],
                                    "simulated_match_score": score_summary,
                                    "official_match_points": None,
                                    "qualification_eligible": False,
                                    "source_identity_sha256": source_hash,
                                })
                                from .meco_simulated_score_ledger import (
                                    validate_replay_score_binding)
                                validate_replay_score_binding(replay_record, record,
                                    score_capture["terminal_score_counts"],
                                    score_estimand_approval)
                            result_path = Path(replay_output_dir) / f"{simulation_id}.result.json"
                            write_replay_bundle(result_path, replay_record)
                            record["dashboard_replay"] = {
                                "simulation_id": simulation_id,
                                "result_path": str(result_path.resolve()),
                                "evaluation_report_path": (str(output_path.resolve())
                                    if output_path is not None else None),
                                "evaluation_record_index": len(records),
                                "evaluation_record_digest": record_digest,
                                "frame_count": len(replay_frames),
                                "checkpoint_source_outdated": source_checkpoint_outdated}
                            replay_exports.append(record["dashboard_replay"])
                        records.append(record)
                        update_evaluation_status("running", len(records))
                    finally:
                        if episode_metric_ledger is not None:
                            episode_metric_ledger.close_collection_bout(
                                "evaluation_error" if sys.exc_info()[0] else
                                "episode_finalization")
                        close = getattr(env, "close", None)
                        if callable(close):
                            close()
        report = {"schema_version": 1, "evaluation_id": eval_id,
            "input_contract_sha256": input_contract_digest(),
            "protocol_id": protocol,
            "qualification_protocol_id": qualification_protocol_id,
            "evaluation_protocol_id": (evaluation_binding["protocol_id"]
                if evaluation_binding else schedule_case_binding["protocol_id"]
                if schedule_case_binding else None),
            "evaluation_protocol_sha256": (evaluation_binding["protocol_sha256"]
                if evaluation_binding else schedule_case_binding["protocol_sha256"]
                if schedule_case_binding else None),
            "evaluation_case": evaluation_binding,
            "schedule_case_binding": schedule_case_binding,
            "diagnostic_frontier_binding": diagnostic_frontier_binding,
            "diagnostic_only": split == "diagnostic",
            "calibration_control_case_binding": calibration_control_case_binding,
            "exact_episode_seeds": (evaluation_binding["seeds"] if evaluation_binding else
                schedule_case_binding["seeds"] if schedule_case_binding else None),
            "evaluation_design": ("frozen_protocol_held_out" if evaluation_binding
                else "diagnostic_unbound"),
            "sources": sources,
            "tracking_loss_reset_policy": TRACK_LOSS_RESET_POLICY,
            "safety_metrics_schema": "meco-safety-rates-v1",
            "safety_metrics_source_identity_sha256": source_hash,
            "contact_geometry_id": contact_geometry_id,
            "contact_geometry_config": contact_geometry_config,
            "phase4_start_profile_assignment": phase4_start_profile_assignment,
            "phase4_focal_pose_assignment": phase4_focal_pose_assignment,
            "phase4_teammate_behavior_assignment": phase4_teammate_behavior_assignment,
            "qualification_eligible": False,
            "episode_metric_ledger_schema": {
                "schema_version": 4,
                "record_field": "meco_metric_ledger",
                "scope": "one ledger per evaluation record, focal slot 3",
                "outer_step_sampling": "fuel event tensors are sampled after each outer CurriculumEnv.step; contact metrics use cumulative low-level TensorPhysics.step counters",
                "collection_action_bouts": "contiguous focal slot-3 executed actions 0..3; each bout stores inclusive action sequence/counts, zero-based start and exclusive end outer-step ticks, acquired quantity and closure reason; acquisition success means focal fuel_acquired_event increased during the bout; missing action/acquisition signals make bout metrics unavailable",
                "collection_action_bout_aggregate": "count, acquired count, unsuccessful_collection_action_bouts and acquired proportion are computed over finalized bouts; unsuccessful_collection_action_bouts counts only bouts with authoritative acquired quantity known to be zero; zero-bout proportion is null",
                "unsuccessful_collection_action_bouts": "descriptive finalized focal collection-bout outcome, not causal intake failure",
                "hub_denied": "remains a separate simulator fuel_denied event and is not interpreted as failed collection; abandoned fuel remains unavailable without an authoritative event producer",
                "safety_metrics_schema": "meco-safety-rates-v1",
                "safety_metrics": {
                    "safety_robot_contact_rate": "focal slot-3 robot-contact ticks / focal active physics ticks",
                    "safety_field_contact_rate": "focal slot-3 field-contact ticks / focal active physics ticks",
                    "safety_wall_contact_rate": "focal slot-3 wall-contact ticks / focal active physics ticks",
                },
                "safety_contact_scope": "observed per-class contacts, not policy-rule violations; missing counters remain null",
                "distance": "XY chord proxy between outer-step pose samples; not exact curved path length",
                "route_switches": "selected candidate route ID changes; not AD* replans",
                "route_aba_proxy": "three consecutive selected route IDs A-B-A; not interception or feint detection",
                "heading_alignment_attempts": "macro rows with a planned heading and any within-0.04m XY opportunity during that macro",
                "heading_joint_endpoint": "success requires terminal XY distance <= 0.04m and terminal heading error <= 0.1 rad; an earlier in-macro XY opportunity alone is not endpoint success",
                "unavailable_counts": ["abandoned fuel", "failed collection"],
            },
            "executed_planner_route_trace_schema": {
                "schema_version": 1,
                "record_field": "executed_planner_route_trace",
                "scope": "post-macro snapshot of latest focal slot-3 AD* route for A1/A2/A3 and P0 evaluation records",
                "route_points": "planner.last_path[slot 3, :last_lengths[slot 3]] after the macro; latest planner route snapshot, not every within-macro replan",
                "selected_goal": "macro-selected goal passed to the executor",
                "observed_goal": "slot-3 planner.last_goal after macro execution",
                "actions": "selected macro action and actual per-physics-step env.last_actions trace",
                "p_map_sha256": "SHA256 of canonical shape/dtype metadata plus contiguous CPU tensor bytes for the effective frozen map used by the latest focal route; null when that route was replanned without P, including P0",
                "p_map_applied": "whether the latest focal route was produced with a P map",
                "p_map_shape_and_dtype": "tensor shape and dtype paired with p_map_sha256; null when no map was applied, including P0",
                "p_map_scale": "scale used with the map for the latest focal route; null when no map was applied, including P0",
                "p_map_applied_goal_and_action": "goal and action signature paired with the map used for the latest focal route; may differ from the current macro selected goal/action when the planner retains its prior route",
                "trajectory_claim": "route planner snapshot, not an exact actuator trajectory",
            },
            "training_protocol_ids": ([common_training_protocol]
                if common_training_protocol else []),
            "shared_training_configuration": common_training_config,
            "scenario_id": scenario_id, "categories": list(categories),
            "episodes_per_category": episodes, "master_seed": master_seed,
            "candidate_k": candidate_k,
            "execution_device": execution_device,
            "checkpoint_sha256": {track: policy_manifest[track]["checkpoint_sha256"]
                for track in selected},
            "candidate_k_checkpoint": checkpoint_candidate_k,
            "candidate_k_evaluation_override": candidate_k_override,
            "prediction": {"model": PREDICTION_MODEL_ID,
                "learned_mode": effective_prediction_mode,
                "control_modes": effective_control_modes,
                "explicit_learned_override": prediction_mode,
                "explicit_control_override": control_prediction_mode,
                "contract": prediction_contract(effective_prediction_mode)},
            "inference_disable_fault": ({
                "requested_tick": inference_disable_tick,
                "trigger_boundary": "first_macro_boundary_at_or_after_requested_tick",
                "fallback_policy": "D1",
                "trained_policies": list(selected),
                "paired_normal_and_fault_records": True}
                if inference_disable_tick is not None else None),
            "phase4_factor": phase4_factor,
            "phase4_opponent_schedule": phase4_opponent_schedule,
            "phase4_direct_pursuit_binding": phase4_direct_pursuit_binding,
            "phase4_trench_interception_binding": phase4_trench_interception_binding,
            "phase4_opponent_outcomes": ({
                "interception_rate": None,
                "policy_family_coverage": None,
                "paired_outcome_by_family": None,
                "qualification_eligible": False,
            } if phase4_opponent_schedule is not None else None),
            "plan_a_candidate_k_used": False,
            "plan_a_support_order": SUPPORT_ORDER,
            "plan_a_support_contract": "fixed node-major eight-descriptor tensor per legal graph node; masked by one model-independent graph shortest-path pass and vectorized native behavior endpoint/target and heading constraints; carrying dumper ferry pairs are masked because native release may preempt their selected endpoint",
            "plan_a_prediction_risk_provenance": "risk is computed for each legal graph node with at least one support-valid behavior/descriptor pair; finite nonnegative ETA is required for every such node; unreachable and fully masked nodes receive inert zero risk placeholders and are excluded from CV endpoint forecasts and horizon min/max; CV contract horizon is the maximum valid-node ETA and is null when no valid endpoint exists; current mode uses observed pose with contract horizon zero",
            "plan_a_heading_contract": "selected Plan A goals persist across macro steps until XY and heading completion; C1/C2/D1/D2 retain their original per-macro action semantics. The simulator pauses at the selected XY goal, samples the rectangular chassis sweep every 5 degrees, requires chassis translation to settle, and applies a point-forecast clearance check against known teammate poses and the observed opponent using constant velocity over estimated turn time plus stopping drift. This estimate has no uncertainty inflation, and the 5-degree static samples do not prove continuous swept clearance; native route and collision checks remain authoritative. Slot-3 score/pass events before the selected goal are recorded as execution failures and excluded from Plan A actor credit. The native slot-3 dumper ferry action can release fuel anywhere; Plan A masks that carrying dumper ferry pair, and the event censor remains for any actual early slot-3 pass. Intake uses target capture geometry, score uses hub-facing tolerance, and executable non-dumper ferry actions use the native release-facing tolerance",
            "seeds": (list(episode_seeds) if episode_seeds is not None else
                      [master_seed + i for i in range(episodes)]),
            "tracks": list(selected), "ablations": [f"{track}_P0" for track in ("A1", "A2", "A3")
                if track in selected],
            "controls": list(controls) if include_controls else [],
            "prediction_parameters": {"model": PREDICTION_MODEL_ID,
                "parameters": {"cv_max_speed_mps": PREDICTION_MAX_SPEED_MPS,
                    "cv_max_track_age_s": PREDICTION_MAX_AGE_S,
                    "fixed_default_horizon_s": 1.0,
                    "candidate_horizon": "route_eta_s",
                    "plan_a_horizon": "maximum_finite_nonnegative_eta_over_nodes_with_at_least_one_support_valid_pair; unreachable_and_fully_masked_nodes_excluded; null_when_no_valid_endpoint",
                    "stale_cv_fallback": "observed_current_pose",
                    "invalid_or_absent_observation": "no_risk_penalty"}},
            "plan_a_penalty": {"trained_mode": p_mode, "p0_ablation_mode": "p0",
                "cost_scale": p_cost_scale,
                "points_per_grid_edge": p_points_per_grid_edge,
                "projection": PENALTY_MAP_PROJECTION, "map_units": PENALTY_MAP_UNITS,
                "path_units": PENALTY_PATH_UNITS,
                **plan_a_ppo_contract_metadata(),
                "native_route_execution": "projected P map routed to focal slot-3 AD* on every replan and eligible depot retry; other robots unweighted; P0 passes no map",
                "runtime_route_validation": "pending_representative_simulator_replay"},
            "control_definitions": {"D0": "production deterministic offense controller; native 20 ms physics-step cadence; actual executed actions recorded; not constrained by the MECO candidate table",
                "D1": "hand-weighted scorer using current-pose prediction by default (explicit evaluation override applies); 12-step macro cadence",
                "D2": "D1 scorer using bounded constant-velocity prediction by default (explicit evaluation override applies); 12-step macro cadence"},
            "policy_manifest": policy_manifest,
            "paired": True, "split": split, "diagnostic": episodes < 100,
            "inference_disable_fault_design": ({
                "requested_tick": inference_disable_tick,
                "paired_same_seed_normal_records": True,
                "fault_variants": list(selected),
                "trigger": "first macro decision boundary at or after requested physics tick",
                "unreached_trigger_is_not_success": True,
                "fallback": "D1 deterministic scorer using the checkpoint prediction mode",
                "state_cleared": ["Plan A pending goal", "C1/C2/D1/D2 commitment", "recurrent hidden state"],
                "post_fault_safety": "controlled-slot-3 robot, field, and wall contact physics-tick counts from cumulative counters",
                "post_fault_recovery": "not inferred; fallback action counts, outcome/progress deltas, and safety ticks are reported",
            } if inference_disable_tick is not None else None),
            "component_timing": {
                "provenance": timing_provenance,
                "coverage_note": "episode construction/reset, checkpoint loading, and native D0 per-physics-step execution are outside these component samples",
                "components": {name: _timing_summary(values)
                               for name, values in component_timing_ms.items()},
                "component_boundaries": {
                    "legal_graph_build": "legal pose graph construction per non-D0 episode",
                    "policy_decision_pipeline": "candidate/observation/features/model/scoring preparation, or D1/D2 control candidate generation and scoring; intentionally not labeled inference-only",
                    "d1_fallback_pipeline": "fallback candidate generation and deterministic D1 scoring after fault trigger",
                    "simulator_macro_step": "one selected action's complete physics-tick macro execution",
                },
            },
            "qualification": qualification_case is not None,
            "qualification_case": qualification_case,
            "simulated_score_ledger": ({
                "schema": "meco-simulated-score-ledger-v1",
                "rules_id": "meco_simulated_fuel_score_v1",
                "rules_sha256": score_estimand_rules_sha256,
                "estimand_approval_sha256": score_estimand_approval_sha256,
                "official_match_points": None,
                "qualification_eligible": False,
                "collection_requires_full_episode_replay": True,
            } if score_estimand_approval is not None else None),
            "candidate_boundary": CANDIDATE_BOUNDARY,
            "diagnostic_boundary": "matched validation sample only; promotion requires separate held-out gates",
            "source_identity_sha256": source_hash,
            "dashboard_verified": False, "records": records,
            "dashboard_replays": replay_exports,
            "replay_selection": (None if replay_policy_seeds is None else {
                policy: sorted(seeds) for policy, seeds in replay_policy_seeds.items()}),
            "generated_utc": datetime.now(timezone.utc).isoformat()}
        _assert_score_estimand_unchanged(score_estimand_path,
                                         score_estimand_approval_sha256)
        if output:
            _atomic_json(Path(output), report)
            dashboard_summary_sha256 = None
            try:
                from .meco_dashboard import (_stable_file_sha256,
                    _write_dashboard_comparison_summary)
                if _write_dashboard_comparison_summary(Path(output), report):
                    dashboard_summary_sha256, _, _ = _stable_file_sha256(
                        Path(str(output) + ".dashboard-summary.json"),
                        max_bytes=1024 * 1024)
            except (OSError, TypeError, ValueError):
                print("Dashboard comparison summary unavailable; raw report remains authoritative.",
                      file=sys.stderr)
            update_evaluation_status("complete", len(records),
                dashboard_summary_sha256=dashboard_summary_sha256)
        return report
    except BaseException:
        try:
            update_evaluation_status("failed", len(records))
        except OSError:
            pass
        raise


def _parser():
    parser = argparse.ArgumentParser(description="Matched MECO five-track research trainer")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--track", choices=MECO_TRACKS, required=True)
    train.add_argument("--scenario", default="pickup_simple")
    train.add_argument("--category", default="single")
    train.add_argument("--updates", type=int, default=1000)
    train.add_argument("--rollout-steps", type=int, default=32)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--action-ticks", type=int, default=12)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device, for example cuda:0")
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--gamma", type=float, default=.993)
    train.add_argument("--gae-lambda", type=float, default=.95)
    train.add_argument("--clip-coef", type=float, default=.2)
    train.add_argument("--entropy-coef", type=float, default=.01)
    train.add_argument("--value-coef", type=float, default=.5)
    train.add_argument("--max-grad-norm", type=float, default=.5)
    train.add_argument("--candidate-k", type=int, choices=(8, 16, 32), default=16)
    train.add_argument("--route-cost-points-per-second", type=float,
                        default=ROUTE_COST_POINTS_PER_SECOND)
    train.add_argument("--p-mode", choices=("learned", "p0"), default="learned")
    train.add_argument("--p-cost-scale", type=float, default=1.)
    train.add_argument("--p-points-per-grid-edge", type=float, default=.05)
    train.add_argument("--prediction-mode", choices=PREDICTION_MODES, default="cv")
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--warm-start",
        help="initialize model weights from a versioned supervised artifact; optimizer/RNG/cursor remain fresh")
    train.add_argument("--qualification-protocol",
        help="frozen qualification artifact required with --warm-start")
    train.add_argument("--stop-file")
    train.add_argument("--abort-file",
        help="invocation-scoped sibling-failure signal; honored after an atomic update checkpoint")
    train.add_argument("--invocation-id",
        help="supervisor invocation identity bound to a sibling abort marker")
    train.add_argument("--run-id")
    train.add_argument("--protocol-id")
    train.add_argument("--task-assignment-json")
    train.add_argument("--expected-source-sha256",
        help="supervisor source identity digest required for supervised campaign workers")
    train.add_argument("--native-extension-attestation-json",
        help="supervisor-pinned loaded HIP binary identity required for campaign workers")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--scenario", default="pickup_simple")
    ev.add_argument("--categories", default="single,cluster")
    ev.add_argument("--episodes", type=int, default=1)
    ev.add_argument("--seed", type=int, default=20261024)
    ev.add_argument("--split", choices=("diagnostic", "train", "validation", "test"),
        default="diagnostic")
    ev.add_argument("--evaluation-protocol",
        help="immutable protocol binding for an exact validation/test evaluation")
    ev.add_argument("--evaluation-case-json",
        help="JSON object selecting the frozen layout/family/start/sensor stratum")
    ev.add_argument("--episode-seeds",
        help="comma-separated exact held-out seed list; required with --evaluation-protocol")
    ev.add_argument("--schedule-case-binding-json",
        help="frozen campaign schedule case binding for durable evaluations")
    ev.add_argument("--diagnostic-frontier-binding-json",
        help="manifest/source/configuration-bound diagnostic checkpoint frontier")
    ev.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device, for example cuda:0")
    ev.add_argument("--prediction-mode", choices=PREDICTION_MODES,
        help="override learned A/C prediction mode; default preserves checkpoint mode")
    ev.add_argument("--control-prediction-mode", choices=PREDICTION_MODES,
        help="override D1 and D2 together; default preserves D1=current and D2=cv")
    ev.add_argument("--control-prediction-modes", type=json.loads,
        help='JSON mapping for per-control modes, e.g. \'{"D1":"current","D2":"multi_model_v1"}\'; cannot combine with --control-prediction-mode')
    ev.add_argument("--candidate-k-override", type=int, choices=(8, 16, 32),
        help="evaluation-only candidate K for C tracks and controls")
    ev.add_argument("--inference-disable-tick", type=int,
        help="also run same-seed learned-policy pairs that disable inference at the first macro boundary at or after this physics tick")
    ev.add_argument("--replay-output-dir", type=Path,
        help="opt-in dashboard replay directory, typically <run>/.scenario-progress")
    ev.add_argument("--replay-seeds", type=int, nargs="+",
        help="exact subset of evaluated seeds to export as dashboard replay bundles")
    ev.add_argument("--simulated-score-estimand", type=Path,
        help="human-approved immutable FUEL-only estimand JSON; requires replay export for every evaluated policy and seed")
    ev.add_argument("--output", required=True)
    for track in MECO_TRACKS:
        ev.add_argument(f"--{track.lower()}-checkpoint", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "train":
        config = MecoTrainConfig(track=args.track, scenario_id=args.scenario,
            category=args.category, updates=args.updates, rollout_steps=args.rollout_steps,
            epochs=args.epochs, action_ticks=args.action_ticks, seed=args.seed,
            device=args.device, learning_rate=args.learning_rate, gamma=args.gamma,
            gae_lambda=args.gae_lambda, clip_coef=args.clip_coef,
            entropy_coef=args.entropy_coef, value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            candidate_k=args.candidate_k,
            p_mode=args.p_mode, p_cost_scale=args.p_cost_scale,
            p_points_per_grid_edge=args.p_points_per_grid_edge,
            prediction_mode=args.prediction_mode,
            route_cost_points_per_second=args.route_cost_points_per_second,
            output=args.output)
        assignment = (json.loads(args.task_assignment_json)
            if args.task_assignment_json else {"scenario_id": args.scenario,
                "category": args.category, "seed": config.seed})
        result = train_meco(config, resume=args.resume, warm_start=args.warm_start,
                            qualification_protocol=args.qualification_protocol,
                            stop_file=args.stop_file,
                            abort_file=args.abort_file,
                            invocation_id=args.invocation_id,
                            run_id_override=args.run_id, protocol_id=args.protocol_id,
                            task_assignment=assignment,
                            expected_source_sha256=args.expected_source_sha256,
                            native_extension_attestation=(json.loads(
                                args.native_extension_attestation_json)
                                if args.native_extension_attestation_json else None))
        print(json.dumps(result, indent=2))
        return
    checkpoints = {track: getattr(args, f"{track.lower()}_checkpoint") for track in MECO_TRACKS}
    if (args.replay_output_dir is None) != (args.replay_seeds is None):
        raise SystemExit("--replay-output-dir and explicit --replay-seeds must be supplied together")
    report = evaluate_meco(checkpoints, scenario_id=args.scenario,
        categories=tuple(item for item in args.categories.split(",") if item),
        episodes=args.episodes, master_seed=args.seed, device=args.device,
        split=args.split,
        evaluation_protocol=args.evaluation_protocol,
        evaluation_case=(json.loads(args.evaluation_case_json)
            if args.evaluation_case_json else None),
        schedule_case_binding=(json.loads(args.schedule_case_binding_json)
            if args.schedule_case_binding_json else None),
        diagnostic_frontier_binding=(json.loads(args.diagnostic_frontier_binding_json)
            if args.diagnostic_frontier_binding_json else None),
        episode_seeds=(tuple(int(item) for item in args.episode_seeds.split(",") if item)
            if args.episode_seeds else None),
        simulated_score_estimand=args.simulated_score_estimand,
        output=args.output, prediction_mode=args.prediction_mode,
        control_prediction_mode=args.control_prediction_mode,
        control_prediction_modes=args.control_prediction_modes,
        candidate_k_override=args.candidate_k_override,
        inference_disable_tick=args.inference_disable_tick,
        replay_output_dir=args.replay_output_dir,
        replay_seeds=(None if args.replay_seeds is None else tuple(args.replay_seeds)))
    print(json.dumps({"evaluation_id": report["evaluation_id"],
        "protocol_id": report["protocol_id"], "records": len(report["records"]),
        "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
