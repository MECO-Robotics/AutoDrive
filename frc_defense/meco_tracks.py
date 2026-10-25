"""Five MECO research scorers and a fail-closed decision adapter.

This module is deliberately separate from the production controller. Models
score goals/candidates; the caller supplies deterministic feasibility and
legality, and only the deterministic arbiter may execute an action.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
import queue
import sys
import threading
from contextlib import contextmanager
from types import MappingProxyType
import time
from typing import Mapping
import uuid
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from .learning_contract import CandidateTable, ObservationSnapshot, ScorerSnapshot, masked_argmax
from .learning_inputs import build_observation_snapshot


_MecoAdapterScope = threading.local()


@contextmanager
def managed_meco_adapters():
    """Close every production adapter created in this thread's scope."""
    previous = getattr(_MecoAdapterScope, "adapters", None)
    adapters = []
    _MecoAdapterScope.adapters = adapters
    try:
        yield
    finally:
        if previous is None:
            del _MecoAdapterScope.adapters
        else:
            _MecoAdapterScope.adapters = previous
        active_exception = sys.exc_info()[0] is not None
        close_error = None
        for adapter in reversed(adapters):
            try:
                adapter.close()
            except Exception as exc:
                close_error = close_error or exc
        if close_error is not None and not active_exception:
            raise close_error


MECO_INFERENCE_SCHEMA = "meco-inference-v2"
MECO_MAX_INFERENCE_AGE_S = 0.25
MECO_INFERENCE_OUTCOME_CODES = MappingProxyType({
    "accepted": 0,
    "deterministic_fallback": 1,
    "no_feasible_candidate": 2,
})
MECO_INPUT_NAMES = {
    "A1": ("raster", "globals_", "hidden"),
    "A2": ("node_features", "edges", "edge_mask", "node_valid", "region_ids", "globals_",
           "coarse_edges", "coarse_edge_mask", "hidden"),
    "A3": ("node_features", "edges", "edge_mask", "node_valid", "region_ids", "globals_", "hidden"),
    "C1": ("globals_", "candidate_features", "hidden"),
    "C2": ("raster", "globals_", "candidate_features", "hidden"),
}


@dataclass(frozen=True, slots=True)
class MecoInferenceRequest:
    """Versioned production scorer request tied to real planner tensors."""
    track: str
    snapshot_id: str
    checkpoint_sha256: str
    protocol_id: str
    observation_schema: int
    observation_timestamp: float
    capture_monotonic: float
    dispatch_monotonic: float
    expires_monotonic: float
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    candidate_ids: tuple[str, ...]
    valid_mask: tuple[bool, ...]
    fallback_index: int
    request_sequence: int = 0
    generation_id: str = ""
    candidate_mask_digest: str = ""
    legal_pose_order_digest: str = ""
    input_contract_sha256: str = ""
    model_id: str = ""
    schema: str = MECO_INFERENCE_SCHEMA
    candidate_support_mode: str = "table-mask"

    def __post_init__(self):
        if (self.schema != MECO_INFERENCE_SCHEMA or self.track not in MECO_TRACKS
                or self.observation_schema != 1 or not self.snapshot_id):
            raise ValueError("unsupported MECO track or observation schema")
        if (len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256)
                or not self.protocol_id or type(self.request_sequence) is not int
                or self.request_sequence < 1 or not self.generation_id
                or self.model_id != f"meco:{self.track}"):
            raise ValueError("checkpoint SHA-256 and protocol identity are required")
        if (not math.isfinite(self.observation_timestamp) or self.observation_timestamp < 0
                or not math.isfinite(self.capture_monotonic) or self.capture_monotonic < 0
                or not math.isfinite(self.dispatch_monotonic)
                or self.dispatch_monotonic < self.capture_monotonic
                or not math.isfinite(self.expires_monotonic)
                or self.expires_monotonic != self.capture_monotonic + MECO_MAX_INFERENCE_AGE_S
                or self.dispatch_monotonic > self.expires_monotonic):
            raise ValueError("invalid simulation/capture/dispatch timestamp or expiry")
        if self.input_names != MECO_INPUT_NAMES[self.track] or len(self.input_shapes) != len(self.input_names):
            raise ValueError("track input names/order do not match the versioned contract")
        if (not self.candidate_ids or any(not isinstance(item, str) or not item for item in self.candidate_ids)
                or len(set(self.candidate_ids)) != len(self.candidate_ids)
                or any(not shape or any(type(size) is not int or size <= 0 for size in shape)
                       for shape in self.input_shapes)):
            raise ValueError("candidate ordering, mask, or deterministic fallback is invalid")
        if self.candidate_support_mode == "table-mask":
            if (len(self.candidate_ids) != len(self.valid_mask)
                    or any(type(v) is not bool for v in self.valid_mask)
                    or not 0 <= self.fallback_index < len(self.candidate_ids)
                    or not self.valid_mask[self.fallback_index]):
                raise ValueError("candidate ordering, mask, or deterministic fallback is invalid")
            expected_mask_digest = ordered_candidate_mask_digest(self.candidate_ids, self.valid_mask)
        elif self.candidate_support_mode == "device-mask":
            if self.valid_mask or self.fallback_index != -1:
                raise ValueError("device candidate support cannot contain a host mask or fallback")
            expected_mask_digest = ordered_candidate_order_digest(self.candidate_ids)
        else:
            raise ValueError("unsupported candidate support mode")
        if self.candidate_mask_digest != expected_mask_digest:
            raise ValueError("candidate IDs/mask digest does not match request ordering")
        from .meco_input_provenance import input_contract_digest
        if self.input_contract_sha256 != input_contract_digest():
            raise ValueError("model input provenance digest does not match registry")
        if self.track.startswith("A") and (len(self.legal_pose_order_digest) != 64
                or any(c not in "0123456789abcdef" for c in self.legal_pose_order_digest)):
            raise ValueError("Plan A request requires the canonical legal-pose order digest")
        if self.track.startswith("C") and self.legal_pose_order_digest:
            raise ValueError("Plan C request cannot carry a Plan A legal-pose digest")


@dataclass(frozen=True, slots=True)
class MecoInferenceResponse:
    """Scorer values plus immutable provenance copied from its request."""
    schema: str
    track: str
    checkpoint_sha256: str
    protocol_id: str
    model_id: str
    observation_timestamp: float
    capture_monotonic: float
    dispatch_monotonic: float
    completed_monotonic: float
    expires_monotonic: float
    request_sequence: int
    generation_id: str
    snapshot_id: str
    candidate_mask_digest: str
    legal_pose_order_digest: str
    input_contract_sha256: str
    outputs: Mapping[str, torch.Tensor]


def ordered_candidate_mask_digest(candidate_ids, valid_mask):
    payload = {"candidate_ids": tuple(candidate_ids), "valid_mask": tuple(valid_mask)}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"),
                           ensure_ascii=False).encode()).hexdigest()


def ordered_candidate_order_digest(candidate_ids):
    """Digest stable row identity when the authoritative mask stays on device."""
    payload = {"candidate_ids": tuple(candidate_ids),
               "support_mode": "device-mask"}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"),
                           ensure_ascii=False).encode()).hexdigest()


def make_meco_inference_request(adapter: "ProductionMecoAdapter", *,
                                table: CandidateTable,
                                observation_timestamp: float,
                                capture_monotonic: float,
                                inputs: Mapping[str, torch.Tensor],
                                legal_pose_order_digest: str = "",
                                candidate_batch=None) -> MecoInferenceRequest:
    """Freeze request identity/order/shapes from current planner tensors."""
    names = MECO_INPUT_NAMES[adapter.track]
    if tuple(inputs) != names:
        raise ValueError("inputs must be supplied in the track's canonical order")
    from .meco_input_provenance import validate_serialized_inputs
    contract_digest = validate_serialized_inputs(adapter.track, inputs)
    dispatch = time.monotonic()
    sequence, generation = adapter.next_request_identity()
    device_support = candidate_batch is not None
    if device_support and (adapter.track not in ("C1", "C2") or
            tuple(candidate_batch.candidate_ids) != tuple(table.candidate_ids)):
        raise ValueError("device candidate support requires aligned C1/C2 candidate rows")
    request_mask = () if device_support else table.valid_mask
    request_fallback = -1 if device_support else next(
        (i for i, valid in enumerate(table.valid_mask) if valid), -1)
    if not device_support and request_fallback < 0:
        raise ValueError("request requires a valid deterministic fallback")
    return MecoInferenceRequest(track=adapter.track, snapshot_id=table.snapshot_id,
        checkpoint_sha256=adapter.checkpoint_sha256, protocol_id=adapter.protocol_id,
        observation_schema=1, observation_timestamp=float(observation_timestamp),
        capture_monotonic=float(capture_monotonic), dispatch_monotonic=dispatch,
        expires_monotonic=float(capture_monotonic) + MECO_MAX_INFERENCE_AGE_S,
        input_names=names,
        input_shapes=tuple(tuple(inputs[name].shape) for name in names),
        candidate_ids=table.candidate_ids, valid_mask=request_mask,
        fallback_index=request_fallback,
        request_sequence=sequence, generation_id=generation,
        candidate_mask_digest=(ordered_candidate_order_digest(table.candidate_ids)
            if device_support else ordered_candidate_mask_digest(table.candidate_ids, table.valid_mask)),
        legal_pose_order_digest=legal_pose_order_digest,
        input_contract_sha256=contract_digest, model_id=f"meco:{adapter.track}",
        candidate_support_mode=("device-mask" if device_support else "table-mask"))


class ProductionMecoAdapter:
    """Validate real track tensors and scorer outputs before planner use.

    Invalid, expired, stale, or faulting inference returns ``None``; caller
    must use its already-feasible deterministic fallback.
    """
    def __init__(self, track: str, model: nn.Module, *, checkpoint_sha256: str,
                 protocol_id: str, source_snapshot_sha256: str = ""):
        if track not in MECO_TRACKS:
            raise ValueError(f"unsupported MECO track: {track}")
        if (len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
                or not protocol_id):
            raise ValueError("production adapter requires checkpoint and protocol identity")
        parameters = tuple(model.parameters())
        if not parameters:
            raise RuntimeError("production MECO model must have accelerator-resident parameters")
        devices = {parameter.device for parameter in parameters}
        if (len(devices) != 1 or next(iter(devices)).type != "cuda"
                or not torch.cuda.is_available()
                or next(iter(devices)).index is None
                or next(iter(devices)).index >= torch.cuda.device_count()):
            raise RuntimeError("production MECO adapter rejects CPU or unavailable model devices")
        self.track, self.model = track, model
        self.device = next(iter(devices))
        self.checkpoint_sha256, self.protocol_id = checkpoint_sha256, protocol_id
        self.source_snapshot_sha256 = source_snapshot_sha256
        self._last_inference_exception_type = None
        self._request_lock = threading.Lock()
        self._request_sequence = 0
        self._generation_id = uuid.uuid4().hex
        self._broker = _AsyncInferenceBroker(self)
        scope = getattr(_MecoAdapterScope, "adapters", None)
        if scope is not None:
            scope.append(self)

    def next_request_identity(self):
        with self._request_lock:
            self._request_sequence += 1
            return self._request_sequence, self._generation_id

    @property
    def inference_counters(self):
        return self._broker.counters()

    def submit_inference(self, request: MecoInferenceRequest,
                         inputs: Mapping[str, torch.Tensor]):
        # Diagnostic capture is strictly opt-in and bounded by an explicit
        # request count. Capture failures are visible but never alter scoring.
        if "MECO_REQUEST_CAPTURE_DIR" in os.environ or "MECO_REQUEST_CAPTURE_MAX" in os.environ:
            try:
                from .meco_request_capture import capture_request
                capture_inputs = {name: inputs[name] for name in request.input_names
                                  if name in inputs}
                capture_request(request, capture_inputs,
                    source_snapshot_sha256=self.source_snapshot_sha256)
            except Exception as exc:
                warnings.warn(f"MECO production request capture failed: {type(exc).__name__}: {exc}",
                              RuntimeWarning, stacklevel=2)
        return self._broker.submit(request, inputs)

    def poll_inference(self, request: MecoInferenceRequest, handle):
        return self._broker.poll(request, handle)

    def close(self):
        """Stop the adapter worker and release its model references once idle."""
        self._broker.close()

    def _infer_sync(self, request: MecoInferenceRequest, inputs: Mapping[str, torch.Tensor], *, now: float):
        self._last_inference_exception_type = None
        try:
            from .meco_input_provenance import input_contract_digest
            support_key = "__candidate_valid_mask"
            candidate_mask = inputs.get(support_key)
            model_inputs = {name: inputs[name] for name in request.input_names
                            if name in inputs}
            if (request.track != self.track or request.model_id != f"meco:{self.track}"
                    or request.checkpoint_sha256 != self.checkpoint_sha256
                    or request.protocol_id != self.protocol_id or request.observation_schema != 1
                    or not math.isfinite(now) or now < request.dispatch_monotonic
                    or now - request.capture_monotonic > MECO_MAX_INFERENCE_AGE_S
                    or now > request.expires_monotonic
                    or request.input_contract_sha256 != input_contract_digest()
                    or tuple(model_inputs) != request.input_names
                    or set(inputs) != (set(request.input_names) |
                        ({support_key} if candidate_mask is not None else set()))):
                return None
            if (request.candidate_support_mode == "device-mask" and
                    (self.track not in ("A1", "A2", "A3", "C1", "C2")
                     or candidate_mask is None)):
                return None
            tensors = tuple(model_inputs[name] for name in request.input_names)
            if tuple(tuple(t.shape) for t in tensors) != request.input_shapes:
                return None
            if any(not isinstance(t, torch.Tensor) for t in tensors):
                return None
            if any(t.device != self.device or t.device.type != "cuda" for t in tensors):
                return None
            # Plan C device-mask requests and all A2/A3 graph requests keep
            # dynamic validity predicates on device. A2/A3 status is runtime
            # tensor state even when their legacy CandidateTable API wraps it.
            device_validity_mode = (request.candidate_support_mode == "device-mask"
                                    or self.track in ("A2", "A3"))
            inference_valid = None
            if device_validity_mode:
                # Plan C device-mask requests cannot synchronize dynamic CUDA
                # predicates. Carry validity to its selector and sanitize.
                inference_valid = torch.isfinite(tensors[0]).all()
                safe_tensors = []
                for tensor in tensors:
                    if tensor.is_floating_point():
                        inference_valid &= torch.isfinite(tensor).all()
                        safe_tensors.append(torch.where(torch.isfinite(tensor), tensor,
                                                        torch.zeros_like(tensor)))
                    else:
                        safe_tensors.append(tensor)
            else:
                # Preserve the established A1/A2/A3 adapter contract from
                # mainline; Plan C's no-sync change does not modify Plan A.
                if any(tensor.is_floating_point() and
                       not bool(torch.isfinite(tensor).all()) for tensor in tensors):
                    return None
                safe_tensors = list(tensors)
            for name, tensor in zip(request.input_names, tensors):
                if name in ("edge_mask", "coarse_edge_mask", "node_valid") and tensor.dtype != torch.bool:
                    return None
                if name in ("edges", "coarse_edges", "region_ids") and tensor.dtype != torch.long:
                    return None
            by_name = dict(zip(request.input_names, safe_tensors))
            if candidate_mask is not None:
                if (request.candidate_support_mode != "device-mask"
                        or not isinstance(candidate_mask, torch.Tensor)
                        or candidate_mask.dtype != torch.bool
                        or candidate_mask.device != self.device
                        or candidate_mask.shape != (1, len(request.candidate_ids))):
                    return None
                by_name["candidate_mask"] = candidate_mask
            batch = tensors[0].shape[0]
            if by_name["globals_"].shape != (batch, 7):
                return None
            if "raster" in by_name and (by_name["raster"].ndim != 4 or by_name["raster"].shape[1] != 6):
                return None
            if by_name.get("hidden") is not None and by_name["hidden"].shape != (batch, HIDDEN):
                return None
            if "candidate_features" in by_name and (
                    by_name["candidate_features"].ndim != 3
                    or by_name["candidate_features"].shape[0] != batch
                    or by_name["candidate_features"].shape[1] != len(request.candidate_ids)
                    or by_name["candidate_features"].shape[2] != MECO_CANDIDATE_MODEL_INPUT_WIDTH):
                return None
            if "node_features" in by_name and (
                    by_name["node_features"].ndim != 3 or by_name["node_features"].shape[0] != batch
                    or by_name["node_features"].shape[2] != 9):
                return None
            if "edges" in by_name and (by_name["edges"].ndim != 3
                    or by_name["edge_mask"].shape != by_name["edges"].shape
                    or by_name["edges"].shape[:2] != by_name["node_features"].shape[:2]):
                return None
            if "region_ids" in by_name and by_name["region_ids"].shape != by_name["node_features"].shape[:2]:
                return None
            if "node_valid" in by_name and (
                    by_name["node_valid"].shape != by_name["node_features"].shape[:2]
                    or by_name["node_valid"].dtype != torch.bool):
                return None
            if "coarse_edges" in by_name and (by_name["coarse_edges"].ndim != 3
                    or by_name["coarse_edge_mask"].shape != by_name["coarse_edges"].shape):
                return None
            outputs = self.model(**by_name)
            expected = (("utility", "penalty", "behavior_logits", "hidden") if self.track.startswith("A")
                        else ("candidate_utility", "uncertainty", "hidden"))
            if not isinstance(outputs, Mapping) or any(name not in outputs for name in expected):
                return None
            if self.track in ("A2", "A3"):
                model_input_valid = outputs.get("device_input_valid")
                if (not isinstance(model_input_valid, torch.Tensor)
                        or model_input_valid.dtype != torch.bool
                        or model_input_valid.numel() != 1
                        or model_input_valid.device != self.device):
                    return None
                if device_validity_mode:
                    inference_valid &= model_input_valid.reshape(())
            values = {name: outputs[name] for name in expected}
            if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
                   or value.device != self.device or value.device.type != "cuda"
                   or (not device_validity_mode and not bool(torch.isfinite(value).all()))
                   for value in values.values()):
                return None
            if device_validity_mode:
                for value in values.values():
                    inference_valid &= torch.isfinite(value).all()
            if values["hidden"].shape != (batch, HIDDEN):
                return None
            if self.track.startswith("A"):
                utility_shape_ok = (
                    values["utility"].ndim == 4 and values["utility"].shape[:2] == (batch, 8)
                    and values["utility"].shape[-2:] == by_name["raster"].shape[-2:]
                    if self.track == "A1" else
                    values["utility"].ndim == 2 and values["utility"].shape == by_name["node_features"].shape[:2])
                if (not utility_shape_ok or values["penalty"].shape != values["utility"].shape
                        or values["behavior_logits"].shape != (batch, 5)):
                    return None
                if device_validity_mode:
                    inference_valid &= (values["penalty"] >= 0).all()
                elif bool((values["penalty"] < 0).any()):
                    return None
            elif (values["candidate_utility"].shape != (batch, len(request.candidate_ids))
                  or values["uncertainty"].shape != values["candidate_utility"].shape):
                return None
            else:
                inference_valid &= (values["uncertainty"] >= 0).all()
            if device_validity_mode:
                values = {name: torch.where(inference_valid, value,
                                            torch.zeros_like(value))
                          for name, value in values.items()}
                values["__inference_valid"] = inference_valid
            done = time.monotonic()
            if (done - request.capture_monotonic > MECO_MAX_INFERENCE_AGE_S
                    or done > request.expires_monotonic):
                return None
            return MecoInferenceResponse(schema=MECO_INFERENCE_SCHEMA,
                track=self.track, checkpoint_sha256=self.checkpoint_sha256,
                protocol_id=self.protocol_id, model_id=request.model_id,
                observation_timestamp=request.observation_timestamp,
                capture_monotonic=request.capture_monotonic,
                dispatch_monotonic=request.dispatch_monotonic,
                completed_monotonic=done, expires_monotonic=request.expires_monotonic,
                request_sequence=request.request_sequence,
                generation_id=request.generation_id, snapshot_id=request.snapshot_id,
                candidate_mask_digest=request.candidate_mask_digest,
                legal_pose_order_digest=request.legal_pose_order_digest,
                input_contract_sha256=request.input_contract_sha256,
                outputs=MappingProxyType(dict(values)))
        except Exception:
            self._last_inference_exception_type = "model_exception"
            return None


class _AsyncInferenceBroker:
    """One-worker, single-pending-request broker with replace-oldest semantics."""
    _STOP = object()

    def __init__(self, adapter):
        self.adapter = adapter
        self.pending = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest_published_sequence = 0
        self.latest_submitted_sequence = 0
        self.closed = False
        self.dropped = 0
        self.late = 0
        self.faults = 0
        self.recent_outcomes = deque(maxlen=128)
        self.active_item = None
        self.thread = threading.Thread(target=self._run,
            name=f"meco-inference-{adapter.track}", daemon=True)
        self.thread.start()

    def counters(self):
        with self.lock:
            return {"dropped_requests": self.dropped,
                    "late_responses": self.late,
                    "inference_faults": self.faults,
                    "generation_id": self.adapter._generation_id,
                    "last_published_sequence": self.latest_published_sequence,
                    "outcome_schema": "meco-inference-outcomes-v1",
                    "recent_outcomes": [copy.deepcopy(row) for row in self.recent_outcomes],
                    "closed": self.closed}

    @staticmethod
    def _identity(request):
        return {"track": getattr(request, "track", None),
                "request_sequence": getattr(request, "request_sequence", None),
                "generation_id": getattr(request, "generation_id", None),
                "snapshot_id": getattr(request, "snapshot_id", None)}

    def _finish_locked(self, item, status, *, now=None):
        if item.get("terminal"):
            if item.get("status") == "published" and status in (
                    "consumed", "superseded", "identity_mismatch", "expired_completion"):
                terminal_time = time.monotonic() if now is None else now
                item.setdefault("times", {})[status] = terminal_time
                if status == "consumed":
                    item.setdefault("events", []).append("consumed")
                item["status"] = status
                outcome = item.get("outcome")
                if outcome is not None:
                    outcome["status"] = status
                    outcome["times_monotonic"] = dict(item["times"])
                    outcome["events"] = list(item["events"])
                    outcome["durations_s"]["request_age_at_terminal"] = max(0.,
                        terminal_time - getattr(item["request"], "capture_monotonic", terminal_time))
            return
        item["terminal"] = True
        item["status"] = status
        item.setdefault("times", {})[status] = time.monotonic() if now is None else now
        times = dict(item.get("times", {}))
        queued = times.get("queued")
        item_identity = self._identity(item["request"])
        outcome = {"schema": "meco-inference-outcome-v1",
            **item_identity, "status": status,
            "times_monotonic": times,
            "durations_s": ({"queue_wait": max(0., times["started"] - queued)
                if queued is not None and times.get("started") is not None else None,
                "inference": max(0., times["completed"] - times["started"])
                if times.get("completed") is not None and times.get("started") is not None else None,
                "request_age_at_terminal": max(0., times[status] - getattr(
                    item["request"], "capture_monotonic", times[status]))}),
            "events": list(item.get("events", ()))}
        item["outcome"] = outcome
        self.recent_outcomes.append(outcome)

    def _record_completion_locked(self, item, completed_at):
        item.setdefault("times", {})["completed"] = completed_at
        if "completed" not in item.setdefault("events", []):
            item["events"].append("completed")
        outcome = item.get("outcome")
        if outcome is not None:
            outcome["times_monotonic"] = dict(item["times"])
            outcome["events"] = list(item["events"])
            started = item["times"].get("started")
            outcome["durations_s"]["inference"] = (max(0., completed_at - started)
                if started is not None else None)

    def _reject_locked(self, request, status):
        item = {"request": request, "times": {}, "events": [], "terminal": False}
        self._finish_locked(item, status)

    def submit(self, request, inputs):
        if (request.generation_id != self.adapter._generation_id
                or request.request_sequence <= 0
                or request.request_sequence > self.adapter._request_sequence
                or request.checkpoint_sha256 != self.adapter.checkpoint_sha256
                or request.protocol_id != self.adapter.protocol_id
                or request.model_id != f"meco:{self.adapter.track}"):
            with self.lock:
                self.faults += 1
                self._reject_locked(request, "identity_mismatch")
            return None
        try:
            frozen_inputs = {name: tensor.detach().clone()
                             for name, tensor in inputs.items()}
        except Exception:
            with self.lock:
                self.faults += 1
                self._reject_locked(request, "model_exception")
            return None
        item = {"request": request, "inputs": frozen_inputs,
                "response": None, "done": False, "consumed": False,
                "discarded": False, "late_counted": False, "terminal": False,
                "times": {"queued": time.monotonic()}, "events": ["queued"]}
        with self.lock:
            if self.closed:
                self.faults += 1
                self._finish_locked(item, "close")
                return None
            if request.request_sequence <= self.latest_submitted_sequence:
                self.faults += 1
                self._finish_locked(item, "identity_mismatch")
                return None
            # Sequence acceptance and queue replacement are one serialized
            # operation, so an older submitter cannot replace a newer request.
            try:
                self.pending.put_nowait(item)
            except queue.Full:
                try:
                    replaced_item = self.pending.get_nowait()
                    self.pending.task_done()
                except queue.Empty:
                    replaced_item = None
                if replaced_item is not None:
                    replaced_item["discarded"] = True
                    self.dropped += 1
                    self._finish_locked(replaced_item, "superseded")
                try:
                    self.pending.put_nowait(item)
                except queue.Full:
                    item["discarded"] = True
                    self.dropped += 1
                    self._finish_locked(item, "superseded")
                    return None
            self.latest_submitted_sequence = request.request_sequence
        return item

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                while True:
                    try:
                        item = self.pending.get_nowait()
                    except queue.Empty:
                        break
                    if item is not self._STOP:
                        item["discarded"] = True
                        item["done"] = True
                        self.dropped += 1
                        self._finish_locked(item, "close")
                    self.pending.task_done()
                self.pending.put_nowait(self._STOP)
        if threading.current_thread() is not self.thread:
            self.thread.join()

    def poll(self, request, item):
        if item is None:
            return None
        with self.lock:
            if item.get("request") != request:
                self._reject_locked(request, "identity_mismatch")
                return None
            if item["consumed"] or item["discarded"]:
                return None
            if request.generation_id != self.adapter._generation_id:
                item["discarded"] = True
                item["consumed"] = True
                self._mark_late_locked(item)
                self._finish_locked(item, "identity_mismatch")
                return None
            if request.request_sequence != self.latest_submitted_sequence:
                item["discarded"] = True
                item["consumed"] = True
                self._mark_late_locked(item)
                self._finish_locked(item, "superseded" if
                    request.request_sequence < self.latest_submitted_sequence else "identity_mismatch")
                return None
            if not item["done"]:
                # This poll is the action boundary. Closing the handle prevents
                # a later worker completion from being published or reused.
                item["consumed"] = True
                item["times"]["consumed"] = time.monotonic()
                item["events"].append("consumed")
                self._finish_locked(item, "consumed_before_completion")
                if time.monotonic() >= request.expires_monotonic:
                    self._mark_late_locked(item)
                return None
            response = item["response"]
            if response is None:
                item["consumed"] = True
                self._finish_locked(item, item.get("status") or "model_none")
                return None
            if (time.monotonic() > request.expires_monotonic
                    or response.completed_monotonic > request.expires_monotonic):
                self._mark_late_locked(item)
                self._finish_locked(item, "expired_completion")
                return None
            item["consumed"] = True
            self._finish_locked(item, "consumed")
            return response

    def _mark_late(self, item):
        with self.lock:
            self._mark_late_locked(item)

    def _mark_late_locked(self, item):
        if not item["late_counted"]:
            self.late += 1
            item["late_counted"] = True

    def _run(self):
        while True:
            item = self.pending.get()
            if item is self._STOP:
                self.pending.task_done()
                return
            request = item["request"]
            try:
                if (request.generation_id != self.adapter._generation_id
                        or time.monotonic() > request.expires_monotonic):
                    self._mark_late(item)
                    with self.lock:
                        self._finish_locked(item, "expiry_before_start" if time.monotonic() > request.expires_monotonic else "identity_mismatch")
                    continue
                with self.lock:
                    if self.closed:
                        self._finish_locked(item, "close")
                        continue
                    self.active_item = item
                    item["times"]["started"] = time.monotonic()
                    item["events"].append("started")
                with torch.inference_mode():
                    if hasattr(self.adapter, "_last_inference_exception_type"):
                        self.adapter._last_inference_exception_type = None
                    response = self.adapter._infer_sync(request, item["inputs"],
                                                        now=time.monotonic())
                completed_at = (getattr(response, "completed_monotonic", None)
                    if response is not None else None) or time.monotonic()
                late_response = False
                with self.lock:
                    self._record_completion_locked(item, completed_at)
                    if (self.closed or response is None or time.monotonic() > request.expires_monotonic
                            or request.request_sequence != self.latest_submitted_sequence
                            or request.request_sequence <= self.latest_published_sequence
                            or item["consumed"]):
                        if self.closed:
                            self._finish_locked(item, "close")
                        elif response is None:
                            self.faults += 1
                            status = ("expired_completion" if time.monotonic() > request.expires_monotonic
                                else getattr(self.adapter, "_last_inference_exception_type", None)
                                or "model_none")
                            self._finish_locked(item, status)
                        else:
                            late_response = True
                            reason = ("expired_completion" if time.monotonic() > request.expires_monotonic
                                or response.completed_monotonic > request.expires_monotonic else
                                "consumed_before_completion" if item["consumed"] else "superseded")
                            self._finish_locked(item, reason)
                    else:
                        # Publish only after the complete output mapping is validated.
                        item["response"] = response
                        self.latest_published_sequence = request.request_sequence
                        item["times"]["published"] = time.monotonic()
                        item["events"].append("published")
                        self._finish_locked(item, "published")
                    # Make completion visible in the same critical section as
                    # any complete response publication.
                    item["done"] = True
                if late_response:
                    self._mark_late(item)
            except Exception:
                # Adapter faults are isolated to this request; keep the broker
                # worker alive for subsequent snapshots.
                with self.lock:
                    self._record_completion_locked(item, time.monotonic())
                    self.faults += 1
                    item["response"] = None
                    self._finish_locked(item, "close" if self.closed else "model_exception")
            finally:
                with self.lock:
                    item["done"] = True
                    if self.active_item is item:
                        self.active_item = None
                self.pending.task_done()


MECO_TRACKS = ("A1", "A2", "A3", "C1", "C2")
MECO_SCHEMA_VERSION = 1
HIDDEN = 64
PREDICTION_MODES = ("current", "cv", "multi_model_v1")
PREDICTION_MODEL_ID = "meco_opponent_forecast_v1"
PREDICTION_MAX_SPEED_MPS = 3.0
PREDICTION_MAX_AGE_S = .5
PREDICTION_DEFAULT_HORIZON_S = 1.0
PREDICTION_THREAT_RADIUS_M = 2.0
PREDICTION_MAX_PENALTY_POINTS = .75
SAFE_ACTION_FEATURES = (
    "action_index", "sensor_track_buffer_index", "goal_x_m", "goal_y_m",
    "heading_bin", "behavior_index", "field_length_m", "field_width_m",
    "target_distance_m", "nearby_sensed_track_count", "track_confidence",
    "track_age_s", "opponent_distance_m", "route_time_s",
)
EXTRA_FEATURES = ("legal_node_index", "route_distance_m",
                  "risk_penalty_points", "target_heading_rad",
                  "is_continue", "commitment_switch_cost")
MECO_CANDIDATE_RAW_FEATURE_COUNT = len(SAFE_ACTION_FEATURES) + len(EXTRA_FEATURES)
MECO_CANDIDATE_MODEL_INPUT_WIDTH = 2 * MECO_CANDIDATE_RAW_FEATURE_COUNT

# Shared opponent forecast contract used by learned policies and controls.
PREDICTION_MODES = ("current", "cv", "multi_model_v1")
PREDICTION_MODEL_ID = "meco_opponent_forecast_v1"
PREDICTION_MAX_SPEED_MPS = 3.0
PREDICTION_MAX_AGE_S = 0.5
PREDICTION_DEFAULT_HORIZON_S = 1.0
PREDICTION_THREAT_RADIUS_M = 2.0
PREDICTION_MAX_PENALTY_POINTS = 0.75
MULTIMODEL_ID = "meco_opponent_kinematics_mm_v1"
MULTIMODEL_MAX_HORIZON_S = 2.5
MULTIMODEL_TURN_RATE_MIN_RAD_S = .6
MULTIMODEL_TURN_RATE_MAX_RAD_S = 1.5
MULTIMODEL_DAMPING_PER_S = .8
MULTIMODEL_HYPOTHESES = (
    ("constant_velocity", .40),
    ("turn_left", .15),
    ("turn_right", .15),
    ("damped_velocity", .30),
)


def prediction_contract(mode: str, *, horizon_s: float | None = None) -> dict:
    """Describe forecast semantics in immutable training/evaluation provenance."""
    if mode not in PREDICTION_MODES:
        raise ValueError(f"prediction mode must be one of {PREDICTION_MODES}")
    horizon = (0.0 if mode == "current" else
               None if horizon_s is None else float(horizon_s))
    if horizon is not None and (not math.isfinite(horizon) or horizon < 0.):
        raise ValueError("prediction horizon must be finite and nonnegative")
    contract = {"model": MULTIMODEL_ID if mode == "multi_model_v1" else PREDICTION_MODEL_ID,
        "mode": mode,
        "horizon_s": horizon,
        "horizon_semantics": ("observed_current_pose" if mode == "current" else
            "caller_supplied_eta_or_fixed_horizon" if horizon is None else "fixed_or_caller_supplied"),
        "max_speed_mps": PREDICTION_MAX_SPEED_MPS if mode == "cv" else None,
        "max_track_age_s": PREDICTION_MAX_AGE_S if mode == "cv" else None,
        "threat_radius_m": PREDICTION_THREAT_RADIUS_M,
        "max_penalty_points": PREDICTION_MAX_PENALTY_POINTS,
        "fallback": ("none_if_observed_pose_missing_or_nonfinite" if mode == "current"
            else "current_finite_observed_pose_when_cv_unavailable; otherwise_none")}
    if mode == "multi_model_v1":
        contract.update({
            "horizon_semantics": "candidate_route_eta_clamped_to_max_horizon",
            "max_horizon_s": MULTIMODEL_MAX_HORIZON_S,
            "max_speed_mps": PREDICTION_MAX_SPEED_MPS,
            "max_track_age_s": PREDICTION_MAX_AGE_S,
            "hypotheses": [{"id": name, "weight": weight}
                           for name, weight in MULTIMODEL_HYPOTHESES],
            "turn_rate_min_rad_s": MULTIMODEL_TURN_RATE_MIN_RAD_S,
            "turn_rate_max_rad_s": MULTIMODEL_TURN_RATE_MAX_RAD_S,
            "damping_per_s": MULTIMODEL_DAMPING_PER_S,
            "fallback": "finite_observed_pose_if_track_missing_stale_or_nonfinite",
            "risk_semantics": "weighted_expected_penalty_over_hypothesis_endpoints",
        })
    return contract


def observation_digest(snapshot: ObservationSnapshot) -> str:
    payload = {"snapshot_id": snapshot.snapshot_id, "episode_id": snapshot.episode_id,
               "sequence": snapshot.sequence, "timestamp": snapshot.timestamp,
               "raster_shape": snapshot.raster_shape, "raster": snapshot.raster,
               "globals": snapshot.globals, "schema_version": snapshot.schema_version}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def candidate_digest(table: CandidateTable) -> str:
    payload = {"snapshot_id": table.snapshot_id, "candidate_ids": table.candidate_ids,
               "features": table.features, "valid_mask": table.valid_mask,
               "feature_names": table.feature_names, "route_costs": table.route_costs,
               "schema_version": table.schema_version,
               "context_digest": table.context_digest}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def new_meco_episode_id(label="episode") -> str:
    """Return an opaque, one-run generation ID suitable for snapshot binding."""
    return f"{uuid.uuid4().hex}:{label}"


def _groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class RecurrentTrack(nn.Module):
    """Per-episode GRU context. Caller resets hidden state at episode/loss boundaries."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(HIDDEN, HIDDEN)

    def forward(self, features, hidden=None):
        if hidden is None:
            hidden = features.new_zeros((features.shape[0], HIDDEN))
        elif hidden.shape != (features.shape[0], HIDDEN) or not bool(torch.isfinite(hidden).all()):
            raise ValueError("GRU hidden state must be finite with shape [batch, 64]")
        return self.gru(features, hidden)


def _fixed_order_candidate_linear(features, layer):
    """Evaluate a small candidate projection with a fixed reduction order.

    ROCm may select a split-K GEMM for these candidate-by-hidden projections;
    its accumulation order can vary between identical calls. The elementwise
    product plus fixed-axis reduction is stable on the accelerator and keeps
    the same linear parameters and gradients without changing Torch's global
    deterministic-algorithm mode.
    """
    projected = (features.unsqueeze(-2) * layer.weight).sum(dim=-1)
    return projected if layer.bias is None else projected + layer.bias


class A1MultiscaleUNetGRU(nn.Module):
    """A1: multiscale raster U-Net with dense U/P and five behavior values."""
    track_id = "A1"

    def __init__(self):
        super().__init__()
        self.enc0 = ConvBlock(6, 24)
        self.enc1 = ConvBlock(24, 32)
        self.enc2 = ConvBlock(32, HIDDEN)
        self.pool = nn.AvgPool2d(2, ceil_mode=True)
        self.global_mlp = nn.Sequential(nn.Linear(7, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
        self.context = RecurrentTrack()
        self.fuse = nn.Conv2d(HIDDEN * 2, HIDDEN, 1)
        self.dec1 = ConvBlock(HIDDEN + 32, 32)
        self.dec0 = ConvBlock(32 + 24, 24)
        self.pose_head = nn.Conv2d(24, 16, 1)  # 8 headings × [utility, penalty]
        self.behavior_head = nn.Linear(HIDDEN, 5)

    def forward(self, raster, globals_, hidden=None, **_):
        x0 = self.enc0(raster)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        pooled = x2.mean((-2, -1)) + self.global_mlp(globals_)
        state = self.context(pooled, hidden)
        x = self.fuse(torch.cat((x2, state[:, :, None, None].expand_as(x2)), 1))
        x = self.dec1(torch.cat((F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1), 1))
        x = self.dec0(torch.cat((F.interpolate(x, size=x0.shape[-2:], mode="bilinear", align_corners=False), x0), 1))
        pose = self.pose_head(x).reshape(x.shape[0], 8, 2, *x.shape[-2:])
        return {"utility": pose[:, :, 0], "penalty": F.softplus(pose[:, :, 1]),
                "behavior_logits": self.behavior_head(state), "hidden": state}


class SE2MessageBlock(nn.Module):
    """Masked directed transition aggregation over an SE(2) node lattice."""
    def __init__(self, width=HIDDEN):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.update = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, nodes, edges, edge_mask, *, node_valid=None, return_status=False):
        # edges [B,N,D] contains neighbor node indices; edge_mask marks legal transitions.
        if (nodes.ndim != 3 or edges.ndim != 3 or edge_mask.shape != edges.shape
                or edges.shape[:2] != nodes.shape[:2] or edges.dtype != torch.long
                or edge_mask.dtype != torch.bool):
            raise ValueError("invalid SE(2) graph tensor shapes or dtypes")
        if node_valid is None:
            node_valid = torch.ones(nodes.shape[:2], device=nodes.device, dtype=torch.bool)
        if node_valid.shape != nodes.shape[:2] or node_valid.dtype != torch.bool:
            raise ValueError("SE(2) node validity must match the node batch")
        edge_valid = (edges >= 0) & (edges < nodes.shape[1])
        safe_index = edges.clamp(0, nodes.shape[1] - 1)
        destination_valid = node_valid.gather(1, safe_index.reshape(nodes.shape[0], -1)).reshape_as(edges)
        source_valid = node_valid[:, :, None]
        status = ~(edge_mask & (~edge_valid | ~destination_valid | ~source_valid)).any()
        safe_mask = edge_mask & edge_valid & destination_valid & source_valid
        safe = torch.where(safe_mask, edges, torch.zeros_like(edges))
        expanded = nodes[:, :, None, :].expand(-1, -1, safe.shape[-1], -1)
        neighbors = expanded.gather(1, safe[..., None].expand(-1, -1, -1, nodes.shape[-1]))
        messages = self.message(neighbors)
        mask = safe_mask[..., None].to(messages.dtype)
        aggregate = (messages * mask).sum(2) / mask.sum(2).clamp_min(1.)
        result = self.norm(nodes + self.update(torch.cat((nodes, aggregate), -1)))
        result = torch.where(node_valid[..., None], result, torch.zeros_like(result))
        return (result, status) if return_status else result


def _se2_node_inputs(node_features, globals_, node_valid, *, return_status=False):
    """Encode headings as sin/cos so bins 7 and 0 obey angular wraparound."""
    if (node_features.ndim != 3 or node_features.shape[-1] != 9
            or globals_.shape != (node_features.shape[0], 7)):
        raise ValueError("SE(2) nodes require six raster values, x/y, heading-bin and seven globals")
    if node_valid.shape != node_features.shape[:2] or node_valid.dtype != torch.bool:
        raise ValueError("SE(2) node validity must match node features")
    checked_nodes = torch.where(node_valid[..., None], node_features,
                                torch.zeros_like(node_features))
    finite_nodes = torch.isfinite(checked_nodes).all()
    finite_globals = torch.isfinite(globals_).all()
    clean_nodes = torch.nan_to_num(checked_nodes)
    clean_globals = torch.nan_to_num(globals_)
    coords = clean_nodes[..., 6:8]
    heading = clean_nodes[..., 8]
    prefix_valid = ~(node_valid[..., 1:] & ~node_valid[..., :-1]).any()
    status = (finite_nodes & finite_globals
        & prefix_valid
        & ((coords >= 0.) & (coords <= 1.)).all()
        & ((heading >= 0.) & (heading <= 7.)
           & (heading == heading.round())).all())
    coords = coords.clamp(0., 1.)
    heading = heading.round().clamp(0., 7.)
    angle = heading * (2. * math.pi / 8.)
    heading_vector = torch.stack((angle.sin(), angle.cos()), -1)
    embedded = torch.cat((clean_nodes[..., :6], coords, heading_vector,
                      clean_globals[:, None, :].expand(-1, node_features.shape[1], -1)), -1)
    return (embedded, status) if return_status else embedded


def _pool_regions(nodes, region_ids, region_tokens, node_valid):
    batch, count, width = nodes.shape
    if region_ids.shape != (batch, count) or region_ids.dtype != torch.long:
        raise ValueError("region IDs must be integer indices within the configured regions")
    if node_valid.shape != (batch, count) or node_valid.dtype != torch.bool:
        raise ValueError("region pooling node validity must match node rows")
    id_valid = (region_ids >= 0) & (region_ids < region_tokens)
    ids_safe = torch.where(id_valid, region_ids, torch.zeros_like(region_ids))
    pooled = nodes.new_zeros((batch, region_tokens, width))
    counts = nodes.new_zeros((batch, region_tokens, 1))
    contribution_valid = node_valid & id_valid
    pooled.scatter_add_(1, ids_safe[..., None].expand_as(nodes),
                        nodes * contribution_valid[..., None].to(nodes.dtype))
    counts.scatter_add_(1, ids_safe[..., None],
        contribution_valid[..., None].to(nodes.dtype))
    occupied = counts.squeeze(-1) > 0
    has_region = occupied.any(-1)
    status = (~(node_valid & ~id_valid).any()) & node_valid.any(-1).all() & has_region.all()
    occupied = occupied.clone()
    occupied[:, 0] |= ~has_region
    pooled = pooled / counts.clamp_min(1.)
    return pooled, occupied, status


class A2StructuredSE2UNetGRU(nn.Module):
    """A2: SE(2) graph U-Net with wrapped headings, regional attention, and GRU."""
    track_id = "A2"

    def __init__(self, region_tokens=64, heads=4):
        super().__init__()
        if not 32 <= region_tokens <= 128 or heads < 1 or HIDDEN % heads:
            raise ValueError("A2 requires 32–128 regions and a dividing attention head count")
        self.region_tokens = region_tokens
        self.lift = nn.Sequential(nn.Linear(17, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
        self.fine_layers = nn.ModuleList([SE2MessageBlock(), SE2MessageBlock()])
        self.coarse_layer = SE2MessageBlock()
        self.attention = nn.MultiheadAttention(HIDDEN, heads, batch_first=True)
        self.context = RecurrentTrack()
        self.decoder = nn.Sequential(nn.Linear(HIDDEN * 3, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 2))
        self.behavior_head = nn.Linear(HIDDEN, 5)

    def forward(self, node_features, edges, edge_mask, region_ids, globals_,
                coarse_edges, coarse_edge_mask, node_valid, hidden=None, **_):
        node_inputs, status = _se2_node_inputs(node_features, globals_, node_valid,
                                               return_status=True)
        nodes = self.lift(node_inputs)
        nodes = torch.where(node_valid[..., None], nodes, torch.zeros_like(nodes))
        for layer in self.fine_layers:
            nodes, layer_status = layer(nodes, edges, edge_mask,
                node_valid=node_valid, return_status=True)
            status = status & layer_status
        batch, count, _ = nodes.shape
        coarse, occupied, pool_status = _pool_regions(nodes, region_ids,
            self.region_tokens, node_valid=node_valid)
        status = status & pool_status
        coarse, coarse_status = self.coarse_layer(coarse, coarse_edges,
            coarse_edge_mask, node_valid=occupied, return_status=True)
        status = status & coarse_status
        attended, _ = self.attention(coarse, coarse, coarse,
                                     key_padding_mask=~occupied, need_weights=False)
        state = self.context((attended * occupied[..., None]).sum(1) /
                             occupied.sum(1, keepdim=True).clamp_min(1.), hidden)
        safe_region_ids = torch.where((region_ids >= 0) & (region_ids < self.region_tokens),
                                      region_ids, torch.zeros_like(region_ids))
        upsampled = coarse.gather(1, safe_region_ids[..., None].expand(-1, -1, HIDDEN))
        decoded = self.decoder(torch.cat((nodes, upsampled,
                                          state[:, None].expand(-1, count, -1)), -1))
        return {"utility": decoded[..., 0], "penalty": F.softplus(decoded[..., 1]),
                "behavior_logits": self.behavior_head(state), "hidden": state,
                "region_tokens": coarse, "occupied_regions": occupied,
                "device_input_valid": status}


class A3HierarchicalSE2GNNGRU(nn.Module):
    """A3: fine SE(2) graph propagation with connectivity-aware region pooling."""
    track_id = "A3"

    def __init__(self, region_tokens=64, heads=4):
        super().__init__()
        if region_tokens != 64 or heads < 1 or HIDDEN % heads:
            raise ValueError("A3 prototype fixes 64 region tokens")
        self.region_tokens = region_tokens
        self.lift = nn.Sequential(nn.Linear(17, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
        self.layers = nn.ModuleList([SE2MessageBlock() for _ in range(3)])
        self.region_attention = nn.MultiheadAttention(HIDDEN, heads, batch_first=True)
        self.context = RecurrentTrack()
        self.region_decoder = nn.Sequential(nn.Linear(HIDDEN * 2, HIDDEN), nn.SiLU())
        self.node_head = nn.Linear(HIDDEN, 2)
        self.behavior_head = nn.Linear(HIDDEN, 5)

    def forward(self, node_features, edges, edge_mask, region_ids, globals_,
                node_valid, hidden=None, **_):
        node_inputs, status = _se2_node_inputs(node_features, globals_, node_valid,
                                               return_status=True)
        nodes = self.lift(node_inputs)
        nodes = torch.where(node_valid[..., None], nodes, torch.zeros_like(nodes))
        for layer in self.layers:
            nodes, layer_status = layer(nodes, edges, edge_mask,
                node_valid=node_valid, return_status=True)
            status = status & layer_status
        batch, count, _ = nodes.shape
        pooled, occupied, pool_status = _pool_regions(nodes, region_ids,
            self.region_tokens, node_valid=node_valid)
        status = status & pool_status
        regions, _ = self.region_attention(pooled, pooled, pooled,
                                           key_padding_mask=~occupied, need_weights=False)
        state = self.context((regions * occupied[..., None]).sum(1) /
                             occupied.sum(1, keepdim=True).clamp_min(1.), hidden)
        contextual = self.region_decoder(torch.cat((regions, state[:, None].expand(-1, self.region_tokens, -1)), -1))
        safe_region_ids = torch.where((region_ids >= 0) & (region_ids < self.region_tokens),
                                      region_ids, torch.zeros_like(region_ids))
        node_context = contextual.gather(1, safe_region_ids[..., None].expand(-1, -1, HIDDEN))
        outputs = self.node_head(nodes + node_context)
        return {"utility": outputs[..., 0], "penalty": F.softplus(outputs[..., 1]),
                "behavior_logits": self.behavior_head(state), "hidden": state,
                "region_tokens": regions, "occupied_regions": occupied,
                "device_input_valid": status}


class CandidateMLPGRU(nn.Module):
    """C1: global/candidate encoders, recurrent state, shared utility head."""
    track_id = "C1"

    def __init__(self, candidate_features: int, width=HIDDEN):
        super().__init__()
        if candidate_features < 1:
            raise ValueError("candidate_features must be positive")
        self.global_encoder = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width))
        self.candidate_encoder = nn.Sequential(nn.Linear(candidate_features, width), nn.SiLU(), nn.Linear(width, width))
        self.context = RecurrentTrack()
        self.scorer = nn.Sequential(nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, 1))
        self.uncertainty = nn.Linear(width * 3, 1)

    def forward(self, globals_, candidate_features, hidden=None, **_):
        state = self.context(self.global_encoder(globals_), hidden)
        candidates = self.candidate_encoder(candidate_features)
        fused = torch.cat((candidates, state[:, None].expand(-1, candidates.shape[1], -1),
                           self.global_encoder(globals_)[:, None].expand(-1, candidates.shape[1], -1)), -1)
        scored = F.silu(_fixed_order_candidate_linear(fused, self.scorer[0]))
        utility = _fixed_order_candidate_linear(scored, self.scorer[2]).squeeze(-1)
        return {"candidate_utility": utility,
                "uncertainty": F.softplus(self.uncertainty(fused).squeeze(-1)), "hidden": state}


class C2MapCrossAttentionGRU(nn.Module):
    """C2: compact raster tokens query-attended by each candidate."""
    track_id = "C2"

    def __init__(self, candidate_features: int, tokens=64, heads=4):
        super().__init__()
        if candidate_features < 1 or not 32 <= tokens <= 128 or heads < 1 or HIDDEN % heads:
            raise ValueError("invalid C2 dimensions")
        self.tokens = tokens
        self.map_encoder = nn.Sequential(ConvBlock(6, 32), nn.AvgPool2d(2, ceil_mode=True),
            ConvBlock(32, HIDDEN), nn.AdaptiveAvgPool2d((8, 8)))
        self.global_encoder = nn.Sequential(nn.Linear(7, HIDDEN), nn.SiLU())
        self.candidate_encoder = nn.Sequential(nn.Linear(candidate_features, HIDDEN), nn.SiLU())
        self.cross_attention = nn.MultiheadAttention(HIDDEN, heads, batch_first=True)
        self.context = RecurrentTrack()
        self.scorer = nn.Sequential(nn.Linear(HIDDEN * 3, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 1))
        self.uncertainty = nn.Linear(HIDDEN * 3, 1)

    def forward(self, raster, globals_, candidate_features, hidden=None,
                candidate_mask=None, **_):
        map_tokens = self.map_encoder(raster).flatten(2).transpose(1, 2)
        if map_tokens.shape[1] > self.tokens:
            map_tokens = F.adaptive_avg_pool1d(map_tokens.transpose(1, 2), self.tokens).transpose(1, 2)
        global_state = self.global_encoder(globals_)
        candidate = self.candidate_encoder(candidate_features)
        queried, _ = self.cross_attention(candidate, map_tokens, map_tokens, need_weights=False)
        if candidate_mask is None:
            candidate_mask = torch.ones(candidate_features.shape[:2],
                device=candidate_features.device, dtype=torch.bool)
        if (candidate_mask.shape != candidate_features.shape[:2]
                or candidate_mask.dtype != torch.bool
                or candidate_mask.device != candidate_features.device):
            raise ValueError("C2 candidate mask must be bool [B,C] on the candidate device")
        # Preserve the original all-query context. Support masks gate action
        # selection; using them to reweight this pooled state changes C2's
        # recurrent context whenever any candidate is infeasible.
        candidate_context = queried.mean(1)
        state = self.context(global_state + candidate_context, hidden)
        fused = torch.cat((queried, state[:, None].expand(-1, queried.shape[1], -1), candidate), -1)
        scored = F.silu(_fixed_order_candidate_linear(fused, self.scorer[0]))
        utility = _fixed_order_candidate_linear(scored, self.scorer[2]).squeeze(-1)
        return {"candidate_utility": utility,
                "uncertainty": F.softplus(self.uncertainty(fused).squeeze(-1)), "hidden": state,
                "map_token_count": map_tokens.shape[1]}


def make_meco_track(track: str, *, candidate_features: int = MECO_CANDIDATE_RAW_FEATURE_COUNT, **kwargs) -> nn.Module:
    """Build a scorer; C1/C2 receive raw columns plus same-width missingness."""
    if type(candidate_features) is not int or candidate_features < 1:
        raise ValueError("candidate_features must be a positive raw feature count")
    factories = {"A1": A1MultiscaleUNetGRU, "A2": A2StructuredSE2UNetGRU,
                 "A3": A3HierarchicalSE2GNNGRU,
                 "C1": lambda: CandidateMLPGRU(candidate_features * 2, **kwargs),
                 "C2": lambda: C2MapCrossAttentionGRU(candidate_features * 2, **kwargs)}
    if track not in factories:
        raise ValueError(f"unknown MECO track {track!r}; expected one of {MECO_TRACKS}")
    return factories[track]()


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    observation: ObservationSnapshot
    candidates: CandidateTable
    now: float
    max_age: float
    fallback_candidate_id: str
    expected_model_id: str

    def __post_init__(self):
        if not self.expected_model_id:
            raise ValueError("expected_model_id is required")
        try:
            uuid.UUID(self.observation.episode_id.split(":", 1)[0])
        except (ValueError, AttributeError):
            raise ValueError("episode_id must begin with a unique UUID generation")
        if self.observation.snapshot_id != f"{self.observation.episode_id}:{self.observation.sequence}":
            raise ValueError("observation snapshot ID must bind episode generation and sequence")
        if self.candidates.snapshot_id != self.observation.snapshot_id:
            raise ValueError("candidate table must bind the exact observation snapshot")
        if not math.isfinite(self.now) or not math.isfinite(self.max_age) or self.max_age < 0.:
            raise ValueError("decision time and maximum age must be finite and nonnegative")


def choose_candidate(request: DecisionRequest, output: ScorerSnapshot | None) -> tuple[str, str]:
    """Validate exact fresh model output, then choose among feasible rows.

    Returns `(candidate_id, source)`. The candidate table is assumed to have
    been emitted by a deterministic feasibility service. Invalid, stale,
    mismatched, unavailable, or non-finite outputs all use the deterministic
    safe fallback. The model cannot alter candidate validity or constraints.
    """
    table = request.candidates
    try:
        if (not isinstance(output, ScorerSnapshot) or
                not output.matches(request.observation, table,
                                   now=request.now, max_age=request.max_age)):
            raise ValueError("missing, stale, or mismatched scorer result")
        if output.model_id != request.expected_model_id:
            raise ValueError("scorer model identity mismatch")
        if (output.observation_digest != observation_digest(request.observation)
                or output.candidate_digest != candidate_digest(table)):
            raise ValueError("scorer content digest mismatch")
        if request.fallback_candidate_id not in table.candidate_ids:
            raise ValueError("fallback is absent from candidate table")
        fallback = table.candidate_ids.index(request.fallback_candidate_id)
        selected = masked_argmax(output.scores, table.valid_mask, fallback_index=fallback)
        if selected is None:
            raise ValueError("no deterministic-feasible candidate")
        return table.candidate_ids[selected], "model"
    except (TypeError, ValueError, OverflowError):
        if request.fallback_candidate_id not in table.candidate_ids:
            raise ValueError("deterministic fallback missing from candidate table")
        index = table.candidate_ids.index(request.fallback_candidate_id)
        if not table.valid_mask[index]:
            raise ValueError("deterministic fallback is not marked feasible")
        return request.fallback_candidate_id, "deterministic_fallback"


def make_scorer_snapshot(model_id: str, observation: ObservationSnapshot,
                         candidates: CandidateTable, scores, timestamp: float) -> ScorerSnapshot:
    """Bind scorer output to exact observation, feature rows, mask, IDs, and order."""
    if candidates.snapshot_id != observation.snapshot_id:
        raise ValueError("candidate table and observation snapshot differ")
    return ScorerSnapshot(model_id, observation.snapshot_id, candidates.candidate_ids,
                          tuple(float(score) for score in scores), float(timestamp),
                          schema_version=2,
                          observation_digest=observation_digest(observation),
                          candidate_digest=candidate_digest(candidates))


def candidate_policy_logits(utility: torch.Tensor, valid_mask: torch.Tensor,
                            route_cost: torch.Tensor | None = None,
                            risk_cost: torch.Tensor | None = None,
                            path_cost: torch.Tensor | None = None, *,
                            validate_inputs: bool = True) -> torch.Tensor:
    """Gross reward minus ETA, risk, and integrated Plan A path costs once."""
    if utility.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
        raise ValueError("utility and boolean feasibility mask shapes must match")
    if validate_inputs and not bool(torch.isfinite(utility[valid_mask]).all()):
        raise ValueError("valid candidate utilities must be finite")
    utility_finite = torch.isfinite(utility)
    valid_mask = valid_mask & utility_finite
    value = torch.where(utility_finite, utility, torch.zeros_like(utility))
    if route_cost is not None:
        if route_cost.shape != value.shape:
            raise ValueError("route costs must match utility")
        if validate_inputs and not bool(torch.isfinite(route_cost[valid_mask]).all()):
            raise ValueError("valid route costs must be finite and match utility")
        route_finite = torch.isfinite(route_cost)
        valid_mask = valid_mask & route_finite
        value = value - torch.where(route_finite, route_cost,
                                    torch.zeros_like(route_cost))
    if risk_cost is not None:
        if risk_cost.shape != value.shape:
            raise ValueError("risk costs must match utility")
        if validate_inputs and not bool(torch.isfinite(risk_cost[valid_mask]).all()):
            raise ValueError("valid risk costs must be finite and match utility")
        risk_finite = torch.isfinite(risk_cost)
        valid_mask = valid_mask & risk_finite
        value = value - torch.where(risk_finite, risk_cost,
                                    torch.zeros_like(risk_cost))
    if path_cost is not None:
        if path_cost.shape != value.shape:
            raise ValueError("path costs must match utility")
        if validate_inputs and not bool(torch.isfinite(path_cost[valid_mask]).all()):
            raise ValueError("valid path costs must be finite and match utility")
        path_finite = torch.isfinite(path_cost)
        valid_mask = valid_mask & path_finite
        value = value - torch.where(path_finite, path_cost,
                                    torch.zeros_like(path_cost))
    if validate_inputs and not bool(valid_mask.any(dim=-1).all()):
        raise ValueError("each row must include at least one feasible candidate")
    return value.masked_fill(~valid_mask, -torch.inf)


def candidate_feature_tensor(table: CandidateTable, *, device, dtype=torch.float32,
                             raw_features: torch.Tensor | None = None,
                             missing_mask: torch.Tensor | None = None):
    """Numerically encode named candidate rows and append missingness indicators.

    Missing estimates stay distinguishable from a real zero. The returned
    feature names and unavailable masks are part of the evaluation record.
    """
    if raw_features is not None or missing_mask is not None:
        if (not isinstance(raw_features, torch.Tensor) or
                not isinstance(missing_mask, torch.Tensor)):
            raise ValueError("raw_features and missing_mask must be supplied together")
        expected = (len(table.candidate_ids), len(table.feature_names))
        if (raw_features.shape != expected or missing_mask.shape != expected or
                raw_features.device != torch.device(device) or
                missing_mask.device != torch.device(device) or
                raw_features.dtype != dtype or missing_mask.dtype != dtype or
                not raw_features.is_floating_point() or
                not missing_mask.is_floating_point()):
            raise ValueError("pre-encoded candidate tensors differ from the table contract")
        return raw_features, missing_mask
    values = []
    missing = []
    for row in table.features:
        values.append([0. if value is None else float(value) for value in row])
        missing.append([1. if value is None else 0. for value in row])
    return (torch.tensor(values, device=device, dtype=dtype),
            torch.tensor(missing, device=device, dtype=dtype))


def score_candidates(track: str, model: nn.Module, *, raster: torch.Tensor,
                     globals_: torch.Tensor, table: CandidateTable,
                     candidate_batch=None,
                     hidden: torch.Tensor | None = None,
                     graph: Mapping[str, torch.Tensor] | None = None,
                     route_cost: torch.Tensor | None = None,
                     risk_cost: torch.Tensor | None = None,
                     inference_request: MecoInferenceRequest | None = None,
                     inference_now: float | None = None,
                     observation_timestamp: float | None = None,
                     capture_monotonic: float | None = None):
    """Run one track and return masked categorical logits plus recurrent state.

    Candidate rows for A tracks must name `goal_x_m`, `goal_y_m`,
    `heading_bin`, and `behavior_index`; A2/A3 additionally require
    `legal_node_index` and graph tensors. Thus the model only scores rows the
    deterministic generator already declared valid.
    """
    if raster.ndim != 4 or raster.shape[1] != 6 or globals_.shape != (raster.shape[0], 7):
        raise ValueError("expected raster [B,6,H,W] and globals [B,7]")
    if raster.shape[0] != 1:
        raise ValueError("CandidateTable adapter currently scores one decision at a time")
    if track in ("C1", "C2"):
        expected_features = tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)
        if (tuple(table.feature_names) != expected_features or
                any(len(row) != MECO_CANDIDATE_RAW_FEATURE_COUNT for row in table.features)):
            raise ValueError("C1/C2 candidate rows differ from the registered shared feature schema")
    if candidate_batch is None:
        candidate, unavailable = candidate_feature_tensor(
            table, device=raster.device, dtype=raster.dtype)
        candidate = torch.cat((candidate, unavailable), -1)[None]
        valid_candidates = torch.tensor(table.valid_mask, device=raster.device,
                                        dtype=torch.bool)[None]
    else:
        if (tuple(candidate_batch.candidate_ids) != tuple(table.candidate_ids) or
                tuple(candidate_batch.feature_names) != tuple(table.feature_names)):
            raise ValueError("device candidate metadata differs from CandidateTable order/schema")
        candidate_feature_tensor(table,
            device=raster.device, dtype=raster.dtype,
            raw_features=candidate_batch.raw_features[0],
            missing_mask=candidate_batch.missing_mask[0])
        candidate = candidate_batch.candidate_features
        if (candidate.shape != (1, len(table.candidate_ids),
                                MECO_CANDIDATE_MODEL_INPUT_WIDTH) or
                candidate.device != raster.device or candidate.dtype != raster.dtype):
            raise ValueError("device candidate feature tensor violates model input contract")
        valid_candidates = candidate_batch.valid_mask
        if (valid_candidates.shape != (1, len(table.candidate_ids)) or
                valid_candidates.dtype != torch.bool or
                valid_candidates.device != raster.device):
            raise ValueError("device candidate validity mask violates decision contract")
    if isinstance(model, ProductionMecoAdapter) and hidden is None:
        hidden = raster.new_zeros((raster.shape[0], HIDDEN))
    inputs = {"raster": raster, "globals_": globals_, "candidate_features": candidate,
              "hidden": hidden}
    if track in ("A2", "A3"):
        if graph is None:
            raise ValueError(f"{track} requires a deterministic legal SE(2) graph")
        expected = ({"node_features", "edges", "edge_mask", "region_ids",
                     "coarse_edges", "coarse_edge_mask"} if track == "A2" else
                    {"node_features", "edges", "edge_mask", "region_ids"})
        if set(graph) != expected:
            raise ValueError(f"{track} graph tensors must be exactly {sorted(expected)}")
        if "globals_" in graph or "hidden" in graph:
            raise ValueError("graph data cannot override observation state or recurrent context")
        inputs.update(graph)
    if isinstance(model, ProductionMecoAdapter):
        response = None
        broker_handle = None
        ordered_inputs = ({name: inputs[name] for name in MECO_INPUT_NAMES[track]
                           if name in inputs} if track == model.track else {})
        if track == model.track and inference_request is None:
            dispatch = time.monotonic()
            if (observation_timestamp is not None and capture_monotonic is not None
                    and dispatch <= capture_monotonic + MECO_MAX_INFERENCE_AGE_S):
                try:
                    inference_request = make_meco_inference_request(model, table=table,
                        observation_timestamp=observation_timestamp,
                        capture_monotonic=capture_monotonic, inputs=ordered_inputs,
                        candidate_batch=candidate_batch)
                except Exception:
                    inference_request = None
        expected_support_mode = ("device-mask" if candidate_batch is not None else "table-mask")
        expected_candidate_digest = (ordered_candidate_order_digest(table.candidate_ids)
            if candidate_batch is not None else
            ordered_candidate_mask_digest(table.candidate_ids, table.valid_mask))
        request_matches = (inference_request is not None and track == model.track
            and inference_request.track == track
            and inference_request.snapshot_id == table.snapshot_id
            and inference_request.candidate_ids == table.candidate_ids
            and inference_request.candidate_support_mode == expected_support_mode
            and (candidate_batch is not None or inference_request.valid_mask == table.valid_mask)
            and inference_request.candidate_mask_digest == expected_candidate_digest)
        if request_matches:
            ordered_inputs = {name: inputs[name] for name in inference_request.input_names
                              if name in inputs}
            broker_inputs = dict(ordered_inputs)
            if track in ("C1", "C2") and candidate_batch is not None:
                broker_inputs["__candidate_valid_mask"] = valid_candidates
            broker_handle = model.submit_inference(inference_request, broker_inputs)

        # Compute the complete deterministic decision while accelerator
        # inference is in flight. A missing result uses this exact fallback.
        utility = torch.zeros((len(table.candidate_ids),), device=raster.device, dtype=raster.dtype)
        valid = valid_candidates.to(device=utility.device)
        if route_cost is not None or risk_cost is not None:
            fallback_logits = candidate_policy_logits(utility[None], valid, route_cost, risk_cost,
                validate_inputs=candidate_batch is None)
            fallback_available = valid.any(dim=-1)
        else:
            fallback = (inference_request.fallback_index if inference_request is not None else
                        next((i for i, ok in enumerate(table.valid_mask) if ok), -1))
            fallback_logits = torch.full((1, len(table.candidate_ids)), -torch.inf,
                                         device=utility.device, dtype=utility.dtype)
            if candidate_batch is not None:
                fallback = candidate_batch.fallback_index
                in_range = (fallback >= 0) & (fallback < len(table.candidate_ids))
                support = torch.zeros_like(fallback_logits, dtype=torch.bool)
                support.scatter_(1, fallback.clamp(0, len(table.candidate_ids) - 1).reshape(1, 1), True)
                support = support & valid & in_range.reshape(1, 1)
                fallback_logits = fallback_logits.masked_fill(support, 0.)
                fallback_available = support.any(dim=-1)
            elif fallback < 0 or not table.valid_mask[fallback]:
                raise ValueError("invalid deterministic fallback for failed inference")
            else:
                fallback_logits[0, fallback] = 0.
                fallback_available = torch.ones((1,), device=utility.device, dtype=torch.bool)

        if request_matches:
            response = model.poll_inference(inference_request, broker_handle)
        if response is not None and (
                response.schema != MECO_INFERENCE_SCHEMA
                or response.track != inference_request.track
                or response.model_id != inference_request.model_id
                or response.checkpoint_sha256 != inference_request.checkpoint_sha256
                or response.protocol_id != inference_request.protocol_id
                or response.request_sequence != inference_request.request_sequence
                or response.generation_id != inference_request.generation_id
                or response.snapshot_id != inference_request.snapshot_id
                or response.candidate_mask_digest != inference_request.candidate_mask_digest
                or response.legal_pose_order_digest != inference_request.legal_pose_order_digest
                or response.input_contract_sha256 != inference_request.input_contract_sha256
                or response.observation_timestamp != inference_request.observation_timestamp
                or response.capture_monotonic != inference_request.capture_monotonic
                or response.dispatch_monotonic != inference_request.dispatch_monotonic
                or not math.isfinite(response.completed_monotonic)
                or response.completed_monotonic < inference_request.dispatch_monotonic
                or response.expires_monotonic != inference_request.expires_monotonic
                or response.completed_monotonic > inference_request.expires_monotonic
                or time.monotonic() > inference_request.expires_monotonic):
            response = None
        if response is None:
            provenance = {"schema": MECO_INFERENCE_SCHEMA,
                "model_id": f"meco:{track}", "checkpoint_sha256": model.checkpoint_sha256,
                "protocol_id": model.protocol_id,
                "request_sequence": (inference_request.request_sequence
                    if inference_request is not None else None),
                "generation_id": (inference_request.generation_id
                    if inference_request is not None else None),
                "snapshot_id": table.snapshot_id,
                "observation_timestamp": (inference_request.observation_timestamp
                    if inference_request is not None else observation_timestamp),
                "candidate_mask_digest": (inference_request.candidate_mask_digest
                    if inference_request is not None else None),
                "captured_monotonic": (inference_request.capture_monotonic
                    if inference_request is not None else capture_monotonic),
                "dispatch_monotonic": (inference_request.dispatch_monotonic
                    if inference_request is not None else None),
                "completed_monotonic": None,
                "expires_monotonic": (inference_request.expires_monotonic
                    if inference_request is not None else None),
                "counters": model.inference_counters}
            return {"logits": fallback_logits, "utility": utility, "hidden": hidden,
                    "uncertainty": None, "raw": None,
                    "inference_status": "deterministic_fallback",
                    "inference_status_code": torch.where(fallback_available,
                        torch.ones((), device=utility.device, dtype=torch.uint8),
                        torch.full((), 2, device=utility.device, dtype=torch.uint8)),
                    "inference_status_codes": dict(MECO_INFERENCE_OUTCOME_CODES),
                    "inference_provenance": provenance}
        output = response.outputs
    else:
        if track == "C2" and candidate_batch is not None:
            output = model(**inputs, candidate_mask=valid_candidates)
        else:
            output = model(**inputs)
    if track in ("C1", "C2"):
        utility = output["candidate_utility"][0]
    elif track == "A1":
        utility = _gather_dense_candidates(output, table, raster.shape[-2:])[0]
    else:
        utility = _gather_graph_candidates(output, table)[0]
    valid = valid_candidates.to(device=utility.device)
    if candidate_batch is not None:
        valid = valid & torch.isfinite(utility[None])
        logits = candidate_policy_logits(utility[None], valid, route_cost, risk_cost,
                                         validate_inputs=False)
    else:
        logits = candidate_policy_logits(utility[None], valid, route_cost, risk_cost)
    inference_valid = output.get("__inference_valid")
    if inference_valid is not None:
        # Invalid numeric responses must be one-hot at the precomputed fallback
        # index. Route-cost logits are not sufficient: stochastic sampling
        # could otherwise select a different action after an inference fault.
        fallback_index = (candidate_batch.fallback_index if candidate_batch is not None
                          else torch.tensor(inference_request.fallback_index,
                              device=utility.device, dtype=torch.long))
        if candidate_batch is None:
            # Keep the established table-mask fallback behavior used by
            # A1/A2/A3, including route/risk-cost selection when supplied.
            logits = torch.where(inference_valid.reshape(1, 1), logits,
                                 fallback_logits)
            inference_status_code = torch.where(inference_valid,
                torch.zeros((), device=utility.device, dtype=torch.uint8),
                torch.ones((), device=utility.device, dtype=torch.uint8))
        else:
            candidate_count = logits.shape[-1]
            fallback_in_range = (fallback_index >= 0) & (fallback_index < candidate_count)
            safe_fallback_index = fallback_index.clamp(0, candidate_count - 1)
            fallback_support = torch.zeros_like(logits, dtype=torch.bool)
            fallback_support.scatter_(1, safe_fallback_index.reshape(1, 1), True)
            # A placeholder index (currently zero) may exist when route
            # construction found no feasible candidate. It cannot become
            # selectable unless the device feasibility mask authorizes it.
            fallback_support = (fallback_support & valid_candidates &
                                fallback_in_range.reshape(1, 1))
            numeric_fallback_logits = torch.full_like(logits, -torch.inf)
            numeric_fallback_logits = numeric_fallback_logits.masked_fill(fallback_support, 0.)
            fallback_available = fallback_support.any(dim=-1, keepdim=True)
            logits = torch.where(inference_valid.reshape(1, 1), logits,
                                 numeric_fallback_logits)
            no_feasible_candidates = ~valid_candidates.any(dim=-1)
            inference_status_code = torch.where(no_feasible_candidates.squeeze(0),
                torch.full((), 2, device=utility.device, dtype=torch.uint8),
                torch.where(inference_valid,
                    torch.zeros((), device=utility.device, dtype=torch.uint8),
                    torch.where(fallback_available.squeeze(-1),
                        torch.ones((), device=utility.device, dtype=torch.uint8),
                        torch.full((), 2, device=utility.device, dtype=torch.uint8))))
        output_hidden = torch.where(inference_valid, output["hidden"], hidden)
        inference_status = ("device_outcome" if candidate_batch is not None
                            else "device_validated")
    else:
        output_hidden = output["hidden"]
        inference_status_code = None
        inference_status = ("accepted" if isinstance(model, ProductionMecoAdapter)
                            else "trainer_direct")
    provenance = ({"schema": MECO_INFERENCE_SCHEMA,
        "model_id": inference_request.model_id,
        "checkpoint_sha256": inference_request.checkpoint_sha256,
        "protocol_id": inference_request.protocol_id,
        "request_sequence": inference_request.request_sequence,
        "generation_id": inference_request.generation_id,
        "snapshot_id": inference_request.snapshot_id,
        "observation_timestamp": inference_request.observation_timestamp,
        "candidate_mask_digest": inference_request.candidate_mask_digest,
        "captured_monotonic": inference_request.capture_monotonic,
        "dispatch_monotonic": inference_request.dispatch_monotonic,
        "completed_monotonic": response.completed_monotonic,
        "expires_monotonic": inference_request.expires_monotonic,
        "counters": model.inference_counters}
        if isinstance(model, ProductionMecoAdapter) else None)
    return {"logits": logits, "utility": utility, "hidden": output_hidden,
            "uncertainty": output.get("uncertainty"), "raw": output,
            "inference_status": inference_status,
            "inference_status_code": inference_status_code,
            "inference_status_codes": dict(MECO_INFERENCE_OUTCOME_CODES),
            "inference_provenance": provenance}


def _table_columns(table: CandidateTable):
    return {name: i for i, name in enumerate(table.feature_names)}


def _required_column(columns, name):
    if name not in columns:
        raise ValueError(f"candidate table is missing {name!r}")
    return columns[name]


def _gather_dense_candidates(output, table, raster_shape):
    columns = _table_columns(table)
    x_col = _required_column(columns, "goal_x_m")
    y_col = _required_column(columns, "goal_y_m")
    heading_col = _required_column(columns, "heading_bin")
    behavior_col = _required_column(columns, "behavior_index")
    height, width = raster_shape
    # Raster is field-aligned; callers supply dimensions/scales as table metadata
    # features to avoid assuming a particular field geometry.
    sx_col = _required_column(columns, "field_length_m")
    sy_col = _required_column(columns, "field_width_m")
    utility = output["utility"]
    scores = []
    field_dimensions = None
    behavior_logits = output["behavior_logits"]
    for row_index, row in enumerate(table.features):
        vals = (row[x_col], row[y_col], row[heading_col], row[behavior_col], row[sx_col], row[sy_col])
        if any(value is None for value in vals):
            if table.valid_mask[row_index]:
                raise ValueError("valid dense-pose candidate is missing required pose fields")
            scores.append(utility.new_tensor(float("nan")))
            continue
        x, y, heading, behavior, field_length, field_width = map(float, vals)
        if field_length <= 0. or field_width <= 0.:
            raise ValueError("candidate field dimensions must be positive")
        if field_dimensions is None:
            field_dimensions = (field_length, field_width)
        elif (field_length, field_width) != field_dimensions:
            raise ValueError("candidate rows must share exact field dimensions")
        if not (0. <= x < field_length and 0. <= y < field_width):
            raise ValueError("candidate goal pose is outside the configured field")
        if heading != round(heading) or not 0. <= heading < 8.:
            raise ValueError("candidate heading_bin must be an integer in [0, 7]")
        if behavior != round(behavior) or not 0. <= behavior < 5.:
            raise ValueError("candidate behavior_index must be an integer in [0, 4]")
        col = int(x / field_length * width)
        line = int(y / field_width * height)
        hbin = int(heading)
        bidx = int(behavior)
        # P(q) is a traversal-field output; it belongs in deterministic route
        # integration, not as an endpoint penalty. Keep it available in `raw`.
        scores.append(utility[0, hbin, line, col] + behavior_logits[0, bidx])
    return torch.stack(scores)[None]


def _gather_graph_candidates(output, table):
    columns = _table_columns(table)
    node_col = _required_column(columns, "legal_node_index")
    behavior_col = _required_column(columns, "behavior_index")
    utility = output["utility"]
    behavior_logits = output["behavior_logits"]
    scores = []
    for row_index, row in enumerate(table.features):
        node, behavior = row[node_col], row[behavior_col]
        if node is None or behavior is None:
            if table.valid_mask[row_index]:
                raise ValueError("valid graph candidate is missing node or behavior index")
            scores.append(utility.new_tensor(float("nan")))
        else:
            if float(node) != round(float(node)) or not 0 <= float(node) < utility.shape[1]:
                raise ValueError("candidate legal_node_index must reference an existing graph node")
            if float(behavior) != round(float(behavior)) or not 0 <= float(behavior) < 5:
                raise ValueError("candidate behavior_index must be an integer in [0, 4]")
            idx, bidx = int(node), int(behavior)
            scores.append(utility[0, idx] + behavior_logits[0, bidx])
    return torch.stack(scores)[None]


def categorical_from_candidate_logits(logits: torch.Tensor):
    """Build the MECO policy distribution from translation-stabilized logits.

    Categorical normalization is invariant to a shared additive constant per
    row. Removing each row's maximum before constructing the distribution keeps
    logsumexp near zero and avoids losing float32 precision when candidate
    scores have a large common offset. Invalid candidates represented by
    ``-inf`` remain ``-inf`` after centering, so the legal action support is
    unchanged.
    """
    if logits.ndim < 1:
        raise ValueError("candidate logits must have an action dimension")
    centered_logits = logits - logits.amax(dim=-1, keepdim=True)
    # The async assertion above is the device-safe validity check. PyTorch's
    # default distribution validation converts that predicate to a host bool.
    return torch.distributions.Categorical(logits=centered_logits,
                                           validate_args=False)


def sample_candidate_action(logits: torch.Tensor, *, deterministic=False):
    """Sample a legal categorical choice and return action and PPO log-probability."""
    if logits.ndim != 2:
        raise ValueError("each decision requires at least one finite candidate logit")
    torch._assert_async(torch.isfinite(torch.logsumexp(logits, dim=-1)).all(),
                        "each decision requires at least one finite candidate logit")
    distribution = categorical_from_candidate_logits(logits)
    action = logits.argmax(-1) if deterministic else distribution.sample()
    return action, distribution.log_prob(action), distribution.entropy()


@dataclass(frozen=True, slots=True)
class SafeActionCandidates:
    """Currently executable curriculum action subset, not a general planner API."""
    observation: ObservationSnapshot
    table: CandidateTable
    fallback_candidate_id: str
    action_indices: tuple[int, ...]
    device_features: torch.Tensor | None = None
    device_missing_mask: torch.Tensor | None = None
    device_valid_mask: torch.Tensor | None = None
    device_track_indices: torch.Tensor | None = None
    device_fallback_index: torch.Tensor | None = None


@torch.no_grad()
def build_safe_action_candidates(env, observation: torch.Tensor, *, world=0, slot=3,
                                 episode_id: str | None = None) -> SafeActionCandidates:
    """Expose only the simulator's existing eight masked choices in stable row order.

    The existing action mask and action executor remain authoritative. This
    diagnostic subset has no AD* route ETA, capture-pose guarantee, or recall
    claim beyond those eight already executable action rows.
    """
    if env.n != 1 or world != 0 or slot != 3:
        raise ValueError("safe curriculum candidate source currently supports learner slot 3 in one world")
    if observation.device != torch.device(env.device):
        raise ValueError("safe curriculum observation is not on the assigned device")
    mask = observation[world, slot, -8:].bool()
    indices, available, _ = env._candidates()
    track_indices = indices[world, slot]
    candidate_available = available[world, slot]
    points = env.track_pos[world, slot].index_select(0, track_indices)
    pose = env.policy_pose[world, slot]
    delta = points - pose[:2]
    distances = delta.norm(dim=-1)
    confidence = env.track_confidence[world, slot].index_select(0, track_indices).clamp(0., 1.)
    ages = env.track_age[world, slot].index_select(0, track_indices).clamp_min(0.)
    all_points = env.track_pos[world, slot]
    all_mask = env.track_mask[world, slot]
    nearby = (((points[:, None] - all_points[None]).norm(dim=-1) < 1.25) & all_mask[None]).sum(-1)
    opponent_distance = None
    if bool(env.opponent_valid[world, slot]):
        opponent_distance = (points - env.opponent_pose[world, slot, :2]).norm(dim=-1)
    field_length, field_width = float(env.sim.field_length), float(env.sim.field_width)
    # Existing behavior rows retain canonical names and stable action order.
    fixed = {4: ("score", 2), 5: ("intercept", 4), 6: ("ferry", 3), 7: ("hold", 0)}
    ids, rows = [], []
    for action_index in range(8):
        if action_index < 4 and bool(candidate_available[action_index]):
            track = int(track_indices[action_index])
            target = points[action_index]
            angle = torch.atan2(delta[action_index, 1], delta[action_index, 0])
            heading_bin = int(torch.round(torch.remainder(angle, 2. * math.pi) * (8. / (2. * math.pi))).item()) % 8
            ids.append(f"collect:track:{track}")
            rows.append((float(action_index), float(track), float(target[0]), float(target[1]),
                         float(heading_bin), 1., field_length, field_width,
                         float(distances[action_index]), float(nearby[action_index]),
                         float(confidence[action_index]), float(ages[action_index]),
                         (float(opponent_distance[action_index]) if opponent_distance is not None else None),
                         None))
        elif action_index < 4:
            # Keep invalid action rows at their fixed positions for categorical parity.
            ids.append(f"collect:unavailable-slot:{action_index}")
            rows.append((float(action_index), None, None, None, None, 1., field_length,
                         field_width, None, None, None, None, None, None))
        else:
            name, behavior = fixed[action_index]
            ids.append(name)
            if action_index == 4:
                target = env.hub_centers[env.team_ids[slot]]
            elif action_index == 6:
                target = env.hub_centers[env.team_ids[slot]]
            elif action_index == 5 and bool(env.opponent_valid[world, slot]):
                target = env.opponent_pose[world, slot, :2]
            else:
                target = pose[:2]
            distance = float((target - pose[:2]).norm().item())
            angle = torch.atan2(target[1] - pose[1], target[0] - pose[0])
            heading_bin = int(torch.round(torch.remainder(angle, 2. * math.pi) * (8. / (2. * math.pi))).item()) % 8
            rows.append((float(action_index), None, float(target[0]), float(target[1]),
                         float(heading_bin), float(behavior), field_length, field_width,
                         distance, None, None, None, None, None))
    if len(ids) != 8:
        raise RuntimeError("candidate source must preserve all eight action rows")
    generation = new_meco_episode_id(episode_id or "curriculum")
    snap = build_observation_snapshot(env, world=world, slot=slot, episode_id=generation)
    table = CandidateTable(snap.snapshot_id, tuple(ids), tuple(rows),
                           tuple(bool(x) for x in mask.cpu().tolist()), SAFE_ACTION_FEATURES,
                           route_costs=(None,) * 8)
    from .curriculum_scenarios import teacher_actions
    torch._assert_async(mask.any(), "simulator exposed no legal fallback action")
    teacher_action = teacher_actions(env, observation)[world, slot].to(
        device=mask.device, dtype=torch.long)
    safe_teacher_action = teacher_action.clamp(0, 7)
    fallback_index = torch.where(
        (teacher_action >= 0) & (teacher_action < 8) & mask[safe_teacher_action],
        teacher_action, mask.to(torch.long).argmax())
    fallback_action = int(fallback_index.detach().to(device="cpu").item())
    device = observation.device
    dtype = points.dtype
    device_values = torch.zeros((8, len(SAFE_ACTION_FEATURES)),
                                device=device, dtype=dtype)
    device_missing = torch.ones_like(device_values)
    action_ids = torch.arange(8, device=device, dtype=torch.long)
    device_values[:, 0] = action_ids.to(dtype)
    device_values[:, 5] = torch.tensor((1., 1., 1., 1., 2., 4., 3., 0.),
                                      device=device, dtype=dtype)
    device_values[:, 6] = float(env.sim.field_length)
    device_values[:, 7] = float(env.sim.field_width)
    device_missing[:, (0, 5, 6, 7)] = 0.
    collect_valid = candidate_available.to(device=device, dtype=torch.bool)
    collect_heading_bin = torch.round(torch.remainder(
        torch.atan2(delta[:, 1], delta[:, 0]), 2. * math.pi) *
        (8. / (2. * math.pi))).remainder(8).to(dtype)
    device_values[:4, 1] = track_indices.to(dtype)
    device_values[:4, 2:4] = points
    device_values[:4, 4] = collect_heading_bin
    device_values[:4, 8] = distances
    device_values[:4, 9] = nearby.to(dtype)
    device_values[:4, 10] = confidence
    device_values[:4, 11] = ages
    device_missing[:4, (1, 2, 3, 4, 8, 9, 10, 11)] = (~collect_valid[:, None]).to(dtype)
    opponent_valid = env.opponent_valid[world, slot].to(device=device, dtype=torch.bool)
    opponent_xy = env.opponent_pose[world, slot, :2]
    collect_opponent_distance = torch.linalg.vector_norm(points - opponent_xy, dim=-1)
    device_values[:4, 12] = collect_opponent_distance
    device_missing[:4, 12] = (~collect_valid | ~opponent_valid).to(dtype)
    fixed_targets = torch.stack((
        env.hub_centers[env.team_ids[slot]],
        torch.where(opponent_valid, opponent_xy, pose[:2]),
        env.hub_centers[env.team_ids[slot]], pose[:2]), dim=0)
    fixed_delta = fixed_targets - pose[:2]
    device_values[4:, 2:4] = fixed_targets
    device_values[4:, 4] = torch.round(torch.remainder(
        torch.atan2(fixed_delta[:, 1], fixed_delta[:, 0]), 2. * math.pi) *
        (8. / (2. * math.pi))).remainder(8).to(dtype)
    device_values[4:, 8] = torch.linalg.vector_norm(fixed_delta, dim=-1)
    device_missing[4:, (2, 3, 4, 8)] = 0.
    device_missing[:, 13] = 1.
    device_values = torch.where(device_missing.bool(),
                                torch.zeros_like(device_values), device_values)
    torch._assert_async(torch.isfinite(device_values).all(),
                        "base candidate features contain non-finite device values")
    return SafeActionCandidates(snap, table, table.candidate_ids[fallback_action],
        tuple(range(8)), device_values, device_missing, mask,
        track_indices.to(device=observation.device, dtype=torch.long),
        fallback_index.to(device=observation.device, dtype=torch.long))


def deterministic_candidate_scores(table: CandidateTable, *, policy="D1",
                                    predicted_opponent_xy=None):
    """Hand-weighted comparator over exactly the same current action rows."""
    if policy not in ("D1", "D2"):
        raise ValueError("policy must be D1 or D2")
    columns = _table_columns(table)
    scores = []
    for row in table.features:
        action = int(row[columns["action_index"]])
        if action < 4 and row[columns["target_distance_m"]] is not None:
            distance = row[columns["target_distance_m"]]
            nearby = row[columns["nearby_sensed_track_count"]] or 0.
            confidence = row[columns["track_confidence"]] or 0.
            age = row[columns["track_age_s"]] or 0.
            score = 1. / (1. + distance) + .12 * nearby + .10 * confidence - .05 * age
            if policy == "D2" and predicted_opponent_xy is not None:
                x, y = row[columns["goal_x_m"]], row[columns["goal_y_m"]]
                d = math.hypot(x - predicted_opponent_xy[0], y - predicted_opponent_xy[1])
                score -= .75 * max(0., min(1., (2. - d) / 2.))
            scores.append(float(score))
        else:
            scores.append({"score": .65, "intercept": -.25,
                           "ferry": .20, "hold": -.5}.get(table.candidate_ids[action], -1.))
    return tuple(scores)


def predict_opponent_endpoint(*, valid, position_xy, velocity_xy, age_s,
                              horizon_s=1., max_speed_mps=3., max_age_s=.5):
    """D2's isolated bounded constant-velocity forecast from an observed track."""
    values = tuple(position_xy) + tuple(velocity_xy) + (age_s, horizon_s, max_speed_mps, max_age_s)
    if not valid or len(position_xy) != 2 or len(velocity_xy) != 2 or not all(math.isfinite(float(v)) for v in values):
        return None
    if age_s < 0. or age_s > max_age_s or horizon_s < 0. or max_speed_mps <= 0.:
        return None
    vx, vy = map(float, velocity_xy)
    speed = math.hypot(vx, vy)
    scale = min(1., max_speed_mps / max(speed, 1e-8))
    return (float(position_xy[0]) + vx * scale * horizon_s,
            float(position_xy[1]) + vy * scale * horizon_s)


def prediction_contract(mode: str, *, horizon_s: float | None = None) -> dict:
    """Canonical forecast parameters included in matched-evaluation provenance."""
    if mode not in PREDICTION_MODES:
        raise ValueError(f"prediction mode must be one of {PREDICTION_MODES}")
    horizon = (0.0 if mode == "current" else
               None if horizon_s is None else float(horizon_s))
    if horizon is not None and (not math.isfinite(horizon) or horizon < 0.):
        raise ValueError("prediction horizon must be finite and nonnegative")
    contract = {"model": MULTIMODEL_ID if mode == "multi_model_v1" else PREDICTION_MODEL_ID,
        "mode": mode,
        "horizon_s": horizon,
        "horizon_semantics": ("observed_current_pose" if mode == "current" else
            "caller_supplied_eta_or_fixed_horizon" if horizon is None else "fixed_or_caller_supplied"),
        "max_speed_mps": PREDICTION_MAX_SPEED_MPS if mode == "cv" else None,
        "max_track_age_s": PREDICTION_MAX_AGE_S if mode == "cv" else None,
        "threat_radius_m": PREDICTION_THREAT_RADIUS_M,
        "max_penalty_points": PREDICTION_MAX_PENALTY_POINTS,
        "fallback": ("none_if_observed_pose_missing_or_nonfinite" if mode == "current"
            else "current_finite_observed_pose_when_cv_unavailable; otherwise_none")}
    if mode == "multi_model_v1":
        contract.update({
            "horizon_semantics": "candidate_route_eta_clamped_to_max_horizon",
            "max_horizon_s": MULTIMODEL_MAX_HORIZON_S,
            "max_speed_mps": PREDICTION_MAX_SPEED_MPS,
            "max_track_age_s": PREDICTION_MAX_AGE_S,
            "hypotheses": [{"id": name, "weight": weight}
                           for name, weight in MULTIMODEL_HYPOTHESES],
            "turn_rate_min_rad_s": MULTIMODEL_TURN_RATE_MIN_RAD_S,
            "turn_rate_max_rad_s": MULTIMODEL_TURN_RATE_MAX_RAD_S,
            "damping_per_s": MULTIMODEL_DAMPING_PER_S,
            "fallback": "finite_observed_pose_if_track_missing_stale_or_nonfinite",
            "risk_semantics": "weighted_expected_penalty_over_hypothesis_endpoints",
        })
    return contract


def opponent_forecast(*, mode: str, valid, position_xy, velocity_xy,
                      age_s, horizon_s: float | None = None,
                      angular_velocity_rad_s: float | None = None) -> dict:
    """Return an observed-pose, bounded CV, or deterministic kinematic mixture."""
    if mode not in PREDICTION_MODES:
        raise ValueError(f"prediction mode must be one of {PREDICTION_MODES}")
    pose = None
    if valid and position_xy is not None and len(position_xy) == 2:
        values = tuple(float(v) for v in position_xy)
        if all(math.isfinite(v) for v in values):
            pose = values
    horizon = (0. if mode == "current" else PREDICTION_DEFAULT_HORIZON_S
               if horizon_s is None else float(horizon_s))
    if not math.isfinite(horizon) or horizon < 0.:
        raise ValueError("prediction horizon must be finite and nonnegative")
    forecast = None
    fallback = None
    if mode == "current":
        forecast = pose
    elif pose is not None and mode == "cv":
        if velocity_xy is not None and age_s is not None:
            forecast = predict_opponent_endpoint(valid=True, position_xy=pose,
                velocity_xy=velocity_xy, age_s=float(age_s), horizon_s=horizon,
                max_speed_mps=PREDICTION_MAX_SPEED_MPS,
                max_age_s=PREDICTION_MAX_AGE_S)
        if forecast is None:
            forecast = pose
            fallback = "observed_current_pose"
    hypotheses = []
    horizon_clamped = bool(mode == "multi_model_v1" and
                           horizon > MULTIMODEL_MAX_HORIZON_S)
    if mode == "multi_model_v1" and pose is not None:
        fresh_motion = (velocity_xy is not None and age_s is not None
            and math.isfinite(float(age_s)) and 0. <= float(age_s) <= PREDICTION_MAX_AGE_S
            and len(velocity_xy) == 2
            and all(math.isfinite(float(v)) for v in velocity_xy))
        if fresh_motion:
            t = min(horizon, MULTIMODEL_MAX_HORIZON_S)
            vx, vy = map(float, velocity_xy)
            speed = math.hypot(vx, vy)
            scale = min(1., PREDICTION_MAX_SPEED_MPS / max(speed, 1e-8))
            vx, vy = vx * scale, vy * scale
            observed_omega = (float(angular_velocity_rad_s)
                if angular_velocity_rad_s is not None and
                   math.isfinite(float(angular_velocity_rad_s)) else 0.)
            turn_rate = min(MULTIMODEL_TURN_RATE_MAX_RAD_S,
                            max(MULTIMODEL_TURN_RATE_MIN_RAD_S, abs(observed_omega)))
            cv_xy = (pose[0] + vx * t, pose[1] + vy * t)
            damped_distance_time = -math.expm1(-MULTIMODEL_DAMPING_PER_S * t) / MULTIMODEL_DAMPING_PER_S
            damped_xy = (pose[0] + vx * damped_distance_time,
                         pose[1] + vy * damped_distance_time)
            for name, weight in MULTIMODEL_HYPOTHESES:
                if name == "constant_velocity":
                    xy = cv_xy
                elif name.startswith("turn_"):
                    omega = turn_rate * (1. if name == "turn_left" else -1.)
                    angle = omega * t
                    if abs(omega) < 1e-8:
                        dx, dy = vx * t, vy * t
                    else:
                        dx = (vx * math.sin(angle) + vy * (math.cos(angle) - 1.)) / omega
                        dy = (vx * (1. - math.cos(angle)) + vy * math.sin(angle)) / omega
                    xy = (pose[0] + dx, pose[1] + dy)
                else:
                    xy = damped_xy
                hypotheses.append({"id": name, "weight": weight, "xy": xy})
            forecast = tuple(sum(item["weight"] * item["xy"][axis]
                                 for item in hypotheses) for axis in (0, 1))
        else:
            fallback = "observed_current_pose"
            hypotheses = [{"id": "observed_current_pose", "weight": 1., "xy": pose}]
            forecast = pose
    if mode == "multi_model_v1" and pose is None:
        fallback = "none_no_finite_observed_pose"
    result = {"xy": forecast, "mode": mode, "horizon_s": horizon,
        "observed_age_s": (float(age_s) if age_s is not None and
                            math.isfinite(float(age_s)) else None),
        "used_cv": bool(mode == "cv" and forecast is not None and fallback is None),
        "fallback": fallback}
    if mode == "multi_model_v1":
        result.update({"model": MULTIMODEL_ID,
            "horizon_s": min(horizon, MULTIMODEL_MAX_HORIZON_S),
            "requested_horizon_s": horizon,
            "horizon_clamped": horizon_clamped,
            "hypotheses": hypotheses,
            "used_multimodel": len(hypotheses) > 1})
    return result


def opponent_risk_penalty(goal_xy, forecast: Mapping) -> float | None:
    """Return bounded expected proximity cost over a forecast's hypotheses."""
    hypotheses = forecast.get("hypotheses") or []
    if not hypotheses:
        xy = forecast.get("xy")
        hypotheses = [] if xy is None else [{"weight": 1., "xy": xy}]
    if not hypotheses:
        return None
    weighted = total = 0.
    for item in hypotheses:
        xy = item.get("xy")
        weight = float(item.get("weight", 0.))
        if (xy is None or len(xy) != 2 or not math.isfinite(weight) or weight <= 0.
                or not all(math.isfinite(float(value)) for value in xy)):
            continue
        separation = math.hypot(float(goal_xy[0]) - float(xy[0]),
                                float(goal_xy[1]) - float(xy[1]))
        penalty = PREDICTION_MAX_PENALTY_POINTS * max(0., min(1.,
            (PREDICTION_THREAT_RADIUS_M - separation) / PREDICTION_THREAT_RADIUS_M))
        weighted += weight * penalty
        total += weight
    return None if total <= 0. else weighted / total
