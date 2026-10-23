"""Five MECO research scorers and a fail-closed decision adapter.

This module is deliberately separate from the production controller. Models
score goals/candidates; the caller supplies deterministic feasibility and
legality, and only the deterministic arbiter may execute an action.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Mapping
import uuid

import torch
from torch import nn
from torch.nn import functional as F

from .learning_contract import CandidateTable, ObservationSnapshot, ScorerSnapshot, masked_argmax
from .learning_inputs import build_observation_snapshot


MECO_INFERENCE_SCHEMA = "meco-inference-v1"
MECO_INPUT_NAMES = {
    "A1": ("raster", "globals_", "hidden"),
    "A2": ("node_features", "edges", "edge_mask", "region_ids", "globals_",
           "coarse_edges", "coarse_edge_mask", "hidden"),
    "A3": ("node_features", "edges", "edge_mask", "region_ids", "globals_", "hidden"),
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
    expires_at: float
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    candidate_ids: tuple[str, ...]
    valid_mask: tuple[bool, ...]
    fallback_index: int
    schema: str = MECO_INFERENCE_SCHEMA

    def __post_init__(self):
        if (self.schema != MECO_INFERENCE_SCHEMA or self.track not in MECO_TRACKS
                or self.observation_schema != 1 or not self.snapshot_id):
            raise ValueError("unsupported MECO track or observation schema")
        if (len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256)
                or not self.protocol_id):
            raise ValueError("checkpoint SHA-256 and protocol identity are required")
        if (not math.isfinite(self.observation_timestamp) or not math.isfinite(self.expires_at)
                or self.expires_at < self.observation_timestamp):
            raise ValueError("invalid observation timestamp or expiry")
        if self.input_names != MECO_INPUT_NAMES[self.track] or len(self.input_shapes) != len(self.input_names):
            raise ValueError("track input names/order do not match the versioned contract")
        if (not self.candidate_ids or any(not isinstance(item, str) or not item for item in self.candidate_ids)
                or len(set(self.candidate_ids)) != len(self.candidate_ids)
                or any(not shape or any(type(size) is not int or size <= 0 for size in shape)
                       for shape in self.input_shapes)
                or len(self.candidate_ids) != len(self.valid_mask)
                or any(type(v) is not bool for v in self.valid_mask)
                or not 0 <= self.fallback_index < len(self.candidate_ids)
                or not self.valid_mask[self.fallback_index]):
            raise ValueError("candidate ordering, mask, or deterministic fallback is invalid")


@dataclass(frozen=True, slots=True)
class MecoInferenceResponse:
    """Scorer values plus immutable provenance copied from its request."""
    schema: str
    track: str
    checkpoint_sha256: str
    protocol_id: str
    observation_timestamp: float
    completed_at: float
    outputs: Mapping[str, torch.Tensor]


def make_meco_inference_request(adapter: "ProductionMecoAdapter", *,
                                table: CandidateTable,
                                observation_timestamp: float, expires_at: float,
                                inputs: Mapping[str, torch.Tensor]) -> MecoInferenceRequest:
    """Freeze request identity/order/shapes from current planner tensors."""
    names = MECO_INPUT_NAMES[adapter.track]
    if tuple(inputs) != names:
        raise ValueError("inputs must be supplied in the track's canonical order")
    return MecoInferenceRequest(track=adapter.track, snapshot_id=table.snapshot_id,
        checkpoint_sha256=adapter.checkpoint_sha256, protocol_id=adapter.protocol_id,
        observation_schema=1, observation_timestamp=float(observation_timestamp),
        expires_at=float(expires_at), input_names=names,
        input_shapes=tuple(tuple(inputs[name].shape) for name in names),
        candidate_ids=table.candidate_ids, valid_mask=table.valid_mask,
        fallback_index=next((i for i, valid in enumerate(table.valid_mask) if valid), -1))


class ProductionMecoAdapter:
    """Validate real track tensors and scorer outputs before planner use.

    Invalid, expired, stale, or faulting inference returns ``None``; caller
    must use its already-feasible deterministic fallback.
    """
    def __init__(self, track: str, model: nn.Module, *, checkpoint_sha256: str,
                 protocol_id: str):
        if track not in MECO_TRACKS:
            raise ValueError(f"unsupported MECO track: {track}")
        if (len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
                or not protocol_id):
            raise ValueError("production adapter requires checkpoint and protocol identity")
        self.track, self.model = track, model
        self.checkpoint_sha256, self.protocol_id = checkpoint_sha256, protocol_id

    def infer(self, request: MecoInferenceRequest, inputs: Mapping[str, torch.Tensor], *, now: float):
        try:
            if (request.track != self.track or request.checkpoint_sha256 != self.checkpoint_sha256
                    or request.protocol_id != self.protocol_id or request.observation_schema != 1
                    or not math.isfinite(now) or now < request.observation_timestamp or now > request.expires_at
                    or tuple(inputs) != request.input_names):
                return None
            tensors = tuple(inputs[name] for name in request.input_names)
            if tuple(tuple(t.shape) for t in tensors) != request.input_shapes:
                return None
            if any(not isinstance(t, torch.Tensor) for t in tensors):
                return None
            if any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for t in tensors):
                return None
            for name, tensor in zip(request.input_names, tensors):
                if name in ("edge_mask", "coarse_edge_mask") and tensor.dtype != torch.bool:
                    return None
                if name in ("edges", "coarse_edges", "region_ids") and tensor.dtype != torch.long:
                    return None
            by_name = dict(inputs)
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
            if "coarse_edges" in by_name and (by_name["coarse_edges"].ndim != 3
                    or by_name["coarse_edge_mask"].shape != by_name["coarse_edges"].shape):
                return None
            outputs = self.model(**dict(inputs))
            expected = (("utility", "penalty", "behavior_logits", "hidden") if self.track.startswith("A")
                        else ("candidate_utility", "uncertainty", "hidden"))
            if not isinstance(outputs, Mapping) or any(name not in outputs for name in expected):
                return None
            values = {name: outputs[name] for name in expected}
            if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
                   or not bool(torch.isfinite(value).all()) for value in values.values()):
                return None
            if values["hidden"].shape != (batch, HIDDEN):
                return None
            if self.track.startswith("A"):
                utility_shape_ok = (
                    values["utility"].ndim == 4 and values["utility"].shape[:2] == (batch, 8)
                    and values["utility"].shape[-2:] == by_name["raster"].shape[-2:]
                    if self.track == "A1" else
                    values["utility"].ndim == 2 and values["utility"].shape == by_name["node_features"].shape[:2])
                if (not utility_shape_ok or values["penalty"].shape != values["utility"].shape
                        or values["behavior_logits"].shape != (batch, 5)
                        or bool((values["penalty"] < 0).any())):
                    return None
            elif (values["candidate_utility"].shape != (batch, len(request.candidate_ids))
                  or values["uncertainty"].shape != values["candidate_utility"].shape
                  or bool((values["uncertainty"] < 0).any())):
                return None
            done = time.monotonic()
            if done > request.expires_at:
                return None
            return MecoInferenceResponse(MECO_INFERENCE_SCHEMA, self.track,
                self.checkpoint_sha256, self.protocol_id, request.observation_timestamp,
                done, values)
        except Exception:
            return None


MECO_TRACKS = ("A1", "A2", "A3", "C1", "C2")
MECO_SCHEMA_VERSION = 1
HIDDEN = 64
PREDICTION_MODES = ("current", "cv")
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
PREDICTION_MODES = ("current", "cv")
PREDICTION_MODEL_ID = "meco_opponent_forecast_v1"
PREDICTION_MAX_SPEED_MPS = 3.0
PREDICTION_MAX_AGE_S = 0.5
PREDICTION_DEFAULT_HORIZON_S = 1.0
PREDICTION_THREAT_RADIUS_M = 2.0
PREDICTION_MAX_PENALTY_POINTS = 0.75


def prediction_contract(mode: str, *, horizon_s: float | None = None) -> dict:
    """Describe forecast semantics in immutable training/evaluation provenance."""
    if mode not in PREDICTION_MODES:
        raise ValueError(f"prediction mode must be one of {PREDICTION_MODES}")
    horizon = (0.0 if mode == "current" else
               None if horizon_s is None else float(horizon_s))
    if horizon is not None and (not math.isfinite(horizon) or horizon < 0.):
        raise ValueError("prediction horizon must be finite and nonnegative")
    return {"model": PREDICTION_MODEL_ID, "mode": mode,
        "horizon_s": horizon,
        "horizon_semantics": ("observed_current_pose" if mode == "current" else
            "caller_supplied_eta_or_fixed_horizon" if horizon is None else "fixed_or_caller_supplied"),
        "max_speed_mps": PREDICTION_MAX_SPEED_MPS if mode == "cv" else None,
        "max_track_age_s": PREDICTION_MAX_AGE_S if mode == "cv" else None,
        "threat_radius_m": PREDICTION_THREAT_RADIUS_M,
        "max_penalty_points": PREDICTION_MAX_PENALTY_POINTS,
        "fallback": ("none_if_observed_pose_missing_or_nonfinite" if mode == "current"
            else "current_finite_observed_pose_when_cv_unavailable; otherwise_none")}


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

    def forward(self, nodes, edges, edge_mask):
        # edges [B,N,D] contains neighbor node indices; edge_mask marks legal transitions.
        if (nodes.ndim != 3 or edges.ndim != 3 or edge_mask.shape != edges.shape
                or edges.shape[:2] != nodes.shape[:2] or edges.dtype != torch.long
                or edge_mask.dtype != torch.bool):
            raise ValueError("invalid SE(2) graph tensor shapes or dtypes")
        if bool((edge_mask & ((edges < 0) | (edges >= nodes.shape[1]))).any()):
            raise ValueError("a legal SE(2) edge references an out-of-range node")
        safe = torch.where(edge_mask, edges, torch.zeros_like(edges))
        expanded = nodes[:, :, None, :].expand(-1, -1, safe.shape[-1], -1)
        neighbors = expanded.gather(1, safe[..., None].expand(-1, -1, -1, nodes.shape[-1]))
        messages = self.message(neighbors)
        mask = edge_mask[..., None].to(messages.dtype)
        aggregate = (messages * mask).sum(2) / mask.sum(2).clamp_min(1.)
        return self.norm(nodes + self.update(torch.cat((nodes, aggregate), -1)))


def _se2_node_inputs(node_features, globals_):
    """Encode headings as sin/cos so bins 7 and 0 obey angular wraparound."""
    if (node_features.ndim != 3 or node_features.shape[-1] != 9
            or globals_.shape != (node_features.shape[0], 7)):
        raise ValueError("SE(2) nodes require six raster values, x/y, heading-bin and seven globals")
    if not bool(torch.isfinite(node_features).all()) or not bool(torch.isfinite(globals_).all()):
        raise ValueError("SE(2) node features and globals must be finite")
    if bool(((node_features[..., 6:8] < 0.) | (node_features[..., 6:8] > 1.)).any()):
        raise ValueError("SE(2) node x/y coordinates must be normalized to [0, 1]")
    heading = node_features[..., 8]
    if bool(((heading < 0.) | (heading > 7.) | (heading != heading.round())).any()):
        raise ValueError("heading bins must be integers in [0, 7]")
    angle = heading * (2. * math.pi / 8.)
    heading_vector = torch.stack((angle.sin(), angle.cos()), -1)
    return torch.cat((node_features[..., :8], heading_vector,
                      globals_[:, None, :].expand(-1, node_features.shape[1], -1)), -1)


def _pool_regions(nodes, region_ids, region_tokens):
    batch, count, width = nodes.shape
    if (region_ids.shape != (batch, count) or region_ids.dtype != torch.long
            or bool(((region_ids < 0) | (region_ids >= region_tokens)).any())):
        raise ValueError("region IDs must be integer indices within the configured regions")
    pooled = nodes.new_zeros((batch, region_tokens, width))
    counts = nodes.new_zeros((batch, region_tokens, 1))
    pooled.scatter_add_(1, region_ids[..., None].expand_as(nodes), nodes)
    counts.scatter_add_(1, region_ids[..., None], nodes.new_ones((batch, count, 1)))
    occupied = counts.squeeze(-1) > 0
    if not bool(occupied.any(-1).all()):
        raise ValueError("each SE(2) graph must contain at least one legal node")
    pooled = pooled / counts.clamp_min(1.)
    return pooled, occupied


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
                coarse_edges, coarse_edge_mask, hidden=None, **_):
        nodes = self.lift(_se2_node_inputs(node_features, globals_))
        for layer in self.fine_layers:
            nodes = layer(nodes, edges, edge_mask)
        batch, count, _ = nodes.shape
        coarse, occupied = _pool_regions(nodes, region_ids, self.region_tokens)
        coarse = self.coarse_layer(coarse, coarse_edges, coarse_edge_mask)
        attended, _ = self.attention(coarse, coarse, coarse,
                                     key_padding_mask=~occupied, need_weights=False)
        state = self.context((attended * occupied[..., None]).sum(1) /
                             occupied.sum(1, keepdim=True).clamp_min(1.), hidden)
        upsampled = coarse.gather(1, region_ids[..., None].expand(-1, -1, HIDDEN))
        decoded = self.decoder(torch.cat((nodes, upsampled,
                                          state[:, None].expand(-1, count, -1)), -1))
        return {"utility": decoded[..., 0], "penalty": F.softplus(decoded[..., 1]),
                "behavior_logits": self.behavior_head(state), "hidden": state,
                "region_tokens": coarse, "occupied_regions": occupied}


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

    def forward(self, node_features, edges, edge_mask, region_ids, globals_, hidden=None, **_):
        nodes = self.lift(_se2_node_inputs(node_features, globals_))
        for layer in self.layers:
            nodes = layer(nodes, edges, edge_mask)
        batch, count, _ = nodes.shape
        pooled, occupied = _pool_regions(nodes, region_ids, self.region_tokens)
        regions, _ = self.region_attention(pooled, pooled, pooled,
                                           key_padding_mask=~occupied, need_weights=False)
        state = self.context((regions * occupied[..., None]).sum(1) /
                             occupied.sum(1, keepdim=True).clamp_min(1.), hidden)
        contextual = self.region_decoder(torch.cat((regions, state[:, None].expand(-1, self.region_tokens, -1)), -1))
        node_context = contextual.gather(1, region_ids[..., None].expand(-1, -1, HIDDEN))
        outputs = self.node_head(nodes + node_context)
        return {"utility": outputs[..., 0], "penalty": F.softplus(outputs[..., 1]),
                "behavior_logits": self.behavior_head(state), "hidden": state,
                "region_tokens": regions, "occupied_regions": occupied}


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
        return {"candidate_utility": self.scorer(fused).squeeze(-1),
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

    def forward(self, raster, globals_, candidate_features, hidden=None, **_):
        map_tokens = self.map_encoder(raster).flatten(2).transpose(1, 2)
        if map_tokens.shape[1] > self.tokens:
            map_tokens = F.adaptive_avg_pool1d(map_tokens.transpose(1, 2), self.tokens).transpose(1, 2)
        global_state = self.global_encoder(globals_)
        candidate = self.candidate_encoder(candidate_features)
        queried, _ = self.cross_attention(candidate, map_tokens, map_tokens, need_weights=False)
        state = self.context(global_state + queried.mean(1), hidden)
        fused = torch.cat((queried, state[:, None].expand(-1, queried.shape[1], -1), candidate), -1)
        return {"candidate_utility": self.scorer(fused).squeeze(-1),
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
                            path_cost: torch.Tensor | None = None) -> torch.Tensor:
    """Gross reward minus ETA, risk, and integrated Plan A path costs once."""
    if utility.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
        raise ValueError("utility and boolean feasibility mask shapes must match")
    if not bool(torch.isfinite(utility[valid_mask]).all()):
        raise ValueError("valid candidate utilities must be finite")
    value = utility
    if route_cost is not None:
        if route_cost.shape != value.shape or not bool(torch.isfinite(route_cost[valid_mask]).all()):
            raise ValueError("valid route costs must be finite and match utility")
        value = value - route_cost
    if risk_cost is not None:
        if risk_cost.shape != value.shape or not bool(torch.isfinite(risk_cost[valid_mask]).all()):
            raise ValueError("valid risk costs must be finite and match utility")
        value = value - risk_cost
    if path_cost is not None:
        if path_cost.shape != value.shape or not bool(torch.isfinite(path_cost[valid_mask]).all()):
            raise ValueError("valid path costs must be finite and match utility")
        value = value - path_cost
    if not bool(valid_mask.any(dim=-1).all()):
        raise ValueError("each row must include at least one feasible candidate")
    return value.masked_fill(~valid_mask, -torch.inf)


def candidate_feature_tensor(table: CandidateTable, *, device, dtype=torch.float32):
    """Numerically encode named candidate rows and append missingness indicators.

    Missing estimates stay distinguishable from a real zero. The returned
    feature names and unavailable masks are part of the evaluation record.
    """
    values = []
    missing = []
    for row in table.features:
        values.append([0. if value is None else float(value) for value in row])
        missing.append([1. if value is None else 0. for value in row])
    return (torch.tensor(values, device=device, dtype=dtype),
            torch.tensor(missing, device=device, dtype=dtype))


def score_candidates(track: str, model: nn.Module, *, raster: torch.Tensor,
                     globals_: torch.Tensor, table: CandidateTable,
                     hidden: torch.Tensor | None = None,
                     graph: Mapping[str, torch.Tensor] | None = None,
                     route_cost: torch.Tensor | None = None,
                     risk_cost: torch.Tensor | None = None,
                     inference_request: MecoInferenceRequest | None = None,
                     inference_now: float | None = None):
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
    candidate, unavailable = candidate_feature_tensor(table, device=raster.device, dtype=raster.dtype)
    candidate = torch.cat((candidate, unavailable), -1)[None]
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
        if model.track != track:
            response = None
        else:
            ordered_inputs = {name: inputs[name] for name in MECO_INPUT_NAMES[track] if name in inputs}
            if inference_request is None:
                started = time.monotonic() if inference_now is None else inference_now
                fallback_index = next((i for i, valid_value in enumerate(table.valid_mask)
                                       if valid_value), -1)
                if fallback_index < 0:
                    raise ValueError("production scorer requires a valid deterministic fallback")
                inference_request = MecoInferenceRequest(
                    track=track, snapshot_id=table.snapshot_id,
                    checkpoint_sha256=model.checkpoint_sha256,
                    protocol_id=model.protocol_id, observation_schema=1,
                    observation_timestamp=started, expires_at=started + .25,
                    input_names=MECO_INPUT_NAMES[track],
                    input_shapes=tuple(tuple(ordered_inputs[name].shape)
                                       for name in MECO_INPUT_NAMES[track]),
                    candidate_ids=table.candidate_ids,
                    valid_mask=table.valid_mask, fallback_index=fallback_index)
                inference_now = time.monotonic()
        if (inference_request is None or inference_now is None
                or inference_request.track != track
                or inference_request.snapshot_id != table.snapshot_id
                or inference_request.candidate_ids != table.candidate_ids
                or inference_request.valid_mask != table.valid_mask):
            response = None
        else:
            # Keep exact track-specific tensor order as part of the wire contract.
            ordered_inputs = {name: inputs[name] for name in inference_request.input_names if name in inputs}
            response = model.infer(inference_request, ordered_inputs, now=inference_now)
            if response is not None and (
                    response.schema != MECO_INFERENCE_SCHEMA
                    or response.track != inference_request.track
                    or response.checkpoint_sha256 != inference_request.checkpoint_sha256
                    or response.protocol_id != inference_request.protocol_id
                    or response.observation_timestamp != inference_request.observation_timestamp
                    or not math.isfinite(response.completed_at)
                    or response.completed_at > inference_request.expires_at
                    or response.completed_at < inference_request.observation_timestamp):
                response = None
        if response is None:
            utility = torch.zeros((len(table.candidate_ids),), device=raster.device, dtype=raster.dtype)
            valid = torch.tensor(table.valid_mask, device=utility.device, dtype=torch.bool)[None]
            if route_cost is not None or risk_cost is not None:
                logits = candidate_policy_logits(utility[None], valid, route_cost, risk_cost)
            else:
                fallback = inference_request.fallback_index if inference_request is not None else next(
                    (i for i, ok in enumerate(table.valid_mask) if ok), -1)
                logits = torch.full((1, len(table.candidate_ids)), -torch.inf,
                                    device=utility.device, dtype=utility.dtype)
                if fallback < 0 or fallback >= len(table.candidate_ids) or not table.valid_mask[fallback]:
                    raise ValueError("invalid deterministic fallback for failed inference")
                logits[0, fallback] = 0.
            return {"logits": logits, "utility": utility, "hidden": hidden,
                    "uncertainty": None, "raw": None,
                    "inference_status": "deterministic_fallback"}
        output = response.outputs
    else:
        output = model(**inputs)
    if track in ("C1", "C2"):
        utility = output["candidate_utility"][0]
    elif track == "A1":
        utility = _gather_dense_candidates(output, table, raster.shape[-2:])[0]
    else:
        utility = _gather_graph_candidates(output, table)[0]
    valid = torch.tensor(table.valid_mask, device=utility.device, dtype=torch.bool)[None]
    logits = candidate_policy_logits(utility[None], valid, route_cost, risk_cost)
    return {"logits": logits, "utility": utility, "hidden": output["hidden"],
            "uncertainty": output.get("uncertainty"), "raw": output,
            "inference_status": "accepted" if isinstance(model, ProductionMecoAdapter) else "trainer_direct"}


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


def sample_candidate_action(logits: torch.Tensor, *, deterministic=False):
    """Sample a legal categorical choice and return action and PPO log-probability."""
    if logits.ndim != 2 or not bool(torch.isfinite(torch.logsumexp(logits, dim=-1)).all()):
        raise ValueError("each decision requires at least one finite candidate logit")
    distribution = torch.distributions.Categorical(logits=logits)
    action = logits.argmax(-1) if deterministic else distribution.sample()
    return action, distribution.log_prob(action), distribution.entropy()


@dataclass(frozen=True, slots=True)
class SafeActionCandidates:
    """Currently executable curriculum action subset, not a general planner API."""
    observation: ObservationSnapshot
    table: CandidateTable
    fallback_candidate_id: str
    action_indices: tuple[int, ...]


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
    mask = observation[world, slot, -8:].bool()
    indices, available, _ = env._candidates()
    track_indices = indices[world, slot]
    candidate_available = available[world, slot]
    points = env.track_pos[world, slot].index_select(0, track_indices)
    pose = env.sim.pose[world, slot]
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
    fallback_action = int(teacher_actions(env, observation)[world, slot].item())
    if fallback_action < 0 or fallback_action >= 8 or not table.valid_mask[fallback_action]:
        valid = [i for i, allowed in enumerate(table.valid_mask) if allowed]
        if not valid:
            raise RuntimeError("simulator exposed no legal fallback action")
        fallback_action = valid[0]
    return SafeActionCandidates(snap, table, table.candidate_ids[fallback_action], tuple(range(8)))


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
    return {"model": PREDICTION_MODEL_ID, "mode": mode,
        "horizon_s": horizon,
        "horizon_semantics": ("observed_current_pose" if mode == "current" else
            "caller_supplied_eta_or_fixed_horizon" if horizon is None else "fixed_or_caller_supplied"),
        "max_speed_mps": PREDICTION_MAX_SPEED_MPS if mode == "cv" else None,
        "max_track_age_s": PREDICTION_MAX_AGE_S if mode == "cv" else None,
        "threat_radius_m": PREDICTION_THREAT_RADIUS_M,
        "max_penalty_points": PREDICTION_MAX_PENALTY_POINTS,
        "fallback": ("none_if_observed_pose_missing_or_nonfinite" if mode == "current"
            else "current_finite_observed_pose_when_cv_unavailable; otherwise_none")}


def opponent_forecast(*, mode: str, valid, position_xy, velocity_xy,
                      age_s, horizon_s: float | None = None) -> dict:
    """Return an observed-pose or bounded CV forecast with explicit fallback."""
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
    elif pose is not None:
        if velocity_xy is not None and age_s is not None:
            forecast = predict_opponent_endpoint(valid=True, position_xy=pose,
                velocity_xy=velocity_xy, age_s=float(age_s), horizon_s=horizon,
                max_speed_mps=PREDICTION_MAX_SPEED_MPS,
                max_age_s=PREDICTION_MAX_AGE_S)
        if forecast is None:
            forecast = pose
            fallback = "observed_current_pose"
    return {"xy": forecast, "mode": mode, "horizon_s": horizon,
        "observed_age_s": (float(age_s) if age_s is not None and
                            math.isfinite(float(age_s)) else None),
        "used_cv": bool(mode == "cv" and forecast is not None and fallback is None),
        "fallback": fallback}
