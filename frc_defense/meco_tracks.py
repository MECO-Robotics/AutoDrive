"""Five MECO research scorers and a fail-closed decision adapter.

This module is deliberately separate from the production controller. Models
score goals/candidates; the caller supplies deterministic feasibility and
legality, and only the deterministic arbiter may execute an action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .learning_contract import CandidateTable, ObservationSnapshot, ScorerSnapshot, masked_argmax
from .learning_inputs import build_observation_snapshot


MECO_TRACKS = ("A1", "A2", "A3", "C1", "C2")
MECO_SCHEMA_VERSION = 1
HIDDEN = 64
SAFE_ACTION_FEATURES = (
    "action_index", "sensor_track_buffer_index", "goal_x_m", "goal_y_m",
    "heading_bin", "behavior_index", "field_length_m", "field_width_m",
    "target_distance_m", "nearby_sensed_track_count", "track_confidence",
    "track_age_s", "opponent_distance_m", "route_time_s",
)


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
        safe = edges.clamp(0, nodes.shape[1] - 1)
        expanded = nodes[:, :, None, :].expand(-1, -1, safe.shape[-1], -1)
        neighbors = expanded.gather(1, safe[..., None].expand(-1, -1, -1, nodes.shape[-1]))
        messages = self.message(neighbors)
        mask = edge_mask[..., None].to(messages.dtype)
        aggregate = (messages * mask).sum(2) / mask.sum(2).clamp_min(1.)
        return self.norm(nodes + self.update(torch.cat((nodes, aggregate), -1)))


class A2StructuredSE2UNetGRU(nn.Module):
    """A2: structured heading-aware transition net, regional attention, and GRU."""
    track_id = "A2"

    def __init__(self, region_tokens=64, heads=4):
        super().__init__()
        if not 32 <= region_tokens <= 128 or HIDDEN % heads:
            raise ValueError("A2 requires 32–128 regions and a dividing attention head count")
        self.region_tokens = region_tokens
        self.lift = nn.Sequential(nn.Linear(6 + 7 + 3, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
        self.layers = nn.ModuleList([SE2MessageBlock(), SE2MessageBlock()])
        self.region_embedding = nn.Embedding(region_tokens, HIDDEN)
        self.attention = nn.MultiheadAttention(HIDDEN, heads, batch_first=True)
        self.context = RecurrentTrack()
        self.decoder = nn.Sequential(nn.Linear(HIDDEN * 2, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 2))
        self.behavior_head = nn.Linear(HIDDEN, 5)

    def forward(self, node_features, edges, edge_mask, region_ids, globals_, hidden=None, **_):
        nodes = self.lift(torch.cat((node_features, globals_[:, None, :].expand(-1, node_features.shape[1], -1)), -1))
        for layer in self.layers:
            nodes = layer(nodes, edges, edge_mask)
        batch, count, _ = nodes.shape
        if region_ids.shape != (batch, count) or bool(((region_ids < 0) | (region_ids >= self.region_tokens)).any()):
            raise ValueError("A2 region IDs must map each node to a configured region")
        ids = region_ids
        sums = nodes.new_zeros((batch, self.region_tokens, HIDDEN))
        counts = nodes.new_zeros((batch, self.region_tokens, 1))
        sums.scatter_add_(1, ids[..., None].expand_as(nodes), nodes)
        counts.scatter_add_(1, ids[..., None], nodes.new_ones((batch, count, 1)))
        tokens = sums / counts.clamp_min(1.) + self.region_embedding.weight[None]
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        state = self.context(attended.mean(1), hidden)
        decoded = self.decoder(torch.cat((nodes, state[:, None].expand(-1, count, -1)), -1))
        return {"utility": decoded[..., 0], "penalty": F.softplus(decoded[..., 1]),
                "behavior_logits": self.behavior_head(state), "hidden": state,
                "region_tokens": tokens}


class A3HierarchicalSE2GNNGRU(nn.Module):
    """A3: fine SE(2) graph propagation with connectivity-aware region pooling."""
    track_id = "A3"

    def __init__(self, region_tokens=64, heads=4):
        super().__init__()
        if region_tokens != 64 or HIDDEN % heads:
            raise ValueError("A3 prototype fixes 64 region tokens")
        self.region_tokens = region_tokens
        self.lift = nn.Sequential(nn.Linear(16, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
        self.layers = nn.ModuleList([SE2MessageBlock() for _ in range(3)])
        self.region_attention = nn.MultiheadAttention(HIDDEN, heads, batch_first=True)
        self.context = RecurrentTrack()
        self.region_decoder = nn.Sequential(nn.Linear(HIDDEN * 2, HIDDEN), nn.SiLU())
        self.node_head = nn.Linear(HIDDEN, 2)
        self.behavior_head = nn.Linear(HIDDEN, 5)

    def forward(self, node_features, edges, edge_mask, region_ids, globals_, hidden=None, **_):
        nodes = self.lift(torch.cat((node_features, globals_[:, None].expand(-1, node_features.shape[1], -1)), -1))
        for layer in self.layers:
            nodes = layer(nodes, edges, edge_mask)
        batch, count, _ = nodes.shape
        if region_ids.shape != (batch, count) or bool(((region_ids < 0) | (region_ids >= self.region_tokens)).any()):
            raise ValueError("A3 region IDs must map each node to a configured region")
        ids = region_ids
        pooled = nodes.new_zeros((batch, self.region_tokens, HIDDEN))
        counts = nodes.new_zeros((batch, self.region_tokens, 1))
        pooled.scatter_add_(1, ids[..., None].expand_as(nodes), nodes)
        counts.scatter_add_(1, ids[..., None], nodes.new_ones((batch, count, 1)))
        pooled = pooled / counts.clamp_min(1.)
        regions, _ = self.region_attention(pooled, pooled, pooled, need_weights=False)
        state = self.context(regions.mean(1), hidden)
        contextual = self.region_decoder(torch.cat((regions, state[:, None].expand(-1, self.region_tokens, -1)), -1))
        node_context = contextual.gather(1, ids[..., None].expand(-1, -1, HIDDEN))
        outputs = self.node_head(nodes + node_context)
        return {"utility": outputs[..., 0], "penalty": F.softplus(outputs[..., 1]),
                "behavior_logits": self.behavior_head(state), "hidden": state,
                "region_tokens": regions}


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
        if candidate_features < 1 or not 32 <= tokens <= 128 or HIDDEN % heads:
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


def make_meco_track(track: str, *, candidate_features: int = 24, **kwargs) -> nn.Module:
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
    expected_model_id: str | None = None


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
        if request.expected_model_id is not None and output.model_id != request.expected_model_id:
            raise ValueError("scorer model identity mismatch")
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


def candidate_policy_logits(utility: torch.Tensor, valid_mask: torch.Tensor,
                            route_cost: torch.Tensor | None = None,
                            risk_cost: torch.Tensor | None = None) -> torch.Tensor:
    """Gross-reward utilities with deterministic cost subtraction exactly once."""
    if utility.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
        raise ValueError("utility and boolean feasibility mask shapes must match")
    if not bool(torch.isfinite(utility[valid_mask]).all()):
        raise ValueError("valid candidate utilities must be finite")
    value = utility
    if route_cost is not None:
        if route_cost.shape != value.shape or not bool(torch.isfinite(route_cost).all()):
            raise ValueError("route costs must be finite and match utility")
        value = value - route_cost
    if risk_cost is not None:
        if risk_cost.shape != value.shape or not bool(torch.isfinite(risk_cost).all()):
            raise ValueError("risk costs must be finite and match utility")
        value = value - risk_cost
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
                     risk_cost: torch.Tensor | None = None):
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
    inputs = {"raster": raster, "globals_": globals_, "candidate_features": candidate,
              "hidden": hidden}
    if track in ("A2", "A3"):
        if graph is None:
            raise ValueError(f"{track} requires a deterministic legal SE(2) graph")
        inputs.update(graph)
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
            "uncertainty": output.get("uncertainty"), "raw": output}


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
    penalty = output["penalty"]
    scores = []
    behavior_logits = output["behavior_logits"]
    for row in table.features:
        vals = (row[x_col], row[y_col], row[heading_col], row[behavior_col], row[sx_col], row[sy_col])
        if any(value is None for value in vals):
            scores.append(utility.new_tensor(float("nan")))
            continue
        x, y, heading, behavior, field_length, field_width = map(float, vals)
        if field_length <= 0. or field_width <= 0.:
            raise ValueError("candidate field dimensions must be positive")
        col = max(0, min(width - 1, int(x / field_length * width)))
        line = max(0, min(height - 1, int(y / field_width * height)))
        hbin = max(0, min(7, int(heading)))
        bidx = max(0, min(4, int(behavior)))
        scores.append(utility[0, hbin, line, col] - penalty[0, hbin, line, col] + behavior_logits[0, bidx])
    return torch.stack(scores)[None]


def _gather_graph_candidates(output, table):
    columns = _table_columns(table)
    node_col = _required_column(columns, "legal_node_index")
    behavior_col = _required_column(columns, "behavior_index")
    utility, penalty = output["utility"], output["penalty"]
    behavior_logits = output["behavior_logits"]
    scores = []
    for row in table.features:
        node, behavior = row[node_col], row[behavior_col]
        if node is None or behavior is None:
            scores.append(utility.new_tensor(float("nan")))
        else:
            idx, bidx = int(node), max(0, min(4, int(behavior)))
            if idx < 0 or idx >= utility.shape[1]:
                scores.append(utility.new_tensor(float("nan")))
            else:
                scores.append(utility[0, idx] - penalty[0, idx] + behavior_logits[0, bidx])
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
    snap = build_observation_snapshot(env, world=world, slot=slot, episode_id=episode_id)
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
