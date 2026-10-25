"""Snapshot-bound simulator candidates with route and capture-geometry estimates.

The existing eight masked actions remain the behavior primitives. K=16/32 add
stable, locally offset collection approach goals; C1/C2 can append a
snapshot-revalidated continue candidate after the K-prefix. A separate expanded
goal set is route-qualified as a bounded candidate-coverage oracle. The selected
endpoint is passed to AD* while the simulator retains action legality and
mechanisms.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import torch

from .experimental_route_evaluation import (MAX_ROUTE_QUERY_BATCH, RouteQuery,
    DeviceRouteBatch, DeviceRouteQuery, evaluate_device_routes, evaluate_routes)
from .learning_contract import CandidateTable
from .meco_pose_graph import LegalPoseGraph
from .meco_tracks import (SafeActionCandidates, build_safe_action_candidates,
    opponent_forecast, opponent_risk_penalty, prediction_contract, PREDICTION_MODES,
    SAFE_ACTION_FEATURES, EXTRA_FEATURES)


ROUTE_COST_POINTS_PER_SECOND = 0.05
CANDIDATE_BOUNDARY = {
    "version": 9,
    "capacity": [8, 16, 32],
    "source": "all currently sensed fuel tracks plus existing simulator action slots",
    "configurable_k": True,
    "k16_k32_supported": True,
    "within_candidate_pool_recall_only": False,
    "reference_pool": "all route-qualified ordinary rows plus independent 18-offset route-qualified oracle rows for every currently sensed target",
    "recall_limit": "snapshot-bounded deterministic oracle over currently sensed tracks, not a simulator-optimality oracle",
    "expanded_goals": "one selectable direct approach per sensed target in stable distance/buffer order, followed by bounded approach offsets; oracle adds 18 offsets per sensed target",
    "execution": "selected XY endpoint and sensor buffer identity flow through the paired native goal override; slot-3 native controller revalidates the sensed track each physics step and steers pickup bearing toward its current XY throughout approach; endpoint alignment uses native swept clearance checks; action mask/mechanisms remain simulator-owned",
    "target_controller_override": "restricted to the neural offense in slot 3 and a paired collect action; each physics step requires the same sensor-buffer track to remain present, within perception timeout, and within 1.0 m of snapshot XY; use current track delta for slot-3 fuel_delta and pickup target throughout approach, then derive terminal intake bearing from current track XY relative to the selected endpoint; on loss or buffer reuse, clear goal/heading override and end the macro for a fresh candidate decision",
    "target_execution_boundary": "native collector is geometric and may collect any physically contacted eligible fuel; selected sensed target is an intended route goal and terminal bearing, not an exclusive pickup constraint",
    "selectable_order": "legacy eight action rows, then direct goals for sensed targets beyond the native top-four ordered by robot distance and sensor buffer index, then legacy offsets; K8 prefix is unchanged",
    "heading_execution": "selected collection track supplies a dynamically revalidated pickup bearing throughout approach; at the endpoint native controller applies swept static/observed-robot turn-clearance checks before final alignment",
    "future_swept_dynamic_collision_check": False,
    "continue_behavior_candidate": True,
    "continue_gap": "one dynamically revalidated continue row follows the stable K-prefix; target loss, route invalidity, expiry, completion, phase incompatibility, and episode reset clear it",
    "selectable_pool": "ordinary prefix at K=8/16/32 plus at most one appended continue row; K8 IDs/order remain unchanged, while K16/K32 prioritize direct recall of sensed targets beyond top-four before approach-offset variants",
    "commitment_expiry_macro_actions": 10,
    "commitment_target_matching_tolerance_m": 1.0,
    "commitment_switch_cost_points": 0.10,
    "hysteresis": "when a commitment is active, apply the same fixed switch-cost utility penalty to every feasible non-continue row for C1/C2/D1/D2; continuation remains dynamically revalidated",
    "route_feasibility": "route and endpoint clearance plus bearing-aligned collection capture-envelope eligibility; selected-target pickup bearing is commanded through approach, with terminal alignment subject to native swept-clearance checks",
    "capture_estimate": "bearing-aligned capture envelope eligibility proxy from sensed target geometry; native controller steers toward the selected track, while physical capture remains simulator-authoritative",
    "capture_feasibility": {
        "basis": "candidate goal XY and target XY projected into inferred target-bearing frame",
        "longitudinal_min_m": "robot_length_m/2 - 0.075 - env._fuel_radius (native collector radius)",
        "longitudinal_max_m": "robot_length_m/2 + 0.35 + 0.3048 + env._fuel_radius (native collector radius)",
        "lateral_max_m": "robot_width_m/2 + 0.075 + env._fuel_radius (native collector radius)",
        "orientation": "bearing-aligned eligibility geometry; selected target drives native pickup bearing and terminal alignment remains subject to swept-clearance checks",
        "authority": "candidate eligibility proxy only; native gamepiece collision/capture logic is authoritative",
        "missing_target": "ineligible"
    },
    "plan_a_policy_support": "all reachable legal SE(2) graph nodes crossed with currently executable drive/intake/score/ferry/defense behaviors; carrying dumper ferry pairs are excluded because native release may preempt the selected endpoint",
    "plan_a_support_order": "legal_graph_node_ascending_then_behavior_0_to_4_then_descriptor_ascending_v2",
    "plan_a_support_membership": "one model-independent shortest-path pass over legal SE(2) graph plus vectorized behavior-specific endpoint/target constraints; independent of learned P",
    "plan_a_intake_descriptor": "intake pairs retain exact sensed target slot and require a bearing-aligned capture-geometry envelope",
    "plan_a_heading_semantics": "each selected legal SE(2) node carries its terminal heading; the simulator stops at the XY goal, checks sampled static rotation and observed-robot point forecast, then aligns; intake headings are capture-geometry constrained, score headings are hub-facing within 5 degrees; controllable ferry headings satisfy the native 45-degree release cone",
    "plan_a_drive_semantics": "action 7 holds unless the Plan A adapter pairs it with its explicit validated planner goal, retaining native route and obstacle guards; legal dumper traversal remains supported while loaded, while carrying dumper ferry action 6 remains excluded because it may release fuel before reaching the selected endpoint",
    "plan_a_goal_constraints": "score pairs require a safe score-grid endpoint; executable ferry pairs require carried fuel and current ferry endpoint, while dumper ferry pairs are excluded; defense pairs require the current intercept/post endpoint",
    "plan_a_behavior_heading_constraints": "intake headings must keep the exact sensed target within the forward capture envelope; score headings must face the HUB within 5 degrees; controllable ferry headings satisfy the simulator's 45-degree release cone; drive and defense headings use guarded terminal alignment",
    "plan_a_ppo": "sample and recompute log probability over the exact frozen node-major support mask and order",
    "plan_a_route_metadata": "frozen shortest-path parent tree including the root-pose connector and terminal headings; this is graph-route metadata, while execution remains XY AD* with swept SE(2) checks and guarded terminal alignment",
    "plan_a_candidate_k": "not used; candidate_k applies only to Plan C candidate scorers",
}
# The learned candidate scorers receive both these named table columns and
# one missingness bit per column. Keep the contract shared with training and
# checkpoint export rather than inferring it from a stale export fixture.
CANDIDATE_BASE_FEATURE_COUNT = len(SAFE_ACTION_FEATURES) + len(EXTRA_FEATURES)
CANDIDATE_MODEL_FEATURE_COUNT = 2 * CANDIDATE_BASE_FEATURE_COUNT


@dataclass(frozen=True)
class CandidateCommitment:
    """Task-level choice carried between snapshots and requalified each time."""
    candidate_id: str
    action_index: int
    goal_variant: int
    source_xy_m: tuple[float, float] | None
    created_sequence: int
    expires_sequence: int

    def __post_init__(self):
        if (not isinstance(self.candidate_id, str) or not self.candidate_id
                or type(self.action_index) is not int or self.action_index not in range(8)
                or type(self.goal_variant) is not int or self.goal_variant not in range(7)
                or type(self.created_sequence) is not int or self.created_sequence < 0
                or type(self.expires_sequence) is not int
                or self.expires_sequence <= self.created_sequence):
            raise ValueError("invalid Plan C commitment descriptor")
        if self.source_xy_m is not None:
            source = tuple(float(value) for value in self.source_xy_m)
            if len(source) != 2 or not all(math.isfinite(value) for value in source):
                raise ValueError("commitment source position must be a finite XY pair")
            object.__setattr__(self, "source_xy_m", source)


@dataclass(frozen=True)
class RouteFeasibleCandidates:
    observation: object
    table: CandidateTable
    fallback_candidate_id: str
    action_indices: tuple[int, ...]
    route_cost_points: torch.Tensor
    risk_cost_points: torch.Tensor
    route_records: tuple[dict, ...]
    candidate_recall: dict
    commitment: CandidateCommitment | None = None
    device_batch: "DeviceCandidateBatch | None" = None


@dataclass(frozen=True)
class DeviceCandidateBatch:
    """Device-resident numeric Plan C candidates with host-only stable IDs."""

    candidate_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    candidate_index: torch.Tensor
    action_index: torch.Tensor
    target_track_index: torch.Tensor
    target_xy: torch.Tensor
    goal_xy: torch.Tensor
    goal_heading: torch.Tensor
    route_override_flags: torch.Tensor
    raw_features: torch.Tensor
    missing_mask: torch.Tensor
    valid_mask: torch.Tensor
    eta_s: torch.Tensor
    route_cost_points: torch.Tensor
    risk_cost_points: torch.Tensor
    route_batch: DeviceRouteBatch
    fallback_index: torch.Tensor

    def __post_init__(self):
        count = len(self.candidate_ids)
        if (not count or len(set(self.candidate_ids)) != count or
                any(not isinstance(item, str) or not item for item in self.candidate_ids)):
            raise ValueError("candidate IDs must be unique nonempty host metadata")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("candidate feature names must be unique and ordered")
        device = self.valid_mask.device
        if self.candidate_index.shape != (count,) or self.candidate_index.dtype != torch.long:
            raise ValueError("candidate_index must be int64 [K]")
        if self.action_index.shape != (count,) or self.action_index.dtype != torch.long:
            raise ValueError("action_index must be int64 [K]")
        if self.target_track_index.shape != (count,) or self.target_track_index.dtype != torch.long:
            raise ValueError("target_track_index must be int64 [K]")
        for name, tensor in (("target_xy", self.target_xy), ("goal_xy", self.goal_xy)):
            if tensor.shape != (count, 2) or not tensor.is_floating_point():
                raise ValueError(f"{name} must be floating [K,2]")
        if self.goal_heading.shape != (count,) or not self.goal_heading.is_floating_point():
            raise ValueError("goal_heading must be floating [K]")
        if self.route_override_flags.shape != (count, 2) or self.route_override_flags.dtype != torch.bool:
            raise ValueError("route_override_flags must be bool [K,2]")
        feature_shape = (1, count, len(self.feature_names))
        if (self.raw_features.shape != feature_shape or
                self.missing_mask.shape != feature_shape or
                not self.raw_features.is_floating_point() or
                not self.missing_mask.is_floating_point()):
            raise ValueError("raw_features and missing_mask must be floating [1,K,F]")
        if self.valid_mask.shape != (1, count) or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [1,K]")
        for name, tensor in (("eta_s", self.eta_s),
                             ("route_cost_points", self.route_cost_points),
                             ("risk_cost_points", self.risk_cost_points)):
            if tensor.shape != (1, count) or not tensor.is_floating_point():
                raise ValueError(f"{name} must be floating [1,K]")
        if self.fallback_index.shape != () or self.fallback_index.dtype != torch.long:
            raise ValueError("fallback_index must be a scalar int64 tensor")
        tensors = (self.candidate_index, self.action_index,
            self.target_track_index, self.target_xy, self.goal_xy,
            self.goal_heading, self.route_override_flags, self.raw_features,
            self.missing_mask, self.valid_mask, self.eta_s,
            self.route_cost_points, self.risk_cost_points, self.fallback_index)
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all device candidate tensors must share one device")
        if (self.route_batch.device != device or len(self.route_batch.queries) != count or
                tuple(query.candidate_id for query in self.route_batch.queries) !=
                tuple(self.candidate_ids)):
            raise ValueError("route batch must align with device candidate order")

    @property
    def device(self):
        return self.valid_mask.device

    @property
    def candidate_features(self):
        return torch.cat((self.raw_features, self.missing_mask), dim=-1)

    def gather_action_payload(self, selected_index: torch.Tensor) -> torch.Tensor:
        """Gather the sole numeric selection payload for host simulator dispatch."""
        if selected_index.numel() != 1 or selected_index.device != self.device:
            raise ValueError("selected_index must be one device scalar")
        index = selected_index.reshape(1).to(dtype=torch.long)
        action = self.action_index[index].to(torch.float64)
        target_track = self.target_track_index[index].to(torch.float64)
        target_xy = self.target_xy[index].to(torch.float64)
        goal_xy = self.goal_xy[index].to(torch.float64)
        goal_heading = self.goal_heading[index].to(torch.float64)[:, None]
        flags = self.route_override_flags[index].to(torch.float64)
        candidate_index = self.candidate_index[index].to(torch.float64)
        return torch.cat((candidate_index[:, None], action[:, None],
            target_track[:, None], target_xy, goal_xy, goal_heading, flags), dim=-1)


def _candidate_target(env, action: int, base: SafeActionCandidates, *,
                      target_row=None):
    slot = 3
    pose = env.policy_pose[0, slot]
    xy = pose[:2].clone()
    heading = float(pose[2].item())
    valid = bool(base.table.valid_mask[action])
    intake = action < 4
    if action < 4:
        row = base.table.features[action] if target_row is None else target_row
        if row[2] is None or row[3] is None:
            return xy, heading, False, True
        fuel = torch.tensor((row[2], row[3]), device=env.device, dtype=xy.dtype)
        delta = fuel - xy
        norm = float(delta.norm().item())
        if not math.isfinite(norm) or norm <= 1e-6:
            return xy, heading, False, True
        direction = delta / norm
        target = fuel - direction * 0.45
        radius = float((.5 * torch.sqrt(env.sim.length[0, slot].square() +
                                        env.sim.width[0, slot].square())).item())
        target[0].clamp_(radius, float(env.sim.field_length) - radius)
        target[1].clamp_(radius, float(env.sim.field_width) - radius)
        heading = math.atan2(float(direction[1]), float(direction[0]))
        return target, heading, valid, intake
    if action == 4:  # score: nearest currently safe alliance-zone point
        points = env._score_grid_points[0, slot]
        clear = env._score_grid_clear[0, slot]
        if points.shape[0]:
            distance = (points - xy).norm(dim=-1).masked_fill(~clear, float("inf"))
            index = int(distance.argmin().item())
            if not bool(torch.isfinite(distance[index])):
                return xy, heading, False, False
            target = points[index].clone()
        else:
            radius = .5 * torch.sqrt(env.sim.length[0, slot].square() + env.sim.width[0, slot].square())
            if int(env.team_ids[slot]) == 0:
                low, high = float(radius), float(env.alliance_zone_depth - radius)
            else:
                low, high = float(env.sim.field_length - env.alliance_zone_depth + radius), float(env.sim.field_length - radius)
            target = xy.clone()
            target[0].clamp_(low, high)
        hub = env.hub_centers[env.team_ids[slot]]
        delta = hub - target
        heading = math.atan2(float(delta[1]), float(delta[0]))
        return target, heading, valid, False
    if action == 5:  # offense intercept is only available for a fresh sensed opponent
        if not bool(env.opponent_valid[0, slot]):
            return xy, heading, False, False
        target = env.opponent_pose[0, slot, :2].clone()
        delta = target - xy
        if float(delta.norm()) > 1e-6:
            heading = math.atan2(float(delta[1]), float(delta[0]))
        return target, heading, valid, False
    if action == 6:
        target = env._ferry_targets_for_current_motion()[0, slot].clone()
        delta = target - xy
        if float(delta.norm()) > 1e-6:
            heading = math.atan2(float(delta[1]), float(delta[0]))
        return target, heading, valid, False
    return xy, heading, valid, False


def _endpoint_clear_batch(env, xy, intake_mask):
    """Tensor-only endpoint envelope check for an aligned candidate batch."""
    if xy.ndim != 2 or xy.shape[-1] != 2 or intake_mask.shape != xy.shape[:1]:
        raise ValueError("endpoint batch requires XY [K,2] and intake mask [K]")
    slot = 3
    length = env.sim.length[0, slot] + intake_mask.to(env.sim.length.dtype) * .6096
    width = env.sim.width[0, slot].expand_as(length)
    side = torch.sqrt(length.square() + width.square())
    path = xy[:, None, :].expand(-1, 2, -1)
    static_clear = env.planner._footprint_path_clear(path,
        torch.zeros(xy.shape[0], device=xy.device, dtype=xy.dtype), side, side)
    circles = env._perceived_robot_obstacles()[slot]
    circle_mask = circles[:, 2] > 0.
    separation = torch.linalg.vector_norm(xy[:, None, :] - circles[None, :, :2], dim=-1)
    circles_clear = ((separation > side[:, None] * .5 + circles[None, :, 2]) |
                     ~circle_mask[None, :]).all(dim=-1)
    return static_clear & circles_clear


def _goal_nodes_batch(env, graph: LegalPoseGraph, xy, heading, intake_mask):
    """Resolve legal pose nodes and endpoint clearance without scalar reads."""
    device = xy.device
    hbin = torch.round(torch.remainder(heading, 2. * math.pi) *
                       (8. / (2. * math.pi))).to(torch.long).remainder(8)
    if graph.node_count == 0:
        return (torch.full_like(hbin, -1), xy, hbin,
                torch.zeros_like(hbin, dtype=torch.bool))
    finite = torch.isfinite(xy).all(dim=-1) & torch.isfinite(heading)
    in_field = ((xy[:, 0] >= 0.) & (xy[:, 0] < float(env.sim.field_length)) &
                (xy[:, 1] >= 0.) & (xy[:, 1] < float(env.sim.field_width)))
    ix = torch.floor(xy[:, 0] / graph.resolution_m).to(torch.long)
    iy = torch.floor(xy[:, 1] / graph.resolution_m).to(torch.long)
    cell_valid = (ix >= 0) & (ix < graph.nx) & (iy >= 0) & (iy < graph.ny)
    safe_ix, safe_iy = ix.clamp(0, graph.nx - 1), iy.clamp(0, graph.ny - 1)
    lookup = graph.lookup.to(device=device)
    requested = lookup[hbin, safe_ix, safe_iy]
    all_headings = lookup[:, safe_ix, safe_iy].transpose(0, 1)
    heading_ids = torch.arange(8, device=device).expand_as(all_headings)
    fallback_heading = torch.where(all_headings >= 0, heading_ids, 8).amin(dim=-1)
    fallback = lookup[fallback_heading.clamp_max(7), safe_ix, safe_iy]
    node = torch.where(requested >= 0, requested, fallback)
    valid = finite & in_field & cell_valid & (node >= 0)
    safe_node = node.clamp_min(0)
    center = graph.xy.to(device=device)[safe_node]
    close = torch.linalg.vector_norm(center - xy, dim=-1) <= graph.resolution_m * .72
    clear = _endpoint_clear_batch(env, center, intake_mask)
    valid &= close & clear
    selected_hbin = graph.heading_bin.to(device=device)[safe_node].to(torch.long)
    selected_hbin = torch.where(valid, selected_hbin, hbin)
    node = torch.where(valid, node, torch.full_like(node, -1))
    center = torch.where(valid[:, None], center, xy)
    return node, center, selected_hbin, valid


def _candidate_geometry_device(env, graph, specs, fuel_xy, fuel_available,
                               base_valid):
    """Build route goals/capture masks as tensors after the host spec list."""
    device = torch.device(env.device)
    dtype = env.sim.pose.dtype
    count = len(specs)
    if fuel_xy.shape != (count, 2) or fuel_available.shape != (count,):
        raise ValueError("candidate fuel tensors must align with the ordered spec list")
    actions = torch.tensor([row[0] for row in specs], device=device, dtype=torch.long)
    variants = torch.tensor([row[1] for row in specs], device=device, dtype=torch.long)
    forward_offsets = torch.tensor([row[2] for row in specs], device=device, dtype=dtype)
    lateral_offsets = torch.tensor([row[3] for row in specs], device=device, dtype=dtype)
    pose = env.sim.pose[0, 3]
    pose_xy = pose[:2].expand(count, -1)
    pose_heading = pose[2].to(torch.float64).expand(count)
    collect = actions < 4
    allowed = base_valid[actions]
    fuel_delta = fuel_xy - pose_xy
    fuel_distance = torch.linalg.vector_norm(fuel_delta, dim=-1)
    fuel_ok = (fuel_available & torch.isfinite(fuel_xy).all(dim=-1) &
               torch.isfinite(fuel_distance) & (fuel_distance > 1e-6))
    direction = fuel_delta / fuel_distance.clamp_min(1e-12)[:, None]
    radius = .5 * torch.sqrt(env.sim.length[0, 3].square() +
                             env.sim.width[0, 3].square())
    fuel_target = fuel_xy - direction * .45
    fuel_target = torch.stack((
        fuel_target[:, 0].clamp(radius, float(env.sim.field_length) - radius),
        fuel_target[:, 1].clamp(radius, float(env.sim.field_width) - radius)), dim=-1)
    targets = torch.where(collect[:, None] & fuel_ok[:, None], fuel_target, pose_xy)
    fuel_heading = torch.atan2(direction[:, 1].to(torch.float64),
                               direction[:, 0].to(torch.float64))
    headings = torch.where(collect & fuel_ok, fuel_heading, pose_heading)
    allowed &= ~collect | fuel_ok

    score_mask = actions == 4
    score_points = env._score_grid_points[0, 3]
    score_clear = env._score_grid_clear[0, 3]
    if score_points.shape[0]:
        score_dist = torch.linalg.vector_norm(score_points - pose[:2], dim=-1)
        score_dist = score_dist.masked_fill(~score_clear, float("inf"))
        score_target = score_points[score_dist.argmin()]
        score_ok = torch.isfinite(score_dist).any()
    else:
        team_zero = env.team_ids[3] == 0
        low = torch.where(team_zero, radius,
            env.sim.field_length - env.alliance_zone_depth + radius)
        high = torch.where(team_zero, env.alliance_zone_depth - radius,
            env.sim.field_length - radius)
        score_target = pose[:2].clone()
        score_target[0].clamp_(low, high)
        score_ok = torch.ones((), device=device, dtype=torch.bool)
    hub = env.hub_centers[env.team_ids[3]]
    score_delta = hub - score_target
    score_heading = torch.atan2(score_delta[1].to(torch.float64),
                                score_delta[0].to(torch.float64))
    targets = torch.where(score_mask[:, None], score_target.expand(count, -1), targets)
    headings = torch.where(score_mask, score_heading.expand(count), headings)
    allowed &= ~score_mask | score_ok

    intercept_mask = actions == 5
    opponent_xy = env.opponent_pose[0, 3, :2]
    opponent_ok = env.opponent_valid[0, 3] & torch.isfinite(opponent_xy).all()
    intercept_delta = opponent_xy - pose[:2]
    intercept_heading = torch.atan2(intercept_delta[1].to(torch.float64),
                                    intercept_delta[0].to(torch.float64))
    targets = torch.where(intercept_mask[:, None], opponent_xy.expand(count, -1), targets)
    headings = torch.where(intercept_mask &
        (torch.linalg.vector_norm(intercept_delta) > 1e-6),
        intercept_heading.expand(count), headings)
    allowed &= ~intercept_mask | opponent_ok

    ferry_mask = actions == 6
    ferry_target = env._ferry_targets_for_current_motion()[0, 3]
    ferry_delta = ferry_target - pose[:2]
    ferry_heading = torch.atan2(ferry_delta[1].to(torch.float64),
                                ferry_delta[0].to(torch.float64))
    targets = torch.where(ferry_mask[:, None], ferry_target.expand(count, -1), targets)
    headings = torch.where(ferry_mask &
        (torch.linalg.vector_norm(ferry_delta) > 1e-6),
        ferry_heading.expand(count), headings)

    offsets = variants != 0
    forward = torch.stack((headings.cos(), headings.sin()), dim=-1).to(dtype)
    left = torch.stack((-headings.sin(), headings.cos()), dim=-1).to(dtype)
    offset_xy = forward * forward_offsets[:, None] + left * lateral_offsets[:, None]
    targets = targets + torch.where(offsets[:, None], offset_xy,
                                    torch.zeros_like(offset_xy))
    # Collection bearing is recomputed against the offset target, matching the
    # existing candidate semantics before legal-node snapping.
    target_delta = fuel_xy - targets
    target_distance = torch.linalg.vector_norm(target_delta, dim=-1)
    target_heading = torch.atan2(target_delta[:, 1].to(torch.float64),
                                 target_delta[:, 0].to(torch.float64))
    headings = torch.where(collect & (target_distance > 1e-8), target_heading, headings)
    nodes, centers, hbins, node_ok = _goal_nodes_batch(
        env, graph, targets, headings, collect)
    allowed &= node_ok

    snapped_delta = fuel_xy - centers
    snapped_distance = torch.linalg.vector_norm(snapped_delta, dim=-1)
    snapped_heading = torch.atan2(snapped_delta[:, 1].to(torch.float64),
                                  snapped_delta[:, 0].to(torch.float64))
    recheck = collect & node_ok
    headings = torch.where(recheck & (snapped_distance > 1e-8), snapped_heading, headings)
    nodes2, centers2, hbins2, valid2 = _goal_nodes_batch(
        env, graph, centers, headings, collect)
    nodes = torch.where(recheck, nodes2, nodes)
    centers = torch.where(recheck[:, None], centers2, centers)
    hbins = torch.where(recheck, hbins2, hbins)
    allowed &= ~recheck | valid2

    capture_delta = fuel_xy - centers
    safe_heading = torch.where(torch.isfinite(headings), headings,
                               torch.zeros_like(headings))
    forward = torch.stack((safe_heading.cos(), safe_heading.sin()), dim=-1).to(dtype)
    left = torch.stack((-safe_heading.sin(), safe_heading.cos()), dim=-1).to(dtype)
    longitudinal = (capture_delta * forward).sum(-1)
    lateral = (capture_delta * left).sum(-1).abs()
    fuel_radius = torch.as_tensor(env._fuel_radius, device=device, dtype=dtype)
    half_length = env.sim.length[0, 3] * .5
    half_width = env.sim.width[0, 3] * .5
    capture = (torch.isfinite(capture_delta).all(-1) &
        (longitudinal >= half_length - .075 - fuel_radius) &
        (longitudinal <= half_length + .35 + .3048 + fuel_radius) &
        (lateral <= half_width + .075 + fuel_radius))
    allowed &= ~collect | capture
    return {"actions": actions, "collect": collect, "fuel_xy": fuel_xy,
        "fuel_available": fuel_available, "targets": centers, "headings": headings,
        "nodes": nodes, "heading_bins": hbins, "allowed": allowed,
        "capture_distance": torch.linalg.vector_norm(capture_delta, dim=-1),
        "capture_longitudinal": longitudinal, "capture_lateral": lateral,
        "capture_inside": capture}


def _endpoint_clear(env, xy, heading: float, *, intake_extended: bool):
    slot = 3
    length = env.sim.length[0, slot].clone()
    if intake_extended:
        length = length + .6096
    width = env.sim.width[0, slot].clone()
    # The selected heading is only a planner feature. Enclose the full robot
    # (including intake) in a square so endpoint clearance holds at every yaw.
    side = torch.sqrt(length.square() + width.square())
    point = xy.reshape(1, 1, 2).expand(1, 2, 2)
    static_ok = bool(env.planner._footprint_path_clear(point,
        torch.zeros(1, device=env.device, dtype=point.dtype),
        side.reshape(1), side.reshape(1))[0].item())
    circles = env._perceived_robot_obstacles()[slot]
    circles = circles[circles[:, 2] > 0]
    if circles.numel() == 0:
        return static_ok
    delta = circles[:, :2] - xy
    clearance_radius = float(side) * .5
    circles_clear = bool((delta.norm(dim=-1) > clearance_radius + circles[:, 2]).all().item())
    return static_ok and circles_clear


def _goal_node(env, graph: LegalPoseGraph, xy, heading: float, *, intake_extended=False):
    if not (0. <= float(xy[0]) < float(env.sim.field_length)
            and 0. <= float(xy[1]) < float(env.sim.field_width)):
        return None, None, 0
    angle = heading % (2 * math.pi)
    hbin = int(round(angle * 8 / (2 * math.pi))) % 8
    # Bind the requested target to its nearest planner-cell center. A blocked
    # cell is rejected; no nearest-free projection is permitted here.
    ix = int(torch.floor(xy[0] / graph.resolution_m).item())
    iy = int(torch.floor(xy[1] / graph.resolution_m).item())
    if not (0 <= ix < graph.nx and 0 <= iy < graph.ny):
        return None, None, hbin
    node = int(graph.lookup[hbin, ix, iy].item())
    if node < 0:
        # The selected sensed target drives native pickup bearing during
        # execution. This graph heading qualifies the candidate's capture
        # geometry; bind lookup to the first legal heading at this cell when
        # that inferred heading is blocked.
        legal_nodes = graph.lookup[:, ix, iy]
        legal_nodes = legal_nodes[legal_nodes >= 0]
        if legal_nodes.numel() == 0:
            return None, None, hbin
        node = int(legal_nodes[0].item())
    center = graph.xy[node]
    if float((center - xy).norm().item()) > graph.resolution_m * .72:
        return None, None, hbin
    if not _endpoint_clear(env, center, float(hbin) * 2 * math.pi / 8,
                           intake_extended=intake_extended):
        return None, None, hbin
    # A2/A3 consume the legal node's heading as part of its SE(2) feature.
    # If the sensed target bearing had to fall back to another legal heading,
    # keep the candidate feature aligned with the actual node identity.
    selected_hbin = int(graph.heading_bin[node].item())
    return node, center, selected_hbin


def _capture_pose_estimate(env, base: SafeActionCandidates, action: int,
                           goal_xy, goal_heading: float | None = None, *,
                           target_row=None) -> dict:
    """Qualify sensed-target geometry in the candidate's inferred heading frame.

    This is a conservative candidate eligibility proxy, not a command to the
    native controller. Simulator capture and mechanism behavior remain the
    final authority.
    """
    if action >= 4:
        return {"status": "not_applicable", "basis": "non_collection_action",
                "feasible": True}
    row = base.table.features[action] if target_row is None else target_row
    if row[2] is None or row[3] is None:
        return {"status": "unavailable", "basis": "target_not_sensed",
                "feasible": False, "reason": "target_not_sensed"}
    slot = 3
    fuel = goal_xy.new_tensor((row[2], row[3]))
    delta = fuel - goal_xy
    distance = float(delta.norm().item())
    if not math.isfinite(distance):
        return {"status": "unavailable", "basis": "nonfinite_target_geometry",
                "feasible": False, "reason": "nonfinite_target_geometry"}
    # Match the native collector predicates in gamepiece_actions.py exactly;
    # fuel_physics.config.radius can differ from the collector's effective
    # radius and would incorrectly reject otherwise capturable candidates.
    radius = float(env._fuel_radius)
    half_length = float(env.sim.length[0, slot]) * .5
    half_width = float(env.sim.width[0, slot]) * .5
    forward_min = half_length - .075 - radius
    forward_limit = (half_length + .35 + .3048 + radius)
    lateral_limit = half_width + .075 + radius
    heading = (math.atan2(float(delta[1]), float(delta[0]))
               if goal_heading is None and distance > 1e-12 else
               (0. if goal_heading is None else float(goal_heading)))
    forward = goal_xy.new_tensor((math.cos(heading), math.sin(heading)))
    left = goal_xy.new_tensor((-math.sin(heading), math.cos(heading)))
    longitudinal = float(torch.dot(delta, forward).item())
    lateral = abs(float(torch.dot(delta, left).item()))
    inside = (forward_min <= longitudinal <= forward_limit
              and lateral <= lateral_limit)
    reason = None
    if longitudinal < forward_min:
        reason = "target_behind_or_inside_minimum_capture_distance"
    elif longitudinal > forward_limit:
        reason = "target_beyond_forward_capture_limit"
    elif lateral > lateral_limit:
        reason = "target_outside_lateral_capture_limit"
    return {"status": "inside_estimate" if inside else "outside_estimate",
            "basis": "sensed_target_bearing_geometry_only",
            "feasible": bool(inside), "reason": reason,
            "distance_m": distance,
            "estimated_longitudinal_m": longitudinal,
            "estimated_lateral_m": lateral,
            "estimated_min_forward_m": forward_min,
            "estimated_forward_limit_m": forward_limit,
            "estimated_lateral_limit_m": lateral_limit}


def _opponent_prediction(env, slot=3, horizon=0., mode="cv"):
    valid = bool(env.opponent_valid[0, slot])
    age = float(env.opponent_age[0, slot])
    position = env.opponent_pose[0, slot, :2].detach().cpu().tolist()
    velocity = env.opponent_velocity[0, slot, :2].detach().cpu().tolist()
    angular_velocity = (float(env.opponent_velocity[0, slot, 2])
                        if env.opponent_velocity.shape[-1] > 2 else None)
    return opponent_forecast(mode=mode, valid=valid, position_xy=position,
        velocity_xy=velocity, age_s=age, horizon_s=max(0., float(horizon)),
        angular_velocity_rad_s=angular_velocity)


def _world_zero_perceived_fuel(env):
    """Read slot-3 tracks through the batched native accessor for one world."""
    if env.n != 1:
        raise ValueError("Plan C perceived-target candidates require one world")
    points, velocities, sensed = env._perceived_fuel(3)
    if (not isinstance(points, torch.Tensor)
            or not isinstance(velocities, torch.Tensor)
            or not isinstance(sensed, torch.Tensor)
            or points.ndim != 3 or points.shape[0] != env.n
            or points.shape[-1] != 2 or velocities.shape != points.shape
            or sensed.shape != points.shape[:2]):
        raise ValueError("slot-3 perceived FUEL buffers have misaligned batched shapes")
    # Keep the native accessor's [world, track, ...] contract intact, then
    # select the sole supported world explicitly for candidate construction.
    return points[0], velocities[0], sensed[0]


def _additional_sensed_target_rows(env, base: SafeActionCandidates):
    """Return candidate rows for sensed tracks omitted by the native top four.

    The action id stays a native collect class; the goal endpoint and the
    sensor buffer identity live in this candidate row. Native intake remains
    geometric and authoritative, so this does not turn a sensor track into an
    exclusive piece reservation.
    """
    base_track_ids = {int(base.table.features[i][1]) for i in range(4)
                      if base.table.valid_mask[i] and
                      base.table.features[i][1] is not None}
    valid_collect_actions = [i for i in range(4) if base.table.valid_mask[i]]
    if not valid_collect_actions:
        return []
    fallback_action = valid_collect_actions[0]
    points, _, sensed = _world_zero_perceived_fuel(env)
    sensed_ids = torch.nonzero(sensed, as_tuple=False).flatten().tolist()
    pose_xy = env.policy_pose[0, 3, :2]
    confidence = env.track_confidence[0, 3]
    ages = env.track_age[0, 3]
    opponent_distance = None
    if bool(env.opponent_valid[0, 3]):
        opponent_distance = (points - env.opponent_pose[0, 3, :2]).norm(dim=-1)
    distances = (points - pose_xy).norm(dim=-1)
    ordered = sorted((int(i) for i in sensed_ids if int(i) not in base_track_ids),
                     key=lambda i: (float(distances[i]), i))
    nearby = (((points[:, None, :] - points[None, :, :]).norm(dim=-1) < 1.25)
              & sensed[None, :]).sum(-1)
    field_length, field_width = float(env.sim.field_length), float(env.sim.field_width)
    template = base.table.features[fallback_action]
    result = []
    for track in ordered:
        target = points[track]
        delta = target - pose_xy
        angle = torch.atan2(delta[1], delta[0])
        heading_bin = int(torch.round(torch.remainder(angle, 2. * math.pi) *
                                      (8. / (2. * math.pi))).item()) % 8
        row = list(template)
        row[0] = float(fallback_action)
        row[1] = float(track)
        row[2], row[3] = float(target[0]), float(target[1])
        row[4] = float(heading_bin)
        row[6], row[7] = field_length, field_width
        row[8] = float(distances[track])
        row[9] = float(nearby[track])
        row[10] = float(confidence[track].clamp(0., 1.))
        row[11] = float(ages[track].clamp_min(0.))
        row[12] = (None if opponent_distance is None else
                   float(opponent_distance[track]))
        row[13] = None
        result.append((fallback_action, tuple(row), track))
    return result


@torch.no_grad()
def build_route_feasible_candidates(env, observation: torch.Tensor,
                                    graph: LegalPoseGraph, *, episode_id: str,
                                    action_ticks: int = 12,
                                    candidate_k: int = 8,
                                    prediction_mode: str = "cv",
                                    learned_traversal_cost=None,
                                    learned_cost_scale: float = 1.0,
                                    commitment: CandidateCommitment | None = None) -> RouteFeasibleCandidates:
    """Build the current bounded decision table and qualify every route row.

    Route feasibility uses a fresh uncached TensorADStar workspace over current
    perceived robot obstacles. The inferred target-bearing heading selects a
    legal graph node and contributes an estimate to candidate eligibility;
    endpoint footprint clearance is checked at every yaw. The paired native
    executor steers pickup bearing toward the selected sensed track and guards
    terminal alignment. ETA is distance/speed plus heading-change/turn-rate.
    Dynamic interaction is a point-prediction risk cost; full time-varying
    swept-robot planning remains out of scope and is recorded in provenance.
    """
    if env.n != 1 or not isinstance(episode_id, str) or not episode_id:
        raise ValueError("candidate source requires one world and a unique episode_id")
    if type(action_ticks) is not int or action_ticks < 1:
        raise ValueError("action_ticks must be a positive integer")
    if type(candidate_k) is not int or candidate_k not in (8, 16, 32):
        raise ValueError("candidate_k must be one of 8, 16, or 32")
    if prediction_mode not in PREDICTION_MODES:
        raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
    base = build_safe_action_candidates(env, observation, world=0, slot=3,
                                        episode_id=episode_id)
    sequence = int(env.steps[0].item())
    if base.observation.sequence != sequence:
        raise ValueError("candidate snapshot does not match current simulator step")
    snapshot = replace(base.observation, episode_id=episode_id,
                       snapshot_id=f"{episode_id}:{sequence}")
    base = SafeActionCandidates(snapshot,
        CandidateTable(snapshot.snapshot_id, base.table.candidate_ids,
            base.table.features, base.table.valid_mask, base.table.feature_names,
            route_costs=base.table.route_costs),
        base.fallback_candidate_id, base.action_indices)
    target_rows = {int(base.table.features[action][1]): base.table.features[action]
                   for action in range(4) if base.table.valid_mask[action] and
                   base.table.features[action][1] is not None}
    specs = [(action, 0, 0., 0.,
              (int(base.table.features[action][1])
               if action < 4 and base.table.features[action][1] is not None else None))
             for action in range(8)]
    additional_targets = _additional_sensed_target_rows(env, base)
    for action, row, track in additional_targets:
        target_rows[track] = row
        specs.append((action, 0, 0., 0., track))
    offsets = ((.22, 0.), (-.22, 0.), (0., .22), (0., -.22),
               (.34, 0.), (-.34, 0.))
    for variant, (forward, lateral) in enumerate(offsets, 1):
        for action in range(4):
            track = (int(base.table.features[action][1])
                     if base.table.features[action][1] is not None else None)
            specs.append((action, variant, forward, lateral, track))
    # Reconcile a carried task against this snapshot before adding it. Intake
    # commitments follow the same sensed target by position, not a transient
    # top-four buffer index. Any lost, illegal, or expired commitment is dropped.
    active_commitment = commitment
    continue_spec = None
    if active_commitment is not None:
        action = active_commitment.action_index
        if sequence >= active_commitment.expires_sequence:
            active_commitment = None
        elif action < 4:
            source = active_commitment.source_xy_m
            matches = []
            if source is not None:
                points, _, sensed = _world_zero_perceived_fuel(env)
                for track in torch.nonzero(sensed, as_tuple=False).flatten().tolist():
                    point = points[track]
                    gap = math.hypot(float(point[0]) - source[0], float(point[1]) - source[1])
                    matches.append((gap, int(track)))
            if not matches or min(matches)[0] > CANDIDATE_BOUNDARY["commitment_target_matching_tolerance_m"]:
                active_commitment = None
            else:
                track = min(matches)[1]
                action = next((i for i in range(4) if base.table.valid_mask[i]), None)
                if action is None:
                    active_commitment = None
                else:
                    continue_spec = (action, active_commitment.goal_variant, 0., 0., track)
        else:
            if not base.table.valid_mask[action]:
                active_commitment = None
            else:
                continue_spec = (action, active_commitment.goal_variant, 0., 0., None)
    continue_index = None
    if continue_spec is not None:
        continue_index = len(specs)
        specs.append(continue_spec)
    oracle_offsets = ((.50, 0.), (-.50, 0.), (0., .50), (0., -.50),
        (.50, .50), (.50, -.50), (-.50, .50), (-.50, -.50),
        (.68, 0.), (-.68, 0.), (0., .68), (0., -.68),
        (.68, .25), (.68, -.25), (-.68, .25), (-.68, -.25),
        (.25, .68), (-.25, .68))
    oracle_start_index = len(specs)
    oracle_track_ids = sorted(target_rows)
    for variant, (forward, lateral) in enumerate(oracle_offsets, 7):
        for track in oracle_track_ids:
            action = next((i for i in range(4) if base.table.valid_mask[i]), None)
            if action is not None:
                specs.append((action, variant, forward, lateral, track))
    targets, headings, route_queries, goal_nodes = [], [], [], []
    capture_estimates = []
    hbins, requested, ids = [], [], []
    for row_index, (action, variant, forward_offset, lateral_offset,
                    target_track) in enumerate(specs):
        candidate_row = target_rows.get(target_track) if target_track is not None else None
        target, heading, allowed, intake = _candidate_target(
            env, action, base, target_row=candidate_row)
        is_continue = continue_index is not None and row_index == continue_index
        if is_continue:
            candidate_id = "continue:" + active_commitment.candidate_id
        elif target_track is not None:
            base_tracks = {int(base.table.features[i][1]) for i in range(4)
                           if base.table.features[i][1] is not None}
            candidate_id = (base.table.candidate_ids[action] if not variant and
                            target_track in base_tracks else
                            f"collect:track:{target_track}" if not variant else
                            f"collect:track:{target_track}:approach:{variant}")
        else:
            candidate_id = (base.table.candidate_ids[action] if not variant else
                            f"{base.table.candidate_ids[action]}:approach:{variant}")
        ids.append(candidate_id)
        if variant:
            forward = target.new_tensor((math.cos(heading), math.sin(heading)))
            left = target.new_tensor((-math.sin(heading), math.cos(heading)))
            target = target + forward * forward_offset + left * lateral_offset
        if action < 4:
            fuel_row = candidate_row or base.table.features[action]
            if fuel_row[2] is not None and fuel_row[3] is not None:
                delta = target.new_tensor((fuel_row[2], fuel_row[3])) - target
                if float(delta.norm()) > 1e-8:
                    heading = math.atan2(float(delta[1]), float(delta[0]))
        node, center, hbin = _goal_node(env, graph, target, heading,
                                        intake_extended=intake)
        allowed &= node is not None
        if center is not None:
            target = center
            if action < 4:
                fuel_row = candidate_row or base.table.features[action]
                if fuel_row[2] is not None and fuel_row[3] is not None:
                    delta = target.new_tensor((fuel_row[2], fuel_row[3])) - target
                    if float(delta.norm()) > 1e-8:
                        heading = math.atan2(float(delta[1]), float(delta[0]))
                    # Resolve the legal heading bin again from the actual
                    # target bearing at the snapped endpoint.
                    node, center, hbin = _goal_node(env, graph, target, heading,
                                                    intake_extended=intake)
                    allowed &= node is not None
                    if center is not None:
                        target = center
        capture_estimate = _capture_pose_estimate(env, base, action, target,
            goal_heading=heading, target_row=candidate_row)
        if action < 4:
            allowed &= bool(capture_estimate["feasible"])
        capture_estimates.append(capture_estimate)
        targets.append(target)
        headings.append(heading)
        goal_nodes.append(node)
        hbins.append(hbin)
        requested.append(allowed)
        if allowed:
            route_queries.append(RouteQuery(candidate_id=candidate_id,
                snapshot_id=base.observation.snapshot_id, world=0, slot=3,
                goal_xy=(float(center[0]), float(center[1])), intake_extended=intake))

    # Validate the complete candidate set before probing any routes. The
    # evaluator intentionally caps each native AD* workspace at 64 rows; split
    # only after global snapshot and duplicate checks so chunk boundaries cannot
    # hide a cross-chunk contract violation.
    if route_queries:
        snapshot_ids = {query.snapshot_id for query in route_queries}
        if len(snapshot_ids) != 1:
            raise ValueError("one route batch must use a single snapshot_id")
        query_keys = [(query.world, query.slot, query.candidate_id)
                      for query in route_queries]
        if len(set(query_keys)) != len(query_keys):
            raise ValueError("route batch contains duplicate candidate keys")

    route_results_list = []
    for offset in range(0, len(route_queries), MAX_ROUTE_QUERY_BATCH):
        route_results_list.extend(evaluate_routes(
            env, route_queries[offset:offset + MAX_ROUTE_QUERY_BATCH],
            goal_tolerance_m=graph.resolution_m * .72,
            endpoint_tolerance_m=graph.resolution_m * .72,
            learned_traversal_cost=learned_traversal_cost,
            learned_cost_scale=learned_cost_scale,
            learned_cost_provenance="meco_legal_heading_penalty_projection_v1"))
    route_results = tuple(route_results_list)
    result_by_id = {result.candidate_id: result for result in route_results}
    speed = max(float(env.sim.speed[0, 3]), .1)
    omega = max(float(env.sim.omega_limit[0, 3]), .1)
    pose_heading = float(env.policy_pose[0, 3, 2])
    rows = []
    valid_mask = []
    routes = []
    route_costs = []
    risk_costs = []
    for row_index, (action, variant, _, _, target_track) in enumerate(specs):
        old = (target_rows[target_track] if target_track is not None
               else base.table.features[action])
        candidate_id = ids[row_index]
        result = result_by_id.get(candidate_id)
        distance = result.distance_m if result is not None and result.feasible else None
        route_ok = bool(requested[row_index] and result is not None and result.feasible and
                        goal_nodes[row_index] is not None)
        turn = abs(math.atan2(math.sin(headings[row_index] - pose_heading),
                              math.cos(headings[row_index] - pose_heading)))
        eta = (None if distance is None else distance / speed + turn / omega
               + action_ticks * float(env.dt))
        if eta is None:
            # An unreachable route has no arrival-time forecast. Keep the
            # missing forecast explicit instead of silently treating it as a
            # zero-second arrival at the current opponent pose.
            forecast = {"xy": None, "mode": prediction_mode,
                "horizon_s": 0. if prediction_mode == "current" else None,
                "observed_age_s": (float(env.opponent_age[0, 3])
                    if math.isfinite(float(env.opponent_age[0, 3])) else None),
                "used_cv": False, "fallback": "not_applicable_unreachable_route"}
        else:
            forecast = _opponent_prediction(env, horizon=eta, mode=prediction_mode)
        predicted = forecast["xy"]
        goal_xy = targets[row_index]
        separation = None if predicted is None else math.hypot(
            float(goal_xy[0]) - predicted[0], float(goal_xy[1]) - predicted[1])
        risk = opponent_risk_penalty(goal_xy, forecast)
        valid = bool(base.table.valid_mask[action] and route_ok)
        valid_mask.append(valid)
        # Update the shared target to the legal route endpoint, and preserve
        # every missing estimate explicitly for the model's missingness mask.
        row = list(old)
        row[2] = float(goal_xy[0])
        row[3] = float(goal_xy[1])
        row[4] = float(hbins[row_index])
        # route_time_s is already part of SAFE_ACTION_FEATURES; replace the
        # coarse base estimate with the route-derived ETA instead of appending
        # a duplicate column name.
        row[13] = eta
        row.extend((None if goal_nodes[row_index] is None else float(goal_nodes[row_index]),
                    distance, risk, float(headings[row_index]), float(is_continue),
                    float(active_commitment is not None and not is_continue)))
        rows.append(tuple(row))
        route_costs.append(0. if eta is None else eta * ROUTE_COST_POINTS_PER_SECOND)
        risk_costs.append(0. if risk is None else risk)
        capture_estimate = capture_estimates[row_index]
        routes.append({"candidate_id": candidate_id,
            "action_index": action,
            "candidate_row": row_index,
            "is_continue_candidate": bool(is_continue),
            "commitment_id": (active_commitment.candidate_id
                              if active_commitment is not None else None),
            "goal_variant": variant,
            "route_status": "not_queried" if result is None else result.status,
            "feasible": valid, "route_feasible": route_ok,
            "capture_estimate": capture_estimate,
            "goal_xy_m": [float(goal_xy[0]), float(goal_xy[1])],
            "source_fuel_xy_m": ([float(old[2]), float(old[3])]
                if action < 4 and old[2] is not None and old[3] is not None else None),
            "sensor_track_buffer_index": (target_track if action < 4 else None),
            "target_identity_execution": ("slot3_revalidated_track_terminal_bearing_with_native_geometric_capture"
                if target_track is not None and action < 4 else "native_action_target"),
            "target_track_snapshot_tolerance_m": (1.0 if target_track is not None
                and action < 4 else None),
            "goal_heading_rad": float(headings[row_index]),
            "legal_node_index": goal_nodes[row_index], "distance_m": distance,
            "eta_s": eta, "predicted_opponent_xy_m": predicted,
            "opponent_prediction": {**forecast,
            "contract": prediction_contract(prediction_mode,
                    horizon_s=(eta if eta is not None else None))},
            "endpoint_separation_m": separation, "risk_penalty_points": risk,
            **({"expected_risk_semantics": "weighted_expectation_over_prediction_hypotheses"}
               if prediction_mode == "multi_model_v1" else {}),
            "intake_extended": action < 4,
            "path_points_m": (None if result is None else result.path_points_m),
            "weighted_route_cost_grid_edges": (None if result is None
                else result.weighted_cost),
            "route_cost_units": (None if result is None else result.cost_units),
            "route_cost_provenance": (None if result is None
                else result.cost_provenance)})
    if continue_index is not None and not valid_mask[continue_index]:
        active_commitment = None
        for index, row in enumerate(rows):
            if index != continue_index:
                updated = list(row)
                updated[-1] = 0.
                rows[index] = tuple(updated)
    selectable_indices = list(range(candidate_k))
    if continue_index is not None and valid_mask[continue_index]:
        selectable_indices.append(continue_index)
    if not any(valid_mask[index] for index in selectable_indices):
        raise RuntimeError("candidate service found no feasible row, including safe fallback")
    fallback = base.fallback_candidate_id
    if fallback not in ids or ids.index(fallback) not in selectable_indices or not valid_mask[ids.index(fallback)]:
        hold = ids.index("hold")
        if valid_mask[hold]:
            fallback = "hold"
        else:
            fallback = min((i for i in selectable_indices if valid_mask[i]),
                           key=lambda i: route_costs[i] + risk_costs[i])
            fallback = ids[fallback]
    names = base.table.feature_names + EXTRA_FEATURES
    # Higher-budget reference: every ordinary candidate plus 18 separately
    # route-qualified offsets for every sensed target. Oracle rows never enter
    # model order and never affect the K8 baseline.
    oracle_ids = {ids[i] for i in range(oracle_start_index)
                  if i != continue_index and valid_mask[i]}
    oracle_ids.update(ids[i] for i in range(oracle_start_index, len(ids)) if valid_mask[i])
    oracle_ids.discard(ids[continue_index] if continue_index is not None else "")
    oracle_capacity = (8 + len(additional_targets) + 4 * len(offsets) +
                       len(oracle_track_ids) * len(oracle_offsets))
    oracle_count = len(oracle_ids)
    requested_ids = {ids[i] for i in range(candidate_k) if valid_mask[i]}
    oracle_indices = [i for i in range(oracle_start_index)
                      if i != continue_index and valid_mask[i]] + [
        i for i in range(oracle_start_index, len(ids)) if valid_mask[i]]
    oracle_cost_values = [route_costs[i] for i in oracle_indices]
    selected_cost_values = [route_costs[i] for i in range(candidate_k) if valid_mask[i]]
    recall = {"candidate_capacity": candidate_k,
              "prediction_mode": prediction_mode,
              "reference_pool_capacity": oracle_capacity,
              "valid_executable_candidates": len(requested_ids),
              "reference_scope": "independent_expanded_goal_oracle",
              "within_candidate_pool_recall_measured": False,
              "route_feasible_rows_in_reference_pool": oracle_count,
              "oracle_route_evaluated": True,
              "candidate_recall": (len(requested_ids & oracle_ids) / oracle_count if oracle_count else 1.),
              "oracle_min_route_cost_points": min(oracle_cost_values) if oracle_cost_values else None,
              "oracle_mean_route_cost_points": (sum(oracle_cost_values) / len(oracle_cost_values)
                                                if oracle_cost_values else None),
              "selected_min_route_cost_points": min(selected_cost_values) if selected_cost_values else None,
              "selected_mean_route_cost_points": (sum(selected_cost_values) / len(selected_cost_values)
                                                  if selected_cost_values else None),
              "recall_limitation": "bounded deterministic expanded-goal oracle; not a simulator-optimality oracle"}
    selected_indices = list(range(candidate_k))
    if continue_index is not None and continue_index in selectable_indices:
        selected_indices.append(continue_index)
    commitment_digest = ("" if active_commitment is None else hashlib.sha256(
        json.dumps({"candidate_id": active_commitment.candidate_id,
            "action_index": active_commitment.action_index,
            "goal_variant": active_commitment.goal_variant,
            "source_xy_m": active_commitment.source_xy_m,
            "created_sequence": active_commitment.created_sequence,
            "expires_sequence": active_commitment.expires_sequence},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    table = CandidateTable(base.observation.snapshot_id,
        tuple(ids[i] for i in selected_indices), tuple(rows[i] for i in selected_indices),
        tuple(valid_mask[i] for i in selected_indices), names,
        route_costs=tuple(route_costs[i] for i in selected_indices),
        schema_version=base.table.schema_version, context_digest=commitment_digest)
    if active_commitment is not None and continue_index is not None:
        active_commitment = replace(active_commitment,
            action_index=int(specs[continue_index][0]))
    return RouteFeasibleCandidates(base.observation, table, fallback,
        tuple(specs[i][0] for i in selected_indices),
        torch.tensor([route_costs[i] for i in selected_indices], device=env.device, dtype=torch.float32),
        torch.tensor([risk_costs[i] for i in selected_indices], device=env.device, dtype=torch.float32),
        tuple(routes[i] for i in selected_indices), recall, active_commitment)
