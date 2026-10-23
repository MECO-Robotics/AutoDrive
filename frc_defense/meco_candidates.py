"""Snapshot-bound simulator candidates with route and capture-geometry estimates.

The existing eight masked actions remain the behavior primitives. K=16/32 add
stable, locally offset collection approach goals. The selected endpoint is
passed to AD* while the simulator retains action legality and mechanisms. K32
reports within-pool coverage only; it is not an independent oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from .experimental_route_evaluation import RouteQuery, evaluate_routes
from .learning_contract import CandidateTable
from .meco_pose_graph import LegalPoseGraph
from .meco_tracks import SafeActionCandidates, build_safe_action_candidates, predict_opponent_endpoint


ROUTE_COST_POINTS_PER_SECOND = 0.05
RISK_RADIUS_M = 2.0
RISK_MAX_POINTS = 0.75
CANDIDATE_BOUNDARY = {
    "version": 2,
    "capacity": [8, 16, 32],
    "source": "existing simulator action slots",
    "configurable_k": True,
    "k16_k32_supported": True,
    "within_candidate_pool_recall_only": True,
    "reference_pool": "route-qualified K32 from the same sensed action slots and fixed offsets",
    "recall_limit": "K32 recall is tautologically 1; no distinct >32 reference pool is measured",
    "expanded_goals": "bounded collection approach offsets",
    "execution": "selected XY endpoint overrides planner goal; action mask/mechanisms remain simulator-owned",
    "heading_execution": "no terminal heading command; sensed-target bearing is used for graph lookup and nonbinding estimate; native controller owns orientation",
    "future_swept_dynamic_collision_check": False,
    "continue_behavior_candidate": False,
    "continue_gap": "no persistent selected commitment exists; hold is the stationary fallback",
    "route_feasibility": "route and endpoint clearance only; inferred terminal heading is not commanded",
    "capture_estimate": "nonbinding geometry estimate from sensed target position; never used as a legality mask",
}
EXTRA_FEATURES = ("legal_node_index", "route_distance_m", "route_time_s",
                  "risk_penalty_points", "target_heading_rad")


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


def _candidate_target(env, action: int, base: SafeActionCandidates):
    slot = 3
    pose = env.sim.pose[0, slot]
    xy = pose[:2].clone()
    heading = float(pose[2].item())
    valid = bool(base.table.valid_mask[action])
    intake = action < 4
    if action < 4:
        row = base.table.features[action]
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
        # Heading comes from sensed target bearing and is not commanded by the
        # native controller. It may select model features, but it cannot make
        # an otherwise legal XY endpoint infeasible. Bind the graph row to the
        # first legal heading at this cell for A2/A3 node lookup.
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
                           goal_xy) -> dict:
    """Report sensed-target geometry without implying commanded orientation."""
    if action >= 4:
        return {"status": "not_applicable", "basis": "non_collection_action"}
    row = base.table.features[action]
    if row[2] is None or row[3] is None:
        return {"status": "unavailable", "basis": "target_not_sensed"}
    slot = 3
    fuel = goal_xy.new_tensor((row[2], row[3]))
    delta = fuel - goal_xy
    distance = float(delta.norm().item())
    if not math.isfinite(distance):
        return {"status": "unavailable", "basis": "nonfinite_target_geometry"}
    radius = (float(env.fuel_physics.config.radius) if env.fuel_physics is not None
              else float(env._fuel_radius))
    half_length = float(env.sim.length[0, slot]) * .5
    half_width = float(env.sim.width[0, slot]) * .5
    forward_limit = (half_length + .35 + .3048 + radius)
    lateral_limit = half_width + .075 + radius
    # Bearing-aligned point geometry only; actual terminal orientation and
    # capture are controlled by the native controller and simulator.
    inside = (-half_length - .075 - radius <= distance <= forward_limit
              and 0. <= lateral_limit)
    return {"status": "inside_estimate" if inside else "outside_estimate",
            "basis": "sensed_target_bearing_geometry_only",
            "distance_m": distance,
            "estimated_forward_limit_m": forward_limit,
            "estimated_lateral_limit_m": lateral_limit}


def _opponent_prediction(env, slot=3, horizon=0.):
    valid = bool(env.opponent_valid[0, slot])
    age = float(env.opponent_age[0, slot])
    position = env.opponent_pose[0, slot, :2].detach().cpu().tolist()
    velocity = env.opponent_velocity[0, slot, :2].detach().cpu().tolist()
    return predict_opponent_endpoint(valid=valid, position_xy=position,
        velocity_xy=velocity, age_s=age, horizon_s=max(0., float(horizon)))


@torch.no_grad()
def build_route_feasible_candidates(env, observation: torch.Tensor,
                                    graph: LegalPoseGraph, *, episode_id: str,
                                    action_ticks: int = 12,
                                    candidate_k: int = 8,
                                    learned_traversal_cost=None,
                                    learned_cost_scale: float = 1.0) -> RouteFeasibleCandidates:
    """Build the current bounded decision table and qualify every route row.

    Route feasibility uses a fresh uncached TensorADStar workspace over current
    perceived robot obstacles. The inferred target-bearing heading selects a
    legal graph node and contributes an estimate only; endpoint footprint
    clearance is checked at every yaw. The native controller owns terminal
    orientation; this adapter does not command it. ETA is
    distance/speed plus heading-change/turn-rate. Dynamic interaction is a
    point-prediction risk cost; full time-varying swept-robot planning remains
    out of scope and is recorded in provenance.
    """
    if env.n != 1 or not isinstance(episode_id, str) or not episode_id:
        raise ValueError("candidate source requires one world and a unique episode_id")
    if type(action_ticks) is not int or action_ticks < 1:
        raise ValueError("action_ticks must be a positive integer")
    if type(candidate_k) is not int or candidate_k not in (8, 16, 32):
        raise ValueError("candidate_k must be one of 8, 16, or 32")
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
    specs = [(action, 0, 0., 0.) for action in range(8)]
    offsets = ((.22, 0.), (-.22, 0.), (0., .22), (0., -.22),
               (.34, 0.), (-.34, 0.))
    for variant, (forward, lateral) in enumerate(offsets, 1):
        for action in range(4):
            specs.append((action, variant, forward, lateral))
    targets, headings, route_queries, goal_nodes = [], [], [], []
    hbins, requested, ids = [], [], []
    for row_index, (action, variant, forward_offset, lateral_offset) in enumerate(specs):
        target, heading, allowed, intake = _candidate_target(env, action, base)
        candidate_id = (base.table.candidate_ids[action] if not variant else
                        f"{base.table.candidate_ids[action]}:approach:{variant}")
        ids.append(candidate_id)
        if variant:
            forward = target.new_tensor((math.cos(heading), math.sin(heading)))
            left = target.new_tensor((-math.sin(heading), math.cos(heading)))
            target = target + forward * forward_offset + left * lateral_offset
        if action < 4:
            fuel_row = base.table.features[action]
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
                fuel_row = base.table.features[action]
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
        targets.append(target)
        headings.append(heading)
        goal_nodes.append(node)
        hbins.append(hbin)
        requested.append(allowed)
        if allowed:
            route_queries.append(RouteQuery(candidate_id=candidate_id,
                snapshot_id=base.observation.snapshot_id, world=0, slot=3,
                goal_xy=(float(center[0]), float(center[1])), intake_extended=intake))

    route_results = evaluate_routes(env, route_queries,
        goal_tolerance_m=graph.resolution_m * .72,
        endpoint_tolerance_m=graph.resolution_m * .72,
        learned_traversal_cost=learned_traversal_cost,
        learned_cost_scale=learned_cost_scale,
        learned_cost_provenance="meco_legal_heading_penalty_projection_v1")
    result_by_id = {result.candidate_id: result for result in route_results}
    speed = max(float(env.sim.speed[0, 3]), .1)
    omega = max(float(env.sim.omega_limit[0, 3]), .1)
    pose_heading = float(env.sim.pose[0, 3, 2])
    rows = []
    valid_mask = []
    routes = []
    route_costs = []
    risk_costs = []
    for row_index, (action, variant, _, _) in enumerate(specs):
        old = base.table.features[action]
        candidate_id = ids[row_index]
        result = result_by_id.get(candidate_id)
        distance = result.distance_m if result is not None and result.feasible else None
        route_ok = bool(requested[row_index] and result is not None and result.feasible and
                        goal_nodes[row_index] is not None)
        turn = abs(math.atan2(math.sin(headings[row_index] - pose_heading),
                              math.cos(headings[row_index] - pose_heading)))
        eta = (None if distance is None else distance / speed + turn / omega
               + action_ticks * float(env.dt))
        predicted = _opponent_prediction(env, horizon=eta or 0.)
        goal_xy = targets[row_index]
        if predicted is None and bool(env.opponent_valid[0, 3]):
            predicted = tuple(float(x) for x in env.opponent_pose[0, 3, :2])
        separation = None if predicted is None else math.hypot(
            float(goal_xy[0]) - predicted[0], float(goal_xy[1]) - predicted[1])
        risk = (None if separation is None else RISK_MAX_POINTS *
                max(0., min(1., (RISK_RADIUS_M - separation) / RISK_RADIUS_M)))
        valid = bool(base.table.valid_mask[action] and route_ok)
        valid_mask.append(valid)
        # Update the shared target to the legal route endpoint, and preserve
        # every missing estimate explicitly for the model's missingness mask.
        row = list(old)
        row[2] = float(goal_xy[0])
        row[3] = float(goal_xy[1])
        row[4] = float(hbins[row_index])
        row.extend((None if goal_nodes[row_index] is None else float(goal_nodes[row_index]),
                    distance, eta, risk, float(headings[row_index])))
        rows.append(tuple(row))
        route_costs.append(0. if eta is None else eta * ROUTE_COST_POINTS_PER_SECOND)
        risk_costs.append(0. if risk is None else risk)
        capture_estimate = _capture_pose_estimate(env, base, action, goal_xy)
        routes.append({"candidate_id": candidate_id,
            "action_index": action,
            "candidate_row": row_index,
            "goal_variant": variant,
            "route_status": "not_queried" if result is None else result.status,
            "feasible": valid, "route_feasible": route_ok,
            "capture_estimate": capture_estimate,
            "goal_xy_m": [float(goal_xy[0]), float(goal_xy[1])],
            "goal_heading_rad": float(headings[row_index]),
            "legal_node_index": goal_nodes[row_index], "distance_m": distance,
            "eta_s": eta, "predicted_opponent_xy_m": predicted,
            "endpoint_separation_m": separation, "risk_penalty_points": risk,
            "intake_extended": action < 4,
            "path_points_m": (None if result is None else result.path_points_m),
            "weighted_route_cost_grid_edges": (None if result is None
                else result.weighted_cost),
            "route_cost_units": (None if result is None else result.cost_units),
            "route_cost_provenance": (None if result is None
                else result.cost_provenance)})
    if not any(valid_mask[:candidate_k]):
        raise RuntimeError("candidate service found no feasible row, including safe fallback")
    fallback = base.fallback_candidate_id
    if fallback not in ids or not valid_mask[ids.index(fallback)]:
        hold = ids.index("hold")
        if valid_mask[hold]:
            fallback = "hold"
        else:
            fallback = min((i for i, valid in enumerate(valid_mask[:candidate_k]) if valid),
                           key=lambda i: route_costs[i] + risk_costs[i])
            fallback = ids[fallback]
    names = base.table.feature_names + EXTRA_FEATURES
    reference_ids = {ids[i] for i, valid in enumerate(valid_mask) if valid}
    requested_ids = {ids[i] for i, valid in enumerate(valid_mask[:candidate_k]) if valid}
    recall = {"candidate_capacity": candidate_k, "reference_pool_capacity": 32,
              "valid_executable_candidates": sum(valid_mask[:candidate_k]),
              "reference_scope": "within_candidate_pool",
              "within_candidate_pool_recall_measured": True,
              "route_feasible_rows_in_reference_pool": len(reference_ids),
              "within_candidate_pool_recall": (len(requested_ids & reference_ids) / len(reference_ids) if reference_ids else 1.),
              "recall_limitation": "K32 is the same candidate pool; its recall is tautologically 1 and no distinct >32 reference is measured"}
    table = CandidateTable(base.observation.snapshot_id, tuple(ids[:candidate_k]),
        tuple(rows[:candidate_k]), tuple(valid_mask[:candidate_k]), names,
        route_costs=tuple(route_costs[:candidate_k]), schema_version=base.table.schema_version)
    return RouteFeasibleCandidates(base.observation, table, fallback,
        tuple(spec[0] for spec in specs[:candidate_k]),
        torch.tensor(route_costs[:candidate_k], device=env.device, dtype=torch.float32),
        torch.tensor(risk_costs[:candidate_k], device=env.device, dtype=torch.float32),
        tuple(routes[:candidate_k]), recall)
