"""Snapshot-bound simulator candidates with route and terminal-pose checks.

This first bounded candidate generator uses the existing eight executable
curriculum actions as its primitive set. It adds observed target poses, legal
SE(2) endpoints, current observed-robot route feasibility, and ETA estimates.
Its coverage is therefore limited to those eight actions. The report includes
only a sensed-track coverage proxy; it does not build a larger executable
oracle or measure candidate recall against one. K16/K32 enumeration and an
explicit continue candidate remain unimplemented.
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
    "version": 1,
    "capacity": 8,
    "source": "existing simulator action slots",
    "configurable_k": False,
    "k16_k32_supported": False,
    "higher_budget_oracle_recall_measured": False,
    "future_swept_dynamic_collision_check": False,
    "continue_behavior_candidate": False,
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
    point = xy.reshape(1, 1, 2).expand(1, 2, 2)
    static_ok = bool(env.planner._footprint_path_clear(point,
        torch.tensor([heading], device=env.device, dtype=point.dtype),
        length.reshape(1), width.reshape(1))[0].item())
    circles = env._perceived_robot_obstacles()[slot]
    circles = circles[circles[:, 2] > 0]
    if circles.numel() == 0:
        return static_ok
    delta = circles[:, :2] - xy
    c, s = math.cos(heading), math.sin(heading)
    forward = delta[:, 0] * c + delta[:, 1] * s
    lateral = -delta[:, 0] * s + delta[:, 1] * c
    dx = (forward.abs() - float(length) * .5).clamp_min(0.)
    dy = (lateral.abs() - float(width) * .5).clamp_min(0.)
    circles_clear = bool((dx.square() + dy.square() > circles[:, 2].square()).all().item())
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
        return None, None, hbin
    center = graph.xy[node]
    if float((center - xy).norm().item()) > graph.resolution_m * .72:
        return None, None, hbin
    if not _endpoint_clear(env, center, float(hbin) * 2 * math.pi / 8,
                           intake_extended=intake_extended):
        return None, None, hbin
    return node, center, hbin


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
                                    action_ticks: int = 12) -> RouteFeasibleCandidates:
    """Build the current bounded decision table and qualify every route row.

    Route feasibility uses a fresh uncached TensorADStar workspace over current
    perceived robot obstacles. The endpoint is additionally checked in the
    requested terminal heading against the static legal SE(2) lattice. ETA is
    distance/speed plus heading-change/turn-rate. Dynamic interaction is a
    point-prediction risk cost; full time-varying swept-robot planning remains
    out of scope and is recorded in provenance.
    """
    if env.n != 1 or not isinstance(episode_id, str) or not episode_id:
        raise ValueError("candidate source requires one world and a unique episode_id")
    if type(action_ticks) is not int or action_ticks < 1:
        raise ValueError("action_ticks must be a positive integer")
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
    targets = []
    headings = []
    route_queries = []
    goal_nodes: list[int | None] = []
    goal_centers = []
    hbins = []
    requested = []
    for action in range(8):
        target, heading, allowed, intake = _candidate_target(env, action, base)
        node, center, hbin = _goal_node(env, graph, target, heading,
                                        intake_extended=intake)
        allowed &= node is not None
        targets.append(target)
        headings.append(heading)
        goal_nodes.append(node)
        goal_centers.append(center)
        hbins.append(hbin)
        requested.append(allowed)
        if allowed:
            route_queries.append(RouteQuery(candidate_id=base.table.candidate_ids[action],
                snapshot_id=base.observation.snapshot_id, world=0, slot=3,
                goal_xy=(float(center[0]), float(center[1])), intake_extended=intake))

    route_results = evaluate_routes(env, route_queries,
        goal_tolerance_m=graph.resolution_m * .72,
        endpoint_tolerance_m=graph.resolution_m * .72)
    result_by_id = {result.candidate_id: result for result in route_results}
    speed = max(float(env.sim.speed[0, 3]), .1)
    omega = max(float(env.sim.omega_limit[0, 3]), .1)
    pose_heading = float(env.sim.pose[0, 3, 2])
    rows = []
    valid_mask = []
    routes = []
    route_costs = []
    risk_costs = []
    for action, old in enumerate(base.table.features):
        result = result_by_id.get(base.table.candidate_ids[action])
        distance = result.distance_m if result is not None and result.feasible else None
        route_ok = bool(requested[action] and result is not None and result.feasible and
                        goal_nodes[action] is not None)
        turn = abs(math.atan2(math.sin(headings[action] - pose_heading),
                              math.cos(headings[action] - pose_heading)))
        eta = (None if distance is None else distance / speed + turn / omega
               + action_ticks * float(env.dt))
        predicted = _opponent_prediction(env, horizon=eta or 0.)
        goal_xy = targets[action]
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
        row[4] = float(hbins[action])
        row.extend((None if goal_nodes[action] is None else float(goal_nodes[action]),
                    distance, eta, risk, float(headings[action])))
        rows.append(tuple(row))
        route_costs.append(0. if eta is None else eta * ROUTE_COST_POINTS_PER_SECOND)
        risk_costs.append(0. if risk is None else risk)
        routes.append({"candidate_id": base.table.candidate_ids[action],
            "action_index": action,
            "route_status": "not_queried" if result is None else result.status,
            "feasible": valid, "goal_xy_m": [float(goal_xy[0]), float(goal_xy[1])],
            "goal_heading_rad": float(headings[action]),
            "legal_node_index": goal_nodes[action], "distance_m": distance,
            "eta_s": eta, "predicted_opponent_xy_m": predicted,
            "endpoint_separation_m": separation, "risk_penalty_points": risk,
            "intake_extended": action < 4})
    if not any(valid_mask):
        raise RuntimeError("candidate service found no feasible row, including safe fallback")
    fallback = base.fallback_candidate_id
    if fallback not in base.table.candidate_ids or not valid_mask[base.table.candidate_ids.index(fallback)]:
        hold = base.table.candidate_ids.index("hold")
        if valid_mask[hold]:
            fallback = "hold"
        else:
            fallback = min((i for i, valid in enumerate(valid_mask) if valid),
                           key=lambda i: route_costs[i] + risk_costs[i])
            fallback = base.table.candidate_ids[fallback]
    names = base.table.feature_names + EXTRA_FEATURES
    table = CandidateTable(base.observation.snapshot_id, base.table.candidate_ids,
        tuple(rows), tuple(valid_mask), names,
        route_costs=tuple(route_costs), schema_version=base.table.schema_version)
    sensed = int(env.track_mask[0, 3].sum().item())
    recall = {"candidate_capacity": 8, "valid_executable_candidates": sum(valid_mask),
              "sensed_track_oracle_count": sensed,
              "executable_track_rows": sum(valid_mask[:4]),
              "track_candidate_coverage_proxy": (sum(valid_mask[:4]) / sensed if sensed else None),
              "higher_budget_oracle_recall_measured": False}
    return RouteFeasibleCandidates(base.observation, table, fallback, tuple(range(8)),
        torch.tensor(route_costs, device=env.device, dtype=torch.float32),
        torch.tensor(risk_costs, device=env.device, dtype=torch.float32),
        tuple(routes), recall)
