"""Isolated route feasibility probes for experimental candidate scoring.

This module does not participate in controller execution. It evaluates routes
against the same current obstacle geometry used by ``TensorThreeVsThreeEnv``
and returns geometric distance only; it does not estimate travel time.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Literal

import torch

from .tensor_adstar import TensorADStar


RouteStatus = Literal["feasible", "at_goal", "no_path", "invalid"]
MAX_ROUTE_QUERY_BATCH = 64


@dataclass(frozen=True, slots=True)
class RouteQuery:
    """A requested XY route for one observed world/robot at one snapshot."""

    candidate_id: str
    snapshot_id: str
    world: int
    slot: int
    goal_xy: tuple[float, float]
    intake_extended: bool = False


@dataclass(frozen=True, slots=True)
class RouteResult:
    candidate_id: str
    snapshot_id: str
    world: int
    slot: int
    status: RouteStatus
    feasible: bool
    distance_m: float | None


def _finite_pair(value) -> bool:
    try:
        return len(value) == 2 and all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def _inside_field(point, length: float, width: float) -> bool:
    return (0.0 <= float(point[0]) <= length and
            0.0 <= float(point[1]) <= width)


def _scratch_planner(env, query_rows: list[tuple[RouteQuery, torch.Tensor]]) -> TensorADStar:
    """Build a fresh planner workspace; never borrow live planner buffers."""
    live = env.planner
    device = env.device
    count = len(query_rows)
    starts = torch.stack([env.sim.pose[q.world, q.slot, :2] for q, _ in query_rows])
    headings = torch.stack([env.sim.pose[q.world, q.slot, 2] for q, _ in query_rows])
    velocity = torch.stack([env.sim.velocity[q.world, q.slot] for q, _ in query_rows])
    sim = SimpleNamespace(
        pose=torch.cat((starts, headings[:, None]), dim=-1)[:, None, :],
        velocity=velocity[:, None, :],
        field_length=env.sim.field_length,
        field_width=env.sim.field_width,
        field_colliders=env.sim.field_colliders,
        obstacles=env.sim.obstacles,
        swerve=getattr(env.sim, "swerve", None),
        omega_limit=torch.stack([env.sim.omega_limit[q.world, q.slot]
                                 for q, _ in query_rows]),
        alpha=torch.stack([env.sim.alpha[q.world, q.slot]
                           for q, _ in query_rows]),
    )
    scratch_env = SimpleNamespace(device=device, n=count, sim=sim,
                                  field_boxes=env.field_boxes)
    planner = TensorADStar(
        scratch_env, resolution=live.resolution, max_points=live.max_points,
        sweeps=live.sweeps, avoid_bumps=live.avoid_bumps,
        early_convergence=live.early_convergence)
    planner.footprint_clearance = live.footprint_clearance
    # Occupancy behavior and geometry must match the active implementation.
    planner._fused_blocked_hip_enabled = live._fused_blocked_hip_enabled
    return planner


@torch.no_grad()
def evaluate_routes(env, queries: tuple[RouteQuery, ...] | list[RouteQuery],
                    *, goal_tolerance_m: float = 0.05,
                    endpoint_tolerance_m: float = 0.05) -> tuple[RouteResult, ...]:
    """Evaluate an ordered batch against current observed robot obstacles.

    Starts, footprint and motion limits come from the live simulator state.
    Dynamic geometry is taken from ``env._perceived_robot_obstacles()``, the
    exact helper used by the controller. Teammates use its current localization
    feed and opponents use its observed track; hidden opponent truth is not read.

    This is synchronous-only and uncached: all rows must name one snapshot, and
    callers must build queries from the current environment state. This is a
    snapshot XY route feasibility check. It does not enforce terminal
    heading, intake capture, mechanism legality, or time-varying collision
    prediction. Callers must validate those task constraints separately.
    """
    items = tuple(queries)
    if not math.isfinite(goal_tolerance_m) or goal_tolerance_m < 0:
        raise ValueError("goal_tolerance_m must be finite and nonnegative")
    if not math.isfinite(endpoint_tolerance_m) or endpoint_tolerance_m < 0:
        raise ValueError("endpoint_tolerance_m must be finite and nonnegative")
    if not items:
        return ()
    if len(items) > MAX_ROUTE_QUERY_BATCH:
        raise ValueError(f"route batch exceeds {MAX_ROUTE_QUERY_BATCH} queries")

    snapshot_ids = {q.snapshot_id for q in items
                    if isinstance(q, RouteQuery) and isinstance(q.snapshot_id, str)}
    if len(snapshot_ids) > 1:
        raise ValueError("one route batch must use a single snapshot_id")

    results: list[RouteResult | None] = [None] * len(items)
    valid_rows: list[tuple[int, RouteQuery]] = []
    seen: set[tuple[int, int, str]] = set()
    for index, query in enumerate(items):
        if not isinstance(query, RouteQuery):
            raise TypeError("queries must contain RouteQuery values")
        key = ((query.world, query.slot, query.candidate_id)
               if (isinstance(query.candidate_id, str) and
                   type(query.world) is int and type(query.slot) is int) else None)
        valid = (isinstance(query.candidate_id, str) and bool(query.candidate_id) and
                 isinstance(query.snapshot_id, str) and bool(query.snapshot_id) and
                 type(query.world) is int and 0 <= query.world < env.n and
                 type(query.slot) is int and 0 <= query.slot < 6 and
                 type(query.intake_extended) is bool and
                 _finite_pair(query.goal_xy) and key is not None and key not in seen and
                 _inside_field(query.goal_xy, float(env.sim.field_length),
                               float(env.sim.field_width)))
        if valid:
            seen.add(key)
            valid_rows.append((index, query))
        else:
            results[index] = RouteResult(query.candidate_id, query.snapshot_id,
                query.world, query.slot, "invalid", False, None)

    if valid_rows:
        obstacle_snapshot = env._perceived_robot_obstacles()
        planner_rows: list[tuple[RouteQuery, torch.Tensor]] = []
        for _, query in valid_rows:
            goal = torch.as_tensor(query.goal_xy, device=env.device,
                                    dtype=env.sim.pose.dtype)
            planner_rows.append((query, goal))
        device = env.device
        starts = torch.stack([env.sim.pose[q.world, q.slot, :2]
                              for q, _ in planner_rows])
        headings = torch.stack([env.sim.pose[q.world, q.slot, 2]
                                for q, _ in planner_rows])
        goals = torch.stack([goal for _, goal in planner_rows])
        lengths = torch.stack([
            env.sim.length[q.world, q.slot] +
            (0.6096 if q.intake_extended else 0.0)
            for q, _ in planner_rows])
        widths = torch.stack([env.sim.width[q.world, q.slot]
                              for q, _ in planner_rows])
        speed = torch.stack([env.sim.speed[q.world, q.slot]
                             for q, _ in planner_rows])
        friction = torch.stack([env.sim.lateral_mu[q.world, q.slot]
                                for q, _ in planner_rows])
        acceleration = torch.stack([env.sim.accel[q.world, q.slot]
                                    for q, _ in planner_rows])
        obstacle_rows = torch.stack([
            obstacle_snapshot[q.world * 6 + q.slot]
            for q, _ in planner_rows
        ])
        values_finite = (torch.isfinite(starts).all(-1) &
                         torch.isfinite(headings) & torch.isfinite(goals).all(-1) &
                         torch.isfinite(lengths) & torch.isfinite(widths) &
                         torch.isfinite(speed) & torch.isfinite(friction) &
                         torch.isfinite(acceleration) &
                         torch.isfinite(obstacle_rows).all(dim=(-1, -2)))
        values_finite &= ((lengths > 0) & (widths > 0) & (speed > 0) &
                          (friction > 0) & (acceleration > 0))
        values_finite &= ((starts[:, 0] >= 0) &
                          (starts[:, 0] <= float(env.sim.field_length)) &
                          (starts[:, 1] >= 0) &
                          (starts[:, 1] <= float(env.sim.field_width)))

        valid_indices = torch.nonzero(values_finite, as_tuple=False).flatten()
        invalid_indices = torch.nonzero(~values_finite, as_tuple=False).flatten()
        for local_index in invalid_indices.tolist():
            original_index, query = valid_rows[local_index]
            results[original_index] = RouteResult(query.candidate_id,
                query.snapshot_id, query.world, query.slot, "invalid", False, None)

        if valid_indices.numel():
            valid_queries = [planner_rows[i] for i in valid_indices.tolist()]
            scratch = _scratch_planner(env, valid_queries)
            start = starts[valid_indices]
            heading = headings[valid_indices]
            goal = goals[valid_indices]
            length = lengths[valid_indices]
            width = widths[valid_indices]
            obstacle = obstacle_rows[valid_indices]
            goal_distance = (goal - start).norm(dim=-1)
            at_goal_mask = goal_distance <= goal_tolerance_m

            # Reject blocked endpoints before plan() can project them onto a
            # nearby free cell and accidentally report the wrong candidate.
            sx, sy = scratch._indices(start)
            gx, gy = scratch._indices(goal)
            blocked = scratch._blocked(heading, length, width)
            blocked = scratch._block_robot_obstacles(blocked, length, width, obstacle)
            rows = torch.arange(start.shape[0], device=device)
            endpoint_blocked = blocked[rows, sx, sy] | blocked[rows, gx, gy]
            at_goal_valid = at_goal_mask & ~endpoint_blocked
            for local_index in torch.nonzero(at_goal_valid, as_tuple=False).flatten().tolist():
                original_index, query = valid_rows[valid_indices[local_index].item()]
                results[original_index] = RouteResult(query.candidate_id,
                    query.snapshot_id, query.world, query.slot, "at_goal", True, 0.0)

            at_goal_blocked = at_goal_mask & endpoint_blocked
            for local_index in torch.nonzero(at_goal_blocked, as_tuple=False).flatten().tolist():
                original_index, query = valid_rows[valid_indices[local_index].item()]
                results[original_index] = RouteResult(query.candidate_id,
                    query.snapshot_id, query.world, query.slot, "no_path", False, None)

            route_mask = ~at_goal_mask & ~endpoint_blocked
            rejected_indices = torch.nonzero((~at_goal_mask) & endpoint_blocked,
                                             as_tuple=False).flatten()
            for local_index in rejected_indices.tolist():
                original_index, query = valid_rows[valid_indices[local_index].item()]
                results[original_index] = RouteResult(query.candidate_id,
                    query.snapshot_id, query.world, query.slot, "no_path", False, None)

            if route_mask.any():
                scratch.plan(start, goal, heading, length, width,
                    speed=speed[valid_indices],
                    lateral_friction=friction[valid_indices],
                    acceleration=acceleration[valid_indices],
                    active_mask=route_mask, robot_obstacles=obstacle)
                for batch_local in torch.nonzero(route_mask, as_tuple=False).flatten().tolist():
                    original_index, query = valid_rows[valid_indices[batch_local].item()]
                    count = int(scratch.last_lengths[batch_local].item())
                    points = scratch.last_path[batch_local, :count]
                    if count <= 1:
                        result = RouteResult(query.candidate_id, query.snapshot_id,
                            query.world, query.slot, "no_path", False, None)
                    else:
                        endpoint_error = float((points[-1] - goal[batch_local]).norm().item())
                        clear = bool(scratch._footprint_path_clear(
                            points[None], heading[batch_local:batch_local+1],
                            length[batch_local:batch_local+1],
                            width[batch_local:batch_local+1])[0].item())
                        distance = float((points[1:] - points[:-1]).norm(dim=-1).sum().item())
                        if (not clear or not math.isfinite(distance) or
                                endpoint_error > endpoint_tolerance_m):
                            result = RouteResult(query.candidate_id, query.snapshot_id,
                                query.world, query.slot, "no_path", False, None)
                        else:
                            result = RouteResult(query.candidate_id, query.snapshot_id,
                                query.world, query.slot, "feasible", True, distance)
                    results[original_index] = result

    return tuple(result for result in results if result is not None)
