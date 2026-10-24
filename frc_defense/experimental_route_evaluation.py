"""Isolated route feasibility probes for experimental candidate scoring.

This module does not participate in controller execution. It evaluates routes
against the same current obstacle geometry used by ``TensorThreeVsThreeEnv``
and returns geometric distance only; it does not estimate travel time.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
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
    path_points_m: tuple[tuple[float, float], ...] | None = None
    weighted_cost: float | None = None
    cost_units: str = "normalized_grid_edge_units"
    cost_provenance: str = "legacy_bump_cost_path_integral_v1"


def _finite_pair(value) -> bool:
    try:
        return len(value) == 2 and all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def _inside_field(point, length: float, width: float) -> bool:
    return (0.0 <= float(point[0]) <= length and
            0.0 <= float(point[1]) <= width)


def _pose_clear_of_circles(points, heading, length, width, circles) -> torch.Tensor:
    """Exact oriented-rectangle clearance against center/radius circles."""
    if circles.shape[-2] == 0:
        return torch.ones(points.shape[0], device=points.device, dtype=torch.bool)
    delta = circles[..., :2] - points[:, None, :]
    c, s = heading.cos()[:, None], heading.sin()[:, None]
    forward = delta[..., 0] * c + delta[..., 1] * s
    lateral = -delta[..., 0] * s + delta[..., 1] * c
    dx = (forward.abs() - length[:, None] * .5).clamp_min(0.)
    dy = (lateral.abs() - width[:, None] * .5).clamp_min(0.)
    overlap = (dx.square() + dy.square() <= circles[..., 2].clamp_min(0.).square())
    overlap &= circles[..., 2] > 0.
    return ~overlap.any(dim=-1)


def _pose_clear(scratch, point, heading, length, width, robot_obstacles):
    """Check an exact pose against field geometry and observed robot circles."""
    path = point[:, None, :].expand(-1, 2, -1)
    static_clear = scratch._footprint_path_clear(path, heading, length, width)
    static_circles = scratch._extra_circles.reshape(1, -1, 3).expand(
        point.shape[0], -1, -1)
    circles = torch.cat((static_circles,
                         robot_obstacles), dim=1)
    circle_clear = _pose_clear_of_circles(point, heading, length, width, circles)
    return static_clear & circle_clear


def _cost_map_digest(cost_map, *, resolution: float,
                     field_length: float, field_width: float) -> str:
    """Hash raster geometry and canonical row-major float64 map data."""
    values = cost_map.detach().to(device="cpu", dtype=torch.float64).contiguous()
    nx, ny = values.shape
    packed_values = struct.pack(f"<{values.numel()}d", *values.reshape(-1).tolist())
    # Cell (i, j) is at ((i + .5) * resolution, (j + .5) * resolution).
    # Encode the field origin and full affine index-to-meter transform, along
    # with the field bounds, so the same matrix on another grid hashes apart.
    geometry = struct.pack(
        "<11d", float(resolution), float(field_length), float(field_width),
        0.0, 0.0,
        float(resolution), 0.0, float(resolution) * 0.5,
        0.0, float(resolution), float(resolution) * 0.5)
    payload = (b"autodrive.learned_traversal_cost_map.xy_cell_center_affine_v2\0" +
               struct.pack("<II", nx, ny) + geometry + packed_values)
    return hashlib.sha256(payload).hexdigest()


def _path_weighted_cost(scratch, points, learned_cost, scale: float) -> float:
    """Integrate terrain multipliers along points in cardinal-grid-step units.

    Each straight segment is split at every crossed raster boundary, then its
    length fraction is charged at that cell's multiplier. Bump traversal
    retains the planner's +0.12 cost; learned map values are additional
    dimensionless cost per cardinal step.
    """
    if points.shape[0] < 2:
        return 0.0
    terrain = scratch._bump_cost
    if learned_cost is not None:
        terrain = terrain + learned_cost * scale
    coordinates = points.detach().to(device="cpu", dtype=torch.float64).tolist()
    terrain_values = terrain.detach().to(device="cpu", dtype=torch.float64).tolist()
    resolution = float(scratch.resolution)
    total = 0.0
    for first, second in zip(coordinates, coordinates[1:]):
        x0, y0 = first
        x1, y1 = second
        dx, dy = x1 - x0, y1 - y0
        segment_length = math.hypot(dx, dy)
        if segment_length == 0.0:
            continue
        crossings = {0.0, 1.0}
        if dx != 0.0:
            low, high = sorted((x0, x1))
            first_boundary = math.floor(low / resolution) + 1
            last_boundary = math.ceil(high / resolution) - 1
            for cell_boundary in range(first_boundary, last_boundary + 1):
                fraction = (cell_boundary * resolution - x0) / dx
                if 0.0 < fraction < 1.0:
                    crossings.add(fraction)
        if dy != 0.0:
            low, high = sorted((y0, y1))
            first_boundary = math.floor(low / resolution) + 1
            last_boundary = math.ceil(high / resolution) - 1
            for cell_boundary in range(first_boundary, last_boundary + 1):
                fraction = (cell_boundary * resolution - y0) / dy
                if 0.0 < fraction < 1.0:
                    crossings.add(fraction)
        ordered = sorted(crossings)
        for begin, end in zip(ordered, ordered[1:]):
            midpoint = (begin + end) * 0.5
            x = x0 + midpoint * dx
            y = y0 + midpoint * dy
            ix = min(max(math.floor(x / resolution), 0), scratch.nx - 1)
            iy = min(max(math.floor(y / resolution), 0), scratch.ny - 1)
            total += (segment_length / resolution) * (end - begin) * terrain_values[ix][iy]
    return total


def _scratch_planner(env, query_rows: list[tuple[RouteQuery, torch.Tensor]]) -> TensorADStar:
    """Build a fresh planner workspace; never borrow live planner buffers."""
    live = env.planner
    device = env.device
    count = len(query_rows)
    starts = torch.stack([env.policy_pose[q.world, q.slot, :2] for q, _ in query_rows])
    headings = torch.stack([env.policy_pose[q.world, q.slot, 2] for q, _ in query_rows])
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
                    endpoint_tolerance_m: float = 0.05,
                    learned_traversal_cost=None,
                    learned_cost_scale: float = 1.0,
                    learned_cost_provenance: str = "caller_supplied") -> tuple[RouteResult, ...]:
    """Evaluate an ordered batch against current observed robot obstacles.

    Starts use policy localization; footprint and motion limits come from
    configured robot parameters. Dynamic geometry is taken from
    ``env._perceived_robot_obstacles()``, the
    exact helper used by the controller. Teammates use its current localization
    feed and opponents use its observed track; hidden opponent truth is not read.

    This is synchronous-only and uncached: all rows must name one snapshot, and
    callers must build queries from the current environment state. This is a
    snapshot XY route feasibility check. It does not enforce terminal
    heading, intake capture, mechanism legality, or time-varying collision
    prediction. Callers must validate those task constraints separately.

    ``learned_traversal_cost`` is a shared ``[nx, ny]`` field map on the
    planner grid. Its finite, nonnegative values are dimensionless additive
    penalties per cardinal grid step (base cardinal cost 1.0, diagonal sqrt(2));
    ``learned_cost_scale`` multiplies those values. They affect route choice,
    never occupancy or footprint safety. Results retain XY path points in
    meters and report a path-integrated cost in normalized grid-edge units.
    Provenance records the producer, scale, raster schema and SHA-256 digest
    of the normalized map values.
    """
    items = tuple(queries)
    if not math.isfinite(goal_tolerance_m) or goal_tolerance_m < 0:
        raise ValueError("goal_tolerance_m must be finite and nonnegative")
    if not math.isfinite(endpoint_tolerance_m) or endpoint_tolerance_m < 0:
        raise ValueError("endpoint_tolerance_m must be finite and nonnegative")
    try:
        learned_cost_scale = float(learned_cost_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("learned_cost_scale must be finite and nonnegative") from exc
    if not math.isfinite(learned_cost_scale) or learned_cost_scale < 0:
        raise ValueError("learned_cost_scale must be finite and nonnegative")
    if not isinstance(learned_cost_provenance, str) or not learned_cost_provenance:
        raise ValueError("learned_cost_provenance must be a nonempty string")
    learned_map = None
    if learned_traversal_cost is None:
        cost_provenance = "legacy_bump_cost_path_integral_v2"
    else:
        learned_map = torch.as_tensor(learned_traversal_cost,
                                      device=env.device,
                                      dtype=env.policy_pose.dtype)
        expected_shape = (env.planner.nx, env.planner.ny)
        if learned_map.shape != expected_shape:
            raise ValueError(f"learned_traversal_cost must have shape {expected_shape}")
        if (not bool(torch.isfinite(learned_map).all().item()) or
                bool((learned_map < 0).any().item())):
            raise ValueError("learned_traversal_cost must be finite and nonnegative")
        effective_map = learned_map * learned_cost_scale
        if not bool(torch.isfinite(effective_map).all().item()):
            raise ValueError("scaled learned_traversal_cost must remain finite")
        resolution = float(env.planner.resolution)
        field_length = float(env.sim.field_length)
        field_width = float(env.sim.field_width)
        digest = _cost_map_digest(learned_map, resolution=resolution,
                                  field_length=field_length,
                                  field_width=field_width)
        geometry_provenance = (
            f"grid=xy_cell_center_affine_v2:r={resolution.hex()}:"
            f"L={field_length.hex()}:W={field_width.hex()}:"
            "origin=(0x0.0p+0,0x0.0p+0):"
            f"transform=((r,0,{(resolution * 0.5).hex()}),"
            f"(0,r,{(resolution * 0.5).hex()}))")
        cost_provenance = (
            f"{learned_cost_provenance}:schema=xy_grid_f64le_v2:"
            f"{geometry_provenance}:"
            f"sha256={digest}:scale={learned_cost_scale.hex()}:"
            "path_integral_grid_edge_v2")
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
                query.world, query.slot, "invalid", False, None,
                cost_provenance=cost_provenance)

    if valid_rows:
        obstacle_snapshot = env._perceived_robot_obstacles()
        planner_rows: list[tuple[RouteQuery, torch.Tensor]] = []
        for _, query in valid_rows:
            goal = torch.as_tensor(query.goal_xy, device=env.device,
                                    dtype=env.policy_pose.dtype)
            planner_rows.append((query, goal))
        device = env.device
        starts = torch.stack([env.policy_pose[q.world, q.slot, :2]
                              for q, _ in planner_rows])
        headings = torch.stack([env.policy_pose[q.world, q.slot, 2]
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
                query.snapshot_id, query.world, query.slot, "invalid", False, None,
                cost_provenance=cost_provenance)

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

            # Goals on blocked cells are not silently accepted after AD* projects
            # them. Starts may occupy blocked raster cells: AD* has a visible
            # escape path for those while preserving the measured pose.
            gx, gy = scratch._indices(goal)
            blocked = scratch._blocked(heading, length, width)
            blocked = scratch._block_robot_obstacles(blocked, length, width, obstacle)
            rows = torch.arange(start.shape[0], device=device)
            goal_blocked = blocked[rows, gx, gy]
            at_goal_clear = at_goal_mask & _pose_clear(
                scratch, start, heading, length, width, obstacle)
            at_goal_clear &= _pose_clear(
                scratch, goal, heading, length, width, obstacle)
            for local_index in torch.nonzero(at_goal_clear, as_tuple=False).flatten().tolist():
                original_index, query = valid_rows[valid_indices[local_index].item()]
                results[original_index] = RouteResult(query.candidate_id,
                    query.snapshot_id, query.world, query.slot, "at_goal", True, 0.0,
                    tuple((float(x), float(y)) for x, y in start[local_index:local_index+1].tolist()),
                    0.0, cost_provenance=cost_provenance)

            at_goal_blocked = at_goal_mask & ~at_goal_clear
            for local_index in torch.nonzero(at_goal_blocked, as_tuple=False).flatten().tolist():
                original_index, query = valid_rows[valid_indices[local_index].item()]
                results[original_index] = RouteResult(query.candidate_id,
                    query.snapshot_id, query.world, query.slot, "no_path", False, None,
                    cost_provenance=cost_provenance)

            route_mask = ~at_goal_mask & ~goal_blocked
            rejected_indices = torch.nonzero((~at_goal_mask) & goal_blocked,
                                             as_tuple=False).flatten()
            for local_index in rejected_indices.tolist():
                original_index, query = valid_rows[valid_indices[local_index].item()]
                results[original_index] = RouteResult(query.candidate_id,
                    query.snapshot_id, query.world, query.slot, "no_path", False, None,
                    cost_provenance=cost_provenance)

            if route_mask.any():
                # Supplying this keyword only when present preserves the
                # planner's original fused/legacy route path by default.
                planner_kwargs = dict(
                    speed=speed[valid_indices],
                    lateral_friction=friction[valid_indices],
                    acceleration=acceleration[valid_indices],
                    active_mask=route_mask, robot_obstacles=obstacle)
                if learned_map is None:
                    scratch.plan(start, goal, heading, length, width,
                                 **planner_kwargs)
                else:
                    scratch.plan(start, goal, heading, length, width,
                        learned_traversal_cost=learned_map,
                        learned_cost_scale=learned_cost_scale,
                        **planner_kwargs)
                for batch_local in torch.nonzero(route_mask, as_tuple=False).flatten().tolist():
                    original_index, query = valid_rows[valid_indices[batch_local].item()]
                    count = int(scratch.last_lengths[batch_local].item())
                    points = scratch.last_path[batch_local, :count]
                    if count <= 1:
                        result = RouteResult(query.candidate_id, query.snapshot_id,
                            query.world, query.slot, "no_path", False, None,
                            cost_provenance=cost_provenance)
                    else:
                        # AD* may start from a projected visible cell when the
                        # measured pose lies in a blocked raster cell. Include
                        # that escape segment in both clearance and distance.
                        points = torch.cat((start[batch_local:batch_local+1], points), dim=0)
                        endpoint_error = float((points[-1] - goal[batch_local]).norm().item())
                        clear = bool(scratch._footprint_path_clear(
                            points[None], heading[batch_local:batch_local+1],
                            length[batch_local:batch_local+1],
                            width[batch_local:batch_local+1])[0].item())
                        distance = float((points[1:] - points[:-1]).norm(dim=-1).sum().item())
                        endpoint_circle_clear = _pose_clear_of_circles(
                            points[-1:], heading[batch_local:batch_local+1],
                            length[batch_local:batch_local+1],
                            width[batch_local:batch_local+1],
                            torch.cat((scratch._extra_circles.reshape(1, -1, 3),
                                       obstacle[batch_local:batch_local+1]), dim=1))
                        path_points = tuple((float(x), float(y))
                                            for x, y in points.tolist())
                        weighted_cost = _path_weighted_cost(
                            scratch, points, learned_map, learned_cost_scale)
                        if (not clear or not bool(endpoint_circle_clear[0].item()) or
                                not math.isfinite(distance) or
                                endpoint_error > endpoint_tolerance_m):
                            result = RouteResult(query.candidate_id, query.snapshot_id,
                                query.world, query.slot, "no_path", False, None,
                                path_points, weighted_cost,
                                cost_provenance=cost_provenance)
                        else:
                            result = RouteResult(query.candidate_id, query.snapshot_id,
                                query.world, query.slot, "feasible", True, distance,
                                path_points, weighted_cost,
                                cost_provenance=cost_provenance)
                    results[original_index] = result

    return tuple(result for result in results if result is not None)
