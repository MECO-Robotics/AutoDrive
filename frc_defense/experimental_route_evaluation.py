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
class DeviceRouteChunkMap:
    """Deterministic global-to-chunk mapping for an ordered route support."""

    row_count: int
    spans: tuple[tuple[int, int], ...]
    global_row: torch.Tensor
    chunk_index: torch.Tensor
    local_row: torch.Tensor

    def __post_init__(self):
        if type(self.row_count) is not int or self.row_count < 0:
            raise ValueError("row_count must be a nonnegative exact integer")
        if (self.global_row.shape != (self.row_count,) or
                self.chunk_index.shape != (self.row_count,) or
                self.local_row.shape != (self.row_count,)):
            raise ValueError("chunk mapping tensors must be [row_count]")
        if any(t.dtype != torch.long for t in
               (self.global_row, self.chunk_index, self.local_row)):
            raise ValueError("chunk mapping tensors must be int64")
        if any(t.device != self.global_row.device for t in
               (self.chunk_index, self.local_row)):
            raise ValueError("chunk mapping tensors must share a device")
        expected = tuple((start, min(start + MAX_ROUTE_QUERY_BATCH, self.row_count))
                         for start in range(0, self.row_count, MAX_ROUTE_QUERY_BATCH))
        if self.spans != expected:
            raise ValueError("route chunk spans must be contiguous, ordered, and complete")


def make_device_route_chunk_map(row_count: int, *, device) -> DeviceRouteChunkMap:
    """Build host-known 64-row spans and device-resident row coordinates."""
    if type(row_count) is not int or row_count < 0:
        raise ValueError("row_count must be a nonnegative exact integer")
    spans = tuple((start, min(start + MAX_ROUTE_QUERY_BATCH, row_count))
                  for start in range(0, row_count, MAX_ROUTE_QUERY_BATCH))
    global_row = torch.arange(row_count, device=device, dtype=torch.long)
    chunk_index = torch.div(global_row, MAX_ROUTE_QUERY_BATCH,
                            rounding_mode="floor")
    local_row = global_row.remainder(MAX_ROUTE_QUERY_BATCH)
    return DeviceRouteChunkMap(row_count, spans, global_row, chunk_index, local_row)


def validate_device_route_batch(batch: "DeviceRouteBatch", queries, *, device):
    """Return an on-device scalar contract bit; malformed batches fail closed.

    Python checks cover immutable identity/order/shape metadata. Numeric validity
    stays as a device predicate and can be combined with the support mask without
    synchronizing the route decision.
    """
    items = tuple(queries)
    if not isinstance(batch, DeviceRouteBatch):
        raise TypeError("batch must be a DeviceRouteBatch")
    if batch.queries != items or len(batch.queries) != len(items):
        raise ValueError("route batch identity/order differs from requested rows")
    target = torch.device(device)
    if batch.device != target:
        raise ValueError("route batch is on the wrong device")
    if batch.cost_units != "normalized_grid_edge_units":
        return torch.zeros((), device=target, dtype=torch.bool)
    count = len(items)
    expected_shapes = ((count,), (count,), (count, 2), (count,), (count,),
                       (count,), (count,))
    actual = (batch.status_code.shape, batch.feasible.shape,
              batch.path_xy.shape[:1] + batch.path_xy.shape[2:],
              batch.path_length.shape, batch.distance_m.shape,
              batch.endpoint_error_m.shape, batch.weighted_cost_grid_edges.shape)
    if actual != expected_shapes:
        raise ValueError("route batch numeric shapes are malformed")
    rows = torch.arange(count, device=target, dtype=torch.long)
    lengths_ok = ((batch.path_length >= 0) &
                  (batch.path_length <= batch.path_xy.shape[1]))
    finite = (torch.isfinite(batch.path_xy).all() &
              torch.isfinite(batch.distance_m).all() &
              torch.isfinite(batch.endpoint_error_m).all() &
              torch.isfinite(batch.weighted_cost_grid_edges).all())
    for optional in (batch.bump_cost_grid_edges, batch.learned_cost_grid_edges):
        if optional is not None:
            finite = finite & torch.isfinite(optional).all()
    indices_ok = (batch.candidate_index == rows).all()
    status_ok = ((batch.status_code >= DeviceRouteBatch.INVALID) &
                 (batch.status_code <= DeviceRouteBatch.FEASIBLE)).all()
    feasibility_ok = (batch.feasible ==
                      ((batch.status_code == DeviceRouteBatch.AT_GOAL) |
                       (batch.status_code == DeviceRouteBatch.FEASIBLE))).all()
    return lengths_ok.all() & finite & indices_ok & status_ok & feasibility_ok


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


@dataclass(frozen=True, slots=True)
class DeviceRouteQuery:
    """Host identity for a route whose numeric goal remains on the device."""

    candidate_id: str
    snapshot_id: str
    world: int
    slot: int
    intake_extended: bool = False


@dataclass(frozen=True, slots=True)
class DeviceRouteBatch:
    """Fixed-shape numeric route result; all decision values stay device-side."""

    queries: tuple[DeviceRouteQuery, ...]
    status_code: torch.Tensor
    feasible: torch.Tensor
    path_xy: torch.Tensor
    path_length: torch.Tensor
    distance_m: torch.Tensor
    endpoint_error_m: torch.Tensor
    weighted_cost_grid_edges: torch.Tensor
    candidate_index: torch.Tensor
    cost_units: str = "normalized_grid_edge_units"
    cost_provenance: str = "legacy_bump_cost_path_integral_v2"
    bump_cost_grid_edges: torch.Tensor | None = None
    learned_cost_grid_edges: torch.Tensor | None = None

    INVALID = 0
    AT_GOAL = 1
    NO_PATH = 2
    FEASIBLE = 3

    def __post_init__(self):
        count = len(self.queries)
        if self.status_code.shape != (count,) or self.status_code.dtype != torch.long:
            raise ValueError(f"status_code must be int64 [Q], got {self.status_code.shape} {self.status_code.dtype}")
        if self.feasible.shape != (count,) or self.feasible.dtype != torch.bool:
            raise ValueError("feasible must be bool [Q]")
        if self.path_xy.ndim != 3 or self.path_xy.shape[0] != count or self.path_xy.shape[2] != 2:
            raise ValueError("path_xy must have shape [Q, P, 2]")
        if self.path_length.shape != (count,) or self.path_length.dtype != torch.long:
            raise ValueError("path_length must be int64 [Q]")
        for name, value in (("distance_m", self.distance_m),
                            ("endpoint_error_m", self.endpoint_error_m),
                            ("weighted_cost_grid_edges", self.weighted_cost_grid_edges)):
            if value.shape != (count,) or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating [Q] tensor")
        if self.candidate_index.shape != (count,) or self.candidate_index.dtype != torch.long:
            raise ValueError("candidate_index must be int64 [Q]")
        tensors = (self.feasible, self.path_xy, self.path_length, self.distance_m,
                   self.endpoint_error_m, self.weighted_cost_grid_edges,
                   self.candidate_index)
        tensors += tuple(value for value in (self.bump_cost_grid_edges,
                                              self.learned_cost_grid_edges)
                         if value is not None)
        if any(value.device != self.status_code.device for value in tensors):
            raise ValueError("DeviceRouteBatch tensors must share one device")
        for name, value in (("bump_cost_grid_edges", self.bump_cost_grid_edges),
                            ("learned_cost_grid_edges", self.learned_cost_grid_edges)):
            if value is not None and (value.shape != (count,) or
                                      not value.is_floating_point()):
                raise ValueError(f"{name} must be a floating [Q] tensor")

    @property
    def device(self):
        return self.status_code.device

    def project_results(self) -> tuple[RouteResult, ...]:
        """Materialize report records after selection, never in the live path."""
        status = self.status_code.detach().to(device="cpu").tolist()
        feasible = self.feasible.detach().to(device="cpu").tolist()
        lengths = self.path_length.detach().to(device="cpu").tolist()
        distances = self.distance_m.detach().to(device="cpu", dtype=torch.float64).tolist()
        endpoint_errors = self.endpoint_error_m.detach().to(device="cpu", dtype=torch.float64).tolist()
        costs = self.weighted_cost_grid_edges.detach().to(device="cpu", dtype=torch.float64).tolist()
        paths = self.path_xy.detach().to(device="cpu", dtype=torch.float64).tolist()
        labels = {self.INVALID: "invalid", self.AT_GOAL: "at_goal",
                  self.NO_PATH: "no_path", self.FEASIBLE: "feasible"}
        rows = []
        for index, query in enumerate(self.queries):
            code = status[index]
            count = lengths[index]
            path = tuple((point[0], point[1]) for point in paths[index][:count])
            rows.append(RouteResult(query.candidate_id, query.snapshot_id,
                query.world, query.slot, labels.get(code, "invalid"),
                bool(feasible[index]), distances[index] if feasible[index] else None,
                path if count else None, costs[index] if count else None, self.cost_units,
                self.cost_provenance))
        return tuple(rows)


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


def _path_weighted_cost_device(scratch, path_xy: torch.Tensor,
                               path_length: torch.Tensor, learned_cost,
                               scale: float) -> torch.Tensor:
    """Integrate padded route batches without converting route data to host."""
    device = path_xy.device
    dtype = torch.float64
    count, point_count = path_xy.shape[:2]
    if point_count < 2:
        return torch.zeros((count,), device=device, dtype=dtype)
    path = path_xy.to(dtype=dtype)
    terrain = scratch._bump_cost.to(device=device, dtype=dtype)
    if learned_cost is not None:
        terrain = terrain + learned_cost.to(device=device, dtype=dtype) * scale
    resolution = float(scratch.resolution)
    starts, ends = path[:, :-1], path[:, 1:]
    delta = ends - starts
    dx, dy = delta[..., 0:1], delta[..., 1:2]
    x_boundaries = (torch.arange(scratch.nx + 1, device=device, dtype=dtype) *
                    resolution).reshape(1, 1, -1)
    y_boundaries = (torch.arange(scratch.ny + 1, device=device, dtype=dtype) *
                    resolution).reshape(1, 1, -1)
    safe_dx = torch.where(dx != 0., dx, torch.ones_like(dx))
    safe_dy = torch.where(dy != 0., dy, torch.ones_like(dy))
    tx = (x_boundaries - starts[..., 0:1]) / safe_dx
    ty = (y_boundaries - starts[..., 1:2]) / safe_dy
    tx = torch.where((dx != 0.) & (tx > 0.) & (tx < 1.), tx,
                     torch.ones_like(tx))
    ty = torch.where((dy != 0.) & (ty > 0.) & (ty < 1.), ty,
                     torch.ones_like(ty))
    zero = torch.zeros_like(tx[..., :1])
    one = torch.ones_like(tx[..., :1])
    crossings = torch.cat((zero, one, tx, ty), dim=-1).sort(dim=-1).values
    begin, end = crossings[..., :-1], crossings[..., 1:]
    midpoint = (begin + end) * .5
    location = starts[..., None, :] + midpoint[..., None] * delta[..., None, :]
    ix = torch.floor(location[..., 0] / resolution).to(torch.long).clamp(0, scratch.nx - 1)
    iy = torch.floor(location[..., 1] / resolution).to(torch.long).clamp(0, scratch.ny - 1)
    flat_index = ix * scratch.ny + iy
    values = terrain.reshape(-1)[flat_index]
    interval = (values * (end - begin)).sum(dim=-1)
    segment_steps = torch.linalg.vector_norm(delta, dim=-1) / resolution
    segment_valid = (torch.arange(point_count - 1, device=device)[None, :] <
                     (path_length - 1).clamp_min(0)[:, None])
    per_segment = torch.where(segment_valid, interval * segment_steps,
                              torch.zeros_like(interval))
    return per_segment.sum(dim=-1)


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
def evaluate_device_routes(env, queries: tuple[DeviceRouteQuery, ...] | list[DeviceRouteQuery],
                           goal_xy: torch.Tensor, *, goal_tolerance_m: float = 0.05,
                           endpoint_tolerance_m: float = 0.05,
                           learned_traversal_cost=None,
                           learned_cost_scale: float = 1.0,
                           learned_cost_provenance: str = "caller_supplied") -> DeviceRouteBatch:
    """Evaluate numeric route goals on device and retain fixed-shape tensor rows.

    Query identity and order are immutable host metadata. Goal coordinates,
    route status, path, endpoint checks, distance, and weighted cost remain
    tensors on ``env.device``. Call ``DeviceRouteBatch.project_results`` only
    after the decision when report-compatible ``RouteResult`` objects are
    needed.
    """
    items = tuple(queries)
    device = torch.device(env.device)
    if not math.isfinite(goal_tolerance_m) or goal_tolerance_m < 0:
        raise ValueError("goal_tolerance_m must be finite and nonnegative")
    if not math.isfinite(endpoint_tolerance_m) or endpoint_tolerance_m < 0:
        raise ValueError("endpoint_tolerance_m must be finite and nonnegative")
    learned_cost_scale = float(learned_cost_scale)
    if not math.isfinite(learned_cost_scale) or learned_cost_scale < 0:
        raise ValueError("learned_cost_scale must be finite and nonnegative")
    if not isinstance(learned_cost_provenance, str) or not learned_cost_provenance:
        raise ValueError("learned_cost_provenance must be a nonempty string")
    if not items:
        empty_float = torch.empty((0,), device=device, dtype=torch.float64)
        return DeviceRouteBatch((), torch.empty((0,), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.bool),
            torch.empty((0, 0, 2), device=device, dtype=env.policy_pose.dtype),
            torch.empty((0,), device=device, dtype=torch.long), empty_float,
            empty_float.clone(), empty_float.clone(),
            torch.empty((0,), device=device, dtype=torch.long),
            cost_provenance=("legacy_bump_cost_path_integral_v2"
                if learned_traversal_cost is None else learned_cost_provenance),
            bump_cost_grid_edges=empty_float.clone(),
            learned_cost_grid_edges=empty_float.clone())
    if len(items) > MAX_ROUTE_QUERY_BATCH:
        raise ValueError(f"route batch exceeds {MAX_ROUTE_QUERY_BATCH} queries")
    if any(not isinstance(query, DeviceRouteQuery) for query in items):
        raise TypeError("queries must contain DeviceRouteQuery values")
    snapshot_ids = {query.snapshot_id for query in items
                    if isinstance(query.snapshot_id, str)}
    if len(snapshot_ids) > 1:
        raise ValueError("one route batch must use a single snapshot_id")
    seen = set()
    metadata_valid = []
    worlds, slots, intake = [], [], []
    for query in items:
        key = ((query.world, query.slot, query.candidate_id)
               if isinstance(query.candidate_id, str) and type(query.world) is int
               and type(query.slot) is int else None)
        valid = (isinstance(query.candidate_id, str) and bool(query.candidate_id)
                 and isinstance(query.snapshot_id, str) and bool(query.snapshot_id)
                 and type(query.world) is int and 0 <= query.world < env.n
                 and type(query.slot) is int and 0 <= query.slot < 6
                 and type(query.intake_extended) is bool and key is not None
                 and key not in seen)
        if valid:
            seen.add(key)
        metadata_valid.append(valid)
        worlds.append(query.world if valid else 0)
        slots.append(query.slot if valid else 0)
        intake.append(query.intake_extended if valid else False)
    count = len(items)
    if (not isinstance(goal_xy, torch.Tensor) or goal_xy.shape != (count, 2)
            or goal_xy.device != device or not goal_xy.is_floating_point()):
        raise ValueError("goal_xy must be floating [Q,2] on env.device")
    goals = goal_xy.to(dtype=env.policy_pose.dtype)
    world_index = torch.tensor(worlds, device=device, dtype=torch.long)
    slot_index = torch.tensor(slots, device=device, dtype=torch.long)
    query_mask = torch.tensor(metadata_valid, device=device, dtype=torch.bool)
    intake_mask = torch.tensor(intake, device=device, dtype=torch.bool)
    obstacle_snapshot = env._perceived_robot_obstacles()
    flat_robot = world_index * 6 + slot_index
    starts = env.policy_pose[world_index, slot_index, :2]
    headings = env.policy_pose[world_index, slot_index, 2]
    lengths = env.sim.length[world_index, slot_index] + intake_mask.to(
        env.sim.length.dtype) * 0.6096
    widths = env.sim.width[world_index, slot_index]
    speed = env.sim.speed[world_index, slot_index]
    friction = env.sim.lateral_mu[world_index, slot_index]
    acceleration = env.sim.accel[world_index, slot_index]
    obstacles = obstacle_snapshot[flat_robot]
    inside_goal = ((goals[:, 0] >= 0.) & (goals[:, 0] <= float(env.sim.field_length)) &
                   (goals[:, 1] >= 0.) & (goals[:, 1] <= float(env.sim.field_width)))
    finite = (torch.isfinite(starts).all(-1) & torch.isfinite(headings) &
        torch.isfinite(goals).all(-1) & torch.isfinite(lengths) &
        torch.isfinite(widths) & torch.isfinite(speed) &
        torch.isfinite(friction) & torch.isfinite(acceleration) &
        torch.isfinite(obstacles).all(dim=(-1, -2)) & inside_goal &
        (lengths > 0.) & (widths > 0.) & (speed > 0.) &
        (friction > 0.) & (acceleration > 0.) &
        (starts[:, 0] >= 0.) & (starts[:, 0] <= float(env.sim.field_length)) &
        (starts[:, 1] >= 0.) & (starts[:, 1] <= float(env.sim.field_width)))
    active = query_mask & finite
    safe_start = torch.nan_to_num(starts, nan=0., posinf=0., neginf=0.)
    safe_goal = torch.nan_to_num(goals, nan=0., posinf=0., neginf=0.)
    safe_heading = torch.nan_to_num(headings, nan=0., posinf=0., neginf=0.)
    safe_length = torch.where(torch.isfinite(lengths) & (lengths > 0.), lengths,
                              torch.ones_like(lengths))
    safe_width = torch.where(torch.isfinite(widths) & (widths > 0.), widths,
                             torch.ones_like(widths))
    safe_obstacles = torch.nan_to_num(obstacles, nan=0., posinf=0., neginf=0.)
    goal_distance = torch.linalg.vector_norm(safe_goal - safe_start, dim=-1)
    at_goal = active & (goal_distance <= goal_tolerance_m)
    safe_queries = tuple(DeviceRouteQuery(query.candidate_id, query.snapshot_id,
        worlds[index], slots[index], intake[index])
        for index, query in enumerate(items))
    query_rows = [(query, safe_goal[index])
                  for index, query in enumerate(safe_queries)]
    scratch = _scratch_planner(env, query_rows)
    gx, gy = scratch._indices(safe_goal)
    blocked = scratch._blocked(safe_heading, safe_length, safe_width)
    blocked = scratch._block_robot_obstacles(blocked, safe_length, safe_width,
                                             safe_obstacles)
    row_index = torch.arange(count, device=device)
    goal_blocked = blocked[row_index, gx, gy]
    at_goal_clear = at_goal & _pose_clear(scratch, safe_start, safe_heading,
        safe_length, safe_width, safe_obstacles)
    at_goal_clear &= _pose_clear(scratch, safe_goal, safe_heading,
        safe_length, safe_width, safe_obstacles)
    route_active = active & ~at_goal & ~goal_blocked
    planner_kwargs = dict(speed=torch.where(torch.isfinite(speed) & (speed > 0.),
        speed, torch.ones_like(speed)), lateral_friction=torch.where(
        torch.isfinite(friction) & (friction > 0.), friction,
        torch.ones_like(friction)), acceleration=torch.where(
        torch.isfinite(acceleration) & (acceleration > 0.), acceleration,
        torch.ones_like(acceleration)), active_mask=route_active,
        robot_obstacles=safe_obstacles)
    learned_map = None
    cost_provenance = "legacy_bump_cost_path_integral_v2"
    if learned_traversal_cost is not None:
        learned_map = torch.as_tensor(learned_traversal_cost, device=device,
                                      dtype=env.policy_pose.dtype)
        if learned_map.shape != (env.planner.nx, env.planner.ny):
            raise ValueError("learned_traversal_cost has the wrong route-grid shape")
        cost_provenance = learned_cost_provenance
        scratch.plan(safe_start, safe_goal, safe_heading, safe_length, safe_width,
            learned_traversal_cost=learned_map,
            learned_cost_scale=learned_cost_scale, **planner_kwargs)
    else:
        scratch.plan(safe_start, safe_goal, safe_heading, safe_length, safe_width,
                     **planner_kwargs)
    planned_length = scratch.last_lengths
    route_has_path = route_active & (planned_length > 1)
    route_diagnostic = route_active & (planned_length > 1)
    raw_path = torch.cat((safe_start[:, None, :], scratch.last_path), dim=1)
    path_length = torch.where(route_diagnostic, planned_length + 1,
                              torch.zeros_like(planned_length))
    path_length = torch.where(at_goal_clear, torch.ones_like(path_length), path_length)
    at_goal_path = safe_start[:, None, :].expand(-1, raw_path.shape[1], -1)
    path_xy = torch.where(at_goal_clear[:, None, None], at_goal_path, raw_path)
    last_index = (planned_length - 1).clamp(0, scratch.max_points - 1)
    endpoint = scratch.last_path.gather(1,
        last_index[:, None, None].expand(-1, 1, 2)).squeeze(1)
    endpoint = torch.where(at_goal_clear[:, None], safe_start, endpoint)
    tail_index = torch.arange(path_xy.shape[1], device=device)[None, :]
    padded_tail = tail_index >= path_length[:, None]
    path_xy = torch.where(padded_tail[:, :, None], endpoint[:, None, :], path_xy)
    endpoint_error = torch.linalg.vector_norm(endpoint - safe_goal, dim=-1)
    path_for_clear = torch.where((path_length > 0)[:, None, None], path_xy,
                                 safe_start[:, None, :].expand_as(path_xy))
    static_clear = scratch._footprint_path_clear(path_for_clear, safe_heading,
        safe_length, safe_width)
    endpoint_circles = torch.cat((scratch._extra_circles.reshape(1, -1, 3).expand(
        count, -1, -1), safe_obstacles), dim=1)
    circle_clear = _pose_clear_of_circles(endpoint,
        safe_heading, safe_length, safe_width, endpoint_circles)
    segment_lengths = torch.linalg.vector_norm(path_xy[:, 1:] - path_xy[:, :-1], dim=-1)
    segment_mask = (torch.arange(path_xy.shape[1] - 1, device=device)[None, :] <
                    (path_length - 1).clamp_min(0)[:, None])
    distance = torch.where(segment_mask, segment_lengths,
                           torch.zeros_like(segment_lengths)).sum(dim=-1)
    endpoint_ok = endpoint_error <= endpoint_tolerance_m
    route_feasible = (route_has_path & static_clear & circle_clear & endpoint_ok &
                      torch.isfinite(distance))
    feasible = at_goal_clear | route_feasible
    status = torch.full((count,), DeviceRouteBatch.NO_PATH, device=device,
                        dtype=torch.long)
    status = torch.where(~active, torch.full_like(status, DeviceRouteBatch.INVALID), status)
    status = torch.where(at_goal_clear, torch.full_like(status, DeviceRouteBatch.AT_GOAL), status)
    status = torch.where(route_feasible, torch.full_like(status, DeviceRouteBatch.FEASIBLE), status)
    path_length = torch.where(~active, torch.zeros_like(path_length), path_length)
    path_length = torch.where(route_has_path | at_goal_clear, path_length,
                              torch.zeros_like(path_length))
    path_xy = torch.where((path_length > 0)[:, None, None], path_xy,
                          torch.zeros_like(path_xy))
    distance = torch.where(feasible, distance, distance)
    endpoint_error = torch.where(at_goal_clear, torch.zeros_like(endpoint_error),
                                 endpoint_error)
    bump_cost = _path_weighted_cost_device(scratch, path_xy, path_length,
                                           None, 1.)
    if learned_map is not None:
        weighted_cost = _path_weighted_cost_device(scratch, path_xy, path_length,
                                                   learned_map, learned_cost_scale)
        learned_cost = weighted_cost - bump_cost
    else:
        weighted_cost = bump_cost
        learned_cost = torch.zeros_like(weighted_cost)
    return DeviceRouteBatch(items, status, feasible, path_xy, path_length,
        distance.to(torch.float64), endpoint_error.to(torch.float64),
        weighted_cost, torch.arange(count, device=device, dtype=torch.long),
        cost_provenance=cost_provenance, bump_cost_grid_edges=bump_cost,
        learned_cost_grid_edges=learned_cost)


@torch.no_grad()
def evaluate_device_route_chunks(env, queries, goal_xy: torch.Tensor, **kwargs):
    """Evaluate ordered device goals in bounded planner batches and concatenate tensors."""
    items = tuple(queries)
    if (not isinstance(goal_xy, torch.Tensor) or goal_xy.ndim != 2
            or goal_xy.shape != (len(items), 2)):
        raise ValueError("goal_xy must be floating [Q,2] aligned with queries")
    if goal_xy.device != torch.device(env.device) or not goal_xy.is_floating_point():
        raise ValueError("goal_xy must be floating on env.device")
    mapping = make_device_route_chunk_map(len(items), device=goal_xy.device)
    chunks = []
    all_chunks_valid = torch.ones((), device=goal_xy.device, dtype=torch.bool)
    common_metadata = None
    for start, stop in mapping.spans:
        chunk_queries = items[start:stop]
        chunk = evaluate_device_routes(env, chunk_queries,
            goal_xy[start:stop], **kwargs)
        contract_ok = validate_device_route_batch(chunk, chunk_queries,
                                                  device=goal_xy.device)
        metadata = (chunk.cost_units, chunk.cost_provenance,
                    chunk.bump_cost_grid_edges is not None,
                    chunk.learned_cost_grid_edges is not None)
        if common_metadata is None:
            common_metadata = metadata
        elif metadata != common_metadata:
            contract_ok = torch.zeros_like(contract_ok)
        if chunk.cost_units != "normalized_grid_edge_units":
            contract_ok = torch.zeros_like(contract_ok)
        all_chunks_valid = all_chunks_valid & contract_ok
        # A malformed or non-finite chunk invalidates the complete route batch.
        # Keep that decision on device so no good chunk can remain selectable.
        chunk = DeviceRouteBatch(chunk.queries,
            torch.where(all_chunks_valid, chunk.status_code,
                        torch.full_like(chunk.status_code, DeviceRouteBatch.INVALID)),
            chunk.feasible & all_chunks_valid, chunk.path_xy, chunk.path_length,
            chunk.distance_m, chunk.endpoint_error_m,
            chunk.weighted_cost_grid_edges,
            torch.arange(stop - start, device=goal_xy.device, dtype=torch.long),
            cost_units=chunk.cost_units,
            cost_provenance=chunk.cost_provenance,
            bump_cost_grid_edges=chunk.bump_cost_grid_edges,
            learned_cost_grid_edges=chunk.learned_cost_grid_edges)
        chunks.append(chunk)
    if not chunks:
        return evaluate_device_routes(env, (), goal_xy, **kwargs)
    def concatenate_optional(name):
        values = [getattr(batch, name) for batch in chunks]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"route chunks disagree on optional {name} output")
        return torch.cat(values)
    result = DeviceRouteBatch(items,
        torch.cat([batch.status_code for batch in chunks]),
        torch.cat([batch.feasible for batch in chunks]),
        torch.cat([batch.path_xy for batch in chunks]),
        torch.cat([batch.path_length for batch in chunks]),
        torch.cat([batch.distance_m for batch in chunks]),
            torch.cat([batch.endpoint_error_m for batch in chunks]),
            torch.cat([batch.weighted_cost_grid_edges for batch in chunks]),
            mapping.global_row, cost_units=chunks[0].cost_units,
            cost_provenance=chunks[0].cost_provenance,
            bump_cost_grid_edges=concatenate_optional("bump_cost_grid_edges"),
            learned_cost_grid_edges=concatenate_optional("learned_cost_grid_edges"))
    result_contract_ok = validate_device_route_batch(result, items,
                                                      device=goal_xy.device)
    batch_valid = all_chunks_valid & result_contract_ok
    return DeviceRouteBatch(result.queries,
        torch.where(batch_valid, result.status_code,
                    torch.full_like(result.status_code, DeviceRouteBatch.INVALID)),
        result.feasible & batch_valid, result.path_xy,
        result.path_length, result.distance_m, result.endpoint_error_m,
        result.weighted_cost_grid_edges, result.candidate_index,
        cost_units=result.cost_units,
        cost_provenance=result.cost_provenance,
        bump_cost_grid_edges=result.bump_cost_grid_edges,
        learned_cost_grid_edges=result.learned_cost_grid_edges)


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
