"""PathPlanner LocalADStar-style pathfinding and holonomic path following.

The AD* search and Bézier waypoint construction follow PathPlanner's
``LocalADStar`` implementation (MIT; copyright Michael Jansen, 2022), adapted
for this simulator's field boxes, bump speed zones, and drivetrain limits.
Source: https://github.com/mjansen4857/pathplanner/tree/main/pathplannerlib-python/pathplannerlib/pathfinders.py
"""
from __future__ import annotations

from functools import lru_cache
from math import ceil, floor
from heapq import heappop, heappush
from math import atan2, cos, hypot, sin

from .field import BUMP_RAMP_RISE, BUMP_ROLLING_RESISTANCE


TURN_LATERAL_ACCEL_SCALE = .65
BRAKING_ACCEL_SCALE = .65


@lru_cache(maxsize=32)
def _static_planner_grids(length, width, resolution, robot_length,
                          robot_width, colliders, bumps):
    """Share immutable field grids between equivalent local repair planners."""
    nx, ny = int(ceil(length / resolution)), int(ceil(width / resolution))
    radius = min(robot_length, robot_width) / 2
    cell_pad = resolution / 2
    blocked = []
    bump_grid = []
    for ix in range(nx):
        x = (ix + .5) * resolution
        blocked_col = []
        bump_col = []
        for iy in range(ny):
            y = (iy + .5) * resolution
            blocked_cell = (x < radius or x > length - radius or
                            y < radius or y > width - radius)
            if not blocked_cell:
                for cx, cy, hx, hy in colliders:
                    nearest_x = max(abs(x - cx) - hx, 0.)
                    nearest_y = max(abs(y - cy) - hy, 0.)
                    if (nearest_x * nearest_x + nearest_y * nearest_y <=
                            (radius + cell_pad) ** 2):
                        blocked_cell = True
                        break
            blocked_col.append(blocked_cell)
            bump_col.append(any(abs(x - cx) <= hx and abs(y - cy) <= hy
                                for cx, cy, hx, hy in bumps))
        blocked.append(tuple(blocked_col))
        bump_grid.append(tuple(bump_col))
    return tuple(blocked), tuple(bump_grid)


def _box_geometry(box):
    if hasattr(box, "length"):
        return (float(box.x), float(box.y), float(box.length) / 2,
                float(box.width) / 2)
    return tuple(float(value) for value in box)


class ADStarPlanner:
    """Incremental Anytime Dynamic A* on a 8-connected field grid."""

    def __init__(self, length: float, width: float, colliders=(), bumps=(), *,
                 resolution: float = .2, robot_length: float = .9,
                 robot_width: float = .9, robot_radius: float | None = None,
                 robot_heading: float = 0.,
                 max_speed: float = 4.8,
                 max_acceleration: float = 8., steering_rate: float = 12.,
                 lateral_friction: float = 1.2,
                 max_angular_speed: float = 8.,
                 max_angular_acceleration: float = 18.):
        self.length, self.width = float(length), float(width)
        self.resolution = float(resolution)
        self.robot_length = max(.01, float(robot_length))
        self.robot_width = max(.01, float(robot_width))
        self.robot_heading = float(robot_heading)
        self.robot_radius = float(robot_radius or 0.)
        self.nx = int(ceil(self.length / self.resolution))
        self.ny = int(ceil(self.width / self.resolution))
        self.colliders = tuple(colliders)
        self.bumps = tuple(bumps)
        self.max_speed = max(.1, float(max_speed))
        self.max_acceleration = max(.1, float(max_acceleration))
        self.steering_rate = max(.1, float(steering_rate))
        self.lateral_friction = max(.1, float(lateral_friction))
        self.max_angular_speed = max(.1, float(max_angular_speed))
        self.max_angular_acceleration = max(.1, float(max_angular_acceleration))
        static_colliders = tuple(_box_geometry(box) for box in self.colliders)
        static_bumps = tuple(_box_geometry(box) for box in self.bumps)
        self._blocked, self._bump = _static_planner_grids(
            self.length, self.width, self.resolution, self.robot_length,
            self.robot_width, static_colliders, static_bumps)
        self._dynamic = ()
        self._g = {}
        self._rhs = {}
        self._open = {}
        self._closed = set()
        self._incons = set()
        self._heap = []
        self._serial = 0
        self._start = self._goal = None
        self._epsilon = 2.5
        self._neighbors = ((1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.),
                           (1, 1, 1.41421356237), (1, -1, 1.41421356237),
                           (-1, 1, 1.41421356237), (-1, -1, 1.41421356237))
        self.last_poses = []

    @staticmethod
    def _inside_box(x, y, box, margin):
        # FieldBox instances and simulator (cx, cy, half_x, half_y) tuples.
        if hasattr(box, "length"):
            cx, cy, hx, hy = box.x, box.y, box.length / 2, box.width / 2
        else:
            cx, cy, hx, hy = box
        return abs(x - cx) <= hx + margin and abs(y - cy) <= hy + margin

    def _rect_hits_box(self, x, y, box):
        """SAT overlap of the oriented bumper and a collider expanded by half a grid cell."""
        if hasattr(box, "length"):
            cx, cy, hx, hy = box.x, box.y, box.length / 2, box.width / 2
        else:
            cx, cy, hx, hy = box
        # Expanding the axis-aligned box by half a cell covers every point in
        # the cell, preventing a smoothed route from clipping between samples.
        pad = self.resolution / 2
        hx, hy = hx + pad, hy + pad
        dx, dy = x - cx, y - cy
        c, s = cos(self.robot_heading), sin(self.robot_heading)
        half_l, half_w = self.robot_length / 2, self.robot_width / 2
        return (abs(dx) <= hx + half_l * abs(c) + half_w * abs(s) and
                abs(dy) <= hy + half_l * abs(s) + half_w * abs(c) and
                abs(dx * c + dy * s) <= half_l + hx * abs(c) + hy * abs(s) and
                abs(-dx * s + dy * c) <= half_w + hx * abs(s) + hy * abs(c))

    def _pose_clear(self, x, y, heading=None):
        """Exact rotated-rectangle clearance at a predicted chassis pose."""
        heading = self.robot_heading if heading is None else heading
        c, s = cos(heading), sin(heading)
        half_l, half_w = self.robot_length / 2, self.robot_width / 2
        extent_x = half_l * abs(c) + half_w * abs(s)
        extent_y = half_l * abs(s) + half_w * abs(c)
        if (x < extent_x or x > self.length - extent_x or
                y < extent_y or y > self.width - extent_y):
            return False
        # Circumscribed and inscribed circles cheaply classify clear and
        # definitely-colliding cases. SAT is reserved for the ambiguous band.
        robot_outer = hypot(half_l, half_w)
        robot_inner = min(half_l, half_w)
        for box in (*self.colliders, *self._dynamic):
            if hasattr(box, "length"):
                cx, cy, hx, hy = box.x, box.y, box.length / 2, box.width / 2
            else:
                cx, cy, hx, hy = box
            dx, dy = x - cx, y - cy
            nearest_x = max(abs(dx) - hx, 0.)
            nearest_y = max(abs(dy) - hy, 0.)
            separation = hypot(nearest_x, nearest_y)
            if separation > robot_outer + hypot(hx, hy):
                continue
            if separation <= robot_inner:
                return False
            if (abs(dx * c + dy * s) > half_l + hx * abs(c) + hy * abs(s) or
                    abs(dx * -s + dy * c) > half_w + hx * abs(s) + hy * abs(c)):
                continue
            return False
        return True

    def _swept_rect_clear(self, start, end, start_heading=None, end_heading=None):
        """Sweep the predicted oriented bumper along a trajectory segment.

        When headings are omitted, predict that the chassis follows the local
        travel tangent. Explicit headings allow callers to include rotation
        through a corner as well as translation.
        """
        distance = hypot(end[0] - start[0], end[1] - start[1])
        tangent = atan2(end[1] - start[1], end[0] - start[0])
        start_heading = tangent if start_heading is None else start_heading
        end_heading = tangent if end_heading is None else end_heading
        heading_delta = atan2(sin(end_heading - start_heading),
                              cos(end_heading - start_heading))
        # Small steps bound translation between exact SAT checks; route samples
        # and grid cells can otherwise skip a thin support on a long segment.
        count = max(1, int(ceil(max(distance, abs(heading_delta) *
                                    hypot(self.robot_length, self.robot_width) / 2) /
                                min(.025, self.resolution / 8))))
        for i in range(count + 1):
            t = i / count
            if not self._pose_clear(start[0] + (end[0] - start[0]) * t,
                                    start[1] + (end[1] - start[1]) * t,
                                    start_heading + heading_delta * t):
                return False
        return True

    def _trajectory_clear(self, points):
        """Validate a path using its predicted chassis tangent at each point."""
        if len(points) < 2:
            return bool(points) and self._pose_clear(*points[0])
        return self._pose_path_clear(self._poses_for_path(points))

    def _poses_for_path(self, points):
        poses = []
        for i, point in enumerate(points):
            before, after = points[max(0, i - 1)], points[min(len(points) - 1, i + 1)]
            dx, dy = after[0] - before[0], after[1] - before[1]
            heading = atan2(dy, dx) if hypot(dx, dy) > 1e-9 else self.robot_heading
            poses.append((float(point[0]), float(point[1]), heading))
        if poses:
            poses[0] = (poses[0][0], poses[0][1], self.robot_heading)
        return poses

    def _pose_path_clear(self, poses):
        return (bool(poses) and all(
            self._swept_rect_clear(a[:2], b[:2], a[2], b[2])
            for a, b in zip(poses, poses[1:])) and
            all(self._pose_clear(*pose) for pose in poses))

    def _first_collision(self, poses):
        for i, (a, b) in enumerate(zip(poses, poses[1:])):
            distance = hypot(b[0] - a[0], b[1] - a[1])
            dtheta = atan2(sin(b[2] - a[2]), cos(b[2] - a[2]))
            count = max(1, int(ceil(max(distance, abs(dtheta) *
                                        hypot(self.robot_length, self.robot_width) / 2) /
                                    min(.025, self.resolution / 8))))
            for j in range(count + 1):
                t = j / count
                pose = (a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t, a[2] + dtheta * t)
                if not self._pose_clear(*pose):
                    return i, t, pose
        return None

    @staticmethod
    def _motion_time(distance, speed_limit, acceleration, initial_speed=0.):
        distance = max(0., float(distance))
        vmax, accel = max(.1, float(speed_limit)), max(.1, float(acceleration))
        v0 = min(vmax, max(0., float(initial_speed)))
        accelerate_distance = max(0., (vmax * vmax - v0 * v0) / (2 * accel))
        if distance <= accelerate_distance:
            return (hypot(v0, (2 * accel * distance) ** .5) - v0) / accel
        return (vmax - v0) / accel + (distance - accelerate_distance) / vmax

    def _repair_heading(self, poses, collision, initial_velocity,
                        initial_angular_velocity):
        i, t, pose = collision
        inserted = list(poses)
        at = i + 1
        base = pose[2]
        inserted.insert(at, pose)
        period = (1.5707963267948966 if
                  abs(self.robot_length - self.robot_width) <= 1e-5 else
                  3.141592653589793)
        best = None
        degree_step = .5
        for step_index in range(1, int(ceil(period * 180 /
                                            (3.141592653589793 * degree_step))) + 1):
            degree = step_index * degree_step
            for direction in (-1., 1.):
                candidate = list(inserted)
                heading = base + direction * degree * 3.141592653589793 / 180
                candidate[at] = (pose[0], pose[1], heading)
                lo, hi = max(0, at - 2), min(len(candidate), at + 3)
                if not self._pose_path_clear(candidate[lo:hi]):
                    continue
                delta = degree * 3.141592653589793 / 180
                rotation_time = self._motion_time(delta, self.max_angular_speed,
                                                  self.max_angular_acceleration,
                                                  initial_angular_velocity * direction)
                span = hypot(candidate[at + 1][0] - candidate[at - 1][0],
                             candidate[at + 1][1] - candidate[at - 1][1])
                if span > 1e-9:
                    ux = (candidate[at + 1][0] - candidate[at - 1][0]) / span
                    uy = (candidate[at + 1][1] - candidate[at - 1][1]) / span
                    translation_speed = max(0., initial_velocity[0] * ux +
                                            initial_velocity[1] * uy)
                else:
                    translation_speed = 0.
                translation_time = self._motion_time(
                    span, self.max_speed, self.max_acceleration,
                    translation_speed)
                time = max(rotation_time, translation_time)
                if best is None or time < best[0]:
                    best = (time, candidate)
            if best:
                break
        return best

    def _repair_translation(self, points, collision, initial_velocity,
                            initial_angular_velocity):
        i, t, pose = collision
        a, b = points[i], points[i + 1]
        tangent = atan2(b[1] - a[1], b[0] - a[0])
        normals = [(cos(tangent + sign * 1.5707963267948966),
                    sin(tangent + sign * 1.5707963267948966)) for sign in (-1, 1)]
        # Prefer directions away from the nearest obstacle face.
        nearest = None
        for box in (*self.colliders, *self._dynamic):
            if hasattr(box, "length"):
                cx, cy, hx, hy = box.x, box.y, box.length / 2, box.width / 2
            else:
                cx, cy, hx, hy = box
            qx = max(cx - hx, min(pose[0], cx + hx))
            qy = max(cy - hy, min(pose[1], cy + hy))
            dx, dy = pose[0] - qx, pose[1] - qy
            distance = hypot(dx, dy)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, dx, dy, pose[0] - cx, pose[1] - cy, hx, hy)
        if nearest and nearest[0] > 1e-8:
            normals.insert(0, (nearest[1] / nearest[0], nearest[2] / nearest[0]))
        elif nearest:
            _, _, _, dx, dy, hx, hy = nearest
            faces = ((hx - dx, (1., 0.)), (hx + dx, (-1., 0.)),
                     (hy - dy, (0., 1.)), (hy + dy, (0., -1.)))
            normals.insert(0, min(faces, key=lambda item: item[0])[1])
        half_l, half_w = self.robot_length / 2, self.robot_width / 2
        c, sn = cos(pose[2]), sin(pose[2])
        ex, ey = half_l * abs(c) + half_w * abs(sn), half_l * abs(sn) + half_w * abs(c)
        boundary_margins = ((pose[0] - ex, (1., 0.)),
                            (self.length - ex - pose[0], (-1., 0.)),
                            (pose[1] - ey, (0., 1.)),
                            (self.width - ey - pose[1], (0., -1.)))
        nearest_boundary = min(boundary_margins, key=lambda item: item[0])
        if nearest_boundary[0] < self.robot_length:
            normals.insert(0, nearest_boundary[1])
        best = None
        for normal in normals:
            for step in range(1, 41):
                displacement = step * .025
                candidate_points = list(points)
                candidate_points.insert(i + 1,
                    (pose[0] + normal[0] * displacement,
                     pose[1] + normal[1] * displacement))
                candidate_poses = self._poses_for_path(candidate_points)
                local = candidate_poses[max(0, i - 1):min(len(candidate_poses), i + 4)]
                if not self._pose_path_clear(local):
                    continue
                extra = (hypot(candidate_points[i][0] - candidate_points[i + 1][0],
                               candidate_points[i][1] - candidate_points[i + 1][1]) +
                         hypot(candidate_points[i + 1][0] - candidate_points[i + 2][0],
                               candidate_points[i + 1][1] - candidate_points[i + 2][1]) -
                         hypot(candidate_points[i][0] - candidate_points[i + 2][0],
                               candidate_points[i][1] - candidate_points[i + 2][1]))
                segment_length = hypot(b[0] - a[0], b[1] - a[1])
                initial_speed = (max(0., (initial_velocity[0] * (b[0] - a[0]) +
                                          initial_velocity[1] * (b[1] - a[1])) /
                                         segment_length)
                                 if segment_length > 1e-9 else 0.)
                translation_time = self._motion_time(
                    extra, self.max_speed, self.max_acceleration, initial_speed)
                old_poses = self._poses_for_path(points)
                angle_changes = []
                for candidate_index in range(max(0, i - 1),
                                             min(len(candidate_poses), i + 4)):
                    if candidate_index == i + 1:
                        old_heading = pose[2]
                    else:
                        old_index = candidate_index - 1 if candidate_index > i + 1 else candidate_index
                        old_heading = old_poses[min(old_index, len(old_poses) - 1)][2]
                    new_heading = candidate_poses[candidate_index][2]
                    angle_changes.append(atan2(sin(new_heading - old_heading),
                                               cos(new_heading - old_heading)))
                signed_delta = max(angle_changes, key=abs, default=0.)
                delta = abs(signed_delta)
                rotation_time = self._motion_time(
                    delta, self.max_angular_speed, self.max_angular_acceleration,
                    max(0., initial_angular_velocity * (1. if signed_delta >= 0 else -1.)))
                time = max(translation_time, rotation_time)
                if best is None or time < best[0]:
                    best = (time, candidate_points, candidate_poses)
                break
        return best

    def _local_se2_repair(self, poses, collision, expansion_budget=3500):
        """Search a small orientation-aware patch and splice it into the route."""
        i, _, hit = collision
        entry_index = max(0, i - 5)
        exit_index = min(len(poses) - 1, i + 6)
        entry, exit = poses[entry_index], poses[exit_index]
        radius = max(.8, 4 * self.resolution)
        x0, y0 = self._cell(entry[:2])
        x1, y1 = self._cell(exit[:2])
        hc = self._cell(hit[:2])
        cells = int(ceil(radius / self.resolution))
        xmin = max(0, min(x0, x1, hc[0]) - cells)
        xmax = min(self.nx - 1, max(x0, x1, hc[0]) + cells)
        ymin = max(0, min(y0, y1, hc[1]) - cells)
        ymax = min(self.ny - 1, max(y0, y1, hc[1]) + cells)
        period = (1.5707963267948966 if
                  abs(self.robot_length - self.robot_width) <= 1e-5 else
                  3.141592653589793)
        bins = 8
        step = period / bins
        start_angle = entry[2] % period
        start_bin = int(round(start_angle / step)) % bins
        start_theta = entry[2] + (start_bin * step - start_angle + period / 2) % period - period / 2
        start_center = self._cell_center((x0, y0))
        if not self._swept_rect_clear(entry[:2], start_center,
                                      entry[2], start_theta):
            return None
        start_state = (x0, y0, start_bin)
        frontier, costs, parent, closed = [], {start_state: 0.}, {}, set()
        headings = {start_state: start_theta}
        serial = 0

        def heuristic(ix, iy):
            dx, dy = abs(ix - x1), abs(iy - y1)
            return self.resolution * (max(dx, dy) + .41421356237 * min(dx, dy))

        heappush(frontier, (heuristic(x0, y0), serial, start_state))
        terminal = None
        expanded = 0
        while frontier and expanded < expansion_budget:
            _, _, state = heappop(frontier)
            if state in closed:
                continue
            closed.add(state)
            ix, iy, it = state
            point = self._cell_center((ix, iy))
            theta = headings[state]
            if (ix, iy) == (x1, y1) and self._swept_rect_clear(point, exit[:2], theta, exit[2]):
                terminal = state
                break
            expanded += 1
            for dx, dy, scale in self._neighbors:
                nx, ny = ix + dx, iy + dy
                if not (xmin <= nx <= xmax and ymin <= ny <= ymax):
                    continue
                target = self._cell_center((nx, ny))
                if not self._swept_rect_clear(point, target, theta, theta):
                    continue
                nxt, edge_cost = (nx, ny, it), scale * self.resolution
                value = costs[state] + edge_cost
                if value < costs.get(nxt, float("inf")):
                    costs[nxt], parent[nxt] = value, state
                    headings[nxt] = theta
                    serial += 1
                    heappush(frontier, (value + heuristic(nx, ny), serial, nxt))
            for turn in (-1, 1):
                next_it = (it + turn) % bins
                next_theta = theta + turn * step
                if not self._swept_rect_clear(point, point, theta, next_theta):
                    continue
                nxt, edge_cost = (ix, iy, next_it), .2 * self.resolution
                value = costs[state] + edge_cost
                if value < costs.get(nxt, float("inf")):
                    costs[nxt], parent[nxt] = value, state
                    headings[nxt] = next_theta
                    serial += 1
                    heappush(frontier, (value + heuristic(ix, iy), serial, nxt))
        if terminal is None:
            return None
        states = [terminal]
        while states[-1] != start_state:
            states.append(parent[states[-1]])
        states.reverse()
        patch = [entry]
        for ix, iy, it in states:
            theta = headings[(ix, iy, it)]
            center = self._cell_center((ix, iy))
            if (hypot(center[0] - patch[-1][0], center[1] - patch[-1][1]) > 1e-7 or
                    abs(theta - patch[-1][2]) > 1e-7):
                patch.append((center[0], center[1], theta))
        patch.append(exit)
        candidate = poses[:entry_index] + patch + poses[exit_index + 1:]
        return candidate if self._pose_path_clear(candidate) else None

    def _cell_blocked(self, ix, iy, dynamic):
        x, y = self._cell_center((ix, iy))
        radius = min(self.robot_length, self.robot_width) / 2
        cell_pad = self.resolution / 2
        if (x < radius or x > self.length - radius or
                y < radius or y > self.width - radius):
            return True
        for box in (*self.colliders, *dynamic):
            if hasattr(box, "length"):
                cx, cy, hx, hy = box.x, box.y, box.length / 2, box.width / 2
            else:
                cx, cy, hx, hy = box
            dx, dy = abs(x - cx), abs(y - cy)
            nearest_x, nearest_y = max(dx - hx, 0.), max(dy - hy, 0.)
            if nearest_x * nearest_x + nearest_y * nearest_y <= (radius + cell_pad) ** 2:
                return True
        return False

    def configure_footprint(self, robot_length, robot_width, heading):
        """Refresh the inexpensive inscribed-circle search map if size changed."""
        length, width, heading = (max(.01, float(robot_length)),
                                  max(.01, float(robot_width)), float(heading))
        dimensions_changed = (length, width) != (self.robot_length, self.robot_width)
        self.robot_length, self.robot_width, self.robot_heading = length, width, heading
        if dimensions_changed:
            self._blocked = [[self._cell_blocked(ix, iy, self._dynamic)
                              for iy in range(self.ny)] for ix in range(self.nx)]
            if self._goal is not None:
                self._start = self._goal = None

    def _cell(self, point):
        x = max(0, min(self.nx - 1, floor(float(point[0]) / self.resolution)))
        y = max(0, min(self.ny - 1, floor(float(point[1]) / self.resolution)))
        return x, y

    def _cell_center(self, cell):
        return ((cell[0] + .5) * self.resolution,
                (cell[1] + .5) * self.resolution)

    def _nearest_free(self, cell):
        if not self._blocked[cell[0]][cell[1]]:
            return cell
        for radius in range(1, max(self.nx, self.ny)):
            candidates = ((cell[0] + dx, cell[1] + dy)
                          for dx in range(-radius, radius + 1)
                          for dy in range(-radius, radius + 1)
                          if max(abs(dx), abs(dy)) == radius)
            for x, y in candidates:
                if 0 <= x < self.nx and 0 <= y < self.ny and not self._blocked[x][y]:
                    return x, y
        return cell

    def _reset_search(self, start, goal):
        self._g.clear(); self._rhs.clear(); self._open.clear(); self._closed.clear()
        self._incons.clear(); self._heap.clear(); self._serial = 0
        self._start, self._goal, self._epsilon = start, goal, 2.5
        self._rhs[goal] = 0.
        self._push(goal)

    def _heuristic(self, a, b):
        return hypot(a[0] - b[0], a[1] - b[1])

    def configure_kinematics(self, max_speed, max_acceleration, steering_rate=12.,
                             lateral_friction=1.2, max_angular_speed=None,
                             max_angular_acceleration=None):
        """Set trajectory limits; geometric AD* search costs are unchanged."""
        limits = (max(.1, float(max_speed)), max(.1, float(max_acceleration)),
                  max(.1, float(steering_rate)), max(.1, float(lateral_friction)))
        if limits != (self.max_speed, self.max_acceleration, self.steering_rate,
                      self.lateral_friction):
            self.max_speed, self.max_acceleration, self.steering_rate, self.lateral_friction = limits
        if max_angular_speed is not None:
            self.max_angular_speed = max(.1, float(max_angular_speed))
        if max_angular_acceleration is not None:
            self.max_angular_acceleration = max(.1, float(max_angular_acceleration))

    def _speed_cap(self, node):
        # Terrain affects acceleration through grade and rolling resistance;
        # don't impose a blanket low speed over the entire bump footprint.
        return self.max_speed

    def _ramp_grade(self, point):
        x, y = map(float, point)
        grades = []
        for box in self.bumps:
            if hasattr(box, "length"):
                cx, cy, hx, hy = box.x, box.y, box.length / 2, box.width / 2
            else:
                cx, cy, hx, hy = box
            if abs(x - cx) <= hx and abs(y - cy) <= hy:
                grades.append(-((x > cx) - (x < cx)) * BUMP_RAMP_RISE / max(hx, 1e-6))
        return sum(grades) / len(grades) if grades else 0.

    def _segment_accelerations(self, path, index):
        """Positive propulsion and braking limits projected onto path tangent."""
        distance = hypot(path[index + 1][0] - path[index][0],
                         path[index + 1][1] - path[index][1])
        if distance <= 1e-9:
            return self.max_acceleration, self.max_acceleration
        tx = (path[index + 1][0] - path[index][0]) / distance
        grade = .5 * (self._ramp_grade(path[index]) + self._ramp_grade(path[index + 1]))
        gravity_along_path = -9.81 * grade * tx
        bump_fraction = .5 * (float(self._bump[self._cell(path[index])[0]][self._cell(path[index])[1]]) +
                              float(self._bump[self._cell(path[index + 1])[0]][self._cell(path[index + 1])[1]]))
        rolling = BUMP_ROLLING_RESISTANCE * 9.81 * bump_fraction
        propulsion = max(.2, self.max_acceleration + gravity_along_path - rolling)
        braking = max(.2, self.max_acceleration - gravity_along_path + rolling)
        return propulsion, braking

    def _edge_time(self, source, target, grid_distance):
        # PathPlanner LocalADStar uses Euclidean grid distance for edge cost.
        # Terrain and drivetrain constraints are applied in the trajectory
        # speed profile after geometric pathfinding.
        return float(grid_distance)

    def _key(self, node):
        g, rhs = self._g.get(node, float("inf")), self._rhs.get(node, float("inf"))
        if g > rhs:
            return rhs + self._epsilon * self._heuristic(self._start, node), rhs
        return g + self._heuristic(self._start, node), g

    def _push(self, node):
        self._serial += 1
        key = self._key(node)
        self._open[node] = (key, self._serial)
        heappush(self._heap, (key, self._serial, node))

    def _remove(self, node):
        self._open.pop(node, None)

    def _neighbors_of(self, node):
        x, y = node
        for dx, dy, distance in self._neighbors:
            nxt = x + dx, y + dy
            if not (0 <= nxt[0] < self.nx and 0 <= nxt[1] < self.ny):
                continue
            if self._blocked[nxt[0]][nxt[1]]:
                continue
            if dx and dy and (self._blocked[x + dx][y] or self._blocked[x][y + dy]):
                continue
            yield nxt, self._edge_time(node, nxt, distance)

    def _predecessors(self, node):
        x, y = node
        if self._blocked[x][y]:
            return
        for dx, dy, distance in self._neighbors:
            pred = x + dx, y + dy
            if not (0 <= pred[0] < self.nx and 0 <= pred[1] < self.ny):
                continue
            if self._blocked[pred[0]][pred[1]]:
                continue
            if dx and dy and (self._blocked[pred[0]][y] or
                              self._blocked[x][pred[1]]):
                continue
            yield pred, self._edge_time(pred, node, distance)

    def _line_is_clear(self, start, end):
        """PathPlanner's integer-grid walkability check for a line segment."""
        x0, y0 = self._cell(start)
        x1, y1 = self._cell(end)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y, n = x0, y0, 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        while n > 0:
            if self._blocked[x][y]:
                return False
            if error > 0:
                x += x_inc
                error -= dy
            elif error < 0:
                y += y_inc
                error += dx
            else:
                # A diagonal corner touch is not clear space for a finite
                # bumper. Require both side cells before stepping through it.
                if self._blocked[x + x_inc][y] or self._blocked[x][y + y_inc]:
                    return False
                x += x_inc
                y += y_inc
                error += dx - dy
                n -= 1
            n -= 1
        return True

    def _smooth_path(self, points):
        """Greedily keep the farthest line-of-sight waypoint on every route."""
        if len(points) < 2:
            return list(points)
        smoothed = [points[0]]
        anchor = 0
        while anchor < len(points) - 1:
            target = len(points) - 1
            while target > anchor + 1 and not self._line_is_clear(points[anchor], points[target]):
                target -= 1
            if not self._line_is_clear(points[anchor], points[target]):
                # Search edges should be line-clear; if malformed input reaches
                # here, truncate safely instead of creating a shortcut through
                # a blocked cell.
                return smoothed
            smoothed.append(points[target])
            anchor = target
        return smoothed

    def _finalize_path(self, points, initial_velocity=(0., 0.),
                       initial_angular_velocity=0.):
        """Shortcut, spline, then repair only locally against the actual chassis."""
        route = self._curve_path(self._smooth_path(points))
        if len(route) < 2:
            if route and not self._pose_clear(*route[0], self.robot_heading):
                self.last_poses = []
                return []
            self.last_poses = self._poses_for_path(route)
            return route
        poses = self._poses_for_path(route)
        for _ in range(6):
            collision = self._first_collision(poses)
            if collision is None:
                self.last_poses = poses
                return [(p[0], p[1]) for p in poses]
            points = [(p[0], p[1]) for p in poses]
            heading = self._repair_heading(poses, collision, initial_velocity,
                                           initial_angular_velocity)
            translation = self._repair_translation(points, collision,
                                                   initial_velocity,
                                                   initial_angular_velocity)
            options = []
            if heading is not None:
                options.append((heading[0], heading[1]))
            if translation is not None:
                options.append((translation[0], translation[2]))
            if options:
                _, poses = min(options, key=lambda item: item[0])
            else:
                poses = self._local_se2_repair(poses, collision)
                if poses is None:
                    self.last_poses = []
                    return []
            if not self._pose_path_clear(poses):
                continue
        if self._first_collision(poses) is None:
            self.last_poses = poses
            return [(p[0], p[1]) for p in poses]
        self.last_poses = []
        return []

    def _curve_path(self, points):
        """Build PathPlanner-style Bézier waypoints, then sample the curves."""
        if len(points) < 2:
            return points

        # LocalADStar turns every corner into two poses, 80% along its
        # incoming and outgoing legs, then waypointsFromPoses assigns 0.4
        # auto-control handles in each pose's travel direction.
        poses = [(points[0], atan2(points[1][1] - points[0][1],
                                   points[1][0] - points[0][0]))]
        for i in range(1, len(points) - 1):
            previous, corner, following = points[i - 1], points[i], points[i + 1]
            incoming = atan2(corner[1] - previous[1], corner[0] - previous[0])
            outgoing = atan2(following[1] - corner[1], following[0] - corner[0])
            p1 = (previous[0] + .8 * (corner[0] - previous[0]),
                  previous[1] + .8 * (corner[1] - previous[1]))
            p2 = (corner[0] + .2 * (following[0] - corner[0]),
                  corner[1] + .2 * (following[1] - corner[1]))
            poses.extend(((p1, incoming), (p2, outgoing)))
        poses.append((points[-1], atan2(points[-1][1] - points[-2][1],
                                        points[-1][0] - points[-2][0])))

        def bezier(control_factor):
            controls = []
            for i, (anchor, heading) in enumerate(poses):
                prev = None
                nxt = None
                if i:
                    prev_dist = hypot(anchor[0] - poses[i - 1][0][0],
                                      anchor[1] - poses[i - 1][0][1]) * control_factor
                    prev = (anchor[0] - prev_dist * cos(heading),
                            anchor[1] - prev_dist * sin(heading))
                if i + 1 < len(poses):
                    next_dist = hypot(poses[i + 1][0][0] - anchor[0],
                                      poses[i + 1][0][1] - anchor[1]) * control_factor
                    nxt = (anchor[0] + next_dist * cos(heading),
                           anchor[1] + next_dist * sin(heading))
                controls.append((anchor, nxt, prev))

            result = [poses[0][0]]
            for i in range(len(poses) - 1):
                p0, p1 = controls[i][0], controls[i][1]
                p2, p3 = controls[i + 1][2], controls[i + 1][0]
                if p1 is None or p2 is None:
                    p1, p2 = p0, p3
                length = sum(hypot(b[0] - a[0], b[1] - a[1])
                             for a, b in ((p0, p1), (p1, p2), (p2, p3)))
                samples = max(2, int(ceil(length / .08)))
                for j in range(1, samples + 1):
                    t = j / samples
                    u = 1 - t
                    result.append((u**3*p0[0] + 3*u*u*t*p1[0] +
                                   3*u*t*t*p2[0] + t**3*p3[0],
                                   u**3*p0[1] + 3*u*u*t*p1[1] +
                                   3*u*t*t*p2[1] + t**3*p3[1]))
            return result

        for factor in (.4, .28, .18, .1, 0.):
            curve = bezier(factor)
            if all(self._line_is_clear(a, b) for a, b in zip(curve, curve[1:])):
                return curve
        # Collision repair runs after smoothing, so return the LOS route even
        # when the actual bumper clips it; repair is explicitly local to the hit.
        return points

    def _update_vertex(self, node):
        if node != self._goal:
            best = min((cost + self._g.get(nxt, float("inf"))
                        for nxt, cost in self._neighbors_of(node)), default=float("inf"))
            self._rhs[node] = best
        if node in self._open:
            self._remove(node)
        if self._g.get(node, float("inf")) != self._rhs.get(node, float("inf")):
            if node not in self._closed:
                self._push(node)
            else:
                self._incons.add(node)

    def _peek(self):
        while self._heap:
            key, serial, node = self._heap[0]
            if self._open.get(node) == (key, serial):
                return key, node
            heappop(self._heap)
        return (float("inf"), float("inf")), None

    def _compute_or_improve_path(self, budget):
        expanded = 0
        while expanded < budget:
            top_key, node = self._peek()
            if node is None or (top_key >= self._key(self._start) and
                                self._g.get(self._start, float("inf")) ==
                                self._rhs.get(self._start, float("inf"))):
                break
            heappop(self._heap); self._remove(node)
            old_key = top_key
            new_key = self._key(node)
            if old_key < new_key:
                self._push(node)
            elif self._g.get(node, float("inf")) > self._rhs.get(node, float("inf")):
                self._g[node] = self._rhs[node]
                self._closed.add(node)
                for pred, _ in self._predecessors(node):
                    self._update_vertex(pred)
            else:
                self._g[node] = float("inf")
                self._update_vertex(node)
                for pred, _ in self._predecessors(node):
                    self._update_vertex(pred)
            expanded += 1
        return expanded

    def set_dynamic_obstacles(self, obstacles):
        """Replace moving colliders and update only vertices whose edges changed."""
        obstacles = tuple(obstacles)
        old_blocked = self._blocked
        blocked = [[False] * self.ny for _ in range(self.nx)]
        for ix in range(self.nx):
            for iy in range(self.ny):
                blocked[ix][iy] = self._cell_blocked(ix, iy, obstacles)
        changed = [(x, y) for x in range(self.nx) for y in range(self.ny)
                   if old_blocked[x][y] != blocked[x][y]]
        self._dynamic, self._blocked = obstacles, blocked
        if self._goal is not None and changed:
            affected = {(x + dx, y + dy) for x, y in changed
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                        if 0 <= x + dx < self.nx and 0 <= y + dy < self.ny}
            for node in affected:
                self._update_vertex(node)
            self._epsilon = max(self._epsilon, 2.5)
            for node in self._incons:
                self._push(node)
            self._incons.clear(); self._closed.clear()
            self._rekey_open()

    def _rekey_open(self):
        items = list(self._open)
        self._open.clear(); self._heap.clear()
        for node in items:
            self._push(node)

    def plan(self, start, goal, *, expansion_budget: int = 12000,
             initial_velocity=(0., 0.), initial_angular_velocity=0.):
        """Incrementally plan in 2D, then repair footprint collisions locally."""
        start, goal = tuple(map(float, start[:2])), tuple(map(float, goal[:2]))
        self.last_poses = []
        start_cell, goal_cell = self._cell(start), self._cell(goal)
        s, g = self._nearest_free(start_cell), self._nearest_free(goal_cell)
        if (s == g and self._line_is_clear(start, goal)):
            return self._finalize_path([start, goal], initial_velocity,
                                       initial_angular_velocity)
        if self._goal != g or self._start is None:
            self._reset_search(s, g)
        else:
            if self._start != s:
                self._start = s
                for node in self._incons:
                    self._push(node)
                self._incons.clear(); self._closed.clear()
            else:
                self._start = s
            self._rekey_open()
        spent = 0
        while spent < expansion_budget:
            expanded = self._compute_or_improve_path(expansion_budget - spent)
            spent += expanded
            if self._epsilon <= 1. or (not self._open and not self._incons):
                break
            self._epsilon = max(1., self._epsilon - .5)
            for node in self._incons:
                self._push(node)
            self._incons.clear(); self._closed.clear()
            self._rekey_open()
            if expanded == 0 and spent >= expansion_budget:
                break
        if self._g.get(s, float("inf")) == float("inf"):
            return []
        best, seen = [s], {s}
        while best[-1] != g:
            node = best[-1]
            options = [(self._g.get(nxt, float("inf")), nxt)
                       for nxt, _ in self._neighbors_of(node)]
            if not options:
                break
            _, nxt = min(options)
            if nxt in seen:
                break
            best.append(nxt); seen.add(nxt)
        if best[-1] != g:
            return []
        points = [self._cell_center(node) for node in best]
        points[0] = start
        points[-1] = goal
        return self._finalize_path(points, initial_velocity,
                                   initial_angular_velocity)

    @staticmethod
    def predict_intercept(attacker, goal, defender, defender_velocity,
                          attacker_speed: float, clearance: float = .96,
                          max_horizon: float = 1.5):
        """Predict the defender's constant-velocity position at an approach intercept."""
        ax, ay = map(float, attacker)
        gx, gy = map(float, goal)
        dx, dy = gx - ax, gy - ay
        distance = hypot(dx, dy)
        if distance < 1e-6:
            return tuple(map(float, defender)), 0.
        va = (dx / distance * float(attacker_speed), dy / distance * float(attacker_speed))
        q = (float(defender[0]) - ax, float(defender[1]) - ay)
        vd = tuple(map(float, defender_velocity))
        relative = (vd[0] - va[0], vd[1] - va[1])
        c = q[0] * q[0] + q[1] * q[1] - clearance * clearance
        a = relative[0] * relative[0] + relative[1] * relative[1]
        b = 2 * (q[0] * relative[0] + q[1] * relative[1])
        time_to_intercept = None
        if c <= 0:
            time_to_intercept = 0.
        elif a > 1e-9:
            discriminant = b * b - 4 * a * c
            if discriminant >= 0:
                roots = [t for t in ((-b - discriminant ** .5) / (2 * a),
                                     (-b + discriminant ** .5) / (2 * a)) if t >= 0]
                if roots:
                    time_to_intercept = min(roots)
        if time_to_intercept is None:
            # If the constant-velocity lines miss, project to closest approach.
            time_to_intercept = max(0., -(q[0] * relative[0] + q[1] * relative[1]) /
                                    max(a, 1e-9))
        time_to_intercept = min(float(time_to_intercept), max_horizon)
        return (vd[0] * time_to_intercept + float(defender[0]),
                vd[1] * time_to_intercept + float(defender[1])), time_to_intercept

    def plan_attacker(self, attacker, goal, defender, defender_velocity,
                      attacker_speed: float, *, attacker_acceleration: float = 8.,
                      steering_rate: float = 12., lateral_friction: float = 1.2,
                      max_angular_speed: float = 8.,
                      max_angular_acceleration: float = 18.,
                      attacker_heading: float | None = None,
                      attacker_length: float | None = None,
                      attacker_width: float | None = None,
                      attacker_velocity=None, attacker_angular_velocity: float = 0.,
                      expansion_budget: int = 4500):
        """Replan an attack route around the defender's velocity-predicted intercept."""
        self.configure_kinematics(attacker_speed, attacker_acceleration, steering_rate,
                                  lateral_friction, max_angular_speed,
                                  max_angular_acceleration)
        self.configure_footprint(attacker_length or self.robot_length,
                                 attacker_width or self.robot_width,
                                 self.robot_heading if attacker_heading is None else attacker_heading)
        intercept, intercept_time = self.predict_intercept(
            attacker, goal, defender, defender_velocity, attacker_speed)
        half_extent = .45
        dynamic = (intercept[0], intercept[1], half_extent, half_extent)
        self.set_dynamic_obstacles((dynamic,))
        path = self.plan(attacker, goal, expansion_budget=expansion_budget,
                         initial_velocity=((0., 0.) if attacker_velocity is None
                                           else attacker_velocity),
                         initial_angular_velocity=attacker_angular_velocity)
        # If the moving intercept closes the start/goal corridor completely,
        # do not return a stationary one-cell route. Fall back to static-field
        # avoidance and let the next velocity update choose a new intercept.
        if len(path) <= 1 and hypot(float(goal[0]) - float(attacker[0]),
                                    float(goal[1]) - float(attacker[1])) > .75:
            self.set_dynamic_obstacles(())
            path = self.plan(attacker, goal, expansion_budget=max(12000, expansion_budget * 3),
                             initial_velocity=((0., 0.) if attacker_velocity is None
                                               else attacker_velocity),
                             initial_angular_velocity=attacker_angular_velocity)
            if len(path) <= 1:
                # Dynamic-map repair can leave an incomplete incremental search
                # when its budget is exhausted. Reinitialize against the static
                # map before returning a one-point route that commands a stop.
                self._start = self._goal = None
                path = self.plan(attacker, goal, expansion_budget=max(24000, expansion_budget * 6),
                                 initial_velocity=((0., 0.) if attacker_velocity is None
                                                   else attacker_velocity),
                                 initial_angular_velocity=attacker_angular_velocity)
        return path, intercept, intercept_time

    def speed_profile(self, path, initial_velocity=(0., 0.), module_angles=None):
        """Return a feasible per-waypoint speed profile for the swerve drivetrain."""
        count = len(path)
        if count == 0:
            return []
        if count == 1:
            return [0.]
        distances = [hypot(path[i + 1][0] - path[i][0],
                           path[i + 1][1] - path[i][1]) for i in range(count - 1)]
        caps = []
        for point in path:
            cell = self._cell(point)
            caps.append(self._speed_cap(cell))
        headings = [
            (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(count - 1)]
        for i in range(1, count - 1):
            ax, ay = headings[i - 1]; bx, by = headings[i]
            turn = abs(atan2(ax * by - ay * bx, ax * bx + ay * by))
            arc = max((distances[i - 1] + distances[i]) * .5, 1e-6)
            curvature = turn / arc
            if curvature > 1e-6:
                grade = self._ramp_grade(path[i])
                lateral_accel = (TURN_LATERAL_ACCEL_SCALE * self.lateral_friction *
                                 9.81 / hypot(1., grade))
                caps[i] = min(caps[i], (lateral_accel / curvature) ** .5,
                              self.steering_rate / curvature)
        first_dx, first_dy = headings[0]
        first_norm = max(hypot(first_dx, first_dy), 1e-8)
        speed = [min(caps[0], max(0.,
            (initial_velocity[0] * first_dx + initial_velocity[1] * first_dy) / first_norm))]
        speed.extend(caps[1:])
        for i in range(1, count):
            accel, _ = self._segment_accelerations(path, i - 1)
            speed[i] = min(speed[i], (speed[i - 1] ** 2 + 2 * accel * distances[i - 1]) ** .5)
        speed[-1] = 0.
        for i in range(count - 2, -1, -1):
            _, accel = self._segment_accelerations(path, i)
            accel *= BRAKING_ACCEL_SCALE
            if i == 0 and module_angles is not None:
                dx, dy = headings[0]
                heading = atan2(dy, dx)
                lateral_share = sum(sin(heading-angle)**2
                                    for angle in module_angles) / max(len(module_angles), 1)
                grade = self._ramp_grade(path[i])
                lateral_accel = (TURN_LATERAL_ACCEL_SCALE * self.lateral_friction *
                                 9.81 / hypot(1., grade))
                accel = accel * (1. - lateral_share) + lateral_accel * lateral_share
            speed[i] = min(speed[i], (speed[i + 1] ** 2 + 2 * accel * distances[i]) ** .5)
        return speed

    def next_waypoint_index(self, path, position, lookahead: float = .35):
        if not path:
            return 0
        px, py = float(position[0]), float(position[1])
        nearest = min(range(len(path)), key=lambda i: (path[i][0] - px) ** 2 + (path[i][1] - py) ** 2)
        for index in range(nearest + 1, len(path)):
            if hypot(path[index][0] - px, path[index][1] - py) >= lookahead:
                return index
        return len(path) - 1

    def next_waypoint(self, path, position, lookahead: float = .35):
        if not path:
            return position
        return path[self.next_waypoint_index(path, position, lookahead)]

    @staticmethod
    def path_reference(path, speeds, position, lookahead_distance=.35, *,
                       progress_hint=None, return_projection=False):
        """Sample a path reference without allowing projection to jump backward.

        ``progress_hint`` is the distance-along-path estimate from the previous
        control step. It makes projection stable where a route doubles back or
        has nearby parallel sections. The optional projection is the closest
        point on the path at the robot's current station, separate from the
        lookahead target used for feed-forward sampling.
        """
        if not path:
            empty = (tuple(position), (0., 0.), 0.)
            return (*empty, tuple(position), 0.) if return_projection else empty
        if len(path) == 1:
            one = (path[0], (0., 0.), 0.)
            return (*one, path[0], 0.) if return_projection else one
        px, py = float(position[0]), float(position[1])
        lengths = [hypot(path[i + 1][0] - path[i][0],
                         path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
        cumulative = [0.]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        best_distance, progress, projection = float("inf"), 0., path[0]
        for i, length in enumerate(lengths):
            if length <= 1e-9:
                continue
            if progress_hint is not None and cumulative[i + 1] < max(0., progress_hint - .08):
                continue
            ax, ay = path[i]
            dx, dy = path[i + 1][0] - ax, path[i + 1][1] - ay
            t = max(0., min(1., ((px - ax) * dx + (py - ay) * dy) / (length * length)))
            q = (ax + dx * t, ay + dy * t)
            distance = (q[0] - px) ** 2 + (q[1] - py) ** 2
            if distance < best_distance:
                best_distance, progress, projection = distance, cumulative[i] + t * length, q
        if progress_hint is not None:
            progress = max(progress, min(cumulative[-1], float(progress_hint)))
        target_s = min(cumulative[-1], progress + max(.12, float(lookahead_distance)))
        target_index = len(path) - 1
        for i, length in enumerate(lengths):
            if target_s <= cumulative[i + 1] or i == len(lengths) - 1:
                target_index = i
                t = 0. if length <= 1e-9 else max(0., min(1., (target_s - cumulative[i]) / length))
                target = (path[i][0] + (path[i + 1][0] - path[i][0]) * t,
                          path[i][1] + (path[i + 1][1] - path[i][1]) * t)
                tangent = (0., 0.) if length <= 1e-9 else (
                    (path[i + 1][0] - path[i][0]) / length,
                    (path[i + 1][1] - path[i][1]) / length)
                v0 = speeds[i] if i < len(speeds) else 0.
                v1 = speeds[i + 1] if i + 1 < len(speeds) else v0
                # The reference is sampled at the lookahead station. The
                # follower adds measured overspeed braking against this target.
                result = (target, tangent, float(v0 + (v1 - v0) * t))
                return (*result, projection, progress) if return_projection else result
        result = (path[-1], (0., 0.), float(speeds[-1] if speeds else 0.))
        return (*result, projection, progress) if return_projection else result
