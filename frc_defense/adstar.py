"""PathPlanner LocalADStar-style pathfinding and holonomic path following.

The AD* search and Bézier waypoint construction follow PathPlanner's
``LocalADStar`` implementation (MIT; copyright Michael Jansen, 2022), adapted
for this simulator's field boxes, bump speed zones, and drivetrain limits.
Source: https://github.com/mjansen4857/pathplanner/tree/main/pathplannerlib-python/pathplannerlib/pathfinders.py
"""
from __future__ import annotations

from math import ceil, floor, sqrt
from heapq import heappop, heappush
from math import atan2, cos, hypot, sin

from .field import BUMP_RAMP_RISE, BUMP_ROLLING_RESISTANCE


TURN_LATERAL_ACCEL_SCALE = .65
BRAKING_ACCEL_SCALE = .65


class ADStarPlanner:
    """Incremental Anytime Dynamic A* on a 8-connected field grid."""

    def __init__(self, length: float, width: float, colliders=(), bumps=(), *,
                 resolution: float = .2, robot_length: float = .9,
                 robot_width: float = .9, robot_radius: float | None = None,
                 robot_heading: float = 0.,
                 max_speed: float = 4.5,
                 max_acceleration: float = 8., steering_rate: float = 12.,
                 lateral_friction: float = 1.2):
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
        self._blocked = [[False] * self.ny for _ in range(self.nx)]
        for ix in range(self.nx):
            x = (ix + .5) * self.resolution
            for iy in range(self.ny):
                y = (iy + .5) * self.resolution
                self._blocked[ix][iy] = self._cell_blocked(ix, iy, ())
        self._bump = [[False] * self.ny for _ in range(self.nx)]
        for ix in range(self.nx):
            x = (ix + .5) * self.resolution
            for iy in range(self.ny):
                y = (iy + .5) * self.resolution
                self._bump[ix][iy] = any(self._inside_box(x, y, b, 0.) for b in self.bumps)
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
        self.orientation_bins = 8
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
        # No grid padding here: the footprint is tested directly against every
        # collider. Grid padding is useful for search, but must not stand in for
        # checking the actual swept bumper on a smoothed trajectory.
        dx_axis = (c, s)
        dy_axis = (-s, c)
        for box in (*self.colliders, *self._dynamic):
            if hasattr(box, "length"):
                cx, cy, hx, hy = box.x, box.y, box.length / 2, box.width / 2
            else:
                cx, cy, hx, hy = box
            dx, dy = x - cx, y - cy
            if (abs(dx) > hx + extent_x or abs(dy) > hy + extent_y or
                    abs(dx * c + dy * s) > half_l + hx * abs(c) + hy * abs(s) or
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
        period = (0.5 * 3.141592653589793
                  if abs(self.robot_length - self.robot_width) <= 1e-5 else
                  3.141592653589793)
        heading_delta = (end_heading - start_heading + period / 2) % period - period / 2
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
        """Validate translation and predicted tangent rotation along a route."""
        if len(points) < 2:
            return bool(points) and self._pose_clear(*points[0])
        headings = []
        for i, point in enumerate(points):
            before = points[max(0, i - 1)]
            after = points[min(len(points) - 1, i + 1)]
            headings.append(atan2(after[1] - before[1], after[0] - before[0]))
        return all(self._swept_rect_clear(a, b, headings[i], headings[i + 1])
                   for i, (a, b) in enumerate(zip(points, points[1:])))

    def _cell_blocked(self, ix, iy, dynamic):
        x, y = self._cell_center((ix, iy))
        c, s = cos(self.robot_heading), sin(self.robot_heading)
        pad = self.resolution / 2
        ex = self.robot_length / 2 * abs(c) + self.robot_width / 2 * abs(s) + pad
        ey = self.robot_length / 2 * abs(s) + self.robot_width / 2 * abs(c) + pad
        return (x < ex or x > self.length - ex or y < ey or y > self.width - ey or
                any(self._rect_hits_box(x, y, b) for b in (*self.colliders, *dynamic)))

    def configure_footprint(self, robot_length, robot_width, heading):
        """Refresh grid clearance for the current chassis size and orientation."""
        values = (max(.01, float(robot_length)), max(.01, float(robot_width)),
                  float(heading))
        if values == (self.robot_length, self.robot_width, self.robot_heading):
            return
        self.robot_length, self.robot_width, self.robot_heading = values
        self._blocked = [[self._cell_blocked(ix, iy, self._dynamic)
                          for iy in range(self.ny)] for ix in range(self.nx)]
        # The footprint changed globally; incremental edge repair is not valid.
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
                             lateral_friction=1.2):
        """Set trajectory limits; geometric AD* search costs are unchanged."""
        limits = (max(.1, float(max_speed)), max(.1, float(max_acceleration)),
                  max(.1, float(steering_rate)), max(.1, float(lateral_friction)))
        if limits != (self.max_speed, self.max_acceleration, self.steering_rate,
                      self.lateral_friction):
            self.max_speed, self.max_acceleration, self.steering_rate, self.lateral_friction = limits

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
            # Grid-cell occupancy uses the robot's current heading. Also test
            # each candidate move with its predicted travel heading so the
            # long bumper corners cannot clip an obstacle on a diagonal edge.
            source, target = self._cell_center(node), self._cell_center(nxt)
            heading = atan2(target[1] - source[1], target[0] - source[0])
            if not self._swept_rect_clear(source, target, heading, heading):
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
            source, target = self._cell_center(pred), self._cell_center(node)
            heading = atan2(target[1] - source[1], target[0] - source[0])
            if not self._swept_rect_clear(source, target, heading, heading):
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

    def _finalize_path(self, points):
        """Apply the same safe LOS reduction and curve fit to every AD* route."""
        if len(points) < 2:
            return list(points)
        route = self._curve_path(self._smooth_path(points))
        if self._trajectory_clear(route):
            return route
        # Never publish a route that failed the same predictive footprint
        # check used by search edges. An empty route is a safe planning failure.
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
            if (all(self._line_is_clear(a, b) for a, b in zip(curve, curve[1:])) and
                    self._trajectory_clear(curve)):
                return curve
        # A smooth spline may bulge outside the AD* clearance corridor. Try
        # line-of-sight segments with the same full-body sweep validation.
        if self._trajectory_clear(points):
            return points
        # The caller retains the original discrete route as the final fallback.
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

    def plan(self, start, goal, *, expansion_budget: int = 12000):
        """Plan a swept-footprint route in discretized SE(2).

        Square symmetry reduces chassis orientation to a quarter-turn. Rectangular
        footprints retain their half-turn symmetry. ``last_poses`` stores
        (x, y, theta) samples; the return value remains the legacy XY route.
        """
        start = tuple(map(float, start[:2]))
        goal = tuple(map(float, goal[:2]))
        self.last_poses = []
        period = (0.5 * 3.141592653589793
                  if abs(self.robot_length - self.robot_width) <= 1e-5 else
                  3.141592653589793)
        step = period / self.orientation_bins
        angle_bin = int(round((self.robot_heading % period) / step)) % self.orientation_bins
        initial_heading = angle_bin * step
        sc, gc = self._cell(start), self._cell(goal)
        sx, sy = self._cell_center(sc)
        gx, gy = self._cell_center(gc)
        if not self._pose_clear(*start, self.robot_heading):
            return []
        if not self._swept_rect_clear(start, (sx, sy), self.robot_heading,
                                      self.robot_heading):
            return []
        if not self._swept_rect_clear((sx, sy), (sx, sy), self.robot_heading,
                                      initial_heading):
            return []

        origin = (sc[0], sc[1], angle_bin)
        def heuristic(ix, iy):
            dx, dy = abs(ix - gc[0]), abs(iy - gc[1])
            return self.resolution * (max(dx, dy) + (1.41421356237 - 1.) * min(dx, dy))

        frontier = []
        serial = 0
        heappush(frontier, (heuristic(sc[0], sc[1]), serial, origin))
        costs = {origin: 0.}
        parent = {}
        closed = set()
        terminal = None

        expanded = 0
        while frontier and expanded < max(1, int(expansion_budget)):
            _, _, state = heappop(frontier)
            if state in closed:
                continue
            closed.add(state)
            ix, iy, it = state
            x, y = self._cell_center((ix, iy))
            theta = it * step
            if (ix, iy) == gc and self._swept_rect_clear((x, y), goal, theta, theta):
                terminal = state
                break
            expanded += 1
            transitions = []
            for dx, dy, scale in self._neighbors:
                nx, ny = ix + dx, iy + dy
                if not (0 <= nx < self.nx and 0 <= ny < self.ny):
                    continue
                target = self._cell_center((nx, ny))
                if not self._swept_rect_clear((x, y), target, theta, theta):
                    continue
                transitions.append(((nx, ny, it), self.resolution * scale))
            for turn in (-1, 1):
                next_it = (it + turn) % self.orientation_bins
                next_theta = next_it * step
                if self._swept_rect_clear((x, y), (x, y), theta, next_theta):
                    # Rotation has a small positive cost to avoid gratuitous
                    # spin while keeping the heuristic purely translational.
                    transitions.append(((ix, iy, next_it), self.resolution * .2))
            for nxt, edge_cost in transitions:
                new_cost = costs[state] + edge_cost
                if new_cost >= costs.get(nxt, float("inf")):
                    continue
                costs[nxt] = new_cost
                parent[nxt] = state
                serial += 1
                heappush(frontier, (new_cost + heuristic(nxt[0], nxt[1]),
                                    serial, nxt))

        if terminal is None:
            return []
        states = [terminal]
        while states[-1] != origin:
            states.append(parent[states[-1]])
        states.reverse()
        poses = [(start[0], start[1], self.robot_heading)]
        poses.append((sx, sy, initial_heading))
        for ix, iy, it in states[1:]:
            px, py = self._cell_center((ix, iy))
            poses.append((px, py, it * step))
        if hypot(poses[-1][0] - goal[0], poses[-1][1] - goal[1]) > 1e-7:
            poses.append((goal[0], goal[1], poses[-1][2]))

        # Greedy SE(2) shortcutting: interpolate both translation and angle,
        # accepting a shortcut only after checking its complete swept square.
        compact = [poses[0]]
        anchor = 0
        while anchor < len(poses) - 1:
            chosen = anchor + 1
            for candidate in range(len(poses) - 1, anchor, -1):
                a, b = poses[anchor], poses[candidate]
                if self._swept_rect_clear(a[:2], b[:2], a[2], b[2]):
                    chosen = candidate
                    break
            compact.append(poses[chosen])
            anchor = chosen
        self.last_poses = compact
        return [(pose[0], pose[1]) for pose in compact]

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
                      attacker_heading: float | None = None,
                      attacker_length: float | None = None,
                      attacker_width: float | None = None,
                      expansion_budget: int = 4500):
        """Replan an attack route around the defender's velocity-predicted intercept."""
        self.configure_kinematics(attacker_speed, attacker_acceleration, steering_rate,
                                  lateral_friction)
        self.configure_footprint(attacker_length or self.robot_length,
                                 attacker_width or self.robot_width,
                                 self.robot_heading if attacker_heading is None else attacker_heading)
        intercept, intercept_time = self.predict_intercept(
            attacker, goal, defender, defender_velocity, attacker_speed)
        half_extent = .45
        dynamic = (intercept[0], intercept[1], half_extent, half_extent)
        self.set_dynamic_obstacles((dynamic,))
        path = self.plan(attacker, goal, expansion_budget=expansion_budget)
        # If the moving intercept closes the start/goal corridor completely,
        # do not return a stationary one-cell route. Fall back to static-field
        # avoidance and let the next velocity update choose a new intercept.
        if len(path) <= 1 and hypot(float(goal[0]) - float(attacker[0]),
                                    float(goal[1]) - float(attacker[1])) > .75:
            self.set_dynamic_obstacles(())
            path = self.plan(attacker, goal, expansion_budget=max(12000, expansion_budget * 3))
            if len(path) <= 1:
                # Dynamic-map repair can leave an incomplete incremental search
                # when its budget is exhausted. Reinitialize against the static
                # map before returning a one-point route that commands a stop.
                self._start = self._goal = None
                path = self.plan(attacker, goal, expansion_budget=max(24000, expansion_budget * 6))
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
