"""Raster sweep route generation and cursor management for the 3v3 env."""
from __future__ import annotations

from math import ceil, hypot

import torch

from ._tensor_3v3_constants import TEAM_IDS


class TensorThreeVsThreeNavigationMixin:
    """Build and advance each robot's deterministic sweep route."""
    def _build_raster_spline_points(self, sample_count=256):
        """Build closed, smooth serpentine midfield routes for each robot."""
        field_width=float(self.sim.field_width)
        y_bands = ((.9, 2.65), (2.65, 5.42),
                   (5.42, field_width-.9))
        x_min = self.alliance_zone_depth + 1.45
        x_max = self.field_length - self.alliance_zone_depth - 1.45
        column_count = 4
        column_spacing = (x_max - x_min) / (column_count - 1)
        paths = []

        for robot, team in enumerate(TEAM_IDS):
            lane = robot % 3
            y_min, y_max = y_bands[lane]
            order = (0, 1, 2, 3, 2, 1, 0)
            x_at = lambda column: (x_min + column * column_spacing
                                   if team == 0 else
                                   x_max - column * column_spacing)
            start = (x_at(order[0]), y_min)
            control = [start]
            current_y = y_min
            end_y = y_max
            for column in order:
                x = x_at(column)
                if abs(control[-1][0] - x) > 1e-6:
                    control.append((x, current_y))
                control.append((x, end_y))
                current_y = end_y
                end_y = y_min if end_y == y_max else y_max
            if hypot(control[-1][0] - start[0], control[-1][1] - start[1]) > 1e-6:
                control.append(start)
            else:
                control.pop()

            dense = []
            count = len(control)
            for i in range(count):
                p0, p1 = control[(i - 1) % count], control[i]
                p2, p3 = control[(i + 1) % count], control[(i + 2) % count]
                chord = hypot(p2[0] - p1[0], p2[1] - p1[1])
                samples = max(3, ceil(chord / .25))
                for sample in range(samples):
                    t = sample / samples
                    t2, t3 = t * t, t * t * t
                    point = []
                    h00, h10 = 2.*t3-3.*t2+1., t3-2.*t2+t
                    h01, h11 = -2.*t3+3.*t2, t3-t2
                    for axis in range(2):
                        m1=.2*(p2[axis]-p0[axis])
                        m2=.2*(p3[axis]-p1[axis])
                        point.append(h00*p1[axis]+h10*m1+h01*p2[axis]+h11*m2)
                    dense.append(tuple(point))
            dense.append(dense[0])
            cumulative = [0.]
            for a, b in zip(dense, dense[1:]):
                cumulative.append(cumulative[-1] + hypot(b[0] - a[0], b[1] - a[1]))
            total = cumulative[-1]
            sampled = []
            segment = 0
            for i in range(sample_count):
                distance = total * i / sample_count
                while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < distance:
                    segment += 1
                span = max(cumulative[segment + 1] - cumulative[segment], 1e-9)
                fraction = (distance - cumulative[segment]) / span
                a, b = dense[segment], dense[segment + 1]
                sampled.append((a[0] + (b[0] - a[0]) * fraction,
                                a[1] + (b[1] - a[1]) * fraction))
            paths.append(sampled)
        return paths

    def _closest_raster_indices(self, positions):
        points = self.raster_sweep_points[None].expand(self.n, -1, -1, -1)
        distance = (points - positions[:, :, None, :]).square().sum(-1)
        return distance.argmin(-1)

    def _prepare_raster_cursor(self, active):
        positions=self.sim.pose[:,:,:2]
        global_nearest=self._closest_raster_indices(positions)
        offsets=torch.arange(33,device=self.device,dtype=torch.long)
        forward_indices=(self._raster_cursor[:,:,None]+offsets[None,None]) % self.raster_sweep_points.shape[1]
        points=self.raster_sweep_points[None].expand(self.n,-1,-1,-1)
        forward_points=points.gather(
            2,forward_indices[...,None].expand(-1,-1,-1,2))
        forward_distance=(forward_points-positions[:,:,None,:]).square().sum(-1)
        forward_nearest=(self._raster_cursor+
                         forward_distance.argmin(-1)) % self.raster_sweep_points.shape[1]
        next_cursor=torch.where(self._raster_reanchor,global_nearest,forward_nearest)
        self._raster_cursor.copy_(torch.where(active[:,None],next_cursor,self._raster_cursor))

    def _finish_raster_cursor(self, active, sweep_mask):
        positions=self.sim.pose[:,:,:2]
        offsets=torch.arange(33,device=self.device,dtype=torch.long)
        forward_indices=(self._raster_cursor[:,:,None]+offsets[None,None]) % self.raster_sweep_points.shape[1]
        points=self.raster_sweep_points[None].expand(self.n,-1,-1,-1)
        forward_points=points.gather(
            2,forward_indices[...,None].expand(-1,-1,-1,2))
        forward_distance=(forward_points-positions[:,:,None,:]).square().sum(-1)
        forward_nearest=(self._raster_cursor+
                         forward_distance.argmin(-1)) % self.raster_sweep_points.shape[1]
        update=active[:,None]&sweep_mask
        self._raster_cursor.copy_(torch.where(update,forward_nearest,self._raster_cursor))
        self._raster_reanchor.copy_(torch.where(
            active[:,None],~sweep_mask,self._raster_reanchor))

    def _raster_waypoint_targets(self, positions):
        points = self.raster_sweep_points[None].expand(self.n, -1, -1, -1)
        cursor = self._raster_cursor
        target_index = (cursor + self._raster_waypoint_lookahead) % points.shape[2]
        next_index = (target_index + 1) % points.shape[2]
        target = points.gather(2, target_index[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        following = points.gather(2, next_index[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        tangent = following - target
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return cursor, target, tangent
