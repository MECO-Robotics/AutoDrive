"""Raster sweep route generation and cursor management for the 3v3 env."""
from __future__ import annotations

from math import ceil, hypot

import torch

from ._tensor_3v3_constants import TEAM_IDS


class TensorThreeVsThreeNavigationMixin:
    """Build and advance each robot's deterministic sweep route."""
    def _build_raster_spline_points(self, sample_count=256):
        """Build long, full-height raster loops with broad end turns.

        The two straight passes span almost the whole field height and the
        full available midfield depth. Rounded U-turns shift the next pass
        toward midfield without the tight, multi-axis cornering of the old
        compact serpentine route.
        """
        field_width = float(self.sim.field_width)
        y_min, y_max = .65, field_width - .65
        turn_radius = .32
        x_start = self.alliance_zone_depth + .55
        x_mid = self.field_length * .5
        x_end = x_mid - .35
        paths = []

        for robot, team in enumerate(TEAM_IDS):
            # Keep each alliance in its own midfield half. The full x range
            # is used by every route so each robot sweeps the complete field
            # height while progressively moving farther from its alliance.
            if team == 1:
                x1, x2 = (self.field_length-x_start-turn_radius,
                          self.field_length-x_end+turn_radius)
            else:
                x1, x2 = x_start+turn_radius, x_end-turn_radius
            # Two Bezier U-turns join full-height straight passes. Their
            # horizontal tangents distribute heading change over distance.
            top_a, top_b = (x1, y_max-turn_radius), (x2, y_max-turn_radius)
            bot_a, bot_b = (x2, y_min+turn_radius), (x1, y_min+turn_radius)
            dense = []

            def line(a, b):
                count = max(2, ceil(hypot(b[0]-a[0], b[1]-a[1])/.10))
                for i in range(count):
                    t = i/count
                    dense.append((a[0]+(b[0]-a[0])*t,
                                  a[1]+(b[1]-a[1])*t))

            def bezier(a, c1, c2, b):
                count = max(8, ceil(hypot(b[0]-a[0], b[1]-a[1])/.04))
                for i in range(count):
                    t = i/count
                    u = 1.-t
                    dense.append((u**3*a[0]+3*u*u*t*c1[0]+3*u*t*t*c2[0]+t**3*b[0],
                                  u**3*a[1]+3*u*u*t*c1[1]+3*u*t*t*c2[1]+t**3*b[1]))

            line((x1, y_min+turn_radius), top_a)
            bezier(top_a, (x1, y_max+turn_radius*.55),
                   (x2, y_max+turn_radius*.55), top_b)
            line(top_b, bot_a)
            bezier(bot_a, (x2, y_min-turn_radius*.55),
                   (x1, y_min-turn_radius*.55), bot_b)
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
