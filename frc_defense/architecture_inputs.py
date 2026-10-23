"""Perception-only full-field raster and entity inputs for policy experiments.

The field is world aligned, with 64 columns along field length and 32 rows
along field width. Channels: solid geometry, bumps, log FUEL count, mean FUEL
vx, mean FUEL vy, mean confidence times freshness, self, teammates, opponent
track, own hub, enemy hub, retained FUEL track support. Support says nothing
about whether empty space has been observed. Retained tracks may be stale.

Entities contain every retained FUEL slot, self, two teammates, and one tracked
opponent (fuel_count + 4 slots). Features are relative x/y, vx/vy, age,
confidence, four type indicators (self/ally/opponent/FUEL), and x/y extent.
Positions/extents use field dimensions, velocities use own maximum speed,
and ages use the FUEL track timeout. No FUEL truth or enemy truth is read.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class PerceptionInputs:
    def __init__(self, env, robot_ids: tuple[int, ...] | list[int]) -> None:
        self.env = env
        self.robot_ids = tuple(int(robot) for robot in robot_ids)
        if not self.robot_ids or len(set(self.robot_ids)) != len(self.robot_ids):
            raise ValueError("robot_ids must be nonempty and unique")
        if any(robot < 0 or robot >= 6 for robot in self.robot_ids):
            raise ValueError("robot_ids must be in [0, 6)")
        self.ids = torch.tensor(self.robot_ids, device=env.device, dtype=torch.long)
        self.scales = torch.tensor(
            (env.sim.field_length, env.sim.field_width), device=env.device,
            dtype=torch.float32,
        )
        self.grid_scales = self.scales.new_tensor((64, 32))
        self.timeout = max(float(env.perception_track_timeout), 1e-3)
        teams = env.team_ids
        self.teammates = torch.stack([
            torch.nonzero((teams == teams[robot]) &
                          (torch.arange(6, device=env.device) != robot)).flatten()
            for robot in self.robot_ids
        ])
        if self.teammates.shape != (len(self.robot_ids), 2):
            raise ValueError("PerceptionInputs requires two teams of three robots")
        self.static = self._static_field()

    def _static_field(self) -> Tensor:
        env = self.env
        ys = (torch.arange(32, device=env.device) + .5) * self.scales[1] / 32
        xs = (torch.arange(64, device=env.device) + .5) * self.scales[0] / 64
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack((xx, yy), dim=-1)
        half_cell = self.scales / self.grid_scales / 2
        result = torch.zeros((len(self.robot_ids), 12, 32, 64), device=env.device)
        for channel, boxes in ((0, env.sim.field_colliders), (1, env.sim.bump_regions)):
            if boxes.numel():
                overlap = ((centers[..., None, :] - boxes[:, :2]).abs() <=
                           boxes[:, 2:4] + half_cell).all(-1).any(-1)
                result[:, channel] = overlap.float()
        circles = env.sim.obstacles
        if circles.numel():
            nearest = ((centers[..., None, :] - circles[:, :2]).abs() -
                       half_cell).clamp_min(0)
            overlap = (nearest.square().sum(-1) <= circles[:, 2].square()).any(-1)
            result[:, 0] = torch.maximum(result[:, 0], overlap.float())
        hub_cells = self._cells(env.hub_centers)
        own_team = env.team_ids[self.ids]
        rows = torch.arange(len(self.robot_ids), device=env.device)
        flat = result.flatten(2)
        flat[rows, 9, hub_cells[own_team]] = 1.
        flat[rows, 10, hub_cells[1 - own_team]] = 1.
        return result

    def _cells(self, positions: Tensor) -> Tensor:
        xy = torch.nan_to_num(positions, nan=0., posinf=0., neginf=0.)
        grid = (xy / self.scales * self.grid_scales).floor().long()
        return grid[..., 1].clamp(0, 31) * 64 + grid[..., 0].clamp(0, 63)

    def _selected(self, values: Tensor) -> Tensor:
        selected = values.index_select(1, self.ids)
        return selected.reshape(self.env.n * len(self.robot_ids), *values.shape[2:])

    @torch.no_grad()
    def __call__(self, obs: Tensor | None = None) -> dict[str, Tensor]:
        env = self.env
        batch = env.n * len(self.robot_ids)
        if obs is None:
            obs = env.observe()
        if obs.ndim == 3:
            scalars = self._selected(obs)
        elif obs.shape == (env.n * 6, 137):
            scalars = self._selected(obs.reshape(env.n, 6, 137))
        elif obs.shape == (batch, 137):
            scalars = obs
        else:
            raise ValueError("expected full or selected 137-feature observation batch")
        scalars = torch.nan_to_num(scalars, nan=0., posinf=0., neginf=0.)
        if getattr(env, "architecture_pickup_only", False):
            # Isolated pickup cannot score/ferry and reacquire the same balls.
            scalars = scalars.clone()
            scalars[:, -4:-1] = 0.

        fuel_mask = self._selected(env.track_mask).bool()
        fuel_pos = self._selected(env.track_pos)
        fuel_vel = self._selected(env.track_vel)
        fuel_age = self._selected(env.track_age)
        fuel_confidence = self._selected(env.track_confidence)
        fuel_extent = self._selected(env.track_spatial_extent)
        fuel_mask = fuel_mask & torch.isfinite(fuel_pos).all(-1)
        fuel_pos = torch.nan_to_num(fuel_pos.masked_fill(~fuel_mask[..., None], 0.))
        fuel_vel = torch.nan_to_num(fuel_vel.masked_fill(~fuel_mask[..., None], 0.),
                                     nan=0., posinf=0., neginf=0.)
        fuel_age = torch.nan_to_num(fuel_age, nan=self.timeout, posinf=self.timeout,
                                    neginf=0.).clamp_min(0.)
        fuel_confidence = torch.nan_to_num(fuel_confidence, nan=0.).clamp(0., 1.)
        fuel_extent = torch.nan_to_num(fuel_extent, nan=0., posinf=0., neginf=0.)

        own_pos = self._selected(env.sim.pose)[..., :2]
        own_vel = self._selected(env.sim.velocity)[..., :2]
        own_extent = torch.stack((self._selected(env.sim.length),
                                  self._selected(env.sim.width)), dim=-1)
        speed = self._selected(env.sim.speed).clamp_min(.1)[:, None, None]
        # Index only the two same-team slots. Enemy simulator poses and
        # velocities never enter this representation.
        ally_pos = env.sim.pose[:, self.teammates, :2].reshape(batch, 2, 2)
        ally_vel = env.sim.velocity[:, self.teammates, :2].reshape(batch, 2, 2)
        ally_extent = torch.stack((env.sim.length[:, self.teammates],
                                   env.sim.width[:, self.teammates]), -1).reshape(batch, 2, 2)
        opponent_mask = self._selected(env.opponent_valid).bool()
        opponent_pos = self._selected(env.opponent_pose)[..., :2]
        opponent_vel = self._selected(env.opponent_velocity)[..., :2]
        opponent_extent = self._selected(env.opponent_size)[..., :2]
        opponent_age = self._selected(env.opponent_age)
        opponent_mask &= torch.isfinite(opponent_pos).all(-1)
        positions = torch.cat((fuel_pos, own_pos[:, None], ally_pos,
                               opponent_pos[:, None]), dim=1)
        velocities = torch.cat((fuel_vel, own_vel[:, None], ally_vel,
                                opponent_vel[:, None]), dim=1)
        extents = torch.cat((fuel_extent, own_extent[:, None], ally_extent,
                             opponent_extent[:, None]), dim=1)
        ages = torch.cat((fuel_age, fuel_age.new_zeros((batch, 3)),
                          opponent_age[:, None]), dim=1)
        confidences = torch.cat((fuel_confidence, fuel_age.new_ones((batch, 3)),
                                 opponent_mask[:, None].float()), dim=1)
        mask = torch.cat((fuel_mask, fuel_mask.new_ones((batch, 3)),
                           opponent_mask[:, None]), dim=1)
        count = fuel_mask.shape[1]
        types = positions.new_zeros((batch, count + 4, 4))
        types[:, :count, 3] = 1.
        types[:, count, 0] = 1.
        types[:, count + 1:count + 3, 1] = 1.
        types[:, count + 3, 2] = 1.
        entities = torch.cat(((positions - own_pos[:, None]) / self.scales,
                              velocities / speed,
                              (ages / self.timeout).clamp(0., 1.)[..., None],
                              confidences[..., None], types,
                              extents / self.scales), dim=-1)
        entities = torch.nan_to_num(entities.masked_fill(~mask[..., None], 0.),
                                     nan=0., posinf=0., neginf=0.)

        field = self.static[None].expand(env.n, -1, -1, -1, -1).reshape(
            batch, 12, 32, 64).clone()
        fuel_cells = self._cells(fuel_pos)
        fuel_weights = fuel_mask.float()
        sums = field.new_zeros((batch, 4, 32 * 64))
        confidence_freshness = fuel_confidence * torch.exp(-fuel_age / self.timeout)
        values = torch.stack((fuel_weights,
                              fuel_vel[..., 0] / speed[:, 0],
                              fuel_vel[..., 1] / speed[:, 0],
                              confidence_freshness), dim=1) * fuel_weights[:, None]
        sums.scatter_add_(2, fuel_cells[:, None].expand(-1, 4, -1), values)
        flat = field.flatten(2)
        cell_count = sums[:, 0]
        flat[:, 2] = cell_count.log1p() / math.log1p(max(count, 1))
        flat[:, 3:6] = sums[:, 1:4] / cell_count[:, None].clamp_min(1.)
        flat[:, 11] = (cell_count > 0).float()
        robot_cells = self._cells(positions[:, count:])
        flat[:, 6].scatter_add_(1, robot_cells[:, :1], field.new_ones((batch, 1)))
        flat[:, 7].scatter_add_(1, robot_cells[:, 1:3], field.new_ones((batch, 2)))
        flat[:, 8].scatter_add_(1, robot_cells[:, 3:4], opponent_mask[:, None].float())
        return {"scalars": scalars, "field": field, "entities": entities,
                "entity_mask": mask}
