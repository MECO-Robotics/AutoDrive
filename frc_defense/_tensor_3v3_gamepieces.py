"""Fuel staging and match timing for the 3v3 environment."""
from __future__ import annotations

import math
import torch

from .field import grid_array_points
from ._tensor_3v3_observation import NUM_ROBOTS
from .gamepiece_actions import update_fuel_actions


def _edge_biased_unit(unit):
    """Mix a broad uniform distribution with a mild, symmetric edge bias."""
    toward_upper=unit>=.5
    local=torch.where(toward_upper,(unit-.5)*2.,unit*2.)
    edge_fraction=torch.where(toward_upper,
        1.-.5*(1.-local).square(),.5*local.square())
    return .65*unit+.35*edge_fraction


class TensorThreeVsThreeGamepieceMixin:
    """Manage gamepiece placement, ownership, scoring, and match timing."""
    _fuel_radius = .0825

    def _update_fuel(self, active, score_intent):
        return update_fuel_actions(self, active, score_intent)

    def _randomize_pass_zone_positions(self, active=None):
        """Scatter passed fuel through alliance zones with an edge/corner bias."""
        if active is None:
            active=torch.ones(self.n,device=self.device,dtype=torch.bool)
        width=self.sim.field_width
        depth=self.alliance_zone_depth
        samples=self._random_active(active,(2,self.fuel_count,2))

        margin=.28
        # Keep clear of the HUB approach lane while spreading drops over the
        # whole safe band, with a mild preference for either x edge.
        outer_depth=depth-1.3
        red_x=margin+(outer_depth-margin)*_edge_biased_unit(samples[:,0,:,0])
        red_y=margin+(width-2.*margin)*_edge_biased_unit(samples[:,0,:,1])
        blue_near_x=margin+(outer_depth-margin)*_edge_biased_unit(samples[:,1,:,0])
        blue_y=margin+(width-2.*margin)*_edge_biased_unit(samples[:,1,:,1])
        red=torch.stack((red_x,red_y),-1)
        blue=torch.stack((self.field_length-blue_near_x,blue_y),-1)
        sampled=torch.stack((red,blue),1)
        torch.where(active[:,None,None,None],sampled,self.pass_zone_positions,
                    out=self.pass_zone_positions)

        respawn_x=(self._midfield_respawn_positions[:,0] /
                   self._pickup_grid_cell_size).floor().long().clamp_(
                       0,self._pickup_grid_nx-1)
        respawn_y=(self._midfield_respawn_positions[:,1] /
                   self._pickup_grid_cell_size).floor().long().clamp_(
                       0,self._pickup_grid_ny-1)
        respawn_cell=respawn_y*self._pickup_grid_nx+respawn_x
        pass_x=(self.pass_zone_positions[...,0] /
                self._pickup_grid_cell_size).floor().long().clamp_(
                    0,self._pickup_grid_nx-1)
        pass_y=(self.pass_zone_positions[...,1] /
                self._pickup_grid_cell_size).floor().long().clamp_(
                    0,self._pickup_grid_ny-1)
        pass_cell=pass_y*self._pickup_grid_nx+pass_x
        cells=torch.stack((respawn_cell[None].expand(self.n,-1),
                           pass_cell[:,0],pass_cell[:,1]),-1).to(torch.int16)
        torch.where(active[:,None,None],cells,self._midfield_respawn_cells,
                    out=self._midfield_respawn_cells)

    def _robots_clear_of_bumps(self, positions):
        """Check robot footprints against the rectangular BUMP regions."""
        bumps=self.sim.bump_regions
        if not bumps.numel():
            return torch.ones(positions.shape[:-1],device=self.device,dtype=torch.bool)
        radii=.5*torch.sqrt(self.sim.length.square()+self.sim.width.square())
        dx=(positions[...,None,0]-bumps[None,None,:,0]).abs()
        dy=(positions[...,None,1]-bumps[None,None,:,1]).abs()
        overlaps=((dx<=bumps[None,None,:,2]+radii[:,:,None]) &
                  (dy<=bumps[None,None,:,3]+radii[:,:,None]))
        return ~overlaps.any(-1)

    def _initialize_fuel(self, mask=None):
        mask = (torch.ones(self.n, device=self.device, dtype=torch.bool) if mask is None else mask)
        positions = torch.zeros_like(self.piece_pos)
        active = torch.zeros_like(self.piece_active)
        owners = torch.full_like(self.piece_owner, -1)
        zones = torch.full_like(self.piece_zone, -1)
        half = .075
        depots = {box.name: box for box in self.field_boxes}
        for begin, name in ((0, "red_depot"), (24, "blue_depot")):
            depot = depots[name]
            x_min = max(half, depot.x - depot.length / 2)
            x_max = min(self.sim.field_length - half, depot.x + depot.length / 2)
            y_min = max(half, depot.y - depot.width / 2)
            y_max = min(self.sim.field_width - half, depot.y + depot.width / 2)
            positions[:, begin:begin + 24] = torch.tensor(
                grid_array_points(24, x_min, x_max, y_min, y_max),
                device=self.device, dtype=positions.dtype)
        active[:, :48] = True; zones[:, :24] = 1; zones[:, 24:48] = 2
        neutral_count = (self.fuel_count - 96 - 6 * self.preloaded_per_robot -
                         self._probe_respawn_capacity)
        self._probe_respawn_start=96+neutral_count
        self._probe_respawn_cursor[mask]=0
        if neutral_count:
            positions[:, 96:96 + neutral_count] = torch.tensor(
                grid_array_points(neutral_count, 8.27-.915, 8.27+.915,
                                  4.035-2.615, 4.035+2.615),
                device=self.device, dtype=positions.dtype)
            active[:, 96:96 + neutral_count] = True
            zones[:, 96:96 + neutral_count] = 0
        if self.preloaded_per_robot:
            for robot in range(NUM_ROBOTS):
                begin = self.fuel_count - 6 * self.preloaded_per_robot + robot * self.preloaded_per_robot
                end = begin + self.preloaded_per_robot
                positions[:, begin:end] = self.sim.pose[:, robot, None, :2]
                active[:, begin:end] = True; owners[:, begin:end] = robot
                zones[:, begin:end] = 5
        self.piece_pos.copy_(torch.where(mask[:, None, None], positions, self.piece_pos))
        self.piece_vel[mask] = 0.
        self.piece_active.copy_(torch.where(mask[:, None], active, self.piece_active))
        self.piece_owner.copy_(torch.where(mask[:, None], owners, self.piece_owner))
        self.piece_zone.copy_(torch.where(mask[:, None], zones, self.piece_zone))
        cell_size = self._pickup_grid_cell_size
        initial_x = (positions[..., 0] / cell_size).floor().long().clamp_(0, self._pickup_grid_nx - 1)
        initial_y = (positions[..., 1] / cell_size).floor().long().clamp_(0, self._pickup_grid_ny - 1)
        initial_cell = initial_y * self._pickup_grid_nx + initial_x
        pass_x = (self.pass_zone_positions[..., 0] / cell_size).floor().long().clamp_(0, self._pickup_grid_nx - 1)
        pass_y = (self.pass_zone_positions[..., 1] / cell_size).floor().long().clamp_(0, self._pickup_grid_ny - 1)
        pass_cell = pass_y * self._pickup_grid_nx + pass_x
        possible = torch.stack((initial_cell, pass_cell[:,0],pass_cell[:,1]),-1).to(torch.int16)
        self._pickup_possible_cells.copy_(torch.where(
            mask[:, None, None], possible, self._pickup_possible_cells))

    def _update_match(self, active):
        self.match_elapsed.copy_(torch.where(active,(self.match_elapsed+self.dt).clamp(max=160.),self.match_elapsed))
        self.match_remaining.copy_(torch.where(active,(160.-self.match_elapsed).clamp_min(0.),self.match_remaining))
        if self.behavior_probe is not None:
            values=torch.ones_like(self.hub_active)
            values[:,0]=self.behavior_probe_hub_active
            values[:,1]=not self.behavior_probe_hub_active
            self.hub_active.copy_(torch.where(active[:,None],values,self.hub_active))
            return
        first = torch.where(self.auto_fuel_scores[:,0] > self.auto_fuel_scores[:,1], 0,
                torch.where(self.auto_fuel_scores[:,1] > self.auto_fuel_scores[:,0], 1,
                            torch.zeros_like(self.steps)))
        # REBUILT: AUTO + transition are 30 s; four 25 s alliance shifts
        # run from 30–130 s, followed by a 30 s end game. The AUTO winner's
        # HUB is inactive in shift 1, then HUB status alternates each shift.
        cycle=((self.match_elapsed-30.).clamp_min(0.)/25.).floor().long()
        inactive=torch.where(cycle%2==0,first,1-first)
        values=torch.ones_like(self.hub_active)
        in_alliance_shifts=((self.match_elapsed>=30.) & (self.match_elapsed<130.))
        values.scatter_(1,inactive[:,None],(~in_alliance_shifts)[:,None])
        values=torch.where((self.match_elapsed >= 130.)[:,None],
                           torch.ones_like(values),values)
        self.hub_active.copy_(torch.where(active[:,None],values,self.hub_active))
