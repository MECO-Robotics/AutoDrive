"""Fuel lifecycle, pickup, passing, and scoring for the 3v3 env."""
from __future__ import annotations

import math

import torch

from . import tensor_3v3_pickup_grid_hip as _pickup_grid_hip
from ._tensor_3v3_constants import TEAM_IDS
from ._tensor_3v3_observation import NUM_ROBOTS


class TensorThreeVsThreeGamepieceMixin:
    """Manage gamepiece placement, ownership, scoring, and match timing."""
    def _randomize_pass_zone_positions(self, active=None):
        """Scatter passed fuel through alliance zones with an edge/corner bias."""
        if active is None:
            active=torch.ones(self.n,device=self.device,dtype=torch.bool)
        width=self.sim.field_width
        depth=self.alliance_zone_depth
        samples=self._random_active(active,(2,self.fuel_count,2))

        def edge_biased(unit, low, high):
            # Mix a mild edge preference with uniform placement. Strong edge
            # bias made most ferry drops collect along the zone borders.
            toward_upper=unit>=.5
            local=torch.where(toward_upper,(unit-.5)*2.,unit*2.)
            edge_fraction=torch.where(toward_upper,
                1.-.5*(1.-local).square(),.5*local.square())
            fraction=.65*unit+.35*edge_fraction
            return low+(high-low)*fraction

        margin=.28
        # The inner edge of each alliance zone faces its HUB. Restrict ferry
        # respawns to the outer portion of the zone so edge bias cannot put
        # every carrier's fuel into the HUB approach lane.
        outer_depth=depth-1.3
        red_x=margin+(outer_depth-margin)*samples[:,0,:,0].square()
        red_y=edge_biased(samples[:,0,:,1],margin,width-margin)
        blue_near_x=margin+(outer_depth-margin)*samples[:,1,:,0].square()
        blue_y=edge_biased(samples[:,1,:,1],margin,width-margin)
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
        active_count=self.n if mask is None else int(mask.sum().item())
        mask = (torch.ones(self.n, device=self.device, dtype=torch.bool) if mask is None else mask)
        positions = torch.zeros_like(self.piece_pos)
        active = torch.zeros_like(self.piece_active)
        owners = torch.full_like(self.piece_owner, -1)
        zones = torch.full_like(self.piece_zone, -1)
        half = .075
        positions[:, :24, 0] = .35 + self._random_active(mask,(24,),active_count=active_count) * .69
        positions[:, :24, 1] = 2.60 + self._random_active(mask,(24,),active_count=active_count) * .53
        positions[:, 24:48, 0] = 15.50 + self._random_active(mask,(24,),active_count=active_count) * .69
        positions[:, 24:48, 1] = 4.94 + self._random_active(mask,(24,),active_count=active_count) * .53
        active[:, :48] = True; zones[:, :24] = 1; zones[:, 24:48] = 2
        neutral_count = self.fuel_count - 96 - 6 * self.preloaded_per_robot
        if neutral_count:
            positions[:, 96:96 + neutral_count, 0] = 8.27 + (self._random_active(mask,(neutral_count,),active_count=active_count) - .5) * 1.83
            positions[:, 96:96 + neutral_count, 1] = 4.035 + (self._random_active(mask,(neutral_count,),active_count=active_count) - .5) * 5.23
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

    def _update_fuel(self, active, score_intent):
        self.fuel_acquired_event[active]=0; self.fuel_scored_event[active]=0
        self.fuel_passed_event[active]=0; self.fuel_denied_event[active]=0
        free = self.piece_active & (self.piece_owner < 0)
        # Track invalidation is independent of robot order. Collect all pieces
        # touched in this physics tick and clear the large [world, robot, piece]
        # track tensors once after pickup and passing are resolved.
        track_clear_mask = torch.zeros_like(free)
        # Fixed six-robot loop; all worlds and all 504 pieces stay tensorized.
        for robot in range(6):
            deterministic_offense = (
                self.control_modes[robot] == "deterministic" and
                self.robot_roles[robot] == "offense")
            policy_blocked = torch.zeros_like(free)
            if deterministic_offense:
                team = TEAM_IDS[robot]
                piece_in_alliance = (
                    self.piece_pos[..., 0] <= self.alliance_zone_depth if team == 0 else
                    self.piece_pos[..., 0] >= self.field_length - self.alliance_zone_depth)
                # During an active HUB shift the sweep can collect FUEL from
                # midfield and the rest of the field. During a closed shift,
                # gather outside the alliance zone for ferrying.
                allowed_zone = torch.where(
                    self.hub_active[:, team, None], torch.ones_like(piece_in_alliance),
                    ~piece_in_alliance)
                policy_blocked = free & ~allowed_zone
                free &= allowed_zone
            if (self.fuel_count <= 512 and
                    _pickup_grid_hip.integrated_available(self.device)):
                _pickup_grid_hip.pickup_robot(
                    active=active, pose=self.sim.pose, length=self.sim.length,
                    width=self.sim.width, piece_pos=self.piece_pos,
                    piece_active=self.piece_active, piece_owner=self.piece_owner,
                    free=free, possible_cells=self._pickup_possible_cells,
                    next_intake=self.next_intake, elapsed=self.match_elapsed,
                    controlled=self._controlled_mode_mask,
                    deterministic=self._deterministic_mode_mask,
                    defense_role=self._defense_role_mask,
                    acquired_event=self.fuel_acquired_event,
                    track_clear_mask=track_clear_mask, robot=robot,
                    nx=self._pickup_grid_nx, ny=self._pickup_grid_ny,
                    cell_size=self._pickup_grid_cell_size,
                    capacity=self.robot_fuel_capacity_values[robot],
                    allow_sweep=deterministic_offense)
                free |= policy_blocked
                continue
            pose = self.sim.pose[:,robot]
            delta = self.piece_pos-pose[:,None,:2]
            forward=torch.stack((pose[:,2].cos(),pose[:,2].sin()),-1)
            left=torch.stack((-pose[:,2].sin(),pose[:,2].cos()),-1)
            longitudinal=(delta*forward[:,None]).sum(-1)
            lateral=(delta*left[:,None]).sum(-1).abs()
            intake=(longitudinal>=self.sim.length[:,robot,None]*.5-.075)&(longitudinal<=self.sim.length[:,robot,None]*.5+.35)&(lateral<=self.sim.width[:,robot,None]*.5+.075)
            possession=(self.piece_owner==robot).sum(-1)
            robot_controlled=(self._nn_mode_mask[robot] |
                              self._deterministic_mode_mask[robot])
            sweep_enabled = torch.zeros((),device=self.device,dtype=torch.bool)
            sweep_picked = torch.zeros_like(free)
            if deterministic_offense:
                # Deterministic offense sweeps loose FUEL with its front
                # intake. Capture every contacted piece on this
                # physics tick instead of one piece per 200 ms intake cycle.
                sweep_enabled=(self._deterministic_mode_mask[robot] &
                               ~self._defense_role_mask[robot])
                sweep_candidates=(free&intake&active[:,None]&
                                  sweep_enabled&robot_controlled)
                room=(self.robot_fuel_capacities[robot]-possession).clamp_min(0)
                contact_rank=sweep_candidates.to(torch.int32).cumsum(-1)
                sweep_picked=sweep_candidates&(contact_rank<=room[:,None])
            eligible=(free&intake&(possession<self.robot_fuel_capacities[robot])[:,None]&
                     (self.match_elapsed[:,None]+1e-6>=self.next_intake[:,robot,None])&
                     active[:,None]&robot_controlled&~sweep_enabled)
            distance=delta.square().sum(-1).masked_fill(~eligible,float("inf"))
            nearest,index=distance.min(-1); picked=torch.isfinite(nearest)
            owner=self.piece_owner[self._world_indices,index]
            self.piece_owner[self._world_indices,index]=torch.where(picked,robot,owner)
            torch.where(sweep_picked,self._robot_owner_ids[robot],self.piece_owner,
                        out=self.piece_owner)
            picked_piece=sweep_picked.clone()
            picked_piece.scatter_(1,index[:,None],picked[:,None])
            track_clear_mask |= picked_piece
            self.fuel_acquired_event[:,robot]=torch.where(active,
                                                           picked.long()+sweep_picked.sum(-1),
                                                           self.fuel_acquired_event[:,robot])
            free[self._world_indices,index] &= ~picked
            free &= ~sweep_picked
            self.next_intake[:,robot]=torch.where(picked,self.match_elapsed+.20,self.next_intake[:,robot])
            free |= policy_blocked

        # Piece positions are world-major while ownership is per-piece. Gather
        # the owning robot's pose once, then apply pass-zone overrides in one
        # batched write before scoring is evaluated.
        owner_before_pass = self.piece_owner
        safe_owner = owner_before_pass.clamp(0, NUM_ROBOTS - 1)
        held = (self.piece_active & (owner_before_pass >= 0) &
                (owner_before_pass < NUM_ROBOTS) & active[:,None])
        carried_pose = self.sim.pose[self._world_indices[:,None], safe_owner, :2]
        torch.where(held[...,None], carried_pose, self.piece_pos,
                    out=self.piece_pos)

        at_ferry = ((self.sim.pose[:,:,:2] - self.ferry_targets[None]).norm(dim=-1)
                    <= .35)
        home_zone_center=self.hub_centers[self.team_ids].clone()
        home_zone_center[:,0]=torch.where(
            self.team_ids==0,self.alliance_zone_depth*.5,
            self.field_length-self.alliance_zone_depth*.5)
        ferry_vector=home_zone_center[None]-self.sim.pose[:,:,:2]
        ferry_bearing=torch.atan2(ferry_vector[...,1],ferry_vector[...,0])
        ferry_heading_error=torch.atan2(
            torch.sin(ferry_bearing-self.sim.pose[:,:,2]),
            torch.cos(ferry_bearing-self.sim.pose[:,:,2])).abs()
        # The Dumper ejects only through its front. A ferry release therefore
        # waits until the robot is turned toward its friendly alliance zone.
        facing_home=((ferry_heading_error<=math.pi/4)|self._turret_mask[None])
        rate_ready=(self.match_elapsed[:,None]+1e-6>=self.next_ferry)
        at_release_position=at_ferry|self._dumper_mask[None]
        robot_pass = ((self.last_actions == 6) & at_release_position & facing_home &
                      rate_ready & active[:,None])
        first_held_piece=torch.stack([
            (owner_before_pass==robot).to(torch.int64).argmax(-1)
            for robot in range(NUM_ROBOTS)],dim=1)
        selected_for_pass=(self._fuel_indices==first_held_piece.gather(
            1,safe_owner))
        passed = held & selected_for_pass & robot_pass.gather(1, safe_owner)
        # Draw a fresh landing point for each robot each physics tick. Only
        # robots that actually pass this tick use their sample. Reusing a
        # fixed location per FUEL id caused repeated ferry cycles to stack at
        # the same points.
        pass_random=torch.rand((self.n,NUM_ROBOTS,2),device=self.device,
                               generator=self.generator)
        outer_depth=self.alliance_zone_depth-1.3
        margin=.28
        pass_x_unit=pass_random[...,0]
        pass_x_fraction=.65*pass_x_unit+.35*pass_x_unit.square()
        pass_x_near=margin+(outer_depth-margin)*pass_x_fraction
        upper=pass_random[...,1]>=.5
        local=torch.where(upper,(pass_random[...,1]-.5)*2.,pass_random[...,1]*2.)
        edge_y_fraction=torch.where(upper,
            1.-.5*(1.-local).square(),.5*local.square())
        pass_y_fraction=.65*pass_random[...,1]+.35*edge_y_fraction
        pass_y=margin+(self.sim.field_width-2.*margin)*pass_y_fraction
        pass_x=torch.where(self.team_ids[None]==0,pass_x_near,
                           self.sim.field_length-pass_x_near)
        pass_destinations=torch.stack((pass_x,pass_y),-1)
        pass_positions=pass_destinations[self._world_indices[:,None],safe_owner]
        torch.where(passed[...,None], pass_positions, self.piece_pos,
                    out=self.piece_pos)
        pass_cell_size=self._pickup_grid_cell_size
        pass_cell_x=(pass_positions[...,0]/pass_cell_size).floor().long().clamp_(
            0,self._pickup_grid_nx-1)
        pass_cell_y=(pass_positions[...,1]/pass_cell_size).floor().long().clamp_(
            0,self._pickup_grid_ny-1)
        pass_cell=pass_cell_y*self._pickup_grid_nx+pass_cell_x
        pass_slot=(self.team_ids[safe_owner]+1).clamp_(1,2)
        world_index=self._world_indices[:,None].expand_as(safe_owner)
        piece_index=self._fuel_indices.expand_as(safe_owner)
        old_cell=self._pickup_possible_cells[world_index,piece_index,pass_slot]
        self._pickup_possible_cells[world_index,piece_index,pass_slot]=torch.where(
            passed,pass_cell.to(old_cell.dtype),old_cell)
        self.piece_vel.masked_fill_(passed[...,None], 0.)
        self.piece_owner.masked_fill_(passed, -1)
        home_zone=self.team_ids[safe_owner]+1
        torch.where(passed,home_zone,self.piece_zone,out=self.piece_zone)
        passed_counts = torch.zeros_like(self.fuel_passed_event)
        passed_counts.scatter_add_(1, safe_owner,
                                   passed.to(self.fuel_passed_event.dtype))
        self.fuel_passed_event.copy_(torch.where(
            active[:,None], passed_counts, self.fuel_passed_event))
        for robot in range(NUM_ROBOTS):
            passed_one=passed_counts[:,robot]>0
            interval=self.robot_score_interval_values[robot]
            self.next_ferry[:,robot]=torch.where(
                passed_one,self.match_elapsed+interval,self.next_ferry[:,robot])
        track_clear_mask |= passed

        newly_scored=torch.zeros_like(self.piece_active)
        bump_clear=self._robots_clear_of_bumps(self.sim.pose[:,:,:2])
        for robot in range(6):
            # Alliance membership is a fixed robot-slot constant. Reading the
            # equivalent CUDA tensor with int(...) synchronizes the device for
            # each robot on every 50 Hz physics tick.
            team=TEAM_IDS[robot]; held=self.piece_active&(self.piece_owner==robot)&active[:,None]
            hub=self.hub_centers[team]; delta=hub-self.sim.pose[:,robot,:2]
            bearing=torch.atan2(delta[:,1],delta[:,0])
            error=torch.atan2(torch.sin(bearing-self.sim.pose[:,robot,2]),torch.cos(bearing-self.sim.pose[:,robot,2])).abs()
            if self.robot_types[robot]=="dumper":
                stationary=(self.sim.velocity[:,robot,:2].norm(dim=-1)<=.10)&(
                    self.sim.velocity[:,robot,2].abs()<=.10)
                aimed=(error<=math.pi/18)&stationary
            else:
                aimed=torch.ones_like(error,dtype=torch.bool)
            robot_x=self.sim.pose[:,robot,0]
            in_alliance_zone=(robot_x<=self.alliance_zone_depth if team==0 else
                              robot_x>=self.field_length-self.alliance_zone_depth)
            ready=in_alliance_zone&aimed&bump_clear[:,robot]
            has_fuel=held.any(-1)
            entering=(ready&~self.last_hub_zone[:,robot]&has_fuel&active&
                      (~self.hub_active[:,team]|~in_alliance_zone))
            self.fuel_denied_event[:,robot]=torch.where(active,entering.long(),
                                                         self.fuel_denied_event[:,robot])
            self.fuel_denied_count[:,robot]+=entering.long()
            can_score=(ready&score_intent[:,robot]&
                (self.match_elapsed+1e-6>=self.next_score[:,robot])&
                self.hub_active[:,team]&in_alliance_zone&active)
            eligible=held&can_score[:,None]
            chosen=eligible&(self._fuel_indices==eligible.to(torch.int64).argmax(-1,keepdim=True))
            count=chosen.sum(-1); newly_scored|=chosen
            self.fuel_scored_event[:,robot]=torch.where(active,count,
                                                         self.fuel_scored_event[:,robot])
            self.fuel_score_count[:,team]+=count
            self.auto_fuel_scores[:,team]+=count*(self.match_elapsed<=20.+self.dt).long()
            self.next_score[:,robot]=torch.where(
                count>0,self.match_elapsed+self.robot_score_interval_values[robot],
                self.next_score[:,robot])
            self.last_hub_zone[:,robot]=torch.where(active,ready,self.last_hub_zone[:,robot])

        track_clear_mask |= newly_scored
        clear_tracks = track_clear_mask[:,None,:]
        self.track_mask.masked_fill_(clear_tracks, False)
        self.track_age.masked_fill_(clear_tracks, float("inf"))
        torch.where(newly_scored[...,None],self._midfield_respawn_positions[None],
                    self.piece_pos,out=self.piece_pos)
        self.piece_vel.masked_fill_(newly_scored[...,None],0.)
        torch.where(newly_scored,torch.full_like(self.piece_owner,-1),
                    self.piece_owner,out=self.piece_owner)
        torch.where(newly_scored,torch.zeros_like(self.piece_zone),
                    self.piece_zone,out=self.piece_zone)
        torch.where(newly_scored[...,None],self._midfield_respawn_cells,
                    self._pickup_possible_cells,out=self._pickup_possible_cells)
        self.fuel_acquisition_count += self.fuel_acquired_event*active[:,None]
