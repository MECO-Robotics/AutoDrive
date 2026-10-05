"""Perception updates and observation construction for the 3v3 environment."""
from __future__ import annotations

import math

import torch

from .tensor_sim import normalize_tensor_observation_batch_in_place
from . import tensor_fuel_candidate_rank as _fuel_candidate_rank_hip


NUM_ROBOTS = 6
OBS_DIM = 137
ACTION_DIM = 8


class TensorThreeVsThreeObservationMixin:
    """Methods that maintain sensor tracks and build policy observations."""

    @property
    def num_agents(self):
        return self.n * NUM_ROBOTS

    def _random(self, shape):
        return torch.rand(shape, device=self.device, generator=self.generator)

    def _random_active(self, active, tail_shape, *, normal=False, dtype=None,
                       active_count=None):
        """Draw only for active worlds so stopped episodes do not consume RNG."""
        count=(int(active.sum().item()) if active_count is None else int(active_count))
        result=torch.zeros((self.n,*tail_shape),device=self.device,
                           dtype=dtype or torch.float32)
        if count:
            shape=(count,*tail_shape)
            sample=(torch.randn(shape,device=self.device,generator=self.generator)
                    if normal else torch.rand(shape,device=self.device,generator=self.generator))
            result[active]=sample
        return result

    def _update_perception(self, active, active_count=None):
        """Update per-robot free-FUEL and nearest-opponent tracks."""
        pose = self.sim.pose
        free = self.piece_active & (self.piece_owner < 0)
        from . import tensor_perception
        fused_visible = tensor_perception.visibility_mask_3v3(
            pose.contiguous(), self.piece_pos.contiguous(), self.piece_active,
            self.piece_owner, active, self.perception_range,
            self.perception_fov_degrees, self.field_feature_obstacles,
            self._robot_occlusion_radius)
        fused_tracks = (self.fused_sensor_rng and fused_visible is not None and
            tensor_perception.perception_commit_3v3(
                fused_visible, active.contiguous(), self._perception_rng_ticks,
                self.piece_pos.contiguous(), self.piece_vel.contiguous(),
                self.track_pos, self.track_vel, self.track_age, self.track_mask,
                self._perception_rng_seed, self.dt, self.perception_dropout,
                self.position_noise, self.velocity_noise))
        self._fused_sensor_rng_used |= bool(fused_tracks)
        if not fused_tracks:
            self.track_age = torch.where(active[:, None, None] & self.track_mask,
                                         self.track_age + self.dt, self.track_age)
        fused_opponents = (fused_tracks and
            tensor_perception.opponent_tracks_3v3(
                pose.contiguous(), self.sim.length.contiguous(),
                self.sim.width.contiguous(), self.sim.accel.contiguous(),
                self._robot_occlusion_radius.contiguous(),
                self.field_feature_obstacles.contiguous(), active.contiguous(),
                self._controlled_mode_mask.contiguous(),
                self._defense_role_mask.contiguous(), self._perception_rng_ticks,
                self.opponent_pose, self.opponent_velocity, self.opponent_size,
                self.opponent_age, self.opponent_valid, self._perception_rng_seed,
                self.perception_range, self.perception_fov_degrees,
                self.perception_dropout, self.position_noise,
                self.velocity_noise, self.dt))
        self._fused_opponent_tracks_used |= bool(fused_opponents)
        if not fused_opponents:
            self.opponent_age = torch.where(active[:, None] & self.opponent_valid,
                                            self.opponent_age + self.dt,
                                            self.opponent_age)
        for robot in range(NUM_ROBOTS):
            own = pose[:, robot]
            if fused_visible is not None:
                visible = fused_visible[:, robot]
            else:
                delta = self.piece_pos - own[:, None, :2]
                distance = delta.norm(dim=-1)
                visible = free & (distance <= self.perception_range)
                if self.perception_fov < math.pi:
                    bearing = torch.atan2(delta[..., 1], delta[..., 0])
                    error = torch.atan2(torch.sin(bearing - own[:, None, 2]),
                                        torch.cos(bearing - own[:, None, 2])).abs()
                    visible &= error <= self.perception_fov
                if self.field_feature_obstacles.numel():
                    blocked = tensor_perception.piece_occlusion_mask(
                        own[:, :2].contiguous(), delta,
                        self.field_feature_obstacles.contiguous(),
                        (visible & active[:, None]).contiguous())
                    visible &= ~blocked
                # Evaluate all other chassis together; the old inner robot loop
                # launched five separate GPU kernels per tracked FUEL row.
                denom=delta.square().sum(-1).clamp_min(1e-8)
                peers=self.sim.pose[:,:,:2]
                rel=peers-own[:,None,:2]
                fraction=(rel[:,:,None,:]*delta[:,None,:,:]).sum(-1)/denom[:,None,:]
                closest=own[:,None,None,:2]+fraction.clamp(0.,1.)[...,None]*delta[:,None,:,:]
                radius=self._robot_occlusion_radius
                peer_blocked=((closest-peers[:,:,None,:]).norm(dim=-1)<=radius[:,:,None])
                peer_blocked &= (fraction>.02)&(fraction<.98)
                peer_ids=self._peer_robot_ids[robot]
                visible &= ~peer_blocked[:,peer_ids].any(1)
            if not fused_tracks:
                if self.perception_dropout:
                    dropout=self._random_active(active,(self.fuel_count,),active_count=active_count)
                    visible &= dropout >= self.perception_dropout
                noise = self._random_active(active,(self.fuel_count,2),normal=True,active_count=active_count) * self.position_noise
                vnoise = self._random_active(active,(self.fuel_count,2),normal=True,active_count=active_count) * self.velocity_noise
                row_visible = visible & active[:, None]
                self.track_pos[:, robot] = torch.where(row_visible[..., None],
                    self.piece_pos + noise, self.track_pos[:, robot])
                self.track_vel[:, robot] = torch.where(row_visible[..., None],
                    self.piece_vel + vnoise, self.track_vel[:, robot])
                self.track_age[:, robot] = torch.where(row_visible, torch.zeros_like(self.track_age[:, robot]),
                                                       self.track_age[:, robot])
                self.track_mask[:, robot] |= row_visible

            if fused_opponents:
                continue
            enemy_ids = self._enemy_robot_ids[robot]
            enemy_pose = pose[:, enemy_ids]
            enemy_delta = enemy_pose[..., :2] - own[:, None, :2]
            enemy_distance = enemy_delta.norm(dim=-1)
            enemy_bearing = torch.atan2(enemy_delta[..., 1], enemy_delta[..., 0])
            bearing_error = torch.atan2(torch.sin(enemy_bearing-own[:, None, 2]),
                                        torch.cos(enemy_bearing-own[:, None, 2])).abs()
            in_view = (enemy_distance <= self.perception_range) & (bearing_error <= self.perception_fov)
            # In mixed 1v1 scenario playback, unassigned slots remain as
            # stationary field obstacles. Do not let a scripted defender
            # treat those empty slots as an attacking robot when an assigned
            # opponent is present.
            if self.robot_roles[robot]=="defense":
                in_view &= self._controlled_mode_mask[enemy_ids][None]
            in_view &= active[:, None]
            if self.field_feature_obstacles.numel():
                from .tensor_perception import piece_occlusion_mask
                in_view &= ~piece_occlusion_mask(own[:,:2].contiguous(),enemy_delta,
                    self.field_feature_obstacles.contiguous(),in_view.contiguous())
            # For each of the three enemy tracks, test all six possible robot
            # occluders together. Exclude the observing robot and the tracked
            # enemy chassis itself, matching the original visibility rule.
            peers=self.sim.pose[:,:,:2]
            peer_rel=peers-own[:,None,:2]
            denom=enemy_delta.square().sum(-1).clamp_min(1e-8)
            fraction=(enemy_delta[:,:,None,:]*peer_rel[:,None,:,:]).sum(-1)/denom[:,:,None]
            closest=own[:,None,None,:2]+fraction.clamp(0.,1.)[...,None]*enemy_delta[:,:,None,:]
            blocking_radius=self._robot_occlusion_radius
            occluded=((closest-peers[:,None,:,:]).norm(dim=-1)<=blocking_radius[:,None,:])
            occluded &= (fraction>.02)&(fraction<.98)
            valid_occluder=self._valid_enemy_occluders[robot]
            in_view &= ~ (occluded & valid_occluder[None]).any(-1)
            nearest = enemy_distance.masked_fill(~in_view, float("inf")).argmin(-1)
            has_enemy = in_view.any(-1)
            chosen = enemy_pose[self._world_indices, nearest]
            if self.perception_dropout:
                has_enemy &= self._random_active(active,(),active_count=active_count) >= self.perception_dropout
            pn = self._random_active(active,(2,),normal=True,active_count=active_count) * self.position_noise
            hn = self._random_active(active,(),normal=True,active_count=active_count) * min(.05, self.position_noise)
            vn = self._random_active(active,(3,),normal=True,active_count=active_count) * self.velocity_noise
            chosen = chosen.clone(); chosen[:, :2] += pn; chosen[:, 2] += hn
            chosen_velocity = enemy_pose[self._world_indices, nearest].clone() + vn
            self.opponent_pose[:, robot] = torch.where(has_enemy[:, None], chosen,
                                                       self.opponent_pose[:, robot])
            self.opponent_velocity[:, robot] = torch.where(has_enemy[:, None], chosen_velocity,
                                                           self.opponent_velocity[:, robot])
            enemy_size = torch.stack((self.sim.length[:, enemy_ids],
                                      self.sim.width[:, enemy_ids],
                                      self.sim.accel[:, enemy_ids] / 10.), -1)
            chosen_size = enemy_size[self._world_indices, nearest]
            self.opponent_size[:, robot] = torch.where(has_enemy[:, None], chosen_size,
                                                       self.opponent_size[:, robot])
            self.opponent_age[:, robot] = torch.where(has_enemy, torch.zeros_like(self.opponent_age[:, robot]),
                                                      self.opponent_age[:, robot])
            self.opponent_valid[:, robot] |= has_enemy
        if fused_tracks:
            self._perception_rng_ticks.add_(active.long())
        else:
            self.track_mask &= ~((self.track_age > 1.) & active[:, None, None])
        if not fused_opponents:
            self.opponent_valid &= ~((self.opponent_age > 1.) & active[:, None])

    def _candidates(self, candidate_mask=None):
        candidate_mask = self.track_mask if candidate_mask is None else candidate_mask
        if (_fuel_candidate_rank_hip is not None and
                _fuel_candidate_rank_hip.integrated_available(self.device)):
            indices, valid, nearest = _fuel_candidate_rank_hip.candidates(
                self.track_pos.reshape(self.n * NUM_ROBOTS, self.fuel_count, 2),
                self.sim.pose[:, :, :2].reshape(self.n * NUM_ROBOTS, 2),
                candidate_mask.reshape(self.n * NUM_ROBOTS, self.fuel_count))
            return (indices.reshape(self.n, NUM_ROBOTS, 4),
                    valid.reshape(self.n, NUM_ROBOTS, 4),
                    nearest.reshape(self.n, NUM_ROBOTS, 4))
        delta = self.track_pos - self.sim.pose[:, :, None, :2]
        distance = delta.norm(dim=-1).masked_fill(~candidate_mask, float("inf"))
        nearest, indices = distance.topk(4, dim=-1, largest=False)
        return indices, torch.isfinite(nearest), nearest

    def _action_mask(self, indices, valid):
        mask = torch.zeros((self.n, NUM_ROBOTS, ACTION_DIM), device=self.device,
                           dtype=torch.bool)
        possession = torch.stack([(self.piece_owner == r).sum(-1) for r in range(NUM_ROBOTS)], dim=1)
        mask[..., :4] = valid & (possession[..., None] <
                                  self.robot_fuel_capacities[None,:,None])
        mask[..., 4] = possession > 0
        mask[..., 5] = self.opponent_valid | self._defense_role_mask[None]
        mask[..., 6] = True
        mask[..., 7] = True
        return mask, possession

    def observe(self, active_mask=None):
        active = (torch.ones(self.n, device=self.device, dtype=torch.bool) if active_mask is None else
                  torch.as_tensor(active_mask, device=self.device, dtype=torch.bool).reshape(self.n))
        p, v = self.sim.pose, self.sim.velocity
        indices, valid, eta = self._candidates()
        action_mask, possession = self._action_mask(indices, valid)
        rows = self._world_indices[:, None]
        same_team = (self.team_ids[:, None] == self.team_ids[None, :]) & ~torch.eye(NUM_ROBOTS, device=self.device, dtype=torch.bool)
        teammate_dist = (p[:, None, :, :2] - p[:, :, None, :2]).norm(dim=-1)
        teammate_dist = teammate_dist.masked_fill(~same_team[None], float("inf"))
        team_idx = teammate_dist.topk(2, dim=-1, largest=False).indices
        team_pose = p[:,None].expand(-1,6,-1,-1).gather(
            2, team_idx[...,None].expand(-1,-1,-1,3))
        team_vel = v[:,None].expand(-1,6,-1,-1).gather(
            2, team_idx[...,None].expand(-1,-1,-1,3))
        robot_possession=torch.stack([(self.piece_owner==r).sum(-1)
                                      for r in range(NUM_ROBOTS)],1)
        teammate_possession=robot_possession[:,None,:].expand(-1,NUM_ROBOTS,-1).gather(2,team_idx)
        # Nearest opponent's measured track is the same one the controller sees.
        opponent = self.opponent_pose
        opponent_v = self.opponent_velocity
        other_valid = self.opponent_valid.to(p.dtype)
        other_age = (self.opponent_age / 1.).clamp(0., 1.)
        lengths = self.sim.length; widths = self.sim.width; accels = self.sim.accel
        own_size = torch.stack((lengths, widths, accels / 10.), -1)
        # Four nearest fixed field features as (x, y, radius) pairs.
        feature = self.field_feature_obstacles
        if feature.numel():
            fdist = (feature[None, None, :, :2] - p[:, :, None, :2]).square().sum(-1)
            fsel = fdist.topk(min(4, feature.shape[0]), dim=-1, largest=False).indices
            obs_feat = feature[fsel].reshape(self.n, NUM_ROBOTS, -1)
            if obs_feat.shape[-1] < 12:
                obs_feat = torch.nn.functional.pad(obs_feat, (0, 12-obs_feat.shape[-1]))
        else:
            obs_feat = torch.zeros((self.n, NUM_ROBOTS, 12), device=self.device)
        field = self._field_scales
        base = torch.cat((p, v, opponent, opponent_v, other_valid[..., None], other_age[..., None],
                          torch.full((self.n, 6, 1), .595, device=self.device),
                          field.expand(self.n, 6, 2), own_size, self.opponent_size, obs_feat), -1)
        # Three route values from current cached AD* route state.
        path = self.planner.last_path.reshape(self.n, 6, -1, 2)
        path_length = self.planner.last_lengths.reshape(self.n, 6)
        first = path[:, :, 1] - p[:, :, :2]
        segments = (path[:, :, 1:] - path[:, :, :-1]).norm(dim=-1)
        path_mask = self._route_segment_indices[None, None, :] < (path_length-1).clamp_min(0)[..., None]
        route_cost = (segments * path_mask).sum(-1)
        route = torch.stack((first[..., 0]/self.sim.field_length,
            first[..., 1]/self.sim.field_width,
            route_cost/math.hypot(self.sim.field_length,self.sim.field_width)), -1)
        route = torch.where((path_length > 1)[..., None], route, torch.zeros_like(route))
        local_fuel_points = self.track_pos.gather(2, indices[...,None].expand(-1,-1,-1,2))
        local_fuel_vel = self.track_vel.gather(2, indices[...,None].expand(-1,-1,-1,2))
        local = torch.cat(((local_fuel_points-p[:, :, None, :2]) /
            self._field_scales,
            local_fuel_vel / self.sim.speed[:, :, None, None].clamp_min(.1),
            valid[..., None].to(p.dtype)), -1).reshape(self.n,6,20)
        teammate = torch.cat(((team_pose[..., :2]-p[:, :, None, :2]) /
            self._field_scales,
            team_vel[..., :2] / self.sim.speed[:, :, None, None].clamp_min(.1),
            team_pose[..., 2:3].sin(), team_pose[..., 2:3].cos(),
            teammate_possession[...,None].to(p.dtype) /
                max(self.fuel_capacity,1),
            self.agent_train_mask[team_idx].to(p.dtype)[...,None],
            torch.ones((self.n,6,2,2),device=self.device)), -1).reshape(self.n,6,20)
        own_count = possession.to(p.dtype) / max(self.fuel_capacity,1)
        same_team_float=same_team.to(p.dtype)
        teammate_count=(torch.einsum("ij,nj->ni",same_team_float,possession.to(p.dtype)) /
                        (2*max(self.fuel_capacity,1)))
        possession_feat = torch.stack((own_count, teammate_count), -1)
        team_scores = self.fuel_score_count / max(self.fuel_count,1)
        relative_hubs = ((self.hub_centers[None,None] - p[:, :, None, :2]) /
            self._field_scales)
        match = torch.cat((self.match_elapsed[:,None,None].expand(-1,6,1)/160.,
                           self.hub_active[:,None].expand(-1,6,-1).to(p.dtype),
                           team_scores[:,None].expand(-1,6,-1), relative_hubs.reshape(self.n,6,4)), -1)
        candidate_points = local_fuel_points
        own_eta = torch.where(valid,
            eta / self.sim.speed[:, :, None].clamp_min(.1),torch.zeros_like(eta))
        enemy_distance = (candidate_points - opponent[:, :, None, :2]).norm(dim=-1)
        enemy_eta = enemy_distance / self.sim.speed.mean(-1)[:, None, None].clamp_min(.1)
        team = self.team_ids[None, :, None]
        hub = self.hub_centers[self.team_ids][None]
        to_score = (candidate_points-hub[:,:,None]).norm(dim=-1) / self.sim.speed[:, :, None].clamp_min(.1)
        zone = (candidate_points[...,0]/self.sim.field_length*6.).floor().clamp(0,5)/6.
        risk = torch.where(valid,torch.sigmoid((own_eta-enemy_eta)*2.),
                           torch.zeros_like(own_eta))
        candidates = torch.stack((
            (candidate_points[...,0]-p[:,:,None,0])/self.sim.field_length,
            (candidate_points[...,1]-p[:,:,None,1])/self.sim.field_width,
            own_eta/20.,enemy_eta/20.,to_score/20.,risk,zone,
            own_count[...,None].expand(-1,-1,4),
            (1-own_count[...,None]).expand(-1,-1,4),valid.to(p.dtype)), -1).reshape(self.n,6,40)
        obs = torch.cat((base, route, local, teammate, possession_feat, match,
                         candidates, action_mask.to(p.dtype)), -1)
        if obs.shape[-1] != OBS_DIM:
            raise RuntimeError(f"3v3 observation width {obs.shape[-1]} != {OBS_DIM}")
        if self.normalize_observations:
            flat = obs.reshape(-1, OBS_DIM)
            sp = torch.stack((self.sim.speed, self.sim.speed), -1).reshape(-1, 2)
            om = torch.stack((self.sim.omega_limit,self.sim.omega_limit),-1).reshape(-1,2)
            normalize_tensor_observation_batch_in_place(flat,self.sim.field_length,
                                                        self.sim.field_width,sp,om)
        return torch.where(active[:,None,None], obs, self._last_obs)
