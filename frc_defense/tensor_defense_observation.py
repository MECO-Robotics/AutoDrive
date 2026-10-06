"""Perception and observation behavior for the tensorized defense simulator."""
from __future__ import annotations

import math

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None

from .observation import normalize_tensor_observation_batch_in_place


class TensorDefenseObservationMixin:
    def perception_state(self, focal):
        """Per-robot controller view; contains no simulator piece-owner data."""
        from .tensor_sim import PerceptionState
        other=1-int(focal)
        opponent_pose,opponent_velocity=self._observed_robot(focal,other)
        return PerceptionState(
            own_pose=self.sim.pose[:,focal], own_velocity=self.sim.velocity[:,focal],
            own_possession=self._own_possession(focal), capacity=self.fuel_capacity,
            opponent_pose=opponent_pose, opponent_velocity=opponent_velocity,
            opponent_valid=self._opponent_track_valid[:,focal],
            opponent_track_age=self._opponent_track_age[:,focal],
            fuel_position=self._track_pos[:,focal], fuel_velocity=self._track_vel[:,focal],
            fuel_mask=self._track_mask[:,focal], fuel_track_age=self._track_age[:,focal],
            hub_active=self.hub_active, hub_centers=self.hub_centers,
            match_elapsed=self.match_elapsed)

    def _perceived_fuel(self, focal):
        """Return tracked FUEL position, velocity, and validity without truth IDs/owners."""
        return (self._track_pos[:,focal],self._track_vel[:,focal],self._track_mask[:,focal])

    def _observed_robot(self, focal, robot):
        if focal==robot:
            return self.sim.pose[:,robot],self.sim.velocity[:,robot]
        valid=self._opponent_track_valid[:,focal]
        return (torch.where(valid[:,None],self._opponent_track_pose[:,focal],
                            torch.zeros_like(self._opponent_track_pose[:,focal])),
                torch.where(valid[:,None],self._opponent_track_velocity[:,focal],
                            torch.zeros_like(self._opponent_track_velocity[:,focal])))

    def _opponent_track_features(self, focal):
        valid=self._opponent_track_valid[:,focal].to(self.sim.pose.dtype)
        age=(self._opponent_track_age[:,focal]/max(self.perception_track_timeout,1e-6)).clamp(0.,1.)
        return torch.stack((valid,age),dim=-1)

    def _random_active(self, active_mask, tail_shape, *, normal=False, high=None,
                       active_count=None):
        count=int(active_mask.sum().item()) if active_count is None else int(active_count)
        dtype=torch.long if high is not None else self.sim.pose.dtype
        if count:
            shape=(count,*tail_shape)
            if normal:
                values=torch.randn(shape,device=self.device,generator=self.generator)
            elif high is not None:
                values=torch.randint(high,shape,device=self.device,generator=self.generator)
            else:
                values=torch.rand(shape,device=self.device,generator=self.generator)
            if count == self.n:
                return values
            out=torch.zeros((self.n,*tail_shape),device=self.device,dtype=dtype)
            out[active_mask]=values
            return out
        return torch.zeros((self.n,*tail_shape),device=self.device,dtype=dtype)

    def _update_perception(self, reset_mask=None, active_mask=None, active_count=None):
        """Advance simple per-robot FUEL tracking from simulated sensor detections."""
        from . import tensor_sim as _tensor_sim
        if (reset_mask is None and _tensor_sim._perception_commit_proto is not None and
                _tensor_sim._perception_commit_proto.fused_commit_available(self.device)):
            _tensor_sim._perception_commit_proto.update_perception(
                self, active_mask=active_mask, active_count=active_count)
            return
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                     else torch.as_tensor(active_mask,device=self.device,dtype=torch.bool))
        dt=self.dt
        if reset_mask is None:
            self._track_age=torch.where(active_mask[:,None,None]&self._track_mask,
                self._track_age+dt,self._track_age)
            self._opponent_track_age=torch.where(active_mask[:,None]&self._opponent_track_valid,
                self._opponent_track_age+dt,self._opponent_track_age)
        else:
            reset_mask=reset_mask.to(device=self.device,dtype=torch.bool)
            self._track_mask[reset_mask]=False
            self._track_age[reset_mask]=float("inf")
            self._opponent_track_valid[reset_mask]=False
            self._opponent_track_age[reset_mask]=float("inf")
        for robot in range(2):
            pose=self.sim.pose[:,robot]
            delta=self.piece_pos-pose[:,None,:2]
            occ=self.field_feature_obstacles
            other=1-robot
            other_pos=self.sim.pose[:,other,:2]
            other_radius=.5*torch.sqrt(self.sim.length[:,other].square()+self.sim.width[:,other].square())
            if _tensor_sim.FUSED_PERCEPTION_HIP_ENABLED:
                visible=_tensor_sim.visibility_mask(
                    pose.contiguous(), self.piece_pos.contiguous(),
                    self.piece_active.contiguous(), self.piece_owner.contiguous(),
                    active_mask.contiguous(), self.perception_range, self.perception_fov,
                    occ.contiguous(), other_pos.contiguous(), other_radius.contiguous())
            else:
                # Preserve the original Euclidean boundary predicate exactly.
                distance=delta.norm(dim=-1)
                visible=(self.piece_active & (self.piece_owner < 0) &
                         (distance <= self.perception_range))
                # Retain the original wrapped-angle path for narrow FOVs.
                if self.perception_fov < 360.:
                    bearing=torch.atan2(delta[...,1],delta[...,0])
                    angle=torch.atan2(torch.sin(bearing-pose[:,None,2]),
                                      torch.cos(bearing-pose[:,None,2])).abs()
                    visible &= angle <= math.radians(self.perception_fov)*.5
                # Circle/radius occlusion approximates the field structures.
                if occ.numel():
                    eligible=(visible & active_mask[:,None]).contiguous()
                    blocked=_tensor_sim.piece_occlusion_mask(pose[:,:2].contiguous(),delta,occ,eligible)
                    visible &= ~blocked
                rel=other_pos[:,None,:]-pose[:,None,:2]
                denom=delta.square().sum(-1).clamp_min(1e-8)
                t=(rel*delta).sum(-1)/denom
                closest=pose[:,None,:2]+t.clamp(0.,1.)[...,None]*delta
                occluded=((closest-other_pos[:,None,:]).norm(dim=-1)<=other_radius[:,None]) & (t>.02) & (t<.98)
                visible &= ~occluded
            if self.perception_dropout:
                detected=self._random_active(active_mask,visible.shape[1:],active_count=active_count)>=self.perception_dropout
                visible &= detected
            noise=self._random_active(active_mask,self.piece_pos.shape[1:],normal=True,
                                      active_count=active_count)*self.perception_position_noise
            measured=self.piece_pos+noise
            updated_velocity=self.piece_vel.clone()
            if self.perception_velocity_noise:
                updated_velocity += self._random_active(active_mask,self.piece_vel.shape[1:],normal=True,
                    active_count=active_count)*self.perception_velocity_noise
            self._track_pos[:,robot]=torch.where((visible&active_mask[:,None])[...,None],measured,self._track_pos[:,robot])
            self._track_vel[:,robot]=torch.where((visible&active_mask[:,None])[...,None],updated_velocity,self._track_vel[:,robot])
            self._track_age[:,robot]=torch.where(visible&active_mask[:,None],torch.zeros_like(self._track_age[:,robot]),self._track_age[:,robot])
            self._track_mask[:,robot] |= visible&active_mask[:,None]
            observed_pose=self.sim.pose[:,1-robot].clone()
            observed_velocity=self.sim.velocity[:,1-robot].clone()
            robot_delta=observed_pose[:,:2]-self.sim.pose[:,robot,:2]
            robot_bearing=torch.atan2(robot_delta[:,1],robot_delta[:,0])
            bearing_error=torch.atan2(torch.sin(robot_bearing-self.sim.pose[:,robot,2]),
                torch.cos(robot_bearing-self.sim.pose[:,robot,2])).abs()
            detected=(self._random_active(active_mask,(),active_count=active_count)>=self.perception_dropout)
            detected &= (robot_delta.norm(dim=-1)<=self.perception_range)
            detected &= bearing_error<=math.radians(self.perception_fov)*.5
            if occ.numel():
                denom=robot_delta.square().sum(-1).clamp_min(1e-8)
                ray=occ[None,:,:2]-self.sim.pose[:,robot,None,:2]
                t=(ray*robot_delta[:,None,:]).sum(-1)/denom[:,None]
                closest=self.sim.pose[:,robot,None,:2]+t.clamp(0.,1.)[...,None]*robot_delta[:,None,:]
                obstacle_radius=occ[None,:,2]+.03
                blocked=((obstacle_radius>=0) &
                         ((closest-occ[None,:,:2]).square().sum(-1)<=obstacle_radius.square()))
                blocked &= (t>.02)&(t<.98)
                detected &= ~blocked.any(-1)
            position_noise=self._random_active(active_mask,(2,),normal=True,
                                               active_count=active_count)*self.perception_position_noise
            heading_noise=self._random_active(active_mask,(),normal=True,active_count=active_count)*min(
                .05,self.perception_position_noise)
            velocity_noise=self._random_active(active_mask,(3,),normal=True,
                                               active_count=active_count)*self.perception_velocity_noise
            observed_pose[:,:2]+=position_noise
            observed_pose[:,2]+=heading_noise
            observed_velocity+=velocity_noise
            detected &= active_mask
            self._opponent_track_pose[:,robot]=torch.where(detected[:,None],observed_pose,
                self._opponent_track_pose[:,robot])
            self._opponent_track_velocity[:,robot]=torch.where(detected[:,None],observed_velocity,
                self._opponent_track_velocity[:,robot])
            self._opponent_track_age[:,robot]=torch.where(detected,torch.zeros_like(self._opponent_track_age[:,robot]),
                self._opponent_track_age[:,robot])
            self._opponent_track_valid[:,robot] |= detected
        expired=self._track_age>self.perception_track_timeout
        self._track_mask &= ~(expired&active_mask[:,None,None])
        self._opponent_track_valid &= ~((self._opponent_track_age>self.perception_track_timeout)&active_mask[:,None])

    def _fuel_candidates(self, focal):
        from . import tensor_sim as _tensor_sim
        points,_,free=self._perceived_fuel(focal)
        if (not self.squared_fuel_candidate_distance and
                _tensor_sim._fuel_candidate_rank_hip is not None and
                _tensor_sim._fuel_candidate_rank_hip.integrated_available(self.device)):
            # The HIP kernel fuses Euclidean distance and free-mask handling;
            # ATen topk remains responsible for backend-specific tie ordering.
            return _tensor_sim._fuel_candidate_rank_hip.candidates(
                points, self.sim.pose[:,focal,:2], free)
        delta=points-self.sim.pose[:,focal,None,:2]
        if self.squared_fuel_candidate_distance:
            distance_sq=delta.square().sum(-1).masked_fill(~free,float("inf"))
            nearest_sq,indices=distance_sq.topk(4,dim=-1,largest=False)
            nearest=nearest_sq.clamp_min(0.).sqrt()
        else:
            distance=delta.norm(dim=-1).masked_fill(~free,float("inf"))
            nearest,indices=distance.topk(4,dim=-1,largest=False)
        return indices,torch.isfinite(nearest),nearest

    def _strategic_candidate_features(self, focal, *, _candidate_data=None,
                                      _own_count=None):
        other=1-focal
        indices,valid,own_distance=(self._fuel_candidates(focal) if _candidate_data is None
                                    else _candidate_data)
        rows=torch.arange(self.n,device=self.device)[:,None]
        points,_,_=self._perceived_fuel(focal)
        points=points[rows,indices]
        own_eta=own_distance/self.sim.speed[:,focal,None].clamp_min(.1)
        opponent_pose,_=self._observed_robot(focal,other)
        opponent_distance=(points-opponent_pose[:,None,:2]).norm(dim=-1)
        opponent_eta=opponent_distance/self.sim.speed[:,other,None].clamp_min(.1)
        hub=self.hub_centers[focal]
        robot_radius=.5*torch.sqrt(self.sim.length[:,focal].square()+self.sim.width[:,focal].square())
        score_distance=(points-hub).norm(dim=-1)-(.595+robot_radius[:,None])
        to_score=score_distance.clamp_min(0.)/self.sim.speed[:,focal,None].clamp_min(.1)
        risk=torch.sigmoid((own_eta-opponent_eta)*2.)
        zone=(points[...,0]/self.sim.field_length*6.).floor().clamp(0,5).to(points.dtype)/6.
        own_count=(self._own_possession(focal) if _own_count is None else
                   _own_count).to(points.dtype)
        own_pos=(own_count/max(self.fuel_capacity,1))[:,None].expand(-1,4)
        capacity_left=((self.fuel_capacity-own_count).clamp_min(0)/max(self.fuel_capacity,1))[:,None].expand(-1,4)
        features=torch.stack(((points[...,0]-self.sim.pose[:,focal,None,0])/self.sim.field_length,
            (points[...,1]-self.sim.pose[:,focal,None,1])/self.sim.field_width,
            own_eta/20.,opponent_eta/20.,to_score/20.,risk,zone,own_pos,capacity_left,
            valid.to(points.dtype)),dim=-1)
        features=torch.where(valid[...,None],features,torch.zeros_like(features))
        return features.reshape(self.n,40)

    def strategic_action_mask(self, focal=0, *, _candidate_data=None):
        """Valid categorical choices for the current game state and robot role."""
        _,valid,_=(self._fuel_candidates(focal) if _candidate_data is None
                   else _candidate_data)
        offense=(self.task=="counter_defense" and focal==0) or (self.task=="defense" and focal==1)
        mask=torch.zeros((self.n,8),device=self.device,dtype=torch.bool)
        if offense:
            mask[:,:4]=valid
            carrying=(self.piece_active&(self.piece_owner==focal)).any(-1)
            mask[:,4]=carrying
            # Passing remains unavailable in the 1v1 field model until a teammate
            # state is supplied; the categorical slot is reserved for that action.
            mask[:,6]=True
            mask[:,7]=True
        else:
            mask[:,0]=True  # intercept the attacker's observed route
            mask[:,1:5]=valid  # deny one of the visible candidate FUEL pieces
            mask[:,5]=True  # block the likely scoring lane
            mask[:,6]=True  # shadow the attacker
        return mask

    def _strategic_observation(self, focal, *, return_candidate_data=False,
                               active_mask=None, cache_active_mask=None,
                               row_indices=None, candidate_data=None,
                               action_mask=None, own_count=None):
        from . import tensor_sim as _tensor_sim
        if (_tensor_sim._full_strategic_observation_available is not None and
                _tensor_sim._full_strategic_observation_available(self.device)):
            self._full_observation_hip_available=True
            raw, candidate_data, action_mask = _tensor_sim._full_strategic_observation(
                self, focal, active_mask=active_mask, return_candidate_data=True,
                row_indices=row_indices, candidates=candidate_data,
                action_mask=action_mask, own_count=own_count)
            self._cache_strategic_own_candidates(
                candidate_data, action_mask, focal=focal,
                active_mask=(active_mask if cache_active_mask is None
                             else cache_active_mask))
            if return_candidate_data:
                return raw, candidate_data, action_mask
            return raw
        fuse_candidate_local=(_tensor_sim._fused_strategic_observation_available is not None and
                              _tensor_sim._fused_strategic_observation_available(self.device))
        p,v=self.sim.pose,self.sim.velocity
        other=1-focal
        candidate_data=self._fuel_candidates(focal)
        own_count=self._own_possession(focal)
        action_mask=self.strategic_action_mask(focal,_candidate_data=candidate_data)
        other_pose,other_velocity=self._observed_robot(focal,other)
        base=torch.cat((p[:,focal],v[:,focal],other_pose,other_velocity,
            self._opponent_track_features(focal),
            torch.full((self.n,1),.595,device=self.device),
            torch.full((self.n,1),self.sim.field_length,device=self.device),
            torch.full((self.n,1),self.sim.field_width,device=self.device),
            self.sim.length[:,[focal,other]],self.sim.width[:,[focal,other]],
            self.sim.accel[:,[focal,other]]/10.,self._obstacle_features(robot_index=focal)),dim=-1)
        route=torch.zeros((self.n,3),device=self.device)
        planner=(self._adstar_tactical_planner if focal==0 else
                 self._adstar_planners if self.task=="defense" else self._adstar_defender_planners)
        if planner is not None:
            has_route=planner.last_lengths>1
            first=planner.last_path[:,1,:]-p[:,focal,:2]
            segment=(planner.last_path[:,1:,:]-planner.last_path[:,:-1,:]).norm(dim=-1)
            valid_path=(torch.arange(segment.shape[1],device=self.device)[None,:]
                        <(planner.last_lengths-1).clamp_min(0)[:,None])
            cost=(segment*valid_path).sum(-1)
            route=torch.stack((first[:,0]/self.sim.field_length,first[:,1]/self.sim.field_width,
                cost/math.hypot(self.sim.field_length,self.sim.field_width)),dim=-1)
            route=torch.where(has_route[:,None],route,torch.zeros_like(route))
        local=[]
        for robot in (focal,other):
            if robot!=focal:
                # Do not fuse the opponent's private sensor tracks into this
                # robot's observation; no teammate communications are modeled.
                local.append(torch.zeros((self.n,20),device=self.device,dtype=base.dtype))
                continue
            if fuse_candidate_local:
                # The fused packer computes these 4 × 5 local-track features.
                local.append(torch.zeros((self.n,20),device=self.device,dtype=base.dtype))
                continue
            tracked_points,tracked_velocities,_=self._perceived_fuel(robot)
            indices,valid,nearest=candidate_data
            rows=torch.arange(self.n,device=self.device)[:,None]
            points=tracked_points[rows,indices];velocities=tracked_velocities[rows,indices]
            relative=points-self.sim.pose[:,robot,None,:2]
            normalized_velocity=velocities/self.sim.speed[:,robot,None,None].clamp_min(.1)
            slot=torch.cat((relative[...,0:1]/self.sim.field_length,
                relative[...,1:2]/self.sim.field_width,normalized_velocity,
                valid[...,None].to(base.dtype)),dim=-1)
            local.append(torch.where(valid[...,None],slot,torch.zeros_like(slot)).reshape(self.n,-1))
        possession=torch.stack((own_count,torch.zeros_like(own_count)),dim=-1).to(base.dtype)
        possession=possession/max(self.fuel_capacity,1)
        match=torch.cat((self.match_elapsed[:,None]/160.,self.hub_active.to(base.dtype),
            self.fuel_score_count[:,[focal,other]].to(base.dtype)/max(self.fuel_count,1),
            (self.hub_centers[None,:,:]-p[:,focal,None,:2]).reshape(self.n,4)/
                torch.tensor((self.sim.field_length,self.sim.field_width,self.sim.field_length,
                              self.sim.field_width),device=self.device)),dim=-1)
        action_values=action_mask.to(base.dtype)
        if fuse_candidate_local:
            raw=_tensor_sim._pack_fused_strategic_observation(self,focal,base=base,route=route,
                local1=local[1],possession=possession,match=match,
                action_values=action_values,other_pose=other_pose,own_count=own_count,
                candidate_data=candidate_data,normalize=self.normalize_observations)
            if raw is None:
                raise RuntimeError("fused strategic observation became unavailable mid-call")
        else:
            raw=torch.cat((base,route,*local,possession,match,
                self._strategic_candidate_features(focal,_candidate_data=candidate_data,
                                                   _own_count=own_count),
                action_values),dim=-1)
            if self.normalize_observations:
                normalize_tensor_observation_batch_in_place(raw,self.sim.field_length,self.sim.field_width,
                    self.sim.speed[:,[focal,other]],self.sim.omega_limit[:,[focal,other]])
        self._cache_strategic_own_candidates(
            candidate_data, action_mask, focal=focal,
            active_mask=(active_mask if cache_active_mask is None
                         else cache_active_mask))
        if return_candidate_data:
            return raw,candidate_data,action_mask
        return raw

    def _cache_strategic_own_candidates(self, candidate_data, action_mask, *,
                                        focal, active_mask=None):
        """Preserve per-world candidate cache rows for either observation path."""
        if focal != 0 or not self.reuse_strategic_own_candidates:
            return
        # The next normal physics step starts from this exact observation state,
        # so it can consume this ranking instead of repeating topk. Keep
        # inactive rows intact under active-mask stepping.
        if active_mask is None:
            self._pending_strategic_own_candidates = (candidate_data, action_mask)
            return
        selected = torch.as_tensor(active_mask, device=self.device,
                                   dtype=torch.bool).reshape(self.n)
        pending = self._pending_strategic_own_candidates
        if pending is None:
            pending = (tuple(torch.zeros_like(value) for value in candidate_data),
                       torch.zeros_like(action_mask))
        previous_data, previous_mask = pending
        merged_data = tuple(torch.where(selected[:, None], current, previous)
                            for current, previous in zip(candidate_data, previous_data))
        merged_mask = torch.where(selected[:, None], action_mask, previous_mask)
        self._pending_strategic_own_candidates = (merged_data, merged_mask)

    def _raw_obs(self, active_mask=None, cache_active_mask=None, row_indices=None,
                 candidate_data=None, action_mask=None, own_count=None):
        if self.action_mode=="strategic":
            return self._strategic_observation(
                0, active_mask=active_mask,
                cache_active_mask=cache_active_mask, row_indices=row_indices,
                candidate_data=candidate_data, action_mask=action_mask,
                own_count=own_count)
        p,v=self.sim.pose,self.sim.velocity
        opponent_pose,opponent_velocity=self._observed_robot(0,1)
        # Legacy continuous policies keep their feature width, but receive no
        # hidden episode goal; strategy must be inferred from observable state.
        rel=self._opponent_track_features(0)
        # Tactical defense needs the defended region to choose an interception
        # point. Both tasks expose the goal and radius at the shared indices.
        goal_radius=torch.full((self.n,1),.595,device=self.device)
        raw=torch.cat((p[:,0],v[:,0],opponent_pose,opponent_velocity,rel,goal_radius,
                       torch.full((self.n,1),self.sim.field_length,device=self.device),torch.full((self.n,1),self.sim.field_width,device=self.device),
                       self.sim.length,self.sim.width,self.sim.accel/10.,self._obstacle_features()),-1)
        if self.action_mode in ("tactical","strategic"):
            # Cached AD* route to the previous tactical waypoint. The final
            # three features are next-route-point delta (field-normalized) and
            # remaining path length (field-diagonal-normalized); zeros indicate
            # that a route is not available yet.
            route_features=torch.zeros((self.n,3),device=self.device)
            if self._adstar_tactical_planner is not None:
                planner=self._adstar_tactical_planner
                has_route=planner.last_lengths>1
                first=planner.last_path[:,1,:]-p[:,0,:2]
                segment=(planner.last_path[:,1:,:]-planner.last_path[:,:-1,:]).norm(dim=-1)
                valid=(torch.arange(segment.shape[1],device=self.device)[None,:]
                       <(planner.last_lengths-1).clamp_min(0)[:,None])
                path_cost=(segment*valid).sum(-1)
                route_features=torch.stack((first[:,0]/self.sim.field_length,
                    first[:,1]/self.sim.field_width,
                    path_cost/math.hypot(self.sim.field_length,self.sim.field_width)),dim=-1)
                route_features=torch.where(has_route[:,None],route_features,torch.zeros_like(route_features))
            raw=torch.cat((raw,route_features),dim=-1)
        # Feature scales are fixed and shared with external policy adapters;
        # no rollout-fitted running statistics can leak into evaluation.
        if self.normalize_observations:
            normalize_tensor_observation_batch_in_place(
                raw,self.sim.field_length,self.sim.field_width,self.sim.speed,self.sim.omega_limit)
        return raw

    def _obs(self, initialize_mask=None, active_mask=None, active_count=None,
             capture_output=True, capture_raw_mask=None, capture_raw_rows=None):
        # PPO supplies this scalar for its fixed, synchronized full-active
        # batch. Do not inspect the device mask here: active_count is already
        # the trusted shape/RNG count used by _random_active.
        full_active = (active_mask is None or
                       (active_count is not None and int(active_count) == self.n))
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                     else torch.as_tensor(active_mask,device=self.device,dtype=torch.bool))
        raw_capture_mask=(active_mask if capture_raw_mask is None else
                          (active_mask & torch.as_tensor(
                              capture_raw_mask, device=self.device,
                              dtype=torch.bool).reshape(self.n)))
        candidate_data=action_mask=own_count=None
        if capture_raw_rows is not None:
            # Candidate ranking and masks still refresh for every active world:
            # the next physics tick consumes them in the strategic controller.
            candidate_data=self._fuel_candidates(0)
            own_count=self._own_possession(0)
            action_mask=self.strategic_action_mask(
                0, _candidate_data=candidate_data)
            self._cache_strategic_own_candidates(
                candidate_data, action_mask, focal=0, active_mask=active_mask)
            raw=None
            if capture_raw_rows.numel():
                raw=self._raw_obs(
                    active_mask=raw_capture_mask, cache_active_mask=active_mask,
                    row_indices=capture_raw_rows, candidate_data=candidate_data,
                    action_mask=action_mask, own_count=own_count)
        else:
            raw=self._raw_obs(active_mask=raw_capture_mask,
                              cache_active_mask=active_mask)
        history_length=self.observation_history.shape[1]
        if initialize_mask is None:
            next_index=(self._observation_history_index+1).remainder(history_length)
            if capture_raw_rows is not None:
                self._observation_history_index.copy_(
                    torch.where(active_mask, next_index,
                                self._observation_history_index))
                if capture_raw_rows.numel():
                    self.observation_history[
                        self._observation_world_index[capture_raw_rows],
                        next_index[capture_raw_rows]]=raw
            elif full_active and capture_raw_mask is None:
                # PPO supplies a trusted full-batch active count for ordinary
                # strategic ticks. Avoid all-true mask selects while retaining
                # the same ring slot, values, storage, and subsequent RNG calls.
                self.observation_history[
                    self._observation_world_index,next_index]=raw
                self._observation_history_index.copy_(next_index)
            else:
                write_index=torch.where(active_mask,next_index,self._observation_history_index)
                current=self.observation_history[self._observation_world_index,write_index]
                updated=torch.where(raw_capture_mask[:,None],raw,current)
                self.observation_history[self._observation_world_index,write_index]=updated
                self._observation_history_index.copy_(write_index)
        else:
            initialize_mask=torch.as_tensor(initialize_mask,device=self.device,dtype=torch.bool)
            fresh=raw[:,None,:].expand(-1,self.observation_history.shape[1],-1)
            self.observation_history.copy_(torch.where(
                initialize_mask[:,None,None],fresh,self.observation_history))
            self._observation_history_index.copy_(torch.where(
                initialize_mask,torch.zeros_like(self._observation_history_index),
                self._observation_history_index))
        if not capture_output:
            # PPO only consumes the observation at the end of a strategic
            # held-action interval (and at an episode boundary). Keep writing
            # the complete raw history above and advance the same observation
            # RNG streams, but skip the delayed gather and output transforms
            # whose intermediate results the trainer discards.
            if self.randomize and self.observation_noise>0:
                self._random_active(active_mask,(18,),normal=True,
                                    active_count=active_count)
            if self.randomize and self.observation_dropout>0:
                self._random_active(active_mask,(18,),active_count=active_count)
            return self._last_observation
        read_index=(self._observation_history_index-self.observation_delay).remainder(history_length)
        obs=self.observation_history.gather(
            1,read_index[:,None,None].expand(-1,1,self.obs_dim)).squeeze(1)
        if self.randomize and self.observation_noise>0:
            # gather() above allocated a fresh tensor, separate from history.
            # Mutate only its physical-state slice to avoid rebuilding all 137
            # features. Keep noise before dropout and preserve both RNG calls.
            obs[:,:18].add_(self._random_active(
                active_mask,(18,),normal=True,active_count=active_count)*self.observation_noise)
        if self.randomize and self.observation_dropout>0:
            keep=self._random_active(active_mask,(18,),active_count=active_count)>=self.observation_dropout
            obs[:,:18].mul_(keep)
        if full_active:
            return obs
        return torch.where(active_mask[:,None],obs,self._last_observation)

    def _obstacle_features(self,robot_index=0,row_indices=None):
        pose=(self.sim.pose[:,robot_index] if row_indices is None else
              self.sim.pose.index_select(0,row_indices)[:,robot_index])
        row_count=pose.shape[0]
        out=torch.zeros((row_count,12),device=self.device)
        obstacles=(torch.cat((self.sim.obstacles,self.field_feature_obstacles),0)
                   if self.field_feature_obstacles.shape[0] else self.sim.obstacles)
        if obstacles.shape[0]:
            dist=(obstacles[None,:,:2]-pose[:,None,:2]).square().sum(-1)
            chosen=obstacles[dist.topk(min(4,obstacles.shape[0]),dim=-1,largest=False).indices]
            out[:,:chosen.shape[1]*3]=chosen.reshape(row_count,-1)
        return out
