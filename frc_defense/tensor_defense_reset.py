"""Episode setup and reset lifecycle for the tensor defense environment."""
from __future__ import annotations

import math
from .fuel_physics_runtime import reset_fuel_physics

from .field import ALLIANCE_ZONE_DEPTH

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None


class TensorDefenseResetMixin:
    """Sample episode layouts and reset full or partial environment batches."""

    def _assert_device_condition(self, condition, message):
        """Queue a fail-closed tensor assertion without reading its value on CPU."""
        if (not isinstance(condition, torch.Tensor) or condition.device != self.device or
                condition.numel() != 1):
            raise ValueError("reset validation must be a device scalar tensor")
        assert_async=getattr(torch,"_assert_async",None)
        if assert_async is None:
            raise RuntimeError("device-side reset validation is unavailable")
        assert_async(condition,message)

    def _sample_episode_endpoints(self, goal_override=None, start_zone=None, goal_zone=None, mask=None):
        """Sample endpoints in requested zones, or random distinct field zones."""
        if (mask is not None and self.device.type != "cpu" and
                (not isinstance(mask, torch.Tensor) or mask.device != self.device)):
            raise ValueError("GPU endpoint mask must already reside on the environment device")
        mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if mask is None
              else torch.as_tensor(mask,device=self.device,dtype=torch.bool).reshape(self.n))
        length, width = self.sim.field_length, self.sim.field_width
        margin = .85
        depth = ALLIANCE_ZONE_DEPTH if self.field_boxes else length / 3
        depth = min(depth, (length - 2 * margin) / 3)
        edges = torch.tensor((0., depth, length - depth, length), device=self.device)
        zone_ids={"red":0,"center":1,"middle":1,"blue":2}
        def zone_tensor(value):
            if value is None or (isinstance(value,str) and value.lower()=="random"):
                return None
            if isinstance(value,str):
                if value.lower() not in zone_ids:
                    raise ValueError("zone must be red, center, blue, or random")
                return torch.full((self.n,),zone_ids[value.lower()],device=self.device,dtype=torch.long)
            zones=torch.as_tensor(value,device=self.device,dtype=torch.long).reshape(-1)
            if zones.numel()==1: zones=zones.expand(self.n)
            if zones.numel()!=self.n:
                raise ValueError("zone ids must contain one value per environment in 0..2")
            self._assert_device_condition(((zones>=0)&(zones<=2)).all(),
                                          "zone ids must be in 0..2")
            return zones
        requested_start=zone_tensor(start_zone)
        requested_goal=zone_tensor(goal_zone)
        if isinstance(start_zone,str) and isinstance(goal_zone,str) and start_zone in zone_ids and start_zone==goal_zone:
            raise ValueError("start and goal zones must be different")
        random_start=self._random_active(mask,(),high=3).long()
        start_zone=(requested_start if requested_start is not None else random_start)
        if goal_override is None:
            if requested_goal is not None:
                goal_zone=requested_goal
                same=start_zone==goal_zone
                self._assert_device_condition((~same).all(),
                                              "start and goal zones must be different")
            elif requested_start is not None:
                goal_zone=(start_zone+self._random_active(mask,(),high=2).long()+1)%3
            else:
                goal_zone = (start_zone + self._random_active(mask,(),high=2).long()+1) % 3
        else:
            if requested_goal is not None:
                raise ValueError("goal_zone cannot be combined with an explicit goal coordinate")
            goal = torch.as_tensor(goal_override, device=self.device, dtype=self.sim.pose.dtype).expand(self.n, 2)
            goal_zone = torch.where(goal[:, 0] < depth, 0,
                                    torch.where(goal[:, 0] >= length - depth, 2, 1)).long()
            if requested_start is None:
                choices = self._random_active(mask,(),high=2).long()
                start_zone = torch.where(goal_zone == 0, choices + 1,
                             torch.where(goal_zone == 1, choices * 2, choices))
            elif isinstance(start_zone,torch.Tensor) and isinstance(goal_zone,torch.Tensor):
                self._assert_device_condition((start_zone!=goal_zone).all(),
                                              "start and goal zones must be different")

        def sample_points(zones, avoid=None):
            low, high = edges[zones] + margin, edges[zones + 1] - margin
            points = torch.zeros((self.n, 2), device=self.device, dtype=self.sim.pose.dtype)
            unresolved = mask.clone()
            for _ in range(12):
                candidate = torch.stack((low + self._random_active(mask,()) * (high - low),
                    margin + self._random_active(mask,()) * (width - 2 * margin)), -1)
                valid = torch.ones((self.n,), device=self.device, dtype=torch.bool)
                for box in self.sim.field_colliders:
                    valid &= (((candidate[:, 0] - box[0]).abs() > box[2] + .8) |
                              ((candidate[:, 1] - box[1]).abs() > box[3] + .8))
                for obstacle in self.sim.obstacles:
                    valid &= ((candidate - obstacle[:2]).square().sum(-1) > (obstacle[2] + .8) ** 2)
                if avoid is not None:
                    valid &= (candidate - avoid).norm(dim=-1) >= 3.
                accept = unresolved & valid
                points = torch.where(accept[:, None], candidate, points)
                unresolved &= ~accept
            # Rejection sampling can miss in dense layouts. Fall back to a
            # random choice from a legal 0.2 m lattice; never return the last
            # unchecked sample. If the 3 m separation preference leaves no
            # legal point, relax only that preference, not collider clearance.
            xs=torch.arange(margin,self.sim.field_length-margin+.001,.2,
                                device=self.device,dtype=self.sim.pose.dtype)
            ys=torch.arange(margin,width-margin+.001,.2,
                                device=self.device,dtype=self.sim.pose.dtype)
            gx,gy=torch.meshgrid(xs,ys,indexing="ij")
            lattice=torch.stack((gx.flatten(),gy.flatten()),-1)
            zone_of=torch.where(lattice[:,0]<depth,0,
                         torch.where(lattice[:,0]>=length-depth,2,1))
            clear=torch.ones((lattice.shape[0],),device=self.device,dtype=torch.bool)
            for box in self.sim.field_colliders:
                clear &= ((lattice[:,0]-box[0]).abs()>box[2]+.8)|((lattice[:,1]-box[1]).abs()>box[3]+.8)
            for obstacle in self.sim.obstacles:
                clear &= ((lattice-obstacle[:2]).square().sum(-1)>(obstacle[2]+.8)**2)
            allowed=clear[None,:]&(zone_of[None,:]==zones[:,None])
            if avoid is not None:
                separated=allowed&((lattice[None,:,:]-avoid[:,None,:]).square().sum(-1)>=9.)
                has_separated=separated.any(-1,keepdim=True)
                allowed=torch.where(has_separated,separated,allowed)
            has_legal_point=allowed.any(-1)
            self._assert_device_condition((has_legal_point | ~unresolved).all(),
                                          "no legal endpoint exists in a selected field zone")
            random_rank=self._random_active(mask,(lattice.shape[0],))
            fallback_index=random_rank.masked_fill(~allowed,-1.).argmax(-1)
            fallback=lattice[fallback_index]
            points=torch.where(unresolved[:,None],fallback,points)
            return points

        start = sample_points(start_zone)
        goal = (torch.as_tensor(goal_override, device=self.device, dtype=self.sim.pose.dtype)
                .expand(self.n, 2).clone() if goal_override is not None
                else sample_points(goal_zone, avoid=start))
        return start, goal

    def _place_defenders_near_clear_adstar_paths(self, mask=None):
        """Place each reset defender near a randomized point on the clear AD* route."""
        if self._adstar_planners is None or not self.adstar_spawn_hint:
            return
        starts=self.sim.pose[:,1,:2]
        if (mask is not None and self.device.type != "cpu" and
                (not isinstance(mask,torch.Tensor) or mask.device != self.device)):
            raise ValueError("GPU defender placement mask must already reside on the environment device")
        selected=(torch.ones(self.n,device=self.device,dtype=torch.bool) if mask is None
                  else torch.as_tensor(mask,device=self.device,dtype=torch.bool))
        planner=self._adstar_planners
        planner.plan(starts,self._attacker_objective(),self.sim.pose[:,1,2],self.sim.length[:,1],
                     self.sim.width[:,1],active_mask=selected,
                     static_active_mask=True)
        # Preserve defender_spawn's eligible-rank, route fallback and jitter
        # semantics while drawing from each selected world's reset stream.
        distance=(planner.last_path-starts[:,None,:]).norm(dim=-1)
        valid=(torch.arange(planner.max_points,device=self.device)[None,:] <
               planner.last_lengths[:,None])
        eligible=valid&(distance>=1.25)&(distance<=2.75)
        weights=self._random_active(selected,(planner.max_points,))
        weights=weights.masked_fill(~(eligible&selected[:,None]),-1.)
        index=weights.argmax(-1)
        fallback=(valid&(distance>=1.1)).float().argmax(-1)
        index=torch.where((eligible&selected[:,None]).any(-1),index,fallback)
        row=torch.arange(self.n,device=self.device)
        point=planner.last_path[row,index]
        jitter=self._random_active(selected,(2,))-.5
        candidate=torch.where(selected[:,None],point+jitter*.18,starts)
        self.sim.pose[:,0,:2]=torch.where(selected[:,None],candidate,self.sim.pose[:,0,:2])

    def _move_adstar_defenders_to_clear_start(self, mask=None):
        """Keep an AD* guard from spawning inside its inflated obstacle map."""
        bumps=self.sim.bump_regions
        if self._adstar_defender_planners is None or not bumps.numel():
            return
        if (mask is not None and self.device.type != "cpu" and
                (not isinstance(mask,torch.Tensor) or mask.device != self.device)):
            raise ValueError("GPU defender placement mask must already reside on the environment device")
        selected=(torch.ones(self.n,device=self.device,dtype=torch.bool) if mask is None
                  else torch.as_tensor(mask,device=self.device,dtype=torch.bool))
        position=self.sim.pose[:,1,:2]
        radius=.5*torch.sqrt(self.sim.length[:,1].square()+self.sim.width[:,1].square())+.15
        dx=(position[:,None,0]-bumps[None,:,0]).abs()
        dy=(position[:,None,1]-bumps[None,:,1]).abs()
        inside=((dx<=bumps[None,:,2]+radius[:,None]) &
                (dy<=bumps[None,:,3]+radius[:,None])).any(-1)
        boxes=self.sim.field_colliders
        if boxes.numel():
            box_dx=(position[:,None,0]-boxes[None,:,0]).abs()
            box_dy=(position[:,None,1]-boxes[None,:,1]).abs()
            inside|=((box_dx<=boxes[None,:,2]+radius[:,None]) &
                     (box_dy<=boxes[None,:,3]+radius[:,None])).any(-1)
        obstacles=self.sim.obstacles
        if obstacles.numel():
            obstacle_distance=(position[:,None,:]-obstacles[None,:,:2]).norm(dim=-1)
            inside|=(obstacle_distance<=obstacles[None,:,2]+radius[:,None]).any(-1)
        attacker=self.sim.pose[:,0,:2]
        separation=(position-attacker).norm(dim=-1)
        needs_move=selected&(inside|(separation<1.4)|(separation>3.5))
        margin=radius.clamp_min(.9)
        lane=self._attacker_objective()-attacker
        lane_length=lane.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        preferred=attacker+lane/lane_length*torch.minimum(lane_length*.55,
            torch.full_like(lane_length,2.5))
        for _ in range(32):
            angle=self._random_active(needs_move,())*(2*math.pi)
            spawn_radius=.25+self._random_active(needs_move,())*1.35
            candidate=torch.stack((
                preferred[:,0]+angle.cos()*spawn_radius,
                preferred[:,1]+angle.sin()*spawn_radius),dim=-1)
            cdx=(candidate[:,None,0]-bumps[None,:,0]).abs()
            cdy=(candidate[:,None,1]-bumps[None,:,1]).abs()
            clear=~((cdx<=bumps[None,:,2]+radius[:,None]) &
                    (cdy<=bumps[None,:,3]+radius[:,None])).any(-1)
            if boxes.numel():
                box_dx=(candidate[:,None,0]-boxes[None,:,0]).abs()
                box_dy=(candidate[:,None,1]-boxes[None,:,1]).abs()
                hits=((box_dx<=boxes[None,:,2]+radius[:,None]) &
                      (box_dy<=boxes[None,:,3]+radius[:,None])).any(-1)
                clear&=~hits
            obstacles=self.sim.obstacles
            if obstacles.numel():
                obstacle_distance=(candidate[:,None,:]-obstacles[None,:,:2]).norm(dim=-1)
                clear&=~(obstacle_distance<=obstacles[None,:,2]+radius[:,None]).any(-1)
            attacker_clearance=.5*torch.sqrt(self.sim.length[:,0].square()+self.sim.width[:,0].square())+radius+.25
            candidate_separation=(candidate-attacker).norm(dim=-1)
            clear&=(candidate_separation>=attacker_clearance)
            clear&=(candidate[:,0]>=margin)&(candidate[:,0]<=self.sim.field_length-margin)
            clear&=(candidate[:,1]>=margin)&(candidate[:,1]<=self.sim.field_width-margin)
            place=needs_move&clear
            position.copy_(torch.where(place[:,None],candidate,position))
            needs_move&=~place
        self._assert_device_condition((~needs_move).all(),
                                      "could not place defender in a clear start position")

    def reset(self,seed=None,options=None):
        self._ensure_world_rng_state()
        if seed is not None:
            self.generator.manual_seed(int(seed))
            world_ids=torch.arange(self.n,device=self.device,dtype=torch.long)
            self._world_rng_seed.copy_((world_ids*0x1F123BB5+int(seed))&0x7fffffff)
        self._world_rng_counters.zero_()
        self.sim.reset(seed); self.steps.zero_(); self.last_contact.zero_()
        self._sample_static_opponents(torch.ones(self.n,device=self.device,dtype=torch.bool))
        self.last_static_contact.zero_()
        if self._adstar_planners is not None:
            self._adstar_contact_latched.zero_()
        pidx=0 if self.task=="counter_defense" else 1
        start, goal = self._sample_episode_endpoints(
            options.get("goal") if options and "goal" in options else None,
            options.get("start_zone") if options else None,
            options.get("goal_zone") if options else None)
        self.goal.copy_(goal)
        self.sim.pose[:, pidx, :2] = start
        if self.randomize:
            full_mask=torch.ones((self.n,),device=self.device,dtype=torch.bool)
            self.sim.velocity[:,:,:2]=(self._random_active(full_mask,(2,2))*2-1)*.25*self.sim.speed[:,:,None]
            self.sim.velocity[:,:,2]=(self._random_active(full_mask,(2,))*2-1)*.2*self.sim.omega_limit
            self.goal_radius=.3+self._random_active(full_mask,())*.5
        else:
            self.goal_radius.fill_(self.base_goal_radius)
        self.sim.velocity[:,1]=torch.where(self.static_opponent_mask[:,None],
            torch.zeros_like(self.sim.velocity[:,1]),self.sim.velocity[:,1])
        if options and "goal_radius" in options:
            self.goal_radius.copy_(torch.as_tensor(options["goal_radius"],device=self.device,dtype=self.goal.dtype).expand_as(self.goal_radius))
        self.match_elapsed.zero_(); self.match_remaining.fill_(160.)
        self.hub_active.fill_(True)
        self.hub_inactive_first=self._random_active(
            torch.ones(self.n,device=self.device,dtype=torch.bool),(),high=2).long()
        self.auto_fuel_scores.zero_()
        for value in (self.fuel_acquisition_count,self.fuel_score_count,self.fuel_denied_count,
                      self.fuel_abandoned_count,self.fuel_acquired_event,self.fuel_scored_event,
                      self.fuel_denied_event,self.fuel_abandoned_event,
                      self.fuel_passed_event):
            value.zero_()
        for value in (self.next_intake_time,self.next_score_time,self.next_score,
                      self.next_ferry):
            value.zero_()
        self.last_actions.fill_(7)
        self._intake_collecting.zero_()
        self._initialize_gamepieces()
        reset_fuel_physics(self)
        self._update_perception(torch.ones((self.n,),device=self.device,dtype=torch.bool))
        self._move_adstar_defenders_to_clear_start()
        self._place_defenders_near_clear_adstar_paths()
        self._last_hub_zone.zero_(); self.last_hub_zone.zero_()
        self._last_strategic_action.zero_()
        self._planner_tick.fill_(self.adstar_replan_interval-1)
        self._planner_tick_scalar=self.adstar_replan_interval-1
        self._planner_tick_aligned=True
        self._planner_full_batch_active=False
        self.action_history.zero_()
        self.control_delay.zero_()
        if self.randomize and self.max_control_latency_steps:
            self.control_delay.copy_(self._random_active(
                torch.ones(self.n,device=self.device,dtype=torch.bool),(),
                high=self.max_control_latency_steps+1).long())
        self.observation_delay.zero_()
        if self.randomize and self.max_observation_latency_steps:
            self.observation_delay.copy_(self._random_active(
                torch.ones(self.n,device=self.device,dtype=torch.bool),(),
                high=self.max_observation_latency_steps+1).long())
        attacker_index=0 if self.task=="counter_defense" else 1
        self.previous.copy_((self._attacker_objective()-self.sim.pose[:,attacker_index,:2]).norm(dim=-1))
        self.initial_distance.copy_(self.previous)
        self.path_length.zero_(); self.contact_steps.zero_(); self.contact_count.zero_()
        self.blocked_steps.zero_(); self.out_of_bounds_steps.zero_(); self.last_contact.zero_(); self.last_action.zero_()
        self.observation_history.zero_()
        self._last_observation=self._obs(torch.ones((self.n,),device=self.device,dtype=torch.bool))
        return self._last_observation,{"goal":self._attacker_objective(),
            "goal_radius":torch.full_like(self.goal_radius,.595) if self.action_mode=="strategic" else self.goal_radius,
            "field_layout":"2026_rebuilt" if self.field_boxes else "custom",
            "field_colliders":self.field_boxes}

    def reset_done(self,mask):
        """Reset selected environment slots; return the full fresh observation batch."""
        if (self.device.type != "cpu" and
                (not isinstance(mask, torch.Tensor) or mask.device != self.device)):
            raise ValueError("GPU reset mask must already reside on the environment device")
        mask=torch.as_tensor(mask,device=self.device,dtype=torch.bool).reshape(self.n)
        # A device reset mask cannot select a host-side scalar fast path.
        # Keep per-world cadence active after every reset call.
        self._planner_tick_aligned=False
        self._planner_full_batch_active=False
        self.sim.reset_done(mask)
        self._sample_static_opponents(mask)
        if self._adstar_planners is not None:
            self._adstar_contact_latched &= ~mask
        pidx=0 if self.task=="counter_defense" else 1
        start, goal = self._sample_episode_endpoints(mask=mask)
        self.sim.pose[:, pidx, :2] = torch.where(mask[:, None], start, self.sim.pose[:, pidx, :2])
        self.goal.copy_(torch.where(mask[:,None],goal,self.goal))
        if self.randomize:
            vx=(self._random_active(mask,(2,2))*2-1)*.25*self.sim.speed[:,:,None]
            vw=(self._random_active(mask,(2,))*2-1)*.2*self.sim.omega_limit
            randomized_velocity=torch.cat((vx,vw[...,None]),-1)
            self.sim.velocity.copy_(torch.where(mask[:,None,None],
                randomized_velocity,self.sim.velocity))
            radii=.3+self._random_active(mask,())*.5
        else:
            radii=torch.full_like(self.goal_radius,self.base_goal_radius)
        self.goal_radius.copy_(torch.where(mask,radii,self.goal_radius))
        static_velocity=torch.where(self.static_opponent_mask[:,None],
            torch.zeros_like(self.sim.velocity[:,1]),self.sim.velocity[:,1])
        self.sim.velocity[:,1]=torch.where(mask[:,None],static_velocity,self.sim.velocity[:,1])
        self.match_elapsed.copy_(torch.where(mask,torch.zeros_like(self.match_elapsed),self.match_elapsed))
        self.match_remaining.copy_(torch.where(mask,torch.full_like(self.match_remaining,160.),self.match_remaining))
        self.hub_active.copy_(torch.where(mask[:,None],torch.ones_like(self.hub_active),self.hub_active))
        tie_choice=self._random_active(mask,(),high=2).long()
        self.hub_inactive_first.copy_(torch.where(mask,tie_choice,self.hub_inactive_first))
        self.auto_fuel_scores.copy_(torch.where(mask[:,None],torch.zeros_like(self.auto_fuel_scores),self.auto_fuel_scores))
        self._last_hub_zone.copy_(torch.where(mask[:,None],torch.zeros_like(self._last_hub_zone),self._last_hub_zone))
        self.last_hub_zone.copy_(torch.where(mask[:,None],torch.zeros_like(self.last_hub_zone),self.last_hub_zone))
        self._initialize_gamepieces(mask)
        reset_fuel_physics(self,mask)
        self._update_perception(mask,active_mask=mask)
        self._move_adstar_defenders_to_clear_start(mask)
        self._place_defenders_near_clear_adstar_paths(mask)
        self._last_strategic_action.copy_(torch.where(mask,torch.zeros_like(self._last_strategic_action),self._last_strategic_action))
        for value in (self.fuel_acquisition_count,self.fuel_score_count,self.fuel_denied_count,
                      self.fuel_abandoned_count,self.fuel_acquired_event,self.fuel_scored_event,
                      self.fuel_denied_event,self.fuel_abandoned_event,
                      self.fuel_passed_event):
            value.masked_fill_(mask[:,None],0)
        for value in (self.next_intake_time,self.next_score_time,self.next_score,
                      self.next_ferry):
            value.masked_fill_(mask[:,None],0)
        self.last_actions.masked_fill_(mask[:,None],7)
        self._intake_collecting.masked_fill_(mask[:,None],False)
        self._planner_tick.copy_(torch.where(mask,torch.full_like(self._planner_tick,
            self.adstar_replan_interval-1),self._planner_tick))
        # Per-world cadence remains active for full and partial reset batches.
        self.steps.copy_(torch.where(mask,torch.zeros_like(self.steps),self.steps))
        self.last_contact.copy_(torch.where(mask,torch.zeros_like(self.last_contact),self.last_contact))
        self.last_static_contact.copy_(torch.where(mask,torch.zeros_like(self.last_static_contact),self.last_static_contact))
        for values in (self.path_length,self.contact_steps,self.contact_count,self.blocked_steps,self.out_of_bounds_steps):
            values.masked_fill_(mask,0)
        self.last_action.copy_(torch.where(mask[:,None],torch.zeros_like(self.last_action),self.last_action))
        self.action_history.copy_(torch.where(mask[:,None,None],torch.zeros_like(self.action_history),self.action_history))
        if self.randomize and self.max_control_latency_steps:
            delays=self._random_active(mask,(),high=self.max_control_latency_steps+1).long()
            self.control_delay.copy_(torch.where(mask,delays,self.control_delay))
        if self.randomize and self.max_observation_latency_steps:
            delays=self._random_active(mask,(),high=self.max_observation_latency_steps+1).long()
            self.observation_delay.copy_(torch.where(mask,delays,self.observation_delay))
        distance=(self._attacker_objective()-self.sim.pose[:,pidx,:2]).norm(dim=-1)
        self.previous.copy_(torch.where(mask,distance,self.previous))
        fresh=self._obs(mask,active_mask=mask)
        self._last_observation.copy_(torch.where(mask[:,None],fresh,self._last_observation))
        return self._last_observation

    def _sample_static_opponents(self,mask):
        if self.static_opponent_fraction<=0.:
            self.static_opponent_mask.copy_(torch.where(mask,
                torch.zeros_like(self.static_opponent_mask),self.static_opponent_mask))
            return
        sampled=self._random_active(mask,())<self.static_opponent_fraction
        self.static_opponent_mask.copy_(torch.where(mask,sampled,self.static_opponent_mask))
