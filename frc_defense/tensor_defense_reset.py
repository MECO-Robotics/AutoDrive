"""Episode setup and reset lifecycle for the tensor defense environment."""
from __future__ import annotations

import math

from .field import ALLIANCE_ZONE_DEPTH

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None


class TensorDefenseResetMixin:
    """Sample episode layouts and reset full or partial environment batches."""

    def _sample_episode_endpoints(self, goal_override=None, start_zone=None, goal_zone=None, mask=None):
        """Sample endpoints in requested zones, or random distinct field zones."""
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
            if zones.numel()!=self.n or bool(((zones<0)|(zones>2)).any().item()):
                raise ValueError("zone ids must contain one value per environment in 0..2")
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
                if bool(same.any().item()):
                    raise ValueError("start and goal zones must be different")
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
            elif bool((start_zone==goal_zone).any().item()):
                raise ValueError("start and goal zones must be different")

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
            if bool(unresolved.any().item()):
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
                    allowed=torch.where(separated.any(-1,keepdim=True),separated,allowed)
                has_legal_point=allowed.any(-1)
                if not bool(has_legal_point[unresolved].all().item()):
                    raise RuntimeError("no legal endpoint exists in the selected field zone")
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
        selected=(torch.ones(self.n,device=self.device,dtype=torch.bool) if mask is None
                  else torch.as_tensor(mask,device=self.device,dtype=torch.bool))
        planner=self._adstar_planners
        planner.plan(starts,self._attacker_objective(),self.sim.pose[:,1,2],self.sim.length[:,1],
                     self.sim.width[:,1],active_mask=selected)
        candidate=planner.defender_spawn(starts,selected,self.generator)
        self.sim.pose[:,0,:2]=torch.where(selected[:,None],candidate,self.sim.pose[:,0,:2])

    def _move_adstar_defenders_to_clear_start(self, mask=None):
        """Keep an AD* guard from spawning inside its inflated obstacle map."""
        bumps=self.sim.bump_regions
        if self._adstar_defender_planners is None or not bumps.numel():
            return
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
            if not bool(needs_move.any().item()):
                break

    def reset(self,seed=None,options=None):
        if seed is not None:self.generator.manual_seed(int(seed))
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
            self.sim.velocity[:,:,:2]=(torch.rand((self.n,2,2),device=self.device,generator=self.generator)*2-1)*.25*self.sim.speed[:,:,None]
            self.sim.velocity[:,:,2]=(torch.rand((self.n,2),device=self.device,generator=self.generator)*2-1)*.2*self.sim.omega_limit
            self.goal_radius=.3+torch.rand((self.n,),device=self.device,generator=self.generator)*.5
        else:
            self.goal_radius.fill_(self.base_goal_radius)
        self.sim.velocity[:,1]=torch.where(self.static_opponent_mask[:,None],
            torch.zeros_like(self.sim.velocity[:,1]),self.sim.velocity[:,1])
        if options and "goal_radius" in options:
            self.goal_radius.copy_(torch.as_tensor(options["goal_radius"],device=self.device,dtype=self.goal.dtype).expand_as(self.goal_radius))
        self.match_elapsed.zero_(); self.match_remaining.fill_(160.)
        self.hub_active.fill_(True)
        self.hub_inactive_first=torch.randint(2,(self.n,),device=self.device,generator=self.generator)
        self.auto_fuel_scores.zero_()
        for value in (self.fuel_acquisition_count,self.fuel_score_count,self.fuel_denied_count,
                      self.fuel_abandoned_count,self.fuel_acquired_event,self.fuel_scored_event,
                      self.fuel_denied_event,self.fuel_abandoned_event):
            value.zero_()
        for value in (self.next_intake_time,self.next_score_time):
            value.zero_()
        self._initialize_gamepieces()
        self._update_perception(torch.ones((self.n,),device=self.device,dtype=torch.bool))
        self._move_adstar_defenders_to_clear_start()
        self._place_defenders_near_clear_adstar_paths()
        self._last_hub_zone.zero_(); self._last_strategic_action.zero_()
        self._last_learned_opponent_action.zero_()
        self._planner_tick.fill_(self.adstar_replan_interval-1)
        self._planner_tick_scalar=self.adstar_replan_interval-1
        self._planner_tick_aligned=True
        self._planner_full_batch_active=False
        self.action_history.zero_()
        self.control_delay.zero_()
        if self.randomize and self.max_control_latency_steps:
            self.control_delay.random_(self.max_control_latency_steps+1,generator=self.generator)
        self.observation_delay.zero_()
        if self.randomize and self.max_observation_latency_steps:
            self.observation_delay.random_(self.max_observation_latency_steps+1,generator=self.generator)
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
        mask=torch.as_tensor(mask,device=self.device,dtype=torch.bool).reshape(self.n)
        full_reset=bool(mask.all().item())
        # Per-world reset phases may now differ; only a full-batch reset can
        # restore the common scalar cadence.
        self._planner_tick_aligned=full_reset
        self._planner_full_batch_active=False
        self.sim.reset_done(mask)
        self._sample_static_opponents(mask)
        if self._adstar_planners is not None:
            self._adstar_contact_latched &= ~mask
        if self.randomize:
            vx=(self._random_active(mask,(2,2))*2-1)*.25*self.sim.speed[:,:,None]
            vw=(self._random_active(mask,(2,))*2-1)*.2*self.sim.omega_limit
            self.sim.velocity=torch.where(mask[:,None,None],torch.cat((vx,vw[...,None]),-1),self.sim.velocity)
        static_velocity=torch.where(self.static_opponent_mask[:,None],
            torch.zeros_like(self.sim.velocity[:,1]),self.sim.velocity[:,1])
        self.sim.velocity[:,1]=torch.where(mask[:,None],static_velocity,self.sim.velocity[:,1])
        pidx=0 if self.task=="counter_defense" else 1
        start, goal = self._sample_episode_endpoints(mask=mask)
        self.sim.pose[:, pidx, :2] = torch.where(mask[:, None], start, self.sim.pose[:, pidx, :2])
        self.goal=torch.where(mask[:,None],goal,self.goal)
        self.match_elapsed=torch.where(mask,torch.zeros_like(self.match_elapsed),self.match_elapsed)
        self.match_remaining=torch.where(mask,torch.full_like(self.match_remaining,160.),self.match_remaining)
        self.hub_active=torch.where(mask[:,None],torch.ones_like(self.hub_active),self.hub_active)
        tie_choice=self._random_active(mask,(),high=2).long()
        self.hub_inactive_first=torch.where(mask,tie_choice,self.hub_inactive_first)
        self.auto_fuel_scores=torch.where(mask[:,None],torch.zeros_like(self.auto_fuel_scores),self.auto_fuel_scores)
        self._last_hub_zone=torch.where(mask[:,None],torch.zeros_like(self._last_hub_zone),self._last_hub_zone)
        self._initialize_gamepieces(mask)
        self._update_perception(mask,active_mask=mask)
        self._move_adstar_defenders_to_clear_start(mask)
        self._place_defenders_near_clear_adstar_paths(mask)
        self._last_strategic_action=torch.where(mask,torch.zeros_like(self._last_strategic_action),self._last_strategic_action)
        self._last_learned_opponent_action=torch.where(mask,
            torch.zeros_like(self._last_learned_opponent_action),
            self._last_learned_opponent_action)
        for value in (self.fuel_acquisition_count,self.fuel_score_count,self.fuel_denied_count,
                      self.fuel_abandoned_count,self.fuel_acquired_event,self.fuel_scored_event,
                      self.fuel_denied_event,self.fuel_abandoned_event):
            value.masked_fill_(mask[:,None],0)
        for value in (self.next_intake_time,self.next_score_time):
            value.masked_fill_(mask[:,None],0)
        self._planner_tick=torch.where(mask,torch.full_like(self._planner_tick,
            self.adstar_replan_interval-1),self._planner_tick)
        if full_reset:
            self._planner_tick_scalar=self.adstar_replan_interval-1
        radii=.3+self._random_active(mask,())*.5
        self.goal_radius=torch.where(mask,radii if self.randomize else torch.full_like(radii,self.base_goal_radius),self.goal_radius)
        self.steps=torch.where(mask,torch.zeros_like(self.steps),self.steps)
        self.last_contact=torch.where(mask,torch.zeros_like(self.last_contact),self.last_contact)
        self.last_static_contact=torch.where(mask,torch.zeros_like(self.last_static_contact),self.last_static_contact)
        for values in (self.path_length,self.contact_steps,self.contact_count,self.blocked_steps,self.out_of_bounds_steps):
            values.masked_fill_(mask,0)
        self.last_action=torch.where(mask[:,None],torch.zeros_like(self.last_action),self.last_action)
        self.action_history=torch.where(mask[:,None,None],torch.zeros_like(self.action_history),self.action_history)
        if self.randomize and self.max_control_latency_steps:
            delays=self._random_active(mask,(),high=self.max_control_latency_steps+1).long()
            self.control_delay=torch.where(mask,delays,self.control_delay)
        if self.randomize and self.max_observation_latency_steps:
            delays=self._random_active(mask,(),high=self.max_observation_latency_steps+1).long()
            self.observation_delay=torch.where(mask,delays,self.observation_delay)
        self._ppo_observation_delay_groups=None
        distance=(self._attacker_objective()-self.sim.pose[:,pidx,:2]).norm(dim=-1)
        self.previous=torch.where(mask,distance,self.previous)
        fresh=self._obs(mask,active_mask=mask)
        self._last_observation=torch.where(mask[:,None],fresh,self._last_observation)
        return self._last_observation

    def _sample_static_opponents(self,mask):
        if self.static_opponent_fraction<=0.:
            self.static_opponent_mask=torch.where(mask,torch.zeros_like(self.static_opponent_mask),
                self.static_opponent_mask)
            return
        sampled=self._random_active(mask,())<self.static_opponent_fraction
        self.static_opponent_mask=torch.where(mask,sampled,self.static_opponent_mask)
