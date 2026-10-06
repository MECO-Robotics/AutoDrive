"""Opponent and strategic AD* planning for the tensor defense environment."""
from __future__ import annotations

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None

from .tensor_physics import swerve_heading_rate


class TensorDefenseOpponentMixin:
    """Planning and tactical motion methods used by ``TensorDefenseEnv``."""

    def _advance_adstar_cadence(self, active_mask):
        """Return whether to plan and the mask, avoiding a reduction when aligned."""
        if self._planner_full_batch_active and self._planner_tick_aligned:
            self._planner_tick_scalar += 1
            self._planner_tick.add_(1)
            due=self._planner_tick_scalar % self.adstar_replan_interval == 0
            return due,None
        self._planner_tick_aligned=False
        self._planner_full_batch_active=False
        self._planner_tick=torch.where(active_mask,self._planner_tick+1,self._planner_tick)
        replan=active_mask&(self._planner_tick%self.adstar_replan_interval==0)
        return bool(replan.any()),replan

    def _adstar_tactical_velocity(self, waypoint_action, active_mask=None):
        """Decode a normalized field waypoint and route robot 0 to it."""
        target=torch.stack(((waypoint_action[:,0]+1.)*.5*self.sim.field_length,
                            (waypoint_action[:,1]+1.)*.5*self.sim.field_width),dim=-1)
        return self._adstar_target_velocity(target,active_mask)

    def _face_hub_omega(self, robot, intent):
        """Turn a dumper toward its HUB while it executes a scoring objective."""
        pose=self.sim.pose[:,robot]
        delta=self.hub_centers[robot]-pose[:,:2]
        bearing=torch.atan2(delta[:,1],delta[:,0])
        error=torch.atan2(torch.sin(bearing-pose[:,2]),torch.cos(bearing-pose[:,2]))
        rate=swerve_heading_rate(error,self.sim.omega_limit[:,robot],
                                 self.sim.alpha[:,robot])
        return torch.where(intent,rate,torch.zeros_like(rate))

    def _adstar_target_velocity(self, waypoint, active_mask=None):
        """Route robot 0 to an absolute strategic waypoint with AD*."""
        planner=self._adstar_tactical_planner
        defender=self.sim.pose[:,0]
        margin=.5*torch.maximum(self.sim.length[:,0],self.sim.width[:,0])
        low=margin[:,None]
        high=torch.stack((self.sim.field_length-margin,self.sim.field_width-margin),dim=-1)
        waypoint=torch.maximum(torch.minimum(waypoint,high),low)
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                     else active_mask)
        should_replan,replan_mask=self._advance_adstar_cadence(active_mask)
        if should_replan:
            planner.plan(defender[:,:2],waypoint,defender[:,2],self.sim.length[:,0],
                self.sim.width[:,0],self.sim.speed[:,0],lateral_friction=self.sim.lateral_mu[:,0],
                acceleration=self.sim.accel[:,0],active_mask=replan_mask)
        command,_=planner.path_reference(defender[:,:2],self.sim.velocity[:,0,:2],self.sim.speed[:,0])
        distance=(waypoint-defender[:,:2]).norm(dim=-1,keepdim=True)
        command=torch.where(distance<=.3,torch.zeros_like(command),command)
        return torch.cat((command,torch.zeros((self.n,1),device=self.device)),dim=-1)

    def _strategic_target(self,action_class,*,_candidate_data=None):
        """Resolve masked strategic choices into waypoints for the AD* controller."""
        position=self.sim.pose[:,0,:2]
        candidates,valid,_=(self._fuel_candidates(0) if _candidate_data is None
                            else _candidate_data)
        rows=torch.arange(self.n,device=self.device)
        if self.task=="defense":
            attacker_pose,attacker_velocity=self._observed_robot(0,1)
            attacker=attacker_pose[:,:2]
            velocity=attacker_velocity[:,:2]
            speed=velocity.norm(dim=-1,keepdim=True)
            to_defender=position-attacker
            fallback=to_defender/to_defender.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            heading=torch.where(speed>.15,velocity/speed.clamp_min(.15),fallback)
            eta=((position-attacker).norm(dim=-1,keepdim=True)/self.sim.speed[:,0,None].clamp_min(.1)).clamp(0.,1.25)
            contest=attacker+velocity*eta+heading*.55
            hub_delta=self.hub_centers[None,:,:]-attacker[:,None,:]
            hub_distance=hub_delta.norm(dim=-1).clamp_min(1e-6)
            alignment=(heading[:,None,:]*hub_delta/hub_distance[...,None]).sum(-1)
            likely_hub=alignment.argmax(-1)
            target_hub=self.hub_centers[likely_hub]
            target_direction=target_hub-attacker
            target_distance=target_direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            target_direction=target_direction/target_distance
            block_lane=attacker+target_direction*torch.minimum(target_distance*.35,
                torch.full_like(target_distance,.9))
            slot=(action_class-1).clamp(0,3)
            points,_,_=self._perceived_fuel(0)
            candidate=points[rows,candidates[rows,slot]]
            candidate_valid=valid[rows,slot]
            deny=torch.where(candidate_valid[:,None],candidate,block_lane)
            target=torch.where((action_class==0)[:,None],contest,
                torch.where(((action_class>=1)&(action_class<=4))[:,None],deny,
                    torch.where((action_class==5)[:,None],block_lane,
                        torch.where((action_class==6)[:,None],attacker+velocity*.35,contest))))
            fuel_action=(action_class>=1)&(action_class<=4)
            available=self._opponent_track_valid[:,0]|fuel_action
            return torch.where(available[:,None],target,position)
        else:
            own_hub=self.hub_centers[0].expand(self.n,-1)
            hub_direction=position-own_hub
            hub_direction=hub_direction/hub_direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            approach=own_hub+hub_direction*(.595+.5*torch.sqrt(
                self.sim.length[:,0].square()+self.sim.width[:,0].square())+.02)[:,None]
            slot=action_class.clamp(0,3)
            points,_,_=self._perceived_fuel(0)
            collect=points[rows,candidates[rows,slot]]
            collect=torch.where(valid[rows,slot,None],collect,approach)
            direction=own_hub-position
            direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            tactical=position+direction*1.5
            return torch.where((action_class<=3)[:,None],collect,
                torch.where((action_class==4)[:,None],approach,
                    torch.where((action_class==6)[:,None],tactical,position)))

    def _adstar_opponent_velocity(self,target,active_mask=None):
        """Plan robot 1's route to a game-state objective."""
        planner=(self._adstar_planners if self.task=="defense"
                 else self._adstar_defender_planners)
        if planner is None:
            raise RuntimeError("strategic opponent motion requires its AD* planner")
        pose=self.sim.pose[:,1]
        observed_defender,observed_velocity=self._observed_robot(1,0)
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                     else active_mask)
        should_replan,replan_mask=self._advance_adstar_cadence(active_mask)
        if should_replan:
            planner.plan(pose[:,:2],target,pose[:,2],self.sim.length[:,1],self.sim.width[:,1],
                self.sim.speed[:,1],observed_defender[:,:2],observed_velocity[:,:2],
                self.task=="defense",self.sim.lateral_mu[:,1],self.sim.accel[:,1],
                active_mask=replan_mask)
        command,_=planner.path_reference(pose[:,:2],self.sim.velocity[:,1,:2],self.sim.speed[:,1])
        magnitude=command.norm(dim=-1,keepdim=True)
        return command*torch.minimum(torch.ones_like(magnitude),
            self.sim.speed[:,1,None]/magnitude.clamp_min(1e-6))

    def _velocity_intercept_target(self):
        """Deterministic attacker waypoint that evades the defender's predicted lane intercept."""
        attacker=self.sim.pose[:,1,:2]
        defender_pose,defender_velocity=self._observed_robot(1,0)
        defender=defender_pose[:,:2]
        goal=self._attacker_objective()
        route=goal-attacker
        distance=route.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        direction=route/distance
        defender_along=((defender-attacker)*direction).sum(-1).clamp_min(0.)
        time_to_intercept=(defender_along/self.sim.speed[:,1].clamp_min(.1)).clamp(0.,1.2)
        predicted_defender=defender+defender_velocity[:,:2]*time_to_intercept[:,None]
        along=((predicted_defender-attacker)*direction).sum(-1).clamp_min(0.).minimum(distance[:,0])
        closest=attacker+direction*along[:,None]
        lateral=predicted_defender-closest
        threatened=(lateral.norm(dim=-1)<1.35)&(along>.3)&(along<distance[:,0]-.4)
        signed_side=direction[:,0]*lateral[:,1]-direction[:,1]*lateral[:,0]
        side=torch.where(signed_side>=0.,1.,-1.)
        perpendicular=torch.stack((-direction[:,1],direction[:,0]),-1)
        bypass=closest+direction*.9-perpendicular*(side[:,None]*1.2)
        return torch.where(threatened[:,None],bypass,goal)

    def _adstar_attacker_velocity(self,active_mask=None):
        """Batched GPU route planning and velocity tracking; no host state copies."""
        planner=self._adstar_planners
        pose=self.sim.pose[:,1]
        observed_defender,_=self._observed_robot(1,0)
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                     else active_mask)
        should_replan,replan_mask=self._advance_adstar_cadence(active_mask)
        # A host-side fixed cadence avoids the device synchronization caused by
        # checking the per-world mask. The aligned common case uses a scalar;
        # contact replanning remains bounded by the configured cadence.
        if should_replan:
            planner.plan(pose[:,:2],self._attacker_objective(),pose[:,2],self.sim.length[:,1],
                self.sim.width[:,1],self.sim.speed[:,1],observed_defender[:,:2],
                self._observed_robot(1,0)[1][:,:2],True,self.sim.lateral_mu[:,1],self.sim.accel[:,1],
                active_mask=replan_mask)
        command,_=planner.path_reference(pose[:,:2],self.sim.velocity[:,1,:2],self.sim.speed[:,1])
        magnitude=command.norm(dim=-1,keepdim=True)
        return command*torch.minimum(torch.ones_like(magnitude),
            self.sim.speed[:,1,None]/magnitude.clamp_min(1e-6))

    def _defense_target(self, mode, defender_index, attacker_index):
        """Create a defense waypoint from observable tracks, then AD* executes it."""
        defender_pose,defender_velocity=self._observed_robot(defender_index,defender_index)
        attacker_pose,attacker_velocity=self._observed_robot(defender_index,attacker_index)
        defender=defender_pose[:,:2]
        attacker=attacker_pose[:,:2]
        velocity=attacker_velocity[:,:2]
        speed=velocity.norm(dim=-1,keepdim=True)
        toward=defender-attacker
        fallback=toward/toward.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        heading=torch.where(speed>.15,velocity/speed.clamp_min(.15),fallback)
        distance=(defender-attacker).norm(dim=-1,keepdim=True)
        eta=(distance/self.sim.speed[:,defender_index,None].clamp_min(.1)).clamp(0.,1.25)
        intercept=attacker+velocity*eta+heading*.55
        hub=self.hub_centers[attacker_index].expand(self.n,-1)
        to_hub=hub-attacker
        hub_distance=to_hub.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        lane=to_hub/hub_distance
        block=attacker+lane*torch.minimum(hub_distance*.4,torch.full_like(hub_distance,1.0))
        guard=hub-lane*(.595+.5*torch.sqrt(self.sim.length[:,defender_index].square()+
            self.sim.width[:,defender_index].square())+.15)[:,None]
        if mode=="FUEL_DENIAL":
            candidates,valid,distances=self._fuel_candidates(defender_index)
            points,_,_=self._perceived_fuel(defender_index)
            index=distances.masked_fill(~valid,float("inf")).argmin(-1)
            fuel=points[torch.arange(self.n,device=self.device),candidates.gather(1,index[:,None]).squeeze(1)]
            target=torch.where(valid.any(-1)[:,None],fuel,block)
        elif mode=="LANE_BLOCK":
            target=block
        elif mode=="SHADOW":
            target=attacker+velocity*.35
        elif mode=="HUB_GUARD":
            target=guard
        else:
            target=intercept
        tracked=self._opponent_track_valid[:,defender_index].clone()
        if mode=="FUEL_DENIAL":
            tracked |= self._fuel_candidates(defender_index)[1].any(-1)
        return torch.where(tracked[:,None],target,defender)

    def _adstar_defender_velocity(self, target=None, active_mask=None):
        """Route a defender toward a perception-derived objective with AD*."""
        planner=self._adstar_defender_planners
        defender=self.sim.pose[:,1]
        attacker_pose,attacker_observed_velocity=self._observed_robot(1,0)
        attacker=attacker_pose[:,:2]
        attacker_velocity=attacker_observed_velocity[:,:2]
        if target is None:
            attacker_speed=attacker_velocity.norm(dim=-1,keepdim=True)
            to_attacker=attacker-defender[:,:2]
            fallback=to_attacker/to_attacker.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            velocity_direction=torch.where(attacker_speed>.15,
                attacker_velocity/attacker_speed.clamp_min(.15),fallback)
            defender_distance=(attacker-defender[:,:2]).norm(dim=-1,keepdim=True)
            reach_time=(defender_distance/self.sim.speed[:,1,None].clamp_min(.1)).clamp(0.,1.25)
            target=attacker+attacker_velocity*reach_time+velocity_direction*.55
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                     else active_mask)
        should_replan,replan_mask=self._advance_adstar_cadence(active_mask)
        if should_replan:
            planner.plan(defender[:,:2],target,defender[:,2],self.sim.length[:,1],
                self.sim.width[:,1],self.sim.speed[:,1],attacker,attacker_velocity,False,
                self.sim.lateral_mu[:,1],self.sim.accel[:,1],active_mask=replan_mask)
        command,_=planner.path_reference(defender[:,:2],self.sim.velocity[:,1,:2],self.sim.speed[:,1])
        magnitude=command.norm(dim=-1,keepdim=True)
        return command*torch.minimum(torch.ones_like(magnitude),
            self.sim.speed[:,1,None]/magnitude.clamp_min(1e-6))
