"""Physics stepping and reward finalization for the tensor defense environment."""
from __future__ import annotations

import math
from .fuel_physics_runtime import advance_with_fuel

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None

from .reward import (ROTATION_COMMAND_DELTA_PENALTY, ROTATION_COMMAND_PENALTY,
                     SPIN_RATE_PENALTY)


class TensorDefenseStepMixin:
    """Advance the environment and assemble per-step reward and info."""

    def step(self,action,active_mask=None,*,_active_count=None,_return_info=True,
             _capture_observation=True):
        full_batch=active_mask is None
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool)
                     if active_mask is None else
                     torch.as_tensor(active_mask,device=self.device,dtype=torch.bool).reshape(self.n))
        if _active_count is None:
            active_count=self.n if full_batch else int(active_mask.sum().item())
        else:
            active_count=int(_active_count)
        if active_count != self.n:
            self._planner_tick_aligned=False
        self._planner_full_batch_active=(self._planner_tick_aligned and
                                         active_count == self.n)
        if active_count == 0:
            return (self._last_observation,torch.zeros(self.n,device=self.device),
                    torch.zeros(self.n,device=self.device,dtype=torch.bool),
                    torch.zeros(self.n,device=self.device,dtype=torch.bool),{})
        a=torch.as_tensor(action,device=self.device,dtype=self.sim.pose.dtype)
        if self.action_mode=="strategic":
            if a.ndim==2 and a.shape[-1]==1:a=a.squeeze(-1)
            if a.ndim==0:a=a.expand(self.n)
            if tuple(a.shape)!=(self.n,):raise ValueError(f"strategic action must be {(self.n,)} class indices")
            strategy=a.long().clamp(0,7)
            stored_action=torch.zeros((self.n,3),device=self.device,dtype=self.sim.pose.dtype)
            stored_action[:,0]=strategy.to(self.sim.pose.dtype)
        else:
            action_dim=2 if self.action_mode=="tactical" else 3
            if a.shape==(action_dim,):a=a.expand(self.n,action_dim)
            if tuple(a.shape)!=(self.n,action_dim):raise ValueError(f"action must be {(self.n,action_dim)}")
            a=torch.nan_to_num(a).clamp(-1,1)
            stored_action=(torch.cat((a,torch.zeros((self.n,1),device=self.device)),dim=-1)
                           if action_dim==2 else a)
        # Keep the three-channel latency buffer for continuous policies;
        # strategic choices store their categorical id in channel zero.
        shifted=torch.cat((stored_action[:,None,:],self.action_history[:,:-1,:]),dim=1)
        self.action_history=torch.where(active_mask[:,None,None],shifted,self.action_history)
        applied=self.action_history.gather(1,self.control_delay[:,None,None].expand(-1,1,3)).squeeze(1)
        a=applied
        if self.task=="defense" and self.defense_mode is not None:
            target=self._defense_target(self.defense_mode,0,1)
            own=self._adstar_target_velocity(target,active_mask)
            a=torch.cat((own[:,:2]/self.sim.speed[:,0,None].clamp_min(.1),
                         torch.zeros((self.n,1),device=self.device)),dim=-1)
        elif self.action_mode=="tactical":
            own=self._adstar_tactical_velocity(a[:,:2],active_mask)
            a=torch.cat((own[:,:2]/self.sim.speed[:,0,None].clamp_min(.1),
                         torch.zeros((self.n,1),device=self.device)),dim=-1)
        elif self.action_mode=="strategic":
            proposed=applied[:,0].long().clamp(0,7)
            pending=(self._pending_strategic_own_candidates
                     if self.reuse_strategic_own_candidates else None)
            if pending is None:
                candidate_data=self._fuel_candidates(0)
                action_mask=self.strategic_action_mask(_candidate_data=candidate_data)
            else:
                candidate_data,action_mask=pending
            valid=action_mask.gather(1,proposed[:,None]).squeeze(1)
            fallback=action_mask.to(torch.int64).argmax(-1)
            self._last_strategic_action=torch.where(active_mask,
                torch.where(valid,proposed,fallback),self._last_strategic_action)
            target=self._strategic_target(self._last_strategic_action,
                                          _candidate_data=candidate_data)
            own=self._adstar_target_velocity(target,active_mask)
            dumping=self._last_strategic_action==4
            ferrying=self._last_strategic_action==6
            omega=self._face_hub_omega(0,dumping)+self._face_ferry_omega(0,ferrying)
            a=torch.cat((own[:,:2]/self.sim.speed[:,0,None].clamp_min(.1),
                         (omega/self.sim.omega_limit[:,0].clamp_min(.1))[:,None]),dim=-1)
        else:
            own=torch.cat((a[:,:2]*self.sim.speed[:,0,None],(a[:,2]*self.sim.omega_limit[:,0])[:,None]),-1)
        p=self.sim.pose
        # Strategic offense rewards gamepiece events only. Its objective is
        # used below solely for diagnostic distance/path metrics, so avoid a
        # 504-piece nearest-objective reduction on every physics tick in PPO's
        # no-info path. Evaluation still computes the exact original metrics.
        metric_only_objective=(self.skip_strategic_offense_metric_objective and
                               self.action_mode=="strategic" and
                               self.task=="counter_defense" and not _return_info)
        if metric_only_objective:
            goal=None
            old_position=None
            old_distance=None
        else:
            goal=self._attacker_objective()
            old_position=p[:,0,:2].clone()
            old_distance=((goal-p[:,0,:2]).norm(dim=-1)
                           if self.task=="counter_defense" else
                           (goal-p[:,1,:2]).norm(dim=-1))
        # Opponent motion remains batched and device-local. Each scripted
        # behavior supplies a target, then the common speed-limited controller
        # tracks it. ``offense`` aliases pursuit for the defense task.
        kind=self.opponent.get("value","guard") if isinstance(self.opponent,dict) else self.opponent
        if not isinstance(kind,str): kind="guard"
        kind=kind.lower()
        if self.task=="counter_defense":
            kind={"cutoff":"intercept","mirror":"shadow","velocity_intercept":"intercept",
                  "pursuit":"shadow"}.get(kind,kind)
        if self.task=="defense" and kind in ("guard","pursuit"): kind="offense"
        elif self.task=="counter_defense" and kind=="guard": kind="adstar_defender"
        controlled=p[:,0,:2]
        goal_target=goal
        if self.task=="defense":
            # The second robot is the attacker; its default target is the goal.
            target=goal_target
        else:
            target=controlled
        if kind=="guard":
            target=.5*(controlled+goal_target)
        elif kind=="cutoff":
            route=goal_target-controlled
            unit=route/route.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            target=controlled+self.sim.velocity[:,0,:2]*.45+unit*.65
        elif kind=="velocity_intercept":
            target=self._velocity_intercept_target()
        elif kind=="mirror":
            target=2*goal_target-controlled
        elif kind=="pursuit":
            target=goal_target if self.task=="defense" else controlled
        elif kind in ("adstar_defender","intercept","lane_block","fuel_denial","shadow","hub_guard"):
            if self.task!="counter_defense" or self._adstar_defender_planners is None:
                raise ValueError("scripted defense modes are available for the counter-defense task")
            mode=self.defense_mode if kind=="adstar_defender" else kind.upper()
            target=self._defense_target(mode,1,0)
            oppxy=self._opponent_velocity_command(self._adstar_defender_velocity(target,active_mask),active_mask)
            opp=torch.cat((oppxy,torch.zeros((self.n,1),device=self.device)),dim=-1)
            commands=torch.stack((own,opp),dim=1)
            advance_with_fuel(self,commands,active_mask); self.steps+=active_mask.long()
            return self._finish_step(goal,old_position,old_distance,a,active_mask,active_count,
                                     _return_info,_capture_observation)
        elif kind in ("adstar","offense"):
            if self.task!="defense" or self._adstar_planners is None:
                raise ValueError("the AD* attacker is available for the defense task")
            # _swerve accepts field-relative chassis velocity and performs the
            # body-frame conversion internally for module kinematics.
            oppxy=self._opponent_velocity_command(self._adstar_attacker_velocity(active_mask),active_mask)
            carrying=self._own_possession(1)>0
            omega=self._face_hub_omega(1,carrying)
            opp=torch.cat((oppxy,omega[:,None]),dim=-1)
            commands=torch.stack((own,opp),dim=1)
            advance_with_fuel(self,commands,active_mask); self.steps+=active_mask.long()
            return self._finish_step(goal,old_position,old_distance,a,active_mask,active_count,
                                     _return_info,_capture_observation)
        elif kind=="random":
            count=active_count
            theta_values=torch.rand((count,),device=self.device,generator=self.generator)*(2*math.pi)
            speed_values=torch.rand((count,),device=self.device,generator=self.generator)*self.sim.speed[active_mask,1]
            theta=torch.zeros((self.n,),device=self.device).masked_scatter(active_mask,theta_values)
            speed=torch.zeros((self.n,),device=self.device).masked_scatter(active_mask,speed_values)
            oppxy=torch.stack((theta.cos()*speed,theta.sin()*speed),-1)
            oppxy=self._opponent_velocity_command(oppxy,active_mask)
            opp=torch.cat((oppxy,torch.zeros((self.n,1),device=self.device)),dim=-1)
            commands=torch.stack((own,opp),dim=1)
            advance_with_fuel(self,commands,active_mask); self.steps+=active_mask.long()
            return self._finish_step(goal,old_position,old_distance,a,active_mask,active_count,
                                     _return_info,_capture_observation)
        direction=target-p[:,1,:2]
        oppxy=direction/(direction.norm(dim=-1,keepdim=True).clamp_min(1e-8))*self.sim.speed[:,1,None]
        oppxy=self._opponent_velocity_command(oppxy,active_mask)
        carrying=self._own_possession(1)>0
        omega=self._face_hub_omega(1,carrying) if self.task=="defense" else torch.zeros((self.n,),device=self.device)
        opp=torch.cat((oppxy,omega[:,None]),dim=-1)
        commands=torch.stack((own,opp),dim=1)
        advance_with_fuel(self,commands,active_mask); self.steps+=active_mask.long()
        return self._finish_step(goal,old_position,old_distance,a,active_mask,active_count,
                                 _return_info,_capture_observation)

    def _finish_step(self,goal,old_position,old_distance,action,active_mask=None,
                     active_count=None,return_info=True,capture_observation=True):
        metric_only_objective=(self.skip_strategic_offense_metric_objective and
                               self.action_mode=="strategic" and
                               self.task=="counter_defense" and not return_info)
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                     else active_mask)
        opponent_velocity=torch.where(self.static_opponent_mask[:,None],
            torch.zeros_like(self.sim.velocity[:,1]),self.sim.velocity[:,1])
        self.sim.velocity[:,1]=torch.where(active_mask[:,None],opponent_velocity,self.sim.velocity[:,1])
        p=self.sim.pose
        acquisitions,scores,denied=self._update_gamepieces(active_mask)
        self._update_perception(active_mask=active_mask,active_count=active_count)
        d=(torch.zeros_like(self.previous) if metric_only_objective else
           (goal-p[:,1,:2]).norm(dim=-1))
        attacker_index=0 if self.task=="counter_defense" else 1
        if self.action_mode=="strategic":
            attack_score=scores[:,attacker_index].float()
            # A score is one cycle event, not an episode boundary. Strategic
            # episodes continue until the full physics-time match horizon.
            terminated=torch.zeros_like(attack_score,dtype=torch.bool)
            if self.task=="counter_defense":
                reward=attack_score+.05*acquisitions[:,0].float()-.001
                score=torch.zeros_like(reward)
            else:
                reward=-attack_score-.02*acquisitions[:,1].float()-.001
                route=goal-p[:,1,:2]; rs=route.square().sum(-1).clamp_min(1e-8)
                proj=(((p[:,0,:2]-p[:,1,:2])*route).sum(-1)/rs).clamp(0,1)
                closest=p[:,1,:2]+proj[:,None]*route; lane=(p[:,0,:2]-closest).norm(dim=-1)
                score=torch.exp(-lane/.7)*torch.exp(-((proj-.55)/.35).square())
                reward+=.01*score
        elif self.task=="counter_defense":
            controlled_d=(goal-p[:,0,:2]).norm(dim=-1)
            reward=(self.previous-controlled_d)*2-.01
            terminated=controlled_d<self.goal_radius
            reward+=torch.where(terminated,10.,0.)
            self.previous=torch.where(active_mask,controlled_d,self.previous)
            score=torch.zeros_like(reward)
        else:
            reward=(d-self.previous)*2-.01
            terminated=d<self.goal_radius
            reward+=torch.where(terminated,-10.,0.)
            route=goal-p[:,1,:2]; rs=route.square().sum(-1).clamp_min(1e-8)
            proj=(((p[:,0,:2]-p[:,1,:2])*route).sum(-1)/rs).clamp(0,1)
            closest=p[:,1,:2]+proj[:,None]*route; lane=(p[:,0,:2]-closest).norm(dim=-1)
            score=torch.exp(-lane/.7)*torch.exp(-((proj-.55)/.35).square()); reward+=.04*score
            self.previous=torch.where(active_mask,d,self.previous)
        if self.action_mode=="strategic":
            pass
        elif self.task=="counter_defense":
            reward+=.05*acquisitions[:,0].float()+scores[:,0].float()-scores[:,1].float()
        else:
            reward-=.02*acquisitions[:,1].float()+scores[:,1].float()
        if metric_only_objective:
            moved=None
        else:
            new_distance=((goal-p[:,0,:2]).norm(dim=-1)
                          if self.task=="counter_defense" else d)
            blocked=(old_distance-new_distance)<.002
            moved=(p[:,0,:2]-old_position).norm(dim=-1)
        contact=self.sim.robot_contact
        contact_started=contact&~self.last_contact
        action_delta=action-self.last_action
        smoothness=action_delta.norm(dim=-1)
        spin_rate_ratio=self.sim.velocity[:,0,2]/self.sim.omega_limit[:,0].clamp_min(.1)
        maneuver_penalty=(SPIN_RATE_PENALTY*spin_rate_ratio.square()+
                          ROTATION_COMMAND_PENALTY*action[:,2].square()+
                          ROTATION_COMMAND_DELTA_PENALTY*action_delta[:,2].square())
        wall=self.sim.wall_contact[:,0].any(-1)
        if not metric_only_objective:
            self.path_length+=moved*active_mask
            self.contact_steps+=contact.float()*active_mask
            self.contact_count+=contact_started.float()*active_mask
            self.blocked_steps+=blocked.float()*active_mask
            self.out_of_bounds_steps+=wall.float()*active_mask
        self.last_contact.copy_(torch.where(active_mask,contact,self.last_contact))
        self.last_action.copy_(torch.where(active_mask[:,None],action,self.last_action))
        truncated=(self.steps>=self.horizon)&active_mask
        if return_info:
            if self.action_mode=="strategic":
                success=(truncated&(self.fuel_score_count[:,0]>0)) if self.task=="counter_defense" else (
                    truncated&(self.fuel_score_count[:,1]==0))
            else:
                success=terminated if self.task=="counter_defense" else (truncated&~terminated)
        static_contact=self.sim.field_contact[:,0]|self.sim.wall_contact[:,0].any(-1)
        if return_info:
            opponent_wall=self.sim.wall_contact[:,1].any(-1)
            opponent_static_contact=self.sim.field_contact[:,1]|opponent_wall
        static_contact_started=static_contact&~self.last_static_contact&active_mask
        self.last_static_contact.copy_(torch.where(active_mask,static_contact,self.last_static_contact))
        # Allow useful bumper engagement with the attacker. Charge only once
        # when the controlled robot newly hits static field geometry.
        reward-=static_contact_started.float()*.05+.001*smoothness+maneuver_penalty
        if self.task=="defense":
            reward+=torch.where(truncated&~terminated,10.,0.)
        if not return_info:
            reward=torch.where(active_mask,reward,torch.zeros_like(reward))
            terminated &= active_mask
            truncated &= active_mask
            obs=self._obs(active_mask=active_mask,active_count=active_count,
                          capture_output=capture_observation,
                          capture_raw_mask=getattr(
                              self, "_ppo_observation_capture_mask", None),
                          capture_raw_rows=getattr(
                              self, "_ppo_observation_capture_rows", None))
            if capture_observation:
                if active_count == self.n:
                    self._last_observation=obs.clone()
                else:
                    self._last_observation=torch.where(active_mask[:,None],obs,self._last_observation)
            return obs,reward,terminated,truncated,{}
        info={"contact":contact.float(),"contact_duration":contact.float()*self.dt,
              "opponent_contact":self.sim.opponent_contact.float(),
              "field_contact":self.sim.field_contact[:,0].float(),
              "wall_contact":self.sim.wall_contact[:,0].any(-1).float(),
              "static_contact_started":static_contact_started.float(),
              "opponent_field_contact":self.sim.field_contact[:,1].float(),
              "opponent_wall_contact":opponent_wall.float(),
              "opponent_static_contact":opponent_static_contact.float(),
              "contact_count":contact_started.float(),"time_blocked":blocked.float()*self.dt,
              "time_to_goal":((self.match_elapsed*(scores[:,attacker_index]>0)
                  if self.action_mode=="strategic" else self.steps*self.dt*terminated)).float(),"success":success.float(),
              "defensive_delay":torch.full_like(self.steps,self.dt,dtype=torch.float32) if self.task=="defense" else torch.zeros_like(self.steps,dtype=torch.float32),
              "useful_position":score,"path_length":moved,"command_smoothness":smoothness,
              "spin_rate_ratio":spin_rate_ratio.abs(),"maneuver_penalty":maneuver_penalty,
              "path_efficiency":torch.where(terminated,self.initial_distance/self.path_length.clamp_min(1e-6),torch.zeros_like(self.path_length)),
              "out_of_bounds":wall.float()}
        possession=(self.piece_active[:,:,None] &
                    (self.piece_owner[:,:,None]==torch.arange(2,device=self.device)[None,None,:])).sum(1)
        info.update({
            "fuel_acquired_event":acquisitions*active_mask[:,None],
            "fuel_scored_event":scores*active_mask[:,None],
            "fuel_denied_event":denied*active_mask[:,None],
            "fuel_abandoned_event":self.fuel_abandoned_event*active_mask[:,None],
            "fuel_acquisition_count":self.fuel_acquisition_count,
            "fuel_score_count":self.fuel_score_count,
            "fuel_denied_count":self.fuel_denied_count,
            "fuel_abandoned_count":self.fuel_abandoned_count,
            "fuel_possession_count":possession,
            "fuel_capacity_per_robot":self.fuel_capacity,
            "fuel_intake_interval_s":self.intake_interval,
            "fuel_score_interval_s":self.score_interval,
            "match_elapsed":self.match_elapsed,
            "match_remaining":self.match_remaining,
            "match_progress":self.match_elapsed/160.,
            "hub_active":self.hub_active,
            "game_state":{
                "piece_pos":self.piece_pos,
                "piece_vel":self.piece_vel,
                "piece_active":self.piece_active,
                "piece_owner":self.piece_owner,
                "piece_zone":self.piece_zone,
                "piece_type":self.piece_type,
                "fuel_acquired_event":acquisitions*active_mask[:,None],
                "fuel_scored_event":scores*active_mask[:,None],
                "fuel_denied_event":denied*active_mask[:,None],
                "fuel_abandoned_event":self.fuel_abandoned_event*active_mask[:,None],
                "fuel_possession_count":possession,
                "fuel_capacity_per_robot":self.fuel_capacity,
                "fuel_intake_interval_s":self.intake_interval,
                "fuel_score_interval_s":self.score_interval,
                "fuel_score_count":self.fuel_score_count,
                "match_elapsed":self.match_elapsed,
                "match_remaining":self.match_remaining,
                "match_progress":self.match_elapsed/160.,
                "hub_active":self.hub_active,
                "hub_centers":self.hub_centers,
                "fuel_count":self.fuel_count,
                "preloads_per_robot":self.preloads_per_robot,
                "piece_owner_codes":{"free_or_off_field":-1,"scored":-2,"robots_or_teammates": "0..5"},
                "piece_zone_codes":{"neutral":0,"red_depot":1,"blue_depot":2,
                    "red_outpost_stock":3,"blue_outpost_stock":4,"preload":5,"scored":6},
            }})
        if hasattr(self, "_last_opponent_command"):
            info["opponent_effort_vector"] = self._last_opponent_command
        if self._adstar_tactical_planner is not None:
            info["controlled_adstar_path"] = self._adstar_tactical_planner.last_path
            info["controlled_adstar_path_lengths"] = self._adstar_tactical_planner.last_lengths
        opponent_kind=self.opponent.get("value","") if isinstance(self.opponent,dict) else self.opponent
        if self._adstar_planners is not None and opponent_kind == "adstar":
            info["adstar_paths"]=self._adstar_planners.last_path
            info["adstar_path_lengths"]=self._adstar_planners.last_lengths
            info["predicted_intercepts"]=self._adstar_planners.last_intercept
            info["predicted_intercept_times"]=self._adstar_planners.last_intercept_time
        elif self._adstar_defender_planners is not None and opponent_kind in ("guard", "adstar_defender"):
            info["adstar_paths"]=self._adstar_defender_planners.last_path
            info["adstar_path_lengths"]=self._adstar_defender_planners.last_lengths
        if self._adstar_tactical_planner is not None:
            info["tactical_adstar_paths"]=self._adstar_tactical_planner.last_path
            info["tactical_adstar_path_lengths"]=self._adstar_tactical_planner.last_lengths
        for key in ("contact","contact_duration","opponent_contact","field_contact",
                    "wall_contact","static_contact_started","opponent_field_contact",
                    "opponent_wall_contact","opponent_static_contact","contact_count",
                    "time_blocked","time_to_goal","success","defensive_delay",
                    "useful_position","path_length","command_smoothness",
                    "spin_rate_ratio","maneuver_penalty","path_efficiency","out_of_bounds"):
            value=info.get(key)
            if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==self.n:
                shape=(self.n,)+(1,)*(value.ndim-1)
                info[key]=torch.where(active_mask.reshape(shape),value,torch.zeros_like(value))
        reward=torch.where(active_mask,reward,torch.zeros_like(reward))
        terminated &= active_mask
        truncated &= active_mask
        obs=self._obs(active_mask=active_mask,active_count=active_count,
                      capture_output=capture_observation)
        if capture_observation:
            if active_count == self.n:
                self._last_observation=obs.clone()
            else:
                self._last_observation=torch.where(active_mask[:,None],obs,self._last_observation)
        return obs,reward,terminated,truncated,info
