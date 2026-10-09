"""Gamepiece and match lifecycle behavior for the tensor defense environment."""
from __future__ import annotations

import math
from .fuel_physics_runtime import launch_fuel, shooter_respawn_targets

from .field import ALLIANCE_ZONE_DEPTH, grid_array_points

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None

if torch is not None:
    from .tensor_gamepieces import (
        FUSED_GAMEPIECES_HIP_ENABLED,
        update_gamepieces as _update_gamepieces_hip,
    )
else:
    FUSED_GAMEPIECES_HIP_ENABLED = False
    _update_gamepieces_hip = None


class TensorDefenseGamepieceMixin:
    """Manage staged FUEL, match timing, pickup, and HUB scoring."""
    def _clear_intersected_fuel_tracks(self, active_mask=None):
        """Forget fuel observations inside a robot's contact/intake envelope."""
        delta=self._track_pos[:,:,:,None,:]-self.sim.pose[:,None,None,:,:2]
        forward=torch.stack((self.sim.pose[:,:,2].cos(),
                             self.sim.pose[:,:,2].sin()),-1)
        left=torch.stack((-forward[...,1],forward[...,0]),-1)
        longitudinal=(delta*forward[:,None,None,:,:]).sum(-1)
        lateral=(delta*left[:,None,None,:,:]).sum(-1).abs()
        contact_envelope=(
            (longitudinal>=-self.sim.length[:,None,None,:]*.5-.0825)&
            (longitudinal<=self.sim.length[:,None,None,:]*.5+.35+.0825)&
            (lateral<=self.sim.width[:,None,None,:]*.5+.075+.0825))
        intersected=contact_envelope.any(-1)
        if active_mask is not None:
            intersected &= active_mask[:,None,None]
        self._track_mask.masked_fill_(intersected,False)
        self._track_age.masked_fill_(intersected,float("inf"))

    def release_outpost_fuel(self, alliance, position, count=1, mask=None):
        """Model a human-player fuel release at a caller-supplied chute opening.

        ``position`` is the measured field-relative release point; the simulator
        deliberately does not guess the in-field chute opening coordinate.
        Returns the number released per environment.
        """
        side=str(alliance).lower()
        if side not in ("red","blue"):
            raise ValueError("alliance must be red or blue")
        if count<0: raise ValueError("count must be nonnegative")
        selected=(torch.ones((self.n,),device=self.device,dtype=torch.bool) if mask is None
                  else torch.as_tensor(mask,device=self.device,dtype=torch.bool).reshape(self.n))
        point=torch.as_tensor(position,device=self.device,dtype=self.sim.pose.dtype)
        if point.shape==(2,):point=point.expand(self.n,2)
        if tuple(point.shape)!=(self.n,2):raise ValueError(f"position must be {(self.n,2)}")
        zone=3 if side=="red" else 4
        released=torch.zeros((self.n,),device=self.device,dtype=torch.long)
        rows=torch.arange(self.n,device=self.device)
        for _ in range(min(int(count),24)):
            stock=(self.piece_zone==zone)&~self.piece_active
            available=stock.any(-1)&selected
            index=stock.to(torch.int32).argmax(-1)
            self.piece_pos[rows[available],index[available]]=point[available]
            self.piece_vel[rows[available],index[available]]=0.
            self.piece_zone[rows[available],index[available]]=0
            self.piece_owner[rows[available],index[available]]=-1
            self.piece_active[rows[available],index[available]]=True
            released+=available.long()
        return released

    def _initialize_gamepieces(self, mask=None):
        """Reset 2026 FUEL staging in a fixed-size tensor catalog of up to 600 slots.

        Default count follows the official 504-piece match staging. The 2D
        model places 24 at each DEPOT and the neutral count (360..408) in the
        official neutral-pile footprint. The 24 per OUTPOST remain inactive
        off-field stock behind the chute. Ball-ball and vertical dynamics are
        intentionally omitted.
        """
        if mask is None:
            mask=torch.ones((self.n,),device=self.device,dtype=torch.bool)
        else:
            mask=torch.as_tensor(mask,device=self.device,dtype=torch.bool).reshape(self.n)
        positions=torch.zeros_like(self.piece_pos)
        active=torch.zeros_like(self.piece_active)
        owner=torch.full_like(self.piece_owner,-1)
        zone=torch.full_like(self.piece_zone,-1)
        kind=torch.full_like(self.piece_type,-1)
        count=self.fuel_count
        # The manual's 360..408 neutral count results from 0..48 robot preloads.
        # Default to a randomized preload count for the six-robot alliance; a
        # fixed per-robot count can be supplied for controlled experiments.
        max_preload=min(8,max(0,(count-96)//6))
        per_robot=(self._random_active(mask,(),high=max_preload+1).long()
                   if self.preloaded_per_robot is None else
                   torch.full((self.n,),self.preloaded_per_robot,device=self.device,dtype=torch.long))
        neutral_count=count-96-6*per_robot
        half_ball=.075
        def grid_rect(num,x0,x1,y0,y1):
            return torch.tensor(grid_array_points(num,x0,x1,y0,y1),
                                device=self.device,dtype=positions.dtype).reshape(num,2).expand(self.n,-1,2)
        depots={getattr(box,"name",""):box for box in self.field_boxes}
        red_depot=depots.get("red_depot")
        blue_depot=depots.get("blue_depot")
        if red_depot is None:
            red_depot=(.35,self.sim.field_width/2-2.39,.69,.53)
            blue_depot=(self.sim.field_length-.35,self.sim.field_width/2+2.39,.69,.53)
        else:
            red_depot=(red_depot.x,red_depot.y,red_depot.length,red_depot.width)
            blue_depot=(blue_depot.x,blue_depot.y,blue_depot.length,blue_depot.width)
        for first,rect,front in ((0,red_depot,1),(24,blue_depot,-1)):
            x,y,length,width=rect
            # Stage within/along the low DEPOT footprint; the planar solver
            # treats these FUEL pieces as non-colliding objects.
            dep_pos=grid_rect(24,max(half_ball,x-length/2),min(self.sim.field_length-half_ball,x+length/2),
                                max(half_ball,y-width/2),min(self.sim.field_width-half_ball,y+width/2))
            positions[:,first:first+24]=dep_pos
            zone[:,first:first+24]=1 if front==1 else 2
        center_y=self.sim.field_width/2
        # Outpost stock remains off-field behind the chute/door; no guessed
        # in-field spawn is used. It is available to future explicit release events.
        zone[:,48:72]=3
        zone[:,72:96]=4
        neutral_start=96
        neutral_max=count-96
        # Official neutral pile: 206 x 72 in (~5.23 x 1.83 m), roughly split
        # across the center line in a regular grid array.
        neutral=grid_rect(neutral_max,
            max(half_ball,self.sim.field_length/2-.915),
            min(self.sim.field_length-half_ball,self.sim.field_length/2+.915),
            max(half_ball,center_y-2.615),min(self.sim.field_width-half_ball,center_y+2.615))
        positions[:,neutral_start:neutral_start+neutral_max]=neutral
        neutral_slots=torch.arange(neutral_max,device=self.device)[None,:]<neutral_count[:,None]
        zone[:,neutral_start:neutral_start+neutral_max]=torch.where(neutral_slots,0,-1)
        rows=torch.arange(self.n,device=self.device)[:,None]
        within=torch.arange(8,device=self.device)[None,:]
        for robot in range(6):
            indices=neutral_start+neutral_count[:,None]+robot*per_robot[:,None]+within
            valid=within<per_robot[:,None]
            if robot<2:
                preload_position=self.sim.pose[:,robot,None,:2].expand(-1,8,-1)
            else:
                side=0 if robot in (2,3) else 1
                x=(1.2 if side==0 else self.sim.field_length-1.2)
                preload_position=torch.zeros((self.n,8,2),device=self.device)
                preload_position[:,:,0]=x
                preload_position[:,:,1]=self.sim.field_width/2+(robot-3.5)*.55
            rr=rows.expand(-1,8)[valid]
            cc=indices[valid]
            positions[rr,cc]=preload_position[valid]
            active[rr,cc]=True
            zone[rr,cc]=5
            owner[rr,cc]=robot
        if getattr(self, "random_gamepiece_placement", False):
            # Scatter all active fuel across the field with ball-diameter spacing.
            min_distance=2*half_ball
            for row in range(self.n):
                active_indices=torch.nonzero(active[row],as_tuple=False).flatten().tolist()
                accepted=[]
                attempts=0
                while len(accepted)<len(active_indices) and attempts<max(10000,len(active_indices)*1000):
                    attempts+=1
                    candidate=(half_ball+float(self._random((1,))[0].item())*(self.sim.field_length-2*half_ball),
                               half_ball+float(self._random((1,))[0].item()*(self.sim.field_width-2*half_ball)))
                    if all((candidate[0]-x)**2+(candidate[1]-y)**2 >= min_distance**2
                           for x,y in accepted):
                        accepted.append(candidate)
                if len(accepted)!=len(active_indices):
                    raise RuntimeError("could not place non-overlapping random gamepieces")
                if accepted:
                    positions[row,active_indices]=torch.tensor(accepted,device=self.device,
                                                                  dtype=positions.dtype)
        active[:,:48]=True
        active[:,neutral_start:neutral_start+neutral_max]|=neutral_slots
        kind[:,:count]=0
        self.piece_pos=torch.where(mask[:,None,None],positions,self.piece_pos)
        self.piece_vel=torch.where(mask[:,None,None],torch.zeros_like(positions),self.piece_vel)
        self.piece_active=torch.where(mask[:,None],active,self.piece_active)
        self.piece_owner=torch.where(mask[:,None],owner,self.piece_owner)
        self.piece_zone=torch.where(mask[:,None],zone,self.piece_zone)
        self.piece_type=torch.where(mask[:,None],kind,self.piece_type)
        self.preloads_per_robot=torch.where(mask,per_robot,self.preloads_per_robot)

    def _update_match_clock(self, active_mask=None):
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                     else active_mask)
        # REBUILT runs 20 s autonomous followed by 140 s of teleop (2:40 total).
        # Training uses 8,000 x 20 ms steps so the dynamics advance in real time.
        elapsed_next=(self.match_elapsed+self.match_clock_step).clamp(max=160.)
        self.match_elapsed=torch.where(active_mask,elapsed_next,self.match_elapsed)
        remaining_next=(160.-self.match_elapsed).clamp_min(0.)
        self.match_remaining=torch.where(active_mask,remaining_next,self.match_remaining)
        red_auto=self.auto_fuel_scores[:,0]
        blue_auto=self.auto_fuel_scores[:,1]
        first_inactive=torch.where(red_auto>blue_auto,torch.zeros_like(self.hub_inactive_first),
            torch.where(blue_auto>red_auto,torch.ones_like(self.hub_inactive_first),self.hub_inactive_first))
        elapsed=self.match_elapsed
        # The requested training schedule alternates the inactive HUB every
        # 30 s of teleop. Both HUBs remain active during autonomous.
        teleop_elapsed=(elapsed-20.).clamp_min(0.)
        cycle=(teleop_elapsed/30.).floor().long()
        inactive=torch.where((cycle%2)==0,first_inactive,1-first_inactive)
        new_hub_active=torch.ones((self.n,2),device=self.device,dtype=torch.bool)
        new_hub_active.scatter_(1,inactive[:,None],(~(elapsed>=20.))[:,None])
        self.hub_active=torch.where(active_mask[:,None],new_hub_active,self.hub_active)

    def _update_gamepieces(self,active_mask=None):
        """Vectorized pickup/score state; HUB score is a 2D range surrogate.

        REBUILT scoring requires FUEL through a 1.06 m top opening 1.83 m above
        carpet. This planar simulator approximates that event at the HUB
        perimeter and applies official active/inactive timing; it does not model
        launch trajectories or sensor-array passage.
        """
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                     else active_mask)
        if (self.fuel_physics is None and _update_gamepieces_hip is not None and FUSED_GAMEPIECES_HIP_ENABLED and
                _update_gamepieces_hip(self,active_mask)):
            self._clear_intersected_fuel_tracks(active_mask)
            return (self.fuel_acquired_event,self.fuel_scored_event,
                    self.fuel_denied_event)
        return self._update_gamepieces_torch(active_mask)

    def _update_gamepieces_torch(self,active_mask=None):
        """Reference Torch implementation, also used when fused HIP is unavailable."""
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                     else active_mask)
        self._update_match_clock(active_mask)
        for event in (self.fuel_acquired_event,self.fuel_scored_event,
                      self.fuel_denied_event,self.fuel_abandoned_event):
            event.copy_(torch.where(active_mask[:,None],torch.zeros_like(event),event))
        newly_scored=torch.zeros_like(self.piece_active)
        score_origins=torch.zeros_like(self.piece_pos)
        rows=torch.arange(self.n,device=self.device)
        free=self.piece_active&(self.piece_owner<0)
        if self.fuel_physics is not None:
            free &= self.fuel_physics.pos[...,2]<=.20
        # Intake is on the robot's local +X/front side. Its capture band starts
        # just inside the front bumper and extends 0.35 m ahead, across the
        # bumper width plus a small game-piece margin. Rear/side contacts do
        # not acquire pieces. The configured capacity and intake interval bound
        # possession even when the robot remains over a pile.
        for robot in (0,1):
            pose=self.sim.pose[:,robot]
            position=pose[:,:2]
            delta=self.piece_pos-position[:,None,:]
            heading=pose[:,2]
            forward=torch.stack((torch.cos(heading),torch.sin(heading)),-1)
            left=torch.stack((-torch.sin(heading),torch.cos(heading)),-1)
            longitudinal=(delta*forward[:,None,:]).sum(-1)
            lateral=(delta*left[:,None,:]).sum(-1).abs()
            front_edge=self.sim.length[:,robot,None]*.5
            intake_reach=.35
            intake_half_width=self.sim.width[:,robot,None]*.5+.075
            intake=(longitudinal>=front_edge-.075)&(longitudinal<=front_edge+intake_reach)&(lateral<=intake_half_width)
            possession=(self.piece_active&(self.piece_owner==robot)).sum(-1)
            ready=self.match_elapsed+1e-6>=self.next_intake_time[:,robot]
            can_intake=(possession<self.fuel_capacity)&ready&active_mask
            team=self._team_ids_host[robot]
            piece_in_alliance=(self.piece_pos[...,0]<=ALLIANCE_ZONE_DEPTH
                if team==0 else
                self.piece_pos[...,0]>=self.sim.field_length-ALLIANCE_ZONE_DEPTH)
            allowed_zone=torch.where(self.hub_active[:,team,None],
                                     torch.ones_like(piece_in_alliance),
                                     ~piece_in_alliance)
            # Only the nearest eligible piece matters, so squared distance is
            # order preserving and avoids sqrt over every piece each tick.
            distance=delta.square().sum(-1).masked_fill(
                ~(free&allowed_zone&intake&can_intake[:,None]),float("inf"))
            nearest,index=distance.min(-1)
            picked=torch.isfinite(nearest)
            selected_owner=self.piece_owner[rows,index]
            self.piece_owner[rows,index]=torch.where(
                picked,torch.full_like(selected_owner,robot),selected_owner)
            self.fuel_acquired_event[:,robot]=torch.where(
                active_mask,picked.long(),self.fuel_acquired_event[:,robot])
            picked &= active_mask
            self.next_intake_time[:,robot]=torch.where(picked,
                self.match_elapsed+self.intake_interval,self.next_intake_time[:,robot])
            selected_free=free[rows,index]
            free[rows,index]=torch.where(picked,torch.zeros_like(selected_free),selected_free)
        for robot in (0,1):
            held=self.piece_active&(self.piece_owner==robot)&active_mask[:,None]
            self.piece_pos=torch.where(held[...,None],self.sim.pose[:,robot,None,:2],self.piece_pos)
            self.piece_vel=torch.where(held[...,None],self.sim.velocity[:,robot,None,:2],self.piece_vel)
            center=self.hub_centers[robot]
            robot_radius=.5*torch.sqrt(self.sim.length[:,robot].square()+self.sim.width[:,robot].square())
            near_hub=(self.sim.pose[:,robot,:2]-center).norm(dim=-1)<=(.595+robot_radius+.075)
            team=self._team_ids_host[robot]
            robot_x=self.sim.pose[:,robot,0]
            in_alliance_zone=(robot_x<=ALLIANCE_ZONE_DEPTH if team==0 else
                robot_x>=self.sim.field_length-ALLIANCE_ZONE_DEPTH)
            has_fuel=held.any(-1)
            score_intent=((self._last_strategic_action==4) if self.action_mode=="strategic"
                          else torch.ones((self.n,),device=self.device,dtype=torch.bool))
            hub_vector=center-self.sim.pose[:,robot,:2]
            hub_bearing=torch.atan2(hub_vector[:,1],hub_vector[:,0])
            angle_error=torch.atan2(torch.sin(hub_bearing-self.sim.pose[:,robot,2]),
                                    torch.cos(hub_bearing-self.sim.pose[:,robot,2])).abs()
            # A Dumper only releases FUEL while stationary and aimed directly
            # at the HUB. Turret behavior is modeled separately in 3v3.
            stationary=(self.sim.velocity[:,robot,:2].norm(dim=-1)<=.10)&(
                self.sim.velocity[:,robot,2].abs()<=.10)
            aimed=(angle_error<=math.pi/18.)&stationary
            scoring_ready=near_hub&aimed&score_intent
            entering=scoring_ready&~self._last_hub_zone[:,robot]
            denied=entering&has_fuel&(~self.hub_active[:,robot]|~in_alliance_zone)
            denied &= active_mask
            self.fuel_denied_event[:,robot]=torch.where(
                active_mask,denied.long(),self.fuel_denied_event[:,robot])
            self.fuel_denied_count[:,robot]+=denied.long()
            score_ready=self.match_elapsed+1e-6>=self.next_score_time[:,robot]
            eligible=(held&scoring_ready[:,None]&self.hub_active[:,robot,None]&
                in_alliance_zone[:,None]&score_ready[:,None]&active_mask[:,None])
            eligible_distance=torch.arange(self.fuel_count,device=self.device)[None,:].expand(self.n,-1)
            selected=eligible&(eligible_distance==eligible.to(torch.int64).argmax(-1,keepdim=True))
            scored=selected
            newly_scored|=scored
            if self.fuel_physics is not None:
                score_origins=torch.where(scored[...,None],center,score_origins)
            score_count=scored.sum(-1)
            self.fuel_scored_event[:,robot]=torch.where(
                active_mask,score_count,self.fuel_scored_event[:,robot])
            self.fuel_score_count[:,robot]+=score_count
            self.next_score_time[:,robot]=torch.where(score_count>0,
                self.match_elapsed+self.score_interval,self.next_score_time[:,robot])
            in_auto=self.match_elapsed<=20.+self.dt
            self.auto_fuel_scores[:,robot]+=score_count*in_auto.long()
            self.piece_owner=torch.where(scored,-2,self.piece_owner)
            self.piece_zone=torch.where(scored,torch.full_like(self.piece_zone,6),self.piece_zone)
            self.piece_vel=torch.where(scored[...,None],torch.zeros_like(self.piece_vel),self.piece_vel)
            self._last_hub_zone[:,robot]=torch.where(
                active_mask,scoring_ready,self._last_hub_zone[:,robot])
        newly_scored &= active_mask[:,None]
        if self.fuel_physics is None:
            torch.where(newly_scored[...,None],self._midfield_respawn_positions[None],
                        self.piece_pos,out=self.piece_pos)
            self.piece_vel.masked_fill_(newly_scored[...,None],0.)
        else:
            launch_fuel(self,newly_scored,shooter_respawn_targets(self,score_origins),
                        origins=score_origins,height=1.83,flight_time=1.,
                        horizontal_velocity_scale=.1)
        torch.where(newly_scored,torch.full_like(self.piece_owner,-1),
                    self.piece_owner,out=self.piece_owner)
        torch.where(newly_scored,torch.zeros_like(self.piece_zone),
                    self.piece_zone,out=self.piece_zone)
        self._clear_intersected_fuel_tracks(active_mask)
        self.fuel_acquisition_count+=self.fuel_acquired_event*active_mask[:,None]
        self.fuel_abandoned_count+=self.fuel_abandoned_event*active_mask[:,None]
        return self.fuel_acquired_event,self.fuel_scored_event,self.fuel_denied_event
