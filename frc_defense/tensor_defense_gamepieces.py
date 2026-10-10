"""Gamepiece and match lifecycle behavior for the tensor defense environment."""
from __future__ import annotations

from .field import ALLIANCE_ZONE_DEPTH, grid_array_points
from .gamepiece_actions import update_fuel_actions

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None

class TensorDefenseGamepieceMixin:
    """Manage staged FUEL, match timing, pickup, and HUB scoring."""
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

    def _robots_clear_of_bumps(self, positions):
        bumps=self.sim.bump_regions
        if not bumps.numel():
            return torch.ones(positions.shape[:-1],device=self.device,dtype=torch.bool)
        radii=.5*torch.sqrt(self.sim.length.square()+self.sim.width.square())
        dx=(positions[...,None,0]-bumps[None,None,:,0]).abs()
        dy=(positions[...,None,1]-bumps[None,None,:,1]).abs()
        overlaps=((dx<=bumps[None,None,:,2]+radii[:,:,None]) &
                  (dy<=bumps[None,None,:,3]+radii[:,:,None]))
        return ~overlaps.any(-1)

    def _update_gamepieces(self,active_mask=None):
        """Run the canonical offense collect, ferry, and score rules."""
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool)
                     if active_mask is None else active_mask)
        self._update_match_clock(active_mask)
        offense_robot=0
        other_robot=1-offense_robot
        if self.action_mode=="strategic":
            selected=self._last_strategic_action
            own_action=torch.where(selected<4,selected,
                torch.where(selected==4,torch.full_like(selected,4),
                torch.where(selected==6,torch.full_like(selected,6),
                            torch.full_like(selected,7))))
        else:
            possession=(self.piece_active &
                        (self.piece_owner==offense_robot)).sum(-1)
            own_action=torch.where(possession>0,
                torch.where(self.hub_active[:,self.team_ids[offense_robot]],
                            torch.full_like(possession,4),
                            torch.full_like(possession,6)),
                torch.zeros_like(possession))
        actions=torch.full((self.n,2),7,device=self.device,dtype=torch.long)
        actions[:,offense_robot]=own_action
        other_count=(self.piece_active & (self.piece_owner==other_robot)).sum(-1)
        actions[:,other_robot]=torch.where(other_count>0,
            torch.where(self.hub_active[:,self.team_ids[other_robot]],
                        torch.full_like(other_count,4),
                        torch.full_like(other_count,6)),
            torch.zeros_like(other_count))
        self.last_actions.copy_(torch.where(active_mask[:,None],actions,self.last_actions))
        self._intake_collecting.copy_(torch.where(
            active_mask[:,None],self.last_actions<4,self._intake_collecting))
        score_intent=self.last_actions==4
        return update_fuel_actions(self,active_mask,score_intent)
