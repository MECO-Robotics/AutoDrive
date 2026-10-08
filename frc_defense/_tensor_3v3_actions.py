"""Action target selection and robot contention behavior for the 3v3 env."""
from __future__ import annotations

import math
import torch
from . import tensor_collision_multi_hip as _collision_multi_hip
from . import tensor_3v3_pickup_grid_hip as _pickup_grid_hip


class TensorThreeVsThreeActionMixin:
    """Methods that translate strategic actions into robot targets."""

    def _prepare_score_grid(self):
        """Cache fixed-field score points and their chassis-clearance mask."""
        robot_radius=.5*torch.sqrt(self.sim.length.square()+self.sim.width.square())
        red_min=robot_radius
        red_max=(self.alliance_zone_depth-robot_radius).clamp_min(0.)
        blue_min=self.field_length-self.alliance_zone_depth+robot_radius
        blue_max=torch.full_like(blue_min,self.field_length)-robot_radius
        x_min=torch.where(self.team_ids[None,:]==0,red_min,blue_min)
        x_max=torch.where(self.team_ids[None,:]==0,red_max,blue_max)
        boxes=torch.cat((self.planner._boxes,self.planner._bumps,
                         self.sim.bump_regions),0)
        self._score_grid_robot_radius=robot_radius
        self._score_grid_x_min=x_min
        self._score_grid_x_max=x_max
        self._score_grid_boxes=boxes
        if boxes.numel()==0:
            self._score_grid_points=torch.empty(
                (self.n,self.team_ids.numel(),0,2),device=self.device)
            self._score_grid_clear=torch.empty(
                (self.n,self.team_ids.numel(),0),device=self.device,dtype=torch.bool)
            return
        fractions=self._score_x_fractions
        y_fractions=self._score_y_fractions
        grid_x=x_min[...,None]+(x_max-x_min)[...,None]*fractions
        grid_y=robot_radius[...,None]+(
            self.sim.field_width-2.*robot_radius)[...,None]*y_fractions
        grid_x=grid_x[..., :,None].expand(-1,-1,-1,y_fractions.numel())
        grid_y=grid_y[...,None,:].expand(-1,-1,fractions.numel(),-1)
        points=torch.stack((grid_x,grid_y),-1).reshape(
            self.n,self.team_ids.numel(),-1,2)
        box_x=boxes[None,None,:,0]
        box_y=boxes[None,None,:,1]
        box_hx=boxes[None,None,:,2]
        box_hy=boxes[None,None,:,3]
        dx=(points[...,None,0]-box_x[:,:,None,:]).abs()
        dy=(points[...,None,1]-box_y[:,:,None,:]).abs()
        overlap=((dx<=box_hx[:,:,None,:]+robot_radius[...,None,None]+.15) &
                 (dy<=box_hy[:,:,None,:]+robot_radius[...,None,None]+.15))
        self._score_grid_points=points
        self._score_grid_clear=~overlap.any(-1)

    def _target_for_actions(self, actions, active):
        p = self.sim.pose
        defer_initial_candidates=(self._has_deterministic_offense and
                                  all(mode=="deterministic"
                                      for mode in self.control_modes))
        if defer_initial_candidates:
            indices=valid=None
            if (_pickup_grid_hip.integrated_available(self.device) and
                    all(mode!="none" for mode in self.control_modes)):
                possession=_pickup_grid_hip.possession_counts(self.piece_owner)
            else:
                possession=torch.stack(
                    [(self.piece_owner==robot).sum(-1) for robot in range(6)],dim=1)
        else:
            indices, valid, _ = self._candidates()
            masks, possession = self._action_mask(indices, valid)
        robot_hub_active = self.hub_active[:, self.team_ids]
        hub=self.hub_centers[self.team_ids][None].expand(self.n,-1,-1)
        intake_indices, intake_eligible = indices, valid
        intake_available = valid.any(-1) if valid is not None else None
        if self._has_deterministic_offense:
            # Collect fuel across the field. The inactive-HUB rule excludes
            # only fuel in this robot's friendly alliance zone.
            piece_in_alliance = torch.where(
                self.team_ids[None, :, None] == 0,
                self.track_pos[..., 0] <= self.alliance_zone_depth,
                self.track_pos[..., 0] >= self.field_length-self.alliance_zone_depth)
            shift_collection_zone = torch.where(
                robot_hub_active[..., None],torch.ones_like(piece_in_alliance),
                ~piece_in_alliance)
            offense_mode=(self._deterministic_mode_mask &
                          ~self._defense_role_mask)[None]
            strategy_eligible=shift_collection_zone
            # Remember fuel across brief occlusions, with an age penalty in
            # ranking. Pickup legality still comes from live free-fuel tracks.
            live_tracks=(self.track_mask &
                         (self.track_age <= self.perception_track_timeout))
            intake_forward=torch.stack((torch.cos(p[:,:,2]),
                                        torch.sin(p[:,:,2])),-1)
            intake_right=torch.stack((torch.sin(p[:,:,2]),
                                      -torch.cos(p[:,:,2])),-1)
            intake_reference=(p[:,:,:2] +
                intake_forward*(self.sim.length*.5+.15)[...,None] +
                intake_right*(self.sim.width*.5)[...,None])
            eligible_tracks=live_tracks & torch.where(
                offense_mode[...,None],strategy_eligible,
                torch.ones_like(strategy_eligible))
            # Availability is based on every currently known fuel track, not
            # just the four nearest targets exposed as intake actions.
            intake_available=eligible_tracks.any(-1)
            intake_tracks=eligible_tracks
            intake_distance=(self.track_pos-intake_reference[:,:,None,:]).norm(dim=-1)
            intake_distance+=self.track_age.clamp(0.,2.)*.4
            nearest,intake_indices=intake_distance.masked_fill(
                ~intake_tracks,float("inf")).topk(4,dim=-1,largest=False)
            intake_eligible=torch.isfinite(nearest)
            if defer_initial_candidates:
                indices,valid=intake_indices,intake_eligible
            else:
                masks[..., :4] = intake_eligible & (
                    possession[..., None] < self.robot_fuel_capacities[None, :, None])
        candidate_points = self.track_pos.gather(
            2, intake_indices[..., None].expand(-1, -1, -1, 2))
        # A deterministic defender holds the opponent HUB approach even when
        # the attacker is temporarily occluded; giving up the post lets a
        # scorer approach freely during perception dropouts.
        defense_action = torch.full_like(possession,5)
        if self._has_deterministic_offense:
            # Active hubs reward steady scoring. Send partial loads so the
            # dumper spends less time carrying fuel across long return routes.
            active_batch_limit=torch.ceil(
                self.robot_fuel_capacities[None].to(torch.float32)*.5).long()
            active_batch_limit=active_batch_limit.expand_as(possession)
            batch_limit=torch.where(
                robot_hub_active,active_batch_limit,
                self.robot_fuel_capacities[None].expand_as(possession))
            hub_bearing=torch.atan2(hub[...,1]-p[...,1],hub[...,0]-p[...,0])
            hub_heading_error=torch.atan2(
                torch.sin(hub_bearing-p[...,2]),torch.cos(hub_bearing-p[...,2])).abs()
            bump_clear=self._robots_clear_of_bumps(p[:,:,:2])
            gather_more = (possession < batch_limit) & intake_available
            score_commit = (robot_hub_active & (possession>0) &
                (self._score_committed |
                 (possession>=batch_limit) | ~intake_available |
                 (nearest[...,0]>2.5)))
            self._score_committed.copy_(torch.where(
                active[:,None],score_commit,self._score_committed))
            # All robot types fill to capacity when an inactive HUB requires
            # ferrying. Ferry early only when no legal pickup target remains.
            ferry_batch_limit=self.robot_fuel_capacities[None].expand_as(possession)
            begin_ferry=(possession>=ferry_batch_limit)|~intake_available
            ferry_committed=(~robot_hub_active & (possession>0) &
                (self._ferry_committed|begin_ferry))
            self._ferry_committed.copy_(torch.where(
                active[:,None],ferry_committed,self._ferry_committed))
            ferry_inactive_fuel=(~robot_hub_active & (possession > 0) & ferry_committed)
            # Settle dumpers before ferrying so carried fuel stays inside the
            # robot while its chassis turns toward the alliance zone.
            dumper_speed=self.sim.velocity[:,:,:2].norm(dim=-1)
            dumper_braking_for_ferry=(ferry_inactive_fuel &
                self._dumper_mask[None] & (dumper_speed>.15))
            offense_action=torch.where(ferry_inactive_fuel,
                torch.where(dumper_braking_for_ferry,
                    torch.full_like(possession,7),torch.full_like(possession,6)),
                torch.where(score_commit,torch.full_like(possession,4),
                torch.where(gather_more,torch.zeros_like(possession),
                            torch.full_like(possession,6))))
            # Turrets can fire while following their collection route. Their
            # strategy keeps collecting whenever legal fuel is available and
            # raises scoring intent independently of the movement action.
            active_turret_collect=(robot_hub_active & self._turret_mask[None] &
                                   intake_available &
                                   (possession<self.robot_fuel_capacities[None]))
            offense_action=torch.where(active_turret_collect,
                                       torch.zeros_like(possession),offense_action)
            deterministic=torch.where(self._defense_role_mask[None],defense_action,
                                      offense_action)
        else:
            # With no deterministic offense controller, all deterministic
            # controllers in this batch are defenders and use action 5.
            deterministic=defense_action
        if defer_initial_candidates:
            # This branch constructs only legal deterministic actions: pickup
            # requires an eligible intake track, score requires possession,
            # and defense is restricted to defensive robot slots.
            action=deterministic
        else:
            action = torch.as_tensor(actions, device=self.device, dtype=torch.long).reshape(self.n, 6)
            action = torch.where(self._nn_mode_mask[None], action, torch.where(
                self._deterministic_mode_mask[None],deterministic,torch.full_like(action,7)))
            action = action.clamp(0, 7)
            action = torch.where(masks.gather(-1, action[...,None]).squeeze(-1), action,
                                 masks.to(torch.int64).argmax(-1))
        self.last_actions.copy_(torch.where(active[:,None],action,self.last_actions))
        score_intent = action == 4
        if self._has_deterministic_offense:
            deterministic_offense_mode=(self._deterministic_mode_mask &
                                         ~self._defense_role_mask)[None]
            active_turret_scoring=(deterministic_offense_mode & self._turret_mask[None] &
                                   robot_hub_active & (possession>0) &
                                   ~ferry_inactive_fuel)
            score_intent |= active_turret_scoring
        self._last_score_intent.copy_(torch.where(
            active[:,None],score_intent,self._last_score_intent))
        rows = self._world_indices[:,None]
        selected = indices.gather(2,action.clamp(max=3)[...,None]).squeeze(-1)
        deterministic_offense = (self._deterministic_mode_mask &
                                 ~self._defense_role_mask)[None]
        deterministic_robot = self._deterministic_mode_mask[None]
        target_based_approach=(not self.sweeping_enabled and
            deterministic_offense & ((action<4)|((action==6)&~(possession>0))))
        locked_index=self._target_fuel_index.clamp(0,self.fuel_count-1)
        # Rank deterministic intake targets by distance from the robot.
        if self._has_deterministic_offense:
            intake_reference=(p[:,:,:2] +
                torch.stack((torch.cos(p[:,:,2]),torch.sin(p[:,:,2])),-1)*
                    (self.sim.length*.5+.15)[...,None] +
                torch.stack((torch.sin(p[:,:,2]),-torch.cos(p[:,:,2])),-1)*
                    (self.sim.width*.5)[...,None])
            local_dist = (candidate_points - intake_reference[:,:,None,:]).norm(dim=-1)
            candidate_age=self.track_age.gather(2,intake_indices).clamp(0.,2.)
            intake_cost=local_dist+candidate_age*.4
            staging_cost = intake_cost
            staging_idx = staging_cost.masked_fill(~intake_eligible,float("inf")).argmin(-1)
            stage_selected = intake_indices.gather(2,staging_idx[...,None]).squeeze(-1)
            stage_mask = deterministic_offense & ~robot_hub_active & intake_available
            selected = torch.where(stage_mask,stage_selected,selected)
            # Keep later teammates from selecting destinations inside an
            # earlier teammate's inscribed chassis circle. This reserves a
            # useful patch of collection area, rather than only the exact
            # selected fuel track.
            candidate_cost = torch.where(
                (~robot_hub_active)[..., None], staging_cost, intake_cost)
            assigned = []
            for robot in range(self.team_ids.numel()):
                wants_fuel = deterministic_offense[:, robot] & (action[:, robot] < 4)
                available = intake_eligible[:, robot].clone()
                for prior_robot, prior_index, prior_active in assigned:
                    same_alliance = self.team_ids[robot] == self.team_ids[prior_robot]
                    destination = self.track_pos[
                        self._world_indices,prior_robot,
                        prior_index.clamp(0,self.fuel_count-1)]
                    destination_radius=.5*torch.minimum(
                        self.sim.length[:,prior_robot],self.sim.width[:,prior_robot])
                    occupied_area=((candidate_points[:,robot]-destination[:,None,:])
                                   .norm(dim=-1) < destination_radius[:,None])
                    reserved=(prior_active & wants_fuel & same_alliance)[:,None]
                    available &= ~(occupied_area & reserved)
                costs = candidate_cost[:, robot].masked_fill(~available, float("inf"))
                rank = costs.argmin(-1)
                has_assignment = torch.isfinite(costs.gather(1, rank[:, None]).squeeze(1))
                proposed = intake_indices[:, robot].gather(1, rank[:, None]).squeeze(1)
                locked_candidate=((intake_indices[:,robot]==locked_index[:,robot,None]) &
                                  available)
                has_locked=locked_candidate.any(-1)
                proposed=torch.where(has_locked,locked_index[:,robot],proposed)
                selected[:, robot] = torch.where(
                    wants_fuel & has_assignment, proposed, selected[:, robot])
                # Later robots only write their own columns, so this view of
                # the completed prior slot stays stable without a clone.
                assigned.append((robot, selected[:, robot],
                                 wants_fuel & has_assignment))
        has_selected=target_based_approach & intake_available
        next_target=torch.where(has_selected,selected,torch.full_like(selected,-1))
        target_changed=has_selected&(next_target!=self._target_fuel_index)
        self._target_fuel_index.copy_(torch.where(
            active[:,None],next_target,self._target_fuel_index))
        next_selected_step=torch.where(target_changed,self.steps[:,None],
            torch.where(has_selected,self._target_fuel_selected_step,
                        torch.full_like(self._target_fuel_selected_step,-1)))
        self._target_fuel_selected_step.copy_(torch.where(
            active[:,None],next_selected_step,self._target_fuel_selected_step))
        fuel_target = self.track_pos.gather(
            2,selected[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        self._last_audit_cluster_count=torch.zeros_like(selected)
        possession_now = possession > 0
        hub = self.hub_centers[self.team_ids][None].expand(self.n,-1,-1)
        # Score from the nearest point inside the alliance zone. Robots already
        # in the zone stay put; dumpers turn directly toward the HUB center.
        robot_radius=self._score_grid_robot_radius
        x_min=self._score_grid_x_min
        x_max=self._score_grid_x_max
        score_target=p[:,:,:2].clone()
        score_target[...,0]=torch.maximum(x_min,torch.minimum(score_target[...,0],x_max))
        score_target[...,1]=torch.maximum(
            robot_radius,torch.minimum(score_target[...,1],self.sim.field_width-robot_radius))
        # The release gate checks simulator bump regions directly. Include
        # those same rectangles here so the selected staging point can satisfy
        # that gate instead of stopping at a target that still overlaps a bump.
        boxes=self._score_grid_boxes
        fused_score_target=_collision_multi_hip.safe_score_targets(
            self.sim,score_target,robot_radius,x_min,x_max,boxes,
            self._score_grid_points,self._score_grid_clear)
        if fused_score_target is not None:
            score_target=fused_score_target
        elif boxes.numel():
            box_x=boxes[None,None,:,0]
            box_y=boxes[None,None,:,1]
            box_hx=boxes[None,None,:,2]
            box_hy=boxes[None,None,:,3]
            radius=robot_radius[...,None]
            left=score_target[:,:,None,:].expand(-1,-1,boxes.shape[0],-1).clone()
            right=left.clone()
            lower=left.clone()
            upper=left.clone()
            # Leave room for stopping error and the sim's rectangular bump
            # clearance check around the robot's center point.
            clearance=.40
            left[...,0]=box_x-box_hx-radius-clearance
            right[...,0]=box_x+box_hx+radius+clearance
            lower[...,1]=box_y-box_hy-radius-clearance
            upper[...,1]=box_y+box_hy+radius+clearance
            candidates=torch.cat((score_target[:,:,None,:],left,right,lower,upper),2)
            candidates[...,0]=torch.maximum(
                x_min[...,None],torch.minimum(candidates[...,0],x_max[...,None]))
            candidates[...,1]=candidates[...,1].clamp_min(robot_radius[...,None])
            candidates[...,1]=torch.minimum(
                candidates[...,1],self.sim.field_width-robot_radius[...,None])
            # If the nearest pose is pinned beside a BUMP, point the scorer
            # deeper into its alliance zone. A pose that barely clears a
            # BUMP on paper can remain physically wedged against it and never
            # satisfy the release gate.
            side_count=candidates.shape[2]
            dx=(candidates[...,None,0]-box_x[:,:,None,:]).abs()
            dy=(candidates[...,None,1]-box_y[:,:,None,:]).abs()
            overlap=((dx<=box_hx[:,:,None,:]+robot_radius[...,None,None]+.15) &
                     (dy<=box_hy[:,:,None,:]+robot_radius[...,None,None]+.15))
            clear=~overlap.any(-1)
            side_distance=(candidates-p[:,:,None,:2]).norm(dim=-1)
            side_distance=side_distance.masked_fill(~clear,float('inf'))
            grid_distance=(self._score_grid_points-p[:,:,None,:2]).norm(dim=-1)
            grid_distance=grid_distance.masked_fill(~self._score_grid_clear,float('inf'))
            distance=torch.cat((side_distance,grid_distance),2)
            nearest=distance.argmin(-1)
            has_clear=clear.any(-1)
            has_clear |= self._score_grid_clear.any(-1)
            side_target=candidates.gather(
                2,nearest.clamp(max=side_count-1)[...,None,None].expand(-1,-1,1,2)).squeeze(2)
            grid_index=(nearest-side_count).clamp(
                min=0,max=self._score_grid_points.shape[2]-1)
            grid_target=self._score_grid_points.gather(
                2,grid_index[...,None,None].expand(-1,-1,1,2)).squeeze(2)
            safe_target=torch.where((nearest<side_count)[...,None],side_target,grid_target)
            score_target=torch.where(has_clear[...,None],safe_target,score_target)
        enemy_mask=self.team_ids[:,None] != self.team_ids[None,:]
        enemy_delta=p[:,None,:,:2]-p[:,:,None,:2]
        enemy_dist=enemy_delta.norm(dim=-1).masked_fill(~enemy_mask[None],float("inf"))
        nearest_enemy=enemy_dist.argmin(-1)
        nearest_enemy_pose=p.gather(1,nearest_enemy[...,None].expand(-1,-1,3))
        # Defend the perceived attacker's scoring lane around the HUB. When
        # that opponent is temporarily occluded, keep a post on the standard
        # alliance-side approach rather than giving up the HUB.
        opponent_team=1-self.team_ids
        opponent_hub=self.hub_centers[opponent_team][None]
        approach_side=torch.where(opponent_team[None]==0,-1.,1.)
        defender_radius=.5*torch.sqrt(self.sim.length.square()+self.sim.width.square())
        # Post just outside the HUB footprint, rather than at the outer edge
        # of the scoring radius. This puts the chassis in the attacker's
        # preferred scoring ring while leaving clearance for the HUB collider.
        scoring_standoff=(.595+defender_radius-.15).clamp_min(.85)
        standard_post=opponent_hub.expand(self.n,-1,-1).clone()
        standard_post[...,0]+=approach_side*scoring_standoff
        tracked_enemy=self.opponent_pose[...,:2]
        tracked_velocity=self.opponent_velocity[...,:2]
        enemy_speed=tracked_velocity.norm(dim=-1,keepdim=True)
        fallback_direction=(p[:,:,:2]-tracked_enemy)
        fallback_direction=fallback_direction/fallback_direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        travel_direction=torch.where(enemy_speed>.15,
            tracked_velocity/enemy_speed.clamp_min(.15),fallback_direction)
        intercept_time=((tracked_enemy-p[:,:,:2]).norm(dim=-1,keepdim=True)/
                        self.sim.speed[:,:,None].clamp_min(.1)).clamp(0.,1.25)
        intercept_target=(tracked_enemy+tracked_velocity*intercept_time+
                         travel_direction*.55)
        # Stay on the attacker's body/drive line throughout the match. A fixed
        # HUB post protects only one approach; a scorer can turn around it and
        # reach the scoring ring. Predict an intercept from the observed
        # velocity so AD* closes on the robot and uses chassis contact to deny
        # its route. Keep the HUB post only as a perception-loss fallback.
        guarded_target=intercept_target
        guarded_score_pose=torch.where(self.opponent_valid[...,None],guarded_target,
                                       standard_post)
        defense_intercept=guarded_score_pose
        intercept=torch.where(self._defense_role_mask[None,:,None],
                              defense_intercept,nearest_enemy_pose[...,:2])
        # Empty robots follow their indexed spline raster through midfield.
        ferry_target=self.ferry_targets[None].expand(self.n,-1,-1)
        dumper_ferry=(self._dumper_mask[None]&possession_now&~robot_hub_active)
        action_six_target=torch.where(dumper_ferry[...,None],p[:,:,:2],ferry_target)
        deterministic_offense=(self._deterministic_mode_mask &
                                ~self._defense_role_mask)[None]
        sweeping=(self.sweeping_enabled & deterministic_offense & ((action<4)|
                   ((action==6)&~possession_now)))
        _, raster_target, tangent_vector=self._raster_waypoint_targets(p[:,:,:2])
        # The intake follows the direction of travel on the raster spline.
        fuel_delta = fuel_target - p[:, :, :2]
        fuel_delta=torch.where(sweeping[...,None],tangent_vector,fuel_delta)
        target=torch.where(sweeping[...,None],raster_target,
            torch.where((action<4)[...,None],fuel_target,
            torch.where((action==4)[...,None],score_target,
            torch.where((action==5)[...,None],intercept,
            torch.where((action==6)[...,None],action_six_target,p[:,:,:2])))))
        if not self.sweeping_enabled and self._has_deterministic_offense:
            target_based_collect = deterministic_offense & (
                (action < 4) | ((action == 6) & ~possession_now))
            candidate_reference=intake_reference
            candidate_distance=(candidate_points-candidate_reference[:,:,None,:]).norm(dim=-1)
            cluster_count=(intake_eligible & (candidate_distance<1.7)).sum(-1)
            self._last_audit_cluster_count=cluster_count
            nearest_distance=candidate_distance.masked_fill(
                ~intake_eligible,float("inf")).amin(-1)
            at_cluster=(cluster_count>=2)&(nearest_distance<=1.25)
            # Depot tracks can satisfy the generic cluster threshold before
            # the chassis is lined up for the intake. Delay corridor collection
            # until it has reached a usable depot approach pose.
            depot_target=torch.zeros_like(at_cluster)
            for box in self.field_boxes:
                if box.name.endswith("_depot"):
                    depot_target |= (
                        (fuel_target[...,0]-box.x).abs()<=box.length/2) & (
                        (fuel_target[...,1]-box.y).abs()<=box.width/2)
            fuel_bearing=torch.atan2(fuel_delta[...,1],fuel_delta[...,0])
            fuel_heading_error=torch.atan2(
                torch.sin(fuel_bearing-p[:,:,2]),
                torch.cos(fuel_bearing-p[:,:,2])).abs()
            depot_approach_ready=(fuel_delta.norm(dim=-1)<=1.15) & (
                fuel_heading_error<=.35)
            at_cluster &= ~depot_target | depot_approach_ready
            current_cluster=(cluster_count>=2)&(candidate_distance.masked_fill(
                ~intake_eligible,float("inf")).amin(-1)<=2.4)
            cluster_members=intake_eligible & (candidate_distance<1.7)
            cluster_center=torch.where(
                cluster_members[...,None],candidate_points,
                torch.zeros_like(candidate_points)).sum(-2)
            cluster_center=cluster_center/cluster_count[...,None].clamp_min(1)
            old_collecting=self._target_collecting
            empty_ticks=torch.where(old_collecting & ~current_cluster,
                self._target_cluster_empty_ticks+1,
                torch.zeros_like(self._target_cluster_empty_ticks))
            cluster_grace_ticks=max(1,int(math.ceil(.4/self.dt)))
            still_collecting=old_collecting & (empty_ticks<cluster_grace_ticks)
            next_collecting=torch.where(old_collecting,still_collecting,at_cluster)
            next_collecting=torch.where(target_based_collect,next_collecting,
                                        torch.zeros_like(next_collecting))
            entering_cluster=next_collecting & ~old_collecting & target_based_collect
            # Enter a depot/field cluster on the line to its selected fuel.
            # Latching the old travel heading can make the chassis sweep past
            # a dense cluster with its intake pointed sideways.
            target_distance=fuel_delta.norm(dim=-1)
            target_heading=torch.atan2(fuel_delta[...,1],fuel_delta[...,0])
            course_heading=torch.where(target_distance>.1,target_heading,p[:,:,2])
            self._target_collect_heading.copy_(torch.where(
                active[:,None]&entering_cluster,course_heading,
                self._target_collect_heading))
            self._target_turn_anchor.copy_(torch.where(
                (active[:,None]&entering_cluster)[...,None],p[:,:,:2],
                self._target_turn_anchor))
            self._target_collecting.copy_(torch.where(
                active[:,None],next_collecting,self._target_collecting))
            self._target_cluster_empty_ticks.copy_(torch.where(
                active[:,None],empty_ticks,self._target_cluster_empty_ticks))

            # Approach mode heads straight to the closest visible FUEL. With
            # no legal target, move toward the open field interior so AD* can
            # route around field elements while perception acquires new fuel.
            target_based_search=(target_based_collect & ~self._target_collecting &
                                 (action==6))
            search_target=torch.zeros_like(p[:,:,:2])
            search_target[...,0]=self.field_length*.5
            search_target[...,1]=self.sim.field_width*.5
            target=torch.where(target_based_search[...,None],search_target,target)

            # Trace the local fuel-cloud edge with the intake's right side.
            # A smoothed density gradient estimates the inward normal; its
            # perpendicular gives a tangent that adapts to irregular shapes.
            collecting=target_based_collect & self._target_collecting
            course=self._target_collect_heading
            forward=torch.stack((torch.cos(course),torch.sin(course)),-1)
            intake_right=torch.stack((forward[...,1],-forward[...,0]),-1)
            boundary_reference=(p[:,:,:2] + forward*(self.sim.length*.5+.15)[...,None]
                                + intake_right*(self.sim.width*.5)[...,None])
            cloud_delta=self.track_pos-boundary_reference[:,:,None,:]
            cloud_distance=cloud_delta.norm(dim=-1)
            # A wider-than-intake kernel blends neighboring edge normals so
            # the traced tangent rounds corners instead of following noise.
            sigma=torch.maximum(self.sim.width*1.5,
                                torch.full_like(self.sim.width,.7))[...,None]
            cloud_weight=torch.exp(-.5*(cloud_distance/sigma).square())
            cloud_weight*=torch.exp(-self.track_age.clamp_min(0.)*.2)
            cloud_weight*=intake_tracks.to(cloud_weight.dtype)
            inward=(cloud_delta*cloud_weight[...,None]).sum(-2)
            inward_norm=inward.norm(dim=-1,keepdim=True)
            gradient_valid=inward_norm.squeeze(-1)>.15
            center_inward=cluster_center-boundary_reference
            center_inward_norm=center_inward.norm(dim=-1,keepdim=True)
            center_valid=(cluster_count>=2)&(center_inward_norm.squeeze(-1)>.15)
            normal=inward/inward_norm.clamp_min(1e-6)
            center_normal=center_inward/center_inward_norm.clamp_min(1e-6)
            normal=torch.where(gradient_valid[...,None],normal,center_normal)
            tangent_a=torch.stack((-normal[...,1],normal[...,0]),-1)
            tangent_b=-tangent_a
            # Keep the contour direction that best continues the current
            # travel, avoiding a reversal at a rounded cluster corner.
            tangent=torch.where(((tangent_a*forward).sum(-1)>=0.)[...,None],
                                tangent_a,tangent_b)
            tangent_heading=torch.atan2(tangent[...,1],tangent[...,0])
            angle_delta=torch.atan2(torch.sin(tangent_heading-course),
                                    torch.cos(tangent_heading-course))
            # The density normal is noisy at the edge of sparse clusters.
            # Limit contour-course changes below chassis turn rate so a
            # single changing neighbor cannot spin the collector in place.
            turn_limit=self.sim.omega_limit*.5*self.dt
            next_course=course+angle_delta.clamp(-turn_limit,turn_limit)
            # Trace only when the smoothed density gradient gives a stable
            # cluster edge; otherwise keep the reachable direct pickup target.
            edge_trace=collecting & (gradient_valid|center_valid)
            trace_forward=torch.stack((torch.cos(next_course),torch.sin(next_course)),-1)
            radius=.5*torch.sqrt(self.sim.length.square()+self.sim.width.square())
            trace_target=p[:,:,:2]+trace_forward*3.0
            trace_inside=(
                (trace_target[...,0]>=radius) &
                (trace_target[...,0]<=self.field_length-radius) &
                (trace_target[...,1]>=radius) &
                (trace_target[...,1]<=self.sim.field_width-radius))
            # A clamped tangent can point along a wall into an unrelated
            # corner. Use the selected-fuel approach for that step and resume
            # contour tracing when its lookahead stays inside the field.
            edge_trace &= trace_inside
            self._target_collect_heading.copy_(torch.where(
                active[:,None]&edge_trace,next_course,self._target_collect_heading))
            target=torch.where(edge_trace[...,None],trace_target,target)
            fuel_delta=torch.where(edge_trace[...,None],trace_forward,fuel_delta)
            # Aim the chassis center just before the ball, leaving it in the
            # front intake strip when the route reaches its zero-speed end.
            # Driving through the ball center sends depot pickups into the
            # depot wall and leaves the ball behind the intake.
            approach = fuel_target - p[:, :, :2]
            approach = approach / approach.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            pass_target = fuel_target - approach * .45
            radius = .5 * torch.sqrt(self.sim.length.square() + self.sim.width.square())
            pass_target[..., 0].clamp_(radius, self.field_length - radius)
            pass_target[..., 1].clamp_(radius, self.sim.field_width - radius)
            depots={box.name:box for box in self.field_boxes
                    if box.name in ("red_depot","blue_depot")}
            for team,depot_name in ((0,"red_depot"),(1,"blue_depot")):
                depot=depots.get(depot_name)
                if depot is None:
                    continue
                in_depot=((fuel_target[...,0]-depot.x).abs()<=depot.length/2) & (
                    (fuel_target[...,1]-depot.y).abs()<=depot.width/2)
                team_mask=self.team_ids[None,:]==team
                half_length=self.sim.length*.5
                half_width=self.sim.width*.5
                xmin=half_length
                xmax=self.field_length-half_length
                ymin=half_width
                ymax=self.sim.field_width-half_width
                # Depots are open floor in the planar model. Let the chassis
                # straddle the depot footprint slightly so balls near its
                # back edge stay inside the measured front-intake reach; the
                # fully exterior pose can be just beyond reach and forces a
                # long route around an open end.
                side_x=(depot.x+depot.length/2+half_length-.20 if team==0 else
                        depot.x-depot.length/2-half_length+.20)
                side_y_low=torch.full_like(side_x,depot.y-depot.width/2-.015)-half_width
                side_y_high=torch.full_like(side_x,depot.y+depot.width/2+.015)+half_width
                xside=torch.stack((
                    torch.stack((side_x,
                        fuel_target[...,1].clamp(ymin,ymax)),-1),
                    torch.stack((fuel_target[...,0].clamp(xmin,xmax),
                        side_y_low),-1),
                    torch.stack((fuel_target[...,0].clamp(xmin,xmax),
                        side_y_high),-1)
                ),2)
                # A depot ball can be reachable through its field side or
                # either open end. Choose the nearest chassis pose that puts
                # the ball inside the front intake footprint.
                offsets=fuel_target[:,:,None,:]-xside
                along=torch.where(
                    torch.arange(3,device=self.device)[None,None,:]==0,
                    offsets[...,0].abs(),offsets[...,1].abs())
                across=torch.where(
                    torch.arange(3,device=self.device)[None,None,:]==0,
                    offsets[...,1].abs(),offsets[...,0].abs())
                reach_along=half_length[...,None]+.35+.083
                reach_across=half_width[...,None]+.075+.083
                reachable=(along<=reach_along)&(across<=reach_across)
                reachable &= (xside[...,0]>=xmin[...,None]) & (xside[...,0]<=xmax[...,None])
                reachable &= (xside[...,1]>=ymin[...,None]) & (xside[...,1]<=ymax[...,None])
                side_distance=(xside-p[:,:,None,:2]).norm(dim=-1)
                side_index=side_distance.masked_fill(~reachable,float("inf")).argmin(-1)
                side_target=xside.gather(2,side_index[...,None,None].expand(-1,-1,1,2)).squeeze(2)
                has_reachable=reachable.any(-1)
                use_depot_side=in_depot & team_mask & has_reachable
                pass_target=torch.where(use_depot_side[...,None],side_target,pass_target)
            approaching_fuel=(target_based_collect & (action<4) &
                              (~collecting | (depot_target & ~depot_approach_ready)))
            target = torch.where(approaching_fuel[..., None], pass_target, target)
        # These are read-only snapshots; sharing their storage avoids two
        # full tensor copies per tick (CUDA graph replays update in place).
        self._last_audit_targets=target.detach()
        self._last_audit_fuel_targets=fuel_target.detach()
        return target, action, possession_now, fuel_delta

    def _avoid_robot_contention(self, command, targets, active=None):
        """Bend commanded motion around robots on near-term collision courses."""
        if active is not None:
            fused = _collision_multi_hip.avoid_robot_contention(
                self.sim, command, targets, active, self._avoidance_winner,
                self._controlled_mode_mask,self.teammate_intent_knowledge)
            if fused is not None:
                return fused
        controlled=self._controlled_mode_mask[None]
        command=torch.where(controlled[...,None],command,torch.zeros_like(command))
        pair_i, pair_j = self.sim._collision_pair_i, self.sim._collision_pair_j
        position = self.sim.pose[:, :, :2]
        pos_i, pos_j = position[:, pair_i], position[:, pair_j]
        cmd_i, cmd_j = command[:, pair_i], command[:, pair_j]
        relative = pos_i - pos_j
        relative_velocity = cmd_i - cmd_j
        closing_speed_sq = relative_velocity.square().sum(-1)
        time = (-(relative * relative_velocity).sum(-1) /
                closing_speed_sq.clamp_min(1e-6)).clamp(0., .7)
        closest = relative + relative_velocity * time[..., None]
        closest_distance = closest.norm(dim=-1)
        current_distance = relative.norm(dim=-1)
        radii = .5 * torch.sqrt(self.sim.length.square() + self.sim.width.square())
        clearance = radii[:, pair_i] + radii[:, pair_j] + .25
        conflict = (((closing_speed_sq > .0225) &
                     (closest_distance < clearance)) |
                    (current_distance < clearance - .08))

        goal_distance = (targets - position).norm(dim=-1)
        distance_i, distance_j = goal_distance[:, pair_i], goal_distance[:, pair_j]
        teammate_pair=(self.sim.team_ids[pair_i]==self.sim.team_ids[pair_j])
        use_intent=teammate_pair & self.teammate_intent_knowledge
        if not self.teammate_intent_knowledge:
            # Teammates remain hard AD* and physics obstacles, but with intent
            # knowledge disabled, only react to close physical proximity.
            near_teammate=current_distance < (clearance + .05)
            conflict=torch.where(teammate_pair[None],near_teammate,conflict)
        # Keep right of way fixed for the whole encounter. Recomputing this
        # choice every frame makes robots swap who yields as their target
        # distances fluctuate, which can trap both in a sidestep loop.
        intent_winner = torch.where(
            distance_i <= distance_j, pair_i[None], pair_j[None])
        stable_winner=torch.minimum(pair_i,pair_j)[None].expand(self.n,-1)
        preferred_winner=torch.where(use_intent[None],intent_winner,stable_winner)
        i_controlled=self._controlled_mode_mask[pair_i]
        j_controlled=self._controlled_mode_mask[pair_j]
        preferred_winner=torch.where(
            (i_controlled&~j_controlled)[None],pair_j[None],preferred_winner)
        preferred_winner=torch.where(
            (~i_controlled&j_controlled)[None],pair_i[None],preferred_winner)
        conflict &= (i_controlled|j_controlled)[None]
        retained = ((self._avoidance_winner >= 0) &
                    (current_distance < clearance + .35) &
                    (~teammate_pair[None] | self.teammate_intent_knowledge))
        winner = torch.where(retained,self._avoidance_winner,
            torch.where(conflict,preferred_winner,
                        torch.full_like(preferred_winner,-1)))
        if active is None:
            self._avoidance_winner.copy_(winner)
        else:
            self._avoidance_winner.copy_(torch.where(
                active[:,None],winner,self._avoidance_winner))
        i_priority = winner == pair_i[None]
        loser = torch.where(i_priority, pair_j[None], pair_i[None])
        loser_command = torch.where(i_priority[..., None], cmd_j, cmd_i)
        separation = torch.where(i_priority[..., None], -closest, closest)
        separation = torch.where(
            (separation.norm(dim=-1) < .1)[...,None],
            torch.where(i_priority[...,None],-relative,relative),separation)
        radial = separation / separation.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        tangent = torch.stack((-radial[..., 1], radial[..., 0]), -1)
        tangent = torch.where(((tangent * loser_command).sum(-1) < 0.)[..., None],
                              -tangent, tangent)
        desired_speed = loser_command.norm(dim=-1).clamp_min(.65)
        avoid_velocity = desired_speed[..., None] * (tangent + .75 * radial)
        avoid_velocity *= conflict[..., None]

        avoidance = torch.zeros_like(command)
        avoidance.scatter_add_(1, loser[..., None].expand(-1, -1, 2), avoid_velocity)
        count = torch.zeros_like(goal_distance)
        count.scatter_add_(1, loser, conflict.to(count.dtype))
        adjusted = avoidance / count.clamp_min(1.)[..., None]
        adjusted_speed = adjusted.norm(dim=-1, keepdim=True)
        adjusted *= torch.minimum(torch.ones_like(adjusted_speed),
                                  self.sim.speed[..., None] /
                                  adjusted_speed.clamp_min(1e-6))
        adjusted=torch.where((count > 0.)[..., None], adjusted, command)
        return torch.where(controlled[...,None],adjusted,torch.zeros_like(adjusted))
