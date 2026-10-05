"""Action target selection and robot contention behavior for the 3v3 env."""
from __future__ import annotations

import torch


class TensorThreeVsThreeActionMixin:
    """Methods that translate strategic actions into robot targets."""

    def _target_for_actions(self, actions, active):
        p = self.sim.pose
        indices, valid, _ = self._candidates()
        masks, possession = self._action_mask(indices, valid)
        robot_hub_active = self.hub_active[:, self.team_ids]
        hub=self.hub_centers[self.team_ids][None].expand(self.n,-1,-1)
        intake_indices, intake_eligible = indices, valid
        allowed_tracks=self.track_mask
        if self._has_deterministic_offense:
            track_in_alliance = torch.where(
                self.team_ids[None, :, None] == 0,
                self.track_pos[..., 0] <= self.alliance_zone_depth,
                self.track_pos[..., 0] >= self.field_length - self.alliance_zone_depth)
            # Collection runs as a continuous field sweep. During a closed
            # shift, keep intake targets outside the alliance zone so fuel
            # gathered in midfield can be returned home.
            allowed_tracks = self.track_mask & torch.where(
                robot_hub_active[..., None], torch.ones_like(track_in_alliance),
                ~track_in_alliance)
            intake_indices, intake_eligible, _ = self._candidates(allowed_tracks)
        candidate_points = self.track_pos.gather(
            2, intake_indices[..., None].expand(-1, -1, -1, 2))
        intake_available = intake_eligible.any(-1)
        # A deterministic defender holds the opponent HUB approach even when
        # the attacker is temporarily occluded; giving up the post lets a
        # scorer approach freely during perception dropouts.
        defense_action = torch.full_like(possession,5)
        if self._has_deterministic_offense:
            # Keep batches small enough to finish scoring during an active hub
            # shift. Endgame batches shrink to leave time to score.
            # These offense-only rules do not affect NN attackers or defenders.
            active_batch_limit=torch.minimum(
                self.robot_fuel_capacities[None].expand_as(possession),
                torch.full_like(possession,24))
            active_batch_limit=torch.where(self._dumper_mask[None],
                self.robot_fuel_capacities[None].expand_as(possession),
                active_batch_limit)
            endgame_remaining = (160. - self.match_elapsed[:, None]).clamp_min(0.)
            endgame_batch_limit = torch.ceil(
                active_batch_limit * endgame_remaining / 40.).long().clamp_min(1)
            endgame_batch_limit = endgame_batch_limit.expand_as(possession)
            endgame_batch_limit = torch.minimum(endgame_batch_limit,active_batch_limit)
            active_batch_limit = torch.where(
                (self.match_elapsed[:, None] >= 130.),endgame_batch_limit,active_batch_limit)
            active_batch_limit=torch.where(self._dumper_mask[None],
                self.robot_fuel_capacities[None].expand_as(possession),
                active_batch_limit)
            batch_limit = torch.where(
                robot_hub_active,active_batch_limit,
                self.robot_fuel_capacities[None].expand_as(possession))
            hub_bearing=torch.atan2(hub[...,1]-p[...,1],hub[...,0]-p[...,0])
            hub_heading_error=torch.atan2(
                torch.sin(hub_bearing-p[...,2]),torch.cos(hub_bearing-p[...,2])).abs()
            bump_clear=self._robots_clear_of_bumps(p[:,:,:2])
            gather_more = (possession < batch_limit) & intake_available
            ready_to_score = ((possession > 0) & robot_hub_active &
                              ((possession >= batch_limit) | ~intake_available))
            partial_ferry_limit=torch.minimum(
                torch.full_like(possession,6),
                self.robot_fuel_capacities[None].expand_as(possession))
            ferry_batch_limit=torch.where(self._dumper_mask[None],
                self.robot_fuel_capacities[None].expand_as(possession),
                partial_ferry_limit)
            begin_ferry=(possession>=ferry_batch_limit)|~intake_available
            ferry_committed=(~robot_hub_active & (possession>0) &
                (self._ferry_committed|begin_ferry))
            self._ferry_committed.copy_(torch.where(
                active[:,None],ferry_committed,self._ferry_committed))
            ferry_inactive_fuel=(~robot_hub_active & (possession > 0) & ferry_committed)
            offense_action=torch.where(ferry_inactive_fuel,
                torch.full_like(possession,6),
                torch.where(ready_to_score,torch.full_like(possession,4),
                torch.where(gather_more,torch.zeros_like(possession),
                            torch.full_like(possession,6))))
            deterministic=torch.where(self._defense_role_mask[None],defense_action,
                                      offense_action)
        else:
            # With no deterministic offense controller, all deterministic
            # controllers in this batch are defenders and use action 5.
            deterministic=defense_action
        action = torch.as_tensor(actions, device=self.device, dtype=torch.long).reshape(self.n, 6)
        action = torch.where(self._nn_mode_mask[None], action, torch.where(
            self._deterministic_mode_mask[None],deterministic,torch.full_like(action,7)))
        action = action.clamp(0, 7)
        action = torch.where(masks.gather(-1, action[...,None]).squeeze(-1), action,
                             masks.to(torch.int64).argmax(-1))
        self.last_actions.copy_(torch.where(active[:,None],action,self.last_actions))
        rows = self._world_indices[:,None]
        selected = indices.gather(2,action.clamp(max=3)[...,None]).squeeze(-1)
        deterministic_offense = (self._deterministic_mode_mask &
                                 ~self._defense_role_mask)[None]
        deterministic_robot = self._deterministic_mode_mask[None]
        # Rank deterministic intake targets by distance from the robot.
        if self._has_deterministic_offense:
            local_dist = (candidate_points - p[:,:,None,:2]).norm(dim=-1)
            intake_cost=local_dist
            staging_cost = intake_cost
            staging_idx = staging_cost.masked_fill(~intake_eligible,float("inf")).argmin(-1)
            stage_selected = intake_indices.gather(2,staging_idx[...,None]).squeeze(-1)
            stage_mask = deterministic_offense & ~robot_hub_active & intake_available
            selected = torch.where(stage_mask,stage_selected,selected)
            # Assign different pieces to same-alliance deterministic robots.
            # Without this, all three bots can choose the identical nearest
            # track and converge on one intake target. Reserve each chosen
            # track for later robots on that alliance, using the same closed-
            # hub staging cost as the individual selector.
            candidate_cost = torch.where(
                (~robot_hub_active)[..., None], staging_cost, intake_cost)
            assigned = []
            for robot in range(self.team_ids.numel()):
                wants_fuel = deterministic_offense[:, robot] & (action[:, robot] < 4)
                available = intake_eligible[:, robot].clone()
                for prior_robot, prior_index, prior_active in assigned:
                    same_alliance = self.team_ids[robot] == self.team_ids[prior_robot]
                    duplicate = intake_indices[:, robot] == prior_index[:, None]
                    available &= ~(duplicate & (prior_active & wants_fuel & same_alliance)[:, None])
                costs = candidate_cost[:, robot].masked_fill(~available, float("inf"))
                rank = costs.argmin(-1)
                has_assignment = torch.isfinite(costs.gather(1, rank[:, None]).squeeze(1))
                proposed = intake_indices[:, robot].gather(1, rank[:, None]).squeeze(1)
                selected[:, robot] = torch.where(
                    wants_fuel & has_assignment, proposed, selected[:, robot])
                assigned.append((robot, selected[:, robot].clone(),
                                 wants_fuel & has_assignment))
        fuel_target = self.track_pos.gather(
            2,selected[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        possession_now = possession > 0
        hub = self.hub_centers[self.team_ids][None].expand(self.n,-1,-1)
        # Score from the nearest point inside the alliance zone. Robots already
        # in the zone stay put; dumpers turn toward the HUB before firing.
        robot_radius=.5*torch.sqrt(self.sim.length.square()+self.sim.width.square())
        red_min=robot_radius
        red_max=(self.alliance_zone_depth-robot_radius).clamp_min(0.)
        blue_min=self.field_length-self.alliance_zone_depth+robot_radius
        blue_max=torch.full_like(blue_min,self.field_length)-robot_radius
        x_min=torch.where(self.team_ids[None,:]==0,red_min,blue_min)
        x_max=torch.where(self.team_ids[None,:]==0,red_max,blue_max)
        score_target=p[:,:,:2].clone()
        score_target[...,0]=torch.maximum(x_min,torch.minimum(score_target[...,0],x_max))
        score_target[...,1]=torch.maximum(
            robot_radius,torch.minimum(score_target[...,1],self.sim.field_width-robot_radius))
        # The release gate checks simulator bump regions directly. Include
        # those same rectangles here so the selected staging point can satisfy
        # that gate instead of stopping at a target that still overlaps a bump.
        boxes=torch.cat((self.planner._boxes,self.planner._bumps,
                         self.sim.bump_regions),0)
        if boxes.numel():
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
            dx=(candidates[...,None,0]-box_x[:,:,None,:]).abs()
            dy=(candidates[...,None,1]-box_y[:,:,None,:]).abs()
            overlap=((dx<=box_hx[:,:,None,:]+robot_radius[...,None,None]) &
                     (dy<=box_hy[:,:,None,:]+robot_radius[...,None,None]))
            clear=~overlap.any(-1)
            distance=(candidates-p[:,:,None,:2]).norm(dim=-1)
            nearest=distance.masked_fill(~clear,float('inf')).argmin(-1)
            has_clear=clear.any(-1)
            safe_target=candidates.gather(
                2,nearest[...,None,None].expand(-1,-1,1,2)).squeeze(2)
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
        sweeping=(deterministic_offense & ((action<4)|
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
        return target, action, possession_now, fuel_delta

    def _avoid_robot_contention(self, command, targets, active=None):
        """Bend commanded motion around robots on near-term collision courses."""
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
        return torch.where((count > 0.)[..., None], adjusted, command)
