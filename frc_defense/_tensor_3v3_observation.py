"""Perception updates and observation construction for the 3v3 environment."""
from __future__ import annotations

import math

import torch

from .tensor_sim import normalize_tensor_observation_batch_in_place
from . import tensor_fuel_candidate_rank as _fuel_candidate_rank_hip


NUM_ROBOTS = 6
OBS_DIM = 137
ACTION_DIM = 8


def _gated_angular_track_association(track_pos,track_mask,track_age,
                                     detections,detected,observer_pose,
                                     timeout,gate_m):
    """Associate sorted anonymous detections to nearby angular-rank tracks.

    Each detection checks only three neighboring prior bearing ranks. A
    conflict reduction keeps the assignment one-to-one; unmatched detections
    use expired/free slots. Complexity is O(n log n) for the two orderings and
    O(n) for candidate checks, independent of world count on the host.
    """
    worlds,slots=detected.shape
    old_delta=track_pos-observer_pose[:,None,:2]
    old_bearing=torch.atan2(old_delta[...,1],old_delta[...,0])
    old_bearing=torch.atan2(torch.sin(old_bearing-observer_pose[:,None,2]),
                            torch.cos(old_bearing-observer_pose[:,None,2]))
    old_order=torch.argsort(torch.where(track_mask,old_bearing,float("inf")),
                            dim=-1,stable=True)
    old_sorted=track_pos.gather(1,old_order[...,None].expand(-1,-1,2))
    old_valid=track_mask.gather(1,old_order)
    ranks=torch.arange(slots,device=track_pos.device)[None].expand(worlds,-1)
    candidate_ranks=torch.stack((ranks-1,ranks,ranks+1),-1).clamp(0,slots-1)
    candidate_pos=old_sorted.gather(
        1,candidate_ranks.reshape(worlds,-1)[...,None].expand(-1,-1,2)
    ).reshape(worlds,slots,3,2)
    candidate_valid=old_valid.gather(1,candidate_ranks.reshape(worlds,-1)).reshape(
        worlds,slots,3)
    distance=(candidate_pos-detections[:,:,None,:]).norm(dim=-1)
    distance=distance.masked_fill(~candidate_valid,float("inf"))
    nearest_distance,nearest_choice=distance.min(-1)
    nearest_rank=candidate_ranks.gather(-1,nearest_choice[...,None]).squeeze(-1)
    old_slot=old_order.gather(1,nearest_rank)
    matched=(detected & torch.isfinite(nearest_distance) &
             (nearest_distance<=gate_m))
    # Resolve collisions deterministically: the lowest sorted detection rank
    # wins an old slot; the rest are treated as new anonymous detections.
    winner=torch.full((worlds,slots),slots+1,device=track_pos.device,dtype=torch.long)
    winner.scatter_reduce_(1,old_slot,torch.where(matched,ranks+1,slots+1),
                           reduce="amin",include_self=True)
    matched &= winner.gather(1,old_slot)==ranks+1
    available=(~track_mask)|(track_age>timeout)
    free_order=torch.argsort(~available,dim=-1,stable=True)
    free_count=available.sum(-1,keepdim=True)
    needs_new=detected&~matched
    new_rank=needs_new.long().cumsum(-1)-1
    free_slot=free_order.gather(1,new_rank.clamp(0,slots-1))
    allocate=needs_new & (new_rank<free_count)
    target=torch.where(matched,old_slot,free_slot)
    accepted=matched|allocate
    return target,accepted


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
        if not count:
            return torch.zeros((self.n,*tail_shape),device=self.device,
                               dtype=dtype or torch.float32)
        shape=(count,*tail_shape)
        sample=(torch.randn(shape,device=self.device,generator=self.generator)
                if normal else torch.rand(shape,device=self.device,generator=self.generator))
        if count == self.n:
            return sample
        result=torch.zeros((self.n,*tail_shape),device=self.device,
                           dtype=dtype or torch.float32)
        result[active]=sample
        return result

    def _update_perception(self, active, active_count=None):
        """Update per-robot free-FUEL and nearest-opponent tracks."""
        pose = self.sim.pose
        free = self.piece_active & (self.piece_owner < 0)
        from . import tensor_perception
        all_robots_controlled=all(mode!="none" for mode in self.control_modes)
        camera_poses=None
        camera_visibility=None
        if all_robots_controlled:
            # Model each intake/opposite stereo pair at its chassis mount.
            # The two lenses in each pair use their own optical centers at
            # +/- half-baseline; the rear pair faces opposite the intake pair.
            baseline=self.stereo_baseline_m
            forward=torch.stack((torch.cos(pose[...,2]),
                                 torch.sin(pose[...,2])),-1)
            lateral=torch.stack((-torch.sin(pose[...,2]),
                                  torch.cos(pose[...,2])),-1)
            intake_center=pose.clone()
            intake_center[...,:2] += forward*self.sim.length[...,None]*.5
            opposite_center=pose.clone()
            opposite_center[...,:2] -= forward*self.sim.length[...,None]*.5
            opposite_center[...,2]=torch.atan2(torch.sin(pose[...,2]+math.pi),
                                                torch.cos(pose[...,2]+math.pi))
            intake_left=intake_center.clone()
            intake_left[...,:2] += lateral*(baseline*.5)
            intake_right=intake_center.clone()
            intake_right[...,:2] -= lateral*(baseline*.5)
            opposite_left=opposite_center.clone()
            opposite_left[...,:2] += lateral*(baseline*.5)
            opposite_right=opposite_center.clone()
            opposite_right[...,:2] -= lateral*(baseline*.5)
            visible_intake_left=tensor_perception.visibility_mask_3v3(
                intake_left.contiguous(), self.piece_pos.contiguous(), self.piece_active,
                self.piece_owner, active, self.perception_range,
                self.perception_fov_degrees, self.field_feature_obstacles,
                self._robot_occlusion_radius)
            visible_intake_right=tensor_perception.visibility_mask_3v3(
                intake_right.contiguous(), self.piece_pos.contiguous(), self.piece_active,
                self.piece_owner, active, self.perception_range,
                self.perception_fov_degrees, self.field_feature_obstacles,
                self._robot_occlusion_radius)
            visible_opposite_left=tensor_perception.visibility_mask_3v3(
                opposite_left.contiguous(), self.piece_pos.contiguous(), self.piece_active,
                self.piece_owner, active, self.perception_range,
                self.perception_fov_degrees, self.field_feature_obstacles,
                self._robot_occlusion_radius)
            visible_opposite_right=tensor_perception.visibility_mask_3v3(
                opposite_right.contiguous(), self.piece_pos.contiguous(), self.piece_active,
                self.piece_owner, active, self.perception_range,
                self.perception_fov_degrees, self.field_feature_obstacles,
                self._robot_occlusion_radius)
            camera_poses=torch.stack((intake_left,intake_right,
                                      opposite_left,opposite_right),dim=2)
            camera_visibility=torch.stack((visible_intake_left,visible_intake_right,
                                            visible_opposite_left,
                                            visible_opposite_right),dim=2)
            fused_visible=None
        else:
            fused_visible=None
        fused_tracks = False

        # The dashboard's one-world CPU rollout was doing six separate
        # [world, piece] visibility passes here. Flatten the observer dimension
        # for static-feature occlusion, then evaluate robot-to-robot occlusion
        # as one [world, observer, peer, piece] tensor. Keep random sampling in
        # its original per-robot order so seeded sensor noise is reproducible.
        if (self.device.type == "cpu" and camera_poses is None and fused_visible is None and
                all_robots_controlled):
            own_xy = pose[:, :, :2]
            delta = self.piece_pos[:, None, :, :] - own_xy[:, :, None, :]
            visible = (free[:, None, :] & active[:, None, None] &
                       (delta.norm(dim=-1) <= self.perception_range))
            if self.perception_fov < math.pi:
                bearing = torch.atan2(delta[..., 1], delta[..., 0])
                error = torch.atan2(
                    torch.sin(bearing - pose[:, :, None, 2]),
                    torch.cos(bearing - pose[:, :, None, 2])).abs()
                visible &= ((error <= self.perception_fov) |
                            (error >= math.pi-self.perception_fov))
            if self.field_feature_obstacles.numel():
                blocked = tensor_perception.piece_occlusion_mask(
                    own_xy.reshape(self.n * NUM_ROBOTS, 2).contiguous(),
                    delta.reshape(self.n * NUM_ROBOTS, self.fuel_count, 2),
                    self.field_feature_obstacles.contiguous(),
                    visible.reshape(self.n * NUM_ROBOTS, self.fuel_count).contiguous(),
                ).reshape(self.n, NUM_ROBOTS, self.fuel_count)
                visible &= ~blocked

            peers = pose[:, :, :2]
            peer_delta = peers[:, None, :, :] - own_xy[:, :, None, :]
            denom = delta.square().sum(-1).clamp_min(1e-8)
            fraction = (peer_delta[:, :, :, None, :] *
                        delta[:, :, None, :, :]).sum(-1) / denom[:, :, None, :]
            closest = (own_xy[:, :, None, None, :] +
                       fraction.clamp(0., 1.)[..., None] * delta[:, :, None, :, :])
            peer_blocked = ((closest - peers[:, None, :, None, :]).norm(dim=-1) <=
                            self._robot_occlusion_radius[:, None, :, None])
            peer_blocked &= (fraction > .02) & (fraction < .98)
            peer_blocked &= ~torch.eye(NUM_ROBOTS, device=self.device,
                                       dtype=torch.bool)[None, :, :, None]
            visible &= ~peer_blocked.any(dim=2)

            # The remaining loop updates only six opponent tracks, which are
            # tiny compared with repeating all 504-piece visibility work.
            batched_visible = visible
            piece_tracks_batched = True
        elif camera_poses is not None and all_robots_controlled:
            batched_visible=camera_visibility[:,:,0]
            piece_tracks_batched=True
        else:
            piece_tracks_batched = False
        has_controlled_defender=any(
            mode!="none" and self.robot_roles[robot]=="defense"
            for robot,mode in enumerate(self.control_modes))
        anonymous_piece_positions=None
        anonymous_piece_velocities=None
        anonymous_visible_fraction=None
        anonymous_spatial_extent=None
        anonymous_angular_extent=None
        anonymous_merged=None
        if piece_tracks_batched:
            if camera_poses is not None:
                per_camera=tensor_perception.angular_cluster_camera_detections(
                    camera_poses.contiguous(),self.piece_pos.contiguous(),
                    self.piece_vel.contiguous(),camera_visibility.contiguous(),
                    self.perception_ball_diameter,
                    math.radians(self.perception_fov_degrees)*.5,self)
                packed_detection=tensor_perception.stereo_fusion_pack(
                    pose.contiguous(),per_camera,self,self.fuel_count)
                if packed_detection is None:
                    # Front and rear camera pairs have disjoint 120-degree views.
                    # Fuse each stereo pair only after its lenses have independently
                    # generated occlusion-aware clusters. Keeping the two directions
                    # separate halves the final angular sweep's working set.
                    pair_pos=torch.stack((torch.cat((per_camera[0][:,:,0],
                                                       per_camera[0][:,:,1]),dim=2),
                                          torch.cat((per_camera[0][:,:,2],
                                                       per_camera[0][:,:,3]),dim=2)),dim=2)
                    pair_vel=torch.stack((torch.cat((per_camera[1][:,:,0],
                                                       per_camera[1][:,:,1]),dim=2),
                                          torch.cat((per_camera[1][:,:,2],
                                                       per_camera[1][:,:,3]),dim=2)),dim=2)
                    pair_valid=torch.stack((torch.cat((per_camera[2][:,:,0],
                                                         per_camera[2][:,:,1]),dim=2),
                                            torch.cat((per_camera[2][:,:,2],
                                                         per_camera[2][:,:,3]),dim=2)),dim=2)
                    pair_confidence=torch.stack((torch.cat((per_camera[3][:,:,0],
                                                              per_camera[3][:,:,1]),dim=2),
                                                 torch.cat((per_camera[3][:,:,2],
                                                              per_camera[3][:,:,3]),dim=2)),dim=2)
                    pair_spatial=torch.stack((torch.cat((per_camera[4][:,:,0],
                                                           per_camera[4][:,:,1]),dim=2),
                                              torch.cat((per_camera[4][:,:,2],
                                                           per_camera[4][:,:,3]),dim=2)),dim=2)
                    pair_angular=torch.stack((torch.cat((per_camera[5][:,:,0],
                                                           per_camera[5][:,:,1]),dim=2),
                                              torch.cat((per_camera[5][:,:,2],
                                                           per_camera[5][:,:,3]),dim=2)),dim=2)
                    pair_merged=torch.stack((torch.cat((per_camera[6][:,:,0],
                                                          per_camera[6][:,:,1]),dim=2),
                                             torch.cat((per_camera[6][:,:,2],
                                                          per_camera[6][:,:,3]),dim=2)),dim=2)
                    pair_poses=torch.stack((pose,pose.clone()),dim=2)
                    pair_poses[:,:,1,2]=torch.atan2(torch.sin(pose[:,:,2]+math.pi),
                                                     torch.cos(pose[:,:,2]+math.pi))
                    all_pos=pair_pos.reshape(self.n,NUM_ROBOTS*2,-1,2)
                    all_vel=pair_vel.reshape(self.n,NUM_ROBOTS*2,-1,2)
                    all_valid=pair_valid.reshape(self.n,NUM_ROBOTS*2,-1)
                    all_confidence=pair_confidence.reshape(self.n,NUM_ROBOTS*2,-1)
                    all_spatial=pair_spatial.reshape(self.n,NUM_ROBOTS*2,-1,2)
                    all_angular=pair_angular.reshape(self.n,NUM_ROBOTS*2,-1)
                    all_merged=pair_merged.reshape(self.n,NUM_ROBOTS*2,-1)
                    fusion_poses=pair_poses.reshape(self.n,NUM_ROBOTS*2,3)
                    (anonymous_piece_positions,anonymous_piece_velocities,
                     batched_visible,anonymous_visible_fraction,
                     anonymous_spatial_extent,anonymous_angular_extent,
                     anonymous_merged)=tensor_perception.fuse_camera_cluster_detections(
                        fusion_poses.contiguous(),all_pos,all_vel,all_valid,
                        all_confidence,all_spatial,all_angular,all_merged,
                        self.perception_ball_diameter,
                        math.radians(self.perception_fov_degrees)*.5)
                    anonymous_piece_positions=anonymous_piece_positions.reshape(
                        self.n,NUM_ROBOTS,2,-1,2)
                    anonymous_piece_velocities=anonymous_piece_velocities.reshape(
                        self.n,NUM_ROBOTS,2,-1,2)
                    batched_visible=batched_visible.reshape(
                        self.n,NUM_ROBOTS,2,-1)
                    anonymous_visible_fraction=anonymous_visible_fraction.reshape(
                        self.n,NUM_ROBOTS,2,-1)
                    anonymous_spatial_extent=anonymous_spatial_extent.reshape(
                        self.n,NUM_ROBOTS,2,-1,2)
                    anonymous_angular_extent=anonymous_angular_extent.reshape(
                        self.n,NUM_ROBOTS,2,-1)
                    anonymous_merged=anonymous_merged.reshape(
                        self.n,NUM_ROBOTS,2,-1)
                    # Front/rear views cannot overlap physically; concatenate their
                    # anonymous detections into the observation slots after fusion.
                    anonymous_piece_positions=anonymous_piece_positions.reshape(
                        self.n,NUM_ROBOTS,-1,2)
                    anonymous_piece_velocities=anonymous_piece_velocities.reshape(
                        self.n,NUM_ROBOTS,-1,2)
                    batched_visible=batched_visible.reshape(self.n,NUM_ROBOTS,-1)
                    anonymous_visible_fraction=anonymous_visible_fraction.reshape(
                        self.n,NUM_ROBOTS,-1)
                    anonymous_spatial_extent=anonymous_spatial_extent.reshape(
                        self.n,NUM_ROBOTS,-1,2)
                    anonymous_angular_extent=anonymous_angular_extent.reshape(
                        self.n,NUM_ROBOTS,-1)
                    anonymous_merged=anonymous_merged.reshape(self.n,NUM_ROBOTS,-1)
                    # Restore one bearing-ordered anonymous stream per robot before
                    # association. Invalid padding sorts after real detections.
                    delta=anonymous_piece_positions-pose[:,:,:2][:,:,None,:]
                    bearing=torch.atan2(delta[...,1],delta[...,0])
                    bearing=torch.atan2(torch.sin(bearing-pose[:,:,None,2]),
                                        torch.cos(bearing-pose[:,:,None,2]))
                    fused_order=torch.argsort(torch.where(
                        batched_visible,bearing,float("inf")),dim=-1)
                    batched_visible=batched_visible.gather(-1,fused_order)
                    anonymous_piece_positions=anonymous_piece_positions.gather(
                        2,fused_order[...,None].expand(-1,-1,-1,2))
                    anonymous_piece_velocities=anonymous_piece_velocities.gather(
                        2,fused_order[...,None].expand(-1,-1,-1,2))
                    anonymous_visible_fraction=anonymous_visible_fraction.gather(
                        -1,fused_order)
                    anonymous_spatial_extent=anonymous_spatial_extent.gather(
                        2,fused_order[...,None].expand(-1,-1,-1,2))
                    anonymous_angular_extent=anonymous_angular_extent.gather(
                        -1,fused_order)
                    anonymous_merged=anonymous_merged.gather(-1,fused_order)
                    # At most the physical FUEL count can be distinct detections;
                    # extra slots can only be residual cross-camera duplicates.
                    limit=self.fuel_count
                    anonymous_piece_positions=anonymous_piece_positions[:,:,:limit]
                    anonymous_piece_velocities=anonymous_piece_velocities[:,:,:limit]
                    batched_visible=batched_visible[:,:,:limit]
                    anonymous_visible_fraction=anonymous_visible_fraction[:,:,:limit]
                    anonymous_spatial_extent=anonymous_spatial_extent[:,:,:limit]
                    anonymous_angular_extent=anonymous_angular_extent[:,:,:limit]
                    anonymous_merged=anonymous_merged[:,:,:limit]
                else:
                    (anonymous_piece_positions,anonymous_piece_velocities,
                     batched_visible,anonymous_visible_fraction,
                     anonymous_spatial_extent,anonymous_angular_extent,
                     anonymous_merged)=packed_detection
            else:
                delta=self.piece_pos[:,None]-pose[:,:,:2][:,:,None,:]
                bearing=torch.atan2(delta[...,1],delta[...,0])
                bearing=torch.atan2(torch.sin(bearing-pose[:,:,None,2]),
                                    torch.cos(bearing-pose[:,:,None,2]))
                order=torch.argsort(torch.where(batched_visible,bearing,float("inf")),
                                    dim=-1,stable=True)
                batched_visible=batched_visible.gather(-1,order)
                anonymous_piece_positions=self.piece_pos[:,None].expand(
                    -1,NUM_ROBOTS,-1,-1).gather(
                        2,order[...,None].expand(-1,-1,-1,2))
                anonymous_piece_velocities=self.piece_vel[:,None].expand(
                    -1,NUM_ROBOTS,-1,-1).gather(
                        2,order[...,None].expand(-1,-1,-1,2))
                (anonymous_piece_positions,anonymous_piece_velocities,
                 batched_visible,anonymous_visible_fraction,
                 anonymous_spatial_extent,anonymous_angular_extent,
                 anonymous_merged)=tensor_perception.angular_cluster_detections(
                    pose.contiguous(),anonymous_piece_positions,
                    anonymous_piece_velocities,batched_visible,
                    self.perception_ball_diameter)
        if piece_tracks_batched and anonymous_piece_positions is not None:
            fused_tracks=tensor_perception.anonymous_track_update(
                pose.contiguous(),anonymous_piece_positions.contiguous(),
                anonymous_piece_velocities.contiguous(),batched_visible.contiguous(),
                anonymous_visible_fraction.contiguous(),
                anonymous_spatial_extent.contiguous(),
                anonymous_angular_extent.contiguous(),anonymous_merged.contiguous(),
                active.contiguous(),self._perception_rng_ticks,
                self.track_pos,self.track_vel,self.track_age,self.track_mask,
                self.perception_quality_state,self.track_confidence,
                self.track_spatial_extent,self.track_angular_extent,
                self.track_merged,self._current_fuel_visibility,
                self._perception_rng_seed,self.dt,self.perception_dropout,
                self.position_noise,self.velocity_noise,
                self.perception_track_timeout,.35+self.position_noise*4.,
                self.perception_ball_diameter)
        self._fused_sensor_rng_used |= bool(fused_tracks)
        if not fused_tracks:
            self.track_age.copy_(torch.where(
                active[:, None, None] & self.track_mask,
                self.track_age + self.dt, self.track_age))

        opponent_robot_rows=[]
        opponent_has_rows=[]
        opponent_pose_rows=[]
        opponent_velocity_rows=[]
        opponent_size_rows=[]
        fused_opponents = (has_controlled_defender and self.fused_sensor_rng and
            camera_poses is not None and
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
        self._fused_sensor_rng_used |= bool(fused_opponents)
        batch_opponent_commit=(piece_tracks_batched and has_controlled_defender and
                               not fused_opponents)
        if has_controlled_defender and not fused_opponents:
            self.opponent_age.copy_(torch.where(
                active[:, None] & self.opponent_valid,
                self.opponent_age + self.dt, self.opponent_age))
        for robot in range(NUM_ROBOTS):
            if self.control_modes[robot] == "none":
                continue
            own = pose[:, robot]
            if not fused_tracks:
                observed_positions=None
                observed_velocities=None
                if piece_tracks_batched:
                    visible = batched_visible[:, robot]
                    observed_positions=anonymous_piece_positions[:,robot]
                    observed_velocities=anonymous_piece_velocities[:,robot]
                elif fused_visible is not None:
                    visible = fused_visible[:, robot]
                else:
                    delta = self.piece_pos - own[:, None, :2]
                    distance = delta.norm(dim=-1)
                    visible = free & (distance <= self.perception_range)
                    if self.perception_fov < math.pi:
                        bearing = torch.atan2(delta[..., 1], delta[..., 0])
                        error = torch.atan2(torch.sin(bearing - own[:, None, 2]),
                                            torch.cos(bearing - own[:, None, 2])).abs()
                        visible &= ((error <= self.perception_fov) |
                                    (error >= math.pi-self.perception_fov))
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
                if not piece_tracks_batched:
                    rel=self.piece_pos-own[:,None,:2]
                    bearing=torch.atan2(rel[...,1],rel[...,0])
                    bearing=torch.atan2(torch.sin(bearing-own[:,None,2]),
                                        torch.cos(bearing-own[:,None,2]))
                    order=torch.argsort(torch.where(visible,bearing,float("inf")),
                                        dim=-1,stable=True)
                    visible=visible.gather(-1,order)
                    observed_positions=self.piece_pos.gather(
                        1,order[...,None].expand(-1,-1,2))
                    observed_velocities=self.piece_vel.gather(
                        1,order[...,None].expand(-1,-1,2))
                    (group_pos,group_vel,group_valid,group_fraction,
                     group_spatial_extent,group_angular_extent,group_merged)=(
                        tensor_perception.angular_cluster_detections(
                            own[:,None].contiguous(),observed_positions[:,None],
                            observed_velocities[:,None],visible[:,None],
                            self.perception_ball_diameter))
                    observed_positions=group_pos[:,0]
                    observed_velocities=group_vel[:,0]
                    visible=group_valid[:,0]
                    anonymous_visible_fraction=group_fraction[:,0]
                    anonymous_spatial_extent=group_spatial_extent[:,0]
                    anonymous_angular_extent=group_angular_extent[:,0]
                    anonymous_merged=group_merged[:,0]
                self._current_fuel_visibility[:, robot] = torch.where(
                    active[:,None],torch.zeros_like(visible),
                    self._current_fuel_visibility[:,robot])
                distance=(observed_positions-own[:,None,:2]).norm(dim=-1)
                angular_width=2.*torch.atan(
                    (self.perception_ball_diameter*.5)/distance.clamp_min(.05))
                instantaneous_quality=(torch.exp(-distance/8.)*
                    (angular_width/.012).clamp(max=1.)).clamp(0.,1.)
                if anonymous_visible_fraction is not None:
                    fraction=(anonymous_visible_fraction[:,robot]
                        if piece_tracks_batched else anonymous_visible_fraction)
                    instantaneous_quality*=fraction
                old_quality=self.perception_quality_state[:,robot]
                current_fraction=(anonymous_visible_fraction[:,robot]
                    if piece_tracks_batched else anonymous_visible_fraction)
                quality=torch.where(visible,
                    (.8*old_quality+.2*instantaneous_quality)*current_fraction,
                                    old_quality)
                self.perception_quality_state[:,robot].copy_(torch.where(
                    active[:,None],quality,old_quality))
                if not fused_tracks:
                    detection_draw=self._random_active(
                        active,(self.fuel_count,),active_count=active_count)
                    visible &= detection_draw < quality*(1.-self.perception_dropout)
                    size_quality=(angular_width/.012).clamp(.15,1.)
                    noise_scale=(self.position_noise*(1.+distance/8.)/
                                 size_quality.sqrt())
                    noise = self._random_active(active,(self.fuel_count,2),normal=True,active_count=active_count) * noise_scale[...,None]
                    vnoise = self._random_active(active,(self.fuel_count,2),normal=True,active_count=active_count) * self.velocity_noise
                    row_visible = visible & active[:, None]
                    target,accepted=_gated_angular_track_association(
                        self.track_pos[:,robot],self.track_mask[:,robot],
                        self.track_age[:,robot],observed_positions,row_visible,
                        own,self.perception_track_timeout,
                        .35+self.position_noise*4.)
                    rows=torch.arange(self.n,device=self.device)[:,None].expand_as(target)
                    self.track_pos[rows[accepted],robot,target[accepted]]=\
                        (observed_positions+noise)[accepted]
                    self.track_vel[rows[accepted],robot,target[accepted]]=\
                        (observed_velocities+vnoise)[accepted]
                    self.track_age[rows[accepted],robot,target[accepted]]=0.
                    self.track_mask[rows[accepted],robot,target[accepted]]=True
                    spatial=(anonymous_spatial_extent[:,robot]
                        if piece_tracks_batched else anonymous_spatial_extent)
                    angular=(anonymous_angular_extent[:,robot]
                        if piece_tracks_batched else anonymous_angular_extent)
                    merged=(anonymous_merged[:,robot]
                        if piece_tracks_batched else anonymous_merged)
                    self.track_confidence[rows[accepted],robot,target[accepted]]=\
                        quality[accepted]
                    self.track_spatial_extent[rows[accepted],robot,target[accepted]]=\
                        spatial[accepted]
                    self.track_angular_extent[rows[accepted],robot,target[accepted]]=\
                        angular[accepted]
                    self.track_merged[rows[accepted],robot,target[accepted]]=\
                        merged[accepted]
                    self._current_fuel_visibility[
                        rows[accepted],robot,target[accepted]]=True

            if self.robot_roles[robot] != "defense":
                continue
            if fused_opponents:
                continue
            enemy_ids = self._enemy_robot_ids[robot]
            enemy_pose = pose[:, enemy_ids]
            enemy_delta = enemy_pose[..., :2] - own[:, None, :2]
            enemy_distance = enemy_delta.norm(dim=-1)
            enemy_bearing = torch.atan2(enemy_delta[..., 1], enemy_delta[..., 0])
            bearing_error = torch.atan2(torch.sin(enemy_bearing-own[:, None, 2]),
                                        torch.cos(enemy_bearing-own[:, None, 2])).abs()
            in_view = ((enemy_distance <= self.perception_range) &
                       ((bearing_error <= self.perception_fov) |
                        (bearing_error >= math.pi-self.perception_fov)))
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
            enemy_size = torch.stack((self.sim.length[:, enemy_ids],
                                      self.sim.width[:, enemy_ids],
                                      self.sim.accel[:, enemy_ids] / 10.), -1)
            chosen_size = enemy_size[self._world_indices, nearest]
            if batch_opponent_commit:
                opponent_robot_rows.append(robot)
                opponent_has_rows.append(has_enemy)
                opponent_pose_rows.append(chosen)
                opponent_velocity_rows.append(chosen_velocity)
                opponent_size_rows.append(chosen_size)
            else:
                self.opponent_pose[:, robot] = torch.where(has_enemy[:, None], chosen,
                                                           self.opponent_pose[:, robot])
                self.opponent_velocity[:, robot] = torch.where(has_enemy[:, None], chosen_velocity,
                                                               self.opponent_velocity[:, robot])
                self.opponent_size[:, robot] = torch.where(has_enemy[:, None], chosen_size,
                                                           self.opponent_size[:, robot])
                self.opponent_age[:, robot] = torch.where(has_enemy, torch.zeros_like(self.opponent_age[:, robot]),
                                                          self.opponent_age[:, robot])
                self.opponent_valid[:, robot] |= has_enemy
        if batch_opponent_commit and opponent_robot_rows:
            rows=self._defense_robot_ids
            has_enemy=torch.stack(opponent_has_rows,dim=1)
            chosen=torch.stack(opponent_pose_rows,dim=1)
            chosen_velocity=torch.stack(opponent_velocity_rows,dim=1)
            chosen_size=torch.stack(opponent_size_rows,dim=1)
            old_pose=self.opponent_pose[:,rows]
            old_velocity=self.opponent_velocity[:,rows]
            old_size=self.opponent_size[:,rows]
            old_age=self.opponent_age[:,rows]
            old_valid=self.opponent_valid[:,rows]
            self.opponent_pose.index_copy_(1,rows,torch.where(
                has_enemy[...,None],chosen,old_pose))
            self.opponent_velocity.index_copy_(1,rows,torch.where(
                has_enemy[...,None],chosen_velocity,old_velocity))
            self.opponent_size.index_copy_(1,rows,torch.where(
                has_enemy[...,None],chosen_size,old_size))
            self.opponent_age.index_copy_(1,rows,torch.where(
                has_enemy,torch.zeros_like(old_age),old_age))
            self.opponent_valid.index_copy_(1,rows,old_valid|has_enemy)
        # Track lifetime is observation-driven. Do not reveal collection or
        # removal through simulator-owned FUEL state; stale detections expire
        # through the configured timeout just as on a real robot.
        if fused_tracks or fused_opponents:
            self._perception_rng_ticks.add_(active.long())
        if not fused_tracks:
            self.track_mask &= ~((self.track_age > self.perception_track_timeout) &
                                 active[:, None, None])
        if not fused_opponents:
            self.opponent_valid &= ~((self.opponent_age > 1.) & active[:, None])

    def _candidates(self, candidate_mask=None):
        candidate_mask = self.track_mask if candidate_mask is None else candidate_mask
        active_count=self._controlled_robot_ids.numel()
        if active_count<NUM_ROBOTS:
            rows=self._controlled_robot_ids
            track_pos=self.track_pos[:,rows].reshape(
                self.n*active_count,self.fuel_count,2)
            track_mask=candidate_mask[:,rows].reshape(
                self.n*active_count,self.fuel_count)
            robot_pos=self.sim.pose[:,rows,:2].reshape(self.n*active_count,2)
            if active_count:
                if (_fuel_candidate_rank_hip is not None and
                        _fuel_candidate_rank_hip.integrated_available(self.device)):
                    indices,valid,nearest=_fuel_candidate_rank_hip.candidates(
                        track_pos,robot_pos,track_mask)
                else:
                    distance=(track_pos-robot_pos[:,None]).norm(dim=-1)
                    nearest,indices=distance.masked_fill(
                        ~track_mask,float("inf")).topk(4,dim=-1,largest=False)
                    valid=torch.isfinite(nearest)
                shape=(self.n,active_count,4)
                indices=indices.reshape(shape); valid=valid.reshape(shape)
                nearest=nearest.reshape(shape)
            else:
                indices=torch.empty((self.n,0,4),device=self.device,dtype=torch.long)
                valid=torch.empty((self.n,0,4),device=self.device,dtype=torch.bool)
                nearest=torch.empty((self.n,0,4),device=self.device)
            all_indices=torch.zeros((self.n,NUM_ROBOTS,4),device=self.device,dtype=torch.long)
            all_valid=torch.zeros((self.n,NUM_ROBOTS,4),device=self.device,dtype=torch.bool)
            all_nearest=torch.full((self.n,NUM_ROBOTS,4),float("inf"),device=self.device)
            all_indices.index_copy_(1,rows,indices)
            all_valid.index_copy_(1,rows,valid)
            all_nearest.index_copy_(1,rows,nearest)
            return all_indices,all_valid,all_nearest
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
