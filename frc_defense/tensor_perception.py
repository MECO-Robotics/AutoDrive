"""Optional fused HIP implementation of FUEL piece occlusion checks.

The exact-parity accelerator path is enabled by default on HIP. Set
AUTODRIVE_FUSED_PERCEPTION_HIP=0 to force the Torch reference path. The fused
kernel does not use or advance any RNG state.
"""
from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import torch


_HIP_PERCEPTION_EXTENSION = None
_HIP_PERCEPTION_EXTENSION_ATTEMPTED = False
_HIP_PERCEPTION_ENABLED = os.environ.get("AUTODRIVE_FUSED_PERCEPTION_HIP", "1") != "0"
FUSED_PERCEPTION_HIP_ENABLED = _HIP_PERCEPTION_ENABLED
FUSED_TRACK_AGING_HIP_ENABLED = (
    os.environ.get("AUTODRIVE_FUSED_TRACK_AGING_HIP", "1") != "0")


def _hip_perception_extension():
    """Load the optional HIP kernel, returning None on unsupported systems."""
    global _HIP_PERCEPTION_EXTENSION, _HIP_PERCEPTION_EXTENSION_ATTEMPTED
    if not _HIP_PERCEPTION_ENABLED:
        return None
    if _HIP_PERCEPTION_EXTENSION_ATTEMPTED:
        return _HIP_PERCEPTION_EXTENSION
    _HIP_PERCEPTION_EXTENSION_ATTEMPTED = True
    if not (torch.cuda.is_available() and torch.version.hip):
        return None
    try:
        from torch.utils import cpp_extension

        bundled_sdk = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
        rocm_sdk = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
                        (bundled_sdk if bundled_sdk.exists() else cpp_extension._find_rocm_home()))
        rocm_sdk = rocm_sdk.resolve()
        # The bundled wrapper mishandles toolchain paths containing spaces.
        rocm_alias = rocm_sdk
        if " " in str(rocm_sdk):
            rocm_alias = Path("/tmp/autodrive-rocm-sdk")
            if rocm_alias.is_symlink() and rocm_alias.resolve() != rocm_sdk:
                rocm_alias.unlink()
            if not rocm_alias.exists():
                rocm_alias.symlink_to(rocm_sdk, target_is_directory=True)
            os.environ.setdefault("ROCM_HOME", str(rocm_alias))
            os.environ.setdefault("HIP_CLANG_PATH", str(rocm_alias / "lib/llvm/bin"))

        torch_lib = Path(torch.__file__).parent / "lib"
        torch_lib_alias = torch_lib.resolve()
        if " " in str(torch_lib_alias):
            torch_lib_alias = Path("/tmp/autodrive-torch-lib")
            if torch_lib_alias.is_symlink() and torch_lib_alias.resolve() != torch_lib.resolve():
                torch_lib_alias.unlink()
            if not torch_lib_alias.exists():
                torch_lib_alias.symlink_to(torch_lib.resolve(), target_is_directory=True)

        original_torch_lib = cpp_extension.TORCH_LIB_PATH
        original_rocm_home = cpp_extension.ROCM_HOME
        original_hip_home = cpp_extension.HIP_HOME
        cpp_extension.TORCH_LIB_PATH = str(torch_lib_alias)
        cpp_extension.ROCM_HOME = str(rocm_alias)
        cpp_extension.HIP_HOME = str(rocm_alias / "hip")
        source_dir = Path(__file__).parent / "csrc"
        device_lib = next((p for p in (
            rocm_alias / "lib/llvm/amdgcn/bitcode",
            rocm_alias / "amdgcn/bitcode",
        ) if p.is_dir()), None)
        cuda_flags = ["-O3", "-ffp-contract=off"]
        if device_lib is not None:
            cuda_flags.append(f"--rocm-device-lib-path={device_lib}")
        try:
            _HIP_PERCEPTION_EXTENSION = cpp_extension.load(
                name="autodrive_tensor_perception_hip",
                sources=[str(source_dir / "tensor_perception_hip.cpp"),
                         str(source_dir / "tensor_perception_hip_kernel.cu")],
                with_cuda=True,
                extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags,
                verbose=False,
            )
        finally:
            cpp_extension.TORCH_LIB_PATH = original_torch_lib
            cpp_extension.ROCM_HOME = original_rocm_home
            cpp_extension.HIP_HOME = original_hip_home
    except Exception as exc:  # An optional kernel must not break training.
        warnings.warn(f"Fused perception HIP kernel unavailable; using Torch: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _HIP_PERCEPTION_EXTENSION


def piece_occlusion_mask_torch(pose_xy, segment, obstacles, eligible):
    """Reference predicate, skipping pieces already known to be invisible."""
    if obstacles.shape[-1] == 4:
        origin=pose_xy[:,None,None,:]
        direction=segment[:,:,None,:]
        center=obstacles[None,None,:,:2]
        half=obstacles[None,None,:,2:4]+.03
        parallel=direction.abs()<1e-8
        outside_parallel=(origin-center).abs()>half
        safe_direction=torch.where(parallel,torch.ones_like(direction),direction)
        first=(center-half-origin)/safe_direction
        second=(center+half-origin)/safe_direction
        near=torch.minimum(first,second)
        far=torch.maximum(first,second)
        near=torch.where(parallel,float("-inf"),near)
        far=torch.where(parallel,float("inf"),far)
        enters=near.amax(-1).clamp_min(.02)
        exits=far.amin(-1).clamp_max(.98)
        intersects=(exits>=enters)&~(parallel&outside_parallel).any(-1)
        return intersects.any(-1)&eligible
    denom = segment.square().sum(-1).clamp_min(1e-8)
    rel = obstacles[None, None, :, :2] - pose_xy[:, None, None, :]
    t = (rel * segment[:, :, None, :]).sum(-1) / denom[:, :, None]
    closest = pose_xy[:, None, None, :] + t.clamp(0., 1.)[..., None] * segment[:, :, None, :]
    obstacle_radius = obstacles[None, None, :, 2] + .03
    blocked = ((obstacle_radius > 0) &
               ((closest - obstacles[None, None, :, :2]).norm(dim=-1) <=
                obstacle_radius) &
                (t > .02) & (t < .98)).any(-1)
    return blocked & eligible


def visibility_mask_torch(pose, pieces, piece_active, piece_owner, active,
                          perception_range, fov_degrees, obstacles,
                          other_xy, other_radius):
    """Torch reference for deterministic FUEL visibility before sensor noise."""
    segment = pieces - pose[:, None, :2]
    distance = segment.norm(dim=-1)
    visible = (piece_active & (piece_owner < 0) & active[:, None] &
               (distance <= float(perception_range)))
    if float(fov_degrees) < 360.:
        bearing = torch.atan2(segment[..., 1], segment[..., 0])
        angle = torch.atan2(torch.sin(bearing - pose[:, None, 2]),
                            torch.cos(bearing - pose[:, None, 2])).abs()
        visible &= angle <= math.radians(float(fov_degrees)) * .5
    if obstacles.numel():
        visible &= ~piece_occlusion_mask_torch(
            pose[:, :2], segment, obstacles, visible.contiguous())
    rel = other_xy[:, None, :] - pose[:, None, :2]
    denom = segment.square().sum(-1).clamp_min(1e-8)
    t = (rel * segment).sum(-1) / denom
    closest = pose[:, None, :2] + t.clamp(0., 1.)[..., None] * segment
    occluded = (((closest - other_xy[:, None, :]).norm(dim=-1) <=
                 other_radius[:, None]) & (t > .02) & (t < .98))
    return visible & ~occluded


def piece_occlusion_mask(pose_xy, segment, obstacles, eligible):
    """Return [world,piece] occlusion, using HIP only when explicitly enabled.

    ``pose_xy`` is [world,2], ``segment`` is piece-minus-pose [world,piece,2],
    and ``obstacles`` is [circle,3] containing x, y, radius. The fused kernel
    is opt-in through AUTODRIVE_FUSED_PERCEPTION_HIP=1.
    """
    extension = _hip_perception_extension() if _HIP_PERCEPTION_ENABLED else None
    if (extension is not None and obstacles.shape[-1] in (3, 4) and
            pose_xy.is_cuda and pose_xy.dtype == torch.float32
            and segment.dtype == torch.float32 and obstacles.dtype == torch.float32
            and pose_xy.is_contiguous() and segment.is_contiguous()
            and obstacles.is_contiguous() and eligible.dtype == torch.bool
            and eligible.is_contiguous()):
        return extension.piece_occlusion(pose_xy, segment, obstacles, eligible)
    return piece_occlusion_mask_torch(pose_xy, segment, obstacles, eligible)


def visibility_mask(pose, pieces, piece_active, piece_owner, active,
                    perception_range, fov_degrees, obstacles,
                    other_xy, other_radius):
    """Deterministic visibility; dropout/noise and tracking remain in Torch."""
    extension = _hip_perception_extension() if _HIP_PERCEPTION_ENABLED else None
    tensors = (pose, pieces, piece_active, piece_owner, active, obstacles,
              other_xy, other_radius)
    if (extension is not None and pose.is_cuda and pose.dtype == torch.float32
            and all(t.is_contiguous() for t in tensors)
            and all(t.device == pose.device for t in tensors)
            and piece_active.dtype == torch.bool and piece_owner.dtype == torch.long
            and active.dtype == torch.bool and obstacles.dtype == torch.float32
            and other_xy.dtype == torch.float32 and other_radius.dtype == torch.float32):
        return extension.visibility_mask(
            pose, pieces, piece_active, piece_owner, active,
            float(perception_range), float(fov_degrees), obstacles,
            other_xy, other_radius)
    return visibility_mask_torch(
        pose, pieces, piece_active, piece_owner, active,
        perception_range, fov_degrees, obstacles, other_xy, other_radius)


def visibility_mask_3v3(pose, pieces_xy, piece_active, piece_owner, active,
                        perception_range, fov_degrees, obstacles, robot_radius):
    """Fused per-piece visibility for all six observers; None means fallback."""
    extension = _hip_perception_extension() if _HIP_PERCEPTION_ENABLED else None
    tensors = (pose, pieces_xy, piece_active, piece_owner, active, obstacles,
               robot_radius)
    if (extension is not None and obstacles.shape[-1] in (3, 4) and
            pose.is_cuda and pose.dtype == torch.float32 and
            all(t.is_contiguous() and t.device == pose.device for t in tensors) and
            piece_active.dtype == torch.bool and piece_owner.dtype == torch.long and
            active.dtype == torch.bool and obstacles.dtype == torch.float32 and
            robot_radius.dtype == torch.float32):
        return extension.visibility_mask_3v3(
            pose, pieces_xy, piece_active, piece_owner, active,
            float(perception_range), float(fov_degrees), obstacles, robot_radius)
    # Batched Torch reference, also used on CUDA devices without the optional
    # HIP extension. Keep camera rows independent through geometric visibility.
    delta=pieces_xy[:,None]-pose[:,:,:2][:,:,None,:]
    distance=delta.norm(dim=-1)
    visible=(piece_active[:,None] & (piece_owner[:,None]<0) & active[:,None,None] &
             (distance<=float(perception_range)))
    half=math.radians(float(fov_degrees))*.5
    if half<math.pi:
        bearing=torch.atan2(delta[...,1],delta[...,0])
        error=torch.atan2(torch.sin(bearing-pose[:,:,None,2]),
                          torch.cos(bearing-pose[:,:,None,2])).abs()
        visible &= error<=half
    worlds,robots,pieces=visible.shape
    if obstacles.numel():
        blocked=piece_occlusion_mask_torch(
            pose[:,:,:2].reshape(worlds*robots,2),
            delta.reshape(worlds*robots,pieces,2),obstacles,
            visible.reshape(worlds*robots,pieces)).reshape_as(visible)
        visible &= ~blocked
    peer_xy=pose[:,:,:2]
    peer_delta=peer_xy[:,None]-pose[:,:,:2][:,:,None,:]
    denom=delta.square().sum(-1).clamp_min(1e-8)
    frac=(peer_delta[:,:,:,None,:]*delta[:,:,None,:,:]).sum(-1)/denom[:,:,None,:]
    closest=pose[:,:,:2][:,:,None,None,:]+frac.clamp(0.,1.)[...,None]*delta[:,:,None,:,:]
    peer_blocked=((closest-peer_xy[:,None,:,None,:]).norm(dim=-1)<=
                  robot_radius[:,None,:,None])
    peer_blocked &= (frac>.02)&(frac<.98)
    peer_blocked &= ~torch.eye(robots,device=pose.device,dtype=torch.bool)[None,:,:,None]
    return visible & ~peer_blocked.any(2)


def angular_cluster_detections(pose, positions, velocities, visible,
                               ball_diameter,half_fov=math.pi):
    """Apply adjacent angular occlusion and merge connected apparent groups.

    Inputs are bearing-sorted [world,robot,piece,...] tensors. The overlap
    sweep is O(n) after sorting, with segmented GPU reductions for clusters.
    It never returns simulator piece indices or exact group counts.
    """
    extension=_hip_perception_extension() if _HIP_PERCEPTION_ENABLED else None
    if (extension is not None and pose.is_cuda and pose.dtype==torch.float32 and
            positions.dtype==torch.float32 and velocities.dtype==torch.float32 and
            visible.dtype==torch.bool and pose.device==positions.device==
            velocities.device==visible.device==pose.device):
        lead=visible.shape[:-1]
        rows=math.prod(lead)
        packed=(extension.angular_cluster(
            pose.reshape(rows,3).contiguous(),
            positions.reshape(rows,positions.shape[-2],2).contiguous(),
            velocities.reshape(rows,velocities.shape[-2],2).contiguous(),
            visible.reshape(rows,visible.shape[-1]).contiguous(),
            float(ball_diameter),float(half_fov)))
        return tuple(value.reshape(*lead,*value.shape[1:]) for value in packed)
    pose_xy=pose[...,:2].unsqueeze(-2)
    pose_heading=pose[...,2].unsqueeze(-1)
    raw_delta=positions-pose_xy
    raw_bearing=(torch.atan2(raw_delta[...,1],raw_delta[...,0])-
                 pose_heading)
    raw_bearing=torch.atan2(torch.sin(raw_bearing),torch.cos(raw_bearing))
    order=torch.argsort(torch.where(visible,raw_bearing,float("inf")),dim=-1)
    gather_xy=order[...,None].expand(*order.shape,2)
    positions=positions.gather(-2,gather_xy)
    velocities=velocities.gather(-2,gather_xy)
    visible=visible.gather(-1,order)
    delta=positions-pose_xy
    distance=delta.norm(dim=-1).clamp_min(1e-4)
    bearing=torch.atan2(delta[...,1],delta[...,0])-pose_heading
    bearing=torch.atan2(torch.sin(bearing),torch.cos(bearing))
    half_width=torch.atan(torch.full_like(distance,ball_diameter*.5)/distance)
    lo=(bearing-half_width).clamp(min=-half_fov,max=half_fov)
    hi=(bearing+half_width).clamp(min=-half_fov,max=half_fov)
    left_overlap=torch.zeros_like(distance)
    right_overlap=torch.zeros_like(distance)
    left_nearer=torch.zeros_like(visible)
    right_nearer=torch.zeros_like(visible)
    pair_overlap=(torch.minimum(hi[...,:-1],hi[...,1:])-
                  torch.maximum(lo[...,:-1],lo[...,1:])).clamp_min(0.)
    pair_valid=visible[...,:-1]&visible[...,1:]
    pair_overlap*=pair_valid
    left_overlap[...,1:]=pair_overlap
    right_overlap[...,:-1]=pair_overlap
    left_nearer[...,1:]=distance[...,:-1]<distance[...,1:]
    right_nearer[...,:-1]=distance[...,1:]<distance[...,:-1]
    covered=(left_overlap*left_nearer+right_overlap*right_nearer)
    fraction=(1.-covered/(hi-lo).clamp_min(1e-5)).clamp(0.,1.)
    fraction=torch.where(visible,fraction,0.)
    connected=(pair_overlap>0.) & ((distance[...,:-1]-distance[...,1:]).abs()<=.20)
    starts=visible.clone()
    starts[...,1:] &= ~connected
    labels=(starts.long().cumsum(-1)-1).clamp_min(0)
    valid_flat=visible.reshape(-1,visible.shape[-1])
    label_flat=labels.reshape_as(valid_flat)
    frac_flat=fraction.reshape_as(valid_flat)
    contributes=valid_flat & (frac_flat>.05)
    pos_flat=positions.reshape(-1,positions.shape[-2],2)
    vel_flat=velocities.reshape_as(pos_flat)
    rows,slots=valid_flat.shape
    row=torch.arange(rows,device=positions.device)[:,None].expand(rows,slots)
    weight=valid_flat.to(positions.dtype)*frac_flat
    denom=torch.zeros((rows,slots),device=positions.device,dtype=positions.dtype)
    denom.scatter_add_(1,label_flat,weight)
    pos_sum=torch.zeros((rows,slots,2),device=positions.device,dtype=positions.dtype)
    vel_sum=torch.zeros_like(pos_sum)
    pos_sum.scatter_add_(1,label_flat[...,None].expand(-1,-1,2),
                         pos_flat*weight[...,None])
    vel_sum.scatter_add_(1,label_flat[...,None].expand(-1,-1,2),
                         vel_flat*weight[...,None])
    group_valid=denom>0.
    group_pos=pos_sum/denom.clamp_min(1e-6)[...,None]
    group_vel=vel_sum/denom.clamp_min(1e-6)[...,None]
    # Average visible fraction is a proxy for group detection quality; avoid
    # exposing the simulator's number of members as an exact count.
    members=torch.zeros_like(denom)
    members.scatter_add_(1,label_flat,contributes.to(positions.dtype))
    group_fraction=denom/members.clamp_min(1.)
    inf=torch.full_like(pos_flat,float("inf"))
    ninf=torch.full_like(pos_flat,-float("inf"))
    extent_min=torch.full_like(pos_sum,float("inf"))
    extent_max=torch.full_like(pos_sum,-float("inf"))
    index_xy=label_flat[...,None].expand(-1,-1,2)
    extent_min.scatter_reduce_(1,index_xy,
        torch.where(contributes[...,None],pos_flat,inf),reduce="amin",include_self=True)
    extent_max.scatter_reduce_(1,index_xy,
        torch.where(contributes[...,None],pos_flat,ninf),reduce="amax",include_self=True)
    spatial_extent=(extent_max-extent_min+ball_diameter).clamp_min(0.)
    angular_min=torch.full_like(denom,float("inf"))
    angular_max=torch.full_like(denom,-float("inf"))
    angular_min.scatter_reduce_(1,label_flat,
        torch.where(contributes,lo.reshape_as(valid_flat),float("inf")),
        reduce="amin",include_self=True)
    angular_max.scatter_reduce_(1,label_flat,
        torch.where(contributes,hi.reshape_as(valid_flat),-float("inf")),
        reduce="amax",include_self=True)
    angular_extent=(angular_max-angular_min).clamp_min(0.)
    merged=members>1.
    shape=visible.shape
    return (group_pos.reshape(*shape,2),group_vel.reshape(*shape,2),
            group_valid.reshape(shape),group_fraction.reshape(shape),
            spatial_extent.reshape(*shape,2),angular_extent.reshape(shape),
            merged.reshape(shape))


def fuse_camera_cluster_detections(pose, positions, velocities, visible,
                                   confidence, spatial_extent,
                                   angular_extent, merged, ball_diameter,
                                   half_fov):
    """Fuse already clustered camera detections without discarding metadata."""
    extension=_hip_perception_extension() if _HIP_PERCEPTION_ENABLED else None
    tensors=(pose,positions,velocities,visible,confidence,spatial_extent,
             angular_extent,merged)
    if (extension is not None and pose.is_cuda and pose.dtype==torch.float32 and
            all(t.is_contiguous() and t.device==pose.device for t in tensors)):
        lead=visible.shape[:-1]
        rows=math.prod(lead)
        packed=extension.fuse_camera_clusters(
            pose.reshape(rows,3).contiguous(),
            positions.reshape(rows,positions.shape[-2],2).contiguous(),
            velocities.reshape(rows,velocities.shape[-2],2).contiguous(),
            visible.reshape(rows,visible.shape[-1]).contiguous(),
            confidence.reshape(rows,confidence.shape[-1]).contiguous(),
            spatial_extent.reshape(rows,spatial_extent.shape[-2],2).contiguous(),
            angular_extent.reshape(rows,angular_extent.shape[-1]).contiguous(),
            merged.reshape(rows,merged.shape[-1]).contiguous(),
            float(ball_diameter),float(half_fov))
        return tuple(value.reshape(*lead,*value.shape[1:]) for value in packed)
    # Torch fallback performs the same post-camera interval fusion, carrying
    # extents and confidence from each contributing cluster.
    lead=visible.shape[:-1]; slots=visible.shape[-1]; rows=math.prod(lead)
    pos=positions.reshape(rows,slots,2); vel=velocities.reshape_as(pos)
    mask=visible.reshape(rows,slots); conf=confidence.reshape(rows,slots)
    spatial=spatial_extent.reshape(rows,slots,2)
    angular=angular_extent.reshape(rows,slots); was_merged=merged.reshape(rows,slots)
    p=pose.reshape(rows,3)
    delta=pos-p[:,:2][:,None,:]
    distance=delta.norm(dim=-1).clamp_min(1e-4)
    bearing=torch.atan2(delta[...,1],delta[...,0])-p[:,None,2]
    bearing=torch.atan2(torch.sin(bearing),torch.cos(bearing))
    order=torch.argsort(torch.where(mask,bearing,float("inf")),dim=-1)
    gather_xy=order[...,None].expand(-1,-1,2)
    pos=pos.gather(1,gather_xy); vel=vel.gather(1,gather_xy)
    conf=conf.gather(1,order); spatial=spatial.gather(1,gather_xy)
    angular=angular.gather(1,order); was_merged=was_merged.gather(1,order)
    distance=distance.gather(1,order); bearing=bearing.gather(1,order)
    mask=mask.gather(1,order)
    lo=bearing-angular*.5; hi=bearing+angular*.5
    overlap=(torch.minimum(hi[:,:-1],hi[:,1:])-
             torch.maximum(lo[:,:-1],lo[:,1:])).clamp_min(0.)
    join=(overlap>0.) & ((distance[:,:-1]-distance[:,1:]).abs()<=.20)
    starts=mask.clone(); starts[:,1:] &= ~join
    labels=(starts.long().cumsum(-1)-1).clamp_min(0)
    contributes=mask & (conf>.01)
    weight=conf*contributes.to(conf.dtype)
    denom=torch.zeros_like(conf); denom.scatter_add_(1,labels,weight)
    pos_sum=torch.zeros_like(pos); vel_sum=torch.zeros_like(vel)
    pos_sum.scatter_add_(1,labels[...,None].expand(-1,-1,2),pos*weight[...,None])
    vel_sum.scatter_add_(1,labels[...,None].expand(-1,-1,2),vel*weight[...,None])
    group_pos=pos_sum/denom.clamp_min(1e-6)[...,None]
    group_vel=vel_sum/denom.clamp_min(1e-6)[...,None]
    members=torch.zeros_like(conf)
    members.scatter_add_(1,labels,contributes.to(conf.dtype))
    inf=torch.full_like(pos,float("inf")); ninf=-inf
    extent_min=torch.full_like(pos,float("inf")); extent_max=-extent_min
    extent_min.scatter_reduce_(1,labels[...,None].expand(-1,-1,2),
        torch.where(contributes[...,None],pos-spatial*.5,inf),reduce="amin",include_self=True)
    extent_max.scatter_reduce_(1,labels[...,None].expand(-1,-1,2),
        torch.where(contributes[...,None],pos+spatial*.5,ninf),reduce="amax",include_self=True)
    group_spatial=(extent_max-extent_min).clamp_min(0.)
    conf_out=torch.zeros_like(conf)
    conf_out.scatter_reduce_(1,labels,torch.where(contributes,conf,0.),
                             reduce="amax",include_self=True)
    angle_lo=torch.full_like(conf,float("inf")); angle_hi=-angle_lo
    angle_lo.scatter_reduce_(1,labels,torch.where(contributes,lo,float("inf")),
                             reduce="amin",include_self=True)
    angle_hi.scatter_reduce_(1,labels,torch.where(contributes,hi,-float("inf")),
                             reduce="amax",include_self=True)
    merged_value=torch.zeros_like(conf)
    merged_value.scatter_reduce_(1,labels,
        torch.where(contributes,was_merged.to(conf.dtype),0.),reduce="amax",include_self=True)
    out_valid=denom>0.
    out_merged=(merged_value>0.) | (members>1.)
    shape=visible.shape
    return (group_pos.reshape(*shape,2),group_vel.reshape(*shape,2),
            out_valid.reshape(shape),conf_out.reshape(shape),
            group_spatial.reshape(*shape,2),(angle_hi-angle_lo).clamp_min(0.).reshape(shape),
            out_merged.reshape(shape))


def angular_cluster_camera_detections(camera_pose,positions,velocities,visible,
                                      ball_diameter,half_fov,owner=None):
    """Cluster per-camera observations while reading shared world geometry."""
    extension=_hip_perception_extension() if _HIP_PERCEPTION_ENABLED else None
    if (extension is not None and camera_pose.is_cuda and
            camera_pose.dtype==torch.float32 and positions.is_contiguous() and
            velocities.is_contiguous() and visible.is_contiguous() and
            camera_pose.is_contiguous() and positions.device==camera_pose.device and
            velocities.device==camera_pose.device and visible.device==camera_pose.device):
        worlds,robots,cameras,pieces=visible.shape
        key=(worlds,robots,cameras,pieces,camera_pose.device)
        cached=getattr(owner,"_camera_cluster_buffers",None) if owner is not None else None
        if cached is None or cached[0]!=key:
            shape=(worlds,robots,cameras,pieces)
            outputs=(torch.empty((*shape,2),device=camera_pose.device),
                     torch.empty((*shape,2),device=camera_pose.device),
                     torch.empty(shape,device=camera_pose.device,dtype=torch.bool),
                     torch.empty(shape,device=camera_pose.device),
                     torch.empty((*shape,2),device=camera_pose.device),
                     torch.empty(shape,device=camera_pose.device),
                     torch.empty(shape,device=camera_pose.device,dtype=torch.bool))
            cached=(key,outputs)
            if owner is not None: owner._camera_cluster_buffers=cached
        _,outputs=cached
        extension.angular_cluster_cameras(camera_pose,positions,velocities,visible,
            *outputs,float(ball_diameter),float(half_fov))
        return outputs
    worlds,robots,cameras,pieces=visible.shape
    pos=positions[:,None,None].expand(-1,robots,cameras,-1,-1)
    vel=velocities[:,None,None].expand_as(pos)
    return angular_cluster_detections(camera_pose,pos,vel,visible,ball_diameter,half_fov)


def stereo_fusion_pack(pose, per_camera, owner, pieces):
    """Fuse stereo detections and pack a bearing-ordered tracker input in HIP.

    Pair and packed output buffers are retained on the environment and reused
    across perception ticks. Returns None when the HIP extension is unavailable.
    """
    extension=_hip_perception_extension() if _HIP_PERCEPTION_ENABLED else None
    if (extension is None or not pose.is_cuda or pose.dtype!=torch.float32 or
            not all(t.is_contiguous() and t.device==pose.device for t in per_camera)):
        return None
    worlds=pose.shape[0]; key=(worlds,int(pieces),pose.device)
    buffers=getattr(owner,"_stereo_fusion_buffers",None)
    if buffers is None or buffers[0]!=key:
        if pieces<=512:
            pair=(torch.empty((0,),device=pose.device),
                  torch.empty((0,),device=pose.device),
                  torch.empty((0,),device=pose.device,dtype=torch.bool),
                  torch.empty((0,),device=pose.device),
                  torch.empty((0,),device=pose.device),
                  torch.empty((0,),device=pose.device),
                  torch.empty((0,),device=pose.device,dtype=torch.bool))
        else:
            pair=(torch.empty((worlds,6,2,pieces,2),device=pose.device),
                  torch.empty((worlds,6,2,pieces,2),device=pose.device),
                  torch.empty((worlds,6,2,pieces),device=pose.device,dtype=torch.bool),
                  torch.empty((worlds,6,2,pieces),device=pose.device),
                  torch.empty((worlds,6,2,pieces,2),device=pose.device),
                  torch.empty((worlds,6,2,pieces),device=pose.device),
                  torch.empty((worlds,6,2,pieces),device=pose.device,dtype=torch.bool))
        packed=(torch.empty((worlds,6,pieces,2),device=pose.device),
                torch.empty((worlds,6,pieces,2),device=pose.device),
                torch.empty((worlds,6,pieces),device=pose.device,dtype=torch.bool),
                torch.empty((worlds,6,pieces),device=pose.device),
                torch.empty((worlds,6,pieces,2),device=pose.device),
                torch.empty((worlds,6,pieces),device=pose.device),
                torch.empty((worlds,6,pieces),device=pose.device,dtype=torch.bool))
        buffers=(key,pair,packed)
        owner._stereo_fusion_buffers=buffers
    _,pair,packed=buffers
    extension.stereo_fusion_pack(pose,*per_camera,*pair,*packed)
    return packed


def perception_commit_3v3(visible, active, ticks, piece_pos, piece_vel,
                          track_pos, track_vel, track_age, track_mask,
                          seed, dt, dropout, position_noise, velocity_noise,
                          track_timeout=2.0):
    """Fuse 3v3 per-piece sensor noise and track writes when HIP is available."""
    extension = _hip_perception_extension() if _HIP_PERCEPTION_ENABLED else None
    tensors = (visible, active, ticks, piece_pos, piece_vel, track_pos,
               track_vel, track_age, track_mask)
    if (extension is None or not visible.is_cuda or visible.dtype != torch.bool or
            not all(t.is_contiguous() and t.device == visible.device for t in tensors)):
        return False
    if (active.dtype != torch.bool or ticks.dtype != torch.int64 or
            piece_pos.dtype != torch.float32 or piece_vel.dtype != torch.float32 or
            track_pos.dtype != torch.float32 or track_vel.dtype != torch.float32 or
            track_age.dtype != torch.float32 or track_mask.dtype != torch.bool):
        return False
    extension.perception_commit_3v3(
        visible, active, ticks, piece_pos, piece_vel, track_pos, track_vel,
        track_age, track_mask, int(seed), float(dt), float(dropout),
        float(position_noise), float(velocity_noise),float(track_timeout))
    return True


def anonymous_track_update(observer, detections, velocities, detected,
                           confidence, spatial_extent, angular_extent, merged,
                           active, ticks, track_pos, track_vel, track_age,
                           track_mask, quality_state, track_confidence,
                           track_spatial_extent, track_angular_extent,
                           track_merged, current_visibility, seed, dt, dropout,
                           position_noise, velocity_noise, timeout, gate,
                           ball_diameter):
    """Run anonymous association, sensor quality/noise and track writes in HIP."""
    extension=_hip_perception_extension() if _HIP_PERCEPTION_ENABLED else None
    tensors=(observer,detections,velocities,detected,confidence,spatial_extent,
             angular_extent,merged,active,ticks,track_pos,track_vel,track_age,
             track_mask,quality_state,track_confidence,track_spatial_extent,
             track_angular_extent,track_merged,current_visibility)
    if (extension is None or not observer.is_cuda or observer.dtype!=torch.float32 or
            not all(t.is_contiguous() and t.device==observer.device for t in tensors)):
        return False
    extension.anonymous_track_update(
        observer,detections,velocities,detected,confidence,spatial_extent,
        angular_extent,merged,active,ticks,track_pos,track_vel,track_age,
        track_mask,quality_state,track_confidence,track_spatial_extent,
        track_angular_extent,track_merged,current_visibility,int(seed),float(dt),
        float(dropout),float(position_noise),float(velocity_noise),float(timeout),
        float(gate),float(ball_diameter))
    return True


def opponent_tracks_3v3(pose, length, width, acceleration, robot_radius,
                       obstacles, active, controlled, defense_role, ticks,
                       opponent_pose, opponent_velocity, opponent_size,
                       opponent_age, opponent_valid, seed, perception_range,
                       fov_degrees, dropout, position_noise, velocity_noise, dt):
    """Fuse field/chassis occlusion, nearest-opponent choice, and track writes."""
    extension = _hip_perception_extension() if _HIP_PERCEPTION_ENABLED else None
    tensors = (pose, length, width, acceleration, robot_radius, obstacles, active,
               controlled, defense_role, ticks, opponent_pose, opponent_velocity,
               opponent_size, opponent_age, opponent_valid)
    if (extension is None or not pose.is_cuda or pose.dtype != torch.float32 or
            not all(t.is_contiguous() and t.device == pose.device for t in tensors)):
        return False
    if (any(t.dtype != torch.float32 for t in
            (pose, length, width, acceleration, robot_radius, obstacles,
             opponent_pose, opponent_velocity, opponent_size, opponent_age)) or
            active.dtype != torch.bool or controlled.dtype != torch.bool or
            defense_role.dtype != torch.bool or ticks.dtype != torch.int64 or
            opponent_valid.dtype != torch.bool):
        return False
    extension.opponent_tracks_3v3(
        pose, length, width, acceleration, robot_radius, obstacles, active,
        controlled, defense_role, ticks, opponent_pose, opponent_velocity,
        opponent_size, opponent_age, opponent_valid, int(seed),
        float(perception_range), float(fov_degrees), float(dropout),
        float(position_noise), float(velocity_noise), float(dt))
    return True


def age_tracks_3v3(track_pos, track_vel, track_age, track_mask,
                   opponent_age, opponent_valid, active, dt, timeout):
    """Age known fuel and opponent tracks in one in-place HIP kernel."""
    if not FUSED_TRACK_AGING_HIP_ENABLED:
        return False
    extension = _hip_perception_extension()
    tensors = (track_pos, track_vel, track_age, track_mask, opponent_age,
               opponent_valid, active)
    if (extension is None or not track_pos.is_cuda or
            track_pos.dtype != torch.float32 or
            not all(t.is_contiguous() and t.device == track_pos.device
                    for t in tensors)):
        return False
    if (any(t.dtype != torch.float32 for t in
            (track_pos, track_vel, track_age, opponent_age)) or
            track_mask.dtype != torch.bool or opponent_valid.dtype != torch.bool or
            active.dtype != torch.bool):
        return False
    extension.age_tracks_3v3(track_pos, track_vel, track_age, track_mask,
                             opponent_age, opponent_valid, active,
                             float(dt), float(timeout))
    return True
