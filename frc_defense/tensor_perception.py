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
    if (extension is not None and pose_xy.is_cuda and pose_xy.dtype == torch.float32
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
    if (extension is None or not pose.is_cuda or pose.dtype != torch.float32 or
            not all(t.is_contiguous() and t.device == pose.device for t in tensors) or
            piece_active.dtype != torch.bool or piece_owner.dtype != torch.long or
            active.dtype != torch.bool or obstacles.dtype != torch.float32 or
            robot_radius.dtype != torch.float32):
        return None
    return extension.visibility_mask_3v3(
        pose, pieces_xy, piece_active, piece_owner, active,
        float(perception_range), float(fov_degrees), obstacles, robot_radius)


def perception_commit_3v3(visible, active, ticks, piece_pos, piece_vel,
                          track_pos, track_vel, track_age, track_mask,
                          seed, dt, dropout, position_noise, velocity_noise):
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
        float(position_noise), float(velocity_noise))
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
