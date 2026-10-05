"""Optional HIP path for committing perception tracks after Torch RNG.

Sensor visibility, dropout, and every RNG/noise draw stay on the current Torch
path; only the ordinary (no reset mask) in-place track-state commit is fused.
The production simulator uses this only when its explicit opt-in is enabled.
"""
from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import torch

_EXTENSION = None
_ATTEMPTED = False
FUSED_PERCEPTION_COMMIT_HIP_ENABLED = (
    os.environ.get("AUTODRIVE_FUSED_PERCEPTION_COMMIT_HIP", "1") != "0")


def extension():
    global _EXTENSION, _ATTEMPTED
    if _ATTEMPTED:
        return _EXTENSION
    _ATTEMPTED = True
    if not (torch.cuda.is_available() and torch.version.hip):
        return None
    try:
        from torch.utils import cpp_extension

        bundled = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
        rocm = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
                    (bundled if bundled.exists() else cpp_extension._find_rocm_home())).resolve()
        torch_lib = (Path(torch.__file__).parent / "lib").resolve()

        def alias(path: Path, alias_path: Path) -> Path:
            if " " not in str(path):
                return path
            if alias_path.is_symlink() and alias_path.resolve() != path:
                alias_path.unlink()
            if not alias_path.exists():
                alias_path.symlink_to(path, target_is_directory=True)
            return alias_path

        rocm_alias = alias(rocm, Path("/tmp/autodrive-rocm-sdk"))
        lib_alias = alias(torch_lib, Path("/tmp/autodrive-torch-lib"))
        if rocm_alias != rocm:
            os.environ.setdefault("ROCM_HOME", str(rocm_alias))
            os.environ.setdefault("HIP_CLANG_PATH", str(rocm_alias / "lib/llvm/bin"))
        old = (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME,
               cpp_extension.HIP_HOME)
        cpp_extension.TORCH_LIB_PATH = str(lib_alias)
        cpp_extension.ROCM_HOME = str(rocm_alias)
        cpp_extension.HIP_HOME = str(rocm_alias / "hip")
        source = Path(__file__).parent / "csrc"
        device_lib = next((p for p in (rocm_alias / "lib/llvm/amdgcn/bitcode",
                                       rocm_alias / "amdgcn/bitcode") if p.is_dir()), None)
        cuda_flags = ["-O3", "-ffp-contract=off"]
        if device_lib is not None:
            cuda_flags.append(f"--rocm-device-lib-path={device_lib}")
        try:
            _EXTENSION = cpp_extension.load(
                name="autodrive_tensor_perception_commit_proto_hip",
                sources=[str(source / "tensor_perception_commit_proto.cpp"),
                         str(source / "tensor_perception_commit_proto_kernel.cu")],
                with_cuda=True, extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags, verbose=False)
        finally:
            (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME,
             cpp_extension.HIP_HOME) = old
    except Exception as exc:
        warnings.warn(f"Perception commit prototype unavailable: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXTENSION


def fused_commit_available(device):
    """Whether the opt-in fused commit is available for this device."""
    if (not FUSED_PERCEPTION_COMMIT_HIP_ENABLED or torch.version.hip is None or
            torch.device(device).type != "cuda"):
        return False
    return extension() is not None


def update_perception(env, active_mask=None, active_count=None):
    """Prototype equivalent of `_update_perception(reset_mask=None, ...)`.

    Requires HIP and mutates only the same perception track tensors as the
    reference method. The sensor/RNG calculations intentionally mirror the
    current Torch implementation and run before the fused commit.
    """
    ext = extension()
    if ext is None or env.device.type != "cuda":
        raise RuntimeError("perception commit prototype requires HIP")
    if active_mask is None:
        active_mask = torch.ones(env.n, device=env.device, dtype=torch.bool)
    else:
        active_mask = torch.as_tensor(active_mask, device=env.device,
                                      dtype=torch.bool).reshape(env.n)

    visible_rows = []
    measured_rows = []
    velocity_rows = []
    detected_rows = []
    opponent_pose_rows = []
    opponent_velocity_rows = []
    dt = env.dt
    occ = env.field_feature_obstacles
    for robot in range(2):
        pose = env.sim.pose[:, robot]
        delta = env.piece_pos - pose[:, None, :2]
        other = 1 - robot
        other_pos = env.sim.pose[:, other, :2]
        other_radius = .5 * torch.sqrt(env.sim.length[:, other].square() +
                                       env.sim.width[:, other].square())
        from .tensor_perception import visibility_mask
        visible = visibility_mask(
            pose.contiguous(), env.piece_pos.contiguous(),
            env.piece_active.contiguous(), env.piece_owner.contiguous(),
            active_mask.contiguous(), env.perception_range, env.perception_fov,
            occ.contiguous(), other_pos.contiguous(), other_radius.contiguous())
        if env.perception_dropout:
            detected_pieces = env._random_active(
                active_mask, visible.shape[1:], active_count=active_count) >= env.perception_dropout
            visible &= detected_pieces
        noise = env._random_active(active_mask, env.piece_pos.shape[1:], normal=True,
                                   active_count=active_count) * env.perception_position_noise
        measured = env.piece_pos + noise
        updated_velocity = env.piece_vel.clone()
        if env.perception_velocity_noise:
            updated_velocity += env._random_active(
                active_mask, env.piece_vel.shape[1:], normal=True,
                active_count=active_count) * env.perception_velocity_noise
        visible_rows.append(visible & active_mask[:, None])
        measured_rows.append(measured)
        velocity_rows.append(updated_velocity)

        observed_pose = env.sim.pose[:, other].clone()
        observed_velocity = env.sim.velocity[:, other].clone()
        robot_delta = observed_pose[:, :2] - env.sim.pose[:, robot, :2]
        robot_bearing = torch.atan2(robot_delta[:, 1], robot_delta[:, 0])
        bearing_error = torch.atan2(
            torch.sin(robot_bearing - env.sim.pose[:, robot, 2]),
            torch.cos(robot_bearing - env.sim.pose[:, robot, 2])).abs()
        detected = (env._random_active(active_mask, (), active_count=active_count) >=
                    env.perception_dropout)
        detected &= robot_delta.norm(dim=-1) <= env.perception_range
        detected &= bearing_error <= math.radians(env.perception_fov) * .5
        if occ.numel():
            denom = robot_delta.square().sum(-1).clamp_min(1e-8)
            ray = occ[None, :, :2] - env.sim.pose[:, robot, None, :2]
            t = (ray * robot_delta[:, None, :]).sum(-1) / denom[:, None]
            closest = (env.sim.pose[:, robot, None, :2] +
                       t.clamp(0., 1.)[..., None] * robot_delta[:, None, :])
            obstacle_radius = occ[None, :, 2] + .03
            blocked = ((obstacle_radius >= 0) &
                       ((closest - occ[None, :, :2]).square().sum(-1) <=
                        obstacle_radius.square()))
            blocked &= (t > .02) & (t < .98)
            detected &= ~blocked.any(-1)
        position_noise = env._random_active(
            active_mask, (2,), normal=True, active_count=active_count) * env.perception_position_noise
        heading_noise = env._random_active(
            active_mask, (), normal=True, active_count=active_count) * min(
                .05, env.perception_position_noise)
        velocity_noise = env._random_active(
            active_mask, (3,), normal=True, active_count=active_count) * env.perception_velocity_noise
        observed_pose[:, :2] += position_noise
        observed_pose[:, 2] += heading_noise
        observed_velocity += velocity_noise
        detected &= active_mask
        detected_rows.append(detected)
        opponent_pose_rows.append(observed_pose)
        opponent_velocity_rows.append(observed_velocity)

    ext.commit(
        active_mask.contiguous(), torch.stack(visible_rows, dim=1).contiguous(),
        torch.stack(measured_rows, dim=1).contiguous(),
        torch.stack(velocity_rows, dim=1).contiguous(),
        torch.stack(detected_rows, dim=1).contiguous(),
        torch.stack(opponent_pose_rows, dim=1).contiguous(),
        torch.stack(opponent_velocity_rows, dim=1).contiguous(),
        env._track_pos, env._track_vel, env._track_mask, env._track_age,
        env._opponent_track_pose, env._opponent_track_velocity,
        env._opponent_track_valid, env._opponent_track_age,
        float(dt), float(env.perception_track_timeout))
