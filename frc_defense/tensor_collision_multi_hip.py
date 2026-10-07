"""Fused six-robot SAT contact resolution for HIP simulator batches."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

_EXT = None
_ATTEMPTED = False
ENABLED = os.environ.get("AUTODRIVE_FUSED_ROBOT_COLLISION_6_HIP", "1") != "0"


def _extension():
    global _EXT, _ATTEMPTED
    if not ENABLED or _ATTEMPTED:
        return _EXT
    _ATTEMPTED = True
    if not (torch.cuda.is_available() and torch.version.hip):
        return None
    try:
        from torch.utils import cpp_extension

        bundled = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
        rocm = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
                    (bundled if bundled.exists() else cpp_extension._find_rocm_home())).resolve()
        rocm_alias = rocm
        if " " in str(rocm):
            rocm_alias = Path("/tmp/autodrive-rocm-sdk")
            if rocm_alias.is_symlink() and rocm_alias.resolve() != rocm:
                rocm_alias.unlink()
            if not rocm_alias.exists():
                rocm_alias.symlink_to(rocm, target_is_directory=True)
            os.environ.setdefault("ROCM_HOME", str(rocm_alias))
            os.environ.setdefault("HIP_CLANG_PATH", str(rocm_alias / "lib/llvm/bin"))
        torch_lib = Path(torch.__file__).parent / "lib"
        if " " in str(torch_lib.resolve()):
            torch_alias = Path("/tmp/autodrive-torch-lib")
            if torch_alias.is_symlink() and torch_alias.resolve() != torch_lib.resolve():
                torch_alias.unlink()
            if not torch_alias.exists():
                torch_alias.symlink_to(torch_lib.resolve(), target_is_directory=True)
            torch_lib = torch_alias
        saved = (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME,
                 cpp_extension.HIP_HOME)
        cpp_extension.TORCH_LIB_PATH = str(torch_lib)
        cpp_extension.ROCM_HOME = str(rocm_alias)
        cpp_extension.HIP_HOME = str(rocm_alias / "hip")
        source_dir = Path(__file__).parent / "csrc"
        device_lib = next((p for p in (rocm_alias / "lib/llvm/amdgcn/bitcode",
                                        rocm_alias / "amdgcn/bitcode") if p.is_dir()), None)
        flags = ["-O3", "-ffp-contract=off"]
        if device_lib is not None:
            flags.append(f"--rocm-device-lib-path={device_lib}")
        try:
            _EXT = cpp_extension.load(
                name="autodrive_tensor_collision_multi_hip",
                sources=[str(source_dir / "tensor_collision_multi_hip.cpp"),
                         str(source_dir / "tensor_collision_multi_hip_kernel.cu")],
                with_cuda=True, extra_cflags=["-O3"], extra_cuda_cflags=flags,
                verbose=False)
        finally:
            (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME,
             cpp_extension.HIP_HOME) = saved
    except Exception as exc:
        warnings.warn(f"Fused six-robot collision HIP unavailable: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXT


def robot_contacts(sim, active_mask):
    """Resolve six-robot pair contacts in place; false requests Torch fallback."""
    if (not ENABLED or sim.device.type != "cuda" or not torch.version.hip or
            sim.num_robots != 6):
        return False
    active = torch.as_tensor(active_mask, device=sim.device, dtype=torch.bool)
    tensors = (sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
               sim.yaw_inertia_multiplier, sim.mu, active, sim.team_ids,
               sim.robot_contact, sim.opponent_contact)
    if any(t.device != sim.pose.device or not t.is_contiguous() for t in tensors):
        return False
    ext = _extension()
    if ext is None:
        return False
    ext.robot_contacts(sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
                       sim.yaw_inertia_multiplier, sim.mu, active, sim.team_ids,
                       sim.robot_contact, sim.opponent_contact)
    return True


def field_contacts(sim, active_mask):
    """Resolve six robots against field boxes in one HIP kernel."""
    if (not ENABLED or sim.device.type != "cuda" or not torch.version.hip or
            sim.num_robots != 6 or not sim.field_colliders.numel()):
        return False
    active = torch.as_tensor(active_mask, device=sim.device,
                             dtype=torch.bool).reshape(sim.n).contiguous()
    tensors = (sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
               sim.yaw_inertia_multiplier, sim.wall_mu, sim.field_colliders,
               active, sim.field_contact)
    if any(t.device != sim.pose.device or not t.is_contiguous() for t in tensors):
        return False
    ext = _extension()
    if ext is None or not hasattr(ext, "field_contacts"):
        return False
    ext.field_contacts(sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
                       sim.yaw_inertia_multiplier, sim.wall_mu,
                       sim.field_colliders, active, sim.field_contact)
    sim.robot_contact |= sim.field_contact.any(-1)
    return True


def field_sweep_contacts(sim, active_mask, sweep_steps, substep_dt):
    """Integrate and resolve a full six-robot field sweep in one HIP launch."""
    if (not ENABLED or sim.device.type != "cuda" or not torch.version.hip or
            sim.num_robots != 6 or not sim.field_colliders.numel()):
        return False
    active = torch.as_tensor(active_mask, device=sim.device,
                             dtype=torch.bool).reshape(sim.n).contiguous()
    tensors = (sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
               sim.yaw_inertia_multiplier, sim.wall_mu, sim.field_colliders,
               active, sim.field_contact, sim.robot_contact)
    if any(t.device != sim.pose.device or not t.is_contiguous() for t in tensors):
        return False
    ext = _extension()
    if ext is None or not hasattr(ext, "field_sweep_contacts"):
        return False
    ext.field_sweep_contacts(
        sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
        sim.yaw_inertia_multiplier, sim.wall_mu, sim.field_colliders,
        active, sim.field_contact, int(sweep_steps), float(substep_dt))
    sim.robot_contact |= sim.field_contact.any(-1)
    return True


def safe_score_targets(sim, score_target, robot_radius, x_min, x_max,
                       boxes, grid_points, grid_clear):
    """Choose the nearest legal score pose for six robots in one HIP launch."""
    if (not ENABLED or sim.device.type != "cuda" or not torch.version.hip or
            sim.num_robots != 6 or not boxes.numel()):
        return None
    tensors=(sim.pose,score_target,robot_radius,x_min,x_max,boxes,
             grid_points,grid_clear)
    if any(t.device != sim.pose.device or not t.is_contiguous() for t in tensors):
        return None
    ext=_extension()
    if ext is None or not hasattr(ext,"safe_score_targets"):
        return None
    return ext.safe_score_targets(
        sim.pose,score_target,robot_radius,x_min,x_max,boxes,
        grid_points,grid_clear,float(sim.field_width))


def avoid_robot_contention(sim, command, targets, active, winner, controlled,
                           teammate_intent_knowledge):
    """Fuse all 15 pair checks and per-robot avoidance reductions on HIP."""
    if (not ENABLED or sim.device.type != "cuda" or not torch.version.hip or
            sim.num_robots != 6):
        return None
    active = torch.as_tensor(active, device=sim.device,
                             dtype=torch.bool).reshape(sim.n).contiguous()
    tensors = (sim.pose, command, targets, sim.length, sim.width, sim.speed,
               controlled, sim.team_ids, active, winner)
    if any(t.device != sim.pose.device for t in tensors):
        return None
    command = command.contiguous()
    targets = targets.contiguous()
    controlled = controlled.contiguous()
    ext = _extension()
    if ext is None or not hasattr(ext, "avoid_robot_contention"):
        return None
    return ext.avoid_robot_contention(
        sim.pose, command, targets, sim.length, sim.width, sim.speed,
        controlled, sim.team_ids, active, winner,
        bool(teammate_intent_knowledge))
