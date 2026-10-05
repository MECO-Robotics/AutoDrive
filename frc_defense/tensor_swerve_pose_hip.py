"""Optional HIP fusion for swerve dynamics plus pose integration.

The production simulator loads this lazily on HIP for supported two- and
six-robot batches; the Torch path remains available as a fallback.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch

_EXT = None


def extension():
    """Build/load the experimental extension on demand (never at import time)."""
    global _EXT
    if _EXT is not None:
        return _EXT
    if not (torch.cuda.is_available() and torch.version.hip):
        raise RuntimeError("experimental swerve fusion requires a HIP device")
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
        alias = Path("/tmp/autodrive-torch-lib")
        if alias.is_symlink() and alias.resolve() != torch_lib.resolve():
            alias.unlink()
        if not alias.exists():
            alias.symlink_to(torch_lib.resolve(), target_is_directory=True)
        torch_lib = alias
    old_paths = (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME,
                 cpp_extension.HIP_HOME)
    cpp_extension.TORCH_LIB_PATH = str(torch_lib)
    cpp_extension.ROCM_HOME = str(rocm_alias)
    cpp_extension.HIP_HOME = str(rocm_alias / "hip")
    source_dir = Path(__file__).parent / "csrc"
    device_lib = next((p for p in (rocm_alias / "lib/llvm/amdgcn/bitcode",
                                  rocm_alias / "amdgcn/bitcode") if p.is_dir()), None)
    cuda_flags = ["-O3", "-ffp-contract=off"]
    if device_lib is not None:
        cuda_flags.append(f"--rocm-device-lib-path={device_lib}")
    try:
        _EXT = cpp_extension.load(
            name="autodrive_tensor_swerve_pose_hip",
            sources=[str(source_dir / "tensor_swerve_pose_hip.cpp"),
                     str(source_dir / "tensor_swerve_pose_kernel.cu")],
            with_cuda=True, extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_flags, verbose=False)
    finally:
        (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME,
         cpp_extension.HIP_HOME) = old_paths
    return _EXT


def step(sim, command, active_mask, *, debug=True):
    """Run fused swerve + pose update; debug storage is optional."""
    if sim.device.type != "cuda" or torch.version.hip is None:
        raise RuntimeError("experimental swerve fusion requires HIP tensors")
    active_mask = torch.as_tensor(active_mask, device=sim.device,
                                  dtype=torch.bool).reshape(sim.n).contiguous()
    command = torch.as_tensor(command, device=sim.device,
                              dtype=sim.pose.dtype).contiguous()
    if command.shape == (sim.num_robots, 3):
        command = command.expand(sim.n, sim.num_robots, 3).contiguous()
    if command.shape != (sim.n, sim.num_robots, 3):
        raise ValueError(f"command must be {(sim.n, sim.num_robots, 3)}")
    command = torch.nan_to_num(command)
    debug_tensor = (torch.empty((sim.n, sim.num_robots, 4, 26), device=sim.device,
                                dtype=sim.pose.dtype) if debug else
                    torch.empty((0,), device=sim.device, dtype=sim.pose.dtype))
    p = sim.swerve
    resistance = p.nominal_voltage / p.stall_current_a
    stall = p.free_speed_rpm * 2 * math.pi / 60
    kv = stall / (p.nominal_voltage - resistance * p.free_current_a)
    kt = p.stall_torque_nm / p.stall_current_a
    inverse_wheel_radius = 1 / p.wheel_radius
    inverse_steer_current_limit = 1 / max(p.steer_current_limit, 1.0e-6)
    inverse_resistance = 1 / resistance
    inverse_stall = 1 / stall
    inverse_kv = 1 / kv
    inverse_four_dt = 1 / (4 * sim.dt)
    extension().step(
        sim.pose, sim.velocity, command, active_mask,
        sim.length, sim.width, sim.mass, sim.speed, sim.accel,
        sim.omega_limit, sim.alpha, sim.ground_mu, sim.lateral_mu,
        sim.yaw_inertia_multiplier, sim.drive_ratio,
        sim.drive_current_limit, sim.drive_supply_limit,
        sim.robot_supply_limit, sim.battery_resistance,
        sim.module_angle, sim.module_steer_rate, sim.module_drive_speed,
        sim.module_current, sim.module_supply_current, sim.robot_current,
        sim.bump_regions.contiguous(), debug_tensor, sim.dt,
        p.module_x_offset, p.module_y_offset, inverse_wheel_radius,
        p.steer_current_limit, p.steer_rate_limit, p.steer_acceleration,
        inverse_steer_current_limit, inverse_resistance, inverse_stall,
        inverse_kv, kt, p.free_current_a, p.motor_efficiency,
        p.battery_voltage, inverse_four_dt)
    return debug_tensor if debug else None
