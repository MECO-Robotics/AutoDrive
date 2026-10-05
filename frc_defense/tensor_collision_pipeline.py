"""Fused HIP contact iteration pipeline with a bitwise Torch fallback."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

_EXT = None
_ATTEMPTED = False
FUSED_CONTACT_PIPELINE_HIP_ENABLED = (
    os.environ.get("AUTODRIVE_FUSED_CONTACT_PIPELINE_HIP", "1") != "0")


def _extension():
    global _EXT, _ATTEMPTED
    if not FUSED_CONTACT_PIPELINE_HIP_ENABLED:
        return None
    if _ATTEMPTED:
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
            alias = Path("/tmp/autodrive-torch-lib")
            if alias.is_symlink() and alias.resolve() != torch_lib.resolve():
                alias.unlink()
            if not alias.exists():
                alias.symlink_to(torch_lib.resolve(), target_is_directory=True)
            torch_lib = alias
        original = (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME,
                    cpp_extension.HIP_HOME)
        cpp_extension.TORCH_LIB_PATH = str(torch_lib)
        cpp_extension.ROCM_HOME = str(rocm_alias)
        cpp_extension.HIP_HOME = str(rocm_alias / "hip")
        source_dir = Path(__file__).parent / "csrc"
        device_lib = next((path for path in (
            rocm_alias / "lib/llvm/amdgcn/bitcode", rocm_alias / "amdgcn/bitcode",
        ) if path.is_dir()), None)
        cuda_flags = ["-O3", "-ffp-contract=off"]
        if device_lib is not None:
            cuda_flags.append(f"--rocm-device-lib-path={device_lib}")
        try:
            _EXT = cpp_extension.load(
                name="autodrive_tensor_collision_pipeline_hip",
                sources=[str(source_dir / "tensor_collision_pipeline_hip.cpp"),
                         str(source_dir / "tensor_collision_pipeline_hip_kernel.cu")],
                with_cuda=True, extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags, verbose=False)
        finally:
            (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME,
             cpp_extension.HIP_HOME) = original
    except Exception as exc:
        warnings.warn(f"Fused contact pipeline unavailable; using Torch: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXT


def contact_pipeline(sim, active_mask):
    """Mutate contact state once, executing all ordered iterations; bool success."""
    if (sim.device.type != "cuda" or not torch.version.hip or
            any(not t.is_contiguous() for t in (
                sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
                sim.yaw_inertia_multiplier, sim.mu, sim.wall_mu, active_mask,
                sim.robot_contact, sim.opponent_contact, sim.field_contact,
                sim.wall_contact, sim.obstacles, sim.field_colliders))):
        return False
    ext = _extension()
    if ext is None:
        return False
    ext.contact_pipeline(sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
        sim.yaw_inertia_multiplier, sim.mu, sim.wall_mu, active_mask,
        sim.robot_contact, sim.opponent_contact, sim.field_contact,
        sim.wall_contact, sim.obstacles, sim.field_colliders,
        sim.contact_iterations, sim.field_length, sim.field_width, -1)
    return True
