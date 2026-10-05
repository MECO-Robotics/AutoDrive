"""Optional fused HIP implementation for sequential wall contacts.

The Torch implementation remains the reference and fallback. Set
``AUTODRIVE_FUSED_COLLISION_HIP=0`` to force the reference path.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch


_EXT = None
_ATTEMPTED = False
_ENABLED = os.environ.get("AUTODRIVE_FUSED_COLLISION_HIP", "1") != "0"
FUSED_COLLISION_HIP_ENABLED = _ENABLED


def _extension():
    global _EXT, _ATTEMPTED
    if not _ENABLED:
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
            rocm_alias / "lib/llvm/amdgcn/bitcode",
            rocm_alias / "amdgcn/bitcode",
        ) if path.is_dir()), None)
        cuda_flags = ["-O3", "-ffp-contract=off"]
        if device_lib is not None:
            cuda_flags.append(f"--rocm-device-lib-path={device_lib}")
        try:
            _EXT = cpp_extension.load(
                name="autodrive_tensor_collision_hip",
                sources=[str(source_dir / "tensor_collision_hip.cpp"),
                         str(source_dir / "tensor_collision_hip_kernel.cu")],
                with_cuda=True, extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags, verbose=False)
        finally:
            (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME,
             cpp_extension.HIP_HOME) = original
    except Exception as exc:
        warnings.warn(f"Fused collision HIP kernel unavailable; using Torch: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXT


def wall_contacts(sim, active_mask):
    """Run fused in-place wall contacts. Return false when Torch should run."""
    ext = _extension()
    tensors = (sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
               sim.yaw_inertia_multiplier, sim.wall_mu, active_mask,
               sim.wall_contact)
    if (ext is None or sim.device.type != "cuda" or not torch.version.hip or
            any(t.device != sim.pose.device for t in tensors) or
            any(not t.is_contiguous() for t in tensors)):
        return False
    ext.walls(sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
              sim.yaw_inertia_multiplier, sim.wall_mu, active_mask,
              sim.wall_contact, sim.field_length, sim.field_width)
    return True
