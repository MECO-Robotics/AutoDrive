"""HIP implementation for strategic fuel candidate ranking.

The device kernel fuses Euclidean distance and invalid-candidate masking. The
final top-k intentionally uses ATen so selection follows the active Torch
backend. Equal-distance and +inf index ordering is not deterministic on the
current ROCm build. Disable simulator dispatch with
``AUTODRIVE_FUSED_FUEL_CANDIDATE_RANK_HIP=0``.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

_EXT = None
_ATTEMPTED = False
FUSED_FUEL_CANDIDATE_RANK_HIP_ENABLED = (
    os.environ.get("AUTODRIVE_FUSED_FUEL_CANDIDATE_RANK_HIP", "1") != "0")


def _extension():
    global _EXT, _ATTEMPTED
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
        old = (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME)
        cpp_extension.TORCH_LIB_PATH = str(torch_lib)
        cpp_extension.ROCM_HOME = str(rocm_alias)
        cpp_extension.HIP_HOME = str(rocm_alias / "hip")
        src = Path(__file__).parent / "csrc"
        device_lib = next((p for p in (rocm_alias / "lib/llvm/amdgcn/bitcode",
                                        rocm_alias / "amdgcn/bitcode") if p.is_dir()), None)
        flags = ["-O3", "-ffp-contract=off"]
        if device_lib is not None:
            flags.append(f"--rocm-device-lib-path={device_lib}")
        try:
            _EXT = cpp_extension.load(
                name="autodrive_fuel_candidate_rank_hip",
                sources=[str(src / "tensor_fuel_candidate_rank.cpp"),
                         str(src / "tensor_fuel_candidate_rank_kernel.cu")],
                with_cuda=True, extra_cflags=["-O3"], extra_cuda_cflags=flags,
                verbose=False)
        finally:
            cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = old
    except Exception as exc:
        warnings.warn(f"Fused HIP fuel candidate rank unavailable: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXT


def available(device="cuda"):
    return torch.version.hip is not None and torch.device(device).type == "cuda" and _extension() is not None


def integrated_available(device="cuda"):
    """Whether feature-flagged HIP candidate dispatch is available."""
    return FUSED_FUEL_CANDIDATE_RANK_HIP_ENABLED and available(device)


def candidates(points: torch.Tensor, robot_xy: torch.Tensor, free: torch.Tensor,
               active_mask: torch.Tensor | None = None):
    """Return (indices, valid, nearest_distance), matching _fuel_candidates.

    Inputs are explicitly made contiguous so this isolated API has clear layout
    requirements and does not change their values or generator state.
    """
    if not available(points.device):
        raise RuntimeError("fuel candidate rank prototype requires HIP")
    if active_mask is not None and tuple(active_mask.shape) != (points.shape[0],):
        raise ValueError(f"active_mask must have shape {(points.shape[0],)}")
    # Like observation construction, candidate generation is read-only and
    # returns each row's current data even for inactive worlds.
    return _extension().candidates(points.contiguous(), robot_xy.contiguous(), free.contiguous())
