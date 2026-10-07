"""Opt-in HIP pickup kernel for the six-robot batched simulator."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

_EXT = None
_ATTEMPTED = False
ENABLED = os.environ.get("AUTODRIVE_3V3_PICKUP_GRID_HIP", "1") != "0"


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
        old = (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME,
               cpp_extension.HIP_HOME)
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
                name="autodrive_3v3_pickup_grid_hip_v5",
                sources=[str(src / "tensor_3v3_pickup_grid_hip.cpp"),
                         str(src / "tensor_3v3_pickup_grid_hip_kernel.cu")],
                with_cuda=True, extra_cflags=["-O3"], extra_cuda_cflags=flags,
                verbose=False)
        finally:
            cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = old
    except Exception as exc:
        warnings.warn(f"Fused HIP 3v3 pickup kernel unavailable: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXT


def integrated_available(device="cuda"):
    return ENABLED and torch.version.hip is not None and \
        torch.device(device).type == "cuda" and _extension() is not None


def pickup_robot(*, active, pose, length, width, piece_pos, piece_active,
                 piece_owner, free, possible_cells, next_intake, elapsed,
                 controlled, deterministic, defense_role, acquired_event,
                 track_clear_mask, robot, nx, ny, cell_size, capacity,
                 allow_sweep):
    ext = _extension()
    if ext is None:
        raise RuntimeError("HIP 3v3 pickup extension is unavailable")
    ext.pickup(active.contiguous(), pose.contiguous(), length.contiguous(),
        width.contiguous(), piece_pos.contiguous(), piece_active.contiguous(),
        piece_owner, free, possible_cells.contiguous(), next_intake, elapsed,
        controlled.contiguous(), deterministic.contiguous(),
        defense_role.contiguous(), acquired_event, track_clear_mask,
        int(robot), int(nx), int(ny), float(cell_size), int(capacity),
        bool(allow_sweep))


def pickup_all_robots(*, active, pose, length, width, piece_pos, piece_active,
                      piece_owner, free, possible_cells, next_intake, elapsed,
                      controlled, deterministic, defense_role, hub_active,
                      capacities, acquired_event, track_clear_mask,
                      nx, ny, cell_size, alliance_depth, field_length):
    """Resolve all six robots in one ordered kernel, preserving slot priority."""
    ext = _extension()
    if ext is None:
        raise RuntimeError("HIP 3v3 pickup extension is unavailable")
    ext.pickup_all(active.contiguous(), pose.contiguous(), length.contiguous(),
        width.contiguous(), piece_pos.contiguous(), piece_active.contiguous(),
        piece_owner, free, possible_cells.contiguous(), next_intake, elapsed,
        controlled.contiguous(), deterministic.contiguous(),
        defense_role.contiguous(), hub_active.contiguous(),
        capacities.contiguous(), acquired_event, track_clear_mask,
        int(nx), int(ny), float(cell_size), float(alliance_depth),
        float(field_length))


def possession_counts(piece_owner):
    """Count owned pieces for six robot slots in one HIP kernel launch."""
    ext=_extension()
    if ext is None or not hasattr(ext,"possession_counts"):
        raise RuntimeError("HIP 3v3 pickup extension is unavailable")
    return ext.possession_counts(piece_owner.contiguous())
