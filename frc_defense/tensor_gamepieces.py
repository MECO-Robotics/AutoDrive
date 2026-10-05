"""Optional fused HIP update for per-world gamepiece pickup and scoring."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

from .field import ALLIANCE_ZONE_DEPTH

_EXT = None
_ATTEMPTED = False
# Enable the exact-parity HIP fast path by default when running on ROCm.
# Set AUTODRIVE_FUSED_GAMEPIECES_HIP=0 to force the Torch reference path.
_ENABLED = os.environ.get("AUTODRIVE_FUSED_GAMEPIECES_HIP", "1") != "0"
FUSED_GAMEPIECES_HIP_ENABLED = _ENABLED


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
        orig = (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME)
        cpp_extension.TORCH_LIB_PATH = str(torch_lib)
        cpp_extension.ROCM_HOME = str(rocm_alias)
        cpp_extension.HIP_HOME = str(rocm_alias / "hip")
        src = Path(__file__).parent / "csrc"
        device_lib = next((path for path in (
            rocm_alias / "lib/llvm/amdgcn/bitcode",
            rocm_alias / "amdgcn/bitcode",
        ) if path.is_dir()), None)
        cuda_flags = ["-O3", "-ffp-contract=off"]
        if device_lib is not None:
            cuda_flags.append(f"--rocm-device-lib-path={device_lib}")
        try:
            _EXT = cpp_extension.load(
                name="autodrive_tensor_gamepieces_hip_v2",
                sources=[str(src / "tensor_gamepieces_hip.cpp"),
                         str(src / "tensor_gamepieces_hip_kernel.cu")],
                with_cuda=True, extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags, verbose=False)
        finally:
            cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = orig
    except Exception as exc:
        warnings.warn(f"Fused gamepiece HIP kernel unavailable; using Torch: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXT


def update_gamepieces(env, active_mask):
    """Run the in-place fused update, or return False for the Torch fallback."""
    ext = _extension()
    if ext is None or env.device.type != "cuda":
        return False
    tensors = (
        active_mask, env.sim.pose, env.sim.velocity, env.sim.length, env.sim.width,
        env.piece_pos, env.piece_vel, env.piece_active, env.piece_owner,
        env.piece_zone, env.match_elapsed, env.match_remaining, env.hub_active,
        env.hub_inactive_first, env.auto_fuel_scores, env.next_intake_time,
        env.next_score_time, env._last_hub_zone, env._last_strategic_action,
        env.fuel_acquisition_count, env.fuel_score_count, env.fuel_denied_count,
        env.fuel_abandoned_count, env.fuel_acquired_event, env.fuel_scored_event,
        env.fuel_denied_event, env.fuel_abandoned_event, env.hub_centers,
        env._midfield_respawn_positions, env._track_mask, env._track_age,
    )
    if (not torch.version.hip or any(t.device != env.sim.pose.device for t in tensors)
            or any(not t.is_contiguous() for t in tensors)):
        return False
    ext.update(
        *tensors, float(env.dt), float(env.intake_interval), float(env.score_interval),
        int(env.fuel_capacity), bool(env.action_mode == "strategic"), int(env.fuel_count),
        float(env.sim.field_length), float(ALLIANCE_ZONE_DEPTH))
    return True
