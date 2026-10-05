"""Fused HIP assembler for strategic raw observations.

The assembler fuses base/route/possession/match/action-mask assembly,
candidate/local feature generation, packing, and normalization. Perception,
opponent observation selection, and obstacle top-k remain on their existing
paths. Fuel candidate distance/masking has a separate optional HIP path; its
top-k selection remains ATen. Set
``AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_FULL_HIP=0`` to opt out of this
assembler on HIP systems.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

_EXT = None
_ATTEMPTED = False
FUSED_FULL_STRATEGIC_OBSERVATION_HIP_ENABLED = (
    os.environ.get("AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_FULL_HIP", "1") != "0")


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
                name="autodrive_strategic_observation_full_proto_hip",
                sources=[str(src / "tensor_strategic_observation_full_proto.cpp"),
                         str(src / "tensor_strategic_observation_full_proto_kernel.cu")],
                with_cuda=True, extra_cflags=["-O3"], extra_cuda_cflags=flags,
                verbose=False)
        finally:
            cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = old
    except Exception as exc:
        warnings.warn(f"Full strategic observation prototype unavailable: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXT


def available(device):
    return torch.version.hip is not None and torch.device(device).type == "cuda" and _extension() is not None


def integrated_available(device):
    """Whether the exact-parity HIP assembler is enabled and available."""
    return FUSED_FULL_STRATEGIC_OBSERVATION_HIP_ENABLED and available(device)


def observe(env, focal=0, *, normalize=None, candidates=None, route_available=None,
            active_mask=None, return_candidate_data=False, row_indices=None,
            action_mask=None, own_count=None):
    """Assemble full 137-vector from current env state; does not alter env state.

    ``active_mask`` is accepted to make masked-batch parity cases explicit.
    ``active_mask`` gates raw-feature assembly for PPO's delayed-observation
    capture path. Nonselected output rows are zero and must not be consumed.
    """
    if not available(env.device):
        raise RuntimeError("full observation prototype requires HIP")
    if active_mask is None:
        active_mask = torch.ones((env.n,), device=env.device, dtype=torch.bool)
    else:
        active_mask = torch.as_tensor(active_mask, device=env.device,
                                      dtype=torch.bool).reshape(env.n).contiguous()
    rows = (torch.arange(env.n, device=env.device) if row_indices is None else
            torch.as_tensor(row_indices, device=env.device, dtype=torch.long).reshape(-1))
    def take(value):
        return value if row_indices is None else value.index_select(0, rows)

    p, v = env.sim.pose, env.sim.velocity
    other = 1 - focal
    other_pose, other_velocity = env._observed_robot(focal, other)
    other_pose, other_velocity = take(other_pose), take(other_velocity)
    opponent = take(env._opponent_track_features(focal))
    obstacles = env._obstacle_features(robot_index=focal, row_indices=row_indices)
    if candidates is None:
        candidates = env._fuel_candidates(focal)
    indices, valid, nearest = (take(value) for value in candidates)
    if own_count is None:
        own_count = env._own_possession(focal)
    own_count = take(own_count).to(p.dtype)
    planner = (env._adstar_tactical_planner if focal == 0 else
               env._adstar_planners if env.task == "defense" else
               env._adstar_defender_planners)
    if route_available is False or planner is None:
        path = torch.zeros((env.n, 2, 2), device=env.device, dtype=p.dtype)
        path_lengths = torch.zeros((env.n,), device=env.device, dtype=torch.long)
    else:
        path, path_lengths = take(planner.last_path), take(planner.last_lengths)
    action_offense = ((env.task == "counter_defense" and focal == 0) or
                      (env.task == "defense" and focal == 1))
    do_normalize = env.normalize_observations if normalize is None else bool(normalize)
    if action_mask is None:
        action_mask = env.strategic_action_mask(focal, _candidate_data=candidates)
    full_action_mask = action_mask
    raw = _extension().assemble(
        take(p[:, focal]).contiguous(), take(v[:, focal]).contiguous(),
        other_pose.contiguous(), other_velocity.contiguous(),
        opponent.contiguous(), obstacles.contiguous(),
        take(env.sim.length[:, [focal, other]]).contiguous(),
        take(env.sim.width[:, [focal, other]]).contiguous(),
        take(env.sim.accel[:, [focal, other]]).contiguous(),
        take(env.sim.speed[:, [focal, other]]).contiguous(),
        take(env.sim.omega_limit[:, [focal, other]]).contiguous(),
        path.contiguous(), path_lengths.contiguous(),
        take(env._track_pos[:, focal]), take(env._track_vel[:, focal]),
        indices.contiguous(), valid.contiguous(), nearest.contiguous(),
        own_count.contiguous(), take(env.piece_active).contiguous(),
        take(env.piece_owner).contiguous(),
        take(env.match_elapsed).contiguous(), take(env.hub_active).contiguous(),
        take(env.fuel_score_count[:, [focal, other]]).contiguous(),
        env.hub_centers.reshape(-1).contiguous(), env.hub_centers[focal].contiguous(),
        torch.ones((rows.shape[0],), device=env.device, dtype=torch.bool),
        float(env.sim.field_length), float(env.sim.field_width),
        int(env.fuel_capacity), int(env.fuel_count), bool(action_offense),
        bool(do_normalize))
    if return_candidate_data:
        return raw, candidates, full_action_mask
    return raw
