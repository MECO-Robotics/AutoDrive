"""Optional HIP kernel for exact strategic-observation feature assembly.

Candidate ranking, history, latency, noise, dropout, and all random draws stay
in Torch. On HIP, the optional fused path computes local-track and candidate
features and packs the raw vector in one kernel; all other configurations use
the simulator's Torch implementation. Set
``AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_HIP=0`` to opt out.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

_EXT = None
_ATTEMPTED = False
FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = (
    os.environ.get("AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_HIP", "1") != "0")


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
                name="autodrive_strategic_observation_proto_hip",
                sources=[str(src / "tensor_strategic_observation_proto.cpp"),
                         str(src / "tensor_strategic_observation_proto_kernel.cu")],
                with_cuda=True, extra_cflags=["-O3"], extra_cuda_cflags=flags,
                verbose=False)
        finally:
            cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = old
    except Exception as exc:
        warnings.warn(f"Strategic observation prototype unavailable: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXT


def pack_blocks(base, route, local0, local1, possession, match, candidate,
                action_mask, speed, omega, field_length, field_width,
                normalize=True):
    """Pack the simulator's exact 35+3+20+20+2+9+40+8 layout."""
    tensors = (base, route, local0, local1, possession, match, candidate,
               action_mask, speed, omega)
    if (not torch.version.hip or base.device.type != "cuda" or
            base.dtype != torch.float32 or any(t.dtype != torch.float32 for t in tensors) or
            any(t.device != base.device for t in tensors) or
            any(not t.is_contiguous() for t in tensors)):
        raise RuntimeError("prototype requires contiguous float32 HIP tensors on one device")
    n = base.shape[0]
    expected = ((n, 35), (n, 3), (n, 20), (n, 20), (n, 2), (n, 9),
                (n, 40), (n, 8), (n, 2), (n, 2))
    if tuple(tuple(t.shape) for t in tensors) != expected:
        raise ValueError("invalid strategic observation feature block shapes")
    ext = _extension()
    if ext is None:
        raise RuntimeError("HIP extension is unavailable; no Torch fallback is provided")
    return ext.pack(*tensors, float(field_length), float(field_width), bool(normalize))


def fused_candidate_local_available(device):
    """Whether the flag-gated fused feature kernel is usable for ``device``."""
    if (not FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED or torch.version.hip is None or
            torch.device(device).type != "cuda"):
        return False
    return _extension() is not None


def pack_fused_candidate_local(env, focal, *, base, route, local1, possession,
                               match, action_values, other_pose, own_count,
                               candidate_data, normalize):
    """Return fused raw features, or None to use the exact Torch fallback."""
    if (not FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED or torch.version.hip is None or
            env.device.type != "cuda"):
        return None
    ext = _extension()
    if ext is None:
        return None
    tracked_points, tracked_velocities, _ = env._perceived_fuel(focal)
    indices, valid, nearest = candidate_data
    speed = env.sim.speed[:, [focal, 1 - focal]].contiguous()
    omega = env.sim.omega_limit[:, [focal, 1 - focal]].contiguous()
    return ext.pack_fused(base.contiguous(), route.contiguous(), local1.contiguous(),
        possession.contiguous(), match.contiguous(), action_values.contiguous(),
        speed, omega, tracked_points, tracked_velocities, indices.contiguous(),
        valid.contiguous(), nearest.contiguous(), env.sim.pose[:, focal].contiguous(),
        other_pose[:, :2].contiguous(), own_count.to(base.dtype).contiguous(),
        env.sim.length[:, focal].contiguous(), env.sim.width[:, focal].contiguous(),
        env.hub_centers[focal].contiguous(), float(env.sim.field_length),
        float(env.sim.field_width), int(env.fuel_capacity), bool(normalize))


def _prototype_blocks(env, focal, *, fuse_candidate_local=False):
    """Mirror only the block-building part of TensorDefenseEnv's observation."""
    p, v = env.sim.pose, env.sim.velocity
    other = 1 - focal
    candidate_data = env._fuel_candidates(focal)
    own_count = env._own_possession(focal)
    action_mask = env.strategic_action_mask(focal, _candidate_data=candidate_data)
    other_pose, other_velocity = env._observed_robot(focal, other)
    base = torch.cat((p[:, focal], v[:, focal], other_pose, other_velocity,
        env._opponent_track_features(focal),
        torch.full((env.n, 1), .595, device=env.device),
        torch.full((env.n, 1), env.sim.field_length, device=env.device),
        torch.full((env.n, 1), env.sim.field_width, device=env.device),
        env.sim.length[:, [focal, other]], env.sim.width[:, [focal, other]],
        env.sim.accel[:, [focal, other]] / 10.,
        env._obstacle_features(robot_index=focal)), dim=-1).contiguous()

    planner = (env._adstar_tactical_planner if focal == 0 else
               env._adstar_planners if env.task == "defense" else
               env._adstar_defender_planners)
    route = torch.zeros((env.n, 3), device=env.device)
    if planner is not None:
        has_route = planner.last_lengths > 1
        first = planner.last_path[:, 1, :] - p[:, focal, :2]
        segment = (planner.last_path[:, 1:, :] - planner.last_path[:, :-1, :]).norm(dim=-1)
        valid_path = (torch.arange(segment.shape[1], device=env.device)[None, :] <
                      (planner.last_lengths - 1).clamp_min(0)[:, None])
        cost = (segment * valid_path).sum(-1)
        route = torch.stack((first[:, 0] / env.sim.field_length,
            first[:, 1] / env.sim.field_width,
            cost / __import__("math").hypot(env.sim.field_length, env.sim.field_width)), dim=-1)
        route = torch.where(has_route[:, None], route, torch.zeros_like(route))

    tracked_points, tracked_velocities, _ = env._perceived_fuel(focal)
    indices, valid, nearest = candidate_data
    if fuse_candidate_local:
        # Placeholders are not read by pack_fused; keep a stable tuple layout.
        local0 = torch.zeros((env.n, 20), device=env.device, dtype=base.dtype)
    else:
        rows = torch.arange(env.n, device=env.device)[:, None]
        points = tracked_points[rows, indices]
        velocities = tracked_velocities[rows, indices]
        relative = points - p[:, focal, None, :2]
        normalized_velocity = velocities / env.sim.speed[:, focal, None, None].clamp_min(.1)
        slot = torch.cat((relative[..., 0:1] / env.sim.field_length,
            relative[..., 1:2] / env.sim.field_width, normalized_velocity,
            valid[..., None].to(base.dtype)), dim=-1)
        local0 = torch.where(valid[..., None], slot, torch.zeros_like(slot)).reshape(env.n, -1)
    local1 = torch.zeros((env.n, 20), device=env.device, dtype=base.dtype)

    possession = torch.stack((own_count, torch.zeros_like(own_count)), dim=-1).to(base.dtype)
    possession = possession / max(env.fuel_capacity, 1)
    match = torch.cat((env.match_elapsed[:, None] / 160.,
        env.hub_active.to(base.dtype),
        env.fuel_score_count[:, [focal, other]].to(base.dtype) / max(env.fuel_count, 1),
        (env.hub_centers[None, :, :] - p[:, focal, None, :2]).reshape(env.n, 4) /
          torch.tensor((env.sim.field_length, env.sim.field_width,
                        env.sim.field_length, env.sim.field_width), device=env.device)), dim=-1)
    candidate = (torch.zeros((env.n, 40), device=env.device, dtype=base.dtype)
                 if fuse_candidate_local else
                 env._strategic_candidate_features(
                     focal, _candidate_data=candidate_data, _own_count=own_count))
    action_values = action_mask.to(base.dtype)
    speed = env.sim.speed[:, [focal, other]].contiguous()
    omega = env.sim.omega_limit[:, [focal, other]].contiguous()
    feature_inputs = None
    if fuse_candidate_local:
        feature_inputs = (tracked_points, tracked_velocities, indices, valid, nearest,
            p[:, focal].contiguous(), other_pose[:, :2].contiguous(),
            own_count.to(base.dtype).contiguous(),
            env.sim.length[:, focal].contiguous(), env.sim.width[:, focal].contiguous(),
            env.hub_centers[focal].contiguous())
    return (base, route.contiguous(), local0.contiguous(), local1,
            possession.contiguous(), match.contiguous(), candidate.contiguous(),
            action_values.contiguous(), speed, omega, candidate_data, action_mask,
            feature_inputs)


def strategic_observation(env, focal=0, *, active_mask=None, return_candidate_data=False,
                          normalize=None, fuse_candidate_local=False):
    """Build prototype raw 137-vector; Torch RNG/history stay outside this function."""
    blocks = _prototype_blocks(env, focal, fuse_candidate_local=fuse_candidate_local)
    (base, route, local0, local1, possession, match, candidate, action, speed,
     omega, data, mask, feature_inputs) = blocks
    do_normalize = env.normalize_observations if normalize is None else bool(normalize)
    if fuse_candidate_local:
        if feature_inputs is None:
            raise RuntimeError("fused candidate/local inputs were not constructed")
        if not torch.version.hip:
            raise RuntimeError("fused candidate/local prototype requires HIP")
        ext = _extension()
        if ext is None:
            raise RuntimeError("HIP extension is unavailable")
        (track_pos, track_vel, indices, valid, nearest, own_pose, other_pose,
         own_count, robot_length, robot_width, hub) = feature_inputs
        raw = ext.pack_fused(base, route, local1, possession, match, action,
            speed, omega, track_pos, track_vel, indices, valid, nearest,
            own_pose, other_pose, own_count, robot_length, robot_width, hub,
            float(env.sim.field_length), float(env.sim.field_width),
            int(env.fuel_capacity), bool(do_normalize))
    else:
        raw = pack_blocks(base, route, local0, local1, possession, match,
            candidate, action, speed, omega, env.sim.field_length, env.sim.field_width,
            normalize=do_normalize)
    if focal == 0 and env.reuse_strategic_own_candidates:
        if active_mask is None:
            env._pending_strategic_own_candidates = (data, mask)
        else:
            selected = torch.as_tensor(active_mask, device=env.device,
                                       dtype=torch.bool).reshape(env.n)
            pending = env._pending_strategic_own_candidates
            if pending is None:
                pending = (tuple(torch.zeros_like(value) for value in data),
                           torch.zeros_like(mask))
            prev_data, prev_mask = pending
            merged_data = tuple(torch.where(selected[:, None], now, prev)
                                for now, prev in zip(data, prev_data))
            env._pending_strategic_own_candidates = (
                merged_data, torch.where(selected[:, None], mask, prev_mask))
    if return_candidate_data:
        return raw, data, mask
    return raw


def torch_history_from_raw(env, raw, *, active_mask=None, active_count=None):
    """Apply the unchanged Torch history, latency, noise, and dropout stage.

    This test-only helper mirrors ``TensorDefenseEnv._obs`` after its raw
    observation call. It intentionally uses the environment's existing RNG
    helper and does not fuse or reorder random draws.
    """
    active_mask = (torch.ones(env.n, device=env.device, dtype=torch.bool)
                   if active_mask is None else
                   torch.as_tensor(active_mask, device=env.device, dtype=torch.bool))
    history_length = env.observation_history.shape[1]
    next_index = (env._observation_history_index + 1).remainder(history_length)
    write_index = torch.where(active_mask, next_index, env._observation_history_index)
    current = env.observation_history[env._observation_world_index, write_index]
    updated = torch.where(active_mask[:, None], raw, current)
    env.observation_history[env._observation_world_index, write_index] = updated
    env._observation_history_index.copy_(write_index)
    read_index = (env._observation_history_index - env.observation_delay).remainder(history_length)
    obs = env.observation_history.gather(
        1, read_index[:, None, None].expand(-1, 1, env.obs_dim)).squeeze(1)
    if env.randomize and env.observation_noise > 0:
        state = obs[:, :18] + env._random_active(active_mask, (18,), normal=True,
            active_count=active_count) * env.observation_noise
        obs = torch.cat((state, obs[:, 18:]), -1)
    if env.randomize and env.observation_dropout > 0:
        keep = env._random_active(active_mask, (18,), active_count=active_count) >= env.observation_dropout
        obs = torch.cat((obs[:, :18] * keep, obs[:, 18:]), -1)
    return torch.where(active_mask[:, None], obs, env._last_observation)
