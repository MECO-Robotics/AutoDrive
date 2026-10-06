#!/usr/bin/env python3
"""Measure strategic tensor-simulator throughput across environment batches.

This isolates environment/planner stepping from PPO optimization. For full PPO
throughput, use the same environment counts with tensor_training.train and a
fixed transition budget; generational training changes total transitions when
the environment count changes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time

import torch

# Enable the optional extension before importing tensor_sim, which snapshots
# the opt-in environment flag at module import time.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--fused-gamepieces", action="store_true")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.fused_gamepieces:
    os.environ["AUTODRIVE_FUSED_GAMEPIECES_HIP"] = "1"

from frc_defense import tensor_perception
from frc_defense import tensor_gamepieces
from frc_defense import tensor_adstar
from frc_defense import tensor_collision
from frc_defense import tensor_strategic_observation_proto
from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense.tensor_training import _held_action_interval


def benchmark(env_count: int, device: torch.device, seed: int,
              decisions: int, ticks_per_decision: int, opponent: str,
              return_info: bool = False,
              skip_metric_objective: bool = True,
              profile_components: bool = False,
              squared_fuel_candidate_distance: bool = False,
              reuse_own_candidates: bool = False) -> dict:
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda":
        # ROCm's allocator rejects a ``torch.device`` argument here on some
        # builds even though the process has a single selected GPU.
        torch.cuda.reset_peak_memory_stats()
    env = TensorDefenseEnv(
        num_envs=env_count, task="counter_defense", device=device, seed=seed,
        opponent=opponent, action_mode="strategic", horizon=8000,
        skip_strategic_offense_metric_objective=skip_metric_objective,
        reuse_strategic_own_candidates=reuse_own_candidates,
        squared_fuel_candidate_distance=squared_fuel_candidate_distance,
    )
    component_host_seconds: dict[str, float] = {}
    component_calls: dict[str, int] = {}
    if profile_components:
        profiled = (
            (env, "step", "env.step"), (env, "_finish_step", "_finish_step"),
            (env, "_obs", "_obs"), (env, "_raw_obs", "_raw_obs"),
            (env, "_strategic_observation", "_strategic_observation"),
            (env, "_fuel_candidates", "_fuel_candidates"),
            (env, "_update_perception", "_update_perception"),
            (env, "_update_gamepieces", "_update_gamepieces"),
            (env.sim, "step", "sim.step"),
            (env.sim, "_step_eager", "sim._step_eager"),
        )
        profiled_owners = list(profiled)
        planner_seen = set()
        for attr in ("_adstar_tactical_planner", "_adstar_planners",
                     "_adstar_defender_planners"):
            planner = getattr(env, attr, None)
            if isinstance(planner, (tuple, list)):
                planners = [(f"{attr}.{index}", value)
                            for index, value in enumerate(planner)]
            elif planner is None:
                planners = []
            else:
                planners = [(attr, planner)]
            for label, planner in planners:
                if id(planner) in planner_seen:
                    continue
                planner_seen.add(id(planner))
                for name in ("plan", "path_reference"):
                    profiled_owners.append((planner, name, f"{label}.{name}"))
        for owner, name, label in profiled_owners:
            original = getattr(owner, name, None)
            if not callable(original):
                continue
            component_host_seconds[label] = 0.0
            component_calls[label] = 0
            def timed(*args, _label=label, _original=original, **kwargs):
                before = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    component_host_seconds[_label] += time.perf_counter() - before
                    component_calls[_label] += 1
            setattr(owner, name, timed)
    obs = env._last_observation
    action = env.strategic_action_mask().to(torch.int64).argmax(-1)

    # Warm planner kernels and allocator before measuring the batch.
    warmup_started = time.perf_counter()
    _held_action_interval(env, action, ticks_per_decision, obs,
                          known_remaining_ticks=8000, return_info=return_info,
                          )
    obs = env._last_observation
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_elapsed = time.perf_counter() - warmup_started
    if profile_components:
        for name in component_host_seconds:
            component_host_seconds[name] = 0.0
            component_calls[name] = 0

    start = time.perf_counter()
    physics_ticks = 0
    episode_ticks = ticks_per_decision
    reward_total = torch.zeros(env_count, device=device)
    reward_trace = torch.empty((decisions, env_count), device=device,
                               dtype=reward_total.dtype)
    terminal_seen = torch.zeros(env_count, device=device, dtype=torch.bool)
    truncated_seen = torch.zeros_like(terminal_seen)
    for decision_index in range(decisions):
        rewards, dones, truncated, obs, _, ticks, _, _ = _held_action_interval(
            env, action, ticks_per_decision, obs,
            known_remaining_ticks=8000 - episode_ticks,
            return_info=return_info,
            )
        reward_total += rewards
        reward_trace[decision_index].copy_(rewards)
        terminal_seen |= dones
        truncated_seen |= truncated
        physics_ticks += ticks
        episode_ticks += ticks
        env._last_observation = obs
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    reward_trace_sha256 = hashlib.sha256(
        reward_trace.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

    # Copy only after the timed region so A/B cases can verify that execution
    # produced identical simulator, perception, and planner state.
    state_hash = hashlib.sha256()
    owners = [("env", env), ("sim", env.sim)]
    for name in ("_adstar_tactical_planner", "_adstar_planners",
                 "_adstar_defender_planners"):
        planner = getattr(env, name, None)
        if isinstance(planner, (tuple, list)):
            owners.extend((f"{name}.{index}", item) for index, item in enumerate(planner))
        elif planner is not None:
            owners.append((name, planner))
    for label, owner in owners:
        for name, value in sorted(vars(owner).items()):
            # These are implementation details, not simulator state; the
            # one-shot cache and selected planner backend can differ by case.
            # All resulting simulator/planner state below must still match.
            if name in ("_pending_strategic_own_candidates",
                        "_fused_blocked_hip_enabled",
                        "_fused_swerve_pose_hip_enabled",
                        "_fused_swerve_pose_hip_failed"):
                continue
            key = f"{label}.{name}".encode()
            if torch.is_tensor(value):
                state_hash.update(key)
                state_hash.update(value.detach().contiguous().cpu().numpy().tobytes())
            elif label.startswith("_adstar") and isinstance(value, (bool, int, float)):
                state_hash.update(key + repr(value).encode())
    for name, generator in (("env_rng", env.generator), ("sim_rng", env.sim.generator)):
        state_hash.update(name.encode())
        state_hash.update(generator.get_state().cpu().numpy().tobytes())
    for name in ("_planner_tick_scalar", "_planner_tick_aligned",
                 "_planner_full_batch_active"):
        state_hash.update(name.encode())
        state_hash.update(repr(getattr(env, name, None)).encode())
    for name, tensor in (("reward_total", reward_total),
                         ("terminal_seen", terminal_seen),
                         ("truncated_seen", truncated_seen)):
        state_hash.update(name.encode())
        state_hash.update(tensor.detach().contiguous().cpu().numpy().tobytes())

    result = {
        "num_envs": env_count,
        "decisions_per_env": decisions,
        "physics_ticks_per_decision": ticks_per_decision,
        "warmup_seconds": warmup_elapsed,
        "returns_info": return_info,
        "skip_strategic_offense_metric_objective": skip_metric_objective,
        "reuse_strategic_own_candidates": reuse_own_candidates,
        "squared_fuel_candidate_distance": squared_fuel_candidate_distance,
        "component_host_seconds_nested": component_host_seconds if profile_components else None,
        "component_host_seconds_inclusive": component_host_seconds if profile_components else None,
        "component_calls": component_calls if profile_components else None,
        "component_timing_note": ("inclusive nested host wall timings; component totals overlap"
                                  if profile_components else None),
        "fused_perception_requested": tensor_perception.FUSED_PERCEPTION_HIP_ENABLED,
        "fused_perception_loaded": (tensor_perception.FUSED_PERCEPTION_HIP_ENABLED and
                                    tensor_perception._hip_perception_extension() is not None),
        "fused_gamepieces_requested": tensor_gamepieces.FUSED_GAMEPIECES_HIP_ENABLED,
        "fused_gamepieces_loaded": (tensor_gamepieces.FUSED_GAMEPIECES_HIP_ENABLED and
                                     tensor_gamepieces._extension() is not None),
        "fused_contact_pipeline_hip_enabled": env.sim._fused_contact_pipeline_hip_enabled,
        "fused_swerve_pose_hip_enabled": env.sim._fused_swerve_pose_hip_enabled,
        "fused_collision_hip_enabled": tensor_collision.FUSED_COLLISION_HIP_ENABLED,
        "fused_adstar_occupancy_hip_enabled": tensor_adstar._HIP_OCCUPANCY_ENABLED,
        "fused_adstar_path_reference_hip_enabled": tensor_adstar._HIP_PATH_REFERENCE_ENABLED,
        "fused_strategic_observation_hip_enabled": (
            tensor_strategic_observation_proto.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED),
        "torch_compile_swerve_enabled": env.sim._torch_compile_swerve_enabled,
        "torch_compile_swerve_failed": env.sim._torch_compile_swerve_failed,
        "torch_compile_step_enabled": env.sim._torch_compile_step_enabled,
        "torch_compile_step_failed": env.sim._torch_compile_step_failed,
        "elapsed_seconds": elapsed,
        "strategic_transitions_per_second": env_count * decisions / elapsed,
        "physics_world_ticks_per_second": env_count * physics_ticks / elapsed,
        "device": str(device),
        "reward_total": float(reward_total.sum().item()),
        "reward_trace_sha256": reward_trace_sha256,
        "terminal_worlds_seen": int(terminal_seen.sum().item()),
        "truncated_worlds_seen": int(truncated_seen.sum().item()),
        "final_state_sha256": state_hash.hexdigest(),
    }
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        result.update({
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "free_gib_after_run": free_bytes / 1024**3,
            "total_gib": total_bytes / 1024**3,
        })
    del env, obs, action
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, nargs="+", default=(512, 1024, 2048, 4096))
    parser.add_argument("--decisions", type=int, default=8)
    parser.add_argument("--physics-ticks", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opponent", default="guard")
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--return-info", action="store_true",
                        help="include environment diagnostics to measure the training no-info fast path")
    parser.add_argument("--fused-gamepieces", action="store_true",
                        help="enable the opt-in fused HIP gamepiece update")
    parser.add_argument("--keep-metric-objective", action="store_true",
                        help="compute the legacy metric-only strategic offense objective")
    parser.add_argument("--reuse-strategic-own-candidates", action="store_true",
                        help="reuse controlled-side FUEL candidates from the current observation in the next physics step")
    parser.add_argument("--profile-step-components", action="store_true",
                        help="record nested host-side call timing by simulator step component")
    parser.add_argument("--squared-fuel-candidate-distance", action="store_true",
                        help="rank candidates with squared distances and sqrt only the selected four")
    args = parser.parse_args()
    if min(args.envs) < 1 or min(args.decisions, args.physics_ticks) < 1:
        parser.error("environment counts, decisions, and physics ticks must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but unavailable")
    results = [benchmark(n, device, args.seed, args.decisions,
                         args.physics_ticks, args.opponent, args.return_info,
                         not args.keep_metric_objective,
                         args.profile_step_components,
                         args.squared_fuel_candidate_distance,
                         args.reuse_strategic_own_candidates) for n in args.envs]
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
