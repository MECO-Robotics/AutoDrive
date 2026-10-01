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
import json
import time

import torch

from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense.tensor_training import _held_action_interval


def benchmark(env_count: int, device: torch.device, seed: int,
              decisions: int, ticks_per_decision: int, opponent: str) -> dict:
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda":
        # ROCm's allocator rejects a ``torch.device`` argument here on some
        # builds even though the process has a single selected GPU.
        torch.cuda.reset_peak_memory_stats()
    env = TensorDefenseEnv(
        num_envs=env_count, task="counter_defense", device=device, seed=seed,
        opponent=opponent, action_mode="strategic", horizon=8000,
    )
    obs = env._last_observation
    action = env.strategic_action_mask().to(torch.int64).argmax(-1)

    # Warm planner kernels and allocator before measuring the batch.
    _held_action_interval(env, action, ticks_per_decision, obs)
    obs = env._last_observation
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    physics_ticks = 0
    for _ in range(decisions):
        _, _, _, obs, _, ticks, _ = _held_action_interval(
            env, action, ticks_per_decision, obs)
        physics_ticks += ticks
        env._last_observation = obs
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    result = {
        "num_envs": env_count,
        "decisions_per_env": decisions,
        "physics_ticks_per_decision": ticks_per_decision,
        "elapsed_seconds": elapsed,
        "strategic_transitions_per_second": env_count * decisions / elapsed,
        "physics_world_ticks_per_second": env_count * physics_ticks / elapsed,
        "device": str(device),
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
    args = parser.parse_args()
    if min(args.envs) < 1 or min(args.decisions, args.physics_ticks) < 1:
        parser.error("environment counts, decisions, and physics ticks must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but unavailable")
    results = [benchmark(n, device, args.seed, args.decisions,
                         args.physics_ticks, args.opponent) for n in args.envs]
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
