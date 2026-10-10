#!/usr/bin/env python3
"""Benchmark 3v3 match latency and batch throughput with fixed seeds."""
from __future__ import annotations

import argparse
import json
import time

import torch

from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def run_case(device: str, envs: int, ticks: int, seed: int,
             controllers: bool, fuel_physics: bool,
             replan_interval: int = 20,
             perception_interval: int = 2) -> dict:
    modes = (("deterministic",) * 6 if controllers else ("none",) * 6)
    started = time.perf_counter()
    env = TensorThreeVsThreeEnv(
        num_envs=envs, device=device, seed=seed, control_modes=modes,
        robot_roles=("offense",) * 3 + ("defense",) * 3,
        horizon=ticks, dt=.02,
        randomize=True, fuel_count=504, fused_sensor_rng=True,
        fuel_physics=fuel_physics, replan_interval=replan_interval,
        perception_interval=perception_interval,
    )
    env.reset(seed=seed)
    actions = torch.full((envs, 6), 7, device=device, dtype=torch.long)
    # Build lazy HIP extensions and initialize scratch buffers outside the
    # steady-state rollout timing. Reset after warmup to preserve the seed.
    warmup_ticks = min(40, max(1, ticks // 10))
    for _ in range(warmup_ticks):
        env.step(actions, capture_observation=False, capture_info=False)
    env.reset(seed=seed)
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(ticks):
        env.step(actions, capture_observation=False, capture_info=False)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "envs": envs,
        "ticks": ticks,
        "seed": seed,
        "controllers": controllers,
        "fuel_physics": fuel_physics,
        "replan_interval": replan_interval,
        "perception_interval": perception_interval,
        "setup_seconds": setup_seconds,
        "elapsed_seconds": elapsed,
        "seconds_per_batch_rollout_projection": elapsed * 8000 / ticks,
        "seconds_per_world_full_match_projection": elapsed * 8000 / (ticks * envs),
        "world_ticks_per_second": envs * ticks / elapsed,
        "score_by_team": env.fuel_score_count.detach().cpu().tolist(),
        "fuel_acquisitions_by_robot": env.fuel_acquisition_count.detach().cpu().tolist(),
        "fuel_backend": env.fuel_physics.stats["backend"] if env.fuel_physics else None,
        "physics_graph_enabled": env._physics_graph_enabled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=64821)
    parser.add_argument("--controllers", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--fuel-physics", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--replan-interval", type=int, default=20)
    parser.add_argument("--perception-interval", type=int, default=2)
    args = parser.parse_args()
    if args.envs < 1 or args.ticks < 1:
        parser.error("envs and ticks must be positive")
    if args.replan_interval < 1 or args.perception_interval < 1:
        parser.error("planner and perception intervals must be positive")
    result = run_case(args.device, args.envs, args.ticks, args.seed,
                      args.controllers, args.fuel_physics,
                      args.replan_interval, args.perception_interval)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
