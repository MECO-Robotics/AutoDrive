#!/usr/bin/env python3
"""Paired 3v3 rollout benchmark for single-world robot contact resolution."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv
from frc_defense import tensor_collision_multi_hip


def run_case(parallel: bool, ticks: int, seed: int) -> dict:
    tensor_collision_multi_hip.PARALLEL_SINGLE_WORLD = parallel
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cuda:0", seed=seed,
        control_modes=("deterministic",) * 6,
        robot_roles=("offense",) * 3 + ("defense",) * 3,
        horizon=8000, randomize=True, fuel_count=504, fused_sensor_rng=True)
    env.reset(seed=seed)
    actions = torch.full((1, 6), 7, device="cuda:0", dtype=torch.long)
    for _ in range(40):
        env.step(actions, capture_observation=False, capture_info=False)
    env.reset(seed=seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(ticks):
        env.step(actions, capture_observation=False, capture_info=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "parallel_single_world_contacts": parallel,
        "ticks": ticks,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "projected_full_match_seconds": elapsed * 8000 / ticks,
        "score_by_team": env.fuel_score_count.detach().cpu().tolist(),
        "fuel_acquisitions_by_robot": env.fuel_acquisition_count.detach().cpu().tolist(),
    }
    del env
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=800)
    parser.add_argument("--seed", type=int, default=64821)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("Requires a ROCm/HIP PyTorch device")
    # Interleave paired cases to reduce clock and temperature ordering bias.
    cases = [run_case(False, args.ticks, args.seed),
             run_case(True, args.ticks, args.seed),
             run_case(True, args.ticks, args.seed),
             run_case(False, args.ticks, args.seed)]
    baseline = statistics.median(
        case["elapsed_seconds"] for case in cases
        if not case["parallel_single_world_contacts"])
    parallel = statistics.median(
        case["elapsed_seconds"] for case in cases
        if case["parallel_single_world_contacts"])
    print(json.dumps({
        "scenario": "full-scale deterministic 3v3",
        "device": "cuda:0",
        "ticks_per_case": args.ticks,
        "seed": args.seed,
        "cases": cases,
        "median_speedup": baseline / parallel,
        "baseline_median_seconds": baseline,
        "parallel_median_seconds": parallel,
        "scope": "full environment rollout; capture and info output disabled",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
