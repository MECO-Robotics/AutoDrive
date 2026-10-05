#!/usr/bin/env python3
"""Integrated strategic-observation A/B for isolated fuel candidate ranking."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense import tensor_strategic_observation_full_proto as full_obs
from frc_defense.tensor_fuel_candidate_rank import candidates, available


def reference_candidates(env, focal):
    points, _, free = env._perceived_fuel(focal)
    delta = points - env.sim.pose[:, focal, None, :2]
    distance = delta.norm(dim=-1).masked_fill(~free, float("inf"))
    nearest, indices = distance.topk(4, dim=-1, largest=False)
    return indices, torch.isfinite(nearest), nearest


def check_semantic_candidates(a, b):
    if not torch.equal(a[1], b[1]) or not torch.equal(a[2], b[2]):
        raise AssertionError("candidate validity or nearest distance mismatch")
    if not torch.equal(a[0][a[1]], b[0][b[1]]):
        raise AssertionError("valid candidate indices mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not available("cuda"):
        raise RuntimeError("fuel candidate rank HIP extension unavailable")
    if not full_obs.available("cuda"):
        raise RuntimeError("full strategic observation assembler unavailable")

    torch.manual_seed(873311)
    torch.cuda.manual_seed_all(873311)
    env = TensorDefenseEnv(num_envs=args.envs, task="counter_defense", device="cuda:0",
        seed=873311, action_mode="strategic", randomize=True,
        reuse_strategic_own_candidates=False)
    active = (torch.arange(args.envs, device=env.device) % 5 != 1)
    old = env._fuel_candidates

    def baseline():
        return full_obs.observe(env, 0, active_mask=active, return_candidate_data=True)

    def prototype():
        def fused(focal):
            points, _, free = env._perceived_fuel(focal)
            return candidates(points, env.sim.pose[:, focal, :2], free,
                              active_mask=active)
        env._fuel_candidates = fused
        try:
            return full_obs.observe(env, 0, active_mask=active, return_candidate_data=True)
        finally:
            env._fuel_candidates = old

    ref = baseline()
    got = prototype()
    if not torch.equal(ref[0], got[0]):
        raise AssertionError("integrated 137-feature observation mismatch")
    check_semantic_candidates(ref[1], got[1])
    if not torch.equal(ref[2], got[2]):
        raise AssertionError("integrated strategic action mask mismatch")

    # Warm kernels and extension dispatch before timing. Use events to measure
    # device elapsed time without inserting per-operation host synchronizations.
    for _ in range(8):
        baseline(); prototype()
    torch.cuda.synchronize()

    def timed(fn):
        samples = []
        for _ in range(args.repeat):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return samples

    order = []
    for rep in range(args.repeat):
        order.append(("baseline", baseline) if rep % 2 == 0 else ("prototype", prototype))
        order.append(("prototype", prototype) if rep % 2 == 0 else ("baseline", baseline))
    samples = {"baseline": [], "prototype": []}
    for name, fn in order:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize()
        samples[name].append(start.elapsed_time(end))
    a, b = samples["baseline"], samples["prototype"]
    result = {"device": torch.cuda.get_device_name(), "envs": args.envs,
        "candidate_count": int(env.fuel_count), "active_fraction": float(active.float().mean().item()),
        "observation_and_valid_candidate_parity": True,
        "baseline_ms_median": statistics.median(a), "prototype_ms_median": statistics.median(b),
        "speedup": statistics.median(a) / statistics.median(b),
        "baseline_ms_samples": a, "prototype_ms_samples": b}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
