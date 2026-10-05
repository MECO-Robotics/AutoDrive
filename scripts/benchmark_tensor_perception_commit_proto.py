#!/usr/bin/env python3
"""Short A/B/A/B/A timing for production vs isolated perception commit."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense import tensor_perception_commit_proto as prototype


TRACK_NAMES = ("_track_pos", "_track_vel", "_track_mask", "_track_age",
               "_opponent_track_pose", "_opponent_track_velocity",
               "_opponent_track_valid", "_opponent_track_age")


def _new_env(seed, device, worlds, pieces):
    env = TensorDefenseEnv(
        num_envs=worlds, task="counter_defense", device=device, seed=seed,
        action_mode="direct", randomize=True, observation_noise=0.0,
        observation_dropout=0.0,
        perception_config={"detection_dropout": .02,
            "position_noise_m": .015, "velocity_noise_mps": .05},
        fuel_count=pieces, horizon=2000)
    env.reset(seed=seed)
    return env


def _assert_parity(reference, candidate, label):
    for name in TRACK_NAMES:
        if not torch.equal(getattr(reference, name), getattr(candidate, name)):
            raise AssertionError(f"{label}: {name} differs")
    if not torch.equal(reference.generator.get_state(), candidate.generator.get_state()):
        raise AssertionError(f"{label}: RNG state differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--pieces", type=int, default=504)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=24)
    parser.add_argument("--seed", type=int, default=73151)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("This timing harness requires HIP.")
    ext = prototype.extension()
    if ext is None:
        raise SystemExit("Perception commit HIP extension failed to load.")
    device = torch.device(args.device)
    reference = _new_env(args.seed, device, args.worlds, args.pieces)
    candidate = _new_env(args.seed, device, args.worlds, args.pieces)
    _assert_parity(reference, candidate, "initial state")
    active = torch.arange(args.worlds, device=device).remainder(7).ne(2)
    active_count = int(active.sum().item())

    def call_reference():
        reference._update_perception(active_mask=active, active_count=active_count)

    def call_candidate():
        prototype.update_perception(candidate, active_mask=active,
                                    active_count=active_count)

    torch.cuda.synchronize(device)
    for _ in range(args.warmup):
        call_reference()
        call_candidate()
    torch.cuda.synchronize(device)
    _assert_parity(reference, candidate, "warmup parity")

    order = ("production", "prototype", "production", "prototype", "production")
    blocks = []
    for variant in order:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        call = call_reference if variant == "production" else call_candidate
        for _ in range(args.ticks):
            call()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        # Advance the untimed counterpart equally so every A/B block starts
        # from matching tracks and generator state.
        other_call = call_candidate if variant == "production" else call_reference
        for _ in range(args.ticks):
            other_call()
        torch.cuda.synchronize(device)
        _assert_parity(reference, candidate, f"after {variant} block")
        blocks.append({"variant": variant, "ticks": args.ticks,
            "elapsed_seconds": elapsed,
            "seconds_per_tick": elapsed / args.ticks,
            "physics_world_ticks_per_second": args.worlds * args.ticks / elapsed})

    production = [b["seconds_per_tick"] for b in blocks if b["variant"] == "production"]
    fused = [b["seconds_per_tick"] for b in blocks if b["variant"] == "prototype"]
    report = {
        "workload": {"device": str(device), "worlds": args.worlds,
            "fuel_pieces": args.pieces, "active_worlds": active_count,
            "warmup_ticks_per_variant": args.warmup,
            "measured_ticks_per_block": args.ticks,
            "order": list(order), "calls": "_update_perception only; no env.step"},
        "blocks": blocks,
        "production_median_seconds_per_tick": sorted(production)[len(production) // 2],
        "prototype_median_seconds_per_tick": sorted(fused)[len(fused) // 2],
        "median_speedup": (sorted(production)[len(production) // 2] /
                           sorted(fused)[len(fused) // 2]),
        "exact_track_and_rng_parity": True,
        "timing_note": "host wall with HIP synchronize before and after each block; A/B/A/B/A",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
