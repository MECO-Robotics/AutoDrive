#!/usr/bin/env python3
"""A/B/A simulator benchmark for fused six-robot HIP contacts."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from frc_defense.tensor_sim import TensorVectorizedSimulator
from frc_defense import tensor_collision_multi_hip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--ticks", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3,
                        help="A/B pairs, each with a fresh identical simulator state")
    parser.add_argument("--active-fraction", type=float, default=.8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=819)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.envs < 1 or args.ticks < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("envs/ticks/repeats must be positive and warmup nonnegative")
    if not 0. < args.active_fraction <= 1.:
        parser.error("active-fraction must be in (0, 1]")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("this benchmark requires a HIP device")
    sim = TensorVectorizedSimulator(args.envs, device=device, num_robots=6,
                                    seed=args.seed, contact_iterations=3)
    gen = torch.Generator(device=device).manual_seed(args.seed + 42)
    centers_x = torch.tensor((1.2, 1.2, 1.2, 15.4, 15.4, 15.4), device=device)
    centers_y = torch.tensor((1.4, 4., 6.6, 1.4, 4., 6.6), device=device)
    pose = torch.empty_like(sim.pose)
    pose[:, :, 0] = centers_x + torch.randn(
        (args.envs, 6), device=device, generator=gen) * .08
    pose[:, :, 1] = centers_y + torch.randn(
        (args.envs, 6), device=device, generator=gen) * .08
    pose[:, :, 2] = torch.randn(
        (args.envs, 6), device=device, generator=gen) * .15
    sim.pose.copy_(pose)
    sim.velocity.copy_(torch.randn(
        (args.envs, 6, 3), device=device, generator=gen) * .4)
    sim._fused_swerve_pose_hip_enabled = False

    command = torch.zeros((args.envs, 6, 3), device=device)
    command[:, :, 0] = .15
    # Use an exact-count mask so A and B execute the same number of active rows.
    active_count = round(args.envs * args.active_fraction)
    active = torch.arange(args.envs, device=device) < active_count
    snapshot = {key: value.clone() for key, value in vars(sim).items()
                if torch.is_tensor(value) and value.numel()}
    generator_state = sim.generator.get_state()

    def restore() -> None:
        for key, value in snapshot.items():
            getattr(sim, key).copy_(value)
        sim.generator.set_state(generator_state)

    def run(enabled: bool) -> tuple[float, dict[str, torch.Tensor]]:
        tensor_collision_multi_hip.ENABLED = enabled
        sim._fused_robot_collision_multi_hip_enabled = enabled
        restore()
        for _ in range(args.warmup):
            sim.step(command, active, _active_nonempty=True)
        torch.cuda.synchronize(device)
        restore()
        started = time.perf_counter()
        for _ in range(args.ticks):
            sim.step(command, active, _active_nonempty=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        state = {key: value.clone() for key, value in vars(sim).items()
                 if torch.is_tensor(value) and value.numel()}
        return elapsed, state

    torch_times, fused_times = [], []
    torch_states, fused_states = [], []
    for _ in range(args.repeats):
        torch_time, torch_state = run(False)
        fused_time, fused_state = run(True)
        torch_times.append(torch_time)
        fused_times.append(fused_time)
        torch_states.append(torch_state)
        fused_states.append(fused_state)
    tensor_collision_multi_hip.ENABLED = True
    reference = torch_states[0]
    max_state_diff = max(float((reference[key].float() - candidate[key].float()).abs().max())
        for candidate in fused_states for key in reference)
    max_repeat_diff = max(float((reference[key].float() - candidate[key].float()).abs().max())
        for candidate in torch_states[1:] for key in reference)
    median_torch = statistics.median(torch_times)
    median_fused = statistics.median(fused_times)
    report = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "envs": args.envs,
        "physics_ticks": args.ticks,
        "warmup_ticks": args.warmup,
        "active_fraction": active_count / args.envs,
        "repeats": args.repeats,
        "torch_seconds": torch_times,
        "fused_seconds": fused_times,
        "median_torch_seconds": median_torch,
        "median_fused_seconds": median_fused,
        "speedup_vs_median_torch": median_torch / median_fused,
        "max_final_state_diff": max_state_diff,
        "max_torch_repeat_diff": max_repeat_diff,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    if max_state_diff > 5.e-5 or max_repeat_diff > 5.e-5:
        raise SystemExit("A/B/A state parity exceeded tolerance")


if __name__ == "__main__":
    main()
