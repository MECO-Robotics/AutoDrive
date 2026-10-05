#!/usr/bin/env python3
"""Narrow test-only HIP graph check for TensorSimulator.step."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from frc_defense.tensor_sim import TensorDefenseEnv


def make(device, n, seed):
    env = TensorDefenseEnv(
        num_envs=n, task="counter_defense", device=device, seed=seed,
        opponent="shadow", action_mode="strategic", horizon=8000,
        randomize=False, max_observation_latency_steps=0,
        max_control_latency_steps=0,
    )
    env.reset(seed=seed)
    return env


def tensor_state(sim):
    result = {}
    for name, value in vars(sim).items():
        if torch.is_tensor(value):
            result[name] = value.detach().clone()
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                if torch.is_tensor(item):
                    result[f"{name}[{i}]"] = item.detach().clone()
        elif isinstance(value, dict):
            for key, item in value.items():
                if torch.is_tensor(item):
                    result[f"{name}.{key}"] = item.detach().clone()
    return result


def compare(left, right, tick):
    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            return f"tick {tick}: state tensor {name} exists on only one side"
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            return f"tick {tick}: {name} shape/dtype differs"
        if not torch.equal(a, b):
            delta = (a.float() - b.float()).abs()
            return f"tick {tick}: {name} differs (max_abs={delta.max().item()})"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=93211)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("Requires a ROCm/HIP PyTorch device")
    eager, captured = make(args.device, args.envs, args.seed), make(args.device, args.envs, args.seed)
    device = torch.device(args.device)
    commands = torch.tensor((.3, -.2, .15, -.25, .1, -.1),
                            device=device, dtype=eager.sim.pose.dtype).reshape(1, 2, 3).expand(args.envs, 2, 3).contiguous()
    active = torch.ones((args.envs,), device=device, dtype=torch.bool)
    torch.cuda.synchronize(device)
    # Warm extension compilation and allocator paths, then reset both sims to
    # identical seeds before capture and comparison.
    for _ in range(2):
        eager.sim.step(commands, active, _active_nonempty=True)
        captured.sim.step(commands, active, _active_nonempty=True)
    eager.reset(seed=args.seed)
    captured.reset(seed=args.seed)
    torch.cuda.synchronize(device)
    graph = None
    report = {"status": "blocked", "envs": args.envs, "ticks_requested": args.ticks,
              "scope": "TensorSimulator.step only; env.step/AD*/perception/observation/reward eager or excluded",
              "exact_ticks_passed": 0, "parity": None, "error": None}
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured.sim.step(commands, active, _active_nonempty=True)
        torch.cuda.synchronize(device)
        for tick in range(1, args.ticks + 1):
            eager.sim.step(commands, active, _active_nonempty=True)
            graph.replay()
            torch.cuda.synchronize(device)
            difference = compare(tensor_state(eager.sim), tensor_state(captured.sim), tick)
            if difference:
                raise AssertionError(difference)
            report["exact_ticks_passed"] = tick
        report["status"] = "exact"
        report["parity"] = "all simulator tensor state exact after each replay"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if report["status"] != "exact":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
