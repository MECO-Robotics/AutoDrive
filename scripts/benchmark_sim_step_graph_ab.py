#!/usr/bin/env python3
"""Matched eager/graph TensorSimulator.step parity and wall-time microbench."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

import torch

from frc_defense.tensor_sim import TensorDefenseEnv


def make_env(device: str, n: int, seed: int):
    env = TensorDefenseEnv(
        num_envs=n, task="counter_defense", device=device, seed=seed,
        opponent="shadow", action_mode="strategic", horizon=8000,
        randomize=False, max_observation_latency_steps=0,
        max_control_latency_steps=0,
    )
    env.reset(seed=seed)
    return env


def state_snapshot(sim):
    tensors = {}
    for name, value in vars(sim).items():
        if torch.is_tensor(value):
            tensors[name] = value.detach().clone()
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                if torch.is_tensor(item):
                    tensors[f"{name}[{i}]"] = item.detach().clone()
        elif isinstance(value, dict):
            for key, item in value.items():
                if torch.is_tensor(item):
                    tensors[f"{name}.{key}"] = item.detach().clone()
    if hasattr(sim, "generator"):
        tensors["generator_state"] = sim.generator.get_state().clone()
    return tensors


def first_difference(left, right, label):
    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            return f"{label}: tensor {name} exists on only one side"
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            return f"{label}: {name} shape/dtype differs"
        if not torch.equal(a, b):
            delta = (a.float() - b.float()).abs()
            return f"{label}: {name} differs (max_abs={delta.max().item()})"
    return None


def run_block(env, mode, graph, commands, active, ticks, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    if mode == "eager":
        for _ in range(ticks):
            env.sim.step(commands, active, _active_nonempty=True)
    else:
        for _ in range(ticks):
            graph.replay()
    torch.cuda.synchronize(device)
    return time.perf_counter() - start, state_snapshot(env.sim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--ticks", type=int, default=256)
    parser.add_argument("--parity-ticks", type=int, default=384)
    parser.add_argument("--seed", type=int, default=78147)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("Requires a ROCm/HIP PyTorch device")
    device = torch.device(args.device)
    env = make_env(args.device, args.envs, args.seed)
    commands = torch.tensor((.3, -.2, .15, -.25, .1, -.1), device=device,
                            dtype=env.sim.pose.dtype).reshape(1, 2, 3).expand(
                                args.envs, 2, 3).contiguous()
    active = torch.ones((args.envs,), device=device, dtype=torch.bool)

    # Warm lazy extensions and allocation paths, then capture with persistent
    # fixed-shape action and active-mask buffers.
    for _ in range(4):
        env.sim.step(commands, active, _active_nonempty=True)
    env.reset(seed=args.seed)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        env.sim.step(commands, active, _active_nonempty=True)
    torch.cuda.synchronize(device)

    report = {
        "status": "blocked", "device": str(device), "envs": args.envs,
        "seed": args.seed, "parity_ticks": args.parity_ticks,
        "measured_ticks_per_block": args.ticks,
        "scope": "TensorSimulator.step only; full-active, stable command/mask buffers",
        "production_changed": False,
        "parity": {"status": "not_run"}, "blocks": [], "speedup": None,
    }
    try:
        env.reset(seed=args.seed)
        for _ in range(args.parity_ticks):
            env.sim.step(commands, active, _active_nonempty=True)
        eager_final = state_snapshot(env.sim)

        env.reset(seed=args.seed)
        for _ in range(args.parity_ticks):
            graph.replay()
        graph_final = state_snapshot(env.sim)
        difference = first_difference(eager_final, graph_final, "pre-benchmark parity")
        if difference:
            report["parity"] = {"status": "failed", "difference": difference}
            report["status"] = "parity_failed"
            return write_report(args.output, report, 1)
        report["parity"] = {
            "status": "exact", "ticks": args.parity_ticks,
            "state": "all direct TensorSimulator tensors and generator state",
        }

        # A/B/A/B/A matched blocks. Each starts from the same seed/state; only
        # synchronization occurs at the block boundaries, never per tick.
        timed_reference = None
        for index, mode in enumerate(("eager", "graph", "eager", "graph", "eager"), 1):
            env.reset(seed=args.seed)
            seconds, final_state = run_block(env, mode, graph, commands, active,
                                             args.ticks, device)
            if timed_reference is None:
                timed_reference = final_state
            else:
                difference = first_difference(timed_reference, final_state,
                                              f"timed block {index} parity")
                if difference:
                    report["status"] = "timed_parity_failed"
                    report["error"] = difference
                    return write_report(args.output, report, 1)
            report["blocks"].append({"index": index, "mode": mode,
                                     "seconds": seconds,
                                     "world_ticks_per_second": args.envs * args.ticks / seconds})
        eager_times = [b["seconds"] for b in report["blocks"] if b["mode"] == "eager"]
        graph_times = [b["seconds"] for b in report["blocks"] if b["mode"] == "graph"]
        eager_median, graph_median = statistics.median(eager_times), statistics.median(graph_times)
        report["speedup"] = eager_median / graph_median
        report["summary"] = {
            "eager_median_seconds": eager_median,
            "graph_median_seconds": graph_median,
            "eager_spread_seconds": max(eager_times) - min(eager_times),
            "graph_spread_seconds": max(graph_times) - min(graph_times),
            "speedup_eager_over_graph": report["speedup"],
            "exact_timed_state_parity": True,
        }
        report["status"] = "complete"
        return write_report(args.output, report, 0)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        return write_report(args.output, report, 1)


def write_report(path, report, exit_code):
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2)
    path.write_text(rendered + "\n")
    print(rendered)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
