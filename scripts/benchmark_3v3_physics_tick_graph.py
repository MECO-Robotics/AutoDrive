#!/usr/bin/env python3
"""Compare eager and HIP graph replay for the coupled physics substeps."""
from __future__ import annotations

import argparse
import json
import time

import torch

from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def make_env(device: str, seed: int):
    env = TensorThreeVsThreeEnv(
        num_envs=1, device=device, seed=seed, control_modes=("none",) * 6,
        robot_roles=("offense",) * 3 + ("defense",) * 3,
        horizon=8000, randomize=True, fused_sensor_rng=True)
    env.reset(seed=seed)
    return env


def substeps(env, active):
    return env.fuel_physics.required_substeps(
        env.sim.dt, None, active, env.piece_active, env.piece_owner)


def eager_tick(env, command, active):
    physics = env.fuel_physics
    outer_dt = env.sim.dt
    count = substeps(env, active)
    dt = outer_dt / count
    try:
        env.sim.dt = dt
        for _ in range(count):
            env.sim.step(command, active, _active_nonempty=True)
            physics.step(active, env.piece_active, env.piece_owner,
                         dt=dt, substeps=1, adaptive=False)
    finally:
        env.sim.dt = outer_dt
    return count


def state(env):
    values = {}
    for prefix, owner in (("sim", env.sim), ("fuel", env.fuel_physics)):
        for name, value in vars(owner).items():
            if torch.is_tensor(value) and value.numel():
                values[f"{prefix}.{name}"] = value.detach().clone()
    hip = env.fuel_physics._hip
    if hip is not None:
        # HIP counters are instrumentation and may be updated differently by
        # graph capture/replay. Compare durable solver and neighbor state.
        for name in ("reference", "oldfree", "rebuild", "counts", "overflow",
                     "contact_count", "robot_candidates", "support_height"):
            value = getattr(hip, name, None)
            if torch.is_tensor(value) and value.numel():
                values[f"hip.{name}"] = value.detach().clone()
    return values


def first_difference(left, right):
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            return f"{key} missing from one state"
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            return f"{key} shape/dtype mismatch"
        if not torch.equal(a, b):
            delta = (a.float() - b.float()).abs()
            return f"{key} differs, max_abs={delta.max().item()}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ticks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=64821)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("Requires a ROCm/HIP PyTorch device")

    eager, graphed = make_env(args.device, args.seed), make_env(args.device, args.seed)
    device = torch.device(args.device)
    command = torch.zeros((1, 6, 3), device=device)
    command[:, :, 0] = torch.tensor((.3, -.2, .1, -.25, .15, -.1), device=device)
    command[:, :, 1] = torch.tensor((.05, .1, -.08, -.04, .06, -.03), device=device)
    eager_command, graph_command = command.clone(), command.clone()
    active = torch.ones((1,), device=device, dtype=torch.bool)
    eager_active, graph_active = active.clone(), active.clone()

    # Warm JIT extensions and caches, then reset both simulations identically.
    for _ in range(2):
        eager_tick(eager, eager_command, eager_active)
        eager.reset(seed=args.seed)
    graphed.reset(seed=args.seed)
    torch.cuda.synchronize(device)

    graphs = {}

    def get_graph(count):
        if count in graphs:
            return graphs[count]
        outer_dt = graphed.sim.dt
        dt = outer_dt / count
        graph = torch.cuda.CUDAGraph()
        try:
            graphed.sim.dt = dt
            with torch.cuda.graph(graph):
                graphed.sim.step(graph_command, graph_active, _active_nonempty=True)
                graphed.fuel_physics.step(
                    graph_active, graphed.piece_active, graphed.piece_owner,
                    dt=dt, substeps=1, adaptive=False)
        finally:
            graphed.sim.dt = outer_dt
        graphs[count] = graph
        return graph

    eager_counts, graph_counts = [], []
    error = None
    for tick in range(args.ticks):
        eager_count = eager_tick(eager, eager_command, eager_active)
        graph_count = substeps(graphed, graph_active)
        eager_counts.append(eager_count)
        graph_counts.append(graph_count)
        if eager_count != graph_count:
            error = f"tick {tick}: substeps {eager_count} != {graph_count}"
            break
        graph = get_graph(graph_count)
        outer_dt = graphed.sim.dt
        dt = outer_dt / graph_count
        graphed.sim.dt = dt
        try:
            for _ in range(graph_count):
                graph.replay()
        finally:
            graphed.sim.dt = outer_dt
        torch.cuda.synchronize(device)
        error = first_difference(state(eager), state(graphed))
        if error:
            error = f"tick {tick}: {error}"
            break

    if error:
        print(json.dumps({"status": "parity_failed", "ticks_checked": len(eager_counts),
                          "eager_substeps": eager_counts, "graph_substeps": graph_counts,
                          "difference": error}, indent=2))
        raise SystemExit(1)

    eager.reset(seed=args.seed)
    graphed.reset(seed=args.seed)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.ticks):
        eager_tick(eager, eager_command, eager_active)
    torch.cuda.synchronize(device)
    eager_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(args.ticks):
        count = substeps(graphed, graph_active)
        graph = get_graph(count)
        outer_dt = graphed.sim.dt
        dt = outer_dt / count
        graphed.sim.dt = dt
        try:
            for _ in range(count):
                graph.replay()
        finally:
            graphed.sim.dt = outer_dt
    torch.cuda.synchronize(device)
    graph_seconds = time.perf_counter() - start
    error = first_difference(state(eager), state(graphed))
    print(json.dumps({
        "status": "exact" if error is None else "parity_failed",
        "ticks": args.ticks, "seed": args.seed,
        "eager_seconds": eager_seconds, "graph_seconds": graph_seconds,
        "speedup": eager_seconds / max(graph_seconds, 1e-12),
        "substeps_per_tick": graph_counts,
        "graphs_captured": sorted(graphs), "difference": error,
        "scope": "simulator and HIP fuel substep only; controllers/perception excluded",
    }, indent=2, sort_keys=True))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
