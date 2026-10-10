#!/usr/bin/env python3
"""Check repeated graph replay parity/speed for no-replan 3v3 physics ticks."""
from __future__ import annotations

import argparse
import json
import time

import torch

from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def make_env(device: str, n: int, seed: int) -> TensorThreeVsThreeEnv:
    env = TensorThreeVsThreeEnv(
        num_envs=n, device=device, seed=seed,
        control_modes=("deterministic", "deterministic", "deterministic", "nn", "nn", "nn"),
        horizon=8000, randomize=False, perception_dropout=0.,
        position_noise=0., velocity_noise=0., fused_sensor_rng=True,
        perception_interval=1,
    )
    env.reset(seed=seed)
    # The outer-step capture probe uses a fixed five-substep physics schedule.
    # Adaptive substeps call .item() and the normal inner physics graph cannot
    # be nested inside the graph being measured.
    if env.fuel_physics is not None:
        env.fuel_physics.config.adaptive_substeps = False
    env._physics_graph_enabled = False
    return env


def set_no_replan_phase(env: TensorThreeVsThreeEnv) -> None:
    env.planner_ticks.zero_()
    env._planner_tick_scalar = 0
    env._planner_tick_aligned = True


def tensor_snapshot(env: TensorThreeVsThreeEnv) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    seen: set[int] = set()

    def visit(obj, prefix: str, depth: int) -> None:
        if depth > 3 or id(obj) in seen:
            return
        if torch.is_tensor(obj):
            snapshot[prefix] = obj.detach().clone()
            return
        if isinstance(obj, (str, bytes, int, float, bool, type(None))):
            return
        if isinstance(obj, (tuple, list)):
            seen.add(id(obj))
            for i, value in enumerate(obj):
                visit(value, f"{prefix}[{i}]", depth + 1)
            return
        if isinstance(obj, dict):
            seen.add(id(obj))
            for key, value in obj.items():
                visit(value, f"{prefix}.{key}", depth + 1)
            return
        if hasattr(obj, "__dict__"):
            seen.add(id(obj))
            for key, value in vars(obj).items():
                if key in ("generator", "generator_state", "_compiled_step",
                           "_target_action_graph_actions",
                           "_target_action_graph_active",
                           "_target_action_graph_outputs",
                           "_target_action_graph_audit",
                           "_physics_graph_command", "_physics_graph_active",
                           "_hip"):
                    continue
                visit(value, f"{prefix}.{key}", depth + 1)

    visit(env, "env", 0)
    return snapshot


def first_difference(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]):
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            return f"{key} is missing on one side"
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            return f"{key} shape/dtype differs"
        if not torch.equal(a[key], b[key]):
            delta = (a[key].float() - b[key].float()).abs()
            return f"{key} differs (max_abs={delta.max().item()})"
    return None


def step(env: TensorThreeVsThreeEnv, action: torch.Tensor) -> None:
    env.step(action, active_mask=None, capture_observation=False,
             capture_info=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--ticks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=64821)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("Requires a ROCm/HIP device")
    if args.envs < 1 or not 1 <= args.ticks < 20:
        raise SystemExit("envs must be positive and ticks must be in [1, 19]")

    device = torch.device(args.device)
    eager = make_env(args.device, args.envs, args.seed)
    graphed = make_env(args.device, args.envs, args.seed)
    action_storage = torch.full((args.envs, 6), 7, device=device, dtype=torch.long)
    actions = [torch.full_like(action_storage, 7) for _ in range(args.ticks)]
    # Exercise policy-controlled action changes while keeping this interval
    # before the next AD* replan boundary.
    for t in range(args.ticks):
        actions[t][:, :3] = t % 8
        actions[t][:, 3:] = 5

    # Warm compiled HIP modules, then reset both trajectories identically.
    for _ in range(2):
        step(eager, action_storage)
        step(graphed, action_storage)
    eager.reset(seed=args.seed)
    graphed.reset(seed=args.seed)
    set_no_replan_phase(eager)
    set_no_replan_phase(graphed)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    action_storage.copy_(actions[0])
    with torch.cuda.graph(graph):
        graph_output = graphed.step(action_storage, active_mask=None,
                                    capture_observation=False, capture_info=False)
    # Capture records operations without applying their tensor updates; the
    # first replay performs the same tick as one eager call.
    graph.replay()
    graphed._planner_tick_scalar = 1
    graphed._full_batch_step_scalar = 1
    step(eager, actions[0])
    if graphed._planner_tick_scalar != eager._planner_tick_scalar:
        raise AssertionError("planner host counters differ after graph capture")
    torch.cuda.synchronize(device)
    difference = first_difference(tensor_snapshot(eager), tensor_snapshot(graphed))
    if difference:
        raise AssertionError(f"capture parity failed: {difference}")

    # Repeat dynamic action inputs. Python's aligned planner scalar is advanced
    # alongside the graph's captured device counter.
    for t in range(1, args.ticks):
        step(eager, actions[t])
        action_storage.copy_(actions[t])
        graph.replay()
        graphed._planner_tick_scalar += 1
        graphed._full_batch_step_scalar += 1
        if graphed._planner_tick_scalar != eager._planner_tick_scalar:
            raise AssertionError("planner host counters differ during replay")
    torch.cuda.synchronize(device)
    difference = first_difference(tensor_snapshot(eager), tensor_snapshot(graphed))
    if difference:
        raise AssertionError(f"replay parity failed: {difference}")

    # Matched timing blocks from reset, after graph capture and warmup.
    eager.reset(seed=args.seed)
    graphed.reset(seed=args.seed)
    set_no_replan_phase(eager)
    set_no_replan_phase(graphed)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for action in actions:
        step(eager, action)
    torch.cuda.synchronize(device)
    eager_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for t, action in enumerate(actions):
        action_storage.copy_(action)
        graph.replay()
        graphed._planner_tick_scalar += 1
        graphed._full_batch_step_scalar += 1
    torch.cuda.synchronize(device)
    graph_seconds = time.perf_counter() - start
    difference = first_difference(tensor_snapshot(eager), tensor_snapshot(graphed))
    report = {
        "status": "exact" if difference is None else "parity_failed",
        "envs": args.envs,
        "ticks": args.ticks,
        "eager_seconds": eager_seconds,
        "graph_seconds": graph_seconds,
        "step_graph_speedup": eager_seconds / max(graph_seconds, 1e-12),
        "difference": difference,
        "scope": "3v3 TensorThreeVsThreeEnv.step, full active batch, no-replan interval",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if difference is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
