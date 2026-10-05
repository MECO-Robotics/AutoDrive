#!/usr/bin/env python3
"""Isolated HIP graph capture parity test for raw strategic observations.

This captures only TensorDefenseEnv._strategic_observation. It deliberately
does not capture _obs history/noise, env.step, AD* planning, reset, or PPO.
Run only when a GPU has been explicitly released for graph testing.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import textwrap
import time
import types
from pathlib import Path

import torch

from frc_defense.tensor_sim import TensorDefenseEnv
import frc_defense.tensor_sim as tensor_sim_module
from frc_defense import tensor_strategic_observation_proto as strategic_proto


def _env(device: str, envs: int, seed: int) -> TensorDefenseEnv:
    return TensorDefenseEnv(
        num_envs=envs, task="counter_defense", device=device, seed=seed,
        opponent="guard", action_mode="strategic", horizon=8000,
        randomize=False, reuse_strategic_own_candidates=True,
        skip_strategic_offense_metric_objective=True,
    )


def _pending(env: TensorDefenseEnv):
    pending = env._pending_strategic_own_candidates
    if pending is None:
        return None
    data, mask = pending
    return tuple(x.clone() for x in data), mask.clone()


def _assert_pending_equal(left, right, label: str):
    if (left is None) != (right is None):
        raise AssertionError(f"{label}: pending-cache presence differs")
    if left is None:
        return
    left_indices, left_valid, left_nearest = left[0]
    right_indices, right_valid, right_nearest = right[0]
    if not torch.equal(left_valid, right_valid):
        raise AssertionError(f"{label}: candidate cache validity differs")
    valid = left_valid
    if not torch.equal(left_indices[valid], right_indices[valid]):
        raise AssertionError(f"{label}: valid candidate indices differ")
    if not torch.equal(left_nearest[valid], right_nearest[valid]):
        raise AssertionError(f"{label}: valid nearest candidate distances differ")
    if not torch.equal(left[1], right[1]):
        raise AssertionError(f"{label}: action-mask cache differs")


def _set_candidate_state(env: TensorDefenseEnv, enabled: bool):
    env._track_mask.zero_()
    if enabled:
        rows = torch.arange(env.n, device=env.device)
        # Vary candidate count by row, retaining a fixed tensor shape.
        counts = (rows % 5) + 1
        slots = torch.arange(env.fuel_count, device=env.device)[None, :]
        env._track_mask[:, 0] = slots < counts[:, None]
        env._track_pos[:, 0, :, 0] = 1.0 + slots.to(torch.float32) * .001
        env._track_pos[:, 0, :, 1] = 2.0 + rows[:, None] * .0001
        env._track_age[:, 0] = torch.where(env._track_mask[:, 0], 0.0, float("inf"))
        env._track_age[:, 1] = float("inf")


def _capture_friendly_method():
    """Make an isolated method copy replacing list indices with scalar stacks."""
    source = textwrap.dedent(inspect.getsource(TensorDefenseEnv._strategic_observation))
    replacements = {
        "self.sim.length[:,[focal,other]]":
            "torch.stack((self.sim.length[:,focal],self.sim.length[:,other]),dim=1)",
        "self.sim.width[:,[focal,other]]":
            "torch.stack((self.sim.width[:,focal],self.sim.width[:,other]),dim=1)",
        "self.sim.accel[:,[focal,other]]":
            "torch.stack((self.sim.accel[:,focal],self.sim.accel[:,other]),dim=1)",
        "self.fuel_score_count[:,[focal,other]]":
            "torch.stack((self.fuel_score_count[:,focal],self.fuel_score_count[:,other]),dim=1)",
        "self.sim.speed[:,[focal,other]]":
            "torch.stack((self.sim.speed[:,focal],self.sim.speed[:,other]),dim=1)",
        "self.sim.omega_limit[:,[focal,other]]":
            "torch.stack((self.sim.omega_limit[:,focal],self.sim.omega_limit[:,other]),dim=1)",
    }
    for old, new in replacements.items():
        if old not in source:
            raise AssertionError(f"capture-friendly rewrite did not find {old}")
        source = source.replace(old, new)
    source, constant_count = re.subn(
        r"torch\.tensor\(\(self\.sim\.field_length,self\.sim\.field_width,"
        r"self\.sim\.field_length,\s*self\.sim\.field_width\),device=self\.device\)",
        "self._graph_field_scales", source)
    if constant_count != 1:
        raise AssertionError("capture-friendly rewrite did not find match field scales")
    cache_old = (
        "            merged_data=tuple(torch.where(selected[:,None],current,previous)\n"
        "                              for current,previous in zip(candidate_data,previous_data))\n"
        "            merged_mask=torch.where(selected[:,None],action_mask,previous_mask)\n"
        "            self._pending_strategic_own_candidates=(merged_data,merged_mask)")
    cache_new = (
        "            for current,previous in zip(candidate_data,previous_data):\n"
        "                previous.copy_(torch.where(selected[:,None],current,previous))\n"
        "            previous_mask.copy_(torch.where(selected[:,None],action_mask,previous_mask))\n"
        "            self._pending_strategic_own_candidates=(previous_data,previous_mask)")
    if cache_old not in source:
        raise AssertionError("capture-friendly rewrite did not find candidate cache merge")
    source = source.replace(cache_old, cache_new)
    namespace = vars(tensor_sim_module).copy()
    namespace["_pack_fused_strategic_observation"] = _capture_friendly_pack
    exec(compile(source, "<capture-friendly-strategic-observation>", "exec"), namespace)
    return namespace["_strategic_observation"]


def _capture_friendly_pack(env, focal, *, base, route, local1, possession,
                           match, action_values, other_pose, own_count,
                           candidate_data, normalize):
    """Fused pack helper with capture-safe scalar indexing for speed/omega."""
    if strategic_proto._extension() is None:
        raise RuntimeError("fused strategic observation extension is unavailable")
    tracked_points, tracked_velocities, _ = env._perceived_fuel(focal)
    indices, valid, nearest = candidate_data
    other = 1 - focal
    speed = torch.stack((env.sim.speed[:, focal], env.sim.speed[:, other]), dim=1).contiguous()
    omega = torch.stack((env.sim.omega_limit[:, focal], env.sim.omega_limit[:, other]), dim=1).contiguous()
    return strategic_proto._extension().pack_fused(
        base.contiguous(), route.contiguous(), local1.contiguous(),
        possession.contiguous(), match.contiguous(), action_values.contiguous(),
        speed, omega, tracked_points, tracked_velocities, indices.contiguous(),
        valid.contiguous(), nearest.contiguous(), env.sim.pose[:, focal].contiguous(),
        other_pose[:, :2].contiguous(), own_count.to(base.dtype).contiguous(),
        env.sim.length[:, focal].contiguous(), env.sim.width[:, focal].contiguous(),
        env.hub_centers[focal].contiguous(), float(env.sim.field_length),
        float(env.sim.field_width), int(env.fuel_capacity), bool(normalize))


def _run(device: str, envs: int, seed: int, measure: bool = False) -> dict:
    eager = _env(device, envs, seed)
    captured = _env(device, envs, seed)
    active_static = torch.ones((envs,), device=device, dtype=torch.bool)

    # Warm lazy extensions and allocator before capture. Keep both environments
    # aligned, including the pending candidate cache mutation.
    eager._strategic_observation(0, active_mask=active_static)
    captured._strategic_observation(0, active_mask=active_static)
    torch.cuda.synchronize()

    eager_raw_before = eager._strategic_observation(0, active_mask=active_static)
    graph = torch.cuda.CUDAGraph()
    captured._graph_field_scales = torch.tensor(
        (captured.sim.field_length, captured.sim.field_width,
         captured.sim.field_length, captured.sim.field_width),
        device=device, dtype=captured.sim.pose.dtype)
    captured._strategic_observation = types.MethodType(
        _capture_friendly_method(), captured)
    capture_friendly_eager = captured._strategic_observation(
        0, active_mask=active_static)
    if not torch.equal(eager_raw_before, capture_friendly_eager):
        raise AssertionError("capture-friendly eager output differs from production eager")
    _assert_pending_equal(_pending(eager), _pending(captured),
                          "capture-friendly eager")
    with torch.cuda.graph(graph):
        graph_raw = captured._strategic_observation(0, active_mask=active_static)
    torch.cuda.synchronize()

    cases = []
    rows = torch.arange(envs, device=device)
    scenarios = (
        ("all_active_candidates", torch.ones_like(active_static), True),
        ("mixed_active_candidates", (rows % 3) != 1, True),
        ("all_inactive_candidates", torch.zeros_like(active_static), True),
        ("mixed_active_empty_candidates", (rows % 2) == 0, False),
        ("all_active_empty_candidates", torch.ones_like(active_static), False),
    )
    for label, active, candidates in scenarios:
        _set_candidate_state(eager, candidates)
        _set_candidate_state(captured, candidates)
        active_static.copy_(active)
        torch.cuda.synchronize()
        eager_raw = eager._strategic_observation(0, active_mask=active)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(eager_raw, graph_raw):
            mismatch = int((eager_raw != graph_raw).sum().item())
            raise AssertionError(f"{label}: raw observation differs in {mismatch} values")
        _assert_pending_equal(_pending(eager), _pending(captured), label)
        cases.append({"case": label, "raw_exact": True, "cache_exact": True})
    timing = None
    if measure:
        # Warm both paths before timing. Each timed block is a batch with one
        # synchronization at either end, so the measurement includes host
        # launch overhead and completed device work without syncing per call.
        all_active = torch.ones((envs,), device=device, dtype=torch.bool)
        _set_candidate_state(eager, True)
        _set_candidate_state(captured, True)
        active_static.copy_(all_active)
        for _ in range(8):
            eager_last = eager._strategic_observation(0, active_mask=all_active)
            graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(eager_last, graph_raw):
            raise AssertionError("post-warmup eager/graph observation mismatch")
        _assert_pending_equal(_pending(eager), _pending(captured), "timing warmup")

        repeats = 60
        timings = []
        for label, use_graph in (("eager_a", False), ("graph_b", True),
                                 ("eager_a_repeat", False), ("graph_b_repeat", True),
                                 ("eager_a_final", False)):
            torch.cuda.synchronize()
            started = time.perf_counter()
            last = None
            if use_graph:
                for _ in range(repeats):
                    graph.replay()
            else:
                for _ in range(repeats):
                    last = eager._strategic_observation(0, active_mask=all_active)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if use_graph:
                last = graph_raw
            timings.append({"variant": label, "replays": repeats,
                            "elapsed_seconds": elapsed,
                            "seconds_per_observation": elapsed / repeats})
        if not torch.equal(eager_last, graph_raw):
            raise AssertionError("timed eager/graph observation mismatch")
        _assert_pending_equal(_pending(eager), _pending(captured), "timed cache")
        eager_mean = sum(t["seconds_per_observation"] for t in timings
                         if t["variant"].startswith("eager")) / 3
        graph_mean = sum(t["seconds_per_observation"] for t in timings
                         if t["variant"].startswith("graph")) / 2
        timing = {"replays_per_block": repeats, "order": [t["variant"] for t in timings],
                  "blocks": timings, "eager_mean_seconds_per_observation": eager_mean,
                  "graph_mean_seconds_per_observation": graph_mean,
                  "speedup": eager_mean / graph_mean,
                  "timing_note": "host wall time for batched calls; one device sync at each block boundary"}
    return {"envs": envs, "cases": cases, "all_exact": True, "timing": timing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=415903)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing-output", type=Path)
    parser.add_argument("--measure", action="store_true",
                        help="run the warmed repeated eager-versus-graph timing")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("This graph parity harness requires a ROCm/HIP device.")
    if not hasattr(torch.cuda, "CUDAGraph"):
        raise SystemExit("Installed PyTorch build does not expose CUDAGraph.")
    report = {
        "workload": "raw strategic observation only",
        "device": args.device,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "excluded": ["_obs history/noise", "env.step", "AD* planning", "reset", "PPO"],
        "result": _run(args.device, args.envs, args.seed, measure=args.measure),
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if args.timing_output is not None and report["result"]["timing"] is not None:
        args.timing_output.parent.mkdir(parents=True, exist_ok=True)
        args.timing_output.write_text(json.dumps(report["result"]["timing"], indent=2) + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
