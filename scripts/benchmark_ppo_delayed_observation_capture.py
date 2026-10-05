#!/usr/bin/env python3
"""Parity and throughput check for PPO delayed-observation row capture.

The default run uses a small fixed-seed batch and checks 12/13 tick intervals,
per-world latency 0/8, rewards, returned action masks, RNG, simulator state,
terminal horizon observations, and post-reset observations. Add
--benchmark-envs 24576 for a warmed rollout-only throughput comparison.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from frc_defense.tensor_training import _env, _held_action_interval


def _make_env(n, device, seed, *, fast, horizon=8000):
    env = _env(n, "counter_defense", device, seed, "guard", horizon,
               architecture="strategic_adstar")
    obs, _ = env.reset()
    env._ppo_sparse_delayed_observation_capture = bool(
        fast and env._full_observation_hip_available)
    return env, obs


def _assert_equal(a, b, label):
    if not torch.equal(a, b):
        if a.dtype.is_floating_point:
            diff = (a - b).abs()
            detail = f"max_abs={float(diff.max().item())}"
        else:
            detail = "tensor values differ"
        raise AssertionError(f"{label}: {detail}")


def _assert_environment_parity(legacy, fast, label):
    for name, left in vars(legacy).items():
        right = getattr(fast, name, None)
        if isinstance(left, torch.Tensor):
            if name == "observation_history":
                # Nonselected raw history slots are intentionally not written
                # by the PPO-only fast path. Returned delayed observations and
                # the ring indices are checked separately.
                continue
            if isinstance(right, torch.Tensor):
                _assert_equal(left, right, f"{label}.{name}")
    for name, left in vars(legacy.sim).items():
        right = getattr(fast.sim, name, None)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            _assert_equal(left, right, f"{label}.sim.{name}")
    for planner_name in ("_adstar_planners", "_adstar_defender_planners",
                         "_adstar_tactical_planner"):
        left_planner = getattr(legacy, planner_name, None)
        right_planner = getattr(fast, planner_name, None)
        if left_planner is None:
            continue
        for name, left in vars(left_planner).items():
            right = getattr(right_planner, name, None)
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                _assert_equal(left, right, f"{label}.{planner_name}.{name}")
    _assert_equal(legacy.generator.get_state(), fast.generator.get_state(),
                  f"{label}.generator")


def _interval(legacy, fast, action, ticks, remaining, label):
    args = dict(known_remaining_ticks=remaining, return_info=False)
    l = _held_action_interval(legacy, action, ticks, legacy._last_observation, **args)
    f = _held_action_interval(fast, action, ticks, fast._last_observation, **args)
    lr, ld, lt, lo = l[:4]
    fr, fd, ft, fo = f[:4]
    _assert_equal(lr, fr, f"{label}.reward")
    _assert_equal(ld, fd, f"{label}.done")
    _assert_equal(lt, ft, f"{label}.truncated")
    _assert_equal(lo, fo, f"{label}.observation")
    _assert_equal(lo[:, -8:].bool(), fo[:, -8:].bool(), f"{label}.action_mask")
    _assert_environment_parity(legacy, fast, label)
    return l, f


def parity(device):
    n, seed = 18, 441903
    legacy, legacy_obs = _make_env(n, device, seed, fast=False)
    fast, fast_obs = _make_env(n, device, seed, fast=True)
    if not (fast._full_observation_hip_available and
            fast._ppo_sparse_delayed_observation_capture):
        raise RuntimeError("fast capture unavailable; refusing false parity pass")
    _assert_equal(legacy_obs, fast_obs, "initial observation")
    # Explicitly include both latency endpoints and all-world cache refresh.
    delays = torch.tensor(([0, 8] * (n // 2) + ([0] if n % 2 else [])),
                          device=device, dtype=torch.long)
    legacy.observation_delay.copy_(delays)
    fast.observation_delay.copy_(delays)
    action = torch.full((n,), 7, device=device, dtype=torch.long)
    for index, ticks in enumerate((12, 13, 12, 13)):
        _interval(legacy, fast, action, ticks, 7000, f"interval_{index}_{ticks}")

    # Reproduce a 15-tick match remainder: legacy fallback must retain source
    # rows needed by terminal observations whose delay exceeds the final
    # interval's remaining tick count.
    for env in (legacy, fast):
        env.steps.fill_(env.horizon - 15)
        env.match_elapsed.fill_(159.7)
        env.match_remaining.fill_(.3)
    _interval(legacy, fast, action, 12, 15, "horizon_penultimate")
    terminal_legacy, terminal_fast = _interval(
        legacy, fast, action, 12, 3, "horizon_terminal")
    _assert_equal(terminal_legacy[2], torch.ones_like(terminal_legacy[2]),
                  "legacy terminal mask")
    _assert_equal(terminal_fast[2], torch.ones_like(terminal_fast[2]),
                  "fast terminal mask")
    reset_legacy = legacy.reset_done(terminal_legacy[2])
    reset_fast = fast.reset_done(terminal_fast[2])
    _assert_equal(reset_legacy, reset_fast, "post-reset observation")
    _assert_environment_parity(legacy, fast, "post_reset")
    return {
        "status": "passed",
        "envs": n,
        "device": str(device),
        "interval_ticks": [12, 13, 12, 13],
        "forced_observation_delays": [0, 8],
        "terminal_remainder_ticks": 15,
        "terminal_truncations": int(terminal_fast[2].sum().item()),
        "generator_and_state_parity": True,
        "post_reset_parity": True,
        "full_observation_hip_available": fast._full_observation_hip_available,
        "sparse_capture_enabled": fast._ppo_sparse_delayed_observation_capture,
    }


def _rollout_case(n, device, seed, fast, decisions, warmup):
    env, obs = _make_env(n, device, seed, fast=fast)
    action = torch.full((n,), 7, device=device, dtype=torch.long)
    phase = 0.
    episode_ticks = 0
    for _ in range(warmup):
        phase += 12.5
        ticks = max(1, int(phase))
        phase -= ticks
        _held_action_interval(env, action, ticks, obs,
                              known_remaining_ticks=8000 - episode_ticks,
                              return_info=False)
        episode_ticks += ticks
        obs = env._last_observation
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    transitions = 0
    physics_world_ticks = 0
    for _ in range(decisions):
        phase += 12.5
        ticks = max(1, int(phase))
        phase -= ticks
        result = _held_action_interval(
            env, action, ticks, obs,
            known_remaining_ticks=8000 - episode_ticks,
            return_info=False)
        transitions += n
        physics_world_ticks += result[6]
        episode_ticks += ticks
        obs = result[3]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "fast_capture": fast,
        "envs": n,
        "decisions": decisions,
        "transitions": transitions,
        "physics_world_ticks": physics_world_ticks,
        "elapsed_seconds": elapsed,
        "strategic_transitions_per_second": transitions / elapsed,
        "physics_world_ticks_per_second": physics_world_ticks / elapsed,
        "full_observation_hip_available": env._full_observation_hip_available,
        "sparse_capture_enabled": env._ppo_sparse_delayed_observation_capture,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark-envs", type=int, default=0)
    parser.add_argument("--benchmark-decisions", type=int, default=32)
    parser.add_argument("--warmup-decisions", type=int, default=3)
    parser.add_argument("--reverse-order", action="store_true",
                        help="run sparse before legacy to reduce order bias")
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device unavailable: {device}")
    report = {"parity": parity(device)}
    if args.benchmark_envs:
        cases = []
        order = (True, False) if args.reverse_order else (False, True)
        for fast in order:
            cases.append(_rollout_case(
                args.benchmark_envs, device, 441904, fast,
                args.benchmark_decisions, args.warmup_decisions))
        report["benchmark"] = cases
        by_fast = {case["fast_capture"]: case for case in cases}
        report["throughput_ratio_fast_over_legacy"] = (
            by_fast[True]["strategic_transitions_per_second"] /
            by_fast[False]["strategic_transitions_per_second"])
    serialized = json.dumps(report, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
