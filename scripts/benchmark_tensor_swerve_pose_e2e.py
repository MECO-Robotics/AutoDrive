#!/usr/bin/env python3
"""Paired strategic-environment A/B/A/B/A benchmark for fused swerve + pose."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import torch

os.environ["AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_HIP"] = "1"
from frc_defense.tensor_sim import TensorDefenseEnv


def _equal(a, b, where):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
            if isinstance(b, torch.Tensor) and a.shape == b.shape:
                diff = (a - b).abs()
                more = f"max_abs={float(diff.max().item()):.9g}"
            else:
                more = f"shapes={getattr(a, 'shape', None)}/{getattr(b, 'shape', None)}"
            raise AssertionError(f"{where} differs: {more}")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f"{where} mapping keys differ")
        for k in a:
            _equal(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(f"{where} sequence shape differs")
        for i, (x, y) in enumerate(zip(a, b)):
            _equal(x, y, f"{where}[{i}]")
    elif a is None or isinstance(a, (bool, int, float, str)):
        if a != b:
            raise AssertionError(f"{where}: {a!r} != {b!r}")


def _views(env):
    result = {f"env.{k}": v for k, v in vars(env).items()
              if isinstance(v, torch.Tensor)}
    result.update({f"sim.{k}": v for k, v in vars(env.sim).items()
                  if isinstance(v, torch.Tensor)})
    for name in ("_adstar_planners", "_adstar_defender_planners",
                 "_adstar_tactical_planner"):
        planner = getattr(env, name, None)
        if planner is not None:
            result.update({f"{name}.{k}": v for k, v in vars(planner).items()
                           if isinstance(v, torch.Tensor)})
    result.update({
        "scalar.planner_tick_scalar": env._planner_tick_scalar,
        "scalar.planner_tick_aligned": env._planner_tick_aligned,
        "scalar.planner_full_batch_active": env._planner_full_batch_active,
        "rng.env": env.generator.get_state(),
        "rng.sim": env.sim.generator.get_state(),
    })
    return result


def _make_pair(worlds, device):
    env_name = "AUTODRIVE_FUSED_SWERVE_POSE_HIP"
    os.environ[env_name] = "0"
    torch_env = TensorDefenseEnv(
        num_envs=worlds, device=device, seed=98031,
        task="counter_defense", opponent="guard", action_mode="strategic",
        randomize=False, observation_noise=0.0, observation_dropout=0.0,
        perception_config={"detection_dropout": 0.0, "position_noise_m": 0.0},
        horizon=8000)
    os.environ[env_name] = "1"
    hip_env = TensorDefenseEnv(
        num_envs=worlds, device=device, seed=98031,
        task="counter_defense", opponent="guard", action_mode="strategic",
        randomize=False, observation_noise=0.0, observation_dropout=0.0,
        perception_config={"detection_dropout": 0.0, "position_noise_m": 0.0},
        horizon=8000)
    if not hip_env.sim._fused_swerve_pose_hip_enabled:
        raise AssertionError("candidate env did not enable fused swerve + pose")
    return torch_env, hip_env


def _run(env, seed, action_bank, ticks_per_decision):
    env.reset(seed=seed)
    trace = []
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for action in action_bank:
        for _ in range(ticks_per_decision):
            obs, reward, done, truncated, info = env.step(
                action, _return_info=False)
            trace.append((obs.clone(), reward.clone(), done.clone(),
                          truncated.clone(), info))
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / 1000.0, trace


def _pair_check(torch_env, hip_env, trace_a, trace_b, where):
    _equal(trace_a, trace_b, f"{where}.reward-observation-trace")
    _equal(_views(torch_env), _views(hip_env), f"{where}.full-state")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--decisions", type=int, default=4)
    parser.add_argument("--ticks-per-decision", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        print(json.dumps({"status": "skipped", "reason": "HIP unavailable"}))
        return 0
    free, total = torch.cuda.mem_get_info()
    if free < 3 * 1024**3:
        print(json.dumps({"status": "skipped", "reason": "less than 3 GiB free",
                          "free_gib": free / 1024**3,
                          "total_gib": total / 1024**3}))
        return 0
    torch_env, hip_env = _make_pair(args.envs, args.device)
    from frc_defense import tensor_swerve_pose_hip
    tensor_swerve_pose_hip.extension()
    gen = torch.Generator(device=args.device).manual_seed(13103)
    action_bank = torch.randint(0, 8, (args.decisions, args.envs),
                                device=args.device, generator=gen)
    seed = 98031

    # Warm both variants, check parity, then reset for measurement.
    warm_actions = action_bank[:1]
    _, warm_a = _run(torch_env, seed, warm_actions, args.ticks_per_decision)
    _, warm_b = _run(hip_env, seed, warm_actions, args.ticks_per_decision)
    _pair_check(torch_env, hip_env, warm_a, warm_b, "warmup")
    del warm_a, warm_b

    measurements = []
    pair_traces = []
    for phase, env, tag in (
        ("A1", torch_env, "torch"), ("B1", hip_env, "hip"),
        ("A2", torch_env, "torch"), ("B2", hip_env, "hip"),
        ("A3", torch_env, "torch"),
    ):
        seconds, trace = _run(env, seed, action_bank, args.ticks_per_decision)
        measurements.append({"phase": phase, "variant": tag, "seconds": seconds})
        pair_traces.append(trace)
        if phase in ("B1", "B2"):
            _pair_check(torch_env, hip_env, pair_traces[-2], pair_traces[-1], phase)
            del pair_traces[-2:]

    # Untimed B3 is a parity sentinel for the final A3 while preserving the
    # requested measured A/B/A/B/A ordering.
    _, sentinel_b = _run(hip_env, seed, action_bank, args.ticks_per_decision)
    _pair_check(torch_env, hip_env, pair_traces[-1], sentinel_b, "A3/B3-sentinel")
    del pair_traces, sentinel_b

    a = [m["seconds"] for m in measurements if m["variant"] == "torch"]
    b = [m["seconds"] for m in measurements if m["variant"] == "hip"]
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    result = {
        "status": "exact-e2e-parity-passed",
        "environment": "TensorDefenseEnv strategic, AD* + fused observation active in both variants",
        "worlds": args.envs,
        "strategic_decisions": args.decisions,
        "physics_ticks_per_decision": args.ticks_per_decision,
        "physics_ticks_per_run": args.decisions * args.ticks_per_decision,
        "initial_free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
        "measured_phases": measurements,
        "mean_torch_seconds": mean_a,
        "mean_hip_seconds": mean_b,
        "e2e_speedup": mean_a / mean_b,
        "parity_pairs": 3,
        "final_hip_sentinel": "exact",
        "default_enabled_on_hip": True,
        "opt_out": "AUTODRIVE_FUSED_SWERVE_POSE_HIP=0",
    }
    out_dir = Path("evaluations/strategic-ppo-validation-20261001")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "swerve_pose_e2e_ababa.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["artifact"] = str(out_path.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
