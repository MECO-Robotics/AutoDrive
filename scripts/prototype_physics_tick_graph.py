#!/usr/bin/env python3
"""Test-only full-active strategic env.step graph-capture feasibility probe.

No production code is changed. This intentionally starts at a small batch and
captures no-info physics ticks while keeping AD* planning eager.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import re
import textwrap
import traceback
import types
import time
from pathlib import Path

import torch

from frc_defense.tensor_sim import TensorDefenseEnv
import frc_defense.tensor_sim as tensor_sim_module


def make_env(device: str, n: int, seed: int):
    return TensorDefenseEnv(
        num_envs=n, task="counter_defense", device=device, seed=seed,
        opponent="shadow", action_mode="strategic", horizon=8000,
        randomize=False, reuse_strategic_own_candidates=True,
        skip_strategic_offense_metric_objective=True,
        max_observation_latency_steps=0, max_control_latency_steps=0,
    )


def eager_external_plan(env, action, active):
    """Mirror _adstar_target_velocity cadence and planning before a tick."""
    env._planner_tick_scalar += 1
    env._planner_tick.add_(1)
    due = env._planner_tick_scalar % env.adstar_replan_interval == 0
    if not due:
        return
    env._last_strategic_action.copy_(action)
    target = env._strategic_target(env._last_strategic_action)
    pose = env.sim.pose[:, 0]
    margin = .5 * torch.maximum(env.sim.length[:, 0], env.sim.width[:, 0])
    low = margin[:, None]
    high = torch.stack((env.sim.field_length - margin,
                        env.sim.field_width - margin), dim=-1)
    target = torch.maximum(torch.minimum(target, high), low)
    env._adstar_tactical_planner.plan(
        pose[:, :2], target, pose[:, 2], env.sim.length[:, 0],
        env.sim.width[:, 0], env.sim.speed[:, 0],
        lateral_friction=env.sim.lateral_mu[:, 0],
        acceleration=env.sim.accel[:, 0], active_mask=active)


def eager_tick(env, action, active):
    return env.step(action, active_mask=active, _active_count=env.n,
                    _return_info=False)


def install_capture_friendly_observation(env):
    helper_path = Path(__file__).with_name("test_strategic_observation_graph.py")
    spec = importlib.util.spec_from_file_location("_strategic_graph_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    env._graph_field_scales = torch.tensor(
        (env.sim.field_length, env.sim.field_width,
         env.sim.field_length, env.sim.field_width),
        device=env.device, dtype=env.sim.pose.dtype)
    tensor_sim_module._full_strategic_observation_available = None
    source = textwrap.dedent(inspect.getsource(TensorDefenseEnv._strategic_observation))
    replacements = {
        "self.sim.length[:,[focal,other]]": "torch.stack((self.sim.length[:,focal],self.sim.length[:,other]),dim=1)",
        "self.sim.width[:,[focal,other]]": "torch.stack((self.sim.width[:,focal],self.sim.width[:,other]),dim=1)",
        "self.sim.accel[:,[focal,other]]": "torch.stack((self.sim.accel[:,focal],self.sim.accel[:,other]),dim=1)",
        "self.fuel_score_count[:,[focal,other]]": "torch.stack((self.fuel_score_count[:,focal],self.fuel_score_count[:,other]),dim=1)",
        "self.sim.speed[:,[focal,other]]": "torch.stack((self.sim.speed[:,focal],self.sim.speed[:,other]),dim=1)",
        "self.sim.omega_limit[:,[focal,other]]": "torch.stack((self.sim.omega_limit[:,focal],self.sim.omega_limit[:,other]),dim=1)",
    }
    for old, new in replacements.items():
        if old not in source:
            raise AssertionError(f"capture-friendly rewrite missed {old}")
        source = source.replace(old, new)
    source, count = re.subn(
        r"torch\.tensor\(\(self\.sim\.field_length,self\.sim\.field_width,"
        r"self\.sim\.field_length,\s*self\.sim\.field_width\),device=self\.device\)",
        "self._graph_field_scales", source)
    if count != 1:
        raise AssertionError("capture-friendly rewrite missed match scales")
    namespace = vars(tensor_sim_module).copy()
    namespace["_pack_fused_strategic_observation"] = module._capture_friendly_pack
    exec(compile(source, "<physics-tick-capture-observation>", "exec"), namespace)
    env._strategic_observation = types.MethodType(
        namespace["_strategic_observation"], env)


def tensor_state(obj, prefix="root", seen=None):
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return {}
    seen.add(oid)
    result = {}
    if torch.is_tensor(obj):
        result[prefix] = obj.detach().clone()
    elif isinstance(obj, (tuple, list)):
        for i, value in enumerate(obj):
            result.update(tensor_state(value, f"{prefix}[{i}]", seen))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            result.update(tensor_state(value, f"{prefix}.{key}", seen))
    elif hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("frc_defense"):
        for key, value in vars(obj).items():
            if key.startswith("_") or key in ("sim", "generator"):
                # Include private tensor state and nested planner state too.
                result.update(tensor_state(value, f"{prefix}.{key}", seen))
            elif torch.is_tensor(value):
                result[f"{prefix}.{key}"] = value.detach().clone()
    return result


def snapshot(env):
    state = tensor_state(env)
    state["env.generator_state"] = env.generator.get_state().clone()
    return state


def first_difference(left, right):
    keys = sorted(set(left) | set(right))
    for key in keys:
        if key not in left or key not in right:
            return f"state key {key} exists in only one environment"
        if left[key].shape != right[key].shape or left[key].dtype != right[key].dtype:
            return f"{key} shape/dtype differs: {left[key].shape}/{left[key].dtype} vs {right[key].shape}/{right[key].dtype}"
        if not torch.equal(left[key], right[key]):
            delta = (left[key].float() - right[key].float()).abs()
            return f"{key} first mismatch max_abs={delta.max().item()}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=28107)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("Requires a ROCm/HIP PyTorch device")
    device = torch.device(args.device)
    eager, graphed = make_env(args.device, args.envs, args.seed), make_env(args.device, args.envs, args.seed)
    eager.reset(seed=args.seed)
    graphed.reset(seed=args.seed)
    install_capture_friendly_observation(graphed)
    torch.cuda.synchronize(device)
    action = torch.full((args.envs,), 7, device=device, dtype=torch.long)
    active = torch.ones((args.envs,), device=device, dtype=torch.bool)

    # Prewarm kernels and planner on eager once, then reset both trajectories
    # from the same seed so the parity interval begins identically.
    for _ in range(2):
        eager_tick(eager, action, active)
        graphed.step(action, active_mask=active, _active_count=args.envs,
                     _return_info=False)
    eager.reset(seed=args.seed)
    graphed.reset(seed=args.seed)
    torch.cuda.synchronize(device)

    capture_error = None
    original_cadence = graphed._advance_adstar_cadence
    try:
        # AD* plan() has dynamic host-side work and remains eager. Cadence and
        # replanning are performed immediately before each graph replay.
        graphed._advance_adstar_cadence = lambda active_mask: (False, None)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_result = graphed.step(action, active_mask=active,
                                        _active_count=args.envs,
                                        _return_info=False)
        torch.cuda.synchronize(device)
        eager_results = None
        for _ in range(args.ticks):
            eager_results = eager_tick(eager, action, active)
            eager_obs = eager_results[0].clone()
            eager_reward = eager_results[1].clone()
            eager_done = eager_results[2].clone()
            eager_truncated = eager_results[3].clone()
            eager_external_plan(graphed, action, active)
            graph.replay()
            torch.cuda.synchronize(device)
            got = graph_result
            mismatch = None
            for name, expected, actual in (
                ("observation", eager_obs, got[0]),
                ("reward", eager_reward, got[1]),
                ("done", eager_done, got[2]),
                ("truncated", eager_truncated, got[3]),
            ):
                if not torch.equal(expected, actual):
                    mismatch = f"{name} differs, max_abs={(expected.float()-actual.float()).abs().max().item()}"
                    break
            if mismatch:
                raise AssertionError(f"tick parity failed: {mismatch}")
            state_diff = first_difference(snapshot(eager), snapshot(graphed))
            if state_diff:
                raise AssertionError(f"tick state parity failed: {state_diff}")
        result = {"status": "exact", "envs": args.envs, "ticks": args.ticks,
                  "capture_error": capture_error,
                  "graph_capture": "passed", "parity": "exact full tensor state, outputs, explicit generator state"}
    except Exception as exc:
        capture_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        result = {"status": "blocked", "envs": args.envs, "ticks": args.ticks,
                  "capture_error": capture_error}
    finally:
        graphed._advance_adstar_cadence = original_cadence
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if result["status"] != "exact":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
