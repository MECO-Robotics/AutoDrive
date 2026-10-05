#!/usr/bin/env python3
"""Reproducible Torch-vs-production-HIP path_reference rollout parity audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

# Must be enabled before importing tensor_adstar so the production loader is
# exercised when the fused branch is selected.
os.environ["AUTODRIVE_FUSED_PATH_REFERENCE_HIP"] = "1"
from frc_defense import tensor_adstar
from frc_defense.tensor_sim import TensorDefenseEnv


def owners(env):
    result = {"env": env, "sim": env.sim}
    for name in ("_adstar_tactical_planner", "_adstar_planners",
                 "_adstar_defender_planners"):
        value = getattr(env, name, None)
        if isinstance(value, (tuple, list)):
            result.update({f"{name}.{i}": item for i, item in enumerate(value)})
        elif value is not None:
            result[name] = value
    return result


def state_map(env):
    values = {}
    for label, owner in owners(env).items():
        for name, value in vars(owner).items():
            if torch.is_tensor(value):
                values[f"{label}.{name}"] = value.detach()
            elif isinstance(value, (bool, int, float, str, type(None))):
                values[f"{label}.{name}"] = value
    return values


def compare(a, b):
    sa, sb = state_map(a), state_map(b)
    keys = sorted(set(sa) | set(sb))
    missing = [key for key in keys if key not in sa or key not in sb]
    mismatches, details, max_abs, max_key = [], [], 0.0, None
    for key in keys:
        if key not in sa or key not in sb:
            continue
        va, vb = sa[key], sb[key]
        if torch.is_tensor(va) and torch.is_tensor(vb):
            if va.shape != vb.shape or va.dtype != vb.dtype:
                mismatches.append(key)
                details.append({"key": key, "shape_a": list(va.shape), "shape_b": list(vb.shape),
                                "dtype_a": str(va.dtype), "dtype_b": str(vb.dtype)})
                continue
            equal = torch.equal(va, vb)
            if not equal:
                mismatches.append(key)
                detail = {"key": key, "shape": list(va.shape), "dtype": str(va.dtype)}
                if va.is_floating_point():
                    diff = (va - vb).abs()
                    local = float(torch.nan_to_num(diff, nan=float("inf")).max().item())
                    detail["max_abs_delta"] = local
                    if local > max_abs:
                        max_abs, max_key = local, key
                else:
                    detail["different_elements"] = int((va != vb).sum().item())
                details.append(detail)
        elif va != vb:
            mismatches.append(key)
            details.append({"key": key, "value_a": repr(va), "value_b": repr(vb)})
    return {"equal": not missing and not mismatches,
            "mismatch_count": len(mismatches), "missing_count": len(missing),
            "mismatches": details,
            "first_mismatch": (mismatches[0] if mismatches else (missing[0] if missing else None)),
            "max_abs_float_delta": max_abs, "max_delta_key": max_key}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=96)
    parser.add_argument("--seed", type=int, default=741029)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("A HIP-enabled PyTorch build and visible HIP GPU are required")

    # Build/load using the same isolated strict-math loader used by production.
    extension = tensor_adstar._hip_path_reference_extension()
    if extension is None or not hasattr(extension, "path_reference"):
        raise SystemExit("Production AD* HIP extension did not load path_reference")
    cuda_flags = ["-O3", "-ffp-contract=off"]

    def make_env():
        return TensorDefenseEnv(
            num_envs=args.envs, task="counter_defense", device=device, seed=args.seed,
            opponent="guard", action_mode="strategic", horizon=8000,
            skip_strategic_offense_metric_objective=True,
            reuse_strategic_opponent_candidates=True)

    torch_env = make_env()
    hip_env = make_env()
    torch.cuda.synchronize(device)
    init = compare(torch_env, hip_env)
    if not init["equal"]:
        raise RuntimeError(f"Initial states differ before stepping: {init}")
    action = torch.zeros((args.envs,), dtype=torch.int64, device=device)
    per_tick = []
    started = time.perf_counter()
    for tick in range(args.ticks):
        tensor_adstar._HIP_PATH_REFERENCE_ENABLED = False
        out_t = torch_env.step(action)
        tensor_adstar._HIP_PATH_REFERENCE_ENABLED = True
        out_h = hip_env.step(action)
        torch.cuda.synchronize(device)
        transition = []
        for index, (left, right) in enumerate(zip(out_t[:4], out_h[:4])):
            if torch.is_tensor(left) and torch.is_tensor(right):
                equal = torch.equal(left, right)
                delta = 0.0
                if not equal and left.is_floating_point():
                    delta = float(torch.nan_to_num((left-right).abs(), nan=float("inf")).max().item())
                transition.append({"index": index, "equal": equal, "max_abs_delta": delta})
        state = compare(torch_env, hip_env)
        per_tick.append({"tick": tick, "transition": transition, "state": state})
        torch_env._last_observation = out_t[0]
        hip_env._last_observation = out_h[0]
    elapsed = time.perf_counter() - started
    first_state_diff = next((item for item in per_tick if not item["state"]["equal"]), None)
    first_transition_diff = next((item for item in per_tick
                                  if any(not x["equal"] for x in item["transition"])), None)
    report = {
        "fixture": {"envs": args.envs, "physics_ticks": args.ticks, "seed": args.seed,
                    "task": "counter_defense", "opponent": "guard",
                    "action_mode": "strategic", "strategic_action_each_tick": 0,
                    "device": str(device), "initial_state_equal": init["equal"]},
        "build": {"loader": "frc_defense.tensor_adstar._hip_path_reference_extension()",
                  "isolated_extension_cuda_flags": cuda_flags,
                  "isolated_extension_extra_cflags": ["-O3"],
                  "source": str(Path(tensor_adstar.__file__).resolve())},
        "result": {"exact_all_ticks": all(t["state"]["equal"] for t in per_tick),
                   "first_state_difference": first_state_diff,
                   "first_transition_difference": first_transition_diff,
                   "elapsed_seconds": elapsed},
        "per_tick": per_tick,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), **report["fixture"], **report["build"],
                      **report["result"]}, indent=2))


if __name__ == "__main__":
    main()
