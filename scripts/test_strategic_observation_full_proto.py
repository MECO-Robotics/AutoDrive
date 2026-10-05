#!/usr/bin/env python3
"""Exact parity and focused timing for the isolated full observation HIP proto."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense import tensor_sim
from frc_defense import tensor_strategic_observation_proto as packer
from frc_defense import tensor_strategic_observation_full_proto as full


def _env(task, seed, device, n, normalize):
    return tensor_sim.TensorDefenseEnv(num_envs=n, task=task, device=device,
        seed=seed, action_mode="strategic", randomize=False,
        reuse_strategic_own_candidates=False, normalize_observations=normalize)


def _seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fixture(env, focal, empty, route):
    env._track_mask[:, focal].zero_()
    if not empty:
        env._track_mask[:, focal, :4] = True
        offsets = torch.tensor(((.19, .07), (.43, .16), (.71, .31), (1.03, .52)),
            device=env.device, dtype=env.sim.pose.dtype)
        env._track_pos[:, focal, :4] = env.sim.pose[:, focal, None, :2] + offsets
        env._track_vel[:, focal, :4] = torch.tensor((.11, -.07), device=env.device)
    planner = (env._adstar_tactical_planner if focal == 0 else
        env._adstar_planners if env.task == "defense" else env._adstar_defender_planners)
    if planner is not None:
        planner.last_path.zero_()
        if route:
            planner.last_path[:, 0] = env.sim.pose[:, focal, :2]
            planner.last_path[:, 1, 0] = env.sim.pose[:, focal, 0] + .5
            planner.last_path[:, 1, 1] = env.sim.pose[:, focal, 1] + .25
            planner.last_path[:, 2, 0] = env.sim.pose[:, focal, 0] + 1.
            planner.last_path[:, 2, 1] = env.sim.pose[:, focal, 1] + .5
            planner.last_lengths.fill_(3)
        else:
            planner.last_lengths.zero_()


def _case(task, focal, normalize, empty, route, active_fraction, seed, device, n):
    _seed(seed)
    env = _env(task, seed, device, n, normalize)
    _fixture(env, focal, empty, route)
    active = (torch.arange(n, device=device) < max(1, int(n * active_fraction)))
    # Establish the documented reference with the current Torch observation path.
    packer.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = False
    expected, expected_candidates, expected_mask = env._strategic_observation(
        focal, active_mask=active, return_candidate_data=True)
    packer.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = True
    got_candidates = env._fuel_candidates(focal)
    got = full.observe(env, focal, normalize=normalize, candidates=got_candidates,
                       active_mask=active)
    if not torch.equal(expected, got):
        delta = (expected - got).abs()
        max_error = float(delta.max().item())
        mismatch = int((expected != got).sum().item())
        at = torch.nonzero(expected != got, as_tuple=False)[0].tolist()
        raise AssertionError(f"mismatch {task=} {focal=} {normalize=} {empty=} {route=}: "
            f"{mismatch} values, max_abs={max_error}, first={at}, "
            f"expected={expected[tuple(at)].item()}, got={got[tuple(at)].item()}")
    for left, right, label in ((expected_candidates[1], got_candidates[1], "candidate validity"),
                               (expected_mask, env.strategic_action_mask(focal, _candidate_data=got_candidates), "action mask")):
        if not torch.equal(left, right):
            raise AssertionError(f"{label} mismatch in {task=} {focal=}")
    # Active-mask selection is accepted by reference only to protect pending state;
    # verify inactive output rows still describe their unchanged state exactly.
    inactive = ~active
    if inactive.any() and not torch.equal(expected[inactive], got[inactive]):
        raise AssertionError("inactive observation rows differ")
    return {"task": task, "focal": focal, "normalize": normalize,
            "empty_candidates": empty, "route_available": route,
            "active_fraction": active_fraction, "exact": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("requires HIP")
    if full._extension() is None:
        raise SystemExit("full strategic observation extension failed to build")
    cases = []
    for task, focal, norm, empty, route, active_frac in itertools.product(
            ("counter_defense", "defense"), (0, 1), (False, True),
            (False, True), (False, True), (.5, 1.)):
        cases.append(_case(task, focal, norm, empty, route, active_frac,
                           73021, args.device, min(args.envs, 64)))

    _seed(831)
    env = _env("counter_defense", 831, args.device, args.envs, True)
    _fixture(env, 0, False, True)
    # Warm up both current production subpath and full assembler.
    packer.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = False
    for _ in range(10):
        ref = env._strategic_observation(0)
    packer.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = True
    for _ in range(10):
        fused = full.observe(env, 0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        ref = env._strategic_observation(0)
    torch.cuda.synchronize()
    ref_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        fused = full.observe(env, 0)
    torch.cuda.synchronize()
    fused_s = time.perf_counter() - t0
    report = {"cases": cases, "case_count": len(cases), "all_exact": True,
        "envs": args.envs, "repeats": args.repeats,
        "reference_ms_per_call": ref_s * 1000 / args.repeats,
        "full_assembler_ms_per_call": fused_s * 1000 / args.repeats,
        "speedup": ref_s / fused_s,
        "scope": "observation call including existing Torch candidate ranking, opponent/obstacle prep, and fused full-vector assembly"}
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
