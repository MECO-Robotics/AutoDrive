#!/usr/bin/env python3
"""Run a reproducible batch of full 3v3 matches for offense experiments.

Each vector-environment slot is a distinct randomized match generated from the
same batch seed. Run candidate worktrees with the same seed, slot count, and
simulator settings to pair the resulting scenarios.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import torch

from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def _revision() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def evaluate(args: argparse.Namespace) -> dict:
    modes = ("deterministic",) * 6
    env = TensorThreeVsThreeEnv(
        num_envs=args.envs,
        device=args.device,
        seed=args.seed,
        control_modes=modes,
        robot_roles=("offense",) * 3 + ("defense",) * 3,
        horizon=args.ticks,
        dt=0.02,
        randomize=True,
        fuel_count=504,
        fused_sensor_rng=True,
        fuel_physics=args.fuel_physics,
        replan_interval=args.replan_interval,
        perception_interval=args.perception_interval,
    )
    env.reset(seed=args.seed)
    actions = torch.full((args.envs, 6), 7, device=args.device, dtype=torch.long)

    started = time.perf_counter()
    for _ in range(args.ticks):
        env.step(actions, capture_observation=False, capture_info=False)
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started

    scores = env.fuel_score_count.detach().cpu().tolist()
    acquisitions = env.fuel_acquisition_count.detach().cpu().tolist()
    differentials = [int(row[0]) - int(row[1]) for row in scores]
    return {
        "schema_version": 1,
        "label": args.label,
        "revision": _revision(),
        "scenario": {
            "batch_seed": args.seed,
            "world_slots": list(range(args.envs)),
            "scenario_ids": [f"{args.seed}:{slot}" for slot in range(args.envs)],
            "seed_semantics": "one seeded vector batch; slots are distinct worlds, not independently seeded resets",
        },
        "settings": {
            "ticks": args.ticks,
            "seconds_per_match": args.ticks * 0.02,
            "device": args.device,
            "controllers": list(modes),
            "robot_roles": ["offense"] * 3 + ["defense"] * 3,
            "fuel_physics": args.fuel_physics,
            "replan_interval": args.replan_interval,
            "perception_interval": args.perception_interval,
        },
        "elapsed_seconds": elapsed,
        "matches_per_second": args.envs / elapsed,
        "per_match": [
            {
                "scenario_id": f"{args.seed}:{slot}",
                "score_by_team": score,
                "score_differential_team0_minus_team1": differentials[slot],
                "fuel_acquisitions_by_robot": acquisition,
            }
            for slot, (score, acquisition) in enumerate(zip(scores, acquisitions))
        ],
        "summary": {
            "mean_team0_score": sum(row[0] for row in scores) / args.envs,
            "mean_team1_score": sum(row[1] for row in scores) / args.envs,
            "mean_score_differential": sum(differentials) / args.envs,
            "total_team0_score": sum(row[0] for row in scores),
            "total_team1_score": sum(row[1] for row in scores),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="short candidate or worktree name")
    parser.add_argument("--seed", type=int, default=64821, help="batch seed; keep fixed across candidates")
    parser.add_argument("--envs", type=int, default=32, help="randomized match worlds in one batch")
    parser.add_argument("--ticks", type=int, default=8000, help="physics ticks per full match")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fuel-physics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replan-interval", type=int, default=20)
    parser.add_argument("--perception-interval", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.envs < 1 or args.ticks < 1:
        parser.error("envs and ticks must be positive")
    if args.replan_interval < 1 or args.perception_interval < 1:
        parser.error("planner and perception intervals must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error("label may contain only letters, numbers, dot, underscore, and hyphen")

    result = evaluate(args)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"wrote {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
