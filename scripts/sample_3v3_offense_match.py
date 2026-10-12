#!/usr/bin/env python3
"""Run one full randomized 3v3 match and save a compact sampled trajectory.

This is an explanatory profiling run, not the end-state batch evaluator. Keep
the vector batch size equal to the paired batch evaluator and select a slot;
changing num_envs can change seeded random draws. Ten-tick sampling captures
0.2 s intervals without synchronizing the device every physics tick.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def provenance() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        repo_root = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True,
            stderr=subprocess.DEVNULL).strip()).resolve()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                             stderr=subprocess.DEVNULL).strip())
        module = str(Path(__import__("frc_defense.tensor_3v3", fromlist=["__file__"]).__file__).resolve())
        module_matches_checkout = Path(module).is_relative_to(repo_root)
        return {"commit": commit, "dirty": dirty, "module": module,
                "module_matches_checkout": module_matches_checkout}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "module": None}


def run(args: argparse.Namespace) -> dict:
    revision = provenance()
    if revision.get("module_matches_checkout") is False:
        raise RuntimeError(
            "simulator module is outside the current Git checkout; set PYTHONPATH "
            "to the intended worktree before sampling")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    env = TensorThreeVsThreeEnv(
        num_envs=args.envs, device=args.device, seed=args.seed,
        control_modes=("deterministic",) * 6,
        robot_roles=("offense",) * 3 + ("defense",) * 3,
        horizon=args.ticks, dt=0.02, randomize=True, fuel_count=504,
        fused_sensor_rng=True, fuel_physics=True, replan_interval=20,
        perception_interval=2, teammate_coordination_enabled=False,
    )
    env.reset(seed=args.seed)
    actions = torch.full((args.envs, 6), 7, device=args.device, dtype=torch.long)
    frame_snapshots = []
    start = time.perf_counter()
    for tick in range(args.ticks):
        env.step(actions, capture_observation=False, capture_info=False)
        if (tick + 1) % args.sample_every_ticks == 0 or tick + 1 == args.ticks:
            # Keep captures on device and transfer them once after the rollout.
            # Per-frame .cpu() synchronization can dominate GPU simulation time.
            possession = torch.stack([
                (env.piece_owner[args.world_slot] == robot).sum()
                for robot in range(6)
            ]).to(env.sim.pose.dtype)
            frame_snapshots.append(torch.cat((
                env.sim.pose[args.world_slot].reshape(-1),
                env.sim.velocity[args.world_slot, :, :2].reshape(-1),
                env.last_actions[args.world_slot].to(env.sim.pose.dtype),
                env._last_audit_targets[args.world_slot].reshape(-1),
                env._last_audit_fuel_targets[args.world_slot].reshape(-1),
                env.planner.last_goal.reshape(args.envs, 6, 2)[args.world_slot].reshape(-1),
                env._target_collecting[args.world_slot].to(env.sim.pose.dtype),
                env.fuel_acquisition_count[args.world_slot].to(env.sim.pose.dtype),
                env.fuel_score_count[args.world_slot].to(env.sim.pose.dtype),
                env.fuel_scored_event[args.world_slot].to(env.sim.pose.dtype),
                env.fuel_acquired_event[args.world_slot].to(env.sim.pose.dtype),
                possession,
                env.hub_active[args.world_slot].to(env.sim.pose.dtype),
                env.match_elapsed[args.world_slot:args.world_slot+1],
            )).detach())
        if (tick + 1) % 500 == 0 or tick + 1 == args.ticks:
            elapsed = time.perf_counter() - start
            print(f"sample progress: {tick + 1}/{args.ticks} ticks, "
                  f"{elapsed:.1f}s", file=sys.stderr, flush=True)

    rows = torch.stack(frame_snapshots).cpu().tolist()
    frames = []
    for sample_index, snapshot in enumerate(rows):
        tick = min((sample_index + 1) * args.sample_every_ticks, args.ticks)
        i = 0

        def take(n: int):
            nonlocal i
            values = snapshot[i:i+n]
            i += n
            return values

        pose = [take(3) for _ in range(6)]
        velocity = [take(2) for _ in range(6)]
        robot_actions = [int(x) for x in take(6)]
        targets = [take(2) for _ in range(6)]
        fuel_targets = [take(2) for _ in range(6)]
        route_goals = [take(2) for _ in range(6)]
        collecting = [bool(x) for x in take(6)]
        acquisitions = [int(x) for x in take(6)]
        scores = [int(x) for x in take(2)]
        scored_events = [int(x) for x in take(6)]
        acquired_events = [int(x) for x in take(6)]
        possession = [int(x) for x in take(6)]
        hubs = [bool(x) for x in take(2)]
        elapsed = take(1)[0]
        if i != len(snapshot):
            raise RuntimeError("sample layout mismatch")
        frames.append({
            "tick": tick, "time_s": elapsed, "robots": pose,
            "velocity_mps": velocity, "actions": robot_actions,
            "targets": targets, "fuel_targets": fuel_targets,
            "route_goals": route_goals, "collecting": collecting,
            "acquisitions_by_robot": acquisitions,
            "score_by_team": scores,
            "scored_event_by_robot": scored_events,
            "acquired_event_by_robot": acquired_events,
            "possession_by_robot": possession,
            "hub_active_by_team": hubs,
        })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "schema_version": 1, "label": args.label,
        "revision": revision, "seed": args.seed,
        "scenario_id": f"{args.seed}:{args.world_slot}",
        "scenario_world_slot": args.world_slot,
        "field": {"length_m": env.field_length, "width_m": float(env.sim.field_width),
                  "alliance_zone_depth_m": env.alliance_zone_depth,
                  "hub_centers": env.hub_centers.cpu().tolist()},
        "settings": {"ticks": args.ticks, "dt": 0.02,
                     "sample_every_ticks": args.sample_every_ticks,
                     "envs": args.envs, "world_slot": args.world_slot,
                     "device": args.device, "teammate_coordination": False,
                     "replan_interval": 20, "perception_interval": 2,
                     "fuel_physics": True},
        "elapsed_seconds": time.perf_counter() - start,
        "frames": frames,
        "final": {"score_by_team": env.fuel_score_count[args.world_slot].cpu().tolist(),
                  "acquisitions_by_robot": env.fuel_acquisition_count[args.world_slot].cpu().tolist()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=4,
                        help="match the vector batch size used by the paired evaluator")
    parser.add_argument("--world-slot", type=int, default=0,
                        help="record this world slot from the exact seeded vector batch")
    parser.add_argument("--ticks", type=int, default=8000)
    parser.add_argument("--sample-every-ticks", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.ticks, args.sample_every_ticks, args.envs) < 1:
        parser.error("ticks, sample interval, and env count must be positive")
    if not 0 <= args.world_slot < args.envs:
        parser.error("world slot must be in range for env count")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "revision": result["revision"],
                      "scenario_id": result["scenario_id"],
                      "samples": len(result["frames"]),
                      "final": result["final"]}, indent=2))


if __name__ == "__main__":
    main()
