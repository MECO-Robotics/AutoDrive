#!/usr/bin/env python3
"""Read per-frame robot audit signals from a randomized playback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


def load_record(args):
    if args.playback:
        return json.loads(args.playback.read_text())
    if args.dashboard_url:
        simulation_id = args.simulation_id
        if not simulation_id:
            with urlopen(args.dashboard_url.rstrip("/") +
                         "/api/zone-playback-active", timeout=20) as response:
                active = json.load(response).get("job")
            if not active:
                raise SystemExit("Dashboard has no saved randomized playback")
            simulation_id = active["simulation_id"]
        url = (args.dashboard_url.rstrip("/") +
               "/api/zone-playback-result?simulation_id=" + simulation_id)
        with urlopen(url, timeout=20) as response:
            return json.load(response)
    progress_dir = args.run_dir / ".scenario-progress"
    if args.simulation_id:
        path = progress_dir / f"{args.simulation_id}.result.json"
    else:
        candidates = list(progress_dir.glob("*.result.json"))
        if not candidates:
            raise SystemExit(f"No saved scenario results under {progress_dir}")
        path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--playback", type=Path,
                        help="saved playback/result JSON file")
    source.add_argument("--dashboard-url",
                        help="dashboard base URL, e.g. http://127.0.0.1:8765")
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/tensor-ppo"),
                        help="dashboard run directory for latest or named saved result")
    parser.add_argument("--simulation-id", help="scenario ID from /api/zone-playback-active")
    parser.add_argument("--robot", type=int, default=0, choices=range(6))
    parser.add_argument("--all-frames", action="store_true",
                        help="include every captured frame instead of only state/target changes")
    args = parser.parse_args()
    record = load_record(args)
    scenarios = record.get("scenarios") or []
    frames = scenarios[0].get("frames", []) if scenarios else record.get("frames", [])
    if not frames:
        raise SystemExit("Playback contains no frames")

    robot = args.robot
    rows = []
    previous = None
    for frame in frames:
        pose = frame.get("robots", [])[robot]
        target = (frame.get("robot_targets") or [None] * 6)[robot]
        fuel = (frame.get("robot_fuel_targets") or [None] * 6)[robot]
        route_goal = (frame.get("robot_route_goals") or [None] * 6)[robot]
        action = (frame.get("robot_actions") or [None] * 6)[robot]
        collecting = (frame.get("robot_collecting") or [False] * 6)[robot]
        cluster = (frame.get("robot_cluster_counts") or [None] * 6)[robot]
        row = {"time": frame.get("match_elapsed"), "pose": pose,
               "action": action, "collecting": collecting,
               "cluster_tracks": cluster, "fuel_target": fuel,
               "control_target": target, "adstar_goal": route_goal}
        signature = (action, collecting, tuple(fuel or ()), tuple(target or ()),
                     tuple(route_goal or ()))
        if args.all_frames or signature != previous:
            rows.append(row)
        previous = signature
    print(json.dumps({"seed": record.get("seed"), "simulation_id": record.get("simulation_id"),
                      "robot": robot, "frames_captured": len(frames),
                      "audit_rows": rows}, separators=(",", ":")))


if __name__ == "__main__":
    main()
