#!/usr/bin/env python3
"""Summarize robot traversal, collection, carrying, and scoring from a trace."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ACTION_NAMES = {0: "collect", 1: "collect", 2: "collect", 3: "collect",
                4: "score_or_return", 5: "defend", 6: "search_or_ferry",
                7: "wait"}


def analyze(path: Path) -> dict:
    record = json.loads(path.read_text())
    frames = record["frames"]
    field = record["field"]
    zone = float(field["alliance_zone_depth_m"])
    length = float(field["length_m"])
    per_robot = []
    for robot in range(6):
        stats = {"robot": robot, "team": 0 if robot < 3 else 1,
                 "samples": len(frames), "distance_m": 0., "moving_seconds": 0.,
                 "targetward_progress_m": 0., "command_target_changes": 0,
                 "fuel_target_assignments": 0, "fuel_target_switches": 0,
                 "fuel_target_losses": 0, "fuel_target_active_seconds": 0.,
                 "fuel_target_dwell_seconds": [],
                 "low_speed_far_seconds": 0., "collecting_seconds": 0.,
                 "cluster_sweep_seconds": 0.,
                 "loaded_seconds": 0., "loaded_hub_inactive_seconds": 0.,
                 "loaded_in_alliance_zone_seconds": 0.,
                 "loaded_active_score_zone_seconds": 0.,
                 "alliance_zone_seconds": 0., "score_zone_active_seconds": 0.,
                 "acquisitions": 0, "sampled_scored_event_samples": 0,
                 "loaded_trips": 0, "loaded_return_to_zone_seconds": [],
                 "action_seconds": {}}
        last = None
        load_started_s = None
        reached_zone_this_load = False
        fuel_target_started_s = None
        for frame in frames:
            dt = (float(frame["time_s"]) - float(last["time_s"])) if last else 0.
            pose = frame["robots"][robot]
            speed = math.hypot(*frame["velocity_mps"][robot])
            target = frame["targets"][robot]
            target_dist = math.dist(pose[:2], target)
            fuel_target_index = int(frame.get("fuel_target_indices", [-1] * 6)[robot])
            fuel_target_step = int(frame.get("fuel_target_selected_steps", [-1] * 6)[robot])
            possession = int(frame["possession_by_robot"][robot])
            action = int(frame["actions"][robot])
            action_name = ACTION_NAMES.get(action, "other")
            if last is None:
                if fuel_target_index >= 0:
                    stats["fuel_target_assignments"] += 1
                    fuel_target_started_s = fuel_target_step * .02
            else:
                previous_fuel_target = int(last.get("fuel_target_indices", [-1] * 6)[robot])
                if fuel_target_index != previous_fuel_target:
                    if previous_fuel_target >= 0:
                        stats["fuel_target_dwell_seconds"].append(max(
                            0., float(frame["time_s"]) -
                            float(fuel_target_started_s if fuel_target_started_s is not None
                                  else last["time_s"])))
                    if fuel_target_index >= 0:
                        if previous_fuel_target >= 0:
                            stats["fuel_target_switches"] += 1
                        else:
                            stats["fuel_target_assignments"] += 1
                        fuel_target_started_s = fuel_target_step * .02
                    else:
                        stats["fuel_target_losses"] += 1
                        fuel_target_started_s = None
            if dt > 0:
                if last:
                    prev_pose = last["robots"][robot]
                    stats["distance_m"] += math.dist(pose[:2], prev_pose[:2])
                    previous_target = last["targets"][robot]
                    if math.dist(target, previous_target) <= .25:
                        stats["targetward_progress_m"] += (
                            math.dist(prev_pose[:2], target) - target_dist)
                    else:
                        stats["command_target_changes"] += 1
                if speed >= .3:
                    stats["moving_seconds"] += dt
                if speed < .3 and target_dist > 1.0:
                    stats["low_speed_far_seconds"] += dt
                # "Collect" is an issued planner mode; it does not guarantee
                # the intake is actually engaged. Use the simulator's sampled
                # collecting state for throughput denominators.
                if frame["collecting"][robot]:
                    stats["collecting_seconds"] += dt
                if frame.get("target_collecting", [False] * 6)[robot]:
                    stats["cluster_sweep_seconds"] += dt
                if fuel_target_index >= 0:
                    stats["fuel_target_active_seconds"] += dt
                if possession > 0:
                    if last is None or int(last["possession_by_robot"][robot]) == 0:
                        stats["loaded_trips"] += 1
                        load_started_s = float(frame["time_s"])
                        reached_zone_this_load = False
                    stats["loaded_seconds"] += dt
                    if not frame["hub_active_by_team"][stats["team"]]:
                        stats["loaded_hub_inactive_seconds"] += dt
                    if ((stats["team"] == 0 and pose[0] <= zone) or
                            (stats["team"] == 1 and pose[0] >= length - zone)):
                        stats["loaded_in_alliance_zone_seconds"] += dt
                in_zone = ((stats["team"] == 0 and pose[0] <= zone) or
                           (stats["team"] == 1 and pose[0] >= length - zone))
                if (possession > 0 and in_zone and not reached_zone_this_load
                        and load_started_s is not None):
                    stats["loaded_return_to_zone_seconds"].append(
                        float(frame["time_s"]) - load_started_s)
                    reached_zone_this_load = True
                if possession == 0:
                    load_started_s = None
                    reached_zone_this_load = False
                if in_zone:
                    stats["alliance_zone_seconds"] += dt
                    if frame["hub_active_by_team"][stats["team"]]:
                        stats["score_zone_active_seconds"] += dt
                        if possession > 0:
                            stats["loaded_active_score_zone_seconds"] += dt
                stats["action_seconds"][action_name] = (
                    stats["action_seconds"].get(action_name, 0.) + dt)
                acquired = int(frame["acquisitions_by_robot"][robot]) - int(
                    last["acquisitions_by_robot"][robot])
                stats["acquisitions"] += max(0, acquired)
                # Simulator events are per-tick pulses, sampled every N ticks;
                # this count is only a sparse indicator, not an event total.
                stats["sampled_scored_event_samples"] += int(
                    frame["scored_event_by_robot"][robot])
            last = frame
        if last is not None and int(last.get("fuel_target_indices", [-1] * 6)[robot]) >= 0:
            stats["fuel_target_dwell_seconds"].append(max(
                0., float(last["time_s"]) -
                float(fuel_target_started_s if fuel_target_started_s is not None
                      else last["time_s"])))
        elapsed = max(0., float(frames[-1]["time_s"]) - float(frames[0]["time_s"])) if frames else 0.
        stats["elapsed_s"] = elapsed
        stats["mean_speed_mps"] = stats["distance_m"] / elapsed if elapsed else 0.
        dwell = stats["fuel_target_dwell_seconds"]
        stats["mean_fuel_target_dwell_s"] = sum(dwell) / len(dwell) if dwell else None
        stats["median_fuel_target_dwell_s"] = (
            statistics.median(dwell) if dwell else None)
        stats["mean_loaded_return_to_zone_s"] = (
            sum(stats["loaded_return_to_zone_seconds"]) /
            len(stats["loaded_return_to_zone_seconds"])
            if stats["loaded_return_to_zone_seconds"] else None)
        stats["acquisitions_per_collect_min"] = (
            60. * stats["acquisitions"] / stats["collecting_seconds"]
            if stats["collecting_seconds"] else 0.)
        per_robot.append(stats)

    team_summary = {}
    for team in (0, 1):
        score_active = score_inactive = 0
        for previous, current in zip(frames, frames[1:]):
            delta = max(0, int(current["score_by_team"][team]) -
                        int(previous["score_by_team"][team]))
            if current["hub_active_by_team"][team]:
                score_active += delta
            else:
                score_inactive += delta
        team_summary[str(team)] = {
            "distance_m": sum(r["distance_m"] for r in per_robot if r["team"] == team),
            "acquisitions": sum(r["acquisitions"] for r in per_robot if r["team"] == team),
            "scored_fuel": int(record["final"]["score_by_team"][team]),
            "score_deltas_during_active_hub": score_active,
            "score_deltas_during_inactive_hub": score_inactive,
            "low_speed_far_seconds": sum(r["low_speed_far_seconds"] for r in per_robot if r["team"] == team),
            "loaded_seconds": sum(r["loaded_seconds"] for r in per_robot if r["team"] == team),
            "loaded_inactive_seconds": sum(r["loaded_hub_inactive_seconds"] for r in per_robot if r["team"] == team),
            "loaded_active_score_zone_seconds": sum(r["loaded_active_score_zone_seconds"] for r in per_robot if r["team"] == team),
            "score_zone_active_seconds": sum(r["score_zone_active_seconds"] for r in per_robot if r["team"] == team),
        }

    return {
        "source": str(path), "label": record["label"],
        "revision": record["revision"], "scenario_id": record["scenario_id"],
        "settings": record["settings"], "profile_elapsed_s": record["elapsed_seconds"],
        "final": record["final"], "per_robot": per_robot,
        "team_summary": team_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = analyze(args.trace)
    encoded = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
