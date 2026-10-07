#!/usr/bin/env python3
"""Export per-frame and collect-only metrics from a focused replay."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ALLIANCE_ZONE_DEPTH = 4.028


def analyze(source: Path, output: Path) -> dict:
    replay = json.loads(source.read_text())
    frames = replay["frames"]
    records = []
    for index, frame in enumerate(frames):
        robot = frame["robots"][0]
        action = int(frame["robot_actions"][0])
        pieces = frame["fuel_pieces"]
        held = sum(piece[2] == 0 for piece in pieces)
        red_zone = sum(piece[2] == -1 and piece[0] < ALLIANCE_ZONE_DEPTH
                       for piece in pieces)
        phase = ("collect" if action == 0 else
                 "ferry" if action == 6 and held > 0 else
                 "ferry-wait" if action == 7 and held > 0 else
                 "search" if action == 6 else
                 "wait" if action == 7 else
                 "score" if action == 4 else "other")
        time_s = float(frame["match_elapsed"])
        previous = frames[index - 1] if index else None
        interval_s = (time_s - float(previous["match_elapsed"]) if previous else 0.)
        same_collect = bool(previous and action == 0 and
                            int(previous["robot_actions"][0]) == 0)
        acquisition_delta = (int(frame["fuel_acquisition_count"][0]) -
            int(previous["fuel_acquisition_count"][0]) if previous else 0)
        distance = (math.dist(robot[:2], previous["robots"][0][:2])
                    if previous else 0.)
        heading_delta = (math.atan2(
            math.sin(robot[2] - previous["robots"][0][2]),
            math.cos(robot[2] - previous["robots"][0][2])) if previous else 0.)
        records.append({
            "time_s": time_s,
            "action": action,
            "phase": phase,
            "ferrying": phase in ("ferry", "ferry-wait"),
            "collecting": bool(frame["robot_collecting"][0]),
            "x_m": robot[0], "y_m": robot[1], "heading_rad": robot[2],
            "speed_mps": distance / interval_s if interval_s > 0 else 0.,
            "turn_rate_rad_s": heading_delta / interval_s if interval_s > 0 else 0.,
            "acquired_total": int(frame["fuel_acquisition_count"][0]),
            "acquired_delta": acquisition_delta,
            "collect_only_delta": acquisition_delta if same_collect else "",
            "collect_only_rate_s": acquisition_delta / interval_s
                if same_collect and interval_s > 0 else "",
            "hopper_held": held,
            "red_zone_loose": red_zone,
            "held_plus_red_zone": held + red_zone,
            "cluster_tracks": int(frame["robot_cluster_counts"][0]),
            "target_x_m": frame["robot_targets"][0][0],
            "target_y_m": frame["robot_targets"][0][1],
            "route_goal_x_m": frame["robot_route_goals"][0][0],
            "route_goal_y_m": frame["robot_route_goals"][0][1],
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    episodes = []
    index = 0
    while index < len(records):
        if records[index]["action"] != 0:
            index += 1
            continue
        start = index
        while index + 1 < len(records) and records[index + 1]["action"] == 0:
            index += 1
        end = index
        elapsed = records[end]["time_s"] - records[start]["time_s"]
        acquisitions = records[end]["acquired_total"] - records[start]["acquired_total"]
        distance = sum(math.dist(
            (records[i - 1]["x_m"], records[i - 1]["y_m"]),
            (records[i]["x_m"], records[i]["y_m"])) for i in range(start + 1, end + 1))
        if end > start:
            episodes.append({
                "start_s": records[start]["time_s"],
                "end_s": records[end]["time_s"],
                "duration_s": elapsed,
                "acquisitions": acquisitions,
                "acquisitions_per_collect_s": acquisitions / max(elapsed, 1e-9),
                "path_m": distance,
                "mean_speed_mps": distance / max(elapsed, 1e-9),
            })
        index += 1

    collect_bins = []
    # The focused replay may save one frame just beyond its nominal 30s window;
    # don't let that extra frame create a spurious final one-second bin.
    for second in range(int(records[-1]["time_s"]) if records else 0):
        intervals = [(i, record) for i, record in enumerate(records)
                     if second <= record["time_s"] < second + 1 and
                     record["collect_only_delta"] != ""]
        if not intervals:
            continue
        duration = sum(float(record["time_s"]) - float(records[i - 1]["time_s"])
                       for i, record in intervals if i > 0)
        if duration <= 0:
            continue
        selected = [record for _, record in intervals]
        collect_bins.append({
            "start_s": second,
            "collect_seconds": duration,
            "acquisitions": sum(int(record["collect_only_delta"])
                                 for record in selected),
            "acquisitions_per_collect_s": sum(int(record["collect_only_delta"])
                for record in selected) / duration,
            "mean_speed_mps": sum(float(record["speed_mps"])
                for record in selected) / len(selected),
            "mean_abs_turn_rate_rad_s": sum(abs(float(record["turn_rate_rad_s"]))
                for record in selected) / len(selected),
        })

    summary = {
        "simulation_id": replay.get("simulation_id"),
        "seed": replay.get("seed"),
        "frame_count": len(records),
        "duration_s": records[-1]["time_s"] if records else 0.,
        "final_acquired": records[-1]["acquired_total"] if records else 0,
        "final_held_plus_red_zone": records[-1]["held_plus_red_zone"] if records else 0,
        "collect_episodes": episodes,
        "collect_only_one_second_bins": collect_bins,
        "csv": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="focused replay JSON")
    parser.add_argument("--output", type=Path,
                        help="CSV output (default: beside replay)")
    args = parser.parse_args()
    output = args.output or args.replay.with_name(args.replay.stem + ".timeseries.csv")
    print(json.dumps(analyze(args.replay, output), indent=2))


if __name__ == "__main__":
    main()
