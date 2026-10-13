#!/usr/bin/env python3
"""Analyze frozen per-world batches and same-rollout trace pairs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from analyze_3v3_offense_trace import analyze as analyze_trace


SEEDS = (339752, 199448, 247246, 973688, 144124, 762406)
T95_DF5 = 2.5705818366147395


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def mean_ci(values: list[float]) -> dict:
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"mean": mean, "low_95": None, "high_95": None}
    half_width = T95_DF5 * statistics.stdev(values) / len(values) ** .5
    return {"mean": mean, "low_95": mean - half_width,
            "high_95": mean + half_width}


def team0_acquisitions(match: dict) -> int:
    return sum(int(value) for value in match["fuel_acquisitions_by_robot"][:3])


def team0_points(match: dict) -> int:
    return int(match["score_by_team"][0])


def trace_metrics(path: Path) -> dict:
    summary = analyze_trace(path)
    offense = [row for row in summary["per_robot"] if row["team"] == 0]
    total_intake_s = sum(row["collecting_seconds"] for row in offense)
    dwell = [value for row in offense
             for value in row["fuel_target_dwell_seconds"]]
    team = summary["team_summary"]["0"]
    return {
        "scenario_id": summary["scenario_id"],
        "score": int(summary["final"]["score_by_team"][0]),
        "acquisitions": sum(row["acquisitions"] for row in offense),
        "distance_m": sum(row["distance_m"] for row in offense),
        "low_speed_far_s": sum(row["low_speed_far_seconds"] for row in offense),
        "targetward_progress_m": sum(row["targetward_progress_m"] for row in offense),
        "command_target_changes": sum(row["command_target_changes"] for row in offense),
        "fuel_target_assignments": sum(row["fuel_target_assignments"] for row in offense),
        "fuel_target_switches": sum(row["fuel_target_switches"] for row in offense),
        "fuel_target_losses": sum(row["fuel_target_losses"] for row in offense),
        "fuel_target_active_s": sum(row["fuel_target_active_seconds"] for row in offense),
        "mean_fuel_target_dwell_s": statistics.mean(dwell) if dwell else None,
        "intake_enabled_s": total_intake_s,
        "cluster_sweep_s": sum(row["cluster_sweep_seconds"] for row in offense),
        "acquisitions_per_intake_min": 60 * sum(
            row["acquisitions"] for row in offense) / total_intake_s
            if total_intake_s else 0.,
        "loaded_s": sum(row["loaded_seconds"] for row in offense),
        "loaded_hub_inactive_s": sum(
            row["loaded_hub_inactive_seconds"] for row in offense),
        "loaded_active_score_zone_s": sum(
            row["loaded_active_score_zone_seconds"] for row in offense),
        "mean_loaded_return_to_zone_s": statistics.mean(
            value for row in offense
            for value in row["loaded_return_to_zone_seconds"])
            if any(row["loaded_return_to_zone_seconds"] for row in offense)
            else None,
        "score_during_active_hub": team["score_deltas_during_active_hub"],
        "score_during_inactive_hub": team["score_deltas_during_inactive_hub"],
        "active_score_zone_s": team["score_zone_active_seconds"],
    }


def analyze(root: Path, seeds: tuple[int, ...]) -> dict:
    paired_rows = []
    seed_rows = []
    trace_rows = []
    seed_score_deltas = []
    seed_fuel_deltas = []
    baseline_points = []
    candidate_points = []
    baseline_fuel = []
    candidate_fuel = []

    for seed in seeds:
        baseline_path = root / "baseline" / f"baseline-{seed}.json"
        candidate_path = root / "candidate" / f"candidate-{seed}.json"
        baseline = load(baseline_path)
        candidate = load(candidate_path)
        if baseline["scenario"]["batch_seed"] != seed or candidate["scenario"]["batch_seed"] != seed:
            raise ValueError(f"seed metadata mismatch for {seed}")
        if baseline["settings"]["teammate_coordination_enabled"] or candidate["settings"]["teammate_coordination_enabled"]:
            raise ValueError(f"teammate coordination unexpectedly enabled for {seed}")
        if baseline["settings"]["device"] != candidate["settings"]["device"]:
            raise ValueError(f"paired arms used different devices for {seed}")
        b_worlds = {row["scenario_id"]: row for row in baseline["per_match"]}
        c_worlds = {row["scenario_id"]: row for row in candidate["per_match"]}
        if set(b_worlds) != set(c_worlds) or len(b_worlds) != 4:
            raise ValueError(f"scenario IDs do not pair as four worlds for {seed}")

        seed_b_score = seed_c_score = seed_b_fuel = seed_c_fuel = 0
        for scenario_id in sorted(b_worlds):
            b_world = b_worlds[scenario_id]
            c_world = c_worlds[scenario_id]
            b_points, c_points = team0_points(b_world), team0_points(c_world)
            b_fuel, c_fuel = team0_acquisitions(b_world), team0_acquisitions(c_world)
            row = {
                "seed": seed, "scenario_id": scenario_id,
                "baseline_offense_acquisitions": b_fuel,
                "candidate_offense_acquisitions": c_fuel,
                "delta_offense_acquisitions": c_fuel - b_fuel,
                "baseline_team0_points": b_points,
                "candidate_team0_points": c_points,
                "delta_team0_points": c_points - b_points,
            }
            paired_rows.append(row)
            seed_b_score += b_points
            seed_c_score += c_points
            seed_b_fuel += b_fuel
            seed_c_fuel += c_fuel
            baseline_points.append(b_points)
            candidate_points.append(c_points)
            baseline_fuel.append(b_fuel)
            candidate_fuel.append(c_fuel)

        baseline_trace_path = root / "trace" / f"baseline-{seed}-trace.json"
        candidate_trace_path = root / "trace" / f"candidate-{seed}-trace.json"
        bt = trace_metrics(baseline_trace_path)
        ct = trace_metrics(candidate_trace_path)
        if bt["scenario_id"] != f"{seed}:0" or ct["scenario_id"] != f"{seed}:0":
            raise ValueError(f"trace scenario mismatch for {seed}")
        trace_row = {"seed": seed, "baseline": bt, "candidate": ct}
        for metric in bt:
            if metric in {"scenario_id", "score", "acquisitions"}:
                continue
            bval, cval = bt[metric], ct[metric]
            trace_row[f"delta_{metric}"] = (
                cval - bval if bval is not None and cval is not None else None)
        trace_rows.append(trace_row)

        b_mean_score = seed_b_score / 4
        c_mean_score = seed_c_score / 4
        b_mean_fuel = seed_b_fuel / 4
        c_mean_fuel = seed_c_fuel / 4
        seed_score_deltas.append(c_mean_score - b_mean_score)
        seed_fuel_deltas.append(c_mean_fuel - b_mean_fuel)
        seed_rows.append({
            "seed": seed, "worlds": 4,
            "baseline_mean_team0_points": b_mean_score,
            "candidate_mean_team0_points": c_mean_score,
            "delta_mean_team0_points": c_mean_score - b_mean_score,
            "baseline_mean_offense_acquisitions": b_mean_fuel,
            "candidate_mean_offense_acquisitions": c_mean_fuel,
            "delta_mean_offense_acquisitions": c_mean_fuel - b_mean_fuel,
        })

    points_ci = mean_ci(seed_score_deltas)
    fuel_ci = mean_ci(seed_fuel_deltas)
    return {
        "seeds": list(seeds), "seed_clusters": len(seeds),
        "worlds_per_arm": len(paired_rows),
        "analysis_unit": "six seed-cluster means; worlds are nested within seed",
        "confidence_interval": "two-sided 95% Student-t, df=5",
        "score_delta_per_world_ci": points_ci,
        "fuel_delta_per_world_ci": fuel_ci,
        "arms": {
            "baseline": {
                "mean_team0_points": statistics.mean(baseline_points),
                "median_team0_points": statistics.median(baseline_points),
                "worlds_over_800": sum(value > 800 for value in baseline_points),
                "offense_acquisitions": sum(baseline_fuel),
            },
            "candidate": {
                "mean_team0_points": statistics.mean(candidate_points),
                "median_team0_points": statistics.median(candidate_points),
                "worlds_over_800": sum(value > 800 for value in candidate_points),
                "offense_acquisitions": sum(candidate_fuel),
            },
        },
        "paired_worlds": paired_rows,
        "seed_clusters_detail": seed_rows,
        "trace_slot0_pairs": trace_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "evaluations/expected-yield-cycle-rank-replication-V"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    report = analyze(args.root, tuple(args.seeds))
    json_path = args.root / "analysis" / "paired-analysis.json"
    csv_path = args.root / "analysis" / "paired-worlds.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    with csv_path.open("w", newline="") as output:
        rows = report["paired_worlds"]
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "report": str(json_path),
        "worlds_per_arm": report["worlds_per_arm"],
        "baseline": report["arms"]["baseline"],
        "candidate": report["arms"]["candidate"],
        "fuel_delta_per_world_95_ci": report["fuel_delta_per_world_ci"],
        "score_delta_per_world_95_ci": report["score_delta_per_world_ci"],
    }, indent=2))


if __name__ == "__main__":
    main()
