#!/usr/bin/env python3
"""Compare two offense batch artifacts from identical randomized scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported result schema in {path}")
    return data


def compare(baseline: dict, candidate: dict) -> dict:
    if baseline["scenario"] != candidate["scenario"]:
        raise ValueError("batch seed or world slots differ; paired comparison is invalid")
    baseline_settings = dict(baseline["settings"])
    candidate_settings = dict(candidate["settings"])
    baseline_device = str(baseline_settings.pop("device", "cpu"))
    candidate_device = str(candidate_settings.pop("device", "cpu"))
    # cuda:N / hip:N selects a physical device, not different simulator
    # behavior. Permit paired candidate runs on separate GPUs while requiring
    # the same backend and all other simulation settings.
    baseline_backend = baseline_device.split(":", 1)[0]
    candidate_backend = candidate_device.split(":", 1)[0]
    if (baseline_backend != candidate_backend or
            baseline_settings != candidate_settings):
        raise ValueError("simulator settings differ; paired comparison is invalid")
    baseline_rows = baseline["per_match"]
    candidate_rows = candidate["per_match"]
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("match counts differ")

    rows = []
    for old, new in zip(baseline_rows, candidate_rows):
        if old["scenario_id"] != new["scenario_id"]:
            raise ValueError("scenario order differs; paired comparison is invalid")
        old_diff = old["score_differential_team0_minus_team1"]
        new_diff = new["score_differential_team0_minus_team1"]
        old_fuel = sum(old["fuel_acquisitions_by_robot"][:3])
        new_fuel = sum(new["fuel_acquisitions_by_robot"][:3])
        rows.append({
            "scenario_id": old["scenario_id"],
            "baseline_score_by_team": old["score_by_team"],
            "candidate_score_by_team": new["score_by_team"],
            "team0_score_gain": new["score_by_team"][0] - old["score_by_team"][0],
            "score_differential_gain": new_diff - old_diff,
            "baseline_team0_fuel_acquisitions": old_fuel,
            "candidate_team0_fuel_acquisitions": new_fuel,
            "team0_fuel_acquisition_gain": new_fuel - old_fuel,
        })

    count = len(rows)
    gains = [row["score_differential_gain"] for row in rows]
    team0_gains = [row["team0_score_gain"] for row in rows]
    fuel_gains = [row["team0_fuel_acquisition_gain"] for row in rows]
    by_seed: dict[str, list[dict]] = {}
    for row in rows:
        seed = row["scenario_id"].split(":", 1)[0]
        by_seed.setdefault(seed, []).append(row)
    return {
        "schema_version": 1,
        "scenario": baseline["scenario"],
        "settings": baseline["settings"],
        "device_selection": {
            "baseline": baseline_device,
            "candidate": candidate_device,
        },
        "baseline": {"label": baseline["label"], "revision": baseline["revision"]},
        "candidate": {"label": candidate["label"], "revision": candidate["revision"]},
        "summary": {
            "matches": count,
            "mean_team0_score_gain": sum(team0_gains) / count,
            "mean_score_differential_gain": sum(gains) / count,
            "candidate_win_count": sum(gain > 0 for gain in gains),
            "tie_count": sum(gain == 0 for gain in gains),
            "candidate_loss_count": sum(gain < 0 for gain in gains),
            "baseline_team0_fuel_acquisitions": sum(
                row["baseline_team0_fuel_acquisitions"] for row in rows),
            "candidate_team0_fuel_acquisitions": sum(
                row["candidate_team0_fuel_acquisitions"] for row in rows),
            "mean_team0_fuel_acquisition_gain": sum(fuel_gains) / count,
            "fuel_gain_world_count": sum(gain > 0 for gain in fuel_gains),
            "fuel_tie_world_count": sum(gain == 0 for gain in fuel_gains),
            "fuel_loss_world_count": sum(gain < 0 for gain in fuel_gains),
        },
        "per_seed": {
            seed: {
                "worlds": len(seed_rows),
                "mean_team0_score_gain": sum(
                    row["team0_score_gain"] for row in seed_rows) / len(seed_rows),
                "mean_score_differential_gain": sum(
                    row["score_differential_gain"] for row in seed_rows) / len(seed_rows),
                "mean_team0_fuel_acquisition_gain": sum(
                    row["team0_fuel_acquisition_gain"] for row in seed_rows) / len(seed_rows),
                "fuel_gain_world_count": sum(
                    row["team0_fuel_acquisition_gain"] > 0 for row in seed_rows),
                "fuel_loss_world_count": sum(
                    row["team0_fuel_acquisition_gain"] < 0 for row in seed_rows),
            }
            for seed, seed_rows in sorted(by_seed.items())
        },
        "per_match": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(_load(args.baseline), _load(args.candidate))
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"wrote {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
