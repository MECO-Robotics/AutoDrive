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
    if baseline["settings"] != candidate["settings"]:
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
        rows.append({
            "scenario_id": old["scenario_id"],
            "baseline_score_by_team": old["score_by_team"],
            "candidate_score_by_team": new["score_by_team"],
            "team0_score_gain": new["score_by_team"][0] - old["score_by_team"][0],
            "score_differential_gain": new_diff - old_diff,
        })

    count = len(rows)
    gains = [row["score_differential_gain"] for row in rows]
    team0_gains = [row["team0_score_gain"] for row in rows]
    return {
        "schema_version": 1,
        "scenario": baseline["scenario"],
        "settings": baseline["settings"],
        "baseline": {"label": baseline["label"], "revision": baseline["revision"]},
        "candidate": {"label": candidate["label"], "revision": candidate["revision"]},
        "summary": {
            "matches": count,
            "mean_team0_score_gain": sum(team0_gains) / count,
            "mean_score_differential_gain": sum(gains) / count,
            "candidate_win_count": sum(gain > 0 for gain in gains),
            "tie_count": sum(gain == 0 for gain in gains),
            "candidate_loss_count": sum(gain < 0 for gain in gains),
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
