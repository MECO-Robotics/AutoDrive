"""Read-only API payload builders used by the dashboard HTTP handler."""
from __future__ import annotations

import json
import math


def curriculum_status_payload(run_dir, curriculum_root=None, project_root=None) -> bytes:
    """Return a read-only curriculum summary and registry metadata."""
    from pathlib import Path
    from scripts.report_curriculum_progress import summarize
    from .training_curriculum import SCENARIOS, CURRICULUM_VERSION

    root = _curriculum_report_root(run_dir, project_root=project_root, curriculum_root=curriculum_root)
    try:
        report = summarize(root)  # Deliberately omit cache_dir: dashboard reads never write caches.
        error = None
    except Exception:  # Keep the independent panel usable when evidence is malformed.
        report = {"runs": [], "queue": None, "errors": []}
        error = "Could not read saved curriculum status."
    scenarios = {
        key: {"stage": item.stage, "prerequisites": list(item.prerequisites),
              "required_categories": list(item.required_categories),
              "success_criterion": item.success_criterion,
              "min_episodes": item.gate.min_episodes,
              "success_rate": item.gate.success_rate,
              "throughput_ratio": item.gate.throughput_ratio,
              "min_useful_throughput": item.gate.min_useful_throughput,
              "consecutive_evaluations": item.gate.consecutive_evaluations}
        for key, item in SCENARIOS.items()
    }
    valid_runs = []
    tracks = {name: {"track": name, "status": "unknown", "updates": None,
                     "device": "unknown", "current": "unknown", "evaluation": "unknown"}
              for name in ("A1", "A2", "A3", "C1", "C2")}
    for lane in report.get("queue_lanes", []):
        if isinstance(lane, dict) and lane.get("track") in tracks:
            name = lane["track"]
            tracks[name] = {"track": name, "status": lane.get("status", "unknown"),
                            "updates": lane.get("completed_updates"),
                            "device": lane.get("device", "unknown"),
                            "current": lane.get("current", "unknown"),
                            "evaluation": lane.get("evaluation", "unknown")}
    for run in report.get("runs", []):
        for lane in run.get("queue_lanes", []):
            if isinstance(lane, dict) and lane.get("track") in tracks:
                name = lane["track"]
                tracks[name] = {"track": name, "status": lane.get("status", "unknown"),
                                "updates": lane.get("completed_updates"),
                                "device": lane.get("device", "unknown"),
                                "current": lane.get("current", "unknown"),
                                "evaluation": lane.get("evaluation", "unknown")}
        rows = run.get("latest_evaluations", [])
        by_scenario = {}
        duplicate_categories = set()
        for row in rows:
            if isinstance(row, dict) and row.get("scenario") in SCENARIOS:
                scenario = row["scenario"]
                category = row.get("category")
                categories = by_scenario.setdefault(scenario, {})
                if category in categories:
                    duplicate_categories.add(scenario)
                categories[category] = row
        qualified_candidates = set()
        raw_qualified = run.get("qualified", [])
        evaluation_groups = {}
        for row in run.get("evaluation_history", []):
            if isinstance(row, dict) and row.get("scenario") in SCENARIOS:
                evaluation_groups.setdefault((row.get("scenario"), row.get("evaluation_id")), []).append(row)
        if isinstance(raw_qualified, list):
            for scenario, definition in SCENARIOS.items():
                required = definition.gate.consecutive_evaluations if definition else 0
                canonical_streak = run.get("consecutive_passes", {}).get(scenario, 0)
                canonical_policy = run.get("last_qualified_checkpoint", {}).get(scenario)
                canonical_protocol = run.get("qualified_protocol", {}).get(scenario)
                groups = [(evaluation_id, rows) for (name, evaluation_id), rows in evaluation_groups.items()
                          if name == scenario and evaluation_id]
                groups.sort(key=lambda item: max((row.get("update", -1) for row in item[1]), default=-1),
                            reverse=True)
                chain = []
                binding = None
                for evaluation_id, proof_rows in groups:
                    categories = {row.get("category"): row for row in proof_rows}
                    names = [row.get("category") for row in proof_rows]
                    candidate = (proof_rows[0].get("policy_id"), proof_rows[0].get("protocol_id")) if proof_rows else None
                    valid = (candidate and candidate[0] and candidate[1]
                             and len(names) == len(definition.required_categories)
                             and len(set(names)) == len(names)
                             and set(names) == set(definition.required_categories)
                             and all(row.get("evaluation_id") == evaluation_id
                                     and row.get("policy_id") == candidate[0]
                                     and row.get("protocol_id") == candidate[1]
                                     and _category_passes(row, definition, row.get("category"), CURRICULUM_VERSION)
                                     for row in proof_rows))
                    if not valid:
                        break
                    if binding is None:
                        binding = candidate
                    if candidate != binding or evaluation_id in [item["evaluation_id"] for item in chain]:
                        break
                    chain.append({"evaluation_id": evaluation_id, "policy_id": candidate[0],
                                  "protocol_id": candidate[1]})
                streak = min(len(chain), required)
                if (scenario in raw_qualified and type(canonical_streak) is int
                        and canonical_streak >= required and streak >= required
                        and binding == (canonical_policy, canonical_protocol)):
                    qualified_candidates.add(scenario)
                # Evaluation details accompany the run so the UI can show why a gate is unmet.
                run.setdefault("qualification_streaks", {})[scenario] = {
                    "streak": streak, "required": required,
                    "evaluation_ids": [entry["evaluation_id"] for entry in chain]}
        valid_qualified = []
        while True:
            ready = [scenario for scenario in SCENARIOS
                     if scenario in qualified_candidates and scenario not in valid_qualified
                     and set(SCENARIOS[scenario].prerequisites) <= set(valid_qualified)]
            if not ready:
                break
            valid_qualified.extend(ready)
        queue_status = run.get("queue_status")
        stale = bool(queue_status in ("queued", "running", "pending") and report.get("queue")
                     and report["queue"].get("age_seconds", 0) > 180)
        valid_runs.append({
            "run": run.get("run"), "algorithm": run.get("algorithm"),
            "architecture": run.get("architecture"), "status": "stale" if stale else queue_status or "saved",
            "updates": run.get("updates", 0), "current": run.get("current"),
            "qualified": valid_qualified, "consecutive_passes": run.get("consecutive_passes", {}),
            "qualification_streaks": run.get("qualification_streaks", {}),
            "diagnostic": run.get("diagnostic", False),
            "latest_evaluations": [{key: value for key, value in row.items()
                                    if key != "evidence"} for row in rows],
        })
    errors = report.get("errors", [])
    queue = report.get("queue") or {}
    payload = {"curriculum_version": CURRICULUM_VERSION, "scenarios": scenarios,
               "runs": valid_runs,
               "tracks": [tracks[name] for name in ("A1", "A2", "A3", "C1", "C2")],
               "three_pass_provenance": _three_pass_provenance(),
               "queue": {"age_seconds": queue.get("age_seconds")} if queue else None,
               "errors": [{"error": "Saved evidence could not be verified."}
                          for _ in errors[:10]], "error": error,
               "generated_at": report.get("generated_at"),
               "stale_after_seconds": 180}
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()


def _curriculum_report_root(run_dir, project_root=None, curriculum_root=None):
    """Use an explicit in-project root, or run_dir only when it is a curriculum root."""
    from pathlib import Path

    project = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    selected = Path(curriculum_root) if curriculum_root is not None else Path(run_dir)
    resolved = selected.resolve()
    if not (resolved == project or project in resolved.parents):
        return None
    if _has_curriculum_data(resolved) or curriculum_root is not None and resolved.is_dir():
        return resolved
    return None


def _has_curriculum_data(root):
    from pathlib import Path

    root = Path(root)
    if any((root / name).is_file() for name in
           ("orchestration.json", "queue-status.json", "queue.json", "manifest.json")):
        return True
    for current, dirs, files in __import__("os").walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        if "stats.json" in files:
            return True
    return False


def _three_pass_provenance():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "docs" / "MECO_THREE_PASS_PROVENANCE.json"
    try:
        data = json.loads(path.read_text())
        if data.get("schema") != "meco-three-pass-provenance-v1":
            raise ValueError("unsupported provenance schema")
        return data
    except (OSError, ValueError, TypeError):
        return {"status": "unknown", "tranches": []}


def _category_passes(row, definition, category, curriculum_version):
    gate = definition.gate
    return (isinstance(row, dict)
            and row.get("scenario") == definition.scenario_id
            and row.get("category") == category
            and row.get("scenario_version") == definition.version
            and row.get("curriculum_version") == curriculum_version
            and row.get("split") == "validation"
            and row.get("diagnostic") is False
            and row.get("paired") is True
            and row.get("reference_qualified") is True
            and type(row.get("episodes")) is int
            and row["episodes"] >= gate.min_episodes
            and type(row.get("successes")) is int
            and 0 <= row["successes"] <= row["episodes"]
            and isinstance(row.get("success_rate"), (int, float))
            and math.isfinite(row["success_rate"])
            and row["success_rate"] >= gate.success_rate
            and isinstance(row.get("useful_throughput"), (int, float))
            and math.isfinite(row["useful_throughput"])
            and row["useful_throughput"] >= gate.min_useful_throughput
            and (not gate.throughput_ratio
                 or isinstance(row.get("throughput_ratio"), (int, float))
                 and math.isfinite(row["throughput_ratio"])
                 and row["throughput_ratio"] >= gate.throughput_ratio))


def field_layout_payload() -> bytes:
    """Return the compact field geometry document served by ``/api/field-layout``."""
    from .field import (
        ALLIANCE_ZONE_DEPTH,
        bump_boxes,
        rebuilt_field,
        static_collision_boxes,
    )

    boxes = rebuilt_field()
    field_length = 651.22 * 0.0254
    field_width = 317.7 * 0.0254
    return json.dumps(
        {
            "length": field_length,
            "width": field_width,
            "alliance_zone_depth": ALLIANCE_ZONE_DEPTH,
            "elements": [box.as_dict() for box in boxes],
            "colliders": [box.as_dict() for box in static_collision_boxes(boxes)],
            "bump_regions": [box.as_dict() for box in bump_boxes(boxes)],
        },
        separators=(",", ":"),
    ).encode()
