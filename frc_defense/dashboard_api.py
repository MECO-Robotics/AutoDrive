"""Read-only API payload builders used by the dashboard HTTP handler."""
from __future__ import annotations

import json
import math

_EXPECTED_THREE_PASS_PROVENANCE_SHA256 = "fde7ec26feeceafa13f960a2bc65dcba5d8ab81cb09618f7a35e259dd13295d5"
_EXPECTED_PLAN_SHA256 = "718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10"
_QUEUE_STATUS_FILES = {"queue-status.json", "orchestration.json", "queue.json",
                      "task-scheduler-state.json"}


def _curriculum_error_projection(item):
    """Return a bounded error class without projecting raw paths or details."""
    if isinstance(item, dict):
        path = item.get("path")
        if (isinstance(path, str) and len(path) <= 512
                and path.rsplit("/", 1)[-1] in _QUEUE_STATUS_FILES):
            return {"error": "Training queue status could not be projected."}
        detail = item.get("error")
        if isinstance(detail, str) and len(detail) <= 128 and detail.startswith("queue "):
            return {"error": "Training queue status could not be projected."}
    return {"error": "Saved evaluation evidence could not be verified."}


def curriculum_status_payload(run_dir, curriculum_root=None, project_root=None,
                              meco_data_root=None) -> bytes:
    """Return a read-only curriculum summary and registry metadata."""
    from pathlib import Path
    from scripts.report_curriculum_progress import summarize
    from .training_curriculum import SCENARIOS, CURRICULUM_VERSION

    root = _curriculum_report_root(
        run_dir, project_root=project_root, curriculum_root=curriculum_root,
        meco_data_root=meco_data_root)
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
                eval_every = run.get("eval_every")
                cadence_valid = type(eval_every) is int and eval_every > 0
                canonical_streak = run.get("consecutive_passes", {}).get(scenario, 0)
                canonical_policy = run.get("last_qualified_checkpoint", {}).get(scenario)
                canonical_protocol = run.get("qualified_protocol", {}).get(scenario)
                consecutive_protocol = run.get("consecutive_protocol", {}).get(scenario)
                groups = []
                for (name, evaluation_id), proof_rows in evaluation_groups.items():
                    if name != scenario:
                        continue
                    updates = {row.get("update") for row in proof_rows}
                    update = next(iter(updates)) if len(updates) == 1 else None
                    groups.append((update, evaluation_id, proof_rows))
                groups.sort(key=lambda item: item[0] if type(item[0]) is int else -1, reverse=True)
                chain = []
                binding_protocol = None
                for item_update, evaluation_id, proof_rows in groups:
                    categories = {row.get("category"): row for row in proof_rows}
                    names = [row.get("category") for row in proof_rows]
                    candidate = (proof_rows[0].get("policy_id"), proof_rows[0].get("protocol_id")) if proof_rows else None
                    valid = (cadence_valid and type(item_update) is int and item_update > 0
                             and item_update % eval_every == 0
                             and isinstance(evaluation_id, str) and evaluation_id
                             and candidate and candidate[0] and candidate[1]
                             and len(names) == len(definition.required_categories)
                             and len(set(names)) == len(names)
                             and set(names) == set(definition.required_categories)
                             and all(row.get("evaluation_id") == evaluation_id
                                     and row.get("policy_id") == candidate[0]
                                     and row.get("protocol_id") == candidate[1]
                                     and _category_passes(row, definition, row.get("category"), CURRICULUM_VERSION)
                                     for row in proof_rows))
                    if not valid or run.get("evaluation_history_incomplete"):
                        break
                    if binding_protocol is None:
                        binding_protocol = candidate[1]
                    if (candidate[1] != binding_protocol
                            or evaluation_id in [item["evaluation_id"] for item in chain]
                            or chain and item_update != chain[-1]["update"] - eval_every):
                        break
                    chain.append({"evaluation_id": evaluation_id, "policy_id": candidate[0],
                                  "protocol_id": candidate[1], "update": item_update})
                streak = min(len(chain), required)
                if (scenario in raw_qualified and type(canonical_streak) is int
                        and canonical_streak >= required and streak >= required
                        and binding_protocol == canonical_protocol == consecutive_protocol
                        and chain and chain[0]["policy_id"] == canonical_policy):
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
            "updates": run.get("updates", 0), "eval_every": run.get("eval_every"),
            "current": run.get("current"),
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
               "errors": [_curriculum_error_projection(item)
                          for item in errors[:10]], "error": error,
               "generated_at": report.get("generated_at"),
               "stale_after_seconds": 180}
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()


def _curriculum_report_root(run_dir, project_root=None, curriculum_root=None,
                            meco_data_root=None):
    """Use an explicit checkout/data root, or an implicit in-checkout run root."""
    from pathlib import Path

    project = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    selected = Path(curriculum_root) if curriculum_root is not None else Path(run_dir)
    if _contains_symlink(selected):
        return None
    resolved = selected.resolve()
    allowed_roots = [project]
    if curriculum_root is not None and meco_data_root is not None:
        allowed_roots.append(Path(meco_data_root).resolve())
    if not any(resolved == allowed or allowed in resolved.parents
               for allowed in allowed_roots):
        return None
    if _has_curriculum_data(resolved) or curriculum_root is not None and resolved.is_dir():
        return resolved
    return None


def _has_curriculum_data(root):
    from pathlib import Path

    root = Path(root)
    if any((root / name).is_file() and not (root / name).is_symlink() for name in
           ("orchestration.json", "queue-status.json", "queue.json", "manifest.json")):
        return True
    for current, dirs, files in __import__("os").walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        if "stats.json" in files:
            return True
    return False


def _contains_symlink(path):
    from pathlib import Path
    absolute = Path(path).absolute()
    return any(part.is_symlink() for part in (absolute, *absolute.parents))


def _three_pass_provenance():
    import hashlib
    from pathlib import Path
    project = Path(__file__).resolve().parents[1]
    relative_manifest = "docs/MECO_THREE_PASS_PROVENANCE.json"
    try:
        path = _verified_repo_file(project, relative_manifest)
        if path is None:
            return _untrusted_three_pass("manifest path is missing, escaped, or symlinked")
        if hashlib.sha256(path.read_bytes()).hexdigest() != _EXPECTED_THREE_PASS_PROVENANCE_SHA256:
            return _untrusted_three_pass("manifest hash does not match the dashboard-pinned digest")
        data = json.loads(path.read_text())
        return _validate_three_pass_manifest(project, data)
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
        return _untrusted_three_pass(f"manifest could not be read: {type(exc).__name__}")


def _untrusted_three_pass(reason):
    return {"status": "untrusted", "evidence_class": "scoped_transcript_summary",
            "tranches": [], "untrusted_reason": reason}


def _validate_three_pass_manifest(project, data):
    """Validate summary-only role evidence and exact commit linkage."""
    import hashlib
    import re

    def reject(reason):
        return _untrusted_three_pass(reason)

    if not isinstance(data, dict) or data.get("schema") != "meco-three-pass-provenance-v2":
        return reject("unsupported provenance schema")
    if (data.get("status") != "scoped-transcript-summaries"
            or data.get("evidence_class") != "scoped_transcript_summary"):
        return reject("evidence is not explicitly classified as scoped transcript summaries")
    if data.get("plan_sha256") != _EXPECTED_PLAN_SHA256:
        return reject("plan digest does not match the dashboard-pinned digest")
    plan = _verified_repo_file(project, data.get("plan_path"))
    if plan is None or hashlib.sha256(plan.read_bytes()).hexdigest() != _EXPECTED_PLAN_SHA256:
        return reject("plan path or digest is invalid")
    if (not isinstance(data.get("scope"), str) or not data["scope"].strip()
            or not isinstance(data.get("limits"), list) or not data["limits"]
            or any(not isinstance(item, str) or not item.strip() for item in data["limits"])):
        return reject("scope and summary-only limits are required")
    tranches = data.get("tranches")
    if not isinstance(tranches, list) or not tranches:
        return reject("tranche list is missing or empty")
    seen = set()
    commit_pattern = re.compile(r"^[0-9a-f]{40}$")
    path_pattern = re.compile(r"^[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)*$")

    def valid_artifact(role):
        artifact = role.get("artifact") if isinstance(role, dict) else None
        if not isinstance(artifact, dict):
            return False
        if (artifact.get("kind") != "scoped-transcript-summary"
                or artifact.get("status") not in {"transcript-summary", "review-summary"}
                or not isinstance(artifact.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])):
            return False
        path = _verified_repo_file(project, artifact.get("path"))
        return path is not None and hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    for tranche in tranches:
        if not isinstance(tranche, dict):
            return reject("tranche entry is not an object")
        requirement_id = tranche.get("requirement_id")
        if (not isinstance(requirement_id, str)
                or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", requirement_id)
                or requirement_id in seen):
            return reject("tranche requirement ID is malformed or duplicated")
        seen.add(requirement_id)
        if not isinstance(tranche.get("title"), str) or not tranche["title"].strip():
            return reject(f"{requirement_id}: title is missing")
        planner, implementer, reviewer = (tranche.get(key) for key in ("planner", "implementer", "reviewer"))
        if (not isinstance(planner, dict) or planner.get("status") != "summary-only"
                or not valid_artifact(planner)
                or planner["artifact"].get("status") != "transcript-summary"):
            return reject(f"{requirement_id}: planner summary artifact is invalid")
        if not isinstance(implementer, dict) or implementer.get("status") != "implemented":
            return reject(f"{requirement_id}: implementation status is invalid")
        commits = [value for key, value in implementer.items() if key.endswith("_commit") or key == "commit"]
        if not commits or any(not isinstance(value, str) or not commit_pattern.fullmatch(value) for value in commits):
            return reject(f"{requirement_id}: implementation commit IDs must be exact 40-character hashes")
        changed_paths = implementer.get("paths")
        if (not isinstance(changed_paths, list) or not changed_paths
                or any(not isinstance(item, str) for item in changed_paths)
                or len(set(changed_paths)) != len(changed_paths)):
            return reject(f"{requirement_id}: implementation paths are missing or duplicated")
        for changed_path in changed_paths:
            if (not isinstance(changed_path, str) or not path_pattern.fullmatch(changed_path)
                    or _verified_repo_file(project, changed_path) is None):
                return reject(f"{requirement_id}: implementation path is missing or unsafe")
        if (not isinstance(reviewer, dict) or reviewer.get("status") != "approved-summary"
                or reviewer.get("verdict") != "approved" or not valid_artifact(reviewer)
                or reviewer["artifact"].get("status") != "review-summary"):
            return reject(f"{requirement_id}: reviewer evidence does not record an approved summary")
        approved = reviewer.get("approved_commit")
        if not isinstance(approved, str) or not commit_pattern.fullmatch(approved):
            return reject(f"{requirement_id}: reviewer approved commit is not an exact 40-character hash")
        if not isinstance(reviewer.get("verdict_detail"), str) or not reviewer["verdict_detail"].strip():
            return reject(f"{requirement_id}: reviewer verdict detail is missing")
        if not isinstance(tranche.get("limits"), str) or not tranche["limits"].strip():
            return reject(f"{requirement_id}: tranche limits are missing")
    result = dict(data)
    result["validation_status"] = "validated_summary_only"
    result["message"] = "Validated scoped summaries; this is not whole-plan approval or agent-liveness evidence."
    return result


def _verified_repo_file(project, relative_path):
    from pathlib import Path
    if not isinstance(relative_path, str) or not relative_path:
        return None
    try:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            return None
        lexical = project / relative
        if _contains_symlink(lexical):
            return None
        resolved = lexical.resolve()
        if resolved != project and project not in resolved.parents:
            return None
        return resolved if resolved.is_file() and not resolved.is_symlink() else None
    except (OSError, ValueError, RuntimeError):
        return None


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
