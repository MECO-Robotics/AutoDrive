"""Package explicit Plan A human annotations for the replay-bound consumer.

This module only packages labels. It does not authenticate an annotator or
establish expertise, optimality, or replay validity. The existing annotation
loader remains the acceptance boundary for exact simulator replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from . import meco_supervised as supervised
from . import meco_counterfactual as counterfactual


IDENTITY = ("scenario_id", "category", "seed", "decision_index")
AUTHORED_KEYS = set(IDENTITY) | {
    "selected_goal", "executed_action", "selected_behavior_id",
    "censored_feasible_behavior_ids", "u_status", "u_target",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(path: Path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value} at {path}:{number}")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row at {path}:{number} must be an object")
        rows.append(row)
    return rows


def _key(row):
    key = tuple(row.get(k) for k in IDENTITY)
    if (not isinstance(key[0], str) or not key[0] or
            not isinstance(key[1], str) or not key[1] or
            type(key[2]) is not int or key[2] < 0 or
            type(key[3]) is not int or key[3] < 0):
        raise ValueError("annotation identity must be (scenario, category, seed, decision_index)")
    return key


def _index(rows, label):
    out = {}
    for row in rows:
        key = _key(row)
        if key in out:
            raise ValueError(f"duplicate {label} identity: {key}")
        out[key] = row
    return out


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def _preflight_destinations(output: Path, input_paths):
    output = _resolved(output)
    manifest = _resolved(output.with_suffix(output.suffix + ".manifest.json"))
    if output == manifest:
        raise ValueError("dataset and manifest output paths resolve to the same file")
    for label, input_path in input_paths:
        if _resolved(input_path) in (output, manifest):
            raise ValueError(f"output path collides with input {label}: {_resolved(input_path)}")
    return output, manifest


def _stage_bytes(data: bytes, destination: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.meco-tmp-",
                                dir=destination.parent)
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_pair(output: Path, manifest_path: Path, data: bytes, manifest_data: bytes):
    """Publish two staged files and restore the prior pair if either replace fails."""
    staged, backups = [], []
    preserve_backups = False
    destinations = (output, manifest_path)
    try:
        for destination, payload in zip(destinations, (data, manifest_data)):
            staged.append(_stage_bytes(payload, destination))
        for destination in destinations:
            if destination.is_dir():
                raise ValueError(f"output destination is a directory: {destination}")
            backups.append(None)
            if destination.exists():
                backups[-1] = _stage_bytes(destination.read_bytes(), destination)
        os.replace(staged[0], output)
        staged[0] = None
        os.replace(staged[1], manifest_path)
        staged[1] = None
        _fsync_directory(output.parent)
    except BaseException as exc:
        rollback_errors = []
        for index, (destination, backup) in enumerate(zip(destinations, backups)):
            try:
                if backup is not None:
                    os.replace(backup, destination)
                    backups[index] = None
                else:
                    destination.unlink(missing_ok=True)
            except BaseException as rollback_exc:
                rollback_errors.append(f"{destination}: {rollback_exc}")
        try:
            _fsync_directory(output.parent)
        except BaseException as rollback_exc:
            rollback_errors.append(f"directory fsync: {rollback_exc}")
        preserve_backups = bool(rollback_errors)
        detail = ("; rollback errors: " + "; ".join(rollback_errors)) if rollback_errors else ""
        if preserve_backups:
            detail += "; recovery backups retained: " + ", ".join(str(p) for p in backups if p is not None)
        raise RuntimeError(f"failed to publish annotation dataset/manifest pair: {exc}{detail}") from exc
    finally:
        for path in staged + ([] if preserve_backups else backups):
            if path is not None:
                path.unlink(missing_ok=True)


def materialize_plan_a_annotations(*, authored_rows: Path, source_dataset: Path,
        legal_node_dataset: Path, qualification_protocol: Path,
        authority_json: Path, output: Path):
    """Write loader-schema JSONL and manifest from explicit authored sidecars."""
    authored_rows, source_dataset, legal_node_dataset, qualification_protocol, authority_json = map(
        Path, (authored_rows, source_dataset, legal_node_dataset, qualification_protocol, authority_json))
    source_manifest_path = _resolved(source_dataset.with_suffix(source_dataset.suffix + ".manifest.json"))
    legal_manifest_path = _resolved(legal_node_dataset.with_suffix(legal_node_dataset.suffix + ".manifest.json"))
    output, manifest_path = _preflight_destinations(output, (
        ("annotation sidecar", authored_rows), ("source JSONL", source_dataset),
        ("source manifest", source_manifest_path), ("legal-node JSONL", legal_node_dataset),
        ("legal-node manifest", legal_manifest_path),
        ("qualification protocol", qualification_protocol), ("authority JSON", authority_json)))
    source_manifest_bytes = source_manifest_path.read_bytes()
    legal_manifest_bytes = legal_manifest_path.read_bytes()
    source_manifest, legal_manifest = json.loads(source_manifest_bytes), json.loads(legal_manifest_bytes)
    source_bytes, legal_bytes = source_dataset.read_bytes(), legal_node_dataset.read_bytes()
    source_sha, legal_sha = _sha(source_bytes), _sha(legal_bytes)
    protocol, protocol_id, protocol_sha, _ = supervised._load_protocol(qualification_protocol)
    authority_bytes = authority_json.read_bytes()
    authority = json.loads(authority_bytes)
    if not isinstance(authority, dict) or not isinstance(authority.get("source_id"), str) or not authority["source_id"].strip():
        raise ValueError("authority JSON must declare a nonempty source_id")
    # This is a declared label source only, not authenticated identity or proof of expertise.
    if (source_manifest.get("schema") != counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA or
            source_manifest.get("target_schema") != counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA or
            source_manifest.get("dataset_sha256") != source_sha or
            source_manifest.get("split") != "train" or
            source_manifest.get("protocol_id") != protocol_id or
            source_manifest.get("qualification_protocol_sha256") != protocol_sha or
            source_manifest.get("records_have_replay_bound_selected_goal") is not True):
        raise ValueError("explicit-goal source does not bind its bytes and frozen train protocol")
    base_name = source_manifest.get("base_dataset")
    if not isinstance(base_name, str) or not base_name or Path(base_name).is_absolute():
        raise ValueError("explicit-goal source has invalid base dataset reference")
    base_path = (source_dataset.parent / base_name).resolve()
    base_manifest_path = base_path.with_suffix(base_path.suffix + ".manifest.json")
    output, manifest_path = _preflight_destinations(output, (
        ("annotation sidecar", authored_rows), ("source JSONL", source_dataset),
        ("source manifest", source_manifest_path), ("legal-node JSONL", legal_node_dataset),
        ("legal-node manifest", legal_manifest_path),
        ("qualification protocol", qualification_protocol), ("authority JSON", authority_json),
        ("base source JSONL", base_path), ("base source manifest", base_manifest_path)))
    base_sha, base_manifest_sha = _sha(base_path.read_bytes()), _sha(base_manifest_path.read_bytes())
    if (source_manifest.get("base_dataset_sha256") != base_sha or
            source_manifest.get("base_manifest_sha256") != base_manifest_sha or
            legal_manifest.get("schema") != counterfactual.PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            legal_manifest.get("target_schema") != counterfactual.PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            legal_manifest.get("dataset_sha256") != legal_sha or
            legal_manifest.get("source_dataset_sha256") != base_sha or
            legal_manifest.get("source_manifest_sha256") != base_manifest_sha or
            legal_manifest.get("expert_goal_source_sha256") != source_sha or
            legal_manifest.get("expert_goal_source_manifest_sha256") != _sha(source_manifest_bytes) or
            legal_manifest.get("protocol_id") != protocol_id or
            legal_manifest.get("qualification_protocol_sha256") != protocol_sha or
            legal_manifest.get("split") != "train"):
        raise ValueError("legal-node/source/protocol provenance does not match current bytes")
    source = _index(_jsonl(source_dataset), "source")
    legal = _index(_jsonl(legal_node_dataset), "legal-node")
    authored = _index(_jsonl(authored_rows), "authored")
    if (source_manifest.get("record_count") != len(source) or
            legal_manifest.get("record_count") != len(legal)):
        raise ValueError("source/legal-node manifest record count mismatch")
    if set(source) != set(legal):
        raise ValueError("source and legal-node identities do not match exactly")
    if set(authored) != set(source):
        missing, extra = set(source) - set(authored), set(authored) - set(source)
        raise ValueError(f"authored identities must exactly cover source rows (missing={len(missing)}, extra={len(extra)})")
    expected_source_identity = source_manifest.get("source_identity_sha256")
    if (source_manifest.get("sources") != legal_manifest.get("sources") or
            source_manifest.get("source_identity_sha256") != legal_manifest.get("source_identity_sha256")):
        raise ValueError("source identity declarations differ between source and legal-node manifests")
    rows = []
    for key in sorted(source):
        src, node_row, label = source[key], legal[key], authored[key]
        base = src.get("base_record")
        if (src.get("schema") != counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_SCHEMA or
                src.get("target_schema") != counterfactual.PLAN_A_EXPERT_GOAL_SOURCE_TARGET_SCHEMA or
                not isinstance(base, dict) or _key(base) != key or
                src.get("snapshot_id") != base.get("branch_state", {}).get("snapshot_id") or
                src.get("state_sha256") != base.get("predecision_simulator_state_sha256") or
                src.get("support_digest") != node_row.get("support_digest") or
                src.get("snapshot_id") != node_row.get("snapshot_id") or
                src.get("state_sha256") != node_row.get("state_sha256")):
            raise ValueError("source/legal-node row provenance does not match exact base state and support")
        if (node_row.get("schema") != counterfactual.PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
                node_row.get("target_schema") != counterfactual.PLAN_A_LEGAL_NODE_TARGET_SCHEMA):
            raise ValueError("legal-node row schema mismatch")
        unknown = set(label) - AUTHORED_KEYS
        missing_fields = {"selected_goal", "executed_action", "selected_behavior_id",
                          "censored_feasible_behavior_ids", "u_status"} - set(label)
        if unknown or missing_fields:
            raise ValueError(f"authored row has unsupported/incomplete fields (unknown={sorted(unknown)}, missing={sorted(missing_fields)})")
        allowed = supervised._protocol_train_seeds(protocol, key[0], key[1])
        if key[2] not in allowed:
            raise ValueError(f"authored identity uses a heldout or undeclared seed: {key}")
        selected = src.get("selected_goal")
        if (not isinstance(selected, dict) or not isinstance(label["selected_goal"], dict) or
                label["selected_goal"].get("node_id") != selected.get("node_id") or
                label["selected_goal"].get("descriptor_sha256") != src.get("teacher_selected_goal_descriptor_sha256") or
                selected.get("descriptor_sha256") != src.get("teacher_selected_goal_descriptor_sha256") or
                type(src.get("executed_action")) is not int or type(label["executed_action"]) is not int or
                label["executed_action"] != src["executed_action"] or
                type(selected.get("behavior_id")) is not int or type(label["selected_behavior_id"]) is not int or
                label["selected_behavior_id"] != selected["behavior_id"]):
            raise ValueError("authored goal/action/behavior labels mismatch replay-bound executed source")
        if src.get("execution_failure") is not False:
            raise ValueError("authored goal is not the exact successfully executed replay-bound goal")
        node = label["selected_goal"].get("node_id")
        pose = next((p for p in node_row.get("canonical_poses", []) if p.get("node_id") == node), None)
        if (type(node) is not int or node not in node_row.get("feasible_node_ids", []) or
                pose is None or any(label["selected_goal"].get(k) != pose.get(k)
                    for k in ("node_id", "x_m", "y_m", "heading_bin"))):
            raise ValueError("authored goal is missing, inferred, mismatched, or infeasible")
        behavior = label["selected_behavior_id"]
        if not any(p.get("node_id") == node and p.get("behavior_id") == behavior and
                   p.get("status") == "available" for p in node_row.get("pairs", [])):
            raise ValueError("authored behavior is infeasible or unavailable at selected goal")
        target = next((v for v in node_row.get("node_u_targets", []) if v.get("node_id") == node), None)
        censored = label["censored_feasible_behavior_ids"]
        if (target is None or not isinstance(censored, list) or any(type(x) is not int for x in censored) or
                len(set(censored)) != len(censored) or set(censored) != set(target.get("censored_feasible_behavior_ids", []))):
            raise ValueError("U support/censoring declaration differs from bound feasible support")
        u_status, u_target = label["u_status"], label.get("u_target")
        if u_status == "available":
            if (type(u_target) not in (int, float) or not math.isfinite(u_target) or censored or
                    target.get("status") != "available" or
                    set(target.get("feasible_behavior_ids", [])) != set(target.get("available_behavior_ids", []))):
                raise ValueError("available U requires finite value and complete uncensored feasible support")
        elif u_status in ("unavailable", "censored"):
            if u_target is not None or (u_status == "censored" and not censored):
                raise ValueError("unavailable/censored U has inconsistent value or support")
        else:
            raise ValueError("U status must be explicitly available, unavailable, or censored")
        rows.append({
            "schema": supervised.PLAN_A_EXPERT_GOAL_DATASET_SCHEMA,
            "target_schema": supervised.PLAN_A_EXPERT_GOAL_TARGET_SCHEMA,
            **dict(zip(IDENTITY, key)), "annotation_source_id": authority["source_id"],
            "snapshot_id": src["snapshot_id"], "state_sha256": src.get("simulator_state_sha256", src.get("state_sha256")),
            "support_digest": node_row["support_digest"],
            "executed_goal_descriptor_sha256": src["teacher_selected_goal_descriptor_sha256"],
            "executed_action": label["executed_action"], "selected_behavior_id": behavior,
            "selected_goal": label["selected_goal"], "censored_feasible_behavior_ids": sorted(censored),
            "u_status": u_status, "u_target": u_target,
        })
    data = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode() for row in rows)
    manifest = {
        "schema": supervised.PLAN_A_EXPERT_GOAL_DATASET_SCHEMA,
        "target_schema": supervised.PLAN_A_EXPERT_GOAL_TARGET_SCHEMA,
        "target_semantics": supervised.PLAN_A_EXPERT_GOAL_SEMANTICS,
        "split": "train", "source_dataset_sha256": base_sha,
        "source_manifest_sha256": base_manifest_sha,
        "expert_goal_source_sha256": source_sha,
        "expert_goal_source_manifest_sha256": _sha(source_manifest_bytes),
        "legal_node_dataset_sha256": legal_sha,
        "legal_node_manifest_sha256": _sha(legal_manifest_bytes),
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "sources": source_manifest["sources"], "source_identity_sha256": expected_source_identity,
        "dataset_sha256": _sha(data),
        "annotation_source": {"source_id": authority["source_id"], "sha256": _sha(authority_bytes)},
        "authority_declared_only": True,
        "authority_is_authenticated_identity": False,
        "authority_proves_expertise_or_optimality": False,
        "replay_validation_required": True,
        "support_order": supervised.SUPPORT_ORDER, "record_count": len(rows),
        "materializer_schema": "meco-plan-a-human-annotation-materializer-v1",
        "authored_rows_sha256": _sha(authored_rows.read_bytes()),
    }
    manifest_data = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish_pair(output, manifest_path, data, manifest_data)
    return {"dataset": str(output), "rows": len(rows), "dataset_sha256": _sha(data),
            "manifest": str(manifest_path), "manifest_sha256": _sha(manifest_data),
            "replay_validation_required": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authored-rows", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--legal-node-dataset", type=Path, required=True)
    parser.add_argument("--qualification-protocol", type=Path, required=True)
    parser.add_argument("--authority-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(materialize_plan_a_annotations(authored_rows=args.authored_rows,
        source_dataset=args.source_dataset, legal_node_dataset=args.legal_node_dataset,
        qualification_protocol=args.qualification_protocol, authority_json=args.authority_json,
        output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
