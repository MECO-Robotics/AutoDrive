"""Read-only, fail-closed summary of Plan A legal-node U/B labels.

This audits stored artifacts only. It deliberately does not replay or refit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha(value) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode())


def _jsonl(data: bytes, label: str):
    try:
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSONL") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must contain nonempty object rows")
    return rows


def _number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value))


def audit_plan_a_legal_nodes(dataset: Path, source_dataset: Path,
        qualification_protocol: Path) -> dict:
    """Verify artifact bindings and summarize recorded sampled U/B coverage."""
    from .meco_counterfactual import (DATASET_SCHEMA,
        TARGET_SCHEMA, PLAN_A_LEGAL_NODE_DATASET_SCHEMA, PLAN_A_LEGAL_NODE_TARGET_SCHEMA,
        _artifact_identity_sha256)
    from .meco_supervised import _load_protocol, _protocol_train_seeds

    dataset, source_dataset, qualification_protocol = map(Path.resolve,
        (Path(dataset), Path(source_dataset), Path(qualification_protocol)))
    legal_manifest_path = dataset.with_suffix(dataset.suffix + ".manifest.json")
    source_manifest_path = source_dataset.with_suffix(source_dataset.suffix + ".manifest.json")
    legal_bytes, legal_manifest_bytes = dataset.read_bytes(), legal_manifest_path.read_bytes()
    source_bytes, source_manifest_bytes = source_dataset.read_bytes(), source_manifest_path.read_bytes()
    legal_manifest, source_manifest = json.loads(legal_manifest_bytes), json.loads(source_manifest_bytes)
    if not isinstance(legal_manifest, dict) or not isinstance(source_manifest, dict):
        raise ValueError("dataset manifests must be JSON objects")
    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(qualification_protocol)
    legal_rows, source_rows = _jsonl(legal_bytes, "legal-node dataset"), _jsonl(source_bytes, "source dataset")
    legal_sha, legal_manifest_sha = _sha(legal_bytes), _sha(legal_manifest_bytes)
    source_sha, source_manifest_sha = _sha(source_bytes), _sha(source_manifest_bytes)

    sources = legal_manifest.get("sources")
    source_identity_sha = legal_manifest.get("source_identity_sha256")
    if not isinstance(sources, dict) or source_identity_sha != _canonical_sha(sources):
        raise ValueError("legal-node manifest source identity digest is missing or invalid")
    if (legal_manifest.get("schema") != PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            legal_manifest.get("target_schema") != PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            legal_manifest.get("split") != "train" or
            legal_manifest.get("dataset_sha256") != legal_sha or
            legal_manifest.get("record_count") != len(legal_rows) or
            legal_manifest.get("source_dataset_sha256") != source_sha or
            legal_manifest.get("source_dataset") != source_dataset.name or
            legal_manifest.get("source_manifest_sha256") != source_manifest_sha or
            legal_manifest.get("protocol_id") != protocol_id or
            legal_manifest.get("qualification_protocol_sha256") != protocol_sha or
            legal_manifest.get("qualification_protocol_file") != protocol_path.name):
        raise ValueError("legal-node manifest bytes, source, protocol, or train-split binding mismatch")
    if (source_manifest.get("schema") != DATASET_SCHEMA or
            source_manifest.get("target_schema") != TARGET_SCHEMA or
            source_manifest.get("records_have_prefix_replay_parity") is not True or
            source_manifest.get("split") != "train" or
            source_manifest.get("dataset_sha256") != source_sha or
            source_manifest.get("record_count") != len(source_rows) or
            source_manifest.get("protocol_id") != protocol_id or
            source_manifest.get("qualification_protocol_sha256") != protocol_sha or
            source_manifest.get("qualification_protocol_file") != protocol_path.name or
            source_manifest.get("sources") != sources or
            source_manifest.get("source_identity_sha256") != source_identity_sha):
        raise ValueError("source manifest bytes, source identity, protocol, or train-split binding mismatch")
    native_identity = source_manifest.get("native_extension_identity")
    if (source_manifest.get("native_extension_identity_schema") != 1 or
            source_manifest.get("artifact_identity_sha256") !=
                _artifact_identity_sha256(source_identity_sha, native_identity)):
        raise ValueError("source native-extension artifact identity binding is missing or invalid")
    if (legal_manifest.get("sources") != source_manifest.get("sources") or
            legal_manifest.get("source_identity_sha256") != source_manifest.get("source_identity_sha256") or
            legal_manifest.get("device") != source_manifest.get("device")):
        raise ValueError("legal-node/source source-identity or runtime-device binding mismatch")
    scenario, category = source_manifest.get("scenario_id"), source_manifest.get("category")
    if not isinstance(scenario, str) or not isinstance(category, str):
        raise ValueError("source manifest scenario/category is missing")
    train_seeds = _protocol_train_seeds(protocol, scenario, category)
    declared_seeds = source_manifest.get("seeds")
    if (not isinstance(declared_seeds, list) or not declared_seeds or
            any(type(seed) is not int for seed in declared_seeds) or
            len(set(declared_seeds)) != len(declared_seeds)):
        raise ValueError("source manifest has an invalid exact seed list")
    allowed_seeds = set(declared_seeds)
    if (not allowed_seeds or
            not allowed_seeds <= train_seeds):
        raise ValueError("source manifest seeds are outside the frozen protocol train split")

    def index(rows, label):
        result = {}
        for row in rows:
            key = (row.get("scenario_id"), row.get("category"), row.get("seed"),
                   row.get("decision_index"))
            if (type(key[2]) is not int or type(key[3]) is not int or key in result or
                    key[0] != scenario or key[1] != category or key[2] not in allowed_seeds):
                raise ValueError(f"{label} contains malformed, duplicate, or out-of-train identities")
            result[key] = row
        return result

    source_by_key, legal_by_key = index(source_rows, "source dataset"), index(legal_rows, "legal-node dataset")
    if set(source_by_key) != set(legal_by_key):
        raise ValueError("source and legal-node rows do not bind the exact same branch identities")

    totals = Counter()
    heading_region = Counter()
    feasible_behavior_counts = Counter()
    sampled_behavior_counts = Counter()
    feasible_descriptor_counts = Counter()
    per_behavior_variances = {str(i): [] for i in range(5)}
    node_weights = []
    descriptor_weights = []
    selected_nodes = feasible_nodes = 0
    for key, legal in legal_by_key.items():
        source = source_by_key[key]
        branch_state = source.get("branch_state")
        source_snapshot_id = (branch_state.get("snapshot_id")
            if isinstance(branch_state, dict) else None)
        if (legal.get("schema") != PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
                legal.get("target_schema") != PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
                source.get("source_identity_sha256") != source_identity_sha or
                legal.get("source_identity_sha256") != source_identity_sha or
                legal.get("snapshot_id") != source_snapshot_id or
                legal.get("state_sha256") != source.get("predecision_simulator_state_sha256") or
                legal.get("support_digest") != source.get("support_digest")):
            raise ValueError("legal-node row does not bind exact source replay identity/state/support")
        feasible = legal.get("feasible_node_ids")
        selected = legal.get("selected_node_ids")
        poses = legal.get("canonical_poses")
        if (not isinstance(feasible, list) or not feasible or
                any(type(node) is not int for node in feasible) or len(set(feasible)) != len(feasible) or
                not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or
                any(type(node) is not int for node in selected) or
                not set(selected) <= set(feasible) or not isinstance(poses, list)):
            raise ValueError("legal-node sampled support or canonical pose manifest is malformed")
        pose_by_node = {pose.get("node_id"): pose for pose in poses if isinstance(pose, dict)}
        if len(pose_by_node) != len(poses) or not set(feasible) <= set(pose_by_node):
            raise ValueError("canonical legal-node poses do not cover feasible nodes uniquely")
        layout = legal.get("full_pair_layout")
        if not isinstance(layout, list):
            raise ValueError("full feasible legal-node behavior layout is missing")
        if any(not isinstance(item, dict) or type(item.get("feasible")) is not bool
                for item in layout):
            raise ValueError("full feasible legal-node behavior layout contains malformed entries")
        feasible_pairs = [item for item in layout if item["feasible"]]
        if legal.get("full_pair_layout_sha256") != _canonical_sha(layout):
            raise ValueError("full feasible legal-node behavior layout digest mismatch")
        behavior_by_node = Counter()
        for index_in_layout, pair in enumerate(layout):
            if (pair.get("pair_index") != index_in_layout or
                    not isinstance(pair.get("descriptor_id"), str) or not pair["descriptor_id"]):
                raise ValueError("full pair layout indices or descriptor identities are malformed")
        for pair in feasible_pairs:
            if (type(pair.get("node_id")) is not int or pair["node_id"] not in feasible or
                    type(pair.get("behavior_id")) is not int or not 0 <= pair["behavior_id"] < 5):
                raise ValueError("full pair layout contains invalid feasible node/behavior identity")
            behavior_by_node[pair["node_id"]] += 1
            feasible_descriptor_counts[str(pair["behavior_id"])] += 1
        feasible_node_behaviors = {(pair["node_id"], pair["behavior_id"])
            for pair in feasible_pairs}
        sampled_node_behaviors = {(node, behavior) for node, behavior in feasible_node_behaviors
            if node in selected}
        for _, behavior in feasible_node_behaviors:
            feasible_behavior_counts[str(behavior)] += 1
        for _, behavior in sampled_node_behaviors:
            sampled_behavior_counts[str(behavior)] += 1
        for node in selected:
            pose = pose_by_node[node]
            heading_region[f"{pose.get('heading_bin')}:{pose.get('region_id')}"] += 1
        selected_nodes += len(selected)
        feasible_nodes += len(feasible)
        totals["branches"] += 1
        targets = legal.get("node_u_targets")
        if (not isinstance(targets, list) or any(not isinstance(item, dict) or
                type(item.get("node_id")) is not int for item in targets) or
                {item.get("node_id") for item in targets} != set(selected)):
            raise ValueError("U targets do not cover exactly the selected legal nodes")
        target_by_node = {item["node_id"]: item for item in targets}
        for node in selected:
            feasible_behaviors = target_by_node[node].get("feasible_behavior_ids")
            layout_behaviors = sorted({pair["behavior_id"] for pair in feasible_pairs
                if pair["node_id"] == node})
            if (not isinstance(feasible_behaviors, list) or
                    any(type(behavior) is not int or not 0 <= behavior < 5
                        for behavior in feasible_behaviors) or
                    sorted(set(feasible_behaviors)) != layout_behaviors):
                raise ValueError("U target behavior coverage differs from the full feasible layout")
        for target in targets:
            if (target.get("status") not in ("available", "unavailable") or
                    (target.get("status") == "available") != (target.get("u_target") is not None)):
                raise ValueError("U target status/value is inconsistent")
            if target["status"] == "available" and not _number(target["u_target"]):
                raise ValueError("available U target is not finite")
            totals["u_available_nodes" if target["status"] == "available" else "u_unavailable_nodes"] += 1
            censored = target.get("censored_feasible_behavior_ids")
            if not isinstance(censored, list) or any(type(item) is not int for item in censored):
                raise ValueError("U target lacks feasible-behavior censoring inventory")
            if (len(set(censored)) != len(censored) or
                    not set(censored) <= set(target["feasible_behavior_ids"])):
                raise ValueError("U target censoring inventory differs from feasible behaviors")
            totals["censored_feasible_behavior_count"] += len(censored)
            unavailable_behaviors = target.get("unavailable_behavior_ids")
            if (not isinstance(unavailable_behaviors, list) or
                    any(type(item) is not int for item in unavailable_behaviors) or
                    len(set(unavailable_behaviors)) != len(unavailable_behaviors) or
                    set(unavailable_behaviors) & set(target["feasible_behavior_ids"])):
                raise ValueError("U target lacks unavailable-behavior inventory")
            totals["unavailable_behavior_count"] += len(unavailable_behaviors)
            totals["u_censored_nodes" if censored else "u_uncensored_nodes"] += 1
            if target["status"] == "available" and censored:
                raise ValueError("available U target is marked censored")
            available_behaviors = target.get("available_behavior_ids")
            if (not isinstance(available_behaviors, list) or
                    any(type(item) is not int for item in available_behaviors) or
                    len(set(available_behaviors)) != len(available_behaviors) or
                    not set(available_behaviors) <= set(target["feasible_behavior_ids"]) or
                    (target["status"] == "available" and
                        set(available_behaviors) != set(target["feasible_behavior_ids"]))):
                raise ValueError("U target available-behavior inventory is inconsistent")
            pi = target.get("node_inclusion_probability")
            weight = target.get("node_inverse_probability_weight")
            if (not _number(pi) or not 0 < pi <= 1 or not _number(weight) or
                    not math.isclose(weight, 1 / pi)):
                raise ValueError("node inclusion probability/weight is invalid")
            node_weights.append(weight)
        pairs = legal.get("pairs")
        if not isinstance(pairs, list) or any(not isinstance(pair, dict) for pair in pairs):
            raise ValueError("legal-node pair outcomes are missing")
        if any(type(pair.get("node_id")) is not int or
                type(pair.get("behavior_id")) is not int for pair in pairs):
            raise ValueError("sampled descriptor outcomes have malformed node/behavior IDs")
        selected_pair_keys = [(pair.get("node_id"), pair.get("behavior_id")) for pair in pairs]
        if (len(selected_pair_keys) != len(set(selected_pair_keys)) or
                set(selected_pair_keys) != sampled_node_behaviors):
            raise ValueError("sampled descriptor outcomes do not cover each selected feasible behavior once")
        selection_digest = _canonical_sha({"protocol_id": protocol_id, "key": key,
            "support_digest": legal["support_digest"], "selected_node_ids": selected,
            "selected_pair_indices": [pair.get("pair_index") for pair in pairs]})
        if legal.get("selection_digest") != selection_digest:
            raise ValueError("sampled legal-node selection digest mismatch")
        for pair in pairs:
            pair_index = pair.get("pair_index")
            if (type(pair_index) is not int or not 0 <= pair_index < len(layout) or
                    layout[pair_index].get("feasible") is not True or
                    pair.get("node_id") not in selected or
                    pair.get("node_id") != layout[pair_index].get("node_id") or
                    pair.get("behavior_id") != layout[pair_index].get("behavior_id") or
                    pair.get("descriptor_id") != layout[pair_index].get("descriptor_id") or
                    pair.get("status") not in ("available", "unavailable")):
                raise ValueError("sampled descriptor outcome differs from exact feasible pair support")
            pi = pair.get("descriptor_inclusion_probability")
            weight = pair.get("descriptor_inverse_probability_weight")
            repeats = pair.get("repeat_returns", pair.get("repeat_outcomes"))
            if (not _number(pi) or not 0 < pi <= 1 or not _number(weight) or
                    not math.isclose(weight, 1 / pi) or not isinstance(repeats, list) or
                    len(repeats) != 2 or any(not isinstance(repeat, dict) or
                    repeat.get("status") not in ("available", "unavailable") for repeat in repeats)):
                raise ValueError("descriptor inclusion or paired-repeat provenance is invalid")
            repeats_available = all(repeat["status"] == "available" for repeat in repeats)
            if (pair["status"] == "available") != repeats_available:
                raise ValueError("descriptor outcome status differs from its paired repeats")
            descriptor_weights.append(weight)
            totals["pair_outcomes"] += 1
            for repeat in repeats:
                if repeat["status"] == "available" and not _number(repeat.get("gross_horizon_return")):
                    raise ValueError("available repeat return is not finite")
                totals["repeat_available" if repeat["status"] == "available" else "repeat_unavailable"] += 1
            behavior = pair.get("behavior_id")
            if pair.get("status") == "available":
                totals["available_behavior_outcomes"] += 1
                node_target = target_by_node[pair["node_id"]]
                variance = node_target.get("behavior_repeat_return_variances", {}).get(str(behavior))
                if not _number(variance) or variance < 0:
                    raise ValueError("available behavior outcome lacks finite repeat variance")
                per_behavior_variances[str(behavior)].append(variance)
            else:
                totals["unavailable_behavior_outcomes"] += 1
        preferences = legal.get("behavior_preferences")
        if not isinstance(preferences, list) or any(not isinstance(item, dict) for item in preferences):
            raise ValueError("B behavior preferences are missing")
        for preference in preferences:
            status = preference.get("status")
            if status not in ("comparable", "tie", "uncertain"):
                raise ValueError("B preference status is malformed")
            totals[f"b_{status}"] += 1

    def weight_summary(weights):
        if not weights:
            return {"count": 0, "sum": 0.0, "effective_sample_size": 0.0}
        total = sum(weights)
        return {"count": len(weights), "sum": total,
            "effective_sample_size": total * total / sum(weight * weight for weight in weights)}

    variance_summary = {behavior: (None if not values else {
        "count": len(values), "mean": sum(values) / len(values),
        "maximum": max(values)}) for behavior, values in per_behavior_variances.items()}
    return {"schema": "meco-plan-a-legal-node-oracle-audit-v1",
        "provenance": {"legal_node_dataset": str(dataset), "legal_node_dataset_sha256": legal_sha,
            "legal_node_manifest": str(legal_manifest_path), "legal_node_manifest_sha256": legal_manifest_sha,
            "source_dataset": str(source_dataset), "source_dataset_sha256": source_sha,
            "source_manifest": str(source_manifest_path), "source_manifest_sha256": source_manifest_sha,
            "qualification_protocol": str(protocol_path), "qualification_protocol_id": protocol_id,
            "qualification_protocol_sha256": protocol_sha,
            "source_identity_sha256": source_identity_sha, "split": "train",
            "scenario_id": scenario, "category": category,
            "device_identity": legal_manifest["device"]},
        "coverage": {"branch_count": totals["branches"], "sampled_legal_nodes": selected_nodes,
            "feasible_legal_nodes": feasible_nodes,
            "sampled_node_coverage_fraction": selected_nodes / feasible_nodes,
            "sampled_heading_region_counts": dict(sorted(heading_region.items())),
            "feasible_node_behavior_counts": dict(sorted(feasible_behavior_counts.items())),
            "sampled_node_behavior_counts": dict(sorted(sampled_behavior_counts.items())),
            "feasible_behavior_descriptor_pair_counts": dict(sorted(feasible_descriptor_counts.items()))},
        "u_summary": {"available_node_count": totals["u_available_nodes"],
            "unavailable_node_count": totals["u_unavailable_nodes"],
            "uncensored_node_count": totals["u_uncensored_nodes"],
            "censored_node_count": totals["u_censored_nodes"],
            "censored_feasible_behavior_count": totals["censored_feasible_behavior_count"],
            "unavailable_behavior_count": totals["unavailable_behavior_count"],
            "available_behavior_outcome_count": totals["available_behavior_outcomes"],
            "unavailable_behavior_outcome_count": totals["unavailable_behavior_outcomes"],
            "repeat_available_count": totals["repeat_available"],
            "repeat_unavailable_count": totals["repeat_unavailable"],
            "per_behavior_repeat_return_variance": variance_summary},
        "b_summary": {"preference_count": sum(totals[f"b_{key}"] for key in
            ("comparable", "tie", "uncertain")), "comparable_count": totals["b_comparable"],
            "tie_count": totals["b_tie"], "uncertain_count": totals["b_uncertain"]},
        "sampling_weights": {"node": weight_summary(node_weights),
            "descriptor": weight_summary(descriptor_weights)},
        "evidence_status": {"replay_validated": False,
            "empirical_oracle_established": False, "warm_start_artifact_produced": False}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--qualification-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write report JSON here; default is stdout")
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            protected = {Path(value).resolve() for value in (
                args.dataset, args.source_dataset, args.qualification_protocol)}
            protected.update(path.with_suffix(path.suffix + ".manifest.json")
                for path in (args.dataset, args.source_dataset))
            if args.output.resolve() in protected:
                raise ValueError("audit report output must not overwrite an input artifact")
        report = audit_plan_a_legal_nodes(args.dataset, args.source_dataset,
            args.qualification_protocol)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        else:
            sys.stdout.write(encoded)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
