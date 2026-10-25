"""Offline corpus coverage audit for source-bound Plan C repeated labels.

This summarizes the existing sampled candidate set and its repeated-return
labels. It does not estimate data sufficiency, full-action recall, calibration,
warm-start readiness, or simulator performance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .curriculum_training import source_identity
from .meco_supervised import _read_counterfactual_dataset
from .meco_counterfactual import PLAN_C_REPEAT_TARGET_SCHEMA

REPORT_SCHEMA = "meco-plan-c-repeated-label-corpus-audit-v1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(value) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode())


def audit_dataset(dataset: Path, *, qualification_protocol: Path) -> dict:
    """Validate and summarize one frozen train-split repeated-label dataset."""
    dataset = Path(dataset).resolve()
    protocol_path = Path(qualification_protocol).resolve()
    rows, dataset_sha, manifest_sha, _protocol, protocol_id, protocol_sha, manifest = \
        _read_counterfactual_dataset(dataset, qualification_protocol=protocol_path,
                                     expected_sources=source_identity())
    if (manifest.get("split") != "train" or
            manifest.get("plan_c_target_schema") != PLAN_C_REPEAT_TARGET_SCHEMA):
        raise ValueError("Plan C corpus audit requires repeated-label train-split data")

    candidate_count = pair_count = stable_count = tied_count = mixed_count = 0
    action_counts: dict[str, int] = {}
    skipped_by_reason: dict[str, int] = {}
    skipped_count = 0
    feasible_prefixes = len(rows)
    for row in rows:
        outcomes = row["plan_c_counterfactuals"]
        labels = row["plan_c_pairwise_labels"]
        candidate_count += len(outcomes)
        pair_count += len(labels)
        for outcome in outcomes:
            action = outcome["descriptor"].get("action")
            key = str(action) if type(action) is int else "unknown"
            action_counts[key] = action_counts.get(key, 0) + 1
        for label in labels:
            if label["stable_preference"]:
                stable_count += 1
            elif any(value == 0 for value in label["repeat_differences"]):
                tied_count += 1
            else:
                mixed_count += 1
    for summary in manifest["seed_branch_summaries"].values():
        skipped_count += summary["skipped_count"]
        for reason, count in summary["skipped_reason_counts"].items():
            skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + count
    report = {
        "schema": REPORT_SCHEMA,
        "audit_identity": None,
        "dataset": {
            "file": dataset.name,
            "dataset_sha256": dataset_sha,
            "manifest_sha256": manifest_sha,
            "record_count": len(rows),
            "protocol_id": protocol_id,
            "qualification_protocol_sha256": protocol_sha,
            "protocol_file": protocol_path.name,
            "split": "train",
            "scenario_id": manifest["scenario_id"],
            "category": manifest["category"],
            "source_identity_sha256": manifest["source_identity_sha256"],
            "source_runtime_device_bound": manifest["source_runtime_device_bound"],
            "plan_c_repeat_protocol_id": manifest["plan_c_repeat_protocol"]["repeat_protocol_id"],
        },
        "coverage": {
            "prefix_count": len(rows),
            "feasible_prefix_count": feasible_prefixes,
            "skipped_branch_count": skipped_count,
            "skipped_branch_counts_by_reason": dict(sorted(skipped_by_reason.items())),
            "candidate_count": candidate_count,
            "candidate_count_per_prefix": {
                "minimum": min(len(row["plan_c_counterfactuals"]) for row in rows),
                "maximum": max(len(row["plan_c_counterfactuals"]) for row in rows),
            },
            "pair_comparison_count": pair_count,
            "pair_label_counts": {
                "stable_preference": stable_count,
                "tied_or_zero_repeat_difference": tied_count,
                "mixed_sign": mixed_count,
            },
            "candidate_descriptor_action_counts": dict(sorted(action_counts.items())),
            "categories": {manifest["category"]: len(rows)},
        },
        "limits": [
            "Counts describe only archived protocol-bound feasible train prefixes and sampled Plan C candidates.",
            "Candidate descriptor action coverage is not recall over the full action space.",
            "Two matched repeats support stability abstention, not precise variance calibration.",
            "This audit does not establish data sufficiency, warm-start readiness, or simulator outcomes.",
        ],
    }
    report["audit_identity"] = _identity(
        {key: value for key, value in report.items() if key != "audit_identity"})
    return report
