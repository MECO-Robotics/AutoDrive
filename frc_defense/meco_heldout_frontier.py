"""Strict offline compiler and verifier for the provisional MECO held-out frontier.

The frontier is an immutable validation-input lock. It is not an evaluation result
and it never qualifies a model or exposes final-test seeds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .meco_tracks import MECO_TRACKS
from .meco_validation_campaign import (
    _canonical_bytes,
    _sha,
    _validate_sha,
    compile_validation_campaign,
)


FRONTIER_SCHEMA = "meco-heldout-frontier-v1-provisional"
_BASELINE_POLICIES = ("D0", "D1", "D2")
_NEURAL_CONTROLS = ("A1_P0", "A2_P0", "A3_P0")
_DISABLED = tuple(f"{track}__INFERENCE_DISABLED_TO_D1" for track in MECO_TRACKS)


def _source_binding(source_identity: dict) -> dict:
    if not isinstance(source_identity, dict) or not source_identity:
        raise ValueError("frontier requires a nonempty canonical source identity")
    raw = json.dumps(source_identity, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return {"identity": source_identity, "sha256": hashlib.sha256(raw).hexdigest()}


def _require_five_hashes(values: dict, label: str) -> dict[str, str]:
    if not isinstance(values, dict) or set(values) != set(MECO_TRACKS):
        raise ValueError(f"frontier requires {label} for all five tracks")
    return {track: _validate_sha(values[track], f"{track} {label}")
            for track in MECO_TRACKS}


def _contrast_roster() -> list[dict[str, str]]:
    """Plan §§3/6/7 roster, limited to registered schema-7 matrix factors."""
    rows = [
        {"id": f"architecture:{track}:vs:D0", "condition": "architecture_matched_cv_k16",
         "policies": f"{track},D0", "plan_refs": "§§3,6"} for track in MECO_TRACKS
    ]
    rows += [
        {"id": "architecture:A1:vs:C2", "condition": "architecture_matched_cv_k16",
         "policies": "A1,C2", "plan_refs": "§§6,7"},
        {"id": "architecture:C1:vs:C2", "condition": "architecture_matched_cv_k16",
         "policies": "C1,C2", "plan_refs": "§§6,7"},
    ]
    rows += [
        {"id": f"architecture:{left}:vs:{right}", "condition": "architecture_matched_cv_k16",
         "policies": f"{left},{right}", "plan_refs": "§§3,6,7"}
        for left, right in (("A1", "A2"), ("A1", "A3"), ("A2", "A3"))
    ]
    rows += [
        {"id": "baseline:D0:vs:D1", "condition": "controls_default_d1_current_d2_cv_k16",
         "policies": "D0,D1", "plan_refs": "§§3,6"},
        {"id": "prediction:D1:vs:D2", "condition": "controls_default_d1_current_d2_cv_k16",
         "policies": "D1,D2", "plan_refs": "§§3,6"},
    ]
    rows += [
        {"id": f"candidate-utility:{track}:vs:D1", "condition": "controls_default_d1_current_d2_cv_k16",
         "policies": f"{track},D1", "plan_refs": "§§3,6"} for track in ("C1", "C2")
    ]
    rows += [
        {"id": f"prediction:{track}:current:vs:cv", "condition": "matched_prediction_current_k16",
         "policies": track, "plan_refs": "§§6,7"} for track in MECO_TRACKS
    ]
    rows += [
        {"id": f"candidate-recall:{track}:K{cap}", "condition": (
             "architecture_matched_cv_k16" if cap == 16 else f"candidate_recall_k{cap}_cv"),
         "policies": track, "plan_refs": "§§3,6,7"}
        for track in ("C1", "C2") for cap in (8, 16, 32)
    ]
    rows += [
        {"id": f"penalty:P0:{track}", "condition": "architecture_matched_cv_k16",
         "policies": f"{track}_P0", "plan_refs": "§§3,6,7"} for track in ("A1", "A2", "A3")
    ]
    rows += [
        {"id": f"fallback:{track}:to:D1", "condition": "inference_disabled_fallback_current_k16",
         "policies": f"{track}__INFERENCE_DISABLED_TO_D1", "plan_refs": "§§6,7"} for track in MECO_TRACKS
    ]
    return rows


def _validate_campaign_roster(campaign: dict) -> None:
    if campaign.get("split") != "validation":
        raise ValueError("held-out frontier only accepts validation split")
    if campaign.get("final_test_seed_handling") != "not_exported":
        raise ValueError("final-test seeds must remain unexported")
    if campaign.get("qualification_claim") is not False or campaign.get("qualification_eligible") is not False:
        raise ValueError("provisional frontier cannot make or enable qualification claims")
    if campaign.get("job_count") != len(campaign.get("jobs", [])) or not campaign.get("jobs"):
        raise ValueError("validation campaign has missing jobs")

    coverage_by_condition: dict[str, dict[str, set]] = {}
    for job in campaign["jobs"]:
        if job.get("seed_manifest", {}).get("split") != "validation":
            raise ValueError("frontier contains a non-validation seed manifest")
        manifest_bytes = (json.dumps(job["seed_manifest"], sort_keys=True, indent=2,
                                     allow_nan=False) + "\n").encode()
        if hashlib.sha256(manifest_bytes).hexdigest() != job.get("seed_manifest_sha256"):
            raise ValueError("frontier seed manifest hash mismatch")
        for condition in job.get("conditions", []):
            name = condition.get("condition_name")
            expected = condition.get("expected_coverage")
            if not name or not isinstance(expected, dict):
                raise ValueError("frontier condition lacks expected coverage")
            policies = expected.get("policies")
            categories = expected.get("categories")
            seeds = expected.get("seed_list")
            keys = sorted((policy, category, seed) for policy in policies
                          for category in categories for seed in seeds)
            if (len(keys) != expected.get("record_count") or
                    hashlib.sha256(_canonical_bytes([list(key) for key in keys])).hexdigest()
                    != expected.get("policy_category_seed_keys_sha256")):
                raise ValueError("frontier policy/category/seed coverage digest mismatch")
            state = coverage_by_condition.setdefault(name, {"policies": set(), "categories": set(), "seeds": set()})
            state["policies"].update(policies)
            state["categories"].update(categories)
            state["seeds"].update(seeds)

    expected_conditions = {
        "architecture_matched_cv_k16", "controls_default_d1_current_d2_cv_k16",
        "matched_prediction_current_k16", "candidate_recall_k8_cv",
        "candidate_recall_k32_cv", "inference_disabled_fallback_current_k16",
    }
    if not expected_conditions.issubset(coverage_by_condition):
        raise ValueError("frontier is missing a required registered contrast condition")
    expected_policy_sets = {
        "architecture_matched_cv_k16": set(MECO_TRACKS) | set(_BASELINE_POLICIES) | set(_NEURAL_CONTROLS),
        "controls_default_d1_current_d2_cv_k16": set(MECO_TRACKS) | set(_BASELINE_POLICIES) | set(_NEURAL_CONTROLS),
        "matched_prediction_current_k16": set(MECO_TRACKS) | set(_BASELINE_POLICIES) | set(_NEURAL_CONTROLS),
        "candidate_recall_k8_cv": set(MECO_TRACKS) | set(_BASELINE_POLICIES) | set(_NEURAL_CONTROLS),
        "candidate_recall_k32_cv": set(MECO_TRACKS) | set(_BASELINE_POLICIES) | set(_NEURAL_CONTROLS),
        "inference_disabled_fallback_current_k16": set(MECO_TRACKS) | set(_BASELINE_POLICIES) | set(_NEURAL_CONTROLS) | set(_DISABLED),
    }
    for condition, required in expected_policy_sets.items():
        if not required.issubset(coverage_by_condition[condition]["policies"]):
            raise ValueError(f"frontier condition {condition} lacks required track/control coverage")
    for condition in expected_conditions:
        if not coverage_by_condition[condition]["categories"] or not coverage_by_condition[condition]["seeds"]:
            raise ValueError(f"frontier condition {condition} has incomplete category/seed coverage")

    roster = _contrast_roster()
    available = {condition: coverage["policies"]
                 for condition, coverage in coverage_by_condition.items()}
    for contrast in roster:
        wanted = set(contrast["policies"].split(","))
        if not wanted.issubset(available.get(contrast["condition"], set())):
            raise ValueError(f"frontier is missing required contrast {contrast['id']}")


def compile_heldout_frontier(*, protocol_path: str | Path,
        checkpoint_paths: dict[str, str | Path], source_identity: dict,
        training_configuration_sha256: dict[str, str]) -> dict:
    """Bind complete validation inputs into one provisional immutable root object."""
    if not isinstance(checkpoint_paths, dict) or set(checkpoint_paths) != set(MECO_TRACKS):
        raise ValueError("frontier requires checkpoint files for all five tracks")
    checkpoints = _require_five_hashes({track: hashlib.sha256(Path(checkpoint_paths[track]).read_bytes()).hexdigest()
                                        for track in checkpoint_paths}, "checkpoint hash")
    configs = _require_five_hashes(training_configuration_sha256, "configuration hash")
    source = _source_binding(source_identity)
    campaign = compile_validation_campaign(protocol_path=protocol_path,
        checkpoint_paths=checkpoint_paths, source_identity=source_identity,
        training_configuration_sha256=configs, qualification_claim=False)
    _validate_campaign_roster(campaign)
    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol_raw = protocol_file.read_bytes()
    protocol = json.loads(protocol_raw)
    root = {
        "schema": FRONTIER_SCHEMA,
        "version_status": "provisional",
        "qualification_claim": False,
        "qualification_eligible": False,
        "split": "validation",
        "protocol": {
            "raw_sha256": hashlib.sha256(protocol_raw).hexdigest(),
            "protocol_id": protocol.get("protocol_id"),
            "protocol_version": protocol.get("protocol_version"),
            "gate_calibration_status": protocol.get("gate_calibration_status"),
            "safety_gate_status": protocol.get("safety_gate_status"),
        },
        "source": source,
        "checkpoints_sha256": checkpoints,
        "training_configuration_sha256": configs,
        "campaign_plan_digest": campaign["plan_digest"],
        "seed_manifests": [
            {"file": job["seed_manifest_file"], "sha256": job["seed_manifest_sha256"],
             "scenario_id": job["scenario_id"],
             "coverage": [condition["expected_coverage"] for condition in job["conditions"]]}
            for job in campaign["jobs"]
        ],
        "contrast_roster": _contrast_roster(),
        "unsupported_optional_contrasts": [{"id": "D3", "status": "not_implemented",
            "reason": "optional learned opponent prediction is not registered in schema-7 matrix"}],
        "final_test_seed_handling": "not_exported",
    }
    _validate_frontier_shape(root)
    root["frontier_sha256"] = _sha(root)
    return root


def _validate_frontier_shape(frontier: dict) -> None:
    if not isinstance(frontier, dict) or frontier.get("schema") != FRONTIER_SCHEMA:
        raise ValueError("unsupported held-out frontier schema")
    if frontier.get("version_status") != "provisional" or frontier.get("split") != "validation":
        raise ValueError("frontier must be provisional and validation-only")
    if frontier.get("qualification_claim") is not False or frontier.get("qualification_eligible") is not False:
        raise ValueError("frontier qualification fields must remain false")
    if frontier.get("final_test_seed_handling") != "not_exported":
        raise ValueError("frontier must not export final-test seeds")
    if frontier.get("contrast_roster") != _contrast_roster():
        raise ValueError("frontier contrast roster differs from the registered plan roster")
    if frontier.get("unsupported_optional_contrasts") != [{"id": "D3", "status": "not_implemented",
            "reason": "optional learned opponent prediction is not registered in schema-7 matrix"}]:
        raise ValueError("optional D3 contrast must remain explicitly unsupported")
    protocol = frontier.get("protocol")
    if not isinstance(protocol, dict) or set(protocol) != {
        "raw_sha256", "protocol_id", "protocol_version", "gate_calibration_status", "safety_gate_status"
    }:
        raise ValueError("frontier protocol binding is incomplete")
    _validate_sha(protocol.get("raw_sha256"), "protocol raw hash")
    for label in ("protocol_id",):
        if not isinstance(protocol.get(label), str) or not protocol[label]:
            raise ValueError(f"frontier {label} is missing")
    if not isinstance(frontier.get("source"), dict) or set(frontier["source"]) != {"identity", "sha256"}:
        raise ValueError("frontier source binding is incomplete")
    _validate_sha(frontier["source"].get("sha256"), "source identity hash")
    _require_five_hashes(frontier.get("checkpoints_sha256"), "checkpoint hash")
    _require_five_hashes(frontier.get("training_configuration_sha256"), "configuration hash")
    if not isinstance(frontier.get("campaign_plan_digest"), str):
        raise ValueError("frontier campaign plan digest is missing")
    _validate_sha(frontier["campaign_plan_digest"], "campaign plan digest")
    manifests = frontier.get("seed_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("frontier validation seed manifest bindings are missing")
    for manifest in manifests:
        if not isinstance(manifest, dict) or not isinstance(manifest.get("file"), str):
            raise ValueError("frontier contains a malformed seed manifest binding")
        _validate_sha(manifest.get("sha256"), "seed manifest hash")
        if not manifest.get("coverage"):
            raise ValueError("frontier seed manifest lacks expected coverage digests")


def validate_heldout_frontier(frontier: dict, *, protocol_path: str | Path,
        checkpoint_paths: dict[str, str | Path], source_identity: dict,
        training_configuration_sha256: dict[str, str]) -> None:
    """Verify exact current bindings by deterministically rebuilding the frontier."""
    _validate_frontier_shape(frontier)
    observed_digest = frontier.get("frontier_sha256")
    _validate_sha(observed_digest, "frontier digest")
    if observed_digest != _sha({key: value for key, value in frontier.items()
                                if key != "frontier_sha256"}):
        raise ValueError("frontier root digest mismatch")
    rebuilt = compile_heldout_frontier(protocol_path=protocol_path,
        checkpoint_paths=checkpoint_paths, source_identity=source_identity,
        training_configuration_sha256=training_configuration_sha256)
    if rebuilt != frontier:
        raise ValueError("current protocol/source/checkpoint/configuration/coverage differs from frozen frontier")
