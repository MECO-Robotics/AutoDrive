import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from frc_defense.meco_phase4 import PHASE4_FACTORS, phase4_factor_digest
from frc_defense.meco_ablation_matrix import POLICY_CONTRASTS
from frc_defense.meco_tracks import MECO_TRACKS

from scripts.summarize_meco_ablation_matrix import (
    PHASE4_POLICIES, _canonical_hash, _cluster_bootstrap_ci,
    _content_bound_seed, _equal_category_summaries, _paired_finite_values,
    _resolve_conditions, _stratified_paired_ci, phase4_teammate_behavior_record,
    summarize_matrix,
    _validate_required_contrasts, summarize_matrices,
)


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _standard_contrasts(common, records, condition_id):
    provenance = {**common, "scenario_id": "pickup_simple"}
    rows = []
    pairs = [(name, candidate, reference)
        for name, policy_pairs in POLICY_CONTRASTS.items()
        for candidate, reference in policy_pairs]
    pairs.extend((f"{track}_vs_D0_architecture_condition", track, "D0")
                 for track in MECO_TRACKS)
    for name, candidate_policy, reference_policy in pairs:
        contrast = {"contrast": name, "paired": True,
            "condition_id": condition_id, "candidate_policy": candidate_policy,
            "reference_policy": reference_policy,
            "common_bundle_provenance": dict(provenance), "paired_episode_rows": []}
        for category, seed in (("single", seed) for seed in (10, 11, 12, 13)):
            pair = {(row["policy"], row["category"], row["seed"]): row
                    for row in records}
            candidate = pair.get((candidate_policy, category, seed))
            reference = pair.get((reference_policy, category, seed))
            if candidate is None or reference is None:
                continue
            contrast["paired_episode_rows"].append({"category": category, "seed": seed,
                "candidate_policy": candidate_policy, "reference_policy": reference_policy,
                "candidate_record_policy_id": candidate["policy_id"],
                "reference_record_policy_id": reference["policy_id"],
                "candidate_metrics": candidate["metrics"],
                "reference_metrics": reference["metrics"]})
        rows.append(contrast)
    return rows


def _contrast_catalog_rows():
    rows = [{"contrast": name, "candidate_policy": candidate,
             "reference_policy": reference}
        for name, pairs in POLICY_CONTRASTS.items()
        for candidate, reference in pairs]
    rows.extend({"contrast": f"{track}_vs_D0_architecture_condition",
                 "candidate_policy": track, "reference_policy": "D0"}
                for track in MECO_TRACKS)
    return rows


@pytest.mark.parametrize("omitted_name", [*POLICY_CONTRASTS,
    *(f"{track}_vs_D0_architecture_condition" for track in MECO_TRACKS)])
def test_rejects_rehashed_matrix_missing_whole_required_contrast(tmp_path, omitted_name):
    path = _fixture(tmp_path)
    matrix = json.loads(path.read_text())
    matrix["contrasts"] = [row for row in matrix["contrasts"]
                           if row["contrast"] != omitted_name]
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    _write_json(path, matrix)
    with pytest.raises(ValueError, match="missing required contrast pairs"):
        summarize_matrix(path, replicates=100)


def test_rejects_matrix_missing_one_a_track_pairwise_contrast():
    rows = [row for row in _contrast_catalog_rows()
            if (row["contrast"], row["candidate_policy"], row["reference_policy"])
            != ("A1_A2_A3", "A2", "A3")]
    with pytest.raises(ValueError, match="missing required contrast pairs"):
        _validate_required_contrasts(rows)


def test_rejects_unsupported_pair_for_named_contrast():
    rows = _contrast_catalog_rows()
    rows.append({"contrast": "A1_A2_A3", "candidate_policy": "A1",
                 "reference_policy": "D1"})
    with pytest.raises(ValueError, match="unsupported named contrast pair"):
        _validate_required_contrasts(rows)


def _fixture(tmp_path, *, duplicate=False, missing=False, bad_provenance=False,
             schema_version=4):
    case = {"layout_id": "layout-x", "defender_family": "family-x",
            "start_condition": "start-x", "sensor_profile": "sensor-x"}
    seeds = [10, 11, 12, 13]
    policies = (*MECO_TRACKS, "D0", "D1", "D2")
    hashes = {policy: hashlib.sha256(policy.encode()).hexdigest() for policy in MECO_TRACKS}
    configs = {policy: hashlib.sha256((policy + "config").encode()).hexdigest()
               for policy in MECO_TRACKS}
    protocols = {policy: "train-protocol" for policy in MECO_TRACKS}
    spec = {"condition_name": "architecture_matched_cv_k16"}
    qualification_protocol = {"protocol_id": "qual-protocol",
        "protocol_sha256": "qual-protocol-sha", "protocol_file": "synthetic-v2.json"}
    records = []
    for seed, offset in zip(seeds, (1.0, 2.0, 3.0, 4.0)):
        cluster = "layout-x"
        for policy in policies:
            score_delta = {"A1": .5, "A2": .4, "A3": .3, "C1": .2}.get(policy, 0.0)
            records.append({"policy": policy, "policy_id": f"id-{policy}",
                "category": "single", "seed": seed,
                "layout_id": cluster, "defender_family": "family-x",
                "start_condition": "start-x", "sensor_profile": "sensor-x",
                "metrics": {"score": offset + score_delta,
                            "fuel": 5.0}, "candidate_recall": offset / 10})
    if duplicate:
        records.append(dict(records[0]))
    if missing:
        records = [r for r in records if not (r["policy"] == "C2" and r["seed"] == 13)]
    manifest = {policy: {"policy_id": f"id-{policy}",
        "checkpoint_sha256": hashes[policy],
        "training_configuration_sha256": configs[policy],
        "training_protocol_id": protocols[policy],
        "source_identity_sha256": "source-hash"} for policy in MECO_TRACKS}
    manifest.update({policy: {"policy_id": f"id-{policy}"}
                     for policy in ("D0", "D1", "D2")})
    report = {"schema_version": 1, "evaluation_id": "eval-one", "protocol_id": "eval-protocol",
        "qualification_protocol_id": "qual-protocol",
        "evaluation_protocol_id": "qual-protocol",
        "evaluation_protocol_sha256": "qual-protocol-sha",
        "evaluation_case": {"protocol_id": "qual-protocol",
            "protocol_sha256": "qual-protocol-sha", "protocol_file": "synthetic-v2.json",
            "split": "test", "scenario_id": "pickup_simple", "categories": ["single"],
            "qualification_case": case, "seeds": seeds,
            "case_ids": [{"category": "single", **case}]},
        "exact_episode_seeds": seeds, "evaluation_design": "frozen_protocol_held_out",
        "split": "test",
        "source_identity_sha256": "source-hash", "scenario_id": "pickup_simple",
        "categories": ["single"], "seeds": seeds,
        "qualification_case": case,
        "execution_device": {"device": "cuda:1", "accelerator_backend": "rocm"},
        "candidate_k": {"effective": 16}, "policy_manifest": manifest, "records": records,
        "experiment_matrix": {"matrix_id": "matrix-id", "condition_id": "condition-id",
            "condition": spec, "evaluation_design": "frozen_protocol_held_out",
            "evaluation_split": "test", "qualification_protocol_id": "qual-protocol",
            "qualification_protocol": qualification_protocol,
            "seed_split_assignments": {"single": {str(seed): "test" for seed in seeds}}}}
    report_path = tmp_path / "condition.json"
    _write_json(report_path, report)
    condition = {"condition_id": "condition-id", "evaluation_id": "eval-one",
        "protocol_id": "eval-protocol", "report_path": str(report_path),
        "checkpoint_sha256": hashes,
        "source_identity_sha256": "source-hash", "qualification_case": case,
        "execution_device": {"device": "cuda:1", "accelerator_backend": "rocm"},
        "qualification_protocol": qualification_protocol, "policy_manifest": manifest, "condition": spec}
    common = {"checkpoint_sha256": hashes,
        "training_configuration_sha256": configs,
        "training_protocol_ids": protocols,
        "source_identity_sha256": "source-hash",
        "execution_device": {"device": "cuda:1", "accelerator_backend": "rocm"},
        "evaluation_design": "frozen_protocol_held_out", "evaluation_split": "test",
        "qualification_protocol_id": "qual-protocol", "qualification_protocol": qualification_protocol,
        "qualification_case": case,
        "seed_split_assignments": {"single": {str(seed): "test" for seed in seeds}}}
    contrasts = _standard_contrasts(common, records, "condition-id")
    if schema_version >= 5:
        condition["evaluation_report_sha256"] = hashlib.sha256(
            report_path.read_bytes()).hexdigest()
    matrix_protocol = {"schema_version": schema_version, "matrix_id": "matrix-id", "scenario_id": "pickup_simple",
        "categories": ["single"], "seed_list": seeds,
        "conditions": [spec], **common}
    matrix = {"schema_version": schema_version, "matrix_id": "matrix-id", "protocol": matrix_protocol,
        "conditions": {"architecture_matched_cv_k16": condition},
        "contrasts": contrasts, **common}
    recall = {"contrast": "candidate_recall_K_sweep", "paired": True,
        "candidate_policy": "A1@K=16", "reference_policy": "expanded_route_qualified_candidate_pool",
        "condition_id": "condition-id", "common_bundle_provenance": dict(common)}
    recall["common_bundle_provenance"]["scenario_id"] = "pickup_simple"
    matrix["contrasts"].extend({**recall, "category": "single", "seed": seed,
        "candidate_recall": next(r["candidate_recall"] for r in records
                                  if r["policy"] == "A1" and r["seed"] == seed)}
        for seed in seeds)
    if bad_provenance:
        matrix["contrasts"][0]["common_bundle_provenance"]["source_identity_sha256"] = "other-source"
    matrix.pop("report_sha256", None)
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    matrix_path = tmp_path / "matrix.json"
    _write_json(matrix_path, matrix)
    return matrix_path


def _second_stratum(path, *, layout="layout-y"):
    matrix = json.loads(path.read_text())
    matrix_id = "matrix-id-" + layout
    condition_id = "condition-id-" + layout
    report_path = path.parent / f"condition-{layout}.json"
    old_condition = matrix["conditions"]["architecture_matched_cv_k16"]
    report = json.loads(Path(old_condition["report_path"]).read_text())
    case = dict(matrix["qualification_case"])
    case["layout_id"] = layout
    for row in report["records"]:
        row["layout_id"] = layout
    report["qualification_case"] = case
    report["evaluation_case"]["qualification_case"] = case
    report["evaluation_case"]["case_ids"] = [{"category": "single", **case}]
    report["experiment_matrix"].update({"matrix_id": matrix_id,
                                        "condition_id": condition_id})
    _write_json(report_path, report)
    condition = dict(old_condition)
    condition.update({"condition_id": condition_id, "report_path": str(report_path),
                      "qualification_case": case})
    matrix["conditions"]["architecture_matched_cv_k16"] = condition
    matrix.update({"matrix_id": matrix_id, "qualification_case": case})
    matrix["protocol"].update({"matrix_id": matrix_id, "qualification_case": case})
    matrix["contrasts"] = [dict(row) for row in matrix["contrasts"]]
    for row in matrix["contrasts"]:
        row["condition_id"] = condition_id
        row["common_bundle_provenance"] = dict(row["common_bundle_provenance"])
        row["common_bundle_provenance"]["qualification_case"] = case
        for pair in row.get("paired_episode_rows", []):
            pair["candidate_metrics"] = dict(pair["candidate_metrics"])
            pair["reference_metrics"] = dict(pair["reference_metrics"])
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    output = path.parent / f"matrix-{layout}.json"
    _write_json(output, matrix)
    return output


def _reuse_condition_id(path, condition_id="condition-id"):
    matrix = json.loads(path.read_text())
    name, condition = next(iter(matrix["conditions"].items()))
    report_path = Path(condition["report_path"])
    report = json.loads(report_path.read_text())
    old_id = condition["condition_id"]
    condition["condition_id"] = condition_id
    report["experiment_matrix"]["condition_id"] = condition_id
    _write_json(report_path, report)
    for contrast in matrix["contrasts"]:
        if contrast.get("condition_id") == old_id:
            contrast["condition_id"] = condition_id
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    _write_json(path, matrix)


def _matched_prediction_fixture(tmp_path):
    path = _fixture(tmp_path)
    matrix = json.loads(path.read_text())
    matrix_id = "matched-matrix"
    specs = [
        {"condition_name": "matched_prediction_current_k16", "prediction_mode": "current"},
        {"condition_name": "architecture_matched_cv_k16", "prediction_mode": "cv"},
    ]
    reports = {}
    conditions = {}
    for name, condition_id, delta in ((specs[0]["condition_name"], "current-id", 0.0),
                                      (specs[1]["condition_name"], "cv-id", 2.0)):
        old = matrix["conditions"]["architecture_matched_cv_k16"]
        report = json.loads(Path(old["report_path"]).read_text())
        report["evaluation_id"] = f"eval-{condition_id}"
        spec = next(x for x in specs if x["condition_name"] == name)
        report["experiment_matrix"].update({"matrix_id": matrix_id,
            "condition_id": condition_id, "condition": spec})
        for row in report["records"]:
            row["metrics"]["score"] += delta
        output_path = tmp_path / f"{condition_id}.json"
        _write_json(output_path, report)
        condition = dict(old)
        condition.update({"condition_id": condition_id,
            "evaluation_id": report["evaluation_id"], "report_path": str(output_path)})
        conditions[name] = condition
        reports[name] = report
    matrix["matrix_id"] = matrix_id
    matrix["protocol"].update({"matrix_id": matrix_id, "conditions": specs})
    matrix["conditions"] = conditions
    provenance = {key: matrix.get(key) for key in (
        "checkpoint_sha256", "training_configuration_sha256", "training_protocol_ids",
        "source_identity_sha256", "execution_device", "evaluation_design",
        "evaluation_split", "qualification_case", "qualification_protocol_id",
        "qualification_protocol", "seed_split_assignments")}
    matrix["contrasts"] = _standard_contrasts(provenance,
        reports[specs[1]["condition_name"]]["records"], "cv-id")
    for policy in MECO_TRACKS:
        for seed in [10, 11, 12, 13]:
            current = next(row for row in reports[specs[0]["condition_name"]]["records"]
                           if row["policy"] == policy and row["seed"] == seed)
            cv = next(row for row in reports[specs[1]["condition_name"]]["records"]
                      if row["policy"] == policy and row["seed"] == seed)
            matched_provenance = {**provenance, "scenario_id": "pickup_simple"}
            matrix["contrasts"].append({"contrast": "matched_prediction_current_vs_cv",
                "paired": True, "candidate_policy": f"{policy}@cv",
                "reference_policy": f"{policy}@current",
                "condition_ids": {"current": "current-id", "cv": "cv-id"},
                "category": "single", "seed": seed,
                "current_metrics": current["metrics"], "cv_metrics": cv["metrics"],
                "common_bundle_provenance": matched_provenance})
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    output = tmp_path / "matched-matrix.json"
    _write_json(output, matrix)
    return output


def _phase4_fixture(tmp_path):
    path = _fixture(tmp_path)
    matrix = json.loads(path.read_text())
    matrix_id = matrix["matrix_id"]
    baseline = matrix["conditions"]["architecture_matched_cv_k16"]
    baseline_report = json.loads(Path(baseline["report_path"]).read_text())
    for policy in PHASE4_POLICIES:
        if policy not in baseline_report["policy_manifest"]:
            baseline_report["policy_manifest"][policy] = {"policy_id": f"id-{policy}"}
        if not any(row["policy"] == policy for row in baseline_report["records"]):
            for seed in baseline_report["seeds"]:
                baseline_report["records"].append({"policy": policy,
                    "policy_id": f"id-{policy}", "category": "single", "seed": seed,
                    "layout_id": "layout-x", "defender_family": "family-x",
                    "start_condition": "start-x", "sensor_profile": "sensor-x",
                    "metrics": {"score": seed / 10.0, "fuel": 5.0}})
    _write_json(Path(baseline["report_path"]), baseline_report)
    baseline["policy_manifest"] = baseline_report["policy_manifest"]
    base_spec = matrix["protocol"]["conditions"][0]
    common = dict(matrix["contrasts"][0]["common_bundle_provenance"])
    for factor, delta in (("feinting_opponent", 1.0),
                          ("observation_delay", 3.0)):
        condition_name = f"phase4_{factor}"
        condition_id = f"condition-{factor}"
        factor_spec = PHASE4_FACTORS[factor]
        spec = {**base_spec, "condition_name": condition_name,
                "phase4_factor": factor_spec}
        report = json.loads(json.dumps(baseline_report))
        report["evaluation_id"] = f"eval-{factor}"
        report["phase4_factor"] = factor_spec
        report["experiment_matrix"].update({"condition_id": condition_id,
            "condition": spec})
        for row in report["records"]:
            if row["policy"] == "A1":
                row["metrics"]["score"] += delta
        report_path = tmp_path / f"{condition_name}.json"
        _write_json(report_path, report)
        matrix["conditions"][condition_name] = {
            **baseline, "condition_id": condition_id,
            "evaluation_id": report["evaluation_id"],
            "report_path": str(report_path), "phase4_factor": factor_spec}
        matrix["protocol"]["conditions"].append(spec)
        for policy in PHASE4_POLICIES:
            for seed in report["seeds"]:
                treated = next(row for row in report["records"]
                               if row["policy"] == policy and row["seed"] == seed)
                control = next(row for row in baseline_report["records"]
                               if row["policy"] == policy and row["seed"] == seed)
                matrix["contrasts"].append({
                    "contrast": "phase4_factor_vs_no_factor", "factor": factor,
                    "factor_sha256": phase4_factor_digest(factor_spec),
                    "candidate_policy": policy, "reference_policy": policy, "paired": True,
                    "condition_ids": {"factor": condition_id,
                                      "baseline": baseline["condition_id"]},
                    "qualification_case": matrix["qualification_case"],
                    "category": "single", "seed": seed,
                    "factor_metrics": treated["metrics"],
                    "baseline_metrics": control["metrics"],
                    "common_bundle_provenance": dict(common)})
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    output = tmp_path / "phase4-matrix.json"
    _write_json(output, matrix)
    return output


def test_known_paired_interval_is_deterministic_and_clustered(tmp_path):
    path = _fixture(tmp_path)
    one = summarize_matrix(path, bootstrap_seed=42, replicates=500)
    two = summarize_matrix(path, bootstrap_seed=42, replicates=500)
    summary = next(x for x in one["summaries"] if x["metric"] == "score")
    assert summary["mean_difference"] == pytest.approx(0.5)
    assert summary["ci_95"] == [0.5, 0.5]
    assert summary["cluster_count"] == 1
    recall = next(x for x in one["summaries"] if x["contrast"] == "candidate_recall_K_sweep")
    assert recall["candidate_k"] == 16
    assert recall["ci_95"]
    assert "candidate_minus_reference" not in recall
    assert one == two
    assert one["diagnostic_only"] is True
    assert one["promotion_inference"] is False


def test_schema5_freezes_condition_report_bytes(tmp_path):
    path = _fixture(tmp_path, schema_version=5)
    summarize_matrix(path, replicates=100)
    report_path = tmp_path / "condition.json"
    report = json.loads(report_path.read_text())
    report["records"][0]["metrics"]["score"] += 1
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="evaluation report SHA-256"):
        summarize_matrix(path, replicates=100)


def test_cluster_bootstrap_resamples_clusters_then_rows():
    groups = {("a", "family"): [0.0, 0.0], ("b", "family"): [10.0]}
    assert _cluster_bootstrap_ci(groups, seed=7, replicates=1000) == \
        _cluster_bootstrap_ci(groups, seed=7, replicates=1000)
    assert _cluster_bootstrap_ci(groups, seed=7, replicates=1000) == [0.0, 10.0]


@pytest.mark.parametrize("kwargs,match", [
    ({"duplicate": True}, "duplicate evaluation record"),
    ({"missing": True}, "missing candidate record|missing reference record|missing"),
    ({"bad_provenance": True}, "contrast common provenance mismatch"),
])
def test_fails_closed_on_duplicate_missing_or_incompatible_inputs(tmp_path, kwargs, match):
    path = _fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        summarize_matrix(path, replicates=100)


def test_fails_closed_when_matrix_digest_is_wrong(tmp_path):
    path = _fixture(tmp_path)
    value = json.loads(path.read_text())
    value["matrix_id"] = "tampered"
    _write_json(path, value)
    with pytest.raises(ValueError, match="matrix report SHA-256"):
        summarize_matrix(path, replicates=100)


def test_rejects_unsupported_matrix_schema(tmp_path):
    path = _fixture(tmp_path)
    value = json.loads(path.read_text())
    value["schema_version"] = 3
    value.pop("report_sha256", None)
    value["report_sha256"] = _canonical_hash(value)
    _write_json(path, value)
    with pytest.raises(ValueError, match="supported schema version 4 through 7"):
        summarize_matrix(path, replicates=100)


def test_rejects_schema_eight(tmp_path):
    path = _fixture(tmp_path)
    value = json.loads(path.read_text())
    value["schema_version"] = 8
    value["protocol"]["schema_version"] = 8
    value.pop("report_sha256", None)
    value["report_sha256"] = _canonical_hash(value)
    _write_json(path, value)
    with pytest.raises(ValueError, match="supported schema version 4 through 7"):
        summarize_matrix(path, replicates=100)


def _schema6_fixture(path):
    from frc_defense.meco_phase4_context import phase4_start_profile_assignment
    matrix = json.loads(path.read_text())
    matrix["schema_version"] = matrix["protocol"]["schema_version"] = 6
    matrix["protocol"]["scenario_id"] = "full_match_offense"
    matrix["source_identity_sha256"] = matrix["protocol"]["source_identity_sha256"] = "b" * 64
    matrix["evaluation_split"] = matrix["protocol"]["evaluation_split"] = "validation"
    matrix["seed_split_assignments"] = matrix["protocol"]["seed_split_assignments"] = {
        "single": {str(seed): "validation" for seed in matrix["protocol"]["seed_list"]}}
    frozen = dict(matrix["qualification_protocol"])
    frozen["protocol_sha256"] = "a" * 64
    matrix["qualification_protocol"] = matrix["protocol"]["qualification_protocol"] = frozen
    matrix["conditions"]["architecture_matched_cv_k16"]["qualification_protocol"] = frozen
    assignment = phase4_start_profile_assignment("team_start_set_a",
        scenario_id="full_match_offense", split="validation",
        protocol_id=frozen["protocol_id"], protocol_sha256=frozen["protocol_sha256"],
        qualification_case=matrix["qualification_case"], categories=["single"],
        seeds=matrix["protocol"]["seed_list"])
    matrix["phase4_start_profile_assignment"] = assignment
    matrix["protocol"]["phase4_start_profile_assignment"] = assignment
    condition = matrix["conditions"]["architecture_matched_cv_k16"]
    condition["phase4_start_profile_assignment"] = assignment
    condition["source_identity_sha256"] = "b" * 64
    report_path = Path(condition["report_path"])
    report = json.loads(report_path.read_text())
    report["scenario_id"] = "full_match_offense"
    report["source_identity_sha256"] = "b" * 64
    report["split"] = "validation"
    report["evaluation_case"].update({"scenario_id": "full_match_offense",
        "split": "validation", "protocol_sha256": frozen["protocol_sha256"]})
    for entry in report["policy_manifest"].values():
        entry["source_identity_sha256"] = "b" * 64
    condition["policy_manifest"] = report["policy_manifest"]
    report["evaluation_protocol_sha256"] = frozen["protocol_sha256"]
    report["experiment_matrix"].update({
        "evaluation_split": "validation",
        "seed_split_assignments": matrix["seed_split_assignments"],
        "qualification_protocol": frozen,
        "phase4_start_profile_assignment": assignment})
    report["phase4_start_profile_assignment"] = assignment
    _write_json(report_path, report)
    condition["evaluation_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    _write_json(path, matrix)
    return matrix


def test_schema6_accepts_and_binds_start_profile_assignment(tmp_path):
    path = _fixture(tmp_path, schema_version=6)
    _schema6_fixture(path)
    matrix = json.loads(path.read_text())
    loaded = _resolve_conditions(matrix, path)
    assert loaded["common"]["evaluation_split"] == "validation"


def test_schema6_rejects_condition_manifest_assignment_tampering(tmp_path):
    path = _fixture(tmp_path, schema_version=6)
    _schema6_fixture(path)
    matrix = json.loads(path.read_text())
    matrix["conditions"]["architecture_matched_cv_k16"][
        "phase4_start_profile_assignment"] = None
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    _write_json(path, matrix)
    with pytest.raises(ValueError, match="condition manifest start-profile assignment"):
        _resolve_conditions(matrix, path)


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["protocol"].update(phase4_start_profile_assignment=None),
     "matching start-profile assignment"),
    (lambda m: (m["phase4_start_profile_assignment"].update(registry_profile_id="bogus"),
                m["protocol"]["phase4_start_profile_assignment"].update(
                    registry_profile_id="bogus")),
     "unknown full-team start profile"),
])
def test_schema6_rejects_start_profile_assignment_tampering(tmp_path, mutate, match):
    path = _fixture(tmp_path, schema_version=6)
    _schema6_fixture(path)
    matrix = json.loads(path.read_text())
    mutate(matrix)
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    _write_json(path, matrix)
    with pytest.raises(ValueError, match=match):
        _resolve_conditions(matrix, path)


def _schema7_fixture(path):
    from frc_defense.meco_phase4_context import phase4_teammate_behavior_assignment
    matrix = _schema6_fixture(path)
    matrix["schema_version"] = matrix["protocol"]["schema_version"] = 7
    frozen = matrix["qualification_protocol"]
    case = matrix["qualification_case"]
    seeds = matrix["protocol"]["seed_list"]
    condition_name = "architecture_matched_cv_k16"
    old = matrix["conditions"][condition_name]
    old_report_path = Path(old["report_path"])
    old_report = json.loads(old_report_path.read_text())
    new_specs, new_conditions = [], {}
    for arm in ("no_factor", "behavior_pool"):
        name = f"phase4_teammate_behaviors_{arm}"
        condition_id = f"condition-{arm}"
        spec = {"condition_name": name, "prediction_mode": "cv",
            "control_prediction_mode": "cv", "candidate_k": 16,
            "teammate_behavior_arm": arm, "teammate_behavior_pool_seed": 2}
        assignment = phase4_teammate_behavior_assignment(arm=arm,
            scenario_id="full_match_offense", split="validation",
            protocol_id=frozen["protocol_id"], protocol_sha256=frozen["protocol_sha256"],
            qualification_case=case, categories=["single"], seeds=seeds,
            pool_seed=2)
        report = json.loads(json.dumps(old_report))
        report.update(evaluation_id=f"eval-{arm}",
                      phase4_teammate_behavior_assignment=assignment)
        report["experiment_matrix"].update({"condition_id": condition_id,
            "condition": spec,
            "phase4_teammate_behavior_assignment": assignment})
        for row in report["records"]:
            row["episode_id"] = f"episode-{arm}-{row['policy']}-{row['seed']}"
            row["elapsed_ticks"] = 2
            if arm == "no_factor":
                runtime = {"schema": "phase4-teammate-behavior-runtime-v1",
                    "arm": arm, "assignments": None, "assignment_applied": False,
                    "events": []}
            else:
                scheduled = assignment["behavior_schedule"][seeds.index(row["seed"])]["assignments"]
                events = []
                for tick in range(2):
                    slots = []
                    for slot, behavior in scheduled.items():
                        action = {"collect": tick % 4, "score": 4, "defend": 5,
                                  "ferry": 6, "hold_position": 7}[behavior]
                        slots.append({"slot": slot, "requested_behavior_action": action,
                            "helper_selected_action": action, "actually_applied_action": action,
                            "requested_legal": True, "helper_fallback": False,
                            "environment_remask_fallback": False, "active": True})
                    events.append({"tick": tick, "slots": slots})
                runtime = {"schema": "phase4-teammate-behavior-runtime-v1",
                    "arm": arm,
                    "assignments": {str(slot): behavior for slot, behavior in scheduled.items()},
                    "assignment_applied": True, "events": events}
            row["phase4_teammate_behavior"] = phase4_teammate_behavior_record(
                assignment, runtime, episode_id=row["episode_id"],
                scenario_id="full_match_offense", category=row["category"],
                seed=row["seed"], policy_id=row["policy_id"],
                condition_id=condition_id,
                source_identity_sha256=report["source_identity_sha256"],
                outcome_metrics=row.get("metrics"),
                expected_action_ticks=len(runtime["events"]),
                checkpoint_sha256=row.get("checkpoint_sha256"),
                execution_device=row.get("execution_device"))
        report_path = path.parent / f"{arm}.condition.json"
        _write_json(report_path, report)
        condition = dict(old)
        condition.update(condition_id=condition_id, evaluation_id=report["evaluation_id"],
            report_path=str(report_path), condition=spec,
            phase4_teammate_behavior_assignment=assignment,
            evaluation_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
        new_specs.append(spec)
        new_conditions[name] = condition
    matrix["conditions"] = new_conditions
    matrix["protocol"]["conditions"] = new_specs
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    _write_json(path, matrix)
    return matrix


def test_schema7_accepts_paired_teammate_runtime_assignments(tmp_path):
    path = _fixture(tmp_path, schema_version=6)
    matrix = _schema7_fixture(path)
    loaded = _resolve_conditions(matrix, path)
    assert set(loaded["conditions"]) == {
        "phase4_teammate_behaviors_no_factor",
        "phase4_teammate_behaviors_behavior_pool"}


@pytest.mark.parametrize("damage,match", [
    ("remove_arm", "exactly one of each arm"),
    ("duplicate_arm", "exactly one of each arm"),
    ("bad_schedule", "differs from frozen spec"),
    ("bad_event", "runtime record differs"),
])
def test_schema7_rejects_arm_schedule_and_event_tampering(tmp_path, damage, match):
    path = _fixture(tmp_path, schema_version=6)
    matrix = _schema7_fixture(path)
    if damage in ("remove_arm", "duplicate_arm"):
        specs = matrix["protocol"]["conditions"]
        if damage == "remove_arm":
            specs.pop()
            matrix["conditions"].pop("phase4_teammate_behaviors_behavior_pool")
        else:
            specs[-1]["teammate_behavior_arm"] = "no_factor"
        matrix.pop("report_sha256", None)
        matrix["report_sha256"] = _canonical_hash(matrix)
        _write_json(path, matrix)
    elif damage == "bad_schedule":
        matrix["conditions"]["phase4_teammate_behaviors_behavior_pool"][
                "phase4_teammate_behavior_assignment"]["behavior_schedule"][0][
                    "assignments"][4] = "defend"
        matrix.pop("report_sha256", None)
        matrix["report_sha256"] = _canonical_hash(matrix)
        _write_json(path, matrix)
    else:
        condition = matrix["conditions"]["phase4_teammate_behaviors_behavior_pool"]
        report_path = Path(condition["report_path"])
        report = json.loads(report_path.read_text())
        report["records"][0]["phase4_teammate_behavior"]["assignment_pair_sha256"] = "0" * 64
        _write_json(report_path, report)
        condition["evaluation_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        matrix.pop("report_sha256", None)
        matrix["report_sha256"] = _canonical_hash(matrix)
        _write_json(path, matrix)
    with pytest.raises(ValueError, match=match):
        _resolve_conditions(matrix, path)


@pytest.mark.parametrize("field,value", [
    ("evaluation_protocol_id", "wrong-protocol"),
    ("evaluation_protocol_sha256", "wrong-sha"),
    ("exact_episode_seeds", [10, 11, 12]),
])
def test_fails_closed_on_protocol_v2_binding_mismatch(tmp_path, field, value):
    path = _fixture(tmp_path)
    matrix = json.loads(path.read_text())
    report_path = Path(matrix["conditions"]["architecture_matched_cv_k16"]["report_path"])
    report = json.loads(report_path.read_text())
    report[field] = value
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="protocol-v2 binding"):
        summarize_matrix(path, replicates=100)


def test_fails_closed_on_missing_condition_report(tmp_path):
    path = _fixture(tmp_path)
    matrix = json.loads(path.read_text())
    matrix["conditions"]["architecture_matched_cv_k16"]["report_path"] = str(
        tmp_path / "missing-report.json")
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    _write_json(path, matrix)
    with pytest.raises(ValueError, match="cannot read evaluation report"):
        summarize_matrix(path, replicates=100)


def test_diagnostic_matrix_requires_unbound_evaluator_reports(tmp_path):
    path = _fixture(tmp_path)
    matrix = json.loads(path.read_text())
    matrix.update({"evaluation_design": "diagnostic_only_unbound_seeds",
                   "evaluation_split": "diagnostic",
                   "qualification_protocol_id": None,
                   "qualification_protocol": None,
                   "seed_split_assignments": None})
    matrix["protocol"].update({"evaluation_design": "diagnostic_only_unbound_seeds",
        "evaluation_split": "diagnostic", "qualification_protocol_id": None,
        "qualification_protocol": None, "seed_split_assignments": None})
    for condition in matrix["conditions"].values():
        condition["qualification_protocol"] = None
        report_path = Path(condition["report_path"])
        report = json.loads(report_path.read_text())
        report.update({"qualification_protocol_id": None,
            "evaluation_protocol_id": None, "evaluation_protocol_sha256": None,
            "evaluation_case": None, "exact_episode_seeds": None,
            "evaluation_design": "diagnostic_unbound", "split": "diagnostic"})
        report["experiment_matrix"].update({
            "evaluation_design": "diagnostic_only_unbound_seeds",
            "evaluation_split": "diagnostic", "qualification_protocol_id": None,
            "qualification_protocol": None, "seed_split_assignments": None})
        _write_json(report_path, report)
    for contrast in matrix["contrasts"]:
        provenance = contrast["common_bundle_provenance"]
        provenance.update({"evaluation_design": "diagnostic_only_unbound_seeds",
            "evaluation_split": "diagnostic", "qualification_protocol_id": None,
            "qualification_protocol": None, "seed_split_assignments": None})
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    _write_json(path, matrix)
    summary = summarize_matrix(path, replicates=100)
    assert summary["diagnostic_only"] is True
    assert summary["provenance"]["evaluation_split"] == "diagnostic"


def test_pools_compatible_disjoint_qualification_strata(tmp_path):
    first = _fixture(tmp_path)
    second = _second_stratum(first)
    result = summarize_matrices([first, second], bootstrap_seed=9, replicates=500)
    score = next(row for row in result["summaries"]
                 if row.get("contrast") == "A1_vs_C2" and row.get("metric") == "score")
    assert score["n_episodes"] == 8
    assert score["cluster_count"] == 2
    assert score["mean_difference"] == pytest.approx(0.5)


def test_pooled_hash_retains_reports_with_duplicate_condition_ids(tmp_path):
    first = _fixture(tmp_path)
    second = _second_stratum(first)
    _reuse_condition_id(second)
    result = summarize_matrices([first, second], bootstrap_seed=17, replicates=100)
    assert len(result["evaluation_reports"]) == 2
    assert {row["condition_id"] for row in result["evaluation_reports"].values()} == {
        "condition-id"}
    expected_hash = _canonical_hash({
        "matrices": sorted(row["file_sha256"] for row in result["matrices"]),
        "evaluation_reports": sorted(({
            "matrix_id": row["matrix_id"], "condition_id": row["condition_id"],
            "sha256": row["sha256"]} for row in result["evaluation_reports"].values()),
            key=lambda row: (str(row["matrix_id"]), str(row["condition_id"]), row["sha256"]))})
    assert result["bootstrap"]["input_sha256"] == expected_hash


def test_equal_category_paired_effects_use_explicit_equal_weights_and_content_seed():
    summaries = []
    pairs = [(track, "D0") for track in ("A1", "A2", "A3", "C1", "C2")]
    pairs.extend([("C1", "D1"), ("D1", "D2"), ("A1", "C2"),
                  ("C1", "C2"), ("A1", "A2"), ("A1", "A3"),
                  ("A2", "A3")])
    for candidate, reference in pairs:
        for category, difference in (("cluster", 1.0), ("trench", 3.0)):
            summaries.append({"contrast": f"{candidate}_vs_{reference}",
                "candidate_policy": candidate, "reference_policy": reference,
                "metric": "score", "difference_direction": "candidate_minus_reference",
                "category": category, "_observations": [
                    {"seed": seed, "difference": difference} for seed in (11, 19)]})
    result = _equal_category_summaries(summaries, bootstrap_seed=17, replicates=500)
    assert len(result) == len(pairs)
    by_pair = {(row["candidate_policy"], row["reference_policy"]): row for row in result}
    assert set(by_pair) == set(pairs)
    row = by_pair[("A1", "D0")]
    assert row["category_weights"] == {"cluster": 0.5, "trench": 0.5}
    assert row["mean_difference"] == pytest.approx(2.0)
    assert row["ci_95"] == [2.0, 2.0]
    assert row["bootstrap"]["method"] == "within_category_paired_episode_resampling"
    assert row["bootstrap"]["seed"] == _content_bound_seed(
        17, row["bootstrap"]["input_sha256"])
    assert row["n_pairs"] == 4
    cv_summary = {**summaries[0], "difference_direction": "cv_minus_current"}
    cv_row = _equal_category_summaries([cv_summary], bootstrap_seed=17,
        replicates=100)[0]
    assert cv_row["difference_direction"] == "cv_minus_current"


def test_stratified_bootstrap_zero_seed_uses_nonzero_prng_state():
    interval = _stratified_paired_ci({"cluster": [0.0, 1.0]}, seed=0,
        replicates=1000)
    assert interval == [0.0, 1.0]


def test_equal_category_paired_effects_reject_duplicate_exact_pair_key():
    item = {"contrast": "A1_vs_D0", "candidate_policy": "A1",
        "reference_policy": "D0", "metric": "score",
        "difference_direction": "candidate_minus_reference", "category": "cluster",
        "_observations": [{"seed": 11, "difference": 1.0},
                          {"seed": 11, "difference": 2.0}]}
    with pytest.raises(ValueError, match="duplicate paired category/seed/policy"):
        _equal_category_summaries([item], bootstrap_seed=3, replicates=100)


def test_equal_category_paired_effects_reject_nonfinite_difference():
    item = {"contrast": "A1_vs_D0", "candidate_policy": "A1",
        "reference_policy": "D0", "metric": "score",
        "difference_direction": "candidate_minus_reference", "category": "cluster",
        "_observations": [{"seed": 11, "difference": float("inf")}]}
    with pytest.raises(ValueError, match="finite number"):
        _equal_category_summaries([item], bootstrap_seed=3, replicates=100)


def test_both_null_paired_values_fail_with_explicit_unavailable_reason():
    with pytest.raises(ValueError, match="metric score unavailable: both paired values are null"):
        _paired_finite_values(None, None, "score", "Phase 4 paired")


def test_matrix_summary_rejects_metric_missing_from_one_paired_policy(tmp_path):
    path = _fixture(tmp_path, schema_version=4)
    matrix = json.loads(path.read_text())
    report_path = Path(matrix["conditions"]["architecture_matched_cv_k16"]["report_path"])
    report = json.loads(report_path.read_text())
    for record in report["records"]:
        if record["policy"] == "A1":
            record["metrics"].pop("fuel")
    _write_json(report_path, report)
    for pair in matrix["contrasts"][0]["paired_episode_rows"]:
        pair["candidate_metrics"].pop("fuel")
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    _write_json(path, matrix)
    with pytest.raises(ValueError, match="missing or unpaired metric fields"):
        summarize_matrix(path, replicates=100)


def test_rejects_overlapping_category_seed_stratum_across_matrices(tmp_path):
    first = _fixture(tmp_path)
    second = _second_stratum(first, layout="layout-x")
    with pytest.raises(ValueError, match="overlapping category/seed/stratum"):
        summarize_matrices([first, second], replicates=100)


def test_rejects_same_protocol_id_with_different_frozen_protocol_hash(tmp_path):
    first = _fixture(tmp_path)
    second = _second_stratum(first)
    for path, digest in ((first, "protocol-hash-one"),
                         (second, "protocol-hash-two")):
        matrix = json.loads(path.read_text())
        provenance = {"protocol_id": "qual-protocol", "protocol_sha256": digest,
                      "protocol_file": "protocol.json"}
        matrix["qualification_protocol"] = provenance
        matrix["protocol"]["qualification_protocol"] = provenance
        for condition in matrix["conditions"].values():
            condition["qualification_protocol"] = provenance
            report_path = Path(condition["report_path"])
            report = json.loads(report_path.read_text())
            report["experiment_matrix"]["qualification_protocol"] = provenance
            report["evaluation_protocol_sha256"] = digest
            report["evaluation_case"]["protocol_sha256"] = digest
            report["evaluation_case"]["protocol_file"] = "protocol.json"
            _write_json(report_path, report)
        for contrast in matrix["contrasts"]:
            contrast["common_bundle_provenance"]["qualification_protocol"] = provenance
        matrix.pop("report_sha256", None)
        matrix["report_sha256"] = _canonical_hash(matrix)
        _write_json(path, matrix)
    with pytest.raises(ValueError, match="incompatible qualification protocol ID/SHA-256"):
        summarize_matrices([first, second], replicates=100)


def test_matched_prediction_interval_uses_cv_minus_current_metrics(tmp_path):
    path = _matched_prediction_fixture(tmp_path)
    result = summarize_matrices([path], bootstrap_seed=3, replicates=500)
    row = next(row for row in result["summaries"]
               if row.get("contrast") == "matched_prediction_current_vs_cv"
               and row.get("metric") == "score")
    assert row["difference_direction"] == "cv_minus_current"
    assert row["mean_difference"] == pytest.approx(2.0)
    assert row["ci_95"] == [2.0, 2.0]


def test_phase4_multiple_factors_keep_distinct_paired_intervals(tmp_path):
    path = _phase4_fixture(tmp_path)
    result = summarize_matrices([path], bootstrap_seed=7, replicates=500)
    rows = {row["factor"]: row for row in result["summaries"]
            if row.get("contrast") == "phase4_factor_vs_no_factor"
            and row.get("candidate_policy") == "A1"
            and row.get("metric") == "score"}
    assert set(rows) == {"feinting_opponent", "observation_delay"}
    assert rows["feinting_opponent"]["mean_difference"] == pytest.approx(1.0)
    assert rows["feinting_opponent"]["ci_95"] == [1.0, 1.0]
    assert rows["observation_delay"]["mean_difference"] == pytest.approx(3.0)
    assert rows["observation_delay"]["ci_95"] == [3.0, 3.0]


def test_phase4_rejects_entire_missing_policy_for_each_factor(tmp_path):
    path = _phase4_fixture(tmp_path)
    matrix = json.loads(path.read_text())
    matrix["contrasts"] = [row for row in matrix["contrasts"]
        if not (row.get("contrast") == "phase4_factor_vs_no_factor"
                and row.get("candidate_policy") == "D2")]
    matrix.pop("report_sha256", None)
    matrix["report_sha256"] = _canonical_hash(matrix)
    _write_json(path, matrix)
    with pytest.raises(ValueError, match="missing a required factor/policy pair"):
        summarize_matrix(path, replicates=100)


def test_cli_direct_invocation_imports_project_package():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "summarize_meco_ablation_matrix.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=root,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--input-matrix" in result.stdout
