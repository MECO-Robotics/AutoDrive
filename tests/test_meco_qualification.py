"""CPU-only qualification statistics and fail-closed promotion gate tests."""

from __future__ import annotations

import copy
import hashlib

import pytest

from frc_defense import meco_qualification as qualification
from frc_defense.training_curriculum import CurriculumState, SCENARIOS


def test_cluster_bootstrap_is_seeded_bounded_and_handles_degenerate_groups():
    groups = {("layout-a", "family-a"): [0.2, 0.4, 0.8],
              ("layout-b", "family-b"): [1.0, 1.4, 1.8]}
    first = qualification._cluster_bootstrap_ci(groups, seed=7021, replicates=2000)
    again = qualification._cluster_bootstrap_ci(groups, seed=7021, replicates=2000)
    assert first == again
    assert 0 <= first[0] <= first[1] <= 1.8

    degenerate = qualification._cluster_bootstrap_ci(
        {("one", "cluster"): [0.375] * 5}, seed=9, replicates=200)
    assert degenerate == [0.375, 0.375]


def test_cluster_bootstrap_resamples_clusters_as_units():
    # These two clusters have opposite within-cluster patterns. A clustered
    # bootstrap can draw either whole cluster; a flat episode bootstrap can
    # instead mix their rows in every replicate.
    groups = {("small", "a"): [-1.0] * 4,
              ("large", "b"): [1.0] * 4}
    clustered = qualification._cluster_bootstrap_ci(groups, seed=17, replicates=1000)
    flat = qualification._cluster_bootstrap_ci({("all", "episodes"): [-1.0] * 4 + [1.0] * 4},
                                               seed=17, replicates=1000)
    assert clustered[0] == -1.0 and clustered[1] == 1.0
    assert flat != clustered


def _synthetic_round(monkeypatch, *, useful_margin=0.1, difference=0.3,
                     safety_values=(0.0, 0.0), omit_safety=False, drop_candidate_seed=None,
                     candidate_policy="A1"):
    scenario = "pickup_simple"
    categories = SCENARIOS[scenario].required_categories
    seeds = tuple(range(100, 200))
    cases = tuple(qualification.QualificationCase(scenario, category, "validation",
        layout, family, "start-a", "sensor-a", seeds)
        for category in categories for layout, family in
        (("layout-a", "family-a"), ("layout-b", "family-b")))
    monkeypatch.setattr(qualification, "protocol_cases",
        lambda *args, **kwargs: cases)
    monkeypatch.setattr(qualification, "_validate_v2_protocol", lambda *args: None)
    monkeypatch.setattr(qualification, "_validate_report_protocol_binding", lambda *args: None)
    monkeypatch.setattr(qualification, "_validate_evidence_digest", lambda *args: None)
    monkeypatch.setattr(qualification, "BOOTSTRAP_REPLICATES", 300)
    import frc_defense.curriculum_training as training
    monkeypatch.setattr(training, "source_identity", lambda: {"test": "source"})
    source_hash = hashlib.sha256(b'{"test": "source"}').hexdigest()
    manifest = {candidate_policy: {"policy_id": f"{candidate_policy}-policy",
                  "checkpoint_sha256": "b" * 64},
                "D0": {"policy_id": "D0_production_deterministic",
                  "checkpoint_sha256": None}}
    protocol = {"promotion_target_scenario": scenario,
        "protocol_version": qualification.LEGACY_CALIBRATED_PROTOCOL_VERSION,
        "paired_reference_policy": qualification.REFERENCE_POLICY,
        "calibration_source_identity_sha256": source_hash,
        "protocol_id": "protocol-test", "episodes_per_category_stratum": 100,
        "master_seed": 12, "scenario_ids": [scenario],
        "promotion_gates": {scenario: qualification.asdict(SCENARIOS[scenario].gate)},
        "safety_gate_status": "metrics_schema_verified",
        "minimum_useful_gain": {category: useful_margin for category in categories},
        "safety_noninferiority_margins": {
            name: 0.1 for name in qualification.SAFETY_METRIC_NAMES}}
    expected_case_ids = [{"category": case.category, "layout_id": case.layout_id,
        "defender_family": case.defender_family,
        "start_condition": case.start_condition, "sensor_profile": case.sensor_profile}
        for case in cases]
    records = []
    for case in cases:
        for seed in case.seeds:
            for policy in (candidate_policy, "D0"):
                if policy == candidate_policy and seed == drop_candidate_seed:
                    continue
                metrics = {"success": 1, "useful_throughput": (1.0 + difference
                    if policy == candidate_policy else 1.0)}
                if not omit_safety or policy != candidate_policy:
                    metrics.update({name: safety_values[0 if policy == candidate_policy else 1]
                                    for name in qualification.SAFETY_METRIC_NAMES})
                records.append({"scenario_id": scenario, "category": case.category,
                    "layout_id": case.layout_id, "defender_family": case.defender_family,
                    "start_condition": case.start_condition,
                    "sensor_profile": case.sensor_profile, "seed": seed,
                    "policy": policy, "policy_id": manifest[policy]["policy_id"],
                    "checkpoint_sha256": manifest[policy]["checkpoint_sha256"],
                    "source_identity_sha256": source_hash, "split": "validation",
                    "protocol_id": "protocol-test", "metrics": metrics})
    report = {"qualification": True, "split": "validation", "scenario_id": scenario,
        "protocol_id": "protocol-test",
        "protocol_version": qualification.LEGACY_CALIBRATED_PROTOCOL_VERSION,
        "evaluation_id": "evaluation-test", "candidate_policy": candidate_policy,
        "bundle_id": qualification.bundle_identity(manifest), "policy_manifest": manifest,
        "reference_report": {"policy_manifest": copy.deepcopy(manifest)},
        "source_identity_sha256": source_hash, "gate_definition": protocol["promotion_gates"][scenario],
        "case_ids": expected_case_ids, "records": records,
        "safety_metrics_schema": qualification.SAFETY_METRICS_SCHEMA,
        "safety_metrics_source_identity_sha256": source_hash}
    return report, protocol, CurriculumState()


def test_useful_gain_gate_requires_lower_ci_strictly_above_margin(monkeypatch):
    report, protocol, state = _synthetic_round(monkeypatch, useful_margin=0.25, difference=0.25)
    decision, summary = qualification.consume_validation_round(report, protocol=protocol, state=state)
    for category in SCENARIOS["pickup_simple"].required_categories:
        assert summary["confidence_intervals"][category]["paired_difference_95_ci"] == [0.25, 0.25]
    assert not decision.passed
    assert any("does not exceed frozen margin" in failure for failure in decision.failures)


def test_safety_noninferiority_blocks_and_missing_values_fail_closed(monkeypatch):
    report, protocol, state = _synthetic_round(monkeypatch, safety_values=(0.25, 0.0))
    decision, summary = qualification.consume_validation_round(report, protocol=protocol, state=state)
    assert not decision.passed
    assert any("safety noninferiority CI exceeds" in failure for failure in decision.failures)
    assert all(summary["confidence_intervals"][category]["safety_noninferiority"]
               for category in SCENARIOS["pickup_simple"].required_categories)

    report, protocol, state = _synthetic_round(monkeypatch, omit_safety=True)
    # Historical v1 input is allowed to omit instrumentation, but the
    # promotion gate must still fail closed when it cannot calculate CIs.
    report["protocol_version"] = qualification.PROTOCOL_VERSION
    decision, summary = qualification.consume_validation_round(report, protocol=protocol, state=state)
    assert not decision.passed
    assert all(value is None for category in SCENARIOS["pickup_simple"].required_categories
               for value in summary["confidence_intervals"][category]
               ["safety_noninferiority"].values())
    assert any("safety metric" in failure and "absent or invalid" in failure
               for failure in decision.failures)


def test_invalid_safety_rate_and_incomplete_paired_coverage_are_rejected(monkeypatch):
    report, protocol, state = _synthetic_round(monkeypatch, safety_values=(float("nan"), 0.0))
    with pytest.raises(ValueError, match="finite rate"):
        qualification.consume_validation_round(report, protocol=protocol, state=state)

    report, protocol, state = _synthetic_round(monkeypatch, drop_candidate_seed=100)
    with pytest.raises(ValueError, match="incomplete qualification coverage"):
        qualification.consume_validation_round(report, protocol=protocol, state=state)


def _calibration_runtime_identity(**overrides):
    identity = {"device": "cuda:0", "device_name": "WX9100",
        "backend": "hip", "torch_version": "2.8.0+rocm7.0",
        "hip_version": "7.0.0"}
    identity.update(overrides)
    return identity


@pytest.mark.parametrize("device", ("cuda", "cuda:-1", "cuda:1junk", "cuda:01", "cuda:0.0"))
def test_calibration_runtime_identity_rejects_malformed_device(device):
    with pytest.raises(ValueError):
        qualification._validate_calibration_runtime_device(
            _calibration_runtime_identity(device=device))


@pytest.mark.parametrize("identity", (
    _calibration_runtime_identity(torch_version=None),
    _calibration_runtime_identity(torch_version=7),
    _calibration_runtime_identity(torch_version=" "),
    _calibration_runtime_identity(backend="hip", hip_version=None),
    _calibration_runtime_identity(backend="hip", hip_version=7),
    _calibration_runtime_identity(backend="hip", hip_version=" "),
    _calibration_runtime_identity(backend="cuda", hip_version="7.0.0"),
    _calibration_runtime_identity(backend="vulkan", hip_version=None),
))
def test_calibration_runtime_identity_rejects_malformed_versions(identity):
    with pytest.raises(ValueError):
        qualification._validate_calibration_runtime_device(identity)


def test_calibration_runtime_identity_accepts_typed_cuda_and_hip():
    qualification._validate_calibration_runtime_device(_calibration_runtime_identity())
    qualification._validate_calibration_runtime_device(_calibration_runtime_identity(
        device="cuda:1", backend="cuda", hip_version=None))


def test_d0_calibration_manifest_is_source_bound_and_checkpoint_free():
    source = "a" * 64
    manifest = {"D0": {"policy_id": "D0_production_deterministic",
        "source_identity_sha256": source, "checkpoint_sha256": None}}
    qualification._validate_calibration_d0_manifest(manifest, source)

    wrong_source = copy.deepcopy(manifest)
    wrong_source["D0"]["source_identity_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="manifest"):
        qualification._validate_calibration_d0_manifest(wrong_source, source)

    checkpointed = copy.deepcopy(manifest)
    checkpointed["D0"]["checkpoint_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="manifest"):
        qualification._validate_calibration_d0_manifest(checkpointed, source)

    with pytest.raises(ValueError, match="manifest"):
        qualification._validate_calibration_d0_manifest({}, source)


def _synthetic_d0_calibration_bundle(monkeypatch):
    scenario = "pickup_simple"
    case = qualification.QualificationCase(scenario, "single", "calibration",
        "layout-a", "family-a", "start-a", "sensor-a", (103,))
    monkeypatch.setattr(qualification, "protocol_cases", lambda *args, **kwargs: (case,))
    source = "a" * 64
    runtime = _calibration_runtime_identity()
    policy_manifest = {"D0": {"policy_id": "D0_production_deterministic",
        "source_identity_sha256": source, "checkpoint_sha256": None}}
    rows = []
    for repetition in range(2):
        rows.append({"repetition": repetition, "category": case.category,
            "layout_id": case.layout_id, "defender_family": case.defender_family,
            "start_condition": case.start_condition, "sensor_profile": case.sensor_profile,
            "seed": 103, "policy": "D0", "split": "calibration",
            "policy_id": "D0_production_deterministic", "checkpoint_sha256": None,
            "source_identity_sha256": source,
            "calibration_runtime_device": copy.deepcopy(runtime),
            "metrics": {"useful_throughput": 1.0,
                **{name: 0.0 for name in qualification.SAFETY_METRIC_NAMES}}})
    stratum = (case.category, case.layout_id, case.defender_family,
               case.start_condition, case.sensor_profile)
    summary = {"/".join(stratum): {"category": case.category,
        "layout_id": case.layout_id, "defender_family": case.defender_family,
        "start_condition": case.start_condition, "sensor_profile": case.sensor_profile,
        "seed_count": 1, "repetition_count": 2,
        "mean_useful_throughput": 1.0, "between_seed_variance": 0.0,
        "mean_within_seed_repeat_variance": 0.0,
        "seed_digest": hashlib.sha256(b"[103]").hexdigest()}}
    report = {"schema_version": 2, "calibration": True,
        "protocol_version": qualification.D0_CALIBRATION_VERSION,
        "scenario_id": scenario, "episodes_per_category_stratum": qualification.MIN_EPISODES,
        "repetitions": 2, "master_seed": 23,
        "case_ids": [{"category": case.category, "layout_id": case.layout_id,
            "defender_family": case.defender_family,
            "start_condition": case.start_condition, "sensor_profile": case.sensor_profile}],
        "policy_manifest": policy_manifest, "source_identity_sha256": source,
        "runtime_device": runtime,
        "normalized_runtime_identity": qualification._normalized_calibration_runtime(runtime),
        "safety_metrics_schema": qualification.SAFETY_METRICS_SCHEMA,
        "safety_metrics_source_identity_sha256": source,
        "summary": summary, "records": rows}
    report["calibration_id"] = qualification._canonical_sha256(report)
    return report


def _rehash_calibration_report(report):
    report.pop("calibration_id", None)
    report["calibration_id"] = qualification._canonical_sha256(report)


def test_calibration_intake_binds_row_runtime_while_allowing_device_ordinal(monkeypatch):
    report = _synthetic_d0_calibration_bundle(monkeypatch)
    qualification._calibration_report_index({"pickup_simple": report}, ("pickup_simple",))

    report = _synthetic_d0_calibration_bundle(monkeypatch)
    del report["records"][0]["calibration_runtime_device"]
    _rehash_calibration_report(report)
    with pytest.raises(ValueError, match="row runtime provenance"):
        qualification._calibration_report_index({"pickup_simple": report}, ("pickup_simple",))

    report = _synthetic_d0_calibration_bundle(monkeypatch)
    report["records"][0]["calibration_runtime_device"]["device"] = "cuda:1"
    _rehash_calibration_report(report)
    qualification._calibration_report_index({"pickup_simple": report}, ("pickup_simple",))


@pytest.mark.parametrize("runtime_change", (
    {"backend": "cuda", "hip_version": None},
    {"device_name": "different GPU model"},
    {"torch_version": "2.9.0+rocm7.0"},
    {"hip_version": "7.1.0"},
))
def test_calibration_intake_rejects_row_runtime_backend_model_or_version_mismatch(
        monkeypatch, runtime_change):
    report = _synthetic_d0_calibration_bundle(monkeypatch)
    report["records"][0]["calibration_runtime_device"].update(runtime_change)
    _rehash_calibration_report(report)

    with pytest.raises(ValueError, match="row runtime provenance"):
        qualification._calibration_report_index({"pickup_simple": report}, ("pickup_simple",))


@pytest.mark.parametrize("corruption", ("manifest_source", "manifest_checkpoint",
                                         "runtime_device", "normalized_runtime"))
def test_calibration_intake_rejects_manifest_or_runtime_corruption(monkeypatch, corruption):
    report = _synthetic_d0_calibration_bundle(monkeypatch)
    if corruption == "manifest_source":
        report["policy_manifest"]["D0"]["source_identity_sha256"] = "b" * 64
    elif corruption == "manifest_checkpoint":
        report["policy_manifest"]["D0"]["checkpoint_sha256"] = "c" * 64
    elif corruption == "runtime_device":
        report["runtime_device"]["device"] = "cuda:1junk"
    else:
        report["normalized_runtime_identity"]["torch_version"] = "different"
    _rehash_calibration_report(report)

    with pytest.raises(ValueError):
        qualification._calibration_report_index({"pickup_simple": report}, ("pickup_simple",))


@pytest.mark.parametrize("corruption", (None, "source", "checkpoint"))
def test_shard_artifact_validation_binds_d0_manifest_to_source(monkeypatch, corruption):
    source = "a" * 64
    manifest = {"manifest_id": "manifest-test", "source_identity_sha256": source}
    policy_manifest = {"D0": {"policy_id": "D0_production_deterministic",
        "source_identity_sha256": source, "checkpoint_sha256": None}}
    if corruption == "source":
        policy_manifest["D0"]["source_identity_sha256"] = "b" * 64
    elif corruption == "checkpoint":
        policy_manifest["D0"]["checkpoint_sha256"] = "c" * 64
    artifact = {"schema_version": 1,
        "artifact_type": qualification.D0_CALIBRATION_SHARD_ARTIFACT,
        "manifest_id": manifest["manifest_id"], "manifest": manifest,
        "policy_manifest": policy_manifest,
        "safety_metrics_schema": qualification.SAFETY_METRICS_SCHEMA,
        "safety_metrics_source_identity_sha256": source, "entries": []}
    artifact["artifact_sha256"] = qualification._canonical_sha256(artifact)
    monkeypatch.setattr(qualification, "_validate_d0_shard_entries",
        lambda artifact, manifest, *, require_complete: [])

    if corruption is None:
        assert qualification._validate_d0_shard_artifact(artifact, manifest) == []
    else:
        with pytest.raises(ValueError, match="manifest"):
            qualification._validate_d0_shard_artifact(artifact, manifest)
