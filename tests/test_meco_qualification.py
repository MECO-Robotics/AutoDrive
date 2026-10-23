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
        "protocol_id": "protocol-test", "protocol_version": qualification.CALIBRATED_PROTOCOL_VERSION,
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
