import pytest

from frc_defense.meco_ablation_matrix import _architecture_track_vs_d0
from frc_defense.meco_tracks import MECO_TRACKS


@pytest.fixture
def paired_inputs():
    categories = ("cluster", "trench")
    seeds = (11, 19)
    records = {}
    for policy in ("A1", "D0"):
        for category in categories:
            for seed in seeds:
                records[(policy, category, seed)] = {
                    "policy_id": f"{policy}-id",
                    "metrics": {"score": seed + (1 if policy == "A1" else 0)},
                }
    report = {
        "categories": list(categories), "seeds": list(seeds),
        "evaluation_id": "eval-1", "protocol_id": "protocol-1",
        "source_identity_sha256": "source-hash",
        "policy_manifest": {"D0": {"policy_id": "D0_production_deterministic",
                                     "prediction_mode": None}},
        "prediction": {"learned_mode": "cv", "control_modes": {"D1": "current", "D2": "cv"}},
        "candidate_k": {"effective": 16},
    }
    return categories, seeds, records, report


@pytest.mark.parametrize("track", MECO_TRACKS)
def test_architecture_track_d0_rows_preserve_exact_pair_and_provenance(paired_inputs, track):
    categories, seeds, records, report = paired_inputs
    if track != "A1":
        for category in categories:
            for seed in seeds:
                records[(track, category, seed)] = {
                    "policy_id": f"{track}-id", "metrics": {"score": seed + 2}}
    contrast = _architecture_track_vs_d0(
        track=track, condition_id="condition-1", report=report,
        records=records, categories=categories, seeds=seeds)

    assert contrast["contrast"] == f"{track}_vs_D0_architecture_condition"
    assert contrast["comparison_scope"] == "architecture_condition_outcomes"
    assert contrast["condition_id"] == "condition-1"
    assert contrast["report_provenance"] == {
        "evaluation_id": "eval-1", "protocol_id": "protocol-1",
        "source_identity_sha256": "source-hash", "seed_list": list(seeds),
        "policy_manifest": report["policy_manifest"],
    }
    assert [(row["category"], row["seed"]) for row in contrast["paired_episode_rows"]] == [
        (category, seed) for category in categories for seed in seeds]
    first = contrast["paired_episode_rows"][0]
    assert first["candidate_record_policy_id"] == f"{track}-id"
    assert first["reference_record_policy_id"] == "D0-id"
    assert first["candidate_metrics"] == {"score": (12 if track == "A1" else 13)}
    assert first["reference_metrics"] == {"score": 11}
    semantics = contrast["condition_semantics"]
    assert semantics["learned_track_prediction_mode"] == "cv"
    assert semantics["d0_prediction_mode"] is None
    assert "not assigned" in semantics["fairness_caveat"]
    assert "prediction-mode-matched" in semantics["fairness_caveat"]


def test_architecture_d0_pair_rejects_missing_or_extra_rows(paired_inputs):
    categories, seeds, records, report = paired_inputs
    missing = dict(records)
    del missing[("D0", "cluster", 11)]
    with pytest.raises(ValueError, match="coverage differs"):
        _architecture_track_vs_d0(track="A1", condition_id="c", report=report,
            records=missing, categories=categories, seeds=seeds)

    extra = dict(records)
    extra[("D0", "unexpected", 11)] = {"policy_id": "D0-id", "metrics": {}}
    with pytest.raises(ValueError, match="coverage differs"):
        _architecture_track_vs_d0(track="A1", condition_id="c", report=report,
            records=extra, categories=categories, seeds=seeds)


def test_architecture_d0_pair_rejects_report_coverage_mismatch(paired_inputs):
    categories, seeds, records, report = paired_inputs
    report["seeds"] = [11]
    with pytest.raises(ValueError, match="report category/seed coverage differs"):
        _architecture_track_vs_d0(track="A1", condition_id="c", report=report,
            records=records, categories=categories, seeds=seeds)


def test_architecture_d0_pair_requires_metrics_and_record_policy_ids(paired_inputs):
    categories, seeds, records, report = paired_inputs
    records[("D0", "cluster", 11)]["metrics"] = None
    with pytest.raises(ValueError, match="lacks policy IDs/metrics"):
        _architecture_track_vs_d0(track="A1", condition_id="c", report=report,
            records=records, categories=categories, seeds=seeds)
