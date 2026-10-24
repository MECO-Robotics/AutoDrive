import importlib.util
from pathlib import Path

import pytest

from frc_defense import meco_qualification as q
from frc_defense.training_curriculum import SCENARIOS


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/meco_d0_calibration_campaign.py"
SPEC = importlib.util.spec_from_file_location("meco_d0_calibration_campaign", SCRIPT)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def test_campaign_closure_is_full_registered_scenario_order():
    assert len(campaign._closure()) == 12
    assert campaign._closure() == tuple(SCENARIOS)
    assert campaign._closure()[-1] == "full_match_offense"


def test_scenario_shard_partition_is_deterministic_and_disjoint(monkeypatch):
    monkeypatch.setattr(q, "_require_calibration_gpu", lambda device: {
        "device": device, "device_name": "test WX9100", "backend": "cuda",
        "torch_version": "test", "hip_version": None})
    monkeypatch.setattr(q, "_calibration_source_identity_sha256", lambda: "f" * 64)
    kwargs = dict(scenario_id="pickup_simple", episodes=100, repetitions=3,
                  master_seed=20261024, shard_count=4, device="cuda:0")
    first = q._d0_calibration_shard_manifests(**kwargs)
    second = q._d0_calibration_shard_manifests(**kwargs)
    assert [m["manifest_id"] for m in first] == [m["manifest_id"] for m in second]
    buckets = [{q._unit_key(unit) for unit in manifest["work_units"]} for manifest in first]
    assert all(buckets[i].isdisjoint(buckets[j])
               for i in range(len(buckets)) for j in range(i + 1, len(buckets)))
    assert len(set.union(*buckets)) == first[0]["work_unit_count"]


def test_campaign_runner_reuses_and_resumes_through_existing_shard_api(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "validate_campaign", lambda value: value)
    calls = []
    monkeypatch.setattr(q, "run_d0_calibration_shard",
                        lambda manifest, **kwargs: calls.append((manifest["shard_index"], kwargs)))
    plans = [{"scenario_id": name, "manifests": [{"shard_index": 0, "shard_count": 1}]}
             for name in campaign._closure()]
    value = {"campaign_id": "c" * 64, "config": {"device": "cuda:1"},
             "scenario_plans": plans}
    result = campaign.run_campaign(value, tmp_path)
    assert result["shards_validated_or_run"] == 12
    assert len(calls) == 12
    assert calls[0][1]["device"] == "cuda:1"
    assert calls[0][1]["output"].name == "shard-0000-of-0001.artifact.json"


def test_merge_refuses_missing_shards_before_emitting_report(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "validate_campaign", lambda value: value)
    value = {"campaign_id": "c" * 64, "scenario_ids": list(campaign._closure()),
             "source_identity_sha256": "f" * 64,
             "runtime_device": {"device": "cuda:0"},
             "total_record_count": 12,
             "scenario_plans": [{"scenario_id": name, "manifests": [{
                 "shard_index": 0, "shard_count": 1}]} for name in campaign._closure()]}
    with pytest.raises(ValueError, match="missing shard"):
        campaign.merge_campaign(value, tmp_path / "missing", tmp_path / "reports")
    assert not (tmp_path / "reports" / "campaign.report.json").exists()


def test_merge_requires_and_merges_all_scenarios(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "validate_campaign", lambda value: value)
    monkeypatch.setattr(q, "_load_json", lambda path: {"path": str(path)})
    monkeypatch.setattr(q, "_validate_d0_shard_artifact", lambda artifact, manifest: None)
    monkeypatch.setattr(q, "_merged_d0_calibration_report", lambda manifests, artifacts: {
        "records": [{}], "case_ids": [{}], "summary": {},
        "calibration_id": "a" * 64})
    monkeypatch.setattr(q, "write_immutable_report", lambda path, value: "b" * 64)
    scenarios = campaign._closure()
    value = {"campaign_id": "c" * 64, "scenario_ids": list(scenarios),
             "source_identity_sha256": "f" * 64,
             "runtime_device": {"device": "cuda:0"},
             "total_record_count": 12,
             "scenario_plans": [{"scenario_id": name, "manifests": [{
                 "shard_index": 0, "shard_count": 1}]} for name in scenarios]}
    root = tmp_path / "artifacts"
    for name in scenarios:
        output = campaign._artifact_path(root, name, 0, 1)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}")
    result = campaign.merge_campaign(value, root, tmp_path / "reports")
    assert result["scenario_count"] == 12
    assert result["record_count"] == 12
    assert result["output"].endswith("campaign.report.json")


def test_calibration_variance_summary_matches_population_variance():
    cases = [{"category": "single", "layout_id": "L", "defender_family": "none",
              "start_condition": "start", "sensor_profile": "standard", "seeds": [3, 7]}]
    records = []
    for seed, values in ((3, (1., 3., 5.)), (7, (5., 7., 9.))):
        for repetition, value in enumerate(values):
            records.append({"category": "single", "layout_id": "L", "defender_family": "none",
                "start_condition": "start", "sensor_profile": "standard", "seed": seed,
                "repetition": repetition, "metrics": {"useful_throughput": value}})
    summary = q._d0_calibration_summary(cases, 3, records, "pickup_simple")
    row = next(iter(summary.values()))
    assert row["mean_useful_throughput"] == 5.
    assert row["between_seed_variance"] == 4.
    assert row["mean_within_seed_repeat_variance"] == pytest.approx(8 / 3)
