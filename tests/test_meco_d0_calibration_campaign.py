import importlib.util
import hashlib
import json
from pathlib import Path
import threading

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


def test_merge_is_restartable_after_partial_write_and_rejects_conflicts(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "validate_campaign", lambda value: value)
    monkeypatch.setattr(q, "_validate_d0_shard_artifact", lambda artifact, manifest: None)
    monkeypatch.setattr(q, "_merged_d0_calibration_report", lambda manifests, artifacts: {
        "records": [{}], "case_ids": [{}], "summary": {},
        "calibration_id": "a" * 64})
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
    reports = tmp_path / "reports"
    original_writer = campaign._write_immutable_or_verify_stable
    writes = 0

    def fail_after_four_writes(path, payload):
        nonlocal writes
        result = original_writer(path, payload)
        writes += 1
        if writes == 4:
            raise RuntimeError("simulated interruption")
        return result

    monkeypatch.setattr(campaign, "_write_immutable_or_verify_stable", fail_after_four_writes)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        campaign.merge_campaign(value, root, reports)
    assert len(list(reports.glob("*.calibration.json"))) == 4

    monkeypatch.setattr(campaign, "_write_immutable_or_verify_stable", original_writer)
    result = campaign.merge_campaign(value, root, reports)
    assert result["scenario_count"] == 12
    assert result["record_count"] == 12
    assert result["output"].endswith("campaign.report.json")
    first_digest = result["report_sha256"]
    resumed = campaign.merge_campaign(value, root, reports)
    assert resumed["report_sha256"] == first_digest

    conflict = reports / f"{scenarios[0]}.calibration.json"
    conflict.write_text('{"conflict":true}')
    with pytest.raises(ValueError, match="different content"):
        campaign.merge_campaign(value, root, reports)


def test_atomic_writer_concurrent_identical_publication_is_idempotent(tmp_path, monkeypatch):
    destination = tmp_path / "same-report.json"
    payload = {"schema": 1, "rows": list(range(100))}
    barrier = threading.Barrier(2)
    original = campaign._write_and_sync_temp

    def synchronized_write(path, encoded):
        original(path, encoded)
        barrier.wait(timeout=5)

    monkeypatch.setattr(campaign, "_write_and_sync_temp", synchronized_write)
    results = []
    failures = []

    def publish():
        try:
            results.append(campaign._write_immutable_or_verify_stable(destination, payload))
        except BaseException as exc:  # propagate worker-thread failures into the assertion
            failures.append(exc)

    workers = [threading.Thread(target=publish) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert not any(worker.is_alive() for worker in workers)
    assert failures == []
    assert results == [results[0], results[0]]
    assert json.loads(destination.read_text()) == payload
    assert list(tmp_path.glob(".*.tmp")) == []
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, indent=2,
        allow_nan=False).encode()).hexdigest()
    assert results[0] == expected


def test_atomic_writer_cleans_interrupted_temp_and_can_retry(tmp_path, monkeypatch):
    destination = tmp_path / "interrupted-report.json"
    payload = {"schema": 1, "records": ["complete", "payload"]}
    original = campaign._write_and_sync_temp

    def interrupted_write(path, encoded):
        with open(path, "wb") as stream:
            stream.write(encoded[:8])
            stream.flush()
        raise OSError("simulated interruption during temporary write")

    monkeypatch.setattr(campaign, "_write_and_sync_temp", interrupted_write)
    with pytest.raises(OSError, match="simulated interruption"):
        campaign._write_immutable_or_verify_stable(destination, payload)
    assert not destination.exists()
    assert list(tmp_path.glob(".*.tmp")) == []

    # A process kill can leave a uniquely named temporary file behind. It is
    # never used as the final destination and must not prevent a fresh publish.
    orphan = tmp_path / ".interrupted-report.json.orphan.tmp"
    orphan.write_bytes(b"partial")
    monkeypatch.setattr(campaign, "_write_and_sync_temp", original)
    digest = campaign._write_immutable_or_verify_stable(destination, payload)
    assert json.loads(destination.read_text()) == payload
    assert campaign._write_immutable_or_verify_stable(destination, payload) == digest
    assert list(tmp_path.glob(".*.tmp")) == [orphan]


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
