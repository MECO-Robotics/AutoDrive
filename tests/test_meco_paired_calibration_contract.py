"""Pure manifest contracts for the resumable paired Phase 0 calibration."""

import copy

import pytest

from frc_defense import meco_qualification as q


def _workers():
    common = {"device_model": "AMD Instinct MI300X WX9100",
        "backend": "hip", "backend_version": "rocm-test",
        "torch_version": "torch-test"}
    return [
        {"worker_id": "wx-a", "device": "cuda:0", "device_index": 0,
         "device_uuid": "GPU-a", **common},
        {"worker_id": "wx-b", "device": "cuda:1", "device_index": 1,
         "device_uuid": "GPU-b", **common},
    ]


def _bindings():
    freeze_sha = "b" * 64
    return ({"path": "artifacts/baseline-freeze.json",
             "artifact_sha256": "a" * 64,
             "schema": "meco-baseline-freeze-v2",
             "freeze_sha256": freeze_sha},
            {"path": "artifacts/partitions.json",
             "artifact_sha256": "c" * 64,
             "schema": "meco-heldout-partitions-v1",
             "partition_sha256": "d" * 64,
             "baseline_freeze_sha256": freeze_sha},
            {"path": "approvals/estimand.json",
             "artifact_sha256": "e" * 64,
             "schema": "meco-simulated-score-estimand-review-v1",
             "approval_sha256": "f" * 64})


def _cases(monkeypatch):
    def fake_cases(scenario_id, split, *, episodes, master_seed):
        assert split == "calibration"
        assert episodes == 100
        assert master_seed == 123
        return (q.QualificationCase(scenario_id, "category-a", "calibration",
            "layout-a", "family-a", "start-a", "sensor-a", (3, 7)),)
    monkeypatch.setattr(q, "protocol_cases", fake_cases)


def _campaign(monkeypatch, **overrides):
    _cases(monkeypatch)
    freeze, partition, approval = _bindings()
    values = {"episodes": 100, "repetitions": 2, "master_seed": 123,
        "shard_count": 2, "source_identity_sha256": "9" * 64,
        "baseline_freeze_binding": freeze, "partition_binding": partition,
        "estimand_approval_binding": approval,
        "worker_assignments": _workers()}
    values.update(overrides)
    return q.build_paired_calibration_campaign(**values)


def test_paired_campaign_binds_all_controls_and_complete_calibration_schedule(monkeypatch):
    campaign = _campaign(monkeypatch)
    units = [unit for shard in campaign["shards"] for unit in shard["work_units"]]
    assert campaign["campaign_type"] == q.PAIRED_CALIBRATION_VERSION
    assert campaign["config"]["control_ids"] == list(q.PAIRED_CALIBRATION_CONTROLS)
    multi_model = next(row for row in campaign["controls"]
        if row["control_id"] == "D2/multi_model_v1")
    assert multi_model["execution_supported"] is False
    assert "not implemented" in multi_model["support_limit"]
    assert {unit["control_id"] for unit in units} == set(
        q.PAIRED_CALIBRATION_CONTROLS[1:])
    assert len(units) == campaign["work_unit_count"] == 12 * 2 * 2 * 3
    assert {unit["split"] for unit in units} == {"calibration"}
    pairs = {}
    for unit in units:
        pairs.setdefault(unit["pair_id"], set()).add(unit["control_id"])
    assert all(controls == set(q.PAIRED_CALIBRATION_CONTROLS[1:])
               for controls in pairs.values())
    assert len(campaign["shards"]) == 2
    assert {shard["worker_id"] for shard in campaign["shards"]} == {"wx-a", "wx-b"}
    assert q.validate_paired_calibration_campaign(campaign) is campaign


def test_paired_campaign_rejects_schedule_or_binding_mutation(monkeypatch):
    campaign = _campaign(monkeypatch)
    altered = copy.deepcopy(campaign)
    altered["shards"][0]["work_units"].pop()
    with pytest.raises(ValueError, match="shard assignment"):
        q.validate_paired_calibration_campaign(altered)

    altered = copy.deepcopy(campaign)
    altered["estimand_approval"]["approval_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="score/replay/approval contract"):
        q.validate_paired_calibration_campaign(altered)

    altered = copy.deepcopy(campaign)
    altered["partition"]["baseline_freeze_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="not bound"):
        q.validate_paired_calibration_campaign(altered)


@pytest.mark.parametrize("field,value", [
    ("device", "cuda"),
    ("device", "cpu"),
    ("device_model", "NVIDIA P600"),
    ("backend", "cuda"),
])
def test_paired_worker_selection_rejects_fallback_and_non_wx9100(field, value):
    workers = _workers()
    workers[0][field] = value
    with pytest.raises(ValueError, match="indexed WX9100"):
        q._paired_worker_assignments(workers)


def test_paired_worker_selection_rejects_duplicate_or_mixed_runtime():
    workers = _workers()
    workers[1]["device_uuid"] = workers[0]["device_uuid"]
    with pytest.raises(ValueError, match="two distinct devices"):
        q._paired_worker_assignments(workers)
    workers = _workers()
    workers[1]["backend_version"] = "different"
    with pytest.raises(ValueError, match="mixed runtime"):
        q._paired_worker_assignments(workers)


def test_paired_campaign_rejects_unsafe_artifact_paths_and_bad_repetition(monkeypatch):
    freeze, partition, approval = _bindings()
    freeze["path"] = "../outside/freeze.json"
    with pytest.raises(ValueError, match="path is unsafe"):
        _campaign(monkeypatch, baseline_freeze_binding=freeze)
    with pytest.raises(ValueError, match="at least two repetitions"):
        _campaign(monkeypatch, repetitions=1)
