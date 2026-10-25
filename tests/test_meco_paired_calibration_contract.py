"""Pure manifest contracts for the resumable paired Phase 0 calibration."""

import copy
import hashlib
import json
from pathlib import Path

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
             "freeze_sha256": freeze_sha, "source_commit": "1" * 40,
             "source_tree": "2" * 40, "dependency_lock_sha256": "3" * 64},
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
        "shard_count": 2, "source_binding": {"commit": "1" * 40,
            "tree": "2" * 40, "source_identity_sha256": "9" * 64,
            "source_tree_identity_sha256": "8" * 64,
            "dependency_lock_sha256": "3" * 64},
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
    assert {unit["control_id"] for unit in units} == set(q.PAIRED_CALIBRATION_CONTROLS)
    assert len(units) == campaign["work_unit_count"] == 12 * 2 * 2 * 3 * 2
    assert {unit["split"] for unit in units} == {"calibration"}
    pairs = {}
    for unit in units:
        pairs.setdefault(unit["pair_id"], set()).add(unit["control_id"])
    assert all(controls == {"D0", pair["paired_control_id"]}
               for pair_id, controls in pairs.items()
               for pair in [next(unit for unit in units if unit["pair_id"] == pair_id)])
    shard_for_pair = {}
    for shard in campaign["shards"]:
        for unit in shard["work_units"]:
            previous = shard_for_pair.setdefault(unit["pair_id"], shard["shard_index"])
            assert previous == shard["shard_index"]
    assert len(campaign["shards"]) == 2
    assert {shard["worker_id"] for shard in campaign["shards"]} == {"wx-a", "wx-b"}
    assert q.paired_calibration_execution_blockers(campaign) == ["D2/multi_model_v1"]
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
    ("repeat_semantics", "resample rows"),
    ("summary_method", "choose-best-observed"),
    ("bootstrap_replicates", 1),
])
def test_paired_campaign_rejects_unpinned_summary_configuration(monkeypatch, field, value):
    campaign = _campaign(monkeypatch)
    campaign["config"][field] = value
    with pytest.raises(ValueError, match="config is invalid"):
        q.validate_paired_calibration_campaign(campaign)


def test_paired_progress_is_manifest_bound_unique_and_resume_stable(monkeypatch):
    campaign = _campaign(monkeypatch)
    shard = campaign["shards"][0]
    unit = shard["work_units"][0]
    row = {"schema": q.PAIRED_CALIBRATION_ARTIFACT,
        "campaign_id": campaign["campaign_id"],
        "work_unit_id": q._canonical_sha256(unit), "work_unit": unit,
        "row_payload": {"replay_sha256": "a" * 64}}
    progress = q.build_paired_calibration_progress(campaign, 0, [row])
    assert progress["complete"] is False
    assert q.validate_paired_calibration_progress(campaign, 0, progress) == progress

    changed = copy.deepcopy(progress)
    changed["completed_rows"].append(row)
    body = {key: value for key, value in changed.items() if key != "progress_id"}
    changed["progress_id"] = q._canonical_sha256(body)
    with pytest.raises(ValueError, match="duplicate, unordered"):
        q.validate_paired_calibration_progress(campaign, 0, changed)

    altered = copy.deepcopy(progress)
    altered["shard_manifest_id"] = "b" * 64
    body = {key: value for key, value in altered.items() if key != "progress_id"}
    altered["progress_id"] = q._canonical_sha256(body)
    with pytest.raises(ValueError, match="another manifest"):
        q.validate_paired_calibration_progress(campaign, 0, altered)


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


def test_gate_artifact_reader_binds_exact_bytes_and_rejects_symlinks(tmp_path):
    path = tmp_path / "freeze.json"
    raw = json.dumps({"freeze_sha256": "a" * 64}, sort_keys=True).encode()
    path.write_bytes(raw)
    binding = {"path": "freeze.json", "artifact_sha256": hashlib.sha256(raw).hexdigest()}
    assert q._read_paired_bound_json(tmp_path, binding, kind="baseline freeze")[
        "freeze_sha256"] == "a" * 64

    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="bytes changed"):
        q._read_paired_bound_json(tmp_path, binding, kind="baseline freeze")

    outside = tmp_path.parent / "outside-freeze.json"
    outside.write_bytes(raw)
    link = tmp_path / "linked.json"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        q._paired_repo_json_file(tmp_path, "linked.json", kind="baseline freeze")

    with pytest.raises(ValueError, match="unsafe"):
        q._paired_repo_json_file(tmp_path, "../outside-freeze.json", kind="baseline freeze")


def test_gate_binding_reports_git_lookup_failure_without_name_error(monkeypatch, tmp_path):
    for name in ("freeze.json", "partition.json", "approval.json"):
        (tmp_path / name).write_text("{}")
    def fail_git(*args, **kwargs):
        raise q.subprocess.CalledProcessError(128, args[0], stderr="not a git checkout")
    monkeypatch.setattr(q.subprocess, "run", fail_git)
    with pytest.raises(ValueError, match="cannot inspect source commit/tree"):
        q.prepare_paired_calibration_gate_bindings(repo_root=tmp_path,
            baseline_freeze_path="freeze.json", partition_path="partition.json",
            estimand_approval_path="approval.json", configuration_artifacts={})


@pytest.mark.parametrize("source_key,value", [
    ("tree", "4" * 40), ("dependency_lock_sha256", "5" * 64),
])
def test_campaign_rejects_source_tree_and_runtime_lock_drift(monkeypatch, source_key, value):
    campaign = _campaign(monkeypatch)
    campaign["source"][source_key] = value
    with pytest.raises(ValueError, match="source/runtime binding"):
        q.validate_paired_calibration_campaign(campaign)


def test_campaign_rejects_partition_leakage_and_semantic_approval_drift(monkeypatch):
    campaign = _campaign(monkeypatch)
    campaign["partition"]["baseline_freeze_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="not bound"):
        q.validate_paired_calibration_campaign(campaign)

    campaign = _campaign(monkeypatch)
    campaign["estimand_approval"]["approval_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="score/replay/approval contract"):
        q.validate_paired_calibration_campaign(campaign)


def test_paired_campaign_rejects_unsafe_artifact_paths_and_bad_repetition(monkeypatch):
    freeze, partition, approval = _bindings()
    freeze["path"] = "../outside/freeze.json"
    with pytest.raises(ValueError, match="path is unsafe"):
        _campaign(monkeypatch, baseline_freeze_binding=freeze)
    with pytest.raises(ValueError, match="at least two repetitions"):
        _campaign(monkeypatch, repetitions=1)
