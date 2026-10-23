import json
import hashlib
from pathlib import Path

from frc_defense.meco_dashboard import (
    CONTROLS, TRACKS, _collect_status, meco_status_payload,
)


def _report():
    categories, seeds = ["open", "defended"], [11, 12]
    sources = {"simulator": "abc", "policy": "def"}
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return {
        "schema_version": 1, "evaluation_id": "eval-001", "protocol_id": "proto-a",
        "sources": sources, "source_identity_sha256": source_hash,
        "scenario_id": "fuel-test", "categories": categories, "seeds": seeds,
        "tracks": list(TRACKS), "controls": list(CONTROLS), "paired": True,
        "split": "validation", "diagnostic": True,
        "records": [{"scenario_id": "fuel-test", "category": category,
                     "seed": seed, "policy": policy, "policy_id": policy,
                     "protocol_id": "proto-a", "split": "validation",
                     "source_identity_sha256": source_hash,
                     "metrics": {"success": True, "useful_throughput": 2.5,
                                 "collisions": 0, "p95_latency_ms": 3.5},
                     "candidate_recall": .9}
                    for category in categories for seed in seeds
                    for policy in TRACKS + CONTROLS],
    }


def test_fixed_checkout_roots_and_all_policy_rows(tmp_path):
    payload = json.loads(meco_status_payload(tmp_path))
    assert Path(payload["checkpoint_root"]) == tmp_path / "checkpoints/meco-five"
    assert [row["policy_id"] for row in payload["policies"]] == list(TRACKS + CONTROLS)
    assert payload["policies"][0]["training_state"] == "not_started"
    assert payload["policies"][-1]["training_state"] == "not_applicable"


def test_queued_track_without_manifest_is_visible_without_claiming_liveness(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "checkpoints", tmp_path / "evaluations"
    lane = checkpoint_root / "run-a"
    lane.mkdir(parents=True)
    evaluation_root.mkdir()
    now = 1_800_000_000.
    stamp = "2027-01-15T08:00:00+00:00"
    (checkpoint_root / "queue-status.json").write_text(json.dumps({
        "schema_version": 1, "updated_utc": stamp,
        "lanes": [{"track": "A1", "run_id": "r1", "status": "queued"}],
    }))
    result = _collect_status(checkpoint_root, evaluation_root, now=now,
                             process_probe=lambda _pid: True)
    policy = next(row for row in result["policies"] if row["policy_id"] == "A1")
    assert result["queue"]["state"] == "fresh"
    assert policy["training_state"] == "queued"
    assert policy["run"]["heartbeat"]["state"] == "not_started"


def test_live_requires_fresh_consistent_heartbeat_and_process(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    lane = checkpoint_root / "run-a"
    lane.mkdir(parents=True)
    evaluation_root.mkdir()
    now = 1_800_000_000.
    (lane / "run.json").write_text(json.dumps({"run_id": "r1", "track": "C1", "status": "running"}))
    (lane / "heartbeat.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "r1", "track": "C1", "pid": 99,
        "heartbeat_utc": "2027-01-15T08:00:00+00:00",
    }))
    result = _collect_status(checkpoint_root, evaluation_root, now=now,
                             process_probe=lambda pid: pid == 99)
    policy = next(row for row in result["policies"] if row["policy_id"] == "C1")
    assert policy["training_state"] == "running"
    assert policy["run"]["heartbeat"]["state"] == "confirmed"


def test_only_complete_matched_evaluation_is_exposed_with_category_uncertainty(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    report = _report()
    (evaluation_root / "good.json").write_text(json.dumps(report))
    incomplete = _report()
    incomplete["records"].pop()
    (evaluation_root / "incomplete.json").write_text(json.dumps(incomplete))
    result = _collect_status(checkpoint_root, evaluation_root, now=1_800_000_000.)
    assert len(result["comparison_groups"]) == 1
    group = result["comparison_groups"][0]
    assert group["matched_status"] == "diagnostic matched"
    assert group["record_count"] == 4
    assert group["policy_metrics"]["A1"]["open"]["success"]["n"] == 2
    assert group["policy_metrics"]["A1"]["open"]["success"]["normal_95_interval"] is not None
    assert next(row for row in result["policies"] if row["policy_id"] == "D2")["evaluation_state"] == "diagnostic matched"


def test_record_metrics_and_provenance_are_required(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    bad_metrics = _report()
    del bad_metrics["records"][0]["metrics"]["success"]
    bad_provenance = _report()
    bad_provenance["records"][0]["split"] = "test"
    (evaluation_root / "metrics.json").write_text(json.dumps(bad_metrics))
    (evaluation_root / "provenance.json").write_text(json.dumps(bad_provenance))
    result = _collect_status(checkpoint_root, evaluation_root, now=1_800_000_000.)
    assert result["comparison_groups"] == []


def test_evaluator_manifest_reports_live_progress_and_stale_state(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    report = _report()
    report["diagnostic"] = False
    report["evaluation_id"] = "eval-live"
    (evaluation_root / "run.json").write_text(json.dumps(report))
    manifest = {"schema_version": 1, "evaluation_id": "eval-live",
        "protocol_id": "proto-a", "split": "validation", "status": "running",
        "pid": 42, "heartbeat_utc": "2027-01-15T08:00:00+00:00",
        "output_file": "run.json", "progress": {"completed_records": 3, "total_records": 24}}
    status_path = evaluation_root / "run.json.status.json"
    status_path.write_text(json.dumps(manifest))
    live = _collect_status(checkpoint_root, evaluation_root, now=1_800_000_000.,
                           process_probe=lambda _pid: True)
    assert live["evaluation_runs"][0]["status"] == "running"
    assert live["evaluation_runs"][0]["progress"]["completed_records"] == 3
    stale = _collect_status(checkpoint_root, evaluation_root, now=1_800_001_000.,
                            process_probe=lambda _pid: False)
    assert stale["evaluation_runs"][0]["status"] == "stale/unconfirmed"


def test_running_evaluator_can_report_progress_before_evidence_file_exists(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    status = {"schema_version": 1, "evaluation_id": "eval-live",
        "protocol_id": "proto-a", "split": "validation", "status": "running",
        "pid": 42, "heartbeat_utc": "2027-01-15T08:00:00+00:00",
        "output_file": "pending-report.json", "progress": {"completed_records": 1, "total_records": 24}}
    (evaluation_root / "pending-report.json.status.json").write_text(json.dumps(status))
    result = _collect_status(checkpoint_root, evaluation_root, now=1_800_000_000.,
                             process_probe=lambda _pid: True)
    assert result["evaluation_runs"][0]["status"] == "running"
    assert result["evaluation_runs"][0]["error"] is None


def test_active_run_precedes_higher_update_completed_run(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    old = checkpoint_root / "old"
    active = checkpoint_root / "active"
    old.mkdir()
    active.mkdir()
    (old / "run.json").write_text(json.dumps({"run_id": "old", "track": "A2", "status": "complete", "completed_updates": 900}))
    (active / "run.json").write_text(json.dumps({"run_id": "active", "track": "A2", "status": "running", "completed_updates": 4}))
    result = _collect_status(checkpoint_root, evaluation_root, now=1_800_000_000.,
                             process_probe=lambda _pid: False)
    policy = next(row for row in result["policies"] if row["policy_id"] == "A2")
    assert policy["run"]["run_id"] == "active"
