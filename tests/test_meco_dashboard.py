import json
from pathlib import Path

from frc_defense.meco_dashboard import (
    CONTROLS, TRACKS, _collect_status, meco_status_payload,
)


def _report():
    categories, seeds = ["open", "defended"], [11, 12]
    return {
        "schema_version": 1, "evaluation_id": "eval-001", "protocol_id": "proto-a",
        "sources": {"simulator": "abc", "policy": "def"},
        "scenario_id": "fuel-test", "categories": categories, "seeds": seeds,
        "tracks": list(TRACKS), "controls": list(CONTROLS), "paired": True,
        "diagnostic": False,
        "records": [{"scenario_id": "fuel-test", "category": category,
                     "seed": seed, "policy": policy,
                     "metrics": {"collisions": 0, "p95_latency_ms": 3.5},
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


def test_fresh_queue_does_not_claim_worker_liveness(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "checkpoints", tmp_path / "evaluations"
    lane = checkpoint_root / "run-a"
    lane.mkdir(parents=True)
    evaluation_root.mkdir()
    now = 1_800_000_000.
    stamp = "2027-01-15T08:00:00+00:00"
    (checkpoint_root / "queue-status.json").write_text(json.dumps({
        "schema_version": 1, "updated_utc": stamp,
        "lanes": [{"track": "A1", "run_id": "r1", "status": "running"}],
    }))
    (lane / "run.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "r1", "track": "A1", "status": "running",
    }))
    result = _collect_status(checkpoint_root, evaluation_root, now=now,
                             process_probe=lambda _pid: True)
    policy = next(row for row in result["policies"] if row["policy_id"] == "A1")
    assert result["queue"]["state"] == "fresh"
    assert policy["training_state"] == "unconfirmed"
    assert policy["run"]["heartbeat"]["state"] == "missing"


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


def test_only_complete_matched_evaluation_is_exposed(tmp_path):
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
    assert group["decision_grade"] is True
    assert group["record_count"] == 4
    assert group["policy_metrics"]["A1"][0]["candidate_recall"] == .9
    assert next(row for row in result["policies"] if row["policy_id"] == "D2")["evaluation_state"] == "decision_grade"

