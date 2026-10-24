import json
import hashlib
import shutil
import subprocess
from pathlib import Path

from frc_defense.meco_dashboard import (
    ABLATIONS, CONTROLS, PENALTY_MAP_UNITS, PENALTY_PATH_UNITS,
    PENALTY_PPO_CONTRACT, PENALTY_PROJECTION, TRACKS, _collect_status,
    _three_pass_provenance, meco_status_payload,
)


def _report(episodes=2):
    categories = ["open", "defended"]
    seeds = list(range(11, 11 + episodes))
    sources = {"simulator": "abc", "policy": "def"}
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    checkpoint_hashes = {
        policy: hashlib.sha256(policy.encode()).hexdigest()
        for policy in TRACKS + ABLATIONS
    }
    policy_manifest = {
        policy: {"policy_id": f"model-{policy}",
                 "checkpoint_sha256": checkpoint_hashes[policy],
                 "source_identity_sha256": source_hash}
        for policy in TRACKS + ABLATIONS
    }
    for policy in TRACKS[:3]:
        policy_manifest[policy]["p_mode"] = "learned"
    for policy in ABLATIONS:
        policy_manifest[policy]["p_mode"] = "p0"
    control_ids = {"D0": "D0_production_deterministic", "D1": "D1_hand_weighted",
                   "D2": "D2_kinematic_prediction"}
    policy_manifest.update({policy: {"policy_id": control_ids[policy], "checkpoint_sha256": None,
                                    "source_identity_sha256": source_hash}
                           for policy in CONTROLS})
    report = {
        "schema_version": 1, "evaluation_id": "eval-001", "protocol_id": "proto-a",
        "sources": sources, "source_identity_sha256": source_hash,
        "scenario_id": "fuel-test", "categories": categories, "seeds": seeds,
        "episodes_per_category": len(seeds),
        "tracks": list(TRACKS), "ablations": list(ABLATIONS),
        "controls": list(CONTROLS), "paired": True,
        "split": "validation", "diagnostic": len(seeds) < 100,
        "control_definitions": {"D0": "production deterministic offense controller; native 20 ms physics-step cadence; actual executed actions recorded; not constrained by the MECO candidate table",
            "D1": "hand-weighted scorer over the shared feasible candidate table",
            "D2": "D1 scorer with kinematic opponent endpoint prediction"},
        "plan_a_penalty": {
            "trained_mode": "learned", "p0_ablation_mode": "p0",
            "cost_scale": 2.0, "points_per_grid_edge": 1.0,
            "projection": PENALTY_PROJECTION,
            "map_units": PENALTY_MAP_UNITS,
            "path_units": PENALTY_PATH_UNITS,
            "ppo_recomputation": PENALTY_PPO_CONTRACT},
        "policy_manifest": policy_manifest,
        "records": [],
    }
    for category in categories:
        for seed in seeds:
            for policy in TRACKS + ABLATIONS + CONTROLS:
                provenance = policy_manifest[policy]
                row = {
                    "scenario_id": "fuel-test", "category": category,
                    "seed": seed, "policy": policy,
                    "policy_id": provenance["policy_id"],
                    "checkpoint_sha256": provenance["checkpoint_sha256"],
                    "protocol_id": "proto-a", "split": "validation",
                    "source_identity_sha256": source_hash,
                    "metrics": {"success": True, "useful_throughput": 2.5,
                                "collisions": 0, "p95_latency_ms": 3.5},
                    "candidate_recall": .9,
                    "p_mode": provenance.get("p_mode"),
                }
                if policy == "D0":
                    row.update({
                        "action_source": "production_deterministic",
                        "controller_cadence": "native_20ms_physics_step",
                        "action_count_semantics": "executed_per_physics_step",
                        "elapsed_ticks": 3,
                        "executed_action_trace": [0, 1, 2],
                        "actions": [1, 1, 1, 0, 0, 0, 0, 0],
                    })
                if policy in TRACKS[:3] + ABLATIONS:
                    p_mode = "p0" if policy in ABLATIONS else "learned"
                    route_provenance = (
                        "legacy_bump_cost_path_integral_v2" if p_mode == "p0" else
                        "meco_legal_heading_penalty_projection_v1:schema=xy_grid_f64le_v2:"
                        + "sha256=" + "a" * 64
                        + ":scale=" + float(2.0).hex()
                        + ":path_integral_grid_edge_v2")
                    row["route_trace"] = [{
                        "p_mode": p_mode,
                        "ppo_path_penalty_points": 0.0 if p_mode == "p0" else 1.25,
                        "ppo_path_penalty_units": "reward_points",
                        "p_map_scale": 2.0,
                        "p_points_per_grid_edge": 1.0,
                        "path_points_m": [[0.0, 0.0], [1.0, 0.0]],
                        "route_cost_units": PENALTY_PATH_UNITS,
                        "route_cost_provenance": route_provenance,
                    }]
                report["records"].append(row)
    return report


def test_fixed_checkout_roots_and_all_policy_rows(tmp_path):
    payload = json.loads(meco_status_payload(tmp_path))
    assert Path(payload["checkpoint_root"]) == tmp_path / "checkpoints/meco-five"
    assert [row["policy_id"] for row in payload["policies"]] == list(
        TRACKS + ABLATIONS + CONTROLS)
    assert payload["policies"][0]["training_state"] == "not_started"
    assert payload["policies"][-1]["training_state"] == "not_applicable"


def test_three_pass_dashboard_reports_campaign_workflow_without_inventing_review_evidence():
    project = Path(__file__).resolve().parents[1]
    provenance = _three_pass_provenance(project)
    assert provenance["status"] == "validated_summary_only"
    assert [row["tranche_id"] for row in provenance["tranches"]] == [
        "dashboard-three-pass-availability", "bounded-multimodel-prediction",
        "d1-d2-multimodel-paired-matrix",
        "three-pass-summary-validation"]
    assert all(row["commits"]["feature_commit"] == row["commits"]["reviewer_approved_commit"]
               for row in provenance["tranches"])
    assert "not full" in " ".join(provenance["limits"])
    source = (project / "frc_defense/dashboard_static/meco.js").read_text()
    assert "Campaign pass status" in source
    assert "does not report agent liveness" in source
    assert "Scoped implementation evidence" in source
    node = shutil.which("node")
    if node:
        subprocess.run([node, "--check", str(project / "frc_defense/dashboard_static/meco.js")],
                       check=True, capture_output=True, text=True)


def test_queued_track_without_manifest_is_visible_without_claiming_liveness(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "checkpoints", tmp_path / "evaluations"
    lane = checkpoint_root / "run-a"
    lane.mkdir(parents=True)
    evaluation_root.mkdir()
    now = 1_800_000_000.
    stamp = "2027-01-15T08:00:00+00:00"
    (checkpoint_root / "queue-status.json").write_text(json.dumps({
        "schema_version": 1, "updated_utc": stamp,
        "phase": "training", "cycle": 3,
        "evaluation": {"active": True, "status": "running", "pid": 101},
        "lanes": [{"track": "A1", "run_id": "r1", "status": "queued",
                   "pid": 99, "device": "cuda:0", "completed_updates": 12,
                   "chunk_updates": 100}],
    }))
    result = _collect_status(checkpoint_root, evaluation_root, now=now,
                             process_probe=lambda _pid: True)
    policy = next(row for row in result["policies"] if row["policy_id"] == "A1")
    assert result["queue"]["state"] == "fresh"
    assert policy["training_state"] == "queued"
    assert policy["run"]["heartbeat"]["state"] == "not_started"
    assert policy["run"]["worker_pid"] == 99
    assert policy["run"]["completed_updates"] == 12
    assert result["queue"]["lanes"]["A1"]["device"] == "cuda:0"
    assert result["queue"]["evaluation"]["pid"] == 101


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
    assert len(report["records"]) == 44
    assert group["record_count"] == 4
    assert group["control_definitions"]["D0"].startswith("production deterministic offense")
    assert set(group["control_definitions"]) == {"D0"}
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
    bad_identity = _report()
    bad_identity["sources"]["simulator"] = "changed"
    bad_policy = _report()
    bad_policy["records"][0]["policy_id"] = "different-weights"
    bad_split = _report()
    bad_split["split"] = "test"
    bad_diagnostic = _report()
    bad_diagnostic["diagnostic"] = "false"
    (evaluation_root / "metrics.json").write_text(json.dumps(bad_metrics))
    (evaluation_root / "provenance.json").write_text(json.dumps(bad_provenance))
    (evaluation_root / "identity.json").write_text(json.dumps(bad_identity))
    (evaluation_root / "policy.json").write_text(json.dumps(bad_policy))
    (evaluation_root / "split.json").write_text(json.dumps(bad_split))
    (evaluation_root / "diagnostic.json").write_text(json.dumps(bad_diagnostic))
    result = _collect_status(checkpoint_root, evaluation_root, now=1_800_000_000.)
    assert result["comparison_groups"] == []


def test_episode_count_and_diagnostic_flag_must_agree(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    falsely_final = _report()
    falsely_final["diagnostic"] = False
    falsely_final["episodes_per_category"] = 100
    (evaluation_root / "false-final.json").write_text(json.dumps(falsely_final))
    wrong_count = _report()
    wrong_count["episodes_per_category"] = 3
    (evaluation_root / "wrong-count.json").write_text(json.dumps(wrong_count))
    result = _collect_status(checkpoint_root, evaluation_root, now=1_800_000_000.)
    assert result["comparison_groups"] == []


def test_evaluator_manifest_reports_live_progress_and_stale_state(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    report = _report(100)
    report["diagnostic"] = False
    report["evaluation_id"] = "eval-live"
    (evaluation_root / "run.json").write_text(json.dumps(report))
    manifest = {"schema_version": 1, "evaluation_id": "eval-live",
        "protocol_id": "proto-a", "split": "validation", "status": "running",
        "pid": 42, "heartbeat_utc": "2027-01-15T08:00:00+00:00",
        "output_file": "run.json", "progress": {"completed_records": 3, "total_records": 2200}}
    status_path = evaluation_root / "run.json.status.json"
    status_path.write_text(json.dumps(manifest))
    live = _collect_status(checkpoint_root, evaluation_root, now=1_800_000_000.,
                           process_probe=lambda _pid: True)
    assert live["evaluation_runs"][0]["status"] == "running"
    assert live["evaluation_runs"][0]["progress"]["completed_records"] == 3
    stale = _collect_status(checkpoint_root, evaluation_root, now=1_800_001_000.,
                            process_probe=lambda _pid: False)
    assert stale["evaluation_runs"][0]["status"] == "stale/unconfirmed"


def test_evaluator_complete_requires_valid_report(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    report = _report(100)
    report.update(evaluation_id="eval-complete", diagnostic=False)
    # This complete-scale report is invalid only because one record is missing
    # from its otherwise fully matched matrix.
    report["records"].pop()
    (evaluation_root / "report.json").write_text(json.dumps(report))
    status = {"schema_version": 1, "evaluation_id": "eval-complete",
        "protocol_id": "proto-a", "split": "validation", "status": "complete",
        "pid": 42, "heartbeat_utc": "2027-01-15T08:00:00+00:00",
        "output_file": "report.json", "progress": {"completed_records": 2199, "total_records": 2200}}
    (evaluation_root / "report.json.status.json").write_text(json.dumps(status))
    result = _collect_status(checkpoint_root, evaluation_root, now=1_800_000_000.,
                             process_probe=lambda _pid: False)
    assert result["evaluation_runs"][0]["status"] == "invalid"
    assert result["evaluation_runs"][0]["error"]


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


def test_three_pass_summary_manifest_rejects_artifact_tampering_and_unsafe_paths(tmp_path, monkeypatch):
    import frc_defense.meco_dashboard as dashboard
    project = Path(__file__).resolve().parents[1]
    docs = tmp_path / "docs"
    shutil.copytree(project / "docs", docs)
    manifest_path = docs / "MECO_THREE_PASS_PROVENANCE.json"
    data = json.loads(manifest_path.read_text())

    # A modified artifact is rejected even when the test pins the manifest digest.
    record = data["tranches"][0]["summaries"]["planner"]
    (tmp_path / record["path"]).write_text("tampered\n")
    monkeypatch.setattr(dashboard, "_EXPECTED_THREE_PASS_PROVENANCE_SHA256",
                        hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    result = dashboard._three_pass_provenance(tmp_path)
    assert result["status"] == "untrusted"
    assert "hash mismatch" in result["untrusted_reason"]

    # Re-pin a manifest containing a traversal path; it must fail closed.
    shutil.copy(project / record["path"], tmp_path / record["path"])
    data["tranches"][0]["summaries"]["planner"]["path"] = "../outside.md"
    manifest_path.write_text(json.dumps(data, indent=2) + "\n")
    monkeypatch.setattr(dashboard, "_EXPECTED_THREE_PASS_PROVENANCE_SHA256",
                        hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    result = dashboard._three_pass_provenance(tmp_path)
    assert result["status"] == "untrusted"
    assert "path or digest is invalid" in result["untrusted_reason"]


def test_three_pass_summary_manifest_rejects_unmatched_reviewer_commit(tmp_path, monkeypatch):
    import frc_defense.meco_dashboard as dashboard
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / "docs", tmp_path / "docs")
    manifest_path = tmp_path / "docs/MECO_THREE_PASS_PROVENANCE.json"
    data = json.loads(manifest_path.read_text())
    data["tranches"][0]["commits"]["reviewer_approved_commit"] = "0" * 40
    manifest_path.write_text(json.dumps(data, indent=2) + "\n")
    monkeypatch.setattr(dashboard, "_EXPECTED_THREE_PASS_PROVENANCE_SHA256",
                        hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    result = dashboard._three_pass_provenance(tmp_path)
    assert result["status"] == "untrusted"
    assert "must match exact commit ids" in result["untrusted_reason"]


def test_three_pass_summary_manifest_rejects_nonexistent_commit_object(tmp_path, monkeypatch):
    import frc_defense.meco_dashboard as dashboard
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / "docs", tmp_path / "docs")
    manifest_path = tmp_path / "docs/MECO_THREE_PASS_PROVENANCE.json"
    data = json.loads(manifest_path.read_text())
    data["tranches"][0]["commits"] = {
        "feature_commit": "0" * 40,
        "reviewer_approved_commit": "0" * 40,
    }
    manifest_path.write_text(json.dumps(data, indent=2) + "\n")
    monkeypatch.setattr(dashboard, "_EXPECTED_THREE_PASS_PROVENANCE_SHA256",
                        hashlib.sha256(manifest_path.read_bytes()).hexdigest())

    # A poisoned PATH must not let a fake git approve nonexistent commit IDs.
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nexit 0\n")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    result = dashboard._three_pass_provenance(tmp_path)
    assert result["status"] == "untrusted"
    assert "not present as a commit object" in result["untrusted_reason"]


def test_three_pass_summary_manifest_rejects_symlinked_artifact(tmp_path):
    import frc_defense.meco_dashboard as dashboard
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project / "docs", tmp_path / "docs")
    manifest = json.loads((tmp_path / "docs/MECO_THREE_PASS_PROVENANCE.json").read_text())
    relpath = manifest["tranches"][0]["summaries"]["planner"]["path"]
    artifact = tmp_path / relpath
    target = tmp_path / "external-summary.md"
    target.write_text(artifact.read_text())
    artifact.unlink()
    artifact.symlink_to(target)
    result = dashboard._three_pass_provenance(tmp_path)
    assert result["status"] == "untrusted"
    assert "symlinked" in result["untrusted_reason"]
