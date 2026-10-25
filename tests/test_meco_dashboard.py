import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from frc_defense.meco_dashboard import (
    ABLATIONS, CONTROLS, TRACKS, _collect_status, _qualification_status,
    _copy_dashboard_comparison_summary, _read_evaluation_comparison,
    _write_dashboard_comparison_summary, _queue_attempt_projection,
    _queue_state, _training_curriculum_projection,
    meco_status_payload,
)
from frc_defense.meco_path_penalty import PLAN_A_PPO_RECOMPUTATION_CONTRACT
from frc_defense.meco_qualification import (CALIBRATED_PROTOCOL_VERSION,
    D0_PAIRED_REFERENCE_PROTOCOL_VERSION, LEGACY_CALIBRATED_PROTOCOL_VERSION)
from frc_defense.training_curriculum import SCENARIOS, _ancestors


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
    for policy in ("A1", "A2", "A3"):
        policy_manifest[policy]["p_mode"] = "learned"
    for policy in ABLATIONS:
        policy_manifest[policy]["p_mode"] = "p0"
    control_ids = {"D0": "D0_teacher_proxy", "D1": "D1_hand_weighted",
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
        "control_definitions": {
            "D0": "production deterministic offense controller; native 20 ms physics-step cadence; actual executed actions recorded; not constrained by the MECO candidate table",
            "D1": "hand-weighted scorer over the shared feasible candidate table",
            "D2": "D1 scorer with kinematic opponent endpoint prediction"},
        "plan_a_penalty": {
            "trained_mode": "learned", "p0_ablation_mode": "p0",
            "cost_scale": 2.0, "points_per_grid_edge": 1.0,
            "projection": "mean_over_legal_heading_bins_xy_grid_v1",
            "map_units": "dimensionless_additive_penalty_per_cardinal_grid_step",
            "path_units": "normalized_grid_edge_units",
            "ppo_recomputation": PLAN_A_PPO_RECOMPUTATION_CONTRACT},
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
                if policy in ("A1", "A2", "A3") + ABLATIONS:
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
                        "route_cost_units": "normalized_grid_edge_units",
                        "route_cost_provenance": route_provenance,
                    }]
                report["records"].append(row)
    return report


def test_fixed_checkout_roots_and_all_policy_rows(tmp_path):
    payload = json.loads(meco_status_payload(tmp_path))
    assert Path(payload["checkpoint_root"]) == tmp_path / "checkpoints/meco-five"
    assert Path(payload["evaluation_root"]) == tmp_path / "evaluations/meco-five"
    assert [row["policy_id"] for row in payload["policies"]] == list(
        TRACKS + ABLATIONS + CONTROLS)
    assert payload["policies"][0]["training_state"] == "not_started"
    assert payload["policies"][-1]["training_state"] == "not_applicable"


def test_direct_campaign_root_reads_track_runs_queue_scheduler_and_evaluations(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    evaluation_root = tmp_path / "evaluations"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    for track in TRACKS:
        run_dir = checkpoint_root / track
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": f"run-{track}", "track": track, "status": "complete",
            "completed_updates": 100}))

    now = datetime.now(timezone.utc)
    scenario = "offense_vs_scripted_pressure"
    category = SCENARIOS[scenario].required_categories[0]
    previous = {"cycle": 1, "scenario_id": scenario, "category": category,
        "seed": 17, "assignment_id": "previous-1", "sampling_role": "stage_exposure"}
    assignment = {"cycle": 2, "scenario_id": scenario, "category": category,
        "seed": 18, "assignment_id": "pending-2", "sampling_role": "stage_exposure"}
    counts = {"exposure_chunks": {scenario: 1},
        "category_exposures": {f"{scenario}/{category}": 1},
        "recovery_pressure": {}, "phase4_exposure_counts": {}}
    scheduler = {"schema_version": 1, "completed_cycle": 1,
        "pending_assignment": assignment, **counts,
        "history": [{"assignment": previous}]}
    (tmp_path / "task-scheduler-state.json").write_text(json.dumps(scheduler))
    queue = {"schema_version": 1, "updated_utc": now.isoformat(), "cycle": 2,
        "task_assignment": assignment,
        "task_scheduler": {"completed_cycle": 1,
            "pending_assignment": assignment, **counts},
        "lanes": [_queue_lane(track) for track in TRACKS]}
    (tmp_path / "queue-status.json").write_text(json.dumps(queue))
    (evaluation_root / "matched.json").write_text(json.dumps(_report()))

    payload = json.loads(meco_status_payload(tmp_path))

    assert Path(payload["checkpoint_root"]) == checkpoint_root
    assert Path(payload["evaluation_root"]) == evaluation_root
    assert payload["queue"]["state"] == "fresh"
    assert payload["queue"]["curriculum"]["state"] == "fresh"
    assert payload["queue"]["curriculum"]["assignment"]["scenario_id"] == scenario
    assert len(payload["comparison_groups"]) == 1
    for track in TRACKS:
        row = next(policy for policy in payload["policies"]
                   if policy["policy_id"] == track)
        assert row["training_state"] == "complete"
        assert row["run"]["completed_updates"] == 100


def test_large_report_uses_digest_bound_strict_comparison_summary(tmp_path, monkeypatch):
    import frc_defense.meco_dashboard as dashboard

    monkeypatch.setattr(dashboard, "MAX_DASHBOARD_INLINE_REPORT_BYTES", 1)
    report = _report()
    source = tmp_path / "evaluation.json"
    source.write_text(json.dumps(report, separators=(",", ":")))
    assert _write_dashboard_comparison_summary(source, report)

    def bind_status(report_path):
        summary_bytes = Path(str(report_path) + ".dashboard-summary.json").read_bytes()
        Path(str(report_path) + ".status.json").write_text(json.dumps({
            "schema_version": 1, "evaluation_id": report["evaluation_id"],
            "protocol_id": report["protocol_id"], "split": "validation",
            "status": "complete", "output_file": report_path.name,
            "dashboard_summary_sha256": hashlib.sha256(summary_bytes).hexdigest()}))

    bind_status(source)

    comparison, binding, error, visible = _read_evaluation_comparison(
        source, tmp_path, {"files": 0, "summary_bytes": 0,
                           "raw_bytes": 0, "records": 0})

    assert error is None and visible
    assert comparison["policy_metrics"] == dashboard._comparison(report)["policy_metrics"]
    assert binding == {"report_schema_version": 1, "evaluation_id": "eval-001",
        "protocol_id": "proto-a", "split": "validation",
        "source_identity_sha256": report["source_identity_sha256"], "record_count": 44}

    summary_path = Path(str(source) + ".dashboard-summary.json")
    forged = json.loads(summary_path.read_text())
    forged["comparison"]["policy_metrics"]["A1"]["open"]["success"] = "forged"
    forged["comparison_sha256"] = hashlib.sha256(
        dashboard._dashboard_summary_json_bytes(forged["comparison"])).hexdigest()
    summary_path.write_text(json.dumps(forged, separators=(",", ":")))
    unbound, _, unbound_error, _ = _read_evaluation_comparison(
        source, tmp_path, {"files": 0, "summary_bytes": 0,
                           "raw_bytes": 0, "records": 0})
    assert unbound is None and unbound_error == "dashboard summary is unavailable or invalid"
    bind_status(source)
    rejected, _, forged_error, _ = _read_evaluation_comparison(
        source, tmp_path, {"files": 0, "summary_bytes": 0,
                           "raw_bytes": 0, "records": 0})
    assert rejected is None and forged_error == "dashboard summary is unavailable or invalid"

    for mutate in (
        lambda projection: projection["policy_metrics"]["A1"]["open"].pop("success"),
        lambda projection: projection["policy_metrics"]["A1"]["open"]
            ["collection_action_bouts"].update(count=-1, acquired_count=-1),
    ):
        assert _write_dashboard_comparison_summary(source, report)
        malformed_sidecar = json.loads(summary_path.read_text())
        mutate(malformed_sidecar["comparison"])
        malformed_sidecar["comparison_sha256"] = hashlib.sha256(
            dashboard._dashboard_summary_json_bytes(
                malformed_sidecar["comparison"])).hexdigest()
        summary_path.write_text(json.dumps(malformed_sidecar, separators=(",", ":")))
        bind_status(source)
        malformed, _, malformed_error, _ = _read_evaluation_comparison(
            source, tmp_path, {"files": 0, "summary_bytes": 0,
                               "raw_bytes": 0, "records": 0})
        assert malformed is None and malformed_error == "dashboard summary is unavailable or invalid"

    published = tmp_path / "scheduled-evaluation.json"
    published.write_bytes(source.read_bytes())
    assert not _copy_dashboard_comparison_summary(source, published)
    assert _write_dashboard_comparison_summary(source, report)
    bind_status(source)
    assert _copy_dashboard_comparison_summary(source, published)
    bind_status(published)
    copied, _, copy_error, _ = _read_evaluation_comparison(
        published, tmp_path, {"files": 0, "summary_bytes": 0,
                            "raw_bytes": 0, "records": 0})
    assert copy_error is None and copied == comparison

    source.write_bytes(source.read_bytes() + b" ")
    stale, _, stale_error, _ = _read_evaluation_comparison(
        source, tmp_path, {"files": 0, "summary_bytes": 0,
                           "raw_bytes": 0, "records": 0})
    assert stale is None and stale_error == "dashboard summary is unavailable or invalid"


def test_incomplete_report_cannot_produce_dashboard_summary(tmp_path):
    report = _report()
    report["records"].pop()
    source = tmp_path / "incomplete.json"
    source.write_text(json.dumps(report))
    assert not _write_dashboard_comparison_summary(source, report)
    assert not Path(str(source) + ".dashboard-summary.json").exists()


def test_mixed_metric_ledger_versions_allow_partial_unsuccessful_coverage(tmp_path, monkeypatch):
    import frc_defense.meco_dashboard as dashboard

    monkeypatch.setattr(dashboard, "MAX_DASHBOARD_INLINE_REPORT_BYTES", 1)
    report = _report(episodes=2)
    rows = [row for row in report["records"]
            if row["policy"] == "A1" and row["category"] == "open"]
    rows[0]["meco_metric_ledger"] = {
        "schema_version": 3, "availability": {"collection_action_bouts": True},
        "collection_action_bouts": {"count": 2, "acquired_count": 1,
                                     "acquired_proportion": .5}}
    rows[1]["meco_metric_ledger"] = {
        "schema_version": 4, "availability": {"collection_action_bouts": True},
        "collection_action_bouts": {"count": 3, "acquired_count": 2,
            "unsuccessful_collection_action_bouts": 1,
            "acquired_proportion": 2 / 3}}
    expected = dashboard._comparison(report)
    assert expected["policy_metrics"]["A1"]["open"]["collection_action_bouts"] == {
        "available_records": 2, "count": 5, "acquired_count": 3,
        "unsuccessful_collection_action_bouts": 1,
        "unsuccessful_collection_action_bouts_available_records": 1,
        "unsuccessful_collection_action_bouts_total_records": 2,
        "acquired_proportion": .6,
        "semantics": "unsuccessful_collection_action_bouts counts finalized focal collection-action bouts with authoritative acquired quantity known to be zero; it is a descriptive bout outcome, not causal intake failure"}
    source = tmp_path / "mixed-ledger.json"
    source.write_text(json.dumps(report, separators=(",", ":")))
    assert _write_dashboard_comparison_summary(source, report)
    summary_bytes = Path(str(source) + ".dashboard-summary.json").read_bytes()
    Path(str(source) + ".status.json").write_text(json.dumps({
        "evaluation_id": report["evaluation_id"], "protocol_id": report["protocol_id"],
        "split": report["split"], "status": "complete", "output_file": source.name,
        "dashboard_summary_sha256": hashlib.sha256(summary_bytes).hexdigest()}))

    actual, _, error, _ = _read_evaluation_comparison(
        source, tmp_path, {"files": 0, "summary_bytes": 0,
                           "raw_bytes": 0, "records": 0})

    assert error is None
    assert actual["policy_metrics"] == expected["policy_metrics"]


def test_summary_writer_concurrency_is_bounded(tmp_path):
    import frc_defense.meco_dashboard as dashboard

    report = _report()
    source = tmp_path / "evaluation.json"
    source.write_text(json.dumps(report))
    assert dashboard._DASHBOARD_SUMMARY_WRITES.acquire(blocking=False)
    try:
        assert not _write_dashboard_comparison_summary(source, report)
    finally:
        dashboard._DASHBOARD_SUMMARY_WRITES.release()


def test_direct_campaign_layout_rejects_partial_track_directories(tmp_path):
    (tmp_path / "checkpoints" / "A1").mkdir(parents=True)

    with pytest.raises(ValueError, match="real, non-symlink in-root track directories"):
        meco_status_payload(tmp_path)


def test_direct_campaign_layout_rejects_in_root_track_symlink(tmp_path):
    direct_root = tmp_path / "checkpoints"
    direct_root.mkdir()
    (direct_root / "A1-target").mkdir()
    (direct_root / "A1").symlink_to(direct_root / "A1-target", target_is_directory=True)
    for track in TRACKS[1:]:
        (direct_root / track).mkdir()

    with pytest.raises(ValueError, match="real, non-symlink in-root"):
        meco_status_payload(tmp_path)


def test_direct_campaign_layout_rejects_in_root_checkpoints_parent_symlink(tmp_path):
    target = tmp_path / "checkpoints-target"
    for track in TRACKS:
        (target / track).mkdir(parents=True)
    (tmp_path / "checkpoints").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="checkpoints root must be a real in-root directory"):
        meco_status_payload(tmp_path)


def test_direct_campaign_layout_rejects_in_root_evaluations_root_symlink(tmp_path):
    direct_root = tmp_path / "checkpoints"
    for track in TRACKS:
        (direct_root / track).mkdir(parents=True)
    evaluations_target = tmp_path / "evaluations-target"
    evaluations_target.mkdir()
    (tmp_path / "evaluations").symlink_to(evaluations_target, target_is_directory=True)

    with pytest.raises(ValueError, match="evaluations root must be a real in-root directory"):
        meco_status_payload(tmp_path)


def test_direct_campaign_layout_rejects_ambiguity_with_legacy_roots(tmp_path):
    direct_root = tmp_path / "checkpoints"
    for track in TRACKS:
        (direct_root / track).mkdir(parents=True)
    (direct_root / "meco-five").mkdir()

    with pytest.raises(ValueError, match="ambiguous legacy and direct"):
        meco_status_payload(tmp_path)


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
        "lanes": ([{"track": "A1", "run_id": "r1", "run_directory": "run-a",
                    "status": "queued", "pid": 99, "device": "cuda:0",
                    "completed_updates": 12, "chunk_updates": 100}] +
                  [{"track": track, "run_id": f"r-{track}",
                    "run_directory": track, "status": "complete", "pid": None,
                    "device": "cuda:0", "completed_updates": 0,
                    "chunk_updates": 100} for track in TRACKS if track != "A1"]),
    }))
    result = _collect_status(checkpoint_root, evaluation_root, now=now,
                             process_probe=lambda _pid: True)
    policy = next(row for row in result["policies"] if row["policy_id"] == "A1")
    assert result["queue"]["state"] == "fresh"
    assert policy["training_state"] == "queued"
    assert policy["run"]["heartbeat"]["state"] == "not_started"
    assert policy["run"]["worker_pid"] is None
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
    assert report["control_definitions"]["D0"].startswith("production deterministic offense")
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


@pytest.mark.parametrize("version,gate_status", [
    (LEGACY_CALIBRATED_PROTOCOL_VERSION, "calibrated_baseline_frozen"),
    (D0_PAIRED_REFERENCE_PROTOCOL_VERSION,
     "d0_operational_margin_frozen_reference_variance_unverified"),
    (CALIBRATED_PROTOCOL_VERSION,
     "matched_reference_variance_and_reviewed_margin_frozen"),
])
def test_qualification_status_accepts_version_matched_calibration(version, gate_status, tmp_path):
    scenario = "pickup_simple"
    closure = {scenario, *_ancestors(scenario)}
    calibration_ids = {name: hashlib.sha256(name.encode()).hexdigest() for name in closure}
    calibration_id = hashlib.sha256(json.dumps(calibration_ids, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    status = {"schema_version": 1, "split": "validation", "scenario_id": scenario,
        "protocol_id": "a" * 64, "bundle_id": "b" * 64, "checkpoint_sha256": None,
        "confidence_intervals": {}, "promotion": {"passed": True},
        "protocol_version": version, "gate_calibration_status": gate_status,
        "safety_gate_status": "metrics_schema_verified", "calibration_id": calibration_id,
        "calibration_ids": calibration_ids,
        "retention_scenario_ids": [name for name in SCENARIOS if name in closure and name != scenario]}
    root = tmp_path / "qualifications" / "meco"
    root.mkdir(parents=True)
    (root / "qualification-status.json").write_text(json.dumps(status))
    errors = []
    projected = _qualification_status(tmp_path, errors, tmp_path)
    assert errors == []
    assert projected["protocol_version"] == version
    assert projected["gate_calibration_status"] == gate_status


def test_v4_qualification_status_rejects_legacy_gate_label(tmp_path):
    status = {"schema_version": 1, "split": "validation", "scenario_id": "pickup_simple",
        "protocol_id": "a" * 64, "bundle_id": "b" * 64, "checkpoint_sha256": None,
        "confidence_intervals": {}, "promotion": {"passed": True},
        "protocol_version": CALIBRATED_PROTOCOL_VERSION,
        "gate_calibration_status": "calibrated_baseline_frozen",
        "safety_gate_status": "metrics_schema_verified", "calibration_id": "c" * 64,
        "calibration_ids": {"pickup_simple": "d" * 64}, "retention_scenario_ids": []}
    root = tmp_path / "qualifications" / "meco"
    root.mkdir(parents=True)
    (root / "qualification-status.json").write_text(json.dumps(status))
    errors = []
    assert _qualification_status(tmp_path, errors, tmp_path) is None
    assert errors and "invalid provenance" in errors[0]["error"]


def test_evaluator_complete_requires_valid_report(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    report = _report(100)
    report.update(evaluation_id="eval-complete", diagnostic=False)
    # A complete non-diagnostic report is invalid only because one record is
    # missing from its otherwise fully matched 100-seed matrix.
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


def test_diagnostic_status_completion_stays_separate_from_unbound_reports(tmp_path, monkeypatch):
    import frc_defense.meco_dashboard as dashboard

    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    monkeypatch.setattr(dashboard, "MAX_DASHBOARD_INLINE_REPORT_BYTES", 32)
    stable_hash = dashboard._stable_file_sha256
    def forbid_unbound_report_read(path, *args, **kwargs):
        if Path(path).name.startswith("diagnostic-"):
            raise AssertionError("unbound report bytes must not be read for identity")
        return stable_hash(path, *args, **kwargs)
    monkeypatch.setattr(dashboard, "_stable_file_sha256", forbid_unbound_report_read)
    for index in range(5):
        output_name = f"diagnostic-{index}.json"
        # The oversized report contains a misleading header, but without a
        # summary binding its bytes are never read for report identity.
        (evaluation_root / output_name).write_bytes(
            b'{"evaluation_id":"header-id","protocol_id":"header-protocol"}' + b" " * 4096)
        (evaluation_root / f"{output_name}.status.json").write_text(json.dumps({
            "schema_version": 1, "evaluation_id": f"status-eval-{index}",
            "protocol_id": f"status-protocol-{index}", "split": "diagnostic",
            "status": "complete", "pid": 999999,
            "heartbeat_utc": "2027-01-15T08:00:00+00:00",
            "output_file": output_name,
            "progress": {"completed_records": 2200, "total_records": 2200}}))

    result = _collect_status(checkpoint_root, evaluation_root,
        project_root=tmp_path, now=1_800_000_000., process_probe=lambda _pid: False)

    assert len(result["evaluation_runs"]) == 5
    for row in result["evaluation_runs"]:
        assert row["execution_status"] == "status-reported complete"
        assert row["status"] == "status-reported complete"
        assert row["diagnostic"] is True
        assert row["report_validation"] == "unvalidated"
        assert row["evidence_error"] == "oversized report lacks evaluator summary binding"
        assert row["evaluation_id"].startswith("status-eval-")
        assert "report_identity" not in row
    assert result["comparison_groups"] == []
    assert result["qualification"] is None
    assert sum(error["error"] == "oversized report lacks evaluator summary binding"
               for error in result["errors"]) == 5


def test_diagnostic_status_cannot_promote_report_into_comparison(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    report = _report(100)
    report["evaluation_id"] = "diag-status-eval"
    report_path = evaluation_root / "diagnostic.json"
    report_path.write_text(json.dumps(report))
    (evaluation_root / "diagnostic.json.status.json").write_text(json.dumps({
        "schema_version": 1, "evaluation_id": "diag-status-eval",
        "protocol_id": "proto-a", "split": "diagnostic", "status": "complete",
        "pid": 999999, "heartbeat_utc": "2027-01-15T08:00:00+00:00",
        "output_file": report_path.name,
        "progress": {"completed_records": 2200, "total_records": 2200}}))

    result = _collect_status(checkpoint_root, evaluation_root,
        project_root=tmp_path, now=1_800_000_000., process_probe=lambda _pid: False)

    row = result["evaluation_runs"][0]
    assert row["execution_status"] == "status-reported complete"
    assert row["diagnostic"] is True
    assert row["report_validation"] == "invalid"
    assert row["evidence_error"] == "evaluator status/report provenance mismatch"
    assert result["comparison_groups"] == []
    assert result["qualification"] is None


def test_diagnostic_completion_survives_missing_report_but_not_bad_status_identity(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    base = {"schema_version": 1, "evaluation_id": "missing-report-eval",
        "protocol_id": "missing-report-protocol", "split": "diagnostic",
        "status": "complete", "pid": 999999,
        "heartbeat_utc": "2027-01-15T08:00:00+00:00",
        "output_file": "missing.json"}
    (evaluation_root / "missing.json.status.json").write_text(json.dumps(base))
    bad_id = dict(base, evaluation_id="")
    bad_id["output_file"] = "bad-id.json"
    (evaluation_root / "bad-id.json.status.json").write_text(json.dumps(bad_id))
    bad_split = dict(base, evaluation_id="bad-split-eval", split="unknown")
    bad_split["output_file"] = "bad-split.json"
    (evaluation_root / "bad-split.json.status.json").write_text(json.dumps(bad_split))

    result = _collect_status(checkpoint_root, evaluation_root,
        project_root=tmp_path, now=1_800_000_000., process_probe=lambda _pid: False)

    by_id = {row["evaluation_id"]: row for row in result["evaluation_runs"]}
    missing = by_id["missing-report-eval"]
    assert missing["execution_status"] == "status-reported complete"
    assert missing["diagnostic"] is True
    assert missing["report_validation"] == "unavailable"
    assert missing["evidence_error"] == "evaluation report is unavailable"
    assert by_id[""]["execution_status"] == "invalid"
    assert by_id[""]["diagnostic"] is False
    assert by_id["bad-split-eval"]["execution_status"] == "invalid"
    assert by_id["bad-split-eval"]["diagnostic"] is False


def test_bound_report_validation_remains_independent_from_execution_lifecycle(tmp_path, monkeypatch):
    import frc_defense.meco_dashboard as dashboard

    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    monkeypatch.setattr(dashboard, "MAX_DASHBOARD_INLINE_REPORT_BYTES", 1)
    report = _report(100)
    report_path = evaluation_root / "bound.json"
    report_path.write_text(json.dumps(report, separators=(",", ":")))
    assert _write_dashboard_comparison_summary(report_path, report)
    summary = Path(str(report_path) + ".dashboard-summary.json").read_bytes()
    (evaluation_root / "bound.json.status.json").write_text(json.dumps({
        "schema_version": 1, "evaluation_id": report["evaluation_id"],
        "protocol_id": report["protocol_id"], "split": "validation",
        "status": "complete", "pid": 999999,
        "heartbeat_utc": "2027-01-15T08:00:00+00:00",
        "output_file": report_path.name,
        "dashboard_summary_sha256": hashlib.sha256(summary).hexdigest()}))

    result = _collect_status(checkpoint_root, evaluation_root,
        project_root=tmp_path, now=1_800_000_000., process_probe=lambda _pid: False)

    row = result["evaluation_runs"][0]
    assert row["execution_status"] == "complete"
    assert row["report_validation"] == "validated"
    assert row["evidence_error"] is None
    assert len(result["comparison_groups"]) == 1


@pytest.mark.parametrize("mismatch", ["digest", "evaluation_id", "protocol_id", "split"])
def test_bounded_report_binding_mismatches_remain_evidence_invalid(
        tmp_path, monkeypatch, mismatch):
    import frc_defense.meco_dashboard as dashboard

    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    monkeypatch.setattr(dashboard, "MAX_DASHBOARD_INLINE_REPORT_BYTES", 1)
    report = _report(100)
    report_path = evaluation_root / f"{mismatch}.json"
    report_path.write_text(json.dumps(report, separators=(",", ":")))
    assert _write_dashboard_comparison_summary(report_path, report)
    summary = Path(str(report_path) + ".dashboard-summary.json").read_bytes()
    status = {"schema_version": 1, "evaluation_id": report["evaluation_id"],
        "protocol_id": report["protocol_id"], "split": "validation", "status": "complete",
        "pid": 999999, "heartbeat_utc": "2027-01-15T08:00:00+00:00",
        "output_file": report_path.name,
        "dashboard_summary_sha256": hashlib.sha256(summary).hexdigest()}
    if mismatch == "digest":
        status["dashboard_summary_sha256"] = "f" * 64
    elif mismatch == "evaluation_id":
        status["evaluation_id"] = "different-evaluation"
    elif mismatch == "protocol_id":
        status["protocol_id"] = "different-protocol"
    else:
        status["split"] = "test"
    (evaluation_root / f"{report_path.name}.status.json").write_text(json.dumps(status))

    result = _collect_status(checkpoint_root, evaluation_root,
        project_root=tmp_path, now=1_800_000_000., process_probe=lambda _pid: False)

    row = result["evaluation_runs"][0]
    assert row["execution_status"] == "invalid"
    assert row["report_validation"] == "invalid"
    assert row["evidence_error"]
    assert result["comparison_groups"] == []


def test_invalid_or_absent_status_never_projects_completion_or_report_identity(tmp_path,
                                                                               monkeypatch):
    import frc_defense.meco_dashboard as dashboard

    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    monkeypatch.setattr(dashboard, "MAX_DASHBOARD_INLINE_REPORT_BYTES", 32)
    invalid = {"schema_version": 1, "evaluation_id": "bad-eval",
        "protocol_id": "bad-protocol", "split": "diagnostic", "status": "complete",
        "pid": 999999, "heartbeat_utc": "2027-01-15T08:00:00+00:00",
        "output_file": "../outside.json"}
    (evaluation_root / "unsafe.json.status.json").write_text(json.dumps(invalid))
    (evaluation_root / "malformed.json.status.json").write_text("{")
    (evaluation_root / "absent-status.json").write_bytes(
        b'{"evaluation_id":"report-header-id"}' + b" " * 4096)

    result = _collect_status(checkpoint_root, evaluation_root,
        project_root=tmp_path, now=1_800_000_000., process_probe=lambda _pid: False)

    by_id = {row["evaluation_id"]: row for row in result["evaluation_runs"]}
    unsafe = by_id["bad-eval"]
    malformed = next(row for row in result["evaluation_runs"]
                     if row["evaluation_id"] is None)
    assert unsafe["execution_status"] == "invalid"
    assert unsafe["diagnostic"] is False
    assert unsafe["evidence_error"] == "evaluator output file is not a safe in-root basename"
    assert malformed["execution_status"] == "unconfirmed/invalid"
    assert malformed["evaluation_id"] is None
    assert malformed["evidence_error"]
    assert result["comparison_groups"] == []
    assert all("report-header-id" not in str(row) for row in result["evaluation_runs"])


def test_running_diagnostic_status_remains_running_while_report_is_pending(tmp_path):
    checkpoint_root, evaluation_root = tmp_path / "ckpt", tmp_path / "eval"
    checkpoint_root.mkdir()
    evaluation_root.mkdir()
    (evaluation_root / "running.json.status.json").write_text(json.dumps({
        "schema_version": 1, "evaluation_id": "running-eval",
        "protocol_id": "running-protocol", "split": "diagnostic", "status": "running",
        "pid": 42, "heartbeat_utc": "2027-01-15T08:00:00+00:00",
        "output_file": "running.json",
        "progress": {"completed_records": 3, "total_records": 2200}}))

    result = _collect_status(checkpoint_root, evaluation_root,
        project_root=tmp_path, now=1_800_000_000., process_probe=lambda _pid: True)

    row = result["evaluation_runs"][0]
    assert row["execution_status"] == "running"
    assert row["diagnostic"] is False
    assert row["report_validation"] == "pending"
    assert row["evidence_error"] is None


def test_evaluator_card_labels_status_claim_and_report_validation_separately():
    source = Path("frc_defense/dashboard_static/meco.js").read_text()
    assert 'run.execution_status || run.status' in source
    assert '"status-reported complete · diagnostic · report unvalidated"' in source
    assert 'Status-file evaluation ${esc(run.evaluation_id)}' in source
    assert 'Report validation ${esc(run.report_validation || "unconfirmed")}' in source
    assert '${esc(evidenceError)}' in source


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


def _queue_lane(track):
    return {"track": track, "run_id": f"run-{track}",
        "run_directory": track, "device": "cuda:0", "status": "complete",
        "pid": None, "completed_updates": 0, "chunk_updates": 100}


def _read_queue_for_lanes(tmp_path, lanes, *, updated_utc=None, **extra):
    now = datetime.now(timezone.utc)
    root = tmp_path / "checkpoints"
    root.mkdir(parents=True)
    (root / "queue-status.json").write_text(json.dumps({
        "schema_version": 1, "updated_utc": updated_utc or now.isoformat(),
        "lanes": lanes, **extra}))
    return _queue_state(root, now.timestamp(), tmp_path)


def _diagnostic_monitor_summary():
    pending = {"frontier_id": "cycle-4-update-400", "cycle": 4, "update": 400,
        "status": "running", "checkpoint_snapshots": [{"path": "/private/A1.pt"}],
        "attempts": [{"attempt": 2, "status": "running",
                      "output_path": "/private/report.json"}],
        "result": {"evaluation_id": "diag-eval-4", "report_sha256": "c" * 64,
            "status_sha256": "d" * 64, "diagnostic_only": True,
            "qualification_eligible": False, "status_path": "/private/status.json"}}
    return {"enabled": True, "diagnostic_only": True,
        "qualification_eligible": False, "journal_sha256": "b" * 64,
        "journal_events": 12, "completed_count": 3,
        "active_frontier": pending, "queued_frontier": None,
        "pending_frontier": pending,
        "skipped_frontiers": [{"frontier_id": "old-frontier", "cycle": 2,
            "update": 200, "superseded_by": "new-frontier"}],
        "missed_frontiers": [{"cycle": 1, "update": 100, "reason": "frontier unavailable"}]}


def test_queue_projects_fresh_diagnostic_monitor_with_allowlisted_frontiers(tmp_path):
    monitor = _diagnostic_monitor_summary()
    schedule = {"schedule_count": 5, "completed_count": 1,
        "pending_count": 2, "failed_count": 0}
    result = _read_queue_for_lanes(tmp_path, [_queue_lane(t) for t in TRACKS],
        diagnostic_monitor=monitor, evaluation_schedule=schedule,
        evaluation={"status": "running", "pid": 41,
            "diagnostic_monitor": monitor,
            "diagnostic_monitor_error": "nested monitor failure"})

    projected = result["diagnostic_monitor"]
    assert result["state"] == "fresh"
    assert projected["state"] == "fresh"
    assert projected["configured"] is True and projected["enabled"] is True
    assert projected["diagnostic_only"] is True
    assert projected["qualification_eligible"] is False
    assert projected["completed_count"] == 3
    assert projected["pending_frontier"] == {
        "frontier_id": "cycle-4-update-400", "cycle": 4,
        "update": 400, "status": "running", "attempt": 2,
        "result": {"evaluation_id": "diag-eval-4", "report_sha256": "c" * 64,
            "status_sha256": "d" * 64, "diagnostic_only": True,
            "qualification_eligible": False}}
    assert projected["skipped_count"] == 1
    assert projected["skipped_frontiers"][0]["status"] == "skipped"
    assert projected["missed_count"] == 1
    assert "checkpoint_snapshots" not in json.dumps(projected)
    assert "/private" not in json.dumps(projected)
    assert result["evaluation_schedule"] == schedule
    assert result["evaluation"] == {"status": "running", "pid": 41}


def test_queue_absent_diagnostic_monitor_is_not_configured(tmp_path):
    result = _read_queue_for_lanes(tmp_path, [_queue_lane(t) for t in TRACKS])

    projected = result["diagnostic_monitor"]
    assert projected["state"] == "not_configured"
    assert projected["configured"] is False
    assert projected["pending_frontier"] is None
    assert projected["diagnostic_only"] is True
    assert projected["qualification_eligible"] is False


def test_queue_malformed_diagnostic_monitor_fails_closed_and_bounds_error(tmp_path):
    error = "E" * 500
    result = _read_queue_for_lanes(tmp_path, [_queue_lane(t) for t in TRACKS],
        diagnostic_monitor={"enabled": True, "diagnostic_only": False,
                            "qualification_eligible": True},
        evaluation={"diagnostic_monitor_error": error})

    projected = result["diagnostic_monitor"]
    assert projected["state"] == "untrusted"
    assert projected["enabled"] is False
    assert projected["diagnostic_only"] is True
    assert projected["qualification_eligible"] is False
    assert projected["pending_frontier"] is None
    assert len(projected["error"]) == 300


def test_stale_queue_diagnostic_monitor_does_not_expose_pending_progress(tmp_path):
    stale_stamp = "2000-01-01T00:00:00+00:00"
    result = _read_queue_for_lanes(tmp_path, [_queue_lane(t) for t in TRACKS],
        updated_utc=stale_stamp, diagnostic_monitor=_diagnostic_monitor_summary())

    projected = result["diagnostic_monitor"]
    assert result["state"] == "stale"
    assert projected["state"] == "stale"
    assert projected["fresh"] is False
    assert projected["pending_frontier"] is None
    assert projected["active_frontier"] is None
    assert projected["completed_count"] == 3
    assert projected["qualification_eligible"] is False


def test_diagnostic_monitor_frontend_contract_uses_text_content_and_keeps_schedule():
    root = Path(__file__).resolve().parents[1]
    html = (root / "frc_defense/dashboard_static/meco.html").read_text()
    source = (root / "frc_defense/dashboard_static/meco.js").read_text()
    start = source.index("const monitor = data.queue?.diagnostic_monitor;")
    end = source.index('document.querySelector("#policies")', start)
    rendering = source[start:end]

    assert 'id="diagnostic-monitor"' in html
    assert 'document.querySelector("#diagnostic-monitor")' in rendering
    assert "monitorLine.textContent" in rendering
    assert "innerHTML" not in rendering
    assert "never qualification eligible" in rendering
    assert "pending frontier unconfirmed" in rendering
    assert "evaluation_schedule" in source


def test_native_queue_list_requires_exactly_one_lane_per_track(tmp_path):
    result = _read_queue_for_lanes(
        tmp_path, [_queue_lane(track) for track in TRACKS])

    assert result["state"] == "fresh"
    assert set(result["lanes"]) == set(TRACKS)


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(_queue_lane("A1")),
    lambda rows: rows.append(_queue_lane("unknown")),
])
def test_native_queue_list_rejects_missing_duplicate_and_unknown_tracks(tmp_path, mutate):
    rows = [_queue_lane(track) for track in TRACKS]
    mutate(rows)

    result = _read_queue_for_lanes(tmp_path, rows)

    assert result["state"] == "stale"
    assert result["worker_state"] == "unknown"
    assert result["error"]


def test_dict_queue_requires_exact_keys_and_matching_lane_records(tmp_path):
    rows = {track: {key: value for key, value in _queue_lane(track).items()
                    if key != "track"} for track in TRACKS}
    result = _read_queue_for_lanes(tmp_path, rows)
    assert result["state"] == "fresh"

    rows["A1"]["track"] = "A2"
    mismatch = _read_queue_for_lanes(tmp_path / "mismatch", rows)
    assert mismatch["state"] == "stale"
    assert mismatch["worker_state"] == "unknown"


def test_live_assignment_projection_requires_matching_scheduler_snapshot(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    scenario = "offense_vs_scripted_pressure"
    category = SCENARIOS[scenario].required_categories[0]
    previous = {"cycle": 1, "scenario_id": scenario, "category": category,
        "seed": 17, "assignment_id": "previous-1", "sampling_role": "stage_exposure"}
    assignment = {"cycle": 2, "scenario_id": scenario, "category": category,
        "seed": 18, "assignment_id": "pending-2", "sampling_role": "stage_exposure"}
    counts = {"exposure_chunks": {scenario: 1},
        "category_exposures": {f"{scenario}/{category}": 1},
        "recovery_pressure": {}, "phase4_exposure_counts": {}}
    scheduler = {"schema_version": 1, "completed_cycle": 1,
        "pending_assignment": assignment, **counts,
        "history": [{"assignment": previous}]}
    (root / "task-scheduler-state.json").write_text(json.dumps(scheduler))
    queue_payload = {"cycle": 2, "task_assignment": assignment,
        "task_scheduler": {"completed_cycle": 1, "pending_assignment": assignment,
            **counts}}

    projected = _training_curriculum_projection(root, queue_payload, "fresh", tmp_path)

    assert projected["state"] == "fresh"
    assert projected["assignment"]["scenario_id"] == scenario
    assert projected["assignment"]["category"] == category

    scheduler["pending_assignment"] = {**assignment, "scenario_id": "pickup_simple"}
    (root / "task-scheduler-state.json").write_text(json.dumps(scheduler))
    mismatch = _training_curriculum_projection(root, queue_payload, "fresh", tmp_path)
    assert mismatch["state"] == "unavailable"
    assert mismatch["assignment"] is None


def test_deferred_queue_attempt_is_queued_and_not_started():
    attempt = _queue_attempt_projection("A1", {
        "status": "deferred_for_evaluation", "run_id": "deferred-run",
        "run_directory": "A1", "device": "cuda:0", "completed_updates": 0,
    }, queue_fresh=True)

    assert attempt["training_state"] == "queued"
    assert attempt["heartbeat"]["state"] == "not_started"
    assert attempt["queue_process_probe"]["confirmed"] is False
