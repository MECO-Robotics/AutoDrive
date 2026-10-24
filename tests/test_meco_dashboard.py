import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from frc_defense.meco_dashboard import (
    ABLATIONS, CONTROLS, TRACKS, _collect_status, _queue_attempt_projection, _queue_state,
    _training_curriculum_projection, meco_status_payload,
)
from frc_defense.meco_path_penalty import PLAN_A_PPO_RECOMPUTATION_CONTRACT
from frc_defense.training_curriculum import SCENARIOS


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
    assert [row["policy_id"] for row in payload["policies"]] == list(
        TRACKS + ABLATIONS + CONTROLS)
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


def _read_queue_for_lanes(tmp_path, lanes):
    now = datetime.now(timezone.utc)
    root = tmp_path / "checkpoints"
    root.mkdir(parents=True)
    (root / "queue-status.json").write_text(json.dumps({
        "schema_version": 1, "updated_utc": now.isoformat(), "lanes": lanes}))
    return _queue_state(root, now.timestamp(), tmp_path)


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
