"""Pure manifest contracts for the resumable paired Phase 0 calibration."""

import copy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from frc_defense import meco_qualification as q
from frc_defense import meco_external_release_gate as release_gate
from frc_defense.meco_simulated_score_ledger import (
    ESTIMAND_SCHEMA, OUTCOME_NAME, PHASE0_PLAN_SHA256, RULES, RULES_ID,
    RULES_SHA256, SOURCE_PLAN_SHA256, MecoScoreEventRecorder,
    estimand_approval_digest, make_event_stream, sha256,
    validate_estimand_approval, validate_replay_score_binding)


def _workers():
    common = {"device_model": "AMD Radeon Pro WX 9100",
        "backend": "hip", "backend_version": "rocm-test",
        "torch_version": "torch-test"}
    return [
        {"worker_id": "wx-a", "device": "cuda:0", "device_index": 0,
         "device_uuid": "30323133-6632-6261-3734-313632316134", **common},
        {"worker_id": "wx-b", "device": "cuda:1", "device_index": 1,
         "device_uuid": "30323133-6661-3836-3737-346334386134", **common},
    ]


def _bindings():
    freeze_sha = "b" * 64
    approval_sha = estimand_approval_digest(_approval())
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
             "approval_sha256": approval_sha})


def _approval():
    return {"schema": ESTIMAND_SCHEMA, "approval_id": "human-review-fixture",
        "approved": True, "approver": "fixture@example.org",
        "approved_at": "2026-10-25T12:00:00Z",
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "phase0_plan_sha256": PHASE0_PLAN_SHA256,
        "simulator_source_sha256": "9" * 64,
        "rules_id": RULES_ID, "rules_sha256": RULES_SHA256,
        "primary_estimand": OUTCOME_NAME,
        "formula": "sum score_units by alliance over accepted FUEL score-award events",
        "unit": RULES["unit"], "included_outcomes": RULES["included_outcomes"],
        "excluded_outcomes": RULES["excluded_outcomes"],
        "interpretation_limits": "Synthetic contract fixture, not official FRC points.",
        "intended_use": "Descriptive simulator outcome research only."}


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


def _release_observation(campaign):
    devices = {device: {**identity, "active_owners": []}
        for device, identity in sorted(release_gate.WX9100_BY_INDEX.items())}
    owners = {device: [] for device in release_gate.WX9100_BY_INDEX}
    body = {"schema": "meco-external-release-observation-v1",
        "campaign_id": release_gate.R4_CAMPAIGN_ID,
        "manifest_path": str(release_gate.R4_MANIFEST_PATH),
        "manifest_id": "acaf745eb84eb66ea8447a884ad6eb036166abe2f596b168ee392fb6d423bc03",
        "devices": devices,
        "observations": [{"r4_processes": [], "device_owners": owners} for _ in range(2)],
        "settle_seconds": 1.0, "observed_utc": "2026-10-25T00:00:00+00:00",
        "method": "proc-cmdline-and-proc-fd-render-node-bdf-v1"}
    return {**body, "observation_sha256": q._canonical_sha256(body)}


def _outcome_fixture(monkeypatch, campaign, unit, replay_root):
    shard_index = next(shard["shard_index"] for shard in campaign["shards"]
        if unit in shard["work_units"])
    worker = campaign["worker_assignments"][shard_index %
        len(campaign["worker_assignments"])]
    unit_id = q._canonical_sha256(unit)
    simulation_id = f"sim-{unit_id[:16]}"
    device = worker["device"]
    control = next(row for row in campaign["controls"]
        if row["control_id"] == unit["control_id"])
    source_hash = campaign["source"]["source_identity_sha256"]
    metrics = {name: .1 for name in q.SAFETY_METRIC_NAMES}
    metrics.update({"success": 1.0, "useful_throughput": 3.5})
    approval = _approval()
    approval_sha = estimand_approval_digest(approval)
    scoring_configuration = {"dt": 0.02, "alliance_zone_depth_m": 3.0,
        "hub_schedule": "source-bound-current", "robot_control_modes": ["deterministic"] * 6,
        "robot_types": ["dumper"] * 6, "robot_roles": ["offense"] * 3 + ["defense"] * 3,
        "robot_fuel_capacities": [8] * 6, "robot_score_interval_s": [0.2] * 6,
        "robot_dimensions_m": {"length": [1.] * 6, "width": [.7] * 6},
        "field_geometry": {"field_length_m": 16., "field_width_m": 8.,
            "hub_centers_m": [[4., 4.], [12., 4.]], "bump_regions": []},
        "robot_readiness": {"dumper_stationary_linear_mps": 0.2,
            "dumper_stationary_angular_radps": 0.2,
            "dumper_aim_tolerance_rad": 3.14159 / 9,
            "bump_clearance_source": "TensorThreeVsThreeEnv._robots_clear_of_bumps"},
        "score_intent_source": "effective policy action trace passed to update_fuel_actions",
        "score_rules_id": RULES_ID, "robot_team_slots": [0, 0, 0, 1, 1, 1],
        "fuel_count": 32}
    context = {"episode_id": f"paired-fixture-{unit_id[:16]}", "seed": unit["seed"],
        "scenario_id": unit["scenario_id"], "robot_team_slots": [0, 0, 0, 1, 1, 1],
        "world_count": 1, "source_identity_sha256": source_hash,
        "effective_scoring_configuration": scoring_configuration,
        "effective_scoring_configuration_sha256": sha256(scoring_configuration),
        "fuel_count": 32, "terminal_state": "complete", "terminal_tick": 8}
    recorder = MecoScoreEventRecorder(context, approval, approval_sha)
    elapsed = 0.
    dt = struct.unpack("=f", struct.pack("=f", .02))[0]
    for _ in range(2):
        elapsed = struct.unpack("=f", struct.pack("=f", elapsed + dt))[0]
    recorder.record_award(tick=1, match_elapsed_s=elapsed, world_slot=0,
        alliance=0, robot_slot=0, piece_slot=4, piece_generation=0,
        source_site="gamepiece_actions.update_fuel_actions.vectorized_score_mutation")
    captured = recorder.finish(terminal_tick=8, terminal_state="complete",
        terminal_score_counts=[1, 0])
    ledger = captured["ledger"]
    evaluation = {**{key: unit[key] for key in (
        "scenario_id", "category", "layout_id", "defender_family",
        "start_condition", "sensor_profile", "split", "seed")},
        "policy": control["policy"], "policy_id": control["policy_id"],
        "prediction_mode": control["prediction_mode"],
        "prediction_contract": control["prediction_contract"],
        "source_identity_sha256": source_hash,
        "safety_metrics_schema": q.SAFETY_METRICS_SCHEMA,
        "safety_metrics_source_identity_sha256": source_hash,
        "meco_metric_ledger": {"safety_metrics": {name: metrics[name]
            for name in q.SAFETY_METRIC_NAMES}},
        "metrics": metrics, "score_terminal_counters": [1, 0],
        "fuel_score_count": [1, 0], "official_match_points": None,
        "qualification_eligible": False,
        "execution_device": {"device": device, "accelerator_backend": "ROCm"},
        "execution_failures": [],
        "simulated_score_ledger": ledger,
        "simulated_match_score": ledger["simulated_match_score"]}
    replay_digest = q._canonical_sha256(evaluation)
    replay_ref = {"simulation_id": simulation_id,
        "result_path": f"{simulation_id}.result.json",
        "evaluation_record_digest": replay_digest}
    outcome_digest = q._canonical_sha256({"metrics": metrics,
        "safety_metrics": evaluation["meco_metric_ledger"]["safety_metrics"],
        "execution_failures": []})
    replay = {"simulation_id": simulation_id, "evaluation_record_digest": replay_digest,
        "replay_contract_schema": q.PAIRED_CALIBRATION_REPLAY_CONTRACT,
        "paired_calibration_binding": q.paired_calibration_replay_binding(
            campaign, shard_index, unit),
        "source_identity_sha256": source_hash,
        "condition_id": q.paired_calibration_condition_id(
            q.paired_calibration_replay_binding(campaign, shard_index, unit)),
        "paired_calibration_worker_identity": worker,
        "device": {"device": device, "accelerator_backend": "ROCm"},
        "execution_device": {"device": device, "accelerator_backend": "ROCm"},
        "simulation_device": device, "outcome_digest": outcome_digest,
        "safety_digest": q._canonical_sha256(evaluation["meco_metric_ledger"]["safety_metrics"]),
        "scenario_id": unit["scenario_id"], "category": unit["category"],
        "seed": unit["seed"], "split": unit["split"],
        "qualification_case": {key: unit[key] for key in (
            "layout_id", "defender_family", "start_condition", "sensor_profile")},
        "policy": control["policy"], "policy_id": control["policy_id"],
        "prediction_mode": control["prediction_mode"],
        "evaluation_record": {"metrics": metrics,
            "safety_metrics": evaluation["meco_metric_ledger"]["safety_metrics"],
            "simulated_score_ledger_sha256": ledger["ledger_sha256"],
            "simulated_match_score": ledger["simulated_match_score"],
            "source_identity_sha256": source_hash},
        "score_event_stream": captured["event_stream"],
        "simulated_score_ledger": ledger,
        "simulated_match_score": ledger["simulated_match_score"],
        "estimand_approval_sha256": approval_sha,
        "score_terminal_counters": [1, 0]}
    replay_raw = json.dumps(replay, sort_keys=True).encode()
    (replay_root / replay_ref["result_path"]).write_bytes(replay_raw)
    validate_estimand_approval(approval, source_identity_sha256=source_hash)
    assert estimand_approval_digest(approval) == campaign["estimand_approval"]["approval_sha256"]
    validate_replay_score_binding(replay, evaluation, [1, 0], approval)
    row = q.build_paired_calibration_outcome_row(campaign, shard_index, unit, evaluation,
        replay_raw, replay_ref, approval)
    return row, approval, ledger, replay_raw, evaluation, replay_ref


def test_paired_campaign_binds_all_controls_and_complete_calibration_schedule(monkeypatch):
    campaign = _campaign(monkeypatch)
    units = [unit for shard in campaign["shards"] for unit in shard["work_units"]]
    assert campaign["campaign_type"] == q.PAIRED_CALIBRATION_VERSION
    assert campaign["config"]["control_ids"] == list(q.PAIRED_CALIBRATION_CONTROLS)
    d0 = next(row for row in campaign["controls"] if row["control_id"] == "D0")
    assert d0["prediction_mode"] is None
    assert d0["prediction_contract"] is None
    assert d0["prediction_contract_sha256"] is None
    assert d0["control_binding_sha256"] == q._canonical_sha256(
        {key: value for key, value in d0.items()
            if key != "control_binding_sha256"})
    multi_model = next(row for row in campaign["controls"]
        if row["control_id"] == "D2/multi_model_v1")
    assert multi_model["execution_supported"] is True
    assert multi_model["support_limit"] is None
    assert multi_model["prediction_contract"]["model"] == "meco_opponent_kinematics_mm_v1"
    assert multi_model["prediction_contract"]["mode"] == "multi_model_v1"
    assert multi_model["prediction_contract_sha256"] == q._canonical_sha256(
        multi_model["prediction_contract"])
    for shard in campaign["shards"]:
        assert shard["control_bindings"]["D2/multi_model_v1"] == \
            multi_model["control_binding_sha256"]
        assert shard["control_binding_sha256"] == q._canonical_sha256(
            shard["control_bindings"])
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
    assert q.paired_calibration_execution_blockers(campaign) == []
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

    altered = copy.deepcopy(campaign)
    altered["controls"][2]["prediction_contract"]["max_speed_mps"] = 999.
    with pytest.raises(ValueError, match="predictor identity"):
        q.validate_paired_calibration_campaign(altered)


def test_paired_runner_groups_exact_case_seed_tuples_and_revalidates_gate_before_eval(monkeypatch):
    campaign = _campaign(monkeypatch)
    shard = campaign["shards"][0]
    groups = q._paired_calibration_evaluation_groups(shard)
    assert groups
    for group in groups:
        assert group["seeds"] == [3, 7]
        assert set(group["controls"]) == {"D0", group["key"][-1]}
        for control_id, units in group["controls"].items():
            assert [unit["seed"] for unit in units] == group["seeds"]
    from frc_defense import meco_external_release_gate as release_gate
    ordering = []
    monkeypatch.setattr(release_gate, "verify_external_release_before_paired_run",
        lambda campaign: ordering.append("release") or {})
    monkeypatch.setattr(q, "validate_paired_calibration_gate_artifacts",
        lambda *args, **kwargs: ordering.append("artifacts") or
            (_ for _ in ()).throw(ValueError("gate validation reached")))
    with pytest.raises(ValueError, match="gate validation reached"):
        q.run_paired_calibration_shard(campaign, 0, repo_root="unavailable",
            replay_root="unavailable", output_root="unavailable",
            configuration_artifacts={})
    assert ordering == ["release", "artifacts"]


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


def test_paired_progress_is_manifest_bound_unique_and_resume_stable(monkeypatch, tmp_path):
    campaign = _campaign(monkeypatch)
    shard = campaign["shards"][0]
    unit = shard["work_units"][0]
    row, approval, _, _, _, _ = _outcome_fixture(monkeypatch, campaign, unit, tmp_path)
    progress = q.build_paired_calibration_progress(campaign, 0, [row],
        replay_root=tmp_path, approval=approval)
    assert progress["complete"] is False
    assert q.validate_paired_calibration_progress(campaign, 0, progress,
        replay_root=tmp_path, approval=approval) == progress

    changed = copy.deepcopy(progress)
    changed["completed_rows"].append(row)
    body = {key: value for key, value in changed.items() if key != "progress_id"}
    changed["progress_id"] = q._canonical_sha256(body)
    with pytest.raises(ValueError, match="duplicate, unordered"):
        q.validate_paired_calibration_progress(campaign, 0, changed,
            replay_root=tmp_path, approval=approval)

    altered = copy.deepcopy(progress)
    altered["shard_manifest_id"] = "b" * 64
    body = {key: value for key, value in altered.items() if key != "progress_id"}
    altered["progress_id"] = q._canonical_sha256(body)
    with pytest.raises(ValueError, match="another manifest"):
        q.validate_paired_calibration_progress(campaign, 0, altered,
            replay_root=tmp_path, approval=approval)

    with pytest.raises(ValueError, match="progress is incomplete"):
        q.build_paired_calibration_final_shard(campaign, 0, progress,
            replay_root=tmp_path, approval=approval,
            release_observation=_release_observation(campaign))


def test_paired_artifact_writer_is_atomic_idempotent_and_never_overwrites(tmp_path):
    path = tmp_path / "progress.json"
    artifact = {"schema": q.PAIRED_CALIBRATION_PROGRESS, "progress_id": "a" * 64}
    digest = q.write_paired_calibration_artifact_immutable(path, artifact)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert q.write_paired_calibration_artifact_immutable(path, artifact) == digest
    with pytest.raises(ValueError, match="conflicting bytes"):
        q.write_paired_calibration_artifact_immutable(path,
            {"schema": q.PAIRED_CALIBRATION_PROGRESS, "progress_id": "b" * 64})


def test_progress_writer_is_atomic_monotonic_and_compare_and_swap(tmp_path):
    path = tmp_path / "shard.progress.json"

    def progress(ids):
        body = {"schema": q.PAIRED_CALIBRATION_PROGRESS,
            "campaign_id": "campaign", "shard_index": 0,
            "shard_manifest_id": "manifest",
            "completed_work_unit_ids": ids, "completed_rows": [],
            "complete": False}
        return {**body, "progress_id": q._canonical_sha256(body)}

    first = progress(["a"])
    q.write_paired_calibration_progress_atomic(path, first)
    second = progress(["a", "b"])
    q.write_paired_calibration_progress_atomic(path, second,
        expected_previous_progress_id=first["progress_id"])
    assert json.loads(path.read_bytes()) == second
    with pytest.raises(ValueError, match="CAS token is stale"):
        q.write_paired_calibration_progress_atomic(path, progress(["a", "b", "c"]),
            expected_previous_progress_id=first["progress_id"])
    with pytest.raises(ValueError, match="not monotonic"):
        q.write_paired_calibration_progress_atomic(path, progress(["b"]),
            expected_previous_progress_id=second["progress_id"])


def test_paired_merge_requires_exact_shard_set_after_registered_controls_enable(monkeypatch):
    campaign = _campaign(monkeypatch)
    assert q.paired_calibration_execution_blockers(campaign) == []
    with pytest.raises(ValueError, match="exactly one final artifact per shard"):
        q.build_paired_calibration_merged_report(campaign, [], replay_root="unused",
            approval={})


def test_paired_control_summary_uses_same_tuple_and_separate_alliance_deltas(monkeypatch):
    campaign = _campaign(monkeypatch)
    units = [unit for shard in campaign["shards"] for unit in shard["work_units"]]
    candidate = next(unit for unit in units if unit["control_id"] == "D1/current")
    baseline = next(unit for unit in units if unit["pair_id"] == candidate["pair_id"]
        and unit["control_id"] == "D0")
    safety = {name: .2 for name in q.SAFETY_METRIC_NAMES}
    rows = {candidate["pair_id"]: {
        "D0": {"work_unit": baseline, "simulated_match_score": {"red": 2, "blue": 1},
            "safety_metrics": safety, "useful_throughput": 3.0},
        "D1/current": {"work_unit": candidate,
            "simulated_match_score": {"red": 4, "blue": 0},
            "safety_metrics": {name: .1 for name in q.SAFETY_METRIC_NAMES},
            "useful_throughput": 4.0}}}
    summary = q._paired_control_summary(campaign, "D1/current", rows)
    assert summary["simulated_match_score_difference"]["red"]["mean"] == 2.0
    assert summary["simulated_match_score_difference"]["blue"]["mean"] == -1.0
    assert summary["safety_rate_difference"][next(iter(q.SAFETY_METRIC_NAMES))]["mean"] == -0.1
    assert len(summary["stratum_summaries"]) == 1
    stratum = summary["stratum_summaries"][0]
    assert stratum["pair_count"] == 1
    assert stratum["category"] == candidate["category"]
    assert stratum["simulated_match_score_difference"]["red"]["mean"] == 2.0
    assert stratum["safety_rate_difference"][next(iter(q.SAFETY_METRIC_NAMES))]["mean"] == -0.1


def test_paired_seed_bootstrap_resamples_clusters_and_retains_repetitions():
    interval = q._paired_seed_cluster_bootstrap_ci(
        {"seed-a": [0.0, 100.0], "seed-b": [10.0, 10.0]},
        seed=17, replicates=1000)
    assert interval == [10.0, 50.0]

    with pytest.raises(ValueError, match="populated seed clusters"):
        q._paired_seed_cluster_bootstrap_ci({}, seed=17, replicates=10)


def test_paired_control_bundle_is_individually_digest_bound_and_nonqualifying(monkeypatch):
    campaign = _campaign(monkeypatch)
    summary = {"control_id": "D1/current", "pair_count": 3,
        "simulated_match_score_difference": {"red": {"mean": 1.0}}}
    bundle, artifact_sha256 = q._build_paired_control_bundle(campaign,
        summary, ["a" * 64, "b" * 64])
    raw = json.dumps(bundle, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()
    assert hashlib.sha256(raw).hexdigest() == artifact_sha256
    assert bundle["bundle_id"] == q._canonical_sha256({
        key: value for key, value in bundle.items() if key != "bundle_id"})
    assert bundle["final_shard_sha256"] == ["a" * 64, "b" * 64]
    assert bundle["official_match_points"] is None
    assert bundle["qualification_eligible"] is False

    changed, changed_sha = q._build_paired_control_bundle(campaign,
        {**summary, "pair_count": 4}, ["a" * 64, "b" * 64])
    assert changed["bundle_id"] != bundle["bundle_id"]
    assert changed_sha != artifact_sha256


def test_paired_outcome_row_binds_evaluation_and_exact_replay(monkeypatch, tmp_path):
    campaign = _campaign(monkeypatch)
    unit = campaign["shards"][0]["work_units"][0]
    row, approval, ledger, replay_raw, evaluation, replay_ref = _outcome_fixture(
        monkeypatch, campaign, unit, tmp_path)
    assert row["replay_sha256"] == hashlib.sha256(replay_raw).hexdigest()
    assert row["ledger_sha256"] == ledger["ledger_sha256"]
    assert row["simulated_match_score"] == ledger["simulated_match_score"]
    assert row["official_match_points"] is None
    assert row["qualification_eligible"] is False
    assert q.validate_paired_calibration_outcome_row(campaign, 0, row,
        replay_root=tmp_path, approval=approval) == row
    replay_path = tmp_path / replay_ref["result_path"]
    replay_path.write_bytes(replay_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifact bytes changed"):
        q.validate_paired_calibration_outcome_row(campaign, 0, row,
            replay_root=tmp_path, approval=approval)
    replay_path.write_bytes(replay_raw)

    mutated = copy.deepcopy(row)
    mutated["event_stream_sha256"] = "0" * 64
    body = {key: value for key, value in mutated.items() if key != "outcome_row_sha256"}
    mutated["outcome_row_sha256"] = q._canonical_sha256(body)
    with pytest.raises(ValueError, match="replay identity/event stream changed"):
        q.validate_paired_calibration_outcome_row(campaign, 0, mutated,
            replay_root=tmp_path, approval=approval)

    changed = dict(replay_ref, evaluation_record_digest="0" * 64)
    with pytest.raises(ValueError, match="evaluation digest differs"):
        q.build_paired_calibration_outcome_row(campaign, 0, unit, evaluation,
            replay_raw, changed, {"approved": True})


def test_multimodel_control_identity_survives_outcome_and_progress_validation(monkeypatch, tmp_path):
    campaign = _campaign(monkeypatch)
    unit = next(unit for unit in campaign["shards"][0]["work_units"]
        if unit["control_id"] == "D2/multi_model_v1")
    shard_index = 0
    row, approval, _, replay_raw, evaluation, replay_ref = _outcome_fixture(
        monkeypatch, campaign, unit, tmp_path)
    control = next(control for control in campaign["controls"]
        if control["control_id"] == "D2/multi_model_v1")
    assert evaluation["prediction_mode"] == "multi_model_v1"
    assert evaluation["prediction_contract"] == control["prediction_contract"]
    assert row["control_binding_sha256"] == control["control_binding_sha256"]
    assert q.validate_paired_calibration_outcome_row(campaign, shard_index, row,
        replay_root=tmp_path, approval=approval) == row
    progress = q.build_paired_calibration_progress(campaign, shard_index, [row],
        replay_root=tmp_path, approval=approval)
    assert progress["completed_rows"] == [row]

    changed = copy.deepcopy(row)
    changed["control_binding"]["prediction_contract"]["max_horizon_s"] = 99.0
    body = {key: value for key, value in changed.items() if key != "outcome_row_sha256"}
    changed["outcome_row_sha256"] = q._canonical_sha256(body)
    with pytest.raises(ValueError, match="predictor/control identity changed"):
        q.validate_paired_calibration_outcome_row(campaign, shard_index, changed,
            replay_root=tmp_path, approval=approval)


def test_multimodel_contract_digest_survives_final_shard_and_merged_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(q, "BOOTSTRAP_REPLICATES", 25)
    monkeypatch.setattr(q, "PAIRED_CALIBRATION_CONTROLS",
        ("D0", "D2/multi_model_v1"))
    campaign = _campaign(monkeypatch)
    approval = _approval()
    finals = []
    for shard in campaign["shards"]:
        rows = [_outcome_fixture(monkeypatch, campaign, unit, tmp_path)[0]
            for unit in shard["work_units"]]
        progress = q.build_paired_calibration_progress(campaign,
            shard["shard_index"], rows, replay_root=tmp_path, approval=approval)
        final = q.build_paired_calibration_final_shard(campaign,
            shard["shard_index"], progress, replay_root=tmp_path, approval=approval,
            release_observation=_release_observation(campaign))
        expected_binding = next(control["control_binding_sha256"]
            for control in campaign["controls"]
            if control["control_id"] == "D2/multi_model_v1")
        assert final["control_bindings"]["D2/multi_model_v1"] == expected_binding
        assert all(row["control_binding_sha256"] == expected_binding
            for row in final["completed_rows"]
            if row["control_id"] == "D2/multi_model_v1")
        finals.append(json.dumps(final, sort_keys=True, separators=(",", ":")).encode())

    report = q.build_paired_calibration_merged_report(campaign, finals,
        replay_root=tmp_path, approval=approval)
    bundle = next(bundle for bundle in report["_control_bundle_artifacts"]
        if bundle["control_id"] == "D2/multi_model_v1")
    assert bundle["control_binding_sha256"] == expected_binding
    assert bundle["qualification_eligible"] is False
    assert report["official_match_points"] is None
    assert any(ref["bundle_id"] == bundle["bundle_id"]
        for ref in report["control_bundles"])

@pytest.mark.parametrize("mutation,message", [
    ("qualification_case", "outcome/case/safety digest differs"),
    ("metrics", "outcome/case/safety digest differs"),
    ("safety", "outcome/case/safety digest differs"),
    ("worker", "identity/event stream changed"),
])
def test_paired_outcome_rejects_replay_tuple_metrics_safety_and_device_drift(
        monkeypatch, tmp_path, mutation, message):
    campaign = _campaign(monkeypatch)
    unit = campaign["shards"][0]["work_units"][0]
    case_root = tmp_path / mutation
    case_root.mkdir()
    row, approval, _, replay_raw, _, replay_ref = _outcome_fixture(
        monkeypatch, campaign, unit, case_root)
    replay = json.loads(replay_raw)
    if mutation == "qualification_case":
        replay["qualification_case"]["layout_id"] = "wrong-layout"
    elif mutation == "metrics":
        replay["evaluation_record"]["metrics"]["success"] = 0.0
    elif mutation == "safety":
        name = next(iter(q.SAFETY_METRIC_NAMES))
        replay["evaluation_record"]["safety_metrics"][name] = .9
    else:
        replay["paired_calibration_worker_identity"]["device_uuid"] = "other-gpu"
    changed_raw = json.dumps(replay, sort_keys=True).encode()
    (case_root / replay_ref["result_path"]).write_bytes(changed_raw)
    row["replay_sha256"] = hashlib.sha256(changed_raw).hexdigest()
    body = {key: value for key, value in row.items() if key != "outcome_row_sha256"}
    row["outcome_row_sha256"] = q._canonical_sha256(body)
    with pytest.raises(ValueError, match=message):
        q.validate_paired_calibration_outcome_row(campaign, 0, row,
            replay_root=case_root, approval=approval)


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
