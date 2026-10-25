import copy
import hashlib
import json
from pathlib import Path

import pytest

from frc_defense import dashboard_api


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "docs/MECO_THREE_PASS_PROVENANCE.json"


@pytest.fixture
def evidence_case(tmp_path, monkeypatch):
    data = json.loads(MANIFEST.read_text())
    plan = tmp_path / data["plan_path"]
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_bytes((REPO / data["plan_path"]).read_bytes())
    monkeypatch.setattr(dashboard_api, "_EXPECTED_PLAN_SHA256", hashlib.sha256(plan.read_bytes()).hexdigest())
    for tranche in data["tranches"]:
        for role_name in ("planner", "reviewer"):
            artifact = tranche[role_name].get("artifact")
            if not isinstance(artifact, dict):
                continue
            destination = tmp_path / artifact["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / artifact["path"]).read_bytes())
        for artifact_key in ("code_review_artifact", "ledger_candidate_summary_artifact"):
            artifact = tranche["implementer"].get(artifact_key)
            if isinstance(artifact, dict):
                destination = tmp_path / artifact["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((REPO / artifact["path"]).read_bytes())
        for relative in tranche["implementer"]["paths"]:
            destination = tmp_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / relative).read_bytes())
    # Exercise the previously approved prefix as a validated summary while D
    # remains intentionally pending in the actual repository manifest.
    data["tranches"] = data["tranches"][:-1]
    return tmp_path, data


def validate(project, data):
    return dashboard_api._validate_three_pass_manifest(project, data)


def test_prior_manifest_prefix_remains_validated_as_summary_only(evidence_case):
    current = dashboard_api._three_pass_provenance()
    assert current["status"] == "untrusted"
    assert "reviewer evidence" in current["untrusted_reason"]
    project, data = evidence_case
    result = validate(project, data)
    assert result["validation_status"] == "validated_summary_only"
    assert result["status"] == "scoped-transcript-summaries"
    assert result["evidence_class"] == "scoped_transcript_summary"
    assert len(result["tranches"]) == 88
    assert result["tranches"][-1]["requirement_id"] == (
        "phase4-runtime-status-ledger-dashboard")
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == (
        dashboard_api._EXPECTED_THREE_PASS_PROVENANCE_SHA256)
    assert hashlib.sha256(json.dumps(
        result["tranches"], sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest() == (
        "af5303953982a989276a62bcc47abb78700e5d5b4273f93796fcc3a356423829")
    phase4_status = result["tranches"][-1]
    assert phase4_status["planner"]["artifact"] == {
        "path": "docs/plans/phase4-runtime-status-ledger-dashboard-20261025.md",
        "sha256": "3fd84e8adaab8b8ab92d08123dd10d7486890255476259a17de7c31d6a95549c",
        "kind": "scoped-transcript-summary",
        "status": "transcript-summary",
    }
    assert phase4_status["implementer"]["focal_status_pass2_commit"] == (
        "2357ae4a9661c7cb6d6c038a72881a55fbdcfbec")
    assert phase4_status["implementer"]["focal_status_approved_commit"] == (
        phase4_status["implementer"]["focal_status_pass2_commit"])
    assert phase4_status["implementer"]["focal_status_integration_commit"] == (
        "252eb3d65e27a0d96352cf2abb83265384ce1d8d")
    assert phase4_status["implementer"]["runtime_status_pass2_commit"] == (
        "3ed5c86e5108db24eb1588afc890d1cf379cad39")
    assert phase4_status["implementer"]["runtime_status_approved_commit"] == (
        phase4_status["implementer"]["runtime_status_pass2_commit"])
    assert phase4_status["implementer"]["runtime_status_integration_commit"] == (
        "384c9a2baa9a97a27271cc6ed26fb316f01be8ed")
    assert phase4_status["implementer"]["status"] == "implemented"
    assert "candidate_review_pending" not in phase4_status["implementer"]
    assert phase4_status["implementer"]["candidate_commit"] == (
        phase4_status["reviewer"]["approved_commit"])
    assert phase4_status["reviewer"]["status"] == "approved-summary"
    assert phase4_status["reviewer"]["verdict"] == "approved"
    assert phase4_status["reviewer"]["artifact"]["path"] == (
        "docs/implementation-provenance/phase4-runtime-status-ledger-dashboard-20261025.md")
    phase4_status_summary = (REPO / phase4_status["reviewer"]["artifact"]["path"]).read_text()
    for required_reference in (
        "phase4-focal-pose-mislocalization-wiring-20261025.md",
        "phase4-runtime-status-corrections-20261025.md",
        "2357ae4a9661c7cb6d6c038a72881a55fbdcfbec",
        "3ed5c86e5108db24eb1588afc890d1cf379cad39",
        "252eb3d65e27a0d96352cf2abb83265384ce1d8d",
        "384c9a2baa9a97a27271cc6ed26fb316f01be8ed",
        "cfadae21999fdbd31597fd096bc916485d20ec98",
    ):
        assert required_reference in phase4_status_summary
    phase2_complete_set = next(t for t in result["tranches"]
                               if t["requirement_id"] == "phase2-five-track-complete-set")
    assert phase2_complete_set["reviewer"]["verdict"] == "approved"
    lease = next(t for t in result["tranches"]
                 if t["requirement_id"] == "meco-evaluation-device-lease")
    assert lease["implementer"]["commit"] == "9e41078fd438767217d051b89a58b2110449b3fb"
    assert lease["reviewer"]["approved_commit"] == lease["implementer"]["commit"]
    phase4_training = next(t for t in result["tranches"]
                           if t["requirement_id"] == "phase4-training-exposure-manifest")
    assert phase4_training["implementer"]["commit"] == (
        "e843d7ee6662b9f664f05bb8a1885aeb6b30ac8e")
    assert phase4_training["reviewer"]["approved_commit"] == (
        "ba38d8799226c05c498d14d1f6593ef2c73dc738")
    plan_a_hip = next(t for t in result["tranches"]
                      if t["requirement_id"] == "plan-a-hip-exact-heap-hardening")
    assert plan_a_hip["implementer"]["commit"] == (
        "5f199c68efdc4e17ae77fa4e286c15adaf9093d6")
    assert plan_a_hip["reviewer"]["approved_commit"] == (
        "5f199c68efdc4e17ae77fa4e286c15adaf9093d6")
    durable_schedule = next(t for t in result["tranches"]
        if t["requirement_id"] == "meco-durable-evaluation-schedule")
    assert durable_schedule["implementer"]["commit"] == (
        "d6bcab27637dbc2b39240e95ca994224d8809f31")
    assert durable_schedule["implementer"]["candidate_integration_commit"] == (
        "53157b4f81807126a38b6e2832e0eced745bcbd4")
    assert durable_schedule["reviewer"]["approved_commit"] == (
        durable_schedule["implementer"]["commit"])
    assert "No GPU, simulator, or replay" in durable_schedule["limits"]
    trace = next(t for t in result["tranches"]
                 if t["requirement_id"] == "meco-inference-observability-decision-trace")
    assert trace["implementer"]["commit"] == "7cfe3d07696167d931ced416780435c4f13ddfc1"
    assert trace["reviewer"]["approved_commit"] == trace["implementer"]["commit"]
    d0 = next(t for t in result["tranches"]
              if t["requirement_id"] == "d0-campaign-orchestration")
    assert d0["implementer"]["commit"] == "dbd84d8e67a92db7bacef46043cae4a2bd316e8b"
    assert d0["implementer"]["initial_commit"] == "91f24a1ffe4feef37d70f2cba414676412811be2"
    assert d0["reviewer"]["approved_commit"] == d0["implementer"]["commit"]
    matched = next(t for t in result["tranches"]
                   if t["requirement_id"] == "matched-evaluation-controls")
    assert matched["implementer"]["commit"] == "0db1e6436fbd753d82e49fbab02850d81fce1f07"
    assert matched["reviewer"]["approved_commit"] == matched["implementer"]["commit"]
    planner = next(t for t in result["tranches"]
                   if t["requirement_id"] == "validation-campaign-planner")
    assert planner["implementer"]["commit"] == "4a5ef0d71b4a5db218deb6f7b85ca90cd86c6ad0"
    assert planner["reviewer"]["approved_commit"] == planner["implementer"]["commit"]
    assert "not whole-plan approval" in result["message"]
    assert all(t["planner"]["artifact"]["kind"] == "scoped-transcript-summary"
               for t in result["tranches"])
    plan_c_repeats = next(t for t in result["tranches"]
                          if t["requirement_id"] == "plan-c-matched-repeated-rollout-labels")
    assert plan_c_repeats["implementer"]["candidate_implementation_commit"] == (
        "3186636a559a7729387a4b7989ca6caee0569b4d")
    assert plan_c_repeats["implementer"]["coverage_followup_commit"] == (
        "6c397fe17570363cc3d14f7232bc8c660b076eb3")
    assert plan_c_repeats["reviewer"]["approved_commit"] == (
        "6c397fe17570363cc3d14f7232bc8c660b076eb3")
    assert "No Plan C labels were collected" in plan_c_repeats["limits"]
    plan_a_heap = next(t for t in result["tranches"]
                       if t["requirement_id"] == "plan-a-exact-heap-device-prototype")
    assert plan_a_heap["implementer"]["commit"] == (
        "eae7c8ebfea8dce971845b95597ee95ca90a157d")
    assert plan_a_heap["reviewer"]["approved_commit"] == (
        plan_a_heap["implementer"]["commit"])
    assert "No HIP compilation/execution" in plan_a_heap["limits"]
    contact_geometry = next(t for t in result["tranches"]
        if t["requirement_id"] == "phase4-contact-geometry-path")
    assert contact_geometry["implementer"]["commit"] == (
        "1d4e4eb15562fb7b490310394a3f4f832e95a1b9")
    assert contact_geometry["reviewer"]["approved_commit"] == (
        contact_geometry["implementer"]["commit"])
    pair_contact = next(t for t in result["tranches"]
                        if t["requirement_id"] == "phase4-pair-contact-events")
    assert pair_contact["implementer"]["commit"] == (
        "b3adb19b055acece92952c69bed7c1e9e1290699")
    assert pair_contact["reviewer"]["approved_commit"] == (
        pair_contact["implementer"]["commit"])
    route_progress = next(t for t in result["tranches"]
        if t["requirement_id"] == "phase4-first-route-progress-interval")
    assert route_progress["implementer"]["commit"] == (
        "9ee3369eb816c187e7ecd5fecfde520f89be0f78")
    assert route_progress["reviewer"]["approved_commit"] == (
        route_progress["implementer"]["commit"])
    assert route_progress["implementer"]["paths"] == [
        "frc_defense/meco_phase4_runtime_adapter.py",
        "frc_defense/meco_track_training.py",
        "tests/test_meco_phase4_runtime_adapter_contract.py"]
    assert "recovery_time_after_bump_s" in route_progress["limits"]
    assert "qualification remains false" in route_progress["limits"]
    endpoint = next(t for t in result["tranches"]
        if t["requirement_id"] == "phase4-selected-goal-endpoint-diagnostic")
    assert endpoint["implementer"]["commit"] == (
        "9462bbd36756d79c1001d6b90c9c5f31d7c94f89")
    assert endpoint["reviewer"]["approved_commit"] == endpoint["implementer"]["commit"]
    assert endpoint["implementer"]["paths"] == [
        "frc_defense/meco_metric_ledger.py",
        "frc_defense/meco_phase4_runtime_adapter.py",
        "frc_defense/meco_track_training.py",
        "tests/test_meco_selected_goal_endpoint_contract.py"]
    assert "episode-wide" in endpoint["limits"]
    assert "D0 has no selected-goal ledger" in endpoint["limits"]
    assert "not contact-caused recovery" in endpoint["limits"]
    onnx = next(t for t in result["tranches"]
                if t["requirement_id"] == "phase5-onnx-runtime-parity-followup")
    assert onnx["implementer"]["commit"] == "8cc96a9270930ca86147b00efa60404e062bbe6e"
    assert onnx["implementer"]["candidate_integration_commit"] == (
        "7c932da57573b6fd0b0adaaf7b2d01083f275d1b")
    assert onnx["reviewer"]["approved_commit"] == onnx["implementer"]["commit"]
    assert onnx["reviewer"]["integrated_approved_commit"] == (
        onnx["implementer"]["candidate_integration_commit"])
    assert onnx["implementer"]["paths"] == [
        "frc_defense/meco_deployment.py",
        "scripts/verify_meco_onnx_parity.py",
        "tests/test_meco_deployment_contracts.py",
        "tests/test_meco_onnx_runtime_parity.py",
        "tests/test_meco_parameter_inventory.py",
        "docs/implementation-provenance/production-input-onnx-parity.md"]
    assert "synthetic inputs" in onnx["limits"]
    assert "No real production capture" in onnx["limits"]
    rollout_u = next(t for t in result["tranches"]
        if t["requirement_id"] == "phase2-plan-a-matched-rollout-u-oracle")
    assert rollout_u["implementer"]["commit"] == (
        "88b0bb9270ad912fb0cc651c33990e57c111b9b0")
    assert rollout_u["implementer"]["candidate_integration_commit"] == (
        rollout_u["implementer"]["commit"])
    assert rollout_u["reviewer"]["approved_commit"] == rollout_u["implementer"]["commit"]
    assert rollout_u["implementer"]["paths"] == [
        "frc_defense/meco_counterfactual.py",
        "frc_defense/meco_plan_a_oracle_audit.py",
        "frc_defense/meco_supervised.py",
        "tests/test_meco_plan_a_oracle_audit.py",
        "docs/MECO_SUPERVISED_WARMSTART.md"]
    assert "Seventeen focused offline audit tests passed" in rollout_u["limits"]
    assert "No labels" in rollout_u["limits"]
    assert "No simulator, replay, GPU training" in rollout_u["limits"]
    assert "no global optimality claim" in rollout_u["limits"]
    assert "P remains P0" in rollout_u["limits"]
    oracle_summary = (REPO / rollout_u["planner"]["artifact"]["path"]).read_text()
    assert "/root/phase2_goal_oracle_implementer" not in oracle_summary
    assert "88b0bb9270ad912fb0cc651c33990e57c111b9b0" in oracle_summary
    matrix_ci = next(t for t in result["tranches"]
                     if t["requirement_id"] == "matrix-paired-effects-confidence-intervals")
    assert matrix_ci["implementer"]["commit"] == (
        "185aba4b411eed9698df0e251c469dfc580bb141")
    assert matrix_ci["implementer"]["candidate_integration_commit"] == (
        "3d5f4b0df3ff97795c8733d743040cb23d14c3bb")
    assert matrix_ci["reviewer"]["approved_commit"] == matrix_ci["implementer"]["commit"]
    assert matrix_ci["planner"]["artifact"]["path"] == (
        "docs/implementation-provenance/matrix-paired-effects-confidence-intervals.md")
    assert matrix_ci["planner"]["artifact"]["sha256"] == matrix_ci["reviewer"]["artifact"]["sha256"]
    assert matrix_ci["implementer"]["paths"] == [
        "scripts/summarize_meco_ablation_matrix.py",
        "tests/test_summarize_meco_ablation_matrix.py"]
    assert "diagnostic paired effects only" in matrix_ci["limits"]
    matrix_summary = (REPO / matrix_ci["planner"]["artifact"]["path"]).read_text()
    assert "/root/phase4_next_planner" in matrix_summary
    assert "/root/phase4_pair_contact_review" in matrix_summary
    assert "185aba4b411eed9698df0e251c469dfc580bb141" in matrix_summary
    reactive = next(t for t in result["tranches"]
        if t["requirement_id"] == "phase4-reactive-opponent-validation-wiring")
    assert reactive["implementer"]["commit"] == (
        "49c8ba7d15c507c1208600daf3b3fa3b4312a15e")
    assert reactive["reviewer"]["reviewed_integration_commit"] == (
        reactive["implementer"]["commit"])
    assert {entry["reviewer"] for entry in reactive["reviewer"]["approvals"]} == {
        "/root/phase4_pair_contact_review", "/root/phase4_dispatch_review"}
    assert all(entry["verdict"] == "approved"
        and entry["reviewed_head"] ==
        "983183665fc32fc4f8d6d501a0258ebb7e25a6a0"
        for entry in reactive["reviewer"]["approvals"])
    assert "19 focused CPU contract tests passed" in reactive["limits"]
    assert "slot 2 is out of scope" in reactive["limits"]
    assert "No simulator behavior validation or replay was run" in reactive["limits"]
    assert "qualification eligibility remains false" in reactive["limits"]
    reactive_summary = (REPO / reactive["planner"]["artifact"]["path"]).read_text()
    assert "helper-only API was not connected" in reactive_summary
    assert "Family-specific outcomes, including interception rate" in reactive_summary
    start_profiles = next(t for t in result["tranches"]
        if t["requirement_id"] == "phase4-legal-team-start-profiles")
    assert start_profiles["implementer"]["commit"] == (
        "28f906961115adffb11a3e7e11e4188e4660dcd9")
    assert start_profiles["reviewer"]["approved_commit"] == (
        start_profiles["implementer"]["commit"])
    assert start_profiles["implementer"]["paths"] == [
        "frc_defense/architecture_scenarios.py",
        "frc_defense/curriculum_scenarios.py",
        "frc_defense/meco_ablation_matrix.py",
        "frc_defense/meco_phase4_campaign_plan.py",
        "frc_defense/meco_phase4_context.py",
        "frc_defense/meco_phase4_runtime_adapter.py",
        "frc_defense/meco_track_training.py",
        "frc_defense/meco_validation_campaign_runner.py",
        "frc_defense/tensor_3v3.py",
        "scripts/run_meco_ablation_matrix.py",
        "tests/test_meco_phase4_campaign_plan.py",
        "tests/test_meco_phase4_context.py",
        "tests/test_meco_phase4_runtime_adapter_contract.py"]
    assert start_profiles["planner"]["artifact"]["sha256"] == (
        start_profiles["reviewer"]["artifact"]["sha256"])
    assert start_profiles["limits"].count("No simulator") == 1
    assert "outcome_by_start_state is null/unavailable" in start_profiles["limits"]
    assert "qualification eligibility remains false" in start_profiles["limits"]
    campaign_manifest = next(t for t in result["tranches"]
        if t["requirement_id"] == "meco-five-track-campaign-manifest")
    assert campaign_manifest["implementer"]["commit"] == (
        "bc7eb512512a134272d84b96ba11c8767d67a596")
    assert campaign_manifest["implementer"]["candidate_integration_commit"] == (
        "50aaef203da7cee6e97569d865628444241070c1")
    assert campaign_manifest["reviewer"]["approved_commit"] == (
        campaign_manifest["implementer"]["commit"])
    assert "launch readiness remain false" in campaign_manifest["limits"]
    context_dispatch = next(t for t in result["tranches"]
        if t["requirement_id"] == "phase4-context-paired-dispatch")
    assert context_dispatch["implementer"]["commit"] == (
        "509b219d3dc6c18ff177b03e8532785268be4253")
    assert context_dispatch["reviewer"]["approved_commit"] == (
        context_dispatch["implementer"]["commit"])
    assert "No simulator" in context_dispatch["limits"]
    manifest_supervisor = next(t for t in result["tranches"]
        if t["requirement_id"] == "meco-supervisor-manifest-device-lease")
    assert manifest_supervisor["implementer"]["commit"] == (
        "5d81713310e9400b12fead981f118913f83222f0")
    assert manifest_supervisor["implementer"]["candidate_integration_commit"] == (
        "55e9885961f5c3c5d2643b6652d7d2b238aac40d")
    assert manifest_supervisor["reviewer"]["approved_commit"] == (
        manifest_supervisor["implementer"]["commit"])
    assert "Phase 2" in manifest_supervisor["limits"]
    assert "qualification remains false" in manifest_supervisor["limits"]
    phase2_campaign = next(t for t in result["tranches"]
        if t["requirement_id"] == "meco-c1-c2-phase2-campaign")
    assert phase2_campaign["implementer"]["commit"] == (
        "82a1e8146aec2ad8d2172c0786d10a7e7f41d895")
    assert phase2_campaign["implementer"]["candidate_integration_commit"] == (
        "4d80247942a3453651def4546a2735184cd5092b")
    assert phase2_campaign["reviewer"]["approved_commit"] == (
        phase2_campaign["implementer"]["commit"])
    assert "No simulator" in phase2_campaign["limits"]
    profile_summary = (REPO / start_profiles["planner"]["artifact"]["path"]).read_text()
    assert "defers its initial perception" in profile_summary
    assert "legal-start rate and outcome-by-start-state" in profile_summary
    action7 = next(t for t in result["tranches"]
        if t["requirement_id"] == "phase4-action7-remask-taxonomy")
    assert action7["implementer"]["commit"] == (
        "accce779be3d1a9e87092e22ac45ae7a6543d01d")
    assert action7["implementer"]["candidate_integration_commit"] == (
        "4d040f5e2238e652e90c745300394e07aa1a575f")
    assert action7["reviewer"]["approved_commit"] == action7["implementer"]["commit"]
    assert "not shown to be a learned policy choice" in action7["limits"]
    root_fix = next(t for t in result["tranches"]
        if t["requirement_id"] == "dashboard-curriculum-external-data-root")
    assert root_fix["implementer"]["commit"] == (
        "819836eba213eb4ae1141044f0d2a9993d062b56")
    assert root_fix["implementer"]["candidate_integration_commit"] == (
        "9597d16e33d18d6dea5ea4f2ee6d6e0715191428")
    assert root_fix["reviewer"]["approved_commit"] == root_fix["implementer"]["commit"]
    assert "No simulator evaluation or replay was run" in root_fix["limits"]
    route_outcome = next(t for t in result["tranches"]
        if t["requirement_id"] == "phase4-route-outcome-capture")
    assert route_outcome["implementer"]["commit"] == (
        "387a0ef0e5a50bb986af923800245e0ef965624d")
    assert route_outcome["implementer"]["source_head"] == (
        "54219fece5ca0d3ba9820d77e38108a3a8f71ce4")
    assert route_outcome["implementer"]["reviewed_diff_sha256"] == (
        "5664272c5dfac610cefbf486f14119e6ea9196d7888428befa299253df82610c")
    assert route_outcome["implementer"]["candidate_integration_commit"] == (
        "b8f30d9e5a92675d069a740cdca070d04bc330c5")
    assert route_outcome["reviewer"]["approved_commit"] == (
        route_outcome["implementer"]["commit"])
    assert route_outcome["reviewer"]["integrated_approved_commit"] == (
        route_outcome["implementer"]["candidate_integration_commit"])
    assert route_outcome["planner"]["artifact"]["sha256"] == (
        "435c2d45e891a910894b4b3148684db6842be842c76cfbaab518ae2c3001d4a8")
    assert "41 focused route/runtime contracts passed before" in route_outcome["limits"]
    assert "50 selected-goal, Phase 4 runtime, and execution-failure taxonomy tests" in route_outcome["limits"]
    assert "No simulator evaluation, replay, GPU/HIP run, training, dashboard restart" in route_outcome["limits"]
    assert "qualification eligibility remains false" in route_outcome["limits"]
    route_summary = (REPO / route_outcome["planner"]["artifact"]["path"]).read_text()
    assert "lifecycle identity" in route_summary
    assert "selected destination node" in route_summary
    assert "D0 has no selected-goal ledger" in route_summary
    focal_pose = next(t for t in result["tranches"]
        if t["requirement_id"] == "phase4-focal-pose-assignment")
    assert focal_pose["implementer"]["commit"] == (
        "fd9ba22eaa35919fcd206323f7615ca8c8166c07")
    assert focal_pose["implementer"]["source_head"] == (
        "b8f30d9e5a92675d069a740cdca070d04bc330c5")
    assert focal_pose["implementer"]["reviewed_diff_sha256"] == (
        "1c9ff7ced88c5175dfc0fba97a83d6a2f6cbdd078839c7b8fa5a8ee0371aee71")
    assert focal_pose["implementer"]["candidate_integration_commit"] == (
        "c4425862bdaac32e9f5206bdcd2a22fe0626692d")
    assert focal_pose["reviewer"]["approved_commit"] == (
        focal_pose["implementer"]["commit"])
    assert focal_pose["implementer"]["paths"] == [
        "frc_defense/curriculum_scenarios.py",
        "frc_defense/meco_ablation_matrix.py",
        "frc_defense/meco_phase4_context.py",
        "frc_defense/meco_phase4_runtime_adapter.py",
        "frc_defense/meco_track_training.py",
        "scripts/run_meco_ablation_matrix.py",
        "tests/test_meco_focal_pose_boundary.py",
        "tests/test_meco_phase4_runtime_adapter_contract.py"]
    assert focal_pose["planner"]["artifact"]["sha256"] == (
        "0b4096e1588b5fffd80fc9d4a0010d47102720fe1f69fa531db02d9c5c780372")
    assert "131 focused CPU contract tests" in focal_pose["limits"]
    for excluded in ("No simulator evaluation", "E2E replay", "GPU/HIP run",
                     "training", "dashboard currentness check", "qualification"):
        assert excluded in focal_pose["limits"]
    focal_summary = (REPO / focal_pose["planner"]["artifact"]["path"]).read_text()
    assert "generic `phase4_factor_exposure`" in focal_summary
    assert "initial-only" in focal_summary



def test_plan_c_checkpoint_d_is_pending_and_fail_closed():
    data = json.loads(MANIFEST.read_text())
    assert len(data["tranches"]) == 89
    row = data["tranches"][-1]
    assert row["requirement_id"] == "meco-plan-c-route-device-residency"
    assert row["planner"]["artifact"] == {
        "path": "docs/plans/gpu-resident-route-decision-slice-20261025.md",
        "sha256": "8c6c67ffb2934fa3a4ce40b54c5cc2f2e2cb4877b5eedac9c31a2936a6879b5a",
        "kind": "scoped-transcript-summary",
        "status": "transcript-summary",
    }
    implementer = row["implementer"]
    assert implementer["code_candidate_commit"] == (
        "cdeb9391bf23e5c2460527959132008aee6f8eb1")
    assert implementer["code_review_approved_commit"] == (
        implementer["code_candidate_commit"])
    assert implementer["code_review_verdict"] == "approved"
    assert implementer["code_review_artifact"]["status"] == "review-summary"
    code_review = REPO / implementer["code_review_artifact"]["path"]
    assert hashlib.sha256(code_review.read_bytes()).hexdigest() == (
        implementer["code_review_artifact"]["sha256"])
    d_summary = REPO / implementer["ledger_candidate_summary_artifact"]["path"]
    assert hashlib.sha256(d_summary.read_bytes()).hexdigest() == (
        implementer["ledger_candidate_summary_artifact"]["sha256"])
    assert (REPO / "docs/implementation-provenance/plan-c-device-resident-selection-currentness-20261025.md").is_file()
    status_text = (REPO / "docs/MECO_FULL_PLAN_EXECUTION_STATUS.md").read_text()
    assert "Ledger checkpoint D records that C approval but remains pending" in status_text
    assert implementer["code_candidate_paths"] == implementer["paths"]
    assert len(implementer["code_candidate_paths"]) == 6
    assert "candidate_commit" not in implementer
    assert "approved_commit" not in row["reviewer"]
    assert "artifact" not in row["reviewer"]
    assert row["reviewer"]["status"] == "pending"
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == (
        dashboard_api._EXPECTED_THREE_PASS_PROVENANCE_SHA256)
    result = validate(REPO, data)
    assert result["status"] == "untrusted"
    assert "reviewer evidence" in result["untrusted_reason"]
    api_result = dashboard_api._three_pass_provenance()
    assert api_result["status"] == "untrusted"
    assert api_result["tranches"] == []
    assert "reviewer evidence" in api_result["untrusted_reason"]

def test_valid_fixture_passes_structural_and_hash_checks(evidence_case):
    project, data = evidence_case
    result = validate(project, data)
    assert result["validation_status"] == "validated_summary_only"
    assert [item["requirement_id"] for item in result["tranches"]]


@pytest.mark.parametrize("mutate,reason", [
    (lambda d: d["tranches"].append(copy.deepcopy(d["tranches"][0])), "duplicated"),
    (lambda d: d["tranches"][0]["implementer"].update(commit="abc123"), "40-character"),
    (lambda d: d["tranches"][0]["implementer"].update(paths=[]), "paths are missing"),
    (lambda d: d["tranches"][0]["reviewer"].update(verdict="requested_changes"), "approved summary"),
    (lambda d: d["tranches"][0]["reviewer"].update(approved_commit="abc123"), "reviewer approved commit is not an exact"),
    (lambda d: d["tranches"][0]["implementer"].update(status="running"), "implementation status"),
    (lambda d: d["tranches"][0]["planner"].update(status="original-transcript"), "planner summary artifact"),
    (lambda d: d["tranches"][0]["reviewer"]["artifact"].update(sha256="0" * 64), "reviewer evidence"),
    (lambda d: d["tranches"][0].update(limits=""), "limits are missing"),
])
def test_malformed_or_tampered_evidence_fails_closed(evidence_case, mutate, reason):
    project, data = evidence_case
    changed = copy.deepcopy(data)
    mutate(changed)
    result = validate(project, changed)
    assert result["status"] == "untrusted"
    assert reason in result["untrusted_reason"]
    assert result["tranches"] == []


def test_artifact_path_escape_and_changed_path_symlink_fail_closed(evidence_case):
    project, data = evidence_case
    escaped = copy.deepcopy(data)
    escaped["tranches"][0]["planner"]["artifact"]["path"] = "../outside.md"
    assert validate(project, escaped)["status"] == "untrusted"

    linked = copy.deepcopy(data)
    first_path = linked["tranches"][0]["implementer"]["paths"][0]
    target = project / "target-file"
    target.write_text("target")
    path = project / first_path
    path.unlink()
    path.symlink_to(target)
    result = validate(project, linked)
    assert result["status"] == "untrusted"
    assert "implementation path is missing or unsafe" in result["untrusted_reason"]


def test_untrusted_api_payload_explains_manifest_digest_failure(monkeypatch):
    monkeypatch.setattr(dashboard_api, "_EXPECTED_THREE_PASS_PROVENANCE_SHA256", "0" * 64)
    result = dashboard_api._three_pass_provenance()
    assert result["status"] == "untrusted"
    assert result["tranches"] == []
    assert "manifest hash" in result["untrusted_reason"]


def test_dashboard_frontend_renders_validated_payload_separately_from_liveness():
    source = (REPO / "frc_defense/dashboard_static/meco.js").read_text()
    assert "/api/curriculum-status" in source
    assert "validation_status === \"validated_summary_only\"" in source
    assert "untrusted_reason" in source
    assert 'roleEvidence("planner"' in source
    assert 'roleEvidence("implementer"' in source
    assert 'roleEvidence("reviewer"' in source
    assert "does not report agent liveness" in source
    assert "whole-plan approval" in source
