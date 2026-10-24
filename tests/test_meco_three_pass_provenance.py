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
            artifact = tranche[role_name]["artifact"]
            destination = tmp_path / artifact["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / artifact["path"]).read_bytes())
        for relative in tranche["implementer"]["paths"]:
            destination = tmp_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO / relative).read_bytes())
    return tmp_path, data


def validate(project, data):
    return dashboard_api._validate_three_pass_manifest(project, data)


def test_current_manifest_is_validated_as_summary_only():
    result = dashboard_api._three_pass_provenance()
    assert result["validation_status"] == "validated_summary_only"
    assert result["evidence_class"] == "scoped_transcript_summary"
    assert len(result["tranches"]) == 33
    lease = next(t for t in result["tranches"]
                 if t["requirement_id"] == "meco-evaluation-device-lease")
    assert lease["implementer"]["commit"] == "9e41078fd438767217d051b89a58b2110449b3fb"
    assert lease["reviewer"]["approved_commit"] == lease["implementer"]["commit"]
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
