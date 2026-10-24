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
    assert len(result["tranches"]) == 16
    lease = next(t for t in result["tranches"]
                 if t["requirement_id"] == "meco-evaluation-device-lease")
    assert lease["implementer"]["commit"] == "9e41078fd438767217d051b89a58b2110449b3fb"
    assert lease["reviewer"]["approved_commit"] == lease["implementer"]["commit"]
    trace = next(t for t in result["tranches"]
                 if t["requirement_id"] == "meco-inference-observability-decision-trace")
    assert trace["implementer"]["commit"] == "7cfe3d07696167d931ced416780435c4f13ddfc1"
    assert trace["reviewer"]["approved_commit"] == trace["implementer"]["commit"]
    assert "not whole-plan approval" in result["message"]
    assert all(t["planner"]["artifact"]["kind"] == "scoped-transcript-summary"
               for t in result["tranches"])


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
