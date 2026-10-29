import hashlib
import json

import pytest

from scripts.verify_plan_a_graph_gate_evidence import (
    CLOSED, OPEN, PREWEIGHTED_CLOSED, normalized_graph_source, check_report,
)


def test_normalization_only_accepts_the_graph_gate_literal():
    closed = PREWEIGHTED_CLOSED + b"\n" + CLOSED + b"\nimplementation\n"
    opened = closed.replace(CLOSED, OPEN)
    assert normalized_graph_source(opened) == closed
    assert normalized_graph_source(closed) == closed
    altered = opened.replace(b"implementation", b"changed implementation")
    assert hashlib.sha256(normalized_graph_source(altered)).digest() != hashlib.sha256(closed).digest()
    with pytest.raises(ValueError):
        normalized_graph_source(opened.replace(PREWEIGHTED_CLOSED, PREWEIGHTED_CLOSED.replace(b"False", b"True")))
    with pytest.raises(ValueError):
        normalized_graph_source(opened + b"\n" + CLOSED)


def test_report_must_match_independently_supplied_hash(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"passed": true}')
    with pytest.raises(ValueError, match="reviewed hash"):
        check_report(tmp_path, report, "0" * 64, {}, b"")


def test_device_attestation_requires_uuid_model_and_bdf():
    from copy import deepcopy
    from scripts.verify_plan_a_graph_gate_evidence import check_device

    report = {
        "requested_device": "cuda:0",
        "device": {"index": 0, "model": "AMD Radeon Pro WX 9100", "uuid": "uuid0"},
        "lease_authorization": {
            "logical_lease_devices": ["cuda:0"], "logical_lease_id": "lease",
            "physical_attestation": {"devices": [{
                "device": "cuda:0", "hip_uuid": "uuid0",
                "model": "AMD Radeon Pro WX 9100", "pci_bdf": "0000:19:00.0",
            }]},
        },
    }
    assert check_device(report, "report")["pci_bdf"] == "0000:19:00.0"
    for key in ("hip_uuid", "model", "pci_bdf"):
        changed = deepcopy(report)
        changed["lease_authorization"]["physical_attestation"]["devices"][0][key] = None
        if key == "hip_uuid":
            changed["device"]["uuid"] = None
        with pytest.raises(ValueError, match="attestation incomplete"):
            check_device(changed, "report")


@pytest.fixture
def valid_graph_report(tmp_path, monkeypatch):
    from scripts import verify_plan_a_graph_gate_evidence as verifier

    graph_source = PREWEIGHTED_CLOSED + b"\n" + CLOSED
    sources = {verifier.GRAPH_SOURCE: hashlib.sha256(graph_source).hexdigest()}
    case = {"id": "fixture", "route_kind": "geometry", "graph_sha256": "graph",
            "tensor_sha256": "tensor", "route_output_field_sha256": "route",
            "support_digest_sha256": "support"}
    manifest = {"source_sha256": sources, "cases": [case]}
    manifest["manifest_payload_sha256"] = verifier.sha256_bytes(verifier.canonical_bytes(manifest))
    manifest_bytes = json.dumps(manifest).encode()
    manifest_path = tmp_path / verifier.FIXTURE_RELATIVE
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(manifest_bytes)
    monkeypatch.setattr(verifier, "source_hashes", lambda root: sources)
    monkeypatch.setattr(verifier, "git_bytes", lambda root, commit, path:
                        manifest_bytes if path == verifier.FIXTURE_RELATIVE else graph_source)
    graph = {"path": "graph_geometry", "status": "candidate_pass", "exact_output_match": True,
             "support_digest_match": True, "route_output_field_sha256": "route",
             "support_digest_sha256": "support", "warmups": 1, "repeats": 3,
             "hip_elapsed_ms": [1.0, 1.1, 1.2],
             "compared_output_fields": [{"field": name, "exact_equal": True} for name in verifier.OUTPUT_KEYS]}
    report = {"schema_version": 1, "evidence_class": "hip_parity_profile_candidate", "passed": True,
              "git_dirty": False, "source_identity_verified_at_start_and_end": True,
              "production_gates_open": {"graph": False, "preweighted": False},
              "git_commit": "a" * 40, "source_sha256": sources,
              "fixture_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "cases": [{"id": "fixture", "graph_sha256": "graph", "tensor_sha256": "tensor", "hip_paths": [graph]}],
              "requested_device": "cuda:0", "device": {"index": 0, "model": "AMD Radeon Pro WX 9100", "uuid": "uuid0"},
              "lease_authorization": {"logical_lease_devices": ["cuda:0"], "logical_lease_id": "lease",
                  "physical_attestation": {"devices": [{"device": "cuda:0", "hip_uuid": "uuid0",
                      "model": "AMD Radeon Pro WX 9100", "pci_bdf": "0000:19:00.0"}]}}}

    def verify():
        raw = json.dumps(report).encode()
        path = tmp_path / "report.json"
        path.write_bytes(raw)
        return verifier.check_report(tmp_path, path, hashlib.sha256(raw).hexdigest(), sources, graph_source)

    return report, verify


def test_valid_report_is_accepted(valid_graph_report):
    _, verify = valid_graph_report
    assert verify()["device"] == "cuda:0"


def test_duplicate_path_is_rejected(valid_graph_report):
    report, verify = valid_graph_report
    paths = report["cases"][0]["hip_paths"]
    paths.insert(0, {"path": "graph_geometry", "status": "candidate_failed"})
    with pytest.raises(ValueError, match="duplicate HIP path"):
        verify()


@pytest.mark.parametrize("paths", [None, [None], [{}], [{"path": ""}]])
def test_malformed_paths_are_rejected(valid_graph_report, paths):
    report, verify = valid_graph_report
    report["cases"][0]["hip_paths"] = paths
    with pytest.raises(ValueError, match="malformed HIP path"):
        verify()


@pytest.mark.parametrize("field", ["warmups", "repeats", "hip_elapsed_ms"])
def test_boolean_metrics_are_rejected(valid_graph_report, field):
    report, verify = valid_graph_report
    graph = report["cases"][0]["hip_paths"][0]
    graph[field] = [True, 1.1, 1.2] if field == "hip_elapsed_ms" else True
    with pytest.raises(ValueError, match="parity/profile incomplete"):
        verify()


@pytest.mark.parametrize("field,value", [
    ("compared_output_fields", None), ("compared_output_fields", {}),
    ("compared_output_fields", [None]), ("hip_elapsed_ms", None),
    ("hip_elapsed_ms", 1), ("hip_elapsed_ms", "1.0"),
])
def test_malformed_comparisons_and_timings_are_rejected(valid_graph_report, field, value):
    report, verify = valid_graph_report
    report["cases"][0]["hip_paths"][0][field] = value
    with pytest.raises(ValueError, match="parity/profile incomplete"):
        verify()
