import hashlib

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
