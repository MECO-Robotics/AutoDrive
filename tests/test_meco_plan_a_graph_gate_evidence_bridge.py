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
