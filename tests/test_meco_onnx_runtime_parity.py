"""CPU-only synthetic ONNX graph execution contracts; not production captures."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime", reason="optional ONNX Runtime parity dependencies")
onnx = pytest.importorskip("onnx", reason="optional ONNX exporter dependencies")

from frc_defense.meco_deployment import ExportAdapter, dummy_inputs, input_names
from frc_defense.meco_tracks import MECO_TRACKS, make_meco_track

_PARITY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_meco_onnx_parity.py"
_PARITY_SPEC = importlib.util.spec_from_file_location("meco_parity_verifier", _PARITY_PATH)
parity = importlib.util.module_from_spec(_PARITY_SPEC)
_PARITY_SPEC.loader.exec_module(parity)


def _synthetic_provenance(track: str) -> dict:
    return {"sha256": hashlib.sha256(track.encode()).hexdigest(),
            "run_id": f"synthetic-{track}", "update": 1,
            "protocol_id": "synthetic-contract", "sources": {"fixture.py": "0" * 64}}


@pytest.fixture(scope="module")
def exported_graphs(tmp_path_factory):
    """Export real model constructors with seeded random weights, then run via ORT."""
    root = tmp_path_factory.mktemp("onnx-synthetic")
    models = {}
    torch.manual_seed(713)
    for track in MECO_TRACKS:
        model = make_meco_track(track).cpu().eval()
        models[track] = model
        wrapper = ExportAdapter(track, model).eval()
        tensors = dummy_inputs(track)
        graph = root / f"{track}.onnx"
        output_names = parity.OUTPUT_NAMES[track]
        torch.onnx.export(wrapper, tensors, str(graph),
                          input_names=input_names(track), output_names=output_names,
                          opset_version=17, do_constant_folding=True, dynamo=False)
    return root, models


def _bind_synthetic_inputs(monkeypatch, models):
    def load(track, checkpoint):
        return models[track], _synthetic_provenance(track)
    monkeypatch.setattr(parity, "load_track_model", load)


def test_executes_exported_graphs_for_all_five_tracks(tmp_path, monkeypatch, exported_graphs):
    graph_root, models = exported_graphs
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    for track in MECO_TRACKS:
        checkpoint = checkpoint_root / track / "latest.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(f"synthetic checkpoint {track}".encode())
    _bind_synthetic_inputs(monkeypatch, models)

    report = parity.verify_onnx_parity(checkpoint_root, graph_root, 8128,
                                       "CPUExecutionProvider", 1e-5, 1e-4)

    assert report["status"] == "pass", [(row["track"], row.get("error"), row.get("traceback"),
                                          [(out.get("error"), out.get("max_abs_error"))
                                           for out in row.get("outputs", [])])
                                         for row in report["tracks"]]
    assert report["input_mode"] == "synthetic_seeded"
    assert report["onnxruntime_version"] == ort.__version__
    assert "CPUExecutionProvider" in report["available_providers"]
    assert len(report["tracks"]) == 5
    for row in report["tracks"]:
        assert row["status"] == "pass"
        assert row["onnx_sha256"]
        assert row["checkpoint_sha256"]
        assert row["input_sha256"]
        assert row["inputs"]
        assert row["outputs"] and all(item["allclose"] for item in row["outputs"])


def test_rejects_onnx_input_name_drift(tmp_path, monkeypatch, exported_graphs):
    source_root, models = exported_graphs
    graph_root = tmp_path / "graphs"
    graph_root.mkdir()
    for track in MECO_TRACKS:
        (graph_root / f"{track}.onnx").write_bytes(
            (source_root / f"{track}.onnx").read_bytes())
    track = MECO_TRACKS[0]
    graph_path = graph_root / f"{track}.onnx"
    graph = onnx.load(str(graph_path))
    old_name = graph.graph.input[0].name
    graph.graph.input[0].name = "wrong_input_name"
    for node in graph.graph.node:
        node.input[:] = ["wrong_input_name" if name == old_name else name
                         for name in node.input]
    onnx.save(graph, str(graph_path))
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / track / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint")
    _bind_synthetic_inputs(monkeypatch, models)

    report = parity.verify_onnx_parity(checkpoint_root, graph_root, 8128,
                                       "CPUExecutionProvider", 1e-5, 1e-4)

    assert report["status"] == "failed"
    assert "ONNX input names/order mismatch" in report["tracks"][0]["error"]


def test_accepts_production_export_adapter_tuple_outputs(tmp_path, monkeypatch, exported_graphs):
    graph_root, models = exported_graphs
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    for track in MECO_TRACKS:
        checkpoint = checkpoint_root / track / "latest.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(f"synthetic checkpoint {track}".encode())
    monkeypatch.setattr(parity, "load_track_model",
                        lambda track, checkpoint: (ExportAdapter(track, models[track]).eval(),
                                                    _synthetic_provenance(track)))

    report = parity.verify_onnx_parity(checkpoint_root, graph_root, 8128,
                                       "CPUExecutionProvider", 1e-5, 1e-4)

    assert report["status"] == "pass", [(row["track"], row.get("error"))
                                          for row in report["tracks"]]
    assert all(row["status"] == "pass" for row in report["tracks"])


def test_rejects_output_values_outside_declared_tolerance():
    expected = (torch.zeros((1, 2), dtype=torch.float32),) * 4
    actual = [np.full((1, 2), 1.0, dtype=np.float32) for _ in range(4)]
    compared = parity._compare_outputs("A1", expected, actual, atol=1e-5, rtol=1e-4)
    assert not any(item["allclose"] for item in compared)
