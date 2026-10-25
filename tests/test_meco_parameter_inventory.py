import pytest
import torch

from frc_defense import meco_deployment
from frc_defense.meco_tracks import MECO_TRACKS

EXPECTED_PARAMETER_COUNTS = {
    "A1": 171717,
    "A2": 122311,
    "A3": 118215,
    "C1": 49026,
    "C2": 124162,
}


def test_parameter_inventory_counts_are_stable_and_complete(monkeypatch):
    original_factory = meco_deployment.make_meco_track
    devices = []

    def cpu_factory(track, **kwargs):
        model = original_factory(track, **kwargs)
        devices.extend({parameter.device.type for parameter in model.parameters()})
        return model

    monkeypatch.setattr(meco_deployment, "make_meco_track", cpu_factory)
    first = meco_deployment.parameter_inventory()
    second = meco_deployment.parameter_inventory()
    assert devices and set(devices) == {"cpu"}
    assert first["schema"] == second["schema"] == "meco-parameter-inventory-v1"
    assert first["tracks"] == second["tracks"] == list(MECO_TRACKS)
    assert set(first["models"]) == set(MECO_TRACKS)
    assert first["source_files"] == second["source_files"]
    assert first["source_files_sha256"] == second["source_files_sha256"]
    assert any("do not prove checkpoint/export parity" in item
               for item in first["limitations"])
    assert set(first["source_files"]) == {
        "frc_defense/meco_deployment.py", "frc_defense/meco_tracks.py",
        "frc_defense/meco_candidates.py"}

    for track in MECO_TRACKS:
        model = first["models"][track]
        repeated = second["models"][track]
        for key in ("parameter_count", "trainable_parameter_count", "dtype_totals",
                    "dtype_shape_totals", "per_module_counts", "configuration_sha256"):
            assert model[key] == repeated[key]
        assert model["parameter_count"] > 0
        assert model["parameter_count"] == EXPECTED_PARAMETER_COUNTS[track]
        assert model["trainable_parameter_count"] == model["parameter_count"]
        assert sum(item["parameters"] for item in model["dtype_totals"]) == model["parameter_count"]
        assert sum(item["parameters"] for item in model["dtype_shape_totals"]) == model["parameter_count"]
        assert sum(item["parameters"] for item in model["per_module_counts"]) == model["parameter_count"]
        assert model["configuration"]["device"] == "cpu"
        assert model["configuration"]["output_contract"] == model["output_contract"]
        assert model["configuration_sha256"]


def test_export_and_profile_manifests_include_five_track_inventory(monkeypatch,
                                                                    tmp_path):
    monkeypatch.setattr(meco_deployment, "benchmark",
        lambda track, *_args, **_kwargs: {"track": track, "synthetic_diagnostic_only": True})
    monkeypatch.setattr(meco_deployment.torch.cuda, "is_available", lambda: False)
    export = meco_deployment.export_manifest([{"track": "A1"}])
    profile = meco_deployment.profile_report(("A1",), tmp_path,
        iterations=1, warmup=0)
    for manifest in (export, profile):
        inventory = manifest["parameter_inventory"]
        assert inventory["tracks"] == list(MECO_TRACKS)
        assert set(inventory["models"]) == set(MECO_TRACKS)
    assert export["schema"] == "meco-onnx-export-v1"
    assert profile["schema"] == "meco-synthetic-scorer-profile-v1"


@pytest.mark.parametrize("track,expected_inputs,expected_outputs", [
    ("A1", (("raster", [1, 6, 64, 64]), ("globals_", [1, 7]), ("hidden", [1, 64])),
     (("utility", [1, 8, 64, 64]), ("penalty", [1, 8, 64, 64]),
      ("behavior_logits", [1, 5]), ("hidden_out", [1, 64]))),
    ("A2", (("node_features", [1, 12096, 9]), ("edges", [1, 12096, 10]),
      ("edge_mask", [1, 12096, 10]), ("node_valid", [1, 12096]),
      ("region_ids", [1, 12096]), ("globals_", [1, 7]),
      ("coarse_edges", [1, 64, 64]),
      ("coarse_edge_mask", [1, 64, 64]), ("hidden", [1, 64])),
     (("utility", [1, 12096]), ("penalty", [1, 12096]),
      ("behavior_logits", [1, 5]), ("hidden_out", [1, 64]))),
    ("A3", (("node_features", [1, 12096, 9]), ("edges", [1, 12096, 10]),
      ("edge_mask", [1, 12096, 10]), ("node_valid", [1, 12096]),
      ("region_ids", [1, 12096]), ("globals_", [1, 7]), ("hidden", [1, 64])),
     (("utility", [1, 12096]), ("penalty", [1, 12096]),
      ("behavior_logits", [1, 5]), ("hidden_out", [1, 64]))),
    ("C1", (("globals_", [1, 7]), ("candidate_features", [1, 16, 40]),
      ("hidden", [1, 64])),
     (("candidate_utility", [1, 16]), ("uncertainty", [1, 16]), ("hidden_out", [1, 64]))),
    ("C2", (("raster", [1, 6, 64, 64]), ("globals_", [1, 7]),
      ("candidate_features", [1, 16, 40]), ("hidden", [1, 64])),
     (("candidate_utility", [1, 16]), ("uncertainty", [1, 16]), ("hidden_out", [1, 64]))),
])
def test_inventory_records_export_input_and_output_contracts(track, expected_inputs,
                                                              expected_outputs):
    inventory = meco_deployment.parameter_inventory()
    model = inventory["models"][track]
    assert tuple((item["name"], item["shape"]) for item in model["input_contract"]) == expected_inputs
    assert tuple((item["name"], item["shape"]) for item in model["output_contract"]) == expected_outputs
    assert all(item["dtype"] == "torch.float32" for item in model["output_contract"])
    assert model["configuration"]["track"] == track

    scorer = meco_deployment.make_meco_track(track,
        candidate_features=meco_deployment.CANDIDATE_BASE_FEATURE_COUNT).cpu().eval()
    adapter = meco_deployment.ExportAdapter(track, scorer).cpu().eval()
    inputs = meco_deployment.dummy_inputs(track, device="cpu")
    assert all(tensor.device.type == "cpu" for tensor in inputs)
    with torch.inference_mode():
        outputs = adapter(*inputs)
    assert tuple(tuple(output.shape) for output in outputs) == tuple(
        tuple(shape) for _, shape in expected_outputs)
    assert all(output.device.type == "cpu" for output in outputs)
    assert all(output.dtype == torch.float32 for output in outputs)
