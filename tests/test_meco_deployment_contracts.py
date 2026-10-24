import hashlib
import json

import pytest
import torch

from frc_defense import meco_deployment
from frc_defense.meco_tracks import MECO_TRACKS


@pytest.mark.parametrize("track", MECO_TRACKS)
def test_export_adapter_values_match_scorer(track):
    scorer = meco_deployment.make_meco_track(
        track, candidate_features=meco_deployment.CANDIDATE_BASE_FEATURE_COUNT
    ).cpu().eval()
    adapter = meco_deployment.ExportAdapter(track, scorer).cpu().eval()
    inputs = meco_deployment.dummy_inputs(track, device="cpu")
    names = meco_deployment.input_names(track)

    with torch.inference_mode():
        expected = scorer(**dict(zip(names, inputs)))
        actual = adapter(*inputs)

    if "utility" in expected:
        expected_values = (expected["utility"], expected["penalty"],
                           expected["behavior_logits"], expected["hidden"])
    else:
        expected_values = (expected["candidate_utility"], expected["uncertainty"],
                           expected["hidden"])
    assert len(actual) == len(expected_values)
    for exported, scorer_value in zip(actual, expected_values):
        assert exported.shape == scorer_value.shape
        assert exported.dtype == scorer_value.dtype
        assert torch.equal(exported, scorer_value)


def test_profiler_fails_closed_when_checkpoint_run_manifest_is_missing(tmp_path):
    checkpoint = tmp_path / "A1" / "latest.pt"
    checkpoint.parent.mkdir()
    model = meco_deployment.make_meco_track("A1")
    torch.save({"schema_version": 1, "config": {"track": "A1"},
                "sources": {"trainer": "a" * 64}, "run_id": "run-1",
                "update": 1, "protocol_id": "protocol-1",
                "model": model.state_dict()}, checkpoint)

    with pytest.raises(ValueError, match="run manifest is missing"):
        meco_deployment.benchmark("A1", checkpoint, "cpu", iterations=1, warmup=0)
    with pytest.raises(ValueError, match="run manifest is missing"):
        meco_deployment.export_track("A1", checkpoint, tmp_path / "A1.onnx")


def _write_valid_a1_checkpoint(checkpoint):
    scorer = meco_deployment.make_meco_track("A1")
    config = {"track": "A1", "updates": 1}
    sources = {"trainer": "a" * 64}
    protocol_id = "protocol-1"
    payload = {"schema_version": 1, "config": config, "sources": sources,
               "run_id": "run-1", "update": 1, "protocol_id": protocol_id,
               "model": scorer.state_dict()}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    source_digest = hashlib.sha256(json.dumps(
        sources, sort_keys=True, allow_nan=False).encode()).hexdigest()
    immutable_config = {key: value for key, value in config.items() if key != "updates"}
    config_digest = hashlib.sha256(json.dumps(
        immutable_config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    manifest = {"schema_version": 1, "run_id": "run-1", "sources": sources,
        "config": config, "protocol_id": protocol_id, "status": "complete",
        "completed_updates": 1, "source_identity_sha256": source_digest,
        "configuration_sha256": config_digest}
    (checkpoint.parent / "run.json").write_text(json.dumps(manifest))
    return hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_deployment_export_records_checkpoint_and_tensor_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "A1" / "latest.pt"
    checkpoint_digest = _write_valid_a1_checkpoint(checkpoint)
    captured = {}

    def fake_onnx_export(model, tensors, output, *, input_names, output_names, **_kwargs):
        captured["input_names"] = input_names
        captured["output_names"] = output_names
        with torch.inference_mode():
            captured["outputs"] = model(*tensors)
        from pathlib import Path
        Path(output).write_bytes(b"onnx-fixture")

    monkeypatch.setattr(meco_deployment.torch.onnx, "export", fake_onnx_export)
    monkeypatch.setattr(meco_deployment, "operator_inventory",
                        lambda _path: {"inspected": False})
    report = meco_deployment.export_track("A1", checkpoint, tmp_path / "A1.onnx")

    assert captured["input_names"] == meco_deployment.input_names("A1")
    assert captured["output_names"] == ("utility", "penalty", "behavior_logits", "hidden")
    assert len(captured["outputs"]) == len(captured["output_names"])
    assert report["checkpoint_provenance"]["sha256"] == checkpoint_digest
    assert report["size_bytes"] == len(b"onnx-fixture")
    assert report["graph"] == {"inspected": False}


def test_synthetic_profile_keeps_timing_labels_and_checkpoint_provenance(tmp_path):
    checkpoint = tmp_path / "A1" / "latest.pt"
    checkpoint_digest = _write_valid_a1_checkpoint(checkpoint)
    result = meco_deployment.benchmark("A1", checkpoint, "cpu",
                                       iterations=1, warmup=0)

    provenance = result["checkpoint_provenance"]
    assert provenance["sha256"] == checkpoint_digest
    assert provenance["run_id"] == "run-1"
    assert provenance["update"] == 1
    assert provenance["sources"] == {"trainer": "a" * 64}
    assert provenance["training_status"] == "complete"
    assert provenance["manifest_completed_updates"] == 1
    assert result["synthetic_diagnostic_only"] is True
    assert result["deployment_evidence"] is False
    assert result["feature_route_feasibility_action_pipeline_measured"] is False
    assert result["age_proxy_independent_measurement"] is False
    assert result["same_timing_relabelled_as_age_proxy_ms"] == result[
        "synthetic_raw_scorer_argmax_ms"]
    assert result["host_to_device_transfer"] == (
        "excluded: inputs preloaded on device; measure separately on hardware")


def test_profile_report_labels_synthetic_evidence_and_unvalidated_hailo(monkeypatch,
                                                                        tmp_path):
    monkeypatch.setattr(meco_deployment, "benchmark",
        lambda track, *_args, **_kwargs: {"track": track,
                                          "synthetic_diagnostic_only": True})
    monkeypatch.setattr(meco_deployment.torch.cuda, "is_available", lambda: False)
    report = meco_deployment.profile_report(("A1",), tmp_path, iterations=1, warmup=0)

    assert report["evidence_class"] == (
        "synthetic trained-scorer microbenchmark; not deployment qualification")
    assert report["hailo"]["validated"] is False
    assert "no physical actuation is measured" in report["limitations"][3]
