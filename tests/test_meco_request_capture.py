import time

import pytest
import torch

from frc_defense import meco_request_capture as capture
from frc_defense.meco_input_provenance import input_contract_digest
from frc_defense.meco_tracks import (
    MECO_INPUT_NAMES, MecoInferenceRequest, ordered_candidate_mask_digest)


def _request_and_inputs():
    names = ("globals_", "candidate_features", "hidden")
    candidate_ids = ("candidate-0", "candidate-1")
    valid_mask = (True, True)
    now = time.monotonic()
    request = MecoInferenceRequest(
        track="C1", snapshot_id="snapshot-1", checkpoint_sha256="a" * 64,
        protocol_id="protocol-1", observation_schema=1,
        observation_timestamp=1., capture_monotonic=now,
        dispatch_monotonic=now, expires_monotonic=now + .25,
        input_names=names, input_shapes=((1, 7), (1, 2, 40), (1, 64)),
        candidate_ids=candidate_ids, valid_mask=valid_mask, fallback_index=0,
        request_sequence=1, generation_id="generation-1",
        candidate_mask_digest=ordered_candidate_mask_digest(candidate_ids, valid_mask),
        input_contract_sha256=input_contract_digest(), model_id="meco:C1")
    inputs = {
        "globals_": torch.arange(7, dtype=torch.float32).reshape(1, 7),
        "candidate_features": torch.arange(80, dtype=torch.float32).reshape(1, 2, 40),
        "hidden": torch.zeros((1, 64), dtype=torch.float32),
    }
    return request, inputs


def test_capture_round_trip_binds_input_contract_and_tensor_content(tmp_path, monkeypatch):
    request, inputs = _request_and_inputs()
    monkeypatch.setattr(capture, "_CAPTURED", 0)
    monkeypatch.setenv("MECO_REQUEST_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("MECO_REQUEST_CAPTURE_MAX", "1")
    path = capture.capture_request(request, inputs, source_snapshot_sha256="b" * 64)
    metadata, loaded = capture.load_capture(path, expected_track="C1")
    assert metadata["input_contract_sha256"] == input_contract_digest()
    assert tuple(torch.equal(a, b) for a, b in zip(inputs.values(), loaded)) == (True, True, True)
    capture.validate_capture_inputs(metadata, loaded, expected_track="C1")


def test_capture_rejects_tensor_content_drift(tmp_path, monkeypatch):
    request, inputs = _request_and_inputs()
    monkeypatch.setattr(capture, "_CAPTURED", 0)
    monkeypatch.setenv("MECO_REQUEST_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("MECO_REQUEST_CAPTURE_MAX", "1")
    path = capture.capture_request(request, inputs, source_snapshot_sha256="b" * 64)
    metadata, loaded = capture.load_capture(path, expected_track="C1")
    changed = list(loaded)
    changed[0] = changed[0].clone()
    changed[0][0, 0] += 1
    with pytest.raises(ValueError, match="captured tensor"):
        capture.validate_capture_inputs(metadata, tuple(changed), expected_track="C1")


def test_capture_load_rejects_metadata_digest_tampering(tmp_path, monkeypatch):
    request, inputs = _request_and_inputs()
    monkeypatch.setattr(capture, "_CAPTURED", 0)
    monkeypatch.setenv("MECO_REQUEST_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("MECO_REQUEST_CAPTURE_MAX", "1")
    path = capture.capture_request(request, inputs, source_snapshot_sha256="b" * 64)
    metadata, loaded = capture.load_capture(path, expected_track="C1")
    metadata["input_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="metadata digest mismatch"):
        capture.validate_capture_inputs(metadata, loaded, expected_track="C1")


def test_capture_load_rejects_registry_drift(tmp_path, monkeypatch):
    request, inputs = _request_and_inputs()
    monkeypatch.setattr(capture, "_CAPTURED", 0)
    monkeypatch.setenv("MECO_REQUEST_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("MECO_REQUEST_CAPTURE_MAX", "1")
    path = capture.capture_request(request, inputs, source_snapshot_sha256="b" * 64)
    altered = dict(MECO_INPUT_NAMES)
    altered["C1"] = ("candidate_features", "globals_", "hidden")
    monkeypatch.setattr("frc_defense.meco_tracks.MECO_INPUT_NAMES", altered)
    with pytest.raises(ValueError, match="serialized track input order"):
        capture.load_capture(path, expected_track="C1")
