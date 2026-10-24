
import pytest

from frc_defense import meco_input_provenance as provenance
from frc_defense.meco_tracks import MECO_INPUT_NAMES


def test_registry_is_complete_ordered_and_c1_c2_candidate_schema_matches():
    record = provenance.validate_registry()
    assert record["serialized_inputs_by_track"] == {
        track: list(names) for track, names in MECO_INPUT_NAMES.items()}
    assert record["candidate_schema_by_track"]["C1"] == record["candidate_schema_by_track"]["C2"]
    assert len(provenance.input_contract_digest()) == 64


def test_registry_rejects_missing_model_input(monkeypatch):
    monkeypatch.setattr(provenance, "_INPUTS", {
        key: value for key, value in provenance._INPUTS.items() if key != "raster"})
    with pytest.raises(ValueError, match="serialized track input order"):
        provenance.validate_registry()


def test_digest_rejects_registry_drift(monkeypatch):
    monkeypatch.setattr(provenance, "_INPUTS", {
        key: value for key, value in provenance._INPUTS.items() if key != "raster"})
    with pytest.raises(ValueError, match="serialized track input order"):
        provenance.input_contract_digest()


def test_registry_rejects_runtime_input_order_mismatch(monkeypatch):
    altered = dict(MECO_INPUT_NAMES)
    altered["C1"] = ("candidate_features", "globals_", "hidden")
    monkeypatch.setattr("frc_defense.meco_tracks.MECO_INPUT_NAMES", altered)
    with pytest.raises(ValueError, match="serialized track input order"):
        provenance.validate_registry()


def test_registry_rejects_c1_c2_candidate_schema_mismatch(monkeypatch):
    original = provenance.registry

    def mismatched():
        record = original()
        record["candidate_schema_by_track"]["C2"] = ["different"]
        return record

    monkeypatch.setattr(provenance, "registry", mismatched)
    with pytest.raises(ValueError, match="C1/C2 candidate schema mismatch"):
        provenance.validate_registry()


@pytest.mark.parametrize("track,order", [
    ("A1", ("raster", "globals_", "hidden")),
    ("C1", ("globals_", "candidate_features", "hidden")),
])
def test_serialized_input_names_and_order_are_checked(track, order):
    assert provenance.validate_serialized_inputs(track, dict.fromkeys(order)) == provenance.input_contract_digest()
    with pytest.raises(ValueError, match="registered names/order"):
        provenance.validate_serialized_inputs(track, dict.fromkeys(reversed(order)))


def test_digest_and_inference_request_reject_runtime_registry_drift(monkeypatch):
    from frc_defense.meco_tracks import (
        MECO_INFERENCE_SCHEMA, MECO_INPUT_NAMES, MecoInferenceRequest,
        ordered_candidate_mask_digest,
    )

    altered = dict(MECO_INPUT_NAMES)
    altered["C1"] = ("candidate_features", "globals_", "hidden")
    monkeypatch.setattr("frc_defense.meco_tracks.MECO_INPUT_NAMES", altered)
    with pytest.raises(ValueError, match="serialized track input order"):
        provenance.input_contract_digest()
    with pytest.raises(ValueError, match="track input names/order"):
        MecoInferenceRequest(
            track="C1", snapshot_id="snapshot", checkpoint_sha256="a" * 64,
            protocol_id="protocol", observation_schema=1,
            observation_timestamp=1., capture_monotonic=1.,
            dispatch_monotonic=1., expires_monotonic=1.25,
            input_names=("globals_", "candidate_features", "hidden"),
            input_shapes=((1, 7), (1, 1, 40), (1, 64)),
            candidate_ids=("candidate",), valid_mask=(True,), fallback_index=0,
            request_sequence=1, generation_id="generation",
            candidate_mask_digest=ordered_candidate_mask_digest(("candidate",), (True,)),
            model_id="meco:C1", schema=MECO_INFERENCE_SCHEMA,
            input_contract_sha256="0" * 64)


def test_deployment_export_order_uses_and_enforces_registry(monkeypatch):
    from frc_defense import meco_deployment

    canonical = meco_deployment.input_names("C1")
    tensors = meco_deployment.dummy_inputs("C1")
    assert meco_deployment.validate_export_input_order("C1", canonical, tensors) == canonical
    monkeypatch.setattr(meco_deployment, "input_names",
        lambda track: tuple(reversed(canonical)))
    with pytest.raises(ValueError, match="registered names/order"):
        meco_deployment.validate_export_input_order(
            "C1", tuple(reversed(canonical)), tensors)


def test_deployment_input_names_reject_runtime_order_drift(monkeypatch):
    from frc_defense import meco_deployment

    altered = dict(MECO_INPUT_NAMES)
    altered["C2"] = ("globals_", "raster", "candidate_features", "hidden")
    monkeypatch.setattr("frc_defense.meco_tracks.MECO_INPUT_NAMES", altered)
    with pytest.raises(ValueError, match="serialized track input order"):
        meco_deployment.input_names("C2")
