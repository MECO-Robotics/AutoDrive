
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
