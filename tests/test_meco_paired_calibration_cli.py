"""Pure CLI input contracts for paired campaign planning/merging."""

import json

import pytest

from frc_defense.meco_baseline_freeze import REQUIRED_CONFIGURATION_ARTIFACT_IDS
from scripts import meco_d0_calibration_campaign as cli


def _config_index(tmp_path):
    index = {}
    for artifact_id in sorted(REQUIRED_CONFIGURATION_ARTIFACT_IDS):
        path = tmp_path / f"{artifact_id}.json"
        path.write_text(json.dumps({"id": artifact_id}), encoding="utf-8")
        index[artifact_id] = str(path)
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path, index


def test_paired_cli_loads_exact_logical_config_artifacts_without_path_identity(tmp_path):
    index_path, index = _config_index(tmp_path)

    loaded = cli._load_config_artifact_bytes(index_path)

    assert set(loaded) == REQUIRED_CONFIGURATION_ARTIFACT_IDS
    assert loaded["field_layout"] == (tmp_path / "field_layout.json").read_bytes()
    assert all(isinstance(value, bytes) for value in loaded.values())
    assert all(str(tmp_path).encode() not in value for value in loaded.values())


def test_paired_cli_rejects_missing_or_extra_config_artifact_ids(tmp_path):
    index_path, index = _config_index(tmp_path)
    del index["field_layout"]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="exact required IDs"):
        cli._load_config_artifact_bytes(index_path)

    index["field_layout"] = str(tmp_path / "field_layout.json")
    index["extra"] = str(tmp_path / "field_layout.json")
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="exact required IDs"):
        cli._load_config_artifact_bytes(index_path)


def test_paired_cli_rejects_symlinked_config_artifact_source(tmp_path):
    index_path, index = _config_index(tmp_path)
    target = tmp_path / "field_layout.json"
    link = tmp_path / "field-layout-link.json"
    link.symlink_to(target)
    index["field_layout"] = str(link)
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="missing or symlinked"):
        cli._load_config_artifact_bytes(index_path)
