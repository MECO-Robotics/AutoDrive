"""Pure CLI input contracts for paired campaign planning/merging."""

import hashlib
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


def test_paired_merge_publishes_each_control_bundle_before_report(tmp_path, monkeypatch):
    from frc_defense import meco_qualification as q

    campaign = {"campaign_id": "campaign", "phase0_binding": {
        "partition_sha256": "p" * 64}, "calibration_tuples": [{"tuple_id": "t" * 64}],
        "controls": [{
        "control_id": "D1/current", "control_binding_sha256": "c" * 64}]}
    summary = {"control_id": "D1/current", "pair_count": 1}
    bundle, digest = q._build_paired_control_bundle(campaign, summary, ["a" * 64])
    report = {"control_bundles": [{"control_id": "D1/current",
        "bundle_id": bundle["bundle_id"], "artifact_sha256": digest}],
        "_control_bundle_artifacts": [bundle]}
    output = tmp_path / "merged.json"
    published = cli._publish_paired_control_bundles(output, report)

    bundle_path = tmp_path / "merged.control-D1-current.json"
    assert json.loads(bundle_path.read_bytes()) == bundle
    assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == digest
    assert published == [{"control_id": "D1/current", "bundle_id": bundle["bundle_id"],
        "artifact_sha256": digest, "output": str(bundle_path)}]
    assert "_control_bundle_artifacts" not in report

    monkeypatch.setattr(q, "write_paired_calibration_artifact_immutable",
        lambda path, artifact: (_ for _ in ()).throw(ValueError("conflicting bytes")))
    report["_control_bundle_artifacts"] = [bundle]
    with pytest.raises(ValueError, match="conflicting bytes"):
        cli._publish_paired_control_bundles(output, report)
