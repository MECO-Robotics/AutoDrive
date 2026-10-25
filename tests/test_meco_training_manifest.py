import copy
import hashlib
import json
from pathlib import Path

import pytest

from frc_defense import meco_training_manifest as manifest_api
from frc_defense.meco_tracks import MECO_TRACKS


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _protocol(path):
    body = {"protocol_version": "meco-qualification-v1", "master_seed": 17,
        "cases": [
            {"scenario_id": "pickup_simple", "category": "single", "split": "train", "seeds": [1, 2]},
            {"scenario_id": "pickup_simple", "category": "single", "split": "validation", "seeds": [3]},
            {"scenario_id": "pickup_simple", "category": "single", "split": "test", "seeds": [4]},
        ]}
    body["protocol_id"] = _canonical(body)
    path.write_text(json.dumps(body))
    return body


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol = _protocol(protocol_path)
    monkeypatch.setattr(manifest_api, "source_identity", lambda: {"source": "fixture"})
    monkeypatch.setattr(manifest_api, "_git_binding", lambda _: {
        "commit": "a" * 40, "clean": True, "status_sha256": "b" * 64})
    monkeypatch.setattr(manifest_api, "_model_schema", lambda track: {
        "sha256": _canonical([track, "schema"]), "parameters": [track, "schema"]})
    seed_partitions = {"train": [1, 2], "validation": [3], "test": [4]}
    tracks = {}
    campaign_id = "meco-five-fixture"
    campaign_root = tmp_path / campaign_id
    for track in MECO_TRACKS:
        tracks[track] = {"track": track, "scenario_id": "pickup_simple",
            "category": "single", "seed": 17, "device": "cuda:0",
            "output": str(campaign_root / "checkpoints" / track)}
    config = {"seed": 17, "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "evaluation_output": str(campaign_root / "evaluations"),
        "qualification_protocol": str(protocol_path),
        "evaluation_protocol": str(protocol_path), "training_devices": ["cuda:0", "cuda:1"],
        "lease_contract": manifest_api.LEASE_CONTRACT,
        "evaluation_device": "cuda:1", "tracks": tracks,
        "scheduler": {"algorithm": "MecoTaskScheduler", "version": 1,
            "master_seed": 17, "seed_partitions": seed_partitions,
            "task_assignment_contract": ("sequential-cycle deterministic curriculum assignment; all five tracks share "
                "scenario/category/seed; train seed indexed by cycle modulo frozen protocol pool")},
        "evaluation": {"split": "diagnostic", "scenario_id": "pickup_simple",
            "category": "single", "episode_seeds": [100]}}
    return tmp_path, config


def _seal(value):
    value["canonical_sha256"] = _canonical({k: v for k, v in value.items()
        if k != "canonical_sha256"})
    return value


def test_describe_is_no_write_by_default_and_records_false_phase2(campaign):
    tmp_path, config = campaign
    result = manifest_api.describe_campaign(config, source_root=tmp_path)
    assert set(result["tracks"]) == set(MECO_TRACKS)
    assert result["campaign"]["phase4_train_exposure"] is False
    assert result["readiness"]["implementation_ready"] is True
    assert result["readiness"]["hardware_ready"] is True
    assert result["readiness"]["phase2_ready"] is False
    assert result["readiness"]["phase2_complete"] is False
    assert result["readiness"]["validation_ready"] is False
    assert result["readiness"]["qualification_ready"] is False
    assert result["evaluation_protocol"]["qualification_eligible"] is False
    assert result["evaluation_protocol"]["gate_calibration"]["calibrated_and_reviewed"] is False
    assert not (tmp_path / "meco-five-fixture" / "manifest.json").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_phase4_train_exposure_is_frozen_in_manifest_and_digest(campaign, enabled):
    tmp_path, config = campaign
    config = copy.deepcopy(config)
    config["phase4_train_exposure"] = enabled
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    assert manifest["campaign"]["phase4_train_exposure"] is enabled
    assert manifest_api.validate_campaign_manifest(
        manifest, source_root=tmp_path)["valid"]


def test_legacy_manifest_defaults_phase4_train_exposure_to_false(campaign):
    tmp_path, config = campaign
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    del manifest["campaign"]["phase4_train_exposure"]
    _seal(manifest)
    assert manifest_api.validate_campaign_manifest(
        manifest, source_root=tmp_path)["valid"]


@pytest.mark.parametrize("value", [0, 1, "true", None, [], {}])
def test_phase4_train_exposure_requires_strict_boolean(campaign, value):
    tmp_path, config = campaign
    bad = copy.deepcopy(config)
    bad["phase4_train_exposure"] = value
    with pytest.raises(ValueError, match="phase4_train_exposure.*boolean"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)


def test_phase4_train_exposure_tampering_fails_digest_and_strict_validation(campaign):
    tmp_path, config = campaign
    config = copy.deepcopy(config)
    config["phase4_train_exposure"] = True
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered["campaign"]["phase4_train_exposure"] = False
    with pytest.raises(ValueError, match="canonical digest mismatch"):
        manifest_api.validate_campaign_manifest(tampered, source_root=tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered["campaign"]["phase4_train_exposure"] = 1
    _seal(tampered)
    with pytest.raises(ValueError, match="phase4_train_exposure.*boolean"):
        manifest_api.validate_campaign_manifest(tampered, source_root=tmp_path)


def test_optional_evaluation_schedule_is_frozen_and_validated(campaign):
    tmp_path, config = campaign
    protocol_path = Path(config["evaluation_protocol"])
    protocol = json.loads(protocol_path.read_text())
    config = copy.deepcopy(config)
    config["evaluation_schedule"] = [{
        "scenario_id": "pickup_simple", "category": "single", "seeds": [101],
        "split": "diagnostic", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()}]
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    assert manifest["evaluation_schedule"][0]["index"] == 0
    assert manifest_api.validate_campaign_manifest(
        manifest, source_root=tmp_path)["valid"]
    bad = copy.deepcopy(manifest)
    bad["evaluation_schedule"][0]["seeds"] = [1]
    bad = _seal(bad)
    with pytest.raises(ValueError, match="leaks"):
        manifest_api.validate_campaign_manifest(bad, source_root=tmp_path)


def test_rejects_cpu_unindexed_and_duplicate_devices(campaign):
    tmp_path, config = campaign
    bad = copy.deepcopy(config)
    bad["training_devices"] = ["cpu"]
    with pytest.raises(ValueError, match="indexed CUDA"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["training_devices"] = ["cuda:0", "cuda:0"]
    with pytest.raises(ValueError, match="duplicate"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["tracks"]["C2"]["device"] = "cuda"
    with pytest.raises(ValueError, match="indexed CUDA"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["training_devices"] = ["cuda:0"]
    with pytest.raises(ValueError, match="exactly two"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["lease_contract"] = {"policy": "trust config"}
    with pytest.raises(ValueError, match="exclusive ownership"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)


def test_rejects_duplicate_seed_partition_and_incomplete_track_map(campaign):
    tmp_path, config = campaign
    bad = copy.deepcopy(config)
    bad["scheduler"]["seed_partitions"]["validation"] = [2, 3]
    with pytest.raises(ValueError, match="seed partitions"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)


def test_rejects_bad_numeric_ranges_and_campaign_path_collisions(campaign):
    tmp_path, config = campaign
    bad = copy.deepcopy(config)
    bad["tracks"]["A1"]["gamma"] = 1.5
    with pytest.raises(ValueError, match="optimizer or discount"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["campaign_root"] = str(manifest_api.PINNED_LIVE_ROOTS[0] / "fake-campaign")
    with pytest.raises(ValueError, match="namespaced|pinned live"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["tracks"]["A2"]["output"] = bad["tracks"]["A1"]["output"]
    with pytest.raises(ValueError, match="unique"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)
    output = Path(config["campaign_root"]) / "campaign-manifest.json"
    written = manifest_api.describe_campaign(config, source_root=tmp_path, output=output)
    assert json.loads(output.read_text()) == written
    assert manifest_api.validate_campaign_manifest(written, source_root=tmp_path)["valid"]
    assert manifest_api.validate_campaign_manifest(written, source_root=tmp_path)["launch_ready"] is False
    with pytest.raises(ValueError, match="already exists|existing frozen manifest"):
        manifest_api.describe_campaign(config, source_root=tmp_path, output=output)
    bad = copy.deepcopy(config)
    del bad["tracks"]["A3"]
    with pytest.raises(ValueError, match="exactly"):
        manifest_api.describe_campaign(bad, source_root=tmp_path)


def test_rejects_preexisting_campaign_namespace(campaign):
    tmp_path, config = campaign
    root = Path(config["campaign_root"])
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="namespace already exists"):
        manifest_api.describe_campaign(config, source_root=tmp_path)


def test_validate_rederives_config_seed_device_and_source_bindings(campaign):
    tmp_path, config = campaign
    base = manifest_api.describe_campaign(config, source_root=tmp_path)
    assert manifest_api.validate_campaign_manifest(base, source_root=tmp_path)["valid"]
    bad = copy.deepcopy(base)
    bad["tracks"]["C1"]["config"]["learning_rate"] = .2
    bad["tracks"]["C1"]["config_sha256"] = _canonical(bad["tracks"]["C1"]["config"])
    with pytest.raises(ValueError, match="shared training"):
        manifest_api.validate_campaign_manifest(_seal(bad), source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["scheduler"]["seed_partitions"]["train"] = [2]
    with pytest.raises(ValueError, match="seed partitions"):
        manifest_api.validate_campaign_manifest(_seal(bad), source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["devices"]["training_pool"] = ["cuda:0", "cuda:0"]
    with pytest.raises(ValueError, match="duplicate"):
        manifest_api.validate_campaign_manifest(_seal(bad), source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["source"]["identity_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="source binding"):
        manifest_api.validate_campaign_manifest(_seal(bad), source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["source"]["identity"] = {"forged": "identity"}
    with pytest.raises(ValueError, match="source binding"):
        manifest_api.validate_campaign_manifest(_seal(bad), source_root=tmp_path)


def test_validate_rejects_protocol_and_false_coverage_mutations(campaign):
    tmp_path, config = campaign
    base = manifest_api.describe_campaign(config, source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["evaluation_protocol"]["episode_seeds"] = [3]
    bad["evaluation_protocol"]["split"] = "validation"
    bad["evaluation_protocol"]["qualification_eligible"] = True
    with pytest.raises(ValueError, match="qualification"):
        manifest_api.validate_campaign_manifest(_seal(bad), source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["warm_starts"]["A1"]["phase2_covered"] = True
    bad["readiness"]["phase2_ready"] = False
    with pytest.raises(ValueError, match="warm-start bytes|coverage"):
        manifest_api.validate_campaign_manifest(_seal(bad), source_root=tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(protocol_path.read_text() + " ")
    with pytest.raises(ValueError, match="protocol binding is stale"):
        manifest_api.validate_campaign_manifest(base, source_root=tmp_path)


def test_calibrated_gate_fields_and_validation_readiness_are_separate(campaign):
    tmp_path, config = campaign
    config["evaluation"]["split"] = "validation"
    config["evaluation"]["episode_seeds"] = [3]
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    assert manifest["readiness"]["validation_ready"] is True
    assert manifest["readiness"]["qualification_ready"] is False
    assert manifest["evaluation_protocol"]["qualification_eligible"] is False


def test_forged_v1_calibration_sidecar_never_enables_qualification(campaign):
    tmp_path, config = campaign
    protocol_path = tmp_path / "protocol.json"
    protocol = _protocol(protocol_path)
    evidence_path = tmp_path / "baseline-variance-review.txt"
    evidence_path.write_text("Independent baseline variance review approved; gates frozen.\n")
    protocol["gate_calibration_status"] = "calibrated_baseline_frozen"
    protocol["gate_calibration_evidence"] = {
        "baseline_variance_review": "approved", "frozen": True,
        "evidence_id": "review-2026-10-24",
        "evidence_path": str(evidence_path),
        "evidence_sha256": manifest_api._file_sha(evidence_path)}
    body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    protocol["protocol_id"] = _canonical(body)
    protocol_path.write_text(json.dumps(protocol))
    config["evaluation"]["split"] = "test"
    config["evaluation"]["episode_seeds"] = [4]
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    assert manifest["evaluation_protocol"]["qualification_eligible"] is False
    assert manifest["evaluation_protocol"]["gate_calibration"]["trusted_calibration_report_binding"] is False
    assert manifest["readiness"]["qualification_ready"] is False
    assert manifest["readiness"]["launch_ready"] is False


def test_protocol_parser_accepts_v2_through_official_validator(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol-v2.json"
    protocol = _protocol(protocol_path)
    protocol["protocol_version"] = "meco-qualification-v2"
    protocol["promotion_target_scenario"] = "pickup_simple"
    body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    protocol["protocol_id"] = _canonical(body)
    protocol_path.write_text(json.dumps(protocol))
    from frc_defense import meco_qualification
    monkeypatch.setattr(meco_qualification, "_validate_v2_protocol", lambda *_: None)
    _, parsed, _, _ = manifest_api._protocol(protocol_path)
    assert parsed["protocol_version"] == "meco-qualification-v2"


def test_protocol_parser_accepts_v4_but_training_manifest_does_not_attest_qualification(
        tmp_path, monkeypatch):
    from frc_defense import meco_qualification

    protocol_path = tmp_path / "protocol-v4.json"
    protocol = _protocol(protocol_path)
    protocol.update({"protocol_version": meco_qualification.CALIBRATED_PROTOCOL_VERSION,
        "promotion_target_scenario": "pickup_simple",
        "gate_calibration_status": "matched_reference_variance_and_reviewed_margin_frozen",
        "safety_gate_status": "metrics_schema_verified"})
    protocol["protocol_id"] = _canonical({key: value for key, value in protocol.items()
        if key != "protocol_id"})
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(meco_qualification, "_validate_v2_protocol", lambda *_: None)
    _, parsed, _, _ = manifest_api._protocol(protocol_path)
    assert parsed["protocol_version"] == meco_qualification.CALIBRATED_PROTOCOL_VERSION
    calibration = manifest_api._gate_calibration(parsed)
    assert calibration["calibrated_and_reviewed"] is False
    assert calibration["trusted_calibration_report_binding"] is False


def test_hardware_inventory_requires_matching_wx9100_evidence(campaign, monkeypatch):
    tmp_path, config = campaign
    original = json.loads(manifest_api.HOST_GPU_INVENTORY.read_text())
    bad = copy.deepcopy(original)
    bad["devices"][0]["pci_address"] = "0000:ff:00.0"
    path = tmp_path / "host-gpu-inventory.json"
    path.write_text(json.dumps(bad))
    monkeypatch.setattr(manifest_api, "HOST_GPU_INVENTORY", path)
    monkeypatch.setattr(manifest_api, "HOST_GPU_INVENTORY_SHA256", manifest_api._file_sha(path))
    with pytest.raises(ValueError, match="index/BDF/model"):
        manifest_api.describe_campaign(config, source_root=tmp_path)


def test_fabricated_target_coverage_never_qualifies(campaign, monkeypatch):
    tmp_path, config = campaign
    import torch
    path = tmp_path / "fabricated.pt"
    path.write_bytes(b"not a torch artifact")
    payload = {"track": "A1", "schema": "fake",
        "source_identity_sha256": _canonical({"source": "fixture"}),
        "protocol_id": _protocol(tmp_path / "protocol.json")["protocol_id"],
        "protocol_sha256": manifest_api._file_sha(tmp_path / "protocol.json"),
        "target_coverage": {name: {"coverage_valid": True, "labeled_count": 100,
            "target_schema": "invented", "dataset_sha256": "a" * 64,
            "source_dataset_sha256": "b" * 64, "label_provenance": "invented"}
            for name in ("U", "P", "behavior")}}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    from frc_defense import meco_supervised
    monkeypatch.setattr(meco_supervised, "validate_warm_start_artifact",
        lambda *args, **kwargs: payload)
    config["warm_start_artifacts"] = {"A1": str(path)}
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    assert manifest["warm_starts"]["A1"]["phase2_covered"] is False
    assert not any(row["valid"] for row in manifest["warm_starts"]["A1"]["targets"].values())
    assert manifest["readiness"]["phase2_ready"] is False


def test_validate_rehashes_warm_start_bytes(campaign, monkeypatch):
    tmp_path, config = campaign
    artifact = tmp_path / "warm.pt"
    artifact.write_bytes(b"stable-artifact")
    config["warm_start_artifacts"] = {"A1": str(artifact)}
    monkeypatch.setattr(manifest_api, "_warm_start", lambda path, track, expected_sources,
        protocol_path, protocol_id, protocol_sha: {
            "present": True, "path": str(artifact.resolve()), "sha256": manifest_api._file_sha(artifact),
            "binding_valid": True, "targets": {"U": {"valid": False},
                "P": {"valid": False}, "behavior": {"valid": False}},
            "phase2_covered": False, "reason": None})
    base = manifest_api.describe_campaign(config, source_root=tmp_path)
    artifact.write_bytes(b"mutated-artifact")
    with pytest.raises(ValueError, match="warm-start bytes"):
        manifest_api.validate_campaign_manifest(base, source_root=tmp_path)


def test_source_projection_excludes_only_native_state_and_accepts_prepare_transition(
        campaign, monkeypatch):
    tmp_path, config = campaign
    unprepared = {"source": "fixture", "_runtime_environment": {"FRC_MODE": "test"},
        "_drivetrain_config_sha256": "d" * 64,
        "_native_fuel_extension": {"status": "unprepared"}}
    monkeypatch.setattr(manifest_api, "source_identity", lambda: unprepared)
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    projection = {key: value for key, value in unprepared.items()
                  if key != "_native_fuel_extension"}
    assert manifest["source"]["identity"] == projection
    assert manifest["source"]["identity_sha256"] == manifest_api._sha(projection)
    assert manifest_api.validate_campaign_manifest(manifest, source_root=tmp_path)["valid"]

    loaded = {**unprepared, "_native_fuel_extension": {
        "status": "present", "module": "autodrive_fuel_physics_hip",
        "path": "/native/fuel.so", "sha256": "a" * 64,
        "build_fingerprint": "b" * 64, "device": 1, "inode": 2,
        "size": 3, "mtime_ns": 4}}
    monkeypatch.setattr(manifest_api, "source_identity", lambda: loaded)
    assert manifest_api.validate_campaign_manifest(manifest, source_root=tmp_path)["valid"]


@pytest.mark.parametrize("change", ["code", "environment", "drivetrain"])
def test_source_projection_still_rejects_code_environment_and_drivetrain_drift(
        campaign, monkeypatch, change):
    tmp_path, config = campaign
    source = {"module.py": "a" * 64, "_runtime_environment": {"FRC_MODE": "test"},
        "_drivetrain_config_sha256": "d" * 64,
        "_native_fuel_extension": {"status": "unprepared"}}
    monkeypatch.setattr(manifest_api, "source_identity", lambda: source)
    manifest = manifest_api.describe_campaign(config, source_root=tmp_path)
    drifted = copy.deepcopy(source)
    if change == "code":
        drifted["module.py"] = "c" * 64
    elif change == "environment":
        drifted["_runtime_environment"]["FRC_MODE"] = "changed"
    else:
        drifted["_drivetrain_config_sha256"] = "e" * 64
    monkeypatch.setattr(manifest_api, "source_identity", lambda: drifted)
    with pytest.raises(ValueError, match="source binding"):
        manifest_api.validate_campaign_manifest(manifest, source_root=tmp_path)


def test_warm_start_metadata_binds_source_projection_and_defers_loaded_binary_parity(
        tmp_path, monkeypatch):
    native = {"status": "present", "module": "autodrive_fuel_physics_hip",
        "path": "/native/fuel.so", "sha256": "a" * 64,
        "build_fingerprint": "b" * 64, "device": 1, "inode": 2,
        "size": 3, "mtime_ns": 4}
    sources = {"module.py": "c" * 64, "_runtime_environment": {"FRC_MODE": "test"},
        "_native_fuel_extension": native}
    checkpoint = tmp_path / "artifact.pt"
    checkpoint.write_bytes(b"artifact")
    payload = {"sources": sources,
        "source_identity_sha256": manifest_api._source_artifact_hash(sources),
        "track": "A1", "protocol_id": "protocol",
        "protocol_sha256": "p" * 64}
    import torch
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    projection = {key: value for key, value in sources.items()
                  if key != "_native_fuel_extension"}
    result = manifest_api._warm_start(checkpoint, "A1", projection,
        tmp_path / "protocol.json", "protocol", "p" * 64)
    assert result["binding_valid"] is True
    assert result["native_extension_identity"] == native
    assert result["native_extension_identity_sha256"] == manifest_api._sha(native)
    assert result["native_runtime_attested"] is False
    assert result["reason"] == "loaded HIP parity is required at runtime"
