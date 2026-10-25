import copy
import hashlib
import json
from pathlib import Path

import pytest

from frc_defense import meco_training_manifest as manifest_api
from frc_defense import meco_supervisor as supervisor_api
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


def _legacy_describe(config, **kwargs):
    return manifest_api.describe_campaign(config, legacy_diagnostic=True, **kwargs)


def _legacy_validate(manifest, **kwargs):
    return manifest_api.validate_campaign_manifest(manifest, allow_legacy_v2=True, **kwargs)


def _seal(value):
    value["canonical_sha256"] = _canonical({k: v for k, v in value.items()
        if k != "canonical_sha256"})
    return value


def test_describe_is_no_write_by_default_and_records_false_phase2(campaign):
    tmp_path, config = campaign
    result = _legacy_describe(config, source_root=tmp_path)
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


def test_new_manifest_requires_phase0_and_v2_is_explicitly_legacy_read_only(campaign):
    tmp_path, config = campaign
    with pytest.raises(ValueError, match="require Phase 0 materialization"):
        manifest_api.describe_campaign(config, source_root=tmp_path)
    legacy = manifest_api.describe_campaign(config, source_root=tmp_path,
        legacy_diagnostic=True)
    assert legacy["schema"] == manifest_api.SCHEMA
    assert legacy["legacy_diagnostic_only"] is True
    with pytest.raises(ValueError, match="read-only"):
        manifest_api.validate_campaign_manifest(legacy, source_root=tmp_path)
    assert manifest_api.validate_campaign_manifest(legacy, source_root=tmp_path,
        allow_legacy_v2=True)["valid"]


@pytest.mark.parametrize("enabled", [False, True])
def test_phase4_train_exposure_is_frozen_in_manifest_and_digest(campaign, enabled):
    tmp_path, config = campaign
    config = copy.deepcopy(config)
    config["phase4_train_exposure"] = enabled
    manifest = _legacy_describe(config, source_root=tmp_path)
    assert manifest["campaign"]["phase4_train_exposure"] is enabled
    assert _legacy_validate(
        manifest, source_root=tmp_path)["valid"]


def test_legacy_manifest_defaults_phase4_train_exposure_to_false(campaign):
    tmp_path, config = campaign
    manifest = _legacy_describe(config, source_root=tmp_path)
    del manifest["campaign"]["phase4_train_exposure"]
    _seal(manifest)
    assert _legacy_validate(
        manifest, source_root=tmp_path)["valid"]


@pytest.mark.parametrize("value", [0, 1, "true", None, [], {}])
def test_phase4_train_exposure_requires_strict_boolean(campaign, value):
    tmp_path, config = campaign
    bad = copy.deepcopy(config)
    bad["phase4_train_exposure"] = value
    with pytest.raises(ValueError, match="phase4_train_exposure.*boolean"):
        _legacy_describe(bad, source_root=tmp_path)


def test_phase4_train_exposure_tampering_fails_digest_and_strict_validation(campaign):
    tmp_path, config = campaign
    config = copy.deepcopy(config)
    config["phase4_train_exposure"] = True
    manifest = _legacy_describe(config, source_root=tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered["campaign"]["phase4_train_exposure"] = False
    with pytest.raises(ValueError, match="canonical digest mismatch"):
        _legacy_validate(tampered, source_root=tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered["campaign"]["phase4_train_exposure"] = 1
    _seal(tampered)
    with pytest.raises(ValueError, match="phase4_train_exposure.*boolean"):
        _legacy_validate(tampered, source_root=tmp_path)


def test_optional_evaluation_schedule_is_frozen_and_validated(campaign):
    tmp_path, config = campaign
    protocol_path = Path(config["evaluation_protocol"])
    protocol = json.loads(protocol_path.read_text())
    config = copy.deepcopy(config)
    config["evaluation_schedule"] = [{
        "scenario_id": "pickup_simple", "category": "single", "seeds": [101],
        "split": "diagnostic", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()}]
    manifest = _legacy_describe(config, source_root=tmp_path)
    assert manifest["evaluation_schedule"][0]["index"] == 0
    assert _legacy_validate(
        manifest, source_root=tmp_path)["valid"]
    bad = copy.deepcopy(manifest)
    bad["evaluation_schedule"][0]["seeds"] = [1]
    bad = _seal(bad)
    with pytest.raises(ValueError, match="leaks"):
        _legacy_validate(bad, source_root=tmp_path)


def test_manifest_supervisor_constructor_keeps_roots_absent_until_claim(campaign, monkeypatch):
    tmp_path, config = campaign
    protocol_path = Path(config["evaluation_protocol"])
    protocol = json.loads(protocol_path.read_text())
    config = copy.deepcopy(config)
    config["evaluation_schedule"] = [{
        "scenario_id": "pickup_simple", "category": "single", "seeds": [101],
        "split": "diagnostic", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()}]
    manifest = _legacy_describe(config, source_root=tmp_path)
    identity = {"status": "present", "module": "autodrive_fuel_physics_hip",
        "path": "/native/fuel.so", "sha256": "a" * 64,
        "build_fingerprint": "b" * 64, "device": 1, "inode": 2,
        "size": 3, "mtime_ns": 4}
    monkeypatch.setattr(supervisor_api, "source_identity", lambda: {"source": "fixture"})
    monkeypatch.setattr(supervisor_api, "_validate_runtime_source_binding",
        lambda *_args, **_kwargs: "c" * 64)
    monkeypatch.setattr("frc_defense.meco_device.require_indexed_cuda_device",
        lambda *_args, **_kwargs: None)
    monkeypatch.setattr("frc_defense.fuel_physics_hip.prepare_native_fuel_extension",
        lambda _device: identity)
    monkeypatch.setattr("frc_defense.fuel_physics_hip.native_extension_identity",
        lambda: identity)

    cfg = manifest["tracks"][MECO_TRACKS[0]]["config"]
    with pytest.raises(ValueError, match="legacy v2 manifests are read-only"):
        supervisor_api.MecoSupervisor(output=manifest["campaign"]["root"],
            evaluation_output=manifest["campaign"]["evaluation_output"],
            devices=tuple(manifest["devices"]["training_pool"]),
            scenario=cfg["scenario_id"], category=cfg["category"], seed=cfg["seed"],
            chunk_updates=cfg["updates"], rollout_steps=cfg["rollout_steps"],
            epochs=cfg["epochs"], action_ticks=cfg["action_ticks"],
            learning_rate=cfg["learning_rate"], gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"], clip_coef=cfg["clip_coef"],
            entropy_coef=cfg["entropy_coef"], value_coef=cfg["value_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            route_cost_weight=cfg["route_cost_points_per_second"],
            candidate_k=cfg["candidate_k"], evaluation_categories=("single",),
            evaluation_episodes=1, evaluation_seed=101,
            evaluation_device=manifest["devices"]["evaluation"], cycles=0,
            stop_file=None, p_mode=cfg["p_mode"], p_cost_scale=cfg["p_cost_scale"],
            p_points_per_grid_edge=cfg["p_points_per_grid_edge"],
            prediction_mode=cfg["prediction_mode"],
            qualification_protocol=manifest["qualification_protocol"]["path"],
            evaluation_split="diagnostic", campaign_manifest=manifest)

    campaign_root = Path(manifest["campaign"]["root"])
    evaluation_root = Path(manifest["campaign"]["evaluation_output"])
    assert not campaign_root.exists()
    assert not evaluation_root.exists()


def test_rejects_cpu_unindexed_and_duplicate_devices(campaign):
    tmp_path, config = campaign
    bad = copy.deepcopy(config)
    bad["training_devices"] = ["cpu"]
    with pytest.raises(ValueError, match="indexed CUDA"):
        _legacy_describe(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["training_devices"] = ["cuda:0", "cuda:0"]
    with pytest.raises(ValueError, match="duplicate"):
        _legacy_describe(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["tracks"]["C2"]["device"] = "cuda"
    with pytest.raises(ValueError, match="indexed CUDA"):
        _legacy_describe(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["training_devices"] = ["cuda:0"]
    with pytest.raises(ValueError, match="exactly two"):
        _legacy_describe(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["lease_contract"] = {"policy": "trust config"}
    with pytest.raises(ValueError, match="exclusive ownership"):
        _legacy_describe(bad, source_root=tmp_path)


def test_rejects_duplicate_seed_partition_and_incomplete_track_map(campaign):
    tmp_path, config = campaign
    bad = copy.deepcopy(config)
    bad["scheduler"]["seed_partitions"]["validation"] = [2, 3]
    with pytest.raises(ValueError, match="seed partitions"):
        _legacy_describe(bad, source_root=tmp_path)


def test_rejects_bad_numeric_ranges_and_campaign_path_collisions(campaign):
    tmp_path, config = campaign
    bad = copy.deepcopy(config)
    bad["tracks"]["A1"]["gamma"] = 1.5
    with pytest.raises(ValueError, match="optimizer or discount"):
        _legacy_describe(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["campaign_root"] = str(manifest_api.PINNED_LIVE_ROOTS[0] / "fake-campaign")
    with pytest.raises(ValueError, match="namespaced|pinned live"):
        _legacy_describe(bad, source_root=tmp_path)
    bad = copy.deepcopy(config)
    bad["tracks"]["A2"]["output"] = bad["tracks"]["A1"]["output"]
    with pytest.raises(ValueError, match="unique"):
        _legacy_describe(bad, source_root=tmp_path)
    output = Path(config["campaign_root"]) / "campaign-manifest.json"
    written = _legacy_describe(config, source_root=tmp_path, output=output)
    assert json.loads(output.read_text()) == written
    assert _legacy_validate(written, source_root=tmp_path)["valid"]
    assert _legacy_validate(written, source_root=tmp_path)["launch_ready"] is False
    with pytest.raises(ValueError, match="already exists|existing frozen manifest"):
        _legacy_describe(config, source_root=tmp_path, output=output)
    bad = copy.deepcopy(config)
    del bad["tracks"]["A3"]
    with pytest.raises(ValueError, match="exactly"):
        _legacy_describe(bad, source_root=tmp_path)


def test_rejects_preexisting_campaign_namespace(campaign):
    tmp_path, config = campaign
    root = Path(config["campaign_root"])
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="namespace already exists"):
        _legacy_describe(config, source_root=tmp_path)


def test_validate_rederives_config_seed_device_and_source_bindings(campaign):
    tmp_path, config = campaign
    base = _legacy_describe(config, source_root=tmp_path)
    assert _legacy_validate(base, source_root=tmp_path)["valid"]
    bad = copy.deepcopy(base)
    bad["tracks"]["C1"]["config"]["learning_rate"] = .2
    bad["tracks"]["C1"]["config_sha256"] = _canonical(bad["tracks"]["C1"]["config"])
    with pytest.raises(ValueError, match="shared training"):
        _legacy_validate(_seal(bad), source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["scheduler"]["seed_partitions"]["train"] = [2]
    with pytest.raises(ValueError, match="seed partitions"):
        _legacy_validate(_seal(bad), source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["devices"]["training_pool"] = ["cuda:0", "cuda:0"]
    with pytest.raises(ValueError, match="duplicate"):
        _legacy_validate(_seal(bad), source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["source"]["identity_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="source binding"):
        _legacy_validate(_seal(bad), source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["source"]["identity"] = {"forged": "identity"}
    with pytest.raises(ValueError, match="source binding"):
        _legacy_validate(_seal(bad), source_root=tmp_path)


def test_validate_rejects_protocol_and_false_coverage_mutations(campaign):
    tmp_path, config = campaign
    base = _legacy_describe(config, source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["evaluation_protocol"]["episode_seeds"] = [3]
    bad["evaluation_protocol"]["split"] = "validation"
    bad["evaluation_protocol"]["qualification_eligible"] = True
    with pytest.raises(ValueError, match="qualification"):
        _legacy_validate(_seal(bad), source_root=tmp_path)
    bad = copy.deepcopy(base)
    bad["warm_starts"]["A1"]["phase2_covered"] = True
    bad["readiness"]["phase2_ready"] = False
    with pytest.raises(ValueError, match="warm-start bytes|coverage"):
        _legacy_validate(_seal(bad), source_root=tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(protocol_path.read_text() + " ")
    with pytest.raises(ValueError, match="protocol binding is stale"):
        _legacy_validate(base, source_root=tmp_path)


def test_calibrated_gate_fields_and_validation_readiness_are_separate(campaign):
    tmp_path, config = campaign
    config["evaluation"]["split"] = "validation"
    config["evaluation"]["episode_seeds"] = [3]
    manifest = _legacy_describe(config, source_root=tmp_path)
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
    manifest = _legacy_describe(config, source_root=tmp_path)
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
        _legacy_describe(config, source_root=tmp_path)


def test_fabricated_target_coverage_never_qualifies(campaign, monkeypatch):
    tmp_path, config = campaign
    import torch
    protocol = _protocol(tmp_path / "protocol.json")
    payloads = {}
    paths = {}
    payload_sources = {"source": "fixture", "_native_fuel_extension": {
        "status": "present", "module": "autodrive_fuel_physics_hip",
        "path": "/native/fuel.so", "sha256": "a" * 64,
        "build_fingerprint": "b" * 64, "device": 1, "inode": 2,
        "size": 3, "mtime_ns": 4}}
    for track in MECO_TRACKS:
        path = tmp_path / f"fabricated-{track}.pt"
        path.write_bytes(f"not a torch artifact {track}".encode())
        paths[track] = str(path)
        payloads[str(path.resolve())] = {"track": track, "schema": "fake",
            "sources": payload_sources,
            "source_identity_sha256": manifest_api._source_artifact_hash(payload_sources),
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": manifest_api._file_sha(tmp_path / "protocol.json"),
            "target_coverage": {name: {"coverage_valid": True, "labeled_count": 100,
                "target_schema": "invented", "dataset_sha256": "a" * 64,
                "source_dataset_sha256": "b" * 64, "label_provenance": "invented"}
                for name in ("U", "P", "behavior")}}
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: payloads[str(Path(path).resolve())])
    config["warm_start_artifacts"] = paths
    manifest = _legacy_describe(config, source_root=tmp_path)
    for track in MECO_TRACKS:
        assert manifest["warm_starts"][track]["phase2_covered"] is False
        assert not any(row["valid"] for row in manifest["warm_starts"][track]["targets"].values())
    assert manifest["readiness"]["phase2_ready"] is False


def test_validate_rehashes_warm_start_bytes(campaign, monkeypatch):
    tmp_path, config = campaign
    artifacts = {}
    config["warm_start_artifacts"] = {}
    for track in MECO_TRACKS:
        artifact = tmp_path / f"warm-{track}.pt"
        artifact.write_bytes(f"stable-{track}".encode())
        artifacts[track] = artifact
        config["warm_start_artifacts"][track] = str(artifact)
    monkeypatch.setattr(manifest_api, "_warm_start", lambda path, track, expected_sources,
        protocol_path, protocol_id, protocol_sha: {
            "present": True, "path": str(Path(path).resolve()), "sha256": manifest_api._file_sha(path),
            "binding_valid": True, "targets": {"U": {"valid": False},
                "P": {"valid": False}, "behavior": {"valid": False}},
            "phase2_covered": False, "reason": None})
    base = _legacy_describe(config, source_root=tmp_path)
    artifacts["A1"].write_bytes(b"mutated-artifact")
    with pytest.raises(ValueError, match="warm-start bytes"):
        _legacy_validate(base, source_root=tmp_path)


def test_partial_five_track_warm_start_set_is_rejected(campaign):
    tmp_path, config = campaign
    artifact = tmp_path / "warm-A1.pt"
    artifact.write_bytes(b"warm-start")
    config["warm_start_artifacts"] = {"A1": str(artifact)}
    with pytest.raises(ValueError, match="all five tracks or none"):
        _legacy_describe(config, source_root=tmp_path)


def test_source_projection_excludes_only_native_state_and_accepts_prepare_transition(
        campaign, monkeypatch):
    tmp_path, config = campaign
    unprepared = {"source": "fixture", "_runtime_environment": {"FRC_MODE": "test"},
        "_drivetrain_config_sha256": "d" * 64,
        "_native_fuel_extension": {"status": "unprepared"}}
    monkeypatch.setattr(manifest_api, "source_identity", lambda: unprepared)
    manifest = _legacy_describe(config, source_root=tmp_path)
    projection = {key: value for key, value in unprepared.items()
                  if key != "_native_fuel_extension"}
    assert manifest["source"]["identity"] == projection
    assert manifest["source"]["identity_sha256"] == manifest_api._sha(projection)
    assert _legacy_validate(manifest, source_root=tmp_path)["valid"]

    loaded = {**unprepared, "_native_fuel_extension": {
        "status": "present", "module": "autodrive_fuel_physics_hip",
        "path": "/native/fuel.so", "sha256": "a" * 64,
        "build_fingerprint": "b" * 64, "device": 1, "inode": 2,
        "size": 3, "mtime_ns": 4}}
    monkeypatch.setattr(manifest_api, "source_identity", lambda: loaded)
    assert _legacy_validate(manifest, source_root=tmp_path)["valid"]


@pytest.mark.parametrize("change", ["code", "environment", "drivetrain"])
def test_source_projection_still_rejects_code_environment_and_drivetrain_drift(
        campaign, monkeypatch, change):
    tmp_path, config = campaign
    source = {"module.py": "a" * 64, "_runtime_environment": {"FRC_MODE": "test"},
        "_drivetrain_config_sha256": "d" * 64,
        "_native_fuel_extension": {"status": "unprepared"}}
    monkeypatch.setattr(manifest_api, "source_identity", lambda: source)
    manifest = _legacy_describe(config, source_root=tmp_path)
    drifted = copy.deepcopy(source)
    if change == "code":
        drifted["module.py"] = "c" * 64
    elif change == "environment":
        drifted["_runtime_environment"]["FRC_MODE"] = "changed"
    else:
        drifted["_drivetrain_config_sha256"] = "e" * 64
    monkeypatch.setattr(manifest_api, "source_identity", lambda: drifted)
    with pytest.raises(ValueError, match="source binding"):
        _legacy_validate(manifest, source_root=tmp_path)


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


def test_multicase_warm_start_binding_requires_exact_five_artifact_hashes(tmp_path, monkeypatch):
    descriptor_path = tmp_path / "descriptor.json"
    bundle_path = tmp_path / "bundle.json"
    descriptor = {"descriptor_sha256": "d" * 64, "case_set_sha256": "c" * 64}
    artifacts = {track: tmp_path / f"{track}.pt" for track in MECO_TRACKS}
    for path in artifacts.values():
        path.write_bytes(b"artifact")
    bundle = {"descriptor_sha256": "d" * 64, "case_set_sha256": "c" * 64,
        "phase2_ready": False, "phase2_complete": False,
        "qualification_eligible": False, "artifacts": {track: {"sha256": manifest_api._file_sha(path)}
            for track, path in artifacts.items()}}
    descriptor_path.write_text(json.dumps(descriptor))
    bundle_path.write_text(json.dumps(bundle))
    binding = {"schema": "meco-phase2-multi-case-five-track-v1.training-binding",
        "descriptor_path": str(descriptor_path),
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "descriptor_file_sha256": manifest_api._file_sha(descriptor_path),
        "bundle_manifest_path": str(bundle_path),
        "bundle_manifest_sha256": manifest_api._file_sha(bundle_path),
        "case_set_sha256": "c" * 64,
        "artifact_sha256_by_track": {track: manifest_api._file_sha(path)
            for track, path in artifacts.items()}}
    starts = {track: {"present": True, "sha256": manifest_api._file_sha(path),
        "binding_valid": True} for track, path in artifacts.items()}
    calls = []
    monkeypatch.setattr("frc_defense.meco_phase2_multicase.validate_multicase_warm_start_bundle",
        lambda desc, bundle, **kwargs: calls.append((desc, bundle, kwargs)) or {
            "training_readiness_proven": False})
    assert manifest_api._validate_multicase_warm_start_binding(binding,
        {track: str(path) for track, path in artifacts.items()}, starts) == binding
    assert len(calls) == 1
    assert calls[0][2]["expected_case_set_sha256"] == "c" * 64
    bad = copy.deepcopy(binding)
    bad["artifact_sha256_by_track"].pop("A1")
    with pytest.raises(ValueError, match="exactly five"):
        manifest_api._validate_multicase_warm_start_binding(bad,
            {track: str(path) for track, path in artifacts.items()}, starts)


def test_phase0_multicase_binding_preserves_freeze_approval_crosswalk_and_envelopes(
        tmp_path, monkeypatch):
    from frc_defense import meco_phase2_multicase as multicase
    descriptor_path = tmp_path / "descriptor.json"
    bundle_path = tmp_path / "bundle.json"
    phase0 = {"partition_sha256": "1" * 64, "baseline_freeze_sha256": "2" * 64}
    approval = {"artifact_sha256": "3" * 64, "approval_id": "human-approved"}
    crosswalk = {"crosswalk_sha256": "4" * 64, "tuple_ids": ["tuple-a"]}
    descriptor = {"schema": multicase.PRODUCTION_SCHEMA, "descriptor_sha256": "d" * 64,
        "case_set_sha256": "c" * 64, "phase0_binding": phase0,
        "score_estimand_approval": approval, "tuple_crosswalk": crosswalk}
    artifacts = {track: tmp_path / f"{track}.pt" for track in MECO_TRACKS}
    for path in artifacts.values():
        path.write_bytes(b"artifact")
    envelope_digests = {track: chr(ord("a") + i) * 64 for i, track in enumerate(MECO_TRACKS)}
    bundle = {"descriptor_sha256": "d" * 64, "case_set_sha256": "c" * 64,
        "phase0_binding": phase0, "score_estimand_approval": approval,
        "tuple_crosswalk": crosswalk, "phase2_ready": False, "phase2_complete": False,
        "qualification_eligible": False,
        "artifacts": {track: {"sha256": manifest_api._file_sha(path),
            "phase0_artifact_envelope_sha256": envelope_digests[track]}
            for track, path in artifacts.items()}}
    descriptor_path.write_text(json.dumps(descriptor))
    bundle_path.write_text(json.dumps(bundle))
    binding = {"schema": "meco-phase2-multi-case-five-track-v1.training-binding-phase0-v2",
        "descriptor_path": str(descriptor_path), "descriptor_sha256": "d" * 64,
        "descriptor_file_sha256": manifest_api._file_sha(descriptor_path),
        "bundle_manifest_path": str(bundle_path),
        "bundle_manifest_sha256": manifest_api._file_sha(bundle_path),
        "case_set_sha256": "c" * 64,
        "artifact_sha256_by_track": {track: manifest_api._file_sha(path)
            for track, path in artifacts.items()},
        "phase0_binding": phase0, "score_estimand_approval": approval,
        "tuple_crosswalk": crosswalk,
        "phase0_binding_sha256": multicase._digest(phase0),
        "score_estimand_approval_sha256": approval["artifact_sha256"],
        "tuple_crosswalk_sha256": crosswalk["crosswalk_sha256"],
        "artifact_phase0_envelope_sha256_by_track": envelope_digests}
    starts = {track: {"present": True, "sha256": manifest_api._file_sha(path),
        "binding_valid": True} for track, path in artifacts.items()}
    monkeypatch.setattr("frc_defense.meco_phase2_multicase.validate_multicase_warm_start_bundle",
        lambda *_args, **_kwargs: {"training_readiness_proven": False})
    assert manifest_api._validate_multicase_warm_start_binding(binding,
        {track: str(path) for track, path in artifacts.items()}, starts) == binding
    bad = copy.deepcopy(binding)
    bad["tuple_crosswalk_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Phase 0 digests"):
        manifest_api._validate_multicase_warm_start_binding(bad,
            {track: str(path) for track, path in artifacts.items()}, starts)


def test_current_phase0_campaign_rejects_archival_multicase_binding():
    phase0_binding = {"partition_sha256": "p" * 64}
    legacy_binding = {"schema": "meco-phase2-multi-case-five-track-v1.training-binding",
        "descriptor": {"schema": "meco-phase2-multi-case-five-track-v1"}}
    with pytest.raises(ValueError, match="require a matching production Phase 0"):
        manifest_api._require_current_phase0_multicase_binding(legacy_binding, phase0_binding)
    production_binding = {"schema": "meco-phase2-multi-case-five-track-v1.training-binding-phase0-v2",
        "descriptor": {"schema": "meco-phase2-multi-case-five-track-phase0-v2"},
        "phase0_binding": phase0_binding}
    manifest_api._require_current_phase0_multicase_binding(production_binding, phase0_binding)
    production_binding["phase0_binding"] = {"partition_sha256": "x" * 64}
    with pytest.raises(ValueError, match="require a matching production Phase 0"):
        manifest_api._require_current_phase0_multicase_binding(production_binding, phase0_binding)


def test_phase0_manifest_builder_rejects_v1_multicase_binding(campaign, monkeypatch):
    tmp_path, config = campaign
    phase0_binding = {"partition_sha256": "p" * 64}
    legacy_binding = {"schema": "meco-phase2-multi-case-five-track-v1.training-binding",
        "descriptor": {"schema": "meco-phase2-multi-case-five-track-v1"}}
    config = copy.deepcopy(config)
    config.update({"phase0_materialization": {"fixture": True},
        "multicase_warm_start_binding": legacy_binding})
    monkeypatch.setattr(manifest_api, "validate_effective_phase0_bundle",
        lambda *_args: phase0_binding)
    monkeypatch.setattr(manifest_api, "_validate_multicase_warm_start_binding",
        lambda binding, *_args: binding)
    with pytest.raises(ValueError, match="require a matching production Phase 0"):
        manifest_api.describe_campaign(config, source_root=tmp_path)
