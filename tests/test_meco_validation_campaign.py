import hashlib
import json

import pytest

from frc_defense.meco_qualification import protocol_document
from frc_defense.meco_tracks import MECO_TRACKS
from frc_defense.meco_validation_campaign import compile_validation_campaign
from frc_defense.meco_validation_campaign_runner import _validate_plan
from frc_defense.training_curriculum import SCENARIOS


def _write_protocol(path, protocol):
    path.write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def test_compiles_complete_validation_only_campaign(tmp_path):
    protocol = protocol_document()
    path = tmp_path / "protocol.json"
    _write_protocol(path, protocol)
    plan = compile_validation_campaign(protocol_path=path)
    assert plan == compile_validation_campaign(protocol_path=path)

    assert plan["scenario_ids"] == list(SCENARIOS)
    assert plan["job_count"] == 28
    assert plan["split"] == "validation"
    assert plan["qualification_eligible"] is False
    assert plan["final_test_seed_handling"] == "not_exported"
    assert plan["plan_digest"] == hashlib.sha256(json.dumps(
        {key: value for key, value in plan.items() if key != "plan_digest"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert all(job["seed_manifest"]["split"] == "validation" for job in plan["jobs"])
    assert all(job["seed_manifest"]["protocol_id"] == protocol["protocol_id"]
               for job in plan["jobs"])
    assert all(job["seed_manifest"]["protocol_sha256"] == hashlib.sha256(
        path.read_bytes()).hexdigest() for job in plan["jobs"])
    for job in plan["jobs"]:
        encoded_manifest = (json.dumps(job["seed_manifest"], sort_keys=True,
            indent=2, allow_nan=False) + "\n").encode()
        assert job["seed_manifest_sha256"] == hashlib.sha256(encoded_manifest).hexdigest()
    assert all(job["seed_manifest"]["categories"] == [
        category for category in SCENARIOS[job["scenario_id"]].required_categories]
        for job in plan["jobs"])
    assert all(job["conditions"] for job in plan["jobs"])
    for job in plan["jobs"]:
        for condition in job["conditions"]:
            coverage = condition["expected_coverage"]
            assert coverage["record_count"] == (len(coverage["policies"])
                * len(coverage["categories"]) * len(coverage["seed_list"]))
            assert len(coverage["policy_category_seed_keys_sha256"]) == 64
    assert all("expected_replay_keys" in job["conditions"][0] for job in plan["jobs"])


def test_provisional_protocol_cannot_make_qualification_claim(tmp_path):
    path = tmp_path / "protocol.json"
    _write_protocol(path, protocol_document())
    with pytest.raises(ValueError, match="new qualification claims require"):
        compile_validation_campaign(protocol_path=path, qualification_claim=True)


def test_v4_protocol_qualifies_only_with_exact_reviewed_gate_status(tmp_path, monkeypatch):
    from frc_defense import meco_qualification

    protocol = protocol_document()
    protocol.update({"protocol_version": meco_qualification.CALIBRATED_PROTOCOL_VERSION,
        "gate_calibration_status": "matched_reference_variance_and_reviewed_margin_frozen",
        "safety_gate_status": "metrics_schema_verified",
        "safety_metrics_schema": meco_qualification.SAFETY_METRICS_SCHEMA,
        "safety_metrics_source_identity_sha256": "a" * 64})
    protocol["protocol_id"] = hashlib.sha256(json.dumps(
        {key: value for key, value in protocol.items() if key != "protocol_id"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path = tmp_path / "v4.json"
    _write_protocol(path, protocol)
    validation_calls = []
    monkeypatch.setattr(meco_qualification, "_validate_v2_protocol",
        lambda protocol, scenario: validation_calls.append((protocol["protocol_id"], scenario)))
    plan = compile_validation_campaign(protocol_path=path, qualification_claim=True)
    assert validation_calls
    assert plan["qualification_claim"] is True
    assert plan["qualification_eligible"] is True
    assert plan["protocol"]["protocol_version"] == meco_qualification.CALIBRATED_PROTOCOL_VERSION

    protocol["gate_calibration_status"] = "calibrated_baseline_frozen"
    protocol["protocol_id"] = hashlib.sha256(json.dumps(
        {key: value for key, value in protocol.items() if key != "protocol_id"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _write_protocol(path, protocol)
    with pytest.raises(ValueError, match="reviewed matched-reference margins"):
        compile_validation_campaign(protocol_path=path, qualification_claim=True)


@pytest.mark.parametrize("version,gate_status", [
    ("meco-qualification-v2", "calibrated_baseline_frozen"),
    ("meco-qualification-v3",
     "d0_operational_margin_frozen_reference_variance_unverified"),
])
def test_legacy_v2_v3_are_diagnostic_only_not_qualification_inputs(
        tmp_path, monkeypatch, version, gate_status):
    from frc_defense import meco_qualification

    monkeypatch.setattr(meco_qualification, "_validate_v2_protocol", lambda *_: None)
    protocol = protocol_document()
    protocol.update({"protocol_version": version,
        "gate_calibration_status": gate_status,
        "safety_gate_status": "metrics_schema_verified",
        "safety_metrics_schema": meco_qualification.SAFETY_METRICS_SCHEMA,
        "safety_metrics_source_identity_sha256": "a" * 64})
    protocol["protocol_id"] = hashlib.sha256(json.dumps(
        {key: value for key, value in protocol.items() if key != "protocol_id"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path = tmp_path / f"{version}.json"
    _write_protocol(path, protocol)
    checkpoint_paths, config_hashes = {}, {}
    for track in MECO_TRACKS:
        checkpoint = tmp_path / f"{version}-{track}.pt"
        checkpoint.write_bytes(track.encode())
        checkpoint_paths[track] = checkpoint
        config_hashes[track] = hashlib.sha256((version + track).encode()).hexdigest()
    plan = compile_validation_campaign(protocol_path=path,
        checkpoint_paths=checkpoint_paths, source_identity={"fixture": version},
        training_configuration_sha256=config_hashes)
    assert plan["protocol"]["protocol_version"] == version
    assert plan["qualification_eligible"] is False
    _validate_plan(plan)
    with pytest.raises(ValueError, match="new qualification claims require"):
        compile_validation_campaign(protocol_path=path, qualification_claim=True)


def test_evaluator_protocol_loader_routes_calibrated_versions_to_validator(monkeypatch):
    from frc_defense import meco_qualification, meco_track_training

    calls = []
    monkeypatch.setattr(meco_qualification, "_validate_v2_protocol",
        lambda protocol, scenario: calls.append((protocol["protocol_version"], scenario)))
    for version in (meco_qualification.LEGACY_CALIBRATED_PROTOCOL_VERSION,
                    meco_qualification.D0_PAIRED_REFERENCE_PROTOCOL_VERSION,
                    meco_qualification.CALIBRATED_PROTOCOL_VERSION):
        protocol = protocol_document()
        protocol["protocol_version"] = version
        protocol["protocol_id"] = hashlib.sha256(json.dumps(
            {key: value for key, value in protocol.items() if key != "protocol_id"},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        loaded, protocol_id, _, _ = meco_track_training.load_evaluation_protocol(
            protocol, scenario_id="pickup_simple")
        assert loaded is protocol
        assert protocol_id == protocol["protocol_id"]
    assert calls == [(meco_qualification.LEGACY_CALIBRATED_PROTOCOL_VERSION, "pickup_simple"),
                     (meco_qualification.D0_PAIRED_REFERENCE_PROTOCOL_VERSION, "pickup_simple"),
                     (meco_qualification.CALIBRATED_PROTOCOL_VERSION, "pickup_simple")]


def test_rejects_split_seed_overlap(tmp_path):
    protocol = protocol_document()
    validation_seed = next(row["seeds"][0] for row in protocol["cases"]
                           if row["split"] == "validation")
    train_row = next(row for row in protocol["cases"] if row["split"] == "train")
    train_row["seeds"][0] = validation_seed
    protocol["protocol_id"] = hashlib.sha256(json.dumps(
        {key: value for key, value in protocol.items() if key != "protocol_id"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path = tmp_path / "overlap.json"
    _write_protocol(path, protocol)
    with pytest.raises(ValueError, match="split seed pools overlap"):
        compile_validation_campaign(protocol_path=path)


@pytest.mark.parametrize("mutation", ["omit_stratum", "add_unregistered_stratum"])
def test_rejects_incomplete_or_extra_validation_strata(tmp_path, mutation):
    protocol = protocol_document()
    target = next(row for row in protocol["cases"]
                  if row["scenario_id"] == "pickup_simple"
                  and row["split"] == "validation")
    stratum = {field: target[field] for field in (
        "layout_id", "defender_family", "start_condition", "sensor_profile")}
    if mutation == "omit_stratum":
        protocol["cases"] = [row for row in protocol["cases"]
            if not (row["scenario_id"] == "pickup_simple"
                and row["split"] == "validation"
                and all(row[field] == value for field, value in stratum.items()))]
    else:
        extra = dict(target)
        extra["start_condition"] = "unregistered_validation_stratum"
        protocol["cases"].append(extra)
    protocol["protocol_id"] = hashlib.sha256(json.dumps(
        {key: value for key, value in protocol.items() if key != "protocol_id"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path = tmp_path / f"{mutation}.json"
    _write_protocol(path, protocol)
    with pytest.raises(ValueError, match="complete deterministic scenario/stratum/category/seed schedule"):
        compile_validation_campaign(protocol_path=path)


def test_optional_checkpoint_source_and_config_provenance_is_bound(tmp_path):
    path = tmp_path / "protocol.json"
    _write_protocol(path, protocol_document())
    checkpoint_paths = {}
    config_hashes = {}
    for track in MECO_TRACKS:
        checkpoint = tmp_path / f"{track}.pt"
        checkpoint.write_bytes(track.encode())
        checkpoint_paths[track] = checkpoint
        config_hashes[track] = hashlib.sha256(track.encode()).hexdigest()
    source = {"checkout": "test", "source": ["tracked-file"]}
    plan = compile_validation_campaign(protocol_path=path,
        checkpoint_paths=checkpoint_paths, source_identity=source,
        training_configuration_sha256=config_hashes)
    assert all(job["checkpoint_sha256"] == {
        track: hashlib.sha256((tmp_path / f"{track}.pt").read_bytes()).hexdigest()
        for track in MECO_TRACKS} for job in plan["jobs"])
    assert all(job["source_identity"]["identity"] == source for job in plan["jobs"])
    assert all(job["training_configuration_sha256"] == config_hashes
               for job in plan["jobs"])
