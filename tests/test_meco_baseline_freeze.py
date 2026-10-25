"""Offline contract tests for source-bound Phase 0 freeze and split contracts."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from frc_defense.meco_baseline_freeze import (
    AUTHORITATIVE_FIELD_SOURCES,
    PARTITION_NAMES,
    PARTITION_WORKFLOWS,
    REQUIRED_CONFIGURATION_ARTIFACT_IDS,
    canonical_sha256,
    compile_baseline_freeze,
    compile_partition_contract,
    compile_effective_capture_receipt,
    compile_effective_baseline_freeze,
    compile_tuple_partition_materialization,
    consume_final_test_authorization,
    validate_baseline_freeze,
    validate_effective_capture_receipt,
    validate_effective_baseline_freeze,
    validate_partition_contract,
    validate_tuple_partition_materialization,
)


SOURCE_PLAN = "docs/source-plan.md"


def test_authoritative_sources_follow_meco_curriculum_evaluation_path():
    assert AUTHORITATIVE_FIELD_SOURCES == {
        "robot_dynamics": (
            "frc_defense/curriculum_scenarios.py",
            "frc_defense/tensor_3v3.py",
            "frc_defense/tensor_sim.py",
            "frc_defense/tensor_physics.py",
            "frc_defense/fuel_physics_runtime.py",
        ),
        "field_geometry": (
            "frc_defense/curriculum_scenarios.py",
            "frc_defense/tensor_3v3.py",
            "frc_defense/field.py",
        ),
        "legal_connectivity": (
            "frc_defense/curriculum_scenarios.py",
            "frc_defense/tensor_3v3.py",
            "frc_defense/field.py",
            "frc_defense/tensor_adstar.py",
            "frc_defense/_tensor_3v3_navigation.py",
        ),
        "opponent_controllers": (
            "frc_defense/curriculum_scenarios.py",
            "frc_defense/training_curriculum.py",
            "frc_defense/meco_track_training.py",
            "frc_defense/meco_phase4_opponents.py",
            "frc_defense/tensor_3v3.py",
            "frc_defense/_tensor_3v3_actions.py",
        ),
        "action_update_cadence": (
            "frc_defense/curriculum_scenarios.py",
            "frc_defense/tensor_3v3.py",
            "frc_defense/_tensor_3v3_actions.py",
            "frc_defense/tensor_physics.py",
        ),
        "sensor_noise_profiles": (
            "frc_defense/curriculum_scenarios.py",
            "frc_defense/tensor_3v3.py",
            "frc_defense/_tensor_3v3_observation.py",
            "frc_defense/tensor_perception.py",
        ),
        "scoring_observation_semantics": (
            "frc_defense/curriculum_scenarios.py",
            "frc_defense/tensor_3v3.py",
            "frc_defense/_tensor_3v3_gamepieces.py",
            "frc_defense/meco_track_training.py",
            "frc_defense/meco_metric_ledger.py",
        ),
        "safety_event_semantics": (
            "frc_defense/tensor_physics.py",
            "frc_defense/tensor_3v3.py",
            "frc_defense/_tensor_3v3_actions.py",
            "frc_defense/curriculum_scenarios.py",
            "frc_defense/meco_track_training.py",
            "frc_defense/meco_metric_ledger.py",
        ),
    }


def _init_repo(root: Path) -> None:
    paths = {SOURCE_PLAN}
    for required in AUTHORITATIVE_FIELD_SOURCES.values():
        paths.update(required)
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic source fixture: {relative}\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Fixture"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "fixture@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "synthetic contract fixture"], check=True)


def _runtime_identity() -> dict[str, str]:
    digest = hashlib.sha256(b"synthetic fixture only").hexdigest()
    return {
        "simulator_id": "synthetic-fixture-simulator",
        "simulator_version": "synthetic-fixture-version",
        "simulator_build_sha256": digest,
        "python_version": "synthetic-fixture-python",
        "torch_version": "synthetic-fixture-torch",
        "backend": "synthetic-fixture-backend",
        "backend_version": "synthetic-fixture-backend-version",
        "device_id": "synthetic-fixture-device",
        "dependency_lock_sha256": digest,
    }


def _configuration_artifacts() -> dict[str, bytes]:
    return {
        artifact_id: f'{{"artifact":"{artifact_id}","fixture":true}}'.encode()
        for artifact_id in sorted(REQUIRED_CONFIGURATION_ARTIFACT_IDS)
    }


def _field_records(root: Path) -> dict[str, dict]:
    records = {}
    for field, paths in AUTHORITATIVE_FIELD_SOURCES.items():
        if field == "scoring_observation_semantics":
            value = {
                "simulator_metric": "fuel_score_count",
                "metric_semantics": "scored_fuel_piece_count",
                "official_match_points_available": False,
            }
        else:
            value = {"fixture_only": True, "field": field}
        records[field] = {
            "value": value,
            "selector": f"synthetic_fixture.{field}",
            "sources": [
                {
                    "path": path,
                    "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
                }
                for path in paths
            ],
        }
    return records


def _valid_partitions() -> dict[str, dict[str, list]]:
    return {
        "train": {
            "seeds": [1, 2], "defender_families": ["train_policy"],
            "layouts": ["train_layout"], "start_conditions": ["train_start"],
        },
        "calibration": {
            "seeds": [3, 4], "defender_families": ["cal_policy"],
            "layouts": ["cal_layout"], "start_conditions": ["cal_start"],
        },
        "validation": {
            "seeds": [5, 6], "defender_families": ["validation_policy"],
            "layouts": ["validation_layout"], "start_conditions": ["validation_start"],
        },
        "final_test": {
            "seeds": [7, 8], "defender_families": ["heldout_policy"],
            "layouts": ["heldout_layout"], "start_conditions": ["heldout_start"],
        },
    }


def _valid_uses() -> dict[str, list[str]]:
    return {name: sorted(PARTITION_WORKFLOWS[name]) for name in PARTITION_NAMES}


def _factory_arguments():
    protocol = {"protocol_version": "fixture-v1", "cases": []}
    protocol_bytes = __import__("json").dumps(protocol, sort_keys=True,
        separators=(",", ":")).encode()
    return {"scenario_id": "pickup_simple", "category": "single", "num_envs": 1,
        "device": "cuda:0", "seed": 13, "fuel_physics": True,
        "production_d0": False, "layout_id": "2026_rebuilt",
        "defender_family": "category_scripted", "start_condition": "scenario_defined",
        "sensor_profile": "standard",
        "evaluation_protocol_object": protocol,
        "evaluation_protocol_content_base64": __import__("base64").b64encode(
            protocol_bytes).decode(),
        "evaluation_protocol_content_sha256": hashlib.sha256(protocol_bytes).hexdigest()}


def _effective_runtime_identity():
    digest = hashlib.sha256(b"fixture-runtime").hexdigest()
    return {"simulator_id": "fixture-sim", "simulator_version": "1",
        "simulator_build_sha256": digest, "python_version": "3.fixture",
        "torch_version": "fixture", "backend": "ROCm", "backend_version": "fixture",
        "device_id": "GPU-0123456789abcdef", "device_index": 0,
        "device_model": "fixture WX9100", "dependency_lock_sha256": digest}


@pytest.fixture
def stub_effective_device_probe(monkeypatch):
    from frc_defense import curriculum_scenarios
    monkeypatch.setattr(curriculum_scenarios, "_discover_effective_runtime_identity",
        lambda device, _root: {**_effective_runtime_identity(),
            "device_index": int(str(device).rsplit(":", 1)[1])})


def _registry_and_assignment():
    tuples = []
    assignments = {}
    for partition_i, partition in enumerate(PARTITION_NAMES):
        for j in range(2):
            item = {"scenario_id": "pickup_simple", "case_id": f"case-{partition_i}-{j}",
                "category": "single", "defender_family": f"family-{partition_i}-{j}",
                "layout_id": f"layout-{partition_i}-{j}",
                "start_condition": f"start-{partition_i}-{j}",
                "sensor_profile": "standard", "seed": partition_i * 10 + j,
                "repetition": 0, "evaluation_phase": partition}
            tuple_id = canonical_sha256(item)
            tuples.append({**item, "eligible": True})
            assignments[tuple_id] = partition
    registry = {"schema": "meco-evaluation-tuple-registry-v1", "tuples": tuples}
    eligible = sorted(assignments)
    budget = {"episodes_per_tuple": 1, "repetitions": 2}
    allocation = {"authored_by": "human-review-fixture", "authored_at_utc": "2026-10-25T00:00:00Z",
        "source_sha256": hashlib.sha256(b"fixed table").hexdigest(), "outcomes_accessed": False}
    tracks = {track: {"tuple_ids": eligible, "evaluation_budget": budget}
        for track in ("A1", "A2", "A3", "C1", "C2")}
    table = {"schema": "meco-partition-assignment-table-v1", "assignments": assignments,
        "model_tracks": tracks, "evaluation_budget": budget,
        "allocation_provenance": allocation}
    return registry, table


def test_baseline_freeze_is_bound_to_clean_source_runtime_and_field_sources(tmp_path):
    _init_repo(tmp_path)
    freeze = compile_baseline_freeze(
        tmp_path,
        source_plan_path=SOURCE_PLAN,
        runtime_identity=_runtime_identity(),
        field_records=_field_records(tmp_path),
        configuration_artifacts=_configuration_artifacts(),
    )

    assert freeze["qualification_eligible"] is False
    assert freeze == compile_baseline_freeze(
        tmp_path,
        source_plan_path=SOURCE_PLAN,
        runtime_identity=_runtime_identity(),
        field_records=_field_records(tmp_path),
        configuration_artifacts=_configuration_artifacts(),
    )
    assert validate_baseline_freeze(
        freeze, tmp_path, configuration_artifacts=_configuration_artifacts()) == freeze
    assert freeze["configuration_artifacts"] == [
        {"artifact_id": artifact_id,
         "content_sha256": hashlib.sha256(config_bytes).hexdigest()}
        for artifact_id, config_bytes in sorted(_configuration_artifacts().items())]
    assert "path" not in freeze["configuration_artifacts"][0]

    changed = dict(freeze)
    changed["runtime"] = dict(freeze["runtime"], device_id="different-device")
    with pytest.raises(ValueError, match="runtime identity digest"):
        validate_baseline_freeze(
            changed, tmp_path, configuration_artifacts=_configuration_artifacts())


def test_baseline_freeze_rejects_missing_runtime_field_and_dirty_source(tmp_path):
    _init_repo(tmp_path)
    runtime = _runtime_identity()
    runtime.pop("device_id")
    with pytest.raises(ValueError, match="runtime identity"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=runtime,
            field_records=_field_records(tmp_path),
            configuration_artifacts=_configuration_artifacts(),
        )

    (tmp_path / "frc_defense/curriculum_scenarios.py").write_text("dirty\n")
    with pytest.raises(ValueError, match="clean source checkout"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=_runtime_identity(),
            field_records=_field_records(tmp_path),
            configuration_artifacts=_configuration_artifacts(),
        )


def test_baseline_freeze_rejects_missing_source_field_or_changed_source_hash(tmp_path):
    _init_repo(tmp_path)
    records = _field_records(tmp_path)
    del records["legal_connectivity"]
    with pytest.raises(ValueError, match="baseline fields must be complete"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=_runtime_identity(),
            field_records=records,
            configuration_artifacts=_configuration_artifacts(),
        )

    records = _field_records(tmp_path)
    records["field_geometry"]["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source digest mismatch"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=_runtime_identity(),
            field_records=records,
            configuration_artifacts=_configuration_artifacts(),
        )


def test_scoring_field_cannot_claim_official_match_points_or_convert_fuel_counts(tmp_path):
    _init_repo(tmp_path)
    records = _field_records(tmp_path)
    records["scoring_observation_semantics"]["value"] = {
        "simulator_metric": "match_points",
        "metric_semantics": "official_match_points",
        "official_match_points_available": True,
    }
    with pytest.raises(ValueError, match="FUEL-piece-count-only"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=_runtime_identity(),
            field_records=records,
            configuration_artifacts=_configuration_artifacts(),
        )

    records = _field_records(tmp_path)
    records["scoring_observation_semantics"]["value"]["official_match_points_available"] = 0
    with pytest.raises(ValueError, match="FUEL-piece-count-only"):
        compile_baseline_freeze(
            tmp_path,
            source_plan_path=SOURCE_PLAN,
            runtime_identity=_runtime_identity(),
            field_records=records,
            configuration_artifacts=_configuration_artifacts(),
        )


def test_baseline_freeze_rejects_missing_or_drifted_configuration_artifacts(tmp_path):
    _init_repo(tmp_path)
    arguments = {
        "source_plan_path": SOURCE_PLAN,
        "runtime_identity": _runtime_identity(),
        "field_records": _field_records(tmp_path),
    }
    for missing_id in REQUIRED_CONFIGURATION_ARTIFACT_IDS:
        incomplete = _configuration_artifacts()
        del incomplete[missing_id]
        with pytest.raises(ValueError, match="configuration artifact IDs"):
            compile_baseline_freeze(
                tmp_path, **arguments, configuration_artifacts=incomplete)

    config = _configuration_artifacts()
    freeze = compile_baseline_freeze(
        tmp_path, **arguments, configuration_artifacts=config)
    with pytest.raises(ValueError, match="configuration artifact identity drifted"):
        validate_baseline_freeze(
            freeze, tmp_path,
            configuration_artifacts={**config, "field_layout": b'{"layout":"shifted"}'})

    invalid_ids = dict(config)
    invalid_ids["/tmp/config.json"] = invalid_ids.pop("field_layout")
    with pytest.raises(ValueError, match="configuration artifact IDs"):
        compile_baseline_freeze(
            tmp_path, **arguments,
            configuration_artifacts=invalid_ids)


def test_partition_contract_is_deterministic_disjoint_and_final_test_isolated():
    baseline_digest = hashlib.sha256(b"synthetic freeze").hexdigest()
    contract = compile_partition_contract(
        baseline_freeze_sha256=baseline_digest,
        partitions=_valid_partitions(),
        allowed_uses=_valid_uses(),
    )
    assert contract["qualification_eligible"] is False
    assert contract == compile_partition_contract(
        baseline_freeze_sha256=baseline_digest,
        partitions=_valid_partitions(),
        allowed_uses=_valid_uses(),
    )
    assert validate_partition_contract(
        contract, baseline_freeze_sha256=baseline_digest) == contract
    with pytest.raises(ValueError, match="does not bind"):
        validate_partition_contract(
            contract,
            baseline_freeze_sha256=hashlib.sha256(b"different freeze").hexdigest(),
        )

    changed = dict(contract)
    changed["partition_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest or canonical form"):
        validate_partition_contract(changed, baseline_freeze_sha256=baseline_digest)


@pytest.mark.parametrize("axis", ["seeds", "defender_families", "layouts", "start_conditions"])
def test_partition_contract_rejects_cross_partition_leakage(axis):
    partitions = _valid_partitions()
    partitions["final_test"][axis] = list(partitions["train"][axis])
    with pytest.raises(ValueError, match="leaks across"):
        compile_partition_contract(
            baseline_freeze_sha256=hashlib.sha256(b"baseline").hexdigest(),
            partitions=partitions,
            allowed_uses=_valid_uses(),
        )


def test_partition_contract_rejects_final_test_use_for_training_or_margin_review():
    uses = _valid_uses()
    uses["final_test"] = ["final_evaluation", "training"]
    with pytest.raises(ValueError, match="preregistered policy"):
        compile_partition_contract(
            baseline_freeze_sha256=hashlib.sha256(b"baseline").hexdigest(),
            partitions=_valid_partitions(),
            allowed_uses=uses,
        )


def test_effective_receipt_and_freeze_bind_factory_snapshot_bytes_and_source(tmp_path,
        stub_effective_device_probe):
    _init_repo(tmp_path)
    receipt = compile_effective_capture_receipt(tmp_path,
        source_plan_path=SOURCE_PLAN, factory_arguments=_factory_arguments(),
        runtime_identity=_effective_runtime_identity(),
        captured_at_utc="2026-10-25T00:00:00Z")
    assert validate_effective_capture_receipt(receipt, tmp_path) == receipt
    freeze = compile_effective_baseline_freeze(receipt, tmp_path)
    assert freeze["qualification_eligible"] is False
    assert validate_effective_baseline_freeze(freeze, receipt, tmp_path) == freeze

    changed_receipt = __import__("copy").deepcopy(receipt)
    changed_receipt["fields"]["robot_dynamics"]["value"]["forged"] = True
    with pytest.raises(ValueError, match="digest"):
        validate_effective_capture_receipt(changed_receipt, tmp_path)


@pytest.mark.parametrize("mutation", ["missing_field", "default_only", "noncanonical_config",
    "protocol_object_mismatch", "duplicate_protocol_key"])
def test_effective_receipt_rejects_unresolved_or_noncanonical_snapshot(tmp_path, mutation,
        stub_effective_device_probe):
    _init_repo(tmp_path)
    arguments = _factory_arguments()
    if mutation == "missing_field":
        arguments.pop("sensor_profile")
    elif mutation == "default_only":
        arguments["sensor_profile"] = None
    elif mutation == "protocol_object_mismatch":
        arguments["evaluation_protocol_object"] = {"scenario_id": "forged"}
    elif mutation == "duplicate_protocol_key":
        raw = b'{"scenario_id":"first","scenario_id":"second"}'
        arguments["evaluation_protocol_content_base64"] = __import__("base64").b64encode(raw).decode()
        arguments["evaluation_protocol_content_sha256"] = hashlib.sha256(raw).hexdigest()
        arguments["evaluation_protocol_object"] = {"scenario_id": "second"}
    if mutation != "noncanonical_config":
        with pytest.raises(ValueError):
            compile_effective_capture_receipt(tmp_path, source_plan_path=SOURCE_PLAN,
                factory_arguments=arguments, runtime_identity=_effective_runtime_identity(),
                captured_at_utc="2026-10-25T00:00:00Z")
        return
    receipt = compile_effective_capture_receipt(tmp_path, source_plan_path=SOURCE_PLAN,
        factory_arguments=arguments, runtime_identity=_effective_runtime_identity(),
        captured_at_utc="2026-10-25T00:00:00Z")
    item = next(value for value in receipt["configuration_artifacts"]
        if value["artifact_id"] == "field_layout")
    item["content_base64"] = __import__("base64").b64encode(b'{ "not": "canonical" }').decode()
    item["content_sha256"] = hashlib.sha256(
        __import__("base64").b64decode(item["content_base64"])).hexdigest()
    receipt["receipt_sha256"] = canonical_sha256({k: v for k, v in receipt.items()
        if k != "receipt_sha256"})
    with pytest.raises(ValueError, match="canonical"):
        validate_effective_capture_receipt(receipt, tmp_path)


@pytest.mark.parametrize("mismatch", ["selector", "index", "model", "uuid",
    "torch", "backend", "simulator", "dependency"])
def test_effective_capture_rejects_runtime_device_not_selected_by_factory(tmp_path,
        stub_effective_device_probe, mismatch):
    _init_repo(tmp_path)
    arguments = _factory_arguments()
    runtime = _effective_runtime_identity()
    if mismatch == "selector":
        arguments["device"] = "cuda:1"
    elif mismatch == "index":
        runtime["device_index"] = 1
    elif mismatch == "model":
        runtime["device_model"] = "different model"
    elif mismatch == "uuid":
        runtime["device_id"] = "GPU-ffffffffffffffff"
    elif mismatch == "torch":
        runtime["torch_version"] = "forged"
    elif mismatch == "backend":
        runtime["backend_version"] = "forged"
    elif mismatch == "simulator":
        runtime["simulator_version"] = "forged"
    else:
        runtime["dependency_lock_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="runtime identity differs"):
        compile_effective_capture_receipt(tmp_path, source_plan_path=SOURCE_PLAN,
            factory_arguments=arguments, runtime_identity=runtime,
            captured_at_utc="2026-10-25T00:00:00Z")


def test_receipt_reproduces_snapshot_instead_of_trusting_resealed_snapshot(tmp_path,
        stub_effective_device_probe):
    import copy
    _init_repo(tmp_path)
    receipt = compile_effective_capture_receipt(tmp_path,
        source_plan_path=SOURCE_PLAN, factory_arguments=_factory_arguments(),
        runtime_identity=_effective_runtime_identity(),
        captured_at_utc="2026-10-25T00:00:00Z")
    changed = copy.deepcopy(receipt)
    snapshot = changed["effective_snapshot"]
    snapshot["factory_arguments"]["layout_id"] = "forged-layout"
    snapshot["snapshot_sha256"] = canonical_sha256({key: value
        for key, value in snapshot.items() if key != "snapshot_sha256"})
    changed["effective_snapshot_sha256"] = snapshot["snapshot_sha256"]
    changed["receipt_sha256"] = canonical_sha256({key: value
        for key, value in changed.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError, match="does not reproduce"):
        validate_effective_capture_receipt(changed, tmp_path)


def test_effective_runtime_identity_is_discovered_without_caller_assertion(tmp_path,
        stub_effective_device_probe):
    _init_repo(tmp_path)
    receipt = compile_effective_capture_receipt(tmp_path,
        source_plan_path=SOURCE_PLAN, factory_arguments=_factory_arguments(),
        captured_at_utc="2026-10-25T00:00:00Z")
    assert receipt["runtime"] == _effective_runtime_identity()


def test_tuple_partition_closes_registry_for_five_identical_tracks():
    registry, assignments = _registry_and_assignment()
    freeze_sha = hashlib.sha256(b"fixture freeze").hexdigest()
    materialized = compile_tuple_partition_materialization(freeze_sha256=freeze_sha,
        registry=registry, assignment_table=assignments)
    assert set(materialized["partitions"]) == set(PARTITION_NAMES)
    assert materialized["final_test_access_ledger"]["status"] == "locked"
    assert materialized["final_test_access_ledger"]["outcomes_read"] is False
    assert validate_tuple_partition_materialization(materialized,
        freeze_sha256=freeze_sha, registry=registry, assignment_table=assignments) == materialized


def test_final_test_authorization_transition_is_single_use_and_does_not_read_outcomes():
    registry, assignments = _registry_and_assignment()
    freeze_sha = hashlib.sha256(b"fixture freeze").hexdigest()
    partition = compile_tuple_partition_materialization(freeze_sha256=freeze_sha,
        registry=registry, assignment_table=assignments)
    ledger = dict(partition["final_test_access_ledger"])
    approval_sha = hashlib.sha256(b"approved authorization fixture").hexdigest()
    authorization = {"schema": "meco-final-test-single-use-authorization-v1",
        "partition_sha256": partition["partition_sha256"],
        "approval_artifact_sha256": approval_sha, "authorized_by": "reviewer",
        "purpose": "final_evaluation", "issued_at_utc": "2026-10-25T00:00:00Z"}
    authorization["authorization_sha256"] = canonical_sha256(authorization)
    state, receipt = consume_final_test_authorization(partition,
        prior_ledger_state=ledger, authorization=authorization,
        consumer_id="evaluation-run-1", consumed_at_utc="2026-10-25T00:01:00Z")
    assert state["authorization_count"] == 1
    assert state["outcomes_read"] is False
    assert receipt["outcomes_read"] is False
    assert "outcomes" not in receipt
    with pytest.raises(ValueError, match="consumed|single-use"):
        consume_final_test_authorization(partition, prior_ledger_state=state,
            authorization=authorization, consumer_id="evaluation-run-2",
            consumed_at_utc="2026-10-25T00:02:00Z")


@pytest.mark.parametrize("mutation", ["missing_tuple", "track_membership", "axis_leak", "exclusion_reason"])
def test_tuple_partition_rejects_incomplete_or_leaking_assignments(mutation):
    registry, assignments = _registry_and_assignment()
    if mutation == "missing_tuple":
        assignments["assignments"].pop(next(iter(assignments["assignments"])))
    elif mutation == "track_membership":
        assignments["model_tracks"]["C2"]["tuple_ids"].pop()
    elif mutation == "axis_leak":
        train_id = next(key for key, value in assignments["assignments"].items() if value == "train")
        cal_id = next(key for key, value in assignments["assignments"].items() if value == "calibration")
        train_tuple = next(item for item in registry["tuples"] if canonical_sha256(
            {k: v for k, v in item.items() if k != "eligible"}) == train_id)
        cal_tuple = next(item for item in registry["tuples"] if canonical_sha256(
            {k: v for k, v in item.items() if k != "eligible"}) == cal_id)
        cal_tuple["seed"] = train_tuple["seed"]
        assignments["assignments"].pop(cal_id)
        assignments["assignments"][canonical_sha256({k: v for k, v in cal_tuple.items()
            if k != "eligible"})] = "calibration"
        for track in assignments["model_tracks"].values():
            track["tuple_ids"] = sorted(assignments["assignments"])
    else:
        registry["tuples"][0]["eligible"] = False
        registry["tuples"][0]["exclusion_reason"] = ""
    with pytest.raises(ValueError):
        compile_tuple_partition_materialization(freeze_sha256=hashlib.sha256(b"fixture").hexdigest(),
            registry=registry, assignment_table=assignments)
