import pytest

from frc_defense.meco_track_training import (
    _canonical_digest,
    _validate_paired_calibration_replay_binding,
    _validate_paired_calibration_replay_bindings,
    _validate_paired_calibration_campaign_inputs,
)


def _binding_and_worker():
    work_unit = {
        "scenario_id": "pickup_simple", "category": "single", "split": "calibration",
        "seed": 41, "pair_id": "pair-41", "paired_control_id": "D1",
        "control_id": "D1", "repetition": 2,
        "layout_id": "layout-a", "defender_family": "family-a",
        "start_condition": "start-a", "sensor_profile": "sensor-a",
    }
    binding = {
        "campaign_id": "campaign-1", "shard_index": 0,
        "shard_manifest_id": "manifest-1", "work_unit_id": _canonical_digest(work_unit),
        "work_unit": work_unit, "pair_id": "pair-41", "control_id": "D1",
        "source_binding": {"source": "frozen"}, "baseline_freeze_sha256": "a" * 64,
        "partition_sha256": "b" * 64, "estimand_approval_sha256": "c" * 64,
        "scoring": {"replay_contract_schema": "dashboard-replay-generation-v1"},
        "control_binding": {"control_id": "D1", "policy": "D1",
            "prediction_mode": "current", "prediction_contract": {"model": "current"},
            "prediction_contract_sha256": _canonical_digest({"model": "current"})},
    }
    binding["control_binding_sha256"] = _canonical_digest(binding["control_binding"])
    binding["control_binding"]["control_binding_sha256"] = binding["control_binding_sha256"]
    worker = {"worker_id": "worker-0", "device": "cuda:0", "device_index": 0,
        "device_model": "AMD Instinct MI300X WX9100", "device_uuid": "GPU-uuid",
        "backend": "hip", "backend_version": "6.3", "torch_version": "2.8"}
    runtime = {key: worker[key] for key in (
        "device_model", "device_uuid", "backend", "backend_version", "torch_version")}
    return binding, worker, runtime


def test_paired_replay_binding_accepts_exact_campaign_and_runtime_identity():
    binding, worker, runtime = _binding_and_worker()
    _validate_paired_calibration_replay_binding(binding, worker, device="cuda:0",
        execution_device={"device": "cuda:0", "accelerator_backend": "ROCm"},
        scenario_id="pickup_simple", split="calibration", policy="D1",
        prediction_mode="current", seed=41,
        qualification_case={"category": "single", "layout_id": "layout-a", "defender_family": "family-a",
            "start_condition": "start-a", "sensor_profile": "sensor-a"},
        condition_id="paired-cal-campaign-1-pair-41-D1-r0002",
        runtime_identity=runtime)


@pytest.mark.parametrize("mutation", ["digest", "pair", "device", "runtime", "schema", "control", "control_body"])
def test_paired_replay_binding_rejects_changed_provenance(mutation):
    binding, worker, runtime = _binding_and_worker()
    if mutation == "digest":
        binding["work_unit_id"] = "0" * 64
    elif mutation == "pair":
        binding["pair_id"] = "other"
    elif mutation == "device":
        worker["device"] = "cuda:1"
    elif mutation == "runtime":
        runtime["device_uuid"] = "GPU-other"
    elif mutation == "control":
        binding["control_binding_sha256"] = "d" * 64
    elif mutation == "control_body":
        binding["control_binding"]["candidate_configuration"] = {"candidate_k": 32}
    else:
        binding["scoring"]["replay_contract_schema"] = "old"
    with pytest.raises(ValueError):
        _validate_paired_calibration_replay_binding(binding, worker, device="cuda:0",
            execution_device={"device": "cuda:0", "accelerator_backend": "ROCm"},
            scenario_id="pickup_simple", split="calibration", policy="D1",
            prediction_mode="current", seed=41,
            qualification_case={"category": "single", "layout_id": "layout-a", "defender_family": "family-a",
                "start_condition": "start-a", "sensor_profile": "sensor-a"},
        condition_id="paired-cal-campaign-1-pair-41-D1-r0002",
            runtime_identity=runtime)


def test_per_seed_bindings_cover_each_score_seed_and_derive_exact_conditions():
    binding_a, worker, runtime = _binding_and_worker()
    binding_b, _, _ = _binding_and_worker()
    binding_b["work_unit"] = {**binding_b["work_unit"], "seed": 42,
        "pair_id": "pair-42", "paired_control_id": "D1"}
    binding_b["work_unit_id"] = _canonical_digest(binding_b["work_unit"])
    binding_b["pair_id"] = "pair-42"
    bindings = {41: binding_a, 42: binding_b}
    _validate_paired_calibration_replay_bindings(bindings, (41, 42), worker,
        device="cuda:0", execution_device={"device": "cuda:0", "accelerator_backend": "ROCm"},
        scenario_id="pickup_simple", split="calibration", policy="D1",
        prediction_mode="current", qualification_case={"category": "single",
            "layout_id": "layout-a", "defender_family": "family-a",
            "start_condition": "start-a", "sensor_profile": "sensor-a"},
        runtime_identity=runtime)


@pytest.mark.parametrize("seeds", [(41, 42), (41, 43)])
def test_per_seed_bindings_reject_incomplete_or_extra_seed_coverage(seeds):
    binding, worker, runtime = _binding_and_worker()
    with pytest.raises(ValueError, match="cover every exact scored seed"):
        _validate_paired_calibration_replay_bindings({41: binding}, seeds, worker,
            device="cuda:0", execution_device={"device": "cuda:0", "accelerator_backend": "ROCm"},
            scenario_id="pickup_simple", split="calibration", policy="D1",
            prediction_mode="current", qualification_case={"category": "single",
                "layout_id": "layout-a", "defender_family": "family-a",
                "start_condition": "start-a", "sensor_profile": "sensor-a"},
            runtime_identity=runtime)


def _trusted_campaign_fixture(monkeypatch):
    from frc_defense import meco_qualification as qualification

    worker = {"worker_id": "worker-0", "device": "cuda:0", "device_index": 0,
        "device_model": "AMD Instinct WX9100", "device_uuid": "GPU-uuid",
        "backend": "hip", "backend_version": "6.3", "torch_version": "2.8"}
    control = {"control_id": "D1", "policy": "D1", "policy_id": "D1_ref",
        "prediction_mode": "current", "prediction_contract": {"model": "current"},
        "prediction_contract_sha256": _canonical_digest({"model": "current"}),
        "candidate_configuration": {"candidate_k": 16,
            "route_cost_points_per_second": 2.5}, "execution_supported": True,
        "support_limit": None}
    control["control_binding_sha256"] = _canonical_digest(control)
    control_binding = dict(control)
    units = [{"scenario_id": "pickup_simple", "category": "single",
        "split": "calibration", "seed": seed, "pair_id": f"pair-{seed}",
        "paired_control_id": "D1", "control_id": "D1", "repetition": 2,
        "layout_id": "layout-a",
        "defender_family": "family-a", "start_condition": "start-a",
        "sensor_profile": "sensor-a"} for seed in (41, 42)]
    campaign = {"source": {"source_identity_sha256": "source-ok"},
        "baseline_freeze": {"freeze_sha256": "freeze-ok"},
        "partition": {"partition_sha256": "partition-ok"},
        "estimand_approval": {"approval_sha256": "approval-ok"},
        "worker_assignments": [worker], "controls": [control]}
    bindings = {unit["seed"]: {"campaign_id": "campaign-1",
        "shard_index": 0, "shard_manifest_id": "manifest-1",
        "work_unit_id": _canonical_digest(unit), "work_unit": unit,
        "pair_id": unit["pair_id"], "control_id": unit["control_id"],
        "source_binding": campaign["source"],
        "baseline_freeze_sha256": "freeze-ok", "partition_sha256": "partition-ok",
        "estimand_approval_sha256": "approval-ok",
        "scoring": {"replay_contract_schema": "dashboard-replay-generation-v1"},
        "control_binding": control_binding,
        "control_binding_sha256": control["control_binding_sha256"]}
        for unit in units}
    campaign["bindings"] = bindings
    shard = {"worker_id": worker["worker_id"], "work_units": units}

    def validate_gate(value, **kwargs):
        del kwargs
        if (value["source"] != {"source_identity_sha256": "source-ok"}
                or value["baseline_freeze"].get("freeze_sha256") != "freeze-ok"
                or value["partition"].get("partition_sha256") != "partition-ok"
                or value["estimand_approval"].get("approval_sha256") != "approval-ok"):
            raise ValueError("trusted artifact reference mismatch")
        return {"approval": {"bytes": "validated"}}

    monkeypatch.setattr(qualification, "validate_paired_calibration_gate_artifacts", validate_gate,
        raising=False)
    monkeypatch.setattr(qualification, "_paired_shard_for_index",
        lambda value, index: shard if value is campaign and index == 0 else None,
        raising=False)
    monkeypatch.setattr(qualification, "paired_calibration_replay_binding",
        lambda value, index, unit: value["bindings"][unit["seed"]]
        if value is campaign and index == 0 else None, raising=False)
    return campaign, bindings, worker


def _validate_trusted_fixture(campaign, bindings, worker):
    return _validate_paired_calibration_campaign_inputs(campaign, 0, bindings,
        worker, (41, 42), device="cuda:0", repo_root="/unused", configuration_artifacts={},
        policy="D1", calibration_control_policy="D1", candidate_k=16,
        route_cost_weight=2.5, prediction_mode="current")


def test_campaign_inputs_resolve_to_exact_trusted_per_seed_bindings(monkeypatch):
    campaign, bindings, worker = _trusted_campaign_fixture(monkeypatch)
    assert _validate_trusted_fixture(campaign, bindings, worker) == {
        "approval": {"bytes": "validated"}}


@pytest.mark.parametrize("field", ["source", "baseline_freeze", "partition", "estimand_approval"])
def test_campaign_inputs_reject_mutated_trusted_artifact_references(monkeypatch, field):
    campaign, bindings, worker = _trusted_campaign_fixture(monkeypatch)
    campaign[field] = {"source_identity_sha256": "tampered"} if field == "source" else {"sha256": "tampered"}
    with pytest.raises(ValueError, match="trusted artifact reference mismatch"):
        _validate_trusted_fixture(campaign, bindings, worker)


@pytest.mark.parametrize("mutation", ["control_body", "binding", "worker"])
def test_campaign_inputs_reject_control_binding_and_worker_mutations(monkeypatch, mutation):
    campaign, bindings, worker = _trusted_campaign_fixture(monkeypatch)
    mutated_bindings = dict(bindings)
    mutated_worker = dict(worker)
    if mutation == "control_body":
        changed = dict(bindings[41])
        changed_control = dict(changed["control_binding"])
        changed_control["candidate_configuration"] = {"candidate_k": 32}
        changed["control_binding"] = changed_control
        changed["control_binding_sha256"] = _canonical_digest(changed_control)
        mutated_bindings[41] = changed
    elif mutation == "binding":
        changed = dict(bindings[42])
        changed["work_unit_id"] = "f" * 64
        mutated_bindings[42] = changed
    else:
        mutated_worker["device_uuid"] = "GPU-other"
    with pytest.raises(ValueError):
        _validate_trusted_fixture(campaign, mutated_bindings, mutated_worker)


@pytest.mark.parametrize("mutation", ["source", "freeze", "partition", "approval",
    "worker", "device", "binding", "control"])
def test_evaluate_rejects_campaign_mutations_before_device_or_environment(monkeypatch, mutation):
    from frc_defense import meco_track_training

    campaign, bindings, worker = _trusted_campaign_fixture(monkeypatch)
    mutated_bindings = dict(bindings)
    mutated_worker = dict(worker)
    requested_device = "cpu"
    if mutation == "source":
        campaign["source"] = {"source_identity_sha256": "wrong"}
    elif mutation == "freeze":
        campaign["baseline_freeze"] = {"freeze_sha256": "wrong"}
    elif mutation == "partition":
        campaign["partition"] = {"partition_sha256": "wrong"}
    elif mutation == "approval":
        campaign["estimand_approval"] = {"approval_sha256": "wrong"}
    elif mutation == "worker":
        mutated_worker["device_index"] = 1
    elif mutation == "device":
        requested_device = "cuda:1"
    elif mutation == "binding":
        changed = dict(bindings[42])
        changed["work_unit_id"] = "0" * 64
        mutated_bindings[42] = changed
    else:
        changed = dict(bindings[41])
        altered_control = dict(changed["control_binding"])
        altered_control["candidate_configuration"] = {"candidate_k": 32}
        changed["control_binding"] = altered_control
        mutated_bindings[41] = changed
    with pytest.raises(ValueError):
        meco_track_training.evaluate_meco({}, scenario_id="pickup_simple",
            categories=("single",), episodes=2, master_seed=41, device=requested_device,
            episode_seeds=(41, 42), split="calibration", qualification_case={
                "layout_id": "layout-a", "defender_family": "family-a",
                "start_condition": "start-a", "sensor_profile": "sensor-a"},
            evaluation_policies=(), control_policies=("D0", "D1"),
            control_prediction_modes={"D1": "current"},
            calibration_control_policy="D1", candidate_k_override=16,
            calibration_route_cost_points_per_second=2.5,
            replay_output_dir="/tmp/unused-paired-replays",
            replay_selection={"D1": [41, 42]}, simulated_score_estimand="unused-approval.json",
            paired_calibration_bindings=mutated_bindings,
            paired_calibration_worker_identity=mutated_worker,
            paired_calibration_campaign=campaign, paired_calibration_shard_index=0,
            paired_calibration_configuration_artifacts={})
