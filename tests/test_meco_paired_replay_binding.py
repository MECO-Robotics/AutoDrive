import pytest

from frc_defense.meco_track_training import (
    _canonical_digest,
    _validate_paired_calibration_replay_binding,
)


def _binding_and_worker():
    work_unit = {
        "scenario_id": "pickup_simple", "category": "single", "split": "calibration",
        "seed": 41, "pair_id": "pair-1", "control_id": "D1", "repetition": 2,
        "layout_id": "layout-a", "defender_family": "family-a",
        "start_condition": "start-a", "sensor_profile": "sensor-a",
    }
    binding = {
        "campaign_id": "campaign-1", "shard_index": 0,
        "shard_manifest_id": "manifest-1", "work_unit_id": _canonical_digest(work_unit),
        "work_unit": work_unit, "pair_id": "pair-1", "control_id": "D1",
        "source_binding": {"source": "frozen"}, "baseline_freeze_sha256": "a" * 64,
        "partition_sha256": "b" * 64, "estimand_approval_sha256": "c" * 64,
        "scoring": {"replay_contract_schema": "dashboard-replay-generation-v1"},
    }
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
        scenario_id="pickup_simple", split="calibration", policy="D1", seed=41,
        qualification_case={"category": "single", "layout_id": "layout-a", "defender_family": "family-a",
            "start_condition": "start-a", "sensor_profile": "sensor-a"},
        condition_id="paired-cal-campaign-1-pair-1-D1-r0002",
        runtime_identity=runtime)


@pytest.mark.parametrize("mutation", ["digest", "pair", "device", "runtime", "schema"])
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
    else:
        binding["scoring"]["replay_contract_schema"] = "old"
    with pytest.raises(ValueError):
        _validate_paired_calibration_replay_binding(binding, worker, device="cuda:0",
            execution_device={"device": "cuda:0", "accelerator_backend": "ROCm"},
            scenario_id="pickup_simple", split="calibration", policy="D1", seed=41,
            qualification_case={"category": "single", "layout_id": "layout-a", "defender_family": "family-a",
                "start_condition": "start-a", "sensor_profile": "sensor-a"},
        condition_id="paired-cal-campaign-1-pair-1-D1-r0002",
            runtime_identity=runtime)
