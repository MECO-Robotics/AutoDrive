import hashlib
import json

import pytest

from frc_defense.meco_ablation_matrix import (
    DEFAULT_EVALUATION_STRATUM, _load_seed_manifest,
    _held_out_seed_assignments, _seal_final_test_consumption, _validate_replay_selection,
    default_conditions,
)
from scripts.summarize_meco_ablation_matrix import _diagnostic_performance_summaries
from frc_defense.meco_final_test_ledger import assert_training_not_released_after_final_test
from frc_defense.meco_ablation_matrix import run_matrix
from frc_defense.meco_tracks import MECO_TRACKS
from scripts.run_meco_ablation_matrix import _parser


def _seed_manifest(tmp_path):
    value = {"schema": "meco-exact-seed-manifest-v1",
        "protocol_id": "protocol-a", "protocol_sha256": "sha-a",
        "scenario_id": "pickup_simple", "categories": ["single"],
        "split": "validation", "qualification_case": DEFAULT_EVALUATION_STRATUM,
        "seed_list": [11, 12]}
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def test_exact_seed_manifest_is_bound_to_protocol_split_stratum_and_categories(tmp_path):
    path, value = _seed_manifest(tmp_path)
    assert _load_seed_manifest(path, protocol_id="protocol-a", protocol_sha256="sha-a",
        scenario_id="pickup_simple", categories=("single",), split="validation",
        qualification_case=DEFAULT_EVALUATION_STRATUM) == (11, 12)
    for key, wrong in (("split", "test"), ("protocol_id", "other"),
                       ("protocol_sha256", "other-sha"),
                       ("scenario_id", "pickup_scattered"),
                       ("categories", ["cluster"]),
                       ("qualification_case", {**DEFAULT_EVALUATION_STRATUM,
                           "layout_id": "other-layout"})):
        altered = dict(value, **{key: wrong})
        path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError, match="mismatch"):
            _load_seed_manifest(path, protocol_id="protocol-a", protocol_sha256="sha-a",
                scenario_id="pickup_simple", categories=("single",), split="validation",
                qualification_case=DEFAULT_EVALUATION_STRATUM)
    path.write_text(json.dumps(dict(value, seed_list=[11, 11])), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        _load_seed_manifest(path, protocol_id="protocol-a", protocol_sha256="sha-a",
            scenario_id="pickup_simple", categories=("single",), split="validation",
            qualification_case=DEFAULT_EVALUATION_STRATUM)


def test_protocol_seed_lookup_rejects_a_seed_from_another_stratum():
    protocol = {"cases": [{"scenario_id": "pickup_simple", "category": "single",
        "split": "validation", **DEFAULT_EVALUATION_STRATUM, "seeds": [11]},
        {"scenario_id": "pickup_simple", "category": "single", "split": "test",
         **{**DEFAULT_EVALUATION_STRATUM, "layout_id": "other-layout"}, "seeds": [12]}]}
    assert _held_out_seed_assignments(protocol, "pickup_simple", ("single",), (11,),
        DEFAULT_EVALUATION_STRATUM) == {"single": {"11": "validation"}}
    with pytest.raises(ValueError, match="not assigned"):
        _held_out_seed_assignments(protocol, "pickup_simple", ("single",), (12,),
            DEFAULT_EVALUATION_STRATUM)


def test_replay_selection_binds_only_requested_condition_policy_and_seeds():
    conditions = default_conditions("pickup_simple", categories=("single",),
        qualification_case=DEFAULT_EVALUATION_STRATUM)
    selection = {"architecture_matched_cv_k16": {"A1": [11], "D0": [12]}}
    assert _validate_replay_selection(selection, conditions, ("single",), (11, 12)) == selection
    with pytest.raises(ValueError, match="unknown or empty condition"):
        _validate_replay_selection({"other": {"A1": [11]}}, conditions,
            ("single",), (11, 12))
    with pytest.raises(ValueError, match="invalid matrix seed"):
        _validate_replay_selection({"architecture_matched_cv_k16": {"A1": [13]}},
            conditions, ("single",), (11, 12))


def test_final_test_consumption_is_one_exclusive_event_bound_to_exact_inputs(tmp_path):
    path = tmp_path / "sealed-final-test-consumption.json"
    protocol = {"protocol_id": "protocol-test", "protocol_sha256": "protocol-sha"}
    checkpoints = {"A1": hashlib.sha256(b"checkpoint").hexdigest()}
    kwargs = dict(matrix_id="matrix-digest", protocol=protocol,
        checkpoint_hashes=checkpoints, source_hash="source-digest",
        seed_list=(101, 102), scenario_id="pickup_simple", categories=("single",),
        qualification_case=DEFAULT_EVALUATION_STRATUM)
    _seal_final_test_consumption(path, **kwargs)
    ledger = json.loads(path.read_text(encoding="utf-8"))
    assert ledger["matrix_sha256"] == "matrix-digest"
    assert ledger["protocol_sha256"] == "protocol-sha"
    assert ledger["checkpoint_sha256"] == checkpoints
    assert ledger["source_identity_sha256"] == "source-digest"
    assert ledger["seed_list"] == [101, 102]
    assert ledger["split"] == "test"
    with pytest.raises(ValueError, match="already consumed"):
        _seal_final_test_consumption(path, **kwargs)


def test_performance_summary_bootstraps_record_level_paired_diagnostics():
    case = {**DEFAULT_EVALUATION_STRATUM}
    candidate = {"layout_id": case["layout_id"],
        "defender_family": case["defender_family"],
        "diagnostic_performance": {"end_to_end_wall_ms": 12.0,
            "fallback_count": 1, "fallback_rate": .1,
            "route_policy_pipeline_ms": {"p50_ms": 4., "p95_ms": 7., "p99_ms": 8.},
            "inference_ms": {"p50_ms": 2., "p95_ms": 3., "p99_ms": 3.5},
            "simulator_macro_ms": {"p50_ms": 5., "p95_ms": 8., "p99_ms": 9.}}}
    reference = {**candidate,
        "diagnostic_performance": {"end_to_end_wall_ms": 10.0,
            "fallback_count": 0, "fallback_rate": 0.,
            "route_policy_pipeline_ms": {"p50_ms": 3., "p95_ms": 6., "p99_ms": 7.},
            "inference_ms": {"p50_ms": 1., "p95_ms": 2., "p99_ms": 2.5},
            "simulator_macro_ms": {"p50_ms": 4., "p95_ms": 7., "p99_ms": 8.}}}
    matrix = {"contrasts": [{"paired_episode_rows": [{"category": "single", "seed": 11}],
        "condition_id": "condition", "candidate_policy": "A1", "reference_policy": "D0",
        "contrast": "A1_vs_D0"}]}
    loaded = {"reports": {"condition": ({}, {
        ("A1", "single", 11): candidate, ("D0", "single", 11): reference})}}
    result = _diagnostic_performance_summaries(matrix, loaded,
        bootstrap_seed=7, replicates=100)
    indexed = {row["metric"]: row for row in result}
    assert indexed["end_to_end_wall_ms"]["mean_difference"] == 2.
    assert indexed["fallback_count"]["mean_difference"] == 1.
    assert indexed["inference_ms_p99_ms"]["mean_difference"] == 1.
    assert indexed["route_policy_pipeline_ms_p95_ms"]["ci_95"] == [1., 1.]
    end_to_end_quantiles = next(row for row in result
        if row["metric"] == "end_to_end_wall_latency_quantiles")
    assert end_to_end_quantiles["quantile_differences"]["p99_ms"] == {
        "difference": 2., "ci_95": [2., 2.]}


def test_sealed_final_test_blocks_training_and_detects_source_drift(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol_id, protocol_sha, source_hash = "p1", "p-sha", "source-sha"
    ledger_path = protocol_path.with_name(f"{protocol_id}.sealed-final-test-consumption.json")
    ledger_path.write_text(json.dumps({"protocol_id": protocol_id,
        "protocol_sha256": protocol_sha, "source_identity_sha256": source_hash}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="source identity changed"):
        assert_training_not_released_after_final_test(
            protocol_path, protocol_id, protocol_sha, "modified-source")
    with pytest.raises(ValueError, match="training is forbidden"):
        assert_training_not_released_after_final_test(
            protocol_path, protocol_id, protocol_sha, source_hash)


@pytest.mark.parametrize("initialization", ["fresh", "resume", "warm_start"])
def test_train_meco_api_blocks_after_final_test_for_all_initialization_modes(
        tmp_path, monkeypatch, initialization):
    import frc_defense.fuel_physics_hip as hip
    import frc_defense.meco_supervised as supervised
    import frc_defense.meco_track_training as training
    import frc_defense.meco_device as device

    protocol_path = tmp_path / "protocol.json"
    protocol_id, protocol_sha = "p1", "p-sha"
    source = {"source": "prepared"}
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    protocol_path.with_name(
        f"{protocol_id}.sealed-final-test-consumption.json").write_text(
            json.dumps({"protocol_id": protocol_id,
                "protocol_sha256": protocol_sha,
                "source_identity_sha256": source_hash}), encoding="utf-8")
    monkeypatch.setattr(supervised, "_load_protocol",
        lambda path: ({"protocol_id": protocol_id}, protocol_id, protocol_sha, path))
    monkeypatch.setattr(supervised, "validate_training_assignment", lambda *args: None)
    monkeypatch.setattr(training, "source_identity", lambda: source)
    monkeypatch.setattr(training.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(device, "require_indexed_cuda_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(hip, "prepare_native_fuel_extension", lambda device: None)
    kwargs = {}
    if initialization == "resume":
        kwargs["resume"] = str(tmp_path / "resume.pt")
    elif initialization == "warm_start":
        kwargs.update(warm_start=str(tmp_path / "warm.pt"), protocol_id="train-p1")
    config = training.MecoTrainConfig(track="A1", device="cuda:0",
        output=str(tmp_path / "output"))
    with pytest.raises(ValueError, match="training is forbidden"):
        training.train_meco(config, qualification_protocol=str(protocol_path), **kwargs)
    assert not (tmp_path / "output").exists()


def test_matrix_cli_defaults_to_accelerator_and_rejects_cpu_before_gpu_setup(tmp_path):
    required_checkpoints = []
    for track in MECO_TRACKS:
        required_checkpoints.extend([f"--{track.lower()}-checkpoint", f"{track}.pt"])
    args = _parser().parse_args(["--scenario", "pickup_simple", "--categories", "single",
        *required_checkpoints])
    assert args.device == "cuda:0"
    with pytest.raises(ValueError, match="CPU is unsupported"):
        run_matrix(checkpoints={track: f"{track}.pt" for track in MECO_TRACKS},
            scenario_id="pickup_simple", categories=("single",), seeds=(1,),
            device="cpu", output_dir=tmp_path)
