import math

import pytest
import torch
from types import SimpleNamespace

from frc_defense.meco_tracks import (
    MULTIMODEL_HYPOTHESES,
    MULTIMODEL_ID,
    PREDICTION_MODES,
    opponent_forecast,
    opponent_risk_penalty,
    prediction_contract,
)


def test_multimodel_contract_is_explicit_and_versioned_for_matched_tracks():
    assert "multi_model_v1" in PREDICTION_MODES
    contract = prediction_contract("multi_model_v1", horizon_s=5.)
    assert contract["model"] == MULTIMODEL_ID == "meco_opponent_kinematics_mm_v1"
    assert contract["mode"] == "multi_model_v1"
    assert contract["max_horizon_s"] == 2.5
    assert contract["horizon_semantics"] == "candidate_route_eta_clamped_to_max_horizon"
    assert [(row["id"], row["weight"]) for row in contract["hypotheses"]] == list(
        MULTIMODEL_HYPOTHESES)
    assert math.isclose(sum(row["weight"] for row in contract["hypotheses"]), 1.)


@pytest.mark.parametrize("invalid_seed", [True, 1.0])
def test_matrix_record_index_rejects_non_integer_seed_aliases(invalid_seed):
    from frc_defense.meco_ablation_matrix import _records_by_key

    with pytest.raises(ValueError, match="record seed must be an integer"):
        _records_by_key({"records": [{"policy": "A1", "category": "single",
                                      "seed": invalid_seed}]})


def test_matrix_record_index_keeps_valid_integer_seed_pairing():
    from frc_defense.meco_ablation_matrix import _records_by_key

    candidate = {"policy": "A1", "category": "single", "seed": 1}
    reference = {"policy": "D0", "category": "single", "seed": 1}
    records = _records_by_key({"records": [candidate, reference]})

    assert records[("A1", "single", 1)] is candidate
    assert records[("D0", "single", 1)] is reference


def test_multimodel_forecast_is_bounded_deterministic_and_risk_uses_mixture():
    kwargs = dict(mode="multi_model_v1", valid=True, position_xy=(1., 2.),
                  velocity_xy=(8., 0.), age_s=.1, horizon_s=4.,
                  angular_velocity_rad_s=.2)
    first = opponent_forecast(**kwargs)
    second = opponent_forecast(**kwargs)
    assert first == second
    assert first["model"] == MULTIMODEL_ID
    assert first["horizon_s"] == 2.5
    assert first["requested_horizon_s"] == 4.
    assert first["horizon_clamped"] is True
    assert first["used_multimodel"] is True
    assert len(first["hypotheses"]) == 4
    # Every scenario is capped at 3 m/s before its kinematics are integrated.
    assert all(math.dist((1., 2.), row["xy"]) <= 7.5000001
               for row in first["hypotheses"])
    assert opponent_risk_penalty((4., 2.), first) == pytest.approx(
        sum(weight * .75 * max(0., min(1., (2. - math.dist((4., 2.), xy)) / 2.))
            for (name, weight), xy in zip(MULTIMODEL_HYPOTHESES,
                (row["xy"] for row in first["hypotheses"]))))


@pytest.mark.parametrize("kwargs,reason", [
    (dict(valid=True, position_xy=(1., 2.), velocity_xy=(1., 0.), age_s=.6),
     "observed_current_pose"),
    (dict(valid=True, position_xy=(1., 2.), velocity_xy=(float("nan"), 0.), age_s=.1),
     "observed_current_pose"),
])
def test_multimodel_stale_or_nonfinite_motion_falls_back_to_finite_pose(kwargs, reason):
    result = opponent_forecast(mode="multi_model_v1", horizon_s=1., **kwargs)
    assert result["xy"] == (1., 2.)
    assert result["fallback"] == reason
    assert result["used_multimodel"] is False
    assert result["hypotheses"] == [{"id": "observed_current_pose",
                                     "weight": 1., "xy": (1., 2.)}]
    assert opponent_risk_penalty((1., 2.), result) == pytest.approx(.75)


def test_multimodel_missing_or_nonfinite_pose_has_no_risk():
    missing = opponent_forecast(mode="multi_model_v1", valid=False,
        position_xy=(0., 0.), velocity_xy=(1., 0.), age_s=.1, horizon_s=1.)
    nonfinite = opponent_forecast(mode="multi_model_v1", valid=True,
        position_xy=(float("nan"), 0.), velocity_xy=(1., 0.), age_s=.1, horizon_s=1.)
    assert missing["xy"] is None
    assert nonfinite["xy"] is None
    assert missing["fallback"] == "none_no_finite_observed_pose"
    assert opponent_risk_penalty((0., 0.), missing) is None
    assert opponent_risk_penalty((0., 0.), nonfinite) is None


def test_plan_a_node_risk_matches_shared_mixture_per_route_eta():
    from frc_defense.meco_track_training import _plan_a_prediction_risk

    goals = torch.tensor(((2., 1.), (4., 2.), (7., 3.)))
    support = SimpleNamespace(goal_xy=goals,
        valid_mask=torch.ones(3 * 8, dtype=torch.bool))
    env = SimpleNamespace(
        opponent_valid=torch.tensor([[False, False, False, True]]),
        opponent_pose=torch.tensor([[[0., 0., 0.], [0., 0., 0.],
                                    [0., 0., 0.], [1., 2., 0.]]]),
        opponent_velocity=torch.tensor([[[0., 0., 0.], [0., 0., 0.],
                                         [0., 0., 0.], [2., .5, .8]]]),
        opponent_age=torch.tensor([[0., 0., 0., .1]]))
    eta = torch.tensor((.3, 1.4, 4.))
    actual, metadata = _plan_a_prediction_risk(env, support, eta, "multi_model_v1")
    expected = []
    for goal, horizon in zip(goals.tolist(), eta.tolist()):
        forecast = opponent_forecast(mode="multi_model_v1", valid=True,
            position_xy=(1., 2.), velocity_xy=(2., .5), age_s=.1,
            horizon_s=horizon, angular_velocity_rad_s=.8)
        expected.append(opponent_risk_penalty(goal, forecast))
    assert actual.tolist() == pytest.approx(expected, abs=2e-6)
    assert metadata["used_multimodel"] is True
    assert metadata["risk_semantics"] == "weighted_expected_penalty_over_hypothesis_endpoints"


def test_matrix_accepts_multimodel_in_matched_learned_and_control_contract():
    from frc_defense.meco_ablation_matrix import MatrixCondition, default_conditions
    from frc_defense.meco_track_training import (
        _parser, _prediction_record_fields, _resolve_control_prediction_modes)

    condition = MatrixCondition("matched_mm", "multi_model_v1", "multi_model_v1", 16)
    condition.validate()
    with pytest.raises(ValueError, match="both"):
        MatrixCondition("ambiguous", "cv", "cv", 16,
            control_prediction_modes={"D1": "current"}).validate()
    args = _parser().parse_args(["evaluate", "--device", "cuda:0", "--output", "report.json",
        "--prediction-mode", "multi_model_v1", "--control-prediction-mode", "multi_model_v1",
        "--a1-checkpoint", "a1.pt", "--a2-checkpoint", "a2.pt",
        "--a3-checkpoint", "a3.pt", "--c1-checkpoint", "c1.pt", "--c2-checkpoint", "c2.pt"])
    assert args.prediction_mode == args.control_prediction_mode == "multi_model_v1"
    row_provenance = _prediction_record_fields(args.control_prediction_mode)
    assert row_provenance["prediction_mode"] == "multi_model_v1"
    assert row_provenance["prediction_contract"] == prediction_contract("multi_model_v1")

    args = _parser().parse_args(["evaluate", "--device", "cuda:0", "--output", "report.json",
        "--control-prediction-modes", '{"D1":"current","D2":"multi_model_v1"}',
        "--a1-checkpoint", "a1.pt", "--a2-checkpoint", "a2.pt",
        "--a3-checkpoint", "a3.pt", "--c1-checkpoint", "c1.pt", "--c2-checkpoint", "c2.pt"])
    assert args.control_prediction_modes == {"D1": "current", "D2": "multi_model_v1"}
    assert _resolve_control_prediction_modes(
        control_prediction_modes=args.control_prediction_modes) == args.control_prediction_modes
    assert _resolve_control_prediction_modes(control_prediction_mode="multi_model_v1") == {
        "D1": "multi_model_v1", "D2": "multi_model_v1"}
    assert _resolve_control_prediction_modes() == {"D1": "current", "D2": "cv"}
    with pytest.raises(ValueError, match="either"):
        _resolve_control_prediction_modes("cv", {"D2": "multi_model_v1"})
    with pytest.raises(ValueError, match="map"):
        _resolve_control_prediction_modes(control_prediction_modes={"D3": "cv"})

    controls = {condition.condition_name: condition for condition in
        default_conditions("pickup_simple")}
    assert controls["controls_default_d1_current_d2_cv_k16"].control_prediction_modes is None
    multi = controls["controls_d1_current_d2_multimodel_v1_k16"]
    assert multi.control_prediction_modes == {"D1": "current", "D2": "multi_model_v1"}
    assert _resolve_control_prediction_modes(
        control_prediction_modes=multi.control_prediction_modes) == {
            "D1": "current", "D2": "multi_model_v1"}


@pytest.mark.parametrize("row_provenance_tamper", [
    None, "missing", "tampered",
    "categories_missing", "categories_empty", "categories_truncated",
    "seeds_missing", "seeds_empty", "seeds_truncated",
    "seeds_bool_equal", "seeds_float_equal",
])
def test_matrix_runner_executes_and_records_per_control_condition(
        monkeypatch, tmp_path, row_provenance_tamper):
    import hashlib
    import json

    import frc_defense.curriculum_training as curriculum_training
    import frc_defense.fuel_physics_hip as fuel_physics_hip
    import frc_defense.meco_ablation_matrix as matrix
    import frc_defense.meco_device as meco_device
    import frc_defense.meco_track_training as track_training
    from frc_defense.meco_tracks import MECO_TRACKS, prediction_contract

    source = {"source": "synthetic matrix runner fixture"}
    monkeypatch.setattr(curriculum_training, "source_identity", lambda: source)
    monkeypatch.setattr(fuel_physics_hip, "prepare_native_fuel_extension", lambda device: None)
    monkeypatch.setattr(meco_device, "normalized_device_info", lambda device: {
        "device": "cpu", "accelerator_backend": "cpu"})
    monkeypatch.setattr(matrix, "normalized_device_info", lambda device: {
        "device": "cpu", "accelerator_backend": "cpu"})
    config_hashes = {track: f"config-{track}" for track in MECO_TRACKS}
    protocol_ids = {track: "shared-training-protocol" for track in MECO_TRACKS}
    shared_config = {"seed": 12}
    monkeypatch.setattr(matrix, "_checkpoint_configuration_provenance",
        lambda checkpoints: (config_hashes, protocol_ids, shared_config))

    calls = []

    def fake_evaluate(checkpoints, **kwargs):
        calls.append(kwargs)
        source_hash = hashlib.sha256(json.dumps(source, sort_keys=True,
            allow_nan=False).encode()).hexdigest()
        modes = track_training._resolve_control_prediction_modes(
            kwargs["control_prediction_mode"], kwargs["control_prediction_modes"])
        manifest = {}
        for track in MECO_TRACKS:
            manifest[track] = {"checkpoint_sha256": hashlib.sha256(
                Path(checkpoints[track]).read_bytes()).hexdigest(),
                "source_identity_sha256": source_hash,
                "training_configuration_sha256": config_hashes[track],
                "training_protocol_id": protocol_ids[track]}
        for control, mode in (("D0", None), ("D1", modes["D1"]),
                              ("D2", modes["D2"])):
            manifest[control] = {"prediction_mode": mode,
                "prediction_contract": prediction_contract(mode) if mode else None}
        for track in ("A1", "A2", "A3"):
            manifest[f"{track}_P0"] = dict(manifest[track])
        records = []
        policy_names = (*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2")
        if kwargs["inference_disable_tick"] is not None:
            policy_names += tuple(f"{track}__INFERENCE_DISABLED_TO_D1"
                                  for track in MECO_TRACKS)
            for track in MECO_TRACKS:
                manifest[f"{track}__INFERENCE_DISABLED_TO_D1"] = dict(manifest[track])
        for policy in policy_names:
            for category in kwargs["categories"]:
                for seed in kwargs["episode_seeds"]:
                    records.append({"policy": policy, "category": category,
                        "seed": seed, "policy_id": f"synthetic-{policy}",
                        **track_training._prediction_record_fields(
                            modes[policy] if policy in modes else None),
                        "metrics": {"score": 0.}, "candidate_recall": 1.})
        if row_provenance_tamper is not None:
            # Corrupt the multi-model condition's D2 row, preserving the report
            # manifest and summary prediction modes to exercise row-level checks.
            target = next((row for row in records if row["policy"] == "D2"), None)
            if target is not None and kwargs["control_prediction_modes"] == {
                    "D1": "current", "D2": "multi_model_v1"}:
                if row_provenance_tamper == "missing":
                    target.pop("prediction_mode")
                else:
                    target["prediction_contract"] = prediction_contract("cv")
        report_categories = list(kwargs["categories"])
        report_seeds = list(kwargs["episode_seeds"])
        if row_provenance_tamper and row_provenance_tamper.startswith("categories_"):
            if row_provenance_tamper == "categories_missing":
                report_categories = None
            elif row_provenance_tamper == "categories_empty":
                report_categories = []
            else:
                report_categories = report_categories[:-1]
        if row_provenance_tamper and row_provenance_tamper.startswith("seeds_"):
            if row_provenance_tamper == "seeds_missing":
                report_seeds = None
            elif row_provenance_tamper == "seeds_empty":
                report_seeds = []
            elif row_provenance_tamper == "seeds_bool_equal":
                report_seeds[0] = True
            elif row_provenance_tamper == "seeds_float_equal":
                report_seeds[1] = 27.0
            else:
                report_seeds = report_seeds[:-1]
        device_info = {"device": "cpu", "accelerator_backend": "cpu"}
        return {"evaluation_id": f"eval-{len(calls)}", "protocol_id": f"proto-{len(calls)}",
            "qualification_protocol_id": None, "source_identity_sha256": source_hash,
            "execution_device": device_info,
            "component_timing": {"provenance": device_info},
            "policy_manifest": manifest, "shared_training_configuration": shared_config,
            "phase4_factor": kwargs["phase4_factor"],
            "prediction": {"control_modes": modes},
            "candidate_k": kwargs["candidate_k_override"],
            "inference_disable_fault": None, "categories": report_categories,
            "seeds": report_seeds, "records": records}

    from pathlib import Path
    monkeypatch.setattr(track_training, "evaluate_meco", fake_evaluate)
    checkpoints = {}
    for track in MECO_TRACKS:
        path = tmp_path / f"{track}.pt"
        path.write_bytes(track.encode())
        checkpoints[track] = str(path)
    if row_provenance_tamper is None:
        report = matrix.run_matrix(checkpoints=checkpoints, scenario_id="pickup_simple",
            categories=("single", "cluster"), seeds=(1, 27), device="cpu",
            output_dir=tmp_path / "matrix")
    else:
        with pytest.raises(ValueError, match="evaluation report|row prediction provenance"):
            matrix.run_matrix(checkpoints=checkpoints, scenario_id="pickup_simple",
                categories=("single", "cluster"), seeds=(1, 27), device="cpu",
                output_dir=tmp_path / "matrix")
        return

    assert any(call["control_prediction_modes"] == {
        "D1": "current", "D2": "multi_model_v1"} for call in calls)
    condition = report["conditions"]["controls_d1_current_d2_multimodel_v1_k16"]
    assert condition["policy_manifest"]["D1"]["prediction_mode"] == "current"
    assert condition["policy_manifest"]["D2"]["prediction_mode"] == "multi_model_v1"
    contrast = next(row for row in report["contrasts"]
        if row["contrast"] == "D1_vs_D2_multi_model_v1")
    assert contrast["condition_id"] == condition["condition_id"]
    assert contrast["report_provenance"]["policy_manifest"]["D2"][
        "prediction_contract"] == prediction_contract("multi_model_v1")
    paired_row = contrast["paired_episode_rows"][0]
    assert paired_row["candidate_prediction_provenance"] == {
        "prediction_mode": "current",
        "prediction_contract": prediction_contract("current")}
    assert paired_row["reference_prediction_provenance"] == {
        "prediction_mode": "multi_model_v1",
        "prediction_contract": prediction_contract("multi_model_v1")}
