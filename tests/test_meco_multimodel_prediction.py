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


def test_multimodel_contract_is_explicit_and_versioned():
    assert "multi_model_v1" in PREDICTION_MODES
    contract = prediction_contract("multi_model_v1", horizon_s=5.)
    assert contract["model"] == MULTIMODEL_ID == "meco_opponent_kinematics_mm_v1"
    assert contract["mode"] == "multi_model_v1"
    assert contract["max_horizon_s"] == 2.5
    assert contract["horizon_semantics"] == "candidate_route_eta_clamped_to_max_horizon"
    assert [(row["id"], row["weight"]) for row in contract["hypotheses"]] == list(
        MULTIMODEL_HYPOTHESES)
    assert math.isclose(sum(row["weight"] for row in contract["hypotheses"]), 1.)


def test_multimodel_forecast_is_bounded_deterministic_and_risk_uses_mixture():
    kwargs = dict(mode="multi_model_v1", valid=True, position_xy=(1., 2.),
                  velocity_xy=(8., 0.), age_s=.1, horizon_s=4.,
                  angular_velocity_rad_s=.2)
    first = opponent_forecast(**kwargs)
    assert first == opponent_forecast(**kwargs)
    assert first["model"] == MULTIMODEL_ID
    assert first["horizon_s"] == 2.5
    assert first["requested_horizon_s"] == 4.
    assert first["horizon_clamped"] is True
    assert first["used_multimodel"] is True
    assert len(first["hypotheses"]) == 4
    assert all(math.dist((1., 2.), row["xy"]) <= 7.5000001
               for row in first["hypotheses"])
    expected = sum(weight * .75 * max(0., min(1.,
        (2. - math.dist((4., 2.), row["xy"])) / 2.))
        for (_, weight), row in zip(MULTIMODEL_HYPOTHESES, first["hypotheses"]))
    assert opponent_risk_penalty((4., 2.), first) == pytest.approx(expected)


@pytest.mark.parametrize("kwargs", [
    dict(valid=True, position_xy=(1., 2.), velocity_xy=(1., 0.), age_s=.6),
    dict(valid=True, position_xy=(1., 2.), velocity_xy=(float("nan"), 0.), age_s=.1),
])
def test_multimodel_stale_or_nonfinite_motion_falls_back_to_finite_pose(kwargs):
    result = opponent_forecast(mode="multi_model_v1", horizon_s=4., **kwargs)
    assert result["xy"] == (1., 2.)
    assert result["fallback"] == "observed_current_pose"
    assert result["used_multimodel"] is False
    assert result["horizon_s"] == 2.5
    assert result["requested_horizon_s"] == 4.
    assert result["horizon_clamped"] is True
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


def test_plan_a_tensor_node_risk_matches_scalar_mixture_per_eta():
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


def test_plan_a_stale_multimodel_track_falls_back_to_observed_pose_and_masks_nodes():
    from frc_defense.meco_track_training import _plan_a_prediction_risk

    goals = torch.tensor(((0., 0.), (5., 0.)))
    support = SimpleNamespace(goal_xy=goals,
        valid_mask=torch.tensor([True] * 8 + [False] * 8))
    env = SimpleNamespace(
        opponent_valid=torch.tensor([[False, False, False, True]]),
        opponent_pose=torch.tensor([[[0., 0., 0.], [0., 0., 0.],
                                    [0., 0., 0.], [0., 0., 0.]]]),
        opponent_velocity=torch.tensor([[[0., 0., 0.], [0., 0., 0.],
                                         [0., 0., 0.], [1., 0., 0.]]]),
        opponent_age=torch.tensor([[0., 0., 0., .6]]))
    risk, metadata = _plan_a_prediction_risk(env, support,
        torch.tensor((.5, float("nan"))), "multi_model_v1")
    assert risk.tolist() == pytest.approx((.75, 0.))
    assert metadata["fallback"] == "observed_current_pose"
    assert metadata["used_multimodel"] is False


def test_candidate_opponent_batch_seam_passes_heading_rate_to_canonical_predictor():
    from frc_defense.meco_candidates import _opponent_prediction

    env = SimpleNamespace(
        opponent_valid=torch.tensor([[False, False, False, True]]),
        opponent_age=torch.tensor([[0., 0., 0., .1]]),
        opponent_pose=torch.tensor([[[0., 0., 0.], [0., 0., 0.],
                                    [0., 0., 0.], [1., 2., 0.]]]),
        opponent_velocity=torch.tensor([[[0., 0., 0.], [0., 0., 0.],
                                         [0., 0., 0.], [2., .5, .8]]]))
    result = _opponent_prediction(env, horizon=1.4, mode="multi_model_v1")
    expected = opponent_forecast(mode="multi_model_v1", valid=True,
        position_xy=(1., 2.), velocity_xy=(2., .5), age_s=.1,
        horizon_s=1.4, angular_velocity_rad_s=.8)
    assert result["mode"] == expected["mode"]
    assert result["hypotheses"] and len(result["hypotheses"]) == 4
    assert result["xy"] == pytest.approx(expected["xy"], abs=1e-6)
    for actual_hypothesis, expected_hypothesis in zip(
            result["hypotheses"], expected["hypotheses"]):
        assert actual_hypothesis["id"] == expected_hypothesis["id"]
        assert actual_hypothesis["xy"] == pytest.approx(expected_hypothesis["xy"], abs=1e-6)
    assert opponent_risk_penalty((4., 2.), result) is not None


def test_explicit_paired_matrix_condition_and_evaluator_override_preserve_defaults():
    from frc_defense.meco_ablation_matrix import MatrixCondition, default_conditions
    from frc_defense.meco_track_training import _parser, evaluate_meco

    conditions = {row.condition_name: row for row in default_conditions("pickup_simple")}
    default = conditions["controls_default_d1_current_d2_cv_k16"]
    multimodel = conditions["controls_d1_current_d2_multimodel_v1_k16"]
    assert default.control_prediction_mode is None
    assert default.control_prediction_modes is None
    assert multimodel.control_prediction_mode is None
    assert multimodel.control_prediction_modes == {
        "D1": "current", "D2": "multi_model_v1"}
    multimodel.validate()
    with pytest.raises(ValueError, match="cannot be combined"):
        MatrixCondition("invalid", "cv", "cv", 16,
            control_prediction_modes={"D2": "multi_model_v1"}).validate()

    args = _parser().parse_args(["evaluate", "--device", "cuda:0", "--output", "report.json",
        "--control-prediction-modes", '{"D1":"current","D2":"multi_model_v1"}',
        "--a1-checkpoint", "a1.pt", "--a2-checkpoint", "a2.pt",
        "--a3-checkpoint", "a3.pt", "--c1-checkpoint", "c1.pt", "--c2-checkpoint", "c2.pt"])
    assert args.control_prediction_modes == multimodel.control_prediction_modes
    with pytest.raises(ValueError, match="must map D1 and D2"):
        evaluate_meco({}, scenario_id="pickup_simple", categories=("single",),
            episodes=1, master_seed=1, device="cpu",
            control_prediction_modes={"D3": "multi_model_v1"})
