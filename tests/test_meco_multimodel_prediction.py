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
    from frc_defense.meco_ablation_matrix import MatrixCondition
    from frc_defense.meco_track_training import _parser, _prediction_record_fields

    condition = MatrixCondition("matched_mm", "multi_model_v1", "multi_model_v1", 16)
    condition.validate()
    args = _parser().parse_args(["evaluate", "--device", "cuda:0", "--output", "report.json",
        "--prediction-mode", "multi_model_v1", "--control-prediction-mode", "multi_model_v1",
        "--a1-checkpoint", "a1.pt", "--a2-checkpoint", "a2.pt",
        "--a3-checkpoint", "a3.pt", "--c1-checkpoint", "c1.pt", "--c2-checkpoint", "c2.pt"])
    assert args.prediction_mode == args.control_prediction_mode == "multi_model_v1"
    row_provenance = _prediction_record_fields(args.control_prediction_mode)
    assert row_provenance["prediction_mode"] == "multi_model_v1"
    assert row_provenance["prediction_contract"] == prediction_contract("multi_model_v1")
