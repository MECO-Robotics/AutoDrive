import math

import pytest

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
    result = opponent_forecast(mode="multi_model_v1", horizon_s=1., **kwargs)
    assert result["xy"] == (1., 2.)
    assert result["fallback"] == "observed_current_pose"
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
