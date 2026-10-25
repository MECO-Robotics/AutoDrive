"""Pure metadata contract for MECO's opponent prediction modes."""
from __future__ import annotations

import math


PREDICTION_MODES = ("current", "cv", "multi_model_v1")
PREDICTION_MODEL_ID = "meco_opponent_forecast_v1"
PREDICTION_MAX_SPEED_MPS = 3.0
PREDICTION_MAX_AGE_S = 0.5
PREDICTION_DEFAULT_HORIZON_S = 1.0
PREDICTION_THREAT_RADIUS_M = 2.0
PREDICTION_MAX_PENALTY_POINTS = 0.75
MULTIMODEL_ID = "meco_opponent_kinematics_mm_v1"
MULTIMODEL_MAX_HORIZON_S = 2.5
MULTIMODEL_TURN_RATE_MIN_RAD_S = 0.6
MULTIMODEL_TURN_RATE_MAX_RAD_S = 1.5
MULTIMODEL_DAMPING_PER_S = 0.8
MULTIMODEL_HYPOTHESES = (
    ("constant_velocity", 0.40),
    ("turn_left", 0.15),
    ("turn_right", 0.15),
    ("damped_velocity", 0.30),
)


def prediction_contract(mode: str, *, horizon_s: float | None = None) -> dict:
    """Describe forecast semantics in immutable training/evaluation provenance."""
    if mode not in PREDICTION_MODES:
        raise ValueError(f"prediction mode must be one of {PREDICTION_MODES}")
    horizon = (0.0 if mode == "current" else
               None if horizon_s is None else float(horizon_s))
    if horizon is not None and (not math.isfinite(horizon) or horizon < 0.):
        raise ValueError("prediction horizon must be finite and nonnegative")
    contract = {"model": MULTIMODEL_ID if mode == "multi_model_v1" else PREDICTION_MODEL_ID,
        "mode": mode,
        "horizon_s": horizon,
        "horizon_semantics": ("observed_current_pose" if mode == "current" else
            "caller_supplied_eta_or_fixed_horizon" if horizon is None else "fixed_or_caller_supplied"),
        "max_speed_mps": PREDICTION_MAX_SPEED_MPS if mode == "cv" else None,
        "max_track_age_s": PREDICTION_MAX_AGE_S if mode == "cv" else None,
        "threat_radius_m": PREDICTION_THREAT_RADIUS_M,
        "max_penalty_points": PREDICTION_MAX_PENALTY_POINTS,
        "fallback": ("none_if_observed_pose_missing_or_nonfinite" if mode == "current"
            else "current_finite_observed_pose_when_cv_unavailable; otherwise_none")}
    if mode == "multi_model_v1":
        contract.update({
            "horizon_semantics": "candidate_route_eta_clamped_to_max_horizon",
            "max_horizon_s": MULTIMODEL_MAX_HORIZON_S,
            "max_speed_mps": PREDICTION_MAX_SPEED_MPS,
            "max_track_age_s": PREDICTION_MAX_AGE_S,
            "hypotheses": [{"id": name, "weight": weight}
                           for name, weight in MULTIMODEL_HYPOTHESES],
            "turn_rate_min_rad_s": MULTIMODEL_TURN_RATE_MIN_RAD_S,
            "turn_rate_max_rad_s": MULTIMODEL_TURN_RATE_MAX_RAD_S,
            "damping_per_s": MULTIMODEL_DAMPING_PER_S,
            "fallback": "finite_observed_pose_if_track_missing_stale_or_nonfinite",
            "risk_semantics": "weighted_expected_penalty_over_hypothesis_endpoints",
        })
    return contract
