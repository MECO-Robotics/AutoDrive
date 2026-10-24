"""Deterministic, sensor-driven opponent policies for Phase 4 evaluation.

This module only produces ordinary planar velocity commands. The caller owns
legal spawn placement, chassis limits, collision handling, and simulator step.
It deliberately accepts sensed positions and velocities rather than simulator
truth state.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import torch


POLICY_POOL = (
    "stationary_blockage", "direct_pursuit", "trench_interceptor",
    "telegraph_reversal", "reactive_defender_a", "reactive_defender_b",
    "orbit_to_pursuit_switch",
)
POLICY_FACTOR_IDS = {
    "stationary_blockage": "blockage.stationary",
    "direct_pursuit": "opponent.direct_pursuit",
    "trench_interceptor": "opponent.trench_interception",
    "telegraph_reversal": "opponent.feint_reversal",
    "reactive_defender_a": "defender.reactive_pool",
    "reactive_defender_b": "defender.reactive_pool",
    "orbit_to_pursuit_switch": "opponent.abrupt_policy_change",
}
POLICY_CONFIG = {
    "schema": "meco-phase4-opponent-pool-v1",
    "command": "bounded observed-state planar velocity; simulator applies chassis physics",
    "max_command_mps": 1.5,
    "intercept_horizon_s": 0.8,
    "orbit_radius_m": 0.8,
    "orbit_angular_rate_rad_s": 1.2,
    "feint_distance_m": 1.2,
    "switch_tick": 360,
    "tick_semantics": "20ms physics ticks from scenario reset",
    "policies": list(POLICY_POOL),
    "factor_ids": POLICY_FACTOR_IDS,
    "sensing_boundary": "policy inputs may contain only controller-visible sensed state",
}


def policy_pool_identity() -> dict[str, str]:
    raw = json.dumps(POLICY_CONFIG, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return {"schema": POLICY_CONFIG["schema"],
            "config_sha256": hashlib.sha256(raw).hexdigest()}


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _bounded(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    magnitude = vector.norm(dim=-1, keepdim=True)
    maximum = maximum[..., None] if maximum.ndim else maximum
    return vector * torch.clamp(maximum / magnitude.clamp_min(1e-6), max=1.0)


def _read(state: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        value = state[key]
    except KeyError as exc:
        raise ValueError(f"Phase 4 opponent state is missing {key!r}") from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Phase 4 opponent state {key!r} must be a tensor")
    return value


def opponent_velocity_command(policy_id: str, sensed_state: Mapping[str, torch.Tensor],
                              *, tick: int | torch.Tensor,
                              max_speed_mps: float | torch.Tensor = 1.5
                              ) -> torch.Tensor:
    """Return ``[...,2]`` legal planar velocity requests from sensed state.

    Required state keys: ``own_xy``, ``target_xy``, ``target_velocity_xy``,
    ``target_valid``, ``anchor_xy``, and ``objective_xy``. Optional ``phase``
    defaults to zero. Invalid/missing target tracks make reactive policies
    hold at zero; they never fall back to hidden simulator truth.
    """
    if policy_id not in POLICY_POOL:
        raise ValueError(f"unknown Phase 4 opponent policy: {policy_id}")
    own = _read(sensed_state, "own_xy")
    target = _read(sensed_state, "target_xy")
    velocity = _read(sensed_state, "target_velocity_xy")
    valid = _read(sensed_state, "target_valid").bool()
    anchor = _read(sensed_state, "anchor_xy")
    objective = _read(sensed_state, "objective_xy")
    if own.shape[-1:] != (2,) or target.shape != own.shape or velocity.shape != own.shape:
        raise ValueError("own/target positions and sensed target velocity must share [...,2] shape")
    if anchor.shape != own.shape or objective.shape != own.shape or valid.shape != own.shape[:-1]:
        raise ValueError("anchor/objective/valid tensors must match the policy batch shape")
    speed = torch.as_tensor(max_speed_mps, dtype=own.dtype, device=own.device)
    if speed.ndim and speed.shape != valid.shape:
        raise ValueError("max_speed_mps must be scalar or match the policy batch shape")
    phase = sensed_state.get("phase")
    if phase is None:
        phase = torch.zeros_like(valid, dtype=own.dtype)
    else:
        phase = torch.as_tensor(phase, dtype=own.dtype, device=own.device)
    if policy_id == "stationary_blockage":
        command = torch.zeros_like(own)
    elif policy_id in ("direct_pursuit", "reactive_defender_b"):
        command = target - own
    elif policy_id == "trench_interceptor":
        # Lead the observed target toward the objective and meet it at the
        # closest point along that sensed route.
        route = objective - target
        direction = _unit(route)
        horizon = torch.as_tensor(POLICY_CONFIG["intercept_horizon_s"],
                                  dtype=own.dtype, device=own.device)
        predicted = target + velocity * horizon
        along = ((predicted - target) * direction).sum(-1, keepdim=True).clamp_min(0.)
        along = torch.minimum(along, route.norm(dim=-1, keepdim=True))
        command = target + direction * along - own
    elif policy_id == "telegraph_reversal":
        sign = torch.where(torch.as_tensor(tick, device=own.device) >= POLICY_CONFIG["switch_tick"],
                           -1., 1.).to(dtype=own.dtype)
        route = _unit(objective - anchor)
        perpendicular = torch.stack((-route[..., 1], route[..., 0]), dim=-1)
        command = anchor + perpendicular * (sign * POLICY_CONFIG["feint_distance_m"]) - own
    else:
        # Reactive A is a deterministic orbit around its legal anchor. The
        # switch policy uses that same behavior until the frozen switch tick.
        orbit_phase = phase * POLICY_CONFIG["orbit_angular_rate_rad_s"]
        orbit_target = anchor + POLICY_CONFIG["orbit_radius_m"] * torch.stack(
            (torch.cos(orbit_phase), torch.sin(orbit_phase)), dim=-1)
        if policy_id == "orbit_to_pursuit_switch":
            pursuit = target - own
            use_pursuit = torch.as_tensor(tick, device=own.device) >= POLICY_CONFIG["switch_tick"]
            command = torch.where(use_pursuit[..., None], pursuit, orbit_target - own)
        else:
            command = orbit_target - own
    if policy_id not in ("stationary_blockage", "telegraph_reversal"):
        command = torch.where(valid[..., None], command, torch.zeros_like(command))
    return _bounded(command, speed.clamp_min(0.))
