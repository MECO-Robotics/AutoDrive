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

# Frozen category domain from the registry's exact route-segment selector.
TRENCH_INTERCEPTION_CATEGORIES = tuple(
    f"{activity}/trench/{pressure}" for activity in ("active", "inactive")
    for pressure in (
        "level_0", "level_1_stationary", "level_2_patrol",
        "level_3_seeded_choice", "level_3_trench", "level_4_reactive",
        "level_5_mixed"))


def opponent_policy_runtime_domain_supported(scenario_id: str, category: str) -> bool:
    if scenario_id not in ("offense_vs_scripted_pressure", "full_match_offense"):
        return False
    for part in category.split("/"):
        if part.startswith("level_"):
            try:
                return int(part.split("_")[1]) in (1, 2, 3, 4, 5)
            except ValueError:
                return False
    return False


def trench_interception_runtime_domain_supported(scenario_id: str,
                                                  category: str) -> bool:
    """Whether a category is in the frozen trench-interception domain."""
    return (scenario_id == "offense_vs_scripted_pressure" and
            category in TRENCH_INTERCEPTION_CATEGORIES)


def opponent_policy_runtime_slots(category: str, *, allow_level_zero: bool = False) -> tuple[int, ...]:
    """Configured pressure slots: levels 1–4 use slot 0; level 5 uses 0/1."""
    level = next((part for part in category.split("/") if part.startswith("level_")), None)
    digits = None if level is None else level.removeprefix("level_").split("_", 1)[0]
    if digits is None or not digits.isdigit():
        raise ValueError("opponent slot scope requires a registered pressure category")
    value = int(digits)
    if value == 0 and allow_level_zero:
        return ()
    if value in (1, 2, 3, 4):
        return (0,)
    if value == 5:
        return (0, 1)
    raise ValueError("opponent slot scope supports pressure levels 1 through 5 only")


def policy_pool_identity() -> dict[str, str]:
    raw = json.dumps(POLICY_CONFIG, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return {"schema": POLICY_CONFIG["schema"],
            "config_sha256": hashlib.sha256(raw).hexdigest()}


def policy_assignment_identity(*, policy_id: str, scenario_id: str,
                               category: str, qualification_case: Mapping[str, str],
                               seed: int, condition: str,
                               factor_id: str = "defender.reactive_pool") -> dict[str, str]:
    """Bind one scheduled policy episode to its validation case and seed.

    This is a scheduling identity only. It does not claim that the simulator
    applied the policy or produced family-specific outcome signals.
    """
    if policy_id not in POLICY_POOL:
        raise ValueError(f"unknown Phase 4 opponent policy: {policy_id}")
    if (not isinstance(scenario_id, str) or not scenario_id or
            not isinstance(category, str) or not category):
        raise ValueError("opponent assignment requires scenario and category IDs")
    case_fields = ("layout_id", "defender_family", "start_condition", "sensor_profile")
    if (not isinstance(qualification_case, Mapping) or
            set(qualification_case) != set(case_fields) or
            any(not isinstance(qualification_case[key], str) or
                not qualification_case[key] for key in case_fields)):
        raise ValueError("opponent assignment requires a complete validation case")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("opponent assignment seed must be a signed-63-bit non-negative integer")
    if condition not in ("no_factor", "factor_treatment"):
        raise ValueError("opponent assignment condition is invalid")
    if not isinstance(factor_id, str) or not factor_id:
        raise ValueError("opponent assignment factor ID must be nonempty")
    pool_identity = policy_pool_identity()
    evaluation_case = {"scenario_id": scenario_id, "category": category,
                       "qualification_case": dict(qualification_case),
                       "split": "validation", "seed": seed,
                       "condition": condition, "policy_id": policy_id}
    stimulus = {"assignment_schema": "meco-phase4-opponent-assignment-v1",
                "evaluation_case": evaluation_case,
                "factor_id": factor_id,
                "policy_config_sha256": pool_identity["config_sha256"],
                "sensing_boundary": POLICY_CONFIG["sensing_boundary"]}
    stimulus_sha256 = hashlib.sha256(json.dumps(stimulus, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"policy_id": policy_id,
            "policy_family": POLICY_FACTOR_IDS[policy_id],
            "policy_config_sha256": pool_identity["config_sha256"],
            "stimulus_identity_sha256": stimulus_sha256}


def compile_opponent_policy_schedule(*, policy_id: str, scenario_id: str,
        categories: list[str] | tuple[str, ...], qualification_case: Mapping[str, str],
        seeds: list[int] | tuple[int, ...], split: str,
        factor_id: str = "defender.reactive_pool",
        protocol_id: str | None = None) -> dict:
    """Build the evaluator input binding for one exact validation policy run."""
    if split != "validation":
        raise ValueError("Phase 4 opponent policy schedules are validation-only")
    if factor_id == "opponent.trench_interception":
        if (policy_id != "trench_interceptor" or
                scenario_id != "offense_vs_scripted_pressure" or
                not isinstance(protocol_id, str) or not protocol_id or
                not categories or
                any(category not in TRENCH_INTERCEPTION_CATEGORIES
                    for category in categories)):
            raise ValueError("trench interception requires a category subset of its exact 14-category validation domain, scenario, factor, protocol, and policy")
    if factor_id == "opponent.direct_pursuit":
        if (policy_id != "direct_pursuit" or
                scenario_id != "offense_vs_scripted_pressure" or
                not isinstance(protocol_id, str) or not protocol_id or
                not categories or any(not opponent_policy_runtime_domain_supported(
                    scenario_id, category) for category in categories)):
            raise ValueError("direct pursuit requires its exact pressure-category validation domain, scenario, factor, protocol, and policy")
    if (not isinstance(categories, (list, tuple)) or not categories or
            any(not isinstance(category, str) or not category for category in categories) or
            len(set(categories)) != len(categories)):
        raise ValueError("opponent schedule requires unique nonempty categories")
    if (not isinstance(seeds, (list, tuple)) or not seeds or
            any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("opponent schedule requires unique signed-63-bit validation seeds")
    # Reuse the exact case validation and policy/domain checks for every row.
    identities = [policy_assignment_identity(policy_id=policy_id,
        scenario_id=scenario_id, category=category,
        qualification_case=qualification_case, seed=seed,
        condition="factor_treatment", factor_id=factor_id)
        for category in categories for seed in seeds]
    payload = {"schema": "meco-phase4-opponent-evaluator-schedule-v1",
        "policy_id": policy_id, "scenario_id": scenario_id,
        "categories": list(categories), "qualification_case": dict(qualification_case),
        "factor_id": factor_id,
        "protocol_id": protocol_id,
        "category_domain": (list(TRENCH_INTERCEPTION_CATEGORIES)
            if factor_id == "opponent.trench_interception" else
            list(categories) if factor_id == "opponent.direct_pursuit" else None),
        "policy_family": POLICY_FACTOR_IDS[policy_id],
        "configured_pressure_slots": {category: list(opponent_policy_runtime_slots(
                                      category, allow_level_zero=(
                                          factor_id == "opponent.trench_interception")))
                                      for category in categories},
        "split": split, "seed_list": list(seeds),
        "policy_config_sha256": policy_pool_identity()["config_sha256"],
        "per_episode_assignment_sha256": hashlib.sha256(json.dumps(identities,
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "policy_inputs": POLICY_CONFIG["sensing_boundary"],
        "runtime_status": "scheduled_not_simulator_validated",
        "executable": True, "qualification_eligible": False}
    payload["schedule_sha256"] = hashlib.sha256(json.dumps(payload,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return payload


def validate_opponent_policy_schedule(schedule: dict, *, scenario_id: str,
        categories: list[str] | tuple[str, ...], qualification_case: Mapping[str, str],
        seeds: list[int] | tuple[int, ...], split: str,
        factor_id: str = "defender.reactive_pool",
        protocol_id: str | None = None) -> dict:
    if not isinstance(schedule, dict) or schedule.get("schema") != "meco-phase4-opponent-evaluator-schedule-v1":
        raise ValueError("invalid Phase 4 opponent evaluator schedule")
    expected = compile_opponent_policy_schedule(policy_id=schedule.get("policy_id"),
        scenario_id=scenario_id, categories=categories,
        qualification_case=qualification_case, seeds=seeds, split=split,
        factor_id=factor_id, protocol_id=protocol_id)
    if schedule != expected:
        raise ValueError("opponent evaluator schedule differs from exact protocol case/seed/config")
    return expected


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
    raw_valid = _read(sensed_state, "target_valid")
    valid_metadata = (torch.isfinite(raw_valid) &
                      ((raw_valid == 0) | (raw_valid == 1)))
    valid = raw_valid.bool()
    anchor = _read(sensed_state, "anchor_xy")
    objective = _read(sensed_state, "objective_xy")
    if own.shape[-1:] != (2,) or target.shape != own.shape or velocity.shape != own.shape:
        raise ValueError("own/target positions and sensed target velocity must share [...,2] shape")
    if (anchor.shape != own.shape or objective.shape != own.shape or
            valid.shape != own.shape[:-1]):
        raise ValueError("anchor/objective/valid tensors must match the policy batch shape")
    # Sanitize before arithmetic so a corrupt sensor/metadata value cannot
    # escape as a NaN command. Keep validity masks on-device; bad rows fail
    # closed to a zero velocity request without a host synchronization.
    finite_state = (torch.isfinite(own).all(-1) & torch.isfinite(target).all(-1) &
                    torch.isfinite(velocity).all(-1) & torch.isfinite(anchor).all(-1) &
                    torch.isfinite(objective).all(-1) & valid_metadata)
    own = torch.nan_to_num(own, nan=0., posinf=0., neginf=0.)
    target = torch.nan_to_num(target, nan=0., posinf=0., neginf=0.)
    velocity = torch.nan_to_num(velocity, nan=0., posinf=0., neginf=0.)
    anchor = torch.nan_to_num(anchor, nan=0., posinf=0., neginf=0.)
    objective = torch.nan_to_num(objective, nan=0., posinf=0., neginf=0.)
    speed = torch.as_tensor(max_speed_mps, dtype=own.dtype, device=own.device)
    if speed.ndim and speed.shape != valid.shape:
        raise ValueError("max_speed_mps must be scalar or match the policy batch shape")
    finite_speed = torch.isfinite(speed)
    speed = torch.nan_to_num(speed, nan=0., posinf=0., neginf=0.)
    phase = sensed_state.get("phase")
    if phase is None:
        phase = torch.zeros_like(valid, dtype=own.dtype)
    else:
        phase = torch.as_tensor(phase, dtype=own.dtype, device=own.device)
        if phase.shape != valid.shape:
            raise ValueError("phase must be scalar or match the policy batch shape")
    finite_phase = torch.isfinite(phase)
    phase = torch.nan_to_num(phase, nan=0., posinf=0., neginf=0.)
    tick_tensor = torch.as_tensor(tick, device=own.device)
    finite_tick = torch.isfinite(tick_tensor) if tick_tensor.is_floating_point() else True
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
        sign = torch.where(tick_tensor >= POLICY_CONFIG["switch_tick"],
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
            use_pursuit = tick_tensor >= POLICY_CONFIG["switch_tick"]
            command = torch.where(use_pursuit[..., None], pursuit, orbit_target - own)
        else:
            command = orbit_target - own
    if policy_id not in ("stationary_blockage", "telegraph_reversal"):
        command = torch.where(valid[..., None], command, torch.zeros_like(command))
    command = _bounded(command, speed.clamp_min(0.))
    safe_row = finite_state & finite_phase
    if isinstance(finite_tick, torch.Tensor):
        safe_row = safe_row & finite_tick
    if isinstance(finite_speed, torch.Tensor) and finite_speed.ndim:
        safe_row = safe_row & finite_speed
    elif isinstance(finite_speed, torch.Tensor):
        safe_row = safe_row & finite_speed
    return torch.where(safe_row[..., None], command, torch.zeros_like(command))
