"""Phase 4 runtime context fixtures, bound to the design registry axes.

The helpers are opt-in and build checked reset poses, paired teammate action
assignments, policy-consistent pose perturbations, and physical contact/bump
approaches. Simulator truth is only changed by explicit reset helpers.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

import torch


CONTEXT_FACTOR_VERSION = 1
_FACTOR_AXES = {
    "legal_team_pose_sets": ("start.legal_team_poses", {
        "team_start_set_a", "team_start_set_b", "team_start_set_c"}),
    "teammate_behaviors": ("teammate.behavior_pool", {
        "collect", "score", "ferry", "defend", "hold_position"}),
    "robot_contact_geometry": ("contact.robot_bump", {
        "crossing_contact", "side_contact", "recovery_contact"}),
    "focal_pose_offsets": ("observation.focal_pose_offset", {
        "translation_small", "translation_large", "heading_offset"}),
    "truth_pose_boundary": ("observation.focal_pose_offset", {
        "simulator_truth_only_not_policy_input"}),
}

TEAMMATE_BEHAVIORS = ("collect", "score", "ferry", "defend", "hold_position")
START_PROFILE_NAMES = ("midfield_spread", "crossing_lanes", "opposing_lane_stagger")
START_PROFILE_REGISTRY_IDS = {
    "midfield_spread": "team_start_set_a",
    "crossing_lanes": "team_start_set_b",
    "opposing_lane_stagger": "team_start_set_c",
}
CONTACT_GEOMETRIES = ("crossing_contact", "side_contact", "recovery_contact")
_CONTACT_APPROACHES = {
    "crossing_contact": {"opponent_heading_rad": math.pi/4,
                         "focal_heading_rad": -math.pi/4,
                         "exposure": "crossing_diagonal_axes"},
    "side_contact": {"opponent_heading_rad": math.pi/2,
                      "focal_heading_rad": 0.,
                      "exposure": "orthogonal_side_impact"},
    "recovery_contact": {"opponent_heading_rad": math.radians(25.),
                          "focal_heading_rad": 0.,
                          "exposure": "oblique_corner_impact"},
}


def validate_contact_geometry_split(geometry_id: str | None, split: str) -> None:
    """Contact geometry is diagnostic only and cannot enter held-out final test."""
    if geometry_id is not None and split == "test":
        raise ValueError("contact geometry evaluation is diagnostic only; final-test use is disabled")
FOCAL_POSE_OFFSETS = {
    "translation_small": {"offset_xy_m": (0.30, 0.0), "heading_offset_rad": 0.0},
    "translation_large": {"offset_xy_m": (0.60, 0.0), "heading_offset_rad": 0.0},
    "heading_offset": {"offset_xy_m": (0.0, 0.0), "heading_offset_rad": math.radians(10.)},
}
_TRUTH_POSE_BOUNDARY = {"value": "simulator_truth_only_not_policy_input"}

# Legal, robot-slot-ordered full-team reset states. Coordinates use REBUILT's
# 16.54 x 8.07 m field; they are scaled to a layout's dimensions at reset.
_START_LAYOUTS = {
    "midfield_spread": (
        (5.95, .95, 0.), (6.65, 4.035, 0.), (5.95, 7.12, 0.),
        (10.59, .95, math.pi), (9.89, 4.035, math.pi),
        (10.59, 7.12, math.pi)),
    "crossing_lanes": (
        (6.25, 1.25, .18), (6.65, 4.035, 0.), (6.25, 6.82, -.18),
        (10.29, 1.25, math.pi-.18), (9.89, 4.035, math.pi),
        (10.29, 6.82, math.pi+.18)),
    "opposing_lane_stagger": (
        (6.05, 1.30, .12), (6.60, 3.00, 0.), (7.15, 5.40, -.12),
        (10.49, 6.77, math.pi-.12), (9.94, 5.07, math.pi),
        (9.39, 2.67, math.pi+.12)),
}


def phase4_context_identity(axis_name: str, value_name: str) -> dict:
    """Return registry-axis identity plus canonical config digest."""
    if axis_name not in _FACTOR_AXES:
        raise ValueError(f"unregistered Phase 4 context axis: {axis_name}")
    factor_id, allowed = _FACTOR_AXES[axis_name]
    if value_name not in allowed:
        raise ValueError(f"unregistered value {value_name!r} for axis {axis_name}")
    config = _context_config(axis_name, value_name)
    payload = {"version": CONTEXT_FACTOR_VERSION, "factor_id": factor_id,
               "axis_name": axis_name, "value_name": value_name,
               "config": config}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {**payload, "config_sha256": digest,
            "factor_config_id": f"{factor_id}:{value_name}"}


def start_profile_identity(runtime_profile: str) -> dict:
    """Return the exact registry identity for a runtime profile name."""
    try:
        registry_id = START_PROFILE_REGISTRY_IDS[runtime_profile]
    except KeyError as exc:
        raise ValueError(f"unknown full-team start profile: {runtime_profile}") from exc
    return phase4_context_identity("legal_team_pose_sets", registry_id)


def start_profile_runtime_name(registry_profile_id: str) -> str:
    """Resolve a frozen legal-team-pose registry value to its runtime profile."""
    try:
        return next(name for name, registry_id in START_PROFILE_REGISTRY_IDS.items()
                    if registry_id == registry_profile_id)
    except StopIteration as exc:
        raise ValueError(f"unknown full-team start profile registry ID: {registry_profile_id}") from exc


def phase4_start_profile_assignment(registry_profile_id: str, *,
        scenario_id: str, split: str, protocol_id: str,
        protocol_sha256: str, qualification_case: dict,
        categories: tuple[str, ...] | list[str], seeds: tuple[int, ...] | list[int]) -> dict:
    """Bind one registered team pose set to an exact held-out protocol cell."""
    if start_profile_runtime_name(registry_profile_id) not in START_PROFILE_NAMES:
        raise ValueError("start profile is not registered")
    if scenario_id != "full_match_offense" or split != "validation":
        raise ValueError("legal full-team start profiles are restricted to full-match validation")
    if (not isinstance(protocol_id, str) or not protocol_id or
            not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in protocol_sha256)):
        raise ValueError("start profile requires frozen protocol identity")
    case_fields = {"layout_id", "defender_family", "start_condition", "sensor_profile"}
    if (not isinstance(qualification_case, dict) or set(qualification_case) != case_fields or
            any(not isinstance(value, str) or not value for value in qualification_case.values())):
        raise ValueError("start profile requires the exact frozen qualification case")
    if (not isinstance(categories, (tuple, list)) or not categories or
            any(not isinstance(category, str) or not category for category in categories) or
            len(set(categories)) != len(categories)):
        raise ValueError("start profile categories must be unique nonempty strings")
    if (not isinstance(seeds, (tuple, list)) or not seeds or
            any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("start profile seeds must be unique signed-63-bit integers")
    profile = phase4_context_identity("legal_team_pose_sets", registry_profile_id)
    seed_sha256 = hashlib.sha256(json.dumps(list(seeds), separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    payload = {"schema": "meco-phase4-start-profile-assignment-v1",
        "factor_id": "start.legal_team_poses",
        "registry_profile_id": registry_profile_id,
        "runtime_profile_name": start_profile_runtime_name(registry_profile_id),
        "profile_config_sha256": profile["config_sha256"],
        "scenario_id": scenario_id, "split": split,
        "protocol_id": protocol_id, "protocol_sha256": protocol_sha256,
        "qualification_case": dict(qualification_case),
        "categories": list(categories), "seed_count": len(seeds),
        "seed_assignment_sha256": seed_sha256,
        "legal_start_validation_rate": None,
        "outcome_by_start_state": None,
        "qualification_eligible": False}
    assignment_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {**payload, "assignment_sha256": assignment_sha256}


def validate_phase4_start_profile_assignment(assignment: dict) -> dict:
    """Validate the immutable assignment and its registry/config binding."""
    if not isinstance(assignment, dict):
        raise ValueError("start profile assignment must be an object")
    profile_id = assignment.get("registry_profile_id")
    runtime_name = start_profile_runtime_name(profile_id)
    profile = phase4_context_identity("legal_team_pose_sets", profile_id)
    if (assignment.get("schema") != "meco-phase4-start-profile-assignment-v1" or
            assignment.get("factor_id") != "start.legal_team_poses" or
            assignment.get("runtime_profile_name") != runtime_name or
            assignment.get("profile_config_sha256") != profile["config_sha256"] or
            assignment.get("scenario_id") != "full_match_offense" or
            assignment.get("split") != "validation" or
            assignment.get("qualification_eligible") is not False or
            assignment.get("legal_start_validation_rate") is not None or
            assignment.get("outcome_by_start_state") is not None):
        raise ValueError("start profile assignment differs from its frozen registry contract")
    digest = assignment.get("assignment_sha256")
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("start profile assignment digest is malformed")
    payload = {key: value for key, value in assignment.items()
               if key != "assignment_sha256"}
    expected_digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != expected_digest:
        raise ValueError("start profile assignment digest mismatch")
    if (not isinstance(assignment.get("protocol_id"), str) or
            not assignment["protocol_id"] or
            not isinstance(assignment.get("protocol_sha256"), str) or
            len(assignment["protocol_sha256"]) != 64 or
            any(char not in "0123456789abcdef"
                for char in assignment["protocol_sha256"]) or
            type(assignment.get("seed_count")) is not int or assignment["seed_count"] < 1 or
            not isinstance(assignment.get("seed_assignment_sha256"), str) or
            len(assignment["seed_assignment_sha256"]) != 64 or
            any(char not in "0123456789abcdef"
                for char in assignment["seed_assignment_sha256"]) or
            not isinstance(assignment.get("categories"), list) or
            not assignment["categories"] or
            any(not isinstance(category, str) or not category
                for category in assignment["categories"]) or
            len(set(assignment["categories"])) != len(assignment["categories"]) or
            not isinstance(assignment.get("qualification_case"), dict) or
            set(assignment["qualification_case"]) != {
                "layout_id", "defender_family", "start_condition", "sensor_profile"} or
            any(not isinstance(value, str) or not value
                for value in assignment["qualification_case"].values())):
        raise ValueError("start profile assignment has incomplete protocol binding")
    return dict(assignment)


def phase4_focal_pose_assignment(offset_name: str | None, *,
        paired_offset_name: str, scenario_id: str, split: str,
        protocol_id: str, protocol_sha256: str, qualification_case: dict,
        categories: tuple[str, ...] | list[str], seeds: tuple[int, ...] | list[int]) -> dict:
    """Bind a no-offset control or registered pose offset to one validation cell."""
    if scenario_id != "full_match_offense" or split != "validation":
        raise ValueError("focal pose offsets are restricted to full-match validation")
    pair_identity = phase4_context_identity("focal_pose_offsets", paired_offset_name)
    if offset_name is not None and offset_name != paired_offset_name:
        raise ValueError("focal pose arm must match its registered paired offset")
    if (not isinstance(protocol_id, str) or not protocol_id or
            not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in protocol_sha256)):
        raise ValueError("focal pose assignment requires frozen protocol identity")
    case_fields = {"layout_id", "defender_family", "start_condition", "sensor_profile"}
    if (not isinstance(qualification_case, dict) or set(qualification_case) != case_fields or
            any(not isinstance(v, str) or not v for v in qualification_case.values())):
        raise ValueError("focal pose assignment requires the exact frozen case")
    if (not isinstance(categories, (tuple, list)) or not categories or
            any(not isinstance(v, str) or not v for v in categories) or
            len(set(categories)) != len(categories)):
        raise ValueError("focal pose categories must be unique nonempty strings")
    if (not isinstance(seeds, (tuple, list)) or not seeds or
            any(type(v) is not int or v < 0 or v >= 2**63 for v in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("focal pose seeds must be unique signed-63-bit integers")
    payload = {"schema": "meco-phase4-focal-pose-assignment-v1",
        "factor_id": "observation.focal_pose_offset",
        "paired_offset_name": paired_offset_name,
        "paired_offset_config_sha256": pair_identity["config_sha256"],
        "arm": "no_offset" if offset_name is None else "offset",
        "requested_offset_name": offset_name,
        "requested_offset_config_sha256": (None if offset_name is None else pair_identity["config_sha256"]),
        "scenario_id": scenario_id, "split": split,
        "protocol_id": protocol_id, "protocol_sha256": protocol_sha256,
        "qualification_case": dict(qualification_case),
        "categories": list(categories), "seeds": list(seeds),
        "relocalization_time_s": None, "pose_error_recovery_rate": None,
        "safety_contact_rate": None, "qualification_eligible": False}
    pair_payload = {k: v for k, v in payload.items()
                    if k not in ("arm", "requested_offset_name", "requested_offset_config_sha256")}
    payload["pair_assignment_sha256"] = hashlib.sha256(json.dumps(
        pair_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    payload["assignment_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return payload


def validate_phase4_focal_pose_assignment(assignment: dict) -> dict:
    if not isinstance(assignment, dict):
        raise ValueError("focal pose assignment must be an object")
    expected = phase4_focal_pose_assignment(
        assignment.get("requested_offset_name"),
        paired_offset_name=assignment.get("paired_offset_name"),
        scenario_id=assignment.get("scenario_id"), split=assignment.get("split"),
        protocol_id=assignment.get("protocol_id"), protocol_sha256=assignment.get("protocol_sha256"),
        qualification_case=assignment.get("qualification_case"),
        categories=assignment.get("categories"), seeds=assignment.get("seeds"))
    if expected != assignment:
        raise ValueError("focal pose assignment differs from its frozen pair/protocol contract")
    return assignment


@torch.no_grad()
def apply_phase4_focal_pose_assignment(env, assignment: dict) -> dict:
    """Apply a registered estimate offset after reset without changing simulator truth."""
    validate_phase4_focal_pose_assignment(assignment)
    if not hasattr(env, "sim") or not hasattr(env, "estimated_pose") or not hasattr(env, "policy_pose"):
        raise ValueError("environment lacks physical and policy pose state")
    if (env.estimated_pose.data_ptr() == env.sim.pose.data_ptr() or
            env.policy_pose.data_ptr() == env.sim.pose.data_ptr()):
        raise ValueError("policy pose state aliases simulator truth")
    if (env.estimated_pose.shape != env.sim.pose.shape or
            env.policy_pose.shape != env.sim.pose.shape or env.sim.pose.ndim != 3 or
            env.sim.pose.shape[1] <= 3 or env.sim.pose.shape[2] != 3):
        raise ValueError("environment pose tensors do not satisfy the six-robot pose contract")
    truth = env.sim.pose[:, 3].clone()
    if assignment["requested_offset_name"] is not None:
        config = FOCAL_POSE_OFFSETS[assignment["requested_offset_name"]]
        dx, dy = config["offset_xy_m"]
        env.estimated_pose[:, 3, 0] += dx
        env.estimated_pose[:, 3, 1] += dy
        env.estimated_pose[:, 3, 2] += config["heading_offset_rad"]
        env.estimated_pose[:, 3, 2] = torch.atan2(
            torch.sin(env.estimated_pose[:, 3, 2]), torch.cos(env.estimated_pose[:, 3, 2]))
    env.policy_pose.copy_(env.estimated_pose)
    event = {"assignment_sha256": assignment["assignment_sha256"],
        "pair_assignment_sha256": assignment["pair_assignment_sha256"],
        "arm": assignment["arm"], "requested_offset_name": assignment["requested_offset_name"],
        "requested_offset_config_sha256": assignment["requested_offset_config_sha256"],
        "applied_before_policy_observation": True,
        "physical_truth_unchanged": bool(torch.equal(truth, env.sim.pose[:, 3])),
        "relocalization_time_s": None, "pose_error_recovery_rate": None,
        "safety_contact_rate": None, "qualification_eligible": False}
    env.phase4_focal_pose_runtime = event
    return event


@torch.no_grad()
def phase4_focal_pose_evaluator_diagnostics(env, runtime_event: dict) -> dict:
    """Create truth/estimate diagnostics for evaluator serialization only."""
    truth = env.sim.pose[:, 3].clone()
    estimate = env.policy_pose[:, 3].clone()
    error_xy = torch.linalg.vector_norm(estimate[:, :2] - truth[:, :2], dim=-1)
    error_heading = torch.atan2(torch.sin(estimate[:, 2] - truth[:, 2]),
                                torch.cos(estimate[:, 2] - truth[:, 2]))
    return {**runtime_event,
        "sim_truth_pose_m_rad": truth.detach().cpu().tolist(),
        "policy_pose_estimate_m_rad": estimate.detach().cpu().tolist(),
        "pose_error_xy_m": error_xy.detach().cpu().tolist(),
        "pose_error_heading_rad": error_heading.detach().cpu().tolist()}


def _context_config(axis_name: str, value_name: str) -> dict:
    if axis_name == "legal_team_pose_sets":
        runtime_name = next((name for name, registry_id in
                             START_PROFILE_REGISTRY_IDS.items()
                             if registry_id == value_name), None)
        if runtime_name is None:
            raise ValueError(f"registry ID has no runtime start profile: {value_name}")
        return {"robot_slot_poses_m_rad": [list(pose)
                for pose in _START_LAYOUTS[runtime_name]],
                "runtime_profile_name": runtime_name,
                "registry_profile_id": value_name,
                "field_reference_m": [16.54, 8.07],
                "legal_start_validation": "field_bounds_robot_clearance_solids_pairwise"}
    if axis_name == "teammate_behaviors":
        return {"behavior": value_name, "macro_action": {
            "collect": "nearest_legal_sensed_candidate_0_to_3",
            "score": 4, "ferry": 6, "defend": 5,
            "hold_position": 7}[value_name]}
    if axis_name == "robot_contact_geometry":
        return {"geometry": value_name, "robot_pair_slots": [2, 3],
                "geometry_contract": _CONTACT_APPROACHES[value_name],
                "contact_approach": {"opponent_slot": 2, "focal_slot": 3,
                    "opponent_action": 5, "focal_action": 7,
                    "target": "sensed_opponent_track",
                    "command_path": "bounded_native_motion_controller"},
                "contact_injection": False,
                "contact_metric": {"approach": "slot 2 action 5 targets a fresh valid sensed track verified as focal slot 3",
                    "contact_proxy": "distance-gated co-contact physics ticks from per-robot flags; not registry contact signal",
                    "recovery_proxy": "first stable separated low-speed interval after last proxy contact; not route-following recovery"},
                "physics": "ordinary"}
    if axis_name == "focal_pose_offsets":
        values = FOCAL_POSE_OFFSETS[value_name]
        return {"offset_xy_m": list(values["offset_xy_m"]),
                "heading_offset_rad": values["heading_offset_rad"],
                "dependent_channel_policy": "recompute_relative_or_invalidate"}
    return dict(_TRUTH_POSE_BOUNDARY)


def _dimensions(env) -> tuple[float, float]:
    return float(env.sim.field_length), float(env.sim.field_width)


def _robot_radii(env) -> torch.Tensor:
    # Conservative circumradius, valid for every chassis orientation.
    return .5 * torch.sqrt(env.sim.length.square() + env.sim.width.square())


def full_team_start_profile(env, profile: str) -> torch.Tensor:
    """Return a legal ``[world, 6, 3]`` start pose tensor without mutation."""
    if profile not in START_PROFILE_NAMES:
        raise ValueError(f"unknown full-team start profile: {profile}")
    length, width = _dimensions(env)
    coords = torch.tensor(_START_LAYOUTS[profile], device=env.device,
                          dtype=env.sim.pose.dtype)
    coords[:, 0] *= length / 16.54
    coords[:, 1] *= width / 8.07
    poses = coords.expand(env.n, -1, -1).clone()
    validate_full_team_start_profile(env, poses)
    return poses


def apply_full_team_start_profile(env, profile: str) -> torch.Tensor:
    """Apply a checked profile at reset and clear stale physical velocities."""
    poses = full_team_start_profile(env, profile)
    env.sim.pose.copy_(poses)
    env.sim.velocity.zero_()
    return poses


def validate_full_team_start_profile(env, poses: torch.Tensor) -> None:
    """Reject bounds, inter-robot envelope, and solid-field collisions."""
    if not isinstance(poses, torch.Tensor) or poses.shape != (env.n, 6, 3):
        raise ValueError("start poses must have shape [num_worlds, 6, 3]")
    if not torch.isfinite(poses).all():
        raise ValueError("start poses must be finite")
    length, width = _dimensions(env)
    radius = _robot_radii(env)
    xy = poses[..., :2]
    if ((xy[..., 0] < radius) | (xy[..., 0] > length-radius) |
            (xy[..., 1] < radius) | (xy[..., 1] > width-radius)).any():
        raise ValueError("start profile violates legal field bounds")
    distance = torch.linalg.vector_norm(xy[:, :, None, :] - xy[:, None, :, :], dim=-1)
    pair_radius = radius[:, :, None] + radius[:, None, :]
    pair_mask = ~torch.eye(6, device=env.device, dtype=torch.bool)
    if (distance[:, pair_mask] < pair_radius[:, pair_mask] + .02).any():
        raise ValueError("start profile has overlapping robot chassis envelopes")

    from .field import static_collision_boxes
    for box in static_collision_boxes(env.field_boxes):
        dx = (xy[..., 0] - float(box.x)).abs()
        dy = (xy[..., 1] - float(box.y)).abs()
        if ((dx <= float(box.length)/2 + radius) &
                (dy <= float(box.width)/2 + radius)).any():
            raise ValueError(f"start profile intersects solid field geometry: {box.name}")


def contact_exposure_profile(env, geometry: str) -> torch.Tensor:
    """Return a legal contact approach start for opponent slots 2 and 3.

    Slots 2 and 3 begin with chassis clearance, facing one another. Their
    deterministic defend actions below move toward the sensed opponent; any
    eventual chassis contact is generated and resolved by normal physics.
    """
    if geometry not in CONTACT_GEOMETRIES:
        raise ValueError(f"unknown contact geometry: {geometry}")
    length, width = _dimensions(env)
    radius = _robot_radii(env)
    separation = float((radius[:, 2] + radius[:, 3]).max().item()) + .08
    approach = _CONTACT_APPROACHES[geometry]
    opponent_heading = approach["opponent_heading_rad"]
    focal_heading = approach["focal_heading_rad"]
    cx, cy = length*.5, width*.5
    # Slot 2 advances along its heading toward focal slot 3.  This gives each
    # registered geometry a distinct pair pose/orientation while preserving
    # the same conservative chassis clearance.
    dx, dy = separation*math.cos(opponent_heading), separation*math.sin(opponent_heading)
    poses = torch.tensor((
        (length*.365, width*.15, opponent_heading),
        (length*.365, width*.85, opponent_heading),
        (cx-dx, cy-dy, opponent_heading),
        (cx, cy, focal_heading),
        (length*.635, width*.15, focal_heading),
        (length*.635, width*.85, focal_heading)),
        device=env.device, dtype=env.sim.pose.dtype)
    validate_full_team_start_profile(env, poses.expand(env.n, -1, -1))
    return poses.expand(env.n, -1, -1).clone()


def contact_exposure_actions(env, *, focal_slot: int = 3,
                             opponent_slot: int = 2) -> torch.Tensor:
    """Hold the focal robot while a scripted opponent approaches its track.

    Action 5 uses the existing opponent-track target and ordinary motion and
    collision code. Fail closed until observer slot 2 has a fresh valid track
    for focal slot 3 and its action mask allows the approach.
    """
    if (focal_slot, opponent_slot) != (3, 2):
        raise ValueError("registered contact geometry uses focal/opponent slots (3, 2)")
    if not all(hasattr(env, attr) for attr in
               ("opponent_valid", "opponent_age", "opponent_pose", "team_ids", "dt")):
        raise ValueError("contact approach requires sensed opponent validity, age, pose, team, and dt")
    if (env.opponent_valid.shape[1] <= opponent_slot or
            env.opponent_age.shape[1] <= opponent_slot or
            env.opponent_pose.shape[1] <= opponent_slot):
        raise ValueError("contact approach opponent track tensors have invalid shape")
    fresh_limit = max(float(env.dt), float(getattr(env, "perception_interval", 1)) * float(env.dt))
    track_valid = env.opponent_valid[:, opponent_slot]
    track_fresh = env.opponent_age[:, opponent_slot] <= fresh_limit + 1e-6
    if not (track_valid & track_fresh).all():
        raise ValueError("contact approach requires a fresh valid opponent track")
    # Verify that the anonymous sensed track currently corresponds to focal
    # slot 3.  Truth is used only as an evaluation-harness identity check; the
    # command itself targets the sensed opponent track through action 5.
    if env.team_ids.shape != (6,):
        raise ValueError("contact approach requires six robot team IDs")
    opposing = env.team_ids != env.team_ids[opponent_slot]
    if not opposing[focal_slot] or int(opposing.sum().item()) != 3:
        raise ValueError("contact approach team assignment does not identify focal as opponent")
    sensed_xy = env.opponent_pose[:, opponent_slot, :2]
    true_opponents = env.sim.pose[:, opposing, :2]
    distances = torch.linalg.vector_norm(sensed_xy[:, None, :] - true_opponents, dim=-1)
    focal_index = torch.nonzero(torch.arange(6, device=env.device)[opposing] == focal_slot).item()
    if not (distances.argmin(-1) == focal_index).all():
        raise ValueError("fresh sensed opponent track is not directed at focal slot 3")
    indices, valid, _ = env._candidates()
    legal, _ = env._action_mask(indices, valid)
    if not legal[:, opponent_slot, 5].all():
        raise ValueError("contact approach requires action mask to allow approach action 5")
    actions = torch.full((env.n, 6), 7, device=env.device, dtype=torch.long)
    actions[:, opponent_slot] = 5
    return actions


class ContactExposureProxyRecorder:
    """Proxy recorder for ordinary-physics robot-pair contact and clearance.

    The simulator exposes per-robot contact flags, not the contacting partner.
    A pair event therefore requires both robots' same-tick flags and their
    measured chassis centers within the conservative pair envelope. This is an
    explicit proxy and cannot satisfy the registry's pair-impulse or
    return-to-precontact-route-following measurements.
    """
    def __init__(self, worlds: int, *, device="cpu", dt: float = .02,
                 geometry: str = "crossing_contact",
                 pair: tuple[int, int] = (2, 3), low_speed_m_s: float = .20,
                 recovery_clearance_m: float = .05, stable_ticks: int = 5):
        if type(worlds) is not int or worlds < 1:
            raise ValueError("worlds must be a positive integer")
        if pair != (2, 3):
            raise ValueError("registered contact geometry uses robot pair slots (2, 3)")
        if geometry not in CONTACT_GEOMETRIES:
            raise ValueError("geometry is not a registered robot-contact exposure")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if (not math.isfinite(low_speed_m_s) or low_speed_m_s < 0 or
                not math.isfinite(recovery_clearance_m) or recovery_clearance_m < 0):
            raise ValueError("recovery speed and clearance must be finite and non-negative")
        if type(stable_ticks) is not int or stable_ticks < 1:
            raise ValueError("stable_ticks must be a positive integer")
        self.worlds, self.device = worlds, torch.device(device)
        self.dt, self.pair, self.geometry = float(dt), pair, geometry
        self.factor_identity = phase4_context_identity("robot_contact_geometry", geometry)
        self.low_speed_m_s = float(low_speed_m_s)
        self.recovery_clearance_m = float(recovery_clearance_m)
        self.stable_ticks_required = stable_ticks
        self.tick = 0
        self.pair_contact_proxy_first_tick = torch.full((worlds,), -1,
            device=self.device, dtype=torch.long)
        self.pair_contact_proxy_last_tick = torch.full((worlds,), -1,
            device=self.device, dtype=torch.long)
        self.pair_contact_proxy_tick_count = torch.zeros((worlds,),
            device=self.device, dtype=torch.long)
        self._recovery_stable_ticks = torch.zeros((worlds,),
            device=self.device, dtype=torch.long)
        self.post_contact_clearance_proxy_s = torch.full((worlds,), float("nan"),
            device=self.device, dtype=torch.float32)

    def record_tick(self, pose: torch.Tensor, velocity: torch.Tensor,
                    robot_contact_by_robot: torch.Tensor,
                    robot_radius: torch.Tensor) -> None:
        """Record one completed physics tick from simulator tensors."""
        i, j = self.pair
        if pose.shape != (self.worlds, 6, 3):
            raise ValueError("pose must have shape [worlds, 6, 3]")
        if velocity.shape[:2] != (self.worlds, 6) or velocity.shape[-1] < 2:
            raise ValueError("velocity must have shape [worlds, 6, >=2]")
        if robot_contact_by_robot.shape != (self.worlds, 6):
            raise ValueError("robot_contact_by_robot must have shape [worlds, 6]")
        if robot_radius.shape != (self.worlds, 6):
            raise ValueError("robot_radius must have shape [worlds, 6]")
        distance = torch.linalg.vector_norm(pose[:, i, :2]-pose[:, j, :2], dim=-1)
        pair_radius = robot_radius[:, i]+robot_radius[:, j]
        near_pair = distance <= pair_radius + .04
        contact = (robot_contact_by_robot[:, i] & robot_contact_by_robot[:, j] & near_pair)
        first = contact & (self.pair_contact_proxy_first_tick < 0)
        self.pair_contact_proxy_first_tick = torch.where(first,
            torch.full_like(self.pair_contact_proxy_first_tick, self.tick),
            self.pair_contact_proxy_first_tick)
        self.pair_contact_proxy_last_tick = torch.where(contact,
            torch.full_like(self.pair_contact_proxy_last_tick, self.tick),
            self.pair_contact_proxy_last_tick)
        self.pair_contact_proxy_tick_count += contact.long()
        contacted = self.pair_contact_proxy_first_tick >= 0
        speeds = torch.linalg.vector_norm(velocity[:, (i, j), :2], dim=-1)
        recovered_clearance = distance >= pair_radius+self.recovery_clearance_m
        stable = (contacted & recovered_clearance &
                  (speeds <= self.low_speed_m_s).all(-1) &
                  torch.isnan(self.post_contact_clearance_proxy_s) & ~contact)
        self._recovery_stable_ticks = torch.where(stable,
            self._recovery_stable_ticks+1, torch.zeros_like(self._recovery_stable_ticks))
        recovered = (self._recovery_stable_ticks >= self.stable_ticks_required) & \
                    torch.isnan(self.post_contact_clearance_proxy_s)
        # self.tick is the 0-based tick currently being recorded.  The proxy
        # timestamp is end-of-tick; elapsed time is the number of complete
        # clear physics intervals after the final contact tick.
        duration_ticks = self.tick-self.pair_contact_proxy_last_tick
        duration = duration_ticks.clamp_min(0).to(torch.float32)*self.dt
        self.post_contact_clearance_proxy_s = torch.where(
            recovered, duration, self.post_contact_clearance_proxy_s)
        self.tick += 1

    def summary(self) -> dict:
        """Return explicit observed, not-qualified contact/recovery fields."""
        ticks = self.pair_contact_proxy_first_tick.tolist()
        last_ticks = self.pair_contact_proxy_last_tick.tolist()
        counts = self.pair_contact_proxy_tick_count.tolist()
        recovery = self.post_contact_clearance_proxy_s.tolist()
        return {"schema": "meco-phase4-contact-exposure-proxy-v2",
                "factor_identity": self.factor_identity,
                "pair_slots": list(self.pair), "physics_tick_count": self.tick,
                "pair_contact_proxy_first_tick": [None if tick < 0 else tick for tick in ticks],
                "pair_contact_proxy_last_tick": [None if tick < 0 else tick for tick in last_ticks],
                "pair_contact_proxy_tick_count": counts,
                "post_contact_clearance_proxy_s": [None if math.isnan(value) else value for value in recovery],
                "registry_contact_signal_status": "unavailable_without_pair_impulse_signal",
                "registry_recovery_time_after_bump_s": None,
                "proxy_scope": "distance_gated_pair_proxy_from_per_robot_flags",
                "promotion_status": "proxy_only_requires_pair_impulse_and_route_state_timestamps"}


@dataclass(frozen=True)
class TeammateBehaviorPool:
    """Deterministic balanced behavior schedule for selected robot slots."""
    seed: int
    robot_ids: tuple[int, ...] = (4, 5)

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (not self.robot_ids or len(set(self.robot_ids)) != len(self.robot_ids)
                or any(type(robot) is not int or robot not in range(6)
                       for robot in self.robot_ids)):
            raise ValueError("robot_ids must be unique robot slots in [0, 6)")

    def for_episode(self, episode: int) -> dict[int, str]:
        if type(episode) is not int or episode < 0:
            raise ValueError("episode must be a non-negative integer")
        offset = self.seed % len(TEAMMATE_BEHAVIORS)
        return {robot: TEAMMATE_BEHAVIORS[(offset + episode + 2*index) %
                                           len(TEAMMATE_BEHAVIORS)]
                for index, robot in enumerate(self.robot_ids)}

    def metadata_for_episode(self, episode: int) -> dict:
        assignments = self.for_episode(episode)
        return {"episode": episode,
                "assignments": {str(robot): phase4_context_identity(
                    "teammate_behaviors", behavior)
                    for robot, behavior in assignments.items()}}


def teammate_behavior_actions(env, assignments: dict[int, str]) -> torch.Tensor:
    """Build legal macro actions from sensed candidates and current masks."""
    return teammate_behavior_action_details(env, assignments)["actions"]


def teammate_behavior_action_details(env, assignments: dict[int, str]) -> dict:
    """Return requested legal actions and explicit helper fallback diagnostics."""
    if not isinstance(assignments, dict):
        raise TypeError("assignments must map robot slot to behavior name")
    if any(type(robot) is not int or robot not in range(6) for robot in assignments):
        raise ValueError("behavior assignment contains an invalid robot slot")
    if any(name not in TEAMMATE_BEHAVIORS for name in assignments.values()):
        raise ValueError("unknown teammate behavior")
    indices, valid, _ = env._candidates()
    action_legal, _ = env._action_mask(indices, valid)
    actions = torch.full((env.n, 6), 7, dtype=torch.long, device=env.device)
    requested_actions = {"score": 4, "defend": 5, "ferry": 6, "hold_position": 7}
    desired = torch.full((env.n, 6), 7, dtype=torch.long, device=env.device)
    requested_legal = torch.zeros((env.n, 6), dtype=torch.bool, device=env.device)
    helper_fallback = torch.zeros((env.n, 6), dtype=torch.bool, device=env.device)
    for robot, behavior in assignments.items():
        if behavior == "collect":
            candidates = action_legal[:, robot, :4] & valid[:, robot]
            chosen = candidates.to(torch.int64).argmax(-1)
            has_candidate = candidates.any(-1)
            chosen = torch.where(has_candidate, chosen, torch.full_like(chosen, 7))
            desired[:, robot] = torch.where(has_candidate, chosen,
                                             torch.full_like(chosen, 0))
        else:
            chosen = torch.full((env.n,), requested_actions[behavior],
                                dtype=torch.long, device=env.device)
            desired[:, robot] = chosen
        in_range = chosen < action_legal.shape[-1]
        allowed = in_range & action_legal[:, robot].gather(1, chosen.clamp_max(7)[:, None]).squeeze(1)
        fallback = torch.where(action_legal[:, robot, 7], torch.full_like(chosen, 7),
                               action_legal[:, robot].to(torch.int64).argmax(-1))
        actions[:, robot] = torch.where(allowed, chosen, fallback)
        intended = desired[:, robot]
        requested_in_range = intended < action_legal.shape[-1]
        requested_legal[:, robot] = (requested_in_range &
            action_legal[:, robot].gather(1, intended.clamp(0, 7)[:, None]).squeeze(1))
        helper_fallback[:, robot] = actions[:, robot] != intended
    return {"actions": actions, "requested_actions": desired,
            "requested_legal": requested_legal, "helper_fallback": helper_fallback}


def phase4_teammate_behavior_assignment(*, arm: str, scenario_id: str,
        split: str, protocol_id: str, protocol_sha256: str,
        qualification_case: dict, categories, seeds, pool_seed: int = 0) -> dict:
    """Build a paired, exact validation assignment for full-match slots 4/5."""
    import hashlib, json
    if arm not in ("no_factor", "behavior_pool"):
        raise ValueError("teammate behavior arm must be no_factor or behavior_pool")
    if scenario_id != "full_match_offense" or split != "validation":
        raise ValueError("teammate behavior assignments are validation-only full-match lanes")
    if (not isinstance(protocol_id, str) or not protocol_id or
            not isinstance(protocol_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None):
        raise ValueError("teammate behavior assignment requires exact protocol identity")
    case_fields = {"layout_id", "defender_family", "start_condition", "sensor_profile"}
    if (not isinstance(qualification_case, dict) or
            set(qualification_case) != case_fields or
            any(not isinstance(value, str) or not value
                for value in qualification_case.values())):
        raise ValueError("teammate behavior assignment requires the exact frozen case strata")
    categories = tuple(categories); seeds = tuple(seeds)
    if (not categories or len(set(categories)) != len(categories) or
            any(not isinstance(category, str) or not category for category in categories)):
        raise ValueError("teammate behavior categories must be nonempty and unique")
    if (not seeds or len(set(seeds)) != len(seeds) or
            any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)):
        raise ValueError("teammate behavior seeds must be exact unique non-negative integers")
    if type(pool_seed) is not int or pool_seed < 0:
        raise ValueError("teammate behavior pool seed must be non-negative")
    pool = TeammateBehaviorPool(pool_seed, (4, 5))
    schedule = [{"seed": seed, "assignments": pool.for_episode(index)}
                for index, seed in enumerate(seeds)]
    identity = {"scenario_id": scenario_id, "split": split,
        "protocol_id": protocol_id, "protocol_sha256": protocol_sha256,
        "qualification_case": qualification_case, "categories": list(categories),
        "seeds": list(seeds), "slots": [4, 5], "pool_seed": pool_seed,
        "pool_seed_schedule": [entry["assignments"] for entry in schedule]}
    pair_sha = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return {"schema": "phase4-teammate-behavior-assignment-v1",
        **identity, "arm": arm, "pair_sha256": pair_sha,
        "behavior_schedule": schedule if arm == "behavior_pool" else None,
        "assignment_runtime_event": "phase4_teammate_behavior",
        "assignment_path_supported": True,
        "outcome_by_teammate_behavior": None, "teammate_collision_rate": None,
        "score_contribution": None, "qualification_eligible": False}


def validate_phase4_teammate_behavior_assignment(assignment: dict) -> dict:
    if not isinstance(assignment, dict):
        raise ValueError("teammate behavior assignment must be a mapping")
    arm = assignment.get("arm")
    expected = phase4_teammate_behavior_assignment(arm=arm,
        scenario_id=assignment.get("scenario_id"), split=assignment.get("split"),
        protocol_id=assignment.get("protocol_id"),
        protocol_sha256=assignment.get("protocol_sha256"),
        qualification_case=assignment.get("qualification_case"),
        categories=assignment.get("categories", ()), seeds=assignment.get("seeds", ()),
        pool_seed=assignment.get("pool_seed"))
    if assignment != expected:
        raise ValueError("teammate behavior assignment differs from its exact paired protocol binding")
    return expected


def phase4_teammate_behavior_record(assignment: dict, runtime: dict, *,
        episode_id: str, scenario_id: str, category: str, seed: int,
        policy_id: str, condition_id: str | None,
        source_identity_sha256: str) -> dict:
    """Bind evaluator action trace to its exact assignment and provenance."""
    assignment = validate_phase4_teammate_behavior_assignment(assignment)
    if (not isinstance(runtime, dict) or
            runtime.get("schema") != "phase4-teammate-behavior-runtime-v1" or
            runtime.get("arm") != assignment["arm"] or
            not isinstance(runtime.get("events"), list)):
        raise ValueError("teammate behavior runtime record is missing or mismatched")
    if (not isinstance(episode_id, str) or not episode_id or
            scenario_id != assignment["scenario_id"] or
            category not in assignment["categories"] or seed not in assignment["seeds"] or
            not isinstance(policy_id, str) or not policy_id or
            not isinstance(source_identity_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", source_identity_sha256) is None):
        raise ValueError("teammate behavior record provenance is not assignment-bound")
    expected_assignments = (None if assignment["arm"] == "no_factor" else
        {str(slot): behavior for slot, behavior in
         assignment["behavior_schedule"][assignment["seeds"].index(seed)]["assignments"].items()})
    if (runtime.get("assignments") != expected_assignments or
            runtime.get("assignment_applied") != (expected_assignments is not None)):
        raise ValueError("teammate behavior runtime does not match its assigned seed/arm")
    return {"schema": "phase4-teammate-behavior-runtime-v1",
        "assignments": runtime["assignments"],
        "assignment_applied": runtime["assignment_applied"],
        "events": runtime["events"],
        "assignment_pair_sha256": assignment["pair_sha256"],
        "episode_id": episode_id, "scenario_id": scenario_id,
        "category": category, "seed": seed, "policy_id": policy_id,
        "condition_id": condition_id,
        "source_identity_sha256": source_identity_sha256,
        "protocol_id": assignment["protocol_id"],
        "protocol_sha256": assignment["protocol_sha256"],
        "qualification_case": assignment["qualification_case"],
        "arm": assignment["arm"],
        "outcome_by_teammate_behavior": None,
        "teammate_collision_rate": None, "score_contribution": None,
        "qualification_eligible": False}


def apply_focal_pose_offset(observation: torch.Tensor, offset_name: str, *,
                            field_length: float, field_width: float,
                            normalized: bool = True,
                            focal_robot: int | None = None) -> tuple[torch.Tensor, dict]:
    """Perturb self pose and make dependent observation channels consistent.

    The 137-value 3v3 observation has its self pose in channels 0..2. Relative
    geometry is translated into the estimated-pose frame. Route length is
    invalidated because only its first segment is available, while candidate
    own ETA and risk are recomputed from corrected candidate-relative vectors.
    No simulator state or input observation is mutated. This post-observation
    transform does not prevent pre-transform candidate ranking, masks, route
    roots, or direct ``env.sim.pose`` readers from consulting physical truth;
    it is not a complete training no-leak boundary until those readers consume
    an estimated-pose state.
    """
    if not isinstance(observation, torch.Tensor) or observation.ndim < 1 or observation.shape[-1] < 137:
        raise ValueError("pose perturbation requires a full 137-channel 3v3 observation")
    identity = phase4_context_identity("focal_pose_offsets", offset_name)
    if field_length <= 0 or field_width <= 0:
        raise ValueError("field dimensions must be positive")
    if observation.ndim >= 3 and observation.shape[-2] == 6:
        if type(focal_robot) is not int or focal_robot not in range(6):
            raise ValueError("focal_robot in [0, 6) is required for six-agent observations")
        view = observation[..., focal_robot, :]
    else:
        if focal_robot is not None:
            raise ValueError("focal_robot is only valid for six-agent observations")
        view = observation

    config = FOCAL_POSE_OFFSETS[offset_name]
    dx_m, dy_m = config["offset_xy_m"]
    dtheta = config["heading_offset_rad"]
    scale_x = field_length if normalized else 1.
    scale_y = field_width if normalized else 1.
    delta_x, delta_y = dx_m/field_length, dy_m/field_width
    result = observation.clone()
    output = result[..., focal_robot, :] if focal_robot is not None else result
    original = view

    # The first three channels are absolute self pose. Base field-feature
    # centers (four x/y/radius triples at 23:35) are also absolute.
    output[..., 0] = original[..., 0] + dx_m/scale_x
    output[..., 1] = original[..., 1] + dy_m/scale_y
    theta_scale = math.pi if normalized else 1.
    theta = original[..., 2]*theta_scale + dtheta
    output[..., 2] = torch.atan2(torch.sin(theta), torch.cos(theta))/theta_scale
    for x_idx, y_idx, radius_idx in ((23, 24, 25), (26, 27, 28),
                                     (29, 30, 31), (32, 33, 34)):
        present = original[..., radius_idx] > 0
        output[..., x_idx] = torch.where(present,
            original[..., x_idx]-dx_m/scale_x, original[..., x_idx])
        output[..., y_idx] = torch.where(present,
            original[..., y_idx]-dy_m/scale_y, original[..., y_idx])

    # All following x/y pairs are relative field-normalized vectors.
    route_present = original[..., 35:37].abs().sum(-1) > 0
    output[..., 35] = torch.where(route_present, original[..., 35]-delta_x, 0.)
    output[..., 36] = torch.where(route_present, original[..., 36]-delta_y, 0.)
    output[..., 37] = 0.  # route cost depends on hidden route geometry from truth
    for base in (38, 43, 48, 53, 58, 68, 85, 87):
        output[..., base] = original[..., base]-delta_x
        output[..., base+1] = original[..., base+1]-delta_y
    old_candidates = original[..., 89:129].reshape(*original.shape[:-1], 4, 10)
    candidate_output = output[..., 89:129].reshape(*output.shape[:-1], 4, 10)
    candidate_output[..., 0] = old_candidates[..., 0]-delta_x
    candidate_output[..., 1] = old_candidates[..., 1]-delta_y
    old_xy = torch.stack((old_candidates[..., 0]*field_length,
                          old_candidates[..., 1]*field_width), -1)
    new_xy = torch.stack((candidate_output[..., 0]*field_length,
                          candidate_output[..., 1]*field_width), -1)
    old_distance = torch.linalg.vector_norm(old_xy, dim=-1).clamp_min(1e-6)
    new_distance = torch.linalg.vector_norm(new_xy, dim=-1)
    candidate_output[..., 2] = old_candidates[..., 2]*new_distance/old_distance
    candidate_output[..., 5] = torch.sigmoid(
        (candidate_output[..., 2]-old_candidates[..., 3])*40.)

    evidence = {"applied": True, "factor_identity": identity,
                "focal_robot": focal_robot,
                "physical_truth_mutated": False,
                "policy_boundary_scope": "post_observation_137_channel_transform_only",
                "training_no_leak_complete": False,
                "downstream_truth_readers_modified": False,
                "unresolved_truth_readers": [
                    "candidate ranking and masks computed before observation transform",
                    "route planning roots and path cache",
                    "direct env.sim.pose readers in policy/candidate/Plan A/training code"],
                "dependent_channel_policy": "recompute_relative_or_invalidate",
                "invalidated_channels": [37],
                "recomputed_channel_groups": ["field_feature_positions",
                    "route_first_segment", "local_fuel_positions",
                    "teammate_positions", "hub_relative_positions",
                    "candidate_positions", "candidate_own_eta", "candidate_risk"]}
    return result, evidence


def bump_contact_exposure_geometry(env, side: str = "lower", *,
                                   alliance: str = "blue", robot_slot: int = 3) -> dict:
    """Return a chassis-clear pose facing inward toward a real bump surface.

    Field bump terrain is separate from the robot-to-robot contact factor. This
    only describes a physical terrain approach; ordinary physics produces any
    resulting traversal or contact response.
    """
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    if alliance not in ("red", "blue"):
        raise ValueError("alliance must be 'red' or 'blue'")
    if type(robot_slot) is not int or robot_slot not in range(6):
        raise ValueError("robot_slot must be in [0, 6)")
    from .field import bump_boxes, static_collision_boxes
    box = next((item for item in bump_boxes(env.field_boxes)
                if item.name.startswith(f"{alliance}_bump_{side}")), None)
    if box is None:
        raise ValueError(f"no {alliance} {side} bump geometry in current layout")
    radii = _robot_radii(env)[:, robot_slot]
    radius = float(radii.max().item())
    # Red starts west of its bump and approaches east; blue starts east and
    # approaches west. The full conservative chassis radius clears the bump.
    direction = 1. if alliance == "red" else -1.
    x = box.x - direction*(box.length/2 + radius + .05)
    y = box.y
    length, width = _dimensions(env)
    if not (radius <= x <= length-radius and radius <= y <= width-radius):
        raise ValueError("bump approach pose violates legal field bounds")
    for solid in static_collision_boxes(env.field_boxes):
        if (abs(x-solid.x) <= solid.length/2+radius and
                abs(y-solid.y) <= solid.width/2+radius):
            raise ValueError(f"bump approach pose intersects solid geometry: {solid.name}")
    heading = 0. if direction > 0 else math.pi
    return {"bump_name": box.name, "entry_pose": (x, y, heading),
            "alliance": alliance, "side": side,
            "terrain_source": "simulator_bump_regions",
            "chassis_clearance_m": .05,
            "contact_injected": False, "physics": "ordinary"}
