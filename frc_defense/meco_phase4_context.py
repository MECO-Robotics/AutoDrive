"""Runtime context fixtures for Phase 4 full-team evaluation.

This module deliberately does not wire itself into curriculum scenarios or
qualification protocols. Callers opt into a named start profile, teammate
behavior assignment, or observation-only pose offset at the evaluation seam.
Physical contacts and bump traversal remain the simulator's normal physics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


TEAMMATE_BEHAVIORS = ("collect", "score", "ferry", "defend", "hold")
START_PROFILE_NAMES = ("midfield_spread", "crossing_lanes")
CONTACT_GEOMETRIES = ("crossing_contact", "side_contact", "recovery_contact")

# Start coordinates are generated from actual field dimensions so the same
# profiles work on the supported qualification layouts. Poses are slot ordered
# (0..5); the two alliances face each other across the midfield lanes.
_START_LAYOUTS = {
    "midfield_spread": (
        (5.95, .95, 0.), (6.65, 4.035, 0.), (5.95, 7.12, 0.),
        (10.59, .95, math.pi), (9.89, 4.035, math.pi),
        (10.59, 7.12, math.pi)),
    "crossing_lanes": (
        (6.25, 1.25, .18), (6.65, 4.035, 0.), (6.25, 6.82, -.18),
        (10.29, 1.25, math.pi-.18), (9.89, 4.035, math.pi),
        (10.29, 6.82, math.pi+.18)),
}


def _field_dimensions(env) -> tuple[float, float]:
    return float(env.sim.field_length), float(env.sim.field_width)


def _max_robot_radius(env) -> torch.Tensor:
    # Circumradius is conservative for every heading and therefore cannot
    # admit a pose that clips a wall or a solid field structure.
    return .5 * torch.sqrt(env.sim.length.square() + env.sim.width.square())


def full_team_start_profile(env, profile: str) -> torch.Tensor:
    """Return a validated ``[world, 6, 3]`` physical start pose tensor.

    The returned tensor is a proposal. It does not modify ``env.sim.pose``;
    callers apply it at reset and then let ordinary physics advance the scene.
    """
    if profile not in _START_LAYOUTS:
        raise ValueError(f"unknown full-team start profile: {profile}")
    length, width = _field_dimensions(env)
    coords = torch.tensor(_START_LAYOUTS[profile], device=env.device,
                          dtype=env.sim.pose.dtype)
    # Profiles are relative to the canonical 16.54 x 8.07 m field. Scale only
    # the coordinates, not robot headings, for a layout with different bounds.
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


def contact_exposure_profile(env, geometry: str) -> torch.Tensor:
    """Return a legal six-robot profile with one close, non-overlapping pair.

    The pair starts with a small physical clearance. It is not teleported into
    collision; subsequent contact, if reached, is resolved by normal physics.
    """
    if geometry not in CONTACT_GEOMETRIES:
        raise ValueError(f"unknown contact exposure geometry: {geometry}")
    length, width = _field_dimensions(env)
    cx, cy = length * .5, width * .5
    poses = torch.tensor((
        (length*.365, width*.15, 0.),
        (length*.365, width*.85, 0.),
        (cx-.68, cy, 0.),
        (cx+.68, cy, math.pi),
        (length*.635, width*.15, math.pi),
        (length*.635, width*.85, math.pi)),
        device=env.device, dtype=env.sim.pose.dtype)
    if geometry == "side_contact":
        poses[2] = poses.new_tensor((cx, cy-.68, math.pi/2))
        poses[3] = poses.new_tensor((cx, cy+.68, -math.pi/2))
    elif geometry == "recovery_contact":
        # A glancing recovery approach: a small longitudinal stagger while
        # retaining separation above the combined conservative chassis radii.
        poses[2] = poses.new_tensor((cx-.78, cy-.12, .12))
        poses[3] = poses.new_tensor((cx+.78, cy+.12, math.pi+.12))
    validate_full_team_start_profile(env, poses.expand(env.n, -1, -1))
    return poses.expand(env.n, -1, -1).clone()


def validate_full_team_start_profile(env, poses: torch.Tensor) -> None:
    """Raise if any pose is out of bounds, overlaps a robot, or blocks a solid.

    Bump and trench footprints are excluded from solid geometry. The physical
    simulator owns those surfaces and their safety/contact response.
    """
    if not isinstance(poses, torch.Tensor) or poses.shape != (env.n, 6, 3):
        raise ValueError("start poses must have shape [num_worlds, 6, 3]")
    if not torch.isfinite(poses).all():
        raise ValueError("start poses must be finite")
    length, width = _field_dimensions(env)
    radius = _max_robot_radius(env)
    xy = poses[..., :2]
    if ((xy[..., 0] < radius[None]) | (xy[..., 0] > length-radius[None]) |
            (xy[..., 1] < radius[None]) | (xy[..., 1] > width-radius[None])).any():
        raise ValueError("start profile violates legal field bounds")
    delta = xy[:, :, None, :] - xy[:, None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    pair_radius = radius[:, :, None] + radius[:, None, :]
    pair_mask = ~torch.eye(6, device=env.device, dtype=torch.bool)
    if (distance[:, pair_mask] < pair_radius[:, pair_mask] + .02).any():
        raise ValueError("start profile has overlapping robot chassis envelopes")

    # Use the simulator's authoritative collision footprints; traversable bump
    # and trench regions are intentionally not static solid obstacles.
    from .field import static_collision_boxes
    solids = static_collision_boxes(env.field_boxes)
    for box in solids:
        dx = (xy[..., 0] - float(box.x)).abs()
        dy = (xy[..., 1] - float(box.y)).abs()
        # Circumradius inflated AABB test is conservative across headings.
        intersects = ((dx <= float(box.length)/2 + radius[None]) &
                      (dy <= float(box.width)/2 + radius[None]))
        if intersects.any():
            raise ValueError(f"start profile intersects solid field geometry: {box.name}")


@dataclass(frozen=True)
class TeammateBehaviorPool:
    """Deterministic, balanced behavior schedule for selected teammate slots."""
    seed: int
    robot_ids: tuple[int, ...] = (4, 5)

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (not self.robot_ids or len(set(self.robot_ids)) != len(self.robot_ids)
                or any(type(robot) is not int or robot not in (0, 1, 2, 3, 4, 5)
                       for robot in self.robot_ids)):
            raise ValueError("robot_ids must be unique robot slots in [0, 6)")

    def for_episode(self, episode: int) -> dict[int, str]:
        if type(episode) is not int or episode < 0:
            raise ValueError("episode must be a non-negative integer")
        offset = self.seed % len(TEAMMATE_BEHAVIORS)
        return {robot: TEAMMATE_BEHAVIORS[(offset + episode + 2*index) %
                                           len(TEAMMATE_BEHAVIORS)]
                for index, robot in enumerate(self.robot_ids)}


def teammate_behavior_actions(env, assignments: dict[int, str]) -> torch.Tensor:
    """Build legal macro actions from the environment's sensed masks.

    Actions use existing simulator semantics: 0..3 collect a ranked sensed
    target, 4 score, 5 defend/intercept, 6 ferry/search, and 7 hold. No hidden
    piece ownership or opponent truth is read here.
    """
    if not isinstance(assignments, dict):
        raise TypeError("assignments must map robot slot to behavior name")
    if any(type(robot) is not int or robot not in range(6) for robot in assignments):
        raise ValueError("behavior assignment contains an invalid robot slot")
    if any(name not in TEAMMATE_BEHAVIORS for name in assignments.values()):
        raise ValueError("unknown teammate behavior")
    indices, valid, _ = env._candidates()
    legal, _ = env._action_mask(indices, valid)
    actions = torch.full((env.n, 6), 7, dtype=torch.long, device=env.device)
    requested = {"score": 4, "defend": 5, "ferry": 6, "hold": 7}
    for robot, behavior in assignments.items():
        if behavior == "collect":
            candidates = legal[:, robot, :4] & valid[:, robot]
            chosen = candidates.to(torch.int64).argmax(-1)
            chosen = torch.where(candidates.any(-1), chosen, torch.full_like(chosen, 7))
        else:
            chosen = torch.full((env.n,), requested[behavior],
                                dtype=torch.long, device=env.device)
        in_range = chosen < legal.shape[-1]
        allowed = in_range & legal[:, robot].gather(1, chosen.clamp_max(7)[:, None]).squeeze(1)
        fallback = legal[:, robot].to(torch.int64).argmax(-1)
        actions[:, robot] = torch.where(allowed, chosen, fallback)
    return actions


def apply_focal_pose_offset(observation: torch.Tensor, offset_xy_m: tuple[float, float],
                            heading_offset_rad: float, *, field_length: float,
                            field_width: float, normalized: bool = True,
                            focal_robot: int | None = None) -> tuple[torch.Tensor, dict]:
    """Return a cloned policy observation with only focal self-pose channels changed.

    For 3v3 observations the first three values are own x, y, heading. When
    normalized, x/y are scaled by field dimensions and heading by pi. The
    original observation and physical simulator state are never mutated.
    Derived route features remain as supplied by the caller, so this helper is
    an observation-channel perturbation rather than a full localization stack.
    """
    if not isinstance(observation, torch.Tensor) or observation.ndim < 1 or observation.shape[-1] < 3:
        raise ValueError("observation must be a tensor with at least 3 channels")
    dx, dy = offset_xy_m
    values = (float(dx), float(dy), float(heading_offset_rad),
              float(field_length), float(field_width))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("pose offsets and field dimensions must be finite")
    if field_length <= 0 or field_width <= 0:
        raise ValueError("field dimensions must be positive")
    if observation.ndim >= 3 and observation.shape[-2] == 6:
        if type(focal_robot) is not int or focal_robot not in range(6):
            raise ValueError("focal_robot in [0, 6) is required for six-agent observations")
        pose_channels = observation[..., focal_robot, :3]
    else:
        if focal_robot is not None:
            raise ValueError("focal_robot is only valid for six-agent observations")
        pose_channels = observation[..., :3]
    result = observation.clone()
    pose_result = (result[..., focal_robot, :3] if focal_robot is not None
                   else result[..., :3])
    xy_scale_x = field_length if normalized else 1.
    xy_scale_y = field_width if normalized else 1.
    theta_scale = math.pi if normalized else 1.
    pose_result[..., 0] = pose_channels[..., 0] + dx / xy_scale_x
    pose_result[..., 1] = pose_channels[..., 1] + dy / xy_scale_y
    theta = pose_channels[..., 2] * theta_scale + heading_offset_rad
    pose_result[..., 2] = torch.atan2(torch.sin(theta), torch.cos(theta)) / theta_scale
    evidence = {"applied": True, "channel_indices": [0, 1, 2],
                "offset_xy_m": [float(dx), float(dy)],
                "heading_offset_rad": float(heading_offset_rad),
                "physical_truth_mutated": False,
                "derived_features_recomputed": False,
                "focal_robot": focal_robot}
    return result, evidence


def bump_contact_exposure_geometry(env, side: str = "lower", *, alliance: str = "blue") -> dict:
    """Describe legal bump-edge exposure from the simulator's ordinary geometry."""
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    if alliance not in ("red", "blue"):
        raise ValueError("alliance must be 'red' or 'blue'")
    from .field import bump_boxes
    boxes = bump_boxes(env.field_boxes)
    box = next((item for item in boxes if item.name.startswith(alliance + "_bump_" + side)), None)
    if box is None:
        raise ValueError(f"no {alliance} {side} bump geometry in current layout")
    # Return a point just outside the bump footprint, not a fabricated impact.
    # Evaluators may start a robot there and drive across using normal physics.
    direction = -1. if alliance == "blue" else 1.
    edge_x = box.x + direction * (box.length/2 + .05)
    edge_y = box.y
    if not (0. <= edge_x <= env.sim.field_length and 0. <= edge_y <= env.sim.field_width):
        raise ValueError("bump exposure point is outside legal field bounds")
    heading = 0. if direction < 0 else math.pi
    return {"bump_name": box.name, "entry_pose": (edge_x, edge_y, heading),
            "terrain_source": "simulator_bump_regions",
            "contact_injected": False, "physics": "ordinary"}
