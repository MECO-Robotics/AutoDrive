"""Phase 4 runtime context fixtures, bound to the design registry axes.

The helpers are opt-in and do not alter curriculum/protocol wiring. They build
checked full-team reset poses, sensed behavior actions, policy-consistent pose
perturbations, and physical contact/bump approaches. Simulator truth is only
changed by the explicit start-profile reset helper; pose-error factors edit an
observation clone and its dependent features.
"""
from __future__ import annotations

import hashlib
import json
import math
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
    if not isinstance(assignments, dict):
        raise TypeError("assignments must map robot slot to behavior name")
    if any(type(robot) is not int or robot not in range(6) for robot in assignments):
        raise ValueError("behavior assignment contains an invalid robot slot")
    if any(name not in TEAMMATE_BEHAVIORS for name in assignments.values()):
        raise ValueError("unknown teammate behavior")
    indices, valid, _ = env._candidates()
    legal, _ = env._action_mask(indices, valid)
    actions = torch.full((env.n, 6), 7, dtype=torch.long, device=env.device)
    requested = {"score": 4, "defend": 5, "ferry": 6, "hold_position": 7}
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
        fallback = torch.where(legal[:, robot, 7], torch.full_like(chosen, 7),
                               legal[:, robot].to(torch.int64).argmax(-1))
        actions[:, robot] = torch.where(allowed, chosen, fallback)
    return actions


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
