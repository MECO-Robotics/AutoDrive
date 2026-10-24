"""Snapshot-bound simulator candidates with route and capture-geometry estimates.

The existing eight masked actions remain the behavior primitives. K=16/32 add
stable, locally offset collection approach goals; C1/C2 can append a
snapshot-revalidated continue candidate after the K-prefix. A separate expanded
goal set is route-qualified as a bounded candidate-coverage oracle. The selected
endpoint is passed to AD* while the simulator retains action legality and
mechanisms.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import torch

from .experimental_route_evaluation import (MAX_ROUTE_QUERY_BATCH, RouteQuery,
    evaluate_routes)
from .learning_contract import CandidateTable
from .meco_pose_graph import LegalPoseGraph
from .meco_tracks import (SafeActionCandidates, build_safe_action_candidates,
    opponent_forecast, opponent_risk_penalty, prediction_contract, PREDICTION_MODES,
    SAFE_ACTION_FEATURES, EXTRA_FEATURES)


ROUTE_COST_POINTS_PER_SECOND = 0.05
CANDIDATE_BOUNDARY = {
    "version": 9,
    "capacity": [8, 16, 32],
    "source": "all currently sensed fuel tracks plus existing simulator action slots",
    "configurable_k": True,
    "k16_k32_supported": True,
    "within_candidate_pool_recall_only": False,
    "reference_pool": "all route-qualified ordinary rows plus independent 18-offset route-qualified oracle rows for every currently sensed target",
    "recall_limit": "snapshot-bounded deterministic oracle over currently sensed tracks, not a simulator-optimality oracle",
    "expanded_goals": "one selectable direct approach per sensed target in stable distance/buffer order, followed by bounded approach offsets; oracle adds 18 offsets per sensed target",
    "execution": "selected XY endpoint and sensor buffer identity flow through the paired native goal override; slot-3 native controller revalidates the sensed track each physics step and steers pickup bearing toward its current XY throughout approach; endpoint alignment uses native swept clearance checks; action mask/mechanisms remain simulator-owned",
    "target_controller_override": "restricted to the neural offense in slot 3 and a paired collect action; each physics step requires the same sensor-buffer track to remain present, within perception timeout, and within 1.0 m of snapshot XY; use current track delta for slot-3 fuel_delta and pickup target throughout approach, then derive terminal intake bearing from current track XY relative to the selected endpoint; on loss or buffer reuse, clear goal/heading override and end the macro for a fresh candidate decision",
    "target_execution_boundary": "native collector is geometric and may collect any physically contacted eligible fuel; selected sensed target is an intended route goal and terminal bearing, not an exclusive pickup constraint",
    "selectable_order": "legacy eight action rows, then direct goals for sensed targets beyond the native top-four ordered by robot distance and sensor buffer index, then legacy offsets; K8 prefix is unchanged",
    "heading_execution": "selected collection track supplies a dynamically revalidated pickup bearing throughout approach; at the endpoint native controller applies swept static/observed-robot turn-clearance checks before final alignment",
    "future_swept_dynamic_collision_check": False,
    "continue_behavior_candidate": True,
    "continue_gap": "one dynamically revalidated continue row follows the stable K-prefix; target loss, route invalidity, expiry, completion, phase incompatibility, and episode reset clear it",
    "selectable_pool": "ordinary prefix at K=8/16/32 plus at most one appended continue row; K8 IDs/order remain unchanged, while K16/K32 prioritize direct recall of sensed targets beyond top-four before approach-offset variants",
    "commitment_expiry_macro_actions": 10,
    "commitment_target_matching_tolerance_m": 1.0,
    "commitment_switch_cost_points": 0.10,
    "hysteresis": "when a commitment is active, apply the same fixed switch-cost utility penalty to every feasible non-continue row for C1/C2/D1/D2; continuation remains dynamically revalidated",
    "route_feasibility": "route and endpoint clearance plus bearing-aligned collection capture-envelope eligibility; selected-target pickup bearing is commanded through approach, with terminal alignment subject to native swept-clearance checks",
    "capture_estimate": "bearing-aligned capture envelope eligibility proxy from sensed target geometry; native controller steers toward the selected track, while physical capture remains simulator-authoritative",
    "capture_feasibility": {
        "basis": "candidate goal XY and target XY projected into inferred target-bearing frame",
        "longitudinal_min_m": "robot_length_m/2 - 0.075 - env._fuel_radius (native collector radius)",
        "longitudinal_max_m": "robot_length_m/2 + 0.35 + 0.3048 + env._fuel_radius (native collector radius)",
        "lateral_max_m": "robot_width_m/2 + 0.075 + env._fuel_radius (native collector radius)",
        "orientation": "bearing-aligned eligibility geometry; selected target drives native pickup bearing and terminal alignment remains subject to swept-clearance checks",
        "authority": "candidate eligibility proxy only; native gamepiece collision/capture logic is authoritative",
        "missing_target": "ineligible"
    },
    "plan_a_policy_support": "all reachable legal SE(2) graph nodes crossed with currently executable drive/intake/score/ferry/defense behaviors; carrying dumper ferry pairs are excluded because native release may preempt the selected endpoint",
    "plan_a_support_order": "legal_graph_node_ascending_then_behavior_0_to_4_then_descriptor_ascending_v2",
    "plan_a_support_membership": "one model-independent shortest-path pass over legal SE(2) graph plus vectorized behavior-specific endpoint/target constraints; independent of learned P",
    "plan_a_intake_descriptor": "intake pairs retain exact sensed target slot and require a bearing-aligned capture-geometry envelope",
    "plan_a_heading_semantics": "each selected legal SE(2) node carries its terminal heading; the simulator stops at the XY goal, checks sampled static rotation and observed-robot point forecast, then aligns; intake headings are capture-geometry constrained, score headings are hub-facing within 5 degrees; controllable ferry headings satisfy the native 45-degree release cone",
    "plan_a_drive_semantics": "action 7 holds unless the Plan A adapter pairs it with its explicit validated planner goal, retaining native route and obstacle guards; legal dumper traversal remains supported while loaded, while carrying dumper ferry action 6 remains excluded because it may release fuel before reaching the selected endpoint",
    "plan_a_goal_constraints": "score pairs require a safe score-grid endpoint; executable ferry pairs require carried fuel and current ferry endpoint, while dumper ferry pairs are excluded; defense pairs require the current intercept/post endpoint",
    "plan_a_behavior_heading_constraints": "intake headings must keep the exact sensed target within the forward capture envelope; score headings must face the HUB within 5 degrees; controllable ferry headings satisfy the simulator's 45-degree release cone; drive and defense headings use guarded terminal alignment",
    "plan_a_ppo": "sample and recompute log probability over the exact frozen node-major support mask and order",
    "plan_a_route_metadata": "frozen shortest-path parent tree including the root-pose connector and terminal headings; this is graph-route metadata, while execution remains XY AD* with swept SE(2) checks and guarded terminal alignment",
    "plan_a_candidate_k": "not used; candidate_k applies only to Plan C candidate scorers",
}
# The learned candidate scorers receive both these named table columns and
# one missingness bit per column. Keep the contract shared with training and
# checkpoint export rather than inferring it from a stale export fixture.
CANDIDATE_BASE_FEATURE_COUNT = len(SAFE_ACTION_FEATURES) + len(EXTRA_FEATURES)
CANDIDATE_MODEL_FEATURE_COUNT = 2 * CANDIDATE_BASE_FEATURE_COUNT


@dataclass(frozen=True)
class CandidateCommitment:
    """Task-level choice carried between snapshots and requalified each time."""
    candidate_id: str
    action_index: int
    goal_variant: int
    source_xy_m: tuple[float, float] | None
    created_sequence: int
    expires_sequence: int

    def __post_init__(self):
        if (not isinstance(self.candidate_id, str) or not self.candidate_id
                or type(self.action_index) is not int or self.action_index not in range(8)
                or type(self.goal_variant) is not int or self.goal_variant not in range(7)
                or type(self.created_sequence) is not int or self.created_sequence < 0
                or type(self.expires_sequence) is not int
                or self.expires_sequence <= self.created_sequence):
            raise ValueError("invalid Plan C commitment descriptor")
        if self.source_xy_m is not None:
            source = tuple(float(value) for value in self.source_xy_m)
            if len(source) != 2 or not all(math.isfinite(value) for value in source):
                raise ValueError("commitment source position must be a finite XY pair")
            object.__setattr__(self, "source_xy_m", source)


@dataclass(frozen=True)
class RouteFeasibleCandidates:
    observation: object
    table: CandidateTable
    fallback_candidate_id: str
    action_indices: tuple[int, ...]
    route_cost_points: torch.Tensor
    risk_cost_points: torch.Tensor
    route_records: tuple[dict, ...]
    candidate_recall: dict
    commitment: CandidateCommitment | None = None


def _candidate_target(env, action: int, base: SafeActionCandidates, *,
                      target_row=None):
    slot = 3
    pose = env.sim.pose[0, slot]
    xy = pose[:2].clone()
    heading = float(pose[2].item())
    valid = bool(base.table.valid_mask[action])
    intake = action < 4
    if action < 4:
        row = base.table.features[action] if target_row is None else target_row
        if row[2] is None or row[3] is None:
            return xy, heading, False, True
        fuel = torch.tensor((row[2], row[3]), device=env.device, dtype=xy.dtype)
        delta = fuel - xy
        norm = float(delta.norm().item())
        if not math.isfinite(norm) or norm <= 1e-6:
            return xy, heading, False, True
        direction = delta / norm
        target = fuel - direction * 0.45
        radius = float((.5 * torch.sqrt(env.sim.length[0, slot].square() +
                                        env.sim.width[0, slot].square())).item())
        target[0].clamp_(radius, float(env.sim.field_length) - radius)
        target[1].clamp_(radius, float(env.sim.field_width) - radius)
        heading = math.atan2(float(direction[1]), float(direction[0]))
        return target, heading, valid, intake
    if action == 4:  # score: nearest currently safe alliance-zone point
        points = env._score_grid_points[0, slot]
        clear = env._score_grid_clear[0, slot]
        if points.shape[0]:
            distance = (points - xy).norm(dim=-1).masked_fill(~clear, float("inf"))
            index = int(distance.argmin().item())
            if not bool(torch.isfinite(distance[index])):
                return xy, heading, False, False
            target = points[index].clone()
        else:
            radius = .5 * torch.sqrt(env.sim.length[0, slot].square() + env.sim.width[0, slot].square())
            if int(env.team_ids[slot]) == 0:
                low, high = float(radius), float(env.alliance_zone_depth - radius)
            else:
                low, high = float(env.sim.field_length - env.alliance_zone_depth + radius), float(env.sim.field_length - radius)
            target = xy.clone()
            target[0].clamp_(low, high)
        hub = env.hub_centers[env.team_ids[slot]]
        delta = hub - target
        heading = math.atan2(float(delta[1]), float(delta[0]))
        return target, heading, valid, False
    if action == 5:  # offense intercept is only available for a fresh sensed opponent
        if not bool(env.opponent_valid[0, slot]):
            return xy, heading, False, False
        target = env.opponent_pose[0, slot, :2].clone()
        delta = target - xy
        if float(delta.norm()) > 1e-6:
            heading = math.atan2(float(delta[1]), float(delta[0]))
        return target, heading, valid, False
    if action == 6:
        target = env._ferry_targets_for_current_motion()[0, slot].clone()
        delta = target - xy
        if float(delta.norm()) > 1e-6:
            heading = math.atan2(float(delta[1]), float(delta[0]))
        return target, heading, valid, False
    return xy, heading, valid, False


def _endpoint_clear(env, xy, heading: float, *, intake_extended: bool):
    slot = 3
    length = env.sim.length[0, slot].clone()
    if intake_extended:
        length = length + .6096
    width = env.sim.width[0, slot].clone()
    # The selected heading is only a planner feature. Enclose the full robot
    # (including intake) in a square so endpoint clearance holds at every yaw.
    side = torch.sqrt(length.square() + width.square())
    point = xy.reshape(1, 1, 2).expand(1, 2, 2)
    static_ok = bool(env.planner._footprint_path_clear(point,
        torch.zeros(1, device=env.device, dtype=point.dtype),
        side.reshape(1), side.reshape(1))[0].item())
    circles = env._perceived_robot_obstacles()[slot]
    circles = circles[circles[:, 2] > 0]
    if circles.numel() == 0:
        return static_ok
    delta = circles[:, :2] - xy
    clearance_radius = float(side) * .5
    circles_clear = bool((delta.norm(dim=-1) > clearance_radius + circles[:, 2]).all().item())
    return static_ok and circles_clear


def _goal_node(env, graph: LegalPoseGraph, xy, heading: float, *, intake_extended=False):
    if not (0. <= float(xy[0]) < float(env.sim.field_length)
            and 0. <= float(xy[1]) < float(env.sim.field_width)):
        return None, None, 0
    angle = heading % (2 * math.pi)
    hbin = int(round(angle * 8 / (2 * math.pi))) % 8
    # Bind the requested target to its nearest planner-cell center. A blocked
    # cell is rejected; no nearest-free projection is permitted here.
    ix = int(torch.floor(xy[0] / graph.resolution_m).item())
    iy = int(torch.floor(xy[1] / graph.resolution_m).item())
    if not (0 <= ix < graph.nx and 0 <= iy < graph.ny):
        return None, None, hbin
    node = int(graph.lookup[hbin, ix, iy].item())
    if node < 0:
        # The selected sensed target drives native pickup bearing during
        # execution. This graph heading qualifies the candidate's capture
        # geometry; bind lookup to the first legal heading at this cell when
        # that inferred heading is blocked.
        legal_nodes = graph.lookup[:, ix, iy]
        legal_nodes = legal_nodes[legal_nodes >= 0]
        if legal_nodes.numel() == 0:
            return None, None, hbin
        node = int(legal_nodes[0].item())
    center = graph.xy[node]
    if float((center - xy).norm().item()) > graph.resolution_m * .72:
        return None, None, hbin
    if not _endpoint_clear(env, center, float(hbin) * 2 * math.pi / 8,
                           intake_extended=intake_extended):
        return None, None, hbin
    # A2/A3 consume the legal node's heading as part of its SE(2) feature.
    # If the sensed target bearing had to fall back to another legal heading,
    # keep the candidate feature aligned with the actual node identity.
    selected_hbin = int(graph.heading_bin[node].item())
    return node, center, selected_hbin


def _capture_pose_estimate(env, base: SafeActionCandidates, action: int,
                           goal_xy, goal_heading: float | None = None, *,
                           target_row=None) -> dict:
    """Qualify sensed-target geometry in the candidate's inferred heading frame.

    This is a conservative candidate eligibility proxy, not a command to the
    native controller. Simulator capture and mechanism behavior remain the
    final authority.
    """
    if action >= 4:
        return {"status": "not_applicable", "basis": "non_collection_action",
                "feasible": True}
    row = base.table.features[action] if target_row is None else target_row
    if row[2] is None or row[3] is None:
        return {"status": "unavailable", "basis": "target_not_sensed",
                "feasible": False, "reason": "target_not_sensed"}
    slot = 3
    fuel = goal_xy.new_tensor((row[2], row[3]))
    delta = fuel - goal_xy
    distance = float(delta.norm().item())
    if not math.isfinite(distance):
        return {"status": "unavailable", "basis": "nonfinite_target_geometry",
                "feasible": False, "reason": "nonfinite_target_geometry"}
    # Match the native collector predicates in gamepiece_actions.py exactly;
    # fuel_physics.config.radius can differ from the collector's effective
    # radius and would incorrectly reject otherwise capturable candidates.
    radius = float(env._fuel_radius)
    half_length = float(env.sim.length[0, slot]) * .5
    half_width = float(env.sim.width[0, slot]) * .5
    forward_min = half_length - .075 - radius
    forward_limit = (half_length + .35 + .3048 + radius)
    lateral_limit = half_width + .075 + radius
    heading = (math.atan2(float(delta[1]), float(delta[0]))
               if goal_heading is None and distance > 1e-12 else
               (0. if goal_heading is None else float(goal_heading)))
    forward = goal_xy.new_tensor((math.cos(heading), math.sin(heading)))
    left = goal_xy.new_tensor((-math.sin(heading), math.cos(heading)))
    longitudinal = float(torch.dot(delta, forward).item())
    lateral = abs(float(torch.dot(delta, left).item()))
    inside = (forward_min <= longitudinal <= forward_limit
              and lateral <= lateral_limit)
    reason = None
    if longitudinal < forward_min:
        reason = "target_behind_or_inside_minimum_capture_distance"
    elif longitudinal > forward_limit:
        reason = "target_beyond_forward_capture_limit"
    elif lateral > lateral_limit:
        reason = "target_outside_lateral_capture_limit"
    return {"status": "inside_estimate" if inside else "outside_estimate",
            "basis": "sensed_target_bearing_geometry_only",
            "feasible": bool(inside), "reason": reason,
            "distance_m": distance,
            "estimated_longitudinal_m": longitudinal,
            "estimated_lateral_m": lateral,
            "estimated_min_forward_m": forward_min,
            "estimated_forward_limit_m": forward_limit,
            "estimated_lateral_limit_m": lateral_limit}


def _opponent_prediction(env, slot=3, horizon=0., mode="cv"):
    valid = bool(env.opponent_valid[0, slot])
    age = float(env.opponent_age[0, slot])
    position = env.opponent_pose[0, slot, :2].detach().cpu().tolist()
    velocity = env.opponent_velocity[0, slot, :2].detach().cpu().tolist()
    return opponent_forecast(mode=mode, valid=valid, position_xy=position,
        velocity_xy=velocity, age_s=age, horizon_s=max(0., float(horizon)),
        angular_velocity_rad_s=float(env.opponent_velocity[0, slot, 2]))


def _world_zero_perceived_fuel(env):
    """Read slot-3 tracks through the batched native accessor for one world."""
    if env.n != 1:
        raise ValueError("Plan C perceived-target candidates require one world")
    points, velocities, sensed = env._perceived_fuel(3)
    if (not isinstance(points, torch.Tensor)
            or not isinstance(velocities, torch.Tensor)
            or not isinstance(sensed, torch.Tensor)
            or points.ndim != 3 or points.shape[0] != env.n
            or points.shape[-1] != 2 or velocities.shape != points.shape
            or sensed.shape != points.shape[:2]):
        raise ValueError("slot-3 perceived FUEL buffers have misaligned batched shapes")
    # Keep the native accessor's [world, track, ...] contract intact, then
    # select the sole supported world explicitly for candidate construction.
    return points[0], velocities[0], sensed[0]


def _additional_sensed_target_rows(env, base: SafeActionCandidates):
    """Return candidate rows for sensed tracks omitted by the native top four.

    The action id stays a native collect class; the goal endpoint and the
    sensor buffer identity live in this candidate row. Native intake remains
    geometric and authoritative, so this does not turn a sensor track into an
    exclusive piece reservation.
    """
    base_track_ids = {int(base.table.features[i][1]) for i in range(4)
                      if base.table.valid_mask[i] and
                      base.table.features[i][1] is not None}
    valid_collect_actions = [i for i in range(4) if base.table.valid_mask[i]]
    if not valid_collect_actions:
        return []
    fallback_action = valid_collect_actions[0]
    points, _, sensed = _world_zero_perceived_fuel(env)
    sensed_ids = torch.nonzero(sensed, as_tuple=False).flatten().tolist()
    pose_xy = env.sim.pose[0, 3, :2]
    confidence = env.track_confidence[0, 3]
    ages = env.track_age[0, 3]
    opponent_distance = None
    if bool(env.opponent_valid[0, 3]):
        opponent_distance = (points - env.opponent_pose[0, 3, :2]).norm(dim=-1)
    distances = (points - pose_xy).norm(dim=-1)
    ordered = sorted((int(i) for i in sensed_ids if int(i) not in base_track_ids),
                     key=lambda i: (float(distances[i]), i))
    nearby = (((points[:, None, :] - points[None, :, :]).norm(dim=-1) < 1.25)
              & sensed[None, :]).sum(-1)
    field_length, field_width = float(env.sim.field_length), float(env.sim.field_width)
    template = base.table.features[fallback_action]
    result = []
    for track in ordered:
        target = points[track]
        delta = target - pose_xy
        angle = torch.atan2(delta[1], delta[0])
        heading_bin = int(torch.round(torch.remainder(angle, 2. * math.pi) *
                                      (8. / (2. * math.pi))).item()) % 8
        row = list(template)
        row[0] = float(fallback_action)
        row[1] = float(track)
        row[2], row[3] = float(target[0]), float(target[1])
        row[4] = float(heading_bin)
        row[6], row[7] = field_length, field_width
        row[8] = float(distances[track])
        row[9] = float(nearby[track])
        row[10] = float(confidence[track].clamp(0., 1.))
        row[11] = float(ages[track].clamp_min(0.))
        row[12] = (None if opponent_distance is None else
                   float(opponent_distance[track]))
        row[13] = None
        result.append((fallback_action, tuple(row), track))
    return result


@torch.no_grad()
def build_route_feasible_candidates(env, observation: torch.Tensor,
                                    graph: LegalPoseGraph, *, episode_id: str,
                                    action_ticks: int = 12,
                                    candidate_k: int = 8,
                                    prediction_mode: str = "cv",
                                    learned_traversal_cost=None,
                                    learned_cost_scale: float = 1.0,
                                    commitment: CandidateCommitment | None = None) -> RouteFeasibleCandidates:
    """Build the current bounded decision table and qualify every route row.

    Route feasibility uses a fresh uncached TensorADStar workspace over current
    perceived robot obstacles. The inferred target-bearing heading selects a
    legal graph node and contributes an estimate to candidate eligibility;
    endpoint footprint clearance is checked at every yaw. The paired native
    executor steers pickup bearing toward the selected sensed track and guards
    terminal alignment. ETA is distance/speed plus heading-change/turn-rate.
    Dynamic interaction is a point-prediction risk cost; full time-varying
    swept-robot planning remains out of scope and is recorded in provenance.
    """
    if env.n != 1 or not isinstance(episode_id, str) or not episode_id:
        raise ValueError("candidate source requires one world and a unique episode_id")
    if type(action_ticks) is not int or action_ticks < 1:
        raise ValueError("action_ticks must be a positive integer")
    if type(candidate_k) is not int or candidate_k not in (8, 16, 32):
        raise ValueError("candidate_k must be one of 8, 16, or 32")
    if prediction_mode not in PREDICTION_MODES:
        raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
    base = build_safe_action_candidates(env, observation, world=0, slot=3,
                                        episode_id=episode_id)
    sequence = int(env.steps[0].item())
    if base.observation.sequence != sequence:
        raise ValueError("candidate snapshot does not match current simulator step")
    snapshot = replace(base.observation, episode_id=episode_id,
                       snapshot_id=f"{episode_id}:{sequence}")
    base = SafeActionCandidates(snapshot,
        CandidateTable(snapshot.snapshot_id, base.table.candidate_ids,
            base.table.features, base.table.valid_mask, base.table.feature_names,
            route_costs=base.table.route_costs),
        base.fallback_candidate_id, base.action_indices)
    target_rows = {int(base.table.features[action][1]): base.table.features[action]
                   for action in range(4) if base.table.valid_mask[action] and
                   base.table.features[action][1] is not None}
    specs = [(action, 0, 0., 0.,
              (int(base.table.features[action][1])
               if action < 4 and base.table.features[action][1] is not None else None))
             for action in range(8)]
    additional_targets = _additional_sensed_target_rows(env, base)
    for action, row, track in additional_targets:
        target_rows[track] = row
        specs.append((action, 0, 0., 0., track))
    offsets = ((.22, 0.), (-.22, 0.), (0., .22), (0., -.22),
               (.34, 0.), (-.34, 0.))
    for variant, (forward, lateral) in enumerate(offsets, 1):
        for action in range(4):
            track = (int(base.table.features[action][1])
                     if base.table.features[action][1] is not None else None)
            specs.append((action, variant, forward, lateral, track))
    # Reconcile a carried task against this snapshot before adding it. Intake
    # commitments follow the same sensed target by position, not a transient
    # top-four buffer index. Any lost, illegal, or expired commitment is dropped.
    active_commitment = commitment
    continue_spec = None
    if active_commitment is not None:
        action = active_commitment.action_index
        if sequence >= active_commitment.expires_sequence:
            active_commitment = None
        elif action < 4:
            source = active_commitment.source_xy_m
            matches = []
            if source is not None:
                points, _, sensed = _world_zero_perceived_fuel(env)
                for track in torch.nonzero(sensed, as_tuple=False).flatten().tolist():
                    point = points[track]
                    gap = math.hypot(float(point[0]) - source[0], float(point[1]) - source[1])
                    matches.append((gap, int(track)))
            if not matches or min(matches)[0] > CANDIDATE_BOUNDARY["commitment_target_matching_tolerance_m"]:
                active_commitment = None
            else:
                track = min(matches)[1]
                action = next((i for i in range(4) if base.table.valid_mask[i]), None)
                if action is None:
                    active_commitment = None
                else:
                    continue_spec = (action, active_commitment.goal_variant, 0., 0., track)
        else:
            if not base.table.valid_mask[action]:
                active_commitment = None
            else:
                continue_spec = (action, active_commitment.goal_variant, 0., 0., None)
    continue_index = None
    if continue_spec is not None:
        continue_index = len(specs)
        specs.append(continue_spec)
    oracle_offsets = ((.50, 0.), (-.50, 0.), (0., .50), (0., -.50),
        (.50, .50), (.50, -.50), (-.50, .50), (-.50, -.50),
        (.68, 0.), (-.68, 0.), (0., .68), (0., -.68),
        (.68, .25), (.68, -.25), (-.68, .25), (-.68, -.25),
        (.25, .68), (-.25, .68))
    oracle_start_index = len(specs)
    oracle_track_ids = sorted(target_rows)
    for variant, (forward, lateral) in enumerate(oracle_offsets, 7):
        for track in oracle_track_ids:
            action = next((i for i in range(4) if base.table.valid_mask[i]), None)
            if action is not None:
                specs.append((action, variant, forward, lateral, track))
    targets, headings, route_queries, goal_nodes = [], [], [], []
    capture_estimates = []
    hbins, requested, ids = [], [], []
    for row_index, (action, variant, forward_offset, lateral_offset,
                    target_track) in enumerate(specs):
        candidate_row = target_rows.get(target_track) if target_track is not None else None
        target, heading, allowed, intake = _candidate_target(
            env, action, base, target_row=candidate_row)
        is_continue = continue_index is not None and row_index == continue_index
        if is_continue:
            candidate_id = "continue:" + active_commitment.candidate_id
        elif target_track is not None:
            base_tracks = {int(base.table.features[i][1]) for i in range(4)
                           if base.table.features[i][1] is not None}
            candidate_id = (base.table.candidate_ids[action] if not variant and
                            target_track in base_tracks else
                            f"collect:track:{target_track}" if not variant else
                            f"collect:track:{target_track}:approach:{variant}")
        else:
            candidate_id = (base.table.candidate_ids[action] if not variant else
                            f"{base.table.candidate_ids[action]}:approach:{variant}")
        ids.append(candidate_id)
        if variant:
            forward = target.new_tensor((math.cos(heading), math.sin(heading)))
            left = target.new_tensor((-math.sin(heading), math.cos(heading)))
            target = target + forward * forward_offset + left * lateral_offset
        if action < 4:
            fuel_row = candidate_row or base.table.features[action]
            if fuel_row[2] is not None and fuel_row[3] is not None:
                delta = target.new_tensor((fuel_row[2], fuel_row[3])) - target
                if float(delta.norm()) > 1e-8:
                    heading = math.atan2(float(delta[1]), float(delta[0]))
        node, center, hbin = _goal_node(env, graph, target, heading,
                                        intake_extended=intake)
        allowed &= node is not None
        if center is not None:
            target = center
            if action < 4:
                fuel_row = candidate_row or base.table.features[action]
                if fuel_row[2] is not None and fuel_row[3] is not None:
                    delta = target.new_tensor((fuel_row[2], fuel_row[3])) - target
                    if float(delta.norm()) > 1e-8:
                        heading = math.atan2(float(delta[1]), float(delta[0]))
                    # Resolve the legal heading bin again from the actual
                    # target bearing at the snapped endpoint.
                    node, center, hbin = _goal_node(env, graph, target, heading,
                                                    intake_extended=intake)
                    allowed &= node is not None
                    if center is not None:
                        target = center
        capture_estimate = _capture_pose_estimate(env, base, action, target,
            goal_heading=heading, target_row=candidate_row)
        if action < 4:
            allowed &= bool(capture_estimate["feasible"])
        capture_estimates.append(capture_estimate)
        targets.append(target)
        headings.append(heading)
        goal_nodes.append(node)
        hbins.append(hbin)
        requested.append(allowed)
        if allowed:
            route_queries.append(RouteQuery(candidate_id=candidate_id,
                snapshot_id=base.observation.snapshot_id, world=0, slot=3,
                goal_xy=(float(center[0]), float(center[1])), intake_extended=intake))

    # Validate the complete candidate set before probing any routes. The
    # evaluator intentionally caps each native AD* workspace at 64 rows; split
    # only after global snapshot and duplicate checks so chunk boundaries cannot
    # hide a cross-chunk contract violation.
    if route_queries:
        snapshot_ids = {query.snapshot_id for query in route_queries}
        if len(snapshot_ids) != 1:
            raise ValueError("one route batch must use a single snapshot_id")
        query_keys = [(query.world, query.slot, query.candidate_id)
                      for query in route_queries]
        if len(set(query_keys)) != len(query_keys):
            raise ValueError("route batch contains duplicate candidate keys")

    route_results_list = []
    for offset in range(0, len(route_queries), MAX_ROUTE_QUERY_BATCH):
        route_results_list.extend(evaluate_routes(
            env, route_queries[offset:offset + MAX_ROUTE_QUERY_BATCH],
            goal_tolerance_m=graph.resolution_m * .72,
            endpoint_tolerance_m=graph.resolution_m * .72,
            learned_traversal_cost=learned_traversal_cost,
            learned_cost_scale=learned_cost_scale,
            learned_cost_provenance="meco_legal_heading_penalty_projection_v1"))
    route_results = tuple(route_results_list)
    result_by_id = {result.candidate_id: result for result in route_results}
    speed = max(float(env.sim.speed[0, 3]), .1)
    omega = max(float(env.sim.omega_limit[0, 3]), .1)
    pose_heading = float(env.sim.pose[0, 3, 2])
    rows = []
    valid_mask = []
    routes = []
    route_costs = []
    risk_costs = []
    for row_index, (action, variant, _, _, target_track) in enumerate(specs):
        old = (target_rows[target_track] if target_track is not None
               else base.table.features[action])
        candidate_id = ids[row_index]
        result = result_by_id.get(candidate_id)
        distance = result.distance_m if result is not None and result.feasible else None
        route_ok = bool(requested[row_index] and result is not None and result.feasible and
                        goal_nodes[row_index] is not None)
        turn = abs(math.atan2(math.sin(headings[row_index] - pose_heading),
                              math.cos(headings[row_index] - pose_heading)))
        eta = (None if distance is None else distance / speed + turn / omega
               + action_ticks * float(env.dt))
        if eta is None:
            # An unreachable route has no arrival-time forecast. Keep the
            # missing forecast explicit instead of silently treating it as a
            # zero-second arrival at the current opponent pose.
            forecast = {"xy": None, "mode": prediction_mode,
                "horizon_s": 0. if prediction_mode == "current" else None,
                "observed_age_s": (float(env.opponent_age[0, 3])
                    if math.isfinite(float(env.opponent_age[0, 3])) else None),
                "used_cv": False, "fallback": "not_applicable_unreachable_route"}
        else:
            forecast = _opponent_prediction(env, horizon=eta, mode=prediction_mode)
        predicted = forecast["xy"]
        goal_xy = targets[row_index]
        separation = None if predicted is None else math.hypot(
            float(goal_xy[0]) - predicted[0], float(goal_xy[1]) - predicted[1])
        risk = opponent_risk_penalty(goal_xy, forecast)
        valid = bool(base.table.valid_mask[action] and route_ok)
        valid_mask.append(valid)
        # Update the shared target to the legal route endpoint, and preserve
        # every missing estimate explicitly for the model's missingness mask.
        row = list(old)
        row[2] = float(goal_xy[0])
        row[3] = float(goal_xy[1])
        row[4] = float(hbins[row_index])
        # route_time_s is already part of SAFE_ACTION_FEATURES; replace the
        # coarse base estimate with the route-derived ETA instead of appending
        # a duplicate column name.
        row[13] = eta
        row.extend((None if goal_nodes[row_index] is None else float(goal_nodes[row_index]),
                    distance, risk, float(headings[row_index]), float(is_continue),
                    float(active_commitment is not None and not is_continue)))
        rows.append(tuple(row))
        route_costs.append(0. if eta is None else eta * ROUTE_COST_POINTS_PER_SECOND)
        risk_costs.append(0. if risk is None else risk)
        capture_estimate = capture_estimates[row_index]
        routes.append({"candidate_id": candidate_id,
            "action_index": action,
            "candidate_row": row_index,
            "is_continue_candidate": bool(is_continue),
            "commitment_id": (active_commitment.candidate_id
                              if active_commitment is not None else None),
            "goal_variant": variant,
            "route_status": "not_queried" if result is None else result.status,
            "feasible": valid, "route_feasible": route_ok,
            "capture_estimate": capture_estimate,
            "goal_xy_m": [float(goal_xy[0]), float(goal_xy[1])],
            "source_fuel_xy_m": ([float(old[2]), float(old[3])]
                if action < 4 and old[2] is not None and old[3] is not None else None),
            "sensor_track_buffer_index": (target_track if action < 4 else None),
            "target_identity_execution": ("slot3_revalidated_track_terminal_bearing_with_native_geometric_capture"
                if target_track is not None and action < 4 else "native_action_target"),
            "target_track_snapshot_tolerance_m": (1.0 if target_track is not None
                and action < 4 else None),
            "goal_heading_rad": float(headings[row_index]),
            "legal_node_index": goal_nodes[row_index], "distance_m": distance,
            "eta_s": eta, "predicted_opponent_xy_m": predicted,
            "opponent_prediction": {**forecast,
            "contract": prediction_contract(prediction_mode,
                    horizon_s=(eta if eta is not None else None))},
            "endpoint_separation_m": separation, "risk_penalty_points": risk,
            **({"expected_risk_semantics": "weighted_expectation_over_prediction_hypotheses"}
               if prediction_mode == "multi_model_v1" else {}),
            "intake_extended": action < 4,
            "path_points_m": (None if result is None else result.path_points_m),
            "weighted_route_cost_grid_edges": (None if result is None
                else result.weighted_cost),
            "route_cost_units": (None if result is None else result.cost_units),
            "route_cost_provenance": (None if result is None
                else result.cost_provenance)})
    if continue_index is not None and not valid_mask[continue_index]:
        active_commitment = None
        for index, row in enumerate(rows):
            if index != continue_index:
                updated = list(row)
                updated[-1] = 0.
                rows[index] = tuple(updated)
    selectable_indices = list(range(candidate_k))
    if continue_index is not None and valid_mask[continue_index]:
        selectable_indices.append(continue_index)
    if not any(valid_mask[index] for index in selectable_indices):
        raise RuntimeError("candidate service found no feasible row, including safe fallback")
    fallback = base.fallback_candidate_id
    if fallback not in ids or ids.index(fallback) not in selectable_indices or not valid_mask[ids.index(fallback)]:
        hold = ids.index("hold")
        if valid_mask[hold]:
            fallback = "hold"
        else:
            fallback = min((i for i in selectable_indices if valid_mask[i]),
                           key=lambda i: route_costs[i] + risk_costs[i])
            fallback = ids[fallback]
    names = base.table.feature_names + EXTRA_FEATURES
    # Higher-budget reference: every ordinary candidate plus 18 separately
    # route-qualified offsets for every sensed target. Oracle rows never enter
    # model order and never affect the K8 baseline.
    oracle_ids = {ids[i] for i in range(oracle_start_index)
                  if i != continue_index and valid_mask[i]}
    oracle_ids.update(ids[i] for i in range(oracle_start_index, len(ids)) if valid_mask[i])
    oracle_ids.discard(ids[continue_index] if continue_index is not None else "")
    oracle_capacity = (8 + len(additional_targets) + 4 * len(offsets) +
                       len(oracle_track_ids) * len(oracle_offsets))
    oracle_count = len(oracle_ids)
    requested_ids = {ids[i] for i in range(candidate_k) if valid_mask[i]}
    oracle_indices = [i for i in range(oracle_start_index)
                      if i != continue_index and valid_mask[i]] + [
        i for i in range(oracle_start_index, len(ids)) if valid_mask[i]]
    oracle_cost_values = [route_costs[i] for i in oracle_indices]
    selected_cost_values = [route_costs[i] for i in range(candidate_k) if valid_mask[i]]
    recall = {"candidate_capacity": candidate_k,
              "prediction_mode": prediction_mode,
              "reference_pool_capacity": oracle_capacity,
              "valid_executable_candidates": len(requested_ids),
              "reference_scope": "independent_expanded_goal_oracle",
              "within_candidate_pool_recall_measured": False,
              "route_feasible_rows_in_reference_pool": oracle_count,
              "oracle_route_evaluated": True,
              "candidate_recall": (len(requested_ids & oracle_ids) / oracle_count if oracle_count else 1.),
              "oracle_min_route_cost_points": min(oracle_cost_values) if oracle_cost_values else None,
              "oracle_mean_route_cost_points": (sum(oracle_cost_values) / len(oracle_cost_values)
                                                if oracle_cost_values else None),
              "selected_min_route_cost_points": min(selected_cost_values) if selected_cost_values else None,
              "selected_mean_route_cost_points": (sum(selected_cost_values) / len(selected_cost_values)
                                                  if selected_cost_values else None),
              "recall_limitation": "bounded deterministic expanded-goal oracle; not a simulator-optimality oracle"}
    selected_indices = list(range(candidate_k))
    if continue_index is not None and continue_index in selectable_indices:
        selected_indices.append(continue_index)
    commitment_digest = ("" if active_commitment is None else hashlib.sha256(
        json.dumps({"candidate_id": active_commitment.candidate_id,
            "action_index": active_commitment.action_index,
            "goal_variant": active_commitment.goal_variant,
            "source_xy_m": active_commitment.source_xy_m,
            "created_sequence": active_commitment.created_sequence,
            "expires_sequence": active_commitment.expires_sequence},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    table = CandidateTable(base.observation.snapshot_id,
        tuple(ids[i] for i in selected_indices), tuple(rows[i] for i in selected_indices),
        tuple(valid_mask[i] for i in selected_indices), names,
        route_costs=tuple(route_costs[i] for i in selected_indices),
        schema_version=base.table.schema_version, context_digest=commitment_digest)
    if active_commitment is not None and continue_index is not None:
        active_commitment = replace(active_commitment,
            action_index=int(specs[continue_index][0]))
    return RouteFeasibleCandidates(base.observation, table, fallback,
        tuple(specs[i][0] for i in selected_indices),
        torch.tensor([route_costs[i] for i in selected_indices], device=env.device, dtype=torch.float32),
        torch.tensor([risk_costs[i] for i in selected_indices], device=env.device, dtype=torch.float32),
        tuple(routes[i] for i in selected_indices), recall, active_commitment)
