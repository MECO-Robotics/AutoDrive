"""Full legal-pose stochastic action support for Plan A PPO tracks.

Support order is graph-node major, then behavior index (drive, intake, score,
ferry, defense), then sensed-target slot for intake. Reachability comes from one model-independent shortest-path
tree on the deterministic legal SE(2) graph. Learned P changes action scores,
never support membership or ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch


BEHAVIORS = ("drive", "intake", "score", "ferry", "defense")
SUPPORT_ORDER = "legal_graph_node_ascending_then_behavior_0_to_4_then_descriptor_ascending_v2"


@dataclass(frozen=True)
class PlanAActionSupport:
    """Frozen canonical node × behavior descriptor support and path metadata."""
    node_ids: torch.Tensor       # [N], graph's canonical ascending node IDs
    behavior_ids: torch.Tensor  # [P], canonical pair behavior IDs
    action_ids: torch.Tensor     # [P], simulator action for each pair
    pair_node_ids: torch.Tensor  # [P], graph node for each pair
    descriptor_ids: torch.Tensor  # [P], sensed target slot or -1
    valid_mask: torch.Tensor    # [P], deterministic support mask
    distances_m: torch.Tensor   # [N], root-to-node shortest legal graph distance
    route_cost: torch.Tensor    # [N], deterministic graph edge cost incl. heading change
    parent_ids: torch.Tensor    # [N], rooted shortest-path tree
    parent_edge_m: torch.Tensor  # [N], edge length along that tree
    traversal_order: torch.Tensor  # [N], parent-before-child ordering
    ancestor_jumps: torch.Tensor  # [ceil(log2 N)+1,N], binary-lifting parent table
    root_node: torch.Tensor | int
    root_connector_m: torch.Tensor | float
    # `fallback_pair` remains for old serialized test fixtures. Runtime
    # selection uses the device tensor so choosing a fallback does not read a
    # CUDA/HIP scalar before the action boundary.
    fallback_pair: int
    goal_xy: torch.Tensor       # [N,2], frozen legal node endpoints
    goal_heading: torch.Tensor  # [N], frozen legal SE(2) heading endpoints
    snapshot_id: str
    support_digest: str
    fallback_pair_tensor: torch.Tensor | None = None
    order_digest: str = ""

    @property
    def pair_count(self):
        return int(self.valid_mask.numel())


@dataclass(frozen=True)
class PlanAEmptySupportDiagnostic:
    """Source-state evidence for an observation with no legal Plan A pair."""
    snapshot_id: str
    scenario_id: str | None
    category: str | None
    graph_node_count: int
    reachable_node_count: int
    root_node: int
    root_connector_m: float
    learner_pose_m_rad: tuple[float, float, float]
    legal_action_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]
    support_pair_count: int
    valid_pair_counts_by_behavior: tuple[tuple[str, int], ...]
    available_fuel_targets: int
    safe_score_points: int
    possession_count: int
    dumper_with_fuel: bool
    ferry_target_m: tuple[float, float]
    opponent_valid: bool


class PlanAEmptySupportError(RuntimeError):
    """No action is legal for this frozen Plan A observation."""
    def __init__(self, diagnostic: PlanAEmptySupportDiagnostic):
        self.diagnostic = diagnostic
        super().__init__("Plan A support contains no legal node/behavior pair")


def _plan_a_support_tensors(support: PlanAActionSupport, graph):
    """Return the canonical, ordered tensor set bound to the support digest."""
    return {
        "node_ids": support.node_ids,
        "behavior_ids": support.behavior_ids,
        "action_ids": support.action_ids,
        "pair_node_ids": support.pair_node_ids,
        "descriptor_ids": support.descriptor_ids,
        "valid_mask": support.valid_mask,
        "distances_m": support.distances_m,
        "route_cost": support.route_cost,
        "parent_ids": support.parent_ids,
        "parent_edge_m": support.parent_edge_m,
        "traversal_order": support.traversal_order,
        "ancestor_jumps": support.ancestor_jumps,
        "goal_xy": support.goal_xy,
        "goal_heading": support.goal_heading,
        "heading_bin": graph.heading_bin,
        "graph_xy": graph.xy,
        "edges": graph.edges,
        "edge_mask": graph.edge_mask,
    }


def plan_a_support_digest(support: PlanAActionSupport, graph) -> str:
    """Recompute the integrity digest for frozen PPO support/path metadata."""
    digest = hashlib.sha256(SUPPORT_ORDER.encode())
    digest.update(support.snapshot_id.encode())
    root_node = (int(support.root_node.item()) if torch.is_tensor(support.root_node)
                 else support.root_node)
    root_connector = (float(support.root_connector_m.item())
                      if torch.is_tensor(support.root_connector_m)
                      else support.root_connector_m)
    digest.update(f"root_node={root_node};root_connector_m={root_connector:.17g};"
                  f"fallback_pair={support.fallback_pair}".encode())
    digest.update(support.order_digest.encode())
    for name, tensor in _plan_a_support_tensors(support, graph).items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def plan_a_support_order_digest(support: PlanAActionSupport, graph) -> str:
    """Bind scorer row order from immutable metadata without reading device data."""
    digest = hashlib.sha256()
    digest.update(SUPPORT_ORDER.encode())
    digest.update(support.snapshot_id.encode())
    digest.update((f"plan-a-order-v1;n={graph.node_count};nx={graph.nx};ny={graph.ny};"
                   f"resolution={float(graph.resolution_m):.17g};pairs={support.pair_count};"
                   f"rows_per_node=8").encode())
    return digest.hexdigest()


@torch.no_grad()
def build_plan_a_support(env, graph, observation, snapshot_id: str, *,
                         require_hip_shortest_path: bool = False) -> PlanAActionSupport:
    """Build all reachable legal node × currently legal behavior pairs.

    One Dijkstra pass over the legal SE(2) graph supplies deterministic route
    reachability, distance, parent, and frozen path metadata. This is a graph
    route contract; the executor still performs its current XY AD* planning
    with swept SE(2) checks and native safety arbitration.
    """
    if env.n != 1 or not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("Plan A support requires one world and a snapshot ID")
    device = env.device
    n = graph.node_count
    if n < 1:
        raise ValueError("Plan A legal-pose graph is empty")
    pose = env.policy_pose[0, 3]
    angle = torch.atan2(torch.sin(graph.heading - pose[2]),
                        torch.cos(graph.heading - pose[2])).abs()
    start_cost = (graph.xy - pose[:2]).norm(dim=-1) + angle * .25
    root = start_cost.argmin()
    root_connector_m = (pose[:2] - graph.xy.index_select(0,
        root.reshape(1)).reshape(2)).norm().to(torch.float64)
    if require_hip_shortest_path:
        # This mode is used by the Plan A P-route collector. It must not copy
        # graph edges to Python or fall back to the CPU reference when the
        # exact-heap HIP parity/profile gate is closed.
        from .meco_plan_a_shortest_path import plan_a_dijkstra_hip_for_graph
        paths = plan_a_dijkstra_hip_for_graph(graph, device=device, root=root,
            root_start_cost=start_cost.index_select(0, root.reshape(1)).reshape(()).to(torch.float64),
            root_connector_m=root_connector_m)
    else:
        from .meco_plan_a_shortest_path import (cached_plan_a_edge_weights,
            plan_a_dijkstra_cpu)
        edge_dst, edge_costs, edge_translations, edge_allowed = \
            cached_plan_a_edge_weights(graph)
        paths = plan_a_dijkstra_cpu(edge_dst, edge_costs, edge_translations,
            edge_allowed, root=int(root.item()),
            root_start_cost=float(start_cost[root].item()),
            root_connector_m=float(root_connector_m.item()))
    distances = paths["route_cost"]
    physical_distances = paths["physical_distance"]
    parents = paths["parent_ids"]
    edge_lengths = paths["parent_edge_m"]
    order = paths["traversal_order"]
    if torch.is_tensor(distances):
        reachable = torch.isfinite(distances)
    else:
        reachable = torch.tensor([math.isfinite(d) for d in distances],
                                 device=device, dtype=torch.bool)
    if isinstance(paths, dict) and torch.is_tensor(paths.get("status")):
        reachable = reachable & (paths["status"] == 0)
    action_mask = observation[0, 3, -8:].bool()
    # Pair constraints reflect the native executor's actual goal semantics.
    # Intake rows bind a sensed target slot and require a bearing-aligned
    # capture neighborhood. Score rows bind to a safe score-grid endpoint;
    # ferry rows require carried fuel and bind to the current ferry endpoint;
    # dumper ferry rows are excluded while carrying because its native action
    # can release fuel before reaching that selected endpoint;
    # defense rows bind to an observed opponent endpoint. Drive can use any
    # reachable legal node. In Plan A's adapter, action 7 becomes a traversal
    # command only when paired with this explicit frozen planner goal.
    target_indices, target_available, _ = env._candidates()
    fuel_ids = target_indices[0, 3]
    fuel_valid = target_available[0, 3]
    fuel_xy = env.track_pos[0, 3].index_select(0, fuel_ids)
    score_points = env._score_grid_points[0, 3]
    score_clear = env._score_grid_clear[0, 3]
    possession = (env.piece_owner[0] == 3).sum()
    dumper_flag = torch.as_tensor(env._dumper_mask[3], device=device,
                                  dtype=torch.bool)
    dumper_with_fuel = dumper_flag & (possession > 0)
    ferry_target = env._ferry_targets_for_current_motion()[0, 3]
    opponent_valid = env.opponent_valid[0, 3].bool()
    robot_xy = env.policy_pose[0, 3, :2]
    opponent_target = env.opponent_pose[0, 3, :2]
    opponent_velocity = env.opponent_velocity[0, 3, :2]
    enemy_speed = opponent_velocity.norm()
    away = robot_xy - opponent_target
    travel_direction = torch.where(enemy_speed > .15,
        opponent_velocity / enemy_speed.clamp_min(1e-6),
        away / away.norm().clamp_min(1e-6))
    robot_speed = env.sim.speed[0, 3].clamp_min(.1)
    intercept_time = ((opponent_target - robot_xy).norm() /
                      robot_speed).clamp(0., 1.25)
    observed_opponent_target = (opponent_target + opponent_velocity * intercept_time +
                                travel_direction * .55)
    team_id = torch.as_tensor(env.team_ids[3], device=device, dtype=torch.long)
    enemy_team = 1 - team_id
    fallback_target = env.hub_centers.index_select(0,
        enemy_team.reshape(1)).reshape(2).clone()
    approach_side = torch.where(enemy_team == 0,
        fallback_target.new_tensor(-1.), fallback_target.new_tensor(1.))
    defender_radius = .5 * torch.sqrt(env.sim.length[0, 3].square() +
                                       env.sim.width[0, 3].square())
    standoff = torch.maximum(defender_radius.new_tensor(.85),
                             .595 + defender_radius - .15)
    fallback_target[0] += approach_side * standoff
    opponent_target = torch.where(opponent_valid, observed_opponent_target,
                                  fallback_target)
    half_length = env.sim.length[0, 3] * .5
    half_width = env.sim.width[0, 3] * .5
    fuel_radius = torch.as_tensor(env._fuel_radius, device=device,
                                  dtype=graph.xy.dtype)
    capture_radius = half_length + .35 + .3048 + fuel_radius
    lateral_radius = half_width + .075 + fuel_radius
    resolution = float(graph.resolution_m)
    # All geometric predicates are batched over the full node set. Only the
    # resulting small boolean vectors cross to Python for canonical ordering.
    heading_xy = torch.stack((graph.heading.cos(), graph.heading.sin()), -1)
    left_xy = torch.stack((-graph.heading.sin(), graph.heading.cos()), -1)
    delta = fuel_xy[None, :, :] - graph.xy[:, None, :]
    along = (delta * heading_xy[:, None, :]).sum(-1)
    lateral = (delta * left_xy[:, None, :]).sum(-1).abs()
    capture_low = half_length - .075 - fuel_radius
    intake_mask = ((along >= capture_low) & (along <= capture_radius) &
                   (lateral <= lateral_radius) & fuel_valid[None, :] &
                   action_mask[:4][None, :] & reachable[:, None])
    safe_points = score_points[score_clear]
    hub = env.hub_centers.index_select(0, team_id.reshape(1)).reshape(2)
    score_bearing = torch.atan2(hub[1] - graph.xy[:, 1],
                                hub[0] - graph.xy[:, 0])
    score_heading_error = torch.atan2(
        torch.sin(graph.heading - score_bearing),
        torch.cos(graph.heading - score_bearing)).abs()
    score_heading_legal = score_heading_error <= math.radians(5.)
    if safe_points.numel():
        score_mask = ((graph.xy[:, None] - safe_points[None]).norm(dim=-1).amin(-1)
                      <= resolution * .72) & reachable & action_mask[4] & score_heading_legal
    else:
        score_mask = torch.zeros((n,), device=device, dtype=torch.bool)
    from .fuel_physics_runtime import ferry_aim_targets
    team_nodes = team_id.expand(1, n)
    ferry_aim = ferry_aim_targets(env, graph.xy[None], team_nodes)[0]
    ferry_bearing = torch.atan2(ferry_aim[:, 1] - graph.xy[:, 1],
                                ferry_aim[:, 0] - graph.xy[:, 0])
    ferry_heading_error = torch.atan2(
        torch.sin(graph.heading - ferry_bearing),
        torch.cos(graph.heading - ferry_bearing)).abs()
    ferry_heading_legal = (~dumper_with_fuel) | (ferry_heading_error <= math.pi / 4.)
    ferry_mask = (((graph.xy - ferry_target).norm(dim=-1) <= resolution * .72) &
                  reachable & action_mask[6] & (possession > 0) & ferry_heading_legal &
                  ~dumper_with_fuel)
    defense_mask = (((graph.xy - opponent_target).norm(dim=-1) <= resolution * .72) &
                    reachable & action_mask[5])
    # Fixed eight-column descriptor layout per graph node: drive, four
    # target-bound intake choices, score, ferry, defense. Invalid rows remain
    # present with a false mask, so pair IDs and tensor shape are canonical
    # across observations and independent of the learned model.
    pair_node_ids = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(8)
    behavior_ids = torch.tensor((0, 1, 1, 1, 1, 2, 3, 4),
                                device=device, dtype=torch.long).repeat(n)
    action_ids = torch.tensor((7, 0, 1, 2, 3, 4, 6, 5),
                              device=device, dtype=torch.long).repeat(n)
    descriptor_ids = torch.tensor((-1, 0, 1, 2, 3, -1, -1, -1),
                                  device=device, dtype=torch.long).repeat(n)
    reachable_rows = reachable[:, None].expand(-1, 8)
    valid_matrix = torch.stack((
        # A loaded dumper may still select a legal traversal goal. The native
        # action is only allowed to move through the paired explicit-goal
        # override in the Plan A adapter; without that override action 7 holds.
        reachable & action_mask[7],
        intake_mask[:, 0], intake_mask[:, 1], intake_mask[:, 2], intake_mask[:, 3],
        score_mask, ferry_mask, defense_mask), -1)
    valid = valid_matrix.reshape(-1) & reachable_rows.reshape(-1)
    if not bool(valid.any()):
        counts = valid_matrix.sum(dim=0).detach().cpu().tolist()
        pose_values = tuple(float(value) for value in pose.detach().cpu().tolist())
        legal_ids = tuple(int(value) for value in torch.nonzero(
            action_mask, as_tuple=False).flatten().detach().cpu().tolist())
        ferry_values = tuple(float(value) for value in ferry_target.detach().cpu().tolist())
        if len(ferry_values) != 2 or not all(math.isfinite(value) for value in ferry_values):
            raise RuntimeError("Plan A empty-support ferry target is not a finite XY pair")
        curriculum = getattr(env, "curriculum", {})
        diagnostic = PlanAEmptySupportDiagnostic(
            snapshot_id=snapshot_id,
            scenario_id=curriculum.get("scenario_id"),
            category=curriculum.get("category"),
            graph_node_count=n,
            reachable_node_count=int(reachable.sum().item()),
            root_node=int(root.item()),
            root_connector_m=float(root_connector_m.item()),
            learner_pose_m_rad=pose_values,
            legal_action_ids=legal_ids,
            action_mask=tuple(bool(value) for value in action_mask.detach().cpu().tolist()),
            support_pair_count=int(valid.sum().item()),
            valid_pair_counts_by_behavior=tuple(zip(
                ("drive", "intake_0", "intake_1", "intake_2", "intake_3",
                 "score", "ferry", "defense"), (int(value) for value in counts))),
            available_fuel_targets=int(fuel_valid.sum().item()),
            safe_score_points=int(score_clear.sum().item()),
            possession_count=int(possession.item()),
            dumper_with_fuel=bool(dumper_with_fuel.item()),
            ferry_target_m=ferry_values,
            opponent_valid=bool(opponent_valid.item()))
        raise PlanAEmptySupportError(diagnostic)
    # Use the closest graph pose with any deterministic executable behavior,
    # then its first valid descriptor in canonical order.
    physical_tensor = (physical_distances.to(device=device, dtype=graph.xy.dtype)
        if torch.is_tensor(physical_distances) else
        torch.tensor(physical_distances, device=device, dtype=graph.xy.dtype))
    # argmin gives the first node on ties; argmax over 0/1 slots gives the
    # first behavior/descriptor in canonical order.
    fallback_node = physical_tensor.masked_fill(
        ~valid_matrix.any(dim=-1), torch.inf).argmin()
    fallback_slot = valid_matrix[fallback_node].to(torch.long).argmax()
    fallback_pair_tensor = fallback_node * valid_matrix.shape[1] + fallback_slot
    fallback_pair = -1
    parent_tensor = torch.where(parents >= 0, parents,
                                torch.full_like(parents, root))
    jumps = [parent_tensor]
    while (1 << len(jumps)) < n:
        previous = jumps[-1]
        jumps.append(previous.index_select(0, previous))
    support = PlanAActionSupport(
        node_ids=torch.arange(n, device=device, dtype=torch.long),
        behavior_ids=behavior_ids, action_ids=action_ids,
        pair_node_ids=pair_node_ids, descriptor_ids=descriptor_ids,
        valid_mask=valid,
        distances_m=(physical_distances.to(device=device, dtype=graph.xy.dtype)
            if torch.is_tensor(physical_distances) else
            torch.tensor(physical_distances, device=device, dtype=graph.xy.dtype)),
        route_cost=(distances.to(device=device, dtype=graph.xy.dtype)
            if torch.is_tensor(distances) else
            torch.tensor(distances, device=device, dtype=graph.xy.dtype)),
        parent_ids=(parents.to(device=device, dtype=torch.long)
            if torch.is_tensor(parents) else
            torch.tensor(parents, device=device, dtype=torch.long)),
        parent_edge_m=(edge_lengths.to(device=device, dtype=graph.xy.dtype)
            if torch.is_tensor(edge_lengths) else
            torch.tensor(edge_lengths, device=device, dtype=graph.xy.dtype)),
        traversal_order=(order.to(device=device, dtype=torch.long)
            if torch.is_tensor(order) else
            torch.tensor(order, device=device, dtype=torch.long)),
        ancestor_jumps=torch.stack(jumps),
        root_node=root, root_connector_m=root_connector_m,
        fallback_pair=fallback_pair,
        goal_xy=graph.xy.detach().clone(),
        goal_heading=graph.heading.detach().clone(), snapshot_id=snapshot_id,
        support_digest="", fallback_pair_tensor=fallback_pair_tensor.reshape(()))
    return PlanAActionSupport(**{
        **support.__dict__,
        "order_digest": plan_a_support_order_digest(support, graph),
    })


def gather_pose_behavior_logits(track: str, raw, graph, raster, support: PlanAActionSupport,
                                *, field_length: float, field_width: float):
    """Vectorized `U(node)+behavior_logit` over the complete frozen support."""
    if track == "A1":
        utility = raw["utility"]
        h, w = raster.shape[-2:]
        col = (graph.xy[:, 0] / field_length * w).floor().long().clamp(0, w - 1)
        row = (graph.xy[:, 1] / field_width * h).floor().long().clamp(0, h - 1)
        node_values = utility[0, graph.heading_bin, row, col]
    elif track in ("A2", "A3"):
        node_values = raw["utility"][0]
    else:
        raise ValueError("full pose support is only defined for A1/A2/A3")
    behavior = raw["behavior_logits"][0]
    logits = node_values[support.pair_node_ids] + behavior[support.behavior_ids]
    return logits.masked_fill(~support.valid_mask, -torch.inf)


def graph_path_integral(node_cost: torch.Tensor, support: PlanAActionSupport):
    """Differentiable integral on the frozen shortest-path tree for every node."""
    if node_cost.ndim != 1 or node_cost.numel() != support.node_ids.numel():
        raise ValueError("node costs must match the frozen legal graph")
    total = node_cost * support.parent_edge_m
    # Double the covered ancestor span each round. Keep the jump table and
    # aggregate recurrence aligned so shared path prefixes are counted once.
    for parent in support.ancestor_jumps:
        total = total + total.index_select(0, parent)
    root_node = support.root_node
    if torch.is_tensor(root_node):
        root_node = root_node.to(device=node_cost.device, dtype=torch.long)
    root_connector = support.root_connector_m
    if torch.is_tensor(root_connector):
        root_connector = root_connector.to(device=node_cost.device,
                                            dtype=node_cost.dtype)
    total = total + node_cost[root_node] * root_connector
    return total
