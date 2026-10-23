"""Full legal-pose stochastic action support for Plan A PPO tracks.

Support order is graph-node major, then behavior index (drive, intake, score,
ferry, defense), then sensed-target slot for intake. Reachability comes from one model-independent shortest-path
tree on the deterministic legal SE(2) graph. Learned P changes action scores,
never support membership or ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
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
    root_node: int
    root_connector_m: float
    fallback_pair: int
    goal_xy: torch.Tensor       # [N,2], frozen legal node endpoints
    goal_heading: torch.Tensor  # [N], frozen legal SE(2) heading endpoints
    snapshot_id: str
    support_digest: str

    @property
    def pair_count(self):
        return int(self.valid_mask.numel())


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
    digest.update(f"root_node={support.root_node};root_connector_m={support.root_connector_m:.17g};"
                  f"fallback_pair={support.fallback_pair}".encode())
    for name, tensor in _plan_a_support_tensors(support, graph).items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def build_plan_a_support(env, graph, observation, snapshot_id: str) -> PlanAActionSupport:
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
    pose = env.sim.pose[0, 3]
    angle = torch.atan2(torch.sin(graph.heading - pose[2]),
                        torch.cos(graph.heading - pose[2])).abs()
    start_cost = (graph.xy - pose[:2]).norm(dim=-1) + angle * .25
    root = int(start_cost.argmin().item())
    edges = graph.edges[0].detach().cpu().tolist()
    edge_masks = graph.edge_mask[0].detach().cpu().tolist()
    xy = graph.xy.detach().cpu()
    headings = graph.heading.detach().cpu()
    xy_values = xy.tolist()
    heading_values = headings.tolist()
    distances = [math.inf] * n
    physical_distances = [math.inf] * n
    parents = [-1] * n
    edge_lengths = [0.] * n
    root_connector_m = float((pose[:2] - graph.xy[root]).norm())
    distances[root] = float(start_cost[root])
    physical_distances[root] = root_connector_m
    queue = [(distances[root], root)]
    while queue:
        distance, src = heapq.heappop(queue)
        if distance != distances[src]:
            continue
        for dst, allowed in zip(edges[src], edge_masks[src]):
            if not allowed:
                continue
            dx = xy_values[src][0] - xy_values[dst][0]
            dy = xy_values[src][1] - xy_values[dst][1]
            translation = math.hypot(dx, dy)
            delta_heading = heading_values[dst] - heading_values[src]
            turn = abs(math.atan2(math.sin(delta_heading), math.cos(delta_heading)))
            cost = translation + .25 * turn
            proposal = distance + cost
            if proposal < distances[dst] - 1e-9:
                distances[dst] = proposal
                physical_distances[dst] = physical_distances[src] + translation
                parents[dst] = src
                edge_lengths[dst] = translation
                heapq.heappush(queue, (proposal, dst))
    order = sorted((i for i, d in enumerate(distances) if math.isfinite(d)),
                   key=lambda i: (distances[i], i))
    reachable = torch.tensor([math.isfinite(d) for d in distances],
                             device=device, dtype=torch.bool)
    action_mask = observation[0, 3, -8:].bool()
    # Pair constraints reflect the native executor's actual goal semantics.
    # Intake rows bind a sensed target slot and require a bearing-aligned
    # capture neighborhood. Score rows bind to a safe score-grid endpoint;
    # ferry rows require carried fuel and bind to the current ferry endpoint;
    # defense rows bind to an observed opponent endpoint. Drive can use any
    # reachable legal node. In Plan A's adapter, action 7 becomes a traversal
    # command only when paired with this explicit frozen planner goal.
    target_indices, target_available, _ = env._candidates()
    fuel_ids = target_indices[0, 3]
    fuel_valid = target_available[0, 3]
    fuel_xy = env.track_pos[0, 3].index_select(0, fuel_ids)
    score_points = env._score_grid_points[0, 3]
    score_clear = env._score_grid_clear[0, 3]
    possession = int((env.piece_owner[0] == 3).sum().item())
    dumper_with_fuel = bool(env._dumper_mask[3]) and possession > 0
    ferry_target = env._ferry_targets_for_current_motion()[0, 3]
    opponent_valid = bool(env.opponent_valid[0, 3])
    robot_xy = env.sim.pose[0, 3, :2]
    opponent_target = env.opponent_pose[0, 3, :2]
    opponent_velocity = env.opponent_velocity[0, 3, :2]
    if opponent_valid:
        enemy_speed = float(opponent_velocity.norm())
        if enemy_speed > .15:
            travel_direction = opponent_velocity / enemy_speed
        else:
            away = robot_xy - opponent_target
            travel_direction = away / away.norm().clamp_min(1e-6)
        intercept_time = float((opponent_target - robot_xy).norm() /
                               max(float(env.sim.speed[0, 3]), .1))
        intercept_time = min(1.25, max(0., intercept_time))
        opponent_target = (opponent_target + opponent_velocity * intercept_time +
                           travel_direction * .55)
    else:
        enemy_team = 1 - int(env.team_ids[3])
        opponent_target = env.hub_centers[enemy_team].clone()
        approach_side = -1. if enemy_team == 0 else 1.
        defender_radius = .5 * torch.sqrt(env.sim.length[0, 3].square() +
                                           env.sim.width[0, 3].square())
        standoff = max(.85, .595 + float(defender_radius) - .15)
        opponent_target[0] += approach_side * standoff
    half_length = float(env.sim.length[0, 3]) * .5
    half_width = float(env.sim.width[0, 3]) * .5
    capture_radius = half_length + .35 + .3048 + float(env._fuel_radius)
    lateral_radius = half_width + .075 + float(env._fuel_radius)
    resolution = float(graph.resolution_m)
    # All geometric predicates are batched over the full node set. Only the
    # resulting small boolean vectors cross to Python for canonical ordering.
    heading_xy = torch.stack((graph.heading.cos(), graph.heading.sin()), -1)
    left_xy = torch.stack((-graph.heading.sin(), graph.heading.cos()), -1)
    delta = fuel_xy[None, :, :] - graph.xy[:, None, :]
    along = (delta * heading_xy[:, None, :]).sum(-1)
    lateral = (delta * left_xy[:, None, :]).sum(-1).abs()
    capture_low = half_length - .075 - float(env._fuel_radius)
    intake_mask = ((along >= capture_low) & (along <= capture_radius) &
                   (lateral <= lateral_radius) & fuel_valid[None, :] &
                   action_mask[:4][None, :] & reachable[:, None])
    safe_points = score_points[score_clear]
    team_id = int(env.team_ids[3])
    hub = env.hub_centers[team_id]
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
    team_nodes = torch.full((1, n), team_id, device=device, dtype=torch.long)
    ferry_aim = ferry_aim_targets(env, graph.xy[None], team_nodes)[0]
    ferry_bearing = torch.atan2(ferry_aim[:, 1] - graph.xy[:, 1],
                                ferry_aim[:, 0] - graph.xy[:, 0])
    ferry_heading_error = torch.atan2(
        torch.sin(graph.heading - ferry_bearing),
        torch.cos(graph.heading - ferry_bearing)).abs()
    ferry_heading_legal = ((not dumper_with_fuel) |
                           (ferry_heading_error <= math.pi / 4.))
    ferry_mask = (((graph.xy - ferry_target).norm(dim=-1) <= resolution * .72) &
                  reachable & action_mask[6] & (possession > 0) & ferry_heading_legal)
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
        reachable & action_mask[7] & (not dumper_with_fuel),
        intake_mask[:, 0], intake_mask[:, 1], intake_mask[:, 2], intake_mask[:, 3],
        score_mask, ferry_mask, defense_mask), -1)
    valid = valid_matrix.reshape(-1) & reachable_rows.reshape(-1)
    if not bool(valid.any()):
        raise RuntimeError("Plan A support has no feasible node/behavior pair")
    # Use the closest graph pose with any deterministic executable behavior,
    # then its first valid descriptor in canonical order.
    node_has_action = valid.reshape(n, 8).any(-1)
    fallback_nodes = torch.nonzero(node_has_action, as_tuple=False).flatten()
    if fallback_nodes.numel() == 0:
        raise RuntimeError("Plan A safe fallback is absent from the support mask")
    fallback_node = int(fallback_nodes[torch.argmin(
        torch.tensor([physical_distances[i] for i in fallback_nodes.tolist()],
                     device=device, dtype=graph.xy.dtype))].item())
    fallback_behaviors = torch.nonzero(valid[fallback_node * 8:(fallback_node + 1) * 8],
                                       as_tuple=False).flatten()
    fallback_pair = fallback_node * 8 + int(fallback_behaviors[0].item())
    parent_tensor = torch.tensor([p if p >= 0 else root for p in parents],
                                 device=device, dtype=torch.long)
    jumps = [parent_tensor]
    while (1 << len(jumps)) < n:
        previous = jumps[-1]
        jumps.append(previous.index_select(0, previous))
    support = PlanAActionSupport(
        node_ids=torch.arange(n, device=device, dtype=torch.long),
        behavior_ids=behavior_ids, action_ids=action_ids,
        pair_node_ids=pair_node_ids, descriptor_ids=descriptor_ids,
        valid_mask=valid,
        distances_m=torch.tensor(physical_distances, device=device, dtype=graph.xy.dtype),
        route_cost=torch.tensor(distances, device=device, dtype=graph.xy.dtype),
        parent_ids=torch.tensor(parents, device=device, dtype=torch.long),
        parent_edge_m=torch.tensor(edge_lengths, device=device, dtype=graph.xy.dtype),
        traversal_order=torch.tensor(order, device=device, dtype=torch.long),
        ancestor_jumps=torch.stack(jumps),
        root_node=root, root_connector_m=root_connector_m,
        fallback_pair=fallback_pair,
        goal_xy=graph.xy.detach().clone(),
        goal_heading=graph.heading.detach().clone(), snapshot_id=snapshot_id,
        support_digest="")
    return PlanAActionSupport(**{
        **support.__dict__,
        "support_digest": plan_a_support_digest(support, graph),
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
    total = total + node_cost[support.root_node] * support.root_connector_m
    return total
