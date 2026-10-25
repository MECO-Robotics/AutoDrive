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
from torch.nn import functional as F


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
    def __init__(self, diagnostic: PlanAEmptySupportDiagnostic | None):
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
        "fallback_pair_tensor": support.fallback_pair_tensor,
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
        "lookup": graph.lookup,
        "edges": graph.edges,
        "edge_mask": graph.edge_mask,
        "node_valid": graph.node_valid,
        "region_ids": graph.region_ids,
        "coarse_edges": graph.coarse_edges,
        "coarse_edge_mask": graph.coarse_edge_mask,
        "graph_status": graph.graph_status,
    }


def plan_a_support_digest(support: PlanAActionSupport, graph) -> str:
    """Hash frozen dynamic graph/support bytes on device after action selection."""
    device = graph.xy.device
    if device.type != "cuda" or torch.version.hip is None:
        raise RuntimeError("dynamic Plan A support SHA-256 requires the HIP device path")
    if not torch.is_tensor(support.root_node) or not torch.is_tensor(support.root_connector_m):
        raise RuntimeError("dynamic Plan A root provenance must remain device-resident")
    if not torch.is_tensor(support.fallback_pair_tensor):
        raise RuntimeError("dynamic Plan A fallback provenance must remain device-resident")
    tensors = {"root_node": support.root_node.reshape(()),
        "root_connector_m": support.root_connector_m.reshape(()),
        "fallback_pair_tensor": support.fallback_pair_tensor,
        **_plan_a_support_tensors(support, graph)}
    header_fields = ["plan-a-support-sha256-v1", support.snapshot_id,
        support.order_digest, f"legacy_fallback={support.fallback_pair}"]
    payload = [torch.tensor(list(("\n".join(header_fields) + "\n").encode()),
                            device=device, dtype=torch.uint8)]
    for name, tensor in tensors.items():
        if tensor is None:
            header_fields.append(f"{name}=none")
            continue
        if tensor.device != device:
            raise RuntimeError(f"Plan A digest tensor {name} is not device-resident")
        header_fields.append(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}")
        payload.append(tensor.detach().contiguous().view(torch.uint8).reshape(-1))
    header = torch.tensor(list(("\n".join(header_fields) + "\n").encode()),
                          device=device, dtype=torch.uint8)
    payload[0] = header
    from .meco_plan_a_shortest_path_hip import extension
    digest_bytes = extension().sha256_bytes(torch.cat(payload).contiguous())
    # This fixed 32-byte copy is only exposed after the chosen-action boundary.
    return bytes(digest_bytes.detach().cpu().tolist()).hex()


def plan_a_support_digest_cpu_reference(support: PlanAActionSupport, graph) -> str:
    """SHA-256 oracle for offline CPU fixtures; never called by live decisions."""
    if graph.xy.device.type != "cpu":
        raise RuntimeError("CPU Plan A digest oracle only accepts CPU reference graphs")
    digest = hashlib.sha256(SUPPORT_ORDER.encode())
    digest.update(support.snapshot_id.encode())
    digest.update(str(support.root_node).encode())
    digest.update(str(support.root_connector_m).encode())
    digest.update(f"fallback_pair={support.fallback_pair}".encode())
    digest.update(support.order_digest.encode())
    for name, tensor in _plan_a_support_tensors(support, graph).items():
        if tensor is None:
            digest.update(f"{name}=none".encode())
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def plan_a_hip_fixture_support_digest_cpu_reference(
        support: PlanAActionSupport, graph) -> str:
    """Preserve the v1 offline route-fixture digest, never used by live code.

    These preweighted HIP parity fixtures do not model the live fixed-capacity
    pose graph. Their historical digest binds their route tree and geometry;
    the required graph-contract tensors are supplied for shape completeness,
    but are deliberately outside this separately versioned fixture digest.
    """
    if graph.xy.device.type != "cpu":
        raise RuntimeError("Plan A HIP fixture digest only accepts CPU artifacts")
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
    fixture_only_names = {"lookup", "node_valid", "region_ids", "coarse_edges",
                          "coarse_edge_mask", "graph_status"}
    for name, tensor in _plan_a_support_tensors(support, graph).items():
        if name in fixture_only_names:
            continue
        if tensor is None:
            digest.update(f"{name}=none".encode())
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def plan_a_support_order_digest(support: PlanAActionSupport, graph) -> str:
    """Bind scorer row order from immutable metadata without reading device data."""
    digest = hashlib.sha256()
    digest.update(SUPPORT_ORDER.encode())
    digest.update(support.snapshot_id.encode())
    digest.update((f"plan-a-order-v2;schema=fixed-capacity-pose-v1;"
                   f"compact=row-major-valid-prefix;Ncap={graph.node_capacity};"
                   f"nx={graph.nx};ny={graph.ny};resolution={float(graph.resolution_m):.17g};"
                   f"headings=8;degree=10;regions={graph.region_count};coarse_width=64;"
                   f"pairs={support.pair_count};rows_per_node=8").encode())
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
    n = graph.node_capacity
    pose = env.policy_pose[0, 3]
    angle = torch.atan2(torch.sin(graph.heading - pose[2]),
                        torch.cos(graph.heading - pose[2])).abs()
    start_cost = (graph.xy - pose[:2]).norm(dim=-1) + angle * .25
    start_cost = start_cost.masked_fill(~graph.node_valid[0], torch.inf)
    root = start_cost.argmin()
    root_connector_m = (pose[:2] - graph.xy.index_select(0,
        root.reshape(1)).reshape(2)).norm().to(torch.float64)
    if require_hip_shortest_path or torch.device(device).type != "cpu":
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
        if torch.device(device).type != "cpu":
            raise RuntimeError("Plan A shortest-path CPU oracle is restricted to CPU references")
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
    reachable = reachable & graph.node_valid[0] & (graph.graph_status == 0)
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
    hub = env.hub_centers.index_select(0, team_id.reshape(1)).reshape(2)
    score_bearing = torch.atan2(hub[1] - graph.xy[:, 1],
                                hub[0] - graph.xy[:, 0])
    score_heading_error = torch.atan2(
        torch.sin(graph.heading - score_bearing),
        torch.cos(graph.heading - score_bearing)).abs()
    score_heading_legal = score_heading_error <= math.radians(5.)
    score_distance = (graph.xy[:, None] - score_points[None]).norm(dim=-1)
    score_distance = score_distance.masked_fill(~score_clear[None], torch.inf).amin(-1)
    score_mask = (score_distance <= resolution * .72) & reachable & action_mask[4] & score_heading_legal
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
    fallback_pair_tensor = torch.where(valid_matrix.any(),
        fallback_node * valid_matrix.shape[1] + fallback_slot,
        torch.full((), -1, device=device, dtype=torch.long))
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


def gather_pose_behavior_logits_batch(track: str, raw, batch, *,
        field_length: float, field_width: float):
    """Batched equivalent of canonical A1/A2/A3 pose-support logits."""
    if track not in ("A1", "A2", "A3") or batch.plan_a is None:
        raise ValueError("batched pose scoring requires an A track PPO batch")
    support = batch.plan_a
    rows = torch.arange(batch.capacity, device=batch.device, dtype=torch.long)[:, None]
    if track == "A1":
        utility = raw["utility"]
        height, width = batch.common.raster.shape[-2:]
        col = (support.xy[..., 0] / float(field_length) * width).floor().long().clamp(0, width - 1)
        row = (support.xy[..., 1] / float(field_width) * height).floor().long().clamp(0, height - 1)
        node_values = utility[rows, support.heading_bins, row, col]
    else:
        node_values = raw["utility"]
    safe_nodes = support.pair_node_ids.clamp(0, batch.node_capacity - 1)
    safe_behaviors = support.behavior_ids.clamp(0, 4)
    pair_utility = node_values.gather(1, safe_nodes)
    behavior = raw["behavior_logits"].gather(1, safe_behaviors)
    return pair_utility + behavior


def _project_plan_a_penalty_batch(track: str, penalty: torch.Tensor, batch, *,
        field_length: float, field_width: float, resolution_m: float):
    """Project per-pose penalty to XY cells with the frozen one-row rules."""
    support = batch.plan_a
    rows = torch.arange(batch.capacity, device=batch.device, dtype=torch.long)[:, None]
    legal_lookup = support.legal_lookup
    legal = legal_lookup >= 0
    safe_lookup = legal_lookup.clamp(0, batch.node_capacity - 1)
    if track in ("A2", "A3"):
        pose_penalty = torch.nan_to_num(penalty).clamp_min(0.)
    elif track == "A1":
        height, width = penalty.shape[-2:]
        x = support.xy[..., 0]
        y = support.xy[..., 1]
        gx = 2. * x / float(field_length) + 1. / width - 1.
        gy = 2. * y / float(field_width) + 1. / height - 1.
        grid = torch.stack((gx, gy), -1).reshape(-1, 1, 1, 2)
        heading_penalty = penalty[rows, support.heading_bins].reshape(
            batch.capacity * batch.node_capacity, 1, height, width)
        sampled = F.grid_sample(torch.nan_to_num(heading_penalty).clamp_min(0.),
            grid, mode="bilinear", padding_mode="border", align_corners=False)
        pose_penalty = sampled.reshape(batch.capacity, batch.node_capacity)
    else:
        raise ValueError("Plan A penalty projection only supports A1/A2/A3")
    heading_maps = pose_penalty.gather(1,
        safe_lookup.reshape(batch.capacity, -1)).reshape(
            batch.capacity, 8, batch.nx, batch.ny)
    counts = legal.sum(1)
    projected = (heading_maps * legal.to(heading_maps.dtype)).sum(1)
    return projected / counts.clamp_min(1).to(projected.dtype)


def plan_a_policy_logits_batch(track: str, raw, batch, *,
        field_length: float, field_width: float, resolution_m: float,
        p_mode: str, p_cost_scale: float, p_points_per_grid_edge: float):
    """Batched A policy distribution and source-parity status tensors."""
    support = batch.plan_a
    logits = gather_pose_behavior_logits_batch(track, raw, batch,
        field_length=field_length, field_width=field_width)
    if p_mode == "learned":
        projected = _project_plan_a_penalty_batch(track, raw["penalty"], batch,
            field_length=field_length, field_width=field_width,
            resolution_m=resolution_m)
        x = (support.xy[..., 0] / float(resolution_m)).floor().long().clamp(0, batch.nx - 1)
        y = (support.xy[..., 1] / float(resolution_m)).floor().long().clamp(0, batch.ny - 1)
        node_cost = projected[torch.arange(batch.capacity, device=batch.device)[:, None], x, y]
        total = node_cost * support.parent_edge_m
        for level in range(support.ancestor_jumps.shape[1]):
            parent = support.ancestor_jumps[:, level].clamp(0, batch.node_capacity - 1)
            total = total + total.gather(1, parent)
        root = support.root_node.clamp(0, batch.node_capacity - 1)
        root_cost = node_cost.gather(1, root[:, None]).squeeze(1)
        node_path_cost = (total + root_cost[:, None] * support.root_connector_m[:, None])
        node_path_cost = (node_path_cost * float(p_cost_scale)
                          * float(p_points_per_grid_edge))
    elif p_mode == "p0":
        node_path_cost = torch.zeros_like(support.route_cost)
    else:
        raise ValueError("Plan A PPO batch p_mode must be learned or p0")
    safe_nodes = support.pair_node_ids.clamp(0, batch.node_capacity - 1)
    route = support.route_cost.gather(1, safe_nodes)
    risk = support.risk_cost.gather(1, safe_nodes)
    path = node_path_cost.gather(1, safe_nodes)
    logits = (logits - route - risk - path).masked_fill(~support.support_valid, -torch.inf)
    status = torch.isfinite(raw["utility"]).all() & torch.isfinite(raw["penalty"]).all()
    if "device_input_valid" in raw:
        status = status & raw["device_input_valid"].all()
    status = status & (torch.isfinite(logits) | ~support.support_valid).all()
    status = status & (~batch.common.slot_valid |
                       support.support_valid.any(dim=1)).all()
    return logits, node_path_cost, status


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
