"""Versioned fixed-capacity device batch contract for five-track PPO.

This module deliberately contains no simulator objects, per-transition mappings,
candidate/support labels, or CPU numeric optimizer.  Construction captures only
static layout metadata; dynamic row state is stored in device tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch


PPO_BATCH_SCHEMA_VERSION = 3
PPO_BATCH_SCHEMA_ID = "five-track-ppo-fixed-capacity-v3"
PPO_OPTIMIZER_SCHEMA_ID = "sequential-adam-masked-noop-v1"
PPO_RNG_SCHEMA_ID = "hip-generator-action-permutation-v1"
PPO_HIDDEN_WIDTH = 64
PPO_GRAPH_DEGREE = 10
PPO_REGION_COUNT = 64
PPO_SUPPORT_ROWS_PER_NODE = 8
PPO_CANDIDATE_MAX = 33
PPO_SUPPORT_DIGEST_BYTES = 32


@dataclass
class CommonPpoTensors:
    raster: torch.Tensor
    globals: torch.Tensor
    hidden_before: torch.Tensor
    action: torch.Tensor
    old_logprob: torch.Tensor
    old_value: torch.Tensor
    next_value: torch.Tensor
    reward: torch.Tensor
    episode_boundary: torch.Tensor
    execution_failure: torch.Tensor
    slot_valid: torch.Tensor
    actor_valid: torch.Tensor
    critic_valid: torch.Tensor
    replay_digest: torch.Tensor


@dataclass
class PlanAPpoTensors:
    node_ids: torch.Tensor
    node_features: torch.Tensor
    xy: torch.Tensor
    heading_bins: torch.Tensor
    headings: torch.Tensor
    node_valid: torch.Tensor
    edges: torch.Tensor
    edge_mask: torch.Tensor
    region_ids: torch.Tensor
    coarse_edges: torch.Tensor
    coarse_edge_mask: torch.Tensor
    legal_lookup: torch.Tensor
    graph_status: torch.Tensor
    distances_m: torch.Tensor
    support_route_cost: torch.Tensor
    pair_node_ids: torch.Tensor
    behavior_ids: torch.Tensor
    action_ids: torch.Tensor
    descriptor_ids: torch.Tensor
    support_valid: torch.Tensor
    old_action_probs: torch.Tensor
    route_cost: torch.Tensor
    risk_cost: torch.Tensor
    parent_edge_m: torch.Tensor
    parent_ids: torch.Tensor
    traversal_order: torch.Tensor
    ancestor_jumps: torch.Tensor
    root_node: torch.Tensor
    root_connector_m: torch.Tensor
    fallback_pair: torch.Tensor
    goal_xy: torch.Tensor
    goal_heading: torch.Tensor


@dataclass
class PlanCPpoTensors:
    candidate_features: torch.Tensor
    raw_features: torch.Tensor
    missing_mask: torch.Tensor
    candidate_ids: torch.Tensor
    action_ids: torch.Tensor
    target_track_ids: torch.Tensor
    target_xy: torch.Tensor
    goal_xy: torch.Tensor
    goal_heading: torch.Tensor
    route_override_flags: torch.Tensor
    candidate_present: torch.Tensor
    candidate_valid: torch.Tensor
    old_action_probs: torch.Tensor
    fallback_index: torch.Tensor
    route_cost: torch.Tensor
    risk_cost: torch.Tensor


@dataclass
class FiveTrackPpoBatchV2:
    """One rollout's homogeneous-track, fixed-capacity tensor payload."""

    track: str
    device: torch.device
    capacity: int
    node_capacity: int
    pair_capacity: int
    candidate_capacity: int
    nx: int
    ny: int
    feature_count: int
    common: CommonPpoTensors
    plan_a: PlanAPpoTensors | None
    plan_c: PlanCPpoTensors | None

    @classmethod
    def allocate(cls, *, track: str, capacity: int, device,
                 raster_shape: tuple[int, int], node_capacity: int,
                 nx: int, ny: int, feature_count: int = 0,
                 dtype: torch.dtype = torch.float32) -> "FiveTrackPpoBatchV2":
        if track not in ("A1", "A2", "A3", "C1", "C2"):
            raise ValueError("PPO batch track must be A1, A2, A3, C1, or C2")
        if (type(capacity) is not int or capacity < 1 or
                type(node_capacity) is not int or node_capacity < 1 or
                type(nx) is not int or nx < 1 or type(ny) is not int or ny < 1):
            raise ValueError("PPO batch capacities must be positive static integers")
        if track in ("A1", "A2", "A3") and node_capacity != 8 * nx * ny:
            raise ValueError("Plan A node capacity must equal eight headings times nx times ny")
        device = torch.device(device)
        if device.type != "cuda" or torch.version.hip is None:
            raise RuntimeError("five-track PPO batch v2 requires an assigned HIP device")
        height, width = raster_shape
        if height < 1 or width < 1:
            raise ValueError("raster dimensions must be positive")
        if track in ("C1", "C2") and (type(feature_count) is not int or feature_count < 1):
            raise ValueError("C-track PPO batches require a positive feature count")

        R, N, P = capacity, node_capacity, node_capacity * PPO_SUPPORT_ROWS_PER_NODE
        K = PPO_CANDIDATE_MAX

        def empty(shape, *, dtype_=dtype):
            return torch.zeros(shape, device=device, dtype=dtype_)

        common = CommonPpoTensors(
            raster=empty((R, 6, height, width)), globals=empty((R, 7)),
            hidden_before=empty((R, PPO_HIDDEN_WIDTH)),
            action=empty((R,), dtype_=torch.long),
            old_logprob=empty((R,)), old_value=empty((R,)), next_value=empty((R,)),
            reward=empty((R,)),
            episode_boundary=empty((R,), dtype_=torch.bool),
            execution_failure=empty((R,), dtype_=torch.bool),
            slot_valid=empty((R,), dtype_=torch.bool),
            actor_valid=empty((R,), dtype_=torch.bool),
            critic_valid=empty((R,), dtype_=torch.bool),
            replay_digest=empty((R, PPO_SUPPORT_DIGEST_BYTES), dtype_=torch.uint8))

        plan_a = None
        if track in ("A1", "A2", "A3"):
            jump_count = max(1, (N - 1).bit_length() + 1)
            plan_a = PlanAPpoTensors(
                node_ids=empty((R, N), dtype_=torch.long),
                node_features=empty((R, N, 9)), xy=empty((R, N, 2)),
                heading_bins=empty((R, N), dtype_=torch.long),
                headings=empty((R, N)),
                node_valid=empty((R, N), dtype_=torch.bool),
                edges=empty((R, N, PPO_GRAPH_DEGREE), dtype_=torch.long),
                edge_mask=empty((R, N, PPO_GRAPH_DEGREE), dtype_=torch.bool),
                region_ids=empty((R, N), dtype_=torch.long),
                coarse_edges=empty((R, PPO_REGION_COUNT, PPO_REGION_COUNT), dtype_=torch.long),
                coarse_edge_mask=empty((R, PPO_REGION_COUNT, PPO_REGION_COUNT), dtype_=torch.bool),
                legal_lookup=empty((R, 8, nx, ny), dtype_=torch.long),
                graph_status=empty((R,), dtype_=torch.long),
                distances_m=empty((R, N)), support_route_cost=empty((R, N)),
                pair_node_ids=empty((R, P), dtype_=torch.long),
                behavior_ids=empty((R, P), dtype_=torch.long),
                action_ids=empty((R, P), dtype_=torch.long),
                descriptor_ids=empty((R, P), dtype_=torch.long),
                support_valid=empty((R, P), dtype_=torch.bool),
                old_action_probs=empty((R, P)),
                route_cost=empty((R, N)), risk_cost=empty((R, N)),
                parent_edge_m=empty((R, N)),
                parent_ids=empty((R, N), dtype_=torch.long),
                traversal_order=empty((R, N), dtype_=torch.long),
                ancestor_jumps=empty((R, jump_count, N), dtype_=torch.long),
                root_node=empty((R,), dtype_=torch.long),
                root_connector_m=empty((R,)), fallback_pair=empty((R,), dtype_=torch.long),
                goal_xy=empty((R, N, 2)), goal_heading=empty((R, N)))

        plan_c = None
        if track in ("C1", "C2"):
            F = feature_count
            plan_c = PlanCPpoTensors(
                candidate_features=empty((R, K, 2 * F)),
                raw_features=empty((R, K, F)),
                missing_mask=empty((R, K, F)),
                candidate_ids=empty((R, K), dtype_=torch.long),
                action_ids=empty((R, K), dtype_=torch.long),
                target_track_ids=empty((R, K), dtype_=torch.long),
                target_xy=empty((R, K, 2)), goal_xy=empty((R, K, 2)),
                goal_heading=empty((R, K)),
                route_override_flags=empty((R, K), dtype_=torch.bool),
                candidate_present=empty((R, K), dtype_=torch.bool),
                candidate_valid=empty((R, K), dtype_=torch.bool),
                old_action_probs=empty((R, K)),
                fallback_index=empty((R,), dtype_=torch.long),
                route_cost=empty((R, K)), risk_cost=empty((R, K)))

        return cls(track, device, R, N, P, K if plan_c is not None else 0,
                   nx, ny, feature_count, common, plan_a, plan_c)

    def validate_static(self, *, track: str, device, capacity: int,
                        node_capacity: int, nx: int, ny: int) -> None:
        """Validate only host-known schema metadata, never tensor values."""
        expected_device = torch.device(device)
        if (track != self.track or expected_device != self.device or
                capacity != self.capacity or node_capacity != self.node_capacity or
                nx != self.nx or ny != self.ny):
            raise RuntimeError("PPO batch v2 static schema/device mismatch")
        for group in (self.common, self.plan_a, self.plan_c):
            if group is None:
                continue
            for field in fields(group):
                value = getattr(group, field.name)
                if value.device != self.device:
                    raise RuntimeError(f"PPO batch field {field.name} left its assigned device")

    def select_rows(self, indices: torch.Tensor) -> "FiveTrackPpoBatchV2":
        """Gather a tensor-selected view; indices are never read by Python."""
        if indices.device != self.device or indices.dtype != torch.long or indices.ndim != 1:
            raise RuntimeError("PPO row selection must be a device int64 vector")

        def select_group(group):
            if group is None:
                return None
            return type(group)(**{
                field.name: getattr(group, field.name).index_select(0, indices)
                for field in fields(group)})

        return FiveTrackPpoBatchV2(
            track=self.track, device=self.device, capacity=indices.shape[0],
            node_capacity=self.node_capacity, pair_capacity=self.pair_capacity,
            candidate_capacity=self.candidate_capacity, nx=self.nx, ny=self.ny,
            feature_count=self.feature_count, common=select_group(self.common),
            plan_a=select_group(self.plan_a), plan_c=select_group(self.plan_c))

    def device_contract_status(self) -> torch.Tensor:
        """Combine dynamic row, mask, range and finite checks on the HIP device."""
        common = self.common
        active = common.slot_valid
        status = (common.actor_valid == (active & ~common.execution_failure)).all()
        status = status & (common.critic_valid == active).all()
        status = status & torch.isfinite(common.raster).all()
        status = status & torch.isfinite(common.globals).all()
        status = status & torch.isfinite(common.hidden_before).all()
        status = status & (torch.isfinite(common.old_logprob) | ~active).all()
        status = status & (torch.isfinite(common.old_value) | ~active).all()
        status = status & (torch.isfinite(common.next_value) | ~active).all()
        status = status & (torch.isfinite(common.reward) | ~active).all()
        if self.plan_a is not None:
            a = self.plan_a
            status = status & ((common.action >= 0) &
                (common.action < self.pair_capacity) | ~active).all()
            safe_pair_nodes = a.pair_node_ids.clamp(0, self.node_capacity - 1)
            pair_route = a.route_cost.gather(1, safe_pair_nodes)
            pair_risk = a.risk_cost.gather(1, safe_pair_nodes)
            status = status & ((a.pair_node_ids >= 0) &
                (a.pair_node_ids < self.node_capacity) | ~a.support_valid).all()
            status = status & ((a.behavior_ids >= 0) & (a.behavior_ids < 5) |
                               ~a.support_valid).all()
            status = status & ((torch.isfinite(pair_route) & torch.isfinite(pair_risk)) |
                               ~a.support_valid).all()
            status = status & (torch.isfinite(a.old_action_probs) &
                (a.old_action_probs >= 0.) & (a.old_action_probs <= 1.) |
                ~common.slot_valid[:, None]).all()
            status = status & ((a.old_action_probs.sum(dim=1) - 1.).abs() <= 5e-5 |
                               ~active).all()
            status = status & (~active | a.support_valid.any(dim=1)).all()
            status = status & ((a.graph_status == 0) | ~active).all()
            safe_fallback = a.fallback_pair.clamp(0, self.pair_capacity - 1)
            fallback_valid = a.support_valid.gather(1, safe_fallback[:, None]).squeeze(1)
            status = status & ((a.fallback_pair >= 0) & fallback_valid | ~active).all()
        if self.plan_c is not None:
            c = self.plan_c
            status = status & ((common.action >= 0) &
                (common.action < self.candidate_capacity) | ~active).all()
            status = status & ((torch.isfinite(c.raw_features) &
                                torch.isfinite(c.missing_mask)) |
                               ~c.candidate_present[..., None]).all()
            status = status & ((torch.isfinite(c.route_cost) &
                                torch.isfinite(c.risk_cost)) |
                               ~c.candidate_valid).all()
            status = status & (torch.isfinite(c.old_action_probs) &
                (c.old_action_probs >= 0.) & (c.old_action_probs <= 1.) |
                ~common.slot_valid[:, None]).all()
            status = status & ((c.old_action_probs.sum(dim=1) - 1.).abs() <= 5e-5 |
                               ~active).all()
            status = status & (~active | c.candidate_valid.any(dim=1)).all()
            status = status & (~c.candidate_valid | c.candidate_present).all()
            status = status & (~c.candidate_present[:, 1:] |
                               c.candidate_present[:, :-1]).all()
            safe_fallback = c.fallback_index.clamp(0, self.candidate_capacity - 1)
            fallback_valid = c.candidate_valid.gather(1, safe_fallback[:, None]).squeeze(1)
            status = status & ((c.fallback_index >= 0) &
                (c.fallback_index < self.candidate_capacity) & fallback_valid | ~active).all()
            status = status & ((c.action_ids >= 0) & (c.action_ids < 8) |
                               ~c.candidate_valid).all()
        return status


def write_common_slot(batch: FiveTrackPpoBatchV2, slot: int, *, raster: torch.Tensor,
                      globals_: torch.Tensor, hidden_before: torch.Tensor | None,
                      action: torch.Tensor, old_logprob: torch.Tensor,
                      old_value: torch.Tensor, next_value: torch.Tensor,
                      reward: torch.Tensor,
                      episode_boundary: torch.Tensor,
                      execution_failure: torch.Tensor) -> None:
    """Write one predetermined rollout slot without inspecting tensor values."""
    if type(slot) is not int or not 0 <= slot < batch.capacity:
        raise IndexError("PPO rollout slot is outside its fixed capacity")
    common = batch.common
    common.raster[slot].copy_(raster.reshape_as(common.raster[slot]))
    common.globals[slot].copy_(globals_.reshape_as(common.globals[slot]))
    if hidden_before is not None:
        common.hidden_before[slot].copy_(hidden_before.reshape_as(common.hidden_before[slot]))
    else:
        common.hidden_before[slot].zero_()
    common.action[slot].copy_(action.reshape(()).to(dtype=torch.long))
    common.old_logprob[slot].copy_(old_logprob.reshape(()))
    common.old_value[slot].copy_(old_value.reshape(()))
    common.next_value[slot].copy_(next_value.reshape(()))
    common.reward[slot].copy_(reward.reshape(()))
    if (episode_boundary.device != batch.device or execution_failure.device != batch.device or
            episode_boundary.dtype != torch.bool or execution_failure.dtype != torch.bool):
        raise RuntimeError("PPO boundary/failure flags must be device boolean tensors")
    common.episode_boundary[slot].copy_(episode_boundary.reshape(()))
    common.execution_failure[slot].copy_(execution_failure.reshape(()))
    common.slot_valid[slot] = True
    common.actor_valid[slot].copy_(~execution_failure.reshape(()))
    common.critic_valid[slot] = True


def write_plan_a_slot(batch: FiveTrackPpoBatchV2, slot: int, *, graph,
                      support, graph_node_features: torch.Tensor,
                      route_cost: torch.Tensor, risk_cost: torch.Tensor,
                      old_action_probs: torch.Tensor) -> None:
    """Copy fixed-shape A support/graph tensors into the chosen static slot."""
    if batch.plan_a is None or type(slot) is not int or not 0 <= slot < batch.capacity:
        raise ValueError("invalid Plan A PPO slot")
    destination = batch.plan_a
    fields_to_copy = (
        (destination.node_features, graph_node_features.reshape_as(destination.node_features[slot])),
        (destination.node_ids, support.node_ids),
        (destination.xy, graph.xy),
        (destination.heading_bins, graph.heading_bin),
        (destination.headings, graph.heading),
        (destination.node_valid, graph.node_valid.reshape_as(destination.node_valid[slot])),
        (destination.edges, graph.edges.reshape_as(destination.edges[slot])),
        (destination.edge_mask, graph.edge_mask.reshape_as(destination.edge_mask[slot])),
        (destination.region_ids, graph.region_ids.reshape_as(destination.region_ids[slot])),
        (destination.coarse_edges, graph.coarse_edges.reshape_as(destination.coarse_edges[slot])),
        (destination.coarse_edge_mask, graph.coarse_edge_mask.reshape_as(destination.coarse_edge_mask[slot])),
        (destination.legal_lookup, graph.lookup),
        (destination.graph_status, graph.graph_status.reshape(())),
        (destination.distances_m, support.distances_m),
        (destination.support_route_cost, support.route_cost),
        (destination.pair_node_ids, support.pair_node_ids),
        (destination.behavior_ids, support.behavior_ids),
        (destination.action_ids, support.action_ids),
        (destination.descriptor_ids, support.descriptor_ids),
        (destination.support_valid, support.valid_mask),
        (destination.old_action_probs, old_action_probs.reshape_as(destination.old_action_probs[slot])),
        (destination.route_cost, route_cost.reshape_as(destination.route_cost[slot])),
        (destination.risk_cost, risk_cost.reshape_as(destination.risk_cost[slot])),
        (destination.parent_edge_m, support.parent_edge_m),
        (destination.parent_ids, support.parent_ids),
        (destination.traversal_order, support.traversal_order),
        (destination.ancestor_jumps, support.ancestor_jumps),
        (destination.root_connector_m, support.root_connector_m.reshape(())),
        (destination.goal_xy, support.goal_xy),
        (destination.goal_heading, support.goal_heading),
    )
    for target, source in fields_to_copy:
        target[slot].copy_(source)
    root = support.root_node
    fallback = support.fallback_pair_tensor
    if not isinstance(root, torch.Tensor) or not isinstance(fallback, torch.Tensor):
        raise RuntimeError("Plan A root and fallback identities must be device tensors")
    destination.root_node[slot].copy_(root.reshape(()))
    destination.fallback_pair[slot].copy_(fallback.reshape(()))


def write_plan_c_slot(batch: FiveTrackPpoBatchV2, slot: int, *, device_batch,
                      route_cost: torch.Tensor, risk_cost: torch.Tensor,
                      old_action_probs: torch.Tensor) -> None:
    """Copy the frozen C device payload, padding only by the preallocated zeros."""
    if batch.plan_c is None or type(slot) is not int or not 0 <= slot < batch.capacity:
        raise ValueError("invalid Plan C PPO slot")
    destination = batch.plan_c
    count = device_batch.candidate_index.shape[-1]
    if count > batch.candidate_capacity:
        raise RuntimeError("Plan C candidates exceed the fixed PPO capacity")
    source_fields = (
        (destination.candidate_features, device_batch.candidate_features),
        (destination.raw_features, device_batch.raw_features),
        (destination.missing_mask, device_batch.missing_mask),
        (destination.candidate_ids, device_batch.candidate_index),
        (destination.action_ids, device_batch.action_index),
        (destination.target_track_ids, device_batch.target_track_index),
        (destination.target_xy, device_batch.target_xy),
        (destination.goal_xy, device_batch.goal_xy),
        (destination.goal_heading, device_batch.goal_heading),
        (destination.route_override_flags, device_batch.route_override_flags),
        (destination.candidate_present, torch.ones((count,), device=batch.device,
                                                     dtype=torch.bool)),
        (destination.candidate_valid, device_batch.valid_mask),
        (destination.old_action_probs, old_action_probs),
        (destination.route_cost, route_cost),
        (destination.risk_cost, risk_cost),
    )
    for target, source in source_fields:
        row = source.reshape((count, *source.shape[2:])) if source.ndim > 2 and source.shape[0] == 1 else source.reshape(count, *source.shape[1:])
        target[slot, :count].copy_(row)
    destination.fallback_index[slot].copy_(device_batch.fallback_index.reshape(()))


def _slot_digest_payload(batch: FiveTrackPpoBatchV2, slot: int) -> torch.Tensor:
    """Serialize one slot from tensors only; metadata is fixed schema text."""
    if type(slot) is not int or not 0 <= slot < batch.capacity:
        raise IndexError("PPO digest slot is outside its fixed capacity")
    metadata = (f"{PPO_BATCH_SCHEMA_ID}\ntrack={batch.track}\n"
                f"R={batch.capacity};N={batch.node_capacity};P={batch.pair_capacity};"
                f"K={batch.candidate_capacity};nx={batch.nx};ny={batch.ny};"
                f"F={batch.feature_count};slot={slot}\n")
    pieces = [torch.tensor(list(metadata.encode("ascii")), device=batch.device,
                           dtype=torch.uint8)]
    for group in (batch.common, batch.plan_a, batch.plan_c):
        if group is None:
            continue
        for field in fields(group):
            if field.name == "replay_digest":
                continue
            value = getattr(group, field.name)[slot].detach().contiguous()
            pieces.append(value.view(torch.uint8).reshape(-1))
    return torch.cat(pieces).contiguous()


def write_slot_digest(batch: FiveTrackPpoBatchV2, slot: int) -> None:
    """Store the fixed 32-byte HIP digest after writing the complete slot."""
    if batch.device.type != "cuda" or torch.version.hip is None:
        raise RuntimeError("PPO replay digest requires an assigned HIP device")
    from .meco_plan_a_shortest_path_hip import extension
    digest = extension().sha256_bytes(_slot_digest_payload(batch, slot))
    batch.common.replay_digest[slot].copy_(digest)


def validate_slot_digests(batch: FiveTrackPpoBatchV2) -> torch.Tensor:
    """Return a scalar device verdict; never inspect a slot or digest on host."""
    from .meco_plan_a_shortest_path_hip import extension
    valid = torch.ones((), device=batch.device, dtype=torch.bool)
    for slot in range(batch.capacity):
        actual = extension().sha256_bytes(_slot_digest_payload(batch, slot))
        matches = (actual == batch.common.replay_digest[slot]).all()
        valid = valid & (~batch.common.slot_valid[slot] | matches)
    return valid


def gae_v2(batch: FiveTrackPpoBatchV2, *, gamma: float,
           gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-slot GAE; all transition values and masks stay on the device."""
    common = batch.common
    advantages = torch.zeros_like(common.reward)
    last = torch.zeros((), device=batch.device, dtype=common.reward.dtype)
    for slot in reversed(range(batch.capacity)):
        active = common.slot_valid[slot]
        nonterminal = (~common.episode_boundary[slot]).to(common.reward.dtype)
        delta = (common.reward[slot] + gamma * common.next_value[slot] * nonterminal
                 - common.old_value[slot])
        candidate = delta + gamma * gae_lambda * nonterminal * last
        last = torch.where(active, candidate,
            torch.where(common.episode_boundary[slot], torch.zeros_like(last), last))
        advantages[slot] = torch.where(active, last, torch.zeros_like(last))
    returns = torch.where(common.slot_valid, advantages + common.old_value,
                          torch.zeros_like(advantages))
    return advantages, returns


def normalized_actor_advantage(batch: FiveTrackPpoBatchV2,
                               advantages: torch.Tensor) -> torch.Tensor:
    """Population-normalize actor rows with the frozen 1e-8 denominator floor."""
    actor = batch.common.actor_valid & batch.common.slot_valid
    weights = actor.to(dtype=advantages.dtype)
    count = weights.sum().clamp_min(1.)
    mean = (advantages * weights).sum() / count
    centered = advantages - mean
    variance = (centered.square() * weights).sum() / count
    normalized = centered / variance.sqrt().clamp_min(1e-8)
    return torch.where(actor, normalized, torch.zeros_like(normalized))


class MaskedSequentialAdamV1:
    """Adam state whose parameters, moments, steps and no-op gate stay on HIP.

    The caller performs one backward pass per fixed permutation position and
    passes a device scalar `active`. Padded or rejected rows preserve every
    parameter and state tensor exactly. Static parameter iteration is schema
    metadata; this class does not implement weight decay or AMSGrad.
    """

    def __init__(self, parameters, *, device, learning_rate: float,
                 beta1: float = .9, beta2: float = .999, eps: float = 1e-8):
        self.parameters = tuple(parameters)
        self.device = torch.device(device)
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        if (not self.parameters or self.device.type != "cuda" or torch.version.hip is None or
                any(parameter.device != self.device for parameter in self.parameters)):
            raise RuntimeError("sequential Adam requires nonempty HIP parameters on one device")
        if (self.learning_rate <= 0. or not 0. <= self.beta1 < 1. or
                not 0. <= self.beta2 < 1. or self.eps <= 0.):
            raise ValueError("invalid sequential Adam hyperparameters")
        self.first_moment = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.second_moment = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.step_count = [torch.zeros((), device=self.device, dtype=torch.long)
                           for _ in self.parameters]

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self, active: torch.Tensor, *, max_grad_norm: float) -> torch.Tensor:
        if (active.device != self.device or active.dtype != torch.bool or
                active.numel() != 1):
            raise RuntimeError("Adam update predicate must be one HIP bool tensor")
        gradients = [parameter.grad for parameter in self.parameters]
        present = [gradient for gradient in gradients if gradient is not None]
        if not present:
            raise RuntimeError("PPO loss produced no gradients for this track")
        if any(gradient.device != self.device or gradient.is_sparse for gradient in present):
            raise RuntimeError("PPO Adam requires dense gradients on its assigned HIP device")
        norms = torch.stack([torch.linalg.vector_norm(gradient.detach(), ord=2)
                             for gradient in present])
        total_norm = torch.linalg.vector_norm(norms, ord=2)
        clip_factor = (float(max_grad_norm) / (total_norm + 1e-6)).clamp(max=1.)
        active_scalar = active.reshape(())
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            parameter = self.parameters[index]
            first = self.first_moment[index]
            second = self.second_moment[index]
            step_count = self.step_count[index]
            clipped = gradient.detach() * clip_factor
            step_next = step_count + active_scalar.to(dtype=torch.long)
            first_next = self.beta1 * first + (1. - self.beta1) * clipped
            second_next = self.beta2 * second + (1. - self.beta2) * clipped.square()
            step_float = step_next.to(dtype=parameter.dtype)
            correction1 = 1. - torch.pow(parameter.new_tensor(self.beta1), step_float)
            correction2 = 1. - torch.pow(parameter.new_tensor(self.beta2), step_float)
            denominator = second_next.sqrt() / correction2.sqrt() + self.eps
            parameter_next = parameter - (self.learning_rate / correction1) * first_next / denominator
            parameter.copy_(torch.where(active_scalar, parameter_next, parameter))
            first.copy_(torch.where(active_scalar, first_next, first))
            second.copy_(torch.where(active_scalar, second_next, second))
            step_count.copy_(torch.where(active_scalar, step_next, step_count))
        return total_norm

    def state_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the ordered opaque v1 tensor state for checkpoint serialization."""
        flattened = []
        for step, first, second in zip(self.step_count,
                                       self.first_moment, self.second_moment):
            flattened.extend((step, first, second))
        return tuple(flattened)

    def state_dict(self) -> dict[str, object]:
        return {"schema_id": PPO_OPTIMIZER_SCHEMA_ID,
            "learning_rate": self.learning_rate, "beta1": self.beta1,
            "beta2": self.beta2, "eps": self.eps,
            "parameter_count": len(self.parameters),
            "state": [tuple(value.detach().clone() for value in triple)
                      for triple in zip(self.step_count, self.first_moment,
                                        self.second_moment)]}

    def validate_state_dict(self, state: dict[str, object]):
        if (not isinstance(state, dict) or set(state) != {
                "schema_id", "learning_rate", "beta1", "beta2", "eps",
                "parameter_count", "state"} or
                state.get("schema_id") != PPO_OPTIMIZER_SCHEMA_ID or
                state.get("learning_rate") != self.learning_rate or
                state.get("beta1") != self.beta1 or
                state.get("beta2") != self.beta2 or
                state.get("eps") != self.eps or
                state.get("parameter_count") != len(self.parameters)):
            raise ValueError("incompatible sequential Adam checkpoint schema")
        saved = state["state"]
        if not isinstance(saved, (list, tuple)) or len(saved) != len(self.parameters):
            raise ValueError("sequential Adam checkpoint parameter count mismatch")
        staged = []
        for index, triple in enumerate(saved):
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                raise ValueError("sequential Adam checkpoint state tuple is malformed")
            parameter = self.parameters[index]
            step, first, second = triple
            if (not isinstance(step, torch.Tensor) or step.dtype != torch.long or
                    step.numel() != 1 or int(step.item()) < 0 or
                    not isinstance(first, torch.Tensor) or
                    not isinstance(second, torch.Tensor) or
                    first.shape != parameter.shape or second.shape != parameter.shape or
                    first.dtype != parameter.dtype or second.dtype != parameter.dtype):
                raise ValueError("sequential Adam checkpoint tensor shape mismatch")
            if (not bool(torch.isfinite(first).all()) or
                    not bool(torch.isfinite(second).all())):
                raise ValueError("sequential Adam checkpoint moments must be finite")
            if bool((second < 0).any()):
                raise ValueError("sequential Adam second moments must be nonnegative")
            step = step.to(device=self.device)
            first = first.to(device=self.device)
            second = second.to(device=self.device)
            staged.append((step, first, second))
        return tuple(staged)

    def load_state_dict(self, state: dict[str, object]) -> None:
        staged = self.validate_state_dict(state)
        for index, (step, first, second) in enumerate(staged):
            self.step_count[index].copy_(step.reshape(()))
            self.first_moment[index].copy_(first)
            self.second_moment[index].copy_(second)


class DevicePpoRngV1:
    """Independent explicit HIP streams for policy actions and row orders."""

    def __init__(self, *, device, action_seed: int, permutation_seed: int):
        self.device = torch.device(device)
        if self.device.type != "cuda" or torch.version.hip is None:
            raise RuntimeError("five-track PPO RNG streams require an assigned HIP device")
        self.action = torch.Generator(device=self.device)
        self.permutation = torch.Generator(device=self.device)
        self.action.manual_seed(int(action_seed))
        self.permutation.manual_seed(int(permutation_seed))

    @classmethod
    def from_seed(cls, seed: int, *, device) -> "DevicePpoRngV1":
        if type(seed) is not int or seed < 0:
            raise ValueError("PPO RNG seed must be a nonnegative integer")
        # Separate streams keep rollout action draws independent from epoch
        # order draws while binding both streams to the host-known run seed.
        modulus = (1 << 63) - 1
        return cls(device=device,
            action_seed=(seed * 6364136223846793005 + 1442695040888963407) % modulus,
            permutation_seed=(seed * 2862933555777941757 + 3037000493) % modulus)

    def action_index(self, logits: torch.Tensor, *, deterministic: bool = False):
        from .meco_tracks import sample_candidate_action
        if logits.device != self.device:
            raise RuntimeError("action logits do not match the PPO RNG device")
        return sample_candidate_action(logits, deterministic=deterministic,
                                       generator=self.action)

    def permutation_indices(self, capacity: int) -> torch.Tensor:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("permutation capacity must be a positive static integer")
        return torch.randperm(capacity, device=self.device, generator=self.permutation)

    def state_dict(self) -> dict[str, object]:
        return {"schema_id": PPO_RNG_SCHEMA_ID,
                "action": self.action.get_state(),
                "permutation": self.permutation.get_state()}

    def validate_state_dict(self, state: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
        if (not isinstance(state, dict) or
                state.get("schema_id") != PPO_RNG_SCHEMA_ID or
                set(state) != {"schema_id", "action", "permutation"}):
            raise ValueError("incompatible five-track PPO RNG checkpoint schema")
        if any(not isinstance(state[name], torch.Tensor) for name in ("action", "permutation")):
            raise ValueError("five-track PPO RNG checkpoint states must be tensors")
        action = torch.Generator(device=self.device)
        permutation = torch.Generator(device=self.device)
        try:
            action.set_state(state["action"])
            permutation.set_state(state["permutation"])
        except (RuntimeError, TypeError) as exc:
            raise ValueError("five-track PPO RNG checkpoint state is malformed") from exc
        return action.get_state(), permutation.get_state()

    def load_state_dict(self, state: dict[str, object]) -> None:
        action_state, permutation_state = self.validate_state_dict(state)
        self.action.set_state(action_state)
        self.permutation.set_state(permutation_state)
