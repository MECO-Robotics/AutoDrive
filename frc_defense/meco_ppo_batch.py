"""Versioned fixed-capacity device batch contract for five-track PPO.

This module deliberately contains no simulator objects, per-transition mappings,
candidate/support labels, or CPU numeric optimizer.  Construction captures only
static layout metadata; dynamic row state is stored in device tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch


PPO_BATCH_SCHEMA_VERSION = 2
PPO_BATCH_SCHEMA_ID = "five-track-ppo-fixed-capacity-v2"
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
    reward: torch.Tensor
    episode_boundary: torch.Tensor
    execution_failure: torch.Tensor
    slot_valid: torch.Tensor
    actor_valid: torch.Tensor
    critic_valid: torch.Tensor
    replay_digest: torch.Tensor


@dataclass
class PlanAPpoTensors:
    node_features: torch.Tensor
    xy: torch.Tensor
    heading_bins: torch.Tensor
    node_valid: torch.Tensor
    edges: torch.Tensor
    edge_mask: torch.Tensor
    region_ids: torch.Tensor
    coarse_edges: torch.Tensor
    coarse_edge_mask: torch.Tensor
    legal_lookup: torch.Tensor
    pair_node_ids: torch.Tensor
    behavior_ids: torch.Tensor
    action_ids: torch.Tensor
    descriptor_ids: torch.Tensor
    support_valid: torch.Tensor
    route_cost: torch.Tensor
    risk_cost: torch.Tensor
    parent_edge_m: torch.Tensor
    traversal_order: torch.Tensor
    ancestor_jumps: torch.Tensor
    root_node: torch.Tensor
    root_connector_m: torch.Tensor
    fallback_pair: torch.Tensor


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
    candidate_valid: torch.Tensor
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
            old_logprob=empty((R,)), old_value=empty((R,)), reward=empty((R,)),
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
                node_features=empty((R, N, 9)), xy=empty((R, N, 2)),
                heading_bins=empty((R, N), dtype_=torch.long),
                node_valid=empty((R, N), dtype_=torch.bool),
                edges=empty((R, N, PPO_GRAPH_DEGREE), dtype_=torch.long),
                edge_mask=empty((R, N, PPO_GRAPH_DEGREE), dtype_=torch.bool),
                region_ids=empty((R, N), dtype_=torch.long),
                coarse_edges=empty((R, PPO_REGION_COUNT, PPO_REGION_COUNT), dtype_=torch.long),
                coarse_edge_mask=empty((R, PPO_REGION_COUNT, PPO_REGION_COUNT), dtype_=torch.bool),
                legal_lookup=empty((R, 8, nx, ny), dtype_=torch.long),
                pair_node_ids=empty((R, P), dtype_=torch.long),
                behavior_ids=empty((R, P), dtype_=torch.long),
                action_ids=empty((R, P), dtype_=torch.long),
                descriptor_ids=empty((R, P), dtype_=torch.long),
                support_valid=empty((R, P), dtype_=torch.bool),
                route_cost=empty((R, N)), risk_cost=empty((R, N)),
                parent_edge_m=empty((R, N)),
                traversal_order=empty((R, N), dtype_=torch.long),
                ancestor_jumps=empty((R, jump_count, N), dtype_=torch.long),
                root_node=empty((R,), dtype_=torch.long),
                root_connector_m=empty((R,)), fallback_pair=empty((R,), dtype_=torch.long))

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
                candidate_valid=empty((R, K), dtype_=torch.bool),
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


def write_common_slot(batch: FiveTrackPpoBatchV2, slot: int, *, raster: torch.Tensor,
                      globals_: torch.Tensor, hidden_before: torch.Tensor | None,
                      action: torch.Tensor, old_logprob: torch.Tensor,
                      old_value: torch.Tensor, reward: torch.Tensor,
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
                      route_cost: torch.Tensor, risk_cost: torch.Tensor) -> None:
    """Copy fixed-shape A support/graph tensors into the chosen static slot."""
    if batch.plan_a is None or type(slot) is not int or not 0 <= slot < batch.capacity:
        raise ValueError("invalid Plan A PPO slot")
    destination = batch.plan_a
    fields_to_copy = (
        (destination.node_features, graph_node_features.reshape_as(destination.node_features[slot])),
        (destination.xy, graph.xy),
        (destination.heading_bins, graph.heading_bin),
        (destination.node_valid, graph.node_valid.reshape_as(destination.node_valid[slot])),
        (destination.edges, graph.edges.reshape_as(destination.edges[slot])),
        (destination.edge_mask, graph.edge_mask.reshape_as(destination.edge_mask[slot])),
        (destination.region_ids, graph.region_ids.reshape_as(destination.region_ids[slot])),
        (destination.coarse_edges, graph.coarse_edges.reshape_as(destination.coarse_edges[slot])),
        (destination.coarse_edge_mask, graph.coarse_edge_mask.reshape_as(destination.coarse_edge_mask[slot])),
        (destination.legal_lookup, graph.lookup),
        (destination.pair_node_ids, support.pair_node_ids),
        (destination.behavior_ids, support.behavior_ids),
        (destination.action_ids, support.action_ids),
        (destination.descriptor_ids, support.descriptor_ids),
        (destination.support_valid, support.valid_mask),
        (destination.route_cost, route_cost.reshape_as(destination.route_cost[slot])),
        (destination.risk_cost, risk_cost.reshape_as(destination.risk_cost[slot])),
        (destination.parent_edge_m, support.parent_edge_m),
        (destination.traversal_order, support.traversal_order),
        (destination.ancestor_jumps, support.ancestor_jumps),
        (destination.root_connector_m, support.root_connector_m.reshape(())),
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
                      route_cost: torch.Tensor, risk_cost: torch.Tensor) -> None:
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
        (destination.candidate_valid, device_batch.valid_mask),
        (destination.route_cost, route_cost),
        (destination.risk_cost, risk_cost),
    )
    for target, source in source_fields:
        row = source.reshape((count, *source.shape[2:])) if source.ndim > 2 and source.shape[0] == 1 else source.reshape(count, *source.shape[1:])
        target[slot, :count].copy_(row)


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
