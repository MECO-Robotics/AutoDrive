"""Exact-weight Plan A shortest-path primitives and gated HIP heap prototype.

The CPU heap is an offline/reference path. Live HIP support dispatch remains
fail-closed until the WX9100 parity/profile gate is independently accepted.
Route chunk coordinates are preparation-only; evaluation is deferred until the
shared device-query batch API is available to Plan A.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import threading
import weakref

_EDGE_WEIGHT_CACHE = {}
_EDGE_WEIGHT_CACHE_LOCK = threading.RLock()
PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED = False


@dataclass(frozen=True)
class PlanARootBinding:
    """Device root and connector values for one canonical pose graph."""

    root_index: "torch.Tensor"
    root_start_cost: "torch.Tensor"
    root_connector_m: "torch.Tensor"
    valid: "torch.Tensor"


@dataclass(frozen=True)
class PlanARouteChunkMap:
    """Shared route chunk coordinates plus Plan A support-to-node mapping."""

    row_count: int
    spans: tuple[tuple[int, int], ...]
    global_row: "torch.Tensor"
    chunk_index: "torch.Tensor"
    local_row: "torch.Tensor"
    support_to_route_row: "torch.Tensor"
    support_route_row_valid: "torch.Tensor"

    def __post_init__(self):
        import torch
        from .experimental_route_evaluation import MAX_ROUTE_QUERY_BATCH

        if type(self.row_count) is not int or self.row_count < 1:
            raise ValueError("Plan A route row count must be a positive static integer")
        expected = tuple((start, min(start + MAX_ROUTE_QUERY_BATCH, self.row_count))
                         for start in range(0, self.row_count, MAX_ROUTE_QUERY_BATCH))
        if self.spans != expected:
            raise ValueError("Plan A route chunk spans must be contiguous and complete")
        if any(not torch.is_tensor(value) for value in
               (self.global_row, self.chunk_index, self.local_row,
                self.support_to_route_row, self.support_route_row_valid)):
            raise ValueError("Plan A route chunk mappings must be device tensors")
        if any(value.shape != (self.row_count,) for value in
               (self.global_row, self.chunk_index, self.local_row)):
            raise ValueError("Plan A global/chunk/local maps must be [N]")
        if (self.support_to_route_row.ndim != 1 or
                self.support_route_row_valid.shape != self.support_to_route_row.shape):
            raise ValueError("Plan A support route map must be [P]")
        if any(value.dtype != torch.long for value in
               (self.global_row, self.chunk_index, self.local_row,
                self.support_to_route_row)) or \
                self.support_route_row_valid.dtype != torch.bool:
            raise ValueError("Plan A route map indices/mask have invalid dtypes")
        if any(value.device != self.global_row.device for value in
               (self.chunk_index, self.local_row, self.support_to_route_row,
                self.support_route_row_valid)):
            raise ValueError("Plan A route mappings must share one device")


def plan_a_root_binding(graph, pose, *, device) -> PlanARootBinding:
    """Select the nearest legal SE(2) root and connector entirely on-device.

    The root tie break is the first canonical graph row, matching ``argmin`` in
    the existing Plan A support builder. ``valid`` remains a device predicate;
    callers must bind it into path status instead of reading it on the host.
    """
    import torch

    selected = torch.device(device)
    graph.require_device(selected)
    if (not torch.is_tensor(pose) or pose.shape != (3,) or
            pose.device != selected or graph.xy.device != selected or
            not torch.is_floating_point(pose) or
            graph.node_valid.shape != (1, graph.node_capacity)):
        raise ValueError("Plan A root selection requires one same-device [x,y,heading] pose")
    angle = torch.atan2(torch.sin(graph.heading - pose[2]),
                        torch.cos(graph.heading - pose[2])).abs()
    start_cost = (graph.xy - pose[:2]).norm(dim=-1) + angle * .25
    start_cost = start_cost.masked_fill(~graph.node_valid[0], torch.inf)
    root = start_cost.argmin()
    root_start_cost = start_cost.index_select(0, root.reshape(1)).reshape(()).to(torch.float64)
    root_xy = graph.xy.index_select(0, root.reshape(1)).reshape(2)
    root_connector_m = (pose[:2] - root_xy).norm().to(torch.float64)
    valid = (graph.graph_status == 0) & graph.node_valid[0].any() & \
        torch.isfinite(pose).all() & torch.isfinite(root_start_cost) & \
        torch.isfinite(root_connector_m) & (root_start_cost >= 0.) & \
        (root_connector_m >= 0.)
    return PlanARootBinding(root.to(torch.int64), root_start_cost,
                            root_connector_m, valid)


def make_plan_a_route_chunk_map(route_row_count: int, support_to_route_row,
                                *, device) -> PlanARouteChunkMap:
    """Prepare shared deterministic 64-row coordinates without evaluation.

    The canonical device-query producer/consumer contract belongs to the shared
    route API. Plan A must not build host query records or call its legacy host
    evaluator; route evaluation is deferred until that API is approved.
    """
    import torch
    from .experimental_route_evaluation import make_device_route_chunk_map

    if type(route_row_count) is not int or route_row_count < 1:
        raise ValueError("Plan A route row count must be a positive static integer")
    selected = torch.device(device)
    if (not torch.is_tensor(support_to_route_row) or
            support_to_route_row.ndim != 1 or
            support_to_route_row.dtype != torch.long or
            support_to_route_row.device != selected):
        raise ValueError("Plan A support-to-route rows must be same-device int64 [P]")
    mapping = make_device_route_chunk_map(route_row_count, device=selected)
    support_valid = ((support_to_route_row >= 0) &
                     (support_to_route_row < route_row_count))
    return PlanARouteChunkMap(route_row_count, mapping.spans,
        mapping.global_row, mapping.chunk_index, mapping.local_row,
        support_to_route_row, support_valid)


def _require_wx9100_device(device):
    """Reject non-WX9100 devices before loading the experimental extension."""
    import torch

    selected = torch.device(device)
    if selected.type != "cuda" or selected.index is None:
        raise ValueError("Plan A HIP heap requires an explicitly indexed HIP device")
    if (torch.version.hip is None or not torch.cuda.is_available() or
            selected.index >= torch.cuda.device_count()):
        raise RuntimeError("Plan A HIP heap requires an available HIP device")
    name = torch.cuda.get_device_name(selected).casefold().replace("-", " ").replace("_", " ")
    if "wx 9100" not in name and "wx9100" not in name:
        raise RuntimeError("Plan A HIP heap is restricted to WX9100 devices")
    return selected


def precompute_plan_a_edge_weights(xy, headings, edges, edge_mask):
    """Precompute directed edge cost/translation using the canonical Python math."""
    xy_values = xy.detach().cpu().tolist() if hasattr(xy, "detach") else xy
    heading_values = headings.detach().cpu().tolist() if hasattr(headings, "detach") else headings
    edge_values = edges.detach().cpu().tolist() if hasattr(edges, "detach") else edges
    mask_values = edge_mask.detach().cpu().tolist() if hasattr(edge_mask, "detach") else edge_mask
    n = len(xy_values)
    slots = len(edge_values[0]) if n else 0
    costs = [[math.inf] * slots for _ in range(n)]
    lengths = [[0.] * slots for _ in range(n)]
    allowed_values = [[False] * slots for _ in range(n)]
    destinations = [[0] * slots for _ in range(n)]
    for src in range(n):
        for slot, (dst, allowed) in enumerate(zip(edge_values[src], mask_values[src])):
            destinations[src][slot] = int(dst)
            if not allowed:
                continue
            dx = xy_values[src][0] - xy_values[dst][0]
            dy = xy_values[src][1] - xy_values[dst][1]
            translation = math.hypot(dx, dy)
            delta_heading = heading_values[dst] - heading_values[src]
            turn = abs(math.atan2(math.sin(delta_heading), math.cos(delta_heading)))
            costs[src][slot] = translation + .25 * turn
            lengths[src][slot] = translation
            allowed_values[src][slot] = True
    return destinations, costs, lengths, allowed_values


def cached_plan_a_edge_weights(graph):
    """Reuse precomputed weights for the lifetime of one static pose graph."""
    graph_id = id(graph)
    with _EDGE_WEIGHT_CACHE_LOCK:
        cached = _EDGE_WEIGHT_CACHE.get(graph_id)
        if cached is not None and cached[0]() is graph:
            return cached[1]["host"]
        edge_values = graph.edges[0].detach().cpu().tolist()
        mask_values = graph.edge_mask[0].detach().cpu().tolist()
        weights = precompute_plan_a_edge_weights(graph.xy, graph.heading,
            edge_values, mask_values)
        try:
            reference = weakref.ref(graph,
                lambda reference, key=graph_id: _discard_graph_weights(key, reference))
        except TypeError as exc:
            raise TypeError("static legal pose graph must support weak references") from exc
        _EDGE_WEIGHT_CACHE[graph_id] = (reference,
            {"host": weights, "device_tensors": {}})
        return weights


def _discard_graph_weights(graph_id, reference):
    with _EDGE_WEIGHT_CACHE_LOCK:
        current = _EDGE_WEIGHT_CACHE.get(graph_id)
        if current is not None and current[0] is reference:
            _EDGE_WEIGHT_CACHE.pop(graph_id, None)


def plan_a_dijkstra_cpu(edge_dst, edge_costs, edge_lengths, edge_allowed, *, root,
                        root_start_cost, root_connector_m):
    """CPU reference matching the original heap order and epsilon contract."""
    n = len(edge_costs)
    if (n < 1 or len(edge_dst) != n or len(edge_lengths) != n or len(edge_allowed) != n or
            type(root) is not int or not 0 <= root < n):
        raise ValueError("invalid Plan A graph or root")
    distances = [math.inf] * n
    physical = [math.inf] * n
    parents = [-1] * n
    parent_edge_m = [0.] * n
    distances[root] = float(root_start_cost)
    physical[root] = float(root_connector_m)
    pending = [(distances[root], root)]
    settled_order = []
    while pending:
        distance, src = heapq.heappop(pending)
        if distance != distances[src]:
            continue
        settled_order.append(src)
        for dst, cost, length, allowed in zip(edge_dst[src], edge_costs[src],
                                               edge_lengths[src], edge_allowed[src]):
            if not allowed:
                continue
            proposal = distance + cost
            if proposal < distances[dst] - 1e-9:
                distances[dst] = proposal
                physical[dst] = physical[src] + length
                parents[dst] = src
                parent_edge_m[dst] = length
                heapq.heappush(pending, (proposal, dst))
    traversal = sorted((i for i, value in enumerate(distances) if math.isfinite(value)),
                       key=lambda i: (distances[i], i))
    return {"route_cost": distances, "physical_distance": physical,
        "parent_ids": parents, "parent_edge_m": parent_edge_m,
        "traversal_order": traversal}


def plan_a_fallback_pair(valid_matrix, physical_distance):
    """Choose closest node by physical path distance, then first valid slot."""
    if len(valid_matrix) != len(physical_distance):
        raise ValueError("fallback support and distances differ in size")
    candidates = [node for node, row in enumerate(valid_matrix) if any(row)]
    if not candidates:
        raise ValueError("Plan A safe fallback is absent from the support mask")
    node = min(candidates, key=lambda index: (physical_distance[index], index))
    return node * len(valid_matrix[node]) + next(
        slot for slot, valid in enumerate(valid_matrix[node]) if valid)


def plan_a_dijkstra_hip(edge_dst, edge_costs, edge_lengths, edge_allowed, *, root,
                        root_start_cost, root_connector_m):
    """Legacy harness prototype with host status read; excluded from live dispatch.

    Training/evaluation use ``plan_a_dijkstra_hip_for_graph`` and its
    device-resident status tensor instead.
    """
    import torch

    if not PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED:
        raise RuntimeError("Plan A HIP dispatch is disabled pending WX9100 parity and profiling")
    selected = _require_wx9100_device(edge_costs.device)
    if (type(root) is not int or type(root_start_cost) not in (int, float) or
            type(root_connector_m) not in (int, float) or
            not math.isfinite(float(root_start_cost)) or float(root_start_cost) < 0 or
            not math.isfinite(float(root_connector_m)) or float(root_connector_m) < 0):
        raise ValueError("HIP heap root and connector costs must be finite and nonnegative")

    from .meco_plan_a_shortest_path_hip import extension

    if not (torch.is_tensor(edge_dst) and edge_dst.ndim == 2 and
            torch.is_tensor(edge_costs) and edge_costs.is_cuda and
            torch.is_tensor(edge_lengths) and torch.is_tensor(edge_allowed) and
            edge_costs.dtype == torch.float64 and edge_costs.ndim == 2 and
            edge_costs.shape[0] > 0 and edge_costs.shape[1] > 0 and
            edge_costs.shape[0] <= (torch.iinfo(torch.int64).max - 1) // edge_costs.shape[1] and
            edge_lengths.shape == edge_costs.shape and edge_lengths.dtype == torch.float64 and
            edge_allowed.shape == edge_costs.shape and edge_allowed.dtype == torch.bool and
            edge_costs.device == edge_lengths.device == edge_allowed.device == edge_dst.device == selected and
            edge_dst.dtype == torch.int64 and edge_dst.shape == edge_costs.shape and
            edge_dst.is_cuda):
        raise ValueError("HIP heap requires same-device float64 costs/lengths and bool mask")
    route, physical, parent, parent_length, _settled = extension().dijkstra(
        edge_dst.contiguous(), edge_costs.contiguous(), edge_lengths.contiguous(),
        edge_allowed.contiguous(), int(root), float(root_start_cost),
        float(root_connector_m))
    reachable = torch.isfinite(route)
    reachable_nodes = torch.arange(route.numel(), device=route.device,
                                   dtype=torch.long)[reachable]
    # reachable_nodes is ascending, so stable route sorting implements the
    # historical (final distance, node_id) order exactly.
    order = reachable_nodes[torch.argsort(route[reachable], stable=True)]
    return {"route_cost": route, "physical_distance": physical,
        "parent_ids": parent, "parent_edge_m": parent_length,
        "traversal_order": order}


def plan_a_dijkstra_hip_for_graph(graph, *, device, root, root_start_cost,
                                  root_connector_m, root_valid=None):
    """Run exact graph geometry and Dijkstra on device without host graph copies.

    This prototype entry remains disabled for live support until WX9100
    parity/profile evidence opens the dispatch gate. Root validity, graph
    validity and the kernel status are combined on-device; a failed route
    returns only invalid sentinels rather than a partial shortest-path tree.
    """
    import torch

    if not PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED:
        raise RuntimeError("Plan A HIP dispatch is disabled pending WX9100 parity and profiling")
    selected = torch.device(device)
    xy = graph.xy
    heading = graph.heading
    edge_dst = graph.edges[0]
    edge_allowed = graph.edge_mask[0]
    n = int(graph.node_capacity)
    if (not all(torch.is_tensor(value) and value.device == selected for value in
                (xy, heading, edge_dst, edge_allowed, graph.node_valid[0],
                 graph.graph_status)) or
            xy.shape != (n, 2) or heading.shape != (n,) or
            edge_dst.ndim != 2 or edge_dst.shape[0] != n or
            edge_allowed.shape != edge_dst.shape or
            graph.node_valid.shape != (1, n) or graph.node_valid.dtype != torch.bool or
            graph.graph_status.shape != () or graph.graph_status.dtype != torch.long or
            edge_dst.dtype != torch.int64 or edge_allowed.dtype != torch.bool or
            xy.dtype not in (torch.float32, torch.float64) or
            heading.dtype not in (torch.float32, torch.float64) or
            not torch.is_tensor(root) or root.shape != () or root.dtype != torch.int64 or
            not torch.is_tensor(root_start_cost) or root_start_cost.shape != () or
            root_start_cost.dtype != torch.float64 or
            not torch.is_tensor(root_connector_m) or root_connector_m.shape != () or
            root_connector_m.dtype != torch.float64 or
            root.device != selected or root_start_cost.device != selected or
            root_connector_m.device != selected):
        raise ValueError("HIP graph route requires valid same-device graph tensors and root costs")
    if n < 1 or edge_dst.shape[1] < 1:
        raise ValueError("HIP graph route requires positive node capacity and degree")
    if root_valid is not None and (not torch.is_tensor(root_valid) or
            root_valid.shape != () or root_valid.dtype != torch.bool or
            root_valid.device != selected):
        raise ValueError("HIP root validity must be a same-device bool scalar")
    from .meco_plan_a_shortest_path_hip import dijkstra_graph
    route, physical, parent, parent_length, settled, status = dijkstra_graph(
        xy, heading, edge_dst, edge_allowed, graph.node_valid[0], root,
        root_start_cost, root_connector_m)
    destination_in_range = (edge_dst >= 0) & (edge_dst < n)
    safe_destination = edge_dst.clamp(0, n - 1)
    destination_valid = graph.node_valid[0].index_select(
        0, safe_destination.reshape(-1)).reshape_as(edge_dst)
    active_edges_valid = ((~edge_allowed) |
                          (destination_in_range & destination_valid)).all()
    active_sources_valid = ((~edge_allowed) | graph.node_valid[0][:, None]).all()
    root_in_range = (root >= 0) & (root < n)
    safe_root = root.clamp(0, n - 1)
    root_node_valid = graph.node_valid[0].index_select(
        0, safe_root.reshape(1)).reshape(())
    inputs_valid = (torch.isfinite(xy).all() & torch.isfinite(heading).all() &
        active_edges_valid & active_sources_valid & root_in_range & root_node_valid &
        (graph.graph_status == 0) & torch.isfinite(root_start_cost) &
        torch.isfinite(root_connector_m) & (root_start_cost >= 0.) &
        (root_connector_m >= 0.))
    if root_valid is not None:
        inputs_valid = inputs_valid & root_valid
    parent_valid = (parent == -1) | ((parent >= 0) & (parent < n))
    traversal_valid = (settled == -1) | ((settled >= 0) & (settled < n))
    route_outputs_valid = ((~torch.isnan(route)).all() &
        (~torch.isnan(physical)).all() & torch.isfinite(parent_length).all() &
        (parent_length >= 0.).all() & parent_valid.all() & traversal_valid.all() &
        ((~torch.isfinite(route)) | torch.isfinite(physical)).all())
    status = torch.where(inputs_valid & route_outputs_valid, status,
                         torch.ones_like(status))
    result_valid = status == 0
    route = torch.where(result_valid, route, torch.full_like(route, torch.inf))
    physical = torch.where(result_valid, physical,
                           torch.full_like(physical, torch.inf))
    parent = torch.where(result_valid, parent, torch.full_like(parent, -1))
    parent_length = torch.where(result_valid, parent_length,
                                torch.zeros_like(parent_length))
    settled = torch.where(result_valid, settled, torch.full_like(settled, -1))
    return {"route_cost": route, "physical_distance": physical,
        "parent_ids": parent, "parent_edge_m": parent_length,
        "traversal_order": settled, "status": status, "valid": result_valid}


def plan_a_dijkstra_hip_for_root_binding(graph, *, device,
                                         root_binding: PlanARootBinding):
    """Consume a device-computed root binding without scalar projection."""
    if not isinstance(root_binding, PlanARootBinding):
        raise TypeError("root_binding must be a PlanARootBinding")
    return plan_a_dijkstra_hip_for_graph(graph, device=device,
        root=root_binding.root_index,
        root_start_cost=root_binding.root_start_cost,
        root_connector_m=root_binding.root_connector_m,
        root_valid=root_binding.valid)
