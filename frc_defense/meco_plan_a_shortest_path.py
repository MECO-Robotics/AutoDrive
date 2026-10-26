"""Exact-weight Plan A shortest-path primitives and gated HIP heap prototype.

Runtime support construction continues to use the existing CPU heap until
the separately exposed HIP prototype passes WX9100 parity and profiling.
"""
from __future__ import annotations

import heapq
import math
import threading
import weakref

_EDGE_WEIGHT_CACHE = {}
_EDGE_WEIGHT_CACHE_LOCK = threading.RLock()
PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED = False


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
    """Fail-closed prototype entry point; never called by training dispatch."""
    import torch

    if not PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED:
        raise RuntimeError("Plan A HIP dispatch is disabled pending WX9100 parity and profiling")
    selected = _require_wx9100_device(edge_costs.device)

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
            edge_dst.is_cuda and
            torch.is_tensor(root) and root.ndim == 0 and root.dtype == torch.int64 and
            root.is_cuda and root.device == selected and root.is_contiguous() and
            torch.is_tensor(root_start_cost) and root_start_cost.ndim == 0 and
            root_start_cost.dtype == torch.float64 and root_start_cost.is_cuda and
            root_start_cost.device == selected and root_start_cost.is_contiguous() and
            torch.is_tensor(root_connector_m) and root_connector_m.ndim == 0 and
            root_connector_m.dtype == torch.float64 and root_connector_m.is_cuda and
            root_connector_m.device == selected and root_connector_m.is_contiguous()):
        raise ValueError("HIP heap requires same-device float64 costs/lengths and bool mask")
    route, physical, parent, parent_length, _settled, status = extension().dijkstra(
        edge_dst.contiguous(), edge_costs.contiguous(), edge_lengths.contiguous(),
        edge_allowed.contiguous(), root.contiguous(), root_start_cost.contiguous(),
        root_connector_m.contiguous())
    reachable = torch.isfinite(route)
    reachable_nodes = torch.arange(route.numel(), device=route.device,
                                   dtype=torch.long)[reachable]
    # reachable_nodes is ascending, so stable route sorting implements the
    # historical (final distance, node_id) order exactly.
    order = reachable_nodes[torch.argsort(route[reachable], stable=True)]
    return {"route_cost": route, "physical_distance": physical,
        "parent_ids": parent, "parent_edge_m": parent_length,
        "traversal_order": order, "status": status}


def plan_a_dijkstra_hip_for_graph(graph, *, device, root, root_start_cost,
                                  root_connector_m):
    """Run exact graph geometry and Dijkstra on device without host graph copies.

    This is an explicit prototype entry point. Runtime support construction
    remains on the CPU oracle until WX9100 parity and profile evidence opens
    the dispatch gate.
    """
    import torch

    if not PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED:
        raise RuntimeError("Plan A HIP dispatch is disabled pending WX9100 parity and profiling")
    device = torch.device(device)
    selected = _require_wx9100_device(device)
    xy = graph.xy
    heading = graph.heading
    edge_dst = graph.edges[0]
    edge_allowed = graph.edge_mask[0]
    n = int(graph.node_count)
    if (not all(torch.is_tensor(value) and value.device == selected for value in
                (xy, heading, edge_dst, edge_allowed)) or
            xy.shape != (n, 2) or heading.shape != (n,) or
            edge_dst.ndim != 2 or edge_dst.shape[0] != n or
            edge_allowed.shape != edge_dst.shape or
            edge_dst.dtype != torch.int64 or edge_allowed.dtype != torch.bool or
            xy.dtype not in (torch.float32, torch.float64) or
            heading.dtype not in (torch.float32, torch.float64) or
            not torch.is_tensor(root) or root.device != selected or
            root.ndim != 0 or root.dtype != torch.int64 or not root.is_contiguous() or
            not torch.is_tensor(root_start_cost) or root_start_cost.device != selected or
            root_start_cost.ndim != 0 or root_start_cost.dtype != torch.float64 or
            not root_start_cost.is_contiguous() or
            not torch.is_tensor(root_connector_m) or root_connector_m.device != selected or
            root_connector_m.ndim != 0 or root_connector_m.dtype != torch.float64 or
            not root_connector_m.is_contiguous()):
        raise ValueError("HIP graph route requires valid same-device graph tensors and root costs")
    from .meco_plan_a_shortest_path_hip import extension
    route, physical, parent, parent_length, settled, status = extension().dijkstra_graph(
        xy.contiguous(), heading.contiguous(), edge_dst.contiguous(),
        edge_allowed.contiguous(), root.contiguous(), root_start_cost.contiguous(),
        root_connector_m.contiguous())
    # The device heap settles nodes by (distance, node_id); remove only the
    # unreachable -1 tail without materializing any route data on the host.
    order = settled[settled >= 0]
    return {"route_cost": route, "physical_distance": physical,
        "parent_ids": parent, "parent_edge_m": parent_length,
        "traversal_order": order, "status": status}
