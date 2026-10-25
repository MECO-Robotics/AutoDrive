"""Static legal SE(2) pose lattice used by the MECO research tracks.

The lattice is a research feasibility representation. Candidate routes still
pass through the simulator's independent AD* footprint and observed-robot
checks before execution. It does not replace the production planner.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch


HEADING_BINS = 8
REGION_COUNT = 64


def compact_legal_pose_mask(legal: torch.Tensor):
    """Return a stable fixed-capacity row-major compaction contract."""
    if legal.ndim != 3 or legal.shape[0] != HEADING_BINS or legal.dtype != torch.bool:
        raise ValueError("legal pose mask must be bool [8,nx,ny]")
    flat = legal.reshape(-1)
    capacity = flat.numel()
    original = torch.arange(capacity, device=legal.device, dtype=torch.long)
    permutation = torch.argsort((~flat).to(torch.long) * capacity + original,
                                stable=True)
    inverse = torch.empty_like(permutation).scatter_(0, permutation, original)
    node_valid = flat[permutation]
    return permutation, inverse, node_valid


@dataclass(frozen=True)
class LegalPoseGraph:
    xy: torch.Tensor                 # [N,2], field meters
    heading_bin: torch.Tensor        # [N]
    heading: torch.Tensor            # [N], radians
    edges: torch.Tensor              # [1,N,D], legal destination node IDs
    edge_mask: torch.Tensor          # [1,N,D]
    node_valid: torch.Tensor          # [1,N], compact valid prefix, device resident
    region_ids: torch.Tensor         # [1,N], connected graph regions
    coarse_edges: torch.Tensor       # [1,R,Dc], legal inter-region transitions
    coarse_edge_mask: torch.Tensor   # [1,R,Dc]
    lookup: torch.Tensor             # [8,nx,ny], -1 for illegal poses
    graph_status: torch.Tensor       # scalar device status; nonzero fails closed
    nx: int
    ny: int
    resolution_m: float
    region_count: int

    @property
    def node_capacity(self) -> int:
        return self.xy.shape[0]

    @property
    def valid_count(self) -> torch.Tensor:
        return self.node_valid.sum()

    def node_features(self, env, raster: torch.Tensor, *, world: int = 0,
                      slot: int = 3) -> torch.Tensor:
        """Return raster tracks plus normalized pose geometry per legal node."""
        if raster.ndim != 4 or raster.shape[1] != 6:
            raise ValueError("raster must have shape [batch,6,height,width]")
        height, width = raster.shape[-2:]
        field_length = float(env.sim.field_length)
        field_width = float(env.sim.field_width)
        col = (self.xy[:, 0] / field_length * width).floor().long().clamp(0, width - 1)
        row = (self.xy[:, 1] / field_width * height).floor().long().clamp(0, height - 1)
        sensed = raster[world, :, row, col].transpose(0, 1)
        geom = torch.stack((self.xy[:, 0] / field_length,
                            self.xy[:, 1] / field_width,
                            self.heading_bin.to(self.xy.dtype)), -1)
        features = torch.cat((sensed, geom), -1)
        features = torch.where(self.node_valid[0, :, None], features,
                               torch.zeros_like(features))
        return features[None]


def _connected_region_ids(edges: list[list[int]], edge_mask: list[list[bool]],
                          node_xy: torch.Tensor, region_count: int,
                          node_valid: list[bool] | None = None) -> list[int]:
    """Partition every connected component into connected graph regions."""
    count = len(edges)
    node_valid = ([True] * count if node_valid is None else node_valid)
    adjacency = [[dst for dst, allowed in zip(row, mask) if allowed]
                 for row, mask in zip(edges, edge_mask)]
    component = [-1] * count
    components: list[list[int]] = []
    for start in range(count):
        if not node_valid[start]:
            continue
        if component[start] >= 0:
            continue
        label = len(components)
        members = []
        queue = [start]
        component[start] = label
        for current in queue:
            members.append(current)
            for nxt in adjacency[current]:
                if node_valid[nxt] and component[nxt] < 0:
                    component[nxt] = label
                    queue.append(nxt)
        components.append(members)
    if len(components) > region_count:
        raise ValueError(f"legal pose graph has {len(components)} components; "
                         f"cannot represent them in {region_count} regions")

    allocations = [1] * len(components)
    remaining = region_count - len(components)
    # Split spare regions over the largest components. Each region is seeded
    # at a stable, evenly spaced node and then expanded only along legal edges.
    for _ in range(remaining):
        largest = max(range(len(components)), key=lambda i: len(components[i]) / allocations[i])
        allocations[largest] += 1

    labels = [-1] * count
    region = 0
    for members, allocation in zip(components, allocations):
        seeds = [members[min(len(members) - 1, (2 * i + 1) * len(members) // (2 * allocation))]
                 for i in range(allocation)]
        queue = deque()
        for seed in seeds:
            if labels[seed] == -1:
                labels[seed] = region
                queue.append(seed)
                region += 1
        while queue:
            current = queue.popleft()
            for nxt in adjacency[current]:
                if labels[nxt] == -1:
                    labels[nxt] = labels[current]
                    queue.append(nxt)
    if any(node_valid[i] and label < 0 for i, label in enumerate(labels)):
        raise RuntimeError("region partition did not reach every legal pose")
    labels = [max(0, label) for label in labels]
    return labels


@torch.no_grad()
def build_legal_pose_graph(env, *, slot: int = 3, resolution_m: float | None = None,
                           region_count: int = REGION_COUNT) -> LegalPoseGraph:
    """Build an 8-heading static lattice from production planner geometry.

    Vertices use the production planner's configured grid and oriented robot
    footprint. Translation edges are checked with its swept-footprint routine;
    heading changes require legal endpoints and a legal intermediate heading.
    Dynamic robot circles remain snapshot-specific and are checked by the
    candidate route evaluator, not baked into this reusable graph.
    """
    if env.n < 1 or not 0 <= slot < 6:
        raise ValueError("environment must contain a valid robot slot")
    planner = env.planner
    if resolution_m is not None and not math.isclose(float(resolution_m), planner.resolution):
        raise ValueError("pose graph resolution must match the active planner grid")
    device = env.device
    dtype = env.sim.pose.dtype
    nx, ny = planner.nx, planner.ny
    xx, yy = torch.meshgrid(planner.x, planner.y, indexing="ij")
    xy_grid = torch.stack((xx, yy), -1)
    heading_values = torch.arange(HEADING_BINS, device=device, dtype=dtype) * (2 * math.pi / HEADING_BINS)
    length = env.sim.length[0, slot].expand(HEADING_BINS).clone()
    width = env.sim.width[0, slot].expand(HEADING_BINS).clone()
    blocked = planner._blocked(heading_values, length, width)
    legal = ~blocked
    # Stable legality sort creates the same row-major legal prefix as boolean
    # indexing, with a fixed grid-derived output capacity and no dynamic slice.
    capacity = HEADING_BINS * nx * ny
    permutation, inverse, node_valid = compact_legal_pose_mask(legal)
    legal_flat = legal.reshape(-1)
    all_xy = xy_grid[None].expand(HEADING_BINS, -1, -1, -1).reshape(capacity, 2)
    all_heading_bins = torch.arange(HEADING_BINS, device=device)[:, None, None].expand(
        -1, nx, ny).reshape(capacity)
    node_grid = all_xy[permutation]
    heading_bins = all_heading_bins[permutation]
    headings = heading_values[heading_bins]
    lookup_flat = torch.where(legal_flat, inverse, torch.full_like(inverse, -1))
    lookup = lookup_flat.reshape(HEADING_BINS, nx, ny)

    # Fixed slot order is the historical eight translations followed by the
    # two heading turns. Padded rows receive safe in-range IDs and false masks.
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1),
               (1, 1), (1, -1), (-1, 1), (-1, -1))
    original_h = permutation // (nx * ny)
    original_x = (permutation % (nx * ny)) // ny
    original_y = permutation % ny
    dst_columns, candidate_masks = [], []
    for dx, dy in offsets:
        target_x, target_y = original_x + dx, original_y + dy
        in_bounds = ((target_x >= 0) & (target_x < nx) &
                     (target_y >= 0) & (target_y < ny))
        target_original = (original_h * (nx * ny) + target_x.clamp(0, nx - 1) * ny
                           + target_y.clamp(0, ny - 1))
        target_node = lookup_flat[target_original]
        dst_columns.append(target_node.clamp_min(0))
        candidate_masks.append(node_valid & in_bounds & (target_node >= 0))
    for delta in (-1, 1):
        target_h = (original_h + delta) % HEADING_BINS
        target_original = target_h * (nx * ny) + original_x * ny + original_y
        target_node = lookup_flat[target_original]
        dst_columns.append(target_node.clamp_min(0))
        candidate_masks.append(node_valid & (target_node >= 0))
    edges = torch.stack(dst_columns, -1)
    candidate_valid = torch.stack(candidate_masks, -1)
    rotations = torch.tensor((False,) * 8 + (True, True),
                             device=device, dtype=torch.bool)[None, :]
    path = torch.stack((node_grid[:, None, :].expand(-1, 10, -1),
                        node_grid[edges]), dim=2)
    edge_length = length[heading_bins][:, None].expand(-1, 10)
    edge_width = width[heading_bins][:, None].expand(-1, 10)
    edge_heading = headings[:, None].expand(-1, 10)
    flat_path = path.reshape(-1, 2, 2)
    flat_length, flat_width = edge_length.reshape(-1), edge_width.reshape(-1)
    edge_clear = planner._footprint_path_clear(flat_path,
        edge_heading.reshape(-1), flat_length, flat_width).reshape(capacity, 10)
    # A discrete 45-degree rotation has a 22.5-degree swept midpoint. Check
    # this orientation on the same field grid before admitting the turn edge.
    angle_delta = torch.atan2(torch.sin(headings[edges] - headings[:, None]),
                              torch.cos(headings[edges] - headings[:, None]))
    mid_heading = torch.atan2(torch.sin(headings[:, None] + angle_delta * .5),
                              torch.cos(headings[:, None] + angle_delta * .5))
    mid_path = torch.stack((node_grid[:, None, :].expand(-1, 10, -1),
                            node_grid[:, None, :].expand(-1, 10, -1)), dim=2)
    mid_clear = planner._footprint_path_clear(mid_path.reshape(-1, 2, 2),
        mid_heading.reshape(-1), flat_length, flat_width).reshape(capacity, 10)
    edge_mask = candidate_valid & edge_clear & (~rotations | mid_clear)
    edge_mask &= node_valid[:, None] & node_valid[edges]
    if device.type == "cuda" and torch.version.hip is not None:
        # Preserve the historical ordered directed-BFS partition exactly on
        # device. The fixed 64-wide coarse row is padding-inert for the graph
        # message block and avoids a D2H maximum-degree read.
        from .meco_plan_a_shortest_path_hip import extension
        region_tensor, coarse_edges, coarse_mask, region_status = \
            extension().pose_regions(edges.contiguous(), edge_mask.contiguous(),
                                     node_valid.contiguous(), int(region_count))
        graph_status = region_status
        coarse_degree = region_count
    elif device.type == "cpu":
        edge_rows = edges.cpu().tolist()
        mask_rows = edge_mask.cpu().tolist()
        valid_rows = node_valid.tolist()
        if not any(valid_rows):
            regions = [0] * capacity
            graph_status = torch.ones((), device=device, dtype=torch.long)
        else:
            regions = _connected_region_ids(edge_rows, mask_rows, node_grid, region_count,
                                            node_valid=valid_rows)
            graph_status = torch.zeros((), device=device, dtype=torch.long)
        coarse_adj = [set() for _ in range(region_count)]
        region_tensor = torch.tensor(regions, device=device, dtype=torch.long)
        for src_node, row in enumerate(edge_rows):
            if not valid_rows[src_node] or not any(mask_rows[src_node]):
                continue
            src_region = regions[src_node]
            for dst_node, allowed in zip(row, mask_rows[src_node]):
                if allowed and valid_rows[dst_node]:
                    dst_region = regions[dst_node]
                    if src_region != dst_region:
                        coarse_adj[src_region].add(dst_region)
        coarse_degree = region_count
        coarse_edges = torch.zeros((region_count, coarse_degree), device=device,
                                   dtype=torch.long)
        coarse_mask = torch.zeros_like(coarse_edges, dtype=torch.bool)
        for region, neighbors in enumerate(coarse_adj):
            ordered = sorted(neighbors)
            if ordered:
                coarse_edges[region, :len(ordered)] = torch.tensor(ordered, device=device)
                coarse_mask[region, :len(ordered)] = True
    else:
        raise RuntimeError("Plan A pose graph requires HIP; CUDA must not use CPU graph fallback")
    return LegalPoseGraph(
        xy=node_grid, heading_bin=heading_bins, heading=headings,
        edges=edges[None],
        edge_mask=edge_mask[None],
        node_valid=node_valid[None],
        region_ids=region_tensor[None],
        coarse_edges=coarse_edges[None],
        coarse_edge_mask=coarse_mask[None],
        lookup=lookup, nx=nx, ny=ny, resolution_m=planner.resolution,
        region_count=region_count, graph_status=graph_status)
