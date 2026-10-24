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


@dataclass(frozen=True)
class LegalPoseGraph:
    xy: torch.Tensor                 # [N,2], field meters
    heading_bin: torch.Tensor        # [N]
    heading: torch.Tensor            # [N], radians
    edges: torch.Tensor              # [1,N,D], legal destination node IDs
    edge_mask: torch.Tensor          # [1,N,D]
    region_ids: torch.Tensor         # [1,N], connected graph regions
    coarse_edges: torch.Tensor       # [1,R,Dc], legal inter-region transitions
    coarse_edge_mask: torch.Tensor   # [1,R,Dc]
    lookup: torch.Tensor             # [8,nx,ny], -1 for illegal poses
    nx: int
    ny: int
    resolution_m: float
    region_count: int

    @property
    def node_count(self) -> int:
        return self.xy.shape[0]

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
        return torch.cat((sensed, geom), -1)[None]


def _connected_region_ids(edges: list[list[int]], edge_mask: list[list[bool]],
                          node_xy: torch.Tensor, region_count: int) -> list[int]:
    """Partition every connected component into connected graph regions."""
    count = len(edges)
    adjacency = [[dst for dst, allowed in zip(row, mask) if allowed]
                 for row, mask in zip(edges, edge_mask)]
    component = [-1] * count
    components: list[list[int]] = []
    for start in range(count):
        if component[start] >= 0:
            continue
        label = len(components)
        members = []
        queue = [start]
        component[start] = label
        for current in queue:
            members.append(current)
            for nxt in adjacency[current]:
                if component[nxt] < 0:
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
    if any(label < 0 for label in labels):
        raise RuntimeError("region partition did not reach every legal pose")
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
    lookup = torch.full((HEADING_BINS, nx, ny), -1, device=device, dtype=torch.long)
    total = int(legal.sum().item())
    if total == 0:
        raise ValueError("static field geometry contains no legal SE(2) poses")
    lookup[legal] = torch.arange(total, device=device)
    node_grid = xy_grid[None].expand(HEADING_BINS, -1, -1, -1)[legal]
    heading_bins = torch.arange(HEADING_BINS, device=device)[:, None, None].expand(-1, nx, ny)[legal]
    headings = heading_values[heading_bins]

    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1),
               (1, 1), (1, -1), (-1, 1), (-1, -1))
    candidate_src: list[torch.Tensor] = []
    candidate_dst: list[torch.Tensor] = []
    candidate_slot: list[torch.Tensor] = []
    candidate_rot: list[torch.Tensor] = []
    for hbin in range(HEADING_BINS):
        for slot, (dx, dy) in enumerate(offsets):
            sx0, sx1 = max(0, -dx), min(nx, nx - dx)
            sy0, sy1 = max(0, -dy), min(ny, ny - dy)
            source_legal = legal[hbin, sx0:sx1, sy0:sy1]
            target_legal = legal[hbin, sx0 + dx:sx1 + dx, sy0 + dy:sy1 + dy]
            coords = torch.nonzero(source_legal & target_legal, as_tuple=False)
            if coords.numel():
                ix, iy = coords[:, 0] + sx0, coords[:, 1] + sy0
                candidate_src.append(lookup[hbin, ix, iy])
                candidate_dst.append(lookup[hbin, ix + dx, iy + dy])
                candidate_slot.append(torch.full_like(candidate_src[-1], slot))
                candidate_rot.append(torch.zeros_like(candidate_src[-1], dtype=torch.bool))
        for side, delta in enumerate((-1, 1)):
            next_h = (hbin + delta) % HEADING_BINS
            valid = legal[hbin] & legal[next_h]
            coords = torch.nonzero(valid, as_tuple=False)
            if coords.numel():
                ix, iy = coords[:, 0], coords[:, 1]
                candidate_src.append(lookup[hbin, ix, iy])
                candidate_dst.append(lookup[next_h, ix, iy])
                candidate_slot.append(torch.full_like(candidate_src[-1], 8 + side))
                candidate_rot.append(torch.ones_like(candidate_src[-1], dtype=torch.bool))

    if not candidate_src:
        raise ValueError("legal pose lattice has no transitions")
    src = torch.cat(candidate_src)
    dst = torch.cat(candidate_dst)
    slots = torch.cat(candidate_slot)
    is_rotation = torch.cat(candidate_rot)
    path = torch.stack((node_grid[src], node_grid[dst]), dim=1)
    # `length`/`width` above are heading-sized vectors used by _blocked;
    # transition source indices are legal-node IDs, so map them through each
    # node's heading bin before indexing the footprint dimensions.
    edge_length = length[heading_bins[src]]
    edge_width = width[heading_bins[src]]
    edge_heading = headings[src]
    edge_clear = planner._footprint_path_clear(path, edge_heading,
        edge_length, edge_width)
    # A discrete 45-degree rotation has a 22.5-degree swept midpoint. Check
    # this orientation on the same field grid before admitting the turn edge.
    mid_heading = torch.atan2(torch.sin(headings[src] +
        torch.atan2(torch.sin(headings[dst] - headings[src]),
                    torch.cos(headings[dst] - headings[src])) * .5),
        torch.cos(headings[src] +
        torch.atan2(torch.sin(headings[dst] - headings[src]),
                    torch.cos(headings[dst] - headings[src])) * .5))
    mid_clear = planner._footprint_path_clear(
        node_grid[src, None].expand(-1, 2, -1), mid_heading,
        edge_length, edge_width)
    edge_clear &= ~is_rotation | mid_clear

    # Store a fixed maximum degree of 10 (8 translations + 2 turns).
    degree = 10
    edges = torch.zeros((total, degree), device=device, dtype=torch.long)
    edge_mask = torch.zeros((total, degree), device=device, dtype=torch.bool)
    edges[src, slots] = dst
    edge_mask[src, slots] = edge_clear
    edge_rows = edges.cpu().tolist()
    mask_rows = edge_mask.cpu().tolist()
    regions = _connected_region_ids(edge_rows, mask_rows, node_grid, region_count)
    coarse_adj = [set() for _ in range(region_count)]
    region_tensor = torch.tensor(regions, device=device, dtype=torch.long)
    for src_node, row in enumerate(edge_rows):
        if not any(mask_rows[src_node]):
            continue
        src_region = regions[src_node]
        for dst_node, allowed in zip(row, mask_rows[src_node]):
            if allowed:
                dst_region = regions[dst_node]
                if src_region != dst_region:
                    coarse_adj[src_region].add(dst_region)
    coarse_degree = max(1, max((len(row) for row in coarse_adj), default=0))
    coarse_edges = torch.zeros((region_count, coarse_degree), device=device,
                               dtype=torch.long)
    coarse_mask = torch.zeros_like(coarse_edges, dtype=torch.bool)
    for region, neighbors in enumerate(coarse_adj):
        ordered = sorted(neighbors)
        if ordered:
            coarse_edges[region, :len(ordered)] = torch.tensor(ordered, device=device)
            coarse_mask[region, :len(ordered)] = True
    return LegalPoseGraph(
        xy=node_grid, heading_bin=heading_bins, heading=headings,
        edges=edges[None],
        edge_mask=edge_mask[None],
        region_ids=region_tensor[None],
        coarse_edges=coarse_edges[None],
        coarse_edge_mask=coarse_mask[None],
        lookup=lookup, nx=nx, ny=ny, resolution_m=planner.resolution,
        region_count=region_count)
