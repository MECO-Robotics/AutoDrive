"""Differentiable Plan A penalty projection and frozen-route integration.

Model outputs are dimensionless nonnegative penalties per cardinal planner
step. The weighted AD* API accepts XY maps, so heading-specific A fields are
projected by averaging only legal headings at each XY cell. PPO integrates the
same projected map over the retained XY route, then converts normalized grid
edge units to reward points with an explicit configured scale.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


PENALTY_MAP_PROJECTION = "mean_over_legal_heading_bins_xy_grid_v1"
PENALTY_MAP_UNITS = "dimensionless_additive_penalty_per_cardinal_grid_step"
PENALTY_PATH_UNITS = "normalized_grid_edge_units"
PLAN_A_PPO_RECOMPUTATION_CONTRACT = (
    "projected_xy_penalty_node_integral_over_frozen_legal_se2_shortest_path_tree"
    "_including_root_pose_connector_v1")


def plan_a_ppo_contract_metadata() -> dict[str, str]:
    """Return the canonical Plan A PPO recomputation provenance field."""
    return {"ppo_recomputation": PLAN_A_PPO_RECOMPUTATION_CONTRACT}


def project_pose_penalty(track: str, penalty: torch.Tensor, legal_lookup: torch.Tensor,
                         planner, field_length: float, field_width: float,
                         *, mode: str = "learned", node_valid=None) -> torch.Tensor:
    """Project an A track's output to the XY map consumed by weighted AD*."""
    if track not in ("A1", "A2", "A3"):
        raise ValueError("Plan A penalty projection only supports A1/A2/A3")
    if mode not in ("learned", "p0"):
        raise ValueError("p_mode must be 'learned' or 'p0'")
    nx = planner.nx if hasattr(planner, "nx") else int(planner["nx"])
    ny = planner.ny if hasattr(planner, "ny") else int(planner["ny"])
    expected = (nx, ny)
    if legal_lookup.shape != (8, *expected):
        raise ValueError("legal pose lookup must match the planner XY grid")
    if mode == "p0":
        return penalty.new_zeros(expected)
    if penalty.ndim != (4 if track == "A1" else 2) or penalty.shape[0] != 1:
        raise ValueError(f"{track} penalty field has an invalid shape")
    penalty = torch.nan_to_num(penalty).clamp_min(0.)
    legal = legal_lookup >= 0
    if track == "A1":
        if penalty.shape[1] != 8:
            raise ValueError("A1 penalty field must contain eight heading bins")
        height, width = penalty.shape[-2:]
        x = planner.x if hasattr(planner, "x") else planner["x"]
        y = planner.y if hasattr(planner, "y") else planner["y"]
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        # The observation raster's field-aligned coordinates use floor(x/L*W).
        gx = 2. * xx / float(field_length) + 1. / width - 1.
        gy = 2. * yy / float(field_width) + 1. / height - 1.
        sampling_grid = torch.stack((gx, gy), -1)[None]
        heading_maps = F.grid_sample(penalty, sampling_grid, mode="bilinear",
            padding_mode="border", align_corners=False)[0]
    else:
        if node_valid is None:
            raise ValueError("A2/A3 penalty projection requires node_valid")
        if (node_valid.shape != (1, penalty.shape[1]) or node_valid.dtype != torch.bool):
            raise ValueError("A2/A3 penalty projection requires the fixed-capacity node mask")
        if node_valid.device != penalty.device:
            raise ValueError("A2/A3 node validity must remain with the penalty tensor")
        safe_penalty = torch.where(node_valid[0], penalty[0], torch.zeros_like(penalty[0]))
        heading_maps = safe_penalty[legal_lookup.clamp_min(0)]
    counts = legal.sum(0)
    projected = (heading_maps * legal.to(heading_maps.dtype)).sum(0)
    return projected / counts.clamp_min(1).to(projected.dtype)


def integrate_frozen_routes(cost_map: torch.Tensor, route_records, resolution_m: float,
                            *, map_scale: float, points_per_grid_edge: float) -> torch.Tensor:
    """Differentiably integrate an XY penalty map over retained route segments.

    Segment splitting and cell selection match the weighted route evaluator's
    grid-edge integral. Route coordinates are frozen evidence; map values retain
    gradients through the current model output.
    """
    if cost_map.ndim != 2 or not bool(torch.isfinite(cost_map).all()):
        raise ValueError("cost_map must be a finite [nx,ny] tensor")
    if bool((cost_map < 0).any()):
        raise ValueError("cost_map must be nonnegative")
    resolution_m, map_scale, points_per_grid_edge = map(float,
        (resolution_m, map_scale, points_per_grid_edge))
    if (not all(math.isfinite(value) and value >= 0.
                for value in (resolution_m, map_scale, points_per_grid_edge))
            or resolution_m == 0.):
        raise ValueError("route integral scales must be finite and nonnegative")
    nx, ny = cost_map.shape
    integrals = []
    for record in route_records:
        points = record.get("path_points_m")
        total = cost_map.new_zeros(())
        if points and len(points) > 1:
            for first, second in zip(points, points[1:]):
                x0, y0 = map(float, first)
                x1, y1 = map(float, second)
                dx, dy = x1 - x0, y1 - y0
                length = math.hypot(dx, dy)
                if length == 0.:
                    continue
                crossings = {0., 1.}
                if dx != 0.:
                    low, high = sorted((x0, x1))
                    for cell in range(math.floor(low / resolution_m) + 1,
                                      math.ceil(high / resolution_m)):
                        fraction = (cell * resolution_m - x0) / dx
                        if 0. < fraction < 1.:
                            crossings.add(fraction)
                if dy != 0.:
                    low, high = sorted((y0, y1))
                    for cell in range(math.floor(low / resolution_m) + 1,
                                      math.ceil(high / resolution_m)):
                        fraction = (cell * resolution_m - y0) / dy
                        if 0. < fraction < 1.:
                            crossings.add(fraction)
                stations = sorted(crossings)
                for start, end in zip(stations, stations[1:]):
                    middle = (start + end) * .5
                    mx = x0 + dx * middle
                    my = y0 + dy * middle
                    ix = min(nx - 1, max(0, math.floor(mx / resolution_m)))
                    iy = min(ny - 1, max(0, math.floor(my / resolution_m)))
                    fraction_length = length * (end - start) / resolution_m
                    total = total + cost_map[ix, iy] * fraction_length
        integrals.append(total * map_scale * points_per_grid_edge)
    return torch.stack(integrals)[None]
