"""Deterministic, device-free geometry recipes for frozen Plan A HIP cases."""
from __future__ import annotations

import math


def open_pose_lattice(recipe: dict) -> dict:
    """Build the frozen open-field 8-heading lattice shape, degree 10."""
    nx, ny = int(recipe["nx"]), int(recipe["ny"])
    if nx < 2 or ny < 2 or nx * ny * 8 != recipe["node_count"]:
        raise ValueError("invalid large pose-lattice fixture recipe")
    spacing = float(recipe["spacing_m"])
    xy, heading, edge_dst, edge_allowed = [], [], [], []
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1),
               (1, 1), (1, -1), (-1, 1), (-1, -1))
    for hbin in range(8):
        theta = hbin * (2. * math.pi / 8.)
        for ix in range(nx):
            for iy in range(ny):
                xy.append([ix * spacing, iy * spacing])
                heading.append(theta)
                destinations, active = [], []
                for dx, dy in offsets:
                    tx, ty = ix + dx, iy + dy
                    destinations.append((hbin * nx + tx) * ny + ty
                                        if 0 <= tx < nx and 0 <= ty < ny else 0)
                    active.append(0 <= tx < nx and 0 <= ty < ny)
                for delta in (-1, 1):
                    destinations.append((((hbin + delta) % 8) * nx + ix) * ny + iy)
                    active.append(True)
                if len(destinations) != 10:
                    raise RuntimeError("pose-lattice recipe degree changed")
                edge_dst.append(destinations)
                edge_allowed.append(active)
    return {"xy": xy, "heading": heading, "edge_dst": edge_dst,
            "edge_allowed": edge_allowed,
            "root": int(recipe["root"]),
            "root_start_cost": float(recipe["root_start_cost"]),
            "root_connector_m": float(recipe["root_connector_m"])}
