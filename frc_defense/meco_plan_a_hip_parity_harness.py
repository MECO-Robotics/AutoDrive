"""Source-bound offline and explicitly leased Plan A HIP parity evidence.

The offline fixture path imports only Python/CPU reference code.  The device
path is deliberately not a production dispatcher and never opens the normal
Plan A gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

SCHEMA_VERSION = 1
EVIDENCE_CLASSES = {"offline_fixture_only", "hip_parity_candidate",
                    "hip_parity_profile_candidate"}
FIXTURE_RELATIVE = Path("tests/fixtures/plan_a_hip/manifest.json")
PINNED_SOURCES = (
    "frc_defense/meco_plan_a_shortest_path.py",
    "frc_defense/meco_pose_graph.py",
    "frc_defense/meco_plan_a_policy.py",
    "frc_defense/csrc/meco_plan_a_shortest_path_hip.cpp",
    "frc_defense/csrc/meco_plan_a_shortest_path_hip_kernel.cu",
    "frc_defense/meco_plan_a_shortest_path_hip.py",
)
OUTPUT_KEYS = ("route_cost", "physical_distance", "parent_ids",
               "parent_edge_m", "traversal_order")


def canonical_bytes(value) -> bytes:
    def normalize(item):
        if isinstance(item, float) and not math.isfinite(item):
            return "Infinity" if item > 0 else "-Infinity" if item < 0 else "NaN"
        if isinstance(item, dict):
            return {key: normalize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        return item
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_hashes(root: Path | None = None) -> dict[str, str]:
    root = root or _root()
    result = {}
    for relative in PINNED_SOURCES:
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise ValueError(f"pinned source is missing or escapes checkout: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _large_pose_lattice(recipe: dict) -> dict:
    """Build a deterministic open-field 8-heading lattice with degree 10."""
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
                source = (hbin * nx + ix) * ny + iy
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
                assert len(destinations) == 10
                edge_dst.append(destinations)
                edge_allowed.append(active)
    return {"xy": xy, "heading": heading, "edge_dst": edge_dst,
            "edge_allowed": edge_allowed,
            "root": int(recipe["root"]),
            "root_start_cost": float(recipe["root_start_cost"]),
            "root_connector_m": float(recipe["root_connector_m"])}


def materialize_case(case: dict) -> dict:
    if "recipe" in case:
        if case["recipe"].get("kind") != "open_pose_lattice_v1":
            raise ValueError(f"unknown fixture recipe: {case['id']}")
        return _large_pose_lattice(case["recipe"])
    graph = case.get("graph")
    if not isinstance(graph, dict):
        raise ValueError(f"fixture graph missing: {case['id']}")
    def decode(value):
        if value == "NaN":
            return math.nan
        if value == "Infinity":
            return math.inf
        if value == "-Infinity":
            return -math.inf
        if isinstance(value, list):
            return [decode(item) for item in value]
        if isinstance(value, dict):
            return {key: decode(item) for key, item in value.items()}
        return value
    return decode(graph)


def _route_inputs(graph: dict) -> tuple[dict, str]:
    from .meco_plan_a_shortest_path import precompute_plan_a_edge_weights
    if "xy" in graph:
        dst, costs, lengths, allowed = precompute_plan_a_edge_weights(
            graph["xy"], graph["heading"], graph["edge_dst"],
            graph["edge_allowed"])
        return {"edge_dst": dst, "edge_costs": costs,
                "edge_lengths": lengths, "edge_allowed": allowed}, "geometry"
    return {"edge_dst": graph["edge_dst"],
            "edge_costs": graph["edge_costs"],
            "edge_lengths": graph["edge_lengths"],
            "edge_allowed": graph["edge_allowed"]}, "preweighted"


def _route_hash(result: dict) -> str:
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        return value
    return sha256_bytes(canonical_bytes(finite_json(
        {key: result[key] for key in OUTPUT_KEYS})))


def _support_digest(graph: dict, result: dict, case_id: str) -> str:
    """Digest a canonical one-behavior support over the fixture route tree."""
    import torch
    from .meco_plan_a_policy import PlanAActionSupport, plan_a_support_digest
    if "xy" not in graph:
        # A geometry identity is required because the support digest binds the
        # graph endpoints, heading bins, edges and edge mask as well as route.
        xy = [[float(i), 0.] for i in range(len(result["route_cost"]))]
        heading = [0.] * len(xy)
        edge_dst, edge_allowed = graph["edge_dst"], graph["edge_allowed"]
        graph = {**graph, "xy": xy, "heading": heading}
    else:
        xy, heading = graph["xy"], graph["heading"]
        edge_dst, edge_allowed = graph["edge_dst"], graph["edge_allowed"]
    node_count = len(result["route_cost"])
    reachable = [i for i, distance in enumerate(result["route_cost"])
                 if math.isfinite(float(distance))]
    if not reachable:
        raise ValueError(f"fixture root did not reach a support node: {case_id}")
    root = int(graph["root"])
    physical = result["physical_distance"]
    fallback_node = min(reachable, key=lambda i: (physical[i], i))
    row_parent = [p if p >= 0 else root for p in result["parent_ids"]]
    jumps = [torch.tensor(row_parent, dtype=torch.long)]
    while (1 << len(jumps)) < node_count:
        prev = jumps[-1]
        jumps.append(prev.index_select(0, prev))
    tensor = lambda values, dtype=None: torch.tensor(values, dtype=dtype)
    xy_tensor = tensor(xy, torch.float64)
    heading_tensor = tensor(heading, torch.float64)
    heading_bins = tensor([int(round(value / (math.pi / 4.))) % 8
                            for value in heading], torch.long)
    degree = max(1, max((len(row) for row in edge_dst), default=0))
    padded_edges = [list(row) + [0] * (degree - len(row)) for row in edge_dst]
    padded_mask = [list(row) + [False] * (degree - len(row))
                   for row in edge_allowed]
    edges_tensor = tensor([padded_edges], torch.long)
    mask_tensor = tensor([padded_mask], torch.bool)
    pair_nodes = torch.tensor(reachable, dtype=torch.long)
    support = PlanAActionSupport(
        node_ids=torch.arange(node_count),
        behavior_ids=torch.zeros(len(reachable), dtype=torch.long),
        action_ids=torch.full((len(reachable),), 7, dtype=torch.long),
        pair_node_ids=pair_nodes,
        descriptor_ids=torch.full((len(reachable),), -1, dtype=torch.long),
        valid_mask=torch.ones(len(reachable), dtype=torch.bool),
        distances_m=tensor(physical, torch.float64),
        route_cost=tensor(result["route_cost"], torch.float64),
        parent_ids=tensor(result["parent_ids"], torch.long),
        parent_edge_m=tensor(result["parent_edge_m"], torch.float64),
        traversal_order=tensor(result["traversal_order"], torch.long),
        ancestor_jumps=torch.stack(jumps), root_node=root,
        root_connector_m=float(graph["root_connector_m"]),
        fallback_pair=reachable.index(fallback_node),
        goal_xy=xy_tensor, goal_heading=heading_tensor,
        snapshot_id=f"plan-a-hip-fixture-v1:{case_id}", support_digest="")
    graph_object = type("FixtureGraph", (), {
        "heading_bin": heading_bins, "xy": xy_tensor,
        "heading": heading_tensor, "edges": edges_tensor,
        "edge_mask": mask_tensor})()
    return plan_a_support_digest(support, graph_object)


def _fixture_result(root: Path) -> tuple[dict, list[dict]]:
    path = root / FIXTURE_RELATIVE
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Plan A HIP fixture schema")
    manifest_digest = raw.get("manifest_payload_sha256")
    unsigned = {key: value for key, value in raw.items()
                if key != "manifest_payload_sha256"}
    if manifest_digest != sha256_bytes(canonical_bytes(unsigned)):
        raise ValueError("Plan A HIP fixture manifest canonical digest mismatch")
    if raw.get("source_sha256") != source_hashes(root):
        raise ValueError("pinned Plan A HIP source hash changed; refresh/review fixtures")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture manifest must contain cases")
    if len({case.get("id") for case in cases}) != len(cases):
        raise ValueError("fixture IDs must be unique")
    results = []
    from .meco_plan_a_shortest_path import plan_a_dijkstra_cpu
    for case in cases:
        graph = materialize_case(case)
        graph_digest = sha256_bytes(canonical_bytes(graph))
        if graph_digest != case.get("graph_sha256"):
            raise ValueError(f"fixture graph hash mismatch: {case['id']}")
        try:
            route_inputs, route_kind = _route_inputs(graph)
            result = plan_a_dijkstra_cpu(**route_inputs,
                root=int(graph["root"]),
                root_start_cost=float(graph["root_start_cost"]),
                root_connector_m=float(graph["root_connector_m"]))
        except Exception as exc:
            if case.get("expected_error") != type(exc).__name__:
                raise ValueError(f"unexpected CPU fixture failure in {case['id']}: {exc}") from exc
            results.append({"id": case["id"], "status": "expected_rejection",
                            "error_type": type(exc).__name__,
                            "graph_sha256": graph_digest})
            continue
        if case.get("expected_error"):
            raise ValueError(f"fixture expected {case['expected_error']} but was accepted: {case['id']}")
        route_digest = _route_hash(result)
        if route_digest != case.get("route_output_sha256"):
            raise ValueError(f"CPU reference output mismatch: {case['id']}")
        support_digest = _support_digest(graph, result, case["id"])
        if support_digest != case.get("support_digest_sha256"):
            raise ValueError(f"Plan A support digest mismatch: {case['id']}")
        results.append({"id": case["id"], "status": "pass",
            "route_kind": route_kind, "node_count": len(result["route_cost"]),
            "active_edge_count": sum(sum(bool(v) for v in row)
                                      for row in graph["edge_allowed"]),
            "graph_sha256": graph_digest, "route_output_sha256": route_digest,
            "support_digest_sha256": support_digest})
    return raw, results


def run_offline(root: Path | None = None) -> dict:
    root = (root or _root()).resolve()
    fixture, results = _fixture_result(root)
    git = __import__("subprocess").run(["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(__import__("subprocess").run(["git", "-C", str(root), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout)
    fixture_path = root / FIXTURE_RELATIVE
    return {"schema_version": SCHEMA_VERSION,
        "evidence_class": "offline_fixture_only", "passed": True,
        "production_gate_open": False, "hip_imported": False,
        "device_discovered": False,
        "git_commit": git, "git_dirty": dirty,
        "source_sha256": fixture["source_sha256"],
        "fixture_manifest_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "cases": results}


def _load_lease_handoff(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("schema_version", "authorization_id", "supervisor_id",
                "training_leases_released", "exclusive_lease_id", "device_index",
                "device_model", "device_uuid", "authorized_by", "authorized_at_utc")
    missing = [key for key in required if key not in payload]
    if missing or payload.get("schema_version") != 1 or \
            payload.get("training_leases_released") is not True or \
            not all(isinstance(payload.get(key), str) and payload[key]
                    for key in required if key not in
                    ("schema_version", "training_leases_released", "device_index")):
        raise ValueError(f"invalid exclusive WX9100 lease handoff: {missing}")
    if payload["device_index"] not in (0, 1):
        raise ValueError("lease handoff must identify WX9100 index 0 or 1")
    return payload


def run_device(root: Path, device: str, lease_handoff_path: Path, *,
               warmups: int = 3, repeats: int = 10) -> dict:
    """Post-lease candidate runner. This function is intentionally never used offline."""
    if device not in ("cuda:0", "cuda:1"):
        raise ValueError("device verification requires explicit cuda:0 or cuda:1")
    if warmups < 1 or repeats < 3:
        raise ValueError("device profiling requires warmups >= 1 and repeats >= 3")
    fixture, cases = _fixture_result(root)
    lease = _load_lease_handoff(lease_handoff_path)
    import torch
    from . import meco_plan_a_shortest_path as shortest
    selected = shortest._require_wx9100_device(device)
    properties = torch.cuda.get_device_properties(selected)
    model = torch.cuda.get_device_name(selected)
    uuid = str(getattr(properties, "uuid", ""))
    if "wx9100" not in model.casefold().replace("-", "").replace("_", ""):
        raise RuntimeError("selected accelerator is not a WX9100")
    if int(selected.index) != lease["device_index"] or model != lease["device_model"] or \
            not uuid or uuid != lease["device_uuid"]:
        raise RuntimeError("WX9100 identity does not match the authorized lease handoff")
    if not shortest.PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED is False:
        raise RuntimeError("candidate harness must not run with production gate changed")
    # Import/compile is reached only after explicit device and lease checks.
    from .meco_plan_a_shortest_path_hip import extension
    ext = extension()
    from .meco_plan_a_shortest_path import plan_a_dijkstra_cpu
    output = []
    for case in fixture["cases"]:
        graph = materialize_case(case)
        if case.get("expected_error"):
            # Invalid fixtures remain in the contract report. Device failure is
            # checked by the native status contract without fallback.
            output.append({"id": case["id"], "status": "invalid_case_deferred"})
            continue
        inputs, kind = _route_inputs(graph)
        cpu = plan_a_dijkstra_cpu(**inputs, root=graph["root"],
            root_start_cost=graph["root_start_cost"],
            root_connector_m=graph["root_connector_m"])
        dst = torch.tensor(inputs["edge_dst"], device=selected, dtype=torch.long).contiguous()
        allowed = torch.tensor(inputs["edge_allowed"], device=selected, dtype=torch.bool).contiguous()
        arrays = {key: torch.tensor(inputs[key], device=selected,
                    dtype=torch.float64).contiguous()
                  for key in ("edge_costs", "edge_lengths")}
        xy = heading = None
        if kind == "geometry":
            xy = torch.tensor(graph["xy"], device=selected, dtype=torch.float64).contiguous()
            heading = torch.tensor(graph["heading"], device=selected,
                                   dtype=torch.float64).contiguous()
        def invoke():
            if kind == "geometry":
                route, physical, parent, parent_length, traversal = ext.dijkstra_graph(
                    xy, heading, dst, allowed, graph["root"],
                    graph["root_start_cost"], graph["root_connector_m"])
            else:
                route, physical, parent, parent_length, traversal = ext.dijkstra(
                    dst, arrays["edge_costs"], arrays["edge_lengths"], allowed,
                    graph["root"], graph["root_start_cost"], graph["root_connector_m"])
            return {"route_cost": route, "physical_distance": physical,
                    "parent_ids": parent, "parent_edge_m": parent_length,
                    "traversal_order": traversal}
        for _ in range(warmups):
            invoke()
        torch.cuda.synchronize(selected)
        torch.cuda.reset_peak_memory_stats(selected)
        timings = []
        device_result = None
        for _ in range(repeats):
            torch.cuda.synchronize(selected)
            start = time.perf_counter_ns()
            device_result = invoke()
            torch.cuda.synchronize(selected)
            timings.append((time.perf_counter_ns() - start) / 1.e6)
        device_host = {key: value.detach().cpu().tolist()
                       for key, value in device_result.items()}
        mismatches = [key for key in OUTPUT_KEYS if device_host[key] != cpu[key]]
        if mismatches:
            raise RuntimeError(f"exact HIP parity mismatch in {case['id']}: {mismatches}")
        output.append({"id": case["id"], "status": "candidate_pass",
            "route_kind": kind, "exact_output_match": True,
            "support_digest_match": _support_digest(graph, device_host, case["id"]) ==
                case["support_digest_sha256"],
            "warmups": warmups, "repeats": repeats,
            "elapsed_ms": timings,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(selected)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(selected))})
    return {"schema_version": SCHEMA_VERSION,
        "evidence_class": "hip_parity_profile_candidate",
        "production_gate_open": False, "passed": all(
            row.get("status") == "candidate_pass" and row.get("support_digest_match")
            for row in output if row.get("status") != "invalid_case_deferred"),
        "device": {"index": int(selected.index), "model": model, "uuid": uuid},
        "lease_authorization": lease, "fixture_manifest_sha256":
            hashlib.sha256((root / FIXTURE_RELATIVE).read_bytes()).hexdigest(),
        "cases": output}
