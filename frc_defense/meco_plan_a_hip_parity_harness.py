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
import subprocess
import time

SCHEMA_VERSION = 1
EVIDENCE_CLASSES = {"offline_fixture_only", "hip_parity_candidate",
                    "hip_parity_profile_candidate"}
FIXTURE_RELATIVE = Path("tests/fixtures/plan_a_hip/manifest.json")
PINNED_SOURCES = (
    "frc_defense/meco_plan_a_shortest_path.py",
    "frc_defense/meco_pose_graph.py",
    "frc_defense/meco_plan_a_policy.py",
    "frc_defense/meco_plan_a_hip_fixture_graph.py",
    "frc_defense/csrc/meco_plan_a_shortest_path_hip.cpp",
    "frc_defense/csrc/meco_plan_a_shortest_path_hip_kernel.cu",
    "frc_defense/meco_plan_a_shortest_path_hip.py",
    "frc_defense/meco_device_lease.py",
    "frc_defense/meco_gpu_attestation.py",
    "frc_defense/meco_training_manifest.py",
    "frc_defense/meco_plan_a_hip_parity_harness.py",
    "scripts/verify_plan_a_hip_parity.py",
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


def _git_source_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout)
    return commit, dirty


def _large_pose_lattice(recipe: dict) -> dict:
    from .meco_plan_a_hip_fixture_graph import open_pose_lattice
    return open_pose_lattice(recipe)


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



def _field_hash(value) -> str:
    return sha256_bytes(canonical_bytes(value))


def _output_field_hashes(result: dict) -> dict[str, str]:
    return {key: _field_hash(result[key]) for key in OUTPUT_KEYS}


def _fixture_tensor_hashes(graph: dict) -> dict[str, str]:
    route_inputs, _ = _route_inputs(graph)
    names = ("xy", "heading", "edge_dst", "edge_costs", "edge_lengths",
             "edge_allowed")
    values = {key: graph[key] for key in ("xy", "heading", "edge_dst", "edge_allowed")
              if key in graph}
    values.update({key: route_inputs[key] for key in
                   ("edge_costs", "edge_lengths")})
    return {key: _field_hash(values[key]) for key in names if key in values}

def _support_digest(graph: dict, result: dict, case_id: str) -> str:
    """Digest a canonical one-behavior support over the fixture route tree."""
    import torch
    from .meco_plan_a_policy import (PlanAActionSupport,
        plan_a_support_digest_cpu_reference)
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
    jumps = [torch.tensor(row_parent, dtype=torch.long, device="cpu")]
    while (1 << len(jumps)) < node_count:
        prev = jumps[-1]
        jumps.append(prev.index_select(0, prev))
    tensor = lambda values, dtype=None: torch.tensor(values, dtype=dtype, device="cpu")
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
    pair_nodes = torch.tensor(reachable, dtype=torch.long, device="cpu")
    support = PlanAActionSupport(
        node_ids=torch.arange(node_count, device="cpu"),
        behavior_ids=torch.zeros(len(reachable), dtype=torch.long, device="cpu"),
        action_ids=torch.full((len(reachable),), 7, dtype=torch.long, device="cpu"),
        pair_node_ids=pair_nodes,
        descriptor_ids=torch.full((len(reachable),), -1, dtype=torch.long, device="cpu"),
        valid_mask=torch.ones(len(reachable), dtype=torch.bool, device="cpu"),
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
    return plan_a_support_digest_cpu_reference(support, graph_object)


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
    if any(not isinstance(case, dict) or not isinstance(case.get("id"), str)
           for case in cases) or len({case.get("id") for case in cases}) != len(cases):
        raise ValueError("fixture IDs must be unique")
    by_id = {case["id"]: case for case in cases}
    required = {"equal_distance_node_and_edge_slot_ties",
        "strict_relaxation_epsilon", "disconnected_defaults",
        "geometry_root_connector_and_turn_cost",
        "invalid_active_edge_destination", "invalid_active_geometry",
        "production_pose_lattice_8heading_100x80"}
    if not required.issubset(by_id):
        raise ValueError(f"required frozen fixture cases are missing: {sorted(required - set(by_id))}")
    if by_id["invalid_active_edge_destination"].get("expected_error") != "IndexError" or \
            any(by_id[key].get("device_rejects") is not True for key in
                ("invalid_active_edge_destination", "invalid_active_geometry")):
        raise ValueError("invalid active-edge and geometry fixtures must require device rejection")
    large = by_id["production_pose_lattice_8heading_100x80"].get("recipe", {})
    if large.get("node_count") != 64000 or large.get("nx") * large.get("ny") * 8 != 64000:
        raise ValueError("large pose-lattice fixture no longer represents the pinned production-size shape")
    results = []
    from .meco_plan_a_shortest_path import plan_a_dijkstra_cpu
    for case in cases:
        graph = materialize_case(case)
        graph_digest = sha256_bytes(canonical_bytes(graph))
        tensor_hashes = _fixture_tensor_hashes(graph)
        if tensor_hashes != case.get("tensor_sha256"):
            raise ValueError(f"fixture tensor hash mismatch: {case['id']}")
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
                            "graph_sha256": graph_digest,
                            "tensor_sha256": tensor_hashes,
                            "expected_error": case.get("expected_error"),
                            "device_rejects": case.get("device_rejects", False)})
            continue
        if case.get("expected_error"):
            raise ValueError(f"fixture expected {case['expected_error']} but was accepted: {case['id']}")
        route_digest = _route_hash(result)
        if route_digest != case.get("route_output_sha256"):
            raise ValueError(f"CPU reference output mismatch: {case['id']}")
        output_field_hashes = _output_field_hashes(result)
        if output_field_hashes != case.get("route_output_field_sha256"):
            raise ValueError(f"CPU reference field digest mismatch: {case['id']}")
        support_digest = _support_digest(graph, result, case["id"])
        if support_digest != case.get("support_digest_sha256"):
            raise ValueError(f"Plan A support digest mismatch: {case['id']}")
        results.append({"id": case["id"], "status": "pass",
            "route_kind": route_kind, "node_count": len(result["route_cost"]),
            "active_edge_count": sum(sum(bool(v) for v in row)
                                      for row in graph["edge_allowed"]),
            "graph_sha256": graph_digest, "tensor_sha256": tensor_hashes,
            "route_output_sha256": route_digest,
            "route_output_field_sha256": output_field_hashes,
            "support_digest_sha256": support_digest,
            "expected_route_output_sha256": case["route_output_sha256"],
            "expected_support_digest_sha256": case["support_digest_sha256"]})
    return raw, results


def run_offline(root: Path | None = None) -> dict:
    root = (root or _root()).resolve()
    fixture, results = _fixture_result(root)
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout)
    fixture_path = root / FIXTURE_RELATIVE
    import torch
    return {"schema_version": SCHEMA_VERSION,
        "evidence_class": "offline_fixture_only", "passed": True,
        "production_gate_open": False, "training_configuration_modified": False,
        "hip_imported": False, "hip_compiled": False, "device_discovered": False,
        "git_commit": git, "git_dirty": dirty,
        "source_sha256": fixture["source_sha256"],
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256((root / "scripts/verify_plan_a_hip_parity.py").read_bytes()).hexdigest(),
        "fixture_manifest_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "fixture_cases_expected": sorted(case["id"] for case in fixture["cases"]),
        "fixture_cases_reported": sorted(row["id"] for row in results),
        "cases_expected": len(fixture["cases"]),
        "cases_reported": len(results),
        "cases": results}


def _device_inputs(graph: dict, kind: str, selected, torch, ext):
    route_inputs = _route_inputs(graph)[0]
    slots = max(1, max((len(row) for row in route_inputs["edge_dst"]), default=0))
    def pad(rows, fill):
        return [list(row) + [fill] * (slots - len(row)) for row in rows]
    edge_dst_host = pad(route_inputs["edge_dst"], 0)
    edge_allowed_host = pad(route_inputs["edge_allowed"], False)
    edge_costs_host = pad(route_inputs["edge_costs"], math.inf)
    edge_lengths_host = pad(route_inputs["edge_lengths"], 0.)
    def prepare_preweighted():
        edge_dst = torch.tensor(edge_dst_host, device=selected,
                                dtype=torch.long).contiguous()
        edge_allowed = torch.tensor(edge_allowed_host, device=selected,
                                     dtype=torch.bool).contiguous()
        edge_costs = torch.tensor(edge_costs_host, device=selected,
                                  dtype=torch.float64).contiguous()
        edge_lengths = torch.tensor(edge_lengths_host, device=selected,
                                    dtype=torch.float64).contiguous()
        def invoke():
            return ext.dijkstra(edge_dst, edge_costs, edge_lengths, edge_allowed,
                graph["root"], graph["root_start_cost"], graph["root_connector_m"])
        return invoke
    paths = {"preweighted": prepare_preweighted}
    if kind == "geometry":
        def prepare_graph_geometry():
            edge_dst = torch.tensor(edge_dst_host, device=selected,
                                    dtype=torch.long).contiguous()
            edge_allowed = torch.tensor(edge_allowed_host, device=selected,
                                        dtype=torch.bool).contiguous()
            xy = torch.tensor(graph["xy"], device=selected,
                              dtype=torch.float64).contiguous()
            heading = torch.tensor(graph["heading"], device=selected,
                                   dtype=torch.float64).contiguous()
            node_valid = torch.ones((xy.shape[0],), device=selected,
                                    dtype=torch.bool)
            def invoke():
                return ext.dijkstra_graph(xy, heading, edge_dst, edge_allowed, node_valid,
                    graph["root"], graph["root_start_cost"], graph["root_connector_m"])
            return invoke
        paths["graph_geometry"] = prepare_graph_geometry
    return paths


def _compiler_metadata(torch) -> dict:
    compiler_path = subprocess.run(["which", "hipcc"], capture_output=True,
                                   text=True, check=False).stdout.strip() or None
    compiler = subprocess.run([compiler_path, "--version"], capture_output=True,
                              text=True, check=False) if compiler_path else None
    bundled = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
    import os
    from torch.utils import cpp_extension
    rocm_value = (os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
                  (str(bundled) if bundled.exists() else cpp_extension._find_rocm_home()))
    rocm = Path(rocm_value).resolve() if rocm_value else None
    bitcode = next((path for path in ((rocm / "lib/llvm/amdgcn/bitcode",
        rocm / "amdgcn/bitcode") if rocm else ()) if path.is_dir()), None)
    return {"hipcc_path": compiler_path,
        "hipcc_exit_code": compiler.returncode if compiler else None,
        "hipcc_version": ((compiler.stdout + compiler.stderr).strip()
                          if compiler else None),
        "rocm_home": str(rocm) if rocm else None,
        "device_bitcode_path": str(bitcode) if bitcode else None,
        "hipconfig_version": (subprocess.run(["hipconfig", "--version"],
            capture_output=True, text=True, check=False).stdout.strip()
            if subprocess.run(["which", "hipconfig"], capture_output=True,
                text=True, check=False).stdout.strip() else None)}


def run_device(root: Path, device: str, *,
               warmups: int = 3, repeats: int = 10) -> dict:
    """Run only while holding the repository lease and physical attestation."""
    if device not in ("cuda:0", "cuda:1"):
        raise ValueError("device verification requires explicit cuda:0 or cuda:1")
    if warmups < 1 or repeats < 3:
        raise ValueError("device profiling requires warmups >= 1 and repeats >= 3")
    root = root.resolve()
    start_commit, start_dirty = _git_source_state(root)
    if start_dirty:
        raise RuntimeError("device candidate requires a clean checkout at invocation start")
    start_sources = source_hashes(root)
    fixture_path = root / FIXTURE_RELATIVE
    start_fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    fixture, _ = _fixture_result(root)
    if fixture.get("source_sha256") != start_sources:
        raise RuntimeError("fixture source pins do not match invocation-start source identity")
    from .meco_device_lease import lease_devices
    from .meco_gpu_attestation import verify_post_lease_wx9100
    lease = lease_devices((device,), owner="plan-a-hip-parity-profile")
    with lease:
        attestation = verify_post_lease_wx9100(lease, (device,))
        report = _run_device_under_lease(root, device, fixture, lease, attestation,
                                         warmups=warmups, repeats=repeats)
    end_commit, end_dirty = _git_source_state(root)
    end_sources = source_hashes(root)
    end_fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    if (end_commit != start_commit or end_dirty or end_sources != start_sources or
            end_fixture_sha256 != start_fixture_sha256):
        raise RuntimeError("checkout, pinned sources, or fixture changed during device candidate run")
    report.update({"git_commit": start_commit, "git_dirty": False,
        "source_sha256": start_sources,
        "fixture_manifest_sha256": start_fixture_sha256,
        "source_identity_verified_at_start_and_end": True})
    return report


def _run_device_under_lease(root: Path, device: str, fixture: dict, lease,
                            attestation: dict, *, warmups: int, repeats: int) -> dict:
    import torch
    from . import meco_plan_a_shortest_path as shortest
    selected = shortest._require_wx9100_device(device)
    properties = torch.cuda.get_device_properties(selected)
    model = torch.cuda.get_device_name(selected)
    uuid = str(getattr(properties, "uuid", ""))
    if "wx9100" not in model.casefold().replace("-", "").replace("_", "").replace(" ", ""):
        raise RuntimeError("selected accelerator is not a WX9100")
    attested = {row["device"]: row for row in attestation["devices"]}
    if (device not in attested or int(selected.index) != int(device.split(":")[1]) or
            model != attested[device]["model"] or not uuid or
            uuid.casefold() != str(attested[device]["hip_uuid"]).casefold()):
        raise RuntimeError("selected WX9100 identity does not match post-lease physical attestation")
    if shortest.PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED is not False:
        raise RuntimeError("candidate harness must not run with production gate changed")
    # Import/compile is reached only after explicit device and lease checks.
    from .meco_plan_a_shortest_path_hip import extension
    import frc_defense.meco_plan_a_shortest_path_hip as hip_loader
    extension_was_loaded = hip_loader._EXTENSION is not None
    compile_start = time.perf_counter_ns()
    ext = extension()
    compile_elapsed_ms = (time.perf_counter_ns() - compile_start) / 1.e6
    from .meco_plan_a_shortest_path import plan_a_dijkstra_cpu
    output = []
    cases_by_id = {case["id"]: case for case in fixture["cases"]}
    for case in fixture["cases"]:
        graph = materialize_case(case)
        kind = case.get("route_kind", "geometry" if "xy" in graph else "preweighted")
        path_invocations = _device_inputs(graph, kind, selected, torch, ext)
        if case.get("device_rejects"):
            rejection_rows = []
            for path_name, prepare in path_invocations.items():
                invoke = prepare()
                try:
                    invoke()
                    torch.cuda.synchronize(selected)
                except Exception as exc:
                    rejection_rows.append({"path": path_name,
                        "status": "candidate_rejected_input",
                        "device_error_type": type(exc).__name__,
                        "device_error": str(exc)[:500]})
                    continue
                raise RuntimeError(
                    f"HIP {path_name} accepted invalid fixture input: {case['id']}")
            output.append({"id": case["id"], "status": "candidate_rejected_input",
                "graph_sha256": case["graph_sha256"],
                "tensor_sha256": case["tensor_sha256"],
                "cpu_expected_error": case.get("expected_error"),
                "expected_route_output_sha256": case.get("route_output_sha256"),
                "expected_route_output_field_sha256": case.get("route_output_field_sha256"),
                "expected_support_digest_sha256": case.get("support_digest_sha256"),
                "expected_device_rejection": True, "hip_paths": rejection_rows})
            continue
        if case.get("expected_error"):
            raise RuntimeError(
                f"fixture error contract missing device rejection: {case['id']}")
        route_inputs, _ = _route_inputs(graph)
        cpu = plan_a_dijkstra_cpu(**route_inputs, root=graph["root"],
            root_start_cost=graph["root_start_cost"],
            root_connector_m=graph["root_connector_m"])
        active_edges = sum(sum(bool(value) for value in row)
                           for row in graph["edge_allowed"])
        node_count = len(graph["edge_dst"])
        edge_slots = len(graph["edge_dst"][0]) if node_count else 0
        hip_rows = []
        for path_name, prepare in path_invocations.items():
            # Input construction and transfer stay outside the timing boundary.
            # HIP timing covers host launch through synchronized device completion.
            invoke = prepare()
            def cpu_path_invoke():
                cpu_path_inputs = (_route_inputs(graph)[0]
                    if path_name == "graph_geometry" else route_inputs)
                return plan_a_dijkstra_cpu(**cpu_path_inputs, root=graph["root"],
                    root_start_cost=graph["root_start_cost"],
                    root_connector_m=graph["root_connector_m"])
            cpu_first_start = time.perf_counter_ns()
            cpu_path_invoke()
            cpu_first_ms = (time.perf_counter_ns() - cpu_first_start) / 1.e6
            cpu_repeat_ms = []
            for _ in range(repeats):
                cpu_start = time.perf_counter_ns()
                cpu_path_invoke()
                cpu_repeat_ms.append((time.perf_counter_ns() - cpu_start) / 1.e6)
            median_cpu = sorted(cpu_repeat_ms)[len(cpu_repeat_ms) // 2]
            def invoke_mapping():
                route, physical, parent, parent_length, traversal = invoke()
                return {"route_cost": route, "physical_distance": physical,
                    "parent_ids": parent, "parent_edge_m": parent_length,
                    "traversal_order": traversal}
            first_use_start = time.perf_counter_ns()
            invoke_mapping()
            torch.cuda.synchronize(selected)
            first_use_ms = (time.perf_counter_ns() - first_use_start) / 1.e6
            for _ in range(warmups - 1):
                invoke_mapping()
            torch.cuda.synchronize(selected)
            torch.cuda.reset_peak_memory_stats(selected)
            timings = []
            device_result = None
            for _ in range(repeats):
                torch.cuda.synchronize(selected)
                start = time.perf_counter_ns()
                device_result = invoke_mapping()
                torch.cuda.synchronize(selected)
                timings.append((time.perf_counter_ns() - start) / 1.e6)
            device_host = {}
            for key, value in device_result.items():
                host = value.detach().cpu().tolist()
                if key == "traversal_order":
                    host = [node for node in host if node >= 0]
                device_host[key] = host
            mismatches = [key for key in OUTPUT_KEYS if device_host[key] != cpu[key]]
            if mismatches:
                raise RuntimeError(
                    f"exact HIP parity mismatch in {case['id']} {path_name}: {mismatches}")
            output_comparisons = [{"field": key, "exact_equal": True,
                "cpu_sha256": _field_hash(cpu[key]),
                "hip_sha256": _field_hash(device_host[key])} for key in OUTPUT_KEYS]
            support_digest = _support_digest(graph, device_host, case["id"])
            if support_digest != case["support_digest_sha256"]:
                raise RuntimeError(
                    f"Plan A support digest mismatch in {case['id']} {path_name}")
            median_hip = sorted(timings)[len(timings) // 2]
            hip_rows.append({"path": path_name, "exact_output_match": True,
                "compared_output_fields": output_comparisons,
                "support_digest_match": True,
                "route_output_sha256": _route_hash(device_host),
                "route_output_field_sha256": _output_field_hashes(device_host),
                "support_digest_sha256": support_digest,
                "warmups": warmups, "repeats": repeats,
                "first_use_elapsed_ms": first_use_ms,
                "hip_elapsed_ms": timings,
                "cpu_first_call_ms": cpu_first_ms,
                "cpu_elapsed_ms": cpu_repeat_ms,
                "median_route_speedup": median_cpu / median_hip,
                "speedup_scope": "CPU and HIP route path boundary recorded here only; host-to-device input transfers and extension compilation excluded",
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(selected)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(selected))})
            del invoke_mapping, invoke, device_result
        output.append({"id": case["id"], "status": "candidate_pass",
            "route_kind": kind, "exact_output_match": True,
            "support_digest_match": True, "graph_sha256": case["graph_sha256"],
            "tensor_sha256": case["tensor_sha256"],
            "expected_route_output_sha256": case["route_output_sha256"],
            "expected_route_output_field_sha256": case["route_output_field_sha256"],
            "expected_support_digest_sha256": case["support_digest_sha256"],
            "node_count": node_count, "edge_slots_per_node": edge_slots,
            "active_edge_count": active_edges, "warmups": warmups,
            "repeats": repeats,
            "timing_boundary": "Per hip_paths entry; geometry CPU reference includes edge-weight precomputation while graph_geometry HIP computes geometry in the kernel; host-to-device input transfers and extension compilation are excluded",
            "hip_paths": hip_rows})
    source_id = source_hashes(root)
    git_commit, git_dirty = _git_source_state(root)
    rows_by_id = {row.get("id"): row for row in output}
    all_ids_present = set(rows_by_id) == set(cases_by_id) and len(output) == len(cases_by_id)
    passed = all_ids_present and all(
        (rows_by_id[case_id].get("status") == "candidate_rejected_input" and
         {path_row.get("path") for path_row in rows_by_id[case_id].get("hip_paths", [])} ==
             ({"preweighted", "graph_geometry"} if "xy" in case.get("graph", {}) or
              "recipe" in case else {"preweighted"}) and
         all(path_row.get("status") == "candidate_rejected_input"
             for path_row in rows_by_id[case_id].get("hip_paths", [])))
        if case.get("device_rejects") else
        (rows_by_id[case_id].get("status") == "candidate_pass" and
         rows_by_id[case_id].get("exact_output_match") is True and
         rows_by_id[case_id].get("support_digest_match") is True and
         {path_row.get("path") for path_row in rows_by_id[case_id].get("hip_paths", [])} ==
             ({"preweighted", "graph_geometry"} if "xy" in case.get("graph", {}) or
              "recipe" in case else {"preweighted"}) and
         all(path_row.get("exact_output_match") is True and
             path_row.get("support_digest_match") is True and
             path_row.get("route_output_sha256") == case.get("route_output_sha256") and
             path_row.get("support_digest_sha256") == case.get("support_digest_sha256") and
             path_row.get("route_output_field_sha256") == case.get("route_output_field_sha256")
             for path_row in rows_by_id[case_id].get("hip_paths", [])))
        for case_id, case in cases_by_id.items())
    architecture = str(getattr(properties, "gcnArchName", ""))
    if not architecture:
        raise RuntimeError("selected WX9100 architecture identity is unavailable")
    compiler = _compiler_metadata(torch)
    hip_flags = ["-O3", "-ffp-contract=off"]
    if compiler["device_bitcode_path"]:
        hip_flags.append("--rocm-device-lib-path=" + compiler["device_bitcode_path"])
    import os
    build_config = {"extension_name": "autodrive_meco_plan_a_heap_hip_prototype",
        "sources": ["frc_defense/csrc/meco_plan_a_shortest_path_hip.cpp",
                    "frc_defense/csrc/meco_plan_a_shortest_path_hip_kernel.cu"],
        "host_flags": ["-O3", "-ffp-contract=off"], "hip_flags": hip_flags,
        "rocm_home": compiler["rocm_home"],
        "effective_rocm_home_env": os.environ.get("ROCM_HOME"),
        "effective_hip_clang_path_env": os.environ.get("HIP_CLANG_PATH"),
        "build_loader_sha256": source_id["frc_defense/meco_plan_a_shortest_path_hip.py"]}
    binary_path = getattr(ext, "__file__", None)
    binary_sha256 = (hashlib.sha256(Path(binary_path).read_bytes()).hexdigest()
                     if binary_path and Path(binary_path).is_file() else None)
    compile_identity = {"source_sha256": source_id, "build_config": build_config,
        "compiler": compiler, "hip_architecture": architecture,
        "binary_sha256": binary_sha256}
    compile_identity_sha256 = sha256_bytes(canonical_bytes(compile_identity))
    return {"schema_version": SCHEMA_VERSION,
        "evidence_class": "hip_parity_profile_candidate",
        "production_gate_open": False, "training_configuration_modified": False,
        "passed": passed,
        "git_commit": git_commit, "git_dirty": git_dirty,
        "source_sha256": source_id,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256((root / "scripts/verify_plan_a_hip_parity.py").read_bytes()).hexdigest(),
        "training_configuration_modified": False,
        "fixture_manifest_sha256": hashlib.sha256(
            (root / FIXTURE_RELATIVE).read_bytes()).hexdigest(),
        "fixture_cases_expected": sorted(cases_by_id),
        "fixture_cases_reported": sorted(row["id"] for row in output),
        "torch_version": str(torch.__version__),
        "hip_runtime_version": str(torch.version.hip),
        "compiler": compiler, "hip_architecture": architecture,
        "build_config": build_config,
        "extension_binary_path": binary_path, "extension_binary_sha256": binary_sha256,
        "compile_identity_sha256": compile_identity_sha256,
        "compile_elapsed_ms": compile_elapsed_ms,
        "extension_was_already_loaded": extension_was_loaded,
        "extension_compile_boundary": "on-demand extension loader call; excludes device-specific fixture timings",
        "device": {"index": int(selected.index), "model": model, "uuid": uuid},
        "lease_authorization": {"logical_lease_id": lease.lease_id,
            "logical_lease_owner": lease.owner,
            "logical_lease_devices": list(lease.devices),
            "physical_attestation": attestation},
        "profile_scope": "single exact-heap route call only; no policy, simulator, training, deployment, or end-to-end claims",
        "cases": output}
