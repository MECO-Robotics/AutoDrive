#!/usr/bin/env python3
"""Profile and replay-check one captured Plan A HIP graph route."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT /
        "profiling/next-speedup-capture-20261030/route/a1_plan_a_dijkstra_inputs.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--active-frontier", action="store_true",
                        help="profile the opt-in active-candidate kernel")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("route profile requires HIP")
    torch.cuda.set_device(args.device)
    saved = torch.load(args.inputs, map_location="cpu", weights_only=False)
    keys = ("xy", "heading", "edge_dst", "edge_allowed", "node_valid", "root",
            "root_start_cost", "root_connector_m", "heading_values", "turns")
    device_inputs = {key: saved[key].to(args.device).contiguous() for key in keys}
    from frc_defense.meco_plan_a_shortest_path_hip import extension
    ext = extension()

    route_entry = (ext.dijkstra_graph_active if args.active_frontier
                   else ext.dijkstra_graph)

    def solve():
        return route_entry(*(device_inputs[key] for key in keys))

    expected = saved["reference_outputs"]
    def check(outputs, label):
        names = ("route_cost", "physical_distance", "parent_ids", "parent_edge_m",
                 "traversal_order", "status")
        mismatch = [name for name, value in zip(names, outputs)
                    if not torch.equal(value.cpu(), expected[name])]
        if mismatch:
            raise AssertionError(f"captured-input {label} parity mismatch: {mismatch}")

    check(solve(), "warmup")
    torch.cuda.synchronize(args.device)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, with_stack=True, profile_memory=False) as prof:
        outputs = solve()
        torch.cuda.synchronize(args.device)
    check(outputs, "profiled run")

    out_dir = args.inputs.parent
    variant = "active" if args.active_frontier else "full_scan"
    trace = out_dir / f"a1_route_solver_trace_{variant}.json"
    prof.export_chrome_trace(str(trace))
    rows = []
    for event in prof.key_averages(group_by_input_shape=True):
        rows.append({"name": event.key, "count": int(event.count),
            "self_cpu_us": float(getattr(event, "self_cpu_time_total", 0.0)),
            "self_device_us": float(getattr(event, "self_device_time_total",
                                   getattr(event, "self_cuda_time_total", 0.0))),
            "input_shapes": getattr(event, "input_shapes", None)})
    rows.sort(key=lambda row: row["self_device_us"], reverse=True)
    report = {"schema": "autodrive-plan-a-route-profiler-v1",
        "evidence_type": "single captured-input attribution; profiler timings are not A/B timing",
        "git_head": subprocess.check_output(["git", "-C", str(ROOT),
            "rev-parse", "HEAD"], text=True).strip(),
        "kernel_variant": "active-candidate" if args.active_frontier else "full-scan",
        "input_path": str(args.inputs), "input_sha256": sha256(args.inputs),
        "device": args.device, "device_name": torch.cuda.get_device_name(args.device),
        "hip_version": torch.version.hip, "torch_version": torch.__version__,
        "input_nodes": int(device_inputs["xy"].shape[0]),
        "input_edges": int(device_inputs["edge_allowed"].sum().item()),
        "route_status": int(outputs[5].item()),
        "exact_output_replay_match": True,
        "trace_path": str(trace), "trace_sha256": sha256(trace),
        "source_hashes": {path: sha256(ROOT / path) for path in (
            "frc_defense/meco_plan_a_shortest_path.py",
            "frc_defense/csrc/meco_plan_a_shortest_path_hip.cpp",
            "frc_defense/csrc/meco_plan_a_shortest_path_hip_kernel.cu",
            "profiling/profile_plan_a_graph_inputs.py")},
        "operators": rows}
    if args.active_frontier:
        frontier_trace = outputs[6].cpu()
        report["frontier_candidate_counts"] = frontier_trace[
            frontier_trace[:, 0] >= 0].tolist()
        report["frontier_rounds"] = len(report["frontier_candidate_counts"])
    summary = out_dir / f"a1_route_solver_trace_{variant}_summary.json"
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "evidence_type", "input_nodes", "input_edges", "route_status",
        "exact_output_replay_match", "trace_path", "trace_sha256")}, indent=2))
    print(json.dumps(rows[:30], indent=2))


if __name__ == "__main__":
    main()
