#!/usr/bin/env python3
"""Paired A/B/A benchmark for the exact full-scan and active-list routes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT /
        "profiling/next-speedup-capture-20261030/route/a1_plan_a_dijkstra_inputs.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--candidate-block-size", type=int, default=0,
        help="0 uses the 1024-thread device maximum; otherwise use a power of two")
    parser.add_argument("--output", type=Path, default=ROOT /
        "profiling/next-speedup-capture-20261030/route/a1_route_frontier_ab_a.json")
    args = parser.parse_args()
    if args.pairs < 6:
        parser.error("--pairs must be at least 6")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("route A/B/A requires HIP")
    torch.cuda.set_device(args.device)
    saved = torch.load(args.inputs, map_location="cpu", weights_only=False)
    keys = ("xy", "heading", "edge_dst", "edge_allowed", "node_valid", "root",
            "root_start_cost", "root_connector_m", "heading_values", "turns")
    inputs = [saved[key].to(args.device).contiguous() for key in keys]
    from frc_defense.meco_plan_a_shortest_path_hip import extension
    ext = extension()
    expected = saved["reference_outputs"]
    output_names = ("route_cost", "physical_distance", "parent_ids",
                    "parent_edge_m", "traversal_order", "status")

    def invoke(candidate: bool):
        if not candidate:
            return ext.dijkstra_graph(*inputs)
        if args.candidate_block_size:
            return ext.dijkstra_graph_active_block(
                *inputs, args.candidate_block_size)
        return ext.dijkstra_graph_active(*inputs)

    def assert_exact(outputs, variant: str, pair: int, slot: str):
        for name, actual in zip(output_names, outputs[:6]):
            if not torch.equal(actual.cpu(), expected[name]):
                raise AssertionError(f"{variant} pair {pair} {slot}: {name} differs from capture")

    assert_exact(invoke(False), "A", 0, "warmup")
    assert_exact(invoke(True), "B", 0, "warmup")
    torch.cuda.synchronize(args.device)
    samples = {"A0_baseline_ms": [], "B_active_candidate_ms": [],
               "A1_baseline_repeat_ms": []}
    last_trace = []
    for pair in range(1, args.pairs + 1):
        row = {}
        for slot, candidate, key in (
                ("A0", False, "A0_baseline_ms"),
                ("B", True, "B_active_candidate_ms"),
                ("A1", False, "A1_baseline_repeat_ms")):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = invoke(candidate)
            end.record()
            end.synchronize()
            elapsed = float(start.elapsed_time(end))
            assert_exact(outputs, "B" if candidate else "A", pair, slot)
            samples[key].append(elapsed)
            row[slot] = elapsed
            if candidate:
                trace = outputs[6].cpu()
                last_trace = trace[trace[:, 0] >= 0].tolist()
        row["exact_output_parity"] = True
    medians = {key: statistics.median(values) for key, values in samples.items()}
    report = {
        "schema": "autodrive-plan-a-route-frontier-ab-a-v1",
        "evidence_type": "captured production A1 route microbenchmark; not full-match qualification",
        "git_head": subprocess.check_output(["git", "-C", str(ROOT),
            "rev-parse", "HEAD"], text=True).strip(),
        "device": args.device, "device_name": torch.cuda.get_device_name(args.device),
        "hip_version": torch.version.hip, "torch_version": torch.__version__,
        "scenario": "full_match_offense", "category": "active/open/level_5_mixed",
        "seed": 2970053994459693056, "input_file": str(args.inputs),
        "input_file_sha256": sha256(args.inputs),
        "input_nodes": int(inputs[0].shape[0]),
        "input_edges": int(inputs[3].sum().item()), "pairs": args.pairs,
        "candidate_block_size": args.candidate_block_size or 1024,
        "samples_ms": samples, "medians_ms": medians,
        "active_speedup_vs_A0": medians["A0_baseline_ms"] /
            medians["B_active_candidate_ms"],
        "baseline_repeat_ratio_A0_over_A1": medians["A0_baseline_ms"] /
            medians["A1_baseline_repeat_ms"],
        "exact_output_parity_every_call": True,
        "active_frontier_candidate_counts": last_trace,
        "outputs_compared": list(output_names),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "device_name", "input_nodes", "input_edges", "pairs", "medians_ms",
        "active_speedup_vs_A0", "baseline_repeat_ratio_A0_over_A1",
        "exact_output_parity_every_call", "outputs_compared")}, indent=2))


if __name__ == "__main__":
    main()
