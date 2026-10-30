#!/usr/bin/env python3
"""Capture the exact HIP Dijkstra inputs from a seeded A1 support build."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def cpu_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu().clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2970053994459693056)
    parser.add_argument("--scenario", default="full_match_offense")
    parser.add_argument("--category", default="active/open/level_5_mixed")
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "profiling/next-speedup-capture-20261030/route")
    parser.add_argument("--checkpoint-sha256", default=
        "a68ece9464d4a507c681495ec60e655b6b1d6186b1fdc49d1feddab3815ac49c",
        help="source A1 diagnostic checkpoint identity; support graph input does not depend on policy inference")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("route input capture requires HIP")
    torch.cuda.set_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from frc_defense import meco_plan_a_policy as plan_a
    from frc_defense import meco_plan_a_shortest_path as shortest_path
    from frc_defense import meco_track_training as training
    from frc_defense.curriculum_scenarios import make_curriculum_env
    from frc_defense.curriculum_training import source_identity
    from frc_defense.meco_pose_graph import build_legal_pose_graph, HEADING_BINS

    env = make_curriculum_env(args.scenario, args.category, 1, args.device,
        args.seed, fuel_physics=training._training_fuel_physics_config("A1"))
    training._reset_episode(env, args.seed)
    graph = build_legal_pose_graph(env, slot=3)
    captures = []
    original = shortest_path.plan_a_dijkstra_hip_for_graph

    def capture_then_run(graph_arg, *, device, root, root_start_cost,
                         root_connector_m):
        headings = (torch.arange(HEADING_BINS, dtype=graph_arg.heading.dtype) *
                    (2 * torch.pi / HEADING_BINS)).tolist()
        heading_values, turns = shortest_path.plan_a_cpu_turn_table(
            headings, device=torch.device(device))
        tensors = {
            "xy": cpu_copy(graph_arg.xy),
            "heading": cpu_copy(graph_arg.heading),
            "edge_dst": cpu_copy(graph_arg.edges[0]),
            "edge_allowed": cpu_copy(graph_arg.edge_mask[0]),
            "node_valid": cpu_copy(graph_arg.node_valid[0]),
            "root": cpu_copy(root),
            "root_start_cost": cpu_copy(root_start_cost),
            "root_connector_m": cpu_copy(root_connector_m),
            "heading_values": cpu_copy(heading_values),
            "turns": cpu_copy(turns),
        }
        output = original(graph_arg, device=device, root=root,
            root_start_cost=root_start_cost, root_connector_m=root_connector_m)
        tensors["reference_outputs"] = {key: cpu_copy(value) for key, value in output.items()
                                        if torch.is_tensor(value)}
        captures.append(tensors)
        return output

    shortest_path.plan_a_dijkstra_hip_for_graph = capture_then_run
    try:
        support = plan_a.build_plan_a_support(env, graph, env._last_obs,
            f"a1-route-input-capture-{args.seed}", require_hip_shortest_path=True)
    finally:
        shortest_path.plan_a_dijkstra_hip_for_graph = original
    if len(captures) != 1:
        raise RuntimeError(f"expected one Dijkstra capture, got {len(captures)}")

    input_path = args.output_dir / "a1_plan_a_dijkstra_inputs.pt"
    torch.save(captures[0], input_path)
    state_sources = source_identity()
    source_manifest = {key: value for key, value in state_sources.items()
                       if isinstance(value, (str, int, float, bool)) or value is None}
    artifact = {
        "schema": "autodrive-plan-a-dijkstra-input-capture-v1",
        "capture_mode": "production Plan A support-builder call; initial seeded A1 episode state",
        "scenario": args.scenario, "category": args.category, "seed": args.seed,
        "device": args.device, "device_name": torch.cuda.get_device_name(args.device),
        "backend": "ROCm/HIP", "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "a1_checkpoint_sha256_context": args.checkpoint_sha256,
        "checkpoint_used_for_support_geometry": False,
        "call_count": len(captures), "snapshot_id": f"a1-route-input-capture-{args.seed}",
        "graph_shape": {key: list(value.shape) for key, value in captures[0].items()
                        if torch.is_tensor(value)},
        "graph_dtype": {key: str(value.dtype) for key, value in captures[0].items()
                        if torch.is_tensor(value)},
        "source_identity": source_manifest,
        "git_head": subprocess.check_output(["git", "-C", str(ROOT),
            "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {path: sha256(ROOT / path) for path in (
            "frc_defense/meco_plan_a_policy.py",
            "frc_defense/meco_plan_a_shortest_path.py",
            "frc_defense/meco_pose_graph.py",
            "frc_defense/csrc/meco_plan_a_shortest_path_hip.cpp",
            "frc_defense/csrc/meco_plan_a_shortest_path_hip_kernel.cu",
            "profiling/capture_plan_a_graph_inputs.py")},
        "input_file": str(input_path), "input_file_sha256": sha256(input_path),
        "support_pair_count": int(support.pair_count),
        "route_status": int(captures[0]["reference_outputs"]["status"].item()),
    }
    manifest_path = args.output_dir / "a1_plan_a_dijkstra_inputs.json"
    manifest_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: artifact[key] for key in (
        "capture_mode", "scenario", "category", "seed", "device_name",
        "call_count", "graph_shape", "input_file", "input_file_sha256",
        "route_status", "support_pair_count")}, indent=2))


if __name__ == "__main__":
    main()
