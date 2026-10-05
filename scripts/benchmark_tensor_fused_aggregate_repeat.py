#!/usr/bin/env python3
"""Repeated all-Torch/all-fused aggregate A/B/A/B/A parity/timing check."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--decisions", type=int, default=4)
    parser.add_argument("--physics-ticks", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9017)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-strategic-observation", action="store_true",
                        help="toggle the production fused strategic-observation path with the other kernels")
    parser.add_argument("--include-fused-swerve-pose", action="store_true",
                        help="toggle the exact fused swerve + pose path with the other kernels")
    parser.add_argument("--include-full-strategic-observation", action="store_true",
                        help="toggle the opt-in full 137-feature HIP assembler with the other kernels")
    args = parser.parse_args()

    base = Path(__file__).with_name("benchmark_tensor_fused_aggregate_ab_a.py")
    spec = importlib.util.spec_from_file_location("aggregate_benchmark", base)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    benchmark = Path(__file__).with_name("benchmark_tensor_scaling.py").resolve()
    cases = []
    for i, enabled in enumerate((False, True, False, True, False)):
        label = f"{'fused' if enabled else 'torch'}_{i + 1}"
        cases.append(module.run_case(label, enabled, benchmark, args,
            include_strategic_observation=args.include_strategic_observation,
            include_fused_swerve_pose=args.include_fused_swerve_pose,
            include_full_strategic_observation=args.include_full_strategic_observation))

    torch_times = [case["elapsed_seconds"] for case in cases if not case["fused_contact_pipeline_hip"]]
    fused_times = [case["elapsed_seconds"] for case in cases if case["fused_contact_pipeline_hip"]]
    hashes = {case["final_state_sha256"] for case in cases}
    rewards = {case["reward_total"] for case in cases}
    reward_traces = {case["reward_trace_sha256"] for case in cases}
    report = {
        "envs": args.envs,
        "decisions": args.decisions,
        "physics_ticks_per_decision": args.physics_ticks,
        "device": args.device,
        "seed": args.seed,
        "include_strategic_observation": args.include_strategic_observation,
        "include_fused_swerve_pose": args.include_fused_swerve_pose,
        "include_full_strategic_observation": args.include_full_strategic_observation,
        "sequence": [case["case"] for case in cases],
        "cases": cases,
        "torch_median_seconds": statistics.median(torch_times),
        "fused_median_seconds": statistics.median(fused_times),
        "median_speedup": statistics.median(torch_times) / statistics.median(fused_times),
        "torch_samples_seconds": torch_times,
        "fused_samples_seconds": fused_times,
        "torch_range_seconds": max(torch_times) - min(torch_times),
        "fused_range_seconds": max(fused_times) - min(fused_times),
        "torch_relative_range": ((max(torch_times) - min(torch_times)) /
                                 statistics.median(torch_times)),
        "fused_relative_range": ((max(fused_times) - min(fused_times)) /
                                 statistics.median(fused_times)),
        "all_hashes_equal": len(hashes) == 1,
        "all_rewards_equal": len(rewards) == 1,
        "all_reward_traces_equal": len(reward_traces) == 1,
        "candidate_paths": ["perception", "gamepieces", "full_contact_pipeline",
                            "strict_adstar_path_reference", "adstar_occupancy"],
    }
    if args.include_strategic_observation:
        report["candidate_paths"].append("strategic_observation_features_and_pack")
    if args.include_fused_swerve_pose:
        report["candidate_paths"].append("swerve_dynamics_and_pose_integration")
    if args.include_full_strategic_observation:
        report["candidate_paths"].append("full_strategic_observation_137_feature_assembler")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if (not report["all_hashes_equal"] or not report["all_rewards_equal"] or
            not report["all_reward_traces_equal"]):
        raise SystemExit("A/B/A/B/A state or reward parity failed")


if __name__ == "__main__":
    main()
