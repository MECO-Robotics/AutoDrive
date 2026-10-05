#!/usr/bin/env python3
"""Full TensorDefenseEnv A/B/A benchmark for the current fused simulator set."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run_case(label: str, enabled: bool, benchmark_script: Path, args,
             include_strategic_observation: bool = False,
             include_fused_swerve_pose: bool = False,
             include_full_strategic_observation: bool = False) -> dict:
    env = os.environ.copy()
    bit = "1" if enabled else "0"
    env.update({
        "AUTODRIVE_FUSED_PERCEPTION_HIP": bit,
        "AUTODRIVE_FUSED_GAMEPIECES_HIP": bit,
        "AUTODRIVE_FUSED_COLLISION_HIP": bit,
        # Measure the strict isolated route-reference kernel only in the
        # candidate, alongside production fused simulator paths.
        "AUTODRIVE_FUSED_PATH_REFERENCE_HIP": "1" if enabled else "0",
        "AUTODRIVE_FUSED_CONTACT_PIPELINE_HIP": bit,
        "AUTODRIVE_FUSED_ADSTAR_OCCUPANCY_HIP": bit,
        # Keep pre-existing aggregate measurements on the Torch observation
        # path unless the repeated benchmark explicitly includes this kernel.
        "AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_HIP": (
            bit if include_strategic_observation else "0"),
        "AUTODRIVE_FUSED_SWERVE_POSE_HIP": (
            bit if include_fused_swerve_pose else "0"),
        "AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_FULL_HIP": (
            bit if include_full_strategic_observation else "0"),
    })
    command = [sys.executable, str(benchmark_script), "--envs", str(args.envs),
               "--decisions", str(args.decisions), "--physics-ticks", str(args.physics_ticks),
               "--device", args.device, "--seed", str(args.seed), "--opponent", "guard",
               "--reuse-strategic-own-candidates"]
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process_elapsed = time.perf_counter() - started
    result = json.loads(completed.stdout)["results"][0]
    result.update({
        "case": label,
        "fused_perception_and_gamepieces_collision": enabled,
        "adstar_path_reference_hip": enabled,
        "fused_contact_pipeline_hip": enabled,
        "fused_adstar_occupancy_hip": enabled,
        "fused_strategic_observation_hip": bool(enabled and include_strategic_observation),
        "fused_swerve_pose_hip": bool(enabled and include_fused_swerve_pose),
        "fused_full_strategic_observation_hip": bool(
            enabled and include_full_strategic_observation),
        "process_wall_seconds": process_elapsed,
    })
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-3000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--decisions", type=int, default=4)
    parser.add_argument("--physics-ticks", type=int, default=12)
    parser.add_argument("--include-strategic-observation", action="store_true")
    parser.add_argument("--include-fused-swerve-pose", action="store_true")
    parser.add_argument("--include-full-strategic-observation", action="store_true",
                        help="toggle full 137-feature HIP assembler with the other kernels")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9017)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    benchmark_script = Path(__file__).with_name("benchmark_tensor_scaling.py").resolve()
    cases = [run_case("torch_a", False, benchmark_script, args,
                      include_strategic_observation=args.include_strategic_observation,
                      include_fused_swerve_pose=args.include_fused_swerve_pose,
                      include_full_strategic_observation=args.include_full_strategic_observation),
             run_case("fused_b", True, benchmark_script, args,
                      include_strategic_observation=args.include_strategic_observation,
                      include_fused_swerve_pose=args.include_fused_swerve_pose,
                      include_full_strategic_observation=args.include_full_strategic_observation),
             run_case("torch_a_repeat", False, benchmark_script, args,
                      include_strategic_observation=args.include_strategic_observation,
                      include_fused_swerve_pose=args.include_fused_swerve_pose,
                      include_full_strategic_observation=args.include_full_strategic_observation)]
    mean_a = (cases[0]["elapsed_seconds"] + cases[2]["elapsed_seconds"]) * .5
    report = {
        "envs": args.envs,
        "decisions": args.decisions,
        "physics_ticks_per_decision": args.physics_ticks,
        "device": args.device,
        "seed": args.seed,
        "enabled_candidate_paths": ["perception", "gamepieces", "full_contact_pipeline",
                                    "adstar_occupancy"],
        "candidate_only_paths": ["adstar_path_reference", "full_contact_pipeline",
                                  "adstar_occupancy"],
        "cases": cases,
        "baseline_repeat_ratio": cases[2]["elapsed_seconds"] / cases[0]["elapsed_seconds"],
        "speedup_vs_mean_torch": mean_a / cases[1]["elapsed_seconds"],
        "include_strategic_observation": args.include_strategic_observation,
        "include_fused_swerve_pose": args.include_fused_swerve_pose,
        "include_full_strategic_observation": args.include_full_strategic_observation,
        "all_final_state_hashes_equal": len({case["final_state_sha256"] for case in cases}) == 1,
    }
    if args.include_strategic_observation:
        report["enabled_candidate_paths"].append("strategic_observation_feature_pack")
    if args.include_fused_swerve_pose:
        report["enabled_candidate_paths"].append("fused_swerve_pose")
    if args.include_full_strategic_observation:
        report["enabled_candidate_paths"].append("full_strategic_observation_assembler")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_final_state_hashes_equal"]:
        raise SystemExit("A/B/A final state hashes differ")


if __name__ == "__main__":
    main()
