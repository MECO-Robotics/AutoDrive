#!/usr/bin/env python3
"""A/B/A/B/A E2E benchmark toggling only the optional perception commit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run_case(label: str, commit_enabled: bool, benchmark_script: Path,
             args) -> dict:
    env = os.environ.copy()
    env.update({
        "AUTODRIVE_FUSED_PERCEPTION_HIP": "1",
        "AUTODRIVE_FUSED_PERCEPTION_COMMIT_HIP": "1" if commit_enabled else "0",
        "AUTODRIVE_FUSED_GAMEPIECES_HIP": "1",
        "AUTODRIVE_FUSED_COLLISION_HIP": "1",
        "AUTODRIVE_FUSED_PATH_REFERENCE_HIP": "1",
        "AUTODRIVE_FUSED_CONTACT_PIPELINE_HIP": "1",
        "AUTODRIVE_FUSED_ADSTAR_OCCUPANCY_HIP": "1",
        "AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_HIP": "1",
        "AUTODRIVE_FUSED_SWERVE_POSE_HIP": "1",
        "AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_FULL_HIP": "1",
    })
    command = [sys.executable, str(benchmark_script), "--envs", str(args.envs),
        "--decisions", "1", "--physics-ticks", str(args.physics_ticks),
        "--device", args.device, "--seed", str(args.seed),
        "--opponent", "guard", "--reuse-strategic-own-candidates"]
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - started
    result = json.loads(completed.stdout)["results"][0]
    result.update({"case": label, "perception_commit_enabled": commit_enabled,
                   "process_wall_seconds": elapsed})
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-2000:]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--physics-ticks", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=92391)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark_script = Path(__file__).with_name("benchmark_tensor_scaling.py").resolve()
    variants = (("baseline_a", False), ("commit_b", True),
                ("baseline_a_repeat", False), ("commit_b_repeat", True),
                ("baseline_a_final", False))
    cases = [run_case(name, enabled, benchmark_script, args)
             for name, enabled in variants]
    baseline = [case["elapsed_seconds"] for case in cases
                if not case["perception_commit_enabled"]]
    commit = [case["elapsed_seconds"] for case in cases
              if case["perception_commit_enabled"]]
    median_baseline = sorted(baseline)[len(baseline) // 2]
    median_commit = sorted(commit)[len(commit) // 2]
    report = {
        "workload": {"envs": args.envs,
            "strategic_decisions": 1, "physics_ticks": args.physics_ticks,
            "device": args.device, "seed": args.seed,
            "all_other_fused_paths_held_on": True,
            "strategic_full_assembler": "enabled/default-on"},
        "cases": cases,
        "median_baseline_seconds": median_baseline,
        "median_commit_seconds": median_commit,
        "speedup": median_baseline / median_commit,
        "baseline_repeat_ratio": max(baseline) / min(baseline),
        "commit_repeat_ratio": max(commit) / min(commit),
        "all_final_state_hashes_equal": len(
            {case["final_state_sha256"] for case in cases}) == 1,
        "all_reward_trace_hashes_equal": len(
            {case["reward_trace_sha256"] for case in cases}) == 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_final_state_hashes_equal"]:
        raise SystemExit("A/B/A/B/A final state hashes differ")
    if not report["all_reward_trace_hashes_equal"]:
        raise SystemExit("A/B/A/B/A reward trace hashes differ")


if __name__ == "__main__":
    main()
