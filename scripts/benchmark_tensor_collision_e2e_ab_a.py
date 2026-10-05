#!/usr/bin/env python3
"""Full TensorDefenseEnv A/B/A benchmark for fused wall contacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run_case(label: str, enabled: bool, benchmark_script: Path, args) -> dict:
    env = os.environ.copy()
    env["AUTODRIVE_FUSED_COLLISION_HIP"] = "1" if enabled else "0"
    command = [sys.executable, str(benchmark_script), "--envs", str(args.envs),
               "--decisions", str(args.decisions), "--physics-ticks", str(args.physics_ticks),
               "--device", args.device, "--seed", str(args.seed), "--opponent", "guard",
               "--reuse-strategic-own-candidates"]
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - started
    payload = json.loads(completed.stdout)
    result = payload["results"][0]
    result["case"] = label
    result["collision_hip_enabled"] = enabled
    result["process_wall_seconds"] = elapsed
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-3000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--decisions", type=int, default=4)
    parser.add_argument("--physics-ticks", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9017)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    benchmark_script = Path(__file__).with_name("benchmark_tensor_scaling.py").resolve()
    cases = [run_case("torch_a", False, benchmark_script, args),
             run_case("hip_b", True, benchmark_script, args),
             run_case("torch_a_repeat", False, benchmark_script, args)]
    a = (cases[0]["elapsed_seconds"] + cases[2]["elapsed_seconds"]) * .5
    report = {
        "envs": args.envs,
        "decisions": args.decisions,
        "physics_ticks_per_decision": args.physics_ticks,
        "device": args.device,
        "seed": args.seed,
        "cases": cases,
        "baseline_repeat_ratio": cases[2]["elapsed_seconds"] / cases[0]["elapsed_seconds"],
        "speedup_vs_mean_torch": a / cases[1]["elapsed_seconds"],
        "all_final_state_hashes_equal": len({case["final_state_sha256"] for case in cases}) == 1,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_final_state_hashes_equal"]:
        raise SystemExit("A/B/A final state hashes differ")


if __name__ == "__main__":
    main()
