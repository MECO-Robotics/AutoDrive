#!/usr/bin/env python3
"""Warm simulator A/B/A for Torch vs isolated strict-HIP path reference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

os.environ["AUTODRIVE_FUSED_PATH_REFERENCE_HIP"] = "1"
from frc_defense import tensor_adstar
from benchmark_tensor_scaling import benchmark


def run_case(label, enabled, args):
    tensor_adstar._HIP_PATH_REFERENCE_ENABLED = enabled
    result = benchmark(
        args.envs, torch.device(args.device), args.seed,
        args.decisions, args.physics_ticks, "guard",
        return_info=False, skip_metric_objective=True,
        reuse_opponent_candidates=True)
    return {"label": label, "path_reference": "strict_hip" if enabled else "torch", **result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--decisions", type=int, default=16)
    parser.add_argument("--physics-ticks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=92001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("A HIP-enabled PyTorch build and visible HIP GPU are required")
    extension = tensor_adstar._hip_path_reference_extension()
    if extension is None:
        raise SystemExit("isolated strict HIP path-reference extension failed to load")

    # Warm both implementations in advance so neither timed A entry is a cold start.
    warm_a = run_case("warm_torch", False, args)
    warm_b = run_case("warm_strict_hip", True, args)
    a1 = run_case("A1_torch", False, args)
    b = run_case("B_strict_hip", True, args)
    a2 = run_case("A2_torch", False, args)
    a_tps = statistics.mean((a1["strategic_transitions_per_second"],
                             a2["strategic_transitions_per_second"]))
    report = {
        "fixture": {"envs": args.envs, "decisions": args.decisions,
                    "physics_ticks_per_decision": args.physics_ticks, "seed": args.seed,
                    "opponent": "guard", "device": args.device},
        "compile": {"loader": "_hip_path_reference_extension",
                    "extra_cuda_cflags": ["-O3", "-ffp-contract=off"],
                    "extra_cflags": ["-O3"]},
        "warmup": [warm_a, warm_b],
        "cases": [a1, b, a2],
        "comparison": {
            "warm_a_mean_tps": a_tps,
            "b_tps": b["strategic_transitions_per_second"],
            "speedup": b["strategic_transitions_per_second"] / a_tps - 1.,
            "a1_a2_hash_equal": a1["final_state_sha256"] == a2["final_state_sha256"],
            "all_hashes_equal": len({a1["final_state_sha256"], b["final_state_sha256"],
                                     a2["final_state_sha256"]}) == 1,
            "all_rewards_equal": len({a1["reward_total"], b["reward_total"],
                                       a2["reward_total"]}) == 1,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), **report["fixture"],
                      **report["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
