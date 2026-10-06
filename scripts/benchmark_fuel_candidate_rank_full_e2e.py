#!/usr/bin/env python3
"""Parity-first A/B/A test of candidate rank in full 2048x10x12 episodes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frc_defense import tensor_fuel_candidate_rank as candidate_rank


def load_scaling_benchmark():
    path = ROOT / "scripts" / "benchmark_tensor_scaling.py"
    spec = importlib.util.spec_from_file_location("fuel_candidate_scaling_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--decisions", type=int, default=10)
    parser.add_argument("--physics-ticks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=90217)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.envs != 2048 or args.decisions != 10 or args.physics_ticks != 12:
        raise ValueError("this validation is specified for exactly 2048 x 10 x 12")

    scaling = load_scaling_benchmark()
    def run(label, enabled):
        candidate_rank.FUSED_FUEL_CANDIDATE_RANK_HIP_ENABLED = enabled
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        result = scaling.benchmark(args.envs, torch.device(args.device), args.seed,
            args.decisions, args.physics_ticks, "guard", return_info=False,
            skip_metric_objective=True, profile_components=False,
            squared_fuel_candidate_distance=False, reuse_own_candidates=True)
        result["case"] = label
        result["fused_candidate_rank_hip_enabled"] = enabled
        return result

    # First pass is only a gate: persist parity evidence and abort before the
    # reported timing sequence if any state/reward trace differs.
    parity = [run("torch_parity", False), run("proto_parity", True)]
    parity_ok = (parity[0]["final_state_sha256"] == parity[1]["final_state_sha256"] and
                 parity[0]["reward_trace_sha256"] == parity[1]["reward_trace_sha256"] and
                 parity[0]["reward_total"] == parity[1]["reward_total"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parity_path = args.output.with_name(args.output.stem + "-parity.json")
    parity_record = {"device": args.device, "envs": args.envs,
        "decisions": args.decisions, "physics_ticks_per_decision": args.physics_ticks,
        "seed": args.seed, "parity_pass": parity_ok,
        "cases": [{k: v for k, v in item.items() if k not in (
            "component_host_seconds_nested", "component_host_seconds_inclusive",
            "component_calls")} for item in parity]}
    parity_path.write_text(json.dumps(parity_record, indent=2, sort_keys=True) + "\n")
    if not parity_ok:
        print(json.dumps(parity_record, indent=2, sort_keys=True))
        raise SystemExit("parity gate failed; timing sequence aborted")

    timed = [run("torch_a1", False), run("proto_b", True), run("torch_a2", False)]
    a = [timed[0]["elapsed_seconds"], timed[2]["elapsed_seconds"]]
    b = [timed[1]["elapsed_seconds"]]
    report = {"device": args.device, "envs": args.envs, "decisions": args.decisions,
        "physics_ticks_per_decision": args.physics_ticks, "seed": args.seed,
        "candidate_only_change": "production flag toggles fused Euclidean distance + free mask; ATen topk/isfinite stays in use",
        "parity_gate_artifact": str(parity_path), "parity_pass": True,
        "sequence": [item["case"] for item in timed], "cases": timed,
        "torch_median_seconds": statistics.median(a), "prototype_median_seconds": statistics.median(b),
        "speedup": statistics.median(a) / statistics.median(b),
        "torch_repeat_relative_spread": abs(a[1] - a[0]) / statistics.median(a),
        "gain_outside_torch_repeat_spread": statistics.median(a) - statistics.median(b) > abs(a[1] - a[0]),
        "all_timed_final_states_equal": len({item["final_state_sha256"] for item in timed}) == 1,
        "all_timed_reward_traces_equal": len({item["reward_trace_sha256"] for item in timed}) == 1}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_timed_final_states_equal"] or not report["all_timed_reward_traces_equal"]:
        raise SystemExit("timed A/B/A state or reward mismatch")


if __name__ == "__main__":
    main()
