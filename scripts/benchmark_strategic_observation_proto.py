#!/usr/bin/env python3
"""Small warmed A/B/A/B timing for isolated strategic-observation packing."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense.tensor_strategic_observation_proto import strategic_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=77319)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fuse-candidate-local", action="store_true",
                        help="benchmark fused local-track/candidate feature arithmetic")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("Benchmark requires the HIP runtime.")
    env = TensorDefenseEnv(num_envs=args.envs, task="counter_defense",
        device=args.device, seed=args.seed, action_mode="strategic", randomize=True)
    env._track_mask[:, 0].zero_()
    origin = env.sim.pose[:, 0, :2]
    offsets = torch.tensor(((.19, .07), (.43, .16), (.71, .31), (1.03, .52)),
                           device=env.device, dtype=env.sim.pose.dtype)
    env._track_pos[:, 0, :4] = origin[:, None, :] + offsets[None, :, :]
    env._track_vel[:, 0, :4] = torch.tensor((.11, -.07), device=env.device,
        dtype=env.sim.pose.dtype)
    env._track_mask[:, 0, :4] = True

    reference = lambda: env._strategic_observation(0)
    prototype = lambda: strategic_observation(
        env, 0, fuse_candidate_local=args.fuse_candidate_local)
    reference_out = reference()
    prototype_out = prototype()
    if not torch.equal(reference_out, prototype_out):
        raise AssertionError("reference and prototype observations differ")
    for _ in range(12):
        reference(); prototype()
    torch.cuda.synchronize(args.device)

    results = []
    for block in range(args.blocks):
        order = ([("reference", reference), ("prototype", prototype)]
                 if block % 2 == 0 else
                 [("prototype", prototype), ("reference", reference)])
        for name, fn in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repetitions):
                fn()
            end.record()
            end.synchronize()
            milliseconds = start.elapsed_time(end)
            results.append({"block": block, "case": name,
                "milliseconds": milliseconds,
                "observations_per_second": args.repetitions * args.envs /
                                           (milliseconds / 1000.)})
    ref = [x["milliseconds"] for x in results if x["case"] == "reference"]
    proto = [x["milliseconds"] for x in results if x["case"] == "prototype"]
    report = {"device": str(args.device), "envs": args.envs,
        "repetitions_per_block": args.repetitions, "blocks": args.blocks,
        "valid_candidate_rows": int(env._track_mask[:, 0].any(-1).sum().item()),
        "reference_median_ms": statistics.median(ref),
        "prototype_median_ms": statistics.median(proto),
        "speedup": statistics.median(ref) / statistics.median(proto),
        "fuse_candidate_local": args.fuse_candidate_local,
        "all_case_outputs_exact_before_timing": True, "blocks_results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
