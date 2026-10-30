#!/usr/bin/env python3
"""Compare replaying one-substep vs whole-tick A1 HIP graphs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "profiling"))

import combined_sim_accounting_ab_a as bench
from frc_defense.fuel_physics_runtime import advance_with_fuel


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(env, command, active, *, whole_tick: bool, device: str) -> float:
    os.environ["AUTODRIVE_EXPERIMENTAL_FUEL_WHOLE_TICK_GRAPH"] = (
        "1" if whole_tick else "0")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    advance_with_fuel(env, command, active)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=bench.DEFAULT_SEED)
    parser.add_argument("--category", default=bench.DEFAULT_CATEGORY)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT /
        "profiling/next-speedup-capture-20261030/simulator/whole-tick-graph-ab-a.json")
    args = parser.parse_args()
    if args.pairs < 3:
        parser.error("--pairs must be at least 3")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("whole-tick graph benchmark requires HIP")
    torch.cuda.set_device(args.device)
    os.environ["AUTODRIVE_EXPERIMENTAL_FUEL_WHOLE_TICK_GRAPH"] = "0"
    envs = [bench._make_env(device=args.device, seed=args.seed,
        category=args.category, fused=False) for _ in range(3)]
    commands = [bench._commands(env, args.device) for env in envs]
    active = [torch.ones((env.n,), device=args.device, dtype=torch.bool) for env in envs]

    for index, whole_tick in enumerate((False, True, False)):
        run(envs[index], commands[index], active[index], whole_tick=whole_tick,
            device=args.device)
    torch.cuda.synchronize(args.device)
    warm = [bench._tensor_hashes(env) for env in envs]
    warm_mismatch = {key: bench._state_differences(warm[0], warm[index])[:20]
                     for index, key in ((1, "whole_tick"), (2, "baseline_repeat"))
                     if bench._state_differences(warm[0], warm[index])}
    if warm_mismatch:
        raise AssertionError(json.dumps({"stage": "warmup", "mismatches": warm_mismatch}, indent=2))

    samples = {"A0_baseline_ms": [], "B_whole_tick_graph_ms": [],
               "A1_baseline_repeat_ms": []}
    hash_checks = 0
    for _ in range(args.pairs):
        elapsed = []
        for index, whole_tick in enumerate((False, True, False)):
            elapsed.append(run(envs[index], commands[index], active[index],
                whole_tick=whole_tick, device=args.device))
        for value, name in zip(elapsed, samples):
            samples[name].append(value)
        hashes = [bench._tensor_hashes(env) for env in envs]
        for index in (1, 2):
            differences = bench._state_differences(hashes[0], hashes[index])
            if differences:
                raise AssertionError(json.dumps({"pair": len(samples["A0_baseline_ms"]),
                    "variant": "whole_tick" if index == 1 else "baseline_repeat",
                    "differences": differences[:20]}, indent=2))
        hash_checks += 2

    medians = {key: statistics.median(values) for key, values in samples.items()}
    report = {"schema": "autodrive-whole-tick-graph-ab-a-v1",
        "evidence_type": "seeded production A1 physics tick benchmark",
        "git_head": subprocess.check_output(["git", "-C", str(ROOT),
            "rev-parse", "HEAD"], text=True).strip(),
        "device": args.device, "device_name": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__, "hip_version": torch.version.hip,
        "scenario": "full_match_offense", "category": args.category,
        "seed": args.seed, "pairs": args.pairs,
        "accounting_fusion_gates": "both off",
        "whole_tick_graph_gate": "B only; off for A0 and A1",
        "samples_ms": samples, "medians_ms": medians,
        "speedup_vs_A0": medians["A0_baseline_ms"] / medians["B_whole_tick_graph_ms"],
        "baseline_repeat_ratio_A0_over_A1": medians["A0_baseline_ms"] /
            medians["A1_baseline_repeat_ms"],
        "exact_persistent_state_hashes": True,
        "hash_comparisons": hash_checks,
        "source_hashes": {path: sha256(ROOT / path) for path in (
            "frc_defense/fuel_physics_runtime.py", "frc_defense/tensor_physics.py",
            "frc_defense/csrc/tensor_swerve_pose_kernel.cu",
            "frc_defense/csrc/tensor_collision_multi_hip_kernel.hip",
            "profiling/benchmark_whole_tick_graph_ab_a.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "device_name", "pairs", "medians_ms", "speedup_vs_A0",
        "baseline_repeat_ratio_A0_over_A1", "exact_persistent_state_hashes",
        "hash_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
