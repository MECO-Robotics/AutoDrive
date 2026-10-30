#!/usr/bin/env python3
"""Capture a short profiler trace of the production A1 simulator tick."""
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
sys.path.insert(0, str(ROOT / "profiling"))

import combined_sim_accounting_ab_a as bench
from frc_defense.fuel_physics_runtime import advance_with_fuel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=bench.DEFAULT_SEED)
    parser.add_argument("--category", default=bench.DEFAULT_CATEGORY)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "profiling/next-speedup-capture-20261030/simulator")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("simulator trace requires the HIP runtime")
    torch.cuda.set_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Isolate the current production tick path from the two previously merged
    # accounting fusions. This run attributes work; it is not a timing result.
    for gate in ("AUTODRIVE_EXPERIMENTAL_CONTACT_ACCOUNTING_FUSION",
                 "AUTODRIVE_EXPERIMENTAL_FUSED_SIM_TICK_ACCOUNTING"):
        os.environ[gate] = "0"
    env = bench._make_env(device=args.device, seed=args.seed,
                          category=args.category, fused=False)
    command = bench._commands(env, args.device)
    active = torch.ones((env.n,), device=args.device, dtype=torch.bool)
    advance_with_fuel(env, command, active)
    torch.cuda.synchronize(args.device)

    source_files = [ROOT / path for path in (
        "frc_defense/tensor_physics.py",
        "frc_defense/tensor_swerve_pose_hip.py",
        "frc_defense/csrc/tensor_swerve_pose_kernel.cu",
        "frc_defense/csrc/tensor_collision_multi_hip_kernel.hip",
        "frc_defense/fuel_physics_runtime.py",
        "profiling/combined_sim_accounting_ab_a.py",
        __file__.replace(str(ROOT) + "/", ""),
    )]
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, with_stack=True, profile_memory=False) as prof:
        advance_with_fuel(env, command, active)
        torch.cuda.synchronize(args.device)

    trace_path = args.output_dir / "a1_sim_tick_trace.json"
    prof.export_chrome_trace(str(trace_path))
    averages = prof.key_averages(group_by_input_shape=True)
    rows = []
    for event in averages:
        cpu = float(getattr(event, "self_cpu_time_total", 0.0))
        device = float(getattr(event, "self_device_time_total",
                        getattr(event, "self_cuda_time_total", 0.0)))
        rows.append({"name": event.key, "count": int(event.count),
                     "self_cpu_us": cpu, "self_device_us": device,
                     "input_shapes": getattr(event, "input_shapes", None)})
    rows.sort(key=lambda row: (row["self_device_us"], row["count"]), reverse=True)
    identity = {"head": subprocess.check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "source_files": {str(path.relative_to(ROOT)): sha256(path)
                                 for path in source_files},
                "native_extension": bench._native_identity(bench.tensor_collision_multi_hip._extension()),
                "torch_version": torch.__version__, "hip_version": torch.version.hip,
                "device_name": torch.cuda.get_device_name(args.device),
                "device": args.device, "scenario": "full_match_offense",
                "category": args.category, "seed": args.seed,
                "accounting_fusion_gates": "both off", "evidence_type": "profiler attribution only",
                "persistent_state_hashes_after_profile": bench._tensor_hashes(env),
                "trace_sha256": sha256(trace_path)}
    report = {"identity": identity, "trace_path": str(trace_path),
              "operators": rows,
              "target_operator_rows": [row for row in rows if any(
                  token in row["name"].lower() for token in
                  ("nan_to_num", "clamp", "norm", "sqrt", "div", "mul", "cat",
                   "swerve_pose", "swerve"))]}
    report_path = args.output_dir / "a1_sim_tick_trace_summary.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"trace": str(trace_path), "trace_sha256": identity["trace_sha256"],
        "device": identity["device_name"], "device_target": identity["device"],
        "scenario": identity["scenario"], "category": identity["category"],
        "seed": identity["seed"], "summary": str(report_path),
        "top_operators": report["target_operator_rows"][:30]}, indent=2))


if __name__ == "__main__":
    main()
