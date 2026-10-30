#!/usr/bin/env python3
"""Paired A/B/A A1 macro benchmark for fused six-robot tick accounting.

This harness is intentionally an execution-only benchmark; do not run it on
CPU because that would exercise the Torch fallback rather than the HIP path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frc_defense import meco_track_training as training
from frc_defense import tensor_collision_multi_hip
from frc_defense.curriculum_scenarios import make_curriculum_env
from frc_defense.fuel_physics_hip import native_extension_identity as fuel_native_identity
from frc_defense.fuel_physics_runtime import advance_with_fuel

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 2970053994459693056
DEFAULT_CATEGORY = "active/open/level_5_mixed"
GATE = "AUTODRIVE_EXPERIMENTAL_FUSED_SIM_TICK_ACCOUNTING"
HASHED_SOURCE_FILES = (
    "frc_defense/tensor_physics.py",
    "frc_defense/tensor_collision_multi_hip.py",
    "frc_defense/csrc/tensor_collision_multi_hip.cpp",
    "frc_defense/csrc/tensor_collision_multi_hip_kernel.cu",
    "profiling/combined_sim_accounting_ab_a.py",
)


def _make_env(*, device: str, seed: int, category: str, fused: bool):
    os.environ[GATE] = "1" if fused else "0"
    os.environ["AUTODRIVE_EXPERIMENTAL_CONTACT_ACCOUNTING_FUSION"] = "1" if fused else "0"
    env = make_curriculum_env(
        "full_match_offense", category, 1, device, seed,
        fuel_physics=training._training_fuel_physics_config("A1"))
    training._reset_episode(env, seed)
    return env


def _persistent_tensors(env) -> dict[str, torch.Tensor]:
    groups = (("env", env), ("sim", env.sim), ("fuel", env.fuel_physics),
              ("hip", env.fuel_physics._hip))
    result: dict[str, torch.Tensor] = {}
    for prefix, obj in groups:
        for name, value in vars(obj).items():
            if isinstance(value, torch.Tensor):
                key = f"{prefix}.{name}"
                if key in result:
                    raise RuntimeError(f"duplicate persistent tensor key: {key}")
                result[key] = value
    return result


def _tensor_hashes(env) -> dict[str, dict[str, Any]]:
    hashes = {}
    for name, tensor in sorted(_persistent_tensors(env).items()):
        contiguous = tensor.detach().contiguous()
        raw = contiguous.view(torch.uint8).cpu().numpy().tobytes()
        hashes[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return hashes


def _state_differences(reference: dict, actual: dict) -> list[dict[str, Any]]:
    differences = []
    for name in sorted(set(reference) | set(actual)):
        if reference.get(name) != actual.get(name):
            differences.append({
                "tensor": name,
                "reference": reference.get(name),
                "actual": actual.get(name),
            })
    return differences


def _commands(env, device: str):
    # Production A1 scenario macro command profile used by the prior contact
    # accounting A/B run, held fixed so only tick-accounting dispatch changes.
    return torch.tensor(
        [[[.25, .05, 0.], [.15, -.05, 0.], [.2, .1, .02],
          [-.25, .05, 0.], [-.15, -.05, 0.], [-.2, .1, -.02]]],
        device=device, dtype=env.sim.pose.dtype)


def _advance_timed(env, commands, *, fused: bool, device: str) -> float:
    os.environ[GATE] = "1" if fused else "0"
    os.environ["AUTODRIVE_EXPERIMENTAL_CONTACT_ACCOUNTING_FUSION"] = (
        "1" if fused else "0")
    active = torch.ones((env.n,), device=device, dtype=torch.bool)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    advance_with_fuel(env, commands, active)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_identity(module) -> dict[str, Any]:
    path = Path(module.__file__).resolve()
    stat = path.stat()
    return {
        "module": module.__name__,
        "path": str(path),
        "sha256": _sha256_file(path),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "has_sim_tick_accounting": hasattr(module, "sim_tick_accounting"),
    }


def _source_identity() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1"], text=True)
    per_file = {name: _sha256_file(ROOT / name) for name in HASHED_SOURCE_FILES}
    digest = hashlib.sha256()
    for name, value in sorted(per_file.items()):
        digest.update(name.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    diff = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--binary", "HEAD", "--",
         *HASHED_SOURCE_FILES[:-1]])
    diff_hash = hashlib.sha256(diff).hexdigest()
    return {
        "root": str(ROOT),
        "head": head,
        "status_porcelain": status.splitlines(),
        "tracked_source_diff_sha256": diff_hash,
        "source_files": per_file,
        "source_bundle_sha256": digest.hexdigest(),
    }


def _mean_absolute_deviation(values: list[float]) -> float:
    median = statistics.median(values)
    return statistics.median([abs(value - median) for value in values])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--pairs", type=int, default=12,
                        help="number of A/B/A macro triplets after warmup")
    parser.add_argument("--output", default="profiling/sim-tick-accounting-macro-ab-a.json")
    args = parser.parse_args()
    if args.pairs < 3:
        parser.error("--pairs must be at least 3")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("A1 macro benchmark requires a HIP device; refusing fallback timing")
    torch.cuda.set_device(args.device)

    # One extension instance serves both A and B. This proves the candidate
    # native entrypoint exists before timing and avoids cache identity mixing.
    os.environ[GATE] = "0"
    sim_ext = tensor_collision_multi_hip._extension()
    if sim_ext is None or not hasattr(sim_ext, "sim_tick_accounting"):
        raise RuntimeError("built six-robot HIP extension lacks sim_tick_accounting")

    env_a0 = _make_env(device=args.device, seed=args.seed,
                       category=args.category, fused=False)
    env_b = _make_env(device=args.device, seed=args.seed,
                      category=args.category, fused=True)
    env_a1 = _make_env(device=args.device, seed=args.seed,
                       category=args.category, fused=False)
    commands = [_commands(env, args.device) for env in (env_a0, env_b, env_a1)]
    active = [torch.ones((env.n,), device=args.device, dtype=torch.bool)
              for env in (env_a0, env_b, env_a1)]
    torch.cuda.synchronize()

    # Warm each graph/path before capturing timings; retain the post-warm state
    # only if the three variants are exactly equal.
    for env, command, fused in ((env_a0, commands[0], False),
                                (env_b, commands[1], True),
                                (env_a1, commands[2], False)):
        os.environ[GATE] = "1" if fused else "0"
        os.environ["AUTODRIVE_EXPERIMENTAL_CONTACT_ACCOUNTING_FUSION"] = (
            "1" if fused else "0")
        advance_with_fuel(env, command, torch.ones_like(active[0]))
    torch.cuda.synchronize()
    warm_hashes = [_tensor_hashes(env) for env in (env_a0, env_b, env_a1)]
    for label, hashes in (("B", warm_hashes[1]), ("A_repeat", warm_hashes[2])):
        differences = _state_differences(warm_hashes[0], hashes)
        if differences:
            raise AssertionError(json.dumps({"stage": "warmup", "variant": label,
                                             "differences": differences[:20]}, indent=2))

    samples: dict[str, list[float]] = {"A0_baseline_ms": [],
                                       "B_candidate_ms": [],
                                       "A1_baseline_repeat_ms": []}
    macro_hashes = []
    for macro in range(1, args.pairs + 1):
        # Baseline controls bracket the candidate in every paired triplet.
        order = [0, 1, 2]
        elapsed = {}
        for index in order:
            fused = index == 1
            elapsed[index] = _advance_timed(
                (env_a0, env_b, env_a1)[index], commands[index],
                fused=fused, device=args.device)
        for index, key in ((0, "A0_baseline_ms"), (1, "B_candidate_ms"),
                           (2, "A1_baseline_repeat_ms")):
            samples[key].append(elapsed[index])
        hashes = [_tensor_hashes(env) for env in (env_a0, env_b, env_a1)]
        triplet = {"macro": macro, "A0": hashes[0], "B": hashes[1], "A1": hashes[2]}
        for label, reference, actual in (("B", hashes[0], hashes[1]),
                                         ("A1", hashes[0], hashes[2])):
            differences = _state_differences(reference, actual)
            if differences:
                triplet["mismatch"] = {"variant": label, "differences": differences}
                macro_hashes.append(triplet)
                raise AssertionError(json.dumps({"macro": macro, **triplet}, indent=2))
        macro_hashes.append(triplet)

    a_values = samples["A0_baseline_ms"] + samples["A1_baseline_repeat_ms"]
    a_median = statistics.median(a_values)
    b_median = statistics.median(samples["B_candidate_ms"])
    report = {
        "schema_version": 1,
        "purpose": "combined gated six-robot HIP contact and tick accounting A/B/A benchmark",
        "scenario": f"full_match_offense/{args.category}",
        "track": "A1",
        "seed": args.seed,
        "worlds": 1,
        "robots": 6,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "device_index": torch.cuda.current_device(),
        "hip": torch.version.hip,
        "pairs": args.pairs,
        "macro_order": "A0/B/A1 triplets",
        "capture_macro_excluded": True,
        "warmup_macros_excluded": True,
        "gate": GATE,
        "source_identity": _source_identity(),
        "native_identity": {
            "sim_collision_extension": _native_identity(sim_ext),
            "fuel_extension": fuel_native_identity(),
        },
        "samples_ms": samples,
        "medians_ms": {
            "baseline_A_mean_median": a_median,
            "candidate_B": b_median,
            "A0": statistics.median(samples["A0_baseline_ms"]),
            "A1": statistics.median(samples["A1_baseline_repeat_ms"]),
        },
        "baseline_repeat_median_ratio_A1_over_A0": (
            statistics.median(samples["A1_baseline_repeat_ms"]) /
            statistics.median(samples["A0_baseline_ms"])),
        "baseline_median_absolute_deviation_ms": _mean_absolute_deviation(a_values),
        "candidate_speedup_vs_baseline": a_median / b_median,
        "per_macro_exact_state_hashes_equal": True,
        "state_hashes": macro_hashes,
    }
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "state_hashes"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Full per-tensor hashes: {destination}")


if __name__ == "__main__":
    main()
