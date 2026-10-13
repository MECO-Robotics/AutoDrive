#!/usr/bin/env python3
"""Run a reproducible batch of full 3v3 matches for offense experiments.

Each vector-environment slot is a distinct randomized match generated from the
same batch seed. Run candidate worktrees with the same seed, slot count, and
simulator settings to pair the resulting scenarios.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import platform
import re
import subprocess
import sys
import time
from types import ModuleType
from pathlib import Path

import torch

import frc_defense.tensor_3v3 as tensor_3v3_module
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ENV_KEYS = (
    "TORCH_EXTENSIONS_DIR", "PYTORCH_ROCM_ARCH", "TORCH_CUDA_ARCH_LIST",
    "ROCM_HOME", "HIP_HOME", "CUDA_HOME", "HIPCC", "CC", "CXX",
)
NATIVE_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hip", ".hpp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_source(module: ModuleType) -> dict[str, str] | None:
    path = getattr(module, "__file__", None)
    if not path:
        return None
    source_path = Path(path).resolve()
    if source_path.suffix != ".py" or not source_path.is_file():
        return None
    return {"path": str(source_path), "sha256": _sha256(source_path)}


def _checkout_simulator_module() -> dict[str, str]:
    module_path = Path(tensor_3v3_module.__file__).resolve()
    try:
        module_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            "frc_defense.tensor_3v3 resolved outside the evaluator checkout: "
            f"{module_path}; set PYTHONPATH to {REPO_ROOT}"
        ) from exc
    if not module_path.is_file():
        raise RuntimeError(f"simulator module source is missing: {module_path}")
    return {"path": str(module_path), "sha256": _sha256(module_path)}


def _loaded_native_extensions() -> list[dict[str, str]]:
    candidates: dict[str, str] = {}

    def add(module_name: str, module: ModuleType) -> None:
        path = getattr(module, "__file__", None)
        if not path or not (
            module_name.startswith("autodrive_") or
            module_name.startswith("frc_defense.")
        ):
            return
        binary_path = Path(path).resolve()
        is_extension = any(
            str(binary_path).endswith(suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        )
        if is_extension and binary_path.is_file():
            candidates[str(binary_path)] = module_name

    for module_name, module in tuple(sys.modules.items()):
        if isinstance(module, ModuleType):
            add(module_name, module)
            if module_name.startswith("frc_defense."):
                for attribute, value in vars(module).items():
                    if isinstance(value, ModuleType):
                        add(f"{module_name}.{attribute}", value)

    return [
        {"module": module_name, "path": path, "sha256": _sha256(Path(path))}
        for path, module_name in sorted(candidates.items())
    ]


def _runtime_provenance(args: argparse.Namespace,
                        simulator_module: dict[str, str]) -> dict:
    device = torch.device(args.device)
    device_info = {"requested": args.device, "type": device.type, "index": device.index}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device_info.update({
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
            "gcn_arch_name": getattr(properties, "gcnArchName", None),
        })

    python_modules = []
    for module_name, module in sorted(sys.modules.items()):
        if module_name == "frc_defense" or module_name.startswith("frc_defense."):
            if isinstance(module, ModuleType):
                source = _module_source(module)
                if source is not None:
                    python_modules.append({"module": module_name, **source})

    try:
        from torch.utils import cpp_extension
        default_extension_root = cpp_extension.get_default_build_root()
    except (ImportError, AttributeError):
        default_extension_root = None

    native_sources = []
    source_root = REPO_ROOT / "frc_defense"
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in NATIVE_SOURCE_SUFFIXES:
            native_sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})

    return {
        "evaluator_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "invocation": {
            "argv": list(sys.argv),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "working_directory": str(Path.cwd().resolve()),
            "script_path": str(Path(__file__).resolve()),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "torch": {
            "version": torch.__version__,
            "hip_version": torch.version.hip,
            "cuda_version": torch.version.cuda,
        },
        "device": device_info,
        "extension_environment": {
            key: os.environ.get(key) for key in EXTENSION_ENV_KEYS
        },
        "default_torch_extension_root": (
            str(default_extension_root) if default_extension_root is not None else None
        ),
        "simulator_module": simulator_module,
        "python_modules": python_modules,
        "native_sources": native_sources,
        "loaded_native_extensions": _loaded_native_extensions(),
    }


def _revision() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def evaluate(args: argparse.Namespace) -> dict:
    simulator_module = _checkout_simulator_module()
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        # HIP extension launches use the current device stream. Match it to
        # the requested tensor device before importing the simulator's lazy
        # native extensions or constructing the environment.
        torch.cuda.set_device(device.index)
    modes = ("deterministic",) * 6
    env = TensorThreeVsThreeEnv(
        num_envs=args.envs,
        device=args.device,
        seed=args.seed,
        control_modes=modes,
        robot_roles=("offense",) * 3 + ("defense",) * 3,
        horizon=args.ticks,
        dt=0.02,
        randomize=True,
        fuel_count=504,
        fused_sensor_rng=True,
        fuel_physics=args.fuel_physics,
        replan_interval=args.replan_interval,
        perception_interval=args.perception_interval,
        teammate_coordination_enabled=args.teammate_coordination_enabled,
    )
    env.reset(seed=args.seed)
    actions = torch.full((args.envs, 6), 7, device=args.device, dtype=torch.long)

    trace_buffer = None
    trace_sample_count = 0
    if args.trace_slot is not None:
        trace_capacity = (args.ticks + args.trace_every - 1) // args.trace_every
        # Keep the recorder on the simulator device and transfer it once after
        # the rollout. The 125-value record also retains selected fuel indices
        # and target-selection steps for exact target-switch/dwell analysis.
        trace_buffer = torch.empty(
            (trace_capacity, 125), device=args.device, dtype=env.sim.pose.dtype)

    started = time.perf_counter()
    if trace_buffer is None:
        for _ in range(args.ticks):
            env.step(actions, capture_observation=False, capture_info=False)
    else:
        for tick in range(args.ticks):
            env.step(actions, capture_observation=False, capture_info=False)
            if (tick + 1) % args.trace_every == 0 or tick + 1 == args.ticks:
                slot = args.trace_slot
                possession = torch.stack([
                    (env.piece_owner[slot] == robot).sum()
                    for robot in range(6)
                ]).to(env.sim.pose.dtype)
                snapshot = torch.cat((
                    env.sim.pose[slot].reshape(-1),
                    env.sim.velocity[slot, :, :2].reshape(-1),
                    env.last_actions[slot].to(env.sim.pose.dtype),
                    env._last_audit_targets[slot].reshape(-1),
                    env._last_audit_fuel_targets[slot].reshape(-1),
                    env._target_fuel_index[slot].to(env.sim.pose.dtype),
                    env._target_fuel_selected_step[slot].to(env.sim.pose.dtype),
                    env.planner.last_goal.reshape(args.envs, 6, 2)[slot].reshape(-1),
                    env._intake_collecting[slot].to(env.sim.pose.dtype),
                    env._target_collecting[slot].to(env.sim.pose.dtype),
                    env.fuel_acquisition_count[slot].to(env.sim.pose.dtype),
                    env.fuel_score_count[slot].to(env.sim.pose.dtype),
                    env.fuel_scored_event[slot].to(env.sim.pose.dtype),
                    env.fuel_acquired_event[slot].to(env.sim.pose.dtype),
                    possession,
                    env.hub_active[slot].to(env.sim.pose.dtype),
                    env.match_elapsed[slot:slot+1],
                )).detach()
                trace_buffer[trace_sample_count].copy_(snapshot)
                trace_sample_count += 1
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started

    batch_scores = env.fuel_score_count.detach().cpu().tolist()
    batch_acquisitions = env.fuel_acquisition_count.detach().cpu().tolist()
    differentials = [int(row[0]) - int(row[1]) for row in batch_scores]
    revision = _revision()
    provenance = _runtime_provenance(args, simulator_module)
    result = {
        "schema_version": 1,
        "label": args.label,
        "revision": revision,
        "provenance": provenance,
        "scenario": {
            "batch_seed": args.seed,
            "world_slots": list(range(args.envs)),
            "scenario_ids": [f"{args.seed}:{slot}" for slot in range(args.envs)],
            "seed_semantics": "one seeded vector batch; slots are distinct worlds, not independently seeded resets",
        },
        "settings": {
            "ticks": args.ticks,
            "seconds_per_match": args.ticks * 0.02,
            "device": args.device,
            "controllers": list(modes),
            "robot_roles": ["offense"] * 3 + ["defense"] * 3,
            "fuel_physics": args.fuel_physics,
            "fused_sensor_rng": True,
            "replan_interval": args.replan_interval,
            "perception_interval": args.perception_interval,
            "teammate_intent_knowledge": args.teammate_coordination_enabled,
            "teammate_coordination_enabled": args.teammate_coordination_enabled,
        },
        "elapsed_seconds": elapsed,
        "matches_per_second": args.envs / elapsed,
        "per_match": [
            {
                "scenario_id": f"{args.seed}:{slot}",
                "score_by_team": score,
                "score_differential_team0_minus_team1": differentials[slot],
                "fuel_acquisitions_by_robot": acquisition,
            }
            for slot, (score, acquisition) in enumerate(
                zip(batch_scores, batch_acquisitions))
        ],
        "summary": {
            "mean_team0_score": sum(row[0] for row in batch_scores) / args.envs,
            "mean_team1_score": sum(row[1] for row in batch_scores) / args.envs,
            "mean_score_differential": sum(differentials) / args.envs,
            "total_team0_score": sum(row[0] for row in batch_scores),
            "total_team1_score": sum(row[1] for row in batch_scores),
        },
    }
    if trace_buffer is not None:
        expected_trace_samples = (args.ticks + args.trace_every - 1) // args.trace_every
        if trace_sample_count != expected_trace_samples:
            raise RuntimeError(
                f"trace captured {trace_sample_count} samples; "
                f"expected {expected_trace_samples}"
            )
        rows = trace_buffer[:trace_sample_count].cpu().tolist()
        frames = []
        for sample_index, snapshot in enumerate(rows):
            tick = min((sample_index + 1) * args.trace_every, args.ticks)
            i = 0

            def take(count: int):
                nonlocal i
                values = snapshot[i:i + count]
                i += count
                return values

            pose = [take(3) for _ in range(6)]
            velocity = [take(2) for _ in range(6)]
            robot_actions = [int(x) for x in take(6)]
            targets = [take(2) for _ in range(6)]
            fuel_targets = [take(2) for _ in range(6)]
            fuel_target_indices = [int(x) for x in take(6)]
            fuel_target_selected_steps = [int(x) for x in take(6)]
            route_goals = [take(2) for _ in range(6)]
            collecting = [bool(x) for x in take(6)]
            target_collecting = [bool(x) for x in take(6)]
            acquisitions = [int(x) for x in take(6)]
            scores = [int(x) for x in take(2)]
            scored_events = [int(x) for x in take(6)]
            acquired_events = [int(x) for x in take(6)]
            possession = [int(x) for x in take(6)]
            hubs = [bool(x) for x in take(2)]
            match_elapsed = take(1)[0]
            if i != len(snapshot):
                raise RuntimeError("trace sample layout mismatch")
            frames.append({
                "tick": tick, "time_s": match_elapsed, "robots": pose,
                "velocity_mps": velocity, "actions": robot_actions,
                "targets": targets, "fuel_targets": fuel_targets,
                "fuel_target_indices": fuel_target_indices,
                "fuel_target_selected_steps": fuel_target_selected_steps,
                "route_goals": route_goals, "collecting": collecting,
                "target_collecting": target_collecting,
                "acquisitions_by_robot": acquisitions,
                "score_by_team": scores,
                "scored_event_by_robot": scored_events,
                "acquired_event_by_robot": acquired_events,
                "possession_by_robot": possession,
                "hub_active_by_team": hubs,
            })
        expected_ticks = list(range(args.trace_every, args.ticks + 1, args.trace_every))
        if not expected_ticks or expected_ticks[-1] != args.ticks:
            expected_ticks.append(args.ticks)
        actual_ticks = [frame["tick"] for frame in frames]
        if actual_ticks != expected_ticks or actual_ticks[-1] != args.ticks:
            raise RuntimeError(
                f"trace frame ticks do not match expected schedule; "
                f"expected final tick {args.ticks}, got "
                f"{actual_ticks[-1] if actual_ticks else None}"
            )
        if (frames[-1]["score_by_team"] != batch_scores[args.trace_slot] or
                frames[-1]["acquisitions_by_robot"] !=
                batch_acquisitions[args.trace_slot]):
            raise RuntimeError("final trace totals do not match the same batch world")
        result["trace"] = {
            "schema_version": 2,
            "label": f"{args.label}-trace-{args.seed}-{args.trace_slot}",
            "revision": revision,
            "provenance": provenance,
            "seed": args.seed,
            "scenario_id": f"{args.seed}:{args.trace_slot}",
            "scenario_world_slot": args.trace_slot,
            "field": {
                "length_m": env.field_length,
                "width_m": float(env.sim.field_width),
                "alliance_zone_depth_m": env.alliance_zone_depth,
                "hub_centers": env.hub_centers.cpu().tolist(),
            },
            "settings": {
                "ticks": args.ticks, "dt": 0.02,
                "sample_every_ticks": args.trace_every,
                "envs": args.envs, "world_slot": args.trace_slot,
                "device": args.device,
                "teammate_coordination": args.teammate_coordination_enabled,
                "replan_interval": args.replan_interval,
                "perception_interval": args.perception_interval,
                "fuel_physics": args.fuel_physics,
                "fused_sensor_rng": True,
            },
            "elapsed_seconds": elapsed,
            "frames": frames,
            "final": {
                "score_by_team": batch_scores[args.trace_slot],
                "acquisitions_by_robot": batch_acquisitions[args.trace_slot],
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="short candidate or worktree name")
    parser.add_argument("--seed", type=int, default=64821, help="batch seed; keep fixed across candidates")
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        help="evaluate several batch seeds in one Python process; each seed gets a fresh environment",
    )
    parser.add_argument("--envs", type=int, default=32, help="randomized match worlds in one batch")
    parser.add_argument("--ticks", type=int, default=8000, help="physics ticks per full match")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fuel-physics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replan-interval", type=int, default=20)
    parser.add_argument("--perception-interval", type=int, default=2)
    parser.add_argument("--teammate-coordination", "--teammate-intent-knowledge",
                        dest="teammate_coordination_enabled",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="enable teammate coordination (default: siloed robots)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-slot", type=int,
                        help="optionally record one world's trajectory during this batch")
    parser.add_argument("--trace-every", type=int, default=10,
                        help="ticks between read-only device-side trace snapshots")
    parser.add_argument("--trace-output", type=Path,
                        help="write the trace sidecar separately from the batch result")
    parser.add_argument(
        "--output-dir", type=Path,
        help="with --seeds, write one existing-schema JSON artifact per seed here",
    )
    args = parser.parse_args()
    if args.envs < 1 or args.ticks < 1:
        parser.error("envs and ticks must be positive")
    if args.replan_interval < 1 or args.perception_interval < 1:
        parser.error("planner and perception intervals must be positive")
    if args.trace_every < 1:
        parser.error("trace interval must be positive")
    if args.trace_slot is None:
        if args.trace_output is not None:
            parser.error("--trace-output requires --trace-slot")
    elif not 0 <= args.trace_slot < args.envs:
        parser.error("trace slot must be in range for env count")
    if args.trace_slot is not None and args.seeds is not None:
        parser.error("trace capture currently requires a single --seed invocation")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error("label may contain only letters, numbers, dot, underscore, and hyphen")

    if args.seeds is not None:
        if args.output is not None or args.output_dir is None:
            parser.error("--seeds requires --output-dir and cannot be combined with --output")
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must not contain duplicates")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            args.seed = seed
            result = evaluate(args)
            output = args.output_dir / f"{args.label}-{seed}.json"
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(f"wrote {output}")
        return
    if args.output_dir is not None:
        parser.error("--output-dir requires --seeds")

    result = evaluate(args)
    trace = result.pop("trace", None)
    if trace is not None and args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(trace, indent=2) + "\n")
        result["trace_output"] = str(args.trace_output)
    elif trace is not None:
        result["trace"] = trace
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"wrote {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
