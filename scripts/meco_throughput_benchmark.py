#!/usr/bin/env python3
"""Run and compare isolated A-track throughput profiles.

This harness does not modify frc_defense source or campaign checkpoints. It
resumes a checkpoint clone in a new output directory and refuses GPU work
without a cooling attestation and an idle AMD device set.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import statistics


class Recorder:
    def __init__(self, torch, device):
        self.torch = torch
        self.device = device
        self.use_events = (torch.cuda.is_available() and
                           torch.version.hip is not None)
        self.data = {}
        self.events = []

    @contextmanager
    def measure(self, name, *, device_work=False):
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        start_event = end_event = None
        if self.use_events and device_work:
            try:
                stream = self.torch.cuda.current_stream(self.device)
                start_event = self.torch.cuda.Event(enable_timing=True)
                end_event = self.torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
            except Exception:
                start_event = end_event = None
        try:
            yield
        finally:
            sample = self.data.setdefault(name, {
                "calls": 0, "wall_seconds": 0.0, "process_seconds": 0.0,
                "device_seconds": 0.0})
            sample["calls"] += 1
            sample["wall_seconds"] += time.perf_counter() - start_wall
            sample["process_seconds"] += time.process_time() - start_cpu
            if start_event is not None:
                try:
                    end_event.record(stream)
                    self.events.append((name, start_event, end_event))
                except Exception:
                    pass

    def wrap(self, owner, attr, name, *, device_work=False):
        original = getattr(owner, attr)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            with self.measure(name, device_work=device_work):
                return original(*args, **kwargs)
        setattr(owner, attr, measured)
        return owner, attr, original

    def call(self, name, function, *args, device_work=False, **kwargs):
        with self.measure(name, device_work=device_work):
            return function(*args, **kwargs)

    def finish(self):
        if self.events:
            self.torch.cuda.synchronize(self.device)
            for name, start, end in self.events:
                self.data[name]["device_seconds"] += start.elapsed_time(end) / 1000.0
        return self.data


class DeviceSampler:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read(path):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            return None

    def _run(self):
        cards = sorted(Path("/sys/class/drm").glob("card[0-9]*/device"))
        while not self.stop.is_set():
            row = {"utc": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                   "devices": []}
            for device in cards:
                if not (device / "vendor").exists():
                    continue
                hwmons = sorted((device / "hwmon").glob("hwmon*"))
                hwmon = hwmons[0] if hwmons else None
                row["devices"].append({
                    "device": str(device),
                    "busy_percent": self._read(device / "gpu_busy_percent"),
                    "vram_used_bytes": self._read(device / "mem_info_vram_used"),
                    "temperature_millidegrees": (self._read(hwmon / "temp1_input")
                        if hwmon else None),
                    "fan_rpm": (self._read(hwmon / "fan1_input")
                        if hwmon else None),
                    "power_microwatts": (self._read(hwmon / "power1_average")
                        if hwmon else None)})
            self.samples.append(row)
            self.stop.wait(self.interval)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        return self.samples


def _device_summary(samples):
    grouped = {}
    for row in samples:
        for device in row["devices"]:
            item = grouped.setdefault(device["device"], {})
            for key in ("busy_percent", "vram_used_bytes",
                        "temperature_millidegrees", "fan_rpm",
                        "power_microwatts"):
                value = device.get(key)
                if value is not None:
                    item.setdefault(key, []).append(value)
    result = {}
    for device, values in grouped.items():
        result[device] = {key: {"mean": statistics.fmean(items),
                                "max": max(items), "samples": len(items)}
                          for key, items in values.items()}
    return result


def _gpu_preflight(torch, device, attestation_path):
    evidence = Path(attestation_path).expanduser().resolve()
    if not evidence.is_file() or not evidence.read_text().strip():
        raise SystemExit("cooling attestation must be a nonempty file describing verified fan/cooling operation")
    attestation = evidence.read_text().strip()
    parsed = torch.device(device)
    index = parsed.index if parsed.index is not None else torch.cuda.current_device()
    result = subprocess.run(["rocm-smi", "--showpids"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise SystemExit("refusing benchmark: rocm-smi process query failed")
    active_devices = []
    if "No KFD PIDs currently running" not in result.stdout:
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[0].isdigit() and fields[1] != "PROCESS":
                try:
                    active_devices.extend(int(value) for value in fields[2].split(","))
                except ValueError:
                    raise SystemExit("refusing benchmark: cannot parse active ROCm device assignments")
        if not active_devices:
            raise SystemExit("refusing benchmark: rocm-smi listed a process but no device assignments")
    if index in active_devices:
        raise SystemExit(f"refusing benchmark: rocm-smi reports a live process on AMD device {index}")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("refusing benchmark: an available ROCm PyTorch runtime is required")
    if parsed.type != "cuda" or (parsed.index is not None and
                                  parsed.index >= torch.cuda.device_count()):
        raise SystemExit(f"requested ROCm device is unavailable: {device}")
    fan_readings = [int(path.read_text()) for path in
        Path("/sys/class/drm").glob("card[0-9]*/device/hwmon/hwmon*/fan1_input")]
    if fan_readings and min(fan_readings) <= 0 and "fan-stop" not in attestation.casefold():
        raise SystemExit("fan RPM is zero; attest verified fan-stop behavior explicitly or verify fan operation")
    if not fan_readings:
        raise SystemExit("cannot read AMD fan RPM; refusing unverified GPU benchmark")
    return {"attestation_path": str(evidence),
        "attestation_sha256": hashlib.sha256(attestation.encode()).hexdigest(),
        "attestation": attestation,
        "rocm_device_index": index,
        "rocm_process_check": result.stdout.strip(),
        "torch_device_count": torch.cuda.device_count(),
        "torch_hip": torch.version.hip,
        "fan_rpm_preflight": fan_readings}


def _campaign_root(checkpoint):
    path = Path(checkpoint).resolve()
    if path.name != "latest.pt" or path.parent.name not in ("A1", "A2", "A3"):
        raise SystemExit("checkpoint must be an A1/A2/A3 latest.pt file")
    return path.parents[2]


def _inside(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _state_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys() and
                all(_state_equal(left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple)):
        return (isinstance(right, type(left)) and len(left) == len(right) and
                all(_state_equal(a, b) for a, b in zip(left, right)))
    return left == right


def run(args):
    source_root = Path(args.source_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    campaign = _campaign_root(checkpoint)
    if _inside(output, campaign) or _inside(checkpoint, output):
        raise SystemExit("benchmark output must be separate from the campaign and checkpoint")
    if output.exists():
        raise SystemExit("benchmark output path already exists; choose a fresh directory")
    if not (source_root / "frc_defense/meco_track_training.py").is_file():
        raise SystemExit("source root is not an AutoDrive checkout")
    sys.path.insert(0, str(source_root))
    import torch
    preflight = _gpu_preflight(torch, args.device, args.cooling_attestation)
    torch.cuda.set_device(torch.device(args.device))
    from frc_defense import meco_track_training as training
    from frc_defense import meco_plan_a_policy as plan_a
    from frc_defense import meco_tracks as tracks
    from frc_defense import curriculum_scenarios
    from frc_defense.curriculum_training import source_identity
    from frc_defense.fuel_physics_hip import prepare_native_fuel_extension
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("config", {}).get("track") not in ("A1", "A2", "A3"):
        raise SystemExit("checkpoint schema/track is not a resumable A-track checkpoint")
    prepare_native_fuel_extension(args.device)
    current_sources = source_identity()
    current_source_hash = hashlib.sha256(json.dumps(current_sources,
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    old_sources = payload.get("sources")
    rebound = old_sources != current_sources
    if rebound and not args.allow_benchmark_source_rebind:
        raise SystemExit("checkpoint source binding differs; use explicit benchmark-only rebind for candidate builds")
    output.mkdir(parents=True)
    resume_copy = output / "benchmark-input.pt"
    if rebound:
        payload["sources"] = current_sources
    payload.setdefault("config", {}).update(output=str(output), device=args.device)
    torch.save(payload, resume_copy)
    resume_sha = hashlib.sha256(resume_copy.read_bytes()).hexdigest()
    recorder = Recorder(torch, args.device)
    originals = []
    for owner, attr, label, device_work in (
        (training, "source_identity", "source_integrity", False),
        (training, "build_legal_pose_graph", "graph_build_startup", False),
        (training, "_snapshot_tensors", "observation_tensorization", True),
        (training, "_graph_inputs", "graph_feature_construction", True),
        (training, "_prepare_plan_a_decision", "policy_decision", True),
        (training, "_policy_inputs_for_row", "ppo_policy_recompute", True),
        (training, "_run_macro_step", "simulator_macro", True),
        (curriculum_scenarios.CurriculumEnv, "observe", "simulator_observation", True),
        (training, "_ppo_update", "ppo_update_total", True),
        (training.ValueHead, "forward", "value_forward", True),
        (training, "_atomic_torch", "checkpoint_write", False),
        (tracks.SE2MessageBlock, "forward", "graph_message_passing", True),
        (plan_a, "build_plan_a_support", "support_construction", True),
        (plan_a, "plan_a_support_digest", "support_digest", False),
        (tracks, "build_observation_snapshot", "raster_construction", True)):
        originals.append(recorder.wrap(owner, attr, label, device_work=device_work))
    original_json = training._atomic_json
    def measured_atomic_json(path, *json_args, **json_kwargs):
        name = Path(path).name
        label = {"heartbeat.json": "heartbeat_write",
                 "stats.json": "stats_write", "run.json": "run_manifest_write",
                 "task-outcomes.json": "task_outcomes_write"}.get(
                     name, "other_manifest_write")
        return recorder.call(label, original_json, path, *json_args, **json_kwargs)
    training._atomic_json = measured_atomic_json
    originals.append((training, "_atomic_json", original_json))
    original_factory = training.make_meco_track
    @functools.wraps(original_factory)
    def measured_factory(*factory_args, **factory_kwargs):
        model = original_factory(*factory_args, **factory_kwargs)
        original_forward = model.forward
        model.forward = functools.wraps(original_forward)(
            lambda *a, **kw: recorder.call("model_forward", original_forward,
                                            *a, device_work=True, **kw))
        return model
    training.make_meco_track = measured_factory
    originals.append((training, "make_meco_track", original_factory))
    original_backward = torch.Tensor.backward
    try:
        torch.Tensor.backward = functools.wraps(original_backward)(
            lambda self, *a, **kw: recorder.call("ppo_backward", original_backward,
                self, *a, device_work=True, **kw))
        originals.append((torch.Tensor, "backward", original_backward))
    except (AttributeError, TypeError):
        pass
    original_step = torch.optim.Adam.step
    torch.optim.Adam.step = functools.wraps(original_step)(
        lambda self, *a, **kw: recorder.call("optimizer_step", original_step,
            self, *a, device_work=True, **kw))
    originals.append((torch.optim.Adam, "step", original_step))
    sampler = DeviceSampler()
    sampler.start()
    config_data = dict(payload["config"])
    config_data.update(device=args.device, output=str(output), updates=args.updates)
    task_assignment = payload.get("task_assignment") or {}
    manifest = {"schema_version": 1, "phase": args.phase,
        "track": config_data["track"], "updates_requested": args.updates,
        "starting_update": payload["update"], "device": args.device,
        "source_root": str(source_root), "source_identity_sha256": current_source_hash,
        "original_source_identity_sha256": hashlib.sha256(json.dumps(
            old_sources, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "benchmark_input_sha256": resume_sha, "benchmark_source_rebind": rebound,
        "protocol_id": payload.get("protocol_id"), "task_assignment": task_assignment,
        "scenario_id": task_assignment.get("scenario_id"),
        "category": task_assignment.get("category"), "seed": task_assignment.get("seed"),
        "host": {"load_average": os.getloadavg(),
            "cpu_model": next((line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), None)},
        "preflight": preflight, "started_utc": datetime.utcnow().isoformat() + "Z"}
    (output / "benchmark-manifest.json").write_text(json.dumps(
        manifest, indent=2, sort_keys=True) + "\n")
    start = time.perf_counter()
    try:
        result = training.train_meco(training.MecoTrainConfig(**config_data),
            resume=str(resume_copy), protocol_id=payload.get("protocol_id"),
            task_assignment=task_assignment)
        torch.cuda.synchronize(args.device)
    finally:
        elapsed = time.perf_counter() - start
        samples = sampler.close()
        for owner, attr, original in reversed(originals):
            setattr(owner, attr, original)
    components = recorder.finish()
    stats = json.loads((output / "stats.json").read_text())
    updates = stats["history"][-args.updates:]
    intervals = []
    for before, after in zip(updates, updates[1:]):
        try:
            delta = (datetime.fromisoformat(after["update_utc"].replace("Z", "+00:00")) -
                     datetime.fromisoformat(before["update_utc"].replace("Z", "+00:00"))).total_seconds()
            intervals.append(delta)
        except (KeyError, ValueError):
            pass
    warmup = min(args.warmup_updates, max(0, len(updates) - 2))
    timed_intervals = intervals[warmup:]
    active_intervals = [value for value in timed_intervals if value < 180.0]
    checkpoint_out = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    semantic_stats = [{k: v for k, v in row.items() if k != "update_utc"}
                      for row in updates]
    report = {**manifest, "finished_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_wall_seconds": elapsed, "result": result,
        "updates_completed": len(updates),
        "warmup_updates_excluded": warmup,
        "process_wall_seconds_per_update": elapsed / max(1, len(updates)),
        "update_intervals_seconds": intervals,
        "timed_update_intervals_seconds": timed_intervals,
        "active_update_intervals_seconds": active_intervals,
        "active_update_median_seconds": (statistics.median(active_intervals)
            if active_intervals else None),
        "active_update_p95_seconds": (sorted(active_intervals)[
            max(0, int(0.95 * len(active_intervals) + 0.999999) - 1)]
            if active_intervals else None),
        "excluded_intervals_over_180s": sum(value >= 180.0 for value in timed_intervals),
        "components_inclusive": components,
        "device_samples": samples,
        "device_utilization": _device_summary(samples),
        "final_checkpoint_sha256": hashlib.sha256((output / "latest.pt").read_bytes()).hexdigest(),
        "semantic_stats": semantic_stats,
        "final_state": {"model": checkpoint_out["model"],
            "value_head": checkpoint_out["value_head"],
            "optimizer": checkpoint_out["optimizer"]}}
    # State tensors are compared by the compare subcommand; keep the summary small.
    torch.save(report.pop("final_state"), output / "final-state.pt")
    report["final_state_sha256"] = hashlib.sha256((output / "final-state.pt").read_bytes()).hexdigest()
    (output / "profile.json").write_text(json.dumps(report, indent=2,
        sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("track", "phase", "updates_completed",
        "elapsed_wall_seconds", "process_wall_seconds_per_update",
        "final_state_sha256")}, indent=2))


def compare(args):
    import torch
    base_path, candidate_path = Path(args.baseline).resolve(), Path(args.candidate).resolve()
    base = json.loads((base_path / "profile.json").read_text())
    candidate = json.loads((candidate_path / "profile.json").read_text())
    base_state = torch.load(base_path / "final-state.pt", map_location="cpu", weights_only=False)
    candidate_state = torch.load(candidate_path / "final-state.pt", map_location="cpu", weights_only=False)
    same_initial = (base["track"] == candidate["track"] and
        base["checkpoint_sha256"] == candidate["checkpoint_sha256"] and
        base["protocol_id"] == candidate["protocol_id"] and
        base["device"] == candidate["device"] and
        base["starting_update"] == candidate["starting_update"] and
        base["updates_requested"] == candidate["updates_requested"] and
        base["warmup_updates_excluded"] == candidate["warmup_updates_excluded"] and
        base["scenario_id"] == candidate["scenario_id"] and
        base["category"] == candidate["category"] and base["seed"] == candidate["seed"])
    baseline_s = float(base["active_update_median_seconds"])
    candidate_s = float(candidate["active_update_median_seconds"])
    result = {"track": base["track"], "same_initial_state_and_task": same_initial,
        "updates": base["updates_completed"],
        "warmup_updates_excluded": base["warmup_updates_excluded"],
        "baseline_seconds_per_update": baseline_s,
        "candidate_seconds_per_update": candidate_s,
        "speedup": baseline_s / candidate_s if candidate_s else None,
        "percent_faster": 100.0 * (baseline_s - candidate_s) / baseline_s if baseline_s else None,
        "semantic_stats_equal": base["semantic_stats"] == candidate["semantic_stats"],
        "final_model_value_optimizer_state_equal": _state_equal(base_state, candidate_state),
        "baseline_device_samples": base["device_samples"],
        "candidate_device_samples": candidate["device_samples"],
        "baseline_device_utilization": base["device_utilization"],
        "candidate_device_utilization": candidate["device_utilization"],
        "baseline_components": base["components_inclusive"],
        "candidate_components": candidate["components_inclusive"]}
    Path(args.output).write_text(json.dumps(result, indent=2,
        sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("track", "same_initial_state_and_task",
        "baseline_seconds_per_update", "candidate_seconds_per_update", "speedup",
        "semantic_stats_equal", "final_model_value_optimizer_state_equal")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--phase", choices=("baseline", "candidate"), required=True)
    run_parser.add_argument("--source-root", required=True)
    run_parser.add_argument("--checkpoint", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--device", required=True)
    run_parser.add_argument("--updates", type=int, default=10)
    run_parser.add_argument("--warmup-updates", type=int, default=2)
    run_parser.add_argument("--cooling-attestation", required=True,
        help="text file recording independent fan/cooling verification")
    run_parser.add_argument("--allow-benchmark-source-rebind", action="store_true",
        help="rebind only a benchmark checkpoint clone; never writes the campaign checkpoint")
    run_parser.set_defaults(func=run)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.set_defaults(func=compare)
    args = parser.parse_args()
    if args.command == "run" and (args.updates < 4 or args.warmup_updates < 0 or
                                   args.warmup_updates >= args.updates - 1):
        parser.error("use at least four updates and leave at least two timed updates after warmup")
    args.func(args)


if __name__ == "__main__":
    main()
