#!/usr/bin/env python3
"""Prepare or run the 119-case corrected-source study on two GPU slots.

Dry-run is the default. --run requires the frozen runtime source hash and an
empty output directory; each case starts a fresh virtual-environment Python.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

REPO = Path(__file__).resolve().parents[1]
MODEL_SLOTS = {
    0: ("full_field_cnn", "deep_sets", "field_fno"),
    1: ("baseline_mlp", "residual_mlp", "full_field_resnet", "entity_attention"),
}
SCENARIO_BUDGETS = {
    "pickup_simple": (200, 20), "pickup_scattered": (200, 20),
    "deterministic_defense": (400, 40), "defense_3v3": (400, 40),
    "offense_vs_deterministic_defense": (600, 60),
}


def runtime_hash():
    """Match architecture_benchmark.source_provenance without importing Torch."""
    files = sorted((REPO / "frc_defense").glob("*.py"))
    files += sorted((REPO / "frc_defense/csrc").glob("*"))
    files += sorted((REPO / "frc_defense/dashboard_static").glob("*"))
    hashes = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files if path.is_file()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def build_plan(output, python):
    output, python = output.resolve(), python.absolute()
    queues = {slot: [] for slot in MODEL_SLOTS}
    for slot, models in MODEL_SLOTS.items():
        conditions = [(model, scenario, seed, .0003)
                      for scenario in SCENARIO_BUDGETS for model in models for seed in (7, 19, 43)]
        conditions += [(model, "deterministic_defense", 7, rate)
                       for model in models for rate in (.0001, .001)]
        for model, scenario, seed, rate in conditions:
            horizon, rollout = SCENARIO_BUDGETS[scenario]
            directory = output / model / scenario / f"seed-{seed}-lr-{rate:g}"
            command = [str(python), "-m", "frc_defense.architecture_benchmark",
                       "--architectures", model, "--scenarios", scenario,
                       "--seeds", str(seed), "--learning-rates", f"{rate:g}",
                       "--envs", "8", "--eval-envs", "4", "--horizon", str(horizon),
                       "--rollout-steps", str(rollout), "--eval-ticks", str(horizon),
                       "--updates", "4", "--eval-every", "2", "--action-repeat", "10",
                       "--device", f"cuda:{slot}", "--disable-miopen", "--output", str(output)]
            queues[slot].append({"architecture": model, "scenario": scenario, "seed": seed,
                                 "learning_rate": rate, "device": f"cuda:{slot}",
                                 "directory": str(directory), "command": command, "status": "pending"})
    return queues


def run_plan(queues, output, expected_hash):
    stop = threading.Event()
    cancel = threading.Event()
    lock = threading.Lock()
    errors = []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = output.parent / f"v5-worker-logs-{run_id}"
    logs.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "orchestration-v5.json"
    manifest = {"status": "running", "expected_source_sha256": expected_hash,
                "case_count": sum(map(len, queues.values())), "run_id_utc": run_id,
                "logs": str(logs), "queues": queues}

    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)

    def interrupt(signum, frame):
        cancel.set()
        stop.set()

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    save()

    def worker(slot):
        for case in queues[slot]:
            if stop.is_set():
                break
            try:
                if runtime_hash() != expected_hash:
                    raise RuntimeError("runtime source changed before case start")
                stem = f"slot-{slot}-{case['architecture']}-{case['scenario']}-seed-{case['seed']}-lr-{case['learning_rate']:g}"
                log = logs / f"{stem}.log"
                with lock:
                    case.update(status="running", log=str(log))
                    save()
                    print(json.dumps({"event": "start", "slot": slot, "case": stem, "log": str(log)}), flush=True)
                environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with log.open("x") as handle:
                    process = subprocess.Popen(case["command"], cwd=REPO, env=environment,
                                               stdout=handle, stderr=subprocess.STDOUT)
                    with lock:
                        case["pid"] = process.pid
                        save()
                    while process.poll() is None:
                        if cancel.is_set():
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            break
                        time.sleep(.5)
                if cancel.is_set():
                    with lock:
                        case.update(status="cancelled", returncode=process.returncode)
                        save()
                    break
                if process.returncode:
                    raise RuntimeError(f"case process exited {process.returncode}; see {log}")
                metadata = json.loads((Path(case["directory"]) / "metadata.json").read_text())
                if metadata.get("status") != "complete":
                    raise RuntimeError("process ended without complete metadata")
                if metadata["provenance"]["source_sha256"] != expected_hash or runtime_hash() != expected_hash:
                    raise RuntimeError("case or current runtime does not match frozen source")
                with lock:
                    case.update(status="complete", returncode=0)
                    save()
                    print(json.dumps({"event": "complete", "slot": slot, "case": stem,
                                      "completed": sum(c["status"] == "complete" for q in queues.values() for c in q)}), flush=True)
            except Exception as error:
                with lock:
                    case.update(status="failed", error=str(error))
                    errors.append(str(error))
                    stop.set()
                    save()
                    print(json.dumps({"event": "failed", "slot": slot, "error": str(error)}), flush=True)
                break

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker, slot) for slot in queues]
            for future in futures:
                future.result()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        with lock:
            completed = sum(case["status"] == "complete" for queue in queues.values() for case in queue)
            manifest.update(status="cancelled" if cancel.is_set() else "failed" if errors else "complete",
                            completed_cases=completed)
            save()
    print(json.dumps({"status": manifest["status"], "completed": completed, "manifest": str(manifest_path)}), flush=True)
    return 130 if cancel.is_set() else 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "evaluations/architecture-study-20261020/matrix")
    parser.add_argument("--python", type=Path, default=REPO / ".venv/bin/python")
    parser.add_argument("--expected-source", help="required frozen runtime SHA256 for --run")
    parser.add_argument("--run", action="store_true", help="launch cases; otherwise print a read-only plan")
    args = parser.parse_args()
    queues = build_plan(args.output, args.python)
    current = runtime_hash()
    if not args.run:
        print(json.dumps({"mode": "dry-run", "current_source_sha256": current,
                          "total_cases": sum(map(len, queues.values())),
                          "per_slot": {str(slot): {"models": MODEL_SLOTS[slot], "cases": len(queue),
                                                   "first_command": queue[0]["command"],
                                                   "last_command": queue[-1]["command"]}
                                       for slot, queue in queues.items()}}, indent=2))
        return 0
    if not args.expected_source or args.expected_source != current:
        parser.error("--run requires --expected-source matching the current frozen runtime hash")
    if not args.python.is_file():
        parser.error("--python must point to the simulator's installed virtual-environment Python")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be empty; archive the legacy matrix before the uniform rerun")
    return run_plan(queues, output, args.expected_source)


if __name__ == "__main__":
    raise SystemExit(main())
