#!/usr/bin/env python3
"""Benchmark candidate generation against the current Torch expression."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from frc_defense.tensor_fuel_candidate_rank import candidates, available


def reference(points, robot, free):
    delta = points - robot[:, None, :]
    distance = delta.norm(dim=-1).masked_fill(~free, float("inf"))
    nearest, indices = distance.topk(4, dim=-1, largest=False)
    return indices, torch.isfinite(nearest), nearest


def measure(fn, repeat):
    values = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 100.0)
    return {"median_ms_per_call": statistics.median(values), "samples_ms": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if not available("cuda"):
        raise RuntimeError("HIP candidate extension unavailable")
    torch.manual_seed(7729)
    n, p = args.envs, 504
    points = torch.rand((n, p, 2), device="cuda") * 20
    robot = torch.rand((n, 2), device="cuda") * 20
    free = torch.rand((n, p), device="cuda") > 0.4
    ref = reference(points, robot, free)
    got = candidates(points, robot, free)
    parity = [torch.equal(a, b) for a, b in zip(got, ref)]
    if not all(parity):
        raise AssertionError(f"output parity mismatch: {parity}")
    for _ in range(5):
        reference(points, robot, free)
        candidates(points, robot, free)
    torch.cuda.synchronize()
    baseline = measure(lambda: reference(points, robot, free), args.repeat)
    candidate = measure(lambda: candidates(points, robot, free), args.repeat)
    report = {"device": torch.cuda.get_device_name(), "shape": [n, p],
              "exact_parity": parity, "baseline": baseline, "prototype": candidate,
              "speedup": baseline["median_ms_per_call"] / candidate["median_ms_per_call"]}
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(serialized + "\n")


if __name__ == "__main__":
    main()
