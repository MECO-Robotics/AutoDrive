#!/usr/bin/env python3
"""Parity and isolated timing for the non-production AD* occupancy kernel."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from frc_defense.field import INCH, rebuilt_field, static_collision_boxes
from frc_defense.tensor_adstar import TensorADStar, _hip_occupancy_extension


def load_extension():
    extension = _hip_occupancy_extension()
    if extension is None:
        raise RuntimeError("Fused AD* occupancy extension is unavailable or disabled")
    return extension


def make_planner(n: int, device: torch.device, circles: torch.Tensor,
                 fused_occupancy=False):
    field = rebuilt_field()
    length, width = 651.22 * INCH, 317.7 * INCH
    sim = SimpleNamespace(
        pose=torch.zeros((n, 2, 3), dtype=torch.float32, device=device),
        field_length=length, field_width=width,
        field_colliders=[box.as_tensor() for box in static_collision_boxes(field)],
        obstacles=circles,
    )
    env = SimpleNamespace(device=device, n=n, sim=sim, field_boxes=field)
    planner = TensorADStar(env, resolution=.3, avoid_bumps=True)
    planner._fused_blocked_hip_enabled = fused_occupancy
    return planner


def fused(ext, planner, heading, robot_length, robot_width, *, dynamic=None,
          avoid_bumps=True):
    if dynamic is None:
        dynamic = torch.empty((0, 4), dtype=torch.float32, device=heading.device)
    clearance = 0. if avoid_bumps else planner.resolution * .72
    return ext.blocked_grid(heading.contiguous(), robot_length.contiguous(),
        robot_width.contiguous(), planner.x.contiguous(), planner.y.contiguous(),
        planner._boxes.contiguous(), planner._bumps.contiguous(),
        planner._extra_circles.contiguous(), dynamic.contiguous(),
        planner.length, planner.width, clearance, avoid_bumps)


def parity_case(ext, n, device, seed, *, avoid_bumps, with_circles,
                with_dynamic, edge_case=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    circles = (torch.tensor([[3.05, 2.45, .37], [8.4, 4.03, .61]],
                            dtype=torch.float32, device=device) if with_circles else
               torch.empty((0, 3), dtype=torch.float32, device=device))
    planner = make_planner(n, device, circles)
    planner.avoid_bumps = avoid_bumps
    heading = torch.rand((n,), generator=generator, device=device) * (2 * math.pi) - math.pi
    robot_length = torch.rand((n,), generator=generator, device=device) * .45 + .65
    robot_width = torch.rand((n,), generator=generator, device=device) * .35 + .55
    if edge_case:
        fixed = torch.tensor([0., math.pi / 4, -math.pi / 4, math.pi / 2,
                              -math.pi / 2, math.pi, .2, -.2], device=device)
        heading[:min(n, fixed.numel())] = fixed[:n]
        # Add deliberately aligned shapes so >= versus > decisions are covered.
        # At (x[12], y[12]), world 2 touches the oriented box at both SAT limits.
        target_x, target_y = planner.x[12].item(), planner.y[12].item()
        planner._boxes = torch.cat((planner._boxes, torch.tensor(
            [[target_x + .60, target_y + .50, .15, .10]], device=device)), 0)
        planner._bumps = torch.tensor(
            [[target_x + .55, target_y, .10, .10]], dtype=torch.float32, device=device)
        # Circle and dynamic rectangle both meet the inflated x boundary exactly.
        planner._extra_circles = (torch.cat((planner._extra_circles,
            torch.tensor([[target_x + .65, target_y, .20]], dtype=torch.float32,
                         device=device)), 0) if with_circles else planner._extra_circles)
        if n >= 3:
            heading[2] = 0.; robot_length[2] = .9; robot_width[2] = 1.0
        if n >= 4:
            heading[3] = 0.; robot_length[3] = .8; robot_width[3] = .8
        if n >= 7:
            heading[6] = 0.; robot_length[6] = .9; robot_width[6] = .8
        # At the lower-left grid cell, the footprint exactly meets both walls;
        # a neighboring row just over the boundary must be blocked.
        if n >= 2:
            heading[:2] = 0.; robot_length[:2] = torch.tensor([.3, .30000007], device=device)
            robot_width[:2] = torch.tensor([.3, .3], device=device)
    dynamic = None
    if with_dynamic:
        dynamic = torch.zeros((n, 4), dtype=torch.float32, device=device)
        dynamic[:, 0] = torch.rand((n,), generator=generator, device=device) * planner.length
        dynamic[:, 1] = torch.rand((n,), generator=generator, device=device) * planner.width
        dynamic[:, 2:] = .45
        if edge_case and n >= 5:
            heading[4] = 0.; robot_length[4] = .9; robot_width[4] = .9
            dynamic[4] = torch.tensor([target_x + .9, target_y, .45, .35], device=device)
    reference = planner._blocked(heading, robot_length, robot_width,
                                  dynamic if with_dynamic else None)
    candidate = fused(ext, planner, heading, robot_length, robot_width,
                      dynamic=dynamic, avoid_bumps=avoid_bumps)
    mismatch = reference != candidate
    mismatch_count = int(mismatch.sum().item())
    first = None
    if mismatch_count:
        first = [int(v) for v in torch.nonzero(mismatch, as_tuple=False)[0].tolist()]
    return {"n": n, "avoid_bumps": avoid_bumps, "with_circles": with_circles,
            "with_dynamic": with_dynamic, "edge_case": edge_case,
            "mismatches": mismatch_count, "first_mismatch_nxy": first,
            "cells": int(reference.numel())}


def time_case(ext, n, device, seed, repeats):
    generator = torch.Generator(device=device).manual_seed(seed)
    circles = torch.tensor([[3.05, 2.45, .37], [8.4, 4.03, .61]],
                           dtype=torch.float32, device=device)
    planner = make_planner(n, device, circles)
    heading = torch.rand((n,), generator=generator, device=device) * (2 * math.pi) - math.pi
    robot_length = torch.full((n,), .9, dtype=torch.float32, device=device)
    robot_width = torch.full((n,), .75, dtype=torch.float32, device=device)
    dynamic = torch.rand((n, 4), generator=generator, device=device)
    dynamic[:, 0] *= planner.length
    dynamic[:, 1] *= planner.width
    dynamic[:, 2:] = .45
    args = (heading, robot_length, robot_width)
    kwargs = {"dynamic": dynamic, "avoid_bumps": True}
    for _ in range(2):
        planner._blocked(*args, dynamic)
        fused(ext, planner, *args, **kwargs)
    torch.cuda.synchronize(device)

    def measure(fn):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize(device)
        return [s.elapsed_time(e) / 1000. for s, e in zip(starts, ends)]

    reference = measure(lambda: planner._blocked(*args, dynamic))
    candidate = measure(lambda: fused(ext, planner, *args, **kwargs))
    return {"n": n, "grid": [planner.nx, planner.ny], "repeats": repeats,
            "torch_seconds": reference, "hip_seconds": candidate,
            "median_speedup": sorted(reference)[len(reference)//2] /
                              sorted(candidate)[len(candidate)//2]}


def plan_inputs(n, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    field_length, field_width = 651.22 * INCH, 317.7 * INCH
    start = torch.empty((n, 2), device=device).uniform_(.75, 1.5, generator=generator)
    start[:, 0] += torch.rand((n,), generator=generator, device=device) * 1.2
    goal = torch.empty_like(start)
    goal[:, 0] = field_length - start[:, 0]
    goal[:, 1] = field_width - start[:, 1]
    heading = torch.rand((n,), generator=generator, device=device) * (2 * math.pi) - math.pi
    length = torch.full((n,), .9, dtype=torch.float32, device=device)
    width = torch.full((n,), .75, dtype=torch.float32, device=device)
    speed = torch.full((n,), 4.5, dtype=torch.float32, device=device)
    defender = torch.rand((n, 2), generator=generator, device=device)
    defender[:, 0] *= field_length
    defender[:, 1] *= field_width
    defender_velocity = torch.rand((n, 2), generator=generator, device=device) * 2 - 1
    friction = torch.full((n,), 1.2, dtype=torch.float32, device=device)
    acceleration = torch.full((n,), 8., dtype=torch.float32, device=device)
    current_position = start + torch.tensor([.2, -.1], device=device)
    velocity = torch.rand((n, 2), generator=generator, device=device) * 3 - 1.5
    return (start, goal, heading, length, width, speed, defender,
            defender_velocity, friction, acceleration, current_position, velocity)


def plan_once(planner, inputs):
    (start, goal, heading, length, width, speed, defender,
     defender_velocity, friction, acceleration, current_position, velocity) = inputs
    planner.plan(start, goal, heading, length, width, speed, defender,
                 defender_velocity, True, friction, acceleration)
    command, tangent = planner.path_reference(current_position, velocity, speed)
    return {"potential": planner.potential, "path": planner.last_path,
            "lengths": planner.last_lengths, "intercept": planner.last_intercept,
            "intercept_time": planner.last_intercept_time, "start": planner.last_start,
            "goal": planner.last_goal, "heading": planner.last_heading,
            "speed_profile": planner.last_speed_profile,
            "command": command, "tangent": tangent}


def compare_planner_outputs(reference, candidate):
    result = {}
    for key in reference:
        lhs, rhs = reference[key], candidate[key]
        equal = torch.equal(lhs, rhs)
        mismatch_count = int((lhs != rhs).sum().item())
        if lhs.numel() and lhs.dtype != torch.bool:
            finite = torch.isfinite(lhs) & torch.isfinite(rhs)
            max_abs = (float((lhs[finite] - rhs[finite]).abs().max().item())
                       if bool(finite.any().item()) else 0.)
        else:
            max_abs = 0.
        result[key] = {"exact": equal, "mismatches": mismatch_count,
                       "max_abs": max_abs}
    return result


def full_plan_case(ext, n, device, seed, repeats):
    circles = torch.tensor([[3.05, 2.45, .37], [8.4, 4.03, .61]],
                           dtype=torch.float32, device=device)
    inputs = plan_inputs(n, device, seed)
    torch_a = make_planner(n, device, circles, fused_occupancy=False)
    hip_b = make_planner(n, device, circles, fused_occupancy=True)
    torch_a_repeat = make_planner(n, device, circles, fused_occupancy=False)
    torch_outputs = plan_once(torch_a, inputs)
    hip_outputs = plan_once(hip_b, inputs)
    parity = compare_planner_outputs(torch_outputs, hip_outputs)
    if not all(v["exact"] for v in parity.values()):
        return {"n": n, "parity": parity, "all_plan_outputs_exact": False,
                "timings": None}

    torch.cuda.synchronize(device)
    for planner in (torch_a, hip_b, torch_a_repeat):
        plan_once(planner, inputs)
    torch.cuda.synchronize(device)

    def measure(planner):
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            plan_once(planner, inputs)
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
        return samples

    a1 = measure(torch_a)
    b = measure(hip_b)
    a2 = measure(torch_a_repeat)
    a_median = (sorted(a1)[len(a1)//2] + sorted(a2)[len(a2)//2]) / 2
    return {"n": n, "grid": [torch_a.nx, torch_a.ny], "repeats": repeats,
            "parity": parity, "all_plan_outputs_exact": True,
            "timings": {"torch_a_seconds": a1, "hip_b_seconds": b,
                        "torch_a_repeat_seconds": a2,
                        "torch_baseline_median_seconds": a_median,
                        "hip_median_seconds": sorted(b)[len(b)//2],
                        "median_speedup_vs_mean_a": a_median / sorted(b)[len(b)//2],
                        "baseline_repeat_ratio": sorted(a1)[len(a1)//2] /
                                                 sorted(a2)[len(a2)//2]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--bench-sizes", default="512,2048")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", default="evaluations/strategic-ppo-validation-20261001/adstar-occupancy-proto/report.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("HIP accelerator required")
    ext = load_extension()
    parity = []
    for n, edge_case in ((32, True), (257, False)):
        for avoid_bumps in (False, True):
            for with_circles in (False, True):
                for with_dynamic in (False, True):
                    parity.append(parity_case(ext, n, device, 8831 + n,
                        avoid_bumps=avoid_bumps, with_circles=with_circles,
                        with_dynamic=with_dynamic, edge_case=edge_case))
    if any(case["mismatches"] for case in parity):
        report = {"parity": parity, "timings": [], "all_parity_exact": False}
    else:
        sizes = [int(x) for x in args.bench_sizes.split(",") if x]
        occupancy_timings = [time_case(ext, n, device, 9137 + n, args.repeats) for n in sizes]
        full_plans = [full_plan_case(ext, n, device, 10391 + n, args.repeats)
                      for n in sizes]
        report = {"parity": parity, "all_parity_exact": True,
                  "occupancy_timings": occupancy_timings,
                  "full_plan_ab_a": full_plans,
                  "all_full_plan_outputs_exact": all(c["all_plan_outputs_exact"]
                                                       for c in full_plans),
                  "production_wired": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_parity_exact"]:
        raise SystemExit(2)
    if not report["all_full_plan_outputs_exact"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
