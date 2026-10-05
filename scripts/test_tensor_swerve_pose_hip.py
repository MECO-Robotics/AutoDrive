#!/usr/bin/env python3
"""Experimental parity gate for the isolated fused swerve + pose HIP kernel.

This script is intentionally separate from training and production tests. It
builds the experimental extension only when explicitly run on a HIP device.
`--benchmark` runs A/B/A timing only after every parity case passes and at
least 2 GiB of device memory is free.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys

import torch

from frc_defense.tensor_sim import TensorVectorizedSimulator
from frc_defense.field import (BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE,
                               BUMP_ROBOT_CG_HEIGHT, BUMP_ROLLING_RESISTANCE)
from frc_defense.tensor_swerve_pose_hip import step as fused_step


STATE_NAMES = (
    "module_steer_rate", "module_angle", "module_current",
    "module_supply_current", "robot_current", "velocity", "pose",
    "module_drive_speed",
)
DEBUG_FIELDS = (
    "target", "actual", "motor_target", "duty_term1", "duty_term2_base",
    "duty_term2", "duty", "bus", "numerator", "amps_unlimited",
    "amps_current_limited", "supply_pre_controller", "controller_scale",
    "amps_post_controller", "supply_post_controller", "steer_current",
    "supply_sum", "steer_sum", "drive_budget", "cap", "amps_final",
    "supply_final",
)


def _copy_state(source, target):
    for name in STATE_NAMES:
        getattr(target, name).copy_(getattr(source, name))


def _snapshot(sim):
    return {name: getattr(sim, name).clone() for name in STATE_NAMES}


def _restore(sim, state):
    for name, tensor in state.items():
        getattr(sim, name).copy_(tensor)


def _reference_step(sim, command, active):
    command = torch.nan_to_num(command)
    sim._swerve_eager(command, active)
    sim.pose.copy_(torch.where(active[:, None, None],
        sim.pose + sim.velocity * sim.dt, sim.pose))
    wrapped = torch.remainder(sim.pose[..., 2] + math.pi, 2 * math.pi) - math.pi
    sim.pose[..., 2].copy_(torch.where(active[:, None], wrapped, sim.pose[..., 2]))


def _torch_current_debug(sim, command):
    """Return the reference Torch intermediates for the motor/current chain."""
    p = sim.swerve
    theta = sim.pose[..., 2]
    c, s = theta.cos(), theta.sin()
    mx, my = sim._module_x_offsets, sim._module_y_offsets
    speed_limit = sim.speed
    vx = c * command[..., 0] + s * command[..., 1]
    vy = -s * command[..., 0] + c * command[..., 1]
    norm = torch.sqrt(vx.square() + vy.square()).clamp_min(1.0e-8)
    scale = (speed_limit / norm).clamp(max=1)
    vx, vy = vx * scale, vy * scale
    omega = command[..., 2].clamp(-sim.omega_limit, sim.omega_limit)
    dx = vx[..., None] - omega[..., None] * my
    dy = vy[..., None] + omega[..., None] * mx
    target = torch.sqrt(dx.square() + dy.square())
    target *= (speed_limit[..., None] /
        target.amax(-1, keepdim=True).clamp_min(1.0e-8)).clamp(max=1)
    angle = torch.atan2(dy, dx)
    delta = torch.remainder(angle - sim.module_angle + math.pi, 2 * math.pi) - math.pi
    reverse = delta.abs() > math.pi / 2
    angle = torch.where(reverse, angle + math.pi, angle)
    target = torch.where(reverse, -target, target)
    angle = torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
    delta = torch.remainder(angle - sim.module_angle + math.pi, 2 * math.pi) - math.pi
    steer_target = (8 * delta).clamp(-p.steer_rate_limit, p.steer_rate_limit)
    se = steer_target - sim.module_steer_rate
    steer_current = (2 * se.abs()).clamp(max=p.steer_current_limit)

    resistance = p.nominal_voltage / p.stall_current_a
    stall = p.free_speed_rpm * 2 * math.pi / 60
    kv = stall / (p.nominal_voltage - resistance * p.free_current_a)
    actual = sim.module_drive_speed / p.wheel_radius * sim.drive_ratio[..., None]
    motor_target = target / p.wheel_radius * sim.drive_ratio[..., None]
    duty_term1 = motor_target / stall
    duty_term2_base = 0.8 * (motor_target - actual)
    duty_term2 = duty_term2_base / stall
    duty = (duty_term1 + duty_term2).clamp(-1, 1)
    bus = (p.battery_voltage - sim.robot_current * sim.battery_resistance).clamp_min(0)
    numerator = duty * bus[..., None] - actual / kv
    amps_unlimited = numerator / resistance
    amps_limited = amps_unlimited.clamp(-sim.drive_current_limit[..., None],
                                        sim.drive_current_limit[..., None])
    supply_pre = (duty * amps_limited).abs()
    controller_scale = (sim.drive_supply_limit[..., None] /
        supply_pre.clamp_min(1.0e-6)).clamp(max=1)
    amps_post = amps_limited * controller_scale
    supply_post = (duty * amps_post).abs()
    supply_sum = supply_post.sum(-1, keepdim=True).expand_as(supply_post)
    steer_sum = steer_current.sum(-1, keepdim=True).expand_as(steer_current)
    drive_budget = (sim.robot_supply_limit - steer_current.sum(-1)).clamp_min(0)
    cap = (drive_budget / supply_post.sum(-1).clamp_min(1.0e-6)).clamp(max=1)
    amps_final = amps_post * cap[..., None]
    supply_final = supply_post * cap[..., None]
    values = (target, actual, motor_target, duty_term1, duty_term2_base,
              duty_term2, duty,
              bus[..., None].expand_as(target), numerator,
              amps_unlimited, amps_limited, supply_pre, controller_scale,
              amps_post, supply_post, steer_current, supply_sum, steer_sum,
              drive_budget[..., None].expand_as(target), cap[..., None].expand_as(target),
              amps_final, supply_final)
    return torch.stack(values, dim=-1)


def _torch_bump_debug(sim):
    """Return bump-dependent inputs to the velocity update."""
    theta = sim.pose[..., 2]
    c, s = theta.cos(), theta.sin()
    mx, my = sim._module_x_offsets, sim._module_y_offsets
    wheel_x = sim.pose[..., 0, None] + c[..., None] * mx - s[..., None] * my
    wheel_y = sim.pose[..., 1, None] + s[..., None] * mx + c[..., None] * my
    wheel_grade = torch.zeros_like(wheel_x)
    wheel_height = torch.zeros_like(wheel_x)
    on_bump = torch.zeros_like(wheel_x, dtype=torch.bool)
    if sim.bump_regions.shape[0]:
        dx = wheel_x[..., None] - sim.bump_regions[None, None, None, :, 0]
        dy = wheel_y[..., None] - sim.bump_regions[None, None, None, :, 1]
        half_x = sim.bump_regions[None, None, None, :, 2].clamp_min(1e-6)
        within = ((dx.abs() <= half_x) &
                  (dy.abs() <= sim.bump_regions[None, None, None, :, 3]))
        heights = torch.where(within, BUMP_PANEL_THICKNESS + BUMP_RAMP_RISE *
                              (1 - dx.abs() / half_x), 0.)
        grades = torch.where(within, -dx.sign() * BUMP_RAMP_RISE / half_x, 0.)
        selected = heights.argmax(-1, keepdim=True)
        wheel_height = heights.gather(-1, selected).squeeze(-1)
        wheel_grade = grades.gather(-1, selected).squeeze(-1)
        on_bump = within.any(-1)
    front_height = wheel_height[..., 1::2].mean(-1)
    rear_height = wheel_height[..., 0::2].mean(-1)
    pitch = torch.atan2(front_height - rear_height, sim.length)
    normal = (sim.mass[..., None] * 9.81 * pitch[..., None].cos() / 4. -
        sim.mass[..., None] * 9.81 * BUMP_ROBOT_CG_HEIGHT *
        pitch[..., None].sin() * mx.sign() /
        (2 * sim.length[..., None].clamp_min(1e-6))).clamp_min(0.)
    mean_grade = wheel_grade.mean(-1, keepdim=True).expand_as(wheel_grade)
    rolling = (BUMP_ROLLING_RESISTANCE * normal * on_bump).sum(-1) / sim.mass
    rolling = rolling[..., None].expand_as(wheel_grade)
    return torch.stack((wheel_grade, normal, mean_grade, rolling), -1)


def _check_bump_debug(expected, actual, active, label):
    expected, actual = expected[active], actual[active]
    unequal = actual != expected
    if bool(unequal.any()):
        idx = tuple(torch.nonzero(unequal)[0].detach().cpu().tolist())
        names = ("wheel_grade", "normal", "mean_grade", "rolling_accel")
        field = names[idx[-1]]
        raise AssertionError(
            f"{label}: first bump-chain divergence at {field}, index {idx[:-1]}; "
            f"Torch={expected[idx].item():.9g}, HIP={actual[idx].item():.9g}")


def _check_debug(expected, actual, active, label):
    expected = expected[active]
    actual = actual[active]
    unequal = actual != expected
    if not bool(unequal.any()):
        return
    index = torch.nonzero(unequal)[0].detach().cpu().tolist()
    field_index = index[-1]
    field = DEBUG_FIELDS[field_index]
    idx = tuple(index[:-1])
    reduction_trace = ""
    if field == "steer_sum":
        vals = expected[idx[0], idx[1], :, DEBUG_FIELDS.index("steer_current")].detach().cpu().tolist()
        f32 = lambda x: struct.unpack("f", struct.pack("f", float(x)))[0]
        seq = f32(f32(f32(vals[0] + vals[1]) + vals[2]) + vals[3])
        pair = f32(f32(vals[0] + vals[1]) + f32(vals[2] + vals[3]))
        alternate = f32(f32(vals[0] + vals[2]) + f32(vals[1] + vals[3]))
        reduction_trace = (f"; module currents={vals}; sequential={seq:.9g}, "
                           f"pairwise01_23={pair:.9g}, pairwise02_13={alternate:.9g}")
    raise AssertionError(
        f"{label}: first current-chain divergence at {field}, module row {idx}; "
        f"Torch={expected[idx + (field_index,)].item():.9g}, "
        f"HIP={actual[idx + (field_index,)].item():.9g}{reduction_trace}")


def _new_pair(worlds, device, bump_regions):
    reference = TensorVectorizedSimulator(num_envs=worlds, device=device,
        seed=83011, randomize=False, bump_regions=bump_regions)
    candidate = TensorVectorizedSimulator(num_envs=worlds, device=device,
        seed=83011, randomize=False, bump_regions=bump_regions)
    gen = torch.Generator(device=device).manual_seed(119)
    reference.pose[..., 0].uniform_(1.0, reference.field_length - 1.0, generator=gen)
    reference.pose[..., 1].uniform_(1.0, reference.field_width - 1.0, generator=gen)
    reference.pose[..., 2].uniform_(-math.pi, math.pi, generator=gen)
    reference.velocity.uniform_(-2.0, 2.0, generator=gen)
    reference.module_angle.uniform_(-math.pi, math.pi, generator=gen)
    reference.module_steer_rate.uniform_(-reference.swerve.steer_rate_limit,
                                         reference.swerve.steer_rate_limit, generator=gen)
    reference.module_drive_speed.uniform_(-3.0, 3.0, generator=gen)
    reference.module_current.uniform_(-60.0, 60.0, generator=gen)
    reference.module_supply_current.uniform_(0.0, 80.0, generator=gen)
    reference.robot_current.uniform_(0.0, 180.0, generator=gen)
    reference.mass.uniform_(48.0, 64.0, generator=gen)
    reference.length.uniform_(0.82, 1.0, generator=gen)
    reference.width.uniform_(0.82, 1.0, generator=gen)
    reference.speed.uniform_(3.5, 5.0, generator=gen)
    reference.accel.uniform_(6.0, 10.0, generator=gen)
    reference.omega_limit.uniform_(6.0, 9.0, generator=gen)
    reference.alpha.uniform_(14.0, 22.0, generator=gen)
    reference.ground_mu.uniform_(0.5, 1.0, generator=gen)
    reference.lateral_mu.uniform_(0.8, 1.5, generator=gen)
    reference.drive_ratio.uniform_(6.0, 7.2, generator=gen)
    reference.drive_current_limit.uniform_(48.0, 65.0, generator=gen)
    reference.drive_supply_limit.uniform_(60.0, 90.0, generator=gen)
    reference.robot_supply_limit.uniform_(140.0, 190.0, generator=gen)
    reference.battery_resistance.uniform_(0.012, 0.025, generator=gen)
    reference.module_angle[0, 0].zero_()
    reference.module_steer_rate[0, 0].zero_()
    if bump_regions:
        bump_x, bump_y, half_x, half_y = bump_regions[0]
        reference.pose[0, 0] = torch.tensor(
            [bump_x, bump_y, 0.0], device=device)
        if worlds > 1:
            reference.pose[1, 0] = torch.tensor(
                [bump_x - half_x + 0.36, bump_y + half_y - 0.36, 0.0],
                device=device)
    _copy_state(reference, candidate)
    for name in ("mass", "length", "width", "speed", "accel", "omega_limit",
                 "alpha", "ground_mu", "lateral_mu", "drive_ratio",
                 "drive_current_limit", "drive_supply_limit", "robot_supply_limit",
                 "battery_resistance"):
        getattr(candidate, name).copy_(getattr(reference, name))
    return reference, candidate, gen


def _commands(sim, gen, tick):
    command = torch.empty_like(sim.velocity)
    command.uniform_(-1.0, 1.0, generator=gen)
    # Include opposite-direction and exact steering threshold cases as well as
    # controller saturation and zero-command cases.
    if tick % 5 == 0:
        command[:, :, 0] = 0.0
        command[:, :, 1] = 1.0
    elif tick % 5 == 1:
        command[:, :, 0] = 0.0
        command[:, :, 1] = -1.0
    elif tick % 5 == 2:
        command[:, :, :2] = 0.0
        command[:, :, 2] = 0.0
    elif tick % 5 == 3:
        command[:, :, :] = torch.tensor([1.4, -1.3, 9.0], device=sim.device)
    return command


def _check_equal(reference, candidate, label):
    for name in STATE_NAMES:
        expected = getattr(reference, name)
        actual = getattr(candidate, name)
        if not torch.equal(actual, expected):
            diff = (actual - expected).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            unequal = int(torch.count_nonzero(actual != expected).item())
            indices = torch.nonzero(actual != expected)[:8].detach().cpu().tolist()
            details = [(idx, float(expected[tuple(idx)].item()),
                        float(actual[tuple(idx)].item())) for idx in indices]
            raise AssertionError(
                f"{label}: {name} differs in {unequal}/{actual.numel()} values; "
                f"max_abs={max_abs:.9g}; first index/expected/actual={details}")


def _run_parity_case(label, bump_regions, active_pattern, worlds=64, ticks=24):
    device = torch.device("cuda:0")
    reference, candidate, gen = _new_pair(worlds, device, bump_regions)
    no_debug_candidate, _, _ = _new_pair(worlds, device, bump_regions)
    for tick in range(ticks):
        if active_pattern == "all":
            active = torch.ones(worlds, dtype=torch.bool, device=device)
        elif active_pattern == "none":
            active = torch.zeros(worlds, dtype=torch.bool, device=device)
        else:
            active = torch.arange(worlds, device=device).remainder(3).ne(tick % 3)
        command = _commands(reference, gen, tick)
        before_candidate = _snapshot(candidate)
        expected_debug = _torch_current_debug(reference, torch.nan_to_num(command))
        expected_bump = _torch_bump_debug(reference)
        _reference_step(reference, command, active)
        actual_debug = fused_step(candidate, command, active)
        no_debug_result = fused_step(no_debug_candidate, command, active,
                                     debug=False)
        if no_debug_result is not None:
            raise AssertionError("production no-debug path returned debug storage")
        _check_debug(expected_debug, actual_debug[..., :22], active,
                     f"{label}/tick-{tick}")
        _check_bump_debug(expected_bump, actual_debug[..., 22:], active,
                          f"{label}/tick-{tick}")
        _check_equal(reference, candidate, f"{label}/tick-{tick}")
        _check_equal(reference, no_debug_candidate,
                     f"{label}/no-debug-tick-{tick}")
        if active_pattern != "all":
            inactive = ~active
            for name, before in before_candidate.items():
                if not torch.equal(getattr(candidate, name)[inactive], before[inactive]):
                    raise AssertionError(f"{label}/tick-{tick}: inactive {name} mutated")
    return {"case": label, "worlds": worlds, "ticks": ticks,
            "active_pattern": active_pattern, "exact": True}


def _benchmark(worlds, iterations):
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 2 * 1024**3:
        raise RuntimeError(f"refusing benchmark: only {free_bytes / 1024**3:.2f} GiB free")
    empty_bumps = []
    reference, candidate, gen = _new_pair(worlds, torch.device("cuda:0"), empty_bumps)
    active = torch.arange(worlds, device="cuda:0").remainder(5).ne(2)
    commands = [_commands(reference, gen, i) for i in range(8)]

    def block(sim, fused):
        for i in range(iterations):
            command = commands[i % len(commands)]
            if fused:
                fused_step(sim, command, active)
            else:
                _reference_step(sim, command, active)

    initial_ref, initial_candidate = _snapshot(reference), _snapshot(candidate)
    def timed(sim, state, fused):
        _restore(sim, state)
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        block(sim, fused)
        end.record()
        end.synchronize()
        return float(begin.elapsed_time(end)) / 1000.0, _snapshot(sim)

    # Extension compilation and warm-up precede the measured A/B/A sequence.
    fused_step(candidate, commands[0], active)
    _restore(candidate, initial_candidate)
    block(reference, False)
    _restore(reference, initial_ref)
    a1, state_a1 = timed(reference, initial_ref, False)
    b, state_b = timed(candidate, initial_candidate, True)
    a2, state_a2 = timed(reference, initial_ref, False)
    for name in STATE_NAMES:
        if not (torch.equal(state_a1[name], state_b[name]) and
                torch.equal(state_a1[name], state_a2[name])):
            raise AssertionError(f"benchmark trajectory mismatch in {name}")
    return {"worlds": worlds, "iterations": iterations,
            "free_gib_before_benchmark": free_bytes / 1024**3,
            "total_gib": total_bytes / 1024**3,
            "torch_a_seconds": a1, "hip_b_seconds": b,
            "torch_a_repeat_seconds": a2,
            "speedup_vs_mean_torch": ((a1 + a2) * 0.5) / b}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-envs", type=int, default=2048)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        print(json.dumps({"status": "skipped", "reason": "requires HIP GPU"}))
        return 0
    free, total = torch.cuda.mem_get_info()
    if free < 2 * 1024**3:
        print(json.dumps({"status": "skipped", "reason": "less than 2 GiB free",
                          "free_gib": free / 1024**3,
                          "total_gib": total / 1024**3}))
        return 0
    # Empty terrain, the four nominal field bumps, and tied overlapping bumps.
    from frc_defense.field import rebuilt_field, bump_boxes
    standard = [box.as_tensor() for box in bump_boxes(rebuilt_field())]
    tied = [[8.0, 4.0, 1.0, 1.0], [8.0, 4.0, 1.0, 1.0]]
    cases = []
    for bumps, suffix in (([], "flat"), (standard, "field-bumps"),
                          (tied, "overlapping-bump-tie")):
        for active_pattern in ("all", "mixed", "none"):
            cases.append(_run_parity_case(
                f"{suffix}-{active_pattern}", bumps, active_pattern))
    output = {"status": "exact-parity-passed", "parity_cases": cases}
    if args.benchmark:
        cases.append(_run_parity_case("benchmark-flat-mixed",
            [], "mixed", worlds=args.benchmark_envs, ticks=24))
        output["benchmark"] = _benchmark(args.benchmark_envs, args.benchmark_iters)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
