#!/usr/bin/env python3
"""Isolated exact-parity/timing harness for the experimental HIP wall kernel."""
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils import cpp_extension

from frc_defense.field import rebuilt_field, static_collision_boxes
from frc_defense.tensor_sim import TensorVectorizedSimulator


def load_extension():
    sdk = Path(torch.__file__).parent / ".." / "_rocm_sdk_core"
    sdk = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
               (sdk if sdk.exists() else cpp_extension._find_rocm_home())).resolve()
    if " " in str(sdk):
        alias = Path("/tmp/autodrive-rocm-sdk")
        if alias.is_symlink() and alias.resolve() != sdk:
            alias.unlink()
        if not alias.exists():
            alias.symlink_to(sdk, target_is_directory=True)
        sdk = alias
        os.environ.setdefault("ROCM_HOME", str(sdk))
        os.environ.setdefault("HIP_CLANG_PATH", str(sdk / "lib/llvm/bin"))
    torch_lib = Path(torch.__file__).parent / "lib"
    original_lib, original_rocm, original_hip = (
        cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME)
    lib_alias = torch_lib.resolve()
    if " " in str(lib_alias):
        candidate = Path("/tmp/autodrive-torch-lib")
        if candidate.is_symlink() and candidate.resolve() != lib_alias:
            candidate.unlink()
        if not candidate.exists():
            candidate.symlink_to(lib_alias, target_is_directory=True)
        lib_alias = candidate
    cpp_extension.TORCH_LIB_PATH = str(lib_alias)
    cpp_extension.ROCM_HOME = str(sdk)
    cpp_extension.HIP_HOME = str(sdk / "hip")
    device_lib = sdk / "lib/llvm/amdgcn/bitcode"
    src = Path(__file__).resolve().parents[1] / "frc_defense" / "csrc"
    try:
        return cpp_extension.load(
            name="autodrive_tensor_collision_hip",
            sources=[str(src / "tensor_collision_hip.cpp"),
                     str(src / "tensor_collision_hip_kernel.cu")],
            with_cuda=True, extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-ffp-contract=off",
                               f"--rocm-device-lib-path={device_lib}"], verbose=False)
    finally:
        cpp_extension.TORCH_LIB_PATH = original_lib
        cpp_extension.ROCM_HOME = original_rocm
        cpp_extension.HIP_HOME = original_hip


def exact_compare(a, b):
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs": float((a - b).abs().max().item()) if a.numel() else 0.0,
        "mismatch_count": int((a != b).sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9017)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("This harness is for the AMD HIP training host")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    colliders = [box.as_tensor() for box in static_collision_boxes(rebuilt_field())]
    sim = TensorVectorizedSimulator(num_envs=args.envs, device=device, seed=args.seed,
                                    contact_iterations=3, field_colliders=colliders)
    # Seed wall-adjacent and interior robots. Deliberately retain a mix of
    # active and inactive worlds to check masked-row preservation.
    n = args.envs
    pose0 = torch.empty_like(sim.pose)
    pose0[:, :, 0].uniform_(-0.5, sim.field_length + 0.5, generator=sim.generator)
    pose0[:, :, 1].uniform_(-0.5, sim.field_width + 0.5, generator=sim.generator)
    pose0[:, :, 2].uniform_(-torch.pi, torch.pi, generator=sim.generator)
    vel0 = torch.empty_like(sim.velocity)
    vel0.uniform_(-3.0, 3.0, generator=sim.generator)
    active = torch.arange(n, device=device).remainder(5).ne(0)
    contacts0 = torch.arange(n * 4, device=device).reshape(n, 2, 2).remainder(7).eq(0)
    ext = load_extension()

    def restore():
        sim.pose.copy_(pose0)
        sim.velocity.copy_(vel0)
        sim.wall_contact.copy_(contacts0)

    restore()
    sim._walls_torch(active)
    ref_pose, ref_vel, ref_contacts = sim.pose.clone(), sim.velocity.clone(), sim.wall_contact.clone()
    restore()
    ext.walls(sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
              sim.yaw_inertia_multiplier, sim.wall_mu, active, sim.wall_contact,
              sim.field_length, sim.field_width)
    proto_pose, proto_vel, proto_contacts = sim.pose.clone(), sim.velocity.clone(), sim.wall_contact.clone()
    parity = {
        "pose": exact_compare(ref_pose, proto_pose),
        "velocity": exact_compare(ref_vel, proto_vel),
        "wall_contact": bool(torch.equal(ref_contacts, proto_contacts)),
        "active_rows_unchanged": bool(torch.equal(
            proto_pose[~active], pose0[~active]) and torch.equal(proto_vel[~active], vel0[~active])
            and torch.equal(proto_contacts[~active], contacts0[~active])),
    }

    # Verify exact tie and multi-wall cases as well as random wall locations.
    # Use a small deterministic prefix with exact axis-aligned boundary values.
    boundary_pose = pose0[:8].clone()
    boundary_vel = vel0[:8].clone()
    boundary_pose[:, :, 2] = 0.0
    boundary_pose[0, 0, 0] = sim.length[0, 0] * .5  # exact zero penetration
    boundary_pose[0, 1, 0] = sim.field_length - sim.length[0, 1] * .5
    boundary_pose[1, 0, 1] = sim.width[1, 0] * .5
    boundary_pose[1, 1, 1] = sim.field_width - sim.width[1, 1] * .5
    boundary_pose[2, 0, :2] = torch.tensor([.40, .40], device=device)
    boundary_pose[2, 1, :2] = torch.tensor([sim.field_length-.40, sim.field_width-.40], device=device)
    boundary_pose[3, 0, :2] = torch.tensor([-.5, -.5], device=device)
    boundary_pose[3, 1, :2] = torch.tensor([sim.field_length+.5, sim.field_width+.5], device=device)
    boundary_pose[4, :, 2] = torch.pi / 4
    boundary_pose[4, 0, :2] = torch.tensor([.40, .40], device=device)
    boundary_pose[4, 1, :2] = torch.tensor([sim.field_length-.40, sim.field_width-.40], device=device)
    bactive_prefix = torch.tensor([True, True, True, True, True, False, True, False],
                                  device=device)
    bactive = torch.zeros((n,), device=device, dtype=torch.bool)
    bactive[:8] = bactive_prefix
    bcontacts = contacts0[:8].clone()
    boundary_pose_full, boundary_vel_full, boundary_contacts_full = pose0.clone(), vel0.clone(), contacts0.clone()
    boundary_pose_full[:8] = boundary_pose
    boundary_vel_full[:8] = boundary_vel
    sim.pose.copy_(boundary_pose_full)
    sim.velocity.copy_(boundary_vel_full)
    sim.wall_contact.copy_(boundary_contacts_full)
    sim._walls_torch(bactive)
    b_ref_pose, b_ref_vel, b_ref_contacts = (sim.pose[:8].clone(), sim.velocity[:8].clone(),
                                              sim.wall_contact[:8].clone())
    sim.pose.copy_(boundary_pose_full)
    sim.velocity.copy_(boundary_vel_full)
    sim.wall_contact.copy_(boundary_contacts_full)
    ext.walls(sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
              sim.yaw_inertia_multiplier, sim.wall_mu, bactive, sim.wall_contact,
              sim.field_length, sim.field_width)
    b_pose_test, b_vel_test, b_contact_test = (sim.pose[:8].clone(), sim.velocity[:8].clone(),
                                                sim.wall_contact[:8].clone())
    boundary_parity = {
        "pose": exact_compare(b_ref_pose, b_pose_test),
        "velocity": exact_compare(b_ref_vel, b_vel_test),
        "wall_contact": bool(torch.equal(b_ref_contacts, b_contact_test)),
        "inactive_rows_unchanged": bool(torch.equal(b_pose_test[~bactive_prefix], boundary_pose[~bactive_prefix]) and
                                         torch.equal(b_vel_test[~bactive_prefix], boundary_vel[~bactive_prefix]) and
                                         torch.equal(b_contact_test[~bactive_prefix], bcontacts[~bactive_prefix])),
    }

    def collision_pipeline(use_proto):
        for _ in range(sim.contact_iterations):
            sim._robot_collision(active)
            if use_proto:
                ext.walls(sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
                          sim.yaw_inertia_multiplier, sim.wall_mu, active,
                          sim.wall_contact, sim.field_length, sim.field_width)
            else:
                sim._walls_torch(active)
            sim._obstacle_collision(active)
            sim._field_collision(active)

    def restore_pipeline():
        restore()
        sim.robot_contact.zero_()
        sim.opponent_contact.zero_()
        sim.field_contact.zero_()

    def timed(fn):
        samples = []
        for _ in range(args.repeats):
            restore()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1e-3)
        samples.sort()
        return {"median_seconds": samples[len(samples) // 2],
                "min_seconds": samples[0], "max_seconds": samples[-1]}

    # Alternate after warmup so neither path consistently gets first-run/cache effects.
    for _ in range(5):
        restore(); sim._walls_torch(active)
        restore(); ext.walls(sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
                             sim.yaw_inertia_multiplier, sim.wall_mu, active,
                             sim.wall_contact, sim.field_length, sim.field_width)
    baseline_a = timed(lambda: sim._walls_torch(active))
    prototype = timed(lambda: ext.walls(sim.pose, sim.velocity, sim.length, sim.width,
                                         sim.mass, sim.yaw_inertia_multiplier,
                                         sim.wall_mu, active, sim.wall_contact,
                                         sim.field_length, sim.field_width))
    baseline_a_repeat = timed(lambda: sim._walls_torch(active))
    baseline = (baseline_a["median_seconds"] + baseline_a_repeat["median_seconds"]) * 0.5
    result = {"device": args.device, "envs": n, "repeats": args.repeats,
              "seed": args.seed, "parity": parity,
              "baseline_walls_a": baseline_a,
              "prototype_walls_b": prototype,
              "baseline_walls_a_repeat": baseline_a_repeat,
              "baseline_repeat_ratio": baseline_a_repeat["median_seconds"] /
                  baseline_a["median_seconds"],
              "baseline_mean_seconds": baseline,
              "speedup_vs_mean_baseline": baseline / prototype["median_seconds"],
              "boundary_exact_parity": boundary_parity}

    # Measure the full ordered contact-iteration block, including robot SAT,
    # all four walls, optional circles, and the default static field boxes.
    restore_pipeline()
    collision_pipeline(False)
    whole_ref = (sim.pose.clone(), sim.velocity.clone(), sim.robot_contact.clone(),
                 sim.opponent_contact.clone(), sim.wall_contact.clone(), sim.field_contact.clone())
    restore_pipeline()
    collision_pipeline(True)
    whole_proto = (sim.pose.clone(), sim.velocity.clone(), sim.robot_contact.clone(),
                   sim.opponent_contact.clone(), sim.wall_contact.clone(), sim.field_contact.clone())
    result["ordered_contact_pipeline_parity"] = {
        name: (bool(torch.equal(a, b)) if a.dtype == torch.bool else exact_compare(a, b))
        for name, a, b in zip(("pose", "velocity", "robot_contact", "opponent_contact",
                               "wall_contact", "field_contact"), whole_ref, whole_proto)
    }

    def timed_pipeline(use_proto):
        samples = []
        for _ in range(args.repeats):
            restore_pipeline()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            collision_pipeline(use_proto)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1e-3)
        samples.sort()
        return {"median_seconds": samples[len(samples) // 2],
                "min_seconds": samples[0], "max_seconds": samples[-1]}

    # Contact block A/B/A with input restoration outside the timed region.
    result["contact_pipeline_a"] = timed_pipeline(False)
    result["contact_pipeline_b"] = timed_pipeline(True)
    result["contact_pipeline_a_repeat"] = timed_pipeline(False)
    result["contact_pipeline_a_repeat_ratio"] = (
        result["contact_pipeline_a_repeat"]["median_seconds"] /
        result["contact_pipeline_a"]["median_seconds"])
    result["contact_pipeline_speedup_vs_mean_a"] = (
        (result["contact_pipeline_a"]["median_seconds"] +
         result["contact_pipeline_a_repeat"]["median_seconds"]) * .5 /
        result["contact_pipeline_b"]["median_seconds"])
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
