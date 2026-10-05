#!/usr/bin/env python3
"""Parity and timing harness for an isolated full-contact HIP prototype."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from torch.utils import cpp_extension

# This harness supplies its own candidate contact dispatch, and its Torch side
# must remain the reference even if the production HIP extension is installed.
os.environ["AUTODRIVE_FUSED_COLLISION_HIP"] = "0"
os.environ["AUTODRIVE_FUSED_PATH_REFERENCE_HIP"] = "0"

from frc_defense.field import rebuilt_field, static_collision_boxes
from frc_defense.tensor_sim import TensorDefenseEnv, TensorVectorizedSimulator


def load_extension():
    bundled = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
    rocm = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
                (bundled if bundled.exists() else cpp_extension._find_rocm_home())).resolve()
    rocm_alias = rocm
    if " " in str(rocm):
        rocm_alias = Path("/tmp/autodrive-rocm-sdk")
        if rocm_alias.is_symlink() and rocm_alias.resolve() != rocm:
            rocm_alias.unlink()
        if not rocm_alias.exists():
            rocm_alias.symlink_to(rocm, target_is_directory=True)
        os.environ.setdefault("ROCM_HOME", str(rocm_alias))
        os.environ.setdefault("HIP_CLANG_PATH", str(rocm_alias / "lib/llvm/bin"))
    torch_lib = Path(torch.__file__).parent / "lib"
    if " " in str(torch_lib.resolve()):
        lib_alias = Path("/tmp/autodrive-torch-lib")
        if lib_alias.is_symlink() and lib_alias.resolve() != torch_lib.resolve():
            lib_alias.unlink()
        if not lib_alias.exists():
            lib_alias.symlink_to(torch_lib.resolve(), target_is_directory=True)
        torch_lib = lib_alias
    old = (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME)
    cpp_extension.TORCH_LIB_PATH = str(torch_lib)
    cpp_extension.ROCM_HOME = str(rocm_alias)
    cpp_extension.HIP_HOME = str(rocm_alias / "hip")
    src = Path(__file__).resolve().parents[1] / "frc_defense" / "csrc"
    device_lib = rocm_alias / "lib/llvm/amdgcn/bitcode"
    try:
        return cpp_extension.load(
            name="autodrive_tensor_collision_pipeline_proto",
            sources=[str(src / "tensor_collision_pipeline_proto.cpp"),
                     str(src / "tensor_collision_pipeline_proto_kernel.cu")],
            with_cuda=True, extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-ffp-contract=off",
                               f"--rocm-device-lib-path={device_lib}"], verbose=False)
    finally:
        cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = old


def call_fused(ext, sim, active, stage=-1, iterations=None):
    ext.contact_pipeline(sim.pose, sim.velocity, sim.length, sim.width, sim.mass,
        sim.yaw_inertia_multiplier, sim.mu, sim.wall_mu, active,
        sim.robot_contact, sim.opponent_contact, sim.field_contact,
        sim.wall_contact, sim.obstacles, sim.field_colliders,
        sim.contact_iterations if iterations is None else iterations,
        sim.field_length, sim.field_width, stage)


def call_torch(sim, active):
    for _ in range(sim.contact_iterations):
        sim._robot_collision(active)
        sim._walls_torch(active)
        sim._obstacle_collision(active)
        sim._field_collision(active)


def state_tuple(sim):
    return (sim.pose.clone(), sim.velocity.clone(), sim.robot_contact.clone(),
            sim.opponent_contact.clone(), sim.field_contact.clone(), sim.wall_contact.clone())


def state_compare(lhs, rhs):
    return {name: {"equal": bool(torch.equal(a, b)),
                   "max_abs": float((a - b).abs().max().item()) if a.dtype != torch.bool and a.numel() else 0.0,
                   "mismatches": int((a != b).sum().item())}
            for name, a, b in zip(("pose", "velocity", "robot_contact", "opponent_contact",
                                   "field_contact", "wall_contact"), lhs, rhs)}


def create_sim(n, device, seed, obstacle_count):
    boxes = [box.as_tensor() for box in static_collision_boxes(rebuilt_field())]
    obstacles = [[2.0 + 0.7 * j, 2.3 + 0.4 * (j % 4), 0.23 + 0.02 * (j % 3)]
                 for j in range(obstacle_count)]
    sim = TensorVectorizedSimulator(num_envs=n, device=device, seed=seed,
                                    field_colliders=boxes, obstacles=obstacles,
                                    contact_iterations=3)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    pose = torch.empty_like(sim.pose)
    pose[:, :, 0].uniform_(-0.35, sim.field_length + 0.35, generator=gen)
    pose[:, :, 1].uniform_(-0.35, sim.field_width + 0.35, generator=gen)
    pose[:, :, 2].uniform_(-math.pi, math.pi, generator=gen)
    velocity = torch.empty_like(sim.velocity).uniform_(-3.0, 3.0, generator=gen)
    # Deliberately include overlapping robots, exact edge contacts, corners,
    # and rotations that tie SAT axes, in addition to randomized worlds.
    pose[0, 0] = torch.tensor([5.0, 4.0, 0.0], device=device)
    pose[0, 1] = torch.tensor([5.7, 4.0, 0.0], device=device)
    pose[1, 0] = torch.tensor([5.0, 4.0, math.pi / 4], device=device)
    pose[1, 1] = torch.tensor([5.0, 4.0, math.pi / 4], device=device)
    pose[2, 0] = torch.tensor([0.45, 1.0, 0.0], device=device)
    pose[2, 1] = torch.tensor([1.35, 1.0, 0.0], device=device)
    pose[3, 0] = torch.tensor([-0.3, -0.3, 0.0], device=device)
    pose[3, 1] = torch.tensor([sim.field_length + .3, sim.field_width + .3, 0.0], device=device)
    velocity[0, 0, :2] = torch.tensor([1.0, .2], device=device)
    velocity[0, 1, :2] = torch.tensor([-1.0, -.2], device=device)
    sim.pose.copy_(pose)
    sim.velocity.copy_(velocity)
    active = torch.arange(n, device=device).remainder(5).ne(0)
    return sim, active


def run_contact_ab(ext, env_count, device, seed, repeats, obstacle_count):
    sim, active = create_sim(env_count, device, seed, obstacle_count)
    source = (sim.pose.clone(), sim.velocity.clone(), sim.robot_contact.clone(),
              sim.opponent_contact.clone(), sim.field_contact.clone(), sim.wall_contact.clone())

    def restore():
        for dst, src in zip((sim.pose, sim.velocity, sim.robot_contact,
                             sim.opponent_contact, sim.field_contact, sim.wall_contact), source):
            dst.copy_(src)

    restore(); call_torch(sim, active); expected = state_tuple(sim)
    restore(); call_fused(ext, sim, active); actual = state_tuple(sim)
    parity = state_compare(expected, actual)

    def timed(fn):
        samples = []
        for _ in range(repeats):
            restore()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); end.synchronize()
            samples.append(start.elapsed_time(end) / 1000.)
        return statistics.median(samples), min(samples), max(samples)

    for _ in range(3):
        restore(); call_torch(sim, active)
        restore(); call_fused(ext, sim, active)
    a1 = timed(lambda: call_torch(sim, active))
    b = timed(lambda: call_fused(ext, sim, active))
    a2 = timed(lambda: call_torch(sim, active))
    return {"worlds": env_count, "obstacles": obstacle_count,
            "torch_a_median_min_max": a1,
            "fused_b_median_min_max": b,
            "torch_a_repeat_median_min_max": a2,
            "a_repeat_ratio": a2[0] / a1[0],
            "speedup": ((a1[0] + a2[0]) * 0.5) / b[0],
            "parity": parity}


def run_end_to_end_ab_a(ext, args):
    # Reuse the project benchmark's complete TensorDefenseEnv + held-action
    # path. Monkeypatch only the four contact callbacks in the candidate run;
    # constructor reset remains on the original Torch implementation.
    script = Path(__file__).with_name("benchmark_tensor_scaling.py")
    spec = importlib.util.spec_from_file_location("tensor_scaling_benchmark", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = TensorVectorizedSimulator
    original_init = TensorDefenseEnv.__init__
    use_proto = {"value": False}

    def init_with_dispatch(self, *a, **kw):
        original_init(self, *a, **kw)
        if not use_proto["value"]:
            return
        sim = self.sim
        count = {"iteration": 0}
        def fused_robot(mask=None):
            if count["iteration"] == 0:
                call_fused(ext, sim, mask)
            count["iteration"] = (count["iteration"] + 1) % sim.contact_iterations
        sim._robot_collision = fused_robot
        sim._walls = lambda _mask=None: None
        sim._obstacle_collision = lambda _mask=None: None
        sim._field_collision = lambda _mask=None: None

    TensorDefenseEnv.__init__ = init_with_dispatch
    try:
        def benchmark(label, fused):
            use_proto["value"] = fused
            return mod.benchmark(args.envs, torch.device(args.device), args.seed,
                args.decisions, args.physics_ticks, "guard", False, True, True,
                None, False, False, False, True)
        a1 = benchmark("torch_a", False)
        b = benchmark("fused_b", True)
        a2 = benchmark("torch_a_repeat", False)
    finally:
        TensorDefenseEnv.__init__ = original_init
    return {"torch_a": a1, "fused_b": b, "torch_a_repeat": a2,
            "a_repeat_ratio": a2["elapsed_seconds"] / a1["elapsed_seconds"],
            "speedup": (a1["elapsed_seconds"] + a2["elapsed_seconds"]) * .5 /
                       b["elapsed_seconds"],
            "all_hashes_equal": len({a1["final_state_sha256"], b["final_state_sha256"],
                                      a2["final_state_sha256"]}) == 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--decisions", type=int, default=4)
    p.add_argument("--physics-ticks", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=9017)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--obstacles", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("Prototype requires HIP")
    torch_device = torch.device(args.device)
    ext = load_extension()
    contact = run_contact_ab(ext, args.envs, torch_device, args.seed, args.repeats,
                              args.obstacles)
    e2e = run_end_to_end_ab_a(ext, args)
    report = {"device": args.device, "seed": args.seed,
              "contact_pipeline": contact, "end_to_end": e2e}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if any(not item["equal"] for item in contact["parity"].values()) or not e2e["all_hashes_equal"]:
        raise SystemExit("prototype parity failed; see report")


if __name__ == "__main__":
    # Both the project benchmark module and this harness import tensor_sim;
    # path-reference optimization stays off for a stable reference comparison.
    os.environ.setdefault("AUTODRIVE_FUSED_PATH_REFERENCE_HIP", "0")
    os.environ.setdefault("AUTODRIVE_FUSED_COLLISION_HIP", "0")
    main()
