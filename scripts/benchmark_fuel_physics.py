#!/usr/bin/env python3
"""Seeded fuel solver comparisons with synchronized timing and untimed quality metrics.

Run separate processes on cuda:0 and cuda:1; use distinct --output paths.
Unsupported implementations are reported as errors, never relabeled as HIP results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import hashlib
import gc
from pathlib import Path
import statistics
import time

import torch

from frc_defense.fuel_physics import FuelPhysics, FuelPhysicsConfig
from frc_defense.tensor_physics import TensorVectorizedSimulator

VARIANTS = {
    "robot_only": dict(backend="torch"),
    "allpairs": dict(broadphase="all_pairs", cache=False, robot_broadphase="all_pairs"),
    "grid": dict(broadphase="grid", cache=False),
    "cached": dict(broadphase="grid", cache=True),
    "compact": dict(broadphase="grid", cache=True, contact_mode="compact"),
    "colored": dict(broadphase="grid", cache=True, solver="colored"),
    "colored_torch": dict(broadphase="grid", cache=True, solver="colored", backend="torch"),
    "sleep": dict(broadphase="grid", cache=True, sleep=True),
    "graph": dict(broadphase="grid", cache=True),
    "skin025": dict(broadphase="grid", cache=True, skin=.025),
    "skin15": dict(broadphase="grid", cache=True, skin=.15),
    "skin30": dict(broadphase="grid", cache=True, skin=.3),
    "neighbors16": dict(broadphase="grid", cache=True, max_neighbors=16),
    "neighbors32": dict(broadphase="grid", cache=True, max_neighbors=32),
    "island_sleep": dict(broadphase="grid", cache=True, sleep=True, sleep_islands=True),
    "robot_allpairs": dict(broadphase="grid", cache=True, robot_broadphase="all_pairs"),
    "robot_grid": dict(broadphase="grid", cache=True, robot_broadphase="grid"),
    "compact_full": dict(broadphase="grid", cache=True, contact_mode="compact", compact_capacity="full"),
}


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def initialize(args, config, scene):
    """Every variant starts from identical CPU-generated positions and velocities."""
    device = torch.device(args.device)
    sim = TensorVectorizedSimulator(args.worlds, device=device, num_robots=6,
                                   seed=args.seed, dt=.02/config.substeps)
    sim.pose[:, :, :2] = torch.tensor([[5.3, 4.], [2., 1.], [2., 7.],
                                      [14., 1.], [14., 4.], [14., 7.]], device=device)
    sim.pose[:, :, 2] = 0
    generator = torch.Generator().manual_seed(args.seed)
    if scene == "pile":
        # A genuine 3D pile; tangent spheres start without manufactured overlap.
        idx = torch.arange(args.pieces)
        positions = torch.stack((6.+(idx % 9)*.151,
                                 3.45+((idx//9) % 8)*.151,
                                 .075+(idx//72)*.151), -1)
        velocities = torch.zeros((args.pieces, 3))
    else:
        # Filter a separated field lattice against expanded robot footprints.
        # Uniform random points can overlap balls or manufacture bumper ejections.
        positions = torch.rand((34*20, 3), generator=generator)
        idx = torch.arange(34*20)
        positions[:, 0] = .45+(idx % 34)*.46+(positions[:, 0]-.5)*.08
        positions[:, 1] = .45+(idx//34)*.38+(positions[:, 1]-.5)*.08
        positions[:, 2] = config.radius
        robot_xy = sim.pose[0, :, :2].cpu()
        overlaps = ((positions[:, None, :2]-robot_xy).abs() < .45+config.radius+.02).all(-1).any(-1)
        legal = positions[~overlaps]
        if legal.shape[0] < args.pieces:
            raise ValueError("not enough separated legal field slots for requested piece count")
        positions = legal[torch.linspace(0, legal.shape[0]-1, args.pieces).long()]
        velocities = torch.randn((args.pieces, 3), generator=generator)*.3
        velocities[:, 2] = 0
        if scene == "settled":
            velocities.zero_()
    pos2 = positions[:, :2].to(device).expand(args.worlds, -1, -1).clone()
    vel2 = velocities[:, :2].to(device).expand_as(pos2).clone()
    fuel = FuelPhysics(sim, pos2, vel2, config=config)
    fuel.pos.copy_(positions.to(device).expand_as(fuel.pos))
    fuel.vel.copy_(velocities.to(device).expand_as(fuel.vel))
    active = torch.ones(args.worlds, dtype=torch.bool, device=device)
    piece_active = torch.ones((args.worlds, args.pieces), dtype=torch.bool, device=device)
    owner = torch.full((args.worlds, args.pieces), -1, dtype=torch.long, device=device)
    command = torch.zeros((args.worlds, 6, 3), device=device)
    command[:, 0, 0] = args.push_speed
    return sim, fuel, active, piece_active, owner, command


def quality(sim, fuel, initial):
    p = fuel.pos.detach()
    finite = bool(torch.isfinite(p).all() and torch.isfinite(fuel.vel).all()
                  and torch.isfinite(sim.velocity).all())
    max_overlap = 0.
    # Chunk worlds so quality diagnostics do not allocate a giant N*P*P tensor.
    for world in range(p.shape[0]):
        dist = torch.cdist(p[world:world+1], p[world:world+1])[0]
        dist.fill_diagonal_(float("inf"))
        max_overlap = max(max_overlap, float((2*fuel.config.radius-dist).clamp_min(0).max()))
    momentum = fuel.config.mass*fuel.vel[..., :2].sum((0, 1)) + (sim.mass[..., None]*sim.velocity[..., :2]).sum((0, 1))
    return {
        "finite": finite,
        "max_ball_penetration_m": max_overlap,
        "max_floor_penetration_m": float((fuel.config.radius-p[..., 2]).clamp_min(0).max()),
        "mean_robot0_forward_speed_mps": float(sim.velocity[:, 0, 0].mean()),
        "mean_robot0_displacement_m": float((sim.pose[:, 0, 0]-initial[:, 0, 0]).mean()),
        "mean_fuel_planar_speed_mps": float(fuel.vel[..., :2].norm(dim=-1).mean()),
        "max_fuel_speed_mps": float(fuel.vel.norm(dim=-1).max()),
        "fuel_kinetic_energy_j_per_world": float((.5*fuel.config.mass*fuel.vel.square().sum(-1)+.5*.4*fuel.config.mass*fuel.config.radius**2*fuel.angular.square().sum(-1)).sum(-1).mean()),
        "planar_momentum_kg_mps": momentum.cpu().tolist(),
        "momentum_note": "Drivetrain and floor supply external forces; this is not a conservation test.",
        "sleeping_fraction": float(fuel.sleeping.float().mean()),
    }


def initial_geometry(sim, fuel):
    """Initial seeded worlds are identical; inspect one on CPU before timing."""
    pos = fuel.pos[0].detach().cpu()
    robots = sim.pose[0].detach().cpu()
    distances = torch.cdist(pos, pos)
    distances.fill_diagonal_(float("inf"))
    delta = pos[None, :, :2]-robots[:, None, :2]
    c, s = robots[:, 2].cos()[:, None], robots[:, 2].sin()[:, None]
    local = torch.stack((c*delta[..., 0]+s*delta[..., 1],
                         -s*delta[..., 0]+c*delta[..., 1]), -1)
    extent = torch.stack((sim.length[0], sim.width[0]), -1).cpu()/2
    q = local.abs()-extent[:, None, :]
    signed_distance = q.clamp_min(0).norm(dim=-1)+q.max(-1).values.clamp_max(0)
    overlap = (fuel.config.radius-signed_distance).clamp_min(0)
    overlap.masked_fill_(pos[None, :, 2] >= fuel.config.robot_height+fuel.config.radius, 0.)
    return {"max_ball_penetration_m":float((2*fuel.config.radius-distances).clamp_min(0).max()),
            "max_robot_penetration_m":float(overlap.max()),
            "max_floor_penetration_m":float((fuel.config.radius-pos[:, 2]).clamp_min(0).max()),
            "terrain":"flat: no bump regions or static field boxes"}


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return value


def benchmark(args, variant, scene, substeps):
    overrides = dict(VARIANTS[variant])
    config = FuelPhysicsConfig(backend=args.backend, substeps=substeps,
                              adaptive_substeps=args.adaptive and variant != "graph",
                              contact_mode="fused", sleep=False, **{k:v for k,v in overrides.items()
                              if k not in ("backend", "contact_mode", "sleep")})
    config = replace(config, **{k:v for k,v in overrides.items()
                               if k in ("backend", "contact_mode", "sleep")})
    timings, states, metrics, stats, memory, graph_parity = [], [], [], [], [], []
    for repeat in range(args.repeats):
        sim, fuel, active, pa, owner, command = initialize(args, config, scene)
        initial = sim.pose.clone()
        actual_substeps = []
        def tick():
            count = fuel.required_substeps(.02, substeps) if config.adaptive_substeps else substeps
            actual_substeps.append(count)
            sim.dt = .02/count
            for _ in range(count):
                sim.step(command, active, _active_nonempty=True)
                if variant != "robot_only":
                    fuel.step(active, pa, owner, dt=.02/count, substeps=1, adaptive=False)
        # Warm on disposable state: timing starts from exactly the same seed.
        for _ in range(args.warmup):
            tick()
        sim, fuel, active, pa, owner, command = initialize(args, config, scene)
        initial = sim.pose.clone()
        gc.collect()  # Reclaim discarded warmup physics/backend reference cycles.
        initial_contacts = initial_geometry(sim, fuel)
        actual_substeps.clear()
        replay = None
        if variant == "graph":
            # Capture is real and errors are exposed if the solver synchronizes.
            stream = torch.cuda.Stream(device=args.device)
            stream.wait_stream(torch.cuda.current_stream(args.device))
            with torch.cuda.stream(stream):
                tick()
            torch.cuda.current_stream(args.device).wait_stream(stream)
            synchronize(args.device)
            replay = torch.cuda.CUDAGraph()
            with torch.cuda.graph(replay, stream=stream):
                tick()
            # Reset captures' tensors in place; Python counters are excluded from metrics.
            fresh = initialize(args, config, scene)
            def restore_graph_state():
                fuel.reset()
                for target_obj, source_obj in ((sim, fresh[0]), (fuel, fresh[1]),
                                               (fuel._hip, fresh[1]._hip)):
                    if target_obj is None:
                        continue
                    for name, target in vars(target_obj).items():
                        source = getattr(source_obj, name, None)
                        if torch.is_tensor(target) and torch.is_tensor(source) and target.shape == source.shape:
                            target.copy_(source)
            restore_graph_state()
            # Short equality checks detect capture/state bugs before chaotic pile
            # trajectories amplify atomic reduction differences.
            parity_ticks = min(4, args.ticks)
            reference = initialize(args, config, scene)
            for _ in range(parity_ticks):
                replay.replay()
                for _ in range(substeps):
                    reference[0].step(reference[5], reference[2], _active_nonempty=True)
                    reference[1].step(reference[2], reference[3], reference[4],
                                      dt=.02/substeps, substeps=1, adaptive=False)
            short_delta = max(float((a-b).abs().max()) for a,b in
                ((fuel.pos, reference[1].pos), (fuel.vel, reference[1].vel),
                 (sim.velocity, reference[0].velocity)))
            graph_parity.append(short_delta)
            if short_delta > 1.e-4:
                raise RuntimeError(f"graph replay differs from eager short trajectory: {short_delta}")
            restore_graph_state()
        synchronize(args.device)
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        for _ in range(args.ticks):
            replay.replay() if replay is not None else tick()
        synchronize(args.device)
        timings.append((time.perf_counter()-started)/args.ticks)
        memory.append(torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else None)
        states.append((fuel.pos.detach().cpu(), fuel.vel.detach().cpu(), sim.velocity.detach().cpu()))
        if variant == "graph":
            ref_sim, ref_fuel, ref_active, ref_pa, ref_owner, ref_command = initialize(args, config, scene)
            for _ in range(args.ticks):
                for _ in range(substeps):
                    ref_sim.step(ref_command, ref_active, _active_nonempty=True)
                    ref_fuel.step(ref_active, ref_pa, ref_owner, dt=.02/substeps, substeps=1, adaptive=False)
            reference = (ref_fuel.pos.cpu(), ref_fuel.vel.cpu(), ref_sim.velocity.cpu())
            graph_delta = max(float((a-b).abs().max()) for a,b in zip(states[-1], reference))
            # Long pile state differences are reported alongside eager repeat
            # variation; short parity above is the capture correctness gate.
        metrics.append(quality(sim, fuel, initial))
        if variant == "graph":
            metrics[-1]["long_graph_eager_max_state_delta"] = graph_delta
        metrics[-1]["target_forward_speed_mps"] = args.push_speed
        metrics[-1]["mean_forward_speed_deficit_mps"] = args.push_speed-metrics[-1]["mean_robot0_displacement_m"]/(args.ticks*.02)
        stats.append(plain(fuel.refresh_stats()))
        if fuel._hip is not None:
            stats[-1]["contact_buffer_bytes"] = fuel._hip.contacts.numel()*fuel._hip.contacts.element_size()
            stats[-1]["neighbor_buffer_bytes"] = fuel._hip.list.numel()*fuel._hip.list.element_size()
        stats[-1]["actual_substeps_per_tick"] = actual_substeps if variant != "graph" else [substeps]*args.ticks
    repeat_delta = max((float((a-b).abs().max()) for state in states[1:]
                        for a,b in zip(states[0],state)), default=0.)
    result = {"variant":variant, "scene":scene, "config":asdict(config),
              "substep_ms":20/substeps, "initial_geometry":initial_contacts, "seconds_per_control_tick":timings,
              "median_seconds_per_control_tick":statistics.median(timings),
              "world_ticks_per_second":args.worlds/statistics.median(timings),
              "repeat_max_state_delta":repeat_delta, "quality":metrics, "stats":stats,
              "peak_live_device_bytes":memory, "graph_short_parity_max_deltas":graph_parity,
              "timing_scope":"coupled robot and fuel physics; excludes observations, planner, policy"}
    return result, states[0]


def end_to_end(args):
    from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv
    times = {False: [], True: []}
    for repeat in range(args.repeats):
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            env = TensorThreeVsThreeEnv(num_envs=args.worlds, device=args.device,
                seed=args.seed, randomize=False, fuel_physics=enabled,
                perception_dropout=0., position_noise=0., velocity_noise=0.)
            env.reset(seed=args.seed)
            gc.collect()
            action=torch.full((args.worlds, 6), 7, device=args.device, dtype=torch.long)
            for _ in range(args.warmup):
                env.step(action,capture_observation=args.capture_observation,capture_info=False)
            synchronize(args.device)
            start=time.perf_counter()
            for _ in range(args.ticks):
                env.step(action,capture_observation=args.capture_observation,capture_info=False)
            synchronize(args.device)
            times[enabled].append((time.perf_counter()-start)/args.ticks)
    rows = [{"fuel_physics":enabled, "seconds_per_step":times[enabled],
             "median_seconds_per_step":statistics.median(times[enabled])}
            for enabled in (False, True)]
    return {"scope":"full environment step; action 7, seeded staged field and default controllers",
            "capture_observation":args.capture_observation, "capture_info":False,
            "order":"alternating A/B and B/A by repetition",
            "rows":rows,"runtime_ratio":rows[1]["median_seconds_per_step"]/rows[0]["median_seconds_per_step"]}


def source_hashes(include_islands=False, full_environment=False):
    root = Path(__file__).resolve().parents[1]
    files = ("frc_defense/fuel_physics.py", "frc_defense/fuel_physics_hip.py",
             "frc_defense/csrc/fuel_physics_hip.cpp", "frc_defense/csrc/fuel_physics_hip_kernel.cu",
             "frc_defense/fuel_physics_runtime.py", "frc_defense/tensor_physics.py",
             "scripts/benchmark_fuel_physics.py")
    if include_islands:
        files += ("frc_defense/fuel_physics_sleep.py",
                  "frc_defense/csrc/fuel_sleep_hip.cpp",
                  "frc_defense/csrc/fuel_sleep_hip_kernel.cu")
    if full_environment:
        files = tuple(sorted(set(files) | {str(path.relative_to(root))
            for path in (root/"frc_defense").rglob("*.py")} | {str(path.relative_to(root))
            for path in (root/"frc_defense"/"csrc").iterdir()
            if path.suffix in (".cu", ".cpp", ".h", ".hpp") and not path.name.endswith("_hip_hip.cpp")}))
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in files if (root/name).exists()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--worlds",type=int,default=24)
    parser.add_argument("--pieces",type=int,default=504)
    parser.add_argument("--ticks",type=int,default=64)
    parser.add_argument("--warmup",type=int,default=3)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--seed",type=int,default=819)
    parser.add_argument("--push-speed",type=float,default=2.,help="Robot0 forward command in m/s")
    parser.add_argument("--backend",choices=("torch","hip","auto"),default="hip")
    parser.add_argument("--variants",nargs="+",choices=tuple(VARIANTS),default=["robot_only","allpairs","grid","cached","compact","sleep"])
    parser.add_argument("--scenes",nargs="+",choices=("sparse","pile","settled"),default=["sparse","pile"])
    parser.add_argument("--substeps",nargs="+",type=int,default=[4,10])
    parser.add_argument("--end-to-end",action="store_true")
    parser.add_argument("--capture-observation",action="store_true",help="Include observation materialization in full environment A/B")
    parser.add_argument("--adaptive",action="store_true",help="Enable production motion bounds once per control tick; graph variant always uses fixed substeps")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if min(args.worlds,args.pieces,args.ticks,args.repeats,*args.substeps)<1 or args.warmup<0:
        parser.error("counts must be positive; warmup must be nonnegative")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    report={"device":args.device,"device_name":torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU",
            "torch":torch.__version__,"hip":torch.version.hip,"arguments":vars(args)|{"output":str(args.output)},"results":[],"errors":[],"source_hashes":source_hashes("island_sleep" in args.variants, args.end_to_end)}
    references={}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    for scene in args.scenes:
        for nsub in args.substeps:
            for variant in args.variants:
                try:
                    row,state=benchmark(args,variant,scene,nsub)
                    if variant=="allpairs":references[(scene,nsub)]=state
                    if (scene,nsub) in references:
                        row["max_state_delta_vs_allpairs"]=max(float((a-b).abs().max()) for a,b in zip(state,references[(scene,nsub)]))
                    report["results"].append(row)
                    print(json.dumps({k:row[k] for k in ("variant","scene","substep_ms","median_seconds_per_control_tick","repeat_max_state_delta")}),flush=True)
                except Exception as exc:
                    error={"variant":variant,"scene":scene,"substeps":nsub,"error":f"{type(exc).__name__}: {exc}"}
                    report["errors"].append(error);print(json.dumps(error),flush=True)
                save()
    if args.end_to_end:
        try:report["end_to_end"]=end_to_end(args)
        except Exception as exc:report["errors"].append({"end_to_end":f"{type(exc).__name__}: {exc}"})
        save()
    report["source_hashes_after"] = source_hashes("island_sleep" in args.variants, args.end_to_end)
    report["sources_changed_during_run"] = report["source_hashes"] != report["source_hashes_after"]
    save()
    if report["errors"] or report["sources_changed_during_run"]:raise SystemExit(1)


if __name__=="__main__":main()
