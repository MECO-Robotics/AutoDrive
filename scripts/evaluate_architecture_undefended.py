"""Evaluate true undefended references without changing simulator source.

Blue controllers are removed at construction and their bodies are parked
outside the field after every seeded scenario reset. Native red deterministic
controllers, FUEL physics, and simulated perception remain enabled.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import weakref
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from frc_defense import architecture_benchmark as benchmark
from frc_defense import architecture_scenarios as scenarios


def repeat_agreement(first, repeated, tolerance=1e-5):
    """Require exact event metrics; tolerate only tiny frame float differences."""
    maximum, maximum_path, first_difference = 0., None, None
    structural = []

    def compare(a, b, path):
        nonlocal maximum, maximum_path, first_difference
        if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in a:
                compare(a[key], b[key], path + [key])
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + [index])
        elif a != b:
            if first_difference is None:
                first_difference = path
            if isinstance(a, float) and isinstance(b, float):
                delta = abs(a-b)
                if delta > maximum:
                    maximum, maximum_path = delta, path
            else:
                structural.append(path)

    compare(first["frames"], repeated["frames"], [])
    metrics_exact = all(first[key] == repeated.get(key) for key in first if key != "frames")
    first_tick = (first["frames"][first_difference[0]]["tick"]
                  if first_difference is not None else None)
    return {"bitwise_equal": first == repeated, "event_metrics_exact": metrics_exact,
            "frame_float_tolerance": tolerance, "max_frame_float_delta": maximum,
            "max_delta_path": maximum_path, "first_difference_path": first_difference,
            "first_difference_tick": first_tick, "structural_difference_paths": structural,
            "qualified": metrics_exact and not structural and maximum <= tolerance}


def evaluate_undefended(scenario, seed, *, num_envs=4, device="cuda:0",
                       ticks=400, action_repeat=10, repeat_check=True):
    original_constructor = scenarios.TensorThreeVsThreeEnv
    original_reset = scenarios.reset_scenario

    def undefended_constructor(**kwargs):
        modes = list(kwargs["control_modes"])
        for robot in (3, 4, 5):
            modes[robot] = "none"
        kwargs["control_modes"] = tuple(modes)
        return original_constructor(**kwargs)

    reset_checks = []

    def reset_without_blue(env, scenario_name, reset_seed):
        original_reset(env, scenario_name, reset_seed)
        red_pose = env.sim.pose[:, :3].clone()
        fuel_pos, fuel_vel = env.piece_pos.clone(), env.piece_vel.clone()
        fuel_active, fuel_owner = env.piece_active.clone(), env.piece_owner.clone()
        for robot in (3, 4, 5):
            env.sim.pose[:, robot, 0] = -20. - robot * 3.
            env.sim.pose[:, robot, 1] = -20.
            env.sim.pose[:, robot, 2] = 0.
            env.sim.velocity[:, robot].zero_()
        env._target_collect_heading.copy_(env.sim.pose[:, :, 2])
        env._target_turn_anchor.copy_(env.sim.pose[:, :, :2])
        env._raster_cursor.copy_(env._closest_raster_indices(env.sim.pose[:, :, :2]))
        env._raster_reanchor.fill_(True)
        # The shared focused reset places blue slot 3 inside the field before
        # this wrapper parks it. Discard that intermediate sensor observation
        # and regenerate actual perception after removing the blue bodies.
        for name in ("track_pos", "track_vel", "track_confidence", "track_spatial_extent",
                     "track_angular_extent", "perception_quality_state", "opponent_pose",
                     "opponent_velocity", "opponent_size"):
            getattr(env, name).zero_()
        for name in ("track_mask", "track_merged", "_current_fuel_visibility", "opponent_valid"):
            getattr(env, name).zero_()
        env.track_age.fill_(float("inf"))
        env.opponent_age.fill_(float("inf"))
        env._perception_rng_ticks.zero_()
        env._update_perception(torch.ones(env.n, device=env.device, dtype=torch.bool))
        env._last_obs = env.observe()
        checks = {
            "red_placement_preserved": torch.equal(red_pose, env.sim.pose[:, :3]),
            "fuel_placement_preserved": torch.equal(fuel_pos, env.piece_pos),
            "fuel_velocity_preserved": torch.equal(fuel_vel, env.piece_vel),
            "fuel_activity_preserved": torch.equal(fuel_active, env.piece_active),
            "fuel_ownership_preserved": torch.equal(fuel_owner, env.piece_owner),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Undefended parking changed scenario setup: {checks}")
        reset_checks.append(checks)
        return env._last_obs

    with patch.object(scenarios, "TensorThreeVsThreeEnv", undefended_constructor):
        env = scenarios.make_env(scenario, num_envs, device, seed, horizon=ticks)
    parked_pose = env.sim.pose.new_tensor([[-29., -20., 0.], [-32., -20., 0.], [-35., -20., 0.]])
    # None-mode poses are normally restored only after the complete physics
    # tick; wall solves otherwise teleport them into a field corner first.
    # Keep removed blue bodies outside at every robot/FUEL contact boundary.
    env._physics_graph_enabled = False
    env_ref = weakref.ref(env)
    sim_ref = weakref.ref(env.sim)
    physics_ref = weakref.ref(env.fuel_physics)
    original_walls = type(env.sim)._walls
    original_robot_collision = type(env.sim)._robot_collision
    original_physics_step = type(env.fuel_physics).step
    original_step = weakref.WeakMethod(env.step)
    checked_ticks = 0
    absence_checks = {"wall_restores": 0, "robot_contact_before": 0,
                      "robot_contact_after": 0, "fuel_contact_before": 0,
                      "fuel_contact_after": 0}

    def assert_blue_outside(sim, phase):
        if not torch.equal(sim.pose[:, 3:], parked_pose[None].expand(sim.n, -1, -1)):
            raise RuntimeError(f"Removed blue body entered contact solve at {phase}")
        absence_checks[phase] += 1

    def absent_walls(*args, **kwargs):
        sim = sim_ref()
        returned = original_walls(sim, *args, **kwargs)
        sim.pose[:, 3:].copy_(parked_pose[None])
        sim.velocity[:, 3:].zero_()
        assert_blue_outside(sim, "wall_restores")
        return returned

    def absent_robot_collision(*args, **kwargs):
        sim = sim_ref()
        assert_blue_outside(sim, "robot_contact_before")
        returned = original_robot_collision(sim, *args, **kwargs)
        assert_blue_outside(sim, "robot_contact_after")
        return returned

    def absent_physics_step(*args, **kwargs):
        physics = physics_ref()
        assert_blue_outside(physics.sim, "fuel_contact_before")
        returned = original_physics_step(physics, *args, **kwargs)
        assert_blue_outside(physics.sim, "fuel_contact_after")
        return returned

    env.sim._walls = absent_walls
    env.sim._robot_collision = absent_robot_collision
    env.fuel_physics.step = absent_physics_step

    def checked_step(*args, **kwargs):
        nonlocal checked_ticks
        current_env = env_ref()
        result = original_step()(*args, **kwargs)
        if not torch.equal(current_env.sim.pose[:, 3:], parked_pose[None].expand(current_env.n, -1, -1)):
            raise RuntimeError("A removed blue body moved or reentered the field")
        checked_ticks += 1
        return result

    env.step = checked_step
    ids = scenarios.SCENARIOS[scenario]["learned_slots"]
    try:
        with patch.object(benchmark, "reset_scenario", reset_without_blue):
            result = benchmark.evaluate(None, env, scenario, seed, ids, ticks, action_repeat,
                                        reference="hold")
            if repeat_check:
                repeated = benchmark.evaluate(None, env, scenario, seed, ids, ticks, action_repeat,
                                              reference="hold")
                agreement = repeat_agreement(result, repeated)
                if not agreement["qualified"]:
                    differing = [key for key in result if result[key] != repeated.get(key)]
                    error = RuntimeError(f"Undefended seeded repeat differs in {differing}")
                    error.repeat_diagnostic = {"first": result, "repeated": repeated,
                                               "differing_fields": differing,
                                               "repeat_agreement": agreement,
                                               "reset_checks": list(reset_checks),
                                               "substep_blue_contact_checks": dict(absence_checks)}
                    raise error
                result["repeat_run"] = repeated
                result["repeat_agreement"] = agreement
    finally:
        # Remove the closure→bound-method→env cycle before another CUDA graph
        # capture can trigger GC of a prior environment's captured graphs.
        del env.step
        del env.sim._walls
        del env.sim._robot_collision
        del env.fuel_physics.step
    result.update(
        scenario=scenario, reference="blue_absent", num_envs=num_envs,
        action_repeat=action_repeat, control_modes=list(env.control_modes),
        robot_roles=list(env.robot_roles),
        blue_bodies_parked_outside=True, every_tick_blue_pose_checked=True,
        checked_physics_ticks=checked_ticks, seeded_repeat_checked=repeat_check,
        exact_seeded_repeat_checked=(agreement["bitwise_equal"] if repeat_check else False),
        substep_blue_contact_checks=absence_checks,
        absence_intervention={
            "restore_blue_after_every_wall_solve": True,
            "blue_outside_before_and_after_every_robot_contact": True,
            "blue_outside_before_and_after_every_fuel_contact": True,
            "physics_graph_enabled": False,
            "remaining_red_none_bodies_follow_native_simulator_behavior": True,
        },
        red_acquisitions_per_world=env.fuel_acquisition_count[:, :3].sum(-1).cpu().tolist(),
        free_fuel_count_per_world=(env.piece_active & (env.piece_owner < 0)).sum(-1).cpu().tolist(),
        carried_fuel_count_per_world=(env.piece_active & (env.piece_owner >= 0)).sum(-1).cpu().tolist(),
        final_robot_poses_per_world=env.sim.pose.cpu().tolist(),
        reset_checks=reset_checks,
        perception_protocol_note=(
            "Native simulated perception; removed blue controllers are none-mode. "
            "Full 3v3 therefore uses sparse-controller sensor execution rather than "
            "the all-robots-controlled stereo batching path. Same seed/placement "
            "does not promise identical sensor noise streams after removing observers."
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=("deterministic_defense", "defense_3v3"),
                        default=["deterministic_defense", "defense_3v3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[100007, 100019, 100043])
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "evaluations/architecture-study-20261020/references-undefended")
    parser.add_argument("--expected-source", type=Path,
                        help="Require this source manifest's hash before running")
    args = parser.parse_args()
    if min(args.envs, args.ticks, args.action_repeat) < 1:
        parser.error("envs, ticks, and action-repeat must be positive")
    torch.set_num_threads(1)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    torch.backends.cudnn.enabled = False
    provenance = benchmark.source_provenance()
    if args.expected_source:
        expected = json.loads(args.expected_source.read_text())["source_sha256"]
        if provenance["source_sha256"] != expected:
            raise RuntimeError("Current simulator source does not match expected manifest")
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for scenario in args.scenarios:
        for seed in args.seeds:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            gc.collect()
            started = time.perf_counter()
            try:
                result = evaluate_undefended(
                    scenario, seed, num_envs=args.envs, device=args.device,
                    ticks=args.ticks, action_repeat=args.action_repeat, repeat_check=args.repeat_check,
                )
            except RuntimeError as error:
                if hasattr(error, "repeat_diagnostic"):
                    directory = args.output / scenario / f"seed-{seed}-n-{args.envs}-ticks-{args.ticks}"
                    directory.mkdir(parents=True, exist_ok=True)
                    diagnostic = error.repeat_diagnostic
                    diagnostic.update(status="repeat_mismatch", provenance=provenance,
                                      script_sha256=script_sha256)
                    (directory / "repeat-mismatch.json").write_text(json.dumps(diagnostic, indent=2))
                raise
            result.update(provenance=provenance, source_sha256=provenance["source_sha256"],
                          script_sha256=script_sha256, device=args.device,
                          elapsed_seconds=time.perf_counter()-started)
            directory = args.output / scenario / f"seed-{seed}-n-{args.envs}-ticks-{args.ticks}"
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / "undefended.json"
            destination.write_text(json.dumps(result, indent=2))
            print(json.dumps({"scenario": scenario, "seed": seed,
                              "mean_red_score": sum(result["score_red_per_world"])/args.envs,
                              "mean_red_score_integral": result["mean_red_score_integral"],
                              "path": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
