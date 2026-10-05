#!/usr/bin/env python3
"""Integrated exact-parity check for the guarded fused swerve + pose path."""
from __future__ import annotations

import os
import json
from pathlib import Path
import torch

os.environ["AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_HIP"] = "1"
from frc_defense.tensor_sim import TensorDefenseEnv, TensorVectorizedSimulator


def _same(a, b, where):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
            if isinstance(b, torch.Tensor) and a.shape == b.shape:
                diff = (a - b).abs()
                detail = f"max_abs={float(diff.max().item()):.9g}"
            else:
                detail = f"shapes={getattr(a, 'shape', None)}/{getattr(b, 'shape', None)}"
            raise AssertionError(f"{where} differs ({detail})")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f"{where} mapping keys differ")
        for key in a:
            _same(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(f"{where} sequence differs")
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    elif a is None or isinstance(a, (bool, int, float, str)):
        if a != b:
            raise AssertionError(f"{where}: {a!r} != {b!r}")


def _state_views(env):
    state = {f"env.{k}": v for k, v in vars(env).items()
             if isinstance(v, torch.Tensor)}
    state.update({f"sim.{k}": v for k, v in vars(env.sim).items()
                  if isinstance(v, torch.Tensor)})
    for pname in ("_adstar_planners", "_adstar_defender_planners",
                  "_adstar_tactical_planner"):
        planner = getattr(env, pname, None)
        if planner is not None:
            for key, value in vars(planner).items():
                if isinstance(value, torch.Tensor):
                    state[f"{pname}.{key}"] = value
    return state


def _integrated_env_parity(worlds=64, ticks=64, device="cuda:0"):
    from frc_defense.field import bump_boxes, rebuilt_field
    bump = bump_boxes(rebuilt_field())[0].as_tensor()
    bump_x, bump_y = float(bump[0]), float(bump[1])
    env_name = "AUTODRIVE_FUSED_SWERVE_POSE_HIP"
    previous = os.environ.get(env_name)
    os.environ[env_name] = "0"
    torch_env = TensorDefenseEnv(
        num_envs=worlds, device=device, seed=75019, task="counter_defense",
        opponent="guard", action_mode="direct", randomize=False,
        observation_noise=0.0, observation_dropout=0.0,
        perception_config={"detection_dropout": 0.0, "position_noise_m": 0.0},
        fuel_count=96, horizon=2000)
    os.environ[env_name] = "1"
    hip_env = TensorDefenseEnv(
        num_envs=worlds, device=device, seed=75019, task="counter_defense",
        opponent="guard", action_mode="direct", randomize=False,
        observation_noise=0.0, observation_dropout=0.0,
        perception_config={"detection_dropout": 0.0, "position_noise_m": 0.0},
        fuel_count=96, horizon=2000)
    if previous is None:
        os.environ.pop(env_name, None)
    else:
        os.environ[env_name] = previous

    obs_a, _ = torch_env.reset(seed=75019)
    obs_b, _ = hip_env.reset(seed=75019)
    _same(obs_a, obs_b, "reset observation")
    if not hip_env.sim._fused_swerve_pose_hip_enabled:
        raise AssertionError("fused swerve path was not enabled by the opt-in")

    # Place a subset in contact with both a neighboring robot and the field
    # boundary, and refresh observations identically before the rollout.
    for env in (torch_env, hip_env):
        env.sim.pose[:8, 0, 0] = 0.28
        env.sim.pose[:8, 1, 0] = 0.62
        env.sim.pose[:8, :, 1] = 0.46
        env.sim.pose[:8, :, 2] = 0.02
        env.sim.pose[8:16, 0, 0] = bump_x
        env.sim.pose[8:16, 1, 0] = bump_x + 0.16
        env.sim.pose[8:16, :, 1] = bump_y
        env.sim.pose[8:16, :, 2] = 0.0
        env.sim.velocity[:8, :, :2] = 0
        env.sim.velocity[:8, :, 2] = 0
        env._update_perception(torch.ones(worlds, device=device, dtype=torch.bool))
        env._last_observation = env._obs(torch.ones(worlds, device=device,
                                                     dtype=torch.bool))
    _same(torch_env._last_observation, hip_env._last_observation,
          "contact-start observation")

    gen = torch.Generator(device=device).manual_seed(939)
    action_bank = torch.rand((ticks, worlds, 3), device=device, generator=gen) * 2 - 1
    for tick in range(ticks):
        if tick % 17 == 0:
            active = torch.ones(worlds, dtype=torch.bool, device=device)
        elif tick % 17 == 1:
            active = torch.zeros(worlds, dtype=torch.bool, device=device)
        else:
            active = torch.arange(worlds, device=device).remainder(4).ne(tick % 4)
        count = int(active.sum().item())
        result_a = torch_env.step(action_bank[tick], active_mask=active,
                                 _active_count=count)
        result_b = hip_env.step(action_bank[tick], active_mask=active,
                                _active_count=count)
        _same(result_a, result_b, f"step-{tick} result/reward/info")
        for name in ("_planner_tick_scalar", "_planner_tick_aligned",
                     "_planner_full_batch_active"):
            _same(getattr(torch_env, name), getattr(hip_env, name),
                  f"step-{tick}.{name}")
        _same(torch_env.generator.get_state(), hip_env.generator.get_state(),
              f"step-{tick}.env-rng")
        _same(torch_env.sim.generator.get_state(), hip_env.sim.generator.get_state(),
              f"step-{tick}.sim-rng")
        state_a, state_b = _state_views(torch_env), _state_views(hip_env)
        _same(state_a, state_b, f"step-{tick} full-state")
    return {"worlds": worlds, "ticks": ticks, "exact": True,
            "contact_worlds": 8, "planner_cadence_ticks": [20, 40, 60],
            "fused_enabled": hip_env.sim._fused_swerve_pose_hip_enabled}


def _integrated_sim_contact_parity(worlds=64, ticks=64, device="cuda:0"):
    from frc_defense.field import bump_boxes, rebuilt_field
    bump_regions = [box.as_tensor() for box in bump_boxes(rebuilt_field())]
    reference = TensorVectorizedSimulator(num_envs=worlds, device=device,
        seed=621, randomize=False, bump_regions=bump_regions)
    candidate = TensorVectorizedSimulator(num_envs=worlds, device=device,
        seed=621, randomize=False, bump_regions=bump_regions)
    candidate._fused_swerve_pose_hip_enabled = True
    for sim in (reference, candidate):
        sim.pose[:, 0, 0] = 0.28
        sim.pose[:, 1, 0] = 0.62
        sim.pose[:, :, 1] = 0.46
        sim.pose[:, :, 2] = 0.02
        bump = bump_boxes(rebuilt_field())[0].as_tensor()
        sim.pose[8:16, 0, 0] = bump[0]
        sim.pose[8:16, 1, 0] = bump[0] + 0.16
        sim.pose[8:16, :, 1] = bump[1]
        sim.pose[8:16, :, 2] = 0.0
        sim.velocity.zero_()
    generator = torch.Generator(device=device).manual_seed(662)
    command_bank = torch.rand((ticks, worlds, 2, 3), device=device,
                              generator=generator) * 2 - 1
    state_names = ("pose", "velocity", "module_steer_rate", "module_angle",
        "module_current", "module_supply_current", "robot_current",
        "module_drive_speed", "robot_contact", "opponent_contact",
        "field_contact", "wall_contact")
    for tick in range(ticks):
        if tick % 17 == 0:
            active = torch.ones(worlds, dtype=torch.bool, device=device)
        elif tick % 17 == 1:
            active = torch.zeros(worlds, dtype=torch.bool, device=device)
        else:
            active = torch.arange(worlds, device=device).remainder(4).ne(tick % 4)
        reference._step_eager(command_bank[tick], active, _active_nonempty=True,
                              _use_compiled_swerve=False)
        candidate._step_eager(command_bank[tick], active, _active_nonempty=True,
                              _use_compiled_swerve=True)
        for name in state_names:
            _same(getattr(reference, name), getattr(candidate, name),
                  f"sim-step-{tick}.{name}")
        _same(reference.generator.get_state(), candidate.generator.get_state(),
              f"sim-step-{tick}.rng")
    if not bool(reference.robot_contact.any()):
        raise AssertionError("contact parity case failed to exercise contact")
    return {"worlds": worlds, "ticks": ticks, "exact": True,
            "contact_observed": bool(reference.robot_contact.any().item())}


def main():
    if not torch.cuda.is_available() or not torch.version.hip:
        print("SKIP: HIP is not available")
        return 0
    free, total = torch.cuda.mem_get_info()
    if free < 2 * 1024**3:
        print(f"SKIP: only {free / 1024**3:.2f} GiB GPU memory free")
        return 0
    env_name = "AUTODRIVE_FUSED_SWERVE_POSE_HIP"
    old_env = os.environ.pop(env_name, None)
    default_on = TensorVectorizedSimulator(
        num_envs=1, device="cuda:0", seed=17, randomize=False
    )._fused_swerve_pose_hip_enabled
    cpu_fallback = not TensorVectorizedSimulator(
        num_envs=1, device="cpu", seed=17, randomize=False
    )._fused_swerve_pose_hip_enabled
    os.environ[env_name] = "0"
    explicit_opt_out = not TensorVectorizedSimulator(
        num_envs=1, device="cuda:0", seed=17, randomize=False
    )._fused_swerve_pose_hip_enabled
    if old_env is not None:
        os.environ[env_name] = old_env
    else:
        os.environ.pop(env_name, None)
    if not default_on or not explicit_opt_out or not cpu_fallback:
        raise AssertionError("HIP default, =0 opt-out, or CPU fallback check failed")
    results = {
        "free_gib": free / 1024**3,
        "default_on": default_on,
        "explicit_opt_out": explicit_opt_out,
        "cpu_fallback": cpu_fallback,
        "sim_contact_parity": _integrated_sim_contact_parity(),
        "env_reward_trace_parity": _integrated_env_parity(),
    }
    out_dir = Path("evaluations/strategic-ppo-validation-20261001")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "swerve_pose_runtime_parity.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    results["artifact"] = str(out_path.resolve())
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
