#!/usr/bin/env python3
"""Exact HIP parity harness for the isolated perception commit prototype."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense import tensor_perception_commit_proto as prototype


TRACK_NAMES = ("_track_pos", "_track_vel", "_track_mask", "_track_age",
               "_opponent_track_pose", "_opponent_track_velocity",
               "_opponent_track_valid", "_opponent_track_age")


def _assert_equal(lhs, rhs, label):
    if not torch.equal(lhs, rhs):
        raise AssertionError(f"{label} differs")


def _new_env(seed, device, worlds):
    env = TensorDefenseEnv(
        num_envs=worlds, task="counter_defense", device=device, seed=seed,
        action_mode="strategic", randomize=True, observation_noise=0.0,
        observation_dropout=0.0,
        perception_config={"fov_degrees": 270., "range_m": 9.,
            "detection_dropout": .17, "position_noise_m": .025,
            "velocity_noise_mps": .08, "track_timeout_s": .35},
        fuel_count=96, horizon=2000)
    env.reset(seed=seed)
    return env


def _seed_global(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_known_state(env):
    # Ensure both robots have visible, moving pieces, plus stale tracks which
    # exercise age expiry and stale-value retention.
    offsets = torch.tensor(((.15, .04), (.31, -.08), (.55, .17), (.79, -.22)),
                           device=env.device, dtype=env.sim.pose.dtype)
    for robot in range(2):
        origin = env.sim.pose[:, robot, :2]
        piece_slice = slice(robot * 2, robot * 2 + 2)
        env.piece_pos[:, piece_slice] = origin[:, None, :] + offsets[None, robot * 2:robot * 2 + 2, :]
        env.piece_vel[:, piece_slice] = torch.tensor((.23, -.11), device=env.device)
        env.piece_active[:, piece_slice] = True
        env.piece_owner[:, piece_slice] = -1
    env._track_mask[:, :, :8] = True
    env._track_pos[:, :, :8] = 2.25
    env._track_vel[:, :, :8] = -.75
    ages = torch.tensor((.0, .1, .3, .4, .7, .2, .35, .01), device=env.device)
    env._track_age[:, :, :8] = ages
    env._opponent_track_valid[:] = True
    env._opponent_track_pose[:] = .5
    env._opponent_track_velocity[:] = -.25
    env._opponent_track_age[:] = torch.tensor((.1, .4), device=env.device)


def _run_case(case, seed, device, worlds):
    _seed_global(seed)
    reference = _new_env(seed, device, worlds)
    _set_known_state(reference)
    _seed_global(seed)
    candidate = _new_env(seed, device, worlds)
    _set_known_state(candidate)
    active = {
        "all_active": torch.ones(worlds, device=device, dtype=torch.bool),
        "mixed_active": torch.arange(worlds, device=device).remainder(3).ne(1),
        "all_inactive": torch.zeros(worlds, device=device, dtype=torch.bool),
    }[case]
    before = {name: getattr(reference, name).clone() for name in TRACK_NAMES}
    count = int(active.sum().item())
    before_rng = reference.generator.get_state().clone()
    reference._update_perception(active_mask=active, active_count=count)
    reference_rng = reference.generator.get_state().clone()
    prototype.update_perception(candidate, active_mask=active, active_count=count)
    candidate_rng = candidate.generator.get_state().clone()
    for name in TRACK_NAMES:
        _assert_equal(getattr(reference, name), getattr(candidate, name), name)
        if case != "all_active":
            inactive = ~active
            _assert_equal(getattr(reference, name)[inactive], before[name][inactive],
                          f"{name} inactive rows")
    _assert_equal(reference_rng, candidate_rng, "generator state")
    if case == "all_inactive":
        _assert_equal(reference_rng, before_rng, "all-inactive generator state")
    return {"case": case, "active_worlds": count, "exact_state": True,
            "rng_exact": True, "both_robot_track_arrays_checked": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=48)
    parser.add_argument("--seed", type=int, default=70231)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("This prototype parity harness requires HIP.")
    if prototype.extension() is None:
        raise SystemExit("Perception commit HIP extension failed to load.")
    cases = [_run_case(name, args.seed + index, args.device, args.worlds)
             for index, name in enumerate(("all_active", "mixed_active", "all_inactive"))]
    report = {"prototype": "ordinary post-randomization perception commit",
              "device": args.device, "worlds": args.worlds,
              "rng_generation": "Torch reference draw order unchanged",
              "cases": cases, "all_exact": True}
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
