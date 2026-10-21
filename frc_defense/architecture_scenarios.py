"""Seeded perception-driven scenarios for the architecture benchmark.

Scenario setup may place true objects; policies only receive the environment's
normal simulated sensor observations. No tracks are copied from object truth.
"""
from __future__ import annotations

import math
import torch

from .fuel_physics_runtime import reset_fuel_physics
from .tensor_3v3 import TensorThreeVsThreeEnv


SCENARIOS = {
    "pickup_simple": {
        "description": "One blue defender collects six nearby balls in an open midfield lane.",
        "learned_slots": (3,), "fuel_count": 96,
    },
    "pickup_scattered": {
        "description": "One blue defender searches a seeded 24-ball midfield scatter.",
        "learned_slots": (3,), "fuel_count": 96,
    },
    "deterministic_defense": {
        "description": "One blue defender delays a loaded deterministic red attacker approaching its HUB.",
        "learned_slots": (3,), "fuel_count": 96,
    },
    "defense_3v3": {
        "description": "Three blue defenders face three deterministic red attackers on the full field.",
        "learned_slots": (3, 4, 5), "fuel_count": 504,
    },
    "offense_vs_deterministic_defense": {
        "description": (
            "Experimental blue neural offense collects 24 midfield balls against "
            "a deterministic red defender. NN offense uses generic strategic "
            "execution without deterministic-only intake/scoring refinements."
        ),
        "learned_slots": (3,), "fuel_count": 96, "experimental": True,
    },
}


def make_env(scenario: str, num_envs: int, device: str, seed: int,
             horizon: int = 400) -> TensorThreeVsThreeEnv:
    """Construct a real-timestep environment with fuel physics and sensor noise."""
    if scenario == "offense_vs_deterministic_defense":
        from .architecture_offense_scenario import make_offense_env
        return make_offense_env(num_envs, device, seed, horizon=horizon)
    config = SCENARIOS[scenario]
    modes = ["none"] * 6
    for slot in config["learned_slots"]:
        modes[slot] = "nn"
    if scenario == "deterministic_defense":
        modes[0] = "deterministic"
    elif scenario == "defense_3v3":
        modes[:3] = ["deterministic"] * 3
    env = TensorThreeVsThreeEnv(
        num_envs=num_envs, device=device, seed=seed, control_modes=modes,
        robot_roles=("offense",) * 3 + ("defense",) * 3,
        dt=.02, horizon=horizon, randomize=False,
        fuel_count=config["fuel_count"], fuel_physics=True,
        perception_dropout=.02, position_noise=.015, velocity_noise=.05,
    )
    env.architecture_pickup_only = scenario.startswith("pickup_")
    reset_scenario(env, scenario, seed)
    return env


def reset_scenario(env: TensorThreeVsThreeEnv, scenario: str, seed: int) -> torch.Tensor:
    """Reset positions, physics, pickup indexing, and simulated perception."""
    if scenario == "offense_vs_deterministic_defense":
        from .architecture_offense_scenario import reset_offense_scenario
        return reset_offense_scenario(env, seed)
    SCENARIOS[scenario]
    env.reset(seed=seed)
    generator = torch.Generator(device=env.device).manual_seed(int(seed))

    def uniform(shape):
        return torch.rand(shape, device=env.device, generator=generator,
                          dtype=env.sim.pose.dtype)

    if scenario != "defense_3v3":
        # A none-mode robot is restored to this pose after each physics step.
        # Separate inactive bodies outside the field so they cannot block lanes.
        for slot, mode in enumerate(env.control_modes):
            if mode == "none":
                env.sim.pose[:, slot, 0] = -20. - slot * 3.
                env.sim.pose[:, slot, 1] = -20.
        env.piece_active.zero_()
        env.piece_owner.fill_(-1)
        env.piece_zone.fill_(-1)
        env.piece_vel.zero_()
        env.piece_pos.zero_()
        jitter = (uniform((env.n, 2)) - .5) * .2
        if scenario.startswith("pickup_"):
            env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((7.4, 4.035)) + jitter
            env.sim.pose[:, 3, 2] = 0.
            count = 6 if scenario == "pickup_simple" else 24
            if scenario == "pickup_simple":
                # Two rows, 0.32 m spacing: distinct physical balls near intake.
                offsets = env.piece_pos.new_tensor([
                    (.9, -.16), (.9, .16), (1.22, -.16),
                    (1.22, .16), (1.54, -.16), (1.54, .16),
                ])[None].expand(env.n, -1, -1)
            else:
                # Seeded jitter on a separated grid avoids overlapping balls.
                x = torch.linspace(-.8, 2.5, 6, device=env.device)
                y = torch.linspace(-1.5, 1.5, 4, device=env.device)
                yy, xx = torch.meshgrid(y, x, indexing="ij")
                offsets = torch.stack((xx.flatten(), yy.flatten()), -1)[None]
                offsets = offsets + (uniform((env.n, count, 2)) - .5) * .16
            env.piece_pos[:, :count] = env.sim.pose[:, 3, None, :2] + offsets
            env.piece_active[:, :count] = True
            env.piece_zone[:, :count] = 0
        else:
            # Open midfield side of the red HUB, clear of the HUB and BUMPs.
            env.sim.pose[:, 0, :2] = env.sim.pose.new_tensor((8.1, 4.035)) + jitter
            env.sim.pose[:, 0, 2] = math.pi
            env.sim.pose[:, 3, :2] = env.sim.pose.new_tensor((6.8, 4.035)) + jitter
            env.sim.pose[:, 3, 2] = 0.
            count = 24
            env.piece_pos[:, :count] = env.sim.pose[:, 0, None, :2]
            env.piece_active[:, :count] = True
            env.piece_owner[:, :count] = 0
            env.piece_zone[:, :count] = 5

    env.sim.velocity.zero_()
    env._target_collect_heading.copy_(env.sim.pose[:, :, 2])
    env._target_turn_anchor.copy_(env.sim.pose[:, :, :2])
    env._raster_cursor.copy_(env._closest_raster_indices(env.sim.pose[:, :, :2]))
    env._raster_reanchor.fill_(True)
    # observe() exposes cached route features before the first replan. Clear
    # all mutable planner outputs so reset cannot retain a previous policy's
    # route or progress, even when the same environment is reused for evaluation.
    for name in ("last_path", "last_lengths", "last_intercept", "last_intercept_time",
                 "last_start", "last_goal", "last_heading", "last_progress",
                 "last_trench_alignment", "last_speed_profile", "potential"):
        getattr(env.planner, name).zero_()
    env._planner_controller_modes_dirty = True

    # Pickup acceleration caches every possible object cell. Replace the reset
    # spawn cell after placing scenario objects while retaining respawn cells.
    cells_x = (env.piece_pos[..., 0] / env._pickup_grid_cell_size).floor().long()
    cells_y = (env.piece_pos[..., 1] / env._pickup_grid_cell_size).floor().long()
    cells = (cells_y.clamp(0, env._pickup_grid_ny - 1) * env._pickup_grid_nx
             + cells_x.clamp(0, env._pickup_grid_nx - 1))
    env._pickup_possible_cells[:, :, 0] = cells.to(torch.int16)
    reset_fuel_physics(env)
    env.track_pos.zero_()
    env.track_vel.zero_()
    env.track_mask.zero_()
    env.track_age.fill_(float("inf"))
    env.track_confidence.zero_()
    env.track_spatial_extent.zero_()
    env.track_angular_extent.zero_()
    env.track_merged.zero_()
    env._current_fuel_visibility.zero_()
    env.perception_quality_state.zero_()
    env.opponent_valid.zero_()
    env.opponent_pose.zero_()
    env.opponent_velocity.zero_()
    env.opponent_age.fill_(float("inf"))
    env.opponent_size.zero_()
    env._perception_rng_ticks.zero_()
    env._update_perception(torch.ones(env.n, device=env.device, dtype=torch.bool))
    env._last_obs = env.observe()
    return env._last_obs
