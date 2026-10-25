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


def reset_scenario(env: TensorThreeVsThreeEnv, scenario: str, seed: int, *,
                   refresh_perception: bool = True, mask=None) -> torch.Tensor | None:
    """Reset positions and physics, optionally deferring sensor exposure.

    Curriculum resets defer perception so registered start stimuli can be
    validated and applied before the first observation is generated.
    """
    if scenario == "offense_vs_deterministic_defense":
        if mask is not None:
            raise ValueError("offense scenario reset has no selected-row contract")
        from .architecture_offense_scenario import reset_offense_scenario
        if not refresh_perception:
            raise ValueError("this scenario reset does not support deferred perception")
        return reset_offense_scenario(env, seed)
    SCENARIOS[scenario]
    if mask is None:
        selected = torch.ones((env.n,), device=env.device, dtype=torch.bool)
        env.reset(seed=seed, defer_perception=not refresh_perception)
    else:
        if (not isinstance(mask, torch.Tensor) or mask.device != env.device or
                mask.dtype is not torch.bool or tuple(mask.shape) != (env.n,)):
            raise ValueError("scenario reset mask must be device bool [world]")
        selected = mask
        env.reseed_world_rng(selected, seed)
        env.reset_done(selected, defer_perception=True)
    def uniform(shape):
        shape = tuple(shape)
        return env._random_active(selected, shape[1:], dtype=env.sim.pose.dtype,
                                  stream="scenario_reset")

    def write_rows(destination, values):
        values = torch.as_tensor(values, device=destination.device,
                                 dtype=destination.dtype).expand_as(destination)
        row_mask = selected.reshape((env.n,) + (1,) * (destination.ndim - 1))
        destination.copy_(torch.where(row_mask, values, destination))

    if scenario != "defense_3v3":
        # A none-mode robot is restored to this pose after each physics step.
        # Separate inactive bodies outside the field so they cannot block lanes.
        for slot, mode in enumerate(env.control_modes):
            if mode == "none":
                write_rows(env.sim.pose[:, slot, 0], -20. - slot * 3.)
                write_rows(env.sim.pose[:, slot, 1], -20.)
        write_rows(env.piece_active, False)
        write_rows(env.piece_owner, -1)
        write_rows(env.piece_zone, -1)
        write_rows(env.piece_vel, 0.)
        write_rows(env.piece_pos, 0.)
        jitter = (uniform((env.n, 2)) - .5) * .2
        if scenario.startswith("pickup_"):
            write_rows(env.sim.pose[:, 3, :2],
                env.sim.pose.new_tensor((7.4, 4.035)) + jitter)
            write_rows(env.sim.pose[:, 3, 2], 0.)
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
            write_rows(env.piece_pos[:, :count],
                env.sim.pose[:, 3, None, :2] + offsets)
            write_rows(env.piece_active[:, :count], True)
            write_rows(env.piece_zone[:, :count], 0)
        else:
            # Open midfield side of the red HUB, clear of the HUB and BUMPs.
            write_rows(env.sim.pose[:, 0, :2],
                env.sim.pose.new_tensor((8.1, 4.035)) + jitter)
            write_rows(env.sim.pose[:, 0, 2], math.pi)
            write_rows(env.sim.pose[:, 3, :2],
                env.sim.pose.new_tensor((6.8, 4.035)) + jitter)
            write_rows(env.sim.pose[:, 3, 2], 0.)
            count = 24
            write_rows(env.piece_pos[:, :count], env.sim.pose[:, 0, None, :2])
            write_rows(env.piece_active[:, :count], True)
            write_rows(env.piece_owner[:, :count], 0)
            write_rows(env.piece_zone[:, :count], 5)

    write_rows(env.sim.velocity, 0.)
    write_rows(env._target_collect_heading, env.sim.pose[:, :, 2])
    write_rows(env._target_turn_anchor, env.sim.pose[:, :, :2])
    write_rows(env._raster_cursor, env._closest_raster_indices(env.sim.pose[:, :, :2]))
    write_rows(env._raster_reanchor, True)
    # observe() exposes cached route features before the first replan. Clear
    # all mutable planner outputs so reset cannot retain a previous policy's
    # route or progress, even when the same environment is reused for evaluation.
    for name in ("last_path", "last_lengths", "last_intercept", "last_intercept_time",
                 "last_start", "last_goal", "last_heading", "last_progress",
                 "last_trench_alignment", "last_speed_profile", "potential"):
        value = getattr(env.planner, name)
        write_rows(value, 0.)
    if mask is None:
        env._planner_controller_modes_dirty = True
    else:
        env._planner_controller_modes_dirty_mask |= selected

    # Pickup acceleration caches every possible object cell. Replace the reset
    # spawn cell after placing scenario objects while retaining respawn cells.
    cells_x = (env.piece_pos[..., 0] / env._pickup_grid_cell_size).floor().long()
    cells_y = (env.piece_pos[..., 1] / env._pickup_grid_cell_size).floor().long()
    cells = (cells_y.clamp(0, env._pickup_grid_ny - 1) * env._pickup_grid_nx
             + cells_x.clamp(0, env._pickup_grid_nx - 1))
    write_rows(env._pickup_possible_cells[:, :, 0], cells.to(torch.int16))
    reset_fuel_physics(env, selected)
    for tensor in (env.track_pos, env.track_vel, env.track_mask,
                   env.track_confidence, env.track_spatial_extent,
                   env.track_angular_extent, env.track_merged,
                   env._current_fuel_visibility, env.perception_quality_state,
                   env.opponent_valid, env.opponent_pose, env.opponent_velocity,
                   env.opponent_size):
        write_rows(tensor, 0.)
    write_rows(env.track_age, float("inf"))
    write_rows(env.opponent_age, float("inf"))
    if not refresh_perception:
        return None
    env._update_perception(selected)
    fresh = env.observe()
    if env._last_obs is None:
        env._last_obs = fresh
    else:
        write_rows(env._last_obs, fresh)
    return env._last_obs
