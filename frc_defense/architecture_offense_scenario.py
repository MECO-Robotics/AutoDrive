"""Experimental neural offense against a deterministic defender.

Production construction currently restricts neural policies to defense roles.
This benchmark-only factory updates the controller caches after construction
and narrows this instance's action mask to isolate collection and scoring.
Interception is excluded; ferrying is disabled to prevent release/reacquisition
reward loops without scoring.
Core production code remains unchanged.

Important limitation: several intake and scoring execution refinements are
gated specifically on deterministic offense in the current simulator. Neural
offense therefore uses the generic strategic-action execution path. Results
must not be presented as parity with the deterministic offense controller.
"""

from __future__ import annotations

import math
import weakref

import torch

from .architecture_scenarios import reset_scenario
from .tensor_3v3 import TensorThreeVsThreeEnv


SCENARIO = {
    "description": (
        "Experimental blue neural offense collects 24 midfield balls against "
        "a deterministic red defender guarding the blue HUB approach. "
        "Uses generic strategic execution; production NN-offense is unsupported."
    ),
    "learned_slots": (3,),
    "fuel_count": 96,
    "experimental": True,
}


def make_offense_env(num_envs: int, device: str, seed: int,
                     horizon: int = 800) -> TensorThreeVsThreeEnv:
    env = TensorThreeVsThreeEnv(
        num_envs=num_envs, device=device, seed=seed,
        control_modes=("deterministic", "none", "none", "deterministic", "none", "none"),
        robot_roles=("defense", "offense", "offense", "offense", "defense", "defense"),
        dt=.02, horizon=horizon, randomize=False, fuel_count=96,
        fuel_physics=True, perception_dropout=.02, position_noise=.015,
        velocity_noise=.05,
    )
    modes = list(env.control_modes)
    modes[3] = "nn"
    env.control_modes = tuple(modes)
    env.agent_train_mask[3] = True
    env._nn_mode_mask[3] = True
    env._deterministic_mode_mask[3] = False
    # The set of controlled robots is unchanged, while deterministic offense
    # no longer exists. Keep its host and tensor caches consistent.
    env._has_deterministic_offense = False
    env._deterministic_offense_robot_ids = ()
    env._deterministic_offense_robot_index = torch.empty(
        0, device=env.device, dtype=torch.long,
    )
    env._planner_controller_modes_dirty = True
    env._target_action_graph = None
    env._target_action_graph_signature = None
    env.architecture_experimental_nn_offense = True
    original_action_mask = weakref.WeakMethod(env._action_mask)

    def perception_only_offense_mask(indices, valid):
        bound_mask = original_action_mask()
        if bound_mask is None:
            raise ReferenceError("Offense environment no longer exists")
        mask, possession = bound_mask(indices, valid)
        # Keep this solo offense task focused on collection and scoring;
        # preserve the red defender's interception action and exclude the
        # learned slot's interception and ferry recycling actions.
        mask[:, 3, 5:7] = False
        return mask, possession

    env._action_mask = perception_only_offense_mask
    env.architecture_experimental_nn_offense_intercept_disabled = True
    reset_offense_scenario(env, seed)
    return env


def reset_offense_scenario(env: TensorThreeVsThreeEnv, seed: int) -> torch.Tensor:
    # Reuse seeded 24-ball placement, pickup-cell indexing, fuel physics reset,
    # and route-cache clearing from the benchmark's scatter scenario.
    reset_scenario(env, "pickup_scattered", seed)
    env.architecture_pickup_only = False
    env.sim.pose[:, 0, :2] = env.sim.pose.new_tensor((10.1, 4.035))
    env.sim.pose[:, 0, 2] = math.pi
    env._target_collect_heading.copy_(env.sim.pose[:, :, 2])
    env._target_turn_anchor.copy_(env.sim.pose[:, :, :2])
    env._raster_cursor.copy_(env._closest_raster_indices(env.sim.pose[:, :, :2]))
    env._raster_reanchor.fill_(True)
    # Regenerate actual sensor output after moving the defender. Never seed
    # perception tracks directly from game-object or opponent truth.
    env.track_mask.zero_()
    env.track_age.fill_(float("inf"))
    env.track_confidence.zero_()
    env.track_spatial_extent.zero_()
    env.track_angular_extent.zero_()
    env.track_merged.zero_()
    env._current_fuel_visibility.zero_()
    env.perception_quality_state.zero_()
    env.opponent_valid.zero_()
    env.opponent_age.fill_(float("inf"))
    env.opponent_size.zero_()
    env.opponent_pose.zero_()
    env.opponent_velocity.zero_()
    env._perception_rng_ticks.zero_()
    env._update_perception(torch.ones(env.n, device=env.device, dtype=torch.bool))
    env._last_obs = env.observe()
    return env._last_obs
