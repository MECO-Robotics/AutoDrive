import unittest

import torch

from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense.tensor_training import _held_action_interval


def _world_tensor_state(env, index):
    """Capture every batched tensor row, including nested planner caches."""
    objects = {"env": env, "sim": env.sim}
    for name in ("_adstar_planners", "_adstar_defender_planners",
                 "_adstar_tactical_planner"):
        planner = getattr(env, name, None)
        if planner is not None:
            objects[name] = planner
    state = {}
    for object_name, obj in objects.items():
        for name, value in vars(obj).items():
            if (isinstance(value, torch.Tensor) and value.ndim > 0
                    and value.shape[0] == env.n):
                state[f"{object_name}.{name}"] = value[index].clone()
    return state


class TensorEnvActiveMaskTests(unittest.TestCase):
    def test_inactive_world_state_planner_and_rng_are_untouched(self):
        env = TensorDefenseEnv(
            num_envs=2, task="defense", device="cpu", opponent="adstar",
            action_mode="strategic", randomize=False, observation_noise=0.,
            observation_dropout=0., adstar_replan_interval=1,
            perception_config={"detection_dropout": 0., "position_noise_m": 0.,
                               "velocity_noise_mps": 0.},
        )
        action = torch.zeros(2, dtype=torch.long)
        env.step(action)
        env.steps[0] = env.horizon - 1
        _, _, _, truncated, _ = env.step(action, active_mask=torch.ones(2, dtype=torch.bool))
        self.assertTrue(bool(truncated[0]))

        planner = env._adstar_planners
        planner_cache = {
            name: {
                key: getattr(value, key)[0].clone()
                for key in ("last_path", "last_goal", "last_start", "last_heading",
                            "last_lengths", "last_speed_profile", "potential")
            }
            for name, value in (
                ("opponent", env._adstar_planners),
                ("tactical", env._adstar_tactical_planner),
            ) if value is not None
        }
        frozen = {
            "pose": env.sim.pose[0].clone(),
            "velocity": env.sim.velocity[0].clone(),
            "module_angle": env.sim.module_angle[0].clone(),
            "contact": env.sim.robot_contact[0].clone(),
            "clock": env.match_elapsed[0].clone(),
            "pieces": env.piece_pos[0].clone(),
            "piece_velocity": env.piece_vel[0].clone(),
            "piece_active": env.piece_active[0].clone(),
            "piece_owner": env.piece_owner[0].clone(),
            "tracks": env._track_pos[0].clone(),
            "track_age": env._track_age[0].clone(),
            "opponent_tracks": env._opponent_track_pose[0].clone(),
            "opponent_track_age": env._opponent_track_age[0].clone(),
            "planner_tick": env._planner_tick[0].clone(),
            "steps": env.steps[0].clone(),
            "fuel_count": env.fuel_score_count[0].clone(),
            "action_history": env.action_history[0].clone(),
            "last_opponent_command": env._last_opponent_command[0].clone(),
        }
        every_world_tensor = _world_tensor_state(env, 0)
        rng_before = env.generator.get_state().clone()
        sim_rng_before = env.sim.generator.get_state().clone()
        expected_rng = torch.Generator(device="cpu")
        expected_rng.set_state(rng_before)
        for _ in range(2):
            torch.randn((1, env.fuel_count, 2), generator=expected_rng)
            torch.rand((1,), generator=expected_rng)
            torch.randn((1, 2), generator=expected_rng)
            torch.randn((1,), generator=expected_rng)
            torch.randn((1, 3), generator=expected_rng)

        obs, reward, done, truncated, info = env.step(
            action, active_mask=torch.tensor([False, True]))
        self.assertTrue(torch.equal(env.generator.get_state(), expected_rng.get_state()))
        self.assertTrue(torch.equal(env.sim.generator.get_state(), sim_rng_before))
        self.assertTrue(torch.equal(env.sim.pose[0], frozen["pose"]))
        self.assertTrue(torch.equal(env.sim.velocity[0], frozen["velocity"]))
        self.assertTrue(torch.equal(env.sim.module_angle[0], frozen["module_angle"]))
        self.assertTrue(torch.equal(env.sim.robot_contact[0], frozen["contact"]))
        self.assertTrue(torch.equal(env.match_elapsed[0], frozen["clock"]))
        self.assertTrue(torch.equal(env.piece_pos[0], frozen["pieces"]))
        self.assertTrue(torch.equal(env.piece_vel[0], frozen["piece_velocity"]))
        self.assertTrue(torch.equal(env.piece_active[0], frozen["piece_active"]))
        self.assertTrue(torch.equal(env.piece_owner[0], frozen["piece_owner"]))
        self.assertTrue(torch.equal(env._track_pos[0], frozen["tracks"]))
        self.assertTrue(torch.equal(env._track_age[0], frozen["track_age"]))
        self.assertTrue(torch.equal(env._opponent_track_pose[0], frozen["opponent_tracks"]))
        self.assertTrue(torch.equal(env._opponent_track_age[0], frozen["opponent_track_age"]))
        self.assertTrue(torch.equal(env._planner_tick[0], frozen["planner_tick"]))
        self.assertTrue(torch.equal(env.steps[0], frozen["steps"]))
        self.assertTrue(torch.equal(env.fuel_score_count[0], frozen["fuel_count"]))
        self.assertTrue(torch.equal(env.action_history[0], frozen["action_history"]))
        self.assertTrue(torch.equal(env._last_opponent_command[0], frozen["last_opponent_command"]))
        for name, cache in planner_cache.items():
            current = (env._adstar_planners if name == "opponent"
                       else env._adstar_tactical_planner)
            for key, value in cache.items():
                self.assertTrue(torch.equal(getattr(current, key)[0], value), f"{name}.{key}")
        self.assertEqual(float(reward[0]), 0.)
        self.assertFalse(bool(done[0]))
        self.assertFalse(bool(truncated[0]))
        self.assertTrue(torch.equal(obs[0], env._last_observation[0]))
        self.assertEqual(float(info["fuel_acquired_event"][0].sum()), 0.)
        self.assertEqual(float(info["contact"][0]), 0.)
        for name, value in every_world_tensor.items():
            obj_name, attr = name.split(".", 1)
            obj = {"env": env, "sim": env.sim,
                   "_adstar_planners": env._adstar_planners,
                   "_adstar_defender_planners": env._adstar_defender_planners,
                   "_adstar_tactical_planner": env._adstar_tactical_planner}[obj_name]
            self.assertTrue(torch.equal(getattr(obj, attr)[0], value), name)

        # The step returns the terminal transition without auto-resetting it.
        self.assertEqual(int(env.steps[0]), env.horizon)
        continuing = {
            "pose": env.sim.pose[1].clone(),
            "clock": env.match_elapsed[1].clone(),
            "steps": env.steps[1].clone(),
            "tracks": env._track_pos[1].clone(),
            "path": env._adstar_planners.last_path[1].clone(),
            "planner_tick": env._planner_tick[1].clone(),
        }
        env.reset_done(torch.tensor([True, False]))
        self.assertTrue(torch.equal(env.sim.pose[1], continuing["pose"]))
        self.assertTrue(torch.equal(env.match_elapsed[1], continuing["clock"]))
        self.assertTrue(torch.equal(env.steps[1], continuing["steps"]))
        self.assertTrue(torch.equal(env._track_pos[1], continuing["tracks"]))
        self.assertTrue(torch.equal(env._adstar_planners.last_path[1], continuing["path"]))
        self.assertTrue(torch.equal(env._planner_tick[1], continuing["planner_tick"]))

        rng_before_noop = env.generator.get_state().clone()
        sim_rng_before_noop = env.sim.generator.get_state().clone()
        obs_before_noop = env._last_observation.clone()
        obs, reward, done, truncated, _ = env.step(
            action, active_mask=torch.zeros(2, dtype=torch.bool))
        self.assertTrue(torch.equal(env.generator.get_state(), rng_before_noop))
        self.assertTrue(torch.equal(env.sim.generator.get_state(), sim_rng_before_noop))
        self.assertTrue(torch.equal(obs, obs_before_noop))
        self.assertTrue(torch.equal(reward, torch.zeros_like(reward)))
        self.assertFalse(bool(done.any()))
        self.assertFalse(bool(truncated.any()))

    def test_staggered_horizon_ends_stop_on_their_own_physics_tick(self):
        env = TensorDefenseEnv(
            num_envs=3, task="defense", device="cpu", opponent="adstar",
            action_mode="strategic", randomize=False, observation_noise=0.,
            observation_dropout=0., adstar_replan_interval=1,
            perception_config={"detection_dropout": 0., "position_noise_m": 0.,
                               "velocity_noise_mps": 0.},
        )
        action = torch.zeros(3, dtype=torch.long)
        env.steps.copy_(torch.tensor([env.horizon - 1, env.horizon - 2,
                                      env.horizon - 3]))
        initial_obs = env._last_observation.clone()
        result = _held_action_interval(env, action, 6, initial_obs)
        _, done, truncated, terminal_obs, active, _, matches = result
        self.assertEqual(matches, 3)
        self.assertFalse(bool(done.any()))
        self.assertTrue(bool(truncated.all()))
        self.assertFalse(bool(active.any()))
        self.assertTrue(torch.equal(env.steps, torch.full_like(env.steps, env.horizon)))
        self.assertTrue(torch.isfinite(terminal_obs).all())
        state_at_finalize = [_world_tensor_state(env, i) for i in range(env.n)]
        rng_at_finalize = env.generator.get_state().clone()
        env.step(action, active_mask=active)
        self.assertTrue(torch.equal(env.generator.get_state(), rng_at_finalize))
        for i, state in enumerate(state_at_finalize):
            for name, value in state.items():
                object_name, attr = name.split(".", 1)
                obj = {"env": env, "sim": env.sim,
                       "_adstar_planners": env._adstar_planners,
                       "_adstar_defender_planners": env._adstar_defender_planners,
                       "_adstar_tactical_planner": env._adstar_tactical_planner}[object_name]
                self.assertTrue(torch.equal(getattr(obj, attr)[i], value), name)

    def test_none_mask_matches_default_step(self):
        kwargs = dict(num_envs=2, task="defense", device="cpu", opponent="mirror",
                      action_mode="strategic", randomize=False, seed=913,
                      observation_noise=0., observation_dropout=0.)
        implicit = TensorDefenseEnv(**kwargs)
        explicit = TensorDefenseEnv(**kwargs)
        action = torch.tensor([1, 7], dtype=torch.long)
        implicit_obs, implicit_reward, implicit_done, implicit_truncated, _ = implicit.step(action)
        explicit_obs, explicit_reward, explicit_done, explicit_truncated, _ = explicit.step(
            action, active_mask=None)
        self.assertTrue(torch.equal(implicit_obs, explicit_obs))
        self.assertTrue(torch.equal(implicit_reward, explicit_reward))
        self.assertTrue(torch.equal(implicit_done, explicit_done))
        self.assertTrue(torch.equal(implicit_truncated, explicit_truncated))
        for i in range(implicit.n):
            self.assertEqual(_world_tensor_state(implicit, i).keys(),
                             _world_tensor_state(explicit, i).keys())
            for name, value in _world_tensor_state(implicit, i).items():
                self.assertTrue(torch.equal(value, _world_tensor_state(explicit, i)[name]), name)


if __name__ == "__main__":
    unittest.main()
