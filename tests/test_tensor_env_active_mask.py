import unittest

import torch

from frc_defense.tensor_sim import TensorDefenseEnv


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
        rng_before = env.generator.get_state().clone()
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
        obs_before_noop = env._last_observation.clone()
        obs, reward, done, truncated, _ = env.step(
            action, active_mask=torch.zeros(2, dtype=torch.bool))
        self.assertTrue(torch.equal(env.generator.get_state(), rng_before_noop))
        self.assertTrue(torch.equal(obs, obs_before_noop))
        self.assertTrue(torch.equal(reward, torch.zeros_like(reward)))
        self.assertFalse(bool(done.any()))
        self.assertFalse(bool(truncated.any()))


if __name__ == "__main__":
    unittest.main()
