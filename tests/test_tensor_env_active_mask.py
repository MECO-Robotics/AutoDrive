import unittest

import torch

from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense.tensor_training import _env, _held_action_interval


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
    def test_deferred_intermediate_observation_preserves_interval_state(self):
        kwargs = dict(num_envs=4, task="counter_defense", device=torch.device("cpu"),
                      seed=1907, opponent="guard", horizon=2,
                      architecture="strategic_adstar")
        reference = _env(**kwargs)
        deferred = _env(**kwargs)
        initial_reference = reference.reset(seed=1907)[0]
        initial_deferred = deferred.reset(seed=1907)[0]
        torch.testing.assert_close(initial_reference, initial_deferred, rtol=0, atol=0)
        action = torch.tensor((0, 1, 6, 7), dtype=torch.long)
        expected = _held_action_interval(
            reference, action, 4, initial_reference, known_remaining_ticks=2,
            return_info=False, defer_intermediate_observation=False)
        actual = _held_action_interval(
            deferred, action, 4, initial_deferred, known_remaining_ticks=2,
            return_info=False, defer_intermediate_observation=True)
        for index, (left, right) in enumerate(zip(expected, actual)):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0,
                                           msg=f"held interval result {index} differs")
        for object_name in ("env", "sim", "_adstar_planners", "_adstar_defender_planners"):
            lhs = reference if object_name == "env" else getattr(
                reference, "sim" if object_name == "sim" else object_name, None)
            rhs = deferred if object_name == "env" else getattr(
                deferred, "sim" if object_name == "sim" else object_name, None)
            if lhs is None or rhs is None:
                continue
            for name, value in vars(lhs).items():
                candidate = getattr(rhs, name, None)
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, candidate, rtol=0, atol=0,
                                               msg=f"{object_name}.{name} differs")
                elif isinstance(value, tuple) and all(isinstance(v, torch.Tensor) for v in value):
                    for item, (left, right) in enumerate(zip(value, candidate)):
                        torch.testing.assert_close(left, right, rtol=0, atol=0,
                            msg=f"{object_name}.{name}[{item}] differs")
        self.assertTrue(torch.equal(reference.generator.get_state(),
                                    deferred.generator.get_state()))
        self.assertTrue(torch.equal(reference.sim.generator.get_state(),
                                    deferred.sim.generator.get_state()))
        for lhs, rhs in zip(reference._pending_strategic_own_candidates[0],
                             deferred._pending_strategic_own_candidates[0]):
            torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
        torch.testing.assert_close(reference._pending_strategic_own_candidates[1],
                                   deferred._pending_strategic_own_candidates[1],
                                   rtol=0, atol=0)

    def test_training_env_reuses_exact_strategic_observation_candidates(self):
        strategic = _env(1, "counter_defense", torch.device("cpu"), 17, "guard",
                         architecture="strategic_adstar")
        direct = _env(1, "counter_defense", torch.device("cpu"), 17, "guard",
                      architecture="direct")
        self.assertTrue(strategic.reuse_strategic_own_candidates)
        self.assertFalse(direct.reuse_strategic_own_candidates)

    def test_role_swapped_strategic_observation_reuses_candidate_mask(self):
        env = TensorDefenseEnv(
            num_envs=2, task="defense", device="cpu", opponent="learned",
            action_mode="strategic", randomize=False, seed=82,
            observation_noise=0., observation_dropout=0.,
            perception_config={"detection_dropout": 0., "position_noise_m": 0.,
                               "velocity_noise_mps": 0.},
        )
        reference_observation = env._role_swapped_observation()
        reference_candidates = env._fuel_candidates(1)
        reference_mask = env.strategic_action_mask(1, _candidate_data=reference_candidates)
        observation, candidates, action_mask = env._role_swapped_observation(
            return_candidate_data=True)
        torch.testing.assert_close(observation, reference_observation, rtol=0, atol=0)
        for actual, expected in zip(candidates, reference_candidates):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(action_mask, reference_mask, rtol=0, atol=0)

        candidate_calls = []
        original_candidates = env._fuel_candidates

        def count_candidates(focal):
            candidate_calls.append(focal)
            return original_candidates(focal)

        env._fuel_candidates = count_candidates
        env.learned_opponent_fn = lambda _obs: torch.zeros(
            env.n, device=env.device, dtype=torch.long)
        env.learned_opponent_action_dim = 8
        env.step(torch.zeros(env.n, device=env.device, dtype=torch.long),
                 _return_info=False)
        self.assertEqual(candidate_calls.count(1), 1)

    def test_own_strategic_candidate_cache_preserves_step_and_avoids_rerank(self):
        kwargs = dict(
            num_envs=2, task="counter_defense", device="cpu", opponent="guard",
            action_mode="strategic", randomize=True, seed=813,
        )
        legacy = TensorDefenseEnv(**kwargs)
        cached = TensorDefenseEnv(**kwargs, reuse_strategic_own_candidates=True)
        legacy.reset(seed=813)
        cached.reset(seed=813)
        counts = {"legacy": 0, "cached": 0}
        for label, env in (("legacy", legacy), ("cached", cached)):
            original = env._fuel_candidates

            def count(focal, *, _label=label, _original=original):
                if focal == 0:
                    counts[_label] += 1
                return _original(focal)

            env._fuel_candidates = count

        action = torch.tensor([0, 7], dtype=torch.long)
        active_masks = (torch.tensor([False, True]), torch.ones(2, dtype=torch.bool))
        pending_before_step = tuple(value.clone() for value in
            (*cached._pending_strategic_own_candidates[0],
             cached._pending_strategic_own_candidates[1]))
        for step_index, active_mask in enumerate(active_masks):
            expected = legacy.step(action, active_mask=active_mask, _return_info=False)
            actual = cached.step(action, active_mask=active_mask, _return_info=False)
            for index, (left, right) in enumerate(zip(expected[:4], actual[:4])):
                torch.testing.assert_close(left, right, rtol=0, atol=0,
                                           msg=f"cached step {step_index} output {index} differs")
            if step_index == 0:
                pending_after_step = (*cached._pending_strategic_own_candidates[0],
                                      cached._pending_strategic_own_candidates[1])
                for before, after in zip(pending_before_step, pending_after_step):
                    torch.testing.assert_close(after[0], before[0], rtol=0, atol=0,
                                               msg="inactive candidate cache row changed during step")
                pending_before_reset = tuple(value.clone() for value in pending_after_step)
                reset_mask = torch.tensor([True, False])
                legacy_observation = legacy.reset_done(reset_mask)
                cached_observation = cached.reset_done(reset_mask)
                torch.testing.assert_close(legacy_observation, cached_observation,
                                           rtol=0, atol=0,
                                           msg="partial reset observation differs")
                pending_after_reset = (*cached._pending_strategic_own_candidates[0],
                                       cached._pending_strategic_own_candidates[1])
                for before, after in zip(pending_before_reset, pending_after_reset):
                    torch.testing.assert_close(after[1], before[1], rtol=0, atol=0,
                                               msg="unreset candidate cache row changed during reset")
        self.assertEqual(counts, {"legacy": 5, "cached": 3})
        self.assertTrue(torch.equal(legacy.generator.get_state(), cached.generator.get_state()))
        self.assertTrue(torch.equal(legacy.sim.generator.get_state(), cached.sim.generator.get_state()))
        for world in range(legacy.n):
            left = _world_tensor_state(legacy, world)
            right = _world_tensor_state(cached, world)
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0,
                                           msg=f"world {world} {name} differs")

    def test_training_no_info_path_matches_normal_step(self):
        kwargs = dict(
            num_envs=2, task="counter_defense", device="cpu", opponent="guard",
            action_mode="strategic", seed=321, horizon=8000,
            observation_noise=0., observation_dropout=0.,
            perception_config={"detection_dropout": 0., "position_noise_m": 0.,
                               "velocity_noise_mps": 0.},
        )
        with_info = TensorDefenseEnv(**kwargs)
        without_info = TensorDefenseEnv(**kwargs)
        action = torch.tensor([0, 7], dtype=torch.long)
        objective_calls = {"with_info": 0, "without_info": 0}
        with_info_objective = with_info._attacker_objective
        without_info_objective = without_info._attacker_objective

        def count_with_info_objective():
            objective_calls["with_info"] += 1
            return with_info_objective()

        def count_without_info_objective():
            objective_calls["without_info"] += 1
            return without_info_objective()

        with_info._attacker_objective = count_with_info_objective
        without_info._attacker_objective = count_without_info_objective
        expected = with_info.step(action)
        actual = without_info.step(action, _return_info=False)
        for index, (expected_value, actual_value) in enumerate(zip(expected[:4], actual[:4])):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0,
                                       msg=f"step output {index} differs")
        self.assertTrue(expected[4])
        self.assertEqual(actual[4], {})
        self.assertEqual(objective_calls, {"with_info": 1, "without_info": 0})
        for name in ("pose", "velocity", "module_angle"):
            torch.testing.assert_close(getattr(with_info.sim, name),
                                       getattr(without_info.sim, name), rtol=0, atol=0)
        for name in ("piece_pos", "piece_vel", "piece_active", "piece_owner",
                     "steps", "match_elapsed", "match_remaining", "_track_pos",
                     "_track_age", "fuel_score_count", "fuel_acquisition_count"):
            torch.testing.assert_close(getattr(with_info, name), getattr(without_info, name),
                                       rtol=0, atol=0)
        torch.testing.assert_close(with_info.generator.get_state(),
                                   without_info.generator.get_state(), rtol=0, atol=0)

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
        _, done, truncated, terminal_obs, active, _, active_world_ticks, matches = result
        self.assertEqual(matches, 3)
        self.assertEqual(active_world_ticks, 6)
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
