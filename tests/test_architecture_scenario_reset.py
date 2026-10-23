"""Scenario reset regressions; behavioral qualification also requires E2E runs."""

import pytest
import torch

from frc_defense import architecture_offense_scenario, architecture_scenarios
from frc_defense.architecture_inputs import PerceptionInputs
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


@pytest.mark.parametrize("scenario", [
    "pickup_simple", "pickup_scattered", "deterministic_defense",
    "offense_vs_deterministic_defense",
])
def test_seeded_reset_discards_previous_routes_and_sensor_tracks(monkeypatch, scenario):
    # Real simulator and sensor code, without initializing the optional FUEL
    # rigid-body backend: the contract under test is reset, not FUEL motion.
    def cpu_env(**kwargs):
        kwargs["fuel_physics"] = False
        return TensorThreeVsThreeEnv(**kwargs)

    monkeypatch.setattr(architecture_scenarios, "TensorThreeVsThreeEnv", cpu_env)
    monkeypatch.setattr(architecture_offense_scenario, "TensorThreeVsThreeEnv", cpu_env)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        seed = 731
        env = architecture_scenarios.make_env(scenario, 1, "cpu", seed, horizon=400)
        builder = PerceptionInputs(env, [3])
        initial_obs = architecture_scenarios.reset_scenario(env, scenario, seed).clone()
        initial_inputs = {key: value.clone() for key, value in builder(initial_obs).items()}
        if scenario == "offense_vs_deterministic_defense":
            assert env.architecture_experimental_nn_offense_intercept_disabled
            assert not initial_obs[:, 3, -3].bool().any()
            assert not initial_obs[:, 3, -2].bool().any()
            env.opponent_valid[:, 3] = True
            forced_valid_obs = env.observe()
            assert not forced_valid_obs[:, 3, -3].bool().any()
            assert not forced_valid_obs[:, 3, -2].bool().any()
            # Red slot 0 retains normal defense interception permission.
            assert forced_valid_obs[:, 0, -3].bool().all()
        for name in (
            "last_path", "last_lengths", "last_intercept", "last_intercept_time",
            "last_start", "last_goal", "last_heading", "last_progress",
            "last_trench_alignment", "last_speed_profile", "potential",
        ):
            getattr(env.planner, name).fill_(1)
        env.track_pos.fill_(123.)
        env.track_vel.fill_(-37.)
        env.track_mask.fill_(True)
        env.track_age.zero_()
        env.track_confidence.fill_(.8)
        env.opponent_pose.fill_(88.)
        env.opponent_velocity.fill_(-14.)
        env.opponent_valid.fill_(True)
        env.opponent_age.zero_()
        env.opponent_size.fill_(7.)
        reset_obs = architecture_scenarios.reset_scenario(env, scenario, seed)
        torch.testing.assert_close(reset_obs, initial_obs, rtol=0., atol=0.)
        reset_inputs = builder(reset_obs)
        for key in initial_inputs:
            torch.testing.assert_close(reset_inputs[key], initial_inputs[key], rtol=0., atol=0.)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("partial_active", [False, True])
def test_cpu_trajectory_reset_and_fallback_aging_keep_graph_tensor_addresses(partial_active):
    """A graph captures age storage; fallback steps must update that storage."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        seed = 100007
        env = architecture_scenarios.make_env(
            "deterministic_defense", 2, "cpu", seed, horizon=400,
        )
        assert env.fuel_physics is not None
        assert not env._physics_graph_enabled
        actions = torch.full((2, 6), 7, dtype=torch.long)
        actions[:, 3] = 6
        active = torch.tensor([True, False]) if partial_active else None
        pointers = (env.track_age.data_ptr(), env.opponent_age.data_ptr())
        episodes = []
        observed_opponent = False
        for _ in range(2):
            architecture_scenarios.reset_scenario(env, "deterministic_defense", seed)
            frames = []
            for _ in range(12):
                env.step(actions, active_mask=active, capture_observation=False, capture_info=False)
                assert (env.track_age.data_ptr(), env.opponent_age.data_ptr()) == pointers
                observed_opponent |= bool(env.opponent_valid[0].any())
                frames.append(torch.cat((
                    env.sim.pose.flatten(), env.sim.velocity.flatten(),
                    env.fuel_score_count.flatten().float(),
                    env.fuel_acquisition_count.flatten().float(),
                    env.opponent_pose.flatten(), env.opponent_age.flatten(),
                    env.track_age.flatten(),
                )))
            episodes.append(torch.stack(frames))
        assert observed_opponent
        torch.testing.assert_close(episodes[0], episodes[1], rtol=0., atol=0., equal_nan=True)
        # Both branches advance the active world through six sensor updates
        # and six intervening aging ticks; partial stepping leaves world 1 still.
        assert env.steps.tolist() == ([12, 0] if partial_active else [12, 12])
    finally:
        torch.set_num_threads(previous_threads)
