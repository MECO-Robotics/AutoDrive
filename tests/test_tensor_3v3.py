import pytest
import math

torch = pytest.importorskip("torch")

from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv
from frc_defense.tensor_training import ActorCritic


def test_rebuilt_hub_activation_follows_auto_transition_shifts_and_endgame():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=17, horizon=8000,
        randomize=False, fuel_count=96,
    )
    # Red wins AUTO, so its HUB is inactive for shift 1. The status swaps
    # every 25 s, and both HUBS are active during transition and end game.
    env.auto_fuel_scores[0] = torch.tensor([1, 0])
    samples = [
        (29.96, [True, True]),
        (29.98, [False, True]),
        (54.96, [False, True]),
        (54.98, [True, False]),
        (79.98, [False, True]),
        (104.98, [True, False]),
        (129.98, [True, True]),
    ]
    for elapsed_before_tick, expected in samples:
        env.match_elapsed.fill_(elapsed_before_tick)
        env._update_match(torch.tensor([True]))
        assert env.hub_active[0].tolist() == expected


def test_proxy_auto_counts_freeze_at_transition_and_use_seeded_tie():
    from frc_defense.rebuilt_match_scoring import derive_simulation_tie_selection

    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=71, horizon=8000,
        randomize=False, fuel_count=96, fuel_proxy_enabled=True,
        simulation_episode_id="test-episode", simulation_match_id="test-match",
        simulation_seed=71,
    )
    expected_tie = derive_simulation_tie_selection(
        "test-match", 71, world_index=0, reset_generation=int(
            env._fuel_proxy_reset_generation[0].item()))["alliance"]

    # The score tick at exactly 20 seconds is excluded from AUTO; the frozen
    # pair is captured before any transition score can affect shift 1.
    env.auto_fuel_scores[0] = torch.tensor([3, 3])
    env.match_elapsed.fill_(19.98)
    env._update_match(torch.tensor([True]))
    assert env.match_elapsed[0].item() == pytest.approx(20.)
    assert env._frozen_auto_fuel_scores[0].tolist() == [3, 3]
    assert env._auto_fuel_frozen.tolist() == [True]

    env.auto_fuel_scores[0] = torch.tensor([3, 9])
    env.match_elapsed.fill_(29.96)
    env._update_match(torch.tensor([True]))
    assert env.hub_active[0].tolist() == [True, True]
    assert env._frozen_auto_fuel_scores[0].tolist() == [3, 3]
    env.match_elapsed.fill_(29.98)
    env._update_match(torch.tensor([True]))
    expected = [expected_tie != "red", expected_tie != "blue"]
    assert env.hub_active[0].tolist() == expected


def test_runtime_hub_state_matches_pinned_contract_at_all_phase_boundaries():
    from frc_defense.rebuilt_match_scoring import hub_is_active

    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=72, horizon=8000,
        randomize=False, fuel_count=96, fuel_proxy_enabled=True,
        simulation_episode_id="boundary-episode", simulation_match_id="boundary-match",
        simulation_seed=72,
    )
    env._auto_fuel_frozen.fill_(True)
    env._frozen_auto_fuel_scores[0] = torch.tensor([4, 2])
    counts = {"red": 4, "blue": 2}
    for boundary in (0., 20., 30., 55., 80., 105., 130., 160.):
        center_tick = round(boundary / env.dt)
        for tick in (max(0, center_tick-1), center_tick, min(8000, center_tick+1)):
            time_s = min(tick * env.dt, 160.)
            env.steps.fill_(tick)
            env._update_match(torch.tensor([True]),
                clock_time=torch.tensor([time_s], dtype=env.match_elapsed.dtype))
            contract_time = min(time_s, math.nextafter(160., 0.))
            assert env.hub_active[0].tolist() == [
                hub_is_active(alliance, contract_time, counts)
                for alliance in ("red", "blue")]


def test_manual_match_update_uses_device_phase_boundaries():
    from frc_defense.rebuilt_match_scoring import hub_is_active

    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=75, horizon=8000,
        randomize=False, fuel_count=96, fuel_proxy_enabled=True,
        simulation_episode_id="manual-phase-episode",
        simulation_match_id="manual-phase-match", simulation_seed=75,
    )
    env._auto_fuel_frozen.fill_(True)
    env._frozen_auto_fuel_scores[0] = torch.tensor([2, 2])
    tie_counts = {"red": 2, "blue": 2}
    tie = env._fuel_proxy_tie_alliance[0].item()
    selected = ("red", "blue")[tie]
    for boundary in (0., 20., 30., 55., 80., 105., 130., 160.):
        center_tick = round(boundary / env.dt)
        for tick in (max(0, center_tick-1), center_tick, min(8000, center_tick+1)):
            time_s = min(tick * env.dt, 160.)
            env.match_elapsed.fill_(time_s - env.dt)
            env._update_match(torch.tensor([True]))
            contract_time = min(time_s, math.nextafter(160., 0.))
            assert env.hub_active[0].tolist() == [
                hub_is_active(alliance, contract_time, tie_counts,
                              fms_tie_selection=selected)
                for alliance in ("red", "blue")]


def test_inactive_world_match_mask_preserves_phase_and_auto_freeze():
    env = TensorThreeVsThreeEnv(
        num_envs=2, device="cpu", seed=73, horizon=8000,
        randomize=False, fuel_count=96, fuel_proxy_enabled=True,
        simulation_episode_id="mask-episode", simulation_match_id="mask-match",
        simulation_seed=73,
    )
    env.auto_fuel_scores[:] = torch.tensor([[3, 1], [1, 4]])
    env.steps[:] = torch.tensor([1500, 2750])
    env.match_elapsed[:] = torch.tensor([29.98, 54.98])
    env.hub_active[1] = torch.tensor([False, True])

    env._update_match(torch.tensor([True, False]),
        clock_time=torch.tensor([30., 55.]))

    assert env.match_elapsed.tolist() == pytest.approx([30., 54.98])
    assert env.hub_active.tolist() == [[False, True], [False, True]]
    assert env._auto_fuel_frozen.tolist() == [True, False]
    assert env._frozen_auto_fuel_scores.tolist() == [[3, 1], [0, 0]]


def test_partial_reset_clears_only_selected_proxy_world():
    from frc_defense.rebuilt_match_scoring import derive_simulation_tie_selection

    env = TensorThreeVsThreeEnv(
        num_envs=2, device="cpu", seed=74, horizon=32,
        randomize=False, fuel_count=96, fuel_proxy_enabled=True,
        simulation_episode_id="reset-episode", simulation_match_id="reset-match",
        simulation_seed=74,
    )
    before_tie = int(env._fuel_proxy_tie_alliance[0].item())
    env._fuel_proxy_piece_events[0, 0, 0] = 3
    env._fuel_proxy_piece_events[1, 0, 3] = 7
    env._auto_fuel_frozen[:] = True
    env._frozen_auto_fuel_scores[:] = torch.tensor([[2, 1], [1, 2]])
    env.hub_active[1] = torch.tensor([False, True])

    env.reset_done(torch.tensor([True, False]))

    assert env._fuel_proxy_reset_generation.tolist() == [1, 0]
    assert (env._fuel_proxy_piece_events[0] == -1).all()
    assert env._fuel_proxy_piece_events[1, 0, 3].item() == 7
    assert env._auto_fuel_frozen.tolist() == [False, True]
    assert env._frozen_auto_fuel_scores.tolist() == [[0, 0], [1, 2]]
    assert env.hub_active[1].tolist() == [False, True]
    assert env._fuel_proxy_tie_alliance[0].item() != before_tie
    assert env._fuel_proxy_tie_alliance[0].item() == (
        0 if derive_simulation_tie_selection("reset-match", 74,
            world_index=0, reset_generation=1)["alliance"] == "red" else 1)


def test_deterministic_defender_guards_standard_hub_scoring_approach():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=18, horizon=1, randomize=False,
        fuel_count=96, control_modes=("none", "none", "none", "deterministic", "none", "none"),
        robot_roles=("offense", "offense", "offense", "defense", "defense", "defense"),
        perception_dropout=0, position_noise=0, velocity_noise=0,
    )
    # The red opponent is on the far side of the field. The defender should
    # hold the known scoring approach, even when its opponent track is absent.
    env.sim.pose[0, 3, :2] = torch.tensor([10., 4.])
    env.sim.pose[0, 0, :2] = torch.tensor([14., 7.])

    target, action, _, _ = env._target_for_actions(
        torch.full((1, 6), 7), torch.ones(1, dtype=torch.bool))
    red_hub_x = float(env.hub_centers[0, 0])
    red_hub_y = float(env.hub_centers[0, 1])

    assert action[0, 3].item() == 5
    assert target[0, 3, 0].item() < red_hub_x  # block from red's alliance side
    assert target[0, 3, 1].item() == pytest.approx(red_hub_y)
    assert env._action_mask(*env._candidates()[:2])[0][0, 3, 5].item()


def test_defender_intercepts_attacker_even_when_attacker_is_near_hub():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=20, horizon=1, randomize=False,
        fuel_count=96, control_modes=("none", "none", "none", "deterministic", "none", "none"),
        robot_roles=("offense", "offense", "offense", "defense", "defense", "defense"),
        perception_dropout=0, position_noise=0, velocity_noise=0,
    )
    red_hub_x = float(env.hub_centers[0, 0])
    env.opponent_valid[0, 3] = True
    env.opponent_pose[0, 3] = torch.tensor([red_hub_x + 2., env.hub_centers[0, 1], 0.])

    target, action, _, _ = env._target_for_actions(
        torch.full((1, 6), 7), torch.ones(1, dtype=torch.bool))

    assert action[0, 3].item() == 5
    assert (target[0, 3] - env.opponent_pose[0, 3, :2]).norm().item() < 2.1


def test_defender_intercepts_attacker_at_far_and_near_ranges():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=21, horizon=1, randomize=False,
        fuel_count=96, control_modes=("none", "none", "none", "deterministic", "none", "none"),
        robot_roles=("offense", "offense", "offense", "defense", "defense", "defense"),
        perception_dropout=0, position_noise=0, velocity_noise=0,
    )
    env.track_mask[0, 3, 0] = True
    env.track_pos[0, 3, 0] = torch.tensor([13., 1.])
    env.opponent_valid[0, 3] = True
    env.opponent_pose[0, 3] = torch.tensor([1., 1., 0.])
    env.opponent_velocity[0, 3, :2] = torch.tensor([1., 0.])
    env.sim.pose[0, 3, :2] = torch.tensor([10., 4.])
    actions = torch.full((1, 6), 7)
    active = torch.ones(1, dtype=torch.bool)

    target, action, _, _ = env._target_for_actions(actions, active)
    assert action[0, 3].item() == 5
    assert (target[0, 3] - env.opponent_pose[0, 3, :2]).norm().item() < 2.1
    assert target[0, 3, 0].item() > env.opponent_pose[0, 3, 0].item() + .5

    red_hub_x = float(env.hub_centers[0, 0])
    env.opponent_pose[0, 3] = torch.tensor([red_hub_x + 2., env.hub_centers[0, 1], 0.])
    target, action, _, _ = env._target_for_actions(actions, active)
    assert action[0, 3].item() == 5
    assert (target[0, 3] - env.opponent_pose[0, 3, :2]).norm().item() < 2.1


def test_defender_tracks_active_opponent_instead_of_stationary_unassigned_slots():
    modes = ("deterministic", "none", "none", "deterministic", "none", "none")
    roles = ("offense", "offense", "offense", "defense", "defense", "defense")
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=19, horizon=1, randomize=False,
        fuel_count=96, control_modes=modes, robot_roles=roles,
        perception_dropout=0, position_noise=0, velocity_noise=0,
    )
    env.sim.pose[0, 0] = torch.tensor([5., 4., 0.])
    env.sim.pose[0, 1] = torch.tensor([1., 7., 0.])
    env.sim.pose[0, 2] = torch.tensor([1., 6., 0.])
    env.sim.pose[0, 3] = torch.tensor([7., 4., 0.])
    env.sim.pose[0, 4:] = torch.tensor([[15., 1., 0.], [15., 7., 0.]])
    env._update_perception(torch.ones(1, dtype=torch.bool))

    assert env.opponent_valid[0, 3].item()
    assert env.opponent_pose[0, 3, 0].item() == pytest.approx(5.0)
    assert env.opponent_pose[0, 3, 1].item() == pytest.approx(4.0)


def test_shared_policy_infers_six_robots_and_modes_control_each_robot():
    modes = ("none", "nn", "deterministic", "nn", "none", "deterministic")
    roles = ("offense", "defense", "offense", "defense", "defense", "defense")
    env = TensorThreeVsThreeEnv(
        num_envs=1,
        device="cpu",
        seed=91,
        control_modes=modes,
        robot_roles=roles,
        horizon=2,
        randomize=False,
        fuel_count=96,
        perception_dropout=0,
        position_noise=0,
        velocity_noise=0,
        normalize_observations=False,
    )
    obs, info = env.reset(seed=91)

    assert info["robots"] == 6
    assert obs.shape == (1, 6, 137)
    assert torch.isfinite(obs).all()

    policy = ActorCritic(137, 8, "categorical")
    per_robot_obs = obs.reshape(6, 137)
    action_mask = per_robot_obs[:, -8:].bool()
    with torch.no_grad():
        actions, _, _ = policy.sample(per_robot_obs, action_mask=action_mask)

    assert actions.shape == (6,)
    assert torch.all((0 <= actions) & (actions < 8))
    assert torch.all(action_mask.gather(1, actions[:, None]))

    for _ in range(2):
        obs, reward, done, truncated, _ = env.step(actions.reshape(1, 6))

    assert obs.shape == (1, 6, 137)
    assert reward.shape == (1, 6)
    assert torch.isfinite(obs).all()
    assert env.last_actions[0, 0].item() == 7  # none remains a hold action
    assert env.last_actions[0, 4].item() == 7
    assert env.last_actions[0, 1].item() == actions[1].item()
    assert env.last_actions[0, 3].item() == actions[3].item()
    final_action_masks = env.observe()[0, :, -8:].bool()
    assert torch.all(final_action_masks.gather(1, env.last_actions[0, :, None]))
    assert done.tolist() == [False]
    assert truncated.tolist() == [True]
    assert env.steps.tolist() == [2]
    assert env.match_elapsed.item() == pytest.approx(0.04)


def test_three_vs_three_rejects_incomplete_or_unknown_robot_control_modes():
    with pytest.raises(ValueError, match="six robots"):
        TensorThreeVsThreeEnv(
            num_envs=1,
            device="cpu",
            control_modes=("nn", "nn", "nn", "deterministic", "deterministic"),
            horizon=1,
            randomize=False,
            fuel_count=96,
        )
    with pytest.raises(ValueError, match="six robots"):
        TensorThreeVsThreeEnv(
            num_envs=1,
            device="cpu",
            control_modes=("nn", "nn", "nn", "deterministic", "deterministic", "human"),
            horizon=1,
            randomize=False,
            fuel_count=96,
        )


def test_deterministic_offense_and_defense_nn_selection():
    from frc_defense.dashboard import _normalize_robot_control_selections

    selections = (
        "offense_deterministic", "defense_deterministic", "none",
        "defense_nn", "offense_deterministic", "none",
    )
    modes, roles = _normalize_robot_control_selections(selections)
    assert modes == ["deterministic", "deterministic", "none", "nn", "deterministic", "none"]
    assert roles == ["offense", "defense", "offense", "defense", "offense", "defense"]

    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=12, control_modes=modes,
        robot_roles=roles, horizon=1, randomize=False, fuel_count=96,
        perception_dropout=0, position_noise=0, velocity_noise=0,
    )
    assert env.robot_roles == tuple(roles)
    assert env.agent_train_mask.tolist() == [False, False, False, True, False, False]
    assert env._deterministic_mode_mask.tolist() == [True, True, False, False, True, False]
    assert env._defense_role_mask.tolist() == [False, True, False, True, False, True]
