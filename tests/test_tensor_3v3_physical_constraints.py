import math

import pytest

torch = pytest.importorskip("torch")

from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def test_default_robot_limits_are_sixty_fuel_and_twenty_five_bps():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=13, horizon=1,
        randomize=False, fuel_count=96,
    )
    assert env.fuel_capacity == 60
    assert env.max_scoring_bps == 25.0
    assert env.score_interval == pytest.approx(.04)


def test_pickup_respects_per_robot_hopper_capacity():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=14, horizon=1, randomize=False,
        fuel_count=96, max_fuel_capacity=8, perception_dropout=0,
        position_noise=0, velocity_noise=0,
    )
    env.sim.pose[0, 0] = torch.tensor([2.0, 2.0, 0.0])
    env.sim.pose[0, 1:] = torch.tensor([
        [8.0, .5, 0.0], [8.0, 7.5, 0.0], [12.0, 1.0, 0.0],
        [12.0, 7.0, 0.0], [15.0, 4.0, 0.0],
    ])
    env.piece_active[0, :9] = True
    env.piece_owner[0, :8] = 0
    env.piece_pos[0, 8] = torch.tensor([2.55, 2.0])
    env.match_elapsed.fill_(1.0)

    env._update_fuel(torch.ones(1, dtype=torch.bool, device=env.device),
                     torch.zeros((1, 6), dtype=torch.bool, device=env.device))

    assert env.fuel_capacity == 8
    assert env.piece_owner[0, 8].item() == -1
    assert env.fuel_acquired_event[0, 0].item() == 0


def test_pickup_and_pass_clear_tracks_once_without_touching_inactive_worlds():
    env = TensorThreeVsThreeEnv(
        num_envs=2, device="cpu", seed=140, horizon=1, randomize=False,
        fuel_count=96, control_modes=("deterministic", "none", "none", "none", "none", "none"),
        perception_dropout=0, position_noise=0, velocity_noise=0,
    )
    active = torch.tensor([True, False], device=env.device)
    env.match_elapsed.fill_(1.)
    env.next_intake.zero_()
    env.sim.pose[0, 0] = torch.tensor([2., 2., 0.])
    env.piece_active[0, 0] = True
    env.piece_owner[0, 0] = -1
    env.piece_pos[0, 0] = torch.tensor([2.4, 2.])
    env.piece_active[0, 1] = True
    env.piece_owner[0, 1] = 3
    env.last_actions[0, 3] = 6
    env.sim.pose[0, 3, :2] = env.ferry_targets[3]

    env.track_mask.zero_()
    env.track_age.fill_(float("inf"))
    env.track_mask[0, :, 0] = True  # picked
    env.track_mask[0, :, 1] = True  # passed
    env.track_mask[:, :, 2] = True  # unrelated, including inactive world
    env.track_age[env.track_mask] = 0.
    env.fuel_passed_event[1].fill_(9)
    inactive_mask_before = env.track_mask[1].clone()
    inactive_age_before = env.track_age[1].clone()
    score_intent = torch.zeros((2, 6), dtype=torch.bool, device=env.device)

    env._update_fuel(active, score_intent)

    assert env.piece_owner[0, 0].item() == 0
    assert env.piece_owner[0, 1].item() == -1
    assert not env.track_mask[0, :, 0].any()
    assert not env.track_mask[0, :, 1].any()
    assert torch.isinf(env.track_age[0, :, :2]).all()
    assert env.track_mask[0, :, 2].all()
    assert (env.track_age[0, :, 2] == 0.).all()
    assert torch.equal(env.track_mask[1], inactive_mask_before)
    assert torch.equal(env.track_age[1], inactive_age_before)
    assert (env.fuel_passed_event[1] == 9).all()


@pytest.mark.parametrize(("hub_active", "fuel_x", "should_collect"), [
    (True, 2.4, True),       # Active HUB: friendly zone is valid collection area.
    (False, 2.4, False),     # Inactive HUB: friendly zone is prohibited.
    (False, 8.0, True),      # Inactive HUB: neutral field remains collectible.
])
def test_deterministic_collector_zone_eligibility(hub_active, fuel_x,
                                                    should_collect):
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=141, horizon=1, randomize=False,
        fuel_count=96, control_modes=("deterministic", "none", "none",
                                      "none", "none", "none"),
        robot_roles=("offense",) * 6,
        perception_dropout=0, position_noise=0, velocity_noise=0,
    )
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[0, 0] = True
    env.piece_pos[0, 0] = torch.tensor([fuel_x, 2.0])
    env.track_mask.zero_()
    env.track_mask[0, :, 0] = True
    env.track_pos[0, :, 0] = torch.tensor([fuel_x, 2.0])
    env.track_age.fill_(float("inf"))
    env.track_age[0, :, 0] = 0.
    env.sim.pose[0, 0] = torch.tensor([fuel_x - .55, 2.0, 0.0])
    env.hub_active[0, 0] = hub_active
    env.match_elapsed.fill_(1.)
    env.next_intake.zero_()
    active = torch.ones(1, dtype=torch.bool, device=env.device)

    _, _, _, _ = env._target_for_actions(torch.full((1, 6), 7), active)

    env._update_fuel(active, torch.zeros((1, 6), dtype=torch.bool,
                                         device=env.device))
    assert (env.piece_owner[0, 0].item() == 0) is should_collect


def test_scoring_respects_configured_balls_per_second_limit():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=15, horizon=1, randomize=False,
        fuel_count=96, max_fuel_capacity=8, max_scoring_bps=5.0,
        perception_dropout=0, position_noise=0, velocity_noise=0,
    )
    hub = env.hub_centers[0]
    env.sim.pose[0, 0] = torch.tensor([hub[0] - .1, hub[1], 0.0])
    env.piece_active[0, :2] = True
    env.piece_owner[0, :2] = 0
    env.match_elapsed.fill_(1.0)
    active = torch.ones(1, dtype=torch.bool, device=env.device)
    score_intent = torch.zeros((1, 6), dtype=torch.bool, device=env.device)
    score_intent[0, 0] = True

    env._update_fuel(active, score_intent)
    assert env.fuel_score_count[0, 0].item() == 1
    assert env.next_score[0, 0].item() == pytest.approx(1.2)

    # Calling the simulation at the same match time cannot exceed the 5 BPS cap.
    env._update_fuel(active, score_intent)
    assert env.fuel_score_count[0, 0].item() == 1

    env.match_elapsed.fill_(1.2)
    env._update_fuel(active, score_intent)
    assert env.fuel_score_count[0, 0].item() == 2


def test_collect_probe_scores_full_hopper_and_resets_balls_to_midfield():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=151, horizon=1, randomize=False,
        behavior_probe="collect", behavior_probe_robot=0,
        control_modes=("deterministic", "none", "none", "none", "none", "none"),
        robot_roles=("offense",) * 6, max_fuel_capacity=2,
        perception_dropout=0, position_noise=0, velocity_noise=0,
    )
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[0, :2] = True
    env.piece_owner[0, :2] = 0
    env.match_elapsed.fill_(25.)
    active = torch.ones(1, dtype=torch.bool, device=env.device)

    env._update_fuel(active, torch.zeros((1, 6), dtype=torch.bool,
                                         device=env.device))

    assert env.fuel_score_count[0, 0].item() == 2
    assert env.fuel_scored_event[0, 0].item() == 2
    assert torch.equal(env.piece_pos[0, :2], env._midfield_respawn_positions[:2])
    assert (env.piece_owner[0, :2] == -1).all()


def test_robot_finishes_scoring_hopper_before_leaving_hub():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=16, horizon=1, randomize=False,
        fuel_count=96, max_fuel_capacity=8, perception_dropout=0,
        control_modes=("deterministic", "none", "none", "none", "none", "none"),
        position_noise=0, velocity_noise=0,
    )
    hub = env.hub_centers[0]
    env.sim.pose[0, 0] = torch.tensor([hub[0] - .1, hub[1], 0.0])
    env.piece_active[0, :7] = True
    env.piece_owner[0, :7] = 0
    env.hub_active[0, 0] = True

    _, action, _, _ = env._target_for_actions(
        torch.full((1, 6), 7, dtype=torch.long),
        torch.ones(1, dtype=torch.bool),
    )

    assert action[0, 0].item() == 4


def test_deterministic_offense_ferries_full_batch_when_hub_is_inactive():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=18, horizon=1, randomize=False,
        fuel_count=96, max_fuel_capacity=60, perception_dropout=0,
        position_noise=0, velocity_noise=0,
        control_modes=("deterministic", "none", "none", "deterministic", "none", "none"),
    )
    env.piece_active.zero_()
    env.piece_active[0, :24] = True
    env.piece_owner[0, :24] = 0
    env.piece_active[0, 24:48] = True
    env.piece_owner[0, 24:48] = 3
    env.hub_active[0, 0] = False
    env.hub_active[0, 1] = False
    active = torch.ones(1, dtype=torch.bool, device=env.device)
    _, actions, _, _ = env._target_for_actions(torch.full((1, 6), 7), active)
    assert actions[0, 0].item() == 6
    assert actions[0, 3].item() == 6

    env.hub_active[0, :2] = True
    _, actions, _, _ = env._target_for_actions(torch.full((1, 6), 7), active)
    assert actions[0, 0].item() == 4
    assert actions[0, 3].item() == 4

    env.piece_owner[0, 6:24] = -1
    env.piece_owner[0, 30:48] = -1
    env.piece_active[0, 6:24] = False
    env.piece_active[0, 30:48] = False
    env.match_elapsed.fill_(150.)
    _, actions, _, _ = env._target_for_actions(torch.full((1, 6), 7), active)
    assert actions[0, 0].item() == 4
    assert actions[0, 3].item() == 4


def test_ferry_action_drops_carried_fuel_as_spread_alliance_zone_pieces():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=17, horizon=1, randomize=False,
        fuel_count=96, max_fuel_capacity=8, perception_dropout=0,
        position_noise=0, velocity_noise=0,
    )
    active = torch.ones(1, dtype=torch.bool, device=env.device)
    env.piece_active.zero_()
    env.track_mask.zero_()
    env.piece_active[0, :6] = True
    env.piece_owner[0, :3] = 0
    env.piece_owner[0, 3:6] = 3
    env.hub_active[0, :2] = False
    env.sim.pose[0, 0, :2] = env.ferry_targets[0]
    env.sim.pose[0, 3, :2] = env.ferry_targets[3]
    for robot, team in ((0, 0), (3, 1)):
        home_x = (env.alliance_zone_depth * .5 if team == 0 else
                  env.field_length - env.alliance_zone_depth * .5)
        delta = torch.tensor([home_x, env.hub_centers[team, 1]],
                             device=env.device) - env.sim.pose[0, robot, :2]
        env.sim.pose[0, robot, 2] = torch.atan2(delta[1], delta[0])
    env.match_elapsed.fill_(1.)
    _, actions, _, _ = env._target_for_actions(
        torch.tensor([[6, 7, 7, 7, 7, 7]]), active)
    assert actions[0, 0].item() == 6
    env.last_actions[0, 0] = 6
    env.last_actions[0, 3] = 6

    env._update_fuel(active, torch.zeros((1, 6), dtype=torch.bool, device=env.device))

    assert env.fuel_passed_event[0, 0].item() == 1
    assert env.fuel_passed_event[0, 3].item() == 1
    assert env.piece_owner[0, :6].tolist() == [-1, 0, 0, -1, 3, 3]
    assert env.piece_active[0, :6].all()
    assert env.piece_pos[0, 0, 0].item() < env.sim.field_length / 2
    assert env.piece_pos[0, 3, 0].item() > env.sim.field_length / 2
    assert torch.unique(env.piece_pos[0, [0, 3]], dim=0).shape[0] == 2
    assert env.fuel_score_count[0, 0].item() == 0


def test_collect_active_probe_uses_shoot_action_for_full_hopper():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=19, horizon=1, randomize=False,
        fuel_count=504, control_modes=("deterministic", "none", "none",
                                       "none", "none", "none"),
        robot_roles=("offense",) * 6, behavior_probe="collect",
        behavior_probe_hub_active=True, perception_dropout=0,
        position_noise=0, velocity_noise=0,
    )
    assert env._probe_respawn_capacity == 0
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[0, :60] = True
    env.piece_owner[0, :60] = 0
    env.sim.pose[0, 0, :2] = env.ferry_targets[0]
    home = env.hub_centers[0].clone()
    home[0] = env.alliance_zone_depth * .5
    delta = home - env.sim.pose[0, 0, :2]
    env.sim.pose[0, 0, 2] = torch.atan2(delta[1], delta[0])
    env.match_elapsed.fill_(1.)
    _, actions, _, _ = env._target_for_actions(
        torch.full((1, 6), 7, dtype=torch.long, device=env.device),
        torch.ones(1, dtype=torch.bool, device=env.device))
    assert actions[0, 0].item() == 4

    # The dumper keeps shooting after the first ball leaves the full hopper.
    env.piece_owner[0, 0] = -1
    _, actions, _, _ = env._target_for_actions(
        torch.full((1, 6), 7, dtype=torch.long, device=env.device),
        torch.ones(1, dtype=torch.bool, device=env.device))
    assert actions[0, 0].item() == 4

    env._update_fuel(torch.ones(1, dtype=torch.bool, device=env.device),
                     torch.zeros((1, 6), dtype=torch.bool, device=env.device))

    assert env.fuel_passed_event[0, 0].item() == 0
    assert env.piece_owner[0, :60].eq(0).all()


def test_collect_inactive_probe_latches_ferry_until_hopper_is_empty():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=20, horizon=1, randomize=False,
        fuel_count=504, control_modes=("deterministic", "none", "none",
                                       "none", "none", "none"),
        robot_roles=("offense",) * 6, behavior_probe="collect",
        behavior_probe_hub_active=False, perception_dropout=0,
        position_noise=0, velocity_noise=0,
    )
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[0, :61] = True
    env.piece_owner[0, :60] = 0
    env.piece_pos[0, 60] = torch.tensor([8.0, 4.0])
    env.track_mask.zero_()
    env.track_age.fill_(float("inf"))
    env.track_mask[0, 0, 60] = True
    env.track_pos[0, 0, 60] = env.piece_pos[0, 60]
    env.hub_active[0, 0] = False
    env.sim.pose[0, 0] = torch.tensor([6.0, 4.0, math.pi])
    env.next_ferry.zero_()
    env.match_elapsed.fill_(1.)
    active = torch.ones(1, dtype=torch.bool, device=env.device)
    no_action = torch.full((1, 6), 7, dtype=torch.long, device=env.device)
    no_score = torch.zeros((1, 6), dtype=torch.bool, device=env.device)

    _, actions, _, _ = env._target_for_actions(no_action, active)
    assert actions[0, 0].item() == 6
    assert env._ferry_committed[0, 0]

    env._update_fuel(active, no_score)
    assert env.fuel_passed_event[0, 0].item() == 1
    assert (env.piece_owner[0] == 0).sum().item() == 59
    _, actions, _, _ = env._target_for_actions(no_action, active)
    assert actions[0, 0].item() == 6
    assert env._ferry_committed[0, 0]

    env.piece_owner[0, env.piece_owner[0] == 0] = -1
    _, actions, _, _ = env._target_for_actions(no_action, active)
    assert actions[0, 0].item() == 0
    assert not env._ferry_committed[0, 0]


@pytest.mark.parametrize("lane_y,initial_heading", [
    (.8, math.pi / 4), (.8, -math.pi / 4),
    (7.27, math.pi / 4), (7.27, -math.pi / 4),
])
def test_deterministic_offense_rotates_then_intakes_through_trench(
        lane_y, initial_heading):
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=21, horizon=300, randomize=False,
        fuel_count=96, perception_dropout=0, position_noise=0, velocity_noise=0,
        control_modes=("deterministic", "none", "none", "none", "none", "none"),
        robot_roles=("offense",) * 6, replan_interval=5,
    )
    env.sim.length[0, 0] = 1.1
    env.sim.width[0, 0] = .7
    env.sim.pose[0, 0] = torch.tensor([3., lane_y, initial_heading])
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[0, 0] = True
    env.piece_pos[0, 0] = torch.tensor([6., lane_y])
    env.track_mask.zero_()
    env.track_mask[0, 0, 0] = True
    env.track_pos[0, 0, 0] = torch.tensor([6., lane_y])
    env.track_age.fill_(float("inf"))
    env.track_age[0, 0, 0] = 0.
    trench_x = env.hub_centers[0, 0]
    actions = torch.tensor([[0, 7, 7, 7, 7, 7]])

    env.step(actions, capture_observation=False)
    used_trench_alignment = bool(env.planner.last_trench_alignment[0])
    assert env.sim.pose[0, 0, :2].tolist() == pytest.approx([3., lane_y])
    assert abs(float(env.sim.pose[0, 0, 2])) < abs(initial_heading)
    crossed = False
    for _ in range(180):
        env.step(actions, capture_observation=False)
        used_trench_alignment |= bool(env.planner.last_trench_alignment[0])
        crossed |= bool(env.sim.pose[0, 0, 0] > trench_x)
        if env.piece_owner[0, 0].item() == 0:
            break

    assert crossed
    assert used_trench_alignment
    assert env.piece_owner[0, 0].item() == 0
