from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from frc_defense.field import grid_array_points
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv
from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense.dashboard_simulation import generate_zone_playback


def test_gamepiece_grid_has_regular_spacing_and_no_duplicates():
    points = grid_array_points(408, 7.355, 9.185, 1.42, 6.65)
    assert len(set(points)) == 408
    for axis in (0, 1):
        coordinates = sorted({p[axis] for p in points})
        gaps = [b-a for a, b in zip(coordinates, coordinates[1:])]
        assert gaps == pytest.approx([gaps[0]] * len(gaps))


def test_both_environments_stage_same_grid_across_seeds_and_masked_resets():
    for env_type in (TensorThreeVsThreeEnv, TensorDefenseEnv):
        env = env_type(num_envs=2, device="cpu", seed=1, fuel_count=504,
                       preloaded_per_robot=0)
        env.reset(seed=1)
        positions = env.piece_pos.clone()
        assert torch.equal(positions[0], positions[1])
        env.reset(seed=42)
        assert torch.equal(positions, env.piece_pos)
        env.piece_pos[1].fill_(-10)
        env.reset_done(torch.tensor([True, False]))
        assert torch.equal(positions[0], env.piece_pos[0])
        assert bool((env.piece_pos[1] == -10).all())


def test_focused_chassis_stays_twenty_four_inches_after_reset():
    env = TensorThreeVsThreeEnv(num_envs=1, device="cpu", behavior_probe="collect",
        control_modes=["deterministic"]+["none"]*5, robot_roles=["offense"]*6)
    for seed in (0, 42):
        env.reset(seed=seed)
        assert env.sim.length[0, 0].item() == pytest.approx(.6096)
        assert env.sim.width[0, 0].item() == pytest.approx(.6096)
        assert env._controlled_mode_mask.sum().item() == 1


def test_dumper_intake_deployment_is_blocked_by_hub_geometry():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", behavior_probe="collect",
        control_modes=["deterministic"] + ["none"] * 5,
        robot_roles=["offense"] * 6,
    )
    env.reset(seed=0)
    hub = next(box for box in env.field_boxes if box.name == "red_hub")
    env.sim.pose[0, 0] = torch.tensor([hub.x - .7, hub.y, 0.])
    assert not env._intake_deployment_clear()[0, 0].item()

    env.sim.pose[0, 0] = torch.tensor([hub.x - 1.4, hub.y, 0.])
    assert env._intake_deployment_clear()[0, 0].item()


def test_inactive_collect_planner_projects_touching_support_start():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=0, randomize=False,
        control_modes=["deterministic"] + ["none"] * 5,
        robot_roles=["offense"] * 6, behavior_probe="collect",
        behavior_probe_hub_active=False, sweeping_enabled=False,
        replan_interval=10,
    )
    env.reset(seed=0)
    # Match the focused collect-inactive probe: no legal sensed fuel remains,
    # so the empty dumper heads toward its ferry target from a start touching
    # the lower trench support.
    env.hub_active[0, 0] = False
    env.track_mask.zero_()
    env.track_age.fill_(float("inf"))
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    support = next(box for box in env.field_boxes
                   if box.name == "red_trench_support_lower")
    env.sim.pose[0, 0] = torch.tensor(
        [3.722868, 1.346589, .010887], dtype=env.sim.pose.dtype)
    env.sim.velocity.zero_()
    no_op = torch.full((1, 6), 7, dtype=torch.long, device=env.device)

    targets, actions, _, _ = env._target_for_actions(no_op, torch.ones(1, dtype=torch.bool))
    assert actions[0, 0].item() == 6
    center = torch.tensor([env.field_length / 2, env.sim.field_width / 2])
    assert targets[0, 0].tolist() == pytest.approx(center.tolist())

    env.planner.plan(
        env.sim.pose[:, :, :2].reshape(-1, 2), targets.reshape(-1, 2),
        env.sim.pose[:, :, 2].reshape(-1), env.sim.length.reshape(-1),
        env.sim.width.reshape(-1), speed=env.sim.speed.reshape(-1),
        lateral_friction=env.sim.lateral_mu.reshape(-1),
        acceleration=env.sim.accel.reshape(-1),
    )
    route = env.planner.last_path[0, :env.planner.last_lengths[0]]
    assert route.shape[0] > 1
    # The start itself is touching the support; every subsequent waypoint
    # must move away from its inflated footprint.
    clearance_x = support.length / 2 + env.sim.length[0, 0] / 2
    clearance_y = support.width / 2 + env.sim.width[0, 0] / 2
    overlaps_support = ((route[1:, 0] - support.x).abs() < clearance_x) & (
        (route[1:, 1] - support.y).abs() < clearance_y)
    assert not overlaps_support.any()

    for _ in range(125):
        env.step(no_op, capture_observation=False)

    assert not env.sim.field_contact[0, 0].item()


def test_active_collect_probe_can_score_from_low_wall_side_pose():
    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=0, randomize=False,
        control_modes=["deterministic"] + ["none"] * 5,
        robot_roles=["offense"] * 6, robot_types=["dumper"] * 6,
        behavior_probe="collect", behavior_probe_hub_active=True,
        sweeping_enabled=False, perception_dropout=0, position_noise=0,
        velocity_noise=0,
    )
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[0, :60] = True
    env.piece_owner[0, :60] = 0
    # This is the low scoring pose reached by the active collect playback.
    # The HUB center and alliance-side face have different bearings here.
    env.sim.pose[0, 0] = torch.tensor([3.584, .678, 1.27])
    env.sim.velocity.zero_()
    env.next_score.zero_()
    no_op = torch.full((1, 6), 7, dtype=torch.long, device=env.device)

    for _ in range(140):
        env.step(no_op, capture_observation=False, capture_info=False)

    assert env.fuel_score_count[0, 0].item() == 60


@pytest.mark.parametrize("behavior_mode", ("collect_active", "collect_inactive"))
def test_collect_probe_uses_real_fuel_and_does_not_synthesize_scores(behavior_mode):
    record = generate_zone_playback(
        Path("."), "focused", "red", "blue", 0, task="3v3",
        control_modes=["offense_deterministic"] + ["none"] * 5,
        robot_types=["dumper"] * 6, hopper_capacity=60, scoring_bps=25.,
        teammate_intent_knowledge=False, sweeping_enabled=False,
        behavior_mode=behavior_mode,
    )
    frames = record["scenarios"][0]["frames"]
    assert record["simulated_seconds"] == pytest.approx(30.0)
    assert frames[9]["match_elapsed"] == pytest.approx(2.0, abs=.03)
    assert frames[-1]["fuel_acquisition_count"][0] > 0
    if behavior_mode == "collect_active":
        assert frames[-1]["fuel_score_count"][0] > 24
        assert frames[-1]["fuel_score_count"][1] == 0
    else:
        assert frames[-1]["fuel_score_count"] == [0, 0]
    assert len(frames[-1]["fuel_pieces"]) == len(frames[0]["fuel_pieces"])
