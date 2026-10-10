"""Fuel dynamics preserve environment cadence, ownership, and game rules."""
import pytest
import torch

from frc_defense.fuel_physics_runtime import advance_with_fuel
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv
from frc_defense.tensor_sim import TensorDefenseEnv


def make_env(kind="3v3", worlds=2):
    kwargs = dict(num_envs=worlds, device="cpu", seed=188, randomize=False,
                  fuel_count=96, perception_dropout=0, position_noise=0,
                  velocity_noise=0, horizon=100)
    if kind == "3v3":
        return TensorThreeVsThreeEnv(**kwargs, control_modes=("deterministic", "none", "none", "none", "none", "none"),
                                    robot_roles=("offense",) * 6, robot_types=("turret",) * 6)
    return TensorDefenseEnv(**kwargs, task="counter_defense", action_mode="strategic")


@pytest.mark.parametrize("kind", ["2robot", "3v3"])
def test_default_environment_steps_moving_fuel_and_preserves_inactive_world(kind):
    env = make_env(kind)
    assert env.fuel_physics is not None
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[:, 0] = True
    env.piece_pos[:, 0] = torch.tensor([8., 3.])
    env.piece_vel[:, 0] = torch.tensor([1., 0.])
    env.fuel_physics.reset()
    env.fuel_physics.sleeping[1, 0] = True
    before = {name: getattr(env.fuel_physics, name)[1].clone()
              for name in ("pos", "vel", "angular", "sleeping")}
    old_dt = env.sim.dt
    action = torch.zeros((2, 6), dtype=torch.long) if kind == "3v3" else torch.zeros(2, dtype=torch.long)
    env.step(action, active_mask=torch.tensor([True, False]))
    assert env.piece_pos[0, 0, 0].item() > 8.
    assert env.sim.dt == old_dt
    for name, expected in before.items():
        torch.testing.assert_close(getattr(env.fuel_physics, name)[1], expected, rtol=0, atol=0)
    reset_other = env.fuel_physics.pos[1].clone()
    env.reset_done(torch.tensor([True, False]))
    torch.testing.assert_close(env.fuel_physics.pos[1], reset_other, rtol=0, atol=0)
    assert not env.fuel_physics.angular[0].any()


@pytest.mark.parametrize("height,collect", [(.075, True), (.5, False)])
def test_pickup_tracks_current_grid_cell_and_excludes_airborne_fuel(height, collect):
    env = make_env(worlds=1)
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[0, 0] = True
    env.piece_pos[0, 0] = torch.tensor([2.5, 2.])
    env.sim.pose[0, 0] = torch.tensor([2., 2., 0.])
    env.sim.pose[0, 1:, :2] = torch.tensor([14., 7.])
    env._pickup_possible_cells.fill_(-1)  # initial bins deliberately stale
    env.match_elapsed.fill_(1.)
    env.fuel_physics.reset()
    env.fuel_physics.pos[0, 0, 2] = height
    env._update_fuel(torch.ones(1, dtype=torch.bool), torch.zeros((1, 6), dtype=torch.bool))
    assert (env.piece_owner[0, 0].item() == 0) is collect


@pytest.mark.parametrize("release", ["score", "pass", "netpass"])
def test_game_release_retains_motion_on_next_physics_tick(release, monkeypatch):
    from frc_defense import gamepiece_actions as gamepieces
    real_launch = gamepieces.launch_fuel
    release_scales = []
    expected_speeds = []
    env = None

    def record_launch(env, selected, destinations, **kwargs):
        if selected.any():
            release_scales.append(kwargs.get("horizontal_velocity_scale", 1.))
            if release == "pass":
                expected_speeds.extend(
                    ((destinations-kwargs["origins"]).norm(dim=-1)*
                     kwargs.get("horizontal_velocity_scale", 1.)/
                     kwargs.get("flight_time", .8))[selected].tolist())
            if release == "score":
                origin = kwargs["origins"]
                midfield = origin.new_tensor([env.sim.field_length*.5, env.sim.field_width*.5])
                axis = midfield-origin
                delta = destinations-origin
                cross = axis[..., 0]*delta[..., 1]-axis[..., 1]*delta[..., 0]
                angle = torch.atan2(cross, (axis*delta).sum(-1)).abs()
                assert (angle[selected] <= torch.pi/8+1e-6).all()
        return real_launch(env, selected, destinations, **kwargs)

    monkeypatch.setattr(gamepieces, "launch_fuel", record_launch)
    if release == "netpass":
        def force_backer_return(current_env, origins, team_ids, *,
                                return_backer_hits=False):
            positions = current_env._midfield_respawn_positions[0].expand_as(origins).clone()
            hits = torch.ones(origins.shape[:-1], dtype=torch.bool,
                              device=origins.device)
            return (positions, hits) if return_backer_hits else positions
        monkeypatch.setattr(gamepieces, "ferry_respawn_targets", force_backer_return)
    env = make_env(worlds=1)
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[0, 0] = True
    env.piece_owner[0, 0] = 0
    env.match_elapsed.fill_(1.)
    env.next_score.zero_()
    env.next_ferry.zero_()
    env.sim.velocity.zero_()
    env.fuel_physics.sleeping[0, 0] = True
    score_intent = torch.zeros((1, 6), dtype=torch.bool)
    if release == "score":
        env.sim.pose[0, 0, :2] = torch.tensor([2., 2.])
        score_intent[0, 0] = True
    else:
        if release == "netpass":
            env.ferry_targets[0, 1] = env.sim.field_width*.5
        env.sim.pose[0, 0, :2] = env.ferry_targets[0]
        env.last_actions[0, 0] = 6
    env._update_fuel(torch.ones(1, dtype=torch.bool), score_intent)
    assert release_scales == ([.1] if release != "score" else [.15])
    assert env.piece_owner[0, 0].item() == -1
    event = env.fuel_scored_event if release == "score" else env.fuel_passed_event
    assert event[0, 0].item() == 1
    start_height = env.fuel_physics.pos[0, 0, 2].item()
    start_xy = env.piece_pos[0, 0].clone()
    if release == "score":
        assert start_height > .2
        assert env.fuel_physics.vel[0, 0, 2].item() > 0
        hub=env.hub_centers[env.team_ids[0]]
        midfield=start_xy.new_tensor([env.sim.field_length/2,
                                       env.sim.field_width/2])
        assert start_xy[0] > (hub[0]+47.*.0254/2+
                              env.fuel_physics.config.radius)
        assert (env.fuel_physics.vel[0, 0, :2]*(midfield-hub)).sum() > 0
    else:
        if release == "netpass":
            assert start_xy[0] == pytest.approx(env.sim.field_length*.5, abs=1.65)
            assert start_height == pytest.approx(1.83)
            assert env.fuel_physics.vel[0, 0, 2].item() > 0
            assert env.piece_zone[0, 0].item() == 0
            assert env.fuel_physics.vel[0, 0, 0].item() > 0
        else:
            assert start_height == pytest.approx(env.fuel_physics.config.radius)
            assert env.fuel_physics.vel[0, 0, 2].item() == 0
            assert start_xy[0] < env.alliance_zone_depth
            assert env.piece_zone[0, 0].item() == 1
            away_from_robot=start_xy-env.sim.pose[0, 0, :2]
            velocity=env.fuel_physics.vel[0, 0, :2]
            assert (velocity*away_from_robot).sum() > 0
            expected_speed=away_from_robot.norm()*.1/.5
            assert velocity.norm().item() == pytest.approx(expected_speed.item())
    assert not env.fuel_physics.sleeping[0, 0]
    assert env.piece_vel[0, 0].norm().item() > .1
    env.fuel_physics.step(torch.ones(1, dtype=torch.bool), env.piece_active, env.piece_owner)
    if release == "score":
        assert env.fuel_physics.pos[0, 0, 2].item() > start_height
    assert not torch.equal(env.piece_pos[0, 0], start_xy)


def test_substep_runtime_restores_control_dt_after_failure(monkeypatch):
    env = make_env(worlds=1)
    old_dt = env.sim.dt

    def fail(*args, **kwargs):
        raise RuntimeError("intentional contact failure")

    monkeypatch.setattr(env.fuel_physics, "step", fail)
    with pytest.raises(RuntimeError, match="intentional contact failure"):
        advance_with_fuel(env, torch.zeros_like(env.sim.velocity), torch.ones(1, dtype=torch.bool))
    assert env.sim.dt == old_dt


def test_runtime_does_not_substep_for_inactive_world_speeds(monkeypatch):
    env = make_env()
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[1, 0] = True
    env.piece_vel[1, 0, 0] = 1000.
    env.sim.velocity[1, :, 0] = 1000.
    env.fuel_physics.reset()
    calls = []
    original = env.sim.step

    def record_step(*args, **kwargs):
        calls.append(env.sim.dt)
        return original(*args, **kwargs)

    monkeypatch.setattr(env.sim, "step", record_step)
    old_dt = env.sim.dt
    advance_with_fuel(env, torch.zeros_like(env.sim.velocity), torch.tensor([True, False]))
    assert len(calls) == env.fuel_physics.config.substeps
    assert sum(calls) == pytest.approx(old_dt)
    assert env.sim.dt == old_dt


@pytest.mark.parametrize("other_ball_launched", [False, True])
def test_empty_or_partial_launch_does_not_hide_pending_outpost_spawn(other_ball_launched):
    from types import SimpleNamespace
    from frc_defense.fuel_physics_runtime import launch_fuel
    from test_fuel_physics import make_world, advance

    world = make_world(pieces=2)
    physics = world[0]
    env = SimpleNamespace(fuel_physics=physics, piece_pos=physics.piece_pos, piece_vel=physics.piece_vel)
    # Outpost/game-rule spawning writes the planar API before its next sync.
    env.piece_pos[0, 1] = torch.tensor([8., 3.])
    env.piece_vel[0, 1] = torch.tensor([2., 0.])
    selected = torch.tensor([[other_ball_launched, False]])
    destinations = torch.tensor([[[12., 3.], [12., 3.]]])
    launch_fuel(env, selected, destinations)
    advance(world)
    assert physics.pos[0, 1, 0].item() > 8.
    assert physics.vel[0, 1, 0].item() > 0
    torch.testing.assert_close(env.piece_pos[0, 1], physics.pos[0, 1, :2])


@pytest.mark.parametrize("device", ["cpu"] +
                         ([f"cuda:{i}" for i in range(torch.cuda.device_count())]
                          if torch.version.hip else []))
def test_ferry_velocity_multiplier_preserves_vertical_and_unselected_state(device):
    from types import SimpleNamespace
    from frc_defense.fuel_physics_runtime import launch_fuel
    from test_fuel_physics import make_world

    world = make_world(pieces=2, device=device,
                       backend="torch" if device == "cpu" else "hip")
    physics = world[0]
    env = SimpleNamespace(fuel_physics=physics, piece_pos=physics.piece_pos,
                          piece_vel=physics.piece_vel)
    selected = torch.tensor([[True, False]], device=device)
    target = env.piece_pos + torch.tensor([4., 2.], device=device)
    physics.angular[0, 1] = torch.tensor([1., 2., 3.], device=device)
    untouched = {name: getattr(physics, name)[0, 1].clone()
                 for name in ("pos", "vel", "angular")}
    launch_fuel(env, selected, target, horizontal_velocity_scale=.1)
    torch.testing.assert_close(physics.vel[0, 0, :2],
                               torch.tensor([.5, .25], device=device))
    expected_vertical = (.075-.35)/.8 + .5*physics.config.gravity*.8
    assert physics.vel[0, 0, 2].item() == pytest.approx(expected_vertical)
    for name, expected in untouched.items():
        torch.testing.assert_close(getattr(physics, name)[0, 1], expected)


@pytest.mark.parametrize("device", ["cpu"] +
                         ([f"cuda:{i}" for i in range(torch.cuda.device_count())]
                          if torch.version.hip else []))
def test_shooter_respawn_cone_spreads_both_alliances_and_preserves_speed(device):
    from types import SimpleNamespace
    from frc_defense.fuel_physics_runtime import launch_fuel, shooter_respawn_cone
    from test_fuel_physics import make_world

    physics, sim, *_ = make_world(worlds=2, pieces=64, device=device,
                                 backend="torch" if device == "cpu" else "hip")
    origins = torch.tensor([[2., sim.field_width*.5],
                            [sim.field_length-2., sim.field_width*.5]], device=device)
    origins = origins[:, None].expand(-1, 64, -1)
    midfield = origins.new_tensor([sim.field_length*.5, sim.field_width*.5])
    env = SimpleNamespace(sim=sim, fuel_physics=physics,
                          piece_pos=physics.piece_pos, piece_vel=physics.piece_vel,
                          generator=torch.Generator(device=device).manual_seed(42),
                          _midfield_respawn_positions=midfield.expand(64, -1))
    rng = env.generator.get_state()
    spawn, target = shooter_respawn_cone(env, origins)
    env.generator.set_state(rng)
    spawn_again, target_again = shooter_respawn_cone(env, origins)
    torch.testing.assert_close(spawn_again, spawn)
    torch.testing.assert_close(target_again, target)
    delta = target-origins
    axis = midfield-origins
    angle = torch.atan2(axis[..., 0]*delta[..., 1]-axis[..., 1]*delta[..., 0],
                        (axis*delta).sum(-1))
    assert (angle.abs() <= torch.pi/8+1e-6).all()
    assert (angle.min(-1).values < -torch.pi/12).all()
    assert (angle.max(-1).values > torch.pi/12).all()
    torch.testing.assert_close(delta.norm(dim=-1), axis.norm(dim=-1))
    selected = torch.ones((2, 64), dtype=torch.bool, device=device)
    launch_fuel(env, selected, target, origins=origins, height=1.83,
                flight_time=1., horizontal_velocity_scale=.1,
                spawn_positions=spawn)
    torch.testing.assert_close(physics.pos[..., :2], spawn)
    torch.testing.assert_close(physics.vel[..., :2].norm(dim=-1), axis.norm(dim=-1)*.1)


def ferry_test_env(device):
    from types import SimpleNamespace
    from frc_defense.field import ALLIANCE_ZONE_DEPTH, INCH, midfield_respawn_points

    sim = SimpleNamespace(field_length=16.54, field_width=8.21)
    hub_x = ALLIANCE_ZONE_DEPTH+47.*INCH*.5
    return SimpleNamespace(
        sim=sim, alliance_zone_depth=ALLIANCE_ZONE_DEPTH,
        fuel_physics=SimpleNamespace(config=SimpleNamespace(radius=.075)),
        hub_centers=torch.tensor([[hub_x, sim.field_width*.5],
                                  [sim.field_length-hub_x, sim.field_width*.5]], device=device),
        generator=torch.Generator(device=device).manual_seed(19),
        _midfield_respawn_positions=torch.tensor(
            midfield_respawn_points(96, sim.field_length, sim.field_width), device=device))


@pytest.mark.parametrize("device", ["cpu"] +
                         ([f"cuda:{i}" for i in range(torch.cuda.device_count())]
                          if torch.version.hip else []))
def test_ferry_cone_lands_in_friendly_zone_for_both_alliances(device):
    from frc_defense.fuel_physics_runtime import ferry_aim_targets, ferry_respawn_targets

    env = ferry_test_env(device)
    origins = torch.tensor([[8., 1.], [8.54, 7.21]], device=device)
    origins = origins[:, None].expand(-1, 64, -1)
    teams = torch.tensor([[0], [1]], device=device).expand(2, 64)
    targets = ferry_respawn_targets(env, origins, teams)
    assert (targets[0, :, 0] < env.alliance_zone_depth).all()
    assert (targets[1, :, 0] > env.sim.field_length-env.alliance_zone_depth).all()
    center = ferry_aim_targets(env, origins, teams)
    axis, delta = center-origins, targets-origins
    angle = torch.atan2(axis[..., 0]*delta[..., 1]-axis[..., 1]*delta[..., 0],
                        (axis*delta).sum(-1))
    assert (angle.abs() <= torch.pi/36+1e-6).all()
    assert (angle.min(-1).values < -torch.pi/60).all()
    assert (angle.max(-1).values > torch.pi/60).all()
    hub = env.hub_centers[teams]
    face = hub[..., 0] + torch.where(teams == 0, 47.*.0254/2, -47.*.0254/2)
    delta = targets-origins
    fraction = (face-origins[..., 0])/delta[..., 0]
    crossing_y = origins[..., 1]+fraction*delta[..., 1]
    assert ((crossing_y-hub[..., 1]).abs() > .6).all()


def test_repeated_ferry_releases_spawn_in_friendly_alliance_zone():
    from frc_defense.fuel_physics_runtime import ferry_aim_targets

    env = TensorThreeVsThreeEnv(
        num_envs=1, device="cpu", seed=7, randomize=False, fuel_count=160,
        control_modes=("deterministic", "none", "none", "none", "none", "none"),
        robot_roles=("offense",)*6, perception_dropout=0,
        position_noise=0, velocity_noise=0)
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[0, :12] = True
    env.piece_owner[0, :12] = 0
    env.hub_active[0, 0] = False
    env.sim.pose[0, 0, :2] = env.ferry_targets[0]
    aim = ferry_aim_targets(
        env, env.sim.pose[:, :, :2], env.team_ids[None].expand(1, -1))
    delta = aim[0, 0]-env.sim.pose[0, 0, :2]
    env.sim.pose[0, 0, 2] = torch.atan2(delta[1], delta[0])
    env.match_elapsed.fill_(1.)
    env.next_ferry.zero_()
    env.last_actions[0, 0] = 6
    active = torch.ones(1, dtype=torch.bool)
    no_score = torch.zeros((1, 6), dtype=torch.bool)

    peak_speed = 0.
    release_positions = []
    ticks = 0
    while (env.piece_owner[0, :12] >= 0).any() and ticks < 500:
        held_before = env.piece_owner[0, :12] >= 0
        env._update_fuel(active, no_score)
        released = held_before & (env.piece_owner[0, :12] < 0)
        if released.any():
            positions = env.piece_pos[0, :12][released].clone()
            velocities = env.fuel_physics.vel[0, :12, :2][released].clone()
            release_positions.append(positions)
        env.fuel_physics.step(active, env.piece_active, env.piece_owner)
        peak_speed = max(peak_speed,
                         env.fuel_physics.vel[0, :12, :2].norm(dim=-1).max().item())
        env.match_elapsed += env.sim.dt
        ticks += 1

    assert (env.piece_owner[0, :12] == -1).all()
    assert peak_speed < 1.3, f"crowded ferry releases reached {peak_speed:.2f} m/s"
    spawned = torch.cat(release_positions)
    assert (spawned[:, 0] < env.alliance_zone_depth).all()
    assert (spawned[:, 0] < env.sim.field_length/2).all()
    assert spawned[:, 1].std() > .15
    landed = env.piece_pos[0, :12]
    # Released FUEL travels through the randomized midfield-facing HUB region.
    for _ in range(12):
        env.fuel_physics.step(active, env.piece_active, env.piece_owner)
    free = env.piece_owner[0, :12] < 0
    positions = env.fuel_physics.pos[0, :12][free]
    distances = torch.cdist(positions, positions)+torch.eye(positions.shape[0])*100
    assert (distances >= 2*env.fuel_physics.config.radius*.75).all()


@pytest.mark.parametrize("device", ["cpu"] +
                         ([f"cuda:{i}" for i in range(torch.cuda.device_count())]
                          if torch.version.hip else []))
def test_ferry_outside_field_appears_at_edge_and_only_gap_between_bumps_returns_midfield(device):
    from frc_defense.fuel_physics_runtime import resolve_ferry_landings

    env = ferry_test_env(device)
    length, width = env.sim.field_length, env.sim.field_width
    origins = torch.tensor([[[3., 1.], [3., 7.], [length-3., 1.],
                             [length*.5, width*.5], [length*.5, width*.5],
                             [length*.5, 1.]]], device=device)
    targets = torch.tensor([[[-2., -1.], [-1., width+2.], [length+2., -1.],
                             [2., width*.5], [length-2., width*.5], [2., 1.]]], device=device)
    teams = torch.tensor([[0, 0, 1, 0, 1, 0]], device=device)
    result = resolve_ferry_landings(env, origins, targets, teams)
    radius = env.fuel_physics.config.radius
    bottom_fraction = (1.-radius)/2.
    top_fraction = (width-radius-7.)/(width+2.-7.)
    expected_edges = torch.tensor([[3.-5.*bottom_fraction, radius],
                                   [3.-4.*top_fraction, width-radius],
                                   [length-3.+5.*bottom_fraction, radius]], device=device)
    torch.testing.assert_close(result[0, :3], expected_edges)
    side_origins = torch.tensor([[[3., 1.], [length-3., 7.]]], device=device)
    side_targets = torch.tensor([[[-2., 1.], [length+2., 7.]]], device=device)
    side_result = resolve_ferry_landings(
        env, side_origins, side_targets, torch.tensor([[0, 1]], device=device))
    torch.testing.assert_close(side_result,
                               torch.tensor([[[radius, 1.], [length-radius, 7.]]], device=device))
    for index in (3, 4):
        assert (env._midfield_respawn_positions == result[0, index]).all(-1).any()
    # A crossing outside the bump gap does not hit the backer net.
    torch.testing.assert_close(result[0, 5], targets[0, 5])
    torch.testing.assert_close(result[0, 5], targets[0, 5])
