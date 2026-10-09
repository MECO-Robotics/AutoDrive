"""Physical invariants and optimization parity for moving FUEL contacts."""
from types import SimpleNamespace

import pytest
import torch

from frc_defense.fuel_physics import FuelPhysics, FuelPhysicsConfig


def make_world(*, worlds=1, pieces=2, robots=1, device="cpu", bump_regions=(), **options):
    device = torch.device(device)
    sim = SimpleNamespace(
        dt=.02, field_length=16.54, field_width=8.21,
        pose=torch.zeros((worlds, robots, 3), device=device),
        velocity=torch.zeros((worlds, robots, 3), device=device),
        length=torch.full((worlds, robots), .9, device=device),
        width=torch.full((worlds, robots), .9, device=device),
        mass=torch.full((worlds, robots), 55., device=device),
        yaw_inertia_multiplier=torch.full((worlds, robots), 1.5, device=device),
        field_colliders=torch.empty((0, 4), device=device),
        bump_regions=torch.as_tensor(bump_regions, dtype=torch.float32, device=device).reshape(-1, 4),
    )
    # Keep robot geometry away from the isolated particle test region.
    sim.pose[..., :2] = torch.tensor([14., 7.], device=device)
    pos = torch.zeros((worlds, pieces, 2), device=device)
    pos[..., 0] = 3. + torch.arange(pieces, device=device) * .4
    pos[..., 1] = 3.
    vel = torch.zeros_like(pos)
    cfg = FuelPhysicsConfig(**{"backend": "torch", **options})
    physics = FuelPhysics(sim, pos, vel, config=cfg)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    piece_active = torch.ones((worlds, pieces), dtype=torch.bool, device=device)
    owner = torch.full((worlds, pieces), -1, dtype=torch.long, device=device)
    return physics, sim, active, piece_active, owner


def advance(world, ticks=1):
    physics, _, active, piece_active, owner = world
    for _ in range(ticks):
        physics.step(active, piece_active, owner)


def test_fuel_rolls_and_loses_speed_on_carpet():
    world = make_world(pieces=1)
    physics = world[0]
    physics.vel[0, 0, 0] = 1.
    initial_x = physics.pos[0, 0, 0].item()
    advance(world)
    assert physics.angular[0, 0].norm().item() > 0
    advance(world, 24)
    assert physics.pos[0, 0, 0].item() > initial_x
    assert 0 <= physics.vel[0, 0, 0].item() < 1.
    assert physics.pos[0, 0, 2].item() >= physics.config.radius * .5


def test_default_carpet_rolling_resistance_settles_released_fuel_quickly():
    world = make_world(pieces=1)
    physics = world[0]
    physics.vel[0, 0, 0] = .5
    advance(world, 40)
    assert physics.vel[0, 0, :2].norm().item() < .04


def test_airborne_fuel_falls_and_rebounds():
    world = make_world(pieces=1)
    physics = world[0]
    physics.pos[0, 0, 2] = .5
    rebound_speeds = []
    for _ in range(60):
        advance(world)
        if physics.vel[0, 0, 2].item() > .05:
            rebound_speeds.append(physics.vel[0, 0, 2].item())
        assert torch.isfinite(physics.pos).all()
        assert physics.pos[0, 0, 2].item() >= physics.config.radius * .5
    assert rebound_speeds, "floor contact must produce a damped rebound"
    assert max(rebound_speeds) < 1., "foam-ball impacts should dissipate most impact energy"


def test_default_ball_ball_impact_dissipates_kinetic_energy():
    world = make_world(pieces=2, gravity=0., friction=0., rolling_friction=0., sleep=False)
    physics = world[0]
    physics.pos[0, 0] = torch.tensor([3., 3., .075])
    physics.pos[0, 1] = torch.tensor([3.14, 3., .075])
    physics.vel[0, 0, 0] = 1.
    physics.vel[0, 1, 0] = -1.
    energy_before = .5*physics.config.mass*physics.vel.square().sum()
    advance(world)
    energy_after = .5*physics.config.mass*physics.vel.square().sum()
    assert energy_after < energy_before*.5


def test_wall_collision_reverses_approach_velocity():
    world = make_world(pieces=1)
    physics = world[0]
    physics.pos[0, 0, 0] = physics.config.radius + .01
    physics.vel[0, 0, 0] = -2.
    advance(world, 3)
    assert physics.vel[0, 0, 0].item() > 0
    assert physics.pos[0, 0, 0].item() >= physics.config.radius - 1e-5


def test_inactive_worlds_held_and_inactive_fuel_do_not_move():
    world = make_world(worlds=2, pieces=3)
    physics, sim, active, piece_active, owner = world
    active[1] = False
    owner[0, 1] = 0
    piece_active[0, 2] = False
    physics.vel[..., 0] = 1.
    before = {name: getattr(physics, name).clone() for name in ("pos", "vel", "angular")}
    robot_before = sim.velocity.clone()
    advance(world, 4)
    for name, expected in before.items():
        actual = getattr(physics, name)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[0, 1:], expected[0, 1:], rtol=0, atol=0)
    torch.testing.assert_close(sim.velocity[1], robot_before[1], rtol=0, atol=0)
    assert physics.pos[0, 0, 0] > before["pos"][0, 0, 0]


@pytest.mark.parametrize("contact_mode", ["compact", "fused"])
@pytest.mark.parametrize("solver", ["jacobi", "colored"])
def test_grid_and_all_pairs_match_for_moving_contact_clusters(contact_mode, solver):
    options = dict(contact_mode=contact_mode, solver=solver, sleep=False)
    grid = make_world(worlds=2, pieces=12, broadphase="grid", **options)
    reference = make_world(worlds=2, pieces=12, broadphase="all_pairs", **options)
    generator = torch.Generator().manual_seed(907)
    positions = torch.rand((2, 12, 3), generator=generator)
    positions[..., :2] = positions[..., :2] * .5 + 3.
    positions[..., 2] = .075
    velocities = torch.randn((2, 12, 3), generator=generator) * .4
    velocities[..., 2] = 0
    for world in (grid, reference):
        world[0].pos.copy_(positions)
        world[0].vel.copy_(velocities)
    advance(grid, 8)
    advance(reference, 8)
    for name in ("pos", "vel", "angular"):
        torch.testing.assert_close(getattr(grid[0], name), getattr(reference[0], name),
                                   rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(grid[1].velocity, reference[1].velocity, rtol=2e-4, atol=2e-4)


def test_fuel_contact_conserves_horizontal_momentum_without_external_friction():
    world = make_world(gravity=0., friction=0., rolling_friction=0., sleep=False)
    physics = world[0]
    physics.pos[0, 0] = torch.tensor([3., 3., .075])
    physics.pos[0, 1] = torch.tensor([3.15, 3., .075])
    physics.vel[0, 0, 0] = 1.
    initial_momentum = physics.vel[..., :2].sum(dim=1).clone()
    initial_energy = physics.vel.square().sum().item()
    advance(world, 10)
    torch.testing.assert_close(physics.vel[..., :2].sum(dim=1), initial_momentum,
                               rtol=1e-5, atol=1e-5)
    assert physics.vel[0, 1, 0].item() > .1
    assert physics.vel.square().sum().item() <= initial_energy + 1e-4
    assert physics.pos[0, 1, 0] > physics.pos[0, 0, 0]


def test_robot_push_transfers_momentum_and_generates_yaw():
    world = make_world(pieces=1, gravity=0., friction=0., rolling_friction=0., sleep=False)
    physics, sim, *_ = world
    sim.pose[0, 0] = torch.tensor([4., 4., 0.])
    sim.velocity[0, 0, 0] = 3.
    physics.pos[0, 0] = torch.tensor([4.50, 4.20, .075])
    momentum_before = sim.mass[0, 0] * sim.velocity[0, 0, :2].clone()
    advance(world)
    assert physics.vel[0, 0, 0].item() > 0
    assert sim.velocity[0, 0, 0].item() < 3.
    assert sim.velocity[0, 0, 2].item() > 0
    momentum_after = (sim.mass[0, 0] * sim.velocity[0, 0, :2] +
                      physics.config.mass * physics.vel[0, 0, :2])
    torch.testing.assert_close(momentum_after, momentum_before, rtol=1e-5, atol=2e-5)


def test_cached_neighbors_rebuild_after_teleport_and_owner_change():
    cached = make_world(pieces=3, cache=True, sleep=False)
    fresh = make_world(pieces=3, cache=False, sleep=False)
    advance(cached)
    advance(fresh)
    rebuilds_before = cached[0].stats["rebuilds"]
    for world in (cached, fresh):
        physics, _, _, _, owner = world
        # Previously distant ball is suddenly touching ball zero.
        physics.pos[0, 2] = physics.pos[0, 0] + torch.tensor([.145, 0., 0.])
        physics.vel[0, 0, 0] = 1.
        owner[0, 1] = 0
    advance(cached, 3)
    advance(fresh, 3)
    assert cached[0].stats["rebuilds"] > rebuilds_before
    for name in ("pos", "vel", "angular"):
        torch.testing.assert_close(getattr(cached[0], name), getattr(fresh[0], name),
                                   rtol=2e-4, atol=2e-4)
    assert cached[0].vel[0, 2, 0].item() > 0


@pytest.mark.parametrize("contact_mode", ["compact", "fused"])
def test_neighbor_capacity_overflow_keeps_all_dense_contacts(contact_mode):
    limited = make_world(pieces=8, max_neighbors=1, sleep=False, contact_mode=contact_mode)
    reference = make_world(pieces=8, broadphase="all_pairs", sleep=False, contact_mode=contact_mode)
    offsets = torch.tensor([[0., 0.], [.13, 0.], [0., .13], [.13, .13],
                            [.065, .065], [.195, .065], [.065, .195], [.195, .195]])
    for world in (limited, reference):
        world[0].pos[0, :, :2] = offsets + 3.
        world[0].vel[0, 0, 0] = .5
    advance(limited, 3)
    advance(reference, 3)
    assert limited[0].stats["overflows"] > 0
    for name in ("pos", "vel", "angular"):
        torch.testing.assert_close(getattr(limited[0], name), getattr(reference[0], name),
                                   rtol=2e-4, atol=2e-4)


def test_sleeping_pile_wakes_and_transmits_an_impact():
    world = make_world(pieces=4, sleep=True, sleep_time=.04)
    physics = world[0]
    physics.pos[0, :, 0] = torch.tensor([3., 3.15, 3.30, 3.45])
    advance(world, 50)
    assert physics.sleeping.all()
    physics.vel[0, 0, 0] = 2.
    advance(world, 4)
    assert not physics.sleeping[0, 0]
    assert physics.vel[0, 1:, 0].max().item() > .02
    assert torch.isfinite(physics.pos).all()


def test_reset_clears_only_selected_world_contact_and_motion_state():
    world = make_world(worlds=2, pieces=2)
    physics = world[0]
    physics.vel.fill_(.3)
    physics.angular.fill_(.2)
    physics.sleeping.fill_(True)
    inactive_before = {name: getattr(physics, name)[1].clone()
                       for name in ("pos", "vel", "angular", "sleeping")}
    physics.reset(torch.tensor([True, False]))
    assert not physics.angular[0].any()
    assert not physics.sleeping[0].any()
    assert torch.isfinite(physics.pos[0]).all()
    for name, expected in inactive_before.items():
        torch.testing.assert_close(getattr(physics, name)[1], expected, rtol=0, atol=0)


def test_a_row_of_fuel_resists_robot_more_than_one_ball():
    single = make_world(pieces=1, gravity=0., friction=0., rolling_friction=0., sleep=False)
    pile = make_world(pieces=6, gravity=0., friction=0., rolling_friction=0., sleep=False)
    for world in (single, pile):
        physics, sim, *_ = world
        sim.pose[0, 0] = torch.tensor([4., 4., 0.])
        sim.velocity[0, 0, 0] = 3.
        physics.pos[0, :, 0] = 4.5
        physics.pos[0, :, 1] = torch.linspace(3.65, 4.35, physics.pos.shape[1])
    advance(single)
    advance(pile)
    single_loss = 3. - single[1].velocity[0, 0, 0].item()
    pile_loss = 3. - pile[1].velocity[0, 0, 0].item()
    assert single_loss > 0
    assert pile_loss > single_loss * 2


def test_fast_fuel_does_not_cross_another_ball_between_substeps():
    world = make_world(gravity=0., friction=0., rolling_friction=0., sleep=False)
    physics = world[0]
    physics.pos[0, 0] = torch.tensor([3., 3., .075])
    physics.pos[0, 1] = torch.tensor([3.16, 3., .075])
    physics.vel[0, 0, 0] = 25.
    physics.vel[0, 1, 0] = -25.
    advance(world)
    assert physics.pos[0, 0, 0].item() < physics.pos[0, 1, 0].item(), "fast spheres must not tunnel through each other"
    assert physics.vel[0, 0, 0].item() < 25.
    torch.testing.assert_close(physics.vel[..., :2].sum(1), torch.zeros((1, 2)), atol=1e-4, rtol=0)


def test_frictional_fuel_collision_does_not_create_translational_or_spin_energy():
    world = make_world(gravity=0., friction=10., rolling_friction=0., sleep=False)
    physics = world[0]
    physics.pos[0, 0] = torch.tensor([3., 3., .075])
    physics.pos[0, 1] = torch.tensor([3.15, 3., .075])
    physics.vel[0, 0, :2] = torch.tensor([1., 1.])
    physics.angular[0, :, 2] = 10.
    inertia = .4 * physics.config.mass * physics.config.radius ** 2

    def energy():
        return (.5 * physics.config.mass * physics.vel.square().sum() +
                .5 * inertia * physics.angular.square().sum()).item()

    initial = energy()
    for _ in range(6):
        advance(world)
        assert energy() <= initial + 1e-5
    assert energy() < initial
    assert physics.vel[0, 1, :2].norm().item() > 0


def test_adaptive_substeps_ignore_inactive_and_held_fuel_but_include_robot_rotation():
    world = make_world(worlds=2, pieces=3, gravity=0.)
    physics, sim, active, piece_active, owner = world
    active[1] = False
    physics.vel[1, :, 0] = 1000.
    sim.velocity[1, :, 0] = 1000.
    physics.vel[0, 1, 0] = 1000.
    owner[0, 1] = 0
    physics.vel[0, 2, 0] = 1000.
    piece_active[0, 2] = False
    count = physics.required_substeps(active_mask=active, piece_active=piece_active, piece_owner=owner)
    assert count == physics.config.substeps
    sim.velocity[0, 0, 2] = 20.
    count = physics.required_substeps(active_mask=active, piece_active=piece_active, piece_owner=owner)
    assert count > physics.config.substeps


def test_compressed_fuel_contacts_conserve_total_angular_momentum():
    world = make_world(gravity=0., friction=2., rolling_friction=0., sleep=False)
    physics = world[0]
    physics.pos[0, 0] = torch.tensor([3., 3., .075])
    physics.pos[0, 1] = torch.tensor([3.12, 3., .075])
    physics.vel[0, 0, :2] = torch.tensor([.7, 1.])
    physics.vel[0, 1, :2] = torch.tensor([-.3, -.2])
    physics.angular[0, :, 2] = torch.tensor([6., -2.])
    inertia = .4 * physics.config.mass * physics.config.radius ** 2

    def angular_momentum():
        orbital = torch.linalg.cross(physics.pos, physics.config.mass * physics.vel)
        return (orbital + inertia * physics.angular).sum(1)

    initial = angular_momentum().clone()
    advance(world, 5)
    torch.testing.assert_close(angular_momentum(), initial, atol=2e-5, rtol=2e-5)


def test_fuel_rolls_down_field_bump_with_terrain_support():
    from frc_defense.field import BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE

    world = make_world(pieces=1, bump_regions=((3., 3., .6, .8),), sleep=False)
    physics = world[0]
    height = BUMP_PANEL_THICKNESS + BUMP_RAMP_RISE * (1 - .25 / .6)
    normal_z = 1 / (1 + (BUMP_RAMP_RISE / .6) ** 2) ** .5
    physics.pos[0, 0] = torch.tensor([3.25, 3., height + physics.config.radius / normal_z])
    advance(world, 10)
    assert physics.pos[0, 0, 0].item() > 3.25
    assert physics.vel[0, 0, 0].item() > .05
    assert physics._support_height[0, 0].item() > .05
    assert physics.pos[0, 0, 2].item() >= physics._support_height[0, 0].item() + physics.config.radius * .5


def expected_bump_spawn_heights(radius):
    from frc_defense.field import BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE

    slope = BUMP_RAMP_RISE / .6
    return torch.tensor([
        BUMP_PANEL_THICKNESS + BUMP_RAMP_RISE * (1 - .25 / .6) + radius * (1 + slope ** 2) ** .5,
        BUMP_PANEL_THICKNESS + BUMP_RAMP_RISE + radius,
        radius,
    ])


SPAWN_DEVICES = [("cpu", "torch")] + ([(f"cuda:{i}", "hip") for i in range(torch.cuda.device_count())] if torch.version.hip else [])


@pytest.mark.parametrize("device,backend", SPAWN_DEVICES)
def test_reset_places_fuel_above_slope_and_apex_without_changing_inactive_world(device, backend):
    world = make_world(worlds=2, pieces=3, bump_regions=((3., 3., .6, .8),), device=device, backend=backend)
    physics = world[0]
    physics.vel[1].fill_(.3)
    physics.angular[1].fill_(.2)
    physics.sleeping[1].fill_(True)
    physics.sleep_clock[1].fill_(.7)
    inactive = {name: getattr(physics, name)[1].clone()
                for name in ("pos", "vel", "angular", "sleeping", "sleep_clock")}
    physics.piece_pos[0] = torch.tensor([[3.25, 3.], [3., 3.], [4., 3.]])
    physics.reset(torch.tensor([True, False]))
    torch.testing.assert_close(physics.pos[0, :, 2], expected_bump_spawn_heights(physics.config.radius).to(physics.device))
    torch.testing.assert_close(physics.pos[0, :, :2], physics.piece_pos[0])
    for name, expected in inactive.items():
        torch.testing.assert_close(getattr(physics, name)[1], expected, atol=0, rtol=0)


@pytest.mark.parametrize("device,backend", SPAWN_DEVICES)
def test_external_teleport_places_fuel_above_current_terrain_surface(device, backend):
    world = make_world(worlds=2, pieces=3, bump_regions=((3., 3., .6, .8),), gravity=0., sleep=False, device=device, backend=backend)
    physics, _, active, piece_active, owner = world
    active[1] = False
    before = physics.pos[1].clone()
    physics.piece_pos[0] = torch.tensor([[3.25, 3.], [3., 3.], [4., 3.]])
    # Exercise the synchronization used by outpost/game-rule teleports directly,
    # before gravity creates legitimate contact compression.
    changed = physics._sync_external()
    assert changed[0].all()
    torch.testing.assert_close(physics.pos[0, :, 2], expected_bump_spawn_heights(physics.config.radius).to(physics.device))
    torch.testing.assert_close(physics.pos[0, :, :2], physics.piece_pos[0])
    torch.testing.assert_close(physics.pos[1], before, atol=0, rtol=0)

    advance(world)
    assert physics.vel[0].norm().item() < 1e-4, "terrain placement must not manufacture contact impulses"


@pytest.mark.parametrize("device,backend", SPAWN_DEVICES)
def test_masked_reset_does_not_acknowledge_pending_spawn_in_another_world(device, backend):
    world = make_world(worlds=2, pieces=1, device=device, backend=backend,
                       gravity=0., friction=0., rolling_friction=0., sleep=False)
    physics, _, active, _, _ = world
    physics.piece_pos[1, 0] = torch.tensor([8., 3.], device=physics.device)
    physics.piece_vel[1, 0] = torch.tensor([2., 0.], device=physics.device)
    # Reset world A before world B's external spawn has synchronized.
    physics.reset(torch.tensor([True, False], device=physics.device))
    active[0] = False
    advance(world)
    torch.testing.assert_close(physics.pos[1, 0, :2], torch.tensor([8.04, 3.], device=physics.device), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(physics.vel[1, 0, :2], torch.tensor([2., 0.], device=physics.device), atol=1e-6, rtol=1e-6)
