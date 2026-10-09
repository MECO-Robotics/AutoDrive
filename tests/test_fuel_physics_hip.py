"""Contact parity on each available HIP GPU, including inactive worlds."""
import pytest
import torch

from test_fuel_physics import advance, make_world


HIP_DEVICES = list(range(torch.cuda.device_count())) if torch.version.hip else []
pytestmark = pytest.mark.skipif(not HIP_DEVICES, reason="requires ROCm/HIP devices")


@pytest.mark.parametrize("device_index", HIP_DEVICES or [0])
@pytest.mark.parametrize("contact_mode", ["compact", "fused"])
def test_hip_matches_cpu_for_coupled_fuel_robot_contacts(device_index, contact_mode):
    entry_device = torch.cuda.current_device()
    options = dict(worlds=3, pieces=12, robots=2, sleep=False,
                   contact_mode=contact_mode, solver="jacobi")
    cpu = make_world(**options)
    gpu = make_world(device=f"cuda:{device_index}", backend="hip", **options)
    generator = torch.Generator().manual_seed(1731)
    positions = torch.rand((3, 12, 3), generator=generator)
    positions[..., :2] = positions[..., :2] * .7 + 3.
    positions[..., 2] = .075
    positions[:, 0] = torch.tensor([4.50, 4.20, .075])
    velocities = torch.randn((3, 12, 3), generator=generator) * .3
    velocities[..., 2] = 0.
    for world in (cpu, gpu):
        physics, sim, active, piece_active, owner = world
        physics.pos.copy_(positions)
        physics.vel.copy_(velocities)
        sim.pose[:, 0] = torch.tensor([4., 4., 0.], device=sim.pose.device)
        sim.velocity[:, 0, 0] = 3.
        active[2] = False
        physics.sleeping[2, 0] = True
        owner[0, 3] = 0
        piece_active[1, 7] = False
    advance(cpu, 6)
    advance(gpu, 6)
    torch.cuda.synchronize(device_index)
    assert gpu[0].stats["backend"] == "hip"
    assert torch.cuda.current_device() == entry_device
    for name in ("pos", "vel", "angular", "sleeping"):
        actual = getattr(gpu[0], name).cpu()
        expected = getattr(cpu[0], name)
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(gpu[1].velocity.cpu(), cpu[1].velocity, rtol=5e-4, atol=5e-4)
    assert gpu[1].velocity[0, 0, 0].item() < 3.


@pytest.mark.parametrize("device_index", HIP_DEVICES or [0])
def test_hip_fast_fuel_collision_matches_cpu_without_tunneling(device_index):
    options = dict(gravity=0., friction=0., rolling_friction=0., sleep=False)
    cpu = make_world(**options)
    gpu = make_world(device=f"cuda:{device_index}", backend="hip", **options)
    for world in (cpu, gpu):
        physics = world[0]
        physics.pos[0, 0] = torch.tensor([3., 3., .075], device=physics.device)
        physics.pos[0, 1] = torch.tensor([3.16, 3., .075], device=physics.device)
        physics.vel[0, 0, 0] = 25.
        physics.vel[0, 1, 0] = -25.
        advance(world)
    assert gpu[0].pos[0, 0, 0].item() < gpu[0].pos[0, 1, 0].item()
    for name in ("pos", "vel", "angular"):
        torch.testing.assert_close(getattr(gpu[0], name).cpu(), getattr(cpu[0], name),
                                   rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("device_index", HIP_DEVICES or [0])
@pytest.mark.parametrize("contact_mode", ["compact", "fused"])
def test_hip_neighbor_overflow_preserves_dense_pile_contacts(device_index, contact_mode):
    cpu = make_world(pieces=8, broadphase="all_pairs", sleep=False, contact_mode=contact_mode)
    gpu = make_world(pieces=8, device=f"cuda:{device_index}", backend="hip",
                     max_neighbors=1, sleep=False, contact_mode=contact_mode)
    offsets = torch.tensor([[0., 0.], [.13, 0.], [0., .13], [.13, .13],
                            [.065, .065], [.195, .065], [.065, .195], [.195, .195]])
    for world in (cpu, gpu):
        physics = world[0]
        physics.pos[0, :, :2].copy_(offsets + 3.)
        physics.vel[0, 0, 0] = .5
        advance(world, 3)
    gpu[0].refresh_stats()
    assert gpu[0].stats["overflows"] > 0
    for name in ("pos", "vel", "angular"):
        torch.testing.assert_close(getattr(gpu[0], name).cpu(), getattr(cpu[0], name),
                                   rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("device_index", HIP_DEVICES or [0])
def test_hip_sleeping_pile_wakes_and_matches_cpu(device_index):
    options = dict(pieces=4, sleep=True, sleep_time=.04)
    cpu = make_world(**options)
    gpu = make_world(device=f"cuda:{device_index}", backend="hip", **options)
    for world in (cpu, gpu):
        physics = world[0]
        physics.pos[0, :, 0] = torch.tensor([3., 3.15, 3.30, 3.45], device=physics.device)
        advance(world, 50)
        assert physics.sleeping.all()
        physics.vel[0, 0, 0] = 2.
        advance(world, 4)
        assert physics.vel[0, 1:, 0].max().item() > .02
    for name in ("pos", "vel", "angular"):
        torch.testing.assert_close(getattr(gpu[0], name).cpu(), getattr(cpu[0], name),
                                   rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("device_index", HIP_DEVICES or [0])
def test_hip_colored_grid_matches_native_all_pairs_and_conserves_momentum(device_index):
    options = dict(pieces=6, device=f"cuda:{device_index}", backend="hip",
                   solver="colored", sleep=False, gravity=0., friction=0., rolling_friction=0.)
    grid = make_world(broadphase="grid", **options)
    reference = make_world(broadphase="all_pairs", **options)
    for world in (grid, reference):
        physics = world[0]
        physics.pos[0, :, 0] = torch.arange(6, device=physics.device) * .14 + 3.
        physics.vel[0, 0, 0] = 1.
        advance(world, 4)
    for name in ("pos", "vel", "angular"):
        torch.testing.assert_close(getattr(grid[0], name), getattr(reference[0], name),
                                   rtol=5e-4, atol=5e-4)
    expected = torch.tensor([[1., 0.]], device=grid[0].device)
    torch.testing.assert_close(grid[0].vel[..., :2].sum(1), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device_index", HIP_DEVICES or [0])
def test_hip_robot_grid_tracks_moved_robot_fast_fuel_and_changed_ownership(device_index):
    options = dict(worlds=2, pieces=6, device=f"cuda:{device_index}", backend="hip", sleep=False)
    grid = make_world(robot_broadphase="grid", **options)
    reference = make_world(robot_broadphase="all_pairs", **options)
    for world in (grid, reference):
        physics = world[0]
        physics.pos[:, 0, :2] = torch.tensor([4.50, 4.20], device=physics.device)
        physics.pos[:, 1, :2] = torch.tensor([4.5, 3.8], device=physics.device)
        physics.pos[:, 2, :2] = torch.tensor([5., 4.], device=physics.device)
        advance(world)  # cache pieces while robot is still far away
        physics.vel[:, 2, 0] = -20.
        world[1].pose[:, 0] = torch.tensor([4., 4., 0.], device=physics.device)
        world[1].velocity[:, 0, 0] = 3.
        world[4][0, 1] = 0  # ownership change invalidates spatial eligibility
        world[2][1] = False
        advance(world, 4)
    for name in ("pos", "vel", "angular", "sleeping"):
        torch.testing.assert_close(getattr(grid[0], name), getattr(reference[0], name), atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(grid[1].velocity, reference[1].velocity, atol=5e-4, rtol=5e-4)
    assert grid[1].velocity[0, 0, 0].item() < 3.
    assert grid[0].vel[0, 2, 0].item() > -20.


@pytest.mark.parametrize("device_index", HIP_DEVICES or [0])
def test_hip_bump_terrain_contacts_match_cpu(device_index):
    from frc_defense.field import BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE

    options = dict(pieces=1, bump_regions=((3., 3., .6, .8),), sleep=False)
    cpu = make_world(**options)
    gpu = make_world(device=f"cuda:{device_index}", backend="hip", **options)
    height = BUMP_PANEL_THICKNESS + BUMP_RAMP_RISE * (1 - .25 / .6)
    normal_z = 1 / (1 + (BUMP_RAMP_RISE / .6) ** 2) ** .5
    for world in (cpu, gpu):
        physics = world[0]
        physics.pos[0, 0] = torch.tensor([3.25, 3., height + physics.config.radius / normal_z], device=physics.device)
        advance(world, 10)
    for name in ("pos", "vel", "angular", "_support_height"):
        torch.testing.assert_close(getattr(gpu[0], name).cpu(), getattr(cpu[0], name), atol=5e-4, rtol=5e-4)
    assert gpu[0].vel[0, 0, 0].item() > .05
