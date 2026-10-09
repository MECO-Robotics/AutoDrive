"""Whole contact groups sleep only with floor support and wake together."""
import pytest
import torch

from frc_defense.fuel_physics_sleep import update_island_sleep
from test_fuel_physics import advance, make_world


def stacked_world(device="cpu", backend="torch"):
    world = make_world(worlds=2, pieces=4, device=device, backend=backend,
                       sleep_islands=True, sleep_time=.02)
    physics = world[0]
    physics.pos[..., 0] = 3.
    physics.pos[..., 2] = .075 + torch.arange(4, device=physics.device) * .149
    # Populate native cached neighbors before exercising the grouping stage.
    advance(world)
    physics.pos[..., 0] = 3.
    physics.pos[..., 2] = .075 + torch.arange(4, device=physics.device) * .149
    physics.vel.zero_()
    physics.vel[..., 2] = physics.config.gravity * .005
    physics.angular.zero_()
    physics.sleeping.zero_()
    physics._island_sleep["clock"].zero_()
    return world


def free_mask(world):
    _, _, active, piece_active, owner = world
    return active[:, None] & piece_active & (owner < 0)


def settle_stage(world):
    for _ in range(5):
        update_island_sleep(world[0], free_mask(world), .005)


def test_supported_three_dimensional_stack_sleeps_as_one_group():
    world = stacked_world()
    settle_stage(world)
    physics = world[0]
    assert physics.sleeping.all()
    assert not physics.vel.any()
    assert not physics.angular.any()
    assert physics._island_sleep["parent"].unique().numel() == 1


@pytest.mark.parametrize("wake", ["fuel", "robot"])
def test_moving_member_or_robot_contact_wakes_entire_stack(wake):
    world = stacked_world()
    settle_stage(world)
    physics, sim, *_ = world
    if wake == "fuel":
        physics.vel[0, 3, 0] = 1.
    else:
        sim.pose[0, 0] = torch.tensor([2.50, 3., 0.])
        sim.velocity[0, 0, 0] = 1.
    update_island_sleep(physics, free_mask(world), .005)
    assert not physics.sleeping[0].any()
    assert physics.sleeping[1].all()
    assert not physics._island_sleep["clock"][0].any()


def test_airborne_component_and_inactive_world_keep_correct_sleep_state():
    world = stacked_world()
    settle_stage(world)
    physics, _, active, _, _ = world
    active[1] = False
    before = physics.sleeping[1].clone(), physics.vel[1].clone(), physics._island_sleep["clock"][1].clone()
    physics.pos[0, :, 2] += 1.
    update_island_sleep(physics, free_mask(world), .005)
    assert not physics.sleeping[0].any(), "a detached unsupported contact group cannot sleep"
    torch.testing.assert_close(physics.sleeping[1], before[0], atol=0, rtol=0)
    torch.testing.assert_close(physics.vel[1], before[1], atol=0, rtol=0)
    torch.testing.assert_close(physics._island_sleep["clock"][1], before[2], atol=0, rtol=0)


HIP_DEVICES = list(range(torch.cuda.device_count())) if torch.version.hip else []


@pytest.mark.skipif(not HIP_DEVICES, reason="requires HIP GPU")
@pytest.mark.parametrize("device_index", HIP_DEVICES or [0])
def test_native_island_sleep_and_wake_matches_cpu(device_index):
    cpu = stacked_world()
    gpu = stacked_world(device=f"cuda:{device_index}", backend="hip")
    for world in (cpu, gpu):
        settle_stage(world)
        assert world[0].sleeping.all()
        physics = world[0]
        physics.sleep_clock[0, 3] = 0
        physics.sleeping[0, 3] = False
        physics.vel[0, 3, 2] = physics.config.gravity * .005
        update_island_sleep(physics, free_mask(world), .005)
        assert not physics.sleeping[0].any()
        assert physics._island_sleep["clock"][0].max().item() == pytest.approx(.005)
        world[0].vel[0, 3, 0] = 1.
        update_island_sleep(world[0], free_mask(world), .005)
    for name in ("sleeping", "vel", "angular", "sleep_clock"):
        torch.testing.assert_close(getattr(gpu[0], name).cpu(), getattr(cpu[0], name), atol=1e-5, rtol=1e-5)
    assert not gpu[0].sleeping[0].any()
    assert gpu[0].sleeping[1].all()


def test_newly_released_quiet_member_restarts_whole_island_sleep_timer():
    world = stacked_world()
    settle_stage(world)
    physics = world[0]
    physics.sleep_clock[0, 3] = 0
    physics.sleeping[0, 3] = False
    physics.vel[0, 3, 2] = physics.config.gravity * .005
    update_island_sleep(physics, free_mask(world), .005)
    assert not physics.sleeping[0].any()
    torch.testing.assert_close(physics._island_sleep["clock"][0], torch.full((4,), .005))
    assert physics.sleeping[1].all()


def test_integrated_stack_settles_and_reset_clears_island_timers():
    world = make_world(pieces=4, sleep_islands=True)
    physics = world[0]
    physics.pos[0, :, 0] = 3.
    physics.pos[0, :, 2] = .075 + torch.arange(4) * .15
    for _ in range(150):
        advance(world)
        if physics.sleeping.all():
            break
    assert physics.sleeping.all()
    assert physics.pos[0, -1, 2].item() > .4
    physics.reset()
    assert not physics._island_sleep["clock"].any()
    assert not physics.sleeping.any()


def supported_bump_stack(device="cpu", backend="torch"):
    from frc_defense.field import BUMP_HEIGHT

    world = make_world(pieces=2, device=device, backend=backend, sleep_islands=True,
                       sleep_time=.02, bump_regions=((3., 3., .6, .8),))
    physics = world[0]
    physics.pos[..., 0] = 3.
    physics.pos[0, :, 2] = BUMP_HEIGHT + .075 + torch.arange(2, device=physics.device) * .15
    for _ in range(100):
        advance(world)
        if physics.sleeping.all():
            break
    return world


def test_island_support_uses_field_bump_surface_height():
    world = supported_bump_stack()
    physics = world[0]
    assert physics.sleeping.all()
    assert physics._support_height.min().item() > .15
    assert physics.pos[0, 0, 2].item() > .2


@pytest.mark.skipif(not HIP_DEVICES, reason="requires HIP GPU")
@pytest.mark.parametrize("device_index", HIP_DEVICES or [0])
def test_native_island_support_uses_field_bump_surface_height(device_index):
    world = supported_bump_stack(device=f"cuda:{device_index}", backend="hip")
    physics = world[0]
    assert physics.sleeping.all()
    assert physics._support_height.min().item() > .15
    assert physics.pos[0, 0, 2].item() > .2
