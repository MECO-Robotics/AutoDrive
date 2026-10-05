import math

import pytest

torch = pytest.importorskip("torch")


def test_fused_walls_match_torch_bitwise_for_masked_boundary_contacts():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("fused wall contacts require a HIP device")

    from frc_defense import tensor_collision
    from frc_defense.field import rebuilt_field, static_collision_boxes
    from frc_defense.tensor_sim import TensorVectorizedSimulator

    if not tensor_collision.FUSED_COLLISION_HIP_ENABLED:
        pytest.skip("fused wall contacts are disabled by environment")
    if tensor_collision._extension() is None:
        pytest.skip("fused wall contact extension is unavailable")

    device = torch.device("cuda:0")
    n, seed = 64, 8127
    colliders = [box.as_tensor() for box in static_collision_boxes(rebuilt_field())]
    sim = TensorVectorizedSimulator(num_envs=n, device=device, seed=seed,
                                    field_colliders=colliders)
    generator = torch.Generator(device=device).manual_seed(seed)
    source_pose = torch.empty_like(sim.pose)
    source_pose[:, :, 0].uniform_(-0.4, sim.field_length + 0.4, generator=generator)
    source_pose[:, :, 1].uniform_(-0.4, sim.field_width + 0.4, generator=generator)
    source_pose[:, :, 2].uniform_(-math.pi, math.pi, generator=generator)
    source_velocity = torch.empty_like(sim.velocity).uniform_(-3., 3., generator=generator)
    source_wall_contact = torch.arange(n * 4, device=device).reshape(n, 2, 2).remainder(3).eq(0)

    # Exercise exact no-penetration ties, single-wall contacts, simultaneous
    # corner contacts, rotated extents, and the opposing field edges.
    source_pose[:, :, 2] = 0.
    half_length = sim.length[0, 0] * .5
    half_width = sim.width[0, 0] * .5
    source_pose[0, 0, :2] = torch.tensor([half_length, 2.0], device=device)
    source_pose[1, 0, :2] = torch.tensor([-.25, -.30], device=device)
    source_pose[2, 0, :2] = torch.tensor([.40, .40], device=device)
    source_pose[2, 0, 2] = math.pi / 4
    source_pose[3, 0, :2] = torch.tensor([sim.field_length - .40,
                                         sim.field_width - .40], device=device)
    source_pose[4, 0, 0] = half_length
    source_pose[5, 1, 1] = sim.field_width - half_width

    active = torch.arange(n, device=device).remainder(4).ne(1)
    snapshots = (source_pose.clone(), source_velocity.clone(), source_wall_contact.clone())

    sim.pose.copy_(snapshots[0])
    sim.velocity.copy_(snapshots[1])
    sim.wall_contact.copy_(snapshots[2])
    sim._fused_wall_collision_hip_enabled = False
    sim._walls(active)
    expected = (sim.pose.clone(), sim.velocity.clone(), sim.wall_contact.clone())

    sim.pose.copy_(snapshots[0])
    sim.velocity.copy_(snapshots[1])
    sim.wall_contact.copy_(snapshots[2])
    sim._fused_wall_collision_hip_enabled = True
    sim._walls(active)
    actual = (sim.pose, sim.velocity, sim.wall_contact)

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)
    assert torch.equal(sim.pose[~active], snapshots[0][~active])
    assert torch.equal(sim.velocity[~active], snapshots[1][~active])
    assert torch.equal(sim.wall_contact[~active], snapshots[2][~active])
