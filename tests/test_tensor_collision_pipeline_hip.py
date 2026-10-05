import math

import pytest

torch = pytest.importorskip("torch")


def test_fused_contact_pipeline_matches_torch_empty_sets_tangencies_and_masks():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("fused contact pipeline requires a HIP device")
    from frc_defense import tensor_collision_pipeline as fused
    from frc_defense.tensor_sim import TensorVectorizedSimulator

    if not fused.FUSED_CONTACT_PIPELINE_HIP_ENABLED or fused._extension() is None:
        pytest.skip("fused contact pipeline is disabled or unavailable")

    device = torch.device("cuda:0")
    # Exercise each empty collider set independently and both together.
    cases = (([], []), ([[3.0, 3.0, .25]], []), ([], [[3.0, 3.0, .5, .5]]))
    for obstacles, boxes in cases:
        torch_sim = TensorVectorizedSimulator(num_envs=40, device=device, seed=672,
            obstacles=obstacles, field_colliders=boxes, contact_iterations=3)
        hip_sim = TensorVectorizedSimulator(num_envs=40, device=device, seed=672,
            obstacles=obstacles, field_colliders=boxes, contact_iterations=3)
        gen = torch.Generator(device=device).manual_seed(23)
        pose = torch.empty_like(torch_sim.pose)
        pose[:, :, 0].uniform_(-.5, torch_sim.field_length + .5, generator=gen)
        pose[:, :, 1].uniform_(-.5, torch_sim.field_width + .5, generator=gen)
        pose[:, :, 2].uniform_(-math.pi, math.pi, generator=gen)
        # Exact wall tangency, identical-heading robot tangency, and a box tie.
        pose[0, 0] = torch.tensor([torch_sim.length[0, 0] / 2, 2., 0.], device=device)
        pose[1, 0] = torch.tensor([4., 4., 0.], device=device)
        pose[1, 1] = torch.tensor([4. + torch_sim.length[0, 0], 4., 0.], device=device)
        pose[2, :, :2] = torch.tensor([3., 3.], device=device)
        velocity = torch.empty_like(torch_sim.velocity).uniform_(-2., 2., generator=gen)
        active = torch.arange(40, device=device).remainder(3).ne(1)
        flags = (torch.arange(40, device=device).remainder(2).bool(),
                 torch.arange(40, device=device).remainder(3).bool(),
                 torch.arange(80, device=device).reshape(40, 2).remainder(2).bool(),
                 torch.arange(160, device=device).reshape(40, 2, 2).remainder(4).eq(0))
        state = (pose, velocity, *flags)
        for sim in (torch_sim, hip_sim):
            for dst, src in zip((sim.pose, sim.velocity, sim.robot_contact,
                sim.opponent_contact, sim.field_contact, sim.wall_contact), state):
                dst.copy_(src)

        for _ in range(torch_sim.contact_iterations):
            torch_sim._robot_collision(active)
            torch_sim._walls_torch(active)
            torch_sim._obstacle_collision(active)
            torch_sim._field_collision(active)
        expected = tuple(t.clone() for t in (torch_sim.pose, torch_sim.velocity,
            torch_sim.robot_contact, torch_sim.opponent_contact,
            torch_sim.field_contact, torch_sim.wall_contact))
        assert fused.contact_pipeline(hip_sim, active.contiguous())
        actual = (hip_sim.pose, hip_sim.velocity, hip_sim.robot_contact,
                  hip_sim.opponent_contact, hip_sim.field_contact, hip_sim.wall_contact)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert torch.equal(hip_sim.pose[~active], pose[~active])
        assert torch.equal(hip_sim.velocity[~active], velocity[~active])
        for actual_flag, initial_flag in zip(actual[2:], flags):
            assert torch.equal(actual_flag[~active], initial_flag[~active])


def test_fused_contact_pipeline_matches_complete_physics_tick():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("fused contact pipeline requires a HIP device")
    from frc_defense import tensor_collision_pipeline as fused
    from frc_defense.tensor_sim import TensorVectorizedSimulator

    if not fused.FUSED_CONTACT_PIPELINE_HIP_ENABLED or fused._extension() is None:
        pytest.skip("fused contact pipeline is disabled or unavailable")
    device = torch.device("cuda:0")
    reference = TensorVectorizedSimulator(num_envs=48, device=device, seed=311,
        obstacles=[[2.5, 2.5, .25], [3.2, 3.2, .3]],
        field_colliders=[[1.0, 1.0, .3, .5]], contact_iterations=3)
    candidate = TensorVectorizedSimulator(num_envs=48, device=device, seed=311,
        obstacles=[[2.5, 2.5, .25], [3.2, 3.2, .3]],
        field_colliders=[[1.0, 1.0, .3, .5]], contact_iterations=3)
    gen = torch.Generator(device=device).manual_seed(5)
    pose = torch.empty_like(reference.pose)
    pose[:, :, 0].uniform_(-.2, reference.field_length + .2, generator=gen)
    pose[:, :, 1].uniform_(-.2, reference.field_width + .2, generator=gen)
    pose[:, :, 2].uniform_(-math.pi, math.pi, generator=gen)
    velocity = torch.empty_like(reference.velocity).uniform_(-1., 1., generator=gen)
    active = torch.arange(48, device=device).remainder(4).ne(0)
    command = torch.empty_like(reference.velocity).uniform_(-.7, .7, generator=gen)
    for sim in (reference, candidate):
        sim.pose.copy_(pose)
        sim.velocity.copy_(velocity)
        sim._fused_wall_collision_hip_enabled = False
    reference._step_eager(command, active)
    candidate._fused_contact_pipeline_hip_enabled = True
    candidate._step_eager(command, active)
    for name in ("pose", "velocity", "robot_contact", "opponent_contact",
                 "field_contact", "wall_contact", "module_angle",
                 "module_steer_rate", "module_drive_speed", "module_current",
                 "module_supply_current", "robot_current"):
        assert torch.equal(getattr(candidate, name), getattr(reference, name)), name
