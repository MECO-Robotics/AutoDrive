import unittest
import torch

from frc_defense import tensor_perception
from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense.tensor_perception import visibility_mask, visibility_mask_torch


def _torch_visibility_baseline(pose, pieces, piece_active, piece_owner, active,
                               perception_range, fov_degrees, obstacles,
                               other_xy, other_radius):
    """Pre-fusion deterministic visibility sequence used by the simulator."""
    delta = pieces - pose[:, None, :2]
    distance = delta.norm(dim=-1)
    visible = (piece_active & (piece_owner < 0) &
               (distance <= perception_range))
    if fov_degrees < 360.:
        bearing = torch.atan2(delta[..., 1], delta[..., 0])
        angle = torch.atan2(torch.sin(bearing - pose[:, None, 2]),
                            torch.cos(bearing - pose[:, None, 2])).abs()
        visible &= angle <= __import__("math").radians(fov_degrees) * .5
    if obstacles.numel():
        segment = delta
        denom = segment.square().sum(-1).clamp_min(1e-8)
        rel = obstacles[None, None, :, :2] - pose[:, None, None, :2]
        t = (rel * segment[:, :, None, :]).sum(-1) / denom[:, :, None]
        closest = pose[:, None, None, :2] + t.clamp(0., 1.)[..., None] * segment[:, :, None, :]
        radius = obstacles[None, None, :, 2] + .03
        blocked = ((radius > 0.) & ((closest - obstacles[None, None, :, :2]).norm(dim=-1)
                                   <= radius) & (t > .02) & (t < .98)).any(-1)
        visible &= ~blocked
    rel = other_xy[:, None, :] - pose[:, None, :2]
    denom = delta.square().sum(-1).clamp_min(1e-8)
    t = (rel * delta).sum(-1) / denom
    closest = pose[:, None, :2] + t.clamp(0., 1.)[..., None] * delta
    blocked = (((closest - other_xy[:, None, :]).norm(dim=-1)
                <= other_radius[:, None]) & (t > .02) & (t < .98))
    return visible & ~blocked & active[:, None]


def _hidden_fuel_env():
    env = TensorDefenseEnv(
        num_envs=1,
        task="counter_defense",
        device="cpu",
        seed=7,
        action_mode="strategic",
        randomize=False,
        fuel_count=96,
        perception_config={
            "fov_degrees": 360,
            "range_m": 0.01,
            "detection_dropout": 0.0,
            "position_noise_m": 0.0,
            "velocity_noise_mps": 0.0,
            "track_timeout_s": 0.5,
        },
    )
    env.reset(seed=7)
    return env


class PerceptionComplianceTests(unittest.TestCase):
    def test_fused_visibility_matches_torch_for_randomized_and_boundary_cases(self):
        # The same test exercises the HIP implementation when the opt-in flag
        # is enabled on ROCm; otherwise it verifies the production fallback.
        if not (torch.cuda.is_available() and torch.version.hip):
            self.skipTest("requires a ROCm/HIP device")
        if not tensor_perception.FUSED_PERCEPTION_HIP_ENABLED:
            self.skipTest("set AUTODRIVE_FUSED_PERCEPTION_HIP=1 to exercise the HIP kernel")
        extension = tensor_perception._hip_perception_extension()
        self.assertIsNotNone(extension, "fused perception HIP extension did not load")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(1947)
        worlds, count, obstacle_count = 7, 127, 11
        pose = torch.rand((worlds, 3), generator=generator, device=device)
        pose[:, :2] *= torch.tensor([15., 8.], device=device)
        pose[:, 2] = (pose[:, 2] - .5) * 6.
        pieces = torch.rand((worlds, count, 2), generator=generator, device=device)
        pieces *= torch.tensor([16., 9.], device=device)
        piece_active = torch.rand((worlds, count), generator=generator, device=device) > .2
        piece_owner = torch.randint(-2, 3, (worlds, count), generator=generator, device=device)
        active = torch.tensor([True, False, True, True, False, True, True], device=device)
        obstacles = torch.rand((obstacle_count, 3), generator=generator, device=device)
        obstacles[:, :2] *= torch.tensor([16., 9.], device=device)
        obstacles[:, 2] = obstacles[:, 2] * .3 - .02
        other_xy = torch.rand((worlds, 2), generator=generator, device=device)
        other_xy *= torch.tensor([16., 9.], device=device)
        # Distinct radii model differently sized opponents across worlds.
        other_radius = torch.linspace(.2, 1.1, worlds, device=device)
        for fov, distance in ((360., 5.), (90., 8.), (220., 12.)):
            expected = _torch_visibility_baseline(
                pose, pieces, piece_active, piece_owner, active, distance,
                fov, obstacles, other_xy, other_radius)
            reference = visibility_mask_torch(
                pose, pieces, piece_active, piece_owner, active, distance,
                fov, obstacles, other_xy, other_radius)
            actual = visibility_mask(
                pose.contiguous(), pieces.contiguous(), piece_active.contiguous(),
                piece_owner.contiguous(), active.contiguous(), distance, fov,
                obstacles.contiguous(), other_xy.contiguous(), other_radius.contiguous())
            self.assertTrue(torch.equal(reference, expected))
            self.assertTrue(torch.equal(actual, expected))
            self.assertFalse(actual[~active].any())

        # Cover inclusive range/FOV and strict segment endpoint semantics,
        # plus tangent circles and an opponent with a non-default radius.
        pose = torch.tensor([[0., 0., 0.]], device=device)
        pieces = torch.tensor([[[3., 0.], [0., 3.], [1., 0.], [2., 0.],
                                [2.1213203436, 2.1213203436]]], device=device)
        piece_active = torch.ones((1, 5), dtype=torch.bool, device=device)
        piece_owner = torch.full((1, 5), -1, dtype=torch.long, device=device)
        active = torch.ones((1,), dtype=torch.bool, device=device)
        # Circle tangent to ray #0; circle lies at strict endpoint t=.02 for #2.
        obstacles = torch.tensor([[1.5, .53, .5], [.02, 0., .1]], device=device)
        other_xy = torch.tensor([[2., .4]], device=device)
        other_radius = torch.tensor([.4], device=device)
        expected = _torch_visibility_baseline(
            pose, pieces, piece_active, piece_owner, active, 3., 90.,
            obstacles, other_xy, other_radius)
        actual = visibility_mask(
            pose.contiguous(), pieces.contiguous(), piece_active.contiguous(),
            piece_owner.contiguous(), active.contiguous(), 3., 90.,
            obstacles.contiguous(), other_xy.contiguous(), other_radius.contiguous())
        self.assertTrue(torch.equal(actual, expected))

    def test_full_circle_fov_fast_path_preserves_randomized_tracks_and_inactive_rows(self):
        env = TensorDefenseEnv(
            num_envs=4, task="counter_defense", device="cpu", seed=91,
            action_mode="strategic", randomize=False, fuel_count=504,
            field_colliders=[], perception_config={
                "fov_degrees": 360, "range_m": .2, "detection_dropout": 0.,
                "position_noise_m": 0., "velocity_noise_mps": 0.,
            })
        generator = torch.Generator(device="cpu").manual_seed(1234)
        env.sim.pose[:, 0, :2] = torch.tensor([1., 1.])
        env.sim.pose[:, 1, :2] = torch.tensor([15., 7.])
        env.piece_pos.copy_(torch.rand((env.n, env.fuel_count, 2), generator=generator))
        env.piece_pos[..., 0] = env.piece_pos[..., 0] * 14. + 1.
        env.piece_pos[..., 1] = env.piece_pos[..., 1] * 6. + 1.
        env.piece_pos[:, :20] = env.sim.pose[:, 0, None, :2] + (
            torch.rand((env.n, 20, 2), generator=generator) - .5) * .2
        env.piece_pos[:, 20:40] = env.sim.pose[:, 1, None, :2] + (
            torch.rand((env.n, 20, 2), generator=generator) - .5) * .2
        env.piece_active.copy_(torch.rand((env.n, env.fuel_count), generator=generator) > .3)
        env.piece_owner.copy_(torch.where(
            torch.rand((env.n, env.fuel_count), generator=generator) > .5,
            torch.full((env.n, env.fuel_count), -1),
            torch.zeros((env.n, env.fuel_count), dtype=torch.long)))
        active = torch.tensor([True, False, True, False])
        env._track_mask[~active] = True
        env._track_pos[~active] = 123.
        inactive_tracks = env._track_mask[~active].clone()
        inactive_positions = env._track_pos[~active].clone()

        expected = []
        for robot in range(2):
            delta = env.piece_pos - env.sim.pose[:, robot, None, :2]
            distance = delta.norm(dim=-1)
            # With FOV=360, the old wrapped-angle predicate is always true.
            # The other robot cannot occlude a piece inside this short range.
            expected.append(env.piece_active & (env.piece_owner < 0)
                            & (distance <= env.perception_range)
                            & active[:, None])

        rng_expected = torch.Generator(device="cpu")
        rng_expected.set_state(env.generator.get_state())
        for _ in range(2):
            torch.randn((2, env.fuel_count, 2), generator=rng_expected)
            torch.rand((2,), generator=rng_expected)
            torch.randn((2, 2), generator=rng_expected)
            torch.randn((2,), generator=rng_expected)
            torch.randn((2, 3), generator=rng_expected)

        env._update_perception(active_mask=active, active_count=int(active.sum()))

        for robot in range(2):
            self.assertTrue(torch.equal(env._track_mask[active, robot], expected[robot][active]))
            self.assertTrue(torch.equal(env._track_pos[active, robot][expected[robot][active]],
                                        env.piece_pos[active][expected[robot][active]]))
        self.assertTrue(torch.equal(env._track_mask[~active], inactive_tracks))
        self.assertTrue(torch.equal(env._track_pos[~active], inactive_positions))
        self.assertTrue(torch.equal(env.generator.get_state(), rng_expected.get_state()))

    def test_narrow_fov_retains_wrapped_angle_filter(self):
        env = TensorDefenseEnv(
            num_envs=1, task="counter_defense", device="cpu", seed=92,
            action_mode="strategic", randomize=False, fuel_count=96,
            field_colliders=[], perception_config={
                "fov_degrees": 90, "range_m": 2., "detection_dropout": 0.,
                "position_noise_m": 0., "velocity_noise_mps": 0.,
            })
        env.sim.pose[:, 0] = torch.tensor([5., 4., 0.])
        env.sim.pose[:, 1] = torch.tensor([15., 7., 0.])
        env.piece_active.zero_()
        env.piece_owner.fill_(-1)
        env.piece_pos[:, 0] = torch.tensor([6., 4.])
        env.piece_pos[:, 1] = torch.tensor([4., 4.])
        env.piece_active[:, :2] = True
        env._update_perception()
        self.assertTrue(bool(env._track_mask[0, 0, 0]))
        self.assertFalse(bool(env._track_mask[0, 0, 1]))

    def test_controller_observation_ignores_unseen_fuel_and_opponent_possession(self):
        env = _hidden_fuel_env()
        before = env._strategic_observation(0).clone()
        env.piece_active[:] = True
        env.piece_owner[:] = 1
        env.piece_pos[:] = torch.tensor([15.0, 7.5])
        env.piece_vel[:] = torch.tensor([3.0, -2.0])
        env._track_mask[:, 1] = True
        env._track_pos[:, 1] = torch.tensor([2.0, 2.0])
        env._update_perception()
        after = env._strategic_observation(0)
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(env._track_mask[:, 0].any())


    def test_fuel_denial_target_uses_only_visible_tracked_fuel(self):
        env = _hidden_fuel_env()
        before = env._defense_target("FUEL_DENIAL", 1, 0).clone()
        env.piece_active[:] = True
        env.piece_owner[:] = -1
        env.piece_pos[:] = torch.tensor([1.0, 1.0])
        env._update_perception()
        after = env._defense_target("FUEL_DENIAL", 1, 0)
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(env._track_mask[:, 1].any())


    def test_perception_state_has_no_owner_channel(self):
        env = _hidden_fuel_env()
        state = env.perception_state(0)
        self.assertFalse(hasattr(state, "piece_owner"))
        self.assertFalse(hasattr(state, "fuel_owner"))
        self.assertEqual(env._perceived_fuel(0)[2].shape, (1, env.fuel_count))


if __name__ == "__main__":
    unittest.main()
