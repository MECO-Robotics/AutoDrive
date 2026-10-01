import unittest
import torch

from frc_defense.tensor_sim import TensorDefenseEnv


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
