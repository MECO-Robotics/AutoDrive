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
