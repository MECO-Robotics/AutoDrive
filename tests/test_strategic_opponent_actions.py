import unittest

import torch

from frc_defense.tensor_sim import TensorDefenseEnv


class StrategicOpponentActionTests(unittest.TestCase):
    def test_categorical_opponent_actions_0_through_7_are_preserved(self):
        env = TensorDefenseEnv(
            num_envs=8, task="defense", device="cpu", opponent="learned",
            action_mode="strategic", horizon=8000, randomize=False,
        )
        actions = torch.arange(8, dtype=torch.long)
        env.learned_opponent_fn = lambda _obs: actions
        env.strategic_action_mask = lambda focal=0, **_kwargs: torch.ones(
            (env.n, 8), dtype=torch.bool, device=env.device
        )
        captured = []
        original = env._strategic_opponent_target

        def capture(action_class, **kwargs):
            captured.append(action_class.detach().clone())
            return original(action_class, **kwargs)

        env._strategic_opponent_target = capture
        env.step(torch.zeros(env.n, dtype=torch.long))
        self.assertTrue(torch.equal(captured[0].cpu(), actions))

    def test_legacy_continuous_three_action_opponent_builds_velocity_command(self):
        env = TensorDefenseEnv(
            num_envs=2, task="defense", device="cpu", opponent="learned",
            horizon=8000, randomize=False,
        )
        env.learned_opponent_action_kind = "continuous"
        env.learned_opponent_action_dim = 3
        env.learned_opponent_fn = lambda _obs: torch.tensor(
            [[1.0, 0.0, 1.0], [-1.0, 0.0, -1.0]], dtype=torch.float32
        )

        observation, reward, done, truncated, _info = env.step(
            torch.zeros((env.n, env.action_dim), dtype=torch.float32)
        )

        self.assertEqual(tuple(observation.shape), (env.n, env.obs_dim))
        self.assertEqual(tuple(reward.shape), (env.n,))
        self.assertTrue(torch.isfinite(observation).all())
        self.assertTrue(torch.isfinite(reward).all())


if __name__ == "__main__":
    unittest.main()
