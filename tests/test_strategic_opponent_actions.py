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
        env.strategic_action_mask = lambda focal=0: torch.ones(
            (env.n, 8), dtype=torch.bool, device=env.device
        )
        captured = []
        original = env._strategic_opponent_target

        def capture(action_class):
            captured.append(action_class.detach().clone())
            return original(action_class)

        env._strategic_opponent_target = capture
        env.step(torch.zeros(env.n, dtype=torch.long))
        self.assertTrue(torch.equal(captured[0].cpu(), actions))


if __name__ == "__main__":
    unittest.main()
