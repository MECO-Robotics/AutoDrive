import unittest

from frc_defense.tensor_sim import TensorDefenseEnv


class StrategicOpponentActionTests(unittest.TestCase):
    def test_learned_opponents_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "learned opponents are no longer supported"):
            TensorDefenseEnv(
                num_envs=1, task="defense", device="cpu", opponent="learned",
                action_mode="strategic", randomize=False,
            )


if __name__ == "__main__":
    unittest.main()
