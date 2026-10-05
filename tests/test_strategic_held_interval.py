import unittest

import torch

from frc_defense.tensor_training import _gae_advantages, _held_action_interval


class _FakeEnv:
    def __init__(self, horizons, terminations=None):
        self.n = len(horizons)
        self.device = torch.device("cpu")
        self.horizons = torch.tensor(horizons)
        self.terminations = torch.tensor(terminations or [False] * self.n)
        self.steps = torch.zeros(self.n, dtype=torch.long)
        self.applied_actions = torch.zeros(self.n, dtype=torch.long)
        self.reset_calls = 0

    def step(self, action, active_mask=None):
        active_mask = (torch.ones(self.n, dtype=torch.bool) if active_mask is None
                       else active_mask)
        self.steps += active_mask.long()
        self.applied_actions += active_mask.long()
        obs = self.steps[:, None].float()
        ended = (self.steps >= self.horizons) & active_mask
        done = ended & self.terminations
        truncated = ended & ~self.terminations
        reward = torch.ones(self.n)
        return obs, reward, done, truncated, {}

    def reset_done(self, mask):
        self.reset_calls += int(mask.sum())
        self.steps[mask] = 0
        self.applied_actions[mask] = 0
        return self.steps[:, None].float()


class StrategicHeldIntervalTests(unittest.TestCase):
    def test_ended_world_is_not_stepped_or_reset_during_held_interval(self):
        env = _FakeEnv([2])
        initial = env.steps[:, None].float()
        reward, done, truncated, terminal_obs, active, _, active_world_ticks, matches = _held_action_interval(
            env, torch.tensor([4]), 5, initial)
        self.assertEqual(reward.tolist(), [2.0])
        self.assertEqual(env.steps.tolist(), [2])
        self.assertEqual(env.applied_actions.tolist(), [2])
        self.assertEqual(env.reset_calls, 0)
        self.assertEqual(terminal_obs[:, 0].tolist(), [2.0])
        self.assertFalse(bool(active[0]))
        self.assertFalse(bool(done[0]))
        self.assertTrue(bool(truncated[0]))
        self.assertEqual(matches, 1)
        self.assertEqual(active_world_ticks, 2)
        reset_obs = env.reset_done(done | truncated)
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(reset_obs[:, 0].tolist(), [0.0])
        env.step(torch.tensor([7]))
        self.assertEqual(env.steps.tolist(), [1])
        self.assertEqual(env.applied_actions.tolist(), [1])

    def test_reward_transition_stops_at_each_worlds_episode_end(self):
        env = _FakeEnv([1, 2, 3])
        result = _held_action_interval(
            env, torch.tensor([1, 1, 1]), 6, torch.zeros((3, 1)))
        reward, done, truncated, terminal_obs, active, _, active_world_ticks, matches = result
        self.assertEqual(reward.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(terminal_obs[:, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(env.applied_actions.tolist(), [1, 2, 3])
        self.assertEqual(done.tolist(), [False, False, False])
        self.assertEqual(truncated.tolist(), [True, True, True])
        self.assertEqual(active.tolist(), [False, False, False])
        self.assertEqual(matches, 3)
        self.assertEqual(active_world_ticks, 6)

    def test_termination_and_truncation_stop_on_different_ticks(self):
        env = _FakeEnv([1, 2, 4], terminations=[True, False, False])
        reward, done, truncated, terminal_obs, active, _, active_world_ticks, matches = _held_action_interval(
            env, torch.tensor([2, 2, 2]), 6, torch.zeros((3, 1)))
        self.assertEqual(reward.tolist(), [1.0, 2.0, 4.0])
        self.assertEqual(terminal_obs[:, 0].tolist(), [1.0, 2.0, 4.0])
        self.assertEqual(done.tolist(), [True, False, False])
        self.assertEqual(truncated.tolist(), [False, True, True])
        self.assertEqual(active.tolist(), [False, False, False])
        self.assertEqual(env.applied_actions.tolist(), [1, 2, 4])
        self.assertEqual(matches, 2)
        self.assertEqual(active_world_ticks, 7)

    def test_true_termination_and_truncation_bootstrap_differently(self):
        rewards = torch.tensor([[1.0, 1.0]])
        dones = torch.tensor([[True, False]])
        truncated = torch.tensor([[False, True]])
        values = torch.zeros_like(rewards)
        terminal_next_values = torch.tensor([[9.0, 9.0]])
        advantages = _gae_advantages(
            rewards, dones, truncated, values, terminal_next_values,
            gamma=0.9, gae_lambda=0.95)
        self.assertAlmostEqual(float(advantages[0, 0]), 1.0)
        self.assertAlmostEqual(float(advantages[0, 1]), 9.1, places=5)

    def test_8000_tick_match_counts_one_horizon_completion(self):
        env = _FakeEnv([8000])
        obs = torch.zeros((1, 1))
        phase = 0.0
        matches = 0
        active_world_ticks_total = 0
        for _ in range(640):
            phase += 50.0 / 4.0
            ticks = max(1, int(phase))
            phase -= ticks
            remaining = 8000 - int(env.steps[0])
            result = _held_action_interval(env, torch.zeros(1, dtype=torch.long),
                                           ticks, obs)
            _, done, truncated, terminal_obs, _, _, active_world_ticks, count = result
            matches += count
            active_world_ticks_total += active_world_ticks
            self.assertEqual(active_world_ticks, min(ticks, remaining))
            obs = env.reset_done(done | truncated)
        self.assertEqual(env.steps.tolist(), [0])
        self.assertEqual(matches, 1)
        self.assertEqual(active_world_ticks_total, 8000)


if __name__ == "__main__":
    unittest.main()
