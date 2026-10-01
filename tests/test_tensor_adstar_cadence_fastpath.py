import unittest

import torch

from frc_defense.tensor_sim import TensorDefenseEnv


def _env():
    env = TensorDefenseEnv(
        num_envs=3, task="defense", device="cpu", opponent="mirror",
        action_mode="direct", randomize=False, fuel_count=96,
        adstar_replan_interval=4, field_colliders=[],
        observation_noise=0., observation_dropout=0.,
    )
    env.reset(seed=41)
    return env


class TensorADStarCadenceFastPathTests(unittest.TestCase):
    def test_aligned_scalar_cadence_matches_per_world_mask(self):
        scalar = _env()
        masked = _env()
        all_active = torch.ones(scalar.n, dtype=torch.bool)
        scalar_due = []
        masked_due = []

        for _ in range(13):
            scalar._planner_full_batch_active = True
            use_plan, plan_mask = scalar._advance_adstar_cadence(all_active)
            scalar_due.append(use_plan)
            if use_plan:
                self.assertIsNone(plan_mask)

            masked._planner_full_batch_active = False
            use_plan, plan_mask = masked._advance_adstar_cadence(all_active)
            masked_due.append(use_plan)
            if use_plan:
                self.assertTrue(torch.equal(plan_mask, all_active))

            self.assertTrue(torch.equal(scalar._planner_tick, masked._planner_tick))

        self.assertEqual(scalar_due, masked_due)
        self.assertEqual(scalar_due, [True, False, False, False] * 3 + [True])

    def test_partial_reset_disables_fast_path_until_full_reset(self):
        env = _env()
        self.assertTrue(env._planner_tick_aligned)

        env.reset_done(torch.tensor([True, False, False]))
        self.assertFalse(env._planner_tick_aligned)
        env._planner_tick.copy_(torch.tensor([3, 2, 3]))
        active = torch.tensor([True, False, True])
        use_plan, plan_mask = env._advance_adstar_cadence(active)
        self.assertTrue(use_plan)
        self.assertTrue(torch.equal(plan_mask, torch.tensor([True, False, True])))
        self.assertTrue(torch.equal(env._planner_tick, torch.tensor([4, 2, 4])))

        env.reset(seed=42)
        self.assertTrue(env._planner_tick_aligned)
        self.assertEqual(env._planner_tick_scalar, env.adstar_replan_interval - 1)
        self.assertTrue(torch.equal(env._planner_tick,
                                    torch.full_like(env._planner_tick,
                                                    env.adstar_replan_interval - 1)))

    def test_partial_step_mask_disables_fast_path(self):
        env = _env()
        env.step(torch.zeros((env.n, 3)), active_mask=torch.tensor([True, False, True]))
        self.assertFalse(env._planner_tick_aligned)
        self.assertFalse(env._planner_full_batch_active)

    def test_full_reset_done_restores_scalar_alignment(self):
        env = _env()
        env._planner_tick_aligned = False
        env._planner_tick_scalar = 1
        env._planner_tick.copy_(torch.tensor([1, 2, 3]))
        env.reset_done(torch.ones(env.n, dtype=torch.bool))
        self.assertTrue(env._planner_tick_aligned)
        self.assertEqual(env._planner_tick_scalar, env.adstar_replan_interval - 1)
        self.assertTrue(torch.equal(env._planner_tick,
                                    torch.full_like(env._planner_tick,
                                                    env.adstar_replan_interval - 1)))


if __name__ == "__main__":
    unittest.main()
