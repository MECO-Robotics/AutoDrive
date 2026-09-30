import json
import tempfile
import unittest
from pathlib import Path

import torch

from frc_defense.tensor_sim import TensorDefenseEnv, TensorVectorizedSimulator
from frc_defense.tensor_training import generational_train


class MixedDefenseTrainingTests(unittest.TestCase):
    def test_mixed_batch_records_adstar_weighted_per_opponent_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generational_train(
                "defense", 1, Path(tmp), seed=91, num_envs=2, device="cpu",
                opponent="mixed", horizon=6, population_size=2, elite_count=1)
            self.assertEqual(result["opponents"], ["adstar", "offense", "intercept", "velocity_intercept", "mirror"])
            self.assertEqual(result["opponent_weights"], {"adstar": .8, "scripted": .2})
            self.assertEqual(result["completed_timesteps"], 1 * 2 * 2 * 6 * 5)
            self.assertEqual(set(result["opponent_fitness"]), set(result["opponents"]))
            scripted = sum(result["opponent_fitness"][name]
                           for name in ("offense", "intercept", "velocity_intercept", "mirror")) / 4
            expected = .8 * result["opponent_fitness"]["adstar"] + .2 * scripted
            self.assertAlmostEqual(result["best_fitness"], expected, places=4)
            record = json.loads((Path(tmp) / "generation-playback/generation-1.json").read_text())
            self.assertEqual(len(record["candidates"]), 2)
            self.assertEqual(set(record["candidates"][0]["opponent_metrics"]), set(result["opponents"]))
            self.assertTrue(record["candidates"][0]["frames"])

    def test_velocity_intercept_is_deterministic_and_uses_defender_velocity(self):
        env=TensorDefenseEnv(num_envs=1,task="defense",device="cpu",opponent="velocity_intercept",
            randomize=False,field_colliders=[],field_length=30.,field_width=20.,
            perception_config={"range_m":100.,"detection_dropout":0.,
                               "position_noise_m":0.,"velocity_noise_mps":0.})
        env.sim.pose[0,1,:2]=torch.tensor([5.,5.])
        env.sim.pose[0,0,:2]=torch.tensor([9.,5.])
        env.sim.speed[0,1]=4.
        env.goal[0]=torch.tensor([15.,5.])
        env.sim.velocity[0,0,:2]=torch.tensor([0.,.5])
        env._update_perception()
        positive=env._velocity_intercept_target().clone()
        self.assertTrue(torch.equal(positive,env._velocity_intercept_target()))
        env.sim.velocity[0,0,:2]=torch.tensor([0.,-.5])
        env._update_perception()
        negative=env._velocity_intercept_target()
        self.assertNotAlmostEqual(float(positive[0,1]),float(negative[0,1]))
        self.assertAlmostEqual(float(env._observed_robot(1,0)[1][0,1]),-.5,places=6)

    def test_mixed_policy_observation_does_not_receive_adstar_route_or_spawn_hint(self):
        env=TensorDefenseEnv(num_envs=1,task="defense",device="cpu",opponent="adstar",seed=17)
        env.opponent="offense"
        env.adstar_spawn_hint=False
        env._adstar_planners.defender_spawn=lambda *args,**kwargs: (_ for _ in ()).throw(
            AssertionError("AD* route used to place a training defender"))
        obs,_=env.reset(seed=17)
        self.assertEqual(tuple(obs.shape),(1,35))
        next_obs,_,_,_,info=env.step(torch.zeros((1,3)))
        self.assertEqual(tuple(next_obs.shape),(1,35))
        self.assertNotIn("adstar_paths",info)
        self.assertNotIn("predicted_intercepts",info)

    def test_random_endpoints_are_legal_and_in_distinct_zones(self):
        env=TensorDefenseEnv(num_envs=256,task="defense",device="cpu",seed=117,
                             opponent="offense")
        env.reset(seed=117)
        start=env.sim.pose[:,1,:2]
        goal=env.goal
        length,width=env.sim.field_length,env.sim.field_width
        depth=4.028
        zone=lambda x: torch.where(x<depth,0,torch.where(x>=length-depth,2,1))
        self.assertTrue(torch.all(zone(start[:,0])!=zone(goal[:,0])))
        for points in (start,goal):
            self.assertTrue(torch.all(points[:,0]>=.85))
            self.assertTrue(torch.all(points[:,0]<=length-.85))
            self.assertTrue(torch.all(points[:,1]>=.85))
            self.assertTrue(torch.all(points[:,1]<=width-.85))
            for box in env.sim.field_colliders:
                clear=((points[:,0]-box[0]).abs()>box[2]+.8)|((points[:,1]-box[1]).abs()>box[3]+.8)
                self.assertTrue(torch.all(clear))
        env.reset(seed=118,options={"start_zone":"red","goal_zone":"blue"})
        self.assertTrue(torch.all(zone(env.sim.pose[:,1,0])==0))
        self.assertTrue(torch.all(zone(env.goal[:,0])==2))
        with self.assertRaises(ValueError):
            env.reset(seed=119,options={"start_zone":"center","goal_zone":"center"})

    def test_static_and_opponent_contacts_have_distinct_flags(self):
        sim = TensorVectorizedSimulator(
            num_envs=1, device="cpu", randomize=False, field_length=20., field_width=20.,
            field_colliders=[(4., 4., .2, .6)])
        sim.pose[0, 0] = torch.tensor([3.4, 4., 0.])
        sim.pose[0, 1] = torch.tensor([12., 12., 0.])
        sim.velocity.zero_()
        sim.step(torch.zeros((1, 2, 3)))
        self.assertTrue(bool(sim.field_contact[0, 0]))
        self.assertFalse(bool(sim.opponent_contact[0]))

        sim.pose[0, 0] = torch.tensor([10., 10., 0.])
        sim.pose[0, 1] = torch.tensor([10.7, 10., 0.])
        sim.velocity.zero_()
        sim.step(torch.zeros((1, 2, 3)))
        self.assertTrue(bool(sim.opponent_contact[0]))
        self.assertFalse(bool(sim.field_contact.any()))

    def test_useful_opponent_contact_is_not_penalized_and_hold_gets_terminal_bonus(self):
        env = TensorDefenseEnv(
            num_envs=1, task="defense", device="cpu", opponent="offense",
            randomize=False, field_colliders=[], field_length=100., field_width=100., horizon=20)
        env.sim.pose[0, 0] = torch.tensor([50., 50., 0.])
        env.sim.pose[0, 1] = torch.tensor([60., 50., 0.])
        env.goal[0] = torch.tensor([10., 50.])
        env.previous.copy_((env.goal - env.sim.pose[:, 1, :2]).norm(dim=-1))
        env.sim.robot_contact.fill_(True)
        env.sim.opponent_contact.fill_(True)
        env.last_contact.zero_()
        _, reward, _, _, info = env._finish_step(
            env.goal, env.sim.pose[:, 0, :2].clone(), env.previous.clone(),
            torch.zeros((1, 3)))
        self.assertEqual(float(info["opponent_contact"][0]), 1.)
        contact_reward = float(reward[0])
        env.previous.copy_((env.goal - env.sim.pose[:, 1, :2]).norm(dim=-1))
        env.sim.robot_contact.zero_()
        env.sim.opponent_contact.zero_()
        env.last_contact.zero_()
        _, no_contact_reward, _, _, _ = env._finish_step(
            env.goal, env.sim.pose[:, 0, :2].clone(), env.previous.clone(),
            torch.zeros((1, 3)))
        self.assertAlmostEqual(contact_reward, float(no_contact_reward[0]), places=6)

        env.horizon = 1
        env.steps.fill_(0)
        env.last_static_contact.zero_()
        obs = env._obs()
        _, reward, done, truncated, info = env.step(torch.zeros((1, 3)))
        self.assertFalse(bool(done[0]))
        self.assertTrue(bool(truncated[0]))
        self.assertEqual(float(info["success"][0]), 1.)
        self.assertGreater(float(reward[0]), 8.)


if __name__ == "__main__":
    unittest.main()
