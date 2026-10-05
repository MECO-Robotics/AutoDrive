import types
import unittest

import torch

from frc_defense.tensor_sim import TensorDefenseEnv


def _reference_obs(self, initialize_mask=None, active_mask=None, active_count=None):
    """Pre-optimization observation postprocessing used as a parity oracle."""
    active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool) if active_mask is None
                 else torch.as_tensor(active_mask,device=self.device,dtype=torch.bool))
    raw=self._raw_obs(active_mask=active_mask)
    history_length=self.observation_history.shape[1]
    if initialize_mask is None:
        next_index=(self._observation_history_index+1).remainder(history_length)
        write_index=torch.where(active_mask,next_index,self._observation_history_index)
        current=self.observation_history[self._observation_world_index,write_index]
        updated=torch.where(active_mask[:,None],raw,current)
        self.observation_history[self._observation_world_index,write_index]=updated
        self._observation_history_index.copy_(write_index)
    else:
        initialize_mask=torch.as_tensor(initialize_mask,device=self.device,dtype=torch.bool)
        fresh=raw[:,None,:].expand(-1,self.observation_history.shape[1],-1)
        self.observation_history.copy_(torch.where(
            initialize_mask[:,None,None],fresh,self.observation_history))
        self._observation_history_index.copy_(torch.where(
            initialize_mask,torch.zeros_like(self._observation_history_index),
            self._observation_history_index))
    read_index=(self._observation_history_index-self.observation_delay).remainder(history_length)
    obs=self.observation_history.gather(
        1,read_index[:,None,None].expand(-1,1,self.obs_dim)).squeeze(1)
    if self.randomize and self.observation_noise>0:
        state=obs[:,:18]+self._random_active(active_mask,(18,),normal=True,
                                            active_count=active_count)*self.observation_noise
        obs=torch.cat((state,obs[:,18:]),-1)
    if self.randomize and self.observation_dropout>0:
        keep=self._random_active(active_mask,(18,),active_count=active_count)>=self.observation_dropout
        obs=torch.cat((obs[:,:18]*keep,obs[:,18:]),-1)
    return torch.where(active_mask[:,None],obs,self._last_observation)


class TensorObservationPostprocessingTests(unittest.TestCase):
    def _env(self, seed, noisy):
        return TensorDefenseEnv(
            num_envs=2, task="counter_defense", device="cpu", seed=seed,
            opponent="guard", action_mode="strategic", randomize=True,
            observation_noise=.035 if noisy else 0.,
            observation_dropout=.12 if noisy else 0.,
            max_observation_latency_steps=8, max_control_latency_steps=0,
            perception_config={"detection_dropout": 0., "position_noise_m": 0.,
                               "velocity_noise_mps": 0.},
        )

    def test_step_postprocessing_matches_reference_values_and_rng(self):
        seed = 741
        for full_batch in (True, False):
            for delay in (0, 8):
                for noisy in (False, True):
                    for return_info in (False, True):
                        with self.subTest(full_batch=full_batch, delay=delay, noisy=noisy,
                                          return_info=return_info):
                            optimized = self._env(seed, noisy)
                            reference = self._env(seed, noisy)
                            optimized.reset(seed=seed)
                            reference.reset(seed=seed)
                            reference._obs = types.MethodType(_reference_obs, reference)
                            optimized.observation_delay.fill_(delay)
                            reference.observation_delay.fill_(delay)
                            if full_batch:
                                active = torch.ones(2, dtype=torch.bool)
                            else:
                                active = torch.tensor([True, False])
                                optimized.observation_delay[1] = 8 - delay
                                reference.observation_delay[1] = 8 - delay
                            count = int(active.sum())
                            action = torch.tensor([0, 7], dtype=torch.long)
                            expected = reference.step(
                                action, active_mask=active, _active_count=count,
                                _return_info=return_info)
                            actual = optimized.step(
                                action, active_mask=active, _active_count=count,
                                _return_info=return_info)

                            for index, (left, right) in enumerate(zip(actual[:4], expected[:4])):
                                torch.testing.assert_close(
                                    left, right, rtol=0, atol=0,
                                    msg=f"step output {index} differs")
                            torch.testing.assert_close(
                                optimized.observation_history, reference.observation_history,
                                rtol=0, atol=0)
                            torch.testing.assert_close(
                                optimized._observation_history_index,
                                reference._observation_history_index, rtol=0, atol=0)
                            torch.testing.assert_close(
                                optimized._last_observation, reference._last_observation,
                                rtol=0, atol=0)
                            self.assertNotEqual(
                                optimized._last_observation.data_ptr(), actual[0].data_ptr(),
                                "cached last observation must not alias the returned tensor")
                            self.assertTrue(torch.equal(optimized.generator.get_state(),
                                                        reference.generator.get_state()))


if __name__ == "__main__":
    unittest.main()
