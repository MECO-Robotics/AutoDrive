import pytest
import torch

from frc_defense.tensor_3v3 import OBS_DIM, TensorThreeVsThreeEnv


def _env(*, defer=True):
    return TensorThreeVsThreeEnv(num_envs=2, device="cpu", seed=11,
        randomize=True, fuel_count=96, fuel_physics=False,
        normalize_observations=False, defer_initial_perception=defer)


def test_deferred_first_observation_accepts_full_active_batch():
    env = _env()
    assert env._last_obs is None
    observation = env.observe(active_mask=torch.tensor([True, True]))
    assert observation.shape == (2, 6, OBS_DIM)
    assert torch.isfinite(observation).all()


def test_deferred_first_observation_rejects_partial_active_batch():
    env = _env()
    with pytest.raises(RuntimeError, match="before every world has an observation frame"):
        env.observe(active_mask=torch.tensor([True, False]))


def test_existing_observation_preserves_masked_inactive_worlds():
    env = _env(defer=False)
    previous = torch.full_like(env._last_obs, -7.)
    env._last_obs = previous
    observation = env.observe(active_mask=torch.tensor([True, False]))
    torch.testing.assert_close(observation[1], previous[1], rtol=0, atol=0)
    assert not torch.equal(observation[0], previous[0])


def test_deferred_reset_does_not_reuse_previous_episode_observation():
    env = _env(defer=False)
    stale = env._last_obs.clone()
    env.reset(seed=12, defer_perception=True)
    assert env._last_obs is None
    fresh = env.observe()
    assert fresh.shape == stale.shape
    assert not torch.equal(fresh, stale)
