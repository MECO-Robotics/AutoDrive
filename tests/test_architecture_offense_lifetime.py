"""An externally retained action callback must not keep GPU environments alive."""
import gc
import weakref

import pytest
import torch


def test_mask_behavior_and_environment_lifetime(monkeypatch):
    import frc_defense.architecture_offense_scenario as candidate

    class FakeEnv:
        def __init__(self, **kwargs):
            self.control_modes = kwargs["control_modes"]
            self.device = torch.device("cpu")
            self.agent_train_mask = torch.zeros(6, dtype=torch.bool)
            self._nn_mode_mask = torch.zeros(6, dtype=torch.bool)
            self._deterministic_mode_mask = torch.ones(6, dtype=torch.bool)

        def _action_mask(self, indices, valid):
            return torch.ones(1, 6, 8, dtype=torch.bool), torch.zeros(1, 6)

    monkeypatch.setattr(candidate, "TensorThreeVsThreeEnv", FakeEnv)
    monkeypatch.setattr(candidate, "reset_offense_scenario", lambda env, seed: None)
    env = candidate.make_offense_env(1, "cpu", 7)
    callback = env._action_mask
    mask, possession = callback(None, None)
    expected = torch.ones(1, 6, 8, dtype=torch.bool)
    expected[:, 3, 5:7] = False
    torch.testing.assert_close(mask, expected)
    torch.testing.assert_close(possession, torch.zeros(1, 6))
    reference = weakref.ref(env)
    # Keep the callback alive externally. It must neither retain the env nor
    # defer graph destruction until cyclic GC during a later CUDA capture.
    del env
    assert reference() is None
    gc.collect()
    assert reference() is None
    with pytest.raises(ReferenceError, match="environment no longer exists"):
        callback(None, None)
