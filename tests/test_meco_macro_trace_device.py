from types import SimpleNamespace

import torch

from frc_defense.meco_track_training import _run_macro_step, _training_fuel_physics_config


def test_only_plan_a_training_requires_qualifying_hip_fuel():
    for track in ("A1", "A2", "A3"):
        assert _training_fuel_physics_config(track) == {
            "backend": "hip", "gpu_qualifying": True}
    assert _training_fuel_physics_config("C1") is None


def test_macro_trace_snapshots_actions_and_excludes_inactive_ticks():
    class Env:
        n = 1
        device = torch.device("cpu")

        def __init__(self):
            self.last_actions = torch.zeros((1, 6), dtype=torch.long)
            self.sim = SimpleNamespace(pose=torch.zeros((1, 6, 3)))
            self.curriculum_metrics = SimpleNamespace(reward_delta=lambda: torch.ones(1))
            self.steps = 0

        def step(self, actions, **kwargs):
            self.steps += 1
            self.last_actions.fill_(self.steps)
            self._last_planner_collection_target_valid = torch.tensor(self.steps != 2)
            return None, None, torch.tensor([self.steps == 3]), torch.tensor([False]), None

        def observe(self):
            return torch.zeros(1)

    env = Env()
    result = _run_macro_step(env, 0, 4)
    assert result[1] is True
    assert result[3] == [1, 2]
    assert result[8] is True
