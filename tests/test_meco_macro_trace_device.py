from types import SimpleNamespace

import torch
import pytest

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


@pytest.mark.parametrize("action,event_field", [(4, "fuel_scored_event"),
                                               (6, "fuel_passed_event")])
@pytest.mark.parametrize("finish_near", [False, True])
def test_macro_preemption_and_alignment_are_accumulated(action, event_field, finish_near):
    class Env:
        n = 1
        device = torch.device("cpu")

        def __init__(self):
            self.last_actions = torch.zeros((1, 6), dtype=torch.long)
            self.sim = SimpleNamespace(pose=torch.zeros((1, 6, 3)))
            self.fuel_scored_event = torch.zeros((1, 6), dtype=torch.long)
            self.fuel_passed_event = torch.zeros((1, 6), dtype=torch.long)
            self.curriculum_metrics = SimpleNamespace(reward_delta=lambda: torch.ones(1))
            self.steps = 0

        def step(self, actions, **kwargs):
            self.steps += 1
            self.last_actions.fill_(action)
            getattr(self, event_field)[0, 3] = int(self.steps == 1)
            if self.steps == 2 and finish_near:
                self.sim.pose[0, 3, 0] = 1.
            return None, None, torch.tensor([self.steps == 2]), torch.tensor([False]), None

        def observe(self):
            return torch.zeros(1)

    result = _run_macro_step(Env(), action, 4, goal_xy=(1., 0.))
    assert result[1] is True
    assert result[3] == [action, action]
    assert result[6] is finish_near
    assert result[7] is not finish_near

@pytest.mark.parametrize("delay,expected", [(0, False), (2, True)])
def test_macro_observation_deferred_only_without_delay(delay, expected):
    class Env:
        n = 1
        device = torch.device("cpu")
        phase4_observation_delay_ticks = delay

        def __init__(self):
            self.last_actions = torch.zeros((1, 6), dtype=torch.long)
            self.sim = SimpleNamespace(pose=torch.zeros((1, 6, 3)))
            self.curriculum_metrics = SimpleNamespace(reward_delta=lambda: torch.ones(1))
            self.captures = []
            self.observations = 0

        def step(self, actions, **kwargs):
            self.captures.append(kwargs["capture_observation"])
            return None, None, torch.tensor([False]), torch.tensor([False]), None

        def observe(self):
            self.observations += 1
            return torch.zeros(1)

    env = Env()
    _run_macro_step(env, 7, 3, defer_observation=True)
    assert env.captures == [expected] * 3
    assert env.observations == 1
