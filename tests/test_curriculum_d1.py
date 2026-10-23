from types import SimpleNamespace

import torch

from frc_defense.curriculum_training import D1HandWeightedPolicy, _d1_identity


def _sensor_fixture(mask):
    # Intentionally exposes only learner-side pose and sensor track fields.
    env = SimpleNamespace(n=1, device=torch.device('cpu'))
    track_points = torch.tensor([[1., 0.], [3., 0.], [6., 0.], [9., 0.], [2., .2]])
    env.track_pos = track_points.reshape(1, 1, 5, 2).expand(1, 6, 5, 2).clone()
    env.track_confidence = torch.tensor([.9, .8, .7, .6, .95]).reshape(1, 1, 5).expand(1, 6, 5).clone()
    env.track_age = torch.tensor([.1, .2, .3, .4, .1]).reshape(1, 1, 5).expand(1, 6, 5).clone()
    env.track_mask = torch.ones((1, 6, 5), dtype=torch.bool)
    env.sim = SimpleNamespace(pose=torch.tensor([[[0., 0., 0.]]*3 + [[0., 0., 0.]] + [[0., 0., 0.]]*2]))
    env._candidates = lambda: (torch.tensor([[[0, 1, 2, 3]]*6]),
                               torch.tensor([[[True, True, True, True]]*6]),
                               torch.zeros((1, 6, 4)))
    observation = torch.zeros((1, 137))
    observation[:, -8:] = torch.tensor(mask, dtype=torch.float32)
    return env, observation


def test_d1_ranks_nearest_sensor_candidate_and_masks_actions():
    env, observation = _sensor_fixture([1, 1, 1, 1, 0, 0, 0, 1])
    action, trace = D1HandWeightedPolicy().choose(env, observation)
    assert action.tolist() == [0]
    assert trace['candidate_valid'][0].tolist() == [True, True, True, True]
    assert trace['action_scores'][0, 4:7].isneginf().all()
    assert trace['candidate_scores'][0, 0] > trace['candidate_scores'][0, 1]


def test_d1_uses_safe_hold_when_no_candidate_or_other_action_is_legal():
    env, observation = _sensor_fixture([0, 0, 0, 0, 0, 0, 0, 0])
    action, trace = D1HandWeightedPolicy().choose(env, observation)
    assert action.tolist() == [7]
    assert trace['action_scores'][0, :7].isneginf().all()


def test_d1_identity_is_stable_and_versioned_by_weights():
    assert _d1_identity() == _d1_identity()
    assert len(_d1_identity()) == 64
