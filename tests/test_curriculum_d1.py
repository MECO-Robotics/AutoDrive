from types import SimpleNamespace

import torch

from frc_defense.curriculum_training import (D1HandWeightedPolicy, D2PredictivePolicy,
    _d1_artifacts, _d1_identity)
from frc_defense.learning_contract import CandidateTable, ObservationSnapshot


def _sensor_fixture(mask, *, candidate_indices=(0, 1, 2, 3),
                    candidate_valid=(True, True, True, True)):
    # Intentionally exposes only learner-side pose and sensor track fields.
    env = SimpleNamespace(n=1, device=torch.device('cpu'))
    env.steps = torch.tensor([0])
    track_points = torch.tensor([[1., 0.], [3., 0.], [6., 0.], [9., 0.], [2., .2]])
    env.track_pos = track_points.reshape(1, 1, 5, 2).expand(1, 6, 5, 2).clone()
    env.track_confidence = torch.tensor([.9, .8, .7, .6, .95]).reshape(1, 1, 5).expand(1, 6, 5).clone()
    env.track_age = torch.tensor([.1, .2, .3, .4, .1]).reshape(1, 1, 5).expand(1, 6, 5).clone()
    env.track_mask = torch.ones((1, 6, 5), dtype=torch.bool)
    env.sim = SimpleNamespace(pose=torch.tensor([[[0., 0., 0.]]*3 + [[0., 0., 0.]] + [[0., 0., 0.]]*2]))
    env.opponent_pose = torch.zeros((1, 6, 3))
    env.opponent_velocity = torch.zeros((1, 6, 3))
    env.opponent_age = torch.full((1, 6), float('inf'))
    env.opponent_valid = torch.zeros((1, 6), dtype=torch.bool)
    env._candidates = lambda: (torch.tensor([[list(candidate_indices)]*6]),
                               torch.tensor([[list(candidate_valid)]*6]),
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
    env, observation = _sensor_fixture([0, 0, 0, 0, 0, 0, 0, 1])
    action, trace = D1HandWeightedPolicy().choose(env, observation)
    assert action.tolist() == [7]
    assert trace['action_scores'][0, :7].isneginf().all()


def test_d1_candidate_ids_are_unique_with_placeholder_indices():
    env, observation = _sensor_fixture(
        [1, 0, 0, 0, 0, 0, 0, 1], candidate_indices=(0, 0, 0, 0),
        candidate_valid=(True, False, False, False))
    action, trace = D1HandWeightedPolicy().choose(env, observation,
                                                   snapshot_id='episode:0')
    table = trace['candidate_table']
    assert table.candidate_ids == (
        'collect:track:0', 'collect:unavailable-slot:1',
        'collect:unavailable-slot:2', 'collect:unavailable-slot:3',
        'score', 'intercept', 'ferry', 'hold')
    assert len(set(table.candidate_ids)) == 8
    assert table.valid_mask == (True, False, False, False, False, False, False, True)
    assert table.feature_names == (
        'action_index', 'sensor_track_buffer_index', 'target_distance_m',
        'nearby_sensed_track_count', 'track_confidence', 'track_age_s')
    assert table.features[1] == (1., None, None, None, None, None)
    assert table.route_costs == (None,) * 8
    assert action.item() == 0


def test_d1_rejects_a_mask_with_no_legal_fallback():
    env, observation = _sensor_fixture([0, 0, 0, 0, 0, 0, 0, 0])
    import pytest
    with pytest.raises(RuntimeError, match='at least one legal'):
        D1HandWeightedPolicy().choose(env, observation)


def test_d1_identity_is_stable_and_versioned_by_weights():
    assert _d1_identity() == _d1_identity()
    assert len(_d1_identity()) == 64


def test_d2_penalizes_collection_endpoints_but_leaves_fixed_rows_and_mask_unchanged():
    env, observation = _sensor_fixture([1, 1, 1, 1, 1, 1, 1, 1])
    env.opponent_pose[0, 3, :2] = torch.tensor([-2., 0.])
    env.opponent_velocity[0, 3, :2] = torch.tensor([20., 0.])
    env.opponent_age[0, 3] = .1
    env.opponent_valid[0, 3] = True
    _, d1 = D1HandWeightedPolicy().choose(env, observation)
    action, d2 = D2PredictivePolicy().choose(env, observation)
    assert d2['candidate_scores'][0, 0] < d1['candidate_scores'][0, 0]
    assert torch.equal(d2['action_scores'][:, 4:], d1['action_scores'][:, 4:])
    assert torch.equal(torch.isfinite(d2['action_scores']), torch.isfinite(d1['action_scores']))
    assert d2['predicted_opponent_endpoint'][0, 0].item() == 1.
    assert torch.isclose((d2['predicted_opponent_endpoint'][0] -
                          env.opponent_pose[0, 3, :2]).norm(), torch.tensor(3.))
    assert d2['candidate_table'].feature_names[-2:] == (
        'predicted_opponent_endpoint_distance_m', 'threat_penalty')
    assert d2['candidate_table'].features[4][-2:] == (None, None)
    assert action.item() in range(8)


def test_d2_missing_or_stale_opponent_falls_back_exactly_to_d1():
    env, observation = _sensor_fixture([1, 1, 1, 1, 1, 1, 1, 1])
    _, d1 = D1HandWeightedPolicy().choose(env, observation)
    d2_action, d2 = D2PredictivePolicy().choose(env, observation)
    d1_action, _ = D1HandWeightedPolicy().choose(env, observation)
    assert d2_action.tolist() == d1_action.tolist()
    assert torch.equal(d2['action_scores'], d1['action_scores'])
    assert torch.equal(d2['threat_penalty'], torch.zeros_like(d2['threat_penalty']))
    assert torch.isnan(d2['predicted_opponent_endpoint']).all()
    env.opponent_valid[0, 3] = True
    env.opponent_age[0, 3] = .6
    _, stale = D2PredictivePolicy().choose(env, observation)
    assert torch.equal(stale['action_scores'], d1['action_scores'])


def test_scorer_snapshot_ids_are_scoped_to_evaluation_run(monkeypatch):
    import frc_defense.curriculum_training as training

    def snapshot(_env, *, episode_id, sequence):
        return ObservationSnapshot(
            snapshot_id=f'{episode_id}:{sequence}', episode_id=episode_id,
            sequence=sequence, timestamp=0., raster_shape=(6, 1, 1),
            raster=(0.,) * 6, globals=(0.,) * 7)

    monkeypatch.setattr(training, 'build_observation_snapshot', snapshot)
    env = SimpleNamespace(steps=torch.tensor([0]))
    obs = torch.zeros((1, 137))
    obs[0, -8:] = 1.
    def trace_for(run_id):
        snapshot_id=f'{run_id}:pickup_simple:single:seed-9:0'
        table = CandidateTable(
            snapshot_id,
            ('collect:track:0', 'collect:track:1', 'collect:track:2',
             'collect:track:3', 'score', 'intercept', 'ferry', 'hold'),
            tuple((float(i), None, None, None, None, None) for i in range(8)),
            (True,) * 8,
            ('action_index', 'sensor_track_buffer_index', 'target_distance_m',
             'nearby_sensed_track_count', 'track_confidence', 'track_age_s'),
            (None,) * 8)
        return {
            'candidate_distance': torch.zeros((1, 4)),
            'candidate_nearby_tracks': torch.zeros((1, 4)),
            'candidate_confidence': torch.zeros((1, 4)),
            'candidate_age': torch.zeros((1, 4)),
            'action_scores': torch.zeros((1, 8)),
            'candidate_table': table,
            'selected_row': 0,
        }
    first = _d1_artifacts(env, obs, trace_for('run-a'), scenario='pickup_simple',
                          category='single', seed=9, evaluation_id='run-a')
    second = _d1_artifacts(env, obs, trace_for('run-b'), scenario='pickup_simple',
                           category='single', seed=9, evaluation_id='run-b')
    assert first['scorer_snapshot']['snapshot_id'] != second['scorer_snapshot']['snapshot_id']
    assert first['selected_candidate_id'] == 'collect:track:0'
    assert second['selected_candidate_id'] == 'collect:track:0'
