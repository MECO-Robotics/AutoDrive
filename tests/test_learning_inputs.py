from types import SimpleNamespace

import torch

from frc_defense.learning_inputs import build_observation_snapshot


def _env():
    pose = torch.tensor([[[2., 1., 0.], [30., 30., 0.], [31., 30., 0.],
                          [3., 2., .5], [4., 2., 0.], [5., 2., 0.]]])
    sim = SimpleNamespace(pose=pose, field_length=16.54, field_width=8.07)
    return SimpleNamespace(
        n=1,
        device=torch.device('cpu'),
        sim=sim,
        team_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        track_mask=torch.tensor([[[False, False], [False, False], [False, False],
                                  [True, True], [False, False], [False, False]]]),
        track_pos=torch.tensor([[[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]],
                                 [[0., 0.], [0., 0.]], [[1., 1.], [1.1, 1.]],
                                 [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]]),
        track_confidence=torch.tensor([[[0., 0.], [0., 0.], [0., 0.],
                                        [.8, .6], [0., 0.], [0., 0.]]]),
        track_age=torch.tensor([[[0., 0.], [0., 0.], [0., 0.],
                                 [.2, .4], [0., 0.], [0., 0.]]]),
        opponent_valid=torch.tensor([[True, False, False, True, False, False]]),
        opponent_pose=torch.tensor([[[8., 4., 0.], [0., 0., 0.], [0., 0., 0.],
                                     [7., 3., 0.], [0., 0., 0.], [0., 0., 0.]]]),
        opponent_age=torch.tensor([[.5, 0., 0., .3, 0., 0.]]),
        piece_owner=torch.tensor([[3, -1, -1]]),
        match_elapsed=torch.tensor([1.]),
        match_remaining=torch.tensor([159.]),
        hub_active=torch.tensor([[True, False]]),
        curriculum={'seed': 12, 'schedule': ((0., False), (4., True))},
        curriculum_schedule=((0., False), (4., True)),
        steps=torch.tensor([50]),
    )


def test_snapshot_has_proposal_shape_and_negative_track_age():
    snapshot = build_observation_snapshot(_env())
    assert snapshot.raster_shape == (6, 25, 51)
    assert len(snapshot.raster) == 6 * 25 * 51
    assert len(snapshot.globals) == 7
    assert snapshot.snapshot_id == 'curriculum-seed-12:50'
    raster = torch.tensor(snapshot.raster).reshape(snapshot.raster_shape)
    fuel_cell = (int(1. / .3302), int(1. / .3302))
    assert raster[0, fuel_cell[1], fuel_cell[0]] == .8
    assert raster[1, fuel_cell[1], fuel_cell[0]] < 0.
    ally_cell = (int(4. / .3302), int(2. / .3302))
    assert raster[2, ally_cell[1], ally_cell[0]] == 1.
    assert raster[3, ally_cell[1], ally_cell[0]] == 0.


def test_snapshot_does_not_read_hidden_fuel_or_opponent_positions():
    env = _env()
    before = build_observation_snapshot(env)
    env.piece_pos = torch.tensor([[[15., 7.], [2., 2.], [3., 3.]]])
    env.sim.pose[0, 0, :2] = torch.tensor([15., 7.])
    after = build_observation_snapshot(env)
    assert before.raster == after.raster
    assert before.globals == after.globals
