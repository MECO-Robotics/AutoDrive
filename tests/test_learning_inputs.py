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
        policy_pose=pose,
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


def test_ally_channel_uses_explicit_current_team_telemetry():
    env = _env()
    before = build_observation_snapshot(env)
    # Ally slots 4/5 are the documented team telemetry input, so moving one
    # changes only the ally raster channel and never the sensor track channels.
    env.sim.pose[0, 4, :2] = torch.tensor([6., 3.])
    after = build_observation_snapshot(env)
    before_raster = torch.tensor(before.raster).reshape(before.raster_shape)
    after_raster = torch.tensor(after.raster).reshape(after.raster_shape)
    assert not torch.equal(before_raster[2], after_raster[2])
    assert torch.equal(before_raster[:2], after_raster[:2])
    assert torch.equal(before_raster[4:], after_raster[4:])


def test_device_raster_snapshot_matches_legacy_and_keeps_frozen_copy():
    env = _env()
    legacy = build_observation_snapshot(env, episode_id="parity")
    device = build_observation_snapshot(env, episode_id="parity", device_raster=True)
    assert device.snapshot_id == legacy.snapshot_id
    assert device.timestamp == legacy.timestamp
    assert device.raster_shape == legacy.raster_shape
    assert device.globals == legacy.globals
    assert device.raster == legacy.raster
    device_tensor = device.raster_tensor(device=env.device)
    assert device_tensor.device == env.device
    assert torch.equal(device_tensor, torch.tensor(legacy.raster))
    device_tensor.zero_()
    device._device_raster.zero_()
    before = device.raster
    assert before == legacy.raster
    env.track_pos[0, 3, 0] = torch.tensor([6., 3.])
    assert device.raster == before


def test_device_raster_rejects_nonfinite_values():
    import pytest
    from frc_defense.learning_inputs import DeviceRasterObservationSnapshot
    with pytest.raises(ValueError, match="raster values must be finite"):
        DeviceRasterObservationSnapshot(snapshot_id="x:0", episode_id="x",
            sequence=0, timestamp=0., capture_monotonic=0., raster_shape=(1, 1, 1),
            _device_raster=torch.tensor([float("nan")]), globals=(0.,) * 7)


def test_device_raster_rejects_complex_values():
    import pytest
    from frc_defense.learning_inputs import DeviceRasterObservationSnapshot
    with pytest.raises(ValueError, match="raster values must be real"):
        DeviceRasterObservationSnapshot(snapshot_id="x:0", episode_id="x",
            sequence=0, timestamp=0., capture_monotonic=0., raster_shape=(1, 1, 1),
            _device_raster=torch.tensor([1.+2.j]), globals=(0.,) * 7)


def test_device_raster_integer_bool_values_serialize_as_legacy_floats():
    from frc_defense.learning_inputs import DeviceRasterObservationSnapshot
    for values in (torch.tensor([0, 1]), torch.tensor([False, True])):
        snapshot = DeviceRasterObservationSnapshot(snapshot_id="x:0", episode_id="x",
            sequence=0, timestamp=0., capture_monotonic=0., raster_shape=(1, 1, 2),
            _device_raster=values, globals=(0.,) * 7)
        assert snapshot.raster == (0., 1.)
        assert all(type(value) is float for value in snapshot.raster)


def test_device_raster_snapshot_value_equality_and_hash():
    from frc_defense.learning_inputs import DeviceRasterObservationSnapshot
    def snapshot(raster):
        return DeviceRasterObservationSnapshot(snapshot_id="x:0", episode_id="x",
            sequence=0, timestamp=0., capture_monotonic=0., raster_shape=(1, 1, 2),
            _device_raster=torch.tensor(raster), globals=(0.,) * 7)
    assert snapshot([0, 1]) == snapshot([0., 1.])
    assert hash(snapshot([0, 1])) == hash(snapshot([0., 1.]))
    assert snapshot([0, 1]) != snapshot([0, 2])
