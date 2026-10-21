"""Controller-memory privacy checks; simulator qualification is also required."""
from types import SimpleNamespace

import torch

import frc_defense.tensor_3v3  # Initialize the shared simulator import graph.
from frc_defense.gamepiece_actions import _invalidate_perceived_contact_tracks


def contact_state():
    pose = torch.full((1, 6, 3), -20.)
    pose[:, :, 2] = 0.
    pose[0, 3, :2] = torch.tensor((7., 4.))
    tracks = torch.zeros(1, 6, 1, 2)
    tracks[0, 3, 0] = torch.tensor((9., 4.))
    mask = torch.zeros(1, 6, 1, dtype=torch.bool)
    mask[:, 3] = True
    return SimpleNamespace(
        sim=SimpleNamespace(pose=pose, length=torch.full((1, 6), .8),
                            width=torch.full((1, 6), .8)),
        team_ids=torch.tensor((0, 0, 0, 1, 1, 1)),
        track_pos=tracks, track_mask=mask, track_age=torch.zeros(1, 6, 1),
        opponent_pose=torch.zeros(1, 6, 3), opponent_size=torch.zeros(1, 6, 3),
        opponent_valid=torch.zeros(1, 6, dtype=torch.bool), _fuel_radius=.0825,
    )


def test_hidden_opponent_contact_cannot_erase_remembered_fuel():
    env = contact_state()
    env.sim.pose[0, 0, :2] = torch.tensor((9., 4.))
    _invalidate_perceived_contact_tracks(env, torch.tensor([True]))
    assert env.track_mask[0, 3, 0]
    assert env.track_age[0, 3, 0] == 0.


def test_observed_contact_retires_track_and_invalid_estimate_does_not():
    env = contact_state()
    env.opponent_valid[0, 3] = True
    env.opponent_pose[0, 3, :2] = torch.tensor((9., 4.))
    env.opponent_size[0, 3, :2] = .8
    _invalidate_perceived_contact_tracks(env, torch.tensor([True]))
    assert not env.track_mask[0, 3, 0]
    assert torch.isinf(env.track_age[0, 3, 0])
    env = contact_state()
    env.opponent_valid[0, 3] = True
    env.opponent_pose[0, 3] = float("nan")
    env.opponent_size[0, 3, :2] = .8
    _invalidate_perceived_contact_tracks(env, torch.tensor([True]))
    assert env.track_mask[0, 3, 0]


def test_own_and_teammate_contacts_retire_tracks_only_in_active_worlds():
    for robot in (3, 4):
        env = contact_state()
        env.sim.pose[0, robot, :2] = torch.tensor((9., 4.))
        _invalidate_perceived_contact_tracks(env, torch.tensor([False]))
        assert env.track_mask[0, 3, 0]
        _invalidate_perceived_contact_tracks(env, torch.tensor([True]))
        assert not env.track_mask[0, 3, 0]
