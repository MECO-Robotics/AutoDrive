"""Synthetic encoder contracts; no simulator game behavior is exercised."""

from types import SimpleNamespace

import torch

from frc_defense.architecture_inputs import PerceptionInputs


def _fake_env():
    n, slots = 2, 5
    pose = torch.zeros(n, 6, 3)
    pose[..., 0] = torch.arange(6) + 1.
    pose[..., 1] = 2.
    sim = SimpleNamespace(
        field_length=16., field_width=8., pose=pose,
        velocity=torch.ones(n, 6, 3), speed=torch.full((n, 6), 4.),
        length=torch.ones(n, 6), width=torch.ones(n, 6),
        field_colliders=torch.tensor([[8., 4., .5, .5]]),
        bump_regions=torch.tensor([[4., 4., .3, 2.]]),
        obstacles=torch.empty(0, 3),
    )
    mask = torch.zeros(n, 6, slots, dtype=torch.bool)
    mask[..., 0] = True
    positions = torch.zeros(n, 6, slots, 2)
    positions[..., 0, :] = torch.tensor([8., 4.])
    env = SimpleNamespace(
        n=n, device=torch.device("cpu"), sim=sim,
        team_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        hub_centers=torch.tensor([[2., 4.], [14., 4.]]),
        perception_track_timeout=1., track_mask=mask, track_pos=positions,
        track_vel=torch.zeros(n, 6, slots, 2), track_age=torch.zeros(n, 6, slots),
        track_confidence=torch.ones(n, 6, slots),
        track_spatial_extent=torch.full((n, 6, slots, 2), .15),
        opponent_pose=torch.zeros(n, 6, 3),
        opponent_velocity=torch.zeros(n, 6, 3),
        opponent_size=torch.ones(n, 6, 3),
        opponent_age=torch.full((n, 6), float("inf")),
        opponent_valid=torch.zeros(n, 6, dtype=torch.bool),
        # Truth exists in this fake object only to verify it is never consumed.
        piece_pos=torch.randn(n, slots, 2), piece_owner=torch.zeros(n, slots),
        piece_active=torch.ones(n, slots, dtype=torch.bool),
    )
    return env


def test_full_field_input_shapes_and_privileged_truth_is_ignored():
    env = _fake_env()
    encoder = PerceptionInputs(env, [0, 1])
    obs = torch.zeros(env.n, 6, 137)
    before = encoder(obs)
    assert before["field"].shape == (4, 12, 32, 64)
    assert before["entities"].shape == (4, 9, 12)
    assert before["entity_mask"].sum().item() == 16
    assert before["field"][:, 11].sum().item() == 4
    for value in before.values():
        assert torch.isfinite(value).all()
    env.piece_pos.fill_(900.)
    env.piece_owner.fill_(5)
    env.piece_active.zero_()
    env.sim.pose[:, 3:].fill_(float("nan"))
    env.sim.velocity[:, 3:].fill_(float("nan"))
    env.opponent_pose.fill_(float("nan"))
    env.opponent_velocity.fill_(float("nan"))
    env.track_pos[~env.track_mask] = float("nan")
    env.track_vel[~env.track_mask] = float("nan")
    after = encoder(obs)
    for key in before:
        torch.testing.assert_close(before[key], after[key])


def test_retained_track_and_robot_team_affect_expected_channels():
    env = _fake_env()
    encoder = PerceptionInputs(env, [0, 3])
    obs = torch.zeros(env.n, 6, 137)
    before = encoder(obs)
    env.track_vel[..., 0, 0] = 2.
    env.opponent_valid[:, 0] = True
    env.opponent_pose[:, 0, :2] = torch.tensor([12., 6.])
    after = encoder(obs)
    assert torch.all(after["field"][:, 3].amax((-2, -1)) == .5)
    assert after["field"][0, 8].sum() == 1.
    assert after["field"][1, 8].sum() == 0.
    torch.testing.assert_close(after["field"][0, 9], after["field"][1, 10])
    torch.testing.assert_close(after["field"][0, 10], after["field"][1, 9])
    assert not torch.equal(before["entities"], after["entities"])


def test_pickup_only_masks_nonpickup_actions_without_mutating_observation():
    env = _fake_env()
    env.architecture_pickup_only = True
    encoder = PerceptionInputs(env, [0, 3])
    obs = torch.zeros(env.n, 6, 137)
    original_mask = torch.tensor([1., 0., 1., 0., 1., 1., 1., 1.])
    obs[..., -8:] = original_mask
    before = obs.clone()
    inputs = encoder(obs)
    expected = torch.tensor([1., 0., 1., 0., 0., 0., 0., 1.]).expand(4, -1)
    torch.testing.assert_close(inputs["scalars"][:, -8:], expected)
    torch.testing.assert_close(obs, before)
    env.architecture_pickup_only = False
    unrestricted = encoder(obs)
    torch.testing.assert_close(unrestricted["scalars"][:, -8:], original_mask.expand(4, -1))
