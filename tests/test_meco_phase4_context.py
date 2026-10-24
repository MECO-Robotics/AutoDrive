from types import SimpleNamespace

import pytest
import torch
import math

from frc_defense.field import qualification_field
from frc_defense.meco_phase4_context import (
    CONTACT_GEOMETRIES, TEAMMATE_BEHAVIORS, TeammateBehaviorPool,
    apply_focal_pose_offset, apply_full_team_start_profile,
    bump_contact_exposure_geometry, contact_exposure_profile,
    full_team_start_profile, teammate_behavior_actions,
    validate_full_team_start_profile,
)


def _env(n=2):
    pose = torch.zeros((n, 6, 3))
    return SimpleNamespace(
        n=n, device=torch.device("cpu"), field_boxes=qualification_field("2026_rebuilt"),
        sim=SimpleNamespace(
            field_length=16.54, field_width=8.07,
            pose=pose, velocity=torch.ones_like(pose),
            length=torch.full((n, 6), .9), width=torch.full((n, 6), .9)))


@pytest.mark.parametrize("profile", ["midfield_spread", "crossing_lanes"])
def test_full_team_start_profiles_are_legal_and_apply_only_at_reset(profile):
    env = _env()
    before = env.sim.pose.clone()
    poses = full_team_start_profile(env, profile)
    assert poses.shape == (2, 6, 3)
    assert torch.equal(env.sim.pose, before)
    validate_full_team_start_profile(env, poses)

    applied = apply_full_team_start_profile(env, profile)
    assert torch.equal(env.sim.pose, applied)
    assert not env.sim.velocity.any()


@pytest.mark.parametrize("geometry", CONTACT_GEOMETRIES)
def test_contact_exposure_profiles_start_clear_and_keep_all_six_robots_legal(geometry):
    env = _env()
    poses = contact_exposure_profile(env, geometry)
    validate_full_team_start_profile(env, poses)
    pair_distance = torch.linalg.vector_norm(poses[:, 2, :2]-poses[:, 3, :2], dim=-1)
    assert (pair_distance > .9).all()
    assert (pair_distance < 1.8).all()


def test_start_profile_validator_rejects_bounds_overlap_and_solid_obstacle():
    env = _env(1)
    poses = full_team_start_profile(env, "midfield_spread")
    outside = poses.clone()
    outside[0, 0, 0] = -.1
    with pytest.raises(ValueError, match="field bounds"):
        validate_full_team_start_profile(env, outside)
    overlap = poses.clone()
    overlap[0, 1] = overlap[0, 0]
    with pytest.raises(ValueError, match="overlapping"):
        validate_full_team_start_profile(env, overlap)
    blocked = poses.clone()
    blocked[0, 0] = blocked.new_tensor((5.24, 4.035, 0.))
    with pytest.raises(ValueError, match="solid field geometry"):
        validate_full_team_start_profile(env, blocked)


def test_behavior_pool_is_reproducible_and_balances_each_teammate_slot():
    first = TeammateBehaviorPool(seed=17)
    second = TeammateBehaviorPool(seed=17)
    schedule = [first.for_episode(i) for i in range(10)]
    assert schedule == [second.for_episode(i) for i in range(10)]
    for robot in first.robot_ids:
        assert {row[robot] for row in schedule} == set(TEAMMATE_BEHAVIORS)
    with pytest.raises(ValueError):
        first.for_episode(-1)


def test_behavior_actions_respect_existing_sensed_action_masks():
    class FakeActionEnv:
        n = 1
        device = torch.device("cpu")

        def _candidates(self):
            ids = torch.arange(24).reshape(1, 6, 4)
            valid = torch.tensor([[[1, 1, 0, 0]]*6], dtype=torch.bool)
            return ids, valid, torch.zeros((1, 6, 4))

        def _action_mask(self, indices, valid):
            del indices
            mask = torch.zeros((1, 6, 8), dtype=torch.bool)
            mask[..., :4] = valid
            mask[..., 4] = True
            mask[..., 5] = True
            mask[..., 6] = True
            mask[..., 7] = True
            return mask, torch.zeros((1, 6), dtype=torch.long)

    actions = teammate_behavior_actions(FakeActionEnv(), {
        0: "collect", 1: "score", 2: "ferry", 3: "defend", 4: "hold"})
    assert actions.tolist() == [[0, 4, 6, 5, 7, 7]]
    with pytest.raises(ValueError, match="unknown teammate behavior"):
        teammate_behavior_actions(FakeActionEnv(), {4: "idle"})


@pytest.mark.parametrize("normalized", [True, False])
def test_pose_offset_changes_only_policy_self_pose_and_wraps_heading(normalized):
    obs = torch.tensor([[.2, .3, .95, 1., 2., 3.]])
    original = obs.clone()
    updated, evidence = apply_focal_pose_offset(
        obs, (.4, -.2), .4, field_length=16., field_width=8., normalized=normalized)
    assert torch.equal(obs, original)
    scale_x, scale_y, scale_theta = ((16., 8., torch.pi) if normalized else (1., 1., 1.))
    assert updated[0, 0].item() == pytest.approx(.2 + .4/scale_x)
    assert updated[0, 1].item() == pytest.approx(.3 - .2/scale_y)
    angle = .95*scale_theta + .4
    expected_theta = math.atan2(math.sin(angle), math.cos(angle))/scale_theta
    assert updated[0, 2].item() == pytest.approx(expected_theta)
    assert torch.equal(updated[:, 3:], original[:, 3:])
    assert evidence["physical_truth_mutated"] is False
    assert evidence["channel_indices"] == [0, 1, 2]


def test_pose_offset_requires_and_changes_only_selected_row_in_full_team_observation():
    obs = torch.zeros((2, 6, 137))
    obs[..., 2] = .1
    updated, evidence = apply_focal_pose_offset(
        obs, (.2, .1), .05, field_length=16., field_width=8., focal_robot=3)
    assert torch.equal(updated[:, :3], obs[:, :3])
    assert torch.equal(updated[:, 4:], obs[:, 4:])
    assert (updated[:, 3, 0] > obs[:, 3, 0]).all()
    assert evidence["focal_robot"] == 3
    with pytest.raises(ValueError, match="focal_robot"):
        apply_focal_pose_offset(obs, (.2, .1), .05,
                                field_length=16., field_width=8.)


def test_bump_exposure_points_to_real_bump_and_does_not_inject_contact():
    result = bump_contact_exposure_geometry(_env(), "lower")
    assert result["bump_name"] == "blue_bump_lower"
    assert result["terrain_source"] == "simulator_bump_regions"
    assert result["contact_injected"] is False
    assert result["physics"] == "ordinary"
    assert 0 <= result["entry_pose"][0] <= 16.54
    assert 0 <= result["entry_pose"][1] <= 8.07
    red = bump_contact_exposure_geometry(_env(), "upper", alliance="red")
    assert red["bump_name"] == "red_bump_upper"
    assert red["entry_pose"][2] == pytest.approx(3.141592653589793)
