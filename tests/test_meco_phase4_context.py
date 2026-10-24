from types import SimpleNamespace

import pytest
import torch

from frc_defense.field import qualification_field
from frc_defense.meco_phase4_context import (
    CONTACT_GEOMETRIES, START_PROFILE_NAMES, TEAMMATE_BEHAVIORS,
    ContactExposureRecorder,
    TeammateBehaviorPool, apply_focal_pose_offset,
    apply_full_team_start_profile, bump_contact_exposure_geometry,
    contact_exposure_actions, contact_exposure_profile,
    full_team_start_profile, phase4_context_identity,
    teammate_behavior_actions, validate_full_team_start_profile,
)


def _env(n=2):
    pose = torch.zeros((n, 6, 3))
    return SimpleNamespace(
        n=n, device=torch.device("cpu"), field_boxes=qualification_field("2026_rebuilt"),
        sim=SimpleNamespace(
            field_length=16.54, field_width=8.07,
            pose=pose, velocity=torch.ones_like(pose),
            length=torch.full((n, 6), .9), width=torch.full((n, 6), .9)))


@pytest.mark.parametrize("profile", START_PROFILE_NAMES)
def test_full_team_start_profiles_are_legal_and_apply_only_at_reset(profile):
    env = _env()
    before = env.sim.pose.clone()
    poses = full_team_start_profile(env, profile)
    assert poses.shape == (2, 6, 3)
    assert torch.equal(env.sim.pose, before)
    validate_full_team_start_profile(env, poses)
    identity = phase4_context_identity("legal_team_pose_sets", profile)
    assert identity["factor_id"] == "start.legal_team_poses"
    assert identity["config_sha256"] == phase4_context_identity(
        "legal_team_pose_sets", profile)["config_sha256"]

    applied = apply_full_team_start_profile(env, profile)
    assert torch.equal(env.sim.pose, applied)
    assert not env.sim.velocity.any()


@pytest.mark.parametrize("geometry", CONTACT_GEOMETRIES)
def test_contact_exposure_profiles_start_clear_and_face_each_other(geometry):
    env = _env()
    poses = contact_exposure_profile(env, geometry)
    validate_full_team_start_profile(env, poses)
    delta = poses[:, 3, :2] - poses[:, 2, :2]
    pair_distance = torch.linalg.vector_norm(delta, dim=-1)
    radius = .5 * (env.sim.length[:, 2].square()+env.sim.width[:, 2].square()).sqrt()
    radius += .5 * (env.sim.length[:, 3].square()+env.sim.width[:, 3].square()).sqrt()
    assert (pair_distance > radius).all()
    direction = torch.atan2(delta[:, 1], delta[:, 0])
    assert torch.allclose(torch.sin(poses[:, 2, 2]-direction), torch.zeros_like(direction), atol=.01)
    assert torch.allclose(torch.sin(poses[:, 3, 2]-(direction+torch.pi)),
                          torch.zeros_like(direction), atol=.01)
    identity = phase4_context_identity("bump_geometry", geometry)
    assert identity["factor_id"] == "contact.robot_bump"
    assert identity["config"]["physics"] == "ordinary"


def test_contact_actions_require_sensed_opponents_and_use_defend_motion():
    class FakeEnv:
        n = 2
        device = torch.device("cpu")

        def __init__(self, sensed):
            self.sensed = sensed

        def _candidates(self):
            valid = torch.zeros((2, 6, 4), dtype=torch.bool)
            return torch.zeros((2, 6, 4), dtype=torch.long), valid, torch.zeros((2, 6, 4))

        def _action_mask(self, indices, valid):
            del indices, valid
            mask = torch.zeros((2, 6, 8), dtype=torch.bool)
            mask[..., 7] = True
            mask[:, 2, 5] = self.sensed
            mask[:, 3, 5] = self.sensed
            return mask, torch.zeros((2, 6), dtype=torch.long)

    with pytest.raises(ValueError, match="sense the focal robot"):
        contact_exposure_actions(FakeEnv(False))
    actions = contact_exposure_actions(FakeEnv(True))
    assert actions.tolist() == [[7, 7, 5, 7, 7, 7]] * 2


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
    blocked[0, 0] = blocked.new_tensor((4.62534, 4.03479, 0.))
    with pytest.raises(ValueError, match="solid field geometry"):
        validate_full_team_start_profile(env, blocked)


def test_behavior_pool_is_reproducible_balanced_and_registry_bound():
    first = TeammateBehaviorPool(seed=17)
    second = TeammateBehaviorPool(seed=17)
    schedule = [first.for_episode(i) for i in range(10)]
    assert schedule == [second.for_episode(i) for i in range(10)]
    for robot in first.robot_ids:
        assert {row[robot] for row in schedule} == set(TEAMMATE_BEHAVIORS)
    metadata = first.metadata_for_episode(0)
    assert metadata["assignments"]["4"]["factor_id"] == "teammate.behavior_pool"
    assert len(metadata["assignments"]["4"]["config_sha256"]) == 64
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
        0: "collect", 1: "score", 2: "ferry", 3: "defend", 4: "hold_position"})
    assert actions.tolist() == [[0, 4, 6, 5, 7, 7]]
    with pytest.raises(ValueError, match="unknown teammate behavior"):
        teammate_behavior_actions(FakeActionEnv(), {4: "idle"})


def _observation():
    obs = torch.zeros((2, 6, 137))
    obs[..., 0] = .5
    obs[..., 1] = .5
    obs[..., 2] = .95
    obs[..., 3:5] = .2
    for x, y, radius in ((23, 24, 25), (26, 27, 28), (29, 30, 31), (32, 33, 34)):
        obs[..., x] = .6
        obs[..., y] = .4
        obs[..., radius] = .1
    for base in (35, 38, 43, 48, 53, 58, 68, 85, 87):
        obs[..., base] = .1
        obs[..., base+1] = -.05
    obs[..., 37] = .8
    candidates = obs[..., 89:129].reshape(2, 6, 4, 10)
    candidates[..., 0] = .2
    candidates[..., 1] = .1
    candidates[..., 2] = .3
    candidates[..., 3] = .4
    candidates[..., 5] = .45
    candidates[..., 9] = 1.
    return obs


def test_pose_offset_recomputes_relative_features_and_invalidates_route_cost():
    obs = _observation()
    original = obs.clone()
    updated, evidence = apply_focal_pose_offset(
        obs, "translation_small", field_length=16., field_width=8., focal_robot=3)
    assert torch.equal(obs, original)
    assert torch.equal(updated[:, :3], original[:, :3])
    assert torch.equal(updated[:, 4:], original[:, 4:])
    selected = updated[:, 3]
    source = original[:, 3]
    assert selected[0, 0].item() == pytest.approx(source[0, 0].item()+.30/16.)
    assert selected[0, 23].item() == pytest.approx(source[0, 23].item()-.30/16.)
    assert selected[0, 35].item() == pytest.approx(source[0, 35].item()-.30/16.)
    assert selected[:, 37].eq(0).all()
    for base in (38, 43, 48, 53, 58, 68, 85, 87):
        assert selected[0, base].item() == pytest.approx(source[0, base].item()-.30/16.)
    updated_candidates = selected[:, 89:129].reshape(2, 4, 10)
    source_candidates = source[:, 89:129].reshape(2, 4, 10)
    assert (updated_candidates[..., 0] < source_candidates[..., 0]).all()
    assert (updated_candidates[..., 2] < source_candidates[..., 2]).all()
    assert evidence["physical_truth_mutated"] is False
    assert evidence["training_no_leak_complete"] is False
    assert evidence["downstream_truth_readers_modified"] is False
    assert "candidate_own_eta" in evidence["recomputed_channel_groups"]
    assert evidence["invalidated_channels"] == [37]


def test_contact_metric_contract_counts_first_pair_tick_and_stable_recovery():
    recorder = ContactExposureRecorder(1, dt=.02, stable_ticks=3)
    pose = torch.zeros((1, 6, 3))
    velocity = torch.zeros((1, 6, 3))
    radius = torch.full((1, 6), .5)
    contact = torch.zeros((1, 6), dtype=torch.bool)
    pose[0, 3, 0] = 1.0
    contact[0, 2] = contact[0, 3] = True
    recorder.record_tick(pose, velocity, contact, radius)
    assert recorder.contact_realization_tick.tolist() == [0]
    assert recorder.contact_realization_count.tolist() == [1]

    contact.zero_()
    pose[0, 3, 0] = 2.2
    for _ in range(3):
        recorder.record_tick(pose, velocity, contact, radius)
    summary = recorder.summary()
    assert summary["contact_realization_tick"] == [0]
    assert summary["contact_realization_count"] == [1]
    assert summary["recovery_time_s"] == pytest.approx([.02])
    assert summary["factor_identity"]["factor_id"] == "contact.robot_bump"
    assert summary["contact_flag_scope"] == "distance_gated_pair_proxy_from_per_robot_flags"
    assert summary["promotion_status"] == "pending_replay_review"


def test_heading_offset_is_wrapped_and_registry_identity_is_stable():
    obs = _observation()[:, 3]
    updated, evidence = apply_focal_pose_offset(
        obs, "heading_offset", field_length=16., field_width=8.)
    assert (updated[:, 2].abs() <= 1).all()
    identity = evidence["factor_identity"]
    assert identity["factor_id"] == "observation.focal_pose_offset"
    assert identity["config_sha256"] == phase4_context_identity(
        "focal_pose_offsets", "heading_offset")["config_sha256"]


@pytest.mark.parametrize("alliance,side,name,heading", [
    ("blue", "lower", "blue_bump_lower", 3.141592653589793),
    ("red", "upper", "red_bump_upper", 0.)])
def test_bump_approach_starts_chassis_clear_and_faces_inward(alliance, side, name, heading):
    env = _env()
    result = bump_contact_exposure_geometry(env, side, alliance=alliance)
    box = next(box for box in env.field_boxes if box.name == name)
    x, y, theta = result["entry_pose"]
    radius = (.9**2+.9**2)**.5/2
    assert abs(x-box.x) >= box.length/2+radius+.049
    assert y == pytest.approx(box.y)
    assert theta == pytest.approx(heading)
    assert result["contact_injected"] is False
    # Facing the named alliance's bump center is the inward approach direction.
    delta = box.x-x
    assert delta*torch.tensor([torch.cos(torch.tensor(theta)).item()]).item() > 0
