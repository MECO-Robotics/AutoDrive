from types import SimpleNamespace

import pytest
import torch

from frc_defense.field import qualification_field
from frc_defense.meco_phase4_context import (
    CONTACT_GEOMETRIES, START_PROFILE_NAMES, TEAMMATE_BEHAVIORS,
    ContactExposureProxyRecorder,
    TeammateBehaviorPool, apply_focal_pose_offset,
    apply_full_team_start_profile, bump_contact_exposure_geometry,
    contact_exposure_actions, contact_exposure_profile,
    full_team_start_profile, phase4_context_identity,
    phase4_start_profile_assignment, teammate_behavior_actions,
    phase4_teammate_behavior_assignment,
    validate_phase4_teammate_behavior_assignment,
    teammate_behavior_action_details,
    phase4_teammate_behavior_record,
    validate_full_team_start_profile, validate_phase4_start_profile_assignment,
    start_profile_identity,
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
    identity = start_profile_identity(profile)
    assert identity["factor_id"] == "start.legal_team_poses"
    assert identity["value_name"] == {"midfield_spread": "team_start_set_a", "crossing_lanes": "team_start_set_b", "opposing_lane_stagger": "team_start_set_c"}[profile]
    assert identity["config"]["runtime_profile_name"] == profile
    assert identity["config_sha256"] == phase4_context_identity(
        "legal_team_pose_sets", identity["value_name"])["config_sha256"]

    applied = apply_full_team_start_profile(env, profile)
    assert torch.equal(env.sim.pose, applied)
    assert not env.sim.velocity.any()
    assert env.phase4_start_profile_legality == [True, True]
    assert env.phase4_start_profile_validated_world_count == 2
    assert env.phase4_start_profile_validation_rate == 1.


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
    assert torch.allclose(torch.sin(poses[:, 2, 2]-direction),
                          torch.zeros_like(direction), atol=.01)
    identity = phase4_context_identity("robot_contact_geometry", geometry)
    assert identity["factor_id"] == "contact.robot_bump"
    assert identity["config"]["physics"] == "ordinary"


def test_contact_geometries_are_physically_distinct():
    env = _env(1)
    profiles = [contact_exposure_profile(env, geometry)[0, 2:4].clone()
                for geometry in CONTACT_GEOMETRIES]
    assert not torch.equal(profiles[0], profiles[1])
    assert not torch.equal(profiles[0], profiles[2])
    assert not torch.equal(profiles[1], profiles[2])


def test_contact_actions_require_fresh_track_directed_at_focal_and_use_defend_motion():
    class FakeEnv:
        n = 2
        device = torch.device("cpu")
        dt = .02
        perception_interval = 1

        def __init__(self, valid=True, age=0., target_slot=3, mask=True):
            self.opponent_valid = torch.zeros((2, 6), dtype=torch.bool)
            self.opponent_valid[:, 2] = valid
            self.opponent_age = torch.full((2, 6), float("inf"))
            self.opponent_age[:, 2] = age
            self.team_ids = torch.tensor([0, 0, 0, 1, 1, 1])
            self.sim = SimpleNamespace(pose=torch.zeros((2, 6, 3)))
            self.sim.pose[:, 3, :2] = torch.tensor([5., 5.])
            self.sim.pose[:, 4, :2] = torch.tensor([6., 5.])
            self.sim.pose[:, 5, :2] = torch.tensor([7., 5.])
            self.opponent_pose = torch.zeros((2, 6, 3))
            self.opponent_pose[:, 2, :2] = self.sim.pose[:, target_slot, :2]
            self.mask = mask

        def _candidates(self):
            valid = torch.zeros((2, 6, 4), dtype=torch.bool)
            return torch.zeros((2, 6, 4), dtype=torch.long), valid, torch.zeros((2, 6, 4))

        def _action_mask(self, indices, valid):
            del indices, valid
            mask = torch.zeros((2, 6, 8), dtype=torch.bool)
            mask[..., 7] = True
            mask[:, 2, 5] = self.mask
            return mask, torch.zeros((2, 6), dtype=torch.long)

    with pytest.raises(ValueError, match="fresh valid"):
        contact_exposure_actions(FakeEnv(valid=False))
    with pytest.raises(ValueError, match="fresh valid"):
        contact_exposure_actions(FakeEnv(age=.1))
    with pytest.raises(ValueError, match="not directed"):
        contact_exposure_actions(FakeEnv(target_slot=4))
    with pytest.raises(ValueError, match="action mask"):
        contact_exposure_actions(FakeEnv(mask=False))
    actions = contact_exposure_actions(FakeEnv())
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


def test_start_profile_assignment_is_registry_protocol_case_and_seed_bound():
    case = {"layout_id": "2026_rebuilt", "defender_family": "category_scripted",
            "start_condition": "scenario_defined", "sensor_profile": "standard"}
    assignment = phase4_start_profile_assignment("team_start_set_a",
        scenario_id="full_match_offense", split="validation",
        protocol_id="protocol-v1", protocol_sha256="a"*64,
        qualification_case=case, categories=("level_1/open",), seeds=(7, 11))
    assert validate_phase4_start_profile_assignment(assignment) == assignment
    assert assignment["factor_id"] == "start.legal_team_poses"
    assert assignment["seed_count"] == 2
    assert assignment["profile_config_sha256"] == start_profile_identity(
        "midfield_spread")["config_sha256"]
    assert assignment["outcome_by_start_state"] is None
    assert assignment["legal_start_validation_rate"] is None
    assert assignment["qualification_eligible"] is False

    changed_seed = phase4_start_profile_assignment("team_start_set_a",
        scenario_id="full_match_offense", split="validation",
        protocol_id="protocol-v1", protocol_sha256="a"*64,
        qualification_case=case, categories=("level_1/open",), seeds=(7, 12))
    changed_profile = phase4_start_profile_assignment("team_start_set_b",
        scenario_id="full_match_offense", split="validation",
        protocol_id="protocol-v1", protocol_sha256="a"*64,
        qualification_case=case, categories=("level_1/open",), seeds=(7, 11))
    assert changed_seed["assignment_sha256"] != assignment["assignment_sha256"]
    assert changed_profile["profile_config_sha256"] != assignment["profile_config_sha256"]
    with pytest.raises(ValueError, match="validation"):
        phase4_start_profile_assignment("team_start_set_a",
            scenario_id="full_match_offense", split="test",
            protocol_id="protocol-v1", protocol_sha256="a"*64,
            qualification_case=case, categories=("level_1/open",), seeds=(7, 11))
    tampered = dict(assignment, outcome_by_start_state={"team_start_set_a": 1.})
    with pytest.raises(ValueError, match="registry contract"):
        validate_phase4_start_profile_assignment(tampered)


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
    details = teammate_behavior_action_details(FakeActionEnv(), {4: "hold_position", 5: "score"})
    assert details["actions"].tolist() == [[7, 7, 7, 7, 7, 4]]
    assert details["requested_actions"].tolist() == [[7, 7, 7, 7, 7, 4]]
    assert not details["helper_fallback"].any()
    with pytest.raises(ValueError, match="unknown teammate behavior"):
        teammate_behavior_actions(FakeActionEnv(), {4: "idle"})


def test_teammate_behavior_assignment_is_exact_validation_pair_and_fails_closed():
    case = {"layout_id": "2026_rebuilt", "defender_family": "category_scripted",
        "start_condition": "scenario_defined", "sensor_profile": "standard"}
    common = dict(scenario_id="full_match_offense", split="validation",
        protocol_id="heldout-v1", protocol_sha256="a" * 64,
        qualification_case=case, categories=("level_1",), seeds=(11, 12), pool_seed=3)
    control = phase4_teammate_behavior_assignment(arm="no_factor", **common)
    treatment = phase4_teammate_behavior_assignment(arm="behavior_pool", **common)
    assert control["pair_sha256"] == treatment["pair_sha256"]
    assert control["behavior_schedule"] is None
    assert [row["seed"] for row in treatment["behavior_schedule"]] == [11, 12]
    assert all(set(row["assignments"]) == {4, 5}
               for row in treatment["behavior_schedule"])
    assert validate_phase4_teammate_behavior_assignment(treatment) == treatment
    assert treatment["qualification_eligible"] is False
    assert treatment["outcome_by_teammate_behavior"] is None
    assert treatment["teammate_collision_rate"] is None
    assert treatment["score_contribution"] is None
    changed = dict(treatment, protocol_sha256="b" * 64)
    with pytest.raises(ValueError, match="exact paired protocol binding"):
        validate_phase4_teammate_behavior_assignment(changed)
    with pytest.raises(ValueError, match="validation-only"):
        phase4_teammate_behavior_assignment(arm="behavior_pool",
            **{**common, "split": "test"})
    runtime = {"schema": "phase4-teammate-behavior-runtime-v1",
        "arm": "behavior_pool", "assignments": {
            str(slot): behavior for slot, behavior in
            treatment["behavior_schedule"][0]["assignments"].items()},
        "assignment_applied": True, "events": []}
    record = phase4_teammate_behavior_record(treatment, runtime,
        episode_id="episode-11", scenario_id="full_match_offense",
        category="level_1", seed=11, policy_id="D0", condition_id="condition-a",
        source_identity_sha256="c" * 64, expected_action_ticks=0)
    assert record["source_identity_sha256"] == "c" * 64
    assert record["protocol_sha256"] == "a" * 64
    assert record["assignment_pair_sha256"] == treatment["pair_sha256"]
    assert record["outcome_by_teammate_behavior"] is None
    assert record["teammate_collision_rate"] is None
    assert record["score_contribution"] is None
    with pytest.raises(ValueError, match="provenance"):
        phase4_teammate_behavior_record(treatment, runtime,
            episode_id="episode-11", scenario_id="full_match_offense",
            category="level_1", seed=99, policy_id="D0", condition_id="condition-a",
            source_identity_sha256="c" * 64, expected_action_ticks=0)


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
    recorder = ContactExposureProxyRecorder(1, dt=.02, stable_ticks=3)
    pose = torch.zeros((1, 6, 3))
    velocity = torch.zeros((1, 6, 3))
    radius = torch.full((1, 6), .5)
    contact = torch.zeros((1, 6), dtype=torch.bool)
    pose[0, 3, 0] = 1.0
    contact[0, 2] = contact[0, 3] = True
    recorder.record_tick(pose, velocity, contact, radius)
    assert recorder.pair_contact_proxy_first_tick.tolist() == [0]
    assert recorder.pair_contact_proxy_last_tick.tolist() == [0]
    assert recorder.pair_contact_proxy_tick_count.tolist() == [1]

    contact.zero_()
    pose[0, 3, 0] = 2.2
    for _ in range(3):
        recorder.record_tick(pose, velocity, contact, radius)
    summary = recorder.summary()
    assert summary["pair_contact_proxy_first_tick"] == [0]
    assert summary["pair_contact_proxy_last_tick"] == [0]
    assert summary["pair_contact_proxy_tick_count"] == [1]
    assert summary["post_contact_clearance_proxy_s"] == pytest.approx([.06])
    assert summary["factor_identity"]["factor_id"] == "contact.robot_bump"
    assert summary["registry_contact_signal_status"] == "unavailable_without_pair_impulse_signal"
    assert summary["registry_recovery_time_after_bump_s"] is None
    assert "proxy_only" in summary["promotion_status"]


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
