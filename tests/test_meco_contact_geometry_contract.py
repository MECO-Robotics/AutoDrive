import pytest
import torch

from frc_defense.meco_ablation_matrix import (
    MatrixCondition, _expected_contact_policy_ids,
    _validate_contact_condition_records, default_conditions,
)
from frc_defense.meco_phase4_context import (
    phase4_context_identity, validate_contact_geometry_split,
)
from frc_defense.curriculum_scenarios import replace_contact_opponent_action


def _contact_record(geometry_id, geometry_config, policy, policy_id, seed):
    return {
        "phase4_contact_exposure": {
            "geometry_id": geometry_id, "geometry_config": geometry_config,
            "exposure_assigned": True, "pair_slots": [2, 3],
            "opponent_action": 5,
            "opponent_action_source": "fresh_sensed_track_action_5",
            "slot2_env_control_mode": "nn", "approach_action_ticks": 12,
            "focal_action_overridden": False, "route_seen": False,
            "route_seen_time_s": None, "qualification_eligible": False,
            "recovery_time_after_bump_s": None, "route_completion_rate": None,
        },
        "phase4_pair_contact_event": {
            "event_source": "simulator_robot_pair_sat_contact_solver",
            "pair_slots": [2, 3], "qualification_eligible": False,
        },
        "policy": policy, "policy_id": policy_id,
        "category": "active/trench/level_1_stationary", "seed": seed,
    }


def test_contact_override_preserves_focal_slot_and_all_non_opponent_actions():
    selected = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    approach = torch.tensor([[7, 7, 5, 7, 7, 7], [7, 7, 5, 7, 7, 7]])
    result = replace_contact_opponent_action(selected, approach)
    assert result.tolist() == [[1, 2, 5, 4, 5, 6], [6, 5, 5, 3, 2, 1]]
    assert result[:, 3].tolist() == selected[:, 3].tolist()
    assert selected[:, 2].tolist() == [3, 4]


def test_contact_matrix_expected_policy_set_includes_all_three_p0_rows():
    policy_names = (
        "A1", "A2", "A3", "C1", "C2",
        "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2",
    )
    manifest = {name: {"policy_id": f"policy-id:{name}"}
        for name in policy_names}
    expected = _expected_contact_policy_ids(manifest)
    assert set(expected) == set(policy_names)
    assert len(expected) == 11
    del manifest["A2_P0"]
    with pytest.raises(ValueError, match="expected policy ID: A2_P0"):
        _expected_contact_policy_ids(manifest)


@pytest.mark.parametrize("geometry_id", [
    "crossing_contact", "side_contact", "recovery_contact",
])
def test_matrix_condition_binds_geometry_to_frozen_registry_config(geometry_id):
    config = phase4_context_identity("robot_contact_geometry", geometry_id)
    condition = MatrixCondition("contact", "cv", "cv", 16,
        contact_geometry_id=geometry_id, contact_geometry_config=config)
    condition.validate()
    tampered = dict(config)
    tampered["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen registry"):
        MatrixCondition("contact", "cv", "cv", 16,
            contact_geometry_id=geometry_id, contact_geometry_config=tampered).validate()


def test_contact_geometry_is_blocked_from_final_test_split():
    with pytest.raises(ValueError, match="final-test"):
        validate_contact_geometry_split("crossing_contact", "test")
    validate_contact_geometry_split("crossing_contact", "validation")
    validate_contact_geometry_split(None, "test")


def test_matrix_factory_exposes_contact_geometry_conditions_only_on_pressure_domain():
    conditions = default_conditions("offense_vs_scripted_pressure",
        categories=("active/trench/level_1_stationary",),
        contact_geometries=("crossing_contact", "side_contact"))
    contact = [condition for condition in conditions
        if condition.contact_geometry_id is not None]
    assert [condition.contact_geometry_id for condition in contact] == [
        "crossing_contact", "side_contact"]
    assert all(condition.contact_geometry_config == phase4_context_identity(
        "robot_contact_geometry", condition.contact_geometry_id)
        for condition in contact)
    with pytest.raises(ValueError, match="registered pressure/match"):
        default_conditions("pickup_simple", categories=("single",),
            contact_geometries=("crossing_contact",))


def test_contact_matrix_record_requires_separate_exposure_and_solver_event():
    geometry_id = "crossing_contact"
    geometry_config = phase4_context_identity("robot_contact_geometry", geometry_id)
    row = {
        "phase4_contact_exposure": {
            "geometry_id": geometry_id,
            "geometry_config": geometry_config,
            "exposure_assigned": True,
            "pair_slots": [2, 3],
            "opponent_action": 5,
            "opponent_action_source": "fresh_sensed_track_action_5",
            "slot2_env_control_mode": "nn",
            "approach_action_ticks": 0,
            "focal_action_overridden": False,
            "route_seen": False,
            "route_seen_time_s": None,
            "qualification_eligible": False,
            "recovery_time_after_bump_s": None,
            "route_completion_rate": None,
        },
        "phase4_pair_contact_event": {
            "event_source": "simulator_robot_pair_sat_contact_solver",
            "pair_slots": [2, 3],
            "opponent_action": 5,
            "opponent_action_source": "fresh_sensed_track_action_5",
            "slot2_env_control_mode": "nn",
            "distinct_contact_substeps_by_world": [0],
            "qualification_eligible": False,
        },
        "policy": "A1", "policy_id": "policy-a1",
        "category": "active/trench/level_1_stationary", "seed": 37,
    }
    _validate_contact_condition_records([row], geometry_id=geometry_id,
        geometry_config=geometry_config,
        expected_policy_ids={"A1": "policy-a1"},
        expected_category_seed_keys={("active/trench/level_1_stationary", 37)})
    row["phase4_pair_contact_event"] = None
    with pytest.raises(ValueError, match="authoritative pair-contact"):
        _validate_contact_condition_records([row], geometry_id=geometry_id,
            geometry_config=geometry_config,
            expected_policy_ids={"A1": "policy-a1"},
            expected_category_seed_keys={("active/trench/level_1_stationary", 37)})


def test_contact_matrix_records_cannot_claim_recovery_or_qualification():
    geometry_id = "side_contact"
    geometry_config = phase4_context_identity("robot_contact_geometry", geometry_id)
    row = {
        "phase4_contact_exposure": {
            "geometry_id": geometry_id,
            "geometry_config": geometry_config,
            "exposure_assigned": True,
            "pair_slots": [2, 3],
            "opponent_action": 5,
            "opponent_action_source": "fresh_sensed_track_action_5",
            "slot2_env_control_mode": "nn",
            "approach_action_ticks": 1,
            "focal_action_overridden": False,
            "route_seen": True,
            "route_seen_time_s": 0.02,
            "qualification_eligible": False,
            "recovery_time_after_bump_s": 1.2,
            "route_completion_rate": None,
        },
        "phase4_pair_contact_event": {
            "event_source": "simulator_robot_pair_sat_contact_solver",
            "pair_slots": [2, 3],
            "qualification_eligible": False,
        },
        "policy": "A1", "policy_id": "policy-a1",
        "category": "active/trench/level_1_stationary", "seed": 37,
    }
    with pytest.raises(ValueError, match="assigned exposure"):
        _validate_contact_condition_records([row], geometry_id=geometry_id,
            geometry_config=geometry_config,
            expected_policy_ids={"A1": "policy-a1"},
            expected_category_seed_keys={("active/trench/level_1_stationary", 37)})


def test_contact_exposures_are_paired_by_category_seed_across_policies():
    geometry_id = "crossing_contact"
    geometry_config = phase4_context_identity("robot_contact_geometry", geometry_id)
    base = {
        "phase4_contact_exposure": {
            "geometry_id": geometry_id,
            "geometry_config": geometry_config,
            "exposure_assigned": True,
            "pair_slots": [2, 3],
            "opponent_action": 5,
            "opponent_action_source": "fresh_sensed_track_action_5",
            "slot2_env_control_mode": "nn",
            "approach_action_ticks": 12,
            "focal_action_overridden": False,
            "route_seen": False,
            "route_seen_time_s": None,
            "qualification_eligible": False,
            "recovery_time_after_bump_s": None,
            "route_completion_rate": None,
        },
        "phase4_pair_contact_event": {
            "event_source": "simulator_robot_pair_sat_contact_solver",
            "pair_slots": [2, 3],
            "qualification_eligible": False,
        },
        "category": "active/trench/level_1_stationary",
        "seed": 37,
    }
    policy_ids = {"A1": "policy-a1", "D0": "policy-d0"}
    records = [{**base, "policy": policy, "policy_id": policy_ids[policy]}
        for policy in policy_ids]
    expected_pairs = {("active/trench/level_1_stationary", 37)}
    paired = _validate_contact_condition_records(records,
        geometry_id=geometry_id, geometry_config=geometry_config,
        expected_policy_ids=policy_ids,
        expected_category_seed_keys=expected_pairs)
    assert paired["paired_by_category_seed"] is True
    assert paired["approach_action_tick_counts_match"] is True
    with pytest.raises(ValueError, match="expected policy/category/seed"):
        _validate_contact_condition_records(records[:1],
            geometry_id=geometry_id, geometry_config=geometry_config,
            expected_policy_ids=policy_ids,
            expected_category_seed_keys=expected_pairs)


def test_contact_matrix_rejects_missing_whole_pair_and_empty_records():
    geometry_id = "crossing_contact"
    geometry_config = phase4_context_identity("robot_contact_geometry", geometry_id)
    with pytest.raises(ValueError, match="no episode records"):
        _validate_contact_condition_records([], geometry_id=geometry_id,
            geometry_config=geometry_config,
            expected_policy_ids={"A1": "policy-a1"},
            expected_category_seed_keys={("active/trench/level_1_stationary", 37)})
    policies = {"A1": "policy-a1", "D0": "policy-d0"}
    rows_for_first_pair = [_contact_record(geometry_id, geometry_config,
        policy, policy_id, 37) for policy, policy_id in policies.items()]
    with pytest.raises(ValueError, match="expected policy/category/seed"):
        _validate_contact_condition_records(rows_for_first_pair,
            geometry_id=geometry_id,
            geometry_config=geometry_config,
            expected_policy_ids=policies,
            expected_category_seed_keys={
                ("active/trench/level_1_stationary", 37),
                ("active/trench/level_1_stationary", 38)})
