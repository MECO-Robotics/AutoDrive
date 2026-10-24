from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from frc_defense.meco_phase4_context import (
    apply_phase4_focal_pose_assignment, phase4_focal_pose_assignment,
    phase4_focal_pose_evaluator_diagnostics,
    validate_phase4_focal_pose_assignment,
)
from frc_defense.meco_ablation_matrix import (
    _validate_focal_pose_condition_selection, default_conditions)


CASE = {"layout_id": "2026_rebuilt", "defender_family": "category_scripted",
        "start_condition": "scenario_defined", "sensor_profile": "standard"}


def _assignment(offset):
    return phase4_focal_pose_assignment(offset,
        paired_offset_name="translation_small", scenario_id="full_match_offense",
        split="validation", protocol_id="protocol-v1", protocol_sha256="a" * 64,
        qualification_case=CASE, categories=("active/open/level_1_stationary",),
        seeds=(71, 72))


def _state():
    truth = torch.zeros((2, 6, 3), dtype=torch.float64)
    truth[:, 3] = torch.tensor(((5.0, 2.0, .2), (7.0, 3.0, -.3)))
    sim = SimpleNamespace(pose=truth.clone(), velocity=torch.ones_like(truth),
        piece_pos=torch.arange(24, dtype=torch.float64).reshape(2, 4, 3),
        opponent_pose=torch.full((2, 2, 3), 4., dtype=torch.float64),
        contact_flags=torch.tensor(((False, True), (True, False))))
    return SimpleNamespace(n=2, sim=sim,
        estimated_pose=truth.clone(), policy_pose=truth.clone())


def test_focal_pose_pair_is_exactly_bound_and_fail_closed():
    baseline = _assignment(None)
    treatment = _assignment("translation_small")
    assert baseline["pair_assignment_sha256"] == treatment["pair_assignment_sha256"]
    assert baseline["arm"] == "no_offset"
    assert treatment["arm"] == "offset"
    assert validate_phase4_focal_pose_assignment(treatment) == treatment
    changed = deepcopy(treatment)
    changed["seeds"] = [71, 73]
    with pytest.raises(ValueError, match="frozen pair/protocol"):
        validate_phase4_focal_pose_assignment(changed)
    with pytest.raises(ValueError, match="full-match validation"):
        phase4_focal_pose_assignment("translation_small",
            paired_offset_name="translation_small", scenario_id="full_match_offense",
            split="test", protocol_id="protocol-v1", protocol_sha256="a" * 64,
            qualification_case=CASE, categories=("active/open/level_1_stationary",),
            seeds=(71,))


def test_assignment_changes_estimate_only_and_reports_evaluator_diagnostics():
    env = _state()
    physical = {name: getattr(env.sim, name).clone()
                for name in ("pose", "velocity", "piece_pos", "opponent_pose", "contact_flags")}
    event = apply_phase4_focal_pose_assignment(env, _assignment("translation_small"))
    assert all(torch.equal(getattr(env.sim, name), value)
               for name, value in physical.items())
    assert event["physical_truth_unchanged"] is True
    assert event["applied_before_policy_observation"] is True
    assert torch.equal(env.estimated_pose, env.policy_pose)
    torch.testing.assert_close(env.policy_pose[:, 3, 0], physical["pose"][:, 3, 0] + .30)
    assert "sim_truth_pose_m_rad" not in env.phase4_focal_pose_runtime
    record = phase4_focal_pose_evaluator_diagnostics(env, event)
    assert record["sim_truth_pose_m_rad"] == physical["pose"][:, 3].tolist()
    assert record["assignment_sha256"] == _assignment("translation_small")["assignment_sha256"]
    assert record["applied_before_policy_observation"] is True
    assert record["relocalization_time_s"] is None
    assert record["pose_error_recovery_rate"] is None
    assert record["qualification_eligible"] is False


def test_reset_reapplication_is_relative_to_truth_and_not_cumulative():
    env = _state()
    assignment = _assignment("translation_small")
    first_event = apply_phase4_focal_pose_assignment(env, assignment)
    first = phase4_focal_pose_evaluator_diagnostics(
        env, first_event)["policy_pose_estimate_m_rad"]
    # Model each reset's synchronization from physical truth before the reset hook.
    env.estimated_pose.copy_(env.sim.pose)
    env.policy_pose.copy_(env.sim.pose)
    second_event = apply_phase4_focal_pose_assignment(env, assignment)
    second = phase4_focal_pose_evaluator_diagnostics(
        env, second_event)["policy_pose_estimate_m_rad"]
    assert first == second
    torch.testing.assert_close(env.policy_pose[:, 3, 0], env.sim.pose[:, 3, 0] + .30)


def test_matrix_registers_both_arms_for_each_focal_offset():
    categories = ("active/open/level_1_stationary",)
    conditions = default_conditions("full_match_offense", categories=categories,
        qualification_case=CASE, focal_pose_offsets=("translation_small",))
    pair = [condition for condition in conditions
            if condition.focal_pose_pair_offset == "translation_small"]
    assert {(condition.condition_name, condition.focal_pose_offset) for condition in pair} == {
        ("phase4_focal_pose_translation_small_no_offset", None),
        ("phase4_focal_pose_translation_small_offset", "translation_small")}
    _validate_focal_pose_condition_selection(conditions,
        ("translation_small",))
    with pytest.raises(ValueError, match="both arms"):
        _validate_focal_pose_condition_selection(pair[:1], ("translation_small",))
