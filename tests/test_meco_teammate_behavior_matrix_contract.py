import pytest

from frc_defense.meco_ablation_matrix import (
    _matrix_condition_selection_mode,
    _validate_matrix_contrast_inputs,
    default_conditions,
    _validate_teammate_behavior_condition_selection,
)


def test_teammate_behavior_matrix_adds_both_explicit_paired_arms():
    conditions = default_conditions("full_match_offense", categories=("level_1",),
        teammate_behavior_pool_seed=9)
    pair = [condition for condition in conditions
            if condition.teammate_behavior_arm is not None]
    assert [(row.condition_name, row.teammate_behavior_arm) for row in pair] == [
        ("phase4_teammate_behaviors_no_factor", "no_factor"),
        ("phase4_teammate_behaviors_behavior_pool", "behavior_pool"),
    ]
    assert {row.teammate_behavior_pool_seed for row in pair} == {9}
    _validate_teammate_behavior_condition_selection(pair, 9)
    with pytest.raises(ValueError, match="both teammate behavior pair arms"):
        _validate_teammate_behavior_condition_selection(pair[:1], 9)


def test_teammate_behavior_matrix_pair_is_validation_full_match_only():
    with pytest.raises(ValueError, match="full_match_offense"):
        default_conditions("offense_vs_scripted_pressure", categories=("level_1",),
            teammate_behavior_pool_seed=0)
    _validate_teammate_behavior_condition_selection(
        default_conditions("full_match_offense", categories=("level_1",)), None)


@pytest.mark.parametrize("factor", ["focal_pose", "teammate_behavior"])
def test_run_matrix_selection_contract_accepts_complete_paired_only_without_baseline(factor):
    if factor == "focal_pose":
        names = ("phase4_focal_pose_translation_small_no_offset",
                 "phase4_focal_pose_translation_small_offset")
        conditions = default_conditions("full_match_offense", categories=("level_1",),
            focal_pose_offsets=("translation_small",))
    else:
        names = ("phase4_teammate_behaviors_no_factor",
                 "phase4_teammate_behaviors_behavior_pool")
        conditions = default_conditions("full_match_offense", categories=("level_1",),
            teammate_behavior_pool_seed=3)
    selected = tuple(condition for condition in conditions
                     if condition.condition_name in names)
    # This is the same selection-mode validator used by run_matrix immediately
    # before its paired-only (no architecture baseline) summary branch.
    assert _matrix_condition_selection_mode(selected, names) == "paired_context_only"
    _validate_matrix_contrast_inputs("paired_context_only", {
        name: {} for name in names})
    with pytest.raises(ValueError, match="complete paired context factor"):
        _matrix_condition_selection_mode(selected[:1], names[:1])
