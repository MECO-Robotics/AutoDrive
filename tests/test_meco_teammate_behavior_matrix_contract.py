import pytest

from frc_defense.meco_ablation_matrix import (
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
