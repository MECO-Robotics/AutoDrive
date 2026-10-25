import copy

import pytest

from frc_defense.meco_phase4_outcome_bridge import (
    reduce_policy_family_outcomes,
    reduce_teammate_behavior_outcomes,
)


METRICS = {"success": True, "scores": 2.0,
    "useful_throughput": 0.5, "completed_cycles": 1.0}


def _family_row():
    return {"policy_id": "A1", "category": "active/open/level_0", "seed": 7,
        "policy_family": "reactive", "opponent_policy_id": "reactive_a",
        "control_episode_outcome": dict(METRICS),
        "treatment_episode_outcome": dict(METRICS),
        "outcome_semantics": (
            "whole-team paired episode outcome while the policy was assigned; "
            "association by policy family, not per-policy causal contribution")}


def _teammate_row():
    return {"policy_id": "A1", "category": "active/open/level_0", "seed": 7,
        "behavior": "score", "slot": 4,
        "control_whole_team_outcome": dict(METRICS),
        "treatment_whole_team_outcome": dict(METRICS),
        "outcome_semantics": (
            "paired whole-team episode outcomes while this behavior was present; "
            "concurrent behavior slots prevent individual score attribution")}


def _reduce_family(row):
    return reduce_policy_family_outcomes([row], expected_policy_ids={"A1"},
        categories=["active/open/level_0"], seeds=[7],
        family_by_policy={"A1": "reactive"})


def _reduce_teammate(row):
    return reduce_teammate_behavior_outcomes([row],
        expected_keys={("A1", "active/open/level_0", 7, "score")},
        expected_behaviors={"score"})


@pytest.mark.parametrize("reducer,row_factory", [
    (_reduce_family, _family_row),
    (_reduce_teammate, _teammate_row),
])
def test_outcome_reducers_accept_complete_finite_paired_metrics(reducer, row_factory):
    result = reducer(row_factory())
    assert result


@pytest.mark.parametrize("reducer,row_factory,arm", [
    (_reduce_family, _family_row, "control_episode_outcome"),
    (_reduce_family, _family_row, "treatment_episode_outcome"),
    (_reduce_teammate, _teammate_row, "control_whole_team_outcome"),
    (_reduce_teammate, _teammate_row, "treatment_whole_team_outcome"),
])
@pytest.mark.parametrize("damage", ["all_null", "one_null", "missing", "nonfinite"])
def test_outcome_reducers_reject_incomplete_or_censored_paired_metrics(
        reducer, row_factory, arm, damage):
    row = row_factory()
    if damage == "all_null":
        row[arm] = {name: None for name in METRICS}
    elif damage == "one_null":
        row[arm]["scores"] = None
    elif damage == "missing":
        row[arm].pop("completed_cycles")
    else:
        row[arm]["scores"] = float("nan")
    with pytest.raises(ValueError, match="incomplete or invalid whole-team outcome metrics"):
        reducer(row)


def test_outcome_reducers_do_not_mutate_original_episode_records():
    family = _family_row()
    teammate = _teammate_row()
    family_before, teammate_before = copy.deepcopy(family), copy.deepcopy(teammate)
    _reduce_family(family)
    _reduce_teammate(teammate)
    assert family == family_before
    assert teammate == teammate_before
