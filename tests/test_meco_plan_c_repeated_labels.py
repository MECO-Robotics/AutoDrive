import copy
from types import SimpleNamespace

import pytest
import torch

from frc_defense import meco_counterfactual as counterfactual
from frc_defense import meco_supervised as supervised


def _synthetic_row():
    protocol_id = "protocol-test"
    repeat_protocol = {"repeat_protocol_id": "repeat-protocol-test"}
    outcomes = []
    for candidate_index, (candidate_id, values) in enumerate((
            ("left", (4.0, 5.0)), ("right", (2.0, 3.0)))):
        repeats = []
        for repeat_index, gross in enumerate(values):
            repeats.append({
                "gross_macro_return": gross / 2,
                "gross_horizon_return": gross,
                "discounted_horizon_return": gross,
                "continuation_policy": counterfactual.CONTINUATION_POLICY,
                "continuation_macro_horizon": 1,
                "discount_semantics": counterfactual.COUNTERFACTUAL_DISCOUNT_SEMANTICS,
                "terminal": False,
                "repeat_index": repeat_index,
                "repeat_seed": counterfactual.plan_c_repeat_seed(
                    protocol_id=protocol_id, scenario_id="scenario",
                    category="category", seed=7, decision_index=2,
                    repeat_index=repeat_index),
                "repeat_rng_provenance": {"generator_count": 2,
                    "seed_manifest_sha256": "a" * 64},
            })
        outcomes.append(counterfactual._plan_c_repeated_outcome(
            {"candidate_id": candidate_id, "candidate_index": candidate_index}, repeats))
    return {
        "plan_c_target_schema": counterfactual.PLAN_C_REPEAT_TARGET_SCHEMA,
        "plan_c_repeat_protocol_id": repeat_protocol["repeat_protocol_id"],
        "plan_c_counterfactuals": outcomes,
        "plan_c_pairwise_labels": counterfactual._plan_c_pairwise_labels(outcomes),
    }, repeat_protocol, protocol_id


def test_repeat_seed_is_domain_separated_and_matched_across_candidates():
    base = {"protocol_id": "p", "scenario_id": "s", "category": "c",
            "seed": 4, "decision_index": 3}
    first = counterfactual.plan_c_repeat_seed(**base, repeat_index=0)
    assert first == counterfactual.plan_c_repeat_seed(**base, repeat_index=0)
    assert first != counterfactual.plan_c_repeat_seed(**base, repeat_index=1)
    assert first != counterfactual.plan_c_repeat_seed(**{**base, "protocol_id": "other"},
                                                       repeat_index=0)


def test_environment_generators_are_reseeded_and_provenance_is_stable():
    env = SimpleNamespace(sim=SimpleNamespace(generator=torch.Generator(device="cpu")))
    first = counterfactual._seed_plan_c_environment_generators(env, 123)
    first_draw = torch.rand(4, generator=env.sim.generator)
    second = counterfactual._seed_plan_c_environment_generators(env, 123)
    second_draw = torch.rand(4, generator=env.sim.generator)
    assert first == second
    assert torch.equal(first_draw, second_draw)
    assert first["generator_count"] == 1

    counterfactual._seed_plan_c_environment_generators(env, 456)
    assert not torch.equal(first_draw, torch.rand(4, generator=env.sim.generator))

    with pytest.raises(RuntimeError, match="no simulator-local Torch generators"):
        counterfactual._seed_plan_c_environment_generators(SimpleNamespace(value=1), 123)


def test_pair_statistics_abstain_when_matched_repeat_signs_disagree():
    stable = counterfactual.plan_c_repeat_statistics([5, 7], [2, 3])
    assert stable["stable_preference"] is True
    assert stable["preferred_side"] == "left"
    assert stable["repeat_differences"] == [3., 4.]
    assert stable["sample_variance"] == pytest.approx(0.5)
    assert stable["standard_error"] == pytest.approx(0.5)

    unstable = counterfactual.plan_c_repeat_statistics([5, 2], [3, 4])
    assert unstable["stable_preference"] is False
    assert unstable["preferred_side"] is None

    with pytest.raises(ValueError, match="every finite matched repeat"):
        counterfactual.plan_c_repeat_statistics([3], [2, 1])


def test_repeated_label_validator_accepts_complete_matched_contract():
    row, repeat_protocol, protocol_id = _synthetic_row()
    assert supervised._validate_plan_c_repeated_labels(row,
        repeat_protocol=repeat_protocol, protocol_id=protocol_id,
        scenario_id="scenario", category="category", seed=7,
        decision_index=2)
    assert supervised._plan_c_stable_pair_indices(
        row["plan_c_counterfactuals"], row["plan_c_pairwise_labels"]) == [(0, 1, True)]


@pytest.mark.parametrize("corruption", [
    "missing_repeat", "wrong_seed", "variance", "label", "generator_provenance"])
def test_repeated_label_validator_fails_closed_on_incomplete_or_forged_evidence(corruption):
    row, repeat_protocol, protocol_id = _synthetic_row()
    row = copy.deepcopy(row)
    if corruption == "missing_repeat":
        row["plan_c_counterfactuals"][0]["repeat_returns"].pop()
    elif corruption == "wrong_seed":
        row["plan_c_counterfactuals"][0]["repeat_returns"][1]["repeat_seed"] += 1
    elif corruption == "variance":
        row["plan_c_counterfactuals"][0]["repeat_sample_variance_gross_horizon_return"] += 1
    elif corruption == "generator_provenance":
        row["plan_c_counterfactuals"][0]["repeat_returns"][1][
            "repeat_rng_provenance"]["seed_manifest_sha256"] = "b" * 64
    else:
        row["plan_c_pairwise_labels"][0]["preferred_id"] = "right"
    with pytest.raises(ValueError):
        supervised._validate_plan_c_repeated_labels(row,
            repeat_protocol=repeat_protocol, protocol_id=protocol_id,
            scenario_id="scenario", category="category", seed=7,
            decision_index=2)


def test_unstable_pairs_are_excluded_from_fitter_preferences():
    outcomes = [
        {"descriptor": {"candidate_id": "a", "candidate_index": 8}},
        {"descriptor": {"candidate_id": "b", "candidate_index": 3}},
        {"descriptor": {"candidate_id": "c", "candidate_index": 10}},
    ]
    labels = [
        {"left_id": "a", "right_id": "b", "stable_preference": True,
         "preferred_id": "a"},
        {"left_id": "a", "right_id": "c", "stable_preference": False,
         "preferred_id": None},
        {"left_id": "b", "right_id": "c", "stable_preference": True,
         "preferred_id": "c"},
    ]
    assert supervised._plan_c_stable_pair_indices(outcomes, labels) == [
        (8, 3, True), (3, 10, False)]
