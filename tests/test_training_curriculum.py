"""Pure curriculum metadata/progression tests, with no simulator execution."""
from dataclasses import replace
import json

import pytest

from frc_defense.training_curriculum import (
    SCENARIOS, CategoryResult, CurriculumState, EvaluationResult,
    eligible_scenarios, evaluate_promotion, get_scenario, seed_partition, task_mixture,
)


def result(scenario_id, round_id='r', **changes):
    seeds = seed_partition(scenario_id, 'validation', 100)
    categories = tuple(CategoryResult(c, 100, 90, 1., 1., True, True, 'protocol', seeds, seeds)
                       for c in SCENARIOS[scenario_id].required_categories)
    return replace(EvaluationResult(round_id, scenario_id, 'validation', categories,
                                    'checkpoint', protocol_id='protocol'), **changes)


def qualified_state(*names):
    state = CurriculumState()
    for name in names:
        state.qualified.add(name)
        state.last_qualified_checkpoint[name] = 'old'
        state.qualified_protocol[name] = 'protocol'
        state.qualified_throughput[name] = {c: 1. for c in SCENARIOS[name].required_categories}
    return state


def ancestors(name):
    found = set()
    for parent in SCENARIOS[name].prerequisites:
        found.add(parent)
        found.update(ancestors(parent))
    return found


def test_registry_and_branches():
    assert len(SCENARIOS) == 12
    assert set(eligible_scenarios(CurriculumState())) == {'pickup_simple'}
    state = qualified_state('pickup_simple')
    assert {'pickup_scattered', 'active_shoot_preloaded', 'inactive_ferry_preloaded'} <= set(eligible_scenarios(state))
    assert 'active_carry_to_zone' not in eligible_scenarios(state)
    for name, scenario in SCENARIOS.items():
        assert scenario.scenario_id == name
        assert scenario.version == '1'
        assert scenario.time_budget_s > 0
        assert scenario.success_criterion and scenario.failure_criterion
        assert len(scenario.required_categories) == len(set(scenario.required_categories))
        assert scenario.gate.min_episodes >= 100
        assert scenario.gate.consecutive_evaluations == 3
    pressure = SCENARIOS['offense_vs_scripted_pressure'].required_categories
    assert 'active/trench/level_3_trench' in pressure
    assert 'inactive/bump/level_5_mixed' in pressure
    transition = SCENARIOS['hub_transition_cycles'].required_categories
    assert 'inactive_to_active/full/too_little_time/staged_at_home' in transition


def test_seed_pools_are_stable_disjoint_and_runtime_safe():
    pools = [seed_partition(s, split, 200, master_seed=m)
             for s in SCENARIOS for split in ('train', 'validation', 'test') for m in (0, 1)]
    assert sum(map(len, pools)) == len(set().union(*map(set, pools)))
    assert seed_partition('pickup_simple', 'validation', 200) == pools[2]
    assert all(0 <= seed < 2**63 for pool in pools for seed in pool)
    assert seed_partition('pickup_simple', 'train', 0) == ()


@pytest.mark.parametrize('args', [dict(count=True), dict(count=-1), dict(count=2**21),
                                      dict(count=1, master_seed=True), dict(count=1, master_seed=-1),
                                      dict(count=1, split='holdout')])
def test_seed_input_validation(args):
    kwargs = {'split': 'train', 'count': 1} | args
    with pytest.raises(ValueError):
        seed_partition('pickup_simple', **kwargs)


def test_three_consecutive_evaluations_allow_updated_checkpoints_and_ignore_duplicate():
    state = CurriculumState()
    assert evaluate_promotion(state, result('pickup_simple', '1', checkpoint_id='cp-initial')).passed
    before_duplicate = state.to_dict()
    duplicate = evaluate_promotion(state, result('pickup_simple', '1'))
    assert not duplicate.passed
    assert state.to_dict() == before_duplicate
    assert state.consecutive_passes['pickup_simple'] == 1
    assert state.consecutive_checkpoint['pickup_simple'] == 'cp-initial'
    poor = result('pickup_simple', '2')
    poor = replace(poor, categories=(replace(poor.categories[0], successes=89),) + poor.categories[1:])
    assert not evaluate_promotion(state, poor).passed
    assert state.consecutive_passes['pickup_simple'] == 0
    for i in range(3):
        decision = evaluate_promotion(state, result('pickup_simple', f'good-{i}', checkpoint_id=f'cp-{i}'))
        assert decision.promoted == (i == 2)
    assert state.last_qualified_checkpoint == {'pickup_simple': 'cp-2'}
    assert state.consecutive_checkpoint['pickup_simple'] == 'cp-2'
    assert not evaluate_promotion(state, result('pickup_simple', 'again')).promoted


def test_protocol_change_resets_streak_but_checkpoint_change_does_not():
    state = CurriculumState()
    assert evaluate_promotion(state, result('pickup_simple', 'round-1', checkpoint_id='policy-1')).passed
    assert evaluate_promotion(state, result('pickup_simple', 'round-2', checkpoint_id='policy-2')).passed
    assert state.consecutive_passes['pickup_simple'] == 2
    assert state.consecutive_checkpoint['pickup_simple'] == 'policy-2'
    assert evaluate_promotion(state, result('pickup_simple', 'round-3',
        checkpoint_id='policy-3', protocol_id='protocol-v2')).passed
    assert state.consecutive_passes['pickup_simple'] == 1
    assert not evaluate_promotion(state, result('pickup_simple', 'round-4',
        checkpoint_id='policy-4', protocol_id='protocol-v2')).promoted
    decision = evaluate_promotion(state, result('pickup_simple', 'round-5',
        checkpoint_id='policy-5', protocol_id='protocol-v2'))
    assert decision.promoted
    assert state.last_qualified_checkpoint['pickup_simple'] == 'policy-5'


@pytest.mark.parametrize('split', ['train', 'test'])
def test_training_and_final_test_cannot_promote_or_enter_seen_rounds(split):
    state = CurriculumState()
    assert not evaluate_promotion(state, result('pickup_simple', split=split)).passed
    assert state == CurriculumState()


@pytest.mark.parametrize('change', [dict(episodes=99), dict(episodes=True), dict(successes=True),
                                    dict(useful_throughput=float('nan')), dict(reference_throughput=float('inf'))])
def test_invalid_metrics_and_insufficient_sample_fail(change):
    evaluation = result('pickup_simple')
    evaluation = replace(evaluation, categories=(replace(evaluation.categories[0], **change),) + evaluation.categories[1:])
    assert not evaluate_promotion(CurriculumState(), evaluation).passed


def test_missing_category_cannot_be_hidden_by_high_overall_mean():
    evaluation = result('pickup_simple')
    assert not evaluate_promotion(CurriculumState(), replace(evaluation, categories=evaluation.categories[:1])).passed


@pytest.mark.parametrize('change', [dict(reference_qualified=False), dict(paired=False),
                                    dict(reference_protocol_id='other'), dict(reference_seeds=()),
                                    dict(reference_throughput=0), dict(useful_throughput=.79)])
def test_cycle_requires_qualified_paired_reference(change):
    name = 'active_collect_carry_score'
    state = qualified_state(*ancestors(name))
    evaluation = result(name)
    evaluation = replace(evaluation, categories=(replace(evaluation.categories[0], **change),) + evaluation.categories[1:])
    retained = tuple(result(s) for s in ancestors(name))
    assert not evaluate_promotion(state, evaluation, retention_results=retained).passed


def test_pressure_absolute_delivery_minimum_even_with_positive_reference():
    name = 'offense_vs_scripted_pressure'
    state = qualified_state(*ancestors(name))
    evaluation = result(name)
    evaluation = replace(evaluation, categories=tuple(replace(c, useful_throughput=.001, reference_throughput=.001) for c in evaluation.categories))
    assert not evaluate_promotion(state, evaluation, retention_results=tuple(result(s) for s in ancestors(name))).passed


def test_all_transitive_retention_and_regression_reset_streak():
    name = 'active_collect_carry_score'
    parents = ancestors(name)
    state = qualified_state(*parents)
    retained = tuple(result(s, '1') for s in sorted(parents))
    assert evaluate_promotion(state, result(name, '1'), retention_results=retained).passed
    retained = tuple(result(s, '2') for s in sorted(parents))
    poor = replace(retained[0], categories=tuple(replace(c, useful_throughput=.899) for c in retained[0].categories))
    assert not evaluate_promotion(state, result(name, '2'), retention_results=(poor,) + retained[1:]).passed
    assert state.consecutive_passes[name] == 0
    assert state.qualified == parents  # regression prompts practice; no implicit demotion
    assert not evaluate_promotion(state, result(name, '3'), retention_results=retained[:1]).passed
    for i in range(3):
        round_id = f'new-{i}'
        retained = tuple(result(s, round_id) for s in parents)
        assert evaluate_promotion(state, result(name, round_id), retention_results=retained).passed
    assert name in state.qualified


@pytest.mark.parametrize('change', [dict(checkpoint_id='other'), dict(protocol_id='other'),
                                    dict(evaluation_id='other'), dict(split='test'), dict(scenario_version='2')])
def test_retention_provenance_must_match(change):
    state = qualified_state('pickup_simple')
    retained = replace(result('pickup_simple'), **change)
    assert not evaluate_promotion(state, result('pickup_scattered'), retention_results=(retained,)).passed


def test_retention_all_categories_and_success_threshold():
    for categories in ((result('pickup_simple').categories[0],),
                       tuple(replace(c, successes=89) for c in result('pickup_simple').categories)):
        state = qualified_state('pickup_simple')
        retained = replace(result('pickup_simple'), categories=categories)
        assert not evaluate_promotion(state, result('pickup_scattered'), retention_results=(retained,)).passed


def test_json_round_trip_is_independent_and_strict():
    state = qualified_state('pickup_simple')
    state.consecutive_passes['pickup_scattered'] = 2
    state.consecutive_protocol['pickup_scattered'] = 'protocol'
    state.consecutive_checkpoint['pickup_scattered'] = 'latest-bundle'
    state.seen_evaluation_ids.add('round')
    serialized = json.loads(json.dumps(state.to_dict(), allow_nan=False))
    restored = CurriculumState.from_dict(serialized)
    assert restored == state
    serialized['qualified_throughput']['pickup_simple']['single'] = 100
    assert restored.qualified_throughput['pickup_simple']['single'] == 1
    for key, value in [('curriculum_version', 'invalid-version'), ('qualified', ['unknown']),
                       ('consecutive_passes', {'pickup_simple': True}),
                       ('consecutive_passes', {'pickup_simple': 4}),
                       ('last_qualified_checkpoint', {}), ('qualified_throughput', {})]:
        with pytest.raises(ValueError):
            CurriculumState.from_dict(state.to_dict() | {key: value})


def test_mixture_redistributes_and_rejects_impossible_tasks():
    state = qualified_state('pickup_simple')
    mixture = task_mixture(state, ('pickup_scattered',), ('active_shoot_preloaded',))
    assert mixture == pytest.approx({'pickup_scattered': .7, 'pickup_simple': .2, 'active_shoot_preloaded': .1})
    assert task_mixture(CurriculumState(), ('pickup_simple',)) == {'pickup_simple': 1.}
    assert task_mixture(state, ('pickup_scattered',)) == pytest.approx({'pickup_scattered': 7/9, 'pickup_simple': 2/9})
    for current, upcoming in [((), ()), (('full_match_offense',), ()),
                              (('pickup_scattered',), ('inactive_collect_ferry',)),
                              (('pickup_scattered',), ('pickup_scattered',))]:
        with pytest.raises(ValueError):
            task_mixture(state, current, upcoming)


@pytest.mark.parametrize('seeds', [(), tuple(range(100)), (1,) * 100,
                                   tuple(4*i + 2 for i in range(100)),
                                   (True,) + tuple(4*i + 1 for i in range(1, 100))])
def test_primitives_require_actual_unique_validation_seeds(seeds):
    evaluation = result('pickup_simple')
    categories = (replace(evaluation.categories[0], seeds=seeds),) + evaluation.categories[1:]
    assert not evaluate_promotion(CurriculumState(), replace(evaluation, categories=categories)).passed


def test_restore_rejects_completed_unqualified_streak():
    data = CurriculumState().to_dict()
    data['consecutive_passes'] = {'pickup_simple': 3}
    with pytest.raises(ValueError):
        CurriculumState.from_dict(data)


def test_retention_rejects_changed_baseline_protocol():
    state = qualified_state('pickup_simple')
    state.qualified_protocol['pickup_simple'] = 'old-suite'
    decision = evaluate_promotion(state, result('pickup_scattered'), retention_results=(result('pickup_simple'),))
    assert not decision.passed
    assert any('baseline protocol mismatch' in failure for failure in decision.failures)



def test_protocol_change_restarts_partial_streak_and_survives_resume():
    state = CurriculumState()
    assert evaluate_promotion(state, result('pickup_simple', '1')).passed
    assert evaluate_promotion(state, result('pickup_simple', '2')).passed
    state = CurriculumState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert evaluate_promotion(state, result('pickup_simple', '3', protocol_id='new-suite')).passed
    assert state.consecutive_passes['pickup_simple'] == 1
    assert 'pickup_simple' not in state.qualified
    assert state.consecutive_protocol['pickup_simple'] == 'new-suite'
    incomplete = state.to_dict()
    incomplete['consecutive_protocol'] = {}
    with pytest.raises(ValueError):
        CurriculumState.from_dict(incomplete)
