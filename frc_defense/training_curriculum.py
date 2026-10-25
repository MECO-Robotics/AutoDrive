"""Draft curriculum metadata and outcome gates; this module runs no simulation.

Budgets and absolute delivery minima are provisional until scripted-reference
qualification. Category rates are measured externally, never inferred from reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import math

CURRICULUM_VERSION = '2'
_SPLITS = {'train': 0, 'validation': 1, 'test': 2, 'calibration': 3}


@dataclass(frozen=True)
class GateDefinition:
    min_episodes: int = 100
    consecutive_evaluations: int = 3
    success_rate: float = .9
    throughput_ratio: float = 0.
    min_useful_throughput: float = 0.


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    version: str
    stage: str
    prerequisites: tuple[str, ...]
    required_categories: tuple[str, ...]
    time_budget_s: float
    gate: GateDefinition
    success_criterion: str
    failure_criterion: str


_PRIMITIVE = GateDefinition()
_CYCLE = GateDefinition(throughput_ratio=.8)
_PRESSURE = GateDefinition(throughput_ratio=.8, min_useful_throughput=.01)
_ROUTES = ('open', 'trench', 'bump')
_BOTH_ROUTES = tuple(f'{phase}/{route}' for phase in ('active', 'inactive') for route in _ROUTES)
_TRANSITIONS = tuple(
    f'{direction}/{load}/{timing}/{situation}'
    for direction, situations in (
        ('active_to_inactive', ('collecting', 'carrying_home', 'approaching_shot')),
        ('inactive_to_active', ('before_release', 'after_release', 'staged_at_home')),
    )
    for load in ('empty', 'partial', 'full')
    for timing in ('enough_time', 'too_little_time')
    for situation in situations
) + ('empty_collection', 'hold_useful_load')
_PRESSURE_CATEGORIES = tuple(
    f'{phase}/{route}/{blocker}'
    for phase in ('active', 'inactive') for route in _ROUTES
    for blocker in ('level_0', 'level_1_stationary', 'level_2_patrol',
                    'level_3_trench', 'level_3_bump', 'level_3_seeded_choice',
                    'level_4_reactive', 'level_5_mixed')
    if (blocker != 'level_3_trench' or route == 'trench')
    and (blocker != 'level_3_bump' or route == 'bump')
    and (blocker != 'level_3_seeded_choice' or route != 'open')
)


def _scenario(name, stage, prerequisites, categories, budget, gate, success):
    return ScenarioDefinition(name, '1', stage, prerequisites, categories, budget,
                              gate, success, 'Timeout or failure of the stated physical outcome; reward and action requests do not count.')


SCENARIOS = {s.scenario_id: s for s in (
    _scenario('pickup_simple', '1', (), ('single', 'cluster'), 20., _PRIMITIVE,
              'Acquire and retain the required distinct FUEL.'),
    _scenario('pickup_scattered', '2', ('pickup_simple',), ('sparse', 'dense', 'initially_hidden'), 40., _PRIMITIVE,
              'Retain the required fraction of distinct available FUEL without reacquisition credit.'),
    _scenario('pickup_multi_zone', '3', ('pickup_scattered',), ('productive', 'empty', 'depleted'), 60., _PRIMITIVE,
              'Leave empty or depleted zones and retain the required amount from productive zones.'),
    _scenario('active_carry_to_zone', '4A', ('pickup_multi_zone',), ('active/trench', 'active/bump'), 35., _PRIMITIVE,
              'Reach a valid scoring area with required load retained before the active window closes.'),
    _scenario('active_shoot_preloaded', '4B', ('pickup_simple',), ('active/preloaded', 'active/approach', 'active/heading_error'), 25., _PRIMITIVE,
              'Actually score the required FUEL while the friendly HUB is active.'),
    _scenario('inactive_ferry_preloaded', '4C', ('pickup_simple',), ('inactive/preloaded', 'inactive/approach', 'inactive/alignment', 'inactive/partial_load', 'inactive/full_load', 'inactive/landing'), 30., _PRIMITIVE,
              'Distinct eligible outside-to-home FUEL settles in friendly staging and remains available after settling.'),
    _scenario('active_collect_carry_score', '5A', ('pickup_multi_zone', 'active_carry_to_zone', 'active_shoot_preloaded'), tuple(f'active/{r}' for r in _ROUTES), 90., _CYCLE,
              'Repeat collection and actual scoring cycles with useful scored throughput.'),
    _scenario('inactive_collect_ferry', '5B', ('pickup_multi_zone', 'inactive_ferry_preloaded'), tuple(f'inactive/{r}' for r in _ROUTES), 90., _CYCLE,
              'Repeat eligible outside-to-home deliveries with net useful staged FUEL and no recycling credit.'),
    _scenario('hub_transition_cycles', '6', ('active_collect_carry_score', 'inactive_collect_ferry'), _TRANSITIONS, 120., _CYCLE,
              'Choose useful objectives across boundaries and convert staged FUEL to actual scoring after activation.'),
    _scenario('offense_vs_scripted_pressure', '7', ('hub_transition_cycles',), _PRESSURE_CATEGORIES, 120., _PRESSURE,
              'Deliver useful FUEL under each feasible blocker category and recover from interruptions.'),
    _scenario('team_delivery_cycles', '8', ('offense_vs_scripted_pressure',), tuple(f'{c}/{team}' for c in _BOTH_ROUTES for team in ('receiver', 'multiple_teammates', 'opponents')), 150., _CYCLE,
              'Convert deliveries into team scoring while sharing routes and collection areas.'),
    _scenario('full_match_offense', '9', ('team_delivery_cycles', 'offense_vs_scripted_pressure', 'hub_transition_cycles'), _PRESSURE_CATEGORIES, 160., _PRESSURE,
              'Contribute across the complete real schedule, encounters, and staged-FUEL conversion on unseen layouts.'),
)}


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    return SCENARIOS[scenario_id]


def seed_partition(scenario_id: str, split: str, count: int, *, master_seed: int = 0) -> tuple[int, ...]:
    """Stable signed-63-bit seeds; split bits guarantee separation across pools.

    A 40-bit scenario/master hash namespaces a 21-bit counter. Hash collisions
    between namespaces are possible; splits remain disjoint even then.
    """
    get_scenario(scenario_id)
    if split not in _SPLITS:
        raise ValueError('split must be train, validation, test, or calibration')
    if type(count) is not int or not 0 <= count < 2**21:
        raise ValueError('count must be an integer in [0, 2**21)')
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError('master_seed must be a nonnegative integer')
    prefix = int.from_bytes(sha256(f'{CURRICULUM_VERSION}/{scenario_id}/{master_seed}'.encode()).digest()[:5], 'big')
    return tuple((prefix << 23) | (i << 2) | _SPLITS[split] for i in range(count))


@dataclass(frozen=True)
class CategoryResult:
    category: str
    episodes: int
    successes: int
    useful_throughput: float = 0.
    reference_throughput: float = 0.
    paired: bool = False
    reference_qualified: bool = False
    reference_protocol_id: str = ''
    seeds: tuple[int, ...] = ()
    reference_seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_id: str
    scenario_id: str
    split: str
    categories: tuple[CategoryResult, ...]
    checkpoint_id: str = ''
    scenario_version: str = '1'
    curriculum_version: str = CURRICULUM_VERSION
    protocol_id: str = ''
    numeric_gate_verdict: dict | None = None


@dataclass
class CurriculumState:
    qualified: set[str] = field(default_factory=set)
    consecutive_passes: dict[str, int] = field(default_factory=dict)
    last_qualified_checkpoint: dict[str, str] = field(default_factory=dict)
    qualified_throughput: dict[str, dict[str, float]] = field(default_factory=dict)
    seen_evaluation_ids: set[str] = field(default_factory=set)
    qualified_protocol: dict[str, str] = field(default_factory=dict)
    consecutive_protocol: dict[str, str] = field(default_factory=dict)
    consecutive_checkpoint: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'curriculum_version': CURRICULUM_VERSION, 'qualified': sorted(self.qualified),
                'consecutive_passes': dict(self.consecutive_passes),
                'last_qualified_checkpoint': dict(self.last_qualified_checkpoint),
                'qualified_throughput': {s: dict(c) for s, c in self.qualified_throughput.items()},
                'seen_evaluation_ids': sorted(self.seen_evaluation_ids),
                'qualified_protocol': dict(self.qualified_protocol),
                'consecutive_protocol': dict(self.consecutive_protocol),
                'consecutive_checkpoint': dict(self.consecutive_checkpoint)}

    @classmethod
    def from_dict(cls, data: dict) -> CurriculumState:
        expected = set(cls().to_dict())
        if not isinstance(data, dict) or set(data) != expected or data['curriculum_version'] != CURRICULUM_VERSION:
            raise ValueError('invalid state fields or curriculum version')
        for key in ('qualified', 'seen_evaluation_ids'):
            values = data[key]
            if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(values):
                raise ValueError(f'invalid {key}')
        for key in ('consecutive_passes', 'last_qualified_checkpoint', 'qualified_throughput', 'qualified_protocol', 'consecutive_protocol', 'consecutive_checkpoint'):
            if not isinstance(data[key], dict) or any(s not in SCENARIOS for s in data[key]):
                raise ValueError(f'invalid {key}')
        qualified = set(data['qualified'])
        if not qualified <= SCENARIOS.keys():
            raise ValueError('unknown qualified scenario')
        if any(type(n) is not int or not 0 <= n <= 3 for n in data['consecutive_passes'].values()):
            raise ValueError('invalid pass count')
        streaks = {s for s, n in data['consecutive_passes'].items() if n > 0}
        if (set(data['consecutive_protocol']) != streaks
                or set(data['consecutive_checkpoint']) != streaks
                or any(not isinstance(v, str) or not v for key in ('consecutive_protocol', 'consecutive_checkpoint') for v in data[key].values())):
            raise ValueError('positive streaks require protocol and checkpoint identity')
        if set(data['last_qualified_checkpoint']) != qualified or set(data['qualified_throughput']) != qualified or set(data['qualified_protocol']) != qualified:
            raise ValueError('qualified scenarios require checkpoint and category baselines')
        if any(n == 3 and scenario not in qualified for scenario, n in data['consecutive_passes'].items()):
            raise ValueError('completed streak requires historical qualification')
        if any(not isinstance(v, str) or not v for key in ('last_qualified_checkpoint', 'qualified_protocol') for v in data[key].values()):
            raise ValueError('invalid checkpoint')
        for s in qualified:
            if not _ancestors(s) <= qualified:
                raise ValueError('qualified prerequisites missing')
            categories = data['qualified_throughput'][s]
            if not isinstance(categories, dict) or set(categories) != set(SCENARIOS[s].required_categories) or any(not _number(v) for v in categories.values()):
                raise ValueError('invalid throughput baselines')
        return cls(qualified, dict(data['consecutive_passes']), dict(data['last_qualified_checkpoint']),
                   {s: dict(c) for s, c in data['qualified_throughput'].items()}, set(data['seen_evaluation_ids']), dict(data['qualified_protocol']), dict(data['consecutive_protocol']), dict(data['consecutive_checkpoint']))


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    passed: bool
    failures: tuple[str, ...]


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validation_seeds(seeds, episodes):
    return (isinstance(seeds, tuple) and len(seeds) == episodes
            and all(type(seed) is int and 0 <= seed < 2**63
                    and seed % 4 == _SPLITS['validation'] for seed in seeds)
            and len(set(seeds)) == episodes)


def _ancestors(scenario_id):
    result = set()
    for parent in get_scenario(scenario_id).prerequisites:
        result.add(parent)
        result.update(_ancestors(parent))
    return result


def _failures(result):
    scenario = get_scenario(result.scenario_id)
    failures = []
    if result.split != 'validation':
        failures.append('promotion requires validation split')
    if result.scenario_version != scenario.version or result.curriculum_version != CURRICULUM_VERSION:
        failures.append('version mismatch')
    if any(not isinstance(v, str) or not v for v in (result.evaluation_id, result.checkpoint_id, result.protocol_id)):
        failures.append('missing evaluation/checkpoint/protocol identity')
    categories = {}
    for c in result.categories:
        if c.category in categories:
            failures.append(f'duplicate category {c.category}')
        categories[c.category] = c
    if set(categories) != set(scenario.required_categories):
        failures.append('required categories must match exactly')
    numeric_verdict = result.numeric_gate_verdict
    device_category_verdicts = None
    if numeric_verdict is not None:
        required_verdict_categories = (
            'success_rate_pass', 'throughput_ratio_pass',
            'minimum_useful_throughput_pass', 'useful_gain_ci_pass',
            'safety_pass')
        runtime = (numeric_verdict.get('runtime_device')
                   if isinstance(numeric_verdict, dict) else None)
        if (not isinstance(numeric_verdict, dict)
                or numeric_verdict.get('schema_version') != 1
                or numeric_verdict.get('verdict_type') != 'meco-gpu-numeric-gate-verdict-v1'
                or numeric_verdict.get('implementation_id') !=
                    'meco-v4-cluster-episode-device-percentile-v1'
                or numeric_verdict.get('evaluation_id') != result.evaluation_id
                or numeric_verdict.get('protocol_id') != result.protocol_id
                or not isinstance(runtime, dict)
                or runtime.get('backend') != 'hip'
                or not isinstance(runtime.get('device'), str)
                or not runtime['device'].startswith('cuda:')
                or not runtime['device'][5:].isdigit()
                or not isinstance(numeric_verdict.get('category_verdicts'), dict)
                or set(numeric_verdict['category_verdicts']) != set(scenario.required_categories)
                or not isinstance(numeric_verdict.get('failures'), list)
                or any(not isinstance(message, str) or not message
                       for message in numeric_verdict['failures'])):
            failures.append('GPU numeric gate verdict is missing or unbound')
        else:
            device_category_verdicts = numeric_verdict['category_verdicts']
            failures.extend(numeric_verdict['failures'])
    for name, c in categories.items():
        if type(c.episodes) is not int or type(c.successes) is not int or not 0 <= c.successes <= c.episodes or c.episodes < scenario.gate.min_episodes:
            failures.append(f'{name}: invalid or insufficient episode counts')
        elif device_category_verdicts is None and c.successes / c.episodes < scenario.gate.success_rate:
            failures.append(f'{name}: success gate failed')
        elif device_category_verdicts is not None:
            category_verdict = device_category_verdicts.get(name)
            if (not isinstance(category_verdict, dict) or
                    any(type(category_verdict.get(key)) is not bool
                        for key in required_verdict_categories) or
                    not isinstance(category_verdict.get('safety_metric_pass'), dict) or
                    set(category_verdict.get('safety_metric_pass', {})) != {
                        'safety_robot_contact_rate', 'safety_field_contact_rate',
                        'safety_wall_contact_rate'} or
                    any(type(value) is not bool for value in
                        category_verdict['safety_metric_pass'].values()) or
                    category_verdict.get('safety_pass') is not all(
                        category_verdict.get('safety_metric_pass', {}).values())):
                failures.append(f'{name}: GPU numeric category verdict is invalid')
            else:
                # The device computed the numeric predicates. Host orchestration
                # only consumes their typed Boolean outcomes; it never repeats
                # a threshold comparison.
                for key, message in (
                        ('success_rate_pass', 'success gate failed'),
                        ('throughput_ratio_pass', 'qualified paired throughput gate failed'),
                        ('minimum_useful_throughput_pass', 'absolute delivery minimum failed'),
                        ('useful_gain_ci_pass', 'paired useful-throughput CI does not exceed frozen margin'),
                        ('safety_pass', 'safety noninferiority CI failed')):
                    if not category_verdict[key]:
                        failures.append(f'{name}: {message}')
                if category_verdict.get('retention_baseline_pass') is False:
                    failures.append(f'{name}: retention throughput regression or missing baseline')
                for metric, passed in category_verdict['safety_metric_pass'].items():
                    if not passed:
                        failures.append(f'{name}: safety noninferiority failed for {metric}')
        if not _validation_seeds(c.seeds, c.episodes):
            failures.append(f'{name}: validation seed evidence missing or invalid')
        if not _number(c.useful_throughput) or not _number(c.reference_throughput):
            failures.append(f'{name}: invalid throughput')
            continue
        gate = scenario.gate
        if device_category_verdicts is None and gate.throughput_ratio and (
            c.paired is not True or c.reference_qualified is not True
            or c.reference_protocol_id != result.protocol_id
            or not _validation_seeds(c.reference_seeds, c.episodes)
            or c.seeds != c.reference_seeds or c.reference_throughput <= 0
            or c.useful_throughput < gate.throughput_ratio * c.reference_throughput
        ):
            failures.append(f'{name}: qualified paired throughput gate failed')
        if device_category_verdicts is None and c.useful_throughput < gate.min_useful_throughput:
            failures.append(f'{name}: absolute delivery minimum failed')
    return failures


def evaluate_promotion(state: CurriculumState, result: EvaluationResult, *, retention_results=(), additional_failures=()) -> PromotionDecision:
    """Consume one validation round, mutating only progression state.

    Retention results share the primary evaluation ID, checkpoint, and protocol.
    A consumed round cannot be replayed to manufacture consecutive passes.
    Failed retention blocks this promotion and resets its streak, preserving
    historical qualification and checkpoints for callers to schedule recovery.
    """
    scenario = get_scenario(result.scenario_id)
    failures = _failures(result) + [str(item) for item in additional_failures]
    if result.evaluation_id in state.seen_evaluation_ids:
        return PromotionDecision(False, False, ('duplicate evaluation ID',))
    if result.split != 'validation':
        return PromotionDecision(False, False, tuple(failures))
    ancestors = _ancestors(result.scenario_id)
    if not ancestors <= state.qualified:
        failures.append('unqualified prerequisites')
    retention = {}
    for retained in retention_results:
        if retained.scenario_id in retention:
            failures.append('duplicate retention scenario')
        retention[retained.scenario_id] = retained
    if set(retention) != ancestors:
        failures.append('retention must cover all transitive prerequisites exactly')
    for name, retained in retention.items():
        failures.extend(f'{name}: {f}' for f in _failures(retained))
        if (retained.evaluation_id, retained.checkpoint_id, retained.protocol_id) != (result.evaluation_id, result.checkpoint_id, result.protocol_id):
            failures.append(f'{name}: retention round/checkpoint/protocol mismatch')
        if retained.protocol_id != state.qualified_protocol.get(name):
            failures.append(f'{name}: baseline protocol mismatch')
        baseline = state.qualified_throughput.get(name, {})
        for c in retained.categories:
            if retained.numeric_gate_verdict is not None:
                category_verdict = retained.numeric_gate_verdict.get(
                    'category_verdicts', {}).get(c.category, {})
                if type(category_verdict.get('retention_baseline_pass')) is not bool:
                    failures.append(f'{name}/{c.category}: retention GPU verdict is invalid')
            elif c.category not in baseline or not _number(c.useful_throughput) or c.useful_throughput < .9 * baseline[c.category]:
                failures.append(f'{name}/{c.category}: retention throughput regression or missing baseline')
    if isinstance(result.evaluation_id, str) and result.evaluation_id:
        state.seen_evaluation_ids.add(result.evaluation_id)
    if failures:
        state.consecutive_passes[result.scenario_id] = 0
        state.consecutive_protocol.pop(result.scenario_id, None)
        state.consecutive_checkpoint.pop(result.scenario_id, None)
        return PromotionDecision(False, False, tuple(failures))
    previous = state.consecutive_passes.get(result.scenario_id, 0)
    if state.consecutive_protocol.get(result.scenario_id) != result.protocol_id:
        previous = 0
    streak = min(scenario.gate.consecutive_evaluations, previous + 1)
    state.consecutive_protocol[result.scenario_id] = result.protocol_id
    # Record the newest passing round's bundle for reporting and final-test
    # binding. Bundle identity is enforced within each round (target/retention),
    # while the protocol defines comparability across the promotion streak.
    state.consecutive_checkpoint[result.scenario_id] = result.checkpoint_id
    state.consecutive_passes[result.scenario_id] = streak
    promoted = result.scenario_id not in state.qualified and streak >= scenario.gate.consecutive_evaluations
    if promoted:
        state.qualified.add(result.scenario_id)
        state.last_qualified_checkpoint[result.scenario_id] = result.checkpoint_id
        state.qualified_protocol[result.scenario_id] = result.protocol_id
        state.qualified_throughput[result.scenario_id] = {c.category: c.useful_throughput for c in result.categories}
    return PromotionDecision(promoted, True, ())


def eligible_scenarios(state: CurriculumState) -> tuple[str, ...]:
    return tuple(s for s in SCENARIOS if set(SCENARIOS[s].prerequisites) <= state.qualified)


def task_mixture(state: CurriculumState, current_scenario_ids, next_scenario_ids=()) -> dict[str, float]:
    current, upcoming = set(current_scenario_ids), set(next_scenario_ids)
    if not current:
        raise ValueError('at least one current scenario is required')
    eligible = set(eligible_scenarios(state))
    if not current <= eligible or not upcoming <= eligible:
        raise ValueError('unknown scenario or unqualified prerequisites')
    if current & upcoming:
        raise ValueError('current and next tasks must be disjoint')
    earlier = state.qualified - current - upcoming
    groups = ((current, .7), (earlier, .2), (upcoming, .1))
    total = sum(weight for group, weight in groups if group)
    return {s: weight / total / len(group) for group, weight in groups if group for s in sorted(group)}
