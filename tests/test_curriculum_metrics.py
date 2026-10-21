"""Lifecycle accounting tests, paired with retained simulator smoke records."""
from types import SimpleNamespace
import torch

from frc_defense.curriculum_metrics import CurriculumMetrics


def fixture():
    env = SimpleNamespace(n=1, device=torch.device('cpu'), horizon=100, dt=.1,
        field_length=16.54, alliance_zone_depth=4.03, fuel_physics=None,
        piece_owner=torch.full((1, 8), -1), piece_active=torch.ones((1, 8), dtype=torch.bool),
        piece_pos=torch.zeros(1, 8, 2), piece_vel=torch.zeros(1, 8, 2),
        hub_active=torch.tensor([[True, False]]), match_elapsed=torch.zeros(1),
        hub_centers=torch.tensor([[4.6,4.], [11.9,4.]]),
        sim=SimpleNamespace(pose=torch.zeros(1, 6, 3), field_width=8.07),
        curriculum={'scenario_id':'inactive_collect_ferry', 'category':'inactive/open', 'route':'open',
                    'time_budget_s':90., 'schedule':((0.,False),),
                    'targets':{'amount':2,'batch_size':2,'cycles':2}, 'pressure':{'slots':(), 'level':0}})
    env._robots_clear_of_bumps = lambda pose: torch.ones(1,6,dtype=torch.bool)
    env.piece_pos[...,0] = 9.
    return env


def selected(*indices):
    result = torch.zeros(1, 8, dtype=torch.bool)
    result[:,list(indices)] = True
    return result


def settle(env, metrics, ticks=6):
    for _ in range(ticks):
        env.match_elapsed += env.dt
        metrics.update(torch.tensor([True]))


def test_acquisition_credit_once_and_score_starts_new_lifecycle():
    env = fixture(); metrics = CurriculumMetrics(env)
    own = selected(0); owners = torch.full_like(env.piece_owner, 3)
    metrics.acquired(own, owners, env.piece_pos)
    metrics.acquired(own, owners, env.piece_pos)
    assert metrics.acquisitions.item() == 1
    metrics.scored(own, owners)
    assert metrics.lifecycle[0,0].item() == 1
    metrics.acquired(own, owners, env.piece_pos)
    assert metrics.acquisitions.item() == 2
    assert metrics.scores.item() == 1


def test_release_needs_settling_home_stock_recollection_never_double_credits():
    env = fixture(); env.piece_owner[:,0] = 3
    metrics = CurriculumMetrics(env); own = selected(0)
    owners = env.piece_owner.clone()
    metrics.released(own, owners)
    assert metrics.deliveries.item() == 0
    env.piece_owner[:,0] = -1
    env.piece_pos[:,0] = torch.tensor([14.,4.])
    env.piece_vel[:,0,0] = 2.
    settle(env, metrics)
    assert metrics.deliveries.item() == 0
    env.piece_vel.zero_(); settle(env, metrics)
    assert metrics.deliveries.item() == metrics.current_staged.item() == 1
    env.piece_owner[:,0] = 3
    metrics.acquired(own, env.piece_owner, env.piece_pos); settle(env, metrics, 1)
    assert metrics.current_staged.item() == 0
    metrics.released(own, env.piece_owner)
    env.piece_owner[:,0] = -1; settle(env, metrics)
    assert metrics.deliveries.item() == 1
    # A true score clears the old lifecycle; another external acquisition is
    # now eligible for one further physical delivery.
    metrics.scored(own, owners)
    env.piece_pos[:,0,0] = 9.
    metrics.acquired(own, owners, env.piece_pos)
    metrics.released(own, owners)
    env.piece_pos[:,0,0] = 14.; settle(env, metrics)
    assert metrics.deliveries.item() == 2


def test_home_origin_ferry_cannot_earn_delivery():
    env = fixture(); env.piece_pos[:,0,0] = 14.
    metrics = CurriculumMetrics(env); own = selected(0)
    owners = torch.full_like(env.piece_owner,3)
    metrics.acquired(own,owners,env.piece_pos)
    metrics.released(own,owners); settle(env,metrics)
    assert metrics.deliveries.item() == 0


def test_batch_requires_empty_hopper_and_two_pieces_not_each_action():
    env = fixture(); env.piece_owner[:,:2] = 3
    metrics = CurriculumMetrics(env)
    for index in (0,1):
        own = selected(index)
        metrics.released(own, env.piece_owner.clone())
        env.piece_owner[:,index] = -1
        env.piece_pos[:,index] = torch.tensor([14.,4.])
        settle(env, metrics)
    assert metrics.batch_completed.sum().item() == 1
    assert metrics.outcomes()['completed_cycles'].item() == 1
    assert not metrics.outcomes()['success'].item()


def test_teammate_score_preserves_useful_delivery_and_consumes_reward_once():
    env = fixture(); metrics = CurriculumMetrics(env)
    metrics.ferry_credit[:,0] = True
    metrics.inactive_origin[:,0] = True
    metrics.scored(selected(0), torch.full_like(env.piece_owner,4))
    assert metrics.team_conversions.item() == 1
    assert metrics.scores.item() == 0
    assert metrics.reward_delta().item() == 2
    assert metrics.reward_delta().item() == 0


def test_full_match_cannot_pass_before_complete_schedule():
    env = fixture(); env.curriculum.update(scenario_id='full_match_offense', category='active/open/level_0')
    metrics = CurriculumMetrics(env)
    metrics.scores[:] = 10; metrics.batch_completed[:,:2] = True
    assert not metrics.outcomes()['success'].item()


def test_active_cycle_cannot_pass_on_ferry_stock_or_ferry_batches():
    env = fixture(); env.curriculum.update(scenario_id='active_collect_carry_score', category='active/open')
    env.hub_active[:] = torch.tensor([False,True])
    metrics = CurriculumMetrics(env)
    metrics.current_staged[:] = 8
    metrics.inactive_staged[:] = 8
    metrics.ferry_batch_counts[:,:2] = 4
    metrics.batch_ended[:,:2] = True
    metrics.batch_completed[:,:2] = True
    outcome = metrics.outcomes()
    assert not outcome['success'].item()
    assert outcome['completed_cycles'].item() == 0
    assert outcome['useful_throughput'].item() == 0


def test_inactive_full_match_cannot_use_active_origin_stock():
    env = fixture(); env.curriculum.update(scenario_id='full_match_offense', category='inactive/open/level_0')
    env.match_elapsed[:] = 160
    env.curriculum['time_budget_s'] = 160
    metrics = CurriculumMetrics(env)
    metrics.elapsed[:] = 160
    metrics.inactive_seconds[:] = 25
    metrics.current_staged[:] = 8  # all released during active periods
    metrics.ferry_batch_counts[:,:2] = 4
    metrics.batch_ended[:,:2] = True
    assert not metrics.outcomes()['success'].item()
    assert metrics.outcomes()['useful_throughput'].item() == 0


def test_pickup_and_transport_report_physical_useful_throughput():
    env = fixture(); env.curriculum.update(scenario_id='pickup_simple', category='single')
    env.piece_owner[:,0] = 3
    metrics = CurriculumMetrics(env); metrics.elapsed[:] = 2
    assert metrics.outcomes()['useful_throughput'].item() == .5
    env.curriculum.update(scenario_id='active_carry_to_zone', category='active/trench', route='trench')
    env.curriculum['targets']['amount'] = 1
    env.hub_active[:] = torch.tensor([False,True])
    assert metrics.outcomes()['useful_throughput'].item() == 0
    metrics.route_seen[:] = True; env.sim.pose[:,3,0] = 14
    assert metrics.outcomes()['useful_throughput'].item() == .5
