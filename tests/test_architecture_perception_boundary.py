"""Prephysics controller privacy regressions; E2E qualification is also required."""
from types import SimpleNamespace
import pytest
import torch
from frc_defense import architecture_scenarios, tensor_3v3
from frc_defense.architecture_inputs import PerceptionInputs
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv

class BeforePhysics(Exception):
    pass

@pytest.fixture
def env(monkeypatch):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    def factory(**kwargs):
        kwargs['fuel_physics']=False
        return TensorThreeVsThreeEnv(**kwargs)
    monkeypatch.setattr(architecture_scenarios,'TensorThreeVsThreeEnv',factory)
    instance=architecture_scenarios.make_env('deterministic_defense',1,'cpu',731)
    yield instance
    torch.set_num_threads(previous)

def controller_snapshot(env,monkeypatch,action,hidden_xy,observed_xy=None,*,possession=0):
    architecture_scenarios.reset_scenario(env,'deterministic_defense',731)
    if possession:
        env.piece_owner[:,30:30+possession]=3
        env.piece_active[:,30:30+possession]=True
    env.sim.pose[0,3,:2]=torch.tensor([7.4,4.035])
    env.sim.pose[0,3,2]=0.
    env.sim.pose[0,0,:2]=torch.tensor(hidden_xy)
    env.sim.velocity[0,0,:2]=torch.tensor([.2,.1] if hidden_xy[0]<3. else [1.7,-.6])
    env.sim.speed[0,0]=4. if hidden_xy[0]<3. else 12.
    env.opponent_valid[:,3]=observed_xy is not None
    env.opponent_pose[:,3]=0.
    env.opponent_velocity[:,3]=0.
    env.opponent_size[:,3]=0.
    if observed_xy is not None:
        env.opponent_pose[0,3,:2]=torch.tensor(observed_xy)
        env.opponent_velocity[0,3,:2]=torch.tensor([-.6,0.])
        env.opponent_size[0,3,:2]=torch.tensor([.9,.8])
        env.opponent_age[0,3]=0.
    captured={}
    def before_physics(instance,commands,active_mask,**kwargs):
        captured['executed_action']=instance.last_actions[:,3].clone()
        captured['command']=commands[:,3].clone()
        captured['target']=instance._last_audit_targets[:,3].clone()
        captured['route']=instance.planner.last_path[3].clone()
        captured['route_length']=instance.planner.last_lengths[3].clone()
        captured['inputs']={key:value.clone() for key,value in PerceptionInputs(instance,[3])(instance.observe()).items()}
        raise BeforePhysics
    monkeypatch.setattr(tensor_3v3,'advance_with_fuel',before_physics)
    actions=torch.full((1,6),7,dtype=torch.long);actions[:,3]=action
    with pytest.raises(BeforePhysics):env.step(actions)
    return captured

@pytest.mark.parametrize('action',range(8))
def test_full_controller_hidden_enemy_invariance(env,monkeypatch,action):
    first=controller_snapshot(env,monkeypatch,action,(2.,1.))
    second=controller_snapshot(env,monkeypatch,action,(8.1,4.035))
    for key in ('command','target','route','route_length'):
        torch.testing.assert_close(first[key],second[key],rtol=0.,atol=0.)
    for key in first['inputs']:
        torch.testing.assert_close(first['inputs'][key],second['inputs'][key],rtol=0.,atol=0.)

def test_actual_controller_responds_to_observed_anonymous_opponent(env,monkeypatch):
    first=controller_snapshot(env,monkeypatch,5,(2.,1.),(10.,4.035))
    second=controller_snapshot(env,monkeypatch,5,(2.,1.),(8.1,4.035))
    assert not torch.equal(first['target'],second['target'])
    assert not torch.equal(first['command'],second['command'])

def test_contention_ignores_hidden_enemy_intent_and_obeys_valid_track(env):
    env.sim.pose[0,3,:2]=torch.tensor([7.4,4.035])
    env.opponent_valid.zero_()
    command=torch.zeros(1,6,2);command[0,3]=torch.tensor([1.,0.])
    targets=env.sim.pose[:,:,:2].clone()+command
    env.sim.pose[0,0,:2]=torch.tensor([2.,1.]);env._avoidance_winner.fill_(-1)
    first=env._avoid_robot_contention(command.clone(),targets)[:,3].clone()
    env.sim.pose[0,0,:2]=torch.tensor([8.1,4.035]);command[0,0]=torch.tensor([-3.,1.])
    env._avoidance_winner.fill_(-1)
    second=env._avoid_robot_contention(command.clone(),targets)[:,3].clone()
    torch.testing.assert_close(first,second,rtol=0.,atol=0.)
    env.opponent_valid[:,3]=True;env.opponent_pose[0,3,:2]=torch.tensor([8.1,4.035])
    env.opponent_velocity[0,3,:2]=torch.tensor([-1.,0.]);env.opponent_size[0,3,:2]=.8
    tracked=env._avoid_robot_contention(command.clone(),targets)[:,3]
    assert not torch.equal(tracked,second)
    env.opponent_pose[:,3,:2]=float('nan');env._avoidance_winner.fill_(-1)
    invalid=env._avoid_robot_contention(command.clone(),targets)[:,3]
    torch.testing.assert_close(invalid,second,rtol=0.,atol=0.)


def test_experimental_nn_offense_receives_visible_opponent_sensor_track(monkeypatch):
    from frc_defense import architecture_offense_scenario
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        def factory(**kwargs):
            kwargs['fuel_physics']=False
            return TensorThreeVsThreeEnv(**kwargs)
        monkeypatch.setattr(architecture_offense_scenario,'TensorThreeVsThreeEnv',factory)
        instance=architecture_offense_scenario.make_offense_env(1,'cpu',731)
        instance.perception_dropout=0.;instance.position_noise=0.;instance.velocity_noise=0.
        instance.sim.pose[0,3]=torch.tensor([7.4,4.035,0.])
        instance.sim.pose[0,0]=torch.tensor([9.,4.035,0.])
        instance.opponent_valid.zero_();instance.opponent_age.fill_(float('inf'))
        instance._update_perception(torch.ones(1,dtype=torch.bool),active_count=1)
        assert instance.opponent_valid[0,3]
        torch.testing.assert_close(instance.opponent_pose[0,3,:2],instance.sim.pose[0,0,:2],rtol=0.,atol=0.)
        assert (instance.opponent_size[0,3,:2]>0.).all()
    finally:
        torch.set_num_threads(previous)


def test_full_step_hidden_enemy_cannot_retire_retained_track(env,monkeypatch):
    # Freeze observer evidence between paired runs: a remembered detection need
    # not coincide with current FUEL truth. Hidden contact with this memory must
    # not become an extra sensor. Real physics advances, with no robot contact.
    monkeypatch.setattr(env,'_update_perception',lambda *args,**kwargs:None)
    records=[]
    for hidden_xy in ((2.,1.),(3.,1.)):
        architecture_scenarios.reset_scenario(env,'deterministic_defense',731)
        env.piece_active.zero_();env.piece_owner.fill_(-1)
        env.sim.pose[0,3]=torch.tensor([7.4,4.035,0.])
        env.sim.pose[0,0]=torch.tensor([*hidden_xy,0.])
        env.opponent_valid.zero_();env.track_mask.zero_()
        env.track_pos[0,3,0]=torch.tensor([3.,1.])
        env.track_vel.zero_();env.track_mask[0,3,0]=True
        env.track_age[0,3,0]=0.;env.track_confidence[0,3,0]=1.
        actions=torch.full((1,6),7,dtype=torch.long)
        for _ in range(4):env.step(actions)
        assert env.track_mask[0,3,0]
        records.append({key:value.clone() for key,value in PerceptionInputs(env,[3])(env.observe()).items()})
    for key in records[0]:torch.testing.assert_close(records[0][key],records[1][key],rtol=0.,atol=0.)


@pytest.mark.parametrize('action',[4,6])
def test_loaded_score_and_ferry_commands_ignore_hidden_enemy(env,monkeypatch,action):
    # Positive possession makes scoring legal and exercises the loaded ferry
    # branch; an empty-inventory score request is replaced by another tactic.
    first=controller_snapshot(env,monkeypatch,action,(2.,1.),possession=6)
    second=controller_snapshot(env,monkeypatch,action,(8.1,4.035),possession=6)
    assert first['executed_action'].item()==action
    assert second['executed_action'].item()==action
    for key in ('command','target','route','route_length'):
        torch.testing.assert_close(first[key],second[key],rtol=0.,atol=0.)
    for key in first['inputs']:
        torch.testing.assert_close(first['inputs'][key],second['inputs'][key],rtol=0.,atol=0.)
