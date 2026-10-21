"""Runner progression/dataflow checks; actual PPO smoke accompanies each run."""
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frc_defense import curriculum_training as runner
from frc_defense.tensor_training_runtime import ActorCritic


def test_frozen_rows_excluded_and_timeout_bootstraps_pre_reset_observation(monkeypatch):
    torch.set_num_threads(1)
    policy=ActorCritic(137,8,action_kind='categorical')
    for parameter in policy.parameters(): parameter.data.zero_()
    policy.critic.bias.data.fill_(2.)
    env=SimpleNamespace(n=2,device=torch.device('cpu'),horizon=100,_last_obs=torch.zeros(2,6,137))
    env._last_obs[:,:,-8:]=1
    monkeypatch.setattr(runner,'teacher_actions',lambda env,obs:torch.zeros(2,6,dtype=torch.long))
    calls=[]
    def held(env,actions,alive,ticks):
        calls.append(alive.clone());index=len(calls)
        done=torch.tensor([index==1,False])
        truncated=torch.tensor([False,index==3])
        return env._last_obs,torch.tensor([1.,2.])*alive,done,truncated,alive&~(done|truncated)
    monkeypatch.setattr(runner,'_held_step',held)
    batch=runner._collect(policy,env,runner.TrainingConfig(num_envs=2,rollout_steps=3,horizon=100,action_ticks=1))
    assert len(batch['action']) == 4
    assert calls[1].tolist() == [False,True]
    assert batch['returns'][0].item() == pytest.approx(1.)  # true termination
    assert batch['returns'][-1].item() == pytest.approx(2.+.993*2.)  # timeout critic=2


def test_normal_budget_collects_through_horizon_not_only_short_rollout(monkeypatch):
    policy=ActorCritic(137,8,action_kind='categorical')
    env=SimpleNamespace(n=1,device=torch.device('cpu'),horizon=10,_last_obs=torch.zeros(1,6,137))
    env._last_obs[:,:,-8:]=1
    monkeypatch.setattr(runner,'teacher_actions',lambda env,obs:torch.zeros(1,6,dtype=torch.long))
    calls=[]
    def held(env,actions,alive,ticks):
        calls.append(1);timeout=torch.tensor([len(calls)==5])
        return env._last_obs,torch.zeros(1),torch.tensor([False]),timeout,~timeout
    monkeypatch.setattr(runner,'_held_step',held)
    batch=runner._collect(policy,env,runner.TrainingConfig(rollout_steps=1,action_ticks=2))
    assert len(calls)==len(batch['action'])==5


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    path=tmp_path_factory.mktemp('curriculum-runner')
    config=runner.TrainingConfig(updates=1,num_envs=1,rollout_steps=2,epochs=1,seed=31,
        horizon=6,action_ticks=1,eval_every=0,device='cpu',output=str(path/'training'))
    summary=runner.train_curriculum(config)
    return config,summary,path


def test_actual_update_resume_rng_and_configuration(trained):
    config,summary,path=trained
    assert summary['stats'][0]['policy_changed']
    assert summary['diagnostic']
    _,_,checkpoint=runner.load_checkpoint(summary['checkpoint'],device='cpu',restore_rng=True)
    expected=torch.rand(4)
    runner.load_checkpoint(summary['checkpoint'],device='cpu',restore_rng=True)
    assert torch.equal(expected,torch.rand(4))
    with pytest.raises(ValueError,match='immutable'):
        runner.load_checkpoint(summary['checkpoint'],config=replace(config,action_ticks=2),device='cpu')
    continued=runner.train_curriculum(replace(config,updates=2,output=str(path/'resumed')),resume=summary['checkpoint'])
    assert continued['updates']==2
    assert len(continued['stats'])==2
    assert all(torch.isfinite(torch.tensor(row['loss'])) for row in continued['stats'])
    assert continued['stats'][1]['seed'] != continued['stats'][0]['seed']
    _,_,restored=runner.load_checkpoint(continued['checkpoint'])
    assert restored['update']==restored['train_seed_counter']==2
    assert checkpoint['curriculum_state'].to_dict()==restored['curriculum_state'].to_dict()


def test_resume_matches_uninterrupted_next_update(trained):
    config,summary,path=trained
    resumed=runner.train_curriculum(replace(config,updates=2,output=str(path/'repeated-resume')),resume=summary['checkpoint'])
    straight=runner.train_curriculum(replace(config,updates=2,output=str(path/'straight')))
    assert resumed['policy_id']==straight['policy_id']
    assert resumed['stats']==straight['stats']


def test_incompatible_source_and_malformed_state_rejected(trained):
    _,summary,_=trained
    payload=torch.load(summary['checkpoint'],weights_only=False)
    payload['sources']={}
    stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
    with pytest.raises(ValueError,match='source'):
        runner.load_checkpoint(stream)
    payload=torch.load(summary['checkpoint'],weights_only=False);payload['current']='impossible'
    stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
    with pytest.raises(ValueError,match='current'):
        runner.load_checkpoint(stream)


def test_small_paired_evaluation_is_diagnostic_and_final_test_requires_freeze(trained,monkeypatch):
    config,summary,path=trained
    with pytest.raises(ValueError,match='explicit final_test'):
        runner.evaluate_curriculum(summary['checkpoint'],('pickup_simple',),episodes=1,split='test',horizon=2,output=path/'test.json')
    report=runner.evaluate_curriculum(summary['checkpoint'],('pickup_simple',),episodes=1,horizon=2,
        action_ticks=1,categories=('single',),output=path/'validation.json')
    assert report['diagnostic']
    category=report['evaluation_results'][0]['categories'][0]
    assert not category['reference_qualified']
    assert category['seeds']==category['reference_seeds']
    pair=report['records'][0]
    assert pair['learner'][0]['seed']==pair['reference'][0]['seed']
    assert pair['learner'][0]['frames'][-1]['phase']=='terminal'
    assert 'executed_action' in pair['learner'][0]['frames'][0]
    assert 'next_pose' in pair['learner'][0]['frames'][0]
    assert pair['learner'][0]['configuration']['scenario_id']=='pickup_simple'
    # A new output directory is created before freezing bytes on that volume.
    monkeypatch.setattr(runner,'_episode',lambda *args,**kwargs:{'outcome':{'success':False,'useful_throughput':0.},'frames':[]})
    frozen=runner.evaluate_curriculum(summary['checkpoint'],('pickup_simple',),episodes=1,split='test',final_test=True,
        horizon=1,action_ticks=1,categories=('single',),output=path/'missing'/'final.json')
    assert list((path/'missing').glob('frozen-checkpoint-*.pt'))
    assert frozen['checkpoint_file_sha256']


def test_teacher_export_and_atomic_json(trained):
    _,_,path=trained
    exported=runner.export_teacher('pickup_simple','single',episodes=1,horizon=2,action_ticks=1,output=path/'teacher.jsonl')
    rows=[json.loads(line) for line in Path(exported['output']).read_text().splitlines()]
    assert rows and len(rows[0]['observation'])==137 and len(rows[0]['mask'])==8
    assert rows[0]['seed']%4==0
    assert not rows[0]['reference_qualified']
    assert 'outcome' in rows[0] and rows[0]['teacher'].startswith('generic observed')
    destination=path/'atomic.json';runner._atomic_json(destination,{'old':True})
    with pytest.raises(ValueError): runner._atomic_json(destination,{'bad':float('nan')})
    assert json.loads(destination.read_text())=={'old':True}


@pytest.mark.parametrize('reference_successes,throughput,qualified',[(89,1.,False),(90,1.,True),(100,0.,False)])
def test_reference_qualification_requires_measured_success_and_positive_useful_rate(monkeypatch,tmp_path,reference_successes,throughput,qualified):
    teacher_calls=[]
    def episode(policy,*args,**kwargs):
        if policy=='teacher':
            teacher_calls.append(1)
            success=len(teacher_calls)<=reference_successes
            return {'outcome':{'success':success,'useful_throughput':throughput},'frames':[]}
        return {'outcome':{'success':False,'useful_throughput':.5},'frames':[]}
    monkeypatch.setattr(runner,'_episode',episode)
    policy=ActorCritic(137,8,action_kind='categorical')
    report=runner.evaluate_curriculum(policy,('active_collect_carry_score',),episodes=100,
        categories=('active/open',),output=tmp_path/'reference.json')
    category=report['evaluation_results'][0]['categories'][0]
    assert category['reference_qualified'] is qualified
    assert report['diagnostic']  # selected category alone never qualifies a stage


def test_new_run_never_overwrites_existing_checkpoint(trained):
    config,summary,_=trained
    before=Path(summary['checkpoint']).read_bytes()
    with pytest.raises(ValueError,match='already contains'):
        runner.train_curriculum(config)
    assert Path(summary['checkpoint']).read_bytes()==before
