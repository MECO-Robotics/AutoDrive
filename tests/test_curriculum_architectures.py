"""Curriculum model compatibility, accompanied by paired simulator episodes."""
from dataclasses import replace

import pytest
import torch

from frc_defense import curriculum_training as runner
from frc_defense.curriculum_policy import ARCHITECTURES


@pytest.mark.parametrize('architecture', tuple(ARCHITECTURES))
def test_architecture_update_checkpoint_resume_and_paired_simulation(architecture, tmp_path):
    config=runner.TrainingConfig(architecture=architecture,updates=1,num_envs=1,
        rollout_steps=2,epochs=1,seed=31,horizon=6,action_ticks=1,eval_every=0,
        minibatch_size=2,output=str(tmp_path/'training'))
    trained=runner.train_curriculum(config)
    assert trained['stats'][0]['policy_changed']
    policy,_,saved=runner.load_checkpoint(trained['checkpoint'])
    assert saved['config']['architecture']==architecture
    assert torch.isfinite(torch.tensor(trained['stats'][0]['loss']))
    with pytest.raises(ValueError,match='immutable'):
        runner.load_checkpoint(trained['checkpoint'],config=replace(config,
            architecture='full_field_cnn' if architecture!='full_field_cnn' else 'baseline_mlp'))
    resumed=runner.train_curriculum(replace(config,updates=2),resume=trained['checkpoint'])
    straight=runner.train_curriculum(replace(config,updates=2,output=str(tmp_path/'straight')))
    assert resumed['policy_id']==straight['policy_id']
    report=runner.evaluate_curriculum(resumed['checkpoint'],('pickup_simple',),
        episodes=1,horizon=6,action_ticks=1,categories=('single',),
        master_seed=31,output=tmp_path/'paired-simulation.json')
    assert report['architecture']==architecture
    assert report['diagnostic']
    pair=report['records'][0]
    assert pair['learner'][0]['seed']==pair['reference'][0]['seed']
    assert pair['learner'][0]['physics']
    assert 1<=pair['learner'][0]['terminal']['tick']<=6
    if pair['learner'][0]['terminal']['tick']<6:
        assert pair['learner'][0]['outcome']['success']
    assert pair['learner'][0]['configuration']['scenario_id']=='pickup_simple'
    assert pair['learner'][0]['frames'][-1]['phase']=='terminal'


def test_optimization_checkpoint_survives_validation_failure(monkeypatch,tmp_path):
    config=runner.TrainingConfig(updates=1,num_envs=1,rollout_steps=2,epochs=1,
        horizon=4,action_ticks=1,eval_every=1,output=str(tmp_path/'training'))
    def failed(*args,**kwargs):
        raise RuntimeError('interrupted validation')
    monkeypatch.setattr(runner,'evaluate_curriculum',failed)
    with pytest.raises(RuntimeError,match='interrupted validation'):
        runner.train_curriculum(config)
    _,_,payload=runner.load_checkpoint(tmp_path/'training/latest.pt')
    assert payload['update']==1
    assert payload['stats'][0]['evaluation_pending']
    assert payload['stats'][0]['policy_changed']
