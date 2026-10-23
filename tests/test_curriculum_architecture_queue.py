"""Supervisor contracts only; subprocess fixtures do not run a simulator."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_curriculum_architecture_queue.py'
spec = importlib.util.spec_from_file_location('curriculum_architecture_queue', SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_unique_lanes_fixed_devices_and_native_curriculum_command(tmp_path):
    lanes = queue.build_lanes(tmp_path, [7,19,43], ['cuda:0','cuda:1'])
    assert len(lanes)==len({l['directory'] for l in lanes})==21
    assert [l['architecture'] for l in lanes[:7]]==list(queue.ARCHITECTURES)
    config={'python':sys.executable,'envs':8,'rollout_steps':16,'epochs':4,
            'eval_every':25,'evaluation_episodes':100,'action_ticks':12,'minibatch_size':64,
            'learning_rate':.0003,'learning_rate_start':.003,'learning_rate_decay_updates':75,
            'sac_replay_capacity':4096,'sac_batch_size':256,'sac_warmup_steps':1024,
            'sac_updates_per_transition':.25,'sac_target_tau':.005,
            'sac_initial_alpha':.2,'sac_target_entropy_fraction':.98}
    command=queue.training_command(config,lanes[0],5)
    assert '--horizon' not in command and '--imitation-coef' not in command
    assert command[command.index('--evaluation-episodes')+1]=='100'
    assert command[command.index('--minibatch-size')+1]=='64'
    assert command[command.index('--learning-rate-start')+1]=='0.003'
    assert '--disable-miopen' in command and '--resume' not in command
    checkpoint=Path(lanes[0]['directory'])/'latest.pt'
    checkpoint.parent.mkdir(parents=True);checkpoint.touch()
    resumed=queue.training_command(config,lanes[0],10)
    assert resumed[resumed.index('--resume')+1]==str(checkpoint)
    paired=queue.build_lanes(tmp_path,[7],['cuda:0'],('ppo','sac'),('baseline_mlp',))
    assert [lane['algorithm'] for lane in paired]==['ppo','sac']
    sac_command=queue.training_command(config,paired[1],3)
    assert sac_command[sac_command.index('--algorithm')+1]=='sac'
    assert sac_command[sac_command.index('--sac-replay-capacity')+1]=='4096'


def test_fair_rounds_failure_visibility_and_restart_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'REPO',tmp_path)
    monkeypatch.setattr(queue,'source_hash',lambda:'frozen')
    def progress(lane):
        path=Path(lane['directory'])/'progress.json'
        return (json.loads(path.read_text())['update'],[],'pickup_simple') if path.exists() else (0,[],'pickup_simple')
    def command(config,lane,target):
        if lane['architecture']=='residual_mlp':
            return [sys.executable,'-c','raise SystemExit(17)']
        code='import json,sys;from pathlib import Path;p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True);(p/"progress.json").write_text(json.dumps({"update":int(sys.argv[2])}))'
        return [sys.executable,'-c',code,lane['directory'],str(target)]
    monkeypatch.setattr(queue,'committed_progress',progress)
    monkeypatch.setattr(queue,'training_command',command)
    config={'python':sys.executable,'seeds':[7],'devices':['fake:0'],'chunk_updates':1,'max_updates':2}
    output=tmp_path/'queue'
    assert queue.run_queue(output,config,expected_source='frozen')==1
    manifest=json.loads((output/'orchestration.json').read_text())
    assert len(manifest['attempts'])==13
    assert [a['architecture'] for a in manifest['attempts'][:7]]==list(queue.ARCHITECTURES)
    assert manifest['lanes'][1]['status']=='failed'
    assert all(l['completed_updates']==2 for l in manifest['lanes'] if l['status']!='failed')
    assert queue.run_queue(output,config,expected_source='frozen')==1
    restarted=json.loads((output/'orchestration.json').read_text())
    assert len(restarted['attempts'])==13  # completed work is not repeated
    monkeypatch.setattr(queue,'source_hash',lambda:'changed')
    with pytest.raises(ValueError,match='source'):
        queue.run_queue(output,config,expected_source='frozen')
