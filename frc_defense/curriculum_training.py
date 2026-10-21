"""Opt-in offense PPO, paired evaluations, and observed-teacher export.

The standard 137-observation/8-action ActorCritic uses generic NN execution.
Draft scenario targets and scripted teachers remain unqualified until measured.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
import uuid

import torch
from torch.nn import functional as F

from .curriculum_scenarios import make_curriculum_env, reset_curriculum_scenario, teacher_actions
from .tensor_training_runtime import ActorCritic, _gae_advantages
from .training_curriculum import (SCENARIOS, CURRICULUM_VERSION, CategoryResult, EvaluationResult,
    CurriculumState, eligible_scenarios, evaluate_promotion, seed_partition, task_mixture)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TrainingConfig:
    updates: int = 1
    num_envs: int = 4
    rollout_steps: int = 16
    epochs: int = 4
    seed: int = 7
    device: str = 'cpu'
    output: str = str(ROOT / 'checkpoints/curriculum-offense')
    horizon: int | None = None
    action_ticks: int = 12
    evaluation_episodes: int = 100
    eval_every: int = 10
    imitation_coef: float = 0.
    learning_rate: float = 3e-4
    gamma: float = .993
    gae_lambda: float = .95
    clip_coef: float = .2
    value_coef: float = .5
    entropy_coef: float = .01
    max_grad_norm: float = .5

    def validate(self):
        for name in ('updates', 'num_envs', 'rollout_steps', 'epochs', 'action_ticks', 'evaluation_episodes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int or self.seed < 0 or type(self.eval_every) is not int or self.eval_every < 0:
            raise ValueError('seed and eval_every must be nonnegative integers')
        if self.horizon is not None and (type(self.horizon) is not int or self.horizon < 1):
            raise ValueError('horizon must be positive or None')
        for name in ('imitation_coef', 'learning_rate', 'gamma', 'gae_lambda', 'clip_coef', 'value_coef', 'entropy_coef', 'max_grad_norm'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.learning_rate == 0 or self.max_grad_norm == 0 or self.gamma > 1 or self.gae_lambda > 1:
            raise ValueError('invalid learning rate, gradient norm, or discount')


def source_identity():
    files = sorted(p for p in (ROOT/'frc_defense').rglob('*')
                   if p.suffix in ('.py','.cpp','.cu','.h','.cuh','.json','.hip','.hpp','.cc'))
    identity = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    identity['_runtime_environment'] = {k:v for k,v in sorted(os.environ.items())
        if k.startswith(('AUTODRIVE_', 'FRC_', 'PYTORCH_'))}
    config_path=os.environ.get('FRC_DRIVETRAIN_CONFIG')
    if config_path: identity['_drivetrain_config_sha256']=hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    return identity


def _hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def protocol_id(*, action_ticks=12, device='cpu', _sources=None):
    # One shared suite for every scenario and retention result. Per-episode
    # seeds/counts/checkpoints are intentionally recorded outside this identity.
    return _hash({'sources': source_identity() if _sources is None else _sources, 'registry': {k: asdict(v) for k,v in SCENARIOS.items()},
                  'dt': .02, 'fuel_physics': True, 'action_ticks': action_ticks,
                  'device': str(device), 'torch_threads':1,'torch': torch.__version__, 'model': [137,8,'categorical'],
                  'reward': 'lifecycle-acquisition-.02/delivery-1/score-2/completion-5'})


def _atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name); stream.write(content)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        torch.save(payload, temp); os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _policy_id(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'python': random.getstate()}


def _restore_rng(state):
    torch.set_rng_state(state['torch'].cpu()); random.setstate(state['python'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('checkpoint CUDA RNG devices are incompatible')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def load_checkpoint(path, *, config=None, device='cpu', restore_rng=False):
    """Return (standard policy, optimizer, payload); reject incompatible resumes."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    expected={'checkpoint_version','sources','config','policy','optimizer','policy_id','rng','scheduler_rng','update','curriculum_state','current','recovery','train_seed_counter','stats','qualified_artifacts'}
    if not isinstance(payload,dict) or set(payload)!=expected: raise ValueError('invalid checkpoint schema')
    if payload.get('checkpoint_version') != CHECKPOINT_VERSION or payload.get('sources') != source_identity():
        raise ValueError('checkpoint format/source identity mismatch')
    saved = TrainingConfig(**payload['config']); saved.validate()
    if config is not None:
        config.validate()
        mutable = {'updates', 'output'}
        if any(v != asdict(config)[k] for k,v in asdict(saved).items() if k not in mutable):
            raise ValueError('immutable training configuration mismatch')
        if config.updates < payload['update']:
            raise ValueError('target updates precede the saved update')
    payload['curriculum_state'] = CurriculumState.from_dict(payload['curriculum_state'])
    if type(payload['update']) is not int or payload['update']<1 or type(payload['train_seed_counter']) is not int or not 0<=payload['train_seed_counter']<2**21:
        raise ValueError('invalid checkpoint counters')
    if payload['current'] not in eligible_scenarios(payload['curriculum_state']): raise ValueError('invalid current task')
    recovery=payload['recovery']
    if not isinstance(recovery,list) or any(not isinstance(s,str) for s in recovery) or len(recovery)!=len(set(recovery)) or not set(recovery)<=payload['curriculum_state'].qualified:
        raise ValueError('invalid recovery state')
    if not isinstance(payload['stats'],list) or len(payload['stats'])!=payload['update']: raise ValueError('invalid update history')
    json.dumps(payload['stats'],allow_nan=False)
    artifacts=payload['qualified_artifacts']
    if not isinstance(artifacts,dict) or set(artifacts)!=payload['curriculum_state'].qualified or any(not isinstance(p,str) or not Path(p).is_file() for p in artifacts.values()):
        raise ValueError('missing historical qualified checkpoint artifacts')
    policy = ActorCritic(137,8,action_kind='categorical').to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=saved.learning_rate)
    policy.load_state_dict(payload['policy']); optimizer.load_state_dict(payload['optimizer'])
    if _policy_id(policy) != payload['policy_id']:
        raise ValueError('policy identity mismatch')
    if restore_rng:
        _restore_rng(payload['rng'])  # after model initialization consumes RNG
    return policy, optimizer, payload


def _ancestors(name):
    parents = set(SCENARIOS[name].prerequisites)
    return parents | set().union(*(_ancestors(p) for p in parents)) if parents else set()


def _held_step(env, actions, alive, ticks):
    """Keep each world's final pre-reset observation for timeout bootstraps."""
    reward = torch.zeros(env.n, device=env.device)
    done = torch.zeros_like(alive); truncated = torch.zeros_like(alive)
    for _ in range(ticks):
        if not bool(alive.any()): break
        obs, _, ended, timeout, _ = env.step(actions, active_mask=alive, capture_info=False)
        reward += env.curriculum_metrics.reward_delta()
        done |= ended & alive; truncated |= timeout & alive
        alive = alive & ~(ended | timeout)
    return env._last_obs, reward, done, truncated, alive


def _collect(policy, env, config):
    rows = []; alive = torch.ones(env.n, device=env.device, dtype=torch.bool)
    steps=config.rollout_steps if config.horizon is not None else max(config.rollout_steps,math.ceil(env.horizon/config.action_ticks))
    for _ in range(steps):
        valid = alive.clone(); obs = env._last_obs[:,3].clone(); mask = obs[:,-8:].bool()
        with torch.no_grad():
            action, logprob, value = policy.sample(obs, action_mask=mask)
            teacher = teacher_actions(env,env._last_obs)[:,3].clone()
        actions = torch.full((env.n,6),7,device=env.device,dtype=torch.long); actions[:,3] = action
        next_obs,reward,done,truncated,alive = _held_step(env,actions,alive,config.action_ticks)
        with torch.no_grad(): next_value = policy(next_obs[:,3])[2]
        rows.append({'obs':obs,'mask':mask,'action':action,'logprob':logprob,'value':value,
                     'next_value':next_value,'reward':reward,'done':done | ~valid,'truncated':truncated,
                     'valid':valid,'teacher':teacher})
        if not bool(alive.any()): break
    # The environment is deliberately discarded at this update boundary.
    # Remaining worlds are timeouts, not true physical terminations.
    rows[-1]['truncated'] |= alive
    batch = {key:torch.stack([r[key] for r in rows]) for key in rows[0]}
    advantages = _gae_advantages(batch['reward'],batch['done'],batch['truncated'],batch['value'],
                                 batch['next_value'],config.gamma,config.gae_lambda)
    batch['returns'] = advantages + batch['value']; batch['advantages'] = advantages
    return {k:v.reshape((-1,)+v.shape[2:])[batch['valid'].reshape(-1)] for k,v in batch.items()}


def _ppo_update(policy, optimizer, batch, config, update):
    advantage = batch['advantages']; advantage = (advantage-advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    imitation_weight = config.imitation_coef / (1 + update/100)
    for _ in range(config.epochs):
        logprob,entropy,value,logits = policy.score(batch['obs'],batch['action'],batch['mask'])
        ratio = (logprob-batch['logprob']).exp()
        actor = -torch.minimum(ratio*advantage, ratio.clamp(1-config.clip_coef,1+config.clip_coef)*advantage).mean()
        critic = .5*F.mse_loss(value,batch['returns'])
        imitation = F.cross_entropy(logits,batch['teacher'])
        loss = actor + config.value_coef*critic - config.entropy_coef*entropy.mean() + imitation_weight*imitation
        if not torch.isfinite(loss): raise ValueError('nonfinite PPO loss')
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),config.max_grad_norm); optimizer.step()
    return {'loss':float(loss.detach()),'policy_loss':float(actor.detach()),'value_loss':float(critic.detach()),
            'entropy':float(entropy.mean().detach()),'imitation_weight':imitation_weight,'valid_rows':len(advantage),
            'reward':float(batch['reward'].sum())}


def _wilson(successes, count):
    z=1.96; p=successes/count; scale=1+z*z/count
    center=(p+z*z/(2*count))/scale
    radius=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/scale
    return [max(0.,center-radius),min(1.,center+radius)]


def _episode(policy, name, category, seed, device, horizon, action_ticks, *, demonstration=False):
    env=make_curriculum_env(name,category,1,device,seed,horizon=horizon)
    frames=[]; decisions=0
    stride=max(1,math.ceil(env.horizon/action_ticks/64))
    while True:
        obs=env._last_obs[:,3]; mask=obs[:,-8:].bool()
        with torch.no_grad():
            action=(teacher_actions(env,env._last_obs)[:,3] if policy == 'teacher' else policy.sample(obs,deterministic=True,action_mask=mask)[0])
        if demonstration or decisions%stride == 0:
            frame={'tick':int(env.steps[0]),'action':int(action[0]),'mask':mask[0].tolist(),
                   'hub_active':bool(env.hub_active[0,1]),'metrics':{k:v.tolist() for k,v in env.curriculum_metrics.outcomes().items()}}
            if demonstration: frame['observation']=obs[0].tolist()
            frames.append(frame)
        actions=torch.full((1,6),7,device=env.device,dtype=torch.long); actions[:,3]=action
        _,_,done,truncated,alive=_held_step(env,actions,torch.ones(1,device=env.device,dtype=torch.bool),action_ticks)
        if demonstration or decisions%stride == 0:
            frames[-1]['executed_action']=int(env.last_actions[0,3])
            frames[-1]['next_pose']=env.sim.pose[0,3].tolist()
        decisions+=1
        if bool(done[0] | truncated[0]): break
    terminal={'phase':'terminal','tick':int(env.steps[0]),'executed_action':int(env.last_actions[0,3]),
              'pose':env.sim.pose[0,3].tolist(),'hub_active':bool(env.hub_active[0,1]),
              'metrics':{k:v.tolist() for k,v in env.curriculum_metrics.outcomes().items()}}
    if not demonstration: frames.append(terminal)
    return {'scenario_id':name,'scenario_version':SCENARIOS[name].version,'category':category,'seed':seed,
            'physics':True,'dt':env.dt,'action_ticks':action_ticks,'strategic_rate_hz':1/(env.dt*action_ticks),'horizon_steps':env.horizon,
            'configuration':env.curriculum,'control_modes':list(env.control_modes),
            'robot_fuel_capacities':list(env.robot_fuel_capacity_values),'robot_types':list(env.robot_types),'execution':'generic_nn',
            'terminal':terminal,'outcome':{k:v[0].item() for k,v in env.curriculum_metrics.outcomes().items()},'frames':frames}


def evaluate_curriculum(policy_or_checkpoint, scenario_ids, *, episodes, split='validation',master_seed=0,
                        device='cpu',horizon=None,action_ticks=12,categories=None,output=None,
                        final_test=False,evaluation_id=None):
    """Independent paired episodes; abbreviated runs are diagnostic only."""
    torch.set_num_threads(1)
    if type(episodes) is not int or episodes<1 or type(action_ticks) is not int or action_ticks<1:
        raise ValueError('episodes and action_ticks must be positive integers')
    if split not in ('train','validation','test'): raise ValueError('invalid split')
    if isinstance(scenario_ids,str): scenario_ids=(scenario_ids,)
    sources=source_identity()
    checkpoint_hash=None
    if isinstance(policy_or_checkpoint,(str,Path)) and str(policy_or_checkpoint) != 'teacher':
        checkpoint_path=Path(policy_or_checkpoint);checkpoint_bytes=checkpoint_path.read_bytes()
        checkpoint_hash=hashlib.sha256(checkpoint_bytes).hexdigest()
        policy,_,_=load_checkpoint(io.BytesIO(checkpoint_bytes),device=device)
    else: policy=policy_or_checkpoint
    output=Path(output or ROOT/'evaluations/curriculum-offense'/f'{split}-{uuid.uuid4().hex}.json')
    if split == 'test':
        if not final_test or checkpoint_hash is None: raise ValueError('test requires explicit final_test and a selected checkpoint artifact')
        output.parent.mkdir(parents=True,exist_ok=True)
        frozen=output.parent/f'frozen-checkpoint-{checkpoint_hash}.pt'
        if not frozen.exists():
            with tempfile.NamedTemporaryFile(dir=output.parent,delete=False) as stream:
                temporary=Path(stream.name); stream.write(checkpoint_bytes)
            frozen.parent.mkdir(parents=True,exist_ok=True); os.replace(temporary,frozen)
        if hashlib.sha256(frozen.read_bytes()).hexdigest() != checkpoint_hash: raise ValueError('frozen checkpoint digest mismatch')
    if policy != 'teacher' and not isinstance(policy,ActorCritic): raise ValueError('expected standard ActorCritic, checkpoint, or teacher')
    identity=_policy_id(policy) if policy != 'teacher' else 'generic-observed-teacher'
    suite=protocol_id(action_ticks=action_ticks,device=device,_sources=sources); round_id=evaluation_id or uuid.uuid4().hex
    results=[];records=[];saved_rng=_rng_state()
    try:
        for name in scenario_ids:
            definition=SCENARIOS[name]; required=definition.required_categories
            chosen=tuple(categories) if categories is not None else required
            if not chosen or not set(chosen)<=set(required): raise ValueError('unknown or empty category selection')
            seeds=(seed_partition(name,split,episodes+1,master_seed=master_seed)[1:] if split=='test'
                   else seed_partition(name,split,episodes,master_seed=master_seed));category_results=[]
            for category in chosen:
                learner=[_episode(policy,name,category,s,device,horizon,action_ticks) for s in seeds]
                reference=[_episode('teacher',name,category,s,device,horizon,action_ticks) for s in seeds]
                successes=sum(bool(e['outcome']['success']) for e in learner)
                reference_successes=sum(bool(e['outcome']['success']) for e in reference)
                throughput=sum(e['outcome']['useful_throughput'] for e in learner)/episodes
                baseline=sum(e['outcome']['useful_throughput'] for e in reference)/episodes
                qualified=(horizon is None and episodes>=definition.gate.min_episodes
                           and reference_successes/episodes>=definition.gate.success_rate and baseline>0)
                measured=CategoryResult(category,episodes,successes,throughput,baseline,True,qualified,suite,seeds,seeds)
                category_results.append(measured)
                records.append({'scenario_id':name,'category':category,'success_interval_95':_wilson(successes,episodes),
                    'reference_successes':reference_successes,'reference_interval_95':_wilson(reference_successes,episodes),
                    'reference_qualified':qualified,'teacher':'generic NN execution; qualification is category-specific',
                    'learner':learner,'reference':reference})
            result=EvaluationResult(round_id,name,split,tuple(category_results),identity,protocol_id=suite)
            results.append(asdict(result))
    finally: _restore_rng(saved_rng)
    if source_identity()!=sources: raise ValueError('source/runtime changed during evaluation; no current report published')
    report={'curriculum_version':CURRICULUM_VERSION,'sources':sources,'protocol_id':suite,
            'policy_id':identity,'test_seed_index0_reserved':True,'checkpoint_file_sha256':checkpoint_hash,'evaluation_id':round_id,'split':split,
            'dt':.02,'action_ticks':action_ticks,'strategic_rate_hz':1/(.02*action_ticks),
            'diagnostic':horizon is not None or episodes<100 or categories is not None or split!='validation',
            'dashboard_verified':False,'evaluation_results':results,'records':records,'output':str(output)}
    _atomic_json(output,report);return report


def _decode_result(data):
    values=dict(data); values['categories']=tuple(CategoryResult(**(c|{'seeds':tuple(c['seeds']),'reference_seeds':tuple(c['reference_seeds'])})) for c in values['categories'])
    return EvaluationResult(**values)


def train_curriculum(config: TrainingConfig, *, resume=None):
    config.validate();torch.set_num_threads(1);sources=source_identity()
    output=Path(config.output);output.mkdir(parents=True,exist_ok=True)
    if resume is None and (output/'latest.pt').exists():
        raise ValueError('output already contains latest.pt; choose a fresh directory or resume explicitly')
    if resume:
        policy,optimizer,payload=load_checkpoint(resume,config=config,device=config.device,restore_rng=True)
        state=payload['curriculum_state'];artifacts=dict(payload['qualified_artifacts']);current=payload['current'];recovery=payload['recovery'];counter=payload['train_seed_counter']
        scheduler=random.Random();scheduler.setstate(payload['scheduler_rng']);start=payload['update'];stats=list(payload['stats'])
    else:
        torch.manual_seed(config.seed);random.seed(config.seed)
        policy=ActorCritic(137,8,action_kind='categorical').to(config.device)
        optimizer=torch.optim.Adam(policy.parameters(),lr=config.learning_rate)
        state=CurriculumState();artifacts={};current='pickup_simple';recovery=[];counter=0;start=0;stats=[];scheduler=random.Random(config.seed)
    for update in range(start+1,config.updates+1):
        eligible=set(eligible_scenarios(state));upcoming=tuple(s for s in SCENARIOS if s in eligible and s not in state.qualified and s!=current)
        practice=tuple(s for s in recovery if s in eligible and not (_ancestors(s) & set(recovery)))
        mixture=task_mixture(state,practice or (current,),upcoming if not practice else ())
        if practice:
            mixture={s:w for s,w in mixture.items() if not (_ancestors(s) & set(recovery))}
            total=sum(mixture.values());mixture={s:w/total for s,w in mixture.items()}
        name=scheduler.choices(list(mixture),weights=list(mixture.values()),k=1)[0]
        category=scheduler.choice(SCENARIOS[name].required_categories)
        if counter >= 2**21: raise ValueError('training seed partition exhausted')
        seed=seed_partition(name,'train',1,master_seed=config.seed)[0]+4*counter;counter+=1
        env=make_curriculum_env(name,category,config.num_envs,config.device,seed,horizon=config.horizon)
        # Factory uses the normal reset path; each update starts a fresh seeded
        # batch so an atomic checkpoint needs no simulator state.
        batch=_collect(policy,env,config);before=_policy_id(policy)
        metrics=_ppo_update(policy,optimizer,batch,config,update)
        row={'update':update,'scenario_id':name,'category':category,'seed':seed,'policy_changed':before!=_policy_id(policy),**metrics}
        promoted_name=None
        if config.eval_every and update%config.eval_every==0:
            names=tuple(s for s in SCENARIOS if s in _ancestors(current)|{current})
            report=evaluate_curriculum(policy,names,episodes=config.evaluation_episodes,master_seed=config.seed,
                device=config.device,horizon=config.horizon,action_ticks=config.action_ticks,
                output=output/f'evaluation-{update}.json',evaluation_id=f'update-{update}-{_policy_id(policy)}')
            if not report['diagnostic']:
                decoded={r.scenario_id:r for r in map(_decode_result,report['evaluation_results'])}
                decision=evaluate_promotion(state,decoded[current],retention_results=tuple(decoded[s] for s in _ancestors(current)))
                row['promotion']=asdict(decision)
                recovery=[s for s in _ancestors(current) if any(f.startswith(s+':') or f.startswith(s+'/') for f in decision.failures)]
                if decision.promoted:
                    promoted_name=current
                    artifacts[current]=str((output/'qualified'/current/f'{_policy_id(policy)}.pt').resolve())
                    available=[s for s in eligible_scenarios(state) if s not in state.qualified]
                    if available: current=available[0]
        stats.append(row)
        if source_identity()!=sources: raise ValueError('source/runtime changed during training; refusing incompatible checkpoint')
        payload={'checkpoint_version':CHECKPOINT_VERSION,'sources':sources,'config':asdict(config),
                 'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'policy_id':_policy_id(policy),
                 'rng':_rng_state(),'scheduler_rng':scheduler.getstate(),'update':update,
                 'curriculum_state':state.to_dict(),'current':current,'recovery':recovery,'qualified_artifacts':artifacts,'train_seed_counter':counter,'stats':stats}
        if promoted_name is not None:
            artifact=Path(artifacts[promoted_name])
            if artifact.exists():
                saved_policy,_,_=load_checkpoint(artifact,device=config.device)
                if _policy_id(saved_policy)!=payload['policy_id']: raise ValueError('qualified artifact collision')
            else: _atomic_checkpoint(artifact,payload)
            _atomic_checkpoint(output/'last-qualified.pt',payload)
        _atomic_checkpoint(output/'latest.pt',payload)
        _atomic_json(output/'stats.json',{'updates':update,'diagnostic':config.horizon is not None,'dt':.02,'action_ticks':config.action_ticks,'strategic_rate_hz':1/(.02*config.action_ticks),'stats':stats,'curriculum_state':state.to_dict(),'sources':payload['sources']})
    return {'updates':config.updates,'diagnostic':config.horizon is not None,'dt':.02,'action_ticks':config.action_ticks,'strategic_rate_hz':1/(.02*config.action_ticks),'checkpoint':str(output/'latest.pt') if config.updates>start else str(resume),'policy_id':_policy_id(policy),
            'curriculum_state':state.to_dict(),'current':current,'recovery':recovery,'qualified_artifacts':artifacts,'stats':stats}


def export_teacher(scenario_id, category, *, episodes=1, master_seed=0, device='cpu', horizon=None,
                   action_ticks=12, output=None):
    torch.set_num_threads(1)
    if type(episodes) is not int or episodes<1: raise ValueError('episodes must be positive')
    output=Path(output or ROOT/'evaluations/curriculum-offense/teacher.jsonl');output.parent.mkdir(parents=True,exist_ok=True)
    provenance=source_identity();count=0
    with tempfile.NamedTemporaryFile(mode='w',dir=output.parent,delete=False) as stream:
        temp=Path(stream.name)
        try:
            for seed in seed_partition(scenario_id,'train',episodes,master_seed=master_seed):
                record=_episode('teacher',scenario_id,category,seed,device,horizon,action_ticks,demonstration=True)
                for frame in record.pop('frames'):
                    stream.write(json.dumps({'sources':provenance,'teacher':'generic observed NN execution',
                        'reference_qualified':False,'dashboard_verified':False,**record,**frame},allow_nan=False)+'\n');count+=1
        except BaseException:
            temp.unlink(missing_ok=True);raise
    if source_identity()!=provenance:
        temp.unlink(missing_ok=True);raise ValueError('source/runtime changed during teacher export')
    os.replace(temp,output);return {'output':str(output),'rows':count,'reference_qualified':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Draft outcome-gated offense curriculum (unqualified scenarios)')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('plan')
    train=commands.add_parser('train')
    for option,kind,default in [('updates',int,1),('envs',int,4),('rollout-steps',int,16),('epochs',int,4),
        ('seed',int,7),('action-ticks',int,12),('evaluation-episodes',int,100),('eval-every',int,10),('imitation-coef',float,0.)]:
        train.add_argument('--'+option,type=kind,default=default)
    train.add_argument('--device',default='cpu');train.add_argument('--output',default=str(ROOT/'checkpoints/curriculum-offense'))
    train.add_argument('--resume');train.add_argument('--horizon',type=int)
    evaluate=commands.add_parser('evaluate');evaluate.add_argument('--checkpoint');evaluate.add_argument('--policy',choices=('teacher',))
    for command in (evaluate,commands.add_parser('teacher')):
        command.add_argument('--scenario',choices=tuple(SCENARIOS),required=True);command.add_argument('--category')
        command.add_argument('--episodes',type=int,default=1);command.add_argument('--seed',type=int,default=0)
        command.add_argument('--device',default='cpu');command.add_argument('--horizon',type=int)
        command.add_argument('--action-ticks',type=int,default=12);command.add_argument('--output')
    evaluate.add_argument('--split',choices=('train','validation','test'),default='validation');evaluate.add_argument('--final-test',action='store_true')
    args=parser.parse_args(argv);values=vars(args).copy();command=values.pop('command')
    if command=='plan': result={'version':CURRICULUM_VERSION,'qualified':False,'execution':'generic_nn','robot_types':['dumper']*6,'capacity_note':'30 FUEL for pickup primitives, 6 for delivery tasks; deployed 60-capacity qualification pending','scenarios':{s:asdict(d)|{'robot_fuel_capacities':[30 if s.startswith('pickup_') else 6]*6} for s,d in SCENARIOS.items()}}
    elif command=='train':
        resume=values.pop('resume');values['num_envs']=values.pop('envs');result=train_curriculum(TrainingConfig(**values),resume=resume)
    elif command=='evaluate':
        checkpoint=values.pop('checkpoint');policy=values.pop('policy')
        if bool(checkpoint)==bool(policy): parser.error('choose exactly one of --checkpoint or --policy teacher')
        scenario=values.pop('scenario');category=values.pop('category');values['master_seed']=values.pop('seed')
        result=evaluate_curriculum(checkpoint or policy,(scenario,),categories=(category,) if category else None,**values)
        result={k:v for k,v in result.items() if k not in ('records','evaluation_results','sources')}
    else:
        scenario=values.pop('scenario');category=values.pop('category') or SCENARIOS[scenario].required_categories[0]
        values['master_seed']=values.pop('seed');result=export_teacher(scenario,category,**values)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
