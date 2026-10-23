"""Opt-in offense PPO, paired evaluations, and observed-teacher export.

All seven selectable architectures use generic NN execution and sensor tracks.
Draft scenario targets and scripted teachers remain unqualified until measured.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from dataclasses import replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
import tempfile
import uuid

import torch
from torch.nn import functional as F

from .curriculum_scenarios import make_curriculum_env, reset_curriculum_scenario, teacher_actions
from .tensor_training_runtime import ActorCritic, _gae_advantages
from .curriculum_policy import ARCHITECTURES, PerceptionActorCritic, make_policy
from .architecture_inputs import PerceptionInputs
from .learning_contract import CandidateTable, ScorerSnapshot, masked_argmax
from .learning_inputs import build_observation_snapshot
from .discrete_sac import (DiscreteSAC, ScenarioReplayBuffer, make_sac_optimizers,
                           sac_update)
from .training_curriculum import (SCENARIOS, CURRICULUM_VERSION, CategoryResult, EvaluationResult,
    CurriculumState, eligible_scenarios, evaluate_promotion, seed_partition, task_mixture)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_VERSION = 2


class D1HandWeightedPolicy:
    """Opt-in hand-weighted ranking over the curriculum's sensed choices.

    The four collection choices are the environment's nearest valid sensor
    tracks. Other choices are the existing masked actions; this is not a pose
    generator or route planner.
    """
    name = 'd1_hand_weighted'
    weights = {'proximity': 1.0, 'nearby_tracks': .12,
               'confidence': .10, 'track_age': -.05,
               'score': .65, 'ferry': .20, 'intercept': -.25, 'hold': -.5}
    def choose(self, env, observation, *, snapshot_id=None):
        indices, valid, _ = env._candidates()
        slot = 3
        points = env.track_pos[:, slot].gather(
            1, indices[:, slot, :, None].expand(-1, -1, 2))
        distances = (points - env.sim.pose[:, slot, None, :2]).norm(dim=-1)
        confidence = env.track_confidence[:, slot].gather(1, indices[:, slot]).clamp(0., 1.)
        age = env.track_age[:, slot].gather(1, indices[:, slot]).clamp(0., 2.)
        track_points = env.track_pos[:, slot]
        sensed = env.track_mask[:, slot]
        nearby = (((points[:, :, None] - track_points[:, None]).norm(dim=-1) < 1.25) &
                  sensed[:, None]).sum(-1).clamp(max=4).to(points.dtype)
        candidate_valid = valid[:, slot] & torch.isfinite(distances)
        target_scores = (self.weights['proximity'] / (1. + distances) +
                         self.weights['nearby_tracks'] * nearby +
                         self.weights['confidence'] * confidence +
                         self.weights['track_age'] * age)
        target_scores = target_scores.masked_fill(~candidate_valid, float('-inf'))

        if env.n != 1:
            raise ValueError('D1 action-trace evaluation requires one world')
        mask = observation[0, -8:].bool().tolist()
        snapshot_id = snapshot_id or f'd1:{int(env.steps[0].item())}'
        feature_names = ('action_index', 'sensor_track_buffer_index', 'target_distance_m',
                         'nearby_sensed_track_count', 'track_confidence',
                         'track_age_s')
        fixed_ids = {4: 'score', 5: 'intercept', 6: 'ferry', 7: 'hold'}
        candidate_ids = []
        features = []
        for row in range(8):
            if row < 4:
                track_index = int(indices[0, slot, row].item())
                if bool(candidate_valid[0, row].item()):
                    candidate_ids.append(f'collect:track:{track_index}')
                    row_features = (float(row), float(track_index),
                                    float(distances[0, row].item()),
                                    float(nearby[0, row].item()),
                                    float(confidence[0, row].item()),
                                    float(age[0, row].item()))
                else:
                    candidate_ids.append(f'collect:unavailable-slot:{row}')
                    row_features = (float(row), None, None, None, None, None)
            else:
                candidate_ids.append(fixed_ids[row])
                row_features = (float(row), None, None, None, None, None)
            features.append(row_features)
        table = CandidateTable(snapshot_id, tuple(candidate_ids), tuple(features),
                               tuple(bool(value) for value in mask), feature_names,
                               route_costs=(None,) * 8)
        fixed_scores = {'score': self.weights['score'],
                        'intercept': self.weights['intercept'],
                        'ferry': self.weights['ferry'],
                        'hold': self.weights['hold']}
        row_scores = []
        for row, candidate_id in enumerate(table.candidate_ids):
            if row < 4:
                row_scores.append(float(target_scores[0, row].item()))
            else:
                row_scores.append(float(fixed_scores[candidate_id]))
        selected_row = masked_argmax(row_scores, table.valid_mask,
                                     fallback_index=7)
        if selected_row is None:
            raise RuntimeError('D1 requires at least one legal curriculum action')
        # The table's named row and action_index feature define the scorer mapping.
        action_index = int(table.features[selected_row][0])
        action = torch.tensor([action_index], device=env.device, dtype=torch.long)
        scores = torch.tensor([row_scores], device=env.device, dtype=observation.dtype)
        scores = scores.masked_fill(~torch.tensor([table.valid_mask], device=env.device),
                                    float('-inf'))
        trace = {'candidate_scores': target_scores, 'action_scores': scores,
                 'candidate_indices': indices[:, slot], 'candidate_valid': candidate_valid,
                 'candidate_distance': distances, 'candidate_nearby_tracks': nearby,
                 'candidate_confidence': confidence, 'candidate_age': age,
                 'candidate_table': table, 'selected_row': selected_row}
        return action, trace


class D2PredictivePolicy(D1HandWeightedPolicy):
    """D1 action ranking with a bounded observed-opponent endpoint penalty."""
    name = 'd2_predictive_baseline'
    prediction_horizon_s = 1.0
    max_speed_mps = 3.0
    max_track_age_s = 0.5
    threat_radius_m = 2.0
    max_penalty = 0.75

    def choose(self, env, observation, *, snapshot_id=None):
        decision_started = time.perf_counter()
        action, trace = super().choose(env, observation, snapshot_id=snapshot_id)
        prediction_started = time.perf_counter()
        slot = 3
        observed_pose = env.opponent_pose[:, slot, :2]
        observed_velocity = env.opponent_velocity[:, slot, :2]
        age = env.opponent_age[:, slot]
        valid = (env.opponent_valid[:, slot] & torch.isfinite(observed_pose).all(-1) &
                 torch.isfinite(observed_velocity).all(-1) & torch.isfinite(age) &
                 (age >= 0.) & (age <= self.max_track_age_s))
        speed = observed_velocity.norm(dim=-1)
        bounded_velocity = observed_velocity * (self.max_speed_mps / speed.clamp_min(1e-8)).clamp(max=1.)[:, None]
        predicted = observed_pose + bounded_velocity * self.prediction_horizon_s
        predicted = torch.where(valid[:, None], predicted, torch.full_like(predicted, float('nan')))
        candidate_points = env.track_pos[:, slot].gather(
            1, trace['candidate_indices'][..., None].expand(-1, -1, 2))
        endpoint_distance = (candidate_points - predicted[:, None]).norm(dim=-1)
        penalty = self.max_penalty * ((self.threat_radius_m - endpoint_distance) /
                                      self.threat_radius_m).clamp(0., 1.)
        penalty = torch.where(valid[:, None] & trace['candidate_valid'], penalty,
                              torch.zeros_like(penalty))
        candidate_scores = trace['candidate_scores'] - penalty
        row_scores = trace['action_scores'].clone()
        row_scores[:, :4] = candidate_scores
        mask = torch.tensor([trace['candidate_table'].valid_mask], device=env.device,
                            dtype=torch.bool)
        row_scores = row_scores.masked_fill(~mask, float('-inf'))
        selected = masked_argmax(row_scores[0].tolist(), trace['candidate_table'].valid_mask,
                                 fallback_index=7)
        if selected is None:
            raise RuntimeError('D2 requires at least one legal curriculum action')
        action = torch.tensor([int(trace['candidate_table'].features[selected][0])],
                              device=env.device, dtype=torch.long)
        table = trace['candidate_table']
        enriched = replace(table,
            feature_names=table.feature_names + ('predicted_opponent_endpoint_distance_m', 'threat_penalty'),
            features=tuple(row + ((float(endpoint_distance[0, i].item()) if bool(valid[0]) and i < 4 and bool(trace['candidate_valid'][0, i]) else None),
                                  (float(penalty[0, i].item()) if i < 4 else None))
                           for i, row in enumerate(table.features)))
        trace.update({'candidate_scores': candidate_scores, 'action_scores': row_scores,
                      'candidate_table': enriched, 'selected_row': selected,
                      'observed_opponent_pose': observed_pose.detach().clone(),
                      'observed_opponent_velocity': observed_velocity.detach().clone(),
                      'observed_opponent_age': age.detach().clone(),
                      'observed_opponent_valid': env.opponent_valid[:, slot].detach().clone(),
                      'predicted_opponent_endpoint': predicted.detach().clone(),
                      'candidate_endpoint_distance': endpoint_distance,
                      'threat_penalty': penalty,
                      'prediction_seconds': time.perf_counter() - prediction_started,
                      'decision_seconds': time.perf_counter() - decision_started})
        return action, trace


def _d2_identity():
    return _hash({'policy': D2PredictivePolicy.name,
                  'base': _d1_identity(),
                  'prediction': {'horizon_s': D2PredictivePolicy.prediction_horizon_s,
                                'max_speed_mps': D2PredictivePolicy.max_speed_mps,
                                'max_track_age_s': D2PredictivePolicy.max_track_age_s,
                                'threat_radius_m': D2PredictivePolicy.threat_radius_m,
                                'max_penalty': D2PredictivePolicy.max_penalty},
                  'choices': 'd2-endpoint-threat-v1'})


def _d1_identity():
    return _hash({'policy': D1HandWeightedPolicy.name,
                  'weights': D1HandWeightedPolicy.weights,
                  'candidate_source': 'per-snapshot-four-nearest-valid-sensor-track-buffer-slots',
                  'choices': 'd1-action-trace-v2'})


def _d1_artifacts(env, observation, trace, *, scenario, category, seed,
                  evaluation_id):
    """Serialize D1's exact action ordering and score snapshot for audit."""
    tick = int(env.steps[0].item())
    # Bind every observation/scorer snapshot to the enclosing evaluation so a
    # delayed result from a repeated same-seed run cannot match this run.
    episode_id = f'{evaluation_id}:{scenario}:{category}:seed-{seed}'
    observation_snapshot = build_observation_snapshot(
        env, episode_id=episode_id, sequence=tick)
    snapshot_id = observation_snapshot.snapshot_id
    table = trace['candidate_table']
    if table.snapshot_id != snapshot_id:
        raise RuntimeError('D1 candidate table does not match its observation snapshot')
    raw_scores = trace['action_scores'][0].detach().cpu().tolist()
    scores = tuple(float(score) if math.isfinite(float(score)) else 0.
                   for score in raw_scores)
    timestamp = observation_snapshot.timestamp
    scorer_id = _d2_identity() if 'prediction_seconds' in trace else _d1_identity()
    scored = ScorerSnapshot(scorer_id, snapshot_id, table.candidate_ids,
                            scores, timestamp)
    if not scored.matches(observation_snapshot, table, now=timestamp, max_age=0.):
        raise RuntimeError('D1 scorer snapshot does not match its candidate table')
    return {'candidate_table': asdict(table), 'scorer_snapshot': asdict(scored),
            'selected_candidate_id': table.candidate_ids[trace['selected_row']]}


@dataclass(frozen=True)
class TrainingConfig:
    architecture: str = 'baseline_mlp'
    algorithm: str = 'ppo'
    disable_miopen: bool = False
    minibatch_size: int = 0
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
    learning_rate_start: float = 3e-3
    learning_rate_decay_updates: int = 75
    gamma: float = .993
    gae_lambda: float = .95
    clip_coef: float = .2
    value_coef: float = .5
    entropy_coef: float = .01
    max_grad_norm: float = .5
    sac_replay_capacity: int = 4096
    sac_batch_size: int = 256
    sac_warmup_steps: int = 1024
    sac_updates_per_transition: float = .25
    sac_target_tau: float = .005
    sac_initial_alpha: float = .2
    sac_target_entropy_fraction: float = .98

    def validate(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError('unknown architecture')
        if self.algorithm not in ('ppo', 'sac'):
            raise ValueError('algorithm must be ppo or sac')
        if self.algorithm == 'sac' and self.imitation_coef != 0:
            raise ValueError('imitation_coef is only supported by PPO')
        if type(self.disable_miopen) is not bool or type(self.minibatch_size) is not int or self.minibatch_size < 0:
            raise ValueError('invalid backend or minibatch configuration')
        for name in ('updates', 'num_envs', 'rollout_steps', 'epochs', 'action_ticks', 'evaluation_episodes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int or self.seed < 0 or type(self.eval_every) is not int or self.eval_every < 0:
            raise ValueError('seed and eval_every must be nonnegative integers')
        if self.horizon is not None and (type(self.horizon) is not int or self.horizon < 1):
            raise ValueError('horizon must be positive or None')
        for name in ('imitation_coef', 'learning_rate', 'learning_rate_start', 'gamma', 'gae_lambda', 'clip_coef', 'value_coef', 'entropy_coef', 'max_grad_norm'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if (self.learning_rate == 0 or self.learning_rate_start == 0 or
                self.learning_rate_start < self.learning_rate or self.max_grad_norm == 0 or
                self.gamma > 1 or self.gae_lambda > 1):
            raise ValueError('invalid learning rate, gradient norm, or discount')
        if type(self.learning_rate_decay_updates) is not int or self.learning_rate_decay_updates < 1:
            raise ValueError('learning_rate_decay_updates must be a positive integer')
        if self.algorithm == 'sac':
            if (type(self.sac_replay_capacity) is not int or type(self.sac_batch_size) is not int or
                    type(self.sac_warmup_steps) is not int or self.sac_replay_capacity < 1 or
                    self.sac_batch_size < 1 or self.sac_warmup_steps < self.sac_batch_size or
                    self.sac_replay_capacity < self.sac_warmup_steps):
                raise ValueError('SAC replay capacity and warmup must cover a full batch')
            for name in ('sac_updates_per_transition', 'sac_target_tau', 'sac_initial_alpha',
                         'sac_target_entropy_fraction'):
                value = getattr(self, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} must be finite and positive')
            if self.sac_target_tau > 1 or self.sac_target_entropy_fraction > 1:
                raise ValueError('SAC target rates must be at most one')


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


def _stage_learning_rate(update, stats, config):
    stage_start = max((row['update'] + 1 for row in stats
                       if row.get('promotion', {}).get('promoted')), default=1)
    if config.learning_rate_decay_updates == 1:
        return config.learning_rate
    progress = min(max((update - stage_start) /
                       (config.learning_rate_decay_updates - 1), 0.), 1.)
    return math.exp(math.log(config.learning_rate_start) + progress *
                    (math.log(config.learning_rate) - math.log(config.learning_rate_start)))


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
    expected={'checkpoint_version','sources','config','policy','optimizer','replay_buffer','policy_id','rng','scheduler_rng','update','curriculum_state','current','recovery','train_seed_counter','stats','qualified_artifacts'}
    if not isinstance(payload,dict) or set(payload)!=expected: raise ValueError('invalid checkpoint schema')
    if payload.get('checkpoint_version') != CHECKPOINT_VERSION or payload.get('sources') != source_identity():
        raise ValueError('checkpoint format/source identity mismatch')
    saved = TrainingConfig(**payload['config']); saved.validate()
    if saved.disable_miopen: torch.backends.cudnn.enabled=False
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
    if saved.algorithm == 'sac':
        policy = DiscreteSAC(saved.architecture, saved.sac_initial_alpha).to(device)
        optimizer = make_sac_optimizers(policy, saved.learning_rate)
    else:
        policy = make_policy(saved.architecture).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=saved.learning_rate)
    policy.load_state_dict(payload['policy'])
    if saved.algorithm == 'sac':
        if not isinstance(payload['optimizer'], dict) or set(payload['optimizer']) != set(optimizer):
            raise ValueError('invalid SAC optimizer state')
        for name, item in optimizer.items():
            item.load_state_dict(payload['optimizer'][name])
        if not isinstance(payload['replay_buffer'], dict):
            raise ValueError('missing SAC replay state')
    else:
        if payload['replay_buffer'] is not None:
            raise ValueError('PPO checkpoint cannot contain SAC replay')
        optimizer.load_state_dict(payload['optimizer'])
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
    actor = policy.actor if isinstance(policy, DiscreteSAC) else policy
    builder = PerceptionInputs(env, (3,)) if isinstance(actor, PerceptionActorCritic) else None
    rows = []; alive = torch.ones(env.n, device=env.device, dtype=torch.bool)
    steps=config.rollout_steps if config.horizon is not None else max(config.rollout_steps,math.ceil(env.horizon/config.action_ticks))
    for _ in range(steps):
        valid = alive.clone(); obs = env._last_obs[:,3].clone(); mask = obs[:,-8:].bool()
        inputs = ({k:v.clone() for k,v in builder(env._last_obs).items()} if builder else obs)
        with torch.no_grad():
            action, logprob, value = policy.sample(inputs, action_mask=mask)
            teacher = (teacher_actions(env,env._last_obs)[:,3].clone()
                       if config.algorithm == 'ppo' else torch.zeros_like(action))
        actions = torch.full((env.n,6),7,device=env.device,dtype=torch.long); actions[:,3] = action
        next_obs,reward,done,truncated,alive = _held_step(env,actions,alive,config.action_ticks)
        row={'obs':inputs,'mask':mask,'action':action,'logprob':logprob,'value':value,
                     'next_value':None,'reward':reward,'done':done | ~valid,'truncated':truncated,
                     'valid':valid,'teacher':teacher}
        if config.algorithm == 'sac':
            row['next_obs']=({k:v.clone() for k,v in builder(next_obs).items()}
                             if builder else next_obs[:,3].clone())
            row['next_mask']=next_obs[:,3,-8:].bool().clone()
            row['next_value']=torch.zeros_like(value)
        else:
            next_value_input=builder(next_obs) if builder else next_obs[:,3]
            with torch.no_grad(): row['next_value'] = policy(next_value_input)[-1]
        rows.append(row)
        if not bool(alive.any()): break
    # The environment is deliberately discarded at this update boundary.
    # Remaining worlds are timeouts, not true physical terminations.
    rows[-1]['truncated'] |= alive
    batch = {key:({k:torch.stack([r[key][k] for r in rows]) for k in rows[0][key]}
                  if isinstance(rows[0][key],dict) else torch.stack([r[key] for r in rows])) for key in rows[0]}
    advantages = _gae_advantages(batch['reward'],batch['done'],batch['truncated'],batch['value'],
                                 batch['next_value'],config.gamma,config.gae_lambda)
    batch['returns'] = advantages + batch['value']; batch['advantages'] = advantages
    def flatten(value):
        if isinstance(value,dict): return {k:flatten(v) for k,v in value.items()}
        return value.reshape((-1,)+value.shape[2:])[batch['valid'].reshape(-1)]
    return {k:flatten(v) for k,v in batch.items()}


def _ppo_update(policy, optimizer, batch, config, update):
    advantage = batch['advantages']; advantage = (advantage-advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    imitation_weight = config.imitation_coef / (1 + update/100)
    losses=[]
    approx_kl_total=advantage.new_zeros(()); clip_fraction_total=advantage.new_zeros(())
    optimized_samples=0
    for _ in range(config.epochs):
        order=(torch.randperm(len(advantage),device=advantage.device) if config.minibatch_size else
               torch.arange(len(advantage),device=advantage.device))
        size=config.minibatch_size or len(advantage)
        for indices in order.split(size):
            observations=({k:v[indices] for k,v in batch['obs'].items()} if isinstance(batch['obs'],dict)
                          else batch['obs'][indices])
            logprob,entropy,value,logits = policy.score(observations,batch['action'][indices],batch['mask'][indices])
            ratio = (logprob-batch['logprob'][indices]).exp()
            logratio=logprob-batch['logprob'][indices]
            approx_kl_total+=((ratio-1.)-logratio).detach().sum()
            clip_fraction_total+=((ratio-1.).abs()>config.clip_coef).sum()
            optimized_samples+=len(indices)
            actor = -torch.minimum(ratio*advantage[indices], ratio.clamp(1-config.clip_coef,1+config.clip_coef)*advantage[indices]).mean()
            critic = .5*F.mse_loss(value,batch['returns'][indices])
            imitation = F.cross_entropy(logits,batch['teacher'][indices])
            loss = actor + config.value_coef*critic - config.entropy_coef*entropy.mean() + imitation_weight*imitation
            if not torch.isfinite(loss): raise ValueError('nonfinite PPO loss')
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),config.max_grad_norm); optimizer.step()
            losses.append((len(indices),float(loss.detach()),float(actor.detach()),float(critic.detach()),float(entropy.mean().detach())))
    total=sum(row[0] for row in losses)
    averaged=[sum(row[0]*row[column] for row in losses)/total for column in range(1,5)]
    return dict(zip(('loss','policy_loss','value_loss','entropy'),averaged)) | {
            'imitation_weight':imitation_weight,'valid_rows':len(advantage),'reward':float(batch['reward'].sum()),
            'approx_kl':float((approx_kl_total/max(1,optimized_samples)).item()),
            'clip_fraction':float((clip_fraction_total/max(1,optimized_samples)).item())}



def _wilson(successes, count):
    z=1.96; p=successes/count; scale=1+z*z/count
    center=(p+z*z/(2*count))/scale
    radius=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/scale
    return [max(0.,center-radius),min(1.,center+radius)]


def _episode(policy, name, category, seed, device, horizon, action_ticks, *,
             demonstration=False, evaluation_id=None):
    evaluation_id = evaluation_id or uuid.uuid4().hex
    env=make_curriculum_env(name,category,1,device,seed,horizon=horizon)
    actor=policy.actor if isinstance(policy,DiscreteSAC) else policy
    builder=PerceptionInputs(env,(3,)) if isinstance(actor,PerceptionActorCritic) else None
    frames=[]; decisions=0
    stride=max(1,math.ceil(env.horizon/action_ticks/64))
    episode_id=f'{evaluation_id}:{name}:{category}:seed-{seed}'
    while True:
        obs=env._last_obs[:,3]; mask=obs[:,-8:].bool()
        trace=None
        with torch.no_grad():
            if policy == 'teacher':
                action=teacher_actions(env,env._last_obs)[:,3]
            elif isinstance(policy,D1HandWeightedPolicy):
                snapshot_id=f'{episode_id}:{int(env.steps[0].item())}'
                action,trace=policy.choose(env,obs,snapshot_id=snapshot_id)
            else:
                action=policy.sample(builder(env._last_obs) if builder else obs,deterministic=True,action_mask=mask)[0]
        if demonstration or decisions%stride == 0:
            frame={'tick':int(env.steps[0]),'action':int(action[0]),'mask':mask[0].tolist(),
                   'hub_active':bool(env.hub_active[0,1]),'metrics':{k:v.tolist() for k,v in env.curriculum_metrics.outcomes().items()}}
            if isinstance(policy,D1HandWeightedPolicy):
                frame['policy_id']=_d2_identity() if isinstance(policy,D2PredictivePolicy) else _d1_identity()
                frame['candidate_action_trace']={
                    'candidate_indices':trace['candidate_indices'][0].tolist(),
                    'candidate_valid':trace['candidate_valid'][0].tolist(),
                    'candidate_scores':[float(x) if math.isfinite(float(x)) else None for x in trace['candidate_scores'][0].tolist()],
                    'action_scores':[float(x) if math.isfinite(float(x)) else None for x in trace['action_scores'][0].tolist()],
                    'candidate_distance':trace['candidate_distance'][0].tolist(),
                    'candidate_nearby_tracks':trace['candidate_nearby_tracks'][0].tolist(),
                    'candidate_confidence':trace['candidate_confidence'][0].tolist(),
                    'candidate_age':trace['candidate_age'][0].tolist()}
                if isinstance(policy,D2PredictivePolicy):
                    frame['prediction']={
                        'schema_version':1,
                        'observed_state':{'valid':bool(trace['observed_opponent_valid'][0]),
                            'age_s':(float(trace['observed_opponent_age'][0])
                                     if math.isfinite(float(trace['observed_opponent_age'][0])) else None),
                            'pose_xy_m':trace['observed_opponent_pose'][0].tolist(),
                            'velocity_xy_mps':trace['observed_opponent_velocity'][0].tolist()},
                        'predicted_endpoint_xy_m':trace['predicted_opponent_endpoint'][0].tolist()
                            if bool(torch.isfinite(trace['predicted_opponent_endpoint'][0]).all()) else None,
                        'used':bool(trace['observed_opponent_valid'][0] and
                                    trace['observed_opponent_age'][0] <= policy.max_track_age_s),
                        'horizon_s':policy.prediction_horizon_s,
                        'candidate_endpoint_distance_m':[
                            float(x) if math.isfinite(float(x)) else None
                            for x in trace['candidate_endpoint_distance'][0].tolist()],
                        'collection_penalty':trace['threat_penalty'][0].tolist(),
                        'prediction_seconds':trace['prediction_seconds'],
                        'decision_seconds':trace['decision_seconds']}
                frame.update(_d1_artifacts(env, obs, trace, scenario=name,
                                           category=category, seed=seed,
                                           evaluation_id=evaluation_id))
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
    if isinstance(policy_or_checkpoint,(str,Path)) and str(policy_or_checkpoint) == 'd1':
        policy=D1HandWeightedPolicy()
    elif isinstance(policy_or_checkpoint,(str,Path)) and str(policy_or_checkpoint) == 'd2':
        policy=D2PredictivePolicy()
    elif isinstance(policy_or_checkpoint,(str,Path)) and str(policy_or_checkpoint) != 'teacher':
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
    if policy != 'teacher' and not isinstance(policy,(ActorCritic,DiscreteSAC,D1HandWeightedPolicy)):
        raise ValueError('expected standard policy, SAC policy, checkpoint, D1, D2, or teacher')
    identity=(_d2_identity() if isinstance(policy,D2PredictivePolicy) else
              _d1_identity() if isinstance(policy,D1HandWeightedPolicy) else
              _policy_id(policy) if policy != 'teacher' else 'generic-observed-teacher')
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
                learner=[_episode(policy,name,category,s,device,horizon,action_ticks,
                                  evaluation_id=round_id) for s in seeds]
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
            'algorithm':getattr(policy,'algorithm','d2_predictive_hand_weighted' if isinstance(policy,D2PredictivePolicy) else 'd1_hand_weighted' if isinstance(policy,D1HandWeightedPolicy) else 'teacher' if policy == 'teacher' else 'ppo'),
            'architecture':getattr(policy,'architecture','candidate_action_ranker' if isinstance(policy,D1HandWeightedPolicy) else 'baseline_mlp') if policy != 'teacher' else 'deterministic_teacher',
            'policy_provenance':({'kind':'d2_observed_opponent_endpoint_penalty' if isinstance(policy,D2PredictivePolicy) else 'hand_weighted_candidate_action_ranker',
                'candidate_source':'per-snapshot-four-nearest-valid-sensor-track-buffer-slots',
                'action_source':'masked curriculum eight-action interface',
                'weights':D1HandWeightedPolicy.weights,
                'prediction':({'schema_version':1,'horizon_s':policy.prediction_horizon_s,
                    'max_speed_mps':policy.max_speed_mps,'max_track_age_s':policy.max_track_age_s,
                    'threat_radius_m':policy.threat_radius_m,'max_penalty':policy.max_penalty,
                    'penalized_candidates':'collection rows only'} if isinstance(policy,D2PredictivePolicy) else None),
                'uses_hidden_fuel_or_opponent_truth':False}
                if isinstance(policy,D1HandWeightedPolicy) else None),
            'policy_id':identity,'test_seed_index0_reserved':True,'checkpoint_file_sha256':checkpoint_hash,'evaluation_id':round_id,'split':split,
            'dt':.02,'action_ticks':action_ticks,'strategic_rate_hz':1/(.02*action_ticks),
            'diagnostic':horizon is not None or episodes<100 or categories is not None or split!='validation',
            'dashboard_verified':False,'evaluation_results':results,'records':records,'output':str(output)}
    _atomic_json(output,report);return report


def _decode_result(data):
    values=dict(data); values['categories']=tuple(CategoryResult(**(c|{'seeds':tuple(c['seeds']),'reference_seeds':tuple(c['reference_seeds'])})) for c in values['categories'])
    return EvaluationResult(**values)


def train_curriculum(config: TrainingConfig, *, resume=None):
    config.validate();torch.set_num_threads(1)
    if config.disable_miopen: torch.backends.cudnn.enabled=False
    sources=source_identity()
    output=Path(config.output);output.mkdir(parents=True,exist_ok=True)
    if resume is None and (output/'latest.pt').exists():
        raise ValueError('output already contains latest.pt; choose a fresh directory or resume explicitly')
    if resume:
        policy,optimizer,payload=load_checkpoint(resume,config=config,device=config.device,restore_rng=True)
        replay = ScenarioReplayBuffer(config.sac_replay_capacity) if config.algorithm == 'sac' else None
        if replay is not None: replay.load_state_dict(payload['replay_buffer'])
        state=payload['curriculum_state'];artifacts=dict(payload['qualified_artifacts']);current=payload['current'];recovery=payload['recovery'];counter=payload['train_seed_counter']
        scheduler=random.Random();scheduler.setstate(payload['scheduler_rng']);start=payload['update'];stats=list(payload['stats'])
    else:
        torch.manual_seed(config.seed);random.seed(config.seed)
        if config.algorithm == 'sac':
            policy=DiscreteSAC(config.architecture,config.sac_initial_alpha).to(config.device)
            optimizer=make_sac_optimizers(policy,config.learning_rate)
            replay=ScenarioReplayBuffer(config.sac_replay_capacity)
        else:
            policy=make_policy(config.architecture).to(config.device)
            optimizer=torch.optim.Adam(policy.parameters(),lr=config.learning_rate)
            replay=None
        state=CurriculumState();artifacts={};current='pickup_simple';recovery=[];counter=0;start=0;stats=[];scheduler=random.Random(config.seed)
    def checkpoint_payload():
        optimizer_state = ({name:item.state_dict() for name,item in optimizer.items()}
                           if isinstance(optimizer,dict) else optimizer.state_dict())
        return {'checkpoint_version':CHECKPOINT_VERSION,'sources':sources,'config':asdict(config),
                'policy':policy.state_dict(),'optimizer':optimizer_state,
                'replay_buffer':replay.state_dict() if replay is not None else None,
                'policy_id':_policy_id(policy),
                'rng':_rng_state(),'scheduler_rng':scheduler.getstate(),'update':update,
                'curriculum_state':state.to_dict(),'current':current,'recovery':recovery,
                'qualified_artifacts':artifacts,'train_seed_counter':counter,'stats':stats}

    for update in range(start+1,config.updates+1):
        stage_lr = _stage_learning_rate(update,stats,config)
        optimizer_groups=(group for item in optimizer.values() for group in item.param_groups) \
            if isinstance(optimizer,dict) else iter(optimizer.param_groups)
        for group in optimizer_groups:
            group['lr']=stage_lr
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
        if config.algorithm == 'sac':
            replay.add_batch(batch,name)
            if len(replay) >= config.sac_warmup_steps:
                gradient_steps=max(1,round(batch['action'].shape[0]*config.sac_updates_per_transition))
                updates_metrics=[sac_update(policy,optimizer,replay.sample(config.sac_batch_size,config.device),
                    gamma=config.gamma,tau=config.sac_target_tau,
                    target_entropy_fraction=config.sac_target_entropy_fraction)
                    for _ in range(gradient_steps)]
                metric_names=tuple(updates_metrics[0])
                metric_values=torch.stack([
                    torch.stack([item[key] for item in updates_metrics]).mean()
                    for key in metric_names
                ]).detach().cpu().tolist()
                metrics=dict(zip(metric_names,metric_values))
            else:
                metrics={'actor_loss':None,'critic_loss':None,'alpha_loss':None,
                         'alpha':float(policy.alpha.detach()),'entropy':None,'valid_action_count':None}
                gradient_steps=0
            metrics.update({'sac_updates':gradient_steps,'replay_size':len(replay),
                            'replay_scenarios':len(replay.scenario_indices),
                            'approx_kl':None,'clip_fraction':None})
        else:
            metrics=_ppo_update(policy,optimizer,batch,config,update)
        row={'update':update,'algorithm':config.algorithm,'learning_rate':stage_lr,
             'scenario_id':name,'category':category,'seed':seed,
             'policy_changed':before!=_policy_id(policy),**metrics}
        stats.append(row)
        promoted_name=None
        if config.eval_every and update%config.eval_every==0:
            # Preserve completed optimization if paired validation is interrupted.
            # A resumed run retries qualification at its next scheduled gate.
            row['evaluation_pending']=True
            if source_identity()!=sources: raise ValueError('source/runtime changed before evaluation')
            _atomic_checkpoint(output/'latest.pt',checkpoint_payload())
            names=tuple(s for s in SCENARIOS if s in _ancestors(current)|{current})
            report=evaluate_curriculum(policy,names,episodes=config.evaluation_episodes,master_seed=config.seed,
                device=config.device,horizon=config.horizon,action_ticks=config.action_ticks,
                output=output/f'evaluation-{update}.json',evaluation_id=f'update-{update}-{_policy_id(policy)}')
            row['evaluation_pending']=False
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
        if source_identity()!=sources: raise ValueError('source/runtime changed during training; refusing incompatible checkpoint')
        payload=checkpoint_payload()
        if promoted_name is not None:
            artifact=Path(artifacts[promoted_name])
            if artifact.exists():
                saved_policy,_,_=load_checkpoint(artifact,device=config.device)
                if _policy_id(saved_policy)!=payload['policy_id']: raise ValueError('qualified artifact collision')
            else: _atomic_checkpoint(artifact,payload)
            _atomic_checkpoint(output/'last-qualified.pt',payload)
        _atomic_checkpoint(output/'latest.pt',payload)
        _atomic_json(output/'stats.json',{'algorithm':config.algorithm,'config':asdict(config),
            'updates':update,'current':current,'diagnostic':config.horizon is not None,'dt':.02,
            'action_ticks':config.action_ticks,'strategic_rate_hz':1/(.02*config.action_ticks),
            'stats':stats,'architecture':config.architecture,
            'curriculum_state':state.to_dict(),'sources':payload['sources']})
    return {'algorithm':config.algorithm,'architecture':config.architecture,'updates':config.updates,'diagnostic':config.horizon is not None,'dt':.02,'action_ticks':config.action_ticks,'strategic_rate_hz':1/(.02*config.action_ticks),'checkpoint':str(output/'latest.pt') if config.updates>start else str(resume),'policy_id':_policy_id(policy),
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
    train.add_argument('--architecture',choices=tuple(ARCHITECTURES),default='baseline_mlp')
    train.add_argument('--algorithm',choices=('ppo','sac'),default='ppo')
    train.add_argument('--learning-rate',type=float,default=3e-4)
    train.add_argument('--learning-rate-start',type=float,default=3e-3)
    train.add_argument('--learning-rate-decay-updates',type=int,default=75)
    train.add_argument('--disable-miopen',action='store_true')
    train.add_argument('--minibatch-size',type=int,default=0)
    train.add_argument('--sac-replay-capacity',type=int,default=4096)
    train.add_argument('--sac-batch-size',type=int,default=256)
    train.add_argument('--sac-warmup-steps',type=int,default=1024)
    train.add_argument('--sac-updates-per-transition',type=float,default=.25)
    train.add_argument('--sac-target-tau',type=float,default=.005)
    train.add_argument('--sac-initial-alpha',type=float,default=.2)
    train.add_argument('--sac-target-entropy-fraction',type=float,default=.98)
    train.add_argument('--resume');train.add_argument('--horizon',type=int)
    evaluate=commands.add_parser('evaluate');evaluate.add_argument('--checkpoint');evaluate.add_argument('--policy',choices=('teacher','d1','d2'))
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
        if bool(checkpoint)==bool(policy): parser.error('choose exactly one of --checkpoint or --policy teacher|d1|d2')
        scenario=values.pop('scenario');category=values.pop('category');values['master_seed']=values.pop('seed')
        result=evaluate_curriculum(checkpoint or policy,(scenario,),categories=(category,) if category else None,**values)
        result={k:v for k,v in result.items() if k not in ('records','evaluation_results','sources')}
    else:
        scenario=values.pop('scenario');category=values.pop('category') or SCENARIOS[scenario].required_categories[0]
        values['master_seed']=values.pop('seed');result=export_teacher(scenario,category,**values)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
