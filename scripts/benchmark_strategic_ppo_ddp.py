#!/usr/bin/env python3
"""Opt-in, isolated two-GPU DDP PPO throughput prototype.

Launch with ``.venv/bin/python -m torch.distributed.run --nproc_per_node=2 ...``.
Each rank owns an independent strategic environment shard.  A global shuffled
minibatch is partitioned evenly across ranks and DDP averages the gradients.
Worlds use rank-offset seeds; this is throughput comparison, not scenario-level
parity with a monolithic environment batch.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from frc_defense.tensor_training import ActorCritic, _env, _held_action_interval, _reset_obs


class ScoreForward(nn.Module):
    """Route score() through DDP's forward so gradient hooks synchronize."""
    def __init__(self, policy: ActorCritic):
        super().__init__()
        self.policy = policy

    def forward(self, obs, actions, mask):
        return self.policy.score(obs, actions, mask)


def _global_normalize(x: torch.Tensor) -> torch.Tensor:
    # Sufficient statistics make per-minibatch normalization global, not rank-local.
    stats = torch.stack((x.sum(), x.square().sum(), x.new_tensor(x.numel())))
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    mean = stats[0] / stats[2]
    var = (stats[1] / stats[2] - mean.square()).clamp_min(0.)
    return (x - mean) / (var.sqrt() + 1e-8)


def _gae(rewards, dones, truncs, next_values, values, gamma, gae_lambda):
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        nonterminal = (~dones[t]).float()
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * (~truncs[t]).float() * gae
        adv[t] = gae
    return adv, adv + values


def _validate_deferred_next_value_parity():
    """Prove next-value reuse preserves GAE, including done and truncation."""
    device = torch.device('cpu')
    steps, n = 7, 9
    generator = torch.Generator().manual_seed(390271)
    rewards = torch.randn((steps, n), generator=generator)
    values = torch.randn((steps, n), generator=generator)
    dones = torch.zeros((steps, n), dtype=torch.bool)
    truncs = torch.zeros_like(dones)
    dones[2, 1] = True  # true termination inside rollout
    truncs[4, 2] = True  # horizon boundary, requires terminal bootstrap
    # The rollout collector has one explicit horizon-boundary value pass and
    # one final bootstrap pass. Other live next values equal next-step values.
    legacy_next = torch.randn((steps, n), generator=generator)
    for t in range(steps - 1):
        live = ~(dones[t] | truncs[t])
        legacy_next[t] = torch.where(live, values[t + 1], legacy_next[t])
    boundary = torch.zeros(steps, dtype=torch.bool)
    boundary[4] = True
    boundary[-1] = True

    deferred_next = torch.zeros_like(legacy_next)
    for t in range(steps):
        if t:
            ended_prev = dones[t - 1] | truncs[t - 1]
            deferred_next[t - 1] = torch.where(
                ended_prev, deferred_next[t - 1], values[t])
        if boundary[t]:
            deferred_next[t] = torch.where(
                dones[t], torch.zeros_like(legacy_next[t]), legacy_next[t])

    legacy_adv, legacy_returns = _gae(
        rewards, dones, truncs, legacy_next, values, .993, .95)
    deferred_adv, deferred_returns = _gae(
        rewards, dones, truncs, deferred_next, values, .993, .95)
    if not torch.equal(legacy_adv, deferred_adv):
        raise RuntimeError('deferred next-value GAE parity failed')
    if not torch.equal(legacy_returns, deferred_returns):
        raise RuntimeError('deferred next-value return parity failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--envs-per-rank', type=int, default=12288)
    p.add_argument('--rollout-steps', type=int, default=64)
    p.add_argument('--warmup-steps', type=int, default=8,
                   help='untimed strategic decisions to trigger lazy HIP paths')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--minibatch-size', type=int, default=4096,
                   help='global minibatch; must divide evenly across ranks')
    p.add_argument('--seed', type=int, default=94001)
    p.add_argument('--opponent', default='guard')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--task', default='counter_defense')
    p.add_argument('--horizon', type=int, default=8000)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=.993)
    p.add_argument('--gae-lambda', type=float, default=.95)
    p.add_argument('--entropy-coef', type=float, default=.01)
    p.add_argument('--adam-mode', choices=('default', 'foreach', 'fused'),
                   default='default')
    p.add_argument('--gradient-as-bucket-view', action='store_true')
    p.add_argument('--static-graph', action='store_true')
    p.add_argument('--defer-next-values', action='store_true',
                   help='reuse next rollout value; evaluate only horizon/final bootstraps')
    a = p.parse_args()

    _validate_deferred_next_value_parity()

    dist.init_process_group('nccl')  # ROCm's RCCL is exposed via this backend.
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    if world != 2:
        raise ValueError(f'this experiment expects exactly two ranks; got {world}')
    if a.minibatch_size % world:
        raise ValueError('global minibatch must divide evenly across ranks')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    env = _env(a.envs_per_rank, a.task, device, a.seed + rank * 1000003,
               a.opponent, a.horizon, architecture='strategic_adstar')
    obs = _reset_obs(env.reset()).to(device=device, dtype=torch.float32)
    obs_dim, action_dim = int(env.obs_dim), int(env.action_dim)
    policy = ActorCritic(obs_dim, action_dim, 'categorical').to(device)
    if a.checkpoint:
        payload = torch.load(a.checkpoint, map_location=device, weights_only=True)
        if payload.get('architecture') != 'strategic_adstar' or payload.get('obs_dim') != obs_dim:
            raise ValueError('checkpoint architecture/observation dimensions mismatch')
        policy.load_state_dict(payload['model_state_dict'])
    # DDP synchronizes model initialization/checkpoint from rank zero.
    score_model = nn.parallel.DistributedDataParallel(
        ScoreForward(policy), device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=False,
        gradient_as_bucket_view=a.gradient_as_bucket_view,
        static_graph=a.static_graph)
    adam_kwargs = ({'fused': True} if a.adam_mode == 'fused' else
                   {'foreach': True} if a.adam_mode == 'foreach' else {})
    optimizer = torch.optim.Adam(policy.parameters(), lr=a.learning_rate,
                                 eps=1e-5, **adam_kwargs)
    # Diverge action RNG streams after identical initialization.
    torch.manual_seed(a.seed + rank + 1)
    torch.cuda.manual_seed_all(a.seed + rank + 1)

    n, tmax = a.envs_per_rank, a.rollout_steps
    shape = (tmax, n)
    b_obs = torch.empty((*shape, obs_dim), device=device)
    b_actions = torch.empty(shape, device=device, dtype=torch.long)
    b_masks = torch.empty((*shape, action_dim), device=device, dtype=torch.bool)
    b_logp = torch.empty(shape, device=device)
    b_rewards = torch.empty(shape, device=device)
    b_done = torch.empty(shape, device=device, dtype=torch.bool)
    b_trunc = torch.empty(shape, device=device, dtype=torch.bool)
    b_values = torch.empty(shape, device=device)
    b_next = torch.empty(shape, device=device)
    strategic_phase = 0.
    episode_ticks = 0
    next_value_forward_calls = 0
    # The fused simulator paths are loaded lazily on first use. Warm them up
    # before timing so the report reflects steady-state PPO throughput rather
    # than extension loading/compilation on the first transition.
    for _ in range(a.warmup_steps):
        mask = obs[:, -action_dim:].bool()
        with torch.no_grad():
            action = policy.sample(obs, action_mask=mask)[0]
        strategic_phase += 50. / 4.
        ticks = max(1, int(strategic_phase))
        strategic_phase -= ticks
        remaining = a.horizon - episode_ticks
        _, done, trunc, next_obs, _, used_ticks, _, _ = _held_action_interval(
            env, action, ticks, obs, known_remaining_ticks=remaining,
            return_info=False)
        episode_ticks += used_ticks
        if episode_ticks >= a.horizon:
            episode_ticks = 0
        ended = (done.to(device=device, dtype=torch.bool).reshape(n) |
                 trunc.to(device=device, dtype=torch.bool).reshape(n))
        obs = (env.reset_done(ended) if bool(ended.any()) else next_obs)
        obs = obs.to(device=device, dtype=torch.float32)
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    for t in range(tmax):
        b_obs[t] = obs
        mask = obs[:, -action_dim:].bool()
        b_masks[t] = mask
        with torch.no_grad():
            action, logp, value = policy.sample(obs, action_mask=mask)
            if t:
                ended_prev = b_done[t-1] | b_trunc[t-1]
                b_next[t-1] = torch.where(ended_prev, b_next[t-1], value)
        b_actions[t], b_logp[t], b_values[t] = action, logp, value
        strategic_phase += 50. / 4.
        ticks = max(1, int(strategic_phase))
        strategic_phase -= ticks
        remaining = a.horizon - episode_ticks
        reward, done, trunc, next_obs, _, used_ticks, _, _ = _held_action_interval(
            env, action, ticks, obs, known_remaining_ticks=remaining,
            return_info=False)
        episode_ticks += used_ticks
        boundary = episode_ticks >= a.horizon
        if boundary:
            episode_ticks = 0
        reward = reward.to(device=device, dtype=torch.float32).reshape(n)
        done = done.to(device=device, dtype=torch.bool).reshape(n)
        trunc = trunc.to(device=device, dtype=torch.bool).reshape(n)
        b_rewards[t], b_done[t], b_trunc[t] = reward, done, trunc
        # For live rows, the next loop iteration already evaluates this same
        # observation to sample its action and value. Reuse that value there.
        # Horizon rows need the terminal observation's bootstrap before reset;
        # the final rollout step needs its bootstrap as well. True termination
        # is masked from GAE and therefore needs a zero next value.
        need_explicit_value = not a.defer_next_values or boundary or t == tmax - 1
        if need_explicit_value:
            with torch.no_grad():
                next_value = policy(next_obs)[2]
                next_value_forward_calls += 1
                b_next[t] = (torch.where(done, torch.zeros_like(next_value), next_value)
                             if a.defer_next_values else next_value)
        elif a.defer_next_values:
            b_next[t].zero_()
        reset_done = getattr(env, 'reset_done')
        obs = reset_done(done | trunc) if boundary else next_obs
        obs = obs.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        adv, returns = _gae(b_rewards, b_done, b_trunc, b_next, b_values,
                            a.gamma, a.gae_lambda)
    local_size = tmax * n
    global_size = local_size * world
    global_mb, local_mb = a.minibatch_size, a.minibatch_size // world
    if global_size % global_mb:
        raise ValueError(f'global batch {global_size} must be divisible by minibatch {global_mb}')
    batches_per_epoch = global_size // global_mb
    flat_obs = b_obs.reshape(local_size, obs_dim)
    flat_act, flat_mask = b_actions.reshape(-1), b_masks.reshape(local_size, action_dim)
    flat_logp, flat_adv, flat_ret = b_logp.reshape(-1), adv.reshape(-1), returns.reshape(-1)
    # Normalize advantages over the global rollout before epoch shuffling.
    stats = torch.stack((flat_adv.sum(), flat_adv.square().sum(), flat_adv.new_tensor(flat_adv.numel())))
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    gmean = stats[0] / stats[2]
    gstd = (stats[1] / stats[2] - gmean.square()).clamp_min(0.).sqrt()
    flat_adv = (flat_adv - gmean) / (gstd + 1e-8)
    update_start = time.perf_counter()
    for epoch in range(a.epochs):
        if rank == 0:
            # Apply the same local row permutation on both ranks. A global
            # minibatch is then the union of those rows from both independent
            # scenario shards: exactly local_mb rows per rank, with no remap
            # of a sample from one shard onto another rank's local dataset.
            permutation = torch.randperm(local_size, device=device)
        else:
            permutation = torch.empty(local_size, device=device, dtype=torch.long)
        dist.broadcast(permutation, src=0)
        grouped = permutation.reshape(batches_per_epoch, local_mb)
        for step in range(batches_per_epoch):
            local_ids = grouped[step]
            mb_adv = _global_normalize(flat_adv[local_ids])
            new_logp, entropy, value, _logits = score_model(
                flat_obs[local_ids], flat_act[local_ids], flat_mask[local_ids])
            ratio = (new_logp - flat_logp[local_ids]).exp()
            pg = torch.maximum(-mb_adv * ratio,
                -mb_adv * ratio.clamp(1-.2, 1+.2)).mean()
            vloss = .5 * (value - flat_ret[local_ids]).square().mean()
            ent = entropy.mean()
            loss = pg + .5 * vloss - a.entropy_coef * ent + 1e-5 * sum(
                p.square().sum() for p in policy.parameters() if p.ndim > 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    update_elapsed = time.perf_counter() - update_start
    times = torch.tensor([elapsed, update_elapsed], dtype=torch.float64, device=device)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    # DDP must leave every replica identical after the final synchronized step.
    params_flat = torch.cat([p.detach().reshape(-1).double() for p in policy.parameters()])
    low, high = params_flat.clone(), params_flat.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    max_parameter_delta = float((high - low).abs().max().item())
    parameter_parity = max_parameter_delta <= 1e-10
    if not parameter_parity:
        raise RuntimeError(f'DDP parameters diverged across ranks: max delta={max_parameter_delta}')
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=True)
        report = {
            'status': 'completed', 'world_size': world,
            'envs_per_rank': n, 'global_envs': n * world,
            'rollout_steps': tmax, 'global_transitions': global_size,
            'epochs': a.epochs, 'global_minibatch_size': global_mb,
            'local_minibatch_size': local_mb, 'seed': a.seed,
            'adam_mode': a.adam_mode,
            'gradient_as_bucket_view': a.gradient_as_bucket_view,
            'static_graph': a.static_graph,
            'defer_next_values': a.defer_next_values,
            'next_value_forward_calls': next_value_forward_calls,
            'rank_world_seed_offset': 1000003, 'opponent': a.opponent,
            'checkpoint': str(a.checkpoint) if a.checkpoint else None,
            'elapsed_seconds_max_rank': float(times[0]),
            'ppo_update_seconds_max_rank': float(times[1]),
            'transitions_per_second': global_size / float(times[0]),
            'ddp_parameter_parity': parameter_parity,
            'ddp_max_parameter_delta': max_parameter_delta,
            'device_names': [torch.cuda.get_device_name(i) for i in range(world)],
            'scenario_parity': 'rank-offset environment seeds; same seed base but not per-world scenario parity with single-GPU run',
            'note': 'Prototype uses same policy architecture/action mask/held-action env path; no population rotation in this fixed-opponent throughput benchmark.'}
        (a.output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
