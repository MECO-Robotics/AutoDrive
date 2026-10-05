#!/usr/bin/env python3
"""Two-rank synthetic PPO update microbenchmark for ROCm DDP/Adam flags.

Run with torchrun. This isolates optimizer/reducer cost using the strategic
policy shape and the production local minibatch size (2048 examples/rank).
It does not create simulator environments or change optimizer math.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from frc_defense.tensor_training import ActorCritic


class ScoreForward(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, actions, mask):
        return self.policy.score(obs, actions, mask)


def trial(name, device, rank, local_batch, steps, warmup, adam_mode,
          bucket_view, static_graph, seed):
    torch.manual_seed(seed)
    policy = ActorCritic(137, 8, 'categorical').to(device)
    try:
        ddp = nn.parallel.DistributedDataParallel(
            ScoreForward(policy), device_ids=[int(device.index)],
            output_device=int(device.index), broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=bucket_view, static_graph=static_graph)
        optim_kwargs = ({'fused': True} if adam_mode == 'fused' else
                        {'foreach': True} if adam_mode == 'foreach' else {})
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5,
                                     **optim_kwargs)
    except Exception as exc:
        dist.barrier()
        return {'name': name, 'status': 'unsupported', 'error': repr(exc)}

    gen = torch.Generator(device=device).manual_seed(seed + 1009 * (rank + 1))
    obs = torch.randn(local_batch, 137, device=device, generator=gen)
    masks = torch.ones(local_batch, 8, dtype=torch.bool, device=device)
    actions = torch.randint(0, 8, (local_batch,), device=device, generator=gen)
    with torch.no_grad():
        old_logp = policy.score(obs, actions, masks)[0]
    advantages = torch.randn(local_batch, device=device, generator=gen)
    returns = torch.randn(local_batch, device=device, generator=gen)

    def step():
        new_logp, entropy, value, logits = ddp(obs, actions, masks)
        ratio = (new_logp - old_logp).exp()
        pg = torch.maximum(-advantages * ratio,
                           -advantages * ratio.clamp(.8, 1.2)).mean()
        vloss = .5 * (value - returns).square().mean()
        loss = pg + .5 * vloss - .01 * entropy.mean() + 1e-5 * sum(
            p.square().sum() for p in policy.parameters() if p.ndim > 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), .5)
        optimizer.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    dist.barrier()
    started = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    max_elapsed = torch.tensor([elapsed], device=device, dtype=torch.float64)
    dist.all_reduce(max_elapsed, op=dist.ReduceOp.MAX)
    flat = torch.cat([p.detach().reshape(-1).double() for p in policy.parameters()])
    low, high = flat.clone(), flat.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    delta = float((high - low).abs().max().item())
    return {'name': name, 'status': 'completed', 'adam_mode': adam_mode,
            'gradient_as_bucket_view': bucket_view, 'static_graph': static_graph,
            'local_minibatch_size': local_batch,
            'global_minibatch_size': local_batch * dist.get_world_size(),
            'timed_steps': steps, 'warmup_steps': warmup,
            'elapsed_seconds_max_rank': float(max_elapsed.item()),
            'milliseconds_per_global_minibatch': float(max_elapsed.item()) * 1e3 / steps,
            'global_minibatches_per_second': steps / float(max_elapsed.item()),
            'parameter_max_delta': delta, 'parameter_parity': delta <= 1e-10}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--local-minibatch', type=int, default=2048)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--warmup', type=int, default=16)
    p.add_argument('--seed', type=int, default=77101)
    a = p.parse_args()
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise ValueError('requires two ranks')
    device = torch.device('cuda', int(__import__('os').environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    cases = [
        ('baseline_a', 'default', False, False),
        ('bucket_view_a', 'default', True, False),
        ('static_graph', 'default', False, True),
        ('bucket_static', 'default', True, True),
        ('adam_fused', 'fused', False, False),
        ('baseline_b', 'default', False, False),
    ]
    reports = []
    for i, (name, adam, bucket, static) in enumerate(cases):
        result = trial(name, device, rank, a.local_minibatch, a.steps,
                       a.warmup, adam, bucket, static, a.seed)
        reports.append(result)
        if rank == 0:
            print(json.dumps(result), flush=True)
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=True)
        baseline = next(x for x in reports if x['name'] == 'baseline_a')
        for item in reports:
            if item.get('status') == 'completed':
                item['speedup_vs_baseline'] = (baseline['milliseconds_per_global_minibatch'] /
                                               item['milliseconds_per_global_minibatch'])
        report = {'status': 'completed', 'world_size': 2,
                  'devices': [torch.cuda.get_device_name(i) for i in range(2)],
                  'torch': torch.__version__, 'hip': torch.version.hip,
                  'global_minibatch_size': a.local_minibatch * 2,
                  'model': 'strategic categorical 137x128x128x8',
                  'cases': reports,
                  'note': 'Synthetic update-only workload; no environment or rollout/GAE time. PPO loss, minibatch shape and synchronization path match the benchmark implementation.'}
        (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
