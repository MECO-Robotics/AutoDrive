#!/usr/bin/env python3
"""Exact raw-feature parity harness for the isolated strategic-observation HIP proto.

Run after the active GPU campaign releases the selected GPU:
  .venv/bin/python scripts/test_strategic_observation_proto.py --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense.tensor_strategic_observation_proto import (
    strategic_observation, torch_history_from_raw)


def _state_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _pending_clone(pending):
    if pending is None:
        return None
    data, mask = pending
    return tuple(x.clone() for x in data), mask.clone()


def _assert_pending_equal(lhs, rhs, label):
    if (lhs is None) != (rhs is None):
        raise AssertionError(f"{label}: pending cache presence differs")
    if lhs is None:
        return
    for i, (a, b) in enumerate(zip(lhs[0], rhs[0])):
        if not torch.equal(a, b):
            raise AssertionError(f"{label}: pending candidate tensor {i} differs")
    if not torch.equal(lhs[1], rhs[1]):
        raise AssertionError(f"{label}: pending action mask differs")


def _run_case(task, focal, empty_candidates, mixed_active, seed, device, envs,
             fuse_candidate_local=False):
    env = TensorDefenseEnv(num_envs=envs, task=task, device=device, seed=seed,
        action_mode="strategic", randomize=True, perception_config={
            "detection_dropout": .07, "position_noise_m": .025,
            "velocity_noise_mps": .08})
    if empty_candidates:
        env._track_mask[:, focal].zero_()
    else:
        # Provide known, genuinely valid perceived FUEL for the ordinary
        # cases. Distinct offsets avoid ties in the four ranked candidates.
        env._track_mask[:, focal].zero_()
        origin = env.sim.pose[:, focal, :2]
        offsets = torch.tensor(((.19, .07), (.43, .16), (.71, .31), (1.03, .52)),
                               device=env.device, dtype=env.sim.pose.dtype)
        env._track_pos[:, focal, :4] = origin[:, None, :] + offsets[None, :, :]
        env._track_vel[:, focal, :4] = torch.tensor((.11, -.07), device=env.device,
            dtype=env.sim.pose.dtype)
        env._track_mask[:, focal, :4] = True
    # Exercise a nontrivial circular history/index/delay state. The prototype
    # must stay below this layer and neither inspect nor mutate these buffers.
    env.observation_history.normal_(generator=env.generator)
    env._observation_history_index.copy_(torch.randint(
        env.observation_history.shape[1], (env.n,), device=env.device,
        generator=env.generator))
    env.observation_delay.copy_(torch.randint(
        env.max_observation_latency_steps + 1, (env.n,), device=env.device,
        generator=env.generator))
    history_before = env.observation_history.clone()
    index_before = env._observation_history_index.clone()
    delay_before = env.observation_delay.clone()
    rng_before = env.generator.get_state().clone()
    if mixed_active:
        active = (torch.arange(env.n, device=env.device) % 3 != 1)
    else:
        active = torch.ones(env.n, device=env.device, dtype=torch.bool)
    # Seed a distinct prior cache so inactive rows must retain their previous
    # per-world candidate data when the active mask is mixed.
    prior = None
    if focal == 0 and env.reuse_strategic_own_candidates:
        fake = (torch.full((env.n, 4), 17, device=env.device, dtype=torch.long),
                torch.zeros((env.n, 4), device=env.device, dtype=torch.bool),
                torch.full((env.n, 4), -3., device=env.device))
        prior = (fake, torch.zeros((env.n, 8), device=env.device, dtype=torch.bool))
        env._pending_strategic_own_candidates = _pending_clone(prior)

    reference, ref_candidates, ref_mask = env._strategic_observation(
        focal, active_mask=active, return_candidate_data=True)
    ref_pending = _pending_clone(env._pending_strategic_own_candidates)
    # Rewind the only side effect the reference call is allowed to make before
    # running the prototype against identical tensors.
    env._pending_strategic_own_candidates = _pending_clone(prior)
    history_hash_before = _state_hash(history_before)
    prototype, proto_candidates, proto_mask = strategic_observation(
        env, focal, active_mask=active, return_candidate_data=True,
        fuse_candidate_local=fuse_candidate_local)
    if not torch.equal(reference, prototype):
        diff = (reference - prototype).abs()
        rows, cols = torch.nonzero(reference != prototype, as_tuple=True)
        samples = [(int(r.item()), int(c.item()), float(reference[r, c].item()),
                    float(prototype[r, c].item()))
                   for r, c in zip(rows[:12], cols[:12])]
        raise AssertionError(
            f"raw feature mismatch {task=} {focal=} {empty_candidates=} "
            f"{mixed_active=}: differing={(reference != prototype).sum().item()} "
            f"max_abs={diff.max().item()} samples={samples}")
    for index, (left, right) in enumerate(zip(ref_candidates, proto_candidates)):
        if not torch.equal(left, right):
            if index == 0:
                # torch.topk may choose different indices among invalid entries
                # whose distances are all +inf. Those indices are masked out
                # of the observation/action mask and have no policy semantics.
                unequal = (left != right) & ref_candidates[1]
            else:
                unequal = left != right
            if bool(unequal.any().item()):
                rows, cols = torch.nonzero(unequal, as_tuple=True)
                samples = [(int(r.item()), int(c.item()), left[r, c].item(),
                            right[r, c].item())
                           for r, c in zip(rows[:12], cols[:12])]
                raise AssertionError(f"meaningful candidate output {index} differs in "
                    f"{task=}, {focal=}; count={unequal.sum().item()} samples={samples}; "
                    f"valid_equal={torch.equal(ref_candidates[1], proto_candidates[1])}")
    if not torch.equal(ref_mask, proto_mask):
        raise AssertionError(f"action mask differs in {task=}, {focal=}")
    _assert_pending_equal(ref_pending, env._pending_strategic_own_candidates,
                          f"cache {task=} {focal=} {mixed_active=}")
    if not torch.equal(history_before, env.observation_history):
        raise AssertionError("prototype changed randomized observation history")
    if not torch.equal(index_before, env._observation_history_index):
        raise AssertionError("prototype changed observation history indices")
    if not torch.equal(delay_before, env.observation_delay):
        raise AssertionError("prototype changed observation latency")
    if not torch.equal(rng_before, env.generator.get_state()):
        raise AssertionError("raw feature prototype advanced environment RNG")
    if mixed_active and focal == 0 and env.reuse_strategic_own_candidates:
        for tensor_i, (actual, old) in enumerate(zip(
                env._pending_strategic_own_candidates[0], prior[0])):
            if not torch.equal(actual[~active], old[~active]):
                raise AssertionError(f"inactive pending-cache rows changed, field {tensor_i}")
    # The simulator's delayed history is only for focal robot 0. Verify that
    # the fused raw vector enters the unchanged Torch history, latency,
    # noise, and dropout path identically there. Focal 1 is the role-swapped
    # policy input and deliberately bypasses that single-controller history.
    if focal == 0:
        pre_history = history_before.clone()
        pre_index = index_before.clone()
        pre_rng = rng_before.clone()
        env.observation_history.copy_(pre_history)
        env._observation_history_index.copy_(pre_index)
        env.generator.set_state(pre_rng)
        prototype_obs = torch_history_from_raw(env, prototype,
            active_mask=active, active_count=int(active.sum().item()))
        prototype_history = env.observation_history.clone()
        prototype_index = env._observation_history_index.clone()
        prototype_rng = env.generator.get_state().clone()
        env.observation_history.copy_(pre_history)
        env._observation_history_index.copy_(pre_index)
        env.generator.set_state(pre_rng)
        reference_obs = env._obs(active_mask=active,
                                 active_count=int(active.sum().item()))
        if not torch.equal(reference_obs, prototype_obs):
            raise AssertionError(f"history/noise/dropout output differs in {task=}")
        if not torch.equal(env.observation_history, prototype_history):
            raise AssertionError(f"history writes differ in {task=}")
        if not torch.equal(env._observation_history_index, prototype_index):
            raise AssertionError(f"history index differs in {task=}")
        if not torch.equal(env.generator.get_state(), prototype_rng):
            raise AssertionError(f"noise/dropout RNG consumption differs in {task=}")
    return {"task": task, "focal": focal, "empty_candidates": empty_candidates,
            "mixed_active": mixed_active, "raw_sha256": _state_hash(prototype),
            "candidate_valid_rows": int(proto_candidates[1].any(-1).sum().item()),
            "history_sha256": history_hash_before}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=87123)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fuse-candidate-local", action="store_true",
                        help="also fuse local-track and candidate feature arithmetic")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("This parity harness requires the HIP runtime.")
    results = []
    for task in ("counter_defense", "defense"):
        for focal in (0, 1):
            for empty in (False, True):
                for mixed in (False, True):
                    seed = args.seed + len(results)
                    results.append(_run_case(task, focal, empty, mixed,
                                             seed, args.device, args.envs,
                                             args.fuse_candidate_local))
    report = {"cases": results, "all_exact": True,
              "fuse_candidate_local": args.fuse_candidate_local}
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
