#!/usr/bin/env python3
"""Exact runtime parity test for the production flag-gated HIP observation path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense import tensor_sim
from frc_defense import tensor_strategic_observation_proto as strategic_hip


def _assert_candidates_equal(lhs, rhs, label):
    if not torch.equal(lhs[1], rhs[1]):
        raise AssertionError(f"{label}: candidate validity differs")
    valid = lhs[1]
    for name, a, b in (("indices", lhs[0], rhs[0]),
                       ("nearest", lhs[2], rhs[2])):
        if not torch.equal(a[valid], b[valid]):
            raise AssertionError(f"{label}: valid candidate {name} differs")


def _assert_pending_equal(lhs, rhs, label):
    if (lhs is None) != (rhs is None):
        raise AssertionError(f"{label}: pending cache presence differs")
    if lhs is None:
        return
    _assert_candidates_equal(lhs[0], rhs[0], label)
    if not torch.equal(lhs[1], rhs[1]):
        raise AssertionError(f"{label}: pending action masks differ")


def _new_env(task, seed, device, envs, normalize):
    return tensor_sim.TensorDefenseEnv(num_envs=envs, task=task, device=device,
        seed=seed, action_mode="strategic", randomize=True,
        reuse_strategic_own_candidates=True, normalize_observations=normalize,
        perception_config={"detection_dropout": .07,
            "position_noise_m": .025, "velocity_noise_mps": .08})


def _seed_global(seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_known_tracks(env, focal):
    env._track_mask[:, focal].zero_()
    origin = env.sim.pose[:, focal, :2]
    offsets = torch.tensor(((.19, .07), (.43, .16), (.71, .31), (1.03, .52)),
                           device=env.device, dtype=env.sim.pose.dtype)
    env._track_pos[:, focal, :4] = origin[:, None, :] + offsets[None, :, :]
    env._track_vel[:, focal, :4] = torch.tensor((.11, -.07), device=env.device,
                                                 dtype=env.sim.pose.dtype)
    env._track_mask[:, focal, :4] = True


def _run_case(task, focal, seed, device, envs, normalize, empty_candidates):
    strategic_hip.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = False
    _seed_global(seed, device)
    reference = _new_env(task, seed, device, envs, normalize)
    _set_known_tracks(reference, focal)
    if empty_candidates:
        reference._track_mask[:, focal].zero_()
    reference._pending_strategic_own_candidates = None

    strategic_hip.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = True
    _seed_global(seed, device)
    fused = _new_env(task, seed, device, envs, normalize)
    _set_known_tracks(fused, focal)
    if empty_candidates:
        fused._track_mask[:, focal].zero_()
    fused._pending_strategic_own_candidates = None

    active = (torch.arange(envs, device=device) % 3 != 1)
    strategic_hip.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = False
    raw_ref, candidates_ref, mask_ref = reference._strategic_observation(
        focal, active_mask=active, return_candidate_data=True)
    if focal == 0:
        obs_ref = reference._obs(active_mask=active,
                                active_count=int(active.sum().item()))
    else:
        obs_ref = None
    ref_history = reference.observation_history.clone()
    ref_index = reference._observation_history_index.clone()
    ref_rng = reference.generator.get_state().clone()
    ref_pending = reference._pending_strategic_own_candidates

    strategic_hip.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = True
    raw_fused, candidates_fused, mask_fused = fused._strategic_observation(
        focal, active_mask=active, return_candidate_data=True)
    if focal == 0:
        obs_fused = fused._obs(active_mask=active,
                               active_count=int(active.sum().item()))
    else:
        obs_fused = None
    fused_history = fused.observation_history.clone()
    fused_index = fused._observation_history_index.clone()
    fused_rng = fused.generator.get_state().clone()
    fused_pending = fused._pending_strategic_own_candidates

    if not torch.equal(raw_ref, raw_fused):
        raise AssertionError(f"raw observation mismatch in {task=}, {focal=}, {normalize=}")
    _assert_candidates_equal(candidates_ref, candidates_fused,
                             f"candidates {task=} {focal=}")
    if not torch.equal(mask_ref, mask_fused):
        raise AssertionError(f"action mask mismatch in {task=}, {focal=}")
    if focal == 0 and not torch.equal(obs_ref, obs_fused):
        raise AssertionError(f"history/noise observation mismatch in {task=}")
    if not torch.equal(ref_history, fused_history):
        raise AssertionError(f"observation history differs in {task=}, {focal=}")
    if not torch.equal(ref_index, fused_index):
        raise AssertionError(f"observation history index differs in {task=}, {focal=}")
    if not torch.equal(ref_rng, fused_rng):
        raise AssertionError(f"environment RNG state differs in {task=}, {focal=}")
    _assert_pending_equal(ref_pending, fused_pending,
                          f"pending cache {task=} {focal=}")
    return {"task": task, "focal": focal, "normalize": normalize,
            "empty_candidates": empty_candidates,
            "raw_shape": list(raw_fused.shape),
            "valid_candidate_rows": int(candidates_fused[1].any(-1).sum().item()),
            "all_exact": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=64431)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("This focused runtime parity test requires HIP.")
    if strategic_hip._extension() is None:
        raise SystemExit("Fused strategic observation extension failed to load.")
    cases = []
    for task in ("counter_defense", "defense"):
        for focal in (0, 1):
            for normalize in (False, True):
                for empty in (False, True):
                    cases.append(_run_case(task, focal, args.seed + len(cases),
                        args.device, args.envs, normalize, empty))
    strategic_hip.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = True
    report = {"cases": cases, "all_exact": all(case["all_exact"] for case in cases)}
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
