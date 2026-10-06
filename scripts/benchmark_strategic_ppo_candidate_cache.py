#!/usr/bin/env python3
"""Matched PPO A/B/A throughput and policy-parity test for strategic candidate reuse."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from frc_defense.tensor_training import _train_once


def _weights_equal(left: Path, right: Path) -> bool:
    a = torch.load(left, map_location="cpu", weights_only=True)["model_state_dict"]
    b = torch.load(right, map_location="cpu", weights_only=True)["model_state_dict"]
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def _run_case(args, root: Path, name: str, reuse: bool) -> dict:
    output = root / name
    output.mkdir(parents=True, exist_ok=True)
    transitions = args.num_envs * args.rollout_steps
    result = _train_once(
        task="defense", timesteps=transitions, output=output,
        seed=args.seed, num_envs=args.num_envs, device=torch.device(args.device),
        opponent="offense", initial_checkpoint=args.initial_checkpoint,
        rollout_steps=args.rollout_steps, epochs=args.epochs,
        minibatch_size=args.minibatch_size, learning_rate=3e-4, gamma=.993,
        gae_lambda=.95, clip_coef=.2, value_coef=.5, entropy_coef=.01,
        max_grad_norm=.5, l2_coef=1e-5, horizon=8000,
        architecture="strategic_adstar", curriculum=False,
        strategic_rate_hz=4., capture_training_playback=False,
        reuse_strategic_own_candidates=reuse,
        status_context={"timing_profile": False,
                        "requested_timesteps_total": transitions,
                        "candidate_cache_benchmark": True})
    report = {
        "case": name,
        "reuse_strategic_own_candidates": reuse,
        "completed_timesteps": result["completed_timesteps"],
        "elapsed_seconds": result["elapsed_seconds"],
        "strategic_transitions_per_second": result["strategic_transitions_per_second"],
        "physics_world_ticks_per_second": result["physics_world_ticks_per_second"],
        "mean_return": result.get("mean_return"),
        "entropy": result.get("entropy"),
        "action_distribution": result.get("action_distribution"),
        "checkpoint": str(output / "policy.pt"),
    }
    (root / f"{name}.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=18001)
    args = parser.parse_args()
    if args.num_envs < 1 or args.rollout_steps < 1 or args.epochs < 1:
        parser.error("num-envs, rollout-steps, and epochs must be positive")
    if not args.initial_checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.initial_checkpoint}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "status.json").write_text(json.dumps({
        "status": "running", "started_at": time.time(),
        "num_envs": args.num_envs, "rollout_steps": args.rollout_steps,
        "seed": args.seed, "device": args.device,
    }, indent=2))
    cases = []
    try:
        for name, reuse in (("cache_off_a", False), ("cache_on_b", True),
                            ("cache_off_a_repeat", False)):
            cases.append(_run_case(args, args.output, name, reuse))
        parity = {
            name: _weights_equal(Path(cases[0]["checkpoint"]), Path(case["checkpoint"]))
            for name, case in (("cache_on_b", cases[1]),
                               ("cache_off_a_repeat", cases[2]))
        }
        if not all(parity.values()):
            raise RuntimeError(f"trained policy parity failed: {parity}")
        baseline = (cases[0]["strategic_transitions_per_second"] +
                    cases[2]["strategic_transitions_per_second"]) / 2
        report = {
            "cases": cases,
            "policy_weight_parity": parity,
            "cache_speedup_vs_mean_baseline":
                cases[1]["strategic_transitions_per_second"] / baseline,
            "baseline_repeat_ratio":
                cases[2]["strategic_transitions_per_second"] /
                cases[0]["strategic_transitions_per_second"],
        }
        (args.output / "comparison-report.json").write_text(
            json.dumps(report, indent=2))
        (args.output / "status.json").write_text(json.dumps({
            "status": "completed", "finished_at": time.time(),
        }, indent=2))
    except BaseException as exc:
        (args.output / "status.json").write_text(json.dumps({
            "status": "failed", "finished_at": time.time(), "error": repr(exc),
        }, indent=2))
        raise


if __name__ == "__main__":
    main()
