#!/usr/bin/env python3
"""Isolated test-only full simulator A/B/A for strategic observation fusion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense import tensor_strategic_observation_proto as strategic_observation
from frc_defense import tensor_strategic_observation_full_proto as full_observation


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def collect_tensors(root, prefix="env", seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 3 or id(root) in seen:
        return []
    seen.add(id(root))
    found = []
    if isinstance(root, torch.Tensor):
        return [(prefix, tensor_hash(root))]
    if isinstance(root, torch.Generator):
        return [(prefix + ".generator_state", tensor_hash(root.get_state()))]
    if isinstance(root, dict):
        values = root.items()
    elif isinstance(root, (tuple, list)):
        values = enumerate(root)
    elif hasattr(root, "__dict__"):
        values = vars(root).items()
    else:
        return found
    for key, value in values:
        if isinstance(value, (torch.Tensor, torch.Generator, dict, tuple, list)) or hasattr(value, "__dict__"):
            found.extend(collect_tensors(value, f"{prefix}.{key}", seen, depth + 1))
    return found


def run_case(label, fused, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    strategic_observation.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = fused
    full_observation.FUSED_FULL_STRATEGIC_OBSERVATION_HIP_ENABLED = fused
    env = TensorDefenseEnv(num_envs=args.envs, task="counter_defense", device=args.device,
        seed=args.seed, action_mode="strategic", randomize=True,
        reuse_strategic_own_candidates=True)
    # Alternate deterministic policy actions so masks/candidates and held actions
    # exercise multiple branches while keeping A and B inputs identical.
    actions = torch.arange(args.ticks, device=args.device) % 8
    reward_trace = []
    started = time.perf_counter()
    for tick in range(args.ticks):
        active_mask = ((torch.arange(args.envs, device=args.device) + tick) % 5 != 0)
        _, reward, done, truncated, _ = env.step(actions[tick].expand(args.envs),
                                                 active_mask=active_mask,
                                                 _return_info=False)
        reward_trace.append(reward)
        if bool(done.any().item()) or bool(truncated.any().item()):
            raise RuntimeError("unexpected episode end in short e2e benchmark")
    torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started
    trace_hash = hashlib.sha256(torch.stack(reward_trace).contiguous().cpu().numpy().tobytes()).hexdigest()
    state = collect_tensors(env)
    # torch.topk may select arbitrary indices among invalid candidates at +inf
    # distance. Canonicalize only those masked indices; every valid index and
    # the candidate validity/nearest-distance tensors remain part of the hash.
    pending = env._pending_strategic_own_candidates
    valid_index_hash = None
    if pending is not None:
        data, _ = pending
        semantic_indices = torch.where(data[1], data[0], torch.full_like(data[0], -1))
        valid_index_hash = tensor_hash(semantic_indices)
        state = [(path, valid_index_hash if path == "env._pending_strategic_own_candidates.0.0" else digest)
                 for path, digest in state]
    state.sort(key=lambda pair: pair[0])
    state.extend([("global_torch_cpu_rng", tensor_hash(torch.random.get_rng_state())),
                  ("global_torch_device_rng", tensor_hash(torch.cuda.get_rng_state(args.device)))])
    final_hash = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    reward_sum = float(torch.stack(reward_trace).sum().item())
    return {"case": label, "fused_observation": fused, "elapsed_seconds": elapsed,
            "physics_ticks": args.ticks, "world_ticks_per_second": args.envs * args.ticks / elapsed,
            "reward_sum": reward_sum, "reward_trace_sha256": trace_hash,
            "final_state_sha256": final_hash, "final_state_tensor_hashes": state,
            "_observation_history": env.observation_history.detach().cpu().clone(),
            "valid_candidate_index_sha256": valid_index_hash,
            "hashed_tensor_count": len(state)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--ticks", type=int, default=32)
    parser.add_argument("--warmup-ticks", type=int, default=4)
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="run one Torch/full pair and report first observation-history mismatch")
    parser.add_argument("--seed", type=int, default=193871)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("Requires the HIP runtime")
    # Compile/warm both routes before measuring fresh seeded environments.
    for fused in (False, True):
        strategic_observation.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = fused
        full_observation.FUSED_FULL_STRATEGIC_OBSERVATION_HIP_ENABLED = fused
        warm_env = TensorDefenseEnv(num_envs=args.envs, task="counter_defense",
            device=args.device, seed=args.seed, action_mode="strategic", randomize=True,
            reuse_strategic_own_candidates=True)
        for tick in range(args.warmup_ticks):
            active_mask = ((torch.arange(args.envs, device=args.device) + tick) % 5 != 0)
            warm_env.step(torch.full((args.envs,), tick % 8, device=args.device),
                          active_mask=active_mask,
                          _return_info=False)
        torch.cuda.synchronize(args.device)
    if args.diagnostic_only:
        cases = [run_case("torch_diag", False, args),
                 run_case("full_diag", True, args)]
        ref = cases[0]["_observation_history"]
        got = cases[1]["_observation_history"]
        mismatch = ref != got
        positions = torch.nonzero(mismatch, as_tuple=False)
        details = {"history_exact": not bool(mismatch.any()),
                   "mismatch_count": int(mismatch.sum().item()),
                   "history_shape": list(ref.shape)}
        if positions.numel():
            world, history_slot, feature = (int(v) for v in positions[0].tolist())
            details["first_mismatch"] = {"world": world,
                "history_slot": history_slot, "feature": feature,
                "torch_value": float(ref[world, history_slot, feature]),
                "full_value": float(got[world, history_slot, feature]),
                "abs_error": float((ref[world, history_slot, feature] -
                                     got[world, history_slot, feature]).abs())}
            details["mismatch_features"] = torch.nonzero(
                mismatch.sum((0, 1)), as_tuple=False).reshape(-1).tolist()
        report = {"device": args.device, "envs": args.envs, "ticks": args.ticks,
            "seed": args.seed, "diagnostic": details,
            "torch_elapsed_seconds": cases[0]["elapsed_seconds"],
            "full_elapsed_seconds": cases[1]["elapsed_seconds"],
            "all_reward_trace_hashes_equal":
                cases[0]["reward_trace_sha256"] == cases[1]["reward_trace_sha256"]}
        for case in cases:
            case.pop("_observation_history")
        report["cases"] = cases
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    cases = [run_case("torch_a1", False, args),
             run_case("fused_b1", True, args),
             run_case("torch_a2", False, args),
             run_case("fused_b2", True, args),
             run_case("torch_a3", False, args)]
    a_times = [c["elapsed_seconds"] for c in cases if not c["fused_observation"]]
    b_times = [c["elapsed_seconds"] for c in cases if c["fused_observation"]]
    a_median = statistics.median(a_times)
    b_median = statistics.median(b_times)
    strategic_observation.FUSED_STRATEGIC_OBSERVATION_HIP_ENABLED = True
    report = {"device": args.device, "envs": args.envs, "ticks": args.ticks,
        "warmup_ticks_per_path_unmeasured": args.warmup_ticks,
        "production_flag_path": "opt-in full assembler; A cases Torch fallback, B cases full HIP path",
        "opt_in_flag": "AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_FULL_HIP=1",
        "opt_out_flag": "AUTODRIVE_FUSED_STRATEGIC_OBSERVATION_FULL_HIP=0 (default)",
        "seed": args.seed, "cases": cases,
        "torch_median_seconds": a_median, "fused_median_seconds": b_median,
        "speedup_vs_torch_median": a_median / b_median,
        "torch_repeat_ratio_max_min": max(a_times) / min(a_times),
        "all_final_state_hashes_equal": len({c["final_state_sha256"] for c in cases}) == 1,
        "all_reward_trace_hashes_equal": len({c["reward_trace_sha256"] for c in cases}) == 1,
        "all_reward_sums_equal": len({c["reward_sum"] for c in cases}) == 1}
    history_ref = cases[0]["_observation_history"]
    history_checks = []
    for case in cases[1:]:
        history = case["_observation_history"]
        history_checks.append(bool(torch.equal(history_ref, history)))
    report["all_observation_history_equal"] = all(history_checks)
    report["observation_history_mismatch_count_vs_torch_a1"] = [
        int((history_ref != case["_observation_history"]).sum().item()) for case in cases[1:]]
    for case in cases:
        case.pop("_observation_history")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = dict(report)
    summary["cases"] = [{key: value for key, value in case.items()
                         if key != "final_state_tensor_hashes"} for case in cases]
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not report["all_final_state_hashes_equal"] or not report["all_reward_trace_hashes_equal"]:
        raise SystemExit("A/B/A simulator state/rewards differ")


if __name__ == "__main__":
    main()
