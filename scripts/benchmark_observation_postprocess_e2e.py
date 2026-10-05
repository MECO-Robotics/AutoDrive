#!/usr/bin/env python3
"""Parity-first 2048x10x12 check for _obs postprocessing optimization."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frc_defense import tensor_fuel_candidate_rank as candidate_rank
from frc_defense.tensor_sim import TensorDefenseEnv


def old_reference_obs(self, initialize_mask=None, active_mask=None, active_count=None):
    """Exact pre-optimization history/gather/noise/dropout/where sequence."""
    active_mask = (torch.ones(self.n, device=self.device, dtype=torch.bool)
                   if active_mask is None else
                   torch.as_tensor(active_mask, device=self.device, dtype=torch.bool))
    raw = self._raw_obs(active_mask=active_mask)
    history_length = self.observation_history.shape[1]
    if initialize_mask is None:
        next_index = (self._observation_history_index + 1).remainder(history_length)
        write_index = torch.where(active_mask, next_index, self._observation_history_index)
        current = self.observation_history[self._observation_world_index, write_index]
        updated = torch.where(active_mask[:, None], raw, current)
        self.observation_history[self._observation_world_index, write_index] = updated
        self._observation_history_index.copy_(write_index)
    else:
        initialize_mask = torch.as_tensor(initialize_mask, device=self.device, dtype=torch.bool)
        fresh = raw[:, None, :].expand(-1, self.observation_history.shape[1], -1)
        self.observation_history.copy_(torch.where(
            initialize_mask[:, None, None], fresh, self.observation_history))
        self._observation_history_index.copy_(torch.where(
            initialize_mask, torch.zeros_like(self._observation_history_index),
            self._observation_history_index))
    read_index = (self._observation_history_index - self.observation_delay).remainder(history_length)
    obs = self.observation_history.gather(
        1, read_index[:, None, None].expand(-1, 1, self.obs_dim)).squeeze(1)
    if self.randomize and self.observation_noise > 0:
        state = obs[:, :18] + self._random_active(
            active_mask, (18,), normal=True, active_count=active_count) * self.observation_noise
        obs = torch.cat((state, obs[:, 18:]), -1)
    if self.randomize and self.observation_dropout > 0:
        keep = self._random_active(active_mask, (18,), active_count=active_count) >= self.observation_dropout
        obs = torch.cat((obs[:, :18] * keep, obs[:, 18:]), -1)
    return torch.where(active_mask[:, None], obs, self._last_observation)


def load_scaling_benchmark():
    path = ROOT / "scripts" / "benchmark_tensor_scaling.py"
    spec = importlib.util.spec_from_file_location("obs_postprocess_scaling_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _update_hash(hasher, item):
    if torch.is_tensor(item):
        hasher.update(b"tensor")
        hasher.update(str((tuple(item.shape), str(item.dtype))).encode())
        hasher.update(item.detach().contiguous().cpu().numpy().tobytes())
    elif isinstance(item, dict):
        hasher.update(b"dict")
        for key in sorted(item):
            hasher.update(str(key).encode())
            _update_hash(hasher, item[key])
    elif isinstance(item, (tuple, list)):
        hasher.update(type(item).__name__.encode())
        for value in item:
            _update_hash(hasher, value)
    else:
        hasher.update(type(item).__name__.encode())
        hasher.update(repr(item).encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--decisions", type=int, default=10)
    parser.add_argument("--physics-ticks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=90217)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.envs, args.decisions, args.physics_ticks) != (2048, 10, 12):
        raise ValueError("validation is specified for exactly 2048 x 10 x 12")
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("validation requires HIP")

    scaling = load_scaling_benchmark()
    candidate_rank.FUSED_FUEL_CANDIDATE_RANK_HIP_ENABLED = True
    production_obs = TensorDefenseEnv._obs
    original_step = TensorDefenseEnv.step
    capture = None

    def observation_step(self, *positional, **kwargs):
        result = original_step(self, *positional, **kwargs)
        if capture is not None:
            info = result[4]
            capture["calls"] += 1
            capture["key_sets"].append(sorted(info) if isinstance(info, dict) else None)
            if capture["hash_info"]:
                _update_hash(capture["digest"], info)
        return result

    TensorDefenseEnv.step = observation_step

    def run(label, use_old_obs, return_info, *, collect_info=False):
        nonlocal capture
        TensorDefenseEnv._obs = old_reference_obs if use_old_obs else production_obs
        capture = ({"calls": 0, "key_sets": [], "hash_info": return_info,
                    "digest": hashlib.sha256()} if collect_info else None)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        result = scaling.benchmark(args.envs, torch.device(args.device), args.seed,
            args.decisions, args.physics_ticks, "guard", return_info=return_info,
            skip_metric_objective=True, reuse_opponent_candidates=True,
            opponent_policy_decision_each_tick=False, profile_components=False,
            squared_fuel_candidate_distance=False, reuse_own_candidates=True)
        result["case"] = label
        result["old_reference_obs"] = use_old_obs
        result["return_info"] = return_info
        if capture is not None:
            result["info_calls"] = capture["calls"]
            result["info_keys_first_last"] = (capture["key_sets"][0], capture["key_sets"][-1])
            result["info_payload_sha256"] = (capture["digest"].hexdigest()
                                               if return_info else None)
            result["no_info_payloads_empty"] = (not return_info and
                all(not keys for keys in capture["key_sets"]))
        capture = None
        return result

    # Gate both return modes on full state, reward, and RNG parity. The info
    # mode additionally hashes each dictionary returned by env.step.
    parity = [run("production_no_info", False, False, collect_info=True),
              run("reference_no_info", True, False, collect_info=True),
              run("production_with_info", False, True, collect_info=True),
              run("reference_with_info", True, True, collect_info=True)]
    no_info_pair = parity[:2]
    info_pair = parity[2:]

    def equal_pair(pair):
        return (pair[0]["final_state_sha256"] == pair[1]["final_state_sha256"] and
                pair[0]["reward_trace_sha256"] == pair[1]["reward_trace_sha256"] and
                pair[0]["reward_total"] == pair[1]["reward_total"])

    parity_ok = (equal_pair(no_info_pair) and equal_pair(info_pair) and
        no_info_pair[0]["no_info_payloads_empty"] and no_info_pair[1]["no_info_payloads_empty"] and
        info_pair[0]["info_payload_sha256"] == info_pair[1]["info_payload_sha256"] and
        info_pair[0]["info_keys_first_last"] == info_pair[1]["info_keys_first_last"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parity_path = args.output.with_name(args.output.stem + "-parity.json")
    parity_record = {"device": args.device, "envs": args.envs,
        "decisions": args.decisions, "physics_ticks_per_decision": args.physics_ticks,
        "seed": args.seed, "candidate_rank_flag_on_for_all_cases": True,
        "no_info_parity": equal_pair(no_info_pair), "return_info_parity": equal_pair(info_pair),
        "return_info_payload_equal": (
            info_pair[0]["info_payload_sha256"] == info_pair[1]["info_payload_sha256"]),
        "no_info_payloads_empty": all(item["no_info_payloads_empty"] for item in no_info_pair),
        "parity_pass": parity_ok,
        "cases": [{k: v for k, v in item.items() if k not in (
            "component_host_seconds_nested", "component_host_seconds_inclusive", "component_calls")}
            for item in parity]}
    parity_path.write_text(json.dumps(parity_record, indent=2, sort_keys=True) + "\n")
    if not parity_ok:
        print(json.dumps(parity_record, indent=2, sort_keys=True))
        raise SystemExit("parity gate failed; timing sequence aborted")

    # Benchmark the production no-info path. Remove the info recorder so it
    # cannot affect the timings; compare against the exact pre-optimization _obs.
    timed = [run("production_a1", False, False),
             run("reference_b", True, False),
             run("production_a2", False, False)]
    baseline = [timed[0]["elapsed_seconds"], timed[2]["elapsed_seconds"]]
    old = [timed[1]["elapsed_seconds"]]
    report = {"device": args.device, "envs": args.envs, "decisions": args.decisions,
        "physics_ticks_per_decision": args.physics_ticks, "seed": args.seed,
        "candidate_rank_flag_on_for_all_cases": True,
        "compared_paths": "production in-place _obs vs exact pre-optimization test reference",
        "parity_gate_artifact": str(parity_path), "parity_pass": True,
        "sequence": [item["case"] for item in timed], "cases": timed,
        "production_median_seconds": statistics.median(baseline),
        "old_reference_median_seconds": statistics.median(old),
        "speedup": statistics.median(old) / statistics.median(baseline),
        "production_repeat_relative_spread": abs(baseline[1] - baseline[0]) / statistics.median(baseline),
        "gain_outside_production_repeat_spread": (
            statistics.median(old) - statistics.median(baseline) > abs(baseline[1] - baseline[0])),
        "all_timed_final_states_equal": len({item["final_state_sha256"] for item in timed}) == 1,
        "all_timed_reward_traces_equal": len({item["reward_trace_sha256"] for item in timed}) == 1}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    TensorDefenseEnv._obs = production_obs
    TensorDefenseEnv.step = original_step
    if (not report["all_timed_final_states_equal"] or
            not report["all_timed_reward_traces_equal"]):
        raise SystemExit("timed A/B/A state or reward mismatch")


if __name__ == "__main__":
    main()
