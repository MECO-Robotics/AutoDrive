"""Strict matched-prefix counterfactual labels for MECO Phase 2.

This collector branches a frozen diagnostic teacher trajectory at archived
Plan A/Plan C descriptors. It emits simulator returns, never expert labels.
Replay parity failures abort collection; no nearest or approximate match is
accepted. It is intentionally a separate v2 schema from v1 behavior rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
import sys

import torch

from .curriculum_scenarios import make_curriculum_env, teacher_actions
from .meco_candidates import (CANDIDATE_BOUNDARY,
    ROUTE_COST_POINTS_PER_SECOND, build_route_feasible_candidates)
from .meco_plan_a_policy import (SUPPORT_ORDER, build_plan_a_support,
    plan_a_support_digest)
from .meco_pose_graph import build_legal_pose_graph
from .meco_tracks import (PREDICTION_MODES, build_safe_action_candidates,
    candidate_digest, prediction_contract)
from .training_curriculum import get_scenario

DATASET_SCHEMA = "meco-counterfactual-trajectory-v2"
TARGET_SCHEMA = "meco-matched-prefix-gross-return-v2"
DEFAULT_GAMMA = 0.993
CONTINUATION_POLICY = "curriculum_scenarios.teacher_actions; deterministic sensed-state policy"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode()).hexdigest()


def _tensor_digest(tensor):
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _simulator_state_digest(env):
    """Hash every direct simulator tensor and scalar that defines physics state."""
    state = {}
    for name, value in sorted(vars(env.sim).items()):
        if isinstance(value, torch.Tensor):
            state[name] = _tensor_digest(value)
        elif value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not math.isfinite(value):
                state[name] = repr(value)
            else:
                state[name] = value
    if not state:
        raise RuntimeError("simulator exposes no hashable state; refusing weak replay parity")
    return _hash(state)


def _observation_digest(observation):
    if observation is None:
        raise RuntimeError("simulator has no current observation")
    return _tensor_digest(observation)


def _device_identity(device):
    resolved = torch.device(device)
    identity = {"torch": torch.__version__, "python": sys.version,
                "device": str(resolved), "platform": platform.platform()}
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        identity.update(device_index=index,
                        device_name=torch.cuda.get_device_name(index),
                        capability=list(torch.cuda.get_device_capability(index)))
    return identity


def _reward_step(env, action, action_ticks, goal_xy=None, goal_heading=None,
                 allow_action7_traversal=False):
    from .meco_track_training import _run_macro_step
    values = _run_macro_step(env, action, action_ticks, goal_xy,
        goal_heading=goal_heading,
        allow_action7_traversal=allow_action7_traversal)
    reward, ended, timeout = values[:3]
    return (float(reward.detach().cpu().item()), bool(ended), bool(timeout),
            list(values[3]), values[4], values[5], bool(values[7]))


def _prefix_snapshot(env, graph, episode_id, *, action_ticks, candidate_k,
                     prediction_mode, max_c_candidates, max_a_pairs):
    from .meco_track_training import (_candidate_replay_digest, _route_cost_tensor)
    safe = build_safe_action_candidates(env, env._last_obs, world=0, slot=3,
                                        episode_id=episode_id)
    observation = safe.observation
    candidates = build_route_feasible_candidates(env, env._last_obs, graph,
        episode_id=episode_id, action_ticks=action_ticks,
        candidate_k=candidate_k, prediction_mode=prediction_mode)
    route = _route_cost_tensor(candidates, ROUTE_COST_POINTS_PER_SECOND,
                               device=env.device)
    risk = candidates.risk_cost_points.unsqueeze(0)
    candidate_hash = _candidate_replay_digest(candidates.table, route, risk)
    support = build_plan_a_support(env, graph, env._last_obs,
                                   observation.snapshot_id)
    support_hash = plan_a_support_digest(support, graph)
    if support_hash != support.support_digest:
        raise RuntimeError("Plan A support digest failed self-validation")

    valid_c = [index for index, valid in enumerate(candidates.table.valid_mask) if valid]
    valid_c = _deterministic_sample(valid_c, max_c_candidates,
        f"C:{episode_id}:{int(env.steps[0].item())}")
    c_descriptors = []
    for index in valid_c:
        record = candidates.route_records[index]
        c_descriptors.append({"candidate_index": index,
            "candidate_id": candidates.table.candidate_ids[index],
            "action": int(candidates.action_indices[index]),
            "goal_xy_m": list(record["goal_xy_m"]),
            "descriptor_sha256": _hash({"candidate_id": candidates.table.candidate_ids[index],
                "action": int(candidates.action_indices[index]),
                "goal_xy_m": record["goal_xy_m"],
                "route_record": record})})

    valid_a = torch.nonzero(support.valid_mask, as_tuple=False).reshape(-1).tolist()
    valid_a = _deterministic_sample(valid_a, max_a_pairs,
        f"A:{episode_id}:{int(env.steps[0].item())}")
    if len(valid_c) < 2 or len(valid_a) < 2:
        raise RuntimeError("counterfactual branch needs at least two valid C and Plan A descriptors")
    a_descriptors = []
    for pair in valid_a:
        node = int(support.pair_node_ids[pair].item())
        item = {"pair_index": pair, "node_id": node,
            "behavior_id": int(support.behavior_ids[pair].item()),
            "descriptor_id": int(support.descriptor_ids[pair].item()),
            "action": int(support.action_ids[pair].item()),
            "goal_xy_m": [float(value) for value in support.goal_xy[node].tolist()],
            "goal_heading_rad": float(support.goal_heading[node].item())}
        item["descriptor_sha256"] = _hash(item)
        a_descriptors.append(item)
    return {"snapshot_id": observation.snapshot_id,
        "simulator_state_sha256": _simulator_state_digest(env),
        "observation_sha256": _observation_digest(env._last_obs),
        "safe_observation_sha256": _hash({"snapshot_id": observation.snapshot_id,
            "raster": list(observation.raster), "globals": list(observation.globals)}),
        "candidate_digest": candidate_digest(candidates.table),
        "candidate_replay_digest": candidate_hash,
        "support_digest": support_hash,
        "teacher_action": int(teacher_actions(env, env._last_obs)[0, 3].item()),
        "candidate_descriptors": c_descriptors,
        "plan_a_descriptors": a_descriptors}


def _deterministic_sample(indices, limit, key):
    if limit < 1:
        return []
    return sorted(indices, key=lambda index: hashlib.sha256(
        f"{key}:{index}".encode()).digest())[:limit]


def _rebuild_and_check(env, graph, episode_id, expected, *, action_ticks,
                       candidate_k, prediction_mode, max_c_candidates,
                       max_a_pairs):
    actual = _prefix_snapshot(env, graph, episode_id, action_ticks=action_ticks,
        candidate_k=candidate_k, prediction_mode=prediction_mode,
        max_c_candidates=max_c_candidates, max_a_pairs=max_a_pairs)
    for field in ("snapshot_id", "simulator_state_sha256", "observation_sha256",
                  "safe_observation_sha256", "candidate_digest",
                  "candidate_replay_digest", "support_digest",
                  "candidate_descriptors", "plan_a_descriptors"):
        if actual[field] != expected[field]:
            raise RuntimeError(f"strict prefix replay parity mismatch at {field}")
    return actual


def _branch_return(*, scenario_id, category, seed, device, horizon_ticks,
                   action_ticks, decision_index, prefix, descriptor, kind,
                   candidate_k, prediction_mode, max_c_candidates,
                   max_a_pairs, gamma):
    from .meco_track_training import _reset_episode
    env = make_curriculum_env(scenario_id, category, 1, device, seed,
                              horizon=horizon_ticks)
    try:
        graph = build_legal_pose_graph(env, slot=3)
        episode_id = f"counterfactual-{scenario_id}-{category}-{seed}"
        _reset_episode(env, seed)
        for row in prefix[:decision_index]:
            current = _rebuild_and_check(env, graph, episode_id, row,
                action_ticks=action_ticks, candidate_k=candidate_k,
                prediction_mode=prediction_mode,
                max_c_candidates=max_c_candidates, max_a_pairs=max_a_pairs)
            if current["teacher_action"] != row["teacher_action"]:
                raise RuntimeError("teacher prefix action changed during replay")
            _, ended, timeout, trace, _, _, _ = _reward_step(env,
                row["teacher_action"], action_ticks,
                goal_xy=row["prefix_goal_xy_m"],
                goal_heading=row["prefix_goal_heading_rad"],
                allow_action7_traversal=row["prefix_action7_traversal"])
            if trace != row["executed_action_trace"]:
                raise RuntimeError("executed action trace changed during prefix replay")
            if ended or timeout:
                raise RuntimeError("prefix ended before its archived branch point")
        current = _rebuild_and_check(env, graph, episode_id, prefix[decision_index],
            action_ticks=action_ticks, candidate_k=candidate_k,
            prediction_mode=prediction_mode, max_c_candidates=max_c_candidates,
            max_a_pairs=max_a_pairs)
        if kind == "C":
            choices = current["candidate_descriptors"]
            match = next((item for item in choices
                          if item["descriptor_sha256"] == descriptor["descriptor_sha256"]), None)
            if match is None:
                raise RuntimeError("archived Plan C descriptor is absent after strict replay")
            reward, ended, timeout, _, _, _, _ = _reward_step(env, descriptor["action"],
                action_ticks, goal_xy=descriptor["goal_xy_m"])
        else:
            match = next((item for item in current["plan_a_descriptors"]
                          if item["descriptor_sha256"] == descriptor["descriptor_sha256"]), None)
            if match is None:
                raise RuntimeError("archived Plan A descriptor is absent after strict replay")
            reward, ended, timeout, _, _, _, _ = _reward_step(env, descriptor["action"],
                action_ticks, goal_xy=descriptor["goal_xy_m"],
                goal_heading=descriptor["goal_heading_rad"],
                allow_action7_traversal=(descriptor["action"] == 7))
        discounted = reward
        total_macro_return = reward
        ticks = action_ticks
        remaining = max(0, horizon_ticks - (decision_index + 1) * action_ticks)
        while not (ended or timeout) and decision_index * action_ticks + ticks < horizon_ticks:
            action = int(teacher_actions(env, env._last_obs)[0, 3].item())
            step_reward, ended, timeout, _, _, _, _ = _reward_step(env, action, action_ticks)
            total_macro_return += step_reward
            discounted += (gamma ** (ticks / action_ticks)) * step_reward
            ticks += action_ticks
        return {"gross_macro_return": reward,
            "gross_horizon_return": total_macro_return,
            "discounted_horizon_return": discounted,
            "continuation_policy": CONTINUATION_POLICY,
            "continuation_macro_horizon": max(0, math.ceil(remaining / action_ticks)),
            "terminal": bool(ended or timeout)}
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def collect_counterfactual_v2(output: Path, *, scenario_id: str, category: str,
        seeds: list[int], qualification_protocol: Path, device="cpu",
        horizon_ticks=160, action_ticks=12, candidate_k=16,
        prediction_mode="cv", max_c_candidates=8, max_a_pairs=8,
        gamma=DEFAULT_GAMMA):
    """Collect matched-prefix v2 labels; any parity mismatch aborts the file."""
    from .meco_supervised import _load_protocol, _protocol_train_seeds
    from .meco_track_training import _reset_episode
    from .curriculum_training import source_identity

    protocol, protocol_id, protocol_sha, protocol_path = _load_protocol(
        Path(qualification_protocol))
    if prediction_mode not in PREDICTION_MODES:
        raise ValueError("invalid prediction mode")
    definition = get_scenario(scenario_id)
    if category not in definition.required_categories:
        raise ValueError("category is not registered for scenario")
    if (not seeds or len(set(seeds)) != len(seeds) or
            any(type(seed) is not int or seed < 0 for seed in seeds) or
            not set(seeds) <= _protocol_train_seeds(protocol, scenario_id, category)):
        raise ValueError("all seeds must be unique members of the frozen train split")
    if (type(horizon_ticks) is not int or type(action_ticks) is not int or
            action_ticks < 1 or horizon_ticks < action_ticks or
            type(candidate_k) is not int or candidate_k not in (8, 16, 32) or
            type(max_c_candidates) is not int or max_c_candidates < 2 or
            type(max_a_pairs) is not int or max_a_pairs < 2 or
            not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or
            not 0 < gamma <= 1):
        raise ValueError("invalid counterfactual horizon, sample limits, or discount")
    sources = source_identity()
    device_metadata = _device_identity(device)
    manifest = {"schema": DATASET_SCHEMA, "target_schema": TARGET_SCHEMA,
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "qualification_protocol_file": protocol_path.name,
        "split": "train", "scenario_id": scenario_id, "category": category,
        "seeds": list(seeds), "sources": sources,
        "source_identity_sha256": _hash(sources), "device": device_metadata,
        "prediction_mode": prediction_mode,
        "prediction_contract": prediction_contract(prediction_mode),
        "candidate_boundary": CANDIDATE_BOUNDARY,
        "route_cost_points_per_second": ROUTE_COST_POINTS_PER_SECOND,
        "plan_a_support_order": SUPPORT_ORDER,
        "horizon_physics_ticks": horizon_ticks, "action_ticks": action_ticks,
        "gamma_per_physics_tick": gamma,
        "discount_per_macro": gamma ** action_ticks,
        "continuation_policy": CONTINUATION_POLICY,
        "max_plan_c_descriptors_per_prefix": max_c_candidates,
        "max_plan_a_pairs_per_prefix": max_a_pairs,
        "plan_c_target_semantics": "gross simulator reward; exact archived candidate; same deterministic teacher continuation",
        "plan_a_target_semantics": "gross simulator reward per sampled valid node_behavior_descriptor; no P route-cost label",
        "plan_a_p_target": None,
        "plan_a_p_prior": 0.0,
        "plan_a_p_unlabelled_reason": "simulator executor does not execute P-modified routes",
        "source_runtime_device_bound": True}
    rows = []
    for seed in seeds:
        env = make_curriculum_env(scenario_id, category, 1, device, seed,
                                  horizon=horizon_ticks)
        try:
            graph = build_legal_pose_graph(env, slot=3)
            episode_id = f"counterfactual-{scenario_id}-{category}-{seed}"
            _reset_episode(env, seed)
            prefix = []
            max_decisions = max(1, horizon_ticks // action_ticks)
            for _ in range(max_decisions):
                item = _prefix_snapshot(env, graph, episode_id,
                    action_ticks=action_ticks, candidate_k=candidate_k,
                    prediction_mode=prediction_mode,
                    max_c_candidates=max_c_candidates,
                    max_a_pairs=max_a_pairs)
                prefix.append(item)
                reward, ended, timeout, trace, heading_error, goal_distance, preempted = _reward_step(
                    env, item["teacher_action"], action_ticks)
                item.update(prefix_goal_xy_m=None, prefix_goal_heading_rad=None,
                    prefix_action7_traversal=False, executed_action_trace=trace,
                    prefix_gross_return=reward,
                    prefix_terminal_heading_error=heading_error,
                    prefix_terminal_goal_distance_m=goal_distance,
                    prefix_native_goal_preempted=preempted,
                    prefix_candidate_commitment=None)
                if ended or timeout:
                    break
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        for decision_index, item in enumerate(prefix):
            group = {"schema": DATASET_SCHEMA, "target_schema": TARGET_SCHEMA,
                "scenario_id": scenario_id, "category": category, "seed": seed,
                "decision_index": decision_index,
                "prefix_sha256": _hash(prefix[:decision_index + 1]),
                "protocol_id": protocol_id,
                "qualification_protocol_sha256": protocol_sha,
                "source_identity_sha256": manifest["source_identity_sha256"],
                "device": device_metadata,
                "predecision_simulator_state_sha256": item["simulator_state_sha256"],
                "predecision_observation_sha256": item["observation_sha256"],
                "candidate_digest": item["candidate_digest"],
                "candidate_replay_digest": item["candidate_replay_digest"],
                "support_digest": item["support_digest"],
                "action_ticks": action_ticks, "horizon_physics_ticks": horizon_ticks,
                "gamma_per_physics_tick": gamma,
                "discount_per_macro": gamma ** action_ticks,
                "prefix_transcript": [{
                    "predecision_simulator_state_sha256": row["simulator_state_sha256"],
                    "predecision_observation_sha256": row["observation_sha256"],
                    "candidate_digest": row["candidate_digest"],
                    "candidate_replay_digest": row["candidate_replay_digest"],
                    "support_digest": row["support_digest"],
                    "snapshot_id": row["snapshot_id"],
                    "action": row["teacher_action"],
                    "goal_xy_m": row["prefix_goal_xy_m"],
                    "goal_heading_rad": row["prefix_goal_heading_rad"],
                    "allow_action7_traversal": row["prefix_action7_traversal"],
                    "candidate_commitment": row["prefix_candidate_commitment"],
                    "executed_action_trace": row["executed_action_trace"],
                    "gross_macro_return": row["prefix_gross_return"]}
                    for row in prefix[:decision_index + 1]]}
            c_outcomes = []
            for descriptor in item["candidate_descriptors"]:
                outcome = _branch_return(scenario_id=scenario_id, category=category,
                    seed=seed, device=device, horizon_ticks=horizon_ticks,
                    action_ticks=action_ticks, decision_index=decision_index,
                    prefix=prefix, descriptor=descriptor, kind="C",
                    candidate_k=candidate_k, prediction_mode=prediction_mode,
                    max_c_candidates=max_c_candidates, max_a_pairs=max_a_pairs,
                    gamma=gamma)
                c_outcomes.append({"descriptor": descriptor, **outcome})
            c_pairs = [{"left_id": left["descriptor"]["candidate_id"],
                "right_id": right["descriptor"]["candidate_id"],
                "gross_horizon_return_difference": left["gross_horizon_return"] - right["gross_horizon_return"],
                "preferred_id": (left["descriptor"]["candidate_id"]
                    if left["gross_horizon_return"] > right["gross_horizon_return"] else
                    right["descriptor"]["candidate_id"]
                    if right["gross_horizon_return"] > left["gross_horizon_return"] else None)}
                for index, left in enumerate(c_outcomes)
                for right in c_outcomes[index + 1:]]
            a_outcomes = []
            for descriptor in item["plan_a_descriptors"]:
                outcome = _branch_return(scenario_id=scenario_id, category=category,
                    seed=seed, device=device, horizon_ticks=horizon_ticks,
                    action_ticks=action_ticks, decision_index=decision_index,
                    prefix=prefix, descriptor=descriptor, kind="A",
                    candidate_k=candidate_k, prediction_mode=prediction_mode,
                    max_c_candidates=max_c_candidates, max_a_pairs=max_a_pairs,
                    gamma=gamma)
                a_outcomes.append({"descriptor": descriptor, **outcome,
                    "p_target": None, "p_prior": 0.0,
                    "p_unlabelled_reason": manifest["plan_a_p_unlabelled_reason"]})
            a_pairs = [{"left_descriptor_sha256": left["descriptor"]["descriptor_sha256"],
                "right_descriptor_sha256": right["descriptor"]["descriptor_sha256"],
                "gross_horizon_return_difference": left["gross_horizon_return"] - right["gross_horizon_return"],
                "preferred_descriptor_sha256": (left["descriptor"]["descriptor_sha256"]
                    if left["gross_horizon_return"] > right["gross_horizon_return"] else
                    right["descriptor"]["descriptor_sha256"]
                    if right["gross_horizon_return"] > left["gross_horizon_return"] else None)}
                for index, left in enumerate(a_outcomes)
                for right in a_outcomes[index + 1:]]
            rows.append({**group, "plan_c_counterfactuals": c_outcomes,
                "plan_c_pairwise_labels": c_pairs,
                "plan_a_sampled_node_behavior_labels": a_outcomes,
                "plan_a_pairwise_preferences": a_pairs,
                "teacher_prefix_action": item["teacher_action"],
                "teacher_prefix_provenance": CONTINUATION_POLICY})
    if source_identity() != sources:
        raise RuntimeError("source/runtime changed during counterfactual collection")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                      for row in rows).encode()
    output.write_bytes(payload)
    manifest["record_count"] = len(rows)
    manifest["dataset_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["records_have_prefix_replay_parity"] = True
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"path": str(output), "record_count": len(rows),
            "dataset_sha256": manifest["dataset_sha256"],
            "schema": DATASET_SCHEMA}
