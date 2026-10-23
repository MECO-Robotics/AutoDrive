"""Matched PPO training and seeded simulator evaluation for MECO A/C tracks.

This is an opt-in research trainer. It never changes the production controller
or relaxes the simulator's existing action legality and safety checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import uuid

import torch
from torch import nn

from .curriculum_scenarios import make_curriculum_env, reset_curriculum_scenario, teacher_actions
from .curriculum_training import get_scenario, source_identity
from .meco_candidates import (CANDIDATE_BOUNDARY, EXTRA_FEATURES,
    ROUTE_COST_POINTS_PER_SECOND,
    RouteFeasibleCandidates, build_route_feasible_candidates)
from .meco_pose_graph import LegalPoseGraph, build_legal_pose_graph
from .meco_tracks import (MECO_TRACKS, make_meco_track,
    predict_opponent_endpoint, sample_candidate_action, score_candidates)


ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_DT = .02
DEFAULT_CATEGORIES = {"pickup_simple": ("single", "cluster")}


@dataclass(frozen=True)
class MecoTrainConfig:
    track: str
    scenario_id: str = "pickup_simple"
    category: str = "single"
    updates: int = 1000
    rollout_steps: int = 32
    epochs: int = 4
    action_ticks: int = 12
    seed: int = 7
    device: str = "cpu"
    learning_rate: float = 3e-4
    gamma: float = .993
    gae_lambda: float = .95
    clip_coef: float = .2
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = .5
    route_cost_points_per_second: float = ROUTE_COST_POINTS_PER_SECOND
    output: str = ""

    def validate(self):
        if self.track not in MECO_TRACKS:
            raise ValueError(f"track must be one of {MECO_TRACKS}")
        definition = get_scenario(self.scenario_id)
        if self.category not in definition.required_categories:
            raise ValueError("category is not registered for scenario")
        for name in ("updates", "rollout_steps", "epochs", "action_ticks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("requested ROCm/CUDA device is unavailable")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_coef",
                     "entropy_coef", "value_coef", "max_grad_norm",
                     "route_cost_points_per_second"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate <= 0 or self.gamma > 1 or self.gae_lambda > 1 or self.max_grad_norm <= 0:
            raise ValueError("invalid optimizer or discount configuration")


class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(71, 128), nn.Tanh(),
                                 nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, globals_, hidden):
        return self.net(torch.cat((globals_, hidden), -1)).squeeze(-1)


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, allow_nan=False)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_torch(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _model_id(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _episode_id(run_id: str, episode: int) -> str:
    return f"{uuid.uuid4().hex}:{run_id}-episode-{episode:08d}"


def _snapshot_tensors(snapshot, *, device):
    """Recover the exact six-channel/seven-scalar tensors bound by a snapshot."""
    raster = torch.tensor(snapshot.raster, device=device, dtype=torch.float32).reshape(
        1, *snapshot.raster_shape)
    globals_ = torch.tensor(snapshot.globals, device=device, dtype=torch.float32).reshape(1, 7)
    return raster, globals_


def _route_cost_tensor(candidate_set, weight, *, device):
    eta = [0. if row["eta_s"] is None else float(row["eta_s"])
           for row in candidate_set.route_records]
    return torch.tensor(eta, device=device, dtype=torch.float32)[None] * float(weight)


def _graph_inputs(track: str, graph: LegalPoseGraph, env, raster):
    if track not in ("A2", "A3"):
        return None
    data = {"node_features": graph.node_features(env, raster),
            "edges": graph.edges, "edge_mask": graph.edge_mask,
            "region_ids": graph.region_ids}
    if hasattr(graph, "coarse_edges"):
        data.update(coarse_edges=graph.coarse_edges,
                    coarse_edge_mask=graph.coarse_edge_mask)
    return data


def _prepare_decision(track, model, env, graph, episode_id, hidden,
                      *, route_cost_points_per_second=ROUTE_COST_POINTS_PER_SECOND,
                      action_ticks=12, deterministic=False, value_head=None):
    observation = env._last_obs
    candidate_set = build_route_feasible_candidates(env, observation, graph,
        episode_id=episode_id, action_ticks=action_ticks)
    raster, globals_ = _snapshot_tensors(candidate_set.observation, device=env.device)
    graph_data = _graph_inputs(track, graph, env, raster)
    route_cost = _route_cost_tensor(candidate_set, route_cost_points_per_second,
                                    device=env.device)
    risk_cost = candidate_set.risk_cost_points.unsqueeze(0)
    decision = score_candidates(track, model, raster=raster, globals_=globals_,
        table=candidate_set.table, hidden=hidden, graph=graph_data,
        route_cost=route_cost, risk_cost=risk_cost)
    action, logprob, entropy = sample_candidate_action(decision["logits"],
                                                       deterministic=deterministic)
    value = (_value(value_head, globals_, decision["hidden"])
             if value_head is not None else None)
    return {"snapshot": candidate_set.observation, "candidate_set": candidate_set,
            "raster": raster, "globals": globals_, "graph": graph_data,
            "decision": decision, "action": action, "logprob": logprob,
            "entropy": entropy, "hidden_next": decision["hidden"],
            "value": value, "route_cost": route_cost, "risk_cost": risk_cost}


def _value(value_head, globals_, hidden):
    return value_head(globals_, hidden)


def _run_macro_step(env, action_index: int, action_ticks: int):
    actions = torch.zeros((env.n, 6), device=env.device, dtype=torch.long)
    actions[0, 3] = int(action_index)
    total = torch.zeros((), device=env.device)
    ended = timeout = False
    for _ in range(action_ticks):
        _, _, done, truncated, _ = env.step(actions, capture_info=False)
        total += env.curriculum_metrics.reward_delta()[0]
        ended = bool(done[0].item())
        timeout = bool(truncated[0].item())
        if ended or timeout:
            break
    env._last_obs = env.observe()
    return total, ended, timeout


def _reset_episode(env, seed: int):
    reset_curriculum_scenario(env, int(seed))
    env._last_obs = env.observe()
    return env._last_obs


def _gae(rewards, dones, values, next_value, gamma, lam):
    advantages = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for index in reversed(range(len(rewards))):
        next_nonterminal = 1. - dones[index]
        next_v = next_value if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + gamma * next_v * next_nonterminal - values[index]
        last = delta + gamma * lam * next_nonterminal * last
        advantages[index] = last
    return advantages, advantages + values


def _policy_inputs_for_row(track, model, row, hidden):
    return score_candidates(track, model, raster=row["raster"],
        globals_=row["globals"], table=row["candidate_set"].table,
        hidden=hidden, graph=row["graph"],
        route_cost=row["route_cost"], risk_cost=row["risk_cost"])


def _ppo_update(track, model, value_head, optimizer, rows, config, bootstrap_value):
    rewards = torch.stack([row["reward"] for row in rows])
    dones = torch.tensor([row["done"] for row in rows], device=rewards.device,
                         dtype=rewards.dtype)
    old_logprob = torch.stack([row["logprob"] for row in rows]).detach()
    old_value = torch.stack([row["value"] for row in rows]).detach()
    with torch.no_grad():
        next_value = (torch.zeros_like(old_value[-1]) if rows[-1]["done"]
                      else bootstrap_value.reshape(()))
    advantage, returns = _gae(rewards, dones, old_value, next_value,
                              config.gamma, config.gae_lambda)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    policy_losses = []; value_losses = []; entropies = []
    count = len(rows)
    for _ in range(config.epochs):
        order = torch.randperm(count, device=rewards.device)
        for idx in order.tolist():
            row = rows[idx]
            current = _policy_inputs_for_row(track, model, row, row["hidden_before"])
            logits = current["logits"]
            dist = torch.distributions.Categorical(logits=logits)
            action = row["action"].reshape(1)
            logprob = dist.log_prob(action).reshape(())
            entropy = dist.entropy().mean()
            value = _value(value_head, row["globals"], current["hidden"]).reshape(())
            ratio = (logprob - old_logprob[idx]).exp()
            unclipped = ratio * advantage[idx]
            clipped = ratio.clamp(1. - config.clip_coef, 1. + config.clip_coef) * advantage[idx]
            policy_loss = -torch.minimum(unclipped, clipped)
            value_loss = .5 * (value - returns[idx]).square()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(value_head.parameters()),
                                     config.max_grad_norm)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy.detach()))
    return {"policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
            "value_loss": sum(value_losses) / max(1, len(value_losses)),
            "entropy": sum(entropies) / max(1, len(entropies)),
            "rollout_return": float(rewards.sum().detach()),
            "rollout_steps": count}


def train_meco(config: MecoTrainConfig, *, resume: str | None = None,
               stop_file: str | None = None):
    config.validate()
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(config.device))
    run_id = uuid.uuid4().hex
    output = Path(config.output or ROOT / "checkpoints" / "meco-five" / config.track)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_identity()
    model = make_meco_track(config.track,
        candidate_features=len(__import__("frc_defense.meco_tracks", fromlist=["SAFE_ACTION_FEATURES"]).SAFE_ACTION_FEATURES) + len(EXTRA_FEATURES)).to(config.device)
    value_head = ValueHead().to(config.device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(value_head.parameters()),
                                 lr=config.learning_rate)
    update = 0
    episode = 0
    history = []
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1 or payload.get("sources") != sources:
            raise ValueError("MECO checkpoint source/schema mismatch; refusing incompatible resume")
        if payload.get("config") != asdict(config):
            raise ValueError("MECO immutable run configuration mismatch")
        model.load_state_dict(payload["model"])
        value_head.load_state_dict(payload["value_head"])
        optimizer.load_state_dict(payload["optimizer"])
        update, episode, history = payload["update"], payload["episode"], payload["history"]
        run_id = payload["run_id"]
        torch.set_rng_state(payload["rng_torch"])
        random.setstate(payload["rng_python"])
        if payload["rng_device"] and torch.cuda.is_available():
            torch.cuda.set_rng_state(payload["rng_device"], torch.device(config.device))
    env = make_curriculum_env(config.scenario_id, config.category, 1,
        config.device, config.seed)
    try:
        graph = build_legal_pose_graph(env, slot=3)
    except BaseException:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise
    run_manifest = {"schema_version": 1, "run_id": run_id,
        "config": asdict(config), "sources": sources,
        "source_identity_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "graph": {"nodes": graph.node_count, "regions": graph.region_count,
                  "resolution_m": graph.resolution_m, "heading_bins": 8},
        "candidate_boundary": CANDIDATE_BOUNDARY,
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    _atomic_json(output / "run.json", run_manifest)
    heartbeat_path = output / "heartbeat.json"
    def update_heartbeat():
        _atomic_json(heartbeat_path, {"schema_version": 1, "run_id": run_id,
            "track": config.track, "pid": os.getpid(),
            "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "completed_updates": update})
    update_heartbeat()
    hidden = None
    episode_generation = _episode_id(run_id, episode)
    stop_path = Path(stop_file) if stop_file else None
    try:
        while update < config.updates and not (stop_path and stop_path.exists()):
            rows = []
            for _ in range(config.rollout_steps):
                if source_identity() != sources:
                    raise RuntimeError("source/runtime changed during MECO training; stopping before checkpoint write")
                hidden_before = None if hidden is None else hidden.detach().clone()
                with torch.no_grad():
                    prepared = _prepare_decision(config.track, model, env, graph,
                        episode_generation, hidden,
                        route_cost_points_per_second=config.route_cost_points_per_second,
                        action_ticks=config.action_ticks, value_head=value_head)
                    action_index = int(prepared["action"][0].item())
                    hidden = prepared["hidden_next"].detach()
                    value = prepared["value"].detach().reshape(())
                action = prepared["candidate_set"].action_indices[action_index]
                reward, ended, timeout = _run_macro_step(env, action, config.action_ticks)
                update_heartbeat()
                rows.append({"raster": prepared["raster"].detach(),
                    "globals": prepared["globals"].detach(),
                    "graph": prepared["graph"],
                    "candidate_set": prepared["candidate_set"],
                    "route_cost": prepared["route_cost"].detach(),
                    "risk_cost": prepared["risk_cost"].detach(),
                    "action": torch.tensor([action_index], device=config.device),
                    "logprob": prepared["logprob"].detach().reshape(()),
                    "hidden_before": hidden_before,
                    "hidden_next": hidden.detach().clone(), "value": value,
                    "reward": reward.detach().reshape(()), "done": ended or timeout})
                if ended or timeout:
                    episode += 1
                    episode_seed = config.seed + episode
                    _reset_episode(env, episode_seed)
                    hidden = None
                    episode_generation = _episode_id(run_id, episode)
            if rows[-1]["done"]:
                bootstrap_value = torch.zeros_like(rows[-1]["value"])
            else:
                with torch.no_grad():
                    bootstrap = _prepare_decision(config.track, model, env, graph,
                        episode_generation, hidden,
                        route_cost_points_per_second=config.route_cost_points_per_second,
                        action_ticks=config.action_ticks, deterministic=True,
                        value_head=value_head)
                    bootstrap_value = bootstrap["value"].detach().reshape(())
            stats = _ppo_update(config.track, model, value_head, optimizer, rows,
                                config, bootstrap_value)
            update += 1
            update_heartbeat()
            stats.update(update=update, episode=episode, track=config.track,
                         scenario_id=config.scenario_id, category=config.category,
                         learning_rate=config.learning_rate,
                         graph_nodes=graph.node_count,
                         update_utc=datetime.now(timezone.utc).isoformat())
            history.append(stats)
            payload = {"schema_version": 1, "config": asdict(config),
                "sources": sources, "run_id": run_id, "update": update,
                "episode": episode, "model": model.state_dict(),
                "value_head": value_head.state_dict(), "optimizer": optimizer.state_dict(),
                "rng_torch": torch.get_rng_state(), "rng_python": random.getstate(),
                "rng_device": torch.cuda.get_rng_state(torch.device(config.device)) if config.device.startswith("cuda") else None,
                "history": history}
            _atomic_torch(output / "latest.pt", payload)
            _atomic_json(output / "stats.json", {"run_id": run_id, "track": config.track,
                "update": update, "episode": episode, "history": history})
        run_manifest.update(status="stopped" if stop_path and stop_path.exists() else "complete",
                            completed_updates=update,
                            stopped_utc=datetime.now(timezone.utc).isoformat())
        _atomic_json(output / "run.json", run_manifest)
        update_heartbeat()
    except BaseException as exc:
        run_manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                            failed_utc=datetime.now(timezone.utc).isoformat())
        _atomic_json(output / "run.json", run_manifest)
        update_heartbeat()
        raise
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return {"track": config.track, "update": update, "checkpoint": str(output / "latest.pt"),
            "run_id": run_id, "status": run_manifest["status"]}


def _d1_d2_scores(candidate_set: RouteFeasibleCandidates, env, policy: str):
    table = candidate_set.table
    cols = {name: i for i, name in enumerate(table.feature_names)}
    values = []
    opponent_xy = env.opponent_pose[0, 3, :2].detach().cpu().tolist()
    opponent_valid = bool(env.opponent_valid[0, 3])
    opponent_age = float(env.opponent_age[0, 3])
    velocity_xy = env.opponent_velocity[0, 3, :2].detach().cpu().tolist()
    for index, (name, row, feasible) in enumerate(zip(table.candidate_ids,
                                                      table.features,
                                                      table.valid_mask)):
        if not feasible:
            values.append(float("-inf"))
            continue
        action = int(row[cols["action_index"]])
        eta = row[cols["route_time_s"]]
        dist = row[cols["target_distance_m"]]
        conf = row[cols["track_confidence"]]
        age = row[cols["track_age_s"]]
        nearby = row[cols["nearby_sensed_track_count"]]
        if action < 4:
            score = ((1. + .2 * float(nearby or 0.) + .25 * float(conf or 0.)) /
                     max(.25, float(eta or 0.)) - .04 * float(age or 0.))
        else:
            score = {"score": .8, "intercept": .1, "ferry": .45, "hold": .05}.get(name, 0.)
        if opponent_valid:
            if policy == "D2":
                pred = predict_opponent_endpoint(valid=True, position_xy=opponent_xy,
                    velocity_xy=velocity_xy, age_s=opponent_age,
                    horizon_s=float(eta or 0.), max_speed_mps=3., max_age_s=.5)
                enemy = pred or tuple(opponent_xy)
            else:
                enemy = tuple(opponent_xy)
            dx = float(row[cols["goal_x_m"]]) - enemy[0]
            dy = float(row[cols["goal_y_m"]]) - enemy[1]
            separation = math.hypot(dx, dy)
            score -= .75 * max(0., min(1., (2. - separation) / 2.))
        values.append(score)
    return torch.tensor(values, device=env.device, dtype=torch.float32)[None]


@torch.no_grad()
def evaluate_meco(checkpoints: dict[str, str], *, scenario_id: str,
                  categories: tuple[str, ...], episodes: int, master_seed: int,
                  device: str = "cpu", include_controls=True, output: str | None = None):
    """Paired seeded simulator evaluation for D0/D1/D2 and five checkpoints."""
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be positive")
    definition = get_scenario(scenario_id)
    if any(category not in definition.required_categories for category in categories):
        raise ValueError("category is not registered for scenario")
    selected = tuple(MECO_TRACKS)
    if set(checkpoints) != set(selected):
        raise ValueError("evaluation needs exactly one checkpoint for each of A1/A2/A3/C1/C2")
    sources = source_identity()
    loaded = {}
    policy_manifest = {}
    route_weight = None
    for track, path in checkpoints.items():
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload["config"]["track"] != track or payload["sources"] != sources:
            raise ValueError(f"{track} checkpoint is incompatible with current source")
        track_route_weight = float(payload["config"].get(
            "route_cost_points_per_second", ROUTE_COST_POINTS_PER_SECOND))
        if not math.isfinite(track_route_weight) or track_route_weight < 0:
            raise ValueError(f"{track} checkpoint has invalid route cost weight")
        if route_weight is None:
            route_weight = track_route_weight
        elif track_route_weight != route_weight:
            raise ValueError("paired evaluation checkpoints must share one route-cost weight")
        model = make_meco_track(track,
            candidate_features=len(__import__("frc_defense.meco_tracks", fromlist=["SAFE_ACTION_FEATURES"]).SAFE_ACTION_FEATURES) + len(EXTRA_FEATURES)).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        loaded[track] = model
        checkpoint_digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        policy_manifest[track] = {"policy_id": _model_id(model),
            "checkpoint_sha256": checkpoint_digest,
            "source_identity_sha256": hashlib.sha256(
                json.dumps(sources, sort_keys=True).encode()).hexdigest()}
    for control in ("D0", "D1", "D2"):
        policy_manifest[control] = {"policy_id": control, "checkpoint_sha256": None,
            "source_identity_sha256": hashlib.sha256(
                json.dumps(sources, sort_keys=True).encode()).hexdigest()}
    protocol = hashlib.sha256(json.dumps({"sources": sources, "scenario": scenario_id,
        "categories": categories, "episodes": episodes, "master_seed": master_seed,
        "dt": OBSERVATION_DT, "action_ticks": 12, "route_weight": route_weight,
        "candidate_contract": "eight masked actions; verified static terminal SE(2) pose and current-snapshot AD* route"},
        sort_keys=True).encode()).hexdigest()
    eval_id = uuid.uuid4().hex
    output_path = Path(output) if output else None
    status_path = (output_path.with_name(output_path.name + ".status.json")
                   if output_path is not None else None)
    status_last_written = 0.
    status_last_completed = -1
    def update_evaluation_status(status, completed=0):
        nonlocal status_last_written, status_last_completed
        if status_path is None:
            return
        current = time.monotonic()
        if (status == "running" and completed == status_last_completed
                and current - status_last_written < 5.):
            return
        _atomic_json(status_path, {"schema_version": 1, "evaluation_id": eval_id,
            "protocol_id": protocol, "split": "validation", "status": status,
            "pid": os.getpid(), "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "output_file": output_path.name,
            "progress": {"completed_records": completed,
                         "total_records": len(categories) * episodes * (len(selected) + 3)}})
        status_last_written = current
        status_last_completed = completed
    records = []
    try:
        update_evaluation_status("running")
        for category in categories:
            for episode_index in range(episodes):
                seed = master_seed + episode_index
                # Same simulator reset seed and task for every control and model.
                policies = list(selected) + (["D0", "D1", "D2"] if include_controls else [])
                for policy in policies:
                    episode_id = _episode_id(f"eval-{eval_id}-{category}-{policy}",
                                             episode_index)
                    env = make_curriculum_env(scenario_id, category, 1, device, seed)
                    try:
                        graph = build_legal_pose_graph(env, slot=3)
                        hidden = None
                        actions_count = [0] * 8
                        routes_seen = []
                        elapsed = 0
                        while elapsed < env.horizon:
                            obs = env._last_obs
                            candidate_set = build_route_feasible_candidates(env, obs, graph,
                                episode_id=episode_id, action_ticks=12)
                            raster, globals_ = _snapshot_tensors(candidate_set.observation,
                                                                 device=env.device)
                            if policy in MECO_TRACKS:
                                graph_data = _graph_inputs(policy, graph, env, raster)
                                scored = score_candidates(policy, loaded[policy],
                                    raster=raster, globals_=globals_, table=candidate_set.table,
                                    hidden=hidden, graph=graph_data,
                                    route_cost=_route_cost_tensor(candidate_set, route_weight,
                                                                   device=env.device),
                                    risk_cost=candidate_set.risk_cost_points[None])
                                action_tensor, _, _ = sample_candidate_action(scored["logits"], deterministic=True)
                                choice = int(action_tensor[0].item())
                                hidden = scored["hidden"]
                                action_source = "model"
                            elif policy == "D0":
                                teacher = int(teacher_actions(env, obs)[0, 3].item())
                                choice = teacher if candidate_set.table.valid_mask[teacher] else candidate_set.table.candidate_ids.index(candidate_set.fallback_candidate_id)
                                action_source = "d0_teacher"
                            else:
                                logits = _d1_d2_scores(candidate_set, env, policy)
                                choice = int(logits.argmax(-1).item())
                                action_source = policy.lower()
                            action = candidate_set.action_indices[choice]
                            actions_count[action] += 1
                            routes_seen.append(candidate_set.route_records[choice])
                            _, ended, timeout = _run_macro_step(env, action, 12)
                            elapsed += 12
                            update_evaluation_status("running", len(records))
                            if ended or timeout:
                                break
                        outcomes = env.curriculum_metrics.outcomes()
                        metrics = {name: value.detach().cpu().reshape(-1)[0].item()
                                   for name, value in outcomes.items()}
                        records.append({"scenario_id": scenario_id, "category": category,
                            "seed": seed, "policy": policy, "action_source": action_source,
                            "policy_id": policy_manifest[policy]["policy_id"],
                            "checkpoint_sha256": policy_manifest[policy]["checkpoint_sha256"],
                            "protocol_id": protocol, "split": "validation",
                            "source_identity_sha256": hashlib.sha256(json.dumps(
                                sources, sort_keys=True).encode()).hexdigest(),
                            "actions": actions_count, "elapsed_ticks": elapsed,
                            "metrics": metrics, "candidate_recall": candidate_set.candidate_recall,
                            "route_trace": routes_seen,
                            "episode_id": episode_id})
                        update_evaluation_status("running", len(records))
                    finally:
                        close = getattr(env, "close", None)
                        if callable(close):
                            close()
        report = {"schema_version": 1, "evaluation_id": eval_id,
            "protocol_id": protocol, "sources": sources,
            "scenario_id": scenario_id, "categories": list(categories),
            "episodes_per_category": episodes, "master_seed": master_seed,
            "seeds": [master_seed + i for i in range(episodes)],
            "tracks": list(selected), "controls": ["D0", "D1", "D2"] if include_controls else [],
            "policy_manifest": policy_manifest,
        "paired": True, "split": "validation", "diagnostic": episodes < 100,
        "candidate_boundary": CANDIDATE_BOUNDARY,
            "source_identity_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            "dashboard_verified": False, "records": records,
            "generated_utc": datetime.now(timezone.utc).isoformat()}
        if output:
            _atomic_json(Path(output), report)
            update_evaluation_status("complete", len(records))
        return report
    except BaseException:
        try:
            update_evaluation_status("failed", len(records))
        except OSError:
            pass
        raise


def _parser():
    parser = argparse.ArgumentParser(description="Matched MECO five-track research trainer")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--track", choices=MECO_TRACKS, required=True)
    train.add_argument("--scenario", default="pickup_simple")
    train.add_argument("--category", default="single")
    train.add_argument("--updates", type=int, default=1000)
    train.add_argument("--rollout-steps", type=int, default=32)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--action-ticks", type=int, default=12)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="cpu")
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--stop-file")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--scenario", default="pickup_simple")
    ev.add_argument("--categories", default="single,cluster")
    ev.add_argument("--episodes", type=int, default=1)
    ev.add_argument("--seed", type=int, default=20261024)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--output", required=True)
    for track in MECO_TRACKS:
        ev.add_argument(f"--{track.lower()}-checkpoint", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "train":
        config = MecoTrainConfig(track=args.track, scenario_id=args.scenario,
            category=args.category, updates=args.updates, rollout_steps=args.rollout_steps,
            epochs=args.epochs, action_ticks=args.action_ticks, seed=args.seed,
            device=args.device, learning_rate=args.learning_rate, output=args.output)
        result = train_meco(config, resume=args.resume, stop_file=args.stop_file)
        print(json.dumps(result, indent=2))
        return
    checkpoints = {track: getattr(args, f"{track.lower()}_checkpoint") for track in MECO_TRACKS}
    report = evaluate_meco(checkpoints, scenario_id=args.scenario,
        categories=tuple(item for item in args.categories.split(",") if item),
        episodes=args.episodes, master_seed=args.seed, device=args.device,
        output=args.output)
    print(json.dumps({"evaluation_id": report["evaluation_id"],
        "protocol_id": report["protocol_id"], "records": len(report["records"]),
        "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
