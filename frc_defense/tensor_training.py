"""Torch-only PPO for :class:`TensorDefenseEnv`.

Rollouts, advantages, minibatches, and optimizer updates stay as Torch tensors
on the selected device. On ROCm, PyTorch exposes the GPU through ``cuda`` too.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError as exc:  # keep the module's import error actionable
    raise ImportError(
        "Tensor PPO requires PyTorch. Install a Torch build for your platform, "
        "including ROCm or CUDA support when using a GPU."
    ) from exc


OBS_DIM = 35
ACTION_DIM = 3
STRATEGIC_LEGACY_OBS_DIM = 89
STRATEGIC_ACTION_DIM = 8
OBS_NORMALIZATION = "fixed_physical_scale_v1"
MIXED_DEFENSE_OPPONENTS = ("adstar", "offense", "intercept", "velocity_intercept", "mirror")
STATIC_OPPONENT_FRACTION = .20
CURRICULUM_STAGES = ("static_straight", "scripted", "adstar", "mixed", "learned")


def _curriculum_stage(progress: float) -> int:
    return min(len(CURRICULUM_STAGES) - 1,
               max(0, int(max(0., min(.999999, progress)) * len(CURRICULUM_STAGES))))


def _curriculum_opponents(task: str, stage: int, learned_available: bool = True) -> tuple[str, ...]:
    """Expand training from easy motion to durable scripted and learned pools."""
    if task == "defense":
        base = ("offense", "intercept", "mirror", "adstar", "velocity_intercept")
        pools = (("offense",), ("offense", "intercept", "mirror"),
                 ("offense", "intercept", "mirror", "adstar"), base,
                 base + (("learned",) if learned_available else ()))
    else:
        base = ("guard", "intercept", "mirror", "adstar_defender")
        pools = (("guard",), ("guard", "intercept", "mirror"),
                 ("guard", "intercept", "mirror", "adstar_defender"), base,
                 base + (("learned",) if learned_available else ()))
    return pools[min(max(0, stage), len(pools) - 1)]


class ActorCritic(nn.Module):
    """Fixed-size actor/critic for continuous controls or categorical tactics."""

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM,
                 action_kind: str = "continuous"):
        super().__init__()
        if action_kind not in ("continuous", "categorical"):
            raise ValueError("action_kind must be continuous or categorical")
        self.action_kind = action_kind
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(),
                                   nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        if action_kind == "continuous":
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        logits_or_mean = self.actor(features)
        scale = (None if self.action_kind == "categorical"
                 else self.log_std.expand_as(logits_or_mean))
        return logits_or_mean, scale, self.critic(features).squeeze(-1)

    def sample(self, obs: torch.Tensor, deterministic: bool = False,
               action_mask: torch.Tensor | None = None):
        mean, log_std, value = self(obs)
        if self.action_kind == "categorical":
            if action_mask is not None:
                mean=mean.masked_fill(~action_mask.bool(),torch.finfo(mean.dtype).min)
            dist = torch.distributions.Categorical(logits=mean)
            action = mean.argmax(-1) if deterministic else dist.sample()
            return action, dist.log_prob(action), value
        dist = Normal(mean, log_std.exp())
        latent = mean if deterministic else dist.rsample()
        action = torch.tanh(latent)
        # Change of variables for tanh. Clamp only the correction's argument
        # to avoid log(0) at saturated actions.
        log_prob = dist.log_prob(latent).sum(-1) - torch.log(1 - action.square() + 1e-6).sum(-1)
        return action, log_prob, value

    def score(self, obs: torch.Tensor, actions: torch.Tensor,
              action_mask: torch.Tensor | None = None):
        mean, log_std, value = self(obs)
        if self.action_kind == "categorical":
            if action_mask is not None:
                mean=mean.masked_fill(~action_mask.bool(),torch.finfo(mean.dtype).min)
            dist = torch.distributions.Categorical(logits=mean)
            actions = actions.to(torch.long).reshape(-1)
            return dist.log_prob(actions), dist.entropy(), value, mean
        actions = actions.clamp(-0.999999, 0.999999)
        latent = torch.atanh(actions)
        dist = Normal(mean, log_std.exp())
        log_prob = dist.log_prob(latent).sum(-1) - torch.log(1 - actions.square() + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value, mean


def _device(requested: str) -> torch.device:
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested accelerator {requested!r} is unavailable; refusing CPU fallback")
    return dev


def _env(num_envs: int, task: str, device: torch.device, seed: int, opponent: str,
         horizon: int = 8000, normalize_observations: bool = True,
         static_opponent_fraction: float = 0., architecture: str = "direct"):
    try:
        from .tensor_sim import TensorDefenseEnv
    except ImportError as exc:
        raise RuntimeError("Tensor PPO requires frc_defense.tensor_sim.TensorDefenseEnv") from exc
    env = TensorDefenseEnv(num_envs=num_envs, task=task, device=device,
                           seed=seed, opponent=opponent, horizon=horizon,
                           normalize_observations=normalize_observations,
                           static_opponent_fraction=static_opponent_fraction,
                           action_mode={"direct": "direct", "tactical_adstar": "tactical",
                                        "strategic_adstar": "strategic"}.get(architecture, architecture))
    return env


def _attach_historical_opponent(env, task: str, device: torch.device,
                               checkpoint_path: str | Path | None = None) -> bool:
    """Load historical direct or 89-feature strategic policies as opponents."""
    names = (("rebuilt-gamepiece-offense", "rebuilt-counter-defense")
             if task == "defense" else
             ("rebuilt-gamepiece-defense", "rebuilt-defense-generational", "rebuilt-defense"))
    checkpoint = (Path(checkpoint_path) if checkpoint_path else
        next((Path("checkpoints") / name / "policy.pt" for name in names
              if (Path("checkpoints") / name / "policy.pt").is_file()),None))
    if checkpoint is None:
        return False
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        expected_obs=int(payload.get("obs_dim",OBS_DIM))
        if expected_obs not in (OBS_DIM,STRATEGIC_LEGACY_OBS_DIM,137):
            return False
        model = ActorCritic(payload.get("obs_dim", OBS_DIM),
                            payload.get("action_dim", ACTION_DIM),
                            payload.get("action_kind", "continuous")).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
    except (OSError, RuntimeError, KeyError, ValueError):
        return False

    @torch.no_grad()
    def policy(observation):
        expected_obs=int(payload.get("obs_dim",OBS_DIM))
        obs=observation[:,:expected_obs]
        action, _, _ = model(obs)
        if model.action_kind == "categorical":
            if model.actor.out_features==STRATEGIC_ACTION_DIM and observation.shape[-1]>=STRATEGIC_ACTION_DIM:
                mask=observation[:,-STRATEGIC_ACTION_DIM:].bool()
                action=action.masked_fill(~mask,torch.finfo(action.dtype).min)
            return action.argmax(-1)
        return torch.tanh(action)

    env.learned_opponent_fn = policy
    env.learned_opponent_obs_dim=expected_obs
    env.learned_opponent_action_dim=int(payload.get("action_dim",ACTION_DIM))
    env.learned_opponent_architecture=payload.get("architecture","direct")
    env.learned_opponent_checkpoint = str(checkpoint)
    return True


def _reset_obs(result: Any) -> torch.Tensor:
    """Accept Gym-style ``(obs, info)`` as well as the bare tensor contract."""
    return result[0] if isinstance(result, tuple) else result


def _held_action_interval(env, action: torch.Tensor, ticks: int,
                          initial_obs: torch.Tensor):
    """Advance active worlds under one action without crossing episode ends."""
    active_this_decision = torch.ones(env.n, dtype=torch.bool, device=env.device)
    done = torch.zeros_like(active_this_decision)
    truncated = torch.zeros_like(active_this_decision)
    reward = torch.zeros(env.n, dtype=torch.float32, device=env.device)
    transition_next_obs = initial_obs
    complete_matches = 0
    physics_ticks = 0
    for _ in range(ticks):
        if not bool(active_this_decision.any()):
            break
        next_obs, tick_reward, tick_done, tick_truncated, _info = env.step(
            action, active_mask=active_this_decision)
        active_before_tick = active_this_decision
        reward += tick_reward.to(device=env.device, dtype=torch.float32) * active_before_tick
        tick_done = tick_done.to(device=env.device, dtype=torch.bool)
        tick_truncated = tick_truncated.to(device=env.device, dtype=torch.bool)
        done |= tick_done & active_before_tick
        truncated |= tick_truncated & active_before_tick
        transition_next_obs = torch.where(
            active_before_tick[:, None], next_obs, transition_next_obs)
        newly_ended = active_before_tick & (tick_done | tick_truncated)
        complete_matches += int((tick_truncated & active_before_tick).sum().item())
        active_this_decision &= ~newly_ended
        physics_ticks += 1
    return (reward, done, truncated, transition_next_obs, active_this_decision,
            physics_ticks, complete_matches)


def _gae_advantages(rewards: torch.Tensor, dones: torch.Tensor,
                    truncated: torch.Tensor, values: torch.Tensor,
                    next_values: torch.Tensor, gamma: float,
                    gae_lambda: float) -> torch.Tensor:
    """GAE bootstraps horizon truncations, but never true terminations."""
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        nonterminal = (~dones[t]).float()
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        continuation = (~(dones[t] | truncated[t])).float()
        last_gae = delta + gamma * gae_lambda * continuation * last_gae
        advantages[t] = last_gae
    return advantages


def _task_opponent(task: str, opponent: str | None) -> str:
    if opponent is None:
        return "guard" if task == "counter_defense" else "offense"
    if task == "defense" and opponent == "guard":
        return "offense"
    return opponent


def _adstar_reference(env, device, max_worlds: int = 64, world_indices=None,
                      include_all_routes: bool = False):
    """Plan sparse teacher routes with the same device-resident grid search."""
    from .tensor_adstar import TensorADStar
    count = min(max_worlds, env.n)
    # Keep world zero first because it is the representative world exported to
    # the playback recording; fill the remaining references randomly.
    if world_indices is not None:
        indices = torch.as_tensor(world_indices, device=device, dtype=torch.long)
        count = int(indices.numel())
    elif count == 1:
        indices = torch.zeros(1, device=device, dtype=torch.long)
    else:
        indices = torch.cat((torch.zeros(1, device=device, dtype=torch.long),
                             torch.randperm(env.n - 1, device=device)[:count-1] + 1))
    starts=env.sim.pose[:,0,:2]
    goals=env.goal.clone()
    if env.task == "defense":
        # Reference an intercept point on the attacker's straight-line scoring
        # approach; the field planner routes the defender around solid elements.
        goals=.55*env.goal+.45*env.sim.pose[:,1,:2]
    planner=TensorADStar(env)
    dynamic=env.task=="counter_defense"
    padded_all,lengths_all,_,_=planner.plan(starts,goals,env.sim.pose[:,0,2],
        env.sim.length[:,0],env.sim.width[:,0],env.sim.speed[:,0],
        env.sim.pose[:,1,:2],torch.zeros_like(env.sim.velocity[:,1,:2]),dynamic,
        env.sim.lateral_mu[:,0],env.sim.accel[:,0])
    padded=padded_all[indices]
    lengths=lengths_all[indices]
    segments = (padded[:, 1:] - padded[:, :-1]).norm(dim=-1)
    cumulative = torch.cat((torch.zeros((count, 1), device=device), segments.cumsum(-1)), -1)
    total = cumulative.gather(1, (lengths - 1)[:, None]).squeeze(1)
    # Training exports only its representative route. Evaluation playback can
    # request one route per sampled scenario for its ghost rollouts.
    route_count = count if include_all_routes else min(1, count)
    routes=[padded[i,:int(lengths[i].item())].detach().cpu().tolist()
            for i in range(route_count)]
    return indices, padded, lengths, cumulative, total, routes


def _reference_action(routes, positions, headings, speeds):
    indices, points, lengths, _cumulative, _total = routes[:5]
    pos = positions[indices]
    d2 = (points - pos[:, None, :]).square().sum(-1)
    valid = torch.arange(points.shape[1], device=points.device)[None, :] < lengths[:, None]
    nearest = d2.masked_fill(~valid, float("inf")).argmin(-1)
    target_index = torch.minimum(nearest + 1, lengths - 1)
    target = points[torch.arange(len(indices), device=points.device), target_index]
    delta = target - pos
    angle = headings[indices]
    body = torch.stack((angle.cos() * delta[:, 0] + angle.sin() * delta[:, 1],
                        -angle.sin() * delta[:, 0] + angle.cos() * delta[:, 1]), -1)
    return torch.cat(((body / body.norm(dim=-1, keepdim=True).clamp_min(1e-6)),
                      torch.zeros((len(indices), 1), device=points.device)), -1)


def _reference_potential(routes, positions):
    indices, points, lengths, cumulative, total = routes[:5]
    pos = positions[indices]
    d2 = (points - pos[:, None, :]).square().sum(-1)
    valid = torch.arange(points.shape[1], device=points.device)[None, :] < lengths[:, None]
    nearest = d2.masked_fill(~valid, float("inf")).argmin(-1)
    travelled = cumulative.gather(1, nearest[:, None]).squeeze(1)
    return travelled - total


def _reference_tracking_error(routes, positions):
    """Distance from each sampled robot to its nearest AD* route segment."""
    indices, points, lengths, _cumulative, _total = routes[:5]
    pos = positions[indices]
    if points.shape[1] == 1:
        return (pos - points[:, 0]).norm(dim=-1)
    start, end = points[:, :-1], points[:, 1:]
    segment = end - start
    fraction = ((pos[:, None] - start) * segment).sum(-1) / segment.square().sum(-1).clamp_min(1e-8)
    projection = start + fraction.clamp(0, 1)[..., None] * segment
    distance = (pos[:, None] - projection).square().sum(-1)
    valid = torch.arange(points.shape[1] - 1, device=points.device)[None, :] < (lengths - 1)[:, None]
    segment_distance = distance.masked_fill(~valid, float("inf")).min(-1).values.clamp_min(0).sqrt()
    point_distance = (pos - points[:, 0]).norm(dim=-1)
    return torch.where(lengths == 1, point_distance, segment_distance)


def _train_once(task: str, timesteps: int, output: str | Path, *, seed: int,
                num_envs: int, device: torch.device, opponent: str,
                initial_checkpoint: str | Path | None,
                rollout_steps: int, epochs: int, minibatch_size: int,
                learning_rate: float, gamma: float, gae_lambda: float,
                clip_coef: float, value_coef: float, entropy_coef: float,
                max_grad_norm: float, l2_coef: float, horizon: int,
                architecture: str = "direct", curriculum: bool = True,
                strategic_rate_hz: float = 4.0,
                learned_opponent_checkpoint: str | Path | None = None,
                learned_opponent_checkpoints: list[str | Path] | None = None,
                status_context: dict[str, Any] | None = None) -> dict[str, Any]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 2. <= strategic_rate_hz <= 5.:
        raise ValueError("strategic_rate_hz must be between 2 and 5")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    env = _env(num_envs, task, device, seed, opponent, horizon,
               static_opponent_fraction=STATIC_OPPONENT_FRACTION,
               architecture=architecture)
    horizon=int(env.horizon)
    learned_opponent_available = _attach_historical_opponent(
        env, task, device, learned_opponent_checkpoint)
    uses_adstar_teacher = task != "defense" and architecture == "direct"
    obs_dim = int(getattr(env, "obs_dim", OBS_DIM))
    action_dim = int(getattr(env, "action_dim", ACTION_DIM))
    action_kind = "categorical" if architecture == "strategic_adstar" else "continuous"
    model = ActorCritic(obs_dim, action_dim, action_kind).to(device)
    if initial_checkpoint is not None:
        payload = torch.load(initial_checkpoint, map_location=device, weights_only=True)
        if payload.get("drivetrain_config") != env.drivetrain_config:
            raise ValueError("initial checkpoint drivetrain configuration does not match this run; "
                             "retrain without that checkpoint or use its exact FRC_DRIVETRAIN_CONFIG")
        if payload.get("architecture", "direct") != architecture:
            raise ValueError("initial checkpoint architecture differs from this training run")
        model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    obs = _reset_obs(env.reset())
    if not isinstance(obs, torch.Tensor):
        raise TypeError("TensorDefenseEnv.reset() must return a Torch tensor")
    obs = obs.to(device=device, dtype=torch.float32)
    start = time.perf_counter()
    started_at = time.time()
    updates = (timesteps + rollout_steps * num_envs - 1) // (rollout_steps * num_envs)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "policy.pt"
    status_path = out / "status.json"
    playback_path = out / "playback.json"
    status_context = dict(status_context or {})
    completed_timesteps_base = int(status_context.get("completed_timesteps_base", 0))
    requested_timesteps_total = int(status_context.get("requested_timesteps_total", timesteps))
    playback_frames: list[dict[str, Any]] = []
    def write_json(path: Path, value: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(path)
    def write_checkpoint(value: dict[str, Any]) -> None:
        temp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        torch.save(value, temp)
        temp.replace(checkpoint)
    write_json(status_path, {**status_context, "status": "running", "task": task, "opponent": opponent,
        "architecture": architecture, "action_kind": action_kind,
        "curriculum": list(CURRICULUM_STAGES) if curriculum else [],
        "stationary_opponent_fraction": STATIC_OPPONENT_FRACTION,
        "requested_timesteps": requested_timesteps_total, "num_envs": num_envs, "device": str(device),
        "observation_normalization": OBS_NORMALIZATION,
        "l2_coefficient": l2_coef,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "completed_timesteps": completed_timesteps_base, "updates": 0, "total_updates": updates,
        "adstar_action_loss_weight": .25 if uses_adstar_teacher else 0., "task_reward_scale": 1.,
        "adstar_tracking_error_m": None, "started_at": started_at,
        "drivetrain_config": env.drivetrain_config})
    write_json(playback_path, {"task": task, "dt": env.dt, "frames": []})
    # Attacker training can use sparse route imitation; defenders learn only
    # from task reward and the visible robot/field state.
    adstar_action_loss_weight = .25 if uses_adstar_teacher else 0.
    task_reward_scale = 1.
    tracking_error_ema = None
    good_tracking_updates = 0
    strategic_phase = 0.0
    physics_ticks = 0
    complete_matches = 0
    action_counts = torch.zeros(action_dim, device=device)
    opponent_exposure: dict[str, int] = {}
    entropy_mean = 0.0
    for update_index in range(updates):
        stage = _curriculum_stage(update_index / max(1, updates)) if curriculum else -1
        if curriculum:
            available = learned_opponent_available and stage >= 4
            candidates = _curriculum_opponents(task, stage, learned_available=available)
            env.opponent = candidates[update_index % len(candidates)]
            env.static_opponent_fraction = (.4 if stage == 0 else STATIC_OPPONENT_FRACTION)
        exposure_label = env.opponent
        if env.opponent == "learned" and learned_opponent_checkpoints:
            checkpoint_index = update_index % len(learned_opponent_checkpoints)
            active_opponent_checkpoint = learned_opponent_checkpoints[checkpoint_index]
            learned_opponent_available = _attach_historical_opponent(
                env, task, device, active_opponent_checkpoint)
            exposure_label = f"learned:{active_opponent_checkpoint}"
        opponent_exposure[exposure_label] = opponent_exposure.get(exposure_label, 0) + rollout_steps * num_envs
        # AD* is evaluated at rollout boundaries for a sparse reference set.
        # The simulator and PPO tensors stay device resident during each step.
        routes = _adstar_reference(env, device) if uses_adstar_teacher else None
        reference_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)
        if routes is not None:
            reference_mask[routes[0]] = True
        reference_active = reference_mask.clone()
        action_shape = (action_dim,) if action_kind == "continuous" else ()
        action_dtype = torch.float32 if action_kind == "continuous" else torch.long
        b_obs = torch.empty((rollout_steps, num_envs, obs_dim), device=device)
        b_actions = torch.empty((rollout_steps, num_envs, *action_shape), device=device,
                                dtype=action_dtype)
        b_action_masks=(torch.empty((rollout_steps,num_envs,action_dim),device=device,dtype=torch.bool)
                        if action_kind=="categorical" else None)
        b_logprobs = torch.empty((rollout_steps, num_envs), device=device)
        b_rewards = torch.empty((rollout_steps, num_envs), device=device)
        b_dones = torch.empty((rollout_steps, num_envs), device=device, dtype=torch.bool)
        b_truncated = torch.empty((rollout_steps, num_envs), device=device, dtype=torch.bool)
        b_values = torch.empty((rollout_steps, num_envs), device=device)
        b_next_values = torch.empty((rollout_steps, num_envs), device=device)
        b_teacher_actions = torch.zeros((rollout_steps, num_envs, action_dim), device=device)
        b_teacher_mask = torch.zeros((rollout_steps, num_envs), device=device, dtype=torch.bool)
        tracking_error_sum = torch.zeros((), device=device)
        tracking_error_count = torch.zeros((), device=device)
        update_frames = []
        frame_stride = max(1, rollout_steps // 24)
        for t in range(rollout_steps):
            b_obs[t] = obs
            if routes is not None:
                pos_before = env.sim.pose[:, 0, :2]
                tracking_error = _reference_tracking_error(routes, pos_before)
                sampled_active = reference_active[routes[0]]
                tracking_error_sum += (tracking_error * sampled_active.float()).sum()
                tracking_error_count += sampled_active.sum()
                potential_before = _reference_potential(routes, pos_before)
                teacher = _reference_action(routes, pos_before, env.sim.pose[:, 0, 2], env.sim.speed[:, 0])
                b_teacher_actions[t, routes[0]] = teacher
                b_teacher_mask[t] = reference_active
            with torch.no_grad():
                action_mask=(obs[:,-action_dim:].bool() if action_kind=="categorical" and
                             action_dim==STRATEGIC_ACTION_DIM else None)
                action, logprob, value = model.sample(obs,action_mask=action_mask)
                if action_kind == "categorical":
                    action_counts += torch.bincount(action, minlength=action_dim)
            if b_action_masks is not None:
                b_action_masks[t]=action_mask
            if architecture == "strategic_adstar":
                strategic_phase += 50.0 / strategic_rate_hz
                ticks_this_decision = max(1, int(strategic_phase))
                strategic_phase -= ticks_this_decision
            else:
                ticks_this_decision = 1
            if architecture == "strategic_adstar":
                (reward, done, truncated, next_obs, active_this_decision,
                 decision_physics_ticks, decision_matches) = _held_action_interval(
                    env, action, ticks_this_decision, obs)
                physics_ticks += decision_physics_ticks
                complete_matches += decision_matches
            else:
                next_obs, reward, done, truncated, _info = env.step(action)
                physics_ticks += 1
                complete_matches += int(truncated.sum().item())
                if routes is not None:
                    potential_after = _reference_potential(routes, env.sim.pose[:, 0, :2])
                    route_progress = (potential_after - potential_before).clamp(-.25, .25)
                    reward[routes[0]] += .20 * route_progress * reference_active[routes[0]]
            reward = reward.to(device=device, dtype=torch.float32).reshape(num_envs)
            reward *= task_reward_scale
            if t % frame_stride == 0:
                update_frames.append(torch.cat((env.sim.pose[0].reshape(-1), env.goal[0],
                    env.sim.length[0], env.sim.width[0], env.goal_radius[0:1])).detach().clone())
            next_obs = next_obs.to(device=device, dtype=torch.float32)
            reward = reward.to(device=device, dtype=torch.float32).reshape(num_envs)
            done = done.to(device=device, dtype=torch.bool).reshape(num_envs)
            truncated = truncated.to(device=device, dtype=torch.bool).reshape(num_envs)
            b_actions[t], b_logprobs[t], b_values[t] = action, logprob, value
            b_rewards[t], b_dones[t], b_truncated[t] = reward, done, truncated
            ended = done | truncated
            # Bootstrap from the terminal observation, then reset ended rows
            # before sampling the next strategic action.
            transition_next_obs = next_obs.to(device=device, dtype=torch.float32)
            reference_active &= ~ended
            with torch.no_grad():
                b_next_values[t] = model(transition_next_obs)[2]
            reset_done = getattr(env, "reset_done", None)
            if reset_done is None:
                raise RuntimeError("TensorDefenseEnv must provide reset_done(mask) for per-world episode resets")
            next_obs = reset_done(ended) if bool(ended.any()) else transition_next_obs
            obs = next_obs.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            advantages = _gae_advantages(
                b_rewards, b_dones, b_truncated, b_values, b_next_values,
                gamma, gae_lambda)
            returns = advantages + b_values

        flat_obs = b_obs.reshape(-1, obs_dim)
        flat_actions = b_actions.reshape((-1, action_dim) if action_kind == "continuous" else (-1,))
        flat_action_masks=(b_action_masks.reshape(-1,action_dim) if b_action_masks is not None else None)
        flat_logprobs = b_logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_teacher_actions = b_teacher_actions.reshape(-1, action_dim)
        flat_reference_mask = b_teacher_mask.reshape(-1)
        batch_size = flat_obs.shape[0]
        for _epoch in range(epochs):
            indices = torch.randperm(batch_size, device=device)
            for idx in indices.split(minibatch_size):
                action_mask=flat_action_masks[idx] if flat_action_masks is not None else None
                new_logprob, entropy, new_value, mean = model.score(flat_obs[idx], flat_actions[idx],action_mask)
                logratio = new_logprob - flat_logprobs[idx]
                ratio = logratio.exp()
                mb_adv = flat_advantages[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
                pg = torch.maximum(-mb_adv * ratio,
                                   -mb_adv * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)).mean()
                value_loss = 0.5 * (new_value - flat_returns[idx]).square().mean()
                teacher_rows = flat_reference_mask[idx].float()
                teacher_target = torch.atanh(flat_teacher_actions[idx].clamp(-.95, .95))
                teacher_error = (mean - teacher_target).square().mean(-1)
                teacher_loss = (teacher_error * teacher_rows).sum() / teacher_rows.sum().clamp_min(1.)
                l2_norm = sum(parameter.square().sum() for parameter in model.parameters()
                              if parameter.ndim > 1)
                entropy_mean = float(entropy.mean().detach().item())
                update_entropy_coef = entropy_coef * max(0., 1. - update_index / max(1, updates - 1))
                loss = (pg + value_coef * value_loss - update_entropy_coef * entropy.mean() +
                        adstar_action_loss_weight * teacher_loss + l2_coef * l2_norm)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # Keep the latest policy usable for the dashboard and for recovery if
        # a long run is interrupted. Status and playback are atomically replaced.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_tracking_error = None
        if uses_adstar_teacher:
            update_tracking_error = float((tracking_error_sum / tracking_error_count.clamp_min(1)).item())
            tracking_error_ema = (update_tracking_error if tracking_error_ema is None else
                                  .8 * tracking_error_ema + .2 * update_tracking_error)
            if tracking_error_ema <= .35:
                good_tracking_updates += 1
            else:
                good_tracking_updates = 0
            if good_tracking_updates >= 3:
                adstar_action_loss_weight = max(.025, adstar_action_loss_weight * .7)
                task_reward_scale = min(2., task_reward_scale * 1.25)
                good_tracking_updates = 0
        write_checkpoint({"model_state_dict": model.state_dict(), "obs_dim": obs_dim,
                    "action_dim": action_dim, "action_kind": action_kind,
                    "architecture": architecture, "task": task,
                    "observation_normalization": OBS_NORMALIZATION,
                    "drivetrain_config": env.drivetrain_config})
        exported_path = routes[5][0] if routes is not None else []
        playback_frames.extend({"robots": [row[:3], row[3:6]],
            "goal": row[6:8], "sizes": row[8:12], "goal_radius":row[12],
            "adstar_path": exported_path} for row in torch.stack(update_frames).cpu().tolist())
        playback_frames = playback_frames[-480:]
        from .field import (ALLIANCE_ZONE_DEPTH, BUMP_ACCELERATION_SCALE,
                            BUMP_SPEED_SCALE, bump_boxes, static_collision_boxes)
        write_json(playback_path, {"task": task, "dt": env.dt, "field": {
            "length": env.sim.field_length, "width": env.sim.field_width,
            "alliance_zone_depth": ALLIANCE_ZONE_DEPTH,
            "elements": [box.as_dict() for box in env.field_boxes],
            "colliders": [box.as_dict() for box in static_collision_boxes(env.field_boxes)],
            "bump_regions": [box.as_dict() for box in bump_boxes(env.field_boxes)],
            "trench_paths": [box.as_dict() for box in env.field_boxes
                             if "_trench_" in box.name and "_support_" not in box.name],
            "trench_supports": [box.as_dict() for box in env.field_boxes
                                if "_trench_support_" in box.name],
            "bump_speed_scale": BUMP_SPEED_SCALE,
            "bump_acceleration_scale": BUMP_ACCELERATION_SCALE},
            "frames": playback_frames})
        completed = min((update_index + 1) * rollout_steps * num_envs, timesteps)
        elapsed = time.perf_counter() - start
        write_json(status_path, {**status_context, "status": "running", "task": task, "opponent": opponent,
            "stationary_opponent_fraction": STATIC_OPPONENT_FRACTION,
            "requested_timesteps": requested_timesteps_total,
            "completed_timesteps": completed_timesteps_base + completed,
            "num_envs": num_envs, "device": str(device),
            "observation_normalization": OBS_NORMALIZATION,
            "l2_coefficient": l2_coef,
            "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "updates": update_index + 1, "total_updates": updates,
            "adstar_tracking_error_m": tracking_error_ema,
            "adstar_tracking_error_update_m": update_tracking_error,
            "adstar_tracking_target_m": .35,
            "adstar_action_loss_weight": adstar_action_loss_weight,
            "task_reward_scale": task_reward_scale,
            "good_tracking_updates": good_tracking_updates,
                "elapsed_seconds": elapsed, "transitions_per_second": completed / max(elapsed, 1e-12),
                "strategic_rate_hz": strategic_rate_hz if architecture == "strategic_adstar" else 50.,
                "strategic_decisions_per_episode": math.ceil(horizon * strategic_rate_hz / 50.)
                    if architecture == "strategic_adstar" else horizon,
                "strategic_decisions_per_second": strategic_rate_hz if architecture == "strategic_adstar" else 50.,
                "simulated_seconds_trained": physics_ticks * env.dt * num_envs,
                "complete_matches_observed": complete_matches,
                "opponent_exposure_counts": opponent_exposure,
                "entropy": entropy_mean,
                "action_distribution": (action_counts / action_counts.sum().clamp_min(1)).detach().cpu().tolist(),
                "per_action_selection_rate": (action_counts / action_counts.sum().clamp_min(1)).detach().cpu().tolist(),
            "checkpoint": str(checkpoint), "started_at": started_at,
            "drivetrain_config": env.drivetrain_config})

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    write_checkpoint({"model_state_dict": model.state_dict(), "obs_dim": obs_dim,
                "action_dim": action_dim, "action_kind": action_kind,
                "architecture": architecture, "task": task,
                "observation_normalization": OBS_NORMALIZATION,
                "drivetrain_config": env.drivetrain_config})
    metadata = {"task": task, "architecture": architecture, "action_kind": action_kind,
                "observation_dim": obs_dim, "action_dim": action_dim,
                "opponent": opponent, "seed": seed,
                "stationary_opponent_fraction": STATIC_OPPONENT_FRACTION,
                "timesteps": updates * rollout_steps * num_envs, "requested_timesteps": timesteps,
                "reference_planner": "ADStarPlanner" if uses_adstar_teacher else None,
                "reference_worlds_per_update": min(64, num_envs) if uses_adstar_teacher else 0,
                "reference_reward_scale": .20 if uses_adstar_teacher else 0.,
                "reference_action_loss_weight_final": adstar_action_loss_weight,
                "task_reward_scale_final": task_reward_scale,
                "adstar_tracking_error_ema_m": tracking_error_ema,
                "num_envs": num_envs, "device": str(device), "checkpoint": str(checkpoint),
                "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "accelerator_backend": "ROCm" if torch.version.hip else ("CUDA" if torch.version.cuda else "CPU"),
                "domain_randomization": True,
                "observation_normalization": OBS_NORMALIZATION,
                "l2_coefficient": l2_coef,
                "elapsed_seconds": elapsed, "transitions_per_second": updates * rollout_steps * num_envs / max(elapsed, 1e-12),
                "completed_timesteps": updates * rollout_steps * num_envs,
                "strategic_rate_hz": strategic_rate_hz if architecture == "strategic_adstar" else 50.,
                "strategic_decisions_per_episode": math.ceil(horizon * strategic_rate_hz / 50.)
                    if architecture == "strategic_adstar" else horizon,
                "strategic_decisions_per_second": strategic_rate_hz if architecture == "strategic_adstar" else 50.,
                "simulated_seconds_trained": physics_ticks * env.dt * num_envs,
                "complete_matches_observed": complete_matches,
                "opponent_exposure_counts": opponent_exposure,
                "entropy": entropy_mean,
                "action_distribution": (action_counts / action_counts.sum().clamp_min(1)).detach().cpu().tolist(),
                "per_action_selection_rate": (action_counts / action_counts.sum().clamp_min(1)).detach().cpu().tolist(),
                "started_at": started_at,
                "completed_episodes_observed": complete_matches}
    metadata["drivetrain_config"] = env.drivetrain_config
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    write_json(status_path, {**status_context, **metadata, "status": "completed",
                             "requested_timesteps": requested_timesteps_total,
                             "completed_timesteps": completed_timesteps_base + updates * rollout_steps * num_envs,
                             "total_updates": updates, "updates": updates})
    return metadata


def train(task: str = "counter_defense", timesteps: int = 1_000_000,
          output: str | Path = "checkpoints/tensor-ppo", *, seed: int = 7,
          num_envs: int = 2048, device: str = "cuda", opponent: str | None = None,
          initial_checkpoint: str | Path | None = None,
          rollout_steps: int = 128, epochs: int = 4, minibatch_size: int = 4096,
          learning_rate: float = 3e-4, gamma: float = .993, gae_lambda: float = .95,
          clip_coef: float = .2, value_coef: float = .5, entropy_coef: float = .01,
          max_grad_norm: float = .5, l2_coef: float = 1e-5,
          horizon: int = 8000, algorithm: str = "generational", generations: int = 20,
          population_size: int = 8, elite_count: int = 2,
          architecture: str = "strategic_adstar", strategic_rate_hz: float = 4.) -> dict[str, Any]:
    """Train with PPO and a generation-managed opponent/checkpoint population."""
    if algorithm == "generational":
        return generational_train(task, generations, output, seed=seed,
            num_envs=num_envs, device=device, opponent=opponent,
            initial_checkpoint=initial_checkpoint, horizon=horizon, timesteps=timesteps,
            population_size=population_size, elite_count=elite_count,
            rollout_steps=rollout_steps, epochs=epochs, minibatch_size=min(minibatch_size, rollout_steps*num_envs),
            learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
            clip_coef=clip_coef, value_coef=value_coef, entropy_coef=entropy_coef,
            strategic_rate_hz=strategic_rate_hz,
            max_grad_norm=max_grad_norm, l2_coef=l2_coef, architecture=architecture)
    if algorithm != "ppo":
        raise ValueError("algorithm must be 'generational' or 'ppo'")
    if min(timesteps, num_envs, rollout_steps, epochs, minibatch_size, horizon) < 1:
        raise ValueError("timesteps, num_envs, rollout_steps, epochs, minibatch_size and horizon must be positive")
    if not math.isfinite(l2_coef) or l2_coef < 0:
        raise ValueError("l2_coef must be finite and non-negative")
    selected = _device(device)
    chosen_opponent = _task_opponent(task,opponent)
    kwargs = dict(task=task, timesteps=timesteps, output=output, seed=seed,
                  num_envs=num_envs, opponent=chosen_opponent, rollout_steps=rollout_steps,
                  initial_checkpoint=initial_checkpoint,
                  epochs=epochs, minibatch_size=min(minibatch_size, rollout_steps*num_envs),
                  learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
                  clip_coef=clip_coef, value_coef=value_coef, entropy_coef=entropy_coef,
                  max_grad_norm=max_grad_norm, l2_coef=l2_coef, horizon=horizon,
                  architecture=architecture, curriculum=(opponent is None or opponent == "mixed"),
                  strategic_rate_hz=strategic_rate_hz)
    return _train_once(device=selected, **kwargs)


def _opponent_population_paths(task: str, output: Path) -> list[Path]:
    """Find current and retained checkpoints for the opposing role."""
    peer_name = ("rebuilt-gamepiece-offense" if task == "defense"
                 else "rebuilt-gamepiece-defense")
    peer = output.parent / peer_name
    roots = [peer]
    default_roots = (("rebuilt-gamepiece-offense", "rebuilt-counter-defense")
                     if task == "defense" else
                     ("rebuilt-gamepiece-defense", "rebuilt-defense-generational", "rebuilt-defense"))
    roots.extend(Path("checkpoints") / name for name in default_roots)
    paths: list[Path] = []
    for root in roots:
        current = root / "policy.pt"
        if current.is_file():
            paths.append(current)
        paths.extend(sorted((root / "opponent-population").glob("*.pt")))
        paths.extend(sorted((root / "historical-opponents").glob("*.pt")))
    # Include other explicitly role-tagged historical policies without loading
    # unrelated checkpoint files into the opponent pool.
    opponent_task = "defense" if task == "counter_defense" else "counter_defense"
    for current in Path("checkpoints").glob("*/policy.pt"):
        for metadata_path in (current.parent / "metadata.json", current.parent / "status.json"):
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("task") == opponent_task:
                paths.append(current)
                break
    loadable: list[Path] = []
    for path in dict.fromkeys(path.resolve() for path in paths if path.is_file()):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            obs_dim = int(payload.get("obs_dim", OBS_DIM))
            action_dim = int(payload.get("action_dim", ACTION_DIM))
            if ("model_state_dict" in payload and obs_dim in
                    (OBS_DIM, STRATEGIC_LEGACY_OBS_DIM, 137) and action_dim in (3, 8)):
                loadable.append(path)
        except (OSError, RuntimeError, KeyError, ValueError, EOFError):
            continue
    return loadable


def _checkpoint_vector(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return torch.cat([value.detach().float().reshape(-1)
                      for value in payload["model_state_dict"].values()])


def _select_checkpoint_population(entries: list[dict[str, Any]], capacity: int,
                                   elite_count: int) -> list[dict[str, Any]]:
    """Keep high-scoring policies first, then add parameter-diverse candidates."""
    ranked = sorted(entries, key=lambda item: float(item.get("fitness", -math.inf)), reverse=True)
    if len(ranked) <= capacity:
        return ranked
    shortlist = ranked[:max(capacity * 4, elite_count)]
    selected = shortlist[:min(elite_count, capacity)]
    vectors = {item["checkpoint"]: _checkpoint_vector(Path(item["checkpoint"]))
               for item in shortlist}
    while len(selected) < capacity:
        remaining = [item for item in shortlist if item not in selected]
        if not remaining:
            break
        chosen = max(remaining, key=lambda item: (
            min(float(1. - torch.nn.functional.cosine_similarity(
                vectors[item["checkpoint"]], vectors[kept["checkpoint"]], dim=0))
                for kept in selected), float(item.get("fitness", -math.inf))))
        selected.append(chosen)
    return selected


def generational_train(task: str, generations: int, output: str | Path, *,
                       seed: int = 7, num_envs: int = 256, device: str = "cuda",
                       opponent: str | None = None, initial_checkpoint: str | Path | None = None,
                       horizon: int = 8000, population_size: int = 8,
                       elite_count: int = 2, l2_coef: float = 1e-5,
                       architecture: str = "strategic_adstar", curriculum: bool = True,
                       timesteps: int | None = None, rollout_steps: int = 128,
                       epochs: int = 4, minibatch_size: int = 4096,
                       learning_rate: float = 3e-4, gamma: float = .993,
                       gae_lambda: float = .95, clip_coef: float = .2,
                       value_coef: float = .5, entropy_coef: float = .01,
                       max_grad_norm: float = .5,
                       evaluation_episodes: int = 4,
                       strategic_rate_hz: float = 4.) -> dict[str, Any]:
    """Run PPO generations and manage a diverse, evaluated opponent population.

    Each generation continues the prior PPO checkpoint. One member is sampled
    from deterministic baselines and learned opponent checkpoints for training;
    the resulting policy is then evaluated against every fixed baseline and
    each available learned population member on a shared seeded batch.
    """
    if task not in ("counter_defense", "defense"):
        raise ValueError("task must be counter_defense or defense")
    if min(generations, num_envs, horizon, population_size, elite_count,
           rollout_steps, epochs, minibatch_size, evaluation_episodes) < 1:
        raise ValueError("generation, environment, horizon, population, PPO, and evaluation sizes must be positive")
    if elite_count > population_size:
        raise ValueError("elite_count cannot exceed opponent population capacity")
    if timesteps is not None and timesteps < 1:
        raise ValueError("timesteps must be positive")
    if l2_coef < 0 or not math.isfinite(l2_coef):
        raise ValueError("l2_coef must be finite and non-negative")
    selected_device = _device(device)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    archive_dir = out / "opponent-population"
    archive_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "policy.pt"
    status_path = out / "status.json"
    history_path = out / "generation-history.json"
    population_path = archive_dir / "population.json"
    history: list[dict[str, Any]] = []
    try:
        prior = json.loads(history_path.read_text())
        history = list(prior.get("generations", []))
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    previous_checkpoint: str | Path | None = checkpoint if checkpoint.is_file() else initial_checkpoint
    expected_obs = {"direct": OBS_DIM, "tactical_adstar": 38,
                    "strategic_adstar": 137}.get(architecture)
    expected_action = {"direct": ACTION_DIM, "tactical_adstar": 2,
                       "strategic_adstar": STRATEGIC_ACTION_DIM}.get(architecture)
    if previous_checkpoint is not None and expected_obs is not None:
        prior_payload = None
        try:
            prior_payload = torch.load(previous_checkpoint, map_location="cpu", weights_only=True)
            compatible = (prior_payload.get("architecture", "direct") == architecture
                and int(prior_payload.get("obs_dim", OBS_DIM)) == expected_obs
                and int(prior_payload.get("action_dim", ACTION_DIM)) == expected_action)
        except (OSError, RuntimeError, KeyError, ValueError, EOFError):
            compatible = False
        if not compatible:
            # Keep older strategic checkpoints as learned opponents. The new
            # 137-input/8-action policy cannot safely load 89-input/3-action
            # weights, so its first PPO generation starts with a fresh head.
            historical = out / "historical-opponents"
            historical.mkdir(parents=True, exist_ok=True)
            source = Path(previous_checkpoint)
            backup = historical / f"pre-ppo-{source.parent.name}.pt"
            if prior_payload is not None and source.is_file() and source.resolve() != backup.resolve():
                shutil.copy2(source, backup)
            previous_checkpoint = None
    start_generation = len(history)
    if start_generation >= generations:
        start_generation = 0
        history = []
    if not 2. <= strategic_rate_hz <= 5.:
        raise ValueError("strategic_rate_hz must be between 2 and 5")
    decisions_per_episode = (math.ceil(horizon * strategic_rate_hz / 50.)
                             if architecture == "strategic_adstar" else horizon)
    per_generation_steps = (max(num_envs * decisions_per_episode, timesteps // generations)
                            if timesteps is not None else
                            num_envs * max(decisions_per_episode, rollout_steps))
    per_generation_steps = max(num_envs * rollout_steps, per_generation_steps)
    rollout_batch = num_envs * rollout_steps
    per_generation_steps = math.ceil(per_generation_steps / rollout_batch) * rollout_batch
    if task == "counter_defense":
        fixed_baselines = ("guard", "adstar_defender", "intercept", "lane_block",
                           "fuel_denial", "shadow", "hub_guard")
    else:
        fixed_baselines = ("offense", "adstar", "mirror", "cutoff", "velocity_intercept")
    evaluation_count = max(1, min(evaluation_episodes, num_envs))
    started_at = time.time()
    started = time.perf_counter()
    rng = random.Random(seed)
    requested_timesteps = per_generation_steps * generations
    mode_counts: dict[str, int] = {}
    last_opponent_results: dict[str, Any] = {}
    archive_entries: list[dict[str, Any]] = []
    try:
        archive_entries = json.loads(population_path.read_text()).get("members", [])
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    archive_root = archive_dir.resolve()
    archive_entries = [item for item in archive_entries
        if isinstance(item, dict)
        and Path(str(item.get("checkpoint", ""))).parent.resolve() == archive_root
        and Path(str(item.get("checkpoint", ""))).is_file()]
    training_specs: list[dict[str, Any]] = []

    def learned_member_name(path: Path) -> str:
        owner = path.parent.parent.name if path.parent.name in (
            "opponent-population", "historical-opponents") else path.parent.name
        source = path.parent.name if owner != path.parent.name else "current"
        return f"NN · {owner} · {source} · {path.stem}"

    def write_json(path: Path, value: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2, sort_keys=True))
        temp.replace(path)

    for generation in range(start_generation, generations):
        generation_number = generation + 1
        # Refresh the peer pool every generation so both role trainers see the
        # other's newest checkpoint as well as its selected historical members.
        peer_paths = _opponent_population_paths(task, out)
        training_specs = [{"name": mode, "mode": mode, "checkpoint": None}
                          for mode in fixed_baselines]
        training_specs.extend({"name": learned_member_name(path), "mode": "learned",
                               "checkpoint": str(path)} for path in peer_paths)
        if opponent and opponent != "mixed" and opponent not in {item["mode"] for item in training_specs}:
            training_specs.insert(0, {"name": opponent, "mode": _task_opponent(task, opponent),
                                      "checkpoint": None})
        train_spec = training_specs[rng.randrange(len(training_specs))]
        mode_counts[train_spec["name"]] = mode_counts.get(train_spec["name"], 0) + 1
        status_context = {
            "algorithm": "generational", "optimizer": "PPO",
            "generation": generation_number, "current_generation": generation_number,
            "total_generations": generations, "population_size": population_size,
            "elite_count": elite_count, "opponent_member": train_spec["name"],
            "opponent_population_members": [item["name"] for item in training_specs],
            "opponents": sorted({item["mode"] for item in training_specs}),
            "completed_timesteps_base": generation * per_generation_steps,
            "requested_timesteps_total": requested_timesteps,
        }
        ppo_result = _train_once(task=task, timesteps=per_generation_steps,
            output=out, seed=seed + generation * 1009, num_envs=num_envs,
            device=selected_device, opponent=train_spec["mode"],
            initial_checkpoint=previous_checkpoint, rollout_steps=rollout_steps,
            epochs=epochs, minibatch_size=min(minibatch_size, rollout_steps * num_envs),
            learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
            clip_coef=clip_coef, value_coef=value_coef, entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm, l2_coef=l2_coef, horizon=horizon,
            architecture=architecture, curriculum=True,
            strategic_rate_hz=strategic_rate_hz,
            learned_opponent_checkpoint=train_spec["checkpoint"],
            learned_opponent_checkpoints=[item["checkpoint"] for item in training_specs
                if item["mode"] == "learned"],
            status_context=status_context)
        previous_checkpoint = checkpoint
        scenario_seed = 500_000 + generation * 1009
        evaluation_dir = out / "generation-evaluations" / f"generation-{generation_number:04d}"
        opponent_results: dict[str, Any] = {}
        evaluation_specs = [{"name": mode, "mode": mode, "checkpoint": None}
                            for mode in fixed_baselines]
        evaluation_specs.extend({"name": learned_member_name(path), "mode": "learned",
                                 "checkpoint": str(path)} for path in peer_paths)
        for spec in evaluation_specs:
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", spec["name"]).strip("-")
            metrics_path = evaluation_dir / f"{safe_name}.json"
            try:
                metrics = _evaluate_once(checkpoint, task, evaluation_count, scenario_seed,
                    evaluation_count, selected_device,
                    spec["mode"], metrics_path, horizon=horizon,
                    opponent_checkpoint=spec["checkpoint"])
                own_scores = float(metrics.get("mean_scores") or 0.)
                other_scores = float(metrics.get("mean_opponent_scores") or 0.)
                opponent_results[spec["name"]] = {
                    "opponent": spec["mode"], "checkpoint": spec["checkpoint"],
                    "mean_return": metrics.get("mean_return"),
                    "success_rate": metrics.get("mean_success"),
                    "win_rate": metrics.get("mean_success"),
                    "score": own_scores, "score_differential": own_scores - other_scores,
                    "fuel_acquired": metrics.get("mean_acquisitions"),
                    "fuel_scored": metrics.get("mean_scores"),
                    "cycle_time": metrics.get("mean_cycle_time"),
                    "defensive_delay": metrics.get("mean_defensive_delay"),
                    "denial_rate": metrics.get("mean_denied_objectives"),
                    "denied_per_episode": metrics.get("mean_denied_objectives"),
                    "pass_success": (None if metrics.get("mean_passes_attempted") in (None, 0)
                        else float(metrics.get("mean_passes_completed") or 0.) /
                             float(metrics["mean_passes_attempted"])),
                    "mean_opponent_score": metrics.get("mean_opponent_scores"),
                    "seed": scenario_seed, "episodes": evaluation_count,
                    "horizon": metrics.get("horizon", horizon),
                    "metrics_file": str(metrics_path)}
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                opponent_results[spec["name"]] = {"opponent": spec["mode"],
                    "checkpoint": spec["checkpoint"], "seed": scenario_seed,
                    "episodes": evaluation_count, "horizon": horizon,
                    "error": str(exc)}
        last_opponent_results = opponent_results
        successful = [item for item in opponent_results.values() if "error" not in item]
        fitness_values = [float(item.get("success_rate") or item.get("mean_return") or 0.)
                          for item in successful]
        generation_fitness = sum(fitness_values) / max(1, len(fitness_values))
        snapshot = archive_dir / f"generation-{generation_number:04d}.pt"
        temp_snapshot = snapshot.with_suffix(".pt.tmp")
        shutil.copy2(checkpoint, temp_snapshot)
        temp_snapshot.replace(snapshot)
        archive_entries = [item for item in archive_entries
                           if item.get("checkpoint") != str(snapshot.resolve())]
        archive_entries.append({"generation": generation_number,
            "checkpoint": str(snapshot.resolve()), "fitness": generation_fitness,
            "evaluation_seed": scenario_seed,
            "evaluated_opponents": sorted(opponent_results)})
        kept = _select_checkpoint_population(archive_entries, population_size, elite_count)
        kept_paths = {str(Path(item["checkpoint"]).resolve()) for item in kept}
        for item in archive_entries:
            path = Path(item["checkpoint"])
            if str(path.resolve()) not in kept_paths:
                path.unlink(missing_ok=True)
        archive_entries = kept
        write_json(population_path, {"task": task, "capacity": population_size,
            "strong_elites": min(elite_count, population_size), "members": archive_entries})
        generation_record = {"generation": generation_number,
            "training_optimizer": "PPO", "training_opponent": train_spec["name"],
            "training_opponent_checkpoint": train_spec["checkpoint"],
            "ppo_timesteps": ppo_result["completed_timesteps"],
            "strategic_rate_hz": ppo_result.get("strategic_rate_hz"),
            "strategic_decisions_per_second": ppo_result.get("strategic_decisions_per_second"),
            "strategic_decisions_per_episode": ppo_result.get("strategic_decisions_per_episode"),
            "simulated_seconds_trained": ppo_result.get("simulated_seconds_trained"),
            "complete_matches_observed": ppo_result.get("complete_matches_observed"),
            "opponent_exposure_counts": ppo_result.get("opponent_exposure_counts"),
            "entropy": ppo_result.get("entropy"),
            "action_distribution": ppo_result.get("action_distribution"),
            "per_action_selection_rate": ppo_result.get("per_action_selection_rate"),
            "scenario_seed": scenario_seed,
            "fitness": generation_fitness, "opponent_metrics": opponent_results,
            "fixed_baseline_metrics": {name: opponent_results[name] for name in fixed_baselines
                if name in opponent_results},
            "score_differential_by_opponent": {name: item.get("score_differential")
                for name, item in opponent_results.items()},
            "population_members": [item["checkpoint"] for item in archive_entries],
            "population_size": len(archive_entries)}
        history = [item for item in history if int(item.get("generation", 0)) != generation_number]
        history.append(generation_record)
        write_json(history_path, {"algorithm": "ppo_opponent_population",
            "task": task, "evaluation_seed_base": 500_000 + generation * 1009,
            "generations": history})
        elapsed = time.perf_counter() - started
        final_status = {**ppo_result, "status": "running" if generation_number < generations else "completed",
            "algorithm": "generational", "optimizer": "PPO",
            "task": task, "policy_modes": (["INTAKE", "DELIVER"] if task == "counter_defense" else ["DEFENSE"]),
            "generation": generation_number,
            "current_generation": generation_number, "total_generations": generations,
            "population_size": len(archive_entries), "population_capacity": population_size,
            "elite_count": min(elite_count, population_size),
            "opponent": "learned" if train_spec["mode"] == "learned" else train_spec["mode"],
            "opponent_member": train_spec["name"],
            "opponents": sorted({item["mode"] for item in training_specs}),
            "opponent_population_members": [item["name"] for item in training_specs
                if item["mode"] == "learned"],
            "opponent_weights": mode_counts,
            "generation_history": history, "generation_metrics": generation_record,
            "opponent_metrics": opponent_results,
            "fixed_baseline_metrics": generation_record["fixed_baseline_metrics"],
            "best_fitness": max((float(item.get("fitness", -math.inf)) for item in history), default=generation_fitness),
            "best_population_generation": max(history, key=lambda item: float(item.get("fitness", -math.inf))).get("generation"),
            "best_population_fitness": max((float(item.get("fitness", -math.inf)) for item in history), default=generation_fitness),
            "requested_timesteps": requested_timesteps,
            "completed_timesteps": generation_number * per_generation_steps,
            "elapsed_seconds": elapsed,
            "transitions_per_second": generation_number * per_generation_steps / max(elapsed, 1e-12),
            "population_manifest": str(population_path), "generation_history_path": str(history_path)}
        write_json(status_path, final_status)
    final = {**final_status, "status": "completed", "generation_history": history,
        "generation_history_path": str(history_path), "population_manifest": str(population_path),
        "completed_timesteps": generations * per_generation_steps,
        "requested_timesteps": requested_timesteps,
        "elapsed_seconds": time.perf_counter() - started,
        "opponent_metrics": last_opponent_results,
        "opponent_fitness": {name: values.get("success_rate", values.get("mean_return"))
                              for name, values in last_opponent_results.items()},
        "opponents": sorted({item["mode"] for item in training_specs}),
        "opponent_population_members": [item["name"] for item in training_specs
            if item["mode"] == "learned"]}
    write_json(status_path, final)
    write_json(out / "metadata.json", final)
    return final

def evaluate(checkpoint: str | Path, *, task: str = "counter_defense",
             episodes: int = 32, seed: int = 1000, num_envs: int = 32,
             device: str = "cuda", opponent: str = "random",
             output: str | Path = "metrics/tensor-evaluation.json",
             horizon: int = 8000) -> dict[str, Any]:
    """Run deterministic policy evaluation and write a compact JSON report."""
    if episodes < 1 or num_envs < 1 or horizon < 1:
        raise ValueError("episodes, num_envs and horizon must be positive")
    if task == "defense" and opponent == "mixed":
        root = Path(output)
        results = {}
        matched_num_envs = max(num_envs, episodes)
        for mode in MIXED_DEFENSE_OPPONENTS:
            mode_output = root.parent / f"{root.stem}-{mode}" / "metrics.json"
            results[mode] = evaluate(checkpoint, task=task, episodes=episodes, seed=seed,
                num_envs=matched_num_envs, device=device, opponent=mode, output=mode_output,
                horizon=horizon)
        by_opponent = {}
        for mode, metrics in results.items():
            hold_rate = float(metrics.get("mean_success", 0.))
            wins = int(round(hold_rate * episodes))
            lower, upper = _wilson_interval(wins, episodes)
            score_rate = 1. - hold_rate
            mean_score_time = (float(metrics.get("mean_time_to_goal", 0.)) / score_rate
                               if score_rate > 0. else None)
            by_opponent[mode] = {"hold_rate": hold_rate,
                "hold_count": wins, "score_count": episodes - wins,
                "hold_rate_95ci": [lower, upper],
                "attacker_score_rate": score_rate,
                "mean_score_time": mean_score_time,
                "episodes": episodes, "mean_return": metrics.get("mean_return"),
                "mean_episode_length": metrics.get("mean_episode_length"),
                "metrics_file": str(root.parent / f"{root.stem}-{mode}" / "metrics.json")}
        passed = by_opponent["adstar"]["hold_rate"] >= .70 and by_opponent["offense"]["hold_rate"] >= .70
        report = {"task": task, "opponent": "mixed", "checkpoint": str(checkpoint),
            "episodes_per_opponent": episodes, "matched_envs": matched_num_envs,
            "seed": seed, "horizon": horizon,
            "opponent_results": by_opponent, "acceptance_threshold": .70,
            "acceptance_passed": passed,
            "acceptance_required_opponents": ["adstar", "offense"]}
        root.parent.mkdir(parents=True, exist_ok=True)
        root.write_text(json.dumps(report, indent=2, sort_keys=True))
        return report
    selected = _device(device)
    opponent = _task_opponent(task,opponent)
    return _evaluate_once(checkpoint, task, episodes, seed, num_envs,
                          selected, opponent, output, horizon=horizon)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0., 1.
    p = successes / total
    denom = 1. + z * z / total
    center = (p + z * z / (2. * total)) / denom
    radius = z * math.sqrt(p * (1. - p) / total + z * z / (4. * total * total)) / denom
    return max(0., center - radius), min(1., center + radius)


def _scripted_game_action(env, task: str) -> torch.Tensor:
    """Small game-state strategy whose navigation is delegated to simulator AD*."""
    if task == "defense":
        # Deny a visible loose FUEL when available; otherwise block the likely
        # scoring lane. This strategy sees no hidden objective.
        candidates,valid,_=env._fuel_candidates(0)
        slot=valid.to(torch.int64).argmax(-1)
        return torch.where(valid.any(-1),slot+1,torch.full_like(slot,5))
    has_piece = env._own_possession(0) > 0
    can_score = has_piece
    candidates,valid,_=env._fuel_candidates(0)
    collect=valid.to(torch.int64).argmax(-1)
    # Explicit candidate rank 0..3 collects; choice 4 scores. If no loose
    # FUEL is available, select the intermediate-objective behavior.
    return torch.where(can_score,torch.full_like(collect,4),
        torch.where(valid.any(-1),collect,torch.full_like(collect,6)))


def _legacy_strategic_action(action: torch.Tensor, task: str) -> torch.Tensor:
    """Map a historical three-class checkpoint onto the candidate action set."""
    legacy=action.long().clamp(0,2)
    if task=="counter_defense":  # offense: collect, score, tactical
        return torch.where(legacy==0,torch.zeros_like(legacy),
            torch.where(legacy==1,torch.full_like(legacy,4),torch.full_like(legacy,6)))
    # defense: block, deny, intercept
    return torch.where(legacy==0,torch.full_like(legacy,5),
        torch.where(legacy==1,torch.ones_like(legacy),torch.zeros_like(legacy)))


def _evaluate_once(checkpoint, task, episodes, seed, num_envs, device, opponent,
                   output, horizon=8000, scripted_strategy=None,
                   opponent_checkpoint=None):
    payload = (torch.load(checkpoint, map_location=device, weights_only=True)
               if checkpoint is not None else None)
    architecture = (payload.get("architecture", "direct") if payload is not None
                    else "strategic_adstar")
    action_kind = (payload.get("action_kind", "continuous") if payload is not None
                   else "categorical")
    model = None
    if payload is not None:
        model = ActorCritic(payload.get("obs_dim", OBS_DIM),
                            payload.get("action_dim", ACTION_DIM), action_kind).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
    scenario_opponent = ("learned" if opponent_checkpoint else
        "adstar" if task == "defense" and opponent in MIXED_DEFENSE_OPPONENTS else opponent)
    env = _env(num_envs, task, device, seed, scenario_opponent, horizon,
               normalize_observations=bool(payload and
                   payload.get("observation_normalization") == OBS_NORMALIZATION),
               architecture=architecture)
    horizon=int(env.horizon)
    # Planner routes must not determine the defender's initial position or
    # enter its policy inputs during defense evaluation.
    if task == "defense":
        env.adstar_spawn_hint = False
    env.opponent = "learned" if opponent_checkpoint else opponent
    if opponent_checkpoint and not _attach_historical_opponent(
            env, task, device, opponent_checkpoint):
        raise ValueError(f"could not load opponent checkpoint: {opponent_checkpoint}")
    checkpoint_drivetrain_config = payload.get("drivetrain_config") if payload else None
    if (checkpoint_drivetrain_config is not None and
            checkpoint_drivetrain_config != env.drivetrain_config):
        raise ValueError("checkpoint drivetrain configuration does not match this simulator; "
                         "use the same FRC_DRIVETRAIN_CONFIG used during training")
    is_adstar_defense = task == "defense" and opponent == "adstar"
    # The traditional "guard" mode is now the same obstacle-aware AD* policy;
    # keep its public opponent name so existing training/evaluation commands
    # and dashboard links continue to work.
    is_adstar_defender = task == "counter_defense" and opponent in ("guard", "adstar_defender")
    uses_adstar_playback_route = is_adstar_defense or is_adstar_defender
    run_dir = Path(output).parent
    status_path = run_dir / "status.json"
    playback_path = run_dir / "playback.json"
    playback_frames = []
    eval_start = time.perf_counter()

    def atomic_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(path)

    from .field import (ALLIANCE_ZONE_DEPTH, BUMP_ACCELERATION_SCALE,
                        BUMP_SPEED_SCALE, bump_boxes, static_collision_boxes)
    playback_field = {"length": env.sim.field_length, "width": env.sim.field_width,
        "alliance_zone_depth": ALLIANCE_ZONE_DEPTH,
        "elements": [box.as_dict() for box in env.field_boxes],
        "colliders": [box.as_dict() for box in static_collision_boxes(env.field_boxes)],
        "bump_regions": [box.as_dict() for box in bump_boxes(env.field_boxes)],
        "trench_paths": [box.as_dict() for box in env.field_boxes if "_trench_" in box.name and "_support_" not in box.name],
        "trench_supports": [box.as_dict() for box in env.field_boxes if "_trench_support_" in box.name],
        "bump_speed_scale": BUMP_SPEED_SCALE,
        "bump_acceleration_scale": BUMP_ACCELERATION_SCALE}

    playback_scenarios = []
    playback_scenario_done = []
    # Every dashboard mode records the same seeded scenario batch. Endpoint
    # generation lives in TensorDefenseEnv, so policy and opponent modes differ
    # only in motion, never in scenario sampling.
    playback_scenario_count = min(3, num_envs, episodes)
    if playback_scenario_count:
        playback_scenarios = [{"id": str(i + 1), "label": f"Scenario {i + 1}", "frames": []}
            for i in range(playback_scenario_count)]
        playback_scenario_done = [False] * playback_scenario_count

    def write_playback():
        atomic_json(playback_path, {"task": "adstar_attacker_defense" if is_adstar_defense else task,
                                    "scenario_seed": seed, "dt": env.dt,
                                    "field": playback_field, "frames": playback_frames,
                                    "scenarios": playback_scenarios})

    def eval_status(status):
        hold_rate = sum(metric_rows["success"]) / max(1, len(metric_rows["success"]))
        score_times = [value for value in metric_rows["time_to_goal"] if value > 0]
        value = {"status": status, "task": "defense", "opponent": "adstar",
                 "checkpoint": str(checkpoint) if checkpoint else None, "device": str(device),
                 "checkpoint_mtime": Path(checkpoint).stat().st_mtime if checkpoint else None,
                 "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                 "accelerator_backend": "ROCm" if torch.version.hip else ("CUDA" if torch.version.cuda else "CPU"),
                 "num_envs": num_envs,
                 "progress_unit": "episodes", "completed_timesteps": len(scores),
                 "requested_timesteps": episodes, "completed_episodes": len(scores),
                 "requested_episodes": episodes, "adstar_defender_hold_rate": hold_rate,
                 "adstar_attacker_score_rate": len(score_times) / max(1, len(scores)),
                 "adstar_mean_attack_time_to_score": sum(score_times) / max(1, len(score_times)) if score_times else None,
                 "adstar_replan_interval_steps": env.adstar_replan_interval,
                 "elapsed_seconds": time.perf_counter() - eval_start,
                 "drivetrain_config": env.drivetrain_config}
        atomic_json(status_path, value)

    if is_adstar_defender:
        atomic_json(status_path, {"status": "running", "task": task,
            "opponent": opponent, "checkpoint": str(checkpoint),
            "device": str(device), "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda" else "CPU", "num_envs": num_envs,
            "progress_unit": "episodes", "completed_timesteps": 0,
            "requested_timesteps": episodes, "started_at": time.time(),
            "drivetrain_config": env.drivetrain_config,
            "checkpoint_drivetrain_config": checkpoint_drivetrain_config})

    obs = _reset_obs(env.reset()).to(device=device, dtype=torch.float32)
    planning_latencies = []
    planning_calls = 0
    for planner_name in ("_adstar_planners", "_adstar_defender_planners",
                         "_adstar_tactical_planner"):
        planner = getattr(env, planner_name, None)
        if planner is None or not callable(getattr(planner, "plan", None)):
            continue
        original_plan = planner.plan
        def timed_plan(*args, _original=original_plan, **kwargs):
            nonlocal planning_calls
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = _original(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            planning_latencies.append((time.perf_counter() - started) * 1000.)
            planning_calls += 1
            return result
        try:
            planner.plan = timed_plan
        except (AttributeError, TypeError):
            pass
    start_robot = 0 if task == "counter_defense" else 1
    for i, scenario in enumerate(playback_scenarios):
        scenario["start"] = env.sim.pose[i, start_robot, :2].detach().cpu().tolist()
        scenario["goal"] = (None if architecture == "strategic_adstar"
                             else env.goal[i].detach().cpu().tolist())
    playback_reference_routes = []
    if not uses_adstar_playback_route and task != "defense":
        *_, playback_reference_routes = _adstar_reference(
            env, device, max_worlds=playback_scenario_count,
            world_indices=range(playback_scenario_count), include_all_routes=True)
    totals = torch.zeros(num_envs, device=device)
    lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    base_quota, extra_quota = divmod(episodes, num_envs)
    episode_quota = torch.full((num_envs,), base_quota, dtype=torch.long, device=device)
    if extra_quota:
        episode_quota[:extra_quota] += 1
    episode_counts = torch.zeros((num_envs,), dtype=torch.long, device=device)
    active = episode_quota > 0
    scores, episode_lengths = [], []
    metric_names = ("contact", "opponent_contact", "field_contact", "wall_contact",
                    "opponent_field_contact", "opponent_wall_contact", "opponent_static_contact",
                    "static_contact_started", "contact_duration", "contact_count", "time_blocked", "time_to_goal",
                    "success", "defensive_delay", "useful_position", "path_length",
                    "command_smoothness", "spin_rate_ratio", "maneuver_penalty",
                    "path_efficiency", "out_of_bounds")
    metric_totals = {name: torch.zeros(num_envs,device=device) for name in metric_names}
    metric_rows = {name: [] for name in metric_names}
    game_event_names = ("fuel_acquired_event", "fuel_scored_event",
                        "fuel_denied_event", "fuel_abandoned_event")
    game_event_totals = {name: torch.zeros((num_envs, 2), device=device)
                         for name in game_event_names}
    game_metric_rows = {name: [] for name in (
        "acquisitions", "scores", "denied_objectives", "abandoned_objectives",
        "cycle_time", "opponent_acquisitions", "opponent_scores",
        "opponent_denied_objectives", "opponent_abandoned_objectives",
        "total_simulated_score", "passes_attempted", "passes_completed",
        "fuel_transferred", "failed_passes", "pass_to_score_cycle_time")}
    acquired_at = torch.zeros((num_envs, env.piece_owner.shape[1], 2), device=device)
    previous_piece_owner = env.piece_owner.clone()
    cycle_time_sum = torch.zeros((num_envs, 2), device=device)
    cycle_count = torch.zeros((num_envs, 2), device=device)
    inference_latencies = []
    run_dir.mkdir(parents=True, exist_ok=True)
    write_playback()
    if is_adstar_defense:
        eval_status("evaluating")
    while len(scores) < episodes:
        if model is not None:
            inference_start=time.perf_counter()
            with torch.no_grad():
                model_obs=obs[:,:model.trunk[0].in_features]
                action_mask=(model_obs[:,-STRATEGIC_ACTION_DIM:].bool() if action_kind=="categorical" and
                    model.actor.out_features==STRATEGIC_ACTION_DIM else None)
                actions, _, _ = model.sample(model_obs, deterministic=True,action_mask=action_mask)
                if env.action_mode=="strategic" and model.actor.out_features==3:
                    actions=_legacy_strategic_action(actions,task)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_latencies.append((time.perf_counter()-inference_start)*1000.)
        elif scripted_strategy in ("offense", "defense"):
            actions = _scripted_game_action(env, task)
        else:
            raise ValueError("evaluation needs a valid checkpoint or an explicit scripted strategy")
        next_obs, rewards, dones, truncated, info = env.step(actions)
        active_float = active.to(device=device, dtype=torch.float32)
        totals += rewards.to(device=device, dtype=torch.float32).reshape(num_envs) * active_float
        lengths += active.to(device=device, dtype=torch.long)
        for name in metric_names:
            value=info.get(name)
            if value is not None:
                metric_totals[name]+=torch.as_tensor(value,device=device,dtype=torch.float32).reshape(num_envs) * active_float
        for name in game_event_names:
            value = info.get(name)
            if value is not None:
                game_event_totals[name] += torch.as_tensor(value, device=device,
                    dtype=torch.float32).reshape(num_envs, 2) * active_float[:, None]
        current_owner = env.piece_owner
        # Use simulated seconds for robot acquisition-to-score cycles; the
        # strategic match clock may be compressed to fit the rollout horizon.
        episode_clock = lengths.to(dtype=torch.float32) * env.dt
        for robot in (0, 1):
            newly_acquired = (current_owner == robot) & (previous_piece_owner != robot)
            acquired_at[:, :, robot] = torch.where(newly_acquired & active[:, None],
                episode_clock[:, None], acquired_at[:, :, robot])
            just_scored = (current_owner == -2) & (previous_piece_owner == robot)
            elapsed_held = (episode_clock[:, None] - acquired_at[:, :, robot]).clamp_min(0.)
            cycle_time_sum[:, robot] += (elapsed_held * just_scored.float()).sum(-1) * active_float
            cycle_count[:, robot] += just_scored.sum(-1).float() * active_float
            released = (previous_piece_owner == robot) & (current_owner != robot)
            acquired_at[:, :, robot] = torch.where(released, torch.zeros_like(acquired_at[:, :, robot]),
                                                    acquired_at[:, :, robot])
        previous_piece_owner.copy_(current_owner)
        finished = active & (dones.to(device=device, dtype=torch.bool).reshape(num_envs) | truncated.to(device=device, dtype=torch.bool).reshape(num_envs))
        if playback_scenario_count:
            # Keep a few independent worlds as separate rollouts; never splice
            # parallel worlds into one playback timeline.
            sample_finished = False
            wrote_frame = False
            for i in range(playback_scenario_count):
                if playback_scenario_done[i]:
                    continue
                sample_finished = bool(finished[i].item())
                step_index = int(lengths[i].item())
                if step_index % 3 == 0 or sample_finished:
                    paths = info.get("adstar_paths", [])
                    path_lengths = info.get("adstar_path_lengths", [])
                    intercepts = info.get("predicted_intercepts", [])
                    intercept_times = info.get("predicted_intercept_times", [])
                    if uses_adstar_playback_route and len(paths):
                        path_count=int(path_lengths[i].item()) if len(path_lengths) else paths.shape[1]
                        route=paths[i,:path_count].detach().cpu().tolist()
                        predicted_intercept=intercepts[i].detach().cpu().tolist() if is_adstar_defense and len(intercepts) else None
                        predicted_intercept_time=float(intercept_times[i].item()) if is_adstar_defense and len(intercept_times) else None
                    else:
                        route=playback_reference_routes[i] if playback_reference_routes else []
                        predicted_intercept=None
                        predicted_intercept_time=None
                    pose = env.sim.pose[i].detach().cpu().tolist()
                    sizes = torch.stack((env.sim.length[i], env.sim.width[i]), -1).reshape(-1).detach().cpu().tolist()
                    own_effort=((actions[i, :2] * env.sim.speed[i, 0]).detach().cpu().tolist()
                                if actions.ndim >= 2 and actions.shape[-1] >= 2
                                else env.sim.velocity[i, 0, :2].detach().cpu().tolist())
                    game_frame={}
                    for key, attribute in (("hub_active", "hub_active"),
                                           ("match_remaining", "match_remaining"), ("match_elapsed", "match_elapsed"),
                                           ("fuel_score_count", "fuel_score_count")):
                        value=getattr(env,attribute,None)
                        if value is not None:
                            item=value[i]
                            game_frame[key]=item.detach().cpu().tolist() if isinstance(item,torch.Tensor) else item
                    if env.action_mode == "strategic":
                        game_frame["hub_centers"] = env.hub_centers.detach().cpu().tolist()
                        active_fuel = env.piece_active[i]
                        fuel_pieces = torch.cat((env.piece_pos[i],
                            env.piece_owner[i,:,None].to(env.piece_pos.dtype)), -1)
                        game_frame["fuel_pieces"] = fuel_pieces[active_fuel].detach().cpu().tolist()
                        controlled = info.get("controlled_adstar_path", [])
                        controlled_lengths = info.get("controlled_adstar_path_lengths", [])
                        if len(controlled):
                            own_count = (int(controlled_lengths[i].item()) if len(controlled_lengths)
                                         else controlled.shape[1])
                            own_path = controlled[i,:own_count].detach().cpu().tolist()
                            game_frame["adstar_paths"] = [own_path, route]
                    strategic = env.action_mode == "strategic"
                    frame = {"robots": pose,
                    "goal": None if strategic else env.goal[i].detach().cpu().tolist(),
                    "sizes": sizes, "goal_radius": None if strategic else float(env.goal_radius[i].item()),
                    "chassis_effort_vector": own_effort,
                    "robot_effort_vectors": [
                        own_effort,
                        info.get("opponent_effort_vector", torch.zeros((num_envs, 2), device=device))[i].detach().cpu().tolist()],
                    **({} if strategic else {"adstar_path": [list(point) for point in route]}),
                    "predicted_intercept": predicted_intercept,
                    "predicted_intercept_time": predicted_intercept_time}
                    frame.update(game_frame)
                    playback_scenarios[i]["frames"].append(frame)
                    if i == 0:
                        playback_frames.append(frame)
                    wrote_frame = True
                if sample_finished:
                    playback_scenario_done[i] = True
            if wrote_frame and (len(playback_scenarios[0]["frames"]) % 10 == 0 or any(playback_scenario_done)):
                write_playback()
        for index in torch.nonzero(finished, as_tuple=False).flatten():
            if len(scores) < episodes:
                i = int(index.item())
                scores.append(float(totals[i].item()))
                length_i=int(lengths[i].item())
                episode_lengths.append(length_i)
                episode_counts[i] += 1
                for name in metric_names:
                    value=float(metric_totals[name][i].item())
                    metric_rows[name].append(value/length_i if name=="contact" else value)
                for event_name, row_name in (("fuel_acquired_event", "acquisitions"),
                    ("fuel_scored_event", "scores"), ("fuel_denied_event", "denied_objectives")):
                    game_metric_rows[row_name].append(float(game_event_totals[event_name][i, 0].item()))
                    opponent_row = "opponent_" + row_name
                    game_metric_rows[opponent_row].append(float(game_event_totals[event_name][i, 1].item()))
                game_metric_rows["abandoned_objectives"].append(None)
                game_metric_rows["opponent_abandoned_objectives"].append(None)
                cycle_n = float(cycle_count[i, 0].item())
                game_metric_rows["cycle_time"].append(
                    float(cycle_time_sum[i, 0].item()) / cycle_n if cycle_n else None)
                game_metric_rows["total_simulated_score"].append(
                    float(game_event_totals["fuel_scored_event"][i].sum().item()))
                # This environment is currently 1v1 and has no allied receiver;
                # report passing as unavailable rather than silently implying
                # that a no-pass result is a learned behavior.
                for name in ("passes_attempted","passes_completed","fuel_transferred",
                             "failed_passes","pass_to_score_cycle_time"):
                    game_metric_rows[name].append(None)
                totals[i] = 0
                lengths[i] = 0
                for values in metric_totals.values():
                    values[i]=0
                for values in game_event_totals.values():
                    values[i].zero_()
                acquired_at[i].zero_()
                cycle_time_sum[i].zero_()
                cycle_count[i].zero_()
                previous_piece_owner[i].fill_(-1)
                if is_adstar_defense:
                    eval_status("evaluating")
        reset_done = getattr(env, "reset_done", None)
        if reset_done is None:
            raise RuntimeError("TensorDefenseEnv must provide reset_done(mask) for per-world episode resets")
        active &= ~finished
        reset_mask = finished & (episode_counts < episode_quota)
        if len(scores) < episodes and bool(reset_mask.any().item()):
            active |= reset_mask
            next_obs = reset_done(reset_mask)
            totals.masked_fill_(reset_mask, 0.)
            lengths.masked_fill_(reset_mask, 0)
            for values in metric_totals.values():
                values.masked_fill_(reset_mask, 0.)
            for values in game_event_totals.values():
                values[reset_mask] = 0.
            acquired_at[reset_mask] = 0.
            cycle_time_sum[reset_mask] = 0.
            cycle_count[reset_mask] = 0.
            previous_piece_owner[reset_mask] = env.piece_owner[reset_mask]
        obs = next_obs.to(device=device, dtype=torch.float32)
    write_playback()
    result = {"task": task, "opponent": opponent, "checkpoint": str(checkpoint) if checkpoint else None,
              "checkpoint_mtime": Path(checkpoint).stat().st_mtime if checkpoint else None,
              "episodes": episodes, "seed": seed, "device": str(device),
              "horizon": env.horizon, "match_duration_seconds": env.horizon*env.dt,
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "accelerator_backend": "ROCm" if torch.version.hip else ("CUDA" if torch.version.cuda else "CPU"),
              "drivetrain_config": env.drivetrain_config,
              "mean_return": sum(scores)/len(scores),
              "mean_episode_length": sum(episode_lengths)/len(episode_lengths),
              "returns": scores, "episode_lengths": episode_lengths}
    result.update({"mean_"+name:sum(values)/len(values) for name,values in metric_rows.items() if values})
    def optional_mean(values):
        available = [float(value) for value in values if value is not None]
        return sum(available) / len(available) if available else None
    result.update({
        "mean_acquisitions": optional_mean(game_metric_rows["acquisitions"]),
        "mean_scores": optional_mean(game_metric_rows["scores"]),
        "mean_cycle_time": optional_mean(game_metric_rows["cycle_time"]),
        "mean_denied_objectives": optional_mean(game_metric_rows["denied_objectives"]),
        "mean_abandoned_objectives": optional_mean(game_metric_rows["abandoned_objectives"]),
        "mean_opponent_acquisitions": optional_mean(game_metric_rows["opponent_acquisitions"]),
        "mean_opponent_scores": optional_mean(game_metric_rows["opponent_scores"]),
        "mean_opponent_denied_objectives": optional_mean(game_metric_rows["opponent_denied_objectives"]),
        "mean_opponent_abandoned_objectives": optional_mean(game_metric_rows["opponent_abandoned_objectives"]),
        "mean_total_score": optional_mean(game_metric_rows["scores"]),
        "mean_total_simulated_score": optional_mean(game_metric_rows["total_simulated_score"]),
        "mean_passes_attempted": optional_mean(game_metric_rows["passes_attempted"]),
        "mean_passes_completed": optional_mean(game_metric_rows["passes_completed"]),
        "mean_fuel_transferred": optional_mean(game_metric_rows["fuel_transferred"]),
        "mean_failed_passes": optional_mean(game_metric_rows["failed_passes"]),
        "mean_pass_to_score_cycle_time": optional_mean(game_metric_rows["pass_to_score_cycle_time"]),
        "score_semantics": "scored FUEL piece count; simulator does not assign official match points",
        "cycle_time_units": "simulated seconds from piece acquisition to scoring",
        "unavailable_metrics": ["abandoned_objectives (not modeled by simulator)",
            "passing (no allied teammate entity in the current 1v1 simulator)"],
        "game_metrics_by_episode": game_metric_rows,
        "planning_latency_ms_mean": optional_mean(planning_latencies),
        "planning_latency_ms_p95": (sorted(planning_latencies)[min(len(planning_latencies)-1,
            int(.95*(len(planning_latencies)-1)))] if planning_latencies else None),
        "planning_calls": planning_calls,
        "scripted_strategy": scripted_strategy,
    })
    if is_adstar_defense:
        hold_rate = sum(metric_rows["success"]) / max(1, len(metric_rows["success"]))
        score_times = [value for value in metric_rows["time_to_goal"] if value > 0]
        result.update({"adstar_defender_hold_rate": hold_rate,
                       "adstar_attacker_score_rate": len(score_times) / episodes,
                       "adstar_mean_attack_time_to_score": sum(score_times) / max(1, len(score_times)) if score_times else None,
                       "adstar_replan_interval_steps": env.adstar_replan_interval})
    sorted_latency=sorted(inference_latencies)
    result["inference_latency_ms_mean"]=sum(inference_latencies)/max(1,len(inference_latencies))
    result["inference_latency_ms_p95"]=sorted_latency[min(len(sorted_latency)-1,int(.95*(len(sorted_latency)-1)))] if sorted_latency else 0.
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    if is_adstar_defense:
        eval_status("completed")
    elif is_adstar_defender:
        atomic_json(status_path, {**result, "status": "completed",
            "progress_unit": "episodes", "completed_timesteps": episodes,
            "requested_timesteps": episodes,
            "checkpoint_drivetrain_config": checkpoint_drivetrain_config})
    return result


def evaluate_game_ablations(*, offense_checkpoint: str | Path | None = None,
        defense_checkpoint: str | Path | None = None, episodes: int = 8,
        seed: int = 4100, num_envs: int = 8, device: str = "cuda",
        output: str | Path = "metrics/ablations.json", horizon: int = 8000) -> dict[str, Any]:
    """Run matched strategic-policy comparisons and persist the canonical report.

    Scripted sides emit only the environment's semantic collect/score/intercept
    classes; AD* remains responsible for all resulting navigation.
    """
    if episodes < 1 or num_envs < 1 or horizon < 1:
        raise ValueError("episodes, num_envs and horizon must be positive")
    selected = _device(device)
    matched_envs = max(num_envs, episodes)
    output_path = Path(output)
    specs = (
        ("scripted_offense_adstar", "counter_defense", "adstar_defender",
         "offense", None, "offense_pair"),
        ("learned_offense_adstar", "counter_defense", "adstar_defender",
         None, offense_checkpoint, "offense_pair"),
        ("scripted_defense_adstar", "defense", "adstar", "defense", None,
         "defense_pair"),
        ("learned_defense_adstar", "defense", "adstar", None,
         defense_checkpoint, "defense_pair"),
    )
    evaluations: list[dict[str, Any]] = []
    for name, task, opponent, scripted_strategy, checkpoint, group in specs:
        missing_reason = None
        architecture = "strategic_adstar" if scripted_strategy else None
        if not scripted_strategy:
            if not checkpoint or not Path(checkpoint).is_file():
                missing_reason = "No checkpoint supplied for this learned role."
            else:
                checkpoint_path = Path(checkpoint)
                try:
                    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    missing_reason = f"Could not load checkpoint: {exc}"
                    payload = None
                if payload is not None:
                    architecture = payload.get("architecture", "direct")
                    if architecture not in ("tactical_adstar", "strategic_adstar"):
                        missing_reason = ("Checkpoint is direct-control; a tactical+AD* or "
                                          "strategic+AD* checkpoint is required.")
                    else:
                        metadata = None
                        for metadata_path in (checkpoint_path.parent / "metadata.json",
                                              checkpoint_path.parent / "status.json"):
                            try:
                                metadata = json.loads(metadata_path.read_text())
                                break
                            except (OSError, json.JSONDecodeError):
                                continue
                        if metadata is None or metadata.get("task") != task:
                            found_task = metadata.get("task") if metadata else None
                            missing_reason = (f"Checkpoint task role cannot be confirmed: expected {task}, "
                                              f"found {found_task or 'no task metadata'}.")
        if missing_reason:
            evaluations.append({"name": name, "task": task, "architecture": architecture,
                "role": "offense" if task == "counter_defense" else "defense",
                "opponent": opponent, "episodes": 0, "seed": seed,
                "seeds": [], "comparison_group": group, "comparable": False,
                "scenario_comparability": "not evaluable", "evaluated": False,
                "missing_reason": missing_reason, "checkpoint": str(checkpoint) if checkpoint else None})
            continue
        mode_output = output_path.parent / f"{output_path.stem}-{name}" / "metrics.json"
        try:
            metrics = _evaluate_once(checkpoint, task, episodes, seed, matched_envs,
                selected, opponent, mode_output, horizon=horizon,
                scripted_strategy=scripted_strategy)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            evaluations.append({"name": name, "task": task, "architecture": architecture,
                "role": "offense" if task == "counter_defense" else "defense",
                "opponent": opponent, "episodes": 0, "seed": seed,
                "seeds": [], "comparison_group": group, "comparable": False,
                "scenario_comparability": "not evaluable", "evaluated": False,
                "missing_reason": f"Evaluation failed: {exc}",
                "checkpoint": str(checkpoint) if checkpoint else None})
            continue
        score_rate = (float(metrics.get("mean_success", 0.)) if task == "counter_defense"
                      else 1. - float(metrics.get("mean_success", 0.)))
        evaluations.append({
            "name": name, "task": task, "role": "offense" if task == "counter_defense" else "defense",
            "architecture": architecture or "scripted_adstar", "opponent": opponent,
            "episodes": episodes, "seed": seed,
            "seeds": list(range(seed, seed + episodes)),
            "comparison_group": group, "comparable": False,
            "scenario_comparable": False, "scenario_comparability": "awaiting paired role result",
            "evaluated": True, "missing_reason": None,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "metrics_file": str(mode_output),
            "success_rate": float(metrics.get("mean_success", 0.)),
            "scoring_rate": score_rate,
            "concession_rate": score_rate if task == "defense" else None,
            "mean_acquisitions": metrics.get("mean_acquisitions"),
            "mean_scores": metrics.get("mean_scores"),
            "mean_cycle_time": metrics.get("mean_cycle_time"),
            # Defensive delay is computed below as a paired score-time delta;
            # a per-step counter is not a counterfactual delay measurement.
            "mean_defensive_delay": None,
            "mean_attacker_time_to_score": (metrics.get("mean_time_to_goal")
                if task == "defense" and metrics.get("mean_time_to_goal", 0.) > 0 else None),
            "mean_denied_objectives": metrics.get("mean_denied_objectives"),
            "mean_passes_attempted": metrics.get("mean_passes_attempted"),
            "mean_passes_completed": metrics.get("mean_passes_completed"),
            "mean_fuel_transferred": metrics.get("mean_fuel_transferred"),
            "mean_failed_passes": metrics.get("mean_failed_passes"),
            "mean_pass_to_score_cycle_time": metrics.get("mean_pass_to_score_cycle_time"),
            "mean_abandoned_objectives": metrics.get("mean_abandoned_objectives"),
            "mean_contacts": metrics.get("mean_contact_count"),
            "mean_contact_fraction": metrics.get("mean_contact"),
            "mean_total_score": metrics.get("mean_total_score"),
            "mean_total_simulated_score": metrics.get("mean_total_simulated_score"),
            "mean_opponent_score": metrics.get("mean_opponent_scores"),
            "mean_path_efficiency": metrics.get("mean_path_efficiency"),
            "inference_latency_ms_mean": metrics.get("inference_latency_ms_mean"),
            "planning_latency_ms_mean": metrics.get("planning_latency_ms_mean"),
            "planning_calls": metrics.get("planning_calls"),
            "score_semantics": metrics.get("score_semantics"),
            "horizon": metrics.get("horizon",horizon), "num_envs": matched_envs,
            "drivetrain_config": metrics.get("drivetrain_config"),
        })
    for group in ("offense_pair", "defense_pair"):
        pair = [row for row in evaluations if row["comparison_group"] == group]
        complete = len(pair) == 2 and all(row.get("evaluated") for row in pair)
        for row in pair:
            row["comparable"] = complete
            row["scenario_comparable"] = complete
            row["scenario_comparability"] = (
                f"matched seed={seed}, episodes={episodes}, horizon={horizon}"
                if complete else "unpaired: learned checkpoint unavailable or evaluation failed")
            row["comparable_to"] = [other["name"] for other in pair
                                    if other is not row] if complete else []
        if group == "defense_pair" and complete:
            baseline = next((row for row in pair if row["name"] == "scripted_defense_adstar"), None)
            learned = next((row for row in pair if row["name"] == "learned_defense_adstar"), None)
            if baseline is not None and learned is not None:
                baseline_time = baseline.get("mean_attacker_time_to_score")
                learned_time = learned.get("mean_attacker_time_to_score")
                match_duration = horizon * .02
                if baseline_time is not None:
                    observed_time = learned_time if learned_time is not None else match_duration
                    learned["mean_defensive_delay"] = observed_time - baseline_time
                    baseline["mean_defensive_delay"] = 0.
                    learned["defensive_delay_definition"] = (
                        "learned defender attacker-score time minus paired scripted-defense time; "
                        "no score is censored at episode duration")
    report = {"schema_version": 1, "created_at": time.time(),
        "task": "FRC FUEL strategic+AD* ablations", "episodes_per_strategy": episodes,
        "seed": seed, "matched_envs": matched_envs, "horizon": horizon,
        "device": str(selected), "evaluations": evaluations}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    temp_path.replace(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Tensor-resident FRC defense learning")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--task", choices=("counter_defense", "defense"), default="counter_defense")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--strategic-rate-hz", type=float, default=4.)
    p.add_argument("--gamma", type=float, default=.993)
    p.add_argument("--entropy-coef", type=float, default=.01)
    p.add_argument("--horizon", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--opponent")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--algorithm", choices=("generational", "ppo"), default="generational")
    p.add_argument("--architecture", choices=("strategic_adstar", "tactical_adstar", "direct"),
                   default="strategic_adstar")
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--population", "--opponent-pool-size", dest="population", type=int, default=8,
                   help="maximum retained strong/diverse opponent checkpoints")
    p.add_argument("--elites", "--strong-checkpoints", dest="elites", type=int, default=2,
                   help="number of top-scoring checkpoints guaranteed retention")
    p.add_argument("--l2-coef", type=float, default=1e-5)
    p.add_argument("--initial-checkpoint")
    p.add_argument("--output", default="checkpoints/tensor-ppo")
    p.add_argument("--device", default="cuda")
    e = commands.add_parser("evaluate")
    e.add_argument("--task", choices=("counter_defense", "defense"), default="counter_defense")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--episodes", type=int, default=32)
    e.add_argument("--horizon", type=int, default=8000)
    e.add_argument("--envs", type=int, default=32)
    e.add_argument("--seed", type=int, default=1000)
    e.add_argument("--opponent", default="random")
    e.add_argument("--output", default="metrics/tensor-evaluation.json")
    e.add_argument("--device", default="cuda")
    a = commands.add_parser("evaluate-game-ablations")
    a.add_argument("--offense-checkpoint")
    a.add_argument("--defense-checkpoint")
    a.add_argument("--episodes", type=int, default=8)
    a.add_argument("--horizon", type=int, default=8000)
    a.add_argument("--envs", type=int, default=8)
    a.add_argument("--seed", type=int, default=4100)
    a.add_argument("--output", default="metrics/ablations.json")
    a.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        if args.command == "train":
            result = train(args.task, args.steps, args.output, seed=args.seed,
                           num_envs=args.envs, device=args.device, opponent=args.opponent,
                           rollout_steps=args.rollout_steps, epochs=args.epochs,
                           initial_checkpoint=args.initial_checkpoint, horizon=args.horizon,
                           l2_coef=args.l2_coef, algorithm=args.algorithm,
                           generations=args.generations, population_size=args.population,
                           elite_count=args.elites,
                           architecture=args.architecture,
                           strategic_rate_hz=args.strategic_rate_hz,
                           gamma=args.gamma, entropy_coef=args.entropy_coef)
        elif args.command == "evaluate":
            result = evaluate(args.checkpoint, task=args.task, episodes=args.episodes,
                              seed=args.seed, num_envs=args.envs, device=args.device,
                              opponent=args.opponent, output=args.output, horizon=args.horizon)
        else:
            result = evaluate_game_ablations(offense_checkpoint=args.offense_checkpoint,
                defense_checkpoint=args.defense_checkpoint, episodes=args.episodes,
                seed=args.seed, num_envs=args.envs, device=args.device,
                output=args.output, horizon=args.horizon)
    except Exception as exc:
        if args.command == "train":
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            (out / "status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}))
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
