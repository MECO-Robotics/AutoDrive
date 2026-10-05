"""Model and device/runtime helpers for tensor PPO training."""
from __future__ import annotations

import os
import torch
import torch.distributed as dist
from torch import nn
from torch.distributions import Normal
from pathlib import Path
from typing import Any

OBS_DIM = 35
ACTION_DIM = 3
OBS_NORMALIZATION = "fixed_physical_scale_v1"
MIXED_DEFENSE_OPPONENTS = ("adstar", "offense", "intercept", "velocity_intercept", "mirror")
STATIC_OPPONENT_FRACTION = .20
STRATEGIC_LEGACY_OBS_DIM = 89
STRATEGIC_ACTION_DIM = 8
CURRICULUM_STAGES = ("static_straight", "scripted", "adstar", "mixed", "learned")
_PPO_CONTROL_GROUP = None
_PPO_CONTROL_GROUP_OWNED = False


def _curriculum_stage(progress: float) -> int:
    return min(len(CURRICULUM_STAGES) - 1,
               max(0, int(max(0., min(.999999, progress)) * len(CURRICULUM_STAGES))))


def _curriculum_opponents(task: str, stage: int, learned_available: bool = True) -> tuple[str, ...]:
    """Expand training from easy motion to durable scripted and learned pools."""
    if task == "defense":
        # ``intercept`` is a counter-defense-only mode in TensorDefenseEnv;
        # use the attacker-side cutoff policy for the defensive curriculum.
        base = ("offense", "mirror", "adstar", "cutoff", "velocity_intercept")
        pools = (("offense",), ("offense", "mirror"),
                 ("offense", "mirror", "adstar"), base,
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


class _DDPScore(nn.Module):
    """Expose ActorCritic.score through DDP's forward/reducer path."""

    def __init__(self, policy: ActorCritic):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor, actions: torch.Tensor,
                action_mask: torch.Tensor | None = None):
        return self.policy.score(obs, actions, action_mask)


def _distributed_state(device: str | torch.device, device_resolver=None) -> tuple[torch.device, int, int, bool]:
    """Initialize the opt-in torchrun process group and select the local GPU."""
    global _PPO_CONTROL_GROUP, _PPO_CONTROL_GROUP_OWNED
    device_resolver = device_resolver or _device
    def ensure_control_group(world_size: int) -> None:
        global _PPO_CONTROL_GROUP, _PPO_CONTROL_GROUP_OWNED
        if world_size > 1 and _PPO_CONTROL_GROUP is None:
            # Barriers during rank-0-only evaluation should not spin the idle
            # GPU through RCCL; PPO broadcasts/reductions still use default NCCL.
            _PPO_CONTROL_GROUP = dist.new_group(backend="gloo")
            _PPO_CONTROL_GROUP_OWNED = True
    if dist.is_available() and dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if world_size == 1:
            return device_resolver(device), rank, world_size, False
        if not torch.cuda.is_available():
            raise RuntimeError("distributed PPO requires an available CUDA/ROCm device on every rank")
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        ensure_control_group(world_size)
        return torch.device("cuda", local_rank), rank, world_size, False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return device_resolver(device), 0, 1, False
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun PPO requires an available CUDA/ROCm device on every rank")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} is outside the visible GPU set")
    torch.cuda.set_device(local_rank)
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(backend="nccl")
    rank, actual_world = dist.get_rank(), dist.get_world_size()
    if actual_world != world_size:
        raise RuntimeError(f"torchrun WORLD_SIZE={world_size} but process group has {actual_world} ranks")
    ensure_control_group(actual_world)
    return torch.device("cuda", local_rank), rank, actual_world, created


def _device(requested: str) -> torch.device:
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested accelerator {requested!r} is unavailable; refusing CPU fallback")
    return dev


def _env(num_envs: int, task: str, device: torch.device, seed: int, opponent: str,
         horizon: int = 8000, normalize_observations: bool = True,
         static_opponent_fraction: float = 0., architecture: str = "direct",
         reuse_strategic_own_candidates: bool | None = None):
    try:
        from .tensor_sim import TensorDefenseEnv
    except ImportError as exc:
        raise RuntimeError("Tensor PPO requires frc_defense.tensor_sim.TensorDefenseEnv") from exc
    if reuse_strategic_own_candidates is None:
        reuse_strategic_own_candidates=architecture in ("strategic_adstar", "strategic")
    env = TensorDefenseEnv(num_envs=num_envs, task=task, device=device,
                           seed=seed, opponent=opponent, horizon=horizon,
                           normalize_observations=normalize_observations,
                           static_opponent_fraction=static_opponent_fraction,
                           action_mode={"direct": "direct", "tactical_adstar": "tactical",
                                        "strategic_adstar": "strategic"}.get(architecture, architecture),
                           reuse_strategic_own_candidates=reuse_strategic_own_candidates)
    return env


def _attach_historical_opponent(env, task: str, device: torch.device,
                               checkpoint_path: str | Path | None = None,
                               actor_critic_cls=None) -> bool:
    """Load historical direct or 89-feature strategic policies as opponents."""
    actor_critic_cls = actor_critic_cls or ActorCritic
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
        model = actor_critic_cls(payload.get("obs_dim", OBS_DIM),
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
    env.learned_opponent_action_kind=payload.get("action_kind","continuous")
    env.learned_opponent_architecture=payload.get("architecture","direct")
    env.learned_opponent_checkpoint = str(checkpoint)
    return True


def _reset_obs(result: Any) -> torch.Tensor:
    """Accept Gym-style ``(obs, info)`` as well as the bare tensor contract."""
    return result[0] if isinstance(result, tuple) else result


def _held_action_interval(env, action: torch.Tensor, ticks: int,
                          initial_obs: torch.Tensor,
                          known_remaining_ticks: int | None = None,
                          return_info: bool | None = None,
                          opponent_policy_decision_each_tick: bool = False,
                          defer_intermediate_observation: bool = True,
                          coarse_delayed_observation_capture: bool = False):
    """Advance active worlds under one action without crossing episode ends."""
    active_this_decision = torch.ones(env.n, dtype=torch.bool, device=env.device)
    done = torch.zeros_like(active_this_decision)
    truncated = torch.zeros_like(active_this_decision)
    reward = torch.zeros(env.n, dtype=torch.float32, device=env.device)
    transition_next_obs = initial_obs
    complete_matches = torch.zeros((), dtype=torch.long, device=env.device)
    # Strategic episodes only end by horizon truncation. Pull the per-world
    # remaining tick counts once per policy interval, then use the known
    # schedule to avoid a GPU reduction/synchronization on every 50 Hz tick.
    active_counts = None
    if known_remaining_ticks is not None:
        # In strategic mode termination is horizon-only, and every world is
        # reset together from the initial batch. The trainer tracks this scalar
        # episode clock, avoiding a GPU->CPU copy/reduction at every decision.
        active_counts = [env.n if known_remaining_ticks >= tick else 0
                         for tick in range(1, ticks + 1)]
    elif hasattr(env, "horizon"):
        remaining_ticks = (int(env.horizon) - env.steps).detach().cpu()
        active_counts = [int((remaining_ticks >= tick).sum())
                         for tick in range(1, ticks + 1)]
    sparse_observation_capture = bool(
        getattr(env, "_ppo_sparse_delayed_observation_capture", False) and
        getattr(env, "action_mode", None) == "strategic" and
        known_remaining_ticks is not None and return_info is False and
        known_remaining_ticks > ticks +
        int(getattr(env, "max_observation_latency_steps", 0)) and
        getattr(env, "_full_observation_hip_available", False))
    delay_groups = None
    if sparse_observation_capture:
        delay_groups = getattr(env, "_ppo_observation_delay_groups", None)
        if delay_groups is None:
            delays = env.observation_delay.detach().to("cpu").tolist()
            groups = [[] for _ in range(int(env.max_observation_latency_steps) + 1)]
            for row, delay in enumerate(delays):
                groups[int(delay)].append(row)
            delay_groups = [torch.tensor(rows, device=env.device, dtype=torch.long)
                            for rows in groups]
            env._ppo_observation_delay_groups = delay_groups
    active_world_physics_ticks = 0
    for tick_index in range(ticks):
        active_worlds_this_tick = (active_counts[tick_index] if active_counts is not None
                                   else int(active_this_decision.sum().item()))
        step_kwargs = {"active_mask": active_this_decision}
        # Learned strategic opponents share the same policy decision cadence
        # as the controlled policy. Their selected action is held during the
        # remaining 50 Hz physics ticks in this interval.
        if hasattr(env, "action_mode"):
            step_kwargs["_opponent_policy_decision"] = (
                tick_index == 0 or opponent_policy_decision_each_tick)
        if return_info is not None:
            # PPO consumes observations, rewards, and episode flags only. The
            # environment's detailed diagnostics are for evaluation/UI callers.
            step_kwargs["_return_info"] = return_info
        if active_counts is not None:
            step_kwargs["_active_count"] = active_counts[tick_index]
        if sparse_observation_capture:
            # The policy's observation at interval end reads the raw snapshot
            # observation_delay physics ticks earlier. Capture only those rows
            # at their source tick. Episode-end intervals use the legacy path,
            # preserving complete terminal delayed-observation history.
            source_delay = ticks - tick_index - 1
            step_kwargs["_ppo_observation_capture_rows"] = (
                delay_groups[source_delay] if source_delay < len(delay_groups)
                else torch.empty((0,), device=env.device, dtype=torch.long))
        elif (coarse_delayed_observation_capture and
              known_remaining_ticks is not None and return_info is False and
              hasattr(env, "max_observation_latency_steps") and
              tick_index < max(0, ticks -
                              int(env.max_observation_latency_steps) - 1) and
              tick_index + 1 < known_remaining_ticks):
            # A delayed observation at this decision boundary can only read
            # one of the most recent max_latency + 1 raw snapshots. Earlier
            # history slots age out before they can be consumed. Pass an empty
            # capture-row index so _obs still refreshes strategic candidates,
            # advances the ring index, and consumes its normal RNG draws while
            # avoiding construction of an unused 137-feature observation.
            step_kwargs["_ppo_observation_capture_rows"] = torch.empty(
                (0,), device=env.device, dtype=torch.long)
        if (defer_intermediate_observation and known_remaining_ticks is not None
                and return_info is False and hasattr(env, "action_mode")):
            # The PPO policy only consumes this interval's final observation,
            # or the terminal observation if the known match horizon lands
            # inside the interval. Keep all raw history writes and RNG draws;
            # skip only intermediate delayed-output work.
            step_kwargs["_capture_observation"] = (
                tick_index == ticks - 1 or
                tick_index + 1 == known_remaining_ticks)
        capture_rows = step_kwargs.pop("_ppo_observation_capture_rows", None)
        if capture_rows is not None:
            env._ppo_observation_capture_rows = capture_rows
        try:
            next_obs, tick_reward, tick_done, tick_truncated, _info = env.step(
                action, **step_kwargs)
        finally:
            if capture_rows is not None:
                del env._ppo_observation_capture_rows
        active_before_tick = active_this_decision
        full_active_tick = (known_remaining_ticks is not None and
                            active_worlds_this_tick == env.n)
        if full_active_tick:
            reward += tick_reward.to(device=env.device, dtype=torch.float32)
        else:
            reward += (tick_reward.to(device=env.device, dtype=torch.float32) *
                       active_before_tick)
        tick_done = tick_done.to(device=env.device, dtype=torch.bool)
        tick_truncated = tick_truncated.to(device=env.device, dtype=torch.bool)
        if full_active_tick:
            done |= tick_done
            truncated |= tick_truncated
            transition_next_obs = next_obs
        else:
            done |= tick_done & active_before_tick
            truncated |= tick_truncated & active_before_tick
            transition_next_obs = torch.where(
                active_before_tick[:, None], next_obs, transition_next_obs)
        newly_ended = active_before_tick & (tick_done | tick_truncated)
        complete_matches += (tick_truncated & active_before_tick).sum()
        active_this_decision &= ~newly_ended
        active_world_physics_ticks += active_worlds_this_tick
    # Ended worlds remain frozen until reset, so one batched read now returns
    # their terminal observation and the final state for every still-active
    # world without constructing full strategic observations per physics tick.
    return (reward, done, truncated, transition_next_obs, active_this_decision,
            ticks, active_world_physics_ticks, complete_matches)


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
