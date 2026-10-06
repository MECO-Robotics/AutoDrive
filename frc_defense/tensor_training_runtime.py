"""Model and device/runtime helpers for tensor PPO training."""
from __future__ import annotations

import os
import torch
import torch.distributed as dist
from torch import nn
from torch.distributions import Normal
from typing import Any

OBS_DIM = 35
ACTION_DIM = 3
OBS_NORMALIZATION = "fixed_physical_scale_v1"
MIXED_DEFENSE_OPPONENTS = ("adstar", "offense", "intercept", "velocity_intercept", "mirror")
DEFENSE_TRAINING_OPPONENTS = ("offense", "adstar", "mirror", "cutoff", "velocity_intercept")
STATIC_OPPONENT_FRACTION = .20
STRATEGIC_ACTION_DIM = 8
CURRICULUM_STAGES = ("static_straight", "scripted", "adstar", "mixed")
_PPO_CONTROL_GROUP = None
_PPO_CONTROL_GROUP_OWNED = False


def _curriculum_stage(progress: float) -> int:
    return min(len(CURRICULUM_STAGES) - 1,
               max(0, int(max(0., min(.999999, progress)) * len(CURRICULUM_STAGES))))


def _curriculum_opponents(task: str, stage: int) -> tuple[str, ...]:
    """Progress defense training across deterministic attacker behaviors."""
    if task != "defense":
        raise ValueError("only defense training has a curriculum")
    # ``intercept`` is a counter-defense-only mode in TensorDefenseEnv;
    # use the attacker-side cutoff policy for defensive curriculum variants.
    base = DEFENSE_TRAINING_OPPONENTS
    pools = (("offense",), ("offense", "mirror"),
             ("offense", "mirror", "adstar"), base)
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


def _reset_obs(result: Any) -> torch.Tensor:
    """Accept Gym-style ``(obs, info)`` as well as the bare tensor contract."""
    return result[0] if isinstance(result, tuple) else result


def _held_action_interval(env, action: torch.Tensor, ticks: int,
                          initial_obs: torch.Tensor,
                          known_remaining_ticks: int | None = None,
                          return_info: bool | None = None,
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
    if task != "defense":
        raise ValueError("only defense training has an opponent policy")
    if opponent in (None, "guard"):
        return "offense"
    return opponent
