"""Categorical Soft Actor-Critic components for curriculum training."""
from __future__ import annotations

import random
from collections.abc import Mapping

import torch
from torch import nn

from .curriculum_policy import make_policy


def _logits_and_value(policy, observations):
    output = policy(observations)
    if len(output) == 3:
        return output[0], output[2]
    return output[0], output[1]


def masked_distribution(logits, action_mask):
    mask = action_mask.bool()
    if mask.shape != logits.shape:
        raise ValueError("action mask shape must match categorical logits")
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    log_probs = masked_logits.log_softmax(dim=-1)
    probs = log_probs.exp().masked_fill(~mask, 0.)
    safe_log_probs = log_probs.masked_fill(~mask, 0.)
    return probs, safe_log_probs, mask


class _DiscreteQ(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.network = make_policy(architecture)

    def forward(self, observations):
        return _logits_and_value(self.network, observations)[0]


class DiscreteSAC(nn.Module):
    """Categorical actor, twin critics, target critics, and learned temperature."""

    def __init__(self, architecture="baseline_mlp", initial_alpha=.2):
        super().__init__()
        self.algorithm = "sac"
        self.architecture = architecture
        self.actor = make_policy(architecture)
        self.q1 = _DiscreteQ(architecture)
        self.q2 = _DiscreteQ(architecture)
        self.target_q1 = _DiscreteQ(architecture)
        self.target_q2 = _DiscreteQ(architecture)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        for parameter in (*self.target_q1.parameters(), *self.target_q2.parameters()):
            parameter.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(float(initial_alpha)).log())

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(self, observations):
        logits, value = _logits_and_value(self.actor, observations)
        return logits, None, value

    def sample(self, observations, deterministic=False, action_mask=None):
        logits, _, value = self(observations)
        if action_mask is None:
            action_mask = torch.ones_like(logits, dtype=torch.bool)
        probs, log_probs, _ = masked_distribution(logits, action_mask)
        distribution = torch.distributions.Categorical(probs=probs)
        action = probs.argmax(-1) if deterministic else distribution.sample()
        log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        return action, log_prob, value


def make_sac_optimizers(agent, learning_rate):
    return {
        "actor": torch.optim.Adam(agent.actor.parameters(), lr=learning_rate),
        "critic": torch.optim.Adam(
            (*agent.q1.parameters(), *agent.q2.parameters()), lr=learning_rate
        ),
        "alpha": torch.optim.Adam((agent.log_alpha,), lr=learning_rate),
    }


def _tree_map(value, fn):
    if isinstance(value, Mapping):
        return {key: _tree_map(child, fn) for key, child in value.items()}
    return fn(value)


def _tree_index(value, index):
    return _tree_map(value, lambda tensor: tensor[index])


def _tree_stack(rows):
    if isinstance(rows[0], Mapping):
        return {key: _tree_stack([row[key] for row in rows]) for key in rows[0]}
    return torch.stack(rows)


def _to_device(value, device):
    return _tree_map(value, lambda tensor: tensor.to(device=device, non_blocking=True))


class ScenarioReplayBuffer:
    """Bounded CPU replay sampled evenly across scenarios currently in replay."""

    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self.rows = [None] * capacity
        self.scenario_indices = {}
        self.scenario_locations = {}
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    @staticmethod
    def _cpu_tree(value, index):
        return _tree_map(value, lambda tensor: tensor[index].detach().to("cpu").clone())

    def add_batch(self, batch, scenario):
        count = int(batch["action"].shape[0])
        for row_index in range(count):
            index = self.position
            previous = self.rows[index]
            if previous is not None:
                old_scenario = previous["scenario"]
                bucket = self.scenario_indices[old_scenario]
                old_location = self.scenario_locations.pop(index)
                last_location = len(bucket) - 1
                moved = bucket[last_location]
                bucket[old_location] = moved
                bucket.pop()
                if moved != index:
                    self.scenario_locations[moved] = old_location
                if not bucket:
                    del self.scenario_indices[old_scenario]

            row = {
                "scenario": scenario,
                "obs": self._cpu_tree(batch["obs"], row_index),
                "mask": batch["mask"][row_index].detach().cpu().clone(),
                "action": batch["action"][row_index].detach().cpu().clone(),
                "reward": batch["reward"][row_index].detach().cpu().clone(),
                "next_obs": self._cpu_tree(batch["next_obs"], row_index),
                "next_mask": batch["next_mask"][row_index].detach().cpu().clone(),
                "done": batch["done"][row_index].detach().cpu().clone(),
            }
            self.rows[index] = row
            bucket = self.scenario_indices.setdefault(scenario, [])
            self.scenario_locations[index] = len(bucket)
            bucket.append(index)
            self.position = (index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        if type(batch_size) is not int or batch_size < 1 or self.size < batch_size:
            raise ValueError("replay does not contain a full training batch")
        scenarios = tuple(self.scenario_indices)
        chosen = [random.choice(scenarios) for _ in range(batch_size)]
        rows = [self.rows[random.choice(self.scenario_indices[scenario])] for scenario in chosen]
        result = {
            "obs": _tree_stack([row["obs"] for row in rows]),
            "mask": torch.stack([row["mask"] for row in rows]),
            "action": torch.stack([row["action"] for row in rows]).long(),
            "reward": torch.stack([row["reward"] for row in rows]).float(),
            "next_obs": _tree_stack([row["next_obs"] for row in rows]),
            "next_mask": torch.stack([row["next_mask"] for row in rows]),
            "done": torch.stack([row["done"] for row in rows]).float(),
        }
        return {key: _to_device(value, device) for key, value in result.items()}

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "rows": self.rows,
            "scenario_indices": self.scenario_indices,
            "scenario_locations": self.scenario_locations,
            "position": self.position,
            "size": self.size,
        }

    def load_state_dict(self, state):
        expected = {"capacity", "rows", "scenario_indices", "scenario_locations", "position", "size"}
        if not isinstance(state, dict) or set(state) != expected or state["capacity"] != self.capacity:
            raise ValueError("incompatible SAC replay state")
        if (not isinstance(state["rows"], list) or len(state["rows"]) != self.capacity or
                not 0 <= state["position"] < self.capacity or not 0 <= state["size"] <= self.capacity):
            raise ValueError("invalid SAC replay state")
        self.rows = state["rows"]
        self.scenario_indices = state["scenario_indices"]
        self.scenario_locations = state["scenario_locations"]
        self.position = state["position"]
        self.size = state["size"]


def sac_update(agent, optimizers, batch, *, gamma, tau, target_entropy_fraction):
    logits, _ = _logits_and_value(agent.actor, batch["next_obs"])
    next_probs, next_log_probs, next_mask = masked_distribution(logits, batch["next_mask"])
    with torch.no_grad():
        next_q = torch.minimum(agent.target_q1(batch["next_obs"]), agent.target_q2(batch["next_obs"]))
        next_value = (next_probs * (next_q - agent.alpha.detach() * next_log_probs)).sum(-1)
        target = batch["reward"] + gamma * (1. - batch["done"]) * next_value

    q1 = agent.q1(batch["obs"]).gather(-1, batch["action"].unsqueeze(-1)).squeeze(-1)
    q2 = agent.q2(batch["obs"]).gather(-1, batch["action"].unsqueeze(-1)).squeeze(-1)
    critic_loss = .5 * (q1 - target).square().mean() + .5 * (q2 - target).square().mean()
    optimizers["critic"].zero_grad(set_to_none=True)
    critic_loss.backward()
    optimizers["critic"].step()

    actor_logits, _ = _logits_and_value(agent.actor, batch["obs"])
    probs, log_probs, valid_mask = masked_distribution(actor_logits, batch["mask"])
    min_q = torch.minimum(agent.q1(batch["obs"]), agent.q2(batch["obs"])).detach()
    actor_loss = (probs * (agent.alpha.detach() * log_probs - min_q)).sum(-1).mean()
    optimizers["actor"].zero_grad(set_to_none=True)
    actor_loss.backward()
    optimizers["actor"].step()

    entropy = -(probs * log_probs).sum(-1)
    target_entropy = target_entropy_fraction * valid_mask.sum(-1).float().log()
    alpha_loss = (agent.log_alpha * (entropy.detach() - target_entropy.detach())).mean()
    optimizers["alpha"].zero_grad(set_to_none=True)
    alpha_loss.backward()
    optimizers["alpha"].step()

    with torch.no_grad():
        for target_parameter, parameter in zip(agent.target_q1.parameters(), agent.q1.parameters()):
            target_parameter.lerp_(parameter, tau)
        for target_parameter, parameter in zip(agent.target_q2.parameters(), agent.q2.parameters()):
            target_parameter.lerp_(parameter, tau)

    return {
        "actor_loss": actor_loss.detach(),
        "critic_loss": critic_loss.detach(),
        "alpha_loss": alpha_loss.detach(),
        "alpha": agent.alpha.detach(),
        "entropy": entropy.detach().mean(),
        "valid_action_count": valid_mask.sum(-1).float().mean().detach(),
    }
