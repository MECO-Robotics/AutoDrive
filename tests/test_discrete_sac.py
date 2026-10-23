import math

import torch

from frc_defense.curriculum_training import TrainingConfig, train_curriculum
from frc_defense.discrete_sac import (DiscreteSAC, ScenarioReplayBuffer,
                                      masked_distribution)


def test_masked_distribution_excludes_invalid_actions():
    logits = torch.tensor([[0.0, 2.0, -1.0]])
    mask = torch.tensor([[True, False, True]])
    probs, log_probs, valid = masked_distribution(logits, mask)
    assert probs[0, 1].item() == 0.0
    assert log_probs[0, 1].item() == 0.0
    assert valid.tolist() == mask.tolist()
    assert probs.sum(-1).tolist() == [1.0]


def test_replay_is_bounded_and_balances_scenarios():
    replay = ScenarioReplayBuffer(3)
    def batch(start, count):
        return {
            "obs": torch.arange(start, start + count).float().unsqueeze(-1),
            "mask": torch.ones(count, 2, dtype=torch.bool),
            "action": torch.zeros(count, dtype=torch.long),
            "reward": torch.ones(count),
            "next_obs": torch.arange(start + 1, start + count + 1).float().unsqueeze(-1),
            "next_mask": torch.ones(count, 2, dtype=torch.bool),
            "done": torch.zeros(count, dtype=torch.bool),
        }
    replay.add_batch(batch(0, 2), "old_stage")
    replay.add_batch(batch(2, 2), "new_stage")
    assert len(replay) == 3
    assert set(replay.scenario_indices) == {"old_stage", "new_stage"}
    assert replay.scenario_locations.keys() == {index for rows in replay.scenario_indices.values() for index in rows}
    sampled = replay.sample(2, "cpu")
    assert sampled["obs"].shape == (2, 1)
    assert sampled["action"].dtype == torch.long


def test_sac_curriculum_update_and_resume_keep_replay(tmp_path):
    config = TrainingConfig(
        algorithm="sac", updates=1, num_envs=1, rollout_steps=2, epochs=1,
        seed=41, horizon=8, action_ticks=1, eval_every=0, device="cpu",
        output=str(tmp_path / "first"), sac_replay_capacity=8,
        sac_batch_size=2, sac_warmup_steps=2, sac_updates_per_transition=.5,
    )
    result = train_curriculum(config)
    row = result["stats"][0]
    assert row["algorithm"] == "sac"
    assert row["sac_updates"] == 1
    assert row["replay_size"] == 2
    assert math.isfinite(row["critic_loss"])
    assert math.isfinite(row["actor_loss"])
    assert row["alpha"] > 0

    payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert payload["replay_buffer"]["size"] == 2
    extended = TrainingConfig(**(config.__dict__ | {"updates": 2}))
    continued = train_curriculum(
        TrainingConfig(**(extended.__dict__ | {"output": str(tmp_path / "resumed")})),
        resume=result["checkpoint"],
    )
    uninterrupted = train_curriculum(
        TrainingConfig(**(extended.__dict__ | {"output": str(tmp_path / "straight")})),
    )
    assert continued["stats"][1]["replay_size"] == 4
    assert torch.isfinite(torch.tensor(continued["stats"][1]["critic_loss"]))
    assert continued["stats"] == uninterrupted["stats"]
    assert continued["policy_id"] == uninterrupted["policy_id"]


def test_sac_policy_samples_only_valid_actions():
    policy = DiscreteSAC("baseline_mlp")
    obs = torch.zeros(4, 137)
    mask = torch.tensor([[True, False, False, False, False, False, False, True]] * 4)
    actions, log_prob, _ = policy.sample(obs, action_mask=mask)
    assert mask.gather(1, actions[:, None]).all()
    assert torch.isfinite(log_prob).all()
