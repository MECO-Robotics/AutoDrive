"""Uncapped, two-GPU PPO campaign for the six-robot 3v3 simulator."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from frc_defense.tensor_training import train_3v3_generation


ROOT = Path("evaluations/strategic-ppo-validation-20261002/full-scale-3v3")
RANK = int(os.environ.get("RANK", "0"))
CAMPAIGN_ENVS = int(os.environ.get("AUTODRIVE_CAMPAIGN_ENVS", "51200"))
if CAMPAIGN_ENVS < 2 or CAMPAIGN_ENVS % 2:
    raise ValueError("AUTODRIVE_CAMPAIGN_ENVS must be a positive even number")
TASKS = (
    ("offense", 86_106,
     Path("evaluations/strategic-ppo-validation-20261002/full-scale-accelerated/offense/policy.pt")),
    ("defense", 86_107, Path("checkpoints/rebuilt-gamepiece-defense/policy.pt")),
)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    campaign_file = ROOT / "campaign-status.json"
    campaign = _read_json(campaign_file, {}) if RANK == 0 else {}
    if RANK == 0:
        campaign.update(status="running", uncapped=True,
                        architecture="strategic_3v3", robots_per_alliance=3,
                        resumed_at=time.time(), tasks=campaign.get("tasks", {}))
        campaign.setdefault("started_at", time.time())
        campaign_file.write_text(json.dumps(campaign, indent=2))

    while True:
        for role, base_seed, legacy_checkpoint in TASKS:
            output = ROOT / role
            history = _read_json(output / "generation-history.json", {})
            completed = history.get("generations", [])
            next_generation = len(completed) + 1
            checkpoint = output / "policy.pt"
            initial_checkpoint = checkpoint if checkpoint.is_file() else legacy_checkpoint
            if RANK == 0:
                campaign.update(status="running", uncapped=True,
                                current_task=role, current_generation=next_generation)
                campaign.setdefault("tasks", {})[role] = {
                    "status": "running", "generation": next_generation,
                    "generations_completed": len(completed),
                    "generation_limit": None, "output": str(output),
                    "initial_checkpoint": str(initial_checkpoint),
                }
                campaign_file.write_text(json.dumps(campaign, indent=2))

            result = train_3v3_generation(
                role, next_generation, output,
                seed=base_seed + (next_generation - 1) * 1009,
                num_envs=CAMPAIGN_ENVS,
                device="cuda",
                initial_checkpoint=initial_checkpoint,
                horizon=8_000,
                rollout_steps=128,
                epochs=1,
                minibatch_size=8_192,
                learning_rate=3e-4,
                gamma=.993,
                entropy_coef=.01,
                strategic_rate_hz=4.,
            )
            if RANK == 0:
                campaign["tasks"][role] = {
                    "status": "running", "generation": next_generation,
                    "generations_completed": next_generation,
                    "generation_limit": None, "output": str(output),
                    "checkpoint": str(output / "policy.pt"),
                    "complete_matches_observed": result.get("complete_matches_observed"),
                    "strategic_transitions_per_second": result.get("strategic_transitions_per_second"),
                    "mean_return": result.get("mean_return"),
                }
                campaign_file.write_text(json.dumps(campaign, indent=2))


if __name__ == "__main__":
    main()
