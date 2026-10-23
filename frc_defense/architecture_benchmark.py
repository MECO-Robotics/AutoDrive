"""Controlled PPO screening on perception-driven pickup and defense scenarios.

Run with ``python -m frc_defense.architecture_benchmark --help``. This harness
keeps experimental model/checkpoint formats separate from production training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from torch.distributions import Categorical

from .architecture_cnn import FullFieldCNN, FullFieldResNet
from .architecture_dense import BaselineMLP, ResidualMLP, FieldFNOPolicy
from .architecture_entities import DeepSetsPolicy, EntityAttentionPolicy
from .architecture_inputs import PerceptionInputs
from .architecture_scenarios import SCENARIOS, make_env, reset_scenario
from .tensor_training_runtime import _gae_advantages

ARCHITECTURES = {
    "baseline_mlp": BaselineMLP, "residual_mlp": ResidualMLP,
    "deep_sets": DeepSetsPolicy, "entity_attention": EntityAttentionPolicy,
    "full_field_cnn": FullFieldCNN, "full_field_resnet": FullFieldResNet,
    "field_fno": FieldFNOPolicy,
}


def source_provenance():
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "frc_defense").glob("*.py"))
    files += sorted((root / "frc_defense/csrc").glob("*"))
    files += sorted((root / "frc_defense/dashboard_static").glob("*"))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if p.is_file()}
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                              text=True).strip(),
            "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "files": hashes}


def distribution(model, inputs):
    logits, value = model(inputs)
    mask = inputs["scalars"][:, -8:] > .5
    return Categorical(logits=logits.masked_fill(~mask, -1e9)), value


def metrics(env, ids):
    acquired = env.fuel_acquisition_count[:, ids].sum(-1).float()
    score = env.fuel_score_count.float()
    return {"acquired_per_world": acquired.cpu().tolist(),
            "score_blue_per_world": score[:, 1].cpu().tolist(),
            "score_red_per_world": score[:, 0].cpu().tolist(),
            "score_difference_per_world": (score[:, 1] - score[:, 0]).cpu().tolist(),
            "mean_acquired": acquired.mean().item(),
            "mean_score_difference": (score[:, 1] - score[:, 0]).mean().item()}


def snapshot(env, tick, actions):
    """Truth is allowed in evaluation records, never in the policy builder."""
    return {"tick": tick, "match_elapsed": float(env.match_elapsed[0]),
            "robots": env.sim.pose[0].cpu().tolist(),
            "robot_actions": actions[0].cpu().tolist(),
            "robot_acquisitions": env.fuel_acquisition_count[0].cpu().tolist(),
            "team_score": env.fuel_score_count[0].cpu().tolist(),
            "observed_fuel_tracks": env.track_mask[0].sum(-1).cpu().tolist(),
            "observed_opponent_valid": env.opponent_valid[0].cpu().tolist(),
            "fuel": env.piece_pos[0, env.piece_active[0] & (env.piece_owner[0] < 0)].cpu().tolist()}


@torch.no_grad()
def evaluate(model, env, scenario, seed, ids, ticks, repeat, *, reference=None):
    obs = reset_scenario(env, scenario, seed)
    builder = PerceptionInputs(env, ids)
    actions = torch.full((env.n, 6), 7, device=env.device, dtype=torch.long)
    frames = [snapshot(env, 0, actions)]
    counts = torch.zeros(8, device=env.device, dtype=torch.long)
    first_pickup = torch.full((env.n,), float("nan"), device=env.device)
    first_red_score = torch.full_like(first_pickup, float("nan"))
    red_score_integral = torch.zeros_like(first_pickup)
    for tick in range(0, ticks, repeat):
        inputs = builder(obs)
        if reference == "hold":
            selected = torch.full((env.n * len(ids),), 7, device=env.device, dtype=torch.long)
        elif reference == "scripted":
            defense_task = scenario in ("deterministic_defense", "defense_3v3")
            selected = torch.full((env.n * len(ids),), 5 if defense_task else 0,
                                  device=env.device, dtype=torch.long)
            if scenario == "offense_vs_deterministic_defense":
                possession = (env.piece_owner[:, None, :] ==
                              torch.tensor(ids, device=env.device)[None, :, None]).sum(-1)
                selected = torch.where(possession.reshape(-1) >= 6, 4, selected)
        else:
            dist, _ = distribution(model, inputs)
            selected = dist.logits.argmax(-1)
        counts += torch.bincount(selected, minlength=8)
        actions[:, ids] = selected.reshape(env.n, len(ids))
        hold_ticks = min(repeat, ticks - tick)
        for _ in range(hold_ticks):
            env.step(actions, capture_observation=False, capture_info=False)
        time_s = min(ticks, tick + repeat) * env.dt
        first_pickup = torch.where(torch.isnan(first_pickup) &
                                   (env.fuel_acquisition_count[:, ids].sum(-1) > 0),
                                   time_s, first_pickup)
        first_red_score = torch.where(torch.isnan(first_red_score) &
                                      (env.fuel_score_count[:, 0] > 0), time_s, first_red_score)
        red_score_integral += env.fuel_score_count[:, 0] * hold_ticks * env.dt
        obs = env.observe()
        env._last_obs = obs
        if tick % max(repeat, 50) == 0 or tick + repeat >= ticks:
            frames.append(snapshot(env, min(ticks, tick + repeat), actions))
    result = metrics(env, ids)
    def nullable(tensor):
        return [None if value != value else value for value in tensor.cpu().tolist()]
    result.update(first_pickup_seconds_per_world=nullable(first_pickup),
                  first_red_score_seconds_per_world=nullable(first_red_score),
                  red_score_integral_per_world=red_score_integral.cpu().tolist(),
                  mean_red_score_integral=red_score_integral.mean().item(),
                  timing_resolution_seconds=repeat * env.dt)
    result.update(seed=seed, ticks=ticks, physics_dt=env.dt,
                  action_counts=counts.cpu().tolist(), frames=frames,
                  fuel_backend=env.fuel_physics.stats["backend"],
                  perception={"enabled": True, "range_m": env.perception_range,
                              "fov_degrees": env.perception_fov_degrees,
                              "dropout": env.perception_dropout,
                              "position_noise": env.position_noise,
                              "velocity_noise": env.velocity_noise,
                              "interval": env.perception_interval})
    return result


def _potential(env, ids):
    pos = env.sim.pose[:, ids, :2]
    delta = env.track_pos[:, ids] - pos[:, :, None]
    distances = delta.norm(dim=-1).masked_fill(~env.track_mask[:, ids], torch.inf)
    nearest = distances.amin(-1)
    return -torch.where(torch.isfinite(nearest), nearest.clamp(max=6.), 6.)


def train_case(args, architecture, scenario, seed, lr):
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    model = ARCHITECTURES[architecture]().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    ids = (3, 4, 5) if scenario == "defense_3v3" else (3,)
    output = args.output / architecture / scenario / f"seed-{seed}-lr-{lr:g}"
    output.mkdir(parents=True, exist_ok=True)
    train_env = make_env(scenario, args.envs, args.device, seed, horizon=args.horizon)
    eval_env = make_env(scenario, args.eval_envs, args.device, seed + 100000,
                        horizon=args.eval_ticks)
    builder = PerceptionInputs(train_env, ids)
    provenance = source_provenance()
    metadata = {"architecture": architecture, "scenario": scenario, "seed": seed,
                "learning_rate": lr, "parameters": sum(p.numel() for p in model.parameters()),
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "perception_only_dynamic_inputs": True,
                "perception_contract": {
                    "version": 2,
                    "opponents": "anonymous simulated sensor tracks only",
                    "fuel": "retained anonymous simulated sensor tracks only",
                    "own_and_team": "known exact localization, telemetry and possession",
                    "geometry": "public fixed field map",
                    "public_match": "clock, HUB activity and score",
                    "planner_and_avoidance": "same observation boundary as policy inputs",
                    "truth_allowed": "physical dynamics, task rewards and evaluation records",
                }, "provenance": provenance,
                "runtime": {"torch": torch.__version__, "hip": torch.version.hip,
                            "miopen_enabled": torch.backends.cudnn.enabled,
                            "device": args.device,
                            "device_name": (torch.cuda.get_device_name(args.device)
                                            if args.device.startswith("cuda") else "CPU"),
                            "environment": {k: v for k, v in os.environ.items()
                                            if k.startswith("AUTODRIVE_")}},
                "scenario_definition": SCENARIOS[scenario],
                "reward": ("acquisitions + 0.05*(gamma*next_track_potential-current_track_potential)"
                           if "pickup" in scenario else
                           "acquisitions + blue score delta minus red score delta"
                           if scenario == "offense_vs_deterministic_defense" else
                           "blue score delta minus red score delta"),
                "status": "running"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    curve = []
    started = time.perf_counter()
    train_seconds = 0.
    actual_world_ticks = 0
    references = args.output / "references" / scenario / f"seed-{seed}-n-{args.eval_envs}-ticks-{args.eval_ticks}"
    references.mkdir(parents=True, exist_ok=True)
    for update in range(args.updates + 1):
        if update == 0 or update % args.eval_every == 0 or update == args.updates:
            # Hold evaluation worlds fixed across checkpoints and architectures.
            result = evaluate(model, eval_env, scenario, seed + 100000, ids,
                              args.eval_ticks, args.action_repeat)
            result.update(update=update,
                          transitions=update * args.rollout_steps * args.envs * len(ids),
                          world_physics_ticks=actual_world_ticks,
                          train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - started)
            checkpoint = output / f"policy-{update:04d}.pt"
            torch.save({"architecture": architecture, "model_state_dict": model.state_dict(),
                        "scenario": scenario, "seed": seed, "update": update,
                        "provenance": provenance}, checkpoint)
            result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            result["source_sha256"] = provenance["source_sha256"]
            (output / f"eval-{update:04d}.json").write_text(json.dumps(result, indent=2))
            curve.append({k: v for k, v in result.items() if k != "frames"})
            (output / "learning_curve.json").write_text(json.dumps(curve, indent=2))
            print(json.dumps({"architecture": architecture, "scenario": scenario, "seed": seed,
                              "lr": lr, "update": update, "acquired": result["mean_acquired"],
                              "score_difference": result["mean_score_difference"],
                              "train_seconds": round(train_seconds, 2)}), flush=True)
        if update == args.updates:
            break
        # Independent rollout episodes avoid allowing later updates more match time.
        # Train seeds progress identically for each architecture and learning rate.
        update_started = time.perf_counter()
        obs = reset_scenario(train_env, scenario, seed + update * 1009)
        inputs_buffer, actions_buffer, logprobs, values, next_values = [], [], [], [], []
        rewards, terminals, truncations = [], [], []
        rollout_acquired = 0
        rollout_scored = torch.zeros(2, device=args.device, dtype=torch.long)
        full_action = torch.full((args.envs, 6), 7, device=args.device, dtype=torch.long)
        ticks_in_episode = 0
        rollout_inputs = builder(obs)
        for step in range(args.rollout_steps):
            with torch.no_grad():
                dist, value = distribution(model, rollout_inputs)
                selected = dist.sample()
                logprob = dist.log_prob(selected)
                potential_before = _potential(train_env, ids) if "pickup" in scenario else None
                before_acquired = train_env.fuel_acquisition_count[:, ids].clone()
                before_scores = train_env.fuel_score_count.clone()
                full_action[:, ids] = selected.reshape(args.envs, len(ids))
                hold_ticks = min(args.action_repeat, args.horizon - ticks_in_episode)
                for _ in range(hold_ticks):
                    train_env.step(full_action, capture_observation=False, capture_info=False)
                actual_world_ticks += hold_ticks * args.envs
                ticks_in_episode += hold_ticks
                obs = train_env.observe()
                train_env._last_obs = obs
                next_inputs = builder(obs)
                _, next_value = model(next_inputs)
                if "pickup" in scenario:
                    reward = (train_env.fuel_acquisition_count[:, ids] - before_acquired).float()
                    reward += .05 * (args.gamma * _potential(train_env, ids) - potential_before)
                else:
                    scored = train_env.fuel_score_count - before_scores
                    reward = (scored[:, 1] - scored[:, 0]).float()[:, None].expand(-1, len(ids))
                    if scenario == "offense_vs_deterministic_defense":
                        reward = reward + (train_env.fuel_acquisition_count[:, ids] - before_acquired)
                rollout_acquired += int((train_env.fuel_acquisition_count[:, ids] - before_acquired).sum())
                rollout_scored += (train_env.fuel_score_count - before_scores).sum(0)
                truncated = ticks_in_episode >= args.horizon
                inputs_buffer.append(rollout_inputs)
                actions_buffer.append(selected)
                logprobs.append(logprob)
                values.append(value)
                next_values.append(next_value)
                rewards.append(reward.reshape(-1))
                terminals.append(torch.zeros_like(selected, dtype=torch.bool))
                truncations.append(torch.full_like(selected, truncated, dtype=torch.bool))
                if truncated:
                    obs = reset_scenario(train_env, scenario, seed + update * 1009 + step + 1)
                    next_inputs = builder(obs)
                    ticks_in_episode = 0
                rollout_inputs = next_inputs
        reward_tensor = torch.stack(rewards)
        value_tensor = torch.stack(values)
        advantages = _gae_advantages(reward_tensor, torch.stack(terminals), torch.stack(truncations),
                                     value_tensor, torch.stack(next_values), args.gamma, .95)
        returns = (advantages + value_tensor).flatten()
        advantages = advantages.flatten()
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        flat_inputs = {k: torch.cat([b[k] for b in inputs_buffer]) for k in rollout_inputs}
        flat_actions = torch.cat(actions_buffer)
        old_logprobs = torch.cat(logprobs)
        losses = []
        for _ in range(args.epochs):
            permutation = torch.randperm(len(flat_actions), device=args.device)
            for batch in permutation.split(args.minibatch_size):
                minibatch = {k: v[batch] for k, v in flat_inputs.items()}
                dist, prediction = distribution(model, minibatch)
                ratio = (dist.log_prob(flat_actions[batch]) - old_logprobs[batch]).exp()
                surrogate = torch.maximum(-advantages[batch] * ratio,
                                          -advantages[batch] * ratio.clamp(.8, 1.2)).mean()
                value_loss = .5 * (prediction - returns[batch]).square().mean()
                loss = surrogate + .5 * value_loss - .01 * dist.entropy().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite PPO loss in {output}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                if not torch.isfinite(grad):
                    raise RuntimeError(f"Non-finite gradient in {output}")
                optimizer.step()
                losses.append(float(loss.detach()))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        train_seconds += time.perf_counter() - update_started
        with (output / "training.jsonl").open("a") as handle:
            handle.write(json.dumps({"update": update + 1, "mean_reward": reward_tensor.mean().item(),
                                     "mean_loss": sum(losses) / len(losses),
                                     "acquisitions": rollout_acquired,
                                     "score_blue": int(rollout_scored[1]),
                                     "score_red": int(rollout_scored[0]),
                                     "train_seconds": train_seconds}) + "\n")
    for reference in ("hold", "scripted"):
        reference_path = references / f"{reference}.json"
        if reference_path.is_file():
            result = json.loads(reference_path.read_text())
        else:
            result = {}
        if (result.get("source_sha256") != provenance["source_sha256"] or
                result.get("action_repeat") != args.action_repeat):
            result = evaluate(model, eval_env, scenario, seed + 100000, ids,
                              args.eval_ticks, args.action_repeat, reference=reference)
            result.update(source_sha256=provenance["source_sha256"], action_repeat=args.action_repeat)
            temporary = reference_path.with_suffix(f".{architecture}.tmp")
            temporary.write_text(json.dumps(result, indent=2))
            temporary.replace(reference_path)
        (output / f"reference-{reference}.json").write_text(json.dumps(result, indent=2))
    torch.save({"architecture": architecture, "model_state_dict": model.state_dict(),
                "scenario": scenario, "seed": seed, "provenance": provenance}, output / "policy.pt")
    metadata.update(status="complete", train_seconds=train_seconds,
                    elapsed_seconds=time.perf_counter() - started)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 43])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[3e-4])
    parser.add_argument("--output", type=Path, default=Path("evaluations/architecture-study-20261020"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--eval-envs", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--eval-ticks", type=int, default=400)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--rollout-steps", type=int, default=40)
    parser.add_argument("--action-repeat", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=.993)
    parser.add_argument("--disable-miopen", action=argparse.BooleanOptionalAction,
                        default=bool(torch.version.hip),
                        help="use ATen convolution; avoids gfx900 MIOpen weight-gradient faults")
    args = parser.parse_args()
    for name in ("envs", "eval_envs", "horizon", "eval_ticks", "rollout_steps", "action_repeat",
                 "epochs", "minibatch_size", "eval_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.updates < 0 or not 0 < args.gamma <= 1 or any(lr <= 0 for lr in args.learning_rates):
        parser.error("updates must be nonnegative; gamma and learning rates must be positive")
    torch.set_num_threads(1)
    if args.disable_miopen:
        torch.backends.cudnn.enabled = False
    for architecture in args.architectures:
        for scenario in args.scenarios:
            for seed in args.seeds:
                for lr in args.learning_rates:
                    train_case(args, architecture, scenario, seed, lr)


if __name__ == "__main__":
    main()
