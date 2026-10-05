"""Torch-only PPO for :class:`TensorDefenseEnv`.

Rollouts, advantages, minibatches, and optimizer updates stay as Torch tensors
on the selected device. On ROCm, PyTorch exposes the GPU through ``cuda`` too.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.distributed as dist
except ImportError as exc:  # keep the module's import error actionable
    raise ImportError(
        "Tensor PPO requires PyTorch. Install a Torch build for your platform, "
        "including ROCm or CUDA support when using a GPU."
    ) from exc

from .tensor_training_runtime import (
    ACTION_DIM, OBS_DIM, OBS_NORMALIZATION, STRATEGIC_ACTION_DIM,
    STRATEGIC_LEGACY_OBS_DIM, MIXED_DEFENSE_OPPONENTS, STATIC_OPPONENT_FRACTION,
    CURRICULUM_STAGES, ActorCritic, _DDPScore, _curriculum_stage,
    _curriculum_opponents, _device, _env, _reset_obs, _held_action_interval,
    _gae_advantages, _task_opponent, _adstar_reference, _reference_action,
    _reference_potential, _reference_tracking_error,
)
from . import tensor_training_runtime as _runtime

_PPO_CONTROL_GROUP = None
_PPO_CONTROL_GROUP_OWNED = False


def _distributed_state(device: str | torch.device) -> tuple[torch.device, int, int, bool]:
    _runtime._PPO_CONTROL_GROUP = _PPO_CONTROL_GROUP
    _runtime._PPO_CONTROL_GROUP_OWNED = _PPO_CONTROL_GROUP_OWNED
    try:
        return _runtime._distributed_state(device, device_resolver=_device)
    finally:
        globals()["_PPO_CONTROL_GROUP"] = _runtime._PPO_CONTROL_GROUP
        globals()["_PPO_CONTROL_GROUP_OWNED"] = _runtime._PPO_CONTROL_GROUP_OWNED


def _attach_historical_opponent(env, task: str, device: torch.device,
                               checkpoint_path: str | Path | None = None) -> bool:
    return _runtime._attach_historical_opponent(
        env, task, device, checkpoint_path, actor_critic_cls=ActorCritic)


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
                capture_training_playback: bool | None = None,
                reuse_strategic_own_candidates: bool | None = None,
                ppo_sparse_delayed_observation_capture: bool = False,
                status_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one PPO update sequence through the lazily loaded implementation."""
    from . import tensor_training_ppo
    return tensor_training_ppo.train_once(
        task, timesteps, output, seed=seed, num_envs=num_envs, device=device,
        opponent=opponent, initial_checkpoint=initial_checkpoint,
        rollout_steps=rollout_steps, epochs=epochs, minibatch_size=minibatch_size,
        learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
        clip_coef=clip_coef, value_coef=value_coef, entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm, l2_coef=l2_coef, horizon=horizon,
        architecture=architecture, curriculum=curriculum,
        strategic_rate_hz=strategic_rate_hz,
        learned_opponent_checkpoint=learned_opponent_checkpoint,
        learned_opponent_checkpoints=learned_opponent_checkpoints,
        capture_training_playback=capture_training_playback,
        reuse_strategic_own_candidates=reuse_strategic_own_candidates,
        ppo_sparse_delayed_observation_capture=ppo_sparse_delayed_observation_capture,
        status_context=status_context, _runtime=globals())



def train(task: str = "counter_defense", timesteps: int = 1_000_000,
          output: str | Path = "checkpoints/tensor-ppo", *, seed: int = 7,
          num_envs: int = 1024, device: str = "cuda", opponent: str | None = None,
          initial_checkpoint: str | Path | None = None,
          rollout_steps: int = 128, epochs: int = 4, minibatch_size: int = 4096,
          learning_rate: float = 3e-4, gamma: float = .993, gae_lambda: float = .95,
          clip_coef: float = .2, value_coef: float = .5, entropy_coef: float = .01,
          max_grad_norm: float = .5, l2_coef: float = 1e-5,
          horizon: int = 8000, algorithm: str = "generational", generations: int = 20,
          population_size: int = 8, elite_count: int = 2,
          architecture: str = "strategic_adstar", strategic_rate_hz: float = 4.,
          evaluation_episodes: int = 4, evaluation_workers: int = 1) -> dict[str, Any]:
    """Train with PPO and a generation-managed opponent/checkpoint population."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and algorithm != "generational":
        raise ValueError("torchrun multi-GPU training is supported only with algorithm='generational'")
    if algorithm == "generational":
        return generational_train(task, generations, output, seed=seed,
            num_envs=num_envs, device=device, opponent=opponent,
            initial_checkpoint=initial_checkpoint, horizon=horizon, timesteps=timesteps,
            population_size=population_size, elite_count=elite_count,
            rollout_steps=rollout_steps, epochs=epochs, minibatch_size=min(minibatch_size, rollout_steps*num_envs),
            learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
            clip_coef=clip_coef, value_coef=value_coef, entropy_coef=entropy_coef,
            strategic_rate_hz=strategic_rate_hz,
            max_grad_norm=max_grad_norm, l2_coef=l2_coef, architecture=architecture,
            evaluation_episodes=evaluation_episodes,
            evaluation_workers=evaluation_workers)
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


def train_3v3_generation(task: str, generation: int, output: str | Path, *,
                         seed: int = 7, num_envs: int = 1024,
                         device: str = "cuda", initial_checkpoint: str | Path | None = None,
                         rollout_steps: int = 128, epochs: int = 1,
                         minibatch_size: int = 8192, learning_rate: float = 3e-4,
                         gamma: float = .993, gae_lambda: float = .95,
                         clip_coef: float = .2, value_coef: float = .5,
                         entropy_coef: float = .01, max_grad_norm: float = .5,
                         l2_coef: float = 1e-5, horizon: int = 8000,
                         strategic_rate_hz: float = 4.) -> dict[str, Any]:
    """Train one 3v3 PPO generation using the compatibility globals here."""
    from . import tensor_training_3v3

    return tensor_training_3v3.run_with_legacy_globals(
        globals(), task, generation, output, seed=seed, num_envs=num_envs,
        device=device, initial_checkpoint=initial_checkpoint,
        rollout_steps=rollout_steps, epochs=epochs, minibatch_size=minibatch_size,
        learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
        clip_coef=clip_coef, value_coef=value_coef, entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm, l2_coef=l2_coef, horizon=horizon,
        strategic_rate_hz=strategic_rate_hz)



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


def _dispatch_opponent_evaluations(specs, evaluator, workers: int = 1,
                                   device: str | torch.device = "cpu"):
    """Run ordered opponent evaluations, optionally using worker CUDA streams.

    The output list follows input order. Callers must keep evaluator side-effect
    paths private when using more than one worker.
    """
    if workers < 1:
        raise ValueError("evaluation_workers must be positive")
    if workers == 1:
        return [evaluator(index, spec, None) for index, spec in enumerate(specs)]
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation workers requested but CUDA is unavailable")
    local = threading.local()

    def run(index, spec):
        if selected_device.type != "cuda":
            return evaluator(index, spec, None)
        stream = getattr(local, "stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=selected_device)
            local.stream = stream
        with torch.cuda.device(selected_device), torch.cuda.stream(stream):
            result = evaluator(index, spec, stream)
            stream.synchronize()
            return result

    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="ppo-opponent-eval") as pool:
        futures = [pool.submit(run, index, spec)
                   for index, spec in enumerate(specs)]
        return [future.result() for future in futures]


def _opponent_evaluation_paths(evaluation_dir: Path, spec: dict[str, Any],
                               index: int, concurrent: bool) -> tuple[Path, Path]:
    """Return canonical report and isolated execution paths for one spec."""
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", spec["name"]).strip("-")
    canonical = evaluation_dir / f"{safe_name}.json"
    execution = (evaluation_dir / ".workers" / f"worker-{index:03d}" / "metrics.json"
                 if concurrent else canonical)
    return canonical, execution


def _generational_train_impl(task: str, generations: int, output: str | Path, *,
                       seed: int = 7, num_envs: int = 1024, device: str = "cuda",
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
                       strategic_rate_hz: float = 4.,
                       timing_profile: bool = False,
                       evaluation_workers: int = 1) -> dict[str, Any]:
    from . import tensor_training_generational
    options = {key: value for key, value in locals().items()
               if key not in ("task", "generations", "output")}
    return tensor_training_generational._generational_train_impl(
        task, generations, output, _runtime=globals(), **options)

def generational_train(task: str, generations: int, output: str | Path,
                       **kwargs) -> dict[str, Any]:
    """Run opponent-population PPO, owning only process groups we create."""
    global _PPO_CONTROL_GROUP, _PPO_CONTROL_GROUP_OWNED
    device = kwargs.get("device", "cuda")
    selected_device, _rank, _world_size, created = _distributed_state(device)
    kwargs["device"] = selected_device
    try:
        return _generational_train_impl(task, generations, output, **kwargs)
    finally:
        if _PPO_CONTROL_GROUP_OWNED and dist.is_initialized():
            dist.destroy_process_group(_PPO_CONTROL_GROUP)
            _PPO_CONTROL_GROUP = None
            _PPO_CONTROL_GROUP_OWNED = False
            _runtime._PPO_CONTROL_GROUP = None
            _runtime._PPO_CONTROL_GROUP_OWNED = False
        if created and dist.is_initialized():
            # No implicit cleanup of caller-owned groups. On exceptions this
            # releases the local process; torchrun tears down failed peers.
            dist.destroy_process_group()

def evaluate(checkpoint: str | Path, *, task: str = "counter_defense",
             episodes: int = 32, seed: int = 1000, num_envs: int = 32,
             device: str = "cuda", opponent: str = "random",
             output: str | Path = "metrics/tensor-evaluation.json",
             horizon: int = 8000) -> dict[str, Any]:
    from . import tensor_training_evaluation as _evaluation
    _evaluation._bind_training_globals(globals())
    return _evaluation.evaluate(checkpoint, task=task, episodes=episodes, seed=seed,
        num_envs=num_envs, device=device, opponent=opponent, output=output, horizon=horizon)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    from . import tensor_training_evaluation as _evaluation
    _evaluation._bind_training_globals(globals())
    return _evaluation._wilson_interval(successes, total, z)


def _scripted_game_action(env, task: str) -> torch.Tensor:
    from . import tensor_training_evaluation as _evaluation
    _evaluation._bind_training_globals(globals())
    return _evaluation._scripted_game_action(env, task)


def _legacy_strategic_action(action: torch.Tensor, task: str) -> torch.Tensor:
    from . import tensor_training_evaluation as _evaluation
    _evaluation._bind_training_globals(globals())
    return _evaluation._legacy_strategic_action(action, task)


def _evaluate_once(checkpoint, task, episodes, seed, num_envs, device, opponent,
                   output, horizon=8000, scripted_strategy=None,
                   opponent_checkpoint=None, strategic_rate_hz=4.,
                   capture_playback=True, scenario_zone_pair=None):
    from . import tensor_training_evaluation as _evaluation
    _evaluation._bind_training_globals(globals())
    return _evaluation._evaluate_once(checkpoint, task, episodes, seed, num_envs,
        device, opponent, output, horizon, scripted_strategy, opponent_checkpoint,
        strategic_rate_hz, capture_playback, scenario_zone_pair)


def _unevaluated_ablation_row(name: str, task: str, architecture: str | None,
                             opponent: str, group: str, seed: int,
                             checkpoint: str | Path | None, missing_reason: str) -> dict[str, Any]:
    from . import tensor_training_evaluation as _evaluation
    _evaluation._bind_training_globals(globals())
    return _evaluation._unevaluated_ablation_row(name, task, architecture, opponent,
        group, seed, checkpoint, missing_reason)


def evaluate_game_ablations(*, offense_checkpoint: str | Path | None = None,
        defense_checkpoint: str | Path | None = None, episodes: int = 8,
        seed: int = 4100, num_envs: int = 8, device: str = "cuda",
        output: str | Path = "metrics/ablations.json", horizon: int = 8000,
        strategic_rate_hz: float = 4., capture_playback: bool = True) -> dict[str, Any]:
    from . import tensor_training_evaluation as _evaluation
    _evaluation._bind_training_globals(globals())
    return _evaluation.evaluate_game_ablations(
        offense_checkpoint=offense_checkpoint, defense_checkpoint=defense_checkpoint,
        episodes=episodes, seed=seed, num_envs=num_envs, device=device, output=output,
        horizon=horizon, strategic_rate_hz=strategic_rate_hz,
        capture_playback=capture_playback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tensor-resident FRC defense learning")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--task", choices=("counter_defense", "defense"), default="counter_defense")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--envs", type=int, default=1024)
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
    p.add_argument("--evaluation-episodes", type=int, default=4,
                   help="seeded episodes per opponent at each generation boundary")
    p.add_argument("--evaluation-workers", type=int, default=1,
                   help="concurrent CUDA stream workers for generation evaluation")
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
                           gamma=args.gamma, entropy_coef=args.entropy_coef,
                           evaluation_episodes=args.evaluation_episodes,
                           evaluation_workers=args.evaluation_workers)
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
            if int(os.environ.get("RANK", "0")) == 0:
                out.mkdir(parents=True, exist_ok=True)
                (out / "status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}))
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
