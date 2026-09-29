"""Optional PPO training, scripted opponents, and seeded evaluation utilities.

This module deliberately keeps Gymnasium and Stable-Baselines3 out of the
inference import path. Importing :mod:`frc_defense.training` itself is safe on
robot code; the optional packages are required only when training/evaluating.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .types import ChassisCommand, Objective, RobotParameters, RobotState, WorldState
from .reward import (ROTATION_COMMAND_DELTA_PENALTY, ROTATION_COMMAND_PENALTY,
                     SPIN_RATE_PENALTY)


def _optional_rl():
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ImportError as exc:
        raise RuntimeError(
            "Training requires optional dependencies. Install with: "
            "python -m pip install 'gymnasium>=0.29' 'stable-baselines3>=2.2'"
        ) from exc
    return gym, PPO, DummyVecEnv, VecMonitor


@dataclass(frozen=True)
class RewardConfig:
    """Suggested dense reward weights; environments may consume these directly."""
    progress: float = 1.0
    completion: float = 10.0
    time: float = 0.02
    blocked: float = 0.05
    contact: float = 0.03
    out_of_bounds: float = 5.0
    oscillation: float = 0.01
    spin_rate: float = SPIN_RATE_PENALTY
    rotation_command: float = ROTATION_COMMAND_PENALTY
    rotation_command_change: float = ROTATION_COMMAND_DELTA_PENALTY
    delay: float = 1.0
    denial: float = 0.5
    ineffective_position: float = 0.03


class ScriptedOpponent:
    """Simple deterministic goal-seeking opponent controller base."""
    name = "scripted"

    def command(self, own: RobotState, other: RobotState, goal: Objective,
                params: RobotParameters, rng: random.Random | None = None) -> ChassisCommand:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.command(*args, **kwargs)


def _toward(x: float, y: float, speed: float) -> tuple[float, float]:
    d = math.hypot(x, y)
    return (0.0, 0.0) if d < 1e-8 else (speed * x / d, speed * y / d)


class PursuitOpponent(ScriptedOpponent):
    name = "pursuit"
    def command(self, own, other, goal, params, rng=None):
        vx, vy = _toward(other.x - own.x, other.y - own.y, params.max_speed)
        return ChassisCommand(vx, vy, 0.0)


class InterceptOpponent(ScriptedOpponent):
    name = "intercept"
    def command(self, own, other, goal, params, rng=None):
        lead = min(1.0, math.hypot(goal.x - other.x, goal.y - other.y) / max(params.max_speed, .1))
        vx, vy = _toward(goal.x - other.x, goal.y - other.y, params.max_speed)
        return ChassisCommand(*_toward(other.x + vx * lead - own.x, other.y + vy * lead - own.y,
                                      params.max_speed), 0.0)


class GuardOpponent(ScriptedOpponent):
    name = "guard"
    def command(self, own, other, goal, params, rng=None):
        # Guard the midpoint on the goal approach lane.
        tx, ty = (other.x + goal.x) * .5, (other.y + goal.y) * .5
        return ChassisCommand(*_toward(tx - own.x, ty - own.y, params.max_speed), 0.0)


class MirrorOpponent(ScriptedOpponent):
    name = "mirror"
    def command(self, own, other, goal, params, rng=None):
        tx, ty = 2 * goal.x - other.x, 2 * goal.y - other.y
        return ChassisCommand(*_toward(tx - own.x, ty - own.y, params.max_speed), 0.0)


class CutoffOpponent(ScriptedOpponent):
    name = "cutoff"
    def command(self, own, other, goal, params, rng=None):
        # Predict a short-horizon goal-bound position and occupy the lane.
        ux, uy = _toward(goal.x - other.x, goal.y - other.y, 1.0)
        tx, ty = other.x + ux * .8, other.y + uy * .8
        return ChassisCommand(*_toward(tx - own.x, ty - own.y, params.max_speed), 0.0)


class NoisyOpponent(ScriptedOpponent):
    name = "noisy"
    def command(self, own, other, goal, params, rng=None):
        rng = rng or random
        vx, vy = _toward(goal.x - own.x, goal.y - own.y, params.max_speed)
        return ChassisCommand(vx + rng.gauss(0, params.max_speed * .18),
                              vy + rng.gauss(0, params.max_speed * .18),
                              rng.gauss(0, params.max_omega * .1))


class RandomOpponent(ScriptedOpponent):
    name = "random"
    def command(self, own, other, goal, params, rng=None):
        rng = rng or random
        return ChassisCommand(rng.uniform(-params.max_speed, params.max_speed),
                              rng.uniform(-params.max_speed, params.max_speed),
                              rng.uniform(-params.max_omega, params.max_omega))


class OffensiveOpponent(ScriptedOpponent):
    """Simple goal-seeking opponent for the defensive PPO task."""
    name = "offense"
    def command(self, own, other, goal, params, rng=None):
        return ChassisCommand(*_toward(goal.x-own.x, goal.y-own.y, params.max_speed), 0.0)


SCRIPTED_OPPONENTS: dict[str, type[ScriptedOpponent]] = {
    cls.name: cls for cls in (PursuitOpponent, InterceptOpponent, GuardOpponent,
                              MirrorOpponent, CutoffOpponent, NoisyOpponent, RandomOpponent,
                              OffensiveOpponent)
}


class OpponentPool:
    """Weighted mixture of scripted strategies and historical PPO checkpoints."""
    def __init__(self, scripted: Iterable[str] | None = None,
                 checkpoints: Iterable[str | Path] = (), seed: int = 0):
        self.scripted = list(scripted or SCRIPTED_OPPONENTS)
        unknown = set(self.scripted) - set(SCRIPTED_OPPONENTS)
        if unknown:
            raise ValueError(f"Unknown scripted opponents: {sorted(unknown)}")
        self.checkpoints = [str(Path(p)) for p in checkpoints]
        self.rng = random.Random(seed)

    def sample(self) -> dict[str, str]:
        choices = [("scripted", name) for name in self.scripted]
        choices += [("checkpoint", path) for path in self.checkpoints]
        if not choices:
            raise ValueError("Opponent pool is empty")
        kind, value = self.rng.choice(choices)
        return {"kind": kind, "value": value}

    def add_checkpoint(self, path: str | Path) -> None:
        p = str(Path(path))
        if p not in self.checkpoints:
            self.checkpoints.append(p)


def _default_env_factory(task: str, opponent: Any, seed: int, **kwargs):
    try:
        from .sim import make_env
    except ImportError as exc:
        raise RuntimeError("The simulator is unavailable; provide env_factory or implement frc_defense.sim.make_env") from exc
    return make_env(task=task, opponent=opponent, seed=seed, **kwargs)


def _make_vec(task: str, n_envs: int, seed: int, opponent: Any,
              env_factory: Callable[..., Any] | None, env_kwargs: Mapping[str, Any]):
    _, _, DummyVecEnv, VecMonitor = _optional_rl()
    factory = env_factory or _default_env_factory
    env = DummyVecEnv([
        (lambda i=i: factory(task=task, opponent=opponent, seed=seed + i, **dict(env_kwargs)))
        for i in range(n_envs)
    ])
    return VecMonitor(env)


def train(task: str = "counter_defense", total_timesteps: int = 100_000,
          output: str | Path = "checkpoints", *, seed: int = 0, n_envs: int = 8,
          device: str = "cpu", opponent: Any = None, opponent_pool: OpponentPool | None = None,
          env_factory: Callable[..., Any] | None = None, env_kwargs: Mapping[str, Any] | None = None,
          checkpoint: str | Path | None = None, **ppo_kwargs):
    """Train PPO. CPU is the default; unavailable or failing CUDA falls back to CPU."""
    _, PPO, _, _ = _optional_rl()
    if task not in ("counter_defense", "defense"):
        raise ValueError("task must be 'counter_defense' or 'defense'")
    if total_timesteps <= 0 or n_envs <= 0:
        raise ValueError("total_timesteps and n_envs must be positive")
    pool = opponent_pool
    chosen = pool.sample() if pool else (opponent or ("guard" if task == "counter_defense" else "offense"))
    env = _make_vec(task, n_envs, seed, chosen, env_factory, env_kwargs or {})
    selected_device = "cpu" if device == "auto" else device
    if selected_device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available(): selected_device = "cpu"
        except ImportError:
            selected_device = "cpu"
    def build_model(target_device):
        return PPO.load(str(checkpoint), env=env, device=target_device) if checkpoint else PPO(
            "MlpPolicy", env, seed=seed, device=target_device, verbose=1,
            policy_kwargs={"net_arch": [128, 128]}, n_steps=256, batch_size=256,
            **ppo_kwargs)
    try:
        try:
            model = build_model(selected_device)
            model.learn(total_timesteps=int(total_timesteps), reset_num_timesteps=checkpoint is None)
        except Exception:
            if not selected_device.startswith("cuda"):
                raise
            selected_device = "cpu"
            model = build_model(selected_device)
            model.learn(total_timesteps=int(total_timesteps), reset_num_timesteps=checkpoint is None)
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        stamp = f"{task}_{int(time.time())}"
        target = out / stamp
        model.save(str(target))
        metadata = {"task": task, "timesteps": int(total_timesteps), "seed": seed,
                    "device": str(model.device), "opponent": str(chosen)}
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        if pool:
            pool.add_checkpoint(target)
        return model, str(target)
    finally:
        env.close()


def evaluate(model: Any, task: str = "counter_defense", episodes: int = 32, *, seed: int = 1000,
             opponent: Any = "random", env_factory: Callable[..., Any] | None = None,
             env_kwargs: Mapping[str, Any] | None = None, deterministic: bool = True) -> dict[str, Any]:
    """Run a fixed-seed single-env benchmark and aggregate common episode metrics."""
    _optional_rl()
    factory = env_factory or _default_env_factory
    rows: list[dict[str, float]] = []
    latencies: list[float] = []
    for ep in range(episodes):
        env = factory(task=task, opponent=opponent, seed=seed + ep, **dict(env_kwargs or {}))
        try:
            obs, _ = env.reset(seed=seed + ep)
            accum: dict[str, float] = {}
            done = trunc = False
            steps = 0
            while not (done or trunc):
                start = time.perf_counter()
                action, _ = model.predict(obs, deterministic=deterministic)
                latencies.append((time.perf_counter() - start) * 1000)
                obs, reward, done, trunc, info = env.step(action)
                steps += 1
                info = info if isinstance(info, dict) else {}
                for key, value in info.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        accum[key] = accum.get(key, 0.0) + float(value)
                accum["return"] = accum.get("return", 0.0) + float(reward)
            accum["steps"] = float(steps)
            rows.append(accum)
        finally:
            env.close()
    keys = sorted({k for row in rows for k in row})
    metrics = {k: sum(r.get(k, 0.0) for r in rows) / max(1, len(rows)) for k in keys}
    metrics.update({"episodes": episodes, "seed": seed,
                    "inference_latency_ms_mean": sum(latencies) / max(1, len(latencies)),
                    "inference_latency_ms_p95": sorted(latencies)[int(.95 * (len(latencies)-1))] if latencies else 0.0})
    return metrics


def benchmark(model: Any, output: str | Path, **kwargs) -> dict[str, Any]:
    metrics = evaluate(model, **kwargs)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def benchmark_simulator(num_envs: int = 512, steps: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Measure batched simulator throughput without training dependencies."""
    import numpy as np
    from .sim import VectorizedSimulator
    if num_envs < 1 or steps < 1:
        raise ValueError("num_envs and steps must be positive")
    sim = VectorizedSimulator(num_envs=num_envs, randomize=True, seed=seed)
    rng = np.random.default_rng(seed)
    commands = rng.uniform(-4.5, 4.5, (num_envs, 2, 3)).astype(np.float32)
    start = time.perf_counter()
    for _ in range(steps):
        sim.step(commands)
    elapsed = time.perf_counter() - start
    return {"num_envs": num_envs, "steps": steps, "elapsed_seconds": elapsed,
            "transitions": num_envs*steps,
            "transitions_per_second": num_envs*steps/max(elapsed, 1e-12), "seed": seed}


def benchmark_ppo_devices(task: str = "counter_defense", timesteps: int = 4096,
                          n_envs: int = 4, seed: int = 7,
                          output: str | Path = "metrics/device-benchmark") -> dict[str, Any]:
    """Measure short PPO updates on CPU and available CUDA, saving one checkpoint per run."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Device benchmarking requires training extras: pip install -e '.[train]'") from exc
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    results = {}
    base = Path(output)
    for selected in devices:
        start = time.perf_counter()
        model, checkpoint = train(task, timesteps, base / selected.replace(":", "_"),
                                  seed=seed, n_envs=n_envs, device=selected)
        elapsed = time.perf_counter()-start
        actual_device = str(model.device)
        results[actual_device] = {"elapsed_seconds": elapsed,
                             "timesteps_per_second": timesteps/max(elapsed,1e-12),
                             "checkpoint": checkpoint}
        del model
    fastest = max(results, key=lambda d: results[d]["timesteps_per_second"])
    metrics = {"task":task,"timesteps":timesteps,"n_envs":n_envs,"seed":seed,
               "devices":results,"recommended_device":fastest}
    base.mkdir(parents=True,exist_ok=True)
    (base/"results.json").write_text(json.dumps(metrics,indent=2,sort_keys=True))
    return metrics


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train or benchmark FRC defense policies")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("smoke", help="short CPU PPO smoke training")
    p.add_argument("--task", choices=("counter_defense", "defense"), default="counter_defense")
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--envs", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="checkpoints/smoke")
    p.add_argument("--device", default="cpu")
    b = sub.add_parser("throughput", help="benchmark vectorized NumPy simulator")
    b.add_argument("--envs", type=int, default=512)
    b.add_argument("--steps", type=int, default=1000)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--output", default="metrics/throughput.json")
    q = sub.add_parser("ppo-benchmark", help="compare CPU and available CUDA PPO update throughput")
    q.add_argument("--task", choices=("counter_defense", "defense"), default="counter_defense")
    q.add_argument("--steps", type=int, default=4096)
    q.add_argument("--envs", type=int, default=4)
    q.add_argument("--seed", type=int, default=7)
    q.add_argument("--output", default="metrics/device-benchmark")
    e = sub.add_parser("evaluate", help="seeded policy evaluation, saved as JSON")
    e.add_argument("--task", choices=("counter_defense", "defense"), required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--episodes", type=int, default=32)
    e.add_argument("--seed", type=int, default=1000)
    e.add_argument("--opponent", default="random")
    e.add_argument("--output", default="metrics/evaluation.json")
    args = parser.parse_args()
    if args.command == "smoke":
        _, path = train(args.task, args.steps, args.output, seed=args.seed,
                        n_envs=args.envs, device=args.device)
        print(json.dumps({"checkpoint": path, "task": args.task, "timesteps": args.steps}))
    elif args.command == "throughput":
        metrics = benchmark_simulator(args.envs, args.steps, args.seed)
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(json.dumps(metrics))
    elif args.command == "ppo-benchmark":
        print(json.dumps(benchmark_ppo_devices(args.task,args.steps,args.envs,args.seed,args.output)))
    elif args.command == "evaluate":
        _, PPO, _, _ = _optional_rl()
        model = PPO.load(args.checkpoint, device="cpu")
        metrics = benchmark(model,args.output,task=args.task,episodes=args.episodes,
                            seed=args.seed,opponent=args.opponent)
        print(json.dumps(metrics))


if __name__ == "__main__":
    _cli()
