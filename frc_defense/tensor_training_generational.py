"""Generation-managed PPO training implementation."""
from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any


def _generational_train_impl(task: str, generations: int, output: str | Path, *,
                       _runtime: dict[str, Any],
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
    """Run PPO generations and manage a diverse, evaluated opponent population.

    Each generation continues the prior PPO checkpoint. Training uses
    deterministic offense; evaluation compares the defense against fixed,
    deterministic baselines on a shared seeded batch.
    """
    # Resolve through the caller module at invocation time so monkeypatches and
    # the legacy training module's live helpers remain authoritative.
    torch = _runtime["torch"]
    dist = _runtime["dist"]
    OBS_DIM = _runtime["OBS_DIM"]
    ACTION_DIM = _runtime["ACTION_DIM"]
    STRATEGIC_ACTION_DIM = _runtime["STRATEGIC_ACTION_DIM"]
    _PPO_CONTROL_GROUP = _runtime["_PPO_CONTROL_GROUP"]
    _distributed_state = _runtime["_distributed_state"]
    _train_once = _runtime["_train_once"]
    _evaluate_once = _runtime["_evaluate_once"]
    _task_opponent = _runtime["_task_opponent"]
    _select_checkpoint_population = _runtime["_select_checkpoint_population"]
    _dispatch_opponent_evaluations = _runtime["_dispatch_opponent_evaluations"]
    _opponent_evaluation_paths = _runtime["_opponent_evaluation_paths"]

    if task != "defense":
        raise ValueError("only defense policies are trainable; offense is deterministic")
    if min(generations, num_envs, horizon, population_size, elite_count,
           rollout_steps, epochs, minibatch_size, evaluation_episodes) < 1:
        raise ValueError("generation, environment, horizon, population, PPO, and evaluation sizes must be positive")
    if elite_count > population_size:
        raise ValueError("elite_count cannot exceed opponent population capacity")
    if evaluation_workers < 1:
        raise ValueError("evaluation_workers must be positive")
    if timesteps is not None and timesteps < 1:
        raise ValueError("timesteps must be positive")
    if l2_coef < 0 or not math.isfinite(l2_coef):
        raise ValueError("l2_coef must be finite and non-negative")
    selected_device, rank, world_size, created_process_group = _distributed_state(device)
    if num_envs % world_size:
        raise ValueError(f"global num_envs={num_envs} must divide evenly across {world_size} ranks")
    if minibatch_size % world_size:
        raise ValueError(f"global minibatch_size={minibatch_size} must divide evenly across {world_size} ranks")
    local_num_envs = num_envs // world_size
    out = Path(output)
    archive_dir = out / "opponent-population"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier(group=_PPO_CONTROL_GROUP)
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
            compatible = (prior_payload.get("task") == "defense"
                and prior_payload.get("architecture", "direct") == architecture
                and int(prior_payload.get("obs_dim", OBS_DIM)) == expected_obs
                and int(prior_payload.get("action_dim", ACTION_DIM)) == expected_action)
        except (OSError, RuntimeError, KeyError, ValueError, EOFError):
            compatible = False
        if not compatible:
            # An untagged, offense, or incompatible checkpoint must never be
            # treated as a defense initializer or retained as an opponent.
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
    if per_generation_steps % world_size:
        raise ValueError("per-generation transitions must divide evenly across ranks")
    global_minibatch_size = min(minibatch_size, rollout_batch)
    if global_minibatch_size % world_size:
        raise ValueError("effective global minibatch must divide evenly across ranks")
    fixed_baselines = _runtime["DEFENSE_TRAINING_OPPONENTS"]
    training_baselines = ("offense",)
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

    def write_json(path: Path, value: dict[str, Any]) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2, sort_keys=True))
        temp.replace(path)

    for generation in range(start_generation, generations):
        generation_number = generation + 1
        generation_started = time.perf_counter()
        training_specs = [{"name": mode, "mode": mode}
                          for mode in training_baselines]
        if opponent and opponent != "mixed" and opponent not in {item["mode"] for item in training_specs}:
            training_specs.insert(0, {"name": opponent, "mode": _task_opponent(task, opponent)})
        if world_size > 1:
            shared_specs = [training_specs if rank == 0 else None]
            dist.broadcast_object_list(shared_specs, src=0)
            training_specs = shared_specs[0]
        train_spec = (training_specs[rng.randrange(len(training_specs))] if rank == 0 else None)
        if world_size > 1:
            selected_spec = [train_spec]
            dist.broadcast_object_list(selected_spec, src=0)
            train_spec = selected_spec[0]
        assert train_spec is not None
        if rank == 0:
            mode_counts[train_spec["name"]] = mode_counts.get(train_spec["name"], 0) + 1
        status_context = {
            "algorithm": "generational", "optimizer": "PPO",
            "generation": generation_number, "current_generation": generation_number,
            "total_generations": generations, "population_size": population_size,
            "elite_count": elite_count, "opponent_member": train_spec["name"],
            "opponents": sorted({item["mode"] for item in training_specs}),
            "completed_timesteps_base": generation * per_generation_steps,
            "requested_timesteps_total": requested_timesteps,
            "timing_profile": timing_profile,
        }
        ppo_result = _train_once(task=task, timesteps=per_generation_steps // world_size,
            output=out, seed=seed + generation * 1009, num_envs=local_num_envs,
            device=selected_device, opponent=train_spec["mode"],
            initial_checkpoint=previous_checkpoint, rollout_steps=rollout_steps,
            epochs=epochs, minibatch_size=global_minibatch_size // world_size,
            learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
            clip_coef=clip_coef, value_coef=value_coef, entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm, l2_coef=l2_coef, horizon=horizon,
            architecture=architecture, curriculum=True,
            strategic_rate_hz=strategic_rate_hz,
            status_context=status_context)
        previous_checkpoint = checkpoint
        if world_size > 1:
            # Rank 0 owns the just-written checkpoint; all ranks wait before
            # evaluation or population refresh can read it.
            dist.barrier(group=_PPO_CONTROL_GROUP)
            # Rank 0's atomic checkpoint is the population/evaluation artifact.
            # Do not let the other rank refresh or load the next generation
            # until that checkpoint and its evaluation metadata are published.
            if rank != 0:
                dist.barrier(group=_PPO_CONTROL_GROUP)
                continue
        scenario_seed = 500_000 + generation * 1009
        evaluation_dir = out / "generation-evaluations" / f"generation-{generation_number:04d}"
        opponent_results: dict[str, Any] = {}
        evaluation_specs = [{"name": mode, "mode": mode}
                            for mode in fixed_baselines]
        population_evaluation_started = time.perf_counter()
        def evaluate_spec(index, spec, stream):
            metrics_path, _ = _opponent_evaluation_paths(
                evaluation_dir, spec, index, evaluation_workers > 1)
            try:
                # Isolate each concurrent evaluator's status/playback side
                # effects, then publish only its metrics to the canonical path.
                _, worker_path = _opponent_evaluation_paths(
                    evaluation_dir, spec, index, evaluation_workers > 1)
                metrics = _evaluate_once(checkpoint, task, evaluation_count, scenario_seed,
                    evaluation_count, selected_device, spec["mode"], worker_path,
                    horizon=horizon, strategic_rate_hz=strategic_rate_hz,
                    capture_playback=False)
                if evaluation_workers > 1:
                    write_json(metrics_path, metrics)
                return metrics, None
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                return None, str(exc)

        evaluation_outputs = _dispatch_opponent_evaluations(
            evaluation_specs, evaluate_spec, evaluation_workers, selected_device)
        for spec_index, (spec, (metrics, error)) in enumerate(
                zip(evaluation_specs, evaluation_outputs)):
            metrics_path, _ = _opponent_evaluation_paths(
                evaluation_dir, spec, spec_index, evaluation_workers > 1)
            if error is not None:
                opponent_results[spec["name"]] = {"opponent": spec["mode"],
                    "seed": scenario_seed,
                    "episodes": evaluation_count, "horizon": horizon,
                    "error": error}
                continue
            try:
                own_scores = float(metrics.get("mean_scores") or 0.)
                other_scores = float(metrics.get("mean_opponent_scores") or 0.)
                opponent_results[spec["name"]] = {
                    "opponent": spec["mode"],
                    "mean_return": metrics.get("mean_return"),
                    "elapsed_seconds": metrics.get("elapsed_seconds"),
                    "physics_world_ticks_per_second": metrics.get("physics_world_ticks_per_second"),
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
                    "seed": scenario_seed,
                    "episodes": evaluation_count, "horizon": horizon,
                    "error": str(exc)}
        population_evaluation_elapsed = time.perf_counter() - population_evaluation_started
        # Scenario previews are requested by the dashboard with its selected
        # zone constraint. Keep rollout and opponent evaluation independent of
        # this visualization so the simulator work runs only when requested.
        scenario_playback = None
        scenario_pair = None
        scenario_error = None
        scenario_simulation_elapsed = 0.0
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
            "ppo_timesteps": ppo_result["completed_timesteps"],
            "strategic_decision_rate_hz": ppo_result.get("strategic_decision_rate_hz"),
            "strategic_transitions_per_second": ppo_result.get("strategic_transitions_per_second"),
            "strategic_decisions_per_episode": ppo_result.get("strategic_decisions_per_episode"),
            "simulated_seconds_trained": ppo_result.get("simulated_seconds_trained"),
            "complete_matches_observed": ppo_result.get("complete_matches_observed"),
            "mean_return": ppo_result.get("mean_return"),
            "opponent_exposure_counts": ppo_result.get("opponent_exposure_counts"),
            "entropy": ppo_result.get("entropy"),
            "action_distribution": ppo_result.get("action_distribution"),
            "per_action_selection_rate": ppo_result.get("per_action_selection_rate"),
            "training_elapsed_seconds": ppo_result.get("elapsed_seconds"),
            "generation_wall_clock_seconds": time.perf_counter() - generation_started,
            "transitions_per_second": ppo_result.get("transitions_per_second"),
            "physics_world_ticks_per_second": ppo_result.get("physics_world_ticks_per_second"),
            "timing_profile_seconds": ppo_result.get("timing_profile_seconds"),
            "timing_sync_sensitive_counts": ppo_result.get("timing_sync_sensitive_counts"),
            "scenario_seed": scenario_seed,
            "scenario_playback": scenario_playback,
            "scenario_zone_pair": (list(scenario_pair) if scenario_pair is not None else None),
            "scenario_playback_error": scenario_error,
            "population_evaluation_wall_clock_seconds": population_evaluation_elapsed,
            "scenario_simulation_wall_clock_seconds": scenario_simulation_elapsed,
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
            "task": task, "policy_modes": ["DEFENSE"],
            "generation": generation_number,
            "current_generation": generation_number, "total_generations": generations,
            "population_size": len(archive_entries), "population_capacity": population_size,
            "elite_count": min(elite_count, population_size),
            "opponent": train_spec["mode"],
            "opponent_member": train_spec["name"],
            "opponents": sorted({item["mode"] for item in training_specs}),
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
        if world_size > 1:
            # Release other ranks only after evaluation, population updates,
            # and generation status publication are complete.
            dist.barrier(group=_PPO_CONTROL_GROUP)
    if rank != 0:
        if world_size > 1:
            dist.barrier(group=_PPO_CONTROL_GROUP)
        return {"status": "completed", "rank": rank, "world_size": world_size,
                "task": task, "checkpoint": str(checkpoint),
                "completed_timesteps": generations * per_generation_steps}
    final = {**final_status, "status": "completed", "generation_history": history,
        "generation_history_path": str(history_path), "population_manifest": str(population_path),
        "completed_timesteps": generations * per_generation_steps,
        "requested_timesteps": requested_timesteps,
        "elapsed_seconds": time.perf_counter() - started,
        "opponent_metrics": last_opponent_results,
        "opponent_fitness": {name: values.get("success_rate", values.get("mean_return"))
                              for name, values in last_opponent_results.items()},
        "opponents": sorted({item["mode"] for item in training_specs})}
    write_json(status_path, final)
    write_json(out / "metadata.json", final)
    if world_size > 1:
        dist.barrier(group=_PPO_CONTROL_GROUP)
    return final
