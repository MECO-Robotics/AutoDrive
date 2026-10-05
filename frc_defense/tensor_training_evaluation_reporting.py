"""High-level evaluation reports and ablation comparisons.

The episode evaluator lives in :mod:`tensor_training_evaluation`; this module
owns report orchestration and JSON row/schema construction.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


def _bind_evaluation_globals(source: dict[str, Any]) -> None:
    """Bind current evaluator dependencies, including caller monkeypatches."""
    implementation_names = {"_wilson_interval", "_unevaluated_ablation_row"}
    for name in ("torch", "MIXED_DEFENSE_OPPONENTS", "_device",
                 "_task_opponent", "_evaluate_once", "_wilson_interval",
                 "_unevaluated_ablation_row"):
        if name not in source:
            continue
        value = source[name]
        is_forwarder = (
            name in implementation_names
            and getattr(value, "__module__", None) == source.get("__name__")
            and getattr(value, "__name__", None) == name
        )
        if not is_forwarder:
            globals()[name] = value


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



def _unevaluated_ablation_row(name: str, task: str, architecture: str | None,
                             opponent: str, group: str, seed: int,
                             checkpoint: str | Path | None,
                             missing_reason: str) -> dict[str, Any]:
    """Build the shared report row for an ablation that could not run."""
    return {
        "name": name,
        "task": task,
        "architecture": architecture,
        "role": "offense" if task == "counter_defense" else "defense",
        "opponent": opponent,
        "episodes": 0,
        "seed": seed,
        "seeds": [],
        "comparison_group": group,
        "comparable": False,
        "scenario_comparability": "not evaluable",
        "evaluated": False,
        "missing_reason": missing_reason,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }


def evaluate_game_ablations(*, offense_checkpoint: str | Path | None = None,
        defense_checkpoint: str | Path | None = None, episodes: int = 8,
        seed: int = 4100, num_envs: int = 8, device: str = "cuda",
        output: str | Path = "metrics/ablations.json", horizon: int = 8000,
        strategic_rate_hz: float = 4., capture_playback: bool = True) -> dict[str, Any]:
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
            evaluations.append(_unevaluated_ablation_row(
                name, task, architecture, opponent, group, seed, checkpoint,
                missing_reason))
            continue
        mode_output = output_path.parent / f"{output_path.stem}-{name}" / "metrics.json"
        try:
            metrics = _evaluate_once(checkpoint, task, episodes, seed, matched_envs,
                selected, opponent, mode_output, horizon=horizon,
                scripted_strategy=scripted_strategy,
                strategic_rate_hz=strategic_rate_hz,
                capture_playback=capture_playback)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            evaluations.append(_unevaluated_ablation_row(
                name, task, architecture, opponent, group, seed, checkpoint,
                f"Evaluation failed: {exc}"))
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
