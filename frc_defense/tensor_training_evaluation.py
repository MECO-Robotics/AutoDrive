"""Episode-level evaluation for tensor PPO policies.

This module is imported lazily by :mod:`tensor_training` to preserve its
long-standing public API while keeping the training implementation focused.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# Evaluation depends on training's model/environment construction helpers. Bind
# their current values at each public entry point, preserving test and caller
# monkeypatches while avoiding a module import cycle.
def _bind_training_globals(source: dict[str, Any]) -> None:
    names = (
        "torch", "ActorCritic", "OBS_DIM", "ACTION_DIM", "STRATEGIC_ACTION_DIM",
        "OBS_NORMALIZATION", "MIXED_DEFENSE_OPPONENTS", "_device", "_task_opponent",
        "_evaluate_once", "_wilson_interval", "_env",
        "_legacy_strategic_action",
        "_scripted_game_action", "_reset_obs", "_unevaluated_ablation_row",
        "ALLIANCE_ZONE_DEPTH", "BUMP_ACCELERATION_SCALE", "BUMP_SPEED_SCALE",
        "atomic_json", "static_collision_boxes", "write_playback",
    )
    implementation_names = {
        "_evaluate_once", "_wilson_interval", "_scripted_game_action",
        "_legacy_strategic_action", "_unevaluated_ablation_row",
    }
    for name in names:
        if name in source:
            value = source[name]
            # The legacy module now contains forwarding wrappers for these
            # implementations. Keep this module's implementation unless a
            # caller replaced the wrapper (for example, with a test double).
            is_forwarder = (
                name in implementation_names
                and getattr(value, "__module__", None) == source.get("__name__")
                and getattr(value, "__name__", None) == name
            )
            if is_forwarder:
                continue
            globals()[name] = value

def evaluate(checkpoint: str | Path, *, task: str = "defense",
             episodes: int = 32, seed: int = 1000, num_envs: int = 32,
             device: str = "cuda", opponent: str = "random",
             output: str | Path = "metrics/tensor-evaluation.json",
             horizon: int = 8000) -> dict[str, Any]:
    from . import tensor_training_evaluation_reporting as _reporting
    _reporting._bind_evaluation_globals(globals())
    return _reporting.evaluate(checkpoint, task=task, episodes=episodes, seed=seed,
        num_envs=num_envs, device=device, opponent=opponent, output=output, horizon=horizon)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    from . import tensor_training_evaluation_reporting as _reporting
    return _reporting._wilson_interval(successes, total, z)


def _unevaluated_ablation_row(name: str, task: str, architecture: str | None,
                             opponent: str, group: str, seed: int,
                             checkpoint: str | Path | None,
                             missing_reason: str) -> dict[str, Any]:
    from . import tensor_training_evaluation_reporting as _reporting
    return _reporting._unevaluated_ablation_row(name, task, architecture, opponent,
        group, seed, checkpoint, missing_reason)


def evaluate_game_ablations(*, defense_checkpoint: str | Path | None = None,
        episodes: int = 8,
        seed: int = 4100, num_envs: int = 8, device: str = "cuda",
        output: str | Path = "metrics/ablations.json", horizon: int = 8000,
        strategic_rate_hz: float = 4., capture_playback: bool = True) -> dict[str, Any]:
    from . import tensor_training_evaluation_reporting as _reporting
    _reporting._bind_evaluation_globals(globals())
    return _reporting.evaluate_game_ablations(
        defense_checkpoint=defense_checkpoint,
        episodes=episodes, seed=seed, num_envs=num_envs, device=device, output=output,
        horizon=horizon, strategic_rate_hz=strategic_rate_hz,
        capture_playback=capture_playback)


def _scripted_game_action(env, task: str) -> torch.Tensor:
    """Deterministic defensive action; AD* handles resulting navigation."""
    if task != "defense":
        raise ValueError("scripted evaluation only supports deterministic defense")
    candidates, valid, _ = env._fuel_candidates(0)
    slot = valid.to(torch.int64).argmax(-1)
    return torch.where(valid.any(-1), slot + 1, torch.full_like(slot, 5))


def _legacy_strategic_action(action: torch.Tensor) -> torch.Tensor:
    """Map a historical three-class checkpoint onto the candidate action set."""
    legacy=action.long().clamp(0,2)
    # defense: block, deny, intercept
    return torch.where(legacy==0,torch.full_like(legacy,5),
        torch.where(legacy==1,torch.ones_like(legacy),torch.zeros_like(legacy)))


def _evaluate_once(checkpoint, task, episodes, seed, num_envs, device, opponent,
                   output, horizon=8000, scripted_strategy=None,
                   strategic_rate_hz=4.,
                   capture_playback=True, scenario_zone_pair=None):
    if task != "defense":
        raise ValueError("only defense evaluation is supported")
    if not 2. <= strategic_rate_hz <= 5.:
        raise ValueError("strategic_rate_hz must be between 2 and 5")
    payload = (torch.load(checkpoint, map_location=device, weights_only=True)
               if checkpoint is not None else None)
    if payload is not None and payload.get("task") != "defense":
        raise ValueError("checkpoint must be tagged as a defense policy")
    architecture = (payload.get("architecture", "direct") if payload is not None
                    else "strategic_adstar")
    action_kind = (payload.get("action_kind", "continuous") if payload is not None
                   else "categorical")
    model = None
    if payload is not None:
        model = ActorCritic(payload.get("obs_dim", OBS_DIM),
                            payload.get("action_dim", ACTION_DIM), action_kind).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
    scenario_opponent = ("adstar" if opponent in MIXED_DEFENSE_OPPONENTS else opponent)
    env = _env(num_envs, task, device, seed, scenario_opponent, horizon,
               normalize_observations=bool(payload and
                   payload.get("observation_normalization") == OBS_NORMALIZATION),
               architecture=architecture)
    horizon=int(env.horizon)
    # Planner routes must not determine the defender's initial position or
    # enter its policy inputs during defense evaluation.
    env.adstar_spawn_hint = False
    env.opponent = opponent
    checkpoint_drivetrain_config = payload.get("drivetrain_config") if payload else None
    if (checkpoint_drivetrain_config is not None and
            checkpoint_drivetrain_config != env.drivetrain_config):
        raise ValueError("checkpoint drivetrain configuration does not match this simulator; "
                         "use the same FRC_DRIVETRAIN_CONFIG used during training")
    is_adstar_defense = opponent == "adstar"
    # The traditional "guard" mode is now the same obstacle-aware AD* policy;
    # keep its public opponent name so existing training/evaluation commands
    # and dashboard links continue to work.
    uses_adstar_playback_route = is_adstar_defense
    run_dir = Path(output).parent
    status_path = run_dir / "status.json"
    playback_path = run_dir / "playback.json"
    playback_frames = []
    eval_start = time.perf_counter()

    def atomic_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(path)

    from .field import (ALLIANCE_ZONE_DEPTH, BUMP_ACCELERATION_SCALE,
                        BUMP_SPEED_SCALE, bump_boxes, static_collision_boxes)
    playback_field = {"length": env.sim.field_length, "width": env.sim.field_width,
        "alliance_zone_depth": ALLIANCE_ZONE_DEPTH,
        "elements": [box.as_dict() for box in env.field_boxes],
        "colliders": [box.as_dict() for box in static_collision_boxes(env.field_boxes)],
        "bump_regions": [box.as_dict() for box in bump_boxes(env.field_boxes)],
        "trench_paths": [box.as_dict() for box in env.field_boxes if "_trench_" in box.name and "_support_" not in box.name],
        "trench_supports": [box.as_dict() for box in env.field_boxes if "_trench_support_" in box.name],
        "bump_speed_scale": BUMP_SPEED_SCALE,
        "bump_acceleration_scale": BUMP_ACCELERATION_SCALE}

    playback_scenarios = []
    playback_scenario_done = []
    # Every dashboard mode records the same seeded scenario batch. Endpoint
    # generation lives in TensorDefenseEnv, so policy and opponent modes differ
    # only in motion, never in scenario sampling.
    playback_scenario_count = min(3, num_envs, episodes) if capture_playback else 0
    if playback_scenario_count:
        playback_scenarios = [{"id": str(i + 1), "label": f"Scenario {i + 1}", "frames": []}
            for i in range(playback_scenario_count)]
        playback_scenario_done = [False] * playback_scenario_count

    def write_playback():
        atomic_json(playback_path, {"task": "adstar_attacker_defense" if is_adstar_defense else task,
                                    "scenario_seed": seed, "dt": env.dt,
                                    "field": playback_field, "frames": playback_frames,
                                    "scenarios": playback_scenarios})

    def eval_status(status):
        hold_rate = sum(metric_rows["success"]) / max(1, len(metric_rows["success"]))
        score_times = [value for value in metric_rows["time_to_goal"] if value > 0]
        value = {"status": status, "task": "defense", "opponent": "adstar",
                 "checkpoint": str(checkpoint) if checkpoint else None, "device": str(device),
                 "checkpoint_mtime": Path(checkpoint).stat().st_mtime if checkpoint else None,
                 "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                 "accelerator_backend": "ROCm" if torch.version.hip else ("CUDA" if torch.version.cuda else "CPU"),
                 "num_envs": num_envs,
                 "progress_unit": "episodes", "completed_timesteps": len(scores),
                 "requested_timesteps": episodes, "completed_episodes": len(scores),
                 "requested_episodes": episodes, "adstar_defender_hold_rate": hold_rate,
                 "adstar_attacker_score_rate": len(score_times) / max(1, len(scores)),
                 "adstar_mean_attack_time_to_score": sum(score_times) / max(1, len(score_times)) if score_times else None,
                 "adstar_replan_interval_steps": env.adstar_replan_interval,
                 "elapsed_seconds": time.perf_counter() - eval_start,
                 "drivetrain_config": env.drivetrain_config}
        atomic_json(status_path, value)

    reset_options = None
    if scenario_zone_pair is not None:
        start_zone, goal_zone = scenario_zone_pair
        reset_options = {"start_zone": start_zone, "goal_zone": goal_zone}
    obs = _reset_obs(env.reset(options=reset_options)).to(device=device, dtype=torch.float32)
    planning_latencies = []
    planning_device_events = []
    planning_calls = 0
    timing_stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    for planner_name in ("_adstar_planners", "_adstar_defender_planners",
                         "_adstar_tactical_planner"):
        planner = getattr(env, planner_name, None)
        if planner is None or not callable(getattr(planner, "plan", None)):
            continue
        original_plan = planner.plan
        def timed_plan(*args, _original=original_plan, **kwargs):
            nonlocal planning_calls
            if device.type == "cuda":
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record(timing_stream)
            else:
                started = time.perf_counter()
            result = _original(*args, **kwargs)
            if device.type == "cuda":
                event_end.record(timing_stream)
                planning_device_events.append((event_start, event_end))
            else:
                planning_latencies.append((time.perf_counter() - started) * 1000.)
            planning_calls += 1
            return result
        try:
            planner.plan = timed_plan
        except (AttributeError, TypeError):
            pass
    for i, scenario in enumerate(playback_scenarios):
        scenario["start"] = env.sim.pose[i, 1, :2].detach().cpu().tolist()
        scenario["goal"] = (None if architecture == "strategic_adstar"
                             else env.goal[i].detach().cpu().tolist())
    totals = torch.zeros(num_envs, device=device)
    lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    base_quota, extra_quota = divmod(episodes, num_envs)
    episode_quota = torch.full((num_envs,), base_quota, dtype=torch.long, device=device)
    if extra_quota:
        episode_quota[:extra_quota] += 1
    episode_counts = torch.zeros((num_envs,), dtype=torch.long, device=device)
    active = episode_quota > 0
    active_episode_count = min(episodes, num_envs)
    scores, episode_lengths = [], []
    metric_names = ("contact", "opponent_contact", "field_contact", "wall_contact",
                    "opponent_field_contact", "opponent_wall_contact", "opponent_static_contact",
                    "static_contact_started", "contact_duration", "contact_count", "time_blocked", "time_to_goal",
                    "success", "defensive_delay", "useful_position", "path_length",
                    "command_smoothness", "spin_rate_ratio", "maneuver_penalty",
                    "path_efficiency", "out_of_bounds")
    metric_totals = {name: torch.zeros(num_envs,device=device) for name in metric_names}
    metric_rows = {name: [] for name in metric_names}
    game_event_names = ("fuel_acquired_event", "fuel_scored_event",
                        "fuel_denied_event", "fuel_abandoned_event")
    game_event_totals = {name: torch.zeros((num_envs, 2), device=device)
                         for name in game_event_names}
    game_metric_rows = {name: [] for name in (
        "acquisitions", "scores", "denied_objectives", "abandoned_objectives",
        "cycle_time", "opponent_acquisitions", "opponent_scores",
        "opponent_denied_objectives", "opponent_abandoned_objectives",
        "total_simulated_score", "passes_attempted", "passes_completed",
        "fuel_transferred", "failed_passes", "pass_to_score_cycle_time")}
    acquired_at = torch.zeros((num_envs, env.piece_owner.shape[1], 2), device=device)
    previous_piece_owner = env.piece_owner.clone()
    cycle_time_sum = torch.zeros((num_envs, 2), device=device)
    cycle_count = torch.zeros((num_envs, 2), device=device)
    inference_latencies = []
    inference_device_events = []
    strategic = env.action_mode == "strategic"
    strategic_phase = 0.0
    strategic_episode_ticks = 0
    strategic_ticks_remaining = 0
    strategic_decision_due = True
    pending_refresh = torch.zeros(num_envs, dtype=torch.bool, device=device)
    pending_refresh_any = False
    held_actions = None
    run_dir.mkdir(parents=True, exist_ok=True)
    if capture_playback:
        write_playback()
    if is_adstar_defense:
        eval_status("evaluating")
    while len(scores) < episodes:
        decision_mask = active if not strategic else (
            active if strategic_decision_due else pending_refresh & active)
        needs_policy_decision = (not strategic or strategic_decision_due or
                                 pending_refresh_any)
        if needs_policy_decision and model is not None:
            if device.type == "cuda":
                inference_event_start = torch.cuda.Event(enable_timing=True)
                inference_event_end = torch.cuda.Event(enable_timing=True)
                inference_event_start.record(timing_stream)
            else:
                inference_start=time.perf_counter()
            with torch.no_grad():
                model_obs=obs[:,:model.trunk[0].in_features]
                action_mask=(model_obs[:,-STRATEGIC_ACTION_DIM:].bool() if action_kind=="categorical" and
                    model.actor.out_features==STRATEGIC_ACTION_DIM else None)
                actions, _, _ = model.sample(model_obs, deterministic=True,action_mask=action_mask)
                if env.action_mode=="strategic" and model.actor.out_features==3:
                    actions=_legacy_strategic_action(actions)
            if device.type == "cuda":
                inference_event_end.record(timing_stream)
                inference_device_events.append((inference_event_start, inference_event_end))
            else:
                inference_latencies.append((time.perf_counter()-inference_start)*1000.)
            if strategic:
                if held_actions is None:
                    held_actions = actions
                else:
                    selector = decision_mask.reshape((-1,) + (1,) * (actions.ndim - 1))
                    held_actions = torch.where(selector, actions, held_actions)
                pending_refresh &= ~decision_mask
                pending_refresh_any = False
        elif needs_policy_decision and scripted_strategy == "defense":
            actions = _scripted_game_action(env, task)
            if strategic:
                if held_actions is None:
                    held_actions = actions
                else:
                    selector = decision_mask.reshape((-1,) + (1,) * (actions.ndim - 1))
                    held_actions = torch.where(selector, actions, held_actions)
                pending_refresh &= ~decision_mask
                pending_refresh_any = False
        elif needs_policy_decision:
            raise ValueError("evaluation needs a valid checkpoint or an explicit scripted strategy")
        if strategic:
            actions = held_actions
            if strategic_decision_due:
                strategic_phase += 50. / strategic_rate_hz
                strategic_ticks_remaining = max(1, int(strategic_phase))
                strategic_phase -= strategic_ticks_remaining
                strategic_decision_due = False
        next_obs, rewards, dones, truncated, info = env.step(
            actions, active_mask=active,
            _active_count=active_episode_count if strategic else None,
            )
        active_float = active.to(device=device, dtype=torch.float32)
        totals += rewards.to(device=device, dtype=torch.float32).reshape(num_envs) * active_float
        lengths += active.to(device=device, dtype=torch.long)
        for name in metric_names:
            value=info.get(name)
            if value is not None:
                metric_totals[name]+=torch.as_tensor(value,device=device,dtype=torch.float32).reshape(num_envs) * active_float
        for name in game_event_names:
            value = info.get(name)
            if value is not None:
                game_event_totals[name] += torch.as_tensor(value, device=device,
                    dtype=torch.float32).reshape(num_envs, 2) * active_float[:, None]
        current_owner = env.piece_owner
        # Use simulated seconds for robot acquisition-to-score cycles; the
        # strategic match clock may be compressed to fit the rollout horizon.
        episode_clock = lengths.to(dtype=torch.float32) * env.dt
        for robot in (0, 1):
            newly_acquired = (current_owner == robot) & (previous_piece_owner != robot)
            acquired_at[:, :, robot] = torch.where(newly_acquired & active[:, None],
                episode_clock[:, None], acquired_at[:, :, robot])
            just_scored = (current_owner == -2) & (previous_piece_owner == robot)
            elapsed_held = (episode_clock[:, None] - acquired_at[:, :, robot]).clamp_min(0.)
            cycle_time_sum[:, robot] += (elapsed_held * just_scored.float()).sum(-1) * active_float
            cycle_count[:, robot] += just_scored.sum(-1).float() * active_float
            released = (previous_piece_owner == robot) & (current_owner != robot)
            acquired_at[:, :, robot] = torch.where(released, torch.zeros_like(acquired_at[:, :, robot]),
                                                    acquired_at[:, :, robot])
        previous_piece_owner.copy_(current_owner)
        strategic_episode_boundary = False
        if strategic:
            strategic_episode_ticks += 1
            strategic_episode_boundary = strategic_episode_ticks >= horizon
            if strategic_episode_boundary:
                # Strategic matches only truncate at the shared physics
                # horizon. Avoid a device-side `nonzero` and host read on
                # every 20 ms tick in this aligned evaluation path.
                finished = active & truncated.to(device=device, dtype=torch.bool).reshape(num_envs)
            else:
                finished = torch.zeros_like(active)
        else:
            finished = active & (dones.to(device=device, dtype=torch.bool).reshape(num_envs) | truncated.to(device=device, dtype=torch.bool).reshape(num_envs))
        if playback_scenario_count:
            # Keep a few independent worlds as separate rollouts; never splice
            # parallel worlds into one playback timeline.
            sample_finished = False
            wrote_frame = False
            for i in range(playback_scenario_count):
                if playback_scenario_done[i]:
                    continue
                if strategic:
                    # The first playback rows are active for the first full
                    # strategic match by construction of playback_scenario_count.
                    sample_finished = strategic_episode_boundary
                    step_index = strategic_episode_ticks
                else:
                    sample_finished = bool(finished[i].item())
                    step_index = int(lengths[i].item())
                playback_stride = max(1, horizon // 500) if strategic else 3
                if step_index % playback_stride == 0 or sample_finished:
                    paths = info.get("adstar_paths", [])
                    path_lengths = info.get("adstar_path_lengths", [])
                    intercepts = info.get("predicted_intercepts", [])
                    intercept_times = info.get("predicted_intercept_times", [])
                    if uses_adstar_playback_route and len(paths):
                        path_count=int(path_lengths[i].item()) if len(path_lengths) else paths.shape[1]
                        route=paths[i,:path_count].detach().cpu().tolist()
                        predicted_intercept=intercepts[i].detach().cpu().tolist() if is_adstar_defense and len(intercepts) else None
                        predicted_intercept_time=float(intercept_times[i].item()) if is_adstar_defense and len(intercept_times) else None
                    else:
                        route=[]
                        predicted_intercept=None
                        predicted_intercept_time=None
                    pose = env.sim.pose[i].detach().cpu().tolist()
                    sizes = torch.stack((env.sim.length[i], env.sim.width[i]), -1).reshape(-1).detach().cpu().tolist()
                    own_effort=((actions[i, :2] * env.sim.speed[i, 0]).detach().cpu().tolist()
                                if actions.ndim >= 2 and actions.shape[-1] >= 2
                                else env.sim.velocity[i, 0, :2].detach().cpu().tolist())
                    game_frame={}
                    for key, attribute in (("hub_active", "hub_active"),
                                           ("match_remaining", "match_remaining"), ("match_elapsed", "match_elapsed"),
                                           ("fuel_score_count", "fuel_score_count")):
                        value=getattr(env,attribute,None)
                        if value is not None:
                            item=value[i]
                            game_frame[key]=item.detach().cpu().tolist() if isinstance(item,torch.Tensor) else item
                    if env.action_mode == "strategic":
                        game_frame["hub_centers"] = env.hub_centers.detach().cpu().tolist()
                        active_fuel = env.piece_active[i]
                        fuel_pieces = torch.cat((env.piece_pos[i],
                            env.piece_owner[i,:,None].to(env.piece_pos.dtype)), -1)
                        game_frame["fuel_pieces"] = fuel_pieces[active_fuel].detach().cpu().tolist()
                        controlled = info.get("controlled_adstar_path", [])
                        controlled_lengths = info.get("controlled_adstar_path_lengths", [])
                        if len(controlled):
                            own_count = (int(controlled_lengths[i].item()) if len(controlled_lengths)
                                         else controlled.shape[1])
                            own_path = controlled[i,:own_count].detach().cpu().tolist()
                            game_frame["adstar_paths"] = [own_path, route]
                    strategic = env.action_mode == "strategic"
                    frame = {"robots": pose,
                    "goal": None if strategic else env.goal[i].detach().cpu().tolist(),
                    "sizes": sizes, "goal_radius": None if strategic else float(env.goal_radius[i].item()),
                    "chassis_effort_vector": own_effort,
                    "robot_effort_vectors": [
                        own_effort,
                        info.get("opponent_effort_vector", torch.zeros((num_envs, 2), device=device))[i].detach().cpu().tolist()],
                    **({} if strategic else {"adstar_path": [list(point) for point in route]}),
                    "predicted_intercept": predicted_intercept,
                    "predicted_intercept_time": predicted_intercept_time}
                    frame.update(game_frame)
                    playback_scenarios[i]["frames"].append(frame)
                    if i == 0:
                        playback_frames.append(frame)
                    wrote_frame = True
                if sample_finished:
                    playback_scenario_done[i] = True
            if (wrote_frame and scenario_zone_pair is None and
                    (len(playback_scenarios[0]["frames"]) % 10 == 0 or any(playback_scenario_done))):
                write_playback()
        finished_indices = (torch.nonzero(finished, as_tuple=False).flatten()
            if (not strategic or strategic_episode_boundary) else ())
        if (strategic and strategic_episode_boundary and
                len(finished_indices) != active_episode_count):
            raise RuntimeError("strategic evaluation worlds did not truncate together at the horizon")
        for index in finished_indices:
            if len(scores) < episodes:
                i = int(index.item())
                scores.append(float(totals[i].item()))
                length_i=int(lengths[i].item())
                episode_lengths.append(length_i)
                episode_counts[i] += 1
                for name in metric_names:
                    value=float(metric_totals[name][i].item())
                    metric_rows[name].append(value/length_i if name=="contact" else value)
                for event_name, row_name in (("fuel_acquired_event", "acquisitions"),
                    ("fuel_scored_event", "scores"), ("fuel_denied_event", "denied_objectives")):
                    game_metric_rows[row_name].append(float(game_event_totals[event_name][i, 0].item()))
                    opponent_row = "opponent_" + row_name
                    game_metric_rows[opponent_row].append(float(game_event_totals[event_name][i, 1].item()))
                game_metric_rows["abandoned_objectives"].append(None)
                game_metric_rows["opponent_abandoned_objectives"].append(None)
                cycle_n = float(cycle_count[i, 0].item())
                game_metric_rows["cycle_time"].append(
                    float(cycle_time_sum[i, 0].item()) / cycle_n if cycle_n else None)
                game_metric_rows["total_simulated_score"].append(
                    float(game_event_totals["fuel_scored_event"][i].sum().item()))
                # This environment is currently 1v1 and has no allied receiver;
                # report passing as unavailable rather than silently implying
                # that a no-pass result is a learned behavior.
                for name in ("passes_attempted","passes_completed","fuel_transferred",
                             "failed_passes","pass_to_score_cycle_time"):
                    game_metric_rows[name].append(None)
                totals[i] = 0
                lengths[i] = 0
                for values in metric_totals.values():
                    values[i]=0
                for values in game_event_totals.values():
                    values[i].zero_()
                acquired_at[i].zero_()
                cycle_time_sum[i].zero_()
                cycle_count[i].zero_()
                previous_piece_owner[i].fill_(-1)
                if is_adstar_defense:
                    eval_status("evaluating")
        reset_done = getattr(env, "reset_done", None)
        if reset_done is None:
            raise RuntimeError("TensorDefenseEnv must provide reset_done(mask) for per-world episode resets")
        active &= ~finished
        reset_mask = finished & (episode_counts < episode_quota)
        reset_count = 0
        if strategic and strategic_episode_boundary:
            reset_count = int(reset_mask.sum().item()) if len(scores) < episodes else 0
            has_reset = reset_count > 0
        else:
            has_reset = len(scores) < episodes and bool(reset_mask.any().item())
        if has_reset:
            active |= reset_mask
            if strategic:
                pending_refresh |= reset_mask
                pending_refresh_any = True
            next_obs = reset_done(reset_mask)
            totals.masked_fill_(reset_mask, 0.)
            lengths.masked_fill_(reset_mask, 0)
            for values in metric_totals.values():
                values.masked_fill_(reset_mask, 0.)
            for values in game_event_totals.values():
                values[reset_mask] = 0.
            acquired_at[reset_mask] = 0.
            cycle_time_sum[reset_mask] = 0.
            cycle_count[reset_mask] = 0.
            previous_piece_owner[reset_mask] = env.piece_owner[reset_mask]
        if strategic and strategic_episode_boundary:
            active_episode_count += reset_count - len(finished_indices)
            strategic_episode_ticks = 0
        if strategic:
            strategic_ticks_remaining -= 1
            if strategic_ticks_remaining <= 0:
                strategic_decision_due = True
        obs = next_obs.to(device=device, dtype=torch.float32)
    if planning_device_events or inference_device_events:
        # Resolve event timings once after the rollout instead of stalling the
        # device around every AD* plan and policy decision.
        torch.cuda.synchronize(device)
        planning_latencies.extend(start.elapsed_time(end)
                                  for start, end in planning_device_events)
        inference_latencies.extend(start.elapsed_time(end)
                                  for start, end in inference_device_events)
    if capture_playback:
        write_playback()
    result = {"task": task, "opponent": opponent, "checkpoint": str(checkpoint) if checkpoint else None,
              "checkpoint_mtime": Path(checkpoint).stat().st_mtime if checkpoint else None,
              "episodes": episodes, "seed": seed, "device": str(device),
              "horizon": env.horizon, "match_duration_seconds": env.horizon*env.dt,
              "strategic_decision_rate_hz": strategic_rate_hz if strategic else 50.,
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "accelerator_backend": "ROCm" if torch.version.hip else ("CUDA" if torch.version.cuda else "CPU"),
              "drivetrain_config": env.drivetrain_config,
              "mean_return": sum(scores)/len(scores),
              "mean_episode_length": sum(episode_lengths)/len(episode_lengths),
              "returns": scores, "episode_lengths": episode_lengths}
    evaluation_elapsed = max(time.perf_counter() - eval_start, 1e-12)
    result["elapsed_seconds"] = evaluation_elapsed
    result["physics_world_ticks_per_second"] = sum(episode_lengths) / evaluation_elapsed
    result.update({"mean_"+name:sum(values)/len(values) for name,values in metric_rows.items() if values})
    def optional_mean(values):
        available = [float(value) for value in values if value is not None]
        return sum(available) / len(available) if available else None
    result.update({
        "mean_acquisitions": optional_mean(game_metric_rows["acquisitions"]),
        "mean_scores": optional_mean(game_metric_rows["scores"]),
        "mean_cycle_time": optional_mean(game_metric_rows["cycle_time"]),
        "mean_denied_objectives": optional_mean(game_metric_rows["denied_objectives"]),
        "mean_abandoned_objectives": optional_mean(game_metric_rows["abandoned_objectives"]),
        "mean_opponent_acquisitions": optional_mean(game_metric_rows["opponent_acquisitions"]),
        "mean_opponent_scores": optional_mean(game_metric_rows["opponent_scores"]),
        "mean_opponent_denied_objectives": optional_mean(game_metric_rows["opponent_denied_objectives"]),
        "mean_opponent_abandoned_objectives": optional_mean(game_metric_rows["opponent_abandoned_objectives"]),
        "mean_total_score": optional_mean(game_metric_rows["scores"]),
        "mean_total_simulated_score": optional_mean(game_metric_rows["total_simulated_score"]),
        "mean_passes_attempted": optional_mean(game_metric_rows["passes_attempted"]),
        "mean_passes_completed": optional_mean(game_metric_rows["passes_completed"]),
        "mean_fuel_transferred": optional_mean(game_metric_rows["fuel_transferred"]),
        "mean_failed_passes": optional_mean(game_metric_rows["failed_passes"]),
        "mean_pass_to_score_cycle_time": optional_mean(game_metric_rows["pass_to_score_cycle_time"]),
        "score_semantics": "scored FUEL piece count; simulator does not assign official match points",
        "cycle_time_units": "simulated seconds from piece acquisition to scoring",
        "unavailable_metrics": ["abandoned_objectives (not modeled by simulator)",
            "passing (no allied teammate entity in the current 1v1 simulator)"],
        "game_metrics_by_episode": game_metric_rows,
        "planning_latency_ms_mean": optional_mean(planning_latencies),
        "planning_latency_ms_p95": (sorted(planning_latencies)[min(len(planning_latencies)-1,
            int(.95*(len(planning_latencies)-1)))] if planning_latencies else None),
        "planning_calls": planning_calls,
        "scripted_strategy": scripted_strategy,
    })
    if is_adstar_defense:
        hold_rate = sum(metric_rows["success"]) / max(1, len(metric_rows["success"]))
        score_times = [value for value in metric_rows["time_to_goal"] if value > 0]
        result.update({"adstar_defender_hold_rate": hold_rate,
                       "adstar_attacker_score_rate": len(score_times) / episodes,
                       "adstar_mean_attack_time_to_score": sum(score_times) / max(1, len(score_times)) if score_times else None,
                       "adstar_replan_interval_steps": env.adstar_replan_interval})
    sorted_latency=sorted(inference_latencies)
    result["inference_latency_ms_mean"]=sum(inference_latencies)/max(1,len(inference_latencies))
    result["inference_latency_ms_p95"]=sorted_latency[min(len(sorted_latency)-1,int(.95*(len(sorted_latency)-1)))] if sorted_latency else 0.
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    if is_adstar_defense:
        eval_status("completed")
    return result
