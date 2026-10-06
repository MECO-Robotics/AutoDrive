"""PPO rollout and optimization implementation for tensor training.

Imported lazily by :mod:`tensor_training`; runtime dependencies are supplied
by that module so its public monkeypatch seams remain live.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def train_once(task: str, timesteps: int, output: str | Path, *, seed: int,
               num_envs: int, device, opponent: str, initial_checkpoint,
               rollout_steps: int, epochs: int, minibatch_size: int,
               learning_rate: float, gamma: float, gae_lambda: float,
               clip_coef: float, value_coef: float, entropy_coef: float,
               max_grad_norm: float, l2_coef: float, horizon: int,
               architecture: str = "direct", curriculum: bool = True,
               strategic_rate_hz: float = 4.0, capture_training_playback=None,
               reuse_strategic_own_candidates=None,
               ppo_sparse_delayed_observation_capture: bool = False,
               status_context=None, _runtime: dict[str, Any]) -> dict[str, Any]:
    """Run one PPO train call using the legacy module's current bindings."""
    dist = _runtime["dist"]
    torch = _runtime["torch"]
    _env = _runtime["_env"]
    STATIC_OPPONENT_FRACTION = _runtime["STATIC_OPPONENT_FRACTION"]
    OBS_DIM = _runtime["OBS_DIM"]
    ACTION_DIM = _runtime["ACTION_DIM"]
    ActorCritic = _runtime["ActorCritic"]
    _DDPScore = _runtime["_DDPScore"]
    nn = _runtime["nn"]
    _reset_obs = _runtime["_reset_obs"]
    time = _runtime["time"]
    Path = _runtime["Path"]
    CURRICULUM_STAGES = _runtime["CURRICULUM_STAGES"]
    OBS_NORMALIZATION = _runtime["OBS_NORMALIZATION"]
    _curriculum_stage = _runtime["_curriculum_stage"]
    _curriculum_opponents = _runtime["_curriculum_opponents"]
    STRATEGIC_ACTION_DIM = _runtime["STRATEGIC_ACTION_DIM"]
    _held_action_interval = _runtime["_held_action_interval"]
    _gae_advantages = _runtime["_gae_advantages"]
    math = _runtime["math"]
    json = _runtime["json"]
    if task != "defense":
        raise ValueError("only defense policies are trainable; offense is deterministic")
    if opponent not in (*DEFENSE_TRAINING_OPPONENTS, "mixed"):
        raise ValueError("opponent must be a deterministic defense-training baseline")
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    rank_seed = seed + rank * 1_000_003
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 2. <= strategic_rate_hz <= 5.:
        raise ValueError("strategic_rate_hz must be between 2 and 5")
    torch.manual_seed(rank_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(rank_seed)
    env = _env(num_envs, task, device, rank_seed, opponent, horizon,
               static_opponent_fraction=STATIC_OPPONENT_FRACTION,
               architecture=architecture,
               reuse_strategic_own_candidates=reuse_strategic_own_candidates)
    horizon=int(env.horizon)
    obs_dim = int(getattr(env, "obs_dim", OBS_DIM))
    action_dim = int(getattr(env, "action_dim", ACTION_DIM))
    action_kind = "categorical" if architecture == "strategic_adstar" else "continuous"
    model = ActorCritic(obs_dim, action_dim, action_kind).to(device)
    if initial_checkpoint is not None:
        payload = torch.load(initial_checkpoint, map_location=device, weights_only=True)
        if payload.get("task") != "defense":
            raise ValueError("initial checkpoint must be tagged as a defense policy")
        if payload.get("drivetrain_config") != env.drivetrain_config:
            raise ValueError("initial checkpoint drivetrain configuration does not match this run; "
                             "retrain without that checkpoint or use its exact FRC_DRIVETRAIN_CONFIG")
        if payload.get("architecture", "direct") != architecture:
            raise ValueError("initial checkpoint architecture differs from this training run")
        model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    score_model = _DDPScore(model)
    if distributed:
        score_model = nn.parallel.DistributedDataParallel(
            score_model, device_ids=[device.index], output_device=device.index,
            broadcast_buffers=False, find_unused_parameters=False)
    obs = _reset_obs(env.reset())
    env._ppo_sparse_delayed_observation_capture = bool(
        ppo_sparse_delayed_observation_capture and
        architecture == "strategic_adstar" and
        env._full_observation_hip_available)
    if not isinstance(obs, torch.Tensor):
        raise TypeError("TensorDefenseEnv.reset() must return a Torch tensor")
    obs = obs.to(device=device, dtype=torch.float32)
    start = time.perf_counter()
    started_at = time.time()
    updates = (timesteps + rollout_steps * num_envs - 1) // (rollout_steps * num_envs)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "policy.pt"
    status_path = out / "status.json"
    playback_path = out / "playback.json"
    status_context = dict(status_context or {})
    if capture_training_playback is None:
        # Strategic playback is generated as a seeded, zone-constrained GPU
        # scenario after each PPO generation. Avoid synchronizing/copying
        # training trajectories to CPU every update unless explicitly requested.
        capture_training_playback = architecture != "strategic_adstar"
    if distributed and rank != 0:
        capture_training_playback = False
    completed_timesteps_base = int(status_context.get("completed_timesteps_base", 0))
    requested_timesteps_total = int(status_context.get("requested_timesteps_total", timesteps))
    global_num_envs = num_envs * world_size
    global_timesteps = timesteps * world_size
    playback_frames: list[dict[str, Any]] = []
    def write_json(path: Path, value: dict[str, Any]) -> None:
        if rank != 0:
            return
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(path)
    def write_checkpoint(value: dict[str, Any]) -> None:
        if rank != 0:
            return
        temp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        torch.save(value, temp)
        temp.replace(checkpoint)
    if rank == 0:
      write_json(status_path, {**status_context, "status": "running", "task": task, "opponent": opponent,
        "architecture": architecture, "action_kind": action_kind,
        "curriculum": list(CURRICULUM_STAGES) if curriculum else [],
        "stationary_opponent_fraction": STATIC_OPPONENT_FRACTION,
        "requested_timesteps": requested_timesteps_total, "num_envs": global_num_envs,
        "local_num_envs": num_envs, "world_size": world_size, "device": str(device),
        "observation_normalization": OBS_NORMALIZATION,
        "l2_coefficient": l2_coef,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "completed_timesteps": completed_timesteps_base, "updates": 0, "total_updates": updates,
        "started_at": started_at,
        "capture_training_playback": capture_training_playback,
        "reuse_strategic_own_candidates": env.reuse_strategic_own_candidates,
        "drivetrain_config": env.drivetrain_config})
      write_json(playback_path, {"task": task, "dt": env.dt, "frames": []})
    strategic_phase = 0.0
    strategic_episode_ticks = 0
    active_world_physics_ticks = 0
    complete_matches = torch.zeros((), dtype=torch.long, device=device)
    action_counts = torch.zeros(action_dim, device=device)
    opponent_exposure: dict[str, int] = {}
    entropy_mean = 0.0
    episode_return_accum = torch.zeros(num_envs, device=device)
    completed_return_sum = torch.zeros((), device=device)
    completed_episode_count = torch.zeros((), device=device)
    timing_enabled = bool(status_context.get("timing_profile", False))
    timing_seconds = {"environment_step_host": 0.0, "simulator_step_host": 0.0,
                      "adstar_planning_host": 0.0,
                      "strategic_observation_host": 0.0,
                      "raw_observation_host": 0.0,
                      "perception_update_host": 0.0,
                      "gamepiece_update_host": 0.0,
                      "fuel_candidate_host": 0.0,
                      "strategic_target_host": 0.0,
                      "policy_inference_host": 0.0, "ppo_update_host": 0.0,
                      "active_mask_bookkeeping_host": 0.0}
    timing_sync_counts = {"active_mask_count_item_reads": 0,
                          "active_count_schedule_device_syncs": 0,
                          "episode_end_any_item_reads": 0,
                          "entropy_item_reads": 0, "update_device_syncs": 0}
    update_gpu_events = []
    if timing_enabled:
        original_step = env.step
        def timed_env_step(*args, **kwargs):
            before = time.perf_counter()
            result = original_step(*args, **kwargs)
            timing_seconds["environment_step_host"] += time.perf_counter() - before
            return result
        env.step = timed_env_step
        original_observation = env._obs
        def timed_observation(*args, **kwargs):
            before = time.perf_counter()
            result = original_observation(*args, **kwargs)
            timing_seconds["strategic_observation_host"] += time.perf_counter() - before
            return result
        env._obs = timed_observation
        original_raw_observation = env._raw_obs
        def timed_raw_observation(*args, **kwargs):
            before = time.perf_counter()
            result = original_raw_observation(*args, **kwargs)
            timing_seconds["raw_observation_host"] += time.perf_counter() - before
            return result
        env._raw_obs = timed_raw_observation
        for method_name, metric_name in (
                ("_update_perception", "perception_update_host"),
                ("_update_gamepieces", "gamepiece_update_host"),
                ("_fuel_candidates", "fuel_candidate_host"),
                ("_strategic_target", "strategic_target_host")):
            original_method = getattr(env, method_name, None)
            if not callable(original_method):
                continue
            def timed_method(*args, _original=original_method,
                             _metric=metric_name, **kwargs):
                before = time.perf_counter()
                result = _original(*args, **kwargs)
                timing_seconds[_metric] += time.perf_counter() - before
                return result
            setattr(env, method_name, timed_method)
        original_sim_step = env.sim.step
        def timed_sim_step(*args, **kwargs):
            before = time.perf_counter()
            result = original_sim_step(*args, **kwargs)
            timing_seconds["simulator_step_host"] += time.perf_counter() - before
            return result
        env.sim.step = timed_sim_step
        for planner_name in ("_adstar_planners", "_adstar_defender_planners",
                             "_adstar_tactical_planner"):
            planner = getattr(env, planner_name, None)
            if planner is None or not callable(getattr(planner, "plan", None)):
                continue
            original_plan = planner.plan
            def timed_plan(*args, _original=original_plan, **kwargs):
                before = time.perf_counter()
                result = _original(*args, **kwargs)
                timing_seconds["adstar_planning_host"] += time.perf_counter() - before
                return result
            try:
                planner.plan = timed_plan
            except (AttributeError, TypeError):
                pass
    for update_index in range(updates):
        stage = _curriculum_stage(update_index / max(1, updates)) if curriculum else -1
        if curriculum:
            candidates = _curriculum_opponents(task, stage)
            env.opponent = candidates[update_index % len(candidates)]
            env.static_opponent_fraction = (.4 if stage == 0 else STATIC_OPPONENT_FRACTION)
        exposure_label = env.opponent
        opponent_exposure[exposure_label] = opponent_exposure.get(exposure_label, 0) + rollout_steps * num_envs
        action_shape = (action_dim,) if action_kind == "continuous" else ()
        action_dtype = torch.float32 if action_kind == "continuous" else torch.long
        b_obs = torch.empty((rollout_steps, num_envs, obs_dim), device=device)
        b_actions = torch.empty((rollout_steps, num_envs, *action_shape), device=device,
                                dtype=action_dtype)
        b_action_masks=(torch.empty((rollout_steps,num_envs,action_dim),device=device,dtype=torch.bool)
                        if action_kind=="categorical" else None)
        b_logprobs = torch.empty((rollout_steps, num_envs), device=device)
        b_rewards = torch.empty((rollout_steps, num_envs), device=device)
        b_dones = torch.empty((rollout_steps, num_envs), device=device, dtype=torch.bool)
        b_truncated = torch.empty((rollout_steps, num_envs), device=device, dtype=torch.bool)
        b_values = torch.empty((rollout_steps, num_envs), device=device)
        b_next_values = torch.empty((rollout_steps, num_envs), device=device)
        entropy_accumulator = torch.zeros((), device=device)
        entropy_batch_count = 0
        update_frames = []
        frame_stride = max(1, rollout_steps // 24)
        for t in range(rollout_steps):
            b_obs[t] = obs
            with torch.no_grad():
                action_mask=(obs[:,-action_dim:].bool() if action_kind=="categorical" and
                             action_dim==STRATEGIC_ACTION_DIM else None)
                inference_started = time.perf_counter() if timing_enabled else 0.
                action, logprob, value = model.sample(obs,action_mask=action_mask)
                if timing_enabled:
                    timing_seconds["policy_inference_host"] += time.perf_counter() - inference_started
                if action_kind == "categorical":
                    action_counts += torch.bincount(action, minlength=action_dim)
                if architecture == "strategic_adstar" and t > 0:
                    # For a continuing transition, next_obs is exactly this
                    # decision's obs and the policy is unchanged until PPO
                    # updates after the rollout. Reuse its value instead of a
                    # second full ActorCritic forward on every physics interval.
                    previous_ended = b_dones[t - 1] | b_truncated[t - 1]
                    b_next_values[t - 1] = torch.where(
                        previous_ended, b_next_values[t - 1], value)
            if b_action_masks is not None:
                b_action_masks[t]=action_mask
            if architecture == "strategic_adstar":
                strategic_phase += 50.0 / strategic_rate_hz
                ticks_this_decision = max(1, int(strategic_phase))
                strategic_phase -= ticks_this_decision
            else:
                ticks_this_decision = 1
            if architecture == "strategic_adstar":
                interval_started = time.perf_counter() if timing_enabled else 0.
                step_time_before_interval = timing_seconds["environment_step_host"] if timing_enabled else 0.
                known_remaining_ticks = horizon - strategic_episode_ticks
                (reward, done, truncated, next_obs, active_this_decision,
                 decision_physics_ticks, decision_active_world_ticks,
                 decision_matches) = _held_action_interval(
                    env, action, ticks_this_decision, obs,
                    known_remaining_ticks=known_remaining_ticks,
                    return_info=False)
                strategic_episode_ticks += decision_physics_ticks
                strategic_episode_boundary = strategic_episode_ticks >= horizon
                if strategic_episode_ticks >= horizon:
                    strategic_episode_ticks = 0
                interval_elapsed = 0.0
                if timing_enabled:
                    interval_elapsed = time.perf_counter() - interval_started
                timing_seconds["active_mask_bookkeeping_host"] += max(
                    0., interval_elapsed -
                    (timing_seconds["environment_step_host"] - step_time_before_interval))
                active_world_physics_ticks += decision_active_world_ticks
                complete_matches += decision_matches
            else:
                next_obs, reward, done, truncated, _info = env.step(action, _return_info=False)
                if timing_enabled:
                    timing_sync_counts["active_mask_count_item_reads"] += 1
                active_world_physics_ticks += num_envs
                complete_matches += truncated.sum()
            reward = reward.to(device=device, dtype=torch.float32).reshape(num_envs)
            if capture_training_playback and t % frame_stride == 0:
                state = torch.cat((env.sim.pose[0].reshape(-1), env.goal[0],
                    env.sim.length[0], env.sim.width[0], env.goal_radius[0:1])).detach().clone()
                active_fuel = env.piece_active[0]
                fuel_pieces = torch.cat((env.piece_pos[0],
                    env.piece_owner[0, :, None].to(env.piece_pos.dtype)), -1)
                update_frames.append({
                    "state": state,
                    "fuel_pieces": fuel_pieces[active_fuel].detach().cpu().tolist(),
                    "hub_active": env.hub_active[0].detach().cpu().tolist(),
                    "hub_centers": env.hub_centers.detach().cpu().tolist(),
                    "match_remaining": float(env.match_remaining[0].item()),
                    "fuel_score_count": env.fuel_score_count[0].detach().cpu().tolist(),
                })
            next_obs = next_obs.to(device=device, dtype=torch.float32)
            reward = reward.to(device=device, dtype=torch.float32).reshape(num_envs)
            done = done.to(device=device, dtype=torch.bool).reshape(num_envs)
            truncated = truncated.to(device=device, dtype=torch.bool).reshape(num_envs)
            episode_return_accum += reward
            if architecture == "strategic_adstar":
                # Strategic episodes have no early termination and share the
                # known physics horizon. Defer metric aggregation and clearing
                # until that scalar boundary instead of reducing an all-false
                # per-world end mask on every 4 Hz policy transition.
                if strategic_episode_boundary:
                    completed_return_sum += episode_return_accum.sum()
                    completed_episode_count += num_envs
                    episode_return_accum.zero_()
            else:
                ended_for_return = done | truncated
                completed_return_sum += torch.where(
                    ended_for_return, episode_return_accum,
                    torch.zeros_like(episode_return_accum)).sum()
                completed_episode_count += ended_for_return.sum()
                episode_return_accum = torch.where(ended_for_return,
                    torch.zeros_like(episode_return_accum), episode_return_accum)
            b_actions[t], b_logprobs[t], b_values[t] = action, logprob, value
            b_rewards[t], b_dones[t], b_truncated[t] = reward, done, truncated
            ended = done | truncated
            # Bootstrap from the terminal observation, then reset ended rows
            # before sampling the next strategic action.
            transition_next_obs = next_obs.to(device=device, dtype=torch.float32)
            # Terminal observations cannot reuse the next policy decision's
            # value because ended worlds reset before that decision. Strategic
            # episodes end together at the known horizon; the last rollout
            # transition also needs an explicit bootstrap value.
            if (architecture != "strategic_adstar" or strategic_episode_boundary or
                    t == rollout_steps - 1):
                with torch.no_grad():
                    b_next_values[t] = model(transition_next_obs)[2]
            reset_done = getattr(env, "reset_done", None)
            if reset_done is None:
                raise RuntimeError("TensorDefenseEnv must provide reset_done(mask) for per-world episode resets")
            if architecture == "strategic_adstar":
                # Strategic episodes have no early termination: every world
                # truncates at the shared physics horizon. Use the scalar
                # episode clock already maintained above instead of reading
                # `ended.any()` back from the accelerator on every decision.
                ended_any = strategic_episode_boundary
            else:
                ended_any = bool(ended.any())
                if timing_enabled:
                    timing_sync_counts["episode_end_any_item_reads"] += 1
            next_obs = reset_done(ended) if ended_any else transition_next_obs
            obs = next_obs.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            advantages = _gae_advantages(
                b_rewards, b_dones, b_truncated, b_values, b_next_values,
                gamma, gae_lambda)
            returns = advantages + b_values

        flat_obs = b_obs.reshape(-1, obs_dim)
        flat_actions = b_actions.reshape((-1, action_dim) if action_kind == "continuous" else (-1,))
        flat_action_masks=(b_action_masks.reshape(-1,action_dim) if b_action_masks is not None else None)
        flat_logprobs = b_logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        batch_size = flat_obs.shape[0]
        ppo_update_started = time.perf_counter() if timing_enabled else 0.
        ppo_gpu_start = torch.cuda.Event(enable_timing=True) if timing_enabled and device.type == "cuda" else None
        ppo_gpu_end = torch.cuda.Event(enable_timing=True) if timing_enabled and device.type == "cuda" else None
        if ppo_gpu_start is not None:
            ppo_gpu_start.record()
        for _epoch in range(epochs):
            indices = torch.randperm(batch_size, device=device)
            for idx in indices.split(minibatch_size):
                action_mask=flat_action_masks[idx] if flat_action_masks is not None else None
                new_logprob, entropy, new_value, _mean = score_model(
                    flat_obs[idx], flat_actions[idx], action_mask)
                logratio = new_logprob - flat_logprobs[idx]
                ratio = logratio.exp()
                mb_adv = flat_advantages[idx]
                if distributed:
                    # Match one-GPU minibatch normalization over the union of
                    # the ranks' equally sized local minibatches.
                    stats = torch.stack((mb_adv.sum(), mb_adv.square().sum(),
                        mb_adv.new_tensor(float(mb_adv.numel()))))
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    adv_mean = stats[0] / stats[2].clamp_min(1.)
                    adv_var = (stats[1] / stats[2].clamp_min(1.) - adv_mean.square()).clamp_min(0.)
                    mb_adv = (mb_adv - adv_mean) / (adv_var.sqrt() + 1e-8)
                else:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
                pg = torch.maximum(-mb_adv * ratio,
                                   -mb_adv * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)).mean()
                value_loss = 0.5 * (new_value - flat_returns[idx]).square().mean()
                l2_norm = sum(parameter.square().sum() for parameter in model.parameters()
                              if parameter.ndim > 1)
                entropy_accumulator += entropy.mean().detach()
                entropy_batch_count += 1
                update_entropy_coef = entropy_coef * max(0., 1. - update_index / max(1, updates - 1))
                loss = pg + value_coef * value_loss - update_entropy_coef * entropy.mean() + l2_coef * l2_norm
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if timing_enabled:
            timing_seconds["ppo_update_host"] += time.perf_counter() - ppo_update_started
            if ppo_gpu_end is not None:
                ppo_gpu_end.record()
                update_gpu_events.append((ppo_gpu_start, ppo_gpu_end))

        # Keep the latest policy usable for the dashboard and for recovery if
        # a long run is interrupted. Status and playback are atomically replaced.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            if timing_enabled:
                timing_sync_counts["update_device_syncs"] += 1
        entropy_mean = float((entropy_accumulator / max(1, entropy_batch_count)).item())
        if timing_enabled:
            timing_sync_counts["entropy_item_reads"] += 1
        if timing_enabled and update_gpu_events:
            timing_seconds["ppo_update_device_event"] = sum(
                start_event.elapsed_time(end_event) / 1000.
                for start_event, end_event in update_gpu_events)
        write_checkpoint({"model_state_dict": model.state_dict(), "obs_dim": obs_dim,
                    "action_dim": action_dim, "action_kind": action_kind,
                    "architecture": architecture, "task": task,
                    "observation_normalization": OBS_NORMALIZATION,
                    "drivetrain_config": env.drivetrain_config})
        if capture_training_playback:
            exported_path = []
            playback_frames.extend({"robots": [row[:3], row[3:6]],
                "goal": row[6:8], "sizes": row[8:12], "goal_radius":row[12],
                "adstar_path": exported_path, "fuel_pieces": frame["fuel_pieces"],
                "hub_active": frame["hub_active"], "hub_centers": frame["hub_centers"],
                "match_remaining": frame["match_remaining"],
                "fuel_score_count": frame["fuel_score_count"]}
                for frame in update_frames
                for row in [frame["state"].cpu().tolist()])
            playback_frames = playback_frames[-480:]
            from .field import (ALLIANCE_ZONE_DEPTH, BUMP_ACCELERATION_SCALE,
                                BUMP_SPEED_SCALE, bump_boxes, static_collision_boxes)
            write_json(playback_path, {"task": task, "dt": env.dt, "field": {
                "length": env.sim.field_length, "width": env.sim.field_width,
                "alliance_zone_depth": ALLIANCE_ZONE_DEPTH,
                "elements": [box.as_dict() for box in env.field_boxes],
                "colliders": [box.as_dict() for box in static_collision_boxes(env.field_boxes)],
                "bump_regions": [box.as_dict() for box in bump_boxes(env.field_boxes)],
                "trench_paths": [box.as_dict() for box in env.field_boxes
                                 if "_trench_" in box.name and "_support_" not in box.name],
                "trench_supports": [box.as_dict() for box in env.field_boxes
                                    if "_trench_support_" in box.name],
                "bump_speed_scale": BUMP_SPEED_SCALE,
                "bump_acceleration_scale": BUMP_ACCELERATION_SCALE},
                "frames": playback_frames})
        completed = min((update_index + 1) * rollout_steps * num_envs, timesteps)
        local_elapsed = time.perf_counter() - start
        if distributed:
            # One compact reduction keeps live status metrics global while
            # preserving local environment/rollout ownership.
            counters = torch.cat((
                complete_matches.to(torch.float64).reshape(1),
                torch.tensor([active_world_physics_ticks], device=device, dtype=torch.float64),
                completed_return_sum.to(torch.float64).reshape(1),
                completed_episode_count.to(torch.float64).reshape(1),
                action_counts.to(torch.float64),
                entropy_accumulator.to(torch.float64).reshape(1),
                torch.tensor([entropy_batch_count], device=device, dtype=torch.float64)))
            dist.all_reduce(counters, op=dist.ReduceOp.SUM)
            elapsed_values = torch.tensor([local_elapsed], device=device, dtype=torch.float64)
            dist.all_reduce(elapsed_values, op=dist.ReduceOp.MAX)
            global_complete_matches = int(counters[0].item())
            global_active_world_ticks = int(counters[1].item())
            global_return_sum = counters[2]
            global_episode_count = counters[3]
            action_offset = 4
            global_action_counts = counters[action_offset:action_offset + action_dim]
            entropy_offset = action_offset + action_dim
            entropy_total = counters[entropy_offset]
            entropy_batches = counters[entropy_offset + 1]
            entropy_mean = float((entropy_total / entropy_batches.clamp_min(1.)).item())
            elapsed = float(elapsed_values.item())
        else:
            global_complete_matches = int(complete_matches.item())
            global_active_world_ticks = active_world_physics_ticks
            global_return_sum = completed_return_sum
            global_episode_count = completed_episode_count
            global_action_counts = action_counts
            entropy_mean = float((entropy_accumulator / max(1, entropy_batch_count)).item())
            elapsed = local_elapsed
        global_completed = completed * world_size
        global_exposure = {key: value * world_size for key, value in opponent_exposure.items()}
        if distributed and timing_enabled:
            timing_values = torch.tensor(list(timing_seconds.values()), device=device,
                                         dtype=torch.float64)
            dist.all_reduce(timing_values, op=dist.ReduceOp.MAX)
            timing_seconds = dict(zip(timing_seconds, timing_values.detach().cpu().tolist()))
            sync_values = torch.tensor(list(timing_sync_counts.values()), device=device,
                                       dtype=torch.int64)
            dist.all_reduce(sync_values, op=dist.ReduceOp.SUM)
            timing_sync_counts = dict(zip(timing_sync_counts,
                                          sync_values.detach().cpu().tolist()))
        write_json(status_path, {**status_context, "status": "running", "task": task, "opponent": opponent,
            "stationary_opponent_fraction": STATIC_OPPONENT_FRACTION,
            "requested_timesteps": requested_timesteps_total,
            "completed_timesteps": completed_timesteps_base + global_completed,
            "num_envs": global_num_envs, "local_num_envs": num_envs,
            "world_size": world_size, "device": str(device),
            "observation_normalization": OBS_NORMALIZATION,
            "l2_coefficient": l2_coef,
            "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "updates": update_index + 1, "total_updates": updates,
                "elapsed_seconds": elapsed, "transitions_per_second": global_completed / max(elapsed, 1e-12),
                "strategic_decision_rate_hz": strategic_rate_hz if architecture == "strategic_adstar" else 50.,
                "strategic_decisions_per_episode": math.ceil(horizon * strategic_rate_hz / 50.)
                    if architecture == "strategic_adstar" else horizon,
                "strategic_transitions_per_second": global_completed / max(elapsed, 1e-12),
                "physics_world_ticks_per_second": global_active_world_ticks / max(elapsed, 1e-12),
                "simulated_seconds_trained": global_active_world_ticks * env.dt,
                "complete_matches_observed": global_complete_matches,
                "mean_return": (float((global_return_sum / global_episode_count.clamp_min(1)).item())
                    if int(global_episode_count.item()) else None),
                "completed_episodes_observed": int(global_episode_count.item()),
                "opponent_exposure_counts": global_exposure,
                "entropy": entropy_mean,
                "action_distribution": (global_action_counts / global_action_counts.sum().clamp_min(1)).detach().cpu().tolist(),
                "per_action_selection_rate": (global_action_counts / global_action_counts.sum().clamp_min(1)).detach().cpu().tolist(),
                "timing_profile_seconds": timing_seconds if timing_enabled else None,
                "timing_sync_sensitive_counts": timing_sync_counts if timing_enabled else None,
            "checkpoint": str(checkpoint), "started_at": started_at,
            "drivetrain_config": env.drivetrain_config})

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    local_elapsed = time.perf_counter() - start
    if distributed:
        elapsed_value = torch.tensor([local_elapsed], device=device, dtype=torch.float64)
        dist.all_reduce(elapsed_value, op=dist.ReduceOp.MAX)
        elapsed = float(elapsed_value.item())
    else:
        elapsed = local_elapsed
    global_completed_total = updates * rollout_steps * num_envs * world_size
    write_checkpoint({"model_state_dict": model.state_dict(), "obs_dim": obs_dim,
                "action_dim": action_dim, "action_kind": action_kind,
                "architecture": architecture, "task": task,
                "observation_normalization": OBS_NORMALIZATION,
                "drivetrain_config": env.drivetrain_config})
    metadata = {"task": task, "architecture": architecture, "action_kind": action_kind,
                "observation_dim": obs_dim, "action_dim": action_dim,
                "opponent": opponent, "seed": seed,
                "stationary_opponent_fraction": STATIC_OPPONENT_FRACTION,
                "timesteps": global_completed_total, "requested_timesteps": global_timesteps,
                "num_envs": global_num_envs, "local_num_envs": num_envs,
                "world_size": world_size, "device": str(device), "checkpoint": str(checkpoint),
                "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "accelerator_backend": "ROCm" if torch.version.hip else ("CUDA" if torch.version.cuda else "CPU"),
                "domain_randomization": True,
                "observation_normalization": OBS_NORMALIZATION,
                "l2_coefficient": l2_coef,
                "elapsed_seconds": elapsed, "transitions_per_second": global_completed_total / max(elapsed, 1e-12),
                "completed_timesteps": global_completed_total,
                "strategic_decision_rate_hz": strategic_rate_hz if architecture == "strategic_adstar" else 50.,
                "strategic_decisions_per_episode": math.ceil(horizon * strategic_rate_hz / 50.)
                    if architecture == "strategic_adstar" else horizon,
                "strategic_transitions_per_second": global_completed_total / max(elapsed, 1e-12),
                "physics_world_ticks_per_second": global_active_world_ticks / max(elapsed, 1e-12),
                "simulated_seconds_trained": global_active_world_ticks * env.dt,
                "complete_matches_observed": global_complete_matches,
                "mean_return": (float((global_return_sum / global_episode_count.clamp_min(1)).item())
                    if int(global_episode_count.item()) else None),
                "opponent_exposure_counts": global_exposure,
                "entropy": entropy_mean,
                "action_distribution": (global_action_counts / global_action_counts.sum().clamp_min(1)).detach().cpu().tolist(),
                "per_action_selection_rate": (global_action_counts / global_action_counts.sum().clamp_min(1)).detach().cpu().tolist(),
                "timing_profile_seconds": timing_seconds if timing_enabled else None,
                "timing_sync_sensitive_counts": timing_sync_counts if timing_enabled else None,
                "started_at": started_at,
                "completed_episodes_observed": int(global_episode_count.item())}
    metadata["drivetrain_config"] = env.drivetrain_config
    if rank == 0:
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    write_json(status_path, {**status_context, **metadata, "status": "completed",
                             "requested_timesteps": requested_timesteps_total,
                             "completed_timesteps": completed_timesteps_base + global_completed_total,
                             "total_updates": updates, "updates": updates})
    return metadata
