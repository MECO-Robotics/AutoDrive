"""3v3 generation training implementation.

The legacy entry point in :mod:`tensor_training` binds this code to its own
globals so existing patches of PPO helpers and Torch symbols keep working.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path


def _train_3v3_generation(task: str, generation: int, output: str | Path, *,
                         seed: int = 7, num_envs: int = 1024,
                         device: str = "cuda", initial_checkpoint: str | Path | None = None,
                         rollout_steps: int = 128, epochs: int = 1,
                         minibatch_size: int = 8192, learning_rate: float = 3e-4,
                         gamma: float = .993, gae_lambda: float = .95,
                         clip_coef: float = .2, value_coef: float = .5,
                         entropy_coef: float = .01, max_grad_norm: float = .5,
                         l2_coef: float = 1e-5, horizon: int = 8000,
                         strategic_rate_hz: float = 4.,
                         _runtime: dict[str, Any]) -> dict[str, Any]:
    """Train one 3v3 PPO generation, sharing the existing 128x128 policy.

    The three defenders use the policy. The opposing offense follows the
    deterministic REBUILT strategy.
    Defense checkpoints can continue training with all trained input columns;
    older non-3v3 defense weights have unused teammate columns zeroed (58:78).
    """
    # These are aliases to the legacy module's live bindings, preserving the
    # established monkeypatch seam without making this module import it.
    ActorCritic = _runtime["ActorCritic"]
    OBS_NORMALIZATION = _runtime["OBS_NORMALIZATION"]
    Path = _runtime["Path"]
    STRATEGIC_ACTION_DIM = _runtime["STRATEGIC_ACTION_DIM"]
    _DDPScore = _runtime["_DDPScore"]
    _PPO_CONTROL_GROUP = _runtime["_PPO_CONTROL_GROUP"]
    _distributed_state = _runtime["_distributed_state"]
    _gae_advantages = _runtime["_gae_advantages"]
    _reset_obs = _runtime["_reset_obs"]
    dist = _runtime["dist"]
    json = _runtime["json"]
    math = _runtime["math"]
    nn = _runtime["nn"]
    time = _runtime["time"]
    torch = _runtime["torch"]

    if task != "defense":
        raise ValueError("only defense policies are trainable; offense is deterministic")
    if min(generation, num_envs, rollout_steps, epochs, minibatch_size, horizon) < 1:
        raise ValueError("generation, environments, rollout, PPO, and horizon must be positive")
    if not 2. <= strategic_rate_hz <= 5.:
        raise ValueError("strategic_rate_hz must be between 2 and 5")
    selected_device, rank, world_size, _created = _distributed_state(device)
    if num_envs % world_size:
        raise ValueError("global num_envs must divide evenly across DDP ranks")
    if minibatch_size % world_size:
        raise ValueError("minibatch_size must divide evenly across DDP ranks")
    local_envs = num_envs // world_size
    rank_seed = seed + rank * 1_000_003
    torch.manual_seed(rank_seed)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(rank_seed)
    nn_team = 1
    modes = ("deterministic", "deterministic", "deterministic", "nn", "nn", "nn")
    from .tensor_3v3 import TensorThreeVsThreeEnv
    env = TensorThreeVsThreeEnv(num_envs=local_envs, device=selected_device,
        seed=rank_seed, control_modes=modes,
        robot_roles=("offense", "offense", "offense", "defense", "defense", "defense"),
        horizon=horizon, dt=.02, max_fuel_capacity=60,
        max_scoring_bps=25.,
        randomize=True, fused_sensor_rng=True)
    # Candidate ranking runs at physics rate for all six robots. Reuse the
    # parity-checked fused distance/mask kernel when HIP supports it; retain
    # ATen top-k so candidate ordering remains backend-defined as before.
    from . import tensor_fuel_candidate_rank as _fuel_candidate_rank
    candidate_rank_accelerated=_fuel_candidate_rank.integrated_available(selected_device)
    if candidate_rank_accelerated:
        def _three_v_three_candidates():
            points=env.track_pos.reshape(-1,env.fuel_count,2)
            robot_xy=env.sim.pose[:,:,:2].reshape(-1,2)
            free=env.track_mask.reshape(-1,env.fuel_count)
            indices,valid,nearest=_fuel_candidate_rank.candidates(
                points,robot_xy,free)
            return (indices.reshape(local_envs,6,4),
                    valid.reshape(local_envs,6,4),
                    nearest.reshape(local_envs,6,4))
        env._candidates=_three_v_three_candidates
    model = ActorCritic(137, STRATEGIC_ACTION_DIM, "categorical").to(selected_device)
    transfer_kind = "fresh"
    if initial_checkpoint is not None and Path(initial_checkpoint).is_file():
        payload = torch.load(initial_checkpoint, map_location=selected_device,
                             weights_only=True)
        if payload.get("task") != "defense":
            raise ValueError("initial checkpoint must be tagged as a defense policy")
        if int(payload.get("obs_dim", -1)) == 137 and int(payload.get("action_dim", -1)) == 8:
            model.load_state_dict(payload["model_state_dict"])
            if payload.get("architecture") != "strategic_3v3":
                with torch.no_grad():
                    model.trunk[0].weight[:, 58:78].zero_()
                transfer_kind = "137x8 weights; zeroed formerly unused teammate columns"
            else:
                transfer_kind = "continued 3v3 checkpoint"
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    score_model = _DDPScore(model)
    if world_size > 1:
        score_model = nn.parallel.DistributedDataParallel(
            score_model, device_ids=[selected_device.index],
            output_device=selected_device.index, broadcast_buffers=False,
            find_unused_parameters=False)
    output = Path(output)
    status_path = output / "status.json"
    checkpoint = output / "policy.pt"
    opponent_robot_ids = torch.arange(0, 3,
                                      device=selected_device)
    deterministic_opponent = "deterministic_offense"
    opponent_names = [deterministic_opponent]
    opponent_exposure_counts = {name: 0 for name in opponent_names}
    generation_started_at=time.time()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps({"status":"running","algorithm":"generational",
            "task":task,"started_at":generation_started_at,
            "architecture":"strategic_3v3","generation":generation,
            "num_envs":num_envs,"local_num_envs":local_envs,
            "robots_per_alliance":3,"policy_robots_per_world":3,
            "observation_dim":137,"action_dim":8,"policy_transfer":transfer_kind,
            "strategic_decision_rate_hz":strategic_rate_hz,
            "deterministic_baseline":deterministic_opponent,
            "fused_fuel_candidate_rank_hip":candidate_rank_accelerated,
            "fused_sensor_rng_hip_requested":env.fused_sensor_rng,
            "fused_sensor_rng_hip":env._fused_sensor_rng_used,
            "fused_opponent_tracks_hip":env._fused_opponent_tracks_used,
            "fused_swerve_pose_hip_requested":env.sim._fused_swerve_pose_hip_enabled,
            "fused_wall_collision_hip_requested":env.sim._fused_wall_collision_hip_enabled,
            "fused_robot_collision_multi_hip_requested":
                env.sim._fused_robot_collision_multi_hip_enabled,
            "training_event_info_capture":False,
            "max_fuel_per_robot":env.fuel_capacity,
            "max_scoring_bps_per_robot":env.max_scoring_bps},indent=2))
    if world_size > 1:
        dist.barrier(group=_PPO_CONTROL_GROUP)

    obs = _reset_obs(env.reset(seed=rank_seed)[0]).to(selected_device)
    robot_ids = torch.arange(0,3,device=selected_device) if nn_team == 0 else torch.arange(3,6,device=selected_device)
    decision_rate_phase = 0.
    episode_physics_ticks = 0
    decisions_per_match = math.ceil(horizon * strategic_rate_hz / 50.)
    decisions_per_generation = math.ceil(max(decisions_per_match, rollout_steps) / rollout_steps) * rollout_steps
    updates = math.ceil(decisions_per_generation / rollout_steps)
    start_time = time.perf_counter()
    complete_matches = torch.zeros((),device=selected_device,dtype=torch.long)
    active_world_ticks = torch.zeros((),device=selected_device,dtype=torch.long)
    returned_sum = torch.zeros((),device=selected_device)
    returned_count = torch.zeros((),device=selected_device)
    episode_returns = torch.zeros((local_envs,3),device=selected_device)
    fuel_acquired = torch.zeros((),device=selected_device)
    fuel_scored = torch.zeros((),device=selected_device)
    action_counts = torch.zeros(8,device=selected_device)
    entropy_sum = torch.zeros((),device=selected_device)
    entropy_batches = 0
    total_decisions = 0
    rollout_wall_seconds_total = 0.
    ppo_update_wall_seconds_total = 0.
    rollout_device_seconds_total = 0.
    ppo_update_device_seconds_total = 0.

    for update in range(updates):
        rollout_len=min(rollout_steps,decisions_per_generation-total_decisions)
        if rollout_len <= 0:
            break
        active_nn_mask=torch.zeros(6,device=selected_device,dtype=torch.bool)
        active_nn_mask[robot_ids]=True
        active_deterministic_mask=torch.zeros_like(active_nn_mask)
        active_deterministic_mask[opponent_robot_ids]=True
        env._nn_mode_mask.copy_(active_nn_mask)
        env._deterministic_mode_mask.copy_(active_deterministic_mask)
        planner_controller_ids=list(range(nn_team*3,nn_team*3+3))
        planner_controller_ids.extend(range(0,3))
        planner_mode_key=tuple(sorted(planner_controller_ids))
        if planner_mode_key != env._planner_controller_mode_key:
            env._planner_controller_modes_dirty=True
            env._planner_controller_mode_key=planner_mode_key
        env._has_deterministic_offense = True
        env.agent_train_mask.copy_(active_nn_mask)
        obs=env.observe()
        env._last_obs=obs
        opponent_exposure_counts[deterministic_opponent] += rollout_len*num_envs
        # Observations dominate rollout storage at large world batches. Keep a
        # half-precision copy on device, then restore float32 for PPO scoring;
        # policy inference and GAE continue to use the original float32 values.
        b_obs=torch.empty((rollout_len,local_envs,3,137),device=selected_device,
                          dtype=torch.float16)
        b_actions=torch.empty((rollout_len,local_envs,3),device=selected_device,dtype=torch.long)
        b_logprobs=torch.empty((rollout_len,local_envs,3),device=selected_device)
        b_action_masks=torch.empty((rollout_len,local_envs,3,8),device=selected_device,dtype=torch.bool)
        b_values=torch.empty((rollout_len,local_envs,3),device=selected_device)
        b_rewards=torch.empty_like(b_values)
        b_dones=torch.empty_like(b_values,dtype=torch.bool)
        b_truncated=torch.empty_like(b_values,dtype=torch.bool)
        b_next_values=torch.empty_like(b_values)
        rollout_wall_started=time.perf_counter()
        if selected_device.type == "cuda":
            rollout_device_start=torch.cuda.Event(enable_timing=True)
            rollout_device_end=torch.cuda.Event(enable_timing=True)
            rollout_device_start.record()
        for t in range(rollout_len):
            selected_obs=obs[:,robot_ids,:]
            flat=selected_obs.reshape(-1,137)
            action_mask=flat[:,-8:].bool()
            with torch.no_grad():
                logits,_,value=model(flat)
                logits=logits.masked_fill(~action_mask,torch.finfo(logits.dtype).min)
                policy_dist=torch.distributions.Categorical(logits=logits)
                sampled=policy_dist.sample()
                logprob=policy_dist.log_prob(sampled)
                entropy=policy_dist.entropy()
            # The current value is also the next-state bootstrap for the
            # previous transition in this rollout. The policy weights stay
            # fixed until the rollout has been collected, so reusing it is
            # exact and avoids a second full-batch value-only forward at every
            # strategic decision.
            if t > 0:
                b_next_values[t-1]=value.reshape(local_envs,3)
            sampled=sampled.reshape(local_envs,3)
            full_action=torch.full((local_envs,6),7,device=selected_device,dtype=torch.long)
            full_action[:,robot_ids]=sampled
            b_obs[t]=selected_obs.to(torch.float16)
            b_actions[t]=sampled
            b_logprobs[t]=logprob.reshape(local_envs,3)
            b_action_masks[t]=action_mask.reshape(local_envs,3,8)
            b_values[t]=value.reshape(local_envs,3)
            action_counts += torch.bincount(sampled.reshape(-1),minlength=8)
            entropy_sum += entropy.mean().detach(); entropy_batches += 1

            decision_rate_phase += 50. / strategic_rate_hz
            ticks=max(1,int(decision_rate_phase))
            decision_rate_phase -= ticks
            remaining_physics_ticks=horizon-episode_physics_ticks
            ticks_this_transition=min(ticks,remaining_physics_ticks)
            # Strategic rewards are the score difference. Accumulate by
            # snapshotting the match counters once per policy interval instead
            # of reducing event tensors and adding a reward batch on every
            # 20 ms physics tick.
            score_count_before=env.fuel_score_count.clone()
            acquisition_count_before=env.fuel_acquisition_count.clone()
            for _ in range(ticks_this_transition):
                _,tick_reward,_,_,_=env.step(full_action,active_mask=None,
                                               capture_observation=False,
                                               capture_info=False)
            score_delta=env.fuel_score_count-score_count_before
            acquired_delta=env.fuel_acquisition_count-acquisition_count_before
            red_delta=score_delta[:,0]
            blue_delta=score_delta[:,1]
            reward_sum=torch.where(env.team_ids[None]==0,
                (red_delta-blue_delta)[:,None],
                (blue_delta-red_delta)[:,None]).expand(-1,6).to(torch.float32)
            active_world_ticks += local_envs*ticks_this_transition
            fuel_acquired += acquired_delta.sum()
            fuel_scored += score_delta.sum()
            episode_physics_ticks += ticks_this_transition
            terminal_obs=env.observe()
            env._last_obs=terminal_obs
            if t == rollout_len-1:
                # The final bootstrap must use the observation before any
                # opponent-mode change or environment reset at the next
                # rollout boundary. Keep this explicit evaluation once per
                # rollout, including the horizon-truncation terminal state.
                next_selected=terminal_obs[:,robot_ids,:].reshape(-1,137)
                with torch.no_grad():
                    b_next_values[t]=model(next_selected)[2].reshape(local_envs,3)
            selected_reward=reward_sum[:,robot_ids]
            b_rewards[t]=selected_reward
            done_rows=torch.zeros(local_envs,device=selected_device,dtype=torch.bool)
            trunc_rows=torch.full_like(done_rows,episode_physics_ticks>=horizon)
            b_dones[t]=done_rows[:,None].expand(-1,3)
            b_truncated[t]=trunc_rows[:,None].expand(-1,3)
            episode_returns += selected_reward
            if episode_physics_ticks>=horizon:
                returned_sum += episode_returns.mean(-1).sum()
                returned_count += local_envs
                complete_matches += local_envs
                episode_returns.zero_()
                reset_mask=torch.ones(local_envs,device=selected_device,dtype=torch.bool)
                obs=env.reset_done(reset_mask)
                episode_physics_ticks=0
            else:
                obs=terminal_obs
        total_decisions += rollout_len
        rollout_wall_seconds_total += time.perf_counter()-rollout_wall_started
        if selected_device.type == "cuda":
            rollout_device_end.record()

        with torch.no_grad():
            advantages=_gae_advantages(b_rewards,b_dones,b_truncated,b_values,
                                       b_next_values,gamma,gae_lambda)
            returns=advantages+b_values
        flat_obs=b_obs.reshape(-1,137)
        flat_actions=b_actions.reshape(-1)
        flat_masks=b_action_masks.reshape(-1,8)
        flat_logp=b_logprobs.reshape(-1)
        flat_adv=advantages.reshape(-1)
        flat_returns=returns.reshape(-1)
        batch_size=flat_obs.shape[0]
        local_minibatch=max(1,min(batch_size,minibatch_size//world_size))
        ppo_wall_started=time.perf_counter()
        if selected_device.type == "cuda":
            ppo_device_start=torch.cuda.Event(enable_timing=True)
            ppo_device_end=torch.cuda.Event(enable_timing=True)
            ppo_device_start.record()
        for epoch in range(epochs):
            order=torch.randperm(batch_size,device=selected_device)
            for idx in order.split(local_minibatch):
                new_logp,entropy,new_value,_=score_model(
                    flat_obs[idx].to(torch.float32),flat_actions[idx],flat_masks[idx])
                ratio=(new_logp-flat_logp[idx]).exp()
                adv=flat_adv[idx]
                adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
                pg=torch.maximum(-adv*ratio,-adv*ratio.clamp(1-clip_coef,1+clip_coef)).mean()
                vloss=.5*(new_value-flat_returns[idx]).square().mean()
                l2=sum(p.square().sum() for p in model.parameters() if p.ndim>1)
                entropy_coef_now=entropy_coef*max(0.,1.-update/max(1,updates-1))
                loss=pg+value_coef*vloss-entropy_coef_now*entropy.mean()+l2_coef*l2
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm); optimizer.step()
        ppo_update_wall_seconds_total += time.perf_counter()-ppo_wall_started
        if selected_device.type == "cuda":
            ppo_device_end.record()

        if world_size > 1:
            counters=torch.stack((complete_matches.to(torch.float64),active_world_ticks.to(torch.float64),
                returned_sum.to(torch.float64),returned_count.to(torch.float64),
                fuel_acquired.to(torch.float64),fuel_scored.to(torch.float64),
                entropy_sum.to(torch.float64),entropy_sum.new_tensor(float(entropy_batches))))
            dist.all_reduce(counters,op=dist.ReduceOp.SUM)
            global_actions=action_counts.to(torch.float64); dist.all_reduce(global_actions,op=dist.ReduceOp.SUM)
        else:
            counters=torch.stack((complete_matches.to(torch.float64),active_world_ticks.to(torch.float64),
                returned_sum.to(torch.float64),returned_count.to(torch.float64),
                fuel_acquired.to(torch.float64),fuel_scored.to(torch.float64),
                entropy_sum.to(torch.float64),entropy_sum.new_tensor(float(entropy_batches))))
            global_actions=action_counts.to(torch.float64)
        # Synchronize before recording throughput. The scalar reads below
        # already synchronize this stream; doing it here makes elapsed include
        # queued simulator and PPO work instead of reporting launch time only.
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
            rollout_device_seconds_total += rollout_device_start.elapsed_time(
                rollout_device_end)/1000.
            ppo_update_device_seconds_total += ppo_device_start.elapsed_time(
                ppo_device_end)/1000.
        elapsed=time.perf_counter()-start_time
        status={"status":"running","algorithm":"generational","task":task,
            "architecture":"strategic_3v3",
            "deterministic_baseline":deterministic_opponent,
            "started_at":generation_started_at,
            "generation":generation,"updates":update+1,"total_updates":updates,
            "completed_timesteps":total_decisions*num_envs*3,
            "requested_timesteps":decisions_per_generation*num_envs*3,
            "num_envs":num_envs,"local_num_envs":local_envs,"world_size":world_size,
            "observation_dim":137,"action_dim":8,"action_kind":"categorical",
            "robots_per_alliance":3,"policy_transfer":transfer_kind,
            "opponents":opponent_names,
            "strategic_decisions_per_episode":decisions_per_match,
            "opponent_exposure_counts":dict(opponent_exposure_counts),
            "opponent_exposure_unit":"world_strategic_steps",
            "opponent_pool":opponent_names,
            "complete_matches_observed":int(counters[0].item()),
            "simulated_seconds_trained":float(counters[1].item())*env.dt,
            "physics_world_ticks_per_second":float(counters[1].item())/max(elapsed,1e-9),
            "strategic_transitions_per_second":total_decisions*num_envs*3/max(elapsed,1e-9),
            "strategic_decision_rate_hz":strategic_rate_hz,
            "max_fuel_per_robot":env.fuel_capacity,
            "max_scoring_bps_per_robot":env.max_scoring_bps,
            "fused_fuel_candidate_rank_hip":candidate_rank_accelerated,
            "fused_sensor_rng_hip":env._fused_sensor_rng_used,
            "fused_opponent_tracks_hip":env._fused_opponent_tracks_used,
            "fused_swerve_pose_hip":(env.sim._fused_swerve_pose_hip_enabled and
                                     not env.sim._fused_swerve_pose_hip_failed),
            "fused_wall_collision_hip":env.sim._fused_wall_collision_hip_used,
            "fused_robot_collision_multi_hip":
                env.sim._fused_robot_collision_multi_hip_used,
            "training_event_info_capture":False,
            "mean_return":(float((counters[2]/counters[3].clamp_min(1)).item())
                           if counters[3].item() else None),
            "fuel_acquired":float(counters[4].item()),"fuel_scored":float(counters[5].item()),
            "entropy":float((counters[6]/counters[7].clamp_min(1)).item()),
            "action_distribution":(global_actions/global_actions.sum().clamp_min(1)).cpu().tolist(),
            "per_action_selection_rate":(global_actions/global_actions.sum().clamp_min(1)).cpu().tolist(),
            "rollout_wall_seconds":rollout_wall_seconds_total,
            "ppo_update_wall_seconds":ppo_update_wall_seconds_total,
            "rollout_device_seconds":rollout_device_seconds_total,
            "ppo_update_device_seconds":ppo_update_device_seconds_total,
            "elapsed_seconds":elapsed,"checkpoint":str(checkpoint),
            "drivetrain_config":env.drivetrain_config}
        if rank == 0:
            status_path.write_text(json.dumps(status,indent=2))
            torch.save({"model_state_dict":model.state_dict(),"obs_dim":137,"action_dim":8,
                "action_kind":"categorical","architecture":"strategic_3v3","task":task,
                "observation_normalization":OBS_NORMALIZATION,
                "observation_spec":"per-robot-3v3-v1",
                "max_fuel_per_robot":env.fuel_capacity,
                "max_scoring_bps_per_robot":env.max_scoring_bps,
                "drivetrain_config":env.drivetrain_config},checkpoint)
    metadata={**status,"status":"completed","completed_timesteps":
              decisions_per_generation*num_envs*3,"complete_matches_observed":int(counters[0].item())}
    if rank == 0:
        status_path.write_text(json.dumps(metadata,indent=2))
        (output/"metadata.json").write_text(json.dumps(metadata,indent=2))
        history_path=output/"generation-history.json"
        try:
            history=json.loads(history_path.read_text())
            generations=history.get("generations",[])
        except (OSError,json.JSONDecodeError,AttributeError):
            generations=[]
        generations=[row for row in generations
                     if int(row.get("generation",-1))!=generation]
        generations.append({key:metadata.get(key) for key in (
            "generation","completed_timesteps","requested_timesteps",
            "complete_matches_observed","simulated_seconds_trained",
            "strategic_transitions_per_second","physics_world_ticks_per_second",
            "strategic_decision_rate_hz","mean_return","fuel_acquired",
            "fuel_scored","entropy","action_distribution",
            "per_action_selection_rate","elapsed_seconds","policy_transfer",
            "opponent_exposure_counts","opponent_pool")})
        generations.sort(key=lambda row:int(row.get("generation",0)))
        history_path.write_text(json.dumps({"task":task,"architecture":"strategic_3v3",
            "generations":generations},indent=2))
    if world_size > 1:
        dist.barrier(group=_PPO_CONTROL_GROUP)
    return metadata


def run_with_legacy_globals(namespace: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run with dependencies from ``tensor_training``'s live globals."""
    return _train_3v3_generation(*args, **kwargs, _runtime=namespace)
