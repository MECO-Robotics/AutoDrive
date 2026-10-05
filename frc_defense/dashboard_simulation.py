"""Zone playback simulation and durable background-job support."""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ZONE_SCENARIO_TICKS = 1300
PPO_CAMPAIGN_UNIT = "frc-defense-ppo-campaign.service"
_SCENARIO_JOB_IDS: set[str] = set()
_SCENARIO_JOB_LOCK = threading.Lock()

def _json_or_default(path: Path, default: dict) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return json.dumps(default).encode()


def _dashboard_simulation_slot(run_dir: Path, *, campaign_unit: str = PPO_CAMPAIGN_UNIT):
    """Serialize dashboard rollouts and yield both GPUs to an active PPO run."""
    lock_path = run_dir / ".dashboard-simulation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = subprocess.run(
            ["systemctl", "--user", "show", "--property=ActiveState", "--value",
             campaign_unit],
            capture_output=True, text=True, check=False, timeout=10,
        )
        paused_training = False
        if state.returncode == 0 and state.stdout.strip() == "active":
            paused = subprocess.run(
                ["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGSTOP",
                 campaign_unit],
                capture_output=True, text=True, check=False, timeout=10,
            )
            if paused.returncode != 0:
                raise RuntimeError("could not reserve GPU time for the requested simulation")
            paused_training = True
        try:
            yield paused_training
        finally:
            if paused_training:
                subprocess.run(
                    ["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGCONT",
                     campaign_unit],
                    capture_output=True, text=True, check=False, timeout=10,
                )


def run_zone_playback(run_dir: Path, run_name: str, start_zone: str,
                      goal_zone: str, seed: int, *, task: str | None = None,
                      control_modes=None, robot_types=None,
                      progress_path: Path | None = None,
                      hopper_capacity: int = 60, scoring_bps: float = 25.,
                      teammate_intent_knowledge: bool = True, _generate_fn=None) -> dict:
    """Use the dashboard runtime when possible, otherwise the project venv."""
    try:
        import torch  # noqa: F401
    except ImportError:
        interpreter=Path.cwd()/".venv"/"bin"/"python"
        if not interpreter.is_file():
            raise RuntimeError("PyTorch is unavailable and the project virtual environment was not found")
        source=("import json,sys; from pathlib import Path; from frc_defense.dashboard import generate_zone_playback; "
                "print(json.dumps(generate_zone_playback(Path(sys.argv[1]),sys.argv[2],sys.argv[3],"
                "sys.argv[4],int(sys.argv[5]),task=sys.argv[6],control_modes=json.loads(sys.argv[7]),"
                "robot_types=json.loads(sys.argv[8]),progress_path=Path(sys.argv[9]) if sys.argv[9] else None,"
                "hopper_capacity=int(sys.argv[10]),scoring_bps=float(sys.argv[11]),"
                "teammate_intent_knowledge=sys.argv[12]=='true'),"
                "separators=(',',':')))")
        result=subprocess.run([str(interpreter),"-c",source,str(run_dir.resolve()),run_name,
            start_zone,goal_zone,str(seed),task or "",json.dumps(control_modes),
            json.dumps(robot_types or ["dumper"]*6),str(progress_path or ""),
            str(hopper_capacity),str(scoring_bps),str(bool(teammate_intent_knowledge)).lower()],
            cwd=Path.cwd(),capture_output=True,text=True,timeout=1800)
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[-1200:] or "zone rollout process failed")
        return json.loads(result.stdout)
    generate_fn = _generate_fn or generate_zone_playback
    return generate_fn(run_dir,run_name,start_zone,goal_zone,seed,
                                  task=task,control_modes=control_modes,
                                  robot_types=robot_types,
                                  progress_path=progress_path,
                                  hopper_capacity=hopper_capacity,
                                  scoring_bps=scoring_bps,
                                  teammate_intent_knowledge=teammate_intent_knowledge)

def _run_zone_playback_job(run_dir: Path, run_name: str, start_zone: str,
                           goal_zone: str, seed: int, task: str,
                           control_modes: list[str], robot_types: list[str],
                           hopper_capacity: int,
                           scoring_bps: float, teammate_intent_knowledge: bool,
                           progress_path: Path,
                           simulation_id: str, *, _slot_fn=None, _run_fn=None, _write_fn=None, _job_ids=None, _job_lock=None) -> None:
    """Run a long scenario outside the HTTP request and publish its result."""
    try:
        slot_fn = _slot_fn or _dashboard_simulation_slot
        run_fn = _run_fn or run_zone_playback
        write_fn = _write_fn or _write_scenario_progress
        with slot_fn(run_dir) as training_paused:
            record=run_fn(run_dir,run_name,start_zone,goal_zone,seed,
                task=task,control_modes=control_modes,robot_types=robot_types,
                progress_path=progress_path,
                hopper_capacity=hopper_capacity,scoring_bps=scoring_bps,
                teammate_intent_knowledge=teammate_intent_knowledge)
        record["compute_scheduling"]="reserved" if training_paused else "shared"
        result_path=progress_path.with_name(progress_path.stem+".result.json")
        temporary=result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record,separators=(",",":")))
        os.replace(temporary,result_path)
        progress=json.loads(_json_or_default(progress_path,{}))
        write_fn(progress_path,progress.get("total_ticks",ZONE_SCENARIO_TICKS),
            progress.get("total_ticks",ZONE_SCENARIO_TICKS),"ready")
    except Exception as exc:
        progress=json.loads(_json_or_default(progress_path,{}))
        write_fn(progress_path,progress.get("tick",0),
            progress.get("total_ticks",ZONE_SCENARIO_TICKS),"error",str(exc))
    finally:
        with (_job_lock or _SCENARIO_JOB_LOCK):
            (_job_ids if _job_ids is not None else _SCENARIO_JOB_IDS).discard(simulation_id)

def _write_scenario_progress(path: Path | None, tick: int, total_ticks: int,
                             status: str = "running", error: str | None = None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload={"status":status,"tick":int(tick),"total_ticks":int(total_ticks),
             "percent":round(100*int(tick)/max(1,int(total_ticks)),1),
             "simulated_seconds":round(int(tick)*.02,2),
             "match_seconds":round(int(total_ticks)*.02,2),"updated_at":time.time()}
    if error:
        payload["error"]=error
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,separators=(",",":")))
    os.replace(temporary,path)

def _ensure_zone_playback_job(run_dir: Path, run_name: str, simulation_id: str,
                              request: dict, *, _job_fn=None, _write_fn=None,
                              _job_ids=None, _job_lock=None) -> dict:
    """Idempotently start or reattach to the durable job identified by its ID."""
    progress_path=run_dir/".scenario-progress"/f"{simulation_id}.json"
    result_path=progress_path.with_name(progress_path.stem+".result.json")
    job_lock = _job_lock or _SCENARIO_JOB_LOCK
    job_ids = _job_ids if _job_ids is not None else _SCENARIO_JOB_IDS
    write_fn = _write_fn or _write_scenario_progress
    job_fn = _job_fn or _run_zone_playback_job
    with job_lock:
        current=json.loads(_json_or_default(progress_path,{}))
        if current.get("status") in ("error","cancelled"):
            return current
        if simulation_id not in job_ids and not result_path.is_file():
            initial_tick=(current.get("tick",0) if current.get("status")=="waiting" else 0)
            write_fn(progress_path,initial_tick,
                current.get("total_ticks",ZONE_SCENARIO_TICKS),"waiting")
            job_ids.add(simulation_id)
            worker=threading.Thread(target=job_fn,
                args=(run_dir,run_name,request["start_zone"],request["goal_zone"],
                    int(request["seed"]),request["task"],request["control_modes"],
                    request.get("robot_types",["dumper"]*6),
                    int(request.get("hopper_capacity",60)),
                    float(request.get("scoring_bps",25.)),
                    bool(request.get("teammate_intent_knowledge",True)),
                    progress_path,simulation_id),daemon=True)
            worker.start()
        return json.loads(_json_or_default(progress_path,{"status":"waiting"}))

def _normalize_robot_control_selections(selections):
    """Resolve each robot's role/controller preset while accepting legacy modes."""
    selections=list(selections or ("offense_nn","offense_nn","offense_nn",
                                   "defense_deterministic","defense_deterministic",
                                   "defense_deterministic"))
    if len(selections)!=6:
        raise ValueError("control_modes must contain six per-robot controller selections")
    fallback_roles=("offense","offense","offense","defense","defense","defense")
    modes=[]
    roles=[]
    for index, selection in enumerate(selections):
        if selection in ("none", "nn", "deterministic"):
            modes.append(selection)
            roles.append(fallback_roles[index])
        elif selection in ("offense_nn", "defense_nn",
                           "offense_deterministic", "defense_deterministic"):
            role, mode = selection.rsplit("_", 1)
            modes.append(mode)
            roles.append(role)
        else:
            raise ValueError("each robot must select none or an offense/defense NN/deterministic controller")
    return modes, roles

def generate_zone_playback(run_dir: Path, run_name: str, start_zone: str,
                           goal_zone: str, seed: int, *, task: str | None = None,
                           control_modes=None, horizon: int = ZONE_SCENARIO_TICKS,
                           robot_types=None,
                           teammate_intent_knowledge: bool = True,
                           capture_stride: int = 16,
                           hopper_capacity: int = 60, scoring_bps: float = 25.,
                           progress_path: Path | None = None, _write_fn=None, _normalize_fn=None) -> dict:
    """Simulate one seeded, six-robot FRC match for dashboard playback."""
    zones = {"red", "center", "blue"}
    if start_zone not in zones or goal_zone not in zones or start_zone == goal_zone:
        raise ValueError("start and goal must be different legal zones")
    if task is None:
        # Compatibility for callers that still identify the scenario role via
        # a saved playback. New dashboard scenarios pass the role explicitly.
        if not re.fullmatch(r"[a-z0-9-]+", run_name):
            raise ValueError("unknown playback run")
        playback = json.loads((run_dir / run_name / "playback.json").read_text())
        task = playback.get("task", "counter_defense")
    sim_task = "defense" if task == "adstar_attacker_defense" else task
    if sim_task not in ("counter_defense", "defense", "3v3"):
        raise ValueError("scenario task must be 3v3")
    import torch
    from .tensor_training import ActorCritic
    from .tensor_3v3 import TensorThreeVsThreeEnv
    write_fn = _write_fn or _write_scenario_progress
    normalize_fn = _normalize_fn or _normalize_robot_control_selections
    control_modes, robot_roles = normalize_fn(control_modes)
    robot_types=list(robot_types or ["dumper"]*6)
    if len(robot_types)!=6 or any(kind not in ("dumper","turret") for kind in robot_types):
        raise ValueError("robot_types must assign dumper or turret to six robots")
    if horizon < 1 or capture_stride < 1:
        raise ValueError("horizon and capture_stride must be positive")
    # Mark the job active before environment setup so the UI reports progress
    # while the one-world CPU rollout initializes and starts.
    write_fn(progress_path, 0, horizon, "running")
    # A one-world dashboard rollout is too small to use the GPU efficiently.
    # Launching its many small HIP kernels and compiling optional extensions
    # costs more than running these 26 simulated seconds on the CPU.
    device = torch.device("cpu")
    env=TensorThreeVsThreeEnv(num_envs=1,device=device,seed=int(seed),
        control_modes=control_modes,robot_roles=robot_roles,
        robot_types=robot_types,
        teammate_intent_knowledge=teammate_intent_knowledge,
        horizon=horizon,dt=.02,randomize=True,
        max_fuel_capacity=hopper_capacity,max_scoring_bps=scoring_bps)
    obs=env.reset(seed=int(seed))[0]
    policies={}
    checkpoint_paths={}
    for role in ("offense", "defense"):
        if not any(mode=="nn" and robot_role==role
                   for mode, robot_role in zip(control_modes, robot_roles)):
            continue
        candidates=(run_dir/role/"policy.pt",Path("checkpoints")/f"rebuilt-gamepiece-{role}"/"policy.pt",
                    Path("checkpoints")/f"rebuilt-{role}"/"policy.pt")
        checkpoint=next((path for path in candidates if path.is_file()),None)
        if checkpoint is None:
            raise RuntimeError(f"NN selected for {role} robot(s), but no compatible {role} checkpoint was found")
        payload=torch.load(checkpoint,map_location=device,weights_only=True)
        if int(payload.get("obs_dim",-1))!=137 or int(payload.get("action_dim",-1))!=8:
            raise RuntimeError(f"{checkpoint} is not a compatible 137-input / 8-action strategic policy")
        model=ActorCritic(137,8,"categorical").to(device)
        model.load_state_dict(payload["model_state_dict"])
        if payload.get("architecture")!="strategic_3v3":
            with torch.no_grad():
                model.trunk[0].weight[:,58:78].zero_()
        model.eval()
        policies[role]=model
        checkpoint_paths[role]=str(checkpoint)

    actions=torch.full((1,6),7,device=device,dtype=torch.long)
    # Keep playback captures on the simulation device. Copying each frame to
    # Python here synchronizes the GPU hundreds of times during a match; one
    # batched copy after the rollout lets simulation stay asynchronous.
    frame_snapshots=[]
    phase=0.
    next_decision=0
    interval=12
    robot_teams=[0,0,0,1,1,1]
    for tick in range(horizon):
        if tick==next_decision:
            obs=env.observe()
            env._last_obs=obs
            for role,model in policies.items():
                robot_ids=torch.tensor([robot for robot in range(6)
                    if control_modes[robot]=="nn" and robot_roles[robot]==role],
                    device=device,dtype=torch.long)
                policy_obs=obs[:,robot_ids,:].reshape(-1,137)
                mask=policy_obs[:,-8:].bool()
                with torch.no_grad():
                    logits=model(policy_obs)[0].masked_fill(~mask,torch.finfo(obs.dtype).min)
                    actions[:,robot_ids]=logits.argmax(-1).reshape(1,-1)
            phase+=50./4.
            interval=max(1,int(phase))
            phase-=interval
            next_decision=tick+interval
        # Dashboard playback does not consume per-tick training/event info.
        # Building its cloned tensors every step launches needless work and
        # holds up this single-world scenario rollout.
        env.step(actions,active_mask=None,capture_observation=False,
                 capture_info=False)
        if tick == 0 or (tick+1) % 100 == 0 or tick+1 == horizon:
            write_fn(progress_path,tick+1,horizon)
        if (tick+1)%capture_stride and tick+1<horizon:
            continue
        path_tensor=env.planner.last_path.reshape(1,6,-1,2)
        path_lengths=env.planner.last_lengths.reshape(1,6)
        snapshot=torch.cat((env.sim.pose[0].reshape(-1),env.sim.velocity[0,:,:2].reshape(-1),
            torch.stack((env.sim.length[0],env.sim.width[0]),-1).reshape(-1),
            path_tensor[0].reshape(-1),path_lengths[0].to(env.sim.pose.dtype),
            env.piece_pos[0].reshape(-1),env.piece_owner[0].to(env.sim.pose.dtype),
            env.piece_active[0].to(env.sim.pose.dtype),env.hub_centers.reshape(-1),
            env.hub_active[0].to(env.sim.pose.dtype),
            env.fuel_score_count[0].to(env.sim.pose.dtype),
            env.match_elapsed[0:1],env.match_remaining[0:1],
            env.last_actions[0].to(env.sim.pose.dtype),
            env.opponent_valid[0].to(env.sim.pose.dtype))).detach()
        frame_snapshots.append(snapshot)
    snapshot_rows=torch.stack(frame_snapshots).cpu().tolist() if frame_snapshots else []
    frames=[]
    for snapshot in snapshot_rows:
        cursor=0
        def take(count):
            nonlocal cursor
            values=snapshot[cursor:cursor+count]
            cursor+=count
            return values
        pose=[take(3) for _ in range(6)]
        effort=[take(2) for _ in range(6)]
        sizes=take(12)
        path_values=take(6*path_tensor.shape[2]*2)
        route_lengths=[int(x) for x in take(6)]
        path_capacity=path_tensor.shape[2]*2
        paths=[]
        for robot,length in enumerate(route_lengths):
            points=path_values[robot*path_capacity:(robot+1)*path_capacity]
            paths.append([points[i:i+2] for i in range(0,length*2,2)])
        piece_positions=take(env.fuel_count*2)
        piece_owners=take(env.fuel_count)
        piece_active=take(env.fuel_count)
        pieces=[[piece_positions[i*2],piece_positions[i*2+1],piece_owners[i]]
                for i in range(env.fuel_count) if piece_active[i]>.5]
        hub_centers=take(4)
        hub_active=[bool(x) for x in take(2)]
        scores=[int(x) for x in take(2)]
        match_elapsed=take(1)[0]
        match_remaining=take(1)[0]
        robot_actions=[int(x) for x in take(6)]
        robot_opponent_visible=[bool(x) for x in take(6)]
        if cursor!=len(snapshot):
            raise RuntimeError("3v3 playback snapshot layout is inconsistent")
        frames.append({
            "robots":pose,
            "robot_teams":robot_teams,
            "robot_control_modes":control_modes,
            "robot_roles":robot_roles,
            "robot_actions":robot_actions,
            "robot_opponent_visible":robot_opponent_visible,
            "robot_effort_vectors":effort,
            "sizes":sizes,
            "adstar_paths":paths,
            "fuel_pieces":pieces,
            "hub_centers":[hub_centers[i:i+2] for i in (0,2)],
            "hub_active":hub_active,
            "fuel_score_count":scores,
            "match_elapsed":match_elapsed,
            "match_remaining":match_remaining,
        })
    field_length=float(env.sim.field_length);field_width=float(env.sim.field_width)
    field={"length":field_length,"width":field_width,"alliance_zone_depth":4.028,
           "elements":[box.as_dict() for box in env.field_boxes]}
    scenario={"id":"3v3","label":f"Randomized REBUILT scenario · {horizon*.02:g} s",
              "start_zone":start_zone,"goal_zone":goal_zone,"frames":frames}
    write_fn(progress_path,horizon,horizon,"completed")
    return {"task":sim_task,"matchup":"six-robot-scenario","architecture":"strategic_3v3",
        "seed":int(seed),"scenario_seed":int(seed),"device":str(device),"field":field,
        "control_modes":control_modes,"robot_roles":robot_roles,
        "robot_types":robot_types,
        "teammate_intent_knowledge":bool(teammate_intent_knowledge),
        "policy_checkpoints":checkpoint_paths,
        "simulation_constraints":{"max_fuel_per_robot":list(env.robot_fuel_capacity_values),
                                  "max_scoring_bps_per_robot":[
                                      round(1./interval,3)
                                      for interval in env.robot_score_interval_values]},
        "robot_teams":robot_teams,"robots_per_alliance":3,"simulated_seconds":horizon*.02,
        "dt":.02,"capture_stride":capture_stride,"scenarios":[scenario],"frames":frames}



def _complete_candidate_adstar_paths(record: dict) -> None:
    """Attach two role-aware AD* paths to every recorded candidate frame."""
    for candidate in record.get("candidates", []):
        _complete_playback_adstar_paths({
            "task": record.get("task"), "field": record.get("field"),
            "frames": candidate.get("frames") or []})

def _complete_playback_adstar_paths(record: dict) -> None:
    """Fill missing per-robot AD* overlays while retaining recorded live paths."""
    from .adstar import ADStarPlanner
    from .field import bump_boxes, rebuilt_field, static_collision_boxes

    field = record.get("field") or {}
    length = float(field.get("length", 16.54))
    width = float(field.get("width", 8.07))
    boxes = rebuilt_field(length, width)
    colliders = static_collision_boxes(boxes)
    bumps = bump_boxes(boxes)
    task = record.get("task", "counter_defense")
    defense_task = task in ("defense", "adstar_attacker_defense")
    attacker_index = 1 if defense_task else 0
    defender_index = 1 - attacker_index
    frame_groups = [scenario.get("frames") or [] for scenario in record.get("scenarios", [])]
    if not frame_groups:
        frame_groups = [record.get("frames") or []]
    for frames in frame_groups:
        if not frames:
            continue
        first = frames[0]
        robots, goal = first.get("robots") or [], first.get("goal")
        if len(robots) < 2 or not goal or len(goal) < 2:
            continue
        sizes = first.get("sizes") or []
        attacker_start = robots[attacker_index][:2]
        reference_routes = [None, None]
        route_targets = [(attacker_index, goal)]
        if not defense_task:
            # The guard reference follows only the attacker's observed position;
            # never derive its overlay from the scoring goal.
            route_targets.append((defender_index, attacker_start))
        for robot_index, target in route_targets:
            robot = robots[robot_index]
            size_offset = robot_index * 2
            robot_length = sizes[size_offset] if len(sizes) > size_offset else .9
            robot_width = sizes[size_offset + 1] if len(sizes) > size_offset + 1 else .9
            planner = ADStarPlanner(length, width, colliders, bumps,
                robot_length=robot_length, robot_width=robot_width,
                robot_heading=robot[2] if len(robot) > 2 else 0.)
            reference_routes[robot_index] = [list(point) for point in planner.plan(robot[:2], target)]
        for frame in frames:
            paths = frame.get("adstar_paths")
            if not isinstance(paths, list) or len(paths) != 2:
                paths = [reference_routes[0], reference_routes[1]]
            else:
                paths = [paths[i] if isinstance(paths[i], list) and len(paths[i]) > 1
                         else reference_routes[i] for i in range(2)]
            if defense_task:
                paths[defender_index] = []
            frame["adstar_paths"] = paths
