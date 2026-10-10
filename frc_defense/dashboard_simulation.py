"""Zone playback simulation and durable background-job support."""
from __future__ import annotations

import hashlib
import json
import gzip
import math
import os
import re
import subprocess
import threading
import time
import datetime
from contextlib import contextmanager
from pathlib import Path

ZONE_SCENARIO_TICKS = 8000  # 160 seconds at the dashboard match's 20 ms timestep
BEHAVIOR_PROBE_TICKS = 1500  # 30 seconds at the dashboard physics timestep
ZONE_PLAYBACK_DT = .02
ZONE_PLAYBACK_TICKS = math.ceil(ZONE_SCENARIO_TICKS * .02 / ZONE_PLAYBACK_DT)
PPO_CAMPAIGN_UNIT = "frc-defense-ppo-campaign.service"
_SCENARIO_JOB_IDS: set[str] = set()
_SCENARIO_JOB_LOCK = threading.Lock()
_SIMULATION_SLOT_CONDITION = threading.Condition()
_ACTIVE_SIMULATION_SLOTS: dict[int, bool] = {}
_SIMULATION_CAMPAIGN_PAUSED = False


def _dashboard_gpu_count() -> int:
    """Return the number of GPUs visible to the project runtime."""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except ImportError:
        pass
    interpreter=Path.cwd()/".venv"/"bin"/"python"
    if not interpreter.is_file():
        return 0
    try:
        result=subprocess.run([str(interpreter),"-c",
            "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"],
            cwd=Path.cwd(),capture_output=True,text=True,timeout=30)
        if result.returncode == 0:
            return max(0,int(result.stdout.strip().splitlines()[-1]))
    except (OSError,ValueError,subprocess.SubprocessError):
        pass
    return 0


def focused_playback_simulation_id(mode: str, robot_type: str) -> str:
    """Key focused replay caches to the current simulator/controller sources."""
    source_root=Path(__file__).parent
    digest=hashlib.sha256()
    suffixes={".py", ".cpp", ".cu", ".hip"}
    for path in sorted(path for path in source_root.rglob("*")
                       if path.is_file() and path.suffix in suffixes):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(path.read_bytes())
    revision=digest.hexdigest()[:12]
    return f"focused-{revision}-{mode}-{robot_type}"


def replay_code_state() -> dict:
    """Describe the checkout used to create a replay and the current checkout."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
            check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
            check=True, capture_output=True, text=True, timeout=5).stdout.splitlines()
        behavior_suffixes={".py", ".cpp", ".cu", ".hip", ".json"}
        changed = sorted({path for line in dirty if len(line) > 3
            for path in [line[3:].split(" -> ")[-1]]
            if Path(path).suffix in behavior_suffixes or path == "pyproject.toml"})
        return {"commit": commit, "dirty_files": changed}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty_files": []}


_PLAYBACK_FRAME_KEYS = frozenset({
    "robots", "behavior_mode", "robot_teams", "robot_control_modes", "robot_roles",
    "robot_actions", "robot_opponent_visible", "robot_detected_opponents",
    "robot_detected_fuel", "robot_fuel_history", "perception_track_timeout_s",
    "camera_layout", "robot_targets", "robot_fuel_targets", "robot_route_goals",
    "robot_collecting", "perception_fov_degrees", "perception_range",
    "robot_cluster_counts", "robot_effort_vectors", "chassis_effort_vector", "sizes",
    "adstar_paths", "fuel_pieces", "hub_centers", "hub_active", "fuel_score_count",
    "fuel_acquisition_count", "match_elapsed", "match_remaining", "predicted_intercept",
    "predicted_intercept_time",
})


def scenario_playback_view(record: dict, *, max_frames: int = 240) -> dict:
    """Return the renderer data without duplicated frames or excess samples."""
    top_level_keys=("task", "matchup", "architecture", "behavior_mode", "seed",
        "scenario_seed", "field", "robot_types", "simulation_constraints", "robot_teams",
        "robot_count", "robots_per_alliance", "simulated_seconds", "simulation_ticks", "dt")
    view={key:record[key] for key in top_level_keys if key in record}
    scenarios=[]
    for scenario in record.get("scenarios",[]) or []:
        frames=scenario.get("frames",[]) or []
        if len(frames)>max_frames:
            stride=math.ceil((len(frames)-1)/max(1,max_frames-1))
            frames=frames[::stride]
            if scenario["frames"][-1] is not frames[-1]:
                frames.append(scenario["frames"][-1])
        compact_frames=[{key:frame[key] for key in _PLAYBACK_FRAME_KEYS if key in frame}
                        for frame in frames]
        scenarios.append({key:scenario[key] for key in ("id","label","start_zone","goal_zone")
                          if key in scenario} | {"frames":compact_frames})
    view["scenarios"]=scenarios
    return view


def replay_commit_age(saved_commit: str | None, current_commit: str | None,
                      *, cwd: Path | None = None) -> int | None:
    """Return commits since saved_commit, or None when ancestry is unknown."""
    if not saved_commit or not current_commit:
        return None
    try:
        result=subprocess.run(["git","rev-list","--count",f"{saved_commit}..{current_commit}"],
            cwd=cwd or Path(__file__).resolve().parent.parent,
            check=True,capture_output=True,text=True,timeout=5)
        return max(0,int(result.stdout.strip()))
    except (OSError,ValueError,subprocess.SubprocessError):
        return None


def prune_scenario_replays(progress_dir: Path, *, now: float | None = None,
                           current_state: dict | None = None) -> list[str]:
    """Remove scenario replay artifacts older than two days or commits."""
    if not progress_dir.is_dir():
        return []
    now=time.time() if now is None else now
    current_state=current_state or replay_code_state()
    pruned=[]
    active_ids=set(_SCENARIO_JOB_IDS)
    root=Path(__file__).resolve().parent.parent
    for metadata_path in progress_dir.glob("*.result.meta.json"):
        try:
            metadata=json.loads(metadata_path.read_text())
            simulation_id=str(metadata.get("simulation_id") or "")
            if not simulation_id.startswith(("scenario-","focused-")) or simulation_id in active_ids:
                continue
            age_days=max(0.,now-metadata_path.stat().st_mtime)/86400.
            code_state=metadata.get("code_state") or {}
            commits=replay_commit_age(code_state.get("commit"),current_state.get("commit"),cwd=root)
            if age_days <= 2 and (commits is None or commits <= 2):
                continue
        except (OSError,json.JSONDecodeError,AttributeError):
            continue
        for path in progress_dir.glob(f"{simulation_id}.*"):
            try:
                path.unlink()
            except OSError:
                pass
        pruned.append(simulation_id)
    # Legacy scenario results have no sidecar provenance. Age is the only
    # reliable signal available for those files.
    for result_path in progress_dir.glob("*.result.json"):
        simulation_id=result_path.name.removesuffix(".result.json")
        if (result_path.with_suffix(".meta.json").exists() or
                not simulation_id.startswith(("scenario-","focused-")) or
                simulation_id in active_ids or simulation_id in pruned):
            continue
        try:
            if now-result_path.stat().st_mtime <= 2*86400:
                continue
        except OSError:
            continue
        for path in progress_dir.glob(f"{simulation_id}.*"):
            try:
                path.unlink()
            except OSError:
                pass
        pruned.append(simulation_id)
    return pruned

def _json_or_default(path: Path, default: dict) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return json.dumps(default).encode()


@contextmanager
def _dashboard_simulation_slot(run_dir: Path, *, dual_gpu: bool = False,
                               campaign_unit: str = PPO_CAMPAIGN_UNIT):
    """Reserve one GPU, or two concurrent slots when dual mode is enabled."""
    global _SIMULATION_CAMPAIGN_PAUSED
    if dual_gpu:
        available_gpus=_dashboard_gpu_count()
        if available_gpus < 1:
            raise RuntimeError("Dual simulation mode requires a visible GPU")
        gpu_count=min(2,available_gpus)
    else:
        gpu_count=1
    use_dual=gpu_count > 1
    with _SIMULATION_SLOT_CONDITION:
        while True:
            compatible=(not _ACTIVE_SIMULATION_SLOTS or
                (use_dual and all(_ACTIVE_SIMULATION_SLOTS.values())))
            if compatible and len(_ACTIVE_SIMULATION_SLOTS) < gpu_count:
                if not _ACTIVE_SIMULATION_SLOTS:
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
                    _SIMULATION_CAMPAIGN_PAUSED=paused_training
                device_index=next(index for index in range(gpu_count)
                                  if index not in _ACTIVE_SIMULATION_SLOTS)
                _ACTIVE_SIMULATION_SLOTS[device_index]=use_dual
                slot={"device_index":device_index,
                      "training_paused":_SIMULATION_CAMPAIGN_PAUSED}
                break
            _SIMULATION_SLOT_CONDITION.wait()
    try:
        yield slot
    finally:
        with _SIMULATION_SLOT_CONDITION:
            _ACTIVE_SIMULATION_SLOTS.pop(slot["device_index"],None)
            if not _ACTIVE_SIMULATION_SLOTS:
                if _SIMULATION_CAMPAIGN_PAUSED:
                    subprocess.run(
                        ["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGCONT",
                         campaign_unit],
                        capture_output=True, text=True, check=False, timeout=10,
                    )
                _SIMULATION_CAMPAIGN_PAUSED=False
            _SIMULATION_SLOT_CONDITION.notify_all()


def run_zone_playback(run_dir: Path, run_name: str, start_zone: str,
                      goal_zone: str, seed: int, *, task: str | None = None,
                      control_modes=None, robot_types=None,
                      progress_path: Path | None = None,
                      hopper_capacity: int = 60, scoring_bps: float = 25.,
                      random_gamepiece_placement: bool = False,
                      teammate_intent_knowledge: bool = True,
                      sweeping_enabled: bool = False, behavior_mode: str = "match",
                      device: str | None = None,
                      _generate_fn=None) -> dict:
    """Run dashboard rollouts on an available GPU runtime."""
    try:
        import torch
        gpu_available=torch.cuda.is_available()
    except ImportError:
        torch=None
        gpu_available=False
    if not gpu_available:
        interpreter=Path.cwd()/".venv"/"bin"/"python"
        if not interpreter.is_file():
            raise RuntimeError("GPU scenario generation requires the project GPU virtual environment")
        gpu_probe=subprocess.run([str(interpreter),"-c",
            "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"],
            cwd=Path.cwd(),capture_output=True,text=True,timeout=30)
        try:
            gpu_count=int(gpu_probe.stdout.strip().splitlines()[-1])
        except (ValueError,IndexError):
            gpu_count=0
        if gpu_probe.returncode or gpu_count < 1:
            raise RuntimeError("GPU scenario generation requires CUDA/HIP-enabled PyTorch in .venv")
        selected_device=device or "cuda:0"
        selected_index=int(selected_device.split(":")[1]) if ":" in selected_device else 0
        if selected_index >= gpu_count:
            raise RuntimeError(f"Requested simulation device {selected_device} is unavailable")
        source=("import json,sys; from pathlib import Path; from frc_defense.dashboard import generate_zone_playback; "
                "print(json.dumps(generate_zone_playback(Path(sys.argv[1]),sys.argv[2],sys.argv[3],"
                "sys.argv[4],int(sys.argv[5]),task=sys.argv[6],control_modes=json.loads(sys.argv[7]),"
                "robot_types=json.loads(sys.argv[8]),progress_path=Path(sys.argv[9]) if sys.argv[9] else None,"
                "hopper_capacity=int(sys.argv[10]),scoring_bps=float(sys.argv[11]),"
                "teammate_intent_knowledge=sys.argv[12]=='true',"
                "sweeping_enabled=sys.argv[13]=='true',behavior_mode=sys.argv[14],"
                "random_gamepiece_placement=sys.argv[15]=='true',device=sys.argv[16]),"
                "separators=(',',':')))")
        result=subprocess.run([str(interpreter),"-c",source,str(run_dir.resolve()),run_name,
            start_zone,goal_zone,str(seed),task or "",json.dumps(control_modes),
            json.dumps(robot_types or ["dumper"]*6),str(progress_path or ""),
            str(hopper_capacity),str(scoring_bps),str(bool(teammate_intent_knowledge)).lower(),
            str(bool(sweeping_enabled)).lower(),behavior_mode,
            str(bool(random_gamepiece_placement)).lower(),selected_device],
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
                                  teammate_intent_knowledge=teammate_intent_knowledge,
                                  sweeping_enabled=sweeping_enabled,
                                  random_gamepiece_placement=random_gamepiece_placement,
                                  behavior_mode=behavior_mode,device=device)

def _run_zone_playback_job(run_dir: Path, run_name: str, start_zone: str,
                           goal_zone: str, seed: int, task: str,
                           control_modes: list[str], robot_types: list[str],
                           hopper_capacity: int,
                           scoring_bps: float, teammate_intent_knowledge: bool,
                           sweeping_enabled: bool,
                           progress_path: Path,
                           simulation_id: str, behavior_mode: str = "match",
                           dual_gpu: bool = False, *,
                           _slot_fn=None, _run_fn=None, _write_fn=None, _job_ids=None, _job_lock=None) -> None:
    """Run a long scenario outside the HTTP request and publish its result."""
    try:
        slot_fn = _slot_fn or _dashboard_simulation_slot
        run_fn = _run_fn or run_zone_playback
        write_fn = _write_fn or _write_scenario_progress
        slot_context=(slot_fn(run_dir,dual_gpu=True) if dual_gpu
                      else slot_fn(run_dir))
        with slot_context as slot:
            if isinstance(slot,dict):
                device_index=int(slot.get("device_index",0))
                training_paused=bool(slot.get("training_paused",False))
            else:
                device_index=0
                training_paused=bool(slot)
            run_kwargs={}
            if device_index:
                run_kwargs["device"]=f"cuda:{device_index}"
            record=run_fn(run_dir,run_name,start_zone,goal_zone,seed,
                task=task,control_modes=control_modes,robot_types=robot_types,
                progress_path=progress_path,
                hopper_capacity=hopper_capacity,scoring_bps=scoring_bps,
                teammate_intent_knowledge=teammate_intent_knowledge,
                sweeping_enabled=sweeping_enabled,behavior_mode=behavior_mode,
                **run_kwargs)
        record["simulation_id"]=simulation_id
        record["code_state"]=replay_code_state()
        record["compute_scheduling"]="reserved" if training_paused else "shared"
        record["simulation_device"]=f"cuda:{device_index}"
        result_path=progress_path.with_name(progress_path.stem+".result.json")
        temporary=result_path.with_suffix(".tmp")
        serialized=json.dumps(record,separators=(",",":" )).encode()
        temporary.write_bytes(serialized)
        os.replace(temporary,result_path)
        metadata={key:record.get(key) for key in (
            "simulation_id", "task", "seed", "behavior_mode", "robot_types", "code_state")}
        scenarios=record.get("scenarios") or []
        metadata["label"]=(scenarios[0].get("label") if scenarios and isinstance(scenarios[0],dict)
                            else simulation_id)
        metadata_path=result_path.with_suffix(".meta.json")
        metadata_temporary=metadata_path.with_suffix(".tmp")
        metadata_temporary.write_text(json.dumps(metadata,separators=(",",":")))
        os.replace(metadata_temporary,metadata_path)
        compressed_path=result_path.with_suffix(result_path.suffix+".gz")
        compressed_temporary=compressed_path.with_suffix(compressed_path.suffix+".tmp")
        compressed_temporary.write_bytes(gzip.compress(serialized,compresslevel=6,mtime=0))
        os.replace(compressed_temporary,compressed_path)
        view_path=result_path.with_suffix(".view.json.gz")
        view_serialized=json.dumps(scenario_playback_view(record),separators=(",",":" )).encode()
        view_temporary=view_path.with_suffix(".tmp")
        view_temporary.write_bytes(gzip.compress(view_serialized,compresslevel=1,mtime=0))
        os.replace(view_temporary,view_path)
        progress=json.loads(_json_or_default(progress_path,{}))
        write_fn(progress_path,progress.get("total_ticks",ZONE_PLAYBACK_TICKS),
            progress.get("total_ticks",ZONE_PLAYBACK_TICKS),"ready")
    except Exception as exc:
        progress=json.loads(_json_or_default(progress_path,{}))
        write_fn(progress_path,progress.get("tick",0),
            progress.get("total_ticks",ZONE_PLAYBACK_TICKS),"error",str(exc))
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
             "simulated_seconds":round(int(tick)*ZONE_PLAYBACK_DT,2),
             "match_seconds":round(int(total_ticks)*ZONE_PLAYBACK_DT,2),
             "updated_at":time.time()}
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
            requested_ticks=int(request.get("total_ticks",ZONE_SCENARIO_TICKS))
            write_fn(progress_path,initial_tick,
                current.get("total_ticks",requested_ticks),"waiting")
            job_ids.add(simulation_id)
            worker=threading.Thread(target=job_fn,
                args=(run_dir,run_name,request["start_zone"],request["goal_zone"],
                    int(request["seed"]),request["task"],request["control_modes"],
                    request.get("robot_types",["dumper"]*6),
                    int(request.get("hopper_capacity",60)),
                    float(request.get("scoring_bps",25.)),
                    bool(request.get("teammate_intent_knowledge",True)),
                    bool(request.get("sweeping_enabled",False)),
                    progress_path,simulation_id,
                    request.get("behavior_mode","match"),
                    bool(request.get("dual_gpu",False))),daemon=True)
            worker.start()
        return json.loads(_json_or_default(progress_path,{"status":"waiting"}))

def _normalize_robot_control_selections(selections):
    """Resolve per-robot control selections; offense always uses deterministic control."""
    selections=list(selections or ("offense_deterministic","offense_deterministic","offense_deterministic",
                                   "defense_deterministic","defense_deterministic",
                                   "defense_deterministic"))
    if len(selections)!=6:
        raise ValueError("control_modes must contain six per-robot controller selections")
    fallback_roles=("offense","offense","offense","defense","defense","defense")
    modes=[]
    roles=[]
    for index, selection in enumerate(selections):
        if selection == "none":
            role, mode = fallback_roles[index], selection
            modes.append(mode)
            roles.append(role)
        elif selection in ("offense_deterministic", "defense_nn",
                           "defense_deterministic"):
            role, mode = selection.rsplit("_", 1)
            modes.append(mode)
            roles.append(role)
        else:
            raise ValueError("each robot must select deterministic offense, NN/deterministic defense, or none")
    return modes, roles

def generate_zone_playback(run_dir: Path, run_name: str, start_zone: str,
                           goal_zone: str, seed: int, *, task: str | None = None,
                           control_modes=None, horizon: int = ZONE_SCENARIO_TICKS,
                           behavior_mode: str = "match",
                           robot_types=None,
                           teammate_intent_knowledge: bool = True,
                           sweeping_enabled: bool = False,
                           random_gamepiece_placement: bool = False,
                           capture_stride: int = 10,
                           physics_dt: float = ZONE_PLAYBACK_DT,
                           perception_interval: int = 2,
                           contact_iterations: int = 1,
                           planner_replan_interval: int | None = None,
                           hopper_capacity: int = 60, scoring_bps: float = 25.,
                           device=None, cuda_graph: bool | None = None,
                           fused_sensor_rng: bool = True,
                           field_sweep_spacing: float = .02,
                           progress_path: Path | None = None, _write_fn=None,
                           _normalize_fn=None) -> dict:
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
    if behavior_mode == "collect":
        behavior_mode="collect_active"
    if behavior_mode not in ("match", "collect_active", "collect_inactive"):
        raise ValueError("behavior_mode must be match or a collect/HUB probe")
    if len(robot_types)!=6 or any(kind not in ("dumper","turret") for kind in robot_types):
        raise ValueError("robot_types must assign dumper or turret to six robots")
    behavior_probe_robot=0
    behavior_probe=None if behavior_mode=="match" else behavior_mode.split("_")[0]
    behavior_probe_hub_active=(behavior_mode.endswith("_active")
                               if behavior_probe else True)
    if behavior_probe:
        selected_type=robot_types[behavior_probe_robot]
        robot_types=[selected_type]+["dumper"]*5
        hopper_capacity=int(hopper_capacity)
        scoring_bps=float(scoring_bps)
        teammate_intent_knowledge=False
        # Isolate one red offense robot and exercise one behavior/state for 30 s.
        control_modes=["deterministic","none","none","none","none","none"]
        robot_roles=["offense"]*6
        horizon=BEHAVIOR_PROBE_TICKS
        capture_stride=10
        sweeping_enabled=False
    if (horizon < 1 or capture_stride < 1 or not math.isfinite(physics_dt) or
            physics_dt <= 0):
        raise ValueError("horizon, capture_stride, and physics_dt must be positive")
    if planner_replan_interval is not None and int(planner_replan_interval) < 1:
        raise ValueError("planner_replan_interval must be positive")
    simulated_seconds = horizon * .02
    simulation_ticks = max(1, int(math.ceil(simulated_seconds / physics_dt)))
    simulation_dt = simulated_seconds / simulation_ticks
    simulation_capture_stride = max(1, int(round(capture_stride * .02 / simulation_dt)))
    # Mark the job active before environment setup so the UI reports progress
    # while the one-world CPU rollout initializes and starts.
    write_fn(progress_path, 0, simulation_ticks, "running")
    # Match playback must use the same 20 ms physics step as training so fuel
    # pickup and scoring contacts are not skipped by coarse integration.
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA scenario generation was requested, but no CUDA/HIP device is available")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    fuel_count=504
    planner_replan_ticks=(max(1,int(math.ceil(.1/simulation_dt)))
                          if behavior_probe=="collect" else
                          max(1,int(math.ceil(.4/simulation_dt))))
    env=TensorThreeVsThreeEnv(num_envs=1,device=device,seed=int(seed),
        control_modes=control_modes,robot_roles=robot_roles,
        robot_types=robot_types,
        teammate_intent_knowledge=teammate_intent_knowledge,
        horizon=simulation_ticks,dt=simulation_dt,randomize=True,
        sweeping_enabled=sweeping_enabled,
        replan_interval=(planner_replan_ticks
                         if planner_replan_interval is None else
                         int(planner_replan_interval)),
        perception_interval=perception_interval,
        contact_iterations=contact_iterations,
        max_fuel_capacity=hopper_capacity,max_scoring_bps=scoring_bps,
        fuel_count=fuel_count,behavior_probe=behavior_probe,
        random_gamepiece_placement=(bool(random_gamepiece_placement) and behavior_probe is not None),
        fused_sensor_rng=fused_sensor_rng,
        field_sweep_spacing=field_sweep_spacing,
        behavior_probe_robot=behavior_probe_robot,
        behavior_probe_hub_active=behavior_probe_hub_active)
    obs=env.reset(seed=int(seed))[0]
    if behavior_probe:
        # Unused fixed tensor slots stay outside the field and cannot obstruct the test.
        env.sim.pose[:,1:,:2]=-100.
    policies={}
    checkpoint_paths={}
    for role in ("defense",):
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
    all_deterministic=all(mode=="deterministic" for mode in control_modes)
    if cuda_graph is None:
        # PyTorch exposes HIP devices through the CUDA API, but planner graph
        # capture is not reliable on ROCm (a failed capture invalidates the
        # stream and aborts scenario creation). Keep graph playback enabled
        # on CUDA and run the same simulation eagerly on HIP.
        cuda_graph=(device.type=="cuda" and torch.version.hip is None and
                    all_deterministic and not policies)
    if cuda_graph and (device.type!="cuda" or not all_deterministic or policies):
        raise ValueError("CUDA graph playback requires a CUDA device and six deterministic controllers")

    actions=torch.full((1,6),7,device=device,dtype=torch.long)
    # Keep playback captures on the simulation device. Copying each frame to
    # Python here synchronizes the GPU hundreds of times during a match; one
    # batched copy after the rollout lets simulation stay asynchronous.
    robot_teams=[0,0,0,1,1,1]
    path_capacity=env.planner.last_path.shape[1]*2
    frame_snapshots=[]
    live_frame_snapshots=[]
    live_stream_path=(progress_path.with_suffix(".stream.jsonl")
                      if progress_path is not None else None)
    field_record={"length":float(env.sim.field_length),"width":float(env.sim.field_width),
                  "alliance_zone_depth":4.028,
                  "elements":[box.as_dict() for box in env.field_boxes]}
    simulation_constraints={"max_fuel_per_robot":list(env.robot_fuel_capacity_values),
        "max_scoring_bps_per_robot":[round(1./interval,3)
            for interval in env.robot_score_interval_values]}
    if live_stream_path is not None:
        live_stream_path.parent.mkdir(parents=True,exist_ok=True)
        live_metadata={"type":"metadata","task":sim_task,"field":field_record,
            "robot_types":robot_types[:1] if behavior_probe else robot_types,
            "robot_roles":robot_roles[:1] if behavior_probe else robot_roles,
            "control_modes":control_modes[:1] if behavior_probe else control_modes,
            "robot_teams":robot_teams[:1] if behavior_probe else robot_teams,
            "simulation_ticks":simulation_ticks,"simulated_seconds":simulated_seconds,
            "behavior_mode":behavior_mode,"simulation_constraints":simulation_constraints,
            "perception_range":env.perception_range,
            "perception_fov_degrees":env.perception_fov_degrees,
            "perception_track_timeout_s":env.perception_track_timeout}
        live_stream_path.write_text(json.dumps(live_metadata,separators=(",",":"))+"\n")
    @torch.inference_mode()
    def simulate_ticks():
        phase=0.
        next_decision=0
        interval=max(1, int(1. / (simulation_dt * 4.)))
        step_graph=None
        graph_audit_outputs=None
        planner_graph=None
        planner_graph_audit_outputs=None

        def build_snapshot(*, include_tracks=True):
            path_tensor=env.planner.last_path.reshape(1,6,-1,2)
            path_lengths=env.planner.last_lengths.reshape(1,6)
            fields = [env.sim.pose[0].reshape(-1),env.sim.velocity[0,:,:2].reshape(-1),
                torch.stack((env.sim.length[0],env.sim.width[0]),-1).reshape(-1),
                path_tensor[0].reshape(-1),path_lengths[0].to(env.sim.pose.dtype),
                env.piece_pos[0].reshape(-1),env.piece_owner[0].to(env.sim.pose.dtype),
                env.piece_active[0].to(env.sim.pose.dtype)]
            if include_tracks:
                fields.extend((env.track_pos[0].permute(1,0,2).reshape(-1),
                    env.track_age[0].transpose(0,1).reshape(-1),
                    env._current_fuel_visibility[0].transpose(0,1).reshape(-1).to(env.sim.pose.dtype)))
            fields.extend((
                env.hub_centers.reshape(-1),
                env.hub_active[0].to(env.sim.pose.dtype),
                env.fuel_score_count[0].to(env.sim.pose.dtype),
                env.fuel_acquisition_count[0].to(env.sim.pose.dtype),
                env.match_elapsed[0:1],env.match_remaining[0:1],
                env.last_actions[0].to(env.sim.pose.dtype),
                env.opponent_valid[0].to(env.sim.pose.dtype),
                env.opponent_pose[0,:,:2].reshape(-1),
                env.opponent_age[0],
                env.track_pos[0].permute(1,0,2).reshape(-1),
                env._current_fuel_visibility[0].transpose(0,1).reshape(-1).to(env.sim.pose.dtype),
                env._last_audit_targets[0].reshape(-1),
                env._last_audit_fuel_targets[0].reshape(-1),
                env.planner.last_goal.reshape(1,6,2)[0].reshape(-1),
                env._target_collecting[0].to(env.sim.pose.dtype),
                env._last_audit_cluster_count[0].to(env.sim.pose.dtype)))
            return torch.cat(fields).detach()

        def write_live_frames(force=False):
            if live_stream_path is None or not live_frame_snapshots:
                return
            if not force and len(live_frame_snapshots)<5:
                return
            rows=torch.stack(live_frame_snapshots).cpu().tolist()
            stream_lines=[]
            for row in rows:
                cursor=0
                def take(count):
                    nonlocal cursor
                    values=row[cursor:cursor+count]
                    cursor+=count
                    return values
                pose=[take(3) for _ in range(6)]
                effort=[take(2) for _ in range(6)]
                sizes=take(12)
                path_values=take(6*path_capacity)
                route_lengths=[int(x) for x in take(6)]
                paths=[]
                for robot,length in enumerate(route_lengths):
                    points=path_values[robot*path_capacity:(robot+1)*path_capacity]
                    paths.append([points[i:i+2] for i in range(0,length*2,2)])
                piece_positions=take(env.fuel_count*2)
                piece_owners=take(env.fuel_count)
                piece_active=take(env.fuel_count)
                pieces=[[piece_positions[i*2],piece_positions[i*2+1],int(piece_owners[i])]
                    for i in range(env.fuel_count) if piece_active[i]>.5]
                hub_centers=take(4)
                hub_active=[bool(x) for x in take(2)]
                scores=[int(x) for x in take(2)]
                acquisitions=[int(x) for x in take(6)]
                match_elapsed=take(1)[0]
                match_remaining=take(1)[0]
                actions=[int(x) for x in take(6)]
                opponent_visible=[bool(x) for x in take(6)]
                opponent_positions=take(12)
                opponent_ages=take(6)
                targets=[take(2) for _ in range(6)]
                fuel_targets=[take(2) for _ in range(6)]
                route_goals=[take(2) for _ in range(6)]
                collecting=[bool(x) for x in take(6)]
                clusters=[int(x) for x in take(6)]
                if cursor!=len(row):
                    raise RuntimeError("live 3v3 frame layout is inconsistent")
                if behavior_probe:
                    pose,effort,sizes,paths=(pose[:1],effort[:1],sizes[:2],paths[:1])
                    actions=actions[:1]
                    opponent_visible=opponent_visible[:1]
                    opponent_positions=opponent_positions[:2]
                    opponent_ages=opponent_ages[:1]
                    targets,fuel_targets,route_goals=(targets[:1],fuel_targets[:1],route_goals[:1])
                    collecting,clusters=collecting[:1],clusters[:1]
                    frame_teams,frame_modes,frame_roles=robot_teams[:1],control_modes[:1],robot_roles[:1]
                    acquisitions=acquisitions[:1]
                else:
                    frame_teams,frame_modes,frame_roles=robot_teams,control_modes,robot_roles
                frame={"robots":pose,"robot_teams":frame_teams,
                    "robot_control_modes":frame_modes,"robot_roles":frame_roles,
                    "robot_actions":actions,"robot_opponent_visible":opponent_visible,
                    "robot_detected_opponents":[
                        [opponent_positions[i*2],opponent_positions[i*2+1],opponent_ages[i]]
                        if opponent_visible[i] else None for i in range(len(opponent_visible))],
                    "robot_targets":targets,"robot_fuel_targets":fuel_targets,
                    "robot_route_goals":route_goals,"robot_collecting":collecting,
                    "robot_cluster_counts":clusters,"robot_effort_vectors":effort,
                    "sizes":sizes,"adstar_paths":paths,"fuel_pieces":pieces,
                    "hub_centers":[hub_centers[i:i+2] for i in (0,2)],
                    "hub_active":hub_active,"fuel_score_count":scores,
                    "fuel_acquisition_count":acquisitions,
                    "match_elapsed":match_elapsed,"match_remaining":match_remaining,
                    "behavior_mode":behavior_mode,"perception_range":env.perception_range,
                    "perception_fov_degrees":env.perception_fov_degrees,
                    "perception_track_timeout_s":env.perception_track_timeout}
                stream_lines.append(json.dumps({"type":"frame","frame":frame},
                                               separators=(",",":")))
            with live_stream_path.open("a") as stream_file:
                stream_file.write("\n".join(stream_lines)+"\n")
                stream_file.flush()
            live_frame_snapshots.clear()

        def capture_frame(tick):
            frame_snapshots.append(build_snapshot())
            if live_stream_path is not None:
                live_frame_snapshots.append(build_snapshot(include_tracks=False))
                write_live_frames()

        if cuda_graph:
            # Resolve lazy HIP extensions before graph capture. The first
            # planner tick is now captured, so extension compilation itself
            # must stay outside the captured region.
            from .tensor_adstar import (_hip_adstar_extension,
                                        _hip_path_reference_extension)
            _hip_adstar_extension()
            _hip_path_reference_extension()
        tick=0
        while tick < simulation_ticks:
            if policies and tick==next_decision:
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
                phase+=1. / (simulation_dt * 4.)
                if phase < 1.:
                    interval=1
                    phase=0.
                else:
                    interval=int(phase)
                    phase-=interval
                next_decision=tick+interval
            # Keep planner calls eager at their existing cadence. The static
            # full-batch step between replans is captured once and replayed on
            # the simulation stream; CUDA graph replays do not execute Python
            # counter updates, so advance the aligned host counter alongside.
            planner_replan_due=(
                (env._planner_tick_scalar+1)%env.replan_interval==0 or
                env._planner_controller_modes_dirty)
            if cuda_graph and planner_replan_due:
                if planner_graph is None:
                    planner_graph=torch.cuda.CUDAGraph()
                    env._capture_planner_retries=True
                    try:
                        with torch.cuda.graph(planner_graph):
                            env.step(actions,active_mask=None,capture_observation=False,
                                     capture_info=False)
                    finally:
                        env._capture_planner_retries=False
                    planner_graph_audit_outputs=(env._last_audit_targets,
                        env._last_audit_fuel_targets,env._last_audit_cluster_count)
                    # Capture records the tick but does not advance the
                    # simulation. Replay it now so this graph branch executes
                    # exactly one step, matching the eager and steady paths.
                    planner_graph.replay()
                    (env._last_audit_targets,env._last_audit_fuel_targets,
                     env._last_audit_cluster_count)=planner_graph_audit_outputs
                else:
                    # Replays do not update this Python cadence counter.
                    env._planner_tick_scalar+=1
                    planner_graph.replay()
                    (env._last_audit_targets,env._last_audit_fuel_targets,
                     env._last_audit_cluster_count)=planner_graph_audit_outputs
            elif cuda_graph:
                if step_graph is None:
                    step_graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(step_graph):
                        env.step(actions,active_mask=None,capture_observation=False,
                                 capture_info=False)
                    graph_audit_outputs=(env._last_audit_targets,
                                         env._last_audit_fuel_targets,
                                         env._last_audit_cluster_count)
                else:
                    # A replay repeats device work but does not rerun Python.
                    env._planner_tick_scalar+=1
                step_graph.replay()
                # Eager replan ticks rebind these Python attributes to fresh
                # tensors. The graph still updates the output buffers captured
                # earlier, so restore those live buffers for the next planner
                # call and for playback snapshots.
                (env._last_audit_targets,env._last_audit_fuel_targets,
                 env._last_audit_cluster_count)=graph_audit_outputs
            else:
                env.step(actions,active_mask=None,capture_observation=False,
                         capture_info=False)
            if tick == 0 or (tick+1) % 100 == 0 or tick+1 == simulation_ticks:
                write_fn(progress_path,tick+1,simulation_ticks)
            if (tick+1)%simulation_capture_stride and tick+1<simulation_ticks:
                tick+=1
                continue
            capture_frame(tick+1)
            tick+=1
        write_live_frames(force=True)
    simulate_ticks()
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
        path_values=take(6*path_capacity)
        route_lengths=[int(x) for x in take(6)]
        paths=[]
        for robot,length in enumerate(route_lengths):
            points=path_values[robot*path_capacity:(robot+1)*path_capacity]
            paths.append([points[i:i+2] for i in range(0,length*2,2)])
        piece_positions=take(env.fuel_count*2)
        piece_owners=take(env.fuel_count)
        piece_active=take(env.fuel_count)
        track_positions=take(env.fuel_count*6*2)
        track_ages=take(env.fuel_count*6)
        current_visibility=take(env.fuel_count*6)
        pieces=[]
        for i in range(env.fuel_count):
            if piece_active[i] <= .5:
                continue
            pieces.append([piece_positions[i*2], piece_positions[i*2+1],
                           piece_owners[i]])
        hub_centers=take(4)
        hub_active=[bool(x) for x in take(2)]
        scores=[int(x) for x in take(2)]
        fuel_acquisition_count=[int(x) for x in take(6)]
        match_elapsed=take(1)[0]
        match_remaining=take(1)[0]
        robot_actions=[int(x) for x in take(6)]
        robot_opponent_visible=[bool(x) for x in take(6)]
        opponent_positions=take(12)
        opponent_ages=take(6)
        detected_fuel_positions=take(env.fuel_count*6*2)
        detected_fuel_visible=take(env.fuel_count*6)
        robot_targets=[take(2) for _ in range(6)]
        robot_fuel_targets=[take(2) for _ in range(6)]
        robot_route_goals=[take(2) for _ in range(6)]
        robot_collecting=[bool(x) for x in take(6)]
        robot_cluster_counts=[int(x) for x in take(6)]
        if cursor!=len(snapshot):
            raise RuntimeError("3v3 playback snapshot layout is inconsistent")
        frames.append({
            "robots":pose,
            "behavior_mode":behavior_mode,
            "robot_teams":robot_teams,
            "robot_control_modes":control_modes,
            "robot_roles":robot_roles,
            "robot_actions":robot_actions,
            "robot_opponent_visible":robot_opponent_visible,
            "robot_detected_opponents":[
                [opponent_positions[i*2],opponent_positions[i*2+1],opponent_ages[i]]
                if robot_opponent_visible[i] else None for i in range(6)],
            "robot_detected_fuel":[[
                [detected_fuel_positions[(piece*6+robot)*2],
                 detected_fuel_positions[(piece*6+robot)*2+1]]
                for piece in range(env.fuel_count)
                if detected_fuel_visible[piece*6+robot] > .5
            ] for robot in range(6)],
            # Track slots are anonymous. Keep stale positions separate from
            # physical FUEL entries so the renderer cannot attach history to a
            # ground-truth ball by matching array indices.
            "robot_fuel_history":[[
                [track_positions[(slot*6+robot)*2],
                 track_positions[(slot*6+robot)*2+1],
                 track_ages[slot*6+robot]]
                for slot in range(env.fuel_count)
                if track_ages[slot*6+robot] < env.perception_track_timeout and
                   detected_fuel_visible[slot*6+robot] <= .5
            ] for robot in range(6)],
            "perception_track_timeout_s":env.perception_track_timeout,
            "camera_layout":{"height_m":0.508,"horizontal_fov_degrees":120,
                             "stereo_baseline_m":0.0635,
                             "mounts":["intake","opposite"]},
            "robot_targets":robot_targets,
            "robot_fuel_targets":robot_fuel_targets,
            "robot_route_goals":robot_route_goals,
            "robot_collecting":robot_collecting,
            "perception_fov_degrees":env.perception_fov_degrees,
            "perception_range":env.perception_range,

            "robot_cluster_counts":robot_cluster_counts,
            "robot_effort_vectors":effort,
            "sizes":sizes,
            "adstar_paths":paths,
            "fuel_pieces":pieces,
            "hub_centers":[hub_centers[i:i+2] for i in (0,2)],
            "hub_active":hub_active,
            "fuel_score_count":scores,
            "fuel_acquisition_count":fuel_acquisition_count,
            "match_elapsed":match_elapsed,
            "match_remaining":match_remaining,
        })
    if behavior_probe:
        for frame in frames:
            for key in ("robots", "robot_teams", "robot_control_modes", "robot_roles",
                        "robot_actions", "robot_opponent_visible", "robot_targets",
                        "robot_fuel_targets", "robot_route_goals", "robot_collecting",
                        "robot_cluster_counts", "robot_effort_vectors", "adstar_paths",
                        "fuel_acquisition_count"):
                frame[key]=frame[key][:1]
            frame["sizes"]=frame["sizes"][:2]
    field=field_record
    scenario_label=(f"{behavior_probe.title()} probe · HUB "
                    f"{'active' if behavior_probe_hub_active else 'inactive'} · "
                    f"{simulated_seconds:g} s" if behavior_probe else
                    f"Randomized REBUILT scenario · {simulated_seconds:g} s")
    scenario={"id":"3v3","label":scenario_label,
              "start_zone":start_zone,"goal_zone":goal_zone,"frames":frames}
    write_fn(progress_path,simulation_ticks,simulation_ticks,"completed")
    return {"task":sim_task,"matchup":("focused-single-robot" if behavior_probe else "six-robot-scenario"),"architecture":"strategic_3v3",
        "behavior_mode":behavior_mode,
        "seed":int(seed),"scenario_seed":int(seed),"device":str(device),"field":field,
        "control_modes":control_modes[:1] if behavior_probe else control_modes,
        "robot_roles":robot_roles[:1] if behavior_probe else robot_roles,
        "robot_types":robot_types[:1] if behavior_probe else robot_types,
        "teammate_intent_knowledge":bool(teammate_intent_knowledge),
        "sweeping_enabled":bool(sweeping_enabled),
        "policy_checkpoints":checkpoint_paths,
        "simulation_constraints":simulation_constraints,
        "robot_teams":robot_teams[:1] if behavior_probe else robot_teams,
        "robot_count":1 if behavior_probe else 6,"robots_per_alliance":1 if behavior_probe else 3,
        "simulated_seconds":simulated_seconds,"simulation_ticks":simulation_ticks,
        "dt":simulation_dt,"capture_stride":simulation_capture_stride,
        "planner_replan_interval_ticks":env.replan_interval,
        "perception_interval":perception_interval,
        "contact_iterations":contact_iterations,
        "scenarios":[scenario],"frames":frames}



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


def prepare_focused_playbacks(run_dir: Path) -> None:
    """Prepare default dumper/turret replays for collect HUB states."""
    progress_dir=run_dir / ".scenario-progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    for behavior in ("collect",):
        for state in ("active", "inactive"):
            mode=f"{behavior}_{state}"
            for robot_type,capacity,scoring_bps in (("dumper",60,25.),("turret",40,15.)):
                simulation_id=focused_playback_simulation_id(mode,robot_type)
                progress_path=progress_dir / f"{simulation_id}.json"
                result_path=progress_dir / f"{simulation_id}.result.json"
                if result_path.is_file():
                    continue
                request={"start_zone":"red","goal_zone":"blue","seed":0,
                    "task":"3v3","control_modes":["offense_deterministic"]+["none"]*5,
                    "behavior_mode":mode,"total_ticks":BEHAVIOR_PROBE_TICKS,
                        "robot_types":[robot_type]+["dumper"]*5,
                        "teammate_intent_knowledge":False,"sweeping_enabled":False,
                        "dual_gpu":True,
                        "hopper_capacity":capacity,"scoring_bps":scoring_bps}
                request_path=progress_path.with_name(progress_path.stem+".request.json")
                if not request_path.is_file():
                    request_path.write_text(json.dumps(request))
                _ensure_zone_playback_job(run_dir,"focused",simulation_id,request)
