"""Small read-only web dashboard for a tensor PPO run."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import re
import time
import secrets
import threading
import math
from contextlib import contextmanager
from .dashboard_api import field_layout_payload


STATIC_DIR = Path(__file__).with_name("dashboard_static")
from .dashboard_simulation import (
    PPO_CAMPAIGN_UNIT,
    BEHAVIOR_PROBE_TICKS,
    ZONE_SCENARIO_TICKS,
    focused_playback_simulation_id,
    replay_code_state,
    prune_scenario_replays,
    ZONE_PLAYBACK_TICKS,
    _SCENARIO_JOB_IDS,
    _SCENARIO_JOB_LOCK,
)


def load_ablation_records(run_dir: Path) -> list[dict]:
    """Read canonical ablation reports for the controller comparison."""
    records: list[dict] = []
    canonical_paths = [run_dir.parent / "metrics" / "ablations.json",
                       run_dir.parent.parent / "metrics" / "ablations.json",
                       run_dir / "metrics" / "ablations.json"]
    if run_dir.is_dir():
        canonical_paths.extend(run_dir.glob("*/ablations.json"))
    canonical_paths = list(dict.fromkeys(canonical_paths))
    for path in canonical_paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        evaluations = payload.get("evaluations", []) if isinstance(payload, dict) else []
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue
            record = dict(evaluation)
            record.setdefault("run_name", "metrics" if path.name == "ablations.json" and path.parent.name == "metrics" else path.parent.name)
            record.setdefault("comparable", False)
            record["source"] = path.name
            records.append(record)

    return records


@contextmanager
def _dashboard_simulation_slot(run_dir: Path, *, dual_gpu: bool = False):
    """Pass dashboard jobs to the single or dual GPU scheduler."""
    from .dashboard_simulation import _dashboard_simulation_slot as simulation_slot
    with simulation_slot(run_dir, dual_gpu=dual_gpu,
                         campaign_unit=PPO_CAMPAIGN_UNIT) as slot:
        yield slot


def create_handler(run_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            content_encoding=None
            playback_decoded_size=None
            parsed=urlparse(self.path)
            params=parse_qs(parsed.query)
            run=params.get("run",[""])[0]
            target_name=run
            target=run_dir/target_name if re.fullmatch(r"[a-z0-9-]+",target_name) else run_dir
            if parsed.path == "/" or parsed.path == "/index.html":
                body = (STATIC_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif parsed.path == "/info":
                body = (STATIC_DIR / "field-guide.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif parsed.path == "/assets/dashboard.css":
                body = (STATIC_DIR / "dashboard.css").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/css; charset=utf-8")
            elif parsed.path == "/assets/dashboard.js":
                body = (STATIC_DIR / "dashboard.js").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
            elif parsed.path in ("/assets/field_rendering.js",
                                 "/assets/training_rendering.js"):
                asset_name = parsed.path.rsplit("/", 1)[-1]
                body = (STATIC_DIR / asset_name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
            elif parsed.path == "/api/training-status":
                runs=[]
                for status_path in run_dir.glob("*/status.json"):
                    try:
                        status=json.loads(status_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    if status.get("algorithm") == "generational":
                        status["run_name"] = status_path.parent.name
                        runs.append(status)
                # A killed trainer can leave a status file marked "running".
                # Treat it as live only while the status file is being refreshed.
                running=[status for status in runs if status.get("status") == "running"
                         and status_path_age(run_dir / status["run_name"]) <
                         (3 * 60 * 60 if status.get("architecture")=="strategic_3v3" else 45 * 60)]
                candidates=running or runs
                try:
                    campaign=json.loads((run_dir / "campaign-status.json").read_text())
                except (OSError, json.JSONDecodeError):
                    campaign={}
                if candidates:
                    selected=max(candidates,key=lambda status: float(status.get("started_at") or 0.))
                    selected_history_path=run_dir/selected.get("run_name","")/"generation-history.json"
                    try:
                        selected_history=json.loads(selected_history_path.read_text())
                        selected["generation_history"]=selected_history.get("generations",[])
                    except (OSError,json.JSONDecodeError,AttributeError):
                        selected["generation_history"]=[]
                    if selected["generation_history"]:
                        selected["latest_generation_metrics"]=selected["generation_history"][-1]
                    if campaign:
                        for task_info in campaign.get("tasks",{}).values():
                            output=task_info.get("output")
                            if output:
                                history_path=Path(output)/"generation-history.json"
                                try:
                                    history_data=json.loads(history_path.read_text())
                                    rows=history_data.get("generations",[])
                                    task_info["latest_generation_metrics"]=rows[-1] if rows else None
                                    task_info["generation_history"]=rows
                                except (OSError,json.JSONDecodeError,AttributeError):
                                    pass
                        selected["campaign"] = campaign
                    game_runs=[{key:status.get(key) for key in (
                        "run_name","task","status","architecture","generation",
                        "total_generations","completed_timesteps","requested_timesteps",
                        "transitions_per_second","device")}
                        for status in candidates if status.get("architecture") in
                        ("strategic_adstar", "strategic_3v3")]
                    for game_run in game_runs:
                        task_info=(campaign.get("tasks",{}).get(game_run.get("task"),{})
                                   if campaign else {})
                        game_run["generations_completed"]=task_info.get("generations_completed")
                        game_run["latest_generation_metrics"]=task_info.get("latest_generation_metrics")
                    body=json.dumps({**selected,"game_runs":game_runs}).encode()
                else:
                    body=json.dumps({"status":"waiting","algorithm":"generational",
                                     "campaign":campaign}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/field-layout":
                body = field_layout_payload()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/playback-runs":
                runs=[]
                for child in run_dir.iterdir():
                    if (child.is_dir() and re.fullmatch(r"[a-z0-9-]+", child.name)
                            and (child / "playback.json").is_file()):
                        try:
                            playback=json.loads((child / "playback.json").read_text())
                            status=json.loads((child / "status.json").read_text())
                        except (OSError, json.JSONDecodeError):
                            continue
                        task=playback.get("task")
                        attacker=defender=None
                        if task == "adstar_attacker_defense":
                            attacker,defender="adstar","nn"
                        elif (task == "defense" and status.get("algorithm") == "generational"
                              and "adstar" in status.get("opponents", [])):
                            # Defense training records deterministic scripted attack.
                            attacker,defender="adstar","nn"
                        if attacker is None:
                            continue
                        runs.append({"id": child.name,
                            "label": child.name.replace("-", " ").title(),
                            "attacker": attacker, "defender": defender,
                            "mtime": (child / "playback.json").stat().st_mtime})
                runs.sort(key=lambda item: item["mtime"], reverse=True)
                for item in runs:
                    item.pop("mtime", None)
                body=json.dumps(runs).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/ablations":
                body=json.dumps(load_ablation_records(run_dir)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/game-evaluations":
                labels={"scripted_defense_adstar":"Scripted defense + AD*",
                        "learned_defense_adstar":"Learned defense + AD*"}
                roots=(run_dir.parent / "metrics", run_dir.parent.parent / "metrics", run_dir / "metrics")
                entries=[{"id":name,"label":label} for name,label in labels.items()
                         if any((root / f"ablations-{name}" / "playback.json").is_file() for root in roots)]
                body=json.dumps(entries).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/game-evaluation-playback":
                name=params.get("name",[""])[0]
                allowed={"scripted_defense_adstar","learned_defense_adstar"}
                if name not in allowed:
                    self.send_error(400,"unknown game evaluation")
                    return
                roots=(run_dir.parent / "metrics", run_dir.parent.parent / "metrics", run_dir / "metrics")
                playback_path=next((root / f"ablations-{name}" / "playback.json" for root in roots
                                    if (root / f"ablations-{name}" / "playback.json").is_file()),None)
                if playback_path is None:
                    self.send_error(404,"game evaluation playback is unavailable")
                    return
                try:
                    record=json.loads(playback_path.read_text())
                    _complete_playback_adstar_paths(record)
                    body=json.dumps(record,separators=(",",":")).encode()
                except (OSError,ValueError,TypeError,KeyError):
                    self.send_error(500,"could not read game evaluation playback")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/status":
                body = _json_or_default(target / "status.json", {"status": "waiting"})
                if target_name == "adstar-attacker-defense":
                    status = json.loads(body)
                    baseline_raw = _json_or_default(target / "direct-baseline-metrics.json", {})
                    baseline = json.loads(baseline_raw)
                    if (baseline.get("checkpoint") == status.get("checkpoint") and
                            (baseline.get("checkpoint_mtime") is None or
                             baseline.get("checkpoint_mtime") == status.get("checkpoint_mtime"))):
                        status["direct_baseline_defender_hold_rate"] = baseline.get("mean_success")
                        status["direct_baseline_mean_attack_time"] = (
                            baseline.get("mean_time_to_goal", 0.) * baseline.get("episodes", 0) /
                            max(1, baseline.get("episodes", 0) - round(baseline.get("mean_success", 0.) * baseline.get("episodes", 0))))
                    body = json.dumps(status).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif parsed.path == "/api/zone-playback":
                start_zone=params.get("start",[""])[0]
                goal_zone=params.get("goal",[""])[0]
                seed_text=params.get("seed",[""])[0]
                scenario_task=params.get("task",["counter_defense"])[0]
                behavior_mode=params.get("behavior_mode",["match"])[0]
                if start_zone not in ("red","center","blue") or goal_zone not in ("red","center","blue") or start_zone==goal_zone:
                    self.send_error(400,"start and goal must be distinct red, center, or blue zones")
                    return
                if scenario_task not in ("counter_defense", "defense", "3v3"):
                    self.send_error(400,"task must be 3v3")
                    return
                if behavior_mode not in ("match", "collect_active", "collect_inactive", "collect"):
                    self.send_error(400,"behavior_mode must be match or a collect/HUB probe")
                    return
                control_modes=[params.get(f"robot{i}",["offense_deterministic" if i < 3 else "defense_deterministic"])[0]
                               for i in range(6)]
                robot_types=[params.get(f"robot_type{i}",["dumper"])[0]
                             for i in range(6)]
                intent_text=params.get("teammate_intent_knowledge",["true"])[0].lower()
                if intent_text not in ("true","false"):
                    self.send_error(400,"teammate_intent_knowledge must be true or false")
                    return
                teammate_intent_knowledge=intent_text=="true"
                sweeping_text=params.get("sweeping_enabled",["false"])[0].lower()
                if sweeping_text not in ("true","false"):
                    self.send_error(400,"sweeping_enabled must be true or false")
                    return
                sweeping_enabled=sweeping_text=="true"
                dual_gpu_text=params.get("dual_gpu",["false"])[0].lower()
                if dual_gpu_text not in ("true","false"):
                    self.send_error(400,"dual_gpu must be true or false")
                    return
                dual_gpu=dual_gpu_text=="true"
                if any(kind not in ("dumper","turret") for kind in robot_types):
                    self.send_error(400,"each robot type must be dumper or turret")
                    return
                try:
                    hopper_capacity=int(params.get("hopper_capacity",["60"])[0])
                    scoring_bps=float(params.get("scoring_bps",["25"])[0])
                    if not 1 <= hopper_capacity <= 504:
                        raise ValueError("hopper_capacity must be between 1 and 504")
                    if not math.isfinite(scoring_bps) or not .1 <= scoring_bps <= 50:
                        raise ValueError("scoring_bps must be between 0.1 and 50")
                except ValueError as exc:
                    self.send_error(400,str(exc))
                    return
                simulation_id = params.get("simulation_id", [""])[0]
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", simulation_id):
                    self.send_error(400,"simulation_id is required")
                    return
                if behavior_mode == "collect":
                    behavior_mode="collect_active"
                if behavior_mode != "match":
                    if not control_modes or control_modes[0] not in ("offense_deterministic",):
                        control_modes[0]="offense_deterministic"
                    control_modes[1:]=["none"]*5
                    robot_types[1:]=["dumper"]*5
                progress_path=run_dir/".scenario-progress"/f"{simulation_id}.json"
                seed=int(seed_text) if seed_text.isdigit() else secrets.randbits(31)
                request={"start_zone":start_zone,"goal_zone":goal_zone,"seed":seed,
                         "task":scenario_task,"control_modes":control_modes,
                         "behavior_mode":behavior_mode,
                         "total_ticks":(BEHAVIOR_PROBE_TICKS if behavior_mode!="match" else ZONE_SCENARIO_TICKS),
                         "robot_types":robot_types,
                         "teammate_intent_knowledge":teammate_intent_knowledge,
                         "sweeping_enabled":sweeping_enabled,
                         "random_gamepiece_placement":(params.get("random_gamepiece_placement",["false"])[0].lower()=="true"),
                         "hopper_capacity":hopper_capacity,"scoring_bps":scoring_bps,
                         "dual_gpu":dual_gpu}
                request_path=progress_path.with_name(progress_path.stem+".request.json")
                if not request_path.is_file():
                    request_path.write_text(json.dumps(request))
                current=_ensure_zone_playback_job(run_dir,target_name,simulation_id,request)
                if current.get("status") in ("error","cancelled"):
                    body=json.dumps({"error":current.get("error","scenario was cancelled")}).encode()
                    self.send_response(409)
                else:
                    body=json.dumps({"simulation_id":simulation_id,
                        "status":current.get("status","waiting")}).encode()
                    self.send_response(202)
                self.send_header("Content-Type","application/json")
            elif parsed.path == "/api/focused-playback":
                mode=params.get("behavior_mode",[""])[0]
                robot_type=params.get("robot_type",[""])[0]
                dual_gpu_text=params.get("dual_gpu",["false"])[0].lower()
                if dual_gpu_text not in ("true","false"):
                    self.send_error(400,"dual_gpu must be true or false")
                    return
                dual_gpu=dual_gpu_text=="true"
                if mode not in ("collect_active","collect_inactive"):
                    self.send_error(400,"unknown focused replay")
                    return
                if robot_type not in ("dumper","turret"):
                    self.send_error(400,"robot_type must be dumper or turret")
                    return
                simulation_id=focused_playback_simulation_id(mode,robot_type)
                path=run_dir/".scenario-progress"/f"{simulation_id}.result.json"
                compressed=path.with_suffix(path.suffix+".gz")
                if not path.is_file():
                    progress_path=run_dir/".scenario-progress"/f"{simulation_id}.json"
                    request={"start_zone":"red","goal_zone":"blue","seed":0,
                        "task":"3v3","control_modes":["offense_deterministic"]+["none"]*5,
                        "behavior_mode":mode,"total_ticks":BEHAVIOR_PROBE_TICKS,
                        "robot_types":[robot_type]+["dumper"]*5,
                        "teammate_intent_knowledge":False,"sweeping_enabled":False,
                        "dual_gpu":dual_gpu,
                        "hopper_capacity":40 if robot_type=="turret" else 60,
                        "scoring_bps":15. if robot_type=="turret" else 25.}
                    request_path=progress_path.with_name(progress_path.stem+".request.json")
                    progress_path.parent.mkdir(parents=True,exist_ok=True)
                    if not request_path.is_file():
                        request_path.write_text(json.dumps(request))
                    current=_ensure_zone_playback_job(run_dir,"focused",simulation_id,request)
                    if current.get("status") in ("error","cancelled"):
                        self.send_error(500,current.get("error","focused replay generation failed"))
                        return
                    body=json.dumps({"simulation_id":simulation_id,
                        "status":current.get("status","waiting")}).encode()
                    self.send_response(202)
                    self.send_header("Content-Type","application/json")
                    self.send_header("Content-Length",str(len(body)))
                    self.send_header("Cache-Control","no-store")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    try:
                        use_gzip="gzip" in self.headers.get("Accept-Encoding","") and compressed.is_file()
                        body=compressed.read_bytes() if use_gzip else path.read_bytes()
                    except OSError:
                        self.send_error(500,"could not read focused replay")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type","application/json")
                    if use_gzip:
                        self.send_header("Content-Encoding","gzip")
                    self.send_header("Content-Length",str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return
            elif parsed.path == "/api/zone-playback-stream":
                simulation_id=params.get("simulation_id",[""])[0]
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", simulation_id):
                    self.send_error(400,"invalid simulation_id")
                    return
                try:
                    offset = max(0, int(params.get("offset", ["0"])[0]))
                except (TypeError,ValueError):
                    self.send_error(400,"offset must be a non-negative integer")
                    return
                progress_path = run_dir / ".scenario-progress" / f"{simulation_id}.json"
                progress = json.loads(_json_or_default(progress_path, {}))
                stream_path = run_dir / ".scenario-progress" / f"{simulation_id}.stream.jsonl"
                metadata = None
                frames = []
                next_offset = offset
                try:
                    if stream_path.is_file():
                        size = stream_path.stat().st_size
                        if offset>size:
                            offset=0
                        with stream_path.open("rb") as stream_file:
                            stream_file.seek(offset)
                            chunk = stream_file.read(1_000_000)
                        newline = chunk.rfind(b"\n")
                        if newline>=0:
                            complete = chunk[:newline + 1]
                            next_offset = offset + len(complete)
                            for line in complete.splitlines():
                                if not line:
                                    continue
                                item = json.loads(line)
                                if item.get("type")=="metadata":
                                    metadata=item
                                elif item.get("type")=="frame":
                                    frame = item.get("frame")
                                    if isinstance(frame,dict):
                                        frames.append(frame)
                except (OSError,ValueError,TypeError,json.JSONDecodeError):
                    self.send_error(500,"could not read live scenario frames")
                    return
                body = json.dumps({"simulation_id": simulation_id,
                    "metadata": metadata, "frames": frames,
                    "next_offset": next_offset,
                    "status": progress.get("status", "waiting")},
                    separators=(",", ":")).encode()
                self.send_response(200)
                self.send_header("Content-Type","application/json")
            elif parsed.path == "/api/zone-playback-result":
                simulation_id=params.get("simulation_id",[""])[0]
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", simulation_id):
                    self.send_error(400,"invalid simulation_id")
                    return
                progress_path=run_dir/".scenario-progress"/f"{simulation_id}.json"
                progress=json.loads(_json_or_default(progress_path,{}))
                if progress.get("status")=="error":
                    body=json.dumps({"error":progress.get("error","scenario failed")}).encode()
                    self.send_response(500)
                    self.send_header("Content-Type","application/json")
                elif progress.get("status")!="ready":
                    body=json.dumps({"error":"scenario playback is not ready"}).encode()
                    self.send_response(409)
                    self.send_header("Content-Type","application/json")
                else:
                    result_path=run_dir/".scenario-progress"/f"{simulation_id}.result.json"
                    try:
                        playback_view=params.get("view",["0"])[0]=="1"
                        view_path=result_path.with_suffix(".view.json.gz")
                        compressed_path=(view_path if playback_view and view_path.is_file()
                            else result_path.with_suffix(result_path.suffix+".gz"))
                        if ("gzip" in self.headers.get("Accept-Encoding","") and
                                compressed_path.is_file()):
                            body=compressed_path.read_bytes()
                            content_encoding="gzip"
                            if playback_view and compressed_path == view_path and len(body) >= 4:
                                playback_decoded_size=int.from_bytes(body[-4:],"little")
                        else:
                            body=result_path.read_bytes()
                    except OSError:
                        body=json.dumps({"error":"scenario result is unavailable"}).encode()
                        self.send_response(500)
                        self.send_header("Content-Type","application/json")
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type","application/json")
            elif parsed.path == "/api/zone-playback-progress":
                simulation_id=params.get("simulation_id",[""])[0]
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", simulation_id):
                    self.send_error(400,"invalid simulation_id")
                    return
                progress_path=run_dir/".scenario-progress"/f"{simulation_id}.json"
                body=_json_or_default(progress_path,{"status":"waiting","tick":0,"total_ticks":ZONE_PLAYBACK_TICKS,"percent":0})
                progress=json.loads(body)
                request_path=progress_path.with_name(progress_path.stem+".request.json")
                if progress.get("status") in ("waiting","running") and request_path.is_file():
                    try:
                        request=json.loads(request_path.read_text())
                        _ensure_zone_playback_job(run_dir,target_name,simulation_id,request)
                    except (OSError,json.JSONDecodeError,KeyError,ValueError):
                        pass
                self.send_response(200)
                self.send_header("Content-Type","application/json")
            elif parsed.path == "/api/zone-playback-active":
                progress_dir=run_dir/".scenario-progress"
                jobs=[]
                for request_path in progress_dir.glob("*.request.json"):
                    simulation_id=request_path.name.removesuffix(".request.json")
                    progress_path=progress_dir/f"{simulation_id}.json"
                    try:
                        request=json.loads(request_path.read_text())
                        progress=json.loads(progress_path.read_text())
                    except (OSError,json.JSONDecodeError):
                        continue
                    if progress.get("status") in ("waiting","running","ready"):
                        jobs.append((request_path.stat().st_mtime,{**request,
                            "simulation_id":simulation_id,"progress":progress}))
                job=max(jobs,key=lambda item:item[0])[1] if jobs else None
                body=json.dumps({"job":job},separators=(",",":")).encode()
                self.send_response(200)
                self.send_header("Content-Type","application/json")
            elif parsed.path == "/api/scenario-replays":
                progress_dir=run_dir/".scenario-progress"
                current=replay_code_state()
                prune_scenario_replays(progress_dir,current_state=current)
                replays=[]
                if progress_dir.is_dir():
                    for metadata_path in progress_dir.glob("*.result.meta.json"):
                        try:
                            record=json.loads(metadata_path.read_text())
                        except (OSError,json.JSONDecodeError):
                            continue
                        if not isinstance(record,dict):
                            continue
                        simulation_id=record.get("simulation_id") or metadata_path.name.removesuffix(".result.meta.json")
                        if not str(simulation_id).startswith(("scenario-", "focused-")):
                            continue
                        saved=record.get("code_state") or {}
                        saved_dirty=sorted(path for path in saved.get("dirty_files",[])
                            if Path(path).suffix in {".py", ".cpp", ".cu", ".hip", ".json"}
                            or path == "pyproject.toml")
                        outdated=(bool(record.get("outdated")) or
                            saved.get("commit") != current.get("commit") or
                            saved_dirty != current.get("dirty_files",[]))
                        view_path=progress_dir/f"{simulation_id}.result.view.json.gz"
                        replays.append({"simulation_id":simulation_id,
                            "label":record.get("label") or simulation_id,
                            "seed":record.get("seed"),"behavior_mode":record.get("behavior_mode","match"),
                            "robot_types":record.get("robot_types",[]),"code_state":saved,
                            "outdated":outdated,
                            "playback_size_bytes":(view_path.stat().st_size if view_path.is_file()
                                else (progress_dir/f"{simulation_id}.result.json.gz").stat().st_size
                                if (progress_dir/f"{simulation_id}.result.json.gz").is_file() else 0),
                            "updated_at":metadata_path.stat().st_mtime})
                    # Legacy replay results can be large; only open a bounded
                    # header window to extract the scalar fields we need.
                    for result_path in progress_dir.glob("*.result.json"):
                        if result_path.with_suffix(".meta.json").exists():
                            continue
                        simulation_id=result_path.name.removesuffix(".result.json")
                        if not simulation_id.startswith(("scenario-", "focused-")):
                            continue
                        try:
                            with result_path.open("rb") as replay_file:
                                header=replay_file.read(16384).decode("utf-8",errors="ignore")
                            seed_match=re.search(r'"seed"\s*:\s*(-?\d+)',header)
                            behavior_match=re.search(r'"behavior_mode"\s*:\s*"([^"]+)"',header)
                            types_match=re.search(r'"robot_types"\s*:\s*(\[[^]]*\])',header)
                            types=json.loads(types_match.group(1)) if types_match else []
                        except (OSError,json.JSONDecodeError):
                            seed_match=behavior_match=None
                            types=[]
                        replays.append({"simulation_id":simulation_id,"label":simulation_id,
                            "seed":int(seed_match.group(1)) if seed_match else None,
                            "behavior_mode":behavior_match.group(1) if behavior_match else "match",
                            "robot_types":types,"code_state":{},"outdated":True,
                            "updated_at":result_path.stat().st_mtime})
                replays.sort(key=lambda item:item["updated_at"],reverse=True)
                body=json.dumps({"current_code_state":current,"replays":replays},separators=(",",":")).encode()
                self.send_response(200)
                self.send_header("Content-Type","application/json")
            elif parsed.path == "/api/playback":
                playback_path = target / "playback.json"
                body = _json_or_default(playback_path, {"frames": []})
                try:
                    stat = playback_path.stat()
                    revision = f'"{stat.st_mtime_ns}-{stat.st_size}"'
                    if self.headers.get("If-None-Match") == revision:
                        self.send_response(304)
                        self.send_header("ETag", revision)
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        return
                except FileNotFoundError:
                    revision = "missing"
                try:
                    record = json.loads(body)
                    _complete_playback_adstar_paths(record)
                    body = json.dumps(record, separators=(",", ":")).encode()
                except (ValueError, TypeError, KeyError):
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Playback-Revision", revision)
                self.send_header("ETag", revision)
            elif parsed.path == "/api/generation":
                generation = params.get("generation", [""])[0]
                if not generation.isdigit() or int(generation) < 1:
                    self.send_error(400, "generation must be a positive integer")
                    return
                generation_path = target / "generation-playback" / f"generation-{int(generation)}.json"
                body = _json_or_default(generation_path, {"generation": int(generation), "candidates": []})
                # Older top-five generations contain poses but predate per-run
                # AD* route recording. Supply a real field-aware route for each
                # candidate so the UI can render planner context for every ghost.
                try:
                    record = json.loads(body)
                    if record.get("candidates"):
                        _complete_candidate_adstar_paths(record)
                        body = json.dumps(record, separators=(",", ":")).encode()
                except (ValueError, TypeError, KeyError):
                    pass
                try:
                    stat = generation_path.stat()
                    revision = f'"{stat.st_mtime_ns}-{stat.st_size}"'
                    if self.headers.get("If-None-Match") == revision:
                        self.send_response(304)
                        self.send_header("ETag", revision)
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        return
                except FileNotFoundError:
                    revision = "missing"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("ETag", revision)
            elif parsed.path == "/api/generation-scenario":
                generation = params.get("generation", [""])[0]
                if not generation.isdigit() or int(generation) < 1:
                    self.send_error(400, "generation must be a positive integer")
                    return
                scenario_path = (target / "scenario-simulations" /
                    f"generation-{int(generation):04d}" / "playback.json")
                body = _json_or_default(scenario_path, {"generation": int(generation),
                    "scenarios": [], "status": "waiting"})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                self.send_error(404)
                return
            self.send_header("Content-Length", str(len(body)))
            if content_encoding:
                self.send_header("Content-Encoding",content_encoding)
                self.send_header("Vary","Accept-Encoding")
            if playback_decoded_size is not None:
                self.send_header("X-Playback-Decoded-Length",str(playback_decoded_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

        def do_DELETE(self):
            parsed=urlparse(self.path)
            params=parse_qs(parsed.query)
            simulation_id=params.get("simulation_id",[""])[0]
            def respond(status,payload):
                body=json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(body)))
                self.send_header("Cache-Control","no-store")
                self.end_headers()
                self.wfile.write(body)

            if (parsed.path != "/api/scenario-replay" or
                    not re.fullmatch(r"(?:scenario|focused)-[a-zA-Z0-9_-]{1,80}",simulation_id)):
                respond(400,{"error":"invalid saved scenario replay"})
                return
            if simulation_id in _SCENARIO_JOB_IDS:
                respond(409,{"error":"this replay is still being generated"})
                return
            progress_dir=run_dir/".scenario-progress"
            paths=list(progress_dir.glob(f"{simulation_id}.*"))
            removed=0
            try:
                for path in paths:
                    if path.is_file():
                        path.unlink()
                        removed+=1
            except OSError:
                respond(500,{"error":"could not remove all files for this replay"})
                return
            if not removed:
                respond(404,{"error":"saved replay was not found"})
                return
            respond(200,{"simulation_id":simulation_id,"deleted_files":removed})
    return Handler















def run_zone_playback(run_dir: Path, run_name: str, start_zone: str,
                      goal_zone: str, seed: int, *, task: str | None = None,
                      control_modes=None, robot_types=None,
                      progress_path: Path | None = None,
                      hopper_capacity: int = 60, scoring_bps: float = 25.,
                      teammate_intent_knowledge: bool = True,
                      sweeping_enabled: bool = False,
                      random_gamepiece_placement: bool = False,
                      behavior_mode: str = "match",
                      device: str | None = None) -> dict:
    """Use the dashboard runtime when possible, otherwise the project venv."""
    from . import dashboard_simulation as simulation
    return simulation.run_zone_playback(
        run_dir, run_name, start_zone, goal_zone, seed, task=task,
        control_modes=control_modes, robot_types=robot_types,
        progress_path=progress_path, hopper_capacity=hopper_capacity,
        scoring_bps=scoring_bps,
        teammate_intent_knowledge=teammate_intent_knowledge,
        sweeping_enabled=sweeping_enabled,
        random_gamepiece_placement=random_gamepiece_placement,
        behavior_mode=behavior_mode,
        device=device,
        _generate_fn=generate_zone_playback)


def _run_zone_playback_job(*args, **kwargs) -> None:
    """Run a playback job with dashboard-local hooks kept patchable."""
    from . import dashboard_simulation as simulation
    return simulation._run_zone_playback_job(
        *args, _slot_fn=_dashboard_simulation_slot,
        _run_fn=run_zone_playback, _write_fn=_write_scenario_progress,
        _job_ids=_SCENARIO_JOB_IDS, _job_lock=_SCENARIO_JOB_LOCK, **kwargs)


def _write_scenario_progress(*args, **kwargs) -> None:
    from .dashboard_simulation import _write_scenario_progress as write_progress
    return write_progress(*args, **kwargs)


def _ensure_zone_playback_job(*args, **kwargs) -> dict:
    """Start/reattach playback jobs using patchable dashboard callbacks."""
    from . import dashboard_simulation as simulation
    return simulation._ensure_zone_playback_job(
        *args, _job_fn=_run_zone_playback_job,
        _write_fn=_write_scenario_progress,
        _job_ids=_SCENARIO_JOB_IDS, _job_lock=_SCENARIO_JOB_LOCK, **kwargs)


def _normalize_robot_control_selections(selections):
    from .dashboard_simulation import _normalize_robot_control_selections as normalize
    return normalize(selections)


def generate_zone_playback(run_dir: Path, run_name: str, start_zone: str,
                           goal_zone: str, seed: int, *, task: str | None = None,
                           control_modes=None, horizon: int = ZONE_SCENARIO_TICKS,
                           behavior_mode: str = "match",
                           robot_types=None,
                           teammate_intent_knowledge: bool = True,
                           sweeping_enabled: bool = False,
                           random_gamepiece_placement: bool = False,
                           capture_stride: int = 10, device: str | None = None,
                           hopper_capacity: int = 60, scoring_bps: float = 25.,
                           progress_path: Path | None = None) -> dict:
    """Simulate one seeded, six-robot FRC match for dashboard playback."""
    from . import dashboard_simulation as simulation
    return simulation.generate_zone_playback(
        run_dir, run_name, start_zone, goal_zone, seed, task=task,
        control_modes=control_modes, horizon=horizon, robot_types=robot_types,
        behavior_mode=behavior_mode,
        teammate_intent_knowledge=teammate_intent_knowledge,
        sweeping_enabled=sweeping_enabled,
        random_gamepiece_placement=random_gamepiece_placement,
        device=device,
        capture_stride=capture_stride, hopper_capacity=hopper_capacity,
        scoring_bps=scoring_bps, progress_path=progress_path,
        _write_fn=_write_scenario_progress,
        _normalize_fn=_normalize_robot_control_selections)






def _complete_candidate_adstar_paths(record: dict) -> None:
    from .dashboard_simulation import _complete_candidate_adstar_paths as complete
    return complete(record)


def _complete_playback_adstar_paths(record: dict) -> None:
    from .dashboard_simulation import _complete_playback_adstar_paths as complete
    return complete(record)

def _json_or_default(path: Path, default: dict) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return json.dumps(default).encode()


def status_path_age(run_dir: Path) -> float:
    """Return age of the status file, or infinity when it is missing."""
    try:
        return max(0., time.time() - (run_dir / "status.json").stat().st_mtime)
    except OSError:
        return float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="checkpoints/tensor-ppo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--also-host", action="append", default=[],
                        help="Bind an additional address on the same port")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), create_handler(Path(args.run_dir)))
    for host in args.also_host:
        extra = ThreadingHTTPServer((host, args.port), create_handler(Path(args.run_dir)))
        threading.Thread(target=extra.serve_forever, daemon=True).start()
    hosts = [args.host, *args.also_host]
    print("FRC training dashboard: " + ", ".join(
        f"http://{host}:{args.port}/" for host in hosts), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
