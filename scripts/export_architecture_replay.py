"""Export an experimental checkpoint as a fresh dashboard-compatible replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from frc_defense.architecture_benchmark import ARCHITECTURES, distribution, metrics, source_provenance
from frc_defense.architecture_inputs import PerceptionInputs
from frc_defense.architecture_scenarios import SCENARIOS, make_env, reset_scenario
from frc_defense.dashboard_simulation import (
    replay_code_state, write_dashboard_replay_bundle,
)


def capture_frame(env, tick):
    """Read truth for rendering only, after the policy decision has completed."""
    active = env.piece_active[0]
    fuel = torch.cat((env.piece_pos[0], env.piece_owner[0, :, None]), -1)[active]
    visible = env._current_fuel_visibility[0].bool()
    tracks = env.track_mask[0].bool()
    opponents = env.opponent_valid[0].cpu().tolist()
    lengths = env.planner.last_lengths.reshape(env.n, 6)[0].cpu().tolist()
    paths = env.planner.last_path.reshape(env.n, 6, -1, 2)[0].cpu().tolist()
    return {
        "tick": tick, "robots": env.sim.pose[0].cpu().tolist(),
        "behavior_mode": "match", "robot_teams": env.team_ids.cpu().tolist(),
        "robot_control_modes": list(env.control_modes), "robot_roles": list(env.robot_roles),
        "robot_actions": env.last_actions[0].cpu().tolist(),
        "robot_opponent_visible": opponents,
        "robot_detected_opponents": [
            [*env.opponent_pose[0, robot, :2].cpu().tolist(), float(env.opponent_age[0, robot])]
            if opponents[robot] else None for robot in range(6)],
        "robot_detected_fuel": [env.track_pos[0, robot, visible[robot]].cpu().tolist()
                                for robot in range(6)],
        "robot_fuel_history": [torch.cat((env.track_pos[0, robot],
            env.track_age[0, robot, :, None]), -1)[tracks[robot] & ~visible[robot]].cpu().tolist()
            for robot in range(6)],
        "perception_track_timeout_s": env.perception_track_timeout,
        "perception_fov_degrees": env.perception_fov_degrees,
        "perception_range": env.perception_range,
        "camera_layout": {"height_m": .508, "horizontal_fov_degrees": env.perception_fov_degrees,
                          "stereo_baseline_m": env.stereo_baseline_m,
                          "mounts": ["intake", "opposite"]},
        "sizes": torch.stack((env.sim.length[0], env.sim.width[0]), -1).flatten().cpu().tolist(),
        "adstar_paths": [path[:int(length)] for path, length in zip(paths, lengths)],
        "fuel_pieces": fuel.cpu().tolist(), "hub_centers": env.hub_centers.cpu().tolist(),
        "hub_active": env.hub_active[0].cpu().tolist(),
        "fuel_score_count": env.fuel_score_count[0].cpu().tolist(),
        "fuel_acquisition_count": env.fuel_acquisition_count[0].cpu().tolist(),
        "match_elapsed": float(env.match_elapsed[0]),
        "match_remaining": float(env.match_remaining[0]),
    }


def write_bundle(result_path, record):
    """Use the dashboard's result, sidecar, compressed-view, and ready conventions."""
    write_dashboard_replay_bundle(result_path, record)


@torch.no_grad()
def export(args):
    torch.set_num_threads(1)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    if torch.version.hip:
        torch.backends.cudnn.enabled = False
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    model = ARCHITECTURES[architecture]().to(args.device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    config_path = args.checkpoint.parent / "metadata.json"
    training_metadata = json.loads(config_path.read_text()) if config_path.is_file() else {}
    repeat = args.action_repeat or training_metadata.get("config", {}).get("action_repeat", 10)
    env = make_env(args.scenario, 1, args.device, args.seed, horizon=args.ticks)
    obs = reset_scenario(env, args.scenario, args.seed)
    ids = SCENARIOS[args.scenario]["learned_slots"]
    builder = PerceptionInputs(env, ids)
    actions = torch.full((1, 6), 7, device=env.device, dtype=torch.long)
    frames = [capture_frame(env, 0)]
    for tick in range(0, args.ticks, repeat):
        dist, _ = distribution(model, builder(obs))
        actions[:, ids] = dist.logits.argmax(-1).reshape(1, len(ids))
        for _ in range(min(repeat, args.ticks - tick)):
            env.step(actions, capture_observation=False, capture_info=False)
        obs = env.observe()
        env._last_obs = obs
        frames.append(capture_frame(env, min(args.ticks, tick + repeat)))
    provenance = source_provenance()
    training_hash = payload.get("provenance", {}).get("source_sha256")
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    identity = json.dumps([architecture, args.scenario, args.seed, args.ticks,
                           repeat, checkpoint_hash, provenance["source_sha256"]]).encode()
    simulation_id = f"scenario-arch-{architecture}-{hashlib.sha256(identity).hexdigest()[:20]}"
    record = {
        "simulation_id": simulation_id, "task": "3v3", "matchup": args.scenario,
        "architecture": architecture, "behavior_mode": "match", "seed": args.seed,
        "scenario_seed": args.seed, "simulation_device": args.device,
        "field": {"length": env.sim.field_length, "width": env.sim.field_width,
                  "alliance_zone_depth": env.alliance_zone_depth,
                  "elements": [box.as_dict() for box in env.field_boxes]},
        "robot_teams": env.team_ids.cpu().tolist(), "robot_count": 6,
        "robots_per_alliance": 3, "robot_types": list(env.robot_types),
        "control_modes": list(env.control_modes), "robot_roles": list(env.robot_roles),
        "simulation_ticks": args.ticks, "simulated_seconds": args.ticks * env.dt,
        "dt": env.dt, "capture_stride": repeat, "action_repeat": repeat,
        "perception_interval": env.perception_interval,
        "simulation_constraints": {"fuel_physics": True, "perception_simulation": True,
                                   "max_fuel_per_robot": list(env.robot_fuel_capacity_values)},
        "code_state": replay_code_state(), "replay_source_sha256": provenance["source_sha256"],
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replay_provenance": provenance, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash, "checkpoint_training_source_sha256": training_hash,
        "checkpoint_source_outdated": training_hash != provenance["source_sha256"],
        "checkpoint_training_scenario": payload.get("scenario"),
        "metrics": metrics(env, ids), "miopen_enabled": torch.backends.cudnn.enabled,
        "runtime": {"torch": torch.__version__, "hip": torch.version.hip},
        "scenarios": [{"id": args.scenario,
                       "label": f"{architecture}: {args.scenario} (seed {args.seed})", "frames": frames}],
        "frames": frames,
    }
    output = args.output or ROOT / "evaluations/architecture-study-20261020/replays" / f"{simulation_id}.result.json"
    if not output.name.endswith(".result.json"):
        output = output.with_name(output.stem + ".result.json")
    write_bundle(output, record)
    if args.dashboard_run_dir:
        write_bundle(args.dashboard_run_dir / ".scenario-progress" / f"{simulation_id}.result.json", record)
    return {"simulation_id": simulation_id, "result": str(output.resolve()),
            "checkpoint_source_outdated": record["checkpoint_source_outdated"],
            "replay_source_sha256": record["replay_source_sha256"], "metrics": record["metrics"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--seed", type=int, default=100007)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dashboard-run-dir", type=Path)
    parser.add_argument("--action-repeat", type=int)
    args = parser.parse_args()
    if args.ticks < 1 or (args.action_repeat is not None and args.action_repeat < 1):
        parser.error("ticks and action repeat must be positive")
    print(json.dumps(export(args)), flush=True)


if __name__ == "__main__":
    main()
