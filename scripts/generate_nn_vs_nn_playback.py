#!/usr/bin/env python3
"""Record three seeded attacker-NN versus defender-NN field scenarios."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from frc_defense.tensor_sim import TensorDefenseEnv
from frc_defense.tensor_training import ActorCritic, OBS_DIM, OBS_NORMALIZATION, _device
from frc_defense.observation import normalize_tensor_observation_batch_in_place
from frc_defense.field import (ALLIANCE_ZONE_DEPTH, BUMP_ACCELERATION_SCALE,
    BUMP_SPEED_SCALE, bump_boxes, static_collision_boxes)


def load_policy(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=True)
    model = ActorCritic(payload.get("obs_dim", OBS_DIM), payload.get("action_dim", 3)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def swapped_observation(env: TensorDefenseEnv) -> torch.Tensor:
    """Build the attacker's standard observation with robot roles exchanged."""
    sim=env.sim
    own,other=1,0
    relative=env.goal-sim.pose[:,own,:2]
    parts=(sim.pose[:,own],sim.velocity[:,own],sim.pose[:,other],sim.velocity[:,other],
        relative,env.goal_radius[:,None],
        torch.full((env.n,1),sim.field_length,device=env.device),
        torch.full((env.n,1),sim.field_width,device=env.device),
        sim.length[:,[own,other]],sim.width[:,[own,other]],
        sim.accel[:,[own,other]]/10.)
    obstacles=(torch.cat((sim.obstacles,env.field_feature_obstacles),0)
               if env.field_feature_obstacles.shape[0] else sim.obstacles)
    features=torch.zeros((env.n,12),device=env.device,dtype=sim.pose.dtype)
    if obstacles.shape[0]:
        distance=(obstacles[None,:,:2]-sim.pose[:,own,None,:2]).square().sum(-1)
        chosen=obstacles[distance.topk(min(4,obstacles.shape[0]),dim=-1,largest=False).indices]
        features[:,:chosen.shape[1]*3]=chosen.reshape(env.n,-1)
    raw=torch.cat(parts+(features,),-1)
    speeds=sim.speed[:,[own,other]]
    omegas=sim.omega_limit[:,[own,other]]
    if env.normalize_observations:
        normalize_tensor_observation_batch_in_place(raw,sim.field_length,sim.field_width,speeds,omegas)
    return raw


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--defender",type=Path,default=Path("checkpoints/rebuilt-defense/policy.pt"))
    parser.add_argument("--attacker",type=Path,default=Path("checkpoints/rebuilt-counter-defense/policy.pt"))
    parser.add_argument("--output",type=Path,default=Path("checkpoints/nn-vs-nn"))
    parser.add_argument("--device",default="cuda:1")
    parser.add_argument("--seed",type=int,default=41027)
    parser.add_argument("--scenarios",type=int,default=9)
    parser.add_argument("--horizon",type=int,default=750)
    args=parser.parse_args()
    device=_device(args.device)
    defender,defender_payload=load_policy(args.defender,device)
    attacker,attacker_payload=load_policy(args.attacker,device)
    if defender_payload.get("observation_normalization") != attacker_payload.get("observation_normalization"):
        raise ValueError("attacker and defender checkpoints use different observation normalization")
    env=TensorDefenseEnv(num_envs=args.scenarios,task="defense",device=device,seed=args.seed,
        opponent="adstar",horizon=args.horizon,
        normalize_observations=defender_payload.get("observation_normalization")==OBS_NORMALIZATION)
    env.randomize=True
    obs,_=env.reset(seed=args.seed)
    planner=env._adstar_planners
    routes=[planner.last_path[i,:int(planner.last_lengths[i].item())].detach().cpu().tolist()
            for i in range(args.scenarios)]
    field={"length":env.sim.field_length,"width":env.sim.field_width,
        "alliance_zone_depth":ALLIANCE_ZONE_DEPTH,
        "elements":[b.as_dict() for b in env.field_boxes],
        "colliders":[b.as_dict() for b in static_collision_boxes(env.field_boxes)],
        "bump_regions":[b.as_dict() for b in bump_boxes(env.field_boxes)],
        "trench_paths":[b.as_dict() for b in env.field_boxes if "_trench_" in b.name and "_support_" not in b.name],
        "trench_supports":[b.as_dict() for b in env.field_boxes if "_trench_support_" in b.name],
        "bump_speed_scale":BUMP_SPEED_SCALE,"bump_acceleration_scale":BUMP_ACCELERATION_SCALE}
    def zone_name(x):
        depth=ALLIANCE_ZONE_DEPTH
        return "red" if x<depth else "blue" if x>=env.sim.field_length-depth else "center"
    scenarios=[{"id":str(i+1),"label":f"NN vs NN · scenario {i+1}","frames":[],
        "start":env.sim.pose[i,1,:2].detach().cpu().tolist(),
        "goal":env.goal[i].detach().cpu().tolist(),
        "start_zone":zone_name(float(env.sim.pose[i,1,0].item())),
        "goal_zone":zone_name(float(env.goal[i,0].item()))} for i in range(args.scenarios)]
    for i,route in enumerate(routes): scenarios[i]["adstar_reference_path"]=route
    alive=torch.ones(args.scenarios,device=device,dtype=torch.bool)
    start=time.perf_counter()
    frame_stride=3
    with torch.no_grad():
        for t in range(args.horizon):
            obs_defender=env._obs()
            obs_attacker=swapped_observation(env)
            action_d,_,_=defender.sample(obs_defender,deterministic=True)
            action_a,_,_=attacker.sample(obs_attacker,deterministic=True)
            command_d=torch.cat((action_d[:,:2]*env.sim.speed[:,0,None],
                (action_d[:,2]*env.sim.omega_limit[:,0])[:,None]),-1)
            command_a=torch.cat((action_a[:,:2]*env.sim.speed[:,1,None],
                (action_a[:,2]*env.sim.omega_limit[:,1])[:,None]),-1)
            commands=torch.stack((command_d,command_a),1)
            commands=torch.where(alive[:,None,None],commands,torch.zeros_like(commands))
            env.sim.step(commands)
            score=(env.goal-env.sim.pose[:,1,:2]).norm(dim=-1)<env.goal_radius
            alive &= ~score
            capture=(t%frame_stride==0 or bool(score.any().item()) or t==args.horizon-1)
            env.sim.velocity[~alive]=0.
            if capture:
                for i,scenario in enumerate(scenarios):
                    if scenario.get("finished"): continue
                    route=scenario["adstar_reference_path"]
                    frame={"robots":env.sim.pose[i].detach().cpu().tolist(),
                        "goal":env.goal[i].detach().cpu().tolist(),
                        "sizes":torch.stack((env.sim.length[i],env.sim.width[i]),-1).reshape(-1).detach().cpu().tolist(),
                        "goal_radius":float(env.goal_radius[i].item()),
                        "robot_effort_vectors":[command_d[i,:2].detach().cpu().tolist(),command_a[i,:2].detach().cpu().tolist()],
                        "adstar_path":route}
                    scenario["frames"].append(frame)
                    if not bool(alive[i].item()): scenario["finished"]=True
            if not bool(alive.any().item()): break
    for scenario in scenarios:
        scenario.pop("finished",None)
    elapsed=time.perf_counter()-start
    args.output.mkdir(parents=True,exist_ok=True)
    record={"task":"defense","matchup":"nn-vs-nn","scenario_seed":args.seed,
        "dt":env.dt,"field":field,"frames":[],"scenarios":scenarios,
        "defender_checkpoint":str(args.defender.resolve()),
        "attacker_checkpoint":str(args.attacker.resolve())}
    (args.output/"playback.json").write_text(json.dumps(record,separators=(",",":")))
    status={"status":"ready","algorithm":"crossplay","task":"defense",
        "matchup":"NN vs NN","opponent":"attacker NN","opponents":["defender NN","attacker NN"],
        "num_envs":args.scenarios,"completed_timesteps":sum(len(s["frames"]) for s in scenarios),
        "requested_timesteps":sum(len(s["frames"]) for s in scenarios),
        "transitions_per_second":sum(len(s["frames"]) for s in scenarios)/max(elapsed,1e-6),
        "elapsed_seconds":elapsed,"device":str(device),
        "device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
        "accelerator_backend":"ROCm" if torch.version.hip else ("CUDA" if torch.version.cuda else "CPU"),
        "defender_checkpoint":str(args.defender.resolve()),
        "attacker_checkpoint":str(args.attacker.resolve())}
    (args.output/"status.json").write_text(json.dumps(status,indent=2))
    print(json.dumps(status))


if __name__=="__main__":
    main()
