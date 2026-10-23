"""Export a fresh curriculum policy or deterministic-teacher dashboard replay."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from frc_defense.curriculum_training import load_checkpoint,source_identity,_held_step
from frc_defense.curriculum_policy import PerceptionActorCritic
from frc_defense.curriculum_scenarios import make_curriculum_env,teacher_actions
from frc_defense.architecture_inputs import PerceptionInputs
from frc_defense.dashboard_simulation import replay_code_state
from export_architecture_replay import capture_frame,write_bundle

def export(args):
 torch.set_num_threads(1)
 if args.device.startswith('cuda'):torch.cuda.set_device(args.device)
 torch.backends.cudnn.enabled=False
 sources=source_identity();source=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
 if args.checkpoint:
  checkpoint_bytes=args.checkpoint.read_bytes();checkpoint_hash=hashlib.sha256(checkpoint_bytes).hexdigest()
  policy,_,payload=load_checkpoint(args.checkpoint,device=args.device);policy.eval()
  architecture=payload['config']['architecture'];training_source=hashlib.sha256(json.dumps(payload['sources'],sort_keys=True).encode()).hexdigest()
 else:policy='teacher';architecture='deterministic_teacher';checkpoint_hash=None;training_source=source
 env=make_curriculum_env(args.scenario,args.category,1,args.device,args.seed)
 builder=PerceptionInputs(env,(3,)) if isinstance(policy,PerceptionActorCritic) else None
 frames=[capture_frame(env,0)];decisions=[]
 while True:
  with torch.no_grad():
   obs=env._last_obs;mask=obs[:,3,-8:].bool()
   action=teacher_actions(env,obs)[:,3] if policy=='teacher' else policy.sample(builder(obs) if builder else obs[:,3],deterministic=True,action_mask=mask)[0]
   actions=torch.full((1,6),7,device=env.device,dtype=torch.long);actions[:,3]=action
   decisions.append({'tick':int(env.steps[0]),'requested_action':int(action[0]),'valid':bool(mask[0,action[0]])})
   _,_,done,truncated,_=_held_step(env,actions,torch.ones(1,device=env.device,dtype=torch.bool),args.action_ticks)
  frames.append(capture_frame(env,int(env.steps[0])))
  if bool(done[0]|truncated[0]):break
 assert source_identity()==sources and all(d['valid'] for d in decisions)
 identity=[architecture,args.scenario,args.category,args.seed,checkpoint_hash,source,args.action_ticks]
 sid='scenario-curriculum-'+architecture+'-'+hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
 outcomes={k:v[0].item() for k,v in env.curriculum_metrics.outcomes().items()}
 record={'simulation_id':sid,'task':'3v3','matchup':args.scenario,'architecture':architecture,'category':args.category,'behavior_mode':'match','seed':args.seed,'scenario_seed':args.seed,'simulation_device':args.device,'field':{'length':env.sim.field_length,'width':env.sim.field_width,'alliance_zone_depth':env.alliance_zone_depth,'elements':[box.as_dict() for box in env.field_boxes]},'robot_teams':env.team_ids.cpu().tolist(),'robot_count':6,'robots_per_alliance':3,'robot_types':list(env.robot_types),'control_modes':list(env.control_modes),'robot_roles':list(env.robot_roles),'simulation_ticks':int(env.steps[0]),'simulated_seconds':int(env.steps[0])*env.dt,'dt':env.dt,'capture_stride':args.action_ticks,'action_repeat':args.action_ticks,'perception_interval':env.perception_interval,'simulation_constraints':{'fuel_physics':True,'perception_simulation':True,'max_fuel_per_robot':list(env.robot_fuel_capacity_values)},'code_state':replay_code_state(),'replay_source_sha256':source,'replay_provenance':sources,'checkpoint':str(args.checkpoint.resolve()) if args.checkpoint else None,'checkpoint_sha256':checkpoint_hash,'checkpoint_training_source_sha256':training_source,'checkpoint_source_outdated':training_source!=source,'curriculum_outcomes':outcomes,'curriculum_configuration':env.curriculum,'policy_decisions':decisions,'scenarios':[{'id':args.scenario,'label':f'{architecture}: {args.scenario}/{args.category} seed {args.seed}','frames':frames}],'frames':frames}
 assert not record['checkpoint_source_outdated'];write_bundle(args.output,record)
 if args.dashboard_run_dir:write_bundle(args.dashboard_run_dir/'.scenario-progress'/(sid+'.result.json'),record)
 return {'simulation_id':sid,'source_sha256':source,'scenario':args.scenario,'category':args.category,'seed':args.seed,'ticks':record['simulation_ticks'],'outcomes':outcomes,'result':str(args.output.resolve())}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);choice=parser.add_mutually_exclusive_group(required=True);choice.add_argument('--checkpoint',type=Path);choice.add_argument('--teacher',action='store_true')
 parser.add_argument('--scenario',default='pickup_simple');parser.add_argument('--category',default='single');parser.add_argument('--seed',type=int,required=True);parser.add_argument('--device',default='cpu');parser.add_argument('--action-ticks',type=int,default=12);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dashboard-run-dir',type=Path)
 args=parser.parse_args();print(json.dumps(export(args)),flush=True)
