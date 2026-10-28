import ast
import dataclasses
import json
import subprocess
import time
from types import SimpleNamespace
import torch
from frc_defense import meco_track_training as training
from frc_defense.curriculum_scenarios import make_curriculum_env
from frc_defense.meco_pose_graph import build_legal_pose_graph
from frc_defense.meco_device_lease import lease_devices
from frc_defense.meco_gpu_attestation import verify_post_lease_wx9100

@dataclasses.dataclass
class Support:
    goal_xy: torch.Tensor
    valid_mask: torch.Tensor
    distances_m: torch.Tensor
    fallback_pair_tensor: torch.Tensor
    snapshot_id: str = 'xy-dedup:71'
    support_digest: str = ''
    order_digest: str = ''
    @property
    def pair_count(self): return self.valid_mask.numel()

import frc_defense.tensor_adstar as adstar

def baseline(*args, **kwargs):
    adstar._LEARNED_FUSION_DIAGNOSTIC = False
    before = len(dispatches)
    result = training._plan_a_route_support(*args, **kwargs)
    assert set(dispatches[before:]) == {'bellman'}
    return result

def candidate(*args, **kwargs):
    adstar._LEARNED_FUSION_DIAGNOSTIC = True
    before = len(dispatches)
    result = training._plan_a_route_support(*args, **kwargs)
    assert set(dispatches[before:]) == {'fused'}
    return result

out=[]
with lease_devices(('cuda:0',), owner='plan-a-xy-dedup-full') as lease:
    verify_post_lease_wx9100(lease, ('cuda:0',))
    extension = adstar._hip_adstar_extension()
    if extension is None: raise RuntimeError('ADStar HIP extension unavailable')
    dispatches = []
    class DispatchSpy:
        def __getattr__(self, name): return getattr(extension, name)
        def fused_route(self, *args):
            dispatches.append('fused')
            return extension.fused_route(*args)
        def bellman_sweep(self, *args):
            dispatches.append('bellman')
            return extension.bellman_sweep(*args)
    adstar._HIP_ADSTAR_EXTENSION = DispatchSpy()
    adstar._HIP_ADSTAR_EXTENSION_ATTEMPTED = True
    env=make_curriculum_env('pickup_simple','single',1,'cuda:0',71,horizon=10,fuel_physics=False)
    env.set_focal_pose_estimate((10.05,4.05,0.))
    graph=build_legal_pose_graph(env)
    n=graph.node_capacity
    valid=graph.node_valid[0].repeat_interleave(8)
    valid &= (torch.arange(n*8,device=env.device)%5 != 0)
    support=Support(graph.xy.clone(), valid, torch.zeros(n,device=env.device), torch.zeros((),device=env.device,dtype=torch.long))
    xx,yy=env.planner.xx,env.planner.yy
    for mode in ('learned',):
        config=SimpleNamespace(p_mode=mode)
        penalty=(xx*0.01 + yy*0.02) if mode=='learned' else None
        results={}; timings={}
        for name,fn in (('baseline',baseline),('candidate',candidate)) + (('candidate',candidate),('baseline',baseline),('baseline',baseline),('candidate',candidate)) * 5:
            torch.cuda.synchronize(); t=time.perf_counter()
            res=fn(env,graph,support,support.snapshot_id,penalty,config)
            torch.cuda.synchronize(); timings.setdefault(name, []).append(time.perf_counter()-t)
            results[name]=res
        a,ar=results['baseline']; b,br=results['candidate']
        compare={}
        for field in dataclasses.fields(ar):
            av,bv=getattr(ar,field.name),getattr(br,field.name)
            compare[field.name]=bool(torch.equal(av,bv)) if isinstance(av,torch.Tensor) else av==bv
        compare['support_mask']=bool(torch.equal(a.valid_mask,b.valid_mask))
        compare['fallback']=bool(torch.equal(a.fallback_pair_tensor,b.fallback_pair_tensor))
        compare['support_order_digest']=a.order_digest==b.order_digest
        compare['route_evidence']=training._plan_a_route_evidence_digest(ar,'a'*64,ar.distance_m,ar.distance_m)==training._plan_a_route_evidence_digest(br,'a'*64,br.distance_m,br.distance_m)
        out.append({'mode':mode,'nodes':n,'grid':env.planner.nx*env.planner.ny,'times':timings,'equal':compare,'status_counts':torch.bincount(ar.status_code,minlength=4).cpu().tolist(), 'valid_pairs':int(a.valid_mask.sum().cpu())})
        print(json.dumps(out[-1]),flush=True)
print(json.dumps(out,indent=2))
