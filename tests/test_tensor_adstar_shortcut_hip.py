import math
import pytest
import torch
from frc_defense.meco_device_lease import lease_devices
from frc_defense.tensor_adstar import _hip_shortcut_extension

def ref(path,lengths,blocked,res,samples):
    n,p=path.shape[:2]; out=torch.zeros_like(path); out[:,0]=path[:,0]; ln=(lengths>0).long(); ai=torch.zeros_like(lengths)
    idx=torch.arange(1,samples+1,device=path.device); flat=blocked.reshape(n,-1); ny=blocked.shape[2]; nx=blocked.shape[1]
    for i in range(1,p-1):
        a=path.gather(1,ai[:,None,None].expand(-1,1,2)).squeeze(1); d=path[:,i+1]-a
        st=torch.ceil(torch.maximum(d[:,0].abs(),d[:,1].abs())/(res*.5)).long().clamp(1,samples)
        frac=idx[None].to(path.dtype)/st[:,None].to(path.dtype); v=a[:,None,:]+d[:,None,:]*frac[:,:,None]
        x=(v[...,0]/res).floor().long().clamp(0,nx-1); y=(v[...,1]/res).floor().long().clamp(0,ny-1)
        hit=flat.gather(1,x*ny+y)&(idx[None]<=st[:,None]); append=(i<lengths-1)&hit.any(-1)
        slot=ln.clamp_max(p-1); out.scatter_(1,slot[:,None,None].expand(-1,1,2),torch.where(append[:,None],path[:,i],torch.zeros_like(path[:,i]))[:,None]); ai=torch.where(append,torch.full_like(ai,i),ai); ln+=append.long()
    final=path.gather(1,(lengths-1).clamp_min(0)[:,None,None].expand(-1,1,2)).squeeze(1); hr=lengths>1
    out.scatter_(1,ln.clamp_max(p-1)[:,None,None].expand(-1,1,2),torch.where(hr[:,None],final,torch.zeros_like(final))[:,None]); ln+=hr.long(); ln=torch.where(lengths>0,ln,lengths); ln=torch.where(hr,ln,torch.ones_like(ln)).clamp_max(p)
    end=out.gather(1,(ln-1).clamp_min(0)[:,None,None].expand(-1,1,2)).squeeze(1); out=torch.where((torch.arange(p,device=path.device)[None]>=ln[:,None])[:,:,None],end[:,None,:],out)
    return out,ln



def test_hip_shortcut_grid_boundaries_and_invalid_rows():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("HIP device required")
    with lease_devices(("cuda:0",), owner="plan-a-los-boundary-regression"):
        device=torch.device("cuda:0");ext=_hip_shortcut_extension()
        if ext is None: raise RuntimeError("HIP shortcut extension unavailable")
        generator=torch.Generator(device=device).manual_seed(51268)
        res=.3;nx,ny=55,28;samples=max(2,int(math.ceil(math.hypot(16.5,8.2)/(res*.5)))+1)
        for case in range(80):
            path=torch.rand((16,72,2),device=device,generator=generator)*torch.tensor([16.5,8.2],device=device)
            if case%4==0: path=(path/res).round()*res
            lengths=torch.randint(0,73,(16,),device=device,generator=generator)
            grid=torch.rand((16,nx,ny),device=device,generator=generator)<(.0 if case%5==0 else .08)
            actual=torch.empty_like(path);lens=torch.empty_like(lengths);expected,elens=ref(path,lengths,grid,res,samples)
            ext.shortcut(path,lengths,grid,actual,lens,samples,res)
            assert torch.equal(expected,actual) and torch.equal(elens,lens), case
        boundaries=torch.arange(1,nx,device=device,dtype=torch.float32)*res
        candidates=torch.cat((boundaries,torch.nextafter(boundaries,torch.full_like(boundaries,-float("inf"))),torch.nextafter(boundaries,torch.full_like(boundaries,float("inf")))))
        path=torch.zeros((len(candidates),4,2),device=device)
        path[:,:,0]=candidates[:,None]
        lengths=torch.full((len(candidates),),4,device=device,dtype=torch.long)
        grid=torch.rand((len(candidates),nx,ny),device=device,generator=generator)<.3
        actual=torch.empty_like(path);lens=torch.empty_like(lengths);expected,elens=ref(path,lengths,grid,res,samples)
        ext.shortcut(path,lengths,grid,actual,lens,samples,res)
        assert torch.equal(expected,actual) and torch.equal(elens,lens)
        lengths[0]=5;lengths[1]=-1;path[2,0,0]=float("nan")
        path[3,0,0]=3.4e38;path[3,2,0]=-3.4e38
        ext.shortcut(path,lengths,grid,actual,lens,samples,res)
        assert lens[:4].cpu().tolist()==[-1,-1,-1,-1]
        with pytest.raises(RuntimeError,match="invalid shortcut parameters"):
            ext.shortcut(path,lengths,grid[:,:0].contiguous(),actual,lens,samples,res)
