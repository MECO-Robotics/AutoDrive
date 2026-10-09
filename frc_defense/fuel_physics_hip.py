"""Device-resident cached linked-list broadphase and fused fuel contacts."""
from __future__ import annotations
import math
import os
import warnings
from pathlib import Path
import torch
_EXT = None
_ATTEMPTED = False

def _extension():
    global _EXT, _ATTEMPTED
    if _ATTEMPTED:
        return _EXT
    _ATTEMPTED = True
    if not (torch.cuda.is_available() and torch.version.hip):
        return None
    try:
        from torch.utils import cpp_extension
        bundled = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
        rocm = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
                    (bundled if bundled.exists() else cpp_extension._find_rocm_home())).resolve()
        rocm_alias = rocm
        if " " in str(rocm):
            rocm_alias = Path("/tmp/autodrive-rocm-sdk")
            if rocm_alias.is_symlink() and rocm_alias.resolve() != rocm:
                rocm_alias.unlink()
            if not rocm_alias.exists():
                rocm_alias.symlink_to(rocm, target_is_directory=True)
            os.environ.setdefault("ROCM_HOME", str(rocm_alias))
            os.environ.setdefault("HIP_CLANG_PATH", str(rocm_alias / "lib/llvm/bin"))
        torch_lib = Path(torch.__file__).parent / "lib"
        if " " in str(torch_lib.resolve()):
            alias = Path("/tmp/autodrive-torch-lib")
            if alias.is_symlink() and alias.resolve() != torch_lib.resolve():
                alias.unlink()
            if not alias.exists():
                alias.symlink_to(torch_lib.resolve(), target_is_directory=True)
            torch_lib = alias
        old = (cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME)
        cpp_extension.TORCH_LIB_PATH = str(torch_lib)
        cpp_extension.ROCM_HOME = str(rocm_alias)
        cpp_extension.HIP_HOME = str(rocm_alias / "hip")
        src = Path(__file__).parent / "csrc"
        device_lib = next((p for p in (rocm_alias / "lib/llvm/amdgcn/bitcode",
                                        rocm_alias / "amdgcn/bitcode") if p.is_dir()), None)
        flags = ["-O3", "-ffp-contract=off"]
        if device_lib is not None:
            flags.append(f"--rocm-device-lib-path={device_lib}")
        build_dir = None
        if not os.environ.get("TORCH_EXTENSIONS_DIR"):
            build_dir = Path(__file__).parent.parent / "outputs/fuel_physics/extensions/native"
            build_dir.mkdir(parents=True, exist_ok=True)
        try:
            _EXT = cpp_extension.load(
                name="autodrive_fuel_physics_hip",
                sources=[str(src / "fuel_physics_hip.cpp"),
                         str(src / "fuel_physics_hip_kernel.cu")],
                with_cuda=True, build_directory=str(build_dir) if build_dir else None,
                extra_cflags=["-O3"], extra_cuda_cflags=flags,
                verbose=False)
        finally:
            cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = old
    except Exception as exc:
        warnings.warn(f"Coupled HIP fuel physics unavailable: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _EXT



class FuelHIP:
    def __init__(self, physics):
        self.physics = physics
        self.ext = _extension()
        self.available = self.ext is not None
        if not self.available:
            return
        p,c,s = physics,physics.config,physics.sim
        if p.pos.dtype != torch.float32:
            raise ValueError("HIP fuel state requires float32")
        self.cell = 2*c.radius+c.skin
        self.xyz = (math.ceil(s.field_length/self.cell)+2,
                    math.ceil(s.field_width/self.cell)+2,math.ceil(8/self.cell)+2)
        kw = dict(device=p.device,dtype=torch.int32)
        self.heads = torch.full((p.n,math.prod(self.xyz)),-1,**kw)
        self.next = torch.empty((p.n,p.p),**kw)
        self.list = torch.empty((p.n,p.p,c.max_neighbors),**kw)
        self.counts = torch.zeros((p.n,p.p),**kw)
        self.overflow = torch.zeros(p.n,**kw)
        self.rebuild = torch.ones(p.n,**kw)
        self.metrics = torch.zeros(2,**kw)
        self.reference = p.pos.clone()
        self.oldfree = torch.zeros_like(p.sleeping)
        self.dv = torch.zeros_like(p.vel)
        self.dw = torch.zeros_like(p.angular)
        self.rdv = torch.zeros_like(s.velocity)
        # Grid compaction bounds storage by neighbor capacity; overflowing
        # contacts use the fused path, preserving every contact impulse.
        capacity = (p.n*p.p*(p.p-1)//2 if c.broadphase == "all_pairs" or c.compact_capacity == "full" else min(p.n*p.p*c.max_neighbors,p.n*p.p*(p.p-1)//2)) if c.contact_mode == "compact" else 1
        self.contacts = torch.empty((max(1,capacity),2),**kw)
        self.contact_count = torch.zeros(1,**kw)
        self.robot_candidates = torch.empty((p.n,s.pose.shape[1],p.p),device=p.device,dtype=torch.bool)
        self.bumps = getattr(s,"bump_regions",torch.empty((0,4),device=p.device)).contiguous()
        self.support_height = torch.zeros((p.n,p.p),device=p.device,dtype=p.pos.dtype)
        p._support_height = self.support_height
        self.spawn_heights = torch.empty((p.n*p.p,),device=p.device,dtype=p.pos.dtype)
        self.boxes = getattr(s,"field_colliders",torch.empty((0,4),device=p.device)).contiguous()

    def spawn_height(self, xy):
        from .field import BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE
        count=xy.numel()//2
        if count>self.spawn_heights.numel():
            raise ValueError("spawn positions exceed allocated fuel slots")
        out=self.spawn_heights[:count].view(xy.shape[:-1])
        self.ext.spawn_height(xy.contiguous(),self.bumps,out,self.physics.config.radius,
                              BUMP_PANEL_THICKNESS,BUMP_RAMP_RISE)
        return out

    def invalidate(self):
        self.rebuild.fill_(1)

    def step(self, free, changed, dt, substeps):
        p,c,s = self.physics,self.physics.config,self.physics.sim
        # Spawns/teleports must invalidate even when displacement ends inside
        # the previous margin. No device-to-host check is needed.
        torch.maximum(self.rebuild, changed.any(-1).to(torch.int32), out=self.rebuild)
        tensors = [p.pos,p.vel,p.angular,p.sleeping,p.sleep_clock,free.contiguous(),
                   s.pose,s.velocity,s.length,s.width,s.mass,s.yaw_inertia_multiplier,
                   self.boxes,self.reference,self.oldfree,self.heads,self.next,self.list,
                   self.counts,self.overflow,self.rebuild,self.metrics,self.dv,self.dw,
                   self.rdv,self.contacts,self.contact_count,self.robot_candidates,self.bumps,self.support_height]
        from .field import BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE
        params = [c.radius,c.mass,c.gravity,c.stiffness,c.damping,c.friction,
                  c.rolling_friction,c.robot_height,c.skin,dt,s.field_length,s.field_width,
                  c.sleep_time,c.sleep_speed,c.sleep_angular_speed,c.max_compression_fraction,BUMP_PANEL_THICKNESS,BUMP_RAMP_RISE]
        dims = [p.n,p.p,s.pose.shape[1],self.boxes.shape[0],c.max_neighbors,*self.xyz,
                int(c.cache),(2 if c.sleep and c.sleep_islands else int(c.sleep)),int(c.broadphase=="all_pairs"),int(c.contact_mode=="compact"),int(c.solver=="colored"),int(c.robot_broadphase=="grid"),self.bumps.shape[0],self.contacts.shape[0]]
        for _ in range(substeps):
            self.ext.substep(tensors,params,dims)
            if c.sleep and c.sleep_islands:
                from .fuel_physics_sleep import update_island_sleep
                update_island_sleep(p,free,dt)

    def refresh_stats(self):
        p=self.physics
        values=self.metrics.cpu().tolist()
        p.stats["rebuilds"],p.stats["overflows"]=values
        p.stats["candidate_pairs"]=int(self.counts.sum().item())
        if p.config.broadphase=="all_pairs":
            p.stats["candidate_pairs"]=p.n*p.p*(p.p-1)//2
        return p.stats
