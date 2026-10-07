"""Batched device-resident grid planner for tensor simulation rollouts.

The planner uses parallel Bellman sweeps on a clearance-inflated 8-connected
grid, then applies PathPlanner LocalADStar-style line-of-sight reduction and
Bézier path shaping before calculating a drivetrain speed profile. The route
search stays batched and device-resident. Path shaping is adapted from the MIT-
licensed PathPlanner implementation:
https://github.com/mjansen4857/pathplanner/tree/main/pathplannerlib-python/pathplannerlib/pathfinders.py
"""
from __future__ import annotations

import math
import os
import warnings

import torch


_HIP_ADSTAR_EXTENSION = None
_HIP_ADSTAR_EXTENSION_ATTEMPTED = False
_HIP_OCCUPANCY_EXTENSION = None
_HIP_OCCUPANCY_EXTENSION_ATTEMPTED = False
_HIP_PATH_REFERENCE_EXTENSION = None
_HIP_PATH_REFERENCE_EXTENSION_ATTEMPTED = False
_HIP_OCCUPANCY_ENABLED = os.environ.get("AUTODRIVE_FUSED_ADSTAR_OCCUPANCY_HIP", "1") != "0"
# The isolated strict-math fused path has exact 256x96 rollout parity. Set to 0
# to opt out and use the Torch reference implementation.
_HIP_PATH_REFERENCE_ENABLED = os.environ.get("AUTODRIVE_FUSED_PATH_REFERENCE_HIP", "1") != "0"


def _hip_adstar_extension():
    """Load the optional fused sweep kernel; Torch remains the portable path."""
    global _HIP_ADSTAR_EXTENSION, _HIP_ADSTAR_EXTENSION_ATTEMPTED
    if os.environ.get("AUTODRIVE_DISABLE_HIP_ADSTAR") == "1":
        return None
    if _HIP_ADSTAR_EXTENSION_ATTEMPTED:
        return _HIP_ADSTAR_EXTENSION
    _HIP_ADSTAR_EXTENSION_ATTEMPTED = True
    if not (torch.cuda.is_available() and torch.version.hip):
        return None
    try:
        from pathlib import Path
        from torch.utils import cpp_extension
        bundled_sdk = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
        rocm_sdk = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
                        (bundled_sdk if bundled_sdk.exists() else cpp_extension._find_rocm_home()))
        rocm_sdk = rocm_sdk.resolve()
        # ROCm's bundled hipcc wrapper cannot safely shell-split its resolved
        # clang path when the checkout lives under a directory with spaces.
        # A short alias keeps the toolchain itself on the same installation.
        rocm_alias = rocm_sdk
        if " " in str(rocm_sdk):
            rocm_alias = Path("/tmp/autodrive-rocm-sdk")
            if rocm_alias.is_symlink() and rocm_alias.resolve() != rocm_sdk:
                rocm_alias.unlink()
            if not rocm_alias.exists():
                rocm_alias.symlink_to(rocm_sdk, target_is_directory=True)
            os.environ.setdefault("ROCM_HOME", str(rocm_alias))
            os.environ.setdefault("HIP_CLANG_PATH", str(rocm_alias / "lib/llvm/bin"))
        load = cpp_extension.load
        torch_lib = Path(torch.__file__).parent / "lib"
        torch_lib_alias = torch_lib.resolve()
        if " " in str(torch_lib_alias):
            torch_lib_alias = Path("/tmp/autodrive-torch-lib")
            if torch_lib_alias.is_symlink() and torch_lib_alias.resolve() != torch_lib.resolve():
                torch_lib_alias.unlink()
            if not torch_lib_alias.exists():
                torch_lib_alias.symlink_to(torch_lib.resolve(), target_is_directory=True)
        # Ninja's link command does not quote -L paths containing spaces.
        original_torch_lib = cpp_extension.TORCH_LIB_PATH
        original_rocm_home = cpp_extension.ROCM_HOME
        original_hip_home = cpp_extension.HIP_HOME
        cpp_extension.TORCH_LIB_PATH = str(torch_lib_alias)
        cpp_extension.ROCM_HOME = str(rocm_alias)
        cpp_extension.HIP_HOME = str(rocm_alias / "hip")

        source_dir = Path(__file__).parent / "csrc"
        device_lib = next((p for p in (
            rocm_alias / "lib/llvm/amdgcn/bitcode",
            rocm_alias / "amdgcn/bitcode",
        ) if p.is_dir()), None)
        cuda_flags = ["-O3"]
        if device_lib is not None:
            cuda_flags.append(f"--rocm-device-lib-path={device_lib}")
        try:
            _HIP_ADSTAR_EXTENSION = load(
                name="autodrive_tensor_adstar_hip",
                sources=[str(source_dir / "tensor_adstar_hip.cpp"),
                         str(source_dir / "tensor_adstar_hip_kernel.cu"),
                         str(source_dir / "tensor_adstar_route_kernel.cu")],
                with_cuda=True,
                extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags,
                verbose=False,
            )
        finally:
            cpp_extension.TORCH_LIB_PATH = original_torch_lib
            cpp_extension.ROCM_HOME = original_rocm_home
            cpp_extension.HIP_HOME = original_hip_home
    except Exception as exc:  # Optional accelerator must not break CPU/portable runs.
        warnings.warn(f"Fused AD* HIP kernel unavailable; using Torch sweeps: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _HIP_ADSTAR_EXTENSION


def _hip_path_reference_extension():
    """Load the isolated path-reference kernel with strict FP contraction off."""
    global _HIP_PATH_REFERENCE_EXTENSION, _HIP_PATH_REFERENCE_EXTENSION_ATTEMPTED
    if os.environ.get("AUTODRIVE_DISABLE_HIP_ADSTAR") == "1":
        return None
    if _HIP_PATH_REFERENCE_EXTENSION_ATTEMPTED:
        return _HIP_PATH_REFERENCE_EXTENSION
    _HIP_PATH_REFERENCE_EXTENSION_ATTEMPTED = True
    if not (torch.cuda.is_available() and torch.version.hip):
        return None
    try:
        from pathlib import Path
        from torch.utils import cpp_extension
        bundled_sdk = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
        rocm_sdk = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
                        (bundled_sdk if bundled_sdk.exists() else cpp_extension._find_rocm_home()))
        rocm_sdk = rocm_sdk.resolve()
        rocm_alias = rocm_sdk
        if " " in str(rocm_sdk):
            rocm_alias = Path("/tmp/autodrive-rocm-sdk")
            if rocm_alias.is_symlink() and rocm_alias.resolve() != rocm_sdk:
                rocm_alias.unlink()
            if not rocm_alias.exists():
                rocm_alias.symlink_to(rocm_sdk, target_is_directory=True)
            os.environ.setdefault("ROCM_HOME", str(rocm_alias))
            os.environ.setdefault("HIP_CLANG_PATH", str(rocm_alias / "lib/llvm/bin"))
        torch_lib = Path(torch.__file__).parent / "lib"
        torch_lib_alias = torch_lib.resolve()
        if " " in str(torch_lib_alias):
            torch_lib_alias = Path("/tmp/autodrive-torch-lib")
            if torch_lib_alias.is_symlink() and torch_lib_alias.resolve() != torch_lib.resolve():
                torch_lib_alias.unlink()
            if not torch_lib_alias.exists():
                torch_lib_alias.symlink_to(torch_lib.resolve(), target_is_directory=True)
        original_torch_lib = cpp_extension.TORCH_LIB_PATH
        original_rocm_home = cpp_extension.ROCM_HOME
        original_hip_home = cpp_extension.HIP_HOME
        cpp_extension.TORCH_LIB_PATH = str(torch_lib_alias)
        cpp_extension.ROCM_HOME = str(rocm_alias)
        cpp_extension.HIP_HOME = str(rocm_alias / "hip")
        source_dir = Path(__file__).parent / "csrc"
        device_lib = next((p for p in (
            rocm_alias / "lib/llvm/amdgcn/bitcode",
            rocm_alias / "amdgcn/bitcode",
        ) if p.is_dir()), None)
        cuda_flags = ["-O3", "-ffp-contract=off"]
        if device_lib is not None:
            cuda_flags.append(f"--rocm-device-lib-path={device_lib}")
        try:
            _HIP_PATH_REFERENCE_EXTENSION = cpp_extension.load(
                name="autodrive_tensor_adstar_path_reference_hip",
                sources=[str(source_dir / "tensor_adstar_reference_hip.cpp"),
                         str(source_dir / "tensor_adstar_reference_kernel.cu")],
                with_cuda=True,
                extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags,
                verbose=False,
            )
        finally:
            cpp_extension.TORCH_LIB_PATH = original_torch_lib
            cpp_extension.ROCM_HOME = original_rocm_home
            cpp_extension.HIP_HOME = original_hip_home
    except Exception as exc:
        warnings.warn(f"Fused AD* path-reference HIP kernel unavailable; using Torch reference: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _HIP_PATH_REFERENCE_EXTENSION


def _hip_occupancy_extension():
    """Load the strict-math fused AD* occupancy builder; Torch is the fallback."""
    global _HIP_OCCUPANCY_EXTENSION, _HIP_OCCUPANCY_EXTENSION_ATTEMPTED
    if (not _HIP_OCCUPANCY_ENABLED or
            os.environ.get("AUTODRIVE_DISABLE_HIP_ADSTAR") == "1"):
        return None
    if _HIP_OCCUPANCY_EXTENSION_ATTEMPTED:
        return _HIP_OCCUPANCY_EXTENSION
    _HIP_OCCUPANCY_EXTENSION_ATTEMPTED = True
    if not (torch.cuda.is_available() and torch.version.hip):
        return None
    try:
        from pathlib import Path
        from torch.utils import cpp_extension
        bundled_sdk = (Path(torch.__file__).parent / ".." / "_rocm_sdk_core").resolve()
        rocm_sdk = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or
                        (bundled_sdk if bundled_sdk.exists() else cpp_extension._find_rocm_home()))
        rocm_sdk = rocm_sdk.resolve()
        rocm_alias = rocm_sdk
        if " " in str(rocm_sdk):
            rocm_alias = Path("/tmp/autodrive-rocm-sdk")
            if rocm_alias.is_symlink() and rocm_alias.resolve() != rocm_sdk:
                rocm_alias.unlink()
            if not rocm_alias.exists():
                rocm_alias.symlink_to(rocm_sdk, target_is_directory=True)
            os.environ.setdefault("ROCM_HOME", str(rocm_alias))
            os.environ.setdefault("HIP_CLANG_PATH", str(rocm_alias / "lib/llvm/bin"))
        torch_lib = Path(torch.__file__).parent / "lib"
        torch_lib_alias = torch_lib.resolve()
        if " " in str(torch_lib_alias):
            torch_lib_alias = Path("/tmp/autodrive-torch-lib")
            if torch_lib_alias.is_symlink() and torch_lib_alias.resolve() != torch_lib.resolve():
                torch_lib_alias.unlink()
            if not torch_lib_alias.exists():
                torch_lib_alias.symlink_to(torch_lib.resolve(), target_is_directory=True)
        original_torch_lib = cpp_extension.TORCH_LIB_PATH
        original_rocm_home = cpp_extension.ROCM_HOME
        original_hip_home = cpp_extension.HIP_HOME
        cpp_extension.TORCH_LIB_PATH = str(torch_lib_alias)
        cpp_extension.ROCM_HOME = str(rocm_alias)
        cpp_extension.HIP_HOME = str(rocm_alias / "hip")
        source_dir = Path(__file__).parent / "csrc"
        device_lib = next((p for p in (
            rocm_alias / "lib/llvm/amdgcn/bitcode",
            rocm_alias / "amdgcn/bitcode",
        ) if p.is_dir()), None)
        cuda_flags = ["-O3", "-ffp-contract=off"]
        if device_lib is not None:
            cuda_flags.append(f"--rocm-device-lib-path={device_lib}")
        try:
            _HIP_OCCUPANCY_EXTENSION = cpp_extension.load(
                name="autodrive_tensor_adstar_occupancy_hip",
                sources=[str(source_dir / "tensor_adstar_occupancy_hip.cpp"),
                         str(source_dir / "tensor_adstar_occupancy_hip_kernel.cu")],
                with_cuda=True,
                extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags,
                verbose=False,
            )
        finally:
            cpp_extension.TORCH_LIB_PATH = original_torch_lib
            cpp_extension.ROCM_HOME = original_rocm_home
            cpp_extension.HIP_HOME = original_hip_home
    except Exception as exc:
        warnings.warn(f"Fused AD* occupancy HIP kernel unavailable; using Torch reference: {exc}",
                      RuntimeWarning, stacklevel=2)
    return _HIP_OCCUPANCY_EXTENSION


class TensorADStar:
    """GPU batched, footprint-aware A* style potential planner."""

    def __init__(self, env, resolution: float = .3, max_points: int = 72,
                 sweeps: int = 72, avoid_bumps: bool = False,
                 early_convergence: bool = True):
        self.device = env.device
        self.dtype = env.sim.pose.dtype
        self.n = env.n
        self._sim = env.sim
        self._all_active = torch.ones((self.n,), device=self.device,
                                      dtype=torch.bool)
        self.resolution = float(resolution)
        self.nx = int(math.ceil(env.sim.field_length / resolution))
        self.ny = int(math.ceil(env.sim.field_width / resolution))
        self.max_points = int(max_points)
        self.sweeps = int(sweeps)
        self.early_convergence = bool(early_convergence)
        self.avoid_bumps = bool(avoid_bumps)
        self.footprint_clearance = 0.
        self.length = env.sim.field_length
        self.width = env.sim.field_width
        self.steer_rate_limit=float(getattr(getattr(env.sim,"swerve",None),
                                             "steer_rate_limit",12.))
        self.x = (torch.arange(self.nx, device=self.device, dtype=self.dtype) + .5) * resolution
        self.y = (torch.arange(self.ny, device=self.device, dtype=self.dtype) + .5) * resolution
        self.xx, self.yy = torch.meshgrid(self.x, self.y, indexing="ij")
        self._boxes = torch.as_tensor(env.sim.field_colliders, device=self.device,
                                      dtype=self.dtype).reshape(-1, 4)
        self._extra_circles = env.sim.obstacles
        self._fused_blocked_hip_enabled = _HIP_OCCUPANCY_ENABLED
        self._empty_dynamic = torch.empty((0, 4), device=self.device, dtype=self.dtype)
        self._bumps = torch.as_tensor(
            [(b.x, b.y, b.length / 2, b.width / 2) for b in env.field_boxes
            if "bump" in getattr(b, "name", "")], device=self.device,
            dtype=self.dtype).reshape(-1, 4)
        self._trench_xs = torch.as_tensor(
            sorted({b.x for b in env.field_boxes
                    if getattr(b, "name", "").endswith("_trench_lower")}),
            device=self.device, dtype=self.dtype)
        # Bump traversal cost depends only on the fixed field geometry. Keep
        # one grid instead of rebuilding and materializing the same grid for
        # every world on every replan.
        self._bump_cost = torch.ones((self.nx, self.ny), device=self.device,
                                     dtype=self.dtype)
        if self._bumps.numel():
            bump_dx = (self.xx[None] - self._bumps[:, 0, None, None]).abs()
            bump_dy = (self.yy[None] - self._bumps[:, 1, None, None]).abs()
            in_bump = ((bump_dx <= self._bumps[:, 2, None, None]) &
                       (bump_dy <= self._bumps[:, 3, None, None]))
            self._bump_cost += in_bump.any(0).to(self.dtype) * .12
        self._neighbors = ((1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.),
                           (1, 1, 1.41421356237), (1, -1, 1.41421356237),
                           (-1, 1, 1.41421356237), (-1, -1, 1.41421356237))
        self._offsets = torch.tensor([(dx, dy) for dx, dy, _ in self._neighbors],
                                     device=self.device)
        self._diagonal_offsets = self._offsets.prod(dim=-1) != 0
        self._batch = torch.arange(self.n, device=self.device)
        self._edge_cost = torch.tensor((1.41421356237, 1., 1.41421356237, 1.,
                                        1., 1., 1.41421356237, 1., 1.41421356237),
                                       device=self.device, dtype=self.dtype)[None, :, None]
        self.last_path = torch.zeros((self.n, self.max_points, 2), device=self.device,
                                     dtype=self.dtype)
        self.last_lengths = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.last_intercept = torch.zeros((self.n, 2), device=self.device, dtype=self.dtype)
        self.last_intercept_time = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_start = torch.zeros((self.n, 2), device=self.device, dtype=self.dtype)
        self.last_goal = torch.zeros_like(self.last_start)
        self.last_heading = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_progress = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_trench_alignment = torch.zeros(self.n, device=self.device,
                                                 dtype=torch.bool)
        self.last_speed_profile = torch.zeros((self.n,self.max_points),device=self.device,dtype=self.dtype)
        self.potential = torch.zeros((self.n, self.nx, self.ny), device=self.device,
                                     dtype=self.dtype)

    def _blocked(self, heading, length, width, dynamic=None):
        if (self._fused_blocked_hip_enabled and self.device.type == "cuda" and
                self.dtype == torch.float32):
            extension = _hip_occupancy_extension()
            if extension is not None:
                dynamic_tensor = self._empty_dynamic if dynamic is None else dynamic
                grid_clearance = (0. if self.avoid_bumps else self.resolution * .72)
                grid_clearance += self.footprint_clearance
                return extension.blocked_grid(
                    heading.contiguous(), length.contiguous(), width.contiguous(),
                    self.x, self.y, self._boxes, self._bumps, self._extra_circles,
                    dynamic_tensor.contiguous(), self.length, self.width,
                    grid_clearance, self.avoid_bumps)
        c, s = heading.cos().abs(), heading.sin().abs()
        # Narrow bump-to-wall lanes can be one grid row wide. The robot
        # footprint itself remains inflated; avoid an extra cell pad there.
        grid_clearance=(0. if self.avoid_bumps else self.resolution*.72)
        grid_clearance += self.footprint_clearance
        ex = (length * c + width * s)[:, None, None] * .5 + grid_clearance
        ey = (length * s + width * c)[:, None, None] * .5 + grid_clearance
        blocked = ((self.xx[None] < ex) | (self.xx[None] > self.length-ex) |
                   (self.yy[None] < ey) | (self.yy[None] > self.width-ey))
        if self._boxes.numel():
            dx = self.xx[None, None] - self._boxes[None, :, 0, None, None]
            dy = self.yy[None, None] - self._boxes[None, :, 1, None, None]
            hx = self._boxes[None, :, 2, None, None] + ex[:, None]
            hy = self._boxes[None, :, 3, None, None] + ey[:, None]
            # Oriented-rectangle SAT against each axis-aligned field collider.
            forward = dx * heading.cos()[:, None, None, None] + dy * heading.sin()[:, None, None, None]
            lateral = -dx * heading.sin()[:, None, None, None] + dy * heading.cos()[:, None, None, None]
            sat_f = length[:, None, None, None] * .5 + self._boxes[None, :, 2, None, None] * heading.cos().abs()[:, None, None, None] + self._boxes[None, :, 3, None, None] * heading.sin().abs()[:, None, None, None] + grid_clearance
            sat_l = width[:, None, None, None] * .5 + self._boxes[None, :, 2, None, None] * heading.sin().abs()[:, None, None, None] + self._boxes[None, :, 3, None, None] * heading.cos().abs()[:, None, None, None] + grid_clearance
            hit = ((dx.abs() <= hx) & (dy.abs() <= hy) &
                   (forward.abs() <= sat_f) & (lateral.abs() <= sat_l))
            blocked |= hit.any(dim=1)
        # Bumps are traversable terrain. Their extra travel time is represented
        # by _bump_cost; marking their footprint blocked made every AD* route
        # detour around a surface that the physics simulator can climb.
        if self._extra_circles.numel():
            dx = self.xx[None, None] - self._extra_circles[None, :, 0, None, None]
            dy = self.yy[None, None] - self._extra_circles[None, :, 1, None, None]
            r = self._extra_circles[None, :, 2, None, None] + torch.maximum(ex, ey)[:, None]
            blocked |= ((dx.square() + dy.square()) <= r.square()).any(dim=1)
        if dynamic is not None:
            dx = self.xx[None, None] - dynamic[:, 0, None, None, None]
            dy = self.yy[None, None] - dynamic[:, 1, None, None, None]
            hx = dynamic[:, 2, None, None, None] + ex[:, None]
            hy = dynamic[:, 3, None, None, None] + ey[:, None]
            forward = dx * heading.cos()[:, None, None, None] + dy * heading.sin()[:, None, None, None]
            lateral = -dx * heading.sin()[:, None, None, None] + dy * heading.cos()[:, None, None, None]
            sf = length[:, None, None, None] * .5 + dynamic[:, 2, None, None, None] * heading.cos().abs()[:, None, None, None] + dynamic[:, 3, None, None, None] * heading.sin().abs()[:, None, None, None]
            sl = width[:, None, None, None] * .5 + dynamic[:, 2, None, None, None] * heading.sin().abs()[:, None, None, None] + dynamic[:, 3, None, None, None] * heading.cos().abs()[:, None, None, None]
            blocked |= ((dx.abs() <= hx) & (dy.abs() <= hy) & (forward.abs() <= sf) & (lateral.abs() <= sl)).squeeze(1)
        return blocked

    def _block_robot_obstacles(self, blocked, length, width, robot_obstacles):
        """Inflate teammate and opponent chassis into each route's grid."""
        if robot_obstacles is None:
            return blocked
        # Bound temporary memory for large batched training rollouts.
        for offset in range(0, blocked.shape[0], 64):
            end=min(offset+64,blocked.shape[0])
            centers=robot_obstacles[offset:end,:,:2]
            radii=robot_obstacles[offset:end,:,2]
            dx=self.xx[None,None]-centers[...,0,None,None]
            dy=self.yy[None,None]-centers[...,1,None,None]
            moving_radius=.5*torch.minimum(length[offset:end],width[offset:end])
            clearance=moving_radius[:,None]+radii.clamp_min(0.)
            blocked[offset:end] |= (((dx.square()+dy.square()) <=
                clearance[...,None,None].square()) &
                (radii[...,None,None] > 0.)).any(dim=1)
        return blocked

    def _indices(self, points):
        ix = (points[:, 0] / self.resolution).floor().long().clamp(0, self.nx - 1)
        iy = (points[:, 1] / self.resolution).floor().long().clamp(0, self.ny - 1)
        return ix, iy

    def _footprint_path_clear(self, path, heading, length, width):
        """Check a predicted tangent-heading trajectory against field boxes."""
        if not self._boxes.numel():
            return torch.ones(path.shape[0],device=path.device,dtype=torch.bool)
        return torch.cat([
            self._footprint_path_clear_chunk(path[offset:offset+8],
                heading[offset:offset+8],length[offset:offset+8],width[offset:offset+8],
                self.footprint_clearance)
            for offset in range(0,path.shape[0],8)])

    def _footprint_path_clear_chunk(self,path,heading,length,width,clearance):
        """Bound peak temporary memory while checking batched path segments."""
        batch_n,point_count=path.shape[:2]
        delta=path[:,1:]-path[:,:-1]
        segment_heading=torch.atan2(delta[...,1],delta[...,0])
        headings=torch.cat((heading[:,None],segment_heading),-1)
        heading_delta=torch.atan2(torch.sin(headings[:,1:]-headings[:,:-1]),
                                  torch.cos(headings[:,1:]-headings[:,:-1]))
        # Keep both translation and corner rotation increments below 2 cm.
        outer=.5*torch.sqrt(length.square()+width.square())
        steps=torch.ceil(torch.maximum(delta.norm(dim=-1),
                    heading_delta.abs()*outer[:,None])/.02).long().clamp_min(1)
        sample_index=torch.arange(1,65,device=path.device,dtype=path.dtype)
        fraction=sample_index[None,None,:]/steps.clamp_max(64)[...,None].to(path.dtype)
        positions=(path[:,:-1,None,:]+delta[:,:,None,:]*fraction[...,None])
        angles=(headings[:,:-1,None]+heading_delta[:,:,None]*fraction)
        valid_sample=sample_index[None,None,:]<=steps.clamp_max(64)[...,None]
        c,s=angles.cos(),angles.sin()
        boxes=self._boxes
        dx=positions[...,None,0]-boxes[None,None,None,:,0]
        dy=positions[...,None,1]-boxes[None,None,None,:,1]
        hx=boxes[None,None,None,:,2]; hy=boxes[None,None,None,:,3]
        nearest_x=(dx.abs()-hx).clamp_min(0.)
        nearest_y=(dy.abs()-hy).clamp_min(0.)
        separation=torch.sqrt(nearest_x.square()+nearest_y.square())
        robot_outer=.5*torch.sqrt(length.square()+width.square())
        box_outer=torch.sqrt(hx.square()+hy.square())
        broad_clear=separation>robot_outer[:,None,None,None]+box_outer+clearance
        robot_inner=.5*torch.minimum(length,width)
        definite_hit=separation<=robot_inner[:,None,None,None]+clearance

        ambiguous=~broad_clear&~definite_hit
        hit=definite_hit.clone()
        capturing=(path.device.type=="cuda" and
                   torch.cuda.is_current_stream_capturing())
        if capturing:
            hl=length[:,None,None,None]*.5; hw=width[:,None,None,None]*.5
            signed=torch.stack((dx,dy,dx*c[...,None]+dy*s[...,None],
                                -dx*s[...,None]+dy*c[...,None]),-1)
            robot_radius=torch.stack((
                (c[...,None].abs()*hl+s[...,None].abs()*hw).expand_as(signed[...,0]),
                (s[...,None].abs()*hl+c[...,None].abs()*hw).expand_as(signed[...,0]),
                hl.expand_as(signed[...,0]),hw.expand_as(signed[...,0])),-1)
            box_radius=torch.stack((hx.expand_as(signed[...,0]),hy.expand_as(signed[...,0]),
                hx*c[...,None].abs()+hy*s[...,None].abs(),
                hx*s[...,None].abs()+hy*c[...,None].abs()),-1)
            sat_hit=(robot_radius+box_radius-signed.abs()+clearance).amin(-1)>=0.
            hit|=ambiguous&sat_hit
        elif bool(ambiguous.any().item()):
            indices=torch.nonzero(ambiguous,as_tuple=False)
            bi,si,ti,oi=indices.unbind(-1)
            dx_a,dy_a=dx[bi,si,ti,oi],dy[bi,si,ti,oi]
            c_a,s_a=c[bi,si,ti],s[bi,si,ti]
            hl_a,hw_a=length[bi]*.5,width[bi]*.5
            hx_a,hy_a=boxes[oi,2],boxes[oi,3]
            penetration=torch.stack((
                c_a.abs()*hl_a+s_a.abs()*hw_a+hx_a-dx_a.abs(),
                s_a.abs()*hl_a+c_a.abs()*hw_a+hy_a-dy_a.abs(),
                hl_a+hx_a*c_a.abs()+hy_a*s_a.abs()-
                    (dx_a*c_a+dy_a*s_a).abs(),
                hw_a+hx_a*s_a.abs()+hy_a*c_a.abs()-
                    (-dx_a*s_a+dy_a*c_a).abs()),-1) + clearance
            hit[bi,si,ti,oi]=penetration.amin(-1)>=0.
        path_hit=(hit&valid_sample[...,None]).any(dim=(1,2,3))
        ex=.5*(length[:,None,None]*c.abs()+width[:,None,None]*s.abs())
        ey=.5*(length[:,None,None]*s.abs()+width[:,None,None]*c.abs())
        outside=((positions[...,0]<ex)|(positions[...,0]>self.length-ex)|
                 (positions[...,1]<ey)|(positions[...,1]>self.width-ey))
        wall_hit=(outside&valid_sample).any(dim=(1,2))
        return ~(path_hit|wall_hit)

    def _repair_paths_locally(self,path,lengths,heading,length,width,blocked,
                              speed,acceleration):
        """Use the existing local SE(2) repair only for failed route rows."""
        from .adstar import ADStarPlanner

        repaired=path.clone()
        failed=torch.nonzero(lengths>1,as_tuple=False).flatten().tolist()
        if not failed:
            return repaired
        boxes=self._boxes.detach().cpu().tolist()
        omega=getattr(self._sim,"omega_limit",None)
        alpha=getattr(self._sim,"alpha",None)
        def row_limit(values,row,default):
            if values is None:
                return default
            flat=values.reshape(-1)
            if flat.numel()==path.shape[0]:
                index=row
            else:
                pose=getattr(self._sim,"pose",None)
                robots=(int(pose.shape[1]) if pose is not None and pose.ndim>=2
                        else 1)
                index=min(row//max(robots,1),flat.numel()-1)
            value=flat[index]
            if value.ndim:
                value=value.max()
            return float(value.item())
        for row in failed:
            route_length=int(lengths[row].item())
            points=path[row,:route_length].detach().cpu().tolist()
            colliders=list(boxes)
            planner=ADStarPlanner(self.length,self.width,colliders,
                resolution=self.resolution,robot_length=float(length[row].item()),
                robot_width=float(width[row].item()),
                robot_heading=float(heading[row].item()),
                max_speed=float(speed[row].item()) if speed is not None else 4.8,
                max_acceleration=(float(acceleration[row].item())
                                  if acceleration is not None else 8.),
                max_angular_speed=row_limit(omega,row,8.),
                max_angular_acceleration=row_limit(alpha,row,18.))
            sim_velocity=getattr(self._sim,"velocity",None)
            if sim_velocity is not None and sim_velocity.reshape(-1,3).shape[0]==path.shape[0]:
                current=sim_velocity.reshape(-1,3)[row]
            elif sim_velocity is not None:
                world=min(row,sim_velocity.shape[0]-1)
                nearest=(self._sim.pose[world,:,:2]-path[row,0]).square().sum(-1).argmin()
                current=sim_velocity[world,nearest]
            else:
                current=None
            initial_velocity=((float(current[0].item()),float(current[1].item()))
                              if current is not None else (0.,0.))
            initial_angular_velocity=(float(current[2].item())
                                      if current is not None else 0.)
            candidate=planner._finalize_path(
                points,initial_velocity,initial_angular_velocity)
            if len(candidate)<2:
                repaired[row]=path[row,0]
                continue
            stations=[0.]
            for a,b in zip(candidate,candidate[1:]):
                stations.append(stations[-1]+math.hypot(b[0]-a[0],b[1]-a[1]))
            total=stations[-1]
            if total<=1.e-8:
                repaired[row]=path[row,0]
                continue
            sampled=[]; segment=0
            for i in range(path.shape[1]):
                target=total*i/(path.shape[1]-1)
                while segment<len(stations)-2 and stations[segment+1]<target:
                    segment+=1
                span=max(stations[segment+1]-stations[segment],1.e-8)
                t=(target-stations[segment])/span
                sampled.append((candidate[segment][0]+(candidate[segment+1][0]-candidate[segment][0])*t,
                                candidate[segment][1]+(candidate[segment+1][1]-candidate[segment][1])*t))
            repaired[row]=torch.tensor(sampled,device=path.device,dtype=path.dtype)
        return repaired

    def _shortcut_path(self, path, lengths, blocked, heading, length, width,
                       speed=None, acceleration=None):
        """Apply PathPlanner's LOS simplification and Bézier corner smoothing.

        Keep normal route operations batched; only failed footprint checks
        enter the host-side local repair path.
        """
        batch_n, point_count = path.shape[:2]
        source=path.clone()
        simplified=torch.zeros_like(path)
        simplified[:,0]=source[:,0]
        simple_length=(lengths>0).long()
        anchor_index=torch.zeros_like(lengths)
        sample_count=max(2,int(math.ceil(math.hypot(self.length,self.width)/
                                         (self.resolution*.5)))+1)
        sample_index=torch.arange(1,sample_count+1,device=path.device)
        blocked_flat=blocked.reshape(batch_n,-1)

        # PathPlanner appends a grid point only when the last retained point
        # cannot see the following point through free cells.
        for index in range(1,point_count-1):
            active=index < (lengths-1)
            anchor=source.gather(1,anchor_index[:,None,None].expand(-1,1,2)).squeeze(1)
            endpoint=source[:,index+1]
            delta=endpoint-anchor
            steps=torch.ceil(torch.maximum(delta[:,0].abs(),delta[:,1].abs())/
                             (self.resolution*.5)).long().clamp(1,sample_count)
            fraction=(sample_index[None].to(path.dtype)/steps[:,None].to(path.dtype))
            points=anchor[:,None,:]+delta[:,None,:]*fraction[:,:,None]
            ix=(points[...,0]/self.resolution).floor().long().clamp(0,self.nx-1)
            iy=(points[...,1]/self.resolution).floor().long().clamp(0,self.ny-1)
            cell=blocked_flat.gather(1,(ix*self.ny+iy).reshape(batch_n,-1)).view(
                batch_n,sample_count)
            line_blocked=(cell & (sample_index[None]<=steps[:,None])).any(-1)
            append=active & line_blocked
            slot=simple_length.clamp_max(point_count-1)
            simplified.scatter_(1,slot[:,None,None].expand(-1,1,2),
                                torch.where(append[:,None],source[:,index],
                                            torch.zeros_like(source[:,index]))[:,None])
            anchor_index=torch.where(append,torch.full_like(anchor_index,index),anchor_index)
            simple_length+=append.long()

        final_index=(lengths-1).clamp_min(0)
        final=source.gather(1,final_index[:,None,None].expand(-1,1,2)).squeeze(1)
        has_route=lengths>1
        simplified.scatter_(1,simple_length.clamp_max(point_count-1)[:,None,None].expand(-1,1,2),
                            torch.where(has_route[:,None],final,torch.zeros_like(final))[:,None])
        simple_length+=has_route.long()
        simple_length=torch.where(lengths>0,simple_length,lengths)
        simple_length=torch.where(has_route,simple_length,torch.ones_like(simple_length))
        simple_length=simple_length.clamp_max(point_count)
        simple_last=simplified.gather(
            1,(simple_length-1).clamp_min(0)[:,None,None].expand(-1,1,2)).squeeze(1)
        pad_index=torch.arange(point_count,device=path.device)[None]
        simplified=torch.where((pad_index>=simple_length[:,None])[:,:,None],
                               simple_last[:,None,:],simplified)

        # PathPlanner converts each corner to an incoming/outgoing pose pair.
        pose_count=torch.where(has_route,2*simple_length-2,torch.ones_like(simple_length))
        pose_slots=2*point_count-2
        poses=torch.zeros((batch_n,pose_slots,2),device=path.device,dtype=path.dtype)
        headings=torch.zeros((batch_n,pose_slots),device=path.device,dtype=path.dtype)
        poses[:,0]=simplified[:,0]
        first_delta=simplified[:,1]-simplified[:,0]
        first_heading=torch.atan2(first_delta[:,1],first_delta[:,0])
        headings[:,0]=first_heading
        if point_count>2:
            previous=simplified[:,:-2]
            corner=simplified[:,1:-1]
            following=simplified[:,2:]
            incoming=corner-previous
            outgoing=following-corner
            corner_slots=torch.arange(1,point_count-1,device=path.device)
            valid_corner=corner_slots[None] < (simple_length[:,None]-1)
            incoming_heading=torch.atan2(incoming[...,1],incoming[...,0])
            outgoing_heading=torch.atan2(outgoing[...,1],outgoing[...,0])
            incoming_anchor=previous+.8*incoming
            outgoing_anchor=corner+.2*outgoing
            odd_slots=(2*corner_slots-1)[None,:,None].expand(batch_n,-1,2)
            even_slots=(2*corner_slots)[None,:,None].expand(batch_n,-1,2)
            poses.scatter_(1,odd_slots,torch.where(valid_corner[...,None],incoming_anchor,
                                                   simple_last[:,None,:]))
            poses.scatter_(1,even_slots,torch.where(valid_corner[...,None],outgoing_anchor,
                                                    simple_last[:,None,:]))
            headings.scatter_(1,(2*corner_slots-1)[None].expand(batch_n,-1),
                              torch.where(valid_corner,incoming_heading,first_heading[:,None]))
            headings.scatter_(1,(2*corner_slots)[None].expand(batch_n,-1),
                              torch.where(valid_corner,outgoing_heading,first_heading[:,None]))
        previous_index=(simple_length-2).clamp_min(0)
        previous=simplified.gather(1,previous_index[:,None,None].expand(-1,1,2)).squeeze(1)
        final_delta=simple_last-previous
        final_heading=torch.atan2(final_delta[:,1],final_delta[:,0])
        end_slot=(pose_count-1).clamp_min(0)
        poses.scatter_(1,end_slot[:,None,None].expand(-1,1,2),simple_last[:,None,:])
        headings.scatter_(1,end_slot[:,None],final_heading[:,None])
        pose_index=torch.arange(pose_slots,device=path.device)[None]
        poses=torch.where((pose_index>=pose_count[:,None])[:,:,None],
                          simple_last[:,None,:],poses)
        headings=torch.where(pose_index>=pose_count[:,None],final_heading[:,None],headings)

        # Sample PathPlanner's cubic segments, validate the smoothed route
        # against the inflated occupancy grid, and retry shorter handles when
        # a curve clips an obstacle. This mirrors adstar.py's .4 -> .1 search.
        curve_samples=8
        t=torch.arange(1,curve_samples+1,device=path.device,dtype=path.dtype)/curve_samples
        segment_delta=poses[:,1:]-poses[:,:-1]
        segment_distance=segment_delta.norm(dim=-1)
        unit_heading=torch.stack((headings.cos(),headings.sin()),-1)
        curve_count=(pose_count-1).clamp_min(0)
        segment_index=torch.arange(pose_slots-1,device=path.device)[None]
        valid_segment=segment_index<curve_count[:,None]
        p0=poses[:,:-1]
        p3=poses[:,1:]
        resolved=torch.zeros((batch_n,),device=path.device,dtype=torch.bool)

        def resample(points, point_lengths):
            delta=points[:,1:]-points[:,:-1]
            distance=delta.norm(dim=-1)
            index=torch.arange(points.shape[1]-1,device=path.device)[None]
            distance=torch.where(index<(point_lengths-1)[:,None],distance,0.)
            station=torch.cat((torch.zeros((batch_n,1),device=path.device,dtype=path.dtype),
                               distance.cumsum(-1)),-1)
            total=station[:,-1]
            target=total[:,None]*torch.linspace(0.,1.,point_count,device=path.device,
                                                dtype=path.dtype)[None]
            hi=torch.searchsorted(station.contiguous(),target.contiguous(),right=True).clamp(
                1,points.shape[1]-1)
            lo=hi-1
            a=station.gather(1,lo); b=station.gather(1,hi)
            fraction=(target-a)/(b-a).clamp_min(1.e-8)
            first=points.gather(1,lo[:,:,None].expand(-1,-1,2))
            second=points.gather(1,hi[:,:,None].expand(-1,-1,2))
            sampled=first+(second-first)*fraction[:,:,None]
            return sampled,total

        # The unsmoothed LOS polyline is the safe fallback, also sampled at
        # equal arc-length spacing so route tracking remains well conditioned.
        line_path,_=resample(simplified,simple_length)
        line_ix=(line_path[...,0]/self.resolution).floor().long().clamp(0,self.nx-1)
        line_iy=(line_path[...,1]/self.resolution).floor().long().clamp(0,self.ny-1)
        line_cells=blocked_flat.gather(1,(line_ix*self.ny+line_iy).reshape(batch_n,-1)).view(
            batch_n,point_count)
        line_clear=~line_cells.any(-1)
        line_clear &= self._footprint_path_clear(line_path,heading,length,width)
        raw_path,_=resample(source,lengths.clamp_min(1))
        # Keep the lattice route as a fallback when the denser, equal-arc
        # resampling lands in an adjacent inflated grid cell. The route kernel
        # already checked its lattice segments against occupancy; rechecking
        # sampled points can reject a valid path at cell boundaries. Retain
        # the continuous chassis sweep check here so field geometry remains
        # the final safety constraint.
        raw_clear=self._footprint_path_clear(raw_path,heading,length,width)
        fallback=torch.where(line_clear[:,None,None],line_path,
                             torch.where(raw_clear[:,None,None],raw_path,
                                         source[:,0:1].expand(-1,point_count,-1)))
        selected_curve=torch.where(has_route[:,None,None],fallback,
                                   source[:,0:1].expand(-1,point_count,-1))
        resolved=~has_route
        for control_factor in (.4,.28,.18,.1,0.):
            next_control=p0+unit_heading[:,:-1]*(control_factor*segment_distance)[...,None]
            prev_control=p3-unit_heading[:,1:]*(control_factor*segment_distance)[...,None]
            tt=t[None,None,:,None]
            u=1.-tt
            curve=(u.pow(3)*p0[:,:,None,:] + 3*u.square()*tt*next_control[:,:,None,:] +
                   3*u*tt.square()*prev_control[:,:,None,:] + tt.pow(3)*p3[:,:,None,:])
            curve=torch.where(valid_segment[:,:,None,None],curve,
                              simple_last[:,None,None,:])
            curve=torch.cat((poses[:,:1],curve.reshape(batch_n,-1,2)),1)
            curve_length=1+curve_count*curve_samples
            curve_end=curve.gather(1,(curve_length-1).clamp_min(0)[:,None,None].expand(-1,1,2)).squeeze(1)
            curve_index=torch.arange(curve.shape[1],device=path.device)[None]
            curve=torch.where((curve_index>=curve_length[:,None])[:,:,None],
                              curve_end[:,None,:],curve)
            curve_ix=(curve[...,0]/self.resolution).floor().long().clamp(0,self.nx-1)
            curve_iy=(curve[...,1]/self.resolution).floor().long().clamp(0,self.ny-1)
            curve_cells=blocked_flat.gather(
                1,(curve_ix*self.ny+curve_iy).reshape(batch_n,-1)).view(batch_n,-1)
            curve_valid=torch.arange(curve.shape[1],device=path.device)[None] < curve_length[:,None]
            curve_clear=~(curve_cells & curve_valid).any(-1)
            sampled,_=resample(curve,curve_length)
            ix=(sampled[...,0]/self.resolution).floor().long().clamp(0,self.nx-1)
            iy=(sampled[...,1]/self.resolution).floor().long().clamp(0,self.ny-1)
            cells=blocked_flat.gather(1,(ix*self.ny+iy).reshape(batch_n,-1)).view(
                batch_n,point_count)
            clear=curve_clear & ~cells.any(-1)
            clear &= self._footprint_path_clear(sampled,heading,length,width)
            use=has_route & ~resolved & clear
            selected_curve=torch.where(use[:,None,None],sampled,selected_curve)
            resolved |= use

        selected_curve=torch.where(has_route[:,None,None],selected_curve,
                                   source[:,0:1].expand(-1,point_count,-1))
        unresolved=has_route&~resolved&~line_clear&~raw_clear
        can_repair=(path.device.type=="cpu" or
                    not torch.cuda.is_current_stream_capturing())
        if can_repair and bool(unresolved.any().item()):
            repair_source=source.clone()
            repair_lengths=torch.where(unresolved,lengths,torch.ones_like(lengths))
            repaired=self._repair_paths_locally(
                repair_source,repair_lengths,heading,length,width,blocked,speed,acceleration)
            repair_delta=repaired[:,1:]-repaired[:,:-1]
            repair_t=torch.arange(1,9,device=path.device,dtype=path.dtype)/8
            dense_repair=(repaired[:,:-1,None]+repair_delta[:,:,None]*repair_t[None,None,:,None])
            dense_repair=torch.cat((repaired[:,:1],dense_repair.reshape(batch_n,-1,2)),1)
            rx=(dense_repair[...,0]/self.resolution).floor().long().clamp(0,self.nx-1)
            ry=(dense_repair[...,1]/self.resolution).floor().long().clamp(0,self.ny-1)
            repaired_clear=self._footprint_path_clear(
                repaired,heading,length,width)
            selected_curve=torch.where((unresolved&repaired_clear)[:,None,None],
                                       repaired,selected_curve)
            failed=unresolved&~repaired_clear
            selected_curve=torch.where(failed[:,None,None],
                source[:,0:1].expand(-1,point_count,-1),selected_curve)
        # Keep a single-point route as a stop command; valid multi-point routes
        # use all slots after resampling their PathPlanner-style spline.
        out_lengths=torch.where(has_route,torch.full_like(lengths,point_count),
                                torch.where(lengths>0,torch.ones_like(lengths),lengths))
        out_lengths=torch.where(lengths>0,out_lengths,torch.zeros_like(out_lengths))
        path.copy_(selected_curve)
        lengths.copy_(out_lengths)
        return path,lengths

    def _speed_profile(self, path, lengths, speed, lateral_friction, acceleration):
        """PathPlanner-style curvature caps and backward braking envelope."""
        batch_n=path.shape[0]
        delta=path[:,1:]-path[:,:-1]
        distance=delta.norm(dim=-1)
        direction=delta/distance[...,None].clamp_min(1.e-6)
        cross=direction[:,:-1,0]*direction[:,1:,1]-direction[:,:-1,1]*direction[:,1:,0]
        dot=(direction[:,:-1]*direction[:,1:]).sum(-1)
        turn=torch.atan2(cross,dot).abs()
        arc=.5*(distance[:,:-1]+distance[:,1:]).clamp_min(1.e-4)
        curvature=turn/arc
        point_curve=torch.zeros((batch_n,path.shape[1]),device=self.device,dtype=self.dtype)
        point_curve[:,1:-1]=curvature
        mu=(lateral_friction if lateral_friction is not None else
            torch.full((batch_n,),1.2,device=self.device,dtype=self.dtype)).clamp_min(.1)
        accel=(acceleration if acceleration is not None else
               torch.full((batch_n,),8.,device=self.device,dtype=self.dtype)).clamp_min(.2)
        vcap=(speed if speed is not None else
              torch.full((batch_n,),4.8,device=self.device,dtype=self.dtype)).clamp_min(.1)[:,None]
        lateral=(.65*mu[:,None]*9.81/point_curve.clamp_min(1.e-6)).sqrt()
        steer=self.steer_rate_limit/point_curve.clamp_min(1.e-6)
        caps=torch.minimum(vcap,torch.minimum(lateral,steer))
        caps=torch.where(point_curve>1.e-6,caps,vcap)
        endpoint=(lengths-1).clamp_min(0)[:,None]
        endpoint_mask=(torch.arange(path.shape[1],device=path.device)[None,:] == endpoint)
        caps=torch.where(endpoint_mask,torch.zeros_like(caps),caps)
        station=torch.cat((torch.zeros((batch_n,1),device=self.device,dtype=self.dtype),
                           distance.cumsum(-1)),-1)
        delta_station=station[:,None,:]-station[:,:,None]
        indices=torch.arange(path.shape[1],device=self.device)
        future=indices[None,None,:]>=indices[None,:,None]
        valid=(indices[None,None,:]<lengths[:,None,None])&future
        braking=.65*accel[:,None,None]
        reachable=(caps[:,None,:].square()+2*braking*delta_station.clamp_min(0)).sqrt()
        return torch.where(valid,reachable,
                           torch.full_like(reachable,float("inf"))).amin(-1)

    def plan_waypoints(self, start, waypoints, waypoint_indices, heading,
                       length, width, *, lookahead=0, fallback_goal=None,
                       waypoint_mask=None, **plan_kwargs):
        """Plan toward indexed points in per-row waypoint chains.

        Waypoints are stored as ``[planner_rows, points, xy]``. Rows masked
        out of the waypoint route use ``fallback_goal`` and share the same AD*
        solve, which lets strategic actions coexist with spline sweeps.
        """
        waypoints = torch.as_tensor(waypoints, device=self.device, dtype=self.dtype)
        if waypoints.ndim != 3 or waypoints.shape[0] != self.n or waypoints.shape[-1] != 2:
            raise ValueError("waypoints must have shape [planner rows, points, 2]")
        if waypoints.shape[1] < 1:
            raise ValueError("waypoint chains must contain at least one point")
        indices = torch.as_tensor(waypoint_indices, device=self.device,
                                  dtype=torch.long).reshape(self.n)
        target_indices = torch.remainder(indices + int(lookahead), waypoints.shape[1])
        rows = torch.arange(self.n, device=self.device)
        waypoint_goal = waypoints[rows, target_indices]
        if fallback_goal is not None:
            fallback_goal = torch.as_tensor(fallback_goal, device=self.device,
                                            dtype=self.dtype).reshape(self.n, 2)
            if waypoint_mask is None:
                goal = waypoint_goal
            else:
                mask = torch.as_tensor(waypoint_mask, device=self.device,
                                       dtype=torch.bool).reshape(self.n)
                goal = torch.where(mask[:, None], waypoint_goal, fallback_goal)
        else:
            goal = waypoint_goal
        return self.plan(start, goal, heading, length, width, **plan_kwargs)

    def plan(self, start, goal, heading, length, width, speed=None,
             defender=None, defender_velocity=None, dynamic_defender=False,
             lateral_friction=None, acceleration=None, active_mask=None,
             robot_obstacles=None, static_active_mask=False):
        """Plan selected worlds and return the persistent batched route state."""
        full_batch = active_mask is None
        selected = None
        masked_full_batch = False
        if full_batch:
            batch_n = self.n
        else:
            active_mask = torch.as_tensor(active_mask, device=self.device,
                                          dtype=torch.bool).reshape(self.n)
            if static_active_mask:
                # CUDA graph replay needs a fixed batch shape. Compute all six
                # routes and mask persistent writes so inactive rows retain
                # exactly the same cached planner state.
                batch_n = self.n
                masked_full_batch = True
            else:
                selected = torch.nonzero(active_mask, as_tuple=False).flatten()
                batch_n = selected.numel()
        if batch_n == 0:
            return self.last_path,self.last_lengths,self.last_intercept,self.last_intercept_time
        def select_rows(value):
            if value is None:
                return None
            return value if (full_batch or masked_full_batch) else value[selected]

        def write_rows(destination, values):
            if full_batch:
                destination.copy_(values)
            elif masked_full_batch:
                destination.copy_(torch.where(active_mask.reshape(
                    (self.n,) + (1,) * (destination.ndim - 1)),
                    values, destination))
            else:
                destination[selected] = values

        heading = select_rows(heading.reshape(self.n)).to(self.dtype)
        start, goal = select_rows(start).to(self.dtype), select_rows(goal).to(self.dtype)
        length,width=select_rows(length),select_rows(width)
        speed=select_rows(speed)
        defender=select_rows(defender)
        defender_velocity=select_rows(defender_velocity)
        lateral_friction=select_rows(lateral_friction)
        acceleration=select_rows(acceleration)
        robot_obstacles=select_rows(robot_obstacles)
        trench_alignment=torch.zeros(batch_n,device=self.device,dtype=torch.bool)
        if self._trench_xs.numel():
            low_x=torch.minimum(start[:,0],goal[:,0])[:,None]
            high_x=torch.maximum(start[:,0],goal[:,0])[:,None]
            crosses_trench=((self._trench_xs[None]>=low_x) &
                            (self._trench_xs[None]<=high_x)).any(-1)
            trench_alignment=crosses_trench
            # The clear openings are at the field edges. A horizontal chassis
            # footprint can route sideways to either opening, cross the trench,
            # then continue toward a center-field goal without clipping corners.
            cross_heading=torch.where(goal[:,0]>=start[:,0],
                torch.zeros_like(heading),torch.full_like(heading,math.pi))
            heading=torch.where(trench_alignment,cross_heading,heading)
        write_rows(self.last_trench_alignment,trench_alignment)
        if dynamic_defender:
            delta = goal - start
            unit = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            speed_values=(torch.full((batch_n,),4.,device=self.device,dtype=self.dtype)
                          if speed is None else speed.reshape(batch_n))
            va = unit * speed_values[:,None]
            q = defender - start
            rel = defender_velocity - va
            a = rel.square().sum(-1).clamp_min(1e-8)
            b = 2 * (q * rel).sum(-1)
            c = q.square().sum(-1) - .45**2
            disc = (b.square() - 4*a*c).clamp_min(0).sqrt()
            t0 = (-b - disc) / (2*a)
            t1 = (-b + disc) / (2*a)
            t = torch.where(c <= 0, torch.zeros_like(c),
                torch.where((b.square() - 4*a*c >= 0) & (t0 >= 0), t0,
                torch.where((b.square() - 4*a*c >= 0) & (t1 >= 0), t1,
                         (-(q*rel).sum(-1)/a).clamp_min(0))))
            t = t.clamp(max=1.5)
            intercept = defender + defender_velocity * t[:, None]
            dynamic = torch.cat((intercept, torch.full((batch_n, 2), .45, device=self.device,
                                                       dtype=self.dtype)), -1)
        else:
            t = torch.zeros(batch_n, device=self.device, dtype=self.dtype)
            intercept = torch.zeros_like(start)
            dynamic = None
        blocked = self._blocked(heading, length, width, dynamic)
        blocked = self._block_robot_obstacles(blocked,length,width,robot_obstacles)
        hip_extension = _hip_adstar_extension() if self.device.type == "cuda" else None
        route_shared_bytes = (2 * self.nx * self.ny + 5 * 72) * 4
        if self.device.type == "cuda":
            try:
                shared_memory_limit = torch.cuda.get_device_properties(self.device).shared_memory_per_block
            except (AttributeError, RuntimeError, ValueError):
                shared_memory_limit = 0
        else:
            shared_memory_limit = 0
        fused_route_supported = (
            hip_extension is not None and self.dtype == torch.float32 and
            self.max_points == 72 and self.nx * self.ny <= 4096 and
            shared_memory_limit >= route_shared_bytes and blocked.is_contiguous()
        )
        # Targets can sit in a footprint-clearance strip at the field edge
        # (notably FUEL staged inside a DEPOT). Project blocked goals to the
        # nearest reachable cell so AD* does not command into the boundary.
        sx, sy = self._indices(start); gx, gy = self._indices(goal)
        batch = torch.arange(batch_n,device=self.device)
        blocked_goal=blocked[batch,gx,gy]
        # Only blocked goals need projection. The dense all-row distance
        # matrix scales as [robots, grid_cells] and can exceed a gigabyte
        # for large PPO batches. Chunk the affected rows so peak memory is
        # bounded while preserving the same nearest-clear-cell argmin.
        if batch_n <= 64:
            # Scenario rollouts plan only six robots. Compacting blocked rows
            # with nonzero forces a device-to-host sync on every replan; the
            # dense six-row projection is small and keeps the decision on GPU.
            row_goal_distance=(
                (self.xx.reshape(1,-1)-goal[:,0,None]).square()+
                (self.yy.reshape(1,-1)-goal[:,1,None]).square())
            row_goal_distance.masked_fill_(blocked.flatten(1),float("inf"))
            nearest=row_goal_distance.argmin(-1)
            projected=torch.stack((self.x[nearest//self.ny],
                                   self.y[nearest%self.ny]),-1)
            goal=torch.where(blocked_goal[:,None],projected,goal)
            gx,gy=self._indices(goal)
        else:
            blocked_rows=torch.nonzero(blocked_goal,as_tuple=False).flatten()
            if blocked_rows.numel():
                goal=goal.clone()
                projection_chunk=8192
                for offset in range(0,blocked_rows.numel(),projection_chunk):
                    rows=blocked_rows[offset:offset+projection_chunk]
                    row_goal=goal[rows]
                    row_blocked=blocked.index_select(0,rows).flatten(1)
                    goal_distance=((self.xx.reshape(1,-1)-row_goal[:,0,None]).square()+
                                   (self.yy.reshape(1,-1)-row_goal[:,1,None]).square())
                    nearest=goal_distance.masked_fill(row_blocked,float("inf")).argmin(-1)
                    nearest_x=nearest//self.ny
                    nearest_y=nearest%self.ny
                    projected=torch.stack((self.x[nearest_x],self.y[nearest_y]),-1)
                    goal[rows]=projected
                gx,gy=self._indices(goal)
        blocked[batch,sx,sy]=False
        blocked[batch,gx,gy]=False
        # Both the HIP kernels and the Torch fallback broadcast this one
        # field-static grid across worlds without constructing [B, nx, ny].
        bump_cost = self._bump_cost
        if not fused_route_supported:
            inf = torch.full((batch_n, self.nx, self.ny), 1.e6,
                             device=self.device, dtype=self.dtype)
            value = torch.where(blocked, inf, inf.clone())
            value[batch, gx, gy] = torch.zeros_like(gx, dtype=self.dtype)
        # Jacobi Bellman sweeps are parallel across both cells and worlds.
        import torch.nn.functional as F
        if fused_route_supported:
            speed_profile_limit = (speed.reshape(batch_n) if speed is not None else
                                   torch.full((batch_n,), 4.8, device=self.device, dtype=self.dtype))
            friction_profile = (lateral_friction.reshape(batch_n) if lateral_friction is not None else
                                torch.full((batch_n,), 1.2, device=self.device, dtype=self.dtype))
            acceleration_profile = (acceleration.reshape(batch_n) if acceleration is not None else
                                    torch.full((batch_n,), 8., device=self.device, dtype=self.dtype))
            # The fused kernel can write the converged potential directly to
            # its persistent cache for the full-batch path. This avoids a
            # dense temporary plus a full cache copy on every replan.
            out_value = (self.potential if full_batch else
                         torch.empty((batch_n,self.nx,self.ny),
                                     device=self.device,dtype=self.dtype))
            out_path = torch.empty((batch_n, 72, 2), device=self.device, dtype=self.dtype)
            out_lengths = torch.empty((batch_n,), device=self.device, dtype=torch.long)
            out_profile = torch.empty((batch_n, 72), device=self.device, dtype=self.dtype)
            out_goal = torch.empty((batch_n, 2), device=self.device, dtype=self.dtype)
            checkpoints = torch.empty((batch_n,), device=self.device, dtype=torch.int32)
            route_active = (self._all_active if full_batch else
                            active_mask if masked_full_batch else
                            torch.ones((batch_n,), device=self.device, dtype=torch.bool))
            hip_extension.fused_route(
                blocked.contiguous(), bump_cost.contiguous(), route_active,
                start.contiguous(),
                goal.contiguous(),
                sx.contiguous(), sy.contiguous(), gx.contiguous(), gy.contiguous(),
                speed_profile_limit.contiguous(), friction_profile.contiguous(),
                acceleration_profile.contiguous(), out_value, out_path, out_lengths,
                out_profile, out_goal, checkpoints, self.steer_rate_limit,
                self.sweeps, self.early_convergence,
                self.avoid_bumps, self.resolution)
            # The fused kernel cleanup zeroes unused path slots, while the
            # shared Torch shortcutter expects each row to repeat its endpoint
            # through max_points. Pad active rows before resampling; otherwise
            # the zero tail creates a fictitious return through the origin and
            # rejects otherwise valid routes.
            tail=torch.arange(72,device=self.device)[None] >= out_lengths[:,None]
            endpoint=out_path.gather(
                1,(out_lengths-1).clamp_min(0)[:,None,None].expand(-1,1,2))
            out_path=torch.where(tail[:,:,None],endpoint,out_path)
            out_path,out_lengths=self._shortcut_path(
                out_path,out_lengths,blocked,heading,length,width,speed,acceleration)
            out_profile=self._speed_profile(
                out_path,out_lengths,speed,lateral_friction,acceleration)
            if out_value is not self.potential:
                write_rows(self.potential, out_value)
            write_rows(self.last_path, out_path)
            write_rows(self.last_progress, torch.zeros_like(self.last_progress[:batch_n]))
            write_rows(self.last_lengths, out_lengths)
            write_rows(self.last_intercept, intercept)
            write_rows(self.last_intercept_time, t)
            write_rows(self.last_start, start)
            write_rows(self.last_goal, out_goal)
            write_rows(self.last_heading, heading)
            write_rows(self.last_speed_profile, out_profile)
            return self.last_path, self.last_lengths, self.last_intercept, self.last_intercept_time
        if (hip_extension is not None and value.dtype == torch.float32 and
                self.nx * self.ny <= 4096):
            # One workgroup owns one world and uses an on-chip Jacobi buffer;
            # block barriers preserve each full-grid sweep while fusing all
            # Bellman iterations into a single launch.
            value_next = torch.empty_like(value)
            hip_extension.bellman_sweep(value, blocked, bump_cost, value_next,
                                        self.sweeps, self.early_convergence)
            value = value_next
        else:
            blocked_neighbors=F.unfold(
                F.pad(blocked[:,None].to(self.dtype),(1,1,1,1),value=1.),3
            ).view(batch_n,9,self.nx*self.ny)>.5
            corner_clear=torch.ones_like(blocked_neighbors)
            corner_clear[:,0]=~(blocked_neighbors[:,1]|blocked_neighbors[:,3])
            corner_clear[:,2]=~(blocked_neighbors[:,1]|blocked_neighbors[:,5])
            corner_clear[:,6]=~(blocked_neighbors[:,3]|blocked_neighbors[:,7])
            corner_clear[:,8]=~(blocked_neighbors[:,5]|blocked_neighbors[:,7])
            for _ in range(self.sweeps):
                padded=F.pad(value[:,None],(1,1,1,1),value=1.e6)
                neighbors=F.unfold(padded,3).view(batch_n,9,self.nx*self.ny)
                candidate=neighbors+self._edge_cost*bump_cost.reshape(1,1,-1)
                candidate=candidate.masked_fill(~corner_clear,float("inf"))
                value=torch.minimum(value.flatten(1),candidate.amin(1)).view(batch_n,self.nx,self.ny)
                value=torch.where(blocked,inf,value)
        path = torch.zeros((batch_n, self.max_points, 2), device=self.device, dtype=self.dtype)
        path[:, 0] = start
        px, py = sx.clone(), sy.clone()
        active = torch.ones(batch_n, device=self.device, dtype=torch.bool)
        lengths = torch.ones(batch_n, device=self.device, dtype=torch.long)
        rows = batch
        for k in range(1, self.max_points):
            dx,dy=self._offsets[:,0],self._offsets[:,1]
            nx=px[:,None]+dx[None,:]
            ny=py[:,None]+dy[None,:]
            valid=(nx>=0)&(nx<self.nx)&(ny>=0)&(ny<self.ny)
            vx,vy=nx.clamp(0,self.nx-1),ny.clamp(0,self.ny-1)
            diagonal_blocked=(blocked[rows[:,None],vx,py[:,None]]|
                              blocked[rows[:,None],px[:,None],vy])
            valid &= ~(diagonal_blocked & self._diagonal_offsets[None,:])
            neighbor_blocked=blocked[rows[:,None],vx,vy]
            scores=torch.where(valid & ~neighbor_blocked,
                               value[rows[:,None],vx,vy],inf[:,0,0,None])
            choice=scores.argmin(-1)
            next_score=scores.gather(1,choice[:,None]).squeeze(1)
            off=self._offsets[choice]
            nx,ny=px+off[:,0],py+off[:,1]
            progressing=active & (next_score < value[rows,px,py] - 1e-5)
            px=torch.where(progressing,nx,px); py=torch.where(progressing,ny,py)
            point=torch.stack(((px.to(self.dtype)+.5)*self.resolution,
                               (py.to(self.dtype)+.5)*self.resolution),-1)
            path[:,k]=torch.where(progressing[:,None],point,path[:,k-1])
            lengths += progressing.long()
            reached=(px==gx)&(py==gy)
            active &= progressing & ~reached
            # Route extraction is sequential, but once every world is done,
            # remaining iterations are no-ops. Check in chunks to avoid a
            # device synchronization on each path point.
            if k % 16 == 0 and not bool(active.any().item()):
                break
        write_rows(self.potential, value)
        path,lengths=self._shortcut_path(
            path,lengths,blocked,heading,length,width,speed,acceleration)
        write_rows(self.last_path, path)
        write_rows(self.last_progress, torch.zeros_like(self.last_progress[:batch_n]))
        write_rows(self.last_lengths, lengths)
        write_rows(self.last_intercept, intercept)
        write_rows(self.last_intercept_time, t)
        write_rows(self.last_start, start)
        write_rows(self.last_goal, goal)
        write_rows(self.last_heading, heading)
        profile=self._speed_profile(path,lengths,speed,lateral_friction,acceleration)
        write_rows(self.last_speed_profile, profile)
        return self.last_path,self.last_lengths,self.last_intercept,self.last_intercept_time

    def path_reference(self, position, velocity, speed_limit, *, lookahead=None):
        """Batched progress projection, lookahead, and speed-aware path command."""
        path=self.last_path
        hip_extension = (_hip_path_reference_extension() if lookahead is None and _HIP_PATH_REFERENCE_ENABLED and
                         self.device.type == "cuda" else None)
        if (hip_extension is not None and self.max_points == 72 and
                path.dtype == torch.float32 and
                path.is_contiguous() and position.dtype == torch.float32 and
                velocity.dtype == torch.float32 and speed_limit.dtype == torch.float32 and
                self.last_goal.is_contiguous() and self.last_speed_profile.is_contiguous() and
                self.last_progress.is_contiguous()):
            return hip_extension.path_reference(
                path, self.last_lengths, self.last_goal, self.last_speed_profile,
                position, velocity, speed_limit, self.last_progress)
        batch=torch.arange(self.n,device=self.device)
        segments=path[:,1:]-path[:,:-1]
        distance=segments.norm(dim=-1)
        station=torch.cat((torch.zeros((self.n,1),device=self.device,dtype=path.dtype),
                           distance.cumsum(-1)),-1)
        segment_index=torch.arange(self.max_points-1,device=self.device)[None,:]
        valid=segment_index<(self.last_lengths-1)[:,None]
        fraction=(((position[:,None,:]-path[:,:-1])*segments).sum(-1)/
                  distance.square().clamp_min(1.e-12)).clamp(0.,1.)
        projection=path[:,:-1]+fraction[...,None]*segments
        squared=(projection-position[:,None,:]).square().sum(-1)
        eligible=valid & (station[:,1:]>= (self.last_progress-.08).clamp_min(0.)[:,None])
        squared=squared.masked_fill(~eligible,float("inf"))
        idx=squared.argmin(-1)
        chosen_fraction=fraction[batch,idx]
        chosen_length=distance[batch,idx]
        progress=station[batch,idx]+chosen_fraction*chosen_length
        progress=torch.maximum(progress,self.last_progress).minimum(station[batch,-1])
        projection=path[batch,idx]+chosen_fraction[:,None]*segments[batch,idx]
        self.last_progress.copy_(progress)
        lookahead=((.7+.12*speed_limit.clamp(max=4.5)).clamp_min(.12)
                   if lookahead is None else
                   torch.full_like(speed_limit,float(lookahead)))
        target_station=(progress+lookahead).minimum(station[batch,-1])
        target_segment=(station[:,1:]<target_station[:,None]).sum(-1).clamp_max(self.max_points-2)
        target_distance=distance[batch,target_segment]
        target_fraction=((target_station-station[batch,target_segment])/
                         target_distance.clamp_min(1.e-8)).clamp(0.,1.)
        target=path[batch,target_segment]+target_fraction[:,None]*segments[batch,target_segment]
        tangent=segments[batch,target_segment]/target_distance[:,None].clamp_min(1.e-6)
        normal=torch.stack((-tangent[:,1],tangent[:,0]),-1)
        cross=((position-projection)*normal).sum(-1)
        cross_v=(velocity*normal).sum(-1)
        lateral=(-3.*cross-4.*cross_v).clamp(-.75,.75)
        target_next=(target_segment+1).clamp_max(self.max_points-1)
        planned=(self.last_speed_profile[batch,target_segment]*(1.-target_fraction)+
                 self.last_speed_profile[batch,target_next]*target_fraction)
        planned=planned.clamp(max=speed_limit).clamp_min(0.)
        current=velocity.norm(dim=-1)
        target=(planned-.55*(current-planned).clamp_min(0)).clamp_min(0)
        command=tangent*target[:,None]+normal*lateral[:,None]
        # A one-point route means no collision-free step was found. Driving
        # directly to the goal here bypasses every field and robot obstacle.
        command=torch.where((self.last_lengths>1)[:,None],command,
                            torch.zeros_like(command))
        tangent=torch.where((self.last_lengths>1)[:,None],tangent,
                            torch.zeros_like(tangent))
        return command, tangent

    def defender_spawn(self, starts, mask, generator):
        """Choose a randomized point near the computed clear route on-device."""
        distance=(self.last_path-starts[:,None,:]).norm(dim=-1)
        valid=(torch.arange(self.max_points,device=self.device)[None,:] < self.last_lengths[:,None])
        eligible=valid&(distance>=1.25)&(distance<=2.75)
        # Random ranks among eligible points, with an ahead-of-start fallback.
        weights=torch.zeros(eligible.shape,device=self.device)
        count=int(mask.sum().item())
        if count:
            weights[mask]=torch.rand((count,self.max_points),device=self.device,generator=generator)
        weights=weights.masked_fill(~eligible,-1.)
        index=weights.argmax(-1)
        fallback=(valid&(distance>=1.1)).float().argmax(-1)
        index=torch.where(eligible.any(-1),index,fallback)
        point=self.last_path[torch.arange(self.n,device=self.device),index]
        jitter=torch.zeros((self.n,2),device=self.device)
        if count:
            jitter[mask]=torch.rand((count,2),device=self.device,generator=generator)-.5
        point=torch.where(mask[:,None],point+jitter*.18,starts)
        return point
