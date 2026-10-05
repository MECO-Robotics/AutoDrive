"""Batched device-resident grid planner for tensor simulation rollouts.

The planner uses parallel Bellman sweeps on a clearance-inflated 8-connected
grid, then follows the resulting potential with batched greedy extraction.
It intentionally keeps all search state on the selected torch device.
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
        self.resolution = float(resolution)
        self.nx = int(math.ceil(env.sim.field_length / resolution))
        self.ny = int(math.ceil(env.sim.field_width / resolution))
        self.max_points = int(max_points)
        self.sweeps = int(sweeps)
        self.early_convergence = bool(early_convergence)
        self.avoid_bumps = bool(avoid_bumps)
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
                grid_clearance = 0. if self.avoid_bumps else self.resolution * .72
                return extension.blocked_grid(
                    heading.contiguous(), length.contiguous(), width.contiguous(),
                    self.x, self.y, self._boxes, self._bumps, self._extra_circles,
                    dynamic_tensor.contiguous(), self.length, self.width,
                    grid_clearance, self.avoid_bumps)
        c, s = heading.cos().abs(), heading.sin().abs()
        # Narrow bump-to-wall lanes can be one grid row wide. The robot
        # footprint itself remains inflated; avoid an extra cell pad there.
        grid_clearance=0. if self.avoid_bumps else self.resolution*.72
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
            sat_f = length[:, None, None, None] * .5 + self._boxes[None, :, 2, None, None] * heading.cos().abs()[:, None, None, None] + self._boxes[None, :, 3, None, None] * heading.sin().abs()[:, None, None, None]
            sat_l = width[:, None, None, None] * .5 + self._boxes[None, :, 2, None, None] * heading.sin().abs()[:, None, None, None] + self._boxes[None, :, 3, None, None] * heading.cos().abs()[:, None, None, None]
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
            moving_radius=.5*torch.sqrt(length[offset:end].square()+width[offset:end].square())
            clearance=moving_radius[:,None]+radii.clamp_min(0.)
            blocked[offset:end] |= (((dx.square()+dy.square()) <=
                clearance[...,None,None].square()) &
                (radii[...,None,None] > 0.)).any(dim=1)
        return blocked

    def _indices(self, points):
        ix = (points[:, 0] / self.resolution).floor().long().clamp(0, self.nx - 1)
        iy = (points[:, 1] / self.resolution).floor().long().clamp(0, self.ny - 1)
        return ix, iy

    def _shortcut_path(self, path, lengths, blocked):
        """Greedily remove grid turns when the straight segment stays clear."""
        if path.device.type != "cpu":
            return path, lengths
        for row in range(path.shape[0]):
            count=int(lengths[row])
            if count < 3:
                continue
            source=path[row,:count].clone()
            result=[source[0]]
            anchor=0
            while anchor < count-1:
                chosen=anchor+1
                for candidate in range(count-1,anchor+1,-1):
                    a,b=source[anchor],source[candidate]
                    steps=max(1,int(torch.maximum((b[0]-a[0]).abs(),
                        (b[1]-a[1]).abs()).item()/(self.resolution*.5))+1)
                    t=torch.arange(1,steps,device=path.device,dtype=path.dtype)/steps
                    points=a[None]+(b-a)[None]*t[:,None]
                    ix,iy=self._indices(points)
                    if not bool(blocked[row,ix,iy].any()):
                        chosen=candidate
                        break
                result.append(source[chosen])
                anchor=chosen
            new_count=len(result)
            path[row,0]=result[0]
            path[row,1:new_count]=torch.stack(result[1:])
            path[row,new_count:]=result[-1]
            lengths[row]=new_count
        return path,lengths

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
             robot_obstacles=None):
        """Plan selected worlds and return the persistent batched route state."""
        full_batch = active_mask is None
        selected = None
        if full_batch:
            batch_n = self.n
        else:
            active_mask = torch.as_tensor(active_mask, device=self.device,
                                          dtype=torch.bool).reshape(self.n)
            selected = torch.nonzero(active_mask, as_tuple=False).flatten()
            batch_n = selected.numel()
        if batch_n == 0:
            return self.last_path,self.last_lengths,self.last_intercept,self.last_intercept_time
        def select_rows(value):
            if value is None:
                return None
            return value if full_batch else value[selected]

        def write_rows(destination, values):
            if full_batch:
                destination.copy_(values)
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
        # Do not let conservative inflation trap a route's source. AD* defender
        # targets can land on a bump while projecting an interception point;
        # move those endpoints to the nearest reachable clearance cell.
        sx, sy = self._indices(start); gx, gy = self._indices(goal)
        batch = torch.arange(batch_n,device=self.device)
        if self.avoid_bumps and not fused_route_supported:
            blocked_goal=blocked[batch,gx,gy]
            # Only blocked goals need projection. The dense all-row distance
            # matrix scales as [robots, grid_cells] and can exceed a gigabyte
            # for large PPO batches. Chunk the affected rows so peak memory is
            # bounded while preserving the same nearest-clear-cell argmin.
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
        if not fused_route_supported:
            blocked[batch, sx, sy] = False
            blocked[batch, gx, gy] = False
        # Both the HIP kernels and the Torch fallback broadcast this one
        # field-static grid across worlds without constructing [B, nx, ny].
        bump_cost = self._bump_cost
        inf = torch.full((batch_n, self.nx, self.ny), 1.e6,
                         device=self.device, dtype=self.dtype)
        value = torch.where(blocked, inf, inf.clone())
        value[batch, gx, gy] = torch.zeros_like(gx, dtype=self.dtype)
        # Jacobi Bellman sweeps are parallel across both cells and worlds.
        import torch.nn.functional as F
        if fused_route_supported:
            speed_profile_limit = (speed.reshape(batch_n) if speed is not None else
                                   torch.full((batch_n,), 4.5, device=self.device, dtype=self.dtype))
            friction_profile = (lateral_friction.reshape(batch_n) if lateral_friction is not None else
                                torch.full((batch_n,), 1.2, device=self.device, dtype=self.dtype))
            acceleration_profile = (acceleration.reshape(batch_n) if acceleration is not None else
                                    torch.full((batch_n,), 8., device=self.device, dtype=self.dtype))
            # The fused kernel can write the converged potential directly to
            # its persistent cache for the full-batch path. This avoids a
            # dense temporary plus a full cache copy on every replan.
            out_value = self.potential if full_batch else torch.empty_like(value)
            out_path = torch.empty((batch_n, 72, 2), device=self.device, dtype=self.dtype)
            out_lengths = torch.empty((batch_n,), device=self.device, dtype=torch.long)
            out_profile = torch.empty((batch_n, 72), device=self.device, dtype=self.dtype)
            out_goal = torch.empty((batch_n, 2), device=self.device, dtype=self.dtype)
            checkpoints = torch.empty((batch_n,), device=self.device, dtype=torch.int32)
            hip_extension.fused_route(
                value, blocked.contiguous(), bump_cost.contiguous(), start.contiguous(),
                goal.contiguous(),
                sx.contiguous(), sy.contiguous(), gx.contiguous(), gy.contiguous(),
                speed_profile_limit.contiguous(), friction_profile.contiguous(),
                acceleration_profile.contiguous(), out_value, out_path, out_lengths,
                out_profile, out_goal, checkpoints, self.steer_rate_limit,
                self.sweeps, self.early_convergence,
                self.avoid_bumps, self.resolution)
            if out_value is not self.potential:
                write_rows(self.potential, out_value)
            write_rows(self.last_path, out_path)
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
        path,lengths=self._shortcut_path(path,lengths,blocked)
        write_rows(self.last_path, path)
        write_rows(self.last_lengths, lengths)
        write_rows(self.last_intercept, intercept)
        write_rows(self.last_intercept_time, t)
        write_rows(self.last_start, start)
        write_rows(self.last_goal, goal)
        write_rows(self.last_heading, heading)
        # Curvature caps and backward braking pass are computed as one batched
        # pairwise tensor operation. The resulting profile respects swerve
        # steering rate, tire lateral acceleration, and stopping distance.
        delta=path[:,1:]-path[:,:-1]
        distance=delta.norm(dim=-1)
        direction=delta/distance[...,None].clamp_min(1.e-6)
        cross=direction[:,:-1,0]*direction[:,1:,1]-direction[:,:-1,1]*direction[:,1:,0]
        dot=(direction[:,:-1]*direction[:,1:]).sum(-1)
        turn=torch.atan2(cross,dot).abs()
        arc=.5*(distance[:,:-1]+distance[:,1:]).clamp_min(1.e-4)
        curvature=turn/arc
        point_curve=torch.zeros((batch_n,self.max_points),device=self.device,dtype=self.dtype)
        point_curve[:,1:-1]=curvature
        mu=(lateral_friction if lateral_friction is not None else torch.full_like(length,1.2)).clamp_min(.1)
        accel=(acceleration if acceleration is not None else torch.full_like(length,8.)).clamp_min(.2)
        vcap=(speed if speed is not None else torch.full_like(length,4.5)).clamp_min(.1)[:,None]
        lateral=(.65*mu[:,None]*9.81/point_curve.clamp_min(1.e-6)).sqrt()
        steer=self.steer_rate_limit/point_curve.clamp_min(1.e-6)
        caps=torch.minimum(vcap,torch.minimum(lateral,steer))
        caps=torch.where(point_curve>1.e-6,caps,vcap)
        caps[batch,lengths-1]=0.
        station=torch.cat((torch.zeros((batch_n,1),device=self.device,dtype=self.dtype),distance.cumsum(-1)),-1)
        delta_station=station[:,None,:]-station[:,:,None]
        future=torch.arange(self.max_points,device=self.device)[None,None,:]>=torch.arange(self.max_points,device=self.device)[None,:,None]
        valid=(torch.arange(self.max_points,device=self.device)[None,None,:]<lengths[:,None,None])&future
        braking=.65*accel[:,None,None]
        reachable=(caps[:,None,:].square()+2*braking*delta_station.clamp_min(0)).sqrt()
        profile=torch.where(valid,reachable,torch.full_like(reachable,float("inf"))).amin(-1)
        write_rows(self.last_speed_profile, profile)
        return self.last_path,self.last_lengths,self.last_intercept,self.last_intercept_time

    def path_reference(self, position, velocity, speed_limit):
        """Batched progress projection, lookahead, and speed-aware path command."""
        path=self.last_path
        hip_extension = (_hip_path_reference_extension() if _HIP_PATH_REFERENCE_ENABLED and
                         self.device.type == "cuda" else None)
        if (hip_extension is not None and self.max_points == 72 and
                path.dtype == torch.float32 and
                path.is_contiguous() and position.dtype == torch.float32 and
                velocity.dtype == torch.float32 and speed_limit.dtype == torch.float32 and
                self.last_goal.is_contiguous() and self.last_speed_profile.is_contiguous()):
            return hip_extension.path_reference(
                path, self.last_lengths, self.last_goal, self.last_speed_profile,
                position, velocity, speed_limit)
        d2=(path-position[:,None,:]).square().sum(-1)
        idx=d2.argmin(-1)
        next_idx=(idx+1).clamp_max(self.max_points-1)
        batch=torch.arange(self.n,device=self.device)
        here=path[batch,idx]
        segments=(path[:,1:]-path[:,:-1]).norm(dim=-1)
        station=torch.cat((torch.zeros((self.n,1),device=self.device,dtype=path.dtype),
                           segments.cumsum(-1)),-1)
        lookahead=.7+.12*speed_limit.clamp(max=4.5)
        target_station=station[batch,idx]+lookahead
        path_index=torch.arange(self.max_points,device=self.device)[None]
        target_valid=((path_index>=idx[:,None]) &
                      (path_index<self.last_lengths[:,None]) &
                      (station>=target_station[:,None]))
        target_idx=torch.where(target_valid,path_index,self.max_points).amin(-1)
        target_idx=torch.where(target_idx==self.max_points,
                               (self.last_lengths-1).clamp_min(0),target_idx)
        target=path[batch,target_idx]
        # Pure-pursuit lookahead follows a curve through grid corners rather
        # than switching direction abruptly at each 8-connected waypoint.
        tangent=target-position
        tangent=tangent/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        normal=torch.stack((-tangent[:,1],tangent[:,0]),-1)
        cross=((position-here)*normal).sum(-1)
        cross_v=(velocity*normal).sum(-1)
        lateral=(-3.*cross-4.*cross_v).clamp(-.75,.75)
        planned=self.last_speed_profile[batch,idx].clamp(max=speed_limit).clamp_min(.2)
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
