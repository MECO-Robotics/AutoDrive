"""Sleep and wake whole connected fuel contact groups, including supported stacks."""
from pathlib import Path
import os
import torch

_EXTENSION = None


def _extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    from torch.utils import cpp_extension
    from .fuel_physics_hip import _extension as load_physics
    if load_physics() is None:
        raise RuntimeError("HIP island sleeping requires the fuel contact extension")
    rocm = Path(os.environ.get("ROCM_HOME", "/tmp/autodrive-rocm-sdk"))
    old = cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME
    cpp_extension.TORCH_LIB_PATH = "/tmp/autodrive-torch-lib" if Path("/tmp/autodrive-torch-lib").exists() else old[0]
    cpp_extension.ROCM_HOME = str(rocm)
    cpp_extension.HIP_HOME = str(rocm / "hip")
    src = Path(__file__).parent / "csrc"
    build_options = {}
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        build_directory = Path(__file__).resolve().parent.parent / "outputs/fuel_physics/extensions/island"
        build_directory.mkdir(parents=True, exist_ok=True)
        build_options["build_directory"] = str(build_directory)
    flags = ["-O3", "-ffp-contract=off"]
    for path in (rocm / "lib/llvm/amdgcn/bitcode", rocm / "amdgcn/bitcode"):
        if path.is_dir():
            flags.append(f"--rocm-device-lib-path={path}")
            break
    try:
        _EXTENSION = cpp_extension.load(
            name="autodrive_fuel_sleep_hip", sources=[str(src / "fuel_sleep_hip.cpp"), str(src / "fuel_sleep_hip_kernel.cu")],
            with_cuda=True, extra_cflags=["-O3"], extra_cuda_cflags=flags, verbose=False, **build_options)
    finally:
        cpp_extension.TORCH_LIB_PATH, cpp_extension.ROCM_HOME, cpp_extension.HIP_HOME = old
    return _EXTENSION


def initialize_island_sleep(physics):
    """Allocate fixed buffers and load native code before graph capture."""
    kw = dict(device=physics.device)
    shape = (physics.n, physics.p)
    physics._island_sleep = dict(
        clock=torch.zeros(shape, dtype=physics.pos.dtype, **kw),
        parent=torch.empty(shape, dtype=torch.int32, **kw),
        quiet=torch.empty(shape, dtype=torch.int32, **kw),
        supported=torch.empty(shape, dtype=torch.int32, **kw),
        minimum_clock=torch.empty(shape, dtype=physics.pos.dtype, **kw),
    )
    if physics._hip is not None:
        _extension()


def reset_island_sleep(physics, mask=None):
    state = getattr(physics, "_island_sleep", None)
    if state is None:
        return
    if mask is None:
        state["clock"].zero_()
    else:
        state["clock"][mask] = 0


def _torch_update(physics, free, dt):
    c, sim = physics.config, physics.sim
    state = physics._island_sleep
    physics._build_pairs(free)
    xyz = physics.pos.detach().cpu()
    enabled = free.detach().cpu()
    parent = list(range(physics.n * physics.p))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for w, i, j in physics.candidate_pairs.detach().cpu().tolist():
        if (xyz[w, i] - xyz[w, j]).square().sum() <= (2 * c.radius) ** 2:
            a, b = root(w * physics.p + i), root(w * physics.p + j)
            parent[max(a, b)] = min(a, b)
    labels = torch.tensor([root(i) for i in range(len(parent))], device=physics.device).reshape_as(free)
    velocity = physics.vel.clone()
    velocity[..., 2] -= (~physics.sleeping) * c.gravity * dt
    quiet = (velocity.norm(dim=-1) <= c.sleep_speed) & (physics.angular.norm(dim=-1) <= c.sleep_angular_speed)
    for robot in range(sim.pose.shape[1]):
        delta = physics.pos[..., :2] - sim.pose[:, robot, None, :2]
        theta = sim.pose[:, robot, 2:3]
        co, si = theta.cos(), theta.sin()
        local = torch.stack((co * delta[..., 0] + si * delta[..., 1], -si * delta[..., 0] + co * delta[..., 1]), -1)
        half = torch.stack((sim.length[:, robot], sim.width[:, robot]), -1)[:, None] * .5
        _, separation = physics._box_normal(local, half)
        quiet &= ~((separation <= c.radius + .002) & (physics.pos[..., 2] < c.robot_height + c.radius))
    clock = state["clock"]
    height = getattr(physics, "_support_height", torch.zeros_like(physics.pos[..., 2]))
    for w in range(physics.n):
        for label in labels[w, enabled[w]].unique().tolist():
            members = free[w] & (labels[w] == label)
            supported = bool((physics.pos[w, members, 2] <= height[w, members] + c.radius + .005).any())
            stable = supported and bool(quiet[w, members].all())
            elapsed = float(torch.minimum(clock[w, members], physics.sleep_clock[w, members]).min()) + dt if stable else 0.
            asleep = stable and elapsed >= c.sleep_time
            clock[w, members] = elapsed
            physics.sleep_clock[w, members] = elapsed
            physics.sleeping[w, members] = asleep
            if asleep:
                physics.vel[w, members] = 0
                physics.angular[w, members] = 0
    state["parent"].copy_((labels % physics.p).to(torch.int32))


def update_island_sleep(physics, free, dt):
    """Enforce one sleeping decision per actual contact component each substep."""
    if not physics.config.sleep:
        return
    if not hasattr(physics, "_island_sleep"):
        initialize_island_sleep(physics)
    if physics._hip is None:
        _torch_update(physics, free, dt)
        return
    state, hip, sim, c = physics._island_sleep, physics._hip, physics.sim, physics.config
    tensors = [physics.pos, physics.vel, physics.angular, physics.sleeping, state["clock"],
               state["parent"], state["quiet"], state["supported"], state["minimum_clock"], free.contiguous(),
               sim.pose, sim.velocity, sim.length, sim.width, hip.list, hip.counts, hip.overflow, physics.sleep_clock, physics._support_height]
    params = [c.radius, c.gravity, float(dt), c.sleep_time, c.sleep_speed, c.sleep_angular_speed, c.robot_height]
    dims = [physics.n, physics.p, sim.pose.shape[1], c.max_neighbors, int(c.broadphase == "all_pairs")]
    _extension().update(tensors, params, dims)
