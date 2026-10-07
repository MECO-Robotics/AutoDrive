"""Tensor-resident swerve and rigid-body physics engine."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any
import warnings

from .field import (BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE,
                    BUMP_ROBOT_CG_HEIGHT, BUMP_ROLLING_RESISTANCE)
from .tensor_physics_collision import (
    FUSED_COLLISION_HIP_ENABLED, FUSED_CONTACT_PIPELINE_HIP_ENABLED,
    TensorPhysicsCollisionMixin, _contact_pipeline_hip,
)

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None


def _require_torch():
    if torch is None:
        raise RuntimeError("Tensor simulation requires PyTorch; install torch first")


def swerve_heading_rate(error, omega_limit, angular_acceleration,
                        current_rate=None, control_dt=None):
    """Return a yaw-rate target with stopping-distance and rate feedback."""
    if control_dt is not None:
        # Coarse playback steps cannot safely track the drivetrain's full yaw
        # rate. Limit per-step heading change while leaving 50 Hz control alone.
        omega_limit=torch.minimum(omega_limit,torch.full_like(omega_limit,
                                      .6/max(float(control_dt),1e-6)))
    max_rate=torch.sqrt((2.*angular_acceleration*error.abs()).clamp_min(0.))
    desired=error.sign()*torch.minimum(omega_limit,max_rate)
    if current_rate is None:
        return desired
    # The stopping envelope alone assumes zero angular speed. Feed back the
    # measured rate so inertia brakes the chassis before it crosses the target.
    return (desired-1.5*(current_rate-desired)).clamp(-omega_limit,omega_limit)



@dataclass(frozen=True)
class TensorState:
    pose: Any
    velocity: Any



@dataclass(frozen=True)
class TensorSwerveParameters:
    wheel_radius: float = .0508
    drive_ratio: float = 6.75
    drive_current_limit: float = 60.
    drive_supply_current_limit: float = 80.
    robot_supply_current_limit: float = 180.
    steer_current_limit: float = 20.
    steer_rate_limit: float = 12.
    steer_acceleration: float = 80.
    motor_efficiency: float = .9
    battery_voltage: float = 12.6
    battery_internal_resistance: float = .018
    stall_torque_nm: float = 7.09
    stall_current_a: float = 366.
    free_speed_rpm: float = 6000.
    free_current_a: float = 2.
    nominal_voltage: float = 12.
    module_x_offset: float = .36
    module_y_offset: float = .36



class TensorVectorizedSimulator(TensorPhysicsCollisionMixin):
    """Batched rigid-body simulator; every state tensor has leading world N.

    ``num_robots`` defaults to two for compatibility with the original
    offense-versus-defense environment. Larger batches use the same per-robot
    swerve, field, obstacle, and pairwise-contact models.
    """
    def __init__(self, num_envs=1, device="cuda", seed=0, dt=.02,
                 field_length=16.54, field_width=8.21, robot_length=.9,
                 robot_width=.9, max_speed=4.8, max_acceleration=8.,
                 max_omega=8., max_alpha=18., mass=55., bumper_friction=.65,
                 field_friction=.7, lateral_friction=1.2,
                 yaw_inertia_multiplier=1.5, current_limit=None, swerve=None,
                 num_robots=2, team_ids=None,
                 obstacles=(), field_colliders=(), bump_regions=(), contact_iterations=3, randomize=False,
                 field_sweep_spacing=.02, **kwargs):
        _require_torch()
        if num_envs < 1 or dt <= 0: raise ValueError("num_envs and dt must be positive")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested accelerator {device!r} is unavailable; refusing CPU simulation")
        self.n, self.device, self.dt = int(num_envs), requested, float(dt)
        self.num_robots = int(num_robots)
        if self.num_robots < 2:
            raise ValueError("num_robots must be at least two")
        default_teams=(list(range(self.num_robots)) if self.num_robots == 2 else
                       [0,0,0,1,1,1] if self.num_robots == 6 else
                       [i % 2 for i in range(self.num_robots)])
        configured_teams=(default_teams if team_ids is None else
                          torch.as_tensor(team_ids).detach().cpu().reshape(-1).tolist())
        self._team_ids_host=tuple(int(team) for team in configured_teams)
        self.team_ids=torch.as_tensor(default_teams if team_ids is None else team_ids,
            device=self.device,dtype=torch.long).reshape(-1)
        if self.team_ids.numel()!=self.num_robots:
            raise ValueError("team_ids must provide one alliance id per robot")
        pairs=torch.triu_indices(self.num_robots,self.num_robots,offset=1,device=self.device)
        self._collision_pair_i,self._collision_pair_j=pairs[0],pairs[1]
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed or 0))
        self.field_length, self.field_width = float(field_length), float(field_width)
        self.contact_iterations = max(1, int(contact_iterations))
        self.field_sweep_spacing = float(field_sweep_spacing)
        if not math.isfinite(self.field_sweep_spacing) or self.field_sweep_spacing <= 0:
            raise ValueError("field_sweep_spacing must be finite and positive")
        self._fused_robot_collision_multi_hip_used = False
        self._fused_field_collision_multi_hip_used = False
        self._fused_robot_collision_multi_hip_enabled = (
            self.num_robots == 6 and
            os.environ.get("AUTODRIVE_FUSED_ROBOT_COLLISION_6_HIP", "1") != "0" and
            self.device.type == "cuda" and torch.version.hip is not None
        )
        self._fused_field_collision_multi_hip_enabled = (
            self._fused_robot_collision_multi_hip_enabled and
            os.environ.get("AUTODRIVE_FUSED_FIELD_COLLISION_6_HIP", "1") != "0"
        )
        self.swerve = swerve or TensorSwerveParameters()
        self.randomize = bool(randomize)
        self._fused_wall_collision_hip_enabled = (
            FUSED_COLLISION_HIP_ENABLED and self.device.type == "cuda" and
            torch.version.hip is not None
        )
        self._fused_wall_collision_hip_used = False
        self._fused_contact_pipeline_hip_enabled = (
            FUSED_CONTACT_PIPELINE_HIP_ENABLED and self.device.type == "cuda" and
            torch.version.hip is not None
        )
        # Exact fused swerve + pose HIP is enabled by default on HIP after
        # exact integrated parity and repeated E2E validation. Set the
        # environment variable to 0 to force the Torch implementation.
        self._fused_swerve_pose_hip_enabled = (
            os.environ.get("AUTODRIVE_FUSED_SWERVE_POSE_HIP", "1") != "0" and
            self.device.type == "cuda" and torch.version.hip is not None
        )
        self._fused_swerve_pose_hip_failed = False
        # Compilation is opt-in because this kernel mutates simulator state
        # in place and compiler support varies across accelerator backends.
        # Keep construction eager; the first active step compiles lazily.
        self._torch_compile_swerve_enabled = (
            os.environ.get("AUTODRIVE_TORCH_COMPILE_SWERVE") == "1" and
            self.device.type == "cuda" and torch is not None and
            hasattr(torch, "compile")
        )
        self._torch_compiled_swerve = None
        self._torch_compile_swerve_failed = False
        self._torch_compile_step_enabled = (
            os.environ.get("AUTODRIVE_TORCH_COMPILE_SIM_STEP") == "1" and
            self.device.type == "cuda" and torch is not None and
            hasattr(torch, "compile")
        )
        self._torch_compiled_step = None
        self._torch_compile_step_failed = False
        self._base = dict(length=robot_length,width=robot_width,mass=mass,speed=max_speed,
                          accel=max_acceleration,omega=max_omega,alpha=max_alpha,
                          bumper_friction=bumper_friction,field_friction=field_friction,
                          lateral_friction=lateral_friction,
                          yaw_inertia_multiplier=yaw_inertia_multiplier,
                          drive_ratio=self.swerve.drive_ratio,
                          drive_current_limit=current_limit or self.swerve.drive_current_limit,
                          drive_supply_limit=self.swerve.drive_supply_current_limit,
                          robot_supply_limit=self.swerve.robot_supply_current_limit,
                          battery_resistance=self.swerve.battery_internal_resistance)
        def full(shape, val): return torch.full(shape, float(val), device=self.device)
        self.pose = torch.zeros((self.n,self.num_robots,3), device=self.device)
        self.velocity = torch.zeros_like(self.pose)
        robot_shape=(self.n,self.num_robots)
        self.length, self.width, self.mass = (full(robot_shape, v) for v in (robot_length,robot_width,mass))
        self.speed, self.accel, self.omega_limit, self.alpha = (full(robot_shape, v) for v in (max_speed,max_acceleration,max_omega,max_alpha))
        self.mu, self.ground_mu, self.wall_mu = (full((self.n,), v) for v in
                                                (bumper_friction, field_friction, field_friction))
        self.lateral_mu=full(robot_shape,lateral_friction)
        self.yaw_inertia_multiplier=full(robot_shape,yaw_inertia_multiplier)
        self.drive_ratio = full(robot_shape,self.swerve.drive_ratio)
        self.drive_current_limit = full(robot_shape,current_limit or self.swerve.drive_current_limit)
        self.drive_supply_limit=full(robot_shape,self.swerve.drive_supply_current_limit)
        self.robot_supply_limit=full(robot_shape,self.swerve.robot_supply_current_limit)
        self.battery_resistance=full(robot_shape,self.swerve.battery_internal_resistance)
        self.module_angle=torch.zeros((self.n,self.num_robots,4),device=self.device)
        self.module_steer_rate=torch.zeros_like(self.module_angle)
        self.module_drive_speed=torch.zeros_like(self.module_angle)
        self.module_current=torch.zeros_like(self.module_angle)
        self.module_supply_current=torch.zeros_like(self.module_angle)
        self.robot_current=torch.zeros(robot_shape,device=self.device)
        # Module locations are drivetrain constants; broadcast them in the
        # 50 Hz swerve step instead of rebuilding full-batch tensors per tick.
        self._module_x_offsets=torch.tensor(
            [-self.swerve.module_x_offset,self.swerve.module_x_offset,
             -self.swerve.module_x_offset,self.swerve.module_x_offset],
            device=self.device,dtype=self.pose.dtype).reshape(1,1,4)
        self._module_y_offsets=torch.tensor(
            [-self.swerve.module_y_offset,-self.swerve.module_y_offset,
             self.swerve.module_y_offset,self.swerve.module_y_offset],
            device=self.device,dtype=self.pose.dtype).reshape(1,1,4)
        self._module_radius=torch.sqrt(
            self._module_x_offsets.square()+self._module_y_offsets.square()).amax(-1)
        self.robot_contact=torch.zeros((self.n,),device=self.device,dtype=torch.bool)
        self.opponent_contact=torch.zeros((self.n,),device=self.device,dtype=torch.bool)
        self.field_contact=torch.zeros(robot_shape,device=self.device,dtype=torch.bool)
        self.wall_contact=torch.zeros((self.n,self.num_robots,2),device=self.device,dtype=torch.bool)
        obs=torch.as_tensor(obstacles,dtype=torch.float32,device=self.device)
        self.obstacles=obs.reshape(-1,3) if obs.numel() else torch.empty((0,3),device=self.device)
        boxes=torch.as_tensor(field_colliders,dtype=torch.float32,device=self.device)
        self.field_colliders=boxes.reshape(-1,4) if boxes.numel() else torch.empty((0,4),device=self.device)
        bumps=torch.as_tensor(bump_regions,dtype=torch.float32,device=self.device)
        self.bump_regions=bumps.reshape(-1,4) if bumps.numel() else torch.empty((0,4),device=self.device)
        self.reset()

    @property
    def state(self): return TensorState(self.pose,self.velocity)

    def reset(self, seed=None, pose=None, velocity=None):
        if seed is not None: self.generator.manual_seed(int(seed))
        self.pose.zero_(); self.velocity.zero_(); self.module_angle.zero_(); self.module_steer_rate.zero_()
        self.module_drive_speed.zero_(); self.module_current.zero_(); self.module_supply_current.zero_()
        self.robot_current.zero_(); self.robot_contact.zero_(); self.opponent_contact.zero_()
        self.field_contact.zero_(); self.wall_contact.zero_()
        if self.randomize: self._randomize(torch.ones((self.n,),device=self.device,dtype=torch.bool))
        else: self._randomize(torch.ones((self.n,),device=self.device,dtype=torch.bool), nominal=True)
        if pose is None:
            margin=max(.9, .9)  # keep placement independent of device-side tensor reads
            self.pose[:,:,0].uniform_(margin,self.field_length-margin,generator=self.generator)
            self.pose[:,:,1].uniform_(margin,self.field_width-margin,generator=self.generator)
            self.pose[:,:,2].uniform_(-math.pi,math.pi,generator=self.generator)
        else: self.pose.copy_(torch.as_tensor(pose,device=self.device,dtype=self.pose.dtype).expand_as(self.pose))
        if velocity is not None: self.velocity.copy_(torch.as_tensor(velocity,device=self.device,dtype=self.velocity.dtype).expand_as(self.velocity))
        self._field_collision()
        return self.state

    def reset_done(self, mask):
        """Reset selected worlds in place without disturbing the other worlds."""
        mask=torch.as_tensor(mask,device=self.device,dtype=torch.bool).reshape(self.n)
        if self.randomize: self._randomize(mask)
        count=int(mask.sum().item())
        x=torch.zeros((self.n,self.num_robots),device=self.device)
        y=torch.zeros_like(x); a=torch.zeros_like(x)
        if count:
            x[mask]=torch.rand((count,self.num_robots),device=self.device,generator=self.generator)
            y[mask]=torch.rand((count,self.num_robots),device=self.device,generator=self.generator)
            a[mask]=torch.rand((count,self.num_robots),device=self.device,generator=self.generator)
        margin=.9
        self.pose[:,:,0]=torch.where(mask[:,None],margin+x*(self.field_length-2*margin),self.pose[:,:,0])
        self.pose[:,:,1]=torch.where(mask[:,None],margin+y*(self.field_width-2*margin),self.pose[:,:,1])
        self.pose[:,:,2]=torch.where(mask[:,None],(2*a-1)*math.pi,self.pose[:,:,2])
        self._field_collision(mask)
        self.velocity=torch.where(mask[:,None,None],torch.zeros_like(self.velocity),self.velocity)
        self.module_angle=torch.where(mask[:,None,None],torch.zeros_like(self.module_angle),self.module_angle)
        self.module_steer_rate=torch.where(mask[:,None,None],torch.zeros_like(self.module_steer_rate),self.module_steer_rate)
        self.module_drive_speed=torch.where(mask[:,None,None],torch.zeros_like(self.module_drive_speed),self.module_drive_speed)
        self.module_current=torch.where(mask[:,None,None],torch.zeros_like(self.module_current),self.module_current)
        self.module_supply_current=torch.where(mask[:,None,None],torch.zeros_like(self.module_supply_current),self.module_supply_current)
        self.robot_current=torch.where(mask[:,None],torch.zeros_like(self.robot_current),self.robot_current)
        self.robot_contact &= ~mask
        self.opponent_contact &= ~mask
        self.field_contact &= ~mask[:,None]
        self.wall_contact &= ~mask[:,None,None]
        return self.state

    def _randomize(self, mask, nominal=False):
        """Sample or restore physical parameters using device-side tensors only."""
        def sample(base, shape, lo, hi):
            if nominal: return torch.full(shape,float(base),device=self.device)
            value=torch.full(shape,float(base),device=self.device)
            count=int(mask.sum().item())
            if count:
                sample_shape=(count,*shape[1:])
                value[mask]=float(base)*(lo+torch.rand(
                    sample_shape,device=self.device,generator=self.generator)*(hi-lo))
            return value
        pair=(self.n,self.num_robots)
        for name,base,lo,hi in (("length",self._base['length'],.82,1.2),("width",self._base['width'],.82,1.2),
                                ("mass",self._base['mass'],.65,1.45),("speed",self._base['speed'],.75,1.3),
                                ("accel",self._base['accel'],.65,1.4),("omega_limit",self._base['omega'],.7,1.3),
                                ("alpha",self._base['alpha'],.65,1.4),("drive_ratio",self._base['drive_ratio'],.9,1.1),
                                ("drive_current_limit",self._base['drive_current_limit'],.75,1.25),
                                ("drive_supply_limit",self._base['drive_supply_limit'],.75,1.25),
                                ("robot_supply_limit",self._base['robot_supply_limit'],.8,1.2),
                                ("battery_resistance",self._base['battery_resistance'],.7,1.6),
                                ("lateral_mu",self._base['lateral_friction'],.8,1.25),
                                ("yaw_inertia_multiplier",self._base['yaw_inertia_multiplier'],.8,1.2)):
            value=sample(base,pair,lo,hi); current=getattr(self,name)
            current.copy_(torch.where(mask[:,None],value,current))
        for name,base,lo,hi in (("mu",self._base['bumper_friction'],.25,1.1),
                                ("ground_mu",self._base['field_friction'],.35,1.2),
                                ("wall_mu",self._base['field_friction'],.25,1.1)):
            value=sample(base,(self.n,),lo,hi); current=getattr(self,name)
            current.copy_(torch.where(mask,value,current))
        torque_per_amp = self.swerve.stall_torque_nm / self.swerve.stall_current_a
        motor_accel = (4*torque_per_amp*self.drive_current_limit*self.drive_ratio*
                       self.swerve.motor_efficiency /
                       (self.swerve.wheel_radius*self.mass).clamp_min(1e-6))
        tire_accel = self.ground_mu[:, None]*9.81
        accel=torch.minimum(self.accel,torch.minimum(motor_accel,tire_accel))
        self.accel.copy_(torch.where(mask[:,None],accel,self.accel))

    def step(self, command, active_mask=None, *, _active_nonempty=False):
        """Advance physics, optionally compiling the complete physics step.

        The full-step route is only selected for explicit-mask callers that
        guarantee a nonempty active batch. This removes the tensor-to-host
        empty-mask check from the captured graph. The fixed tensor shapes and
        static collider/contact configuration are guarded by Dynamo; changing
        them can trigger recompilation. The compiled path mutates state tensors
        in place, so state parity must be checked before enabling it for runs.
        Eager/compiled throughput comparison after the training GPUs are free:
        ``.venv/bin/python scripts/benchmark_tensor_scaling.py --envs 1024
        --decisions 256 --physics-ticks 12 --device cuda:0 --seed 9001`` and
        ``AUTODRIVE_TORCH_COMPILE_SIM_STEP=1 .venv/bin/python
        scripts/benchmark_tensor_scaling.py --envs 1024 --decisions 256
        --physics-ticks 12 --device cuda:0 --seed 9001``.
        """
        if (self.num_robots == 2 and self._torch_compile_step_enabled and not self._torch_compile_step_failed
                and _active_nonempty and active_mask is not None):
            if self._torch_compiled_step is None:
                try:
                    self._torch_compiled_step = torch.compile(
                        self._step_compiled_eager,
                        fullgraph=True,
                        dynamic=False,
                    )
                except Exception as exc:
                    self._disable_compiled_step(exc)
                    self._step_eager(command, active_mask, _active_nonempty=True,
                                     _use_compiled_swerve=False)
                    return self.state
            try:
                self._torch_compiled_step(command, active_mask)
            except Exception as exc:
                if self._is_compile_time_failure(exc):
                    self._disable_compiled_step(exc)
                    self._step_eager(command, active_mask, _active_nonempty=True,
                                     _use_compiled_swerve=False)
                    return self.state
                # A compiled execution can fail after mutating some state.
                # Do not apply the same physics step again through eager code.
                raise
            return self.state
        self._step_eager(command, active_mask, _active_nonempty=_active_nonempty,
                         _use_compiled_swerve=True)
        return self.state

    def _step_compiled_eager(self, command, active_mask):
        """Compilation entry point; bypasses the separately compiled swerve."""
        self._step_eager(command, active_mask, _active_nonempty=True,
                         _use_compiled_swerve=False)
        return self.pose, self.velocity

    def _disable_compiled_step(self, exc):
        self._torch_compile_step_failed = True
        import warnings
        warnings.warn(
            "Compiled simulator step unavailable; using eager implementation: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _step_eager(self, command, active_mask=None, *, _active_nonempty=False,
                    _use_compiled_swerve=True):
        active_mask=(torch.ones(self.n,device=self.device,dtype=torch.bool)
                     if active_mask is None else
                     torch.as_tensor(active_mask,device=self.device,dtype=torch.bool).reshape(self.n))
        cmd=torch.as_tensor(command,device=self.device,dtype=self.pose.dtype)
        if cmd.shape==(self.num_robots,3): cmd=cmd.expand(self.n,self.num_robots,3)
        if tuple(cmd.shape)!=(self.n,self.num_robots,3): raise ValueError(f"command must be {(self.n,self.num_robots,3)}")
        cmd=torch.nan_to_num(cmd)
        omega=cmd[...,2].clamp(-self.omega_limit,self.omega_limit)
        translation_speed=cmd[...,:2].norm(dim=-1).clamp_min(1e-8)
        # Do not subtract the worst-case rotational wheel speed from chassis
        # translation here. Swerve desaturation below scales the four actual
        # module velocity vectors and preserves feasible combined motion.
        translation_scale=(self.speed/translation_speed).clamp(max=1.)
        cmd=torch.cat((cmd[...,:2]*translation_scale[...,None],omega[...,None]),-1)
        if not _active_nonempty and not bool(active_mask.any()):
            return self.state
        field_sweep_pose=(self.pose.clone() if self.field_colliders.shape[0] else None)
        fused_swerve_pose = False
        if (self.num_robots in (2, 6) and _use_compiled_swerve and self._fused_swerve_pose_hip_enabled and
                not self._fused_swerve_pose_hip_failed):
            # Build before mutating simulator state. If compilation is
            # unavailable, safely fall back for this and subsequent steps.
            try:
                from . import tensor_swerve_pose_hip
                tensor_swerve_pose_hip.extension()
            except Exception as exc:
                self._fused_swerve_pose_hip_failed = True
                import warnings
                warnings.warn(
                    "Fused swerve + pose HIP unavailable; using Torch path: "
                    f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
            else:
                # Runtime launch failures propagate: retrying with Torch could
                # apply a partially completed physics step twice.
                tensor_swerve_pose_hip.step(self, cmd, active_mask, debug=False)
                fused_swerve_pose = True
        if not fused_swerve_pose:
            if _use_compiled_swerve:
                self._swerve(cmd,active_mask)
            else:
                self._swerve_eager(cmd,active_mask)
            if field_sweep_pose is None:
                self.pose.copy_(torch.where(active_mask[:,None,None],
                    self.pose+self.velocity*self.dt,self.pose))
                wrapped=torch.remainder(self.pose[...,2]+math.pi,2*math.pi)-math.pi
                self.pose[...,2].copy_(torch.where(active_mask[:,None],wrapped,self.pose[...,2]))
        self.robot_contact &= ~active_mask
        self.opponent_contact &= ~active_mask
        self.field_contact &= ~active_mask[:,None]
        self.wall_contact &= ~active_mask[:,None,None]
        if field_sweep_pose is not None:
            # Endpoint-only SAT misses thin field structures when a chassis
            # crosses them in one physics tick. Integrate the solved chassis
            # velocity in small swept increments and resolve each contact
            # before advancing farther through the structure.
            max_robot_radius=.5*math.hypot(
                self._base["length"]*1.2,self._base["width"]*1.2)
            max_motion=(self._base["speed"]*1.3+
                        self._base["omega"]*1.3*max_robot_radius)*self.dt
            sweep_steps=max(1,math.ceil(max_motion/self.field_sweep_spacing))
            substep=self.dt/sweep_steps
            self.pose.copy_(field_sweep_pose)
            if not self._field_sweep_collision(active_mask,sweep_steps,substep):
                self._field_sweep_torch(active_mask,sweep_steps,substep)
        fused_contacts = False
        if (self.num_robots == 2 and self._fused_contact_pipeline_hip_enabled and
                _contact_pipeline_hip is not None):
            active_mask = active_mask.contiguous()
            fused_contacts = _contact_pipeline_hip(self, active_mask)
        if not fused_contacts:
            for _ in range(self.contact_iterations):
                self._robot_collision(active_mask); self._walls(active_mask)
                self._obstacle_collision(active_mask); self._field_collision(active_mask)
        return self.pose, self.velocity

    def _swerve(self,cmd,active_mask=None):
        """Run the swerve kernel, optionally using a lazily compiled graph.

        Only this fixed-shape tensor kernel is compiled. Environment dispatch,
        planner operations, physics integration, and episode handling remain
        eager. ``fullgraph`` rejects graph breaks; failures during tracing or
        backend compilation fall back to the unchanged eager implementation.
        Inductor guards batch shape and static drivetrain/bump configuration,
        so changing those can trigger recompilation. Fusion can slightly change
        floating-point rounding; verify pose, velocity, module currents, and
        inactive-row preservation against eager before using it for training.

        After the training GPUs are free, compare:
        eager: ``.venv/bin/python scripts/benchmark_tensor_scaling.py --envs
        1024 --decisions 256 --physics-ticks 12 --device cuda:0 --seed 9001``
        compiled: ``AUTODRIVE_TORCH_COMPILE_SWERVE=1 .venv/bin/python
        scripts/benchmark_tensor_scaling.py --envs 1024 --decisions 256
        --physics-ticks 12 --device cuda:0 --seed 9001``
        """
        if not self._torch_compile_swerve_enabled or self._torch_compile_swerve_failed:
            return self._swerve_eager(cmd, active_mask)
        if self._torch_compiled_swerve is None:
            try:
                self._torch_compiled_swerve = torch.compile(
                    self._swerve_eager,
                    fullgraph=True,
                    dynamic=False,
                )
            except Exception as exc:
                self._disable_compiled_swerve(exc)
                return self._swerve_eager(cmd, active_mask)
        try:
            return self._torch_compiled_swerve(cmd, active_mask)
        except Exception as exc:
            if self._is_compile_time_failure(exc):
                self._disable_compiled_swerve(exc)
                return self._swerve_eager(cmd, active_mask)
            # Do not replay a state-mutating eager kernel after an execution
            # failure: some compiled writes may already have reached the GPU.
            raise

    @staticmethod
    def _is_compile_time_failure(exc):
        """Identify compiler failures that occur before the graph is replayed."""
        error_types = []
        try:
            from torch._dynamo.exc import (BackendCompilerFailed, TorchRuntimeError,
                                           Unsupported)
            error_types.extend((BackendCompilerFailed, TorchRuntimeError, Unsupported))
        except (ImportError, AttributeError):
            pass
        try:
            from torch._inductor.exc import InductorError
            error_types.append(InductorError)
        except (ImportError, AttributeError):
            pass
        return bool(error_types) and isinstance(exc, tuple(error_types))

    def _disable_compiled_swerve(self, exc):
        self._torch_compile_swerve_failed = True
        import warnings
        warnings.warn(
            "Compiled swerve kernel unavailable; using eager implementation: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _swerve_eager(self,cmd,active_mask=None):
        if active_mask is None:
            active_mask=torch.ones(self.n,device=self.device,dtype=torch.bool)
        p=self.swerve; th=self.pose[...,2]; c,s=th.cos(),th.sin()
        speed_limit,accel_limit=self.speed,self.accel
        mx,my=self._module_x_offsets,self._module_y_offsets
        wheel_x=self.pose[...,0,None]+c[...,None]*mx-s[...,None]*my
        wheel_y=self.pose[...,1,None]+s[...,None]*mx+c[...,None]*my
        wheel_grade=torch.zeros_like(wheel_x); wheel_height=torch.zeros_like(wheel_x)
        on_bump=torch.zeros_like(wheel_x,dtype=torch.bool)
        if self.bump_regions.shape[0]:
            dx=wheel_x[...,None]-self.bump_regions[None,None,None,:,0]
            dy=wheel_y[...,None]-self.bump_regions[None,None,None,:,1]
            half_x=self.bump_regions[None,None,None,:,2].clamp_min(1e-6)
            within=(dx.abs()<=half_x)&(dy.abs()<=self.bump_regions[None,None,None,:,3])
            heights=torch.where(within,BUMP_PANEL_THICKNESS+BUMP_RAMP_RISE*(1-dx.abs()/half_x),0.)
            grades=torch.where(within,-dx.sign()*BUMP_RAMP_RISE/half_x,0.)
            selected=heights.argmax(-1,keepdim=True)
            wheel_height=heights.gather(-1,selected).squeeze(-1)
            wheel_grade=grades.gather(-1,selected).squeeze(-1)
            on_bump=within.any(-1)
        front_height=wheel_height[...,1::2].mean(-1)
        rear_height=wheel_height[...,0::2].mean(-1)
        chassis_pitch=torch.atan2(front_height-rear_height,self.length)
        normal=(self.mass[...,None]*9.81*chassis_pitch[...,None].cos()/4. -
                self.mass[...,None]*9.81*BUMP_ROBOT_CG_HEIGHT*chassis_pitch[...,None].sin()*
                mx.sign()/(2*self.length[...,None].clamp_min(1e-6))).clamp_min(0.)
        vx=c*cmd[...,0]+s*cmd[...,1]; vy=-s*cmd[...,0]+c*cmd[...,1]
        omega=cmd[...,2].clamp(-self.omega_limit,self.omega_limit)
        norm=torch.sqrt(vx.square()+vy.square()).clamp_min(1e-8)
        scale=(speed_limit/norm).clamp(max=1)
        vx,vy=vx*scale,vy*scale
        dx=vx[...,None]-omega[...,None]*my; dy=vy[...,None]+omega[...,None]*mx
        target=torch.sqrt(dx.square()+dy.square()); target*= (speed_limit[...,None]/target.amax(-1,keepdim=True).clamp_min(1e-8)).clamp(max=1)
        angle=torch.atan2(dy,dx); delta=torch.remainder(angle-self.module_angle+math.pi,2*math.pi)-math.pi
        reverse=delta.abs()>math.pi/2; angle=torch.where(reverse,angle+math.pi,angle); target=torch.where(reverse,-target,target)
        angle=torch.remainder(angle+math.pi,2*math.pi)-math.pi
        delta=torch.remainder(angle-self.module_angle+math.pi,2*math.pi)-math.pi
        # Reduce drive demand while the azimuth motor is still rotating toward
        # its target. This avoids sideways scrub at large steering errors.
        target*=delta.cos().clamp_min(0.)
        steer_target=(8*delta).clamp(-p.steer_rate_limit,p.steer_rate_limit)
        se=steer_target-self.module_steer_rate
        steer_current=(2*se.abs()).clamp(max=p.steer_current_limit)
        steer_rate=(self.module_steer_rate+se.sign()*p.steer_acceleration*
                    steer_current/max(p.steer_current_limit,1e-6)*self.dt).clamp(
                        -p.steer_rate_limit,p.steer_rate_limit)
        self.module_steer_rate.copy_(torch.where(active_mask[:,None,None],steer_rate,
                                                 self.module_steer_rate))
        angle_value=torch.remainder(self.module_angle+self.module_steer_rate*self.dt+math.pi,2*math.pi)-math.pi
        self.module_angle.copy_(torch.where(active_mask[:,None,None],angle_value,self.module_angle))
        resistance=p.nominal_voltage/p.stall_current_a; stall=p.free_speed_rpm*2*math.pi/60
        kv=stall/(p.nominal_voltage-resistance*p.free_current_a); kt=p.stall_torque_nm/p.stall_current_a
        motor_target=target/p.wheel_radius*self.drive_ratio[...,None]
        actual=self.module_drive_speed/p.wheel_radius*self.drive_ratio[...,None]
        duty=(motor_target/stall+.8*(motor_target-actual)/stall).clamp(-1,1)
        bus=(p.battery_voltage-self.robot_current*self.battery_resistance).clamp_min(0)
        amps=((duty*bus[...,None]-actual/kv)/resistance).clamp(-self.drive_current_limit[...,None],self.drive_current_limit[...,None])
        # Supply limits are per Talon/controller; the separate robot supply
        # cap below constrains the sum across all four drive modules.
        supply=(duty*amps).abs(); amps*= (self.drive_supply_limit[...,None]/supply.clamp_min(1e-6)).clamp(max=1)
        supply=(duty*amps).abs()
        drive_budget=(self.robot_supply_limit-steer_current.sum(-1)).clamp_min(0.)
        cap=(drive_budget/supply.sum(-1).clamp_min(1e-6)).clamp(max=1)
        amps*=cap[...,None]; supply*=cap[...,None]
        torque=kt*(amps-p.free_current_a*actual.sign())
        force=torque*self.drive_ratio[...,None]*p.motor_efficiency/p.wheel_radius
        longitudinal_limit=self.ground_mu[:,None,None]*normal
        force=torch.maximum(torch.minimum(force,longitudinal_limit),-longitudinal_limit)
        body_vx=c*self.velocity[...,0]+s*self.velocity[...,1]
        body_vy=-s*self.velocity[...,0]+c*self.velocity[...,1]
        module_vx=body_vx[...,None]-self.velocity[...,2,None]*my
        module_vy=body_vy[...,None]+self.velocity[...,2,None]*mx
        lateral_velocity=-module_vx*self.module_angle.sin()+module_vy*self.module_angle.cos()
        lateral_limit=self.lateral_mu[...,None]*normal
        lateral_force=(-lateral_velocity*self.mass[...,None]/(4*self.dt)).maximum(-lateral_limit).minimum(lateral_limit)
        combined=((force/longitudinal_limit.clamp_min(1e-8)).square()+
                  (lateral_force/lateral_limit.clamp_min(1e-8)).square()).sqrt()
        friction_scale=combined.clamp_min(1.)
        force=force/friction_scale
        lateral_force=lateral_force/friction_scale
        self.module_current.copy_(torch.where(active_mask[:,None,None],amps,self.module_current))
        self.module_supply_current.copy_(torch.where(active_mask[:,None,None],supply,self.module_supply_current))
        self.robot_current.copy_(torch.where(active_mask[:,None],supply.sum(-1)+steer_current.sum(-1),
                                             self.robot_current))
        fx_drive,fy_drive=force*self.module_angle.cos(),force*self.module_angle.sin()
        fx_lateral=-lateral_force*self.module_angle.sin()
        fy_lateral=lateral_force*self.module_angle.cos()
        fx,fy=fx_drive+fx_lateral,fy_drive+fy_lateral
        drive_ax,drive_ay=fx_drive.sum(-1)/self.mass,fy_drive.sum(-1)/self.mass
        drive_scale=(accel_limit/torch.sqrt(drive_ax.square()+drive_ay.square()).clamp_min(1e-8)).clamp(max=1.)
        ax=drive_ax*drive_scale+fx_lateral.sum(-1)/self.mass
        ay=drive_ay*drive_scale+fy_lateral.sum(-1)/self.mass
        inertia=(self.mass*(self.length.square()+self.width.square())/12)*self.yaw_inertia_multiplier
        az=(mx*fy-my*fx).sum(-1)/inertia
        ca,sa=c,s
        vx_new=self.velocity[...,0]+(ca*ax-sa*ay)*self.dt
        vy_new=self.velocity[...,1]+(sa*ax+ca*ay)*self.dt
        self.velocity[...,0].copy_(torch.where(active_mask[:,None],vx_new,self.velocity[...,0]))
        self.velocity[...,1].copy_(torch.where(active_mask[:,None],vy_new,self.velocity[...,1]))
        mean_grade=wheel_grade.mean(-1)
        rolling_accel=(BUMP_ROLLING_RESISTANCE*normal*on_bump).sum(-1)/self.mass
        vx_new=self.velocity[...,0]+(-9.81*mean_grade-rolling_accel*self.velocity[...,0].sign())*self.dt
        self.velocity[...,0].copy_(torch.where(active_mask[:,None],vx_new,self.velocity[...,0]))
        vnorm=self.velocity[...,:2].norm(dim=-1).clamp_min(1e-8)
        xy_new=self.velocity[...,:2]*(speed_limit/vnorm).clamp(max=1)[...,None]
        self.velocity[...,:2].copy_(torch.where(active_mask[:,None,None],xy_new,self.velocity[...,:2]))
        omega_new=(self.velocity[...,2]+az.clamp(-self.alpha,self.alpha)*self.dt).clamp(
            -self.omega_limit,self.omega_limit)
        self.velocity[...,2].copy_(torch.where(active_mask[:,None],omega_new,self.velocity[...,2]))
        bodyx=ca*self.velocity[...,0]+sa*self.velocity[...,1]; bodyy=-sa*self.velocity[...,0]+ca*self.velocity[...,1]
        drive_speed=(bodyx[...,None]-self.velocity[...,2,None]*my)*self.module_angle.cos()+(bodyy[...,None]+self.velocity[...,2,None]*mx)*self.module_angle.sin()
        self.module_drive_speed.copy_(torch.where(active_mask[:,None,None],drive_speed,self.module_drive_speed))
