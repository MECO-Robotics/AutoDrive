"""Fast batched planar holonomic robot simulator with rigid-body contacts.

Translations and commands are in field coordinates; angles are radians.  The
simulator deliberately exposes numeric arrays rather than leaking its state
objects into policy/runtime interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np

from .field import (BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE,
                    BUMP_ROBOT_CG_HEIGHT, BUMP_ROLLING_RESISTANCE)
from .observation import normalize_numpy_observation_in_place
from .reward import (ROTATION_COMMAND_DELTA_PENALTY, ROTATION_COMMAND_PENALTY,
                     SPIN_RATE_PENALTY)

try:  # Training environments need not install Gymnasium.
    import gymnasium as gym
    from gymnasium import spaces
    _EnvBase = gym.Env
except ImportError:  # tiny compatible surface for basic use
    class _EnvBase:
        metadata: dict[str, Any] = {}

    class _Box:
        def __init__(self, low: Any, high: Any, shape: tuple[int, ...] | None = None, dtype=np.float32):
            self.dtype = dtype
            if shape is not None:
                self.low = np.full(shape, low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype)
            else:
                self.low, self.high = np.asarray(low, dtype=dtype), np.asarray(high, dtype=dtype)
            self.shape = self.low.shape
        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class _Spaces:
        Box = _Box
    spaces = _Spaces()


@dataclass
class BatchState:
    """Arrays have leading batch dimension; robots index 0=controlled, 1=other."""
    pose: np.ndarray       # (N,2,3), x y theta
    velocity: np.ndarray   # (N,2,3), vx vy omega


@dataclass(frozen=True)
class DCMotorParameters:
    """Parameterized brushed/brushless motor curve at nominal voltage."""
    stall_torque_nm: float = 7.09
    stall_current_a: float = 366.0
    free_speed_rpm: float = 6000.0
    free_current_a: float = 2.0
    nominal_voltage: float = 12.0


@dataclass(frozen=True)
class SwerveParameters:
    wheel_radius: float = 0.0508
    drive_ratio: float = 6.75
    drive_current_limit: float = 60.0
    drive_supply_current_limit: float = 80.0
    robot_supply_current_limit: float = 180.0
    steer_current_limit: float = 20.0
    steer_rate_limit: float = 12.0
    steer_acceleration: float = 80.0
    motor_efficiency: float = 0.9
    battery_voltage: float = 12.6
    battery_internal_resistance: float = 0.018
    motor: DCMotorParameters = DCMotorParameters()
    # Module centers are inside the bumper perimeter; tune these from CAD.
    module_x_offset: float = 0.36
    module_y_offset: float = 0.36


class VectorizedSimulator:
    """Vectorized dynamics; contact resolution is per-world with vector math."""
    def __init__(self, num_envs: int = 1, dt: float = 0.02, field_length: float = 16.54,
                 field_width: float = 8.21, robot_length: float = .9, robot_width: float = .9,
                 max_speed: float = 4.5, max_acceleration: float = 8., max_omega: float = 8.,
                 max_alpha: float = 18., mass: float = 55., bumper_friction: float = .65,
                 field_friction: float = .7, lateral_friction: float = 1.2,
                 yaw_inertia_multiplier: float = 1.5,
                 randomize: bool = False, seed: int | None = None,
                 contact_iterations: int = 2, swerve: SwerveParameters | None = None,
                 obstacles: tuple[tuple[float, float, float], ...] = (),
                 field_colliders: tuple[tuple[float, float, float, float], ...] = (),
                 bump_regions: tuple[tuple[float, float, float, float], ...] = (),
                 bumper_field_friction: float | None = None):
        if num_envs < 1 or dt <= 0:
            raise ValueError("num_envs and dt must be positive")
        self.n, self.dt = int(num_envs), float(dt)
        self.field_length, self.field_width = float(field_length), float(field_width)
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.randomize = randomize
        self.contact_iterations = max(1, int(contact_iterations))
        self.base = dict(length=robot_length, width=robot_width, speed=max_speed,
                         accel=max_acceleration, omega=max_omega, alpha=max_alpha, mass=mass,
                         bumper_friction=bumper_friction, field_friction=field_friction,
                         lateral_friction=lateral_friction,
                         yaw_inertia_multiplier=yaw_inertia_multiplier)
        self.pose = np.zeros((self.n, 2, 3), np.float32)
        self.velocity = np.zeros_like(self.pose)
        self.length = np.full((self.n, 2), robot_length, np.float32)
        self.width = np.full((self.n, 2), robot_width, np.float32)
        self.mass = np.full((self.n, 2), mass, np.float32)
        self.speed = np.full((self.n, 2), max_speed, np.float32)
        self.accel = np.full((self.n, 2), max_acceleration, np.float32)
        self.omega_limit = np.full((self.n, 2), max_omega, np.float32)
        self.alpha = np.full((self.n, 2), max_alpha, np.float32)
        self.mu = np.full(self.n, bumper_friction, np.float32)
        self.wall_mu = np.full(self.n, field_friction if bumper_field_friction is None else bumper_field_friction,
                               np.float32)
        self.ground_mu = np.full(self.n, field_friction, np.float32)
        self.lateral_mu = np.full((self.n, 2), lateral_friction, np.float32)
        self.yaw_inertia_multiplier = np.full((self.n, 2), yaw_inertia_multiplier, np.float32)
        self.base_bumper_field_friction = field_friction if bumper_field_friction is None else bumper_field_friction
        self.obstacles = tuple(tuple(map(float, obstacle)) for obstacle in obstacles)
        self.field_colliders = tuple(tuple(map(float, box)) for box in field_colliders)
        self.bump_regions = tuple(tuple(map(float, box)) for box in bump_regions)
        self.swerve = swerve or SwerveParameters()
        self.module_angle = np.zeros((self.n, 2, 4), np.float32)
        self.module_steer_rate = np.zeros_like(self.module_angle)
        self.module_drive_speed = np.zeros_like(self.module_angle)
        self.module_current = np.zeros_like(self.module_angle)
        self.module_supply_current = np.zeros_like(self.module_angle)
        self.robot_current = np.zeros((self.n, 2), np.float32)
        self.robot_contact = np.zeros(self.n, dtype=bool)
        self.drive_ratio = np.full((self.n,2), self.swerve.drive_ratio, np.float32)
        self.drive_current_limit = np.full((self.n,2), self.swerve.drive_current_limit, np.float32)
        self.drive_supply_limit = np.full((self.n,2), self.swerve.drive_supply_current_limit, np.float32)
        self.robot_supply_limit = np.full((self.n,2), self.swerve.robot_supply_current_limit, np.float32)
        self.battery_resistance = np.full((self.n,2), self.swerve.battery_internal_resistance, np.float32)
        self.reset()

    def reset(self, seed: int | None = None, pose: np.ndarray | None = None,
              velocity: np.ndarray | None = None) -> BatchState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pose.fill(0); self.velocity.fill(0)
        self.module_angle.fill(0); self.module_steer_rate.fill(0)
        self.module_drive_speed.fill(0); self.module_current.fill(0)
        self.module_supply_current.fill(0); self.robot_current.fill(0)
        self.robot_contact.fill(False)
        if self.randomize:
            self.drive_ratio[:] = self.swerve.drive_ratio*self.rng.uniform(.9,1.1,(self.n,2))
            self.drive_current_limit[:] = self.swerve.drive_current_limit*self.rng.uniform(.75,1.25,(self.n,2))
            self.drive_supply_limit[:] = self.swerve.drive_supply_current_limit*self.rng.uniform(.75,1.25,(self.n,2))
            self.robot_supply_limit[:] = self.swerve.robot_supply_current_limit*self.rng.uniform(.8,1.2,(self.n,2))
            self.battery_resistance[:] = self.swerve.battery_internal_resistance*self.rng.uniform(.7,1.6,(self.n,2))
            self.lateral_mu[:] = self.base['lateral_friction']*self.rng.uniform(.8,1.25,(self.n,2))
            self.yaw_inertia_multiplier[:] = self.base['yaw_inertia_multiplier']*self.rng.uniform(.8,1.2,(self.n,2))
        else:
            self.drive_ratio.fill(self.swerve.drive_ratio)
            self.drive_current_limit.fill(self.swerve.drive_current_limit)
            self.drive_supply_limit.fill(self.swerve.drive_supply_current_limit)
            self.robot_supply_limit.fill(self.swerve.robot_supply_current_limit)
            self.battery_resistance.fill(self.swerve.battery_internal_resistance)
            self.lateral_mu.fill(self.base['lateral_friction'])
            self.yaw_inertia_multiplier.fill(self.base['yaw_inertia_multiplier'])
        if pose is None:
            margin = max(self.base['length'], self.base['width'])
            self.pose[:, :, 0] = self.rng.uniform(margin, self.field_length-margin, (self.n, 2))
            self.pose[:, :, 1] = self.rng.uniform(margin, self.field_width-margin, (self.n, 2))
            self.pose[:, :, 2] = self.rng.uniform(-np.pi, np.pi, (self.n, 2))
        else:
            self.pose[:] = np.broadcast_to(np.asarray(pose, np.float32), self.pose.shape)
        if velocity is not None:
            self.velocity[:] = np.broadcast_to(np.asarray(velocity, np.float32), self.velocity.shape)
        if self.randomize:
            self.length[:] = self.base['length'] * self.rng.uniform(.82, 1.2, (self.n, 2))
            self.width[:] = self.base['width'] * self.rng.uniform(.82, 1.2, (self.n, 2))
            self.mass[:] = self.base['mass'] * self.rng.uniform(.65, 1.45, (self.n, 2))
            self.speed[:] = self.base['speed'] * self.rng.uniform(.75, 1.3, (self.n, 2))
            self.accel[:] = self.base['accel'] * self.rng.uniform(.65, 1.4, (self.n, 2))
            self.omega_limit[:] = self.base['omega'] * self.rng.uniform(.7, 1.3, (self.n, 2))
            self.alpha[:] = self.base['alpha'] * self.rng.uniform(.65, 1.4, (self.n, 2))
            self.mu[:] = self.rng.uniform(.25, 1.1, self.n)
            self.wall_mu[:] = self.rng.uniform(.25, 1.1, self.n)
            self.ground_mu[:] = self.rng.uniform(.35, 1.2, self.n)
        else:
            for a, key in ((self.length,'length'),(self.width,'width'),(self.mass,'mass'),
                           (self.speed,'speed'),(self.accel,'accel'),(self.omega_limit,'omega'),(self.alpha,'alpha')):
                a.fill(self.base[key])
            self.mu.fill(self.base['bumper_friction'])
            self.wall_mu.fill(self.base_bumper_field_friction)
            self.ground_mu.fill(self.base['field_friction'])
        # Clamp the requested acceleration ceiling to what the configured
        # drive motors, gearing, traction, and robot mass can produce.
        motor = self.swerve.motor
        torque_per_amp = motor.stall_torque_nm / motor.stall_current_a
        motor_accel = (4*torque_per_amp*self.drive_current_limit*self.drive_ratio*
                       self.swerve.motor_efficiency /
                       np.maximum(self.swerve.wheel_radius*self.mass, 1e-6))
        self.accel[:] = np.minimum(self.accel, np.minimum(motor_accel, self.ground_mu[:, None]*9.81))
        for i in range(self.n):
            self._field_contact(i)
            self.robot_contact[i] = False
        return self.state

    @property
    def state(self) -> BatchState:
        return BatchState(self.pose.copy(), self.velocity.copy())

    def step(self, command: np.ndarray) -> BatchState:
        """Advance one frame with (N,2,3) desired field vx,vy,omega commands."""
        cmd = np.asarray(command, dtype=np.float32).copy()
        if cmd.shape == (2, 3): cmd = np.broadcast_to(cmd, (self.n, 2, 3))
        if cmd.shape != self.pose.shape:
            raise ValueError(f"command must have shape {(self.n,2,3)}")
        cmd = np.nan_to_num(cmd, nan=0., posinf=0., neginf=0.)
        self._swerve_step(cmd)
        self.pose += self.velocity * self.dt
        self.pose[..., 2] = (self.pose[..., 2] + np.pi) % (2*np.pi) - np.pi
        self.robot_contact.fill(False)
        for _ in range(self.contact_iterations):
            for i in range(self.n):
                self.robot_contact[i] |= self._robot_contact(i)
                self._wall_contact(i)
                self._obstacle_contact(i)
                self._field_contact(i)
        return self.state

    def _swerve_step(self, command: np.ndarray) -> None:
        """Four-module inverse kinematics plus current-limited DC motor forces."""
        p = self.swerve
        theta = self.pose[..., 2]
        c, s = np.cos(theta), np.sin(theta)
        speed_limit, accel_limit = self.speed, self.accel
        mx = np.stack([-np.full_like(self.length, p.module_x_offset),
                       np.full_like(self.length, p.module_x_offset),
                       -np.full_like(self.length, p.module_x_offset),
                       np.full_like(self.length, p.module_x_offset)], axis=-1)
        my = np.stack([-np.full_like(self.width, p.module_y_offset),
                       -np.full_like(self.width, p.module_y_offset),
                       np.full_like(self.width, p.module_y_offset),
                       np.full_like(self.width, p.module_y_offset)], axis=-1)
        # Sample a triangular bump profile at each wheel. This quasi-3D model
        # uses those heights for chassis pitch/load transfer and slope gravity.
        wheel_x = self.pose[..., 0, None] + c[..., None]*mx - s[..., None]*my
        wheel_y = self.pose[..., 1, None] + s[..., None]*mx + c[..., None]*my
        wheel_grade = np.zeros_like(wheel_x)
        wheel_height = np.zeros_like(wheel_x)
        on_bump = np.zeros_like(wheel_x, dtype=bool)
        if self.bump_regions:
            boxes = np.asarray(self.bump_regions, np.float32)
            dx = wheel_x[..., None] - boxes[None, None, None, :, 0]
            dy = wheel_y[..., None] - boxes[None, None, None, :, 1]
            half_x = np.maximum(boxes[None, None, None, :, 2], 1e-6)
            within = (np.abs(dx) <= half_x) & (np.abs(dy) <= boxes[None, None, None, :, 3])
            heights = np.where(within, BUMP_PANEL_THICKNESS +
                               BUMP_RAMP_RISE*(1-np.abs(dx)/half_x), 0.)
            grades = np.where(within, -np.sign(dx)*BUMP_RAMP_RISE/half_x, 0.)
            selected = heights.argmax(axis=-1)
            wheel_height = np.take_along_axis(heights, selected[..., None], axis=-1)[..., 0]
            wheel_grade = np.take_along_axis(grades, selected[..., None], axis=-1)[..., 0]
            on_bump = within.any(axis=-1)
        front_height = wheel_height[..., 1::2].mean(axis=-1)
        rear_height = wheel_height[..., 0::2].mean(axis=-1)
        chassis_pitch = np.arctan2(front_height-rear_height, self.length)
        normal_load = (self.mass[..., None]*9.81*np.cos(chassis_pitch[..., None])/4. -
                       self.mass[..., None]*9.81*BUMP_ROBOT_CG_HEIGHT*np.sin(chassis_pitch[..., None])*
                       np.sign(mx)/(2*np.maximum(self.length[..., None], 1e-6)))
        normal_load = np.maximum(normal_load, 0.)
        # Convert field-relative chassis requests into each robot's body frame.
        vx = c*command[..., 0] + s*command[..., 1]
        vy = -s*command[..., 0] + c*command[..., 1]
        chassis_norm = np.hypot(vx,vy)
        scale = np.minimum(1.,speed_limit/np.maximum(chassis_norm,1e-8))
        vx *= scale; vy *= scale
        omega = np.clip(command[..., 2], -self.omega_limit, self.omega_limit)
        desired_x = vx[..., None] - omega[..., None]*my
        desired_y = vy[..., None] + omega[..., None]*mx
        target_speed = np.hypot(desired_x, desired_y)
        module_scale = np.minimum(1.,speed_limit[...,None]/np.maximum(target_speed.max(axis=-1,keepdims=True),1e-8))
        target_speed *= module_scale
        target_angle = np.arctan2(desired_y, desired_x)
        delta = (target_angle-self.module_angle+np.pi) % (2*np.pi)-np.pi
        reverse = np.abs(delta) > np.pi/2
        target_angle = np.where(reverse, target_angle+np.pi, target_angle)
        target_speed = np.where(reverse, -target_speed, target_speed)
        target_angle = (target_angle+np.pi) % (2*np.pi)-np.pi
        delta = (target_angle-self.module_angle+np.pi) % (2*np.pi)-np.pi
        steer_rate_target = np.clip(8.0*delta, -p.steer_rate_limit, p.steer_rate_limit)
        steer_error = steer_rate_target-self.module_steer_rate
        steer_current = np.minimum(p.steer_current_limit, np.abs(steer_error)*2.0)
        steer_accel = np.sign(steer_error)*p.steer_acceleration*steer_current/max(p.steer_current_limit,1e-6)
        self.module_steer_rate += steer_accel*self.dt
        self.module_steer_rate[:] = np.clip(self.module_steer_rate,
                                            -p.steer_rate_limit, p.steer_rate_limit)
        self.module_angle[:] = (self.module_angle+self.module_steer_rate*self.dt+np.pi)%(2*np.pi)-np.pi

        motor = p.motor
        resistance = motor.nominal_voltage / motor.stall_current_a
        stall_speed = motor.free_speed_rpm * (2*np.pi/60.)
        kv = stall_speed / (motor.nominal_voltage-resistance*motor.free_current_a)
        kt = motor.stall_torque_nm / motor.stall_current_a
        free_motor_speed = stall_speed
        target_motor_speed = target_speed / p.wheel_radius * self.drive_ratio[...,None]
        actual_motor_speed = self.module_drive_speed / p.wheel_radius * self.drive_ratio[...,None]
        duty = np.clip(target_motor_speed/free_motor_speed +
                       .8*(target_motor_speed-actual_motor_speed)/free_motor_speed, -1., 1.)
        bus_voltage = np.maximum(0., p.battery_voltage-self.robot_current*self.battery_resistance)
        applied_voltage = duty*bus_voltage[..., None]
        current = (applied_voltage-actual_motor_speed/kv)/resistance
        current = np.clip(current, -self.drive_current_limit[...,None], self.drive_current_limit[...,None])
        supply_current = np.abs(duty*current)
        current *= np.minimum(1., self.drive_supply_limit[...,None]/np.maximum(supply_current,1e-6))
        supply_current = np.abs(duty*current)
        drive_budget = np.maximum(self.robot_supply_limit-steer_current.sum(axis=-1), 0.)
        current_scale = np.minimum(1., drive_budget/np.maximum(supply_current.sum(axis=-1),1e-6))
        current *= current_scale[...,None]
        supply_current *= current_scale[...,None]
        torque = kt*(current-motor.free_current_a*np.sign(actual_motor_speed))
        wheel_force = torque*p.drive_ratio*p.motor_efficiency/p.wheel_radius
        # Bound motor force along each module's azimuth.
        wheel_force = np.clip(wheel_force, -self.ground_mu[:,None,None]*normal_load,
                              self.ground_mu[:,None,None]*normal_load)
        # Carpet scrub supplies strong passive force orthogonal to the wheel
        # azimuth. The high lateral stiffness arrests slip within one step,
        # bounded by a separate lateral friction coefficient.
        body_vx = c*self.velocity[...,0] + s*self.velocity[...,1]
        body_vy = -s*self.velocity[...,0] + c*self.velocity[...,1]
        module_vx = body_vx[...,None] - self.velocity[...,2,None]*my
        module_vy = body_vy[...,None] + self.velocity[...,2,None]*mx
        lateral_velocity = -module_vx*np.sin(self.module_angle) + module_vy*np.cos(self.module_angle)
        lateral_limit = self.lateral_mu[...,None]*normal_load
        lateral_force = np.clip(-lateral_velocity*self.mass[...,None]/(4*self.dt),
                                -lateral_limit, lateral_limit)
        combined = np.sqrt((wheel_force/np.maximum(self.ground_mu[:,None,None]*normal_load,1e-8))**2 +
                           (lateral_force/np.maximum(lateral_limit,1e-8))**2)
        friction_scale = np.maximum(combined, 1.)
        wheel_force /= friction_scale
        lateral_force /= friction_scale
        self.module_current[:] = current
        self.module_supply_current[:] = supply_current
        fx_drive = wheel_force*np.cos(self.module_angle)
        fy_drive = wheel_force*np.sin(self.module_angle)
        fx_lateral = -lateral_force*np.sin(self.module_angle)
        fy_lateral = lateral_force*np.cos(self.module_angle)
        fx_body, fy_body = fx_drive+fx_lateral, fy_drive+fy_lateral
        total_fx, total_fy = fx_body.sum(axis=-1), fy_body.sum(axis=-1)
        torque_z = (mx*fy_body-my*fx_body).sum(axis=-1)
        # Approximate steering motor draw from rate tracking error and active current cap.
        self.robot_current[:] = supply_current.sum(axis=-1)+steer_current.sum(axis=-1)
        drive_ax, drive_ay = fx_drive.sum(axis=-1)/self.mass, fy_drive.sum(axis=-1)/self.mass
        drive_scale = np.minimum(1., accel_limit/np.maximum(np.hypot(drive_ax, drive_ay), 1e-8))
        ax_body = drive_ax*drive_scale + fx_lateral.sum(axis=-1)/self.mass
        ay_body = drive_ay*drive_scale + fy_lateral.sum(axis=-1)/self.mass
        ca, sa = np.cos(theta), np.sin(theta)
        ax_field = ca*ax_body-sa*ay_body
        ay_field = sa*ax_body+ca*ay_body
        mean_grade = wheel_grade.mean(axis=-1)
        rolling_accel = (BUMP_ROLLING_RESISTANCE*normal_load*on_bump).sum(axis=-1)/self.mass
        ax_field += -9.81*mean_grade - rolling_accel*np.sign(self.velocity[..., 0])
        dv = np.stack([ax_field, ay_field], axis=-1)*self.dt
        self.velocity[..., :2] += dv
        norm = np.linalg.norm(self.velocity[..., :2], axis=-1, keepdims=True)
        self.velocity[..., :2] *= np.minimum(1., speed_limit[..., None]/np.maximum(norm,1e-8))
        inertia = (self.mass*(self.length**2+self.width**2)/12. *
                   self.yaw_inertia_multiplier)
        angular_accel = np.clip(torque_z/inertia, -self.alpha, self.alpha)
        self.velocity[..., 2] += angular_accel*self.dt
        self.velocity[..., 2] = np.clip(self.velocity[...,2], -self.omega_limit, self.omega_limit)
        body_vx = ca*self.velocity[...,0] + sa*self.velocity[...,1]
        body_vy = -sa*self.velocity[...,0] + ca*self.velocity[...,1]
        module_vx = body_vx[...,None] - self.velocity[...,2,None]*my
        module_vy = body_vy[...,None] + self.velocity[...,2,None]*mx
        self.module_drive_speed[:] = module_vx*np.cos(self.module_angle)+module_vy*np.sin(self.module_angle)

    def _robot_contact(self, i: int) -> bool:
        # SAT over the four face normals. Minimum overlap gives stable cheap resolution.
        p = self.pose[i, :, :2]; theta = self.pose[i, :, 2]
        axes = np.array([[np.cos(theta[0]), np.sin(theta[0])],[-np.sin(theta[0]), np.cos(theta[0])],
                         [np.cos(theta[1]), np.sin(theta[1])],[-np.sin(theta[1]), np.cos(theta[1])]])
        half = self.width[i] * .5; halfl = self.length[i] * .5
        radii = np.zeros(4)
        for k, a in enumerate(axes):
            for r in range(2):
                u = np.array([np.cos(theta[r]), np.sin(theta[r])]); v = np.array([-u[1], u[0]])
                radii[k] += abs(a @ u)*halfl[r] + abs(a @ v)*half[r]
        delta = p[1]-p[0]
        signed = axes @ delta
        overlap = radii - np.abs(signed)
        k = int(np.argmin(overlap))
        if overlap[k] <= 0: return False
        normal = axes[k] * (1. if signed[k] >= 0 else -1.)
        invm = 1. / self.mass[i]; total_inv = invm.sum()
        correction = normal * (overlap[k] + 1e-4)
        p[0] -= correction * invm[0]/total_inv
        p[1] += correction * invm[1]/total_inv
        u0 = np.array([np.cos(theta[0]),np.sin(theta[0])]); v0=np.array([-u0[1],u0[0]])
        u1 = np.array([np.cos(theta[1]),np.sin(theta[1])]); v1=np.array([-u1[1],u1[0]])
        support0 = p[0] + np.sign(normal@u0)*u0*halfl[0] + np.sign(normal@v0)*v0*half[0]
        support1 = p[1] - np.sign(normal@u1)*u1*halfl[1] - np.sign(normal@v1)*v1*half[1]
        point = .5*(support0+support1)
        r0, r1 = point-p[0], point-p[1]
        inertia = (self.mass[i]*(self.length[i]**2+self.width[i]**2)/12. *
                   self.yaw_inertia_multiplier[i])
        cross = lambda a,b: a[0]*b[1]-a[1]*b[0]
        vel0 = self.velocity[i,0,:2] + self.velocity[i,0,2]*np.array([-r0[1],r0[0]])
        vel1 = self.velocity[i,1,:2] + self.velocity[i,1,2]*np.array([-r1[1],r1[0]])
        vn = (vel1-vel0) @ normal
        if vn < 0:
            effective = total_inv + cross(r0,normal)**2/inertia[0] + cross(r1,normal)**2/inertia[1]
            impulse = -(1.05 * vn) / effective
            self.velocity[i,0,:2] -= impulse*normal*invm[0]
            self.velocity[i,1,:2] += impulse*normal*invm[1]
            self.velocity[i,0,2] -= impulse*cross(r0,normal)/inertia[0]
            self.velocity[i,1,2] += impulse*cross(r1,normal)/inertia[1]
            tangent = np.array([-normal[1], normal[0]])
            vel0 = self.velocity[i,0,:2] + self.velocity[i,0,2]*np.array([-r0[1],r0[0]])
            vel1 = self.velocity[i,1,:2] + self.velocity[i,1,2]*np.array([-r1[1],r1[0]])
            vt = (vel1-vel0) @ tangent
            tangent_effective = total_inv + cross(r0,tangent)**2/inertia[0] + cross(r1,tangent)**2/inertia[1]
            jt = np.clip(-vt/tangent_effective, -self.mu[i]*impulse, self.mu[i]*impulse)
            self.velocity[i,0,:2] -= jt*tangent*invm[0]
            self.velocity[i,1,:2] += jt*tangent*invm[1]
            self.velocity[i,0,2] -= jt*cross(r0,tangent)/inertia[0]
            self.velocity[i,1,2] += jt*cross(r1,tangent)/inertia[1]
        return True

    def _wall_contact(self, i: int) -> None:
        # Oriented rectangle's axis-aligned half extents against four field walls.
        for r in range(2):
            x,y,t = self.pose[i,r]
            hx = .5*(self.length[i,r]*abs(np.cos(t))+self.width[i,r]*abs(np.sin(t)))
            hy = .5*(self.length[i,r]*abs(np.sin(t))+self.width[i,r]*abs(np.cos(t)))
            bounds = ((0, hx-x, np.array([1.,0.])), (self.field_length, x+hx-self.field_length, np.array([-1.,0.])),
                      (0, hy-y, np.array([0.,1.])), (self.field_width, y+hy-self.field_width, np.array([0.,-1.])))
            for _, penetration, inward in bounds:
                if penetration > 0:
                    self.pose[i,r,:2] += inward*penetration
                    contact = self._support_point(i, r, -inward)
                    self._static_contact_impulse(i, r, contact, inward, self.wall_mu[i])

    def _support_point(self, i: int, r: int, direction: np.ndarray) -> np.ndarray:
        """Return the oriented bumper's support point toward ``direction``."""
        theta = self.pose[i, r, 2]
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-u[1], u[0]])
        return (self.pose[i, r, :2] + np.sign(direction @ u) * u * self.length[i, r] / 2 +
                np.sign(direction @ v) * v * self.width[i, r] / 2)

    def _static_contact_impulse(self, i: int, r: int, point: np.ndarray,
                                normal: np.ndarray, friction: float) -> None:
        """Apply a frictional rigid-body impulse from an immovable surface."""
        lever = point - self.pose[i, r, :2]
        inertia = (self.mass[i, r] * (self.length[i, r]**2 + self.width[i, r]**2) /
                   12. * self.yaw_inertia_multiplier[i, r])
        cross = lambda a, b: a[0]*b[1] - a[1]*b[0]
        omega = self.velocity[i, r, 2]
        contact_velocity = self.velocity[i, r, :2] + omega*np.array([-lever[1], lever[0]])
        vn = contact_velocity @ normal
        if vn >= 0:
            return
        inv_mass = 1. / self.mass[i, r]
        rn = cross(lever, normal)
        normal_impulse = -(1.05*vn) / (inv_mass + rn*rn/inertia)
        impulse = normal_impulse*normal
        self.velocity[i, r, :2] += impulse*inv_mass
        self.velocity[i, r, 2] += cross(lever, impulse)/inertia
        tangent = np.array([-normal[1], normal[0]])
        contact_velocity = self.velocity[i, r, :2] + self.velocity[i, r, 2]*np.array([-lever[1], lever[0]])
        vt = contact_velocity @ tangent
        rt = cross(lever, tangent)
        tangent_impulse = np.clip(-vt/(inv_mass + rt*rt/inertia),
                                  -friction*normal_impulse, friction*normal_impulse)
        impulse = tangent_impulse*tangent
        self.velocity[i, r, :2] += impulse*inv_mass
        self.velocity[i, r, 2] += cross(lever, impulse)/inertia

    def _box_contact_point(self, i: int, r: int, center: np.ndarray,
                           half: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Midpoint of the overlapping contact patch for two oriented boxes."""
        theta = self.pose[i, r, 2]
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-u[1], u[0]])
        tangent = np.array([-normal[1], normal[0]])
        robot_half = np.array([self.length[i, r], self.width[i, r]]) / 2
        robot_normal = abs(normal @ u)*robot_half[0] + abs(normal @ v)*robot_half[1]
        box_normal = abs(normal[0])*half[0] + abs(normal[1])*half[1]
        normal_coordinate = .5*((normal @ self.pose[i, r, :2] - robot_normal) +
                                 (normal @ center + box_normal))
        robot_tangent = abs(tangent @ u)*robot_half[0] + abs(tangent @ v)*robot_half[1]
        box_tangent = abs(tangent[0])*half[0] + abs(tangent[1])*half[1]
        robot_center = tangent @ self.pose[i, r, :2]
        box_center = tangent @ center
        overlap_low = max(robot_center-robot_tangent, box_center-box_tangent)
        overlap_high = min(robot_center+robot_tangent, box_center+box_tangent)
        tangent_coordinate = .5*(overlap_low+overlap_high)
        return normal*normal_coordinate + tangent*tangent_coordinate

    def _obstacle_contact(self, i: int) -> None:
        """Resolve circle obstacles against each oriented rectangular bumper."""
        for ox, oy, radius in self.obstacles:
            for r in range(2):
                x, y, theta = self.pose[i, r]
                u = np.array([np.cos(theta), np.sin(theta)])
                v = np.array([-u[1], u[0]])
                relative = np.array([ox-x, oy-y])
                local = np.array([relative@u, relative@v])
                half = np.array([self.length[i,r], self.width[i,r]])*.5
                delta = local-np.clip(local, -half, half)
                distance = float(np.linalg.norm(delta))
                if distance >= radius:
                    continue
                if distance < 1e-8:
                    gaps = half-np.abs(local)
                    axis = int(np.argmin(gaps))
                    direction = np.zeros(2); direction[axis] = -1. if local[axis] >= 0 else 1.
                    penetration = radius+gaps[axis]
                else:
                    direction = -delta/distance
                    penetration = radius-distance
                normal = direction[0]*u+direction[1]*v
                self.pose[i,r,:2] += normal*(penetration+1e-4)
                robot_point = self._support_point(i, r, -normal)
                circle_point = np.array([ox, oy]) + normal*radius
                self._static_contact_impulse(i, r, .5*(robot_point+circle_point),
                                             normal, self.wall_mu[i])
                self.robot_contact[i] = True


    def _field_contact(self, i: int) -> None:
        """Resolve oriented robot bumpers against axis-aligned field boxes."""
        for cx, cy, box_hl, box_hw in self.field_colliders:
            for r in range(2):
                x,y,t=self.pose[i,r]; c,s=np.cos(t),np.sin(t)
                axes=np.array([[1.,0.],[0.,1.],[c,s],[-s,c]])
                delta=self.pose[i,r,:2]-np.array([cx,cy])
                signed=axes@delta
                hl,hw=self.length[i,r]/2,self.width[i,r]/2
                ac,ass=abs(c),abs(s)
                robot_r=np.array([ac*hl+ass*hw+box_hl,ass*hl+ac*hw+box_hw,
                    hl+box_hl*ac+box_hw*ass,hw+box_hl*ass+box_hw*ac])
                penetration=robot_r-np.abs(signed)
                depth=float(penetration.min())
                if depth<0: continue
                tied = penetration <= depth + max(1e-6, depth*1e-4)
                candidate_normals = axes * np.where(signed[:,None] >= 0, 1., -1.)
                approach = candidate_normals @ self.velocity[i,r,:2]
                approaching = np.where(tied, approach, np.inf)
                axis = int(np.argmin(approaching)) if np.any(approaching < -1e-6) else int(np.argmin(penetration))
                normal=axes[axis]*(1. if signed[axis]>=0 else -1.)
                self.pose[i,r,:2]+=normal*(depth+1e-4)
                point=self._box_contact_point(i,r,np.array([cx,cy]),
                                              np.array([box_hl,box_hw]),normal)
                self._static_contact_impulse(i,r,point,normal,self.wall_mu[i])
                self.robot_contact[i]=True


class CounterDefenseEnv(_EnvBase):
    """Single-world Gymnasium env. The controlled robot navigates to a goal."""
    metadata = {"render_modes": []}
    def __init__(self, mode: str = "counter", dt: float = .02, horizon: int = 750,
                 randomize: bool = True, field_length: float = 16.54, field_width: float = 8.07,
                 seed: int | None = None, observation_noise: float = 0., dropout: float = 0.,
                 opponent: Any = "guard", control_latency: float = 0.,
                 observation_latency: float = 0., randomize_latency: bool = True,
                 swerve: SwerveParameters | None = None,
                 obstacles: tuple[tuple[float, float, float], ...] = (),
                 field_layout: str = "2026_rebuilt",
                 randomize_observation: bool = True, goal_radius: float = .5):
        if mode not in ("counter", "defense"):
            raise ValueError("mode must be 'counter' or 'defense'")
        self.mode, self.dt, self.horizon = mode, dt, horizon
        self.goal_radius = float(goal_radius)
        if self.goal_radius <= 0: raise ValueError("goal_radius must be positive")
        self.opponent = opponent
        from .field import bump_boxes, rebuilt_field, static_collision_boxes
        self.field_boxes=rebuilt_field(field_length,field_width) if field_layout=="2026_rebuilt" else ()
        solids=static_collision_boxes(self.field_boxes)
        self.sim = VectorizedSimulator(1, dt, field_length, field_width,
                                       randomize=randomize, seed=seed, swerve=swerve,
                                       obstacles=obstacles,
                                       field_colliders=tuple(box.as_tensor() for box in solids),
                                       bump_regions=tuple(box.as_tensor() for box in bump_boxes(self.field_boxes)))
        self.rng = np.random.default_rng(seed)
        self.noise, self.dropout = observation_noise, dropout
        self.randomize_observation = randomize_observation
        self.noise_level, self.dropout_probability = observation_noise, dropout
        self.control_latency, self.observation_latency = control_latency, observation_latency
        self.randomize_latency = randomize_latency
        self.control_delay = self.observation_delay = 0
        self.action_history: list[np.ndarray] = []
        self.observation_history: list[np.ndarray] = []
        self.path_length = 0.0
        self.initial_dist = 0.0
        self.contact_count = 0
        self.contact_time = 0.0
        self.blocked_time = 0.0
        self.last_contact = False
        self.last_action = np.zeros(3, np.float32)
        self.goal = np.zeros(2, np.float32); self.steps = 0; self.prev_dist = 0.
        self.action_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (35,), dtype=np.float32)

    def _obs(self):
        p,v = self.sim.pose[0], self.sim.velocity[0]
        relgoal = self.goal-p[0,:2]
        obs = np.concatenate([p[0],v[0],p[1],v[1],relgoal,[self.goal_radius], [self.sim.field_length,self.sim.field_width],
                              self.sim.length[0],self.sim.width[0],self.sim.accel[0]/10.,
                              self._obstacle_features()]).astype(np.float32)
        normalize_numpy_observation_in_place(obs,self.sim.field_length,self.sim.field_width,
                                             self.sim.speed[0],self.sim.omega_limit[0])
        if self.noise_level: obs += self.rng.normal(0,self.noise_level,obs.shape).astype(np.float32)
        if self.dropout_probability and self.rng.random() < self.dropout_probability: obs[:12] = 0
        return obs

    def _obstacle_features(self):
        from .field import box_observation_radius
        from .field import static_collision_boxes, bump_boxes
        feature_boxes=static_collision_boxes(self.field_boxes)+bump_boxes(self.field_boxes)
        obstacles=list(self.sim.obstacles)+[(box.x,box.y,box_observation_radius(box)) for box in feature_boxes]
        nearby = sorted(obstacles,
                        key=lambda o:(o[0]-self.sim.pose[0,0,0])**2+(o[1]-self.sim.pose[0,0,1])**2)[:4]
        data = np.zeros((4,3),np.float32)
        for index, obstacle in enumerate(nearby): data[index] = obstacle
        return data.reshape(-1)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.py_rng = random.Random(seed)
        self.sim.reset(seed=seed)
        if self.sim.randomize:
            self.sim.velocity[:,:,:2] = self.rng.uniform(-.25,.25,self.sim.velocity[:,:,:2].shape)*self.sim.speed[:,:,None]
            self.sim.velocity[:,:,2] = self.rng.uniform(-.2,.2,self.sim.velocity[:,:,2].shape)*self.sim.omega_limit
        self.steps = 0
        self.path_length = self.contact_time = self.blocked_time = 0.0
        self.contact_count = 0; self.last_contact = False; self.last_action.fill(0)
        self.noise_level = (self.rng.uniform(0,.02) if self.randomize_observation and self.sim.randomize
                            else self.noise)
        self.dropout_probability = (self.rng.uniform(0,.05) if self.randomize_observation and self.sim.randomize
                                   else self.dropout)
        if self.randomize_latency:
            self.control_delay = int(self.rng.integers(0, max(1, round(.12/self.dt)+1)))
            self.observation_delay = int(self.rng.integers(0, max(1, round(.16/self.dt)+1)))
        else:
            self.control_delay = max(0, round(self.control_latency/self.dt))
            self.observation_delay = max(0, round(self.observation_latency/self.dt))
        if options and "goal" in options:
            self.goal = np.asarray(options["goal"], np.float32)
        else:
            self.goal = np.array([self.rng.uniform(1,self.sim.field_length-1), self.rng.uniform(1,self.sim.field_width-1)],np.float32)
        if options and "goal_radius" in options:
            self.goal_radius=float(options["goal_radius"])
        elif self.sim.randomize:
            self.goal_radius=float(self.rng.uniform(.3,.8))
        if not np.isfinite(self.goal_radius) or self.goal_radius <= 0:
            raise ValueError("goal_radius must be finite and positive")
        start = self.sim.pose[0,0 if self.mode == "counter" else 1,:2]
        self.prev_dist = float(np.linalg.norm(self.goal-start))
        self.initial_dist = self.prev_dist
        initial = self._obs()
        self.action_history = [np.zeros(3, np.float32) for _ in range(self.control_delay)]
        self.observation_history = [initial.copy() for _ in range(self.observation_delay)]
        return initial, {"goal": self.goal.copy(), "goal_radius":self.goal_radius,
                         "control_delay_steps": self.control_delay,
                         "observation_delay_steps": self.observation_delay}

    def step(self, action):
        action = np.nan_to_num(np.asarray(action,np.float32),nan=0.,posinf=0.,neginf=0.).reshape(3)
        action = np.clip(action,-1,1)
        self.action_history.append(action.copy())
        action = self.action_history.pop(0)
        own_cmd = action.copy()
        own_cmd[:2] *= self.sim.speed[0,0]
        own_cmd[2] *= self.sim.omega_limit[0,0]
        p = self.sim.pose[0]
        # Policies stay in chassis-speed space; the simulator handles swerve/motor response.
        kind = self.opponent.get("value", "guard") if isinstance(self.opponent, dict) else self.opponent
        if not isinstance(kind, str): kind = "guard"
        from .training import SCRIPTED_OPPONENTS
        from .types import Objective, RobotParameters, RobotState
        if self.mode == "defense" and kind not in ("random", "noisy", "offense"):
            kind = "offense"
        controller = SCRIPTED_OPPONENTS.get(kind, SCRIPTED_OPPONENTS["offense"])()
        other = RobotState(*map(float,p[0]),*map(float,self.sim.velocity[0,0]))
        opponent_state = RobotState(*map(float,p[1]),*map(float,self.sim.velocity[0,1]))
        params = RobotParameters(length=float(self.sim.length[0,1]), width=float(self.sim.width[0,1]),
                                 max_speed=float(self.sim.speed[0,1]),
                                 max_acceleration=float(self.sim.accel[0,1]),
                                 max_omega=float(self.sim.omega_limit[0,1]),
                                 max_alpha=float(self.sim.alpha[0,1]),mass=float(self.sim.mass[0,1]))
        opp_cmd = controller.command(opponent_state,other,Objective(*map(float,self.goal),self.goal_radius),params,self.py_rng)
        opp = np.asarray([opp_cmd.vx,opp_cmd.vy,opp_cmd.omega],np.float32)
        commands = np.zeros((1,2,3),np.float32); commands[0,0] = own_cmd; commands[0,1] = opp
        oldown = p[0,:2].copy(); oldopp = p[1,:2].copy()
        old_dist = float(np.linalg.norm(self.goal - (oldown if self.mode == "counter" else oldopp)))
        self.sim.step(commands); self.steps += 1
        newown = self.sim.pose[0,0,:2]; newopp = self.sim.pose[0,1,:2]
        if self.mode == "counter":
            d = float(np.linalg.norm(self.goal-newown)); reward = (self.prev_dist-d)*2. - .01
            terminated = d < self.goal_radius; reward += 10. if terminated else 0.
            self.prev_dist = d
        else:
            d = float(np.linalg.norm(self.goal-newopp)); reward = (d-self.prev_dist)*2. - .01
            terminated = d < self.goal_radius; reward += -10. if terminated else 0.
            self.prev_dist = d
            route = self.goal-newopp
            route_sq = float(route@route)
            projection = 0.0 if route_sq < 1e-8 else float(np.clip((newown-newopp)@route/route_sq,0.,1.))
            closest = newopp+projection*route
            lane_distance = float(np.linalg.norm(newown-closest))
            position_score = float(np.exp(-lane_distance/.7)*np.exp(-((projection-.55)/.35)**2))
            reward += .04*position_score
        # Penalize sustained contact, invalid/outside positioning, and abrupt controls.
        contact = bool(self.sim.robot_contact[0])
        reward -= .05 if contact else 0.
        moved = float(np.linalg.norm(newown-oldown)); self.path_length += moved
        contact_started = contact and not self.last_contact
        if contact:
            self.contact_time += self.dt
            if contact_started: self.contact_count += 1
        self.last_contact = bool(contact)
        new_dist = float(np.linalg.norm(self.goal - (newown if self.mode == "counter" else newopp)))
        if old_dist-new_dist < .002: self.blocked_time += self.dt
        action_delta = action-self.last_action
        smoothness = float(np.linalg.norm(action_delta)); self.last_action = action.copy()
        spin_rate_ratio = float(self.sim.velocity[0,0,2] / max(float(self.sim.omega_limit[0,0]), .1))
        maneuver_penalty = (SPIN_RATE_PENALTY * spin_rate_ratio**2 +
                            ROTATION_COMMAND_PENALTY * float(action[2])**2 +
                            ROTATION_COMMAND_DELTA_PENALTY * float(action_delta[2])**2)
        reward -= .001 * smoothness + maneuver_penalty
        truncated = self.steps >= self.horizon
        info = {"contact": float(contact), "contact_duration": self.dt if contact else 0.0,
                "contact_count": float(contact_started), "time_blocked": self.dt if old_dist-new_dist < .002 else 0.0,
                "time_to_goal": self.steps*self.dt if terminated else 0.0,
                "success": float(terminated and self.mode == "counter"),
                "defensive_delay": self.dt if self.mode == "defense" else 0.0,
                "useful_position": position_score if self.mode == "defense" else 0.0,
                "path_length": moved, "command_smoothness": smoothness,
                "spin_rate_ratio": abs(spin_rate_ratio), "maneuver_penalty": maneuver_penalty,
                "path_efficiency": (self.initial_dist / max(self.path_length, 1e-6)) if terminated else 0.0,
                "out_of_bounds": 0.0}
        observation = self._obs()
        self.observation_history.append(observation)
        observation = self.observation_history.pop(0)
        return observation, float(reward), bool(terminated), bool(truncated), info


DefenseEnv = CounterDefenseEnv


def make_env(task: str = "counter_defense", opponent: Any = "guard", seed: int | None = None, **kwargs):
    """Factory used by the optional trainer; task names follow the CLI/API."""
    modes = {"counter_defense": "counter", "defense": "defense", "counter": "counter"}
    if task not in modes:
        raise ValueError("task must be 'counter_defense' or 'defense'")
    return CounterDefenseEnv(mode=modes[task], opponent=opponent, seed=seed, **kwargs)
