"""Fully tensorized, device-resident FRC defense simulator (optional PyTorch).

All per-world state and hot-path calculations stay on ``device``. This model
uses wheel-level swerve forces and motor/current limits. Bump traversal adds a
quasi-3D triangular height profile, chassis pitch/load transfer, and gravity on
the ramp grade. Suspension, chassis roll/bounce, wheel lift, CAN timing, and a
full electrical network are not resolved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

from .field import (ALLIANCE_ZONE_DEPTH, BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE,
                    BUMP_ROBOT_CG_HEIGHT, BUMP_ROLLING_RESISTANCE)
from .observation import normalize_tensor_observation_batch_in_place
from .reward import (ROTATION_COMMAND_DELTA_PENALTY, ROTATION_COMMAND_PENALTY,
                     SPIN_RATE_PENALTY)

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None


def _require_torch():
    if torch is None:
        raise RuntimeError("Tensor simulation requires PyTorch; install torch first")


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


class TensorVectorizedSimulator:
    """Batched two-robot simulator; every state tensor has leading world N."""
    def __init__(self, num_envs=1, device="cuda", seed=0, dt=.02,
                 field_length=16.54, field_width=8.21, robot_length=.9,
                 robot_width=.9, max_speed=4.5, max_acceleration=8.,
                 max_omega=8., max_alpha=18., mass=55., bumper_friction=.65,
                 field_friction=.7, lateral_friction=1.2,
                 yaw_inertia_multiplier=1.5, current_limit=None, swerve=None,
                 obstacles=(), field_colliders=(), bump_regions=(), contact_iterations=3, randomize=False, **kwargs):
        _require_torch()
        if num_envs < 1 or dt <= 0: raise ValueError("num_envs and dt must be positive")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested accelerator {device!r} is unavailable; refusing CPU simulation")
        self.n, self.device, self.dt = int(num_envs), requested, float(dt)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed or 0))
        self.field_length, self.field_width = float(field_length), float(field_width)
        self.contact_iterations = max(1, int(contact_iterations))
        self.swerve = swerve or TensorSwerveParameters()
        self.randomize = bool(randomize)
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
        self.pose = torch.zeros((self.n,2,3), device=self.device)
        self.velocity = torch.zeros_like(self.pose)
        self.length, self.width, self.mass = (full((self.n,2), v) for v in (robot_length,robot_width,mass))
        self.speed, self.accel, self.omega_limit, self.alpha = (full((self.n,2), v) for v in (max_speed,max_acceleration,max_omega,max_alpha))
        self.mu, self.ground_mu, self.wall_mu = (full((self.n,), v) for v in
                                                (bumper_friction, field_friction, field_friction))
        self.lateral_mu=full((self.n,2),lateral_friction)
        self.yaw_inertia_multiplier=full((self.n,2),yaw_inertia_multiplier)
        self.drive_ratio = full((self.n,2),self.swerve.drive_ratio)
        self.drive_current_limit = full((self.n,2),current_limit or self.swerve.drive_current_limit)
        self.drive_supply_limit=full((self.n,2),self.swerve.drive_supply_current_limit)
        self.robot_supply_limit=full((self.n,2),self.swerve.robot_supply_current_limit)
        self.battery_resistance=full((self.n,2),self.swerve.battery_internal_resistance)
        self.module_angle=torch.zeros((self.n,2,4),device=self.device)
        self.module_steer_rate=torch.zeros_like(self.module_angle)
        self.module_drive_speed=torch.zeros_like(self.module_angle)
        self.module_current=torch.zeros_like(self.module_angle)
        self.module_supply_current=torch.zeros_like(self.module_angle)
        self.robot_current=torch.zeros((self.n,2),device=self.device)
        self.robot_contact=torch.zeros((self.n,),device=self.device,dtype=torch.bool)
        self.opponent_contact=torch.zeros((self.n,),device=self.device,dtype=torch.bool)
        self.field_contact=torch.zeros((self.n,2),device=self.device,dtype=torch.bool)
        self.wall_contact=torch.zeros((self.n,2,2),device=self.device,dtype=torch.bool)
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
        x=torch.rand((self.n,2),device=self.device,generator=self.generator)
        y=torch.rand((self.n,2),device=self.device,generator=self.generator)
        a=torch.rand((self.n,2),device=self.device,generator=self.generator)
        margin=.9
        self.pose[:,:,0]=torch.where(mask[:,None],margin+x*(self.field_length-2*margin),self.pose[:,:,0])
        self.pose[:,:,1]=torch.where(mask[:,None],margin+y*(self.field_width-2*margin),self.pose[:,:,1])
        self.pose[:,:,2]=torch.where(mask[:,None],(2*a-1)*math.pi,self.pose[:,:,2])
        self._field_collision()
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
            return float(base)*(lo+torch.rand(shape,device=self.device,generator=self.generator)*(hi-lo))
        pair=(self.n,2)
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
        self.accel.copy_(torch.minimum(self.accel, torch.minimum(motor_accel, tire_accel)))

    def step(self, command):
        cmd=torch.as_tensor(command,device=self.device,dtype=self.pose.dtype)
        if cmd.shape==(2,3): cmd=cmd.expand(self.n,2,3)
        if tuple(cmd.shape)!=(self.n,2,3): raise ValueError(f"command must be {(self.n,2,3)}")
        cmd=torch.nan_to_num(cmd)
        self._swerve(cmd)
        self.pose.add_(self.velocity*self.dt)
        self.pose[...,2].copy_(torch.remainder(self.pose[...,2]+math.pi,2*math.pi)-math.pi)
        self.robot_contact.zero_()
        self.opponent_contact.zero_()
        self.field_contact.zero_()
        self.wall_contact.zero_()
        for _ in range(self.contact_iterations):
            self._robot_collision(); self._walls(); self._obstacle_collision(); self._field_collision()
        return self.state

    def _swerve(self,cmd):
        p=self.swerve; th=self.pose[...,2]; c,s=th.cos(),th.sin()
        speed_limit,accel_limit=self.speed,self.accel
        mx=torch.stack((-torch.ones_like(self.length)*self.swerve.module_x_offset,
                        torch.ones_like(self.length)*self.swerve.module_x_offset,
                        -torch.ones_like(self.length)*self.swerve.module_x_offset,
                        torch.ones_like(self.length)*self.swerve.module_x_offset),-1)
        my=torch.stack((-torch.ones_like(self.width)*self.swerve.module_y_offset,
                        -torch.ones_like(self.width)*self.swerve.module_y_offset,
                        torch.ones_like(self.width)*self.swerve.module_y_offset,
                        torch.ones_like(self.width)*self.swerve.module_y_offset),-1)
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
        norm=torch.sqrt(vx.square()+vy.square()).clamp_min(1e-8); scale=(speed_limit/norm).clamp(max=1)
        vx,vy=vx*scale,vy*scale; omega=cmd[...,2].clamp(-self.omega_limit,self.omega_limit)
        dx=vx[...,None]-omega[...,None]*my; dy=vy[...,None]+omega[...,None]*mx
        target=torch.sqrt(dx.square()+dy.square()); target*= (speed_limit[...,None]/target.amax(-1,keepdim=True).clamp_min(1e-8)).clamp(max=1)
        angle=torch.atan2(dy,dx); delta=torch.remainder(angle-self.module_angle+math.pi,2*math.pi)-math.pi
        reverse=delta.abs()>math.pi/2; angle=torch.where(reverse,angle+math.pi,angle); target=torch.where(reverse,-target,target)
        angle=torch.remainder(angle+math.pi,2*math.pi)-math.pi
        delta=torch.remainder(angle-self.module_angle+math.pi,2*math.pi)-math.pi
        steer_target=(8*delta).clamp(-p.steer_rate_limit,p.steer_rate_limit)
        se=steer_target-self.module_steer_rate
        steer_current=(2*se.abs()).clamp(max=p.steer_current_limit)
        self.module_steer_rate.add_(se.sign()*p.steer_acceleration*steer_current/max(p.steer_current_limit,1e-6)*self.dt).clamp_(-p.steer_rate_limit,p.steer_rate_limit)
        self.module_angle.copy_(torch.remainder(self.module_angle+self.module_steer_rate*self.dt+math.pi,2*math.pi)-math.pi)
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
        self.module_current.copy_(amps); self.module_supply_current.copy_(supply)
        self.robot_current.copy_(supply.sum(-1)+steer_current.sum(-1))
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
        self.velocity[...,0].add_((ca*ax-sa*ay)*self.dt)
        self.velocity[...,1].add_((sa*ax+ca*ay)*self.dt)
        mean_grade=wheel_grade.mean(-1)
        rolling_accel=(BUMP_ROLLING_RESISTANCE*normal*on_bump).sum(-1)/self.mass
        self.velocity[...,0].add_((-9.81*mean_grade-rolling_accel*self.velocity[...,0].sign())*self.dt)
        vnorm=self.velocity[...,:2].norm(dim=-1).clamp_min(1e-8)
        self.velocity[...,:2].mul_((speed_limit/vnorm).clamp(max=1)[...,None])
        self.velocity[...,2].add_(az.clamp(-self.alpha,self.alpha)*self.dt).clamp_(-self.omega_limit,self.omega_limit)
        bodyx=ca*self.velocity[...,0]+sa*self.velocity[...,1]; bodyy=-sa*self.velocity[...,0]+ca*self.velocity[...,1]
        self.module_drive_speed.copy_((bodyx[...,None]-self.velocity[...,2,None]*my)*self.module_angle.cos()+(bodyy[...,None]+self.velocity[...,2,None]*mx)*self.module_angle.sin())

    def _robot_collision(self):
        # Vectorized SAT over four face normals for the pair of oriented rectangles.
        pos=self.pose[:,:,:2]; th=self.pose[:,:,2]; c,s=th.cos(),th.sin()
        axes=torch.stack((torch.stack((c[:,0],s[:,0]),-1),torch.stack((-s[:,0],c[:,0]),-1),torch.stack((c[:,1],s[:,1]),-1),torch.stack((-s[:,1],c[:,1]),-1)),1)
        u=torch.stack((c,s),-1); v=torch.stack((-s,c),-1)
        # axis [N,4,2], robot basis [N,2,2], projected support radii
        proj_u=torch.einsum('nkd,nrd->nkr',axes,u).abs(); proj_v=torch.einsum('nkd,nrd->nkr',axes,v).abs()
        radii=(proj_u*self.length[:,None,:]/2+proj_v*self.width[:,None,:]/2).sum(-1)
        delta=pos[:,1]-pos[:,0]; signed=(axes*delta[:,None,:]).sum(-1)
        overlap=radii-signed.abs(); depth,k=overlap.max(-1)  # replaced below with minimum positive penetration axis
        valid=overlap.min(-1).values>0
        # SAT penetration is the smallest overlap, not deepest.
        depth,k=overlap.min(-1)
        normal=torch.gather(axes,1,k[:,None,None].expand(-1,1,2)).squeeze(1)
        normal*=torch.where(torch.gather(signed,1,k[:,None]).squeeze(1)>=0,1.,-1.)[:,None]
        corr=(depth.clamp_min(0)+1e-4)*valid
        inv=1/self.mass; total=inv.sum(-1)
        pos[:,0]-=normal*corr[:,None]*(inv[:,0]/total)[:,None]
        pos[:,1]+=normal*corr[:,None]*(inv[:,1]/total)[:,None]
        # Contact impulse at midpoint supports; include angular effective mass and Coulomb friction.
        support0=(pos[:,0]+torch.sign((normal*u[:,0]).sum(-1))[:,None]*u[:,0]*self.length[:,0,None]/2+
                  torch.sign((normal*v[:,0]).sum(-1))[:,None]*v[:,0]*self.width[:,0,None]/2)
        support1=(pos[:,1]-torch.sign((normal*u[:,1]).sum(-1))[:,None]*u[:,1]*self.length[:,1,None]/2-
                  torch.sign((normal*v[:,1]).sum(-1))[:,None]*v[:,1]*self.width[:,1,None]/2)
        cp=.5*(support0+support1); r0=cp-pos[:,0]; r1=cp-pos[:,1]
        vel=self.velocity
        cv0=vel[:,0,:2]+vel[:,0,2,None]*torch.stack((-r0[:,1],r0[:,0]),-1)
        cv1=vel[:,1,:2]+vel[:,1,2,None]*torch.stack((-r1[:,1],r1[:,0]),-1)
        rel=cv1-cv0; vn=(rel*normal).sum(-1)
        I=(self.mass*(self.length.square()+self.width.square())/12)*self.yaw_inertia_multiplier
        cross=lambda a,b:a[:,0]*b[:,1]-a[:,1]*b[:,0]
        rn0=cross(r0,normal); rn1=cross(r1,normal)
        eff=inv.sum(-1)+rn0.square()/I[:,0]+rn1.square()/I[:,1]
        jn=torch.where((valid)&(vn<0),-(1.05*vn)/eff.clamp_min(1e-8),0.)
        normal_impulse=jn[:,None]*normal
        self.velocity[:,0,:2]-=normal_impulse*inv[:,0,None]; self.velocity[:,1,:2]+=normal_impulse*inv[:,1,None]
        self.velocity[:,0,2]-=jn*rn0/I[:,0]; self.velocity[:,1,2]+=jn*rn1/I[:,1]
        cv0=self.velocity[:,0,:2]+self.velocity[:,0,2,None]*torch.stack((-r0[:,1],r0[:,0]),-1)
        cv1=self.velocity[:,1,:2]+self.velocity[:,1,2,None]*torch.stack((-r1[:,1],r1[:,0]),-1)
        rel=cv1-cv0
        tangent=torch.stack((-normal[:,1],normal[:,0]),-1); vt=(rel*tangent).sum(-1)
        rt0=cross(r0,tangent); rt1=cross(r1,tangent)
        efft=inv.sum(-1)+rt0.square()/I[:,0]+rt1.square()/I[:,1]
        jt=(-vt/efft.clamp_min(1e-8)).clamp(-self.mu*jn,self.mu*jn)
        impulse=jt[:,None]*tangent
        self.velocity[:,0,:2]-=impulse*inv[:,0,None]; self.velocity[:,1,:2]+=impulse*inv[:,1,None]
        self.velocity[:,0,2]-=jt*rt0/I[:,0]; self.velocity[:,1,2]+=jt*rt1/I[:,1]
        self.robot_contact|=valid
        self.opponent_contact|=valid

    def _apply_static_contact_impulse(self,point,normal,active,friction):
        """Rigid-body normal and Coulomb-friction impulse against a fixed surface."""
        lever=point-self.pose[...,:2]
        inertia=(self.mass*(self.length.square()+self.width.square())/12)*self.yaw_inertia_multiplier
        cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
        arm=torch.stack((-lever[...,1],lever[...,0]),-1)
        contact_velocity=self.velocity[...,:2]+self.velocity[...,2,None]*arm
        vn=(contact_velocity*normal).sum(-1)
        inv_mass=self.mass.reciprocal()
        rn=cross(lever,normal)
        effective=inv_mass+rn.square()/inertia.clamp_min(1e-8)
        jn=torch.where(active&(vn<0),-1.05*vn/effective.clamp_min(1e-8),0.)
        impulse=jn[...,None]*normal
        self.velocity[...,:2].add_(impulse*inv_mass[...,None])
        self.velocity[...,2].add_(cross(lever,impulse)/inertia.clamp_min(1e-8))
        tangent=torch.stack((-normal[...,1],normal[...,0]),-1)
        contact_velocity=self.velocity[...,:2]+self.velocity[...,2,None]*arm
        vt=(contact_velocity*tangent).sum(-1)
        rt=cross(lever,tangent)
        effective_tangent=inv_mass+rt.square()/inertia.clamp_min(1e-8)
        jt=(-vt/effective_tangent.clamp_min(1e-8)).maximum(-friction*jn).minimum(friction*jn)
        impulse=jt[...,None]*tangent
        self.velocity[...,:2].add_(impulse*inv_mass[...,None])
        self.velocity[...,2].add_(cross(lever,impulse)/inertia.clamp_min(1e-8))

    def _walls(self):
        theta=self.pose[...,2]; c,s=theta.cos(),theta.sin()
        hl,hw=self.length/2,self.width/2
        hx=c.abs()*hl+s.abs()*hw; hy=s.abs()*hl+c.abs()*hw
        x,y=self.pose[...,0],self.pose[...,1]
        walls=((0,hx-x,1.),(0,x+hx-self.field_length,-1.),
               (1,hy-y,1.),(1,y+hy-self.field_width,-1.))
        u=torch.stack((c,s),-1); v=torch.stack((-s,c),-1)
        for axis,penetration,sign in walls:
            normal=torch.zeros_like(self.pose[...,:2]); normal[...,axis]=sign
            active=penetration>0
            self.pose[...,:2].add_(torch.where(active[...,None],normal*(penetration.clamp_min(0)[...,None]+1e-4),
                                                torch.zeros_like(normal)))
            toward_wall=-normal
            point=(self.pose[...,:2]+torch.sign((toward_wall*u).sum(-1))[...,None]*u*hl[...,None]+
                   torch.sign((toward_wall*v).sum(-1))[...,None]*v*hw[...,None])
            self._apply_static_contact_impulse(point,normal,active,self.wall_mu[:,None])
            self.wall_contact[:,:,axis]|=active

    def _obstacle_collision(self):
        if self.obstacles.shape[0]==0:return
        pos=self.pose[...,:2]; theta=self.pose[...,2]; c,s=theta.cos(),theta.sin()
        u=torch.stack((c,s),-1); v=torch.stack((-s,c),-1)
        relative=self.obstacles[None,None,:,:2]-pos[:,:,None,:]
        local_x=(relative*u[:,:,None,:]).sum(-1); local_y=(relative*v[:,:,None,:]).sum(-1)
        hl=self.length/2; hw=self.width/2
        closest_x=torch.maximum(torch.minimum(local_x,hl[...,None]),-hl[...,None])
        closest_y=torch.maximum(torch.minimum(local_y,hw[...,None]),-hw[...,None])
        delta_x=local_x-closest_x; delta_y=local_y-closest_y
        distance=torch.sqrt(delta_x.square()+delta_y.square())
        outside=distance>1e-8
        normal_x_out=-delta_x/distance.clamp_min(1e-8)
        normal_y_out=-delta_y/distance.clamp_min(1e-8)
        gap_x=hl[...,None]-local_x.abs(); gap_y=hw[...,None]-local_y.abs()
        use_x=gap_x<=gap_y
        inside_x=torch.where(local_x>=0,-torch.ones_like(local_x),torch.ones_like(local_x))
        inside_y=torch.where(local_y>=0,-torch.ones_like(local_y),torch.ones_like(local_y))
        normal_x=torch.where(outside,normal_x_out,use_x.to(local_x.dtype)*inside_x)
        normal_y=torch.where(outside,normal_y_out,(~use_x).to(local_y.dtype)*inside_y)
        radius=self.obstacles[:,2][None,None,:]
        penetration=torch.where(outside,radius-distance,radius+torch.minimum(gap_x,gap_y)).clamp_min(0.)
        depth,k=penetration.max(-1); contact=depth>0
        gather=k[...,None]
        nx=normal_x.gather(-1,gather).squeeze(-1); ny=normal_y.gather(-1,gather).squeeze(-1)
        normal=nx[...,None]*u+ny[...,None]*v
        circle_center=self.obstacles[:,:2][k]
        circle_radius=self.obstacles[:,2][k]
        pos.add_(torch.where(contact[...,None],normal*(depth[...,None]+1e-4),torch.zeros_like(pos)))
        toward_obstacle=-normal
        robot_point=(pos+torch.sign((toward_obstacle*u).sum(-1))[...,None]*u*hl[...,None]+
                     torch.sign((toward_obstacle*v).sum(-1))[...,None]*v*hw[...,None])
        circle_point=circle_center+normal*circle_radius[...,None]
        point=.5*(robot_point+circle_point)
        self._apply_static_contact_impulse(point,normal,contact,self.wall_mu[:,None])
        self.robot_contact|=contact.any(-1)
        self.field_contact|=contact

    def _field_collision(self):
        """Resolve the most penetrating oriented-robot/field-box pair per robot."""
        if not self.field_colliders.shape[0]: return
        boxes=self.field_colliders; center,half=boxes[:,:2],boxes[:,2:]
        theta=self.pose[...,2]; c,s=theta.cos(),theta.sin(); ac,ass=c.abs(),s.abs()
        axes=torch.stack((torch.stack((torch.ones_like(c),torch.zeros_like(c)),-1),
            torch.stack((torch.zeros_like(c),torch.ones_like(c)),-1),
            torch.stack((c,s),-1),torch.stack((-s,c),-1)),-2)
        delta=self.pose[:,:,None,:2]-center[None,None,:,:]
        signed=(axes[:,:,None,:,:]*delta[:,:,:,None,:]).sum(-1)
        hl,hw=self.length/2,self.width/2
        robot_r=torch.stack((ac*hl+ass*hw,ass*hl+ac*hw,hl.expand_as(c),hw.expand_as(c)),-1)
        box_r=torch.stack((half[None,None,:,0].expand_as(signed[:,:,:,0]),
            half[None,None,:,1].expand_as(signed[:,:,:,0]),
            half[None,None,:,0]*ac[:,:,None]+half[None,None,:,1]*ass[:,:,None],
            half[None,None,:,0]*ass[:,:,None]+half[None,None,:,1]*ac[:,:,None]),-1)
        penetration=robot_r[:,:,None,:]+box_r-signed.abs()
        box_depth,_=penetration.min(-1)
        valid=box_depth>=0
        depth,box_index=torch.where(valid,box_depth,torch.full_like(box_depth,-1.)).max(-1)
        contact=depth>=0
        box_index=box_index.clamp_min(0)
        candidate_penetration=penetration.gather(2,box_index[...,None,None].expand(-1,-1,1,4)).squeeze(2)
        candidate_signed=signed.gather(2,box_index[...,None,None].expand(-1,-1,1,4)).squeeze(2)
        candidate_normals=axes*torch.where(candidate_signed>=0,1.,-1.)[...,None]
        approach=(candidate_normals*self.velocity[...,:2].unsqueeze(-2)).sum(-1)
        tied=(candidate_penetration<=depth[...,None]+1e-6)&(candidate_penetration>=0)
        approaching=torch.where(tied,approach,torch.full_like(approach,float('inf')))
        approach_speed,approach_axis=approaching.min(-1)
        axis_index=torch.where(approach_speed < -1e-6,approach_axis,candidate_penetration.argmin(-1))
        normal=candidate_normals.gather(-2,axis_index[...,None,None].expand(-1,-1,1,2)).squeeze(-2)
        depth=depth.clamp_min(0)
        self.pose[...,:2].add_(torch.where(contact[...,None],normal*(depth[...,None]+1e-4),torch.zeros_like(normal)))
        selected_center=center[box_index]; selected_half=half[box_index]
        u=torch.stack((c,s),-1); v=torch.stack((-s,c),-1)
        tangent=torch.stack((-normal[...,1],normal[...,0]),-1)
        robot_normal=(normal*u).sum(-1).abs()*hl+(normal*v).sum(-1).abs()*hw
        box_normal=normal.abs().mul(selected_half).sum(-1)
        normal_coordinate=.5*((normal*self.pose[...,:2]).sum(-1)-robot_normal+
                               (normal*selected_center).sum(-1)+box_normal)
        robot_tangent=(tangent*u).sum(-1).abs()*hl+(tangent*v).sum(-1).abs()*hw
        box_tangent=tangent.abs().mul(selected_half).sum(-1)
        robot_center=(tangent*self.pose[...,:2]).sum(-1)
        box_center=(tangent*selected_center).sum(-1)
        overlap_low=torch.maximum(robot_center-robot_tangent,box_center-box_tangent)
        overlap_high=torch.minimum(robot_center+robot_tangent,box_center+box_tangent)
        tangent_coordinate=.5*(overlap_low+overlap_high)
        point=normal*normal_coordinate[...,None]+tangent*tangent_coordinate[...,None]
        self._apply_static_contact_impulse(point,normal,contact,self.wall_mu[:,None])
        self.robot_contact|=contact.any(-1)
        self.field_contact|=contact


class TensorDefenseEnv:
    """Vectorized FRC REBUILT offense/defense environment."""
    def __init__(self,num_envs=1,task="counter_defense",device="cuda",seed=0,opponent="guard",**kwargs):
        _require_torch()
        if task not in ("counter_defense","defense"): raise ValueError("task must be counter_defense or defense")
        # Training/evaluation can load the exact robot configuration without
        # changing the tensor hot path. A hardware config defaults to nominal
        # physics; explicitly set "randomize": true for sim-to-real variation.
        config_path = os.environ.get("FRC_DRIVETRAIN_CONFIG")
        drivetrain_config = None
        if config_path:
            config = json.loads(Path(config_path).read_text())
            drivetrain_config = config
            kwargs.setdefault("randomize", config.get("randomize", False))
            for name in ("robot_length", "robot_width", "mass", "max_speed",
                         "max_acceleration", "max_omega", "max_alpha",
                         "field_friction", "lateral_friction"):
                if name in config:
                    kwargs.setdefault(name, config[name])
            kwargs.setdefault("swerve", TensorSwerveParameters(**config.get("swerve", {})))
        self.n,self.device,self.task,self.opponent=int(num_envs),torch.device(device),task,opponent
        self.action_mode=str(kwargs.pop("action_mode","direct")).lower()
        if self.action_mode not in ("direct","tactical","strategic"):
            raise ValueError("action_mode must be direct, tactical, or strategic")
        self.action_dim=2 if self.action_mode=="tactical" else 3
        self.obs_dim={"direct":35,"tactical":38,"strategic":89}[self.action_mode]
        self.learned_opponent_fn=kwargs.pop("learned_opponent_fn",None)
        self.fuel_count=int(kwargs.pop("fuel_count",504))
        preload_setting=kwargs.pop("preloaded_per_robot",None)
        self.preloaded_per_robot=None if preload_setting is None else int(preload_setting)
        if not 96+6*(self.preloaded_per_robot or 0) <= self.fuel_count <= 600:
            raise ValueError("fuel_count must be 96 + 6*preloaded_per_robot through 600")
        if self.preloaded_per_robot is not None and not 0<=self.preloaded_per_robot<=8:
            raise ValueError("preloaded_per_robot must be between 0 and 8")
        self.dt=float(kwargs.pop("dt",.02)); self.horizon=int(kwargs.pop("horizon",750)); self.base_goal_radius=float(kwargs.pop("goal_radius",.5))
        self.match_clock_step=160./max(self.horizon,1) if self.action_mode=="strategic" else self.dt
        self.adstar_replan_interval=max(1,int(kwargs.pop("adstar_replan_interval",20)))
        self._planner_tick=0
        self.adstar_spawn_hint=bool(kwargs.pop("adstar_spawn_hint",True))
        self.randomize=bool(kwargs.pop("randomize",True))
        self.static_opponent_fraction=float(kwargs.pop("static_opponent_fraction",0.))
        if not 0. <= self.static_opponent_fraction <= 1.:
            raise ValueError("static_opponent_fraction must be between 0 and 1")
        self.normalize_observations=bool(kwargs.pop("normalize_observations",True))
        self.observation_noise=float(kwargs.pop("observation_noise",.01))
        self.observation_dropout=float(kwargs.pop("observation_dropout",.02))
        self.max_control_latency_steps=max(0,int(kwargs.pop("max_control_latency_steps",6)))
        self.max_observation_latency_steps=max(0,int(kwargs.pop("max_observation_latency_steps",8)))
        field_layout=kwargs.pop("field_layout","2026_rebuilt")
        field_colliders=kwargs.pop("field_colliders",None)
        self.field_boxes=()
        self.field_feature_boxes=()
        if field_colliders is None and field_layout=="2026_rebuilt":
            from .field import bump_boxes, rebuilt_field, static_collision_boxes
            self.field_boxes=rebuilt_field()
            solids=static_collision_boxes(self.field_boxes)
            self.field_feature_boxes=solids+bump_boxes(self.field_boxes)
            field_colliders=[box.as_tensor() for box in solids]
            kwargs.setdefault("bump_regions",[box.as_tensor() for box in bump_boxes(self.field_boxes)])
            kwargs.setdefault("field_length",16.54); kwargs.setdefault("field_width",8.07)
        elif field_colliders is not None:
            field_colliders=list(field_colliders)
        self.sim=TensorVectorizedSimulator(self.n,device,seed,dt=self.dt,obstacles=kwargs.pop("obstacles",()),
            field_colliders=field_colliders or (),**kwargs)
        self.sim.randomize=self.randomize
        if drivetrain_config is None:
            drivetrain_config = {"name": "built-in illustrative defaults",
                "randomize": self.randomize,
                "mass": 55.0, "robot_length": .9, "robot_width": .9,
                "max_speed": 4.5, "max_acceleration": 8.0,
                "max_omega": 8.0, "max_alpha": 18.0,
                "swerve": asdict(self.sim.swerve)}
        self.drivetrain_config = drivetrain_config
        self.device=self.sim.device
        self._adstar_planners=None
        self._adstar_defender_planners=None
        self._adstar_tactical_planner=None
        self._adstar_contact_latched=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        opponent_kind=opponent.get("value","") if isinstance(opponent,dict) else opponent
        if self.task=="defense" and (self.action_mode=="strategic" or
                isinstance(opponent_kind,str) and opponent_kind.lower()=="adstar"):
            from .tensor_adstar import TensorADStar
            self._adstar_planners=TensorADStar(self)
        elif (self.task=="counter_defense" and (self.action_mode=="strategic" or
              isinstance(opponent_kind,str) and opponent_kind.lower() in ("guard", "adstar_defender"))):
            from .tensor_adstar import TensorADStar
            self._adstar_defender_planners=TensorADStar(self,avoid_bumps=True)
        if self.action_mode in ("tactical","strategic"):
            from .tensor_adstar import TensorADStar
            self._adstar_tactical_planner=TensorADStar(self,avoid_bumps=True)
        if self.field_feature_boxes:
            from .field import box_observation_radius
            feature_values=[(b.x,b.y,box_observation_radius(b)) for b in self.field_feature_boxes]
            self.field_feature_obstacles=torch.tensor(feature_values,device=self.device,dtype=torch.float32)
        else:
            self.field_feature_obstacles=torch.empty((0,3),device=self.device)
        self.generator=torch.Generator(device=self.device).manual_seed(int(seed or 0)); self.steps=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.match_elapsed=torch.zeros(self.n,device=self.device)
        self.match_remaining=torch.full((self.n,),160.,device=self.device)
        self.hub_active=torch.ones((self.n,2),device=self.device,dtype=torch.bool)
        self.hub_centers=self._game_hub_centers()
        self.hub_inactive_first=torch.randint(2,(self.n,),device=self.device,generator=self.generator)
        self.auto_fuel_scores=torch.zeros((self.n,2),device=self.device,dtype=torch.long)
        self.piece_pos=torch.zeros((self.n,600,2),device=self.device)
        self.piece_vel=torch.zeros_like(self.piece_pos)
        self.piece_active=torch.zeros((self.n,600),device=self.device,dtype=torch.bool)
        self.piece_owner=torch.full((self.n,600),-1,device=self.device,dtype=torch.long)
        self.piece_zone=torch.full((self.n,600),-1,device=self.device,dtype=torch.long)
        self.piece_type=torch.full((self.n,600),-1,device=self.device,dtype=torch.long)
        self.preloads_per_robot=torch.zeros((self.n,),device=self.device,dtype=torch.long)
        self.fuel_acquisition_count=torch.zeros((self.n,2),device=self.device,dtype=torch.long)
        self.fuel_score_count=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_denied_count=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_abandoned_count=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_acquired_event=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_scored_event=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_denied_event=torch.zeros_like(self.fuel_acquisition_count)
        self.fuel_abandoned_event=torch.zeros_like(self.fuel_acquisition_count)
        self._last_hub_zone=torch.zeros((self.n,2),device=self.device,dtype=torch.bool)
        self._last_strategic_action=torch.zeros((self.n,),device=self.device,dtype=torch.long)
        self.static_opponent_mask=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.goal=torch.zeros((self.n,2),device=self.device); self.goal_radius=torch.full((self.n,),self.base_goal_radius,device=self.device); self.previous=torch.zeros(self.n,device=self.device)
        self.last_contact=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.control_delay=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.observation_delay=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.action_history=torch.zeros((self.n,self.max_control_latency_steps+1,3),device=self.device)
        self.observation_history=torch.zeros((self.n,self.max_observation_latency_steps+1,self.obs_dim),device=self.device)
        self._last_observation=torch.zeros((self.n,self.obs_dim),device=self.device)
        self.initial_distance=torch.zeros(self.n,device=self.device)
        self.path_length=torch.zeros(self.n,device=self.device)
        self.contact_steps=torch.zeros(self.n,device=self.device)
        self.contact_count=torch.zeros(self.n,device=self.device)
        self.last_static_contact=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.blocked_steps=torch.zeros(self.n,device=self.device)
        self.out_of_bounds_steps=torch.zeros(self.n,device=self.device)
        self.last_action=torch.zeros((self.n,3),device=self.device)

    def _game_hub_centers(self):
        """Official REBUILT HUB centers, sourced from field.py geometry anchors."""
        by_name={getattr(box,"name",""):box for box in self.field_boxes}
        if "red_hub" in by_name and "blue_hub" in by_name:
            points=[(by_name[name].x,by_name[name].y) for name in ("red_hub","blue_hub")]
        else:
            x=4.03+1.19/2
            points=[(x,self.sim.field_width/2),(self.sim.field_length-x,self.sim.field_width/2)]
        return torch.tensor(points,device=self.device,dtype=self.sim.pose.dtype)

    def _attacker_objective(self):
        if self.action_mode=="strategic":
            side=0 if self.task=="counter_defense" else 1
            hub=self.hub_centers[side].expand(self.n,-1)
            if self.task=="defense" and self.piece_active.shape[1]:
                # Scripted attacker baseline: acquire the nearest available
                # FUEL, then route to its active HUB. This chooses an objective
                # from current game state and never consumes a hidden goal.
                carrying=(self.piece_active&(self.piece_owner==1)).any(-1)
                free=self.piece_active&(self.piece_owner<0)
                distance=(self.piece_pos-self.sim.pose[:,1,None,:2]).norm(dim=-1)
                distance=distance.masked_fill(~free,float("inf"))
                nearest,index=distance.min(-1)
                piece=self.piece_pos[torch.arange(self.n,device=self.device),index]
                direction=self.sim.pose[:,1,:2]-hub
                direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
                radius=.595+.5*torch.sqrt(self.sim.length[:,1].square()+self.sim.width[:,1].square())+.02
                approach=hub+direction*radius[:,None]
                score_target=carrying&self.hub_active[:,side]
                target=torch.where(score_target[:,None],approach,
                    torch.where(torch.isfinite(nearest)[:,None],piece,approach))
                return target
            return hub
        return self.goal

    def release_outpost_fuel(self, alliance, position, count=1, mask=None):
        """Model a human-player fuel release at a caller-supplied chute opening.

        ``position`` is the measured field-relative release point; the simulator
        deliberately does not guess the in-field chute opening coordinate.
        Returns the number released per environment.
        """
        side=str(alliance).lower()
        if side not in ("red","blue"):
            raise ValueError("alliance must be red or blue")
        if count<0: raise ValueError("count must be nonnegative")
        selected=(torch.ones((self.n,),device=self.device,dtype=torch.bool) if mask is None
                  else torch.as_tensor(mask,device=self.device,dtype=torch.bool).reshape(self.n))
        point=torch.as_tensor(position,device=self.device,dtype=self.sim.pose.dtype)
        if point.shape==(2,):point=point.expand(self.n,2)
        if tuple(point.shape)!=(self.n,2):raise ValueError(f"position must be {(self.n,2)}")
        zone=3 if side=="red" else 4
        released=torch.zeros((self.n,),device=self.device,dtype=torch.long)
        rows=torch.arange(self.n,device=self.device)
        for _ in range(min(int(count),24)):
            stock=(self.piece_zone==zone)&~self.piece_active
            available=stock.any(-1)&selected
            index=stock.to(torch.int32).argmax(-1)
            self.piece_pos[rows[available],index[available]]=point[available]
            self.piece_vel[rows[available],index[available]]=0.
            self.piece_zone[rows[available],index[available]]=0
            self.piece_owner[rows[available],index[available]]=-1
            self.piece_active[rows[available],index[available]]=True
            released+=available.long()
        return released

    def _initialize_gamepieces(self, mask=None):
        """Reset 2026 FUEL staging with a bounded 600-slot tensor catalog.

        Default count follows the official 504-piece match staging. The 2D
        model places 24 at each DEPOT and the neutral count (360..408) in the
        official neutral-pile footprint. The 24 per OUTPOST remain inactive
        off-field stock behind the chute. Ball-ball and vertical dynamics are
        intentionally omitted.
        """
        if mask is None:
            mask=torch.ones((self.n,),device=self.device,dtype=torch.bool)
        else:
            mask=torch.as_tensor(mask,device=self.device,dtype=torch.bool).reshape(self.n)
        positions=torch.zeros_like(self.piece_pos)
        active=torch.zeros_like(self.piece_active)
        owner=torch.full_like(self.piece_owner,-1)
        zone=torch.full_like(self.piece_zone,-1)
        kind=torch.full_like(self.piece_type,-1)
        count=self.fuel_count
        # The manual's 360..408 neutral count results from 0..48 robot preloads.
        # Default to a randomized preload count for the six-robot alliance; a
        # fixed per-robot count can be supplied for controlled experiments.
        max_preload=min(8,max(0,(count-96)//6))
        per_robot=(torch.randint(max_preload+1,(self.n,),device=self.device,generator=self.generator)
                   if self.preloaded_per_robot is None else
                   torch.full((self.n,),self.preloaded_per_robot,device=self.device,dtype=torch.long))
        neutral_count=count-96-6*per_robot
        half_ball=.075
        def sample_rect(num,x0,x1,y0,y1):
            rand=torch.rand((self.n,num,2),device=self.device,generator=self.generator)
            return torch.stack((x0+rand[...,0]*(x1-x0),y0+rand[...,1]*(y1-y0)),-1)
        depots={getattr(box,"name",""):box for box in self.field_boxes}
        red_depot=depots.get("red_depot")
        blue_depot=depots.get("blue_depot")
        if red_depot is None:
            red_depot=(.35,self.sim.field_width/2-2.39,.69,.53)
            blue_depot=(self.sim.field_length-.35,self.sim.field_width/2+2.39,.69,.53)
        else:
            red_depot=(red_depot.x,red_depot.y,red_depot.length,red_depot.width)
            blue_depot=(blue_depot.x,blue_depot.y,blue_depot.length,blue_depot.width)
        for first,rect,front in ((0,red_depot,1),(24,blue_depot,-1)):
            x,y,length,width=rect
            # Stage within/along the low DEPOT footprint; the planar solver
            # treats these FUEL pieces as non-colliding objects.
            dep_pos=sample_rect(24,max(half_ball,x-length/2),min(self.sim.field_length-half_ball,x+length/2),
                                max(half_ball,y-width/2),min(self.sim.field_width-half_ball,y+width/2))
            positions[:,first:first+24]=dep_pos
            zone[:,first:first+24]=1 if front==1 else 2
        center_y=self.sim.field_width/2
        # Outpost stock remains off-field behind the chute/door; no guessed
        # in-field spawn is used. It is available to future explicit release events.
        zone[:,48:72]=3
        zone[:,72:96]=4
        neutral_start=96
        neutral_max=count-96
        # Official neutral pile: 206 x 72 in (~5.23 x 1.83 m), roughly split
        # across the center line; random scatter avoids a perfect grid.
        neutral=sample_rect(neutral_max,
            max(half_ball,self.sim.field_length/2-.915),
            min(self.sim.field_length-half_ball,self.sim.field_length/2+.915),
            max(half_ball,center_y-2.615),min(self.sim.field_width-half_ball,center_y+2.615))
        positions[:,neutral_start:neutral_start+neutral_max]=neutral
        neutral_slots=torch.arange(neutral_max,device=self.device)[None,:]<neutral_count[:,None]
        zone[:,neutral_start:neutral_start+neutral_max]=torch.where(neutral_slots,0,-1)
        rows=torch.arange(self.n,device=self.device)[:,None]
        within=torch.arange(8,device=self.device)[None,:]
        for robot in range(6):
            indices=neutral_start+neutral_count[:,None]+robot*per_robot[:,None]+within
            valid=within<per_robot[:,None]
            if robot<2:
                preload_position=self.sim.pose[:,robot,None,:2].expand(-1,8,-1)
            else:
                side=0 if robot in (2,3) else 1
                x=(1.2 if side==0 else self.sim.field_length-1.2)
                preload_position=torch.zeros((self.n,8,2),device=self.device)
                preload_position[:,:,0]=x
                preload_position[:,:,1]=self.sim.field_width/2+(robot-3.5)*.55
            rr=rows.expand(-1,8)[valid]
            cc=indices[valid]
            positions[rr,cc]=preload_position[valid]
            active[rr,cc]=True
            zone[rr,cc]=5
            owner[rr,cc]=robot
        active[:,:48]=True
        active[:,neutral_start:neutral_start+neutral_max]|=neutral_slots
        kind[:,:count]=0
        self.piece_pos=torch.where(mask[:,None,None],positions,self.piece_pos)
        self.piece_vel=torch.where(mask[:,None,None],torch.zeros_like(positions),self.piece_vel)
        self.piece_active=torch.where(mask[:,None],active,self.piece_active)
        self.piece_owner=torch.where(mask[:,None],owner,self.piece_owner)
        self.piece_zone=torch.where(mask[:,None],zone,self.piece_zone)
        self.piece_type=torch.where(mask[:,None],kind,self.piece_type)
        self.preloads_per_robot=torch.where(mask,per_robot,self.preloads_per_robot)

    def _update_match_clock(self):
        self.match_elapsed=(self.match_elapsed+self.match_clock_step).clamp(max=160.)
        self.match_remaining=(160.-self.match_elapsed).clamp_min(0.)
        red_auto=self.auto_fuel_scores[:,0]
        blue_auto=self.auto_fuel_scores[:,1]
        first_inactive=torch.where(red_auto>blue_auto,torch.zeros_like(self.hub_inactive_first),
            torch.where(blue_auto>red_auto,torch.ones_like(self.hub_inactive_first),self.hub_inactive_first))
        elapsed=self.match_elapsed
        in_shifts=(elapsed>=30.)&(elapsed<130.)
        shift=((elapsed-30.).clamp_min(0.)/25.).floor().long().clamp(0,3)
        inactive=torch.where((shift%2)==0,first_inactive,1-first_inactive)
        self.hub_active=torch.ones((self.n,2),device=self.device,dtype=torch.bool)
        self.hub_active.scatter_(1,inactive[:,None],(~in_shifts)[:,None])

    def _update_gamepieces(self):
        """Vectorized pickup/score state; HUB score is a 2D range surrogate.

        REBUILT scoring requires FUEL through a 1.06 m top opening 1.83 m above
        carpet. This planar simulator approximates that event at the HUB
        perimeter and applies official active/inactive timing; it does not model
        launch trajectories or sensor-array passage.
        """
        self._update_match_clock()
        self.fuel_acquired_event.zero_(); self.fuel_scored_event.zero_()
        self.fuel_denied_event.zero_(); self.fuel_abandoned_event.zero_()
        newly_scored=torch.zeros_like(self.piece_active)
        rows=torch.arange(self.n,device=self.device)
        free=self.piece_active&(self.piece_owner<0)
        # Intake is on the robot's local +X/front side. Its capture band starts
        # just inside the front bumper and extends 0.35 m ahead, across the
        # bumper width plus a small game-piece margin. Rear/side contacts do
        # not acquire pieces. One pickup per robot per step; carrying is uncapped.
        for robot in (0,1):
            pose=self.sim.pose[:,robot]
            position=pose[:,:2]
            delta=self.piece_pos-position[:,None,:]
            heading=pose[:,2]
            forward=torch.stack((torch.cos(heading),torch.sin(heading)),-1)
            left=torch.stack((-torch.sin(heading),torch.cos(heading)),-1)
            longitudinal=(delta*forward[:,None,:]).sum(-1)
            lateral=(delta*left[:,None,:]).sum(-1).abs()
            front_edge=self.sim.length[:,robot,None]*.5
            intake_reach=.35
            intake_half_width=self.sim.width[:,robot,None]*.5+.075
            intake=(longitudinal>=front_edge-.075)&(longitudinal<=front_edge+intake_reach)&(lateral<=intake_half_width)
            distance=delta.norm(dim=-1).masked_fill(~(free&intake),float("inf"))
            nearest,index=distance.min(-1)
            picked=torch.isfinite(nearest)
            self.piece_owner[rows[picked],index[picked]]=robot
            self.fuel_acquired_event[:,robot]=picked.long()
            free[rows[picked],index[picked]]=False
        for robot in (0,1):
            held=self.piece_active&(self.piece_owner==robot)
            self.piece_pos=torch.where(held[...,None],self.sim.pose[:,robot,None,:2],self.piece_pos)
            self.piece_vel=torch.where(held[...,None],self.sim.velocity[:,robot,None,:2],self.piece_vel)
            center=self.hub_centers[robot]
            robot_radius=.5*torch.sqrt(self.sim.length[:,robot].square()+self.sim.width[:,robot].square())
            near_hub=(self.sim.pose[:,robot,:2]-center).norm(dim=-1)<=(.595+robot_radius+.075)
            has_fuel=held.any(-1)
            score_intent=((self._last_strategic_action==1) if self.action_mode=="strategic"
                          else torch.ones((self.n,),device=self.device,dtype=torch.bool))
            entering=near_hub&score_intent&~self._last_hub_zone[:,robot]
            denied=entering&has_fuel&~self.hub_active[:,robot]
            self.fuel_denied_event[:,robot]=denied.long()
            self.fuel_denied_count[:,robot]+=denied.long()
            scored=held&near_hub[:,None]&self.hub_active[:,robot,None]&score_intent[:,None]
            newly_scored|=scored
            score_count=scored.sum(-1)
            self.fuel_scored_event[:,robot]=score_count
            self.fuel_score_count[:,robot]+=score_count
            in_auto=self.match_elapsed<=20.+self.dt
            self.auto_fuel_scores[:,robot]+=score_count*in_auto.long()
            self.piece_owner=torch.where(scored,-2,self.piece_owner)
            self.piece_zone=torch.where(scored,torch.full_like(self.piece_zone,6),self.piece_zone)
            self.piece_vel=torch.where(scored[...,None],torch.zeros_like(self.piece_vel),self.piece_vel)
            self._last_hub_zone[:,robot]=near_hub&score_intent
        self.piece_active=torch.where(newly_scored,torch.zeros_like(self.piece_active),self.piece_active)
        self.fuel_acquisition_count+=self.fuel_acquired_event
        self.fuel_abandoned_count+=self.fuel_abandoned_event
        return self.fuel_acquired_event,self.fuel_scored_event,self.fuel_denied_event

    def _role_swapped_observation(self):
        """Build the legacy 35-feature view for robot 1 as policy focal robot."""
        p,v=self.sim.pose,self.sim.velocity
        objective=self._attacker_objective()
        relative_goal=(torch.zeros((self.n,2),device=self.device) if self.action_mode=="strategic"
                       else objective-p[:,1,:2])
        radius=(torch.full_like(self.goal_radius,.595) if self.action_mode=="strategic" else self.goal_radius)
        raw=torch.cat((p[:,1],v[:,1],p[:,0],v[:,0],relative_goal,radius[:,None],
            torch.full((self.n,1),self.sim.field_length,device=self.device),
            torch.full((self.n,1),self.sim.field_width,device=self.device),
            self.sim.length[:,[1,0]],self.sim.width[:,[1,0]],self.sim.accel[:,[1,0]]/10.,
            self._obstacle_features(robot_index=1)),-1)
        if self.normalize_observations:
            normalize_tensor_observation_batch_in_place(raw,self.sim.field_length,self.sim.field_width,
                self.sim.speed[:,[1,0]],self.sim.omega_limit[:,[1,0]])
        return raw

    def _raw_obs(self):
        p,v=self.sim.pose,self.sim.velocity
        objective=(self.hub_centers[0].expand(self.n,-1) if self.action_mode=="strategic" else self.goal)
        rel=(torch.zeros_like(self.goal) if self.action_mode=="strategic"
             else objective-p[:,0,:2])
        # Tactical defense needs the defended region to choose an interception
        # point. Both tasks expose the goal and radius at the shared indices.
        goal_radius=(torch.full((self.n,1),.595,device=self.device) if self.action_mode=="strategic"
                     else self.goal_radius[:,None])
        raw=torch.cat((p[:,0],v[:,0],p[:,1],v[:,1],rel,goal_radius,
                       torch.full((self.n,1),self.sim.field_length,device=self.device),torch.full((self.n,1),self.sim.field_width,device=self.device),
                       self.sim.length,self.sim.width,self.sim.accel/10.,self._obstacle_features()),-1)
        if self.action_mode in ("tactical","strategic"):
            # Cached AD* route to the previous tactical waypoint. The final
            # three features are next-route-point delta (field-normalized) and
            # remaining path length (field-diagonal-normalized); zeros indicate
            # that a route is not available yet.
            route_features=torch.zeros((self.n,3),device=self.device)
            if self._adstar_tactical_planner is not None:
                planner=self._adstar_tactical_planner
                has_route=planner.last_lengths>1
                first=planner.last_path[:,1,:]-p[:,0,:2]
                segment=(planner.last_path[:,1:,:]-planner.last_path[:,:-1,:]).norm(dim=-1)
                valid=(torch.arange(segment.shape[1],device=self.device)[None,:]
                       <(planner.last_lengths-1).clamp_min(0)[:,None])
                path_cost=(segment*valid).sum(-1)
                route_features=torch.stack((first[:,0]/self.sim.field_length,
                    first[:,1]/self.sim.field_width,
                    path_cost/math.hypot(self.sim.field_length,self.sim.field_width)),dim=-1)
                route_features=torch.where(has_route[:,None],route_features,torch.zeros_like(route_features))
            raw=torch.cat((raw,route_features),dim=-1)
        if self.action_mode=="strategic":
            free=self.piece_active&(self.piece_owner<0)
            local_features=[]
            for robot in (0,1):
                distance=(self.piece_pos-self.sim.pose[:,robot,None,:2]).norm(dim=-1)
                distance=distance.masked_fill(~free,float("inf"))
                nearest,indices=distance.topk(4,dim=-1,largest=False)
                rows=torch.arange(self.n,device=self.device)[:,None]
                points=self.piece_pos[rows,indices]
                velocities=self.piece_vel[rows,indices]
                valid=torch.isfinite(nearest)
                relative=points-self.sim.pose[:,robot,None,:2]
                normalized_velocity=velocities/self.sim.speed[:,robot,None,None].clamp_min(.1)
                slot=torch.cat((relative[...,0:1]/self.sim.field_length,
                    relative[...,1:2]/self.sim.field_width,normalized_velocity,
                    valid[...,None].to(raw.dtype)),dim=-1)
                local_features.append(torch.where(valid[...,None],slot,torch.zeros_like(slot)).reshape(self.n,-1))
            possession=torch.stack((
                (self.piece_active&(self.piece_owner==0)).sum(-1).to(raw.dtype),
                (self.piece_active&(self.piece_owner==1)).sum(-1).to(raw.dtype)),dim=-1)
            possession=possession/max(self.fuel_count,1)
            match_features=torch.cat((self.match_elapsed[:,None]/160.,
                self.hub_active.to(raw.dtype),
                self.fuel_score_count.to(raw.dtype)/max(self.fuel_count,1),
                (self.hub_centers[None,:,:]-p[:,0,None,:2]).reshape(self.n,4)/
                    torch.tensor((self.sim.field_length,self.sim.field_width,
                                  self.sim.field_length,self.sim.field_width),device=self.device)),dim=-1)
            raw=torch.cat((raw,*local_features,possession,match_features),dim=-1)
        # Feature scales are fixed and shared with external policy adapters;
        # no rollout-fitted running statistics can leak into evaluation.
        if self.normalize_observations:
            normalize_tensor_observation_batch_in_place(
                raw,self.sim.field_length,self.sim.field_width,self.sim.speed,self.sim.omega_limit)
        return raw

    def _obs(self, initialize_mask=None):
        raw=self._raw_obs()
        if initialize_mask is None:
            self.observation_history=torch.cat((raw[:,None,:],self.observation_history[:,:-1,:]),dim=1)
        else:
            fresh=raw[:,None,:].expand(-1,self.observation_history.shape[1],-1)
            self.observation_history=torch.where(initialize_mask[:,None,None],fresh,self.observation_history)
        obs=self.observation_history.gather(1,self.observation_delay[:,None,None].expand(-1,1,self.obs_dim)).squeeze(1)
        if self.randomize and self.observation_noise>0:
            state=obs[:,:18]+torch.randn((self.n,18),device=self.device,generator=self.generator)*self.observation_noise
            obs=torch.cat((state,obs[:,18:]),-1)
        if self.randomize and self.observation_dropout>0:
            keep=torch.rand((self.n,18),device=self.device,generator=self.generator)>=self.observation_dropout
            obs=torch.cat((obs[:,:18]*keep,obs[:,18:]),-1)
        return obs

    def _obstacle_features(self,robot_index=0):
        out=torch.zeros((self.n,12),device=self.device)
        obstacles=(torch.cat((self.sim.obstacles,self.field_feature_obstacles),0)
                   if self.field_feature_obstacles.shape[0] else self.sim.obstacles)
        if obstacles.shape[0]:
            dist=(obstacles[None,:,:2]-self.sim.pose[:,robot_index,None,:2]).square().sum(-1)
            chosen=obstacles[dist.topk(min(4,obstacles.shape[0]),dim=-1,largest=False).indices]
            out[:,:chosen.shape[1]*3]=chosen.reshape(self.n,-1)
        return out

    def _sample_episode_endpoints(self, goal_override=None, start_zone=None, goal_zone=None):
        """Sample endpoints in requested zones, or random distinct field zones."""
        length, width = self.sim.field_length, self.sim.field_width
        margin = .85
        depth = ALLIANCE_ZONE_DEPTH if self.field_boxes else length / 3
        depth = min(depth, (length - 2 * margin) / 3)
        edges = torch.tensor((0., depth, length - depth, length), device=self.device)
        zone_ids={"red":0,"center":1,"middle":1,"blue":2}
        def zone_tensor(value):
            if value is None or (isinstance(value,str) and value.lower()=="random"):
                return None
            if isinstance(value,str):
                if value.lower() not in zone_ids:
                    raise ValueError("zone must be red, center, blue, or random")
                return torch.full((self.n,),zone_ids[value.lower()],device=self.device,dtype=torch.long)
            zones=torch.as_tensor(value,device=self.device,dtype=torch.long).reshape(-1)
            if zones.numel()==1: zones=zones.expand(self.n)
            if zones.numel()!=self.n or bool(((zones<0)|(zones>2)).any().item()):
                raise ValueError("zone ids must contain one value per environment in 0..2")
            return zones
        requested_start=zone_tensor(start_zone)
        requested_goal=zone_tensor(goal_zone)
        if isinstance(start_zone,str) and isinstance(goal_zone,str) and start_zone in zone_ids and start_zone==goal_zone:
            raise ValueError("start and goal zones must be different")
        random_start=torch.randint(3,(self.n,),device=self.device,generator=self.generator)
        start_zone=(requested_start if requested_start is not None else random_start)
        if goal_override is None:
            if requested_goal is not None:
                goal_zone=requested_goal
                same=start_zone==goal_zone
                if bool(same.any().item()):
                    raise ValueError("start and goal zones must be different")
            elif requested_start is not None:
                goal_zone=(start_zone+torch.randint(1,3,(self.n,),device=self.device,generator=self.generator))%3
            else:
                goal_zone = (start_zone + torch.randint(1, 3, (self.n,), device=self.device,
                                                          generator=self.generator)) % 3
        else:
            if requested_goal is not None:
                raise ValueError("goal_zone cannot be combined with an explicit goal coordinate")
            goal = torch.as_tensor(goal_override, device=self.device, dtype=self.sim.pose.dtype).expand(self.n, 2)
            goal_zone = torch.where(goal[:, 0] < depth, 0,
                                    torch.where(goal[:, 0] >= length - depth, 2, 1)).long()
            if requested_start is None:
                choices = torch.randint(2, (self.n,), device=self.device, generator=self.generator)
                start_zone = torch.where(goal_zone == 0, choices + 1,
                             torch.where(goal_zone == 1, choices * 2, choices))
            elif bool((start_zone==goal_zone).any().item()):
                raise ValueError("start and goal zones must be different")

        def sample_points(zones, avoid=None):
            low, high = edges[zones] + margin, edges[zones + 1] - margin
            points = torch.zeros((self.n, 2), device=self.device, dtype=self.sim.pose.dtype)
            unresolved = torch.ones((self.n,), device=self.device, dtype=torch.bool)
            for _ in range(12):
                candidate = torch.stack((low + torch.rand((self.n,), device=self.device,
                    generator=self.generator) * (high - low),
                    margin + torch.rand((self.n,), device=self.device,
                    generator=self.generator) * (width - 2 * margin)), -1)
                valid = torch.ones((self.n,), device=self.device, dtype=torch.bool)
                for box in self.sim.field_colliders:
                    valid &= (((candidate[:, 0] - box[0]).abs() > box[2] + .8) |
                              ((candidate[:, 1] - box[1]).abs() > box[3] + .8))
                for obstacle in self.sim.obstacles:
                    valid &= ((candidate - obstacle[:2]).square().sum(-1) > (obstacle[2] + .8) ** 2)
                if avoid is not None:
                    valid &= (candidate - avoid).norm(dim=-1) >= 3.
                accept = unresolved & valid
                points = torch.where(accept[:, None], candidate, points)
                unresolved &= ~accept
            # Rejection sampling can miss in dense layouts. Fall back to a
            # random choice from a legal 0.2 m lattice; never return the last
            # unchecked sample. If the 3 m separation preference leaves no
            # legal point, relax only that preference, not collider clearance.
            if bool(unresolved.any().item()):
                xs=torch.arange(margin,self.sim.field_length-margin+.001,.2,
                                device=self.device,dtype=self.sim.pose.dtype)
                ys=torch.arange(margin,width-margin+.001,.2,
                                device=self.device,dtype=self.sim.pose.dtype)
                gx,gy=torch.meshgrid(xs,ys,indexing="ij")
                lattice=torch.stack((gx.flatten(),gy.flatten()),-1)
                zone_of=torch.where(lattice[:,0]<depth,0,
                         torch.where(lattice[:,0]>=length-depth,2,1))
                clear=torch.ones((lattice.shape[0],),device=self.device,dtype=torch.bool)
                for box in self.sim.field_colliders:
                    clear &= ((lattice[:,0]-box[0]).abs()>box[2]+.8)|((lattice[:,1]-box[1]).abs()>box[3]+.8)
                for obstacle in self.sim.obstacles:
                    clear &= ((lattice-obstacle[:2]).square().sum(-1)>(obstacle[2]+.8)**2)
                allowed=clear[None,:]&(zone_of[None,:]==zones[:,None])
                if avoid is not None:
                    separated=allowed&((lattice[None,:,:]-avoid[:,None,:]).square().sum(-1)>=9.)
                    allowed=torch.where(separated.any(-1,keepdim=True),separated,allowed)
                has_legal_point=allowed.any(-1)
                if not bool(has_legal_point[unresolved].all().item()):
                    raise RuntimeError("no legal endpoint exists in the selected field zone")
                random_rank=torch.rand((self.n,lattice.shape[0]),device=self.device,generator=self.generator)
                fallback_index=random_rank.masked_fill(~allowed,-1.).argmax(-1)
                fallback=lattice[fallback_index]
                points=torch.where(unresolved[:,None],fallback,points)
            return points

        start = sample_points(start_zone)
        goal = (torch.as_tensor(goal_override, device=self.device, dtype=self.sim.pose.dtype)
                .expand(self.n, 2).clone() if goal_override is not None
                else sample_points(goal_zone, avoid=start))
        return start, goal

    def _place_defenders_near_clear_adstar_paths(self, mask=None):
        """Place each reset defender near a randomized point on the clear AD* route."""
        if self._adstar_planners is None or not self.adstar_spawn_hint:
            return
        starts=self.sim.pose[:,1,:2]
        selected=(torch.ones(self.n,device=self.device,dtype=torch.bool) if mask is None
                  else torch.as_tensor(mask,device=self.device,dtype=torch.bool))
        planner=self._adstar_planners
        planner.plan(starts,self._attacker_objective(),self.sim.pose[:,1,2],self.sim.length[:,1],
                     self.sim.width[:,1])
        candidate=planner.defender_spawn(starts,selected,self.generator)
        self.sim.pose[:,0,:2]=torch.where(selected[:,None],candidate,self.sim.pose[:,0,:2])

    def _move_adstar_defenders_to_clear_start(self, mask=None):
        """Keep an AD* guard from spawning inside its inflated obstacle map."""
        bumps=self.sim.bump_regions
        if self._adstar_defender_planners is None or not bumps.numel():
            return
        selected=(torch.ones(self.n,device=self.device,dtype=torch.bool) if mask is None
                  else torch.as_tensor(mask,device=self.device,dtype=torch.bool))
        position=self.sim.pose[:,1,:2]
        radius=.5*torch.sqrt(self.sim.length[:,1].square()+self.sim.width[:,1].square())+.15
        dx=(position[:,None,0]-bumps[None,:,0]).abs()
        dy=(position[:,None,1]-bumps[None,:,1]).abs()
        inside=((dx<=bumps[None,:,2]+radius[:,None]) &
                (dy<=bumps[None,:,3]+radius[:,None])).any(-1)
        boxes=self.sim.field_colliders
        if boxes.numel():
            box_dx=(position[:,None,0]-boxes[None,:,0]).abs()
            box_dy=(position[:,None,1]-boxes[None,:,1]).abs()
            inside|=((box_dx<=boxes[None,:,2]+radius[:,None]) &
                     (box_dy<=boxes[None,:,3]+radius[:,None])).any(-1)
        obstacles=self.sim.obstacles
        if obstacles.numel():
            obstacle_distance=(position[:,None,:]-obstacles[None,:,:2]).norm(dim=-1)
            inside|=(obstacle_distance<=obstacles[None,:,2]+radius[:,None]).any(-1)
        attacker=self.sim.pose[:,0,:2]
        separation=(position-attacker).norm(dim=-1)
        needs_move=selected&(inside|(separation<1.4)|(separation>3.5))
        margin=radius.clamp_min(.9)
        lane=self._attacker_objective()-attacker
        lane_length=lane.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        preferred=attacker+lane/lane_length*torch.minimum(lane_length*.55,
            torch.full_like(lane_length,2.5))
        for _ in range(32):
            angle=torch.rand((self.n,),device=self.device,generator=self.generator)*(2*math.pi)
            spawn_radius=.25+torch.rand((self.n,),device=self.device,generator=self.generator)*1.35
            candidate=torch.stack((
                preferred[:,0]+angle.cos()*spawn_radius,
                preferred[:,1]+angle.sin()*spawn_radius),dim=-1)
            cdx=(candidate[:,None,0]-bumps[None,:,0]).abs()
            cdy=(candidate[:,None,1]-bumps[None,:,1]).abs()
            clear=~((cdx<=bumps[None,:,2]+radius[:,None]) &
                    (cdy<=bumps[None,:,3]+radius[:,None])).any(-1)
            if boxes.numel():
                box_dx=(candidate[:,None,0]-boxes[None,:,0]).abs()
                box_dy=(candidate[:,None,1]-boxes[None,:,1]).abs()
                hits=((box_dx<=boxes[None,:,2]+radius[:,None]) &
                      (box_dy<=boxes[None,:,3]+radius[:,None])).any(-1)
                clear&=~hits
            obstacles=self.sim.obstacles
            if obstacles.numel():
                obstacle_distance=(candidate[:,None,:]-obstacles[None,:,:2]).norm(dim=-1)
                clear&=~(obstacle_distance<=obstacles[None,:,2]+radius[:,None]).any(-1)
            attacker_clearance=.5*torch.sqrt(self.sim.length[:,0].square()+self.sim.width[:,0].square())+radius+.25
            candidate_separation=(candidate-attacker).norm(dim=-1)
            clear&=(candidate_separation>=attacker_clearance)
            clear&=(candidate[:,0]>=margin)&(candidate[:,0]<=self.sim.field_length-margin)
            clear&=(candidate[:,1]>=margin)&(candidate[:,1]<=self.sim.field_width-margin)
            place=needs_move&clear
            position.copy_(torch.where(place[:,None],candidate,position))
            needs_move&=~place
            if not bool(needs_move.any().item()):
                break

    def reset(self,seed=None,options=None):
        if seed is not None:self.generator.manual_seed(int(seed))
        self.sim.reset(seed); self.steps.zero_(); self.last_contact.zero_()
        self._sample_static_opponents(torch.ones(self.n,device=self.device,dtype=torch.bool))
        self.last_static_contact.zero_()
        if self._adstar_planners is not None:
            self._adstar_contact_latched.zero_()
        pidx=0 if self.task=="counter_defense" else 1
        start, goal = self._sample_episode_endpoints(
            options.get("goal") if options and "goal" in options else None,
            options.get("start_zone") if options else None,
            options.get("goal_zone") if options else None)
        self.goal.copy_(goal)
        self.sim.pose[:, pidx, :2] = start
        if self.randomize:
            self.sim.velocity[:,:,:2]=(torch.rand((self.n,2,2),device=self.device,generator=self.generator)*2-1)*.25*self.sim.speed[:,:,None]
            self.sim.velocity[:,:,2]=(torch.rand((self.n,2),device=self.device,generator=self.generator)*2-1)*.2*self.sim.omega_limit
            self.goal_radius=.3+torch.rand((self.n,),device=self.device,generator=self.generator)*.5
        else:
            self.goal_radius.fill_(self.base_goal_radius)
        self.sim.velocity[:,1]=torch.where(self.static_opponent_mask[:,None],
            torch.zeros_like(self.sim.velocity[:,1]),self.sim.velocity[:,1])
        if options and "goal_radius" in options:
            self.goal_radius.copy_(torch.as_tensor(options["goal_radius"],device=self.device,dtype=self.goal.dtype).expand_as(self.goal_radius))
        self._move_adstar_defenders_to_clear_start()
        self._place_defenders_near_clear_adstar_paths()
        self._initialize_gamepieces()
        self.match_elapsed.zero_(); self.match_remaining.fill_(160.)
        self.hub_active.fill_(True)
        self.hub_inactive_first=torch.randint(2,(self.n,),device=self.device,generator=self.generator)
        self.auto_fuel_scores.zero_()
        for value in (self.fuel_acquisition_count,self.fuel_score_count,self.fuel_denied_count,
                      self.fuel_abandoned_count,self.fuel_acquired_event,self.fuel_scored_event,
                      self.fuel_denied_event,self.fuel_abandoned_event):
            value.zero_()
        self._last_hub_zone.zero_(); self._last_strategic_action.zero_()
        if (self._adstar_planners is not None or self._adstar_defender_planners is not None
                or self._adstar_tactical_planner is not None):
            self._planner_tick=self.adstar_replan_interval-1
        self.action_history.zero_()
        self.control_delay.zero_()
        if self.randomize and self.max_control_latency_steps:
            self.control_delay.random_(self.max_control_latency_steps+1,generator=self.generator)
        self.observation_delay.zero_()
        if self.randomize and self.max_observation_latency_steps:
            self.observation_delay.random_(self.max_observation_latency_steps+1,generator=self.generator)
        attacker_index=0 if self.task=="counter_defense" else 1
        self.previous.copy_((self._attacker_objective()-self.sim.pose[:,attacker_index,:2]).norm(dim=-1))
        self.initial_distance.copy_(self.previous)
        self.path_length.zero_(); self.contact_steps.zero_(); self.contact_count.zero_()
        self.blocked_steps.zero_(); self.out_of_bounds_steps.zero_(); self.last_contact.zero_(); self.last_action.zero_()
        self.observation_history.zero_()
        self._last_observation=self._obs(torch.ones((self.n,),device=self.device,dtype=torch.bool))
        return self._last_observation,{"goal":self._attacker_objective(),
            "goal_radius":torch.full_like(self.goal_radius,.595) if self.action_mode=="strategic" else self.goal_radius,
            "field_layout":"2026_rebuilt" if self.field_boxes else "custom",
            "field_colliders":self.field_boxes}

    def reset_done(self,mask):
        """Reset selected environment slots; return the full fresh observation batch."""
        mask=torch.as_tensor(mask,device=self.device,dtype=torch.bool).reshape(self.n)
        self.sim.reset_done(mask)
        self._sample_static_opponents(mask)
        if self._adstar_planners is not None:
            self._adstar_contact_latched &= ~mask
        if self.randomize:
            vx=(torch.rand((self.n,2,2),device=self.device,generator=self.generator)*2-1)*.25*self.sim.speed[:,:,None]
            vw=(torch.rand((self.n,2),device=self.device,generator=self.generator)*2-1)*.2*self.sim.omega_limit
            self.sim.velocity=torch.where(mask[:,None,None],torch.cat((vx,vw[...,None]),-1),self.sim.velocity)
        self.sim.velocity[:,1]=torch.where(self.static_opponent_mask[:,None],
            torch.zeros_like(self.sim.velocity[:,1]),self.sim.velocity[:,1])
        pidx=0 if self.task=="counter_defense" else 1
        start, goal = self._sample_episode_endpoints()
        self.sim.pose[:, pidx, :2] = torch.where(mask[:, None], start, self.sim.pose[:, pidx, :2])
        self.goal=torch.where(mask[:,None],goal,self.goal)
        self._move_adstar_defenders_to_clear_start(mask)
        self._place_defenders_near_clear_adstar_paths(mask)
        self._initialize_gamepieces(mask)
        self.match_elapsed=torch.where(mask,torch.zeros_like(self.match_elapsed),self.match_elapsed)
        self.match_remaining=torch.where(mask,torch.full_like(self.match_remaining,160.),self.match_remaining)
        self.hub_active=torch.where(mask[:,None],torch.ones_like(self.hub_active),self.hub_active)
        tie_choice=torch.randint(2,(self.n,),device=self.device,generator=self.generator)
        self.hub_inactive_first=torch.where(mask,tie_choice,self.hub_inactive_first)
        self.auto_fuel_scores=torch.where(mask[:,None],torch.zeros_like(self.auto_fuel_scores),self.auto_fuel_scores)
        self._last_hub_zone=torch.where(mask[:,None],torch.zeros_like(self._last_hub_zone),self._last_hub_zone)
        self._last_strategic_action=torch.where(mask,torch.zeros_like(self._last_strategic_action),self._last_strategic_action)
        for value in (self.fuel_acquisition_count,self.fuel_score_count,self.fuel_denied_count,
                      self.fuel_abandoned_count,self.fuel_acquired_event,self.fuel_scored_event,
                      self.fuel_denied_event,self.fuel_abandoned_event):
            value.masked_fill_(mask[:,None],0)
        if (self._adstar_planners is not None or self._adstar_defender_planners is not None
                or self._adstar_tactical_planner is not None):
            self._planner_tick=self.adstar_replan_interval-1
        radii=.3+torch.rand((self.n,),device=self.device,generator=self.generator)*.5
        self.goal_radius=torch.where(mask,radii if self.randomize else torch.full_like(radii,self.base_goal_radius),self.goal_radius)
        self.steps=torch.where(mask,torch.zeros_like(self.steps),self.steps)
        self.last_contact=torch.where(mask,torch.zeros_like(self.last_contact),self.last_contact)
        self.last_static_contact=torch.where(mask,torch.zeros_like(self.last_static_contact),self.last_static_contact)
        for values in (self.path_length,self.contact_steps,self.contact_count,self.blocked_steps,self.out_of_bounds_steps):
            values.masked_fill_(mask,0)
        self.last_action=torch.where(mask[:,None],torch.zeros_like(self.last_action),self.last_action)
        self.action_history=torch.where(mask[:,None,None],torch.zeros_like(self.action_history),self.action_history)
        if self.randomize and self.max_control_latency_steps:
            delays=torch.randint(self.max_control_latency_steps+1,(self.n,),device=self.device,generator=self.generator)
            self.control_delay=torch.where(mask,delays,self.control_delay)
        if self.randomize and self.max_observation_latency_steps:
            delays=torch.randint(self.max_observation_latency_steps+1,(self.n,),device=self.device,generator=self.generator)
            self.observation_delay=torch.where(mask,delays,self.observation_delay)
        distance=(self._attacker_objective()-self.sim.pose[:,pidx,:2]).norm(dim=-1)
        self.previous=torch.where(mask,distance,self.previous)
        fresh=self._obs(mask)
        self._last_observation=torch.where(mask[:,None],fresh,self._last_observation)
        return self._last_observation

    def _sample_static_opponents(self,mask):
        if self.static_opponent_fraction<=0.:
            self.static_opponent_mask=torch.where(mask,torch.zeros_like(self.static_opponent_mask),
                self.static_opponent_mask)
            return
        sampled=torch.rand((self.n,),device=self.device,generator=self.generator)<self.static_opponent_fraction
        self.static_opponent_mask=torch.where(mask,sampled,self.static_opponent_mask)

    def _opponent_velocity_command(self,velocity):
        command=torch.where(self.static_opponent_mask[:,None],torch.zeros_like(velocity),velocity)
        self._last_opponent_command=command
        return command

    def step(self,action):
        a=torch.as_tensor(action,device=self.device,dtype=self.sim.pose.dtype)
        if self.action_mode=="strategic":
            if a.ndim==2 and a.shape[-1]==1:a=a.squeeze(-1)
            if a.ndim==0:a=a.expand(self.n)
            if tuple(a.shape)!=(self.n,):raise ValueError(f"strategic action must be {(self.n,)} class indices")
            strategy=a.long().clamp(0,2)
            stored_action=torch.nn.functional.one_hot(strategy,num_classes=3).to(self.sim.pose.dtype)
        else:
            action_dim=2 if self.action_mode=="tactical" else 3
            if a.shape==(action_dim,):a=a.expand(self.n,action_dim)
            if tuple(a.shape)!=(self.n,action_dim):raise ValueError(f"action must be {(self.n,action_dim)}")
            a=torch.nan_to_num(a).clamp(-1,1)
            stored_action=(torch.cat((a,torch.zeros((self.n,1),device=self.device)),dim=-1)
                           if action_dim==2 else a)
        # Keep the existing three-channel latency buffer for old direct-control
        # policies; tactical policies use its first two waypoint channels and
        # strategic policies encode their discrete class as a one-hot triple.
        self.action_history=torch.cat((stored_action[:,None,:],self.action_history[:,:-1,:]),dim=1)
        applied=self.action_history.gather(1,self.control_delay[:,None,None].expand(-1,1,3)).squeeze(1)
        a=applied
        if self.action_mode=="tactical":
            own=self._adstar_tactical_velocity(a[:,:2])
            a=torch.cat((own[:,:2]/self.sim.speed[:,0,None].clamp_min(.1),
                         torch.zeros((self.n,1),device=self.device)),dim=-1)
        elif self.action_mode=="strategic":
            self._last_strategic_action=applied.argmax(-1)
            target=self._strategic_target(self._last_strategic_action)
            own=self._adstar_target_velocity(target)
            a=torch.cat((own[:,:2]/self.sim.speed[:,0,None].clamp_min(.1),
                         torch.zeros((self.n,1),device=self.device)),dim=-1)
        else:
            own=torch.cat((a[:,:2]*self.sim.speed[:,0,None],(a[:,2]*self.sim.omega_limit[:,0])[:,None]),-1)
        p=self.sim.pose; goal=self._attacker_objective()
        old_position=p[:,0,:2].clone()
        old_distance=(goal-p[:,0,:2]).norm(dim=-1) if self.task=="counter_defense" else (goal-p[:,1,:2]).norm(dim=-1)
        # Opponent motion remains batched and device-local. Each scripted
        # behavior supplies a target, then the common speed-limited controller
        # tracks it. ``offense`` aliases pursuit for the defense task.
        kind=self.opponent.get("value","guard") if isinstance(self.opponent,dict) else self.opponent
        if not isinstance(kind,str): kind="guard"
        kind=kind.lower()
        if self.task=="defense" and kind=="guard": kind="offense"
        elif self.task=="counter_defense" and kind=="guard": kind="adstar_defender"
        controlled=p[:,0,:2]
        goal_target=goal
        if self.task=="defense":
            # The second robot is the attacker; its default target is the goal.
            target=goal_target
        else:
            target=controlled
        if kind=="guard":
            target=.5*(controlled+goal_target)
        elif kind in ("intercept","cutoff"):
            route=goal_target-controlled
            unit=route/route.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            target=controlled+self.sim.velocity[:,0,:2]*.45+unit*.65
        elif kind=="velocity_intercept":
            target=self._velocity_intercept_target()
        elif kind=="mirror":
            target=2*goal_target-controlled
        elif kind in ("pursuit","offense"):
            target=goal_target if self.task=="defense" else controlled
        elif kind=="adstar_defender":
            if self.task!="counter_defense" or self._adstar_defender_planners is None:
                raise ValueError("the AD* defender is available for the counter-defense task")
            oppxy=self._opponent_velocity_command(self._adstar_defender_velocity())
            opp=torch.cat((oppxy,torch.zeros((self.n,1),device=self.device)),dim=-1)
            commands=torch.stack((own,opp),dim=1)
            self.sim.step(commands); self.steps+=1
            return self._finish_step(goal,old_position,old_distance,a)
        elif kind=="adstar":
            if self.task!="defense" or self._adstar_planners is None:
                raise ValueError("the AD* attacker is available for the defense task")
            # _swerve accepts field-relative chassis velocity and performs the
            # body-frame conversion internally for module kinematics.
            oppxy=self._opponent_velocity_command(self._adstar_attacker_velocity())
            opp=torch.cat((oppxy,torch.zeros((self.n,1),device=self.device)),dim=-1)
            commands=torch.stack((own,opp),dim=1)
            self.sim.step(commands); self.steps+=1
            return self._finish_step(goal,old_position,old_distance,a)
        elif kind=="random":
            theta=torch.rand((self.n,),device=self.device,generator=self.generator)*(2*math.pi)
            speed=torch.rand((self.n,),device=self.device,generator=self.generator)*self.sim.speed[:,1]
            oppxy=torch.stack((theta.cos()*speed,theta.sin()*speed),-1)
            oppxy=self._opponent_velocity_command(oppxy)
            opp=torch.cat((oppxy,torch.zeros((self.n,1),device=self.device)),dim=-1)
            commands=torch.stack((own,opp),dim=1)
            self.sim.step(commands); self.steps+=1
            return self._finish_step(goal,old_position,old_distance,a)
        elif kind=="learned" and callable(self.learned_opponent_fn):
            learned=torch.as_tensor(self.learned_opponent_fn(self._role_swapped_observation()),
                                     device=self.device,dtype=self.sim.pose.dtype)
            learned=torch.nan_to_num(learned).clamp(-1,1)
            if self.action_mode=="strategic":
                if learned.shape==(3,):learned=learned.expand(self.n,3)
                if learned.shape==(self.n,3):
                    # Historical direct-control policies are retained in the
                    # opponent pool as tactical diversity. Their output only
                    # selects a semantic objective; AD* still owns motion.
                    action_class=learned.argmax(-1)
                elif learned.shape==(self.n,):
                    action_class=learned.long().clamp(0,2)
                else:
                    raise ValueError(f"learned strategic opponent must return {(self.n,3)} scores or {(self.n,)} classes")
                target=self._strategic_opponent_target(action_class)
                oppxy=self._adstar_opponent_velocity(target)
                opp=torch.cat((oppxy,torch.zeros((self.n,1),device=self.device)),dim=-1)
                commands=torch.stack((own,opp),dim=1)
                self.sim.step(commands); self.steps+=1
                return self._finish_step(goal,old_position,old_distance,a)
            if learned.shape==(3,):learned=learned.expand(self.n,3)
            if tuple(learned.shape)!=(self.n,3):
                raise ValueError(f"learned_opponent_fn must return {(self.n,3)} normalized controls")
            opp=torch.stack((learned[:,:2]*self.sim.speed[:,1,None],
                learned[:,2]*self.sim.omega_limit[:,1]),dim=-1)
            oppxy=self._opponent_velocity_command(opp[:,:2])
            opp=torch.cat((oppxy,opp[:,2:]),dim=-1)
            commands=torch.stack((own,opp),dim=1)
            self.sim.step(commands); self.steps+=1
            return self._finish_step(goal,old_position,old_distance,a)
        direction=target-p[:,1,:2]
        oppxy=direction/(direction.norm(dim=-1,keepdim=True).clamp_min(1e-8))*self.sim.speed[:,1,None]
        oppxy=self._opponent_velocity_command(oppxy)
        opp=torch.cat((oppxy,torch.zeros((self.n,1),device=self.device)),dim=-1)
        commands=torch.stack((own,opp),dim=1)
        self.sim.step(commands); self.steps+=1
        return self._finish_step(goal,old_position,old_distance,a)

    def _adstar_tactical_velocity(self, waypoint_action):
        """Decode a normalized field waypoint and route robot 0 to it."""
        target=torch.stack(((waypoint_action[:,0]+1.)*.5*self.sim.field_length,
                            (waypoint_action[:,1]+1.)*.5*self.sim.field_width),dim=-1)
        return self._adstar_target_velocity(target)

    def _adstar_target_velocity(self, waypoint):
        """Route robot 0 to an absolute strategic waypoint with AD*."""
        planner=self._adstar_tactical_planner
        defender=self.sim.pose[:,0]
        margin=.5*torch.maximum(self.sim.length[:,0],self.sim.width[:,0])
        low=margin[:,None]
        high=torch.stack((self.sim.field_length-margin,self.sim.field_width-margin),dim=-1)
        waypoint=torch.maximum(torch.minimum(waypoint,high),low)
        self._planner_tick+=1
        if self._planner_tick%self.adstar_replan_interval==0:
            planner.plan(defender[:,:2],waypoint,defender[:,2],self.sim.length[:,0],
                self.sim.width[:,0],self.sim.speed[:,0],lateral_friction=self.sim.lateral_mu[:,0],
                acceleration=self.sim.accel[:,0])
        command,_=planner.path_reference(defender[:,:2],self.sim.velocity[:,0,:2],self.sim.speed[:,0])
        distance=(waypoint-defender[:,:2]).norm(dim=-1,keepdim=True)
        command=torch.where(distance<=.3,torch.zeros_like(command),command)
        return torch.cat((command,torch.zeros((self.n,1),device=self.device)),dim=-1)

    def _strategic_target(self,action_class):
        """Resolve collect, score, and contest classes into AD* waypoint goals."""
        position=self.sim.pose[:,0,:2]
        free=self.piece_active&(self.piece_owner<0)
        distances=(self.piece_pos-position[:,None,:]).norm(dim=-1).masked_fill(~free,float("inf"))
        nearest_distance,nearest_index=distances.min(-1)
        nearest=self.piece_pos[torch.arange(self.n,device=self.device),nearest_index]
        if self.task=="defense":
            attacker=self.sim.pose[:,1,:2]
            velocity=self.sim.velocity[:,1,:2]
            speed=velocity.norm(dim=-1,keepdim=True)
            to_defender=position-attacker
            fallback=to_defender/to_defender.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            heading=torch.where(speed>.15,velocity/speed.clamp_min(.15),fallback)
            eta=((position-attacker).norm(dim=-1,keepdim=True)/self.sim.speed[:,0,None].clamp_min(.1)).clamp(0.,1.25)
            contest=attacker+velocity*eta+heading*.55
            hub_delta=self.hub_centers[None,:,:]-attacker[:,None,:]
            hub_distance=hub_delta.norm(dim=-1).clamp_min(1e-6)
            alignment=(heading[:,None,:]*hub_delta/hub_distance[...,None]).sum(-1)
            likely_hub=alignment.argmax(-1)
            target_hub=self.hub_centers[likely_hub]
            target_direction=target_hub-attacker
            target_distance=target_direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            target_direction=target_direction/target_distance
            block_lane=attacker+target_direction*torch.minimum(target_distance*.35,
                torch.full_like(target_distance,.9))
            deny=torch.where(torch.isfinite(nearest_distance)[:,None],nearest,block_lane)
            # Defender options use visible motion and game state: guard the
            # likely scoring lane, deny the nearest loose FUEL, or contest the
            # attacker's predicted position. No target is supplied to policy.
            return torch.where((action_class==0)[:,None],block_lane,
                torch.where((action_class==1)[:,None],deny,contest))
        else:
            own_hub=self.hub_centers[0].expand(self.n,-1)
            hub_direction=position-own_hub
            hub_direction=hub_direction/hub_direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            approach=own_hub+hub_direction*(.595+.5*torch.sqrt(
                self.sim.length[:,0].square()+self.sim.width[:,0].square())+.02)[:,None]
            opponent=self.sim.pose[:,1,:2]
            contested_distance=(self.piece_pos-opponent[:,None,:]).norm(dim=-1).masked_fill(~free,float("inf"))
            _,contested_index=contested_distance.min(-1)
            contest=self.piece_pos[torch.arange(self.n,device=self.device),contested_index]
            contest=torch.where(torch.isfinite(contested_distance.min(-1).values)[:,None],contest,approach)
        collect=torch.where(torch.isfinite(nearest_distance)[:,None],nearest,approach)
        return torch.where((action_class==0)[:,None],collect,
            torch.where((action_class==1)[:,None],approach,contest))

    def _strategic_opponent_target(self,action_class):
        """Resolve a historical policy's semantic choice for robot 1."""
        position=self.sim.pose[:,1,:2]
        free=self.piece_active&(self.piece_owner<0)
        distance=(self.piece_pos-position[:,None,:]).norm(dim=-1).masked_fill(~free,float("inf"))
        nearest_distance,index=distance.min(-1)
        nearest=self.piece_pos[torch.arange(self.n,device=self.device),index]
        if self.task=="defense":
            side=1
            hub=self.hub_centers[side].expand(self.n,-1)
            direction=position-hub
            direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            radius=.595+.5*torch.sqrt(self.sim.length[:,1].square()+self.sim.width[:,1].square())+.02
            approach=hub+direction*radius[:,None]
            carrying=(self.piece_active&(self.piece_owner==1)).any(-1)
            score=carrying&self.hub_active[:,side]
            collect=torch.where(torch.isfinite(nearest_distance)[:,None],nearest,approach)
            return torch.where((action_class==0)[:,None],collect,
                torch.where((action_class==1)[:,None],approach,collect))

        attacker=position
        defender=self.sim.pose[:,0,:2]
        attacker_velocity=self.sim.velocity[:,0,:2]
        speed=attacker_velocity.norm(dim=-1,keepdim=True)
        to_defender=defender-attacker
        fallback=to_defender/to_defender.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        heading=torch.where(speed>.15,attacker_velocity/speed.clamp_min(.15),fallback)
        hubs=self.hub_centers[None,:,:]-defender[:,None,:]
        hub_distance=hubs.norm(dim=-1).clamp_min(1e-6)
        likely=(heading[:,None,:]*hubs/hub_distance[...,None]).sum(-1).argmax(-1)
        route=self.hub_centers[likely]-defender
        route_distance=route.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        block_lane=defender+route/route_distance*torch.minimum(route_distance*.35,
            torch.full_like(route_distance,.9))
        eta=((attacker-defender).norm(dim=-1,keepdim=True)/self.sim.speed[:,1,None].clamp_min(.1)).clamp(0.,1.25)
        intercept=attacker+attacker_velocity*eta+heading*.55
        deny=torch.where(torch.isfinite(nearest_distance)[:,None],nearest,block_lane)
        return torch.where((action_class==0)[:,None],block_lane,
            torch.where((action_class==1)[:,None],deny,intercept))

    def _adstar_opponent_velocity(self,target):
        """Plan robot 1's route to a game-state objective."""
        planner=(self._adstar_planners if self.task=="defense"
                 else self._adstar_defender_planners)
        if planner is None:
            raise RuntimeError("strategic opponent motion requires its AD* planner")
        pose=self.sim.pose[:,1]
        self._planner_tick+=1
        if self._planner_tick%self.adstar_replan_interval==0:
            planner.plan(pose[:,:2],target,pose[:,2],self.sim.length[:,1],self.sim.width[:,1],
                self.sim.speed[:,1],self.sim.pose[:,0,:2],self.sim.velocity[:,0,:2],
                self.task=="defense",self.sim.lateral_mu[:,1],self.sim.accel[:,1])
        command,_=planner.path_reference(pose[:,:2],self.sim.velocity[:,1,:2],self.sim.speed[:,1])
        magnitude=command.norm(dim=-1,keepdim=True)
        return command*torch.minimum(torch.ones_like(magnitude),
            self.sim.speed[:,1,None]/magnitude.clamp_min(1e-6))

    def _velocity_intercept_target(self):
        """Deterministic attacker waypoint that evades the defender's predicted lane intercept."""
        attacker=self.sim.pose[:,1,:2]
        defender=self.sim.pose[:,0,:2]
        goal=self._attacker_objective()
        route=goal-attacker
        distance=route.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        direction=route/distance
        defender_along=((defender-attacker)*direction).sum(-1).clamp_min(0.)
        time_to_intercept=(defender_along/self.sim.speed[:,1].clamp_min(.1)).clamp(0.,1.2)
        predicted_defender=defender+self.sim.velocity[:,0,:2]*time_to_intercept[:,None]
        along=((predicted_defender-attacker)*direction).sum(-1).clamp_min(0.).minimum(distance[:,0])
        closest=attacker+direction*along[:,None]
        lateral=predicted_defender-closest
        threatened=(lateral.norm(dim=-1)<1.35)&(along>.3)&(along<distance[:,0]-.4)
        signed_side=direction[:,0]*lateral[:,1]-direction[:,1]*lateral[:,0]
        side=torch.where(signed_side>=0.,1.,-1.)
        perpendicular=torch.stack((-direction[:,1],direction[:,0]),-1)
        bypass=closest+direction*.9-perpendicular*(side[:,None]*1.2)
        return torch.where(threatened[:,None],bypass,goal)

    def _adstar_attacker_velocity(self):
        """Batched GPU route planning and velocity tracking; no host state copies."""
        planner=self._adstar_planners
        pose=self.sim.pose[:,1]
        self._planner_tick+=1
        # A host-side fixed cadence avoids the device synchronization caused by
        # branching on per-world contact/replan masks. Contact replanning is
        # bounded by this interval instead of synchronizing every control tick.
        if self._planner_tick % self.adstar_replan_interval == 0:
            planner.plan(pose[:,:2],self._attacker_objective(),pose[:,2],self.sim.length[:,1],
                self.sim.width[:,1],self.sim.speed[:,1],self.sim.pose[:,0,:2],
                self.sim.velocity[:,0,:2],True,self.sim.lateral_mu[:,1],self.sim.accel[:,1])
        command,_=planner.path_reference(pose[:,:2],self.sim.velocity[:,1,:2],self.sim.speed[:,1])
        magnitude=command.norm(dim=-1,keepdim=True)
        return command*torch.minimum(torch.ones_like(magnitude),
            self.sim.speed[:,1,None]/magnitude.clamp_min(1e-6))

    def _adstar_defender_velocity(self):
        """Route toward the attacker's current motion lane without goal access."""
        planner=self._adstar_defender_planners
        defender=self.sim.pose[:,1]
        attacker=self.sim.pose[:,0,:2]
        attacker_velocity=self.sim.velocity[:,0,:2]
        attacker_speed=attacker_velocity.norm(dim=-1,keepdim=True)
        to_attacker=attacker-defender[:,:2]
        fallback=to_attacker/to_attacker.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        velocity_direction=torch.where(attacker_speed>.15,
            attacker_velocity/attacker_speed.clamp_min(.15),fallback)
        defender_distance=(attacker-defender[:,:2]).norm(dim=-1,keepdim=True)
        reach_time=(defender_distance/self.sim.speed[:,1,None].clamp_min(.1)).clamp(0.,1.25)
        intercept_time=reach_time.clamp(max=1.25)
        predicted=attacker+attacker_velocity*intercept_time
        target=predicted+velocity_direction*.55
        self._planner_tick+=1
        if self._planner_tick%self.adstar_replan_interval==0:
            planner.plan(defender[:,:2],target,defender[:,2],self.sim.length[:,1],
                self.sim.width[:,1],self.sim.speed[:,1],attacker,attacker_velocity,False,
                self.sim.lateral_mu[:,1],self.sim.accel[:,1])
        command,_=planner.path_reference(defender[:,:2],self.sim.velocity[:,1,:2],self.sim.speed[:,1])
        magnitude=command.norm(dim=-1,keepdim=True)
        return command*torch.minimum(torch.ones_like(magnitude),
            self.sim.speed[:,1,None]/magnitude.clamp_min(1e-6))

    def _finish_step(self,goal,old_position,old_distance,action):
        self.sim.velocity[:,1]=torch.where(self.static_opponent_mask[:,None],
            torch.zeros_like(self.sim.velocity[:,1]),self.sim.velocity[:,1])
        p=self.sim.pose
        acquisitions,scores,denied=self._update_gamepieces()
        d=(goal-p[:,1,:2]).norm(dim=-1)
        attacker_index=0 if self.task=="counter_defense" else 1
        if self.action_mode=="strategic":
            attack_score=scores[:,attacker_index].float()
            terminated=attack_score>0
            if self.task=="counter_defense":
                reward=attack_score+.05*acquisitions[:,0].float()-.001
                score=torch.zeros_like(reward)
            else:
                reward=-attack_score-.02*acquisitions[:,1].float()-.001
                route=goal-p[:,1,:2]; rs=route.square().sum(-1).clamp_min(1e-8)
                proj=(((p[:,0,:2]-p[:,1,:2])*route).sum(-1)/rs).clamp(0,1)
                closest=p[:,1,:2]+proj[:,None]*route; lane=(p[:,0,:2]-closest).norm(dim=-1)
                score=torch.exp(-lane/.7)*torch.exp(-((proj-.55)/.35).square())
                reward+=.01*score
        elif self.task=="counter_defense":
            controlled_d=(goal-p[:,0,:2]).norm(dim=-1)
            reward=(self.previous-controlled_d)*2-.01
            terminated=controlled_d<self.goal_radius
            reward+=torch.where(terminated,10.,0.)
            self.previous.copy_(controlled_d)
            score=torch.zeros_like(reward)
        else:
            reward=(d-self.previous)*2-.01
            terminated=d<self.goal_radius
            reward+=torch.where(terminated,-10.,0.)
            route=goal-p[:,1,:2]; rs=route.square().sum(-1).clamp_min(1e-8)
            proj=(((p[:,0,:2]-p[:,1,:2])*route).sum(-1)/rs).clamp(0,1)
            closest=p[:,1,:2]+proj[:,None]*route; lane=(p[:,0,:2]-closest).norm(dim=-1)
            score=torch.exp(-lane/.7)*torch.exp(-((proj-.55)/.35).square()); reward+=.04*score
            self.previous.copy_(d)
        if self.action_mode=="strategic":
            pass
        elif self.task=="counter_defense":
            reward+=.05*acquisitions[:,0].float()+scores[:,0].float()-scores[:,1].float()
        else:
            reward-=.02*acquisitions[:,1].float()+scores[:,1].float()
        new_distance=(goal-p[:,0,:2]).norm(dim=-1) if self.task=="counter_defense" else d
        blocked=(old_distance-new_distance)<.002
        contact=self.sim.robot_contact
        contact_started=contact&~self.last_contact
        moved=(p[:,0,:2]-old_position).norm(dim=-1)
        action_delta=action-self.last_action
        smoothness=action_delta.norm(dim=-1)
        spin_rate_ratio=self.sim.velocity[:,0,2]/self.sim.omega_limit[:,0].clamp_min(.1)
        maneuver_penalty=(SPIN_RATE_PENALTY*spin_rate_ratio.square()+
                          ROTATION_COMMAND_PENALTY*action[:,2].square()+
                          ROTATION_COMMAND_DELTA_PENALTY*action_delta[:,2].square())
        wall=self.sim.wall_contact[:,0].any(-1)
        self.path_length+=moved
        self.contact_steps+=contact.float()
        self.contact_count+=contact_started.float()
        self.blocked_steps+=blocked.float()
        self.out_of_bounds_steps+=wall.float()
        self.last_contact.copy_(contact); self.last_action.copy_(action)
        truncated=self.steps>=self.horizon
        if self.action_mode=="strategic":
            success=(scores[:,0]>0) if self.task=="counter_defense" else (
                truncated&(self.fuel_score_count[:,1]==0))
        else:
            success=terminated if self.task=="counter_defense" else (truncated&~terminated)
        static_contact=self.sim.field_contact[:,0]|self.sim.wall_contact[:,0].any(-1)
        opponent_wall=self.sim.wall_contact[:,1].any(-1)
        opponent_static_contact=self.sim.field_contact[:,1]|opponent_wall
        static_contact_started=static_contact&~self.last_static_contact
        self.last_static_contact.copy_(static_contact)
        # Allow useful bumper engagement with the attacker. Charge only once
        # when the controlled robot newly hits static field geometry.
        reward-=static_contact_started.float()*.05+.001*smoothness+maneuver_penalty
        if self.task=="defense":
            reward+=torch.where(truncated&~terminated,10.,0.)
        info={"contact":contact.float(),"contact_duration":contact.float()*self.dt,
              "opponent_contact":self.sim.opponent_contact.float(),
              "field_contact":self.sim.field_contact[:,0].float(),
              "wall_contact":self.sim.wall_contact[:,0].any(-1).float(),
              "static_contact_started":static_contact_started.float(),
              "opponent_field_contact":self.sim.field_contact[:,1].float(),
              "opponent_wall_contact":opponent_wall.float(),
              "opponent_static_contact":opponent_static_contact.float(),
              "contact_count":contact_started.float(),"time_blocked":blocked.float()*self.dt,
              "time_to_goal":((self.match_elapsed if self.action_mode=="strategic" else self.steps*self.dt)*terminated).float(),"success":success.float(),
              "defensive_delay":torch.full_like(self.steps,self.dt,dtype=torch.float32) if self.task=="defense" else torch.zeros_like(self.steps,dtype=torch.float32),
              "useful_position":score,"path_length":moved,"command_smoothness":smoothness,
              "spin_rate_ratio":spin_rate_ratio.abs(),"maneuver_penalty":maneuver_penalty,
              "path_efficiency":torch.where(terminated,self.initial_distance/self.path_length.clamp_min(1e-6),torch.zeros_like(self.path_length)),
              "out_of_bounds":wall.float()}
        possession=(self.piece_active[:,:,None] &
                    (self.piece_owner[:,:,None]==torch.arange(2,device=self.device)[None,None,:])).sum(1)
        info.update({
            "fuel_acquired_event":acquisitions,
            "fuel_scored_event":scores,
            "fuel_denied_event":denied,
            "fuel_abandoned_event":self.fuel_abandoned_event,
            "fuel_acquisition_count":self.fuel_acquisition_count,
            "fuel_score_count":self.fuel_score_count,
            "fuel_denied_count":self.fuel_denied_count,
            "fuel_abandoned_count":self.fuel_abandoned_count,
            "fuel_possession_count":possession,
            "match_elapsed":self.match_elapsed,
            "match_remaining":self.match_remaining,
            "match_progress":self.match_elapsed/160.,
            "hub_active":self.hub_active,
            "game_state":{
                "piece_pos":self.piece_pos,
                "piece_vel":self.piece_vel,
                "piece_active":self.piece_active,
                "piece_owner":self.piece_owner,
                "piece_zone":self.piece_zone,
                "piece_type":self.piece_type,
                "fuel_acquired_event":acquisitions,
                "fuel_scored_event":scores,
                "fuel_denied_event":denied,
                "fuel_abandoned_event":self.fuel_abandoned_event,
                "fuel_possession_count":possession,
                "fuel_score_count":self.fuel_score_count,
                "match_elapsed":self.match_elapsed,
                "match_remaining":self.match_remaining,
                "match_progress":self.match_elapsed/160.,
                "hub_active":self.hub_active,
                "hub_centers":self.hub_centers,
                "fuel_count":self.fuel_count,
                "preloads_per_robot":self.preloads_per_robot,
                "piece_owner_codes":{"free_or_off_field":-1,"scored":-2,"robots_or_teammates": "0..5"},
                "piece_zone_codes":{"neutral":0,"red_depot":1,"blue_depot":2,
                    "red_outpost_stock":3,"blue_outpost_stock":4,"preload":5,"scored":6},
            }})
        if hasattr(self, "_last_opponent_command"):
            info["opponent_effort_vector"] = self._last_opponent_command
        if self._adstar_tactical_planner is not None:
            info["controlled_adstar_path"] = self._adstar_tactical_planner.last_path
            info["controlled_adstar_path_lengths"] = self._adstar_tactical_planner.last_lengths
        opponent_kind=self.opponent.get("value","") if isinstance(self.opponent,dict) else self.opponent
        if self._adstar_planners is not None and opponent_kind in ("adstar", "learned"):
            info["adstar_paths"]=self._adstar_planners.last_path
            info["adstar_path_lengths"]=self._adstar_planners.last_lengths
            info["predicted_intercepts"]=self._adstar_planners.last_intercept
            info["predicted_intercept_times"]=self._adstar_planners.last_intercept_time
        elif self._adstar_defender_planners is not None and opponent_kind in ("guard", "adstar_defender", "learned"):
            info["adstar_paths"]=self._adstar_defender_planners.last_path
            info["adstar_path_lengths"]=self._adstar_defender_planners.last_lengths
        if self._adstar_tactical_planner is not None:
            info["tactical_adstar_paths"]=self._adstar_tactical_planner.last_path
            info["tactical_adstar_path_lengths"]=self._adstar_tactical_planner.last_lengths
        obs=self._obs()
        self._last_observation=obs
        return obs,reward,terminated,truncated,info
