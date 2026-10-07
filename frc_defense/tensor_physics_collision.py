"""Robot, wall, obstacle, and field contact resolution for the tensor simulator."""
from __future__ import annotations

import math
import numpy as np

try:
    import torch
except ImportError:  # Keep package imports usable without the training extra.
    torch = None

if torch is not None:
    from .tensor_collision import (FUSED_COLLISION_HIP_ENABLED,
                                   wall_contacts as _wall_contacts_hip)
    from .tensor_collision_pipeline import (
        FUSED_CONTACT_PIPELINE_HIP_ENABLED,
        contact_pipeline as _contact_pipeline_hip,
    )
else:
    FUSED_COLLISION_HIP_ENABLED = False
    _wall_contacts_hip = None
    FUSED_CONTACT_PIPELINE_HIP_ENABLED = False
    _contact_pipeline_hip = None


class TensorPhysicsCollisionMixin:
    """Contact solvers shared by the vectorized simulator."""
    def _robot_collision(self,active_mask=None):
        if active_mask is None: active_mask=torch.ones(self.n,device=self.device,dtype=torch.bool)
        if self.num_robots != 2:
            return self._robot_collision_multi(active_mask)
        # Vectorized SAT over four face normals for the pair of oriented rectangles.
        pos=self.pose[:,:,:2]; th=self.pose[:,:,2]; c,s=th.cos(),th.sin()
        axes=torch.stack((torch.stack((c[:,0],s[:,0]),-1),torch.stack((-s[:,0],c[:,0]),-1),torch.stack((c[:,1],s[:,1]),-1),torch.stack((-s[:,1],c[:,1]),-1)),1)
        u=torch.stack((c,s),-1); v=torch.stack((-s,c),-1)
        # axis [N,4,2], robot basis [N,2,2], projected support radii
        proj_u=torch.einsum('nkd,nrd->nkr',axes,u).abs(); proj_v=torch.einsum('nkd,nrd->nkr',axes,v).abs()
        radii=(proj_u*self.length[:,None,:]/2+proj_v*self.width[:,None,:]/2).sum(-1)
        delta=pos[:,1]-pos[:,0]; signed=(axes*delta[:,None,:]).sum(-1)
        overlap=radii-signed.abs()
        valid=(overlap.min(-1).values>0)&active_mask
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

    def _robot_collision_multi(self, active_mask):
        """Resolve all robot pairs in parallel using rectangle SAT contacts."""
        if self._fused_robot_collision_multi_hip_enabled:
            # One HIP work item resolves all 15 pair contacts per world. The
            # fallback remains the Torch reference on CPU/CUDA or if HIP cannot
            # build the fused extension.
            from . import tensor_collision_multi_hip
            if tensor_collision_multi_hip.robot_contacts(self, active_mask):
                self._fused_robot_collision_multi_hip_used = True
                return
        pair_i, pair_j = self._collision_pair_i, self._collision_pair_j
        pose, velocity = self.pose, self.velocity
        pos_i, pos_j = pose[:, pair_i, :2], pose[:, pair_j, :2]
        theta_i, theta_j = pose[:, pair_i, 2], pose[:, pair_j, 2]
        ci, si, cj, sj = theta_i.cos(), theta_i.sin(), theta_j.cos(), theta_j.sin()
        ui=torch.stack((ci,si),-1); vi=torch.stack((-si,ci),-1)
        uj=torch.stack((cj,sj),-1); vj=torch.stack((-sj,cj),-1)
        axes=torch.stack((ui,vi,uj,vj),-2)
        pu=(axes*ui[:,:,None,:]).sum(-1).abs()
        pv=(axes*vi[:,:,None,:]).sum(-1).abs()
        qu=(axes*uj[:,:,None,:]).sum(-1).abs()
        qv=(axes*vj[:,:,None,:]).sum(-1).abs()
        hi_l=self.length[:,pair_i]/2; hi_w=self.width[:,pair_i]/2
        hj_l=self.length[:,pair_j]/2; hj_w=self.width[:,pair_j]/2
        radii=pu*hi_l[:,:,None]+pv*hi_w[:,:,None]+qu*hj_l[:,:,None]+qv*hj_w[:,:,None]
        delta=pos_j-pos_i
        signed=(axes*delta[:,:,None,:]).sum(-1)
        overlap=radii-signed.abs()
        depth, axis_index=overlap.min(-1)
        valid=(depth>0)&active_mask[:,None]
        normal=axes.gather(2,axis_index[:,:,None,None].expand(-1,-1,1,2)).squeeze(2)
        normal=normal*torch.where(signed.gather(2,axis_index[:,:,None]).squeeze(2)>=0,1.,-1.)[...,None]
        inv_i=self.mass[:,pair_i].reciprocal(); inv_j=self.mass[:,pair_j].reciprocal()
        total=(inv_i+inv_j).clamp_min(1e-8)
        correction=(depth.clamp_min(0)+1e-4)*valid
        move_i=-normal*(correction*inv_i/total)[...,None]
        move_j= normal*(correction*inv_j/total)[...,None]
        position_delta=torch.zeros_like(pose[...,:2])
        position_delta.scatter_add_(1,pair_i[None,:,None].expand(self.n,-1,2),move_i)
        position_delta.scatter_add_(1,pair_j[None,:,None].expand(self.n,-1,2),move_j)
        pose[...,:2].add_(position_delta)

        # Contact points are midway between the oriented bumper support points.
        ui=torch.stack((ci,si),-1); vi=torch.stack((-si,ci),-1)
        uj=torch.stack((cj,sj),-1); vj=torch.stack((-sj,cj),-1)
        support_i=(pos_i+torch.sign((normal*ui).sum(-1))[...,None]*ui*hi_l[...,None]+
                   torch.sign((normal*vi).sum(-1))[...,None]*vi*hi_w[...,None])
        support_j=(pos_j-torch.sign((normal*uj).sum(-1))[...,None]*uj*hj_l[...,None]-
                   torch.sign((normal*vj).sum(-1))[...,None]*vj*hj_w[...,None])
        cp=.5*(support_i+support_j); r_i=cp-pos_i; r_j=cp-pos_j
        vi_c=velocity[:,pair_i,:2]+velocity[:,pair_i,2,None]*torch.stack((-r_i[...,1],r_i[...,0]),-1)
        vj_c=velocity[:,pair_j,:2]+velocity[:,pair_j,2,None]*torch.stack((-r_j[...,1],r_j[...,0]),-1)
        rel=vj_c-vi_c; vn=(rel*normal).sum(-1)
        inertia=(self.mass*(self.length.square()+self.width.square())/12)*self.yaw_inertia_multiplier
        ii=inertia[:,pair_i].clamp_min(1e-8); ij=inertia[:,pair_j].clamp_min(1e-8)
        cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
        rn_i=cross(r_i,normal); rn_j=cross(r_j,normal)
        eff=total+rn_i.square()/ii+rn_j.square()/ij
        jn=torch.where(valid&(vn<0),-(1.05*vn)/eff.clamp_min(1e-8),0.)
        imp_n=jn[...,None]*normal
        vi_c=velocity[:,pair_i,:2]+velocity[:,pair_i,2,None]*torch.stack((-r_i[...,1],r_i[...,0]),-1)
        vj_c=velocity[:,pair_j,:2]+velocity[:,pair_j,2,None]*torch.stack((-r_j[...,1],r_j[...,0]),-1)
        rel=vj_c-vi_c; tangent=torch.stack((-normal[...,1],normal[...,0]),-1)
        vt=(rel*tangent).sum(-1); rt_i=cross(r_i,tangent); rt_j=cross(r_j,tangent)
        eff_t=total+rt_i.square()/ii+rt_j.square()/ij
        mu=torch.sqrt(self.mu[:,None].clamp_min(0))
        jt=(-vt/eff_t.clamp_min(1e-8)).clamp(-mu*jn,mu*jn)
        impulse=imp_n+jt[...,None]*tangent
        dv_i=-impulse*inv_i[...,None]; dv_j=impulse*inv_j[...,None]
        dw_i=-(jn*rn_i+jt*rt_i)/ii; dw_j=(jn*rn_j+jt*rt_j)/ij
        velocity_delta=torch.zeros_like(velocity[...,:2])
        velocity_delta.scatter_add_(1,pair_i[None,:,None].expand(self.n,-1,2),dv_i)
        velocity_delta.scatter_add_(1,pair_j[None,:,None].expand(self.n,-1,2),dv_j)
        omega_delta=torch.zeros_like(velocity[...,2])
        omega_delta.scatter_add_(1,pair_i[None,:].expand(self.n,-1),dw_i)
        omega_delta.scatter_add_(1,pair_j[None,:].expand(self.n,-1),dw_j)
        velocity[...,:2].add_(velocity_delta); velocity[...,2].add_(omega_delta)
        self.robot_contact |= valid.any(-1)
        cross_alliance=self.team_ids[pair_i] != self.team_ids[pair_j]
        self.opponent_contact |= (valid & cross_alliance[None,:]).any(-1)

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

    def _walls(self,active_mask=None):
        if active_mask is None: active_mask=torch.ones(self.n,device=self.device,dtype=torch.bool)
        if self.num_robots in (2, 6) and self._fused_wall_collision_hip_enabled and _wall_contacts_hip is not None:
            active_mask=torch.as_tensor(active_mask,device=self.device,dtype=torch.bool).reshape(self.n).contiguous()
            if _wall_contacts_hip(self,active_mask):
                self._fused_wall_collision_hip_used = True
                return
        self._walls_torch(active_mask)

    def _walls_torch(self,active_mask=None):
        """Torch reference for the sequential x-min, x-max, y-min, y-max contacts."""
        if active_mask is None: active_mask=torch.ones(self.n,device=self.device,dtype=torch.bool)
        theta=self.pose[...,2]; c,s=theta.cos(),theta.sin()
        hl,hw=self.length/2,self.width/2
        hx=c.abs()*hl+s.abs()*hw; hy=s.abs()*hl+c.abs()*hw
        x,y=self.pose[...,0],self.pose[...,1]
        walls=((0,hx-x,1.),(0,x+hx-self.field_length,-1.),
               (1,hy-y,1.),(1,y+hy-self.field_width,-1.))
        u=torch.stack((c,s),-1); v=torch.stack((-s,c),-1)
        for axis,penetration,sign in walls:
            normal=torch.zeros_like(self.pose[...,:2]); normal[...,axis]=sign
            active=(penetration>0)&active_mask[:,None]
            self.pose[...,:2].add_(torch.where(active[...,None],normal*(penetration.clamp_min(0)[...,None]+1e-4),
                                                torch.zeros_like(normal)))
            toward_wall=-normal
            point=(self.pose[...,:2]+torch.sign((toward_wall*u).sum(-1))[...,None]*u*hl[...,None]+
                   torch.sign((toward_wall*v).sum(-1))[...,None]*v*hw[...,None])
            self._apply_static_contact_impulse(point,normal,active,self.wall_mu[:,None])
        self.wall_contact[:,:,axis]|=active

    def _obstacle_collision(self,active_mask=None):
        if active_mask is None: active_mask=torch.ones(self.n,device=self.device,dtype=torch.bool)
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
        depth,k=penetration.max(-1); contact=(depth>0)&active_mask[:,None]
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

    def _field_collision(self,active_mask=None):
        if active_mask is None: active_mask=torch.ones(self.n,device=self.device,dtype=torch.bool)
        """Resolve the most penetrating oriented-robot/field-box pair per robot."""
        if not self.field_colliders.shape[0]: return
        if self._fused_field_collision_multi_hip_enabled:
            from . import tensor_collision_multi_hip
            if tensor_collision_multi_hip.field_contacts(self,active_mask):
                self._fused_field_collision_multi_hip_used = True
                return
        boxes=self.field_colliders; center,half=boxes[:,:2],boxes[:,2:]
        if self.device.type == "cpu":
            # The chassis AABB gives a conservative broadphase for the exact
            # OBB-vs-AABB SAT calculation below. Most swept
            # substeps are nowhere near a field element, and the remaining
            # substeps generally touch only one or two boxes.
            pose=self.pose.detach().numpy()
            theta=pose[...,2]
            half_length=.5*self.length.detach().numpy()
            half_width=.5*self.width.detach().numpy()
            abs_cos=np.abs(np.cos(theta)); abs_sin=np.abs(np.sin(theta))
            extent_x=abs_cos*half_length+abs_sin*half_width
            extent_y=abs_sin*half_length+abs_cos*half_width
            box_values=boxes.detach().numpy()
            delta=np.abs(pose[:,:,:2][:,:,None,:]-box_values[None,None,:,:2])
            possible=((delta[...,0] <= extent_x[:,:,None]+box_values[None,None,:,2]+1.e-6) &
                      (delta[...,1] <= extent_y[:,:,None]+box_values[None,None,:,3]+1.e-6))
            possible &= active_mask.detach().numpy()[:,None,None]
            box_mask=possible.any(axis=(0,1))
            if not box_mask.any():
                return
            if not box_mask.all():
                boxes=boxes.index_select(0,torch.from_numpy(np.flatnonzero(box_mask)))
                center,half=boxes[:,:2],boxes[:,2:]
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
        contact=(depth>=0)&active_mask[:,None]
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

    def _field_sweep_collision(self,active_mask,sweep_steps,substep_dt):
        """Try the single-launch six-robot field sweep implementation."""
        if not self._fused_field_collision_multi_hip_enabled:
            return False
        from . import tensor_collision_multi_hip
        return tensor_collision_multi_hip.field_sweep_contacts(
            self,active_mask,sweep_steps,substep_dt)

    def _field_sweep_torch(self,active_mask,sweep_steps,substep_dt):
        """Torch reference for the HIP field-sweep position/contact loop."""
        for _ in range(sweep_steps):
            self.pose.copy_(torch.where(active_mask[:,None,None],
                self.pose+self.velocity*substep_dt,self.pose))
            wrapped=torch.remainder(self.pose[...,2]+math.pi,2*math.pi)-math.pi
            self.pose[...,2].copy_(torch.where(active_mask[:,None],wrapped,self.pose[...,2]))
            self._field_collision(active_mask)
