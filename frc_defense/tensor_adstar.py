"""Batched device-resident grid planner for tensor simulation rollouts.

The planner uses parallel Bellman sweeps on a clearance-inflated 8-connected
grid, then follows the resulting potential with batched greedy extraction.
It intentionally keeps all search state on the selected torch device.
"""
from __future__ import annotations

import math

import torch


class TensorADStar:
    """GPU batched, footprint-aware A* style potential planner."""

    def __init__(self, env, resolution: float = .3, max_points: int = 72,
                 sweeps: int = 72, avoid_bumps: bool = False):
        self.device = env.device
        self.dtype = env.sim.pose.dtype
        self.n = env.n
        self.resolution = float(resolution)
        self.nx = int(math.ceil(env.sim.field_length / resolution))
        self.ny = int(math.ceil(env.sim.field_width / resolution))
        self.max_points = int(max_points)
        self.sweeps = int(sweeps)
        self.avoid_bumps = bool(avoid_bumps)
        self.length = env.sim.field_length
        self.width = env.sim.field_width
        self.x = (torch.arange(self.nx, device=self.device, dtype=self.dtype) + .5) * resolution
        self.y = (torch.arange(self.ny, device=self.device, dtype=self.dtype) + .5) * resolution
        self.xx, self.yy = torch.meshgrid(self.x, self.y, indexing="ij")
        self._boxes = torch.as_tensor(env.sim.field_colliders, device=self.device,
                                      dtype=self.dtype).reshape(-1, 4)
        self._extra_circles = env.sim.obstacles
        self._bumps = torch.as_tensor(
            [(b.x, b.y, b.length / 2, b.width / 2) for b in env.field_boxes
            if "bump" in getattr(b, "name", "")], device=self.device,
            dtype=self.dtype).reshape(-1, 4)
        self._neighbors = ((1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.),
                           (1, 1, 1.41421356237), (1, -1, 1.41421356237),
                           (-1, 1, 1.41421356237), (-1, -1, 1.41421356237))
        self.last_path = torch.zeros((self.n, self.max_points, 2), device=self.device,
                                     dtype=self.dtype)
        self.last_lengths = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.last_intercept = torch.zeros((self.n, 2), device=self.device, dtype=self.dtype)
        self.last_intercept_time = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_start = torch.zeros((self.n, 2), device=self.device, dtype=self.dtype)
        self.last_goal = torch.zeros_like(self.last_start)
        self.last_heading = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.last_speed_profile = torch.zeros((self.n,self.max_points),device=self.device,dtype=self.dtype)
        self.potential = torch.empty((self.n, self.nx, self.ny), device=self.device,
                                     dtype=self.dtype)

    def _blocked(self, heading, length, width, dynamic=None):
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
        if self.avoid_bumps and self._bumps.numel():
            dx = (self.xx[None, None] - self._bumps[None, :, 0, None, None]).abs()
            dy = (self.yy[None, None] - self._bumps[None, :, 1, None, None]).abs()
            bump_blocked = ((dx <= self._bumps[None, :, 2, None, None] + ex[:, None]) &
                            (dy <= self._bumps[None, :, 3, None, None] + ey[:, None]))
            blocked |= bump_blocked.any(dim=1)
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

    def _indices(self, points):
        ix = (points[:, 0] / self.resolution).floor().long().clamp(0, self.nx - 1)
        iy = (points[:, 1] / self.resolution).floor().long().clamp(0, self.ny - 1)
        return ix, iy

    def plan(self, start, goal, heading, length, width, speed=None,
             defender=None, defender_velocity=None, dynamic_defender=False,
             lateral_friction=None, acceleration=None):
        """Plan for every world and return fixed-size paths plus intercept data."""
        heading = heading.reshape(self.n).to(self.dtype)
        start, goal = start.to(self.dtype), goal.to(self.dtype)
        if dynamic_defender:
            delta = goal - start
            unit = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            va = unit * (speed if speed is not None else 4.).reshape(self.n, 1)
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
            dynamic = torch.cat((intercept, torch.full((self.n, 2), .45, device=self.device,
                                                       dtype=self.dtype)), -1)
        else:
            t = torch.zeros(self.n, device=self.device, dtype=self.dtype)
            intercept = torch.zeros_like(start)
            dynamic = None
        blocked = self._blocked(heading, length, width, dynamic)
        # Do not let conservative inflation trap a route's source. AD* defender
        # targets can land on a bump while projecting an interception point;
        # move those endpoints to the nearest reachable clearance cell.
        sx, sy = self._indices(start); gx, gy = self._indices(goal)
        batch = torch.arange(self.n, device=self.device)
        if self.avoid_bumps:
            blocked_goal=blocked[batch,gx,gy]
            goal_distance=((self.xx.reshape(1,-1)-goal[:,0,None]).square()+
                           (self.yy.reshape(1,-1)-goal[:,1,None]).square())
            nearest=goal_distance.masked_fill(blocked.flatten(1),float("inf")).argmin(-1)
            nearest_x=nearest//self.ny
            nearest_y=nearest%self.ny
            projected=torch.stack((self.x[nearest_x],self.y[nearest_y]),-1)
            goal=torch.where(blocked_goal[:,None],projected,goal)
            gx,gy=self._indices(goal)
        blocked[batch, sx, sy] = False
        blocked[batch, gx, gy] = False
        bump_cost = torch.ones((self.n, self.nx, self.ny), device=self.device, dtype=self.dtype)
        if self._bumps.numel():
            bump_dx = (self.xx[None, None] - self._bumps[None, :, 0, None, None]).abs()
            bump_dy = (self.yy[None, None] - self._bumps[None, :, 1, None, None]).abs()
            in_bump = ((bump_dx <= self._bumps[None, :, 2, None, None]) &
                       (bump_dy <= self._bumps[None, :, 3, None, None]))
            bump_cost += in_bump.any(1).to(self.dtype) * .12
        inf = torch.full_like(bump_cost, 1.e6)
        value = torch.where(blocked, inf, inf.clone())
        value[batch, gx, gy] = 0.
        # Jacobi Bellman sweeps are parallel across both cells and worlds.
        import torch.nn.functional as F
        edge_cost=torch.tensor((1.41421356237,1.,1.41421356237,1.,
                                1.,1.,1.41421356237,1.,1.41421356237),
                               device=self.device,dtype=self.dtype)[None,:,None]
        blocked_neighbors=F.unfold(
            F.pad(blocked[:,None].to(self.dtype),(1,1,1,1),value=1.),3
        ).view(self.n,9,self.nx*self.ny)>.5
        corner_clear=torch.ones_like(blocked_neighbors)
        corner_clear[:,0]=~(blocked_neighbors[:,1]|blocked_neighbors[:,3])
        corner_clear[:,2]=~(blocked_neighbors[:,1]|blocked_neighbors[:,5])
        corner_clear[:,6]=~(blocked_neighbors[:,3]|blocked_neighbors[:,7])
        corner_clear[:,8]=~(blocked_neighbors[:,5]|blocked_neighbors[:,7])
        for _ in range(self.sweeps):
            padded=F.pad(value[:,None],(1,1,1,1),value=1.e6)
            neighbors=F.unfold(padded,3).view(self.n,9,self.nx*self.ny)
            candidate=neighbors+edge_cost*bump_cost.flatten(1)[:,None,:]
            candidate=candidate.masked_fill(~corner_clear,float("inf"))
            value=torch.minimum(value.flatten(1),candidate.amin(1)).view(self.n,self.nx,self.ny)
            value=torch.where(blocked,inf,value)
        self.potential.copy_(value)
        path = torch.zeros((self.n, self.max_points, 2), device=self.device, dtype=self.dtype)
        path[:, 0] = start
        px, py = sx.clone(), sy.clone()
        active = torch.ones(self.n, device=self.device, dtype=torch.bool)
        lengths = torch.ones(self.n, device=self.device, dtype=torch.long)
        rows = batch
        for k in range(1, self.max_points):
            candidates=[]
            for dx, dy, _cost in self._neighbors:
                nx, ny = px + dx, py + dy
                valid=(nx>=0)&(nx<self.nx)&(ny>=0)&(ny<self.ny)
                vx,vy=nx.clamp(0,self.nx-1),ny.clamp(0,self.ny-1)
                if dx and dy:
                    side_x=(px+dx).clamp(0,self.nx-1)
                    side_y=(py+dy).clamp(0,self.ny-1)
                    valid &= ~(blocked[rows,side_x,py]|blocked[rows,px,side_y])
                score=value[rows,vx,vy]
                candidates.append(torch.where(valid & ~blocked[rows,vx,vy],score,inf[:,0,0]))
            scores=torch.stack(candidates,-1)
            choice=scores.argmin(-1)
            next_score=scores.gather(1,choice[:,None]).squeeze(1)
            offsets=torch.tensor([(dx,dy) for dx,dy,_ in self._neighbors],device=self.device)
            off=offsets[choice]
            nx,ny=px+off[:,0],py+off[:,1]
            progressing=active & (next_score < value[rows,px,py] - 1e-5)
            px=torch.where(progressing,nx,px); py=torch.where(progressing,ny,py)
            point=torch.stack(((px.to(self.dtype)+.5)*self.resolution,
                               (py.to(self.dtype)+.5)*self.resolution),-1)
            path[:,k]=torch.where(progressing[:,None],point,path[:,k-1])
            lengths += progressing.long()
            reached=(px==gx)&(py==gy)
            active &= progressing & ~reached
        self.last_path.copy_(path); self.last_lengths.copy_(lengths)
        self.last_intercept.copy_(intercept); self.last_intercept_time.copy_(t)
        self.last_start.copy_(start); self.last_goal.copy_(goal); self.last_heading.copy_(heading)
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
        point_curve=torch.zeros((self.n,self.max_points),device=self.device,dtype=self.dtype)
        point_curve[:,1:-1]=curvature
        mu=(lateral_friction if lateral_friction is not None else torch.full_like(length,1.2)).clamp_min(.1)
        accel=(acceleration if acceleration is not None else torch.full_like(length,8.)).clamp_min(.2)
        vcap=(speed if speed is not None else torch.full_like(length,4.5)).clamp_min(.1)[:,None]
        lateral=(.65*mu[:,None]*9.81/point_curve.clamp_min(1.e-6)).sqrt()
        steer=12./point_curve.clamp_min(1.e-6)
        caps=torch.minimum(vcap,torch.minimum(lateral,steer))
        caps=torch.where(point_curve>1.e-6,caps,vcap)
        caps[batch,lengths-1]=0.
        station=torch.cat((torch.zeros((self.n,1),device=self.device,dtype=self.dtype),distance.cumsum(-1)),-1)
        delta_station=station[:,None,:]-station[:,:,None]
        future=torch.arange(self.max_points,device=self.device)[None,None,:]>=torch.arange(self.max_points,device=self.device)[None,:,None]
        valid=(torch.arange(self.max_points,device=self.device)[None,None,:]<lengths[:,None,None])&future
        braking=.65*accel[:,None,None]
        reachable=(caps[:,None,:].square()+2*braking*delta_station.clamp_min(0)).sqrt()
        profile=torch.where(valid,reachable,torch.full_like(reachable,float("inf"))).amin(-1)
        self.last_speed_profile.copy_(profile)
        return path, lengths, intercept, t

    def path_reference(self, position, velocity, speed_limit):
        """Batched progress projection, lookahead, and speed-aware path command."""
        path=self.last_path
        d2=(path-position[:,None,:]).square().sum(-1)
        idx=d2.argmin(-1)
        next_idx=(idx+1).clamp_max(self.max_points-1)
        batch=torch.arange(self.n,device=self.device)
        here=path[batch,idx]; ahead=path[batch,next_idx]
        tangent=ahead-here
        # If the route has ended or a path has not yet been planned, head toward goal.
        tangent=torch.where((self.last_lengths<=1)[:,None],self.last_goal-position,tangent)
        tangent=tangent/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        normal=torch.stack((-tangent[:,1],tangent[:,0]),-1)
        cross=((position-here)*normal).sum(-1)
        cross_v=(velocity*normal).sum(-1)
        lateral=(-3.*cross-4.*cross_v).clamp(-.75,.75)
        planned=self.last_speed_profile[batch,idx].clamp(max=speed_limit).clamp_min(.2)
        current=velocity.norm(dim=-1)
        target=(planned-.55*(current-planned).clamp_min(0)).clamp_min(0)
        command=tangent*target[:,None]+normal*lateral[:,None]
        return command, tangent

    def defender_spawn(self, starts, mask, generator):
        """Choose a randomized point near the computed clear route on-device."""
        distance=(self.last_path-starts[:,None,:]).norm(dim=-1)
        valid=(torch.arange(self.max_points,device=self.device)[None,:] < self.last_lengths[:,None])
        eligible=valid&(distance>=1.25)&(distance<=2.75)
        # Random ranks among eligible points, with an ahead-of-start fallback.
        weights=torch.rand(eligible.shape,device=self.device,generator=generator)
        weights=weights.masked_fill(~eligible,-1.)
        index=weights.argmax(-1)
        fallback=(valid&(distance>=1.1)).float().argmax(-1)
        index=torch.where(eligible.any(-1),index,fallback)
        point=self.last_path[torch.arange(self.n,device=self.device),index]
        point=torch.where(mask[:,None],point,starts)
        jitter=torch.rand((self.n,2),device=self.device,generator=generator)-.5
        point=point+jitter*.18
        return point
