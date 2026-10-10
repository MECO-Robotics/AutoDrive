"""Coupled spherical fuel dynamics and a deliberately readable Torch reference.

Fuel is three dimensional; the existing robot chassis remains planar. Compliance
models foam compression, not deformation of the collision sphere. Coefficients
are calibration defaults, not measurements of competition fuel. Wheel climbing
requires a chassis model with vertical degrees of freedom and is not represented.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass
class FuelPhysicsConfig:
    radius: float = .075
    mass: float = .215
    substeps: int = 10
    gravity: float = 9.81
    stiffness: float = 2000.
    damping: float = 40.
    friction: float = .45
    rolling_friction: float = .20
    robot_height: float = .5
    broadphase: str = "grid"
    robot_broadphase: str = "all_pairs"
    cache: bool = True
    skin: float = .075
    max_neighbors: int = 64
    contact_mode: str = "fused"
    compact_capacity: str = "bounded"
    solver: str = "jacobi"
    backend: str = "auto"
    sleep: bool = True
    sleep_islands: bool = False
    sleep_time: float = .5
    sleep_speed: float = .025
    sleep_angular_speed: float = .5
    adaptive_substeps: bool = True
    max_compression_fraction: float = .5
    max_translation_per_substep: float | None = None

    def __post_init__(self):
        for name, choices in (("broadphase", ("grid", "all_pairs")),
                              ("robot_broadphase", ("grid", "all_pairs")),
                              ("contact_mode", ("compact", "fused")),
                              ("compact_capacity", ("bounded", "full")),
                              ("solver", ("jacobi", "colored")),
                              ("backend", ("auto", "torch", "hip"))):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        if not isinstance(self.substeps, int) or isinstance(self.substeps, bool) or not isinstance(self.max_neighbors, int):
            raise ValueError("substeps and max_neighbors must be integers")
        numeric = (self.radius,self.mass,self.gravity,self.stiffness,self.damping,self.friction,
                   self.rolling_friction,self.robot_height,self.skin,self.sleep_time,
                   self.sleep_speed,self.sleep_angular_speed,self.max_compression_fraction)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("fuel coefficients must be finite")
        if not 0 < self.max_compression_fraction <= 1:
            raise ValueError("max_compression_fraction must be in (0,1]")
        if self.max_translation_per_substep is not None and (not math.isfinite(self.max_translation_per_substep) or self.max_translation_per_substep<=0):
            raise ValueError("max_translation_per_substep must be positive and finite")
        if self.radius <= 0 or self.mass <= 0 or self.substeps < 1 or self.max_neighbors < 1:
            raise ValueError("positive radius, mass, substeps and max_neighbors required")
        if min(self.skin, self.stiffness, self.damping, self.friction,
               self.rolling_friction, self.sleep_time, self.sleep_speed,
               self.sleep_angular_speed) < 0:
            raise ValueError("contact and sleep coefficients must be nonnegative")


class FuelPhysics:
    def __init__(self, sim, piece_pos, piece_vel, config=None):
        self.sim = sim
        self.config = FuelPhysicsConfig(**config) if isinstance(config, dict) else (config or FuelPhysicsConfig())
        self.piece_pos, self.piece_vel = piece_pos, piece_vel
        if piece_pos.ndim != 3 or piece_pos.shape[-1] != 2 or piece_vel.shape != piece_pos.shape:
            raise ValueError("piece positions and velocities must be [world,piece,2]")
        self.n, self.p = piece_pos.shape[:2]
        self.device = piece_pos.device
        self.pos = torch.zeros((self.n, self.p, 3), device=self.device, dtype=piece_pos.dtype)
        self.vel = torch.zeros_like(self.pos)
        self.angular = torch.zeros_like(self.pos)
        self.sleeping = torch.zeros((self.n, self.p), device=self.device, dtype=torch.bool)
        self.sleep_clock = torch.zeros_like(self.sleeping, dtype=piece_pos.dtype)
        self._last_xy = piece_pos.clone()
        self._last_vxy = piece_vel.clone()
        self._reference = self.pos.clone()
        self._free = torch.zeros_like(self.sleeping)
        self._pairs = torch.empty((0, 3), device=self.device, dtype=torch.long)
        self._valid_cache = False
        self.stats = dict(rebuilds=0, overflows=0, candidate_pairs=0, backend="torch")
        self._hip = None
        if self.device.type == "cuda" and self.config.backend != "torch":
            try:
                from .fuel_physics_hip import FuelHIP
            except ImportError:
                # The HIP implementation is an optional runtime extension.
                # Keep the supported Torch reference path usable when its
                # source module or compiled extension is unavailable.
                FuelHIP = None
            if FuelHIP is not None:
                self._hip = FuelHIP(self)
                if not self._hip.available:
                    self._hip = None
        if self.config.backend == "hip" and self._hip is None:
            raise RuntimeError("requested HIP fuel backend unavailable")
        self.reset()
        if self.config.sleep and self.config.sleep_islands:
            from .fuel_physics_sleep import initialize_island_sleep
            initialize_island_sleep(self)

    @property
    def candidate_pairs(self):
        return self._pairs

    def _spawn_height(self, xy):
        """Place released ground fuel on the analytic terrain contact plane."""
        bumps=getattr(self.sim,"bump_regions",())
        if len(bumps)==0:
            return self.config.radius
        if self._hip is not None:
            return self._hip.spawn_height(xy)
        from .field import BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE
        height = torch.full_like(xy[..., 0], self.config.radius)
        for bump in getattr(self.sim, "bump_regions", ()):
            dx = xy[..., 0]-bump[0]
            within = (dx.abs()<=bump[2]) & ((xy[..., 1]-bump[1]).abs()<=bump[3])
            terrain = BUMP_PANEL_THICKNESS+BUMP_RAMP_RISE*(1-dx.abs()/bump[2])
            slope = -dx.sign()*BUMP_RAMP_RISE/bump[2]
            surface = terrain+self.config.radius*(1+slope.square()).sqrt()
            height = torch.where(within, height.maximum(surface), height)
        return height

    def reset(self, mask=None):
        mask = torch.ones(self.n, device=self.device, dtype=torch.bool) if mask is None else torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        self.pos[mask, :, :2] = self.piece_pos[mask]
        self.pos[mask, :, 2] = self._spawn_height(self.piece_pos[mask])
        self.vel[mask, :, :2] = self.piece_vel[mask]
        self.vel[mask, :, 2] = 0
        self.angular[mask] = 0
        self.sleeping[mask] = False
        self.sleep_clock[mask] = 0
        self._last_xy[mask] = self.piece_pos[mask]
        self._last_vxy[mask] = self.piece_vel[mask]
        self._valid_cache = False
        if self._hip is not None:
            self._hip.invalidate()
        if self.config.sleep and self.config.sleep_islands:
            from .fuel_physics_sleep import reset_island_sleep
            reset_island_sleep(self,mask)

    def commit_external(self, mask=None):
        """Accept synchronized state only for the deliberately updated slots."""
        if mask is None:
            self._last_xy.copy_(self.piece_pos)
            self._last_vxy.copy_(self.piece_vel)
        else:
            selected=torch.as_tensor(mask,device=self.device,dtype=torch.bool)[...,None]
            self._last_xy.copy_(torch.where(selected,self.piece_pos,self._last_xy))
            self._last_vxy.copy_(torch.where(selected,self.piece_vel,self._last_vxy))

    def separate_spawned(self, selected, free):
        """Nudge crowded spawn contacts apart without adding kinetic energy."""
        selected = torch.as_tensor(selected, device=self.device, dtype=torch.bool)
        free = torch.as_tensor(free, device=self.device, dtype=torch.bool)
        if selected.shape != (self.n, self.p) or free.shape != selected.shape:
            raise ValueError("selected and free masks must have shape [world,piece]")
        # Stream capture cannot record data-dependent compaction. Use the
        # fixed-shape masked formulation only while capturing a simulator step.
        if (self.device.type == "cuda" and
                torch.cuda.is_current_stream_capturing()):
            self._separate_spawned_dense(selected, free)
            return
        i, j = torch.triu_indices(self.p, self.p, offset=1, device=self.device)
        pair_selected = selected[:, i] | selected[:, j]
        valid = free[:, i] & free[:, j] & pair_selected
        delta = self.pos[:, i, :2] - self.pos[:, j, :2]
        distance = delta.norm(dim=-1)
        contact = valid & (distance < 2*self.config.radius)
        world, row = contact.nonzero(as_tuple=True)
        if not world.numel():
            return
        a, b = i[row], j[row]
        delta = delta[world, row]
        distance = distance[world, row]
        normal = delta / distance.clamp_min(1e-8)[:, None]
        fallback = torch.zeros_like(normal); fallback[:, 0] = 1.
        normal = torch.where((distance < 1e-8)[:, None], fallback, normal)
        penetration = 2*self.config.radius-distance
        # Apply only a small fraction of each overlap, then cap the combined
        # displacement per ball. Dense contact counts cannot stack into a kick.
        correction = (penetration*.10).clamp_max(.005)
        displacement = correction[:, None]*normal
        flat_delta = torch.zeros((self.n*self.p, 2), device=self.device,
                                 dtype=self.pos.dtype)
        flat_delta.index_add_(0, world*self.p+a, displacement)
        flat_delta.index_add_(0, world*self.p+b, -displacement)
        flat_delta = flat_delta.reshape(self.n, self.p, 2)
        magnitude = flat_delta.norm(dim=-1, keepdim=True)
        flat_delta *= (.015/magnitude.clamp_min(.015))
        self.pos[..., :2].add_(flat_delta)
        self.piece_pos.add_(flat_delta)
        self._last_xy.copy_(self.piece_pos)
        affected = flat_delta.norm(dim=-1) > 0
        self.sleeping &= ~affected
        self.sleep_clock.masked_fill_(affected, 0.)
        # Ferry placement can start with a small geometric overlap. Treat
        # that as a soft landing: absorb most of the carried ball's launch
        # speed before the ordinary contact solver takes over.
        spawned_world, spawned_piece = selected.nonzero(as_tuple=True)
        touching = (free[spawned_world] &
                    (torch.arange(self.p, device=self.device)[None] !=
                     spawned_piece[:, None]) &
                    ((self.pos[spawned_world, spawned_piece, None, :2] -
                      self.pos[spawned_world, :, :2]).norm(dim=-1) <
                     2*self.config.radius)).any(-1)
        self.vel[spawned_world[touching], spawned_piece[touching], :2] *= .2
        self.piece_vel[spawned_world[touching], spawned_piece[touching]] = \
            self.vel[spawned_world[touching], spawned_piece[touching], :2]
        self._last_vxy[spawned_world[touching], spawned_piece[touching]] = \
            self.piece_vel[spawned_world[touching], spawned_piece[touching]]

    def _separate_spawned_dense(self, selected, free):
        """Fixed-shape equivalent of ``separate_spawned`` for graph capture."""
        i, j = torch.triu_indices(self.p, self.p, offset=1, device=self.device)
        pair_selected = selected[:, i] | selected[:, j]
        valid = free[:, i] & free[:, j] & pair_selected
        delta = self.pos[:, i, :2] - self.pos[:, j, :2]
        distance = delta.norm(dim=-1)
        contact = valid & (distance < 2*self.config.radius)
        normal = delta / distance.clamp_min(1e-8)[..., None]
        fallback = torch.zeros_like(normal)
        fallback[..., 0] = 1.
        normal = torch.where((distance < 1e-8)[..., None], fallback, normal)
        correction = torch.where(contact,
            (2*self.config.radius-distance).mul(.10).clamp_max(.005),
            torch.zeros_like(distance))
        displacement = correction[..., None] * normal
        world = torch.arange(self.n, device=self.device)[:, None]
        flat_delta = torch.zeros((self.n*self.p, 2), device=self.device,
                                 dtype=self.pos.dtype)
        flat_delta.index_add_(0, (world*self.p+i[None]).reshape(-1),
                              displacement.reshape(-1, 2))
        flat_delta.index_add_(0, (world*self.p+j[None]).reshape(-1),
                              -displacement.reshape(-1, 2))
        flat_delta = flat_delta.reshape(self.n, self.p, 2)
        magnitude = flat_delta.norm(dim=-1, keepdim=True)
        flat_delta *= .015/magnitude.clamp_min(.015)
        self.pos[..., :2].add_(flat_delta)
        self.piece_pos.add_(flat_delta)
        self._last_xy.copy_(self.piece_pos)
        affected = flat_delta.norm(dim=-1) > 0
        self.sleeping &= ~affected
        self.sleep_clock.masked_fill_(affected, 0.)

        delta_to_all = self.pos[:, :, None, :2] - self.pos[:, None, :, :2]
        distance_to_all = delta_to_all.norm(dim=-1)
        not_self = ~torch.eye(self.p, device=self.device, dtype=torch.bool)
        touching = (free[:, None, :] & not_self[None] &
                    (distance_to_all < 2*self.config.radius)).any(-1)
        soften = selected & touching
        self.vel[..., :2].copy_(torch.where(
            soften[..., None], self.vel[..., :2]*.2, self.vel[..., :2]))
        self.piece_vel.copy_(torch.where(soften[..., None],
                                         self.vel[..., :2], self.piece_vel))
        self._last_vxy.copy_(torch.where(soften[..., None],
                                         self.piece_vel, self._last_vxy))

    def refresh_stats(self):
        return self._hip.refresh_stats() if self._hip is not None else self.stats

    def _sync_external(self):
        moved = (self.piece_pos != self._last_xy).any(-1)
        changed_velocity = (self.piece_vel != self._last_vxy).any(-1)
        changed = moved | changed_velocity
        self.pos[..., :2] = torch.where(moved[..., None], self.piece_pos, self.pos[..., :2])
        self.pos[..., 2] = torch.where(moved, self._spawn_height(self.piece_pos), self.pos[..., 2])
        self.vel[..., :2] = torch.where(changed_velocity[..., None], self.piece_vel, self.vel[..., :2])
        self.vel[..., 2] = torch.where(moved, 0., self.vel[..., 2])
        self.angular.masked_fill_(moved[..., None], 0.)
        self.sleeping &= ~changed
        self.sleep_clock.masked_fill_(changed, 0.)
        return changed

    def _build_pairs(self, free):
        c = self.config
        rebuild = not self._valid_cache or not c.cache
        if not rebuild:
            rebuild = bool((free != self._free).any()) or bool(((self.pos-self._reference).square().sum(-1) > (.5*c.skin)**2).any())
        if not rebuild:
            return
        self.stats["rebuilds"] += 1
        # The reference deliberately uses host hashing; production uses device
        # linked lists. This avoids disguising all-pairs work as a spatial grid.
        rows = []
        xyz, enabled = self.pos.detach().cpu().tolist(), free.detach().cpu().tolist()
        cutoff = 2*c.radius+c.skin
        overflow = False
        for w in range(self.n):
            if c.broadphase == "all_pairs":
                rows.extend((w,i,j) for i in range(self.p) if enabled[w][i]
                            for j in range(i+1,self.p) if enabled[w][j])
                continue
            grid = {}
            for i, point in enumerate(xyz[w]):
                if enabled[w][i]:
                    key = tuple(math.floor(a/cutoff) for a in point)
                    grid.setdefault(key, []).append(i)
            for key, objects in grid.items():
                for i in objects:
                    count = 0
                    for dx in (-1,0,1):
                        for dy in (-1,0,1):
                            for dz in (-1,0,1):
                                for j in grid.get((key[0]+dx,key[1]+dy,key[2]+dz), ()):
                                    if j <= i:
                                        continue
                                    if sum((xyz[w][i][a]-xyz[w][j][a])**2 for a in range(3)) <= cutoff**2:
                                        rows.append((w,i,j)); count += 1
                    overflow |= count > c.max_neighbors
        # Dynamic reference lists do not drop contacts even above capacity.
        self.stats["overflows"] += int(overflow)
        self._pairs = torch.tensor(rows, device=self.device, dtype=torch.long).reshape(-1,3)
        self.stats["candidate_pairs"] = len(rows)
        self._reference.copy_(self.pos)
        self._free.copy_(free)
        self._valid_cache = True

    def _impulse(self, normal, depth, relative, inv_mass, dt,
                 tangential_inverse_mass=None, damping=None):
        c = self.config
        damping = c.damping if damping is None else damping
        vn = (relative*normal).sum(-1)
        # Backward-Euler spring/damper avoids explicit stiff-force instability.
        jn = (dt*(c.stiffness*depth-damping*vn)/
              (1+dt*damping*inv_mass+dt*dt*c.stiffness*inv_mass)).clamp_min(0)
        # Foam has finite compressibility: prevent center crossing after the
        # calibrated compliant range, dissipating excessive impact energy.
        jn = torch.where(depth>=2*c.radius*c.max_compression_fraction, jn.maximum((-vn/inv_mass).clamp_min(0)), jn)
        tangent = relative-vn[...,None]*normal
        speed = tangent.norm(dim=-1)
        # A sphere's tangential effective mass includes its rotational inertia.
        tangent_mass = inv_mass+2.5/c.mass if tangential_inverse_mass is None else tangential_inverse_mass
        jt = (speed/tangent_mass).minimum(c.friction*jn)
        return jn[...,None]*normal-jt[...,None]*tangent/speed.clamp_min(1e-12)[...,None]

    def _pair_contacts(self, pairs, dt):
        if not pairs.numel():
            return
        c = self.config
        w,i,j = pairs.unbind(-1)
        delta = self.pos[w,i]-self.pos[w,j]
        distance = delta.norm(dim=-1)
        valid = distance < 2*c.radius
        if c.contact_mode == "compact":
            w,i,j,delta,distance = (a[valid] for a in (w,i,j,delta,distance))
            valid = torch.ones_like(distance,dtype=torch.bool)
        normal = delta/distance.clamp_min(1e-12)[:,None]
        normal = torch.where((distance<1e-12)[:,None], torch.tensor([1.,0.,0.],device=self.device),normal)
        awake = ~self.sleeping[w,i] | ~self.sleeping[w,j]
        valid &= awake
        # A shared midpoint contact preserves angular momentum even while
        # compliant foam spheres overlap. Surface arms would add net torque.
        ri = -.5*distance[:,None]*normal
        rj = .5*distance[:,None]*normal
        relative = self.vel[w,i]+torch.linalg.cross(self.angular[w,i],ri)-self.vel[w,j]-torch.linalg.cross(self.angular[w,j],rj)
        inertia = .4*c.mass*c.radius**2
        tangent_inv = 2/c.mass+2*(.5*distance).square()/inertia
        impulse = self._impulse(normal,2*c.radius-distance,relative,2/c.mass,dt,
                                tangent_inv,damping=160.)*valid[:,None]
        flat_i,flat_j = w*self.p+i,w*self.p+j
        dv = torch.zeros_like(self.vel).reshape(-1,3)
        dw = torch.zeros_like(dv)
        dv.index_add_(0,flat_i,impulse/c.mass); dv.index_add_(0,flat_j,-impulse/c.mass)
        inertia = .4*c.mass*c.radius**2
        dw.index_add_(0,flat_i,torch.linalg.cross(ri,impulse)/inertia)
        dw.index_add_(0,flat_j,torch.linalg.cross(rj,-impulse)/inertia)
        dv = dv.reshape_as(self.vel)
        # Preserve pairwise momentum and angular momentum while limiting the
        # aggregate Jacobi response in a crowded pile. One shared scale per
        # world retains equal-and-opposite contact impulses.
        world_delta = dv.norm(dim=-1).amax(-1, keepdim=True)
        degree = torch.zeros((self.n*self.p,), device=self.device,
                             dtype=torch.int32)
        contact_i, contact_j = flat_i[valid], flat_j[valid]
        degree.index_add_(0, contact_i, torch.ones_like(contact_i, dtype=torch.int32))
        degree.index_add_(0, contact_j, torch.ones_like(contact_j, dtype=torch.int32))
        max_degree = degree.reshape(self.n, self.p).amax(-1, keepdim=True)
        crowded = max_degree > 4
        scale = torch.where(crowded,
                            (.08/world_delta.clamp_min(.08)).clamp_max(1.),
                            torch.ones_like(world_delta))
        self.vel.add_(dv*scale[..., None]); self.angular.add_(dw.reshape_as(self.angular)*scale[..., None])
        # Wake on active neighbors even if the current separating impulse is zero.
        self.sleeping[w[valid],i[valid]] = False
        self.sleeping[w[valid],j[valid]] = False

    def _static_contact(self, normal, depth, free, dt):
        c = self.config
        valid = free & (depth > 0) & ~self.sleeping
        arm = -c.radius*normal
        relative = self.vel+torch.linalg.cross(self.angular,arm)
        impulse = self._impulse(normal,depth,relative,1/c.mass,dt)*valid[...,None]
        self.vel.add_(impulse/c.mass)
        self.angular.add_(torch.linalg.cross(arm,impulse)/(.4*c.mass*c.radius**2))
        if c.rolling_friction:
            magnitude = self.angular.norm(dim=-1)
            loss = c.rolling_friction*impulse.norm(dim=-1)*c.radius/(.4*c.mass*c.radius**2)
            self.angular.mul_((1-loss/magnitude.clamp_min(1e-12)).clamp_min(0)[...,None])

    @staticmethod
    def _box_normal(local, half):
        closest = local.maximum(-half).minimum(half)
        delta = local-closest
        distance = delta.norm(dim=-1)
        gap = half-local.abs()
        axis = gap.argmin(-1)
        inside_normal = torch.zeros_like(local)
        inside_normal.scatter_(-1,axis[...,None],torch.where(local.gather(-1,axis[...,None])>=0,1.,-1.))
        normal = torch.where((distance>1e-12)[...,None],delta/distance.clamp_min(1e-12)[...,None],inside_normal)
        signed = torch.where(distance>1e-12,distance,-gap.min(-1).values)
        return normal,signed

    def _robot_contacts(self, free, dt):
        c,s = self.config,self.sim
        robot_delta = torch.zeros_like(s.velocity)
        for robot in range(s.pose.shape[1]):
            theta = s.pose[:,robot,2]
            co,si = theta.cos()[:,None],theta.sin()[:,None]
            delta = self.pos[...,:2]-s.pose[:,robot,None,:2]
            local = torch.stack((co*delta[...,0]+si*delta[...,1],-si*delta[...,0]+co*delta[...,1]),-1)
            half = torch.stack((s.length[:,robot],s.width[:,robot]),-1)[:,None,:]*.5
            normal2,distance = self._box_normal(local,half)
            normal = torch.zeros_like(self.pos)
            normal[...,0]=co*normal2[...,0]-si*normal2[...,1]
            normal[...,1]=si*normal2[...,0]+co*normal2[...,1]
            valid = free & (distance<c.radius) & (self.pos[...,2]<c.robot_height+c.radius)
            # Chassis side contacts transfer momentum and yaw torque.
            arm = self.pos[...,:2]-c.radius*normal[...,:2]-s.pose[:,robot,None,:2]
            rv = s.velocity[:,robot,None,:2]+s.velocity[:,robot,None,2:3]*torch.stack((-arm[...,1],arm[...,0]),-1)
            relative = self.vel+torch.linalg.cross(self.angular,-c.radius*normal)
            relative[...,:2] -= rv
            inertia=s.mass[:,robot]*(s.length[:,robot]**2+s.width[:,robot]**2)/12*s.yaw_inertia_multiplier[:,robot]
            cross=arm[...,0]*normal[...,1]-arm[...,1]*normal[...,0]
            inv=1/c.mass+1/s.mass[:,robot,None]+cross**2/inertia[:,None]
            tangent = relative-(relative*normal).sum(-1)[...,None]*normal
            tangent_unit=tangent/tangent.norm(dim=-1).clamp_min(1e-12)[...,None]
            rt=arm[...,0]*tangent_unit[...,1]-arm[...,1]*tangent_unit[...,0]
            inv_t=3.5/c.mass+1/s.mass[:,robot,None]+rt**2/inertia[:,None]
            impulse=self._impulse(normal,c.radius-distance,relative,inv,dt,inv_t)*valid[...,None]
            self.vel.add_(impulse/c.mass)
            self.angular.add_(torch.linalg.cross(-c.radius*normal,impulse)/(.4*c.mass*c.radius**2))
            robot_delta[:,robot,:2] -= impulse[...,:2].sum(1)/s.mass[:,robot,None]
            robot_delta[:,robot,2] -= (arm[...,0]*impulse[...,1]-arm[...,1]*impulse[...,0]).sum(1)/inertia
            self.sleeping &= ~valid
        s.velocity.add_(robot_delta)

    def _torch_substep(self, free, dt):
        moving = (self.vel.norm(dim=-1)>=self.config.sleep_speed) | (self.angular.norm(dim=-1)>=self.config.sleep_angular_speed)
        self.sleeping &= ~(moving & free)
        awake = free & ~self.sleeping
        self.vel[...,2] -= awake*self.config.gravity*dt
        self.pos.add_(self.vel*awake[...,None]*dt)
        self._build_pairs(free)
        if self.config.solver == "colored":
            pairs = self._pairs
            if pairs.numel():
                w,i,j = pairs.unbind(-1)
                pairs = pairs[(self.pos[w,i]-self.pos[w,j]).norm(dim=-1)<2*self.config.radius]
            rows = sorted(pairs.detach().cpu().tolist())
            groups,used = [],{}
            for row in rows:
                w,i,j=row
                occupied=used.get((w,i),set())|used.get((w,j),set())
                color=0
                while color in occupied: color+=1
                while len(groups)<=color: groups.append([])
                groups[color].append(row)
                used.setdefault((w,i),set()).add(color);used.setdefault((w,j),set()).add(color)
            for group in groups:
                self._pair_contacts(torch.tensor(group,device=self.device,dtype=torch.long),dt)
        else:
            self._pair_contacts(self._pairs,dt)
        self._robot_contacts(free,dt)
        normal = torch.zeros_like(self.pos); normal[...,2]=1
        height = torch.zeros_like(self.pos[...,2])
        from .field import BUMP_PANEL_THICKNESS, BUMP_RAMP_RISE
        for bump in getattr(self.sim,"bump_regions",torch.empty((0,4),device=self.device)):
            dx=self.pos[...,0]-bump[0]
            within=(dx.abs()<=bump[2]) & ((self.pos[...,1]-bump[1]).abs()<=bump[3])
            terrain=BUMP_PANEL_THICKNESS+BUMP_RAMP_RISE*(1-dx.abs()/bump[2])
            take=within & (terrain>height)
            slope=-dx.sign()*BUMP_RAMP_RISE/bump[2]
            ramp_normal=torch.stack((-slope,torch.zeros_like(slope),torch.ones_like(slope)),-1)
            ramp_normal=ramp_normal/ramp_normal.norm(dim=-1)[...,None]
            normal=torch.where(take[...,None],ramp_normal,normal)
            height=torch.where(take,terrain,height)
        self._support_height=height
        self._static_contact(normal,self.config.radius-(self.pos[...,2]-height)*normal[...,2],free,dt)
        for axis,limit in ((0,self.sim.field_length),(1,self.sim.field_width)):
            normal.zero_(); normal[...,axis]=1
            self._static_contact(normal,self.config.radius-self.pos[...,axis],free,dt)
            normal[...,axis]=-1
            self._static_contact(normal,self.pos[...,axis]+self.config.radius-limit,free,dt)
        for box in getattr(self.sim,"field_colliders",torch.empty((0,4),device=self.device)):
            normal2,distance=self._box_normal(self.pos[...,:2]-box[:2],box[2:])
            normal.zero_(); normal[...,:2]=normal2
            self._static_contact(normal,self.config.radius-distance,free & (self.pos[...,2]<self.config.robot_height+self.config.radius),dt)
        if self.config.sleep and self.config.sleep_islands:
            from .fuel_physics_sleep import update_island_sleep
            update_island_sleep(self,free,dt)
        else:
            self._sleep_update(free,dt)

    def _sleep_update(self, free, dt):
        c=self.config
        if not c.sleep:
            self.sleeping.masked_fill_(free,False); return
        if c.sleep_islands:
            return
        support_velocity = self.vel.clone()
        support_height=getattr(self,"_support_height",0.)
        on_floor = self.pos[...,2] < support_height+c.radius+.005
        support_velocity[...,2] -= (on_floor & ~self.sleeping)*c.gravity*dt
        quiet=free & (support_velocity.norm(dim=-1)<c.sleep_speed) & (self.angular.norm(dim=-1)<c.sleep_angular_speed) & (self.pos[...,2]<support_height+c.radius+.005)
        self.sleep_clock.copy_(torch.where(free,torch.where(quiet,self.sleep_clock+dt,0.),self.sleep_clock))
        self.sleeping.copy_(torch.where(free,quiet & (self.sleep_clock>=c.sleep_time),self.sleeping))
        self.vel.masked_fill_((self.sleeping & free)[...,None],0.)
        self.angular.masked_fill_((self.sleeping & free)[...,None],0.)

    def required_substeps(self, dt=None, base=None, active_mask=None, piece_active=None, piece_owner=None):
        """One host reduction per control tick; callers can reuse the result."""
        duration=float(self.sim.dt if dt is None else dt)
        count=self.config.substeps if base is None else int(base)
        if not self.config.adaptive_substeps:
            return count
        fuel_speed=torch.maximum(self.vel.norm(dim=-1),self.piece_vel.norm(dim=-1))
        robot_speed=self.sim.velocity[...,:2].norm(dim=-1)+self.sim.velocity[...,2].abs()*.5*(self.sim.length.square()+self.sim.width.square()).sqrt()
        if piece_active is not None: fuel_speed=fuel_speed*piece_active.bool()
        if piece_owner is not None: fuel_speed=fuel_speed*(piece_owner<0)
        if active_mask is not None:
            fuel_speed=fuel_speed*active_mask[:,None]
            robot_speed=robot_speed*active_mask[:,None]
        speed=torch.maximum(fuel_speed.amax(),robot_speed.amax())
        duration=float(self.sim.dt if dt is None else dt)
        max_motion=self.config.max_translation_per_substep or self.config.radius*.5
        bound=2*float(speed.item())+self.config.gravity*duration
        return max(count,math.ceil(duration*bound/max_motion))

    def step(self, active_mask, piece_active, piece_owner, *, dt=None, substeps=None, adaptive=None):
        active=torch.as_tensor(active_mask,device=self.device,dtype=torch.bool).reshape(self.n)
        free=active[:,None] & piece_active.bool() & (piece_owner<0)
        changed=self._sync_external()
        count=self.config.substeps if substeps is None else int(substeps)
        duration=float(self.sim.dt if dt is None else dt)
        if count<1 or duration<=0:
            raise ValueError("positive timestep and substeps required")
        if self.config.adaptive_substeps if adaptive is None else adaptive:
            count=self.required_substeps(duration,count,active,piece_active,piece_owner)
        h=duration/count
        self._last_substep_dt=h
        if self._hip is not None:
            self._hip.step(free,changed,h,count)
            self.stats["backend"]="hip"
        else:
            for _ in range(count):
                self._torch_substep(free,h)
        self.piece_pos.copy_(self.pos[...,:2])
        self.piece_vel.copy_(self.vel[...,:2])
        self._last_xy.copy_(self.piece_pos); self._last_vxy.copy_(self.piece_vel)
        return self.pos,self.vel
