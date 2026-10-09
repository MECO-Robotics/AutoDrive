"""Couple fuel contact substeps to the existing drivetrain and game rules."""
from __future__ import annotations

import math
import torch


def initialize_fuel_physics(env, config=True):
    """Keep legacy planar slots as the game/observation API."""
    if config is False:
        env.fuel_physics = None
        return
    from .fuel_physics import FuelPhysics
    env.fuel_physics = FuelPhysics(env.sim, env.piece_pos, env.piece_vel,
                                   None if config is True else config)


def bind_fuel_state(env):
    physics = env.fuel_physics
    physics.piece_pos = env.piece_pos
    physics.piece_vel = env.piece_vel
    return physics


def reset_fuel_physics(env, mask=None):
    if env.fuel_physics is not None:
        bind_fuel_state(env).reset(mask)


def advance_with_fuel(env, commands, active_mask, *, active_nonempty=True):
    """Advance robot and fuel together; controller cadence stays unchanged."""
    physics = env.fuel_physics
    if physics is None:
        return env.sim.step(commands, active_mask, _active_nonempty=active_nonempty)
    physics = bind_fuel_state(env)
    active = (torch.ones(env.n, device=env.device, dtype=torch.bool)
              if active_mask is None else active_mask)
    outer_dt = env.sim.dt
    substeps = (physics.required_substeps(outer_dt,None,active,
                                         env.piece_active,env.piece_owner)
                if physics.config.adaptive_substeps else physics.config.substeps)
    substep_dt = outer_dt / substeps
    # The motor and tire model consumes collision-modified velocity on the
    # next substep. Restoring dt also preserves reset/playback timing.
    try:
        env.sim.dt = substep_dt
        for _ in range(substeps):
            env.sim.step(commands, active, _active_nonempty=active_nonempty)
            physics.step(active, env.piece_active, env.piece_owner,
                         dt=substep_dt, substeps=1,adaptive=False)
    finally:
        env.sim.dt = outer_dt
    return env.sim.state


def shooter_respawn_targets(env, origins):
    """Sample a 45-degree total cone toward field center, preserving range."""
    midfield = origins.new_tensor([env.sim.field_length*.5, env.sim.field_width*.5])
    toward_midfield = midfield-origins
    bearing = torch.atan2(toward_midfield[..., 1], toward_midfield[..., 0])
    offset = (torch.rand(origins.shape[:-1], device=origins.device,
                         generator=env.generator)-.5)*(math.pi/4)
    angle = bearing+offset
    direction = torch.stack((angle.cos(), angle.sin()), -1)
    distance = (env._midfield_respawn_positions[None]-origins).norm(dim=-1)
    return origins+distance[..., None]*direction


def resolve_ferry_landings(env, origins, targets, team_ids, *,
                           return_backer_hits=False):
    """Clip ferry destinations, routing hits on the exposed hub net to midfield."""
    from .field import INCH
    radius = env.fuel_physics.config.radius if env.fuel_physics is not None else env._fuel_radius
    delta = targets-origins
    lower = origins.new_tensor([radius, radius])
    upper = origins.new_tensor([env.sim.field_length-radius, env.sim.field_width-radius])
    boundary = torch.where(delta>0, upper, lower)
    moving = delta.abs()>1e-8
    exit_fraction = (boundary-origins)/torch.where(moving, delta, 1.)
    exit_fraction = torch.where(moving, exit_fraction, 1.)
    travel = exit_fraction.amin(-1).clamp(0., 1.)
    landings = origins+travel[..., None]*delta
    landings[..., 0].clamp_(radius, env.sim.field_length-radius)
    landings[..., 1].clamp_(radius, env.sim.field_width-radius)
    hub = env.hub_centers[team_ids]
    # Bumps sit immediately above and below the hub. Only the exposed hub edge
    # between their inner edges is backer net; the outer edge is not a net.
    half_bump_width = 73.0 * INCH / 2
    bump_offset = (47.0 * INCH + 73.0 * INCH) / 2
    net_half_width = bump_offset - half_bump_width - radius
    face = hub[..., 0] + torch.where(team_ids == 0, 47.0 * INCH / 2,
                                     -47.0 * INCH / 2)
    dx = landings[..., 0] - origins[..., 0]
    fraction = (face - origins[..., 0]) / torch.where(dx.abs() > 1e-8, dx, 1.)
    crossing_y = origins[..., 1] + fraction * (landings[..., 1] - origins[..., 1])
    hit_net = ((dx.abs() > 1e-8) & (fraction >= 0) & (fraction <= 1) &
               ((crossing_y - hub[..., 1]).abs() <= net_half_width))
    indices = torch.randint(env._midfield_respawn_positions.shape[0],
                            origins.shape[:-1], device=origins.device,
                            generator=env.generator)
    midfield = env._midfield_respawn_positions[indices]
    resolved = torch.where(hit_net[..., None], midfield, landings)
    return (resolved, hit_net) if return_backer_hits else resolved


def ferry_aim_targets(env, origins, team_ids):
    """Choose a friendly-zone aim point whose path clears the HUB backer net."""
    center = env.hub_centers[team_ids].clone()
    center[..., 0] = torch.where(team_ids == 0, env.alliance_zone_depth*.5,
                                 env.sim.field_length-env.alliance_zone_depth*.5)
    # Aim well to one side of the finite net segment. This keeps a ferry pass
    # from deliberately triggering the backer-net return-to-midfield rule.
    radius = (env.fuel_physics.config.radius if env.fuel_physics is not None
              else env._fuel_radius)
    side_offset = min(3.0, env.sim.field_width*.5-radius-.1)
    center[..., 1] += torch.where(
        origins[..., 1] <= env.hub_centers[team_ids][..., 1],
        -side_offset, side_offset)
    center[..., 1].clamp_(radius, env.sim.field_width-radius)
    return center


def ferry_respawn_targets(env, origins, team_ids, *, return_backer_hits=False):
    """Sample a 10-degree total cone aimed safely into the friendly zone."""
    center = ferry_aim_targets(env, origins, team_ids)
    delta = center-origins
    bearing = torch.atan2(delta[..., 1], delta[..., 0])
    offset = (torch.rand(origins.shape[:-1], device=origins.device,
                         generator=env.generator)-.5)*math.radians(10.)
    angle = bearing+offset
    direction = torch.stack((angle.cos(), angle.sin()), -1)
    targets = origins+delta.norm(dim=-1)[..., None]*direction
    return resolve_ferry_landings(env, origins, targets, team_ids,
                                  return_backer_hits=return_backer_hits)


def allocate_clear_ferry_spawns(env, preferred, selected, team_ids, origins):
    """Choose clear, net-avoiding alliance cells and reserve same-tick spawns."""
    radius = (env.fuel_physics.config.radius if env.fuel_physics is not None
              else env._fuel_radius)
    spacing = 2. * radius + .025
    depth = env.alliance_zone_depth
    width = env.sim.field_width
    nx = max(1, int((depth - 2. * radius) / spacing))
    ny = max(1, int((width - 2. * radius) / spacing))
    x = radius + (torch.arange(nx, device=preferred.device, dtype=preferred.dtype)+.5) * ((depth-2.*radius)/nx)
    y = radius + (torch.arange(ny, device=preferred.device, dtype=preferred.dtype)+.5) * ((width-2.*radius)/ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    red = torch.stack((xx.reshape(-1), yy.reshape(-1)), -1)
    blue = red.clone()
    blue[:, 0] = env.sim.field_length-red[:, 0]
    candidates = torch.stack((red, blue), 0)[team_ids]

    # Preserve the finite backer-net rule after redirecting a crowded landing.
    from .field import INCH
    hubs = env.hub_centers[team_ids]
    face = hubs[:, 0] + torch.where(team_ids == 0, 47.*INCH/2, -47.*INCH/2)
    delta = candidates[None]-origins[:,:,None]
    dx = delta[..., 0]
    fraction = (face[None,:,None]-origins[:,:,None,0]) / torch.where(
        dx.abs()>1e-8, dx, 1.)
    crossing_y = origins[:,:,None,1]+fraction*delta[...,1]
    net_half_width = (47.*INCH+73.*INCH)/2-73.*INCH/2-radius
    route_hits_net = ((dx.abs()>1e-8) & (fraction>=0) & (fraction<=1) &
                      ((crossing_y-hubs[None,:,None,1]).abs()<=net_half_width))
    safe_route = ~route_hits_net

    cell_x_count = math.ceil(env.sim.field_length / spacing)
    cell_y_count = math.ceil(width / spacing)
    occupied_counts = torch.zeros((preferred.shape[0], cell_x_count*cell_y_count),
                                  dtype=torch.int32, device=preferred.device)
    free_fuel = env.piece_active & (env.piece_owner < 0)
    piece_x = (env.piece_pos[..., 0] / spacing).floor().long().clamp_(0, cell_x_count-1)
    piece_y = (env.piece_pos[..., 1] / spacing).floor().long().clamp_(0, cell_y_count-1)
    piece_cell = piece_y*cell_x_count+piece_x
    occupied_counts.scatter_add_(1, piece_cell, free_fuel.to(torch.int32))

    candidate_x = (candidates[..., 0] / spacing).floor().long().clamp_(0, cell_x_count-1)
    candidate_y = (candidates[..., 1] / spacing).floor().long().clamp_(0, cell_y_count-1)
    candidate_cell = candidate_y*cell_x_count+candidate_x
    distance = (candidates[None]-preferred[:,:,None]).square().sum(-1)
    chosen = preferred.clone()
    available = torch.ones_like(selected)
    worlds = torch.arange(preferred.shape[0], device=preferred.device)
    for robot in range(team_ids.numel()):
        occupied = occupied_counts.reshape(
            preferred.shape[0],1,cell_y_count,cell_x_count).to(preferred.dtype)
        blocked_grid = torch.nn.functional.max_pool2d(
            occupied, kernel_size=3, stride=1, padding=1).reshape(
                preferred.shape[0],-1).bool()
        blocked = torch.gather(
            blocked_grid, 1, candidate_cell[robot][None].expand(preferred.shape[0],-1))
        valid = safe_route[:,robot] & ~blocked
        has_clear = valid.any(-1)
        cost = distance[:,robot].masked_fill(~valid,float("inf"))
        index = cost.argmin(-1)
        point = candidates[robot,index]
        reserve = selected[:,robot] & has_clear
        chosen[:,robot] = torch.where(reserve[:,None],point,chosen[:,robot])
        available[:,robot] = has_clear
        cell = candidate_cell[robot,index]
        occupied_counts.scatter_add_(1,cell[:,None],reserve.to(torch.int32)[:,None])
    return chosen, available


def launch_fuel(env, selected, destinations, *, origins=None, height=.35,
                flight_time=.8, horizontal_velocity_scale=1.,
                respawn_at_destination=False, spawn_positions=None):
    """Release moving fuel toward a destination with a ballistic arc.

    Destination sampling remains a game-rule choice; contact dynamics may
    change the actual landing point. The horizontal velocity scale controls
    release speed. Ground respawns use the destination as their starting
    position and retain the reduced horizontal velocity. Launch settings are
    illustrative.
    """
    physics = bind_fuel_state(env)
    start = env.piece_pos if origins is None else origins
    target = destinations.expand_as(env.piece_pos)
    z = max(height, physics.config.radius)
    velocity = (target - start) * (horizontal_velocity_scale / flight_time)
    vertical = ((physics.config.radius - z) / flight_time +
                .5 * physics.config.gravity * flight_time)
    position = (target if respawn_at_destination else
                start if spawn_positions is None else spawn_positions)
    if respawn_at_destination:
        z = physics._spawn_height(position)
        vertical = 0.
    env.piece_pos.copy_(torch.where(selected[..., None], position, env.piece_pos))
    env.piece_vel.copy_(torch.where(selected[..., None], velocity, env.piece_vel))
    physics.pos[..., :2].copy_(torch.where(selected[..., None], position,
                                          physics.pos[..., :2]))
    physics.pos[..., 2].copy_(torch.where(selected, z, physics.pos[..., 2]))
    physics.vel[..., :2].copy_(torch.where(selected[..., None], velocity,
                                          physics.vel[..., :2]))
    physics.vel[..., 2].masked_fill_(selected, vertical)
    physics.angular.masked_fill_(selected[..., None], 0.)
    physics.sleeping.masked_fill_(selected,False)
    physics.sleep_clock.masked_fill_(selected,0.)
    physics.commit_external(selected)
    if respawn_at_destination:
        free = env.piece_active & ((env.piece_owner < 0) | selected)
        physics.separate_spawned(selected, free)
