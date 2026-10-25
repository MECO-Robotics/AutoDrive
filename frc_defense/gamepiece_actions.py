"""Shared robot gamepiece actions used by each simulator roster."""
from __future__ import annotations

import math
import torch

from .fuel_physics_runtime import (launch_fuel, shooter_respawn_cone,
                                  ferry_aim_targets, ferry_respawn_targets)
from . import tensor_3v3_pickup_grid_hip as _pickup_grid_hip
from ._tensor_3v3_observation import NUM_ROBOTS


_SCORE_SOURCE_SITE_CODES = {
    "gamepiece_actions.update_fuel_actions.vectorized_score_mutation": 1,
    "gamepiece_actions.update_fuel_actions.robot_loop_score_mutation": 2,
}


def initialize_fuel_score_event_buffer(env, recorder):
    """Allocate fixed device event slots for one bounded score episode."""
    from .meco_simulated_score_ledger import MAX_SCORE_EVENTS_PER_EPISODE

    world_count = int(env.n)
    horizon = int(env.horizon)
    robot_count = int(env.sim.num_robots)
    capacity = world_count * horizon * robot_count
    if (world_count < 1 or horizon < 1 or robot_count < 1 or
            capacity > MAX_SCORE_EVENTS_PER_EPISODE):
        raise ValueError(
            "GPU score-event buffer bound exceeds the fixed aggregate episode cap")
    if recorder.context.get("world_count") != world_count:
        raise ValueError("GPU score-event buffer worlds differ from recorder context")
    device = env.device
    shape = (world_count, horizon, robot_count)
    env._meco_gpu_score_event_buffer = {
        "event_mask": torch.zeros(shape, device=device, dtype=torch.bool),
        "tick": torch.zeros(shape, device=device, dtype=torch.long),
        "match_elapsed_s": torch.zeros(shape, device=device,
                                        dtype=env.match_elapsed.dtype),
        "world_slot": torch.zeros(shape, device=device, dtype=torch.long),
        "alliance": torch.zeros(shape, device=device, dtype=torch.long),
        "robot_slot": torch.zeros(shape, device=device, dtype=torch.long),
        "piece_slot": torch.zeros(shape, device=device, dtype=torch.long),
        "piece_generation": torch.zeros(shape, device=device, dtype=torch.long),
        "source_site_code": torch.zeros(shape, device=device, dtype=torch.int8),
        "world_counts": torch.zeros((world_count,), device=device,
                                    dtype=torch.long),
        "aggregate_count": torch.zeros((), device=device, dtype=torch.long),
        "overflow_flags": torch.zeros((), device=device, dtype=torch.long),
        "capacity": capacity,
        "episode_binding": {
            "episode_id": recorder.context["episode_id"],
            "seed": recorder.context["seed"],
            "scenario_id": recorder.context["scenario_id"],
            "source_identity_sha256": recorder.context[
                "source_identity_sha256"],
            "estimand_approval_sha256": recorder.approval_sha256,
        },
    }
    env._meco_gpu_score_event_buffer_error = None


def reset_fuel_score_event_buffer(env, mask=None):
    """Clear selected world slots while retaining every other world's rows."""
    buffer = getattr(env, "_meco_gpu_score_event_buffer", None)
    if buffer is None:
        return
    if mask is None:
        reset_mask = torch.ones((env.n,), device=env.device, dtype=torch.bool)
    elif (not isinstance(mask, torch.Tensor) or mask.device != env.device or
          mask.dtype is not torch.bool or tuple(mask.shape) != (env.n,)):
        raise ValueError("score-event reset mask must be device bool [world]")
    else:
        reset_mask = mask
    if reset_mask.ndim != 1 or reset_mask.shape[0] != env.n:
        raise ValueError("score-event reset mask must be device bool [world]")
    rows = reset_mask[:, None, None]
    for name in ("event_mask", "tick", "match_elapsed_s", "world_slot",
                 "alliance", "robot_slot", "piece_slot", "piece_generation",
                 "source_site_code"):
        buffer[name].masked_fill_(rows, 0)
    removed = torch.where(reset_mask, buffer["world_counts"],
                           torch.zeros_like(buffer["world_counts"]))
    buffer["world_counts"].copy_(torch.where(
        reset_mask, torch.zeros_like(buffer["world_counts"]),
        buffer["world_counts"]))
    buffer["aggregate_count"].sub_(removed.sum())


def _capture_gpu_fuel_score_awards(env, chosen, *, source_site, robot_slot=None):
    """Write accepted scorer outputs to fixed slots without host materialization."""
    buffer = getattr(env, "_meco_gpu_score_event_buffer", None)
    source_code = _SCORE_SOURCE_SITE_CODES.get(source_site, 0)
    if buffer is None or source_code == 0:
        env._meco_gpu_score_event_buffer_error = (
            "device score-event buffer or supported source site is missing")
        return
    if chosen.ndim == 3:
        world_count, robot_count, _ = chosen.shape
        if (world_count != env.n or
                robot_count != buffer["event_mask"].shape[2]):
            env._meco_gpu_score_event_buffer_error = (
                "vectorized score-event mask shape changed")
            return
        accepted = chosen.any(-1)
        piece_slots = chosen.to(torch.long).argmax(-1)
        generations = env._meco_piece_generation.gather(1, piece_slots)
        for slot in range(robot_count):
            _write_gpu_score_event_slot(
                env, buffer, accepted[:, slot], piece_slots[:, slot],
                generations[:, slot], slot, source_code)
    elif chosen.ndim == 2 and robot_slot is not None:
        world_count = chosen.shape[0]
        if (world_count != env.n or
                not 0 <= int(robot_slot) < buffer["event_mask"].shape[2]):
            env._meco_gpu_score_event_buffer_error = (
                "robot-loop score-event mask shape changed")
            return
        piece_slots = chosen.to(torch.long).argmax(-1)
        generations = env._meco_piece_generation.gather(
            1, piece_slots[:, None]).squeeze(1)
        _write_gpu_score_event_slot(
            env, buffer, chosen.any(-1), piece_slots, generations,
            int(robot_slot), source_code)
    else:
        env._meco_gpu_score_event_buffer_error = (
            "accepted score mask has an unsupported shape")


def _write_gpu_score_event_slot(env, buffer, accepted, piece_slots,
                                generations, robot_slot, source_code):
    from .meco_simulated_score_ledger import MAX_SCORE_EVENTS_PER_EPISODE

    world_count, horizon, robot_count = buffer["event_mask"].shape
    rows = torch.arange(world_count, device=env.device)
    raw_tick = env.steps.to(torch.long) - 1
    tick_valid = (raw_tick >= 0) & (raw_tick < horizon)
    safe_tick = raw_tick.clamp(0, horizon - 1)
    robot_indices = torch.full_like(rows, int(robot_slot))
    index = (rows, safe_tick, robot_indices)
    previous = buffer["event_mask"][index]
    duplicate = accepted & previous
    write = accepted & ~previous & tick_valid
    slot_values = {
        "tick": raw_tick.clamp_min(0),
        "match_elapsed_s": env.match_elapsed,
        "world_slot": rows,
        "alliance": env.team_ids[robot_slot].expand(world_count),
        "robot_slot": robot_indices,
        "piece_slot": piece_slots,
        "piece_generation": generations,
        "source_site_code": torch.full_like(rows, source_code,
                                             dtype=torch.int8),
    }
    for name, values in slot_values.items():
        current = buffer[name][index]
        buffer[name].index_put_(index, torch.where(write, values, current))
    buffer["event_mask"].index_put_(index, previous | write)
    added = write.to(torch.long)
    buffer["world_counts"].add_(added)
    aggregate_count = buffer["aggregate_count"] + added.sum()
    buffer["aggregate_count"].copy_(aggregate_count)
    flags = ((duplicate | (accepted & ~tick_valid)).any().to(torch.long) * 1)
    flags = flags | (aggregate_count > MAX_SCORE_EVENTS_PER_EPISODE).to(torch.long) * 2
    buffer["overflow_flags"].bitwise_or_(flags)


def materialize_fuel_score_awards(env, recorder):
    """Copy bounded event rows at the passive terminal-report boundary."""
    try:
        return _materialize_fuel_score_awards(env, recorder)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _materialize_fuel_score_awards(env, recorder):
    from .meco_simulated_score_ledger import MAX_SCORE_EVENTS_PER_EPISODE

    buffer = getattr(env, "_meco_gpu_score_event_buffer", None)
    if buffer is None:
        return False, "device score-event buffer is missing"
    expected_binding = {
        "episode_id": recorder.context["episode_id"],
        "seed": recorder.context["seed"],
        "scenario_id": recorder.context["scenario_id"],
        "source_identity_sha256": recorder.context["source_identity_sha256"],
        "estimand_approval_sha256": recorder.approval_sha256,
    }
    if buffer["episode_binding"] != expected_binding:
        return False, "device score-event episode binding changed"
    if getattr(env, "_meco_gpu_score_event_buffer_error", None):
        return False, env._meco_gpu_score_event_buffer_error
    overflow_flags = int(buffer["overflow_flags"].detach().cpu().item())
    aggregate_count = int(buffer["aggregate_count"].detach().cpu().item())
    if overflow_flags or aggregate_count > MAX_SCORE_EVENTS_PER_EPISODE:
        return False, f"device score-event buffer overflow flags={overflow_flags}"
    rows = {name: buffer[name].detach().cpu()
            for name in ("event_mask", "tick", "match_elapsed_s", "world_slot",
                         "alliance", "robot_slot", "piece_slot",
                         "piece_generation", "source_site_code")}
    accepted = torch.nonzero(rows["event_mask"], as_tuple=False).tolist()
    source_site_by_code = {code: source for source, code in
                           _SCORE_SOURCE_SITE_CODES.items()}
    def order_key(position):
        world, tick, slot = position
        source_code = int(rows["source_site_code"][world, tick, slot])
        if source_code == 1:
            return (tick, 0, world, slot)
        if source_code == 2:
            return (tick, slot, world, slot)
        return (tick, 0, world, slot)
    accepted.sort(key=order_key)
    if len(accepted) != aggregate_count:
        return False, "device score-event count differs from captured rows"
    for world, tick, slot in accepted:
        source_site = source_site_by_code.get(
            int(rows["source_site_code"][world, tick, slot]))
        if source_site is None:
            return False, "device score-event source code is unsupported"
        recorder.record_award(
            tick=int(rows["tick"][world, tick, slot]),
            match_elapsed_s=float(rows["match_elapsed_s"][world, tick, slot]),
            world_slot=int(rows["world_slot"][world, tick, slot]),
            alliance=int(rows["alliance"][world, tick, slot]),
            robot_slot=int(rows["robot_slot"][world, tick, slot]),
            piece_slot=int(rows["piece_slot"][world, tick, slot]),
            piece_generation=int(rows["piece_generation"][world, tick, slot]),
            source_site=source_site)
    return True, None


def _capture_fuel_score_awards(env, chosen, *, source_site, robot_slot=None):
    """Queue accepted score masks for serialization after gameplay mutation."""
    recorder = getattr(env, "_meco_score_event_recorder", None)
    if recorder is None:
        return
    physics = getattr(env, "fuel_physics", None)
    if physics is not None and getattr(physics, "_gpu_qualifying", False):
        try:
            _capture_gpu_fuel_score_awards(
                env, chosen, source_site=source_site, robot_slot=robot_slot)
        except Exception as exc:
            env._meco_gpu_score_event_buffer_error = (
                f"{type(exc).__name__}: {exc}")
        return
    pending = getattr(env, "_pending_fuel_score_awards", None)
    if pending is None:
        pending = []
        env._pending_fuel_score_awards = pending
    generation = getattr(env, "_meco_piece_generation", None)
    pending.append((chosen.detach().clone(), source_site, robot_slot,
                    None if generation is None else generation.detach().clone()))


def _flush_fuel_score_awards(env):
    """Serialize queued awards after all gameplay and physics updates."""
    physics = getattr(env, "fuel_physics", None)
    recorder = getattr(env, "_meco_score_event_recorder", None)
    if (physics is not None and getattr(physics, "_gpu_qualifying", False)
            and recorder is not None):
        if getattr(env, "_meco_gpu_score_event_buffer", None) is None:
            env._meco_gpu_score_event_buffer_error = (
                "GPU-qualifying score recorder has no device event buffer")
        return
    pending = getattr(env, "_pending_fuel_score_awards", None)
    if not pending:
        return
    env._pending_fuel_score_awards = []
    recorder = getattr(env, "_meco_score_event_recorder", None)
    if recorder is None:
        return
    try:
        for chosen, source_site, robot_slot, generation in pending:
            if chosen.ndim == 3:
                accepted = torch.nonzero(chosen, as_tuple=False).tolist()
                for world, slot, piece in accepted:
                    recorder.record_award(
                        tick=max(0, int(env.steps[world].item()) - 1),
                        match_elapsed_s=float(env.match_elapsed[world].item()),
                        world_slot=world,
                        alliance=int(env.team_ids[slot].item()),
                        robot_slot=slot,
                        piece_slot=piece,
                        piece_generation=(int(generation[world, piece].item())
                                         if generation is not None else 0),
                        source_site=source_site)
            elif chosen.ndim == 2 and robot_slot is not None:
                accepted = torch.nonzero(chosen, as_tuple=False).tolist()
                for world, piece in accepted:
                    recorder.record_award(
                        tick=max(0, int(env.steps[world].item()) - 1),
                        match_elapsed_s=float(env.match_elapsed[world].item()),
                        world_slot=world,
                        alliance=int(env.team_ids[robot_slot].item()),
                        robot_slot=robot_slot,
                        piece_slot=piece,
                        piece_generation=(int(generation[world, piece].item())
                                         if generation is not None else 0),
                        source_site=source_site)
            else:
                raise ValueError("accepted score mask has an unsupported shape")
    except Exception as exc:
        # Event capture is optional evidence. Recording failures cannot abort a
        # completed gameplay/physics transition or affect its RNG/state.
        env._meco_score_event_recorder_error = (
            f"{type(exc).__name__}: {exc}")


def _invalidate_perceived_contact_tracks(env, active):
    """Retire tracks using known chassis contact, without hidden enemy poses.

    The fused all-robot contact invalidator is incompatible with this boundary:
    an unseen opponent passing over a remembered track must not erase it.
    """
    same_team = env.team_ids[:, None] == env.team_ids[None, :]
    delta = env.track_pos[:, :, :, None, :] - env.sim.pose[:, None, None, :, :2]
    heading = env.sim.pose[:, :, 2]
    forward = torch.stack((heading.cos(), heading.sin()), -1)
    left = torch.stack((-heading.sin(), heading.cos()), -1)
    longitudinal = (delta * forward[:, None, None]).sum(-1)
    lateral = (delta * left[:, None, None]).sum(-1).abs()
    radius = env._fuel_radius
    contact = (
        (longitudinal >= -env.sim.length[:, None, None] * .5 - radius)
        & (longitudinal <= env.sim.length[:, None, None] * .5 + .35 + radius)
        & (lateral <= env.sim.width[:, None, None] * .5 + .075 + radius)
        & same_team[None, :, None, :]
    ).any(-1)

    if hasattr(env, "opponent_pose"):
        pose = env.opponent_pose
        size = env.opponent_size[:, :, :2]
        observed_valid = env.opponent_valid
    else:
        # The two-robot roster has pose tracks without a size measurement.
        # Use the observer's calibrated footprint as a prior, not enemy truth.
        pose = env._opponent_track_pose
        size = torch.stack((env.sim.length, env.sim.width), -1)
        observed_valid = env._opponent_track_valid
    valid = (observed_valid & torch.isfinite(pose).all(-1)
             & torch.isfinite(size).all(-1) & (size > 0.).all(-1))
    observed_delta = env.track_pos - pose[:, :, None, :2]
    observed_forward = torch.stack((pose[:, :, 2].cos(), pose[:, :, 2].sin()), -1)
    observed_left = torch.stack((-pose[:, :, 2].sin(), pose[:, :, 2].cos()), -1)
    observed_longitudinal = (observed_delta * observed_forward[:, :, None]).sum(-1)
    observed_lateral = (observed_delta * observed_left[:, :, None]).sum(-1).abs()
    observed_contact = (
        (observed_longitudinal >= -size[:, :, None, 0] * .5 - radius)
        & (observed_longitudinal <= size[:, :, None, 0] * .5 + .35 + radius)
        & (observed_lateral <= size[:, :, None, 1] * .5 + .075 + radius)
        & valid[:, :, None]
    )
    intersected = (contact | observed_contact) & active[:, None, None]
    env.track_mask.masked_fill_(intersected, False)
    env.track_age.masked_fill_(intersected, float("inf"))


def update_fuel_actions(self, active, score_intent):
    self.fuel_acquired_event[active]=0; self.fuel_scored_event[active]=0
    self.fuel_passed_event[active]=0; self.fuel_denied_event[active]=0
    if hasattr(self,"fuel_abandoned_event"):
        self.fuel_abandoned_event[active]=0
    curriculum_metrics = getattr(self, "curriculum_metrics", None)
    if curriculum_metrics is not None:
        acquisition_owners = self.piece_owner.clone()
        acquisition_positions = self.piece_pos.clone()
    original_piece_team=self.team_ids[self.piece_owner.clamp(
        0,self.sim.num_robots-1)]
    free = self.piece_active & (self.piece_owner < 0)
    # Pickup bins must follow physical motion, rather than initial and
    # planned landing cells. Airborne fuel cannot enter a floor intake.
    intake_active=self.piece_active
    if self.fuel_physics is not None:
        x=(self.piece_pos[...,0]/self._pickup_grid_cell_size).floor().long().clamp_(0,self._pickup_grid_nx-1)
        y=(self.piece_pos[...,1]/self._pickup_grid_cell_size).floor().long().clamp_(0,self._pickup_grid_ny-1)
        cell=(y*self._pickup_grid_nx+x).to(self._pickup_possible_cells.dtype)
        self._pickup_possible_cells.copy_(torch.where(
            active[:,None,None],cell[...,None].expand_as(self._pickup_possible_cells),
            self._pickup_possible_cells))
        intake_active=self.piece_active & (
            (self.piece_owner>=0) | (self.fuel_physics.pos[...,2]<=.20))
        free &= intake_active
    # Turret intake stays deployed. Dumper intake follows its action and
    # only reaches FUEL while the robot is actively collecting.
    intake_extended=(self._turret_mask[None] | self._intake_collecting)
    # Track invalidation is based on the intake contact region, not a
    # presumed one-to-one mapping between observed tracks and fuel IDs.
    track_clear_mask = torch.zeros_like(free)
    # Process any configured roster through one shared gamepiece path.
    robot_count=self.sim.num_robots
    all_robots_controlled=(robot_count==NUM_ROBOTS and
                            all(mode!="none" for mode in self.control_modes))
    pickup_grid_available=(self.fuel_count<=512 and
                           _pickup_grid_hip.integrated_available(self.device))
    # The integrated HIP pickup kernel below owns intake geometry for the
    # standard six-robot roster. Avoid building the equivalent [world,robot,
    # piece] tensors unless the fallback loop will consume them.
    if all_robots_controlled and not pickup_grid_available:
        poses=self.sim.pose
        all_delta=self.piece_pos[:,None]-poses[:,:,:2][:,:,None,:]
        forward=torch.stack((poses[...,2].cos(),poses[...,2].sin()),-1)
        left=torch.stack((-poses[...,2].sin(),poses[...,2].cos()),-1)
        longitudinal=(all_delta*forward[:,:,None]).sum(-1)
        lateral=(all_delta*left[:,:,None]).sum(-1).abs()
        all_intake=(
            (longitudinal>=self.sim.length[:,:,None]*.5-.075-self._fuel_radius)&
            (longitudinal<=self.sim.length[:,:,None]*.5+.35+
             intake_extended[:,:,None]*.3048+self._fuel_radius)&
            (lateral<=self.sim.width[:,:,None]*.5+.075+self._fuel_radius))
        all_distance_squared=all_delta.square().sum(-1)
    pickup_all_fused=(robot_count==NUM_ROBOTS and pickup_grid_available and
                      all_robots_controlled)
    if pickup_all_fused:
        _pickup_grid_hip.pickup_all_robots(
            active=active, pose=self.sim.pose, length=self.sim.length,
            width=self.sim.width, piece_pos=self.piece_pos,
            piece_active=intake_active, intake_extended=intake_extended,
            piece_owner=self.piece_owner, free=free,
            next_intake=self.next_intake, elapsed=self.match_elapsed,
            controlled=self._controlled_mode_mask,
            deterministic=self._deterministic_mode_mask,
            defense_role=self._defense_role_mask, hub_active=self.hub_active,
            capacities=self.robot_fuel_capacities,
            acquired_event=self.fuel_acquired_event,
            track_clear_mask=track_clear_mask,
            alliance_depth=self.alliance_zone_depth,
            field_length=self.field_length)
    for robot in range(robot_count):
        if pickup_all_fused:
            break
        if self.control_modes[robot] == "none":
            continue
        extended=intake_extended[:,robot]
        deterministic_offense = (self.control_modes[robot] == "deterministic" and
                                 self.robot_roles[robot] == "offense")
        policy_blocked = torch.zeros_like(free)
        if deterministic_offense:
            team = self.sim._team_ids_host[robot]
            piece_x=self.piece_pos[...,0]
            piece_in_alliance=(piece_x<=self.alliance_zone_depth if team==0 else
                piece_x>=self.field_length-self.alliance_zone_depth)
            # During an active HUB shift the sweep can collect FUEL from
            # midfield and the rest of the field. During a closed shift,
            # gather outside the alliance zone for ferrying.
            allowed_zone = torch.where(
                self.hub_active[:, team, None], torch.ones_like(piece_in_alliance),
                ~piece_in_alliance)
            policy_blocked = free & ~allowed_zone
            free &= allowed_zone
        if robot_count==NUM_ROBOTS and pickup_grid_available:
            _pickup_grid_hip.pickup_robot(
                active=active & extended, pose=self.sim.pose, length=self.sim.length,
                width=self.sim.width, piece_pos=self.piece_pos,
                piece_active=intake_active, piece_owner=self.piece_owner,
                free=free,
                next_intake=self.next_intake, elapsed=self.match_elapsed,
                controlled=self._controlled_mode_mask,
                deterministic=self._deterministic_mode_mask,
                defense_role=self._defense_role_mask,
                acquired_event=self.fuel_acquired_event,
                track_clear_mask=track_clear_mask, robot=robot,
                capacity=self.robot_fuel_capacity_values[robot],
                allow_sweep=deterministic_offense)
            free |= policy_blocked
            continue
        if all_robots_controlled:
            delta=all_delta[:,robot]
            intake=all_intake[:,robot]&extended[:,None]
        else:
            pose = self.sim.pose[:,robot]
            delta = self.piece_pos-pose[:,None,:2]
            forward=torch.stack((pose[:,2].cos(),pose[:,2].sin()),-1)
            left=torch.stack((-pose[:,2].sin(),pose[:,2].cos()),-1)
            longitudinal=(delta*forward[:,None]).sum(-1)
            lateral=(delta*left[:,None]).sum(-1).abs()
            intake=(longitudinal>=self.sim.length[:,robot,None]*.5-.075-self._fuel_radius)&(longitudinal<=self.sim.length[:,robot,None]*.5+.35+extended[:,None]*.3048+self._fuel_radius)&(lateral<=self.sim.width[:,robot,None]*.5+.075+self._fuel_radius)&extended[:,None]
        possession=(self.piece_owner==robot).sum(-1)
        robot_controlled=(self._nn_mode_mask[robot] |
                          self._deterministic_mode_mask[robot])
        sweep_enabled = torch.zeros((),device=self.device,dtype=torch.bool)
        sweep_picked = torch.zeros_like(free)
        if deterministic_offense:
            # Deterministic offense sweeps loose FUEL with its front
            # intake. Capture every contacted piece on this
            # physics tick instead of one piece per 200 ms intake cycle.
            sweep_enabled=(self._deterministic_mode_mask[robot] &
                           ~self._defense_role_mask[robot])
            sweep_candidates=(free&intake&active[:,None]&
                              sweep_enabled&robot_controlled)
            room=(self.robot_fuel_capacities[robot]-possession).clamp_min(0)
            contact_rank=sweep_candidates.to(torch.int32).cumsum(-1)
            sweep_picked=sweep_candidates&(contact_rank<=room[:,None])
        eligible=(free&intake&(possession<self.robot_fuel_capacities[robot])[:,None]&
                 (self.match_elapsed[:,None]+1e-6>=self.next_intake[:,robot,None])&
                 active[:,None]&robot_controlled&~sweep_enabled)
        distance=(all_distance_squared[:,robot] if all_robots_controlled else
                  delta.square().sum(-1)).masked_fill(~eligible,float("inf"))
        nearest,index=distance.min(-1); picked=torch.isfinite(nearest)
        owner=self.piece_owner[self._world_indices,index]
        self.piece_owner[self._world_indices,index]=torch.where(picked,robot,owner)
        torch.where(sweep_picked,self._robot_owner_ids[robot],self.piece_owner,
                    out=self.piece_owner)
        picked_piece=sweep_picked.clone()
        picked_piece.scatter_(1,index[:,None],picked[:,None])
        track_clear_mask |= picked_piece
        self.fuel_acquired_event[:,robot]=torch.where(active,
                                                       picked.long()+sweep_picked.sum(-1),
                                                       self.fuel_acquired_event[:,robot])
        free[self._world_indices,index] &= ~picked
        free &= ~sweep_picked
        self.next_intake[:,robot]=torch.where(picked,self.match_elapsed+.20,self.next_intake[:,robot])
        free |= policy_blocked

    if curriculum_metrics is not None:
        acquired = (acquisition_owners < 0) & (self.piece_owner >= 0) & active[:, None]
        curriculum_metrics.acquired(acquired, self.piece_owner, acquisition_positions)

    # Piece positions are world-major while ownership is per-piece. Gather
    # the owning robot's pose once, then apply pass-zone overrides in one
    # batched write before scoring is evaluated.
    owner_before_pass = self.piece_owner.clone() if curriculum_metrics is not None else self.piece_owner
    safe_owner = owner_before_pass.clamp(0, robot_count - 1)
    held = (self.piece_active & (owner_before_pass >= 0) &
            (owner_before_pass < robot_count) & active[:,None])
    carried_pose = self.sim.pose[self._world_indices[:,None], safe_owner, :2]
    torch.where(held[...,None], carried_pose, self.piece_pos,
                out=self.piece_pos)

    ferry_targets=(self._ferry_targets_for_current_motion()
                   if hasattr(self,"_ferry_targets_for_current_motion") else
                   self.ferry_targets[None].expand(self.n,-1,-1))
    at_ferry = ((self.sim.pose[:,:,:2] - ferry_targets).norm(dim=-1) <= .35)
    home_zone_center=ferry_aim_targets(
        self,self.sim.pose[:,:,:2],self.team_ids[None].expand(self.n,-1))
    ferry_vector=home_zone_center-self.sim.pose[:,:,:2]
    ferry_bearing=torch.atan2(ferry_vector[...,1],ferry_vector[...,0])
    ferry_heading_error=torch.atan2(
        torch.sin(ferry_bearing-self.sim.pose[:,:,2]),
        torch.cos(ferry_bearing-self.sim.pose[:,:,2])).abs()
    # The Dumper ejects only through its front. A ferry release therefore
    # waits until the robot is turned toward its friendly alliance zone.
    facing_home=((ferry_heading_error<=math.pi/4)|self._turret_mask[None])
    rate_ready=(self.match_elapsed[:,None]+1e-6>=self.next_ferry)
    at_release_position=at_ferry|self._dumper_mask[None]
    robot_pass = ((self.last_actions == 6) & at_release_position & facing_home &
                  rate_ready & active[:,None])
    first_held_piece=(owner_before_pass[:,None,:]==
        self._robot_owner_ids[None,:,None]).to(torch.int64).argmax(-1)
    selected_for_pass=(self._fuel_indices==first_held_piece.gather(
        1,safe_owner))
    passed = held & selected_for_pass & robot_pass.gather(1, safe_owner)
    pass_destinations,backer_hit=ferry_respawn_targets(
        self,self.sim.pose[:,:,:2],self.team_ids[None].expand(self.n,-1),
        return_backer_hits=True)
    hit_for_piece=backer_hit.gather(1,safe_owner)
    ordinary_pass=passed & ~hit_for_piece
    backer_pass=passed & hit_for_piece
    passed=ordinary_pass|backer_pass
    pass_positions=pass_destinations[self._world_indices[:,None],safe_owner]
    if self.fuel_physics is None:
        torch.where(passed[...,None], pass_positions, self.piece_pos,
                    out=self.piece_pos)
    else:
        pass_teams=self.team_ids[safe_owner]
        hub_origins=self.hub_centers[pass_teams]
        ferry_origins=self.sim.pose[:,:,:2][self._world_indices[:,None],safe_owner]
        launch_fuel(self,ordinary_pass,pass_positions,origins=ferry_origins,
                    flight_time=.5,horizontal_velocity_scale=.1,
                    respawn_at_destination=True,spawn_positions=pass_positions)
        # A HUB backer return follows the scored-ball ballistic launch
        # settings, starting from its resolved midfield respawn point.
        _,shot_targets=shooter_respawn_cone(self,hub_origins)
        shot_vectors=shot_targets-hub_origins
        shot_distances=shot_vectors.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        shot_vectors=shot_vectors/shot_distances
        backer_origins=pass_positions
        backer_targets=backer_origins+shot_vectors*shot_distances
        launch_fuel(self,backer_pass,backer_targets,origins=backer_origins,
                    height=1.83,flight_time=1.,horizontal_velocity_scale=.1,
                    spawn_positions=backer_origins)
    pass_cell_size=self._pickup_grid_cell_size
    pass_cell_x=(pass_positions[...,0]/pass_cell_size).floor().long().clamp_(
        0,self._pickup_grid_nx-1)
    pass_cell_y=(pass_positions[...,1]/pass_cell_size).floor().long().clamp_(
        0,self._pickup_grid_ny-1)
    pass_cell=pass_cell_y*self._pickup_grid_nx+pass_cell_x
    pass_slot=(self.team_ids[safe_owner]+1).clamp_(1,2)
    world_index=self._world_indices[:,None].expand_as(safe_owner)
    piece_index=self._fuel_indices.expand_as(safe_owner)
    old_cell=self._pickup_possible_cells[world_index,piece_index,pass_slot]
    self._pickup_possible_cells[world_index,piece_index,pass_slot]=torch.where(
        passed,pass_cell.to(old_cell.dtype),old_cell)
    if self.fuel_physics is None:
        self.piece_vel.masked_fill_(passed[...,None], 0.)
    self.piece_owner.masked_fill_(passed, -1)
    pass_team=self.team_ids[safe_owner]
    in_home=torch.where(pass_team==0,
                        pass_positions[...,0]<=self.alliance_zone_depth,
                        pass_positions[...,0]>=self.sim.field_length-self.alliance_zone_depth)
    home_zone=torch.where(in_home,pass_team+1,torch.zeros_like(pass_team))
    torch.where(passed,home_zone,self.piece_zone,out=self.piece_zone)
    passed_counts = torch.zeros_like(self.fuel_passed_event)
    passed_counts.scatter_add_(1, safe_owner,
                               passed.to(self.fuel_passed_event.dtype))
    if curriculum_metrics is not None:
        curriculum_metrics.released(passed, owner_before_pass)
    self.fuel_passed_event.copy_(torch.where(
        active[:,None], passed_counts, self.fuel_passed_event))
    self.next_ferry.copy_(torch.where(passed_counts>0,
        self.match_elapsed[:,None]+self.robot_score_intervals[None],
        self.next_ferry))
    track_clear_mask |= passed

    newly_scored=torch.zeros_like(self.piece_active)
    bump_clear=self._robots_clear_of_bumps(self.sim.pose[:,:,:2])
    if robot_count==NUM_ROBOTS and all_robots_controlled and self._vectorized_scoring_enabled:
        # Score checks are independent across robot slots. Keep the same
        # per-robot first-piece rule, but evaluate six slots in one tensor
        # pass instead of launching a chain of small kernels per robot.
        held=(self.piece_active[:,None,:]&
              (self.piece_owner[:,None,:]==self._robot_owner_ids[None,:,None])&
              active[:,None,None])
        hub=self.hub_centers[self.team_ids][None].expand(self.n,-1,-1)
        delta=hub-self.sim.pose[:,:,:2]
        bearing=torch.atan2(delta[...,1],delta[...,0])
        error=torch.atan2(torch.sin(bearing-self.sim.pose[:,:,2]),
                          torch.cos(bearing-self.sim.pose[:,:,2])).abs()
        stationary=(self.sim.velocity[:,:,:2].norm(dim=-1)<=.20)&(
            self.sim.velocity[:,:,2].abs()<=.20)
        aimed=torch.where(self._dumper_mask[None],
                          (error<=math.pi/9)&stationary,
                          torch.ones_like(stationary))
        robot_x=self.sim.pose[:,:,0]
        in_alliance_zone=torch.where(self.team_ids[None,:]==0,
            robot_x<=self.alliance_zone_depth,
            robot_x>=self.field_length-self.alliance_zone_depth)
        ready=in_alliance_zone&aimed&bump_clear
        has_fuel=held.any(-1)
        robot_hub_active=self.hub_active[:,self.team_ids]
        entering=(ready&~self.last_hub_zone&has_fuel&active[:,None]&
                  (~robot_hub_active|~in_alliance_zone))
        self.fuel_denied_event.copy_(torch.where(
            active[:,None],entering.long(),self.fuel_denied_event))
        self.fuel_denied_count+=entering.long()
        can_score=(ready&score_intent&robot_hub_active&in_alliance_zone&
            active[:,None]&
            (self.match_elapsed[:,None]+1e-6>=self.next_score))
        eligible=held&can_score[...,None]
        first=eligible.to(torch.int64).argmax(-1,keepdim=True)
        chosen=eligible&(self._fuel_indices[:,None,:]==first)
        count=chosen.sum(-1)
        newly_scored=chosen.any(1)
        self.fuel_scored_event.copy_(torch.where(
            active[:,None],count,self.fuel_scored_event))
        team_scores=torch.zeros_like(self.fuel_score_count)
        team_scores.scatter_add_(1,self.team_ids[None].expand(self.n,-1),count)
        self.fuel_score_count+=team_scores
        _capture_fuel_score_awards(self, chosen,
            source_site="gamepiece_actions.update_fuel_actions.vectorized_score_mutation")
        auto_scores=torch.zeros_like(self.auto_fuel_scores)
        auto_scores.scatter_add_(1,self.team_ids[None].expand(self.n,-1),
            count*(self.match_elapsed[:,None]<=20.+self.dt).long())
        self.auto_fuel_scores+=auto_scores
        self.next_score.copy_(torch.where(count>0,
            self.match_elapsed[:,None]+self.robot_score_intervals[None],
            self.next_score))
        self.last_hub_zone.copy_(torch.where(
            active[:,None],ready,self.last_hub_zone))
    else:
        for robot in range(robot_count):
            if self.control_modes[robot] == "none":
                continue
            team=self.sim._team_ids_host[robot]
            held=self.piece_active&(self.piece_owner==robot)&active[:,None]
            hub=self.hub_centers[team]
            delta=hub-self.sim.pose[:,robot,:2]
            bearing=torch.atan2(delta[:,1],delta[:,0])
            error=torch.atan2(torch.sin(bearing-self.sim.pose[:,robot,2]),
                              torch.cos(bearing-self.sim.pose[:,robot,2])).abs()
            if self.robot_types[robot]=="dumper":
                stationary=(self.sim.velocity[:,robot,:2].norm(dim=-1)<=.20)&(
                    self.sim.velocity[:,robot,2].abs()<=.20)
                aimed=(error<=math.pi/9)&stationary
            else:
                aimed=torch.ones_like(error,dtype=torch.bool)
            robot_x=self.sim.pose[:,robot,0]
            in_alliance_zone=(robot_x<=self.alliance_zone_depth if team==0 else
                robot_x>=self.field_length-self.alliance_zone_depth)
            ready=in_alliance_zone&aimed&bump_clear[:,robot]
            has_fuel=held.any(-1)
            entering=(ready&~self.last_hub_zone[:,robot]&has_fuel&active&
                      (~self.hub_active[:,team]|~in_alliance_zone))
            self.fuel_denied_event[:,robot]=torch.where(active,entering.long(),
                                                         self.fuel_denied_event[:,robot])
            self.fuel_denied_count[:,robot]+=entering.long()
            can_score=(ready&score_intent[:,robot]&
                (self.match_elapsed+1e-6>=self.next_score[:,robot])&
                self.hub_active[:,team]&in_alliance_zone&active)
            eligible=held&can_score[:,None]
            chosen=eligible&(self._fuel_indices==eligible.to(torch.int64).argmax(-1,keepdim=True))
            count=chosen.sum(-1); newly_scored|=chosen
            self.fuel_scored_event[:,robot]=torch.where(active,count,
                                                         self.fuel_scored_event[:,robot])
            self.fuel_score_count[:,team]+=count
            _capture_fuel_score_awards(self, chosen,
                source_site="gamepiece_actions.update_fuel_actions.robot_loop_score_mutation",
                robot_slot=robot)
            self.auto_fuel_scores[:,team]+=count*(self.match_elapsed<=20.+self.dt).long()
            self.next_score[:,robot]=torch.where(
                count>0,self.match_elapsed+self.robot_score_interval_values[robot],
                self.next_score[:,robot])
            self.last_hub_zone[:,robot]=torch.where(active,ready,self.last_hub_zone[:,robot])

    if curriculum_metrics is not None:
        curriculum_metrics.scored(newly_scored, self.piece_owner.clone())
    if hasattr(self, "_meco_piece_generation"):
        self._meco_piece_generation.add_(newly_scored.long())
    _invalidate_perceived_contact_tracks(self, active)
    team=original_piece_team
    origins=self.hub_centers[team]
    spawn_origins,launch_targets=shooter_respawn_cone(self,origins)
    if self.fuel_physics is None:
        torch.where(newly_scored[...,None],spawn_origins,
                    self.piece_pos,out=self.piece_pos)
        velocity=(launch_targets-origins)*.1
        torch.where(newly_scored[...,None],velocity,self.piece_vel,
                    out=self.piece_vel)
    else:
        launch_fuel(self,newly_scored,launch_targets,
                    origins=origins,height=1.83,flight_time=1.,
                    horizontal_velocity_scale=.15,
                    spawn_positions=spawn_origins)
    torch.where(newly_scored,torch.full_like(self.piece_owner,-1),
                self.piece_owner,out=self.piece_owner)
    torch.where(newly_scored,torch.zeros_like(self.piece_zone),
                self.piece_zone,out=self.piece_zone)
    torch.where(newly_scored[...,None],self._midfield_respawn_cells,
                self._pickup_possible_cells,out=self._pickup_possible_cells)
    self.fuel_acquisition_count += self.fuel_acquired_event*active[:,None]
    return (self.fuel_acquired_event,self.fuel_scored_event,
            self.fuel_denied_event)
