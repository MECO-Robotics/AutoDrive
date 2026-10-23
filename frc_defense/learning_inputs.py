"""MECO observation adapter built only from curriculum telemetry and tracks."""
from __future__ import annotations

import math

import torch

from .learning_contract import ObservationSnapshot


RASTER_CELL_METERS = 0.3302
TRACK_TIMEOUT_SECONDS = 2.0


def _signed_hub_timer(env, world: int, slot: int) -> float:
    """Signed seconds to the next known HUB transition, positive when active."""
    elapsed = float(env.match_elapsed[world].item())
    schedule = getattr(env, "curriculum_schedule", None)
    if schedule is None:
        boundaries = (30., 55., 80., 105., 130., 160.)
    else:
        boundaries = tuple(float(boundary) for boundary, _ in schedule[1:])
    next_boundary = next((value for value in boundaries if value > elapsed), elapsed)
    remaining = max(0., next_boundary - elapsed)
    active = bool(env.hub_active[world, env.team_ids[slot]].item())
    return remaining if active else -remaining


def _deposit(raster, channel: int, world: int, positions, confidence, age,
             *, field_length: float, field_width: float) -> None:
    height, width = raster.shape[-2:]
    positions = positions.reshape(-1, 2)
    confidence = confidence.reshape(-1).clamp(0., 1.)
    age = age.reshape(-1).clamp_min(0.)
    valid = (torch.isfinite(positions).all(-1) & torch.isfinite(confidence) &
             torch.isfinite(age) & (confidence > 0.) &
             (positions[:, 0] >= 0.) & (positions[:, 0] < field_length) &
             (positions[:, 1] >= 0.) & (positions[:, 1] < field_width))
    if not bool(valid.any()):
        return
    points = positions[valid]
    weights = confidence[valid]
    times = -age[valid]
    columns = (points[:, 0] / RASTER_CELL_METERS).floor().long().clamp(0, width - 1)
    rows = (points[:, 1] / RASTER_CELL_METERS).floor().long().clamp(0, height - 1)
    cells = rows * width + columns
    confidence_flat = raster[world, channel].view(-1)
    confidence_flat.scatter_reduce_(0, cells, weights, reduce="amax", include_self=True)
    time_flat = raster[world, channel + 1].view(-1)
    time_sum = torch.zeros_like(time_flat)
    weight_sum = torch.zeros_like(time_flat)
    time_sum.scatter_add_(0, cells, times * weights)
    weight_sum.scatter_add_(0, cells, weights)
    occupied = weight_sum > 0.
    time_flat[occupied] = time_sum[occupied] / weight_sum[occupied]


@torch.no_grad()
def build_observation_snapshot(env, *, world: int = 0, slot: int = 3,
                               episode_id: str | None = None,
                               sequence: int | None = None) -> ObservationSnapshot:
    """Pack `[6,H,W]` tracks and the proposal's seven global scalars.

    Signed-time cells contain negative track age (seconds before this snapshot);
    current ally telemetry has age zero. Ally positions come from the
    simulator's explicit team telemetry channel (`sim.pose` for own-team slots),
    which must be shared with every comparison policy. Hardware qualification
    requires confirming an equivalent localization feed. Simulator truth is
    used neither to fill FUEL cells nor to locate opponents.
    """
    if not 0 <= world < env.n or not 0 <= slot < 6:
        raise ValueError("world/slot is outside the curriculum environment")
    height = math.ceil(float(env.sim.field_width) / RASTER_CELL_METERS)
    width = math.ceil(float(env.sim.field_length) / RASTER_CELL_METERS)
    raster = torch.zeros((1, 6, height, width), device=env.device,
                         dtype=env.sim.pose.dtype)
    field_length = float(env.sim.field_length)
    field_width = float(env.sim.field_width)

    fuel_valid = env.track_mask[world, slot]
    _deposit(raster, 0, 0, env.track_pos[world, slot, fuel_valid],
             env.track_confidence[world, slot, fuel_valid],
             env.track_age[world, slot, fuel_valid],
             field_length=field_length, field_width=field_width)

    team_ids = env.team_ids
    ally_ids = torch.nonzero((team_ids == team_ids[slot]) &
                             (torch.arange(6, device=env.device) != slot),
                             as_tuple=False).flatten()
    ally_positions = env.sim.pose[world, ally_ids, :2]
    in_field = ((ally_positions[:, 0] >= 0.) & (ally_positions[:, 0] < field_length) &
                (ally_positions[:, 1] >= 0.) & (ally_positions[:, 1] < field_width))
    ally_positions = ally_positions[in_field]
    ally_confidence = torch.ones(ally_positions.shape[0], device=env.device,
                                 dtype=raster.dtype)
    ally_age = torch.zeros_like(ally_confidence)
    _deposit(raster, 2, 0, ally_positions, ally_confidence, ally_age,
             field_length=field_length, field_width=field_width)

    opponent_valid = bool(env.opponent_valid[world, slot].item())
    if opponent_valid:
        age = env.opponent_age[world, slot].reshape(1).clamp_min(0.)
        confidence = torch.exp(-age / TRACK_TIMEOUT_SECONDS)
        _deposit(raster, 4, 0, env.opponent_pose[world, slot, :2].reshape(1, 2),
                 confidence, age, field_length=field_length, field_width=field_width)

    pose = env.sim.pose[world, slot]
    load = (env.piece_owner[world] == slot).sum().to(raster.dtype)
    globals_ = (float(pose[0].item()), float(pose[1].item()),
                float(torch.sin(pose[2]).item()), float(torch.cos(pose[2]).item()),
                float(load.item()), _signed_hub_timer(env, world, slot),
                float(env.match_remaining[world].item()))
    episode = episode_id or f"curriculum-seed-{int(env.curriculum['seed'])}"
    step = int(env.steps[world].item()) if sequence is None else int(sequence)
    if step < 0:
        raise ValueError("sequence must be nonnegative")
    snapshot_id = f"{episode}:{step}"
    return ObservationSnapshot(
        snapshot_id=snapshot_id,
        episode_id=episode,
        sequence=step,
        timestamp=float(env.match_elapsed[world].item()),
        raster_shape=(6, height, width),
        raster=tuple(raster[0].reshape(-1).cpu().tolist()),
        globals=globals_,
    )
