import pytest
import torch

from frc_defense import tensor_perception
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def _require_fused_hip():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("fused sensor update requires HIP")
    if tensor_perception._hip_perception_extension() is None:
        pytest.skip("fused sensor update extension is unavailable")


def test_fused_perception_commit_respects_inactive_worlds_and_sensor_statistics():
    _require_fused_hip()
    device = torch.device("cuda", torch.cuda.current_device())
    worlds, robots, pieces = 3, 6, 2048
    visible = torch.ones((worlds, robots, pieces), device=device, dtype=torch.bool)
    active = torch.tensor([True, False, True], device=device)
    ticks = torch.tensor([7, 11, 19], device=device, dtype=torch.long)
    piece_pos = torch.zeros((worlds, pieces, 2), device=device)
    piece_vel = torch.zeros_like(piece_pos)
    track_pos = torch.full((worlds, robots, pieces, 2), 9., device=device)
    track_vel = torch.full_like(track_pos, 9.)
    track_age = torch.zeros((worlds, robots, pieces), device=device)
    track_mask = torch.zeros((worlds, robots, pieces), device=device, dtype=torch.bool)
    before = tuple(t[1].clone() for t in
                   (track_pos, track_vel, track_age, track_mask))

    tensor_perception.perception_commit_3v3(
        visible, active, ticks, piece_pos, piece_vel, track_pos, track_vel,
        track_age, track_mask, seed=1234, dt=.02, dropout=.02,
        position_noise=.015, velocity_noise=.05)
    torch.cuda.synchronize(device)

    for tensor, expected in zip((track_pos, track_vel, track_age, track_mask), before):
        assert torch.equal(tensor[1], expected)
    updated = track_mask[active]
    update_rate = updated.float().mean().item()
    assert .97 < update_rate < .99
    observed_pos = track_pos[active][updated]
    observed_vel = track_vel[active][updated]
    assert abs(observed_pos.mean().item()) < .002
    assert .013 < observed_pos.std().item() < .017
    assert abs(observed_vel.mean().item()) < .006
    assert .045 < observed_vel.std().item() < .055
    assert torch.all(track_age[active][updated] == 0)


def test_fused_perception_environment_advances_only_active_sensor_counters():
    _require_fused_hip()
    device = torch.device("cuda", torch.cuda.current_device())
    env = TensorThreeVsThreeEnv(num_envs=2, device=device, seed=53,
        randomize=False, fused_sensor_rng=True)
    assert env._fused_sensor_rng_used
    mask = torch.tensor([True, False], device=device)
    inactive_state = tuple(t[1].clone() for t in
        (env.track_pos, env.track_vel, env.track_age, env.track_mask,
         env.opponent_pose, env.opponent_velocity, env.opponent_age,
         env.opponent_valid))
    ticks_before = env._perception_rng_ticks.clone()
    env._update_perception(mask, active_count=1)
    torch.cuda.synchronize(device)
    for tensor, expected in zip(
            (env.track_pos, env.track_vel, env.track_age, env.track_mask,
             env.opponent_pose, env.opponent_velocity, env.opponent_age,
             env.opponent_valid), inactive_state):
        assert torch.equal(tensor[1], expected)
    assert env._perception_rng_ticks[0] == ticks_before[0] + 1
    assert env._perception_rng_ticks[1] == ticks_before[1]


def test_fused_opponent_tracks_match_reference_without_sensor_noise():
    _require_fused_hip()
    device = torch.device("cuda", torch.cuda.current_device())
    kwargs = dict(num_envs=4, device=device, seed=71, randomize=False,
        perception_dropout=0., position_noise=0., velocity_noise=0.)
    fused = TensorThreeVsThreeEnv(**kwargs, fused_sensor_rng=True)
    reference = TensorThreeVsThreeEnv(**kwargs, fused_sensor_rng=False)
    reference.sim.pose.copy_(fused.sim.pose)
    active = torch.ones(4, device=device, dtype=torch.bool)
    fused._update_perception(active, active_count=4)
    reference._update_perception(active, active_count=4)
    torch.cuda.synchronize(device)
    assert fused._fused_opponent_tracks_used
    for actual, expected in zip(
            (fused.opponent_pose, fused.opponent_velocity, fused.opponent_size,
             fused.opponent_age, fused.opponent_valid),
            (reference.opponent_pose, reference.opponent_velocity,
             reference.opponent_size, reference.opponent_age,
             reference.opponent_valid)):
        if actual.dtype == torch.bool:
            assert torch.equal(actual, expected)
        else:
            assert torch.allclose(actual, expected, rtol=0., atol=1e-6,
                                  equal_nan=True)
