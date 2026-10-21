"""Observed features must not reveal hidden opponent speed parameters."""
import torch

from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def test_observation_ignores_hidden_enemy_physical_speed():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        env = TensorThreeVsThreeEnv(
            num_envs=1, device="cpu", seed=731, randomize=False,
            fuel_count=96, fuel_physics=False,
            control_modes=("none", "none", "none", "nn", "none", "none"),
        )
        # Hold the permitted sensor measurements fixed while hidden physical
        # parameters change. These are fixture observations, not injected
        # simulator truth used for training or behavioral qualification.
        env.track_mask.zero_()
        env.track_mask[0, 3, 0] = True
        env.track_pos[0, 3, 0] = env.sim.pose[0, 3, :2] + torch.tensor((-1., 0.))
        env.opponent_valid.zero_()
        env.opponent_pose.zero_()
        env.opponent_velocity.zero_()
        first = env.observe()[:, 3].clone()
        env.sim.speed[:, :3] = torch.tensor((.1, 17., 39.))
        second = env.observe()[:, 3]
        torch.testing.assert_close(first, second, rtol=0., atol=0.)
    finally:
        torch.set_num_threads(previous)
