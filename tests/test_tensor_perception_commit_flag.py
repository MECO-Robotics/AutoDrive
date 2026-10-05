import torch

from frc_defense import tensor_perception_commit_proto as commit_proto
from frc_defense.tensor_sim import TensorDefenseEnv


def _new_env(seed):
    env = TensorDefenseEnv(
        num_envs=4, task="counter_defense", device="cpu", seed=seed,
        opponent="static", action_mode="direct", randomize=True,
        observation_noise=0.0, observation_dropout=0.0, fuel_count=96,
        perception_config={"detection_dropout": .15,
            "position_noise_m": .02, "velocity_noise_mps": .04})
    env.reset(seed=seed)
    return env


def _assert_perception_equal(lhs, rhs):
    for name in ("_track_pos", "_track_vel", "_track_mask", "_track_age",
                 "_opponent_track_pose", "_opponent_track_velocity",
                 "_opponent_track_valid", "_opponent_track_age"):
        assert torch.equal(getattr(lhs, name), getattr(rhs, name)), name
    assert torch.equal(lhs.generator.get_state(), rhs.generator.get_state())


def test_opted_in_perception_commit_falls_back_on_cpu(monkeypatch):
    monkeypatch.setattr(commit_proto, "FUSED_PERCEPTION_COMMIT_HIP_ENABLED", True)
    assert not commit_proto.fused_commit_available(torch.device("cpu"))
    reference = _new_env(21931)
    candidate = _new_env(21931)
    active = torch.tensor((True, False, True, False))
    count = int(active.sum().item())

    original = commit_proto.fused_commit_available
    monkeypatch.setattr(commit_proto, "fused_commit_available",
                        lambda device: original(device))
    reference._update_perception(active_mask=active, active_count=count)
    candidate._update_perception(active_mask=active, active_count=count)
    _assert_perception_equal(reference, candidate)


def test_reset_mask_keeps_existing_perception_path_when_opted_in(monkeypatch):
    monkeypatch.setattr(commit_proto, "FUSED_PERCEPTION_COMMIT_HIP_ENABLED", True)
    reference = _new_env(21932)
    candidate = _new_env(21932)
    mask = torch.tensor((False, True, False, True))
    count = int(mask.sum().item())
    reference._update_perception(reset_mask=mask, active_mask=mask,
                                 active_count=count)
    candidate._update_perception(reset_mask=mask, active_mask=mask,
                                 active_count=count)
    _assert_perception_equal(reference, candidate)
