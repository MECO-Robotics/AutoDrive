import json

from frc_defense.tensor_training import train


def test_ppo_status_records_policy_kl_and_clip_fraction(tmp_path):
    output = tmp_path / "ppo"
    train(task="defense", timesteps=2, output=output, seed=20261021,
          num_envs=1, device="cpu", opponent="mirror", rollout_steps=2,
          epochs=1, minibatch_size=2, horizon=4, algorithm="ppo",
          architecture="direct")

    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["updates"] == 1
    assert status["approx_kl"] == 0.0
    assert status["clip_fraction"] == 0.0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["approx_kl"] == status["approx_kl"]
    assert metadata["clip_fraction"] == status["clip_fraction"]
