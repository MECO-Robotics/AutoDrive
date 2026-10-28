import torch

from frc_defense.meco_track_training import _ppo_metric_summary


def test_batched_metrics_preserve_python_float_summation():
    policy = [torch.tensor(v, dtype=torch.float32) for v in (1e20, 1., -1e20, -0.125)]
    value = [torch.tensor(v, dtype=torch.float64) for v in (0., 0.2, 1e-30, 3.)]
    entropy = [torch.tensor(v, dtype=torch.float32) for v in (0., 1.2, 0.1, 2.)]
    reward = torch.tensor([1., 2., -0.5])
    summary = _ppo_metric_summary(policy, value, entropy, reward, 4)
    assert summary == {
        'policy_loss': sum(float(v) for v in policy) / len(policy),
        'value_loss': sum(float(v) for v in value) / len(value),
        'entropy': sum(float(v) for v in entropy) / len(entropy),
        'rollout_return': float(reward.sum()), 'rollout_steps': 4,
    }


def test_batched_metrics_handle_empty_epochs():
    summary = _ppo_metric_summary([], [], [], torch.tensor([2., -1.]), 2)
    assert summary == {'policy_loss': 0., 'value_loss': 0., 'entropy': 0.,
                       'rollout_return': 1., 'rollout_steps': 2}
