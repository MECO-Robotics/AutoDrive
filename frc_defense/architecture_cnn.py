"""Full-field semantic-raster actor/critics for architecture comparisons.

``FullFieldCNN`` uses the convolution-plus-spatial-flatten idea of DQN
(Mnih et al., 2015, https://doi.org/10.1038/nature14236), with small 3x3
kernels suited to this 32x64 semantic grid. ``FullFieldResNet`` uses IMPALA's
pooling stages and preactivation residual blocks (Espeholt et al., 2018,
https://proceedings.mlr.press/v80/espeholt18a.html). These are adaptations of
the encoders, not reproductions of those papers' training algorithms.

Both retain a 4x8 spatial map before projection; global averaging would erase
absolute field location. A scalar branch retains precise robot/candidate
measurements. The caller must construct all dynamic raster channels from
simulated perception, never hidden simulator state. Neither model has temporal
memory, so tracking age/velocity must be supplied in its observations.
"""
from __future__ import annotations

import torch
from torch import nn


class FullFieldCNN(nn.Module):
    """Compact full-field convolutional encoder with scalar late fusion."""

    def __init__(self, scalar_dim: int = 137, action_dim: int = 8):
        super().__init__()
        self.field_encoder = nn.Sequential(
            nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 4 * 8, 128), nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 128), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(256, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        field = self.field_encoder(inputs["field"])
        scalars = self.scalar_encoder(inputs["scalars"])
        features = self.fusion(torch.cat((field, scalars), dim=-1))
        return self.actor(features), self.critic(features).squeeze(-1)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class FullFieldResNet(nn.Module):
    """IMPALA-inspired full-field residual encoder with scalar late fusion."""

    def __init__(self, scalar_dim: int = 137, action_dim: int = 8):
        super().__init__()
        stages: list[nn.Module] = []
        in_channels = 12
        for channels in (16, 32, 32):
            stages.extend((
                nn.Conv2d(in_channels, channels, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                _ResidualBlock(channels), _ResidualBlock(channels),
            ))
            in_channels = channels
        self.field_encoder = nn.Sequential(
            *stages, nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 4 * 8, 128), nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 128), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(256, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        field = self.field_encoder(inputs["field"])
        scalars = self.scalar_encoder(inputs["scalars"])
        features = self.fusion(torch.cat((field, scalars), dim=-1))
        return self.actor(features), self.critic(features).squeeze(-1)
