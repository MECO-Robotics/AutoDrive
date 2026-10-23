"""Dense controls and an experimental Fourier field policy for architecture screening.

The field policy adapts spectral mixing from Li et al. (arXiv:2010.08895).
It is a policy architecture experiment, not a pretrained dynamics operator.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BaselineMLP(nn.Module):
    """Match the existing categorical ActorCritic's two 128-unit tanh layers."""

    def __init__(self, scalar_dim: int = 137, action_dim: int = 8):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(scalar_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        features = self.trunk(inputs["scalars"])
        return self.actor(features), self.critic(features).squeeze(-1)


class _ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.5 * self.layers(x)


class ResidualMLP(nn.Module):
    """Larger scalar-only control to distinguish capacity from spatial inputs."""

    def __init__(self, scalar_dim: int = 137, action_dim: int = 8):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(scalar_dim, 128), nn.GELU(),
            *(_ResidualBlock(128) for _ in range(3)), nn.LayerNorm(128),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        features = self.trunk(inputs["scalars"])
        return self.actor(features), self.critic(features).squeeze(-1)


class _SpectralMix(nn.Module):
    """Mix positive/negative vertical modes of a real 2-D field separately."""

    def __init__(self, width: int, modes_y: int = 4, modes_x: int = 6):
        super().__init__()
        self.modes_y, self.modes_x = modes_y, modes_x
        # Real storage makes the number of trainable real scalars explicit.
        self.weights = nn.Parameter(
            torch.randn(2, width, width, modes_y, modes_x, 2) / width
        )

    def forward(self, x: Tensor) -> Tensor:
        spectrum = torch.fft.rfft2(x, norm="ortho")
        output = torch.zeros_like(spectrum)
        my = min(self.modes_y, x.shape[-2] // 2)
        mx = min(self.modes_x, spectrum.shape[-1])
        weights = torch.view_as_complex(self.weights.contiguous())
        output[:, :, :my, :mx] = torch.einsum(
            "bihw,iohw->bohw", spectrum[:, :, :my, :mx], weights[0, :, :, :my, :mx]
        )
        output[:, :, -my:, :mx] = torch.einsum(
            "bihw,iohw->bohw", spectrum[:, :, -my:, :mx], weights[1, :, :, :my, :mx]
        )
        return torch.fft.irfft2(output, s=x.shape[-2:], norm="ortho")


class FieldFNOPolicy(nn.Module):
    """Full-field global spectral mixing with precise scalar features.

    Coordinate channels and spatial pooling preserve absolute field position.
    Zero padding reduces, but does not eliminate, the FFT's periodic-boundary
    bias. The pointwise path retains local information discarded by mode cuts.
    """

    def __init__(self, scalar_dim: int = 137, action_dim: int = 8):
        super().__init__()
        width = 16
        self.lift = nn.Conv2d(14, width, 1)
        self.spectral = nn.ModuleList(_SpectralMix(width) for _ in range(3))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(3))
        self.scalar_branch = nn.Sequential(nn.Linear(scalar_dim, 64), nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(width * 4 * 8 + 64, 128), nn.GELU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        field = inputs["field"]
        batch, _, height, width = field.shape
        ys = torch.linspace(-1, 1, height, device=field.device, dtype=field.dtype)
        xs = torch.linspace(-1, 1, width, device=field.device, dtype=field.dtype)
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        coordinates = torch.stack((x, y)).unsqueeze(0).expand(batch, -1, -1, -1)
        spatial = self.lift(torch.cat((field, coordinates), dim=1))
        spatial = F.pad(spatial, (0, 4, 0, 4))
        for spectral, local in zip(self.spectral, self.local):
            spatial = F.gelu(spectral(spatial) + local(spatial))
        spatial = spatial[:, :, :height, :width]
        spatial = F.adaptive_avg_pool2d(spatial, (4, 8)).flatten(1)
        features = self.trunk(torch.cat((spatial, self.scalar_branch(inputs["scalars"])), dim=1))
        return self.actor(features), self.critic(features).squeeze(-1)
