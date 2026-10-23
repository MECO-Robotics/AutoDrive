"""Permutation-invariant policies over simulated perception tracks.

Compact adaptations of Deep Sets (Zaheer et al., 2017) and attention-based
relational policies (Lee et al., 2019; Zambaldi et al., 2018).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeepSetsPolicy(nn.Module):
    """Encode each track independently, then aggregate mean, max, and count."""

    def __init__(self, scalar_dim: int = 137, action_dim: int = 8) -> None:
        super().__init__()
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 64), nn.Tanh())
        self.entity_encoder = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.fusion = nn.Sequential(nn.Linear(193, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        mask = inputs["entity_mask"].bool()
        entities = inputs["entities"].masked_fill(~mask.unsqueeze(-1), 0.0)
        encoded = self.entity_encoder(entities)
        count = mask.sum(dim=1, keepdim=True)
        mean = encoded.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)
        mean = mean / count.clamp_min(1)
        # Include a masked dummy so even a genuinely zero-length set is valid.
        maximum = torch.cat(
            (
                encoded.masked_fill(~mask.unsqueeze(-1), -torch.inf),
                encoded.new_full((encoded.shape[0], 1, 64), -torch.inf),
            ),
            dim=1,
        ).amax(dim=1)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        features = self.fusion(
            torch.cat(
                (self.scalar_encoder(inputs["scalars"]), mean, maximum, count.log1p()),
                dim=-1,
            )
        )
        return self.actor(features), self.critic(features).squeeze(-1)


class _EntityAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 4, dropout=0.0, batch_first=True)
        self.norm_feedforward = nn.LayerNorm(64)
        self.feedforward = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        normalized = self.norm_attention(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.feedforward(self.norm_feedforward(tokens))
        return tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class EntityAttentionPolicy(nn.Module):
    """Relate observed entities with self-attention and a scalar context token.

    The always-valid context token supplies the final pooled representation and
    avoids all-masked attention when perception has no tracks. There are no
    sequence positions: entity ordering carries no meaning.
    """

    def __init__(self, scalar_dim: int = 137, action_dim: int = 8) -> None:
        super().__init__()
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 64), nn.Tanh())
        self.entity_encoder = nn.Sequential(nn.Linear(12, 64), nn.GELU())
        self.blocks = nn.ModuleList([_EntityAttentionBlock(), _EntityAttentionBlock()])
        self.fusion = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        mask = inputs["entity_mask"].bool()
        entities = inputs["entities"].masked_fill(~mask.unsqueeze(-1), 0.0)
        context = self.scalar_encoder(inputs["scalars"]).unsqueeze(1)
        tokens = torch.cat((context, self.entity_encoder(entities)), dim=1)
        padding_mask = torch.cat((mask.new_zeros((mask.shape[0], 1)), ~mask), dim=1)
        for block in self.blocks:
            tokens = block(tokens, padding_mask)
        features = self.fusion(tokens[:, 0])
        return self.actor(features), self.critic(features).squeeze(-1)
