"""Scoring-grid replacement invalidates captured target tensor addresses."""
from types import SimpleNamespace

import torch

from frc_defense._tensor_3v3_actions import TensorThreeVsThreeActionMixin


def test_prepare_score_grid_invalidates_target_graph():
    owner = SimpleNamespace(
        sim=SimpleNamespace(length=torch.ones(1, 6), width=torch.ones(1, 6),
                            bump_regions=torch.empty(0, 4)),
        planner=SimpleNamespace(_boxes=torch.empty(0, 4), _bumps=torch.empty(0, 4)),
        team_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        alliance_zone_depth=4.028, field_length=16.54, n=1, device="cpu",
        _target_action_graph=object(), _target_action_graph_outputs=object(),
        _target_action_graph_audit=object(), _target_action_graph_signature=object(),
    )
    TensorThreeVsThreeActionMixin._prepare_score_grid(owner)
    for name in ("_target_action_graph", "_target_action_graph_outputs",
                 "_target_action_graph_audit", "_target_action_graph_signature"):
        assert getattr(owner, name) is None
    assert owner._score_grid_points.shape == (1, 6, 0, 2)
