from types import SimpleNamespace

import pytest
import torch

from frc_defense.experimental_route_evaluation import _path_weighted_cost_device


@pytest.mark.parametrize('points', [0, 1, 2, 73])
@pytest.mark.parametrize('count', [0, 1, 7])
@pytest.mark.parametrize('scale', [0., .71, 100.])
def test_shared_crossings_preserve_independent_costs(points, count, scale):
    generator = torch.Generator().manual_seed(341)
    scratch = SimpleNamespace(nx=13, ny=8, resolution=0.25,
        _bump_cost=torch.rand((13, 8), generator=generator))
    path = torch.rand((count, points, 2), generator=generator) * 4
    lengths = torch.randint(0, points + 1, (count,), generator=generator)
    learned = torch.rand((13, 8), generator=generator)
    expected_bump = _path_weighted_cost_device(scratch, path, lengths, None, 1.)
    expected_weighted = _path_weighted_cost_device(scratch, path, lengths, learned, scale)
    weighted, bump = _path_weighted_cost_device(
        scratch, path, lengths, learned, scale, return_bump=True)
    assert torch.equal(expected_weighted, weighted)
    assert torch.equal(expected_bump, bump)
