from dataclasses import dataclass
from types import SimpleNamespace

import torch

from frc_defense import meco_track_training as training
from frc_defense.experimental_route_evaluation import DeviceRouteBatch


@dataclass
class _Support:
    goal_xy: torch.Tensor
    valid_mask: torch.Tensor
    distances_m: torch.Tensor
    fallback_pair_tensor: torch.Tensor
    support_digest: str = ''
    order_digest: str = ''


def _setup(monkeypatch, *, corrupt=False):
    x, y = torch.meshgrid(torch.arange(3.) + .5, torch.arange(2.) + .5, indexing='ij')
    grid = torch.stack((x, y), -1).reshape(-1, 2)
    order = torch.randperm(48, generator=torch.Generator().manual_seed(24))
    goals = grid.repeat(8, 1)[order]
    if corrupt:
        goals[0, 0] += .1
    class Planner:
        nx, ny, xx, yy = 3, 2, x, y
        def _indices(self, points):
            return points[:, 0].floor().long().clamp(0, 2), points[:, 1].floor().long().clamp(0, 1)
    env = SimpleNamespace(planner=Planner())
    support = _Support(goals, torch.ones(48 * 8, dtype=torch.bool),
                       torch.zeros(48), torch.tensor(0))
    graph = SimpleNamespace(node_valid=torch.ones(1, 48, dtype=torch.bool))
    graph.node_valid[0, 3] = False
    calls = []
    def fake_route(env, queries, goals, **kwargs):
        calls.append((queries, goals.clone(), kwargs))
        count = len(queries)
        values = (goals[:, 0] * 10 + goals[:, 1]).to(torch.float64)
        good = goals[:, 0] < 2
        return DeviceRouteBatch(tuple(queries), good.long() * 3, good,
            goals[:, None, :], good.long(), values, values + 1, values + 2,
            torch.arange(count), cost_provenance=kwargs.get('learned_cost_provenance', 'plain'),
            bump_cost_grid_edges=values + 3, learned_cost_grid_edges=values + 4)
    monkeypatch.setattr(training, 'evaluate_device_route_chunks', fake_route)
    monkeypatch.setattr(training, 'plan_a_support_order_digest', lambda *_: 'order')
    return env, graph, support, calls


def test_dedup_restores_order_costs_query_identity_and_masks(monkeypatch):
    env, graph, support, calls = _setup(monkeypatch)
    penalty = torch.ones(3, 2)
    result, routes = training._plan_a_route_support(env, graph, support, 's', penalty,
        SimpleNamespace(p_mode='learned'))
    assert len(calls) == 1 and len(calls[0][0]) == 6
    assert calls[0][2]['learned_traversal_cost'] is penalty
    assert routes.queries == tuple(training.DeviceRouteQuery(f'plan-a-node:{i}', 's', 0, 3) for i in range(48))
    assert torch.equal(routes.candidate_index, torch.arange(48))
    assert torch.equal(routes.path_xy[:, 0], support.goal_xy)
    expected = (support.goal_xy[:, 0] * 10 + support.goal_xy[:, 1]).double()
    for name, add in [('distance_m', 0), ('endpoint_error_m', 1),
                      ('weighted_cost_grid_edges', 2), ('bump_cost_grid_edges', 3),
                      ('learned_cost_grid_edges', 4)]:
        assert torch.equal(getattr(routes, name), expected + add)
    expected_legal = (support.goal_xy[:, 0] < 2) & graph.node_valid[0]
    assert torch.equal(result.valid_mask.reshape(-1, 8), expected_legal[:, None].expand(-1, 8))
    fallback = expected.masked_fill(~expected_legal, torch.inf).argmin() * 8
    assert torch.equal(result.fallback_pair_tensor, fallback)


def test_same_capacity_invalid_geometry_fails_closed(monkeypatch):
    env, graph, support, _ = _setup(monkeypatch, corrupt=True)
    result, routes = training._plan_a_route_support(env, graph, support, 's', None,
        SimpleNamespace(p_mode='none'))
    assert not routes.feasible.any()
    assert (routes.status_code == DeviceRouteBatch.INVALID).all()
    assert not result.valid_mask.any()
    assert result.fallback_pair_tensor.item() == -1


def test_canonical_check_stays_on_device(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:
                raise AssertionError('unexpected device-to-host scalar read')
            return func(*args, **(kwargs or {}))

    env, graph, support, _ = _setup(monkeypatch)
    with NoScalarRead():
        training._plan_a_route_support(env, graph, support, 's', None,
            SimpleNamespace(p_mode='none'))
