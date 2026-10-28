import math
import heapq
import random
from pathlib import Path

import torch

from frc_defense import meco_plan_a_shortest_path as shortest_path
from frc_defense.meco_plan_a_shortest_path import (
    cached_plan_a_edge_weights,
    plan_a_dijkstra_cpu,
    plan_a_fallback_pair,
    precompute_plan_a_edge_weights,
)


def _solve(dst, cost, length, allowed, *, root=0, start=0., connector=0.):
    return plan_a_dijkstra_cpu(dst, cost, length, allowed, root=root,
        root_start_cost=start, root_connector_m=connector)


def test_heap_ties_follow_distance_then_node_id_and_slot_relaxation():
    # Root lists node 2 before node 1. Equal path costs still settle by node ID.
    result = _solve([[2, 1], [], []], [[1., 1.], [], []],
        [[1., 1.], [], []], [[True, True], [], []])
    assert result["traversal_order"] == [0, 1, 2]
    assert result["parent_ids"] == [-1, 0, 0]


def test_near_epsilon_proposal_does_not_replace_existing_parent():
    result = _solve([[2, 1], [2], []], [[1., .5], [.5 - .5e-9], []],
        [[1., .5], [.5 - .5e-9], []], [[True, True], [True], []])
    assert result["route_cost"][2] == 1.
    assert result["parent_ids"][2] == 0
    assert result["parent_edge_m"][2] == 1.


def test_disconnected_nodes_keep_infinite_defaults_and_no_parent():
    result = _solve([[1], [], [1]], [[2.], [], [0.]],
        [[2.], [], [1.]], [[True], [], [True]])
    assert result["route_cost"] == [0., 2., math.inf]
    assert result["physical_distance"] == [0., 2., math.inf]
    assert result["parent_ids"] == [-1, 0, -1]
    assert result["parent_edge_m"] == [0., 2., 0.]
    assert result["traversal_order"] == [0, 1]


def test_root_connector_and_turn_cost_keep_route_and_physical_distance_distinct():
    xy = [[0., 0.], [3., 4.]]
    heading = [0., math.pi / 2]
    dst, costs, translations, allowed = precompute_plan_a_edge_weights(
        xy, heading, [[1], [0]], [[True], [False]])
    assert dst == [[1], [0]]
    assert translations[0][0] == 5.
    assert costs[0][0] == 5. + .25 * abs(math.atan2(math.sin(math.pi / 2),
                                                      math.cos(math.pi / 2)))
    result = _solve(dst, costs, translations, allowed, start=2., connector=3.)
    assert result["route_cost"] == [2., 2. + costs[0][0]]
    assert result["physical_distance"] == [3., 8.]
    assert result["parent_ids"] == [-1, 0]


def test_traversal_order_is_final_distance_then_node_id():
    result = _solve([[2, 1], [3], [3], []], [[1., 1.], [1.], [1.], []],
        [[1., 1.], [1.], [1.], []], [[True, True], [True], [True], []])
    assert result["traversal_order"] == [0, 1, 2, 3]


def test_fallback_uses_physical_path_distance_then_first_descriptor():
    valid = [[False, True, True], [True, False, True], [False, False, False]]
    assert plan_a_fallback_pair(valid, [1., .5, 0.]) == 3
    assert plan_a_fallback_pair(valid, [.5, .5, 0.]) == 1


def _legacy_python_dijkstra(xy, heading, edges, edge_mask, *, root, start, connector):
    n = len(xy)
    dist, physical = [math.inf] * n, [math.inf] * n
    parent, parent_edge = [-1] * n, [0.] * n
    dist[root], physical[root] = float(start), float(connector)
    queue = [(dist[root], root)]
    while queue:
        distance, src = heapq.heappop(queue)
        if distance != dist[src]:
            continue
        for dst, allowed in zip(edges[src], edge_mask[src]):
            if not allowed:
                continue
            dx = xy[src][0] - xy[dst][0]
            dy = xy[src][1] - xy[dst][1]
            translation = math.hypot(dx, dy)
            delta = heading[dst] - heading[src]
            turn = abs(math.atan2(math.sin(delta), math.cos(delta)))
            cost = translation + .25 * turn
            proposal = distance + cost
            if proposal < dist[dst] - 1e-9:
                dist[dst] = proposal
                physical[dst] = physical[src] + translation
                parent[dst] = src
                parent_edge[dst] = translation
                heapq.heappush(queue, (proposal, dst))
    order = sorted((node for node, value in enumerate(dist) if math.isfinite(value)),
                   key=lambda node: (dist[node], node))
    return {"route_cost": dist, "physical_distance": physical,
        "parent_ids": parent, "parent_edge_m": parent_edge,
        "traversal_order": order}


def test_cpu_oracle_matches_previous_python_path_on_synthetic_graphs():
    rng = random.Random(90817)
    for _ in range(20):
        n, slots = 11, 4
        xy = [[rng.uniform(-2., 2.), rng.uniform(-2., 2.)] for _ in range(n)]
        heading = [rng.uniform(-math.pi, math.pi) for _ in range(n)]
        edges, masks = [], []
        for src in range(n):
            destinations = [rng.randrange(n) for _ in range(slots)]
            edges.append(destinations)
            masks.append([rng.random() < .6 for _ in range(slots)])
        dst, cost, length, allowed = precompute_plan_a_edge_weights(
            xy, heading, edges, masks)
        root = rng.randrange(n)
        expected = _legacy_python_dijkstra(xy, heading, edges, masks,
            root=root, start=.37, connector=.19)
        actual = plan_a_dijkstra_cpu(dst, cost, length, allowed,
            root=root, root_start_cost=.37, root_connector_m=.19)
        assert actual == expected


def test_static_edge_weights_are_cached_for_graph_lifetime(monkeypatch):
    class StaticGraph:
        pass
    graph = StaticGraph()
    graph.xy = torch.tensor([[0., 0.], [1., 0.]])
    graph.heading = torch.tensor([0., .1])
    graph.edges = torch.tensor([[[1], [0]]])
    graph.edge_mask = torch.tensor([[[True], [False]]])
    real_precompute = shortest_path.precompute_plan_a_edge_weights
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return real_precompute(*args, **kwargs)
    monkeypatch.setattr(shortest_path, "precompute_plan_a_edge_weights", counted)
    first = cached_plan_a_edge_weights(graph)
    second = cached_plan_a_edge_weights(graph)
    assert first is second
    assert len(calls) == 1


def test_hip_prototype_is_fail_closed():
    assert shortest_path.PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED is False
    try:
        shortest_path.plan_a_dijkstra_hip(None, None, None, None, root=0,
            root_start_cost=0., root_connector_m=0.)
    except RuntimeError as exc:
        assert "disabled pending WX9100 parity and profiling" in str(exc)
    else:
        raise AssertionError("HIP prototype must remain fail-closed before hardware gates")



def test_graph_hip_entry_is_gated_before_any_graph_materialization(monkeypatch):
    assert shortest_path.PLAN_A_HIP_GRAPH_PARITY_PROFILE_GATE_PASSED is False
    def forbidden(_graph):
        raise AssertionError("gated graph path must not materialize graph on CPU")
    monkeypatch.setattr(shortest_path, "cached_plan_a_edge_weights", forbidden)
    try:
        shortest_path.plan_a_dijkstra_hip_for_graph(object(), device="cuda:0",
            root=0, root_start_cost=0., root_connector_m=0.)
    except RuntimeError as exc:
        assert "disabled pending WX9100 parity and profiling" in str(exc)
    else:
        raise AssertionError("device-resident graph path must remain gated")


def test_device_resident_graph_path_has_no_host_graph_copy_or_python_heap():
    import inspect
    source = inspect.getsource(shortest_path.plan_a_dijkstra_hip_for_graph)
    assert "cached_plan_a_edge_weights" not in source
    assert ".cpu(" not in source
    assert "graph.heading.tolist(" not in source
    assert "graph.edges[0].tolist(" not in source
    assert "dijkstra_graph" in source


def test_turn_table_matches_cpu_math_for_float32_headings():
    headings = (torch.arange(8, dtype=torch.float32) * (2 * math.pi / 8)).tolist()
    values, turns = shortest_path.plan_a_cpu_turn_table(headings, device="cpu")
    values, turns = values.tolist(), turns.tolist()
    assert values == sorted(headings)
    for i, src in enumerate(values):
        for j, dst in enumerate(values):
            delta = dst - src
            assert turns[i][j].hex() == abs(math.atan2(math.sin(delta),
                                                       math.cos(delta))).hex()


def test_cpu_traversal_sorts_final_costs_after_zero_cost_discovery():
    result = plan_a_dijkstra_cpu([[2], [], [1]], [[0.], [], [0.]],
        [[0.], [], [0.]], [[True], [], [True]], root=0,
        root_start_cost=0., root_connector_m=0.)
    assert result["traversal_order"] == [0, 1, 2]
    assert result["parent_ids"] == [-1, 2, 0]


def test_hip_kernel_has_device_side_edge_and_heap_guards():
    source_root = Path(__file__).parents[1] / "frc_defense" / "csrc"
    binding = (source_root / "meco_plan_a_shortest_path_hip.cpp").read_text()
    kernel = (source_root / "meco_plan_a_shortest_path_hip_kernel.cu").read_text()
    assert "overflows the HIP heap capacity" in binding
    assert "status.item<int64_t>()" in binding
    assert "dst < 0 || dst >= n" in kernel
    assert "heap_size >= heap_capacity" in kernel


def test_graph_and_preweighted_gates_are_independent(monkeypatch):
    import pytest
    from types import SimpleNamespace

    class ReachedDeviceCheck(Exception):
        pass

    def reached(_device):
        raise ReachedDeviceCheck

    monkeypatch.setattr(shortest_path, "_require_wx9100_device", reached)
    graph = lambda: shortest_path.plan_a_dijkstra_hip_for_graph(
        object(), device="cuda:0", root=0, root_start_cost=0., root_connector_m=0.)
    preweighted = lambda: shortest_path.plan_a_dijkstra_hip(
        None, SimpleNamespace(device="cuda:0"), None, None,
        root=0, root_start_cost=0., root_connector_m=0.)

    monkeypatch.setattr(shortest_path, "PLAN_A_HIP_GRAPH_PARITY_PROFILE_GATE_PASSED", True)
    with pytest.raises(ReachedDeviceCheck):
        graph()
    with pytest.raises(RuntimeError, match="disabled pending WX9100"):
        preweighted()

    monkeypatch.setattr(shortest_path, "PLAN_A_HIP_GRAPH_PARITY_PROFILE_GATE_PASSED", False)
    monkeypatch.setattr(shortest_path, "PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED", True)
    with pytest.raises(RuntimeError, match="disabled pending WX9100"):
        graph()
    with pytest.raises(ReachedDeviceCheck):
        preweighted()


def test_candidate_offline_enforces_both_gates_before_fixtures(monkeypatch):
    import pytest
    from frc_defense import meco_plan_a_hip_parity_harness as harness

    def forbidden(_root):
        raise AssertionError("fixtures must not run with an open gate")

    monkeypatch.setattr(harness, "_fixture_result", forbidden)
    for gate in ("PLAN_A_HIP_GRAPH_PARITY_PROFILE_GATE_PASSED",
                 "PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED"):
        with monkeypatch.context() as patch:
            patch.setattr(shortest_path, gate, True)
            with pytest.raises(RuntimeError, match="production gate changed"):
                harness.run_offline()
    assert harness._closed_gate_provenance(Path(__file__).resolve().parents[1]) == {
        "production_gate_open": False,
        "production_gates_open": {"graph": False, "preweighted": False},
    }


def test_candidate_failure_report_leaves_gate_state_unknown(monkeypatch, tmp_path, capsys):
    import json
    import sys
    from scripts import verify_plan_a_hip_parity as runner

    def fail(_root):
        raise RuntimeError("intentional failure")

    monkeypatch.setattr(runner, "run_offline", fail)
    monkeypatch.setattr(sys, "argv", ["verify", "--mode", "offline", "--report-dir", str(tmp_path)])
    assert runner.main() == 1
    report = json.loads(Path(capsys.readouterr().out.strip()).read_text())
    assert report["production_gate_open"] is None
    assert report["production_gates_open"] == {"graph": None, "preweighted": None}


def test_offline_rejects_gate_provenance_from_another_checkout(tmp_path, monkeypatch):
    import pytest
    from frc_defense import meco_plan_a_hip_parity_harness as harness

    def forbidden(_root):
        raise AssertionError("fixtures must not run with unbound gate provenance")

    monkeypatch.setattr(harness, "_fixture_result", forbidden)
    with pytest.raises(RuntimeError, match="gate provenance.*requested checkout"):
        harness.run_offline(tmp_path)
