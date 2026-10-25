"""CPU-only contracts for the legacy Plan A HIP route fixtures."""
import inspect

from frc_defense import meco_counterfactual
from frc_defense.meco_plan_a_hip_parity_harness import _support_digest


def test_offline_route_fixture_builds_complete_graph_contract_for_digest():
    graph = {
        "xy": [[0.0, 0.0], [1.0, 0.0]],
        "heading": [0.0, 0.0],
        "edge_dst": [[1], [0]],
        "edge_allowed": [[True], [False]],
        "root": 0,
        "root_connector_m": 0.0,
    }
    result = {
        "route_cost": [0.0, 1.0],
        "physical_distance": [0.0, 1.0],
        "parent_ids": [-1, 0],
        "parent_edge_m": [0.0, 1.0],
        "traversal_order": [0, 1],
    }

    digest = _support_digest(graph, result, "synthetic-fixture")

    assert len(digest) == 64


def test_counterfactual_support_digest_is_bound_after_descriptor_materialization():
    source = inspect.getsource(meco_counterfactual._prefix_snapshot)
    assert source.index("a_descriptors.append(item)") < source.index(
        "support_hash = (plan_a_support_digest")
    assert "support = replace(support, support_digest=support_hash)" in source
