import ast
import inspect
from pathlib import Path

from frc_defense import meco_track_training
from frc_defense.meco_dashboard import (
    LEGACY_ROUTE_COST_PROVENANCE,
    _valid_plan_a_penalty_contract,
)
from frc_defense.meco_path_penalty import (
    PLAN_A_PPO_RECOMPUTATION_CONTRACT,
    plan_a_ppo_contract_metadata,
    project_pose_penalty,
)
import pytest
import torch


def _report_penalty(*, mode="learned", contract=PLAN_A_PPO_RECOMPUTATION_CONTRACT):
    return {
        "trained_mode": mode,
        "p0_ablation_mode": "p0",
        "cost_scale": 2.0,
        "points_per_grid_edge": 1.0,
        "projection": "mean_over_legal_heading_bins_xy_grid_v1",
        "map_units": "dimensionless_additive_penalty_per_cardinal_grid_step",
        "path_units": "normalized_grid_edge_units",
        "ppo_recomputation": contract,
    }


def test_protocol_and_report_metadata_share_the_canonical_versioned_contract():
    assert PLAN_A_PPO_RECOMPUTATION_CONTRACT == (
        "projected_xy_penalty_node_integral_over_frozen_legal_se2_shortest_path_tree"
        "_including_root_pose_connector_v1")
    metadata = plan_a_ppo_contract_metadata()
    assert metadata == {"ppo_recomputation": PLAN_A_PPO_RECOMPUTATION_CONTRACT}

    source = inspect.getsource(meco_track_training)
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "plan_a_ppo_contract_metadata"]
    # The same field is included in both the evaluation protocol hash payload
    # and the emitted report metadata.
    assert len(calls) == 2


def test_dashboard_contract_accepts_canonical_and_p0_provenance():
    assert _valid_plan_a_penalty_contract(_report_penalty())
    assert _valid_plan_a_penalty_contract(_report_penalty(mode="p0"))
    assert LEGACY_ROUTE_COST_PROVENANCE == "legacy_bump_cost_path_integral_v2"


def test_dashboard_contract_rejects_legacy_and_unknown_semantics():
    for value in (
        "current_policy_penalty_integrated_on_frozen_rollout_routes",
        "current_policy_penalty_integrated_on_frozen_legal_se2_shortest_path_tree",
        "future_or_unknown_plan_a_penalty_contract_v99",
    ):
        assert not _valid_plan_a_penalty_contract(
            _report_penalty(contract=value))


def test_pending_runtime_validation_and_p0_route_provenance_remain_unchanged():
    source = Path(meco_track_training.__file__).read_text()
    assert source.count('"runtime_route_validation": "pending_representative_simulator_replay"') == 2
    assert 'route_provenance != LEGACY_ROUTE_COST_PROVENANCE' in inspect.getsource(
        __import__("frc_defense.meco_dashboard", fromlist=["_comparison"]))


def test_a2_penalty_projection_ignores_padded_node_values_even_if_lookup_points_there():
    penalty = torch.tensor([[2.0, 9.0e8]])
    lookup = torch.ones((8, 2, 2), dtype=torch.long)
    node_valid = torch.tensor([[True, False]])
    projected = project_pose_penalty("A2", penalty, lookup, None, 2.0, 2.0,
                                    node_valid=node_valid)
    torch.testing.assert_close(projected, torch.zeros((2, 2)))


def test_a2_penalty_projection_requires_explicit_node_validity():
    with pytest.raises(ValueError, match="requires node_valid"):
        project_pose_penalty("A2", torch.zeros((1, 2)),
            torch.zeros((8, 2, 2), dtype=torch.long), None, 2.0, 2.0)
