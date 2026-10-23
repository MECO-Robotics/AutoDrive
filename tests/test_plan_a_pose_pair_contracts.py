import pytest
import torch

from frc_defense.meco_plan_a_policy import PlanAActionSupport, gather_pose_behavior_logits
from frc_defense.meco_tracks import make_meco_track, sample_candidate_action


def _pair_support():
    # Two graph nodes, each paired with the four core behaviors. The action
    # IDs follow the planner's canonical behavior-to-simulator mapping.
    return PlanAActionSupport(
        node_ids=torch.tensor([0, 1]),
        behavior_ids=torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]),
        action_ids=torch.tensor([7, 0, 1, 2, 3, 4, 6, 5]),
        pair_node_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        descriptor_ids=torch.tensor([-1] * 8),
        valid_mask=torch.tensor([True, False, False, False,
                                 True, False, False, False]),
        distances_m=torch.tensor([0.0, 1.0]),
        route_cost=torch.tensor([0.0, 1.0]),
        parent_ids=torch.tensor([0, 0]),
        parent_edge_m=torch.tensor([0.0, 1.0]),
        traversal_order=torch.tensor([0, 1]),
        ancestor_jumps=torch.tensor([[0, 0]]),
        root_node=0,
        root_connector_m=0.0,
        fallback_pair=0,
        goal_xy=torch.tensor([[1.0, 1.0], [3.0, 1.0]]),
        goal_heading=torch.zeros(2),
        snapshot_id="synthetic-support",
        support_digest="synthetic",
    )


@pytest.mark.parametrize("track", ("A1", "A2", "A3"))
def test_plan_a_gathers_expected_pose_plus_behavior_and_masks_invalid_pairs(track):
    support = _pair_support()
    graph = type("Graph", (), {
        "xy": torch.tensor([[1.0, 1.0], [3.0, 1.0]]),
        "heading_bin": torch.tensor([2, 4]),
    })()
    if track == "A1":
        utility = torch.zeros(1, 8, 2, 4)
        utility[0, 2, 1, 1] = 10.0
        utility[0, 4, 1, 3] = 20.0
        raster = torch.zeros(1, 6, 2, 4)
    else:
        utility = torch.tensor([[10.0, 20.0]])
        raster = torch.zeros(1, 6, 2, 4)
    behavior = torch.tensor([[1.5, 50.0, 0.0, 0.0, 1.5]])

    logits = gather_pose_behavior_logits(
        track, {"utility": utility, "behavior_logits": behavior}, graph,
        raster, support, field_length=4.0, field_width=2.0)

    assert torch.equal(logits, torch.tensor([
        11.5, -torch.inf, -torch.inf, -torch.inf,
        21.5, -torch.inf, -torch.inf, -torch.inf,
    ]))
    selected_pair, _, _ = sample_candidate_action(logits.unsqueeze(0), deterministic=True)
    assert selected_pair.tolist() == [4]
    pair_index = int(selected_pair.item())
    assert int(support.action_ids[pair_index].item()) == 3
    assert int(support.pair_node_ids[pair_index].item()) == 1


@pytest.mark.parametrize("track", ("A1", "A2", "A3"))
def test_plan_a_models_have_finite_nonnegative_penalty_on_small_cpu_fixture(track):
    torch.manual_seed(31)
    raster = torch.zeros(1, 6, 25, 51)
    globals_ = torch.zeros(1, 7)
    model = make_meco_track(track).eval()
    graph = None
    if track in ("A2", "A3"):
        nodes = 8
        graph = {
            "node_features": torch.zeros(1, nodes, 9),
            "edges": torch.arange(nodes).view(1, nodes, 1),
            "edge_mask": torch.ones(1, nodes, 1, dtype=torch.bool),
            "region_ids": torch.arange(nodes).view(1, nodes) % 64,
        }
        if track == "A2":
            graph["coarse_edges"] = torch.arange(64).view(1, 64, 1)
            graph["coarse_edge_mask"] = torch.zeros(1, 64, 1, dtype=torch.bool)

    with torch.no_grad():
        output = model(raster=raster, globals_=globals_, **(graph or {}))

    penalty = output["penalty"]
    assert penalty.device.type == "cpu"
    assert torch.isfinite(penalty).all()
    assert (penalty >= 0).all()
