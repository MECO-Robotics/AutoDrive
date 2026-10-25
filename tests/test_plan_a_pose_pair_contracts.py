import pytest
import torch
from types import SimpleNamespace

from frc_defense.meco_plan_a_policy import PlanAActionSupport, gather_pose_behavior_logits
from frc_defense.meco_pose_graph import _connected_region_ids
from frc_defense.meco_tracks import make_meco_track, sample_candidate_action
from frc_defense.meco_track_training import _plan_a_logits, _plan_a_selection_logits


def _pair_support():
    # Two graph nodes with eight canonical rows each. Behaviors 1 occupy the
    # intake-target descriptor slots, and action IDs follow planner order.
    behavior_order = (0, 1, 1, 1, 1, 2, 3, 4)
    action_order = (7, 0, 1, 2, 3, 4, 6, 5)
    return PlanAActionSupport(
        node_ids=torch.tensor([0, 1]),
        behavior_ids=torch.tensor(behavior_order * 2),
        action_ids=torch.tensor(action_order * 2),
        pair_node_ids=torch.tensor([0] * 8 + [1] * 8),
        descriptor_ids=torch.tensor([-1, 0, 1, 2, 3, -1, -1, -1] * 2),
        valid_mask=torch.tensor([True] + [False] * 11 + [True] + [False] * 3),
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


def test_plan_a_device_selector_uses_valid_fallback_for_invalid_logits():
    logits = torch.tensor((float("nan"), 4., 9., 2.))
    valid = torch.tensor((False, True, True, False))
    selected_logits = _plan_a_selection_logits(logits, valid,
        torch.tensor(1), torch.tensor(False))
    assert selected_logits.tolist() == [-torch.inf, 0., -torch.inf, -torch.inf]
    action, logprob, entropy = sample_candidate_action(selected_logits[None])
    assert action.tolist() == [1]
    assert logprob.tolist() == [0.]
    assert entropy.tolist() == [0.]


def test_plan_a_device_selector_fails_closed_when_no_route_is_feasible():
    logits = torch.tensor((0., 1., 2., 3.))
    valid = torch.zeros(4, dtype=torch.bool)
    selected_logits = _plan_a_selection_logits(logits, valid,
        torch.tensor(0), torch.tensor(False))
    assert torch.isneginf(selected_logits).all()


@pytest.mark.parametrize("track", ("A1", "A2", "A3"))
def test_plan_a_full_scorer_and_ppo_logprob_share_selected_pair(track):
    support = _pair_support()
    graph = type("Graph", (), {
        "node_count": 2,
        "xy": torch.tensor([[1.0, 1.0], [3.0, 1.0]]),
        "heading_bin": torch.tensor([2, 4]),
        "lookup": torch.zeros((8, 2, 4), dtype=torch.long),
        "nx": 2, "ny": 4, "resolution_m": 1.0,
    })()
    utility = (torch.zeros(1, 8, 2, 4) if track == "A1" else
               torch.tensor([[0.0, 1.25]]))
    if track == "A1":
        utility[0, 4, 1, 3] = 1.25
    raw = {"utility": utility,
           "behavior_logits": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])}
    raster = torch.zeros(1, 6, 2, 4)
    route_batch = SimpleNamespace(
        distance_m=torch.tensor([0.0, 2.0]),
        learned_cost_grid_edges=torch.tensor([0.0, 0.5]))
    env = SimpleNamespace(planner=SimpleNamespace(length=4.0, width=2.0),
        sim=SimpleNamespace(speed=torch.tensor([[0., 0., 0., 1.]]),
                            field_length=4.0, field_width=2.0))
    config = SimpleNamespace(route_cost_points_per_second=2.0,
        p_mode="learned", p_cost_scale=3.0, p_points_per_grid_edge=4.0)

    logits, path_cost = _plan_a_logits(track, raw, support, graph, raster, env,
        config, route_nodes=torch.tensor([0.0, 4.0]),
        risk_nodes=torch.tensor([0.0, 0.5]), route_batch=route_batch)
    torch.testing.assert_close(path_cost, torch.tensor([0.0, 6.0]))
    torch.testing.assert_close(logits[[0, 12]], torch.tensor([0.1, -9.05]))
    action, logprob, _ = sample_candidate_action(logits.unsqueeze(0))
    replay_logprob = torch.distributions.Categorical(logits=logits).log_prob(
        action.reshape(()))
    torch.testing.assert_close(logprob.reshape(()), replay_logprob)


def test_pose_region_partition_preserves_ordered_directed_bfs_semantics():
    edges = [[1, 0], [2, 0], [0, 0], [3, 0]]
    allowed = [[True, False], [True, False], [False, False], [False, False]]
    labels = _connected_region_ids(edges, allowed,
        torch.zeros((4, 2)), region_count=3)
    assert labels == [0, 0, 1, 2]


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

    assert torch.equal(logits, torch.tensor(
        [11.5] + [-torch.inf] * 11 + [70.0] + [-torch.inf] * 3))
    selected_pair, _, _ = sample_candidate_action(logits.unsqueeze(0), deterministic=True)
    assert selected_pair.tolist() == [12]
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
