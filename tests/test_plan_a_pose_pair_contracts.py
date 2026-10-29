import pytest
import torch
from dataclasses import replace
from types import SimpleNamespace

from frc_defense.meco_plan_a_policy import (PlanAActionSupport,
    gather_pose_behavior_logits, plan_a_support_digest_cpu_reference)
from frc_defense.meco_pose_graph import _connected_region_ids
from frc_defense.meco_tracks import make_meco_track, sample_candidate_action
from frc_defense.meco_track_training import (_plan_a_logits,
    _plan_a_selection_logits, _prepare_plan_a_decision)


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
        torch.tensor(-1), torch.tensor(False))
    assert torch.isneginf(selected_logits).all()


@pytest.mark.parametrize("track", ("A1", "A2", "A3"))
def test_plan_a_full_scorer_and_ppo_logprob_share_selected_pair(track):
    support = _pair_support()
    graph = type("Graph", (), {
        "node_capacity": 2,
        "xy": torch.tensor([[1.0, 1.0], [3.0, 1.0]]),
        "heading_bin": torch.tensor([2, 4]),
        "heading": torch.tensor([0.0, 0.0]),
        "edges": torch.zeros((1, 2, 1), dtype=torch.long),
        "edge_mask": torch.zeros((1, 2, 1), dtype=torch.bool),
        "lookup": torch.zeros((8, 2, 4), dtype=torch.long),
        "nx": 4, "ny": 4, "resolution_m": 1.0,
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
        learned_cost_grid_edges=torch.tensor([0.0, 99.0]))
    penalty_map = torch.zeros(4, 4)
    penalty_map[3, 1] = 0.5
    env = SimpleNamespace(planner=SimpleNamespace(length=4.0, width=2.0),
        sim=SimpleNamespace(speed=torch.tensor([[0., 0., 0., 1.]]),
                            field_length=4.0, field_width=2.0))
    config = SimpleNamespace(route_cost_points_per_second=2.0,
        p_mode="learned", p_cost_scale=3.0, p_points_per_grid_edge=4.0)

    logits, path_cost = _plan_a_logits(track, raw, support, graph, raster, env,
        config, route_nodes=torch.tensor([0.0, 4.0]),
        risk_nodes=torch.tensor([0.0, 0.5]), route_batch=route_batch,
        penalty_map=penalty_map)
    print("PDBG", path_cost)
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


def test_plan_a_support_digest_binds_device_fallback_pair():
    support = _pair_support()
    graph = type("Graph", (), {
        "heading_bin": torch.tensor([2, 4]),
        "xy": support.goal_xy,
        "edges": torch.zeros((1, 2, 1), dtype=torch.long),
        "edge_mask": torch.zeros((1, 2, 1), dtype=torch.bool),
        "node_valid": torch.ones((1, 2), dtype=torch.bool),
        "region_ids": torch.zeros((1, 2), dtype=torch.long),
        "coarse_edges": torch.zeros((1, 64, 64), dtype=torch.long),
        "coarse_edge_mask": torch.zeros((1, 64, 64), dtype=torch.bool),
        "graph_status": torch.zeros((), dtype=torch.long),
    })()
    first = plan_a_support_digest_cpu_reference(
        replace(support, fallback_pair_tensor=torch.tensor(0)), graph)
    second = plan_a_support_digest_cpu_reference(
        replace(support, fallback_pair_tensor=torch.tensor(12)), graph)
    assert first != second
    graph_with_changed_lookup = type("Graph", (), {
        **graph.__dict__,
        "lookup": torch.ones((8, 2, 4), dtype=torch.long),
    })()
    changed_lookup = plan_a_support_digest_cpu_reference(
        replace(support, fallback_pair_tensor=torch.tensor(0)),
        graph_with_changed_lookup)
    assert first != changed_lookup


@pytest.mark.parametrize("route_valid", (False, True))
def test_plan_a_prepare_fails_closed_for_empty_routes_or_invalid_model_input(
        monkeypatch, route_valid):
    from dataclasses import replace
    from types import SimpleNamespace
    import frc_defense.meco_track_training as training

    support = replace(_pair_support(), fallback_pair_tensor=torch.tensor(0),
        order_digest="ordered")
    graph = type("Graph", (), {
        "node_capacity": 2, "xy": support.goal_xy,
        "heading": support.goal_heading,
        "heading_bin": torch.tensor([2, 4]),
        "lookup": torch.zeros((8, 2, 4), dtype=torch.long),
        "edges": torch.zeros((1, 2, 1), dtype=torch.long),
        "edge_mask": torch.zeros((1, 2, 1), dtype=torch.bool),
        "nx": 4, "ny": 4, "resolution_m": 1.0,
    })()
    observation = SimpleNamespace(snapshot_id="synthetic", capture_monotonic=1.,
        timestamp=1., schema_version=1)
    env = SimpleNamespace(device=torch.device("cpu"), _last_obs=torch.zeros(1),
        sim=SimpleNamespace(speed=torch.tensor([[0., 0., 0., 1.]]),
            field_length=4.0, field_width=2.0),
        planner=SimpleNamespace(length=4.0, width=2.0),
        track_pos=torch.zeros((1, 6, 4, 2)))
    env._candidates = lambda: (torch.zeros((1, 6, 4), dtype=torch.long),
                               torch.zeros((1, 6, 4), dtype=torch.bool), None)
    monkeypatch.setattr(training, "build_observation_snapshot",
                        lambda *args, **kwargs: observation)
    monkeypatch.setattr(training, "build_safe_action_candidates",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("Plan A must not build candidates")))
    monkeypatch.setattr(training, "build_plan_a_support",
                        lambda *args, **kwargs: support)
    monkeypatch.setattr(training, "_snapshot_tensors",
                        lambda *args, **kwargs: (torch.zeros(1, 6, 2, 4),
                                                 torch.zeros(1, 7)))
    monkeypatch.setattr(training, "_tracking_reset_hidden",
                        lambda hidden, available, raster: (None, True, False))
    monkeypatch.setattr(training, "_graph_inputs", lambda *args: {})
    monkeypatch.setattr(training, "_plan_a_route_support",
        lambda _env, _graph, incoming, *_args: (
            replace(incoming, valid_mask=(torch.ones_like(incoming.valid_mask)
                    if route_valid else torch.zeros_like(incoming.valid_mask)),
                    fallback_pair_tensor=torch.tensor(0 if route_valid else -1)),
            SimpleNamespace(distance_m=torch.zeros(2),
                            learned_cost_grid_edges=torch.zeros(2))))
    monkeypatch.setattr(training, "_plan_a_logits",
        lambda *args, **kwargs: (torch.full((16,), -torch.inf),
                                 torch.zeros(2)))
    monkeypatch.setattr(training, "_plan_a_prediction_risk",
        lambda _env, _support, _eta, _mode: (torch.zeros(2), {}))
    monkeypatch.setattr(training, "_materialize_plan_a_prediction_metadata",
                        lambda metadata: metadata)
    monkeypatch.setattr(training, "plan_a_support_digest",
                        lambda *_args: "synthetic-digest")
    monkeypatch.setattr(training, "_path_geometry", lambda _env: None)

    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            return {"utility": torch.zeros((1, 2)),
                    "penalty": torch.zeros((1, 2)),
                    "behavior_logits": torch.zeros((1, 5)),
                    "hidden": torch.zeros((1, 64)),
                    "device_input_valid": torch.tensor(not route_valid)}

    prepared = _prepare_plan_a_decision("A2", Model(), env, graph, "episode",
        None, route_cost_points_per_second=1., action_ticks=1, candidate_k=8,
        deterministic=False, value_head=None, p_mode="bump", p_cost_scale=1.,
        p_points_per_grid_edge=1.)
    assert prepared["action"].tolist() == [-1]
    assert prepared["logprob"].tolist() == [0.0]
    assert prepared["entropy"].tolist() == [0.0]
    assert prepared["selection_outcome"].empty_support.item()
    assert prepared["selected_action_payload"][0].item() == -1
    assert prepared["selected_action_payload"][1].item() == -1
    assert prepared["selected_action_payload"][2].item() == -1
    assert prepared["selected_action_payload"][5:7].tolist() == [0.0, 0.0]
    assert prepared["selected_action_payload"][8:11].tolist() == [-1.0, -1.0, -1.0]
    assert prepared["selected_action_payload"][11].item() == 1
    assert prepared["selected_action_payload"][12].item() == 1


def test_plan_a_prepare_has_no_host_tensor_transfer_before_compact_payload():
    import inspect
    import frc_defense.experimental_route_evaluation as routes
    import frc_defense.meco_track_training as training

    source = inspect.getsource(_prepare_plan_a_decision)
    pre_handoff = source.split("selected_action_payload = torch.stack", 1)[0]
    assert ".item(" not in pre_handoff
    assert ".cpu(" not in pre_handoff
    assert ".tolist(" not in pre_handoff
    assert "float(env.sim.speed" not in pre_handoff
    for function in (training._plan_a_logits, training._plan_a_route_support,
                     training._plan_a_prediction_risk_device,
                     routes.evaluate_device_route_chunks):
        body = inspect.getsource(function)
        assert ".item(" not in body
        assert ".cpu(" not in body
        assert ".tolist(" not in body


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
            "node_valid": torch.ones(1, nodes, dtype=torch.bool),
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

@pytest.mark.parametrize('mode', ('current', 'cv', 'multi_model_v1'))
@pytest.mark.parametrize('response_case', ('accepted', 'invalid', 'expired'))
def test_plan_a_reused_risk_production_response_and_forecast(monkeypatch, mode, response_case):
    import frc_defense.meco_track_training as training
    import frc_defense.meco_plan_a_support_tree as tree
    from frc_defense.meco_tracks import MecoInferenceResponse
    monkeypatch.setenv('AUTODRIVE_EXPERIMENTAL_PLAN_A_RISK_REUSE', '1')
    support = replace(_pair_support(), fallback_pair_tensor=torch.tensor(0), order_digest='b' * 64)
    graph = SimpleNamespace(node_capacity=2, lookup=torch.zeros((8, 2, 4), dtype=torch.long),
        nx=4, ny=4, resolution_m=1.0)
    env = SimpleNamespace(device=torch.device('cpu'), _last_obs=torch.zeros(1),
        sim=SimpleNamespace(speed=torch.tensor([[0., 0., 0., 2.]]), field_length=4., field_width=2.),
        planner=SimpleNamespace(length=4., width=2.), track_pos=torch.zeros((1, 6, 4, 2)),
        opponent_valid=torch.tensor([[False, False, False, True]]),
        opponent_pose=torch.tensor([[[0.,0.],[0.,0.],[0.,0.],[1.,1.]]]),
        opponent_velocity=torch.tensor([[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[.2,.1,.3]]]),
        opponent_age=torch.tensor([[0.,0.,0.,.1]]))
    env._candidates = lambda: (torch.zeros((1, 6, 4), dtype=torch.long), torch.zeros((1, 6, 4), dtype=torch.bool), None)
    obs = SimpleNamespace(snapshot_id='synthetic', capture_monotonic=100., timestamp=1., schema_version=1)
    monkeypatch.setattr(training.time, 'monotonic', lambda: 100.)
    monkeypatch.setattr(training, 'build_observation_snapshot', lambda *a, **k: obs)
    monkeypatch.setattr(training, 'build_plan_a_support', lambda *a, **k: support)
    monkeypatch.setattr(training, '_snapshot_tensors', lambda *a, **k: (torch.zeros(1,6,2,4), torch.zeros(1,7)))
    monkeypatch.setattr(training, '_tracking_reset_hidden', lambda *a: (None, True, False))
    monkeypatch.setattr(training, '_graph_inputs', lambda *a: {})
    from frc_defense.meco_input_provenance import input_contract_digest
    monkeypatch.setattr(training, 'validate_serialized_inputs', lambda *a: input_contract_digest())
    monkeypatch.setattr(training, '_path_geometry', lambda *a: None)
    monkeypatch.setattr(tree, 'support_digest_v1_compatible', lambda *a: 'digest')
    monkeypatch.setattr(training, 'gather_pose_behavior_logits', lambda track, raw, *a, **k: raw['utility'].reshape(-1)[:1].expand(16))
    calls = []
    def routes(_env, _graph, incoming, *_args):
        count = len(calls)
        distances = torch.tensor([1., 2.]) if count == 0 else torch.tensor([2., 4.])
        changed = replace(incoming, valid_mask=torch.ones_like(incoming.valid_mask), fallback_pair_tensor=torch.tensor(0))
        calls.append(distances)
        return changed, SimpleNamespace(distance_m=distances)
    monkeypatch.setattr(training, '_plan_a_route_support', routes)
    risk_original = training._plan_a_prediction_risk
    risk_calls = []
    def risk(*args):
        result = risk_original(*args)
        risk_calls.append((args[2].clone(), result[0].clone()))
        return result
    monkeypatch.setattr(training, '_plan_a_prediction_risk', risk)
    class Adapter:
        checkpoint_sha256 = 'c' * 64
        protocol_id = 'synthetic'
        inference_counters = {}
        def next_request_identity(self): return (1, 'g')
        def submit_inference(self, request, inputs): return True
        def poll_inference(self, request, handle):
            completed = request.expires_monotonic + 1 if response_case == 'expired' else 100.
            return MecoInferenceResponse(schema=training.MECO_INFERENCE_SCHEMA, track='A1',
                checkpoint_sha256=request.checkpoint_sha256, protocol_id=request.protocol_id,
                model_id=request.model_id, observation_timestamp=request.observation_timestamp,
                capture_monotonic=request.capture_monotonic, dispatch_monotonic=request.dispatch_monotonic,
                completed_monotonic=completed, expires_monotonic=request.expires_monotonic,
                request_sequence=request.request_sequence, generation_id=request.generation_id,
                snapshot_id=request.snapshot_id, candidate_mask_digest=request.candidate_mask_digest,
                legal_pose_order_digest=request.legal_pose_order_digest if response_case != 'invalid' else 'wrong',
                input_contract_sha256=request.input_contract_sha256,
                outputs={'utility': torch.ones(1,8,2,4), 'penalty': torch.zeros(1,8,2,4),
                    'behavior_logits': torch.zeros(1,5), 'hidden': torch.zeros(1,64),
                    '__inference_valid': torch.tensor(True)})
    monkeypatch.setattr(training, 'ProductionMecoAdapter', Adapter)
    prepared = _prepare_plan_a_decision('A1', Adapter(), env, graph, 'episode', None,
        route_cost_points_per_second=1., action_ticks=1, candidate_k=8, deterministic=True,
        value_head=None, p_mode='bump', p_cost_scale=1., p_points_per_grid_edge=1., prediction_mode=mode)
    accepted = response_case == 'accepted'
    assert prepared['inference_provenance']['status'] == ('accepted' if accepted else 'deterministic_fallback')
    assert len(risk_calls) == (2 if accepted else 1)
    expected_eta = calls[-1] / 2.
    expected_risk, expected_metadata = risk_original(env, prepared['support'], expected_eta, mode)
    torch.testing.assert_close(prepared['risk_cost_nodes'], expected_risk, rtol=0, atol=0)
    torch.testing.assert_close(risk_calls[-1][0], expected_eta, rtol=0, atol=0)
    assert prepared['prediction'] == expected_metadata
    torch.testing.assert_close(prepared['decision']['logits'][0, 0],
        ((1. if accepted else 0.) - expected_eta[0] - expected_risk[0]) if accepted else torch.tensor(0.))
