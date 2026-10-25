import pytest
import math
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from frc_defense.experimental_route_evaluation import (
    DeviceRouteBatch,
    DeviceRouteQuery,
    MAX_ROUTE_QUERY_BATCH,
    evaluate_device_route_chunks,
)
import frc_defense.meco_candidates as meco_candidates
from frc_defense.meco_candidates import (CandidateCommitment, DeviceCandidateBatch,
                                         _goal_nodes_batch)
from frc_defense.meco_track_training import _d1_d2_scores
from types import SimpleNamespace
from frc_defense.learning_contract import CandidateTable
from frc_defense.meco_tracks import (EXTRA_FEATURES, MECO_CANDIDATE_MODEL_INPUT_WIDTH,
                                     SAFE_ACTION_FEATURES,
                                     ProductionMecoAdapter, SafeActionCandidates,
                                     make_meco_track, sample_candidate_action,
                                     score_candidates)


class _NoHostScalarSync(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name == "aten::_local_scalar_dense":
            raise AssertionError("numeric device value synchronized to host before selection")
        return func(*args, **(kwargs or {}))


def test_device_route_batch_keeps_a_fixed_tensor_contract():
    query = DeviceRouteQuery("candidate:score", "episode:4", 0, 3)
    batch = DeviceRouteBatch(
        queries=(query,),
        status_code=torch.tensor([DeviceRouteBatch.FEASIBLE], dtype=torch.long),
        feasible=torch.tensor([True]),
        path_xy=torch.tensor([[[1.0, 2.0], [2.0, 2.0], [2.0, 2.0]]]),
        path_length=torch.tensor([2], dtype=torch.long),
        distance_m=torch.tensor([1.0], dtype=torch.float64),
        endpoint_error_m=torch.tensor([0.0], dtype=torch.float64),
        weighted_cost_grid_edges=torch.tensor([1.12], dtype=torch.float64),
        candidate_index=torch.tensor([0], dtype=torch.long),
    )

    assert batch.device.type == "cpu"  # contract test only; no route evaluation
    assert batch.path_xy.shape == (1, 3, 2)
    assert batch.path_length.tolist() == [2]


def test_device_route_batch_rejects_misaligned_numeric_columns():
    query = DeviceRouteQuery("candidate:score", "episode:4", 0, 3)
    with pytest.raises(ValueError, match="distance_m"):
        DeviceRouteBatch(
            queries=(query,),
            status_code=torch.tensor([DeviceRouteBatch.NO_PATH], dtype=torch.long),
            feasible=torch.tensor([False]),
            path_xy=torch.zeros((1, 2, 2)),
            path_length=torch.tensor([0], dtype=torch.long),
            distance_m=torch.zeros((1, 1)),
            endpoint_error_m=torch.zeros((1,)),
            weighted_cost_grid_edges=torch.zeros((1,)),
            candidate_index=torch.tensor([0], dtype=torch.long),
        )


def test_route_projection_is_an_explicit_report_boundary():
    query = DeviceRouteQuery("candidate:score", "episode:4", 0, 3)
    batch = DeviceRouteBatch(
        queries=(query,),
        status_code=torch.tensor([DeviceRouteBatch.AT_GOAL], dtype=torch.long),
        feasible=torch.tensor([True]),
        path_xy=torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]),
        path_length=torch.tensor([1], dtype=torch.long),
        distance_m=torch.tensor([0.0], dtype=torch.float64),
        endpoint_error_m=torch.tensor([0.0], dtype=torch.float64),
        weighted_cost_grid_edges=torch.tensor([0.0], dtype=torch.float64),
        candidate_index=torch.tensor([0], dtype=torch.long),
    )

    result, = batch.project_results()
    assert (result.candidate_id, result.snapshot_id, result.status) == (
        "candidate:score", "episode:4", "at_goal")
    assert result.feasible is True
    assert result.path_points_m == ((1.0, 2.0),)


def test_device_candidate_batch_gathers_one_compact_host_dispatch_payload():
    query = DeviceRouteQuery("candidate:score", "episode:4", 0, 3)
    routes = DeviceRouteBatch(
        queries=(query,),
        status_code=torch.tensor([DeviceRouteBatch.AT_GOAL], dtype=torch.long),
        feasible=torch.tensor([True]),
        path_xy=torch.tensor([[[1.0, 2.0]]]),
        path_length=torch.tensor([1], dtype=torch.long),
        distance_m=torch.tensor([0.0]),
        endpoint_error_m=torch.tensor([0.0]),
        weighted_cost_grid_edges=torch.tensor([0.0]),
        candidate_index=torch.tensor([0], dtype=torch.long),
    )
    features = torch.zeros((1, 1, 20))
    batch = DeviceCandidateBatch(
        candidate_ids=("candidate:score",),
        feature_names=tuple(f"f{i}" for i in range(20)),
        candidate_index=torch.tensor([0], dtype=torch.long),
        action_index=torch.tensor([4], dtype=torch.long),
        target_track_index=torch.tensor([-1], dtype=torch.long),
        target_xy=torch.tensor([[1.0, 2.0]]),
        goal_xy=torch.tensor([[3.0, 4.0]]),
        goal_heading=torch.tensor([0.5]),
        route_override_flags=torch.tensor([[False, True]]),
        raw_features=features,
        missing_mask=torch.zeros_like(features),
        valid_mask=torch.tensor([[True]]),
        eta_s=torch.tensor([[0.0]]),
        route_cost_points=torch.tensor([[0.0]]),
        risk_cost_points=torch.tensor([[0.0]]),
        route_batch=routes,
        fallback_index=torch.tensor(0, dtype=torch.long),
    )

    payload = batch.gather_action_payload(torch.tensor([0], dtype=torch.long))
    assert payload.shape == (1, 10)
    assert payload[0, :3].tolist() == [0.0, 4.0, -1.0]
    assert batch.candidate_features.shape == (1, 1, 40)


@pytest.mark.parametrize("track", ("C1", "C2"))
def test_production_adapter_consumes_device_candidates_without_host_mask_projection(track):
    ids = ("candidate:valid", "candidate:masked")
    queries = tuple(DeviceRouteQuery(item, "episode:4", 0, 3) for item in ids)
    routes = DeviceRouteBatch(
        queries=queries,
        status_code=torch.tensor([DeviceRouteBatch.AT_GOAL, DeviceRouteBatch.NO_PATH]),
        feasible=torch.tensor([True, False]),
        path_xy=torch.tensor([[[1.0, 2.0]], [[0.0, 0.0]]]),
        path_length=torch.tensor([1, 0]),
        distance_m=torch.tensor([0.0, float("inf")]),
        endpoint_error_m=torch.tensor([0.0, float("inf")]),
        weighted_cost_grid_edges=torch.tensor([0.0, float("inf")]),
        candidate_index=torch.tensor([0, 1]),
    )
    features = tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)
    raw = torch.zeros((1, 2, len(features)))
    batch = DeviceCandidateBatch(
        candidate_ids=ids,
        feature_names=features,
        candidate_index=torch.tensor([0, 1]),
        action_index=torch.tensor([4, 5]),
        target_track_index=torch.tensor([-1, -1]),
        target_xy=torch.zeros((2, 2)),
        goal_xy=torch.zeros((2, 2)),
        goal_heading=torch.zeros(2),
        route_override_flags=torch.zeros((2, 2), dtype=torch.bool),
        raw_features=raw,
        missing_mask=torch.zeros_like(raw),
        valid_mask=torch.tensor([[True, False]]),
        eta_s=torch.tensor([[0.0, 0.0]]),
        route_cost_points=torch.tensor([[0.0, 0.0]]),
        risk_cost_points=torch.tensor([[0.0, 0.0]]),
        route_batch=routes,
        fallback_index=torch.tensor(0),
    )
    table = CandidateTable("episode:4", ids,
        tuple(tuple(None for _ in features) for _ in ids), (True, True), features)

    adapter = object.__new__(ProductionMecoAdapter)
    adapter.track = track
    adapter.checkpoint_sha256 = "a" * 64
    adapter.protocol_id = "contract-test"
    adapter._request_sequence = 0
    adapter._generation_id = "generation-contract-test"
    adapter.next_request_identity = lambda: (1, adapter._generation_id)
    adapter._broker = SimpleNamespace(counters=lambda: {})
    submitted = {}
    adapter.submit_inference = lambda request, inputs: submitted.update(
        request=request, inputs=inputs) or object()
    adapter.poll_inference = lambda request, handle: None

    result = score_candidates(track, adapter,
        raster=torch.zeros((1, 6, 4, 4)),
        globals_=torch.zeros((1, 7)), table=table, candidate_batch=batch,
        route_cost=batch.route_cost_points, risk_cost=batch.risk_cost_points,
        observation_timestamp=1.0,
        capture_monotonic=time.monotonic() - .01)

    assert submitted["request"].candidate_support_mode == "device-mask"
    assert submitted["request"].valid_mask == ()
    assert torch.equal(submitted["inputs"]["candidate_features"], batch.candidate_features)
    assert torch.equal(submitted["inputs"]["__candidate_valid_mask"], batch.valid_mask)
    assert result["inference_status"] == "deterministic_fallback"
    assert torch.isfinite(result["logits"][0, 0])
    assert torch.isneginf(result["logits"][0, 1])


def test_c2_context_matches_unmasked_all_query_baseline_for_empty_and_mixed_support():
    model = make_meco_track("C2").eval()
    raster = torch.randn((1, 6, 16, 16))
    globals_ = torch.randn((1, 7))
    features = torch.randn((1, 4, MECO_CANDIDATE_MODEL_INPUT_WIDTH))
    baseline = model(raster, globals_, features)
    for support in (torch.tensor([[True, False, True, False]]),
                    torch.zeros((1, 4), dtype=torch.bool)):
        masked = model(raster, globals_, features, candidate_mask=support)
        assert torch.equal(masked["hidden"], baseline["hidden"])


@pytest.mark.parametrize("track", ("C1", "C2"))
def test_device_candidate_builder_feeds_scorer_without_materializing_feature_rows(
        monkeypatch, track):
    count = 8
    candidate_count = count + 1
    feature_names = tuple(SAFE_ACTION_FEATURES)
    ids = tuple(f"action:{index}" for index in range(count))
    feature_rows = []
    for index in range(count):
        row = [0.] * len(feature_names)
        row[0] = float(index)
        row[2], row[3] = float(index), 1.
        row[4] = 0.
        row[5] = float(index)
        row[6], row[7] = 10., 6.
        feature_rows.append(tuple(row))
    table = CandidateTable("episode:builder", ids, tuple(feature_rows),
        tuple(True for _ in ids), feature_names)
    base = SafeActionCandidates(SimpleNamespace(snapshot_id="episode:builder"),
        table, "action:7", tuple(range(count)))
    env = SimpleNamespace(
        device=torch.device("cpu"),
        sim=SimpleNamespace(
            pose=torch.zeros((1, 4, 3)), speed=torch.ones((1, 4)),
            omega_limit=torch.ones((1, 4)),
        ),
        dt=.02,
        track_pos=torch.zeros((1, 4, 2, 2)),
        policy_pose=torch.zeros((1, 4, 3)),
        track_confidence=torch.ones((1, 4, 2)),
        track_age=torch.zeros((1, 4, 2)),
        opponent_pose=torch.zeros((1, 4, 3)),
        opponent_valid=torch.ones((1, 4), dtype=torch.bool),
    )
    env._perceived_fuel = lambda _slot: (
        torch.zeros((1, 2, 2)), torch.zeros((1, 2, 2)),
        torch.zeros((1, 2), dtype=torch.bool))

    def fake_geometry(_env, _graph, _specs, fuel_xy, _available, _base_valid):
        headings = torch.zeros(candidate_count)
        return {"collect": torch.zeros(candidate_count, dtype=torch.bool),
            "targets": torch.stack((torch.arange(candidate_count).float(),
                                    torch.ones(candidate_count)), -1),
            "headings": headings, "heading_bins": torch.zeros(candidate_count, dtype=torch.long),
            "nodes": torch.arange(candidate_count),
            "allowed": torch.ones(candidate_count, dtype=torch.bool),
            "capture_inside": torch.ones(candidate_count, dtype=torch.bool)}

    def fake_routes(_env, queries, _goals, **_kwargs):
        route_count = len(queries)
        feasible = torch.tensor([query.candidate_id not in
            ("action:1", "continue:action:7") for query in queries])
        return DeviceRouteBatch(
            queries=tuple(queries),
            status_code=torch.where(feasible,
                torch.full((route_count,), DeviceRouteBatch.AT_GOAL),
                torch.full((route_count,), DeviceRouteBatch.NO_PATH)),
            feasible=feasible,
            path_xy=torch.zeros((route_count, 1, 2)),
            path_length=feasible.to(torch.long),
            distance_m=torch.where(feasible, torch.ones(route_count), torch.zeros(route_count)),
            endpoint_error_m=torch.zeros(route_count),
            weighted_cost_grid_edges=torch.zeros(route_count),
            candidate_index=torch.arange(route_count),
        )

    monkeypatch.setattr(meco_candidates, "_candidate_geometry_device", fake_geometry)
    monkeypatch.setattr(meco_candidates, "evaluate_device_route_chunks", fake_routes)
    monkeypatch.setattr(meco_candidates, "_forecast_risk_batch_device",
        lambda _env, goals, _horizons, available, _mode: (
            torch.zeros_like(goals), available,
            torch.ones_like(available), torch.zeros(candidate_count)))
    commitment = CandidateCommitment("action:7", 7, 0, None, 0, 20)
    with _NoHostScalarSync():
        candidate_set = meco_candidates._build_device_candidate_set(env, base,
            SimpleNamespace(resolution_m=.5),
            [(index, 0, 0., 0., None) for index in range(count)] +
            [(7, 0, 0., 0., None)], {},
            candidate_k=count, action_ticks=1, prediction_mode="current",
            active_commitment=commitment, continue_index=count, oracle_capacity=count,
            learned_traversal_cost=None, learned_cost_scale=1.)

    assert candidate_set.device_batch is not None
    assert all(value is None for row in candidate_set.table.features for value in row)
    assert candidate_set.table.valid_mask == tuple(True for _ in range(candidate_count))
    assert candidate_set.device_batch.valid_mask.tolist() == [
        [True, False] + [True] * 6 + [False]]
    switch_column = candidate_set.table.feature_names.index("commitment_switch_cost")
    assert candidate_set.device_batch.raw_features[0, :, switch_column].tolist() == [
        0.] * candidate_count
    model = make_meco_track(track).eval()
    with _NoHostScalarSync():
        result = score_candidates(track, model,
            raster=torch.zeros((1, 6, 16, 16)), globals_=torch.zeros((1, 7)),
            table=candidate_set.table, candidate_batch=candidate_set.device_batch,
            route_cost=candidate_set.route_cost_points[None],
            risk_cost=candidate_set.risk_cost_points[None])
        selected = sample_candidate_action(result["logits"], deterministic=True)
    assert torch.isfinite(result["logits"][0, 0])
    assert torch.isneginf(result["logits"][0, 1])
    assert selected[0].shape == (1,)


def test_d1_d2_device_batch_scoring_keeps_route_and_risk_cost_on_device():
    feature_names = [f"f{i}" for i in range(20)]
    feature_names[0] = "action_index"
    feature_names[10] = "nearby_sensed_track_count"
    feature_names[11] = "track_confidence"
    feature_names[12] = "track_age_s"
    route = DeviceRouteBatch(
        queries=(DeviceRouteQuery("collect:track:0", "ep:1", 0, 3),
                 DeviceRouteQuery("score", "ep:1", 0, 3)),
        status_code=torch.tensor([3, 3]), feasible=torch.tensor([True, True]),
        path_xy=torch.zeros((2, 1, 2)), path_length=torch.ones(2, dtype=torch.long),
        distance_m=torch.zeros(2), endpoint_error_m=torch.zeros(2),
        weighted_cost_grid_edges=torch.zeros(2), candidate_index=torch.arange(2))
    raw = torch.zeros((1, 2, 20))
    raw[0, 0, 0], raw[0, 0, 10], raw[0, 0, 11], raw[0, 0, 12] = 0, 2, .5, 1
    raw[0, 1, 0] = 4
    batch = DeviceCandidateBatch(
        candidate_ids=("collect:track:0", "score"),
        feature_names=tuple(feature_names),
        candidate_index=torch.arange(2), action_index=torch.tensor([0, 4]),
        target_track_index=torch.tensor([0, -1]), target_xy=torch.zeros((2, 2)),
        goal_xy=torch.zeros((2, 2)), goal_heading=torch.zeros(2),
        route_override_flags=torch.zeros((2, 2), dtype=torch.bool),
        raw_features=raw, missing_mask=torch.zeros_like(raw),
        valid_mask=torch.tensor([[True, True]]), eta_s=torch.zeros((1, 2)),
        route_cost_points=torch.tensor([[.005, .01]]),
        risk_cost_points=torch.tensor([[.1, .2]]), route_batch=route,
        fallback_index=torch.tensor(0))

    scores = _d1_d2_scores(SimpleNamespace(device_batch=batch), None, "D1",
                            route_cost_points_per_second=10.)
    assert torch.allclose(scores, torch.tensor([[.385, -1.4]]), atol=1e-6)
    batch.raw_features[0, 1, 0] = float("nan")
    invalid_scores = _d1_d2_scores(SimpleNamespace(device_batch=batch), None, "D1",
                                   route_cost_points_per_second=10.)
    assert torch.isneginf(invalid_scores[0, 1])


def test_device_route_chunks_keep_query_order_without_host_projection(monkeypatch):
    calls = []

    def fake_evaluate(env, queries, goals, **kwargs):
        calls.append((len(queries), goals.clone()))
        count = len(queries)
        return DeviceRouteBatch(
            queries=tuple(queries), status_code=torch.full((count,), 3, dtype=torch.long),
            feasible=torch.ones(count, dtype=torch.bool),
            path_xy=torch.zeros((count, 1, 2)),
            path_length=torch.ones(count, dtype=torch.long),
            distance_m=goals[:, 0].clone(), endpoint_error_m=torch.zeros(count),
            weighted_cost_grid_edges=torch.zeros(count),
            candidate_index=torch.arange(count, dtype=torch.long))

    import frc_defense.experimental_route_evaluation as route_eval
    monkeypatch.setattr(route_eval, "evaluate_device_routes", fake_evaluate)
    count = MAX_ROUTE_QUERY_BATCH + 2
    queries = tuple(DeviceRouteQuery(f"c{i}", "ep:1", 0, 3) for i in range(count))
    goals = torch.stack((torch.arange(count, dtype=torch.float32),
                         torch.zeros(count)), dim=-1)
    batch = evaluate_device_route_chunks(SimpleNamespace(device="cpu"), queries, goals)

    assert [size for size, _ in calls] == [MAX_ROUTE_QUERY_BATCH, 2]
    assert batch.queries == queries
    assert torch.equal(batch.distance_m, goals[:, 0].to(torch.float64))
    assert torch.equal(batch.candidate_index, torch.arange(count))


def test_goal_node_batch_maps_legal_cells_without_scalar_extraction():
    class Planner:
        @staticmethod
        def _footprint_path_clear(path, heading, length, width):
            return torch.ones(path.shape[0], dtype=torch.bool, device=path.device)

    class Sim:
        field_length = 2.
        field_width = 2.
        length = torch.tensor([[0., 0., 0., .5, 0., 0.]])
        width = torch.tensor([[0., 0., 0., .5, 0., 0.]])

    env = SimpleNamespace(device="cpu", sim=Sim(), planner=Planner(),
        _perceived_robot_obstacles=lambda: torch.zeros((6, 0, 3)))
    class Graph:
        node_count = 1
        resolution_m = 1.
        nx = 2
        ny = 2
        lookup = torch.zeros((8, 2, 2), dtype=torch.long)
        xy = torch.tensor([[.5, .5]])
        heading_bin = torch.tensor([2])

    nodes, centers, bins, valid = _goal_nodes_batch(env, Graph(),
        torch.tensor([[.5, .5], [3., 3.]]), torch.tensor([math.pi / 2, 0.]),
        torch.tensor([False, False]))
    assert nodes.tolist() == [0, -1]
    assert centers[0].tolist() == [.5, .5]
    assert bins.tolist() == [2, 0]
    assert valid.tolist() == [True, False]


@pytest.mark.parametrize("track", ("C1", "C2"))
def test_c_track_scorer_consumes_device_candidate_features_and_mask(track):
    feature_names = tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)
    ids = ("collect:track:0", "hold")
    route = DeviceRouteBatch(
        queries=tuple(DeviceRouteQuery(item, "ep:1", 0, 3) for item in ids),
        status_code=torch.tensor([3, 2]), feasible=torch.tensor([True, False]),
        path_xy=torch.zeros((2, 1, 2)), path_length=torch.tensor([1, 0]),
        distance_m=torch.tensor([1., 0.]), endpoint_error_m=torch.zeros(2),
        weighted_cost_grid_edges=torch.zeros(2), candidate_index=torch.arange(2))
    batch = DeviceCandidateBatch(
        candidate_ids=ids, feature_names=feature_names,
        candidate_index=torch.arange(2), action_index=torch.tensor([0, 7]),
        target_track_index=torch.tensor([0, -1]), target_xy=torch.zeros((2, 2)),
        goal_xy=torch.zeros((2, 2)), goal_heading=torch.zeros(2),
        route_override_flags=torch.zeros((2, 2), dtype=torch.bool),
        raw_features=torch.zeros((1, 2, 20)), missing_mask=torch.zeros((1, 2, 20)),
        valid_mask=torch.tensor([[True, False]]), eta_s=torch.tensor([[.2, 0.]]),
        route_cost_points=torch.tensor([[.01, 0.]]),
        risk_cost_points=torch.zeros((1, 2)), route_batch=route,
        fallback_index=torch.tensor(0))
    table = CandidateTable("ep:1", ids,
        (tuple(None for _ in feature_names), tuple(None for _ in feature_names)),
        (True, True), feature_names)
    model = make_meco_track(track)
    raster = torch.zeros((1, 6, 16, 16))
    globals_ = torch.zeros((1, 7))

    result = score_candidates(track, model, raster=raster, globals_=globals_,
        table=table, candidate_batch=batch,
        route_cost=torch.zeros((1, 2)), risk_cost=torch.zeros((1, 2)))

    assert result["logits"].shape == (1, 2)
    assert torch.isfinite(result["logits"][0, 0])
    assert torch.isneginf(result["logits"][0, 1])


def test_c2_context_preserves_all_query_recurrent_state_and_tolerates_empty_support():
    torch.manual_seed(17)
    model = make_meco_track("C2").eval()
    raster = torch.randn((1, 6, 16, 16))
    globals_ = torch.randn((1, 7))
    hidden = torch.randn((1, 64))
    one = torch.randn((1, 1, 40))
    with_two = torch.cat((one, torch.randn((1, 1, 40))), dim=1)
    single_output = model(raster, globals_, one, hidden)
    masked_output = model(raster, globals_, with_two, hidden,
                          candidate_mask=torch.tensor([[True, False]]))
    empty_output = model(raster, globals_, with_two, hidden,
                         candidate_mask=torch.tensor([[False, False]]))

    all_query_output = model(raster, globals_, with_two, hidden)
    torch.testing.assert_close(masked_output["hidden"], all_query_output["hidden"])
    torch.testing.assert_close(masked_output["candidate_utility"],
                               all_query_output["candidate_utility"])
    assert not torch.allclose(masked_output["hidden"], single_output["hidden"])
    assert torch.isfinite(empty_output["hidden"]).all()
    assert torch.isfinite(empty_output["candidate_utility"]).all()
    with pytest.raises(ValueError, match="candidate mask"):
        model(raster, globals_, with_two, hidden,
              candidate_mask=torch.ones((1, 2)))
