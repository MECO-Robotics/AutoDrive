import pytest
import math
import time
import torch
from dataclasses import replace
from torch.utils._python_dispatch import TorchDispatchMode

from frc_defense.experimental_route_evaluation import (
    DeviceRouteBatch,
    DeviceRouteQuery,
    MAX_ROUTE_QUERY_BATCH,
    evaluate_device_route_chunks,
    evaluate_device_routes,
    evaluate_routes,
    RouteQuery,
)
import frc_defense.meco_candidates as meco_candidates
from frc_defense.meco_candidates import (CandidateCommitment, DeviceCandidateBatch,
                                         _goal_nodes_batch)
from frc_defense.meco_pose_graph import build_legal_pose_graph
from frc_defense.meco_track_training import _d1_d2_scores, _snapshot_tensors
from types import SimpleNamespace
from frc_defense.learning_contract import CandidateTable
from frc_defense.meco_tracks import (EXTRA_FEATURES, MECO_CANDIDATE_MODEL_INPUT_WIDTH,
                                     SAFE_ACTION_FEATURES,
                                     MecoInferenceResponse,
                                     ProductionMecoAdapter, SafeActionCandidates,
                                     candidate_policy_logits,
                                     candidate_feature_tensor,
                                     build_safe_action_candidates,
                                     make_meco_track, sample_candidate_action,
                                     make_meco_inference_request, score_candidates)
from frc_defense.curriculum_scenarios import make_curriculum_env


class _NoHostScalarSync(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name == "aten::_local_scalar_dense":
            raise AssertionError("numeric device value synchronized to host before selection")
        return func(*args, **(kwargs or {}))


class _NoHostTensorProjection:
    def __enter__(self):
        self.originals = {name: getattr(torch.Tensor, name)
                          for name in ("cpu", "tolist", "numpy", "item")}
        def reject(*_args, **_kwargs):
            raise AssertionError("device numeric projection before action selection")
        for name in self.originals:
            setattr(torch.Tensor, name, reject)
        return self

    def __exit__(self, *_exc):
        for name, original in self.originals.items():
            setattr(torch.Tensor, name, original)


class _CudaContractTensor(torch.Tensor):
    """CPU-backed tensor exposing the CUDA device contract for adapter tests."""
    @property
    def device(self):
        return torch.device("cuda:0")


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


def test_c1_production_worker_accepts_device_mask_and_returns_model_response():
    track = "C1"
    ids = ("candidate:valid", "candidate:masked")
    features = tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)
    table = CandidateTable("worker:1", ids,
        tuple(tuple(None for _ in features) for _ in ids), (True, True), features)
    adapter = object.__new__(ProductionMecoAdapter)
    adapter.track = track
    adapter.device = torch.device("cuda:0")
    adapter.model = make_meco_track(track).eval()
    adapter.checkpoint_sha256 = "b" * 64
    adapter.protocol_id = "worker-contract"
    adapter._last_inference_exception_type = None
    adapter.next_request_identity = lambda: (1, "worker-generation")
    fake = lambda value: value.as_subclass(_CudaContractTensor)
    inputs = {
        "globals_": fake(torch.zeros((1, 7))),
        "candidate_features": fake(torch.zeros((1, 2, MECO_CANDIDATE_MODEL_INPUT_WIDTH))),
        "hidden": fake(torch.zeros((1, 64))),
    }
    candidate_batch = SimpleNamespace(candidate_ids=ids)
    capture = time.monotonic() - .01
    request = make_meco_inference_request(adapter, table=table,
        observation_timestamp=1., capture_monotonic=capture,
        inputs=inputs, candidate_batch=candidate_batch)
    mask = fake(torch.tensor([[True, False]]))

    response = adapter._infer_sync(request,
        {**inputs, "__candidate_valid_mask": mask}, now=time.monotonic())

    assert response is not None, adapter._last_inference_exception_type
    assert response.track == "C1"
    assert response.candidate_mask_digest == request.candidate_mask_digest
    assert response.outputs["candidate_utility"].shape == (1, 2)

    bad_inputs = dict(inputs)
    bad_inputs["globals_"] = fake(torch.full((1, 7), float("nan")))
    bad_request = make_meco_inference_request(adapter, table=table,
        observation_timestamp=1., capture_monotonic=capture,
        inputs=bad_inputs, candidate_batch=candidate_batch)
    guarded = adapter._infer_sync(bad_request,
        {**bad_inputs, "__candidate_valid_mask": mask}, now=time.monotonic())
    assert guarded is not None
    assert not guarded.outputs["__inference_valid"].item()
    assert torch.equal(guarded.outputs["candidate_utility"],
                       torch.zeros_like(guarded.outputs["candidate_utility"]))


def test_plan_a_worker_rejects_nonfinite_input_before_model_call():
    track = "A1"
    adapter = object.__new__(ProductionMecoAdapter)
    adapter.track = track
    adapter.device = torch.device("cuda:0")
    adapter.checkpoint_sha256 = "e" * 64
    adapter.protocol_id = "plan-a-input-guard"
    adapter.next_request_identity = lambda: (1, "plan-a-input-generation")
    called = []
    adapter.model = lambda **_inputs: called.append(True)
    fake = lambda value: value.as_subclass(_CudaContractTensor)
    raster = torch.zeros((1, 6, 16, 16))
    raster[0, 0, 0, 0] = float("nan")
    inputs = {"raster": fake(raster), "globals_": fake(torch.zeros((1, 7))),
              "hidden": fake(torch.zeros((1, 64)))}
    features = tuple(SAFE_ACTION_FEATURES)
    table = CandidateTable("plan-a-input-guard:1", ("pose:0",),
        (tuple(None for _ in features),), (True,), features)
    request = make_meco_inference_request(adapter, table=table,
        observation_timestamp=1., capture_monotonic=time.monotonic() - .01,
        inputs=inputs, legal_pose_order_digest="f" * 64)
    result = adapter._infer_sync(request, inputs, now=time.monotonic())
    assert result is None
    assert not called


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


def test_invalid_numeric_adapter_response_is_stochastic_one_hot_fallback(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 100.)
    track = "C1"
    ids = ("fallback", "invalid", "cheap")
    names = tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)
    rows = tuple(tuple(0. for _ in names) for _ in ids)
    table = CandidateTable("guarded-sampler:1", ids, rows,
                           (False, True, True), names)
    raw = torch.zeros((1, len(ids), len(names)))
    raw[0, :, 0] = torch.arange(len(ids), dtype=raw.dtype)
    valid_mask = torch.tensor([[False, True, True]])
    candidate_batch = SimpleNamespace(candidate_ids=ids, feature_names=names,
        raw_features=raw, missing_mask=torch.zeros_like(raw),
        candidate_features=torch.cat((raw, torch.zeros_like(raw)), dim=-1),
        valid_mask=valid_mask, fallback_index=torch.tensor(1))
    adapter = object.__new__(ProductionMecoAdapter)
    adapter.track = track
    adapter.checkpoint_sha256 = "d" * 64
    adapter.protocol_id = "guarded-sampler"
    adapter._request_sequence = 0
    adapter._generation_id = "guarded-sampler-generation"
    adapter.next_request_identity = lambda: (1, adapter._generation_id)
    adapter._broker = SimpleNamespace(counters=lambda: {})
    submitted = {}
    adapter.submit_inference = lambda request, inputs: submitted.update(
        request=request, inputs=inputs) or object()

    def invalid_response(request, _handle):
        completed = max(time.monotonic(), request.dispatch_monotonic)
        return MecoInferenceResponse(
            schema="meco-inference-v2", track=track,
            checkpoint_sha256=request.checkpoint_sha256,
            protocol_id=request.protocol_id, model_id=request.model_id,
            observation_timestamp=request.observation_timestamp,
            capture_monotonic=request.capture_monotonic,
            dispatch_monotonic=request.dispatch_monotonic,
            completed_monotonic=completed,
            expires_monotonic=request.expires_monotonic,
            request_sequence=request.request_sequence,
            generation_id=request.generation_id, snapshot_id=request.snapshot_id,
            candidate_mask_digest=request.candidate_mask_digest,
            legal_pose_order_digest=request.legal_pose_order_digest,
            input_contract_sha256=request.input_contract_sha256,
            outputs={"candidate_utility": torch.zeros((1, len(ids))),
                "uncertainty": torch.zeros((1, len(ids))),
                "hidden": torch.zeros((1, 64)),
                "__inference_valid": torch.tensor(False)})
    adapter.poll_inference = invalid_response

    decision = score_candidates(track, adapter,
        raster=torch.zeros((1, 6, 16, 16)), globals_=torch.zeros((1, 7)),
        table=table, candidate_batch=candidate_batch,
        route_cost=torch.tensor([[0., 4., 0.]]),
        risk_cost=torch.zeros((1, 3)), observation_timestamp=1.,
        capture_monotonic=time.monotonic() - .01)
    assert decision["inference_status"] == "device_outcome"
    assert decision["inference_status_codes"]["deterministic_fallback"] == 1
    assert decision["inference_status_code"].item() == 1
    assert torch.isneginf(decision["logits"][0, 0])
    assert torch.isfinite(decision["logits"][0, 1])
    assert torch.isneginf(decision["logits"][0, 2])
    sampled = [sample_candidate_action(decision["logits"], deterministic=False)[0].item()
               for _ in range(32)]
    assert sampled == [1] * 32

    # An all-infeasible batch may carry index zero as a structural placeholder.
    # It must not become a finite one-hot fallback under a numeric model fault.
    no_feasible_batch = SimpleNamespace(**{
        **candidate_batch.__dict__,
        "valid_mask": torch.zeros_like(valid_mask),
        "fallback_index": torch.tensor(0),
    })
    def valid_response(request, handle):
        response = invalid_response(request, handle)
        return replace(response, outputs={**response.outputs,
                                          "__inference_valid": torch.tensor(True)})
    adapter.poll_inference = valid_response
    no_feasible = score_candidates(track, adapter,
        raster=torch.zeros((1, 6, 16, 16)), globals_=torch.zeros((1, 7)),
        table=table, candidate_batch=no_feasible_batch,
        route_cost=torch.zeros((1, 3)), risk_cost=torch.zeros((1, 3)),
        observation_timestamp=1., capture_monotonic=time.monotonic() - .01)
    assert no_feasible["inference_status_codes"]["no_feasible_candidate"] == 2
    assert no_feasible["inference_status_code"].item() == 2
    assert no_feasible["inference_status"] == "device_outcome"
    assert torch.isneginf(no_feasible["logits"]).all()
    with pytest.raises((AssertionError, RuntimeError),
                       match="finite candidate logit|each decision"):
        sample_candidate_action(no_feasible["logits"], deterministic=False)

    # On an absent response, keep the string label generic and use the typed
    # device code to distinguish deterministic fallback from empty support.
    adapter.poll_inference = lambda _request, _handle: None
    absent = score_candidates(track, adapter,
        raster=torch.zeros((1, 6, 16, 16)), globals_=torch.zeros((1, 7)),
        table=table, candidate_batch=no_feasible_batch,
        route_cost=torch.zeros((1, 3)), risk_cost=torch.zeros((1, 3)),
        observation_timestamp=1., capture_monotonic=time.monotonic() - .01)
    assert absent["inference_status"] == "device_outcome"
    assert absent["inference_status_code"].item() == 2
    assert torch.isneginf(absent["logits"]).all()


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
    model = make_meco_track(track).eval()
    with _NoHostTensorProjection(), _NoHostScalarSync():
        candidate_set = meco_candidates._build_device_candidate_set(env, base,
            SimpleNamespace(resolution_m=.5),
            [(index, 0, 0., 0., None) for index in range(count)] +
            [(7, 0, 0., 0., None)], {},
            candidate_k=count, action_ticks=1, prediction_mode="current",
            active_commitment=commitment, continue_index=count, oracle_capacity=count,
            learned_traversal_cost=None, learned_cost_scale=1.)
        result = score_candidates(track, model,
            raster=torch.zeros((1, 6, 16, 16)), globals_=torch.zeros((1, 7)),
            table=candidate_set.table, candidate_batch=candidate_set.device_batch,
            route_cost=candidate_set.route_cost_points[None],
            risk_cost=candidate_set.risk_cost_points[None])
        selected = sample_candidate_action(result["logits"], deterministic=True)
    assert candidate_set.device_batch is not None
    assert all(value is None for row in candidate_set.table.features for value in row)
    assert candidate_set.table.valid_mask == tuple(True for _ in range(candidate_count))
    assert candidate_set.device_batch.valid_mask.tolist() == [
        [True, False] + [True] * 6 + [False]]
    switch_column = candidate_set.table.feature_names.index("commitment_switch_cost")
    assert candidate_set.device_batch.raw_features[0, :, switch_column].tolist() == [
        0.] * candidate_count
    assert torch.isfinite(result["logits"][0, 0])
    assert torch.isneginf(result["logits"][0, 1])
    assert selected[0].shape == (1,)


@pytest.mark.parametrize("track", ("C1", "C2"))
@pytest.mark.parametrize("candidate_k", (8, 16, 32))
def test_c1_c2_device_candidate_scores_and_logprob_match_legacy_with_continue(
        track, candidate_k):
    torch.manual_seed(43)
    names = tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)
    count = candidate_k + 1
    ids = tuple([f"collect:track:{index}" for index in range(candidate_k)] +
                ["continue:collect:track:0"])
    raw = torch.randn((1, count, len(names)))
    raw[:, :, 0] = torch.arange(count, dtype=raw.dtype)
    raw[:, -1, names.index("is_continue")] = 1.
    valid = torch.ones((1, count), dtype=torch.bool)
    valid[0, 1] = False
    missing = torch.zeros_like(raw)
    confidence_column = names.index("track_confidence")
    opponent_gap_column = names.index("opponent_distance_m")
    raw[0, 2, confidence_column] = 0.
    raw[0, 5, opponent_gap_column] = 0.
    missing[0, 2, confidence_column] = 1.
    missing[0, 5, opponent_gap_column] = 1.
    rows = [list(row) for row in raw[0].tolist()]
    rows[2][confidence_column] = None
    rows[5][opponent_gap_column] = None
    rows = tuple(tuple(value for value in row) for row in rows)
    table = CandidateTable("parity:device", ids, rows,
        tuple(bool(value) for value in valid[0].tolist()), names)
    queries = tuple(DeviceRouteQuery(item, "parity:device", 0, 3) for item in ids)
    route = DeviceRouteBatch(queries, torch.full((count,), DeviceRouteBatch.FEASIBLE,
        dtype=torch.long), valid[0], torch.zeros((count, 1, 2)),
        torch.ones(count, dtype=torch.long), torch.zeros(count), torch.zeros(count),
        torch.zeros(count), torch.arange(count))
    route_cost = torch.linspace(0., .02, count)[None]
    risk_cost = torch.linspace(.1, .2, count)[None]
    batch = DeviceCandidateBatch(ids, names, torch.arange(count),
        torch.zeros(count, dtype=torch.long), torch.full((count,), -1),
        torch.zeros((count, 2)), torch.zeros((count, 2)), torch.zeros(count),
        torch.zeros((count, 2), dtype=torch.bool), raw, missing, valid,
        torch.zeros((1, count)), route_cost, risk_cost, route,
        torch.tensor(0))
    legacy_raw, legacy_missing = candidate_feature_tensor(table, device="cpu",
        dtype=raw.dtype)
    assert torch.equal(legacy_raw, batch.raw_features[0])
    assert torch.equal(legacy_missing, batch.missing_mask[0])
    device_raw, device_missing = candidate_feature_tensor(table, device="cpu",
        dtype=raw.dtype, raw_features=batch.raw_features[0],
        missing_mask=batch.missing_mask[0])
    assert torch.equal(device_raw, legacy_raw)
    assert torch.equal(device_missing, legacy_missing)
    model = make_meco_track(track).eval()
    raster = torch.randn((1, 6, 16, 16))
    globals_ = torch.randn((1, 7))
    hidden = torch.randn((1, 64))

    legacy = score_candidates(track, model, raster=raster, globals_=globals_,
        table=table, hidden=hidden, route_cost=route_cost, risk_cost=risk_cost)
    device = score_candidates(track, model, raster=raster, globals_=globals_,
        table=table, candidate_batch=batch, hidden=hidden,
        route_cost=route_cost, risk_cost=risk_cost)
    torch.testing.assert_close(device["logits"], legacy["logits"])
    torch.testing.assert_close(device["hidden"], legacy["hidden"])
    device_action, device_logprob, _ = sample_candidate_action(
        device["logits"], deterministic=True)
    legacy_action, legacy_logprob, _ = sample_candidate_action(
        legacy["logits"], deterministic=True)
    assert torch.equal(device_action, legacy_action)
    torch.testing.assert_close(device_logprob, legacy_logprob)

    def failed_adapter():
        adapter = object.__new__(ProductionMecoAdapter)
        adapter.track = track
        adapter.checkpoint_sha256 = "c" * 64
        adapter.protocol_id = "fallback-parity"
        adapter._request_sequence = 0
        adapter._generation_id = "fallback-generation"
        adapter.next_request_identity = lambda: (1, adapter._generation_id)
        adapter._broker = SimpleNamespace(counters=lambda: {})
        adapter.submit_inference = lambda _request, _inputs: object()
        adapter.poll_inference = lambda _request, _handle: None
        return adapter

    capture = time.monotonic() - .01
    fallback_legacy = score_candidates(track, failed_adapter(), raster=raster,
        globals_=globals_, table=table, hidden=hidden,
        observation_timestamp=1., capture_monotonic=capture)
    fallback_device = score_candidates(track, failed_adapter(), raster=raster,
        globals_=globals_, table=table, candidate_batch=batch, hidden=hidden,
        observation_timestamp=1., capture_monotonic=capture)
    assert fallback_legacy["inference_status"] == "deterministic_fallback"
    assert fallback_device["inference_status"] == "deterministic_fallback"
    torch.testing.assert_close(fallback_device["logits"], fallback_legacy["logits"])


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


def test_route_and_risk_cost_are_deducted_once_and_ties_choose_lowest_index():
    utility = torch.tensor([[1., 1., 9.]])
    valid = torch.tensor([[True, True, False]])
    route = torch.tensor([[.2, .2, 0.]])
    risk = torch.tensor([[.1, .1, 0.]])
    logits = candidate_policy_logits(utility, valid, route, risk)
    torch.testing.assert_close(logits, torch.tensor([[.7, .7, -torch.inf]]))
    selected, _, _ = sample_candidate_action(logits, deterministic=True)
    assert selected.item() == 0


@pytest.mark.parametrize("policy", ("D1", "D2"))
@pytest.mark.parametrize("count", (8, 16, 32))
def test_d1_d2_device_scoring_matches_legacy_rows_across_capacities(policy, count):
    names = [f"f{i}" for i in range(20)]
    names[0], names[10], names[11], names[12] = (
        "action_index", "nearby_sensed_track_count", "track_confidence", "track_age_s")
    ids = tuple([f"collect:track:{i}" for i in range(4)] +
                ["score", "intercept", "ferry", "hold"] +
                [f"collect:track:{i}" for i in range(4, count - 4)])
    rows = []
    for i in range(count):
        row = [0.] * len(names)
        row[0] = float(i if i < 8 else 0)
        row[10], row[11], row[12] = .5, .75, .25
        rows.append(tuple(row))
    raw = torch.tensor(rows, dtype=torch.float32)[None]
    valid = torch.ones((1, count), dtype=torch.bool)
    valid[0, -1] = False
    route_cost = torch.linspace(0., .03, count)
    goals = torch.stack((torch.linspace(.1, 1., count),
                         torch.full((count,), .25)), dim=-1)
    forecast_env = SimpleNamespace(
        opponent_pose=torch.zeros((1, 4, 3)),
        opponent_velocity=torch.tensor([[[0., 0., 0.], [0., 0., 0.],
                                          [0., 0., 0.], [.8, .1, .4]]]),
        opponent_age=torch.zeros((1, 4)),
        opponent_valid=torch.tensor([[False, False, False, True]]))
    forecast, forecast_valid, fresh, risk_cost = (
        meco_candidates._forecast_risk_batch_device(forecast_env, goals,
            torch.ones(count), valid[0], "multi_model_v1"))
    assert forecast_valid.any() and fresh.all() and risk_cost.sum() > 0.
    queries = tuple(DeviceRouteQuery(item, "parity:1", 0, 3) for item in ids)
    route = DeviceRouteBatch(queries, torch.full((count,), 3, dtype=torch.long), valid[0],
        torch.zeros((count, 1, 2)), torch.ones(count, dtype=torch.long),
        torch.zeros(count), torch.zeros(count), torch.zeros(count),
        torch.arange(count))
    batch = DeviceCandidateBatch(ids, tuple(names), torch.arange(count),
        torch.tensor([int(row[0]) for row in rows]),
        torch.full((count,), -1), torch.zeros((count, 2)), goals,
        torch.zeros(count), torch.zeros((count, 2), dtype=torch.bool), raw,
        torch.zeros_like(raw), valid, torch.zeros((1, count)), route_cost[None],
        risk_cost[None], route, torch.tensor(0), forecast_xy=forecast[None],
        forecast_valid=forecast_valid[None])
    table = CandidateTable("parity:1", ids, tuple(rows),
        tuple(bool(value) for value in valid[0].tolist()), tuple(names))
    candidate_set = SimpleNamespace(device_batch=batch, table=table,
        route_cost_points=route_cost, risk_cost_points=risk_cost)

    device_scores = _d1_d2_scores(candidate_set, None, policy,
                                  route_cost_points_per_second=10.)
    legacy_scores = _d1_d2_scores(SimpleNamespace(device_batch=None,
        table=table, route_cost_points=route_cost, risk_cost_points=risk_cost),
        SimpleNamespace(device="cpu"), policy, route_cost_points_per_second=10.)

    torch.testing.assert_close(device_scores, legacy_scores)
    assert device_scores.argmax(-1).item() == legacy_scores.argmax(-1).item()
    assert torch.isneginf(device_scores[0, -1])


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


def test_device_route_evaluator_matches_legacy_for_obstacle_and_invalid_rows():
    env = make_curriculum_env("pickup_simple", "single", 1, "cpu", 71,
                              horizon=10, fuel_physics=False)
    env.set_focal_pose_estimate((10., 4., 0.))
    obstacles = env._perceived_robot_obstacles()[3]
    blocked_goal = (obstacles[0, :2].clone() if obstacles.shape[0]
                    else torch.tensor((10., 4.)))
    queries = tuple(RouteQuery(item, "route-parity:1", world, 3, goal)
        for item, world, goal in (
        ("clear", 0, (10.8, 4.)),
        ("at-goal-clear", 0, (10., 4.)),
        ("unreachable-goal", 0, tuple(blocked_goal.tolist())),
        ("footprint-collision", 0, (.1, .1)),
        ("endpoint-mismatch", 0, (10.73, 4.19)),
        ("duplicate", 0, (11., 4.)),
        ("duplicate", 0, (11.2, 4.)),
        ("invalid-world", 9, (11., 4.)),
        ("invalid-goal-nan", 0, (float("nan"), 4.)),
        ("invalid-goal-inf", 0, (float("inf"), 4.))))
    device_queries = tuple(DeviceRouteQuery(query.candidate_id,
        query.snapshot_id, query.world, query.slot) for query in queries)
    goals = torch.tensor([query.goal_xy for query in queries], dtype=env.policy_pose.dtype)
    learned_cost = torch.arange(env.planner.nx, dtype=env.policy_pose.dtype)[:, None]
    learned_cost = learned_cost.expand(env.planner.nx, env.planner.ny).clone() * .05

    # This guard spans the real evaluator, including TensorADStar planning,
    # endpoint/collision checks, and cost integration. Projection is allowed
    # only after the decision-valued tensors have left that guarded section.
    with _NoHostTensorProjection(), _NoHostScalarSync():
        batch = evaluate_device_routes(env, device_queries, goals,
            learned_traversal_cost=learned_cost,
            learned_cost_provenance="projection-guard-contract",
            endpoint_tolerance_m=.01)
    device = batch.project_results()
    legacy = evaluate_routes(env, queries, learned_traversal_cost=learned_cost,
        learned_cost_provenance="projection-guard-contract",
        endpoint_tolerance_m=.01)

    assert [result.status for result in device] == [result.status for result in legacy]
    assert [result.feasible for result in device] == [result.feasible for result in legacy]
    for index, (left, right) in enumerate(zip(device, legacy)):
        if left.distance_m is not None or right.distance_m is not None:
            assert left.distance_m == pytest.approx(right.distance_m, abs=1e-6)
        if left.path_points_m is not None and right.path_points_m is not None:
            assert len(left.path_points_m) == len(right.path_points_m)
            for left_point, right_point in zip(left.path_points_m, right.path_points_m):
                assert left_point == pytest.approx(right_point, abs=1e-6)
        if left.weighted_cost is not None or right.weighted_cost is not None:
            assert left.weighted_cost is not None and right.weighted_cost is not None
            assert left.weighted_cost == pytest.approx(right.weighted_cost, abs=1e-6)
        assert left.cost_provenance.startswith("projection-guard-contract")
        assert right.cost_provenance.startswith("projection-guard-contract")
        assert left.cost_units == right.cost_units
        if right.feasible and right.status == "feasible":
            endpoint = right.path_points_m[-1]
            goal = queries[index].goal_xy
            expected_error = math.dist(endpoint, goal)
            assert batch.endpoint_error_m[index].item() == pytest.approx(
                expected_error, abs=1e-6)
    assert device[0].status == "feasible" and device[0].feasible
    assert device[1].status == "at_goal" and device[1].feasible
    assert device[2].status == "no_path" and not device[2].feasible
    assert device[3].status == "no_path" and not device[3].feasible
    assert batch.endpoint_error_m[4].item() > .01
    assert batch.path_xy.shape[1] == env.planner.max_points + 1
    clear_path = batch.path_xy[0, :batch.path_length[0]]
    grid_cells = torch.floor(clear_path / env.planner.resolution).to(torch.long)
    assert (grid_cells[1:] != grid_cells[:-1]).any(), "route must cross a raster boundary"

    # Also check bump-only path integration without a learned raster.
    with _NoHostTensorProjection(), _NoHostScalarSync():
        bump_batch = evaluate_device_routes(env, device_queries[:2], goals[:2])
    bump_device = bump_batch.project_results()
    bump_legacy = evaluate_routes(env, queries[:2])
    for left, right in zip(bump_device, bump_legacy):
        assert left.status == right.status
        assert left.feasible == right.feasible
        if left.weighted_cost is not None and right.weighted_cost is not None:
            assert left.weighted_cost == pytest.approx(right.weighted_cost, abs=1e-6)

    # Target the center of a known bump so this fixture proves positive bump
    # cost, instead of merely comparing two zero-bump routes.
    bump_center = env.planner._bumps[0, :2].clone()
    bump_goal_query = DeviceRouteQuery("bump-center", "route-parity:bump", 0, 3)
    with _NoHostTensorProjection(), _NoHostScalarSync():
        bump_center_batch = evaluate_device_routes(env, (bump_goal_query,),
            bump_center[None])
    bump_path = bump_center_batch.path_xy[0, :bump_center_batch.path_length[0]]
    bump_cells = torch.floor(bump_path / env.planner.resolution).to(torch.long)
    bump_cost = env.planner._bump_cost[bump_cells[:, 0], bump_cells[:, 1]]
    assert (bump_cost > 1.).any(), "route must include a positive bump contribution"
    assert bump_center_batch.weighted_cost_grid_edges[0] > (
        bump_center_batch.distance_m[0] / env.planner.resolution)

    # A synthetic observed circle over the robot start exercises the planner's
    # blocked-start/escape contract and explicit endpoint collision checks.
    obstacles = torch.zeros((6, 1, 3), dtype=env.policy_pose.dtype)
    obstacles[3, 0, :2] = env.policy_pose[0, 3, :2]
    obstacles[3, 0, 2] = .25
    original_obstacles = env._perceived_robot_obstacles
    env._perceived_robot_obstacles = lambda: obstacles
    escape_query = RouteQuery("blocked-start-escape", "route-parity:escape",
                              0, 3, (11.2, 4.))
    escape_device_query = DeviceRouteQuery(escape_query.candidate_id,
        escape_query.snapshot_id, escape_query.world, escape_query.slot)
    escape_goal = torch.tensor([escape_query.goal_xy], dtype=env.policy_pose.dtype)
    with _NoHostTensorProjection(), _NoHostScalarSync():
        escape_batch = evaluate_device_routes(env, (escape_device_query,), escape_goal)
    escape_device, = escape_batch.project_results()
    escape_legacy, = evaluate_routes(env, (escape_query,))
    env._perceived_robot_obstacles = original_obstacles
    assert escape_device.status == escape_legacy.status
    assert escape_device.feasible == escape_legacy.feasible
    assert escape_device.path_points_m == escape_legacy.path_points_m
    assert escape_device.weighted_cost == escape_legacy.weighted_cost
    at_goal_blocked = RouteQuery("at-goal-blocked", "route-parity:escape",
                                 0, 3, tuple(env.policy_pose[0, 3, :2].tolist()))
    at_goal_blocked_device_query = DeviceRouteQuery(at_goal_blocked.candidate_id,
        at_goal_blocked.snapshot_id, at_goal_blocked.world, at_goal_blocked.slot)
    at_goal_blocked_xy = torch.tensor([at_goal_blocked.goal_xy],
                                      dtype=env.policy_pose.dtype)
    with _NoHostTensorProjection(), _NoHostScalarSync():
        blocked_goal_batch = evaluate_device_routes(env,
            (at_goal_blocked_device_query,), at_goal_blocked_xy)
    blocked_goal_device, = blocked_goal_batch.project_results()
    blocked_goal_legacy, = evaluate_routes(env, (at_goal_blocked,))
    assert blocked_goal_device.status == blocked_goal_legacy.status == "no_path"
    assert not blocked_goal_device.feasible and not blocked_goal_legacy.feasible
    env._perceived_robot_obstacles = original_obstacles

    # A full-height collider disconnects two field regions. This asserts a
    # genuine unreachable case rather than a goal that is itself occupied.
    original_colliders = env.sim.field_colliders
    barrier = torch.tensor([11.5, env.sim.field_width * .5, .12,
                            env.sim.field_width * .5 + 1.],
                           dtype=env.policy_pose.dtype)
    try:
        env.sim.field_colliders = list(original_colliders) + [barrier]
        unreachable_query = RouteQuery("disconnected", "route-parity:disconnected",
                                       0, 3, (12.4, 4.))
        unreachable_device_query = DeviceRouteQuery(unreachable_query.candidate_id,
            unreachable_query.snapshot_id, unreachable_query.world, unreachable_query.slot)
        unreachable_goal = torch.tensor([unreachable_query.goal_xy],
                                        dtype=env.policy_pose.dtype)
        with _NoHostTensorProjection(), _NoHostScalarSync():
            unreachable_batch = evaluate_device_routes(env,
                (unreachable_device_query,), unreachable_goal)
        unreachable_device, = unreachable_batch.project_results()
        unreachable_legacy, = evaluate_routes(env, (unreachable_query,))
        assert unreachable_device.status == unreachable_legacy.status == "no_path"
        assert not unreachable_device.feasible and not unreachable_legacy.feasible
    finally:
        env.sim.field_colliders = original_colliders

    # A straight, grid-aligned, reachable route uses every actual planner path
    # slot. A supplied zero learned map disables shortcutting on both paths.
    original_max_points = env.planner.max_points
    try:
        env.planner.max_points = 4
        start_cell = torch.floor(env.policy_pose[0, 3, :2] /
                                 env.planner.resolution).to(torch.long)
        max_goal_xy = ((int(start_cell[0].item()) + env.planner.max_points - 1 + .5)
                       * env.planner.resolution,
                       (int(start_cell[1].item()) + .5) * env.planner.resolution)
        max_query = DeviceRouteQuery("max-length", "route-parity:max", 0, 3)
        max_goal = torch.tensor([max_goal_xy], dtype=env.policy_pose.dtype)
        zero_learned_cost = torch.zeros((env.planner.nx, env.planner.ny),
                                        dtype=env.policy_pose.dtype)
        with _NoHostTensorProjection(), _NoHostScalarSync():
            max_batch = evaluate_device_routes(env, (max_query,), max_goal,
                                               learned_traversal_cost=zero_learned_cost)
        max_device, = max_batch.project_results()
        max_legacy_query = RouteQuery("max-length", "route-parity:max", 0, 3,
                                      max_goal_xy)
        max_legacy, = evaluate_routes(env, (max_legacy_query,),
                                      learned_traversal_cost=zero_learned_cost)
        assert max_batch.path_length[0].item() == env.planner.max_points + 1
        assert max_device.status == max_legacy.status == "feasible"
        assert max_device.feasible and max_legacy.feasible
        assert max_device.path_points_m == max_legacy.path_points_m
        assert max_device.weighted_cost == pytest.approx(max_legacy.weighted_cost,
                                                         abs=1e-6)
    finally:
        env.planner.max_points = original_max_points

    # Nonfinite robot state is rejected identically before route selection.
    original_x = env.policy_pose[0, 3, 0].clone()
    for nonfinite_x in (float("nan"), float("inf")):
        env.policy_pose[0, 3, 0] = nonfinite_x
        state_query = RouteQuery("nonfinite-start", "route-parity:state",
                                 0, 3, (11.2, 4.))
        state_device_query = DeviceRouteQuery(state_query.candidate_id,
            state_query.snapshot_id, state_query.world, state_query.slot)
        state_goal = torch.tensor([state_query.goal_xy], dtype=env.policy_pose.dtype)
        with _NoHostTensorProjection(), _NoHostScalarSync():
            state_batch = evaluate_device_routes(env, (state_device_query,), state_goal)
        state_device, = state_batch.project_results()
        state_legacy, = evaluate_routes(env, (state_query,))
        assert state_device.status == state_legacy.status == "invalid"
        assert not state_device.feasible and not state_legacy.feasible
    env.policy_pose[0, 3, 0] = original_x


def test_live_route_candidate_scorer_and_sampler_path_has_no_projection():
    """Actual route/candidate/model/sampler path; execution is E2E-gated."""
    env = make_curriculum_env("pickup_simple", "single", 1, "cpu", 73,
                              horizon=10, fuel_physics=False)
    env.set_focal_pose_estimate((10., 4., 0.))
    observation = env.observe()
    safe = build_safe_action_candidates(env, observation, world=0, slot=3,
                                        episode_id="live-device-route")
    raster, globals_ = _snapshot_tensors(safe.observation, device=env.device)
    graph = build_legal_pose_graph(env, slot=3)
    target_rows = {int(safe.table.features[i][1]): safe.table.features[i]
                   for i in range(4) if safe.table.valid_mask[i]
                   and safe.table.features[i][1] is not None}
    specs = [(action, 0, 0., 0.,
              (int(safe.table.features[action][1])
               if action < 4 and safe.table.features[action][1] is not None else None))
             for action in range(8)]
    with _NoHostTensorProjection(), _NoHostScalarSync():
        candidate_set = meco_candidates._build_device_candidate_set(env, safe,
            graph, specs, target_rows, candidate_k=8, action_ticks=1,
            prediction_mode="current", active_commitment=None,
            continue_index=None, oracle_capacity=len(specs),
            learned_traversal_cost=None, learned_cost_scale=1.)
        result = score_candidates("C1", make_meco_track("C1").eval(),
            raster=raster, globals_=globals_,
            table=candidate_set.table,
            candidate_batch=candidate_set.device_batch,
            route_cost=candidate_set.route_cost_points[None],
            risk_cost=candidate_set.risk_cost_points[None])
        selected, logprob, _ = sample_candidate_action(result["logits"],
                                                         deterministic=True)
    assert candidate_set.device_batch is not None
    assert selected.shape == logprob.shape == (1,)


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
