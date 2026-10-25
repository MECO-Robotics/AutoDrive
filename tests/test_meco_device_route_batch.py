import pytest
import torch

from frc_defense.experimental_route_evaluation import (
    DeviceRouteBatch,
    DeviceRouteQuery,
)
from frc_defense.meco_candidates import DeviceCandidateBatch


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
