import pytest
import torch

from frc_defense.experimental_route_evaluation import (
    DeviceRouteBatch,
    DeviceRouteQuery,
)


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
