"""Empty robot selections must bypass footprint kernels and chunk reduction."""
import pytest
import torch

from frc_defense.tensor_adstar import TensorADStar


@pytest.mark.parametrize("point_count", [2, 5])
@pytest.mark.parametrize("box_count", [0, 2])
def test_empty_path_batch_returns_empty_boolean(point_count, box_count):
    planner = TensorADStar.__new__(TensorADStar)
    planner._boxes = torch.zeros(box_count, 4)
    # Exercise dispatch eligibility without allocating or launching on a GPU.
    planner.device = torch.device("cuda")
    path = torch.empty(0, point_count, 2)
    footprint = torch.empty(0)
    result = planner._footprint_path_clear(path, footprint, footprint, footprint)
    assert result.shape == (0,)
    assert result.dtype == torch.bool
    assert result.device == path.device
