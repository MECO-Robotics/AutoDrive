"""Synthetic contracts for fixed-capacity Plan A graph tensors.

These tests are authored as software evidence. The graph/simulator behavior
suite remains deferred until the simulator and current-dashboard gate is open.
"""
import torch

from frc_defense.meco_pose_graph import (
    HEADING_BINS,
    _connected_region_ids,
    compact_legal_pose_mask,
)


def test_compaction_matches_legacy_boolean_index_order_and_has_fixed_capacity():
    legal = torch.zeros((HEADING_BINS, 3, 4), dtype=torch.bool)
    legal[0, 0, 1] = True
    legal[0, 2, 3] = True
    legal[3, 1, 2] = True
    legal[7, 0, 0] = True

    permutation, inverse, node_valid = compact_legal_pose_mask(legal)
    legacy_original_ids = torch.arange(legal.numel())[legal.reshape(-1)]

    assert permutation.shape == (legal.numel(),)
    assert node_valid.shape == (legal.numel(),)
    assert torch.equal(permutation[:4], legacy_original_ids)
    assert torch.equal(node_valid, torch.arange(legal.numel()) < 4)
    assert torch.equal(inverse[legacy_original_ids], torch.arange(4))


def test_fixed_capacity_region_reference_ignores_padding_and_preserves_fifo_order():
    # Two directed components: 0->1->2 and 3->4. Padded rows have adversarial
    # in-range edges, but are invalid sources and may not create components.
    edges = [[1, 0], [2, 0], [0, 0], [4, 0], [3, 0], [0, 1], [4, 5]]
    masks = [[True, False], [True, False], [False, False], [True, False],
             [False, False], [True, True], [True, True]]
    valid = [True, True, True, True, True, False, False]

    regions = _connected_region_ids(edges, masks, torch.zeros((7, 2)), 4,
                                    node_valid=valid)

    assert regions == [0, 0, 1, 2, 3, 0, 0]


def test_region_reference_rejects_more_valid_components_than_capacity():
    edges = [[0], [1], [2], [3], [4]]
    masks = [[False], [False], [False], [False], [False]]

    try:
        _connected_region_ids(edges, masks, torch.zeros((5, 2)), 4,
                              node_valid=[True, True, True, True, True])
    except ValueError as exc:
        assert "components" in str(exc)
    else:
        raise AssertionError("component overflow must fail closed")
