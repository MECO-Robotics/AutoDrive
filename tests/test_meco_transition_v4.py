"""Authored fixtures for the standalone V4 transition-row block codec.

Run only after the accepted G0 and reviewed contract bridge gates. These
fixtures require a CUDA/HIP device and do not exercise complete bundle bytes.
"""

import pytest
import torch

from frc_defense.meco_transition_v4 import (
    ROLLOUT_ROWS,
    TRANSITION_FIELDS,
    TRANSITION_ROWS_BYTES,
    pack_transition_rows_device,
    unpack_transition_rows_device,
)


_DTYPES = {
    "bool": torch.bool,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}


def _zero_fields(device):
    return [
        torch.zeros((ROLLOUT_ROWS, *shape), dtype=_DTYPES[scalar_type], device=device)
        for _, scalar_type, shape in TRANSITION_FIELDS
    ]


def test_transition_row_block_field_major_axis_order_and_raw_round_trip():
    device = torch.device("cuda")
    fields = _zero_fields(device)
    for index, tensor in enumerate(fields):
        if tensor.dtype == torch.bool:
            tensor.reshape(-1)[0] = True
        else:
            tensor.reshape(-1)[0] = index + 1

    fields[0] = torch.arange(ROLLOUT_ROWS, dtype=torch.int32, device=device).to(torch.uint16)
    fields[1] = torch.arange(ROLLOUT_ROWS, dtype=torch.uint8, device=device)
    fields[11][::2] = True

    raster_index = next(
        index for index, (name, _, _) in enumerate(TRANSITION_FIELDS)
        if name == "observation_raster"
    )
    raster_shape = (ROLLOUT_ROWS, *TRANSITION_FIELDS[raster_index][2])
    fields[raster_index] = torch.arange(
        raster_shape[0] * raster_shape[1] * raster_shape[2] * raster_shape[3],
        dtype=torch.float32,
        device=device,
    ).reshape(raster_shape)
    fields[raster_index][0, 0, 0, 0] = -0.0

    packed = pack_transition_rows_device(fields)
    assert packed.device == device
    assert packed.dtype == torch.uint8
    assert tuple(packed.shape) == (TRANSITION_ROWS_BYTES,)

    expected_block = torch.cat([
        tensor.contiguous().view(torch.uint8).reshape(-1)
        for tensor in fields
    ])
    assert torch.equal(packed, expected_block)

    first_field_bytes = fields[0].contiguous().view(torch.uint8).reshape(-1)
    second_field_bytes = fields[1].contiguous().view(torch.uint8).reshape(-1)
    assert torch.equal(packed[:64], first_field_bytes)
    assert torch.equal(packed[64:96], second_field_bytes)

    # The 94-byte identity group is repeated for all 32 rows before the
    # observation group. The raster then follows row order and C-order axes.
    raster_start = 94 * ROLLOUT_ROWS
    raster_bytes = fields[raster_index].contiguous().view(torch.uint8).reshape(-1)
    raster_end = raster_start + raster_bytes.numel()
    assert torch.equal(packed[raster_start:raster_end], raster_bytes)

    decoded = unpack_transition_rows_device(packed)
    assert len(decoded) == len(fields)
    for expected, actual in zip(fields, decoded):
        assert actual.device == device
        assert actual.dtype == expected.dtype
        assert tuple(actual.shape) == tuple(expected.shape)
        assert torch.equal(
            actual.contiguous().view(torch.uint8).reshape(-1),
            expected.contiguous().view(torch.uint8).reshape(-1),
        )


def test_transition_row_block_rejects_wrong_shape_and_dtype():
    device = torch.device("cuda")
    fields = _zero_fields(device)

    wrong_shape = list(fields)
    wrong_shape[0] = fields[0][:-1]
    with pytest.raises(ValueError, match="shape"):
        pack_transition_rows_device(wrong_shape)

    wrong_dtype = list(fields)
    wrong_dtype[0] = fields[0].to(torch.int32)
    with pytest.raises(ValueError, match="dtype"):
        pack_transition_rows_device(wrong_dtype)

    packed = pack_transition_rows_device(fields)
    with pytest.raises(ValueError, match="shape"):
        unpack_transition_rows_device(packed[:-1])
