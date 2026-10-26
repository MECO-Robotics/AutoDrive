"""V4 schema declarations and the standalone transition-row-block codec.

The codec covers only the fixed 32-row transition field block. It does not
assemble or parse a complete bundle, inspect event semantics, or compute a
digest. All tensor data transformations remain on the tensor's CUDA/HIP
device; callers must keep the returned byte block device-resident.
"""

ABI_NAME = "FiveTrackPpoTransitionV4"
ENVELOPE_NAME = "FiveTrackPpoBatchV4"
PROFILE_ID = "meco-five-track-shared-command-v4"
ENVELOPE_VERSION = 2
SCHEMA_VERSION = 4
PROFILE_DIGEST_HEX = "0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6"
RECONCILED_PROFILE_SHA256 = "5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3"

WORLD_CAPACITY = 1
ROBOT_SLOTS = 6
ROLLOUT_ROWS = 32
MACRO_TICKS = 12
ROUTE_QUERIES = 9681
PLAN_C_ROWS = 537
OBSERVED_TARGETS = 504
PLAN_A_NODES = 12096
PLAN_A_SUPPORT_ROWS = 96768
PLAN_A_ANCESTOR_LEVELS = 14
PLAN_C_ROUTE_CHUNK_CAPACITY = 64
PLAN_C_ROUTE_CHUNKS = 152
PLAN_C_PATH_POINTS = 73
RECURRENT_WIDTH = 64
RNG_PROTOCOL_PHILOX_V2 = 2
ROUTE_COST_COMPONENT_BUMP_BIT = 0
ROUTE_COST_COMPONENT_LEARNED_BIT = 1
OBSERVATION_GLOBAL_ORDER = (
    "x", "y", "sin_heading", "cos_heading", "load",
    "signed_hub_timer", "match_remaining",
)

TRANSITION_ROW_BYTES = 9822906
TRANSITION_ROWS_BYTES = 314332992
STATIC_IDENTITY_BYTES = 123
SCORE_EVENT_CAPACITY = 2304
SCORE_EVENT_ROW_BYTES = 59
SCORE_EVENT_TABLE_BYTES = 135936
PAYLOAD_BYTES = 314469051
HEADER_BYTES = 96
MAX_BUNDLE_BYTES = 314469147

# (name, scalar type, byte width, byte offset). The V4 envelope header is
# packed little-endian, without alignment or padding.
HEADER_FIELDS = (
    ("magic", "bytes", 8, 0),
    ("envelope_version", "uint16", 2, 8),
    ("schema_version", "uint16", 2, 10),
    ("header_bytes", "uint32", 4, 12),
    ("payload_bytes", "uint64", 8, 16),
    ("row_count", "uint16", 2, 24),
    ("track_code", "uint8", 1, 26),
    ("reserved0", "uint8", 1, 27),
    ("profile_digest", "bytes", 32, 28),
    ("payload_digest", "bytes", 32, 60),
    ("reserved1", "bytes", 4, 92),
)

MAGIC = b"MECO4V4\0"
TRACK_CODES = (("A1", 1), ("A2", 2), ("A3", 3), ("C1", 4), ("C2", 5))

# Bundle-level static identity prefix, before transition row 0:
# (name, scalar type, shape, byte width, byte offset).
STATIC_IDENTITY_FIELDS = (
    ("source_digest", "bytes", (32,), 32, 0),
    ("config_digest", "bytes", (32,), 32, 32),
    ("checkpoint_digest", "bytes", (32,), 32, 64),
    ("rng_root_seed", "uint64", (), 8, 96),
    ("rng_segment_id", "uint64", (), 8, 104),
    ("world_uid", "uint64", (), 8, 112),
    ("rng_protocol_code", "uint16", (), 2, 120),
    ("route_cost_component_presence_mask", "uint8", (), 1, 122),
)

# (name, scalar type, shape). Columns repeat for exactly ROLLOUT_ROWS rows;
# within each group, fields appear in this exact packed order. Array axes are
# canonical and there is no implicit alignment or per-row padding.
TRANSITION_FIELDS = (
    ("schema_version", "uint16", ()),
    ("track_code", "uint8", ()),
    ("rollout_slot", "int64", ()),
    ("rollout_generation", "int64", ()),
    ("decision_ordinal", "int64", ()),
    ("world_slot", "int64", ()),
    ("robot_slot", "int64", ()),
    ("world_reset_generation", "int64", ()),
    ("episode_generation", "uint64", ()),
    ("observation_sequence", "uint64", ()),
    ("environment_instance_id", "uint64", (2,)),
    ("row_valid", "bool", ()),
    ("actor_valid", "bool", ()),
    ("critic_valid", "bool", ()),
    ("command_valid", "bool", ()),
    ("terminal", "bool", ()),
    ("truncated", "bool", ()),
    ("transition_finished", "bool", ()),
    ("bootstrap_continuation", "bool", ()),
    ("reset_mask", "bool", ()),
    ("recurrent_available", "bool", ()),
    ("route_active", "bool", ()),
    ("observation_raster", "float32", (6, 25, 51)),
    ("observation_globals", "float32", (7,)),
    ("recurrent_hidden_before", "float32", (64,)),
    ("recurrent_hidden_after", "float32", (64,)),
    ("candidate_raw", "float32", (537, 20)),
    ("candidate_missing", "bool", (537, 20)),
    ("candidate_model_input", "float32", (537, 40)),
    ("candidate_valid", "bool", (537,)),
    ("candidate_identity", "int64", (537,)),
    ("ordinary_row_map", "int64", (16,)),
    ("continuation_row_map", "int64", (1,)),
    ("candidate_model_input_mask", "bool", (537,)),
    ("candidate_selection_rank", "int64", (537,)),
    ("candidate_policy_eligible", "bool", (537,)),
    ("target_xy", "float32", (504, 2)),
    ("target_velocity", "float32", (504, 2)),
    ("target_age", "float32", (504,)),
    ("target_confidence", "float32", (504,)),
    ("target_valid", "bool", (504,)),
    ("target_handle", "int64", (504, 4)),
    ("support_valid", "bool", (96768,)),
    ("distance_m", "float64", (12096,)),
    ("route_cost", "float64", (12096,)),
    ("parent_edge_m", "float64", (12096,)),
    ("parent_node", "int64", (12096,)),
    ("traversal_order", "int64", (12096,)),
    ("ancestor_jumps", "int64", (14, 12096)),
    ("node_utility", "float32", (12096,)),
    ("behavior_logits", "float32", (5,)),
    ("query_kind", "uint8", (9681,)),
    ("policy_source_row", "int64", (9681,)),
    ("oracle_identity_code", "int64", (9681,)),
    ("route_variant", "int64", (9681,)),
    ("target_handle", "int64", (9681, 4)),
    ("goal_xy", "float32", (9681, 2)),
    ("query_active", "bool", (9681,)),
    ("status_code", "int64", (9681,)),
    ("feasible", "bool", (9681,)),
    ("path_xy", "float32", (9681, 73, 2)),
    ("path_length", "int64", (9681,)),
    ("distance_m", "float64", (9681,)),
    ("endpoint_error_m", "float64", (9681,)),
    ("weighted_cost_grid_edges", "float64", (9681,)),
    ("bump_cost_grid_edges", "float64", (9681,)),
    ("learned_cost_grid_edges", "float64", (9681,)),
    ("candidate_index", "int64", (9681,)),
    ("path_point_mask", "bool", (9681, 73)),
    ("schema_version", "uint16", ()),
    ("track_code", "uint8", ()),
    ("decision_valid", "bool", ()),
    ("policy_local_index", "int64", ()),
    ("source_global_row", "int64", ()),
    ("candidate_identity_code", "int64", ()),
    ("action_id", "int64", ()),
    ("goal_xy", "float32", (2,)),
    ("goal_heading", "float32", ()),
    ("target_identity", "int64", ()),
    ("route_override", "bool", ()),
    ("fallback", "bool", ()),
    ("route_status", "uint8", ()),
    ("route_cost", "float32", ()),
    ("risk_cost", "float32", ()),
    ("legal_node_id", "int64", ()),
    ("behavior_id", "int64", ()),
    ("descriptor_id", "int64", ()),
    ("selected_U", "float32", ()),
    ("selected_P", "float32", ()),
    ("reward", "float32", ()),
    ("old_log_probability", "float32", ()),
    ("entropy", "float32", ()),
    ("value", "float32", ()),
    ("old_value", "float32", ()),
    ("return", "float32", ()),
    ("advantage", "float32", ()),
    ("action_rng_key", "uint32", (2,)),
    ("action_rng_counter_before", "uint32", (4,)),
    ("action_rng_counter_after", "uint32", (4,)),
    ("permutation_rng_key", "uint32", (2,)),
    ("permutation_rng_counter_before", "uint32", (4,)),
    ("permutation_rng_counter_after", "uint32", (4,)),
    ("input_digest", "uint8", (32,)),
    ("support_digest", "uint8", (32,)),
    ("transition_digest", "uint8", (32,)),
    ("validation_status", "uint16", ()),
    ("physics_tick_ordinal", "uint64", ()),
    ("update_id", "uint64", ()),
    ("epoch_id", "uint32", ()),
    ("sample_order_id", "uint32", ()),
    ("tick_valid", "bool", (12,)),
    ("executed_action", "int64", (12,)),
    ("remasked", "bool", (12,)),
    ("target_lost", "bool", (12,)),
    ("preempted", "bool", (12,)),
    ("fallback", "bool", (12,)),
    ("execution_valid", "bool", (12,)),
    ("macro_reward", "float32", (12,)),
    ("gamepiece_outcome_flags", "uint16", (12,)),
    ("macro_tick_status", "uint16", (12,)),
)

# Half-open field-index ranges in TRANSITION_FIELDS. Duplicate names such as
# schema_version, track_code, and route_cost are intentional: each independently
# occurs in both the row identity and selected-command groups.
TRANSITION_FIELD_GROUPS = (
    ("identity_schema", 0, 22),
    ("observation", 22, 24),
    ("recurrent_state", 24, 26),
    ("plan_c_candidates", 26, 36),
    ("observed_targets", 36, 42),
    ("plan_a_support_search", 42, 51),
    ("plan_c_route_sink", 51, 69),
    ("selected_command", 69, 89),
    ("ppo_values", 89, 96),
    ("action_permutation_rng", 96, 102),
    ("provenance_replay", 102, 110),
    ("macro_trace", 110, 120),
)


_TORCH_DTYPES = {
    "bool": "bool",
    "float32": "float32",
    "float64": "float64",
    "int8": "int8",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
}

_SCALAR_BYTES = {
    "bool": 1,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
}


def _expected_field_shape(shape):
    return (ROLLOUT_ROWS, *shape)


def _field_byte_count(scalar_type, shape):
    count = 1
    for axis in _expected_field_shape(shape):
        count *= axis
    return count * _SCALAR_BYTES[scalar_type]


def _require_device_tensor(tensor, *, name):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must already reside on a CUDA/HIP device")


def pack_transition_rows_device(fields):
    """Pack the 32 transition rows into one field-major uint8 device block.

    `fields` is a sequence with one tensor for each entry in
    `TRANSITION_FIELDS`, in declaration order. A sequence is required because
    the accepted schema intentionally repeats names in different groups. Each
    tensor shape is `(ROLLOUT_ROWS, *declared_shape)`. This operation preserves
    each field's raw bool and IEEE-754 bit patterns; it does not validate V4
    mask/finite-value semantics.
    """
    import sys
    import torch

    if sys.byteorder != "little":
        raise RuntimeError("V4 transition rows require a little-endian host/device")
    if len(fields) != len(TRANSITION_FIELDS):
        raise ValueError(
            f"expected {len(TRANSITION_FIELDS)} ordered field tensors; got {len(fields)}"
        )

    device = None
    byte_columns = []
    total_bytes = 0
    for index, ((name, scalar_type, shape), tensor) in enumerate(
            zip(TRANSITION_FIELDS, fields)):
        _require_device_tensor(tensor, name=f"field[{index}] {name}")
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError("all transition fields must reside on the same device")
        expected_shape = _expected_field_shape(shape)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"field[{index}] {name} shape must be {expected_shape}; "
                f"got {tuple(tensor.shape)}"
            )
        expected_dtype = getattr(torch, _TORCH_DTYPES[scalar_type])
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"field[{index}] {name} dtype must be {expected_dtype}; got {tensor.dtype}"
            )

        field_bytes = tensor.contiguous().view(torch.uint8).reshape(-1)
        byte_columns.append(field_bytes)
        total_bytes += _field_byte_count(scalar_type, shape)

    if total_bytes != TRANSITION_ROWS_BYTES:
        raise RuntimeError(
            f"declared transition block is {total_bytes} bytes, "
            f"expected {TRANSITION_ROWS_BYTES}"
        )
    packed = torch.cat(byte_columns, dim=0).contiguous()
    if packed.dtype != torch.uint8 or packed.numel() != TRANSITION_ROWS_BYTES:
        raise RuntimeError("packed transition block has an invalid device byte layout")
    return packed


def unpack_transition_rows_device(byte_block):
    """Decode one exact transition-row block into an ordered tuple of tensors.

    `byte_block` must be a one-dimensional uint8 tensor on the same CUDA/HIP
    device that will consume the decoded fields. Each returned tensor matches
    the corresponding entry in `TRANSITION_FIELDS`; duplicate schema names
    remain distinct tuple positions. No header, identity prefix, event table,
    payload digest, or host-side numeric conversion is involved.
    """
    import torch

    _require_device_tensor(byte_block, name="byte_block")
    if byte_block.dtype != torch.uint8:
        raise ValueError("byte_block dtype must be torch.uint8")
    if tuple(byte_block.shape) != (TRANSITION_ROWS_BYTES,):
        raise ValueError(
            f"byte_block shape must be ({TRANSITION_ROWS_BYTES},); "
            f"got {tuple(byte_block.shape)}"
        )

    decoded = []
    offset = 0
    for index, (name, scalar_type, shape) in enumerate(TRANSITION_FIELDS):
        byte_count = _field_byte_count(scalar_type, shape)
        # Clone each packed column to device storage with offset zero before
        # dtype reinterpretation; packed fields have no alignment padding.
        field_bytes = byte_block.narrow(0, offset, byte_count).clone()
        dtype = getattr(torch, _TORCH_DTYPES[scalar_type])
        values = field_bytes.view(dtype).reshape(_expected_field_shape(shape))
        decoded.append(values)
        offset += byte_count

    if offset != TRANSITION_ROWS_BYTES:
        raise RuntimeError(
            f"decoded {offset} transition bytes, expected {TRANSITION_ROWS_BYTES}"
        )
    return tuple(decoded)
