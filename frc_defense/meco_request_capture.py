"""Opt-in bounded capture and strict loading of real MECO scorer requests.

Capture is enabled only when both MECO_REQUEST_CAPTURE_DIR and a positive
MECO_REQUEST_CAPTURE_MAX are set. Each accepted request is one atomic NPZ
artifact containing named CPU tensor copies and canonical provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import zipfile

import numpy as np
import torch


SCHEMA = "meco-production-request-tensors-v1"
_LOCK = threading.Lock()
_CAPTURED = 0


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _tensor_digest(name: str, value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(_canonical(list(value.shape)) + b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def validate_capture_inputs(metadata: dict, tensors, *, expected_track: str) -> None:
    """Bind export tensors to the exact, self-consistent capture metadata."""
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        raise ValueError("invalid production capture metadata")
    stored_metadata_digest = metadata.get("metadata_sha256")
    unhashed_metadata = dict(metadata)
    unhashed_metadata.pop("metadata_sha256", None)
    if (not isinstance(stored_metadata_digest, str)
            or hashlib.sha256(_canonical(unhashed_metadata)).hexdigest() != stored_metadata_digest):
        raise ValueError("production capture metadata digest mismatch")
    from .meco_input_provenance import validate_serialized_inputs
    from .meco_tracks import MECO_INPUT_NAMES
    names = list(MECO_INPUT_NAMES[expected_track])
    contract_digest = validate_serialized_inputs(expected_track, dict.fromkeys(names))
    if (metadata.get("track") != expected_track or metadata.get("input_names") != names
            or metadata.get("input_contract_sha256") != contract_digest
            or not isinstance(metadata.get("tensors"), list)
            or len(metadata["tensors"]) != len(names)
            or len(tensors) != len(names)):
        raise ValueError("production capture input inventory is invalid")
    integer_names = {"edges", "region_ids", "coarse_edges"}
    boolean_names = {"edge_mask", "coarse_edge_mask"}
    tensor_digests = []
    for index, (name, tensor, record) in enumerate(zip(names, tensors, metadata["tensors"])):
        expected_key = f"tensor_{index:02d}"
        if (not isinstance(record, dict) or record.get("name") != name
                or record.get("key") != expected_key or not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu" or not tensor.is_contiguous()):
            raise ValueError(f"production capture input record is invalid: {name}")
        array = tensor.detach().numpy()
        expected_dtype = ("<i8" if name in integer_names else
                          "|b1" if name in boolean_names else "<f4")
        digest = _tensor_digest(name, array)
        if (array.dtype.str != expected_dtype or list(array.shape) != record.get("shape")
                or array.dtype.str != record.get("dtype") or digest != record.get("sha256")):
            raise ValueError(f"ONNX export input does not match captured tensor: {name}")
        tensor_digests.append(digest)
    expected_content_digest = hashlib.sha256(_canonical(tensor_digests)).hexdigest()
    if expected_content_digest != metadata.get("content_sha256"):
        raise ValueError("production capture aggregate content digest mismatch")


def capture_request(request, inputs, *, source_snapshot_sha256: str = "") -> Path | None:
    """Persist one production request when explicit bounded capture is enabled."""
    global _CAPTURED
    directory = os.environ.get("MECO_REQUEST_CAPTURE_DIR")
    limit_raw = os.environ.get("MECO_REQUEST_CAPTURE_MAX")
    if not directory and not limit_raw:
        return None
    if not directory or not limit_raw:
        raise ValueError("production capture requires both MECO_REQUEST_CAPTURE_DIR and MECO_REQUEST_CAPTURE_MAX")
    if (len(source_snapshot_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in source_snapshot_sha256)):
        raise ValueError("production capture requires a checkpoint source-snapshot digest")
    try:
        limit = int(limit_raw)
    except ValueError as exc:
        raise ValueError("MECO_REQUEST_CAPTURE_MAX must be a positive integer") from exc
    if not 1 <= limit <= 100:
        raise ValueError("MECO_REQUEST_CAPTURE_MAX must be in [1, 100]")
    if tuple(inputs) != request.input_names:
        raise ValueError("capture input order does not match request")
    tensors = []
    total_bytes = 0
    integer_names = {"edges", "region_ids", "coarse_edges"}
    boolean_names = {"edge_mask", "coarse_edge_mask"}
    for name, shape in zip(request.input_names, request.input_shapes):
        tensor = inputs[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"capture input {name} is not a tensor")
        expected_dtype = (torch.int64 if name in integer_names else
                          torch.bool if name in boolean_names else torch.float32)
        if tensor.dtype != expected_dtype or tuple(tensor.shape) != shape:
            raise ValueError(f"capture input {name} has noncanonical shape or dtype")
        tensor_bytes = tensor.numel() * tensor.element_size()
        total_bytes += tensor_bytes
        if total_bytes > 64 * 1024 * 1024:
            raise ValueError("production request capture exceeds the 64 MiB tensor limit")
        tensors.append((name, tensor))
    if len(tensors) != len(request.input_names):
        raise ValueError("capture input count does not match request")
    with _LOCK:
        if _CAPTURED >= limit:
            return None
        _CAPTURED += 1
        ordinal = _CAPTURED
    arrays, tensor_records = {}, []
    for index, (name, tensor) in enumerate(tensors):
        array = tensor.detach().to(device="cpu").contiguous().numpy().copy()
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"capture input {name} contains nonfinite values")
        key = f"tensor_{index:02d}"
        arrays[key] = array
        tensor_records.append({"name": name, "key": key, "shape": list(array.shape),
                               "dtype": array.dtype.str,
                               "sha256": _tensor_digest(name, array)})
    if tuple(tuple(row["shape"]) for row in tensor_records) != request.input_shapes:
        raise ValueError("captured tensor shapes do not match request")
    metadata = {
        "schema": SCHEMA, "track": request.track, "request_schema": request.schema,
        "input_contract_sha256": request.input_contract_sha256,
        "input_names": list(request.input_names), "tensors": tensor_records,
        "checkpoint_sha256": request.checkpoint_sha256,
        "protocol_id": request.protocol_id,
        "source_snapshot_sha256": source_snapshot_sha256,
        "observation_schema": request.observation_schema,
        "observation_timestamp": request.observation_timestamp,
        "capture_monotonic": request.capture_monotonic,
        "dispatch_monotonic": request.dispatch_monotonic,
        "expires_monotonic": request.expires_monotonic,
        "snapshot_id": request.snapshot_id,
        "candidate_ids": list(request.candidate_ids),
        "valid_mask": list(request.valid_mask), "fallback_index": request.fallback_index,
        "candidate_mask_digest": request.candidate_mask_digest,
        "legal_pose_order_digest": request.legal_pose_order_digest,
        "request_sequence": request.request_sequence,
        "generation_id": request.generation_id, "model_id": request.model_id,
        "capture_ordinal": ordinal,
    }
    metadata["content_sha256"] = hashlib.sha256(_canonical(
        [record["sha256"] for record in tensor_records])).hexdigest()
    metadata["metadata_sha256"] = hashlib.sha256(_canonical(metadata)).hexdigest()
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = f"{request.track}-{request.generation_id}-{request.request_sequence:08d}.npz"
    destination = root / name
    fd, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, __metadata__=np.frombuffer(_canonical(metadata), dtype=np.uint8), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


def load_capture(path: Path, *, expected_track: str | None = None) -> tuple[dict, tuple[torch.Tensor, ...]]:
    """Strictly validate a capture artifact and return provenance and ordered tensors."""
    path = Path(path)
    # Inspect ZIP member declarations and NPY headers before asking NumPy to
    # allocate/decompress any tensor arrays.
    with zipfile.ZipFile(path) as zipped:
        members = zipped.infolist()
        if len(members) > 10:
            raise ValueError("capture archive has too many members")
        member_names = [member.filename for member in members]
        if len(member_names) != len(set(member_names)):
            raise ValueError("capture archive contains duplicate members")
        if "__metadata__.npy" not in member_names:
            raise ValueError("capture metadata is missing")
        metadata_member = zipped.getinfo("__metadata__.npy")
        if metadata_member.file_size > 1024 * 1024:
            raise ValueError("capture metadata exceeds the 1 MiB limit")
        declared = {}
        total_bytes = 0
        for member in members:
            if member.filename == "__metadata__.npy":
                continue
            if (not member.filename.endswith(".npy")
                    or member.file_size > 64 * 1024 * 1024 + 65536
                    or member.compress_size > 64 * 1024 * 1024 + 65536):
                raise ValueError("capture tensor member is malformed or oversized")
            key = member.filename[:-4]
            with zipped.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    read_header = np.lib.format.read_array_header_1_0
                elif version == (2, 0):
                    read_header = np.lib.format.read_array_header_2_0
                else:
                    raise ValueError("capture tensor uses an unsupported NPY header version")
                shape, fortran_order, dtype = read_header(
                    stream, max_header_size=10000)
                payload_offset = stream.tell()
            if fortran_order or not shape or any(type(size) is not int or size <= 0 for size in shape):
                raise ValueError("capture tensor header has invalid dimensions or memory order")
            byte_count = int(np.prod(shape, dtype=object)) * dtype.itemsize
            total_bytes += byte_count
            if byte_count != member.file_size - payload_offset or total_bytes > 64 * 1024 * 1024:
                raise ValueError("capture tensor declarations exceed the 64 MiB limit")
            declared[key] = (tuple(shape), dtype.str)
        with np.load(path, allow_pickle=False) as archive:
            if "__metadata__" not in archive.files:
                raise ValueError("capture metadata is missing")
            try:
                metadata = json.loads(archive["__metadata__"].tobytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("capture metadata is malformed") from exc
            if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
                raise ValueError("unsupported production capture schema")
            stored_digest = metadata.pop("metadata_sha256", None)
            if not isinstance(stored_digest, str) or hashlib.sha256(_canonical(metadata)).hexdigest() != stored_digest:
                raise ValueError("capture metadata digest mismatch")
            metadata["metadata_sha256"] = stored_digest
            from .meco_input_provenance import validate_serialized_inputs
            from .meco_tracks import MECO_INPUT_NAMES, MECO_TRACKS, MecoInferenceRequest
            track = metadata.get("track")
            if track not in MECO_TRACKS or (expected_track is not None and track != expected_track):
                raise ValueError("capture track mismatch")
            names = metadata.get("input_names")
            contract_digest = validate_serialized_inputs(track, dict.fromkeys(names or ()))
            if names != list(MECO_INPUT_NAMES[track]) or not isinstance(metadata.get("tensors"), list) or len(metadata["tensors"]) != len(names):
                raise ValueError("capture input names/order do not match canonical track contract")
            if metadata.get("input_contract_sha256") != contract_digest:
                raise ValueError("capture input contract digest does not match current registry")
            if metadata.get("request_schema") != "meco-inference-v2":
                raise ValueError("unsupported captured request schema")
            time_fields = ("observation_timestamp", "capture_monotonic",
                           "dispatch_monotonic", "expires_monotonic")
            if (type(metadata.get("observation_schema")) is not int
                    or any(isinstance(metadata.get(key), bool)
                           or not isinstance(metadata.get(key), (int, float))
                           for key in time_fields)
                    or not isinstance(metadata.get("candidate_ids"), list)
                    or any(not isinstance(value, str) or not value
                           for value in metadata["candidate_ids"])
                    or not isinstance(metadata.get("valid_mask"), list)
                    or any(type(value) is not bool for value in metadata["valid_mask"])
                    or type(metadata.get("fallback_index")) is not int):
                raise ValueError("capture request fields have invalid types")
            if set(archive.files) != {"__metadata__", *(row.get("key") for row in metadata["tensors"])}:
                raise ValueError("capture tensor inventory mismatch")
            if set(declared) != {row.get("key") for row in metadata["tensors"]}:
                raise ValueError("capture tensor declarations do not match metadata")
            shapes = []
            tensors = []
            integer_names = {"edges", "region_ids", "coarse_edges"}
            boolean_names = {"edge_mask", "coarse_edge_mask"}
            for name, row in zip(names, metadata["tensors"]):
                declared_shape, declared_dtype = declared[row["key"]]
                array_shape, array_dtype = declared_shape, declared_dtype
                expected_dtype = ("<i8" if name in integer_names else
                                  "|b1" if name in boolean_names else "<f4")
                if (row.get("name") != name or list(array_shape) != row.get("shape")
                        or array_dtype != row.get("dtype") or array_dtype != expected_dtype):
                    raise ValueError(f"captured tensor declaration mismatch: {name}")
                array = archive[row["key"]]
                if _tensor_digest(name, array) != row.get("sha256"):
                    raise ValueError(f"captured tensor integrity mismatch: {name}")
                if array.dtype.kind == "f" and not np.isfinite(array).all():
                    raise ValueError(f"captured tensor is nonfinite: {name}")
                shapes.append(array_shape)
                tensors.append(torch.from_numpy(array.copy()))
        if hashlib.sha256(_canonical([row["sha256"] for row in metadata["tensors"]])).hexdigest() != metadata.get("content_sha256"):
            raise ValueError("capture content digest mismatch")
        request = MecoInferenceRequest(
            track=track, snapshot_id=metadata["snapshot_id"],
            checkpoint_sha256=metadata["checkpoint_sha256"],
            protocol_id=metadata["protocol_id"],
            observation_schema=metadata["observation_schema"],
            observation_timestamp=metadata["observation_timestamp"],
            capture_monotonic=metadata["capture_monotonic"],
            dispatch_monotonic=metadata["dispatch_monotonic"],
            expires_monotonic=metadata["expires_monotonic"],
            input_names=tuple(names), input_shapes=tuple(shapes),
            candidate_ids=tuple(metadata["candidate_ids"]),
            valid_mask=tuple(metadata["valid_mask"]),
            fallback_index=metadata["fallback_index"],
            request_sequence=metadata["request_sequence"],
            generation_id=metadata["generation_id"],
            candidate_mask_digest=metadata["candidate_mask_digest"],
            legal_pose_order_digest=metadata["legal_pose_order_digest"],
            input_contract_sha256=contract_digest,
            model_id=metadata["model_id"], schema=metadata["request_schema"])
        if (len(metadata.get("source_snapshot_sha256", "")) != 64
                or any(char not in "0123456789abcdef" for char in metadata["source_snapshot_sha256"])):
            raise ValueError("capture source provenance digest is invalid")
        return metadata, tuple(tensors)
