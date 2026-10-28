"""Versioned, fixed-chunk SHA-256 evidence digest for device-resident bytes.

The root binds the scheme, input length, chunk size, and ordered SHA-256
chunk hashes reduced through deterministic levels. It is a different digest from the unchunked v1 and must be labeled v2.
"""
import hashlib
import struct

import torch

CHUNK_BYTES = 1024
DOMAIN = b"meco-plan-a-route-evidence-tree-sha256-v2\0"


def _prefix(length: int, domain: bytes) -> bytes:
    return domain + struct.pack(">QQ", length, CHUNK_BYTES)


def tree_digest_cpu(data: bytes, *, domain: bytes = DOMAIN) -> str:
    if not data:
        raise ValueError("tree digest requires a nonempty byte stream")
    length = len(data)
    while True:
        data = b"".join(hashlib.sha256(data[offset:offset + CHUNK_BYTES]).digest()
                        for offset in range(0, len(data), CHUNK_BYTES))
        if len(data) <= CHUNK_BYTES:
            return hashlib.sha256(_prefix(length, domain) + data).hexdigest()


def tree_digest_hip_bytes(data: torch.Tensor, *, domain: bytes = DOMAIN) -> torch.Tensor:
    if (not isinstance(data, torch.Tensor) or data.device.type != "cuda" or
            torch.version.hip is None or data.dtype != torch.uint8 or
            data.ndim != 1 or not data.is_contiguous() or not data.numel()):
        raise ValueError("tree digest requires nonempty contiguous HIP bytes")
    from .meco_tree_digest_hip import extension as chunk_extension
    from .meco_plan_a_shortest_path_hip import extension as root_extension
    length = data.numel()
    while True:
        data = chunk_extension().sha256_chunks(data, CHUNK_BYTES).reshape(-1)
        if data.numel() <= CHUNK_BYTES:
            break
    prefix = torch.frombuffer(bytearray(_prefix(length, domain)), dtype=torch.uint8).to(data.device)
    return root_extension().sha256_bytes(torch.cat((prefix, data)))


def tree_digest_hip(data: torch.Tensor, *, domain: bytes = DOMAIN) -> str:
    return bytes(tree_digest_hip_bytes(data, domain=domain).cpu().tolist()).hex()
