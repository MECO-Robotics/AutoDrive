"""Device SHA-256 vectors; HIP compilation/execution is deferred by lease policy."""
import hashlib

import pytest
import torch


@pytest.mark.parametrize("payload", (b"", b"abc", bytes(range(256)), b"x" * 1000))
def test_hip_sha256_matches_standard_vectors(payload):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("HIP device lease/runtime gate is closed")
    from frc_defense.meco_plan_a_shortest_path_hip import extension

    source = torch.tensor(list(payload), dtype=torch.uint8, device="cuda")
    actual = bytes(extension().sha256_bytes(source).cpu().tolist()).hex()
    assert actual == hashlib.sha256(payload).hexdigest()
