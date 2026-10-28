import hashlib
import struct

import pytest
import torch

from frc_defense.meco_tree_digest import CHUNK_BYTES, DOMAIN, tree_digest_cpu, tree_digest_hip


@pytest.mark.parametrize('length', [1, 55, 56, 63, 64, 65, CHUNK_BYTES - 1,
                                   CHUNK_BYTES, CHUNK_BYTES + 1, 2 * CHUNK_BYTES + 31, 33 * CHUNK_BYTES + 1,
                                   CHUNK_BYTES * CHUNK_BYTES + 1])
def test_tree_digest_cpu_matches_explicit_definition(length):
    data = bytes((i * 37) % 256 for i in range(length))
    hashes = data
    while True:
        hashes = b''.join(hashlib.sha256(hashes[i:i + CHUNK_BYTES]).digest()
                          for i in range(0, len(hashes), CHUNK_BYTES))
        if len(hashes) <= CHUNK_BYTES:
            break
    expected = hashlib.sha256(DOMAIN + struct.pack('>QQ', length, CHUNK_BYTES) + hashes).hexdigest()
    assert tree_digest_cpu(data) == expected
    changed = bytearray(data)
    changed[length // 2] ^= 1
    assert tree_digest_cpu(changed) != expected


def test_tree_digest_rejects_empty():
    with pytest.raises(ValueError):
        tree_digest_cpu(b'')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None,
                    reason='requires HIP device')
@pytest.mark.parametrize('length', [1, 55, 56, 63, 64, 65, CHUNK_BYTES - 1,
                                   CHUNK_BYTES, CHUNK_BYTES + 1, 2 * CHUNK_BYTES + 31, 33 * CHUNK_BYTES + 1,
                                   CHUNK_BYTES * CHUNK_BYTES + 1])
def test_tree_digest_hip_matches_cpu(length):
    from frc_defense.meco_device_lease import DeviceLease
    try:
        lease = DeviceLease(['cuda:1'], owner='plan-a-tree-digest-test').acquire()
    except RuntimeError:
        pytest.skip('device leased')
    try:
        data = bytes((i * 37) % 256 for i in range(length))
        tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).to('cuda:1')
        assert tree_digest_hip(tensor) == tree_digest_cpu(data)
    finally:
        lease.release()
