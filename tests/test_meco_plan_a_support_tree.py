from types import SimpleNamespace
import hashlib
import struct

import pytest
import torch

from frc_defense.meco_plan_a_support_tree import (DOMAIN, support_tree_digest,
    support_tree_digest_device, support_digest_v1_compatible)
from frc_defense.meco_tree_digest import CHUNK_BYTES

SUPPORT_NAMES = ('node_ids', 'behavior_ids', 'action_ids', 'pair_node_ids',
    'descriptor_ids', 'valid_mask', 'fallback_pair_tensor', 'distances_m',
    'route_cost', 'parent_ids', 'parent_edge_m', 'traversal_order',
    'ancestor_jumps', 'goal_xy', 'goal_heading')
GRAPH_NAMES = ('heading_bin', 'graph_xy', 'lookup', 'edges', 'edge_mask',
    'node_valid', 'region_ids', 'coarse_edges', 'coarse_edge_mask', 'graph_status')


def _case(device='cpu'):
    data = lambda: torch.tensor([1, 2, 3], dtype=torch.int64, device=device)
    support = SimpleNamespace(**{name: data() for name in SUPPORT_NAMES})
    support.root_node = torch.tensor(2, dtype=torch.int64, device=device)
    support.root_connector_m = torch.tensor(.25, dtype=torch.float64, device=device)
    support.fallback_pair_tensor = torch.tensor(1, dtype=torch.int64, device=device)
    support.snapshot_id = 'episode:3'; support.order_digest = 'a' * 64
    support.fallback_pair = -1
    graph = SimpleNamespace(**{name: data() for name in GRAPH_NAMES})
    graph.xy = graph.graph_xy
    return support, graph


def _canonical_bytes(support, graph):
    named = [('root_node', support.root_node),
             ('root_connector_m', support.root_connector_m),
             ('fallback_pair_tensor', support.fallback_pair_tensor)]
    named += [(name, getattr(support, name)) for name in SUPPORT_NAMES if name != 'fallback_pair_tensor']
    named += [(name, getattr(graph, 'xy' if name == 'graph_xy' else name)) for name in GRAPH_NAMES]
    header = ['plan-a-support-sha256-v1', support.snapshot_id, support.order_digest,
              f'legacy_fallback={support.fallback_pair}']
    for name, tensor in named:
        header.append(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}')
    data = ('\n'.join(header) + '\n').encode() + b''.join(
        tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        for _, tensor in named)
    return data


def _independent_oracle(support, graph):
    data = _canonical_bytes(support, graph)
    length = len(data)
    while True:
        data = b''.join(hashlib.sha256(data[i:i+CHUNK_BYTES]).digest()
                        for i in range(0, len(data), CHUNK_BYTES))
        if len(data) <= CHUNK_BYTES:
            break
    return hashlib.sha256(DOMAIN + struct.pack('>QQ', length, CHUNK_BYTES) + data).hexdigest()


def test_support_tree_binds_all_fields_and_scalar_roots():
    support, graph = _case()
    expected = _independent_oracle(support, graph)
    assert support_tree_digest(support, graph, require_hip=False) == expected
    for owner, names in ((support, SUPPORT_NAMES), (graph, GRAPH_NAMES)):
        for name in names:
            tensor = getattr(owner, name)
            tensor.reshape(-1)[0] += 1
            assert support_tree_digest(support, graph, require_hip=False) != expected, name
            tensor.reshape(-1)[0] -= 1
    support.root_node += 1
    assert support_tree_digest(support, graph, require_hip=False) != expected
    with pytest.raises(RuntimeError, match='requires a HIP'):
        support_tree_digest(support, graph)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason='HIP required')
def test_support_tree_device_matches_cpu_and_detects_mutation():
    from frc_defense.meco_device_lease import DeviceLease
    try:
        lease = DeviceLease(['cuda:1'], owner='plan-a-support-tree-test').acquire()
    except RuntimeError:
        pytest.skip('device leased')
    try:
        cpu_support, cpu_graph = _case()
        support, graph = _case('cuda:1')
        expected = _independent_oracle(cpu_support, cpu_graph)
        assert support_digest_v1_compatible(support, graph) == hashlib.sha256(
            _canonical_bytes(cpu_support, cpu_graph)).hexdigest()
        assert support_tree_digest(support, graph) == expected
        raw = support_tree_digest_device(support, graph)
        assert raw.device == graph.xy.device and raw.dtype == torch.uint8 and raw.shape == (32,)
        assert bytes(raw.cpu().tolist()).hex() == expected
        support.root_connector_m += 1
        assert support_tree_digest(support, graph) != expected
    finally:
        lease.release()


@pytest.mark.parametrize('scheme', [None, 'sha256-v1'])
def test_ppo_rejects_legacy_support_before_optimizer(scheme, tmp_path, monkeypatch):
    import frc_defense.meco_track_training as trainer
    row = {'reward': torch.tensor(0.), 'done': False, 'logprob': torch.tensor(0.),
           'value': torch.tensor(0.), 'support': object(),
           'route_evidence_sha256': 'c' * 64, 'route_evidence_scheme': 'tree-sha256-v2',
           'route_evidence_source_identity_sha256': 'a' * 64, 'support_digest': 'b' * 64}
    if scheme is not None:
        row['support_digest_scheme'] = scheme
    monkeypatch.setattr(trainer, '_plan_a_route_evidence_digest', lambda *a, **kw: 'c' * 64)
    with pytest.raises(RuntimeError, match='GPU-only PPO requires v2 support'):
        trainer._ppo_update('A1', None, None, None, [row], None, None, update_number=1,
                            diagnostic_output=tmp_path, source_identity_sha256='a' * 64)
