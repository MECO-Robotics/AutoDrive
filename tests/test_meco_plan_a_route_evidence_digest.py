import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from frc_defense.meco_device_lease import DeviceLease
from frc_defense.meco_track_training import _plan_a_route_evidence_digest


def _original_digest(batch, source, route_cost, risk_cost):
    queries = [{"candidate_id": q.candidate_id, "snapshot_id": q.snapshot_id,
                "world": q.world, "slot": q.slot,
                "intake_extended": q.intake_extended} for q in batch.queries]
    header = {"schema": "meco-plan-a-route-evidence-v1",
              "source_identity_sha256": source,
              "route_source": "shared-device-xy-route-evaluation-v1",
              "query_order": queries, "cost_units": batch.cost_units,
              "cost_provenance": batch.cost_provenance}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode())
    names = ("status_code", "feasible", "path_xy", "path_length", "distance_m",
             "endpoint_error_m", "weighted_cost_grid_edges", "candidate_index",
             "bump_cost_grid_edges", "learned_cost_grid_edges")
    fields = [(name, getattr(batch, name)) for name in names]
    fields.extend((("route_cost_nodes", route_cost), ("risk_cost_nodes", risk_cost)))
    for name, tensor in fields:
        if tensor is None:
            digest.update(f"{name}=none\n".encode())
            continue
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _case(device):
    count = 7
    f = torch.arange(count, device=device, dtype=torch.float32)
    q = tuple(SimpleNamespace(candidate_id=f"node:{i}", snapshot_id="snapshot",
        world=0, slot=3, intake_extended=False) for i in range(count))
    return SimpleNamespace(queries=q, cost_units="units", cost_provenance="test",
        status_code=f.to(torch.int64), feasible=(f > 1),
        path_xy=torch.arange(count * 12, device=device, dtype=torch.float32)
            .reshape(count, 2, 6)[:, :, ::2],
        path_length=f.to(torch.int64), distance_m=f.to(torch.float64),
        endpoint_error_m=f, weighted_cost_grid_edges=f + 1,
        candidate_index=f.to(torch.int64), bump_cost_grid_edges=None,
        learned_cost_grid_edges=f + 2), f + 3, f + 4


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_route_evidence_bytes_match_original_and_detect_mutation(device):
    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    lease = None
    if device != "cpu":
        try:
            lease = DeviceLease([device], owner="route-evidence-digest-test").acquire()
        except RuntimeError:
            pytest.skip("device is leased")
    try:
        batch, route_cost, risk_cost = _case(device)
        source = "a" * 64
        actual = _plan_a_route_evidence_digest(batch, source, route_cost, risk_cost)
        assert actual == _original_digest(batch, source, route_cost, risk_cost)
        batch.path_xy[0, 0, 0] += 1
        changed = _plan_a_route_evidence_digest(batch, source, route_cost, risk_cost)
        assert changed == _original_digest(batch, source, route_cost, risk_cost)
        assert changed != actual
        assert _plan_a_route_evidence_digest(batch, "b" * 64, route_cost, risk_cost) != changed
        if device != "cpu":
            mixed_cost = route_cost.cpu()
            assert _plan_a_route_evidence_digest(batch, source, mixed_cost, risk_cost) == \
                _original_digest(batch, source, mixed_cost, risk_cost)
    finally:
        if lease is not None:
            lease.release()


def test_tree_route_evidence_detects_tensor_and_source_mutation():
    batch, route_cost, risk_cost = _case('cpu')
    first = _plan_a_route_evidence_digest(batch, 'a' * 64, route_cost, risk_cost,
                                           scheme='tree-sha256-v2')
    assert first != _plan_a_route_evidence_digest(batch, 'a' * 64, route_cost, risk_cost)
    assert first != _plan_a_route_evidence_digest(batch, 'b' * 64, route_cost, risk_cost,
                                                  scheme='tree-sha256-v2')
    batch.path_xy[0, 0, 0] += 1
    assert first != _plan_a_route_evidence_digest(batch, 'a' * 64, route_cost, risk_cost,
                                                  scheme='tree-sha256-v2')
    with pytest.raises(ValueError, match='unknown'):
        _plan_a_route_evidence_digest(batch, 'a' * 64, route_cost, risk_cost,
                                      scheme='unknown')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None,
                    reason='requires HIP device')
def test_tree_route_evidence_matches_cpu_oracle():
    try:
        lease = DeviceLease(['cuda:1'], owner='plan-a-route-tree-evidence-test').acquire()
    except RuntimeError:
        pytest.skip('device leased')
    try:
        cpu_batch, cpu_cost, cpu_risk = _case('cpu')
        gpu_batch, gpu_cost, gpu_risk = _case('cuda:1')
        expected = _plan_a_route_evidence_digest(cpu_batch, 'a' * 64, cpu_cost, cpu_risk,
                                                scheme='tree-sha256-v2')
        assert _plan_a_route_evidence_digest(gpu_batch, 'a' * 64, gpu_cost, gpu_risk,
                                             scheme='tree-sha256-v2') == expected
        gpu_batch.path_xy[0, 0, 0] += 1
        assert _plan_a_route_evidence_digest(gpu_batch, 'a' * 64, gpu_cost, gpu_risk,
                                             scheme='tree-sha256-v2') != expected
    finally:
        lease.release()


def test_tree_route_training_contract_rejects_cpu():
    batch, cost, risk = _case('cpu')
    with pytest.raises(RuntimeError, match='requires tensors on one HIP device'):
        _plan_a_route_evidence_digest(batch, 'a' * 64, cost, risk,
            scheme='tree-sha256-v2', require_hip=True)


@pytest.mark.parametrize('scheme', [None, 'sha256-v1'])
def test_plan_a_ppo_rejects_legacy_evidence_before_update(scheme, tmp_path):
    from frc_defense.meco_track_training import _ppo_update
    row = {'reward': torch.tensor(0.), 'done': False, 'logprob': torch.tensor(0.),
           'value': torch.tensor(0.), 'support': object(),
           'route_evidence_sha256': 'c' * 64,
           'route_evidence_source_identity_sha256': 'a' * 64}
    if scheme is not None:
        row['route_evidence_scheme'] = scheme
    with pytest.raises(RuntimeError, match='GPU-only PPO requires v2'):
        _ppo_update('A1', None, None, None, [row], None, None, update_number=1,
                    diagnostic_output=tmp_path, source_identity_sha256='a' * 64)
