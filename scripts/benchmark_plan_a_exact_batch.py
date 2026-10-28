#!/usr/bin/env python3
"""Leased experimental shared-graph batch benchmark; never opens dispatch."""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from frc_defense.meco_device_lease import lease_devices
from frc_defense.meco_gpu_attestation import verify_post_lease_wx9100
from frc_defense.meco_plan_a_hip_parity_harness import (
    _fixture_result, _route_inputs, materialize_case, source_hashes,
)
from frc_defense.meco_plan_a_shortest_path import plan_a_dijkstra_cpu


def ms(start):
    return (time.perf_counter_ns() - start) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True, choices=['cuda:0', 'cuda:1'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('at least three repeats are required')
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain']):
        raise RuntimeError('clean checkout required')
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    manifest, _ = _fixture_result(root)
    from frc_defense import meco_plan_a_shortest_path as shortest
    assert not shortest.PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED
    with lease_devices((args.device,), owner='plan-a-exact-batch-benchmark') as lease:
        attestation = verify_post_lease_wx9100(lease, (args.device,))
        from frc_defense.meco_plan_a_shortest_path_hip import extension
        t = time.perf_counter_ns()
        ext = extension()
        compile_ms = ms(t)
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        cases = {c['id']: c for c in manifest['cases']}
        records = []
        for case_id in ('equal_distance_node_and_edge_slot_ties',
                        'strict_relaxation_epsilon', 'disconnected_defaults',
                        'geometry_root_connector_and_turn_cost',
                        'production_pose_lattice_8heading_100x80'):
            graph = materialize_case(cases[case_id])
            inputs, _ = _route_inputs(graph)
            n = len(inputs['edge_dst'])
            slots = max(map(len, inputs['edge_dst']))
            def padded(rows, fill):
                return [list(row) + [fill] * (slots-len(row)) for row in rows]
            t = time.perf_counter_ns()
            host = [torch.tensor(padded(inputs[k], fill), dtype=dtype) for k, fill, dtype in (
                ('edge_dst', 0, torch.long), ('edge_costs', float('inf'), torch.float64),
                ('edge_lengths', 0., torch.float64), ('edge_allowed', False, torch.bool))]
            host_pack_ms = ms(t)
            batches = (1, 4, 16, 32) if n == 64000 else (4,)
            for b in batches:
                roots_list = [(graph['root'] + i * max(1, n//b)) % n for i in range(b)]
                starts_list = [float(graph['root_start_cost'])] * b
                connectors_list = [float(graph['root_connector_m'])] * b
                cpu_samples = []
                for _ in range(args.repeats):
                    t = time.perf_counter_ns()
                    cpu_results = [plan_a_dijkstra_cpu(**inputs, root=r,
                        root_start_cost=s, root_connector_m=c) for r, s, c in
                        zip(roots_list, starts_list, connectors_list)]
                    cpu_samples.append(ms(t))
                root_host = [torch.tensor(roots_list, dtype=torch.long),
                    torch.tensor(starts_list, dtype=torch.float64),
                    torch.tensor(connectors_list, dtype=torch.float64)]
                def transfer():
                    return [v.to(device) for v in host + root_host]
                def allocate():
                    opts = {'device': device}
                    return [torch.empty((b,n), dtype=d, **opts) for d in
                        (torch.float64, torch.float64, torch.long, torch.float64, torch.long)] + [
                        torch.empty((b,n*slots+1), dtype=d, **opts) for d in
                        (torch.float64, torch.long)] + [torch.empty(b, dtype=torch.long, **opts)]
                def launch(ins, out):
                    ext.dijkstra_batch_into(*ins, out)
                ins, out = transfer(), allocate()
                launch(ins, out)
                torch.cuda.synchronize(device)
                kernel, full, inclusive = [], [], []
                for _ in range(args.repeats):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); launch(ins, out); end.record(); end.synchronize()
                    kernel.append(start.elapsed_time(end))
                    t = time.perf_counter_ns(); temp = allocate(); launch(ins, temp)
                    status = temp[-1].cpu(); torch.cuda.synchronize(device); full.append(ms(t))
                    if status.any().item(): raise RuntimeError(f'batch status {status.tolist()}')
                    t = time.perf_counter_ns(); temp_ins = transfer(); temp = allocate(); launch(temp_ins, temp)
                    host_out = [v.cpu() for v in temp[:5]]; status = temp[-1].cpu()
                    torch.cuda.synchronize(device)
                    lists = [v.tolist() for v in host_out]
                    lists[4] = [[node for node in row if node >= 0] for row in lists[4]]
                    inclusive.append(ms(t))
                    if status.any().item(): raise RuntimeError(f'batch status {status.tolist()}')
                fields = ('route_cost', 'physical_distance', 'parent_ids', 'parent_edge_m', 'traversal_order')
                for i, expected in enumerate(cpu_results):
                    for j, field in enumerate(fields):
                        got = lists[j][i]
                        if got != expected[field]:
                            raise RuntimeError(f'{case_id} batch={b} search={i} mismatch {field}')
                records.append({'case': case_id, 'nodes': n, 'slots': slots, 'batch': b,
                    'roots': roots_list, 'exact_all_fields': True, 'cpu_ms': cpu_samples,
                    'host_pack_ms': host_pack_ms, 'kernel_ms': kernel, 'full_call_ms': full,
                    'transfer_inclusive_ms': inclusive,
                    'cpu_over_kernel': statistics.median(cpu_samples)/statistics.median(kernel),
                    'cpu_over_full': statistics.median(cpu_samples)/statistics.median(full),
                    'cpu_over_transfer_inclusive': statistics.median(cpu_samples)/statistics.median(inclusive)})
        if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain']):
            raise RuntimeError('checkout changed during run')
        report = {'commit': commit, 'source_sha256': source_hashes(root), 'device': args.device,
            'attestation': attestation, 'compile_ms': compile_ms, 'records': records,
            'gate_open': False, 'scope': 'experimental shared CPU-preweighted graph; transfers include output consumption; CPU input weight generation and host packing excluded from ratios'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f: json.dump(report, f, indent=2, sort_keys=True)
        print(args.output)


if __name__ == '__main__':
    main()
