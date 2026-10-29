"""Parity checks for frontier overflow and serial fallback boundaries."""

def duplicate_case(count):
 return {'id':'incoming_'+str(count),'xy':[[0.,0.],[1.,0.]],'dst':[[1]*count,[0]*count], 'mask':[[True]*count,[False]*count], 'expected_status':0,'expected_mode':'serial' if count>4160 else 'parallel'}
cases=[duplicate_case(x) for x in (64,65,4160,4161)]
cases += [
 {'id':'multiple_overflow_destinations','xy':[[0.,0.],[1.,0.],[0.,1.]],'dst':[[1]*70+[2]*70,[0]*140,[0]*140],'mask':[[True]*140,[False]*140,[False]*140],'expected_status':0,'expected_mode':'parallel'},
 {'id':'zero_self_plus_positive','xy':[[0.,0.],[1.,0.]],'dst':[[0,1],[1,0]],'mask':[[True,True],[False,False]],'expected_status':0,'expected_mode':'parallel'},
 {'id':'zero_nonself','xy':[[0.,0.],[0.,0.],[1.,0.]],'dst':[[1,2],[2,0],[0,0]],'mask':[[True,True],[True,False],[False,False]],'expected_status':0,'expected_mode':'serial'},
 {'id':'self_only','xy':[[0.,0.]],'dst':[[0]],'mask':[[True]],'expected_status':0,'expected_mode':'serial'},
 {'id':'unreachable_invalid','xy':[[0.,0.],[1.,0.],[2.,0.]],'dst':[[1],[0],[99]],'mask':[[True],[False],[True]],'expected_status':0,'expected_mode':'serial','ignore_invalid_cpu':True},
 {'id':'reachable_invalid','xy':[[0.,0.],[1.,0.]],'dst':[[1,99],[0,0]],'mask':[[True,True],[False,False]],'expected_status':1,'expected_mode':'serial'},
 {'id':'invalid_root','xy':[[0.,0.]],'dst':[[0]],'mask':[[False]],'root':-1,'expected_status':3,'expected_mode':'serial'},
]


def test_frontier_fallback_boundaries():
    import torch
    import pytest
    from frc_defense.meco_device_lease import lease_devices
    from frc_defense.meco_gpu_attestation import verify_post_lease_wx9100
    from frc_defense.meco_plan_a_shortest_path import precompute_plan_a_edge_weights, plan_a_dijkstra_cpu, plan_a_cpu_turn_table
    from frc_defense.meco_plan_a_shortest_path_hip import extension
    from frc_defense.meco_plan_a_hip_parity_harness import _output_field_hashes, OUTPUT_KEYS

    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("HIP device required")
    device = "cuda:0"
    with lease_devices((device,), owner="plan-a-fallback-regression") as lease:
        verify_post_lease_wx9100(lease, (device,))
        ext = extension()
        hv, turn = plan_a_cpu_turn_table([0.], device=device)
        for c in cases:
            n = len(c["xy"])
            args = [torch.tensor(v, dtype=d, device=device) for v, d in (
                (c["xy"], torch.float64), ([0.] * n, torch.float64),
                (c["dst"], torch.long), (c["mask"], torch.bool),
                ([True] * n, torch.bool), (c.get("root", 0), torch.long),
                (0., torch.float64), (0., torch.float64))]
            result = ext.dijkstra_graph(*args, hv, turn)
            status = int(result[-1].item())
            assert status == c["expected_status"], c["id"]
            if status:
                continue
            mask = [row[:] for row in c["mask"]]
            if c.get("ignore_invalid_cpu"):
                mask[2] = [False]
            dst, cost, length, allowed = precompute_plan_a_edge_weights(
                c["xy"], [0.] * n, c["dst"], mask)
            cpu = plan_a_dijkstra_cpu(dst, cost, length, allowed, root=0,
                                     root_start_cost=0., root_connector_m=0.)
            actual = {key: result[i].cpu().tolist() for i, key in enumerate(OUTPUT_KEYS)}
            actual["traversal_order"] = [v for v in actual["traversal_order"] if v >= 0]
            assert _output_field_hashes(actual) == _output_field_hashes(cpu), c["id"]


def test_frontier_random_graphs():
    import random
    import pytest
    import torch
    from frc_defense.meco_device_lease import lease_devices
    from frc_defense.meco_gpu_attestation import verify_post_lease_wx9100
    from frc_defense.meco_plan_a_shortest_path import precompute_plan_a_edge_weights, plan_a_dijkstra_cpu, plan_a_cpu_turn_table
    from frc_defense.meco_plan_a_shortest_path_hip import extension
    from frc_defense.meco_plan_a_hip_parity_harness import _output_field_hashes, OUTPUT_KEYS

    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("HIP device required")
    device = "cuda:0"
    rng = random.Random(61720261029)
    with lease_devices((device,), owner="plan-a-frontier-random-regression") as lease:
        verify_post_lease_wx9100(lease, (device,))
        ext = extension()
        hv, turn = plan_a_cpu_turn_table([0.], device=device)
        for case in range(128):
            n, slots = rng.randint(2, 30), rng.randint(1, 12)
            xy = [[float(i), float(rng.randrange(12)) / 10.] for i in range(n)]
            dst = [[rng.randrange(n) for _ in range(slots)] for _ in range(n)]
            mask = [[rng.random() < .7 for _ in range(slots)] for _ in range(n)]
            values = ((xy, torch.float64), ([0.] * n, torch.float64),
                      (dst, torch.long), (mask, torch.bool), ([True] * n, torch.bool),
                      (0, torch.long), (0., torch.float64), (0., torch.float64))
            args = [torch.tensor(v, dtype=d, device=device) for v, d in values]
            result = ext.dijkstra_graph(*args, hv, turn)
            assert int(result[-1].item()) == 0, case
            indices, costs, lengths, allowed = precompute_plan_a_edge_weights(
                xy, [0.] * n, dst, mask)
            cpu = plan_a_dijkstra_cpu(indices, costs, lengths, allowed, root=0,
                                     root_start_cost=0., root_connector_m=0.)
            actual = {key: result[i].cpu().tolist() for i, key in enumerate(OUTPUT_KEYS)}
            actual["traversal_order"] = [v for v in actual["traversal_order"] if v >= 0]
            assert _output_field_hashes(actual) == _output_field_hashes(cpu), case
