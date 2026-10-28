# A2/A3 graph message reuse

Each SE(2) edge applies the same message network to its destination node.
The candidate applies that network once per node and gathers its output for
each edge. The masked aggregation and model parameters are unchanged. Floating
point gradient accumulation can differ because the operations are reordered.

Focused CPU checks passed for forward outputs, node and parameter gradients,
invalid edges, padded nodes and status bits. On one CPU thread with 4,096
nodes, 10 edges per node and width 64, eight alternating warm forward/backward
pairs had medians of 87.90 ms before and 55.04 ms after.

On an exclusively leased WX 9100, deterministic algorithms enabled, five
alternating warm forward/backward pairs on a synthetic graph with 12,096 nodes,
ten edge slots and 64 regions produced these medians: A2 84.94 ms before and
50.15 ms after; A3 118.87 ms before and 66.51 ms after. Utility, penalty,
behavior and hidden outputs were bitwise equal in this sample. The largest
absolute parameter-gradient differences were 4.27e-8 for A2 and 2.09e-7 for
A3. Input features and connections were synthetic; the node count matches one
observed pickup graph, not its actual edges. The synchronized measurements and
device attestation are in `AutoDriveDiagnostics/plan-a-graph-message-20261028/benchmark.log`;
`guard.json` records the runtime monitor. Reproduce with
`PYTHONPATH=. python scripts/benchmark_plan_a_message_reuse.py --device cuda:1 --nodes 12096 --repeats 5`.

These measurements do not establish end-to-end training speedup. Outstanding:
compare on captured A-track inputs, measure PPO updates and run a fresh
end-to-end simulator evaluation with current dashboard/replay. The change is a
candidate until those checks pass.
