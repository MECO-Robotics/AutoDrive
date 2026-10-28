# A2/A3 graph message reuse

Each SE(2) edge applies the same message network to its destination node.
The candidate applies that network once per node and gathers its output for
each edge. The masked aggregation and model parameters are unchanged. Floating
point gradient accumulation can differ because the operations are reordered.

Focused CPU checks passed for forward outputs, node and parameter gradients,
invalid edges, padded nodes and status bits. On one CPU thread with 4,096
nodes, 10 edges per node and width 64, eight alternating warm forward/backward
pairs had medians of 87.90 ms before and 55.04 ms after. This synthetic result
does not establish GPU or end-to-end training speedup.

Outstanding: compare A2 and A3 outputs and gradients on the target HIP device
and representative graph, measure forward and PPO update throughput, and run
a fresh end-to-end simulator evaluation with current dashboard/replay. The
change is a candidate until those checks pass.
