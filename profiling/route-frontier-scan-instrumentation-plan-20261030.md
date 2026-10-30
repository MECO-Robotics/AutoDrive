# Exact route-frontier scan investigation

This branch is a design-only follow-up to the A1 route profile. It contains no
runtime change, and no GPU run or test was performed.

## What the current kernel guarantees

The production graph call reaches `plan_a_frontier_kernel()` through
`dijkstra_graph_frontier()` in `meco_plan_a_shortest_path_hip.cpp`. The kernel
currently reduces the minimum unvisited key as `(route_cost, node_id)` on every
round (`meco_plan_a_shortest_path_hip_kernel.cu:473-493`). For the usual
positive-edge path it processes the full frontier
`!visited && route < min_route + min_positive_edge_cost` as one round
(`:495-516`). When zero-cost edges exist, it handles the equal-distance
boundary one node at a time in node-ID order and switches to the serial heap
after 128 such rounds (`:495-512`).

Within each destination, incoming proposals are examined in
`(source_route, source_node, source_slot)` order (`:536-565`). A parent changes
only for a proposal strictly below `route[v] - 1e-9`; equal or near-equal
alternatives preserve the first parent. Returned outputs include route costs,
physical distances, parent node/edge, traversal order, and status. The wrapper
forms traversal order with a stable sort of route costs (`meco_plan_a_shortest_path_hip.cpp:322-327`),
so equal costs must remain node-ID ordered and unreachable/error rows must stay
`-1`.

## Why not prototype a new queue yet

Avoiding a full-node minimum scan requires replacing the current globally
ordered frontier with an active ordered structure. A mutable parallel heap or
an exact bucket structure is not a narrow change: it must deduplicate parallel
decreases, discard stale keys, preserve the strict relaxation tolerance and
parent tie order, and reproduce the zero-edge fallback. Using an approximate
bucket width would change which nodes share a frontier round and can change
parents even when final costs appear equal. Source inspection proves the
current ordering rules but does not prove an alternative data structure's
equivalence. Implementing one before measuring scan counts would be guesswork.

The relevant saved A1 profile is
`AutoDriveCampaigns/meco-full-match-offense-production-20261030-0f13e5bd3/profiling/route-pathclear/candidate-stage-profile-v2/route-pathclear-candidate.json`
in the primary checkout: support build p50 16.35 ms, digest p50 2.41 ms,
route-support wall p50 about 125 ms. Those aggregates do not say how many
frontier rounds ran or whether the serial fallback was used. The available
`.pt` captures contain support outputs, not the graph/root input tensors passed
to `dijkstra_graph`; a suitable low-level solver capture is not currently
available. Capture those inputs from a production A1 call before implementing
the counters below. Save scenario, seed, checkpoint identity, source hashes,
tensor shapes/dtypes, and the call-site provenance with the capture.

## Instrument-only benchmark plan

After capturing the production graph/root inputs, add an opt-in stats output to
the HIP extension for one invocation. Keep the
existing route outputs and dispatch unchanged. Put counters in a separate
fixed-size `int64`/`float64` device tensor and write them only when stats are
requested. Count and return:

1. **Graph preparation:** `n`, `slots`, allowed-edge count, reachable valid
   edge count, positive minimum edge cost, zero-cost edge count, maximum
   incoming degree, incoming capacity, and the reason the precheck chooses
   serial fallback (bit flags for invalid edge, indegree, capacity, no positive
   edge, or too many zero edges).
2. **Frontier loop:** round count; total node-key visits (expected `n` per
   round); frontier nodes total and maximum per round; total active destination
   nodes; incoming predecessor comparisons; relax proposals; successful
   relaxations; zero-boundary single-node rounds; late serial fallback reason
   and late fallback round.
3. **Output checks:** status; finite/reachable node count; traversal length;
   exact hashes for route, physical distance, parent, parent edge length, and
   traversal tensors. The harness should also save the full tensors so a hash
   match is not the only parity evidence.

Time the stages separately in the instrumented wrapper: edge-cost/length
precompute, CSR count/prefix/scatter, frontier kernel, stable sort, and total
extension wall time. Use HIP events on the active stream and report
instrumentation-off timings separately because counters and events perturb the
kernel. The paired production timing should use the same saved graph/root
inputs, warmed extension, and alternating A/B/A order. Do not benchmark a new
queue until the instrument-only data shows `n × rounds` scans dominate total
time and reports actual fallback frequency.

## Exact outputs required for a later candidate

For every captured graph/root, compare exactly: status, route-cost tensor,
physical-distance tensor, parent IDs, parent-edge lengths, stable traversal
order, reachable mask, Plan A support mask, fallback pair, support order
digest, and support digest. Then compare selected decisions from the seeded
full-match replay. Record candidate speedup only after paired timing and these
exact checks pass.
