# Route hot-path research (2026-10-30)

## Evidence inspected

The captured production A1 route profile is in
`AutoDriveCampaigns/meco-full-match-offense-production-20261030-0f13e5bd3/profiling/route-pathclear/candidate-stage-profile-v2/route-pathclear-candidate.json`
in the primary checkout. It records five A1 support builds with a 16.35 ms
median (`build_plan_a_support`), 17.05 ms p95, and a 134.64 ms first sample.
The same profile records 2.41 ms median for `support_tree_digest_device`.
The 12096 support rows correspond to 1512 graph nodes × 8 descriptors.

The profile's `route_support_wall_ms` has a 125.0 ms median after the first
sample; `route_pipeline` reports a 371.5 ms median and an 18.47 s p95. The
artifact does not define those aggregate intervals precisely enough to assign
their excess time to a particular kernel. Per-stage samples show large timing
outliers, so means and p95s should not guide a code change without a fresh
paired capture.

## Runtime source finding

`build_plan_a_support()` in `frc_defense/meco_plan_a_policy.py` invokes
`plan_a_dijkstra_hip_for_graph()` on the HIP path. That dispatch calls
`dijkstra_graph()` in `frc_defense/csrc/meco_plan_a_shortest_path_hip.cpp`,
which dispatches the frontier implementation in
`frc_defense/csrc/meco_plan_a_shortest_path_hip_kernel.cu`.

The normal graph route kernel is therefore not the legacy one-thread heap
kernel. It uses a 1024-thread frontier relaxation, with serial fallback for
invalid/oversized reverse adjacency, high indegree, unsuitable edge weights,
or excessive zero-edge ties. The returned traversal order is then formed by
`at::sort(route, stable=true)` in the C++ wrapper. The tempting optimization of
recording pops from `plan_a_dijkstra_search()` would affect the legacy path,
not the measured normal graph path, and would not improve the ordinary A1
profile unless profiling proves fallback dispatch is frequent.

## Next candidate

Profile the frontier route call separately from its graph-edge precompute,
frontier iterations, fallback dispatch, and stable route sort. A high-upside
candidate is to remove the full-node min scan per frontier round using an
ordered active frontier, while preserving the current equal-distance node
ordering, 1e-9 relaxation rule, zero-edge handling, parent selection, physical
distance, and failure status. This is not a safe narrow edit until actual
frontier-round counts and fallback frequency are measured; no code change is
made here.

## Required next benchmark

On the WX9100 HIP path, first capture the actual graph and root tensors passed
to `dijkstra_graph()` from a production A1 call. Existing `.pt` artifacts are
route support outputs and cannot serve as solver inputs. Include scenario,
seed, checkpoint identity, source hashes, tensor shapes/dtypes, and call-site
provenance. Then run paired A/B/A using identical captured inputs and source
identity. Record kernel-level timings and frontier/fallback counters, then
compare route costs, physical distances, parents, parent-edge lengths,
traversal order, graph status, support masks, support digests, and selected
route results exactly. If it wins, run the representative seeded end-to-end
simulator evaluation and verify the dashboard backend/frontend and fresh replay
provenance per `AGENTS.md` before promoting it. No runtime change or solver
speedup is established by the current evidence.
