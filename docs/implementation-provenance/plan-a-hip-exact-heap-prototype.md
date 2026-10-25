# Plan A exact-heap HIP prototype

This prototype extracts the existing Python Dijkstra contract into an
independently testable CPU oracle and precomputes directed edge translation
and route-cost weights with the original Python `hypot/sin/cos/atan2` and
operation order. Directed destinations, edge costs, translations, and masks
are cached against a weak reference to each static pose graph, so host copies
and geometry math happen once per graph lifetime; an explicit HIP call also
caches its float64 edge tensors on the selected device for that graph lifetime.
The HIP kernel uses a persistent device-side binary
heap with `(distance, node_id)` ordering, stale-entry equality, slot-ordered
outgoing-edge relaxation, and the same strict `proposal < old - 1e-9` update
rule. It retains root connector, physical distance, parent edge, and parent.
The wrapper explicitly sorts reachable nodes by final route distance with
stable ascending-node tie order, matching the support builder's final sort.

The prototype is not dispatched by training or support construction. The
existing support path calls the extracted CPU oracle, preserving the fallback
path until device results are verified. The loader compiles only when called
explicitly and requires HIP. No GPU compile or execution, training, or simulator
evaluation was performed for this tranche.

Current hardening keeps `PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED` false, checks
WX9100 identity and tensor/root shape before an eligible explicit call, checks
edge destinations and finite nonnegative weights on device, and reports heap
overflow or invalid graph data instead of returning partial results. The C++
binding rejects heap-capacity integer overflow before allocation. A campaign
manifest opt-in is deferred while the dispatch gate is false; adding a selectable
backend now would expose an unusable setting and could imply runtime support
that has not passed hardware parity or profiling.

The focused CPU-only contract tests cover direct parity with the prior Python
loop on synthetic graphs, graph-lifetime weight caching, deterministic ties,
near-epsilon non-updates, disconnected defaults, root connector and physical
distance, traversal order, edge weight precomputation, and fallback choice. Before any
runtime dispatch is enabled, a later WX9100 gate must compile the extension,
compare every output and downstream support digest against the CPU oracle on
representative graphs (including production-sized graphs), exercise edge
corner cases, and profile throughput and memory. Until those checks pass, this
prototype is an implementation experiment and provides no GPU speedup or
simulation-parity claim.

## Device-resident graph route follow-up (implementer candidate)

The explicit `plan_a_dijkstra_hip_for_graph` entry point now passes the static
graph's device tensors directly to `dijkstra_graph`. The native binding converts
coordinates and headings to float64 on the selected device, and the HIP kernel
computes directed translation and wrapped heading costs during heap relaxation.
The path no longer calls `cached_plan_a_edge_weights`, creates Python graph
lists, or runs a Python heap. Its result tensors and settled traversal ordering
remain on device. The binding reads a scalar device status to fail closed on
invalid active edges or heap overflow.

This is still an explicit, gated prototype. `build_plan_a_support` continues to
call `plan_a_dijkstra_cpu`; its normal runtime path has not been changed. The
WX9100 parity/profile gate remains false. CPU-only contracts verify gate order,
absence of host graph materialization in the opt-in helper, device-kernel
geometry/epsilon/order expressions, and continued CPU support dispatch. They do
not establish HIP compilation, exact device parity, or runtime speedup. The
focused suite passed 13 tests; Python compilation and `git diff --check` passed.
No GPU compilation/execution, simulator evaluation, or replay was performed.


## Scoped three-pass record

**Planner pass.** The support builder copied the static legal graph to CPU and
ran Python `heapq` Dijkstra on every decision. The implementation preserves its
Python geometry operation order for graph-lifetime precomputed weights, then
prototypes a gated device heap.

**Implementer pass.** Commit `eae7c8ebfea8dce971845b95597ee95ca90a157d` adds the
CPU oracle, weak-reference graph cache, HIP binding/kernel/loader, and focused
contract tests.

**Reviewer pass.** Independent review approved that exact commit as a gated
prototype. The focused CPU suite passed 8 tests; Python compilation and
`git diff --check` passed. The HIP extension was not compiled or run. WX9100
parity, support-digest equality, and performance remain mandatory before
runtime dispatch; no simulator evaluation or replay was performed.

**Hardening implementation.** Commit `5f199c68efdc4e17ae77fa4e286c15adaf9093d6`
adds fail-closed device, root, shape, and finite-cost checks; a C++ heap
capacity overflow guard; and explicit error reporting for invalid active edges
or overflow. The parity/profile gate remains false and training support still
dispatches only to the CPU oracle. Ten focused CPU contracts, Python
compilation, and `git diff --check` passed. No HIP compile/run, simulator
evaluation, replay, or profile occurred.

**Independent exact-head review.** The reviewer approved the hardened commit as
a disabled prototype. Review confirmed the gate is checked before extension
loading, dispatch remains CPU-only, and both Python and C++ fail closed on
invalid device/shape/edge/heap inputs. The review explicitly does not establish
HIP parity, runtime speedup, or simulator behavior. The hardened commit was
integrated as `4164f49885d3c74fb56d8499d9089d1b039f4e17`.

**Device-resident routing follow-up — implementer.** Commit
`07d7f9d68257ee3760a8406f1d30d01a273e6484` adds an opt-in HIP kernel that
computes graph edge geometry and exact-heap routes on the device. The normal
`build_plan_a_support` path remains on the CPU oracle, and the parity/profile
gate remains false. Thirteen focused CPU contract tests, Python compilation,
and `git diff --check` passed. No HIP compile/run, WX9100 parity/profile, or
simulator/replay validation was performed.

**Independent exact-head review.** The reviewer approved
`07d7f9d68257ee3760a8406f1d30d01a273e6484` as a gated prototype. Static review
confirmed the heap ordering, edge-slot traversal, `1e-9` relaxation threshold,
root connector, parent and physical-distance updates, unreachable values, and
failure checks. The review also confirmed that runtime support construction
still dispatches to the CPU oracle. Source-contract tests do not prove compiled
HIP parity; hardware compilation, exact output/support-digest comparison,
profiling, and a seeded simulator replay remain outstanding. The approved code
was integrated on `main` as `fa59b2dcd`.
