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
