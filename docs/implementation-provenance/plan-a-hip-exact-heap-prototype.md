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
