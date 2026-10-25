# Pass 1 addendum: fixed-capacity Plan A pose graph

**Status:** proposed for independent Pass 1 approval; plan only
**Base:** `origin/main` at `2cddbcaf10a72a62e3cc0feb3fa83410a151eaea`
**Plan being clarified:** `docs/plans/gpu-resident-plan-a-decision-slice-20261025.md` (SHA-256 `21b5610319116ee80e95cdcc668c4caa3491dd8b2a204e7fdf71bd8366f87235`)
**Scope:** resolve dynamic legal-pose graph sizing and propagate padding validity through the approved Plan A decision slice. This addendum does not approve implementation or runtime behavior.

The pinned base plan file itself is still marked **Proposed; awaiting independent Pass 1 approval**. This addendum does not change that status; the Pass 1 reviewer must reconcile approval of the base plan and this clarification before Pass 2 implementation is authorized.

## Decision

The device-resident Plan A slice must not derive tensor shapes from device values. In particular, it must remove the host scalar read `int(legal.sum().item())` and all data-dependent shape operations such as `torch.nonzero` in live legal-pose graph construction. Use a grid-derived fixed capacity, with a device validity mask identifying the compact prefix of actually legal poses.

For planner dimensions `nx` and `ny`, define:

```text
M = Ncap = 8 * nx * ny
```

The active planner defaults to resolution `0.3 m` (`TensorADStar`), and the default simulated field is `16.54 m × 8.07 m` (`Tensor3v3`). This yields `nx=ceil(16.54/0.3)=56`, `ny=ceil(8.07/0.3)=27`, and `Ncap=12,096`. This is a source-derived default estimate, not a cap for arbitrary field configurations. Capacity is computed from static planner metadata; if a configured field/resolution changes, the capacity changes accordingly without inspecting device tensor contents.

## Canonical compaction and validity semantics

Flatten the `[heading=8, x=nx, y=ny]` legal-pose mask in the same row-major order as the current boolean indexing. A device prefix scan maps each legal flattened pose to its compact node ID, exactly preserving the current ascending legal-node IDs. Scatter legal pose coordinates and heading metadata into the prefix `[0, valid_count)`. Set `node_valid[i]` true exactly for that prefix; padded rows occupy `[valid_count, Ncap)` and have no legal-pose identity. `valid_count` remains device-resident and must not control host allocation, slicing, branching, or loop bounds.

All graph tensors have capacity-sized node axes. Invalid rows must be inert:

- `xy`, heading, and heading-bin padding use finite safe values; no predicate may treat them as valid poses.
- Edge IDs remain in range, while `edge_mask` is false for every edge with an invalid source or destination. Generate the ten direction/turn slots over fixed-capacity indices; do not build variable-length coordinate lists.
- Root costs for invalid rows are `+inf` before `argmin`, preserving first-valid-node tie behavior. The root must be valid.
- Shortest paths receive the validity mask or equivalent fail-closed invariants. Invalid rows remain unreachable with no parent and cannot enter traversal order. Root, costs, distances, and parent outputs remain device tensors.
- Plan A support retains eight descriptor rows per capacity node, preserving each valid node's current row identity and order. Every padded row has a false support mask, is excluded from scoring and fallback selection, and cannot be sampled. The deterministic fallback is chosen only among valid rows.
- A2/A3 outputs for padded rows are ignored by candidate gathering. They must not affect valid-node outputs, pooled representations, recurrent state, utility, penalties, behavior logits, or PPO probabilities.

## Exact ordered connectivity and pooling

The device connectivity/partition implementation must preserve the reference behavior in `meco_pose_graph.py`, including ascending start-node component discovery, FIFO traversal, edge-slot ordering, component allocation to at most 64 regions, stable evenly spaced seeds from the BFS-ordered component members, duplicate-seed suppression, FIFO region ownership, and sorted coarse-neighbor order. Ignore invalid nodes and edges at every stage. Report more than 64 components through a device status code; do not inspect a device scalar on the host to raise a data-dependent Python exception during the decision.

Implement the exact ordered queue semantics in a device kernel or establish a separately reviewed algorithm with bit-for-bit parity. Component or region label propagation that changes member order or tie ownership is not acceptable. A serial ordered device traversal is an allowed correctness-first implementation; its cost must be measured later and may not be hidden by a CPU fallback.

`region_ids` for padded rows must be a safe in-range value and must not contribute to regional sums or counts. Update `_pool_regions` to consume `node_valid`, scatter only valid rows, and count only valid rows. Each graph must still have at least one valid node. Fine graph message passing must mask padded edges, and A2/A3 decoding/gathering must preserve valid-node results exactly.

Use fixed-width coarse adjacency `[1,64,64]`, with destinations sorted in ascending region ID and `coarse_edge_mask` false for padding. This replaces the current data-dependent `Dc=max(out_degree)` width. Masked coarse slots must be numerically inert: the masked message aggregate and its denominator must match the current variable-width representation exactly. If this cannot be proved for both A2 and the inference/deployment path, stop and request a plan amendment instead of changing the model semantics.

## Contract and evidence propagation

`node_valid` is a required A2/A3 graph input and part of the typed graph contract. Update exact input-key/shape/dtype validators, training and evaluation dispatch, inference request input names, deployment/export schemas, and input provenance. Compatibility must be explicit: parameter-only checkpoints may remain loadable only if the new graph contract is bound and valid-row outputs match; older request/schema records must not silently omit validity.

Bind `node_valid`, the compact row-major ordering, capacity, grid dimensions, region IDs, and graph schema version into the legal-pose order/support digest. The digest and its validation must not call `.cpu()`, `.tolist()`, `.item()`, Python truth conversion, or another host projection over dynamic tensors before action selection. Any digest computed from dynamic graph tensors must be computed on device and exposed only through the approved opaque post-decision artifact boundary. The current support digest's CPU tensor hashing is not an allowed substitute.

`node_count` must no longer ambiguously mean both capacity and actual legal-node count. Use a static capacity for tensor shapes and device tensors for dynamic counts/status. Update all routing, support, candidate gathering, summaries, and consumers to use the appropriate concept. A Python integer derived from `nx`, `ny`, and fixed heading count is static capacity metadata; actual valid count is dynamic and remains on device.

## Memory and work estimate / admission gate

With FP32 pose/features and the default `Ncap=12,096`, the principal resident arrays in a first-order graph estimate are approximately 163 bytes per capacity node: `xy` (8), heading (4), heading-bin (8), lookup (8), ten int64 edge IDs (80), ten bool edge masks (10), region ID (8), node-valid (1), and nine FP32 node features (36). This is about 1,971,648 bytes, plus fixed coarse adjacency (`64×64` int64 IDs and bool masks, about 36 KiB). This estimate excludes allocator overhead and simultaneous temporaries.

For edge construction, the maximum `Ncap×10` candidate set is 120,960 edges. A `[Ncap,10,2,2]` FP32 swept-path buffer is about 1.94 MB; simultaneous int64 source/destination arrays add about 1.94 MB. Prefix-scan scratch and other geometry temporaries add further capacity-scaled memory. The nominal graph-construction working set is therefore already roughly 6 MB before planner footprint intermediates, model activations, route batches, Plan A's eight support rows per node, shortest-path heap/workspace, and allocator overhead. A2/A3 node support has `8*Ncap=96,768` rows at this default. These are explicit-array estimates, not a complete peak-memory proof.

Before implementation is accepted, Pass 2 must inventory the actual simultaneous allocation lifetimes (including `_footprint_path_clear`, Plan A support and shortest path) and produce a checked peak bound for configured grids and dtype. If this bound exceeds the agreed device-memory budget, or fixed-capacity model work is too costly for the training path, stop and return to Pass 1 with measured evidence. Do not silently lower capacity, chunk away legal nodes, allocate from a host-read valid count, or reintroduce CPU connectivity.

Work bounds for an exact ordered device implementation are `O(Ncap*10)` for edge construction, `O(Ncap*10)` for ordered component traversal, `O(64*C)` for the 64-region allocation (`C≤64`), `O(Ncap*10)` for region assignment, and `O(Ncap*10 + 64²)` for coarse adjacency. This is a linear graph-build work target, but an ordered serial kernel may have poor device utilization. Pass 2 must preserve exact ordering first; any parallel optimization needs a separate parity proof.

## Forbidden host shape/synchronization operations

Before the selected action is chosen, no tensor shape, allocation, loop count, dispatch, validity decision, or graph/support ordering may depend on a device scalar read by the host. Prohibited operations include `.item()`, `.tolist()`, `.cpu()`, Python `bool`/`int`/`float` conversion of dynamic tensor values, `nonzero` or boolean indexing when it creates a data-dependent output shape, and host exception branches based on device values. Static shape metadata (`nx`, `ny`, fixed degree, 8 headings, 64 regions, and `Ncap`) is allowed. Device status tensors may fail closed through device-side masks/selection; interpret or serialize dynamic results only at the plan's approved post-selection opaque artifact boundary.

## Required synthetic parity evidence

After the GPU lease and project validation gates allow tests, compare a compact CPU reference against the fixed-capacity device path on deterministic synthetic fixtures:

1. Legal-mask compaction with interleaved illegal cells and all eight heading bins; assert every valid coordinate, heading, lookup ID, and edge endpoint matches legacy row-major order exactly.
2. Tiny connected, disconnected, isolated, cyclic, asymmetric-directed, equal-allocation-ratio, and duplicate-seed graphs; compare valid `region_ids`, component-overflow status, sorted coarse destinations, and masks exactly to the current ordered reference.
3. Padded rows containing adversarial finite features and safe in-range edge IDs; prove changing padded values cannot change any valid A2/A3 utility/penalty, behavior logits, pooled region, hidden state, support validity, fallback, selected distribution, or PPO log-probability.
4. A graph with fewer legal nodes than capacity and one with every pose legal; compare root tie behavior, reachable route outputs/parents, all eight support slots per valid node, and action/fallback ordering. Assert padded rows remain unreachable and unselectable.
5. Coarse adjacency at degree 1, sparse and dense adjacency, duplicate transitions, and all 64 regions; compare the fixed-width masked message aggregate and final A2 region result to the current dynamic-width reference.
6. Digest sensitivity: change one valid bit, node ordering, edge, region label, or schema version and require a different bound identity; change padded numeric sentinels while keeping `node_valid=false` and require no decision identity or score change except where canonical artifact bytes intentionally include sentinels.

Tests are necessary but not sufficient. The original Plan A gates still require HIP parity/profiling, selected-device residency evidence, seeded simulator evaluation, fresh replay inspection, and current dashboard verification before runtime qualification. No GPU, simulator, dashboard, or training validation is authorized by this addendum.

## Stop conditions

Stop Pass 2 and return to Pass 1 if (a) row-major valid-node identity cannot be preserved, (b) padded rows affect any valid-node model or decision output, (c) exact ordered region/coarse semantics cannot be reproduced, (d) the required input/provenance/checkpoint contract cannot be updated compatibly, (e) the memory/work admission gate fails, or (f) any preselection host read or dynamic shape remains. No implementation, test, runtime, or training approval is implied until this addendum receives independent Pass 1 approval and the exact code candidate passes Pass 3.
