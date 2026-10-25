# Plan A HIP parity and profiling harness plan

## Planner pass

**Scope:** define a source-bound, CPU-only offline fixture mode and a separate,
explicitly invoked WX9100 verification mode for the Plan A exact-heap HIP
prototype. The offline mode can be implemented and contract-reviewed without
device access. Compiling or running HIP, recording parity/profile evidence, and
any future dispatch decision require a later exclusive WX9100 lease. Normal
training and support construction must continue to use the CPU oracle; the
production parity/profile gate stays false throughout this tranche.

**Planning base:** `main` at `48df1c7d03d72b2645e36070f34f91bf9a4b92ba`.
The root checkout has an unrelated modified generated file,
`frc_defense/csrc/tensor_collision_multi_hip_kernel.hip`; this plan is in an
isolated worktree and excludes that file.

## Evidence and current blocker

- `frc_defense/meco_plan_a_shortest_path.py` defines
  `PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED = False`. Both public HIP prototype
  entry points fail before extension loading while the gate is closed.
- `frc_defense/meco_plan_a_policy.py::build_plan_a_support` uses
  `plan_a_dijkstra_cpu` by default. Its explicit HIP-required collector path
  calls `plan_a_dijkstra_hip_for_graph` and fails closed while the gate is
  closed. The full Plan A model/training path has no HIP dispatch.
- `frc_defense/csrc/meco_plan_a_shortest_path_hip.cpp` exposes both a
  preweighted-edge entry point and a device-resident graph entry point. The
  latter converts geometry to float64 on-device and runs the exact heap in
  `frc_defense/csrc/meco_plan_a_shortest_path_hip_kernel.cu`.
- `tests/test_meco_plan_a_shortest_path_prototype.py` covers the CPU oracle,
  source contracts, closed-gate ordering, and selected kernel expressions.
  These checks do not compile or execute HIP and do not establish parity.
- `docs/implementation-provenance/plan-a-hip-exact-heap-prototype.md` records
  the outstanding hardware gate: compile the extension; compare every route
  output and the downstream support digest with the CPU reference on corner,
  representative, and production-size graphs; then measure throughput and
  memory. No such compile, execution, exact comparison, or profile is claimed
  here.

## Proposed implementation tranche

Keep this as a verification harness, not a runtime backend selector. Candidate
files are `scripts/verify_plan_a_hip_parity.py`, a compact fixture manifest
and fixture tensors under `tests/fixtures/plan_a_hip/`, contract coverage in
`tests/test_meco_plan_a_shortest_path_prototype.py`, and this plan/provenance
record. Do not modify the normal dispatch or set the production gate true.

### 1. Frozen source-bound fixtures and CPU-only mode

Define a versioned fixture manifest with a canonical digest. Each case records
the graph tensors (or a deterministic graph-construction recipe), root,
`root_start_cost`, `root_connector_m`, case purpose, and expected CPU reference
outputs. Bind the manifest to the exact maintained source file hashes for the
CPU oracle, graph construction used by the fixture, support construction and
digest, native binding, HIP kernel, and build flags. Exclude generated or
unrelated source files. A changed source hash, fixture content, schema, or
expected-output digest must fail closed and require an intentional fixture
refresh with reviewer approval.

Include small deterministic cases for equal-distance/node-ID and edge-slot
tie behavior, the strict `1e-9` relaxation boundary, disconnected nodes,
nonzero root connector, physical distance distinct from route cost, and
invalid active-edge/geometry inputs. Include at least one realistic
production-sized static pose graph case so the heap capacity and memory
behavior are represented. Fixtures must exercise both the preweighted input
contract and geometry computed by `dijkstra_graph`; the geometry case must not
be reduced to testing source text.

The default CPU-only mode validates fixture schema, pinned source hashes,
fixture hashes, CPU reference outputs, and expected support digest. It emits an
immutable report labeled `offline_fixture_only`, with no HIP compilation,
extension import, device discovery, or parity/profile claim. The command must
not alter the closed production gate or any training configuration.

### 2. Explicit, post-lease device verification mode

Provide a separate explicit mode requiring an indexed device argument
(`cuda:0` or `cuda:1`), the HIP runtime, and an attested WX9100 identity. Do
not auto-select a device, accept CPU/P600/other accelerators, fall back to the
CPU path after a device error, or run while an exclusive training lease is
active. The mode must require a documented lease handoff/authorization before
loading the extension. Device compilation and execution are **not authorized
by this planner pass** and are deferred until the orchestrator confirms the
training lease is released.

Once authorized, run the same immutable graph cases through the CPU oracle and
both relevant HIP paths where inputs permit. Compare all five outputs:
`route_cost`, `physical_distance`, `parent_ids`, `parent_edge_m`, and
`traversal_order`. Require exact equality, including finite/infinite locations,
parent identity, and ordering, unless an independently reviewed numerical
contract establishes a narrower justified comparison before collecting
results. Also build equivalent Plan A support from each route result and compare
`plan_a_support_digest`; do not claim downstream parity based only on route
arrays.

The run must fail closed on compilation failure, device mismatch, stale source
or fixture, incomplete cases, any output mismatch, or digest mismatch. It must
never set `PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED` or change production
dispatch. Record a candidate result only; gate-opening requires separate
review and a subsequent implementation decision after both parity and profile
criteria pass.

### 3. Profile evidence

For each eligible case on the exclusively leased WX9100, record explicit
warm-up and repeat counts, synchronized elapsed-time distribution, peak
allocated and reserved device memory, and relevant graph dimensions/edge
counts. Compare against the CPU oracle and report speedup only with the exact
timing boundaries stated. Separate extension compilation and first-use costs
from steady-state measurements. Do not imply end-to-end policy, simulator,
training, or deployment speedup. If the one-thread exact heap is slower or
consumes unacceptable memory, preserve the negative result and keep dispatch
closed.

### 4. Immutable report and provenance

Write a new report path for every invocation; never overwrite prior evidence.
The report includes schema/evidence class, pass/fail and per-case records,
fixture manifest hash, all fixture tensor and expected-output hashes, Git
commit and clean-tree state, hashes of the exact maintained sources and build
configuration, PyTorch/ROCm/compiler versions, HIP architecture, indexed
device model and physical identity, lease authorization reference, compile
identity, parity results for every output and support digest, warm-up/repeat
counts, timing distributions, and peak memory. Clearly distinguish
`offline_fixture_only`, `hip_parity_candidate`, and `hip_parity_profile_candidate`;
none of these labels means the production gate is open.

## Acceptance criteria for the implementer

- CPU-only mode validates and reports the pinned fixtures and CPU oracle
  without importing the extension, probing or touching a GPU, or claiming HIP
  parity.
- Fixture identity binds relevant source hashes, canonical fixture bytes,
  reference outputs, and downstream support digest; stale or modified inputs
  are rejected.
- Fixture set covers tie/slot order, epsilon behavior, disconnected nodes,
  root connector versus physical distance, invalid active data, a
  production-sized graph, and both edge-weight and graph-geometry paths.
- Device mode is separate and explicit, requires an indexed attested WX9100
  and lease handoff, and cannot silently fall back or run on another backend.
- The device report contract compares every returned field and the downstream
  support digest, and records complete source/build/device/runtime provenance
  and throughput/memory methodology.
- Every result remains candidate evidence; the constant stays false, normal
  training/support dispatch stays CPU-only, and no manifest/runtime opt-in is
  added.
- No simulator, training, replay, dashboard, or hardware job is part of the
  implementation tranche. The implementer and reviewer do not run tests while
  the active instruction prohibits tests; if later tests check simulated game
  behavior, the project `AGENTS.md` requires a representative fresh simulator
  evaluation and replay/dashboard currentness checks.

## Independent reviewer criteria

Review the exact implementation commit and verify that (1) offline mode is
actually device-free and proves only fixture/reference integrity; (2) every
fixture and expected output is bound to canonical content and source hashes;
(3) geometry and preweighted HIP paths are both represented; (4) all output
fields and downstream support digest are compared; (5) device mode requires
the exact explicitly selected, physically attested WX9100 and documented
exclusive lease handoff; (6) all failure paths reject without fallback; (7)
profile timing and memory capture are synchronized, reproducible, and
accurately scoped; (8) evidence labels and report provenance cannot be
self-asserted to open the production gate; and (9) normal CPU dispatch and the
false gate remain intact. Any later proposal to enable dispatch needs a new
scoped reviewer pass on complete parity/profile evidence, not merely review of
the harness.

## Explicitly deferred runtime gate

No HIP compile or run, WX9100 parity comparison, support-digest comparison,
throughput/memory profile, simulator evaluation, training, replay, or dashboard
activity occurs in the planner or code-only implementation passes. These
runtime items remain pending until the live continuous-training supervisor
releases the exclusive WX9100 leases and the orchestrator explicitly schedules
the verification run. The currently closed gate remains false until all
required device evidence has been collected and independently reviewed.
