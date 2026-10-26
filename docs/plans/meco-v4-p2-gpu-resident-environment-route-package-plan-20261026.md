# P2 GPU-resident environment, observation, candidate and route package plan

**Date:** 2026-10-26  
**Status:** Amendment candidate; original P2 plan received Pass 1 ACCEPT; this G0/source-authoring amendment requires fresh independent exact-head review
**Planner:** `plan_a_owner` (planner only; not implementer or reviewer)  
**Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`  
**Overall plan:** original accepted commit `6325920252773cfa05d63167ba3c29a90debdb01`; G0/source-authoring amendment candidate `5b25eb1cfba7850a871b9024d834457d0fb755a4` (pending fresh review)
**Original overall-plan file SHA-256:** `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`
**Amended overall-plan file SHA-256:** `f5d1751ae6ca02452f286d756116859f176558a5822711a48f2b1aa4d734b43a`
**Accepted P0 package plan:** `13dbd7f8f953995850e811202f7e9c5264c49dca`  
**P0-plan file SHA-256:** `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`  
**Reconciliation commit:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`  
**Reconciliation artifact SHA-256:** `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`  
**Canonical profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`  
**V4 envelope digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Objective and authority

Implement the P2 portion of the accepted MECO V4 plan: GPU-resident environment state and observation/perception, opponent and fuel tracks, candidate generation, static legal-pose graph construction, batched route search and path output, and explicit device placement/evidence on the two AMD WX9100 devices. Preserve current behavior and safety semantics unless a separately accepted reconciliation or package plan changes them. Match the canonical profile and keep one policy track bound to one process/run identity.

This is a package plan only. It authorizes no action until this amended exact plan receives independent Pass 1 ACCEPT, P0a is accepted, P1 is accepted and integrated, and distinct implementer/reviewer identities are assigned. Then Pass 2 may author only allowlisted source/test files and perform static source/diff review in an isolated ZFS worktree before G0. A pre-G0 independent source/interface review may return SOURCE/API ACCEPT for an exact candidate head; this is not package-final Pass 3 ACCEPT and does not authorize dependent-package integration or runtime use. Builds/compilation, test execution, simulator, profiler, numerical work, calibration/evaluation, dashboard, device/GPU use, training, checkpoint mutation, integration, merge and promotion remain prohibited until the user manually stops the live diagnostic campaign and both WX9100 leases are independently verified released. Do not touch live campaign PID `2750253` or its workers, outputs, checkpoints, dashboard, devices or leases.

P2 is strictly downstream of the accepted P0a preregistration and P1 interfaces. It precedes P3/P4/P5 consumers. P2 does not own score-event schema/semantics (P1), policy model implementations (P3/P4), shared PPO/trainer edits (P5), or qualification/scheduling (P6). It will audit those tensor boundaries and publish interface/residency requirements without editing their paths. The complete training/evaluation graph is a required audit; any unresolved non-P2 residency item must name its owning package and block the global GPU-only completion gate until that owner closes it.

## Contract and source evidence

The accepted profile fixes `N=1`, `S=6`, `R=32`, `T=12`, `E=504`, `K=16`, a one-track-per-run policy identity, 504 target slots and four-int target handles. The Plan A and Plan C shapes, masks and ordering are fixed by the pinned reconciliation. P2 preserves target identity, validity, reset generation, no-wrap semantics, stale-handle rejection, deterministic candidate order and route feasibility. P2 may not revise reconciled field/byte sizes or profile identities.

Read-only source inspection at `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22` found these concrete audit leads:

- `frc_defense/meco_device.py:5-21` normalizes one selected device and explicitly permits CPU. It does not bind or validate a two-WX9100 assignment.
- `frc_defense/_tensor_3v3_observation.py:93-110` counts active worlds with `.item()` in `_random_active` and uses data-dependent host control flow. The observation/perception update starts around line 113 and delegates detection work to `tensor_perception`.
- `frc_defense/tensor_3v3.py:634-680` converts active-mask counts and collection target validation to Python scalars; lines 818-840 use scalar finiteness checks and copy a goal to CPU for a request signature. Further dynamic pulls occur at lines 1385-1387 for perception update count. These sites are in the primary environment step and must be classified by full call graph before deciding whether any transfer is permitted.
- `frc_defense/meco_candidates.py:125-280` contains scalar `.item()` reads in candidate geometry and legality checks; lines 301-308 copy opponent pose/velocity to CPU; lines 344-375 convert detected tracks and features to Python lists/scalars; lines 412 and 445-462 include sequence extraction and host per-track matching loops.
- `frc_defense/meco_pose_graph.py:120-145` constructs legal graph tensors on the environment device but obtains a Python count with `.item()` before allocating the compact legal-node table; lines 150-233 use host `nonzero`/Python loops and convert edge tensors to lists to build graph regions and adjacency. Static field constants may be prepared on host; geometry-dependent masks, counts, connectivity, indexing and IDs may not be computed on CPU.
- `frc_defense/tensor_perception.py:25-89` dynamically loads a HIP extension; the extension dispatch points and Torch reference paths appear throughout the module (for example lines 160-208 and 438-520). A Torch reference path is acceptable only when its operations execute on the input WX9100. “Extension unavailable” must never mean CPU execution.
- `frc_defense/tensor_adstar.py:556-615` falls back to Python AD* repair after `.tolist()`/`.item()` transfers. Lines 825-850 explicitly gate local repair out of CUDA graph capture; this is a CPU algorithmic fallback and must be eliminated or made fail-closed on device. Additional device synchronization sites include lines 997-1008, 1108-1120, 1265-1269 and 1362-1380.
- `frc_defense/meco_track_training.py` has tensor `.item()`, `.tolist()`, `.cpu()` and `.numpy()` in rollout, action/route extraction and report paths (including source line families 178-199, 620-667, 795, 1117-1132, 1883-1991 and 2100). P5 exclusively owns this shared trainer file; P2 must provide a GPU-resident API and report the trainer boundary to P5, not edit it.
- `frc_defense/meco_supervised.py` serializes graph/candidate tensors and pulls losses/metrics to CPU. This belongs to model/data or training packages. P2 must include it in the residency inventory and assign each algorithmic tensor to its future owner; P2 does not edit it.
- `frc_defense/meco_counterfactual.py` contains CPU hashing/reward/route scalar and list conversions. Counterfactual/GAE semantics and trainer integration are owned outside P2; its tensor transfers must be classified in the global inventory and handed to the responsible P5 owner.
- `frc_defense/tensor_adstar.py`, `frc_defense/meco_pose_graph.py`, `frc_defense/meco_candidates.py`, `frc_defense/_tensor_3v3_observation.py` and `frc_defense/tensor_3v3.py` form the principal P2 source surface. Existing tests at the source base include `tests/test_tensor_3v3.py`, `tests/test_tensor_3v3_physical_constraints.py`, `tests/test_tensor_adstar_fused_route.py`, `tests/test_tensor_fuel_candidate_rank.py`, and `tests/test_fuel_physics_graph_device.py`.

These are search leads, not a complete audit or proof that all listed scalar pulls are on an algorithmic path. Pass 2 must trace imports, call sites, dynamic branches, custom extension dispatch and graph-capture behavior on its exact integration base. Do not interpret a test pass, device setting or model-only profile as end-to-end residency evidence.

## Dependencies and worktree

**Required bases/gates:**

1. Accepted amended overall plan and accepted P2 plan; exact isolated ZFS worktree; distinct implementer and reviewer. G0 is an execution/integration gate, not a source-authoring prerequisite. Before G0, no build/compile, test execution, simulator/profiler, numerical work, calibration/evaluation, dashboard, device/GPU use, training, checkpoint mutation, integration, merge or promotion is permitted.
2. Accepted P0a candidate and required human decisions. P0b is not a P2 prerequisite because it occurs after P1–P6 source freeze.
3. Accepted P1 implementation/interface head for the V4 typed event table and score/transition interaction. Record the exact P1 integration commit in the P2 implementation ledger before branching.
4. Pass 1 ACCEPT of this P2 plan and distinct named Pass 2 implementer and Pass 3 reviewer.

Create a new ZFS-backed worktree from the exact reviewed P1 integration head after source-authoring prerequisites are met: `.worktrees/autodrive-v4-p2-gpu-resident` on `codex/v4-p2-gpu-resident-YYYYMMDD`. Never branch from or inspect the active campaign checkout, or reuse its output directories. Before G0, P2 may only author and push coherent source/test slices and perform static review. After the exact head receives SOURCE/API ACCEPT, dependent owners may cite it for source authoring, but may not integrate or runtime-use it. After G0, run required tests and device/simulator evidence; final Pass 3 ACCEPT then permits integration of that exact head. Any shared-path conflict is resolved only after G0 in a reviewed integration branch, not by simultaneous edits.

## Finite Pass 2 allowlist

Pass 2 may edit only the following source paths:

- `frc_defense/meco_device.py` — require explicit HIP device identity and bind an assigned WX9100 UUID/index to each worker; reject CPU, P600, missing, ambiguous or changed device identity; provide a machine-readable device evidence record. Do not implement supervisor scheduling owned by P6.
- `frc_defense/tensor_3v3.py` — sole P2 owner for environment-step, active-mask, observation and target-handle integration changes after accepted P1 is integrated. Keep P1 score-event table work merged first; P2 does not change its event meanings or ABI.
- `frc_defense/_tensor_3v3_observation.py` and `frc_defense/_tensor_3v3_navigation.py` — observation/perception and environment route/cursor paths.
- `frc_defense/meco_candidates.py` — batched, device-resident candidate geometry, opponent prediction, target matching, legality, features and stable ordering.
- `frc_defense/meco_pose_graph.py` — legal graph construction and device-resident data products, preserving the reconciled Plan A/Plan C source interfaces.
- `frc_defense/tensor_adstar.py` — route search, validation, path extraction and fail-closed behavior with no CPU repair/fallback.
- `frc_defense/tensor_perception.py` and `frc_defense/tensor_fuel_candidate_rank.py` — only where the residency audit proves the required perception/candidate operation is algorithmic and on the P2 path; preserve public interfaces and HIP dispatch identity.
- `frc_defense/meco_gpu_residency.py` — new focused module for residency ledger/introspection, transfer/synchronization counters and fail-closed device evidence if current modules cannot provide these without coupling.
- `docs/MECO_GPU_RESIDENCY.md` — new implementation ledger documenting tensor owner, producer/consumer, shape/dtype, device, kernel, permitted transfer boundary and failure behavior across the full training/evaluation call graph.

P2 may edit only these existing test paths and add the stated new tests:

- Existing: `tests/test_tensor_3v3.py`, `tests/test_tensor_3v3_physical_constraints.py`, `tests/test_tensor_adstar_fused_route.py`, `tests/test_tensor_fuel_candidate_rank.py`, `tests/test_fuel_physics_graph_device.py`.
- New: `tests/test_meco_gpu_residency.py`, `tests/test_meco_candidates_gpu.py`, `tests/test_tensor_perception_gpu.py`.

No other files are authorized. In particular, P2 must not edit `frc_defense/meco_track_training.py` (P5-exclusive), `meco_supervised.py` (training/model data owner), `meco_counterfactual.py` (P5-owned PPO/end-cause/counterfactual integration), P1 score/event files after its accepted integration, P3/P4 model files, P6 supervisor/evaluator files, build configuration, dashboard source, or any output/checkpoint/campaign file. If a required source or test file lies outside this allowlist, stop and submit a new finite Pass 1 amendment; do not expand scope during implementation.

Custom HIP kernel source is not pre-authorized by this list. If P2 proves a custom kernel is needed beyond PyTorch HIP tensor operations, add the exact source/registration/test paths in a Pass 1 amendment before creating them.

## Required technical work

### 1. Full call-graph tensor residency ledger

Before changing kernels, document every tensor and data-dependent decision on the representative training and evaluation path: simulator physics/state; perception detections and tracks; target handles and validity; field/raster/legal geometry; candidate rows/features/masks; route query inputs, cost maps, feasibility/status and path points; score/transition events; policy inputs/outputs and chosen action; PPO/replay; metrics, safety gates, digest and qualification data. The ledger must name producing/consuming functions, tensor shape/dtype, owning package, device, kernel/extension, copies/synchronizations, reset/lifetime, overflow/error behavior and proposed owner for each open CPU operation.

For non-P2 owners, record a concrete handoff to P1, P3, P4, P5 or P6 with the source location and acceptance gate. The complete ledger must contain no unexplained algorithmic CPU read or device transfer. A permitted host transfer must be explicitly limited to static configuration before algorithm execution or opaque artifact/file I/O after computation; it may not inspect dynamic tensor values, drive a branch, select a candidate, compute route/reward/score, hash dynamic payload values or validate qualification.

### 2. Device placement and fail-closed behavior

Every worker must declare a pinned ROCm/HIP runtime and explicit WX9100 UUID and PyTorch device index. Map worker processes to the two approved WX9100s through a stable, non-oversubscribed policy coordinated with P6. Record UUID, PCI identity if available, runtime/driver/PyTorch/HIP versions, process ID, worker/track ID, device index and kernel execution identity. Device loss, unexpected device change, CPU tensors on the hot path, unsupported HIP kernels, accidental P600 placement or ambiguous affinity fails before training/evaluation proceeds. No silent CPU fallback or device migration is allowed.

P2 owns placement checks at environment, observation, candidate, route and custom-kernel boundaries. A per-process job may be bound to one WX9100; the complete accepted five-track run plan must assign work across both devices. Evidence must show the P2 hot path executing expected kernels and owning dynamic tensors on each WX9100, with transfer and synchronization traces. Do not assert that the two devices jointly execute one simulation unless the accepted integration actually does so.

### 3. Remove algorithmic host work from P2 paths

Vectorize dynamic candidate and opponent handling over fixed padded capacities/masks. Preserve deterministic tie-break order and no-wrap handles. Dynamic tensor cardinality may not control Python loops or allocate algorithmic result structures on CPU. Move geometric legality, target selection, route-status decisions, overflow/capacity checks, and path extraction to HIP-resident tensor operations or approved HIP kernels. Replace route repair with a GPU-resident method that preserves the independent footprint/swept-heading checks; when no valid device path is available, fail closed with a stop/invalid route row. Do not bypass existing collision, legal-transition, heading, stale-observation or collection constraints.

Replace `.item()`, `.cpu()`, `.tolist()`, `.numpy()`, `nonzero(...).tolist()` and equivalent data-dependent host reads in P2 hot functions with device-side masks/operations. Python integer control is permitted only for static shape/configuration or bounded static loops that do not depend on runtime tensor contents. Avoid scalar-read “optimizations” that still synchronize or move a device-derived branch to host. Preserve exact P0a/P1 score/event interfaces and V4 profile/field ordering.

### 4. Global boundaries and owner handoffs

P2 measures and reports all hot-path copies/synchronizations, including those in P5's trainer, supervised/data paths and counterfactuals. P2 does not silently absorb those owners' code. P5 must remove algorithmic trainer scalar pulls and route conversions under its exclusive `meco_track_training.py` authority; model/data owners must keep all policy learning inputs/results on the assigned device; P6 must keep qualification metrics/decisions GPU-native or clearly define opaque transfer boundaries consistent with the contract. Final global GPU-only acceptance waits for all owner packages and the P6 representative full-path trace.

## Acceptance gates

### Source and contract

- Exact P0a, P1 and V4 profile identities are bound; no field, ordering, count, dtype or semantic change is introduced.
- Tensor shapes/masks and target-handle validity/reset/no-wrap/stale semantics match the accepted declarations.
- Candidate ordering, policy candidate identity and executed-command binding remain deterministic and complete.
- Route output covers all requested rows; statuses distinguish feasible, infeasible and error; no capacity truncation or CPU route repair occurs; unsafe paths fail closed.
- Full-call-graph residency ledger has a named owner and closure plan for every dynamic tensor and every boundary transfer.

### Tests and device evidence

- Focused tests cover device mismatch/rejection, CPU tensor/CPU fallback rejection, track and target masks, cardinality at 0/1/max, stale/reset handles, deterministic candidate ties, opponent/fuel validity, route success/no-path/blocked-start/blocked-goal/capacity errors, footprint/swept-heading checks and graph-capture/device-loss behavior.
- Test files may be authored before G0, but no test may execute until this plan is accepted and G0 is evidenced. Numerical algorithms and simulator state must run on the two WX9100 devices; CPU may supervise processes or handle immutable file/network I/O only.
- Capture profiler/device traces for both WX9100s over representative P2 environment, observation, candidate and route work. Include per-kernel device UUID, tensor device/shape/dtype, transfer bytes, synchronization and launch data. Include a test/trace proving CPU and P600 fallback is rejected.
- P2 unit/regression tests that exercise simulated robot, perception, route, collection or scoring behavior must be accompanied by a representative seeded end-to-end simulator evaluation exercising that behavior. Inspect the resulting simulation record/replay and report scenario, seed, replay ID and outcome; tests alone do not establish simulation behavior.
- Before producing or reviewing any dashboard replay, restart dashboard services on the current accepted integration checkout after source changes; confirm backend and frontend asset versions, active service state and successful endpoint responses. Generate a new replay and inspect provenance; mark old-source replays outdated.
- Report all tests, simulator evaluation, replay/scenario/seed, dashboard freshness and endpoint checks, GPU UUID/runtime placement and trace results. If simulator/dashboard/device validation is gated or unavailable, identify the missing evidence; do not mark P2 fully validated.

### Integration gate

P2 has two review dispositions. Before G0, an independent source reviewer checks exact pushed commit/blob hashes, allowlist, P0a/P1 bindings, source semantics, candidate API shape and static residency ledger and may write SOURCE/API ACCEPT or REVISE. SOURCE/API ACCEPT pins a candidate for dependent authors; it is explicitly not package-final Pass 3 ACCEPT and does not allow dependent-package integration or runtime use. After G0, a distinct independent final reviewer checks the same exact head plus executed tests, both-device traces, seeded simulator evidence and current-dashboard freshness where replay is required. Only final Pass 3 ACCEPT permits integration of that exact head; any source change requires new review. P2 acceptance does not by itself authorize five-track training or claim global GPU-only completion; all P1–P6 owner gates and final P0b are still required.

## Risks and stop conditions

- **CPU repair fallback is safety-relevant:** replacing local route repair must retain swept-footprint and heading checks; if no device implementation meets them, fail closed and stop rather than restoring CPU repair.
- **Dynamic Python list construction:** candidate enumeration and opponent matching currently depend on tensor-derived values. Fixed-capacity padding/masks may affect exact ordering; verify deterministic parity against seeded reference records.
- **Other-package ownership:** the complete graph includes scalar reads in trainer, supervised data and counterfactual paths. Record handoffs and block global residency completion until their owners fix them; do not expand P2 scope.
- **P1/P2 shared environment boundary:** P1 is accepted and integrated first. P2 is sole owner of `tensor_3v3.py` changes after that head; do not rebase P1 modifications away or alter score-event meanings.
- **WX9100 launch/runtime uncertainty:** only validated HIP kernels and explicit device IDs can pass. Do not use CPU, P600, unpinned GPU, stale extension or silent fallback to keep a run alive.
- **G0/live campaign:** source/test authoring and static review may proceed only after plan acceptance and prerequisites; no execution or integration before user stop and independent release of both WX9100 leases. Planning and source review do not imply those facts.

## Three-pass roles and disposition

- **Pass 1 planner:** `plan_a_owner`; cannot implement or review P2.
- **Pass 2 implementer:** unassigned; must work only in the fresh worktree/branch above and only inside the finite allowlist.
- **Pre-G0 source/API reviewer:** unassigned and distinct from planner and implementer; records SOURCE/API ACCEPT or REVISE against exact head. This does not authorize integration or runtime use.
- **Post-G0 final Pass 3 reviewer:** unassigned and independent; reviews exact head and all required runtime evidence, then persists final ACCEPT/REVISE at a stable review path.

Planner disposition: **PENDING independent Pass 1 review.** This plan is not source/test/runtime authorization and claims no implementation, GPU residency, simulator result, dashboard validation, campaign status, human approval, or qualification result.
