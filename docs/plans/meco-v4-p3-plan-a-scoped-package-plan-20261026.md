# P3 Plan A A1/A2/A3 scoped package plan

**Date:** 2026-10-26  
**Status:** G0/source-authoring gate amendment candidate; original P3 plan received Pass 1 ACCEPT; this amendment requires fresh independent exact-head review
**Planner identity:** `/root/closure_wave1_reviewer` (author for this package plan only; not implementer or reviewer)  
**Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`  
**Original accepted overall plan:** `6325920252773cfa05d63167ba3c29a90debdb01`, file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`; Pass 1 ACCEPT record `be845d7fb2d2ece861dcd158ac8b8dd89e3dc59d`, record SHA-256 `5a6e94f978c8a7e18d9623897ceba6f32f5b5eadb0aba6605f21c9d1e941a886`
**Overall-plan G0/source-authoring amendment:** `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; pending fresh independent review
**Accepted P0 package plan:** `13dbd7f8f953995850e811202f7e9c5264c49dca`, file SHA-256 `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`; Pass 1 ACCEPT record `037f0f9386cc0efadb1fb54efe5f38f097926d0e`, record SHA-256 `1ee70a6e9d24dc5201da5eec446820167cb8c4d4436fbbe013eb16de66b0280d`  
**P0 G0/source-authoring amendment:** `3185de5d9108717ff6eb5eb998ecfb41ad356057`, file SHA-256 `51bfcb6739153b6115187c15b381698bfa2510de4c8759e4376198798ba13556`; pending fresh independent review
**Accepted P1 package plan:** `53686a254318484f9c6bc3496ecc1c461572c53a`, file SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`; Pass 1 ACCEPT record `854d5286947074ed33f7ac059d67f2f9ccbf9257`, record SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f` (P1 implementation API head is still a future Pass 3 dependency)  
**Accepted reconciliation:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; Pass 3 ACCEPT record `5cb1a2d525168ff0443089cf329e66c6ecb917b8`, record SHA-256 `0df700dcaef1ed35672efc468bdcfd78fac829c31bcf7a88b3c51f1951f06c32`  
**Reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`  
**V4 proposal/raw envelope profile digest:** proposal commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`; SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`  
**Plan A owner response:** `03b7a9f5cbb161a2b30563e76a360619df144849`, `docs/plans/v4-owner-responses/plan-a.md`, SHA-256 `e6863a2b3b52fc2526062ac081cec0b153742d0f32344d8006d766cfad74df21`; exact Pass 3 ACCEPT record SHA-256 `9a1eea2c57c1c20b59d6fdfd11e7ffb697aa77b0093f7fcb40a44978233f38e2`.

## Objective, authority, and gates

Define the bounded P3 work that turns the adopted Plan A contract into a GPU-resident support/scoring adapter for A1, A2, and A3. P3 exposes an exact typed interface for the P5 trainer integration; it does not own or edit the shared trainer, simulator legality/safety authority, the P2 graph/observation/route core, the P1 V4 producer/reader, or the P6 qualification runner. It does not change the accepted profile, reward estimand, PPO/RNG semantics, or the Plan C path.

This is a documentation-only package plan. It authorizes no work until this amended exact plan receives independent Pass 1 ACCEPT, P0a decisions are accepted, and exact P1/P2 candidate interfaces receive independent SOURCE/API ACCEPT. Then source/test-file authoring and static source review may proceed in an isolated ZFS worktree from the approved source base before G0. Pre-G0 SOURCE/API ACCEPT can pin a candidate for dependent source authoring but is not final Pass 3 ACCEPT and does not authorize integration or runtime use by dependent packages. Builds/compilation, test execution, simulator/numerical work, device/GPU use, dashboard, calibration/evaluation, checkpoint mutation, integration, merge, training or qualification remain prohibited until G0. G0 requires the user's manual stop of PID `2750253` and independent confirmation that both WX9100 leases are released. Do not touch the campaign or its dashboard, workers, checkpoints, outputs, devices or leases. The pinned reconciliation/profile hashes may be cited before G0, but the absent artifact bytes must not be added to or required in a source-only worktree; the reviewed contract-anchor bridge is a post-G0 prerequisite for integration/validation. P0b remains downstream: P3 validation does not require final calibration, while fresh training and P6 runtime qualification do.

G0 is the execution/device/integration boundary: the user manually stops active diagnostic campaign PID `2750253`; both assigned WX9100 leases are independently confirmed released; and the authorized maintainer separately authorizes execution/integration. The docs-only reconciliation bridge is reviewed after G0 and before validation/integration. Source-only worktrees use the approved source base and must not include or depend on the absent artifact. Do not inspect, stop, signal, reconfigure, evaluate through, or otherwise touch the active campaign, its dashboard, workers, checkpoints, outputs, leases, or devices. This plan does not claim G0 or P0a is complete.

## Accepted Plan A contract

The accepted Plan A owner row adopts the exact V4 proposal without changing bytes. Preserve these capacities and field meanings:

- Profile identity is one track per run (`N=1`), six robot slots (`S=6`), 32 transitions (`R=32`), 12 macro ticks (`T=12`), and the reconciled observation raster `[6,25,51]` float32 with seven globals in the declared order. Plan A never uses Plan C's K=16 candidate cap.
- Plan A has a padded graph capacity of 12,096 nodes. The runtime legal subset is the full-capacity `node_valid` mask; padded capacity does not imply all nodes are legal. There are 96,768 ordered support rows, exactly eight descriptor rows per padded node, ten directed edge slots per padded node, and 14 ancestor-jump levels. Preserve the adopted node-major behavior/descriptor ordering and support row mapping; do not reinterpret the eight rows as eight behavior classes.
- The V4 dynamic support/search data are `support_valid:bool[96,768]`; `distance_m`, `route_cost`, and `parent_edge_m` as float64 `[12,096]`; `parent_node` and `traversal_order` as int64 `[12,096]`; `ancestor_jumps:int64[14,12,096]`; `node_utility:float32[12,096]`; and `behavior_logits:float32[5]`. The three float64 cost columns are an adopted serialization promotion, not a claim about current computation dtype. Keep profile-fixed capacity, complete ordered evidence, and the accepted digest boundaries. Any field, type, capacity, order, byte, or digest change stops for reconciliation.
- The accepted common selected-command contract remains the source of command identity: preserve `policy_local_index`, source/global identity as defined for Plan A, `action_id`, goal pose, target identity, override/fallback/route status, route and risk costs, `legal_node_id`, `behavior_id`, `descriptor_id`, `selected_U`, and `selected_P`. P3 returns the support-index mapping and model values needed by P5 to form this command. P5 alone links the exact sampled action and old log probability to the PPO transition row; P3 must not reconstruct or substitute that probability.
- Model architecture is fixed across the three tracks: A1 raster multiscale U-Net plus GRU; A2 structured SE(2) U-Net, regional attention, and GRU; A3 hierarchical graph message passing, regional attention, and GRU. They share observation availability, deterministic legal support, native safety, command identity, and PPO/RNG interfaces. Utility and Plan A's nonnegative traversal-penalty semantics remain distinct from route cost and risk cost; each component is applied at its adopted boundary exactly once. P3 cannot add a per-node serialized field absent from the accepted profile or revise scale/units locally.
- The legal SE(2) graph support/search path is not the simulator's execution route. Preserve deterministic graph ordering, root connector, reachability, and path evidence; the native executor, AD* route, action remasking, collision and safety checks remain authoritative for execution. Learned outputs may rank only already-legal support and cannot widen masks or authorize an unsafe action.

The cited Plan A response supports the shape, ordering, current source dtype, and digest inventory, but does not prove hardware memory admission, device parity, simulator correctness, or qualification. This package must establish those properties for its own path.

## Source evidence and limits

At the pinned source base, `frc_defense/meco_tracks.py` already defines the A1, A2, and A3 model classes (`A1MultiscaleUNetGRU` near line 113, `A2StructuredSE2UNetGRU` near line 204, and `A3HierarchicalSE2GNNGRU` near line 241). A1 returns dense utility/penalty outputs and five behavior logits; A2/A3 return node utility/penalty outputs and five behavior logits. This plan consumes those current model contracts; it does not claim they satisfy the V4 adapter or require model-weight/source changes.

The existing `frc_defense/meco_plan_a_policy.py` support builder currently performs scalar extraction and host-list conversion (`argmin().item()`, graph `.cpu().tolist()`, and Python Dijkstra traversal); its support digest copies tensors through CPU. Those are observed base behaviors, not accepted GPU residency. P3 owns replacing/encapsulating this Plan A support adapter on device after P2's graph interface is pinned. P2 remains owner of graph construction and observation/route producer APIs. Exact source and shape references must be refreshed on the accepted integration head before implementation; no source inference is authorized from the citations alone.

## Dependency gates and handoffs

P3 source-only authoring depends on accepted P1 and P2 candidate interfaces; it may not infer API acceptance from a branch tip or unreviewed draft. Before P3 Pass 2 begins, the coordinator must add exact commit, file/blob hash, and independent SOURCE/API ACCEPT record for each dependency to the implementation ledger. These are candidate source-interface pins only; dependent work may not integrate or runtime-use them before G0 and final package ACCEPT:

| Dependency | Required accepted head/interface before P3 Pass 2 | Status in this candidate |
|---|---|---|
| P1 V4 schema/event/artifact | Accepted P1 package plan `53686a254318484f9c6bc3496ecc1c461572c53a` / SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`; exact Pass 1 ACCEPT record `854d5286947074ed33f7ac059d67f2f9ccbf9257` / SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f`. Before P3 Pass 2, pin a candidate P1 typed profile declaration, fixed support/search field registry, batch/manifest API and producer/reader digest boundary with independent SOURCE/API ACCEPT. P5 consumes the API only after post-G0 final ACCEPT and integration. | Package plan accepted; P1 implementation/API head is not yet pinned. P3 must wait for its SOURCE/API ACCEPT. |
| P2 environment/observation/graph/route | Accepted P2 plan `8d0727f10d46bfc5fee2810a1ec7fa7987ecaf91`, original Pass 1 ACCEPT record `0292f01638bf3180b4952721f5692a4446841013`; gate amendment head `1f918a9e0f99ec49a0baf4c3ce69649c4f095a2a`, file SHA-256 `ab640e4fdcfe63c8f1b97aa421297413d2c1304830031d81bb1a9c6cbdd07f50`, is pending fresh review. Exact P2 candidate API must receive SOURCE/API ACCEPT and bind device, shape/dtype, `node_valid`, coordinates/headings, edges/edge masks, graph identity/digest, and route-cost/feasibility handoff. P3 may consume this interface but may not duplicate or modify P2 graph/route producers. | P2 planning ACCEPT exists; amendment review and implementation/API SOURCE/API ACCEPT remain required before P3 source-only authoring. No final P2 package ACCEPT or integration is implied. |
| P0a | Accepted documentation-only preregistration candidate plus authorized maintainer decision record and exact Pass 3 ACCEPT. Provides the common case/seed/control schedule and freezes evaluation methodology. | Package plan accepted; P0a candidate/decision deliverable remains a future gate. |
| P5 trainer | P3 typed adapter is independently accepted and integrated before P5 consumes it. P5 remains sole owner of `frc_defense/meco_track_training.py`, PPO sampling, log-probability recording, transition assembly, optimizer/RNG/checkpoint atomicity. | P5 implementation interface not yet pinned; P3 must publish a standalone API and avoid trainer edits. |

P0b final calibration is not a P3 source or regression-validation dependency. It follows all reviewed P1–P6 source freeze and gates fresh training/Stage 1 qualification. P4 Plan C is not a P3 code dependency; P3 uses the P2 shared observation/legal-graph/route contract and preserves the reconciled A/C separation. If P1 or P2 changes any required interface after P3 begins, stop and obtain a reviewed P3 amendment or rebase/review against the newly accepted interface before continuing.

## P3 source interface and behavior

P3 creates a standalone Plan A adapter that accepts P2-owned device-resident graph/observation tensors and the existing A1/A2/A3 model outputs, applies the adopted support mapping and masks, and returns typed model/support values to P5. It does not place values into a trainer rollout or artifact itself.

The P3 interface must:

1. validate that one world uses the exact pinned profile and that graph tensors, node validity, edge masks, model outputs, dtypes, devices, and snapshot/profile identity agree before scoring;
2. construct or consume deterministic Plan A support in the adopted node-major/eight-descriptor order, retain all 96,768 capacity rows, and represent invalid/padded rows with the adopted mask/zero semantics;
3. adapt A1 raster utility/penalty and A2/A3 node utility/penalty outputs onto the exact 12,096-node profile using the P2-owned legal-node lookup, preserving the accepted positive penalty constraint, behavior-logit order, hidden-state lifecycle, and fail-closed behavior;
4. combine only the adopted utility, traversal penalty, route, and risk terms with their declared units and once-only application. Return component values separately so P5 can bind common command fields and P5/P6 can audit score decomposition. Do not fold route or risk cost into model utility or apply learned penalty twice;
5. return logits/masks and a stable support-local-to-command mapping on device. P5 owns stochastic sampling, RNG draws, exact sampled action/old-log-probability linkage, command-to-row assembly, checkpoint and update transactions;
6. keep all P3 algorithmic operations, including graph-support lookup/gather, utility/penalty projection and integration, legal masking, scoring, finite validation, and selection-facing tensors on the selected WX9100 device. Only the minimal final command handoff explicitly allowed by the accepted P2/P5 boundary may cross to host; no per-node `.item()`, `.tolist()`, `.cpu()`, NumPy conversion, Python node loop, or CPU/P600 numerical fallback; and
7. fail closed on empty/invalid support, profile/snapshot mismatch, out-of-range descriptors, non-finite valid values, overflow, or stale model output. A failure cannot widen support or mutate PPO/RNG/checkpoint state; P5 owns and tests caller-side atomicity.

The three network implementations currently reside in `frc_defense/meco_tracks.py`. This P3 allowlist deliberately does not edit that shared five-track module: P3 consumes the existing A1/A2/A3 output contract and must stop for a plan amendment if implementation requires changing those model definitions or the shared factory. P4's Plan C classes remain untouched. P3 also does not edit P2 graph producers or P5 trainer callers.

## Finite Pass 2 allowlist

P3 Pass 2 may modify only the following paths in a fresh implementation worktree based on the exact accepted integration head. No additional path is implied by “integration”; if discovery proves another path necessary, stop and request a reviewed plan amendment first.

| Path | P3 authority |
|---|---|
| `frc_defense/meco_plan_a_policy.py` | Plan A support ordering, masks, graph-path evidence adapter, utility/penalty projection and support digest consumer using P2's accepted interface. Do not edit P2 graph construction. |
| `frc_defense/meco_plan_a_v4.py` | New standalone typed Plan A V4 model/support adapter and device-only scoring interface consumed by P5. No PPO sampling, trainer rollout, manifest serialization, or checkpoint mutation. |
| `tests/test_meco_plan_a_v4_contract.py` | New deterministic CPU-free contract/reference-fixture tests for the exact capacities, row order, masks, model output shapes, cost units, digest inputs and fail-closed cases. Fixtures may use CPU to generate expected bytes only; implementation validation of algorithmic GPU behavior remains on device. |
| `tests/test_meco_plan_a_v4_simulator.py` | New seeded end-to-end Plan A simulator evaluation harness/integration assertions for A1/A2/A3, paired controls and record/replay inspection. |

Excluded shared and upstream paths include `frc_defense/meco_track_training.py` (P5 only), `frc_defense/meco_pose_graph.py`, `frc_defense/tensor_3v3.py`, `frc_defense/meco_candidates.py` and route/native kernel files (P2), `frc_defense/meco_tracks.py` (shared A/C factory and model definitions), P1 schema/event/artifact modules, P6 supervisor/qualification/dashboard modules, campaign/checkpoint/output data, and all unlisted tests. No HIP/C++ extension is allowlisted; if profiling proves one is necessary, request an exact ABI/path amendment before editing it.

## Delivery order and commit/review cadence

P3 source-only work starts only after this amended plan receives Pass 1 ACCEPT; P0a decisions and P1/P2 candidate interfaces receive the required independent acceptance; and distinct implementer/source reviewer/final reviewer identities are assigned. G0 is not required for source/test-file authoring. The implementer uses a new ZFS-backed worktree/branch (proposed `.worktrees/autodrive-v4-p3-plan-a-impl` / `codex/v4-p3-plan-a-impl-YYYYMMDD`) from the exact approved source base plus only source-only candidate interfaces. Do not include or depend on the absent reconciliation artifact or contract-anchor bridge pre-G0. This Pass 1 planning worktree is not reused for implementation. Until G0, author and statically review only allowlisted files; do not build, compile, execute tests, run numerical work, use devices, or integrate/merge.

1. Record P1/P2 exact candidate dependency commits, hashes, interface documents, and SOURCE/API ACCEPT reviews in the ledger; obtain the P0a decision and acceptance record. G0 evidence is a later prerequisite for builds, test execution, device/simulator validation, dashboard, integration and final Pass 3.
2. Implement and review the Plan A standalone adapter in coherent slices: accepted input validation/support mapping; A1/A2/A3 projection and cost decomposition; device scoring/mask/digest; then simulator evidence. Push each slice before starting a dependent slice. The exact P1/P2 interfaces and V4 field registry are immutable for the slice.
3. Before G0, an independent source reviewer returns SOURCE/API ACCEPT or REVISE against the exact pushed P3 API head, checking scope, interfaces and static evidence. SOURCE/API ACCEPT only pins a candidate for dependent source authoring; it does not allow integration or runtime use. A changed P3 head requires fresh source review.
4. After G0, complete P3 tests, per-kernel placement/residency records, seeded simulator records/replays, and current-dashboard evidence. An independent final Pass 3 reviewer checks the exact same head and evidence. Only final Pass 3 ACCEPT allows integration; P5 may runtime-use the API only after that integration is reviewed. Any changed API or merge conflict receives fresh exact-head reviews.

The P3 Pass 3 reviewer must not be the planner or implementer. The reviewer checks the exact commit/blob, allowlist, accepted P1/P2 interface pins, Plan A profile/model/command contract, GPU evidence, tests, seeded simulation/replay, current dashboard evidence, and that P1/P2/P5/P4/P6 boundaries are unchanged. REVISE returns only P3 to its owner; no training, ABI publication, or qualification is authorized by a P3 ACCEPT.

## Validation and acceptance evidence

### Software contract fixtures

- Confirm the exact support/profile arithmetic: 12,096 padded nodes, 96,768 ordered support rows, eight descriptor rows per node, ten edge slots per node, and 14 ancestor levels. The P3 contract must reject any capacity or field-order drift.
- Golden fixtures verify node-major behavior/descriptor row order, `node_valid` and `support_valid` masking, deterministic tie order, parent/traversal/jump semantics, legal action-to-command mapping, snapshot generation binding, and empty support/fallback behavior.
- Verify A1 raster-to-node projection and A2/A3 node output shapes against the exact accepted P2 graph adapter; verify utility and nonnegative penalty are finite on valid rows, invalid padded rows cannot be selected, and route/risk/penalty components are each applied exactly once.
- Verify selected support-local identity carries through to the common command fields and P5 consumes exact support/API identity. The P3 tests do not own the PPO sampler or test trainer atomicity; P5 does.
- Prove P3 source code has no algorithmic CPU path, device-to-host synchronization in scoring, silent fallback, P600 substitution, or Plan C K-width cap. CPU expected-value helpers are fixture-generation references only.

### GPU-only and end-to-end validation

No device validation may run before G0 and final P1/P2 package acceptance. Run on both assigned WX9100 devices and record UUID/runtime, source/API hashes, kernel/module placement, memory peaks, transfers/synchronizations, and per-track execution. Fail closed if either device is unavailable; do not use CPU or P600 fallback. Before G0, static source/API review only is allowed.

Every test that checks Plan A decision or simulated game behavior must be accompanied by a representative seeded end-to-end simulator evaluation for A1, A2, and A3. Use the P0a-frozen scenarios/seeds and matched D0/D1/D2 controls under the same profile, available observations, safety contract, and resource budget. Inspect the actual simulation record/replay for selected versus executed command, legal route, support mask, U/P and route/risk decomposition, safety/remasking/fallback, episode end cause, FUEL award/ferry outcome where exercised, and failure status. Record test names, case/category, exact seeds, checkpoint/source/config hashes, replay IDs and outcomes. Tests alone do not establish simulated behavior.

Before generating or reviewing a dashboard replay, restart relevant dashboard serving services from the exact current integration checkout. Verify backend service PID/worktree/head, frontend asset hashes against that checkout, active services, and successful relevant endpoints. Generate a fresh replay only after restart, verify replay provenance, and mark old-code replay IDs outdated. Report which tests and simulations ran, scenario/seed/replay identifiers/results, device UUID/runtime, and dashboard currentness/endpoints. If a required simulator or dashboard check cannot complete, state that evidence is missing and validation remains incomplete.

P0b calibration is not run in P3. P3 validation cannot be called Stage 1 qualification, trusted held-out comparison, or training progress; those remain P0b/P6 gates.

## Risks, stop conditions, and nonclaims

- **P1/P2 interface not yet pinned:** do not start P3 Pass 2 until both accepted package/API heads and exact review records are recorded. A changed dependency requires a P3 rebase/review or plan amendment.
- **Model contract mismatch:** if current A1/A2/A3 output shapes or utility/penalty semantics cannot satisfy the accepted profile through the finite adapter paths, stop and request a scoped P3 amendment. Do not edit `meco_tracks.py` or change the ABI inside this plan.
- **Memory/residency:** the full V4 payload is 314,469,051 bytes before workspace/staging buffers. Measure peak allocation on both WX9100s and preserve full fixed capacities. Any need to truncate, reduce support, transfer algorithmic values to CPU, or alter profile requires reconciliation.
- **Support and execution are separate:** Plan A graph support/search evidence cannot replace the P2/native executor's actual route and safety checks. A selected support row is not proof the action was executed.
- **Trainer ownership:** P5 alone edits `meco_track_training.py`, PPO/RNG/update/checkpoint behavior, and trainer integration tests. P3 exposes only the typed standalone adapter.
- **Campaign boundary:** no interaction with the active campaign or its data/devices is authorized. G0 must be evidenced independently; no elapsed-time or process-absence inference. Source authoring/static review can precede G0 only in the approved source-base worktree; all execution and integration remain blocked.

This plan claims no implementation, GPU residency, simulator outcome, replay freshness, dashboard currentness, training progress, qualification, or Stage 1 passage. No tests, source edits, runtime, simulator, GPU, dashboard, or campaign action has been performed for this document.

## Three-pass disposition

- **Pass 1 planner:** `/root/closure_wave1_reviewer`, author of this package plan only.
- **Pass 2 implementer:** unassigned; must be distinct from planner and Pass 3 reviewer.
- **Pre-G0 source/API reviewer:** unassigned; independently reviews exact source/API candidate and records SOURCE/API ACCEPT or REVISE. This is not final ACCEPT.
- **Post-G0 final Pass 3 reviewer:** unassigned; must independently review the same exact pushed head plus all validation evidence above. Only final ACCEPT permits integration.

Planner disposition: **PENDING independent Pass 1 review of this gate amendment.** The original P3 package plan received Pass 1 ACCEPT. This amendment requests bounded source-only authoring before G0, not execution or integration authority.
