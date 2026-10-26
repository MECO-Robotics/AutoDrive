# P4 Plan C C1/C2 scoped package plan

**Date:** 2026-10-26  
**Status:** Pass 1 planner-authored candidate; pending independent exact-head review  
**Planner identity:** `plan_c_owner` (planner only; not implementer, reviewer, or human approver)  
**Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`  
**Accepted overall plan:** `6325920252773cfa05d63167ba3c29a90debdb01`; file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`  
**Accepted P0 package plan:** `13dbd7f8f953995850e811202f7e9c5264c49dca`; file SHA-256 `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`  
**Accepted P1 package plan:** `53686a254318484f9c6bc3496ecc1c461572c53a`; file SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`  
**Accepted P2 package plan:** `8d0727f10d46bfc5fee2810a1ec7fa7987ecaf91`; file SHA-256 `5eddd42914ab74426257ab8c1ff883b25c0c39aaa40bf7c40ccc2e363fbb3957`  
**Accepted P3 package plan:** `8b641ccc56b7c73a880a095b3c6940cb03b617f3`; file SHA-256 `6958cb883a339a251e6069378d8ebb853be97e4965d30ca0eba8d3003fbc0a9b`  
**Accepted V4 reconciliation:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`; artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`  
**Canonical profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`  
**V4 proposal/envelope digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`  
**Plan C owner response:** `6106e2d236af4a0891d8d72a578ce748aece5743`, file SHA-256 `4bad45f3bb5dcbe87907f666805166364a9f424b2fd95a59b1dd23ef3c93c610`, exact Pass 3 review ACCEPT record SHA-256 `cb16ab85da0cd4778c5fe09042ccc3150ece64139a7ed76dee6d7e5b07c3f1a4`.

## Objective and authority

Implement the accepted Plan C policy/model path for C1 and C2 against the exact reconciled `E=504`, `K=16` profile. P4 consumes observation, fixed candidate descriptors, route results, masks and cost tensors from P2; it does not create or mutate the candidate generator, observer/target handles, route/search kernels or shared PPO trainer. P4 provides identical policy-facing row maps, descriptors, candidate validity and route/risk inputs to both architectures and to D1. C1 scores engineered candidate features. C2 additionally encodes the raster and uses candidate-to-map attention. Both return gross strategic utility through one deterministic Plan C arbiter that applies route and risk costs once, respects the accepted feasible-action mask and fallback/hysteresis contract, and leaves native execution safety authoritative.

This document authorizes only a future bounded source package after independent Pass 1 ACCEPT and all stated gates. It does not authorize source edits, tests, builds, simulator, GPU, dashboard, training, campaign, qualification or checkpoint activity. Do not inspect, stop, signal, reconfigure or evaluate through the live diagnostic campaign PID `2750253`, its workers, outputs, dashboard, checkpoints, device leases or GPUs. G0 requires the user's manual campaign stop, independently confirmed lease release, maintainer authorization and reviewed contract-anchor integration before any package execution.

## Pinned Plan C contract

The accepted reconciliation and Plan C owner response fix the following contract for this package:

- Profile dimensions include `N=1`, `S=6`, `R=32`, `T=12`, `E=504` and `K=16`. Plan C has exactly 537 source rows, 20 raw features, 40 model columns (raw values followed by 20 missingness indicators), 9,681 ordered route sink rows, 152 route chunks of at most 64 entries, and 73 path points. The accepted V4 payload and row bytes remain unchanged.
- Raw feature order is: `action_index`, `sensor_track_buffer_index`, `goal_x_m`, `goal_y_m`, `heading_bin`, `behavior_index`, `field_length_m`, `field_width_m`, `target_distance_m`, `nearby_sensed_track_count`, `track_confidence`, `track_age_s`, `opponent_distance_m`, `route_time_s`, `legal_node_index`, `route_distance_m`, `risk_penalty_points`, `target_heading_rad`, `is_continue`, `commitment_switch_cost`. The missingness bits follow in that same order.
- Source rows 0–7 are base actions (kind 1); 8–511 are sensed-target rows (kind 2), target observer buffer slot `r-8`; 512–535 are approach offsets (kind 3), six variant-major groups of four action rows (512–515 variant 1, 516–519 variant 2, 520–523 variant 3, 524–527 variant 4, 528–531 variant 5, 532–535 variant 6); row 536 is continuation (kind 4). Oracle candidates have a separate domain and never enter the 537 policy rows.
- Source identity code is `r+1`, in 1–537; zero is invalid. Descriptor schema is `meco-c-source-row-v1`; descriptor fields are `kind:u16`, `variant:u16`, `source_identity:i64`, `action_index:i16`, `target_identity:i64`, `goal_ref:i32`, `route_ref:i32`, `valid:u8`. The accepted target handle contract remains generation-safe; a buffer slot alone is not a durable target identity.
- Ordinary and continuation views are separate `[1,16,40]` tensors. `ordinary_row_map[j]` maps ordinary local `j∈[0,16)` to the stable rank-selected source row. `continuation_row_map[0]=536`; no continuation view is active when continuation is invalid. Selector outputs have 17 slots; local 0–15 map through the ordinary map, and local 16 maps through the continuation map.
- Preserve `candidate_valid[r]=route_available[r]` before the K-prefix mask; `candidate_model_input_mask[r]=candidate_policy_mask[r]`; `candidate_selection_rank[r]=selection_rank[r]`; and `candidate_policy_eligible[r]=valid_mask[r]=route_available[r] & candidate_policy_mask[r]`. Effective ordinary query mask is `candidate_valid & candidate_model_input_mask`; the continuation query mask has only row 536 when valid. Preserve the large rank sentinel for ineligible extras and continuation rank `2**60`. Masked feature padding is finite zero, unused row-map entries are `-1`, and maps must be in range, unique where required, and mask-consistent.
- Route sink order binds sink row `q` to source-global row `r=q` for policy rows, carries source row in `policy_source_row` and `route_ref`, maps continuation to row 536, and retains inactive and oracle capacity using separate sentinel domains. Consume all 9,681 sink rows and their support, status, feasibility, target handle, route variant, route cost and risk provenance; do not truncate to selected K rows or infer that 152×64 capacity means extra valid sinks.
- C1 and C2 share exactly the same candidate generator, descriptor maps, feasible policy set, route/risk values, deterministic D1 comparator, fallback and command identity. Model outputs are gross utility. Apply route cost and risk cost exactly once in the common deterministic score/arbiter path; keep their units and values inspectable. No Plan A traversal penalty is added. Model outputs cannot widen the P2 feasible mask or bypass native safety, fallback, preemption or execution checks.

These are adopted target semantics, not evidence that source base `6ed961a` already implements them.

## Source evidence and limitations

Read-only inspection at `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22` found:

- `frc_defense/meco_tracks.py:45-55` declares the current 14 `SAFE_ACTION_FEATURES` followed by six `EXTRA_FEATURES`, and derives raw width 20/model width 40.
- `meco_tracks.py:276-297` defines C1 as a global/candidate feature encoder with recurrent context and a candidate utility/uncertainty head. `299-328` defines C2's raster convolutional encoder, candidate encoder, multi-head cross-attention, recurrent state and candidate utility/uncertainty head. `331-341` registers C1/C2 in the shared factory.
- `meco_tracks.py:442-454` builds candidate values/missingness with Python lists and `torch.tensor`; `457-499` adapts variable candidate rows and scores one decision at a time. This is not the fixed V4 two-view GPU interface.
- `meco_tracks.py:681-704` computes existing D1/D2 comparator scores by iterating Python candidate rows. `candidate_policy_logits` subtracts provided route/risk terms, masks invalid rows and fails when no feasible candidate is present (`meco_tracks.py:420-439`). P4 must preserve D1's pinned semantics while providing a device-resident adapter and common C1/C2 scoring path; it may not redefine the D1 estimand or create a distinct candidate universe.
- `frc_defense/meco_candidates.py:383-407` accepts `candidate_k` in `{8,16,32}`; lines `425-475` construct dynamic Python specs; lines `575-650` build Python feature and route records; lines `673-743` select a variable K prefix and append continuation. The fixed 537-row descriptor axis and full route sink are not present in this source base. `meco_candidates.py` is P2-owned and therefore strictly upstream of P4.

The source row/descriptor mappings and exact two-view semantics come from the accepted Plan C row and reconciliation, not from the cited older source. The observed CPU list/scalar construction is a P2 interface gap; P4 must not patch or duplicate it. P2, P1, P3 and P4 implementation/API heads are future dependencies even though their package plans are accepted.

## Dependency gates and ownership boundaries

P4 Pass 2 requires all of the following exact reviewed heads, recorded in the implementation ledger before branching:

| Gate | Required evidence before implementation | Status at this plan head |
|---|---|---|
| G0 and P0a | Overall-plan G0 evidence; accepted P0a preregistration and required maintainer decisions. P0b is downstream of the P1–P6 source freeze and does not block P4 source implementation/tests. | Not claimed complete; blocks implementation. |
| P1 | Accepted standalone transition/feature/artifact schema API, exact file/blob hashes and Pass 3 review, including Plan C field and digest bindings. | P1 plan accepted; implementation/API head not yet available. |
| P2 | Accepted device-resident observation, target-handle, fixed candidate/descriptor/mask/rank/map and full route-sink producer API, with exact shape/dtype/device/error contracts and source/blob hashes. | P2 plan accepted; implementation/API head not yet available. P2 exclusively owns `meco_candidates.py`, route/search and observation producers. |
| P3 | Accepted Plan A interface head and review, with documented ownership boundaries for any shared code. | P3 plan accepted; implementation/API head not yet available. P3 owns only its Plan A adapter and must not change Plan C behavior. |
| P4 plan/roles | This plan Pass 1 ACCEPT; distinct named Pass 2 implementer and Pass 3 reviewer. | Pending exact-head Pass 1 review; roles unassigned. |
| P5 | P5 remains exclusive owner of `frc_defense/meco_track_training.py`, PPO sampling, transition assembly, RNG, optimizer transaction and checkpoint integration. P4 exposes a typed model/scoring API that P5 can consume after exact-head acceptance. | Trainer integration head is future work; no P4 trainer edits are allowed. |

P2 owns candidate generation, route queries/results, observation and target handles. P4 owns only Plan C C1/C2 model/scoring paths and their fixtures. P5 alone owns shared trainer/caller integration. P6 owns qualification runner and dashboard/qualification data paths. P1 owns V4 schema/artifacts; P3 owns Plan A. If the accepted API requires changing another package's path, stop and request a reviewed amendment rather than expanding this allowlist.

## Typed P4 interface

P4 exports a standalone C1/C2 scoring adapter consumed by P5 only after P4's exact API head is accepted and integrated. It accepts P2-owned device tensors for one profile-pinned world: raster `[1,6,25,51]`, globals `[1,7]`, raw fixed candidate features `[1,537,20]`, missingness `[1,537,20]`, source descriptors/identity, validity and rank masks, ordinary/continuation row maps, route sink rows `[1,9681,...]`, route/risk costs and recurrent state. It validates dimensions, dtype, device, profile/source identity, map range/uniqueness, masks, row sentinels, finite values and route capacity before scoring.

Both C1 and C2 return the same typed result: gross utility for the 16 ordinary slots and continuation slot, uncertainty where produced, updated recurrent state, selector-local-to-source-row mapping, valid selection mask, and separately retained route and risk costs. The shared deterministic arbiter returns selection-facing scores after applying route/risk once and enforces accepted fallback/hysteresis and feasibility semantics. The P2 native executor remains final authority. P4 does not sample actions, consume PPO random words, build transition rows, write artifacts/checkpoints, execute commands or edit the trainer.

C1 encodes the 40-column candidate view with its candidate MLP plus global/recurrent context. C2 uses the same candidate descriptors and recurrent/global context while adding a raster map encoder and candidate-to-map attention. C1 and C2 must receive identical candidate/map/route inputs and expose the same output schema; architecture-specific differences are limited to learned representation and gross utility output.

D1 is the deterministic comparator over the same feasible source rows, maps, route estimates, masks and command identities used for both learned tracks. Its source-bound score rule remains stable; only its representation/execution may be vectorized to meet device residency. No learned score, candidate recall, or route/risk cost may change D1's ordering contract. D1 and both learned paths use the same deterministic arbiter, including one-time route/risk application, safety mask, fallback and hysteresis.

## Finite Pass 2 allowlist

Pass 2 may edit only these files in a fresh, ZFS-backed worktree from the exact accepted integration head containing G0's reviewed contract bridge and accepted P1/P2/P3 interfaces:

- `frc_defense/meco_plan_c_v4.py` — new GPU-resident C1/C2 model wrappers and typed fixed-axis feature/view adapter, common Plan C score arbiter and D1 adapter. Do not generate candidates/routes or duplicate P2 tensors.
- `frc_defense/meco_tracks.py` — C1/C2 class/factory/score-adapter hunks only, if needed to preserve existing public model construction. No A1/A2/A3 classes, shared generic trainer helpers, shared sampler, or unrelated model code may change. If C-specific factory integration cannot be isolated, stop for plan amendment.
- `tests/test_meco_plan_c_v4_contract.py` — new fixed-axis, ordering, identity, feature/missingness, maps, masks, ranks, route capacity, C1/C2 interface, D1 parity and failure fixtures.
- `tests/test_meco_plan_c_v4_simulator.py` — new seeded end-to-end C1/C2 simulator harness/integration assertions and selected-versus-executed replay inspection.

No other path is authorized. In particular do not edit `frc_defense/meco_candidates.py`, `meco_pose_graph.py`, `tensor_adstar.py`, observation/environment files, P1 schema/event/artifact files, `frc_defense/meco_track_training.py` (P5-exclusive), counterfactual/PPO/RNG/checkpoint code, P3 Plan A files, P6 qualification/dashboard code, build files, active campaign data, outputs or checkpoints. No HIP/C++ source is pre-authorized; request a reviewed amendment if required.

## Worktree and three-pass cadence

After Pass 1 ACCEPT and gates, the implementer creates a fresh ZFS worktree at `.worktrees/autodrive-v4-p4-plan-c-impl` on branch `codex/v4-p4-plan-c-impl-YYYYMMDD`, based on the exact accepted integration commit. Do not reuse this planning worktree, the active campaign checkout or its outputs. The implementer records exact dependency commits/blobs and review records before editing. Push coherent, reviewable slices before dependent work: typed row/view contract; C1 and C2 model adapters; common arbiter/D1 parity; GPU-residency fixtures; seeded simulator evidence. Every slice changes only the finite allowlist. Rebase or dependency changes require a new exact-head review.

Pass 1 planner is `plan_c_owner`. Pass 2 implementer is unassigned. Pass 3 reviewer is unassigned and must be distinct from planner and implementer. The reviewer checks exact commit/blob hashes, allowlist, accepted P1/P2/P3 interface pins, Plan C row mapping and profile bytes, C1/C2 input parity, D1 and arbiter consistency, GPU evidence, tests, seeded simulator/replay, current dashboard provenance and no P5/P2/P3/P6 ownership leakage. An ACCEPT applies only to that exact head and does not itself authorize training or qualification.

## Required validation and acceptance evidence

All validation is future-gated; no tests or runtime work were performed for this plan.

1. **Contract fixtures:** assert exactly 537 rows and all row mappings, including variant-major rows 512–535 and continuation 536; exact 20 raw feature names/order and matching 20 missingness bits; identity codes 1–537 and invalid zero; descriptor field widths; generation-safe target handles; no oracle inclusion in policy rows.
2. **Selection fixtures:** verify K=16 ordinary rows, separate continuation slot, stable sort/tie-break, ordinary/continuation maps, `-1` unused map slots, no invalid continuation view, `candidate_valid`, model-input and policy-eligible masks, rank sentinel, finite-zero masked features, uniqueness/range guards and no truncation.
3. **Route fixtures:** preserve all 9,681 ordered sinks and 152×64 chunk addressing, including inactive and oracle sentinel domains; 73 path-point capacity; support digest, route/risk units, statuses, feasibility, route variants and target handles. Inject missing, duplicate, reordered, overflowed and nonfinite route rows and verify fail-closed behavior.
4. **Architecture and D1 parity:** for identical seeded source descriptors, features, masks, raster, route/risk inputs and recurrent state, verify C1, C2 and D1 consume the same candidate universe/maps/costs and command identities. Verify C1 uses candidate/global/recurrent representation; C2 additionally uses raster/map attention; outputs are finite on valid rows and never widen the feasible mask. Assert route/risk are subtracted once and fallback/hysteresis/safety behavior is common. D1 remains deterministic and consumes no PPO RNG.
5. **Residency/failure fixtures:** run model/view construction, masks, rank selection, D1, arbiter, validation and digest-facing arrays on assigned HIP/WX9100 devices. Record per-kernel device placement, transfers and synchronizations; reject CPU tensors, CPU/P600 fallback, device mismatch, NaN/Inf valid values, out-of-range maps, stale identities and unsupported kernels. CPU may generate immutable expected fixture bytes only; it may not compute policy features, scores, masks, routes or decisions.
6. **Seeded end-to-end simulation:** after approved P4 tests, run representative seeded C1 and C2 scenarios from the current integration checkout. Inspect fresh simulator records/replays for selected source row versus executed action, route feasibility and path, costs, fallback/remasking/preemption, score events and episode outcome. Use reproducible scenario/seed and record source/config/checkpoint identity, replay ID, and observed results. Unit tests alone do not establish behavior.
7. **Dashboard provenance:** before generating or reviewing a replay, restart the relevant dashboard services from the exact current integration checkout. Verify each service active, backend endpoint success, backend code/head and frontend asset identity. Generate a fresh replay after restart, inspect provenance, and mark prior-code replays outdated. If simulator, device, dashboard or replay evidence is unavailable, report validation incomplete.
8. **Two-device evidence:** demonstrate approved P4 algorithmic work runs on both assigned WX9100 devices with UUID/runtime/PyTorch/HIP details and per-kernel placement; no CPU/P600 fallback, silent migration or host-side dynamic tensor reads. Do not claim one simulation jointly uses both devices unless the accepted implementation proves it.

G0 and the accepted overall plan prohibit all source/test/build/runtime/device/dashboard/simulator/training work before campaign stop and lease-release evidence and the reviewed contract bridge. These validation requirements are future acceptance gates, not permission to bypass G0.

## Risks and stop conditions

- P1/P2/P3 implementation interface heads do not exist at this plan head. Do not implement against moving branches or draft plan shapes; wait for exact independently accepted API heads and pin them.
- The base candidate builder is variable-width, Python/list based and lives under P2 ownership. Do not clone or modify it to accelerate P4; P2 must supply the fixed descriptor and route tensors.
- `meco_tracks.py` contains shared five-track definitions. Keep changes limited to C1/C2 blocks and explicit adapter/factory seams. Any conflict with P3 or broad refactor requires a reviewed amendment.
- C2 map attention and 9,681 route rows may raise memory use. Measure on both WX9100s; do not reduce profile capacity, omit oracle/inactive sinks, or spill numerical work to CPU.
- Route/risk costs must be applied exactly once, and model scores cannot replace P2/native legal and safety checks. A score parity fixture must detect duplicate subtraction or mask widening.
- Existing D1 loops over host objects. Preserve its exact deterministic rule while moving computation to the accepted device interface; if parity cannot be demonstrated within the allowlist, stop for amendment.
- P5 alone owns the shared trainer. P4 publishes only a standalone model/scoring API; P5 integration is a downstream package and exact-head review.
- This plan does not satisfy G0/P0a, authorize implementation or devices, establish GPU residency, validate simulator behavior, publish V4, or claim human approval or qualification.

## Three-pass disposition

- Pass 1 planner: `plan_c_owner`; cannot implement or review P4.
- Pass 2 implementer: unassigned; must use the exact accepted integration head and finite allowlist.
- Pass 3 reviewer: unassigned; must be independent of planner and implementer and review exact pushed evidence.

Planner disposition: **PENDING independent exact-head Pass 1 review.** No source edits, tests, runtime, device use, simulator/dashboard validation, training, qualification or human approval is claimed.
