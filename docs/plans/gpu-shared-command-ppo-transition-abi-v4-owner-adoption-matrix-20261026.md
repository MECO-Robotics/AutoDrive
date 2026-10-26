# Pass 1 plan: V4 owner adoption matrix and implementation gate

**Status:** Proposed owner-decision plan only. Every owner disposition in this
artifact is **PENDING**. The accepted Pass 1 review of the V4 ABI proposal is
not owner adoption and does not authorize a schema declaration, source change,
test, import, compilation, runtime/device/simulator/dashboard operation,
checkpoint access, campaign change, training, or evaluation.

**Planning base:** `codex/meco-v4-inventory-update-20261026` commit
`f36101ff0efa05a5e2a99ac96ef5a873a9f3a957` (the plan must be refreshed if its
base changes before adoption work begins).

**Proposed contract under owner review:**
`docs/plans/gpu-shared-command-ppo-transition-abi-v4-20261026.md`, candidate
commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, artifact SHA-256
`0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
This is an independent, versioned successor proposal to accepted V3 artifact
commit `1c7e9ef02030fa519325c44aa12d4b07e5d171cc` (SHA-256
`65a4d936deb792b26b61b7569fec992dc8d35ec9da11c02845fcf342fa4b7f30`). V3
remains immutable and valid for its existing records. V4 Pass 1 ACCEPT covers
planning-artifact quality only; this plan does not promote any proposed V4
value into a normative contract.

## Purpose and decision rule

Collect exact, independently attributable decisions from the owners named in
V4 §“Owner adoption gates before any schema plan”. Each owner must either:

- **Adopt:** explicitly accept the exact V4 artifact SHA and identify the
  fields, profile values, semantics, source evidence, and limits within that
  owner's scope; or
- **Reject with replacement:** identify the exact incompatible field or
  assumption and publish a versioned replacement proposal with evidence.

Silence, prior source facts, current defaults, an agent's recommendation, the
V4 Pass 1 review, or agreement that V4 is useful does not count as adoption.
A changed source base or candidate artifact hash resets affected decisions to
**PENDING** until the changed contract is reviewed. No owner may adopt on behalf
of another owner. All dispositions remain **PENDING** in this plan until the
actual owner response and its exact artifact/review evidence are recorded in a
new, separately reviewed status update.

## Owner decision matrix

The evidence listed below is the minimum needed to decide each row. The source
proposal and cited source facts are inputs to review, not evidence of owner
acceptance. V4 field names/shapes below refer to the candidate artifact at the
exact commit and SHA above.

| Owner | Exact decision requested from V4 | Minimum evidence required to resolve | Disposition |
|---|---|---|---|
| Environment/profile | Accept/reject profile `meco-ppo-v3-3v3-f504-n1-s6-r32-t12-v1`: `N=1`, `S=6`, `R=32`, `T=12`, `E=504`, `K=16`; field, raster, grid and one-track-per-run bindings. Accepting `E=504` freezes this profile value, not the source ceiling of 600. | Named environment/profile owner response against V4 SHA; exact source/config provenance for each value; an explicit statement whether this is the production training profile or a versioned replacement. | **PENDING** |
| Plan A | Accept/reject padded node capacity `A_N=12,096`, runtime `node_valid` subset and its digest binding, `A_G=96,768`, ten edge slots per padded node, 14 ancestor levels, full dynamic search/support evidence, and the proposed float64 promotion for three cost columns. | Plan A owner response against V4 SHA; source-backed fixed map/robot profile and `node_valid` semantics; dtype decision for all recorded costs; confirmation of full support/search field ordering and digest boundary or a replacement table. | **PENDING** |
| Plan C | Accept/reject full source axis `G_C=537`, raw/model feature widths 20/40 and order, K=16 ordinary view, separate continuation map, candidate masks/ranks, and source-row identity binding. | Plan C owner response against V4 SHA and the accepted K/K+1 artifact; source-backed row order and feature registry; explicit ordinary/continuation map, padding/sentinel, eligibility and selected-source-row semantics or a replacement. | **PENDING** |
| Route | Accept/reject C-only `Q_C=9,681`, 152 ordered direct-scatter chunks of at most 64, retention of all policy/oracle query, result and path rows, five float64 route-cost columns, and the separation from Plan A support/search. | Route owner response against V4 SHA and accepted route contract; source-backed query count/order/chunk map; explicit inactive-row semantics, policy/oracle identity, path retention, cost units/provenance and selected-row association or a replacement. | **PENDING** |
| Observation/native | Accept/reject the `[504]` target table and V4 fields (`target_xy`, velocity, age, confidence, validity, four-int target handle), snapshot binding, slot-generation allocation/reset rules, stale-handle behavior, generation exhaustion behavior, and fixed 12-tick outputs. The current source's default and upper bound are evidence for profile derivation, not owner adoption or proof of perceived-slot identity lifecycle. | Observation and native simulator owner responses against V4 SHA; exact lifecycle transition points for anonymous sensed slots; target handle/snapshot axes, invalid codes, no-wrap/overflow behavior, reset binding and source citations; explicit reconciliation of any sensor slot generation with physical piece generation. | **PENDING** |
| PPO/RNG | Accept/reject row-validity and actor/critic masks, terminal/truncation/bootstrap and update-atomicity semantics, selected-action versus executed-action/log-probability linkage, Philox4x32-10 action/permutation stream derivation and counters, digest coverage/encoding and device-produced payload rules. | PPO and RNG owner responses against V4 SHA; exact row/tensor field table, status and reset rules, stream key/counter/draw schedule and checkpoint continuation state; canonical byte/digest encoding and failure behavior or a replacement contract. | **PENDING** |
| Gamepiece/score | Accept/reject the 59-byte typed score-event record, source-site/code registry, `R×S×T=2,304` profile event capacity, device-native capture, staging before mutation, and fail-closed overflow semantics. Keep the 32,768 episode recorder bound distinct from the profile batch capacity. | Gamepiece and score owner responses against V4 SHA; event field/code registry and ordering; proof each source event fits the row/tick bound or an adjusted finite capacity; reset, overflow and no-partial-mutation behavior or a replacement. | **PENDING** |
| Artifact/manifest | Accept/reject the 96-byte little-endian envelope V2 header, exact maximum payload calculation, static source/profile digest binding, opaque host persistence and reject-before-parse behavior. V4 proposes 9,822,906 bytes per transition row and a maximum payload of 314,469,051 bytes for a 32-row batch. | Artifact/manifest owner response against V4 SHA; independently reproducible byte arithmetic from the exact field table; byte order, version, length, digest, alignment and overflow rules; reader compatibility/rejection rules or a replacement. | **PENDING** |
| Evaluation/curriculum | Accept/reject reward and terminal estimand alignment, curriculum compatibility, and any optional recall metric definition. V4 does not assert a recall result or Stage 1 outcome. | Evaluation/curriculum owner response against V4 SHA; exact estimand, denominator/reference/empty-reference semantics for any recall metric, protocol identity, and explicit mapping to the approved curriculum gates or a replacement. | **PENDING** |

### Cross-owner ABI publication gate

The shared-ABI owner is responsible for reconciling the nine decisions above.
This gate is **PENDING** until each applicable owner disposition is backed by
its exact acceptance/replacement artifact and independent review. The ABI owner
must then publish a new immutable adopted V4 ABI/profile artifact that names
all accepted field types, capacities, codes, sentinels, lifecycle rules,
digests, byte layout and compatibility behavior. The adopted artifact must
have one exact hash; no owner may pin a different or symbolic version. A
replacement to any field returns that field and every dependent owner row to
review before publication.

## Proposed profile and payload decisions remain provisional

The V4 proposal recommends source-derived defaults to make one complete
contract reviewable. In particular, `E=504` is a current default, while source
permits up to 600; `T=12`, `R=32`, and `K=16` are current defaults/configuration
choices, not universal fixed schema maxima. Plan A capacity is padded capacity,
not a claim that every node is legal. Plan A float64 cost storage is a proposed
promotion, not current source dtype. The Plan C source axis and route pool are
separate from Plan A's graph/support axes. Target handles represent anonymous
observation tracks, not physical gamepiece IDs. The V4 row and batch byte
counts are proposal arithmetic that the artifact/manifest owner must reproduce
before adoption.

Do not resolve an unknown capacity by copying a current run's observed maximum,
Python list length, default config value, or per-call route limit into the ABI.
If an exact bound cannot be established for an owner field, keep the disposition
**PENDING** and request a replacement profile or finite-capacity proposal.

## Three-pass workflow and execution boundary

1. **Pass 1 — owner decision planning.** Review this matrix against the exact
   V4 and source hashes. Verify that every proposal is represented, facts are
   distinguished from decisions, and all dispositions are pending. A Pass 1
   ACCEPT approves this collection workflow only; it cannot adopt V4 for an
   owner.
2. **Owner adoption — before implementation planning.** Obtain each decision
   and evidence listed in the matrix. Record actual owner identity, exact
   decision artifact/hash, independent review result, and unresolved/replaced
   fields. Keep any missing, ambiguous, or conditional answer pending.
3. **ABI reconciliation.** After all owner answers are accepted, publish and
   review the single immutable ABI/profile hash. Update owner plans to pin that
   same hash. If owners reject or replace fields, reconcile the V4 successor
   document/version first; do not edit V3 in place.
4. **Pass 1 — implementation plan.** Only after ABI adoption, write a separate
   narrowly allowlisted schema/producer/consumer plan with exact source base,
   migration and compatibility rules, tests, and runtime gates. This matrix
   authorizes no implementation source path.
5. **Pass 2 — implementation.** Implement only against that accepted plan and
   exact adopted ABI. Keep source edits separate from campaign artifacts,
   checkpoints, runtime and dashboards; create incremental reviewable commits.
6. **Pass 3 — exact-head review.** An independent reviewer examines the pushed
   exact candidate and plan allowlist. Revisions require another exact-head
   review. Device, simulator, dashboard/replay and training gates remain
   separate and require their own authorization/protocol; never infer those
   outcomes from static review or schema acceptance.

The live training campaign, its GPU leases, checkpoints and dashboards remain
untouched. No tests/imports, compilation, runtime, simulator, dashboard,
checkpoint, campaign, training or evaluation action is authorized by this plan.
A later Stage 1 claim requires the approved evaluation records for all five
tracks and evidence that training continues without issues; owner adoption and
ABI publication alone cannot satisfy that objective.

## Acceptance checklist

- The matrix contains all nine V4 owner gates and the cross-owner immutable ABI
  publication gate.
- Every owner row has a specific acceptance/rejection subject, source/evidence
  requirements, and **PENDING** disposition; no owner acceptance is implied.
- V3 immutability and the V4 artifact commit/hash are recorded accurately.
- Proposed defaults, padded capacities, source ceilings, target handles,
  route/support axes, and proposed dtype promotions are distinguished from
  current source facts and accepted decisions.
- The plan preserves the source/runtime/campaign boundary and requires a later
  separately reviewed implementation plan plus exact-head Pass 3.
- Only the allowlisted planning file is changed; static `git diff --check` is
  clean. No runtime, device, simulator, dashboard, checkpoint, or training
  action is performed.
