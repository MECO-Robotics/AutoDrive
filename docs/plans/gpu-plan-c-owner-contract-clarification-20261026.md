# Pass 1 proposal: clarify Plan C policy identity and two-view ownership

**Status:** Planning-only owner-contract proposal. This is not accepted, does
not authorize implementation, tests/imports, runtime/device/simulator/dashboard
work, checkpoint access, training/evaluation, or campaign changes. It is
submitted for independent Pass 1 review and explicit owner disposition.

**Planning base:** `main` commit
`e37b2f30b6bc51b74dd72bf039e94a174ec51bf2` (tree
`8edb9998d7e3e361e83938a345c65d4fef488e1d`). The local user-modified HIP file
is outside this proposal and untouched.

**Reconciliation source:**
`docs/plans/gpu-five-track-ppo-plan-c-source-view-reconciliation-20261026.md`,
proposed artifact SHA-256
`ad3b9f69fb5ade7ebe0844be2a3b3105ffcebcd7f708969631efc9f8133e1924`.
That proposal distinguishes the policy table (`G_policy=33+E`) from the
expanded route pool (`Q_total=105+19E`) and explicitly requires owner approval
before narrowing the accepted K-view wording.

**Accepted K-view authority:**
`docs/plans/gpu-plan-c-k-capped-model-view-contract-20261026-amendment.md`,
artifact SHA-256
`6d9c0e6320acd503c8595fa619e8beeba24af88f36e49337d7325331a899ff0b`,
accepted artifact commit `4cdc30f923c0d948b43c57d452c4ee4b17cf61d9`.
This proposal preserves its K ordinary choices plus optional continuation
selection, two K-width input views, source-global identity mapping, and its
existing continuation descriptor/action/goal/route revalidation semantics.

## Owner decision requested

The current reconciliation proposal identifies an ambiguity but cannot settle
it unilaterally: the accepted K-view artifact calls the source-global row an
index into the full per-decision `specs` table, while the PPO inventory needs a
bounded policy-row identity and the current producer appends oracle-only route
specs to that same Python list. This proposal asks Plan C, route, PPO, and
shared-ABI owners to approve or reject the following explicit coordinate
resolution before code work.

### Proposed identity coordinates

1. **Policy source row (`policy_source_row`):** index only the ordered policy
   source table, including the reserved continuation row. Its domain is
   `0..G_policy-1`, where `G_policy=33+E` and current source bounds give
   `G_policy≤633`. Base, sensed-target, and approach-offset policy rows retain
   the current source order; the continuation remains the last policy row at
   `32+E`. Ordinary K-local and continuation-local view slots map to this axis
   through device row maps. This coordinate is the one stored for a selected
   policy decision in PPO.
2. **Route query ordinal (`route_query_ordinal`):** index the complete ordered
   route work sequence, including both policy route queries and oracle-only
   recall queries. Its domain is `0..Q_total-1`, where current source bounds
   give `Q_total=G_policy+18(4+E)=105+19E≤11,505`. Every route result and chunk
   receipt is associated with this ordinal. For the current ordering, policy
   route query q corresponds to policy row r for q=r in `0..G_policy-1`, but
   the two fields remain distinct and are never interchangeable.
3. **Oracle-only identity:** each oracle query has its route ordinal and a
   typed identity for its source target slot and oracle variant. It has no
   `policy_source_row`, is nonselectable, and never enters either model view or
   the action-selection vector. The exact sentinel and identity encoding are
   left to the route/PPO/shared-ABI owners; this proposal does not invent them.
4. **Mapping:** each route query maps to exactly one policy source row or one
   oracle-only identity. Policy query results write back to their policy row;
   oracle results contribute only to the owner-approved recall/summary
   evidence. Selection maps local K/K+1 indices to `policy_source_row`, then
   obtains selected route evidence through that row's query association.

This interpretation narrows “source-global row indexes the full `specs` list”
in the accepted K-view artifact. It treats `specs` as an implementation
container that may enumerate both policy and oracle work, not as a single
identity namespace. **That narrowing requires an explicit Plan C and
shared-ABI owner decision.** Until accepted and reflected in the canonical ABI,
do not encode `policy_source_row` or infer that its meaning has changed.

### Proposed static identity scope

- Scope the static ordered policy ID/schema table to exactly `G_policy` rows,
  including the continuation *row slot* and excluding every oracle-only
  query. Its order is the policy source order above, not the compact model-view
  order. The compact views carry maps into this policy table.
- Bind that table's static digest only to fields the owners classify as
  static: the stable policy row/schema identity, registered feature schema and
  order, and snapshot/schema version as specified by the ABI. Do not include
  dynamic tensors, row maps, route results, sensed-target values, active
  commitment contents, or an unapproved encoding in a host-computed digest.
- Keep a separately versioned oracle-query identity/schema and route-pool
  receipt keyed by `route_query_ordinal`. Oracle IDs do not extend
  `candidate_ids` and cannot be interpreted as policy candidates.
- **Owner decision required:** the current source's continuation string is
  `continue:inactive` or `continue:<active candidate ID>` and thus its value
  varies per decision. The proposal preserves this semantic identity, but
  does not claim that a per-decision continuation string belongs in a static
  ID tuple or static digest. Plan C/PPO/shared-ABI owners must choose and
  specify how its dynamic binding is represented by the already accepted
  descriptor/transition fields, without changing the established continuation
  action, target, goal, route, or commitment revalidation. If the canonical ABI
  has no lossless field for this identity, stop for an ABI amendment before
  source work; do not silently drop the identity or invent a parallel record.

### Model view and context clauses retained from the accepted K-view contract

- Both C1 and C2 receive separate ordinary and continuation feature tensors,
  each `[1,K,40]`, where `K∈{8,16,32}`. The ordinary view has K
  eligibility-filtered, rank-ordered policy rows. The continuation query is
  slot 0 of its own K-wide view; remaining entries are masked padding. The
  model returns K ordinary utilities and one continuation utility. Only the
  subsequent device selector combines them into K+1 options, ordered ordinary
  slots `0..K-1` then continuation slot K. Continuation never displaces an
  ordinary row.
- **C1:** score the ordinary view and continuation query independently with
  the same registered C1 encoder/scorer parameters and the same global/hidden
  state. C1 does not pool candidate context. Its continuation utility is not
  obtained by appending continuation to the ordinary feature tensor.
- **C2:** form the pooled candidate context from active rows of the ordinary
  K-view only, once. Apply the existing recurrent context update once using
  that pooled ordinary context and the global/hidden state. Score ordinary
  rows and the separate continuation query against the same map tokens and
  that same computed recurrent state. Continuation must not enter the pooled
  context, trigger a second context/GRU update, or replace the shared state.
  If no ordinary row is active, the pooled candidate contribution is finite
  zero, with the existing global/hidden context update retained. The
  continuation query can still be scored if its independent validity mask is
  true; selection/fallback remains governed by the established feasible K+1
  support.
- These are accepted design requirements, not claims about current source
  behavior. Current code still builds an axis-wide feature tensor and its C2
  mask can include continuation in the pool; implementation must change the
  producer, adapter, and scorer paths under a separately reviewed source plan.

## Evidence and current-source mismatch

The following source observations are pinned to `e37b2f30…` and should be
checked by Pass 1 review against that exact tree:

| Observation | Source evidence | Consequence |
|---|---|---|
| Policy source order | `frc_defense/meco_candidates.py:1291-1313` creates 8 base rows, E sensed-target rows, then 24 offsets; `:1418-1420` appends continuation. | Current policy extent is 33+E with continuation last. |
| Oracle rows share the list | `meco_candidates.py:1421-1449` appends 18 variants for the base/sensed target slots after the continuation row. | A raw `specs` index can address a policy row or oracle-only row; it is not inherently a PPO policy-row ID. |
| Current identity creation | `meco_candidates.py:575-600` creates one ID per item in `specs`; the continuation label embeds active commitment identity; duplicate labels receive a row suffix. | Current full ID tuple is not static-only and spans beyond the policy table. The static/dynamic split needs an owner contract. |
| Current policy projection | `meco_candidates.py:788-835` constructs `policy_indices`, appends continuation, derives `policy_ids`, and indexes route tensors on that selection. | A policy projection already exists, but the current selected index/metadata behavior does not establish the ABI field semantics proposed above. |
| Current K configuration | `frc_defense/meco_track_training.py:139,154-155`; `meco_candidates.py:1268-1269`. | Supported K values are 8, 16, and 32. |
| Current C2 pooling | `frc_defense/meco_tracks.py:1118-1135`. The mask-weighted mean uses all entries admitted by `candidate_mask`, and that state is used for scoring. | Existing implementation is not proof of ordinary-only context with a separate continuation query. |
| Current adapter shape | `meco_tracks.py:1343-1362` checks feature/mask width against `len(table.candidate_ids)` and zeros rows under `model_input_mask`. | It remains a full-axis masked request rather than the accepted pair of K-wide views. |

The accepted K-view artifact itself is the authority for the model shape and
context requirements; this proposal does not re-open or weaken them. The
current PPO V3/schema inventory and shared ABI remain authorities for field
names/types and record capacities; this proposal does not change either.

## Required owner answers before implementation

1. **Plan C owner:** accept/reject the policy-only meaning/domain of
   `policy_source_row` and confirm the exact source order, including
   continuation row placement.
2. **Route owner:** accept/reject the separate `route_query_ordinal`, policy
   row association, and oracle-only identity mapping; define any required
   bounded oracle evidence retention in its own amendment.
3. **PPO/shared-ABI owners:** approve/reject the coordinate split and static
   candidate ID scope, then publish the exact field names, dtypes, capacities,
   sentinel/failure rules, and dynamic continuation identity binding in the
   canonical ABI. This proposal does not establish those encodings.
4. **C1/C2 model owner:** confirm that the retained separate-query scoring and
   C2 ordinary-only shared context preserve the approved checkpoint parameter
   contract, with no continuation contribution to pooled context.
5. **Cross-owner reviewer:** verify the exact artifacts/source pins and that
   the coordinated source plan and allowlist account for all producer,
   adapter, scorer, PPO, route, and record consumers before any implementation
   Pass 2 begins.

Until these answers are accepted and the canonical ABI/source plans are
updated, the exact policy-row versus full-`specs` meaning and continuation ID
encoding remain owner decisions. Do not implement this proposal from inference.

## Scope and gates

This artifact changes no source, ABI, weights/checkpoints, training schedule,
campaign state, hardware assignment, dashboard, replay, or qualification
protocol. It establishes no implementation, test, device parity, simulator
outcome, Stage 1 pass, or campaign readiness. No tests/imports/runtime or GPU,
simulator, dashboard, or campaign action is included. If accepted, this is
planning input only; source changes require a new exact allowlist, independent
Pass 1, implementation, and exact-head Pass 3 review plus the previously
deferred device and simulator gates.
