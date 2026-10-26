# Pass 1 proposal: Plan C route-owner streaming contract

**Role:** Route-owner planning proposal; planning-only.
**Status:** Proposed for Plan C, route, observation/native, PPO, and shared-ABI owner decisions. No source implementation or schema/layout approval is implied.
**Planning base:** `origin/main` `e37b2f30b6bc51b74dd72bf039e94a174ec51bf2`.
**Proposed Plan C/PPO capacity reconciliation input (planning-only):** artifact SHA-256 `ad3b9f69fb5ade7ebe0844be2a3b3105ffcebcd7f708969631efc9f8133e1924`.
**Accepted route-plan authority:** `docs/plans/gpu-resident-route-decision-slice-20261025.md`, accepted amendment commit `8567fd92d4dc9e214e8502f2249bb02203ac8c03`, artifact SHA-256 `ac5847d65de2af5ccb17d6a1bc109dceefeab71c5d707253980570020c1e885a`.
**Related accepted K/K+1 authority:** `gpu-plan-c-k-capped-model-view-contract-20261026-amendment.md`, accepted artifact SHA-256 `6d9c0e6320acd503c8595fa619e8beeba24af88f36e49337d7325331a899ff0b`.
**Older accepted shared-ABI schema facts:** commit `1c7e9ef02030fa519325c44aa12d4b07e5d171cc`, artifact SHA-256 `65a4d936deb792b26b61b7569fec992dc8d35ec9da11c02845fcf342fa4b7f30`; these source-bounded field facts require V3 route query/result tensors, `route_query_active`, `path_point_mask`, `path_xy`, and route status fields, but do not freeze the complete production capacity/profile or authorize implementation.
**Current shared-ABI coordination proposal (planning-only):** `docs/plans/gpu-shared-command-ppo-transition-abi-20261026.md`, proposal commit `6d46846b87a7790396ef5e45868869c66ac173f6`, artifact SHA-256 `19349ebd40bbe6c0f11e8001bb716524d0273f8f91862f040096883faf3eb338`. It is not an accepted replacement for the older schema facts and does not independently authorize implementation; any later shared-ABI amendment must be pinned and accepted before it changes these requirements.

## Decision sought

Adopt a route-owner contract that keeps the accepted 64-query call bound, makes the full ordered route pool streamable, and keeps route results unambiguously associated with policy source rows. This proposal makes the identities and memory-retention boundary explicit. It does not choose a digest algorithm, V3 record layout, target-generation encoding, or a recall definition that the cited owners have not approved.

## Source-grounded extents and existing behavior

The proposed reconciliation input distinguishes:

- `E` is the additional target-slot **capacity**, including inactive/padded slots, not the active sensed-target count. Source evidence bounds this capacity by `E ≤ 600`, but does not establish `600` as the frozen production profile. The fixed-capacity producer emits all E target slots and carries a separate device eligibility/active mask; inactive slots remain represented and are not compacted away.
- `G_policy = 33 + E` source-global policy rows: 8 base, one row for each of the E target slots (active or inactive), 24 approach-offset, and one continuation row; hence `G_policy ≤ 633` under the source bound.
- `Q_total = G_policy + 18(4 + E) = 105 + 19E` route queries: policy queries first, then 18 oracle variants for each of 4 base slots plus every one of the E target slots, whether eligible or inactive. At `E=600`, this is 11,505 queries. Active/eligibility masks encode which rows can produce a usable route; they do not shrink or reorder the query table.
- Model views are two independent `[1,K,40]` tensors (`K ∈ {8,16,32}`); selector slots are K ordinary plus optional continuation. Neither is a route-query extent.

The accepted route ABI fixes each invocation to `Q ≤ 64`, `P=73`, and the status, feasibility, path, length, distance, endpoint-error, weighted-cost and provenance semantics. The existing `make_device_route_chunk_map` constructs contiguous, ordered spans of at most 64 and device global/chunk/local row coordinates (`experimental_route_evaluation.py:21-64`). But `evaluate_device_route_chunks` first tuples the entire query sequence, creates every span, holds every result chunk, and concatenates full outputs (`:631-703`). It therefore enforces a per-call cap without establishing bounded streaming of the 11,505-query pool.

At the current candidate producer, ordinary policy rows are built before continuation, and oracle specs follow them. The 18 oracle offsets are enumerated outside the sensed-target loop, yielding variant-major then target-major order (`meco_candidates.py:1291-1382,1421-1453`). The device branch sorts `oracle_track_indices` then transfers them to host with `.cpu().tolist()` to build specs (`:1427-1449`); current route identity is `candidate_id`, snapshot, robot `world`, and robot `slot` (`experimental_route_evaluation.py:112-148`). These source facts do not define a stable target-slot generation identity or prove that sensor track indices remain stable across slot reuse.

Current device candidate construction builds a full route batch, selects policy routes by source-global `policy_indices`, and calculates oracle aggregate counts and cost statistics (`meco_candidates.py:819-830,885-917`). It exposes a `candidate_recall` scalar derived from feasible-policy count divided by feasible-oracle count, while labeling the oracle bounded and not a simulator-optimality oracle. That ratio alone does not establish set-recall or coverage of one pool by the other.

## Proposed route identity and ordering contract

Keep these coordinates distinct in every route/PPO/command interface:

1. **`policy_source_row`**: source-global row in `[0,G_policy)`. It is the identity axis for policy candidate fields and is never a K-local slot or route ordinal.
2. **`route_query_ordinal`**: ordinal in `[0,Q_total)`, assigned by immutable query order. It is never interpreted directly as a policy row, even where the current order makes the values coincide.
3. **`policy_local_index`**: `[0,K]` selector slot. Ordinary slots resolve via `ordinary_row_map`; continuation slot K resolves via the continuation map to the source-global continuation row.
4. **`oracle_identity`**: typed non-policy identity for an oracle target and variant. It is not a candidate ID, policy source row, model row, or selectable action.
5. **`chunk_local_row`**: `[0,chunk_count)` within a fixed host-known chunk span, used only to address that invocation's <=64 result batch.

Assign route queries in two stable regions:

- Policy region: one query per policy source row, in source-global order `0..G_policy-1`. A device association map `policy_source_row_to_route_query_ordinal` names the exact route result for each policy row. It may be identity-valued in the current producer, but must remain an explicit field/map.
- Oracle region: then, for each oracle variant in the current canonical variant order, one query per canonical target slot. Identify targets by a stable observation/native-owned target slot plus generation, and encode the oracle variant as a typed integer. Preserve the current variant-major ordering unless the Plan C route owner explicitly versions it. Do not use robot `slot` or a transient track-buffer index as the oracle target identity.

Each route query has one tagged association: either `{policy_source_row}` or `{oracle_target_slot, target_generation, oracle_variant}`. The `route_query_ordinal` and tag are unique within the decision snapshot. A policy row may not alias an oracle identity. The query also carries the existing world/robot-slot/snapshot binding and its goal/source provenance. A masked or invalid target has a typed inactive query result, not a reused identity.

This leads to the existing bound:

`Q_total = (33+E) + 18(4+E) = 105+19E ≤ 11,505`

and at most `ceil(Q_total/64) ≤ 180` contiguous chunks. The producer computes spans from frozen profile metadata; no span or query is compacted/reordered based on device route outcomes.

## Proposed ordered-stream and retention contract

Process spans in ascending global ordinal. For each span `[start,stop)`:

- construct/consume at most 64 query rows and their dynamic goal tensors;
- invoke the accepted route ABI once, with query-local result rows `[0,stop-start)`;
- bind those outputs to global ordinals `start..stop-1` through device row coordinates;
- write every query/result field, including path and masks, into its preassigned V3 device-table rows; policy rows additionally resolve through `policy_source_row`, while optional oracle summaries/digests are folded alongside the retained rows;
- propagate any malformed/non-finite chunk as a fail-closed decision for the complete snapshot, consistent with the accepted route contract.

The memory target is bounded chunk workspace of `O(64×P)` plus the fixed-capacity V3 route query/result representation and any approved summary/digest state. V3 route evidence is not summary-only: preserve the route query/result tensors and their row bindings, including `route_query_active`, `path_point_mask`, route path (`path_xy`), path length, route status, and the other schema fields required by the accepted shared-ABI field table. Stream each <=64-query invocation into its preassigned device rows in that bounded V3 representation, then release chunk workspace. Do not conditionally drop policy paths or oracle query/result/path/status rows, and do not replace required tensors with aggregate counters or a digest. A separately accepted shared-ABI amendment may revise the retention contract; until then, summary-only oracle retention and policy-only path retention are prohibited. Never materialize a host list of all dynamic queries/goals/results or an unbounded per-query Python object table.

For **policy queries**, retain all route query/result fields required by the V3 schema: query metadata and active mask, explicit source/query association, status, feasibility, path length, path and path-point mask, distance, endpoint error, weighted cost, cost units/provenance, and snapshot binding. The policy destination capacity is at most 633 rows. Selected-row command/evidence binding must resolve `policy_local_index → policy_source_row → policy_source_row_to_route_query_ordinal → route result`; it may not depend on coincidentally equal numeric indices.

For **oracle queries**, retain the complete fixed-capacity V3 route query/result fields for every oracle ordinal, including inactive slots: query and result tensors, active mask, path-point mask, route path, path length, status, costs/provenance, and explicit tagged identity/source binding. The full table is bounded only after the observation/route owners freeze the production E and `Q_total` profile and overflow behavior; source evidence `E≤600` is not itself a production-profile approval. Online counts, cost summaries, coverage counters, and an ordered route-pool receipt/digest may be added, but none substitutes for required V3 rows or paths. Stream each chunk into its predetermined device table slice and release only chunk-local workspace. A separately accepted ABI amendment is required before summary-only oracle retention or conditional path retention can be used.

Do not call a count ratio “recall” unless the owners define its denominator, matching/equivalence rule between policy candidates and oracle variants, invalid-target treatment, and empty-reference behavior. Until then, publish neutral summary names such as `policy_feasible_count`, `oracle_feasible_count`, and `oracle_min_cost`; retain the current `within_candidate_pool_recall_measured=false` limitation. The owners may preserve or replace the legacy `candidate_recall` field only after agreeing on its exact semantics and display label.

## Digest and receipt proposal; encoding remains an owner decision

Keep the static candidate-table order digest separate from the dynamic route receipt. The route receipt must bind, at minimum:

- snapshot and schema/order version;
- `G_policy`, `E`, `Q_total`, and the ordered chunk-span sequence;
- for each route ordinal, its tagged policy-row or oracle identity, active/invalid state, route status, path length, cost-unit/provenance code, and route cost fields included by the selected retention profile;
- the final oracle summary values and identity of the stable minimum-cost oracle row;
- source-to-query association used for each policy route result.

Propose a device-side ordered streaming digest state with constant-size state beyond the current chunk. Its exact primitive, byte/field encoding, integer widths, float canonicalization, seed/domain separation, update order, output dtype, overflow behavior, and invalid-row encoding require explicit shared-ABI/PPO/route-owner approval. Do not implement an unspecified digest, use host hashing of dynamic tensors, or claim that a static metadata digest covers dynamic goals/results. If only an opaque receipt is needed in V3, the ABI owner must specify its type/capacity and how the digest is validated.

## Decisions required from named owners before implementation

1. **Plan C and route owners:** approve the five distinct coordinates, policy-first then variant-major oracle order, explicit policy-row-to-query map, and whether the current order is frozen/versioned. Resolve whether every policy query has exactly one route query (including inactive continuation) and whether any future policy row can map to multiple queries.
2. **Observation/native owner:** define stable target-slot identity and generation/reuse semantics, widths, invalid sentinel, and whether canonical target ordering uses slot order or another deterministic key. The current track-index sort is not enough to settle this.
3. **PPO/shared ABI owners:** preserve the accepted V3 route query/result fields for both policy and oracle rows, including inactive rows, query-active/path-point masks, route paths, path lengths, and route statuses. Specify the per-snapshot/device capacity and overflow behavior, exact digest/receipt field layout, and selected-row-to-route-result binding in V3. Observation/route owners must freeze the configured additional target-slot capacity E (including inactive slots) or provide the actual bound; source evidence `E≤600` alone is not a production profile.
4. **Curriculum/evaluation owner:** define the intended recall/coverage metric, matching relation between selectable policy candidates and expanded oracle variants, and empty/invalid reference behavior. Otherwise keep the value as descriptive counts/cost summaries and do not claim recall.
5. **Cross-owner Pass 1 reviewer:** recheck the exact planning base, accepted route artifact, older accepted shared-ABI schema facts, current planning-only shared-ABI coordination proposal, and proposed reconciliation input; verify source citations and formulas, that inactive target slots remain explicit, that required V3 route rows/paths/statuses are retained, and that the proposal does not weaken the <=64 route ABI or make oracle rows selectable. Implementation remains unauthorized until decisions, any required shared-ABI changes, and exact-head approvals are complete.

These decisions cannot be inferred from existing `RouteQuery`'s candidate string, robot `world`/`slot`, the device track-index sort, current route-cost averages, or the 64-query call helper. In particular, the source does not settle target generations, true recall semantics, fixed production E profile, the complete `Q_total` V3 capacity/overflow profile, or digest encoding; V3 path retention is already required by the pinned shared-ABI schema facts.

## Scope and authorization

This proposal changes no source code, tests, PPO ABI, training/checkpoint/campaign state, hardware leases, dashboard, replay, or evaluation protocol. It performs no import, build, runtime, device/GPU, simulator, or dashboard operation. It makes no training, Stage 1, simulator, or whole-compute GPU-only claim. It authorizes only owner review and a follow-up planning amendment. Code work must wait for the requested owner decisions and exact independent reviews; live training/device leases remain untouched.

## Evidence pointers at `e37b2f30…`

- Candidate row construction and continuation: `frc_defense/meco_candidates.py:1291-1382`.
- Oracle variant and target order / current CPU materialization: `frc_defense/meco_candidates.py:1421-1453`.
- Policy route selection by source-global indices: `frc_defense/meco_candidates.py:815-830`.
- Existing bounded oracle statistics and recall label: `frc_defense/meco_candidates.py:885-917`.
- Route identity fields: `frc_defense/experimental_route_evaluation.py:112-148`.
- 64-row chunk map: `frc_defense/experimental_route_evaluation.py:21-64`.
- Current chunk helper full tuple/materialize/concatenate behavior: `frc_defense/experimental_route_evaluation.py:631-703`.
- Accepted per-call route shapes and fields: accepted route-plan amendment `8567fd92…` / `ac5847d6…`, including the route batch schema in “Required vertical slice.”
