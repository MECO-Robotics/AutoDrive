# Pass 1 proposal: PPO V3 coordinate field table

**Status:** Planning-only amendment proposal for independent Pass 1 review and
cross-owner decisions. It authorizes no schema declaration, source change,
tests/imports, compilation, runtime/device/simulator/dashboard work,
checkpoint access, training/evaluation, or campaign change.

**Planning base:** `main` commit
`e37b2f30b6bc51b74dd72bf039e94a174ec51bf2`, tree
`8edb9998d7e3e361e83938a345c65d4fef488e1d`.

**Shared ABI authority being amended:**
`docs/plans/gpu-shared-command-ppo-transition-abi-20261026.md`, commit
`6d46846b87a7790396ef5e45868869c66ac173f6`, artifact SHA-256
`19349ebd40bbe6c0f11e8001bb716524d0273f8f91862f040096883faf3eb338`.
This proposal does not revise that immutable artifact.

**Planning inputs:**

- PPO V3 source inventory, SHA-256
  `fe4f853ebc1332407b8c40410858aa51ade9daf279c4d4baea98cb10694d850f3`.
- Plan C/PPO source-view reconciliation, SHA-256
  `ad3b9f69fb5ade7ebe0844be2a3b3105ffcebcd7f708969631efc9f8133e1924`.
- Plan C owner proposal, SHA-256
  `9caa8bdd98890c2ec3e9b3ed239d58f4b7b78b789bc873a6c0bd2db37770dbc0`.
- Route owner streaming proposal, SHA-256
  `2ce8c4761f1f6224432993a03bfc176e4f257c4cde2781d9a925c3ed986f1e67`.
- Observation/native target-slot proposal, SHA-256
  `e726ddaad146139f33c3beca4e4e698b9ce3928b2a4fe5701e19bd35202be7fb`.

These are accepted planning inputs for this amendment, not a claim that each
proposal is an owner-approved normative contract. Their proposed fields,
identity semantics, retention choices, or owner assignments remain
non-normative until the named owners explicitly adopt them in a reviewed
shared ABI revision. Where they disagree with the ABI, the ABI remains the
current authority until changed through that process.

## Purpose and coordinate rules

`FiveTrackPpoTransitionV3` has one track-tagged transition row. Its
`FiveTrackPpoBatchV3` representation is the same typed device columns with a
leading fixed rollout axis `[R,...]`. The transition must preserve each
coordinate in its own domain. Numeric equality between two coordinates in a
particular ordering does not make them aliases.

For C tracks, selection proceeds from a compact policy-local view to the
full policy source table, and route evidence is addressed by an explicit
global route-query ordinal. Oracle-only queries have no selectable policy
source row. Chunk coordinates are a third routing-work coordinate and do not
replace query ordinals. For A tracks, `policy_local_index` and
`policy_source_row` retain the accepted local/global Plan A support meanings;
C-only continuation/oracle fields use typed neutral values and masks.

## Coordinate field table

`int64` below corresponds to PyTorch `torch.long` in the current source. A
source upper bound is not a frozen production capacity. `R` remains the
rollout capacity symbol from the ABI and is intentionally not inferred here.

| Field | Proposed transition type and batch column | Source/contract bound | Binding and decision status |
|---|---|---|---|
| `policy_source_row` | `int64` scalar; `[R]` | C policy source rows are ordered as 8 base + E sensed-target + 24 offsets + 1 continuation, so `G_policy=33+E≤633`; A uses global support row. ABI already fixes `source_global_row:int64`. | Canonical name for the selected full support/table coordinate; bind to the ABI `source_global_row` alias without host conversion. For C, it indexes only the policy source table, never the expanded oracle pool. This interpretation must be adopted by Plan C and ABI owners because the reconciliation identifies an ambiguity with “full specs.” `E` profile remains unresolved. |
| `policy_local_index` | `int64` scalar; `[R]` | ABI fixes int64. For C, ordinary selector slots are `0..K-1`, `K∈{8,16,32}`; continuation, when enabled, is selector slot `K`. For A it is the Plan A local support row. | Bind the selected output slot before mapping to `policy_source_row`. Must never be used as a source-table or route-query index. |
| `selected_is_continuation` (continuation selector) | **Do not store as an independent dynamic field in the initial table.** Derive device-side `bool` as `(track_code is C) AND (policy_local_index == K)` and validate against the continuation-valid mask/map. If an owner requires materialization, its proposed type is `bool` `[R]`. | K/slot semantics are fixed by the accepted Plan C K/K+1 contract and ABI; an independent flag, neutral A value, and contradiction handling are not fixed. | Avoid duplicate contradictory state. Explicit owner decision is required before adding a serialized selector column. Continuation identity still maps via `continuation_row_map[0]` to `continue_index`; it is not itself the source row. |
| `route_query_ordinal` | `int64` scalar per association; `[R]` for selected route or a separately typed route association object | Current policy-to-route work is `Q_total=105+19E≤11,505` and per-call `Q≤64`, hence at most 180 chunks under the maximum source bound. ABI fixes query map and `candidate_index` to int64, with concatenated index as global query ordinal. | Selected transition must bind through `policy_local_index → policy_source_row → policy_source_row_to_route_query_ordinal → route result`. Although the current policy-first order has matching numbers, the map remains explicit. Owner decision: whether the transition retains only the selected ordinal plus receipt, or all policy query associations as table fields. |
| `route_query_kind` | Proposed typed integer code, width unresolved; device scalar/per query | Source/reconciliation distinguish policy rows from oracle-only rows; no shared code registry or width is fixed. | Required tag if policy/oracle identities share a route stream. ABI/route/PPO owners must choose dtype, code registry, invalid value, and versioning. Do not encode oracle rows as out-of-range policy source rows. |
| `oracle_identity` | No single packed field is approved. Proposed typed tuple per oracle query: target kind, observation/native target handle (where applicable), oracle variant code, and snapshot binding; exact numeric widths and shape unresolved. | Source yields 18 oracle variants over 4 base plus E sensed targets; maximum oracle count `18(4+E)≤10,872`. Observation/native proposal distinguishes anonymous sensed-slot generation from physical gamepiece identity. | Oracle query identity must be unique within the snapshot and nonselectable. Route/observation/native/PPO/ABI owners must agree on canonical target order, target generation, variant codes, invalid sentinel and whether base targets need a separate identity form. No physical FUEL ID may be substituted for a sensed-target handle. |
| `route_chunk_index` | Transient route-map field: `int64` `[Q_total]`; **not presently a required PPO row column** | Existing `DeviceRouteChunkMap.chunk_index` is `torch.long` on device and `floor(global_row/64)`. At maximum `Q_total=11,505`, domain is `0..179`. | Derivable from global ordinal and fixed chunk size. Persist only if a reviewed replay/receipt contract requires it; otherwise it is execution metadata. If persisted, bind its schema/version and verify it equals the derived value on device. |
| `route_chunk_global_start` | Host-known immutable span metadata, represented in source as exact integer span endpoints; if materialized on device, proposed `int64` `[num_chunks]` | Each span is `[start,min(start+64,Q_total))`; maximum starts are `0,64,...,11456` for `Q_total=11,505`. Source span tuple uses Python integers; device row coordinate is int64. | Not a per-transition field. Its persistence, width, envelope placement, and digest coverage are owner decisions. Do not duplicate a chunk start into each PPO row or use a local chunk row as global ordinal. |
| `route_chunk_local_index` | Transient route-map field: `int64` `[Q_total]`; not a PPO row field by default | Existing source computes `global_row % 64` as `torch.long`, domain `0..63`. | Execution coordinate only; it does not identify a transition or candidate. Persist only if required by approved replay contract. |

### Capacity interpretation

The source-derived formulas are:

```text
E = configured sensor-buffer capacity for one environment instance
G_policy = 8 + E + 24 + 1 = 33 + E, with source ceiling 633
Q_policy = G_policy
Q_oracle = 18 * (4 + E), with source ceiling 10,872
Q_total = Q_policy + Q_oracle = 105 + 19E, with source ceiling 11,505
chunk_size = 64; chunks = ceil(Q_total / 64), source ceiling 180
```

Current environment source permits more than one `fuel_count` and defaults to
504; it does not establish a universal V3 production value of E. The
observation/native proposal suggests E equals allocated `fuel_count`, but
that profile choice still needs explicit adoption. Thus `633`, `11,505`, and
`180` are proven maxima under current source bounds, not necessarily the
allocated batch profile. `R`, production E, and whether route arrays are
stored per row, per envelope, or streamed into a bounded receipt remain
unresolved.

## Required transition bindings and masks

The following checks are proposed as device predicates; no Python consumer may
read dynamic coordinate values, masks, route statuses, identities, or digests.

1. **Selection map:** for C, the chosen local slot is in the active compact
   support; ordinary local slots map through `ordinary_row_map`, while local
   K maps through the active continuation map to `continue_index`. The mapped
   source row is in `[0,G_policy)` and its candidate identity, action,
   descriptor, goal, and route columns agree. For A, the local support row
   maps to the exact Plan A global support row and preserved command payload.
2. **Policy/route association:** every retained policy route result has an
   explicit `route_query_ordinal` and source-row association. Query ordinal
   uniqueness and any policy-row-to-query multiplicity rule must be fixed by
   the route owner. Do not depend on current coincidental equality of ordinal
   and source row.
3. **Oracle exclusion:** an oracle query is tagged oracle-only and has no
   selectable `policy_source_row`; it cannot populate model support, action
   selection or selected-action/logprob binding. The policy-vs-oracle tag and
   tagged identity are validated with the route snapshot.
4. **Chunk integrity:** each streamed chunk has ordered global span
   `[start,stop)`, `0≤stop-start≤64`, matching derived global, chunk and local
   coordinates. Missing, repeated, reordered, malformed or nonfinite chunk
   results fail the full snapshot closed according to the route contract.
5. **Lifecycle masks:** keep policy eligibility, continuation-validity,
   route-query-active, route-path-point, rollout-slot-valid, command-valid,
   actor-valid, critic-valid, terminal, truncated, transition-finished,
   bootstrap-continuation and reset masks distinct. Dtypes are bool where the
   ABI explicitly fixes them; exact axes and any additional status masks
   require owner adoption. A false mask cannot be overridden by finite padded
   coordinates or a sentinel-like numeric value.
6. **Selected transition:** selected local slot, selected full source row,
   candidate identity, route ordinal/result, selected command and action,
   old log probability/value, macro selected-vs-executed actions, reward,
   lifecycle masks, recurrent state and RNG state all bind to exactly one
   rollout row. Remasking does not rewrite the sampled slot or old log
   probability. PPO must gate the complete optimizer update on failed row or
   digest validation as specified by the ABI.

The shared ABI already fixes `policy_local_index:int64`, `source_global_row:int64`,
`candidate_identity_code:int64`, route row-map/index types, route activity and
path masks, and the separation of local/global mappings. This amendment
proposes the clearer canonical alias `policy_source_row` for
`source_global_row` and the additional route/oracle/chunk coordinate treatment
above. Exact names, transition-vs-envelope placement, oracle tag registry,
invalid codes, status granularity and per-query/per-row tensor axes require an
updated shared ABI and cannot be inferred from this planning proposal.

## Route V3 retention and digest boundary

Retain the complete policy-query evidence needed by selection and V3 replay
for at most `G_policy≤633` rows, subject to a frozen E profile: active/status,
feasibility, path length, path and path-point mask when required by the ABI,
distance, endpoint error, weighted cost and units/provenance, explicit source
row/query association, target/snapshot binding, and route receipt. The current
shared ABI defines per-call route result types (`P=73`, `Q≤64`), but it does
not decide whether every full path remains in `FiveTrackPpoBatchV3`, is
per-row retained, or is validated and represented by an opaque device receipt.

For oracle-only queries, the route proposal suggests releasing each chunk's
path storage after device-side accumulation of bounded counts/cost summaries,
stable minimum-cost identity and an ordered route-pool receipt. This is a
proposal, not an adopted retention rule. Before source work, route/PPO/ABI and
evaluation owners must select one of: (a) full per-oracle route fields/paths
with a pinned bounded capacity; (b) per-query scalar fields with a pinned
capacity; or (c) a precisely specified bounded device summary and digest
receipt. The selected form must preserve the evaluation evidence required by
the training plan; no silent truncation, host recomputation, or weakened
replay semantics is allowed. Until then V3 route retention is unresolved and
producer work is prohibited.

Separate **static identity** from **dynamic device evidence**:

- Static immutable metadata may bind schema/profile version, E, candidate ID
  order, feature schema/order, route order version, and configuration labels.
  The static `candidate_table_order_digest` covers only those immutable
  declarations.
- Device-computed dynamic evidence must bind the selected source row, local
  mapping, continuation state, route query ordinal/tag/identity, target and
  snapshot generation, chunk span/ordering, route statuses/results, masks,
  and selected command/transition linkage required by the approved coverage
  definition. A static hash cannot stand in for these values.
- The exact digest primitive, domain separation, byte order, field order,
  float canonicalization, output dtype/shape, status registry, sentinel
  encoding, overflow/failure handling, opaque record capacity and replay
  retention are unresolved. They require PPO/route/observation/native/shared
  ABI decisions. Dynamic fields remain on the selected device; the host may
  persist an opaque byte buffer unchanged only after its exact envelope is
  approved.

## Owner decisions required before the next ABI revision

1. **Plan C owner:** adopt or reject `policy_source_row` as the policy-table
   index, explicitly separated from route ordinal and oracle identities;
   adopt selector K / continuation mapping and say whether a redundant bool
   is prohibited or required. Pin the supported E profile or parameterized
   profile set, row-map sentinel and policy-route association multiplicity.
2. **Route owner:** freeze route query-kind and variant codes, canonical query
   order, explicit policy-row-to-query map semantics, target/snapshot binding,
   chunk receipt layout and the required oracle retention option. Confirm
   whether the current 64-query spans and 11,505 maximum are the V3 limits or
   only current source bounds.
3. **Observation/native owner:** adopt target handle fields, generation
   lifecycle and reset/snapshot sequence boundaries, E capacity profile,
   signed widths, invalid sentinel and overflow behavior. Keep anonymous
   sensed-track identity distinct from physical piece identity.
4. **PPO owner:** publish exact `FiveTrackPpoTransitionV3` and
   `FiveTrackPpoBatchV3` typed columns, including `R`, per-track neutral
   representation, field axes, continuation selector treatment, query/map
   retention, selected-vs-executed binding and digest/status coverage.
5. **Manifest/artifact owner:** choose static provenance scope and immutable
   profile identity; specify versioned opaque envelope, byte capacity,
   alignment, length, overflow, reader fail-closed behavior and compatibility
   boundary. Keep dynamic bytes opaque to host readers.
6. **Shared ABI owner:** after decisions are independently reviewed, publish a
   new immutable ABI document/hash with no symbolic unresolved capacity or
   untyped dynamic coordinate. Every dependent owner plan must adopt that same
   hash. A changed coordinate, mask, retention field or producer path returns
   to Pass 1.

Each owner answer must name its exact source commit/tree and supporting line
citations; identify which values are source facts and which are explicit
owner decisions; and receive independent Pass 1 review. Do not treat the four
planning proposals listed above as unanimous owner approval merely because
they were accepted as planning inputs.

## Next amendment and implementation sequence

1. Independently Pass 1 review this exact proposal. Correct only the planning
   artifact if the field table or source bounds are wrong.
2. Obtain the owner decisions above as immutable planning amendments, each
   exact-head reviewed. Reconcile any conflict before changing the ABI.
3. Produce a new shared ABI field-table amendment that adopts the agreed
   normalized coordinates, types, profile capacities, transition/route
   bindings, masks, route V3 retention, static/dynamic digest boundary and
   opaque envelope. Review it independently at Pass 1; calculate and pin its
   exact SHA-256.
4. Have every owner plan adopt that exact ABI hash and its own exact path
   allowlist, then independently Pass 1 review those plans. Only then may a
   separately planned schema/type shell be proposed for
   `frc_defense/meco_ppo_batch.py`.
5. Implement in isolated worktrees from one pinned source head. Require
   independent exact-head Pass 3 for each candidate and the combined head
   before integration. Runtime/device/simulator/dashboard/campaign gates
   remain governed by the ABI plan and live-campaign ownership rules.

## Evidence pointers at planning base

- Current policy ordering, K selection and continuation source row:
  `frc_defense/meco_candidates.py:788-835,1291-1382` and
  `frc_defense/meco_track_training.py:154-155`.
- Feature widths and current model interface:
  `frc_defense/meco_tracks.py:379-384,771-782,1343-1367`.
- E source ceiling and per-observer buffers:
  `frc_defense/tensor_3v3.py:72-80,368-378` and
  `frc_defense/_tensor_3v3_observation.py:83-98`.
- Route variants and source order:
  `frc_defense/meco_candidates.py:1291-1382,1421-1453`.
- Route chunk mapping and int64 coordinates:
  `frc_defense/experimental_route_evaluation.py:21-64`.
- Existing route result schema and `Q≤64`, `P=73`:
  shared ABI §§ “Device route query/result ABI” and “V3 field schema
  amendment”; the cited immutable ABI hash above.
- ABI local/global coordinate and transition field groups:
  shared ABI §§ “Canonical device command” and “Canonical PPO transition and
  batch”.

No source code, tests, imports, runtime, device, simulator, dashboard,
checkpoint, campaign, or training activity was performed for this plan.
