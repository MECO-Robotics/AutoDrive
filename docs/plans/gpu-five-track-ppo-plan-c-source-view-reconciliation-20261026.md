# Pass 1 amendment: reconcile Plan C source capacity with PPO V3 views

**Status:** Proposed planning-only Pass 1 amendment. It authorizes no source
implementation, tests/imports, compilation, runtime/device/simulator/dashboard
work, checkpoint access, training/evaluation, or campaign change.

**Planning base:** `origin/main` commit
`e37b2f30b6bc51b74dd72bf039e94a174ec51bf2`, tree
`8edb9998d7e3e361e83938a345c65d4fef488e1d`.

**PPO V3 source-inventory authority:**
`docs/plans/gpu-five-track-ppo-v3-schema-source-evidence-followup-20261026.md`,
artifact SHA-256
`fe4f853ebc1332407b8c40410858aa51ade9daf279c4d4baea98cb10694d850f`.
This amendment reconciles that inventory's source bounds with its cited Plan C
owner contract; it does not revise the shared PPO ABI or settle other V3 fields.

**Plan C K/K+1 owner contract:**
`gpu-plan-c-k-capped-model-view-contract-20261026-amendment.md`, accepted
owner artifact commit `4cdc30f923c0d948b43c57d452c4ee4b17cf61d9`, artifact
SHA-256 `6d9c0e6320acd503c8595fa619e8beeba24af88f36e49337d7325331a899ff0b`.
Its normative decision is K ordinary policy-ranked candidates plus one
optional dynamically revalidated continuation choice. The model receives two
separate K-wide feature views; the device selection vector has K+1 choices.
The source-global row map, not a local view index, carries identity through
route/action/PPO evidence.

**Plan C route authority:** `docs/plans/gpu-resident-route-decision-slice-20261025.md`,
accepted amendment commit `8567fd92d4dc9e214e8502f2249bb02203ac8c03`, artifact
SHA-256 `ac5847d65de2af5ccb17d6a1bc109dceefeab71c5d707253980570020c1e885a`.
Retain its route cost/status/provenance and device-residency requirements.

## Audit finding

There are three different extents, not one conflicting capacity. The
`K+1`-versus-`G_policy` comparison is a **coordinate/capacity conflation**.
There is a separate, real implementation gap between the accepted compact
view and the current producer/consumer interface, plus a source-row naming
ambiguity between the accepted Plan C text and the PPO inventory:

1. **Policy model input:** each of the ordinary and continuation views is
   `[1,K,40]`, for `K ∈ {8,16,32}`. The ordinary view has K eligibility-filtered,
   rank-ordered rows. The continuation query uses slot 0 of its own K-wide view;
   unused entries are masked padding. C1/C2 return K ordinary utilities and one
   continuation utility. The post-model device selector sees K+1 choices when
   continuation is valid. These are not a single `[1,K+1,40]` model input.
2. **Full policy source table:** the current device producer has 8 base rows,
   E sensed-target slots, 24 approach-offset rows, and one continuation row.
   `E` is bounded by the environment fuel/perception capacity, `0 ≤ E ≤ 600`.
   Thus `G_policy = 8 + E + 24 + 1 = 33 + E ≤ 633`. This is the source-global
   policy/identity and route-association axis, not the C1/C2 input width.
3. **Expanded route work pool:** route evaluation additionally enumerates 18
   oracle variants for each of 4 base target slots and E sensed-target slots.
   Therefore `Q_total = G_policy + 18(4 + E) = 105 + 19E ≤ 11,505` route
   queries. The accepted route ABI's per-call bound is 64, so this envelope
   requires at most `ceil(11,505/64) = 180` calls/chunks when streamed as one
   ordered sequence. Oracle-only rows support recall/evidence; they are not
   C1/C2 inputs or selectable policy rows.

The accepted K-view plan defines a source-global row as an index into the full
per-decision `specs` list. At the current producer, that list also contains
oracle-only route specs, so this coordinate can range over the Q_total route
pool. The PPO V3 inventory separately names the policy table as
`G_policy=33+E` and leaves oracle-row inclusion in the transition unresolved.
That source-global **identity extent/meaning is a genuine cross-owner
ambiguity**. This amendment proposes to make the transition's
`policy_source_row` range `0..G_policy-1`, use a separate `route_query_ordinal`
range `0..Q_total-1`, and bind each query to either a policy row or an
oracle-only identity. The first G_policy route queries currently have the same
numeric ordinal and policy-row index, but the two fields remain distinct in
the schema. This explicit policy-row mapping requires Plan C and shared-ABI
owner approval because it narrows the accepted K-view phrase “full `specs`” to
the policy source table; no implementation may assume that interpretation
before approval.

The current producer retains a full ordered metadata/tensor axis, assigns
`selection_rank` and a `[1,G_policy]` `model_input_mask`, and keeps continuation
at the source-global row after the ordinary rows. The current C1/C2 inference
path still forms a full-width feature tensor and masks rows before model/broker
inference; C2's current mask also admits continuation to its pooled candidate
context. This does **not** implement the accepted two-view K-wide protocol,
which pools ordinary rows once and scores continuation against that shared
state. The current implementation also does not change the source table's
33+E bound or the expanded route pool. The PPO V3 inventory's source capacities
therefore do not contradict the Plan C model-view contract and must not be used
to widen it.

## Source evidence at the planning base

All line citations below refer to source blobs at `e37b2f30…` and must be
rechecked by Pass 1 review against that exact tree.

| Fact | Evidence at planning base | Disposition |
|---|---|---|
| K choices | `frc_defense/meco_track_training.py:154-155`; candidate bounds/config are in `meco_candidates.py:35-40`. | `{8,16,32}` is the model-view K set, not total source capacity. |
| Policy rows and order | `meco_candidates.py:1291-1313` creates 8 base plus E sensed rows plus 24 offset rows; continuation is reserved/appended at `:1320-1346` and remains at the source row after those rows. | Full policy source axis is `G_policy=33+E`; IDs are generated from the full ordered `specs` at `:575-619`. |
| Stable ordinary policy view | `meco_candidates.py:788-835` ranks base rows, eligible sensed rows, and offsets, then retains the full metadata order and computes a prefix eligibility mask; continuation is separately appended at `:815-835`. | Accepted implementation must compact by device row map into the K-wide ordinary view and preserve the continuation source row independently. The current full-axis mask is not an equivalent ABI shape. |
| Feature/missingness widths | `meco_tracks.py:379-384,771-782`; current features concatenate raw and missingness values in `meco_candidates.py:777-784,836-839`. | 20 raw + 20 missingness columns = 40 per view row. Preserve registered feature order. |
| Current model interface and C2 context | `meco_tracks.py:1343-1367` checks full width against `len(table.candidate_ids)` and zeros rows under `model_input_mask`; C2 pooling is at `:1127-1139`. | This is full-axis masked inference, not the accepted two-view interface. It is a source mismatch requiring implementation work, not a reason to alter the owner contract. |
| Sensor capacity | `tensor_3v3.py:72-73,368-372` bounds fuel slots by 600 and allocates per-slot position/mask buffers; `_tensor_3v3_observation.py:83-98` returns focal-slot views. | E inherits the configured slot-capacity bound; identity/generation semantics remain an observation/native/PPO owner decision. |
| Route pool construction | `meco_candidates.py:1421-1453` defines 18 oracle offsets and appends each to the 4 base plus E target slots. The full `specs` tuple also contains 8 base + E + 24 offsets + continuation. | Policy-route rows and oracle-only rows must have separate identities and use stable query ordinals. |
| Route chunk primitive | `experimental_route_evaluation.py:21-64` fixes per-call `MAX_ROUTE_QUERY_BATCH=64` and source-ordered chunk maps. `:631-660` currently converts `queries` to a tuple, builds all spans, evaluates chunks, and accumulates chunk outputs. | The per-call limit is source-bounded; the current helper still materializes the complete query metadata and outputs. It does not prove bounded streaming of the full 11,505-query pool. |
| Static IDs and current digest | `_build_device_candidate_set` derives `policy_ids` and `policy_queries` by indexing the full `ids`/`query_rows` with `policy_indices` (`meco_candidates.py:815-830,851-855`). The full `ids` tuple also includes oracle-only specs. | Define immutable static policy IDs/schema/snapshot digest over G_policy rows, plus a separate static oracle-query identity/schema and ordered route-pool receipt. Do not silently treat the full internal `ids` tuple as the PPO policy-ID tuple. Dynamic row maps/content require device-produced evidence and cannot be covered by host metadata digest. Exact encoding remains joint Plan C/PPO/ABI decision. |

## Reconciled capacity and identity contract to be adopted

This amendment proposes the following owner mapping for the next Plan C/PPO
Pass 1 decision. It does not authorize implementation until the Plan C,
route, observation/native, PPO and shared-ABI owners accept their parts and
exact hashes are reviewed.

### 1. Full policy source table: `G_policy`

- Source-global policy rows are ordered as: base rows `0..7`; E sensed-target
  rows `8..7+E`; 24 approach-offset rows `8+E..31+E`; and continuation row
  `32+E`. The total count is `G_policy=33+E`, maximum 633.
- Static host policy identity is an immutable ordered `candidate_ids` tuple of
  length `G_policy`, plus snapshot, feature-schema version/order and static
  table-order digest. The digest covers the static tuple/schema/snapshot
  binding only; it does not claim to cover device values or dynamic maps.
- Device source columns share the same source-global row coordinate and device:
  action, target/track identity, goal/heading, override flags, raw features,
  missingness, eligibility/validity, route status/path/length/cost/provenance,
  forecast/risk where registered, and candidate-to-query association.
- The ordinary view map is `ordinary_row_map[j] = selected_policy[j]` for
  `j=0..K-1`, where `selected_policy` is the stable canonical device rank over
  eligible base, sensed-target, and offset rows. Apply eligibility before the
  top-K cap. At K=16/32, entries may point to non-prefix source-global rows.
- The continuation view's active slot 0 maps to the existing source-global
  `continue_index=32+E`; otherwise all continuation mask entries are false and
  row-map entries use the accepted sentinel. Preserve the existing
  `continue:inactive` / `continue:<candidate ID>` descriptor and its original
  action, target, goal, route and commitment revalidation. Do not reserve an
  ordinary slot for continuation.
- Device utility/uncertainty outputs are K ordinary values and one continuation
  value. Build selection support/order on device as ordinary local slots
  `0..K-1` followed by continuation local slot K. Resolve a selected local slot
  through the corresponding map before gathering source-global command,
  descriptor, route, and PPO identity. Never write local indices as source IDs.
- Preserve the existing deterministic fallback precedence: feasible base
  fallback, otherwise feasible hold, otherwise the lowest-index minimum-cost
  feasible row. Map that source-global row into the current compact view on
  device. If no K+1 choice is feasible, retain the typed no-feasible path;
  never sample a structural placeholder or fabricate a valid continuation.

### 2. Route query pool: `Q_total`, distinct from `G_policy`

- Use deterministic global query ordinals over the policy route rows followed
  by oracle-only rows. A route row carries a distinct `route_query_ordinal`
  plus `policy_source_row` (0..G_policy-1) or a typed oracle-only identity.
  The current order makes policy query ordinal q equal its policy source row r
  for q=r<G_policy, but they are separate schema fields, not interchangeable
  indices. Oracle query identity is a typed code/tuple for its source target
  slot and oracle variant, is explicitly marked nonselectable, and must not
  alias a model-view or policy source row. The shared ABI owner must approve
  this policy-row interpretation as an amendment to the K-view contract's
  broader `specs` coordinate.
- Stream query specs and dynamic goals through chunks of at most 64 in stable
  source order. Each chunk carries its global query start/count and a
  device-resident global-to-local chunk mapping. Write policy-route results to
  their `G_policy` destination rows. Consume oracle-only results into the
  required on-device recall/summary accumulators and route digest; do not append
  oracle rows to C1/C2 input or action selection.
- At maximum E=600, the producer must account for 633 policy route queries plus
  10,872 oracle queries (11,505 total, 180 chunks maximum). The implementation
  must not allocate a full `[11,505,...]` route-path tensor or materialize a
  full dynamic query table on host. If exact V3 replay needs any additional
  per-oracle row fields, the route/PPO owners must set an explicit bounded
  device retention profile or opaque digest/summary contract before coding.
- For each query, preserve query ordinal, source row or oracle identity,
  snapshot, goal binding, route status, path length/path content as required by
  the accepted ABI, weighted-cost units/provenance, and associated route result.
  The selected policy row's route and cost must remain exactly associated with
  the global row chosen through the K/K+1 mapping.

### 3. Digest and PPO V3 binding

- Keep the static `candidate_table_order_digest` distinct from every dynamic
  device digest. Freeze one device digest coverage proposal across the ordered
  full policy source table: source-global indices/row count, feature-schema
  identity, all registered feature/missingness and decision columns, validity,
  route status/path-length/path representation/cost units and source mapping.
- A separate route-pool digest/receipt must bind `Q_total`, query ordering,
  each policy-row or oracle identity, chunk spans, goal/source mapping, result
  status, path/cost provenance, and the recall summary inputs. The PPO transition
  binds the selected source-global row, selected descriptor/action, selected
  route result, source-table digest, relevant route digest/receipt and snapshot.
- These are coverage requirements, not an approved hash algorithm or ABI field
  layout. Plan C/PPO owners must propose the exact device-byte encoding,
  deterministic ordering, digest primitive/output dtype, sentinel rules and
  overflow/failure behavior. Shared-ABI owner must approve field names/types and
  capacities. No host `.item()`, `.tolist()`, `.cpu()`, NumPy conversion, Python
  formatting or host digest may inspect dynamic tensor content to construct
  identity, digest, status or a training row.

## Decisions required before source work

1. **Plan C owner:** confirm the two K-wide feature views, K+1 selection order,
   exact continuation placement, policy source-row versus route-query
   coordinates, static ID tuple scope, and device-side dynamic digest coverage
   above. Reconcile the accepted
   Plan C K-view artifact with current full-axis producer, adapter, broker and
   training source paths; publish a versioned owner amendment rather than
   claiming the full mask already satisfies compact views.
2. **Route owner:** accept the `Q_total=105+19E` derivation, policy/oracle
   identity separation, ordered 64-row streaming, route result destination map,
   selected route evidence and bounded oracle recall/digest retention. State
   whether all path rows or only bounded summaries are required for oracle-only
   rows under the ABI.
3. **Observation/native owner:** pin E's production capacity/profile, sensed
   target slot identity and reuse/generation semantics, and invalid-sentinel
   behavior. `fuel_count≤600` is a current source bound, not a frozen V3
   identity lifecycle contract.
4. **PPO/shared ABI owners:** adopt policy-source-global versus policy-local
   versus route-query coordinates and explicitly approve the proposed
   narrowing of the K-view plan's `specs` row identity; decide exact field dtypes/tail dimensions,
   `G_policy≤633`, route streaming receipt and digest/status encoding; confirm
   whether V3 stores the full source table, selected row only, or a bounded
   source-table/route receipt. Do not use the PPO row capacity as the C1/C2
   feature width.
5. **Cross-owner reviewer:** verify the exact source/plan blobs and formulas,
   and reject any implementation authorization until all owner answers and the
   shared ABI revision are independently Pass 1 accepted. A future source
   candidate still needs exact-head Pass 3 and the separately deferred device,
   simulator, replay and dashboard gates.

Until all five decisions are approved, the following remain unresolved and
must fail closed in any schema proposal: digest algorithms/byte encoding,
exact route-query and source-row V3 field layouts, oracle result retention,
target generation width/reuse, invalid/padding sentinel codes, fixed profile
for E, and whether per-oracle path details are retained. No source code is
authorized by this amendment.

## Explicitly out of scope

This artifact changes no source, PPO ABI, model weights/checkpoints, training
schedule, campaign state, hardware assignment, dashboards, reports or
qualification protocol. It establishes no implementation, test, device parity,
simulator outcome, replay, dashboard-currentness, qualification, Stage 1 pass,
or campaign readiness. Keep the live campaign/checkpoints/device leases
untouched. Its only permitted follow-up actions are exact-source review, plan
editing, static diff formatting, and publication of the planning branch.
