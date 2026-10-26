# Pass 1 follow-up: PPO V3 schema source evidence and owner decisions

**Status:** planning-only Pass 1 follow-up proposal. It authorizes no source
implementation, imports, tests, compilation, device access, simulator,
dashboard/replay activity, checkpoint access, training/evaluation, or campaign
change.

**Source base:** integrated `main` commit
`61d3731e1531825ee7524065a158ba4765da0a84`, tree
`294ee27d46190fcb13d1f6eb69434ec1a60ac942`.

**ABI authority under review:** shared command/PPO transition ABI at commit
`6d46846b87a7790396ef5e45868869c66ac173f6`, document SHA-256
`19349ebd40bbe6c0f11e8001bb716524d0273f8f91862f040096883faf3eb338`.
Its existing V3 inventory is §§ “Canonical PPO transition and batch” and “V3
field schema amendment: source bounds and unresolved profile decisions”. This
follow-up adds source-path/line evidence at the source base above and assigns
the remaining cross-owner decisions. It does not alter that ABI.

**PPO source-preparation input:** `e0e3929c6b2e8f9109f77c1dc1105d0b367cffd7`,
artifact SHA-256
`248087999f2f1948b419ea67e012533794f4010f24b608b790f39c81c24833f8`. That
amendment pins older ABI commit `947fc80…` / SHA-256
`7650bbe8ec48dfa07557992f90b821d1dd125bc786f48e8698358ef5c27b308f`; it must
be re-adopted to the ABI authority above before any schema shell or producer
work. This is a dependency update, not permission to edit PPO source.

Accepted owner facts consumed for this inventory are the Plan A tensor-root
amendment `c0e9ae958108af3c56c4b64a709d9a4237cc7a3c` (SHA-256
`01e2507a1d4d458a2c8348323e119395d6f4953d988a53c4a7df2395c06f1c8a`), Plan C
K-view amendment `4cdc30f923c0d948b43c57d452c4ee4b17cf61d9` (SHA-256
`6d9c0e6320acd503c8595fa619e8beeba24af88f36e49337d7325331a899ff0b`), Plan C
route amendment `8567fd92d4dc9e214e8502f2249bb02203ac8c03` (SHA-256
`ac5847d65de2af5ccb17d6a1bc109dceefeab71c5d707253980570020c1e885a`), action
and macro plan `dd278c41fa0ed8e8bb6e0097a322ca05ebbc2282` (SHA-256
`b6b4dbd5fcfca0e91e7ad079c8f172a3e2202113220e8114909ca8ee3b687774`), and
gamepiece RNG v2 amendment `1ca7646672a39f0841db5760f1c2dbf8aa465180` (SHA-256
`953f1c31436d82523b90537ccaaf1df9ebbc7bc912f7cbc07343a09fbf0751a4`). These
facts do not fill any field left open by the ABI or owner plans.

The source-base refresh includes integrated Plan C K-prefix masking. The
accepted K-view owner artifact describes a compact K-plus-continuation view;
current source instead retains the full policy metadata axis, carries an
int64 `selection_rank` and bool `[1,G_policy]` `model_input_mask`, masks model
inputs outside the ranked K prefix, and samples in rank order before mapping
the chosen index back to the source row (`meco_candidates.py:788-835,872-884`;
`meco_tracks.py:1343-1362`; `meco_track_training.py:1817-1827`). This follow-up
records the discrepancy for a Plan C owner decision; it does not silently
rewrite the accepted artifact.

## Source evidence and disposition

The `dtype / shape / capacity` column separates what the current source proves
from an implementation-ready fixed schema. “Unresolved” means source may
describe a current runtime value, but no accepted owner contract freezes that
value for V3. Dynamic fields are required to remain on the selected simulator /
PPO device; static provenance labels may be host metadata under the ABI.

| V3 field or field family | Current source evidence: dtype, shape, capacity | Residency and owner(s) | Source derivability / decision required |
|---|---|---|---|
| Schema/track/protocol/source/runtime/config/checkpoint identity and seed label | ABI fixes command `schema_version:uint16`, `track_code:uint8`; it does not define V3 protocol/source/config/checkpoint field encodings or lengths. Current checkpoints store Python/JSON metadata and source hashes (`meco_track_training.py:2800-2819`). | Immutable host metadata; PPO, manifest and ABI owners | Static-vs-dynamic boundary is derivable from ABI; exact fields, encoding, scope (row vs envelope), version registry and compatibility rules are unresolved. PPO owner must publish one field inventory and manifest owner must confirm matching provenance keys. |
| `N`, `S`, `R`, world/robot slot maps, rollout slot/generation/ordinal | ABI fixes `S=6` for ThreeVsThree or `S=2` for Defense and gives map types `int64[R]` / `bool[R]`; `R` is symbolic. Current training constructs an environment with `n=1` (`meco_track_training.py:2357-2358`) and `MecoTrainConfig.rollout_steps=32` is configurable (`:116-125,142-151`). | Device map/masks; PPO, environment/campaign and ABI owners | Current runner settings do not establish the shared V3 profile. Freeze one production `N,S,R`, whether rollout rows represent decisions or world/robot slots, bounds and duplicate-owner behavior, generation width/wrap and ordinal reset. Never equate a configurable default with a schema capacity. |
| `action_ticks=T`, macro active/result axes | Current `action_ticks` defaults to 12 but is caller-configurable and only checked positive (`meco_track_training.py:116-125,142-151`). Macro loops over this value; executed actions and tick masks are appended to Python lists and may end early (`:1884-1917`, `:1987-2015`). | Device macro trace; action/macro, simulator and PPO owners | `T=12` is a default, not a frozen maximum. Pin allowed configured values or one profile value; decide `[T]` retention, padding/no-op semantics, macro terminal/truncation timing, event overflow and output-to-rollout-row binding. |
| Raster observation | `meco_input_provenance.py:12-15` says `[B,6,H,W]`; `learning_inputs.py:75-78,120-127` derives H/W from field dimensions and raster cell size, constructs `[1,6,H,W]` using `policy_pose.dtype`, then serializes. `_snapshot_tensors` explicitly rebuilds float32 on requested device (`meco_track_training.py:1085-1089`). | Device; observation/PPO owners | Six channels and current float32 conversion are derivable. Fixed H/W, field/layout profile, raster resolution, freeze/capture point, and `[N,...]` to `[R,...]` gathering remain unresolved. Observation owner must pin geometry and profile; PPO owner must pin the gathered row representation. |
| Global observation | Input provenance fixes `[B,7]` (`meco_input_provenance.py:16-18`); `_snapshot_tensors` creates float32 `[1,7]` (`meco_track_training.py:1085-1089`). Current values originate from localization, load, hub timer and match time (`learning_inputs.py:109-114`). | Device; observation/PPO owners | Float32 width 7 is derivable at the V3 row boundary. Observation owner must freeze the exact ordered semantic names and source-generation binding; PPO owner must define when the input is captured and retained. |
| Recurrent state / reset generation | All five architectures use `RecurrentTrack` with `GRUCell(64,64)` and hidden shape `[B,64]` (`meco_tracks.py:771,826-837`); track inputs bind hidden to all five (`:74-80`). Training records hidden-before/next tensors (`meco_track_training.py:1701-1713,2564-2702`). There is no LSTM cell in these five model definitions. | Device; model/recurrent, observation-reset and PPO owners | Width 64 and “no cell state in current five models” are source-derivable. Freeze model architecture/no-cell invariant, hidden dtype and axes, before/after capture, reset predicate/granularity, reset-generation dtype/range and episode-generation mapping. Existing tracking reset is a scalar path (`meco_track_training.py:103-113`), not a per-row V3 lifecycle contract. |
| C policy source table, model features, ordinary/continuation maps | Candidate K is `{8,16,32}` (`meco_track_training.py:154-155`; `meco_candidates.py:35-40`). Source order is 8 base rows, E sensor-buffer slots, 24 offset rows, and one continuation row (`meco_candidates.py:1291-1313,1418-1420`). The device path reserves one row per track-buffer slot, including currently ineligible slots (`:1221-1240`), so `E=fuel_count<=600` and policy axis `G_policy=33+E<=633` (`tensor_3v3.py:72-73,368-371`; `_tensor_3v3_observation.py:83-98`; `meco_candidates.py:1147-1164`). Raw feature width is 20 and model width 40 (`meco_tracks.py:771-782,379-384`). Current implementation retains the full metadata/model axis; `selection_rank` is int64 and `model_input_mask` is bool `[1,G_policy]`, admitting the ranked K prefix plus continuation (`meco_candidates.py:788-835,872-884`; `meco_tracks.py:1343-1362`). | Device; Plan C candidate and PPO owners | The source proves a bounded policy-axis formula and maximum, not the accepted K-view compact `[K+1,...]` layout. Plan C owner must reconcile/adopt the full `33+E` axis, rank/mask semantics, stable source-row mapping, padding/missingness, digest coverage and frozen timing with the accepted K-view artifact. Do not treat K as total tensor capacity. |
| A legal graph and full support | `LegalPoseGraph` has grid-derived node capacity; `build_legal_pose_graph` uses `capacity=8*nx*ny` and edge shape `[1,N,D]` (`meco_pose_graph.py:34-54,152-180`). Accepted Plan A order digest binds `rows_per_node=8`, degree 10 and coarse width 64 (`meco_plan_a_policy.py:202-211`); the support has dynamic pair count and fields shaped by graph node/support dimensions (`:21-47,225-236`). | Device; Plan A graph/support and PPO owners | Width 9, eight rows/node, degree 10 and 64 coarse regions are source/contract-bounded. Exact frozen profile for planner `nx,ny`, legal-node bound `N_A`, edge envelope, edge representation, region feature payload dtype/width, support padding, and graph-to-policy-row mapping is unresolved. Plan A owner must publish the attested geometry/profile; PPO must bind that exact support. |
| Candidate/support local and source-global identities | ABI fixes command index/identity integer fields `int64`; Plan A local/global aliases and C K-view mapping semantics are pinned by owner contracts. Current training row stores Python candidate ID/validity lists and candidate-set/table object (`meco_track_training.py:2641-2656`) alongside replay evidence (`:2657-2672`). | Device; A/C owners, shared ABI and PPO | Meaning and source maps are partly derivable. C policy extent is `33+E<=633`; ABI owner must resolve whether oracle-only route specs appear in the transition record. Invalid sentinels, identity code registry, source-map tensor extent and digest representation remain open. A/C owners must publish profiles compatible with one ABI registry; PPO must not copy strings or table objects into V3. |
| Route query/result association | ABI pins per-call `Q≤64`, path `P=73`, result and query field dtypes/shapes; source defines `MAX_ROUTE_QUERY_BATCH=64` and chunk map fields/spans (`experimental_route_evaluation.py:22,25-64`). C source creates 8 base + E target + 24 policy-offset + 1 continuation specs, then 18 oracle variants for each of 4 base plus E target slots (`meco_candidates.py:1291-1313,1418-1425,1447-1453`). Therefore total route specs `Q_total=105+19E`; with `E<=600`, `Q_total<=11,505`, at most 180 chunks of 64. `evaluate_device_route_chunks` receives the complete `query_rows` tuple (`meco_candidates.py:620-621,700-709`); legacy route code also chunks at 64 (`:1548-1556`). | Device; route, Plan A/C and PPO owners | Per-call and current source whole-pool bounds are derivable for this implementation. Plan C/route owners must reconcile the accepted route plan with exact `Q_total=105+19E`, 180-chunk maximum, source ordering, oracle-versus-policy distinction, route-query-to-transition association and route cost provenance. Any future producer change requires re-deriving the bound. |
| Sensed-target table and target identity | ABI fixes entry field types and target identity `int64`, but no capacity/generation field schema. ThreeVsThree enforces `fuel_count<=600`; track position/velocity/mask are `[n,6,fuel_count,2]`, `[n,6,fuel_count,2]`, `[n,6,fuel_count]` (`tensor_3v3.py:72-73,368-371`). The perceived-fuel accessor returns focal robot buffer views (`_tensor_3v3_observation.py:83-98`) and Plan C validates aligned `[world,track,2]` position/velocity plus `[world,track]` sensed mask for its one-world path (`meco_candidates.py:1147-1164`). | Device; observation, native simulator, Plan C and PPO owners | Current buffer upper bound 600 and tensor shapes are source-derivable; stable V3 target-table semantics are not. Observation/native owners jointly define whether the table is all `fuel_count` slots or a narrower live/sensed subset, generation width, invalid sentinel, slot reuse increment semantics, and target digest. C/PPO adopt it. |
| Selected action, old log probability, entropy, value | Current training stores action as a one-element tensor without explicit dtype (`meco_track_training.py:2698`), while logprob, entropy and value derive from the sampled distribution/model (`:1701-1713,1817-1838`). `_ppo_update` stacks row values, then scalar reshapes values (`:2068-2185`). | Device; PPO/model owners | The source identifies scalar-per-row use but does not freeze all dtypes, action/logprob shape, entropy reduction, output tail width or precision. PPO owner must publish exact storage shape/dtype and selected-local-slot relationship; executed action remains distinct. |
| Return/advantage and learning masks | `_gae` consumes tensor rewards/dones/values and makes outputs with `zeros_like(rewards)` (`meco_track_training.py:2024-2033`). Current row stores only `done=ended or timeout` and host `execution_failure` bool (`meco_track_training.py:2641-2703`); actor validity is reconstructed from Python row flags (`:2156-2162`). | Device; PPO, simulator and macro owners | ABI's distinct masks and terminal/truncation facts cannot be derived from this conflated current row representation. Pin each predicate, row/tick granularity, return/advantage dtype, actor-vs-critic failure behavior, GAE normalization, bootstrap behavior and padding effects. Simulator owner defines terminal/truncated facts; PPO defines learning masks. |
| Macro action/status/reward/event trace | Current macro action is one scalar host argument; action/tick results are Python lists/scalars, reward is accumulated as a tensor initialized without explicit dtype (`meco_track_training.py:1884-1917,1892-1905,1987-2015`). Row fields stringify/booleanize some outcomes (`:2691-2703`). | Device; action/macro, native simulator, gamepiece and PPO owners | ABI requires per-tick typed outputs but current source has no fixed tensor tail. Pin status/action/event dtypes and code registries, T, capacity, padding, event overflow, selected/executed linkage, reward aggregation and state mutation atomicity. |
| Reward and next/gamepiece state binding | Reward is produced by macro stepping and reshaped to scalar in row; no V3 next-state or gamepiece state digest exists (`meco_track_training.py:1892-1917,2641-2703`). | Device; PPO, simulator/gamepiece and ABI owners | Scalar float32 is compatible with current conversion but exact creation dtype is not explicitly fixed at the macro accumulator. Reward estimand/aggregation and state-binding fields/algorithm/encoding remain unresolved. Simulator/gamepiece owners must define outputs; ABI/PPO pin exact representation and coverage. |
| Physics RNG v2 versus PPO action/permutation RNG | Accepted gamepiece RNG v2 pins its own per-world Philox address contract. Current PPO seeds global torch RNG (`meco_track_training.py:2256`), uses device `torch.randperm` then host `.tolist()` (`:2166-2167`), and saves global CPU/Python/device RNG states (`:2829-2830`). | Device; gamepiece RNG, PPO optimizer, ABI owners | Physics RNG v2 does not define PPO sampling streams. PPO owner must select independent action/permutation generators, seed derivation, device state/counter layout, before/after decision boundary, padding draw policy and checkpoint resume binding; RNG v2 owner confirms noninterference. No fixed PPO RNG state schema is derivable from current global streams. |
| Input/support/transition digests and validation status | Current replay uses CPU tensor serialization/hash utilities and string/dict fingerprints (`meco_track_training.py:202-267,328-431,2641-2672`); no fixed device byte digest/status tensors exist. | Device; PPO and shared ABI owners | Existing V2/current evidence cannot derive V3 digest algorithm, byte order, output shape, status dtype/code registry or update-atomicity predicate. PPO defines field coverage/algorithm and ABI freezes code/schema; no host digest or verdict is allowed. |
| Opaque record bundle and byte length/capacity | Current checkpoints use `torch.save` plus JSON stats and global RNG state (`meco_track_training.py:2815-2832`); no V3 opaque row/bundle writer is present. | Device-produced bytes, host opaque persistence; PPO, artifact and ABI owners | No source bound for row byte capacity, alignment, padding, length/version type, overflow behavior or byte order. Artifact owner must define bundle envelope and compatibility boundary; PPO defines row byte layout and device writer under separate scope; ABI pins exact limits. |
| Update/epoch/sample-order identity | Current update number and epoch loop counters are host integers; permutation uses a device tensor but is converted to a host list (`meco_track_training.py:2164-2167,2612-2832`). | Device dynamic identity; PPO optimizer and ABI owners | Current host counters do not fix V3 integer widths, valid ranges/rollover, row-vs-envelope placement, device sample-order encoding or deterministic replay identity. PPO owner pins all; ABI freezes its field surface. |

## Required Pass 1 decisions and ownership

The findings above do not authorize a schema declaration. Before the PPO
schema-only source shell in `frc_defense/meco_ppo_batch.py` can proceed:

1. **PPO owner:** update the source-preparation plan to this exact ABI hash;
   publish the complete row and batch field table, exact dtypes/tail shapes,
   static-vs-dynamic split, selected-action/logprob rules, update identity,
   independent action/permutation RNG, digest/status scheme, and opaque row
   byte encoding. It must map every field to an owner and use no inferred
   capacity.
2. **Environment/observation owner:** pin one environment profile (`N,S`, field
   geometry, raster H/W and resolution, observation capture point, sensed-track
   table capacity/generation, reset/episode lifecycle). State source lines and
   explain whether fields are fixed configuration or dynamic tensors.
3. **Plan A owner:** pin legal graph profile (`nx,ny`, node/edge/region extents
   and dtypes), support extent and padding, Plan A HIP status-to-row binding,
   and exact shared `[N,S]` to `[R]` command/transition ownership.
4. **Plan C/route owners:** pin total source candidate capacity `G`, raw feature
   width `F` and ordering/missingness, K view behavior for all `{8,16,32}`,
   sensed-target capacity adoption, total route-query envelope or an ABI-approved
   bounded streaming representation, and route-query-to-transition mapping.
5. **Action/macro/native simulator/gamepiece owners:** pin fixed `T`, all per-tick
   action/status/reward/event dtypes, shapes/capacities, neutral padding and
   overflow behavior, terminal/truncated/bootstrap precedence, pre-mutation
   failure behavior, reward aggregation, and next/gamepiece-state binding.
6. **Artifact/manifest owners:** pin static provenance fields, V3 bundle
   versioning and byte envelope/capacity, host opaque-persistence rules, and
   reject-before-parse behavior for unsupported V3 bundles.
7. **Shared ABI owner:** after owner answers are independently Pass 1 reviewed,
   publish a new immutable ABI revision with no symbolic unresolved capacity or
   untyped dynamic field. Each owner then adopts that same artifact hash and
   receives its required exact-head review. A new or changed production field
   returns to ABI Pass 1.

Each answer must include exact source commit/tree and line citations; values or
semantics not proven by source need an explicit owner decision plus a new
versioned profile. The reviewer must reject any profile that silently promotes
a configurable default, observed maximum, current campaign setting, Python
list length, or per-call route cap into a fixed capacity.

## Stop conditions and verification boundary

No source shell, producer, consumer, test, import, compilation, runtime,
checkpoint, simulator, dashboard/replay, GPU lease, or campaign action is
authorized by this follow-up. Do not stop or modify the active campaign. Keep
the source-preparation stop conditions in `e0e3929` in force. This artifact
does not claim V3 implementation, qualification, a Stage 1 pass, or training
completion.

The only permitted checks for this planning artifact are exact-head metadata,
source/blob identity, and static diff formatting. Independent Pass 1 review
must verify this document against source base `61d3731e1531825ee7524065a158ba4765da0a84`
and the exact ABI/owner artifact hashes above. Any changed source base or
contract hash requires a fresh evidence pass.
