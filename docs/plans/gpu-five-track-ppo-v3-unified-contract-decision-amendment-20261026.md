# PPO V3 unified cross-owner contract decision amendment

**Status:** planning-only decision proposal for independent Pass 1 review. It
does not amend the shared ABI, authorize a schema declaration or source work,
or authorize tests, compilation, runtime, device, simulator, dashboard,
checkpoint, campaign, training, or evaluation activity.

**Source base:** `main` `e37b2f30b6bc51b74dd72bf039e94a174ec51bf2`, tree
`8edb9998d7e3e361e83938a345c65d4fef488e1d`.

**Purpose:** join the accepted PPO V3 source inventory and the five accepted
planning proposals into one decision table, give source-grounded defaults
where the current implementation already supplies them, and isolate genuine
profile/product choices from ordinary owner implementation decisions. This is
not a claim that proposal acceptance equals owner adoption.

## Authorities and proposal inputs

The currently accepted shared field facts remain the immutable ABI schema
facts at commit `1c7e9ef02030fa519325c44aa12d4b07e5d171cc`, document
`docs/plans/gpu-shared-command-ppo-transition-abi-20261026.md`, SHA-256
`65a4d936deb792b26b61b7569fec992dc8d35ec9da11c02845fcf342fa4b7f30`.
The later ABI coordination proposal at `6d46846b87a7790396ef5e45868869c66ac173f6`,
SHA-256 `19349ebd40bbe6c0f11e8001bb716524d0273f8f91862f040096883faf3eb338`,
does not supersede those facts until adopted by a new reviewed ABI revision.
The schema-only source-preparation amendment `e0e3929c6b2e8f9109f77c1dc1105d0b367cffd7`,
SHA-256 `248087999f2f1948b419ea67e012533794f4010f24b608b790f39c81c24833f8`,
remains accepted with its implementation boundary intact.

Planning inputs reconciled here:

| Input | Exact artifact SHA-256 | Disposition in this amendment |
|---|---|---|
| PPO V3 source evidence follow-up (`22db4266a0913e4f0dca29a11a371d266524ad0a`) | `fe4f853ebc1332407b8c40410858aa51ade9daf279c4d4baea98cb10694d850f` | Accepted inventory; source findings retained. |
| Plan C/PPO source-view reconciliation (`aabe4103e6ee3bfe6019f0ebbfe2771968c05e0e`) | `ad3b9f69fb5ade7ebe0844be2a3b3105ffcebcd7f708969631efc9f8133e1924` | Accepted planning proposal; its coordinate distinction is adopted as the recommendation below, pending owner adoption. |
| Plan C owner clarification (`b58bb518637c8738c11ea4853424dcb2567996fa`) | `9caa8bdd98890c2ec3e9b3ed239d58f4b7b78b789bc873a6c0bd2db37770dbc0` | Accepted owner proposal; policy row and route ordinal remain distinct. |
| Route owner stream contract (`c3df72023e621968d763d25eae3f94da9386cce2`) | `2ce8c4761f1f6224432993a03bfc176e4f257c4cde2781d9a925c3ed986f1e67` | Accepted owner proposal; preserve complete route evidence and stream 64-row calls. |
| Observation/native target-slot contract (`ecfffd6a70f4de6d0add2a32ef2e8450b4a63e5e`) | `e726ddaad146139f33c3beca4e4e698b9ce3928b2a4fe5701e19bd35202be7fb` | Accepted owner proposal; anonymous target handles stay separate from physical-piece IDs. |
| PPO V3 coordinate field table (`425266d017844d86030bdf576e54599618fc9cb0`) | `7f86f8fe68a0700bee3d1b567d4879d5707d2e54c994064a2ce20dc0399fccba` | Accepted field-table proposal; normalized below as a recommendation. |

## Normative facts versus decisions

Treat these as existing normative facts only where the cited accepted ABI or
owner artifact says they are fixed:

- The command/transition ABI fixes its declared integer/bool types, local and
  global row-map separation, rollout-axis symbol `R`, route-call cap `Q<=64`,
  path bound `P=73`, and route query/result evidence including
  `route_query_active`, `path_point_mask`, `path_xy`, lengths, and statuses.
- The accepted Plan C K/K+1 contract fixes two K-wide model feature views for
  `K in {8,16,32}`, a separate K+1 selector, and continuation as a selector
  choice rather than another ordinary model row.
- The source inventory and subsequent accepted K-prefix repair show that the
  implementation retains a full policy metadata axis and uses `selection_rank`
  plus `model_input_mask`; that source fact does not rewrite the accepted
  compact-view contract.
- All five current models use a 64-wide GRU hidden vector, with no LSTM cell
  (`meco_tracks.py:771,826-837`).

All other values below marked **recommended** become normative only after the
named owners adopt them in a new immutable ABI/profile revision. A source
default is a starting profile value, not a universal capacity or product
requirement.

## Unified recommended contract

| Concern | Recommended V3 decision | Evidence / boundary | Adoption owner and gate |
|---|---|---|---|
| Environment/rollout profile | Use one world, six robot slots, 32 decisions per rollout, and 12 maximum action ticks for the initial ThreeVsThree profile: `N=1`, `S=6`, `R=32`, `T=12`. Encode profile identity in immutable run metadata. | Trainer defaults: `rollout_steps=32`, `action_ticks=12` (`meco_track_training.py:117-125`); runner creates `n=1` (`:2357-2358`); ThreeVsThree has six robots. Defaults are configurable and do not bind Defense (`S=2`) or future profiles. | PPO and environment owners adopt. Only the selected competition/product profile is an external gate; additional profile support is not required to define the initial profile. |
| Raster/global observation | Use float32 observations. Keep raster `[N,6,H,W]`, where `H=ceil(field_width/0.3302m)` and `W=ceil(field_length/0.3302m)` from the immutable field profile; keep global observation `[N,7]` in current order `(x,y,sin(heading),cos(heading),load,signed_hub_timer,match_remaining)`. Freeze field dimensions and channel names in the profile digest. | `learning_inputs.py:12,75-78,109-114,120-127`; `RASTER_CELL_METERS=0.3302`. The source serializes host-side evidence too; V3 dynamic values must remain device resident. | Observation and PPO owners adopt the existing source order. Product gate only for choosing a non-default field geometry or a different observation profile. |
| Recurrent state | Store hidden-before and hidden-after as float32 `[N,64]` for each row/world binding; no cell state. Use the current explicit tracking-loss reset predicate for this initial model profile and bind reset state to episode identity. | `meco_tracks.py:771,826-837`; trainer captures hidden-before/next (`meco_track_training.py:1701-1713,2564-2702`); reset policy is `TRACK_LOSS_RESET_POLICY` at `:86-113`. | Model/PPO/observation owners adopt; no product choice is needed unless curriculum requires a different reset estimand. |
| Sensor capacity E | For the initial profile, recommend `E=504`, the current default allocated `fuel_count`. Treat E as capacity, including inactive slots, and freeze it per run. Permit a later profile only by new profile identity. Source-safe ceilings remain `E<=600`, `G_policy=33+E<=633`, `Q_total=105+19E<=11,505`; at E=504 these are 537 and 9,681 respectively, at most 152 route chunks. | `tensor_3v3.py:47,72-80,368-378`; candidate order and route variants: `meco_candidates.py:1291-1313,1418-1453`. 600 is a source ceiling, not an agreed production choice. | Environment, Plan C, route and PPO owners adopt. Confirming 504 as the intended production profile is the one external/product decision that gates freezing this capacity; source alone cannot establish competition configuration. |
| Policy table and model views | Canonical policy coordinate is `policy_source_row:int64` in `[0,G_policy)`, aliasing ABI `source_global_row`. Preserve the full policy table order: 8 base, E sensor-slot rows, 24 offsets, one continuation. Ordinary local slots map through the K-view row map; selector local slot K maps through an explicit continuation map. Never compact away inactive sensor slots. | Source ordering: `meco_candidates.py:1291-1313,1418-1420`; accepted Plan C K view; current full-axis mask/rank source `meco_candidates.py:788-835,872-884`, `meco_tracks.py:1343-1362`. | Plan C/PPO/ABI owners adopt. This resolves the accepted-plan/current-source shape difference explicitly; it does not claim current implementation already implements compact model tensors. |
| Coordinate identities | Keep `policy_local_index`, `policy_source_row`, `route_query_ordinal`, `oracle_identity`, and transient `chunk_local_row` as distinct typed coordinates. Oracle-only identities have no policy row and are never selectable. Preserve an explicit policy-row-to-query map even when it is identity-valued. Derive continuation from `(track=C, local_index=K)` plus the validity/map; do not serialize a duplicate continuation bool unless later ABI revision requires it. | Coordinate/domain evidence in accepted proposals and source `meco_candidates.py:815-830,1291-1382,1421-1453`; ABI already separates local/global indices. | Plan C, route, PPO, ABI owners adopt. This is an internal contract decision, not a product gate. |
| Route pool and retention | Preserve every policy and oracle query/result row in stable order, including inactive slots; retain all ABI-required fields, paths and masks. Process ascending contiguous spans of at most 64 and scatter results to preassigned global rows. No full `[Q_total,...]` route-path workspace and no host materialization of dynamic query/result tables. At E=504, capacity is 9,681 rows / 152 spans; at source maximum E=600, 11,505 / 180. | ABI fixes per-call Q/P and route fields. Source helper constructs chunks but tuples/materializes and concatenates results (`experimental_route_evaluation.py:21-64,631-703`); proposal defines streaming requirement. | Route/PPO/ABI owners adopt. A different retention policy requires an explicit ABI/evaluation evidence change, not a silent implementation optimization. |
| Route labels and recall | Report `policy_feasible_count`, `oracle_feasible_count`, and `oracle_min_cost`; do not label their ratio as recall until the curriculum/evaluation owner defines candidate equivalence, denominator, inactive-target handling and empty-reference result. Preserve the existing `within_candidate_pool_recall_measured=false` status. | Current count ratio is in `meco_candidates.py:885-917`; it does not establish set recall. | Evaluation/curriculum owner decision gates any recall claim; no gate for the neutral counts. |
| Sensed target identity | A target is an anonymous per-observer track handle, not a physical piece ID: `(run/environment identity, world_reset_generation:int64, observer_robot:int64, slot_index:int64, slot_generation:int64, observation_sequence:int64)`. Keep `[N,6,E]` identity axes. `track_mask` is authoritative. Increment generation when a free/expired slot is allocated, including a still-masked slot whose age is already beyond timeout; do not silently inherit the old handle. Increment reset generation on reset. Use `-1` for absent slot/source-row coordinates and an explicit validity mask. | `tensor_3v3.py:368-378`; `_tensor_3v3_observation.py:29-80,555-580,704-706`; physical generation is separate (`_tensor_3v3_gamepieces.py:85-94`). No current sensed generation/sequence tensor exists. | Observation/native/Plan C owners adopt and implement atomically; this is an internal semantic decision recommended to close the expired-but-matched ambiguity. If existing association semantics cannot distinguish it, native owner must add an explicit allocation event before implementation. |
| Route query tags | Use `uint8` kind codes `0=inactive`, `1=policy`, `2=oracle`; keep `int64` global ordinal and `int64` policy row where kind=policy; oracle identity is a typed tuple and has no selectable row. Keep target handle/snapshot on policy and oracle rows that reference a sensed slot. | Source has distinct policy/oracle construction but no shared query-kind registry (`meco_candidates.py:1291-1382,1421-1453`). Types/codes here are a proposal, not source fact. | Route/PPO/ABI owners adopt. Internal code-registry decision; no external gate. |
| Invalid/padding behavior | Use bool activity/validity masks as authoritative. Invalid source row/slot/query ordinals use `-1` only where the field is signed int64; route status uses the accepted typed status registry. Padded floating values are zero, but never indicate validity. Overflow, stale snapshot, malformed chunk or generation exhaustion fails the complete decision closed; no wraparound or host repair. | Consistent with ABI's masks and source's fixed capacity rows; sentinels and error status must be frozen in ABI. | ABI, observation/native, route and PPO owners adopt; implementation cannot proceed until status-code registry is pinned. |
| Rollout/terminal masks | Keep row-valid, actor-valid, critic-valid, command-valid, terminal, truncated, transition-finished, bootstrap-continuation, route-active and reset masks separate. Recommended GAE bootstrap is allowed only for truncation that preserves a valid next observation; true terminal and execution failure do not bootstrap; invalid padding updates neither actor nor critic. | Current row conflates `done=ended or timeout` and tracks execution failure separately (`meco_track_training.py:2641-2703`); GAE consumes dones (`:2024-2033`). The recommendation closes that ambiguity but needs simulator semantics. | Simulator/action-macro/PPO owners adopt; external gate only if the intended curriculum estimand chooses another terminal/truncation policy. |
| Actions, macro outputs and reward | Keep selected policy action, executed macro action and per-tick action separate. Use action code width/type from the existing command ABI, not Python strings. Recommended fixed trace tail `[T]` for action/status/reward/event slots with bool tick-valid mask; neutral padding is masked. Aggregate row reward as float32 sum of valid per-tick rewards; bind one result to one rollout row. | Existing loop uses configurable T and Python lists, early termination, reward tensor and host row values (`meco_track_training.py:1884-1917,1987-2015,2691-2703`). Exact registries and event capacity are not established by those lists. | Action/macro/native/PPO/ABI owners adopt. Product gate only for reward estimand / event semantics that the learning plan specifies; exact code widths are internal ABI decisions. |
| RNG | Separate action-sampling and minibatch-permutation device generators from physics RNG. Seed them deterministically from run seed, track code, and named stream ID; checkpoint both generator states and update/epoch counters. No host `.tolist()` of dynamic permutations. | Current PPO uses global torch RNG and host-converts `randperm` (`meco_track_training.py:2164-2167,2256,2829-2830`); gamepiece RNG contract covers physics only. | PPO/ABI owners adopt; no product gate beyond an already-selected run seed. |
| Digests/status | Keep static profile/order digest separate from device-produced dynamic transition and route evidence. Device evidence covers snapshot, coordinates/maps, masks, selected action/command, route results and status, recurrent/reset binding, reward/lifecycle masks and update identity. Recommend SHA-256 over canonical little-endian typed bytes with finite-float validation; dynamic bytes stay device-produced and host persists only approved opaque output. | Current hash tools serialize tensors through CPU/Python (`meco_track_training.py:202-267,328-431`); no fixed V3 device byte digest exists. SHA-256 and canonical encoding are engineering recommendations, not source-derived ABI facts. | PPO/route/native/ABI owners adopt exact encoding, status registry and device implementation plan. Product gate only if the artifact consumer requires a different externally mandated digest. |
| Opaque record bundle | Recommend versioned envelope v1 with `schema_version`, `profile_id`, `row_count`, payload byte length, and payload bytes; reject unsupported versions before parsing. Host may store bytes unchanged but may not decode dynamic contents. | Current checkpoint uses `torch.save` and JSON, no V3 opaque writer (`meco_track_training.py:2815-2832`). No safe row-byte capacity is source-derived until all chosen E/R/T/path fields are frozen. | Artifact/manifest/PPO/ABI owners adopt envelope. Artifact storage requirements and long-term compatibility policy are external gates; byte capacity follows the selected profile and cannot be guessed now. |

## Cross-owner decisions and implementation gate

The following defaults are recommendations, not silently accepted source
facts. Each owner should record **accept**, **reject with replacement**, or
**not applicable**, cite its exact source tree/lines, and adopt one exact ABI
and profile SHA. The only unresolved product decisions required before that
freeze are: (1) whether ThreeVsThree `fuel_count=504` is the intended initial
production profile; (2) exact evaluation recall equivalence if a recall metric
is required; (3) the reward/terminal estimand if it differs from the existing
curriculum. Remaining choices (field/code widths, sentinels, stream IDs,
digest encoding, event capacity, and envelope layout) are engineering/owner
decisions and should be resolved by the named owners using the defaults above,
not left open-ended.

Before any source shell is proposed:

1. Plan C, route, observation/native, PPO, simulator/action-macro, artifact,
   manifest and shared-ABI owners adopt or replace their rows in reviewed
   owner amendments.
2. The shared-ABI owner publishes a new immutable revision with exact dtypes,
   tail shapes, capacities, code registries, masks, error behavior and profile
   identity; no symbolic dynamic capacity or untyped field remains.
3. Every owner plan adopts that exact ABI SHA and profile SHA. Cross-owner Pass
   1 review verifies that E, G, Q, route retention, target lifecycle, and
   transition bindings agree.
4. Only after those reviews may a separate plan request the declarative
   `meco_ppo_batch.py` schema shell. That shell remains schema-only under the
   existing accepted source-preparation boundary; producer, PPO computation,
   checkpoint/RNG writer, tests, and runtime require their own scoped plans and
   reviews.
5. Any future implementation candidate receives independent exact-head Pass
   3. Runtime, GPU, simulator, dashboard/replay and campaign work continue to
   obey the live-campaign release gates in the integration sequence.

## Evidence limits

This amendment makes no claim of source implementation, test results, GPU
residency, simulator success, replay freshness, dashboard currentness,
qualification, Stage 1 passage, or campaign readiness. No source or runtime
action was performed. The full route result/path retention recommendation is
bounded only after profile E is adopted; it does not authorize allocating a
maximum-size monolithic route workspace. A required end-to-end simulator
evaluation and current dashboard/replay check remain mandatory for relevant
behavior changes after live training is manually stopped and device leases are
verified released.
