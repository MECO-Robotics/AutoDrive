# PPO V3 ThreeVsThree profile freeze proposal

**Status:** planning proposal for independent Pass 1. This is a versioned
successor proposal; it does not edit or supersede the accepted shared ABI at
`1c7e9ef02030fa519325c44aa12d4b07e5d171cc`. It is not evidence of owner
adoption and authorizes no schema declaration, source edits, tests, compilation,
runtime, device, simulator, dashboard, checkpoint, campaign, training, or
evaluation activity.

**Source base:** `e37b2f30b6bc51b74dd72bf039e94a174ec51bf2` (tree
`8edb9998d7e3e361e83938a345c65d4fef488e1d`). The accepted input plan is
`5bc5a67886c7cebaf21454b5618b9649c21bff76`, SHA-256
`6c875e448ac0a88f71f497881fdc1b3a10f197ac909b5176c4c6a896734cec30`.

## Authority and purpose

The immutable ABI facts cited here remain those in
`docs/plans/gpu-shared-command-ppo-transition-abi-20261026.md` at commit
`1c7e9ef02030fa519325c44aa12d4b07e5d171cc`, SHA-256
`65a4d936deb792b26b61b7569fec992dc8d35ec9da11c02845fcf342fa4b7f30`.
The coordination proposal at `6d46846b87a7790396ef5e45868869c66ac173f6`,
SHA-256 `19349ebd40bbe6c0f11e8001bb716524d0273f8f91862f040096883faf3eb338`,
and the accepted reconciliation input above remain proposals. Neither is
rewritten here. This artifact separates existing immutable facts, bounded
source facts, and choices recommended for a new profile/ABI revision. A
schema-only declaration remains gated on owners adopting one exact ABI and
profile hash.

### Exact planning inputs

| Input | Exact artifact SHA-256 | Use here |
|---|---|---|
| PPO V3 source inventory, commit `22db4266a0913e4f0dca29a11a371d266524ad0a` | `fe4f853ebc1332407b8c40410858aa51ade9daf279c4d4baea98cb10694d850f` | Source findings and unresolved producer/schema boundaries. |
| Plan C/PPO view reconciliation, commit `aabe4103e6ee3bfe6019f0ebbfe2771968c05e0e` | `ad3b9f69fb5ade7ebe0844be2a3b3105ffcebcd7f708969631efc9f8133e1924` | Distinguishes model-local K slots from source policy rows and route ordinals. |
| Plan C owner proposal, commit `b58bb518637c8738c11ea4853424dcb2567996fa` | `9caa8bdd98890c2ec3e9b3ed239d58f4b7b78b789bc873a6c0bd2db37770dbc0` | Preserves separate policy-row and route-query identity. |
| Route owner proposal, commit `c3df72023e621968d763d25eae3f94da9386cce2` | `2ce8c4761f1f6224432993a03bfc176e4f257c4cde2781d9a925c3ed986f1e67` | Full policy and oracle query/result/path retention proposal. |
| Observation/native slot proposal, commit `ecfffd6a70f4de6d0add2a32ef2e8450b4a63e5e` | `e726ddaad146139f33c3beca4e4e698b9ce3928b2a4fe5701e19bd35202be7fb` | Separates stable anonymous target handles from observation snapshots. |
| PPO V3 coordinate field-table proposal, commit `425266d017844d86030bdf576e54599618fc9cb0` | `7f86f8fe68a0700bee3d1b567d4879d5707d2e54c994064a2ce20dc0399fccba` | Candidate field inventory. |

Source citations below refer to the exact `e37b2f30b6bc51b74dd72bf039e94a174ec51bf2` tree; these SHA-256 values make the cited source snapshot independently checkable.

| Source file | SHA-256 |
|---|---|
| `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md` | `718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10` |
| `frc_defense/meco_track_training.py` | `e6e2a5001adeaa6db14a2d8b95fe5a94edfa76c19bdc2a303a86d81a56a97486` |
| `frc_defense/meco_candidates.py` | `5fad3dd1227d0c092928950c03cc6023e1300a387dfe3dd7e36381ee1572ede9` |
| `frc_defense/meco_tracks.py` | `1e8d62aae2a94ae99fa6e25abd586ae46cffb669a92f8f6bd495f7f974d3172c` |
| `frc_defense/tensor_3v3.py` | `bd28f3efc96e564dac9956d8beb4502338c97624fd4f65a0b7a6b55e3c591394` |
| `frc_defense/learning_inputs.py` | `5e782f36ae128f0bad54d4a40c757fc45b55739ca7261d632f22c02557bb606c` |
| `frc_defense/experimental_route_evaluation.py` | `988dd5e5f1cc88ad8ed52315a097f319ce068d9ab45c63d9875b99be8b1f27f3` |
| `frc_defense/meco_pose_graph.py` | `62251b6cac235207fc669293bf792e73f1ecf7ce2586099ddb7616a7f8f31b67` |
| `frc_defense/meco_plan_a_policy.py` | `21d18e39043546390dbbeecced9eafbd5e39941ded0247ce0e144aea3519ec54` |
| `frc_defense/tensor_adstar.py` | `16bd37d8c26e751328bc17098a3ebf13132622ef37268f62baaa9e667609e4c0` |
| `frc_defense/meco_simulated_score_ledger.py` | `71427d251f1639a4a113b610b5c8122fe88d1c2aac9c11eb4861b0f82bc05114` |

## Frozen profile recommendation

The following is a concrete initial profile proposal, not a fact that source
defaults are already production configuration. The user-selected default is
ThreeVsThree `fuel_count=504`; the current training defaults support
`N=1, S=6, R=32, T=12`. The profile ID should be the immutable string
`meco-ppo-v3-3v3-f504-n1-s6-r32-t12-v1`; all five tracks run this same profile,
one track per run, with `track_code` fixed for that run.

| Profile field | Value | Basis and limit |
|---|---:|---|
| `world_capacity N` | 1 | `meco_track_training.py:2357-2358` creates one environment. |
| `robot_slots S` | 6 | `tensor_3v3.py:41,368-378`; profile is ThreeVsThree, not Defense. |
| `rollout_rows R` | 32 | `meco_track_training.py:116-125` default `rollout_steps=32`. |
| `macro_ticks T` | 12 | Same config default `action_ticks=12`. This bounds macro trace ticks; it does not itself resolve variable event count. |
| `fuel/target capacity E` | 504 | `tensor_3v3.py:47,72-80`; source allows up to 600. Freeze 504 only in this proposed profile. |
| `field` | 16.54 m × 8.07 m | `tensor_3v3.py:64-65`. |
| `raster cell` | 0.3302 m | `learning_inputs.py:12`; therefore `H=25`, `W=51`, raster `[N,6,25,51]`. |
| `Plan A grid` | resolution 0.3 m; `nx=56`, `ny=27`, heading bins 8; graph nodes `A_N=12,096` | `tensor_adstar.py:254-284`, `meco_pose_graph.py:142-179`. This gives Plan A support capacity `A_G=8*A_N=96,768`, eight ordered descriptors per node (`meco_plan_a_policy.py:359-378`). |
| `Plan C candidate table` | `G_C=537` rows; raw width 20; model-input width 40 | `G_C=8+E+24+1`; source order at `meco_candidates.py:1291-1313,1417-1420`; raw/model widths at `meco_tracks.py:772-782`. |
| `Plan C K` | 16; selector width 17 | Current config default and allowed set at `meco_track_training.py:139,154-155`; K-wide model view and separate K+1 selector remain governed by accepted Plan C contract. |
| `route query capacity` | `Q_total=105+19E=9,681`; 152 contiguous chunks of at most 64; `P=73` | Source-bound candidate/route count in the accepted reconciliation; immutable ABI fixes per-call `Q<=64`, `P=73`. Keep full query/result/path evidence per route-owner proposal. |
| `GRU hidden` | float32 `[64]` | `meco_tracks.py:769-771,826-837`. |
| Score-event source bound | 32,768 per episode | `meco_simulated_score_ledger.py:16-20,391-415`. This is the existing score recorder limit, not yet an adopted PPO per-batch event capacity. |

No capacity is inferred from inactive rows. Padding remains present and
masked. The 504 target capacity is per observer slot; the observation axes
remain `[N,S,E]` as `tensor_3v3.py:368-378` allocates them.

## Fields whose type/shape is source-grounded

The following keep or refine fields already enumerated by the immutable ABI;
all other dimensions added below are recommendations requiring owner adoption.

| Field family | Proposed declaration | Authority / evidence |
|---|---|---|
| Common command | Preserve `FiveTrackDeviceActionCommandV1` fields and dtypes exactly: `schema_version:uint16`, `track_code:uint8`, validity/override/fallback `bool`, local/source rows and IDs `int64`, goal XY/heading/cost/risk/U/P `float32`, route status `uint8`. | Immutable ABI §31-53. No field order or type changes proposed. |
| Route chunk result | Preserve `status_code:int64[Q]` (`INVALID=0, AT_GOAL=1, NO_PATH=2, FEASIBLE=3`), `feasible:bool[Q]`, `path_xy:float32[Q,73,2]`, `path_length:int64[Q]`, distance/error/weighted and optional component costs `float64[Q]`, query ordinals/maps `int64`, `path_point_mask:bool[Q,73]`, `route_query_active:bool[Q]`. | Immutable ABI §65-90. Chunk cap 64; no attempt to revise old ABI. |
| Observation | `raster:float32[1,6,25,51]`; `globals:float32[1,7]` in current order `(x,y,sin heading,cos heading,load,signed hub timer,match remaining)`. | `learning_inputs.py:12,75-127`; field order in `:109-127`. Existing source serialization crosses to CPU; this proposal's device residency is not implemented by this table. |
| Plan A geometry/support | `node_xy,heading:float32`; `heading_bin,edge,region,lookup,node/behavior/descriptor/action IDs:int64`; graph/support masks `bool`; fixed shapes from profile above. Preserve node-major / descriptor-major ordering and exact action/descriptor mapping `(7,0,1,2,3,4,6,5)` / `(-1,0,1,2,3,-1,-1,-1)`. | `meco_pose_graph.py:35-50,169-179,181-209,272-281`; `meco_plan_a_policy.py:359-378,399-428`. Cost fields that originate as float64 in the shortest-path implementation retain float64 unless that owner proves an existing float32 conversion. |
| Plan C feature table | Raw candidate values `float32[537,20]`, missingness `bool[537,20]`, serialized input `float32[537,40]`, row-valid `bool[537]`, candidate IDs and row maps `int64`; ordinary map `int64[16]`, continuation map `int64[1]`. | Feature width at `meco_tracks.py:772-782`; mapping source and full policy-axis semantics in accepted Plan C proposals and `meco_candidates.py:831-884`. |
| Target and snapshot | Existing sensor values retain position/velocity `float32[...,2]`, age/confidence `float32`, mask `bool`. Proposed identity records use `int64` components and explicit validity; absent signed coordinates use `-1`. | Tensor dtypes at `tensor_3v3.py:368-378`; immutable ABI §92-96. Generation/sequence semantics are not present in source and require observation/native adoption. |
| Recurrent state | hidden-before/after `float32[1,64]`, reset/availability masks `bool`; no cell state. | `meco_tracks.py:826-837`; reset predicate at `meco_track_training.py:84-113`. |
| PPO scalar values | rewards/returns/advantages/logprob/entropy/value `float32`; action IDs/local indices `int64`; learning/eligibility/lifecycle masks `bool`. | Existing model/training sources and immutable ABI §104-114. Exact entropy reduction, eligibility and lifecycle semantics require PPO and simulator-owner adoption. |

## Explicit engineering recommendations, not existing ABI facts

These close ordinary design choices to give the next ABI owner review a
concrete candidate. They become normative only in a new immutable successor
ABI/profile accepted by the named owners.

1. **Identity coordinate domains.** Use distinct `int64` fields for
   `policy_local_index`, `policy_source_row`, `route_query_ordinal`,
   `oracle_identity_code`, and transient `chunk_local_row`; never alias them.
   Use `route_query_kind:uint8` codes `0=inactive,1=policy,2=oracle` and keep
   policy-row mapping `-1` for oracle-only rows. Preserve full `G_C=537` order
   and inactive entries. The 5-track code registry is proposed as
   `1=A1,2=A2,3=A3,4=C1,5=C2`; zero is invalid/padding.
2. **Target lifecycle.** Proposed stable handle is four `int64` values
   `(world_reset_generation,observer_robot,slot_index,slot_generation)`;
   proposed snapshot token is `(environment_instance_id[2]:uint64,world_index,
   observer_robot,world_reset_generation,observation_sequence:uint64)`. The
   handle remains stable across observations; every dynamic candidate and route
   query/result binds to one snapshot token. Increment slot generation on each
   free/expired-slot allocation, including the expired-but-still-masked case;
   increment reset generation at reset; never wrap—generation exhaustion fails
   the transition closed. Observation/native owners must accept these rules.
3. **Route sink.** For every transition retain every one of the 9,681 ordered
   query rows, policy and oracle, including inactive rows. Store query kind,
   global ordinal, source policy row or oracle identity, query variant, world,
   robot slot, target handle, snapshot token, goal XY `float32[2]`, query-active
   bool, then all immutable-ABI route result fields. Direct-scatter each
   64-row chunk into its assigned global ordinal; never concatenate a second
   full-pool temporary. Float32 path payload is 5,653,704 bytes per transition
   (`9,681*73*2*4`) and 180,918,528 bytes (172.5 MiB) for a 32-row batch,
   before masks and other columns. Route/PPO/ABI owners must accept this memory
   profile together; a summaries-only substitute requires a new reviewed
   route-owner/ABI/evaluation decision.
4. **Lifecycle and status.** Preserve route status codes exactly. For new V3
   validation and macro status fields, use `uint16` codes and separate
   `validation_status` from `macro_tick_status`; do not overload route status.
   Recommended validation registry: `0=VALID,1=UNSUPPORTED_VERSION,
   2=PROFILE_MISMATCH,3=SHAPE_MISMATCH,4=DEVICE_MISMATCH,5=NONFINITE,
   6=SNAPSHOT_MISMATCH,7=MAP_MISMATCH,8=ROUTE_INCONSISTENT,
   9=COMMAND_MISMATCH,10=TARGET_STALE,11=EMPTY_SUPPORT,12=OVERFLOW,
   13=RNG_MISMATCH,14=DIGEST_MISMATCH`. Nonzero status gates the whole
   decision/update to no-op before simulator or optimizer mutation. This is a
   proposed registry; PPO, route, native and simulator owners must approve it.
   Plan A empty support follows current manifest semantics: abort/reset that
   episode and discard its incomplete rollout rows, never fabricate an action
   (`meco_track_training.py:2395-2401`).
5. **Terminal/truncation.** Retain separate bools for `terminal`, `truncated`,
   `transition_finished`, `bootstrap_continuation`, `actor_valid`,
   `critic_valid`, `slot_valid`, and each of 12 `tick_valid` entries. Recommend
   no bootstrap on true terminal or execution failure; bootstrap on truncation
   only when the next observation is valid. Current implementation collapses
   ended/timeout into `done` (`meco_track_training.py:2691-2703`) and GAE masks
   that value (`:2031-2040`), so this is explicitly a semantic owner decision.
6. **Events.** Keep the score-ledger source maximum `32768` as an episode-level
   upper bound only; do not claim it proves the complete PPO event payload or
   assign an event capacity to each row. Proposed event columns are typed
   `event_type:uint16`, tick/row/world/robot/piece/generation `int64`, and
   numeric payload `float32`; exact event kinds and a rollout-batch capacity
   remain gamepiece/score/PPO owner gates. Until frozen, schema-only batch
   declaration must omit `gamepiece_event` rather than inventing a capacity.
7. **RNG.** Keep physics RNG separate. Recommend two PPO streams, action and
   minibatch permutation, each with `key:uint32[2]`, counter-before/after
   `uint32[4]`, and stream code `uint8` (`1=action,2=permutation`). Recommend
   Philox4x32-10; derive each key from SHA-256 of the domain string
   `meco-ppo-v3-rng-v1`, root seed `uint64` little-endian, track code and
   stream code; first eight digest bytes are the little-endian 64-bit key.
   Counter advances by actual 32-bit words consumed. This is a deterministic
   proposal, not current source behavior: training currently uses global torch
   RNG and host-converts `randperm` (`meco_track_training.py:2164-2167`). PPO
   and RNG owners must adopt the algorithm, draw ordering, counter rollover,
   and checkpoint binding.
8. **Digest.** Recommend `input_digest:uint8[32]`,
   `transition_digest:uint8[32]`, SHA-256 over canonical field-order,
   little-endian typed bytes, with valid floating fields finite and signed zero
   preserved. Exclude padding payload from the digest but include masks and
   profile/schema identity; mask-false floats are canonical zero. Digests are
   device-produced/validated; host only stores bytes. Current source uses
   CPU/JSON hashing, so implementation and byte encoding require explicit
   PPO/ABI adoption.
9. **Opaque envelope.** Recommend fixed 96-byte little-endian header:
   `magic:uint8[8]` (`MECO3V3\0`), `envelope_version:uint16=1`,
   `schema_version:uint16=3`, `header_bytes:uint32=96`,
   `payload_bytes:uint64`, `row_count:uint16=32`, `track_code:uint8`,
   `reserved:uint8=0`, `profile_digest:uint8[32]`,
   `payload_digest:uint8[32]`, `reserved:uint8[4]=0`; payload follows exactly,
   no padding or sidecar. Host may persist/read the bytes unchanged. The maximum
   payload byte capacity is intentionally unresolved until all selected field
   arrays—including the full route sink and Plan A fields—are included in a
   packed-size calculation and the artifact owner accepts the bound. Overflow
   rejects the complete bundle; no truncation or host-side decoding.

## Remaining adoption gates before schema-only declaration

| Gate | Required owner decision/evidence |
|---|---|
| Profile identity | Environment, PPO and evaluation owners accept ThreeVsThree 504-capacity profile and its field/raster/grid derivations, or publish replacements with new profile ID. |
| Plan A graph/support storage | Plan A/PPO/ABI owners confirm fixed node/support graph shapes, static versus per-row graph capture, and memory implications. Source geometry gives 12,096 nodes and 96,768 support rows for the default map; dynamic environment geometry cannot reuse this profile. |
| Plan C input view | Plan C/PPO owners accept K=16 for this profile, raw feature ordering, 20/40 widths, ordinary-only context pooling, continuation map, and 537-row source-axis binding. |
| Route retention | Route/PPO/ABI/evaluation owners accept 9,681 per-transition rows, full oracle paths, 152 chunk scatters, and the 172.5 MiB rollout path floor. |
| Target lifecycle | Observation/native owners accept target handle/snapshot layouts, generation increment points, stale-snapshot behavior, and no-wrap failure. |
| Status/lifecycle | PPO, simulator, action-macro and route owners accept the proposed V3 status registry, atomic no-op behavior, actor/critic eligibility, terminal/truncation and bootstrap rules. |
| Event schema | Gamepiece/score/PPO owners provide event code registry and a source-backed fixed batch capacity; current 32,768 ledger limit only bounds an episode recorder. |
| RNG/digest | PPO/RNG/ABI owners adopt stream algorithm and draw schedule, exact digest coverage/encoding, device production, and checkpoint representation. |
| Envelope bound | Artifact/PPO owners accept header and calculate the exact max payload bytes from the final field table; until then no fixed byte tensor declaration is authorized. |
| Recall | Nonblocking for this profile unless curriculum explicitly requires a recall claim. Until then retain `within_candidate_pool_recall_measured=false` and neutral feasible counts. |

After these decisions, the shared-ABI owner should publish a new immutable
revision containing only adopted values and a profile hash. Each owner then
pins those exact hashes in its own Pass 1 plan. Only the separately accepted
schema-shell plan can authorize `meco_ppo_batch.py`; producer, PPO update,
RNG/checkpoint writer, tests, runtime and campaign work each require their own
scoped three-pass plan. The live training campaign/device/dashboard gates in
the accepted integration sequence continue to apply.

## Evidence limits

This artifact proposes decisions; it is not owner adoption. It does not claim
source implementation, tests, GPU residency, simulator behavior, replay or
dashboard currentness, qualification, Stage 1 passage, or campaign readiness.
No tests or runtime actions were run. The project `AGENTS.md` requires seeded
end-to-end simulator evaluation and current dashboard/frontend/backend/replay
verification after behavior changes, following manual campaign stop and
independent device-lease release.
