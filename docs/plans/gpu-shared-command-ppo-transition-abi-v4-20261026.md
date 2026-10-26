# Proposed shared GPU command and PPO transition ABI V4

**Status:** planning-only successor proposal for independent Pass 1 review. It
is a new immutable artifact; it does not edit or supersede the accepted V3
contract in place, establish human owner adoption, authorize schema/source
changes, tests, compilation, runtime, simulator, device, dashboard, replay,
checkpoint, campaign, training, or evaluation activity.

**Planning base:** `main` at `5bf923a62f1ed0f07cbd2a6901cbf2cd41632bd4`.
**Successor of:** accepted shared ABI artifact at commit
`1c7e9ef02030fa519325c44aa12d4b07e5d171cc`, SHA-256
`65a4d936deb792b26b61b7569fec992dc8d35ec9da11c02845fcf342fa4b7f30`.
**Inputs:** unified cross-owner contract decision proposal commit
`5bc5a67886c7cebaf21454b5618b9649c21bff76`, SHA-256
`6c875e448ac0a88f71f497881fdc1b3a10f197ac909b5176c4c6a896734cec30`; ThreeVsThree
profile freeze proposal commit `fd027f91ae45ca86606a17cd54b47e7e58a12a07`,
SHA-256 `8bf27df729c62a49a605bc6789de9b12529da4c00c69a8ce84549b208353a998`.
Accepted per-world gamepiece RNG contract proposal commit
`1ca7646672a39f0841db5760f1c2dbf8aa465180`, SHA-256
`953f1c31436d82523b90537ccaaf1df9ebbc7bc912f7cbc07343a09fbf0751a4`.

## Authority and decision status

The active user objective authorizes completing the full MECO learning-tracks
plan with five approaches and the requested scoped planning/implementation/review
workflow. This artifact therefore advances the source-supported initial
ThreeVsThree profile as **task-scoped proposed defaults** so owners can review
one exact contract. This objective is not evidence that an environment,
competition, product, PPO, route, Plan A, Plan C, observation/native, simulator,
artifact, or evaluation owner has accepted those defaults. Every value marked
“proposed” becomes normative only after independent owner adoption of this
exact artifact hash. No schema implementation is authorized by this proposal.

The existing V3 ABI remains immutable and available for historical records.
This document proposes `FiveTrackPpoTransitionV4`,
`FiveTrackPpoBatchV4`, and opaque envelope version 2. V4 readers must reject
unknown versions before parsing; V3 bytes are not converted or reinterpreted.
The existing `FiveTrackDeviceActionCommandV1` field sequence and types are
preserved. No V4 path may use a Python/CPU value-dependent command or transition
decision.

## Proposed task-scoped profile

Profile identifier: `meco-ppo-v3-3v3-f504-n1-s6-r32-t12-v1`.
The identifier is retained from the accepted profile proposal; it does not
mean the profile is already production-frozen.

| Symbol | Proposed value | Basis and limit |
|---|---:|---|
| Worlds `N` | 1 | Current trainer constructs one world (`meco_track_training.py:2357-2358`). |
| Robot slots `S` | 6 | ThreeVsThree layout (`tensor_3v3.py:41,368-378`); Defense is a different profile. |
| Rollout rows `R` | 32 | Current trainer default (`meco_track_training.py:116-125`). |
| Macro ticks `T` | 12 | Current trainer default (`meco_track_training.py:116-125`); each row has a fixed 12-tick trace. |
| Target/fuel slots `E` | 504 | Current ThreeVsThree allocation default (`tensor_3v3.py:47,72-80,368-378`); source ceiling is 600, not this profile value. |
| Plan C model width `K` | 16 | Current default/allowed K set (`meco_track_training.py:139,154-155`); selector width is K+1. |
| Raster | `[6,25,51]` float32 | Field 16.54 m × 8.07 m and 0.3302 m cell (`tensor_3v3.py:64-65`, `learning_inputs.py:12,75-127`). |
| Plan A legal lattice | `A_N=12,096` nodes; `A_G=96,768` ordered support rows | 56×27 grid × 8 headings; 8 ordered descriptors/node (`tensor_adstar.py:254-284`, `meco_pose_graph.py:142-179`, `meco_plan_a_policy.py:359-378`). |
| Plan C source table | `G_C=537` rows; raw 20, serialized 40 float features | `8 + E + 24 + 1`; widths and order in `meco_candidates.py:1291-1313,1417-1420`, `meco_tracks.py:772-782`. |
| Plan C route pool | `Q_C=105+19E=9,681` ordered rows; 152 chunks of at most 64; `P=73` points | Source candidate/route count and accepted route contract. This capacity is **Plan C only**. |
| Plan A graph edges | 10 candidate edges/node | Source creates eight translation and two heading edges (`meco_pose_graph.py` graph builder). Exact fixed capacity is `A_N×10=120,960`; valid-edge mask controls use. |
| Plan A ancestor-jump levels | 14 | Source loop grows powers of two until covering 12,096 nodes (`meco_plan_a_policy.py` support construction); proposed field is `[14,A_N]`. |
| GRU hidden | 64 float32 values | Current model (`meco_tracks.py:771,826-837`). |

The active user objective is the authorization basis for using these source
defaults as the proposal’s starting profile. They remain proposed values, not
an assertion of a separate human product decision. Every run binds one track
code for its lifetime: `1=A1, 2=A2, 3=A3, 4=C1, 5=C2`; zero is invalid.

## Independent Plan A and Plan C capacities

The ABI carries two separate policy families. Their index spaces and storage
are not interchangeable.

* **Plan A:** the static legal-pose lattice is `[A_N=12,096]`, with 10
  candidate directed edges per node and `[A_G=96,768]` ordered node/behavior
  support rows. The dynamic PPO record stores its full support-valid mask,
  per-node search costs/parents/traversal order, 14-level ancestor table, and
  policy utility output. Its graph/search evidence is not included in `Q_C`.
* **Plan C:** the immutable full source-policy axis is `[G_C=537]`; K-wide
  ordinary views map through explicit `ordinary_row_map[16]`, and the separate
  continuation map is `[1]` and points to `continue_index`. The route sink is
  `[Q_C=9,681]` and retains all policy and oracle query/result/path rows,
  including inactive rows. At most 152 contiguous calls of Q≤64 feed that sink
  by direct device scatter. There is no second full-pool route temporary.

Never use the C route count to cap, truncate, or summarize Plan A's 96,768-row
support/search evidence. Conversely, A lattice nodes are not C route queries.
The common command includes one selected A or C identity, while each per-row
transition retains the corresponding track-family evidence under the fixed
track tag.

## Canonical V4 payload encoding

The proposed canonical output is one device-produced opaque bundle per ordered
32-row batch. Its 96-byte header is little-endian:

| Header field | Type/width | Value |
|---|---:|---|
| `magic` | byte[8] | ASCII `MECO4V4\0` |
| `envelope_version` | uint16 | 2 |
| `schema_version` | uint16 | 4 |
| `header_bytes` | uint32 | 96 |
| `payload_bytes` | uint64 | Exact payload size for this profile and row count; never exceeds the calculated maximum below. |
| `row_count` | uint16 | 32 |
| `track_code` | uint8 | One of the fixed five codes above. |
| `reserved0` | uint8 | Zero. |
| `profile_digest` | byte[32] | SHA-256 of the exact profile/field-layout declaration. |
| `payload_digest` | byte[32] | Device-produced SHA-256 of the exact payload bytes. |
| `reserved1` | byte[4] | Zero. |

Payload is a packed field-order byte stream: no implicit alignment, no
structure padding, no variable-length columns, and no trailing bytes. Integers
are little-endian; bool is exactly one byte 0 or 1; float32/float64 values are
little-endian IEEE-754, finite when valid, with signed zero preserved. Mask-false
float values are zero. Array axes and row/query/event order below are
canonical. The host can copy the bundle unchanged; it cannot decode, hash,
compare, select, aggregate, or gate on dynamic payload bytes.

The following bundle-level static identity fields precede row 0:
`source_digest:byte[32]`,
`config_digest:byte[32]`, `checkpoint_digest:byte[32]`, followed by
`rng_root_seed:uint64`, `rng_segment_id:uint64`, `world_uid:uint64`,
`rng_protocol_code:uint16`, and
`route_cost_component_presence_mask:uint8` (bit 0=bump cost present,
bit 1=learned cost present). Their bytes total 123. `rng_protocol_code=2`
binds `meco-gamepiece-philox-v2`; draw-site IDs and key-derivation code are
covered by the source digest. The accepted RNG proposal requires segment and
world identities but does not source-bound their storage widths; V4 proposes
uint64 for each and rejects out-of-range values rather than wrapping. Per-row
episode generation and physics tick ordinal complete the accepted
counter-address context.
Immutable geometry and descriptor tables are identified by source/config and
profile digests and are reconstructed from that exact pinned source/profile;
they are not dynamic transition data or a sidecar. A profile mismatch rejects
the entire bundle before action or optimizer mutation.

## Exact dynamic row field inventory

Every column below is repeated for exactly `R=32` rows. The final column is the
packed bytes **per transition**. Fields inside a group are concatenated in the
listed order. Static source tables and the bundle-level 123-byte identity
prefix are excluded from this per-row subtotal and accounted separately.

| Field group and exact fields | Type and shape per row | Bytes/row |
|---|---|---:|
| Identity/schema: `schema_version:uint16`, `track_code:uint8`; `rollout_slot`, `rollout_generation`, `decision_ordinal`, `world_slot`, `robot_slot`, `episode_generation`, `world_reset_generation`, `observation_sequence`: `int64`; `environment_instance_id:uint64[2]`; masks `row_valid`, `actor_valid`, `critic_valid`, `command_valid`, `terminal`, `truncated`, `transition_finished`, `bootstrap_continuation`, `reset_mask`, `recurrent_available`, `route_active`: `bool` | scalars; 8×int64; uint64[2]; 11×bool | 94 |
| Observation: `observation_raster`, `observation_globals` in channel order `(x,y,sin_heading,cos_heading,load,signed_hub_timer,match_remaining)` | float32[6,25,51], float32[7] | 30,628 |
| Recurrent state: `recurrent_hidden_before`, `recurrent_hidden_after` | float32[64] each | 512 |
| Plan C values: `candidate_raw`, `candidate_missing`, `candidate_model_input`, `candidate_valid`, `candidate_identity`, `ordinary_row_map`, `continuation_row_map`, `candidate_model_input_mask`, `candidate_selection_rank`, `candidate_policy_eligible` | float32[537,20]; bool[537,20]; float32[537,40]; bool[537]; int64[537]; int64[16]; int64[1]; bool[537]; int64[537]; bool[537] | 149,959 |
| Observed target table for this row's observer: `target_xy`, `target_velocity`, `target_age`, `target_confidence`, `target_valid`, `target_handle=(reset_generation,observer_slot,buffer_slot,slot_generation)` | float32[504,2] each for xy/velocity; float32[504] each for age/confidence; bool[504]; int64[504,4] | 28,728 |
| Plan A dynamic support/search: `support_valid`; `distance_m`, `route_cost`, `parent_edge_m`; `parent_node`, `traversal_order`; `ancestor_jumps`; `node_utility`; `behavior_logits` | bool[96,768]; float64[12,096] each ×3; int64[12,096] each ×2; int64[14,12,096]; float32[12,096]; float32[5] | 1,983,764 |
| Plan C full route sink, each of Q=9,681 ordered rows: `query_kind:uint8`, `policy_source_row:int64`, `oracle_identity_code:int64`, `route_variant:int64`, `target_handle:int64[4]`, `goal_xy:float32[2]`, `query_active:bool`, `status_code:int64`, `feasible:bool`, `path_xy:float32[73,2]`, `path_length:int64`, `distance_m`, `endpoint_error_m`, `weighted_cost_grid_edges`, `bump_cost_grid_edges`, `learned_cost_grid_edges`: float64 each, `candidate_index:int64`, `path_point_mask:bool[73]` | 788 bytes/query × 9,681 | 7,628,628 |
| Common selected command, exact V1 fields: `schema_version:uint16`, `track_code:uint8`, `decision_valid:bool`, `policy_local_index`, `source_global_row`, `candidate_identity_code`, `action_id`: int64; `goal_xy:float32[2]`, `goal_heading:float32`, `target_identity:int64`, `route_override:bool`, `fallback:bool`, `route_status:uint8`, `route_cost:float32`, `risk_cost:float32`, `legal_node_id`, `behavior_id`, `descriptor_id:int64`, `selected_U:float32`, `selected_P:float32` | scalar fields as named | 99 |
| PPO values: `reward`, `old_log_probability`, `entropy`, `value`, `old_value`, `return`, `advantage` | float32 each | 28 |
| Action/permutation RNG: for each stream `key:uint32[2]`, `counter_before:uint32[4]`, `counter_after:uint32[4]` | 2 streams; 40 bytes/stream | 80 |
| Provenance/replay: `input_digest:uint8[32]`, `support_digest:uint8[32]`, `transition_digest:uint8[32]`, `validation_status:uint16`, `physics_tick_ordinal:uint64`, `update_id:uint64`, `epoch_id:uint32`, `sample_order_id:uint32` | fixed shapes as named | 122 |
| Fixed macro trace: `tick_valid:bool[12]`, `executed_action:int64[12]`, `remasked`, `target_lost`, `preempted`, `fallback`, `execution_valid`: bool[12] each; `macro_reward:float32[12]`; `gamepiece_outcome_flags:uint16[12]`; `macro_tick_status:uint16[12]` | fixed shapes as named | 264 |

The exact per-row byte sum is 9,822,906. For the complete batch it is
`9,822,906 × 32 = 314,332,992` bytes.

### Score-event table and maximum payload

The source constant `MAX_SCORE_EVENTS_PER_EPISODE=32,768` at
`meco_simulated_score_ledger.py:20` remains a separate episode recorder guard;
it is not the PPO batch capacity. The tighter profile bound is one selected
held piece per robot per physics tick (`gamepiece_actions.py:415-429,440-477`),
six robot slots, and at most 12 ticks per macro. That gives at most
`6 × 12 = 72` score events per transition and `32 × 72 = 2,304` records per
batch. A macro exits on done/truncated before the caller resets the episode
(`meco_track_training.py:1915,1992-1996`), so the row index and episode
generation unambiguously bind events when a batch spans resets.

The current qualifying GPU path rejects the CPU score recorder
(`fuel_physics_runtime.py:76-79`, setup in `meco_track_training.py:3922-3927`),
and the current serialization materializes CPU lists
(`gamepiece_actions.py:31-69`). V4 therefore requires device-native typed event
capture and a staged, uncommitted event/transition buffer. The CPU recorder
and JSON stream are not used by qualifying V4 training.

Each fixed 59-byte columnar event record is:

| Field | Type | Bytes |
|---|---:|---:|
| `valid` | bool | 1 |
| `rollout_row_index` | uint8 | 1 |
| `episode_event_sequence` | uint16 | 2 |
| `tick` | int64 | 8 |
| `match_elapsed_seconds` | float64 | 8 |
| `world_slot` | int64 | 8 |
| `alliance` | uint8 | 1 |
| `robot_slot` | int64 | 8 |
| `piece_slot` | int64 | 8 |
| `piece_generation` | int64 | 8 |
| `event_type` | uint16 | 2 |
| `score_units` | uint16 | 2 |
| `source_site_code` | uint16 | 2 |
| **Total** |  | **59** |

`event_type=1` means the current ledger's `fuel_score_awarded`; `score_units`
is the recorded unit. Proposed source-site codes are 1 for the exact source
literal `gamepiece_actions.update_fuel_actions.vectorized_score_mutation` and
2 for `gamepiece_actions.update_fuel_actions.robot_loop_score_mutation`, the
two call sites at `gamepiece_actions.py:429,476`. `rollout_row_index` plus that
row's episode generation replaces the current dynamic episode-ID string;
sequence is episode-local and uint16 covers the source recorder's range
0..32,767. The V4 numeric event identity is `(track_code, world_slot,
episode_generation, episode_event_sequence)` and does not reproduce the
historical Python event-ID string. Any future event type/site needs a new ABI
revision. No human-readable dynamic strings enter V4.

The event table maximum is `2,304 × 59 = 135,936` bytes. The proposed
maximum payload is therefore:

```text
314,332,992  transition row columns
    135,936  score-event table
        123  bundle-level source/config/checkpoint digests and RNG identity
------------
314,469,051  maximum payload bytes
         96  envelope header bytes
------------
314,469,147  maximum complete bundle bytes (299.9012 MiB)
```

This is an exact packed-layout recommendation, not a source measured runtime
memory result. Before an implementation plan can adopt it, reviewers must
verify every V4 field and byte subtotal against this table. Producers reserve
the entire fixed capacity before mutation. Any overflow, invalid score event,
stale snapshot, malformed route chunk, unsupported code, nonfinite valid
float, or digest mismatch sets a typed failure status and makes the complete
decision/update a no-op; there is no truncation, host-side repair, or partial
artifact. If event capture cannot be preflighted before simulator mutation,
the native/simulator owner must stage the decision in an uncommitted buffer and
discard it on failure.

## Codes, lifecycle, and deterministic replay recommendations

These choices make the candidate ABI concrete; each remains an owner-adoption
item, not an existing accepted fact.

* Keep route status values unchanged: `0=INVALID`, `1=AT_GOAL`, `2=NO_PATH`,
  `3=FEASIBLE`. `query_kind:uint8` is `0=inactive`, `1=policy`, `2=oracle`.
  `policy_source_row=-1` for oracle rows. Query ordinal is the row's array index;
  `candidate_index` retains the canonical global ordinal.
* `validation_status:uint16`: `0=VALID`, `1=UNSUPPORTED_VERSION`,
  `2=PROFILE_MISMATCH`, `3=SHAPE_MISMATCH`, `4=DEVICE_MISMATCH`,
  `5=NONFINITE`, `6=SNAPSHOT_MISMATCH`, `7=MAP_MISMATCH`,
  `8=ROUTE_INCONSISTENT`, `9=COMMAND_MISMATCH`, `10=TARGET_STALE`,
  `11=EMPTY_SUPPORT`, `12=OVERFLOW`, `13=RNG_MISMATCH`,
  `14=DIGEST_MISMATCH`. `macro_tick_status:uint16` uses
  `0=INACTIVE`, `1=EXECUTED`, `2=REMASKED`, `3=TARGET_LOST`,
  `4=PREEMPTED`, `5=FALLBACK`, `6=EXECUTION_FAILED`, `7=TERMINAL`,
  `8=TRUNCATED`; if concurrent statuses occur, `gamepiece_outcome_flags`
  carries the independent outcome bits and status precedence is
  `EXECUTION_FAILED > TARGET_LOST > PREEMPTED > REMASKED > FALLBACK > EXECUTED`.
* `gamepiece_outcome_flags:uint16[T]` bits: 0 pickup, 1 accepted score,
  2 ferry release, 3 ferry catch, 4 respawn. This compact status summary does
  not replace the complete score-event table. Unknown bits fail closed.
* A stable target handle is `(world_reset_generation, observer_slot,
  buffer_slot, slot_generation):int64[4]`; the row snapshot token is
  `(environment_instance_id:uint64[2], world_slot:int64, robot_slot:int64,
  world_reset_generation:int64, observation_sequence:uint64)`. Increment
  slot generation on allocation/reuse including expired-but-still-masked
  entries; increment reset generation on reset; never wrap. Exhaustion fails
  closed.
* `actor_valid`, `critic_valid`, and `slot_valid` semantics remain separate.
  For this row layout `row_valid` is slot validity. An invalid/padded row is
  excluded from GAE, replay, RNG draws, optimizer state/steps, and diagnostics.
  True terminal and execution failure do not bootstrap; truncation bootstraps
  only with a valid next observation. Exact product estimand remains with the
  curriculum/simulator owners.
* PPO action and permutation RNG streams use Philox4x32-10. Each stream has a
  64-bit key and 128-bit counter; draw order is stable and counter advances by
  consumed 32-bit words. Seed derivation is SHA-256 of domain
  `meco-ppo-v4-rng-v1`, root seed uint64 little-endian, track code and stream
  code; first eight digest bytes are the little-endian key. Counter overflow
  fails closed. Physics RNG remains distinct and follows its separately
  adopted source contract.
* Digests use SHA-256 of canonical field-order packed bytes with the header's
  profile digest, row masks, and code registries included. Digest columns are
  device-produced; host copies only opaque bytes.
* Preserve selected action, executed action, and exact old stochastic log
  probability as separate facts. Never recompute old log probability from a
  native-remasked action. Preserve Plan C K-wide ordinary maps and separate
  K+1 selector continuation; preserve Plan A node/descriptor ordering and
  empty-support behavior (abort/reset and discard incomplete rollout rows).

## Owner adoption gates before any schema plan

Every owner must adopt this exact artifact SHA or publish a versioned
replacement. “Accept” means adopting the proposed values as that owner's
contract, not just agreeing that this is a useful plan.

| Owner | Required explicit decision |
|---|---|
| Environment/profile | Accept/reject proposed profile `N1/S6/R32/T12/E504/K16`, field/raster/grid identity, and one-track-per-run binding. |
| Plan A | Accept/reject A-only `A_N=12,096`, 10 edges/node, `A_G=96,768`, 14 ancestor levels, cost dtypes, full dynamic search evidence, and static-table digest boundary. |
| Plan C | Accept/reject `G_C=537`, raw/model widths 20/40, K=16 ordinary map plus continuation map, full table masks/ranks, and source-row identity. |
| Route | Accept/reject C-only `Q_C=9,681`, 152 direct-scatter chunks, all inactive/policy/oracle path rows retained, five float64 cost columns, and no A-route conflation. |
| Observation/native | Accept/reject target handle/snapshot schema, generation rules, track values, fail-closed behavior, and the fixed 12-tick outputs. |
| PPO/RNG | Accept/reject row masks, lifecycle/bootstrap semantics, exact selected-logprob linkage, PPO stream algorithm/derivation/counters, and digest coverage. |
| Gamepiece/score | Accept/reject 59-byte typed score event, source-site registry, `R×S×T=2,304` profile capacity, device-native capture, and pre-mutation overflow staging. |
| Artifact/manifest | Accept/reject 96-byte envelope V2 header, exact byte maximum, static source/profile digest binding, and reject-before-parse behavior. |
| Evaluation/curriculum | Accept/reject reward/terminal estimand and any recall metric separately; this ABI does not assert a recall result. |

After those decisions, the shared-ABI owner must publish the adopted exact
field table and hash as an immutable contract. Each owner plan must pin that
same SHA and receive its own Pass 1 review; then a separate schema-only plan
may be written. Producer, consumer, PPO update, RNG/checkpoint writer, parser,
dashboard display reader, tests, simulator evaluation and campaign operation
all need their own scoped planning/review gates. The currently live diagnostic
campaign remains ineligible for qualification and is not modified by this
proposal.

## Reviewer checklist and evidence limits

Independent Pass 1 review should verify:

1. This is a new versioned successor and the accepted V3 artifact remains
   unchanged.
2. Proposed defaults are clearly task-scoped recommendations, not human owner
   acceptance or production configuration.
3. Plan A's `A_N/A_G/edge/jump` capacities and Plan C's `G_C/Q_C/K` capacities
   are distinct; no C route bound is applied to A support/search.
4. The V4 field table covers every required transition group, preserves the
   existing command fields and route status semantics, and includes full C
   policy/oracle route results and paths.
5. The 32,768 score-event recorder bound is kept distinct from the 2,304
   profile batch capacity; the `R×S×T` source bound, row indexing,
   event-sequence width, and fail-closed overflow rule are coherent.
6. The payload arithmetic is reproducible from the exact field table and the
   event table; no uncounted padding, dynamic strings, sidecar, or host decode
   is required.
7. Each unresolved product/owner assumption has an explicit accept/reject
   gate; none is represented as an existing source fact.
8. This proposal does not authorize code, tests, runtime/device access,
   simulator/dashboard/replay, checkpoint or campaign operations.

No tests or runtime actions were run. This artifact is not evidence of source
implementation, owner adoption, GPU residency, simulator behavior, dashboard
currentness, replay freshness, qualification, Stage 1 passage, or readiness to
start/restart training.
