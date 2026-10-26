# Pass 1 owner proposal: sensed-target capacity and slot identity lifecycle

**Status:** Planning-only observation/native owner proposal. This artifact proposes
an implementation-ready contract for owner decision and review. It authorizes no
source implementation, imports, tests, compilation, runtime/device/simulator/
dashboard work, checkpoint access, training/evaluation, or campaign change.

**Planning base:** `main` commit
`e37b2f30b6bc51b74dd72bf039e94a174ec51bf2`, tree
`8edb9998d7e3e361e83938a345c65d4fef488e1d`.

**Input reconciliation:** `docs/plans/gpu-five-track-ppo-plan-c-source-view-reconciliation-20261026.md`,
artifact SHA-256 `ad3b9f69fb5ade7ebe0844be2a3b3105ffcebcd7f708969631efc9f8133e1924`.
This proposal supplies only observation/native owner decisions left open by
that amendment. It does not revise the shared ABI, Plan C K/K+1 owner contract,
route contract, or PPO V3 schema.

**Related source/contract inputs:** current source cited below; PPO source
inventory at `docs/plans/gpu-five-track-ppo-v3-schema-source-evidence-followup-20261026.md`
(artifact SHA-256 `fe4f853ebc1332407b8c40410858aa51ade9daf279c4d4baea98cb10694d850f`);
accepted Plan C K-view contract commit
`4cdc30f923c0d948b43c57d452c4ee4b17cf61d9` (SHA-256
`6d9c0e6320acd503c8595fa619e8beeba24af88f36e49337d7325331a899ff0b`); and
Plan C route amendment commit
`8567fd92d4dc9e214e8502f2249bb02203ac8c03` (SHA-256
`ac5847d65de2af5ccb17d6a1bc109dceefeab71c5d707253980570020c1e885a`).

## Source facts

1. ThreeVsThree accepts configurable `fuel_count`, defaults it to 504, and
   bounds it to `96 + 6*preloaded_per_robot <= fuel_count <= 600`
   (`frc_defense/tensor_3v3.py:47,72-80,105`). The observation table allocates
   `track_pos` and `track_vel` as `[n,6,fuel_count,2]`, with `track_mask` and
   `track_age` as `[n,6,fuel_count]` (`tensor_3v3.py:368-378`).
2. `_perceived_fuel(focal)` returns views of one robot's `track_pos`,
   `track_vel`, and `track_mask`; its documented semantics are anonymous sensor
   semantics and it does not consult physical gamepiece positions, activity,
   or ownership (`frc_defense/_tensor_3v3_observation.py:83-98`).
3. Detection association is explicitly anonymous and angular-rank based.
   `_gated_angular_track_association` matches a detection to a nearby prior
   track or allocates a free/expired slot, and resolves assignment collisions
   deterministically (`_tensor_3v3_observation.py:29-80`). The update writes
   position, velocity, age, mask, confidence and extent into the chosen slot
   (`:555-580`). Thus a buffer index is stable only while the current track
   occupant is retained; it is not a physical FUEL identity.
4. Expired tracks can be reused: association considers an unmasked or timed-out
   slot available (`_tensor_3v3_observation.py:68-80`). Track validity can
   also be cleared when age exceeds the perception timeout (`:704-706`). Full
   and masked world resets clear track masks/age and related observation state
   (`tensor_3v3.py:571-580,643-650`). Current source has no perceived-track
   slot-generation tensor or explicit identity-change event.
5. Physical gamepiece generation is different state. `_initialize_fuel`
   maintains `_meco_piece_generation` and `_meco_world_reset_generation`
   (`_tensor_3v3_gamepieces.py:85-94`), but the perceived-fuel accessor does
   not expose these identities. Physical IDs/generations therefore must not be
   copied into a sensed-target identity or used to claim observer-visible
   association.
6. Plan C reserves one policy row per sensor-buffer slot, including
   currently ineligible rows. `_additional_sensed_target_specs` keeps fixed
   buffer-index order and returns the device eligibility mask
   (`frc_defense/meco_candidates.py:1147-1164`; the source inventory locates
   the enclosing producer at `:1221-1240`). The Plan C source-global policy
   ordering is 8 base rows, E sensed-target rows, 24 approach offsets, and one
   continuation row; hence `G_policy=33+E` and sensed slot i maps to policy
   source row `8+i` (`meco_candidates.py:1291-1313,1418-1420`; reconciliation
   amendment §“Reconciled capacity and identity contract”).
7. Route query ordinal, policy source row and compact model-view row are distinct
   coordinates. The accepted reconciliation proposal maps ordinary-view local
   slots to global policy rows; it explicitly keeps route query ordinals
   separate, even where current ordinal and policy row numbers coincide. A
   sensed-target row may participate in a policy route query, and an oracle
   query may refer to a sensed target, but neither coordinate is the sensor
   slot identity.

## Owner proposal (requires explicit adoption)

### Capacity and axes

- Set `E=fuel_count` for each environment instance, with fixed allocated
  per-instance capacity `96 + 6*preloaded_per_robot <= E <= 600`. Treat E as a
  static profile/configuration value attached to the run, not an observed live
  count. For the currently default configuration this means E=504, not a
  universal frozen capacity. The PPO V3 profile must separately choose and pin
  the production E before schema implementation; source currently permits
  multiple configurations.
- Keep the per-observer sensed table shape `[N,6,E]` for target identity and
  `[N,6,E,2]` for position/velocity. Do not collapse robot views: each robot's
  anonymous association buffer is independently keyed.
- Keep Plan C policy-table capacity `G_policy=33+E` (maximum 633) and route
  query-pool capacity `Q_total=105+19E` (maximum 11,505) as separate axes. The
  sensed slot index i contributes to policy row `r=8+i`; this is a row mapping,
  not an alias between `r` and `i`. Neither the K-wide model feature views nor
  the continuation selector extend E.

### Identity and lifecycle

- Introduce a perceived-track generation tensor with shape `[N,6,E]`, dtype
  signed `int64`, resident on the same device as track state. A valid target
  handle is the tuple `(world_reset_generation, observer_robot, slot_index,
  slot_generation)` bound to the decision snapshot. The generation denotes
  continuity of this anonymous local track slot only, not a physical FUEL ID.
- Preserve a slot generation when a detection is accepted as matched to its
  current unexpired track. Increment it when a newly allocated detection
  occupies a previously free or expired slot, including reuse after timeout.
  Increment the world reset generation on a reset and invalidate all target
  handles from that world. Do not reset a slot generation to zero on world
  reset; it remains monotonic for the lifetime of the environment instance.
- On generation overflow, reset-generation overflow, or a native/device update
  that cannot establish the specified transition, fail closed for candidate
  eligibility and emit a typed invalid/overflow status. Never wrap and silently
  reuse a valid identity. A recreated environment starts a new immutable
  environment/run identity, so its counters need not be globally unique across
  process instances.
- Association may be uncertain or merge/split detections; this contract
  intentionally claims track continuity according to the current accepted
  association decision only. It must not infer exact physical-object identity.

### Invalid values, validity and padding

- Make the device bool `track_mask` authoritative for occupancy/eligibility;
  stale coordinates or generation values behind a false mask have no identity
  semantics and must not enter feature, route or PPO decision logic.
- Use `-1` (`int64`) for an absent sensor-slot index and absent policy-source-row
  mapping. Use a distinct typed route-query kind for policy versus oracle-only
  queries; `-1` is not a valid route ordinal. An invalid target handle has
  `valid=false`, `slot_index=-1`, and `policy_source_row=-1`. Do not use slot 0,
  generation 0, or finite zero coordinates as absence.
- K-view padding uses its own false row mask and the Plan C approved row-map
  sentinel; it is not a sensed-target sentinel. Continuation inactivity uses
  the separately approved continuation mask/sentinel, not a synthetic sensor
  identity. ABI owners must confirm all signedness, widths and sentinel codes.

### Snapshot binding and Plan C/route mapping

- Capture one device-resident observation snapshot token before candidate
  construction. Proposed token fields are `(environment_instance_id,
  world_reset_generation, world_index, observer_robot, observation_sequence)`.
  `observation_sequence` is a monotonically increasing int64 per-world/per-
  robot counter advanced for every policy observation, including observations
  with no detections. Existing source does not currently expose this complete
  token; the observation/PPO owners must choose the exact counter increment and
  reset boundary before implementation.
- Each valid sensed candidate at source row `r=8+i` carries the snapshot token
  and target handle `(world_reset_generation, robot, i, slot_generation[i])`.
  Its dynamic row-map identity remains `policy_source_row=r`, independent of
  `ordinary_view_local_slot=j`. The compact K view stores `j -> r` through the
  Plan C row map; selected local action K (continuation) resolves through its
  own continuation map, not through E.
- A policy route query for that candidate carries its own `route_query_ordinal`
  and `policy_source_row=r`, plus the same snapshot token and target handle.
  An oracle-only route query carries an oracle identity and, where its source is
  a sensed slot, the same snapshot-bound target handle; it has no selectable
  policy source row. A route result is consumable only if its query ordinal,
  target handle and snapshot match the candidate/source mapping. Never derive
  slot identity from route ordinal, local view index, or physical-piece ID.
- Before committing a selected action or PPO transition, device predicates
  verify the current world reset generation, observer, slot bounds, slot mask,
  slot generation and snapshot sequence against the captured handle. A stale
  or invalid handle makes that sensed row ineligible and triggers the
  established typed fail-closed/fallback path; it must not be rebound to a new
  occupant of the same numeric slot.
- Static host digests may bind profile, E, field order and immutable candidate
  IDs only. Dynamic slot generation, mask, snapshot token, row-map values,
  route association and selected target must be included in the device-produced
  per-decision identity/digest coverage required by the accepted PPO/route ABI.
  This proposal does not choose the digest algorithm, byte encoding or output
  dtype.

## Explicit owner decisions still required

1. **Observation owner:** approve per-observer `[N,6,E]` identity axes, the
   observation-sequence increment boundary, and the treatment of missed,
   merged and split anonymous detections. Decide whether the generation changes
   exactly on every free/expired-slot allocation or whether the native
   association layer has a more precise event boundary.
2. **Native simulator owner:** identify the authoritative device-side location
   that performs track allocation, generation increment and reset invalidation;
   establish atomic update ordering with track position/mask writes and specify
   typed behavior if generation state is unavailable or overflows. The current
   association source is Torch code, while physical FUEL generation is a
   separate lifecycle and cannot stand in for sensed identity.
3. **Environment/profile owner:** choose the production E (or an explicit set
   of supported E profiles) and bind it into run/source/config identity. The
   current default 504 and source ceiling 600 are not interchangeable schema
   capacities.
4. **Plan C owner:** adopt `r=8+i`, preserve slot-generation identity through
   candidate construction, ordinary K row compaction, continuation handling,
   route attachment and final command resolution; ensure slot eligibility is
   reevaluated against the captured snapshot and current generation.
5. **Route owner:** bind every policy route result to its policy source row,
   target handle and snapshot; define the oracle-only target-handle and
   retention/receipt behavior without making oracle rows selectable.
6. **PPO/shared ABI owners:** choose field names, dtypes, exact shape tails,
   generation and sequence ranges, environment instance identity, sentinel and
   error-code registries, snapshot boundary, digest encoding/coverage, and
   whether the transition retains the full handle or an approved opaque receipt.
   Adopt the same profile hash across observation, Plan C, route, PPO and ABI.

Until these decisions are accepted against a single source/profile/ABI hash,
this remains a proposal and no schema-only shell or producer may implement its
suggested fields. An accepted answer must distinguish the anonymous track
handle from physical gamepiece IDs and from the three policy/route/view row
coordinates.

## Evidence and scope limits

The only checks authorized for this planning artifact are source/blob identity,
line-citation review, exact-head metadata and static diff formatting. It makes
no claim of behavior implementation, test coverage, device parity, simulator
outcome, replay, dashboard currentness, qualification, Stage 1 pass or campaign
readiness. The live campaign, checkpoints and device leases remain out of
scope and must not be touched under this artifact.
