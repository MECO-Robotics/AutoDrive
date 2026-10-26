# Pass 1 amendment: PPO V3 schema-only source preparation

**Status:** proposed, planning-only amendment. This artifact authorizes no
implementation until its exact head passes independent Pass 3. It does not
authorize numerical operations, imports, tests, compilation, device access,
simulator/dashboard/replay activity, checkpoint access, training, evaluation,
fitting, or campaign changes.

**Planning base:** PPO plan branch head
`6e1e6981fc868f1a956f9c55fad5bd7a0947b726`. The parent PPO plan is
`docs/plans/gpu-five-track-ppo-rollout-candidate-residency-expansion-20261026.md`,
SHA-256
`076c6c062e48b2cb65fbe1c0d80a75218dc32a72bec43d5aeb62d8421262056e`.

**Normative type contract:** shared command and PPO transition ABI at
`docs/plans/gpu-shared-command-ppo-transition-abi-20261026.md`, accepted head
`947fc80ebe046363b64583adcc4c17da186309b5`, SHA-256
`7650bbe8ec48dfa07557992f90b821d1dd125bc786f48e8698358ef5c27b308f`. The
ABI owns all shared names, meanings, exact types, shapes, identities, and
ownership it defines. This amendment may not redefine it or infer missing
schema details from a peer plan.

**Campaign boundary:** preserve the active two-WX9100 campaign, its live
checkout, process, inputs, outputs, checkpoints, and device leases. Training
continues uninterrupted until the user manually stops it. The dirty user file
`frc_defense/csrc/tensor_collision_multi_hip_kernel.hip` is outside scope and
must remain untouched.

## Objective and strict scope

Prepare one non-computing, statically typed shell for the ABI-owned
`FiveTrackPpoTransitionV3` and `FiveTrackPpoBatchV3` names in
`frc_defense/meco_ppo_batch.py`, after a complete field-by-field schema table
has been derived from the exact ABI and source bounds. The shell may declare
fixed record structure and tensor annotations only. It may not create or
mutate tensors, read tensor values, validate rows, serialize records, compute
digests, or implement any part of the training/evaluation pipeline.

This is a schema preparation tranche, not the PPO producer. It does not close
the main PPO plan or authorize edits to training, candidate/route selection,
simulator, reset, RNG, optimizer, checkpoint, manifest, artifact, dashboard,
or test paths. If a complete exact schema cannot be derived without guessing,
stop before editing the source shell and return with the unresolved fields for
a new Pass 1 decision.

## Exact ABI requirements to carry forward

The source shell must preserve the seven field groups from ABI 947fc80, without
omission or reinterpretation:

1. Static schema/protocol/track/source/runtime/config/checkpoint identity and
   fixed rollout, rollout-generation, episode-generation, decision-ordinal,
   environment/world-slot identity.
2. Policy observation or frozen-input binding, recurrent input/output, reset
   mask/generation, and separate action/permutation RNG stream state and
   before/after counters.
3. Support/candidate schema and ordered identity; distinct Plan A local/global
   support rows and C local/source-global rows; ordinary and continuation
   maps; route-query identity/masks/results and static route-cost metadata.
4. Selected action plus its exact old stochastic log probability from the same
   local output slot; entropy/value outputs, value, selected support masks,
   and any declared distribution/RNG replay fields. Never bind probability
   to the remasked executed action.
5. A lossless typed view of the complete `FiveTrackDeviceActionCommandV1`,
   route/execution linkage, target identity, per-tick macro output, fallback,
   remask, target-loss, preemption, and failure status.
6. Separate slot-valid, actor-valid, critic-valid, active-tick, terminal,
   truncated, transition-finished, bootstrap-continuation, reset, and episode
   generation state. Invalid padding is a no-op for GAE, replay, RNG, optimizer
   moments/steps, and diagnostics.
7. Reward and device-derived outcome/next-state binding, ordered table/input/
   transition fingerprint, PPO update/epoch/sample-order identity, and typed
   device validation status. A failed row gates the entire update before
   mutation.

Use the ABI's exact command and envelope shapes where represented:

| ABI field or family | Exact ABI type/shape to preserve |
|---|---|
| `FiveTrackDeviceActionCommandV1.schema_version`, `track_code` | `uint16`, `uint8` |
| command `decision_valid`, `route_override`, `fallback` | `bool` |
| command `policy_local_index`, `source_global_row`, `candidate_identity_code`, `action_id`, `target_identity`, `legal_node_id`, `behavior_id`, `descriptor_id` | `int64` |
| command `goal_xy`, `goal_heading` | `float32[2]`, `float32` |
| command `route_status` | `uint8` |
| command `route_cost`, `risk_cost`, `selected_U`, `selected_P` | `float32` |
| command A/C payload axes | fixed `[N,S,...]`, with a single environment layout; `S=6` for `TensorThreeVsThreeEnv` or `S=2` for `TensorDefenseEnv` |
| `environment_active_mask`, `command_active_mask`, `command_valid` | `bool[N]`, `bool[N,S]`, `bool[N,S]` |
| `rollout_row_world`, `rollout_row_robot_slot`, `rollout_slot_valid`, `rollout_active` | `int64[R]`, `int64[R]`, `bool[R]`, `bool[R]` |
| `macro_tick_active_mask` | `bool[N,S]`; static configured loop length is `action_ticks` |
| route query row/variant/world/robot/generation/active fields | `int64[Q]` for row map, variant, world, robot, generation; `bool[Q]` for active; `goal_xy:float32[Q,2]`; `Q≤64` per call |
| route results | `status_code:int64[Q]`; `feasible:bool[Q]`; `path_xy:float32[Q,73,2]`; `path_length:int64[Q]`; three named distance/cost fields `float64[Q]`; query row map `int64[Q]`; path mask/active `bool[Q,73]` / `bool[Q]`; optional bump/learned costs `float64[Q]` |
| sensed-track table fields | `world`, `observer_robot_slot`, `buffer_index`, and `buffer_generation` are `int64`; `valid:bool`; `observed_xy:float32[2]`; `age_seconds`, `confidence:float32` |

These rows state exact ABI types, not a complete V3 declaration. The command,
route query/result, and sensed-track records are shared ABI inputs whose owners
must supply them through their reviewed contracts; the PPO shell cannot create
competing definitions. The ABI fixes rollout axes symbolically, not the
production values of `N`, `R`, or `action_ticks`. Do not mix the `S=6` and
`S=2` environment layouts in one batch.

## Unresolved V3 schema that blocks a shell until closed

The shared ABI names seven required groups but does not publish a complete
field-by-field `FiveTrackPpoTransitionV3` / `FiveTrackPpoBatchV3` dtype and
static-tail-shape table. The parent PPO plan also leaves several tails as
source-bounded or model-declared. Do not fill those gaps by convention. Before
the schema-only source shell is authored, Pass 2 must read source at its exact
pinned base and create an immutable schema table that gives **every** field:
name, owning contract, dtype, shape including fixed tail capacities, device,
validity/padding semantics, and whether it is immutable static metadata or a
dynamic device tensor.

At minimum, the following currently require source-bounded resolution and are
not authorized for guessed layouts or writers by this amendment:

- actual production `N`, `R`, model/feature widths, recurrent hidden/cell
  widths, support/candidate capacities, route/query aggregation tail layout,
  and `action_ticks`;
- dtype and complete tensor shape for policy observations/features, recurrent
  state, values, entropy, old log probability, returns/advantages, reward,
  RNG state/counters, fixed-width fingerprints/opaque record bytes, and
  simulator/macro result tails;
- exact integer widths, shapes, and enumerated code values for protocol/source
  IDs, decision and generation counters, actor/critic/execution status,
  terminal/truncation/bootstrap/reset reasons, and update/epoch/sample-order
  identity where not explicitly pinned by ABI;
- sensed-track table capacity and table/schema-generation field type; the ABI
  pins selected entry field types but not table capacity or all envelope
  metadata;
- the complete set and fixed dtype/shape of ordered candidate/support
  fingerprints, source-global maps, selected-action distribution data, and
  lossless command/result view fields inside each V3 row;
- exact association from `[N,S,...]` command/macro axes to fixed `[R,...]`
  transition axes for the selected production layout, including range and
  ownership proofs for every padded row.

For each entry, the amendment implementer must cite a line-level source bound
or an immutable accepted ABI field. If exact dimensions, dtypes, generation
semantics, or ownership cannot be proved, mark the row unresolved and stop.
Changing a shared ABI field requires a new immutable ABI amendment, exact-head
Pass 1/Pass 3, and adoption by each owner; this PPO amendment cannot silently
extend the shared contract.

## Exact authorized source path

Only after the schema table above has no unresolved fields, Pass 2 may add the
following single file:

| Path | Authorized content | Not authorized |
|---|---|---|
| `frc_defense/meco_ppo_batch.py` (new) | Declarative, fixed-order definitions for exactly one `FiveTrackPpoTransitionV3` and one `FiveTrackPpoBatchV3`, with tensor-only dynamic fields and exact ABI/source-derived dtypes and shapes; static schema identifiers may be immutable constants. No competing command/route/target schema. | Tensor allocation or operation; validation/replay verdicts; GAE; sampling/permutation; loss/gradient/Adam; serialization/opaque byte writer; checkpoint/RNG logic; Python row dictionaries/lists; value-dependent control; any other file. |

The allowed code shell must contain only fields in the closed schema table and
must preserve its field order, dtype, shape, device, and ownership. It must not
contain field defaults that allocate or inspect tensors, post-init validation,
constructors that normalize values, helper functions, imports with device or
runtime effects, file I/O, training hooks, or compatibility fallbacks. Static
type annotations and inert schema constants only. If exact definitions need
any executable behavior, that behavior is outside this tranche and requires a
new Pass 1 amendment.

No tests, imports, compilation, type-checker execution, or runtime probes are
authorized by this source-preparation amendment. The campaign's WX9100 leases
remain untouched; authorization here is limited to reading source and editing
the one listed file in an isolated worktree after exact-head Pass 3 approves
this amendment.

## Ordered three-pass path

1. **Pass 1 review:** independently verify exact plan head/hash, ABI hash,
   scope, known type table, unresolved-field list, and stop conditions.
2. **Pass 2 schema derivation (read-only first):** select and record one exact
   source base. Read only the source/contracts needed to resolve each pending
   field. Produce the complete immutable field table in the allowed plan
   artifact before any Python source edit. A new file or change to this
   amendment returns to Pass 1 and requires review.
3. **Pass 2 shell (conditional):** only after the complete field table has no
   unresolved entries, edit only `frc_defense/meco_ppo_batch.py` to declare
   those exact records. Commit and push this file alone. Do not implement the
   producer, writers, validation or algorithms. If any field remains
   unresolved, stop before shell authoring and seek a new Pass 1 decision.
4. **Pass 3:** independent exact-head review of the schema table and source
   diff. Verify every declared name/type/shape against ABI and cited source,
   no missing/extra field, no operation/helper/import side effect, and only
   the allowed path. A corrected candidate needs a new exact-head Pass 3.
5. **Next tranche:** after shell acceptance, producer behavior still requires
   a separate Pass 1/Pass 2/Pass 3 for exact producer, replay, GAE, optimizer,
   checkpoint, RNG, and artifact logic/path allowlists. This amendment itself
   does not authorize that work.

## Deferred campaign, runtime and qualification gates

All of these remain closed until the user manually stops training and the
supervisor and owned workers are independently confirmed terminal and both
physical WX9100 leases independently confirmed released:

- imports, tests, type checks that import project code, compilation, GPU probes,
  numerical execution, training/evaluation/fitting, checkpoint/artifact reads,
  simulator behavior tests, dashboard service actions, and replay generation;
- every device-residency/parity/determinism/snapshot/safety/performance test;
- representative seeded end-to-end simulator evaluation of changed behavior,
  inspected fresh simulator record/replay, dashboard restart from the reviewed
  checkout, backend/API and frontend-hash verification, and a fresh dashboard
  replay with source/checkpoint/protocol/seed provenance;
- any training-campaign modification, stop/restart, lease change, or Stage 1
  qualification claim.

No single-device P600 or CPU numeric fallback may satisfy acceptance. An
unbound diagnostic evaluation, update count, source declaration, schema-only
Pass 3, or passing preflight is not a Stage 1 pass. Starting/restarting the
five-track campaign requires the complete parent plan and learning-track
authority gates; this amendment grants no launch authorization. The live run
continues uninterrupted until the user's manual stop.

## Pass 1 review checklist

- Confirm this is one new plan document and records the exact ABI head/hash
  and parent PPO plan head/hash.
- Verify the field/type table directly against the immutable ABI blob; every
  additional claimed shape/dtype has a source-bound citation.
- Confirm unresolved V3 shapes/dtypes/capacities are explicit blockers, with
  no guessed values and a required complete field table before code.
- Confirm any schema shell is a single fixed-order tensor record, only in
  `frc_defense/meco_ppo_batch.py`, and has no allocation, computation,
  validation, writer, algorithm, import/runtime effect, or new path.
- Confirm source integration uses an exact pinned base and requires
  independent exact-head Pass 3 before merge; any moved source or ABI invalidates
  the review.
- Confirm tests, imports, runtime/device/simulator/dashboard/replay,
  checkpoints, and campaign interaction remain prohibited, and runtime gates
  require manual stop plus independently verified release of both WX9100s.
- Confirm the amendment makes no claim of PPO implementation, GPU residency,
  Stage 1 pass, qualification, or training completion.

This artifact is Pass 1 authorization only if and after its exact head receives
independent review. No source code or runtime action was performed to create
it.
