# V4 owner response: observation/native

- Owner role: `observation_native_owner`
- Delegated agent/task identity: `/root/observation_native_owner`
- Response author: delegated technical owner
- Date: 2026-10-26
- Pinned V4 proposal: commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`
- Pinned owner-adoption matrix: commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`
- **Disposition: PENDING**

## Scope decision

This response addresses only the observation/native row: target and snapshot identity, observation-buffer reuse/reset lifecycle, stale-handle and exhaustion behavior, and fixed macro outputs. It does not adopt V4 or authorize implementation.

The proposal defines a stable target handle as `(world_reset_generation, observer_slot, buffer_slot, slot_generation):int64[4]` and a row snapshot token as `(environment_instance_id:uint64[2], world_slot:int64, robot_slot:int64, world_reset_generation:int64, observation_sequence:uint64)`. It requires generation increments on allocation/reuse, including expired-but-still-masked entries, reset-generation increments on reset, no wrap, and fail-closed exhaustion (V4 proposal, lines 274–280). The transition row includes observed target position, velocity, age, confidence, validity, and handle (lines 149–153). The macro trace is fixed at `T=12`, with tick validity, executed action, remask/target-lost/preempted/fallback/execution-valid flags, reward, outcome flags, and status (lines 160 and 265–270).

The owner-adoption matrix explicitly resolves the identity type: target handles represent anonymous observation tracks, not physical gamepiece IDs (matrix, lines 82–83). I accept that distinction as the intended V4 meaning. The remaining identity gap is normative: the consumer contract must state how an anonymous-track handle is validated and when it becomes stale, and must explicitly prohibit treating its `slot_generation` as the physical `piece_generation` used by score events. V4 defines the physical score-event `piece_slot` and `piece_generation` separately (proposal, lines 199–212), but this observation/native response needs an explicit cross-field boundary so consumers cannot conflate the two identities.

## Source evidence and limits

- `frc_defense/tensor_3v3.py:368-379` allocates observer-local track position, velocity, mask, age, confidence, extents, merged-state, and visibility arrays with shape `[world, robot, fuel_count, ...]`. The arrays contain no slot-generation or physical piece identity field.
- `frc_defense/csrc/tensor_perception_hip_kernel.cu:861-872` ages and expires tracks by clearing `track_mask`, while preserving array contents. Lines 911–943 associate detections to nearby sorted active tracks and assign unmatched detections to the next free slot. Lines 959–975 then overwrite that slot with detection-derived state. This is anonymous, observer-specific association and slot reuse; it does not bind a track to a simulated fuel body's index or generation.
- `frc_defense/tensor_3v3.py:571-576` and `:643-649` clear per-observer tracking validity during full and partial resets. They do not advance a reset generation or invalidate externally retained handles because no handles exist in the current source.
- `frc_defense/tensor_3v3.py:791-798` consumes the observation track index as a planner target and validates it using current validity, age, source-position proximity, and action match. That protects this call path heuristically; it does not establish a durable identity across a reset or slot reuse.
- `frc_defense/meco_track_training.py:1883-1909,1915-1932,1962-1969` executes a variable number of loop iterations up to `action_ticks`, collects Python lists of executed actions/ticks, and can stop after terminal/truncated status. It tracks selected-target-lost and action-remasked values, but there is no existing canonical fixed-width 12-tick output record at this boundary. V4 must specify how inactive tail ticks, simultaneous per-tick statuses, and the macro summary relate to these current traces; the listed fixed-width fields alone do not establish that mapping.
- `frc_defense/tensor_perception.py:534-558` dispatches anonymous tracking to the HIP extension when available and returns a boolean dispatch result. The native tracker source confirms the anonymous track semantics above, but not a mapping to physical-piece identity. The matrix's identity-type decision should be normative for consumers: observation track handles are not score-event piece identities.

The above is a source audit only. It does not establish runtime behavior, tracker parity, correctness of the V4 serializer, or implementation feasibility on the active GPU path.

## Blocking decisions before disposition

1. Define the normative consumer and staleness rules for anonymous observation-track handles, including when a handle is rejected after expiry, allocation/reuse, reset, or observer-local reassociation. Preserve the matrix's distinction and explicitly prohibit equating observation `slot_generation` with score-event physical `piece_generation`.
2. Pin lifecycle rules at the tracker boundary: which events count as allocation/reuse, whether an expired-but-still-masked slot is stale immediately, whether a matched re-detection preserves slot generation, how full and partial reset advance `world_reset_generation`, and which counter owns overflow detection. The proposal states the increment/no-wrap requirements but does not bind them to these current source events.
3. Define how `environment_instance_id`, `world_reset_generation`, and `observation_sequence` are initialized and advanced, including the precise snapshot capture boundary relative to perception, reset, and macro tick execution. The current source has no corresponding identity counters.
4. Specify exact fixed macro field population, including tail padding, precedence when multiple flags occur on one tick, and consistency checks against the existing variable-length execution trace and early termination. V4 gives field shapes and status precedence but does not fully define the projection from the current macro callchain.
5. Specify int64/uint64 exhaustion handling as a device-visible status that rejects the affected row before action or optimizer mutation, consistent with V4's fail-closed requirement.

Until these are resolved in a versioned contract or supported by a versioned replacement artifact and exact review, the observation/native row remains **PENDING**. No unpinned interpretation is substituted here.

## Authority statement

This file records only the delegated observation/native technical decision. It is not human maintainer approval, does not imply another owner's adoption, does not publish a shared ABI, and does not authorize implementation.
