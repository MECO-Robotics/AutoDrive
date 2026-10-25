# Pass 1 superseding plan: five-track PPO and selected-row reset residency

**Status:** proposed scope expansion after Pass 2 stopped before edits; no implementation or runtime evidence.
**Planning base:** origin/codex/gpu-five-track-ppo-rollout-candidate-expansion-plan-20261026 at 7ce8dd894e9beff41146543264f44fe420187963.
**Supersedes in full:** docs/plans/gpu-five-track-ppo-rollout-candidate-residency-expansion-20261026.md, SHA-256 835e402398db64fd0ed2b3ac17d24d3e7567ce322c25bccdb7d543656bdb92b4, and its incorporated predecessor docs/plans/gpu-five-track-ppo-batched-residency-expansion-20261026.md at 39c0466cafca4c3fd93848c019bedbca08fb8a3b.
**Requirements:** docs/plans/whole-compute-gpu-only-meco-plan-20261025.md, steps 4–8; docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md, §2, §3, §5 Phase 3; repository AGENTS.md simulator/dashboard validation rules.

## Why this Pass 1 expansion is required

The approved rollout/candidate plan's Pass 2 stopped before edits because its proposed masked curriculum reset cannot be delivered in its authorized file set. Source tracing found two full-batch reset entry points: reset_curriculum_scenario calls architecture_scenarios.reset_scenario, which calls env.reset(seed=..., defer_perception=True) without a row mask; the separate TensorDefenseResetMixin.reset also resets the full environment batch. Although partial-reset helpers exist (TensorThreeVsThreeEnv.reset_done, TensorDefenseResetMixin.reset_done, TensorVectorizedSimulator.reset_done, and FuelPhysics.reset(mask)), their existence alone does not reset the complete selected curriculum scenario or prove nonselected state is unchanged.

The reset scope therefore expands to the scenario adapter, the 3v3 reset coordinator and each row-state owner it invokes, the separate defense reset mixin API, fuel physics, and curriculum episode state/metrics. This is a reset-only expansion. It does not authorize changing step physics, observations, reward definitions, policy support, or candidate enumeration. All five-track PPO batching, terminal/status/candidate residency, exact policy and optimizer semantics, RNG/schema/checkpoint rules, and the prior plan's three-pass sequence remain required without relaxation.

## Objective

Complete the prior all-five device-residency objective while adding a fixed-shape selected-row curriculum reset contract. A device terminal mask must reset every episode-scoped field for selected worlds, preserve every unselected world's state and behavior, and avoid host reads or dynamic row compaction. Reset seed/RNG progression, scenario layout, perception exposure, planner state, gamepiece/fuel state, phase-four profile state, recurrent state, curriculum metrics, and the next device action must remain reproducible under checkpoint/resume.

## Reset call graph and state-owner audit

The implementation must audit and close this full chain; a reset helper that updates only pose or done fields is insufficient.

1. meco_track_training.py receives device terminal/truncation/execution masks, writes the fixed PPO row, and requests selected-row reset without converting the mask to Python or resetting the whole training batch.
2. curriculum_scenarios.py::reset_curriculum_scenario currently mutates shared config, deletes/replaces curriculum_metrics, resets phase-four runtime fields, lays out scenario-specific robots and pieces, installs an action mask and schedule, refreshes perception, and rebuilds metrics. The selected-row version must distinguish static episode configuration from per-world tensors, then update only selected world rows.
3. architecture_scenarios.py::reset_scenario currently invokes env.reset and then changes robot poses, piece catalog/ownership/zones, pickup-cell cache, fuel physics, route/planner caches, tracking/perception buffers and observations. Add a mask-aware curriculum entry point; it must not call whole-batch reset from the training loop. Trace architecture_offense_scenario.py delegation; include it only if the reset adapter actually needs it.
4. tensor_3v3.py::TensorThreeVsThreeEnv.reset_done is the active curriculum environment's partial core reset. Its contract and delegated state owners must cover pose/collision state, action/commitment state, scoring/match clocks, fuel and robot counters, target/search/raster state, planner cadence and outputs, perception RNG/tracks, observation history and cached observations. Existing code keeps perception RNG ticks monotonic across episode resets; preserve that rule unless separately reviewed.
5. tensor_physics.py::TensorVectorizedSimulator.reset_done resets physical parameters, pose/velocity, module state, collision reports, physics time/substep and safety counters using the environment device generator. Verify masked RNG consumption and unchanged unselected rows.
6. _tensor_3v3_gamepieces.py, _tensor_3v3_navigation.py, _tensor_3v3_observation.py and _tensor_3v3_actions.py own additional per-world buffers and behavior. Trace every field they initialize/update; reset selected slices or document why a field is static/derived and safely recomputed. No owner may depend on whole-batch reset to clear stale state.
7. fuel_physics_runtime.py, fuel_physics.py and fuel_physics_sleep.py synchronize planar piece state with 3D dynamics and invalidate/reset sleep/cache/island state. Confirm masked reset keeps graph/kernel caches valid for mixed rows and cannot leak stale selected-row state.
8. curriculum_metrics.py::CurriculumMetrics owns per-world/per-piece lifecycle, batch, reward-delta, route and outcome accumulators. Replace deletion/recreation on the live path with selected-row mask reset/initialization from selected scenario state. Initial staged/external/required flags must match fresh construction exactly.
9. curriculum_scenarios.py also owns phase-four observation history, sensor delay/current frames, opponent/start-profile assignments, runtime metrics, anchors and per-world validity flags. Reset selected-row tensors/history only; immutable protocol assignments and static scenario metadata stay host-side and unchanged.
10. tensor_defense_reset.py::TensorDefenseResetMixin.reset_done is a separate supported vector reset API and currently resets environment-owned row tensors. It delegates piece staging to tensor_defense_gamepieces.py and perception/observation refresh to tensor_defense_observation.py; include and audit those helpers' reset-time state effects. Reconcile this API with the shared fixed-shape reset-mask contract so no host-only/reset alternative remains. TensorDefenseResetMixin.reset remains a full-batch construction/API reset and must not be invoked by device terminal control.

If tracing finds another episode-scoped row state or external helper required to reset an owner, Pass 2 stops before touching that out-of-scope production file and returns for another exact-file Pass 1. No module may be called derived without documenting its recomputation dependency and checking that selected reset cannot modify unselected values.

## Complete authorized production scope

This adds reset work to every production/test/document target in the superseded plan; it does not reduce prior requirements. Prior model/replay, candidate, route, fixed PPO batch, checkpoint and RNG targets remain authorized unchanged and remain mandatory.

| Target | Authorized work | Boundary |
|---|---|---|
| frc_defense/curriculum_scenarios.py | Add device-mask reset API and row-wise initializer for all supported five-track curriculum assignments. Tensorize per-world seed/generation, schedule state, robot/piece placement, action masks, contact geometry/profile state, phase-four validity/metrics and observation-buffer reset. | Scenario/category/route/profile IDs and protocol assignments remain static. No mask-controlled Python branch, scalar extraction, host compaction or deleting shared metrics objects. |
| frc_defense/architecture_scenarios.py | Add mask-aware scenario reset that invokes compatible partial core reset, applies all scenario pose/piece/cache changes using selected-row writes, defers perception consistently, and refreshes only selected observations. | Preserve current full reset/construction and seeded placement behavior. Include architecture_offense_scenario.py only if source tracing proves required. |
| frc_defense/tensor_3v3.py | Complete selected-row reset coordinator for every core and cached per-world state it owns; align reset completion with curriculum reset without terminal host branching. | Preserve full reset API, step cadence and transition behavior. |
| frc_defense/tensor_physics.py | Audit/fix TensorVectorizedSimulator.reset_done for every physical parameter, module state, collision/safety diagnostic and generator rule. | Preserve physics distribution and generator semantics; version changed deterministic semantics. |
| frc_defense/_tensor_3v3_gamepieces.py | Mask-reset piece position/velocity/activity/owner/zone/type, preload counts and every per-world lifecycle/cache field used by scenario reset. | Fixed-shape masks; preserve piece identity/order, staging counts and ownership semantics. |
| frc_defense/_tensor_3v3_navigation.py | Reset selected-row target, raster, reanchor, planner cadence/output, route/intercept/progress/potential and planner-owned row buffers. | Planner configuration, route quality and cadence unchanged. |
| frc_defense/_tensor_3v3_observation.py | Reset selected-row perception tracks, validity/age/confidence/extents, noise/dropout counters, delayed frames, observation histories and cached row outputs. | Preserve observation schema and perception timing/noise semantics. |
| frc_defense/_tensor_3v3_actions.py | Reset selected-row last action, action history, target/commitment/collect/search/intake state and persistent execution state. | Preserve legal-action and macro semantics. |
| frc_defense/fuel_physics_runtime.py | Rebind planar fuel state and call selected-row physics reset once after placements. | No host list of selected worlds, extra physics step or early exposure. |
| frc_defense/fuel_physics.py | Audit every reset state/cache; selected rows reset fully and other rows remain unchanged. | Maintain contact, angular, sleep, velocity and cache invariants. |
| frc_defense/fuel_physics_sleep.py | Reset selected sleep-island bookkeeping without invalidating or rewriting unselected rows. | Sleep policy and solver unchanged. |
| frc_defense/curriculum_metrics.py | Add selected-row reset/initialization for every per-world/per-piece metric tensor from scenario staging. | Reward, lifecycle, outcome and qualification definitions unchanged. |
| frc_defense/tensor_defense_reset.py | Audit/fix separate reset mixin masked API and row state for the same fixed-mask rules; ensure training terminal path never calls full reset. | No second PPO environment or full-reset behavior change. Exclude unnecessary changes and document nonuse. |
| frc_defense/tensor_defense_gamepieces.py | Audit the mask-aware piece initializer invoked by TensorDefenseResetMixin.reset_done; include only selected-row reset state necessary for this contract. | Preserve existing piece initialization and transition semantics. |
| frc_defense/tensor_defense_observation.py | Audit reset-time perception and observation refresh helpers invoked by TensorDefenseResetMixin.reset_done; ensure selected rows only and no terminal-mask host control. | Preserve observation meaning and sensing cadence. |
| frc_defense/meco_track_training.py | After terminal/truncation/status are written to the preassigned row, call masked reset; align hidden-state and commitment resets to the same mask. | Compact final action dispatch remains the only live numeric host payload. |
| Prior plan's tests and schema/model/candidate/route/PPO source targets | Continue unchanged as incorporated requirements; add reset parity and isolation coverage below. | No waiver of A1/A2/A3/C1/C2, D1/D2 matched controls, replay, PPO, RNG, checkpoint or cadence gates. |
| This document | Immutable superseding Pass 1 artifact. | Plan only, no implementation/validation claim. |

No edits are authorized to tensor_defense_step.py, simulator transition/contact/physics equations, observation semantics outside reset-owned state, native HIP/build code, evaluation/qualification/reporting, dashboards, replay generation, policy models, rewards/outcomes, campaign controls or deployment. Reset-only changes to the listed owners are allowed. If complete reset requires a transition owner or another file, stop and submit another exact-file plan.

## Selected-row reset contract

1. The reset mask has fixed shape [world], boolean dtype, and is already on the assigned HIP device. No .item(), .tolist(), .cpu(), nonzero-based host compaction, host indexing by selected values, or Python branch controlled by the mask.
2. Selected rows reproduce the exact declared initial scenario state: robot pose/velocity and randomized drivetrain state, contacts, fuel/gamepiece 2D/3D state, ownership/zones, match/score clock, delays, action/commitment, planner/raster, perception RNG/tracks, observation history, phase-four overlays, metrics, terminal/episode generation and recurrent hidden state.
3. Unselected rows remain unchanged across every owner module, including RNG state affecting future draws, counters, physical values, caches, observations, metrics, profile values and pending planner/sensor work. Prove tensor bitwise equality where applicable; for derived outputs prove semantic equality and document the recompute dependency. Static configuration may be shared only if reset does not mutate it.
4. Define a versioned per-world reset RNG stream/counter so one world's draws do not depend on another world's mask cardinality or scheduling. Action, replay-permutation, simulator, perception and fuel reset streams remain distinct. If this requires an unlisted RNG helper/source file, stop for another Pass 1.
5. Preserve scenario piece identity/order, layout jitter, route/profile overlays, target amounts, allowed actions and deferred-perception ordering. Sensors expose only ordinary simulated observations; never copy ground truth into policy observations.
6. Replace live metric-object deletion/recreation with fixed-layout tensors and selected-row initialization. Preserve staged/external/ferry-credit rules, reward deltas, thresholds, cycle counters and outcomes. Episode-generation identity is device state; host strings are passive reporting only.
7. Write terminal/truncation/execution status to its preassigned PPO row first, then reset the corresponding state, hidden state and commitment; only then create the next decision. Do not overwrite a completed transition.
8. All-true is equivalent to full reset; all-false is a no-op including RNG and caches; mixed masks leave unselected state/RNG unchanged. Checkpoint/resume reproduces the same per-row initialization and next action for the new version.
9. Retain every predecessor invariant: five policy distributions and log-probabilities, Plan A support/digest, C candidate identity/order/route join/fallback/recall, D matched inputs, failure actor eligibility, exact PPO cadence, GAE, critic-only semantics, Adam order/state and checkpoint continuation.

## Reset identity, failure, RNG and checkpoint schema

Reset becomes part of all-five source/protocol identity. Version selected-reset schema, mask semantics, per-world reset RNG algorithm/stream IDs, reset owner manifest, metric layout, phase-four buffer state, episode-generation identity, scenario placement order, action/perception streams and checkpoint compatibility. Retain predecessor status-bit, candidate identity, route-query, fixed-row batch, replay digest/schema, optimizer and checkpoint identities.

Checkpoint must bind exact source hash and include per-world reset generators/counters, action/permutation generators, valid-boundary terminal/episode/reset state, every reset-relevant simulator/fuel/perception/planner/history/metrics tensor, RNG schema, and policy/optimizer state. Resume reproduces the exact next per-row reset, action and permutation or fails closed before mutation. Legacy checkpoints without the explicit new schema cannot silently continue.

Invalid mask/schema/device, incomplete owner manifest, scenario mismatch, impossible partial reset, stale observation, replay/candidate identity mismatch or partial reset failure must make the entire PPO update a no-op. A typed host status may stop orchestration only after the gated no-op. Never fall back to full reset on a terminal event.

## Required tests and documentation

Add to all predecessor plan tests:

| Target | Required evidence |
|---|---|
| tests/test_meco_curriculum_device_reset.py | Every supported scenario/category/route/profile under all-true, all-false, single, alternating and randomized masks; compare selected rows to frozen full-reset reference; bitwise snapshot all unselected owner state; verify staging/metrics/observation, phase-four overlays, RNG isolation and resume continuation. |
| tests/test_meco_architecture_scenario_device_reset.py (new) | Verify selected reset never invokes full-batch env.reset; preserve placement/order and deferred perception; prove writes are mask guarded and other rows/state/observations unchanged. |
| tests/test_tensor_3v3_masked_reset.py (new) | Exercise reset coordinator and all 3v3 state owners, with unselected snapshots across simulator, pieces, fuel, planner, action, observation, perception RNG and caches. |
| tests/test_tensor_defense_masked_reset.py (new or existing) | Verify TensorDefenseResetMixin resets selected state only, preserves other rows/RNG, and training terminal path never invokes full reset. |
| tests/test_tensor_physics_masked_reset.py (new or existing) | Verify physical parameters, module state, contacts/reports, physics counters and per-row RNG under mixed masks; all-false is exact no-op. |
| tests/test_fuel_physics_masked_reset.py (new or existing) | Verify 3D position/velocity/angular/sleep/cache/island state resets only selected worlds and remains synchronized with planar pieces. |
| tests/test_meco_macro_device_status.py and tests/test_meco_five_track_ppo_device.py | Preserve terminal/status and typed batch contracts; test transition-row/reset ordering, reset checkpoint continuation and failure no-op. |
| all predecessor model, replay, candidate, route and PPO tests | Retain policy distribution, support, replay mutation, ordering/recall, route chunks, optimizer and cadence rejection coverage. |

Any test checking simulated robot/FUEL/scoring/collection/shooting/ferrying behavior requires a representative seeded end-to-end simulator evaluation inspecting the resulting fresh replay. Restart and verify dashboard services serve the exact checkout's backend and frontend before generating it, per AGENTS.md. Tensor-only unit tests do not establish simulation behavior. These runtime gates occur only after Pass 3 and WX9100 lease release.

## Three-pass sequence and runtime gates

1. Pass 1: independent reviewer checks exact document/head/hash, predecessor requirements, reset call graph and complete owner list, separate reset APIs, tests/targets, selected/unselected invariants, RNG/checkpoint identity, stop rules and boundaries. No implementation/runtime.
2. Pass 2: only after approval, use a fresh HDD worktree from latest main with the approved plan lineage. Before editing, publish a per-world state owner inventory (field, module, reset action, RNG stream, unselected invariant), fixed schemas/capacities, candidate ordering, route mapping and status bits. Reconcile integrated prerequisites. Commit/push incrementally. No tests/imports/compilation/GPU probes/simulator/dashboard/replay/training while either WX9100 lease is active; no campaign interaction. Any unlisted target means stop before editing it.
3. Pass 3: independent exact-head read-only review of every hunk, reset inventory, five policy/replay paths, row RNG/checkpoint continuity, no-host-sync evidence, unselected preservation, exact cadence and incremental commits. New targets or semantic changes return to Pass 1.
4. Only after Pass 3 approval, explicit user stop and verified release of both WX9100 leases may focused device tests run on attested WX9100s. Simulated-behavior tests also need the seeded end-to-end simulator evaluation, current-checkout dashboard/frontend/backend verification and fresh inspected replay. Report tests, scenario/seed/replay ID, dashboard/source identity and incomplete gates.
5. This plan does not start or alter a campaign. Any later campaign retains existing phase prerequisites and explicit user authorization; update count is not curriculum completion.

## Acceptance and precise stop conditions

Accept only if source/call-graph/device tracing proves: all five tracks use the approved fixed-shape device path; terminal/truncation/status/eligibility/reset/commitment/hidden masks stay device-resident; selected rows reproduce declared initialization; every unselected tensor/RNG/cache/metric stays unchanged; every owner in the inventory is covered; reset RNG/checkpoint/next action match the versioned reference; candidate order/recall, policy distributions/log probabilities, GAE, loss and sequential Adam match frozen references; and no dynamic CPU synchronization or decision remains.

Stop before editing an unapproved target if any row state lacks an owner/invariant; planner/observation/fuel/metric/phase-four state cannot be reset selectively; any unselected value or future RNG stream changes; terminal path invokes full reset; reset APIs cannot share a contract without behavior changes; a device mask controls Python branching/indexing; RNG consumption depends on host-compacted rows; checkpoint cannot reproduce next reset/action; a file/helper outside the table is needed; or any policy/candidate/recall/control/actor eligibility/reward/outcome/physics/observation/GAE/optimizer/cadence semantics would change. Stop on any .item(), .tolist(), .cpu(), host reduction/hash/validation, host selected-index selection, dynamic Python row list/dict, or host terminal/reset branch that can influence a live decision or partial update.

No implementation, tests, imports, compilation, runtime, GPU, simulator, dashboard, replay, training, fitting, evaluation or campaign operation was performed to create this plan. This artifact does not assert that selected-row reset or whole-compute GPU residency currently exists.
