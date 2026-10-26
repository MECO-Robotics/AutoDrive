# Pass 1: shared GPU command and PPO transition ABI

**Status:** coordination proposal only; no runtime implementation, tests, device access, simulator, dashboard, replay, campaign, or training operation is authorized.

**Purpose:** provide one immutable contract for A1/A2/A3 and C1/C2 policy outputs, native simulator command consumption, `FiveTrackPpoTransitionV3` / `FiveTrackPpoBatchV3`, and opaque replay bytes. This document breaks the circular dependency pins among the current PPO, action/macro, and gamepiece plans. Its exact artifact SHA-256 becomes the one shared ABI reference. Owner plans must refer to this exact contract hash; they must not pin one another's moving candidate heads as the ABI authority.

**Planning base:** `origin/main` `5afe639760a3e5ebef4ee392901dca55aa5e15c8`.

**Normative compute boundary:** `docs/plans/whole-compute-gpu-only-meco-plan-20261025.md`, SHA-256 `0fc594792e233c471d3e81c871e595b09bb13f039ddc7f6996afbda389cbaab0`. Dynamic numeric values, IDs, indices, masks, routes, statuses, rewards, recurrent state, and RNG state remain on the assigned device through the decision and transition. The only host artifact boundary is opaque device-produced bytes; the host may persist or retrieve those bytes unchanged. No compact selected-action exception, row-index handoff, host interpretation, CPU fallback, or Python value-dependent command/transition control is permitted.

## Stable prerequisites and candidate snapshot

These exact accepted contracts are stable inputs. They establish semantics but do not, by themselves, authorize code integration under this ABI plan.

| Input | Exact accepted artifact | Semantics used here |
|---|---|---|
| Plan A strict command | `docs/plans/gpu-resident-plan-a-decision-slice-20261025.md`, branch `codex/plan-a-no-d2h-transition-amend-20261026`, head `789e1c9aa1df5f7f3650392e20142d9061ed7c3c`, SHA-256 `f82b29a574a037e0a4dcde8c85070eae24ae6dfc933675478c0821a00f7944db` | Preserve the accepted ordered `DevicePlanAActionCommandV1` fields and Plan A support/action semantics; this shared contract specifies how those fields bind to the common command and V3 transition. |
| Plan C two-view mapping | `docs/plans/gpu-plan-c-k-capped-model-view-contract-20261026-amendment.md`, branch `codex/plan-c-k-capped-model-view-contract-20261026`, head `fc2113d4f088f3777f316054225cdb254118b02f`, SHA-256 `77f118d08d5b2cbb2e1af3e9e343a6140c9af46ca5511d1d35cf9c1cb5d7699c` | Local ordinary slots `j∈[0,K)` map via `ordinary_row_map[j]=selected_policy[j]` to source-global candidate rows; local continuation slot `K` maps via `continuation_row_map[0]=continue_index`. Preserve K=8/16/32, non-prefix selected rows and the existing continuation identity/action/track/goal/route. |
| Whole-compute authority | Above pinned document | Device computes and validates dynamic values; only opaque device-produced bytes cross the final artifact boundary. |

The following are a **non-normative candidate snapshot**, included only to identify the current review queue and stop the circular pin updates. They are not mutually authoritative prerequisites, and none overrides this document. Their new owners must adopt this ABI hash and receive exact-head independent Pass 3 before any shared source integration.

| Candidate input (not ABI authority) | Exact current candidate | State for coordination |
|---|---|---|
| PPO transition plan | `docs/plans/gpu-five-track-ppo-rollout-candidate-residency-expansion-20261026.md`, branch `codex/gpu-five-track-ppo-rollout-candidate-expansion-plan-20261026`, head `c31971479bcc81426af880cc205bdb77a528cada`, SHA-256 `32dd720d2bc8fcb1de8f223b736ddc47af48569f945135a643a4379b8b9d753a` | Current successor candidate, not an accepted implementation authority; amend to reference this shared ABI hash and the exact schema below, then require fresh exact-head Pass 3. |
| Action/macro plan | `docs/plans/meco-device-action-macro-boundary-20261025.md`, branch `codex/meco-device-action-macro-plan-20261025`, head `f1264e276c175eed9049a4a7c358877305b00232`, SHA-256 `b2c296259b7f594623c99e4713c7313c79cc8450dcffdee898bcaf0f67a5cc3c` | Candidate pending exact-head Pass 3 and shared-ABI adoption; stale candidate pins must not be used as ABI authority. |
| Gamepiece/simulator plan | `docs/plans/gpu-simulator-gamepiece-physics-tranche-20261026.md`, branch `codex/gpu-gamepiece-physics-plan-20261026`, head `b948e86b9b91b2de87b22d9805980c1cb4b6c03c`, SHA-256 `4699870f66551a4a1a8e37e1a47351a51c97b2e1236d1010c57e76eb6fa1be9b` | Candidate pending exact-head Pass 3 and shared-ABI adoption; its command/transition dependency pins are inputs for review only. |

A future revision to any candidate plan changes only that candidate's status and source scope. It does not alter this contract. A future change to this contract requires a new immutable amendment artifact with a new hash, independent Pass 1 and Pass 3, and fresh adoption by all owners; do not edit an accepted contract in place. The candidate triplet is pinned only in this coordination plan. In each owner plan, record this ABI artifact path and exact hash, not the other owners' latest plan heads.

## Canonical device command

The common command is a tagged fixed-shape tensor record named `FiveTrackDeviceActionCommandV1`. All fields have fixed dtype/shape, are produced on the selected device, and are consumed by the native simulator without host extraction. A/C union fields not used by the selected track receive neutral values and are ignored under the device-resident `track_code`; a mismatched tag, row map, identity, device, or schema produces a device-gated no-op before simulator state mutation.

| Field | Type | Binding and meaning |
|---|---|---|
| `schema_version`, `track_code` | `uint16`, `uint8` | Static schema revision and one of A1/A2/A3/C1/C2; fixed configuration for a rollout row. |
| `decision_valid` | `bool` | Device validity gate; false is a no-op, never a host branch. |
| `policy_local_index` | `int64` | Index into the selected policy's output support only. For A it is the Plan A local support row; for C it is ordinary slot `0..K-1` or continuation slot `K`. Never treat it as a source-table index. |
| `source_global_row` | `int64` | Immutable full support/candidate-table row, obtained through the device mapping. For A it equals `global_support_row`; for C ordinary it equals `ordinary_row_map[policy_local_index]` and continuation K equals `continue_index`. It indexes static identity, descriptor, goal, route, and action columns on device. |
| `candidate_identity_code` | `int64` | Stable integer identity from the versioned static candidate dictionary/table; C's existing continuation candidate identity is preserved. Human-readable labels are not dynamically constructed or looked up by the host. |
| `action_id` | `int64` | Existing native action identity selected by the policy, not a new action or stop token. |
| `goal_xy`, `goal_heading` | `float32[2]`, `float32` | Selected goal in established coordinates and radians. |
| `target_identity` | `int64` | Existing target identity/sentinel, preserved through native revalidation and remasking. |
| `route_override`, `fallback` | `bool`, `bool` | Device-resident existing route-override and fallback semantics. |
| `route_status` | `uint8` | Versioned route result code; no dynamic host string. |
| `route_cost`, `risk_cost` | `float32`, `float32` | Existing units and values, bound to the selected row. |
| `legal_node_id`, `behavior_id`, `descriptor_id` | `int64` each | Plan A payload, preserving canonical legal-node/behavior/descriptor support and stable repository IDs. C uses neutral values. |
| `selected_U`, `selected_P` | `float32`, `float32` | Plan A selected utility and nonnegative penalty from the unchanged U/P semantics; these remain on device and bind into the transition. C uses neutral values. |

The Plan A command's accepted `DevicePlanAActionCommandV1` field sequence is retained exactly: `global_support_row:int64`, `local_support_row:int64`, `legal_node_id:int64`, `behavior_id:int64`, `descriptor_id:int64`, `action_id:int64`, `goal_xy:float32[2]`, `goal_heading:float32`, `target_identity:int64`, `route_override:bool`, `fallback:bool`, `decision_valid:bool`. The common command fields `source_global_row` and `policy_local_index` are exact aliases of its global and local support row, respectively; those aliases must be device tensor views/fields, not host conversions. `selected_U`, `selected_P`, route status/cost, and risk cost are associated selected-row command fields and transition values; they do not redefine Plan A U/P or route semantics.

For C1/C2, policy-local slots and source-global rows are distinct coordinate systems. The device command binds both values plus `candidate_identity_code`; no string ID, table row, or selected action index crosses to Python during selection, command construction, macro execution, reward, transition commit, or replay validation. For local continuation K, `source_global_row` is the existing `continue_index` row and the existing continuation identity, action, track, goal, and route are preserved. Do not invent `policy_stop`, change the action, or alias local K to a global row. The ordered selected action/logprob slots remain ordinary `0..K-1`, then continuation K.

## Native command consumption and macro result

The simulator step consumes the command tensor directly. It performs all existing legal-action checks, target/source/XY revalidation, route endpoint checks, native remasking, safety and collision checks on device. A remasked executed action retains both the policy-selected action and native executed action in the same transition; actor eligibility follows the PPO rule below. Invalid command, overflow, stale identity, unsupported backend, or invalid device gates the complete transition to a no-op before partial state mutation.

The fixed configured `action_ticks` loop is controlled by device active masks. Do not break on device terminal/truncation/status from Python. Device-produced macro outputs include per-tick executed action, remask, target loss, native preemption, fallback, execution validity, reward, gamepiece outcome/event tensors, terminal/truncation, next physical state identity, and post-macro state. Each output remains linked to the exact input command and one predetermined rollout slot. No Python scalar action dispatch, local/global index transfer, candidate ID lookup, reward decision, or terminal branch is permitted.

## Canonical PPO transition and batch

`FiveTrackPpoTransitionV3` is the single canonical row for training and evaluation. `FiveTrackPpoBatchV3` is a fixed-capacity ordered set of rows with separate device tensors; it is not a Python dictionary/list of heterogeneous row values. The PPO owner controls implementation details but must preserve the following exact field groups and linkage:

1. **Static provenance and slot identity:** schema/protocol version, track code, source/runtime/config/checkpoint identity, rollout slot, rollout generation, episode generation, decision ordinal, environment/world slot, and static seed/config labels. Static metadata may be read by the host; it cannot encode dynamic rows or outcomes.
2. **Policy inputs and state:** observation/features or their device-resident frozen-input binding, recurrent input and next hidden/cell state, reset mask/generation, and separate action/permutation RNG stream counters/state before and after the decision. Dynamic values and identity digests stay on device.
3. **Support and selection:** support/candidate schema and device-computed ordered-table digest; A local/global support rows, node/behavior/descriptor/action and selected U/P; C full source-global table binding, `ordinary_row_map`, `continuation_row_map`, selected policy-local index, selected source-global row and stable candidate identity code. For C K=16/32, maps may select non-prefix source rows; K continuation maps exactly to source-global `continue_index`.
4. **Distribution and critic binding:** selected action, exact old stochastic log probability for that same selected local output slot, entropy/value outputs, old value, return/advantage inputs or device-computed equivalents, selected support masks, and any distribution/RNG counters required for deterministic replay. Never reconstruct log probability from the executed/remasked action.
5. **Command, route, and execution:** the complete `FiveTrackDeviceActionCommandV1` identity/fields or equivalent lossless typed device view, route status/cost/risk and path/query binding, selected-versus-executed action and target identity, per-tick macro status/results, fallback, remask, target-loss, preemption and failure codes. Stable IDs stay device numeric codes, not dynamic host strings.
6. **Learning and lifecycle masks:** slot-valid, actor-valid, critic-valid, active-tick, terminal, truncated, transition-finished, bootstrap-continuation, reset, and episode-generation masks as distinct device fields. Preserve existing semantics: failed/remasked execution is critic-only when required by the current policy contract; padding is a no-op for GAE, replay, RNG, optimizer moments/steps and diagnostics. A row failure must gate the whole update before PPO mutation.
7. **Outcome and deterministic replay:** reward and device-derived gamepiece/simulator outcomes, next state binding, table/input/transition digest, PPO update/epoch/sample-order identity, and explicit validation status. Device replay validation checks selected identity, exact old logprob, route/command/execution linkage, RNG, reset/recurrent state and provenance before mutation. A mismatch yields a device-gated full update no-op.

No shadow route trace, Plan A record, host-selected action sidecar, Python-computed digest, or second PPO batch/schema may supplement this row. `FiveTrackPpoBatchV3` is shared by the five tracks and retains the existing categorical distributions, sequential PPO/Adam cadence, masks, recurrent behavior, reward, termination/truncation and checkpoint/RNG semantics unless a separately reviewed experiment changes them.

## Artifact and reader boundary

The canonical output is a device-produced opaque record bundle for the same V3 rows. Keep the existing artifact naming/bundle boundary under one explicitly versioned envelope; do not create a shadow sidecar. The bundle identifies schema/static provenance and contains the device-produced row bytes, dynamic digest/status, and deterministic replay material. Device code creates and validates all dynamic bytes. Host storage may copy/write/read the opaque byte buffer unchanged; it cannot decode, stringify, compare, hash, aggregate, select, or gate on a dynamic field.

Version-1 historical artifacts remain read-only. Current decoded-JSON campaign/dashboard consumers must reject unsupported V3 bundle bytes before parsing or metric/qualification use. Any campaign parser or display reader needs its own exact-path Pass 1 plan and Pass 3 review. A dashboard exception, if separately approved, is display-only and cannot feed training, route/action choice, evaluation metrics, replay validation, campaign sealing, qualification, readiness, status, artifact digests, or PPO verdicts. Until then, V3 dashboard replay display is blocked; regular inference behavior is not changed by this contract. Export/ONNX and synthetic export paths that cannot prove the same V3 command contract must fail closed under their own reviewed plan; no export path is authorized here.

## Exact coordination allowlist

This document changes only `docs/plans/gpu-shared-command-ppo-transition-abi-20261026.md`. It authorizes no implementation. After this contract receives Pass 1 approval, each owner must amend its own plan to cite its exact hash, list its owned paths below, and obtain independent exact-head Pass 3 before any shared code is integrated. A newly needed path or altered command/transition field returns to Pass 1.

The coordinated implementation candidate may touch only these exact production paths, and only when the corresponding owner plan explicitly includes them:

- Shared policy/PPO/macro command and row: `frc_defense/meco_ppo_batch.py`, `frc_defense/meco_track_training.py`, `frc_defense/meco_candidates.py`, `frc_defense/experimental_route_evaluation.py`, `frc_defense/meco_tracks.py`, `frc_defense/meco_plan_a_policy.py`, `frc_defense/tensor_3v3.py`, `frc_defense/tensor_defense_step.py`, `frc_defense/_tensor_3v3_actions.py`.
- Device reset/metrics/artifact handoff: `frc_defense/curriculum_scenarios.py`, `frc_defense/curriculum_metrics.py`, `frc_defense/meco_training_manifest.py`, `frc_defense/meco_validation_campaign_runner.py`.
- Gamepiece/physics consumer outputs: `frc_defense/gamepiece_actions.py`, `frc_defense/_tensor_3v3_gamepieces.py`, `frc_defense/tensor_defense_gamepieces.py`, `frc_defense/tensor_defense_reset.py`, `frc_defense/fuel_physics.py`, `frc_defense/fuel_physics_runtime.py`, `frc_defense/fuel_physics_hip.py`, `frc_defense/fuel_physics_sleep.py`, `frc_defense/csrc/fuel_physics_hip.cpp`, `frc_defense/csrc/fuel_physics_hip_kernel.cu`, `frc_defense/csrc/fuel_sleep_hip.cpp`, `frc_defense/csrc/fuel_sleep_hip_kernel.cu`.
- Any further native physics file, dashboard/API/static asset, parser, exporter, or build target is outside this allowlist and requires a separate owner amendment. In particular, there is no authorization to change a dashboard reader here.

The exact focused test allowlist for the integrated ABI candidate is:

- Shared command/macro/transition: `tests/test_meco_device_action_macro.py`, `tests/test_meco_five_track_ppo_device.py`, `tests/test_meco_plan_a_ppo_replay.py`, `tests/test_meco_c1_ppo_replay.py`, `tests/test_meco_macro_device_status.py`, `tests/test_meco_ppo_record_v2.py`.
- Candidate mapping, route and track semantics: `tests/test_meco_device_route_batch.py`, `tests/test_meco_device_candidate_order.py`, `tests/test_tensor_fuel_candidate_rank.py`, `tests/test_meco_tracks.py`.
- Native simulator/gamepiece paths: `tests/test_tensor_3v3.py`, `tests/test_tensor_3v3_physical_constraints.py`, `tests/test_fuel_physics.py`, `tests/test_fuel_physics_graph_device.py`, `tests/test_fuel_physics_hip.py`, `tests/test_fuel_physics_integration.py`, `tests/test_fuel_physics_sleep.py`, `tests/test_tensor_gamepieces_hip.py`.
- Reset/provenance: `tests/test_meco_curriculum_device_reset.py`, `tests/test_meco_training_manifest.py`, `tests/test_meco_validation_campaign_runner.py`.

Do not edit or run tests under this Pass 1 task. Dashboard/export changes and their tests require separate accepted plans; their current behavior cannot be treated as a V3 validation reader.

## Three-pass and deferred runtime gates

1. **Pass 1 contract review:** an independent reviewer checks the exact candidate plan hash, field dtypes/shapes/ownership, Plan A field preservation, C local/global mapping including non-prefix K=16/32 and continuation `continue_index`, selected-action/logprob linkage, no host numeric/identity transfer, opaque output boundary, exact allowlists, and status of every dependency. All three owner plans must be revised to cite the same immutable contract hash and agree field-for-field before their own Pass 1 approval.
2. **Pass 2 implementation:** after shared ABI approval, use fresh isolated worktrees from a single pinned source base. Owners implement only their exact accepted path allowlists, in dependency order: fixed candidate/support maps and command producer; native command consumer and gamepiece outputs; canonical PPO transition/batch writer and validation; approved artifact/status binding. No owner integrates another owner’s code before that exact source head passes independent Pass 3. No tests, imports, compilation, simulator, device probe, campaign operation, dashboard action, or training while the active diagnostic campaign owns the two WX9100s.
3. **Pass 3 review:** independent exact-head read-only review of each implementation candidate and the combined integrated source head. Confirm one command schema, one V3 transition/batch, identical hashes in all owner plans/manifests, all paths/tests within accepted allowlists, full call/data flow, device-only checks, no CPU reads or sidecar, stable A/C semantics, replay determinism, and incremental commits. Any changed field, extra path, or weakened invariant returns to Pass 1.
4. **Hardware and behavior tests:** only after user manually stops the live training campaign and both WX9100 leases are independently verified released may the exact approved tests run on the explicitly assigned WX9100s. Do not use utilization telemetry as evidence of lease release. Record device identity, source head, test commands/results and transfer/synchronization evidence.
5. **End-to-end simulator validation:** after the exact-head Pass 3 and device gates, run representative seeded simulator evaluations for every changed track among A1/A2/A3/C1/C2, exercising command consumption, remasking, macro ticks, gamepiece outcome/status, PPO transition and replay binding. Retain scenario, seed, replay ID, source/checkpoint/device identity and result. A unit suite alone does not establish simulated behavior.
6. **Dashboard replay gate:** before generating/reviewing any replay, restart relevant services from the exact reviewed checkout; verify backend/API health and frontend asset hashes; then create a fresh replay and inspect provenance. Report dashboard currentness. If the separately reviewed display-only reader does not exist, mark the V3 dashboard replay gate blocked rather than decoding bytes in an unapproved consumer.
7. **Campaign/training transition:** this contract starts or modifies no training. Restart/passive training requires all five approaches' independent Stage 1 gates, all prerequisites in the original learning-tracks plan, current exact source/checkpoint/manifest bindings, and explicit training authorization. A smoke test, update count, or device-residency result is not Stage 1 success.

## Stop conditions and nonclaims

Stop before implementation if any owner plan continues to pin another moving candidate as ABI authority; if any owner does not cite this exact contract hash; if Plan A or C mappings differ; if the native simulator needs a host-decoded action; if the PPO row omits command/selected-vs-executed/logprob/mask/reset/recurrent/RNG/provenance linkage; if dynamic bytes are decoded or hashed by the host; if the bundle needs a sidecar; if any file/test outside the allowlist is required; if a consumer cannot fail closed on an unsupported version; or if device, simulation, dashboard, or replay gates cannot be met after manual stop and lease release.

This is a proposed ABI and coordination boundary, not evidence that the code implements it. It does not establish simulator correctness, dashboard compatibility, GPU residency, qualification, or Stage 1 completion. No tests or runtime operations were run to create this plan.
