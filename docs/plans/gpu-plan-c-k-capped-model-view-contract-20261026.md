# Pass 1 amendment: Plan C K-capped model view and candidate identity

**Status:** Proposed Pass 1 source-plan amendment; implementation requires exact-head independent Pass 3 acceptance.
**Supersedes:** The model-row/identity coupling in `docs/plans/gpu-resident-route-decision-slice-20261025.md` (source SHA-256 `8c6c67ffb2934fa3a4ce40b54c5cc2f2e2cb4877b5eedac9c31a2936a6879b5a`). All other source-plan requirements remain in force unless this amendment explicitly changes them.
**Planning base:** `origin/main` commit `5afe639760a3e5ebef4ee392901dca55aa5e15c8` at drafting; source tree may advance, so implementation must rebase and reconcile the exact source-plan hash before coding.
**Rejected implementation evidence:** Plan C candidate `8befd2df92a93d9e19c9c721952ab7b645eedebe` was rejected at Pass 3 because rows beyond the policy K prefix reached C1/C2 and C2's mean-pooled candidate context included inactive/excess rows. The repair must also decouple the inference request's static host identity metadata from the compact device model-view width/order. This document does not approve or reuse that candidate.

## Goal and boundary

Make C1/C2 receive exactly a `[1,K,40]` device tensor for `K ∈ {8,16,32}` and compute candidate context from those same K rows only. Preserve enough immutable table identity and a device row map to associate every policy row with its original candidate, without copying or converting dynamic row-map values to host. Keep Plan C controls D1/D2, Plan A, route generation, candidate-source ordering, simulator and training/campaign orchestration outside this amendment.

The model view contains at most K candidates total. When a valid continuation candidate exists, it occupies one view slot and ordinary policy-ranked rows fill the remaining K−1 slots; without a valid continuation, ordinary rows fill up to K slots. This is a policy-view cap, not a change to the larger route/recall evidence pool. The view builder must use the already-defined canonical device rank/order policy, so ineligible high-priority extras cannot consume slots and offset rows retain their specified ordering. No row outside this view may enter C1/C2 inputs, attention, pooling, recurrent context, utility, or action selection. If existing approved curriculum requirements give continuation a different K-cap interpretation, stop and request a new Pass 1 decision rather than silently choosing another width.

## Versioned request and tensor contract

Introduce an explicit C1/C2 candidate-view protocol version (v2); do not infer it from tensor shape or overload the current full-width `candidate_ids` contract.

- **Static host identity:** carry the immutable ordered full candidate-table ID tuple, snapshot ID, candidate-table order/schema digest, and the request's view-protocol version. The tuple identifies the global table of N rows and remains independent of K. Its digest covers static IDs/order/schema and snapshot binding only; it must not claim to digest dynamic device row-map values.
- **Device policy view:** carry `candidate_features: float [1,K,40]`, `candidate_valid_mask: bool [1,K]`, and `candidate_row_map: int64 [1,K]`, all on the selected model device. Each map entry is a global row index into the immutable N-row table, with a documented sentinel only for a padded row when fewer than K entries exist. Features, mask, and map share one local view order. The map is generated from device-side policy ranks, contains no duplicate active global rows, and puts a valid continuation in its reserved slot under the rule above.
- **Separate dimensions:** validation uses `global_candidate_count=N` for static identity and `policy_view_width=K` for model tensors. It must not validate either dimension by comparing `K` to `len(candidate_ids)`. Validate ranks, dtypes, devices, protocol version, and tensor shapes without synchronizing device values to Python. Range, active-row uniqueness, map/mask consistency, rank-prefix semantics, continuation placement, and valid-row source membership must be asserted by device-side predicates or established by construction and device-only guards; no `.item()`, `.tolist()`, `.cpu()`, NumPy conversion, or implicit scalar truth conversion may inspect a dynamic map/mask.
- **Model and broker:** the C1/C2 adapter accepts only v2's K-width tensors, returns utility/uncertainty with width K, and echoes protocol version, static table digest, snapshot, and request identity. Broker request/response validation must bind those static values and expected K input/output shapes. An old v1 full-table request may continue only through its unchanged legacy path; it must not be relabeled v2. A v1/v2 mismatch, malformed shapes, or metadata mismatch fails closed to the already-authorized deterministic fallback.
- **C1/C2 semantics:** C1 scores exactly K feature rows. C2 computes attention, pooling, and recurrent candidate context over exactly the K-view rows; invalid/padded rows are excluded from pooled context with a device mask and numerically safe masked reduction. Valid rows must match the approved K-view reference semantics. No inactive, expanded-offset, or recall-oracle rows outside the view can influence recurrent state. Do not change the registered 20 raw features, missingness encoding, 40-wide input, model weights/checkpoint format, score units, or model architecture beyond required mask-aware context reduction.
- **Identity through selection:** utility, feasibility mask, deterministic/stochastic selection, and selected local index remain device tensors of width K. Gather `selected_global_row = candidate_row_map[selected_local_index]` on device. Use that global row tensor for route/action/target/goal/override evidence and action-macro routing. Candidate ID strings and route records may remain static host metadata, but no dynamic global row or selected candidate identity may be copied to Python before decision-dependent scoring, legality revalidation, route evidence, and action construction have finished. Any later boundary that still requires a host action identifier must be separately authorized by the GPU-resident action-macro plan and may not be smuggled into this amendment.
- **Route/evaluation evidence:** append the selected route and inference provenance to evaluation `route_trace` using the same selected global-row mapping, after dispatch has produced route evidence. Preserve route status, path, cost, source/selection provenance and selected candidate/action linkage. Static candidate-table identity and dynamic device view identity must be distinguishable in provenance; do not serialize a fake host digest of dynamic map contents.

## Compatibility and migration

1. Keep existing `MecoInferenceRequest` v1 validation and consumers working for their existing full-table shapes. Add a distinct v2 type/field set or an explicit discriminated version branch with complete validation; do not weaken v1 checks globally.
2. Limit v2 use to C1/C2 Plan C candidate scoring. D1/D2 and Plan A retain their existing input/request contracts in this tranche.
3. Update the local scorer and production adapter together so neither gets a wider candidate tensor than the v2 request. If any remote broker/worker process uses the current schema, inventory its code and add a versioned, fail-closed compatibility path within the allowlist before enabling v2; do not silently send an unsupported schema.
4. Model checkpoint tensors stay compatible. Add protocol/version metadata only to request/response provenance, not learned parameters. No migration or rewrite of campaign records, checkpoints, manifests, dashboards, or active run state is allowed.
5. Preserve immutable table IDs, candidate order, route/action metadata, tie-breaking, score-cost units, deterministic fallback and the already-approved route selection behavior. The changed output index is local to the K view and must be mapped to the original table row on device before downstream use.

## Invariants

- C1/C2 candidate input and every candidate-dependent intermediate used for context or scoring has width K only; every C2 pooled/recurrent context is calculated from those K rows with invalid slots masked.
- Global table count N and policy width K are separately represented and can differ. Static host IDs always describe N global rows; dynamic device map describes K local rows.
- No dynamic numeric candidate identity, row-map, eligibility, mask, score, or selected index is materialized on CPU before action construction. Device guards and validators cannot force host synchronization.
- Each active map entry resolves to one existing global table row; active rows are unique; all aligned columns use the same local order. Invalid/padded slots cannot score, affect context, win selection, or appear in route trace.
- A valid continuation is included within the K cap and remains dynamically revalidated under existing commitment rules. It cannot be dropped by extra/offset candidates or exceed K. The ordinary rows follow the approved rank ordering after filtering eligibility.
- C2 masked pooling is finite for zero valid rows and has a fail-closed fallback; it cannot leak inactive row embeddings into hidden state.
- Response and route evidence bind the request protocol version, static table identity, snapshot, and selected global row. No fallback/response mismatch can be mistaken for a v2 score.
- This is only the C1/C2 model-view/identity contract. It does not claim the whole application is CPU-free or close any Plan A, D1/D2, simulator, training, campaign, dashboard, or five-track gate.

## Pass 2 implementation allowlist

Only the following paths may change. Stop and request another Pass 1 amendment if a required broker implementation, adapter, or test is outside this list.

- `frc_defense/meco_candidates.py`: construct the exact K-width device view and row map from existing canonical ranking/eligibility/continuation tensors; retain the full static table unchanged.
- `frc_defense/meco_tracks.py`: implement discriminated v2 request/response validation, C1/C2 local and production adapter width handling, row-map-aware selection plumbing, and C2 masked context pooling. Keep v1/other tracks unchanged.
- `frc_defense/meco_track_training.py`: route the v2 device view through C1/C2 scoring and bind selected global-row identity into device action/route evidence and evaluation `route_trace`; do not change training schedule, campaigns, checkpoints, hardware selection or launch orchestration.
- `tests/test_meco_tracks.py`: v1 compatibility and v2 request/broker shape/provenance contracts, C2 masked context semantics, malformed/mismatch fail-closed behavior, and zero-valid-row handling.
- `tests/test_meco_multimodel_prediction.py`: C1/C2 v2 scoring and output-index-to-global-row mapping, preserving D1/D2/A contracts.
- `tests/test_meco_candidates.py` (new focused file; no candidate-generation test file exists at the planning base): K=8/16/32 ordering, K-width, continuation slot, eligibility-before-cap, unique/range-valid map, and extra/offset exclusion.
- `tests/test_meco_track_training_candidate_view.py` (new focused file): route-trace selected identity/provenance mapping and no pre-selection host synchronization guards, using fixtures with no simulator/GPU side effects.

Do not change model checkpoint weights, `MECO_INPUT_NAMES` schemas other than explicit versioned candidate metadata, candidate feature meanings/order, Plan A or D1/D2 behavior, route evaluator/search/safety/native legality, action macros, dashboard/replay code, active campaign, curriculum, or GPU kernels in this tranche.

## Evidence and acceptance criteria for Pass 3

Independent Pass 3 reviews the exact pushed candidate head and must verify all of the following from code and focused test output:

1. The C1/C2 model view is exactly `[1,K,40]` for K=8/16/32 and no wider rows enter C1/C2's attention, pooling, recurrence, utility or selection.
2. Static global IDs/digest and dynamic device row map are separate contracts; production broker validation does not compare K to N and rejects protocol/identity/shape mismatches.
3. Device row-map construction obeys eligible-rank ordering, includes a valid continuation within K, excludes inactive extras/offset/recall rows from context, and maps the selected local index to the correct global row without host numeric transfer.
4. C2 context output is invariant to changes in masked-out candidate feature rows, while active-row reference outputs agree within approved dtype tolerance. Empty/malformed support uses the typed fail-closed fallback.
5. Focused tests prove v1 compatibility, v2 local and production adapter behavior, candidate ID/action identity preservation, route-trace provenance, tie-breaking, and unchanged A/D contracts. Static inspection or grep alone is insufficient for the no-sync claim.
6. Exact source diff is confined to the allowlist above, does not touch campaign/checkpoint/dashboard artifacts, and preserves every unrelated gate in the superseded source plan.

Pass 3 must reject if any CPU `.item()`/`.tolist()`/`.cpu()`/NumPy conversion or implicit scalar check reads dynamic map/mask/score/selection before construction of downstream decision evidence; if inactive rows affect C2 context; or if candidate IDs can become misbound after local K selection.

## Deferred runtime gates and training handoff

- This Pass 1 work performs no tests, imports, device query, runtime, simulator, campaign, checkpoint, dashboard, replay, or GPU operation.
- Pass 2 may implement and run device-independent contract tests while the diagnostic campaign runs, provided those commands do not initialize a GPU context, touch campaign/checkpoint payloads, or alter dashboard state. Static tests that import torch/runtime are not presumed device-independent; inspect their launch path first.
- Defer all GPU tests/profiling, simulator behavior evaluation, replay/dashboard validation, training/evaluation launches and hardware qualification until the active diagnostic process exits by user stop and both WX9100 leases are independently verified released. Missing/low utilization telemetry alone does not prove release. Respect the separate read-only device telemetry allowance in the integration sequence plan; no device acquisition/reconfiguration is permitted here.
- After release, complete the source plan's required current-checkout dashboard restart/asset checks and fresh seeded simulator/replay inspection for changed C1/C2 behavior, reporting scenario/seed, replay IDs, source/test-weight hashes, device identity and outcomes. Tests alone do not establish simulator behavior. Keep the active training uninterrupted until the user manually stops it.
- Require exact-head Pass 3 acceptance before integrating this amendment or its implementation, updating integration status, or claiming the C1/C2 gate closed. An accepted code candidate remains pending integration and the deferred runtime gates until those gates separately pass.

## Pass 1 audit evidence

Static inspection of the planning base found:

- `frc_defense/meco_candidates.py` documents a stable K-prefix plus one appended continuation candidate and creates device candidate IDs/metadata over the full selectable pool. The approved view cap must therefore be explicit and independent from the larger route/recall pool.
- `frc_defense/meco_tracks.py::MecoInferenceRequest` binds host `candidate_ids`, support mask/digest and fallback to the candidate tensor width/order; `ProductionMecoAdapter` validates candidate-mask and feature widths against `len(request.candidate_ids)`. `make_meco_inference_request` requires device candidate IDs to equal `CandidateTable.candidate_ids`.
- `frc_defense/meco_tracks.py::C2MapCrossAttentionGRU.forward` computes `candidate_context = queried.mean(1)`, so masked or inactive rows still change recurrent state unless the model view itself is capped and pooling is made mask-aware.
- `frc_defense/meco_tracks.py::score_candidates` currently requires the device batch ID tuple and feature/mask widths to match the full `CandidateTable`; the amendment must introduce a versioned, separate K-view contract rather than loosening these identity checks.
- Rejected candidate `8befd2df92a93d9e19c9c721952ab7b645eedebe` demonstrates that a full-width input with rank masks does not satisfy the required K cap and allows extra rows to affect C2 context.

