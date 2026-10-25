# Pass 1 plan: GPU candidate eligibility and fallback scoring

**State:** implementation plan only; no code or runtime changes in this tranche.  
**Source base reviewed:** `53f3d9c9a49a7e12bf65b5e1fa2797f4464e63be`.  
**Runtime constraint:** leave the live diagnostic r4 campaign uninterrupted and unchanged. Defer GPU/runtime validation until the user stops r4 and both WX9100 leases are released.

## Objective

Remove host-side candidate construction, eligibility decisions, and fallback scoring from the live five-track decision/evaluation path. Candidate features, masks, ordering, route/risk values, fallback choice, scores, and winner selection must remain on the selected GPU. A host copy is allowed only for the already-final action command and passive reporting/serialization that cannot affect the decision.

This tranche is specifically the base action table/spec path and the remaining host candidate/fallback path. It does not redo device-native Plan A support, routes, or scoring, nor the existing device-resident C1/C2 and D1/D2 scoring branches.

## Source callchains and bounded scope

| Surface | Current path and issue | Planned change |
| --- | --- | --- |
| `frc_defense/meco_tracks.py::build_safe_action_candidates` | Builds the canonical eight rows, but loops in Python over tensor-derived availability, heading, track IDs, features, and opponent distance; `.item()`, scalar conversions, and row-list construction can affect candidate contents and eligibility. | Produce a fixed-capacity tensor table/spec on `env.device`; preserve the eight action slots, row IDs/order, feature schema, unavailable-slot representation, and fallback semantics. Keep host strings/IDs as identity metadata only; never read tensor values to make their ordering or eligibility. |
| `frc_defense/meco_candidates.py::build_route_feasible_candidates` | Calls the base table builder, then host `target_rows`/`specs` expansion, route input construction, route feasibility, ETA/risk, eligibility mask, and fallback selection. Legacy path builds `CandidateTable` features/mask and computes fallback using Python values. | Route the qualifying GPU path through fixed-capacity tensors and the existing route/device interfaces. Keep current device-native route/support implementation intact. Do not add a second route algorithm. Compute validity, risk/ETA features, fallback priority, and candidate selection on device. |
| `_additional_sensed_target_rows` vs `_additional_sensed_target_specs` | The former converts candidate tensors with `.tolist()`/`.item()`, sorts and filters targets in Python. The latter already computes eligibility/order on device and transfers IDs for naming. | Preserve and extend the device-spec branch. Restrict CPU transfer to identity labels after device ordering is fixed. Do not use host IDs/rows to reconstruct or alter validity, ranking, or feature data. |
| `meco_track_training.py::_prepare_decision` | A1/A2/A3 dispatch to `_prepare_plan_a_decision`; C1/C2 request `device_resident=True`, but the C-family path falls back to `_route_cost_tensor` and table scoring when `device_batch is None`. | For the qualifying five-track path, require the device batch and reject missing/incorrect-device batches before scoring. Preserve A1/A2/A3's existing Plan A implementation without porting it again. Remove silent host table/scoring fallback from this qualifying path. |
| `meco_track_training.py::_d1_d2_scores` | Device branch scores `raw_features`, route/risk tensors, validity, and commitment cost on device. Legacy branch loops over `CandidateTable` rows and converts route/risk tensors via `.item()` before creating a tensor. | Retain the device branch. Make qualifying D1/D2 fallback fail closed unless a complete device batch is present; retire the Python score loop from this execution path. |
| `evaluate_meco` fallback (`meco_track_training.py`) | D1/D2 request device residency. Device path takes `argmax` on GPU and copies a gathered action payload to CPU; legacy path copies `argmax` via `.item()`. Candidate evidence may be materialized after choice. | Keep the final `argmax` on device. Permit one compact, validated action payload transfer for simulator dispatch after choice is final. Remove legacy fallback and ensure evidence/log serialization occurs after choice without influencing it. |

The `meco_supervised.py` and `meco_counterfactual.py` candidate-builder callers are not a second implementation target here. Audit their use when wiring the gate: they must not silently enter a qualifying live five-track run through the host builder. Any needed migration of those offline/auxiliary flows is a separately scoped tranche.

## Required invariants

1. Preserve the canonical eight base action slots, stable action indices and candidate identities, feature names/types, unavailable-row representation, candidate-K behavior, route-offset expansion order, and current fallback priority/tie behavior.
2. For a given simulator snapshot, candidate identity, target geometry, action legality, route feasibility, route ETA, risk, feature values, candidate ordering, fallback, logits, and selected index are computed from same-device tensors. No host-derived value may change eligibility, rank, score, fallback, or winner.
3. No `.item()`, `.tolist()`, `.cpu()`, NumPy conversion, or Python scalar/boolean extracted from decision tensors may feed candidate construction, validity, scoring, or selection. Host loops over candidate rows for decision math are prohibited.
4. Candidate batch, features, valid mask, route/risk tensors, logits, and action choice must all be on the explicit expected GPU with compatible shapes and one snapshot/step identity. Assert these properties at the boundary.
5. Missing device support, wrong device, malformed shape/schema, stale snapshot, non-finite decision inputs, or an empty feasible set must fail closed with the established typed/no-action error. Do not substitute the legacy table scorer, CPU selector, or a CPU minimum-cost fallback.
6. Host work is limited to non-decision metadata and dispatch after GPU selection. Candidate IDs may be serialized after device order is fixed. A final action payload may cross to host exactly once for simulator command dispatch; host code must not reconsider or replace it. Passive logs and reports may serialize tensors only after the decision is finalized.
7. Preserve existing device-native Plan A support/routes and C1/C2/D1/D2 device scorer. This plan changes their caller contracts only where needed to reject the host fallback; it does not reimplement those algorithms.

## Implementation sequence (future approved tranche)

1. Map all decision-affecting conversions in the scoped callchains and encode expected device, shape, snapshot identity, row order, and fallback rules in one source-level contract.
2. Convert the base eight-row builder to fixed-capacity device tensors and update candidate expansion to consume those tensors. Retain stable host identity labels only where they cannot influence math or ordering.
3. Wire C1/C2 candidate generation and D1/D2 fallback scoring to require device batches. Remove or guard the host `CandidateTable` branches from qualifying training/evaluation. Keep the already device-native Plan A path unchanged.
4. Centralize fail-closed checks at candidate construction and immediately before score/selection. Verify no alternate caller can silently route a qualifying run through host candidate logic.
5. After implementation review and once r4 has been manually stopped and both WX9100 leases are released, run deterministic GPU fixtures for candidate row/mask/order, route/risk, fallback/tie behavior, and failure cases. Then run the representative seeded end-to-end simulator evaluation and inspect its fresh replay. Restart and verify the current dashboard backend and frontend before replay generation/review, per `AGENTS.md`.
6. Freeze source identity only after all source changes are final. Recompute source hashes and calibration/protocol bindings; do not reuse an earlier source binding or checkpoint claim without the repository's exact source/architecture compatibility checks. Qualification reports must bind to the final immutable source and protocol.

No tests, simulator evaluation, device lease, GPU command, campaign mutation, or service action is authorized by this planning tranche.

## Acceptance evidence for the later implementation

- Static callchain review shows no decision-affecting host conversion or host candidate/score loop remains in the qualifying path; every fallback is explicitly fail-closed.
- GPU fixtures establish exact canonical row order/IDs, masks, feature semantics, fallback priority, tie resolution, and selected action for frozen cases, plus fail-closed behavior for missing/wrong-device batches, stale snapshots, malformed inputs, and no feasible candidate.
- Runtime device assertions and profiling show candidate construction/eligibility, route/risk decision values, scoring, fallback, and `argmax` stay on the assigned GPU. The only decision-result host transfer is the final command payload.
- A seeded end-to-end simulator run exercises the changed candidate/fallback behavior; retain its scenario, seed, and replay/result. Verify dashboard service health, backend response, and frontend asset identity against the reviewed checkout before generating/reviewing the replay.
- Evidence records exact source revision/hash, immutable qualification protocol/calibration binding, and exclusive use of both WX9100 leases after r4 releases them. Existing r4 diagnostic reports do not qualify this implementation.
