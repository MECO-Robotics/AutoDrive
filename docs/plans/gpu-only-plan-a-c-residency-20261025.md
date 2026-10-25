# Pass 1 plan: GPU-only Plan A/C training and inference

**Role:** Pass 1 planner artifact  
**Status:** Proposed; requires independent Pass 3 review before implementation starts  
**Scope:** Enforce the user's persisted `offload everything to gpu: no cpu` constraint across the original MECO Plan A/C training and inference path.  
**Source-plan authority:** `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`  
**Plan basis:** source and documentation audit at `a9a34d8aa50b80b359447ab51ae4d54908c18d09`; no campaign data was inspected.

## Constraint is unresolved, not waived

The user's wording is a literal no-CPU constraint. The current Python/HIP application necessarily uses a host process to create and launch kernels, manage processes/devices, and write artifacts. This plan does **not** presume those duties are approved exceptions and does not relabel them as satisfying the request. Before any implementation claims compliance or a training run is authorized, the owner must resolve whether host process orchestration and artifact serialization are permitted. If the answer is literal zero host computation and zero host data transfer, the existing application cannot meet that requirement; do not start training under a weaker interpretation.

If the owner explicitly approves host duties, record the exact permitted operations in a versioned contract. Until then, the compliance status remains **not satisfied / unresolved**. In particular, CPU-based parsing, feature construction, candidate generation, route evaluation, physics, action selection, learning, optimization, and statistics remain prohibited by the stated constraint. CPU oracle execution for tests is also a separate CPU computation and must be declared rather than hidden.

## Audit: current hot-path gaps

The plan document specifies A1/A2/A3 as the three Plan A dense spatial scorers and C1/C2 as the two Plan C candidate rankers. It keeps legal route and mechanism checks authoritative and requires Plan C to score feasible, route-costed candidates. The implementation status at this source head explicitly records mixed CPU/GPU operation and says the CPU-free request is not satisfied (`docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`, “GPU/CPU execution audit”). The existing HIP exact-heap work is a gated prototype; the status records that normal Plan A support construction still uses CPU Dijkstra.

| Path | Source evidence | CPU work that must be removed from the active algorithm |
|---|---|---|
| Legal pose graph | `frc_defense/meco_pose_graph.py`, `build_legal_pose_graph`, especially the edge-list conversion and connected-region construction after edge masks are built | Device graph edges are copied to Python lists; component/region partition and coarse graph construction run with Python traversal, sets, and loops. Preserve canonical node and region semantics, or version the changed representation and retrain/revalidate affected models. |
| Plan A support | `frc_defense/meco_plan_a_policy.py`, `build_plan_a_support` | Root and geometry scalars are read to Python. The normal path calls `cached_plan_a_edge_weights` and `plan_a_dijkstra_cpu`; `require_hip_shortest_path=True` is a gated special mode. Pair masks are mostly tensorized, but support ordering, IDs, support digest, and downstream selection still have host synchronization/serialization. |
| Plan C candidate table | `frc_defense/meco_candidates.py`, `_candidate_target` and `build_route_feasible_candidates` | Per-target geometry, route specs, commitment reconciliation, candidate IDs/order, fallback, recall, and summaries use Python scalar reads and row lists. The stable K-prefix, commitment and sensed-target identity contracts must remain intact. |
| Shared route evaluator | `frc_defense/experimental_route_evaluation.py`, `evaluate_routes`, `_path_weighted_cost`, `_cost_map_digest` | The planner's tensor computation is device-side, but validation, row filtering, result assembly, endpoint checks and scalar extraction run through Python. Weighted terrain integration copies path/map tensors to CPU and loops over raster crossings there. |
| Plan A/C rollout and inference | `frc_defense/meco_track_training.py`, `_prepare_plan_a_decision`, `_prepare_decision`, rollout loop, `_run_macro_step` | A decisions have host-side support identity materialization and scalar reads; C decisions call the Python candidate builder. Rollout transitions are Python dictionaries. Macro action/termination/safety observations read device scalars per step. Production inference request IDs and artifact evidence are host objects. |
| PPO and statistics | `frc_defense/meco_track_training.py`, `_ppo_update` and update/stat persistence | PPO builds tensors from Python rows, converts device `randperm` to a Python list, updates one sample at a time, and extracts loss values/returns to Python. Report aggregation is Python-side. These are learning/statistics algorithms, not merely serialization. |
| Supervised Plan A and C fitting | `frc_defense/meco_supervised.py`, fitting entry points; orchestration in `frc_defense/meco_plan_a_ubp.py` | Record loading, target construction/masking, fitting loops and epoch loss extraction include per-row Python work. Preserve the separate U, B and P label authorities and their current provenance; do not synthesize labels or change P semantics to ease GPU porting. |
| Simulation macro and record capture | `frc_defense/meco_track_training.py`, `_run_macro_step` and evaluation record builders | The simulator uses tensor state, but macro control, termination, metrics, route traces and event checks repeatedly extract scalars or lists. State transitions and safety decisions must not move to an unreviewed host fallback. |

The existing path does have substantial GPU-capable components: model tensor operations, many simulator tensor operations, TensorADStar tensor sweeps, PPO tensor primitives, and a fail-closed HIP shortest-path prototype. Their existence does not prove end-to-end GPU residency because upstream/downstream work still crosses into host algorithms.

## Implementation checkpoints

Each checkpoint is a separate planner → implementer → independent reviewer tranche. Implementers must commit and push reviewable checkpoints incrementally. A checkpoint does not authorize the next one until its exact head receives Pass 3 approval. Do not change the active campaign or acquire devices during code-only work.

### G0 — Resolve and encode the requirement

- Obtain an explicit owner decision on the literal zero-CPU constraint and, if approved, enumerate host operations that are permitted. Do not infer approval from this plan.
- Add a versioned residency contract to source/config/manifests: selected physical device(s), enabled accelerator backends, forbidden fallbacks, allowed host duties (if any), and observable residency evidence.
- Make a missing/unsupported HIP operation fail closed. A CPU reference may be used as a precomputed test oracle only if the owner explicitly permits CPU test computation; it must never be selected by a training/inference runtime.

**Gate:** unresolved literal wording blocks any claim that the no-CPU requirement is met and blocks training dispatch.

### G1 — Plan A legal graph and support

- Move graph connectivity/region/coarse-edge construction and shortest-path geometry/cost calculation to the selected device. Replace normal CPU Dijkstra dispatch; do not leave a “portable” CPU branch reachable from GPU-only training.
- Keep graph node order, edge legality, heading wrap, region semantics, exact edge costs, `(distance,node_id)` tie order, parent reconstruction, physical distance, support order, legal behavior masks, fallback and empty-support behavior unchanged. If exact graph-region parity is not feasible, stop and submit a versioned-model-input redesign for review; do not silently alter A2/A3 inputs.
- Keep support tensors and action selection on device. Defer stable text IDs, cryptographic evidence, and artifact formatting to the owner-approved boundary, if one exists.
- Relevant APIs: `build_legal_pose_graph` (`meco_pose_graph.py`), `build_plan_a_support` and `plan_a_support_digest` (`meco_plan_a_policy.py`), `plan_a_dijkstra_hip_for_graph` (`meco_plan_a_shortest_path.py`).

### G2 — Plan C candidate and route batch

- Replace Python candidate construction with a fixed-capacity, padded tensor contract. Stable integer identities/order stay device-side through ranking and action selection. Candidate expansion, K-prefix, sensed-track revalidation, continue commitment, goal snapping, heading selection, capture envelope, legal-node masks, candidate recall and fallback must preserve current semantics.
- Convert `evaluate_routes` to a device result batch: feasibility/status masks, path distances, endpoint/footprint checks, ETA, opponent risk and weighted route cost. Remove CPU path/map transfers and Python raster-boundary integration in `_path_weighted_cost`. Any returned string IDs or point lists are artifact views produced only after the decision and only if host serialization is approved.
- Preserve TensorADStar's current occupancy, escape-from-blocked-start, footprint clearance, endpoint tolerance and route-cost units. Plan C remains candidate-level valuation; Plan A P remains route-cost shaping. Do not conflate the two.
- Relevant APIs: `build_route_feasible_candidates` (`meco_candidates.py`), `evaluate_routes` and `_path_weighted_cost` (`experimental_route_evaluation.py`), `_prepare_decision` (`meco_track_training.py`).

### G3 — Rollout, inference, PPO, and statistics

- Keep observation construction/normalization, recurrent state, model inference, support/candidate masks, sampling, action/goal tensors, simulation state updates, safety eligibility, reward, terminal flags and stale/fallback decisions on device.
- Replace per-step host scalar branching with batched device state/masks. Maintain current native legality and mechanism checks; neural output must never authorize illegal movement or override native collection/scoring/ferry safety.
- Store rollout data in device tensors with fixed model-specific shapes (A support over the frozen graph; C table padded to configured K plus the continue slot). Keep GAE, replay identity checks, PPO shuffling/minibatches, actor-credit exclusion for execution failures, advantage normalization, loss reduction, optimizer updates and running statistics device-side.
- Preserve exact replay/log-probability checks, recurrent temporal order, Plan A support binding, C candidate table/risk/route binding, model parity and resume determinism. Do not remove replay checks to avoid host synchronization; design device checks or place the evidence computation at an explicitly approved artifact boundary.
- Relevant APIs: `_prepare_plan_a_decision`, `_prepare_decision`, `_run_macro_step`, `_ppo_update`, and rollout persistence (`meco_track_training.py`); Plan A/C fitting in `meco_supervised.py`.

### G4 — Five-track residency and full-path guard

- Run A1, A2, A3, C1 and C2 through the same residency instrumentation. Exercise neural and deterministic fallback paths; a fallback must be GPU-resident too.
- Audit all active callers, including Plan A U/B/P warm-start fitting and Plan C fitting, not just the focal policy's inference kernel. Gate launch if any algorithm path resolves to a CPU implementation or an unapproved provider.
- Report host-to-device/device-to-host byte counts and synchronization events by phase. Any required host transfer must be listed against the unresolved constraint; it cannot be omitted from the report.

## Acceptance evidence

1. **Offline contract and parity:** reference vectors cover graph node/edge/region ordering, Plan A shortest paths and ties, support masks/fallback, Plan C IDs/order/K prefix/commitment/recall, candidate capture/legality, route feasibility/clearance/cost, action masks and exact rollout log probabilities. Acceptance requires semantic equality or a separately versioned and reviewed model-input change. No CPU oracle is executed in a training run.
2. **Device-residency proof:** on each released, physically attested WX9100, capture HIP/PyTorch profiler traces for the complete A/C path. Show every listed algorithm phase executes on the selected GPU, no CPU fallback/provider is reachable, no per-step scalar `.item()`/`.tolist()`/`.cpu()` synchronization exists in that path, and record all boundary copies. Source scans alone and GPU utilization are insufficient evidence.
3. **Training proof:** execute resumed and fresh short A1/A2/A3/C1/C2 training jobs only after gates open. Verify forward/backward, optimizer state, GPU-resident rollout/GAE/statistics, replay identity and log-probability, checkpoint round-trip, RNG continuity, source/protocol binding, and correct device assignment. Do not use a CPU training run as fallback.
4. **Required simulator validation:** tests of route choice, robot motion, collision/safety, collection, scoring, shooting, ferrying, fallback, or other game behavior must be accompanied by representative seeded end-to-end simulator evaluations and inspected records/replays. Retain scenario, seed, replay/scenario ID and result. No unit-only claim establishes behavior.
5. **Required dashboard validation:** after code changes, restart relevant dashboard services on the current checkout; verify active services and endpoints; confirm current backend provenance and frontend asset hashes; generate a fresh replay after restart; inspect replay provenance and mark any old-code replay outdated. Report the checkout/head, service verification, endpoint results, asset hashes, scenario, seed and replay ID. If unavailable, report the validation as incomplete.
6. **Independent Pass 3:** review the exact pushed head against this plan and the source plan. Verify literal constraint status remains explicit, no CPU algorithm path/fallback was missed, model/safety semantics remain intact, all-five coverage is real, and required simulator/dashboard evidence is correctly claimed.

## No-training-before-release gate

No implementation can stop, reconfigure, resume, or compete with the currently running campaign. Do not start GPU validation, simulator behavior tests, training, fitting, or device leases while it owns the devices. The user already specified two WX9100s, not the P600; after the user manually stops the existing run, independently verify both WX9100 leases are released before any hardware check. Then require: G0 owner resolution; exact-head Pass 3 approvals for G1–G4; source/runtime/freeze/partition and human estimand approvals required by the campaign plan; device parity/profile and residency evidence; and current dashboard/replay validation. Only then may the reviewed five-track manifest launch A1/A2/A3/C1/C2 on the two specified WX9100s. This artifact itself does not authorize that launch.

## Out of scope for this planner pass

No code changes, test execution, simulator evaluation, dashboard service action, GPU probe, campaign-data inspection, training action, device lease, or claim of CPU-free operation was made. This is a plan artifact only; the original MECO plan's remaining datasets, estimands, evaluation matrix, qualification gates and five-track completion obligations remain in force.
