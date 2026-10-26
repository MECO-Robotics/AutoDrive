# P6 five-track supervisor, evaluation and qualification package plan

**Date:** 2026-10-26  
**Status:** Planner-authored documentation candidate; pending independent exact-head Pass 1 review  
**Planner:** `/root/closure_wave1_reviewer` (planning only; not implementer, reviewer, operator or human approver)  
**Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`

## Accepted contract and dependency pins

| Dependency | Exact accepted plan / artifact | File SHA-256 | Acceptance record |
|---|---|---|---|
| Overall plan and P0 sequencing | commit `6325920252773cfa05d63167ba3c29a90debdb01`, `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md` | `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a` | Pass 1 ACCEPT record commit `be845d7fb2d2ece861dcd158ac8b8dd89e3dc59d`, record SHA-256 `5a6e94f978c8a7e18d9623897ceba6f32f5b5eadb0aba6605f21c9d1e941a886` |
| P0 Phase 0 package | commit `13dbd7f8f953995850e811202f7e9c5264c49dca`, `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md` | `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf` | Pass 1 ACCEPT record commit `037f0f9386cc0efadb1fb54efe5f38f097926d0e`, record SHA-256 `1ee70a6e9d24dc5201da5eec446820167cb8c4d4436fbbe013eb16de66b0280d` |
| P1 V4 schema, score and artifact package | commit `53686a254318484f9c6bc3496ecc1c461572c53a`, `docs/plans/meco-five-track-v4-schema-score-artifact-implementation-plan-20261026.md` | `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076` | Pass 1 ACCEPT record commit `854d5286947074ed33f7ac059d67f2f9ccbf9257`, record SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f` |
| P2 GPU-resident environment and route package | commit `8d0727f10d46bfc5fee2810a1ec7fa7987ecaf91`, `docs/plans/meco-v4-p2-gpu-resident-environment-route-package-plan-20261026.md` | `5eddd42914ab74426257ab8c1ff883b25c0c39aaa40bf7c40ccc2e363fbb3957` | Pass 1 ACCEPT record commit `0292f01638bf3180b4952721f5692a4446841013`, record SHA-256 `ad43d8e9c91415eb08fabdced1ba85aabaf0849dc15cc302c8f2965c4c0b31bb` |
| Reconciliation / canonical target profile | commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c` | Reconciled profile SHA-256 `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e` | Pass 1 ACCEPT record commit `5cb1a2d525168ff0443089cf329e66c6ecb917b8`, record SHA-256 `0df700dcaef1ed35672efc468bdcfd78fac829c31bcf7a88b3c51f1951f06c32` |

The envelope digest is the raw proposal digest `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`. The accepted evaluation estimand is `simulated_match_score`: accepted active-HUB FUEL awards only. The fixed strategic horizon is 160 physics ticks, at most 12 ticks per macro (last may be partial), gamma 0.993 per macro, exponent zero on the first branch and one increment per continuation decision. Route and risk costs remain separate from gross reward. These are source and protocol targets, not claims of current conformance.

### P3, P4 and P5 package-plan status; implementation heads remain pending

P3 and P5 have accepted Pass 1 package plans. Those ACCEPT records approve only the plan artifacts; they do not accept implementation code, API interfaces, or integration heads. P4 remains pending Pass 1 review. P6 requires the exact accepted implementation/API and integration heads for all three packages before its Pass 2 source work starts.

| Package | Current planning reference | Required before P6 Pass 2 |
|---|---|---|
| P3 Plan A A1/A2/A3 | Accepted plan commit `8b641ccc56b7c73a880a095b3c6940cb03b617f3`, file SHA-256 `6958cb883a339a251e6069378d8ebb853be97e4965d30ca0eba8d3003fbc0a9b`; Pass 1 ACCEPT record commit `a11e68b6240c6701a14d6fafe932f65f0b05d6e2`, record SHA-256 `8da034c3d3b01b1bf113fb429f4fc0f3ed7fa708d19279d266e71ca61918e685`. | Exact accepted P3 implementation/API and integration head; plan acceptance alone does not establish those interfaces. |
| P4 Plan C C1/C2 | P4 plan and implementation heads not pinned in this document; independent plan/artifact review is pending. | Accepted P4 plan/review, then exact accepted P4 implementation and integration head. |
| P5 PPO/RNG/end-cause/atomic checkpoint integration | Accepted plan commit `2742920fc22a87865bdcd03f162ef3a3b9a67978`, file SHA-256 `b9dee65073c9118745cf0cc0ab08606ff42d5220cf6ec0e25701829d11d30e62`; Pass 1 ACCEPT record commit `43901154cfb59981fca39137e7804cf74dd864c0`, record SHA-256 `59730f012fbe06a7a03a3274bf135c9ac73907890d0a5ab0e24311ac901c6223`. | Exact accepted P5 implementation/API and integration head; plan acceptance alone does not establish those interfaces. P5 remains sole owner of `frc_defense/meco_track_training.py`. |

P6 source implementation also waits for accepted P0a decisions and accepted, integrated P1/P2 interfaces. P0b is deliberately later: it depends on frozen, reviewed P1–P6 source; P6 runtime qualification depends on accepted P0b. P6 source must not wait on P0b, and P0b must not wait on P6 runtime results.

## Objective and authority

Implement the P6 portion of the accepted overall plan: v5 target-scoped qualification schema and runner; five isolated training lanes and supervisor; immutable checkpoint snapshotting; continuous read-only evaluation; Stage 1 qualification and promotion/report state; and qualification status data in the dashboard. Train the five tracks A1, A2, A3, C1 and C2 continuously until the user manually stops the training campaign. Run read-only evaluations from immutable snapshots while training continues. Each track must pass three consecutive Stage 1 rounds under the accepted frozen protocol, with reports bound to the exact source, profile, protocol and checkpoint bundle.

This is a Pass 1 package plan only. It does not start or authorize training, evaluation, tests, simulator, dashboard, build or device use; it does not satisfy G0, P0a, or P0b; and it does not claim human approval, implementation conformance, GPU residency, training progress or qualification. Do not inspect, stop, signal, reconfigure, evaluate through, restart or otherwise touch the existing diagnostic campaign, its workers, checkpoints, outputs, dashboard, devices or leases. Future source or runtime work requires exact independent acceptance and G0 evidence, including the user's manual stop of the prior campaign, independently verified GPU lease release, named distinct roles, and maintainer authorization for the new campaign.

The supervisor must keep training running through evaluation and through Stage 1 completion. It has no update-count or elapsed-time stop target. Only an explicit manual user stop ends training. Qualification success does not auto-stop workers. A manual stop may shut down training workers at committed update boundaries; a read-only evaluation already using immutable snapshots may finish or be explicitly cancelled according to the frozen supervisor configuration, without writing into training state.

## Source evidence and ownership boundary

Read-only inspection at the source base found:

- `frc_defense/meco_qualification.py` implements protocol versions v1–v4, case construction, validation, checkpoint identity, evaluation and promotion logic. Its current code is not the target-scoped v5 protocol and does not establish the accepted three-round-per-track contract.
- `frc_defense/meco_task_scheduler.py` imports seed-pool and case APIs from qualification code. P6 owns this scheduling boundary and must coordinate an exact API handoff with P5; P5 owns the trainer caller and any integration edits in `meco_track_training.py`.
- `frc_defense/meco_track_training.py` owns checkpoint creation and training/evaluation entry points in the current source. P5 is the exclusive owner of changes to this shared trainer. P6 must use P5's reviewed checkpoint-commit and evaluation APIs or subprocess interface; it must not edit trainer internals or write trainer state.
- `frc_defense/meco_dashboard.py` reads qualification-status and training/evaluation provenance for `/api/meco-status`. Existing qualification projection validates v1/v2 status and must be changed to recognize only the accepted v5 shape for new qualification claims while preserving legacy records as historical, non-v5 evidence.
- `frc_defense/dashboard.py` serves `/api/meco-status`; current dashboard page assets include `frc_defense/dashboard_static/meco.html`, `meco.js` and `meco.css`. P6 may update this status view within the finite allowlist below.
- Existing repository `AGENTS.md` requires every test that checks simulated game behavior to be paired with a representative seeded end-to-end simulator evaluation and inspected simulation record/replay. It requires current dashboard service/backend/frontend verification and a fresh replay after code changes before dashboard replay inspection.

These source locations are discovery leads, not proof of current behavior, target conformance, GPU residency, dashboard freshness, or Stage 1 success. P6 Pass 2 must refresh the audit against the exact accepted P1–P5 integration head, document API identities and hash them in its implementation ledger, and submit a plan amendment before editing any unlisted path.

## P6 contracts

### Protocol v5 and qualification decisions

- Encode the accepted target, source/profile/protocol/device identities, deterministic train/calibration/validation/final-test splits, case and seed membership, baseline pairing, score and safety metrics, uncertainty/margin rules, report schema, artifact hashes, promotion conditions and invalidation behavior.
- The product evaluation plan's primary target is to beat the strongest matched deterministic baseline on held-out outcomes, with paired initial states/opponent seeds and confidence intervals, without increased safety violations. P0a must freeze the finite held-out set, acceptance predicates, confidence method, safety margins, baseline identity and seed construction before source work. P6 must not invent or tune those decisions from candidate outcomes.
- P0b binds the final v5 protocol and immutable Phase 0/calibration artifact to the frozen reviewed P1–P6 source. The P6 runner fails closed on missing or mismatched source/profile/protocol/device/checkpoint identities, incomplete coverage, duplicate cases, invalid split membership, stale input, numerical error, insufficient evidence, incomplete rows or digest failure.
- Maintain independent training-progress, diagnostic-evaluation, Stage 1 validation and final-test records. Diagnostic evaluation never changes qualification streak, promotion, curriculum, training state or checkpoint selection. Only protocol-designated validation rounds can advance the Stage 1 streak. Final-test data are read only after policy/checkpoint selection is frozen and never feed selection or promotion.
- Stage 1 requires three consecutive complete passing rounds for each of A1/A2/A3/C1/C2. Every round reports the exact compatible five-track checkpoint bundle and source/profile/protocol identity. A failed case, invalid report, changed protocol/source/checkpoint bundle, or required source/device drift invalidates or resets the affected streak as the frozen protocol specifies; an incomplete round cannot count. A new immutable report is required for every attempt. Old-source, prior 100/100 diagnostics, partial runs, test fixtures and dashboard summaries are not qualification evidence.
- Promotion/report state is append-only and reproducible. It records pass/fail and the protocol-defined stage only; it cannot claim official match score, deployment readiness, or human approval unless a separate explicit human record authorizes that claim.

### Five training lanes and manual-stop behavior

- Start five isolated lane workers from fresh P5-produced, source/profile/protocol/device-bound checkpoints. Bind one track code and identity per lane. Use the accepted task/case/seed schedule, runtime, profile and device mapping. Persist independent progress, RNG, optimizer and checkpoint identities per lane.
- P6 may supervise processes and schedule each lane, but may not perform policy, physics, reward, score or qualification calculations on CPU. It consumes P5's reviewed interfaces without changing the trainer. One lane's failure cannot corrupt, silently restart, reseed, advance or overwrite another lane. Failures and sibling action are recorded; restart requires validation of that lane's exact committed manifest and frozen seed/environment reconstruction.
- Publish snapshots only after P5 confirms an atomic committed update. P6 copies the checkpoint and manifest into content-addressed, immutable storage, verifies bytes and identity, and never edits or hard-links a mutable trainer file as an evaluation artifact. Snapshot creation does not pause or reset the worker.
- Read-only evaluators load only immutable snapshots, open model weights read-only, and do not restore optimizer state, write model state, touch PPO RNG, advance scheduler state, mutate curriculum or write into source checkpoints. Verify hashes before and after each evaluation. Keep diagnostics separate from qualification and final-test splits.
- Schedule continuous evaluation using a preregistered, bounded GPU share on each device. The share must preserve ongoing training service and be reported with throughput/latency and device utilization. Evaluation never requests a trainer pause or lease transfer. No CPU simulation, CPU model, CPU scoring, CPU statistical/qualification compute, P600 fallback or device migration is allowed. CPU is limited to process supervision, static configuration, and opaque artifact/file/network I/O after GPU results are complete.
- Assign all five lanes and evaluation to the two approved WX9100 devices under the accepted non-oversubscribed resource policy. Persist per-process track, device UUID/index/PCI identity, ROCm/HIP/driver/PyTorch identity, kernel/run identity, and GPU-residency evidence. Missing devices, lease contention, unsupported kernels, unexpected device changes, invalid snapshots or fallback attempts fail closed before affected numerical work. Resource policy and evaluator shares must be frozen and independently accepted before the new campaign starts.
- The supervisor runs until an explicit manual stop. It does not stop on a score threshold, qualification success, update count, schedule completion, report creation or timeout. A supervisor crash must not falsely report stopped workers; restart requires a reviewed recovery procedure and exact worker/checkpoint inventory. Every stop, restart and lane failure is recorded.

### Dashboard and reports

- The dashboard shows the five track lane states; last committed update; source/profile/protocol and checkpoint bundle identities; evaluation recency; per-track Stage 1 streak and round state; explicit pending/failure state; and device/run identity sufficient to review freshness. It distinguishes training progress, diagnostic evaluation and qualification. It never presents an unverified or legacy report as v5 qualification.
- The dashboard reads only validated, bounded status projections; raw evaluation rows, held-out seed lists, checkpoint paths, secrets, or free-form exception details are not exposed. Dashboard writes never feed back into workers, snapshots, qualification state or promotion.
- Reports and status projections are atomic and content-addressed or otherwise immutable by evaluation/round ID. The dashboard labels old-source replay/status evidence as stale and surfaces missing freshness checks rather than implying completion.

## Dependencies and gates

1. **G0:** independent acceptance of the overall and applicable package plans; explicit user stop of the old diagnostic campaign; independently verified lease release; distinct named planner/implementer/reviewer/operator; explicit maintainer authorization for the new campaign; accepted contract-anchor integration and exact base. This plan does not claim G0 evidence.
2. **P0a:** accepted preregistration and recorded authorized human decisions before P1–P6 source changes.
3. **P1 and P2:** accepted exact implementation/API heads integrated into the common branch. P1 defines score/event/artifact boundaries; P2 supplies GPU-resident environment, observations, candidates/routes, device evidence and residency ledger. P6 may not alter their contract.
4. **P3/P4/P5:** accepted Pass 1 package plans, exact implementation heads, exact API/review records and reviewed integration heads. P3/P4 supply policy/feature model APIs. P5 supplies PPO/RNG, end-cause, atomic checkpoint, track worker and evaluation interfaces. P5 is the sole `meco_track_training.py` owner. Each remains an explicit pending dependency until those exact heads are pinned.
5. **P6 Pass 1:** independent exact-head ACCEPT of this package plan.
6. **P6 Pass 2/3 source:** implement only after all P0a and P1–P5 source dependencies are accepted and integrated. P6 source, tests, supervisor and dashboard projection must be frozen and independently reviewed before P0b.
7. **P0b:** run final calibration and emit immutable Phase 0 and target-scoped protocol artifacts only after P1–P6 source freeze/review. Any relevant source/config/case/device change invalidates the artifact and requires refreeze and recalibration.
8. **P6 runtime readiness:** accepted P0b and frozen v5 protocol; exact P1–P6 integration head; fresh compatible five-track checkpoints and immutable manifests; both WX9100 devices available; approved resource/evaluation share; clean source identity; current dashboard services; and separate explicit human authorization for the new continuous campaign.
9. **Qualification completion:** each track has three consecutive complete passing v5 rounds with exact bundle/protocol identities, complete immutable reports and no unexplained device/source drift. Training remains active until manually stopped.

The dependency graph is acyclic: P0a → P1–P6 source freeze → P0b → P6 runtime/Stage 1. P6 source does not depend on P0b output; P0b does not depend on P6 runtime results.

## Finite Pass 2 source and test allowlist

Pass 2 may edit only these source paths:

- `frc_defense/meco_qualification.py` — P6 sole owner of v5 schema, validation, runner, report, streak and promotion logic.
- `frc_defense/meco_task_scheduler.py` — five-lane assignment/seed scheduling owned by P6; expose a documented stable interface for P5 without editing P5 caller code.
- New `frc_defense/meco_v5_supervisor.py` — worker lifecycle, resource admission, manual-stop handling, monitoring and immutable progress records.
- New `frc_defense/meco_v5_checkpoint_snapshot.py` — committed checkpoint discovery, immutable snapshot copying, content/hash manifest and validation.
- New `frc_defense/meco_v5_evaluation.py` — continuous read-only evaluator and frozen Stage 1/final-test orchestration against P5's accepted evaluation interface.
- New `frc_defense/meco_v5_dashboard.py` — bounded, validated, read-only v5 status projection if it cannot be implemented cleanly in the existing P6 dashboard path.
- `frc_defense/meco_dashboard.py` — validated v5 status projection and explicit legacy/stale treatment.
- `frc_defense/dashboard.py` — read-only P6 status endpoint integration only; no changes to process control or generic serving behavior outside this endpoint.
- `frc_defense/dashboard_static/meco.html`, `frc_defense/dashboard_static/meco.js`, and `frc_defense/dashboard_static/meco.css` — render the scoped P6 five-lane/evaluation/qualification data.
- `docs/MECO_QUALIFICATION.md` — document accepted v5 schema/operations and historical compatibility after exact source decisions.

P6 may add only these tests:

- `tests/test_meco_qualification.py`
- `tests/test_meco_task_scheduler.py`
- `tests/test_meco_v5_supervisor.py`
- `tests/test_meco_v5_checkpoint_snapshot.py`
- `tests/test_meco_v5_evaluation.py`
- `tests/test_meco_dashboard.py`

No other source, test, static, build, data or output path is authorized. In particular, P6 must not edit `frc_defense/meco_track_training.py` (P5 exclusive), P1 score/event/artifact paths, P2 device/environment/route paths, P3/P4 model files, P0 preregistration/calibration artifacts, generated checkpoint/evaluation/qualification outputs, existing campaign files, dependency/build configuration, or unrelated dashboard modules. If exact integration discovery requires a different path or API owner, stop and submit a new finite Pass 1 amendment before editing it. Output artifacts from a later authorized runtime live under a new source-bound v5 run root outside Git and never overwrite prior runs.

## Acceptance evidence

### Source and contract

- All five track codes, profile fields, Plan A/Plan C capacities, score-event rules, reward/horizon/RNG/end-cause/checkpoint semantics and device identities match the accepted exact profile and integrated owner APIs.
- Protocol v5 has a complete canonical schema, deterministic target/case/seed closure, strict version/hash/source validation, disjoint splits, exact expected coverage, empty/missing behavior, uncertainty/safety computation, promotion rules and immutable provenance. No thresholds or held-out sets are improvised after seeing candidate outcomes.
- Five-track lane schedule and worker failure policy are deterministic and source-bound. Scheduler and P5 interfaces have exact accepted API pins. A single lane cannot change siblings' model, optimizer, RNG, progress, checkpoint or qualification streak.
- Snapshot code accepts only P5-confirmed atomic committed checkpoints and produces byte-verified immutable copies. Evaluation is read-only and has no API that can mutate the trainer or checkpoint.
- Three-round streak logic, candidate bundle selection, failed-round reset, stale-source invalidation and final-test isolation are tested against v5 artifacts. Dashboard projection is bounded and provenance checked.
- P6 has no CPU policy/simulator/scoring/statistical/qualification compute and no CPU/P600 fallback. Both WX9100 execution and continuous evaluator resource share are evidenced on the exact integration head.

### Tests and behavioral validation

- Schema/runner tests cover malformed/legacy/stale protocols, exact hashes, missing/duplicate/extra rows, target/case closure, split leakage, seed reproducibility, empty categories, finite values, uncertainty and safety boundaries, all three consecutive rounds, failure reset, source/checkpoint/device changes, interrupted reports and final-test isolation.
- Scheduler/supervisor tests cover all five lanes, stable track/device/seed binding, resource admission, worker loss, sibling isolation, committed-boundary stop, manual stop, restart inventory, no arbitrary completion stop, and no hidden restart or reseed.
- Snapshot/evaluation tests prove immutable bytes, before/after hashes, concurrent training checkpoint safety, frozen model weights, no optimizer/RNG/scheduler/curriculum mutation, and no writes to source checkpoints. Include failure injection and all-or-none report state.
- Dashboard tests validate current v5 status, stale/legacy evidence handling, no raw held-out data or paths, and correct separation of training/evaluation/qualification state.
- Any test that exercises simulated robot, score, collection, route or other game behavior is paired with a representative seeded end-to-end simulator evaluation that exercises the same behavior. Inspect and retain the simulation record/replay; record scenario, seed, replay ID, source commit and outcome. A unit test or summary metric alone is insufficient.
- Before generating or reviewing any replay, restart the dashboard services on the exact current integration checkout after source changes. Verify service state, backend module identity, frontend asset hashes and successful relevant endpoints; generate a new replay and verify its provenance. Mark old-code replays outdated. Record exactly which tests, simulator evaluation, scenario/seed/replay and dashboard freshness checks ran. If a required simulation, GPU or dashboard check cannot be completed, report it as missing and do not mark validation complete.
- Capture per-process and per-kernel WX9100 UUID/index/runtime/driver/PyTorch/HIP identity, transfer/synchronization traces and evaluator GPU share. Prove forbidden CPU/P600 placement fails closed. Record training throughput before and during continuous evaluation, without stopping or resetting training. No runtime validation occurs under this plan.

## Worktree, roles and commit cadence

After Pass 1 ACCEPT and all implementation prerequisites, use a fresh ZFS-backed worktree based on the exact reviewed integration head: `.worktrees/autodrive-v4-p6-supervisor` and branch `codex/v4-p6-supervisor-YYYYMMDD`. Never use the active campaign checkout or its output directories. The planner does not implement or review. Assign a distinct P6 implementer and independent P6 reviewer before Pass 2; P5 remains the trainer integration owner and an explicit API counterpart. Commit/push each coherent schema, scheduler/API, supervisor, snapshot, evaluator, dashboard, test and evidence slice before dependent work. Record exact base/head commits, blobs, file hashes, interface pins, test results, seeded replay provenance, dashboard freshness, device traces and open gates. Reviewer checks the exact pushed head and writes an ACCEPT/REVISE record. Integration changes form a new exact head and require review before runtime or dependent use.

## Risks and stop conditions

- **P0/P6 cycle:** P6 source precedes P0b; P0b follows P1–P6 source freeze; P6 runtime qualification follows P0b. Keep those milestones separate.
- **Checkpoint race or mutation:** P6 accepts only atomic committed P5 checkpoints. If immutable copy or hash verification fails, skip that evaluation and report it; never pause or alter the trainer to force a snapshot.
- **Evaluation interference:** time-share must be preregistered and measured. If evaluation causes a trainer stop, state mutation, lease conflict or unacceptable starvation, stop evaluator work and revise the resource plan; do not pause training silently.
- **Five tracks / two devices:** the resource policy must bind every lane and evaluator to the two approved WX9100s with no device ambiguity, memory overcommit, CPU fallback or hidden lease transfer. Capacity failure blocks runtime readiness.
- **Protocol ambiguity:** any unresolved human choice about target, score rules, held-out data, confidence method, margins or candidate-checkpoint selection blocks P0a acceptance or runtime qualification. P6 cannot infer approval from a generated protocol or dashboard state.
- **Final-test leakage:** final-test data cannot choose a checkpoint, tune a threshold, change curriculum or advance qualification. Any access before policy/checkpoint freeze invalidates the affected test evidence.
- **Dashboard false confidence:** only validated immutable v5 provenance is shown as qualification. Stale frontend/backend/replay inputs block dashboard evidence and are visibly marked stale.
- **No manual stop:** training remains active until explicit user stop; qualification completion, time or update count never ends it. Do not infer that the earlier campaign was stopped or that leases were released.

## Three-pass disposition

- **Pass 1 planner:** `/root/closure_wave1_reviewer`; planning only.
- **Pass 2 implementer:** unassigned; distinct from planner and reviewer.
- **P5 interface counterpart:** unassigned until P5 exact implementation/API head is accepted.
- **Pass 3 reviewer:** unassigned; distinct from planner and implementer, exact-head review required.
- **Runtime operator / human approver:** unassigned; separate from implementation and review roles.

Planner disposition: **PENDING independent Pass 1 review.** This candidate authorizes no code, tests, runtime, training, device, dashboard, campaign, calibration, human approval or Stage 1 result.
