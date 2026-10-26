# P6 five-track supervisor, evaluation and qualification package plan

**Date:** 2026-10-26  
**Status:** Planner-authored documentation candidate; pending independent exact-head Pass 1 review  
**Planner:** `/root/closure_wave1_reviewer` (planning only; not implementer, reviewer, operator or human approver)  
**Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`

## Accepted contract and dependency pins

| Dependency | Exact accepted plan / artifact | File SHA-256 | Acceptance record |
|---|---|---|---|
| Overall plan and P0 sequencing | amended plan commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md` | `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90` | exact-head Pass 1 ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a` |
| P0 Phase 0 package | commit `13dbd7f8f953995850e811202f7e9c5264c49dca`, `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md` | `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf` | Pass 1 ACCEPT record commit `037f0f9386cc0efadb1fb54efe5f38f097926d0e`, record SHA-256 `1ee70a6e9d24dc5201da5eec446820167cb8c4d4436fbbe013eb16de66b0280d` |
| P1 V4 schema, score and artifact package | commit `53686a254318484f9c6bc3496ecc1c461572c53a`, `docs/plans/meco-five-track-v4-schema-score-artifact-implementation-plan-20261026.md` | `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076` | Pass 1 ACCEPT record commit `854d5286947074ed33f7ac059d67f2f9ccbf9257`, record SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f` |
| P2 GPU-resident environment and route package | commit `8d0727f10d46bfc5fee2810a1ec7fa7987ecaf91`, `docs/plans/meco-v4-p2-gpu-resident-environment-route-package-plan-20261026.md` | `5eddd42914ab74426257ab8c1ff883b25c0c39aaa40bf7c40ccc2e363fbb3957` | Pass 1 ACCEPT record commit `0292f01638bf3180b4952721f5692a4446841013`, record SHA-256 `ad43d8e9c91415eb08fabdced1ba85aabaf0849dc15cc302c8f2965c4c0b31bb` |
| Reconciliation / canonical target profile | commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c` | Reconciled profile SHA-256 `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e` | Pass 1 ACCEPT record commit `5cb1a2d525168ff0443089cf329e66c6ecb917b8`, record SHA-256 `0df700dcaef1ed35672efc468bdcfd78fac829c31bcf7a88b3c51f1951f06c32` |

The envelope digest is the raw proposal digest `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`. The accepted evaluation estimand is `simulated_match_score`: accepted active-HUB FUEL awards only. The fixed strategic horizon is 160 physics ticks, at most 12 ticks per macro (last may be partial), gamma 0.993 per macro, exponent zero on the first branch and one increment per continuation decision. Route and risk costs remain separate from gross reward. These are source and protocol targets, not claims of current conformance.

### P3, P4 and P5 package-plan status; implementation heads remain pending

P3, P4 and P5 have accepted Pass 1 package plans. Those ACCEPT records approve only the plan artifacts; they do not accept implementation code, API interfaces, or integration heads. Before G0, P6 source authoring requires exact candidate source/API heads for each package with a persisted structural SOURCE/API ACCEPT. The exact final integration heads are a post-G0 prerequisite for P6 runtime validation and final acceptance, not for isolated source authoring.

| Package | Current planning reference | Required before P6 Pass 2 |
|---|---|---|
| P3 Plan A A1/A2/A3 | Accepted plan commit `8b641ccc56b7c73a880a095b3c6940cb03b617f3`, file SHA-256 `6958cb883a339a251e6069378d8ebb853be97e4965d30ca0eba8d3003fbc0a9b`; Pass 1 ACCEPT record commit `a11e68b6240c6701a14d6fafe932f65f0b05d6e2`, record SHA-256 `8da034c3d3b01b1bf113fb429f4fc0f3ed7fa708d19279d266e71ca61918e685`. | Exact candidate source/API head with independent structural SOURCE/API ACCEPT before P6 source authoring; final integrated head and post-G0 review before runtime validation/final P6 acceptance. |
| P4 Plan C C1/C2 | Accepted plan commit `8708ba8a96536f1a3ae28133afcd4ebcb4790378`, file SHA-256 `e611a87f4b257477f81241fbc67b3b967c59da600de8e26d696d6dbfacd47b1f`; Pass 1 ACCEPT record commit `0bc35e4deaa723483128e6cc1d15de502f05d4e1`, record SHA-256 `f382b90af519b9ccd4f3e02e9df3d9b99587ee3993d7e7b558a1987459aac33c`. | Exact candidate source/API head with independent structural SOURCE/API ACCEPT before P6 source authoring; final integrated head and post-G0 review before runtime validation/final P6 acceptance. |
| P5 PPO/RNG/end-cause/atomic checkpoint integration | Accepted plan commit `2742920fc22a87865bdcd03f162ef3a3b9a67978`, file SHA-256 `b9dee65073c9118745cf0cc0ab08606ff42d5220cf6ec0e25701829d11d30e62`; Pass 1 ACCEPT record commit `43901154cfb59981fca39137e7804cf74dd864c0`, record SHA-256 `59730f012fbe06a7a03a3274bf135c9ac73907890d0a5ab0e24311ac901c6223`. | Exact candidate source/API head with independent structural SOURCE/API ACCEPT before P6 source authoring; final integrated head and post-G0 review before runtime validation/final P6 acceptance. P5 remains sole owner of `frc_defense/meco_track_training.py`. |

P6 source-only authoring waits for accepted P0a decisions and exact P1–P5 candidate source/API heads with structural SOURCE/API ACCEPT records; these candidate interfaces do not authorize runtime use or integration. P6 source must not duplicate or alter those owner contracts. The contract-anchor bridge and integration against final P1–P5 heads occur after G0 and before P6 runtime validation. P0b is deliberately later: it depends on frozen, reviewed P1–P6 source; P6 runtime qualification depends on accepted P0b. P6 source must not wait on P0b, and P0b must not wait on P6 runtime results.

## Objective and authority

Implement the P6 portion of the accepted overall plan: v5 target-scoped qualification schema and runner; five isolated training lanes and supervisor; immutable checkpoint snapshotting; continuous read-only evaluation; Stage 1 qualification and promotion/report state; and qualification status data in the dashboard. Train the five tracks A1, A2, A3, C1 and C2 continuously until the user manually stops the training campaign. Run read-only evaluations from immutable snapshots while training continues. Each track must pass three consecutive Stage 1 rounds under the accepted frozen protocol, with reports bound to the exact source, profile, protocol and checkpoint bundle.

This is a Pass 1 package plan only. It does not start or authorize training, evaluation, tests, simulator, dashboard, build or device use; it does not satisfy G0, P0a, or P0b; and it does not claim human approval, implementation conformance, GPU residency, training progress or qualification. Do not inspect, stop, signal, reconfigure, evaluate through, restart or otherwise touch the existing diagnostic campaign PID `2750253`, its workers, checkpoints, outputs, dashboard, devices or leases. Once this package plan and all applicable dependencies are accepted, isolated P6 source/test-file authoring and static source/diff review may proceed in a fresh worktree on the exact accepted dependency base. Before G0, no build/compile, test execution, simulator or numerical work, calibration, evaluation, qualification run, dashboard operation, device/GPU use, training, checkpoint mutation, integration, merge or promotion may occur. None of those operations may begin until the user manually stops the existing campaign and both WX9100 leases are independently confirmed released; integration and runtime validation also require the reviewed contract-anchor base, named distinct roles and maintainer authorization for the new campaign.

The supervisor must keep training running through evaluation and through Stage 1 completion. It has no update-count or elapsed-time stop target. Only an explicit manual user stop ends training. Qualification success does not auto-stop workers. A manual stop may shut down training workers at committed update boundaries; a read-only evaluation already using immutable snapshots may finish or be explicitly cancelled according to the frozen supervisor configuration, without writing into training state.

## Source evidence and ownership boundary

Read-only inspection at the source base found:

- `frc_defense/meco_qualification.py` implements protocol versions v1–v4, case construction, validation, checkpoint identity, evaluation and promotion logic. Its current code is not the target-scoped v5 protocol and does not establish the accepted three-round-per-track contract.
- `frc_defense/meco_task_scheduler.py` imports seed-pool and case APIs from qualification code. P6 owns this scheduling boundary and must coordinate an exact API handoff with P5; P5 owns the trainer caller and any integration edits in `meco_track_training.py`.
- `frc_defense/meco_track_training.py` owns checkpoint creation and training/evaluation entry points in the current source. P5 is the exclusive owner of changes to this shared trainer. P6 must use P5's reviewed checkpoint-commit and evaluation APIs or subprocess interface; it must not edit trainer internals or write trainer state.
- `frc_defense/meco_dashboard.py` reads qualification-status and training/evaluation provenance for `/api/meco-status`. Existing qualification projection validates v1/v2 status and must be changed to recognize only the accepted v5 shape for new qualification claims while preserving legacy records as historical, non-v5 evidence.
- `frc_defense/dashboard.py` serves `/api/meco-status`; current dashboard page assets include `frc_defense/dashboard_static/meco.html`, `meco.js` and `meco.css`. P6 may update this status view within the finite allowlist below.
- Existing repository `AGENTS.md` requires every test that checks simulated game behavior to be paired with a representative seeded end-to-end simulator evaluation and inspected simulation record/replay. It requires current dashboard service/backend/frontend verification and a fresh replay after code changes before dashboard replay inspection.

These source locations are discovery leads, not proof of current behavior, target conformance, GPU residency, dashboard freshness, or Stage 1 success. Before G0, P6 Pass 2 refreshes the audit against the exact P1–P5 candidate source/API heads that received structural SOURCE/API ACCEPT and records their API identities in its implementation ledger. After G0, P6 refreshes the integration audit against the exact final P1–P5 integration heads before runtime validation and submits a plan amendment before editing any unlisted path.

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

## Dependencies and staged gates

### Source-only authoring gate

After the exact amended overall plan and this package plan are independently accepted, applicable package plans and source/API prerequisites are accepted, P0a and its required human decisions are accepted, and distinct P6 implementer/reviewer identities are assigned, P6 may author only allowlisted source and test files in a fresh isolated ZFS worktree. Before G0, branch from the approved implementation source base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22` or a pinned P1–P5 candidate source/API head with an exact structural SOURCE/API ACCEPT disposition. Do not add or depend on the absent reconciliation artifact in that checkout. P6's source-only prerequisites are the exact P1–P5 candidate source/API heads and their recorded static dispositions; a plan ACCEPT alone does not provide an implementation API. SOURCE/API ACCEPT authorizes dependent source authoring only, not runtime use, integration or final package acceptance. Final integration-head pins are added after G0 and before validation; they do not gate isolated source authoring.

This stage permits file authoring, static inspection and source/diff review only. It does not permit build/compile, test execution, simulator or other numerical work, calibration, evaluation, qualification, dashboard operation, device/GPU use, training, checkpoint mutation, integration, merge or promotion. Do not inspect, stop, signal, reconfigure, evaluate through, restart or otherwise touch live campaign PID `2750253`, its workers, checkpoints, outputs, dashboard, devices or leases.

### Pre-G0 structural source review

An independent reviewer may inspect the exact candidate for allowlist compliance, source/API shape and identity, static invariants, and P6/P1–P5 ownership boundaries. The record is SOURCE/API ACCEPT or REVISE. It cannot claim test/runtime conformance, accept the final package, permit integration, or authorize any downstream runtime use. P6 source candidates remain isolated until G0.

### G0 execution and integration release

G0 requires acceptance of the exact amended overall plan and applicable package plans; the user's explicit manual stop of diagnostic campaign PID `2750253`; independent confirmation that both WX9100 leases are released; named distinct owners/reviewers/operators; and explicit maintainer authorization for runtime/device/integration work. Before G0, no build/compile, test execution, simulator/numerical work, calibration, evaluation, qualification, dashboard operation, device/GPU use, training, checkpoint mutation, integration, merge or promotion may begin.

After G0, create the reviewed docs-only contract-anchor integration commit and verify its exact reconciliation SHA-256 and canonical profile hash. Pin the exact final accepted P1–P5 integration heads and their post-G0 final review records, then rebase or cherry-pick isolated P6 source candidates onto this reviewed bridge/integration base and review the resulting exact head; do not integrate or merge source-only branches earlier. The old campaign and its outputs are never reused.

### Post-G0 final validation and review

Run only the tests and validation authorized by the accepted P6 plan after G0. Complete required dual-WX9100 device/residency evidence, seeded simulator evaluations and fresh current-dashboard/replay checks. The independent reviewer then reviews the exact final commit, blob hashes, allowlist and all required evidence. Only final exact-head Pass 3 ACCEPT authorizes integration, dependent runtime use or P0b source freeze. Missing evidence blocks final ACCEPT.

### P0b, P6 runtime and completion gates

P0b may run only after P1–P6 source freeze and final review; it emits immutable Phase 0 and target-scoped protocol artifacts. Any relevant source/config/case/device change invalidates those artifacts and requires refreeze and recalibration. P6 runtime qualification waits for accepted P0b and frozen v5 protocol; exact P1–P6 final integration head; fresh compatible five-track checkpoints and immutable manifests; both WX9100 devices available; approved resource/evaluation share; clean source identity; current dashboard services; manual-stop/lease evidence; and separate explicit human authorization for the new continuous campaign. Each track must then complete three consecutive passing v5 rounds with exact bundle/protocol identities, immutable reports and no unexplained device/source drift. Training remains active until the user manually stops it.

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

After Pass 1 ACCEPT and source-only authoring prerequisites, use a fresh ZFS-backed worktree based on the exact approved implementation source base or pinned candidate head: `.worktrees/autodrive-v4-p6-supervisor` and branch `codex/v4-p6-supervisor-YYYYMMDD`. Never use the active campaign checkout or its output directories. Before G0, commit/push only allowlisted source/test-file slices and static source/API review dispositions; keep these branches isolated. After G0, rebase or cherry-pick onto the reviewed contract-anchor integration base, record the new exact head, then perform validation and obtain final exact-head ACCEPT before integration. The planner does not implement or review. Assign a distinct P6 implementer and independent P6 reviewer before Pass 2; P5 remains the trainer integration owner and an explicit API counterpart. Record exact base/head commits, blobs, file hashes, interface pins, test results, seeded replay provenance, dashboard freshness, device traces and open gates.

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

Planner disposition: **PENDING independent Pass 1 review.** After its acceptance and applicable dependency pins, this plan permits only isolated source/test-file authoring and static review before G0. It authorizes no pre-G0 test execution, build, runtime, training, evaluation, simulator, device, dashboard, campaign, calibration, human approval or Stage 1 result. G0 and final post-G0 validation/Pass 3 gates remain mandatory.
