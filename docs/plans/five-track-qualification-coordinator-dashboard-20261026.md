# Five-track qualification coordinator and dashboard rollup

**Status:** Pass 1 plan candidate only. No code, tests, training, evaluation, campaign, device, or dashboard-service actions are authorized by this artifact.

**Planning base:** `664092b93467fd926d8bb74e8e04f311489000c4`.

## Objective and scope

Connect the existing v4 single-candidate qualification CLI to a five-policy coordinator and a truthful dashboard rollup for A1, A2, A3, C1, and C2. Keep each policy's state, reports, checkpoint bindings, and three-round streak independent, while using one immutable calibrated v4 protocol. Schedule each evaluation on one exclusively leased WX9100 while the other WX9100 continues fair, non-starved training across all five lanes.

This tranche does not add or change curriculum gate math. `training_curriculum.evaluate_promotion` and `meco_qualification.consume_validation_round` remain authoritative for whether a report passes or promotes. The coordinator schedules the existing per-policy `run-validation` operation, preserves its immutable reports and state, and surfaces those results without inventing a second qualification decision.

This plan depends on the separate whole-compute GPU-residency work being implemented and reviewed first. Every coordinator-launched numerical or algorithmic operation must obey the user's GPU-only requirement: training updates, candidate generation/scoring, route/risk computation, paired evaluation, calibration, and statistical/bootstrap calculations run on GPU. Host work is limited to orchestration, lease management, and artifact I/O. CPU fallback, a missing GPU assertion, or an unbound status is a hard failure.

## Existing behavior and precise gap

- `frc_defense/meco_qualification.py` accepts one `--candidate-policy` and a state file bound to that policy. It persists the `CurriculumState`, a last-round summary, and a neighboring `qualification-status.json`. The status includes protocol, bundle/checkpoint identity, gate result, CIs, qualification flag, and streak summary.
- The v4 runner validates one full target round and invokes the existing promotion gate. A round uses the frozen protocol and its cases/seeds; promotion state resets after a failed round or protocol change. Reports are immutable and separately content-addressed.
- `frc_defense/meco_supervisor.py` orchestrates the active training/diagnostic campaign and publishes `qualification_eligible=false`; it does not coordinate v4 promotion reports across five learned policies.
- `frc_defense/dashboard_api.py` projects a prior curriculum run's `evaluation_history` and `consecutive_passes`. It does not load or validate the per-policy `qualification-status.json` outputs from the MECO qualification CLI.

Missing work is orchestration, cross-report provenance/indexing, lease-aware evaluation scheduling, and a five-row dashboard projection. Do not recreate the `>=100`/category success checks, v4 CI calculations, or three-pass rule in the coordinator or dashboard. They must consume results from the existing validator and promote path.

## Required state and binding model

1. **One frozen protocol.** Resolve one calibrated v4 protocol before any round. Record and pin its protocol ID, canonical file digest, source identity digest, runtime/native extension identity, target scenario, exact case table and seeds, paired-reference identity, D0 calibration bundle, reviewed margins, and safety schema. Every track invocation and every report must carry that same protocol ID and digest. The coordinator rejects a missing protocol, a protocol change, a report for another target, or a policy status bound to another protocol.
2. **Five independent policy states.** Maintain one immutable state file and one append-only coordinator index per candidate policy (`A1-state.json`, `A2-state.json`, etc.), with the qualification CLI's required `candidate_policy` value matching the filename/manifest key. Store reports under unique, policy-specific paths. Never share a mutable `CurriculumState`, reuse a report ID, or let a streak for one policy contribute to another. Each state references its own latest checkpoint bundle and bound report IDs/digests.
3. **Three-round evidence chain.** Preserve all three consecutive passing round report IDs and content hashes per policy, not just a displayed count. At each completed round, append a record containing track, candidate policy, scenario, evaluation/report IDs and hash, protocol ID and hash, checkpoint bundle ID and hash, category episode/success evidence, case/seed coverage digest, v4 CI summary, device identity/lease ID, completion time, and CLI promotion decision. The coordinator verifies that the underlying report and state file agree before indexing them. It may report `qualified` only when the authoritative state and immutable reports demonstrate three passes for this policy under the pinned protocol. A failure, aborted round, state inconsistency, or protocol mismatch clears the displayed active streak for that track while retaining history.
4. **Freshness and failure semantics.** Missing status is `not_started` or `unknown`, never `passed`. Invalid hash, stale output, malformed state, wrong candidate, wrong source/protocol, incomplete report chain, or unavailable report is `invalid`/`stale` with an explicit reason. Do not retain a previously displayed green state when current evidence cannot be read. Dashboard refresh must never mutate qualification state.

## Bounded implementation sequence after Pass 1 approval

Each numbered item is a separate reviewable tranche. No later step is implied by this plan's approval.

1. **Keep r4 isolated.** Continue the currently authorized r4 diagnostic campaign unchanged and uninterrupted until the user manually stops it. Do not change its manifest, source checkout, status, checkpoints, process, leases, device assignment, outputs, or dashboard serving process. Build only in isolated HDD-backed worktrees. Do not run tests or compute on the two devices while r4 holds them.
2. **Freeze the coordinator contract.** In a docs/schema tranche, define a versioned coordinator manifest containing the five explicit track entries, candidate policy IDs, state/report/status paths, exact pinned protocol ID/hash, source/runtime identity, expected WX9100 identities, output roots, and lease policy. Reject duplicate policy IDs/paths, missing tracks, cross-track path reuse, mutable protocol bytes, and any attempt to map a state/report to the wrong candidate.
3. **Implement a single-flight v4 coordinator.** Add a coordinator that schedules one policy's existing v4 validation invocation at a time on the designated evaluation device, waits for immutable report and status publication, verifies report/state/status digests and bindings, then advances that policy's state before scheduling the next job. Retain a deterministic queue order or a documented fair round-robin order across A1/A2/A3/C1/C2. Keep per-policy round indexes and retries separate. An aborted or failed report uses the existing gate's persisted failure semantics; retry must use a new unique evaluation ID/output and cannot erase the audit record.
4. **Add a shared exclusive lease arbiter.** Before evaluation starts, atomically acquire the evaluation WX9100 lease and verify the training scheduler owns only the other WX9100. Configure the qualifying campaign so the evaluator runs exclusively on `cuda:0` (or the frozen designated evaluator device) while a fair, sequential training queue advances each of the five lanes on `cuda:1`. No two compute processes may hold the same indexed device simultaneously. Training may not be globally stopped during evaluation; each lane must receive non-starved turn-taking on the free device and resume from its latest compatible checkpoint. Release and journal the evaluation lease after every report, including failure, and refuse dispatch if lease state is uncertain. No CPU fallback.
5. **Publish a five-row status projection.** Add a dashboard API projection that reads only the coordinator manifest, policy-specific qualification status/state files, immutable reports, and lease/training status. Validate schema, hashes, policy/protocol/source/runtime/checkpoint bindings, report chain, case/seed coverage digest, CIs, status timestamps, and lease ownership before exposing a row. Display five independent rows with explicit states (`not_started`, `in_progress n/3`, `qualified`, `failed`, `invalid`, or `stale`), latest round/report ID, protocol ID, checkpoint bundle ID, category-level counts/rates, v4 CI gate results, current training lane/device/updates, lease status, and a short blocker/error reason. Never derive `qualified` from exposure, updates, an untrusted boolean, or a status timestamp alone.
6. **Preserve current dashboard telemetry.** Keep diagnostic campaign status and the new qualification rows clearly bound to different campaign/protocol identities. Do not translate r4's diagnostic output into qualification status. Keep existing endpoints compatible or version the projection explicitly; do not mix the old `evaluation_history` streak calculation with the v4 per-policy evidence chain.
7. **Complete source changes before freeze.** Coordinator, API, UI, lease integration, and GPU-residency code all change source identity because `curriculum_training.source_identity()` hashes source/config files under `frc_defense`. Finish implementation, code review, schema review, GPU-residency checks, and required tests before final source/runtime freeze. After that freeze, perform source-bound D0 calibration, matched-reference variance analysis, explicit margin review, and generate the immutable v4 protocol. Any later source/runtime/extension change invalidates that calibration and protocol for a new qualification run.
8. **Stop and release r4 before hardware verification or qualifying execution.** Only after the user manually stops r4 and all processes/leases are confirmed released may the separate GPU residency and lease behavior be exercised on the two WX9100s. Snapshot r4 checkpoints and verify exact source/runtime/model/config identities before considering their use. Current checkpoint loaders require exact `sources == source_identity()`; do not bypass this. If the finalized coordinator/GPU source is incompatible with a checkpoint, record exact mismatches and obtain a separate decision before conversion or retraining. Do not assume a from-scratch restart.
9. **Launch a new qualification campaign only after readiness.** Under a separately approved execution tranche, bind a fresh manifest to the frozen source, runtime, devices, five policy checkpoints, and calibrated v4 protocol. Require whole-compute GPU readiness, exclusive lease arbitration, unique output paths, and all other manifest readiness gates. Start no training/evaluation job before the manifest, source, device identities, protocol, and per-track state/report roots validate. This new campaign never modifies or resumes r4.

## Dashboard row contract

Each of the five rows must show the evidence for that policy only:

| Field | Required meaning |
|---|---|
| Track / policy | Exact manifest identity, e.g. A1/A1; validated against state and report |
| Qualification state | Derived only from verified per-policy v4 reports plus authoritative persisted state |
| Round progress | Consecutive passing count out of 3, with report IDs/hashes for all rounds |
| Protocol | Frozen protocol ID/digest and source/runtime identity match |
| Candidate checkpoint | Latest per-round checkpoint bundle ID/digest; no cross-policy substitution |
| Category evidence | `single` and `cluster` counts/rates, complete frozen strata and seed coverage |
| V4 comparisons | Category-level useful-throughput and safety CI/margin result from the immutable report |
| Training | Lane state, latest update, next fair queue turn, and compatible checkpoint resume status |
| Device / lease | Exact WX9100 identity, indexed device, exclusive owner, and current lease status |
| Freshness / blocker | Generated time, stale/invalid state, and exact failure reason when evidence is incomplete |

Exposure progress may appear in a separate diagnostic field, clearly labeled as scheduler exposure only. It cannot contribute to qualification state or round count.

## Acceptance evidence for a later implementation

- The coordinator processes all five policy identities independently under one immutable protocol, producing up to 15 distinct round reports and five separate state files. Every report is linked by digest to its corresponding policy and checkpoint; no report, state, or streak is shared across tracks.
- The coordinator delegates all promotion decisions to the existing qualification CLI/validator and can reconcile its index against the exact immutable reports and state files. Failure/abort, duplicate report, stale hash, wrong protocol, wrong candidate, partial case/seed evidence, or missing file cannot appear as passed.
- Dashboard API returns five explicit rows and validates evidence provenance. It shows no qualification pass for r4 and does not infer promotion from exposure or training update totals. Stale or invalid evidence visibly loses its qualified projection.
- Live scheduling evidence shows one evaluation lease on one WX9100 and one training lease on the other, never simultaneous compute on the same device. Training rotates fairly across all five policies and resumes deferred lanes; evaluation does not stop the training campaign. Both devices are verified WX9100s, and the whole-compute GPU-only contract is met.
- Reproducible seeded end-to-end simulator evaluation and replay inspection establish the behavior of the finalized GPU path; dashboard services and frontend assets are verified current before generating/reviewing the replay. Record scenario, seed, replay ID, evaluation/report IDs, source/protocol hashes, device/lease IDs, and results.
- The final source/runtime identity was frozen after all code changes; D0 and matched-reference calibration, reviewed margins, and v4 protocol bind that exact identity. Each track's three-pass evidence uses that single protocol and its own source-compatible checkpoints.
- Training remains live after qualification and all five rows continue to report non-starved lane progress with compatible latest checkpoints.

## Stop conditions and boundaries

Stop dispatch if r4 has not been manually stopped and its leases released; if either WX9100 is unavailable or not identity-verified; if device lease ownership is uncertain; if same-device compute overlap or CPU compute/fallback is possible; if the coordinator protocol or report hashes differ; if any status cannot be tied to one of the five candidate IDs; if source/runtime has drifted since v4 calibration; or if dashboard evidence is stale or incomplete. Preserve r4 as diagnostic-only. No device, campaign, service, or source-code actions occur under this Pass 1 plan.

## Pass 1 review criteria

Confirm the plan uses the existing v4 promotion gate without duplicating its math; binds five independent candidate states and report chains to one immutable protocol; defines truthful, provenance-checked dashboard rows; provides exclusive two-WX9100 evaluation/training leases with fair non-starved lane progress; preserves GPU-only fail-closed behavior; sequences all source edits before calibration; and leaves live r4 untouched and uninterrupted. Approval authorizes planning only.
