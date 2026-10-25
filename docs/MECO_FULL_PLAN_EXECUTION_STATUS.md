## Phase 4 runtime-status documentation corrections — 2026-10-25

The reviewed focal-pose status row and the later stationary-blockage, direct-
pursuit, trench-interception, and legal-start rows are integrated in
`docs/MECO_PHASE4_RUNTIME_STATUS.md` at `384c9a2baa9a97a27271cc6ed26fb316f01be8ed`.
The rows describe source-level assignment, capture, and paired-reducer wiring
with their current evidence limits. These documentation corrections do not
provide a fresh end-to-end simulator evaluation or replay, held-out behavior
outcomes, or current-dashboard replay review. The focal-pose outcome measures
remain unavailable; stationary-blockage registry outcomes remain
unavailable/non-authoritative; direct pursuit, trench interception, and legal
start execution remain fail-closed pending their accepted source-bound
evidence. Qualification remains disabled.

The historical dashboard refresh note immediately below records a prior
87-entry ledger and predates this integration. It is not evidence for the
current checkout. Post-merge restart and current-checkout API/asset verification
are recorded separately below. See
[`phase4-runtime-status-ledger-dashboard-20261025.md`](implementation-provenance/phase4-runtime-status-ledger-dashboard-20261025.md)
and its entry in [`MECO_THREE_PASS_PROVENANCE.json`](MECO_THREE_PASS_PROVENANCE.json).

**Post-merge dashboard verification — 2026-10-25.** The provenance update
reached `main` at `637cc4b64fd5d1b22532305dd86fc14deb66a913`; the finalizer
passed exact-head review. The dashboard was restarted from this checkout and
verified active. Both APIs, the page, and both JavaScript assets returned HTTP
200; the curriculum API validated 88 summaries with
`phase4-runtime-status-ledger-dashboard` latest, and both served assets matched
the checkout byte-for-byte. The local ledger SHA-256 matches the dashboard API
pin. This is provenance/serving evidence only; no replay or simulator behavior
result is claimed. See
[`phase4-runtime-status-ledger-dashboard-currentness-20261025.md`](implementation-provenance/phase4-runtime-status-ledger-dashboard-currentness-20261025.md)
for response, digest, asset, and service details. The complete evidence note
was committed and pushed at
`612d5bc18bd371c8ddf6a15e147f489bd13e4fcb`; it is outside the digest-pinned
ledger.

## Dashboard refreshed for reviewed three-pass provenance — 2026-10-25

After the integration reached `main` at `86706ad9d`, `frc-defense-dashboard.service` was restarted from the current checkout (PID `3503505`, active). `/`, `/assets/meco.js`, and `/assets/dashboard.js` returned HTTP 200; each served body matched the current checkout byte-for-byte. The dashboard API provenance loader validates the pinned ledger locally as `validated_summary_only` with 87 tranches. I did not query `/api/curriculum-status` because this dashboard instance is configured to read the active r4 evaluation directory. No replay was generated or reviewed, so no simulator result, replay ID, or replay currentness is claimed. Training was left untouched.

# Phase 0 effective-v3 paired calibration code integration — 2026-10-25

The reviewed integration is pushed at `ac590f5487436735c78aef7671be5a9f8aaee187`. It binds the effective source/runtime receipt, freeze-v3, closed tuple registry and partition into paired D0/D1/D2 calibration, carries registered tuple IDs and full predictor contracts through derived artifacts, and revalidates the same bundle before training setup. Independent Pass 3 approved the freeze, paired, validator-fix, and combined heads. The combined focused offline suites passed **172 tests**; the dashboard provenance manifest now records this scoped tranche and its dashboard API digest pin has been updated.

This is code-contract evidence only. No production freeze or partition, human estimand approval, paired calibration result, margin approval, simulator/replay, current-dashboard service check, GPU/device evidence, or qualification was produced. The running r4 diagnostic campaign was left untouched; the user’s manual stop and verified release of both WX9100s remain required before new execution. The complete five-track plan remains in progress. See [`phase0-effective-bundle-paired-calibration-20261025.md`](implementation-provenance/phase0-effective-bundle-paired-calibration-20261025.md) and the corresponding entry in [`MECO_THREE_PASS_PROVENANCE.json`](MECO_THREE_PASS_PROVENANCE.json).

> **Five-track candidate schema fixture integrated — 2026-10-25.** Reviewed
> implementation `2fb402c69475ab3f43707a5eb0ec797e52b5fa8c` repairs the
> synthetic C1/C2 fixture to use the canonical 20-column raw schema and adds
> malformed name/order/width rejection coverage. The change is test-only; the
> runtime schema remains strict. The implementer reported 17 focused tests
> passing. No simulator behavior, replay, qualification, or deployment evidence
> is claimed. Three-pass summary is recorded in
> [`MECO_THREE_PASS_PROVENANCE.json`](MECO_THREE_PASS_PROVENANCE.json).

> **Phase 0 baseline provenance binding repaired — 2026-10-25.** The curriculum API had rejected the `phase0-baseline-heldout-contracts` summary because its planner digest referred to an obsolete revision. This scoped metadata correction binds the row to the current reviewed plan bytes, source commit `a5aebfb034e0329ed66f097ea38d8cf4e93723ee`, and mainline attribution `81dcf7ab668c6cb40d1e2e9c83648f2894d7c361`; it preserves the existing reviewer and implementation evidence. It adds no baseline or partition freeze, calibration, simulator, runtime, or qualification evidence. Dashboard recovery remains unverified pending integration and a current-checkout restart/check.

> **Phase 0 simulated score ledger integrated — 2026-10-25.** Code support is
> integrated at `457c469a5d2e00c56449aaeae0773d3cb6a689ff`; the independently
> reviewed implementation candidate was `c4f8a591b018e2208dcb3f3b60863155aac6929d`.
> It defines `meco-simulated-score-ledger-v1`, rules
> `meco_simulated_fuel_score_v1`, approval schema
> `meco-simulated-score-estimand-review-v1`, and outcome
> `simulated_match_score`, measured only in `simulated_score_unit` from accepted
> active-HUB FUEL score awards. The implementer reported 26 pure ledger tests
> passing, plus compile and diff checks; both production scorer-path cases remain
> skipped. No human estimand approval artifact exists. No seeded simulator
> evaluation, replay inspection, current-dashboard check, runtime evidence, or
> device evidence was produced. Phase 0 remains incomplete and qualification
> remains false.

> **Next scoped implementation tranche planned — 2026-10-25.** The independently reviewed Phase 4 legal-team-start executable-evidence plan is now carried on this branch from exact planner head `a9fdc9387a3cd8557f005863b4517781ef38ba9a` (artifact SHA-256 `4bf072a9e55c1ff7a9172315c7b895c68ae5e439edf064c8692136244a4bee5b`). The implementation will open at most one registered profile cell, only after per-world legality and exact paired outcome reconciliation plus accepted profile-bound replay/dashboard evidence. Current r4 remains active on its older diagnostic checkout; no GPU or simulator work is being started.

> **Dashboard refreshed for trench-gate provenance — 2026-10-25 06:58 UTC.** `frc-defense-dashboard.service` restarted from current `main` at `dfef1b22c` (PID `3123836`, active). `/`, `/api/meco-status`, `/api/curriculum-status`, `/assets/meco.js`, and `/assets/dashboard.js` returned HTTP 200. The curriculum API reports `validated_summary_only` with 74 three-pass tranches; latest is `phase4-trench-interception-runtime-gate`. Served asset SHA-256 values match this checkout: `meco.js` `51a67f3f77d38b23dad046b2189921724417ccc9cd581d3f9d8864573b32f3a2`, `dashboard.js` `487dff5cfb48d4d7e6d419be343d5e1149c291b4ba9866d227fa312972867219`. No replay was generated and no simulator result is claimed. The live r4 campaign was not interrupted.

> **Phase 4 trench-interception gate integrated for dashboard provenance — 2026-10-25.** The independently reviewed implementation at `40fbf7d4529594bb5575702209ae5c56638b1c3d` is merged into this isolated integration branch at `7b999e7fc`; the planner, implementer, reviewer, and summary-only ledger records are being committed incrementally. Static AST parsing and diff checks only. No tests, simulator, GPU, training, replay, or qualification was run. The gate remains non-executable until fresh current-source paired simulator records and inspected replay/dashboard evidence are accepted; this single cell does not close the 14-category trench requirement. The live r4 campaign was left untouched.

> **Dashboard refreshed for provenance ledger 73 — 2026-10-25 06:27 UTC.**
> Restarted `frc-defense-dashboard.service` from current `main` (PID `3090951`;
> active). `/`, `/api/meco-status`, `/api/curriculum-status`, `/assets/meco.js`,
> and `/assets/dashboard.js` returned HTTP 200. The curriculum API reports 73
> three-pass provenance tranches; `/api/meco-status` exposes the campaign ledger
> path in this checkout. Both served asset SHA-256 hashes match `main`:
> `meco.js` `51a67f3f77d38b23dad046b2189921724417ccc9cd581d3f9d8864573b32f3a2`,
> `dashboard.js` `487dff5cfb48d4d7e6d419be343d5e1149c291b4ba9866d227fa312972867219`.
> No replay was generated, no simulator evidence is claimed, and active training
> was left untouched.

> **Phase 4 direct-pursuit runtime wiring integrated — 2026-10-25 06:24 UTC.**
> Reviewed code is on `main` at `42a57af12`. The runtime now captures exact
> simulator pair-contact events and fixed-objective progress, checks matched
> outcomes, requires all 11 policy rows for both 100-seed arms, and changes
> only one preregistered pursuit slot while preserving the other category
> opponent. The compiler still refuses to mark the cell executable until a
> current-source paired simulator record, inspected fresh replay, and current
> dashboard evidence are accepted. Static AST and diff checks passed. No tests,
> simulator, GPU, replay, or qualification run was made; no behavior result is
> claimed. The r4 campaign remains active and was not interrupted.
>
> **Phase 5 provenance dashboard reverified — 2026-10-25 06:09 UTC.** After
> integrating the 72-entry three-pass ledger and digest pin at
> `577584417`, `frc-defense-dashboard.service` restarted on current `main`
> (PID `3073791`). `/`, both status APIs, and both JavaScript assets returned
> HTTP 200. The curriculum API reported `scoped-transcript-summaries` with 72
> tranches, and served `meco.js`/`dashboard.js` SHA-256 hashes matched this
> checkout. No replay was generated. No simulator behavior or qualification
> is claimed.
>
> **Phase 5 capture-bound scorer profiling implemented — 2026-10-25.** The
> reviewed software tranche is integrated at
> `7f374320011ce2579893b23adf190610e1a6740d`. It validates five track captures
> against selected checkpoints and requires an exclusive lease and physical
> attestation for both WX9100s before capture-backed scorer timing. Reports
> record distinct model-source and profiler-source identity and publish
> without overwriting existing paths. Contract tests were added but not run;
> Python compilation and diff checks passed. The dashboard three-pass ledger
> now contains 72 summaries, and its digest pin was revalidated after the
> ledger update. No production captures, trained five-checkpoint bundle, GPU
> profile, simulator/replay, Hailo, or qualification evidence exists. The
> r4 training campaign was not interrupted.
>
> **Phase 5 capture-backed scorer profiler implementation — 2026-10-25.** An
> isolated implementation adds a separate all-five production-capture profile
> mode, bound to selected checkpoint/source/protocol and a clean profiler Git
> commit. It requires the exclusive `cuda:0`/`cuda:1` lease and physical
> attestation of both pinned WX9100 devices; it has no CPU, single-card,
> P600, or synthetic fallback. Existing synthetic diagnostics retain their
> separate evidence labels. The profiler records capture/input/checkpoint/
> device/runtime provenance and reports scorer-only timing with transfer
> excluded. No tests, profile, GPU probe, simulator/replay, or training action
> was run for this code tranche. It creates no profile artifact or report
> digest, and it does not establish Hailo support, deployment qualification,
> or full-pipeline performance. Runtime evidence still requires a complete
> current-source five-checkpoint bundle, matching fresh captures, and both
> WX9100 leases to become available.

> **Phase 2 provenance and dashboard currentness — 2026-10-25 05:58 UTC.**
> The approved Phase 2 runtime tranche is registered as summary 71 in the
> dashboard three-pass ledger, whose digest pin now matches. After integration
> at `b52883d5b`, `frc-defense-dashboard.service` restarted on this checkout
> (PID `3061667`). `/`, `/api/meco-status`, `/api/curriculum-status`, and both
> served JavaScript assets returned HTTP 200. The API validated 71 scoped
> summaries; served `meco.js` and `dashboard.js` hashes match this checkout.
> No replay was generated. The independently reviewed direct-pursuit Phase 4
> planner plan is integrated in `main`; implementation is in progress in an
> isolated worktree. No qualification or simulator outcome is claimed.
>
> **Phase 2 runtime tranche integrated — 2026-10-25 05:55 UTC.** The
> Phase 2 five-track runtime plan is implemented and integrated at
> `054a21ea0fce42100a8c5eb5993d2e5c9e0bf53b`, after three exact-head review
> rounds. Physical WX9100 identity is checked after lease; production Plan A
> support routes for live A-track decisions, C collection/replay, and A/C
> fitting now fail closed through the gated HIP path. The fresh manifest
> handoff captures describe output outside the not-yet-created campaign root.
> The dashboard's summary-only three-pass ledger now contains 71 tranches,
> including this one. No tests, HIP compile, live device attestation, parity or
> profile probe, simulator/replay, or training launch was run for this tranche.
> The HIP gate remains closed until WX9100 parity/profile evidence exists, and
> the Phase 2 coordinator has not generated datasets, warm starts, or a seal.
>
> At the 05:55:27 UTC live poll, uninterrupted r4 supervisor PID `2750253` was
> still training with `stop_requested=false`: A1 queued at 100, A2 and A3
> complete at 200, C1 running at 185, and C2 running at 153, across `cuda:0`
> and `cuda:1`. It remains a diagnostic campaign on the older source, without
> the reviewed runtime gates, a protocol-bound matched evaluation schedule,
> Phase 2 warm starts, or qualification evidence. This tranche was pushed to
> `main`; the generated HIP source modification in the main worktree remains
> unrelated and untouched.
>
# MECO learning tracks plan: execution status

> **Diagnostic dashboard projection pushed — 2026-10-25.** Planner commit
> `2e9206c518a806c3588b0099503eb290d4d7a653` and reviewed implementation
> `817a6f9076d4c28fea34ba26634458b8f69c5547` are on `origin/main`. The
> dashboard now exposes the supervisor's manifest-bound diagnostic monitor
> through an allowlisted, freshness-aware projection and card; malformed or
> stale data cannot imply active progress, and the monitor remains
> qualification-ineligible. The focused dashboard suite passed 32 tests;
> compilation, JS syntax, and diff checks passed. The three-pass ledger now
> contains 70 summaries with its dashboard digest pin updated. No simulator,
> replay, GPU, or training action was part of this dashboard-only change; the
> active r4 training run was left untouched. After integration commit
> `0be3ff7f7`, `frc-defense-dashboard.service` restarted from this checkout at
> 05:32:03 UTC (PID `3034352`). `/`, both status APIs, and both JavaScript
> assets returned HTTP 200; the curriculum API validated all 70 summaries,
> with this tranche latest. Both served asset SHA-256 hashes matched this
> checkout.

> **Diagnostic monitor tranche reviewed and pushed — 2026-10-25 05:07 UTC.** The
> manifest-bound diagnostic evaluation monitor is approved at exact head
> `f79650107a5c92053d651a7d48d9bcd306257f1d`. The initial implementation and
> two corrections remain separate commits so the rejected review rounds are
> traceable. Its focused suite passed 71 tests; the final independent reviewer
> ran 51 monitor/manifest tests, and compilation/diff checks passed. The
> three-pass ledger now records 69 scoped summaries and the dashboard digest
> pin is updated. Commits `138fb998a`, `0dfd8798`, `12c9c432`, `f7965010`, and
> `f92d3aacf` were pushed to `origin/main`. The dashboard service restarted
> from this checkout; both status APIs and both frontend assets returned HTTP
> 200, the three-pass API validated 69 summaries, and served asset hashes
> matched the checkout. This is opt-in future-campaign plumbing only: no
> simulator replay, GPU execution, training, or qualification was run, and
> active r4 was left untouched.

> **Phase 2 coordinator code integrated — 2026-10-25 04:24 UTC.** The
> reviewed resumable Plan A/C complete-set coordinator is integrated on `main`
> through `59524d21a` and pushed. It provides a shared C1/C2 source campaign,
> A1/A2/A3 U+B+P bundle handoff, immutable stage receipts, a revalidated
> five-track seal, and exact manifest warm-start forwarding. Reviewer-approved
> implementation and test heads were `41efbb1ca6cd08c9cf45ad8811088b129130bfbf`
> and `d77cb19c0bcd91688f2aa3d0ac4d69c10efcd982`. Integrated checks passed:
> 43 manifest/coordinator tests and 2 targeted no-device supervisor launch
> tests; compilation and diff checks passed. The coordinator has not generated
> a dataset, weights, or seal; readiness remains false.
>
> The user-requested r4 run remains active on its older source and was not
> altered. Immutable status snapshot `docs/MECO_CAMPAIGN_SNAPSHOT_20261025T0423Z.json`
> (SHA-256 `c13e4dabb94bf36cbf4bcb0593fb7f4540e3df3e0a78215d2dac1c0996a759ca`)
> records cycle 2, one completed cycle, five completed training chunks, all
> five tracks at their first 100-update chunk, and no evaluation result in the
> supervisor record. At the subsequent live check, the supervisor had A2
> running its next chunk and a diagnostic one-episode evaluation process was
> still active. This run remains diagnostic `pickup_simple` exposure with no
> evaluation protocol binding or warm-start artifacts; chunk completion is
> not a held-out pass or qualification.
>
> The three-pass provenance ledger now records the approved coordinator
> tranche. After updating the ledger and pinned digest, dashboard process
> `2963161` was restarted from this checkout. `/`, both status APIs and both
> frontend assets returned HTTP 200; the curriculum API validated all 67
> scoped summaries, and both served JavaScript hashes matched the checkout.
> No replay was generated because this code tranche did not run a simulator
> behavior evaluation.

> **Live r4 and durable-evaluation update — 2026-10-25 03:46 UTC.** The
> supervisor PID `2750253` is active with both WX9100 lanes. Its queue
> heartbeat at `2026-10-25T03:46:36Z` shows A1, A2, A3, and C1 complete at
> 100/100; C2 is running at 35/100 on `cuda:0`. It is cycle 1, with no matched
> cycle completion or evaluation. This is still the older single-scenario
> `pickup_simple/cluster` diagnostic, without an evaluation binding, warm
> starts, or Phase 4 exposure. The five model tracks are being exercised, but
> this run does not establish that any track passed a curriculum or evaluation
> gate. C2 remains active; do not restart it while the user-requested
> uninterrupted run continues.
>
> The durable manifest evaluation schedule is integrated on `main` as
> `30367daa9`, `e28da846c`, `708c31033`, and `53157b4f8`. Independent exact-head
> review approved feature commit `d6bcab27637dbc2b39240e95ca994224d8809f31`.
> The queue retains exact scheduled cases and five-checkpoint snapshots,
> retries the same index after failed or interrupted evaluation, and advances
> only after verified report/status/history. The integrated focused suite
> passed 115 tests and 5 subtests, with 3 lease-acquiring tests deselected.
> The follow-up Plan A device-resident route prototype is integrated as
> `fa59b2dcd` and independently approved as a gated prototype. Thirteen
> focused CPU source-contract tests passed; runtime support construction still
> uses the CPU oracle, and the WX9100 parity/profile gate remains closed. No
> HIP compile, device parity/profile, or simulator replay has been completed.
> The Plan C repeated-label corpus audit is integrated as `70397a016` and
> `5389e618a`, with independent exact-head approval of `dbf1da3b`. Twenty-two
> focused offline tests passed. No real Plan C dataset has been audited, so
> this provides neither labels or warm starts nor simulator outcomes.
> No simulator evaluation or replay was run. After the route-prototype source
> updates, the r5 manifest was regenerated and validated offline at clean
> source `5389e618a5aad99b32753de20fb9c1d47cdce888`, canonical digest
> `87c5abcb30cacaa6b4cd612d31028a9c1ab5215950a12be46a57b7c40c73c542`.
> The current config/manifest are
> `meco-five-continuous-20261025-r5-plan-c-audit-5389.config.json` and
> `meco-five-continuous-20261025-r5-plan-c-audit-5389.json`. The manifest binds
> all 466 validation cases and 46,600 exact seeds; both campaign output roots
> are absent. Launch, Phase 2, validation, qualification, and warm-start
> readiness remain false. It is not launched while r4 holds both WX9100 leases.

> **Why the overall goal is still incomplete — 2026-10-25 03:46 UTC.** The
> goal has been active about 45 hours, while the current uninterrupted r4
> service has run about 3 hours. The four `100/100` values are one diagnostic
> optimizer chunk per track, not 100 evaluation cases or curriculum passes;
> C2 has not finished that first chunk. The active run covers only
> `pickup_simple/cluster`, has no matched evaluation, no warm starts, and no
> Phase 4 exposure. The plan's remaining work includes usable Plan A/C data
> and warm starts, broad matched held-out evaluation with inspected fresh
> replays, Phase 4 generalization, and deployment/qualification evidence.
> Recent elapsed effort also went into source-bound manifests, training
> orchestration, dashboard/provenance support, and scoped Plan A/C tooling;
> those are prerequisites and do not count as model-performance evidence.
> Therefore elapsed time cannot be reported as plan completion, and the
> current training run alone will not finish the full plan.
> r4 was not changed or interrupted.
>
> The user dashboard was restarted on the integrated checkout at 03:33 UTC
> (PID `2907164`). `/`, `/api/meco-status`, `/api/curriculum-status`,
> `/assets/meco.js`, and `/assets/dashboard.js` returned HTTP 200. The
> curriculum API reports `validated_summary_only` with 65 scoped tranches.
> Served asset hashes match the checkout: `meco.js`
> `203e4c2d680d0c30465e2e868dc82a3b5cbf63511e4cb6273ae3650fbd91f7ba` and
> `dashboard.js`
> `487dff5cfb48d4d7e6d419be343d5e1149c291b4ba9866d227fa312972867219`.
> No fresh replay was generated or reviewed.

> **Live five-track campaign recheck — 2026-10-25 03:00 UTC.** The user
> systemd service `meco-five-continuous-20261025.service` is active (PID
> `2750253`). A1, A2, and C1 have each completed 100/100 updates. A3 is at
> 66/100 on `cuda:0`, and C2 is queued. The status heartbeat is
> `2026-10-25T03:00:10Z`; cycle 1 has three completed chunks, zero completed
> cycles, and no evaluation yet. This is still the pinned older-source
> diagnostic campaign, using only `pickup_simple/cluster`, without evaluation
> protocol binding, warm starts, or Phase 4 training exposure. The first two
> tracks reaching 100/100 means their training chunks completed; it does not
> mean they passed a curriculum stage or evaluation gate. All five remain in
> the continuous training schedule.

> **Phase 4 training exposure manifest — 2026-10-25.** The reviewed tranche is
> integrated in `main` as `24a280402`, `5659d7cfd`, and `e843d7ee6`. Campaign
> manifests can now bind the opt-in Phase 4 exposure flag, and durable
> evaluation history and transition auditing bind the campaign configuration
> and verify exactly one correctly pathed and hashed snapshot per track.
> Independent exact-head review approved the feature head before integration.
> Twenty-five focused CPU checks across the history/transition/scheduler/eval
> suites, compilation, and diff checks passed. A follow-up integrated run of
> continual-evaluation, manifest, Phase 4 exposure, and provenance suites passed
> 64 tests; supervisor dispatch tests were not run because their device-lease
> tests conflict with the active two-GPU campaign. The dashboard provenance
> ledger validates as 62 scoped summaries. After restarting
> `frc-defense-dashboard.service` (PID `2854687`), `/`, both status APIs, and
> both frontend assets returned HTTP 200; served JS hashes match this checkout.
> The active r4 manifest and workers were not changed. No simulator evaluation,
> replay, GPU check, outcome run, or qualification was performed.

> **Prepared r5 handoff — 2026-10-25.** An offline manifest and source config
> are saved under the Campaigns manifest directory for a fresh
> `meco-five-continuous-20261025-r5` root. The clean source worktree is pinned
> to `b02db09f330642ad5498b6e76752d1df582c9089`; the manifest digest is
> `cbc18e88e7d967cf3770bec3335fa6f8963a11e58d0b2d3364684f3ffe98cb1f`.
> It binds A1/A2/A3/C1/C2, two WX9100s, Phase 4 training exposure, and all 466
> protocol validation cases (46,600 exact seeds). Offline manifest validation
> passed, while launch, Phase 2, validation, qualification, and warm-start
> readiness remain false. It is not launched: r4 still owns the two device
> leases. Planner and independent reviewer found that the current evaluator
> replaces pending jobs and advances past failed or incomplete cases; the
> reviewer rejected this manifest as a guarantee of full validation coverage.
> A durable no-loss schedule dispatcher/recovery tranche is required before
> claiming the 466 cases will execute. No r5 training, simulation, GPU
> evaluation, or qualification has occurred.

> **GPU/CPU execution audit — 2026-10-25.** The active A3 and C1 workers are
> assigned to the two WX9100s and use HIP tensors, but execution is hybrid, not
> CPU-free. Plan A transfers graph data to Python and runs `heapq` Dijkstra;
> Python orchestration, host synchronization, and logging remain on CPU as
> well. Fuel contact coloring also has a Python transfer path when the
> `colored` solver is selected, but the configured default is `jacobi`, so that
> coloring path is not claimed as active in r4. No runtime setting moves these
> remaining paths wholly onto GPU. The user's CPU-free request is therefore
> not satisfied; the campaign was not changed. A GPU-hot-path conversion needs
> separate implementation and seeded simulator/replay evidence.

> **Plan A HIP exact-heap hardening — 2026-10-25.** The reviewed prototype is
> integrated as `4164f4988`, with exact feature commit
> `5f199c68efdc4e17ae77fa4e286c15adaf9093d6`. It adds fail-closed WX9100,
> shape/root/cost, and heap-capacity checks. Dispatch remains disabled and
> training still uses CPU Dijkstra; no speedup or CPU-free result is claimed.
> Ten CPU contract tests, compilation, and diff checks passed. No HIP compile,
> GPU parity/profile, simulator evaluation, or replay was performed. The
> dashboard's three-pass ledger now records this as tranche 63. After updating
> the ledger digest, `frc-defense-dashboard.service` was restarted (PID
> `2874864`); the page, both status APIs, and both frontend assets returned
> HTTP 200. Served asset hashes match this checkout, and the API validates all
> 63 tranches as scoped summaries.

> **R4 live training snapshot — 2026-10-25 02:10 UTC.** The service remains
> active (PID `2750253`). A1 completed its first 100-update chunk; A2 reached
> 100/100 updates and was still finalizing that chunk; A3 had started at 6/100
> on `cuda:0`; C1 and C2 remained queued. One chunk is complete, with no full
> five-track cycle or evaluation complete. This campaign is pinned to
> source `9bf7d8d55e75821b67eb1c5d1c01871f666ea85d`, uses only the diagnostic
> `pickup_simple/cluster` assignment, has no evaluation-protocol binding, and
> consumes no warm-start artifacts. Its online task counters at the same
> snapshot were A1 0/30 and A2 19/40 successes; these are training episode
> counters, not held-out evaluation or qualification results. The scheduler is
> progressing through all five lanes two at a time, but this campaign does not
> establish the plan's curriculum or evaluation gates.

> **Three-pass dashboard refresh — 2026-10-25 02:29 UTC.** After integrating
> the reviewed continual evaluation schedule and registering its three-pass
> summary, `frc-defense-dashboard.service` was restarted on this checkout (PID `2840384`,
> working directory `AutoDrive`).
> `/`, `/api/meco-status`, `/api/curriculum-status`, `/assets/meco.js`, and
> `/assets/dashboard.js` returned HTTP 200. The curriculum API reports 61
> validated scoped tranches. Served frontend hashes match this checkout:
> `meco.js` `6178a524853b0223d4c6975c9a11b7dc7dc54131e41b383374a98e9a62533c`
> and `dashboard.js`
> `487dff5cfb48d4d7e6d419be343d5e1149c291b4ba9866d227fa312972867219`.
> No simulator replay was generated or reviewed.

> **Phase 4 legal start-profile outcome wiring — 2026-10-25.** The reviewed
> implementation is integrated at `b2509fc44` and its provenance-bound digest
> follow-up at `1929e355e`. Runtime now records all-world legality after the
> full-team profile is applied and before observation; the paired reducer
> checks exact policy/category/seed coverage, finite outcomes, and binds the
> per-world legality evidence into its coverage digest. Scoped contract checks
> passed and an independent exact-head review approved the final implementation
> commit. No Phase 4 simulator evaluation, replay, or GPU run was performed;
> qualification remains false. The dashboard provenance ledger now contains
> 60 scoped tranches.

> **Plan A signed expert-choice authority — 2026-10-25.** The reviewed code path
> is integrated at `54de5352f` through `c118f0db0`. A protocol-pinned Ed25519
> key authorizes complete-action preferences; the fresh replay loader binds the
> sole fitted selected-versus-min-route action pair to that signature and
> rejects relabeled targets or unsupported U-only goal preferences. The focused
> CPU suite reported 19 passed using mocked replay and a small test model. No
> real annotations, production simulator replay, GPU fit, or warm-start
> artifact exists. This is code/contract coverage, not evidence of human
> expertise or simulator performance. The three-pass ledger now has 60 scoped
> tranches.

> **Continual evaluation schedule — 2026-10-25.** The reviewed implementation
> is integrated at `de03ce27a` with the fresh-root limitation correction at
> `89d0e86db`. Future manifests can pin an ordered diagnostic/validation case
> schedule to an exact protocol ID and digest; selection rotates deterministically
> by completed five-track cycle, and evaluation history hashes all five
> checkpoint snapshots. Existing manifests keep the prior behavior. This is not
> enabled in the active r4 manifest, whose evaluations remain fixed to one
> diagnostic `pickup_simple/cluster` seed. Campaign restart in place is
> unsupported. Twenty-five focused contract tests passed; no simulator, GPU, or
> replay was run. The three-pass ledger now has 61 scoped tranches.

> **Three-pass dashboard re-verification — 2026-10-25 01:55 UTC.** After adding
> the reviewed Plan A exact-heap tranche, `frc-defense-dashboard.service` was
> restarted on current `main` (PID `2810214`). `/`, `/api/meco-status`,
> `/api/curriculum-status`, `/assets/meco.js`, and `/assets/dashboard.js` all
> returned HTTP 200. The curriculum API validated 58 scoped summaries. Served
> frontend assets match the checkout (`meco.js` SHA-256
> `6178a524853b0223d4c6975c9a11b7dc7dc54131e41b383374a98a5e9a62533c`;
> `dashboard.js` SHA-256
> `487dff5cfb48d4d7e6d419be343d5e1149c291b4ba9866d227fa312972867219`). No
> simulator replay was generated or reviewed.

> **Plan A exact-heap prototype — 2026-10-25.** The scoped three-pass tranche
> was integrated at `eae7c8ebfea8dce971845b95597ee95ca90a157d`. It extracts the
> legacy Python Dijkstra as an oracle, caches exact edge weights for each
> static graph, and adds an explicitly gated HIP binary-heap implementation.
> The reviewer approved the exact commit as a prototype. Eight focused CPU
> contract tests, Python compilation, and diff checks passed. The kernel has
> not been compiled or run on WX9100; dispatch remains off pending exact output
> and support-digest parity plus profiling. This is not CPU-free execution and
> no simulator evaluation/replay was performed.

> **r4 cycle/evaluation configuration audit — 2026-10-25 01:00 UTC.** The
> service is active in cycle 1 with A1 at 15/100, A2 at 13/100, and A3/C1/C2
> queued; no evaluation file exists yet. The manifest-mode supervisor uses
> `cycles=0`, meaning it repeats until stopped, and matches all five tracks at
> 100 updates per scheduled cycle across two exclusive WX9100 lanes. It queues
> evaluation only after all five checkpoints reach the matched cycle frontier.
> Each automatic evaluation is one diagnostic `pickup_simple/cluster` episode
> on the same seed `61720261025`; it is not a held-out result, and the manifest
> has no runtime evaluation protocol binding. Only one pending evaluation is
> retained, so a slow evaluator may supersede an older pending cycle. Phase 4
> adversarial training exposure is disabled in this manifest; ordinary
> curriculum assignments can still vary between cycles. No warm-start
> artifacts are supplied. This confirms that five-track training is running
> continuously, but the current campaign alone does not satisfy the plan's
> held-out, adversarial, or qualification gates.

> **Dashboard direct-root verification — 2026-10-25 00:58 UTC.** The
> `frc-defense-dashboard.service` was restarted against this `main` checkout at
> 00:57:14 UTC after integrating the reviewed direct-campaign-root resolver.
> The page, `/api/meco-status`, `/api/curriculum-status`, and `/assets/meco.js`
> all returned HTTP 200. The MECO API now resolves r4's direct `checkpoints/`
> and `evaluations/` roots and reports a fresh queue: A1 10 updates, A2 9,
> A3/C1/C2 queued. Curriculum provenance validates as 56 scoped summaries.
> The served `meco.js` SHA-256 matches this checkout:
> `6178a524853b0223d4c6975c9a11b7dc7dc54131e41b383374a98a5e9a62533c`.
> A follow-up campaign read at 00:57:52 UTC showed A1 11/100 and A2 10/100;
> no complete five-track chunk, cycle, or evaluation had finished. Dashboard
> projection is verified current, but the live run remains early diagnostic
> training rather than held-out or qualification evidence.

> **Live throughput check — 2026-10-25 00:57 UTC.** The r4 service remains
> active (PID `2750253`; started 00:46:34 UTC). Both WX9100 workers are alive.
> From update timestamps, A1 is completing about 90–100 optimizer updates/hour
> and A2 about 83–90/hour; their current counts are 10/100 and 9/100. A3, C1, and C2 are
> queued. At this early rate, a matched five-track 100-update cycle is roughly
> 2.5–3.5 hours before evaluation overhead; C-track rates and later-cycle
> throughput are unknown, so this is not an ETA for the full plan. Both GPUs
> report 98–99% utilization, while workers also consume CPU for simulator,
> planning, and control work. This is mixed CPU/GPU execution, not CPU-free.
> No five-track chunk, cycle, or evaluation has yet completed. The dashboard
> direct-campaign status projection remains under repair.

> **R4 launch snapshot — 2026-10-25 00:51 UTC.** The failed
> `meco-five-continuous-20261024-r3` root remains archived and untouched. Its
> supervisor failed at 00:36:32 UTC because the worker run record omitted
> top-level `track`; A1 had 100 optimizer updates, A2 had 93, and the other
> tracks had 0. The scoped worker-manifest fix was independently approved at
> `a4cc6a45ef2b3053bcc70af08ae265801bb74ed4`.
>
> A fresh, source-bound campaign `meco-five-continuous-20261025-r4` was
> launched under user service `meco-five-continuous-20261025.service` (PID
> `2750253`, started 00:46:34 UTC) from reviewed source commit `9bf7d8d55`.
> Its manifest validates against that commit and assigns the two verified
> WX9100 devices. At this snapshot, cycle 1 targeted 100 optimizer updates per
> track: A1 was 1/100 on `cuda:0`, A2 was 1/100 on `cuda:1`, and A3/C1/C2 were
> queued at 0. No training chunk, cycle, or automatic diagnostic evaluation
> was complete yet.
> The run has no warm-start artifacts, uses the diagnostic split, and is not
> qualification evidence.
>
> At this snapshot, the dashboard service had been restarted on the current checkout at 00:47:51 UTC;
> `/api/meco-status`, `/api/curriculum-status`, and `/assets/meco.js` return
> HTTP 200, the served JavaScript hash matches the checkout, and the
> three-pass ledger validated 55 scoped summaries. The status projection then
> assumed the legacy `checkpoints/meco-five` layout; its direct-root correction
> was completed and verified later. GPU utilization was 94% and 98% at
> 00:51:11 UTC.

> **Plan C repeated-label implementation — 2026-10-25.** The isolated
> implementation based on source `9bf7d8d55e75821b67eb1c5d1c01871f666ea85d`
> was fast-forwarded into `main` at `8cb3177026461669a8f095502430f8616d4582cb`.
> Its reviewed code and test commits are `875b4df9b3ec7e0ceecac27a6c30100aa921bd8e`
> and `5815e48ee91ce3aab932666b60781190a928940a`.
> It adds two common-random-number repeats for each archived Plan C candidate
> branch, with repeat seed derivation and protocol identity recorded in the
> dataset. C1/C2 pair labels require the same strict preference sign in both
> matched repeats; mixed-sign or tied evidence is retained and excluded from
> fitting. The implementation has synthetic contract coverage only. The active
> r4 workers remain pinned to their original source and do not load this change.
> No label
> collection, GPU/training, simulator behavior evaluation, replay, or dashboard
> currentness check was run; this tranche establishes no variance calibration,
> winner, or qualification result. See
> [`plan-c-repeated-rollout-labels.md`](implementation-provenance/plan-c-repeated-rollout-labels.md).

> The dashboard provenance manifest now contains 57 validated scoped
> summaries and its pinned digest matches the checkout. The dashboard restarted
> from current `main` at 01:35 UTC (PID `2796113`); at 01:40 UTC, `/`,
> `/api/meco-status`, `/api/curriculum-status`, `/assets/meco.js`, and
> `/assets/dashboard.js` returned HTTP 200. The curriculum API reported
> `validated_summary_only` with 57 tranches. Both served JavaScript SHA-256
> values match `main`. No simulator replay was generated or reviewed in this
> dashboard verification.
>
> **R4 training snapshot — 2026-10-25 01:40 UTC.** The supervisor heartbeat
> was fresh and reported cycle 1, zero completed 100-update chunks, zero
> completed cycles, and no evaluation. A1 was 68/100 on `cuda:0`, A2 was
> 61/100 on `cuda:1`, and A3/C1/C2 were queued. The run remains diagnostic,
> without warm-start artifacts; these are optimizer updates, not evaluation
> passes or qualification evidence.

> The dashboard provenance manifest now contains 57 validated scoped
> summaries and its pinned digest matches the checkout. The service restarted
> from current `main` at 01:35 UTC, but its new process is blocked in uninterruptible
> ZFS buffer I/O and port 8765 is not listening. Live three-pass API and frontend
> asset verification are therefore pending; the prior service process was not
> treated as current after the manifest change.

> **Runtime identity correction — 2026-10-24 23:25 UTC.** The first relaunch
> passed manifest validation and created its isolated campaign root, but both
> workers exited before checkpoint writes: ROCm hipify reformatted the tracked
> generated `tensor_collision_multi_hip_kernel.hip`, changing the source
> identity during rollout. The scoped three-pass correction is integrated at
> `2cf6c2b63e7ad444566cca1a3c148ce451d97f52`; it excludes that generated file
> while retaining its maintained `.cu` and `.cpp` inputs. The failed campaign
> root is retained with zero completed updates. A fresh campaign root and
> source-bound manifest are required for the next attempt. No training is
> currently running; qualification and all document-wide evidence gates remain
> open.

> **Launch correction — 2026-10-24 23:24 UTC.** A new isolated five-track
> diagnostic campaign was attempted using two WX9100 devices, but the
> supervisor exited before creating its campaign root or starting GPU work.
> The cause was manifest launch wiring: a diagnostic split was passed to the
> supervisor together with a bound evaluation protocol, which requires the
> validation split. The scoped planner/implementer/reviewer fix is integrated
> at `6d60294cd3d1741b0c685b15fb2e71e3d14efa0b`; it retains manifest protocol
> and seed validation while leaving the runtime diagnostic evaluation
> unbound. The dashboard three-pass ledger includes this scoped approval.
> The external manifest must be regenerated against the final clean checkout
> before relaunch. At this snapshot the training service is failed/stopped;
> there are no new training updates or qualification results.

> **Snapshot — 2026-10-24 23:16 UTC.** The implementation candidate was
> fast-forwarded into `main` at `08b93828032ca6561552358142bb6b1781bc5071`;
> evidence, dashboard-pin, and status updates are committed through
> `3425eddc6f5b4db98369ab3f1bfc4a247e2d278b`. The previously untracked GPU
> inventory was preserved and verified byte-identical to the integrated file.
> Two startup/replay defects were fixed and reviewed: deferred initial
> observation, then D0 replay cadence when no learned-policy cadence exists.
> The focused replay module passed 33 tests, plus compilation and diff checks.
>
> A fresh representative GPU diagnostic completed on this checkout: D0,
> `pickup_simple/cluster`, seed `61720261024`, `cuda:0` (AMD Radeon Pro
> WX 9100). The scenario completed after six acquisitions at 1.16 simulated
> seconds. Replay `scenario-meco-105c38cbec61cc109461c3e5` has 59 frames; its
> initial frame marks action audits unavailable and the next frame contains
> action-derived values. The report and replay are under
> `evaluations/meco-five/`; the replay reports source commit `3425eddc6`, no
> dirty files, and `outdated=false`. The dashboard runs from this `main`
> checkout; status, curriculum, replay, page, and asset endpoints return HTTP
> 200, and the API validates 51 provenance tranches.
>
> This runtime check is diagnostic evidence only. The five-track supervisor
> remains stopped; the prior run's A1/A2/A3/C1/C2 counts (525/526/500/500/500)
> are not qualification results. The checked-in qualification protocol is
> still provisional v1; full D0 and matched D1/D2 calibration, reviewed v4
> operational margins, held-out five-track comparisons, Phase 4 outcome
> evidence, Hailo profiling, and a qualification decision remain outstanding.

## Prior status snapshot — 2026-10-24 21:51 UTC

This was the state before the final integration and diagnostic replay below.

## Phase 4 action-7 remask taxonomy and dashboard curriculum data root — 2026-10-24

The action-7 diagnostic taxonomy correction is integrated at
`4d040f5e2238e652e90c745300394e07aa1a575f`; the independent reviewer approved
its exact implementation head `accce779be3d1a9e87092e22ac45ae7a6543d01d`.
Nine focused tests and static checks passed. This corrects fallback-only
records being labeled as remasked; it does not show action 7 was a learned
choice. The dashboard curriculum external-root fix is integrated at
`9597d16e33d18d6dea5ea4f2ee6d6e0715191428`; the exact implementation was
independently reviewed and 16 focused tests plus static checks passed. After
integration, the dashboard service was observed active on this candidate and
its relevant status/API and frontend asset endpoints returned HTTP 200 with
candidate-matching backend/assets. The external root points at live main
campaign data; the generic curriculum endpoint reports zero runs, while
five-track status is separate.

Summary-only provenance tranches #38 and #39 are recorded in
[`MECO_THREE_PASS_PROVENANCE.json`](MECO_THREE_PASS_PROVENANCE.json); the
manifest and dashboard digest pin are updated together. Neither tranche has
simulator/replay evidence or establishes policy qualification. No GPU,
training, or replay work was performed for these integrations.

> **Historical integrated-candidate and training status — 2026-10-24 21:51 UTC.**
> The integrated implementation candidate is
> `codex/meco-full-plan-final-integration` at
> `68997963a76226c1f1dad0a1021d3a4432b98696`, based directly on current
> `main` (`20665a5136eac72138ba1d0759d96bcf0ee68a5f`). The candidate includes
> the reviewed dashboard live-curriculum lane projection and qualification-v4
> workflow gate. Its combined scoped suite passed 128 tests; this is contract
> evidence, not simulator qualification. The manifest follow-up records 49
> scoped tranches. The dashboard has not yet been restarted against this
> candidate or checked for current assets/API provenance.
>
> The five-track supervisor is stopped (`checkpoints/meco-five/STOP`,
> timestamp 2026-10-24 21:48:55 UTC; terminal supervisor heartbeat
> 21:51:36 UTC). Its last recorded totals are A1 525, A2 526, A3 500, C1 500,
> and C2 500 optimizer updates. Cycle 6 was interrupted; only A1/A2 had
> advanced into it. The evaluation split was `diagnostic`, with no bound
> evaluation protocol, and no warm-start artifact. The latest completed
> diagnostic evaluation was cycle 5 (22 records). Curriculum exposure covered
> only `pickup_simple` (five cycles: four `cluster`, one `single`); the other
> planned scenario families had zero exposure. These counts do not establish
> 100/100 evaluation cases, pass status, or a qualified winner.
>
> The five-track supervisor has not yet been restarted on the isolated
> source-bound campaign; the previous supervisor remains stopped. Qualification-v4
> code and workflow admission are implemented and reviewed,
> but the checked-in protocol is still provisional v1. The only calibration
> artifact is D0 `pickup_simple`; there is no v4 protocol, reviewed operational
> margin set, matched held-out result, learned-policy confidence interval, or
> qualification decision. No fresh five-track simulator replay has been
> generated or reviewed. The dashboard currentness check and representative
> seeded simulator evaluation remain outstanding. This is the current
> implementation candidate state, not a completion claim.

## Dashboard currentness after C1/C2 campaign integration — 2026-10-24

After candidate commit `7d16d49dfe57c7f426ec2c28714a6d038692f7a9`,
`frc-defense-dashboard.service` restarted with PID `2072539` and this
candidate worktree as its working directory. `/meco`, `/api/meco-status`,
`/api/curriculum-status`, and `/assets/meco.js` returned HTTP 200. The served
JavaScript SHA-256 matched the checkout asset
(`6178a524853b0223d4c6975c9a11b7dc7dc54131e41b383374a98a5e9a62533c`). The
curriculum API validates the three-pass ledger as `validated_summary_only`
with 47 entries and `meco-c1-c2-phase2-campaign` latest. The data root remains
live `main`. No replay was generated; this verifies currentness and provenance
only.

## Dashboard currentness after manifest-supervisor integration — 2026-10-24

After candidate commit `8e8117bd02a609dc5bb4ddb71de2827380759083`, user
service `frc-defense-dashboard.service` restarted with PID `2062204` and this
candidate worktree as its working directory. `/meco`, `/api/meco-status`,
`/api/curriculum-status`, and `/assets/meco.js` returned HTTP 200. The served
JavaScript SHA-256 matched the checkout asset
(`6178a524853b0223d4c6975c9a11b7dc7dc54131e41b383374a98a5e9a62533c`). The
curriculum API validates the three-pass ledger as `validated_summary_only`
with 46 entries and `meco-supervisor-manifest-device-lease` latest. The
dashboard data root remains the live main checkout. No replay was generated;
this verifies dashboard currentness and provenance only.

## C1/C2 matched-prefix Phase 2 campaign executor — 2026-10-24

Reviewed descriptor/executor commit `82a1e8146aec2ad8d2172c0786d10a7e7f41d895`
is integrated at candidate commit `4d80247942a3453651def4546a2735184cd5092b`.
The new offline descriptor binds a clean source snapshot, exact protocol,
train-only seeds, fixed collection axes, indexed device, and isolated paths.
Explicit execution acquires the shared lease, collects one matched-prefix
dataset, fits C1 then C2 on that exact device/dataset, and validates both
artifacts and hashes. The campaign executor is implemented but has not been
executed because the live two-GPU training run remains active. Twelve focused
offline tests and compilation passed; independent review approved the exact
corrected files and protocol-assigned seed examples.

This covers only C1/C2 pairwise gross-return warm starts. It does not generate
Phase 2 artifacts or qualify tracks; Phase 2 readiness and qualification remain
false. See tranche #47 and
[`phase2-c1-c2-campaign.md`](implementation-provenance/phase2-c1-c2-campaign.md).

## Manifest-driven supervisor and cross-process device leases — 2026-10-24

Reviewed implementation commit `5d81713310e9400b12fead981f118913f83222f0`
is integrated at candidate commit `55e988596`. The supervisor now consumes a
validated five-track manifest, derives configurations and isolated roots from
it, rejects CLI overrides, and requires explicit diagnostic-launch opt-in.
Training and evaluation devices use stable sorted cross-process leases shared
with the validation campaign runner. Fresh campaign and evaluation roots are
claimed exclusively after lease acquisition; a concurrent-root race fails
closed and removes only an empty partial claim. Thirty-five focused offline
tests, compilation, and diff checks passed; independent exact-snapshot review
approved after requiring the race fix.

This is launch wiring, not Phase 2 completion. No dataset or warm-start
artifacts were created, no simulator/GPU/training/replay was run, and the live
legacy supervisor was not changed or retroactively leased. Phase 2 and
qualification remain false. See tranche #46 in
[`MECO_THREE_PASS_PROVENANCE.json`](MECO_THREE_PASS_PROVENANCE.json) and
[`meco-supervisor-manifest-device-lease.md`](implementation-provenance/meco-supervisor-manifest-device-lease.md).

## Candidate dashboard verification — 2026-10-24 14:33 UTC

After the provenance ledger and digest pin were integrated, user service
`frc-defense-dashboard.service` restarted with PID `2047704` from this
candidate worktree. `/meco`, `/api/meco-status`, `/api/curriculum-status`, and
`/assets/meco.js` returned HTTP 200. The served MECO JavaScript SHA-256 matched
the checkout (`6178a524853b0223d4c6975c9a11b7dc7dc54131e41b383374a98a5e9a62533c`).
The curriculum API validated the three-pass ledger as `validated_summary_only`
with 45 entries and `phase4-context-paired-dispatch` latest. The dashboard data
root remains the live main checkout. No replay was generated; this verifies
the current dashboard stack and provenance summary only.

## Phase 4 focal-pose and teammate-behavior paired dispatch — 2026-10-24

The validation runner now dispatches the `observation.focal_pose_offset` and
`teammate.behavior_pool` factors through one matrix invocation per exact
protocol cell. Each matrix contains the no-factor and treatment conditions,
with a shared protocol case, validation seed list, five-checkpoint hashes,
source identity, and execution device. The runner verifies condition-level
assignment digests and exact policy/category/seed rows, and rejects missing,
duplicate, or unexpected rows. The matrix CLI exposes the deterministic
teammate-pool seed used to bind a design-axis behavior to a reproducible pool
rotation. Output is write-once: a pre-existing output path is rejected, and
this dispatcher does not resume or overwrite prior evidence. The matrix runner
accepts these complete paired conditions without also evaluating the
architecture baseline; it emits no outcome contrasts for the pair-only matrix.
Each evaluation report must resolve to a regular file within the fresh case
directory and match the byte hash recorded by its matrix condition entry.

This is dispatch and provenance wiring only. Focal-pose recovery/safety and
teammate-behavior/collision/score outcomes remain null; qualification remains
false. Five focused fake-runner and malformed-row contracts passed. No
simulator evaluation, training, replay, GPU run, or dashboard action was
performed. See the paired dispatch contracts in
[`test_meco_phase4_campaign_plan.py`](../tests/test_meco_phase4_campaign_plan.py).

## Offline five-track campaign manifest — 2026-10-24

Reviewed implementation commit `bc7eb512512a134272d84b96ba11c8767d67a596`
adds a source/config/protocol-bound manifest API and CLI for A1, A2, A3, C1,
and C2. It binds the verified two-WX9100 inventory, campaign paths, scheduler,
and warm-start bytes. The manifest is planning metadata only: Phase 2 coverage,
qualification eligibility, and launch readiness remain false until validated
dataset/replay evidence, trusted calibration binding, and supervisor support
exist. Thirteen offline metadata contracts passed. No simulator, training,
GPU, replay, dashboard verification, or qualification was run. See
[`five-track-campaign-manifest.md`](implementation-provenance/five-track-campaign-manifest.md)
and provenance tranche #44.

## Phase 4 legal full-team start profiles — 2026-10-24

Reviewed implementation commit `28f906961115adffb11a3e7e11e4188e4660dcd9`
connects registry-backed `start.legal_team_poses` assignments to protocol-bound
paired validation dispatch and curriculum resets. Paired arms and per-policy
records are checked against the same checkpoint hashes, source identity, and
execution device. Curriculum construction defers its first perception until
the profile assignment is attached; reset validates six-robot pose clearance
and applies the assigned start state before final observation. Five focused CPU
contracts passed, followed by a separate rerun of the strengthened
reset-order contract (one passed); Python compilation and `git diff --check`
passed. Provenance tranche #37 pins the scoped summary at manifest SHA-256
`ed3781ae5307811989f156938fc594341ca5bab854245bd415d5d069038ee4e2`.

The contracts are reviewed, but simulator behavior and legal-start validation
rate remain unverified. `outcome_by_start_state` remains null/unavailable and
qualification eligibility remains false. No simulator, GPU/HIP, training,
campaign, replay, or dashboard-service validation/restart was performed for
this tranche.
See [`phase4-legal-team-start-profiles.md`](implementation-provenance/phase4-legal-team-start-profiles.md)
and tranche #37 in [`MECO_THREE_PASS_PROVENANCE.json`](MECO_THREE_PASS_PROVENANCE.json).

## Phase 4 reactive opponent validation wiring — 2026-10-24

The helper-only deterministic opponent pool is now connected to a protocol-
bound validation schedule and evaluation runner in commit
`49c8ba7d15c507c1208600daf3b3fa3b4312a15e`. The paired dispatcher selects only
`architecture_matched_cv_k16` in both arms. Two independent reviewers approved
the exact pre-commit source HEAD `983183665fc32fc4f8d6d501a0258ebb7e25a6a0`.
Nineteen focused CPU contract tests passed; Python compilation and
`git diff --check` passed. This is summary-only implementation evidence.

No simulator evaluation or replay was run, so actual policy application and
game behavior remain unverified. The configured pressure slots are slot 0 at
levels 1–4 and slots 0/1 at level 5; slot 2 is out of scope. Family-specific
outcomes remain null, and qualification eligibility is false. No GPU, training,
or dashboard service action was performed. See
[`phase4-reactive-opponent-validation-wiring.md`](implementation-provenance/phase4-reactive-opponent-validation-wiring.md)
and tranche #36 in [`MECO_THREE_PASS_PROVENANCE.json`](MECO_THREE_PASS_PROVENANCE.json).

> **Plan A rollout U contract integrated — 2026-10-24.** Reviewed commit
> `88b0bb9270ad912fb0cc651c33990e57c111b9b0` versions the legal-node U
> estimand, binds each sampled-node target to its exact paired rollout
> evidence, validates complete uncensored feasible-behavior support, and
> records U semantics in fitted artifact metadata while preserving P=P0.
> Seventeen focused offline audit tests, Python compilation, and diff checks
> passed. No labels, dataset, or weights were generated; no simulator, replay,
> GPU training, or qualification was run. The bounded sampled-node return
> target is not global optimality. Three-pass tranche #35 is summary-only;
> dashboard provenance pin updated, restart/endpoint verification pending.

> **Dashboard verified on reviewed Phase 4 candidate — 2026-10-24 09:52 UTC.**
> User unit `frc-defense-dashboard.service` restarted with PID `1745889`; its
> working directory is this candidate checkout and its data root remains live
> `main`. `/meco`, `/api/meco-status`, `/api/curriculum-status`, and
> `/assets/meco.js` returned HTTP 200. Served JavaScript SHA-256 equals the
> checkout asset (`6178a524853b0223d4c6975c9a11b7dc7dc54131e41b383374a98a5e9a62533c`).
> The curriculum API reports `validated_summary_only` with 25 provenance
> entries; latest is `phase4-context-runtime`. It still reports
> `Could not read saved curriculum status.` No replay was generated because
> the Phase 4 APIs remain unwired to the evaluator.

> **Candidate advanced after dashboard verification — 2026-10-24 10:00 UTC.**
> The policy-pose boundary is now integrated and the provenance manifest has
> 26 entries with a new digest. Dashboard PID `1745889` has not yet been
> restarted against this newer candidate, so the prior 09:52 verification is
> historical and the loaded backend is not current for the policy-pose change.
> Restart and reverify before any replay.

> **Phase 4 contract and runtime APIs reviewed — 2026-10-24.** The contract
> scope correction `4033acc8a2a67f0d206247363e6c98ca21ce7a97`, opponent API
> `8a31ae2acaba9b0cb5ab842373f2796c271e65bc`, and context API
> `19737bd51429ae41346a3aa7e4435d00d6d943bb` are integrated through candidate
> `f2867715c08e71fd6365b8ddb5ba742e62109fba` and passed scoped independent
> review. The overlay has 11 requirements and 466 exact validation axes, with
> trench interception limited to 14 trench-route categories. Runtime modules
> define seven opponent policies, team starts, teammate actions, contact
> geometry, and pose-offset transforms, but they are not wired into
> `CurriculumEnv` or campaign execution. Contact and recovery values remain
> proxies/unavailable; pose no-leak remains explicitly incomplete.
> **Policy-pose boundary approved and integrated — 2026-10-24.** Exact follow-up
> `7f8c567c0a2a712235a1e464c06b2b3adb1f86b1` passed independent review and is
> integrated at `8fe3c9daf411a7914e0ee6bcf7f1fc89ae27b729`. The perception
> encoder now consumes the policy pose and delayed observation; truth-variance
> tests cover policy geometry and delayed frames. No simulator replay was run,
> and the pose-offset stimulus is not wired. Provenance now covers 26 scoped
> tranches; its dashboard pin is updated, but the service needs another restart
> after this checkout change before replay.

> **Phase 4 campaign plan compiler integrated — 2026-10-24.** The candidate
> now includes the selective validation-only protocol reader and compact plan
> compiler. It validates split metadata without decoding final-test rows or
> disclosing their seeds, and describes 6,488 design cells / 10,380,800
> theoretical paired episode rows. This is a plan only: runtime stimuli,
> dispatch, outcome aggregation, simulator runs, replay inspection, and
> qualification remain outstanding. The three-pass manifest had 27 entries at this historical point; runtime capture raised it to 28 and the pair-contact tranche below raises it to 29.
> Dashboard currentness is
> historical until restarted against the completed candidate state.

> **Phase 4 realization capture integrated — 2026-10-24.** Candidate commits
> `af6930800` and `e07016509` integrate reviewer-approved capture contract
> `f9e53044d01688069dea54e2ea2d5438b76fafc8`. The matrix now rejects
> unrealized or mismatched captures and validates source, factor, split,
> episode seed handling, qualification case, and event evidence. Four existing
> runtime stimuli can be captured; all eleven registry requirements remain
> non-executable for qualification because required outcome signals are not
> authoritative/available or their stimuli are not wired. The focused review
> suite passed 33 tests. No simulator, GPU, training, or replay was run. The
> dashboard has not been restarted against this candidate state.

> **Simulator pair-contact summary source integrated — 2026-10-24.** Candidate
> commits `0d1e5eabf` and `62b4a00dd` add pair-specific contact summaries from
> the Torch and fused HIP robot collision paths. The record identifies slots
> (2,3), counts distinct contact substeps separately from solver reports,
> timestamps completed substeps, and sums normal/tangential/net impulse. It is
> explicitly a bounded summary, not a full event stream. Contact geometry is
> not yet scheduled by the validation matrix; recovery and route-completion
> signals remain unavailable. Seventeen contract tests passed; no simulator,
> HIP compile, GPU behavior test, or replay ran. Dashboard currentness remains
> historical. The summary is recorded in the three-pass manifest, now at 29
> entries with pinned SHA-256 `f9282381fabb8ff53c66e98b996ad626981d808a7e30550156a896c9f0e7ef82`;
> the dashboard service has not been restarted against this candidate update.

> **Diagnostic contact geometry path integrated — 2026-10-24.** Candidate
> commit `1d4e4eb15` wires three registered contact geometries from matrix
> conditions through reset and evaluation. Slot 2 receives the sensed-target
> approach while slot 3's selected policy action is preserved. Episode rows
> separate assigned exposure/action ticks, solver-confirmed pair (2,3)
> contact, and route-crossing state/time. The matrix requires all 11 expected
> policy rows across the exact category/seed grid and rejects missing,
> unexpected, duplicate, or empty rows. Recovery and route-completion outcomes
> remain unavailable; contact conditions are diagnostic-only, qualification
> remains false, and final-test use is blocked. Eleven focused contract tests,
> Python compilation, and diff checks passed. No simulator, GPU/HIP, campaign,
> replay, or dashboard validation ran. The current checkout has not been
> verified by the serving dashboard at implementation review. After the
> provenance integration commit, the service was restarted at 10:48 UTC with
> this candidate worktree, the Three-Pass API validated all 30 summary entries,
> `/meco`, both relevant APIs, and `/assets/meco.js` returned HTTP 200, and the
> served JS hash matched the checkout. No replay was generated.

> **Dashboard verified on candidate — 2026-10-24 09:22 UTC.** After the 22nd
> three-pass tranche was added, `frc-defense-dashboard.service` was restarted
> and is active as PID `1713013`, with this candidate worktree as its working
> directory and live `main` as the training-data root. `/meco`,
> `/api/meco-status`, `/api/curriculum-status`, and `/assets/meco.js` returned
> HTTP 200. The served MECO JavaScript SHA-256 matches the local asset
> (`6178a524853b0223d4c6975c9a11b7dc7dc54131e41b383374a98a5e9a62533c`); the
> curriculum API validated the provenance manifest as `validated_summary_only`
> with 22 entries and `validation-campaign-runner` latest. The curriculum API
> also reports `Could not read saved curriculum status`; its summary/evaluation
> data remains unavailable from the live data root. No replay was generated.

> **Resumable validation campaign runner integrated — 2026-10-24.** Reviewed
> runner `6c1d5eb04acb39b192f1b491f753ca68d338096c` now executes and seals
> the exact 28 validation-only jobs with protocol/source/checkpoint/config and
> report/replay bindings, safe resume, fail-closed partial outputs, and
> scenario-separated summaries. Ten focused planner/runner tests, compile, and
> diff checks passed; execution was mocked. No campaign, simulator, GPU,
> dashboard, replay, or final-test operation was run. Three-pass provenance
> now contains 22 scoped tranches. Dashboard restart/endpoint verification is
> pending; no replay is present.

> **Plan A offline oracle audit integrated — 2026-10-24.** Reviewed audit
> `18faf13c42bb5e5d5168a0b24c4f48726ca14593` now validates source/protocol
> bindings, recomputes stored U/B evidence and sampling weights, reproduces
> the collector's deterministic node sample, and rejects feasible nodes
> omitted from the full pair layout. Thirteen focused tests, compile, and diff
> checks passed. No Plan A dataset, expert labels, simulator replay, fitting,
> GPU, or training evidence was generated. Provenance now contains 21 scoped
> tranches. Dashboard restart/endpoint verification is pending; no replay is
> present.

> **D0 offline schedule compiler integrated — 2026-10-24.** Reviewed commit
> `b4dbaf4cc75e76fa5dc62471e96eda0373d8961b` adds an offline `describe`
> command that reports the default full calibration schedule as 12 scenarios,
> 466 case strata, 96 shards, and 139,800 records without GPU discovery or
> seed disclosure. Run and merge continue to require execution-bound manifests
> and reject descriptive-only output. Ten focused tests, compile, CLI count,
> and diff checks passed. No simulator, GPU, dashboard, training, or D0 shard
> was run. The three-pass manifest contains 20 scoped tranches; dashboard
> restart and endpoint verification after this change are pending. No replay
> is present, and the full calibration remains unrun.

> **Validation campaign planner integrated — 2026-10-24 09:02 UTC.** The
> reviewed planner is in candidate code commit
> `9cba0a9af8312de28c39a3e2a842364f185165e0` and produces 28 deterministic,
> validation-only scenario/stratum jobs from the exact frozen protocol case
> table. It binds exact seeds and expected policy/category/seed coverage,
> refuses qualification claims against the current provisional protocol, and
> does not serialize final-test seeds. Six focused tests, compilation, a CLI
> dry run, and diff checks passed; independent review rejected a missing-
> stratum case before approving the complete case-table validation. No matrix,
> simulator, GPU, or final-test operation was run. Three-pass provenance is
> now 19 scoped tranches. After integration, `frc-defense-dashboard.service`
> was restarted and is active as PID 1690905 from this candidate worktree.
> `/meco`, `/api/meco-status`, `/api/curriculum-status`, and `/assets/meco.js`
> returned HTTP 200; the served JavaScript SHA-256 matches the checkout
> (`6178a524853b0223d4c6975c9a11b7dc7dc54131e41b383374a98a5e9a62533c`). The
> curriculum API returned `validated_summary_only` with 19 entries and
> `validation-campaign-planner` latest. The dashboard reads training data from
> live main; no replay was generated.

> **Matched-evaluation controls integrated — 2026-10-24 08:49 UTC.** The
> planner → implementer → independent reviewer tranche is integrated into
> candidate HEAD `2e610f88d284680f71857a00e0d8f05d0d00fe87` (code integration
> `049c81e076bba74897590d66621592553bd7ddf8`; implementer fix
> `0db1e6436fbd753d82e49fbab02850d81fce1f07`;
> initial feature commit `415ce63d457f076c112a25b648ad6f58e404ffde`). It adds
> protocol-bound exact seed manifests, condition/policy replay selection,
> paired runtime/fallback diagnostics, report digest validation, and a
> protocol-scoped single-use final-test ledger that blocks both supervisor
> launches and direct `train_meco()` fresh/resume/warm-start calls. The matrix
> CLI now requires an indexed accelerator and defaults to `cuda:0`. Three-pass
> provenance is updated to 18 scoped tranche summaries; see
> [`MECO_THREE_PASS_PROVENANCE.json`](MECO_THREE_PASS_PROVENANCE.json) and
> [`implementation-provenance/matched-evaluation-controls.md`](implementation-provenance/matched-evaluation-controls.md).
> Focused verification reported 78 tests, Python compilation, and diff checks
> passing; the independent reviewer rechecked 10 matched-control and 9
> supervisor-dispatch tests. Provenance integration passed 14 tests. After the
> commit, `frc-defense-dashboard.service` was restarted; user-systemd reports
> it active as PID 1676539 with this candidate worktree as its working
> directory. `/meco`, `/api/meco-status`, `/api/curriculum-status`, and
> `/assets/meco.js` returned HTTP 200; the served MECO JavaScript SHA-256
> matches the candidate asset (`6178a524853b0223d4c6975c9a11b7dc7dc54131e41b383374a98a5e9a62533c`).
> The curriculum API reports `validated_summary_only`, 18 tranches, with
> `matched-evaluation-controls` latest. The dashboard reads training data from
> live main. No simulator, GPU, or live training operation was run for this
> tranche, and no replay was generated. The live main
> supervisor remains a separate checkout at source identity
> `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`; at the
> latest poll cycle 2 is training with no error, A2/A3 running on the two
> WX9100 devices and A1/C1/C2 queued at 100 updates. These are updates, not
> qualification results. No fresh replay, held-out matrix, qualification, or
> deployment evidence is claimed.

> **Dashboard promotion capture — 2026-10-24 06:41:03 UTC.** Snapshot
> [`MECO_DASHBOARD_PROMOTION_SNAPSHOT_20261024T064103Z.json`](MECO_DASHBOARD_PROMOTION_SNAPSHOT_20261024T064103Z.json),
> SHA-256 `9bee3debbd535f973cd79d430d23737f12ab8d34b8b55a39b86b6140a71f8660`.
> PID 1537750 serves the dashboard from the candidate checkout, branch
> `codex/meco-production-input-parity-integration` at
> `e449eb1c2edbdd510d133b63d271c4cb6ccf2996`; its command binds both
> `--run-dir` and `--meco-data-root` to live `main` at
> `84a894961af833f580dad8316a69af02492ff0b7`. All four documented endpoints
> returned HTTP 200, and the served JavaScript matched the candidate-local
> asset. The API reports `validated_summary_only` for the three-pass manifest,
> with 10 tranches and `phase5-production-input-onnx-parity` latest. This is a
> candidate dashboard reading main training data; it does not mean candidate
> source has been integrated into main. At capture, main's supervisor was
> training with no reported error; A1/A2/C1 were at 100/100 and A3 at 74/100
> (its worker heartbeat read 75 during the non-atomic capture). The current
> supervisor lane reported C2 queued at 0/100 while `C2/run.json` still
> described the earlier run ID as `aborted_due_to_sibling_failure`; those are
> recorded as separate states. Evaluation remained diagnostic and unbound to a
> held-out protocol. Reads were non-atomic. No tests, simulator evaluation,
> replay generation, or training changes were made for this capture.

> **Live status refresh — 2026-10-24 06:36:58 UTC.** Snapshot
> [`MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T063658Z.json`](MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T063658Z.json),
> SHA-256 `f0cd1e4509859d73fa0d93972a024cecdd2d00870bf844e53035e4b9025cb16c`.
> The active supervisor is healthy at cycle 1 with no reported error and source
> identity `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`.
> A1, A2, and C1 are at 100/100 updates; A3 is running at 70/100 on `cuda:0`;
> the current supervisor lane reports C2 queued on `cuda:0` at 0/100. The C2
> `run.json` still describes a different, earlier run ID in
> `aborted_due_to_sibling_failure`; it has no error string. This earlier
> manifest status is not the current queued lane and is retained as a distinct
> observation. The evaluation split remains diagnostic with no evaluation or
> held-out protocol binding. Dashboard PID 1403399 owns port 8765 from this
> checkout; `/meco`, both status APIs, and `/assets/meco.js` returned HTTP 200.
> The served JavaScript SHA-256 matches the local asset. Runtime source files
> match the supervisor-bound source identity, with no source-file mismatches.
> Reads were non-atomic; snapshot file pins record the observed runtime state.
> No tests or simulator evaluation were run for this refresh.

> **Current campaign update — 2026-10-24 04:49:02 UTC.** A fresh continuous
> run was launched from `main` at `2ecc067b780a693088ec0992df70560806485628`; current HEAD
> `45ec06796` follows documentation and calibration-manifest commits only,
> source identity `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`.
> Supervisor `5e8f956474594bf8b69417b39b4692fa` writes to this checkout's
> `checkpoints/meco-five` and `evaluations/meco-five`, uses `cuda:0,cuda:1`,
> evaluation device `cuda:1`, `--cycles 0`, and protocol
> `dec8964454769b01fa4d79fe5f1cb7919013d94b5076e3a99a20ba5a4181f6f2`.
> At the recorded heartbeat, A1 and A2 were alive at 30/100 and 27/100 updates
> on the two WX9100s; A3, C1, and C2 were queued; no chunk or evaluation had
> completed. D0 shard indices 0 and 1 each completed 150 entries on `cuda:1`;
> index 2 had 139/150 entries on `cuda:0`, and index 3 had just been launched
> on `cuda:1`. The two-shard result is partial calibration progress, not a
> merged calibration report or qualification.
> This confirms launch only, not completed optimizer progress, evaluation, or
> qualification. The evaluation split is diagnostic, so this run does not by
> itself produce held-out qualification evidence. At capture, the dashboard
> process on port 8765 served this checkout's data root; `/meco`, the status
> API, and both JavaScript assets returned HTTP 200, and served asset hashes
> matched the checkout. The named systemd unit was absent, so process ownership
> and listeners were verified directly. The point-in-time evidence is
> [`MECO_CAMPAIGN_SNAPSHOT_20261024T0448Z.json`](MECO_CAMPAIGN_SNAPSHOT_20261024T0448Z.json),
> SHA-256 `02247f858a2276750cf2c4352f703de33b8827f96b719d87847b08a6a10bbe78`.
> The source identity is recomputed from both live workers' `run.json` source
> manifests and matches the supervisor record; the native HIP extension's
> initialization state changes the result of a later fresh identity call.
> The earlier campaign's subsequent C2 update-20 failure and terminal state
> are hash-pinned in this snapshot. Runtime JSON and process handles remain
> authoritative for later status.

> **D0 calibration milestone — 2026-10-24.** The merged `pickup_simple` D0
> calibration report is `qualifications/meco/d0-calibration-pickup_simple.json`
> (SHA-256 `eaf39d5df985c854b157f4abf9fc48a66517cd4864d4486006d0681dda7f76a6`,
> calibration ID `1be58a36b643f3ed48fc661e0d7454007b8f0f5fb2eb5f9a9bf380c9c4c48030`);
> its bundle is `qualifications/meco/d0-calibration-pickup_simple-bundle.json`
> (SHA-256 `0ba1f570c352d7167ab94c4f84f22706865c6e5da38afdc77a160a2c85087b5a`).
> Scoped planner, implementer, and reviewer evidence covers eight 150-entry shards,
> exact manifest/artifact digests, source identity
> `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`, normalized
> HIP/WX9100 runtime, and 1,200 records across four strata (100 unique seeds ×
> repetitions 0, 1, 2), with required safety metrics and source binding verified.
> This is D0 baseline calibration only; it does not qualify a learned policy.
> Independent D0 calibration coverage and variance audit: [`implementation-provenance/d0-pickup-simple-calibration-audit.md`](implementation-provenance/d0-pickup-simple-calibration-audit.md).
> This milestone does not record live training progress; consult the active
> campaign runtime for current status. Target v2 still needs explicit
> operational margins and review; learned-policy held-out report, CI, and primary
> gate are absent. No margin approval or held-out result is asserted.
>
> **Dashboard/training point-in-time capture — 2026-10-24 05:19:49 UTC.** Snapshot
> [`MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T051949Z.json`](MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T051949Z.json),
> SHA-256 `81991d6d429bcb7300e5ccffe967d51e9ecae5f22f01a9fc29190df717566006`. It records the serving checkout, dashboard/API responses and asset
> equality, plus two active training tracks and three queued. Reads were not atomic;
> this is historical evidence, and current status must be read from runtime.

> **Live status refresh — 2026-10-24 06:02:43 UTC.** Snapshot
> [`MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T060243Z.json`](MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T060243Z.json),
> SHA-256 `c41fbdca73b18c77d09c71c6a2709a0a4f466cd7e2d23461dff8c7c04fdaa51b`.
> The supervisor is healthy on `main` at source identity
> `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`:
> A1 and A2 completed 100/100 updates, A3 is 29/100, C1 is 41/100, and C2
> remains queued. A3 and C1 are confirmed on separate WX9100 GPUs. All four
> dashboard endpoints returned HTTP 200 and the served frontend asset matches
> this checkout. The split is diagnostic and has no held-out protocol binding;
> these are training updates, not qualification results. Reads were
> non-atomic; the snapshot pins each captured runtime file.

> **Live status refresh — 2026-10-24 06:23:13 UTC.** Snapshot
> [`MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T062313Z.json`](MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T062313Z.json),
> SHA-256 `f4f2b4e30faa315333ca1d1783834a50bfb6fb0f6b5283eb186854506ab54b82`.
> The pinned supervisor is still training from `main` at source identity
> `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`:
> A1 and A2 completed 100/100 updates, A3 is 54/100, C1 is 85/100, and C2 is
> queued. The active run reports two completed training chunks and no error.
> Dashboard page, MECO API, curriculum API, and JavaScript asset all returned
> HTTP 200; the served JavaScript hash matches this checkout. Evaluation remains
> diagnostic with no bound held-out protocol, so update counts are not scores
> or passes. Reads were non-atomic; the snapshot pins the captured runtime
> files and response bodies.

> **Training interruption and recovery — 2026-10-24 06:26:48 UTC.** Snapshot
> [`MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T062648Z.json`](MECO_DASHBOARD_TRAINING_SNAPSHOT_20261024T062648Z.json),
> SHA-256 `09ec0c981ceffec1800bf6142f842021857d82708270ac96da1ad5631dfe77b9`.
> A merge command was accidentally run in the live checkout while preparing the
> isolated integration candidate. The supervisor's source guard stopped A3 and
> C1 at their last saved checkpoints (56 and 88 updates) and aborted queued C2;
> A1/A2 remained complete at 100. I reverted the source change and confirmed the
> saved checkpoint file hashes match the restored source files and protocol. The
> same continuous two-device supervisor configuration was relaunched as PID
> `1523650`; at capture it had resumed A3 at 56 and C1 at 88, with C2 queued and
> no supervisor error. This recovery is recorded explicitly; the run was not
> uninterrupted. Dashboard endpoints returned HTTP 200 and the served asset
> matched the checkout. Reads were non-atomic; see pinned files and response
> digests in the snapshot.

> **Hardware availability check — 2026-10-24 06:32 UTC.** `lspci` reports the
> two AMD/ATI Vega 10 XT WX9100 devices used for training. Neither `hailortcli`
> nor a `/dev/hailo*` device is available on this host, so Hailo compilation,
> accelerator execution, latency, and memory evidence remain unavailable here.
> At the 06:33 supervisor heartbeat the campaign was healthy: A1/A2/C1 were at
> 100 updates, A3 at 65, and C2 queued behind A3 on `cuda:0`. The paired
> diagnostic evaluation had not started; these updates are not qualification.

> **Phase 5 production-input parity tranche — 2026-10-24.** Isolated branch
> `codex/meco-production-input-parity`, commit
> `b098d725b7c2c2da90ddd6b228989bfc27d949ac`, passed scoped reviewer re-review.
> It adds opt-in bounded production-request capture and binds the exact captured
> tensors, checkpoint/protocol/source provenance, ONNX graph, and parity report;
> capture loading preflights size limits and validates request metadata. No tests,
> simulator, or model workloads were run for this tranche. Integration candidate
> `codex/meco-production-input-parity-integration` at `eebd4ae57b9f226dade8bac99a6734596381759e`
> merges the approved branch into main `4d93b2f97`; a second scoped review
> approved the clean merge. It is ready to fast-forward after training is manually
> stopped and the final worker/checkpoint source-binding audit passes. The
> representative seeded simulator capture/replay and dashboard restart/currentness
> check remain outstanding.

## Phase 0 scoring outcome status

The integrated ledger reconstructs its limited simulated outcome from versioned accepted FUEL score events. It is not official match scoring: Tower awards, foul penalties, disqualification, ranking points, end-game bonus, and advancement points remain excluded, while official match points remain null. A source- and rules-bound human estimand approval artifact is required before collection; none is currently available. The simulator scorer-path fixture remains skipped pending a seeded end-to-end evaluation and fresh dashboard-currentness evidence. Runtime/device evidence, qualification, and Phase 0 completion remain open.

**Historical analysis snapshot:** 2026-10-24 03:53 UTC
**Plan:** [`MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`](MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md)
**Plan SHA-256:** `718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`
**Current implementation checkout at the historical analysis snapshot:** `main` at `2ecc067b780a693088ec0992df70560806485628`
**Prior implementation candidate at the historical snapshot:** `codex/meco-full-plan-integration` at `0e133780a0fadfb9015b84719e75291be89370e0`
**Prior campaign checkout:** `codex/meco-c1-replay-integrated` at `d7b8dd4b1c1814c2a368ded7604b3c887571fdcc`

This ledger separates code that exists from evidence that demonstrates runtime behavior. A passing contract test, completed optimizer update, or review of a scoped patch does not establish a simulator result, qualification, or deployment readiness.

## Previous campaign (historical)

The earlier continuous campaign was isolated in another checkout and failed at C2 update 20. The snapshot below records its state at heartbeat `2026-10-24T03:53:45Z`; A1, A2, A3, and C1 completed one 100-update chunk, while C2 failed before completing that chunk. An immutable compact observation is recorded in [`MECO_CAMPAIGN_SNAPSHOT_20261024T035345Z.json`](MECO_CAMPAIGN_SNAPSHOT_20261024T035345Z.json), SHA-256 `b13df6632b64ba5cf8f872c7c5b8fec45141762e9445ed49265ee7aed17ad524`. It is a historical point-in-time record, not a live status feed.

At the status heartbeat `2026-10-24T03:53:45Z`:

| Track | Device | Training updates | State |
|---|---|---:|---|
| A1 | `cuda:0` | 100/100 | First chunk complete |
| A2 | `cuda:1` | 100/100 | First chunk complete |
| A3 | `cuda:0` | 100/100 | First chunk complete |
| C1 | `cuda:1` | 100/100 | First chunk complete |
| C2 | `cuda:0` | 1/100 | Running |

At that heartbeat, the supervisor reported `completed_cycles: 0`, `completed_training_chunks: 4`, and `qualification_protocol_id: null`. No held-out evaluation had completed. These update counts are optimizer progress, not a 100-case evaluation or a pass. The campaign had no warm-start artifacts (`warm_start_artifact_sha256: {}`); it began from scratch. At that historical snapshot, training used source identity `c56cc4001441e10e72b191025a97141033907aed40f42c0f3b61f7bacab3e8e2`, while the separate candidate then had source identity `8f3d68c00e2beb52d0d188feaf40785bd79a91f04c5ac4c652a05379aa3d045a`.

The old campaign's checkpoints do not satisfy strict source identity for the current checkout and must not be used for current-source evaluation or promotion. No current learned-policy evaluation or replay ID is available; `20261024` is a training seed, not evidence of an evaluation. Dashboard code and assets are now verified current for the active run, but no fresh learned-policy replay has yet been generated or inspected.

## Completion ledger by plan section

| Section | Implemented | Evidence still required |
|---|---|---|
| **1. Decision and hypothesis** | The repository represents three Plan A spatial models and two Plan C candidate rankers, with deterministic comparisons and a shared safety boundary. | No matched, held-out result establishes that a learned architecture beats D0 or the matched candidate baseline. No winner or promotion is supported. |
| **2. Five models** | A1 U-Net+GRU, A2 structured SE(2) U-Net+regional attention+GRU, A3 hierarchical SE(2) GNN+regional attention+GRU, C1 candidate MLP+GRU, and C2 map encoder/candidate cross-attention+GRU are implemented. Parameter inventory and adapter shape checks exist. The inventory reports 171,717 parameters for A1, 122,311 for A2, 118,215 for A3, 49,026 for C1, and 124,162 for C2. | Counts and synthetic/adapter checks do not establish checkpoint-specific exported graphs, memory, Hailo compatibility, latency, or simulator performance. The current five-track frontier is incomplete. |
| **3. Deterministic and predictive controls** | D0/D1/D2 evaluation paths and prediction-mode controls exist. The schema-5 paired matrix analyzer binds exact seed manifests, reports, source, checkpoints, condition/policy replay selection, and can calculate paired outcome and runtime/fallback diagnostic intervals. Schema-4 report reading remains supported. | No current matrix report or interval exists for the requested architecture, control, prediction, penalty, candidate-recall, and fallback comparisons. D3 is not established. |
| **4. Shared observation and planner contract** | Shared raster/scalar inputs, candidate masks/order, stale-output fallback, legal action support, Plan A penalty provenance, evaluation-to-replay record binding, and generation-based replay publication are implemented. The async broker has deterministic fake-adapter contract coverage, including replacement, freshness, faults, and close; raised inference exceptions no longer kill its worker. No-route replay validation checks complete contiguous action coverage. The matrix can select replay seeds by condition and policy; latency and fallback diagnostics are tagged as non-qualification metrics. | No fresh five-track learned-policy replay has been generated and inspected. Broker contract tests do not establish GPU-load freshness or simulator fallback. Replay selection and timing instrumentation do not prove P-shaped routes, robot behavior, collection/scoring, fallback behavior, or representative end-to-end safety. Global planning remains XY AD* with swept SE(2) checks, not full global SE(2). |
| **5. Training strategy** | Phase 0 protocol/configuration and a limited replay-reconstructable MECO FUEL score-ledger contract, Phase 1 interfaces, stochastic PPO support, Phase 4 train-split exposure scheduling, and the resumable two-device supervisor exist. Phase 2 collectors/loaders/fitters exist. A human-authored Plan A goal annotation materializer packages explicit sidecar labels with source/legal/protocol/authority hashes. The legal-node rollout U contract now defines a bounded, replay-bound matched gross-return estimand, strict repeat evidence digest, explicit censoring/completeness, and fitted-artifact semantics; P stays at P0 absent independent P labels. | No Plan A rollout dataset, human-authored annotations, or warm-start artifact is available; the new U contract has not generated labels or weights. No approved score estimand artifact, seeded score evaluation, or inspected score replay exists; official points remain null and qualification false. The current campaign does not consume warm starts. Baseline variance calibration, a matched evaluation frontier, Phase 4 factor outcome runs, and candidate-source training remain outstanding. |
| **6. Metrics and decision gates** | Metrics, qualification protocol versions, confidence intervals, calibration, promotion interfaces, paired compute/fallback diagnostics, and a protocol-scoped single-use final-test ledger exist. New qualification claims are restricted to validated v4 gate status; legacy v1–v3 are diagnostic/read-only. Plan A PPO penalty semantics use one versioned contract shared by training protocol hashing, reports, and dashboard validation. | The active r4 supervisor remains diagnostic with no evaluation-protocol binding and only `pickup_simple` exposure. D0 calibration exists only for `pickup_simple`; the checked-in protocol remains provisional v1. There is no v4 protocol artifact, reviewed operational margin set, learned-policy held-out report, match-outcome confidence interval, or primary success-gate result. No model is qualified. |
| **7. A/B experiments** | Matrix runs record paired policy rows and bind inputs to schema-5 reports; the analyzer enforces exact seed/category/stratum and policy completeness and calculates paired bootstrap intervals. Schema-4 reports remain readable. Phase 4 factor completeness is checked. | The analyzer has no experiment results to summarize. The listed A/B experiments, K=8/16/32 candidate recall, P=0 ablation, matched prediction and inference-disable comparisons have not been completed as a current results bundle. |
| **8. Implementation boundaries and order** | Candidate generation, model adapters, controls, evaluation, replay capture, provenance, deployment interfaces, and the three-pass dashboard are present in the current checkout. | A frozen baseline/fixture evidence package, complete matched five-track bundle, representative replay, qualification, and promotion evidence remain absent. |
| **9. Unresolved design decisions** | The code exposes candidate K, prediction mode, graph support, penalty mode, profiling, and replay/provenance fields needed to measure several open questions. | Legal graph/bin adequacy, candidate recall, route evaluation cost, prediction calibration/occlusion, held-out generalization, Hailo support, latency, and simulator shortcut resistance remain measurement questions. No result should be inferred from current code or synthetic tests. |

The annotation materializer publishes staged dataset and manifest files with rollback for caught staging/replacement/fsync failures and rejects input/output path aliases. Its two-file rename sequence is not crash-atomic; the loader rejects a mixed pair when the dataset hash differs, and still validates the other provenance fields. No annotation data has been materialized.

## Three-pass evidence and validation

The dashboard's structured evidence validator is at `docs/MECO_THREE_PASS_PROVENANCE.json`. It validates hashes, paths, role records, commit formats, reviewer verdicts, and per-tranche limits, and the UI separates workflow pass status from agent liveness. The integrated candidate at `68997963a76226c1f1dad0a1021d3a4432b98696` carries 49 selected tranches and remains explicitly classified as `scoped_transcript_summary`. Some entries summarize parent approval messages because original planner/reviewer artifacts were not persisted. This is not whole-plan approval, original-transcript evidence, runtime validation, or proof that any agent is currently active. The latest dashboard-lane and qualification-v4 entries are summary-only and explicitly retain their runtime/evaluation limits.

On the integrated candidate at `0e133780a`, the focused suites for curriculum and MECO dashboards, replay integrity, async inference broker, transition history, evaluation binding, deployment contracts, Phase 4 exposure, penalty provenance, paired CI analysis, three-pass provenance, and Plan A annotation materialization passed **156 tests and 5 mutation subtests**. Static Python compilation, JavaScript syntax checks, protocol/provenance JSON parsing, and `git diff --check` passed. The curriculum dashboard fixtures now use the active curriculum version (`2`) instead of a stale hard-coded version (`1`). These are unit/contract checks; no test in this validation run exercised end-to-end simulated game behavior, so no simulator evaluation was run.

**Simulator validation:** not run for this candidate. Scenario, seed, and replay identifier: none.
**Dashboard currentness:** the current checkout's backend, frontend assets, and live data root were verified at 04:25 UTC as recorded in the pinned campaign snapshot; no fresh five-track learned-policy replay has been generated or reviewed.
**Qualification/deployment:** no qualified model, Hailo result, or deployment claim.

## Next evidence gates

1. Collect a source-bound Plan A legal-node rollout dataset under the new U contract, validate its replay evidence, then fit and validate a warm-start artifact. Human-authored annotations remain a separate explicit source for feasible behavior preferences; do not infer them from rollout returns.
2. Continue the active current-checkout campaign and verify all five checkpoint manifests bind to one source identity. The earlier `d7b8dd4b1` campaign failed and remains historical.
3. Complete baseline variance calibration and freeze the matched held-out protocol.
4. Run the full paired five-track/control/ablation matrix and summarize paired intervals. Generate and inspect a fresh seeded simulator replay; restart and verify the dashboard again after any code changes before replay review.
5. Run the Phase 4 adversarial/generalization scenarios and Phase 5 export, Hailo, memory, and end-to-end latency measurements where the actual toolchain/device is available.
6. Apply the predeclared qualification and promotion gates only after the required outcome and safety evidence exists.

## Review-only combined integration candidate

This branch selectively combines the reviewed production-input capture/ONNX parity and dashboard queue reconciliation source with the approved Plan A preference-v2 and A/C input-provenance features. It retains the base checkout's 2026-10-24 dashboard training snapshot and D0 calibration audit. The three-pass manifest remains summary-only and now includes the scoped parity, queue-balancing, queue-reconciliation, preference-v2, and input-provenance tranches; the new preference/provenance entries explicitly state that their original planner/reviewer messages were not persisted verbatim.

This is an isolated review candidate, not a change to live `main`. Its MECO source identity differs from the active training source due to the integrated Python changes. No attempt was made to resume active checkpoints from this candidate. Synthetic ONNX Runtime graph parity for all five randomly initialized model constructors has now passed on CPU; no real production capture or trained-checkpoint parity, simulator evaluation, GPU training, Hailo/device qualification, or deployment qualification is claimed.

The focused Plan A preference-v2, model-input provenance, request-capture, dashboard queue projection, deployment contract, three-pass provenance, and inference broker suites passed **58 tests** at that review point. The separate ONNX Runtime parity follow-up adds 20 focused synthetic tests. Python compilation for changed runtime and script modules and `git diff --check` passed. These checks do not exercise the simulator, GPU production inference, a real request capture, trained-checkpoint or production-capture ONNX parity, Hailo, or device deployment.

## Plan A HIP parity/profile harness planner pass

Planner artifact `docs/plans/plan-a-hip-parity-profile-harness-20261025.md`
was authored against `main` base `48df1c7d03d72b2645e36070f34f91bf9a4b92ba`
and independently approved at exact planner commit
`8dbbfee696b74a5fc4e0b071ccc364dd8db84571`. Its SHA-256 is
`0880e0880899cd737a9ac8f6c4fb14320f75f7abf24164cb53e3f9fb0e7a5505`.
The plan covers a CPU-only, source-bound offline fixture mode and a separate
explicit WX9100 parity/profile run after an exclusive lease handoff. This is
planner approval only: no harness implementation, test, HIP compilation or
execution, parity/profile result, simulator, training, replay, or dashboard
validation is claimed. Normal support/training dispatch remains CPU-only and
`PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED` remains false.


## Plan A HIP parity/profile harness implementation pass

Implementation candidate `ca4604e019826cb6b2a010f84872d0eb15f83d16` on
`codex/plan-a-hip-parity-profile-harness-impl-20261025` was independently
approved at that exact head. It adds a canonical source-bound offline fixture
manifest, a 64,000-node × 10-slot 8-heading pose-lattice recipe, and a distinct
explicit WX9100 candidate mode. Device mode acquires the repository exclusive
device lock, performs post-lease physical WX9100 attestation before loading the
HIP extension, and reports the generated lease identity and attestation. It
requires a clean checkout and checks exact pinned-source and fixture identity
at entry and before returning a passing result. The normal CPU oracle dispatch
and false `PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED` constant are unchanged.

Static Python AST parsing and `git diff --check` passed. Fixture generation
executed the CPU oracle to freeze expected digests; the offline CLI was not run,
and no tests, HIP import/compile/execution, device or lease inspection/acquisition,
simulator, training, replay, dashboard, or deployment run occurred. The exact
head was approved by the independent reviewer, who confirmed lock acquisition,
post-lease attestation ordering, source/fixture revalidation, and the closed
production gate. Device parity, support-digest comparison, profiling, and any
dispatch decision remain deferred until the orchestrator confirms continuous
training has released both exclusive WX9100 leases and schedules verification.

## Plan C route-device provenance checkpoint E — candidate pending second review

The Plan C route-device source/code candidate C (`cdeb9391bf23e5c2460527959132008aee6f8eb1`) has a separate approved exact-head code review. Ledger candidate D (`8d64233d4573600321956c74391842e7623f18f7`) was independently approved for ledger correctness; E records that D approval separately from C's code review. E remains a finalizer candidate awaiting its second exact-head review and integration. No E SHA or second-review verdict is recorded in the digest-pinned ledger.

This status records code and ledger review only. It does not establish Plan C simulator behavior, a fresh replay, GPU/HIP parity or profiling, training, qualification, or whole-system GPU-only execution. See the [Plan C ledger candidate summary](implementation-provenance/plan-c-device-resident-selection-ledger-candidate-20261025.md), [D ledger review summary](implementation-provenance/plan-c-device-resident-selection-ledger-review-20261025.md), and [pending dashboard currentness note](implementation-provenance/plan-c-device-resident-selection-currentness-20261025.md). Dashboard service, API, and frontend asset currentness remain unverified; no service action was performed for E.
