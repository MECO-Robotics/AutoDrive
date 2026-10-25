# MECO learning tracks plan: execution status

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

> **Three-pass dashboard refresh — 2026-10-25 02:13 UTC.** After integrating
> the reviewed Phase 4 start-profile outcome wiring and registering its scoped
> planner/implementer/reviewer evidence, `frc-defense-dashboard.service` was
> restarted on this checkout (PID `2825838`, working directory `AutoDrive`).
> `/`, `/api/meco-status`, `/api/curriculum-status`, `/assets/meco.js`, and
> `/assets/dashboard.js` returned HTTP 200. The curriculum API reports 59
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
> 59 scoped tranches.

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
| **5. Training strategy** | Phase 0 protocol/configuration, Phase 1 interfaces, stochastic PPO support, Phase 4 train-split exposure scheduling, and the resumable two-device supervisor exist. Phase 2 collectors/loaders/fitters exist. A human-authored Plan A goal annotation materializer packages explicit sidecar labels with source/legal/protocol/authority hashes. The legal-node rollout U contract now defines a bounded, replay-bound matched gross-return estimand, strict repeat evidence digest, explicit censoring/completeness, and fitted-artifact semantics; P stays at P0 absent independent P labels. | No Plan A rollout dataset, human-authored annotations, or warm-start artifact is available; the new U contract has not generated labels or weights. The current campaign does not consume warm starts. Baseline variance calibration, a matched evaluation frontier, Phase 4 factor outcome runs, and candidate-source training remain outstanding. |
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
