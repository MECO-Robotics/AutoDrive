# Five-track first curriculum-stage gate plan

**Status:** Pass 1 plan candidate only. No implementation, training, evaluation, simulator, dashboard, or device work is authorized by this artifact.

**Planning base:** `664092b93467fd926d8bb74e8e04f311489000c4`.

## Objective

Define a valid, auditable first-stage curriculum milestone for each of A1, A2, A3, C1, and C2, and identify the smallest prerequisite work needed to make that gate possible. Keep the live r4 diagnostic campaign unchanged and training uninterrupted. Any later qualifying campaign must use the two assigned WX9100 devices, preserve GPU-only execution constraints, and keep evaluation lease-aware so evaluation does not terminate training.

This is a per-policy curriculum milestone. It does not establish the full learning-plan primary claim in `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md` §6 (beating the strongest matched deterministic baseline on held-out match outcomes with confidence intervals and no safety increase).

## Authoritative distinction: exposure versus qualification

The scheduler's `pickup_simple` exposure progression is not a qualification result:

- `frc_defense/meco_task_scheduler.py` defaults to 10 exposure chunks per scenario. It treats a scenario as exposure-complete only after the chunk count is reached and each required category has appeared. This unlocks scheduler progression.
- Scheduler assignments report `promotion_status: "not_evaluated_by_exposure_scheduler"`. Exposure state is scheduling state; it does not establish task success or set persistent qualification state.
- Therefore 10/10 exposure chunks alone are insufficient. The required `single` and `cluster` appearances are also needed for scheduler unlock, and neither condition constitutes a stage pass.
- At the observed r4 status timestamp `2026-10-25T17:42:03Z`, r4 had 5/10 `pickup_simple` chunks (4 cluster, 1 single). Its current assignment was diagnostic `stage_exposure`, with promotion explicitly unevaluated.

The formal first-stage promotion gate is defined by `frc_defense/training_curriculum.py` for `pickup_simple`:

- Evaluate the required categories `single` and `cluster` independently.
- Each validation round needs at least 100 episodes per category and at least 90% success in each category. `pickup_simple` has no additional configured throughput-ratio or absolute useful-throughput threshold in its primitive curriculum definition.
- Require three consecutive passing validation rounds for each policy. Reports must meet the evaluator's split, protocol, checkpoint, category, and binding checks; a failure resets the consecutive-pass streak.
- Track this state separately for A1, A2, A3, C1, and C2. No aggregate five-track result may conceal an individual policy failure.
- When using the current qualification pipeline, also satisfy its calibrated v4 useful-throughput and safety checks against the frozen reference and D0 floor. The primitive curriculum threshold does not waive those qualification checks.

“Pass first stage” therefore means that one named policy has a complete, auditable, three-round formal promotion record under one immutable, source-bound, calibrated v4 protocol, with its required category-level episode counts, success rates, reference-margin and safety results, and checkpoint/report provenance. The five-track milestone is achieved only when all five named policies independently satisfy that definition. Scheduler exposure completion is a prerequisite for progression through the training curriculum, not the qualification evidence itself.

## Current state and blockers

The active campaign is `meco-five-continuous-20261025-r4`, launched as diagnostic work. Its manifest states `qualification_eligible=false`; live status reports diagnostic split and `evaluation_binding=null`. It uses the provisional `meco-qualification-v1` calibration, which has not passed baseline variance review and is nonpromotable. No evaluation protocol is bound. All five warm-start artifacts are absent, and phase-two, validation, and qualification readiness are false. The active scheduler had not yet reached 10 exposure chunks at the evidence timestamp above.

Consequently, r4 training updates and diagnostic evaluations cannot count toward the formal stage-one qualification gate. Do not edit its manifest, checkpoints, scheduler state, process, leases, or output tree; do not relabel its results or attach a qualification binding after the fact. Continue the currently authorized diagnostic training until the user manually stops it. Do not launch a replacement or qualifying run as part of this plan.

The existing `pickup_simple` D0 calibration is tied to an older source identity and is historical D0-only evidence. It cannot be reused as current-source calibration. The repository also lacks a completed current-source freeze/partition capture, reviewed matched-reference variance and margins, calibrated v4 protocol, and complete five-track warm-start/readiness bundle. The detailed preconditions remain governed by `docs/MECO_QUALIFICATION.md`, `docs/plans/phase0-baseline-heldout-calibration-20261025.md`, and `docs/MECO_NEXT_FIVE_TRACK_TRAINING_HANDOFF.md`.

## Bounded execution sequence after Pass 1 approval

Each numbered item is a separate reviewable tranche. No later tranche is implied or authorized by approval of this plan.

1. **Freeze the qualifying source and inputs.** After the user manually stops r4 and its processes and device leases are confirmed released, record the current reviewed source commit, source projection, runtime/build identity, simulator identity, track configs, and immutable artifact digests. Capture disjoint D0 calibration and held-out partitions. Do not migrate r4 checkpoints or reuse its diagnostic reports as qualification evidence.
2. **Complete prerequisite evidence and readiness.** Produce source-bound D0 and matched-reference calibration evidence for the target scenario, complete variance analysis, obtain explicit reviewed useful-throughput and safety margins, and create an immutable calibrated v4 protocol. Satisfy each track's Phase 2 data, production-bundle, and warm-start prerequisites from current source before any non-diagnostic training. Keep the stage-one target scope separate from the full-plan held-out architecture claim.
3. **Implement only missing gate plumbing.** In a separately reviewed code tranche, make per-track qualification progress and final promotion status observable from bound v4 reports. Keep scheduler exposure counters and qualification streaks as distinct state. Validate rejection of diagnostic/v1 reports, missing category coverage, insufficient episodes, failed thresholds, stale or mismatched source/checkpoint/protocol bindings, and interrupted streaks. No code should silently convert exposure into qualification.
4. **Prepare a fresh campaign only after explicit readiness evidence.** Generate a new manifest bound to the frozen source and v4 protocol, with `qualification_eligible=true`, explicit `pickup_simple` single/cluster validation strata and frozen seed sets, unique outputs, and both WX9100 device assignments. Enforce GPU-only routing and fail closed on CPU fallback. Do not modify or resume r4.
5. **Run with non-disruptive evaluation scheduling.** Use the two-device lease-aware supervisor so qualification evaluation holds only its assigned device while training continues on the other; resume deferred lanes when a lease is released. Avoid simultaneous ownership or unbounded queueing on either WX9100. Record every training and evaluation lease, per-track checkpoint, report ID, category counts, and protocol/source identity. The three-round streak is maintained independently per track and only against the same immutable protocol.
6. **Establish the behavior and dashboard evidence.** For each behavior test or qualification run, execute the representative end-to-end simulator evaluation with reproducible seeds and inspect the resulting record/replay. Before replay generation, restart dashboard services on the reviewed checkout and verify service health, API responses, and frontend asset identity. Mark older replay provenance outdated. Retain scenario, seed, replay/report IDs, and outcomes.
7. **Verify the milestone while training remains live.** Confirm that all five policies have three consecutive passing reports, at least 100 episodes per required category per round, at least 90% category success per round, v4 reference useful-throughput CI beyond the reviewed margin, and safety CIs within reviewed margins against both matched reference and D0. Verify distinct checkpoint/report bindings and source/runtime/protocol hashes. Confirm all five lane states are healthy and training continues after qualification; qualification must not terminate the campaign.

## Acceptance evidence

The five-track first-stage milestone is accepted only with:

- Five independently complete promotion records, one each for A1, A2, A3, C1, and C2, each showing three consecutive passing `pickup_simple` validation rounds under the same calibrated v4 protocol.
- For each round and policy: at least 100 evaluated episodes in both `single` and `cluster`, at least 90% success in both categories, complete bound report IDs and checkpoint identities, and passing v4 useful-throughput and safety comparisons under the reviewed margins.
- Source, runtime/build, simulator, protocol, reference, D0, partition, seed-set, and bundle digests matching the frozen manifest and reports.
- No diagnostic or provisional-v1 artifacts counted; no exposure counter used as qualification evidence; no track result substituted for another.
- Simulator records/replays inspected for representative seeded cases, with current dashboard backend and frontend checks recorded.
- Live post-gate evidence that training remains active across all five lanes, all required device leases are respected on the two WX9100s, no CPU fallback occurred, and no lane was starved or stopped by evaluation.

This acceptance is the first curriculum-stage qualification only. The full plan's held-out match-outcome architecture comparison remains a separate milestone.

## Stop conditions and ownership boundaries

Stop before any qualifying run if r4 has not been manually stopped and released, if source or runtime identity is not frozen, if v4 calibration/margins are incomplete, if any track lacks required Phase 2 readiness, if the two WX9100 assignments or GPU-only route cannot be enforced, or if dashboard/replay provenance cannot be verified. Preserve the live diagnostic campaign until the user stops it. Any implementation, manifest creation, evaluation/training run, device interaction, or dashboard restart requires its own approved execution tranche after this plan receives Pass 1 approval.

## Pass 1 review criteria

Review that this plan accurately separates scheduler exposure from qualification; applies the authoritative `pickup_simple` category, episode, success, and consecutive-round rules; preserves v4 reference/safety requirements; accurately describes r4 as diagnostic and ineligible; keeps Phase 2 and full-plan claims distinct; and makes no change to the live campaign or devices. Approval authorizes planning only, not implementation or execution.
