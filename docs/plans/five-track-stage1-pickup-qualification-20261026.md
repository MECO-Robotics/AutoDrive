# Pass 1 plan: five-track Stage 1 pickup qualification

**Status:** Planning artifact only.
**Base:** `origin/main` `d30cdd89b4d404d545254423e9d5bc950804e346`.
**Source plan:** `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`.
**Scope:** Gate and prepare one bounded Stage 1 (`pickup_simple`) outcome campaign for A1, A2, A3, C1 and C2 after the uninterrupted live campaign ends. Phase 0 freeze/calibration/margin decisions are mandatory prerequisites, not excluded work. This is not whole-plan qualification or a promotion decision.

## Why this is the next gate

The original plan requires all five architectures to be compared on matched scenarios and outcomes. The current repository has implementations and evaluation machinery, but its status evidence does not establish a current-source, matched Stage 1 result for each track. The active campaign is diagnostic, does not bind a qualification protocol, and cannot be upgraded after the fact.

The latest status says: “all five stage-one qualifications remain outstanding.” Recent approved source work closes code contracts—Plan A device route/fixed-capacity work, GPU physics, five-track PPO/reset/pressure state, and score-event buffering—but none creates a Stage 1 outcome bundle. The score-buffer implementation commit `82009ecdd7b827024928f2ded0ce56ba9889c863` is on `origin/codex/gpu-score-event-buffer-impl-20261026`, not an ancestor of current main. The approved PPO/reset candidate is separately on `origin/codex/five-track-pressure-snapshot-v2-pass2-20261026` at `54bd1dc4651b039e2924e602c2ddf13a165536af`. Any integration/source frontier change invalidates old checkpoint-source compatibility unless an explicit reviewed continuity contract permits reuse.

Existing machinery already covers training campaign launch, validation reports, paired seeds, curriculum outcomes, and qualification protocol intake. This plan adds no new evaluator, report format, protocol version, score definition, or threshold. It specifies the first matched evidence run that can close the curriculum's Stage 1 gate once its prerequisites exist.

## Mandatory Phase 0 and calibrated-protocol gate

The checked-in `qualifications/meco/protocol.json` is `meco-qualification-v1`; it is provisional/diagnostic and cannot authorize this Stage 1 qualification. Do not pass it to the campaign and describe its results as qualified. Existing one-scenario `pickup_simple` calibration does not close the complete source-bound prerequisites.

Before Stage 1 qualification, complete the approved Phase 0 path in `docs/plans/phase0-baseline-heldout-calibration-20261025.md` and `docs/plans/phase0-effective-freeze-partition-materialization-20261025.md`, using the exact post-integration source frontier:

1. Produce and verify the effective source/runtime freeze and disjoint train/calibration/validation/final-test partition, covering all five model tracks. Bind exact scoring, safety, scenario, physics, configuration and native extension identities. Do not read or split private live-run data.
2. Obtain the immutable human-authored `meco-simulated-score-estimand-review-v1` approval bound to source and scoring-rule digests. Keep official match points null unless the approved event model actually supports them; the existing FUEL-only ledger is not official scoring.
3. On calibration tuples only, complete the target's required D0 calibration closure and matched-reference variance calibration against the predeclared D1/D2 control choice. Validate all report bundles, exact seed disjointness and runtime/device identities. A `pickup_simple` diagnostic report alone is insufficient if the target closure requires more cases.
4. After reviewing those calibration artifacts, obtain a human-authored margin decision bound to their digests. It must specify the primary outcome, comparator-selection rule, minimum useful gain and separate robot/field/wall-contact non-inferiority margins before any validation outcomes are inspected.
5. Generate an immutable calibrated target protocol with the current `meco_qualification create-protocol` path. For this promotion gate the exact required version is **`meco-qualification-v5-gpu-bootstrap`**, using implementation ID **`meco-v4-cluster-episode-device-percentile-v1`**. Supply the target's D0 calibration bundle, matched-reference calibration bundle, immutable reference-margin review, and reviewed safety margins; protocol creation must bind their exact digests and current calibration source identity. `qualifications/meco/protocol.json` is `meco-qualification-v1` and cannot be used. Protocol versions v1-v4 are historical/diagnostic only for this gate, even if some older validators can parse them. Before any validation workload, preflight the exact protocol with the current validators and confirm `consume_validation_round` accepts a source-bound validation report under the exact protocol. If the selected runner cannot produce and validate the exact v5 contract, stop and obtain a separate reviewed implementation plan; do not use v1-v4 or weaken the gate.

No production freeze, split, estimand approval, calibration bundle, margin decision, or calibrated protocol is created by this Pass 1 plan.

## Stage 1 outcome contract

Use the registered `pickup_simple` scenario and its exact current scenario version, normal episode budget, robot type, capacity, action cadence, physics/runtime configuration and fixed seed list. Do not substitute a shorter diagnostic horizon or silently combine cluster/multi-zone cases.

For each policy, report the curriculum's actual Stage 1 outcome: distinct FUEL acquired and retained within the episode budget, with release/reacquisition lifecycle rules preventing duplicate credit. Also report useful throughput, completion time, failed collection attempts, and the three safety contact rates. Record FUEL piece count and simulated score ledger as distinct secondary outcomes; do not call them official match points.

Evaluate A1/A2/A3/C1/C2 independently. For promotion evidence, pair each candidate with exactly the one frozen comparator selected during Phase 0 calibration: `D1/current`, `D2/cv`, or `D2/multi_model_v1`. Include D0 separately as the safety floor, not as a second promotion comparator. Other D1/D2 controls may be run on identical seed/scenario rows as diagnostic matched rows, but they do not enter the v5 promotion verdict. Preserve separate rows per track; no aggregate may conceal a missing or failed model. Stage 1 pass thresholds, sample count, confidence rule, and comparator-selection rule must come from the existing versioned curriculum/qualification contract or a reviewed human decision made before evaluation. Do not derive a threshold from observed candidate outcomes or invent one in this plan.

## Execution sequence after manual training stop

1. **Preserve live training now.** Do not stop, restart, reconfigure, migrate, or change its source, checkpoints, output roots, manifests, or leases. This plan authorizes no competing simulator, calibration, training, test, or GPU job.
2. **Freeze the evaluated source frontier.** After the user manually ends the live run and both WX9100 leases are verified released, integrate only exact approved source candidates. Record source tree, runtime/native extension, protocol, scenario registry, physics, and configuration identities. Select checkpoints only when their full source/config identity matches this frontier. Old checkpoints are historical diagnostics if identities differ.
3. **Close prerequisite artifacts.** Require the verified Phase 0 freeze/partition, approved estimand, complete calibration bundles, reviewed margins and exact v5 GPU-bootstrap target protocol described above, plus a source-bound campaign manifest, disjoint validation/test assignment, registered `pickup_simple` scenario bytes, five checkpoint manifests, and accepted stage outcome thresholds/sample budget. Verify the protocol's `calibration_source_identity_sha256` equals the current frozen source identity and that every validation report carries that same source identity and protocol ID. Keep readiness false if any is missing. Do not require Phase 2 warm starts unless the selected protocol explicitly requires them; report warm-start ancestry honestly.
4. **Run the matched Stage 1 campaign.** Use existing MECO validation/matrix runners for all five tracks plus the single preselected D1/D2 promotion comparator and the separate D0 safety floor, with identical scenario, seed rows, physics, observation, action cadence and episode budget. Additional D1/D2 variants are diagnostic only and must be labelled as such. Capture one fresh replay per required policy/seed selection and bind every report/replay to checkpoint, source, protocol, scenario and seed digests.
5. **Review results independently.** Validate exact row completeness, pairedness, lifecycle-aware distinct acquisition, safety metrics, confidence calculation and source/checkpoint provenance. Mark each of the five tracks passed/failed/inconclusive separately. A failure or missing row blocks that track; no ranking or promotion occurs without the predeclared decision rule.
6. **Verify dashboard before replay review.** Restart the serving dashboard from the exact reviewed checkout after code integration. Confirm backend health, `/api/curriculum-status`, `/api/meco-status`, and frontend asset digests match the checkout. Generate fresh replays only after currentness is verified; inspect replay/scenario outcomes and mark old-source replay records outdated.

## Acceptance evidence

- Five separate Stage 1 report rows, one per A1/A2/A3/C1/C2, each paired to the frozen promotion comparator, with D0 safety-floor rows and exact seed completeness. Any additional D1/D2 comparison is explicitly diagnostic and excluded from the promotion verdict.
- Immutable calibrated protocol version `meco-qualification-v5-gpu-bootstrap`, bound to the Phase 0 source freeze, D0/matched-reference bundles, and reviewed margins; its numeric gate verdict uses implementation ID `meco-v4-cluster-episode-device-percentile-v1`.
- For each row: exact source/tree/runtime/checkpoint/protocol/scenario/seed identity, normal episode budget, actual distinct FUEL acquisition/retention outcome, lifecycle counters, useful throughput, failed collection count, safety metrics, and confidence result under a predeclared gate.
- The exact report passes `_validate_v2_protocol`, report/source/protocol/evidence validation, then `consume_validation_round` under the exact v5 protocol; the function's current-source equality check succeeds. Any stale source binding or unsupported version rejects the round and leaves promotion state unchanged.
- Fresh representative simulator records/replays show the changed evaluation path and actual collection outcomes; replay inspection identifies scenario, seed, model/control and source identity.
- Test list and simulator evaluation are reported separately. Any behavior test is accompanied by a representative seeded end-to-end simulation, per project `AGENTS.md`.
- Dashboard service, backend endpoints, frontend asset hashes, and replay provenance/currentness are recorded for the exact reviewed checkout.
- The qualification decision names all five tracks and stays false/inconclusive where a required threshold, approval, result, or safety signal is absent.

## Explicit limits and stop conditions

This gate covers only Stage 1 simple pickup. It does not establish pickup-scattered, multi-zone, transport, shooting, ferrying, full cycles, Phase 4 robustness, Phase 5 deployment, match-outcome superiority, or final qualification under the full MECO evaluation plan.

Stop before execution if the live training has not been manually stopped, leases are not free, the protocol is not exactly `meco-qualification-v5-gpu-bootstrap` with implementation ID `meco-v4-cluster-episode-device-percentile-v1`, `consume_validation_round` rejects the source-bound report, any Phase 0/human decision artifact is absent, seed partitions overlap, any threshold or sample budget is chosen after results are observed, the matched schedule differs across tracks, or dashboard currentness/replay provenance cannot be verified. No runtime/test/simulator action is part of this Pass 1 plan.

## Checkpoint mismatch and safe continuation

After r4 is manually stopped and its WX9100 leases are verified released, compare **every** A1/A2/A3/C1/C2 checkpoint's embedded source identity, code/config/runtime/protocol digests, optimizer/PPO schema, and complete ancestry against the frozen post-integration frontier. Never change a manifest or copy checkpoint bytes to make old values appear current.

- If all five checkpoints pass the exact compatibility validators for the same frozen source/config/protocol and share the expected matched campaign ancestry, resume that campaign only through its supported resume path; retain its exact campaign ID and checkpoint receipts.
- If any track fails compatibility, if the source integration changed identity, or if checkpoint continuity is ambiguous, mark the entire r4 set historical diagnostic ancestry and **do not mix old and new tracks**. Create a new output root and one fresh five-track campaign on the frozen frontier. Initialize all five tracks under the same approved protocol/config and matched update/seed schedule, from scratch or from a complete Phase 2 warm-start set that independently validates against the same frontier. Do not initialize only the mismatched subset while retaining the other four old checkpoints for a claimed matched qualification.
- If a five-track restart cannot be funded or the required warm-start set is incomplete, wait or run only explicitly diagnostic work; do not claim Stage 1 qualification. No checkpoint migration, source relabeling, or live-r4 interruption is authorized by this plan.
