# MECO V4 Phase 0 preregistration candidate (P0a)

**Status:** Candidate for maintainer decisions and independent exact-head review; not approved for calibration, training, qualification, or runtime use.
**Package plan:** `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md` at `72ee8fce1d4a955d100b97d47b58528374faddec`.
**Overall plan:** `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md` at `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`.
**Source reference base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
**Reconciled profile:** SHA-256 `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.
**Reconciliation artifact:** SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`.
**V4 envelope digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.

This is a preregistration proposal. The authorized maintainer must resolve the explicitly open finite-case decisions in §10 before P0a can receive ACCEPT. It does not claim official match points, select a held-out set, set data-derived margins, or establish successful behavior. P0a performs no execution or calibration. P0b creates the final source-bound protocol and Phase 0 artifact only after P1–P6 source freeze/review and the G0 gate.

## 1. Question and comparison

Evaluate all five planned tracks—A1, A2, A3, C1, and C2—against the strongest matched deterministic strategy on held-out simulator outcomes. The deterministic comparison set is D0 (pinned existing policy), D1 (matched candidate strategy), and D2 (D1 with the shared predictive opponent model). Select the strongest comparator using only the preregistered calibration partition, freeze its identity before validation and final testing, and compare every track against that same selected comparator. Do not select a comparator or tune a policy using validation or final-test outcomes.

The principal outcome is accepted active-HUB FUEL awards for the controlled robot/team, counted from the simulator's authoritative accepted-award event. This is a limited simulated proxy, not official match points. P1 must bind the event producer and all consumers to the exact reviewed source; if that map changes the meaning or aggregation, return the protocol to maintainer review before P0b.

The primary product question follows `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`: whether a learned architecture beats the strongest matched deterministic baseline on held-out outcomes, with paired initial states/opponent seeds and bootstrap confidence intervals, without increased safety violations. The estimand is award count per complete 160-strategic-tick episode. Secondary outcomes include award rate over the same fixed horizon, FUEL-per-second, scoring/ferry efficiency, failed collection attempts, defensive exposure/interception, route feasibility, candidate recall, policy switching, reliability/fallback behavior, and end-to-end compute latency. These secondary metrics cannot replace the primary outcome.

## 2. Reward and decision contract

Use the maintainer-confirmed gross-reward contract and current counterfactual defaults:

- strategic reward horizon: 160 strategic ticks;
- macro action duration: at most 12 simulator ticks;
- discount factor: gamma `0.993`;
- gross accepted active-HUB FUEL-award reward; route and risk costs remain separate quantities and are not subtracted from this reward target;
- retain the current counterfactual default reward definitions and signs; P1/P5 must bind their exact implementation/profile identity before source freeze and P0b.

The exact tick-to-time interpretation and reward aggregation at macro boundaries must match the pinned profile and P5 implementation. P0b fails closed on any mismatch. Curriculum/task reward, throughput, score differential, and official match points are not substitutes for this estimand. Route cost and risk are reported separately, so no comparator may quietly mix them into a candidate's gross-reward label while another does not.

## 3. Experimental unit, pairing, and policy set

The experimental unit is one complete simulator episode under one frozen policy, scenario, initial state, opponent assignment, and seed. Each learned track and deterministic comparator receives the same paired scenario/start/opponent/seed cell. Preserve paired identities in raw records and calculate within-pair differences. A cell is never replaced after observing an outcome; failed or incomplete runs are retained with failure status and handled by the preregistered missing-data rules.

The track set is exactly A1/A2/A3/C1/C2. The deterministic candidates are D0/D1/D2 as defined above. Hold simulator rules, physics, sensing, start state, opponent behavior, available observations, action legality, evaluation budgets, and compute allocation fixed wherever the architectures permit. Record unavoidable information or action-space differences rather than describing the inputs as identical. D3 is excluded from this primary protocol unless the maintainer explicitly amends the finite policy set before any candidate results are observed.

## 4. Partition and seed construction

Four disjoint partitions are required: training, calibration, validation, and final test. Their scenario cells and seed lists must be frozen as canonical bytes before training. No seed, initial state, or opponent-policy instance may cross partitions. Hold out complete opponent-policy families and map/start situations from training, not only random seeds. Calibration may estimate baseline variance, uncertainty, and margins and select the strongest deterministic comparator. Validation is for predeclared interim diagnostics only. Final test is opened once after all policies, comparator identity, source, profile, and decision rules are frozen.

For each approved cell, derive a 64-bit episode seed as the first eight bytes (big-endian) of `SHA256(UTF8("MECO-V4-SEED\0") || canonical_json({"partition": P, "scenario_id": S, "pair_id": Q, "replicate": R}))`. Canonical JSON uses UTF-8, sorted object keys, no insignificant whitespace, and integer replicate values. The resulting integer is the simulator seed. Store the canonical preimage, digest, and integer in the protocol. Reject duplicate seed integers across all partitions; resolve collisions by incrementing the replicate field and recording the resulting preimage before any outcomes exist. A single cell seed is shared across all policies in that paired block.

The seed derivation rule is a proposal. The specific partition membership, IDs, counts, and collision-resolution approval remain subject to the maintainer decisions in §10. The seed generator is not evidence that the case matrix has been selected.

## 5. Metrics and safety

### Primary metric

`active_hub_fuel_awards_per_episode`: integer count of accepted active-HUB FUEL award events for the controlled team/robot as agreed in the approved event map. The exact aggregation scope—controlled robot versus whole allied team—must be selected by the maintainer and then implemented consistently. Retain each accepted event's tick, source identity, HUB eligibility state, and stable event identity in raw evidence. Never count attempted scores, inactive-HUB activity, curriculum reward, or unaccepted event pulses.

### Required secondary families

Record at minimum the evaluation-plan categories: game performance; defensive robustness; navigation/safety; strategic decisions; reliability; compute; and generalization. P1/P5 must provide the exact metric IDs, units, aggregation functions, and missingness representation in the source-bound protocol. Do not invent or silently change definitions during calibration.

### Safety non-regression

The primary gate is that the learned track does not increase safety violations against the selected comparator under the approved noninferiority rule. The violation event definitions, aggregation scope, and any critical-event veto must be explicit in the approved case/metric manifest before training. At minimum, the evaluation must account for illegal or infeasible motion, footprint/swept-transition violations, collision/contact measures, capture/mechanism safety constraints, and failed safety fallback where those are observable in the frozen simulator. A higher score cannot offset a safety veto.

No numerical useful-gain threshold or safety margin is asserted here. P0b computes the preregistered margins from calibration-only baseline variance and the accepted method; final-test outcomes cannot tune margins.

## 6. Statistical method and decision rule

For each track, compute paired episode differences against the single calibration-selected comparator, using the approved primary event scope. Report the mean paired difference and a two-sided 95% paired bootstrap confidence interval. Resample paired blocks, stratified by approved scenario/opponent family, with replacement; retain all paired policy rows in a sampled block. The resampling unit, replicate count, interval construction, and family-wise treatment across five track comparisons must be frozen in the accepted protocol before outcomes are produced. The proposed interval is percentile bootstrap; maintainer approval is required before P0a acceptance.

Promote a track for further consideration only if the lower confidence bound exceeds the calibration-derived minimum useful gain and the safety noninferiority gate passes. This is an experimental advancement gate, not production authorization. If the interval is inconclusive, the result is inconclusive; do not call a tie a pass. Report all five track comparisons, failures, missing cells, and multiplicity treatment. D1/D2 diagnostic contrasts may explain comparator strength but do not alter the primary baseline after it is frozen.

The calibration sample floor, baseline-variance estimator, uncertainty method, bootstrap replicate count, minimum useful gain derivation, safety margin derivation, and multiplicity procedure are proposed protocol fields requiring explicit approval in §10. Numeric margins are calculated in P0b and never chosen after candidate results are inspected.

## 7. Calibration method (P0b only)

P0b uses only approved D0/D1/D2 calibration cells and approved repeats. It estimates baseline variance, checks raw event completeness and paired replay consistency, computes the prescribed uncertainty and margins, and selects the strongest deterministic comparator using the primary outcome only. It does not use candidate validation/final-test results, train on calibration outcomes, or change the preregistered estimand or scenario manifest. If the calibration evidence is insufficient under the approved floor, P0b stops for a maintainer decision; it does not silently add samples or revise the method.

Calibration and final evaluation must run on both assigned WX9100 devices under the G0 device-release gate, with algorithmic numerical computation GPU-only and no CPU numerical fallback. CPU orchestration and file I/O are permitted. Retain device UUID/runtime/kernel/residency evidence and fail closed if either device or the approved execution path is unavailable. P0a itself does not inspect devices or run calibration.

## 8. Protocol identity, artifacts, and invalidation

The eventual v5 protocol must bind, by canonical serialization and cryptographic hash:

1. protocol schema/version and target ID;
2. target-to-prerequisite closure for the approved Stage 1 target;
3. exact simulator/rules/source identity and P1's score producer/consumer map;
4. reconciled profile and counterfactual reward contract;
5. policy IDs, comparator-selection rule and selected comparator;
6. finite scenario/case manifest, partition bytes, paired seed preimages and episode limits;
7. metric definitions, safety rules, statistical method, margin method and promotion predicate;
8. device pair identity and execution/residency evidence;
9. raw calibration/evaluation record identities, approvals, timestamps, software versions and result hashes.

Serialize canonical JSON as UTF-8, sorted keys, compact separators, no NaN/Infinity, and explicit schema version. Hash the exact serialized bytes with SHA-256. Keep raw records immutable and give each run a unique ID. A source, rules, profile, metric, case, seed, or device change invalidates the dependent artifact; stale or mismatched identities fail closed. Human decisions and signatures/approval provenance must be recorded separately from agent-generated hashes. Never overwrite prior protocol or calibration artifacts.

Generated P0b artifacts are outside Git at `qualifications/meco/v5/<protocol-id>/phase0-artifact.json`, `protocol.json`, and `calibration/<case-id>/<unique-run-id>/...`. P0a creates none of these files.

## 9. Missing data, failures, and audit record

Keep one row for every scheduled policy/cell, including launch failure, simulator failure, invalid record, safety stop, timeout, or incomplete episode. Bind failure codes to a frozen finite taxonomy in the v5 source implementation. Never replace a failed pair after seeing policy outcomes. The approved statistical plan must say whether a failure is a safety event, an outcome of zero, excluded from the score estimate, or makes the affected paired block non-evaluable; this mapping is a maintainer decision. Publish both the all-scheduled ledger and analyzed paired set with an exclusion reason for each removed pair.

The final audit record includes the exact protocol hash, source/profile/rules/device hashes, policy/checkpoint hashes, scenario and seed, paired cell ID, run/replay identifiers, outcome rows, metric values, failure status, interval/margin computation, approvals, and artifact hashes. Dashboard replays are generated only after G0 and only from the current reviewed checkout; restart the relevant services, verify backend and frontend freshness/endpoints, create a fresh replay, and verify its provenance. A stale replay is marked outdated and is not evidence.

## 10. Maintainer decisions required before P0a ACCEPT

These questions are intentionally open because the approved product document does not name a finite held-out set or fully define all protocol mappings. Record selected values and rationale in a decision record attached to this exact candidate before Pass 3. No implementing agent may guess these values.

| Decision | Required maintainer selection |
|---|---|
| Outcome aggregation scope | Accepted active-HUB FUEL awards for controlled robot or whole allied team; specify event ownership and tie/duplicate treatment. The estimand remains a limited proxy, not official match points. |
| Stage 1 target mapping | Exact v5 target ID, transitive prerequisite closure, and five-track comparison target. |
| Held-out maps and starts | Finite map/layout and initial robot/game-state IDs, including whether any map variations are distinct holdout domains. |
| Held-out opponents | Finite opponent-family IDs and frozen policy/checkpoint/config identities; identify families wholly excluded from training. |
| Case matrix | Finite case IDs and their map, start, opponent family, sensor profile, teammate behavior, and paired block structure. |
| Partition membership | Exact train/calibration/validation/final-test case and seed membership, counts, and immutable canonical bytes. |
| Repeat/sample floor | Minimum paired blocks and calibration episodes per approved stratum; policy for inadequate calibration evidence. |
| Primary uncertainty | Approve percentile paired bootstrap or specify replacement; resampling unit, replicate count, two-sided 95% interval, stratification and five-comparison multiplicity method. |
| Safety metrics and vetoes | Exact metric IDs/scopes and which critical events veto regardless of score. |
| Missing/failure mapping | Finite failure taxonomy and rule for its effect on safety and primary paired estimate. |
| Reward implementation identity | Confirm source/profile binding of current counterfactual defaults and exact aggregation across macros under 160 strategic ticks, max macro 12, gamma .993, gross reward. |
| D3 | Confirm exclusion from the primary comparator set or amend before any candidate outcomes are seen. |

This candidate adopts the already confirmed fixed horizon, macro limit, discount factor, gross-reward semantics, and active-HUB FUEL-award proxy. It does not infer answers for the remaining rows. If any decision is deferred, P0a remains pending and P1–P6 source authoring that depends on the unresolved item must wait.

## 11. P0a/P0b acceptance boundary

P0a Pass 3 reviews this exact document and the maintainer decision record. Acceptance confirms only a preregistered documentation contract and unlocks package source/test-file authoring under each accepted plan. It does not authorize builds, tests, numerical computation, simulator runs, dashboard operations, devices, training, integration, merge, promotion, calibration, or Stage 1 qualification before G0.

P0b starts only after P1–P6 source freeze and independent review, P0a acceptance, G0, both WX9100 leases independently released to the assigned operation, and the reviewed P6 runner. P0b generates immutable source-bound calibration and v5 protocol artifacts. Any relevant source or preregistration change requires re-freeze/review and regeneration before fresh training or qualification.

No tests, builds, simulator evaluation, calibration, dashboard operation, or device/GPU use were performed to prepare this candidate.
