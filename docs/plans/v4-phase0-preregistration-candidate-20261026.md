# MECO V4 Phase 0 preregistration candidate (P0a)

**Status:** Candidate for maintainer decisions and independent exact-head review; not approved for calibration, training, qualification, or runtime use.
**Package plan:** `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md` at `72ee8fce1d4a955d100b97d47b58528374faddec`; Pass 1 ACCEPT record commit `cb02ef8fd` (record SHA-256 `a8c5164730d62bd04c7a134e0d395a81ce8f76c1088db6e5f5a8e2b5bd62701d`).
**Overall plan:** `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md` at `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`.
**Source reference base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
**Current Stage 1 contract evidence:** `frc_defense/meco_qualification.py` at repository `main` commit `d74b0a15e25ddbc6bacfbbb864674371629bb19e`, where `preflight_stage1` requires `promotion_target_scenario == "pickup_simple"`.
**Reconciled profile:** SHA-256 `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.
**Reconciliation artifact:** SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`.
**V4 envelope digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.

This is a preregistration proposal. The authorized maintainer must resolve the explicitly open finite-case decisions in §10 before P0a can receive ACCEPT. It does not claim official match points, select a held-out set, set data-derived margins, or establish successful behavior. P0a performs no execution or calibration. P0b creates the final source-bound protocol and Phase 0 artifact only after P1–P6 source freeze/review and the G0 gate.

## 1. Question and comparison

Evaluate all five planned tracks—A1, A2, A3, C1, and C2—against the strongest matched deterministic strategy on held-out simulator outcomes. The deterministic comparison set is D0 (pinned existing policy), D1 (matched candidate strategy), and D2 (D1 with the shared predictive opponent model). Select the strongest comparator using only the preregistered calibration partition, freeze its identity before validation and final testing, and compare every track against that same selected comparator. Do not select a comparator or tune a policy using validation or final-test outcomes.

The approved Phase 0 research score estimand is accepted active-HUB FUEL awards for the controlled robot/team, counted from the simulator's authoritative accepted-award event. This is a limited simulated proxy, not official match points. P1 must bind the event producer and all consumers to the exact reviewed source; if that map changes the meaning or aggregation, return the protocol to maintainer review before P0b. This score estimand is separate from the PPO return and from the Stage 1 curriculum pass predicate.

The v5 Stage 1 qualification target is `pickup_simple`, matching the current `preflight_stage1` contract; its registered prerequisite closure is exactly `pickup_simple`. The curriculum defines this target by acquiring and retaining required distinct FUEL. Stage 1 readiness remains bound to that scenario's frozen curriculum gate and v5 qualification predicate.

The research comparison follows `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`: compare all five tracks with the strongest matched deterministic baseline on held-out outcomes, using paired states/seeds, confidence intervals, and no safety regression. The maintainer-confirmed limited score estimand remains accepted active-HUB FUEL awards, not official match points. Because `pickup_simple` is a collection/retention scenario and does not require an award, this proxy may be zero or uninformative for Stage 1; P0a must not substitute curriculum reward or treat it as the Stage 1 pass predicate. The broader `full_match_offense` outcome is proposed only as a separately labeled secondary/generalization comparison, with its separate case set and results not affecting Stage 1 readiness.

For the broader secondary comparison, the proposed estimand is accepted active-HUB FUEL award count per complete 160-physics-tick strategic episode. Other diagnostic outcomes include award rate over the same fixed horizon, FUEL-per-second, scoring/ferry efficiency, failed collection attempts, defensive exposure/interception, route feasibility, candidate recall, policy switching, reliability/fallback behavior, and end-to-end compute latency. These do not replace the Stage 1 curriculum predicate or the user-selected award estimand.

## 2. Reward and decision contract

Use the maintainer-confirmed gross strategic-return contract and current counterfactual defaults:

- strategic episode horizon: 160 physics ticks;
- macro action duration: at most 12 physics ticks, with a partial final macro when the remaining episode horizon is shorter;
- discount factor: gamma `0.993`;
- gross return from the current counterfactual reward defaults; route and risk costs remain separate quantities and are not subtracted from this return target;
- retain the current counterfactual default reward definitions and signs; P1/P5 must bind their exact implementation/profile identity before source freeze and P0b.

The return is duration-aware at macro boundaries: the first reward contribution on each macro branch uses discount exponent zero; each continuation decision advances gamma by one exponent regardless of that macro's physics-tick duration. A truncated final macro executes only the remaining physics ticks and closes the 160-physics-tick episode. P1/P5 must bind this branch/exponent rule and reward aggregation to the pinned profile and implementation; P0b fails closed on any mismatch. The Phase 0 outcome estimand remains accepted active-HUB FUEL awards; it is not asserted to be the sole PPO reward component. Route cost and risk are reported separately, so no comparator may quietly mix them into a candidate's gross-return label while another does not.

## 3. Experimental unit, pairing, and policy set

The experimental unit is one complete simulator episode under one frozen policy, scenario, initial state, opponent assignment, and seed. Each learned track and deterministic comparator receives the same paired scenario/start/opponent/seed cell. Preserve paired identities in raw records and calculate within-pair differences. A cell is never replaced after observing an outcome; failed or incomplete runs are retained with failure status and handled by the preregistered missing-data rules.

The track set is exactly A1/A2/A3/C1/C2. The deterministic candidates are D0/D1/D2 as defined above. Hold simulator rules, physics, sensing, start state, opponent behavior, available observations, action legality, evaluation budgets, and compute allocation fixed wherever the architectures permit. Record unavoidable information or action-space differences rather than describing the inputs as identical. D3 is excluded from this primary protocol unless the maintainer explicitly amends the finite policy set before any candidate results are observed.

## 4. Partition and seed construction

Four disjoint partitions are required: training, calibration, validation, and final test. Their scenario cells and seed lists must be frozen as canonical bytes before training. No seed, initial state, or opponent-policy instance may cross partitions. Hold out complete opponent-policy families and map/start situations from training, not only random seeds. Calibration may estimate baseline variance, uncertainty, and margins and select the strongest deterministic comparator. Validation is for predeclared interim diagnostics only. Final test is opened once after all policies, comparator identity, source, profile, and decision rules are frozen.

Use the deterministic seed construction already present at the source reference base as the default: for each scenario/split/stratum, compute `namespace = int.from_bytes(SHA256(UTF8(layout_id + "/" + defender_family + "/" + start_condition + "/" + sensor_profile))[:4], "big")`; then call `training_curriculum.seed_partition(scenario_id, split, count, master_seed=(20261024 << 32) | namespace)`. Proposed count is 100 seeds per category/stratum, matching the existing `MIN_EPISODES` floor. The scenario ID is separately incorporated by `seed_partition`; the namespace intentionally excludes category, matching `meco_qualification.protocol_cases`, so categories in a scenario/stratum share seeds. Preserve the function's stable 63-bit seed values and split bits, retain exact generated seed lists in canonical protocol bytes, and share each cell seed across all policies in its paired block. P6 must revalidate the source functions against the exact frozen integration base and bind their implementation digest; if v5 changes the seed algorithm, stop for renewed preregistration approval.

The seed derivation rule and default master seed/count are proposals. The specific partition membership, IDs, counts, and collision handling remain subject to the maintainer decisions in §10. The seed generator is not evidence that the case matrix has been selected.

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

For the comparative research outcome, advance a track for further consideration only if the lower confidence bound exceeds the calibration-derived minimum useful gain and the safety noninferiority gate passes. This is an experimental advancement gate, not Stage 1 readiness or production authorization; Stage 1 readiness uses the `pickup_simple` curriculum predicate and its v5 gate. If the interval is inconclusive, the result is inconclusive; do not call a tie a pass. Report all five track comparisons, failures, missing cells, and multiplicity treatment. D1/D2 diagnostic contrasts may explain comparator strength but do not alter the primary baseline after it is frozen.

The calibration sample floor, baseline-variance estimator, uncertainty method, bootstrap replicate count, minimum useful gain derivation, safety margin derivation, and multiplicity procedure are proposed protocol fields requiring explicit approval in §10. Numeric margins are calculated in P0b and never chosen after candidate results are inspected.

## 7. Calibration method (P0b only)

P0b uses only approved D0/D1/D2 calibration cells and approved repeats. It estimates baseline variance, checks raw event completeness and paired replay consistency, computes the prescribed uncertainty and margins, and selects the strongest deterministic comparator using the approved comparative outcome. It does not use candidate validation/final-test results, train on calibration outcomes, or change the preregistered estimand or scenario manifest. Because `pickup_simple` does not require active-HUB awards, zero or near-zero award variation may make score-based comparator ranking and margin calibration impossible. In that case P0b stops for a maintainer decision; it does not substitute the curriculum predicate for the approved award estimand or silently expand to `full_match_offense`. Any full-match award analysis is a separate secondary protocol. If calibration evidence is insufficient under the approved floor, P0b stops rather than silently adding samples or revising the method.

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

### Repository inventory and proposed finite default (for approval)

The v5 Stage 1 preflight source requires `promotion_target_scenario == "pickup_simple"`; the registered `pickup_simple` scenario has no prerequisites, so its exact Stage 1 closure is only `pickup_simple`, with categories `single` and `cluster`. This is the recommended and schema-compatible Stage 1 mapping. Its curriculum success criterion is acquiring and retaining required distinct FUEL. The user-selected active-HUB award estimand is retained as a distinct limited score proxy; it is not guaranteed by these Stage 1 categories and cannot replace their curriculum predicate.

For optional secondary/generalization analysis, `full_match_offense` is the registered complete-schedule target. Its transitive prerequisite closure is all 12 curriculum scenarios, in registry order: `pickup_simple`, `pickup_scattered`, `pickup_multi_zone`, `active_carry_to_zone`, `active_shoot_preloaded`, `inactive_ferry_preloaded`, `active_collect_carry_score`, `inactive_collect_ferry`, `hub_transition_cycles`, `offense_vs_scripted_pressure`, `team_delivery_cycles`, `full_match_offense`. Each scenario's category list is finite in `frc_defense/training_curriculum.py`. This broader result is separate from Stage 1 readiness and requires a separately bound secondary outcome table if selected.

The existing qualification source provides these applicable axes: layouts `2026_rebuilt`, `2026_rebuilt_shifted_structures_v1`, `_v2`, and `_v3`; defender families `category_scripted`, `orbiting_interceptor`, and `split_chaser`; starts `scenario_defined`, `offset_upper_lane`, and `offset_lower_lane`; sensor profiles `standard` and `noisy`. `meco_qualification.py` also defines 100 episodes as the minimum per category/stratum, a 10,000-replicate bootstrap default, `meco-safety-rates-v1`, and metric IDs `safety_robot_contact_rate`, `safety_field_contact_rate`, and `safety_wall_contact_rate`. These are existing v4 identifiers/defaults, not automatically accepted v5 semantics.

Proposed finite v5 Stage 1 matrix. Apply each row to the two registered `pickup_simple` categories, with defender family `none` in every partition (the existing protocol helper uses `none` for non-pressure scenarios); pair all five tracks and matched deterministic policies on the same row and seed:

| Partition | Layout ID | Defender family | Start condition | Sensor profile | Proposed episodes per category/stratum |
|---|---|---|---|---|---:|
| train | `2026_rebuilt` | `none` | `scenario_defined` | `standard` | existing training protocol |
| calibration | `2026_rebuilt_shifted_structures_v1` | `none` | `offset_upper_lane` | `noisy` | 100 |
| validation | `2026_rebuilt_shifted_structures_v2` | `none` | `offset_upper_lane` | `noisy` | 100 |
| final test | `2026_rebuilt_shifted_structures_v3` | `none` | `offset_lower_lane` | `noisy` | 100 |

This Stage 1 matrix uses only the `pickup_simple` closure. It proposes one layout per partition, rather than the existing helper's two layouts for calibration/validation, to keep the split finite and disjoint. This is an explicit v5 axis override requiring maintainer approval. Case identity is the canonical tuple `(scenario_id, category, layout_id, defender_family, start_condition, sensor_profile)`; freeze the exact expanded tuple and seed table as canonical bytes before training.

Optional secondary/generalization matrix for `full_match_offense`: use the same partition-specific layout/start/sensor rows above; for the two pressure scenarios (`offense_vs_scripted_pressure`, `full_match_offense`) use defender family `category_scripted` in train, `orbiting_interceptor` in calibration/validation, and `split_chaser` in final test; for all other closure scenarios use `none`. This makes `split_chaser` wholly held out for that secondary comparison. It overrides the helper's legacy calibration/validation axes, which enumerate two layouts and two families; the v5 protocol must record the approved override, and secondary results must not affect Stage 1 readiness. Whether to include this secondary matrix is a maintainer decision.

Existing safety candidate IDs are the three contact rates listed above. Existing curriculum failure semantics include timeout or failure of the stated physical outcome; current evaluation records also name `selected_target_lost`, `native_behavior_preempted_selected_goal`, `action_was_remasked`, and policy-inference fallback reasons. These do not yet define a complete v5 episode-level failure taxonomy. Proposed v5 finite disposition classes are `complete`, `timeout`, `safety_stop`, `simulator_error`, `worker_or_device_error`, `invalid_or_stale_record`, and `incomplete_episode`; map lower-level execution reasons into these classes with the original reason retained. This mapping and whether each class counts as a safety event or makes a pair non-evaluable require maintainer approval and source-bound implementation by P6.

| Decision | Required maintainer selection |
|---|---|
| Outcome aggregation scope | Accept the active-HUB FUEL award event aggregation scope for the controlled robot/team; specify event ownership and tie/duplicate treatment. The estimand remains a limited proxy, not official match points. |
| Stage 1 score-proxy relation | Confirm that the accepted-active-HUB-award estimand remains recorded on `pickup_simple` despite the collection/retention task not requiring an award; confirm it is not the Stage 1 pass predicate. |
| Held-out maps and starts | Accept the proposed v1/v2/v3 layout split and upper/lower-lane starts, or specify exact alternatives and initial state IDs. |
| Held-out opponents | For Stage 1 `pickup_simple`, accept defender family `none` in each partition. If the optional full-match secondary matrix is selected, accept `category_scripted` for train, `orbiting_interceptor` for calibration/validation, and wholly held-out `split_chaser` for final-test pressure scenarios; provide frozen policy/checkpoint/config identities. |
| Case matrix | Accept the two-category Stage 1 matrix and explicit v5 layout override; separately decide whether to add the optional full-match secondary matrix and its pressure-only defender rule. |
| Partition membership | Accept exact Stage 1 tuples and derived seed bytes after generation using the stated namespace algorithm, including 100 episodes per category/stratum for calibration, validation and final test; training uses the existing training protocol. If secondary full-match cases are included, freeze their separate full-closure tuple/seed table too. |
| Repeat/sample floor | Accept 100 episodes per category/stratum and existing calibration repetition minimum of two, or specify alternatives and the inadequate-evidence rule. |
| Primary uncertainty | Approve the existing 10,000-replicate bootstrap as count default and select percentile paired bootstrap or specify replacement; freeze resampling unit, two-sided 95% interval, stratification and five-comparison multiplicity method. |
| Safety metrics and vetoes | Accept the three existing contact-rate IDs and source-bound definitions, plus exact critical-event vetoes; specify any additions/omissions and scope. |
| Missing/failure mapping | Accept the proposed finite episode disposition classes and define the effect of each on safety and the primary paired estimate. |
| Comparator set | Use the five planned tracks A1/A2/A3/C1/C2 against D0/D1/D2; D3 remains outside the preregistered comparator set. Any D3 addition requires an amendment before candidate outcomes. |

This candidate adopts the already confirmed fixed strategic episode horizon, macro limit, discount factor, gross-reward semantics, and active-HUB FUEL-award proxy. It proposes a finite, source-grounded matrix but does not treat it as approved. If any decision is deferred, P0a remains pending and P1–P6 source authoring that depends on the unresolved item must wait.

## 11. P0a/P0b acceptance boundary

P0a Pass 3 reviews this exact document and the maintainer decision record. Acceptance confirms only a preregistered documentation contract and unlocks package source/test-file authoring under each accepted plan. It does not authorize builds, tests, numerical computation, simulator runs, dashboard operations, devices, training, integration, merge, promotion, calibration, or Stage 1 qualification before G0.

P0b starts only after P1–P6 source freeze and independent review, P0a acceptance, G0, both WX9100 leases independently released to the assigned operation, and the reviewed P6 runner. P0b generates immutable source-bound calibration and v5 protocol artifacts. Any relevant source or preregistration change requires re-freeze/review and regeneration before fresh training or qualification.

No tests, builds, simulator evaluation, calibration, dashboard operation, or device/GPU use were performed to prepare this candidate.
