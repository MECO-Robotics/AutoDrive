# MECO V4 Phase 0 preregistration candidate (P0a)

**Status:** Candidate for maintainer decisions and independent exact-head review; not approved for calibration, training, qualification, or runtime use.
**Accepted package plan:** `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md` at `3071c2496eb8bcb326ad0f1e971f1bd1e8d0858f`, blob `f8e1827fc0354b88a908792541aa515790b4bed0`, SHA-256 `5ac9e99cf0ead523cecf6cdc64753d7801361fc876b862f762ae18f70ef4c831`; exact-head Pass 1 ACCEPT review commit `f2f299a81d266dc5e08c5ce27402151c0e78571d`, record SHA-256 `baf8666f6e93550f31afc441e87eaeb089dffd6e4401376bb943ca978e4d529e`.
**Accepted P6 dual-protocol schema/runner plan:** `docs/plans/meco-v4-p6-five-track-supervisor-qualification-package-plan-20261026.md` at `c1f20c80a15588725b3f346ae06366c2197465b7`, file SHA-256 `8c122a5ea82e2f3b77f222c8954a92ae6d11e54a4ded55bfdb786d42283fc516`; exact-head Pass 1 ACCEPT review commit `f025827abe2d3aec366b65b4fad85debd5de31aa`, record SHA-256 `821a11b053c86780bca35aff45e53373c003d7046468e4e1b6905d39d67df6c8`. This accepts P6 implementation scope only, not this candidate’s protocol choices, source implementation, or runtime use.
**Overall plan:** `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md` at `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`.
**Source reference base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
**Score-ledger implementation evidence:** `frc_defense/meco_simulated_score_ledger.py` at `main` commit `d74b0a15e25ddbc6bacfbbb864674371629bb19e` (blob `39677ce1cc26009dfd25d4c8badb46194b010d41`, SHA-256 `71427d251f1639a4a113b610b5c8122fe88d1c2aac9c11eb4861b0f82bc05114`); source commit `97120b3b41adacc71f33d7750067c89348e58ff5`. The source-bound design is described in `docs/plans/phase0-score-ledger-replay-implementation-20261025.md` at commit `db63533ceeecb7ea9181228234ff620f9e6281ad`, blob `c70e52ca994a6da1fdaeca29996a3270dd19c36d`, SHA-256 `3a553986ad9849e8b482292f19d2b7936e9a050ecc6bc2c7cbde6e9ba3e3672c`.
**Safety-metric source evidence:** `frc_defense/meco_metric_ledger.py` at `main` commit `d74b0a15e25ddbc6bacfbbb864674371629bb19e` (blob `d74bb062bcd15599caa4d801063e9300fc09682c`, SHA-256 `1d370591e5aa76ffc8b891a691268e3d897169c4dfd5456b282141c8596ad2f1`). It defines contact-rate IDs and focal slot-3 per-active-physics-tick rates; it marks illegal-transition and footprint events unavailable (lines 12–17, 206–218, 230–235, 287–302).
**Current Stage 1 contract evidence:** `frc_defense/meco_qualification.py` at repository `main` commit `d74b0a15e25ddbc6bacfbbb864674371629bb19e`, where `preflight_stage1` requires `promotion_target_scenario == "pickup_simple"`.
**Reconciled profile:** SHA-256 `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.
**Reconciliation artifact:** SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`.
**V4 envelope digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.

This is a preregistration proposal. The authorized maintainer must resolve the explicitly open finite-case decisions in §10 before P0a can receive ACCEPT. It does not claim official match points, select a held-out set, set data-derived margins, or establish successful behavior. P0a performs no execution or calibration. P0b creates the final source-bound protocol and Phase 0 artifact only after P1–P6 source freeze/review and the G0 gate.

## 1. Question and comparison

The primary Phase 0 research evaluation is the `simulated_match_score` estimand: accepted active-HUB FUEL awards, a limited simulated proxy and not official match points. Evaluate A1, A2, A3, C1, and C2 against the strongest matched deterministic strategy on `full_match_offense` held-out simulator outcomes. The deterministic comparison set is D0 (pinned existing policy), D1 (matched candidate strategy), and D2 (D1 with the shared predictive opponent model). Select the strongest comparator using only the full-match score-calibration partition, freeze its identity before validation and final testing, and compare every track against that same selected comparator. Do not select a comparator or tune a policy using validation or final-test outcomes.

Define `simulated_match_score` as whole-alliance simulated score totals: sum `score_units` by alliance over unique authoritative accepted FUEL score-award event IDs, yielding the ledger's `red` and `blue` totals for each episode. The existing ledger rejects duplicate event IDs within an episode ledger, binds each event to an alliance and scoring source, and validates that alliance against the robot-team map. It reports no per-robot causal contribution from whole-team scoring events (ledger source: lines 102–104, 287–291, 359–378; implementation plan: lines 40–46). P1 must bind the event producer and consumers to the exact reviewed source and reconcile the approval artifact's Phase 0 digest with this accepted P0a/P0b contract before any outcome collection; if source review finds different event semantics or aggregation, return the protocol to maintainer review before P0b. This score estimand is separate from the PPO return and from the Stage 1 curriculum pass predicate.

The v5 Stage 1 qualification target is independently `pickup_simple`, matching the current `preflight_stage1` contract; its registered prerequisite closure is exactly `pickup_simple`. The curriculum defines this target by acquiring and retaining required distinct FUEL. Stage 1 readiness is bound only to that scenario's source-bound curriculum gate and v5 qualification protocol. The full-match score result cannot establish Stage 1 passage, and a pickup_simple curriculum pass cannot establish the primary score outcome.

The research comparison follows `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`: compare all five tracks with the strongest matched deterministic baseline on held-out outcomes, using paired states/seeds, confidence intervals, and no safety regression. `pickup_simple` remains the Stage 1 qualification target but does not require an active-HUB award. Its curriculum predicate must not be replaced by the award metric. The primary award comparison therefore uses a distinct `full_match_offense` score-evaluation protocol and evidence set; that score result does not establish Stage 1 passage.

The score estimand is accepted active-HUB FUEL award count per complete 160-physics-tick strategic episode. Other diagnostic outcomes include award rate over the same fixed horizon, FUEL-per-second, scoring/ferry efficiency, failed collection attempts, defensive exposure/interception, route feasibility, candidate recall, policy switching, reliability/fallback behavior, and end-to-end compute latency. None replaces the Stage 1 curriculum predicate or the user-selected award estimand.

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

`simulated_match_score`: for each complete episode, retain the source ledger's `red` and `blue` whole-alliance totals, each the integer sum of `score_units` over unique accepted active-HUB FUEL score-award event IDs assigned to that alliance. Retain each event's tick, source identity, HUB eligibility state, alliance, and stable event ID in raw evidence. The protocol must freeze which alliance receives each tested policy in paired cases, while retaining both alliance totals. Never count attempted scores, inactive-HUB activity, curriculum reward, or unaccepted event pulses.

### Required secondary families

Record at minimum the evaluation-plan categories: game performance; defensive robustness; navigation/safety; strategic decisions; reliability; compute; and generalization. P1/P5 must provide the exact metric IDs, units, aggregation functions, and missingness representation in the source-bound protocol. Do not invent or silently change definitions during calibration.

### Safety non-regression

Proposed safety contract (not approved): apply a hard veto if any source-bound event occurs in a scheduled episode for (1) an executed action that violates the deterministic legality contract after the final action gate, (2) a swept robot-footprint transition that violates the frozen geometry constraints, (3) a capture/mechanism state that violates the frozen capture constraints, or (4) a required safe fallback that fails to produce and execute a valid safe action. Any such event fails the policy safety gate regardless of score or confidence interval. A selected action being remasked (`action_was_remasked`) is retained as a separate execution diagnostic and does not by itself establish that an illegal action was executed. The current metric ledger explicitly has no authoritative illegal-transition or footprint-violation signal; until P1/P6 implement and validate source-bound event signals for all proposed hard-veto classes, these gates are unavailable and no safety pass can be claimed.

Separately, compare the existing `meco-safety-rates-v1` contact metrics—`safety_robot_contact_rate`, `safety_field_contact_rate`, and `safety_wall_contact_rate`—under the approved noninferiority margins. Their current definitions are contact ticks divided by active physics ticks for focal slot 3, with robot, field, and wall contacts reported separately; unavailable counters are null, not zero. They measure observed contact, not policy-rule violations, so they do not replace the hard-veto events. Their margins and exact aggregation across strata remain maintainer decisions. A higher score cannot offset a hard veto.

No numerical useful-gain threshold or safety margin is asserted here. P0b computes the preregistered margins from calibration-only baseline variance and the accepted method; final-test outcomes cannot tune margins.

## 6. Statistical method and decision rule

For each track, compute paired episode differences in the whole-alliance `simulated_match_score` total for the alliance assigned that track, against the single calibration-selected comparator; retain both red and blue totals in every episode record. Proposed interval method: use 10,000 paired-block bootstrap replicates, resampling paired case/seed blocks with replacement within each approved scenario/opponent-family stratum and keeping all policy/comparator rows for each sampled block together. In each replicate compute all five track-minus-selected-comparator mean differences. For simultaneous familywise 95% coverage, use Bonferroni-adjusted two-sided percentile intervals for the five contrasts: each interval uses the 0.5th and 99.5th bootstrap percentiles (per-contrast alpha 0.01; total family alpha at most 0.05). Freeze the bootstrap RNG, resampling-unit identity and stratum handling in canonical protocol bytes. This method and all its inputs are proposals pending maintainer approval.

For the comparative research outcome, advance a track for further consideration only if the lower confidence bound exceeds the calibration-derived minimum useful gain and the safety noninferiority gate passes. This is an experimental advancement gate, not Stage 1 readiness or production authorization; Stage 1 readiness uses the `pickup_simple` curriculum predicate and its v5 gate. If the interval is inconclusive, the result is inconclusive; do not call a tie a pass. Report all five track comparisons, failures, missing cells, and multiplicity treatment. D1/D2 diagnostic contrasts may explain comparator strength but do not alter the primary baseline after it is frozen.

The calibration sample floor, baseline-variance estimator, uncertainty method, bootstrap replicate count, minimum useful gain derivation, safety margin derivation, and multiplicity procedure are proposed protocol fields requiring explicit approval in §10. Numeric margins are calculated in P0b and never chosen after candidate results are inspected.

## 7. Calibration method (P0b only)

P0b has two non-interchangeable evidence roles. The Stage 1 qualification protocol binds `pickup_simple` and the curriculum's acquisition/retention predicate; its evidence can establish Stage 1 only. The primary score-evaluation protocol binds `full_match_offense`, accepted active-HUB award outcomes, D0/D1/D2 calibration, paired five-track comparisons, and held-out score analysis. It cannot establish Stage 1 passage. Score calibration uses only the full-match score-calibration partition; it estimates baseline variance, uncertainty, and margins, and selects the strongest deterministic comparator using the score estimand. It does not use candidate validation/final-test outcomes or change the preregistered estimand or scenario manifest. If full-match score calibration is insufficient under the approved floor, P0b stops rather than silently adding samples or revising the method. The two protocols, source identities, target closures, partition bytes, and generated records must be bound separately and cross-referenced by the Phase 0 artifact.

Calibration and final evaluation must run on both assigned WX9100 devices under the G0 device-release gate, with algorithmic numerical computation GPU-only and no CPU numerical fallback. CPU orchestration and file I/O are permitted. Retain device UUID/runtime/kernel/residency evidence and fail closed if either device or the approved execution path is unavailable. P0a itself does not inspect devices or run calibration.

## 8. Protocol identity, artifacts, and invalidation

The v5 design must bind two distinct target roles by canonical serialization and cryptographic hash:

1. Stage 1 qualification protocol: target `pickup_simple`, exact prerequisite closure, current curriculum success predicate, and fields required by `preflight_stage1`;
2. primary score-evaluation protocol: target `full_match_offense`, exact prerequisite closure, active-HUB award metric, paired policies/cases/seeds, calibration and statistical fields;
3. exact simulator/rules/source identity and P1's score producer/consumer map;
4. reconciled profile and counterfactual reward contract;
5. policy IDs, comparator-selection rule and selected comparator for the score protocol;
6. finite case manifests, partition bytes, paired seed preimages and episode limits for each protocol;
7. metric definitions, safety rules, statistical method, margin method and the independent Stage 1/score predicates;
8. device pair identity and execution/residency evidence;
9. raw calibration/evaluation record identities, approvals, timestamps, software versions and result hashes.

Current Stage 1 preflight validates only a protocol whose `promotion_target_scenario` is `pickup_simple`; one target-scoped record cannot serve both targets. The accepted P6 plan at the pinned head above owns schema/runner support for separate Stage 1 and primary score-evaluation records and Phase 0 cross-binding. The required paths are `protocol.json` for Stage 1 `pickup_simple`, `score-evaluation-protocol.json` for primary `full_match_offense`, separate `stage1-calibration/` and `score-calibration/` evidence namespaces, and one immutable `phase0-artifact.json` binding both roles and raw-byte digests. P6 plan acceptance establishes the implementation scope; P6 source/API must still implement and pass exact-head review before P0b. This candidate does not authorize changing qualification code in P0a.

Serialize canonical JSON as UTF-8, sorted keys, compact separators, no NaN/Infinity, and explicit schema version. Hash the exact serialized bytes with SHA-256. Keep raw records immutable and give each run a unique ID. A source, rules, profile, metric, case, seed, or device change invalidates the dependent artifact; stale or mismatched identities fail closed. Human decisions and signatures/approval provenance must be recorded separately from agent-generated hashes. Never overwrite prior protocol or calibration artifacts.

Generated P0b artifacts are outside Git. Proposed paths are `qualifications/meco/v5/<stage1-protocol-id>/phase0-artifact.json`, `protocol.json` (Stage 1 target `pickup_simple`), `score-evaluation-protocol.json` (primary score target `full_match_offense`), `stage1-calibration/<case-id>/<unique-run-id>/...`, and `score-calibration/<case-id>/<unique-run-id>/...`. P0a creates none of these files. The accepted P0 package plan and accepted P6 schema/runner plan now explicitly cover the second protocol, separate evidence paths, and Phase 0 cross-binding. P6 must implement and review that contract before P0b begins.

## 9. Missing data, failures, and audit record

Keep one row for every scheduled policy/cell, including launch failure, simulator failure, invalid record, safety stop, timeout, or incomplete episode. The following finite dispositions and effects are proposed for approval. A normal completion at the preregistered strategic horizon is `complete`; `timeout` means the task or run terminated before a complete horizon/predicate. Never replace a scheduled case or paired block after observing outcomes. Retain the all-scheduled ledger and publish the analyzed paired set with an exclusion reason for every non-evaluable block.

| Disposition | Proposed score-evaluation handling | Proposed Stage 1 handling | Proposed safety handling |
|---|---|---|---|
| `complete` | Include only when the full horizon and source-bound score ledger are complete; a real zero-award complete episode is zero. | Evaluate the registered pickup_simple curriculum predicate; count only a complete passing round. | Evaluate contact-rate noninferiority and all hard-veto event signals. |
| `timeout` | Incomplete score episode: no zero-imputation; mark its paired block non-evaluable. No replacement. | Fail the curriculum round. | Preserve observed contacts/veto events; timeout alone is not a hard-veto event. |
| `safety_stop` | Retain partial record; no score imputation; paired block non-evaluable. No replacement. | Fail the round. | Fail the safety gate; preserve the triggering stop reason and any measured contact/veto events. |
| `simulator_error` | No score estimate for affected block; non-evaluable. No replacement. | Fail the round. | Any already observed hard-veto event still fails; otherwise safety result is unavailable for the affected block. |
| `worker_or_device_error` | No score estimate for affected block; non-evaluable. No replacement. | Fail the round. | Any already observed hard-veto event still fails; otherwise safety result is unavailable for the affected block. |
| `invalid_or_stale_record` | Reject record and block from score analysis; non-evaluable. No replacement. | Round cannot pass. | No safety pass can be claimed from the invalid record. |
| `incomplete_episode` | No score estimate; never impute zero; paired block non-evaluable. No replacement. | Round cannot pass. | Preserve any observed hard-veto event; otherwise safety result is unavailable. |

A failed pair does not become a score of zero. Only a valid, complete ledger with zero accepted award events has a score value of zero. These rules are proposed and require maintainer approval before P0a ACCEPT.

The final audit record includes the exact protocol hash, source/profile/rules/device hashes, policy/checkpoint hashes, scenario and seed, paired cell ID, run/replay identifiers, outcome rows, metric values, failure status, interval/margin computation, approvals, and artifact hashes. Dashboard replays are generated only after G0 and only from the current reviewed checkout; restart the relevant services, verify backend and frontend freshness/endpoints, create a fresh replay, and verify its provenance. A stale replay is marked outdated and is not evidence.

## 10. Maintainer decisions required before P0a ACCEPT

These questions are intentionally open because the approved product document does not name a finite held-out set or fully define all protocol mappings. Record selected values and rationale in a decision record attached to this exact candidate before Pass 3. No implementing agent may guess these values.

### Repository inventory and proposed finite default (for approval)

The v5 Stage 1 preflight source requires `promotion_target_scenario == "pickup_simple"`; the registered `pickup_simple` scenario has no prerequisites, so its exact Stage 1 closure is only `pickup_simple`, with categories `single` and `cluster`. This is the recommended and schema-compatible Stage 1 mapping. Its curriculum success criterion is acquiring and retaining required distinct FUEL. The primary active-HUB award estimand is evaluated on the separate `full_match_offense` score protocol; it is not guaranteed by the Stage 1 categories and cannot replace their curriculum predicate.

For the primary score-estimand analysis, `full_match_offense` is the registered complete-schedule target. Its transitive prerequisite closure is all 12 curriculum scenarios, in registry order: `pickup_simple`, `pickup_scattered`, `pickup_multi_zone`, `active_carry_to_zone`, `active_shoot_preloaded`, `inactive_ferry_preloaded`, `active_collect_carry_score`, `inactive_collect_ferry`, `hub_transition_cycles`, `offense_vs_scripted_pressure`, `team_delivery_cycles`, `full_match_offense`. Each scenario's category list is finite in `frc_defense/training_curriculum.py`. This primary score result is separate from Stage 1 readiness and requires a separately bound score-evaluation protocol and outcome table.

The existing qualification source provides these applicable axes: layouts `2026_rebuilt`, `2026_rebuilt_shifted_structures_v1`, `_v2`, and `_v3`; defender families `category_scripted`, `orbiting_interceptor`, and `split_chaser`; starts `scenario_defined`, `offset_upper_lane`, and `offset_lower_lane`; sensor profiles `standard` and `noisy`. `meco_qualification.py` also defines 100 episodes as the minimum per category/stratum, a 10,000-replicate bootstrap default, `meco-safety-rates-v1`, and metric IDs `safety_robot_contact_rate`, `safety_field_contact_rate`, and `safety_wall_contact_rate`. These are existing v4 identifiers/defaults, not automatically accepted v5 semantics.

Proposed finite v5 Stage 1 matrix. Apply each row to the two registered `pickup_simple` categories, with defender family `none` in every partition (the existing protocol helper uses `none` for non-pressure scenarios); pair all five tracks and matched deterministic policies on the same row and seed:

| Partition | Layout ID | Defender family | Start condition | Sensor profile | Proposed episodes per category/stratum |
|---|---|---|---|---|---:|
| train | `2026_rebuilt` | `none` | `scenario_defined` | `standard` | existing training protocol |
| calibration | `2026_rebuilt_shifted_structures_v1` | `none` | `offset_upper_lane` | `noisy` | 100 |
| validation | `2026_rebuilt_shifted_structures_v2` | `none` | `offset_upper_lane` | `noisy` | 100 |
| final test | `2026_rebuilt_shifted_structures_v3` | `none` | `offset_lower_lane` | `noisy` | 100 |

This Stage 1 matrix uses only the `pickup_simple` closure. It proposes one layout per partition, rather than the existing helper's two layouts for calibration/validation, to keep the split finite and disjoint. This is an explicit v5 axis override requiring maintainer approval. Case identity is the canonical tuple `(scenario_id, category, layout_id, defender_family, start_condition, sensor_profile)`; freeze the exact expanded tuple and seed table as canonical bytes before training.

Primary score-estimand matrix for `full_match_offense`: use the same partition-specific layout/start/sensor rows above; for the two pressure scenarios (`offense_vs_scripted_pressure`, `full_match_offense`) use defender family `category_scripted` in train, `orbiting_interceptor` in calibration/validation, and `split_chaser` in final test; for all other closure scenarios use `none`. This makes `split_chaser` wholly held out for the primary score comparison. It overrides the helper's legacy calibration/validation axes, which enumerate two layouts and two families; the v5 protocol must record the approved override, and these primary score results must not be used to claim Stage 1 readiness. Approval of this score matrix remains a maintainer decision.

Existing safety candidate IDs are the three contact rates listed above. Existing curriculum failure semantics include timeout or failure of the stated physical outcome; current evaluation records also name `selected_target_lost`, `native_behavior_preempted_selected_goal`, `action_was_remasked`, and policy-inference fallback reasons. These do not yet define a complete v5 episode-level failure taxonomy. Proposed v5 finite disposition classes are `complete`, `timeout`, `safety_stop`, `simulator_error`, `worker_or_device_error`, `invalid_or_stale_record`, and `incomplete_episode`; map lower-level execution reasons into these classes with the original reason retained. This mapping and whether each class counts as a safety event or makes a pair non-evaluable require maintainer approval and source-bound implementation by P6.

| Decision | Required maintainer selection |
|---|---|
| Proxy approval and interpretation | Approve the user-selected limited proxy as formally specified: whole-alliance `simulated_match_score`, calculated by summing score units over unique authoritative accepted active-HUB FUEL award event IDs, with red/blue totals and no per-robot causal contribution. Confirm its intended comparison use and the explicitly excluded outcomes; it remains a simulated proxy, not official match points. The cited source contract fixes alliance-level aggregation. |
| Dual-target protocol design | Approve `pickup_simple` as the sole Stage 1 qualification protocol and `full_match_offense` as the primary active-HUB award score protocol, separately bound and cross-referenced in the Phase 0 artifact; neither result substitutes for the other predicate. The accepted P6 plan covers this schema/runner scope; require reviewed P6 source/API implementation before P0b. This design remains subject to authorized maintainer approval. |
| Held-out maps and starts | Accept the proposed v1/v2/v3 layout split and upper/lower-lane starts, or specify exact alternatives and initial state IDs. |
| Held-out opponents | For Stage 1 `pickup_simple`, accept defender family `none` in each partition. For the primary full-match score matrix, accept `category_scripted` for train, `orbiting_interceptor` for calibration/validation, and wholly held-out `split_chaser` for final-test pressure scenarios; provide frozen policy/checkpoint/config identities. |
| Case matrix | Accept the two-category Stage 1 matrix and explicit v5 layout override; approve the primary full-match score matrix, its pressure-only defender rule, and the paired red/blue alliance assignment for each tested policy. |
| Partition membership | Accept exact Stage 1 tuples and derived seed bytes after generation using the stated namespace algorithm, including 100 episodes per category/stratum for calibration, validation and final test; training uses the existing training protocol. For full-match primary score cases, freeze their separate full-closure tuple/seed table too. |
| Repeat/sample floor | Accept 100 episodes per category/stratum and existing calibration repetition minimum of two, or specify alternatives and the inadequate-evidence rule. |
| Primary uncertainty | Approve or replace the proposed 10,000 paired-block bootstrap, resampling within scenario/opponent-family strata, with Bonferroni simultaneous familywise 95% percentile intervals for five track contrasts (per-contrast 99% interval; 0.5th/99.5th percentiles); freeze RNG, block identity, stratum handling and any replacement method. |
| Safety metrics and vetoes | Approve or revise the three source-defined focal-slot contact-rate metrics as separate noninferiority outcomes, their margins and aggregation; approve or revise hard vetoes for executed deterministic-legality, swept-footprint, capture/mechanism, and failed-safe-fallback violations. Require source-bound signals for every veto before any safety pass; contact rates do not substitute for veto evidence. |
| Missing/failure mapping | Approve or revise the proposed per-class outcomes in §9, including no replacements, no zero-imputation of incomplete score episodes, and Stage 1 timeout as a failed curriculum round. |
| Comparator set | Use the five planned tracks A1/A2/A3/C1/C2 against D0/D1/D2; D3 remains outside the preregistered comparator set. Any D3 addition requires an amendment before candidate outcomes. |

This candidate adopts the already confirmed fixed strategic episode horizon, macro limit, discount factor, gross-reward semantics, and active-HUB FUEL-award proxy with source-defined whole-alliance aggregation. It proposes a finite, source-grounded matrix but does not treat it as approved. The accepted P0 and P6 plans establish documentation and future implementation scope only. If any decision is deferred, P0a remains pending and P1–P6 source authoring that depends on the unresolved item must wait.

## 11. P0a/P0b acceptance boundary

P0a Pass 3 reviews this exact document and the maintainer decision record. Acceptance confirms only a preregistered documentation contract and unlocks package source/test-file authoring under each accepted plan. It does not authorize builds, tests, numerical computation, simulator runs, dashboard operations, devices, training, integration, merge, promotion, calibration, or Stage 1 qualification before G0.

P0b starts only after P1–P6 source freeze and independent review, P0a acceptance, G0, both WX9100 leases independently released to the assigned operation, and the reviewed P6 runner. P0b generates immutable source-bound Stage 1 qualification and primary score-evaluation protocols with separate calibration evidence. Any relevant source or preregistration change requires re-freeze/review and regeneration before fresh training or qualification.

No tests, builds, simulator evaluation, calibration, dashboard operation, or device/GPU use were performed to prepare this candidate.
