# Phase 0 baseline variance and operational-margin plan

## Planner pass

### Objective

Close the Phase 0 evidence gate for the primary `full_match_offense` promotion target without changing the live r4 campaign. Produce a target-scoped v4 protocol only after (1) a complete D0 calibration bundle, (2) repeated matched-reference variance evidence on identical calibration seeds, and (3) an explicit, reviewed operational-margin decision. This work establishes calibration inputs; it does not qualify any A1/A2/A3/C1/C2 policy.

### Current evidence and gap

- The source plan requires measuring baseline variance before setting the minimum operationally useful improvement; it explicitly rejects inventing a universal percentage (`docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`, §§5–6).
- The checked-in protocol is provisional v1 with `gate_calibration_status=provisional_pending_baseline_variance_review`. The current qualification instructions require a complete prerequisite closure, immutable D0 bundle, repeated matched-reference bundle, explicit margin review, and v4 before promotion validation (`docs/MECO_QUALIFICATION.md`).
- The only checked-in D0 calibration report is `pickup_simple`: 1,200 rows, four strata, 100 seeds × 3 repeats. It does not cover the other 11 scenarios in the `full_match_offense` closure and its standard errors do not select margins (`docs/implementation-provenance/d0-pickup-simple-calibration-audit.md`).
- D0 calibration has resumable deterministic shards and full-campaign merge support. The default complete D0 schedule is 12 scenarios, 466 cases, 139,800 records (`docs/implementation-provenance/d0-campaign-orchestration.md`).
- Matched-reference calibration is implemented for exactly one preselected `D1/current`, `D2/cv`, or `D2/multi_model_v1` alongside D0, over the same repeated seed tables. Its current runner executes the whole scenario calibration in one process and has no shard/resume workflow (`docs/implementation-provenance/matched-reference-variance-calibration.md`). It binds K=16, the default route-cost weight, 12 action ticks, source and runtime identities, and all required safety metrics.
- The live r4 run is diagnostic and owns the two WX9100 device leases. Its exposure is not a calibration split and cannot be reused as calibration evidence. The run must remain untouched. The active source does not have a qualification binding or warm starts; no readiness or qualification claim follows from its progress.

### Selected bounded tranche

Implement and independently review **restart-safe, deterministic matched-reference calibration shards and merge** before launching the full calibration. Reuse the existing D0 calibration case/seed construction and immutable shard pattern, while preserving the matched-reference contract: each accepted row must bind the selected reference and D0 to the exact same `(scenario, case, repetition, seed)`, source, protocol, predictor mode, K, route-cost weight, cadence, normalized runtime, device attestation, and safety-metric schema. The merge must reject omitted/duplicate/unexpected assignments, mixed reference modes, source/runtime/config drift, missing safety rows, and any overlap with train/validation/test. Publication must be immutable/atomic and no-clobber. Keep `D0` and selected-reference reports distinct in the final bundle.

This is the highest-value bounded code tranche because the existing matched-reference path is monolithic across the intended 12-scenario closure: an interruption can discard a long run, while D0 already has a resumable shard path. This tranche does not change variance formulas, operational thresholds, qualification gates, D0 campaign semantics, policy behavior, Phase 4 work, or the approved legal-start profile plan. The current `matched-reference-variance-calibration.md` implementation remains the source contract; the new shard layer must preserve its existing legacy report/protocol readers and exact report validation.

### Run plan after the code tranche is reviewed

1. **Lease and source gate.** Wait until r4 is terminal and both exclusive WX9100 leases are demonstrably released. Do not stop, pause, reconfigure, or compete with r4. Reconfirm a clean, pinned checkout and HIP runtime on both physical WX9100s; bind each shard manifest and worker to the exact indexed device, loaded native-extension identity, source hash, and normalized runtime. If the requested GPU-only hot path is not met, do not launch and do not describe this calibration as satisfying that requirement.
2. **Freeze the statistical target.** Use `full_match_offense`, whose transitive closure is all 12 registered scenarios. Keep the existing calibration seed partition and master seed `20261024`, `episodes=100`, `repetitions=3`, with four declared cases per scenario (466 cases total). Do not regenerate or hand-edit seed tables. Confirm disjointness from all train, validation, and final-test assignments before scheduling. Use new immutable output roots namespaced to the exact reviewed source/runtime; never overwrite the existing `pickup_simple` report or its shards.
3. **Complete D0 first.** Resume/execute the existing 8-shard-per-scenario D0 campaign, merge each scenario report, then validate the immutable 12-scenario bundle. Preserve raw record coverage and provenance. Any rejected shard or changed source/runtime requires a new namespaced campaign rather than mixing outputs.
4. **Preselect one matched reference.** Before seeing variance outputs, record a short reference-selection decision for exactly one supported variant (`D1/current`, `D2/cv`, or `D2/multi_model_v1`) based on the already-defined deterministic-control contract and intended primary comparison. Do not choose the reference by whichever has a favorable result on these calibration seeds. Bind it in every shard; variant mixing is a hard stop.
5. **Collect repeated matched-reference evidence.** Run the new reviewed sharded runner for the same 466 calibration cases, exact seeds, 100 episodes per stratum and three repeats. Each assignment must evaluate the selected reference and D0 on the same case/seed and record useful throughput plus every declared safety metric. Balance shards across both pinned WX9100s only where normalized runtime equivalence is validated; keep shard-to-device provenance. Resume only digest-matching partial shards.
6. **Bundle and review margins explicitly.** Validate all 12 reference reports and the immutable bundle against the D0 bundle, seed-table hashes, source/runtime and reference identity. A named reviewer reviews per-case means, between-seed and within-seed repeat variance, uncertainty, tails, and every safety metric. The reviewer records the minimum useful-gain margins and safety non-inferiority margins in a separate immutable review artifact, with rationale, identity, timestamp, and exact report IDs/hashes. Do not calculate or auto-fill thresholds from a variance formula, and do not substitute statistical detectability for operational usefulness.
7. **Create and freeze v4.** Only after the explicit margin review is accepted, create and validate a `full_match_offense`-scoped v4 protocol binding the D0 and matched-reference bundle digests, source/runtime, reference variant, seed assignments, and reviewed margins. Keep the train/validation/final-test partitions unchanged and disjoint. The protocol's validation is a calibration-readiness gate, not evidence that a learned track passes the primary outcome gate.

### Device-dependent steps and hard stops

The following must wait until r4 releases both WX9100 leases: native HIP preparation and attestation; all D0 shards not already backed by valid current-source/runtime artifacts; every D1/D2 matched-reference episode; any empirical variance calculation dependent on new rows; and any simulator behavior/replay work. No calibration evaluation may fall back to CPU, a P600, a single card, synthetic data, or stale r4 records. Offline planning, implementation, review, schema inspection, and artifact validation can proceed while r4 runs, but they are not empirical calibration.

Stop the tranche if either card identity differs, the native extension is unprepared or changes mid-campaign, source or normalized runtime identities drift, calibration seeds overlap a held-out split, a required safety metric is missing, D0/reference pair coverage differs, or the requested GPU-only algorithmic path is not demonstrated. Keep protocol status provisional and all qualification/promotion gates false until corrected evidence is independently reviewed.

### Acceptance evidence

- Independent reviewer approval of the exact shard/merge implementation commit and its artifact invariants.
- Immutable D0 report per scenario and validated `full_match_offense` D0 bundle; exact expected 12-scenario closure and 139,800 raw records.
- Immutable matched-reference report per scenario and validated bundle with exact paired assignment coverage, chosen reference identity, all required metrics, and matching D0 seed-table hashes.
- Explicit margin-review artifact tied to those report and bundle hashes; no auto-derived thresholds.
- Immutable v4 protocol validates for `full_match_offense`, remains disjoint from train/validation/test, and records the accepted calibration state.
- Qualification remains false until the separate held-out five-track/control evaluation, Phase 4 outcome gates, safety/reliability gates, and deployment requirements are satisfied.

### Three-pass boundary

- **Planner:** this document scopes the bounded calibration-resilience tranche and post-review empirical sequence from current repository evidence.
- **Implementer:** not started. Expected scope is matched-reference shard planning, execution/resume, merge, and immutable bundle plumbing only.
- **Reviewer:** not started. Must inspect the exact implementation head and verify preservation of calibration bindings and failure-closed coverage rules.

No tests, simulator, GPU, training, replay, dashboard operation, D0 campaign, matched-reference calibration, operational-margin choice, or v4 protocol creation was performed for this planner pass. This is a plan, not calibration evidence.
