# P0 Phase 0 preregistration and finalization package plan

**Date:** 2026-10-26
**Status:** Original package plan ACCEPT at `13dbd7f8f953995850e811202f7e9c5264c49dca`; the G0/source-authoring amendment at `c4f136b267f8e41e0baec53394f28fac28f16eb5` received REVISE (review commit `da3154744`, stale overall-plan status pin). Corrected successor and fresh exact-head Pass 1 review are pending.
**Planner identity:** `plan_a_owner` (planner only; not implementer, reviewer, or human approver)
**Accepted overall plan:** commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; exact Pass 1 ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`
**Overall plan file:** `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md`
**Overall plan file SHA-256:** `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`
**Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
**Accepted reconciliation commit:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`
**Reconciliation artifact SHA-256:** `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`
**Reconciled profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
**V4 envelope digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

**Product evaluation-plan reference:** `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`, source base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.

## Objective and authority

Define bounded P0a documentation that proposes Phase 0 rules, protocol requirements, cases/seeds, and calibration method before source implementation, then P0b runtime calibration and generated immutable artifacts only after relevant P1–P6 source is frozen and reviewed. P0 provides inputs to the five-track MECO A1/A2/A3/C1/C2 implementation and subsequent Stage 1 qualification. P0 owns no source or tests; P6 owns qualification v5 schema/runner implementation and qualification tests.

This document is a package plan, not a protocol decision approval, source change, test authorization, calibration/run instruction, training authorization, ABI adoption, or human approval. It does not satisfy G0. The active diagnostic campaign PID `2750253` and its dashboard, workers, checkpoints, outputs, devices and leases remain outside scope. Do not inspect, stop, signal, reconfigure, evaluate through, restart, or otherwise touch them. P0a is documentation-only and may be authored/reviewed in an isolated ZFS worktree after this package plan receives Pass 1 ACCEPT and the required owner is assigned; it does not require G0. No build/compile, test execution, numerical work, simulator, calibration, evaluation, dashboard, device/GPU use, training, checkpoint mutation, integration, merge or promotion is allowed until the user manually stops the campaign and both WX9100 leases are independently verified released. P0b remains withheld until G0, source freeze, and independent review.

The amended overall plan's sequencing governs. P0a Pass 2 produces a documentation-only candidate; authorized human decisions are resolved before P0a Pass 3 ACCEPT. P1–P6 may then begin only the source/test-file authoring allowed by their accepted package plans and prerequisites in isolated ZFS worktrees. Before G0 they may not build, compile, run tests, execute numerical work, use devices, integrate, merge or promote; dependent packages may not integrate/use candidate APIs until post-G0 validation and final ACCEPT. P6 owns qualification v5 schema/runner/source and related tests. P0b depends on P1–P6 source freeze and review, including P6 source; P6 runtime qualification depends on P0b ACCEPT. P0b never waits for P6 runtime qualification results. This removes the P0/P6 cycle.

## Evidence inspected and limits

Read-only inspection was performed against the implementation checkout at the pinned source base. Relevant direct source evidence:

- `frc_defense/meco_qualification.py:28-40` declares protocol versions v1 through v4, `MIN_EPISODES=100`, a 100-item training seed pool and five policy codes. This is not the reconciled target-scoped v5 protocol. The calibration validator at lines 214-240 checks at least 100 episodes and at least two repetitions; this is evidence of current v4 requirements, not approval to carry them into v5.
- `frc_defense/meco_qualification.py:241-280` checks calibration cases, repeated raw rows and source identity. Runtime-device identity checks are at lines 282-298. The `calibrate-d0` command requires an explicit indexed GPU argument and defaults to 100 episodes/3 repetitions at lines 5044-5051. This does not itself bind a two-WX9100 pair or prove all numerical work stays GPU-resident, so the v5 runner and P0b evidence must enforce and demonstrate both.
- Target and prerequisite scoping is implemented in current v4 protocol construction and validation through `protocol_cases` and `_scenario_closure` (see `frc_defense/meco_qualification.py` protocol builder/validator and `frc_defense/meco_stage1_preflight_contract.py`). Existing target closure must be compared with the accepted reconciliation's required v5 semantics; it is not presumed equivalent.
- `frc_defense/meco_task_scheduler.py:16-18` imports the qualification seed-pool size and case builder. Its assignment validation at lines 90-97 derives and checks seeds from the frozen protocol pool. This is a current coupling to qualification protocol APIs; P6's source package plan must include a compatibility decision and finite allowlist if the v5 API changes.
- `frc_defense/gamepiece_actions.py:407-429` computes active HUB scoring eligibility, the accepted FUEL award pulse (`fuel_scored_event`) and cumulative `fuel_score_count`, then records awards. `frc_defense/tensor_3v3.py:1462-1476` computes red/blue score summaries and returns `fuel_scored_event` and `team_score` in step info. The reconciliation estimand is accepted active-HUB FUEL awards only.
- `frc_defense/curriculum_metrics.py:90-113` accumulates task reward, staged/team conversions and related curriculum metrics. It is not the canonical simulated match score estimand and must not be substituted for the scoring event.
- The source checkout does not contain `frc_defense/meco_simulated_score_ledger.py`. The real event producer/consumer map is thus drawn from existing event tensors and score counters, and P1 owns typed V4 event-table implementation and confirms the canonical event sites on its exact source base.
- Existing docs `docs/MECO_QUALIFICATION.md` describe earlier v1/v2 qualification behavior. They are historical compatibility documentation and must be updated for v5; the new protocol must preserve old records without treating them as qualification evidence.
- `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md:98` requires freezing the baseline and dataset protocol and partitioning held-out defender policies and maps/start situations before training. Lines 106 and 112-122 specify adversarial/generalization cases, metric categories, and primary success: beat the strongest matched deterministic baseline on held-out match outcomes with confidence intervals and without increased safety violations, using paired initial states/opponent seeds and bootstrap confidence intervals. It says to measure baseline variance before predeclaring the minimum useful improvement. This defines the product target; P0a still needs a finite Stage 1 target-to-case mapping and explicit human decisions for held-out set and margins where the source leaves them open.

These citations describe observed source only. They do not prove GPU residency, current dashboard freshness, successful simulator behavior, protocol validity, calibration quality, or Stage 1 passage. Exact line references and source identity must be refreshed by the implementer against the accepted integration base.

## Package split and dependency graph

```text
Package-plan acceptance + assigned owner/reviewer
  └── P0a preregistration (rules, schema, cases/seeds, statistical method)
       └── required human decisions + P0a exact-head ACCEPT
            └── P1–P6 isolated source/test authoring; static candidate API review
                 └── G0 (manual campaign stop + both leases independently released
                      + authorization + reviewed contract-anchor bridge)
                      └── builds/tests/runtime validation + final ACCEPT + integration
                           └── source freeze/review + P0b calibration/artifacts
                                └── fresh checkpoints and P6 runtime Stage 1
```

### P0a: propose preregistration before source work

After this package plan's independent Pass 1 ACCEPT, P0a Pass 2 creates a documentation-only candidate preregistration in an isolated ZFS worktree. It needs no G0 because it performs no source edit, build, test, numerical work or runtime operation. It proposes the decisions needed by source implementers but cannot decide required product/rules questions on behalf of the authorized human maintainer:

1. **Outcome/rules:** `simulated_match_score` means accepted active-HUB FUEL awards, with eligibility and acceptance rules traced to the simulator's authoritative producer. Explicitly distinguish curriculum/task reward, throughput, score differential and official match points.
2. **Protocol v5 schema:** target ID, transitive prerequisite list, case identity (category/layout/opponent family/start/sensor profile as applicable), seed partition, metric IDs and definitions, run/source/profile/device identities, uncertainty and margin fields, promotion predicates, required artifacts, approval provenance, integrity/hash rules, fail-closed cases and versioning policy.
3. **Target/case/seeds:** map the product target—A1/A2/A3/C1/C2 evaluated against the strongest matched deterministic baseline on held-out match outcomes, paired states/opponent seeds, bootstrap confidence intervals, and no increased safety violations—to a finite Stage 1 protocol target and transitive prerequisite closure. Enumerate held-out defender families, maps/start situations, safety metrics, case IDs and deterministic seed generation; propose disjoint train, calibration, validation and final-test partitions as canonical bytes. The overall product target is established by the evaluation plan, but the exact finite held-out set and protocol-scenario mapping require authorized human decision before P0a Pass 3 ACCEPT.
4. **Calibration method:** preregister D0 controls, repetitions, minimum sample policy, uncertainty calculation, useful-gain and safety noninferiority margin methods, missing-data/empty-case behavior, and the decision process. P0a locks the method and evidence floor, but not numeric margins unsupported by data. Numeric margins are computed in P0b from the preregistered evidence, never selected after candidate results.
5. **Identity boundary:** specify proposed canonical source/config/rules/profile/device identities, artifact/protocol serialization and hash boundary, immutable output naming, approval-record fields and stale-input failure rules. Hashes do not substitute for authorized human decisions.

Before P0a Pass 3, the authorized maintainer resolves all required score/rules, finite held-out set, protocol-scenario mapping, case/seed and statistical-method decisions against the candidate. Pass 3 then reviews the exact candidate plus the recorded decisions. P0a ACCEPT enables only the source/test-file authoring stage for P1–P6, subject to each package's accepted plan and exact prerequisites. G0 remains mandatory before builds, test execution, simulator or other numerical work, device use, dashboard operation, calibration, evaluation, integration, merge or promotion. P6 owns qualification code. P0a produces no source, tests, calibration result, final Phase 0 artifact, protocol instance, training run or qualification result.

### P0b: calibrate after all relevant source is reviewed

After P1–P6 source and integration commits are frozen and exact-head reviewed, P0b performs runtime calibration and writes generated immutable artifacts only; it does not modify source or tests. P6 owns the schema and runner used for this operation. P0b must:

1. recompute the frozen source/rules/profile/device identity and verify it matches P0a's declared source inputs and P1's reviewed score producer/consumer map;
2. run only preregistered D0 calibration cases/seeds/repeats on both assigned WX9100 devices, with no CPU numerical fallback, retaining complete immutable raw records;
3. recompute the preregistered variance/uncertainty and margins, validate seed partition disjointness, and record any required human decision separately;
4. emit the immutable Phase 0 artifact and target-scoped v5 protocol bound to the exact P1–P6 integration source, reconciled profile, device pair, scoring rules, cases/seeds, calibration, margins and approvals; and
5. pass independent exact-head review of hashes, recomputation, completeness, source binding and approvals before fresh training or P6 qualification.

Any relevant source/rules/case change after P0b invalidates its source binding. Re-freeze/review the source and regenerate P0b artifacts before any fresh training or qualification. No old protocol or calibration output is overwritten.

## Finite future P0 Pass 2 outputs and boundaries

Pass 2 for P0a starts after this package plan receives independent Pass 1 ACCEPT and distinct P0a author/reviewer identities are assigned; G0 is not a prerequisite. P0a Pass 2 output is exactly one documentation-only candidate file: `docs/plans/v4-phase0-preregistration-candidate-20261026.md`, created in a fresh ZFS-backed worktree based on the accepted planning head. Do not add source or test files. Required authorized-human decisions are resolved and recorded against this candidate before its exact-head Pass 3 review; P1–P6 source-only authoring unlocks after that ACCEPT and each package's own plan/prerequisites, while all execution and integration remain G0-gated.

P0b is a separate, later operational action. It may begin only after reviewed P1–P6 source freeze and renewed G0 device/lease readiness. Its only outputs are generated calibration records and these immutable artifacts outside Git:

- `qualifications/meco/v5/<protocol-id>/phase0-artifact.json`
- `qualifications/meco/v5/<protocol-id>/protocol.json`
- `qualifications/meco/v5/<protocol-id>/calibration/<case-id>/<unique-run-id>/...`

These are generated outputs, not source commits. Do not overwrite existing `qualifications/meco/protocol.json`, old reports, checkpoints, output directories or campaign data. P0 owns no code paths. P6's separate package plan must carry the finite allowlist for v5 schema/runner implementation and tests, including `frc_defense/meco_qualification.py` and its qualification tests. P0 does not edit `meco_task_scheduler.py`, `docs/MECO_QUALIFICATION.md`, `gamepiece_actions.py`, `tensor_3v3.py`, `curriculum_metrics.py`, `meco_metric_ledger.py`, `meco_track_training.py`, or P6 supervisor/report paths.

## Pass 2 delivery and commit cadence

The P0a author records the exact plan base and commits only the candidate preregistration document. The authorized maintainer records required decisions; the independent reviewer records exact-head Pass 3 verdict. P0b is a separately scheduled and authorized runtime operation by the assigned operator using the reviewed P6 runner. It cannot start before P1–P6 source freeze, P0a ACCEPT, G0 and independently available device leases. Generated P0b outputs stay outside Git. Any request for P0 source/tests changes requires a new scoped Pass 1 plan; qualification code/tests belong to P6.

## Validation and acceptance evidence

### P0a

- Candidate preregistration document is complete; all required human decisions are recorded before P0a Pass 3.
- The finite Stage 1 target-to-case mapping and prerequisite closure are complete and accepted.
- Split membership, canonical bytes and deterministic seed-generation rules are fully specified before candidate outcomes; P0a performs no fixture execution or tests.
- Metric names, score-estimand rules, D0 controls, uncertainty/margin method, empty/missing behavior and promotion predicates have no unspecified fields.
- Protocol/Phase 0 hash boundary, source identity and invalidation rules are documented and independently reviewed.
- No calibration or candidate scores were used to tune method/threshold choices.

### P0b

- P1–P6 exact integration source is frozen and independently reviewed; score producer and all consumers are cited to that exact source.
- P6-owned tests validate v5 schema, target closure, complete exact case/seed matrix, seed disjointness, identity/hash binding, stale-input failure, legacy rejection for Stage 1, and immutable output behavior before source freeze.
- Calibration is GPU-only on both pinned WX9100 devices. Retain device UUID/runtime/kernel/residency evidence and prove no CPU numerical fallback. CPU process supervision and file I/O remain allowed.
- Recompute results from complete retained calibration rows; record scenario/case IDs, seeds, repetitions, source/profile/protocol IDs, device IDs, output hashes, computed uncertainty/margins and authorized approval record.
- Any test that checks simulated scoring or other game behavior must be accompanied by a representative seeded end-to-end simulator evaluation per repository `AGENTS.md`; inspect and retain its result/replay provenance.
- Before generating/reviewing a dashboard replay, restart dashboard serving services to the current integration checkout, verify backend and frontend asset freshness, active services and successful endpoints; create a fresh replay and verify its source provenance. Mark old-code replays outdated.
- Report tests, simulator scenario/seed/replay ID and result, dashboard freshness/endpoints, and device-residency evidence. If a required simulator or dashboard check cannot be completed, state the missing evidence; do not call validation complete.

No tests, simulation, calibration, dashboard operation, or device use has been run for this planning document.

## Risks and stop conditions

- **Finite Stage 1 mapping:** the product objective and five-track set are stated in the evaluation plan. The exact held-out opponent/map/start set and mapping into v5 target scenarios remain explicit human decisions; stop before P0a Pass 3 if unresolved.
- **Score-source ambiguity:** no dedicated score-ledger module exists at the source base. P1 must confirm every accepted award source and consumer. Do not use curriculum reward, event counters of uncertain semantics or official-score claims as a substitute.
- **P0/P6 cycle:** avoided only by separating P0a from P0b and P6 source implementation from P6 runtime qualification. If P6 source scope changes metrics/rules, freeze it before P0b; any later relevant edit invalidates P0b.
- **Protocol compatibility:** scheduler currently imports qualification APIs. P6's package plan must preserve or explicitly migrate this interface; P0 cannot edit scheduler or qualification source.
- **Device fallback:** current qualification CLI requires an explicit indexed GPU but does not bind both WX9100 devices or prove the full algorithmic path is GPU-resident; older protocol versions are not v5 qualification. Device identity loss or unsupported numeric work is a gate failure.
- **Approval provenance:** agent-generated manifests and hashes are not human approval/signatures. Required decisions must be recorded by the authorized maintainer.
- **Live campaign/G0:** this planning work does not satisfy manual stop or lease release. P0a document authoring may proceed after plan acceptance, but P0b calibration and all build/test/runtime/device/dashboard/integration/merge/promotion actions remain blocked until G0 is evidenced.

## Three-pass roles and disposition

- **Pass 1 planner:** `plan_a_owner` authored this plan and cannot review or implement its package.
- **P0a Pass 2 author:** unassigned; creates only the candidate preregistration document and must be distinct from planner/reviewer.
- **P0b operator:** unassigned; runs the reviewed P6 calibration runner and emits generated artifacts after source freeze.
- **Pass 3 reviewer:** unassigned; must independently review exact pushed heads, path allowlist, protocol/source evidence, test results and (for behavior changes) simulator/dashboard evidence. Persist ACCEPT/REVISE against exact commit and blob hashes.

Planner disposition: **REVISE received; corrected candidate pending fresh independent Pass 1 review.** This plan requests authorization only for the bounded future P0 package after acceptance and G0. It claims no human approval, no implementation, no successful validation, and no Stage 1 result.
