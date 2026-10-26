# P0 Phase 0 preregistration and finalization package plan

**Date:** 2026-10-26  
**Status:** Pass 1 planner-authored candidate; pending independent exact-head Pass 1 review  
**Planner identity:** `plan_a_owner` (planner only; not implementer, reviewer, or human approver)  
**Overall plan commit:** `5d02eab99738cefa0318d4af4018c0450ecca685`  
**Overall plan file:** `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md`  
**Overall plan file SHA-256:** `c7e6d12c4c9e6c9d48d05015e0eb8eac91fe9b42c26e49a52fb81e52c1a4a37c`  
**Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`  
**Accepted reconciliation commit:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`  
**Reconciliation artifact SHA-256:** `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`  
**Reconciled profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`  
**V4 envelope digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Objective and authority

Define the bounded P0 work needed to preregister Phase 0 rules and protocol before source implementation (P0a), then produce final calibration and immutable source-bound Phase 0/protocol artifacts only after all relevant P1–P6 source is frozen and reviewed (P0b). P0 provides inputs to the five-track MECO A1/A2/A3/C1/C2 implementation and subsequent Stage 1 qualification.

This document is a package plan, not a protocol decision approval, source change, test authorization, calibration/run instruction, training authorization, ABI adoption, or human approval. It does not satisfy G0. The active diagnostic campaign PID `2750253` and its dashboard, workers, checkpoints, outputs, devices and leases remain outside scope. Do not inspect, stop, signal, reconfigure, evaluate through, restart, or otherwise touch them. No CPU/GPU work or dashboard operation is authorized until the user manually stops that campaign and lease release is independently verified. Implementation remains withheld until G0 and independent acceptance of this package plan.

The overall plan's sequencing governs: P0a ACCEPT precedes source-affecting P1–P6; P0b depends on P1–P6 source freeze and review, including P6's qualification-runner source; P6 runtime qualification depends on P0b ACCEPT. P0b never waits for P6 runtime qualification results. This removes the P0/P6 cycle.

## Evidence inspected and limits

Read-only inspection was performed against the implementation checkout at the pinned source base. Relevant direct source evidence:

- `frc_defense/meco_qualification.py:28-40` declares protocol versions v1 through v4, `MIN_EPISODES=100`, a 100-item training seed pool and five policy codes. This is not the reconciled target-scoped v5 protocol. The calibration validator at lines 214-240 checks at least 100 episodes and at least two repetitions; this is evidence of current v4 requirements, not approval to carry them into v5.
- `frc_defense/meco_qualification.py:241-280` checks calibration cases, repeated raw rows and source identity. Runtime-device identity checks are at lines 282-298. The `calibrate-d0` command requires an explicit indexed GPU argument and defaults to 100 episodes/3 repetitions at lines 5044-5051. This does not itself bind a two-WX9100 pair or prove all numerical work stays GPU-resident, so the v5 runner and P0b evidence must enforce and demonstrate both.
- Target and prerequisite scoping is implemented in current v4 protocol construction and validation through `protocol_cases` and `_scenario_closure` (see `frc_defense/meco_qualification.py` protocol builder/validator and `frc_defense/meco_stage1_preflight_contract.py`). Existing target closure must be compared with the accepted reconciliation's required v5 semantics; it is not presumed equivalent.
- `frc_defense/meco_task_scheduler.py:16-18` imports the qualification seed-pool size and case builder. Its assignment validation at lines 90-97 derives and checks seeds from the frozen protocol pool. This is a current coupling to qualification protocol APIs and needs a compatibility plan in P0 Pass 2.
- `frc_defense/gamepiece_actions.py:407-429` computes active HUB scoring eligibility, the accepted FUEL award pulse (`fuel_scored_event`) and cumulative `fuel_score_count`, then records awards. `frc_defense/tensor_3v3.py:1462-1476` computes red/blue score summaries and returns `fuel_scored_event` and `team_score` in step info. The reconciliation estimand is accepted active-HUB FUEL awards only.
- `frc_defense/curriculum_metrics.py:90-113` accumulates task reward, staged/team conversions and related curriculum metrics. It is not the canonical simulated match score estimand and must not be substituted for the scoring event.
- The source checkout does not contain `frc_defense/meco_simulated_score_ledger.py`. The real event producer/consumer map is thus drawn from existing event tensors and score counters, and P1 owns typed V4 event-table implementation and confirms the canonical event sites on its exact source base.
- Existing docs `docs/MECO_QUALIFICATION.md` describe earlier v1/v2 qualification behavior. They are historical compatibility documentation and must be updated for v5; the new protocol must preserve old records without treating them as qualification evidence.

These citations describe observed source only. They do not prove GPU residency, current dashboard freshness, successful simulator behavior, protocol validity, calibration quality, or Stage 1 passage. Exact line references and source identity must be refreshed by the implementer against the accepted integration base.

## Package split and dependency graph

```text
G0 (plan acceptance + manual campaign stop + independently verified lease release
    + explicit maintainer authorization + reviewed contract-anchor bridge)
  └── P0a preregistration (rules, schema, cases/seeds, statistical method)
       └── P1–P6 source packages and integration (all exact-head reviewed)
            └── P0b calibration + immutable Phase 0 artifact + final v5 protocol
                 └── fresh source-bound checkpoints and P6 runtime Stage 1
```

### P0a: preregister before source work

P0a is documentation and configuration specification only. It defines the stable decisions required by source implementers:

1. **Outcome/rules:** `simulated_match_score` means accepted active-HUB FUEL awards, with eligibility and acceptance rules traced to the simulator's authoritative producer. Explicitly distinguish curriculum/task reward, throughput, score differential and official match points.
2. **Protocol v5 schema:** target ID, transitive prerequisite list, case identity (category/layout/opponent family/start/sensor profile as applicable), seed partition, metric IDs and definitions, run/source/profile/device identities, uncertainty and margin fields, promotion predicates, required artifacts, approval provenance, integrity/hash rules, fail-closed cases and versioning policy.
3. **Target/case/seeds:** record the authorized Stage 1 target set and prerequisite closure; enumerate the case matrix and deterministic case/seed generation; freeze disjoint train, calibration, validation and final-test partitions as canonical bytes. The exact targets must be an explicit product/maintainer decision; do not infer a target from the currently selected scenario.
4. **Calibration method:** preregister D0 controls, repetitions, minimum sample policy, uncertainty calculation, useful-gain and safety noninferiority margin methods, missing-data/empty-case behavior, and the decision process. P0a locks the method and evidence floor, but not numeric margins unsupported by data. Numeric margins are computed in P0b from the preregistered evidence, never selected after candidate results.
5. **Identity boundary:** specify canonical source/config/rules/profile/device identities, artifact/protocol serialization and hash boundary, immutable output naming, signatures/approval record fields and stale-input failure rules. Hashes do not substitute for authorized human decisions.

P0a acceptance enables P1–P6 source implementation only. It produces no calibration result, final Phase 0 artifact, protocol instance, training run or qualification result.

### P0b: calibrate after all relevant source is reviewed

After P1–P6 source and integration commits are frozen and exact-head reviewed, P0b must:

1. recompute the frozen source/rules/profile/device identity and verify it matches P0a's declared source inputs and P1's reviewed score producer/consumer map;
2. run only preregistered D0 calibration cases/seeds/repeats on both assigned WX9100 devices, with no CPU numerical fallback, retaining complete immutable raw records;
3. recompute the preregistered variance/uncertainty and margins, validate seed partition disjointness, and record any required human decision separately;
4. emit the immutable Phase 0 artifact and target-scoped v5 protocol bound to the exact P1–P6 integration source, reconciled profile, device pair, scoring rules, cases/seeds, calibration, margins and approvals; and
5. pass independent exact-head review of hashes, recomputation, completeness, source binding and approvals before fresh training or P6 qualification.

Any relevant source/rules/case change after P0b invalidates its source binding. Re-freeze/review the source and regenerate P0b artifacts before any fresh training or qualification. No old protocol or calibration output is overwritten.

## Finite future Pass 2 allowlist

Pass 2 starts only after (a) this package plan receives independent Pass 1 ACCEPT, (b) G0 is evidenced, (c) P0a decisions and required human decisions are accepted, and (d) the implementer and independent reviewer are named and distinct. Use a fresh ZFS-backed worktree based on the reviewed G0 contract-anchor integration commit; proposed worktree `.worktrees/autodrive-v4-p0-phase0` and branch `codex/v4-p0-phase0-YYYYMMDD`.

The finite P0 source/document/test allowlist is:

- `frc_defense/meco_qualification.py` — sole P0 owner for v5 Phase 0/protocol parsing, schema validation, source/device binding and calibration runner integration. P6 may add its own qualification supervisor/report modules but must consume this API; it does not edit this file.
- `frc_defense/meco_task_scheduler.py` — only if the v5 seed/case API requires updating scheduler binding; preserve stable task assignments and fail closed on protocol identity mismatch.
- `docs/MECO_QUALIFICATION.md` — document v5 semantics and legacy compatibility.
- `tests/test_meco_qualification_v5.py` — new protocol, artifact, source/device binding and calibration-schema tests/fixtures.
- `tests/test_meco_task_scheduler_v5.py` — new seed-pool and assignment binding tests, only if scheduler code changes.

Generated immutable output paths (not source-controlled and not included in commits):

- `qualifications/meco/v5/<protocol-id>/phase0-artifact.json`
- `qualifications/meco/v5/<protocol-id>/protocol.json`
- `qualifications/meco/v5/<protocol-id>/calibration/<case-id>/<unique-run-id>/...`

Do not overwrite existing `qualifications/meco/protocol.json`, old reports, checkpoints, output directories or campaign data. If implementation proves one additional source path is unavoidable, stop and request a new exact-head Pass 1 amendment before editing it. P0 does not own or edit `gamepiece_actions.py`, `tensor_3v3.py`, `curriculum_metrics.py`, `meco_metric_ledger.py`, `meco_track_training.py`, or the supervisor/report paths owned by P6.

## Pass 2 delivery and commit cadence

The implementer records source base, worktree and branch; produces P0a decisions as small reviewable commits; commits/pushes each coherent schema/interface/test slice before dependent work; and records exact commit and blob hashes. P0a and P0b are separately gated deliverables. Do not run tests in P0a. After P0a source implementation and P1–P6 source freeze, P0b runtime calibration is a new separately authorized slice and cannot start before G0 remains satisfied and device leases are independently available. Generated outputs stay outside Git. Any amendment to allowlisted paths requires Pass 1 review before implementation.

## Validation and acceptance evidence

### P0a

- Exact P0a schema and canonical case/seed fixture bytes match the accepted preregistration.
- The explicit target set and prerequisite closure are complete and accepted.
- Split disjointness and deterministic generation are checked using fixtures, not candidate outcomes.
- Metric names, score-estimand rules, D0 controls, uncertainty/margin method, empty/missing behavior and promotion predicates have no unspecified fields.
- Protocol/Phase 0 hash boundary, source identity and invalidation rules are documented and independently reviewed.
- No calibration or candidate scores were used to tune method/threshold choices.

### P0b

- P1–P6 exact integration source is frozen and independently reviewed; score producer and all consumers are cited to that exact source.
- Tests validate v5 schema, target closure, complete exact case/seed matrix, seed disjointness, identity/hash binding, stale-input failure, legacy rejection for Stage 1, and immutable output behavior.
- Calibration is GPU-only on both pinned WX9100 devices. Retain device UUID/runtime/kernel/residency evidence and prove no CPU numerical fallback. CPU process supervision and file I/O remain allowed.
- Recompute results from complete retained calibration rows; record scenario/case IDs, seeds, repetitions, source/profile/protocol IDs, device IDs, output hashes, computed uncertainty/margins and authorized approval record.
- Any test that checks simulated scoring or other game behavior must be accompanied by a representative seeded end-to-end simulator evaluation per repository `AGENTS.md`; inspect and retain its result/replay provenance.
- Before generating/reviewing a dashboard replay, restart dashboard serving services to the current integration checkout, verify backend and frontend asset freshness, active services and successful endpoints; create a fresh replay and verify its source provenance. Mark old-code replays outdated.
- Report tests, simulator scenario/seed/replay ID and result, dashboard freshness/endpoints, and device-residency evidence. If a required simulator or dashboard check cannot be completed, state the missing evidence; do not call validation complete.

No tests, simulation, calibration, dashboard operation, or device use has been run for this planning document.

## Risks and stop conditions

- **Target ambiguity:** v4 has a scenario DAG and target/prerequisite closure, but the desired target set requires an explicit authorized decision. Stop P0a if that decision is absent.
- **Score-source ambiguity:** no dedicated score-ledger module exists at the source base. P1 must confirm every accepted award source and consumer. Do not use curriculum reward, event counters of uncertain semantics or official-score claims as a substitute.
- **P0/P6 cycle:** avoided only by separating P0a from P0b and P6 source implementation from P6 runtime qualification. If P6 source scope changes metrics/rules, freeze it before P0b; any later relevant edit invalidates P0b.
- **Protocol compatibility:** scheduler currently imports qualification APIs. Pass 2 must preserve or explicitly migrate this interface in the finite allowlist. If the finite scope is insufficient, request plan revision.
- **Device fallback:** current qualification CLI has a CPU default and older protocol versions; neither can be reused as GPU-only v5 qualification. Device identity loss or unsupported numeric work is a gate failure.
- **Approval provenance:** agent-generated manifests and hashes are not human approval/signatures. Required decisions must be recorded by the authorized maintainer.
- **Live campaign/G0:** this planning work does not satisfy manual stop or lease release. No Pass 2 implementation or calibration can begin while those gates are open.

## Three-pass roles and disposition

- **Pass 1 planner:** `plan_a_owner` authored this plan and cannot review or implement its package.
- **Pass 2 implementer:** unassigned; must be distinct from planner and Pass 3 reviewer.
- **Pass 3 reviewer:** unassigned; must independently review exact pushed heads, path allowlist, protocol/source evidence, test results and (for behavior changes) simulator/dashboard evidence. Persist ACCEPT/REVISE against exact commit and blob hashes.

Planner disposition: **PENDING independent Pass 1 review.** This plan requests authorization only for the bounded future P0 package after acceptance and G0. It claims no human approval, no implementation, no successful validation, and no Stage 1 result.
