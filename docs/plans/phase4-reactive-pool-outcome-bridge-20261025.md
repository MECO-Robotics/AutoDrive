# Phase 4 reactive opponent-pool outcome bridge

## Planner pass

**Scope:** make the existing `defender.reactive_pool` opponent assignment and `teammate.behavior_pool` schedule produce exact, replay-bindable family/behavior outcome rows and fail-closed reducers. This is one bounded Phase 4 evidence-interface tranche: both requirements need the same invariant (protocol-seeded assignment → observed applied behavior → paired per-arm outcome). It is code-only, enables later simulator evidence collection, and does not create that evidence or make any cell qualification-eligible.

**Planning base:** `main` / `origin/main` at `c7b3f97964e6bb0638f1e6e8091337b634fa82fd`. The root checkout has an unrelated modified generated HIP source file; the planner artifact is in an isolated HDD-pool worktree and does not touch it.

## Source requirements and current evidence

The source evaluation plan requires Phase 4 evaluation against reactive defenders and opponent-policy pools, with held-out policy families, paired conditions, policy-family coverage, interceptions, and outcomes (Sections 5 Phase 4, 6 Defensive robustness/Generalization, and 7 experiments 5–6). It also requires deterministic safety and common matched scenarios.

Current `docs/MECO_PHASE4_RUNTIME_STATUS.md` says `defender.reactive_pool` has no runtime event and lacks authoritative interception and paired family outcomes; for `teammate.behavior_pool` it says schedule wiring and behavior-specific outcome, collision, and score-contribution signals are missing. The live source shows teammate assignment/dispatch helpers already exist in `meco_phase4_context.py` and `meco_validation_campaign_runner.py`, but the assignment record leaves all required outcomes null; the status text is stale on the dispatch part. The status explicitly reports zero executable qualification cells. `meco_phase4_opponents.py` defines deterministic policy IDs/schedules; `run_phase4_reactive_cell` dispatches opponent-policy cells, but its result leaves `required_family_outcomes` null for reactive pools and only gives complete paired outcome validation to direct-pursuit and trench branches. `meco_phase4_runtime_adapter.py` has no authoritative runtime/signal mapping for either requirement. This is a concrete data-capture/reducer gap after the start-profile, direct-pursuit and trench-interception tranches.

## Exact implementation boundary

The implementer should inspect and, if needed, change only these interfaces plus their focused tests and status/provenance docs:

- `frc_defense/meco_validation_campaign_runner.py::run_phase4_reactive_cell` and the existing teammate validation dispatch/reduction path: preserve exact protocol case, split, seed list, five checkpoints, policy/behavior configuration, assignment digest, and paired control/treatment dispatch; add strict family outcome collection/reduction for both requirements.
- `frc_defense/meco_phase4_opponents.py::compile_opponent_policy_schedule` and `frc_defense/meco_phase4_context.py::phase4_teammate_behavior_record`: use canonical seeded assignment as the only source of policy-family/teammate-behavior identity; do not infer identity from model label or slot position.
- `frc_defense/meco_track_training.py` or the narrow Phase 4 capture module selected after tracing the actual record writer: persist applied policy IDs/slots per episode, assignment digest, realized stimulus, simulator-confirmed contact/interception events, and exact condition/seed/source identity. Do not add a second competing event ledger.
- `frc_defense/meco_phase4_runtime_adapter.py`: bind both runtime-event and signal-source declarations to actual capture fields and keep `qualification_eligible=false` until replay review.
- `frc_defense/meco_phase4_campaign_plan.py` and `meco_validation_campaign_runner.py`: require exact expected family × policy × seed × paired-condition coverage and fail closed on duplicates, missing/censored rows, policy/config/assignment mismatch, or unbound events.
- `tests/` focused Phase 4 opponent runner/capture/reducer tests and `docs/MECO_PHASE4_RUNTIME_STATUS.md` / provenance summary.

The plan authorizes no broad schedule rewrite. Preserve existing scenario-domain, registered pressure-slot, other-scripted-slot preservation, held-out split, and five-track pairing contracts. Exclude final-test seeds. Do not call selected-goal completion, route progress, aggregate contact frequency, or policy assignment itself an interception or outcome. Define episode-level interception only from simulator-confirmed focal/opponent contact events under an explicit preregistered criterion; report contact frequency separately. Keep actual paired score/outcome by family tied to the same scenario, seed, checkpoint set, condition, and assignment digest.

## Acceptance criteria

1. One frozen validation case compiles to deterministic reactive policy IDs, family IDs, slots and assignments, with a digest recomputable from the protocol and seed manifest.
2. The capture has exact per-episode provenance: scenario/category, validation seed, condition, source/checkpoint identity, policy/config/family, configured and applied slots, assignment digest, and event coverage/completeness.
3. Reducers reject missing/duplicate seeds, unexpected families/behaviors, absent applied-slot/action records, assignment drift, mismatched provenance, censored/incomplete capture, and contradictory aggregates. They report family/behavior coverage and paired outcomes without averaging away group counts. Teammate collision and score contribution must use authoritative simulator ledger sources, with unavailable values remaining null.
4. A complete synthetic fixture can exercise schema, binding and fail-closed logic offline. Such a fixture is explicitly labeled synthetic and cannot set runtime validation, replay review, or qualification gates true.
5. The runtime registry reports exact signal sources and keeps the cell diagnostic-only. `docs/MECO_PHASE4_RUNTIME_STATUS.md` distinguishes code support from simulator-validated status and retains zero executable qualification cells until a representative current-source paired simulator campaign plus fresh replay/dashboard review are accepted.
6. Existing direct-pursuit/trench/start-profile semantics and unrelated campaign rows remain unchanged.

## Reviewer focus and limits

Review the exact integrated candidate for arm/policy/behavior/seed cross-binding, hidden use of final-test seeds, family-count completeness, and whether claimed interception, teammate collision, and score-contribution metrics are truly sourced by simulator ledgers. Check that an empty or partial record cannot be promoted by a synthetic fixture or aggregate proxy. Confirm no lease/GPU/simulator/replay operation is included in implementation validation while r4 owns both WX9100s. Any later simulator behavior test must follow the repository instruction: current dashboard backend and assets, fresh replay, inspected outcome, reproducible scenario/seed, and recorded provenance.

This tranche advances Section 5 Phase 4 implementation readiness for two currently unwired factors. It does not complete Phase 4 evaluation, any of the five-track matched experiments, Phase 0 calibration, Phase 2 data/warm starts, deployment profiling, qualification, or model promotion. The immediate runtime campaign remains blocked until r4 exits and releases its leases; the current diagnostic training cannot be upgraded retroactively.
