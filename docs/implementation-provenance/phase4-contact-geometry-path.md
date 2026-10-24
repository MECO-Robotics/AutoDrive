# Phase 4 diagnostic contact geometry path

## Planner pass

The source audit found that registered robot-pair contact geometries existed
only as helpers: matrix conditions did not carry them into evaluator reset,
and the simulator's pair-contact record was separately opt-in. The scoped path
wires geometry identity through matrix, evaluator, and reset; applies a
sensed-target approach only to slot 2 while preserving slot 3's evaluated
action; and records route crossing separately from solver-confirmed contact.
The current records do not support authoritative post-contact recovery time.
Recovery and route-completion outcomes therefore remain unavailable, and this
path must remain qualification-ineligible.

## Implementer pass

Candidate implementation commit:
`1d4e4eb15562fb7b490310394a3f4f832e95a1b9`.
Changed files are `frc_defense/curriculum_scenarios.py`,
`frc_defense/meco_ablation_matrix.py`,
`frc_defense/meco_phase4_context.py`,
`frc_defense/meco_phase4_runtime_adapter.py`,
`frc_defense/meco_track_training.py`, and
`tests/test_meco_contact_geometry_contract.py`.

Matrix conditions bind the registered geometry config, split gate, and exact
category/seed coverage. Reset applies the checked geometry before the final
perception refresh and metric baseline. Only slot 2 uses the sensed-target
approach action; slot 3's selected policy action remains unchanged. Records
distinguish assigned exposure and action ticks from the simulator SAT solver's
pair (2,3) contact summary and the physical route-crossing bit/time. The
matrix requires all 11 evaluated policy rows, including Plan A `P=0` variants
and D0/D1/D2, for every expected category/seed pair. Empty, missing, duplicate,
unexpected, or mismatched rows fail closed. Qualification remains false and
final-test use is blocked.

## Independent reviewer pass

The reviewer first requested changes because the initial pairing validator
accepted incomplete policy subsets and omitted category/seed pairs. The
corrected exact-head diff was independently approved at
`1d4e4eb15562fb7b490310394a3f4f832e95a1b9`. The reviewer confirmed the
11-policy set, full category/seed Cartesian coverage, slot isolation, reset
ordering, route-crossing timing, truth-only harness identity check, and
fail-closed qualification behavior.

## Checks and limits

Eleven focused CPU contract tests passed; Python compilation and
`git diff --check` passed. No simulator behavior, GPU/HIP compilation,
campaign, replay, or dashboard validation ran. Actual contact geometry and
action behavior remain unverified in simulation. Contact output is a bounded
first/last/count summary, not a complete event stream. `route_seen` is a
separate crossing observation, not route completion. Recovery remains
unavailable because no reviewed route-progress metric or contact-aligned
per-tick trace exists. This tranche does not make any Phase 4 cell executable
for qualification.
