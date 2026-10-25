# Phase 4 route outcome in generic factor capture

## Planner pass

Approved scope (`docs/plans/phase4-route-factor-capture-scope.md`):
connect evaluator route-goal lifecycle outcomes to the generic Phase 4 factor
capture for the route-capable, existing factors only: stationary
blockage, feint reversal, and abrupt opponent policy change. Accept an outcome
only when the entire episode's lifecycle denominator is observed and valid and
the episode, source digest, policy, factor ID, and condition exactly match.
Keep timeout, abandonment, replacement, gaps, and unavailable observations
censored; do not use the selected-goal endpoint proxy. Keep every qualification
flag false. Do not run campaign, GPU, simulator, or training work in this pass.

## Implementer pass

The generic factor capture now embeds a separate
`meco-phase4-route-outcome-observation-v1` diagnostic for the three supported
factor kinds when the runtime stimulus was realized and the route lifecycle
record is fully observed. It reuses the evaluator's existing per-factor route
record. Binding mismatches fail closed; missing, censored, or incomplete route
records are omitted. The observation carries the lifecycle rows and outcome
counts so the capture validator can recheck the complete denominator and
endpoint-event schemas. It remains explicitly non-authoritative and
qualification-ineligible; all other required Phase4 outcomes remain governed
by their existing availability contract.

The evaluator writes the same route record to the episode's existing
`phase4_route_completion_by_factor` field and the generic factor capture, so
the two outputs share the exact episode/source/factor/condition binding.

## Independent reviewer pass

The reviewer initially rejected `d40dd6a1c76195c80bfc51e35ef370575622cf64`
after reproducing acceptance of contradictory aggregate counts. The implementer
fixed count reconciliation and added a tampering contract case in
`fbbdd430f3d828b25e8e4007282dc6451b224aab`. Independent review approved that
exact head. The reviewer confirmed exact source/factor/event bindings,
fail-closed censoring, endpoint/failure event schemas and sources, and
qualification-ineligible diagnostic-only semantics.

## Verification and limits

Implementer and reviewer each ran the offline
`PYTHONPATH=. pytest -q tests/test_meco_phase4_runtime_adapter_contract.py`
contract suite: 23 passed. Python compilation and `git diff --check` passed.
The exact approved implementation head is
`fbbdd430f3d828b25e8e4007282dc6451b224aab`. No simulated behavior test or
simulator evaluation was run, so no scenario, seed, or replay is claimed. No GPU,
checkpoint, training, dashboard process, or qualification state was changed.
The r4 supervisor was observed active during this work; its training and
evaluation artifacts were not altered. The provenance integration refreshes the
dashboard API ledger digest. The code adds diagnostic capture only; it does not
make a Phase 4 cell executable or establish simulator validity.
