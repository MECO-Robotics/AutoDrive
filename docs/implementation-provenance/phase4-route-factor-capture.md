# Phase 4 route outcome in generic factor capture

## Planner pass

Approved scope: connect evaluator route-goal lifecycle outcomes to the generic
Phase4 factor capture for the route-capable, existing factors only: stationary
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

Pending independent review of the exact implementation commit. This artifact
does not claim reviewer approval.

## Verification and limits

The implementer ran the offline
`tests/test_meco_phase4_runtime_adapter_contract.py` contract suite, Python
compilation, and `git diff --check`. No simulated behavior test or simulator
evaluation was run, so no scenario, seed, or replay is claimed. No GPU,
checkpoint, training, dashboard, or qualification state was inspected or
changed. The code adds diagnostic capture only; it does not make a Phase4 cell
executable or establish simulator validity.
