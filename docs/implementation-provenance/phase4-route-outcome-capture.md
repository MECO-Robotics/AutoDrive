# Phase 4 route outcome capture — tranche #40

## Planner pass

The planner identified that the prior endpoint diagnostic counted selected-goal
macro rows rather than route-goal lifecycles, and that contact recovery had no
registered return-state criterion. The scoped route tranche adds evaluator-side
route lifecycle records, a complete/censored attempt accounting contract,
paired expected-lane coverage, and explicit missing treatment/support status.
It does not define contact recovery or infer observed failure from abandonment
or episode horizon.

## Implementer pass

The implementation began at source HEAD
`54219fece5ca0d3ba9820d77e38108a3a8f71ce4` and was committed as
`387a0ef0e5a50bb986af923800245e0ef965624d`. Its reviewed working diff had
SHA-256 `5664272c5dfac610cefbf486f14119e6ea9196d7888428befa299253df82610c`.
The implementation was integrated in candidate commit
`b8f30d9e5a92675d069a740cdca070d04bc330c5`.

The evaluator ledger binds goal attempts to macro indices, executed actions,
endpoint measurements, episode/source provenance, and a stable selected-goal
lifecycle identity. Plan A identity uses the selected destination node with
action/behavior/descriptor; per-macro candidate IDs do not define a lifecycle.
For target-bound actions 0–3, no durable target allocation generation exists,
so lifecycle identity remains unavailable even during assigned-goal
continuation. Rows without stable identity are reported as unsupported coverage
and excluded from the route-attempt denominator. Macro gaps, lifecycle
replacement, missing endpoint evidence, and episode end censor known attempts.
Only the existing evaluator-truth joint endpoint rule emits completion.

The matrix builds expected policy/factor/condition/category/case/seed lanes
independently of observed summaries. Unscheduled treatments, including opt-in
contact geometry, remain explicit missing lanes. The stationary blockage
condition preserves scheduled opponent behavior on other slots and holds only
its assigned slot. Campaign aggregation remains non-authoritative and
non-qualifying.

## Independent reviewer pass

The exact reviewed source diff was SHA-256
`5664272c5dfac610cefbf486f14119e6ea9196d7888428befa299253df82610c` at source
HEAD `54219fece5ca0d3ba9820d77e38108a3a8f71ce4`. Review corrections required
stable lifecycle identity distinct from per-macro route candidates, censoring
on lifecycle switches and macro gaps, independently generated expected lanes,
explicitly missing unscheduled treatments, evaluator endpoint revalidation,
and fail-closed handling when target identity or provenance is unavailable.
The approved implementation is commit
`387a0ef0e5a50bb986af923800245e0ef965624d`, carried by candidate integration
commit `b8f30d9e5a92675d069a740cdca070d04bc330c5`.

## Verification and limits

Before the implementation commit, 41 focused route/runtime contract tests
passed. On the integration candidate, 50 tests passed across
`test_meco_selected_goal_endpoint_contract.py`,
`test_meco_phase4_runtime_adapter_contract.py`, and
`test_meco_execution_failure_taxonomy.py`; Python compilation and
`git diff --check` passed.

No simulator evaluation, replay, GPU/HIP run, training, dashboard restart, or
qualification was performed. Contact recovery remains unavailable until an
objective return-to-route-following criterion is specified. Target-bound route
lifecycles remain unsupported without a durable target allocation generation;
D0 has no selected-goal ledger. Route outcome aggregation is not authoritative
and qualification eligibility remains false.
