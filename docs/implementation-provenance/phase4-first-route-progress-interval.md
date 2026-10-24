# Phase 4 first-route-progress interval observation

## Planner pass

The scoped request was to add a contact-window-only, outer-tick
interval-censored first-route-progress observation for D0's direct loop and
macro execution. Samples carry simulator pair identity/count and latest solver
timestamp, slot-3 truth position, pinned goal, validated executable route,
projected remaining route distance, and a route fingerprint. Later contact
resets the window. Changed goals, invalid routes or clocks, non-monotonic outer
time/ticks, contact-count regressions, and invalid or regressing solver
timestamps/substeps censor the observation with a null reason. This diagnostic
is not recovery time or qualification evidence.

## Implementer pass

The integration commit is `9ee3369eb816c187e7ecd5fecfde520f89be0f78`
(`Capture censored route progress after contact`). It changes
`frc_defense/meco_phase4_runtime_adapter.py`,
`frc_defense/meco_track_training.py`, and
`tests/test_meco_phase4_runtime_adapter_contract.py`. Route endpoint tolerance
uses `planner.resolution * 0.72`. Contact solver time supplies the interval
lower bound; route-distance samples are bounded by outer ticks.

## Reviewer pass

Review requested strict route-distance decrease, solver-time lower-bound
semantics, monotonic outer-tick and contact-count checks, solver timestamp and
substep monotonicity, invalid-count censoring, planner-resolution route
tolerance, and route fingerprints. These corrections are included in the
exact integration commit above. The accepted scope remains diagnostic and
fail-closed; route progress does not establish recovery or qualification.

## Checks and limits

The focused runtime-adapter contract file passed 20 tests. Python compilation
and `git diff --check` passed. No simulator, GPU/HIP, training, campaign, or
replay validation was run. The trace is interval-censored at outer ticks and
does not claim exact substep pose, recovery time, route completion, or
qualification. `recovery_time_after_bump_s` and authoritative signal status
remain unavailable; qualification remains false.
