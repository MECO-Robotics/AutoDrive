# Durable manifest evaluation schedule

## Planner

The r5 handoff was rejected as a full-coverage guarantee because pending
evaluation jobs could be replaced or skipped after failures, and manifest
campaigns could not safely resume. The scoped requirement was a hash-bound,
ordered schedule that retains the exact case and five-track checkpoint bundle
until a verified result is recorded.

## Implementer

Feature commits `cf2b664b741443a67099fa4a2866f8f5b`,
`e21be58220310ee6551e499b6219431057bb8971`,
`cc0a9b456339cdfdcd40e117ccb02fb313cdc1a0`, and
`d6bcab27637dbc2b39240e95ca994224d8809f31` implement the durable queue,
manifest resume binding, report/status/checkpoint verification, first-poll
handling, and persisted failure-time retry delays. The queue retains failed
and interrupted cases at the same index and advances only after verified
artifacts and durable history. Exact implementation paths:

- `frc_defense/meco_evaluation_schedule.py`
- `frc_defense/meco_supervisor.py`
- `frc_defense/meco_track_training.py`
- `frc_defense/meco_dashboard.py`
- `frc_defense/dashboard_static/meco.js`
- `scripts/meco_protocol_transition.py`
- `tests/test_meco_continual_eval_schedule.py`
- `tests/test_meco_evaluation_schedule_journal.py`
- `tests/test_meco_training_manifest.py`

Integrated commits on `main` are `30367daa9`, `e28da846c`, `708c31033`, and
`53157b4f8`. Focused CPU validation passed 115 tests and 5 subtests, with 3
device-lease tests deselected because the active r4 campaign owns both WX9100s.
The schedule-specific restart and retry tests passed separately (20 tests).
Python compilation and `git diff --check` passed.

## Reviewer

Independent exact-head review approved implementation commit
`d6bcab27637dbc2b39240e95ca994224d8809f31`. The review verified fresh roots
are claimed after lease acquisition and before campaign state is written,
failure and restart recovery persist retry timestamps, a failed case remains
the queue head, and the cursor advances only after verified completion.

## Limits

No GPU, simulator, or replay evaluation was run for this dispatcher change.
The active r4 campaign was not touched. The pre-existing r5 manifest remains
bound to source `b02db09f330642ad5498b6e76752d1df582c9089` and must be
regenerated against the integrated source before launch. This tranche does
not establish complete validation coverage, training outcomes, qualification,
or deployment readiness.
