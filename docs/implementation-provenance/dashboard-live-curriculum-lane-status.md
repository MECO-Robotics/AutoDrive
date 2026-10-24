# Dashboard live curriculum and lane status

## Planner

The candidate dashboard did not project the current scheduler assignment and
category onto its five track cards. Its queue parser also needed exact
five-lane validation for both native list and keyed-dictionary forms. The
planner required assignments to be bound to a fresh queue and matching
scheduler snapshot, with malformed or incomplete data failing closed. A
follow-up review found that `deferred_for_evaluation` could be mislabeled as
active; deferred lanes must remain queued and must not inherit an active
sibling's assignment.

## Implementer

Implemented in commits `d63bdf1bb83696ae8694c726756d5e9254fceb9d` and
`db918fc12744c02ff513ed25afc4a64ac86b67b3` on the isolated candidate branch.
The queue parser validates exactly one record for A1, A2, A3, C1, and C2,
supports both list and keyed-dictionary schemas, and rejects malformed lane
records and key/track mismatches. The API projects a validated live
scenario/category only onto running or stalled lanes. Deferred lanes remain
queued; stale or unmatched snapshots use an explicit run-config fallback.
The frontend labels the assignment source and retains saved run/scenario
evidence.

The implementer reported 34 focused tests for the first commit and 38 after
the deferred-lane corrections, plus compilation and diff checks.

## Reviewer

The independent reviewer approved exact head
`db918fc12744c02ff513ed25afc4a64ac86b67b3`. The reviewer confirmed the
deferred-lane fix, fail-closed queue/scheduler matching, and preservation of
saved evidence. The reviewer reported 38 focused dashboard tests passed.

## Limits

This is a summary of the scoped planner, implementation, and review reports;
raw agent transcripts were not persisted. It records static dashboard
contracts and focused tests only. The candidate dashboard service was not
restarted or endpoint/asset-verified by this tranche. No simulator, replay,
training, calibration, or qualification result is claimed.
