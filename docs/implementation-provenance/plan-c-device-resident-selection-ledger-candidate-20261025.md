# Plan C provenance ledger candidate (checkpoint D)

This is the scoped implementation summary for ledger candidate D, based on the
approved provenance plan `docs/plans/meco-plan-c-route-device-residency-ledger-currentness-20261025.md`
(SHA-256 `9356f7ae33550c3a7571d864dcb14854cfdeacc91b5556e5bb350d3526f03238`).
It adds one pending Plan C ledger row while preserving all original 88 rows.

The row binds code candidate C (`cdeb9391bf23e5c2460527959132008aee6f8eb1`),
its exact-head approved code review, and its six implementation paths. D's
ledger-review verdict remains pending: the row has no `candidate_commit`, no
`reviewer.approved_commit`, and no approved D reviewer artifact. The dashboard
API digest matches D's ledger bytes, but the validator is expected to reject
this pending row and return an untrusted result. This checkpoint does not
claim the ledger is valid or servable.

D changes these provenance/status files separately from C's implementation
paths:

- `docs/MECO_THREE_PASS_PROVENANCE.json`
- `frc_defense/dashboard_api.py`
- `tests/test_meco_three_pass_provenance.py`
- `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`
- `docs/implementation-provenance/plan-c-device-resident-selection-20261025.md`
- `docs/implementation-provenance/plan-c-device-resident-selection-ledger-candidate-20261025.md`
- `docs/implementation-provenance/plan-c-device-resident-selection-currentness-20261025.md`
- `docs/plans/meco-plan-c-route-device-residency-ledger-currentness-20261025.md`

No D SHA is recorded in this artifact or in its ledger row. D awaits an
independent exact-head Pass 3 ledger review before a separate E finalizer can
record the approval.
