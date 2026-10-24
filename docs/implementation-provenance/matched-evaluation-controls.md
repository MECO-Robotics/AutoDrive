# Matched evaluation controls: three-pass summary

## Planner pass

The planner audited `MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`, the
matrix runner and summarizer, the qualification protocol, and replay records.
It found remaining controls for exact protocol-bound seed inputs,
condition/policy-specific replay capture, paired runtime/fallback diagnostics,
and single-use final-test release. The full plan still requires final-source
training and simulator, qualification, and deployment evidence.

## Implementer pass

The implementation adds exact seed-manifest and split/stratum validation;
condition/policy-specific replay selection included in matrix identity and
reports; episode-level end-to-end, decision, inference, simulator-macro, and
fallback diagnostics with paired bootstrap summaries; schema-5 matrix reports
with report-byte digest checks; and a protocol-scoped exclusive final-test
ledger. The ledger blocks supervisor launch and the canonical `train_meco()`
API for fresh, resume, and warm-start training after test consumption. The
matrix CLI requires an indexed accelerator and defaults to `cuda:0`; CPU is
rejected. Source identity is frozen only after native HIP extension
preparation, matching evaluator identity.

Implementer commits: `415ce63d457f076c112a25b648ad6f58e404ffde` and
`0db1e6436fbd753d82e49fbab02850d81fce1f07`. Integrated candidate commits:
`54eca564a` and `049c81e07`.

## Independent reviewer pass

The reviewer first rejected `415ce63d` because direct `train_meco()` calls
could bypass the supervisor-only test ledger. After the shared guard was added
at the training API boundary, the reviewer approved exact implementer HEAD
`0db1e6436fbd753d82e49fbab02850d81fce1f07`. The focused tests cover fresh,
resume, and warm-start rejection before output creation.

## Verification and limits

The implementer reports `py_compile`, `git diff --check`, and 78 focused
tests passing on the final implementation head. Reviewer spot verification
passed 10 matched-control tests and 9 supervisor-dispatch tests. No GPU,
simulator, dashboard, or active-training operation was run for this tranche.
No final-test ledger has been consumed. This tranche establishes evaluation
controls only; it supplies no held-out outcome, replay, qualification, or
deployment evidence. The integrated candidate is not the live training
checkout.
