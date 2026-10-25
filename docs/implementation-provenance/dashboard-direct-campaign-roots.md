# Dashboard direct campaign roots

## Planner

The scoped planner found that `meco_status_payload` assumed only the legacy
`checkpoints/meco-five` and `evaluations/meco-five` layout, while the active r4
campaign stores track checkpoints directly under `checkpoints/{A1,A2,A3,C1,C2}`.
The plan required preserving legacy roots, resolving direct campaign roots,
reading queue and scheduler state from the campaign root, and failing closed on
partial, ambiguous, escaping, or symlinked layouts. The planner restricted the
work to the dashboard resolver, its focused tests, and the data-root guide; the
training campaign, service, and GPUs were out of scope.

## Implementation

Implemented on branch `codex/meco-dashboard-direct-campaign-root`, final scoped
commit `d23b22d20352d22eccaf5986c421a624335cb7db`. The implementation preserves
legacy resolution, recognizes a complete five-track direct layout, reads
campaign-root queue/scheduler state and direct evaluation records, and rejects
partial or mixed layouts, symlinked checkpoint/track/evaluation roots, and
resolved roots outside the selected data root. `docs/MECO_DASHBOARD_DATA_ROOT.md`
documents both supported layouts.

The implementer reports `python3 -m pytest tests/test_meco_dashboard.py -q` as
27 passed, plus Python compilation and `git diff --check`. Tests cover legacy
behavior, direct status/queue/evaluation projection, partial and ambiguous
layouts, and symlink rejection.

## Independent review

Reviewer approved exact head
`d23b22d20352d22eccaf5986c421a624335cb7db`. The review confirmed direct and
legacy behavior, symlink checks for checkpoint, evaluation, and track roots,
and documentation alignment. The reviewer did not rerun tests and did not
access the service, campaign, or GPUs.

## Scope limits

This is dashboard projection and path validation only. It does not establish
simulator behavior, training outcomes, a held-out evaluation, qualification,
or deployment evidence. Live dashboard verification is recorded separately
after integration and restart.
