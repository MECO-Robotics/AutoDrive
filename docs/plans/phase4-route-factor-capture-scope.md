# Phase 4 route outcome capture scope

## Planner pass

Wire the evaluator's existing route-goal lifecycle record into the generic
Phase 4 capture for stationary blockage, feint reversal, and abrupt opponent
policy change. A route observation may be included only when every lifecycle
has a valid observed completion or registered failure, with exact episode,
source digest, policy, factor, and condition binding. Preserve censoring for
missing outcomes, timeout, abandonment, replacement, and gaps. Do not use the
selected-goal endpoint proxy, change required-signal authority, or make any
qualification gate pass. Contact remains on its separate record path. Do not
interrupt campaign training, acquire GPUs, or run simulator work in this
tranche.

## Review criteria

- Reconcile aggregate completion/failure counts and rate with individual
  lifecycle rows.
- Reject malformed endpoint events, failure events, and provenance bindings.
- Keep censored/incomplete outcomes absent and all qualification flags false.
- Run offline focused contract tests and static checks only; do not claim
  simulator validation or a replay.
