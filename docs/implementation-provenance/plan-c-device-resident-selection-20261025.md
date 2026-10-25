# Plan C device-resident route and selection slice

## Three-pass record

- Approved Pass 1 plan: `docs/plans/gpu-resident-route-decision-slice-20261025.md`
  (SHA-256 `8c6c67ffb2934fa3a4ce40b54c5cc2f2e2cb4877b5eedac9c31a2936a6879b5a`).
- Implementer candidate and exact-head Pass 3 approval:
  `cdeb9391bf23e5c2460527959132008aee6f8eb1`.
- Candidate integration commit: `e87d757cb70262b523a4342315695bf600cf24c7`.
  This record was added on a separate provenance branch after that integration.
- Implementation history is recorded in the ledger entry's
  `implementation_commits`, from the initial route-reuse audit through the
  approved candidate. The candidate's six changed paths are listed there.

## Reviewed source scope

The approved candidate adds device-returning batched route evaluation and
explicit report projection, device candidate feature/missingness and validity
batches, Plan C route/candidate scoring and selection wiring, compact selected
action transfer, and focused contract tests. It preserves Plan A support and
route semantics. The approved plan and source review describe remaining
host-side candidate specification, perception/extraction, simulator dispatch,
learning/orchestration, and serialization work; this slice does not establish
whole-system GPU-only execution.

The exact-head review approved the source/code candidate. The implementer
reported four focused synthetic contract cases passing (29 deselected),
covering Plan A nonfinite-input rejection, invalid Plan C inference fallback
and empty support behavior, the C1 worker device-mask path, and route/risk
single-deduction plus deterministic tie behavior. The broader route-evaluator
behavior fixtures were authored but not run because repository instructions
require a seeded end-to-end simulator evaluation and current-dashboard replay
for simulated route behavior.

## Evidence limits and pending gates

This is code/source review evidence only. No seeded simulator evaluation,
fresh replay, GPU/HIP parity or profiling, current-dashboard verification,
training run, or qualification is established. Those gates remain pending,
along with the Plan C runtime checks required by the approved plan. No Plan A
implementation or outcome is claimed by this tranche.
