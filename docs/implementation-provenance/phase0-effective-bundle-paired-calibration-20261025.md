# Phase 0 effective bundle and paired calibration integration

**Scope:** reviewed code support for binding effective source/runtime freeze v3 and a closed tuple partition into paired D0/D1/D2 calibration. This summary records the scoped plans and exact reviewed implementation; it does not claim that a production freeze, partition, approval, or calibration dataset exists.

## Pass 1 plans

- `docs/plans/phase0-effective-freeze-partition-materialization-20261025.md`
- `docs/plans/phase0-resumable-paired-d0-d2-calibration-20261025.md`

The first plan adds a captured effective configuration receipt, freeze-v3, closed tuple registry, four disjoint partitions, five-track assignments, and a final-test access ledger. The paired plan binds D0 and the predeclared D1/D2 controls to that bundle and requires exact tuple pairing, replay-bound outcomes, immutable resume/merge, a human estimand approval, and an independently observed release of r4 and both WX9100s before execution.

## Pass 2 implementation

Reviewed implementation checkpoints:

- Freeze, receipt, source/runtime provenance, training-manifest binding, trusted allocation and final-test ledger: `b44d39eca41b4716261d13e1307615d6a098e0a9`.
- Paired schema v2, registry-derived calibration schedule, tuple/control provenance, and training-entry bundle revalidation: `de620bb0c977ca344a4f3b012bc00711fcc32f82`.
- Effective-v3 validator/config-byte integration fix: `1ba974e1af55a1fce8f6493c6dd3e0ece3f33104`.
- Combined integration candidate: `ac590f5487436735c78aef7671be5a9f8aaee187`.

The paired runner now validates the effective v3 receipt → freeze → partition chain, binds the receipt/freeze/partition/tuple-registry/assignment digests, derives work only from eligible registered calibration tuples, and carries full predictor contracts and tuple IDs through shards, replay bindings, rows, bundles, and merge. It has no legacy freeze-v2 schedule fallback. Caller configuration bytes are compared with receipt-authenticated bytes before evaluator setup. The final-test ledger and live-release capability remain fail-closed; missing human approval or release evidence cannot authorize a run.

## Pass 3 and checks

Independent review approved the freeze head `b44d39e`, paired head `de620bb`, v3 validator fix `1ba974e`, and combined head `ac590f5`. The combined focused offline suites passed: **172 tests** across freeze, training manifest, diagnostic monitor, paired calibration contract/CLI/replay binding, multimodel prediction, and release-gate contracts.

No simulator evaluation, replay, dashboard restart/currentness check, GPU query/workload, live release observer, calibration run, or training action was performed for this tranche. The active diagnostic campaign was left untouched.

## Remaining gates

No production effective freeze, tuple partition, human-authored estimand approval, complete paired calibration result, margin approval, held-out result, or qualification evidence was created. The user's manual stop and verified release of the current r4 campaign and both WX9100 devices remain prerequisites to any new execution. This integration establishes code contracts only; it does not establish that any production gate is satisfied or that the full learning-track plan is complete.
