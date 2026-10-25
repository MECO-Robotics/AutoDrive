# Phase 4 training exposure in source-bound campaigns

## Planner pass

The campaign manifest previously omitted Phase 4 training exposure even though
the task scheduler and direct supervisor supported it. A future campaign could
not opt into the registered adversarial training assignments through the
source-bound manifest. The scoped change adds a strict boolean manifest field
and carries that choice through durable evaluation history and protocol
transition auditing. The active r4 campaign remains on its original manifest.

## Implementer pass

Commits: `24a280402`, `5659d7cfd`, and `e843d7ee6`.

The manifest canonical digest now binds `phase4_train_exposure`, rejects
non-boolean values, and defaults the field to false for legacy manifests. The
supervisor writes the manifest digest, source configuration digest, and
exposure flag into each durable evaluation-history row and its five checkpoint
snapshot bindings. The transition auditor verifies those bindings against the
queue anchor, requires exactly one snapshot for each of A1/A2/A3/C1/C2, checks
the expected checkpoint filename and path, and verifies each checkpoint hash.
A regression rehashes forged duplicate-track metadata and confirms rejection.

Focused verification on the feature worktree: 3 targeted history tests, 2
protocol-transition history/tamper tests, and 20 tests across the continual
evaluation schedule and scheduler suites passed. Python compilation and
`git diff --check` passed.

## Reviewer pass

Independent exact-head review approved `ba38d8799226c05c498d14d1f6593ef2c73dc738`.
The reviewer verified strict boolean handling, canonical manifest binding,
legacy false behavior, matching durable configuration bindings, queue-anchored
transition validation, and exact one-per-track checkpoint path/hash checks. No
additional blocking issue was found. The three reviewed commits were integrated
into current `main` as `24a280402`, `5659d7cfd`, and `e843d7ee6`.

## Limits

These are CPU contract and static checks. No simulator evaluation, replay,
dashboard replay, GPU test, new training campaign, Phase 4 outcome measurement,
qualification, or deployment validation was performed. The change makes
Phase 4 exposure configurable for a future source-bound campaign; it does not
change or reconfigure the active r4 run.
