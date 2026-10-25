# Phase 4 direct-pursuit runtime gate: three-pass summary

## Planner pass

The approved scope is `docs/plans/phase4-direct-pursuit-executable-gate-20261025.md`
(SHA-256 `20b9c0d56aafb46a8353f36c1503a33b651a89e6631dda89a0190c4b177762f5`).
It binds one held-out direct-pursuit paired validation cell to simulator-grounded
contact, interception-incidence, and progress-based time-loss outcomes. It
requires exact source, protocol, case, seed, checkpoint, device, assignment,
and pair identity; one designated treatment slot; the remaining category
opponent preserved; all 11 policies and 100 seeds per policy in each arm; and
reviewed current-source replay evidence before executability.

## Implementer pass

Implemented on `codex/phase4-direct-pursuit-exec-gate-20261025` through commits
`35bc80e2b`, `0a7545fac`, `8fadd4b8d`, `23090d38c`, and `0533be228`; integrated
to `main` at `42a57af12`.

The implementation captures per-pair simulator SAT-contact events and
physics-time progress at each evaluator step; binds paired arms with a stable
shared experiment identity while retaining arm-specific condition IDs; checks
contact aggregates against events and requires uncensored matched progress;
uses episode-level interception incidence separately from contact frequency;
requires exact 11-policy coverage with 1,100 rows per condition; preserves the
other registered scripted pressure slot; and keeps promotion fail-closed until
reviewed replay/dashboard evidence is supplied.

## Reviewer pass

Independent exact-head review requested changes for pair-condition identity,
constructor binding order, policy coverage, single-slot assignment, and exact
arm-count keys. These were corrected in separate pushed commits. The reviewer
approved exact head `0533be2283f23ed7a5c3dc7177c1eefc46453ad2` with no remaining
static blocker.

## Validation and limits

Static AST parsing and `git diff --check` passed. No tests, simulator
execution, GPU probe, training, dashboard replay, or qualification run was
performed. The code does not make the cell executable until fresh simulator
records, an inspected current-source replay, and current dashboard evidence
are supplied. No direct-pursuit outcome, Phase 4 qualification, model
performance, or promotion is claimed.
