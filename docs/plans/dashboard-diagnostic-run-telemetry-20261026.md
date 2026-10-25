# Dashboard diagnostic run telemetry plan

**Pass:** 1 — planner
**Status:** proposed; no implementation or runtime changes
**Base:** `main` at `664092b93467fd926d8bb74e8e04f311489000c4`
**Worktree:** HDD-backed `codex/dashboard-diagnostic-telemetry-plan-20261026`

## Decision and evidence

The dashboard is correct to reject these five reports as usable evaluation
evidence. It is misleading to project their execution status as `invalid`.
Keep those two conclusions separate.

At the read-only inspection on 2026-10-26, the live r4 directory contained five
completed status files and five matching report files. For every pair, the
status and report headers had the same `evaluation_id` and `protocol_id`; both
declared `split: diagnostic`. The reports declared `diagnostic: true` and
`evaluation_design: diagnostic_unbound`. Their sizes ranged from 63,310,897 to
78,462,169 bytes. Every status lacked `dashboard_summary_sha256`, and none had
the corresponding `.dashboard-summary.json` sidecar. The live API returned five
`oversized report lacks evaluator summary binding` file errors and projected
the five execution records as `invalid` with the generic
`evaluator status/report provenance mismatch` message. It also reported a
separate incomplete `evaluation-history.json` error.

This rejection is required by the current evidence contract: `_comparison`
accepts only validation-split reports, and `_read_evaluation_comparison`
requires a digest-bound evaluator summary for an oversized report. Diagnostic
split/status matching alone cannot authenticate the report's contents or make
it comparison evidence. Do not accept these reports as comparisons, infer
qualification, or alter their status/split/bindings. The telemetry defect is
that one `status` field conflates a terminal evaluator execution (`complete`)
with report evidence validation (`invalid`).

## Objective

Represent diagnostic evaluator completion accurately while preserving explicit
report/provenance errors. The dashboard must make it clear that a diagnostic
run finished, that its full report was not validated because it lacks the
required oversized-report summary binding, and that it is never qualification
evidence.

## Three-pass execution

### Pass 1 — Planner (this artifact)

Pin the separation between execution lifecycle and report evidence. Keep the
current strict report reader, size limits, comparison eligibility, and
qualification boundary. Specify a minimal projection change and its acceptance
criteria below. No campaign files, process, service, device, report, or
checkpoint is changed in this pass.

### Pass 2 — Implementer

In an isolated HDD-backed worktree, update `frc_defense/meco_dashboard.py` to
expose separate bounded fields for evaluator lifecycle and report validation.
For example, retain the validated report state as `invalid` while projecting
`execution_status: complete`, `diagnostic: true` only when supported by
validated status fields, and a specific bounded `evidence_error` such as
`oversized report lacks evaluator summary binding`. Do not repurpose the
existing `status` into a positive evidence state or suppress the error.

Update `frc_defense/dashboard_static/meco.js` to label the record as completed
diagnostic execution with unvalidated report evidence. Continue displaying the
split, progress, and evidence error. Escape all strings as today. If the
diagnostic flag or lifecycle status cannot be confirmed from status/report
metadata, show an unconfirmed state instead of inferring completion. Do not
change evaluator output, retroactively create bindings, or read raw report
contents in the browser.

Add focused fixture tests for these projection cases: matching diagnostic
status/report IDs with missing summary binding; a complete report with a valid
bound summary; status/report ID or protocol mismatch; malformed or absent
status; and an incomplete/running evaluator. Tests must assert that the
diagnostic case remains absent from comparison groups and qualification inputs,
while lifecycle completion and missing binding are independently visible.
Run only the focused dashboard contract tests and static checks; do not run a
simulator, training, fitting, GPU/HIP/CUDA work, device inventory, or any
campaign workload. CPU-only contract tests may not be used as a training
workaround or performance evidence.

### Pass 3 — Independent reviewer

Review the implementer's exact commit in a separate worktree. Verify the
projection distinguishes execution completion from evidence validity, the
five diagnostic records cannot enter comparison or qualification paths, missing
provenance remains visible, malformed/mismatched inputs fail closed, and no
campaign/runtime files changed. Re-run the focused tests if safe. Approve only
the exact reviewed head; otherwise return a concrete correction request and
require a new review of the amended head.

## Acceptance criteria

- A completed, diagnostic evaluator with matching status/report IDs is shown as
  a completed diagnostic *run*, with report evidence explicitly unvalidated
  when its summary binding is absent.
- The specific missing-binding error remains visible in the API/UI and in the
  aggregate errors list; it is not replaced by a generic success or silently
  hidden.
- Such a run contributes no comparison metrics, promotion decision, or
  qualification evidence.
- A fully bound report continues through the existing bounded reader and
  validation rules without changing its semantics.
- Status/report ID, protocol, split, or lifecycle inconsistencies remain
  unconfirmed/invalid with an actionable error; no fields are inferred from
  filenames or progress counts.
- The five live r4 artifacts remain read-only and unchanged. No process/service
  restart, training interruption, device lease, GPU workload, CPU training, or
  report-sidecar backfill is part of these passes. Preserve the two-WX9100
  policy for any future authorized training; do not use P600 or CPU compute.

## Limits

This is a dashboard telemetry change only. It does not repair or bind the five
historical reports, validate their full records, establish evaluator-result
quality, create a match outcome, pass a curriculum stage, qualify/promote any
track, or establish that the active training campaign is healthy. Training is
to remain uninterrupted. No dashboard restart or live endpoint revalidation is
authorized by this plan; any later currentness check must be separately scoped
and must not disturb training.
