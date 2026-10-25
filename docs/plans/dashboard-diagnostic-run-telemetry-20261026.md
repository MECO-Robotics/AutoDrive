# Dashboard diagnostic run telemetry plan

**Pass:** 1 — planner
**Status:** proposed; no implementation or runtime changes
**Base:** `main` at `664092b93467fd926d8bb74e8e04f311489000c4`
**Worktree:** HDD-backed `codex/dashboard-diagnostic-telemetry-plan-20261026`

## Decision and evidence

The dashboard is correct to reject these five reports as usable evaluation
evidence. It is misleading to project their status-file lifecycle as `invalid`.
Keep the status-reported execution lifecycle separate from report validation.

At the read-only inspection on 2026-10-26, the live r4 directory contained five
completed status files and five report paths. The status files declare
`status: complete`, `split: diagnostic`, evaluation/protocol IDs, and safe
output basenames. A bounded prefix inspection found matching report header
IDs and diagnostic fields, but that does **not** establish report identity:
without the digest-bound summary, the dashboard has no bounded mechanism to
authenticate those oversized reports. Their sizes ranged from 63,310,897 to
78,462,169 bytes. Every status lacked `dashboard_summary_sha256`, and none had
the corresponding `.dashboard-summary.json` sidecar. The live API returned five
`oversized report lacks evaluator summary binding` file errors and projected
the five status-file lifecycle records as `invalid` with the generic
`evaluator status/report provenance mismatch` message. It also reported a
separate incomplete `evaluation-history.json` error.

This rejection is required by the current evidence contract: `_comparison`
accepts only validation-split reports, and `_read_evaluation_comparison`
requires a digest-bound evaluator summary for an oversized report. Diagnostic
split/status matching alone cannot authenticate the report's contents or make
it comparison evidence. Do not accept these reports as comparisons, infer
qualification, or alter their status/split/bindings. The telemetry defect is
that one `status` field conflates a schema-validated status-file claim of
terminal execution (`complete`) with report evidence validation (`invalid`).
The UI must say **status-reported complete** and must not imply independent
confirmation that the referenced report matches that status.

## Objective

Represent the diagnostic evaluator's terminal status claim accurately while
preserving explicit report/provenance errors. The dashboard must make clear
that a schema-validated status file reports completion, that the referenced
full report is unavailable/unvalidated because it lacks the required
oversized-report summary binding, and that neither the status file nor the
diagnostic record is qualification evidence.

## Three-pass execution

### Pass 1 — Planner (this artifact)

Pin the separation between execution lifecycle and report evidence. Keep the
current strict report reader, size limits, comparison eligibility, and
qualification boundary. Specify a minimal projection change and its acceptance
criteria below. No campaign files, process, service, device, report, or
checkpoint is changed in this pass.

### Pass 2 — Implementer

In an isolated HDD-backed worktree, update `frc_defense/meco_dashboard.py` to
expose separate bounded fields for status-file lifecycle and report validation.
Only a schema-validated status with terminal `complete`, `split: diagnostic`,
string evaluation/protocol IDs, and a safe in-root output basename may project
`execution_status: status-reported complete` and `diagnostic: true`. This is a
status-file-only claim. Do not read an oversized report prefix or trust its
header to establish identity. Keep a separate `report_validation` state and
`evidence_error`; absent binding must report unvalidated/unavailable with the
specific `oversized report lacks evaluator summary binding` reason. A
status-reported completion must never clear or weaken that evidence error.

When the report is missing, unsafe/outside-root, malformed, mismatched, or
cannot be validated by the bounded reader, report evidence must remain
invalid/unconfirmed and must not be described as matched to the status. Keep
the evaluator lifecycle claim explicitly tied to the status file; do not infer
report completion or identity from progress, filenames, a report prefix, or
status alone. Do not repurpose a positive lifecycle label into a positive
evidence state or suppress the error.

Update `frc_defense/dashboard_static/meco.js` to label the record as
“status-reported complete · diagnostic · report unvalidated” when those exact
status-only predicates pass. Continue displaying the split, progress, and
evidence error. Escape all strings as today. If status validation, terminal
state, diagnostic split, IDs, or safe output path fail, show status as
unconfirmed/invalid; never infer completion or report identity from progress,
filenames, or report metadata. Do not change evaluator output, retroactively
create bindings, or read raw report contents in the browser.

Add focused fixture tests for these projection cases: schema-valid complete
diagnostic status with an oversized output and no summary binding (lifecycle is
status-reported complete; report identity remains unconfirmed); a complete
report with a valid bound summary; missing or unsafe report path; digest,
ID/protocol/split mismatch after bounded validation; malformed or absent
status; and incomplete/running evaluator. Tests must assert no report identity
is inferred from an unbound oversized report, the diagnostic case remains
absent from comparison groups and qualification inputs, and lifecycle status
and missing binding are independently visible.
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

- A schema-validated terminal status file with diagnostic split, IDs, and a
  safe in-root output basename is shown as **status-reported complete**. This
  does not assert that the referenced report matches the status.
- Without the required bounded report binding, report validation remains
  unvalidated/invalid even when the status file reports completion. No
  oversized report identity is inferred from a header or filename.
- The specific missing-binding error remains visible in the API/UI and in the
  aggregate errors list; it is not replaced by a generic success or silently
  hidden.
- Such a run contributes no comparison metrics, promotion decision, or
  qualification evidence.
- A fully bound report continues through the existing bounded reader and
  validation rules without changing its semantics.
- A missing, unsafe, malformed, digest-mismatched, ID/protocol/split-mismatched,
  or otherwise unvalidated report remains evidence-invalid/unconfirmed with an
  actionable error. A status-file completion claim remains explicitly
  status-only; no report identity or evidence validity is inferred from
  filenames, headers, or progress counts.
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
