# Manifest-bound diagnostic evaluation monitor

## Planner pass

The scoped planner identified that finite evaluation schedules do not provide
continuous diagnostic snapshots, while the legacy unscheduled path coalesces
pending snapshots. The approved design adds an opt-in, manifest-bound
`latest_only` diagnostic monitor for future clean campaigns. It keeps
qualification ineligible, binds exact protocol and seed sets, journals
frontiers and attempts durably, verifies checkpoint/source/configuration
provenance, and uses only the configured evaluation device. It does not alter
the currently running r4 campaign or supervisor-level device leases.

Planner scope: `docs/plans/diagnostic-eval-monitor-scope.md`.

## Implementer pass

The implementation adds `MecoDiagnosticMonitor`, canonical manifest
validation, supervisor scheduling and evaluation-report bindings, resumable
frontier/attempt journaling, and contract tests. Review corrections ensure
that recovery never labels a later checkpoint as an earlier frontier,
monitor seeds are disjoint from both frozen evaluation and qualification
protocol partitions, launch and runtime failures release the evaluation lane,
and missed or failed work remains explicit in durable state.

Implementation commits:

- `0dfd87987bef7aac76db499f42b8615c7735df5c` — initial monitor
- `12c9c432975fa87aa0cf1db03b8b99b7e9d86f52` — recovery, seed isolation,
  and launch-failure handling
- `f79650107a5c92053d651a7d48d9bcd306257f1d` — runtime-failure lane release

## Independent reviewer pass

The exact final head `f79650107a5c92053d651a7d48d9bcd306257f1d` was approved
after two rejected review rounds. The review verified checkpoint-update
matching and durable missed-frontier records, seed isolation against both
protocols, release of the evaluation lane for launch and runtime failures,
and diagnostic-only semantics.

## Verification and limits

The implementation focused suite passed **71 tests**; the final reviewer ran
51 monitor and manifest tests. Python compilation and `git diff --check`
passed. These are offline contract and mock checks. No simulator evaluation or
replay, GPU execution, training, dashboard restart, or qualification was
performed for the implementation itself. The monitor is opt-in for a future
campaign; the active r4 training run was left untouched. Diagnostic reports
cannot qualify or promote a model.
