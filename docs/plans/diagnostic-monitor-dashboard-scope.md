# Diagnostic monitor dashboard scope

## Planner pass

Expose the persisted diagnostic monitor state from `queue-status.json` through
`/api/meco-status` and the MECO dashboard. The supervisor already publishes
`diagnostic_monitor` and `diagnostic_monitor_error` in its evaluation status.
The dashboard's `_queue_state` projection currently returns only `evaluation`
and `evaluation_schedule`, and the MECO UI renders only the finite schedule.
Since `/api/meco-status` forwards the queue projection, no separate endpoint or
supervisor change is needed.

This tranche is read-only presentation work. It is independent of the active
r4 campaign: it reads persisted queue snapshots and must not inspect or control
r4 processes, modify its artifacts, or assume monitor state exists there.

## Acceptance criteria

1. For a fresh queue with the monitor enabled, expose a validated explicit
   object at `queue.diagnostic_monitor`. Include `enabled`, `diagnostic_only`,
   `qualification_eligible`, journal hash and event count, completed count,
   plus active, queued, and pending frontier summaries and skipped/missed
   frontier summaries. Allowlist frontier identity fields such as ID, cycle,
   update, status, attempt, and result. Do not forward arbitrary queue JSON or
   checkpoint paths.
2. Surface `diagnostic_monitor_error` as bounded text. Missing or null monitor
   state means “not configured.” A malformed monitor summary must become
   unavailable or untrusted and must never imply qualification eligibility.
   A stale queue remains stale/unconfirmed and must not be rendered as live
   monitor progress.
3. Add a dedicated dashboard status line or card showing configured state,
   diagnostic-only and never-qualification-eligible status, completed count,
   pending frontier/cycle/status, skipped and missed counts, and any current
   error. Do not label monitor results as formal evaluation or qualification.
4. Preserve the existing finite `evaluation_schedule` display.
5. Add projection tests for fresh enabled, absent, malformed/error, and stale
   queue snapshots; assert the diagnostic-only/qualification flags. Cover
   escaping of dynamic UI values through a frontend contract check or an
   equivalent test.

## Files and verification

Expected implementation files:

- `frc_defense/meco_dashboard.py`
- `frc_defense/dashboard_static/meco.html`
- `frc_defense/dashboard_static/meco.js`
- `tests/test_meco_dashboard.py`
- Optionally extend `tests/test_meco_three_pass_provenance.py` if its existing
  static frontend asset assertions are the best fit.

The projection tests can exercise `meco_status_payload` with temporary queue
fixtures; the API simply forwards that payload. Keep this tranche to mocked
queue data and pure projection/frontend contract tests. Do not run simulator
behavior tests, create a replay, start or stop training, interact with a GPU,
or control any process. No active r4 files or runtime state may be changed.

After the change is integrated, restart the dashboard service on the current
checkout and verify the relevant status endpoint and frontend assets match
that checkout. No simulator replay is applicable to this presentation-only
tranche.
