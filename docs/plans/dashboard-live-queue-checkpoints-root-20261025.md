# Dashboard live queue checkpoint-root correction plan

**Status:** planner scope only; no code, tests, service, training, or data changes
**Planning base:** `7d3273e2c810c2022aac94e16aad487ab643bef4` (`origin/main`)
**Worktree:** HDD-backed `codex/dashboard-live-queue-root-plan-20261025`
**Date:** 2026-10-25

## Objective and evidence

Fix the live curriculum queue projection by completing the root-resolution
acceptance already specified in
[`dashboard-current-error-remediation-20261025.md`](dashboard-current-error-remediation-20261025.md).
That plan required resolving each current lane to
`checkpoints/<run_directory>` and matching it exactly to an in-root stats/run
directory. The current summarizer still resolves `root/<run_directory>`, so
the accepted requirement is not met. This is a correction/completion of that
plan's existing acceptance, not a new queue behavior.

At the planning snapshot:

- The live campaign root is
  `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDriveCampaigns/meco-five-continuous-20261025-r4`.
- Its `queue-status.json` has five lanes with `run_directory` values `A1`,
  `A2`, `A3`, `C1`, and `C2`. At 2026-10-25 10:08 UTC, the lanes showed A1
  running and the other four complete.
- `checkpoints/<track>` exists for all five tracks; `<campaign-root>/<track>`
  does not exist for any of them.
- The current resolver in `scripts/report_curriculum_progress.py` calls
  `_queue_lane_directory(root, run_directory)`, while `summarize()` indexes
  saved runs by the parent of each discovered `stats.json`. The current API
  response reports five sanitized queue projection errors, consistent with
  the failed root lookup.
- `frc_defense/meco_dashboard.py::_queue_state` already validates the queue
  lane's `run_directory`; the defect is the curriculum summarizer's mapping
  from that track name to its actual checkpoint directory.

The existing dashboard-current-error plan also covers oversized evaluation
sidecars. That separate issue is out of scope here; do not alter it or recast
the live queue failure as saved evaluation evidence failure.

## Scope and safe resolution contract

Keep the patch in the curriculum report resolver and its focused tests. The
dashboard API and live queue writer should need no schema change.

1. Resolve a `run_directory` only when it is a single safe track component
   equal to the lane's `track`. Reject absolute names, traversal, separators,
   symlinks at any component, missing paths, and paths outside the configured
   report root.
2. Support the canonical current layout `checkpoints/<track>` and legacy
   in-root layout `<root>/<track>`. Prefer the canonical checkpoint layout
   when it is the sole matching location. If both layouts exist and resolve to
   different stats-backed runs, fail closed as ambiguous. If both are present
   and a legacy `directory` field disambiguates them, accept it only when it
   resolves inside the report root to exactly one of those two track paths and
   agrees with `run_directory`.
3. Preserve the prior in-root legacy `directory` field for queue producers
   that still emit it, including an absolute path only after containment and
   exact candidate-path validation. If both `directory` and
   `run_directory` are present, they must resolve to the same chosen track
   directory. Never accept an arbitrary absolute path just because it is
   somewhere under the campaign root.
4. Join a lane to saved evidence only by exact resolved-directory equality
   with the `stats.json` run directory discovered under the configured root.
   Do not match on a basename, track label alone, or a nearby directory.
   Where a valid queue directory exists but has no `stats.json` yet, retain
   the existing queue-only pending representation with no evaluation,
   qualification, or saved-evidence fields synthesized from queue JSON.
5. Keep queue state separate from saved evaluation evidence. Errors must remain
   bounded and sanitized by the existing API projection; do not expose paths,
   raw status contents, or exceptions.

## Implementation and review passes

1. **Planner:** this artifact pins the failure, historical acceptance, safe
   resolution rules, supported legacy layouts, and test contract. Before
   implementation, rebase the plan on then-current pushed `main` if the
   summarizer, queue schema, or data-root behavior has changed.
2. **Implementer:** use a fresh HDD-backed worktree at the then-current
   `origin/main`. Change only the summarizer path resolver and focused tests.
   Do not touch `frc_defense/**/*.py` used by the live training checkout,
   campaign files, checkpoints, manifest, queue, HIP output, training process,
   or dashboard service.
3. **Independent reviewer:** review the exact pushed implementation HEAD and
   diff against this plan. Check containment, symlink rejection, path ambiguity,
   exact stats/run joining, pending-lane behavior, and that malformed queue
   data cannot create saved evidence or qualification.
4. **Integration:** integrate only after exact-head approval. Then restart the
   serving dashboard from the integrated checkout and perform the currentness
   checks below. No service restart is part of planning or implementation
   review.

## Required focused offline tests

- Campaign-shaped fixture with `queue-status.json` and five
  `checkpoints/<track>/stats.json` files: all five lanes resolve to those exact
  directories, are joined to the corresponding saved runs, and produce no
  queue errors.
- Legacy `<root>/<track>` layout remains readable when it is the only
  matching directory; legacy in-root `directory` and matching dual-field
  inputs remain unambiguous.
- Queue-only pending lane with a valid in-root directory but no `stats.json`
  remains pending and cannot contribute evaluation or qualification evidence.
- Conflicting `run_directory`/`directory`, both distinct layouts present,
  wrong track component, missing track directory, traversal, symlink,
  outside-root absolute path, and non-directory path fail closed.
- A stats-backed run elsewhere under the root with the same basename is never
  joined to a lane whose exact resolved path differs.
- Curriculum API projection continues to return a bounded queue error for
  invalid lanes without revealing private paths or raw queue details; unrelated
  saved-evidence errors remain classified separately.

Do not run simulator, replay, training, GPU, or device work. This is a
read-only dashboard projection change, so simulator behavior acceptance does
not apply.

## Post-merge dashboard currentness

After integration, restart the dashboard service from the integrated checkout
and record UTC time, PID/invocation, working directory, commit/source identity,
and endpoint statuses. Verify `/api/curriculum-status` returns all five
current queue lanes with the expected track names and no queue-projection
errors. Verify `/api/meco-status` separately as the campaign-status endpoint;
do not expect it to return curriculum queue details. Fetch `/`, `/meco`, and
all referenced JavaScript/CSS assets; compare their served bytes or SHA-256
values with files from that same checkout. Do not restart or modify the live
r4 trainer or its data.

## Acceptance and limits

- The current queue resolves to `checkpoints/A1`, `checkpoints/A2`,
  `checkpoints/A3`, `checkpoints/C1`, and `checkpoints/C2` under the configured
  campaign root, and each queue lane joins only the matching in-root stats/run
  directory.
- Legacy in-root layouts remain supported where unambiguous; unsafe, conflicting,
  ambiguous, missing, or out-of-root paths are rejected without leaking details.
- Queue-only pending lanes remain status-only. Queue state cannot supply saved
  evaluations, qualification, or promotion evidence.
- Post-merge endpoint and asset checks establish dashboard currentness only.
  They do not validate training, model performance, simulator behavior,
  qualification, promotion, or the separate large-report-sidecar issue.
- Preserve the active r4 campaign and the unrelated generated HIP working-tree
  modification. No training, source-checkout, checkpoint, manifest, queue,
  device-lease, or reward change is in scope.
