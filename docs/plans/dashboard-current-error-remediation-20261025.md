# Dashboard current error remediation plan

**Status:** planner scope only; no implementation, data, service, or training changes
**Planning base:** `main` at `360c5f074b259b16aee8b2d682ecdbef9e645f1d`
**Worktree:** HDD-backed isolated worktree `codex/dashboard-current-error-remediation-plan-20261025`

## Objective and observed failures

Repair two current read-only dashboard projections without changing evaluation semantics or the live campaign:

1. At about 2026-10-25 09:26 UTC, `/api/curriculum-status` returned five generic `Saved evidence could not be verified.` entries. Calling `scripts.report_curriculum_progress.summarize()` against the configured campaign root identified the exact errors: each of five fresh queue lanes was rejected as `queue lane has no valid directory`. The active `queue-status.json` schema publishes `run_directory` (`A1`, `A2`, `A3`, `C1`, `C2`), while the summarizer currently requires `directory`. These are queue projection errors, not five invalid evaluation records.
2. `/api/meco-status` reported two completed diagnostic evaluation files as over the 32 MiB status-reader limit. `cycle-000001-263b6684.json` is 63,310,897 bytes (evaluation `7e2c4fa7e5cd4f929d7b6d918b9497af`, protocol `1e49e2134ec6877ba588f8c92aa6b461be86a7481941056366ede054edd57c3e`); `cycle-000002-6e9cca09.json` is 78,108,801 bytes (evaluation `473ed11878454e2fad9762ee22cbe0cb`, protocol `b45ba8feed3fc2837102f5dd2349253e12d745b324fb1cc90d358b6919b05924`). Both sidecars mark `complete`, `diagnostic`, 11/11 records. The current dashboard reader rejects each before comparison validation.

These diagnostic reports must remain diagnostic. Surfacing or validating them does not qualify a model, establish a promotion result, or convert FUEL counts into official match points.

## Source paths and scope

- `scripts/report_curriculum_progress.py`: `_stats_files`, `summarize`, and queue-lane projection currently look for `directory`.
- `frc_defense/meco_supervisor.py`: queue writer emits `run_directory` for each lane.
- `frc_defense/meco_dashboard.py`: `_queue_state` already validates `run_directory`; `_read_json` enforces `MAX_STATUS_FILE_BYTES = 32 * 1024 * 1024`; `_collect_status` reads evaluation reports and projects only reports accepted by `_comparison`.
- `frc_defense/dashboard_api.py`: `curriculum_status_payload` intentionally sanitizes errors but currently labels every summary error as saved-evidence verification failure.
- Existing focused test homes: `tests/test_dashboard_curriculum.py`, `tests/test_meco_dashboard.py`; add or extend report-reader tests in the closest existing test module rather than adding a separate test framework.
- Frontend affected only if error wording/status details change: `frc_defense/dashboard_static/dashboard.js`, `frc_defense/dashboard_static/meco.js`, and their HTML templates.

## Implementation sequence

1. Fix queue compatibility in the summarizer. Accept the current `run_directory` field and, if legacy `directory` support is retained, define explicit precedence and reject conflicting values. Resolve the lane under the known campaign checkpoint root (`checkpoints/<run_directory>`), verify it matches the in-root stats/run path for that track, reject traversal and symlinks, and preserve an error for missing or mismatched lane directories. Do not accept arbitrary absolute paths from queue JSON. Keep queue lane live state separate from saved evaluation evidence.
2. Replace the generic curriculum error text with a concise bounded projection message that identifies whether queue/status or saved evaluation evidence was rejected. Do not expose absolute paths, raw report contents, exception traces, or unbounded input strings.
3. Resolve large evaluation reports using a bounded, memory-aware path. Preferred option: produce an atomic compact dashboard summary alongside a completed raw report, bound to the raw report SHA-256 plus evaluation ID, protocol ID, split, source identity, schema version, and record count; dashboard validates the summary schema and binding before projection. Derive the summary through the existing strict `_comparison` semantics and preserve its completeness/uniqueness checks. Keep raw reports authoritative and untouched. If that writer-side format cannot be safely scoped, use a streaming reader or a strictly bounded reader with both per-file and aggregate/concurrency memory limits established from representative measurements. Do not merely remove the limit or raise it without an explicit memory budget and a regression case at/over the configured boundary.
4. Preserve status/report agreement checks, containment and symlink protection, diagnostic labeling, and the rule that unavailable, malformed, or unbound summary artifacts remain errors rather than partial evidence. Do not synthesize missing records or a success state.

## Acceptance and test scope

- A fresh five-lane queue using `run_directory` projects all five lanes without errors, and each resolves only to its corresponding in-root `checkpoints/<track>` stats directory.
- Legacy queue shape, if supported, remains covered. Conflicting `directory`/`run_directory`, missing directories, traversal, symlink, and outside-root cases fail closed and never disclose external data.
- Curriculum API wording distinguishes queue projection errors from saved evidence errors and remains bounded; valid evaluation evidence is not hidden by unrelated queue errors.
- Oversized complete diagnostic report acceptance is tested using a representative generated fixture or compact sidecar fixture; verify exact raw digest/evaluation/protocol/split/source binding, complete record coverage, and equality with the existing `_comparison` projection.
- Reject oversized/unbounded inputs, stale/mismatched sidecars, altered raw digest, duplicate/missing rows, malformed schema, and status/report provenance disagreement. Bound concurrent memory/work; test aggregate limits if the selected design has them.
- Explicitly assert diagnostic reports remain diagnostic and cannot affect qualification or promotion. No tests should reinterpret FUEL throughput as match points.
- Run focused dashboard/report contract tests only. This is presentation/provenance work; no simulator behavior test or replay is applicable, and no training, GPU, or device work is in scope.

## Integration and currentness verification

After implementation is reviewed and integrated, restart the serving dashboard service from the integrated checkout so its Python modules reload. Record UTC time, service active state, PID/invocation, working directory, integrated commit, and source identity. Verify `/api/curriculum-status` and `/api/meco-status` return HTTP 200 and report current queue lanes plus both report IDs as bounded diagnostic summaries with matching provenance. Verify the correct frontend assets are served from that same checkout (at minimum `/`, `/meco`, `/assets/dashboard.js`, `/assets/meco.js`, `/assets/dashboard.css`, and `/assets/meco.css`) by comparing served bytes or SHA-256 values to local files. A fresh simulator replay is not applicable because no simulator behavior changes; do not restart or interfere with training.

## Limits

The endpoint and artifact checks establish only dashboard projection and provenance handling. They do not establish model performance, official match points, qualification, promotion, or plan completion. The two observed evaluations remain diagnostic-only. This plan creates no report, summary sidecar, model artifact, or runtime evidence.
