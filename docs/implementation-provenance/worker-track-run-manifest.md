# Worker run-manifest track binding

## Planner

The failed five-track diagnostic worker wrote its track only inside
`config.track`, while the supervisor required a top-level `track` binding.
Keep the supervisor check strict, add the top-level field to the worker's
single run-manifest construction path, and cover both missing and mismatched
fields while the embedded A1 config remains correct. Treat the old campaign as
non-resumable under the source-bound fresh-root launcher; do not edit its
stop marker or run records.

## Implementation

Commit `a4cc6a45ef2b3053bcc70af08ae265801bb74ed4` adds `track: config.track`
to `frc_defense/meco_track_training.py`. The shared path runs after fresh setup
or resume loading. `tests/test_meco_supervisor_dispatch.py` now rejects absent
and wrong top-level values even when `config.track` is `A1`. The strict
supervisor condition was not changed. Nineteen focused dispatch tests passed;
Python compilation and `git diff --check` passed.

## Review and limits

Independent review approved exact commit `a4cc6a45ef2b3053bcc70af08ae265801bb74ed4`.
The reviewer confirmed embedded config alone is insufficient and the
supervisor's other source, native-attestation, run, task, update, protocol,
and completion checks remain intact. This metadata-only tranche has no
simulator, GPU, campaign, dashboard, evaluation, or qualification evidence.
The failed campaign remains archived; a new source-bound root is needed to
resume training safely.
