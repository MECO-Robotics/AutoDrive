# Continual matched five-track evaluation schedule

## Planner pass

The active r4 supervisor assigns matched tasks to the five tracks and can
advance through curriculum exposure, but its automatic evaluation is fixed to
one diagnostic `pickup_simple/cluster` seed. The live campaign is pinned to an
older source checkout, has Phase 4 training exposure disabled, and does not
evaluate curriculum stage promotion. The planner proposed a manifest-bound
case schedule for a future campaign handoff; the active service and manifest
were left untouched.

## Implementer pass

Implementation commits: `de03ce27a` and `89d0e86db`.

The optional frozen schedule binds each diagnostic or validation case to the
manifest's evaluation protocol ID and file digest. Validation cases must match
an exact protocol case and seed list; diagnostic seeds are rejected if they
overlap frozen train, validation, or test partitions. One case is selected
deterministically per completed five-track cycle. Evaluation history records
the selected case and snapshot hashes for all five checkpoints. Existing
manifests without a schedule retain the previous evaluation path.

Focused verification: 25 schedule/manifest contract tests passed; Python
compilation and `git diff --check` passed. Dispatch/device tests were excluded
after the live campaign held the `cuda:0` lease. No service or GPU was touched.

## Reviewer pass

Independent review approved the integrated current-main state at
`89d0e86db7774c014ce0b1c66c8f7426a969d039`. Review confirmed the implementation
files match the reviewed branch exactly, protocol and seed binding, leakage
guards, deterministic within-run cycle selection, all-five checkpoint snapshot
hashes, and backward compatibility.

## Limits

This schedule is not enabled in the active r4 campaign. Manifest campaigns
require fresh output roots and cannot resume in place after a supervisor
restart; schedule continuation across restarts is unsupported. No simulator
evaluation, replay inspection, GPU run, fresh dashboard replay, or qualification
was performed. The active campaign remains diagnostic and pinned to an older
source commit.
