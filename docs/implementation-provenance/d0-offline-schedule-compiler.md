# D0 offline calibration schedule compiler: three-pass summary

## Planner pass

The D0 orchestration implementation could only produce its campaign plan after
GPU/runtime discovery, even though schedule description needs only the frozen
case table. A read-only audit found that D0 code was already present in the
candidate, while the full all-scenario calibration remained unrun. The bounded
scope was to expose deterministic work counts without initializing a GPU,
building the native extension, or disclosing seed lists, while preserving the
source/runtime checks on execution paths.

## Implementer pass

Added `describe_campaign()` and a `describe` CLI action. The description reports
12 scenarios, 466 case strata, 96 balanced per-scenario shards, and 139,800
records for the default calibration schedule. It contains no seed values, work
assignments, or execution manifests. Existing planning, run, and merge paths
retain their source/runtime/GPU checks; `run` and `merge` reject a descriptive
schedule artifact.

Implementer commit: `b4dbaf4cc75e76fa5dc62471e96eda0373d8961b`.
Candidate integration commit: `92404af63`.

## Independent reviewer pass

The reviewer approved exact implementer HEAD
`b4dbaf4cc75e76fa5dc62471e96eda0373d8961b`. Review verified deterministic
counts, that describe does not call GPU discovery or source identity, and that
the output omits master seeds, seed arrays, work units, and manifests. Tests
also confirmed that `run` and `merge` reject this offline artifact before any
execution side effects.

## Verification and limits

Ten focused D0 campaign tests passed; Python compilation, CLI output, and
`git diff --check` passed. The command reported 12 scenarios, 466 case strata,
96 shards, and 139,800 records. No simulator, GPU, dashboard, training, or
calibration shards were run. The full 139,800-record calibration and its
source/runtime-bound outcomes remain outstanding; this tranche is schedule
description only.
