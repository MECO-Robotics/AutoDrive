# Deferred first observation and replay audit lifecycle

## Planner summary

The deferred curriculum reset path calls `observe()` before any policy action. With no cached observation, the all-active path must return a fresh frame; partial-active observation without a cached frame must fail clearly. Replay capture also needs an explicit no-action state. Action-derived target, fuel-target, and cluster snapshots can be absent before the first policy action and stale after full or partial resets. Preserve CUDA graph output buffers in place and invalidate audit validity per reset world.

## Implementer summary

Commit `079ec0f3d4057e1d0036218030f44c252522236e` fixed deferred observation initialization and added four focused regression cases. Commit `1cfdfeab27b7e14b5ababd488d522d57935a966b` added per-world action-audit validity, full and partial reset invalidation, and replay serialization of unavailable initial targets as `null` with `robot_action_audit_available=false`. Candidate integration commit `0e97316c6255e14bab66b4ef11f2936e1c5f17a1` contains both changes. The focused replay test module passed 31 tests; compilation and diff checks passed.

## Reviewer summary

The independent reviewer approved exact implementation commit `1cfdfeab27b7e14b5ababd488d522d57935a966b`. Validity is a separate per-world bool buffer, reset in place; replay suppresses stale snapshots until a policy action refreshes them. The review confirmed controller consumers run after target recomputation and CUDA graph audit-buffer identity is preserved. The focused tests cover null initial fields and per-world validity invalidation, but do not execute a full CUDA graph reset lifecycle.

## Runtime evidence and limits

The subsequent integrated-candidate simulator evaluation exercised this path on `cuda:0`: D0, `pickup_simple/cluster`, seed `61720261024`, 59 replay frames. Frame 0 reports unavailable action audit with null targets; frame 1 reports action-derived values. This is a diagnostic replay check, not held-out qualification, calibration, a learned-track result, or a full reset/CUDA graph stress test.
