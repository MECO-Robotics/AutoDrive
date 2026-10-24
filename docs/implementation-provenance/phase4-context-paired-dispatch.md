# Phase 4 focal-pose and teammate-behavior paired dispatch

## Planner summary

The validation plan already defines focal-pose-offset and teammate-behavior factors, but the campaign runner did not dispatch them. The implementation scope binds a no-factor and treatment arm to the exact same validation protocol case, seed list, five checkpoint hashes, source identity, and execution device. Assignment and row coverage must be exact; results remain diagnostic until simulator evidence exists.

## Implementer summary

Commit `509b219d3` adds `run_phase4_context_cell` for `observation.focal_pose_offset` and `teammate.behavior_pool`, the deterministic teammate pool seed CLI argument, and paired-only matrix selection without the architecture baseline. It accepts only a complete pair and emits no outcome contrasts. Dispatch is write-once, each condition report must resolve inside the fresh case directory, be a regular file, and match its matrix-declared SHA-256. The runner validates assignment digests, protocol/seed manifests, checkpoint/source/device bindings, exact policy/category/seed coverage, and null outcome fields. Qualification remains false.

The reviewed diff SHA-256 was `005896ca2018a54b833d7f402d1ac192fcf85b1aedb765233f570cbd435e0924`. The focused review suite passed 27 tests in 101.54 seconds. Compilation and `git diff --check` passed. These are contract/fake-runner checks only: no simulator, training, GPU, replay, dashboard restart, outcome measurement, or qualification was performed.

## Reviewer summary

The independent reviewer initially requested changes after identifying that the actual matrix CLI rejected paired-only conditions after evaluation and that report paths could reference stale external files. The corrected exact diff was approved. The reviewer confirmed paired-only finalization bypasses architecture-baseline contrasts, output paths are one-shot, report paths and hashes are checked, and outcomes/qualification remain unavailable/false.

Candidate integration commit: `509b219d3` (`Dispatch paired Phase 4 context validation`). This tranche makes the factors dispatchable; simulator behavior and replay evidence remain outstanding.
