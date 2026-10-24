# Async inference broker contract tranche

**Scope:** Section 4 snapshot freshness and fallback contract; Phase 5A broker lifecycle evidence. This is a scoped summary of the planner, implementer, and reviewer reports, not whole-plan approval or runtime qualification.

## Planner pass

The planner identified missing focused contract coverage for the production asynchronous inference broker. The broker already implemented single-slot replacement, latest-sequence publication, freshness checks, fault counters, and worker shutdown, but its queue-replacement and lifecycle behavior had static review only. The recommended tranche was deterministic CPU testing with a fake adapter and synchronization events, with production changes limited to a demonstrated bug. GPU-load freshness, simulator behavior, dashboard replay, and Hailo measurements remain runtime gates.

## Implementer pass

Commit `52d2c8314de23249129bda037dadc07bc36cdafa` added `tests/test_meco_inference_broker.py` covering pending-request replacement, request identity rejection, one-time publication, poll-before-completion non-reuse, expired/empty/raised inference faults, and idempotent close/join/reject-after-close. The raised-inference test exposed a real failure mode: an exception could terminate the sole worker. `frc_defense/meco_tracks.py` now counts that request as a fault and keeps the worker available for later snapshots. The deployment profile now distinguishes these fake-adapter contract tests from real accelerator, evaluator, simulator, and deployment evidence.

Validation reported by the implementer: 5 focused tests passed; Python compilation and `git diff --check` passed. No GPU, simulator, replay, dashboard service, or active campaign was used.

## Reviewer pass

The reviewer approved exact commit `52d2c8314de23249129bda037dadc07bc36cdafa`. Review confirmed the exception is isolated to one request, the worker continues, event barriers exercise the queue and close paths deterministically, and the deployment profile does not claim accelerator or simulator validation.

## Limits

This tranche verifies local broker contracts with a fake CPU adapter. It does not demonstrate freshness under GPU load, real evaluator checkpoint lifecycle, simulator fallback, replay correctness, Hailo compatibility, latency, or robot actuation.
