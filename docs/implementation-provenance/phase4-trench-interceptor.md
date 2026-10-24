# Phase 4 trench-interceptor validation wiring

## Planner pass

The scoped planner found a registered 14-category trench domain and an existing sensed-state `trench_interceptor` policy, but no validation schedule binding or authoritative interception, alternate-crossing, and trench-crossing outcome signals. The implementation is validation-stimulus wiring only; outcome signals remain unavailable.

## Implementer pass

Commits `8a147de049c656909b97e315212b0b9cf897aa76` and `7c295d92da8b228e8cbd52b5a7aaf66c9dfc59a0` bind per-category validation schedules to the immutable 14-category domain, protocol ID, case, seed list, factor, and policy. Campaign dispatch emits one category per cell and the evaluator validates it against the actual cell. Runtime command slots are level 0: none; levels 1–4: slot 0; level 5: slots 0 and 1. The generic opponent runtime domain remains unchanged.

Reviewed source diff SHA-256: `c0bf31416a8ae24a020354b2d4277d0ce54f57e406d9dd0978f75ea26e014973` at source HEAD `41fb3eb5799c90a66fef098bff851d64daa897f1`.

## Independent reviewer pass

The independent reviewer approved the exact diff after verifying campaign dispatch and evaluator schedule binding, the frozen domain, slot mapping, and compatibility with existing reactive opponent schedules. No source changes were made by the reviewer.

Candidate integration commits: `01fb20c68dbf90ffa921266937c3a2f7c52843ca` and `ed2f986231ddc4e57d7ac68227d7356f89e48c20`.

## Evidence limits

Thirty-seven focused contract tests passed, including dispatch and pre-device evaluator binding checks; compilation and `git diff --check` passed. The checks do not validate simulated game behavior. No simulator evaluation, E2E replay, GPU/HIP run, training, dashboard replay, outcome measurement, or qualification was performed. `interception_rate`, `alternate_crossing_success_rate`, and `trench_crossing_time_s` remain null; qualification remains false.
