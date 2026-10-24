# Phase 4 focal-pose assignment — tranche #41

## Scope and evidence

The isolated implementation added a protocol-bound paired no-offset/registered-offset assignment for `full_match_offense` validation. It applies the requested estimate after final scenario reset and before perception and the first policy observation, updates the estimator and policy pose together, and leaves simulator truth unchanged. The evaluator report binds the assignment and records initial truth-versus-estimate diagnostics.

The implementation is commit `fd9ba22eaa35919fcd206323f7615ca8c8166c07`, based on source HEAD `b8f30d9e5a92675d069a740cdca070d04bc330c5`. Independent review approved the exact cached diff SHA-256 `1c9ff7ced88c5175dfc0fba97a83d6a2f6cbdd078839c7b8fa5a8ee0371aee71`. Candidate integration commit is `c4425862bdaac32e9f5206bdcd2a22fe0626692d`.

After integration, 131 focused CPU contract tests passed, along with Python compilation and `git diff --check`. These checks cover assignment/protocol binding, reset application, physical-state invariance, paired matrix lanes, and the separate evaluator report binding.

## Limits

This is a dedicated protocol-bound assignment/report path; generic `phase4_factor_exposure` and `validate_phase4_capture` do not claim focal-pose factor support. The diagnostic is initial-only. No estimator correction or later relocalization event is implemented. Relocalization time, pose-error recovery, safety outcomes, and qualification remain unavailable or false.

No simulator evaluation, E2E replay, GPU/HIP run, training, dashboard currentness check, or qualification was performed. Simulator behavior and policy-consumer behavior still require later E2E validation and replay when the simulator is available.
