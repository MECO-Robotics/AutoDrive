# Evaluation GPU lease and execution-failure taxonomy

**Scope:** Prevent the MECO evaluator from sharing its indexed GPU with a training worker, while allowing other device lanes to continue. Correct the evaluation failure reason when policy inference falls back or produces a nonfinite decision. This is a scoped transcript summary, not whole-plan approval or qualification evidence.

## Planner pass

The planner traced evaluator polling and worker dispatch and found that a worker could be scheduled on the evaluation device while an evaluator was running. The approved tranche reserves that indexed device during evaluation, exposes deferred lanes, continues work on other devices, and resumes deferred work without changing checkpoint/source bindings or the matched five-track update barrier. It also separates fallback reasons from action remasking, which must reflect the actual selected-versus-executed action trace.

## Implementer pass

Commit `9e41078fd438767217d051b89a58b2110449b3fb` changed `frc_defense/meco_supervisor.py` and `frc_defense/meco_track_training.py`, with focused coverage in `tests/test_meco_supervisor_dispatch.py` and `tests/test_meco_execution_failure_taxonomy.py`. The scheduler waits to start an evaluator until its indexed device is clear, defers only lanes assigned to that device, reports `deferred_for_evaluation`, and keeps the batch alive until those lanes can resume or existing stop/failure handling applies. Evaluation records now distinguish `policy_inference_fallback` and `nonfinite_policy_decision`; the `action_was_remasked` flag derives from the observed action trace. The existing `policy_fallback` field remains for compatibility.

The implementer reported 15 focused tests passing, plus Python compilation and `git diff --check`. These checks cover scheduler leasing/defer/resume and failure-reason classification. No simulator or hardware run, fresh dashboard replay, or qualification evaluation was performed. The change was isolated and was not hot-loaded into the active campaign.

## Reviewer pass

The independent reviewer approved exact commit `9e41078fd438767217d051b89a58b2110449b3fb`. Approval covers the scoped implementation and focused checks only; it does not approve simulator behavior, hardware performance, a fresh replay, or qualification.

## Limits

This is a scoped implementation summary. Simulator evaluation and a fresh replay served by a restarted current-checkout dashboard remain activation gates. No hardware, training-campaign, or qualification result is claimed.
