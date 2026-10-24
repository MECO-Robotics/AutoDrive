# MECO inference outcomes and executed decision traces

**Scope:** Make asynchronous inference outcomes and each learned policy's executed decision trace reviewable, with replay alignment checks. This is a scoped transcript summary, not whole-plan approval or runtime qualification.

## Planner pass

The planner identified that aggregate inference counters could not explain a specific decision's fallback or late response, and that evaluator route traces did not always preserve the exact candidate selected when route capture was absent. The planned tranche records bounded per-request outcome diagnostics and a versioned policy-decision trace bound to candidate identity, selected action, actual tick actions, timing, fallback state, and inference provenance. Replay validation checks that the recorded decisions agree with replay actions.

## Implementer pass

Feature commit `6e2db3e48` introduced inference outcome records and the executed policy-decision trace. Corrective commit `7cfe3d07696167d931ced416780435c4f13ddfc1` finalized terminal outcomes for superseded, stale, and closing requests and records completion timing. The implementation paths are `frc_defense/meco_track_training.py`, `frc_defense/meco_tracks.py`, `tests/test_meco_evaluation_dashboard_replay.py`, and `tests/test_meco_inference_broker.py`. The broker keeps a bounded recent outcome history and records request identity, event times/durations, and terminal status without exposing tensors. The evaluator records the selected candidate and action alongside the actual executed action trace and validates replay alignment.

The implementer reported 35 focused tests passing. These tests cover inference outcomes, decision traces, and replay alignment contracts. They do not constitute a simulator evaluation or a fresh dashboard replay review.

## Reviewer pass

The independent reviewer approved exact commit `7cfe3d07696167d931ced416780435c4f13ddfc1`, with 35 focused tests passing. Approval is limited to this implementation and its focused contract checks.

## Limits

No simulator evaluation, fresh dashboard replay, dashboard service/replay review, or qualification evaluation was performed. Those gates remain open; this summary does not claim runtime behavior or learned-policy performance.
