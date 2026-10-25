# Phase 0 MECO simulated score-ledger replay contract

This summary registers scoped implementation and review evidence only. It does
not establish score collection, official match points, qualification, or Phase 0
completion.

## Planner

The approved documentation plan is
`docs/plans/phase0-score-ledger-provenance-update-20261025.md` at
`5a8d5244e1686149b402dddd0f64e7f596ab2284` (plan artifact SHA-256
`104ea63f9b7d01e36b91f449213a9562732cf97e89560d721e43bbddfc3f4084`). It
registers the integrated Phase 0 score-ledger contract while keeping approval,
simulator, replay, dashboard, runtime, device, and qualification evidence
limits explicit. Its source implementation plan is
`docs/plans/phase0-score-ledger-replay-implementation-20261025.md`, SHA-256
`3a553986ad9849e8b482292f19d2b7936e9a050ecc6bc2c7cbde6e9ba3e3672c`.

## Implementation

The integrated implementation commit is
`457c469a5d2e00c56449aaeae0773d3cb6a689ff`. The exact reviewed feature
candidate is `c4f8a591b018e2208dcb3f3b60863155aac6929d`; these are distinct
identities. The implementation defines `meco-simulated-score-ledger-v1`, rules
`meco_simulated_fuel_score_v1`, approval schema
`meco-simulated-score-estimand-review-v1`, and outcome
`simulated_match_score`. It counts only accepted active-HUB FUEL score awards
in `simulated_score_unit`, reconstructs totals from versioned score events,
and preserves the existing FUEL counters and reward path. Official points stay
null. Tower awards, fouls, disqualification, ranking points, end-game bonus,
and advancement points are excluded.

The exact integrated implementation paths are:

- `frc_defense/_tensor_3v3_gamepieces.py`
- `frc_defense/gamepiece_actions.py`
- `frc_defense/meco_simulated_score_ledger.py`
- `frc_defense/meco_track_training.py`
- `tests/test_meco_score_ledger_scorer_path.py`
- `tests/test_meco_simulated_score_ledger.py`

The implementer reported 26 passing pure ledger-contract tests, plus
`compileall` and `git diff --check`. The docs-only ledger update did not rerun
those checks. Both vectorized and robot-loop production scorer-path fixture
cases remain explicitly skipped; they have not passed or run.

## Independent review

The independent reviewer approved exact candidate
`c4f8a591b018e2208dcb3f3b60863155aac6929d` after the scorer fixture was
corrected to place the robot within the red alliance scoring zone and verify
bump clearance. The final candidate also includes the reviewed event-generation,
approval-artifact digest, tick/timestep, and effective eligibility-configuration
bindings. Review did not produce simulator or device evidence.

## Limits

No approved human estimand artifact is present, so outcome collection is not
authorized by an artifact and must remain blocked. No seeded simulator
evaluation, replay generation or inspection, current-dashboard check, runtime
evidence, or device evidence was produced. The two scorer-path cases are
skipped. This contract does not supply official match points or qualification
eligibility; Phase 0 remains incomplete and qualification remains false.

No original planner or reviewer transcript hashes are asserted here. The
dashboard provenance row hashes this materialized summary's exact raw bytes.
