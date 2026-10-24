# Action-7 evaluation remask taxonomy — scoped summary

## Planner scope

The planner inspected two existing diagnostic evaluation records from source
identity `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350`:
`evaluations/meco-five/cycle-000001-3f55a2d8.json` and
`evaluations/meco-five/cycle-000002-20e2aaab.json`. Action 7 was a genuine
selected action for all policies in the inspected evidence (C1 selected it in
83/84 rows in cycle 2). A2/A3 rows carried deterministic-fallback inference
provenance. Fallback-only records could be mislabeled as action remasking when
neither target loss nor native preemption applied. This was a record
classification defect; the planner did not run code, tests, or simulation.

## Implementation

The implementation in candidate commit
`4d040f5e2238e652e90c745300394e07aa1a575f` adds an explicit comparison of the
selected action against every executed tick action and uses that trace to set
the remask flag. A fallback with a trace consisting only of selected action 7
therefore retains the fallback reason and is not mislabeled as remasking. The
failure-reason precedence for selected-target loss and native preemption is
covered. Changed paths are `frc_defense/meco_track_training.py` and
`tests/test_meco_execution_failure_taxonomy.py`.

## Independent review and checks

The scoped independent reviewer `/root/action7_taxonomy_review` approved
implementer HEAD `accce779be3d1a9e87092e22ac45ae7a6543d01d`, which is carried
in the candidate as `4d040f5e2238e652e90c745300394e07aa1a575f`. Nine focused
tests passed; Python compilation and `git diff --check` passed.

## Limits

This corrects evaluation record taxonomy only. It does not establish that
action 7 was learned or selected because of learned policy behavior, and it
does not establish simulator behavior, qualification, or track success. No
simulator, GPU, training, replay, or dashboard validation was performed for
this tranche.
