# Phase 4 selected-goal endpoint diagnostic

## Planner pass

The scoped request was to add a narrow `selected_goal_endpoint_completion_rate`
diagnostic from valid selected-goal attempt ledger rows and place it inside the
Phase 4 contact exposure capture. Attempts must bind to episode, source
identity, contact geometry, and selected-goal identity. Invalid, fallback,
remasked, preempted, or lost-target attempts are excluded with reasons. A
selected goal carrying a heading requires both endpoint distance and heading
error to meet the existing tolerances. An empty valid denominator returns a
null rate.

## Implementer pass

The implementation commit is
`9462bbd36756d79c1001d6b90c9c5f31d7c94f89` (`Record selected goal endpoint
diagnostic`). It changes `frc_defense/meco_metric_ledger.py`,
`frc_defense/meco_phase4_runtime_adapter.py`,
`frc_defense/meco_track_training.py`, and
`tests/test_meco_selected_goal_endpoint_contract.py`. Attempt rows carry the
execution exclusions and episode/source/geometry/goal bindings. The diagnostic
reports numerator, denominator, exclusion counts, goal IDs, heading semantics,
and the episode-wide observation scope. Contradictory validity flags and
exclusion reasons fail closed.

## Reviewer pass

The reviewer requested rejecting rows that claimed `execution_valid=True`
while also carrying execution exclusion reasons. The correction counts both
the explicit reason and the validity contradiction, and the focused contract
suite covers it. The exact corrected commit above was approved.

## Checks and limits

Thirty-five focused selected-goal, Phase 4 runtime-adapter, and contact-geometry
contract tests passed. Python compilation and `git diff --check` passed. No
simulator, GPU/HIP, training, campaign, dashboard, or replay validation was
run. This is an episode-wide selected-goal endpoint diagnostic during assigned
contact geometry; it does not attribute progress to a contact event, measure
recovery, or establish registry `route_completion_rate`. D0 has no selected-
goal ledger, so its rate is null. Authoritative signal status remains false
and qualification remains disabled.
