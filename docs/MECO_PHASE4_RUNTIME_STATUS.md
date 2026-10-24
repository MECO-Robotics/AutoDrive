# Phase 4 validation runtime status

The validation plan now carries a machine-checked requirement → factor →
runtime-event → required-signal binding for all eleven registry requirements.
The campaign compiler marks a cell executable only when its stimulus is wired
and every required outcome signal has an authoritative source. At this revision
there are **zero executable qualification cells**. A planned row is not an
executed or qualified episode.

| Requirement | Factor ID | Runtime event | Required signals | Current status |
| --- | --- | --- | --- | --- |
| stationary blockage | `blockage.stationary` | none | route completion, time lost to blockage, alternate crossing | Not wired; signals unavailable |
| direct pursuit | `opponent.direct_pursuit` | none | interception, time lost to opponents, robot contact | Sensed-state policy helper exists, but validation slot assignment is not wired; signals unavailable |
| trench interception | `opponent.trench_interception` | none | interception, alternate crossing, trench crossing time | Sensed-state policy helper exists, but validation slot assignment is not wired; signals unavailable |
| feints | `opponent.feint_reversal` | `feint_reversal_tick` | reaction latency, interception, route completion | Existing factor can run and record reversal; all required outcome signals unavailable |
| reactive defenders and opponent pools | `defender.reactive_pool` | none | interception, policy-family coverage, paired outcome by family | Seven-policy helper exists, but validation schedule is not wired; signals unavailable |
| multiple legal team start states | `start.legal_team_poses` | none | legal start validation, outcome by start state | Start-pose helpers exist, but validation reset is not wired; signals unavailable |
| teammate behaviors | `teammate.behavior_pool` | none | outcome by behavior, teammate collision, score contribution | Schedule helper exists, but validation action assignment is not wired; signals unavailable |
| sensor delay and low confidence | `sensor.delay_120ms`, `sensor.low_confidence` | delayed-observation ticks, confidence-cap refresh/reduction | outcome by sensor condition, stale-observation reaction latency, outcome by confidence band | Existing factors capture stimulus; required outcome signals unavailable |
| abrupt opponent change | `opponent.abrupt_policy_change` | `policy_change_tick` | reaction latency, interception, route completion | Existing factor can run and record switch; all required outcome signals unavailable |
| bumps and contact | `contact.robot_bump` | none | robot contact rate, recovery time, route completion | Contact approach helper uses proxy signals only; authoritative pair impulse and route-following recovery are unavailable |
| observation-only focal pose mislocalization | `observation.focal_pose_offset` | none | relocalization time, pose-error recovery, safety contact | Observation-boundary helper exists, but validation reset/observation application is not wired; signals unavailable |

The existing four factor configurations (`feinting_opponent`,
`observation_delay`, `independent_low_confidence`, and
`abrupt_opponent_policy_change`) are accepted by the matched evaluator and emit
a source-bound `meco-phase4-realized-stimulus-capture-v1` record. The record
states whether the expected runtime event was observed, lists only registry
signals with authoritative sources, and marks qualification ineligible when
required signals are missing. An absent or mismatched event remains
`stimulus_not_realized`; proxy signals are never promoted to required signals.
Final-test seed values are omitted from this capture artifact.

The new contract and plan tests are static/contract checks. They do not test
simulator behavior. No simulator episode or fresh replay was run for this
change, and the dashboard stack has not been restarted against this candidate.
The required representative end-to-end simulator evaluation and current-stack
replay inspection remain a separate validation gate.
