# Phase 4 validation runtime status

The validation plan now carries a machine-checked requirement → factor →
runtime-event → required-signal binding for all eleven registry requirements.
The campaign compiler marks a cell executable only when its stimulus is wired
and every required outcome signal has an authoritative source. At this revision
there are **zero executable qualification cells**. A planned row is not an
executed or qualified episode.

| Requirement | Factor ID | Runtime event | Required signals | Current status |
| --- | --- | --- | --- | --- |
| stationary blockage | `blockage.stationary` | route-goal lifecycle endpoint record | route completion, time lost to blockage, alternate crossing | Existing factor capture now retains a fully observed, exact-bound route completion diagnostic when available; no representative replay; other required outcomes remain unavailable and qualification is disabled |
| direct pursuit | `opponent.direct_pursuit` | none | interception, time lost to opponents, robot contact | Sensed-state policy helper exists, but validation slot assignment is not wired; signals unavailable |
| trench interception | `opponent.trench_interception` | none | interception, alternate crossing, trench crossing time | Sensed-state policy helper exists, but validation slot assignment is not wired; signals unavailable |
| feints | `opponent.feint_reversal` | `feint_reversal_tick` plus route-goal lifecycle endpoint record | reaction latency, interception, route completion | Existing factor capture can retain a fully observed, exact-bound route completion diagnostic; reaction latency/interception remain unavailable, no representative replay has validated it, and qualification is disabled |
| reactive defenders and opponent pools | `defender.reactive_pool` | `records[].phase4_opponent_policy.applied_event_coverage_complete` | interception, policy-family coverage, paired outcome by family | Validation rows bind exact per-seed policy/family/slot assignments and require applied command coverage. Exact paired whole-team outcomes can be grouped by policy family as an association. Interception remains unavailable because no approved reactive-pool interception criterion/signal exists; qualification remains disabled. |
| multiple legal team start states | `start.legal_team_poses` | none | legal start validation, outcome by start state | Start-pose helpers exist, but validation reset is not wired; signals unavailable |
| teammate behaviors | `teammate.behavior_pool` | `records[].phase4_teammate_behavior.action_event_coverage.complete` | outcome by behavior, teammate collision, score contribution | Exact assigned/requested/helper-selected/actually-applied action traces are checked for every episode tick and paired by identical policy/category/seed. Reported behavior outcomes are whole-team paired episode metrics while behaviors were present, not individual causal credit. No slot-pair-specific collision ledger or robot-attributed score ledger exists, so teammate collision and score contribution remain null; qualification remains disabled. |
| sensor delay and low confidence | `sensor.delay_120ms`, `sensor.low_confidence` | delayed-observation ticks, confidence-cap refresh/reduction | outcome by sensor condition, stale-observation reaction latency, outcome by confidence band | Existing factors capture stimulus; required outcome signals unavailable |
| abrupt opponent change | `opponent.abrupt_policy_change` | `policy_change_tick` plus route-goal lifecycle endpoint record | reaction latency, interception, route completion | Existing factor capture can retain a fully observed, exact-bound route completion diagnostic; reaction latency/interception remain unavailable, no representative replay has validated it, and qualification is disabled |
| bumps and contact | `contact.robot_bump` | `phase4_contact_exposure`, `phase4_pair_contact_event`, `first_route_progress_interval`, `selected_goal_endpoint_completion_rate` | robot contact rate, recovery time, route completion | Three registered geometries flow through diagnostic matrix conditions and reset; slot 2 receives the sensed-target approach while slot 3 remains the evaluated policy. Records separate exposure, solver-confirmed pair (2,3) contact, route crossing, an outer-tick interval-censored route-progress observation, and an episode-wide selected-goal endpoint diagnostic. Neither diagnostic is contact-caused recovery or registry route completion; D0 has no selected-goal ledger, and qualification stays disabled. |
| observation-only focal pose mislocalization | `observation.focal_pose_offset` | none | relocalization time, pose-error recovery, safety contact | Observation-boundary helper exists, but validation reset/observation application is not wired; signals unavailable |

The generic factor capture also retains the existing route-goal lifecycle
outcome diagnostic for stationary blockage, feint reversal, and abrupt policy
change when every route attempt has a fully observed, valid completion or
registered failure and exact episode/source/policy/factor/condition binding.
The validator reconciles aggregate counts against each attempt row. This signal
remains non-authoritative for the registry and qualification-ineligible;
contact remains on its separate capture path. See
`docs/implementation-provenance/phase4-route-factor-capture.md`. No simulator
replay was run for this wiring tranche.

The existing four factor configurations (`feinting_opponent`,
`observation_delay`, `independent_low_confidence`, and
`abrupt_opponent_policy_change`) are accepted by the matched evaluator and emit
a source-bound `meco-phase4-realized-stimulus-capture-v1` record. The record
states whether the expected runtime event was observed, lists only registry
signals with authoritative sources, and marks qualification ineligible when
required signals are missing. An absent or mismatched event remains
`stimulus_not_realized`; proxy signals are never promoted to required signals.
Final-test seed values are omitted from this capture artifact. The separately
opt-in `contact.robot_bump` record reads simulator-generated pair-contact
summaries from the Torch and fused HIP collision solvers. It binds the
registered slot pair (2,3), first and last completed physics-substep times,
distinct contact-substep count, and accumulated impulse. The contact summary
does not contain a complete event stream and does not infer recovery or route
completion. Contact geometry conditions are diagnostic-only and explicitly
qualification-ineligible. They are blocked from final-test use. Route crossing
is recorded as a separate observed event and is not promoted to route
completion.

For stationary blockage, feint reversal, and abrupt policy change, the generic
factor capture now embeds the existing route-goal lifecycle record only when
all attempts have valid completed or explicitly registered failure outcomes,
with no censoring, and its episode, source digest, policy, factor ID, and
condition exactly match the capture. Timeout, abandonment, replacement, or
missing measurements remain censored. The route record is exposed as a
non-authoritative diagnostic; it does not make any cell executable or
qualification-eligible, and no selected-goal endpoint proxy is used.

The separate `first_route_progress_interval` diagnostic checks whether
remaining distance along the current executable route to the pinned selected
goal later decreases after contact. Its lower bound is the latest solver
contact timestamp; route-distance observation bounds are outer ticks. It
records route fingerprints and censors changed goals or invalid clocks,
counts, or routes. It is not `route_completion_rate`,
`recovery_time_after_bump_s`, or qualification evidence.

The reactive-pool and teammate-behavior outcome bridge is implemented as a
diagnostic evidence interface. Reactive rows retain canonical per-seed policy
assignment digests, configured/applied slot identity, and full physics-tick
command coverage; paired validation reports retain whole-team episode outcome
contrasts grouped by policy family. These contrasts do not claim physical
interception or policy-specific causal contribution. Teammate rows bind the
two scheduled slots to complete per-tick requested/helper-selected/applied
action records and pair exact same-policy/category/seed episode outcomes by
behavior association. Teammate collision and individual score contribution
remain null because the simulator does not expose those slot-specific ledgers.
No representative simulator campaign or replay review has been accepted, and
the Phase 4 qualification gate remains closed.

The nested `selected_goal_endpoint_completion_rate` is computed from valid
goal-attempt ledger rows across the episode while a contact geometry is
assigned. It binds each row to the episode, source identity, geometry, and
selected goal; fallback, remasked, preempted, lost-target, and otherwise
invalid attempts are excluded and counted. Heading is required only when the
selected goal carries a finite heading. It does not claim that a contact
caused the endpoint result, is not registry `route_completion_rate`, and is
not a recovery measurement. D0 lacks selected-goal ledger rows, so its rate is
null.

The new contract and plan tests are static/contract checks. They do not test
simulator behavior. No simulator episode, HIP extension compile, GPU behavior
test, or fresh replay was run for this change. After the prior candidate
provenance integration, the dashboard service was restarted at 10:48 UTC
against that worktree; `/meco`, `/api/meco-status`, `/api/curriculum-status`,
and `/assets/meco.js` returned HTTP 200. That API snapshot validated 30
provenance summaries and the served frontend asset hash matched that checkout.
The current metadata-only update adds summaries 31 and 32 and updates the
dashboard manifest digest; the service has not been restarted or rechecked
after this update. No replay was generated, so simulator behavior and replay
inspection remain outstanding.
The required representative end-to-end simulator evaluation and current-stack
replay inspection remain a separate validation gate.
