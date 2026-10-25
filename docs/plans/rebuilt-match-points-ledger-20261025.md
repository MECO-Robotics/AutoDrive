# REBUILT match-points ledger plan for MECO

**Planner base:** `360c5f074b259b16aee8b2d682ecdbef9e645f1d`
**Scope:** define the official 2026 REBUILT score contract, audit its fit to the currently modeled MECO events, and sequence implementation without interrupting the active r4 supervisor. No implementation, training, simulation, device probe, or replay was run for this plan.

## Pinned authority

Use the English official Game Manual, version TU22, and the official combined Team Updates through TU22 as the only rule sources for this scoring contract. The official Season Materials page states the English current manual is authoritative and links the team updates. Pin the downloaded documents by their exact SHA-256 values in the baseline/rules snapshot; URLs alone are mutable.

| Source | Official URL | Version / retrieved | SHA-256 |
|---|---|---|---|
| 2026 REBUILT Game Manual | <https://firstfrc.blob.core.windows.net/frc2026/Manual/2026GameManual.pdf> | TU22 / 2026-10-25 | `5c67300fc412a1eed8ca9dc37fb6644aee05795a2befed1f69736643cbf95139` |
| REBUILT combined Team Updates | <https://firstfrc.blob.core.windows.net/frc2026/Manual/TeamUpdates/REBUILT_TeamUpdate-Combined.pdf> | Through TU22 / 2026-10-25 | `8ec2754d157b7f35cf93b4be301ebdf576f2abf2650975731dcc3cdf5ab0ecb1` |

The official manual identifies FUEL as scored once it passes through the HUB top opening and sensor array. Its scoring table assigns 1 MATCH point per FUEL in an active HUB and zero in an inactive HUB; AUTO uses the same 1 point per active-HUB FUEL. Both HUBs are active in AUTO, TRANSITION, and END GAME. AUTO FUEL totals determine which alliance is inactive in SHIFT 1; that status alternates at each following alliance shift. If AUTO FUEL ties, FMS randomly selects an alliance for its SHIFT 1 inactive status. The 160-second clock is: AUTO 0–20, TRANSITION 20–30, shifts 30–55, 55–80, 80–105, and 105–130, END GAME 130–160. FUEL assessment extends up to 3 seconds after AUTO ends and after TELEOP ends, and continues for 3 seconds after a HUB deactivates to account for processing time.

TOWER scoring is a separate manual criterion: volunteers assess whether a robot meets the level/contact conditions. AUTO permits only Level 1 (maximum two robots); TELEOP awards one level per robot, with Level 1 worth 10, Level 2 worth 20, and Level 3 worth 30 points. A robot earning AUTO Level 1 may earn additional TELEOP points. MINOR and MAJOR FOULs add 5 and 15 MATCH points to the opponent. The manual defines RED CARD / DISQUALIFICATION consequences differently for qualification and playoff matches. Win/tie and bonus Ranking Points are not MATCH points and must not be added to the MATCH score.

Rule references for implementation: manual §§6.4–6.6, §7.4, §10.5–10.6; TU22 and earlier updates only when they alter those rules. Q&A, cheat sheets, team summaries, and simulator comments are not substitute scoring authority. Re-run the authority pin if official documentation changes; an old protocol stays bound to its original source digest.

## Audit of the current evaluator

- `frc_defense/meco_baseline_freeze.py` now correctly maps scoring semantics to `fuel_score_count` / scored FUEL-piece counts and rejects a claim that this is official match points. It documents the MECO construction path as `tensor_training_runtime` → `TensorDefenseEnv`.
- `frc_defense/tensor_sim.py::TensorDefenseEnv` has two robot slots, one on each alliance, and a planar 2D chassis model. It does not instantiate the three robots per alliance required for a full match. `frc_defense/meco_track_training.py` currently writes replay metadata with literal `robot_count: 6`; this does not match `TensorDefenseEnv`'s two robot slots and must be corrected before score provenance can claim roster coverage.
- `frc_defense/gamepiece_actions.py::update_fuel_actions` records a scored-FUEL event when the simplified action/pose/Hub checks pass. There is no projectile crossing of a modeled HUB opening/sensor array. The event timestamp is an abstract score-action time, not an authoritative sensor-crossing time; there is no in-flight FUEL processing queue to support the manual's 3-second accounting windows.
- `frc_defense/tensor_defense_gamepieces.py::_update_match_clock` advances a 160-second clock and compares AUTO FUEL to choose an inactive-first alliance, but its implementation starts alliance SHIFT 1 at elapsed 20 and changes status every 30 seconds. The manual's TRANSITION keeps both HUBs active from elapsed 20–30, and the four 25-second shifts begin at 30, 55, 80, and 105. Its `auto_fuel_scores` only counts its two simulated robots and is not complete three-robot-per-alliance AUTO FUEL.
- `frc_defense/tensor_defense_reset.py` resets fuel and HUB counters, but there is no independent match-score ledger, per-score event journal, Tower result, referee event ingress, DQ context, or completeness status.
- `frc_defense/tensor_defense_step.py` exposes per-step fuel counts/events and uses them for reward. `frc_defense/meco_track_training.py` emits metrics and replay records but does not emit manual-point totals or a ledger digest. `frc_defense/meco_qualification.py` calibrates throughput and safety metrics; it cannot qualify match points.
- Existing contact, collision, wall, or safety metrics are not Referee decisions. The manual's penalties are assigned as perceived by the Referee and many rules require rule-specific context, intent, repetition, escalation, or human adjudication. Never infer a foul from a collision/contact event or infer “no foul” from no collision.
- No Tower mechanism, vertical robot pose, rung-contact state, bumper-height test, or human Tower assessment is modeled in `TensorDefenseEnv`. A 2D pose or “end-game near Tower” heuristic cannot score Tower points.

## Contract: two explicitly different outputs

### 1. Restricted fuel-points training proxy

Name this output `rebuilt_active_hub_fuel_proxy_v1`; never serialize it as `match_points`. It may sum only the current simulator's explicit `fuel_scored_event` events valued under the corrected HUB schedule. Each record must state `scope=two_robot_2d`, `event_semantics=abstract_score_action_not_sensor_crossing`, `tower_points_complete=false`, `fouls_complete=false`, `alliance_roster_complete=false`, and `qualification_eligible=false`. Preserve acquisitions, fuel throughput, and all prior metrics as separate fields; do not fold them into score. Keep the existing objective available for the active r4 process; do not change its checkout, checkpoints, reward, manifest, or service.

This proxy is a shared training signal only. It can support comparable future A1, A2, A3, C1, and C2 training because it is exposed through the common simulator reward/metrics seam, but it cannot establish an official match result or pass the source plan's match-point gate. A future five-track manifest must bind the same proxy contract/digest, per-track configuration, case/seed partition, and update budget. All five must use the same score/reward units; report each policy's model identity and the proxy's limits. Wait until the user ends r4 and both WX9100 leases are available before launching any new five-track GPU campaign.

### 2. Complete REBUILT MATCH-points ledger

Use `official_match_points` only when all score-bearing inputs are complete for the defined match context. The ledger is alliance-scoped, with per-team/per-robot source attribution where modeled. Suggested immutable schema `rebuilt-match-points-ledger-v1`:

- Bind the pinned rules snapshot digest, source commit/tree, baseline freeze, simulator/runtime config, match/evaluation ID, reproducible seed, qualification/playoff context, exact team/alliance roster, robot-slot/team/alliance map, and the actual seeded FMS tie-break result when AUTO totals tie.
- Store immutable ordered event rows with event ID, simulated time/tick, event class, alliance and team/robot attribution, source event/reviewer, manual rule reference, phase, HUB state, point delta, and prior/result ledger digests. Reject duplicate event IDs, inconsistent clocks, impossible phase transitions, unknown team/alliance IDs, invalid rule points, and event mutation after finalization.
- FUEL rows require a represented sensor-crossing/processing event and score-window timestamp, not only the current action-intent event. Classify AUTO versus TELEOP by the official assessment window; count AUTO FUEL used to determine SHIFT 1 independently of the later alliance-match score sum. Credit a FUEL only under the pinned HUB active/grace rules. A per-team attribution is permitted only when the gamepiece's scoring robot/team is known; the alliance is always the scoring owner.
- TOWER rows require a reviewed human adjudication record (or a future validated 3D rung/contact model plus its review process): robot/team/alliance, period, level, contact/bumper criteria, source replay/frame IDs, adjudicator identity/time, and rule snapshot. Enforce AUTO L1 cap of two robots, one TELEOP level per robot, and no duplicate awards; permit the manual's additional TELEOP scoring after AUTO L1.
- FOUL rows require explicit referee/adjudicator input with team/alliance, rule ID, severity, instance or interval, repetition/escalation linkage, card/DQ state, point attribution, evidence record, and reviewer. A raw simulator contact cannot create one. Enforce MINOR=5 and MAJOR=15 opponent points only when that exact adjudicated event says so; suppress/replace penalties when the authoritative adjudication defines a higher-level consequence.
- DISQUALIFICATION rows bind qualification versus playoff context and exact affected team/alliance. Apply qualification semantics to the disqualified team result and playoff semantics to the alliance score as written in the pinned rules. Do not treat a red-card state as merely another foul-point increment.
- Keep MATCH score points separate from ranking-point calculations (win/tie RP and threshold bonuses). If ranking points are later needed, add a separate ledger and event-context inputs; never conflate the two.
- The final result includes `official_match_points` only if all six alliance robot slots and their score sources, fuel sensor/processing events, Tower adjudications, referee penalty coverage, match context, and end-of-match scoring windows are declared complete. Otherwise set it to null with a machine-readable incompleteness list, and expose only the restricted proxy or partial ledgers by their own names.

## Incremental implementation and review slices

### Slice A — rule snapshot and pure scoring contract

Add `qualifications/meco/rebuilt-rules-TU22.json` with the two official URLs, retrieved timestamp, TU22 version, SHA-256 pins above, applicable sections, time boundaries, point values, Tower/penalty/DQ semantics, and explicit exclusions. Add `frc_defense/rebuilt_match_scoring.py` for pure event validation, HUB schedule derivation, point aggregation, separate proxy/full-ledger result types, canonical event and ledger digests, and completeness reasons. Add `tests/test_rebuilt_match_scoring.py` using hand-constructed deterministic rows; cover period boundaries, AUTO/FMS tie, Hub shift alternation, 3-second scoring windows, active/inactive FUEL, Tower caps/levels, point arithmetic, foul attribution, context-specific DQ, RP separation, duplicate or mutated events, and incomplete-ledger refusal. No simulator construction is needed for these pure contract tests.

### Slice B — correct effective HUB schedule and fuel-proxy event seam

Modify `frc_defense/tensor_defense_gamepieces.py`, `frc_defense/gamepiece_actions.py`, `frc_defense/tensor_defense_reset.py`, and `frc_defense/tensor_defense_step.py` to implement the pinned clock/status boundaries and expose a per-world `rebuilt_active_hub_fuel_proxy_v1` ledger without changing the existing raw `fuel_score_count` contract. Decide explicitly whether raw simulated score events inside the 3-second windows are generated only from pending in-flight sensor events; because the present model has no projectile/sensor crossing, do not invent those events. Until such modeled events exist, keep its point proxy completeness false and label the immediate action-score contribution as a legacy diagnostic submetric. Add focused CPU/scoring behavior tests and device-kernel parity tests only for state changed on device. Any test exercising simulated HUB/fuel behavior must also be accompanied by a representative end-to-end simulator evaluation with fixed scenario/seed and inspected replay; after code changes restart and verify the dashboard backend and frontend hashes before replay review.

### Slice C — replay and training provenance for five tracks

In `frc_defense/meco_track_training.py`, serialize the contract ID/version/hash, scope/completeness flags, alliance/robot attribution, per-event timeline/digest, and source rule digest into evaluation records and replay frames. Replace literal robot count metadata with actual `env.sim.pose.shape[1]`; add validation preventing a two-slot run from claiming full 3v3. Update evaluation-record/replay validators and relevant tests so stale/missing/mismatched ledger data is rejected and legacy score rows remain explicitly diagnostic. Add one shared optional training reward/metric route for all five tracks that uses the restricted proxy only, with a manifest-bound objective identifier. Do not change the ongoing r4 objective or its source. This slice produces no match qualification result.

### Slice D — complete-ledger inputs and official endpoint eligibility

Only after broader match simulation is available, add a roster-complete six-robot path and referee/Tower evidence ingress. The current `TensorDefenseEnv` path is not enough. A candidate route is the existing six-robot `TensorThreeVsThreeEnv`, but it must be audited and upgraded against the same rules snapshot before it can feed the official ledger. Extend replay capture to include all six robot states, exact FUEL sensor-equivalent events and the 3-second windows; capture reviewed Tower result frames and referee annotation event hashes. Add report intake requiring complete roster/event coverage and refusing any `official_match_points` output from two-robot, missing-tower, or missing-referee-input runs.

Only after Slice D meets those contracts may `frc_defense/meco_qualification.py` and the planned `meco-qualification-v5` consume `official_match_points` as a primary outcome. Bind the manual/rules digest, ledger schema, event completeness, match context, CI, and reviewed useful-gain/safety margins. Until then, qualification remains unavailable and reports must say so.

## Invariants and acceptance evidence

- The pinned official manual/rules digest and all point values are required; code constants without the digest are not accepted.
- AUTO/TELEOP score periods and HUB states match the manual's exact boundaries; HUB tie selection is explicit and reproducible. The current 20-second/30-second shift schedule is corrected before new proxy collection.
- Every score uses a uniquely identified FUEL sensor-equivalent event or reviewed TOWER/FOUL/DQ row. Intent, contact, FUEL acquisitions, simulated HUB approach, and policy reward are not scoring evidence.
- Alliance score, per-team attribution, match-point totals, and Ranking Points are different outputs. Partial categories produce `official_match_points=null` plus reason codes, never a zero-filled “complete” result.
- Two-robot training events may feed only the labeled restricted proxy. They cannot create a complete official match score, qualified report, match-point confidence interval, or promotion claim.
- r4 keeps running untouched. No new campaign starts or consumes either WX9100 in this tranche. A later five-track training pass must be a fresh source-bound campaign after r4 stops naturally or the user stops it, and must apply one common proxy contract to A1/A2/A3/C1/C2.
- Validation reporting for behavior changes includes focused tests, the required representative end-to-end seeded simulation and inspected replay ID, and dashboard service/API/frontend currentness evidence. No such evidence was produced by this planner.
