# Phase 0 pass 1: simulated score ledger and replay reconstruction

**Status:** implementation plan only. This document creates no scoring evidence, simulator run, freeze, calibration, or qualification result.

**Planner base:** `a5aebfb034e0329ed66f097ea38d8cf4e93723ee` (`codex/phase0-matched-calibration-plan-20261025`).
**Source plan:** `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`, SHA-256 `718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`.
**Phase 0 parent plan:** `docs/plans/phase0-baseline-heldout-calibration-20261025.md`, SHA-256 `95e1b9185509259af952fab7fc956c12974d50aacf43525302275f859f917d49`.
**Scope:** Phase 0 pass 1, steps 1–2 only: trace the actual score-event path and implement a versioned, replay-reconstructable *simulated* match outcome. This plan is a prerequisite to the source/runtime freeze and all outcome calibration. It does not implement pass 1 steps 3–5 or later calibration/protocol work.

## Goal and semantic boundary

For each scored simulator episode, retain an immutable event ledger whose outcome can be recomputed from its ordered events and frozen simulator scoring contract, and can be checked independently against a fresh replay. Report the simulator's modeled score explicitly as a simulated outcome. Do not call it official FRC match points, ranking points, or policy-attributed causal contribution. Keep the production FRC scorer (`official_match_points`) separate; current simulator events do not contain the sensor, referee, tower adjudication, match roster, and disqualification evidence required by that scorer.

The current six-robot simulator increments `fuel_score_count` by accepted score actions, and the step reward uses the current tick's `fuel_scored_event` as a zero-sum signal. Those values are useful cross-checks and training signals, but are not themselves the new outcome record. The versioned scorer must derive each alliance's simulated score by summing explicit, immutable score-award events with a rule-defined value, then compare the derived result with the simulator counters. Preserve scored-FUEL count and useful throughput under their existing names and meanings. Never alias either field to `match_points`.

The model's current score semantics award one simulated score unit when a held FUEL piece passes the simulator's existing score eligibility gate: valid score intent, active HUB, alliance zone, robot readiness, match cadence, and active world. The implementation must pin these semantics and the source/runtime identity. If source inspection during implementation finds another score-bearing event or terminal adjustment in the evaluated simulator path, include it in the rule contract or keep the outcome incomplete; do not infer it away.

## Source audit: current production paths and gaps

| Stage | Current source | Observed behavior | Required seam |
|---|---|---|---|
| Match clock and phase | `frc_defense/tensor_3v3.py`, `TensorThreeVsThreeEnv.step()` and match update | Advances one physics step, calls match update, then score update | Bind score events to the authoritative simulator step/tick and elapsed match time. |
| Score decision and mutation | `frc_defense/gamepiece_actions.py`, `update_fuel_actions()` (installed as the environment score updater) | Clears per-step event tensors; checks readiness, HUB, zone, score intent and cadence; chooses held FUEL, increments `fuel_score_count`/AUTO count, and sets `fuel_scored_event`. The accepted-piece identity is available at the `chosen` mask but is not durably recorded. | Emit an event at the exact successful mutation site in both vectorized-six-robot and fallback robot paths. Event capture must be inactive by default outside the opted-in evaluation recorder or otherwise have bounded, explicit allocation cost. |
| Reward | `frc_defense/tensor_3v3.py`, `step()` after `_update_fuel()` | Derives zero-sum step reward from per-tick `fuel_scored_event`; no general FRC match reward/bonus/penalty ledger is emitted. | Preserve reward calculation. Tests compare exact reward and counters with score-event capture enabled/disabled. |
| Episode metric row | `frc_defense/meco_metric_ledger.py`, `MecoEpisodeMetricLedger.observe_outer_step()` / `to_record()`; `frc_defense/meco_track_training.py`, episode finalization | Stores focal-robot aggregate `fuel_scored`, safety and other descriptive metrics; not a six-robot event history or alliance score outcome. | Add a separate episode score-ledger object to the evaluation row; do not overload `meco_metric_ledger` or its aggregate `fuel_scored`. |
| Replay frame | `frc_defense/meco_track_training.py`, `_capture_meco_replay_frame()` | Captures renderer snapshots including `fuel_score_count`, `hub_active`, clock and pieces. Frames are decision/macro snapshots; intermediate score events can occur between them. | Store tick-level score events in the replay artifact or a digest-bound event stream embedded in it. Replay reconstruction must not rely on differencing sparse counter snapshots. |
| Replay integrity/export | `meco_track_training.py`, `_validate_meco_replay_alignment()`, `_evaluation_record_digest()`, and replay export construction | Aligns evaluation records and frames and hashes summary/outcome material; score event rows and an independent score recomputation are absent. | Bind score-ledger digest and event-stream digest into evaluation record/replay identity, validate them on export/intake, and independently recompute from replay events. |
| FRC scoring contract | `frc_defense/rebuilt_match_scoring.py`, `build_score_ledger()` / `validate_score_ledger()` | Separates confirmed FUEL sensor events and adjudicated tower/foul/DQ/ranking events from an abstract `rebuilt_active_hub_fuel_proxy_v1`; its official result remains incomplete when source coverage is absent. It is a pure evidence contract and is not wired to simulator events. | Keep this contract and `official_match_points` untouched. Add a distinct simulator-outcome contract; do not feed `fuel_score_count` or proxy score actions into the official scorer as sensor or adjudication evidence. |

Before implementation, confirm the bindings above in the exact source commit selected by the implementer. If the trainer's nested `env.step()` loops bypass one capture site, instrument the common simulator scoring mutation instead of relying on evaluator macro boundaries.

## Implementation contract

Implement one new pure, versioned simulator score-accounting module and integrate it at the simulator's accepted score-event mutation and MECO episode/replay boundaries. Proposed public identifiers:

- Schema: `meco-simulated-score-ledger-v1`.
- Scoring contract: `meco_simulated_fuel_score_v1`.
- Replay event stream: `meco-score-events-v1`.
- Outcome field: `simulated_match_score` with `{schema, rules_id, rules_sha256, red, blue, complete}`. It is explicitly a modeled simulator outcome; `official_match_points` remains null/unavailable unless separately supported by the existing evidence contract.

Each normalized event must bind at least: unique deterministic `event_id`; zero-based physics `tick`; finite `match_elapsed_s`; simulator world slot; alliance/team and robot slot; accepted piece index or stable piece ID; event type `fuel_score_awarded`; integer `score_units` fixed by the pinned rules contract; and source event site/semantics. The event order is `(tick, event_id)`. Reject duplicate IDs, unknown slots/alliances, non-finite or out-of-range clocks, negative units, impossible duplicate piece award in one episode, reordered rows, and event rows inconsistent with the simulator's accepted-score source. Preserve enough runtime/source identity to reject cross-source replay.

The episode ledger must include exact match context (seed, scenario, team/robot slot map, simulator source identity, and effective scoring configuration identity), rules identifier and content digest, canonical event rows and per-event digests, aggregate event digest, red/blue derived totals, independent terminal/censoring state, and its own canonical digest. It must distinguish `complete`, `incomplete`, and `invalid`; incomplete source coverage must yield no complete outcome, never zero. No per-robot causal contribution is reported from whole-team score events. A robot-attributed subtotal may be emitted only as a descriptive event attribution with no causal language.

Replay integration must place the exact ordered event rows (or a lossless, digest-bound event stream) in the replay artifact at their original physics ticks. A pure reconstruction function consumes only replay identity/context, the pinned score rules and replay event stream; it recomputes all team totals and the ledger digest. It must not read the evaluation row's totals, `fuel_score_count`, `fuel_scored_event` aggregates, throughput, or training rewards as inputs. The replay validator then compares reconstructed output with the evaluation row and the terminal simulator counters, failing closed on any disagreement.

The simulator's currently modeled outcome is intentionally limited to its actual FUEL score-award mechanics. `tower_award`, foul/penalty, disqualification, ranking points, end-game bonuses, and official advancement points are not implied or silently synthesized by this simulated contract. If a future simulator adds such mechanics, a new versioned rule/event contract is required or the existing episode must be marked incomplete.

## Planned file changes

1. `frc_defense/gamepiece_actions.py`: add a bounded score-award event capture seam at both successful `chosen`/`count` mutation branches, with identical event semantics. Include deterministic per-episode sequencing and source identity supplied by the evaluation harness; no random draws or changes to scoring/reward.
2. `frc_defense/tensor_3v3.py`: initialize/reset opt-in event capture state and bind events to per-world physics tick, match clock and world identity. Verify the reset/active-mask contract so inactive worlds do not emit events. Preserve all score tensors/reward outputs byte-for-byte under event capture.
3. New `frc_defense/meco_simulated_score_ledger.py`: implement strict schema validation, rules digest, canonicalization, aggregation, event/ledger hashing and replay-only reconstruction. Keep this separate from `rebuilt_match_scoring.py`'s actual-FRC evidence and abstract proxy contracts.
4. `frc_defense/meco_track_training.py`: create a recorder for each episode; drain/attach events without losing nested `env.step()` events; attach ledger to evaluation rows; include event stream and ledger binding in the replay record and record/replay digests; validate replay reconstruction before writing. Keep `meco_metric_ledger`, `metrics.scores`, `useful_throughput`, and official-score fields distinct.
5. Tests: add focused pure-ledger tests and production-path integration tests to existing scoring/replay test modules or new narrowly named modules. Extend replay mutation tests so a well-formed but changed event stream cannot pass by updating only a summary digest.
6. `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` and three-pass provenance: update only after exact-head review and integration, explicitly separating code contract, unit evidence, seeded simulation/replay evidence, and still-missing freeze/calibration/qualification.

The implementer should first inspect the pinned baseline source for exact function names and current test topology, then split commits by coherent unit (event capture, pure ledger/reconstruction, episode/replay binding, tests). Do not update generated production/campaign artifacts as part of this tranche.

## Acceptance tests and mutation matrix

Pure-contract tests must verify deterministic aggregation and digests for hand-constructed event sequences; score value derives from validated event rows/rules; same input produces byte-stable canonical output; missing required context or coverage is incomplete; and unrecognized scoring rules fail closed. Integration tests must execute the real simulator score path and compare emitted events, terminal team counters and observed production reward while toggling capture.

Required mutations must be rejected:

1. Delete one event while leaving replay counters unchanged.
2. Duplicate an event ID or duplicate a piece award; reorder same-tick or cross-tick events.
3. Change tick, clock, alliance, robot/team attribution, piece identity, score units, scenario seed, or source/rules digest.
4. Alter scorer rules/version and replay only a prefix under the prior digest.
5. Change replay events but leave its score ledger/evaluation record unchanged; change the evaluation summary alone; or make summary and replay disagree while recomputing only the outer file hash.
6. Substitute `fuel_score_count`, `fuel_scored_event` aggregate, `useful_throughput`, or reward total for event-derived score; assert rejection and assert these values remain separately named in valid output.
7. Emit an event from an inactive world or outside a successful simulator score mutation; assert rejection or incomplete capture.
8. Attempt to populate `official_match_points` or claim qualification eligibility from this simulator-only ledger; assert rejection.
9. Compare event capture on/off for identical seeded inputs and prove score counters and rewards are unchanged. No test may change raw scoring, reward, cadence, random-number consumption, or dynamics.

Existing regression coverage to inspect and extend includes `tests/test_rebuilt_match_scoring.py`, `tests/test_meco_evaluation_dashboard_replay.py`, and score-action coverage in `tests/test_tensor_3v3_physical_constraints.py`. Passing pure schema tests alone does not establish simulator behavior.

## Runtime/replay and dashboard gates

Every test run that checks score-event production or simulator scoring must be accompanied by a representative end-to-end simulator evaluation, with scenario and reproducible seed recorded. The evaluation must exercise scoring on both enabled and disabled HUB periods if the selected scenario permits, inspect the stored record and replay, and prove that replay-only reconstruction equals the record-derived ledger and terminal score counters. Retain scenario ID, seed, simulator/source identity, replay ID/path, event and ledger digests, and result in a reviewable report. An unrelated or historical replay is insufficient.

Per repository `AGENTS.md`, before generating or reviewing that replay, restart the dashboard services serving the current checkout, verify the relevant API endpoints respond successfully, verify the current backend is running, and compare served frontend JS/CSS hashes with the checked-out assets. Generate a fresh replay only after that currentness check. Record service identity/restart, endpoint result, asset hashes, replay ID, scenario, seed and inspection result. If a required device is unavailable because continuous r4 training still holds the WX9100s, do not stop or alter that run, and do not launch a competing simulator; implementation/review remains code-only pending the user's manual stop and actual device release. CPU fallback and the P600 are prohibited for later eligible simulator evidence by the parent plan.

This plan itself must not run tests, launch a simulator, change device leases, restart services, or touch active training data. The implementer may run code-only/pure tests during implementation, but all scoring-behavior test claims remain provisional until the seeded simulator/replay and dashboard gates pass.

## Independent three-pass workflow and merge gate

1. **Planner:** this exact plan is reviewed and accepted at its pushed commit. Any change to score semantics, score field naming, source paths, or the explicit FRC-vs-simulator boundary requires a plan revision and review before code changes.
2. **Implementer:** works in an isolated worktree from the approved plan head, commits and pushes coherent units incrementally, and reports tests, unrun runtime gates, source tree and candidate head.
3. **Independent reviewer:** reviews the exact pushed candidate head; verifies source mutation point, schema strictness, no reward/scoring mutation, all required mutation cases, replay-only reconstruction and digests. Approval must distinguish code acceptance from deferred runtime evidence; no runtime/outcome claim is approved until the simulator/replay and dashboard gates pass.

Merge only the exact reviewed candidate. Then update the three-pass dashboard provenance and restart/check the current dashboard before presenting dashboard evidence. The next Phase 0 tranche (clean freeze and four-way held-out partition) cannot treat this scorer as sufficient if the episode lacks complete event/replay coverage or if its explicitly simulated outcome does not match the approved estimand. No calibration collection, comparator selection, margin setting or qualification is authorized by this implementation tranche.
