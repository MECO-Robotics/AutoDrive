# MECO V4 P1.3/P1.4 source-semantics owner amendment proposal

**Date:** 2026-10-26  
**Status:** Recommended semantics adopted by the user/maintainer response “Approve all” (2026-10-26); exact-head static review and source-plan gates remain pending.
**Implementation base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.  
**Accepted P1 plan:** `40d151dec437fe405bd84d2918b2b86ae6f4a129`; file SHA-256 `1cfe7ccd56a4355501a7e46a3d3e3cd9bf0ae1c443bb0da571c27e05f6043300`; Pass 1 ACCEPT `2810a315970c56b0161bbfbae025a7a0301ece8e`, record SHA-256 `dedcca460b1cfc7d89c2f13ccd383783f674a5085f99052c4e30caf21f4ba339`.  
**Accepted P1.1 map:** `c902659d1b9079b3afcc2e4796780d97020b6726`, `docs/implementation-provenance/meco-v4-p1-score-source-map-20261026.md`, SHA-256 `c8c1d0594977ec456d8513293fe7ba0ca0020eba52a822e75cad34104aa36b4e`.  
**P0a authority:** candidate `56183cefd2402342ce28f836155164fc850ebc60`; maintainer decision `7007607010636fe7451ad1b59eb1a900e0a4d7b1`; exact-head Pass 3 ACCEPT `d5dab585ef1147f0d9997ece80db6e9a7ba95979`. Accepted P1.2 static event registry: `78b640630fde1aa9c0934463525e322ba57e027e`, file SHA-256 `5e4c803a676e7a445b686d0f73e5d18e7341533119571eae8e9545d7a259d278`; review `abc4a3d712bf829d0af5ec7a3ece9e173f7039c3`, record SHA-256 `47d1426ea8b02821a9521839d8c753a2e9e9a8d7505a42a1480fa615a1e7dff5`. The accepted row-block pack/unpack API is `8bd253042ee88d69afa7653886bf3d744e5c8d5e`, SOURCE/API ACCEPT review `3fe1fcd96b3cdab673f0e8847dafff44e484a12`; it does not authorize producer or ledger semantics.

## Recorded decision and boundary

The user/maintainer response “Approve all,” relayed by the orchestrator on 2026-10-26, adopts the concrete recommended choices below for the P1 contract. The already accepted Gamepiece/score owner response continues to govern episode-local sequence reset. This record preserves the fixed ABI and does not claim that the original delegated owner authored a new response. The proposed choices now have maintainer approval; exact-head static review and any required P1 allowlist/source-base amendment remain required before implementation. P1.3/P1.4 source edits remain stopped until those gates are met.

Accepted P0a/P1 event field names, types, widths, order, codes, source-site codes, identity, batch capacity, and estimand remain fixed. This proposal changes none. Any answer requiring a changed field, width, order, capacity, or code returns to ABI reconciliation and requires a new P0a decision/profile hash.

## Source anchors

All observations refer to the pinned implementation base above and the accepted source map. Hashes identify inspected source files.

| Concern | Anchor and fact | SHA-256 |
|---|---|---|
| Score and respawn | `frc_defense/gamepiece_actions.py::update_fuel_actions`: accepted physical piece is selected in `chosen`/`newly_scored`; score bookkeeping precedes respawn/launch mutation; ownership, zone, and possible-cell state then change (map lines 329–453). | `74c47eb546565f46712bd5328c50c71b07275b897cb1ef3fc9216b67d4ad260c` |
| Clock and dispatch | `_tensor_3v3_gamepieces.py::_update_match` advances `match_elapsed` by `dt` (133–135); `_update_fuel` dispatches scoring. | `480042250dcf281142929c62abf8bac60c007c498fb6105d517c55d7f645746a` |
| Step and reset | `tensor_3v3.py::step` increments `steps`, calls `_update_match`, then `_update_fuel` (1340–1342). Full reset clears `match_elapsed` and `steps` (462–463); per-world reset covers both (521 onward). Neither piece-generation nor event-sequence state exists at this base. | `e4925bfc100a487fbf734d0620cd7ab3b66ffc4e9027edb1cde78cbaca2f982e` |
| Diagnostic consumer | `meco_metric_ledger.py` maps `fuel_scored_event` as a focal-robot diagnostic, not the P0a event-ID ledger. | `aa2a440c056a8a54978196aa4d0c6bf809ff807d78ae0d618a33ba8b26e3fb49` |
| Evaluation consumer | `tensor_training_evaluation.py` aggregates `fuel_scored_event`; it cannot establish unique event IDs or the whole-alliance P0a source. | `72036f22011a03fdbcd5ac8f235608ee3a37716b18b8c449f1b289c2b0500483` |
| P5 seam | `meco_track_training.py`; accepted overall plan reserves trainer rollout, PPO/reward collection, and checkpoint handoff to P5. | `da3cb88c320aa8a2945b6ad0c9d9b4dbd34ef843778eaabde8b0f3660a99b07b` |
| Robot-team map | `frc_defense/_tensor_3v3_constants.py::TEAM_IDS` is `(0,0,0,1,1,1)` at the pinned base; score producer has a fixed six-slot robot-to-team mapping. | `fc8898550d31a08a05195cdebe6802afc4563e87019ddcd77e7eea716f9a410a` |
| Canonical P0a ledger contract | `frc_defense/meco_simulated_score_ledger.py` is absent from the implementation base. Accepted P0a source reference at main `d74b0a15e25ddbc6bacfbbb864674371629bb19e` rejects every duplicate event ID and rejects alliance values differing from `robot_team_slots[robot_slot]`; output totals are indexed as red 0, blue 1. That legacy implementation validates zero-based `tick` and its float32 elapsed-time recurrence. V4 adopts the approved one-based post-step sample below, so the new P1 ledger must validate V4 timing directly rather than copy the legacy clock validator. This is contract/source evidence only; do not copy the later-main implementation into the pinned base. | `71427d251f1639a4a113b610b5c8122fe88d1c2aac9c11eb4861b0f82bc05114` |

Accepted Gamepiece/score response `3738ad58fc883c5853b32c508e89ac657864bca4`, file SHA-256 `f57476d13d6038e9e0109fedc557c617ead25a99e4db49512b135a532c1e13af` (review `8e58451551c82c4713688e3fd228a9b410125e2155157f437e75679b2e3601ac`) fixes `episode_event_sequence` as episode-local, restarting only under a new `episode_generation`. That reset rule remains unchanged. The maintainer has now adopted the proposed same-tick allocation, piece-generation lifecycle, and event clock samples below.

## Adopted Gamepiece/score semantics

### Physical `piece_generation`

**Adopted rule:** keep a nonnegative `int64` generation per physical piece slot. Initialize each slot to zero at initial episode creation. Record the scored incarnation's current generation in its event before respawn mutation. Increment exactly once when that accepted score mutation respawns/reuses the slot, before its next incarnation can score. Ordinary motion/state updates do not increment it. On full or per-world reset, reset that world's generations to zero only as its new `episode_generation` is established; leave other worlds unchanged.

**Adopted exhaustion/fault rule:** never wrap or saturate. If a slot at `INT64_MAX` must increment for reuse, or event staging/sequence capacity fails, abort the staged score/event/respawn mutation before it commits. The fault boundary is limited to score/event/respawn state: physics, `steps`, and `match_elapsed` have already advanced before `_update_fuel` and are not rolled back. Mark the environment step invalid, raise a fail-closed error, emit no valid transition/reward/artifact for it, and do not retry that already-advanced step. The environment must be fully reset under a new episode generation before it can be used again. `INT64_MAX` remains valid for an incarnation that does not require reuse in that update.

### Episode-local sequence order and capacity

**Already fixed:** sequence resets only under a new `episode_generation`, per the accepted owner response above.

**Adopted order:** assign sequences only to accepted mutations, sorted chronologically within each episode by `(tick, rollout_row_index, robot_slot, piece_slot, source_site_code)`. The episode sequence counter persists across batch boundaries; batch-local rollout-row indices never restart the episode sequence. Events from one physics update share `tick`, then use rollout row, robot slot, piece slot, and source-site order. This avoids dependence on reduction order, Python loop order, or GPU scheduling. Sequence starts at zero for a new episode generation and resets only at that boundary.

**Adopted limit:** preserve the source-backed 32,768-event-per-episode limit: sequence values 0–32,767; no wrap or truncation. If the next accepted event exceeds it, use the fail-closed staged score/event/respawn fault behavior above. This episode limit is separate from the fixed 2,304-row batch table (`32 × 6 × 12`). Any change to field width or batch capacity requires ABI reconciliation.

### `tick` and `match_elapsed_seconds`

**Adopted sample:** set event `tick` to the simulator's one-based `steps` after the active physics step increments it. Set `match_elapsed_seconds` from the world clock after `_update_match` and before `_update_fuel` score mutation, preserving the exact simulator accumulator value and casting that value to the event field's `float64` representation; do not recompute elapsed time from tick and timestep. All events from one `_update_fuel` dispatch use that sample. Since source resets `steps` and `match_elapsed` with an episode reset, tick restarts at 1 for that episode; elapsed time uses the existing post-`dt`, capped match clock. Inactive worlds emit no event.

This adopts the one-based post-advance sample and existing 160-second match-clock cap. Terminal/truncated active steps use the same sample because they pass through the same step ordering; inactive worlds emit no event.

## Adopted Artifact/manifest decisions

1. Add `frc_defense/meco_simulated_score_ledger.py` to the finite P1 allowlist as the canonical P0a unique-event-ID aggregator at implementation base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`. The file is absent at that base; implement against accepted P0a rules and do not copy later-main code. Event-table and transition-envelope source authoring remains scoped to the existing P1 paths `frc_defense/meco_score_events_v4.py` and `frc_defense/meco_transition_v4.py`; this decision does not itself accept their full runtime or persistence APIs. P1 publishes typed source/API references for P5 and does not add or edit trainer/checkpoint/manifest paths owned by P5.
2. Reconstruct accepted event identity as `(track_code, world_slot, episode_generation, episode_event_sequence)`: `track_code` comes from the one-track-per-run profile, `world_slot` and sequence come from the fixed event row, and `episode_generation` comes from the referenced rollout row. Do not add fields to the event row.
3. Derive event-row `alliance` from the scored robot slot using the pinned `TEAM_IDS=(0,0,0,1,1,1)` map. The adopted mapping is team/alliance 0 = red and 1 = blue, matching P0a totals `[red, blue]`; validate every event alliance against its `robot_slot` and fail closed on unknown team ID or mismatch.
4. Reject every duplicate event ID within the episode aggregate, including an exact duplicate with identical payload, as required by the accepted P0a ledger contract; do not silently deduplicate. V4 has no `episode_id` field: run ledger validation within one episode scope, take `episode_generation` from its referenced rollout row, and reject a second award for the same `(world_slot, piece_slot, piece_generation)` incarnation within that scope, even if its event ID differs. For each `(world_slot, piece_slot)` in that episode scope, the first observed awarded incarnation must have generation 0 and every subsequent awarded incarnation must have exactly the next generation; gaps, repeats, and regressions fail closed. This validates the scorer's generation lifecycle from accepted award rows without claiming to observe unscored respawns. Reject invalid identity, unknown event/source-site code, or malformed accepted row. Sum `score_units` only from unique valid accepted events by validated alliance into red/blue whole-alliance totals. Each accepted mutation remains one unit under the adopted owner contract. This implements P0a's accepted-awards-only unique-ID estimand; per-robot reward/metric aggregates are not substitutes.
5. Bind event rows to transition rows using the fixed rollout-row index and obtain episode generation from that row. `meco_transition_v4.py` remains the P1 artifact/envelope API path; P5 owns trainer checkpoint/manifest integration. No separate manifest file path is added by this amendment.

The source-path addition requires a reviewed P1 plan/allowlist successor before any affected code is authored. Static fixtures must cover duplicate event IDs (including identical payload), duplicate piece-incarnation awards with different IDs, generation gaps/regressions, valid sequential generations across batches, invalid team/alliance values, and the approved V4 one-based clock samples. The accepted P1 plan still requires its reviewed contract-anchor bridge before validation/integration. P5 alone owns trainer integration: P1 provides a typed API pin and does not edit `meco_track_training.py`.

## Related payload-order decision

The same “Approve all” response selects Option A in the separately reviewed payload-order proposal `32edc197cc53812cdff1b74190629db1a2acdc72`, review record `e14d658e0d46a6f50370a621a63994e5d240df8e`. Its proposed byte ranges are identity prefix `[0,123)`, transition block `[123,314333115)`, and score-event table `[314333115,314469051)`. This is a separate contract choice and does not alter event-field meanings. A successor immutable proposal/profile digest and exact review must pin the selected layout before full-bundle codec authoring; the prior digest is not represented here as a finalized Option A profile digest.

## Required source-plan follow-up and current stop

No event field/type/width/order/code/capacity changes are made. The approved V4 clock meaning is one-based and post-advance, whereas the legacy P0a source validator used a zero-based tick; the new V4 ledger must validate the approved V4 sample and source-bound artifact/profile provenance must identify that contract. The fixed semantics require a reviewed successor P1 allowlist that adds `frc_defense/meco_simulated_score_ledger.py`; until that exact allowlist is accepted, P1.3/P1.4 source edits remain stopped. The accepted P1 plan still requires its reviewed contract-anchor bridge before validation/integration.

P1.1 remains source evidence only. P1.2 static registries and the accepted 32-row transition-block codec do not supply event production or a complete bundle/ledger API. **P1.3/P1.4 source edits remain STOPPED** pending independent exact-head review and acceptance of the successor allowlist. P2 remains behind the complete P1 API gate. This record is static planning only: no source/test edits, tests/builds, simulator, GPU/device, dashboard, campaign/lease, runtime, integration, or training authority is granted.
