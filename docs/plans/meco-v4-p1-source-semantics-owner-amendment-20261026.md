# MECO V4 P1.3/P1.4 source-semantics owner amendment proposal

**Date:** 2026-10-26  
**Status:** Proposal only. Every choice below is **unadopted**.  
**Implementation base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.  
**Accepted P1 plan:** `40d151dec437fe405bd84d2918b2b86ae6f4a129`; file SHA-256 `1cfe7ccd56a4355501a7e46a3d3e3cd9bf0ae1c443bb0da571c27e05f6043300`; Pass 1 ACCEPT `2810a315970c56b0161bbfbae025a7a0301ece8e`, record SHA-256 `dedcca460b1cfc7d89c2f13ccd383783f674a5085f99052c4e30caf21f4ba339`.  
**Accepted P1.1 map:** `c902659d1b9079b3afcc2e4796780d97020b6726`, `docs/implementation-provenance/meco-v4-p1-score-source-map-20261026.md`, SHA-256 `c8c1d0594977ec456d8513293fe7ba0ca0020eba52a822e75cad34104aa36b4e`.  
**P0a authority:** candidate `56183cefd2402342ce28f836155164fc850ebc60`; maintainer decision `7007607010636fe7451ad1b59eb1a900e0a4d7b1`; exact-head Pass 3 ACCEPT `d5dab585ef1147f0d9997ece80db6e9a7ba95979`. Accepted P1.2 static event registry: `78b640630fde1aa9c0934463525e322ba57e027e`, file SHA-256 `5e4c803a676e7a445b686d0f73e5d18e7341533119571eae8e9545d7a259d278`; review `abc4a3d712bf829d0af5ec7a3ece9e173f7039c3`, record SHA-256 `47d1426ea8b02821a9521839d8c753a2e9e9a8d7505a42a1480fa615a1e7dff5`. The accepted row-block pack/unpack API is `8bd253042ee88d69afa7653886bf3d744e5c8d5e`, SOURCE/API ACCEPT review `3fe1fcd96b3cdab673f0e8847dafff44e484a12`; it does not authorize producer or ledger semantics.

## Decision requested

This document asks the delegated Gamepiece/score owner, Artifact/manifest owner, and repository maintainer to accept, reject, or revise the choices below. It adopts nothing. P1.3/P1.4 source implementation must remain stopped until owner responses are recorded, independently reviewed at exact head, and any required P1 allowlist/source-base amendment is accepted.

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
| Canonical ledger | `frc_defense/meco_simulated_score_ledger.py` is absent from the pinned base. Later-main source is not authority to copy into this base. | absent at pinned base |

Accepted Gamepiece/score response `3738ad58fc883c5853b32c508e89ac657864bca4`, file SHA-256 `f57476d13d6038e9e0109fedc557c617ead25a99e4db49512b135a532c1e13af` (review `8e58451551c82c4713688e3fd228a9b410125e2155157f437e75679b2e3601ac`) already fixes `episode_event_sequence` as episode-local, restarting only under a new `episode_generation`. That reset rule is settled and is not reopened here. Same-tick allocation order, piece-generation lifecycle, and event clock samples remain unresolved.

## Proposed Gamepiece/score owner decisions — unadopted

### Physical `piece_generation`

**Proposed rule:** keep a nonnegative `int64` generation per physical piece slot. Initialize each slot to zero at initial episode creation. Record the scored incarnation's current generation in its event before respawn mutation. Increment exactly once when that accepted score mutation respawns/reuses the slot, before its next incarnation can score. Ordinary motion/state updates do not increment it. On full or per-world reset, reset that world's generations to zero only as its new `episode_generation` is established; leave other worlds unchanged.

**Proposed exhaustion rule:** never wrap or saturate. If a slot at `INT64_MAX` must increment for reuse, fail the entire staged simulator update before score counters, event rows, respawn, or other transition state is committed. `INT64_MAX` remains valid for an incarnation that does not require reuse in that update.

**Owner must confirm:** what constitutes a new incarnation, whether every respawn/replacement follows this rule, initialization/reset timing, and failure atomicity. Revise with exact per-path behavior if needed.

### Episode-local sequence order and capacity

**Already fixed:** sequence resets only under a new `episode_generation`, per the accepted owner response above.

**Proposed order:** assign sequences only to accepted mutations, sorted within each episode by `(rollout_row_index, tick, robot_slot, piece_slot, source_site_code)`. Events from one physics update share `tick`, then use robot slot, piece slot, and source-site order. This avoids dependence on reduction order, Python loop order, or GPU scheduling. Sequence starts at zero for a new episode generation.

**Proposed limit:** preserve the source-backed 32,768-event-per-episode limit: sequence values 0–32,767; no wrap or truncation. If the next accepted event exceeds it, fail the whole staged update before committing any state/event. This episode limit is separate from the fixed 2,304-row batch table (`32 × 6 × 12`). The owner must explicitly confirm the existing recorder limit is normative for V4; the `uint16` storage type alone does not settle whether V4 may use values through 65,535.

**Owner must confirm:** total ordering, episode cap, and failure behavior. Any change to field width or batch capacity requires ABI reconciliation.

### `tick` and `match_elapsed_seconds`

**Proposed sample:** set event `tick` to the simulator's one-based `steps` after the active physics step increments it. Set `match_elapsed_seconds` from the world clock after `_update_match` and before `_update_fuel` score mutation. All events from one `_update_fuel` dispatch use that sample. Since source resets `steps` and `match_elapsed` with an episode reset, tick restarts at 1 for that episode; elapsed time uses the existing post-`dt`, capped match clock. Inactive worlds emit no event.

**Owner must confirm or replace:** zero/one-based convention, sample point, and terminal/truncated-step behavior, including the existing 160-second cap. Source line order is evidence, not semantic authority.

## Proposed Artifact/manifest owner decisions — unadopted

1. Identify the canonical P0a unique-event-ID aggregator at the approved base: either provide an exact reviewed source/base pin containing it, or request an explicit P1 allowlist amendment adding `frc_defense/meco_simulated_score_ledger.py`. Do not copy later-main code without approval.
2. Confirm how identity `(track_code, world_slot, episode_generation, episode_event_sequence)` is reconstructed, with `episode_generation` joined from the referenced rollout row. Define duplicate handling (same ID/same payload and same ID/conflicting payload) and how `score_units` feed red/blue whole-alliance totals. The result must implement P0a's accepted-awards-only unique-ID estimand, not focal-robot metrics/reward aggregates.
3. Pin the canonical ledger/API and required manifest/artifact paths, and state whether the missing ledger path or any other path requires a P1 plan amendment. Keep full-envelope ordering/offsets under their separate contract decision; this proposal assumes no payload layout.

Any source-path addition or implementation-base change requires a reviewed P1 plan/allowlist amendment before affected code is authored. P5 alone owns trainer integration: P1 provides a typed API pin and does not edit `meco_track_training.py`.

## Required maintainer response and current stop

The maintainer must record disposition of both owner responses, whether P0a/ABI reconciliation is triggered, and the exact P1 base/finite allowlist for P1.3/P1.4. If a canonical ledger path or base change is needed, approve a successor plan/allowlist and exact-head review before implementation. The accepted P1 plan still requires its reviewed contract-anchor bridge before validation/integration.

P1.1 is accepted as source evidence only. P1.2 static registries and the accepted 32-row transition-block codec do not supply event semantics or a complete bundle/ledger API. **P1.3 producer and P1.4 canonical consumer/artifact source edits remain STOPPED** pending owner responses, independent exact-head review, and any required source-base/allowlist acceptance. P2 remains behind the complete P1 API gate. This document is planning only: no source/test edits, tests/builds, simulator, GPU/device, dashboard, campaign/lease, runtime, integration, or training authority is granted.
