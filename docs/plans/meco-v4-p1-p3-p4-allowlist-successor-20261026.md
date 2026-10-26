# MECO V4 P1.3 producer and P1.4 ledger allowlist successor

**Date:** 2026-10-26  
**Status:** Pass 1 amendment candidate; no implementation is authorized until exact-head acceptance.  
**Planner:** `/root/p1_p2_planner`  
**Implementer:** `/root` (after acceptance)  
**Pre-G0 SOURCE/API reviewer:** `/root/p3_p4_planner`  
**Implementation base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.

## Purpose and inherited authority

This is a narrow successor to the accepted P1 source plan `40d151dec437fe405bd84d2918b2b86ae6f4a129` (plan SHA-256 `1cfe7ccd56a4355501a7e46a3d3e3cd9bf0ae1c443bb0da571c27e05f6043300`; Pass 1 ACCEPT `2810a315970c56b0161bbfbae025a7a0301ece8e`, record SHA-256 `dedcca460b1cfc7d89c2f13ccd383783f674a5085f99052c4e30caf21f4ba339`). The accepted P1 source base remains `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`; source work must use a fresh ZFS worktree at that base.

This successor adds the one absent canonical ledger path and bounds the next authoring slice to device-side event production plus the P0a score ledger. It does not replace accepted P0a decisions, change ABI fields/codes/capacity, authorize full payload assembly, or permit runtime work. Existing source/API and G0 conditions in the accepted P1 plan remain in force.

Prerequisites pinned by exact accepted evidence:

- Corrected P1.1 source map `c902659d1b9079b3afcc2e4796780d97020b6726`, file SHA-256 `c8c1d0594977ec456d8513293fe7ba0ca0020eba52a822e75cad34104aa36b4e`.
- Accepted fixed event registry `78b640630fde1aa9c0934463525e322ba57e027e`, file SHA-256 `5e4c803a676e7a445b686d0f73e5d18e7341533119571eae8e9545d7a259d278`; review `abc4a3d712bf829d0af5ec7a3ece9e173f7039c3`, record SHA-256 `47d1426ea8b02821a9521839d8c753a2e9e9a8d7505a42a1480fa615a1e7dff5`.
- Accepted static transition registry `38fdb1d0423f0e723c4e1fb533e36eb7b88c4715`, file SHA-256 `689b84acfd2d327c9ca4c26b081cf997acf511a0f14ed6f605a6caa91562f52f`; review `2cc1275a9cbf3949419cb4925cb1f2b1fb208f85`, record SHA-256 `3352d0c7a4963a928c8e5f4ab0bc9237647e57459963abf9a933f312f0c58b25`.
- Accepted 32-row transition-block pack/unpack API `8bd253042ee88d69afa7653886bf3d744e5c8d5e`; review `3fe1fcd96b3cdab673f0e8847dafff44e484a12`. It is only the standalone fixed transition block.
- Corrected source-semantics candidate `06e3d0bc4cb8e169c5c419afc9647e809915bdf1`, file SHA-256 `e2134319cb0e6fa8c75d257c64a3b6909abe43375a8767d760bf96347aee0879`; exact-head review is pending. This successor adds the accepted P0a incarnation uniqueness and next-generation guards and supersedes the earlier accepted `6011fd...` ledger details; this allowlist cannot pass until this candidate receives exact-head ACCEPT.
- Accepted readiness packet `a0082d5a909087303c666a66e271b80c586c08e2`; review `59e23fe9a204ee6cf55420fba53cdc4a9b3ce1ce`, record SHA-256 `0f0dcf158a688d27c3c1d916bea4b5eccebb72e5ffce633f59a7a169ce62e7d5`.

## Fixed contract; no ABI changes

The P0a contract remains whole-alliance accepted active-HUB FUEL awards by unique event ID, with red/blue totals; this is a limited simulated proxy, not official match points or a per-robot causal estimate. Preserve the adopted event row exactly: `valid bool`, `rollout_row_index uint8`, `episode_event_sequence uint16`, `tick int64`, `match_elapsed_seconds float64`, `world_slot int64`, `alliance uint8`, `robot_slot int64`, `piece_slot int64`, `piece_generation int64`, `event_type uint16`, `score_units uint16`, `source_site_code uint16`; 59 bytes per row; 2,304 rows per batch; 135,936 event-table bytes.

Preserve `event_type=1` (`fuel_score_awarded`), `score_units=1` per accepted mutation, and source-site codes 1 (vectorized score mutation) and 2 (robot-loop score mutation). All unused event rows are zero except `valid=0`. Sequence resets only under a new `episode_generation`; sequence allocation is chronological by `(tick, rollout_row_index, robot_slot, piece_slot, source_site_code)` and the episode counter persists across batch boundaries. `piece_generation` starts at zero for a new episode, increments exactly once when an accepted score mutation respawns/reuses that piece slot, and never wraps. Clock samples use one-based post-increment `steps` and `match_elapsed` after `_update_match`, before score mutation.

Derive event `alliance` from the scored robot slot via pinned `_tensor_3v3_constants.py::TEAM_IDS=(0,0,0,1,1,1)`: team/alliance 0 maps to red and 1 maps to blue. Validate every event alliance against that robot map and fail closed on unknown IDs or mismatch. The map file SHA-256 is `fc8898550d31a08a05195cdebe6802afc4563e87019ddcd77e7eea716f9a410a`.

For an `INT64_MAX` reuse increment, sequence-cap overflow, or malformed/conflicting event, abort the staged score/event/respawn mutation and fail the environment step. Physics, `steps`, and `match_elapsed` already advanced before `_update_fuel`; they are not rolled back. Emit no valid transition/reward/artifact, do not retry that advanced step, and require full environment reset under a new episode generation before reuse. This failure rule applies to staged score/event/respawn state, not to already advanced physics or clocks.

Canonical ledger behavior: event identity is `(track_code, world_slot, episode_generation, episode_event_sequence)`, with `track_code` from the fixed one-track profile and `episode_generation` joined from the referenced rollout row. Reject every duplicate ID, including exact duplicates with identical payload, as required by P0a; do not silently deduplicate. V4 has no `episode_id` field, so validate the ledger within a single episode scope; reject repeated award rows for `(world_slot, piece_slot, piece_generation)` within that scope even when IDs differ. For each `(world_slot, piece_slot)` in that scope, require the awarded generations to begin at 0 and advance by exactly one; reject gaps, repeats, and regressions. This checks the observed award lifecycle and does not assert visibility into unscored respawns. Derive alliance from the scored robot team, validate it against that robot slot, and use the explicit mapping 0=red, 1=blue; unknown/mismatched values fail closed. Count only valid rows with event code 1 and registered source-site codes 1 or 2, summing their `score_units` into red/blue totals. The accepted P0a ledger source reference `d74b0a15e25ddbc6bacfbbb864674371629bb19e` (file SHA-256 `71427d251f1639a4a113b610b5c8122fe88d1c2aac9c11eb4861b0f82bc05114`) verifies duplicate ID/incarnation rejection, next-generation validation, and team-map validation; it is absent at implementation base and must not be copied. Its legacy clock validator is zero-based, while the approved V4 contract is one-based post-step; implement V4 clock validation from the accepted source-semantics pin and bind that change in source/profile provenance. P1 publishes a typed API pin; P5 alone integrates trainer/checkpoint/manifests.

## Bounded source-authoring slice after Pass 1 ACCEPT

After this successor receives exact-head Pass 1 ACCEPT, the implementer may author only the paths and behaviors below, on a fresh worktree based on the exact source base above. This is static source authoring and review only; no test execution, imports, builds, simulator, dashboard, GPU/device, training, or integration is authorized by this plan.

| Path | Exact authorized purpose |
|---|---|
| `frc_defense/gamepiece_actions.py` | Narrow hunks at the two P1.1-mapped score mutation sites only: capture accepted physical-piece identity before respawn; stage event and generation mutation; preserve existing score, reward, RNG, physics, and branch behavior. |
| `frc_defense/_tensor_3v3_gamepieces.py` | Initialize/reset per-world `piece_generation`, episode sequence, event buffer/status, and fault lifecycle under the approved reset identity. |
| `frc_defense/tensor_3v3.py` | Allocate/reset approved event/generation state; expose typed events/status; latch a failed event step so the advanced world cannot be reused before full reset. No general environment or P2 residency changes. |
| `frc_defense/meco_score_events_v4.py` | Extend the accepted static registry with fixed-capacity event staging/ordering, row validation, failure state, and event-ID projection using the accepted semantics. Preserve fixed field names/types/codes and 2,304-row table. |
| `frc_defense/meco_simulated_score_ledger.py` | **New allowlisted path.** Canonical P0a aggregation from valid V4 rows in one episode scope: reject duplicate IDs and duplicate `(world_slot, piece_slot, piece_generation)` awards, require per-slot awarded generations to progress from zero without gaps/repeats/regressions, validate alliance against robot team, and compute red/blue `score_units` totals. Implement from the pinned base and accepted contract; do not copy later-main source. |
| `tests/test_meco_score_events_v4.py` | Static-authored fixtures for identity/generation/reset, tick-first order across batch boundaries, event capacity, duplicate IDs, duplicate incarnation awards with distinct IDs, generation gaps/regressions, valid sequential generations across batches, fail-closed staging, and alliance totals. Test execution is outside this plan. |

No other path is authorized by this successor. The existing `tests/test_meco_transition_v4.py` row-block fixtures remain pinned to accepted head `8bd253042ee88d69afa7653886bf3d744e5c8d5e`; this successor does not revise or execute them.

Explicitly excluded: full-bundle/Option A payload assembly or offsets; payload/profile digest changes; `frc_defense/meco_transition_v4.py` full envelope reader/writer/digest/persistence work; manifest/checkpoint writing; `frc_defense/meco_track_training.py`; `frc_defense/meco_metric_ledger.py`; `frc_defense/tensor_training_evaluation.py`; dashboards; build configuration; native HIP files; campaign/checkpoint/output data. The user-modified HIP file remains untouched. P5 alone owns trainer and checkpoint/manifest integration.

## Separate Option A dependency

The user/maintainer selected Option A in accepted proposal `32edc197cc53812cdff1b74190629db1a2acdc72`, review `e14d658e0d46a6f50370a621a63994e5d240df8e`: identity prefix `[0,123)`, transition block `[123,314333115)`, event table `[314333115,314469051)`. A successor immutable proposal/profile digest and exact review are still pending. This amendment therefore authorizes no full-bundle assembly, event-table payload offset, digest, envelope writer/reader, or artifact persistence API. Those paths need a separate reviewed profile/plan successor.

## Static acceptance evidence and review gate

The independent pre-G0 reviewer must inspect this exact head and verify:

1. Only the listed paths are authorized; the new ledger file is explicitly additive to the accepted finite allowlist.
2. All P0a/P1 field names, types, widths, order, capacity, and codes remain fixed.
3. The accepted semantics and fault boundary are represented exactly, including tick-first chronology and sequence persistence across batch boundaries.
4. P1.3 producer captures only accepted score mutations from the two mapped sites, with event identity taken from the scored physical incarnation before respawn and alliance derived from that robot’s accepted team slot.
5. The canonical ledger rejects every duplicate ID and duplicate piece-incarnation award, requires next sequential awarded generations per episode/world/piece slot, validates the fixed 0=red/1=blue robot-team mapping, and computes whole-alliance totals without depending on per-robot metrics/reward.
6. The P5 seam, Option A digest gate, G0 requirements, and runtime/device prohibitions are preserved.

SOURCE/API ACCEPT pins source authoring only within this allowlist; it does not accept test execution, runtime, P1 integration, full-bundle API, P2, P5, or training.

## Runtime, device, and integration gates

All execution and integration remain behind G0 in the accepted P1 plan: manual campaign stop, independently confirmed release of both assigned WX9100 leases, explicit maintainer authorization, and a reviewed contract-anchor bridge. Algorithmic score/event/digest work must use both WX9100 devices. CPU numerical work, CPU score/event/digest validation, P600 substitution, silent migration, and CPU fallback are prohibited. Retain per-process device identity, kernel placement, ownership, and transfer/synchronization evidence. If either WX9100 or a required HIP path is unavailable, fail closed. These are future gates; this planner performed no such operation.

Static-only artifact. No source, test, build, simulator, dashboard, GPU/device, campaign/lease, runtime, integration, or training operation was performed.
