# Independent review: P1 V4 ledger semantics alignment with P0a

- **Verdict:** REVISE
- **Reviewed commit:** `06e3d0bc4cb8e169c5c419afc9647e809915bdf1`
- **Reviewed file:** `docs/plans/meco-v4-p1-source-semantics-owner-amendment-20261026.md`
- **File SHA-256:** `e2134319cb0e6fa8c75d257c64a3b6909abe43375a8767d760bf96347aee0879`
- **Review mode:** exact-head static contract/source inspection.

## Supported additions

The successor adds the accepted P0a guard against duplicate awards for the same piece incarnation, sequential awarded-generation validation, and corresponding fixture requirements. It retains strict duplicate-event-ID rejection, exact team-map alliance binding, the V4 one-based clock distinction from the legacy validator, tick-first batch chronology, and runtime/source-plan stops.

## Required correction: use V4 episode identity

The newly added incarnation and generation-validation keys still use `episode_id`:

- duplicate incarnation key: `(episode_id, world_slot, piece_slot, piece_generation)`;
- per-slot generation state: `(episode_id, world_slot, piece_slot)`.

V4 has no `episode_id` field. Its approved event identity uses `track_code`, `world_slot`, `episode_generation` joined from the referenced transition row, and `episode_event_sequence`. Replace `episode_id` in both keys with the V4 episode identity `(track_code, world_slot, episode_generation)`, yielding the duplicate-incarnation key `(track_code, world_slot, episode_generation, piece_slot, piece_generation)` and generation-state key `(track_code, world_slot, episode_generation, piece_slot)`. Alternatively, define a deterministic `episode_id` derivation solely from those accepted V4 identity components and pin it. The current text leaves the new P0a guard unimplementable from the allowed event/transition data as written.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
