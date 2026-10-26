# Independent Pass 1 review: P1 source semantics amendment

- **Verdict:** REVISE
- **Reviewed commit:** `ade229a27a263bbdc56197fe4dd1c053f666e610`
- **Reviewed file:** `docs/plans/meco-v4-p1-source-semantics-owner-amendment-20261026.md`
- **File SHA-256:** `0048ed4d7fd0e8d0d03600441568fae1b6a2c9c03aac3c1d5adcbf0c683b7fef`
- **Review mode:** exact-head static contract inspection.

## Confirmed corrections

The amendment preserves the accepted event ABI and P0a estimand, retains tick-first chronological sequence allocation and persistence across batches, and correctly distinguishes score/event/respawn staging failure from physics and clocks that already advanced. It removes the nonexistent V4 `episode_id` field, states that event identity is reconstructed from `track_code`, `world_slot`, `episode_generation` joined from the referenced transition row, and `episode_event_sequence`, and keeps P1.3/P1.4 stopped pending a reviewed allowlist.

## Required correction

The replacement phrase “within one episode scope” is not yet an exact grouping contract for duplicate-incarnation and next-generation checks. The following rules are keyed by `(world_slot, piece_slot[, piece_generation])` *within that scope*, but the amendment does not define how the scope is selected or validated when rows from multiple generations occur in one batch/aggregate. The V4 identity already supplies the grouping fields: define the checks per `(track_code, world_slot, episode_generation, piece_slot)` (or explicitly define and enforce the ledger API as accepting exactly one such episode scope). Ensure batch-boundary checks preserve that same key. This is needed so a new `episode_generation` correctly resets the per-piece generation sequence without allowing events from distinct episodes to collide.

Until this is resolved and independently accepted, P1.3/P1.4 authoring remains stopped. No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
