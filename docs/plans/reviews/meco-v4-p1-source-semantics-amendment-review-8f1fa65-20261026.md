# Independent review: P1 V4 ledger semantics alignment with P0a

- **Verdict:** REVISE
- **Reviewed commit:** `8f1fa65aa816beb2c120bf0bfb9c9911c5fc1c45`
- **Reviewed file:** `docs/plans/meco-v4-p1-source-semantics-owner-amendment-20261026.md`
- **File SHA-256:** `c9445d9f4d942357a8a270f2bff366d82629ed42f09635d51e975cd7c1378a03`
- **Review mode:** exact-head static contract/source inspection.

## Supported corrections

The revision grounds alliance encoding in the pinned source map (`TEAM_IDS=(0,0,0,1,1,1)`, 0=red/1=blue), requires event alliance to match the scored robot slot, restores rejection of every duplicate event ID in line with P0a, and distinguishes the legacy zero-based clock validator from approved V4 one-based post-step semantics. It correctly keeps source/profile provenance and successor allowlist review ahead of implementation. Tick-first episode sequence, cross-batch persistence, piece generation, and staged failure behavior remain consistent with the prior accepted semantics. It also keeps the fixed ABI, P5 seam, Option A profile gate, and P1.3/P1.4 source stop.

## Required correction: accepted P0a piece-incarnation integrity

The pinned accepted P0a ledger source `d74b0a15e25ddbc6bacfbbb864674371629bb19e` (SHA-256 `71427d251f1639a4a113b610b5c8122fe88d1c2aac9c11eb4861b0f82bc05114`) enforces more than event-ID uniqueness and alliance matching. Its validator also rejects a second award for the same `(episode_id, world_slot, piece_slot, piece_generation)` and verifies that each slot's `piece_generation` is the next expected generation. The new V4 contract and allowlist do not require these guards. Distinct event IDs could therefore count the same physical piece incarnation twice, or accept a skipped/reordered generation, despite the one-award-per-incarnation P0a source semantics.

Add to the V4 ledger contract and test-authoring scope: reject duplicate accepted awards for the same `(track_code, world_slot, episode_generation, piece_slot, piece_generation)` incarnation; enforce generation starts at zero per episode/slot and advances exactly once after each accepted award/respawn; and fail closed on skipped, repeated, or out-of-order generations. Pin how a V4 ledger rebuilds this expectation from chronologically ordered event rows across batch boundaries. Keep the legacy zero-based clock recurrence excluded as already stated; only the accepted generation/award invariants carry forward.

Because this candidate changes the previously accepted duplicate-ID policy from exact-duplicate idempotence to strict rejection, the successor allowlist must cite this new exact semantics head and review rather than `6011fd...`/`d36c7b5...`. P1.3/P1.4 source authoring remains stopped until the corrected allowlist successor receives exact-head acceptance.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
