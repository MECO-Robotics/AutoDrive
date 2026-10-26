# Independent Pass 1 review: P1.3/P1.4 allowlist successor

- **Verdict:** REVISE
- **Reviewed commit:** `78626a86ddf0832c09c035d662b70ed5753fb71f`
- **Reviewed file:** `docs/plans/meco-v4-p1-p3-p4-allowlist-successor-20261026.md`
- **File SHA-256:** `decb2061e0ef51c09427cfc23e1450bb24892da967f9c6219319fb486d3f9833`
- **Review mode:** exact-head static plan/contract inspection.

## Supported correction

The successor explicitly derives alliance from the scored robot's pinned `TEAM_IDS=(0,0,0,1,1,1)` map, validates the event alliance against its robot slot, and fails closed on unknown/mismatched values. It also distinguishes the legacy zero-based/float32 P0a clock validator from the approved V4 one-based post-step clock and requires V4 source/profile provenance. The finite allowlist, no-ABI-change scope, P1.3/P1.4 ownership boundary, Option A digest exclusion, G0/runtime/device restrictions, and P5 seam are otherwise bounded and clear.

## Required corrections

1. The plan pins semantics candidate `6011fd987dc51f1db39a7b1cc6571dce27d7bfdb` and its review, but says to reject every duplicate event ID. That pinned accepted semantics candidate instead makes identical-payload duplicate IDs idempotent. The newer correction `8f1fa65aa816beb2c120bf0bfb9c9911c5fc1c45` changes the rule back to strict rejection to align with P0a, but has not yet received ACCEPT. Re-pin a reviewed exact semantics head and review record before treating strict rejection as authorized.
2. The accepted P0a ledger additionally rejects a second award for the same `(episode, world_slot, piece_slot, piece_generation)` and checks each slot's generation against the next expected generation. This successor lacks both guards. Add duplicate-award-per-incarnation rejection and per-slot expected-generation validation across batch boundaries to ledger behavior and static fixture scope. Otherwise two different event IDs could count the same physical piece incarnation, or a skipped/reordered generation could pass.

Until both corrections receive exact-head acceptance, this amendment does not authorize P1.3/P1.4 source authoring.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
