# Independent Pass 1 review: P1.3/P1.4 allowlist successor

- **Verdict:** REVISE
- **Reviewed commit:** `cb2e80097a74af8d6152d6a672247105c556d59d`
- **Reviewed file:** `docs/plans/meco-v4-p1-p3-p4-allowlist-successor-20261026.md`
- **File SHA-256:** `5290c6c9ddc0dd80971cd3ad24471547bb93b5fe17b623c483124bc7a744caab`
- **Review mode:** exact-head static plan/contract inspection.

## Scope and gates that are correctly bounded

This successor now pins the latest semantics candidate `ade229a...` and expressly prevents acceptance/source authorization until that exact semantics head receives ACCEPT. It preserves the fixed event ABI, event codes/capacity, P0a robot-team mapping, tick-first ordering and cross-batch sequence persistence; limits authoring to the finite listed paths; excludes full-bundle assembly/digests and the P5 trainer/manifest seam; and retains G0, two-WX9100 and no-CPU/P600 fallback restrictions. The authoring gate is explicit.

## Required correction

The pinned semantics candidate `ade229a...` is not accepted; its independent review is REVISE. Both that semantics candidate and this allowlist describe duplicate-incarnation and sequential-generation checks as applying “within one episode scope,” with the actual checks keyed only by `(world_slot, piece_slot[, piece_generation])`. Define and enforce the scope with accepted V4 identity fields, e.g. group per `(track_code, world_slot, episode_generation, piece_slot)`, or explicitly constrain the ledger API to one such scope and validate every input row belongs to it. This is required to reset piece-generation checks at a new `episode_generation` and keep cross-batch checks from conflating separate episodes. Pin a corrected, independently accepted semantics head before this allowlist can authorize source authoring.

Until those corrections receive exact-head ACCEPT, P1.3/P1.4 source edits remain stopped. No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
