# Independent Pass 1 review: P1.3/P1.4 allowlist successor

- **Verdict:** REVISE
- **Reviewed commit:** `c282d54dc6c19f4ef7a339cce0a5f1e8783ec4db`
- **Reviewed file:** `docs/plans/meco-v4-p1-p3-p4-allowlist-successor-20261026.md`
- **File SHA-256:** `87267bee4602b3027718059b308bcf3fdaa9b5195136affa89f7c4170da49dc4`
- **Review mode:** exact-head static plan/contract inspection.

## Scope and gates that are correctly bounded

The candidate preserves the fixed event ABI, event codes, batch capacity, P0a alliance mapping, tick-first ordering and sequence persistence; limits source authoring to the enumerated paths; excludes full-bundle assembly/digests and P5 trainer/manifest work; and retains the G0, two-WX9100, no-CPU/P600 fallback and runtime restrictions. It explicitly says implementation is not authorized before exact-head acceptance and that the semantics candidate `06e3d0b...` must itself receive ACCEPT. That is a valid stop boundary, but the pinned semantics candidate currently has a separate REVISE finding, so this candidate cannot open source work.

## Required correction

The ledger rules still key duplicate piece-incarnation awards and sequential generation checks by `(episode_id, world_slot, piece_slot[, piece_generation])`. The pinned V4 event/transition contract has no `episode_id` field or defined `episode_id` derivation; event identity is `(track_code, world_slot, episode_generation, episode_event_sequence)`, with `episode_generation` joined from the referenced transition row. The pinned semantics candidate `06e3d0b...` repeats this unresolved `episode_id` wording and is not accepted. Before this allowlist can pass, define the ledger grouping key using accepted V4 identity fields (or provide an approved derivation), update the semantics pin to an exact-head ACCEPT, and ensure duplicate-incarnation and next-generation checks use that same defined scope across batch boundaries.

Until then, P1.3/P1.4 authoring remains stopped. No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
