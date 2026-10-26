# Independent Pass 1 review: adopted P1.3/P1.4 source semantics

- **Verdict:** ACCEPT
- **Reviewed commit:** `0883445be428bc02b202178e3082426b056bbda6`
- **Reviewed file:** `docs/plans/meco-v4-p1-source-semantics-owner-amendment-20261026.md`
- **File SHA-256:** `6c6e52dee72f1a14858b6954ba2bd86bdbc9f30697f230bdb9593e8b25238035`
- **Review mode:** exact-head static contract/source inspection.

## Findings

This successor corrects the prior chronology finding. It sorts by `(tick, rollout_row_index, robot_slot, piece_slot, source_site_code)`, explicitly states that the episode sequence counter persists across batch boundaries, and resets it only with a new `episode_generation`. The accepted event registry's `rollout_row_index` is a batch-local index in the fixed 32-row table, so making monotonic within-episode tick the primary key preserves chronology when that index repeats in later batches. Within a tick, row, robot, piece, and source-site keys provide deterministic ordering.

The other adopted semantics remain internally consistent with the accepted pins: event fields/codes/capacity/estimand are held fixed; physical-piece generations are scoped to slot reuse and episode reset; tick and match clock use the pinned step/update ordering; and exhaustion/staging failure aborts staged score/event/respawn mutations without claiming rollback of physics, `steps`, or `match_elapsed` already advanced before `_update_fuel`. It specifies invalidation, no retry/emission, fail-closed error, and a required full reset before reuse.

The artifact/manifest implementation details remain gated rather than represented as complete: canonical ledger/API and identity/de-duplication/award aggregation still require exact source/allowlist work. The document keeps P1.3/P1.4 source edits stopped until the successor allowlist is accepted, preserves the reviewed contract-anchor bridge before validation/integration, and retains the complete-P1-API gate for P2. The selected Option A payload placement remains subject to an immutable successor profile/digest and exact review before full-bundle codec authoring.

This ACCEPT covers the static semantic amendment at this exact head. It does not accept the missing canonical ledger implementation, authorize P1.3/P1.4 source edits, or satisfy allowlist, contract-bridge, runtime, or integration gates.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
