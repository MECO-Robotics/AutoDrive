# Independent Pass 1 review: adopted P1.3/P1.4 source semantics and ledger contract

- **Verdict:** ACCEPT
- **Reviewed commit:** `6011fd987dc51f1db39a7b1cc6571dce27d7bfdb`
- **Reviewed file:** `docs/plans/meco-v4-p1-source-semantics-owner-amendment-20261026.md`
- **File SHA-256:** `0e0e3040fb6a5f530c4a6b6e2016158cd7ad63034f18c8f8376a222f66ca3681`
- **Review mode:** exact-head static contract/source inspection.

## Findings

This amendment closes the prior ledger-decision gap by explicitly adding `frc_defense/meco_simulated_score_ledger.py` to the required successor P1 allowlist at the approved base, while forbidding copying the later-main implementation. It concretely defines event identity reconstruction from `track_code`, event-row `world_slot` and sequence, and `episode_generation` joined from the referenced rollout row, without expanding the fixed event ABI. It defines identical-ID/same-payload duplicates as idempotent, conflicts and malformed accepted rows as fail-closed for the batch, filters accepted fuel-award event type/source codes, fixes one unit per accepted mutation, and sums unique score units by alliance for whole-alliance totals under the P0a estimand. It binds event rows to transition rows with the fixed rollout-row index and keeps trainer/checkpoint/manifest integration with P5.

The previous chronology and failure-boundary corrections remain intact: tick precedes the repeating 0–31 batch-local row index in sequence allocation, and the episode counter persists across batches until a new generation. Score/event/respawn faults do not claim rollback of physics, steps, or match clock already advanced before `_update_fuel`; failure invalidates the step, prevents retry/output, and requires full reset before reuse.

The amendment preserves accepted fields, widths, codes, batch capacity, and estimand. It keeps P1.3/P1.4 source changes stopped pending independent review/acceptance of the successor allowlist, and retains the contract-anchor bridge before validation/integration. Option A payload order remains subject to a new immutable profile/proposal digest and exact review before full-bundle codec work. P2 remains gated on a complete accepted P1 API.

The existing simulator base uses the pinned two-team map `TEAM_IDS=(0,0,0,1,1,1)`; the successor binds aggregation to the event-row alliance and that source-defined map rather than changing the ABI. The successor allowlist/API implementation must preserve that association when materializing event alliance values.

This ACCEPT covers the static semantics amendment only. It does not authorize source edits, tests, builds, runtime, device use, integration, merge, training, or full-bundle work before the stated gates.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
