# Independent Pass 1 review: adopted P1.3/P1.4 source semantics

- **Verdict:** REVISE
- **Reviewed commit:** `5011d7b6e19c965f6cdf866ac80be7efd321d82b`
- **Reviewed file:** `docs/plans/meco-v4-p1-source-semantics-owner-amendment-20261026.md`
- **File SHA-256:** `649f26e6f54333e0ba4328748a075492db2a37b1b676f0367eca1073d3797e42`
- **Review mode:** exact-head static contract/source inspection.

## Supported corrections

The successor records the user/maintainer “Approve all” response while preserving the fixed P0a/P1 ABI and explicitly keeps exact-head review and source-plan gates in force. It correctly adopts the previously proposed physical-generation and clock sampling rules. It also resolves the prior failure-boundary finding: `_update_fuel` runs after physics, `steps`, and `match_elapsed` have advanced; those effects are explicitly not rolled back. Failure aborts staged score/event/respawn state, invalidates the step, emits no valid transition/reward/artifact, does not retry, and requires a full reset under a new episode generation before reuse.

The artifact/manifest section remains appropriately gated: canonical identity de-duplication and blue/red award aggregation still need exact ledger/API and allowlist decisions. It keeps P1.3/P1.4 source work stopped pending successor allowlist review and keeps P2 behind the complete P1 API gate. The separately selected Option A payload layout remains subject to an immutable successor profile/digest and exact review before full-bundle codec work.

## Required correction: sequence chronology

The adopted ordering key is `(rollout_row_index, tick, robot_slot, piece_slot, source_site_code)`, with `rollout_row_index` sorted before `tick`. The accepted event registry defines `rollout_row_index` as a `uint8` index into a fixed 32-row batch (0–31), while the adopted tick is one-based and monotonic within an episode until reset. The episode-local event sequence persists across those rows and successive batches. Therefore the row index repeats in later batches and sorting it first can place a later tick before an earlier one; for example, row 0/tick 33 sorts before row 31/tick 32. That does not define chronological sequence allocation.

Revise the approved ordering so episode chronology is primary, for example `(tick, rollout_row_index, robot_slot, piece_slot, source_site_code)`, or provide another exact key that remains globally chronological across batches. Because the user approved the current recommendation, a changed ordering must be explicitly recorded as a successor decision before implementation. P1.3/P1.4 remain stopped pending that resolution and the accepted successor allowlist.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
