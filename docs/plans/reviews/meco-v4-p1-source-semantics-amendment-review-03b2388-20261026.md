# Independent Pass 1 review: P1.3/P1.4 source-semantics amendment

- **Verdict:** REVISE
- **Reviewed commit:** `03b2388eb38112e36a89706502d35becc82b6c30`
- **Reviewed file:** `docs/plans/meco-v4-p1-source-semantics-owner-amendment-20261026.md`
- **File SHA-256:** `feba601b5d2c74bcc7f1117b1271bd90ac3d18a78d4232b50b0dcc66c59fc39f`
- **Review mode:** exact-head, static contract/source inspection.

## Supported claims and scope

The implementation base, accepted P1 plan, corrected source map, P0a decision pins, event registry, row-block API, and source-anchor file hashes match the cited artifacts. The proposal clearly labels its semantic choices unadopted and states that it does not authorize implementation. It preserves accepted P0a/P1 fields, widths, codes, capacity, identity, and estimand; it separately treats episode-local sequence reset under a new `episode_generation` as already fixed, while leaving same-tick ordering, physical-generation lifecycle, time sampling, and canonical ledger source unresolved. The source/consumer boundary and P5 ownership of `meco_track_training.py` are explicit. P1.3/P1.4 remain stopped pending the required owner/maintainer disposition and any exact-base/allowlist amendment. The user's subsequent “Approve all” does not remove the need to clarify the transaction boundary or record the disposition at exact head.

## Required correction

The proposed exhaustion behavior says to “fail the entire staged simulator update” before score counters, event rows, respawn, or “other transition state” is committed. At the pinned source base, `tensor_3v3.step` advances physics and `steps`, then `_update_match` advances the match clock, before it dispatches `_update_fuel` for score mutation. The current language can therefore be read as claiming whole-step failure atomicity or rollback of physics/clock state that has already been applied.

Bound the proposal explicitly to the staged score/event/respawn mutation commit and state that it does not roll back physics, `steps`, or match-clock changes already applied for that step. Specify the simulator's fail/stop behavior after such a staging failure. Do not describe the entire simulator update as atomic unless the source architecture and contract actually provide that rollback.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
