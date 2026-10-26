# Independent pre-G0 SOURCE/API review: P1.1 score-event source map

- **Verdict:** SOURCE/API ACCEPT
- **Reviewed commit:** `920e9b9dd500e655834c7724a1e17c7e02b81bd9`
- **Reviewed file:** `docs/implementation-provenance/meco-v4-p1-score-source-map-20261026.md`
- **Git blob:** `4fcfb8bfc554f78a56f75e5c4eed975bbede4532`
- **File SHA-256:** `51ba48fed95dc96051a3df560b732d277d3d35854a830c8ecb62ce59904de138`
- **Pinned source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted P1 package plan:** `40d151dec437fe405bd84d2918b2b86ae6f4a129`, file SHA-256 `1cfe7ccd56a4355501a7e46a3d3e3cd9bf0ae1c443bb0da571c27e05f6043300`; Pass 1 ACCEPT review `2810a315970c56b0161bbfbae025a7a0301ece8e`, record SHA-256 `dedcca460b1cfc7d89c2f13ccd383783f674a5085f99052c4e30caf21f4ba339`
- **Approved P0a:** candidate `56183cefd2402342ce28f836155164fc850ebc60`; maintainer decision record `7007607010636fe7451ad1b59eb1a900e0a4d7b1`; exact-head Pass 3 ACCEPT record `d5dab585ef1147f0d9997ece80db6e9a7ba95979`
- **Review mode:** static source and Git-object inspection only; no project code was imported or executed.

## Evidence and contract checks

The candidate commit adds exactly one documentation file and is based directly on the approved source base. It introduces no runtime/test/build path or dependency. The six cited existing source-file hashes match the files at the pinned base: `gamepiece_actions.py`, `_tensor_3v3_gamepieces.py`, `tensor_3v3.py`, `meco_metric_ledger.py`, `tensor_training_evaluation.py`, and `meco_track_training.py`. `meco_simulated_score_ledger.py` is absent at the base and present only at the later checkout cited in the map; the candidate correctly refuses to import that later implementation into the approved source tree.

The source-site mapping matches the base: both vectorized and per-robot accepted scoring mutations select physical pieces in `update_fuel_actions`; downstream respawn handling acts on the selected mask; the environment increments `steps` and `match_elapsed` before `_update_fuel`; full and per-world reset paths clear aggregate event/score state; and step rewards, info, metrics, and evaluation reports consume aggregates rather than unique event IDs. The map does not conflate these diagnostics with P0a's whole-alliance accepted-award estimand. It preserves P5 ownership of `meco_track_training.py` and trainer integration.

The four unresolved semantics are correctly treated as blockers: piece-generation lifecycle, event-sequence allocation/order/reset, tick/elapsed-time sampling, and the canonical P0a event-ID ledger consumer. The source map makes no agent-selected semantics normative. It expressly stops P1.3 producer work and P1.4 artifact binding, proposes a reviewed owner/ABI decision for unresolved items, and permits P1.2 only for the already fixed ABI bytes without defining unresolved event semantics. This matches the accepted P1 plan's P1.1 stop rule and preserves the P0a score contract.

## Acceptance scope and limits

This SOURCE/API ACCEPT pins the exact P1.1 evidence map as complete for static source mapping and safe blocker identification. It does **not** close semantic questions, accept a runtime event API, validate scoring behavior, or authorize P1.3/P1.4 producer/consumer edits. Those remain blocked pending the separately reviewed owner/ABI/source decisions specified by the accepted P1 plan. Any source/base/hash change requires a new exact-head review.

No tests, imports, builds, score calculations, simulator/replay, dashboard, calibration, GPU/device, training, integration, campaign, or lease operation was performed. No claim of runtime correctness, GPU residency, qualification, or campaign status is made.
