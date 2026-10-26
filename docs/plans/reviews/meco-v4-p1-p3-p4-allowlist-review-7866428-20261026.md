# Independent Pass 1 review: P1.3/P1.4 allowlist successor

- **Verdict:** REVISE
- **Reviewed commit:** `786642862c0439c13fabe13b0b7f53a9572ec3c0`
- **Reviewed file:** `docs/plans/meco-v4-p1-p3-p4-allowlist-successor-20261026.md`
- **File SHA-256:** `ed1371c536817650045d518bc3a2c0463884d41c7529dcf68c384430e29d9dbe`
- **Review mode:** exact-head static plan/contract inspection.

## Supported scope and gates

The candidate pins the accepted P1 source base and relevant exact-head evidence, adds only the missing ledger module to a finite source/test allowlist, and keeps the event and transition schema paths limited to their stated role. The P0a/P1 fields, widths, row capacity, codes, tick-first order, sequence persistence, generation behavior, and failure boundary are preserved. It excludes Option A payload assembly/digest, full envelope and persistence APIs, trainer/checkpoint/manifest paths, P2/P5 integration, runtime, and execution. G0, two-WX9100, maintainer, and reviewed contract-anchor bridge gates remain explicit.

## Required correction: alliance-to-team binding

P0a's accepted preregistration requires the ledger to bind each event alliance to the robot-team map. The exact P0a text says the score ledger “validates that alliance against the robot-team map.” At the pinned implementation base, `frc_defense/_tensor_3v3_constants.py` defines `TEAM_IDS = (0, 0, 0, 1, 1, 1)` (file SHA-256 `fc8898550d31a08a05195cdebe6802afc4563e87019ddcd77e7eea716f9a410a`), while the accepted event registry only declares `alliance` as `uint8`; it does not declare the red/blue code mapping.

The current allowlist says the producer preserves event fields and the ledger sums by event-row alliance, but does not require the producer to derive that field from the scored robot's team ID or the ledger to validate it against the robot-team map. Add this binding and the exact source-derived mapping to the fixed contract/authorized ledger behavior, with mismatch and unknown alliance values failing closed. This is needed to implement P0a whole-alliance totals without silently crediting the wrong side.

No tests, imports, builds, simulator/replay, dashboard, GPU/device, campaign, lease, checkpoint, or artifact operations were performed.
