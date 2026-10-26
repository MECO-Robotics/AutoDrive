# Independent Pass 1 review: P1 source package successor

- **Verdict:** ACCEPT (supersedes the REVISE for candidate `376f3913976839a1c6043915c7b7854d449c921b` only)
- **Review date:** 2026-10-26
- **Planner:** `/root/p1_p2_planner`
- **Reviewed commit:** `40d151dec437fe405bd84d2918b2b86ae6f4a129`
- **Reviewed file:** `docs/plans/meco-v4-p1-score-event-source-package-plan-20261026.md`
- **Git blob:** `cba82fb8a5e37141211bb0ce2f08e0b3aa87665e`
- **Plan SHA-256:** `1cfe7ccd56a4355501a7e46a3d3e3cd9bf0ae1c443bb0da571c27e05f6043300`
- **Previous review:** REVISE at `1f201f611920948f79234426dd425a8664aeab03`
- **Review base:** `d74b0a15e25ddbc6bacfbbb864674371629bb19e`
- **Scope:** Resolve the two prior findings and confirm no regressions to the exact source base, allowlist, P1.1 semantics gate, succession relationship, and P5 ownership boundary.

## Exact-head review

Both prior findings are resolved:

1. The post-G0 section now requires all G0 evidence: manual stop of campaign PID `2750253`, independent release confirmation for both WX9100 leases, and explicit user or repository-maintainer authorization for runtime/device/integration work. It additionally requires a docs-only contract-anchor bridge containing the exact accepted reconciliation bytes, verified profile/source hashes, and independent review of that exact bridge before any P1 validation or integration. This matches the accepted overall plan's G0 and bridge order.
2. The plan now requires all P1 algorithmic score/event/digest computation on both assigned AMD WX9100 devices. It limits CPU to process supervision, configuration, and opaque I/O; prohibits CPU numerical/score/event/digest work, P600 substitution, and silent migration; requires failure if either device or a required HIP path is unavailable; and asks for per-device identity, placement, memory, and transfer/synchronization evidence.

No regression was found in the remaining scoped requirements. The approved source base remains `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`; the exact source-map file hashes and finite allowlist are unchanged. The plan remains a successor to accepted P1 plan `53686a254318484f9c6bc3496ecc1c461572c53a`, not a rewrite of its history. P1.1 still must resolve generation, sequence/order/reset, clock, and score-ledger-source questions solely from the pinned base and accepted contracts or stop for a reviewed amendment before P1.3. It does not authorize implementer-selected identity semantics. The shared trainer and caller integration remain exclusively P5-owned, and P5 requires an independent pre-G0 SOURCE/API ACCEPT for the exact P1 API before authoring.

The finite allowlist remains limited to the P1 source-map/provenance document, P1 schema/event modules, two score-mutation/reset/environment seams, and the listed focused fixtures. Pre-G0 work remains source/test-file authoring and static review only; builds, imports/project execution, tests, simulator, dashboard, devices/GPU, training, checkpoints, integration, merge, and promotion remain blocked until G0 and the reviewed bridge gates are met.

## Review limits

This ACCEPT applies only to commit `40d151dec437fe405bd84d2918b2b86ae6f4a129` and the exact file hash above. It accepts the Pass 1 plan only; it does not assign implementation/reviewer roles, accept a source/API head, satisfy G0, or authorize runtime activity. No tests, builds, simulator, dashboard, GPU/device, campaign, or lease operations were performed.
