# Independent Pass 1 review: V4 schema, score-event, and artifact implementation plan

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent Pass 1 plan review assigned by `/root`
- **Plan author:** `/root/plan_c_owner`
- **Reviewed plan commit:** `c9d37b43bbddaa0984665dd918a22c572a5cc59d`
- **Reviewed file:** `docs/plans/meco-five-track-v4-schema-score-artifact-implementation-plan-20261026.md`
- **Git blob:** `af883defbd60e9049e2b6b38afe783a857960245`
- **Plan file SHA-256:** `c3f31af40656e1d070d0297b135d73f7bb912fe8afb3cd01fd7984338104e7ff`
- **Pinned source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted reconciliation:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, file SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; exact Pass 3 ACCEPT record `5cb1a2d525168ff0443089cf329e66c6ecb917b8`, record SHA-256 `0df700dcaef1ed35672efc468bdcfd78fac829c31bcf7a88b3c51f1951f06c32`.
- **Reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **V4 envelope profile digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Review scope and evidence

The reviewed plan is a direct child of the exact pinned source base, and its sole change is this planning document. The reconciliation pins match the accepted reconciliation record: the canonical reconciled profile hash is distinct from the raw proposal digest used by the V4 envelope. The plan bounds the P1 implementation to the typed V4 transition/event schema, producer/consumer interfaces, and artifact envelope. It describes a finite package allowlist and requires a read-only producer/consumer source map before implementation edits, including explicit handling of the accepted Gamepiece response's source-path and line-number mismatch. It does not invent the missing score-ledger module or treat the citations as proof of this base's implementation.

The operational boundaries are sound: no source/runtime/test/device/dashboard/training action is authorized by the plan; the live campaign and its resources are protected behind G0 and separate release authorization; CPU numerical fallback is prohibited; and algorithmic work is assigned to the two WX9100 GPUs with fail-closed behavior. The plan also captures the AGENTS.md validation requirements for seeded end-to-end simulator behavior checks, fresh replay provenance, restarting and verifying the current dashboard stack, and reporting scenario, seed, replay, and result. Planner, implementer, and independent reviewer are distinct roles. These are future gates, not actions performed in this review.

## Blocking finding: P0 sequencing conflicts with required P0a/P0b split

The plan makes the fully accepted P0 score-source/rules artifact a prerequisite for **P1.1**, whose purpose is to perform the read-only producer, mutation, reset, and consumer discovery needed to establish that source-bound contract. The prerequisite appears explicitly under “Exact base inspection and source-discovery gate” (“Until P0 is complete…”) and in P1.1’s dependency (“G0 and accepted P0 artifact”). Yet P0 itself must name the authoritative producer, all metric/report consumers, the physical identity lifecycle, and source-derived event semantics. The plan acknowledges that the cited score-ledger module is absent from the pinned tree and that the authoritative source/consumer remains unresolved. This creates a circular gate: final source-calibrated P0 is required before the read-only work that is supposed to resolve its source evidence.

This also conflicts with the current sequencing amendment separating **P0a preregistration before source edits** from **P0b final calibration/artifact after all source is frozen**. The plan should be amended so P1.1's read-only source mapping depends on G0 and the approved P0a preregistration, not the fully calibrated P0b artifact. Keep schema/source edits and any score-dependent implementation gated on the accepted P0b artifact, with its exact pins and source map. The plan must state where calibration can occur after source freeze without requiring source edits to begin under an unapproved score contract, and preserve the rule that changed source/score semantics return to reconciliation.

Because the plan currently blocks P1.1 on final P0 while final P0 requires P1.1's discovery, and is pinned to a sequencing model being amended, this exact-head plan is **REVISE**. No implementation is authorized by this review. The owner should publish a corrected plan head after the top-level P0a/P0b amendment is pinned, then request a fresh exact-head Pass 1 review.

## Review limits

This REVISE applies only to the exact planning artifact pinned above. It does not authorize implementation, tests, builds, device/runtime work, campaign actions, dashboard work, evaluation, or qualification. No tests, simulator, GPU, dashboard, or training actions were performed during review. No implementation source or owner response was changed.
