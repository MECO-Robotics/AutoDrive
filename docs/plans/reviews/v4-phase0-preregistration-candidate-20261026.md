# Independent Pass 3 review: P0a Phase 0 preregistration candidate

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Reviewer identity:** `plan_c_owner`
- **Reviewer task:** independent exact-head Pass 3 review assigned by `/root`
- **Candidate author:** `p0a_implementer` (documentation author; not reviewer or decision authority)
- **Reviewed commit:** `a16881ddbea41f8efc0d4f32439b8a99964a4d38`
- **Reviewed file:** `docs/plans/v4-phase0-preregistration-candidate-20261026.md`
- **Git blob:** `86429d9f1bf075b8197ddc9c407574c193422303`
- **File SHA-256:** `643a7f18e85afb0bf7692f1cb0d40a97d1abb12a7ac73b2e9b600496da38f449`
- **Pinned P0 package plan:** `72ee8fce1d4a955d100b97d47b58528374faddec`, `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md`, SHA-256 `16c1fbcd881bd77adf178c8d8bf5f17bd3cf3e988cde767fbbc7731c8ff088d9`, blob `4527835ff9d3216e0a0e9e1508edb1e25f0f5cd0`.
- **Pinned overall plan:** `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; exact Pass 1 ACCEPT record `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
- **Source reference base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- **Reconciled profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.

## Finding

**F1 — Fixed-horizon wording conflicts with the adopted evaluation contract.** The candidate calls the estimand an award count per “160-strategic-tick episode” and lists a “160 strategic ticks” reward horizon (`§1`, line 19; `§2`, line 25). The accepted overall plan and reconciliation specify 160 **physics** ticks, at most 12 simulator ticks per macro with a partial final macro, gamma 0.993 per macro, branch exponent zero, and one exponent increment per continuation decision (overall plan `§ Reconciled product and profile decisions`, lines 92–97; P5 target lines 413–416). The candidate lists the duration and gamma but does not state partial-final-macro support or the branch/continuation exponent convention. Update the candidate to use the exact adopted 160-physics-tick horizon and explicitly preregister all accepted macro/discount semantics. This is a content defect independent of the unresolved maintainer choices.

## Review scope and evidence

The candidate pins the requested P0 package and accepted overall-plan heads, source reference base, reconciliation/profile hashes, and raw V4 digest. It is one documentation-only file and claims no tests, runtime, simulation, calibration, dashboard or device use. The proposal consistently identifies accepted active-HUB FUEL awards as a limited simulated proxy rather than official match points, distinguishes curriculum reward, and preserves gross reward separately from route/risk costs. It specifies paired episodes, D0/D1/D2 comparator selection only on calibration data, disjoint train/calibration/validation/final-test partitions, deterministic seed derivation, safety non-regression, bootstrap analysis, source-bound identities, immutable P0b artifacts and failure handling.

The open maintainer choices in §10 are explicit and appropriately block ACCEPT: award aggregation scope; target ID/closure; held-out maps, starts and opponent families; finite case matrix and partition bytes; sample floors; bootstrap/multiplicity choices; safety metrics/vetoes; missing/failure handling; reward implementation identity; and D3 inclusion. No decision record resolving them was supplied, and this review does not infer or grant approval. Also, the pinned P0 package head itself marks its corrected G0/source-authoring amendment pending fresh review; that package-plan gate must be resolved under its own review before P0a can receive final ACCEPT.

The reviewed candidate's exact heading and field names are source-bound to base `6ed961a...`; the source citations in the pinned P0 plan describe current score events and distinguish `curriculum_metrics` from the score estimand. P1's later event-source map is correctly required before P0b. Nothing in this review establishes runtime behavior or calibration results.

No human approval is claimed. **Even after F1 is corrected, P0a cannot receive ACCEPT until the authorized maintainer records the §10 decisions against the exact candidate and those decisions are included in the exact-head review.**

## Review limits

No candidate edits, tests, builds, runtime, simulator, calibration, GPU/device, dashboard, campaign, training, checkpoint or output operations were performed. This REVISE applies only to the exact candidate commit and hashes above.
