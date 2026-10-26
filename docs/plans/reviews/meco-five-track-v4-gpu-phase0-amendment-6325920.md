# Independent Pass 1 review: top-level V4 Phase 0 sequencing amendment

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent exact-head Pass 1 review assigned by `/root`
- **Plan author:** `plan_a_owner`
- **Reviewed plan commit:** `6325920252773cfa05d63167ba3c29a90debdb01`
- **Parent commit:** `134336a822c8006da0c3ce60cecba18a96ff2b77`
- **Reviewed file:** `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md`
- **Git blob:** `521ce753719b9f5f4a7e27fe9b57e2e1edd00b17`
- **Plan file SHA-256:** `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted reconciliation:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; accepted review record commit `5cb1a2d525168ff0443089cf329e66c6ecb917b8`, record SHA-256 `0df700dcaef1ed35672efc468bdcfd78fac829c31bcf7a88b3c51f1951f06c32`.
- **Reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **V4 envelope profile digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Review scope and evidence

The reviewed plan commit descends from the previously accepted top-level plan, and the exact pinned implementation source base is an ancestor of the reviewed commit. The amendment keeps the original source base and accepted reconciliation/profile pins. Its file changes clarify P0a candidate authorship and human decisions, prohibit P0a source/test work, assign the v5 qualification schema, runner, and qualification tests to P6, and limit P0b to runtime calibration and generated immutable artifacts with no qualification-source edits.

P0a now has a viable decision sequence: Pass 2 prepares a documentation-only candidate; the authorized maintainer resolves the required product/rules decisions against it; Pass 3 reviews that exact candidate together with recorded decisions; only P0a ACCEPT plus G0 opens P1–P6 source work. The selected target set remains an explicit product/maintainer decision, not an inference from the current scenario. P0a produces no calibration result, final artifact, protocol instance, training, or qualification result.

The dependency order is acyclic: accepted P0a → P1–P6 source implementation/integration and exact-head review, with P6 owning its qualification runner and tests → source freeze → P0b calibration and generated artifact/protocol → fresh source-bound checkpoints and P6 runtime qualification. P0b expressly does not depend on P6 runtime results, and P6 runtime qualification requires P0b acceptance. The amendment requires P0b to emit artifacts without source edits, verify exact frozen source/rules/profile/device identity and P1's score producer/consumer map, use the preregistered D0 calibration method and assigned two-WX9100 devices without CPU numerical fallback, retain evidence for independent recomputation, and invalidate the source binding after relevant post-freeze changes.

G0 remains a hard boundary: manual stop of the active campaign, independent GPU lease release, explicit authorization, and the reviewed docs-only contract-anchor bridge precede source packages. The plan still bars actions on the live campaign and requires two-WX9100 algorithmic execution. The accepted reconciliation/profile pins and five-track scope remain intact. The plan's seeded end-to-end simulator and current-dashboard evidence requirements remain future validation gates, consistent with the repository AGENTS.md. No such activity was performed during this review.

The commit also contains `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md`. This ACCEPT is scoped to the exact top-level plan file and does not accept that separate package plan. Its own exact-head Pass 1 review remains governed by its separate review record; P0 implementation must not begin based on this top-level ACCEPT alone.

## Review limits

This ACCEPT applies only to the exact top-level plan file and commit pinned above. It does not authorize source implementation, tests, builds, device/runtime work, calibration, dashboard actions, training, evaluation, ABI publication, or qualification. Those actions remain gated by G0 and the accepted scoped package plans. No source, tests, runtime, GPU, dashboard, simulator, or campaign actions were performed during review.
