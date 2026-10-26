# Independent Pass 3 review: P0a Phase 0 preregistration candidate

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Reviewer identity:** `plan_c_owner`
- **Reviewer task:** independent exact-head Pass 3 review assigned by `/root`
- **Candidate author:** `p0a_implementer` (documentation author; not reviewer or decision authority)
- **Reviewed commit:** `da8ba7453d3bab71567f1fb10e14d20b77673b9b`
- **Reviewed file:** `docs/plans/v4-phase0-preregistration-candidate-20261026.md`
- **Git blob:** `5f3f9fb9ccb4f4745c6715ce7d97e7b4f0eeb9a5`
- **File SHA-256:** `0ba9bce185b96921373c27630b99f5f8ba4c68aab4456b8de4f893489938117b`
- **Pinned P0 package plan:** `72ee8fce1d4a955d100b97d47b58528374faddec`, `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md`, SHA-256 `16c1fbcd881bd77adf178c8d8bf5f17bd3cf3e988cde767fbbc7731c8ff088d9`, blob `4527835ff9d3216e0a0e9e1508edb1e25f0f5cd0`.
- **Pinned overall plan:** `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; exact Pass 1 ACCEPT record `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
- **Source reference base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- **Reconciled profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.

## Finding

**F1 — Proposed finite case matrix does not match the cited seed-case helper.** The candidate's §4 says to use `training_curriculum.seed_partition(...)` with `meco_qualification.protocol_cases` stratum namespace construction. Its §10 matrix instead assigns one layout and one defender family to each calibration and validation partition, and says to expand every category across the full twelve-scenario closure. At the pinned source base, `meco_qualification._split_axes` gives both calibration and validation two layouts (`2026_rebuilt_shifted_structures_v1` and `_v2`) and two defender families (`orbiting_interceptor` and `split_chaser`) (`frc_defense/meco_qualification.py:36-48,63-72`). `protocol_cases` uses those axes for the target and pressure scenario, but assigns family `none` for every other scenario (`:75-99`). Therefore the helper does not produce the proposed one-row-per-partition matrix as written, and applying a defender family across the full closure conflicts with its current family semantics. Clarify whether the accepted v5 case matrix deliberately replaces these helper axes and define exactly how approved tuples map to seed namespaces, including `none` for applicable scenarios, or revise the matrix to match the source behavior. Keep this separate from the maintainer's authority to approve the proposed finite set.

## Review scope and evidence

The corrected candidate now uses the accepted 160 **physics**-tick horizon, maximum 12-physics-tick macros, partial final macro, gamma `0.993`, branch exponent zero and one exponent increment per continuation decision (§2). These match the reconciled profile convention and address the prior fixed-horizon finding. The target closure listed in §10 matches the source registry and dependency graph (`frc_defense/training_curriculum.py:70-95`); the proposed layouts, families, starts, sensors, 100-episode floor, bootstrap default and safety metric IDs are present in `frc_defense/meco_qualification.py:24-48`. The candidate correctly labels the finite case selection and statistical choices as proposals, distinguishes accepted active-HUB FUEL awards from official match points and curriculum reward, and defers source-bound event semantics and P0b artifacts until after source freeze/G0.

The candidate pins the requested accepted overall plan and P0 plan, states that it is not approved for calibration/training/qualification/runtime, and reports no tests or runtime operations. Its §10 maintainer choices remain unresolved. Even after F1 is corrected, final ACCEPT must wait for the authorized maintainer's decision record; this review does not infer or grant that approval.

No human approval is claimed. This REVISE applies only to the exact candidate commit and hashes above.

## Review limits

Read-only source and document review only. No candidate edits, tests, builds, runtime, simulator, calibration, GPU/device, dashboard, campaign, training, checkpoint, or output operations were performed.
