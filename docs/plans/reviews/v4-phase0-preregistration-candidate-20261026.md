# Independent Pass 3 review: P0a Phase 0 preregistration candidate

- **Verdict:** PENDING — structurally ready; final ACCEPT withheld pending authorized maintainer decisions
- **Review date:** 2026-10-26
- **Reviewer identity:** `plan_c_owner`
- **Reviewer task:** independent exact-head Pass 3 review assigned by `/root`
- **Candidate author:** `p0a_implementer` (documentation author; not reviewer or decision authority)
- **Reviewed commit:** `d651bae150751a1ed214031dda314dc4d9d2bba5`
- **Reviewed file:** `docs/plans/v4-phase0-preregistration-candidate-20261026.md`
- **Git blob:** `a9e3e597278862105915e3eb99e15896de41ae69`
- **File SHA-256:** `f41fc7d499ca3aed573e36c4efa274d19054703111515073abc93f9b5b3d6e80`
- **Pinned P0 package plan:** `72ee8fce1d4a955d100b97d47b58528374faddec`, `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md`, SHA-256 `16c1fbcd881bd77adf178c8d8bf5f17bd3cf3e988cde767fbbc7731c8ff088d9`, blob `4527835ff9d3216e0a0e9e1508edb1e25f0f5cd0`.
- **Pinned overall plan:** `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; exact Pass 1 ACCEPT record `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
- **Source reference base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- **Reconciled profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.
- **Superseded exact-head review:** REVISE for candidate `da8ba7453d3bab71567f1fb10e14d20b77673b9b`, review commit `cbc157eb5d935697ae8dffd043a00db366986926`, record SHA-256 `6e36a0275bf71164114c8da24c95c9dcd25651d69fdae372b9c98f7ce6f26406`.

## Review result

The prior matrix/helper mismatch is corrected. Candidate §10 now explicitly identifies its one-layout/one-family calibration and validation rows as a proposed v5 override of the legacy axes. It applies defender families only to `offense_vs_scripted_pressure` and `full_match_offense`, with `none` on the other closure scenarios, matching the condition in `frc_defense/meco_qualification.py:82-85`. It gives the exact stratum namespace preimage as UTF-8 `layout/family/start/sensor`, SHA-256 first four bytes interpreted big-endian, then calls `training_curriculum.seed_partition` with the scenario/split and `(20261024 << 32) | namespace`; category is excluded from the namespace as in `meco_qualification.py:91-99`. The candidate accurately labels the narrower axes as an explicit proposal rather than behavior supplied by the legacy helper. The current proposal preserves split-bit separation and distinguishes whole-family holdout for final-test `split_chaser` from the reused calibration/validation `orbiting_interceptor` family; it calls for frozen policy/checkpoint/config identities.

The other prior correction remains sound: §2 specifies 160 physics ticks, at most 12 physics ticks per macro, a partial final macro, gamma `0.993`, branch exponent zero, and one exponent increment per continuation decision. The target closure and category inventory match `training_curriculum.py:70-95`. The listed layouts, defender families, starts, sensors, episode floor, bootstrap default and safety metric identifiers are present in `meco_qualification.py:24-48`. The proposal distinguishes accepted active-HUB FUEL awards from official match points and curriculum reward, and defers source-bound score semantics and P0b artifacts until after source freeze/G0.

No substantive content defect remains in this candidate's reviewed scope. However, §10 explicitly leaves product/rules decisions unresolved, including score aggregation scope, target/case approval, held-out cases and opponent identities, sample/statistical choices, safety vetoes, missing-data mapping, reward implementation identity and D3. The candidate itself requires the authorized maintainer's decision record before P0a ACCEPT. Therefore this record states structural readiness only; it is not a Pass 3 ACCEPT and does not authorize dependent source work.

## Authority and limits

The candidate pins the accepted overall plan and P0 package plan, says it is not approved for calibration/training/qualification/runtime use, and reports no tests or runtime operations. No human approval is claimed. This review makes no product or rules decisions on the maintainer's behalf.

Read-only source and document review only. No candidate edits, tests, builds, runtime, simulator, calibration, GPU/device, dashboard, campaign, training, checkpoint, or output operations were performed.
