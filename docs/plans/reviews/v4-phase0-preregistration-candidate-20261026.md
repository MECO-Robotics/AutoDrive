# Independent Pass 3 review: P0a Phase 0 preregistration candidate

- **Verdict:** PENDING — structurally ready; final ACCEPT withheld pending authorized maintainer decisions
- **Review date:** 2026-10-26
- **Reviewer identity:** `plan_c_owner`
- **Reviewer task:** independent exact-head Pass 3 review assigned by `/root`
- **Candidate author:** `p0a_implementer` (documentation author; not reviewer or decision authority)
- **Reviewed commit:** `8b82c19884ac89f42113ab713cd03c156b7ed5bb`
- **Reviewed file:** `docs/plans/v4-phase0-preregistration-candidate-20261026.md`
- **Git blob:** `2cff64fea41f894910f3269bc094a5d89bb226c9`
- **File SHA-256:** `53682d0eb9f5f5fbb392f9f4259cfb94b7d36d98bef0e9f2c96f8b92db93674d`
- **Pinned P0 package plan:** `72ee8fce1d4a955d100b97d47b58528374faddec`, `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md`, SHA-256 `16c1fbcd881bd77adf178c8d8bf5f17bd3cf3e988cde767fbbc7731c8ff088d9`, blob `4527835ff9d3216e0a0e9e1508edb1e25f0f5cd0`.
- **Pinned overall plan:** `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; exact Pass 1 ACCEPT record `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
- **Source reference base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- **Current Stage 1 contract evidence:** `frc_defense/meco_qualification.py` at `main` commit `d74b0a15e25ddbc6bacfbbb864674371629bb19e`; `preflight_stage1` validates target `pickup_simple` at lines 4756–4760 and campaign binding at 4922–4924.
- **Reconciled profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.
- **Prior exact-head disposition:** PENDING, structurally ready candidate `d651bae150751a1ed214031dda314dc4d9d2bba5`; review commit `129622512cac153476d345c5ca9ddc29f1f94fa1`, record SHA-256 `a5b3a2e73377156f9b0bf8274449b8b2054d5f50888b820c5b119ddbdf3fa5f7`. That candidate still proposed `full_match_offense` as the Stage 1 target; this review checks its replacement with the current preflight target.

## Review result

The new Stage 1 target is source-consistent. The candidate selects `pickup_simple`, whose registered prerequisite list is empty and whose categories are exactly `single` and `cluster` (`frc_defense/training_curriculum.py:70-72`). The current preflight source requires the protocol target to equal `pickup_simple` (`frc_defense/meco_qualification.py:4756-4760`), and the campaign binding also fixes that scenario (`:4922-4924`). The proposed Stage 1 matrix uses those two categories, the `none` defender family for this non-pressure scenario, and one explicit layout per partition. It clearly labels the calibration/validation layout narrowing as a v5 override requiring maintainer approval. The optional `full_match_offense` matrix remains a separate generalization analysis and cannot affect Stage 1 readiness.

The candidate separates three meanings without substituting one for another: (1) `pickup_simple`'s curriculum acquisition/retention gate as the Stage 1 predicate, (2) accepted active-HUB FUEL awards as the user-selected research score proxy, and (3) gross counterfactual PPO return. It explicitly warns that the award proxy may be zero or uninformative on `pickup_simple`, requires P0b to stop if that prevents comparator ranking or margin calibration, and prohibits using curriculum reward or optional full-match outcomes as replacements. This is consistent with the target's actual physical outcome and the documented Stage 1 preflight target.

The exact matrix and seed rule are internally coherent: the case tuple includes scenario/category/layout/family/start/sensor; the stratum namespace is the first four SHA-256 bytes of UTF-8 `layout/family/start/sensor`; the scenario and split are passed separately to `seed_partition`; category is excluded as in `protocol_cases`. Stage 1 uses `none` for its non-pressure target. The optional full-match matrix applies defender families only to the two pressure scenarios and `none` elsewhere, and its v5 override of legacy calibration/validation axes is explicit. The prior review's corrected 160-physics-tick, macro limit, partial-final-macro, gamma exponent, accepted-award proxy, and source-grounded matrix handling remain present.

No substantive content defect remains in the reviewed scope. The candidate's §10 still requires authorized maintainer decisions on award aggregation scope and its relation to Stage 1, map/start and opponent choices, matrix/partition bytes, sample and statistical methods, safety vetoes, missing/failure mapping, and comparator set. The exact decision record was not provided. This record therefore says structurally ready only; it is not a Pass 3 ACCEPT and does not authorize dependent source work.

## Authority and limits

The candidate pins the accepted overall and P0 package plans, says it is not approved for calibration/training/qualification/runtime use, and reports no tests or runtime operations. No human approval is claimed. This review makes no product or rules decisions on the maintainer's behalf.

Read-only source and document review only. No candidate edits, tests, builds, runtime, simulator, calibration, GPU/device, dashboard, campaign, training, checkpoint, or output operations were performed.
