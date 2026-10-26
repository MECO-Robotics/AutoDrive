# Independent Pass 3 review: P0a Phase 0 preregistration candidate

- **Verdict:** PENDING — structurally ready; final ACCEPT withheld pending authorized maintainer decisions
- **Review date:** 2026-10-26
- **Reviewer identity:** `plan_c_owner`
- **Reviewer task:** independent exact-head Pass 3 review assigned by `/root`
- **Candidate author:** `p0a_implementer` (documentation author; not reviewer or decision authority)
- **Reviewed commit:** `e50a9556a57c368208d8537a82ab61f54bee79cd`
- **Reviewed file:** `docs/plans/v4-phase0-preregistration-candidate-20261026.md`
- **Git blob:** `074b106324789a84001a952fc5379a569c5dfa95`
- **File SHA-256:** `2e8db74411b26d9d9c08d1ebd938b43cce4b40ea9c768c17ee3734c041e43add`
- **Pinned P0 package plan:** `72ee8fce1d4a955d100b97d47b58528374faddec`, `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md`, SHA-256 `16c1fbcd881bd77adf178c8d8bf5f17bd3cf3e988cde767fbbc7731c8ff088d9`, blob `4527835ff9d3216e0a0e9e1508edb1e25f0f5cd0`.
- **Pinned overall plan:** `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; exact Pass 1 ACCEPT record `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
- **Source reference base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- **Current Stage 1 contract evidence:** `frc_defense/meco_qualification.py` at `main` commit `d74b0a15e25ddbc6bacfbbb864674371629bb19e`; `preflight_stage1` validates target `pickup_simple` at lines 4756–4760 and campaign binding at 4922–4924.
- **Reconciled profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.
- **Prior exact-head disposition:** PENDING, structurally ready candidate `8b82c19884ac89f42113ab713cd03c156b7ed5bb`; review commit `24e659d33be8e5d9a552321b9f93370287228274`, record SHA-256 `46ca5178b185392311f5780e4439da4d64eba1e66f465772d6336b6bacbc1b33`. This review applies only to the new dual-protocol candidate.

## Review result

The candidate now separates the qualification target from the score estimand in a way consistent with the pinned source and product objective. The Stage 1 protocol is target-scoped to `pickup_simple`, matching `preflight_stage1` (`frc_defense/meco_qualification.py:4756-4760`) and the campaign binding (`:4922-4924`). The source registry gives `pickup_simple` no prerequisites and categories `single` and `cluster`; its physical outcome is acquiring and retaining distinct FUEL (`frc_defense/training_curriculum.py:70-72`). The candidate binds Stage 1 passage only to that curriculum predicate and its v5 qualification protocol.

The primary research score protocol is separately scoped to `full_match_offense` and accepted active-HUB FUEL awards. It explicitly states that neither a full-match award result nor a `pickup_simple` curriculum pass substitutes for the other. It retains the 160-physics-tick outcome and accepted award event, distinguishes that estimand from gross PPO return, and requires P0b to stop on insufficient score calibration rather than silently changing target or outcome. The full-match closure and scenario/category inventory match `training_curriculum.py:70-95`.

The two matrix proposals are internally consistent with their stated roles. Stage 1 uses only the two `pickup_simple` categories and `none` defender family. The full-match matrix uses the two pressure-scenario defender families only for `offense_vs_scripted_pressure` and `full_match_offense`, with `none` for other closure scenarios, and labels its narrower calibration/validation axes as an explicit v5 override. The common seed rule specifies the stratum namespace preimage/hash and delegates scenario/split separation to `seed_partition`, consistent with `meco_qualification.py:82-99` and `training_curriculum.py:102-116`. The candidate continues to distinguish source-grounded defaults from maintainer-approved v5 semantics.

The P0/P6 protocol and artifact dependency is explicit and correctly assigned. The candidate records that the accepted P0 plan currently lists one `protocol.json` and a single calibration path, proposes separate Stage 1 and score-evaluation protocol/evidence paths, and requires the P0 plan's artifact allowlist to be amended and P6's schema/runner to support the binding before P0b. This matches the accepted ownership boundary: P0 owns no source/tests and P6 owns v5 qualification schema/runner/tests (P0 plan `72ee8f...:19,23`; overall plan `8efec9...:474-483`). The candidate does not claim those future artifacts or compatibility code already exist, and it does not authorize P0a to change P6 code.

No substantive content defect remains in the reviewed scope. However, §10 still requires maintainer decisions on score aggregation, dual-target design, finite maps/starts/opponents and case tuples, sample/statistical methods, safety vetoes, missing/failure mapping and comparator set. Those decisions were not supplied. The P0 package plan also needs its stated P0b artifact allowlist amendment, and the accepted P6 plan/schema must resolve the two-protocol binding before P0b. This review therefore records structural readiness only, not Pass 3 ACCEPT or authorization to begin dependent source work.

## Authority and limits

The candidate pins the accepted overall and P0 package plans, states it is not approved for calibration/training/qualification/runtime use, and reports no tests or runtime operations. No human approval is claimed. This review makes no product or rules decisions on the maintainer's behalf.

Read-only source and document review only. No candidate edits, tests, builds, runtime, simulator, calibration, GPU/device, dashboard, campaign, training, checkpoint, or output operations were performed.
