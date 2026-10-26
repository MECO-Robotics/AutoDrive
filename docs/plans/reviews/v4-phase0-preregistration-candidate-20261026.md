# Independent Pass 3 SOURCE/API review: P0a Phase 0 preregistration candidate

- **Disposition:** PENDING — structurally ready; final P0a ACCEPT withheld pending authorized maintainer decision record
- **Review date:** 2026-10-26
- **Reviewer identity:** `plan_c_owner`
- **Reviewer task:** independent exact-head SOURCE/API review assigned by `/root`
- **Candidate author:** `p0a_implementer` (documentation author; not reviewer, P0/P6 implementer, or decision authority)
- **Reviewed commit:** `4534186a324ee292b1c7bc3fdde740c5047c603e`
- **Reviewed file:** `docs/plans/v4-phase0-preregistration-candidate-20261026.md`
- **Git blob:** `903f74db372b81101b21736ae6efa1d76e30f8b8`
- **File SHA-256:** `af3d127086641527052acfeb5f2f847727771980aed4934f90aaab9924c485ac`
- **Accepted P0 package plan:** commit `3071c2496eb8bcb326ad0f1e971f1bd1e8d0858f`, file `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md`, blob `f8e1827fc0354b88a908792541aa515790b4bed0`, SHA-256 `5ac9e99cf0ead523cecf6cdc64753d7801361fc876b862f762ae18f70ef4c831`; exact-head Pass 1 ACCEPT record commit `f2f299a81d266dc5e08c5ce27402151c0e78571d`, record SHA-256 `baf8666f6e93550f31afc441e87eaeb089dffd6e4401376bb943ca978e4d529e`.
- **Accepted P6 dual-protocol schema/runner plan:** commit `c1f20c80a15588725b3f346ae06366c2197465b7`, file `docs/plans/meco-v4-p6-five-track-supervisor-qualification-package-plan-20261026.md`, SHA-256 `8c122a5ea82e2f3b77f222c8954a92ae6d11e54a4ded55bfdb786d42283fc516`; exact-head Pass 1 ACCEPT record commit `f025827abe2d3aec366b65b4fad85debd5de31aa`, record SHA-256 `821a11b053c86780bca35aff45e53373c003d7046468e4e1b6905d39d67df6c8`.
- **Accepted overall plan:** commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; Pass 1 ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
- **Source reference base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- **Current Stage 1 contract evidence:** `frc_defense/meco_qualification.py` at `main` commit `d74b0a15e25ddbc6bacfbbb864674371629bb19e`; `preflight_stage1` checks target `pickup_simple` at lines 4756–4760 and campaign binding at 4922–4924.
- **Reconciled profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.

## Review result

The exact accepted plan and review pins are accurate. The accepted P0 successor has its own exact-head Pass 1 ACCEPT record. The accepted P6 amendment defines the dual-protocol schema/runner ownership and cross-binding scope, with its own exact-head Pass 1 ACCEPT record. These plan ACCEPTs authorize planning scope only; they do not accept P0a's decisions or claim that P6 implementation/API source exists or conforms.

The dual-target design is coherent with current source and accepted ownership. Stage 1 qualification is target-scoped to `pickup_simple`, matching `preflight_stage1` (`frc_defense/meco_qualification.py:4756-4760`) and campaign binding (`:4922-4924`). Its registered prerequisite closure is itself and its registered categories are `single` and `cluster` (`frc_defense/training_curriculum.py:70-72`). The primary research score protocol is separately target-scoped to `full_match_offense` and measures accepted active-HUB FUEL awards; it cannot establish Stage 1 passage, and the pickup curriculum predicate cannot substitute for the score outcome. The candidate preserves distinct protocol identities, case/partition bytes, calibration evidence namespaces and raw digests, cross-referenced by the Phase 0 artifact.

The proposed Stage 1 matrix applies only to the two pickup categories and `none` defender family. The full-match matrix uses defender families only on `offense_vs_scripted_pressure` and `full_match_offense`, with `none` elsewhere, and identifies its calibration/validation axis narrowing as an explicit v5 override. The seed namespace preimage and hash construction agree with `meco_qualification.protocol_cases` (`:82-99`) and `training_curriculum.seed_partition` (`:102-116`). The fixed return contract remains 160 physics ticks, max 12 physics ticks per macro with partial final macro, gamma `0.993`, branch exponent zero and one exponent increment per continuation decision.

P0's accepted plan and P6's accepted plan now cover the second protocol, separate evidence paths and Phase 0 cross-binding. Candidate §8 accurately requires P6 source/API implementation and exact-head review before P0b; P0a does not edit qualification code. P0b remains behind source freeze/review, G0, independent lease release and the reviewed P6 runner. No tests/runtime or training authority is claimed.

No source/API structure defect was found. The authorized maintainer decision record is still absent. Candidate §10 leaves score aggregation ownership/tie treatment, dual-target protocol approval, finite maps/starts/opponent identities, case and partition membership, sample floor, uncertainty/multiplicity, safety vetoes, missing/failure effects and comparator set open. Therefore this record finds structural readiness only; it is not final Pass 3 ACCEPT and does not authorize dependent source work.

## Authority and limits

The candidate states it is not approved for calibration, training, qualification or runtime. No human approval is claimed. This review does not resolve product decisions or accept future P6 source/API implementation.

Read-only source and document review only. No candidate edits, tests, builds, runtime, simulator, calibration, GPU/device, dashboard, campaign, training, checkpoint or output operations were performed.
