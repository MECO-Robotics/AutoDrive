# Independent Pass 1 review: P2 GPU-resident environment and route package plan

- **Verdict:** REVISE (supersedes the earlier ACCEPT for the original P2 plan; this amendment is a new exact head)
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent exact-head staged-gate review assigned by `/root`
- **Plan author:** `plan_a_owner`
- **Reviewed plan commit:** `1f918a9e0f99ec49a0baf4c3ce69649c4f095a2a`
- **Reviewed file:** `docs/plans/meco-v4-p2-gpu-resident-environment-route-package-plan-20261026.md`
- **Git blob:** `8f9df78f8d479e1168ca74f8e4d82e1e34699629`
- **Plan file SHA-256:** `ab640e4fdcfe63c8f1b97aa421297413d2c1304830031d81bb1a9c6cbdd07f50`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted overall staged-gate plan:** commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; Pass 1 ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`
- **Accepted P0 plan:** `13dbd7f8f953995850e811202f7e9c5264c49dca`, file SHA-256 `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`; ACCEPT record `037f0f9386cc0efadb1fb54efe5f38f097926d0e`, record SHA-256 `1ee70a6e9d24dc5201da5eec446820167cb8c4d4436fbbe013eb16de66b0280d`
- **Accepted P1 plan:** `53686a254318484f9c6bc3496ecc1c461572c53a`, file SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`; ACCEPT record `854d5286947074ed33f7ac059d67f2f9ccbf9257`, record SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f`
- **Accepted reconciliation/profile:** artifact commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; profile SHA-256 `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`; raw proposal digest `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Review scope and evidence

The amendment pins the accepted overall G0/source-authoring plan, source base, P0 plan, P1 plan, reconciliation artifact and profile. It retains P2's environment/observation/candidate/route scope, exclusive P5 ownership of the shared trainer, explicit two-WX9100/no-CPU rule, simulator and current-dashboard acceptance evidence, live-campaign boundary, and source/API versus final Pass 3 distinction. It prohibits execution and integration before G0.

### Finding: remove pre-G0 P1 integration prerequisites

The plan still requires integration before the source-only phase it introduces. The objective/authority paragraph requires that “P1 is accepted and integrated” before P2 Pass 2; Dependencies items 3–4 require the accepted P1 integration head before branching; the worktree instruction starts from that integrated head; the `tensor_3v3.py` allowlist ownership statement says P2 changes follow accepted P1 integration; and the risks section says P1 is integrated first. The accepted overall plan and this P2 amendment prohibit integration before G0. These requirements conflict, so P2 cannot both begin its pre-G0 source-only phase and follow its stated prerequisite.

Require an exact P1 candidate source/API head with a persisted pre-G0 SOURCE/API ACCEPT before P2 source authoring. Preserve the P1/P2 shared-file ownership order in the candidate interface contract, but defer integrating the P1 candidate and P2 changes until after G0, the reviewed contract-anchor bridge, and final post-G0 review. Pin the resulting exact integration head before validation and final acceptance. Until corrected, this review authorizes no P2 source work.

## Review limits

This review applies only to the exact P2 plan artifact and commit above. No source, tests, runtime, GPU, simulator, dashboard, campaign, or training actions were performed.
