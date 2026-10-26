# Independent Pass 1 review: V4 implementation-plan gate amendment

- **Verdict:** ACCEPT (supersedes the REVISE record for commit `5b25eb1cfba7850a871b9024d834457d0fb755a4`)
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** fresh exact-head review assigned by `/root`
- **Plan author:** `plan_a_owner`
- **Reviewed commit:** `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`
- **Parent:** `5b25eb1cfba7850a871b9024d834457d0fb755a4`
- **Reviewed file:** `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md`
- **Git blob:** `2f1da1e2e571aaf2826f5f9a11b0b83eb4475fdc`
- **File SHA-256:** `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted reconciliation:** commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; canonical profile hash `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e`; V4 envelope digest `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- **Earlier accepted plan head:** commit `6325920252773cfa05d63167ba3c29a90debdb01`, file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`; Pass 1 ACCEPT record commit `be845d7fb2d2ece861dcd158ac8b8dd89e3dc59d`, record SHA-256 `5a6e94f978c8a7e18d9623897ceba6f32f5b5eadb0aba6605f21c9d1e941a886`.
- **Superseded review record:** commit `00095fe2a421f7fe8c2809e1275d56e9c881fb2e`, REVISE record SHA-256 `775a1439484e83fb172018f89444d57c5e3e65acf3464153222ed2f43a5b743a`.

## Review scope and evidence

The exact-head correction resolves the prior ambiguity about pre-G0 authoring and the contract-anchor bridge. Pre-G0 source/test authoring and static review use fresh, isolated worktrees branched from the approved implementation source base or a pinned candidate source head. They may cite pinned reconciliation/profile hashes, but may not add, copy, or depend on the reconciliation artifact being present in that checkout. The reviewed contract-anchor bridge is explicitly post-G0 and must be accepted before validation or integration; pre-G0 source-only branches remain isolated and do not use the missing artifact as a file dependency.

The staged review process is coherent. Before G0, an independent reviewer may record SOURCE/API ACCEPT or REVISE for the exact candidate source head. SOURCE/API ACCEPT pins a candidate interface for dependent source authoring only; it does not permit integration or runtime use. G0 requires the user's manual stop of campaign PID `2750253`, independent confirmation that both WX9100 leases are released, and explicit authorization for runtime/device/integration work. After G0, the bridge, package validation and runtime evidence are reviewed; only final exact-head Pass 3 ACCEPT authorizes integration. No test execution, build, compile, numerical/runtime work, simulator, calibration, evaluation, dashboard, device/GPU use, training, checkpoint mutation, merge or promotion is permitted before G0.

The no-contact boundary for PID `2750253` and its dashboard, workers, checkpoints, outputs, leases and devices remains explicit. The plan preserves two-WX9100 algorithmic execution, forbids CPU numerical/simulation/scoring fallback and P600/CPU fallback, and requires an explicit stop/replan if the devices or kernels are unavailable. It retains uninterrupted five-track training until the user manually stops it; read-only evaluations use immutable checkpoint snapshots, declare GPU shares, and cannot mutate training state. Stage 1 success does not stop training.

The adopted outcome remains `simulated_match_score` using accepted active-HUB FUEL awards only. The fixed strategic horizon remains 160 physics ticks, at most 12 ticks per macro with a partial final macro allowed, gamma 0.993 per macro, first-branch exponent zero and one increment per continuation decision. Route and risk costs remain separate from gross reward. These match the accepted reconciliation and user's decisions.

P0a remains a documentation-only preregistration and human-decision gate before P1–P6 source authoring. P0b remains after frozen/reviewed P1–P6 source; P6 runtime qualification waits for accepted P0b. The top-level dependency graph has no P0/P6 runtime cycle. Required simulator evaluation and current dashboard/backend/frontend verification remain after G0 and after code changes, consistent with repository AGENTS instructions. No human approval, source conformance, campaign stop, lease release, training progress or qualification result is claimed.

No blocking finding remains for this exact plan head.

## Review limits

This ACCEPT applies only to the exact commit and file hash above. No source, test execution, build, runtime, simulator, device/GPU, dashboard, qualification, campaign or training actions were performed during this review.
