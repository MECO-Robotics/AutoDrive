# Independent Pass 1 review: P4 staged source/runtime gate

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/abi_freeze_planner`
- **Reviewer task:** independent exact-head review assigned by `/root`
- **Plan author:** `plan_c_owner`
- **Reviewed commit:** `257ed8afec359752702ae80e56b873ce4a35f7da`
- **Parent:** `8708ba8a96536f1a3ae28133afcd4ebcb4790378`
- **Reviewed file:** `docs/plans/meco-v4-p4-plan-c-c1-c2-scoped-package-plan-20261026.md`
- **Reviewed file SHA-256:** `3365d403bd6c4da2376b0e5375129c1d6d224e07a9cb4f87182632f8bf02ab20`
- **Accepted overall plan:** `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`
- **Overall plan file SHA-256:** `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Review worktree:** `.worktrees/autodrive-v4-p4-staged-gate-review`

## Review scope and evidence

The amendment correctly stages authoring and execution. The accepted overall plan at `8efec9378d1912f0839d7c6cd8e88467fceb1d5a` permits isolated, docs-authorized source and test-file authoring plus static source/diff review before G0, from the approved implementation source base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`. It prohibits build/compile, test execution, numerical/runtime work, simulator, calibration, evaluation, dashboard, device/GPU use, training, checkpoint mutation, integration, merge, and promotion until the manual campaign stop and independent release of both WX9100 leases.

The P4 amendment's objective, dependency table, worktree cadence, and Pass 3 section preserve that separation. It additionally gates runtime/execution and integration on recorded maintainer authorization and review of the post-G0 contract-anchor integration. It reserves pre-G0 review to structural/source/API disposition and requires a fresh post-G0 final review with tests, GPU, seeded simulator/replay, and current dashboard provenance. Its exact implementation source base matches the accepted overall plan. Its P1/P2/P3 dependencies are explicitly future exact implementation/API heads, so accepted package plans are not misrepresented as implementation APIs. No conflicting permission to touch the campaign or devices was found.

The Plan C scope and interface remain bounded to the P4-owned C1/C2 models and scoring, with P2 candidate/route producers, P5 trainer integration, P1 artifacts, P3 Plan A, and P6 qualification/dashboard kept outside its allowlist. No code, tests, build, runtime, campaign, device, simulator, or dashboard activity was performed for this review.

## Finding

**F1 — The pinned overall-plan reference is stale.** The P4 file labels commit `6325920252773cfa05d63167ba3c29a90debdb01` as its accepted overall plan. That plan has an accepted successor, `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, whose Pass 1 record supersedes the earlier review and clarifies the staged source-only gate. Although P4's gate wording is substantively coherent with the successor, its provenance pin does not identify the current accepted overall plan. Update the P4 header to pin `8efec…` and its file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`, while retaining historical references only when labeled as earlier/superseded. Then request a fresh exact-head review.

## Review limits

This REVISE applies only to the exact P4 candidate commit and file hash above. The staged gate is otherwise coherent with the accepted overall plan, but this review does not accept the P4 plan until F1 is corrected. It does not accept implementation, satisfy G0/P0a, authorize campaign/device/runtime actions, or establish simulator/dashboard validation or human approval.
