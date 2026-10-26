# Independent Pass 1 review: V4 implementation-plan gate amendment

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent exact-head review assigned by `/root`
- **Plan author:** `plan_a_owner`
- **Reviewed commit:** `5b25eb1cfba7850a871b9024d834457d0fb755a4`
- **Parent:** `6325920252773cfa05d63167ba3c29a90debdb01`
- **Reviewed file:** `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md`
- **Git blob:** `68c351c72eae07323d65d95d54375e9fea7ff668`
- **File SHA-256:** `f5d1751ae6ca02452f286d756116859f176558a5822711a48f2b1aa4d734b43a`
- **Source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted reconciliation:** commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; canonical profile hash `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e`; envelope digest `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- **Previously accepted top-level amendment:** commit `6325920252773cfa05d63167ba3c29a90debdb01`, file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`; Pass 1 ACCEPT record commit `be845d7fb2d2ece861dcd158ac8b8dd89e3dc59d`, record SHA-256 `5a6e94f978c8a7e18d9623897ceba6f32f5b5eadb0aba6605f21c9d1e941a886`.

## Finding

**F1 — The pre-G0 source-authoring allowance conflicts with the contract-anchor bridge prerequisite.** The revised hard boundary permits isolated source and test-file authoring before G0. The revised G0 section still requires the reviewed docs-only contract-anchor integration to be created “before source/schema packages start,” and says that bridge becomes the implementation worktree base. These clauses leave it unclear whether pre-G0 P1 or other source-package authoring may start from the pinned source base without the bridge, or whether the bridge must precede that authoring even though integration is explicitly barred until G0. P1's source-only authoring dependency lists P0a and the exact reconciliation/profile pin, but does not resolve this base/bridge conflict. Clarify which milestone “package start” means and specify the exact pre-G0 source-authoring base and reconciliation artifact access, while retaining the prohibition on pre-G0 integration and on all execution.

## Review scope and evidence

The amendment cleanly separates source-file authoring and static source/diff review from build/compile, test execution, simulation, calibration, evaluation, dashboard operation, device/GPU use, qualification, training, checkpoint mutation, integration, merge and promotion. It preserves the no-contact boundary for campaign PID `2750253` and requires manual user stop plus independently verified release of both WX9100 leases before execution. It prohibits CPU numerical fallback, CPU simulation/scoring and P600 fallback, and says five-track training continues until the user manually stops it. It retains the accepted active-HUB FUEL award estimand, 160-physics-tick horizon, up to 12 ticks per macro (partial final macro allowed), gamma 0.993, branch exponent zero and continuation increment convention.

The two-stage review rule is explicit: pre-G0 source-scope/interface review cannot authorize integration or downstream API use; final exact-head acceptance of tests, device evidence and simulator/dashboard evidence happens after G0. P0b remains after frozen/reviewed P1–P6 source, while P6 runtime qualification waits for P0b, avoiding a P0/P6 cycle. No human approval, source conformance, campaign stop, lease release, training or validation is claimed.

The amendment's substantive gate split is sound, but F1 leaves the source branch base and contract-anchor sequencing internally ambiguous. The exact head should be clarified and reviewed again before its overall-plan ACCEPT is renewed.

## Review limits

This REVISE applies only to the exact plan commit/file hash above. No source, test, build, runtime, simulator, device/GPU, dashboard, qualification, campaign or training actions were performed during this review.
