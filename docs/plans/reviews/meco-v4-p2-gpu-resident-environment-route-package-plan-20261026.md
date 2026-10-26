# Independent Pass 1 review: P2 GPU-resident environment and route package plan

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent exact-head Pass 1 review assigned by `/root`
- **Plan author:** `plan_a_owner`
- **Reviewed plan commit:** `8d0727f10d46bfc5fee2810a1ec7fa7987ecaf91`
- **Reviewed file:** `docs/plans/meco-v4-p2-gpu-resident-environment-route-package-plan-20261026.md`
- **Git blob:** `2d5bb88f10ce84ac57ab5ecf97a8a52539ce12c8`
- **Plan file SHA-256:** `5eddd42914ab74426257ab8c1ff883b25c0c39aaa40bf7c40ccc2e363fbb3957`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted overall plan:** commit `6325920252773cfa05d63167ba3c29a90debdb01`, file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`; exact Pass 1 ACCEPT record commit `be845d7fb2d2ece861dcd158ac8b8dd89e3dc59d`, record SHA-256 `5a6e94f978c8a7e18d9623897ceba6f32f5b5eadb0aba6605f21c9d1e941a886`.
- **Accepted P0 package plan:** commit `13dbd7f8f953995850e811202f7e9c5264c49dca`, file SHA-256 `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`; exact Pass 1 ACCEPT record commit `037f0f9386cc0efadb1fb54efe5f38f097926d0e`, record SHA-256 `1ee70a6e9d24dc5201da5eec446820167cb8c4d4436fbbe013eb16de66b0280d`.
- **Accepted reconciliation:** commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; canonical reconciled-profile hash `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e`; V4 proposal digest `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- **Accepted P1 plan dependency:** commit `53686a254318484f9c6bc3496ecc1c461572c53a`, plan SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`; exact Pass 1 ACCEPT record commit `854d5286947074ed33f7ac059d67f2f9ccbf9257`, record SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f`.

## Review scope and evidence

The exact P2 plan is a child of the accepted P0 package plan and pins the corrected overall plan, P0 plan, reconciliation artifact, canonical profile and raw V4 proposal digest. The reconciled capacities are binding: the plan names Plan A's 12,096 nodes, 96,768 support rows, 10 edges per node and 14 ancestor levels; its full contract pin also binds Plan C and Route shapes/masks/order from the reconciliation. The reconciled artifact specifies Plan C's 537 source rows, 20 raw/40 model features, 9,681 route rows, 152 chunks of 64 and 73 path points, including the adopted Route row mappings and provenance.

The finite allowlist covers the P2 environment, observation, navigation, candidates, pose graph, route search, conditional perception/ranking, residency instrumentation and ledger, plus named focused tests. It explicitly keeps `meco_track_training.py` under P5, supervised/model data paths with their model/data owner, counterfactual and trainer integration with P5, P1 score-event semantics with P1, policy models with P3/P4, and qualification/scheduling with P6. The P1/P2 shared `tensor_3v3.py` boundary is ordered: accepted P1 integration is merged first, after which P2 alone owns its listed environment-step integration. The plan requires a complete call-graph ledger and named cross-package handoffs before global GPU-only acceptance.

The dependency order is acyclic and explicit: G0, accepted P0a decisions, accepted P1 interface integration, and independent P2 Pass 1 acceptance gate P2 execution. P0b correctly remains downstream of the P1–P6 source freeze. P2 supplies interfaces to P3/P4/P5 consumers and does not absorb their implementation ownership. The live campaign PID and device leases are protected by G0; the plan requires the user's manual stop and independently verified lease release before P2 execution or device use.

Device requirements are fail-closed: each worker binds an explicit WX9100 identity, rejects CPU/P600/ambiguous or changed placement, and records runtime and kernel evidence. The plan requires traces from both approved WX9100 devices and forbids treating one process as evidence of joint execution. Its focused tests and acceptance gates include simulator behavior, seeded end-to-end evaluation with replay inspection, and current dashboard backend/frontend/service/endpoint verification before fresh replay generation, matching the repository's AGENTS instructions. These are future gates; this review did not run them.

The plan gives distinct planner, implementer and reviewer roles and constrains changes to a fresh ZFS worktree and exact finite allowlist. No blocking Pass 1 finding remains for this exact artifact.

## Review limits

This ACCEPT applies only to the plan commit and file hash above. It accepts the package plan for its future gates; it does not satisfy G0 or P0a, authorize implementation, tests, simulator/dashboard actions, GPU/device use, campaign changes, training, human approval, or a claim of GPU residency. No source, test, runtime, GPU, simulator, dashboard, campaign or training actions were performed during this review.
